import Foundation
import ExternalAccessory
import CoreLocation
import Combine

/// Manages connection to Bad Elf GNSS receiver via ExternalAccessory framework.
/// Falls back to CLLocationManager if no Bad Elf is connected.
class BadElfReceiver: NSObject, ObservableObject {

    // MARK: - Published State

    @Published var isConnected = false
    @Published var latestData: GnssData? = nil
    @Published var fixTypeLabel = "No Fix"

    // MARK: - Private State

    private let primaryProtocol = "com.bad-elf.gnss"
    private let legacyProtocol = "com.bad-elf.gps"

    private var accessory: EAAccessory?
    private var session: EASession?
    private var inputStream: InputStream?
    private var outputStream: OutputStream?
    private var buffer = Data()
    private var isConfigured = false
    private var isLegacy = false

    // Epoch merging: accumulate GGA + GST with same timestamp
    private var pendingGGA: GGAData? = nil
    private var gstTimer: Timer? = nil

    // CLLocationManager fallback
    private var locationManager: CLLocationManager?
    private var fallbackTimer: Timer?
    private var usingFallback = false

    // MARK: - Init

    override init() {
        super.init()

        // Listen for accessory connect/disconnect
        NotificationCenter.default.addObserver(
            self, selector: #selector(accessoryDidConnect(_:)),
            name: .EAAccessoryDidConnect, object: nil
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(accessoryDidDisconnect(_:)),
            name: .EAAccessoryDidDisconnect, object: nil
        )
        EAAccessoryManager.shared().registerForLocalNotifications()

        // Try to connect immediately
        scanAndConnect()

        // Start fallback timer: if no Bad Elf in 5 seconds, use CLLocationManager
        fallbackTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { [weak self] _ in
            guard let self = self, !self.isConnected else { return }
            self.startFallbackLocationManager()
        }
    }

    deinit {
        disconnect()
        stopFallbackLocationManager()
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Connection

    private func scanAndConnect() {
        let accessories = EAAccessoryManager.shared().connectedAccessories

        // Prefer modern protocol
        if let acc = accessories.first(where: { $0.protocolStrings.contains(primaryProtocol) }) {
            connect(to: acc, protocol: primaryProtocol, legacy: false)
        } else if let acc = accessories.first(where: { $0.protocolStrings.contains(legacyProtocol) }) {
            connect(to: acc, protocol: legacyProtocol, legacy: true)
        }
    }

    private func connect(to accessory: EAAccessory, protocol proto: String, legacy: Bool) {
        guard let session = EASession(accessory: accessory, forProtocol: proto) else {
            print("[BadElf] Failed to open session for \(proto)")
            return
        }

        self.accessory = accessory
        self.session = session
        self.isLegacy = legacy
        self.isConfigured = false
        self.inputStream = session.inputStream
        self.outputStream = session.outputStream

        // Set up streams on main run loop
        inputStream?.delegate = self
        inputStream?.schedule(in: .main, forMode: .default)
        inputStream?.open()

        outputStream?.delegate = self
        outputStream?.schedule(in: .main, forMode: .default)
        outputStream?.open()

        DispatchQueue.main.async {
            self.isConnected = true
            self.stopFallbackLocationManager()
        }

        print("[BadElf] Connected to \(accessory.name) via \(proto)")
    }

    private func disconnect() {
        inputStream?.close()
        inputStream?.remove(from: .main, forMode: .default)
        outputStream?.close()
        outputStream?.remove(from: .main, forMode: .default)

        inputStream = nil
        outputStream = nil
        session = nil
        accessory = nil
        buffer = Data()
        isConfigured = false
        pendingGGA = nil
        gstTimer?.invalidate()
        gstTimer = nil

        DispatchQueue.main.async {
            self.isConnected = false
            self.fixTypeLabel = "No Fix"
        }
    }

    // MARK: - Config Message

    private func sendConfigMessage() {
        guard let outputStream = outputStream, outputStream.hasSpaceAvailable, !isConfigured else { return }

        if isLegacy {
            // Binary config for legacy devices
            let bytes: [UInt8] = [
                0x24, 0xbe, 0x00, 0x11, 0x05, 0x01, 0x02, 0x05,
                0x31, 0x01, 0x32, 0x04, 0x33, 0x01, 0x64, 0x0d, 0x0a
            ]
            let written = outputStream.write(bytes, maxLength: bytes.count)
            if written > 0 {
                isConfigured = true
                print("[BadElf] Legacy config sent")
            }
        } else {
            // JSON config for modern devices
            let appName = Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String ?? "SkiframesGPS"
            let appId = Bundle.main.bundleIdentifier ?? "com.skiframes.gps"
            let appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"

            let body = "PBEJS,{\"method\":\"session\",\"params\":{\"appName\":\"\(appName)\",\"appId\":\"\(appId)\",\"appVersion\":\"\(appVersion)\",\"msgs\":\"NMEA\"}}"
            let checksum = NmeaParser.computeChecksum(body)
            let sentence = "$\(body)*\(checksum)\r\n"

            if let data = sentence.data(using: .ascii) {
                let written = data.withUnsafeBytes { ptr -> Int in
                    guard let base = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return 0 }
                    return outputStream.write(base, maxLength: data.count)
                }
                if written > 0 {
                    isConfigured = true
                    print("[BadElf] Modern config sent: \(sentence.trimmingCharacters(in: .whitespacesAndNewlines))")
                }
            }
        }
    }

    // MARK: - Data Reading

    private func readAvailableData() {
        guard let inputStream = inputStream else { return }

        var readBuf = [UInt8](repeating: 0, count: 4096)
        let bytesRead = inputStream.read(&readBuf, maxLength: readBuf.count)
        if bytesRead > 0 {
            buffer.append(contentsOf: readBuf[0..<bytesRead])
            processBuffer()
        }
    }

    private func processBuffer() {
        while let sentence = extractNextSentence() {
            processSentence(sentence)
        }
    }

    /// Extract next complete NMEA sentence ($...\r\n) from buffer
    private func extractNextSentence() -> String? {
        // Find '$' start
        guard let startIdx = buffer.firstIndex(of: 0x24) else {
            buffer.removeAll()
            return nil
        }
        // Discard bytes before '$'
        if startIdx > buffer.startIndex {
            buffer.removeSubrange(buffer.startIndex..<startIdx)
        }

        // Find '\n' (0x0A) or '\r' (0x0D)
        guard let endIdx = buffer.dropFirst().firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) else {
            return nil // incomplete sentence
        }

        let sentenceData = buffer[buffer.startIndex...endIdx]
        let sentence = String(data: Data(sentenceData), encoding: .ascii)?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Remove consumed bytes (including any trailing \r\n)
        var removeEnd = buffer.index(after: endIdx)
        while removeEnd < buffer.endIndex && (buffer[removeEnd] == 0x0A || buffer[removeEnd] == 0x0D) {
            removeEnd = buffer.index(after: removeEnd)
        }
        buffer.removeSubrange(buffer.startIndex..<removeEnd)

        return sentence
    }

    // MARK: - NMEA Processing

    private func processSentence(_ sentence: String) {
        guard NmeaParser.validateChecksum(sentence) else { return }
        guard let parsed = NmeaParser.splitSentence(sentence) else { return }
        guard let type = NmeaParser.sentenceType(parsed.fields) else { return }

        switch type {
        case "GGA":
            if let gga = NmeaParser.parseGGA(parsed.fields) {
                pendingGGA = gga
                // Start timeout waiting for GST
                gstTimer?.invalidate()
                gstTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: false) { [weak self] _ in
                    self?.publishWithoutGST()
                }
            }
        case "GST":
            if let gst = NmeaParser.parseGST(parsed.fields) {
                gstTimer?.invalidate()
                gstTimer = nil
                if let gga = pendingGGA, gga.utcTime == gst.utcTime {
                    publishPosition(gga: gga, gst: gst)
                    pendingGGA = nil
                } else if let gga = pendingGGA {
                    // Timestamp mismatch — publish with GST accuracy anyway
                    publishPosition(gga: gga, gst: gst)
                    pendingGGA = nil
                }
            }
        default:
            break // Ignore RMC and others for now
        }
    }

    private func publishWithoutGST() {
        guard let gga = pendingGGA else { return }
        // Legacy device: estimate accuracy from HDOP
        let accuracy = gga.hdop * 3.9
        publishPosition(gga: gga, accuracy: accuracy)
        pendingGGA = nil
    }

    private func publishPosition(gga: GGAData, gst: GSTData) {
        publishPosition(gga: gga, accuracy: gst.horizontalAccuracy)
    }

    private func publishPosition(gga: GGAData, accuracy: Double) {
        let data = GnssData(
            latitude: gga.latitude,
            longitude: gga.longitude,
            altitude: gga.altitude,
            horizontalAccuracy: accuracy,
            fixType: gga.fixQuality,
            timestamp: Date(),
            hdop: gga.hdop,
            satelliteCount: gga.satellites
        )

        DispatchQueue.main.async {
            self.latestData = data
            self.fixTypeLabel = data.fixTypeLabel
        }
    }

    // MARK: - Accessory Notifications

    @objc private func accessoryDidConnect(_ notification: Notification) {
        guard !isConnected else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.scanAndConnect()
        }
    }

    @objc private func accessoryDidDisconnect(_ notification: Notification) {
        if let disconnected = notification.userInfo?[EAAccessoryKey] as? EAAccessory,
           disconnected == accessory {
            print("[BadElf] Disconnected")
            disconnect()
            // Start fallback after short delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                guard let self = self, !self.isConnected else { return }
                self.startFallbackLocationManager()
            }
        }
    }

    // MARK: - CLLocationManager Fallback

    private func startFallbackLocationManager() {
        guard !usingFallback else { return }
        usingFallback = true

        let manager = CLLocationManager()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.activityType = .fitness
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
        self.locationManager = manager

        print("[BadElf] Fallback to CLLocationManager")
    }

    private func stopFallbackLocationManager() {
        locationManager?.stopUpdatingLocation()
        locationManager = nil
        usingFallback = false
        fallbackTimer?.invalidate()
        fallbackTimer = nil
    }
}

// MARK: - StreamDelegate

extension BadElfReceiver: StreamDelegate {
    func stream(_ aStream: Stream, handle eventCode: Stream.Event) {
        switch eventCode {
        case .openCompleted:
            if aStream == outputStream {
                sendConfigMessage()
            }
        case .hasBytesAvailable:
            if aStream == inputStream {
                readAvailableData()
            }
        case .hasSpaceAvailable:
            if aStream == outputStream && !isConfigured {
                sendConfigMessage()
            }
        case .endEncountered, .errorOccurred:
            print("[BadElf] Stream event: \(eventCode)")
            disconnect()
        default:
            break
        }
    }
}

// MARK: - CLLocationManagerDelegate (fallback)

extension BadElfReceiver: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard usingFallback, let loc = locations.last else { return }

        let data = GnssData(
            latitude: loc.coordinate.latitude,
            longitude: loc.coordinate.longitude,
            altitude: loc.altitude,
            horizontalAccuracy: loc.horizontalAccuracy,
            fixType: 1, // Standard GPS
            timestamp: loc.timestamp,
            hdop: 0,
            satelliteCount: 0
        )

        DispatchQueue.main.async {
            self.latestData = data
            self.fixTypeLabel = "GPS (iOS)"
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        if manager.authorizationStatus == .authorizedWhenInUse ||
           manager.authorizationStatus == .authorizedAlways {
            manager.startUpdatingLocation()
        }
    }
}

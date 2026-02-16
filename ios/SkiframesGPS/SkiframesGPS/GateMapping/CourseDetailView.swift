import SwiftUI
import UIKit
import MapKit
import Combine

/// Main capture screen for mapping gate GPS positions on a course.
/// Shows GPS status, gate list with distances, live course map, and capture controls.
///
/// The course is looked up from the store by ID so all mutations persist.
/// Every capture action saves immediately to disk.
struct CourseDetailView: View {
    @ObservedObject var receiver: BadElfReceiver
    @ObservedObject var store: CourseStore
    let courseId: UUID

    @Environment(\.presentationMode) private var presentationMode

    @State private var selectedColor: GateColor = .red
    @State private var editingGate: GateEntry?
    @State private var scrollTarget: UUID?
    @State private var initialized = false
    @State private var showMap = false
    @State private var captureMode: CaptureMode = .gate

    enum CaptureMode {
        case startWand
        case gate
        case finishLeft
        case finishRight
    }

    // Haptic feedback
    private let impactFeedback = UIImpactFeedbackGenerator(style: .heavy)
    private let notificationFeedback = UINotificationFeedbackGenerator()

    /// Live binding to the course in the store
    private var course: CourseConfig {
        get { store.courses.first(where: { $0.id == courseId }) ?? CourseConfig(courseName: "?", discipline: .gs) }
    }

    /// Mutate and save the course
    private func update(_ mutate: (inout CourseConfig) -> Void) {
        guard var c = store.courses.first(where: { $0.id == courseId }) else { return }
        mutate(&c)
        store.save(c)
    }

    // MARK: - Init

    init(receiver: BadElfReceiver, store: CourseStore, course: CourseConfig) {
        self.receiver = receiver
        self.store = store
        self.courseId = course.id
    }

    var body: some View {
        VStack(spacing: 0) {
            // GPS status bar
            gpsStatusBar

            // Course header with map toggle
            courseHeader

            Divider()

            if showMap {
                // Course map view
                CourseMapView(course: course, currentPosition: receiver.latestData)
                    .frame(maxHeight: .infinity)
            } else {
                // Gate list (scrollable middle area)
                gateList
            }

            Divider()

            // Capture controls (pinned at bottom)
            captureControls
        }
        .navigationTitle(course.courseName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
                .fontWeight(.semibold)
            }
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    // Map toggle
                    Button {
                        withAnimation { showMap.toggle() }
                    } label: {
                        Label(showMap ? "Show List" : "Show Map", systemImage: showMap ? "list.bullet" : "map")
                    }

                    Divider()

                    // Google Maps
                    Button {
                        openInGoogleMaps()
                    } label: {
                        Label("Open in Google Maps", systemImage: "map")
                    }
                    .disabled(course.gates.isEmpty)

                    // Google Earth (KML)
                    Button {
                        openInGoogleEarth()
                    } label: {
                        Label("Open in Google Earth", systemImage: "globe.americas")
                    }
                    .disabled(course.gates.isEmpty)

                    Divider()

                    // Share JSON
                    if let url = store.exportURL(for: course) {
                        ShareLink(item: url) {
                            Label("Share JSON", systemImage: "square.and.arrow.up")
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .navigationBarBackButtonHidden(true)
        .sheet(item: $editingGate) { gate in
            GateEditSheet(gate: gate, course: editingCourseBinding) { updatedGate in
                update { $0.updateGate(updatedGate) }
            } onDelete: { gateId in
                update { $0.removeGate(id: gateId) }
            } onRecapture: { gateId in
                if let gnss = receiver.latestData {
                    update { c in
                        if let idx = c.gates.firstIndex(where: { $0.id == gateId }) {
                            c.gates[idx].position = GPSPoint(from: gnss)
                        }
                    }
                }
            }
        }
        .onAppear {
            if !initialized {
                selectedColor = course.nextExpectedColor
                // Start with Start Wand mode if not yet captured
                if course.startWand == nil {
                    captureMode = .startWand
                } else {
                    captureMode = .gate
                }
                initialized = true
            }
        }
    }

    /// Binding for GateEditSheet
    private var editingCourseBinding: Binding<CourseConfig> {
        Binding(
            get: { course },
            set: { store.save($0) }
        )
    }

    // MARK: - GPS Status Bar

    private var gpsStatusBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(gpsStatusColor)
                .frame(width: 10, height: 10)

            Text(gpsStatusText)
                .font(.system(size: 13, weight: .medium))

            Spacer()

            if let data = receiver.latestData {
                Text(String(format: "\u{00B1}%.2fm", data.horizontalAccuracy))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(accuracyColor(data.horizontalAccuracy).opacity(0.15))
                    .foregroundColor(accuracyColor(data.horizontalAccuracy))
                    .cornerRadius(10)

                if data.satelliteCount > 0 {
                    HStack(spacing: 2) {
                        Image(systemName: "antenna.radiowaves.left.and.right")
                            .font(.system(size: 10))
                        Text("\(data.satelliteCount)")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(.systemBackground))
    }

    private var gpsStatusColor: Color {
        if !receiver.isConnected {
            return receiver.latestData != nil ? .orange : .red
        }
        guard let data = receiver.latestData else { return .yellow }
        return data.fixType >= 4 ? .green : .blue
    }

    private var gpsStatusText: String {
        if receiver.isConnected {
            return "Bad Elf: \(receiver.fixTypeLabel)"
        } else if receiver.latestData != nil {
            return "iOS GPS"
        } else {
            return "Searching..."
        }
    }

    // MARK: - Course Header

    private var courseHeader: some View {
        HStack {
            Text(course.discipline.shortName)
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(disciplineColor)
                .cornerRadius(4)

            Text(course.summary)
                .font(.system(size: 13))
                .foregroundColor(.secondary)

            Spacer()

            if let drop = course.verticalDrop {
                HStack(spacing: 4) {
                    Text(String(format: "%.0fm\u{2193}", drop))
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                    if let slope = course.averageSlope {
                        Text(String(format: "%.0f%%", slope.grade))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                    }
                }
                .foregroundColor(.secondary)
            }

            // Map/List toggle button
            Button {
                withAnimation { showMap.toggle() }
            } label: {
                Image(systemName: showMap ? "list.bullet" : "map")
                    .font(.system(size: 14))
                    .foregroundColor(.primary)
                    .padding(6)
                    .background(Color(.tertiarySystemBackground))
                    .cornerRadius(6)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private var disciplineColor: Color {
        switch course.discipline {
        case .sl: return .blue
        case .gs: return .orange
        case .sg: return .red
        }
    }

    // MARK: - Gate List

    private var gateList: some View {
        ScrollViewReader { proxy in
            List {
                // Start wand section
                Section {
                    if let sw = course.startWand {
                        startWandRow(sw)
                    } else {
                        HStack {
                            Image(systemName: "flag.fill")
                                .foregroundColor(.green.opacity(0.3))
                                .frame(width: 28)
                            Text("Not captured yet")
                                .font(.system(size: 14))
                                .foregroundColor(.secondary)
                            Spacer()
                            Button("Capture") {
                                captureMode = .startWand
                            }
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.green)
                        }
                    }
                } header: {
                    Text("Start")
                }

                // Gates section
                Section {
                    ForEach(course.gates) { gate in
                        gateRow(gate)
                            .id(gate.id)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                editingGate = gate
                            }
                    }
                    .onDelete { offsets in
                        update { $0.gates.remove(atOffsets: offsets) }
                    }
                } header: {
                    HStack {
                        Text("Gates (\(course.gates.count))")
                        Spacer()
                        if let total = course.totalDistance {
                            Text(String(format: "%.0fm", total))
                                .font(.system(size: 12, weight: .medium, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        if let slope = course.averageSlope {
                            Text(String(format: "%.0f%%", slope.grade))
                                .font(.system(size: 12, weight: .medium, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                }

                // Finish line section
                Section {
                    if let fl = course.finishLine.left {
                        finishRow("Left Post", point: fl, recaptureMode: .finishLeft)
                    } else {
                        finishPlaceholder("Left Post", captureAction: { captureMode = .finishLeft })
                    }
                    if let fr = course.finishLine.right {
                        finishRow("Right Post", point: fr, recaptureMode: .finishRight)
                    } else {
                        finishPlaceholder("Right Post", captureAction: { captureMode = .finishRight })
                    }
                } header: {
                    Text("Finish")
                }
            }
            .listStyle(.insetGrouped)
            .onChange(of: scrollTarget) { target in
                if let target {
                    withAnimation {
                        proxy.scrollTo(target, anchor: .bottom)
                    }
                    scrollTarget = nil
                }
            }
        }
    }

    private func startWandRow(_ point: GPSPoint) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "flag.fill")
                .foregroundColor(.green)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 1) {
                Text("Start Wand")
                    .font(.system(size: 15, weight: .medium))
                Text(String(format: "%.1fm alt", point.alt))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(point.accuracyText)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(accuracyColor(point.accuracy))

            Button {
                captureMode = .startWand
            } label: {
                Image(systemName: "location.fill")
                    .font(.system(size: 12))
                    .foregroundColor(.green)
                    .padding(6)
                    .background(Color.green.opacity(0.12))
                    .cornerRadius(6)
            }
        }
        .padding(.vertical, 2)
    }

    private func gateRow(_ gate: GateEntry) -> some View {
        HStack(spacing: 10) {
            // Gate number
            Text("\(gate.number)")
                .font(.system(size: 15, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .frame(width: 28, height: 28)
                .background(gate.color == .red ? Color.red : Color.blue)
                .cornerRadius(6)

            // Gate info
            VStack(alignment: .leading, spacing: 1) {
                Text("Gate \(gate.number)")
                    .font(.system(size: 15, weight: .medium))
                // Distance + slope from previous gate
                if let dist = course.distanceFromPrevious(for: gate) {
                    HStack(spacing: 6) {
                        Text(String(format: "%.1fm", dist))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.secondary)
                        if let slope = course.slopeToPrevious(for: gate) {
                            Text(String(format: "%+.0f%% %.0fm\u{2193}", slope.grade, abs(slope.drop)))
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(slope.grade < -15 ? .orange : .secondary)
                        }
                    }
                }
            }

            Spacer()

            // Accuracy and fix type
            VStack(alignment: .trailing, spacing: 1) {
                Text(gate.position.accuracyText)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(accuracyColor(gate.position.accuracy))
                Text(gate.position.fixTypeLabel)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }

            Image(systemName: "chevron.right")
                .font(.system(size: 10))
                .foregroundColor(.secondary.opacity(0.4))
        }
        .padding(.vertical, 2)
    }

    private func finishRow(_ label: String, point: GPSPoint, recaptureMode: CaptureMode) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "flag.checkered")
                .foregroundColor(.primary)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.system(size: 15, weight: .medium))
                Text(String(format: "%.1fm alt", point.alt))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text(point.accuracyText)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(accuracyColor(point.accuracy))

            Button {
                captureMode = recaptureMode
            } label: {
                Image(systemName: "location.fill")
                    .font(.system(size: 12))
                    .foregroundColor(.primary)
                    .padding(6)
                    .background(Color(.tertiarySystemBackground))
                    .cornerRadius(6)
            }
        }
        .padding(.vertical, 2)
    }

    private func finishPlaceholder(_ label: String, captureAction: @escaping () -> Void) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "flag.checkered")
                .foregroundColor(.secondary.opacity(0.3))
                .frame(width: 28)
            Text(label)
                .font(.system(size: 14))
                .foregroundColor(.secondary)
            Spacer()
            Button("Capture") {
                captureAction()
            }
            .font(.system(size: 13, weight: .medium))
        }
    }

    // MARK: - Capture Controls

    private var captureControls: some View {
        VStack(spacing: 8) {
            // Mode selector tabs
            captureModeSelector

            // Main capture button — changes based on mode
            switch captureMode {
            case .startWand:
                captureButton(
                    title: course.startWand == nil ? "Capture Start Wand" : "Re-capture Start Wand",
                    icon: "flag.fill",
                    color: .green
                ) {
                    captureStartWand()
                }
            case .gate:
                // Color selector for gates
                Picker("Color", selection: $selectedColor) {
                    Text("Red").tag(GateColor.red)
                    Text("Blue").tag(GateColor.blue)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 200)
                .padding(.horizontal, 12)

                captureButton(
                    title: "Capture Gate \(course.nextGateNumber)",
                    icon: nil,
                    color: selectedColor == .red ? .red : .blue
                ) {
                    captureGate()
                }
            case .finishLeft:
                captureButton(
                    title: course.finishLine.left == nil ? "Capture Finish Left" : "Re-capture Finish Left",
                    icon: "flag.checkered",
                    color: Color(.darkGray)
                ) {
                    captureFinishLeft()
                }
            case .finishRight:
                captureButton(
                    title: course.finishLine.right == nil ? "Capture Finish Right" : "Re-capture Finish Right",
                    icon: "flag.checkered",
                    color: Color(.darkGray)
                ) {
                    captureFinishRight()
                }
            }

            // Low accuracy warning
            if let data = receiver.latestData, data.horizontalAccuracy > 3.0 {
                Text("Low accuracy \u{2014} wait for better fix")
                    .font(.system(size: 12))
                    .foregroundColor(.orange)
            }
        }
        .padding(.vertical, 10)
        .background(Color(.systemBackground))
    }

    /// Mode selector: Start | Gates | Finish L | Finish R
    private var captureModeSelector: some View {
        HStack(spacing: 6) {
            modeTab("Start", icon: "flag.fill", mode: .startWand,
                    badge: course.startWand != nil ? "✓" : nil,
                    color: .green)
            modeTab("Gates", icon: nil, mode: .gate,
                    badge: course.gates.isEmpty ? nil : "\(course.gates.count)",
                    color: .blue)
            modeTab("Finish L", icon: "flag.checkered", mode: .finishLeft,
                    badge: course.finishLine.left != nil ? "✓" : nil,
                    color: .primary)
            modeTab("Finish R", icon: "flag.checkered", mode: .finishRight,
                    badge: course.finishLine.right != nil ? "✓" : nil,
                    color: .primary)
        }
        .padding(.horizontal, 12)
    }

    private func modeTab(_ title: String, icon: String?, mode: CaptureMode, badge: String?, color: Color) -> some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { captureMode = mode }
        } label: {
            VStack(spacing: 2) {
                HStack(spacing: 3) {
                    if let icon {
                        Image(systemName: icon)
                            .font(.system(size: 10))
                    }
                    Text(title)
                        .font(.system(size: 11, weight: captureMode == mode ? .bold : .medium))
                }
                if let badge {
                    Text(badge)
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(captureMode == mode ? .white : color)
                }
            }
            .foregroundColor(captureMode == mode ? .white : .primary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(captureMode == mode ? color.opacity(0.85) : Color(.tertiarySystemBackground))
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }

    /// Unified capture button
    private func captureButton(title: String, icon: String?, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                HStack(spacing: 6) {
                    if let icon {
                        Image(systemName: icon)
                            .font(.system(size: 16))
                    }
                    Text(title)
                        .font(.system(size: 18, weight: .bold))
                }

                HStack(spacing: 12) {
                    // Live accuracy
                    if let data = receiver.latestData {
                        Text(String(format: "\u{00B1}%.2fm %@", data.horizontalAccuracy, data.fixTypeLabel))
                            .font(.system(size: 12, weight: .medium))
                            .opacity(0.8)
                    } else {
                        Text("Waiting for GPS...")
                            .font(.system(size: 12, weight: .medium))
                            .opacity(0.6)
                    }

                    // Live distance from last gate (only in gate mode)
                    if captureMode == .gate, let liveDistance = liveDistanceFromLastGate {
                        Text(String(format: "%.1fm away", liveDistance))
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .opacity(0.9)
                    }
                }
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 64)
            .background(receiver.latestData != nil ? color : .gray)
            .cornerRadius(16)
        }
        .disabled(receiver.latestData == nil)
        .padding(.horizontal, 12)
    }

    /// Live distance from the last captured point to current GPS position
    private var liveDistanceFromLastGate: Double? {
        guard let gnss = receiver.latestData,
              let lastPoint = course.lastCapturedPoint else { return nil }
        return lastPoint.distanceTo(lat: gnss.latitude, lon: gnss.longitude)
    }

    // MARK: - Capture Actions

    private func captureGate() {
        guard let gnss = receiver.latestData else { return }

        let gate = GateEntry(
            number: course.nextGateNumber,
            color: selectedColor,
            type: course.discipline.hasPanels ? .panelInside : .pole,
            position: GPSPoint(from: gnss)
        )

        update { $0.addGate(gate) }

        selectedColor = course.nextExpectedColor
        scrollTarget = gate.id
        impactFeedback.impactOccurred()
    }

    private func captureStartWand() {
        guard let gnss = receiver.latestData else { return }
        update { $0.setStartWand(GPSPoint(from: gnss)) }
        notificationFeedback.notificationOccurred(.success)
        // Auto-advance to gate capture mode
        withAnimation { captureMode = .gate }
    }

    private func captureFinishLeft() {
        guard let gnss = receiver.latestData else { return }
        update { $0.setFinishLeft(GPSPoint(from: gnss)) }
        notificationFeedback.notificationOccurred(.success)
        // Auto-advance to Finish Right if not captured yet
        if course.finishLine.right == nil {
            withAnimation { captureMode = .finishRight }
        }
    }

    private func captureFinishRight() {
        guard let gnss = receiver.latestData else { return }
        update { $0.setFinishRight(GPSPoint(from: gnss)) }
        notificationFeedback.notificationOccurred(.success)
        // Auto-advance to Finish Left if not captured yet
        if course.finishLine.left == nil {
            withAnimation { captureMode = .finishLeft }
        }
    }

    // MARK: - Google Maps

    private func openInGoogleMaps() {
        // Build a Google Maps URL with all gate waypoints
        // Uses directions mode with waypoints to show the course path
        var allPoints: [(lat: Double, lon: Double, label: String)] = []

        if let sw = course.startWand {
            allPoints.append((sw.lat, sw.lon, "Start"))
        }
        for gate in course.gates {
            allPoints.append((gate.position.lat, gate.position.lon, "Gate \(gate.number)"))
        }
        if let fl = course.finishLine.left {
            allPoints.append((fl.lat, fl.lon, "Finish"))
        }

        guard !allPoints.isEmpty else { return }

        if allPoints.count == 1 {
            // Single point — just open the location
            let p = allPoints[0]
            let urlStr = "https://www.google.com/maps/search/?api=1&query=\(p.lat),\(p.lon)&map_action=map&basemap=satellite"
            if let url = URL(string: urlStr) {
                UIApplication.shared.open(url)
            }
            return
        }

        // Multiple points — use directions with waypoints
        let origin = allPoints.first!
        let dest = allPoints.last!

        var urlStr = "https://www.google.com/maps/dir/?api=1"
        urlStr += "&origin=\(origin.lat),\(origin.lon)"
        urlStr += "&destination=\(dest.lat),\(dest.lon)"
        urlStr += "&travelmode=walking"

        // Add intermediate waypoints (max ~23 for Google Maps URL)
        if allPoints.count > 2 {
            let waypoints = allPoints[1..<(allPoints.count - 1)]
                .prefix(23)
                .map { "\($0.lat),\($0.lon)" }
                .joined(separator: "|")
            urlStr += "&waypoints=\(waypoints)"
        }

        // Try Google Maps app first, fall back to Safari
        if let gmsURL = URL(string: "comgooglemaps://?saddr=\(origin.lat),\(origin.lon)&daddr=\(dest.lat),\(dest.lon)&directionsmode=walking&maptype=satellite"),
           UIApplication.shared.canOpenURL(gmsURL) {
            UIApplication.shared.open(gmsURL)
        } else if let url = URL(string: urlStr) {
            UIApplication.shared.open(url)
        }
    }

    // MARK: - Google Earth Export

    private func openInGoogleEarth() {
        let kml = course.generateKML()

        // Write KML to temp file
        let tempDir = FileManager.default.temporaryDirectory
        let kmlURL = tempDir.appendingPathComponent("\(course.courseId).kml")
        do {
            try kml.write(to: kmlURL, atomically: true, encoding: .utf8)
        } catch {
            print("[KML] Failed to write: \(error)")
            return
        }

        // Try Google Earth app first, fall back to share sheet
        let geURL = URL(string: "comgoogleearth://")
        if let geURL, UIApplication.shared.canOpenURL(geURL) {
            // Google Earth is installed — share the KML file to it
            shareKMLFile(kmlURL)
        } else {
            // No Google Earth app — share so user can choose
            shareKMLFile(kmlURL)
        }
    }

    private func shareKMLFile(_ url: URL) {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let rootVC = windowScene.windows.first?.rootViewController else { return }

        var vc = rootVC
        while let presented = vc.presentedViewController { vc = presented }

        let activityVC = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        activityVC.popoverPresentationController?.sourceView = vc.view
        activityVC.popoverPresentationController?.sourceRect = CGRect(x: vc.view.bounds.midX, y: 0, width: 0, height: 0)
        vc.present(activityVC, animated: true)
    }

    // MARK: - Helpers

    private func accuracyColor(_ accuracy: Double) -> Color {
        if accuracy <= 0.2 { return .green }
        if accuracy <= 1.0 { return .blue }
        if accuracy <= 3.0 { return .orange }
        return .red
    }
}

#Preview {
    NavigationStack {
        CourseDetailView(
            receiver: BadElfReceiver(),
            store: CourseStore(),
            course: CourseConfig(courseName: "GS Training U14", discipline: .gs)
        )
    }
}

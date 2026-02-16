import Foundation

// MARK: - Enums

/// Ski race discipline type
enum Discipline: String, Codable, CaseIterable, Identifiable {
    case gs
    case sl
    case sg

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .gs: return "Giant Slalom"
        case .sl: return "Slalom"
        case .sg: return "Super-G"
        }
    }

    var shortName: String {
        switch self {
        case .gs: return "GS"
        case .sl: return "SL"
        case .sg: return "SG"
        }
    }

    var poleHeight: Double {
        switch self {
        case .sl: return 1.80
        case .gs, .sg: return 2.20
        }
    }

    /// Whether this discipline uses panel gates (GS/SG have inside+outside panels)
    var hasPanels: Bool {
        switch self {
        case .gs, .sg: return true
        case .sl: return false
        }
    }

    /// Python backend-compatible discipline key (for legacy compatibility)
    var pythonKey: String {
        switch self {
        case .gs: return "gs_panel"
        case .sl: return "sl_adult"
        case .sg: return "sg_panel"
        }
    }
}

/// Training group
enum TrainingGroup: String, Codable, CaseIterable, Identifiable {
    case scored
    case u14
    case u12
    case u10
    case masters

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .scored: return "Scored"
        case .u14: return "U14"
        case .u12: return "U12"
        case .u10: return "U10"
        case .masters: return "Masters"
        }
    }
}

/// Ski site/resort
enum Site: String, Codable, CaseIterable, Identifiable {
    case ragged = "ragged_mountain"
    case sunapee = "mount_sunapee"
    case proctor = "proctor_ski_area"
    case other = "other"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .ragged: return "Ragged Mountain Resort (NH)"
        case .sunapee: return "Mount Sunapee (NH)"
        case .proctor: return "Proctor Ski Area (NH)"
        case .other: return "Other"
        }
    }

    var shortName: String {
        switch self {
        case .ragged: return "Ragged"
        case .sunapee: return "Sunapee"
        case .proctor: return "Proctor"
        case .other: return ""
        }
    }

    /// Course names available at this site
    var courseNames: [String] {
        switch self {
        case .ragged: return ["Flying Yankee", "Village Green"]
        case .sunapee: return ["Lynx"]
        case .proctor: return ["Burden Trail"]
        case .other: return []
        }
    }
}

/// Gate pole color
enum GateColor: String, Codable, CaseIterable {
    case red
    case blue

    /// Toggle to the other color
    var toggled: GateColor {
        self == .red ? .blue : .red
    }
}

/// Gate type within the course
enum GateType: String, Codable, CaseIterable {
    case pole = "pole"                      // Slalom single pole
    case panelInside = "panel_inside"       // GS/SG inside gate (with panel/flag)
    case panelOutside = "panel_outside"     // GS/SG outside pole

    var displayName: String {
        switch self {
        case .pole: return "Pole"
        case .panelInside: return "Inside Panel"
        case .panelOutside: return "Outside Pole"
        }
    }

    var shortName: String {
        switch self {
        case .pole: return "Pole"
        case .panelInside: return "Inside"
        case .panelOutside: return "Outside"
        }
    }
}

// MARK: - GPS Point

/// A single GPS measurement with accuracy metadata
struct GPSPoint: Codable, Equatable {
    var lat: Double
    var lon: Double
    var alt: Double
    var accuracy: Double
    var fixType: Int        // 0=No Fix, 1=GPS, 2=DGPS, 4=RTK Fixed, 5=RTK Float
    var timestamp: Date

    /// Initialize from a GnssData reading
    init(from gnss: GnssData) {
        self.lat = gnss.latitude
        self.lon = gnss.longitude
        self.alt = gnss.altitude
        self.accuracy = gnss.horizontalAccuracy
        self.fixType = gnss.fixType
        self.timestamp = gnss.timestamp
    }

    /// Manual init for testing or importing
    init(lat: Double, lon: Double, alt: Double, accuracy: Double = 0, fixType: Int = 1, timestamp: Date = Date()) {
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.accuracy = accuracy
        self.fixType = fixType
        self.timestamp = timestamp
    }

    var fixTypeLabel: String {
        switch fixType {
        case 4: return "RTK Fixed"
        case 5: return "RTK Float"
        case 2: return "DGPS"
        case 1: return "GPS"
        default: return "No Fix"
        }
    }

    var accuracyText: String {
        String(format: "\u{00B1}%.2fm", accuracy)
    }

    /// Haversine distance to another GPS point (meters)
    func distanceTo(_ other: GPSPoint) -> Double {
        Self.haversine(lat1: lat, lon1: lon, lat2: other.lat, lon2: other.lon)
    }

    /// Haversine distance from this point to a raw lat/lon (meters)
    func distanceTo(lat: Double, lon: Double) -> Double {
        Self.haversine(lat1: self.lat, lon1: self.lon, lat2: lat, lon2: lon)
    }

    /// Haversine formula: great-circle distance between two points (meters)
    static func haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double) -> Double {
        let R = 6371000.0 // Earth radius in meters
        let dLat = (lat2 - lat1) * .pi / 180.0
        let dLon = (lon2 - lon1) * .pi / 180.0
        let a = sin(dLat / 2) * sin(dLat / 2) +
                cos(lat1 * .pi / 180.0) * cos(lat2 * .pi / 180.0) *
                sin(dLon / 2) * sin(dLon / 2)
        let c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    }
}

// MARK: - Gate Entry

/// A single gate on the course with GPS position
struct GateEntry: Codable, Identifiable, Equatable {
    let id: UUID
    var number: Int
    var color: GateColor
    var type: GateType
    var position: GPSPoint

    init(number: Int, color: GateColor, type: GateType, position: GPSPoint) {
        self.id = UUID()
        self.number = number
        self.color = color
        self.type = type
        self.position = position
    }
}

// MARK: - Finish Line

/// Finish line defined by two posts (left and right)
struct FinishLine: Codable, Equatable {
    var left: GPSPoint?
    var right: GPSPoint?

    var isComplete: Bool { left != nil && right != nil }
    var hasAny: Bool { left != nil || right != nil }
}

// MARK: - Course Config

/// Complete course configuration with all gate GPS positions.
/// This is the standalone config file saved by the Gates Mapping feature.
/// Independent of camera calibration — reusable for homography, KML, analytics, etc.
struct CourseConfig: Codable, Identifiable, Hashable {
    static func == (lhs: CourseConfig, rhs: CourseConfig) -> Bool {
        lhs.id == rhs.id
    }
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    let id: UUID
    var version: Int
    var courseId: String
    var courseName: String
    var discipline: Discipline
    var site: Site
    var groups: [TrainingGroup]
    var date: Date
    var location: String
    var created: Date
    var modified: Date

    var startWand: GPSPoint?
    var finishLine: FinishLine
    var gates: [GateEntry]

    // MARK: - Init

    init(courseName: String, discipline: Discipline, site: Site = .ragged, groups: [TrainingGroup] = [], location: String = "Ragged Mountain", date: Date = Date()) {
        self.id = UUID()
        self.version = 1

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmm"
        let dateStr = formatter.string(from: date)
        let slug = courseName.lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "[^a-z0-9_]", with: "", options: .regularExpression)
        self.courseId = "\(dateStr)_\(discipline.rawValue)_\(slug)"

        self.courseName = courseName
        self.discipline = discipline
        self.site = site
        self.groups = groups
        self.date = date
        self.location = location
        self.created = Date()
        self.modified = Date()
        self.startWand = nil
        self.finishLine = FinishLine()
        self.gates = []
    }

    /// Groups display string
    var groupsDisplay: String {
        groups.isEmpty ? "" : groups.map(\.displayName).joined(separator: ", ")
    }

    // MARK: - Computed Properties

    /// Next gate number (auto-increment from highest existing)
    var nextGateNumber: Int {
        (gates.map(\.number).max() ?? 0) + 1
    }

    /// Next expected color based on alternating pattern (toggles every gate)
    var nextExpectedColor: GateColor {
        guard let lastGate = gates.last else { return .red }
        return lastGate.color.toggled
    }

    /// Summary string for gate count
    var summary: String {
        var parts: [String] = []
        parts.append("\(gates.count) gate\(gates.count == 1 ? "" : "s")")
        if startWand != nil { parts.append("Start") }
        if finishLine.isComplete {
            parts.append("Finish")
        } else if finishLine.hasAny {
            parts.append("Finish (1/2)")
        }
        return parts.joined(separator: " | ")
    }

    /// Total vertical drop from first to last gate (meters)
    var verticalDrop: Double? {
        guard let first = gates.first, let last = gates.last else { return nil }
        return first.position.alt - last.position.alt
    }

    // MARK: - Mutation

    /// Add a gate and update modified timestamp
    mutating func addGate(_ gate: GateEntry) {
        gates.append(gate)
        modified = Date()
    }

    /// Remove a gate by ID
    mutating func removeGate(id: UUID) {
        gates.removeAll { $0.id == id }
        modified = Date()
    }

    /// Update a gate by ID
    mutating func updateGate(_ gate: GateEntry) {
        if let idx = gates.firstIndex(where: { $0.id == gate.id }) {
            gates[idx] = gate
            modified = Date()
        }
    }

    /// Set start wand position
    mutating func setStartWand(_ point: GPSPoint) {
        startWand = point
        modified = Date()
    }

    /// Set finish line left post
    mutating func setFinishLeft(_ point: GPSPoint) {
        finishLine.left = point
        modified = Date()
    }

    /// Set finish line right post
    mutating func setFinishRight(_ point: GPSPoint) {
        finishLine.right = point
        modified = Date()
    }

    // MARK: - Distance Helpers

    /// Distance from previous gate (or start wand) to this gate, in meters. Nil if first gate with no start wand.
    func distanceFromPrevious(for gate: GateEntry) -> Double? {
        guard let idx = gates.firstIndex(where: { $0.id == gate.id }) else { return nil }
        if idx > 0 {
            return gates[idx - 1].position.distanceTo(gate.position)
        } else if let sw = startWand {
            return sw.distanceTo(gate.position)
        }
        return nil
    }

    /// Slope/grade info between consecutive gates
    struct SlopeInfo {
        let drop: Double      // altitude change in meters (negative = downhill)
        let grade: Double     // grade percentage (negative = downhill)
        let slopeDeg: Double  // slope angle in degrees
    }

    /// Slope from previous gate to this gate. Nil if first gate with no start wand.
    func slopeToPrevious(for gate: GateEntry) -> SlopeInfo? {
        guard let idx = gates.firstIndex(where: { $0.id == gate.id }) else { return nil }
        let prevPoint: GPSPoint
        if idx > 0 {
            prevPoint = gates[idx - 1].position
        } else if let sw = startWand {
            prevPoint = sw
        } else {
            return nil
        }
        let horizDist = prevPoint.distanceTo(gate.position)
        guard horizDist > 0.5 else { return nil } // too close for meaningful slope
        let drop = gate.position.alt - prevPoint.alt
        let grade = (drop / horizDist) * 100.0
        let slopeDeg = atan2(abs(drop), horizDist) * 180.0 / .pi
        return SlopeInfo(drop: drop, grade: grade, slopeDeg: slopeDeg)
    }

    /// Average slope for the entire course
    var averageSlope: SlopeInfo? {
        guard let total = totalDistance, total > 1, let vDrop = verticalDrop else { return nil }
        let grade = (-vDrop / total) * 100.0
        let slopeDeg = atan2(vDrop, total) * 180.0 / .pi
        return SlopeInfo(drop: -vDrop, grade: grade, slopeDeg: slopeDeg)
    }

    /// Total course length along gate-to-gate path (meters)
    var totalDistance: Double? {
        guard gates.count >= 2 else { return nil }
        var total = 0.0
        if let sw = startWand, let first = gates.first {
            total += sw.distanceTo(first.position)
        }
        for i in 1..<gates.count {
            total += gates[i - 1].position.distanceTo(gates[i].position)
        }
        if let last = gates.last, let fl = finishLine.left {
            total += last.position.distanceTo(fl)
        }
        return total
    }

    /// The last captured point (last gate, or start wand if no gates)
    var lastCapturedPoint: GPSPoint? {
        gates.last?.position ?? startWand
    }

    // MARK: - KML Export

    /// Generate KML string for viewing in Google Earth with snow-white terrain overlay
    func generateKML() -> String {
        var kml = """
        <?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2"
             xmlns:gx="http://www.google.com/kml/ext/2.2">
        <Document>
          <name>\(xmlEscape(courseName))</name>
          <description>\(xmlEscape(discipline.displayName)) - \(xmlEscape(location))
        Gates: \(gates.count)\(verticalDrop.map { String(format: " | Drop: %.0fm", $0) } ?? "")\(averageSlope.map { String(format: " | Slope: %.0f%%", $0.grade) } ?? "")</description>
          <open>1</open>

          <Style id="redGate">
            <IconStyle><color>ff0000ff</color><scale>0.9</scale>
              <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
            </IconStyle>
            <LabelStyle><scale>0.8</scale></LabelStyle>
          </Style>
          <Style id="blueGate">
            <IconStyle><color>ffff0000</color><scale>0.9</scale>
              <Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon>
            </IconStyle>
            <LabelStyle><scale>0.8</scale></LabelStyle>
          </Style>
          <Style id="startPin">
            <IconStyle><color>ff00ff00</color><scale>1.1</scale>
              <Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-stars.png</href></Icon>
            </IconStyle>
            <LabelStyle><scale>0.9</scale></LabelStyle>
          </Style>
          <Style id="finishPin">
            <IconStyle><scale>1.0</scale>
              <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-square.png</href></Icon>
            </IconStyle>
            <LabelStyle><scale>0.9</scale></LabelStyle>
          </Style>
          <Style id="raceLine">
            <LineStyle><color>ff00b4ff</color><width>3</width></LineStyle>
          </Style>
          <Style id="snowCover">
            <PolyStyle><color>b0ffffff</color></PolyStyle>
            <LineStyle><color>00ffffff</color><width>0</width></LineStyle>
          </Style>

        """

        // === Snow-white ground overlay polygon ===
        // Covers the course area with semi-transparent white to simulate snow
        if let bounds = courseBounds {
            let margin = 0.0005 // ~55m margin around course
            let north = bounds.maxLat + margin
            let south = bounds.minLat - margin
            let east = bounds.maxLon + margin
            let west = bounds.minLon - margin
            kml += """
              <Folder>
                <name>Snow Cover</name>
                <description>Semi-transparent snow overlay on terrain</description>
                <visibility>1</visibility>
                <Placemark>
                  <name>Snow Ground</name>
                  <styleUrl>#snowCover</styleUrl>
                  <Polygon>
                    <tessellate>1</tessellate>
                    <altitudeMode>clampToGround</altitudeMode>
                    <outerBoundaryIs>
                      <LinearRing>
                        <coordinates>
                          \(String(format: "%.9f,%.9f,0", west, south))
                          \(String(format: "%.9f,%.9f,0", east, south))
                          \(String(format: "%.9f,%.9f,0", east, north))
                          \(String(format: "%.9f,%.9f,0", west, north))
                          \(String(format: "%.9f,%.9f,0", west, south))
                        </coordinates>
                      </LinearRing>
                    </outerBoundaryIs>
                  </Polygon>
                </Placemark>
              </Folder>

            """
        }

        // === Gates Folder ===
        kml += """
          <Folder>
            <name>Gates</name>
            <open>1</open>

        """

        // Start wand
        if let sw = startWand {
            kml += placemark(name: "Start", style: "startPin", point: sw,
                             description: "Start Wand\nAlt: \(String(format: "%.1fm", sw.alt))\nAccuracy: \(sw.accuracyText)")
        }

        // Gates with elevation info
        for gate in gates {
            let style = gate.color == .red ? "redGate" : "blueGate"
            let label = "Gate \(gate.number)"
            var desc = "\(label) (\(gate.color.rawValue) \(gate.type.shortName))"
            desc += "\nAlt: \(String(format: "%.1fm", gate.position.alt))"
            desc += "\nAccuracy: \(gate.position.accuracyText)"
            if let dist = distanceFromPrevious(for: gate) {
                desc += String(format: "\nDist from prev: %.1fm", dist)
            }
            if let slope = slopeToPrevious(for: gate) {
                desc += String(format: "\nSlope: %.0f%% (%.1f°)", slope.grade, slope.slopeDeg)
                desc += String(format: "\nDrop: %.1fm", slope.drop)
            }
            kml += placemark(name: label, style: style, point: gate.position, description: desc)
        }

        // Finish line
        if let fl = finishLine.left {
            kml += placemark(name: "Finish L", style: "finishPin", point: fl,
                             description: "Finish Left Post\nAlt: \(String(format: "%.1fm", fl.alt))")
        }
        if let fr = finishLine.right {
            kml += placemark(name: "Finish R", style: "finishPin", point: fr,
                             description: "Finish Right Post\nAlt: \(String(format: "%.1fm", fr.alt))")
        }

        kml += """
          </Folder>

        """

        // === Race line through all points ===
        var lineCoords: [(Double, Double, Double)] = []
        if let sw = startWand { lineCoords.append((sw.lon, sw.lat, sw.alt)) }
        for gate in gates { lineCoords.append((gate.position.lon, gate.position.lat, gate.position.alt)) }
        if let fl = finishLine.left { lineCoords.append((fl.lon, fl.lat, fl.alt)) }

        if lineCoords.count >= 2 {
            let coordStr = lineCoords.map { String(format: "%.9f,%.9f,%.1f", $0.0, $0.1, $0.2) }.joined(separator: "\n            ")
            kml += """
              <Placemark>
                <name>Race Line</name>
                <description>\(gates.count) gates\(totalDistance.map { String(format: " | %.0fm", $0) } ?? "")\(verticalDrop.map { String(format: " | %.0fm drop", $0) } ?? "")</description>
                <styleUrl>#raceLine</styleUrl>
                <LineString>
                  <tessellate>1</tessellate>
                  <altitudeMode>clampToGround</altitudeMode>
                  <coordinates>
                    \(coordStr)
                  </coordinates>
                </LineString>
              </Placemark>

            """
        }

        // === LookAt (center on course, heading along fall line) ===
        if let center = courseCenter {
            // Compute heading from start to finish for natural viewing angle
            let heading = courseHeading ?? 0
            kml += """
              <LookAt>
                <latitude>\(center.lat)</latitude>
                <longitude>\(center.lon)</longitude>
                <altitude>0</altitude>
                <range>300</range>
                <tilt>60</tilt>
                <heading>\(String(format: "%.1f", heading))</heading>
                <altitudeMode>clampToGround</altitudeMode>
              </LookAt>

            """
        }

        kml += """
        </Document>
        </kml>
        """
        return kml
    }

    private func placemark(name: String, style: String, point: GPSPoint, description: String = "") -> String {
        let escapedDesc = xmlEscape(description).replacingOccurrences(of: "\n", with: "&#10;")
        return """
          <Placemark>
            <name>\(xmlEscape(name))</name>
            <description>\(escapedDesc)</description>
            <styleUrl>#\(style)</styleUrl>
            <Point>
              <altitudeMode>clampToGround</altitudeMode>
              <coordinates>\(String(format: "%.9f,%.9f,%.1f", point.lon, point.lat, point.alt))</coordinates>
            </Point>
          </Placemark>

        """
    }

    private func xmlEscape(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
         .replacingOccurrences(of: "<", with: "&lt;")
         .replacingOccurrences(of: ">", with: "&gt;")
         .replacingOccurrences(of: "\"", with: "&quot;")
    }

    /// Geographic bounds of all captured points
    private var courseBounds: (minLat: Double, maxLat: Double, minLon: Double, maxLon: Double)? {
        var lats: [Double] = []
        var lons: [Double] = []
        if let sw = startWand { lats.append(sw.lat); lons.append(sw.lon) }
        for g in gates { lats.append(g.position.lat); lons.append(g.position.lon) }
        if let fl = finishLine.left { lats.append(fl.lat); lons.append(fl.lon) }
        if let fr = finishLine.right { lats.append(fr.lat); lons.append(fr.lon) }
        guard let minLat = lats.min(), let maxLat = lats.max(),
              let minLon = lons.min(), let maxLon = lons.max() else { return nil }
        return (minLat, maxLat, minLon, maxLon)
    }

    /// Geographic center of all captured points
    private var courseCenter: (lat: Double, lon: Double)? {
        guard let b = courseBounds else { return nil }
        return ((b.minLat + b.maxLat) / 2, (b.minLon + b.maxLon) / 2)
    }

    /// Heading from first to last point (degrees from north, clockwise)
    private var courseHeading: Double? {
        guard let first = (startWand ?? gates.first?.position),
              let last = (finishLine.left ?? gates.last?.position) else { return nil }
        let dLon = (last.lon - first.lon) * .pi / 180.0
        let lat1 = first.lat * .pi / 180.0
        let lat2 = last.lat * .pi / 180.0
        let x = sin(dLon) * cos(lat2)
        let y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
        var heading = atan2(x, y) * 180.0 / .pi
        if heading < 0 { heading += 360 }
        return heading
    }

    // MARK: - Export to Python-compatible format

    /// Convert to the format expected by gps_calibration.py's from_calibration_json()
    func toPythonCalibrationFormat() -> [String: Any] {
        var gatePositions: [String: [String: Any]] = [:]
        var gateList: [[String: Any]] = []

        for gate in gates {
            let gid = String(gate.number)
            gatePositions[gid] = [
                "lat": gate.position.lat,
                "lon": gate.position.lon,
                "alt": gate.position.alt,
                "accuracy": gate.position.accuracy,
            ]
            gateList.append([
                "id": gate.number,
                "color": gate.color.rawValue,
                "type": gate.type.rawValue,
            ])
        }

        // Include start wand as gate 0
        if let sw = startWand {
            gatePositions["0"] = [
                "lat": sw.lat, "lon": sw.lon, "alt": sw.alt, "accuracy": sw.accuracy,
            ]
            gateList.insert(["id": 0, "color": "red", "type": "start_wand"], at: 0)
        }

        // Include finish line posts
        if let fl = finishLine.left {
            let fid = (gates.map(\.number).max() ?? 0) + 100
            gatePositions[String(fid)] = [
                "lat": fl.lat, "lon": fl.lon, "alt": fl.alt, "accuracy": fl.accuracy,
            ]
            gateList.append(["id": fid, "color": "red", "type": "finish_left"])
        }
        if let fr = finishLine.right {
            let fid = (gates.map(\.number).max() ?? 0) + 101
            gatePositions[String(fid)] = [
                "lat": fr.lat, "lon": fr.lon, "alt": fr.alt, "accuracy": fr.accuracy,
            ]
            gateList.append(["id": fid, "color": "red", "type": "finish_right"])
        }

        return [
            "gps_data": [
                "gate_positions": gatePositions,
            ],
            "gates": gateList,
            "discipline": discipline.pythonKey,
            "calibration_mode": "gps",
            "course_id": courseId,
            "course_name": courseName,
            "site": site.rawValue,
            "groups": groups.map(\.rawValue),
        ]
    }
}

// MARK: - JSON Coding Configuration

extension CourseConfig {
    /// Configured JSON encoder matching Python backend conventions
    static var jsonEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }

    /// Configured JSON decoder matching Python backend conventions
    static var jsonDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

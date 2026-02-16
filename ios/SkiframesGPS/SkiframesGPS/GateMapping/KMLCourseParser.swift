import Foundation

/// Parses a KML file (as exported by the app's Google Earth export) back into a CourseConfig.
/// Extracts gate positions, start wand, and finish line from KML Placemarks.
class KMLCourseParser: NSObject, XMLParserDelegate {

    private let data: Data

    // Parsed result
    private var courseName: String = ""
    private var descriptionText: String = ""
    private var placemarks: [ParsedPlacemark] = []

    // XML parsing state
    private var currentElement: String = ""
    private var currentText: String = ""
    private var inGatesFolder = false
    private var inPlacemark = false
    private var currentPlacemarkName: String = ""
    private var currentPlacemarkDesc: String = ""
    private var currentPlacemarkStyle: String = ""
    private var currentCoordinates: String = ""
    private var isDocumentLevel = true  // True until we enter a Folder
    private var folderDepth = 0

    struct ParsedPlacemark {
        var name: String
        var description: String
        var styleUrl: String
        var lon: Double
        var lat: Double
        var alt: Double
    }

    init(data: Data) {
        self.data = data
    }

    func parse() -> CourseConfig? {
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else { return nil }
        return buildCourseConfig()
    }

    // MARK: - XMLParserDelegate

    func parser(_ parser: XMLParser, didStartElement elementName: String,
                namespaceURI: String?, qualifiedName: String?,
                attributes: [String: String]) {
        currentElement = elementName
        currentText = ""

        if elementName == "Folder" {
            folderDepth += 1
            isDocumentLevel = false
        }
        if elementName == "Placemark" {
            inPlacemark = true
            currentPlacemarkName = ""
            currentPlacemarkDesc = ""
            currentPlacemarkStyle = ""
            currentCoordinates = ""
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        currentText += string
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String,
                namespaceURI: String?, qualifiedName: String?) {
        let trimmed = currentText.trimmingCharacters(in: .whitespacesAndNewlines)

        if elementName == "Folder" {
            if inGatesFolder && folderDepth > 0 {
                inGatesFolder = false
            }
            folderDepth -= 1
        }

        // Document-level name and description
        if elementName == "name" && !inPlacemark && folderDepth == 0 {
            courseName = trimmed
        }
        if elementName == "description" && !inPlacemark && folderDepth == 0 {
            descriptionText = trimmed
        }

        // Folder name — detect Gates folder
        if elementName == "name" && !inPlacemark && folderDepth > 0 {
            if trimmed == "Gates" {
                inGatesFolder = true
            }
        }

        // Placemark fields
        if inPlacemark {
            switch elementName {
            case "name":
                currentPlacemarkName = trimmed
            case "description":
                currentPlacemarkDesc = trimmed
            case "styleUrl":
                currentPlacemarkStyle = trimmed
            case "coordinates":
                currentCoordinates = trimmed
            case "Placemark":
                // End of placemark — save if it has a Point (single coordinate)
                if let (lon, lat, alt) = parsePointCoordinates(currentCoordinates) {
                    placemarks.append(ParsedPlacemark(
                        name: currentPlacemarkName,
                        description: currentPlacemarkDesc,
                        styleUrl: currentPlacemarkStyle,
                        lon: lon, lat: lat, alt: alt
                    ))
                }
                inPlacemark = false
            default:
                break
            }
        }

        currentElement = ""
    }

    // MARK: - Coordinate Parsing

    /// Parse "lon,lat,alt" from a KML Point's coordinates
    private func parsePointCoordinates(_ text: String) -> (Double, Double, Double)? {
        // KML coordinates: "lon,lat,alt" — may have spaces around them
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Skip multi-point coordinates (LineString) — those contain spaces between tuples
        let tuples = cleaned.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        guard tuples.count == 1 else { return nil }  // Only single-point Placemarks

        let parts = tuples[0].split(separator: ",")
        guard parts.count >= 2,
              let lon = Double(parts[0]),
              let lat = Double(parts[1]) else { return nil }
        let alt = parts.count >= 3 ? (Double(parts[2]) ?? 0) : 0
        return (lon, lat, alt)
    }

    // MARK: - Build CourseConfig

    private func buildCourseConfig() -> CourseConfig? {
        // Determine discipline from description
        let discipline = parseDiscipline(from: descriptionText)

        // Determine site from description
        let site = parseSite(from: descriptionText)

        // Use course name from KML document, or filename
        let name = courseName.isEmpty ? "Imported Course" : courseName

        var config = CourseConfig(
            courseName: name,
            discipline: discipline,
            site: site,
            groups: [],
            location: site.displayName
        )

        // Parse accuracy from description text like "Accuracy: ±0.11m"
        func parseAccuracy(from desc: String) -> Double {
            if let range = desc.range(of: #"Accuracy:\s*[±]?([\d.]+)m"#, options: .regularExpression) {
                let match = desc[range]
                let numStr = match.replacingOccurrences(of: "Accuracy:", with: "")
                    .replacingOccurrences(of: "±", with: "")
                    .replacingOccurrences(of: "m", with: "")
                    .trimmingCharacters(in: .whitespaces)
                return Double(numStr) ?? 1.0
            }
            return 1.0
        }

        // Process placemarks
        for pm in placemarks {
            let point = GPSPoint(
                lat: pm.lat, lon: pm.lon, alt: pm.alt,
                accuracy: parseAccuracy(from: pm.description),
                fixType: 1, timestamp: Date()
            )

            if pm.name == "Start" || pm.name.lowercased().contains("start wand") {
                config.setStartWand(point)
            } else if pm.name == "Finish L" || pm.name.lowercased().contains("finish l") {
                config.setFinishLeft(point)
            } else if pm.name == "Finish R" || pm.name.lowercased().contains("finish r") {
                config.setFinishRight(point)
            } else if pm.name.hasPrefix("Gate ") {
                // Parse gate number
                let numStr = pm.name.replacingOccurrences(of: "Gate ", with: "")
                guard let gateNum = Int(numStr) else { continue }

                // Determine color from style URL or description
                let color: GateColor = pm.styleUrl.contains("red") || pm.description.contains("(red") ? .red : .blue

                // Determine gate type from description
                let gateType: GateType
                if pm.description.contains("Outside") {
                    gateType = .panelOutside
                } else if pm.description.contains("Pole)") && !pm.description.contains("Inside") {
                    gateType = .pole
                } else if discipline == .sl {
                    gateType = .pole
                } else {
                    gateType = .panelInside
                }

                let gate = GateEntry(number: gateNum, color: color, type: gateType, position: point)
                config.addGate(gate)
            }
        }

        // Only return if we got at least some gates
        guard !config.gates.isEmpty else { return nil }

        return config
    }

    private func parseDiscipline(from desc: String) -> Discipline {
        let lower = desc.lowercased()
        if lower.contains("slalom") && !lower.contains("giant") && !lower.contains("super") {
            return .sl
        } else if lower.contains("super") {
            return .sg
        }
        return .gs  // Default to GS
    }

    private func parseSite(from desc: String) -> Site {
        let lower = desc.lowercased()
        if lower.contains("sunapee") { return .sunapee }
        if lower.contains("ragged") { return .ragged }
        if lower.contains("proctor") { return .proctor }
        return .other
    }
}

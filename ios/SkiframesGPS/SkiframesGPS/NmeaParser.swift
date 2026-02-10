import Foundation

// MARK: - Data Structures

struct GGAData {
    let utcTime: String
    let latitude: Double
    let longitude: Double
    let fixQuality: Int    // 0=invalid, 1=GPS, 2=DGPS, 4=RTK Fixed, 5=RTK Float
    let satellites: Int
    let hdop: Double
    let altitude: Double
}

struct GSTData {
    let utcTime: String
    let latitudeError: Double   // meters, 1-sigma
    let longitudeError: Double  // meters, 1-sigma
    let altitudeError: Double   // meters, 1-sigma

    var horizontalAccuracy: Double {
        sqrt(latitudeError * latitudeError + longitudeError * longitudeError)
    }
}

struct GnssData {
    let latitude: Double
    let longitude: Double
    let altitude: Double
    let horizontalAccuracy: Double  // meters
    let fixType: Int                // 0,1,2,4,5
    let timestamp: Date
    let hdop: Double
    let satelliteCount: Int

    var fixTypeLabel: String {
        switch fixType {
        case 4: return "RTK Fixed"
        case 5: return "RTK Float"
        case 2: return "DGPS"
        case 1: return "GPS"
        default: return "No Fix"
        }
    }
}

// MARK: - NMEA Parser

struct NmeaParser {

    /// Validate NMEA checksum: XOR of all bytes between '$' and '*'
    static func validateChecksum(_ sentence: String) -> Bool {
        guard let starIdx = sentence.firstIndex(of: "*"),
              sentence.first == "$" else { return false }

        let bodyStart = sentence.index(after: sentence.startIndex) // skip '$'
        let body = String(sentence[bodyStart..<starIdx])
        let checksumStr = String(sentence[sentence.index(after: starIdx)...])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard let expected = UInt8(checksumStr, radix: 16) else { return false }

        var xor: UInt8 = 0
        for byte in body.utf8 {
            xor ^= byte
        }
        return xor == expected
    }

    /// Compute NMEA checksum for sentence body (without '$' prefix and '*' suffix)
    static func computeChecksum(_ body: String) -> String {
        var xor: UInt8 = 0
        for byte in body.utf8 {
            xor ^= byte
        }
        return String(format: "%02X", xor)
    }

    /// Add checksum and CRLF terminator to a raw NMEA sentence starting with '$'
    static func addChecksumAndTerminator(_ sentence: String) -> String? {
        guard sentence.first == "$" else { return nil }
        let body = String(sentence.dropFirst()) // remove '$'
        let checksum = computeChecksum(body)
        return "\(sentence)*\(checksum)\r\n"
    }

    /// Split NMEA sentence into fields and checksum
    static func splitSentence(_ raw: String) -> (fields: [String], checksum: String)? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.first == "$" else { return nil }

        // Remove '$' and split by '*'
        let withoutDollar = String(trimmed.dropFirst())
        let parts = withoutDollar.split(separator: "*", maxSplits: 1).map(String.init)
        guard parts.count >= 1 else { return nil }

        let fields = parts[0].split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        let checksum = parts.count > 1 ? parts[1] : ""

        return (fields, checksum)
    }

    /// Parse NMEA coordinate from ddmm.mmmm (lat) or dddmm.mmmm (lon) to decimal degrees
    static func parseCoordinate(_ raw: String, _ direction: String) -> Double? {
        guard !raw.isEmpty, let value = Double(raw) else { return nil }
        let degrees = floor(value / 100.0)
        let minutes = value - (degrees * 100.0)
        var result = degrees + minutes / 60.0
        if direction == "S" || direction == "W" { result = -result }
        return result
    }

    /// Parse GGA sentence
    /// $GNGGA,hhmmss.ss,lat,N/S,lon,E/W,quality,numSV,HDOP,alt,M,...
    static func parseGGA(_ fields: [String]) -> GGAData? {
        guard fields.count >= 10 else { return nil }

        let type = fields[0]
        guard type.hasSuffix("GGA") else { return nil }

        let utcTime = fields[1]

        guard let latitude = parseCoordinate(fields[2], fields[3]),
              let longitude = parseCoordinate(fields[4], fields[5]) else { return nil }

        let fixQuality = Int(fields[6]) ?? 0
        let satellites = Int(fields[7]) ?? 0
        let hdop = Double(fields[8]) ?? 99.0
        let altitude = Double(fields[9]) ?? 0.0

        return GGAData(
            utcTime: utcTime,
            latitude: latitude,
            longitude: longitude,
            fixQuality: fixQuality,
            satellites: satellites,
            hdop: hdop,
            altitude: altitude
        )
    }

    /// Parse GST sentence
    /// $GNGST,hhmmss.ss,rangeRms,smajor,sminor,orient,latErr,lonErr,altErr
    static func parseGST(_ fields: [String]) -> GSTData? {
        guard fields.count >= 8 else { return nil }

        let type = fields[0]
        guard type.hasSuffix("GST") else { return nil }

        let utcTime = fields[1]
        let latErr = Double(fields[6]) ?? 0.0
        let lonErr = Double(fields[7]) ?? 0.0
        let altErr = fields.count > 8 ? (Double(fields[8]) ?? 0.0) : 0.0

        return GSTData(
            utcTime: utcTime,
            latitudeError: latErr,
            longitudeError: lonErr,
            altitudeError: altErr
        )
    }

    /// Determine sentence type from fields (e.g., "GGA", "GST", "RMC")
    static func sentenceType(_ fields: [String]) -> String? {
        guard let first = fields.first, first.count >= 3 else { return nil }
        // Extract last 3 chars: GNGGA → GGA, GPGGA → GGA
        return String(first.suffix(3))
    }
}

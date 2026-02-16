import Foundation
import Combine

/// Manages persistence of CourseConfig JSON files in the app's Documents directory.
/// Files are stored in Documents/courses/{courseId}.json
class CourseStore: ObservableObject {

    @Published var courses: [CourseConfig] = []

    private let directory: URL
    private let encoder = CourseConfig.jsonEncoder
    private let decoder = CourseConfig.jsonDecoder

    // MARK: - Init

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        directory = docs.appendingPathComponent("courses", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        loadAll()
    }

    // MARK: - File Paths

    /// File URL for a given course
    func fileURL(for config: CourseConfig) -> URL {
        directory.appendingPathComponent("\(config.courseId).json")
    }

    // MARK: - CRUD Operations

    /// Load all course configs from disk
    func loadAll() {
        do {
            let files = try FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
            let jsonFiles = files.filter { $0.pathExtension == "json" }

            var loaded: [CourseConfig] = []
            for file in jsonFiles {
                do {
                    let data = try Data(contentsOf: file)
                    let config = try decoder.decode(CourseConfig.self, from: data)
                    loaded.append(config)
                } catch {
                    print("[CourseStore] Failed to load \(file.lastPathComponent): \(error)")
                }
            }

            // Sort by date descending (most recent first)
            courses = loaded.sorted { $0.date > $1.date }
        } catch {
            print("[CourseStore] Failed to list courses directory: \(error)")
            courses = []
        }
    }

    /// Save a course config to disk
    func save(_ config: CourseConfig) {
        var toSave = config

        // Safety net: if a different course already owns this courseId file, deduplicate
        if courses.contains(where: { $0.courseId == toSave.courseId && $0.id != toSave.id }) {
            let formatter = DateFormatter()
            formatter.dateFormat = "HHmmss"
            toSave.courseId += "_\(formatter.string(from: Date()))"
            print("[CourseStore] Duplicate courseId detected, renamed to \(toSave.courseId)")
        }

        do {
            let data = try encoder.encode(toSave)
            let url = fileURL(for: toSave)
            try data.write(to: url, options: .atomic)

            // Update in-memory list
            if let idx = courses.firstIndex(where: { $0.id == toSave.id }) {
                courses[idx] = toSave
            } else {
                courses.insert(toSave, at: 0)
            }
            // Re-sort
            courses.sort { $0.date > $1.date }
        } catch {
            print("[CourseStore] Failed to save \(toSave.courseId): \(error)")
        }
    }

    /// Delete a course config from disk and memory
    func delete(_ config: CourseConfig) {
        let url = fileURL(for: config)
        try? FileManager.default.removeItem(at: url)
        courses.removeAll { $0.id == config.id }
    }

    /// Delete at offsets (for SwiftUI List onDelete)
    func delete(at offsets: IndexSet) {
        for index in offsets {
            let config = courses[index]
            let url = fileURL(for: config)
            try? FileManager.default.removeItem(at: url)
        }
        courses.remove(atOffsets: offsets)
    }

    // MARK: - Import / Export

    /// Import a course config from an external JSON file URL
    func importCourse(from url: URL) -> CourseConfig? {
        do {
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }

            let data = try Data(contentsOf: url)
            var config = try decoder.decode(CourseConfig.self, from: data)

            // Check for duplicate courseId, append suffix if needed
            let existingIds = Set(courses.map(\.courseId))
            if existingIds.contains(config.courseId) {
                let formatter = DateFormatter()
                formatter.dateFormat = "HHmm"
                config.courseId += "_\(formatter.string(from: Date()))"
            }

            save(config)
            return config
        } catch {
            print("[CourseStore] Failed to import from \(url): \(error)")
            return nil
        }
    }

    /// Import a course from a KML file (e.g., exported to Google Earth)
    func importFromKML(url: URL) -> CourseConfig? {
        do {
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }

            let data = try Data(contentsOf: url)
            guard let config = Self.parseCourseFromKML(data: data) else {
                print("[CourseStore] Failed to parse KML from \(url)")
                return nil
            }

            var toImport = config
            // Deduplicate courseId
            let existingIds = Set(courses.map(\.courseId))
            if existingIds.contains(toImport.courseId) {
                let formatter = DateFormatter()
                formatter.dateFormat = "HHmm"
                toImport.courseId += "_\(formatter.string(from: Date()))"
            }

            save(toImport)
            return toImport
        } catch {
            print("[CourseStore] Failed to import KML from \(url): \(error)")
            return nil
        }
    }

    /// Parse a CourseConfig from KML data
    static func parseCourseFromKML(data: Data) -> CourseConfig? {
        let parser = KMLCourseParser(data: data)
        return parser.parse()
    }

    /// Get a shareable file URL for a course (creates a temporary copy if needed)
    func exportURL(for config: CourseConfig) -> URL? {
        let url = fileURL(for: config)
        guard FileManager.default.fileExists(atPath: url.path) else {
            // Save first if file doesn't exist
            save(config)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
        return url
    }
}

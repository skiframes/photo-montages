import SwiftUI
import UniformTypeIdentifiers

/// List of saved course configurations with create, delete, import, and share actions.
struct CourseListView: View {
    @ObservedObject var receiver: BadElfReceiver
    @ObservedObject var store: CourseStore

    @State private var showNewCourse = false
    @State private var showImporter = false
    @State private var navigationPath = NavigationPath()

    var body: some View {
        Group {
            if store.courses.isEmpty {
                emptyState
            } else {
                courseList
            }
        }
        .navigationTitle("Gates Map")
        .navigationDestination(for: CourseConfig.self) { course in
            CourseDetailView(receiver: receiver, store: store, course: course)
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        showNewCourse = true
                    } label: {
                        Label("New Course", systemImage: "plus")
                    }
                    Button {
                        showImporter = true
                    } label: {
                        Label("Import JSON / KML", systemImage: "square.and.arrow.down")
                    }
                } label: {
                    Image(systemName: "plus")
                }
            }
        }
        .sheet(isPresented: $showNewCourse) {
            NewCourseSheet(store: store) { newCourse in
                navigationPath.append(newCourse)
            }
        }
        .fileImporter(
            isPresented: $showImporter,
            allowedContentTypes: [UTType.json, UTType.xml, UTType.data],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                let imported: CourseConfig?
                let ext = url.pathExtension.lowercased()
                if ext == "kml" || ext == "xml" {
                    imported = store.importFromKML(url: url)
                } else {
                    imported = store.importCourse(from: url)
                }
                if let imported {
                    navigationPath.append(imported)
                }
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "mappin.and.ellipse")
                .font(.system(size: 60))
                .foregroundColor(.secondary.opacity(0.5))

            Text("No Courses")
                .font(.title2.weight(.semibold))

            Text("Create a course to start mapping\ngate GPS positions.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Button {
                showNewCourse = true
            } label: {
                Label("New Course", systemImage: "plus")
                    .font(.headline)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    // MARK: - Course List

    private var courseList: some View {
        List {
            ForEach(store.courses) { course in
                NavigationLink(value: course) {
                    CourseRow(course: course)
                }
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) {
                        store.delete(course)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
                .swipeActions(edge: .leading) {
                    if let url = store.exportURL(for: course) {
                        ShareLink(item: url) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        .tint(.blue)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }
}

// MARK: - Course Row

struct CourseRow: View {
    let course: CourseConfig

    private var dateText: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: course.date)
    }

    var body: some View {
        HStack(spacing: 12) {
            // Discipline badge
            Text(course.discipline.shortName)
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(disciplineColor)
                .cornerRadius(6)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(course.courseName)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.primary)

                    if !course.groups.isEmpty {
                        Text(course.groupsDisplay)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Color(.quaternarySystemFill))
                            .cornerRadius(4)
                    }
                }

                HStack(spacing: 8) {
                    if course.site != .other {
                        Text(course.site.shortName)
                            .font(.system(size: 13))
                            .foregroundColor(.secondary)
                    }

                    Text(dateText)
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)

                    Text(course.summary)
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }
            }

            Spacer()
        }
        .padding(.vertical, 4)
    }

    private var disciplineColor: Color {
        switch course.discipline {
        case .sl: return .blue
        case .gs: return .orange
        case .sg: return .red
        }
    }
}

#Preview {
    NavigationStack {
        CourseListView(receiver: BadElfReceiver(), store: CourseStore())
    }
}

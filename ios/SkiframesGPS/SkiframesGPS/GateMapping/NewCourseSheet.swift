import SwiftUI

/// Modal sheet for creating a new course configuration.
/// Site → Course Name (cascading), Discipline, Group buttons.
struct NewCourseSheet: View {
    @ObservedObject var store: CourseStore
    var onCreated: (CourseConfig) -> Void

    @Environment(\.dismiss) private var dismiss

    // Site selection
    @State private var selectedSite: Site = .ragged
    @State private var customSiteName = ""

    // Course name — either picked from site presets or custom
    @State private var selectedCourseName: String = "Flying Yankee"
    @State private var isCustomCourseName = false
    @State private var customCourseName = ""

    // Discipline
    @State private var discipline: Discipline = .gs

    // Groups (multi-select)
    @State private var selectedGroups: Set<TrainingGroup> = []

    // Date
    @State private var date = Date()

    var body: some View {
        NavigationStack {
            Form {
                // MARK: - Site
                Section("Site") {
                    Picker("Site", selection: $selectedSite) {
                        ForEach(Site.allCases) { site in
                            Text(site.displayName).tag(site)
                        }
                    }

                    if selectedSite == .other {
                        TextField("Site Name", text: $customSiteName)
                            .textInputAutocapitalization(.words)
                    }
                }
                .onChange(of: selectedSite) { _ in
                    // Reset course name when site changes
                    let courses = selectedSite.courseNames
                    if courses.isEmpty {
                        isCustomCourseName = true
                        selectedCourseName = ""
                    } else {
                        isCustomCourseName = false
                        selectedCourseName = courses[0]
                    }
                }

                // MARK: - Course Name
                Section("Course Name") {
                    let courseOptions = selectedSite.courseNames
                    if !courseOptions.isEmpty {
                        Picker("Trail", selection: $selectedCourseName) {
                            ForEach(courseOptions, id: \.self) { name in
                                Text(name).tag(name)
                            }
                            Text("Other...").tag("__other__")
                        }
                        .onChange(of: selectedCourseName) { newVal in
                            isCustomCourseName = (newVal == "__other__")
                        }
                    }

                    if isCustomCourseName || courseOptions.isEmpty {
                        TextField("Course Name", text: $customCourseName)
                            .textInputAutocapitalization(.words)
                    }
                }

                // MARK: - Discipline
                Section("Discipline") {
                    Picker("Discipline", selection: $discipline) {
                        ForEach(Discipline.allCases) { disc in
                            Text(disc.displayName).tag(disc)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                // MARK: - Groups (multi-select toggle buttons)
                Section("Group / Training") {
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 8) {
                        ForEach(TrainingGroup.allCases) { group in
                            GroupToggleButton(
                                group: group,
                                isSelected: selectedGroups.contains(group)
                            ) {
                                if selectedGroups.contains(group) {
                                    selectedGroups.remove(group)
                                } else {
                                    selectedGroups.insert(group)
                                }
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }

                // MARK: - Date
                Section("Date & Time") {
                    DatePicker("Date", selection: $date)
                }

                // MARK: - Preview
                Section {
                    HStack {
                        Text("Course ID")
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(previewCourseId)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.6)
                    }
                    if !effectiveGroupsDisplay.isEmpty {
                        HStack {
                            Text("Groups")
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(effectiveGroupsDisplay)
                                .font(.system(size: 13))
                                .foregroundColor(.secondary)
                        }
                    }
                } header: {
                    Text("Preview")
                }
            }
            .navigationTitle("New Course")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Create") {
                        createCourse()
                    }
                    .fontWeight(.semibold)
                    .disabled(effectiveCourseName.isEmpty)
                }
            }
        }
        .presentationDetents([.large])
    }

    // MARK: - Computed

    private var effectiveCourseName: String {
        if isCustomCourseName || selectedSite.courseNames.isEmpty {
            return customCourseName.trimmingCharacters(in: .whitespaces)
        }
        return selectedCourseName
    }

    private var effectiveLocation: String {
        if selectedSite == .other {
            let custom = customSiteName.trimmingCharacters(in: .whitespaces)
            return custom.isEmpty ? "Other" : custom
        }
        return selectedSite.displayName
    }

    private var effectiveGroupsDisplay: String {
        selectedGroups.sorted(by: { $0.rawValue < $1.rawValue })
            .map(\.displayName)
            .joined(separator: ", ")
    }

    private var previewCourseId: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let dateStr = formatter.string(from: date)
        let name = effectiveCourseName
        let slug = name.lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "[^a-z0-9_]", with: "", options: .regularExpression)
        return slug.isEmpty ? "\(dateStr)_\(discipline.rawValue)" : "\(dateStr)_\(discipline.rawValue)_\(slug)"
    }

    // MARK: - Create

    private func createCourse() {
        let name = effectiveCourseName
        guard !name.isEmpty else { return }

        let groupsArray = Array(selectedGroups).sorted(by: { $0.rawValue < $1.rawValue })

        var config = CourseConfig(
            courseName: name,
            discipline: discipline,
            site: selectedSite,
            groups: groupsArray,
            location: effectiveLocation,
            date: date
        )

        // Set the course ID using the formatted values
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        let dateStr = formatter.string(from: date)
        let slug = name.lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "[^a-z0-9_]", with: "", options: .regularExpression)
        config.courseId = "\(dateStr)_\(discipline.rawValue)_\(slug)"

        store.save(config)
        onCreated(config)
        dismiss()
    }
}

// MARK: - Group Toggle Button

struct GroupToggleButton: View {
    let group: TrainingGroup
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(group.displayName)
                .font(.system(size: 14, weight: isSelected ? .bold : .medium))
                .foregroundColor(isSelected ? .white : .primary)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(isSelected ? groupColor : Color(.tertiarySystemBackground))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(isSelected ? groupColor : Color(.separator), lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }

    private var groupColor: Color {
        switch group {
        case .scored: return .purple
        case .u14: return .blue
        case .u12: return .green
        case .u10: return .orange
        case .masters: return .red
        }
    }
}

#Preview {
    NewCourseSheet(store: CourseStore()) { _ in }
}

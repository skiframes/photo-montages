import SwiftUI
import UIKit

/// Modal sheet for editing a single gate's properties: number, color, type, position.
struct GateEditSheet: View {
    @State var gate: GateEntry
    @Binding var course: CourseConfig

    var onSave: (GateEntry) -> Void
    var onDelete: (UUID) -> Void
    var onRecapture: (UUID) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var showDeleteConfirm = false

    var body: some View {
        NavigationStack {
            Form {
                // Gate properties
                Section("Gate Properties") {
                    // Gate number
                    Stepper("Gate \(gate.number)", value: $gate.number, in: 1...200)

                    // Color
                    Picker("Color", selection: $gate.color) {
                        Text("Red").tag(GateColor.red)
                        Text("Blue").tag(GateColor.blue)
                    }
                    .pickerStyle(.segmented)
                }

                // GPS position (read-only)
                Section("GPS Position") {
                    LabeledContent("Latitude") {
                        Text(String(format: "%.9f", gate.position.lat))
                            .font(.system(size: 13, design: .monospaced))
                    }
                    LabeledContent("Longitude") {
                        Text(String(format: "%.9f", gate.position.lon))
                            .font(.system(size: 13, design: .monospaced))
                    }
                    LabeledContent("Altitude") {
                        Text(String(format: "%.1f m", gate.position.alt))
                            .font(.system(size: 13, design: .monospaced))
                    }
                    LabeledContent("Accuracy") {
                        Text(gate.position.accuracyText)
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(accuracyColor(gate.position.accuracy))
                    }
                    LabeledContent("Fix Type") {
                        Text(gate.position.fixTypeLabel)
                            .font(.system(size: 13))
                    }
                    LabeledContent("Captured") {
                        Text(formattedTimestamp)
                            .font(.system(size: 13))
                    }
                }

                // Re-capture button
                Section {
                    Button {
                        onRecapture(gate.id)
                        // Refresh the gate data from course
                        if let updated = course.gates.first(where: { $0.id == gate.id }) {
                            gate = updated
                        }
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    } label: {
                        HStack {
                            Image(systemName: "location.fill")
                            Text("Re-capture GPS Position")
                        }
                    }
                }

                // Delete
                Section {
                    Button(role: .destructive) {
                        showDeleteConfirm = true
                    } label: {
                        HStack {
                            Image(systemName: "trash")
                            Text("Delete Gate")
                        }
                    }
                }
            }
            .navigationTitle("Edit Gate \(gate.number)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        onSave(gate)
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
            .confirmationDialog("Delete Gate \(gate.number)?", isPresented: $showDeleteConfirm) {
                Button("Delete", role: .destructive) {
                    onDelete(gate.id)
                    dismiss()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will remove gate \(gate.number) from the course.")
            }
        }
        .presentationDetents([.medium, .large])
    }

    // MARK: - Helpers

    private var formattedTimestamp: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter.string(from: gate.position.timestamp)
    }

    private func accuracyColor(_ accuracy: Double) -> Color {
        if accuracy <= 0.2 { return .green }
        if accuracy <= 1.0 { return .blue }
        if accuracy <= 3.0 { return .orange }
        return .red
    }
}

struct GateEditSheet_Previews: PreviewProvider {
    @State static var course = CourseConfig(courseName: "Test", discipline: .gs)
    static var previews: some View {
        GateEditSheet(
            gate: GateEntry(
                number: 1,
                color: .red,
                type: .panelInside,
                position: GPSPoint(lat: 43.476, lon: -71.854, alt: 633.7, accuracy: 1.14, fixType: 4)
            ),
            course: $course,
            onSave: { _ in },
            onDelete: { _ in },
            onRecapture: { _ in }
        )
    }
}

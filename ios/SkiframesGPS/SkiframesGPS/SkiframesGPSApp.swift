import SwiftUI

@main
struct SkiframesGPSApp: App {
    @StateObject private var receiver = BadElfReceiver()

    var body: some Scene {
        WindowGroup {
            ContentView(receiver: receiver)
        }
    }
}

struct ContentView: View {
    @ObservedObject var receiver: BadElfReceiver

    var body: some View {
        VStack(spacing: 0) {
            // Status bar showing Bad Elf connection and accuracy
            StatusBar(receiver: receiver)

            // Full-screen WKWebView
            GeolocationWebView(receiver: receiver)
                .ignoresSafeArea(.container, edges: .bottom)
        }
    }
}

// MARK: - Status Bar

struct StatusBar: View {
    @ObservedObject var receiver: BadElfReceiver

    var body: some View {
        HStack(spacing: 8) {
            // Connection indicator
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)

            // Status text
            Text(statusText)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.primary)

            Spacer()

            // Accuracy badge
            if let data = receiver.latestData {
                Text(accuracyText(data))
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(accuracyColor(data).opacity(0.15))
                    .foregroundColor(accuracyColor(data))
                    .cornerRadius(10)
            }

            // Satellite count
            if let data = receiver.latestData, data.satelliteCount > 0 {
                HStack(spacing: 2) {
                    Image(systemName: "antenna.radiowaves.left.and.right")
                        .font(.system(size: 10))
                    Text("\(data.satelliteCount)")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(.systemBackground))
        .overlay(
            Divider(), alignment: .bottom
        )
    }

    private var statusColor: Color {
        if !receiver.isConnected {
            return receiver.latestData != nil ? .orange : .red
        }
        guard let data = receiver.latestData else { return .yellow }
        return data.fixType >= 4 ? .green : .blue
    }

    private var statusText: String {
        if receiver.isConnected {
            return "Bad Elf: \(receiver.fixTypeLabel)"
        } else if receiver.latestData != nil {
            return "iOS GPS (no Bad Elf)"
        } else {
            return "Searching..."
        }
    }

    private func accuracyText(_ data: GnssData) -> String {
        return String(format: "\u{00B1}%.2fm", data.horizontalAccuracy)
    }

    private func accuracyColor(_ data: GnssData) -> Color {
        let acc = data.horizontalAccuracy
        if acc <= 0.2 { return .green }
        if acc <= 1.0 { return .blue }
        if acc <= 3.0 { return .orange }
        return .red
    }
}

#Preview {
    ContentView(receiver: BadElfReceiver())
}

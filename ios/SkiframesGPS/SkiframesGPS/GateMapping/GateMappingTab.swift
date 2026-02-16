import SwiftUI

/// Root view for the Gates Map tab. Owns the CourseStore and wraps CourseListView in a NavigationStack.
struct GateMappingTab: View {
    @ObservedObject var receiver: BadElfReceiver
    @StateObject private var store = CourseStore()

    var body: some View {
        NavigationStack {
            CourseListView(receiver: receiver, store: store)
        }
    }
}

#Preview {
    GateMappingTab(receiver: BadElfReceiver())
}

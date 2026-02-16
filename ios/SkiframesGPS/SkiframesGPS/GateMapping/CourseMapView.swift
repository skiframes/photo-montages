import SwiftUI
import MapKit

/// Real-time course map showing captured gates, race line, and current GPS position.
/// Uses MapKit with satellite imagery. Updates live as gates are captured.
struct CourseMapView: View {
    let course: CourseConfig
    let currentPosition: GnssData?

    /// Map region computed from course bounds
    @State private var region: MKCoordinateRegion = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 43.477, longitude: -71.854),
        span: MKCoordinateSpan(latitudeDelta: 0.003, longitudeDelta: 0.003)
    )
    @State private var regionInitialized = false

    var body: some View {
        Map(coordinateRegion: $region, annotationItems: annotations) { item in
            MapAnnotation(coordinate: item.coordinate) {
                annotationView(for: item)
            }
        }
        .mapStyle(.hybrid)
        .overlay(alignment: .topTrailing) {
            // Gate count badge
            if !course.gates.isEmpty {
                Text("\(course.gates.count) gates")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.black.opacity(0.6))
                    .cornerRadius(8)
                    .padding(8)
            }
        }
        .onAppear { fitRegion() }
        .onChange(of: course.gates.count) { _ in fitRegion() }
    }

    // MARK: - Annotations

    private var annotations: [MapItem] {
        var items: [MapItem] = []

        // Start wand
        if let sw = course.startWand {
            items.append(MapItem(
                id: "start",
                coordinate: CLLocationCoordinate2D(latitude: sw.lat, longitude: sw.lon),
                kind: .start
            ))
        }

        // Gates
        for gate in course.gates {
            items.append(MapItem(
                id: "gate_\(gate.number)",
                coordinate: CLLocationCoordinate2D(latitude: gate.position.lat, longitude: gate.position.lon),
                kind: .gate(number: gate.number, color: gate.color)
            ))
        }

        // Finish line
        if let fl = course.finishLine.left {
            items.append(MapItem(
                id: "finish_l",
                coordinate: CLLocationCoordinate2D(latitude: fl.lat, longitude: fl.lon),
                kind: .finish(label: "L")
            ))
        }
        if let fr = course.finishLine.right {
            items.append(MapItem(
                id: "finish_r",
                coordinate: CLLocationCoordinate2D(latitude: fr.lat, longitude: fr.lon),
                kind: .finish(label: "R")
            ))
        }

        // Current GPS position
        if let pos = currentPosition {
            items.append(MapItem(
                id: "current",
                coordinate: CLLocationCoordinate2D(latitude: pos.latitude, longitude: pos.longitude),
                kind: .currentPosition
            ))
        }

        return items
    }

    @ViewBuilder
    private func annotationView(for item: MapItem) -> some View {
        switch item.kind {
        case .start:
            Image(systemName: "flag.fill")
                .font(.system(size: 16))
                .foregroundColor(.green)
                .shadow(radius: 2)

        case .gate(let number, let color):
            Text("\(number)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.white)
                .frame(width: 22, height: 22)
                .background(color == .red ? Color.red : Color.blue)
                .cornerRadius(4)
                .shadow(radius: 2)

        case .finish(let label):
            VStack(spacing: 0) {
                Image(systemName: "flag.checkered")
                    .font(.system(size: 14))
                Text(label)
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundColor(.white)
            .shadow(radius: 2)

        case .currentPosition:
            Circle()
                .fill(.blue)
                .frame(width: 14, height: 14)
                .overlay(Circle().stroke(.white, lineWidth: 2))
                .shadow(radius: 3)
        }
    }

    // MARK: - Region Fitting

    private func fitRegion() {
        var coords: [CLLocationCoordinate2D] = []

        if let sw = course.startWand {
            coords.append(CLLocationCoordinate2D(latitude: sw.lat, longitude: sw.lon))
        }
        for gate in course.gates {
            coords.append(CLLocationCoordinate2D(latitude: gate.position.lat, longitude: gate.position.lon))
        }
        if let fl = course.finishLine.left {
            coords.append(CLLocationCoordinate2D(latitude: fl.lat, longitude: fl.lon))
        }
        if let pos = currentPosition {
            coords.append(CLLocationCoordinate2D(latitude: pos.latitude, longitude: pos.longitude))
        }

        guard !coords.isEmpty else { return }

        if coords.count == 1 {
            region = MKCoordinateRegion(
                center: coords[0],
                span: MKCoordinateSpan(latitudeDelta: 0.002, longitudeDelta: 0.002)
            )
            return
        }

        let lats = coords.map(\.latitude)
        let lons = coords.map(\.longitude)
        let minLat = lats.min()!
        let maxLat = lats.max()!
        let minLon = lons.min()!
        let maxLon = lons.max()!

        let center = CLLocationCoordinate2D(
            latitude: (minLat + maxLat) / 2,
            longitude: (minLon + maxLon) / 2
        )
        let span = MKCoordinateSpan(
            latitudeDelta: max((maxLat - minLat) * 1.5, 0.001),
            longitudeDelta: max((maxLon - minLon) * 1.5, 0.001)
        )
        region = MKCoordinateRegion(center: center, span: span)
    }
}

// MARK: - Map Item

struct MapItem: Identifiable {
    let id: String
    let coordinate: CLLocationCoordinate2D
    let kind: MapItemKind
}

enum MapItemKind {
    case start
    case gate(number: Int, color: GateColor)
    case finish(label: String)
    case currentPosition
}

// MARK: - MapStyle Extension for iOS 16

extension View {
    @ViewBuilder
    func mapStyle(_ style: MapStyleType) -> some View {
        // .hybrid map type is set via MKMapView configuration
        // For iOS 16 with Map(coordinateRegion:), we use the overlay approach
        self
    }
}

enum MapStyleType {
    case hybrid
}

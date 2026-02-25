#!/usr/bin/env python3
"""
Slope angle calculator for ski racing video analysis.

Calculates the apparent slope angle (fall line direction) as it appears
in the camera view, for use in body position/pose analysis.

Supports two calibration methods:
1. GPS-only: Uses gate and camera GPS coordinates to compute slope plane
2. Combined GPS + Poles: Adds pole top/bottom pixel positions for camera pose refinement

The combined method is more accurate because it calibrates the camera's
actual orientation, not just its position.
"""

import math
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path


# WGS84 constants for GPS-to-local conversion
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2


@dataclass
class GPSPoint:
    """GPS coordinate with altitude."""
    lat: float
    lon: float
    alt: float

    def to_enu(self, origin: 'GPSPoint') -> Tuple[float, float, float]:
        """Convert to local East-North-Up meters relative to origin."""
        lat_rad = math.radians(origin.lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad) ** 2)
        meters_per_deg_lat = math.pi / 180 * N * (1 - WGS84_E2) / (1 - WGS84_E2 * math.sin(lat_rad) ** 2)
        meters_per_deg_lon = math.pi / 180 * N * math.cos(lat_rad)

        east = (self.lon - origin.lon) * meters_per_deg_lon
        north = (self.lat - origin.lat) * meters_per_deg_lat
        up = self.alt - origin.alt

        return (east, north, up)


@dataclass
class PoleMarker:
    """A gate pole with pixel positions and GPS."""
    gate_id: int
    color: str  # "red" or "blue"
    pixel_top: Tuple[float, float]
    pixel_base: Tuple[float, float]
    gps: GPSPoint
    pole_height_m: float = 1.83  # 72 inches


class SlopeCalculator:
    """
    Calculate apparent slope angle in camera view.

    Uses GPS coordinates to determine 3D slope geometry, then projects
    the fall line direction onto the camera image plane.
    """

    def __init__(self):
        self.gates: List[GPSPoint] = []
        self.camera_gps: Optional[GPSPoint] = None
        self.poles: List[PoleMarker] = []
        self.frame_size: Tuple[int, int] = (1920, 1080)  # width, height
        self.origin: Optional[GPSPoint] = None

        # Computed values
        self.slope_plane_normal: Optional[np.ndarray] = None
        self.fall_line_3d: Optional[np.ndarray] = None
        self.camera_rotation: Optional[np.ndarray] = None

    def add_gate_gps(self, gate_id: int, lat: float, lon: float, alt: float, color: str = "red"):
        """Add a gate GPS position."""
        gps = GPSPoint(lat, lon, alt)
        self.gates.append(gps)
        if self.origin is None:
            self.origin = gps

    def set_camera_gps(self, lat: float, lon: float, alt: float):
        """Set camera GPS position."""
        self.camera_gps = GPSPoint(lat, lon, alt)

    def add_pole_marker(self, gate_id: int, color: str,
                        pixel_top: Tuple[float, float],
                        pixel_base: Tuple[float, float],
                        lat: float, lon: float, alt: float,
                        pole_height_m: float = 1.83):
        """Add a pole with both pixel and GPS coordinates for camera calibration."""
        pole = PoleMarker(
            gate_id=gate_id,
            color=color,
            pixel_top=pixel_top,
            pixel_base=pixel_base,
            gps=GPSPoint(lat, lon, alt),
            pole_height_m=pole_height_m
        )
        self.poles.append(pole)

    def set_frame_size(self, width: int, height: int):
        """Set video frame dimensions."""
        self.frame_size = (width, height)

    def compute_slope_plane(self) -> Optional[np.ndarray]:
        """
        Fit a plane to gate positions to find slope surface.

        Returns:
            Normal vector of slope plane (pointing upward from snow)
        """
        if len(self.gates) < 3 or self.origin is None:
            print(f"Need at least 3 gates for slope plane (have {len(self.gates)})")
            return None

        # Convert gates to local ENU coordinates
        points = []
        for gate in self.gates:
            enu = gate.to_enu(self.origin)
            points.append(enu)
        points = np.array(points)

        # Fit plane using SVD
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, Vt = np.linalg.svd(centered)

        # Normal is the last singular vector (smallest variance direction)
        normal = Vt[-1]

        # Ensure normal points upward (positive Z component)
        if normal[2] < 0:
            normal = -normal

        self.slope_plane_normal = normal
        return normal

    def compute_fall_line(self) -> Optional[np.ndarray]:
        """
        Compute the fall line direction (steepest descent) on the slope plane.

        The fall line is the direction of steepest descent on the slope,
        which is the projection of the gravity vector onto the slope plane.

        Returns:
            3D unit vector in the fall line direction (in local ENU coords)
        """
        if self.slope_plane_normal is None:
            self.compute_slope_plane()

        if self.slope_plane_normal is None:
            return None

        n = self.slope_plane_normal

        # Gravity points downward: (0, 0, -1)
        gravity = np.array([0, 0, -1])

        # Project gravity onto plane: g - (g·n)n
        fall_line = gravity - np.dot(gravity, n) * n

        # Normalize
        length = np.linalg.norm(fall_line)
        if length < 1e-6:
            # Slope is nearly horizontal
            return np.array([0, -1, 0])  # Default to north-facing

        self.fall_line_3d = fall_line / length
        return self.fall_line_3d

    def compute_slope_pitch(self) -> float:
        """
        Compute the slope pitch angle (steepness) in degrees.

        This is the angle of the slope from horizontal.
        """
        if self.slope_plane_normal is None:
            self.compute_slope_plane()

        if self.slope_plane_normal is None:
            return 0.0

        # Pitch is the angle between normal and vertical (Z axis)
        vertical = np.array([0, 0, 1])
        cos_angle = np.dot(self.slope_plane_normal, vertical)
        pitch_rad = math.acos(np.clip(cos_angle, -1, 1))

        # Convert to degrees (0° = horizontal, 90° = vertical cliff)
        return math.degrees(pitch_rad)

    def estimate_camera_pose(self) -> Optional[np.ndarray]:
        """
        Estimate camera rotation matrix from pole markers.

        Uses the known 3D positions (from GPS) and 2D image positions
        (from pole drawing) to solve for camera orientation.

        Returns:
            3x3 rotation matrix (camera orientation)
        """
        if len(self.poles) < 2 or self.origin is None:
            print(f"Need at least 2 poles for camera pose (have {len(self.poles)})")
            return None

        # Build correspondence points
        # For each pole: base has known GPS, top is GPS + height in up direction
        world_points = []
        image_points = []

        for pole in self.poles:
            # Base point in local ENU
            base_enu = pole.gps.to_enu(self.origin)
            world_points.append(base_enu)
            image_points.append(pole.pixel_base)

            # Top point: add pole height in local "up" direction
            # Note: pole is vertical, so top = base + (0, 0, height)
            top_enu = (base_enu[0], base_enu[1], base_enu[2] + pole.pole_height_m)
            world_points.append(top_enu)
            image_points.append(pole.pixel_top)

        world_points = np.array(world_points, dtype=np.float64)
        image_points = np.array(image_points, dtype=np.float64)

        # Camera matrix (approximate, can be refined)
        w, h = self.frame_size
        focal_length = max(w, h) * 1.2  # Approximate
        cx, cy = w / 2, h / 2
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Camera position in local ENU
        if self.camera_gps is None:
            print("Camera GPS position not set")
            return None

        cam_enu = np.array(self.camera_gps.to_enu(self.origin))

        # Use cv2.solvePnP if available
        try:
            import cv2

            # solvePnP needs at least 4 points for RANSAC, use SOLVEPNP_ITERATIVE for fewer
            if len(world_points) >= 4:
                success, rvec, tvec = cv2.solvePnP(
                    world_points, image_points, camera_matrix, None,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    world_points, image_points, camera_matrix, None,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                self.camera_rotation = R
                return R
            else:
                print("solvePnP failed")
                return None

        except ImportError:
            print("OpenCV not available, using simplified camera model")
            return self._estimate_camera_rotation_simple()

    def _estimate_camera_rotation_simple(self) -> Optional[np.ndarray]:
        """
        Simple camera rotation estimate when OpenCV not available.

        Uses the direction from camera to gate centroid as the viewing direction.
        """
        if self.camera_gps is None or self.origin is None or not self.gates:
            return None

        cam_enu = np.array(self.camera_gps.to_enu(self.origin))

        # Compute centroid of gates
        gate_centroid = np.mean([np.array(g.to_enu(self.origin)) for g in self.gates], axis=0)

        # View direction: from camera to gates
        view_dir = gate_centroid - cam_enu
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Build rotation matrix: Z = view direction, Y = up, X = right
        # This is a simplified model assuming camera is roughly level
        z_axis = view_dir
        y_approx = np.array([0, 0, 1])  # World up
        x_axis = np.cross(y_approx, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])
        self.camera_rotation = R
        return R

    def project_to_image(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Project a 3D point (in local ENU) to image coordinates.

        Uses the estimated camera pose and a simple pinhole model.
        """
        if self.camera_rotation is None:
            self._estimate_camera_rotation_simple()

        if self.camera_gps is None or self.origin is None:
            return (0, 0)

        cam_enu = np.array(self.camera_gps.to_enu(self.origin))

        # Transform point to camera coordinates
        p_world = point_3d - cam_enu
        p_cam = self.camera_rotation.T @ p_world

        # Project (pinhole model)
        w, h = self.frame_size
        focal = max(w, h) * 1.2

        if abs(p_cam[2]) < 1e-6:
            return (0, 0)

        x = focal * p_cam[0] / p_cam[2] + w / 2
        y = focal * (-p_cam[1]) / p_cam[2] + h / 2  # Flip Y for image coords

        return (x, y)

    def compute_apparent_slope_angle(self) -> Dict:
        """
        Compute the apparent slope angle as it appears in the camera image.

        This is the angle of the fall line direction when projected onto
        the 2D image plane, measured from horizontal (image X axis).

        Returns:
            Dict with:
            - slope_angle_deg: Apparent angle in image (degrees from horizontal)
            - pitch_deg: Actual slope pitch (steepness)
            - fall_line_direction: 2D unit vector in image
            - method: Calibration method used
        """
        result = {
            'slope_angle_deg': 0.0,
            'pitch_deg': 0.0,
            'fall_line_direction': (0, 1),
            'cross_slope_deg': 0.0,
            'method': 'gps_only'
        }

        # Compute slope geometry from GPS
        self.compute_slope_plane()
        self.compute_fall_line()

        if self.fall_line_3d is None or self.origin is None:
            return result

        # Compute pitch (steepness)
        result['pitch_deg'] = round(self.compute_slope_pitch(), 1)

        # Use poles for camera pose if available
        if len(self.poles) >= 2:
            self.estimate_camera_pose()
            result['method'] = 'gps_poles_combined'
        else:
            self._estimate_camera_rotation_simple()
            result['method'] = 'gps_only'

        # Project fall line to image
        # Use two points along the fall line
        if self.camera_gps is None:
            return result

        # Get a point on the slope (e.g., average gate position)
        if self.gates:
            ref_point = np.mean([np.array(g.to_enu(self.origin)) for g in self.gates], axis=0)
        else:
            ref_point = np.array([0, 0, 0])

        # Two points along fall line
        p1 = ref_point
        p2 = ref_point + self.fall_line_3d * 10  # 10m along fall line

        # Project to image
        img_p1 = self.project_to_image(p1)
        img_p2 = self.project_to_image(p2)

        # Compute angle in image
        dx = img_p2[0] - img_p1[0]
        dy = img_p2[1] - img_p1[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            # Fall line projects to a point (unlikely)
            return result

        # Angle from horizontal (positive = sloping down-right)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize to -90 to +90 range
        while angle_deg > 90:
            angle_deg -= 180
        while angle_deg < -90:
            angle_deg += 180

        result['slope_angle_deg'] = round(angle_deg, 1)

        # Fall line direction as unit vector
        length = math.sqrt(dx ** 2 + dy ** 2)
        result['fall_line_direction'] = (round(dx / length, 3), round(dy / length, 3))

        # Cross-slope component (angle of slope across the frame)
        # This is derived from the slope plane normal
        if self.slope_plane_normal is not None:
            n = self.slope_plane_normal
            # Cross-slope is how much the slope tilts left-right
            cross_slope_rad = math.atan2(n[0], n[2])  # East / Up component
            result['cross_slope_deg'] = round(math.degrees(cross_slope_rad), 1)

        return result

    def summary(self) -> str:
        """Generate human-readable summary of slope calculation."""
        lines = ["═══ Slope Calculation Summary ═══"]

        lines.append(f"Gates: {len(self.gates)}")
        lines.append(f"Poles with pixel markers: {len(self.poles)}")

        if self.camera_gps:
            lines.append(f"Camera: ({self.camera_gps.lat:.6f}, {self.camera_gps.lon:.6f})")

        result = self.compute_apparent_slope_angle()

        lines.append("")
        lines.append(f"Method: {result['method']}")
        lines.append(f"Slope pitch (steepness): {result['pitch_deg']}°")
        lines.append(f"Cross-slope: {result['cross_slope_deg']}°")
        lines.append(f"Apparent slope angle in image: {result['slope_angle_deg']}°")
        lines.append(f"Fall line direction (image): {result['fall_line_direction']}")

        return "\n".join(lines)

    @classmethod
    def from_race_manifest(cls, manifest_path: str, camera_id: str = "Cam1",
                           run: str = "run2") -> 'SlopeCalculator':
        """
        Load slope calculator from a race manifest JSON file.

        Args:
            manifest_path: Path to race_manifest.json
            camera_id: Camera ID (e.g., "Cam1")
            run: Which run's gates to use ("run1" or "run2")

        Returns:
            Configured SlopeCalculator
        """
        with open(manifest_path) as f:
            manifest = json.load(f)

        calc = cls()

        # Load gates
        gates_data = manifest.get('course', {}).get(run, {}).get('gates', [])
        for gate in gates_data:
            calc.add_gate_gps(
                gate_id=gate['number'],
                lat=gate['lat'],
                lon=gate['lon'],
                alt=gate.get('dem_elev', gate.get('elev', 0)),
                color=gate.get('color', 'red')
            )

        # Load camera
        cameras = manifest.get('cameras', [])
        for cam in cameras:
            if cam['id'] == camera_id:
                pos = cam.get('position', {})
                if pos:
                    calc.set_camera_gps(
                        lat=pos['lat'],
                        lon=pos['lon'],
                        alt=pos.get('dem_elev', pos.get('elev', 0))
                    )

                # Get gates covered by this camera
                gates_covered = cam.get(f'gates_covered_{run}', [])
                # Could filter to just these gates if needed
                break

        return calc

    def save_config(self, path: str):
        """Save configuration to JSON."""
        config = {
            'gates': [{'lat': g.lat, 'lon': g.lon, 'alt': g.alt} for g in self.gates],
            'camera': {'lat': self.camera_gps.lat, 'lon': self.camera_gps.lon,
                       'alt': self.camera_gps.alt} if self.camera_gps else None,
            'poles': [{
                'gate_id': p.gate_id,
                'color': p.color,
                'pixel_top': list(p.pixel_top),
                'pixel_base': list(p.pixel_base),
                'gps': {'lat': p.gps.lat, 'lon': p.gps.lon, 'alt': p.gps.alt},
                'pole_height_m': p.pole_height_m
            } for p in self.poles],
            'frame_size': list(self.frame_size)
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> 'SlopeCalculator':
        """Load configuration from JSON."""
        with open(path) as f:
            config = json.load(f)

        calc = cls()

        for g in config.get('gates', []):
            calc.gates.append(GPSPoint(g['lat'], g['lon'], g['alt']))

        if config.get('camera'):
            c = config['camera']
            calc.camera_gps = GPSPoint(c['lat'], c['lon'], c['alt'])

        for p in config.get('poles', []):
            calc.add_pole_marker(
                gate_id=p['gate_id'],
                color=p['color'],
                pixel_top=tuple(p['pixel_top']),
                pixel_base=tuple(p['pixel_base']),
                lat=p['gps']['lat'],
                lon=p['gps']['lon'],
                alt=p['gps']['alt'],
                pole_height_m=p.get('pole_height_m', 1.83)
            )

        if config.get('frame_size'):
            calc.frame_size = tuple(config['frame_size'])

        if calc.gates:
            calc.origin = calc.gates[0]

        return calc


def load_from_calibration_and_manifest(calibration_path: str, manifest_path: str,
                                        camera_id: str = "Cam1",
                                        run: str = "run2") -> SlopeCalculator:
    """
    Create SlopeCalculator by combining:
    1. Calibration JSON - pole pixel positions (top/base)
    2. Race manifest - gate GPS positions
    3. Camera GPS position

    This is the MOST ACCURATE method because it uses both:
    - GPS for 3D slope geometry
    - Pole markers for camera pose refinement

    Args:
        calibration_path: Path to calibration JSON from calibration UI
        manifest_path: Path to race_manifest.json with GPS data
        camera_id: Camera ID in manifest
        run: Which run's gates to use

    Returns:
        Configured SlopeCalculator with both GPS and pole data
    """
    calc = SlopeCalculator()

    # Load calibration (pole pixel positions)
    with open(calibration_path) as f:
        cal = json.load(f)

    # Load race manifest (GPS positions)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get frame size from calibration
    if 'frame_shape' in cal:
        calc.frame_size = (cal['frame_shape'][1], cal['frame_shape'][0])  # [h,w] -> (w,h)

    # Build gate GPS lookup from manifest
    gate_gps = {}
    gates_data = manifest.get('course', {}).get(run, {}).get('gates', [])
    for gate in gates_data:
        gate_gps[gate['number']] = {
            'lat': gate['lat'],
            'lon': gate['lon'],
            'alt': gate.get('dem_elev', gate.get('elev', 0)),
            'color': gate.get('color', 'red')
        }

    # Load camera GPS
    cameras = manifest.get('cameras', [])
    for cam in cameras:
        if cam['id'] == camera_id:
            pos = cam.get('position', {})
            if pos:
                calc.set_camera_gps(
                    lat=pos['lat'],
                    lon=pos['lon'],
                    alt=pos.get('dem_elev', pos.get('elev', 0))
                )
            break

    # Add gates from manifest (for slope plane calculation)
    for gate_id, gps_data in gate_gps.items():
        calc.add_gate_gps(
            gate_id=gate_id,
            lat=gps_data['lat'],
            lon=gps_data['lon'],
            alt=gps_data['alt'],
            color=gps_data['color']
        )

    # Add pole markers from calibration (pixel positions + GPS)
    for gate in cal.get('gates', []):
        gate_id = gate['id']
        if gate_id in gate_gps:
            gps = gate_gps[gate_id]
            calc.add_pole_marker(
                gate_id=gate_id,
                color=gate.get('color', gps.get('color', 'red')),
                pixel_top=tuple(gate['top']),
                pixel_base=tuple(gate['base']),
                lat=gps['lat'],
                lon=gps['lon'],
                alt=gps['alt'],
                pole_height_m=1.83  # 72 inches
            )
            print(f"  Added pole {gate_id} ({gate['color']}): "
                  f"pixels ({gate['top']}, {gate['base']}) + GPS ({gps['lat']:.6f}, {gps['lon']:.6f})")

    if calc.poles:
        print(f"\nLoaded {len(calc.poles)} poles with combined pixel + GPS data")
        print("Using GPS+POLES method (most accurate)")
    else:
        print("\nNo matching poles found - using GPS-only method")
        print("Tip: Make sure gate IDs in calibration match gate numbers in manifest")

    return calc


def compute_slope_from_pixels_only(calibration_path: str) -> Dict:
    """
    Compute apparent slope angle using ONLY pixel positions from calibration.

    This method is MORE RELIABLE than GPS-based calculation because:
    1. No GPS error (pixel positions are exact)
    2. No camera pose estimation needed
    3. Uses the actual gate positions visible in the image

    The slope direction is estimated from the line connecting gate centers,
    assuming gates are arranged along the fall line (typical GS course).

    Args:
        calibration_path: Path to calibration JSON with gate pixel positions

    Returns:
        Dict with slope_angle_deg, fall_line_direction, method
    """
    import json

    with open(calibration_path) as f:
        cal = json.load(f)

    gates = cal.get('gates', [])
    if len(gates) < 2:
        return {
            'slope_angle_deg': 0.0,
            'fall_line_direction': (0, 1),
            'method': 'pixel_only',
            'error': 'Need at least 2 gates'
        }

    # Group gates by color (assuming 2 poles per gate for GS panel)
    blue_poles = [g for g in gates if g.get('color') == 'blue' and not g.get('dismissed')]
    red_poles = [g for g in gates if g.get('color') == 'red' and not g.get('dismissed')]

    # Calculate center of each gate (average of its poles)
    def gate_center(poles):
        if not poles:
            return None
        bases = [p['base'] for p in poles]
        x = sum(b[0] for b in bases) / len(bases)
        y = sum(b[1] for b in bases) / len(bases)
        return (x, y)

    blue_center = gate_center(blue_poles)
    red_center = gate_center(red_poles)

    if not blue_center or not red_center:
        # Fall back to using first and last gate by Y position
        sorted_gates = sorted(gates, key=lambda g: g['base'][1])  # Sort by Y (top to bottom)
        top_gate = sorted_gates[0]
        bottom_gate = sorted_gates[-1]

        dx = bottom_gate['base'][0] - top_gate['base'][0]
        dy = bottom_gate['base'][1] - top_gate['base'][1]
    else:
        # In GS, skiers go from red to blue (or blue to red, alternating)
        # The gate higher in the image (smaller Y) is uphill
        if red_center[1] < blue_center[1]:
            # Red is uphill, blue is downhill - fall line goes red -> blue
            dx = blue_center[0] - red_center[0]
            dy = blue_center[1] - red_center[1]
        else:
            # Blue is uphill, red is downhill - fall line goes blue -> red
            dx = red_center[0] - blue_center[0]
            dy = red_center[1] - blue_center[1]

    # Angle from vertical (0° = straight down, positive = tilted RIGHT)
    # In image coords: Y increases downward, so "down" is +Y
    # Vertical direction in image is (0, 1)
    # Angle = atan2(dx, dy) gives angle from vertical
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    # Normalize direction vector
    length = math.sqrt(dx**2 + dy**2)
    if length > 0:
        direction = (dx / length, dy / length)
    else:
        direction = (0, 1)

    # Determine direction label
    if abs(angle_deg) < 5:
        dir_label = "vertical"
    elif angle_deg > 0:
        dir_label = f"{abs(angle_deg):.1f}° RIGHT"
    else:
        dir_label = f"{abs(angle_deg):.1f}° LEFT"

    return {
        'slope_angle_deg': round(angle_deg, 1),
        'fall_line_direction': (round(direction[0], 3), round(direction[1], 3)),
        'direction_label': dir_label,
        'method': 'pixel_only',
        'blue_center': blue_center,
        'red_center': red_center
    }


def test_with_western_qualifier():
    """Test with Western Qualifier race data."""
    manifest_path = "/Users/paul2/skiframes/photo-montages/web/races/western-q-2026-02-22/race_manifest.json"

    print("Loading race manifest...")
    calc = SlopeCalculator.from_race_manifest(manifest_path, camera_id="Cam1", run="run2")

    print(f"\nLoaded {len(calc.gates)} gates")

    # Compute and display results
    print("\n" + calc.summary())

    # Test both methods
    print("\n═══ Method Comparison ═══")

    # GPS only
    result_gps = calc.compute_apparent_slope_angle()
    print(f"GPS-only:     slope angle = {result_gps['slope_angle_deg']}°, pitch = {result_gps['pitch_deg']}°")

    # If we had pole markers, we could compare
    # For now, demonstrate adding poles
    print("\nTo improve accuracy, add pole markers in calibration UI:")
    print("  calc.add_pole_marker(gate_id=8, color='red',")
    print("                       pixel_top=(x1,y1), pixel_base=(x2,y2),")
    print("                       lat=43.4276891, lon=-71.8282715, alt=291.6)")

    return calc


if __name__ == "__main__":
    test_with_western_qualifier()

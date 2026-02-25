#!/usr/bin/env python3
"""
Pose analysis module using YOLOv8-pose for ski racing video clips.

YOLOv8-pose is more robust than MediaPipe for:
- Fast motion and blur
- Unusual body positions (skiing stance)
- Distinguishing person from ski poles
- Detection from distance

Keypoints (COCO format):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Deque, Dict
from collections import deque

import cv2
import numpy as np


@dataclass
class SlopeGeometry:
    """
    Full 3D slope geometry for accurate angle calculations.

    The slope is defined by:
    - fall_line_angle_deg: Apparent fall line direction in image (degrees from vertical)
    - pitch_deg: Slope steepness (degrees from horizontal)
    - roll_deg: Cross-slope tilt (degrees)
    - homography: 3x3 matrix to transform image points to slope plane

    This allows calculating body angles relative to the TRUE slope plane,
    not just the 2D projection in the image.
    """
    fall_line_angle_deg: float = 0.0  # Direction in image (- = left, + = right)
    pitch_deg: float = 25.0           # Slope steepness (typical ski slope)
    roll_deg: float = 0.0             # Cross-slope tilt
    homography: Optional[np.ndarray] = None  # Image to slope plane transform

    # Computed 3D vectors (in camera coordinate system)
    slope_normal: Optional[np.ndarray] = None      # Normal to slope surface
    fall_line_3d: Optional[np.ndarray] = None      # Fall line direction in 3D
    cross_slope_3d: Optional[np.ndarray] = None    # Across-slope direction in 3D

    @classmethod
    def from_calibration_file(cls, calibration_path: str) -> 'SlopeGeometry':
        """
        Load slope geometry from a calibration JSON file.

        Extracts:
        - gate_line_axis: 2D fall line direction
        - world_coords: 3D positions for slope plane fitting
        - homography: image-to-slope transform
        """
        with open(calibration_path) as f:
            cal = json.load(f)

        geom = cls()

        # Get fall line direction from gate_line_axis
        if 'gate_line_axis' in cal:
            axis = cal['gate_line_axis']
            # axis is [x, y] direction vector in image
            # Convert to angle from vertical (positive Y is down in image)
            geom.fall_line_angle_deg = math.degrees(math.atan2(axis[0], axis[1]))

        # Get homography if available
        if 'homography' in cal:
            geom.homography = np.array(cal['homography'])

        # Compute slope plane from world_coords if available
        if 'world_coords' in cal and len(cal['world_coords']) >= 3:
            world_pts = np.array(cal['world_coords'])
            geom._fit_slope_plane(world_pts)

        # Compute pitch from gate positions if world_coords not available
        if geom.pitch_deg == 25.0 and 'gates' in cal:
            gates = cal['gates']
            if len(gates) >= 2:
                geom._estimate_pitch_from_gates(gates)

        geom._compute_3d_vectors()

        return geom

    def _fit_slope_plane(self, world_pts: np.ndarray):
        """Fit a plane to world coordinates to determine slope orientation."""
        if len(world_pts) < 3:
            return

        # Check if world_coords are coplanar (all same Z)
        z_values = world_pts[:, 2] if world_pts.shape[1] > 2 else np.zeros(len(world_pts))
        z_range = np.max(z_values) - np.min(z_values)

        if z_range < 0.01:
            # World coords are on the slope plane itself (z=0)
            # Can't determine pitch from this - use default for ski racing
            self.pitch_deg = 28.0  # Typical GS slope steepness
            print(f"  Note: world_coords are coplanar, using default pitch: {self.pitch_deg} deg")
            return

        # Fit plane using SVD
        centroid = world_pts.mean(axis=0)
        centered = world_pts - centroid
        _, _, Vt = np.linalg.svd(centered)

        # Normal is the last singular vector (smallest variance direction)
        normal = Vt[-1]

        # Ensure normal points "up" (positive Z in world coords)
        if normal[2] < 0:
            normal = -normal

        self.slope_normal = normal

        # Calculate pitch from normal (angle from vertical)
        # If normal is [0, 0, 1], pitch is 0 (horizontal)
        # If normal is [0, 1, 0], pitch is 90 (vertical wall)
        vertical = np.array([0, 0, 1])
        cos_angle = np.dot(normal, vertical)
        self.pitch_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

        # Calculate roll (cross-slope tilt)
        # Project normal onto XZ plane and measure angle from Z
        if abs(normal[2]) > 0.01:
            self.roll_deg = math.degrees(math.atan2(normal[0], normal[2]))

    def _estimate_pitch_from_gates(self, gates: List[dict]):
        """Estimate pitch from vertical extent of gates in image."""
        # Get gate heights (top to base pixel distance)
        heights = []
        for g in gates:
            if 'top' in g and 'base' in g:
                dy = g['base'][1] - g['top'][1]
                heights.append(dy)

        if not heights:
            return

        avg_height = sum(heights) / len(heights)
        # Typical GS panel is 1.83m (72"). If it appears shorter, we're viewing at an angle.
        # This is a rough heuristic - actual pitch depends on camera angle
        # A 45° slope viewed straight on shows poles at ~70% of their true length
        # For now, estimate based on typical race slopes (25-35°)
        self.pitch_deg = 28.0  # Default for typical GS slope

    def _compute_3d_vectors(self):
        """Compute 3D direction vectors from angles."""
        # Fall line in 3D: direction of steepest descent on slope
        # In camera coords: X=right, Y=down, Z=into screen
        fall_rad = math.radians(self.fall_line_angle_deg)
        pitch_rad = math.radians(self.pitch_deg)

        # Fall line direction in image plane (ignoring depth)
        fx = math.sin(fall_rad)  # Horizontal component
        fy = math.cos(fall_rad)  # Vertical component (down)

        # Add depth component based on pitch
        # When viewing slope from above, fall line goes "into" screen
        fz = math.sin(pitch_rad) * fy
        fy = math.cos(pitch_rad) * fy

        self.fall_line_3d = np.array([fx, fy, fz])
        self.fall_line_3d /= np.linalg.norm(self.fall_line_3d)

        # Cross-slope is perpendicular to fall line, in the slope plane
        # Approximate as horizontal across the image
        self.cross_slope_3d = np.array([math.cos(fall_rad), -math.sin(fall_rad), 0])
        self.cross_slope_3d /= np.linalg.norm(self.cross_slope_3d)

        # Slope normal from cross product of fall line and cross-slope
        if self.slope_normal is None:
            self.slope_normal = np.cross(self.fall_line_3d, self.cross_slope_3d)
            self.slope_normal /= np.linalg.norm(self.slope_normal)

    def image_to_slope_coords(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Transform image pixel coordinates to slope plane coordinates.

        Returns (along_fall_line, across_slope) in arbitrary units.
        Uses homography if available, otherwise projects using angles.
        """
        if self.homography is not None:
            # Apply homography
            pt = np.array([pixel_x, pixel_y, 1.0])
            result = self.homography @ pt
            if abs(result[2]) > 1e-6:
                return (result[0] / result[2], result[1] / result[2])

        # Fallback: simple rotation by fall line angle
        fall_rad = math.radians(self.fall_line_angle_deg)
        cos_a, sin_a = math.cos(fall_rad), math.sin(fall_rad)

        # Rotate so fall line is vertical
        along = pixel_x * sin_a + pixel_y * cos_a
        across = pixel_x * cos_a - pixel_y * sin_a

        return (along, across)

    def angle_to_slope_line(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate angle of a line segment relative to the slope line.

        The slope line is perpendicular to the fall line in the image.
        0° = parallel to slope line (across the hill)
        90° = parallel to fall line (down the hill)

        Args:
            p1, p2: Pixel coordinates of line endpoints

        Returns:
            Angle in degrees from slope line. Positive = rotated toward fall line.
        """
        # Calculate angle of line in image coordinates
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        # Angle from horizontal in image
        line_angle = math.degrees(math.atan2(dy, dx))

        # Slope line angle = fall line angle - 90 (perpendicular)
        # Fall line points "down" at fall_line_angle from vertical
        # In image coords, vertical down = 90° from horizontal
        slope_line_angle = self.fall_line_angle_deg  # Slope line is at this angle from horizontal

        # Angle relative to slope line
        angle = line_angle - slope_line_angle

        # Normalize to -90 to +90
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180

        return angle

    def edge_angle_from_tibia(self, knee: Tuple[float, float], ankle: Tuple[float, float]) -> float:
        """
        Calculate ski edge angle from tibia orientation.

        Edge angle is how much the ski base is tilted from the slope surface.
        If tibia is perpendicular to slope, edge angle is 0.

        Args:
            knee, ankle: Pixel coordinates

        Returns:
            Edge angle in degrees. Positive = edged toward the hill.
        """
        # Tibia direction in image
        dx = ankle[0] - knee[0]
        dy = ankle[1] - knee[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        # Tibia angle from vertical in image (0 = straight down)
        tibia_angle = math.degrees(math.atan2(dx, dy))

        # "Perpendicular to slope" in image is at (fall_line_angle) from vertical
        # Because fall line is the steepest descent direction
        slope_perpendicular = self.fall_line_angle_deg

        # Edge angle = tibia deviation from slope perpendicular
        edge = tibia_angle - slope_perpendicular

        # Normalize to -90 to +90
        while edge > 90:
            edge -= 180
        while edge < -90:
            edge += 180

        return edge

    def inclination_angle(self, base: Tuple[float, float], top: Tuple[float, float]) -> float:
        """
        Calculate body inclination (lean) relative to slope perpendicular.

        Args:
            base: Lower point (e.g., ankle midpoint or hip)
            top: Upper point (e.g., shoulder midpoint)

        Returns:
            Inclination in degrees. 0 = perpendicular to slope, positive = leaning downhill.
        """
        # Body line direction in image
        dx = top[0] - base[0]
        dy = top[1] - base[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        # Body angle from vertical in image (0 = straight up)
        # Note: in image coords, Y increases downward, so we flip
        body_angle = math.degrees(math.atan2(dx, -dy))

        # "Perpendicular to slope" in image is at fall_line_angle from vertical
        slope_perpendicular = self.fall_line_angle_deg

        # Inclination = body deviation from slope perpendicular
        inclination = body_angle - slope_perpendicular

        # Normalize to -90 to +90
        while inclination > 90:
            inclination -= 180
        while inclination < -90:
            inclination += 180

        return inclination

    def summary(self) -> str:
        """Return human-readable summary of slope geometry."""
        lines = [
            "Slope Geometry:",
            f"  Fall line direction: {self.fall_line_angle_deg:+.1f} deg from vertical",
            f"  Pitch (steepness): {self.pitch_deg:.1f} deg",
            f"  Roll (cross-slope): {self.roll_deg:.1f} deg",
        ]
        if self.slope_normal is not None:
            lines.append(f"  Slope normal: [{self.slope_normal[0]:.3f}, {self.slope_normal[1]:.3f}, {self.slope_normal[2]:.3f}]")
        if self.fall_line_3d is not None:
            lines.append(f"  Fall line 3D: [{self.fall_line_3d[0]:.3f}, {self.fall_line_3d[1]:.3f}, {self.fall_line_3d[2]:.3f}]")
        return "\n".join(lines)


try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


# COCO keypoint indices
class Keypoints:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    FACE_KEYPOINTS = {NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR}


# Skeleton connections for drawing
SKELETON = [
    (Keypoints.LEFT_SHOULDER, Keypoints.RIGHT_SHOULDER),
    (Keypoints.LEFT_SHOULDER, Keypoints.LEFT_ELBOW),
    (Keypoints.LEFT_ELBOW, Keypoints.LEFT_WRIST),
    (Keypoints.RIGHT_SHOULDER, Keypoints.RIGHT_ELBOW),
    (Keypoints.RIGHT_ELBOW, Keypoints.RIGHT_WRIST),
    (Keypoints.LEFT_SHOULDER, Keypoints.LEFT_HIP),
    (Keypoints.RIGHT_SHOULDER, Keypoints.RIGHT_HIP),
    (Keypoints.LEFT_HIP, Keypoints.RIGHT_HIP),
    (Keypoints.LEFT_HIP, Keypoints.LEFT_KNEE),
    (Keypoints.LEFT_KNEE, Keypoints.LEFT_ANKLE),
    (Keypoints.RIGHT_HIP, Keypoints.RIGHT_KNEE),
    (Keypoints.RIGHT_KNEE, Keypoints.RIGHT_ANKLE),
]


@dataclass
class PoseMetrics:
    """Biomechanics metrics for a single frame."""
    shoulder_angle_to_slope: float
    hip_angle_to_slope: float
    shoulder_alignment_pct: float  # 100% = parallel to fall line
    hip_alignment_pct: float  # 100% = parallel to fall line
    body_angulation: float
    body_inclination: float  # Torso angle to slope
    knee_angle_left: float
    knee_angle_right: float
    edge_angle_left: float  # Ski edge angle (tibia perpendicular to slope)
    edge_angle_right: float
    edge_symmetry_pct: float  # 100% = perfect symmetry
    fore_aft_left: Optional[float]  # Only in side view
    fore_aft_right: Optional[float]
    confidence: float


def angle_from_horizontal(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle of line from p1 to p2 relative to horizontal (degrees)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-6:
        return 90.0 if dy > 0 else -90.0
    return math.degrees(math.atan2(dy, dx))


def angle_from_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle of line from p1 to p2 relative to vertical (degrees)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dx, dy))


def angle_at_joint(p1: Tuple, p2: Tuple, p3: Tuple) -> float:
    """Calculate angle at p2 formed by p1-p2-p3 (degrees)."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 < 1e-6 or mag2 < 1e-6:
        return 180.0

    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def midpoint(p1: Tuple, p2: Tuple) -> Tuple[float, float]:
    """Calculate midpoint between two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


class MetricsHistory:
    """Stores history of metrics for graphing."""

    def __init__(self, max_frames: int = 150):
        self.max_frames = max_frames
        self.shoulder: Deque[float] = deque(maxlen=max_frames)
        self.hip: Deque[float] = deque(maxlen=max_frames)
        self.angulation: Deque[float] = deque(maxlen=max_frames)
        self.inclination: Deque[float] = deque(maxlen=max_frames)
        self.edge_left: Deque[float] = deque(maxlen=max_frames)
        self.edge_right: Deque[float] = deque(maxlen=max_frames)
        self.edge_symmetry: Deque[float] = deque(maxlen=max_frames)
        self.fore_aft_left: Deque[float] = deque(maxlen=max_frames)
        self.fore_aft_right: Deque[float] = deque(maxlen=max_frames)

    def add(self, m: PoseMetrics):
        self.shoulder.append(m.shoulder_angle_to_slope)
        self.hip.append(m.hip_angle_to_slope)
        self.angulation.append(m.body_angulation)
        self.inclination.append(m.body_inclination)
        self.edge_left.append(m.edge_angle_left)
        self.edge_right.append(m.edge_angle_right)
        self.edge_symmetry.append(m.edge_symmetry_pct)
        # Only track fore/aft if available (side view)
        if m.fore_aft_left is not None:
            self.fore_aft_left.append(m.fore_aft_left)
        if m.fore_aft_right is not None:
            self.fore_aft_right.append(m.fore_aft_right)


class YOLOPoseAnalyzer:
    """Analyzes ski racing video clips using YOLOv8-pose."""

    def __init__(self, slope_geometry: SlopeGeometry = None,
                 slope_angle_deg: float = None, pitch_deg: float = None,
                 calibration_path: str = None,
                 model_size: str = 'l', ski_model_path: str = None):
        """
        Initialize the pose analyzer.

        Args:
            slope_geometry: Full 3D slope geometry (preferred)
            slope_angle_deg: Legacy - fall line angle in degrees (if no geometry)
            pitch_deg: Legacy - slope steepness in degrees (if no geometry)
            calibration_path: Path to calibration JSON to load geometry from
            model_size: YOLO model size (n/s/m/l/x)
            ski_model_path: Path to ski detector model
        """
        if not HAS_YOLO:
            raise ImportError("ultralytics required. Install with: pip install ultralytics")

        # Load slope geometry from calibration file if provided
        if calibration_path and Path(calibration_path).exists():
            print(f"Loading slope geometry from {calibration_path}...")
            self.slope_geometry = SlopeGeometry.from_calibration_file(calibration_path)
            print(self.slope_geometry.summary())
        elif slope_geometry is not None:
            self.slope_geometry = slope_geometry
        else:
            # Create geometry from legacy parameters
            self.slope_geometry = SlopeGeometry(
                fall_line_angle_deg=slope_angle_deg or 0.0,
                pitch_deg=pitch_deg or 25.0
            )
            self.slope_geometry._compute_3d_vectors()

        # Legacy attributes for backward compatibility
        self.slope_angle_deg = self.slope_geometry.fall_line_angle_deg
        self.pitch_deg = self.slope_geometry.pitch_deg

        self.history = MetricsHistory()
        self.ski_detector = None

        model_name = f'yolov8{model_size}-pose.pt'
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        print("Model loaded.")

        # Load custom ski detector if available
        if ski_model_path is None:
            # Look for ski detector model
            for path in ['ski_detector.pt', 'edge/ski_detector.pt',
                        'runs/detect/ski_detector/weights/best.pt']:
                if Path(path).exists():
                    ski_model_path = path
                    break

        if ski_model_path and Path(ski_model_path).exists():
            print(f"Loading ski detector from {ski_model_path}...")
            self.ski_detector = YOLO(ski_model_path)
            print("Ski detector loaded.")
        else:
            print("Ski detector not found. Run train_ski_detector.py to train one.")

    def get_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[Tuple[float, float, float]]:
        if keypoints is None or idx >= len(keypoints):
            return None
        x, y, conf = keypoints[idx]
        if conf < 0.3:
            return None
        return (float(x), float(y), float(conf))

    def compute_metrics(self, keypoints: np.ndarray,
                        left_ski: Optional[dict] = None,
                        right_ski: Optional[dict] = None) -> Optional[PoseMetrics]:
        """
        Compute biomechanics metrics using full 3D slope geometry.

        All angles are calculated relative to the true slope plane,
        not just the 2D image projection.
        """
        left_shoulder = self.get_keypoint(keypoints, Keypoints.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(keypoints, Keypoints.RIGHT_SHOULDER)
        left_hip = self.get_keypoint(keypoints, Keypoints.LEFT_HIP)
        right_hip = self.get_keypoint(keypoints, Keypoints.RIGHT_HIP)
        left_knee = self.get_keypoint(keypoints, Keypoints.LEFT_KNEE)
        right_knee = self.get_keypoint(keypoints, Keypoints.RIGHT_KNEE)
        left_ankle = self.get_keypoint(keypoints, Keypoints.LEFT_ANKLE)
        right_ankle = self.get_keypoint(keypoints, Keypoints.RIGHT_ANKLE)

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        confs = [p[2] for p in [left_shoulder, right_shoulder, left_hip, right_hip] if p]
        if left_knee: confs.append(left_knee[2])
        if right_knee: confs.append(right_knee[2])
        if left_ankle: confs.append(left_ankle[2])
        if right_ankle: confs.append(right_ankle[2])
        avg_conf = sum(confs) / len(confs) if confs else 0

        geom = self.slope_geometry

        # Shoulder angle relative to slope line (using 3D geometry)
        # 0° = parallel to slope line (across the hill)
        shoulder_to_slope = geom.angle_to_slope_line(left_shoulder[:2], right_shoulder[:2])

        # Hip angle relative to slope line
        hip_to_slope = geom.angle_to_slope_line(left_hip[:2], right_hip[:2])

        # Body angulation (0° = aligned, positive = separation between torso and legs)
        shoulder_mid = midpoint(left_shoulder[:2], right_shoulder[:2])
        hip_mid = midpoint(left_hip[:2], right_hip[:2])
        knee_mid = midpoint(left_knee[:2], right_knee[:2]) if left_knee and right_knee else None

        if knee_mid:
            raw_angulation = angle_at_joint(shoulder_mid, hip_mid, knee_mid)
            body_angulation = 180.0 - raw_angulation  # Convert: 180° raw = 0° aligned
        else:
            body_angulation = 0.0

        # Body inclination - torso lean relative to slope perpendicular
        # Uses 3D geometry to account for slope pitch
        ankle_mid = midpoint(left_ankle[:2], right_ankle[:2]) if left_ankle and right_ankle else hip_mid
        body_inclination = geom.inclination_angle(ankle_mid, shoulder_mid)

        # Knee angles (pure joint angles, independent of slope)
        knee_angle_left = 180.0
        knee_angle_right = 180.0
        if left_hip and left_knee and left_ankle:
            knee_angle_left = angle_at_joint(left_hip[:2], left_knee[:2], left_ankle[:2])
        if right_hip and right_knee and right_ankle:
            knee_angle_right = angle_at_joint(right_hip[:2], right_knee[:2], right_ankle[:2])

        # Edge angles using 3D slope geometry
        # How much the ski is tilted from the slope surface
        edge_angle_left = 0.0
        edge_angle_right = 0.0

        if left_knee and left_ankle:
            edge_angle_left = geom.edge_angle_from_tibia(left_knee[:2], left_ankle[:2])

        if right_knee and right_ankle:
            edge_angle_right = geom.edge_angle_from_tibia(right_knee[:2], right_ankle[:2])

        # Edge symmetry as percentage (100% = perfect)
        max_edge = max(abs(edge_angle_left), abs(edge_angle_right), 1)
        edge_diff = abs(edge_angle_left - edge_angle_right)
        edge_symmetry_pct = max(0, 100 - (edge_diff / max_edge * 100))

        # Fore/aft balance - tibia angle vs ski long axis
        # Only meaningful in side view (when ski box is elongated horizontally)
        fore_aft_left = None  # None means not visible/calculable
        fore_aft_right = None

        if left_ski and left_ski.get('is_side_view') and left_knee and left_ankle:
            # Tibia angle from vertical
            tibia_angle = angle_from_vertical(left_knee[:2], left_ankle[:2])
            # Ski direction angle (from bounding box)
            ski_angle = left_ski.get('direction_angle', 0)
            # Fore/aft is tibia angle relative to perpendicular to ski
            fore_aft_left = tibia_angle - (ski_angle + 90)
            while fore_aft_left > 90: fore_aft_left -= 180
            while fore_aft_left < -90: fore_aft_left += 180

        if right_ski and right_ski.get('is_side_view') and right_knee and right_ankle:
            tibia_angle = angle_from_vertical(right_knee[:2], right_ankle[:2])
            ski_angle = right_ski.get('direction_angle', 0)
            fore_aft_right = tibia_angle - (ski_angle + 90)
            while fore_aft_right > 90: fore_aft_right -= 180
            while fore_aft_right < -90: fore_aft_right += 180

        # Shoulder/Hip alignment as percentage (100% = parallel to fall line)
        # At 0° angle to slope = 100%, at 45° = 50%, at 90° = 0%
        shoulder_alignment_pct = max(0, 100 - abs(shoulder_to_slope) * 100 / 45)
        hip_alignment_pct = max(0, 100 - abs(hip_to_slope) * 100 / 45)

        return PoseMetrics(
            shoulder_angle_to_slope=round(shoulder_to_slope, 1),
            hip_angle_to_slope=round(hip_to_slope, 1),
            shoulder_alignment_pct=round(shoulder_alignment_pct, 0),
            hip_alignment_pct=round(hip_alignment_pct, 0),
            body_angulation=round(body_angulation, 1),
            body_inclination=round(body_inclination, 1),
            knee_angle_left=round(knee_angle_left, 1),
            knee_angle_right=round(knee_angle_right, 1),
            edge_angle_left=round(edge_angle_left, 1),
            edge_angle_right=round(edge_angle_right, 1),
            edge_symmetry_pct=round(edge_symmetry_pct, 0),
            fore_aft_left=round(fore_aft_left, 1) if fore_aft_left is not None else None,
            fore_aft_right=round(fore_aft_right, 1) if fore_aft_right is not None else None,
            confidence=round(avg_conf, 3)
        )

    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[PoseMetrics], Optional[dict], Optional[dict]]:
        results = self.model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None, None, None, None

        keypoints = results[0].keypoints.data
        if len(keypoints) == 0:
            return None, None, None, None

        person_kpts = keypoints[0].cpu().numpy()

        # Detect skis
        left_ski, right_ski = self._detect_skis(frame, person_kpts)

        # Compute metrics using ski detection data
        metrics = self.compute_metrics(person_kpts, left_ski, right_ski)

        if metrics:
            self.history.add(metrics)

        return person_kpts, metrics, left_ski, right_ski

    def _find_ski_base_line(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find the ski base line by detecting dark pixels within the bounding box.
        Returns ((x1,y1), (x2,y2)) endpoints of the ski base line, or None if not found."""
        # Ensure bounds are within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Convert to grayscale and find dark pixels (ski base is black)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold to find dark pixels (ski base)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Find contours of dark regions
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest elongated contour (the ski)
        best_contour = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area and area > 100:  # Minimum area threshold
                best_area = area
                best_contour = cnt

        if best_contour is None or len(best_contour) < 5:
            return None

        # Fit a line to the contour using PCA or fitLine
        line = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(line[0][0]), float(line[1][0])
        cx_local, cy_local = float(line[2][0]), float(line[3][0])

        # Calculate line length based on contour extent
        rect = cv2.minAreaRect(best_contour)
        length = max(rect[1]) / 2  # Half the longer dimension

        # Calculate endpoints
        pt1 = (int(x1 + cx_local - vx * length), int(y1 + cy_local - vy * length))
        pt2 = (int(x1 + cx_local + vx * length), int(y1 + cy_local + vy * length))

        return (pt1, pt2)

    def _detect_skis(self, frame: np.ndarray, keypoints: np.ndarray) -> Tuple[Optional[dict], Optional[dict]]:
        """Detect skis and match to ankles. Returns (left_ski, right_ski) dicts with box, angle, fore_aft."""
        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (float(kp[0]), float(kp[1]))
            return None

        la = get_pt(Keypoints.LEFT_ANKLE)
        ra = get_pt(Keypoints.RIGHT_ANKLE)

        left_ski = None
        right_ski = None

        if self.ski_detector is None:
            return None, None

        results = self.ski_detector(frame, verbose=False, conf=0.3)

        if not results or len(results) == 0 or results[0].boxes is None:
            return None, None

        boxes = results[0].boxes

        # Get all detections
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # Find actual ski base direction by analyzing dark pixels in the box
            ski_line = self._find_ski_base_line(frame, int(x1), int(y1), int(x2), int(y2))

            if ski_line is not None:
                # Use detected ski line
                (lx1, ly1), (lx2, ly2) = ski_line
                direction_angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))
            else:
                # Fallback to bounding box orientation
                if width > height:
                    direction_angle = math.degrees(math.atan2(height, width))
                else:
                    direction_angle = 90 - math.degrees(math.atan2(width, height))

            # Side view if ski box is elongated (width > 1.5 * height)
            is_side_view = width > 1.5 * height

            detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'center': (cx, cy),
                'direction_angle': direction_angle,
                'width': width,
                'height': height,
                'is_side_view': is_side_view,
                'ski_line': ski_line  # Actual ski base line endpoints
            })

        # Match skis to ankles - find closest ski to each ankle
        if la and detections:
            best_dist = float('inf')
            for det in detections:
                dist = math.sqrt((det['center'][0] - la[0])**2 + (det['center'][1] - la[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    left_ski = det.copy()

        if ra and detections:
            best_dist = float('inf')
            for det in detections:
                if left_ski and det['box'] == left_ski['box']:
                    continue
                dist = math.sqrt((det['center'][0] - ra[0])**2 + (det['center'][1] - ra[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    right_ski = det.copy()

        return left_ski, right_ski

    def _draw_ski_rectangles(self, frame: np.ndarray, keypoints: np.ndarray, scale: float,
                              left_ski: Optional[dict] = None, right_ski: Optional[dict] = None) -> np.ndarray:
        """Draw ski detections with ski base lines."""
        output = frame.copy()
        thickness = max(2, int(3 * scale))

        # Draw left ski
        if left_ski:
            x1, y1, x2, y2 = left_ski['box']
            # Draw bounding box (orange, thin)
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            # Draw ski base line (black, thick)
            if left_ski.get('ski_line'):
                pt1, pt2 = left_ski['ski_line']
                cv2.line(output, pt1, pt2, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # White outline for visibility
                cv2.line(output, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)

        # Draw right ski
        if right_ski:
            x1, y1, x2, y2 = right_ski['box']
            # Draw bounding box (blue, thin)
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 150, 50), 2)
            # Draw ski base line (black, thick)
            if right_ski.get('ski_line'):
                pt1, pt2 = right_ski['ski_line']
                cv2.line(output, pt1, pt2, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # White outline for visibility
                cv2.line(output, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)

        return output

    def _draw_slope_line(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw slope line (dotted) perpendicular to fall line - the snow surface reference."""
        output = frame.copy()

        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (int(kp[0]), int(kp[1]))
            return None

        la = get_pt(Keypoints.LEFT_ANKLE)
        ra = get_pt(Keypoints.RIGHT_ANKLE)

        if la and ra:
            # Start point: below ankle midpoint (on snow)
            ankle_mid = ((la[0] + ra[0])//2, (la[1] + ra[1])//2)
            snow_offset = int(40 * scale)  # Below ankle
            center_pt = (ankle_mid[0], ankle_mid[1] + snow_offset)

            # Slope line is perpendicular to fall line (horizontal on the slope surface)
            # Fall line angle is slope_angle_deg from vertical, so slope line is at slope_angle_deg from horizontal
            slope_line_angle_rad = math.radians(self.slope_angle_deg)
            line_len = int(200 * scale)

            # Extend in both directions
            dx = int(line_len * math.cos(slope_line_angle_rad))
            dy = int(line_len * math.sin(slope_line_angle_rad))
            pt1 = (center_pt[0] - dx, center_pt[1] - dy)
            pt2 = (center_pt[0] + dx, center_pt[1] + dy)

            # Draw dotted line
            thickness = max(2, int(3 * scale))
            dash_len = int(15 * scale)
            gap_len = int(10 * scale)

            # Calculate line length and draw dashes
            total_dx = pt2[0] - pt1[0]
            total_dy = pt2[1] - pt1[1]
            total_len = math.sqrt(total_dx**2 + total_dy**2)
            if total_len > 0:
                ux, uy = total_dx / total_len, total_dy / total_len
                pos = 0
                drawing = True
                while pos < total_len:
                    if drawing:
                        seg_len = min(dash_len, total_len - pos)
                        x1 = int(pt1[0] + ux * pos)
                        y1 = int(pt1[1] + uy * pos)
                        x2 = int(pt1[0] + ux * (pos + seg_len))
                        y2 = int(pt1[1] + uy * (pos + seg_len))
                        cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), thickness, cv2.LINE_AA)
                        pos += dash_len
                    else:
                        pos += gap_len
                    drawing = not drawing

            # Label "SLOPE" at right end
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "SLOPE"
            label_pos = (pt2[0] + int(10 * scale), pt2[1])
            cv2.putText(output, label, label_pos, font, 0.5 * scale, (255, 255, 255),
                       max(1, int(2 * scale)), cv2.LINE_AA)

        return output

    def _draw_fall_line(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw fall line arrow starting from snow level between skis."""
        output = frame.copy()

        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (int(kp[0]), int(kp[1]))
            return None

        la = get_pt(Keypoints.LEFT_ANKLE)
        ra = get_pt(Keypoints.RIGHT_ANKLE)

        if la and ra:
            # Start point: below ankle midpoint (on snow)
            ankle_mid = ((la[0] + ra[0])//2, (la[1] + ra[1])//2)
            snow_offset = int(40 * scale)  # Below ankle
            start_pt = (ankle_mid[0], ankle_mid[1] + snow_offset)

            # Fall line direction (perpendicular to slope line, pointing downhill)
            line_len = int(150 * scale)
            angle_rad = math.radians(90 + self.slope_angle_deg)
            dx = int(line_len * math.cos(angle_rad))
            dy = int(line_len * math.sin(angle_rad))
            end_pt = (start_pt[0] + dx, start_pt[1] + dy)

            # Draw arrow
            cv2.arrowedLine(output, start_pt, end_pt, (0, 165, 255), int(4 * scale),
                           tipLength=0.15, line_type=cv2.LINE_AA)

            # Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "FALL LINE"
            label_pos = (end_pt[0] + int(10 * scale), end_pt[1])
            cv2.putText(output, label, label_pos, font, 0.6 * scale, (0, 165, 255),
                       max(1, int(2 * scale)), cv2.LINE_AA)

        return output

    def _draw_metrics_graph(self, frame: np.ndarray, scale: float, graph_height: int) -> np.ndarray:
        """Draw a graph of metrics over time above the footer."""
        output = frame.copy()
        h, w = frame.shape[:2]

        if len(self.history.edge_symmetry) < 2:
            return output

        # Graph dimensions
        graph_y = h - graph_height - int(50 * scale)  # Above footer
        graph_w = w - int(40 * scale)
        margin = int(20 * scale)

        # Semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (margin, graph_y), (margin + graph_w, graph_y + graph_height),
                     (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Draw grid lines
        for i in range(5):
            y = graph_y + int(graph_height * i / 4)
            cv2.line(output, (margin, y), (margin + graph_w, y), (60, 60, 60), 1)

        def draw_line(data, color, y_min, y_max):
            if len(data) < 2:
                return
            points = []
            for i, val in enumerate(data):
                x = margin + int(i * graph_w / max(len(data) - 1, 1))
                # Normalize to graph height
                norm_val = (val - y_min) / (y_max - y_min) if y_max != y_min else 0.5
                norm_val = max(0, min(1, norm_val))
                y = graph_y + graph_height - int(norm_val * graph_height)
                points.append((x, y))

            for i in range(len(points) - 1):
                cv2.line(output, points[i], points[i+1], color, max(1, int(2 * scale)), cv2.LINE_AA)

        # Draw different metrics
        draw_line(self.history.edge_symmetry, (0, 255, 0), 0, 100)  # Green - symmetry
        draw_line(self.history.inclination, (255, 180, 100), -45, 45)  # Orange - inclination
        draw_line(self.history.angulation, (0, 255, 255), 0, 45)  # Yellow - angulation (0=aligned)

        # Legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 * scale
        leg_y = graph_y + int(15 * scale)
        cv2.putText(output, "Symmetry", (margin + 5, leg_y), font, font_scale, (0, 255, 0), 1)
        cv2.putText(output, "Inclin", (margin + int(80 * scale), leg_y), font, font_scale, (255, 180, 100), 1)
        cv2.putText(output, "Angul", (margin + int(140 * scale), leg_y), font, font_scale, (0, 255, 255), 1)

        return output

    def _draw_extended_lines(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw extended lines through shoulders and hips with slope reference."""
        output = frame.copy()
        line_ext = int(100 * scale)
        thickness = max(2, int(3 * scale))

        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (int(kp[0]), int(kp[1]))
            return None

        def draw_dotted_line(img, pt1, pt2, color, thickness, dash_len=10, gap_len=8):
            """Draw a dotted line between two points."""
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1:
                return
            ux, uy = dx/dist, dy/dist
            pos = 0
            drawing = True
            while pos < dist:
                if drawing:
                    seg_len = min(dash_len, dist - pos)
                    x1 = int(pt1[0] + ux * pos)
                    y1 = int(pt1[1] + uy * pos)
                    x2 = int(pt1[0] + ux * (pos + seg_len))
                    y2 = int(pt1[1] + uy * (pos + seg_len))
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                    pos += dash_len
                else:
                    pos += gap_len
                drawing = not drawing

        ls = get_pt(Keypoints.LEFT_SHOULDER)
        rs = get_pt(Keypoints.RIGHT_SHOULDER)
        lh = get_pt(Keypoints.LEFT_HIP)
        rh = get_pt(Keypoints.RIGHT_HIP)

        # Slope direction unit vector
        slope_rad = math.radians(self.slope_angle_deg)
        slope_ux = math.cos(slope_rad)
        slope_uy = math.sin(slope_rad)

        # Extended shoulder line (magenta) with dotted slope reference
        if ls and rs:
            dx = rs[0] - ls[0]
            dy = rs[1] - ls[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                ext_l = (int(ls[0] - ux * line_ext), int(ls[1] - uy * line_ext))
                ext_r = (int(rs[0] + ux * line_ext), int(rs[1] + uy * line_ext))
                cv2.line(output, ext_l, ext_r, (255, 0, 255), thickness, cv2.LINE_AA)

                # Dotted slope reference line from extended left end
                slope_ref = (int(ext_l[0] + slope_ux * line_ext * 1.5),
                           int(ext_l[1] + slope_uy * line_ext * 1.5))
                draw_dotted_line(output, ext_l, slope_ref, (255, 0, 255), max(1, thickness-1))

        # Extended hip line (cyan) with dotted slope reference
        if lh and rh:
            dx = rh[0] - lh[0]
            dy = rh[1] - lh[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                ext_l = (int(lh[0] - ux * line_ext), int(lh[1] - uy * line_ext))
                ext_r = (int(rh[0] + ux * line_ext), int(rh[1] + uy * line_ext))
                cv2.line(output, ext_l, ext_r, (255, 255, 0), thickness, cv2.LINE_AA)

                # Dotted slope reference line from extended left end
                slope_ref = (int(ext_l[0] + slope_ux * line_ext * 1.5),
                           int(ext_l[1] + slope_uy * line_ext * 1.5))
                draw_dotted_line(output, ext_l, slope_ref, (255, 255, 0), max(1, thickness-1))

        return output

    def draw_overlay(self, frame: np.ndarray, keypoints: np.ndarray,
                     metrics: Optional[PoseMetrics] = None,
                     left_ski: Optional[dict] = None,
                     right_ski: Optional[dict] = None,
                     show_debug_angles: bool = True) -> np.ndarray:
        """Draw pose skeleton and metrics on frame."""
        output = frame.copy()
        h, w = frame.shape[:2]

        scale = max(1.0, h / 1080)
        thickness = max(2, int(2 * scale))
        point_radius = max(3, int(4 * scale))

        # Draw skeleton connections (including shoulder-hip lines)
        for i, j in SKELETON:
            kp1 = keypoints[i]
            kp2 = keypoints[j]
            if kp1[2] > 0.3 and kp2[2] > 0.3:
                pt1 = (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2[0]), int(kp2[1]))
                if i in [Keypoints.LEFT_HIP, Keypoints.RIGHT_HIP, Keypoints.LEFT_KNEE,
                         Keypoints.RIGHT_KNEE, Keypoints.LEFT_ANKLE, Keypoints.RIGHT_ANKLE]:
                    color = (0, 255, 255)  # Yellow for legs
                else:
                    color = (0, 255, 0)  # Green for upper body
                cv2.line(output, pt1, pt2, color, thickness + 1)

        # Draw extended shoulder and hip lines
        output = self._draw_extended_lines(output, keypoints, scale)

        # Draw keypoints (skip face)
        for i, kp in enumerate(keypoints):
            if i in Keypoints.FACE_KEYPOINTS:
                continue
            if kp[2] > 0.3:
                pt = (int(kp[0]), int(kp[1]))
                if kp[2] > 0.7:
                    color = (0, 255, 0)
                elif kp[2] > 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                cv2.circle(output, pt, point_radius, color, -1)
                cv2.circle(output, pt, point_radius + 1, (255, 255, 255), 1)

        # Draw ski rectangles
        output = self._draw_ski_rectangles(output, keypoints, scale, left_ski, right_ski)

        # Draw slope line (dotted, perpendicular to fall line)
        output = self._draw_slope_line(output, keypoints, scale)

        # Draw fall line from snow level
        output = self._draw_fall_line(output, keypoints, scale)

        # Draw metrics graph
        graph_height = int(80 * scale)
        output = self._draw_metrics_graph(output, scale, graph_height)

        # Draw compact metrics bar at BOTTOM
        if metrics:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 * scale
            bar_height = int(40 * scale)

            overlay = output.copy()
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

            y = h - int(12 * scale)
            margin = int(10 * scale)
            text_thickness = max(1, int(2 * scale))

            # Color coding for percentages
            def pct_color(pct):
                if pct >= 80: return (0, 255, 0)  # Green
                elif pct >= 50: return (0, 255, 255)  # Yellow
                else: return (0, 165, 255)  # Orange

            # Compact format with all metrics - angles in degrees (no degree symbol - font issue)
            metrics_parts = [
                (f"Sh:{metrics.shoulder_angle_to_slope:+.0f}", (255, 0, 255)),  # Magenta
                (f"Hip:{metrics.hip_angle_to_slope:+.0f}", (255, 255, 0)),  # Cyan
                (f"Edge L{metrics.edge_angle_left:+.0f} R{metrics.edge_angle_right:+.0f}", (100, 255, 255)),
                (f"Sym:{metrics.edge_symmetry_pct:.0f}%", pct_color(metrics.edge_symmetry_pct)),
                (f"Ang:{metrics.body_angulation:.0f}", (0, 255, 255)),
                (f"Incl:{metrics.body_inclination:+.0f}", (255, 180, 100)),
            ]
            # Only show fore/aft in side view
            if metrics.fore_aft_left is not None or metrics.fore_aft_right is not None:
                fa_l = f"{metrics.fore_aft_left:+.0f}" if metrics.fore_aft_left is not None else "--"
                fa_r = f"{metrics.fore_aft_right:+.0f}" if metrics.fore_aft_right is not None else "--"
                metrics_parts.append((f"F/A L{fa_l} R{fa_r}", (255, 200, 100)))

            x = margin
            for text, color in metrics_parts:
                cv2.putText(output, text, (x, y), font, font_scale, color, text_thickness, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                x += tw + int(15 * scale)

        return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ski video with YOLOv8-pose')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('slope_angle', nargs='?', type=float, default=None,
                        help='Slope angle in degrees (manual override)')
    parser.add_argument('--output-video', '-v', metavar='FILE', help='Save annotated video')
    parser.add_argument('--output-frames', '-o', metavar='DIR', help='Save annotated frames')
    parser.add_argument('--sample-rate', '-s', type=int, default=1,
                        help='Analyze every Nth frame (default: 1)')
    parser.add_argument('--model', '-m', default='l', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (default: l)')
    parser.add_argument('--race-manifest', metavar='FILE',
                        help='Race manifest JSON for GPS-based slope calculation')
    parser.add_argument('--calibration', metavar='FILE',
                        help='Calibration JSON with pole positions')
    parser.add_argument('--camera', default='Cam1', help='Camera ID')
    parser.add_argument('--run', default='run2', choices=['run1', 'run2'],
                        help='Which run (default: run2)')

    args = parser.parse_args()

    print(f"\nVideo: {args.video_path}")

    # Load slope geometry from calibration file (preferred) or use manual angle
    calibration_path = args.calibration
    slope_angle = args.slope_angle

    if calibration_path and Path(calibration_path).exists():
        # Use full 3D slope geometry from calibration
        print(f"Loading 3D slope geometry from: {calibration_path}")
        analyzer = YOLOPoseAnalyzer(
            calibration_path=calibration_path,
            model_size=args.model
        )
    elif slope_angle is not None:
        # Legacy: manual slope angle
        print(f"Using manual slope angle: {slope_angle} deg")
        analyzer = YOLOPoseAnalyzer(
            slope_angle_deg=slope_angle,
            model_size=args.model
        )
    else:
        # No slope info - use defaults
        print("Warning: No calibration or slope angle provided. Using defaults.")
        analyzer = YOLOPoseAnalyzer(model_size=args.model)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f}fps, {total_frames} frames")

    out_video = None
    if args.output_video:
        print(f"Output: {args.output_video}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))

    if args.output_frames:
        Path(args.output_frames).mkdir(parents=True, exist_ok=True)

    all_metrics = []
    frame_num = 0
    poses_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % args.sample_rate == 0:
            result = analyzer.analyze_frame(frame)
            keypoints, metrics, left_ski, right_ski = result

            if keypoints is not None:
                poses_detected += 1
                if metrics:
                    all_metrics.append(metrics)
                annotated = analyzer.draw_overlay(frame, keypoints, metrics, left_ski, right_ski)
                if out_video:
                    out_video.write(annotated)
                if args.output_frames:
                    cv2.imwrite(f"{args.output_frames}/frame_{frame_num:05d}.jpg", annotated)
            else:
                if out_video:
                    out_video.write(frame)

        print(f"\rProgress: {(frame_num + 1) / total_frames * 100:.1f}% | Poses: {poses_detected}", end='', flush=True)
        frame_num += 1

    print(f"\n\nDone! Detected pose in {poses_detected} frames.")

    cap.release()
    if out_video:
        out_video.release()

    if all_metrics:
        print("\nSummary:")
        print(f"  Shoulder: {sum(m.shoulder_angle_to_slope for m in all_metrics)/len(all_metrics):.1f}°")
        print(f"  Hip: {sum(m.hip_angle_to_slope for m in all_metrics)/len(all_metrics):.1f}°")
        print(f"  Angulation: {sum(m.body_angulation for m in all_metrics)/len(all_metrics):.1f}°")
        print(f"  Inclination: {sum(m.body_inclination for m in all_metrics)/len(all_metrics):.1f}°")
        print(f"  Edge Symmetry: {sum(m.edge_symmetry_pct for m in all_metrics)/len(all_metrics):.0f}%")
        # Fore/aft only available in side view
        fa_left = [m.fore_aft_left for m in all_metrics if m.fore_aft_left is not None]
        fa_right = [m.fore_aft_right for m in all_metrics if m.fore_aft_right is not None]
        if fa_left:
            print(f"  Fore/Aft L: {sum(fa_left)/len(fa_left):.1f}°")
        if fa_right:
            print(f"  Fore/Aft R: {sum(fa_right)/len(fa_right):.1f}°")


if __name__ == "__main__":
    main()

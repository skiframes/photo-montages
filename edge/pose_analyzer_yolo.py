#!/usr/bin/env python3
"""
Pose analysis module using YOLOv8-pose for ski racing video clips.

YOLOv8-pose is more robust than MediaPipe for:
- Fast motion and blur
- Unusual body positions (skiing stance)
- Distinguishing person from ski poles
- Detection from distance

Optional SAM3 integration for precise ski segmentation:
- Uses text prompt "ski" to segment skis precisely
- Fits line to segmentation mask for accurate ski orientation
- Falls back to YOLO bounding box detection if SAM3 unavailable

Keypoints (COCO format):
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import math
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Deque, Dict
from collections import deque

import cv2
import numpy as np

# Import add_logos from montage for consistent logo overlay
try:
    from montage import add_logos
except ImportError:
    add_logos = None

# SAM/SAM2/SAM3 support (optional) - Segment Anything Model for precise ski segmentation
HAS_SAM = False
SAMModel = None
SAM3SemanticPredictor = None
try:
    from ultralytics import SAM as _SAM
    SAMModel = _SAM
    HAS_SAM = True
    # SAM3 text prompts require SAM3SemanticPredictor
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor as _SAM3Predictor
        SAM3SemanticPredictor = _SAM3Predictor
    except ImportError:
        pass
except ImportError:
    try:
        # Try alternative import path
        from ultralytics.models.sam import SAM as _SAM
        SAMModel = _SAM
        HAS_SAM = True
    except ImportError:
        pass

# Depth Anything v2 for monocular depth estimation (for dynamic slope detection)
HAS_DEPTH = False
DepthAnythingModel = None
DepthAnythingProcessor = None
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_DEPTH = True
except ImportError:
    pass


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
    def from_detected_skis(cls, left_ski: Optional[dict], right_ski: Optional[dict]) -> Optional['SlopeGeometry']:
        """
        Compute local slope geometry from detected ski orientations.

        The two skis on snow define the slope plane:
        - Average ski direction = fall line direction
        - Line connecting ski centers = cross-slope direction
        - Slope normal = cross product of these vectors

        Returns None if insufficient ski data.
        """
        # Need at least one ski with a line
        skis_with_lines = []
        if left_ski and left_ski.get('ski_line'):
            skis_with_lines.append(left_ski)
        if right_ski and right_ski.get('ski_line'):
            skis_with_lines.append(right_ski)

        if not skis_with_lines:
            return None

        geom = cls()

        # Compute average ski direction (approximates fall line in image)
        directions = []
        centers = []
        for ski in skis_with_lines:
            pt1, pt2 = ski['ski_line']
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                # Normalize and ensure pointing "downhill" (positive Y)
                dir_x, dir_y = dx / length, dy / length
                if dir_y < 0:
                    dir_x, dir_y = -dir_x, -dir_y
                directions.append((dir_x, dir_y))

            if ski.get('center'):
                centers.append(ski['center'])

        if not directions:
            return None

        # Average direction = fall line direction
        avg_dx = sum(d[0] for d in directions) / len(directions)
        avg_dy = sum(d[1] for d in directions) / len(directions)

        # Fall line angle from vertical (0 = straight down)
        geom.fall_line_angle_deg = math.degrees(math.atan2(avg_dx, avg_dy))

        # Estimate pitch from ski foreshortening (rough heuristic)
        # If skis appear shorter than expected, we're viewing at steeper angle
        # Default to typical race slope pitch
        geom.pitch_deg = 28.0

        # Compute cross-slope direction if we have two ski centers
        if len(centers) >= 2:
            cross_dx = centers[1][0] - centers[0][0]
            cross_dy = centers[1][1] - centers[0][1]
            cross_len = math.sqrt(cross_dx*cross_dx + cross_dy*cross_dy)
            if cross_len > 0:
                # Roll angle from cross-slope tilt
                geom.roll_deg = math.degrees(math.atan2(cross_dy, cross_dx))

        geom._compute_3d_vectors()

        return geom

    @classmethod
    def from_depth_map(cls, depth_map: np.ndarray, snow_mask: np.ndarray,
                       skier_position: Tuple[float, float], local_radius: int = 150) -> Optional['SlopeGeometry']:
        """
        Compute local slope geometry from depth map and snow mask.

        Uses monocular depth estimation to get 3D terrain shape, then fits
        a plane to the snow surface near the skier's position.

        Args:
            depth_map: HxW array of depth values (higher = farther)
            snow_mask: HxW binary mask of snow surface
            skier_position: (x, y) position of skier in image
            local_radius: Radius around skier to sample for local slope

        Returns:
            SlopeGeometry with fall_line, pitch, roll from fitted plane
        """
        if depth_map is None or snow_mask is None:
            return None

        h, w = depth_map.shape[:2]
        cx, cy = int(skier_position[0]), int(skier_position[1])

        # Define local region around skier (below them, where slope is)
        # Sample points below the skier's feet
        y_start = max(0, cy)
        y_end = min(h, cy + local_radius * 2)
        x_start = max(0, cx - local_radius)
        x_end = min(w, cx + local_radius)

        if y_end <= y_start or x_end <= x_start:
            return None

        # Get depth values within snow mask in local region
        local_snow = snow_mask[y_start:y_end, x_start:x_end]
        local_depth = depth_map[y_start:y_end, x_start:x_end]

        # Sample points where snow is detected
        snow_points = np.where(local_snow > 0)
        if len(snow_points[0]) < 100:
            return None

        # Create 3D point cloud: (x, y, depth)
        # x, y are image coordinates, depth from depth estimation
        points_y = snow_points[0] + y_start  # Image Y
        points_x = snow_points[1] + x_start  # Image X
        points_z = local_depth[snow_points]  # Depth

        # Subsample for efficiency (max 1000 points)
        if len(points_x) > 1000:
            indices = np.random.choice(len(points_x), 1000, replace=False)
            points_x = points_x[indices]
            points_y = points_y[indices]
            points_z = points_z[indices]

        # Fit a plane to the 3D points: ax + by + c = z
        # Using least squares: [x, y, 1] @ [a, b, c].T = z
        A = np.column_stack([points_x, points_y, np.ones(len(points_x))])
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, points_z, rcond=None)
        except np.linalg.LinAlgError:
            return None

        a, b, c = coeffs

        # Plane normal vector: (-a, -b, 1) normalized
        normal = np.array([-a, -b, 1.0])
        normal = normal / np.linalg.norm(normal)

        # Fall line direction: steepest descent on the plane
        # Project gravity (0, 1, 0) onto plane and normalize
        # Fall line in image = direction where depth increases fastest
        fall_line_2d = np.array([a, b])
        fall_line_len = np.linalg.norm(fall_line_2d)

        if fall_line_len < 0.001:
            # Nearly horizontal slope
            fall_line_angle = 0.0
            pitch = 0.0
        else:
            fall_line_2d = fall_line_2d / fall_line_len
            # Angle from vertical (positive Y is down in image)
            fall_line_angle = math.degrees(math.atan2(fall_line_2d[0], fall_line_2d[1]))
            # Pitch from the slope of the plane
            pitch = math.degrees(math.atan(fall_line_len))

        geom = cls()
        geom.fall_line_angle_deg = fall_line_angle
        geom.pitch_deg = pitch
        geom.slope_normal = normal

        # Cross-slope direction (perpendicular to fall line in image plane)
        cross_slope_2d = np.array([-fall_line_2d[1], fall_line_2d[0]]) if fall_line_len > 0.001 else np.array([1, 0])
        geom.roll_deg = math.degrees(math.atan2(cross_slope_2d[1], cross_slope_2d[0]))

        geom._compute_3d_vectors()

        return geom

    def edge_angle_from_ski_base(self, ski: dict) -> float:
        """
        Calculate ski edge angle from ski base orientation relative to calibrated slope.

        Uses SAM3 rotated rectangle to get ski base orientation, then compares
        to the slope plane defined by calibration (gate_line_axis = fall line).

        Edge angle = how much ski base tilts away from the slope plane.
        0° = ski flat on slope (base parallel to slope)
        90° = ski on full edge (base perpendicular to slope)

        Args:
            ski: Ski detection dict with 'rotated_rect' from SAM3 segmentation

        Returns:
            Edge angle in degrees
        """
        if not ski:
            return 0.0

        rotated_rect = ski.get('rotated_rect')
        ski_line = ski.get('ski_line')

        if rotated_rect is None or ski_line is None:
            return 0.0

        # SAM3 rotated rectangle has 4 corners
        # Find the short edge (ski base width direction)
        pts = rotated_rect

        edge1_len = math.sqrt((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2)
        edge2_len = math.sqrt((pts[2][0]-pts[1][0])**2 + (pts[2][1]-pts[1][1])**2)

        # Short edge = ski base width direction (perpendicular to ski length)
        if edge1_len < edge2_len:
            base_dx = pts[1][0] - pts[0][0]
            base_dy = pts[1][1] - pts[0][1]
        else:
            base_dx = pts[2][0] - pts[1][0]
            base_dy = pts[2][1] - pts[1][1]

        # Ski base width angle from horizontal
        base_width_angle = math.degrees(math.atan2(base_dy, base_dx))

        # Cross-slope direction from calibration (perpendicular to fall line)
        # fall_line_angle_deg is the fall line direction from vertical
        # Cross-slope is perpendicular = fall_line_angle_deg from horizontal
        cross_slope_angle = self.fall_line_angle_deg

        # Edge angle = difference between ski base orientation and cross-slope
        edge_angle = abs(base_width_angle - cross_slope_angle)

        # Normalize to 0-90 range
        while edge_angle > 90:
            edge_angle = 180 - edge_angle

        return edge_angle

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

# Note: segment_anything (original Meta SAM) is NOT used.
# We use ultralytics SAM (SAM2/SAM3) instead - see HAS_SAM set at top of file.


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


# Skeleton connections for drawing (no arms/forearms, no shoulder-to-hip lines)
SKELETON = [
    (Keypoints.LEFT_SHOULDER, Keypoints.RIGHT_SHOULDER),
    # Removed shoulder-to-hip lines per user request - keep only leg and hip connections
    (Keypoints.LEFT_HIP, Keypoints.RIGHT_HIP),
    (Keypoints.LEFT_HIP, Keypoints.LEFT_KNEE),
    (Keypoints.LEFT_KNEE, Keypoints.LEFT_ANKLE),
    (Keypoints.RIGHT_HIP, Keypoints.RIGHT_KNEE),
    (Keypoints.RIGHT_KNEE, Keypoints.RIGHT_ANKLE),
]

# Keypoints to skip when drawing (face + arms)
ARM_KEYPOINTS = {Keypoints.LEFT_ELBOW, Keypoints.RIGHT_ELBOW,
                 Keypoints.LEFT_WRIST, Keypoints.RIGHT_WRIST}


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
    # Belly button estimated position (pixel coords)
    belly_button: Optional[Tuple[float, float]] = None
    # Center of mass position (pixel coords) - can be outside body during carving
    center_of_mass: Optional[Tuple[float, float]] = None
    # Whether CoM is inside or outside body silhouette
    com_inside_body: bool = True


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
        self.com_inside: Deque[bool] = deque(maxlen=max_frames)

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
        self.com_inside.append(m.com_inside_body)


class YOLOPoseAnalyzer:
    """Analyzes ski racing video clips using YOLOv8-pose and optional SAM3."""

    def __init__(self, slope_geometry: SlopeGeometry = None,
                 slope_angle_deg: float = None, pitch_deg: float = None,
                 calibration_path: str = None,
                 model_size: str = 'l', ski_model_path: str = None,
                 use_sam3: bool = True, sam3_model_path: str = None):
        """
        Initialize the pose analyzer.

        Args:
            slope_geometry: Full 3D slope geometry (preferred)
            slope_angle_deg: Legacy - fall line angle in degrees (if no geometry)
            pitch_deg: Legacy - slope steepness in degrees (if no geometry)
            calibration_path: Path to calibration JSON to load geometry from
            model_size: YOLO model size (n/s/m/l/x)
            ski_model_path: Path to ski detector model (YOLO-based)
            use_sam3: Whether to use SAM3 for precise ski segmentation
            sam3_model_path: Path to SAM3 model (default: sam3.pt)
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

        # Auto-detect CUDA device
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_size = model_size  # Store for display

        model_name = f'yolov8{model_size}-pose.pt'
        # Look for model in local directories first (avoid network download)
        model_path = model_name
        for path in [model_name, f'edge/{model_name}', f'~/photo-montages/edge/{model_name}',
                     f'{Path.home()}/{model_name}', f'{Path.home()}/photo-montages/edge/{model_name}']:
            expanded = Path(path).expanduser()
            if expanded.exists():
                model_path = str(expanded)
                break
        print(f"Loading {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"Model loaded on {self.device}.")

        # Load custom ski detector if available
        if ski_model_path is None:
            # Look for ski detector model
            for path in ['ski_detector.pt', 'edge/ski_detector.pt',
                        'runs/detect/ski_detector/weights/best.pt']:
                if Path(path).exists():
                    ski_model_path = path
                    break

        if ski_model_path and Path(ski_model_path).exists():
            print(f"Loading ski detector from {ski_model_path} on {self.device}...")
            self.ski_detector = YOLO(ski_model_path)
            self.ski_detector.to(self.device)
            print("Ski detector loaded.")
        else:
            print("Ski detector not found. Run train_ski_detector.py to train one.")

        # Initialize SAM for precise ski segmentation (SAM3 preferred, then SAM2)
        self.sam_model = None
        self.sam3_text_predictor = None  # SAM3SemanticPredictor for text prompts
        self.sam_version = None  # 'sam3' or 'sam2'
        self.use_sam = use_sam3 and HAS_SAM

        if use_sam3:
            if not HAS_SAM:
                print("SAM not available. Install with: pip install -U ultralytics")
                print("  SAM3: sam3.pt (text prompts), SAM2: sam2_b.pt")
            else:
                # Find SAM model - SAM3 first (supports text prompts), then SAM2
                sam_model_path = sam3_model_path
                if sam_model_path is None:
                    # Priority: SAM3 > SAM2
                    for path in ['sam3.pt', 'edge/sam3.pt', 'models/sam3.pt',
                                 'sam2_b.pt', 'sam2_l.pt', 'sam2_t.pt',
                                 'edge/sam2_b.pt', 'models/sam2_b.pt']:
                        if Path(path).exists():
                            sam_model_path = path
                            break

                if sam_model_path and Path(sam_model_path).exists():
                    print(f"Loading SAM from {sam_model_path}...")
                    try:
                        self.sam_model = SAMModel(sam_model_path)
                        # Detect version
                        if 'sam3' in sam_model_path.lower():
                            self.sam_version = 'sam3'
                            # Initialize SAM3SemanticPredictor for text prompts
                            if SAM3SemanticPredictor is not None:
                                self.sam3_text_predictor = SAM3SemanticPredictor(overrides=dict(
                                    model=sam_model_path,
                                    task="segment",
                                    mode="predict",
                                    conf=0.25,
                                    verbose=False,
                                ))
                                print("SAM3 loaded - will use text prompts for ski detection.")
                            else:
                                print("SAM3 loaded but SAM3SemanticPredictor not available.")
                                print("  Falling back to bbox prompts.")
                                self.sam_version = 'sam2'  # Fall back to SAM2 API
                        else:
                            self.sam_version = 'sam2'
                            print("SAM2 loaded - will use bbox prompts for ski segmentation.")
                    except Exception as e:
                        print(f"Failed to load SAM: {e}")
                        self.sam_model = None
                else:
                    print("SAM model not found. Download:")
                    print("  SAM3: huggingface-cli download facebook/sam3 sam3.pt")
                    print("  SAM2: yolo download model=sam2_b.pt")
                    print("  Falling back to YOLO-based ski detection.")

        # Track current frame for SAM (to avoid re-setting image)
        self._current_frame_id = None
        self._frame_counter = 0  # Auto-incrementing frame ID
        # Current local slope geometry (computed per-frame from ski detection)
        self._current_local_slope = None
        # Compatibility aliases
        self.sam3_predictor = self.sam_model  # Legacy alias

        # Depth estimation for dynamic slope detection
        self.depth_model = None
        self.depth_processor = None
        self._cached_depth_map = None
        self._cached_snow_mask = None

        if HAS_DEPTH:
            try:
                print("Loading Depth Anything v2 for slope estimation...")
                # Use small model for speed, can use larger for accuracy
                model_id = "depth-anything/Depth-Anything-V2-Small-hf"
                self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
                self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
                self.depth_model.to(self.device)
                self.depth_model.eval()
                print(f"Depth Anything v2 loaded on {self.device}.")
            except Exception as e:
                print(f"Failed to load depth model: {e}")
                print("Falling back to ski-direction slope estimation.")

        # Gate info for display (set via set_gate_info())
        self.current_gate = None

        # Logo overlay (set via set_logos())
        self.selected_logos = None

        # Graph display settings
        self.show_graph_curves = False  # Default: don't show curves (toggle via Graph button)

        # Video timestamp for overlay (set via set_video_timestamp())
        self.video_timestamp = None
        self.video_source_name = None

    def set_video_timestamp(self, timestamp: datetime, source_name: str = None):
        """Set video timestamp and source name for overlay display."""
        self.video_timestamp = timestamp
        self.video_source_name = source_name

    def set_logos(self, logos: List[str]):
        """Set logos to overlay at bottom-left corner."""
        self.selected_logos = logos if logos else None

    def set_gate_info(self, gate_id: int = None, color: str = None,
                      prev_gate_id: int = None, prev_gate_color: str = None,
                      dist_from_prev: float = None, offset_lr: float = None,
                      drop: float = None, gps_accuracy: float = None):
        """
        Set current gate info for display in top-right corner.

        Args:
            gate_id: Gate number (e.g., 9)
            color: Gate color ("red" or "blue")
            prev_gate_id: Previous gate number (e.g., 8)
            prev_gate_color: Previous gate color ("red" or "blue")
            dist_from_prev: Distance from previous gate (meters)
            offset_lr: Left/right offset (negative=left, positive=right, meters)
            drop: Vertical drop from previous gate (meters)
            gps_accuracy: GPS measurement accuracy (meters)
        """
        self.current_gate = {
            'id': gate_id,
            'color': color,
            'prev_id': prev_gate_id,
            'prev_color': prev_gate_color,
            'dist_from_prev': dist_from_prev,
            'offset_lr': offset_lr,
            'drop': drop,
            'gps_accuracy': gps_accuracy,
        }

    def get_keypoint(self, keypoints: np.ndarray, idx: int) -> Optional[Tuple[float, float, float]]:
        if keypoints is None or idx >= len(keypoints):
            return None
        x, y, conf = keypoints[idx]
        if conf < 0.3:
            return None
        return (float(x), float(y), float(conf))

    def _segment_body_sam3(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Use SAM3 text prompt "skier" to segment the skier's body.

        Returns binary mask of the body, or None if unavailable.
        Used to refine center of mass estimation and check if CoM is inside body.
        """
        if self.sam3_text_predictor is None or self.sam_version != 'sam3':
            return None

        try:
            self.sam3_text_predictor.set_image(frame)
            results = self.sam3_text_predictor(text=["skier"])

            if not results or len(results) == 0:
                return None

            # Find the largest person mask
            best_mask = None
            best_area = 0

            for result in results:
                if not hasattr(result, 'masks') or result.masks is None:
                    continue
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    area = mask_binary.sum()
                    if area > best_area:
                        best_area = area
                        best_mask = mask_binary

            return best_mask

        except Exception as e:
            print(f"SAM3 body segmentation error: {e}")
            return None

    def estimate_belly_button(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Estimate belly button (navel) position from keypoints.

        The navel sits roughly 60% of the way from shoulders to hips,
        at the midline of the torso.
        """
        left_hip = self.get_keypoint(keypoints, Keypoints.LEFT_HIP)
        right_hip = self.get_keypoint(keypoints, Keypoints.RIGHT_HIP)
        left_shoulder = self.get_keypoint(keypoints, Keypoints.LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(keypoints, Keypoints.RIGHT_SHOULDER)

        if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
            return None

        shoulder_mid = midpoint(left_shoulder[:2], right_shoulder[:2])
        hip_mid = midpoint(left_hip[:2], right_hip[:2])

        # Belly button is ~60% down from shoulders to hips (closer to hips)
        t = 0.60
        bx = shoulder_mid[0] + t * (hip_mid[0] - shoulder_mid[0])
        by = shoulder_mid[1] + t * (hip_mid[1] - shoulder_mid[1])

        return (bx, by)

    def estimate_center_of_mass(self, keypoints: np.ndarray,
                                body_mask: Optional[np.ndarray] = None) -> Optional[Tuple[Tuple[float, float], bool]]:
        """
        Estimate center of mass using segment mass model.

        Body segment mass percentages (Winter, 2009):
          Head+neck: 8.1%, Torso: 49.7%,
          Upper arm: 2.8% each, Forearm+hand: 2.2% each,
          Thigh: 10.0% each, Shank+foot: 6.1% each

        Each segment's CoM is at its midpoint. The whole-body CoM is
        the mass-weighted average of all segment positions.

        During carving, the body leans inside the turn, shifting CoM
        toward the turn center — potentially outside the body silhouette.

        Returns (com_position, is_inside_body) or None.
        """
        # Gather all available keypoints
        nose = self.get_keypoint(keypoints, Keypoints.NOSE)
        l_shoulder = self.get_keypoint(keypoints, Keypoints.LEFT_SHOULDER)
        r_shoulder = self.get_keypoint(keypoints, Keypoints.RIGHT_SHOULDER)
        l_elbow = self.get_keypoint(keypoints, Keypoints.LEFT_ELBOW)
        r_elbow = self.get_keypoint(keypoints, Keypoints.RIGHT_ELBOW)
        l_wrist = self.get_keypoint(keypoints, Keypoints.LEFT_WRIST)
        r_wrist = self.get_keypoint(keypoints, Keypoints.RIGHT_WRIST)
        l_hip = self.get_keypoint(keypoints, Keypoints.LEFT_HIP)
        r_hip = self.get_keypoint(keypoints, Keypoints.RIGHT_HIP)
        l_knee = self.get_keypoint(keypoints, Keypoints.LEFT_KNEE)
        r_knee = self.get_keypoint(keypoints, Keypoints.RIGHT_KNEE)
        l_ankle = self.get_keypoint(keypoints, Keypoints.LEFT_ANKLE)
        r_ankle = self.get_keypoint(keypoints, Keypoints.RIGHT_ANKLE)

        if not all([l_shoulder, r_shoulder, l_hip, r_hip]):
            return None

        # Segment CoM positions and mass percentages
        segments = []  # list of ((x, y), mass_pct)

        shoulder_mid = midpoint(l_shoulder[:2], r_shoulder[:2])
        hip_mid = midpoint(l_hip[:2], r_hip[:2])

        # Head+neck: from nose (or shoulder midpoint) upward
        if nose:
            head_com = midpoint(nose[:2], shoulder_mid)
            segments.append((head_com, 8.1))
        else:
            # Estimate head above shoulders
            head_offset_y = (hip_mid[1] - shoulder_mid[1]) * 0.3
            head_com = (shoulder_mid[0], shoulder_mid[1] - head_offset_y)
            segments.append((head_com, 8.1))

        # Torso: midpoint of shoulder-center to hip-center
        torso_com = midpoint(shoulder_mid, hip_mid)
        segments.append((torso_com, 49.7))

        # Upper arms (shoulder to elbow)
        if l_elbow:
            segments.append((midpoint(l_shoulder[:2], l_elbow[:2]), 2.8))
        else:
            segments.append((l_shoulder[:2], 2.8))

        if r_elbow:
            segments.append((midpoint(r_shoulder[:2], r_elbow[:2]), 2.8))
        else:
            segments.append((r_shoulder[:2], 2.8))

        # Forearms+hands (elbow to wrist)
        if l_elbow and l_wrist:
            segments.append((midpoint(l_elbow[:2], l_wrist[:2]), 2.2))
        elif l_elbow:
            segments.append((l_elbow[:2], 2.2))
        else:
            segments.append((l_shoulder[:2], 2.2))

        if r_elbow and r_wrist:
            segments.append((midpoint(r_elbow[:2], r_wrist[:2]), 2.2))
        elif r_elbow:
            segments.append((r_elbow[:2], 2.2))
        else:
            segments.append((r_shoulder[:2], 2.2))

        # Thighs (hip to knee)
        if l_knee:
            segments.append((midpoint(l_hip[:2], l_knee[:2]), 10.0))
        else:
            segments.append((l_hip[:2], 10.0))

        if r_knee:
            segments.append((midpoint(r_hip[:2], r_knee[:2]), 10.0))
        else:
            segments.append((r_hip[:2], 10.0))

        # Shanks+feet (knee to ankle)
        if l_knee and l_ankle:
            segments.append((midpoint(l_knee[:2], l_ankle[:2]), 6.1))
        elif l_knee:
            segments.append((l_knee[:2], 6.1))
        else:
            segments.append((l_hip[:2], 6.1))

        if r_knee and r_ankle:
            segments.append((midpoint(r_knee[:2], r_ankle[:2]), 6.1))
        elif r_knee:
            segments.append((r_knee[:2], 6.1))
        else:
            segments.append((r_hip[:2], 6.1))

        # Compute mass-weighted center
        total_mass = sum(m for _, m in segments)
        com_x = sum(pos[0] * m for pos, m in segments) / total_mass
        com_y = sum(pos[1] * m for pos, m in segments) / total_mass

        # Check if CoM is inside body
        com_inside = True
        if body_mask is not None:
            # Use SAM3 body mask to check
            ix, iy = int(round(com_x)), int(round(com_y))
            h, w = body_mask.shape[:2]
            if 0 <= iy < h and 0 <= ix < w:
                com_inside = bool(body_mask[iy, ix] > 0)
            else:
                com_inside = False
        else:
            # Heuristic: check if CoM is within the shoulder-hip bounding box
            # (expanded slightly). During carving, CoM moves laterally outside.
            min_x = min(l_shoulder[0], r_shoulder[0], l_hip[0], r_hip[0])
            max_x = max(l_shoulder[0], r_shoulder[0], l_hip[0], r_hip[0])
            min_y = min(l_shoulder[1], r_shoulder[1])
            max_y = max(l_hip[1], r_hip[1])
            margin_x = (max_x - min_x) * 0.15
            margin_y = (max_y - min_y) * 0.1
            com_inside = (min_x - margin_x <= com_x <= max_x + margin_x and
                          min_y - margin_y <= com_y <= max_y + margin_y)

        return ((com_x, com_y), com_inside)

    def _compute_edge_angle(self, ski: dict, slope_ref_angle: float) -> float:
        """
        Calculate edge angle from ski base orientation relative to slope.

        Args:
            ski: Ski detection dict with 'rotated_rect' from SAM3
            slope_ref_angle: Slope angle from SAM3 snow detection or calibration

        Returns:
            Edge angle in degrees (0 = flat on slope, 90 = full edge)
        """
        if not ski:
            return 0.0

        rotated_rect = ski.get('rotated_rect')
        if rotated_rect is None:
            return 0.0

        pts = rotated_rect

        # Find short edge (ski base width direction)
        edge1_len = math.sqrt((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2)
        edge2_len = math.sqrt((pts[2][0]-pts[1][0])**2 + (pts[2][1]-pts[1][1])**2)

        if edge1_len < edge2_len:
            base_dx = pts[1][0] - pts[0][0]
            base_dy = pts[1][1] - pts[0][1]
        else:
            base_dx = pts[2][0] - pts[1][0]
            base_dy = pts[2][1] - pts[1][1]

        # Ski base width angle
        base_width_angle = math.degrees(math.atan2(base_dy, base_dx))

        # Edge angle = difference from slope reference
        edge_angle = abs(base_width_angle - slope_ref_angle)

        # Normalize to 0-90
        while edge_angle > 90:
            edge_angle = 180 - edge_angle

        return edge_angle

    def compute_metrics(self, keypoints: np.ndarray,
                        left_ski: Optional[dict] = None,
                        right_ski: Optional[dict] = None,
                        body_mask: Optional[np.ndarray] = None) -> Optional[PoseMetrics]:
        """
        Compute biomechanics metrics using full 3D slope geometry.

        All angles are calculated relative to the true slope plane,
        not just the 2D image projection.

        If body_mask (from SAM3) is provided, it's used to:
        - Refine body inclination via mask medial axis
        - Check if center of mass falls inside/outside body silhouette
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

        # If SAM3 body mask available, refine inclination using mask medial axis
        # The mask captures the full body silhouette including tucked positions
        if body_mask is not None:
            try:
                contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if len(largest) >= 5:
                        line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
                        vx, vy = float(line[0][0]), float(line[1][0])
                        cx, cy = float(line[2][0]), float(line[3][0])
                        # Use mask medial axis endpoints as base/top for inclination
                        rect = cv2.minAreaRect(largest)
                        length = max(rect[1]) / 2
                        base_pt = (cx + vx * length, cy + vy * length)
                        top_pt = (cx - vx * length, cy - vy * length)
                        # Ensure base is lower (higher Y) than top
                        if base_pt[1] < top_pt[1]:
                            base_pt, top_pt = top_pt, base_pt
                        body_inclination = geom.inclination_angle(base_pt, top_pt)
            except Exception:
                pass  # Keep keypoint-based inclination

        # Knee angles (pure joint angles, independent of slope)
        knee_angle_left = 180.0
        knee_angle_right = 180.0
        if left_hip and left_knee and left_ankle:
            knee_angle_left = angle_at_joint(left_hip[:2], left_knee[:2], left_ankle[:2])
        if right_hip and right_knee and right_ankle:
            knee_angle_right = angle_at_joint(right_hip[:2], right_knee[:2], right_ankle[:2])

        # Edge angles: ski base tilt relative to slope plane
        # Slope plane from depth estimation (best) or calibration (fallback)
        edge_angle_left = 0.0
        edge_angle_right = 0.0

        # Priority for slope estimation:
        # 1. Depth-based slope (from Depth Anything + SAM3 snow mask)
        # 2. Calibration-based slope (from 4-pole calibration file)
        depth_slope = getattr(self, '_depth_slope_geometry', None)
        if depth_slope is not None:
            slope_ref_angle = depth_slope.fall_line_angle_deg
            self._current_slope_source = 'depth'
        else:
            # Fallback to calibration
            slope_ref_angle = geom.fall_line_angle_deg
            self._current_slope_source = 'calibration'

        self._current_local_slope = slope_ref_angle

        # Edge angle = ski base width orientation vs cross-slope (perpendicular to fall line)
        # Cross-slope angle = fall_line_angle + 90°
        cross_slope_angle = slope_ref_angle + 90

        if left_ski:
            edge_angle_left = self._compute_edge_angle(left_ski, cross_slope_angle)

        if right_ski:
            edge_angle_right = self._compute_edge_angle(right_ski, cross_slope_angle)

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

        # Belly button and center of mass
        belly_button = self.estimate_belly_button(keypoints)
        com_result = self.estimate_center_of_mass(keypoints, body_mask)
        com_pos = com_result[0] if com_result else None
        com_inside = com_result[1] if com_result else True

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
            confidence=round(avg_conf, 3),
            belly_button=belly_button,
            center_of_mass=com_pos,
            com_inside_body=com_inside,
        )

    def analyze_frame(self, frame: np.ndarray, frame_id: int = None) -> Tuple[Optional[np.ndarray], Optional[PoseMetrics], Optional[dict], Optional[dict]]:
        # Auto-increment frame counter if not provided
        if frame_id is None:
            frame_id = self._frame_counter
            self._frame_counter += 1

        results = self.model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None, None, None, None

        keypoints = results[0].keypoints.data
        if len(keypoints) == 0:
            return None, None, None, None

        person_kpts = keypoints[0].cpu().numpy()

        # Detect skis (uses SAM3 if available)
        left_ski, right_ski = self._detect_skis(frame, person_kpts, frame_id)

        # Clear depth/snow cache for new frame
        self._cached_depth_map = None
        self._cached_snow_mask = None

        # Estimate slope from depth map (dynamic 3D terrain detection)
        self._depth_slope_geometry = None
        if self.depth_model is not None:
            # Use ankle position as skier location for local slope estimation
            left_ankle = person_kpts[Keypoints.LEFT_ANKLE]
            right_ankle = person_kpts[Keypoints.RIGHT_ANKLE]
            if left_ankle[2] > 0.3 and right_ankle[2] > 0.3:
                skier_pos = ((left_ankle[0] + right_ankle[0]) / 2,
                            (left_ankle[1] + right_ankle[1]) / 2)
            elif left_ankle[2] > 0.3:
                skier_pos = (left_ankle[0], left_ankle[1])
            elif right_ankle[2] > 0.3:
                skier_pos = (right_ankle[0], right_ankle[1])
            else:
                skier_pos = (frame.shape[1] / 2, frame.shape[0] * 0.7)

            self._depth_slope_geometry = self._estimate_slope_from_depth(frame, skier_pos)

        # Get SAM3 body mask for refined metrics (inclination, CoM inside/outside)
        body_mask = None
        if self.use_sam and self.sam_version == 'sam3':
            body_mask = self._segment_body_sam3(frame)

        # Compute metrics using ski detection data and optional body mask
        metrics = self.compute_metrics(person_kpts, left_ski, right_ski, body_mask)

        if metrics:
            self.history.add(metrics)

        return person_kpts, metrics, left_ski, right_ski

    def _segment_skis_sam(self, frame: np.ndarray, bboxes: List[List[float]]) -> List[dict]:
        """
        Use SAM to refine ski bounding boxes into precise masks.

        Args:
            frame: The video frame
            bboxes: List of bounding boxes [[x1,y1,x2,y2], ...] from YOLO detection

        Returns list of ski detections with:
        - mask: Binary mask of the ski
        - ski_line: Fitted line endpoints
        - direction_angle: Precise ski orientation
        - center: Centroid of the mask
        - box: Bounding box
        """
        if self.sam_model is None or not bboxes:
            return []

        try:
            # Run SAM with bounding box prompts
            results = self.sam_model(frame, bboxes=bboxes, verbose=False)

            if not results or len(results) == 0:
                return []

            detections = []

            for i, result in enumerate(results):
                # Process masks from this result
                if not hasattr(result, 'masks') or result.masks is None:
                    continue

                masks = result.masks.data.cpu().numpy()

                for mask in masks:
                    # Get mask as binary image
                    mask_binary = (mask > 0.5).astype(np.uint8)

                    # Find contours in the mask
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)

                    if area < 500:  # Too small to be a ski
                        continue

                    # Fit line to contour for precise orientation
                    if len(largest_contour) >= 5:
                        line = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        vx, vy = float(line[0][0]), float(line[1][0])
                        cx, cy = float(line[2][0]), float(line[3][0])

                        # Get ski length from min area rect
                        rect = cv2.minAreaRect(largest_contour)
                        length = max(rect[1]) / 2

                        # Calculate endpoints
                        pt1 = (int(cx - vx * length), int(cy - vy * length))
                        pt2 = (int(cx + vx * length), int(cy + vy * length))
                        ski_line = (pt1, pt2)

                        # Direction angle
                        direction_angle = math.degrees(math.atan2(vy, vx))

                        # Get bounding box from contour
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        x1, y1, x2, y2 = x, y, x + w, y + h

                        # Width/height for side view detection
                        width = x2 - x1
                        height = y2 - y1
                        is_side_view = width > 1.5 * height

                        detections.append({
                            'box': (float(x1), float(y1), float(x2), float(y2)),
                            'center': (cx, cy),
                            'direction_angle': direction_angle,
                            'ski_line': ski_line,
                            'width': width,
                            'height': height,
                            'is_side_view': is_side_view,
                            'mask': mask_binary,
                            'conf': 0.95,  # SAM provides high-quality masks
                            'method': 'sam'
                        })

            return detections

        except Exception as e:
            print(f"SAM segmentation error: {e}")
            return []

    def _segment_skis_sam3_text(self, frame: np.ndarray) -> List[dict]:
        """
        Use SAM3 text prompts to detect skis directly without YOLO.

        SAM3 supports text prompts like "ski" to segment objects.
        Uses SAM3SemanticPredictor with set_image() and text= query.
        Returns list of ski detections with mask, line, and angle info.
        """
        if self.sam3_text_predictor is None or self.sam_version != 'sam3':
            return []

        try:
            # Set the image for SAM3SemanticPredictor
            self.sam3_text_predictor.set_image(frame)

            # Query with text prompt "ski"
            results = self.sam3_text_predictor(text=["ski"])

            if not results or len(results) == 0:
                return []

            detections = []

            for result in results:
                if not hasattr(result, 'masks') or result.masks is None:
                    continue

                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

                for i, mask in enumerate(masks):
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if not contours:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)

                    if area < 500:
                        continue

                    if len(largest_contour) >= 5:
                        line = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        vx, vy = float(line[0][0]), float(line[1][0])
                        cx, cy = float(line[2][0]), float(line[3][0])

                        rect = cv2.minAreaRect(largest_contour)
                        rect_center, rect_size, rect_angle = rect
                        # rect_size is (width, height) - longer dimension is ski length
                        ski_length = max(rect_size)
                        ski_base_width = min(rect_size)  # Width of the ski base (for edge angle)
                        length = ski_length / 2

                        pt1 = (int(cx - vx * length), int(cy - vy * length))
                        pt2 = (int(cx + vx * length), int(cy + vy * length))
                        ski_line = (pt1, pt2)
                        direction_angle = math.degrees(math.atan2(vy, vx))

                        # Get rotated rect corners for drawing the full ski base
                        box_points = cv2.boxPoints(rect)
                        box_points = np.int32(box_points)

                        if i < len(boxes):
                            x1, y1, x2, y2 = boxes[i]
                        else:
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            x1, y1, x2, y2 = x, y, x + w, y + h

                        width = x2 - x1
                        height = y2 - y1
                        is_side_view = width > 1.5 * height

                        detections.append({
                            'box': (float(x1), float(y1), float(x2), float(y2)),
                            'center': (cx, cy),
                            'direction_angle': direction_angle,
                            'ski_line': ski_line,
                            'ski_base_width': ski_base_width,  # Width of ski base in pixels
                            'ski_length': ski_length,  # Length of ski in pixels
                            'rotated_rect': box_points,  # 4 corners of rotated bounding box
                            'width': width,
                            'height': height,
                            'is_side_view': is_side_view,
                            'mask': mask_binary,
                            'conf': 0.95,
                            'method': 'sam3_text'
                        })

            return detections

        except Exception as e:
            print(f"SAM3 text segmentation error: {e}")
            return []

    def _detect_slope_from_snow(self, frame: np.ndarray) -> Optional[float]:
        """
        Use SAM3 to segment snow/terrain and derive slope angle.

        SAM3 text prompt "snow" segments the snow surface.
        The slope angle is derived from the snow surface orientation.

        Returns:
            Slope angle in degrees (from horizontal), or None if detection fails
        """
        if self.sam3_text_predictor is None or self.sam_version != 'sam3':
            return None

        try:
            # Set the image for SAM3
            self.sam3_text_predictor.set_image(frame)

            # Query with text prompt "snow"
            results = self.sam3_text_predictor(text=["snow"])

            if not results or len(results) == 0:
                return None

            # Find the largest snow mask (main terrain)
            best_mask = None
            best_area = 0

            for result in results:
                if not hasattr(result, 'masks') or result.masks is None:
                    continue

                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    area = np.sum(mask_binary)
                    if area > best_area:
                        best_area = area
                        best_mask = mask_binary

            if best_mask is None or best_area < 1000:
                return None

            # Find contours of snow surface
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Use the largest contour (main snow surface)
            largest = max(contours, key=cv2.contourArea)

            if len(largest) < 10:
                return None

            # Fit a line to the snow surface boundary
            # The slope of this line represents the terrain angle
            line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(line[0][0]), float(line[1][0])

            # Calculate slope angle from horizontal
            # atan2(vy, vx) gives angle of the line
            slope_angle = math.degrees(math.atan2(vy, vx))

            # Store the detected slope for visualization
            self._sam3_detected_slope = slope_angle
            self._sam3_snow_mask = best_mask

            return slope_angle

        except Exception as e:
            print(f"SAM3 snow detection error: {e}")
            return None

    def _estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map using Depth Anything v2.

        Returns:
            HxW numpy array of depth values (higher = farther)
        """
        if self.depth_model is None or self.depth_processor is None:
            return None

        try:
            import torch
            from PIL import Image

            # Convert BGR to RGB PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # Process image
            inputs = self.depth_processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )

            # Convert to numpy
            depth_map = prediction.squeeze().cpu().numpy()

            return depth_map

        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None

    def _estimate_slope_from_depth(self, frame: np.ndarray,
                                    skier_position: Tuple[float, float]) -> Optional[SlopeGeometry]:
        """
        Estimate local slope using depth map + snow mask.

        Combines monocular depth estimation with SAM3 snow segmentation
        to fit a 3D plane to the terrain near the skier.

        Args:
            frame: Video frame
            skier_position: (x, y) position of skier's feet

        Returns:
            SlopeGeometry with fall_line, pitch, roll from fitted plane
        """
        # Get depth map (cached if already computed this frame)
        if self._cached_depth_map is None:
            self._cached_depth_map = self._estimate_depth(frame)

        if self._cached_depth_map is None:
            return None

        # Get snow mask from SAM3
        if self._cached_snow_mask is None and self.sam3_text_predictor is not None:
            try:
                self.sam3_text_predictor.set_image(frame)
                results = self.sam3_text_predictor(text=["snow"])

                if results and len(results) > 0:
                    for result in results:
                        if hasattr(result, 'masks') and result.masks is not None:
                            masks = result.masks.data.cpu().numpy()
                            if len(masks) > 0:
                                # Use largest snow mask
                                areas = [np.sum(m > 0.5) for m in masks]
                                best_idx = np.argmax(areas)
                                self._cached_snow_mask = (masks[best_idx] > 0.5).astype(np.uint8)
                                break
            except Exception as e:
                print(f"Snow mask error: {e}")

        if self._cached_snow_mask is None:
            # No snow mask - use lower half of frame as terrain estimate
            h, w = frame.shape[:2]
            self._cached_snow_mask = np.zeros((h, w), dtype=np.uint8)
            self._cached_snow_mask[h//2:, :] = 1

        # Fit slope plane to depth + snow mask
        slope_geom = SlopeGeometry.from_depth_map(
            self._cached_depth_map,
            self._cached_snow_mask,
            skier_position,
            local_radius=200
        )

        return slope_geom

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

    def _detect_skis(self, frame: np.ndarray, keypoints: np.ndarray, frame_id: int = 0) -> Tuple[Optional[dict], Optional[dict]]:
        """Detect skis and match to ankles. Returns (left_ski, right_ski) dicts with box, angle, fore_aft.

        Detection priority:
        1. SAM3 text prompts ("ski") - no YOLO needed
        2. YOLO + SAM2 bbox prompts - precise segmentation
        3. YOLO + dark pixel detection - fallback
        """
        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (float(kp[0]), float(kp[1]))
            return None

        la = get_pt(Keypoints.LEFT_ANKLE)
        ra = get_pt(Keypoints.RIGHT_ANKLE)

        left_ski = None
        right_ski = None
        detections = []
        yolo_boxes = None  # Track YOLO boxes for step 3 fallback

        # Step 1: Try SAM3 text prompts first (no YOLO needed)
        if self.sam_model is not None and self.sam_version == 'sam3':
            sam3_detections = self._segment_skis_sam3_text(frame)
            if sam3_detections:
                detections = sam3_detections

        # Step 2: Fall back to YOLO detection
        if not detections and self.ski_detector is not None:
            yolo_results = self.ski_detector(frame, verbose=False, conf=0.3)

            if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                yolo_boxes = yolo_results[0].boxes
                bboxes = []
                for box in yolo_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append([float(x1), float(y1), float(x2), float(y2)])

                # Step 2a: If SAM2 available, use bbox prompts for segmentation
                if self.sam_model is not None and self.sam_version == 'sam2' and bboxes:
                    sam_detections = self._segment_skis_sam(frame, bboxes)
                    if sam_detections:
                        detections = sam_detections

        # Step 3: Fall back to YOLO + dark pixel detection if SAM unavailable or failed
        if not detections and yolo_boxes is not None:
            for box in yolo_boxes:
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

                # Estimate ski base width from bounding box (less accurate than SAM)
                ski_base_width = min(width, height)  # Approximate width
                ski_length = max(width, height)

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'center': (cx, cy),
                    'direction_angle': direction_angle,
                    'width': width,
                    'height': height,
                    'is_side_view': is_side_view,
                    'ski_line': ski_line,
                    'ski_base_width': ski_base_width,  # Width of ski base in pixels
                    'ski_length': ski_length,  # Length of ski in pixels
                    'method': 'yolo'
                })

        if not detections:
            return None, None

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
                              left_ski: Optional[dict] = None, right_ski: Optional[dict] = None,
                              metrics: Optional[PoseMetrics] = None) -> np.ndarray:
        """Draw ski base as 3D rectangle showing edge angle relative to slope."""
        output = frame.copy()
        thickness = max(2, int(3 * scale))

        def draw_ski_3d(ski, color, edge_angle_deg: float):
            """Draw ski as a 3D box showing the base face tilted at edge_angle."""
            if not ski:
                return

            ski_line = ski.get('ski_line')
            if not ski_line:
                return

            pt1, pt2 = ski_line
            ski_length = ski.get('ski_length', 0)
            ski_width = ski.get('ski_base_width', 0)

            if ski_length <= 0:
                ski_length = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
            if ski_width <= 0:
                ski_width = ski_length / 20  # Default aspect ratio

            # Calculate ski direction vector
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1:
                return

            # Unit vectors along ski and perpendicular
            ux, uy = dx / length, dy / length  # Along ski
            px, py = -uy, ux  # Perpendicular (in image plane)

            # The ski base face tilts based on edge angle
            # At 0°, we see the top surface (narrow)
            # At 90°, we see the base face (wider)
            edge_rad = math.radians(edge_angle_deg)

            # Calculate apparent width based on edge angle
            # Base width projects: base_face_visible = width * sin(edge_angle)
            # Ski thickness for 3D effect
            SKI_THICKNESS_RATIO = 0.02  # ~3cm thickness for 150cm ski
            thickness_px = ski_length * SKI_THICKNESS_RATIO

            # 3D box vertices:
            # Bottom face (base) has 4 corners, top face has 4 corners
            # At edge angle θ:
            # - Base face appears tilted, showing its width
            # - We draw parallelogram to show 3D perspective

            cx = (pt1[0] + pt2[0]) / 2
            cy = (pt1[1] + pt2[1]) / 2
            half_len = length / 2
            half_width = ski_width / 2

            # Edge creates a visual offset perpendicular to ski
            # More edge = base face tilts toward camera = appears to shift
            edge_offset = thickness_px * math.sin(edge_rad) * 0.5

            # Draw 3D ski base box:
            # 1. Bottom edge (base edge contacting snow)
            # 2. Top edge (top surface edge)
            # 3. Connecting edges showing the tilted base face

            # Base edge (contact with snow) - shifted based on edge angle
            base1 = (int(pt1[0] - px * edge_offset), int(pt1[1] - py * edge_offset))
            base2 = (int(pt2[0] - px * edge_offset), int(pt2[1] - py * edge_offset))

            # Top surface edge - opposite side
            top_offset = half_width - edge_offset
            top1 = (int(pt1[0] + px * top_offset), int(pt1[1] + py * top_offset))
            top2 = (int(pt2[0] + px * top_offset), int(pt2[1] + py * top_offset))

            # Bottom surface edge - base face showing
            base_face_height = thickness_px * math.cos(edge_rad)  # How much of base face visible
            bottom1 = (int(pt1[0] - px * (edge_offset + base_face_height)), int(pt1[1] - py * (edge_offset + base_face_height)))
            bottom2 = (int(pt2[0] - px * (edge_offset + base_face_height)), int(pt2[1] - py * (edge_offset + base_face_height)))

            # Draw filled 3D shape
            # Side face (base) - darker shade to show depth
            if edge_angle_deg > 5:
                base_face = np.array([base1, base2, bottom2, bottom1], dtype=np.int32)
                # Semi-transparent fill for base face
                overlay = output.copy()
                dark_color = tuple(max(0, int(c * 0.4)) for c in color)
                cv2.fillPoly(overlay, [base_face], dark_color)
                cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

            # Top face
            top_face = np.array([base1, base2, top2, top1], dtype=np.int32)
            overlay = output.copy()
            cv2.fillPoly(overlay, [top_face], color)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

            # Draw edges with thick lines
            # Contact edge (where ski meets snow) - brightest
            cv2.line(output, base1, base2, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.line(output, base1, base2, (0, 255, 255), thickness + 1, cv2.LINE_AA)  # Cyan = contact edge

            # Other edges
            cv2.line(output, top1, top2, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.line(output, top1, top2, color, thickness - 1, cv2.LINE_AA)
            cv2.line(output, base1, top1, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.line(output, base1, top1, color, thickness - 1, cv2.LINE_AA)
            cv2.line(output, base2, top2, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.line(output, base2, top2, color, thickness - 1, cv2.LINE_AA)

            # If edge angle > 5°, draw the base face edge
            if edge_angle_deg > 5:
                cv2.line(output, bottom1, bottom2, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.line(output, bottom1, bottom2, tuple(max(0, int(c * 0.6)) for c in color), thickness - 1, cv2.LINE_AA)
                cv2.line(output, base1, bottom1, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.line(output, base2, bottom2, (0, 0, 0), thickness, cv2.LINE_AA)

            # Show edge angle debug info (calibration-based)
            if ski.get('center'):
                cx_ski, cy_ski = int(ski['center'][0]), int(ski['center'][1])
                method = ski.get('method', '?')
                label = f"{edge_angle_deg:.0f}° ({method})"
                debug_font = 0.65 * scale
                cv2.putText(output, label, (cx_ski - int(80 * scale), cy_ski + int(50 * scale)),
                           cv2.FONT_HERSHEY_SIMPLEX, debug_font, (0, 0, 0),
                           max(2, int(2 * scale)), cv2.LINE_AA)
                cv2.putText(output, label, (cx_ski - int(80 * scale), cy_ski + int(50 * scale)),
                           cv2.FONT_HERSHEY_SIMPLEX, debug_font, (255, 255, 0),
                           max(1, int(scale)), cv2.LINE_AA)

        # Get edge angles from metrics if available
        edge_left = metrics.edge_angle_left if metrics else 45.0
        edge_right = metrics.edge_angle_right if metrics else 45.0

        # Draw 3D ski representations
        draw_ski_3d(left_ski, (255, 255, 0), edge_left)  # Cyan
        draw_ski_3d(right_ski, (255, 255, 255), edge_right)  # White

        # Draw ski edge contact points "L" and "R" under ankles on internal side
        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (int(kp[0]), int(kp[1]))
            return None

        la = get_pt(Keypoints.LEFT_ANKLE)
        ra = get_pt(Keypoints.RIGHT_ANKLE)
        edge_font = 0.6 * scale
        edge_pt_radius = max(5, int(6 * scale))

        # For left ski: edge point is on internal side (toward right ski)
        # Position: below ankle, offset toward the center
        if la:
            # Edge contact point below left ankle, offset right (internal side)
            offset_x = int(15 * scale)  # Offset toward center
            offset_y = int(35 * scale)  # Below ankle
            edge_pt_l = (la[0] + offset_x, la[1] + offset_y)
            # Draw point with "L" label
            cv2.circle(output, edge_pt_l, edge_pt_radius, (0, 0, 0), -1)
            cv2.circle(output, edge_pt_l, edge_pt_radius, (0, 255, 255), 2)  # Cyan ring
            cv2.putText(output, "L", (edge_pt_l[0] - int(5 * scale), edge_pt_l[1] + int(4 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 255),
                       max(1, int(scale)), cv2.LINE_AA)

        # For right ski: edge point is on internal side (toward left ski)
        if ra:
            # Edge contact point below right ankle, offset left (internal side)
            offset_x = int(-15 * scale)  # Offset toward center
            offset_y = int(35 * scale)  # Below ankle
            edge_pt_r = (ra[0] + offset_x, ra[1] + offset_y)
            # Draw point with "R" label
            cv2.circle(output, edge_pt_r, edge_pt_radius, (0, 0, 0), -1)
            cv2.circle(output, edge_pt_r, edge_pt_radius, (255, 255, 255), 2)  # White ring
            cv2.putText(output, "R", (edge_pt_r[0] - int(5 * scale), edge_pt_r[1] + int(4 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 255, 255),
                       max(1, int(scale)), cv2.LINE_AA)

        return output

    def _draw_slope_line(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw slope line (dotted) perpendicular to fall line - the snow surface reference.
        Uses ski-derived local slope when available for accuracy."""
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

            # Use calibrated slope angle (from gate positions / terrain)
            slope_angle = self.slope_angle_deg

            # Slope line is perpendicular to fall line (horizontal on the slope surface)
            slope_line_angle_rad = math.radians(slope_angle)
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

            # Label "SLOPE" at right end - always from calibration
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "SLOPE"
            label_pos = (pt2[0] + int(10 * scale), pt2[1])
            cv2.putText(output, label, label_pos, font, 0.45 * scale, (255, 255, 255),
                       max(1, int(2 * scale)), cv2.LINE_AA)

        return output

    def _draw_fall_line(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw fall line arrow starting from snow level between skis.
        Uses ski-derived local slope when available for accuracy."""
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

            # Use calibrated slope angle (from gate positions / terrain)
            slope_angle = self.slope_angle_deg

            # Fall line direction (perpendicular to slope line, pointing downhill)
            line_len = int(150 * scale)
            angle_rad = math.radians(90 + slope_angle)
            dx = int(line_len * math.cos(angle_rad))
            dy = int(line_len * math.sin(angle_rad))
            end_pt = (start_pt[0] + dx, start_pt[1] + dy)

            # Draw arrow
            cv2.arrowedLine(output, start_pt, end_pt, (0, 165, 255), int(4 * scale),
                           tipLength=0.15, line_type=cv2.LINE_AA)

            # Label - always from calibration
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "FALL LINE"
            label_pos = (end_pt[0] + int(10 * scale), end_pt[1])
            cv2.putText(output, label, label_pos, font, 0.5 * scale, (0, 165, 255),
                       max(1, int(2 * scale)), cv2.LINE_AA)

        return output

    def _draw_metrics_legend(self, frame: np.ndarray, scale: float,
                              metrics: Optional[PoseMetrics] = None) -> np.ndarray:
        """Draw metrics legend panel with current values (always visible).

        This shows real-time measurements. Graphs are rendered separately in the browser
        from saved JSON data, allowing toggle on/off and opacity control.
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_font = 0.6 * scale
        value_font = 0.75 * scale
        text_thickness = max(1, int(2 * scale))
        value_thickness = max(2, int(2 * scale))

        # Position: Lower on left side (30% down from top)
        panel_w = int(340 * scale)
        panel_h = int(520 * scale)  # Taller to fit title + warning + metrics + model info with larger fonts
        panel_x = int(15 * scale)
        panel_y = int(h * 0.25)  # 25% down from top

        # Semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

        # Border
        cv2.rectangle(output, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (80, 80, 80), 1)

        # Title (bigger)
        title_font = 0.8 * scale
        cv2.putText(output, "METRICS PROTOTYPE", (panel_x + int(10 * scale), panel_y + int(28 * scale)),
                   font, title_font, (255, 255, 255), max(2, int(2 * scale)), cv2.LINE_AA)
        # Warning note (yellow, bigger but smaller than title)
        warning_font = 0.55 * scale
        cv2.putText(output, "= There are errors!", (panel_x + int(10 * scale), panel_y + int(50 * scale)),
                   font, warning_font, (0, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)

        # Compute current values
        if metrics:
            edge_angle = (90 - abs(metrics.edge_angle_left) + 90 - abs(metrics.edge_angle_right)) / 2
            edge_sym = metrics.edge_symmetry_pct
            fore_aft = 0.0
            if metrics.fore_aft_left is not None and metrics.fore_aft_right is not None:
                fore_aft = (metrics.fore_aft_left + metrics.fore_aft_right) / 2
            elif metrics.fore_aft_left is not None:
                fore_aft = metrics.fore_aft_left
            elif metrics.fore_aft_right is not None:
                fore_aft = metrics.fore_aft_right
            angulation = metrics.body_angulation
            inclination = metrics.body_inclination
            # Secondary metrics
            shoulder_slope = metrics.shoulder_angle_to_slope
            hip_slope = metrics.hip_angle_to_slope
            edge_left = 90 - abs(metrics.edge_angle_left)
            edge_right = 90 - abs(metrics.edge_angle_right)
            fa_left = metrics.fore_aft_left if metrics.fore_aft_left is not None else 0.0
            fa_right = metrics.fore_aft_right if metrics.fore_aft_right is not None else 0.0
        else:
            edge_angle = edge_sym = fore_aft = angulation = inclination = 0.0
            shoulder_slope = hip_slope = edge_left = edge_right = fa_left = fa_right = 0.0

        # Main metrics (5): Edge Angle, Edge Similarity, Fore/Aft, Angulation, Inclination
        # Note: OpenCV putText doesn't support Unicode, so use "deg" instead of "°"
        main_items = [
            ("Edge Angle", edge_angle, "deg", (255, 100, 100)),
            ("Edge Similarity", edge_sym, "%", (0, 255, 0)),
            ("Fore/Aft", fore_aft, "deg", (255, 200, 100)),
            ("Angulation", angulation, "deg", (0, 255, 255)),
            ("Inclination", inclination, "deg", (255, 180, 255)),
        ]

        # Secondary metrics (smaller): Shoulders/Slope, Hips/Slope, Edge R/L, Fore/Aft R/L
        extra_items = [
            ("Shoulders/Slope", shoulder_slope, "deg", (180, 100, 180)),
            ("Hips/Slope", hip_slope, "deg", (180, 180, 100)),
            ("Edge L/R", f"{edge_left:.0f}/{edge_right:.0f}", "deg", (255, 150, 150)),
            ("Fore/Aft L/R", f"{fa_left:.0f}/{fa_right:.0f}", "deg", (255, 220, 150)),
        ]

        # Draw main metrics
        line_h = int(28 * scale)
        start_y = panel_y + int(70 * scale)  # Account for bigger title + warning

        for i, (name, value, unit, color) in enumerate(main_items):
            y = start_y + i * line_h

            # Color indicator square
            sq_size = int(12 * scale)
            cv2.rectangle(output, (panel_x + int(10 * scale), y - sq_size + 2),
                         (panel_x + int(10 * scale) + sq_size, y + 2), color, -1)

            # Label
            cv2.putText(output, name, (panel_x + int(28 * scale), y),
                       font, label_font, (180, 180, 180), text_thickness, cv2.LINE_AA)

            # Value (right-aligned)
            value_str = f"{value:.1f}{unit}"
            (tw, _), _ = cv2.getTextSize(value_str, font, value_font, value_thickness)
            cv2.putText(output, value_str, (panel_x + panel_w - tw - int(10 * scale), y),
                       font, value_font, (255, 255, 255), value_thickness, cv2.LINE_AA)

        # Draw additional metrics (smaller, below main metrics)
        small_label_font = 0.55 * scale
        small_value_font = 0.65 * scale
        small_line_h = int(26 * scale)
        extra_start_y = start_y + len(main_items) * line_h + int(8 * scale)

        # Separator line
        sep_y = extra_start_y - int(6 * scale)
        cv2.line(output, (panel_x + int(10 * scale), sep_y),
                (panel_x + panel_w - int(10 * scale), sep_y), (60, 60, 60), 1)

        for i, (name, value, unit, color) in enumerate(extra_items):
            y = extra_start_y + i * small_line_h

            # Small color indicator
            sq_size = int(8 * scale)
            cv2.rectangle(output, (panel_x + int(10 * scale), y - sq_size + 2),
                         (panel_x + int(10 * scale) + sq_size, y + 2), color, -1)

            # Label (smaller)
            cv2.putText(output, name, (panel_x + int(24 * scale), y),
                       font, small_label_font, (150, 150, 150), 1, cv2.LINE_AA)

            # Value (right-aligned, smaller) - handle both float and string values
            if isinstance(value, str):
                value_str = f"{value}{unit}"
            else:
                value_str = f"{value:.1f}{unit}"
            (tw, _), _ = cv2.getTextSize(value_str, font, small_value_font, 1)
            cv2.putText(output, value_str, (panel_x + panel_w - tw - int(10 * scale), y),
                       font, small_value_font, (220, 220, 220), 1, cv2.LINE_AA)

        # Model info section at the bottom
        model_start_y = extra_start_y + len(extra_items) * small_line_h + int(12 * scale)

        # Separator line
        cv2.line(output, (panel_x + int(10 * scale), model_start_y - int(6 * scale)),
                (panel_x + panel_w - int(10 * scale), model_start_y - int(6 * scale)), (60, 60, 60), 1)

        # Model info (larger font for better visibility)
        model_font = 0.6 * scale
        model_line_h = int(26 * scale)
        model_color = (150, 150, 150)

        # Pose model
        cv2.putText(output, f"Pose: yolov8{self.model_size}-pose",
                   (panel_x + int(10 * scale), model_start_y),
                   font, model_font, model_color, 1, cv2.LINE_AA)

        # Device
        cv2.putText(output, f"Device: {self.device}",
                   (panel_x + int(10 * scale), model_start_y + model_line_h),
                   font, model_font, model_color, 1, cv2.LINE_AA)

        # SAM version
        sam_str = self.sam_version if self.sam_version else "none"
        cv2.putText(output, f"SAM: {sam_str}",
                   (panel_x + int(10 * scale), model_start_y + 2 * model_line_h),
                   font, model_font, model_color, 1, cv2.LINE_AA)

        # Ski detector
        ski_str = "yolo+sam" if self.ski_detector and self.sam_model else ("yolo" if self.ski_detector else "none")
        cv2.putText(output, f"Ski: {ski_str}",
                   (panel_x + int(10 * scale), model_start_y + 3 * model_line_h),
                   font, model_font, model_color, 1, cv2.LINE_AA)

        return output

    def get_metrics_for_export(self) -> dict:
        """Get all metrics history as a dict for JSON export.

        This data is used by the browser to render interactive graphs.
        """
        # Compute derived metrics
        edge_angle_avg = []
        for i in range(len(self.history.edge_left)):
            l = 90 - abs(self.history.edge_left[i])
            r = 90 - abs(self.history.edge_right[i]) if i < len(self.history.edge_right) else l
            edge_angle_avg.append(round((l + r) / 2, 1))

        fore_aft_avg = []
        for i in range(max(len(self.history.fore_aft_left), len(self.history.fore_aft_right))):
            l = self.history.fore_aft_left[i] if i < len(self.history.fore_aft_left) else 0
            r = self.history.fore_aft_right[i] if i < len(self.history.fore_aft_right) else 0
            fore_aft_avg.append(round((l + r) / 2, 1) if (l != 0 or r != 0) else 0)

        return {
            "metrics": [
                {
                    "name": "Edge Angle",
                    "data": edge_angle_avg,
                    "color": "#ff6464",
                    "min": 0, "max": 90, "unit": "°"
                },
                {
                    "name": "Edge Similarity",
                    "data": [round(x, 1) for x in self.history.edge_symmetry],
                    "color": "#00ff00",
                    "min": 0, "max": 100, "unit": "%"
                },
                {
                    "name": "Fore/Aft",
                    "data": fore_aft_avg,
                    "color": "#ffc864",
                    "min": -30, "max": 30, "unit": "°"
                },
                {
                    "name": "Angulation",
                    "data": [round(x, 1) for x in self.history.angulation],
                    "color": "#00ffff",
                    "min": 0, "max": 60, "unit": "°"
                },
                {
                    "name": "Inclination",
                    "data": [round(x, 1) for x in self.history.inclination],
                    "color": "#ffb4ff",
                    "min": -60, "max": 60, "unit": "°"
                }
            ],
            "summary": {
                "shoulder_avg": round(sum(abs(x) for x in self.history.shoulder) / len(self.history.shoulder), 1) if self.history.shoulder else 0,
                "hip_avg": round(sum(abs(x) for x in self.history.hip) / len(self.history.hip), 1) if self.history.hip else 0,
                "edge_left_avg": round(90 - sum(abs(x) for x in self.history.edge_left) / len(self.history.edge_left), 1) if self.history.edge_left else 0,
                "edge_right_avg": round(90 - sum(abs(x) for x in self.history.edge_right) / len(self.history.edge_right), 1) if self.history.edge_right else 0,
            }
        }

    def _draw_extended_lines(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw extended lines through shoulders and hips (no slope reference lines)."""
        output = frame.copy()
        line_ext = int(50 * scale)  # Reduced by half for cleaner visualization
        thickness = max(2, int(3 * scale))

        def get_pt(idx):
            kp = keypoints[idx]
            if kp[2] > 0.3:
                return (int(kp[0]), int(kp[1]))
            return None

        ls = get_pt(Keypoints.LEFT_SHOULDER)
        rs = get_pt(Keypoints.RIGHT_SHOULDER)
        lh = get_pt(Keypoints.LEFT_HIP)
        rh = get_pt(Keypoints.RIGHT_HIP)

        # Extended shoulder line (magenta) - no slope reference
        if ls and rs:
            dx = rs[0] - ls[0]
            dy = rs[1] - ls[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                ext_l = (int(ls[0] - ux * line_ext), int(ls[1] - uy * line_ext))
                ext_r = (int(rs[0] + ux * line_ext), int(rs[1] + uy * line_ext))
                cv2.line(output, ext_l, ext_r, (255, 0, 255), thickness, cv2.LINE_AA)

        # Extended hip line (cyan) - no slope reference
        if lh and rh:
            dx = rh[0] - lh[0]
            dy = rh[1] - lh[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                ext_l = (int(lh[0] - ux * line_ext), int(lh[1] - uy * line_ext))
                ext_r = (int(rh[0] + ux * line_ext), int(rh[1] + uy * line_ext))
                cv2.line(output, ext_l, ext_r, (255, 255, 0), thickness, cv2.LINE_AA)

        return output

    def _draw_gate_info_box(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """
        Draw gate info box in top-right corner.

        Always displays slope gradient. If gate info is set, also shows:
        - Gate number and color
        - Distance from previous gate
        - Vertical drop
        - GPS accuracy
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 * scale  # Bigger text
        text_thickness = max(1, int(2 * scale))
        line_height = int(32 * scale)
        margin = int(150 * scale)  # Large margin from right edge to avoid truncation
        padding = int(10 * scale)

        lines = []

        # Gate info if available
        if self.current_gate:
            gate = self.current_gate
            # Gate number and color
            if gate.get('id') is not None:
                color_text = f" ({gate['color']})" if gate.get('color') else ""
                lines.append((f"Gate {gate['id']}{color_text}", (255, 255, 255)))

            # Previous gate reference
            if gate.get('prev_id') is not None:
                prev_color_text = f" ({gate['prev_color']})" if gate.get('prev_color') else ""
                lines.append((f"From Gate {gate['prev_id']}{prev_color_text}:", (180, 180, 180)))

            # Distance from previous gate
            if gate.get('dist_from_prev') is not None:
                lines.append((f"Dist: {gate['dist_from_prev']:.1f}m", (200, 200, 200)))

            # Left/right offset from fall line
            if gate.get('offset_lr') is not None:
                offset_val = gate['offset_lr']
                offset_dir = "L" if offset_val < 0 else "R"
                lines.append((f"Offset: {abs(offset_val):.1f}m {offset_dir}", (200, 200, 200)))

            # Vertical drop
            if gate.get('drop') is not None:
                lines.append((f"Drop: {gate['drop']:.1f}m", (200, 200, 200)))

            # GPS accuracy
            if gate.get('gps_accuracy') is not None:
                lines.append((f"GPS: +/-{gate['gps_accuracy']:.2f}m", (180, 180, 180)))

        # Slope gradient removed - now shown in graph legend

        if not lines:
            return frame

        # Calculate box dimensions
        max_text_width = 0
        for text, _ in lines:
            (tw, _), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
            max_text_width = max(max_text_width, tw)

        box_width = max_text_width + 2 * padding
        box_height = len(lines) * line_height + padding

        # Box position (top-right corner)
        box_x = w - box_width - margin
        box_y = margin

        # Draw semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        # Draw border
        cv2.rectangle(output, (box_x, box_y), (box_x + box_width, box_y + box_height),
                     (100, 100, 100), 1)

        # Draw text lines
        y = box_y + padding + int(16 * scale)
        for text, color in lines:
            cv2.putText(output, text, (box_x + padding, y), font, font_scale,
                       color, text_thickness, cv2.LINE_AA)
            y += line_height

        return output

    def draw_overlay(self, frame: np.ndarray, keypoints: np.ndarray,
                     metrics: Optional[PoseMetrics] = None,
                     left_ski: Optional[dict] = None,
                     right_ski: Optional[dict] = None,
                     show_debug_angles: bool = True,
                     frame_num: int = None,
                     total_frames: int = None) -> np.ndarray:
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

        # Draw keypoints (skip only face, include arms/elbows/wrists as points only)
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
                # Arm keypoints get slightly smaller radius but are drawn
                r = point_radius if i not in ARM_KEYPOINTS else max(2, int(3 * scale))
                cv2.circle(output, pt, r, color, -1)
                cv2.circle(output, pt, r + 1, (255, 255, 255), 1)

        # Draw belly button point
        if metrics and metrics.belly_button:
            bx, by = int(metrics.belly_button[0]), int(metrics.belly_button[1])
            r = max(4, int(5 * scale))
            cv2.circle(output, (bx, by), r, (0, 0, 0), -1)       # Black fill
            cv2.circle(output, (bx, by), r, (255, 200, 100), 2)   # Orange ring
            # Small label
            cv2.putText(output, "BB", (bx + r + 3, by + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (255, 200, 100),
                       max(1, int(scale)), cv2.LINE_AA)

            # Draw line from belly button to center of neck (spine line)
            # Neck is between shoulders but higher - use nose as reference
            ls = keypoints[Keypoints.LEFT_SHOULDER]
            rs = keypoints[Keypoints.RIGHT_SHOULDER]
            nose = keypoints[Keypoints.NOSE]
            if ls[2] > 0.3 and rs[2] > 0.3:
                shoulder_mid_x = (ls[0] + rs[0]) / 2
                shoulder_mid_y = (ls[1] + rs[1]) / 2

                # Neck is ~30% of way from shoulder midpoint to nose
                if nose[2] > 0.3:
                    neck_x = int(shoulder_mid_x + 0.3 * (nose[0] - shoulder_mid_x))
                    neck_y = int(shoulder_mid_y + 0.3 * (nose[1] - shoulder_mid_y))
                else:
                    # Fallback: estimate neck 20% above shoulder midpoint toward frame center
                    neck_x = int(shoulder_mid_x)
                    neck_y = int(shoulder_mid_y - 0.15 * abs(ls[1] - keypoints[Keypoints.LEFT_HIP][1]))

                # Draw spine line (orange)
                cv2.line(output, (bx, by), (neck_x, neck_y), (255, 200, 100),
                        max(2, int(3 * scale)), cv2.LINE_AA)

                # Draw neck point
                cv2.circle(output, (neck_x, neck_y), max(3, int(4 * scale)), (255, 200, 100), -1)
                cv2.circle(output, (neck_x, neck_y), max(4, int(5 * scale)), (255, 255, 255), 1)

        # Draw center of mass (CoM)
        if metrics and metrics.center_of_mass:
            cx, cy = int(metrics.center_of_mass[0]), int(metrics.center_of_mass[1])
            r = max(6, int(8 * scale))

            if metrics.com_inside_body:
                # Inside body: solid white diamond
                com_color = (255, 255, 255)
            else:
                # Outside body (carving!): red diamond — the interesting case
                com_color = (0, 0, 255)

            # Draw diamond shape for CoM
            pts = np.array([
                [cx, cy - r],
                [cx + r, cy],
                [cx, cy + r],
                [cx - r, cy],
            ], dtype=np.int32)
            cv2.fillPoly(output, [pts], com_color)
            cv2.polylines(output, [pts], True, (0, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)

            # Label
            label = "CoM"
            if not metrics.com_inside_body:
                label = "CoM (OUT)"
            cv2.putText(output, label, (cx + r + 4, cy + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, com_color,
                       max(1, int(scale)), cv2.LINE_AA)

            # Draw line from belly button to CoM if they differ significantly
            if metrics.belly_button:
                bbx, bby = int(metrics.belly_button[0]), int(metrics.belly_button[1])
                dist = math.sqrt((cx - bbx)**2 + (cy - bby)**2)
                if dist > 10 * scale:
                    # Dashed line from belly button to CoM
                    cv2.line(output, (bbx, bby), (cx, cy), com_color,
                            max(1, int(2 * scale)), cv2.LINE_AA)

        # Draw ski rectangles with metrics labels
        output = self._draw_ski_rectangles(output, keypoints, scale, left_ski, right_ski, metrics)

        # Draw slope line (dotted, perpendicular to fall line)
        output = self._draw_slope_line(output, keypoints, scale)

        # Draw fall line from snow level
        output = self._draw_fall_line(output, keypoints, scale)

        # Draw metrics legend panel (always visible, shows current values)
        # Graphs are rendered in browser from JSON data - allows toggle/opacity control
        output = self._draw_metrics_legend(output, scale, metrics)

        # Draw gate info box in top-right corner
        output = self._draw_gate_info_box(output, scale)

        # Draw date/time overlay in bottom-right corner
        if self.video_timestamp:
            font = cv2.FONT_HERSHEY_SIMPLEX
            date_str = self.video_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if self.video_source_name:
                date_str = f"{self.video_source_name} | {date_str}"

            ts_font_scale = 0.5 * scale
            ts_thickness = max(1, int(scale))
            (tw, th), _ = cv2.getTextSize(date_str, font, ts_font_scale, ts_thickness)

            # Position in bottom-right, above any logos
            margin = int(15 * scale)
            ts_x = w - tw - margin
            ts_y = h - margin - int(80 * scale)  # Above logo area

            # Semi-transparent background
            cv2.rectangle(output, (ts_x - 5, ts_y - th - 5),
                         (ts_x + tw + 5, ts_y + 5), (0, 0, 0), -1)
            cv2.putText(output, date_str, (ts_x, ts_y), font, ts_font_scale,
                       (200, 200, 200), ts_thickness, cv2.LINE_AA)

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
    parser.add_argument('--gate-info', metavar='JSON',
                        help='Gate info JSON: {"gate_id":9,"color":"blue","prev_id":8,...}')
    parser.add_argument('--logos', metavar='LIST',
                        help='Comma-separated logo filenames for bottom-left overlay')
    parser.add_argument('--show-curves', action='store_true',
                        help='Show graph curves (default: off, labels/scale only)')

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

    # Set gate info if provided
    if args.gate_info:
        try:
            gate_data = json.loads(args.gate_info)
            analyzer.set_gate_info(
                gate_id=gate_data.get('gate_id'),
                color=gate_data.get('color'),
                prev_gate_id=gate_data.get('prev_id'),
                prev_gate_color=gate_data.get('prev_color'),
                dist_from_prev=gate_data.get('dist_from_prev'),
                offset_lr=gate_data.get('offset_lr'),
                drop=gate_data.get('drop'),
                gps_accuracy=gate_data.get('gps_accuracy')
            )
            print(f"Gate info: Gate {gate_data.get('gate_id')} from Gate {gate_data.get('prev_id')}")
        except Exception as e:
            print(f"Warning: Could not parse gate info: {e}")

    # Set logos if provided
    if args.logos:
        logos = [l.strip() for l in args.logos.split(',')]
        analyzer.set_logos(logos)
        print(f"Logos: {logos}")

    # Set graph curves visibility
    analyzer.show_graph_curves = args.show_curves
    print(f"Graph curves: {'ON' if args.show_curves else 'OFF (labels/scale only)'}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f}fps, {total_frames} frames")

    # Extract video timestamp from file modification time and set on analyzer
    import os
    video_path = Path(args.video_path)
    try:
        # Try to get file modification time
        mtime = os.path.getmtime(args.video_path)
        video_timestamp = datetime.fromtimestamp(mtime)
        source_name = video_path.stem  # Filename without extension
        analyzer.set_video_timestamp(video_timestamp, source_name)
        print(f"Video timestamp: {video_timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({source_name})")
    except Exception as e:
        print(f"Warning: Could not get video timestamp: {e}")

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

    # Keep track of last known pose for intermediate frames (when sample_rate > 1)
    last_keypoints = None
    last_metrics = None
    last_left_ski = None
    last_right_ski = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose analysis only on sampled frames
        if frame_num % args.sample_rate == 0:
            result = analyzer.analyze_frame(frame, frame_id=frame_num)
            keypoints, metrics, left_ski, right_ski = result

            if keypoints is not None:
                poses_detected += 1
                if metrics:
                    all_metrics.append(metrics)
                # Save for reuse on intermediate frames
                last_keypoints = keypoints
                last_metrics = metrics
                last_left_ski = left_ski
                last_right_ski = right_ski

        # Always write every frame to output video (for smooth playback)
        if last_keypoints is not None:
            try:
                annotated = analyzer.draw_overlay(frame, last_keypoints, last_metrics,
                                                   last_left_ski, last_right_ski,
                                                   frame_num=frame_num, total_frames=total_frames)
                if out_video:
                    out_video.write(annotated)
                if args.output_frames and frame_num % args.sample_rate == 0:
                    cv2.imwrite(f"{args.output_frames}/frame_{frame_num:05d}.jpg", annotated)
            except Exception as e:
                print(f"\nError at frame {frame_num}: {e}")
                if out_video:
                    out_video.write(frame)  # Write original frame on error
        else:
            # No pose detected yet, write original frame
            if out_video:
                out_video.write(frame)

        print(f"\rProgress: {(frame_num + 1) / total_frames * 100:.1f}% | Poses: {poses_detected}", end='', flush=True)
        frame_num += 1

    print(f"\n\nDone! Detected pose in {poses_detected} frames.")

    try:
        cap.release()
    except Exception as e:
        print(f"Warning: Error releasing video capture: {e}")

    if out_video:
        try:
            out_video.release()
        except Exception as e:
            print(f"Warning: Error releasing video writer: {e}")

    # Export metrics JSON for browser-based graph rendering
    if args.output_video and all_metrics:
        try:
            metrics_json_path = Path(args.output_video).with_suffix('.json')
            metrics_data = analyzer.get_metrics_for_export()
            metrics_data['video_info'] = {
                'fps': fps,
                'total_frames': total_frames,
                'frames_analyzed': poses_detected,
                'width': frame_width,
                'height': frame_height,
                'source_name': analyzer.video_source_name,
                'timestamp': analyzer.video_timestamp.isoformat() if analyzer.video_timestamp else None,
                'processed_at': datetime.now().isoformat(),
            }
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_data, f)
            print(f"Metrics JSON saved to: {metrics_json_path}")
        except Exception as e:
            print(f"Warning: Could not save metrics JSON: {e}")

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

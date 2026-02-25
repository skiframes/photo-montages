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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Deque
from collections import deque

import cv2
import numpy as np

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
    body_inclination: float
    knee_angle_left: float
    knee_angle_right: float
    edge_angle_left: float
    edge_angle_right: float
    edge_symmetry_pct: float  # 100% = perfect symmetry
    fore_aft_left: float  # Renamed from tibia
    fore_aft_right: float
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
        self.fore_aft_left.append(m.fore_aft_left)
        self.fore_aft_right.append(m.fore_aft_right)


class YOLOPoseAnalyzer:
    """Analyzes ski racing video clips using YOLOv8-pose."""

    def __init__(self, slope_angle_deg: float = 0.0, model_size: str = 'l',
                 pitch_deg: float = None, ski_model_path: str = None):
        if not HAS_YOLO:
            raise ImportError("ultralytics required. Install with: pip install ultralytics")

        self.slope_angle_deg = slope_angle_deg
        self.pitch_deg = pitch_deg
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

        # Shoulder angle relative to slope
        shoulder_angle = angle_from_horizontal(left_shoulder[:2], right_shoulder[:2])
        shoulder_to_slope = shoulder_angle - self.slope_angle_deg
        while shoulder_to_slope > 90: shoulder_to_slope -= 180
        while shoulder_to_slope < -90: shoulder_to_slope += 180

        # Hip angle relative to slope
        hip_angle = angle_from_horizontal(left_hip[:2], right_hip[:2])
        hip_to_slope = hip_angle - self.slope_angle_deg
        while hip_to_slope > 90: hip_to_slope -= 180
        while hip_to_slope < -90: hip_to_slope += 180

        # Body angulation
        shoulder_mid = midpoint(left_shoulder[:2], right_shoulder[:2])
        hip_mid = midpoint(left_hip[:2], right_hip[:2])
        knee_mid = midpoint(left_knee[:2], right_knee[:2]) if left_knee and right_knee else None

        if knee_mid:
            body_angulation = angle_at_joint(shoulder_mid, hip_mid, knee_mid)
        else:
            body_angulation = 180.0

        # Body inclination
        ankle_mid = midpoint(left_ankle[:2], right_ankle[:2]) if left_ankle and right_ankle else None
        if ankle_mid:
            inclination_angle = angle_from_horizontal(ankle_mid, shoulder_mid)
            body_inclination = 90 - inclination_angle - self.slope_angle_deg
            while body_inclination > 90: body_inclination -= 180
            while body_inclination < -90: body_inclination += 180
        else:
            body_inclination = 0.0

        # Knee angles
        knee_angle_left = 180.0
        knee_angle_right = 180.0
        if left_hip and left_knee and left_ankle:
            knee_angle_left = angle_at_joint(left_hip[:2], left_knee[:2], left_ankle[:2])
        if right_hip and right_knee and right_ankle:
            knee_angle_right = angle_at_joint(right_hip[:2], right_knee[:2], right_ankle[:2])

        # Edge angles
        edge_angle_left = 0.0
        edge_angle_right = 0.0
        if left_knee and left_ankle:
            leg_angle_left = angle_from_vertical(left_knee[:2], left_ankle[:2])
            edge_angle_left = leg_angle_left - self.slope_angle_deg
        if right_knee and right_ankle:
            leg_angle_right = angle_from_vertical(right_knee[:2], right_ankle[:2])
            edge_angle_right = leg_angle_right - self.slope_angle_deg

        # Edge symmetry as percentage (100% = perfect)
        max_edge = max(abs(edge_angle_left), abs(edge_angle_right), 1)
        edge_diff = abs(edge_angle_left - edge_angle_right)
        edge_symmetry_pct = max(0, 100 - (edge_diff / max_edge * 100))

        # Fore/aft balance (tibia angle)
        fore_aft_left = 0.0
        fore_aft_right = 0.0
        if left_knee and left_ankle:
            fore_aft_left = angle_from_vertical(left_ankle[:2], left_knee[:2]) - self.slope_angle_deg
        if right_knee and right_ankle:
            fore_aft_right = angle_from_vertical(right_ankle[:2], right_knee[:2]) - self.slope_angle_deg

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
            fore_aft_left=round(fore_aft_left, 1),
            fore_aft_right=round(fore_aft_right, 1),
            confidence=round(avg_conf, 3)
        )

    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[PoseMetrics]]:
        results = self.model(frame, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None, None

        keypoints = results[0].keypoints.data
        if len(keypoints) == 0:
            return None, None

        person_kpts = keypoints[0].cpu().numpy()
        metrics = self.compute_metrics(person_kpts)

        if metrics:
            self.history.add(metrics)

        return person_kpts, metrics

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

            # Calculate ski angle from bounding box orientation
            # Ski is elongated - the longer dimension indicates direction
            if width > height:
                # Ski is more horizontal - angle from the box diagonal
                ski_angle = math.degrees(math.atan2(height, width))
            else:
                # Ski is more vertical
                ski_angle = 90 - math.degrees(math.atan2(width, height))

            detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'center': (cx, cy),
                'angle': ski_angle,
                'width': width,
                'height': height
            })

        # Match skis to ankles
        if la and detections:
            best_dist = float('inf')
            for det in detections:
                dist = math.sqrt((det['center'][0] - la[0])**2 + (det['center'][1] - la[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    left_ski = det.copy()
                    # Calculate fore/aft: how far forward ankle is relative to ski center
                    # Positive = ankle ahead of ski center (forward lean)
                    left_ski['fore_aft'] = la[0] - det['center'][0]  # In pixels

        if ra and detections:
            best_dist = float('inf')
            for det in detections:
                if left_ski and det['box'] == left_ski['box']:
                    continue
                dist = math.sqrt((det['center'][0] - ra[0])**2 + (det['center'][1] - ra[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    right_ski = det.copy()
                    right_ski['fore_aft'] = ra[0] - det['center'][0]

        return left_ski, right_ski

    def _draw_ski_rectangles(self, frame: np.ndarray, keypoints: np.ndarray, scale: float,
                              left_ski: Optional[dict] = None, right_ski: Optional[dict] = None) -> np.ndarray:
        """Draw ski detections using pre-detected ski data."""
        output = frame.copy()

        # Draw left ski (orange)
        if left_ski:
            x1, y1, x2, y2 = left_ski['box']
            color = (0, 165, 255)  # Orange
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        # Draw right ski (blue)
        if right_ski:
            x1, y1, x2, y2 = right_ski['box']
            color = (255, 150, 50)  # Blue
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

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

            # Fall line direction
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
        draw_line(self.history.angulation, (0, 255, 255), 90, 180)  # Yellow - angulation

        # Legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 * scale
        leg_y = graph_y + int(15 * scale)
        cv2.putText(output, "Symmetry", (margin + 5, leg_y), font, font_scale, (0, 255, 0), 1)
        cv2.putText(output, "Inclin", (margin + int(80 * scale), leg_y), font, font_scale, (255, 180, 100), 1)
        cv2.putText(output, "Angul", (margin + int(140 * scale), leg_y), font, font_scale, (0, 255, 255), 1)

        return output

    def _draw_extended_lines(self, frame: np.ndarray, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Draw extended lines through shoulders and hips."""
        output = frame.copy()
        line_ext = int(100 * scale)
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

        # Extended shoulder line (magenta)
        if ls and rs:
            dx = rs[0] - ls[0]
            dy = rs[1] - ls[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                ext_l = (int(ls[0] - ux * line_ext), int(ls[1] - uy * line_ext))
                ext_r = (int(rs[0] + ux * line_ext), int(rs[1] + uy * line_ext))
                cv2.line(output, ext_l, ext_r, (255, 0, 255), thickness, cv2.LINE_AA)

        # Extended hip line (cyan)
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

    def draw_overlay(self, frame: np.ndarray, keypoints: np.ndarray,
                     metrics: Optional[PoseMetrics] = None,
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
        output = self._draw_ski_rectangles(output, keypoints, scale)

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

            # Compact format with all metrics - shoulder/hip as percentages
            metrics_parts = [
                (f"Sh:{metrics.shoulder_alignment_pct:.0f}%", pct_color(metrics.shoulder_alignment_pct)),
                (f"Hip:{metrics.hip_alignment_pct:.0f}%", pct_color(metrics.hip_alignment_pct)),
                (f"Edge L{metrics.edge_angle_left:+.0f} R{metrics.edge_angle_right:+.0f}", (100, 255, 255)),
                (f"Sym:{metrics.edge_symmetry_pct:.0f}%", pct_color(metrics.edge_symmetry_pct)),
                (f"F/A L{metrics.fore_aft_left:+.0f} R{metrics.fore_aft_right:+.0f}", (255, 200, 100)),
                (f"Ang:{metrics.body_angulation:.0f}", (0, 255, 255)),
                (f"Incl:{metrics.body_inclination:+.0f}", (255, 180, 100)),
            ]

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

    slope_angle = args.slope_angle
    pitch_deg = None

    if slope_angle is None and args.calibration:
        try:
            from slope_calculator import compute_slope_from_pixels_only
            print(f"Computing slope from calibration: {args.calibration}")
            result = compute_slope_from_pixels_only(args.calibration)
            slope_angle = result['slope_angle_deg']
            print(f"Slope: {result['direction_label']}")
        except Exception as e:
            print(f"Warning: {e}")
            slope_angle = 0.0
    elif slope_angle is None:
        slope_angle = 0.0

    print(f"\nVideo: {args.video_path}")
    print(f"Slope angle: {slope_angle}°")

    analyzer = YOLOPoseAnalyzer(slope_angle_deg=slope_angle, model_size=args.model, pitch_deg=pitch_deg)

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
            keypoints, metrics = analyzer.analyze_frame(frame)

            if keypoints is not None:
                poses_detected += 1
                if metrics:
                    all_metrics.append(metrics)
                annotated = analyzer.draw_overlay(frame, keypoints, metrics)
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
        print(f"  Fore/Aft L: {sum(m.fore_aft_left for m in all_metrics)/len(all_metrics):.1f}°")
        print(f"  Fore/Aft R: {sum(m.fore_aft_right for m in all_metrics)/len(all_metrics):.1f}°")


if __name__ == "__main__":
    main()

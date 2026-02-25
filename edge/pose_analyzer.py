#!/usr/bin/env python3
"""
Pose analysis module for ski racing video clips.

Uses MediaPipe Pose to detect body keypoints and calculate biomechanics metrics:
- Shoulder angle relative to slope
- Hip angle relative to slope
- Body angulation (angle between torso and legs)
- Body inclination (overall lean toward slope)
"""

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# MediaPipe Pose keypoint indices
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class PoseKeypoints:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass
class PoseMetrics:
    """Biomechanics metrics for a single frame."""
    shoulder_angle_to_slope: float  # Angle of shoulder line relative to slope (degrees)
    hip_angle_to_slope: float       # Angle of hip line relative to slope (degrees)
    body_angulation: float          # Angle between torso and legs (degrees)
    body_inclination: float         # Overall body lean toward slope (degrees)
    confidence: float               # Average landmark visibility/confidence


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_num: int
    timestamp_sec: float
    keypoints: Optional[dict]  # Raw keypoint coordinates
    metrics: Optional[PoseMetrics]
    pose_detected: bool


@dataclass
class VideoAnalysisResult:
    """Complete analysis result for a video."""
    run_id: str
    video_path: str
    slope_angle_deg: float
    total_frames: int
    analyzed_frames: int
    frames: list  # List of FrameAnalysis
    summary: dict  # Average metrics


def angle_from_horizontal(p1: tuple, p2: tuple) -> float:
    """
    Calculate angle of line from p1 to p2 relative to horizontal.
    Returns angle in degrees (-90 to 90).

    Args:
        p1: (x, y) coordinates of first point
        p2: (x, y) coordinates of second point
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if abs(dx) < 1e-6:
        return 90.0 if dy > 0 else -90.0

    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def midpoint(p1: tuple, p2: tuple) -> tuple:
    """Calculate midpoint between two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def angle_between_lines(line1_angle: float, line2_angle: float) -> float:
    """
    Calculate angle between two lines given their angles from horizontal.
    Returns absolute angle difference (0 to 180).
    """
    diff = abs(line1_angle - line2_angle)
    if diff > 180:
        diff = 360 - diff
    return diff


class PoseAnalyzer:
    """
    Analyzes ski racing video clips to extract body position metrics.

    Uses MediaPipe Pose for keypoint detection and calculates angles
    relative to a calibrated slope angle.
    """

    def __init__(self, slope_angle_deg: float = 0.0):
        """
        Initialize the pose analyzer.

        Args:
            slope_angle_deg: Slope angle in degrees (from horizontal, positive = downhill right)
        """
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe is required for pose analysis. Install with: pip install mediapipe")

        self.slope_angle_deg = slope_angle_deg

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()

    def extract_keypoints(self, landmarks) -> dict:
        """
        Extract relevant keypoints from MediaPipe pose landmarks.

        Returns dict with keypoint names and their (x, y, visibility) values.
        Coordinates are normalized (0-1) relative to image dimensions.
        """
        keypoints = {}

        kp_map = {
            'left_shoulder': PoseKeypoints.LEFT_SHOULDER,
            'right_shoulder': PoseKeypoints.RIGHT_SHOULDER,
            'left_hip': PoseKeypoints.LEFT_HIP,
            'right_hip': PoseKeypoints.RIGHT_HIP,
            'left_knee': PoseKeypoints.LEFT_KNEE,
            'right_knee': PoseKeypoints.RIGHT_KNEE,
            'left_ankle': PoseKeypoints.LEFT_ANKLE,
            'right_ankle': PoseKeypoints.RIGHT_ANKLE,
        }

        for name, idx in kp_map.items():
            lm = landmarks.landmark[idx]
            keypoints[name] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }

        return keypoints

    def compute_metrics(self, keypoints: dict) -> PoseMetrics:
        """
        Compute biomechanics metrics from extracted keypoints.

        All angles are relative to the configured slope angle.
        """
        # Get point coordinates (x, y)
        # Note: In image coordinates, y increases downward
        left_shoulder = (keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y'])
        right_shoulder = (keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y'])
        left_hip = (keypoints['left_hip']['x'], keypoints['left_hip']['y'])
        right_hip = (keypoints['right_hip']['x'], keypoints['right_hip']['y'])
        left_knee = (keypoints['left_knee']['x'], keypoints['left_knee']['y'])
        right_knee = (keypoints['right_knee']['x'], keypoints['right_knee']['y'])
        left_ankle = (keypoints['left_ankle']['x'], keypoints['left_ankle']['y'])
        right_ankle = (keypoints['right_ankle']['x'], keypoints['right_ankle']['y'])

        # Midpoints
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)
        ankle_mid = midpoint(left_ankle, right_ankle)

        # 1. Shoulder angle relative to slope
        shoulder_angle = angle_from_horizontal(left_shoulder, right_shoulder)
        shoulder_angle_to_slope = shoulder_angle - self.slope_angle_deg
        # Normalize to -90 to 90
        while shoulder_angle_to_slope > 90:
            shoulder_angle_to_slope -= 180
        while shoulder_angle_to_slope < -90:
            shoulder_angle_to_slope += 180

        # 2. Hip angle relative to slope
        hip_angle = angle_from_horizontal(left_hip, right_hip)
        hip_angle_to_slope = hip_angle - self.slope_angle_deg
        while hip_angle_to_slope > 90:
            hip_angle_to_slope -= 180
        while hip_angle_to_slope < -90:
            hip_angle_to_slope += 180

        # 3. Body angulation (angle at hip between torso and thigh)
        # This measures the "hip angle" in skiing terms - how much the hips
        # are angled relative to the upper body
        torso_angle = angle_from_horizontal(hip_mid, shoulder_mid)
        thigh_angle = angle_from_horizontal(hip_mid, knee_mid)
        body_angulation = abs(angle_between_lines(torso_angle, thigh_angle))
        # Angulation is typically measured from the inside leg in ski racing
        # A value of ~160-180 means relatively straight, ~120-150 shows angulation

        # 4. Body inclination (overall lean from vertical toward slope)
        # Measure angle of the line from ankle to shoulder relative to vertical
        inclination_angle = angle_from_horizontal(ankle_mid, shoulder_mid)
        # Convert to angle from vertical (90 degrees)
        body_inclination = 90 - inclination_angle - self.slope_angle_deg
        # Positive = leaning into hill, Negative = leaning away
        while body_inclination > 90:
            body_inclination -= 180
        while body_inclination < -90:
            body_inclination += 180

        # Calculate average visibility as confidence score
        visibilities = [
            keypoints['left_shoulder']['visibility'],
            keypoints['right_shoulder']['visibility'],
            keypoints['left_hip']['visibility'],
            keypoints['right_hip']['visibility'],
            keypoints['left_knee']['visibility'],
            keypoints['right_knee']['visibility'],
            keypoints['left_ankle']['visibility'],
            keypoints['right_ankle']['visibility'],
        ]
        confidence = sum(visibilities) / len(visibilities)

        return PoseMetrics(
            shoulder_angle_to_slope=round(shoulder_angle_to_slope, 1),
            hip_angle_to_slope=round(hip_angle_to_slope, 1),
            body_angulation=round(body_angulation, 1),
            body_inclination=round(body_inclination, 1),
            confidence=round(confidence, 3)
        )

    def analyze_frame(self, frame: np.ndarray) -> FrameAnalysis:
        """
        Analyze a single video frame.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            FrameAnalysis with keypoints and metrics (or None if pose not detected)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return FrameAnalysis(
                frame_num=0,
                timestamp_sec=0.0,
                keypoints=None,
                metrics=None,
                pose_detected=False
            )

        # Extract keypoints
        keypoints = self.extract_keypoints(results.pose_landmarks)

        # Compute metrics
        metrics = self.compute_metrics(keypoints)

        return FrameAnalysis(
            frame_num=0,
            timestamp_sec=0.0,
            keypoints=keypoints,
            metrics=metrics,
            pose_detected=True
        )

    def analyze_video(
        self,
        video_path: str,
        sample_rate: int = 3,
        run_id: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> VideoAnalysisResult:
        """
        Analyze an entire video clip frame by frame.

        Args:
            video_path: Path to the video file
            sample_rate: Analyze every Nth frame (1 = every frame, 3 = every 3rd frame)
            run_id: Identifier for this run (defaults to filename)
            progress_callback: Optional callback(current_frame, total_frames)

        Returns:
            VideoAnalysisResult with all frame analyses and summary statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if run_id is None:
            run_id = video_path.stem

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_results = []
        frame_num = 0
        analyzed_count = 0

        # Accumulators for summary statistics
        shoulder_angles = []
        hip_angles = []
        angulations = []
        inclinations = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample rate check
            if frame_num % sample_rate == 0:
                timestamp_sec = frame_num / fps if fps > 0 else 0.0

                analysis = self.analyze_frame(frame)
                analysis.frame_num = frame_num
                analysis.timestamp_sec = round(timestamp_sec, 3)

                # Convert to dict for JSON serialization
                frame_result = {
                    'frame_num': analysis.frame_num,
                    'timestamp_sec': analysis.timestamp_sec,
                    'pose_detected': analysis.pose_detected,
                    'keypoints': analysis.keypoints,
                    'metrics': asdict(analysis.metrics) if analysis.metrics else None
                }
                frames_results.append(frame_result)

                if analysis.pose_detected and analysis.metrics:
                    shoulder_angles.append(analysis.metrics.shoulder_angle_to_slope)
                    hip_angles.append(analysis.metrics.hip_angle_to_slope)
                    angulations.append(analysis.metrics.body_angulation)
                    inclinations.append(analysis.metrics.body_inclination)

                analyzed_count += 1

            frame_num += 1

            if progress_callback:
                progress_callback(frame_num, total_frames)

        cap.release()

        # Calculate summary statistics
        def safe_avg(values):
            return round(sum(values) / len(values), 1) if values else None

        def safe_min(values):
            return round(min(values), 1) if values else None

        def safe_max(values):
            return round(max(values), 1) if values else None

        summary = {
            'avg_shoulder_angle': safe_avg(shoulder_angles),
            'avg_hip_angle': safe_avg(hip_angles),
            'avg_angulation': safe_avg(angulations),
            'avg_inclination': safe_avg(inclinations),
            'min_shoulder_angle': safe_min(shoulder_angles),
            'max_shoulder_angle': safe_max(shoulder_angles),
            'min_hip_angle': safe_min(hip_angles),
            'max_hip_angle': safe_max(hip_angles),
            'min_angulation': safe_min(angulations),
            'max_angulation': safe_max(angulations),
            'min_inclination': safe_min(inclinations),
            'max_inclination': safe_max(inclinations),
            'frames_with_pose': len(shoulder_angles),
            'detection_rate': round(len(shoulder_angles) / analyzed_count, 3) if analyzed_count > 0 else 0
        }

        return VideoAnalysisResult(
            run_id=run_id,
            video_path=str(video_path),
            slope_angle_deg=self.slope_angle_deg,
            total_frames=total_frames,
            analyzed_frames=analyzed_count,
            frames=frames_results,
            summary=summary
        )

    def to_dict(self, result: VideoAnalysisResult) -> dict:
        """Convert VideoAnalysisResult to JSON-serializable dict."""
        return {
            'run_id': result.run_id,
            'video_path': result.video_path,
            'slope_angle_deg': result.slope_angle_deg,
            'total_frames': result.total_frames,
            'analyzed_frames': result.analyzed_frames,
            'frames': result.frames,
            'summary': result.summary
        }


def draw_pose_overlay(frame: np.ndarray, keypoints: dict, metrics: PoseMetrics = None) -> np.ndarray:
    """
    Draw pose keypoints and connections on a frame for visualization.

    Args:
        frame: BGR image (OpenCV format)
        keypoints: Dict of keypoint coordinates (normalized 0-1)
        metrics: Optional metrics to display on frame

    Returns:
        Frame with overlay drawn
    """
    h, w = frame.shape[:2]
    output = frame.copy()

    # Convert normalized coordinates to pixel coordinates
    def to_pixel(kp_name):
        kp = keypoints[kp_name]
        return (int(kp['x'] * w), int(kp['y'] * h))

    # Draw connections
    connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle'),
    ]

    for start, end in connections:
        p1 = to_pixel(start)
        p2 = to_pixel(end)
        cv2.line(output, p1, p2, (0, 255, 0), 2)

    # Draw keypoints
    for name in keypoints:
        pt = to_pixel(name)
        visibility = keypoints[name]['visibility']
        color = (0, 255, 0) if visibility > 0.5 else (0, 165, 255)
        cv2.circle(output, pt, 5, color, -1)
        cv2.circle(output, pt, 6, (255, 255, 255), 1)

    # Draw metrics text
    if metrics:
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        texts = [
            f"Shoulder: {metrics.shoulder_angle_to_slope:.1f}°",
            f"Hip: {metrics.hip_angle_to_slope:.1f}°",
            f"Angulation: {metrics.body_angulation:.1f}°",
            f"Inclination: {metrics.body_inclination:.1f}°",
            f"Confidence: {metrics.confidence:.0%}",
        ]

        for text in texts:
            # Draw background rectangle
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(output, (8, y_offset - text_h - 4), (12 + text_w, y_offset + 4), bg_color, -1)
            cv2.putText(output, text, (10, y_offset), font, font_scale, color, 1, cv2.LINE_AA)
            y_offset += 25

    return output


# Example usage and testing
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ski racing video for body position metrics')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('slope_angle', nargs='?', type=float, default=0.0, help='Slope angle in degrees (default: 0)')
    parser.add_argument('--output-frames', '-o', metavar='DIR', help='Save annotated frames to directory')
    parser.add_argument('--output-video', '-v', metavar='FILE', help='Save annotated video to file')
    parser.add_argument('--sample-rate', '-s', type=int, default=3, help='Analyze every Nth frame (default: 3)')
    parser.add_argument('--all-frames', '-a', action='store_true', help='Save all frames, not just those with pose detected')

    args = parser.parse_args()

    video_path = args.video_path
    slope_angle = args.slope_angle
    output_frames_dir = args.output_frames
    output_video_path = args.output_video
    sample_rate = args.sample_rate
    save_all_frames = args.all_frames

    print(f"Analyzing video: {video_path}")
    print(f"Slope angle: {slope_angle}°")

    # Create output directory if needed
    if output_frames_dir:
        output_dir = Path(output_frames_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving annotated frames to: {output_dir}")

    analyzer = PoseAnalyzer(slope_angle_deg=slope_angle)

    # For saving frames/video, we need to process differently
    if output_frames_dir or output_video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if needed
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for macOS
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps / sample_rate, (width, height))
            if not video_writer.isOpened():
                print(f"Warning: Could not open video writer with avc1, trying MJPG")
                output_video_path = output_video_path.replace('.mp4', '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps / sample_rate, (width, height))
            print(f"Saving annotated video to: {output_video_path}")

        frame_num = 0
        saved_count = 0
        pose_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_rate == 0:
                timestamp_sec = frame_num / fps if fps > 0 else 0.0
                analysis = analyzer.analyze_frame(frame)

                # Draw overlay if pose detected or saving all frames
                if analysis.pose_detected or save_all_frames:
                    if analysis.pose_detected and analysis.keypoints:
                        annotated = draw_pose_overlay(frame, analysis.keypoints, analysis.metrics)
                        pose_count += 1
                    else:
                        annotated = frame.copy()
                        # Add "No pose detected" text
                        cv2.putText(annotated, "No pose detected", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Add frame info
                    cv2.putText(annotated, f"Frame {frame_num} | {timestamp_sec:.2f}s",
                                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Save frame image
                    if output_frames_dir:
                        frame_path = output_dir / f"frame_{frame_num:05d}.jpg"
                        cv2.imwrite(str(frame_path), annotated)
                        saved_count += 1

                    # Write to video
                    if video_writer:
                        video_writer.write(annotated)

                pct = (frame_num + 1) / total_frames * 100
                print(f"\rProgress: {pct:.1f}% | Saved: {saved_count} | Poses: {pose_count}", end='', flush=True)

            frame_num += 1

        cap.release()
        if video_writer:
            video_writer.release()

        print(f"\n\nDone! Saved {saved_count} annotated frames, {pose_count} with pose detected.")

    else:
        # Standard analysis without saving frames
        def progress(current, total):
            pct = current / total * 100
            print(f"\rProgress: {pct:.1f}% ({current}/{total})", end='', flush=True)

        result = analyzer.analyze_video(video_path, sample_rate=sample_rate, progress_callback=progress)
        print()

        print(f"\nResults:")
        print(f"  Total frames: {result.total_frames}")
        print(f"  Analyzed frames: {result.analyzed_frames}")
        print(f"  Frames with pose detected: {result.summary['frames_with_pose']}")
        print(f"  Detection rate: {result.summary['detection_rate']:.0%}")
        print()
        print(f"Summary (averages):")
        print(f"  Shoulder angle to slope: {result.summary['avg_shoulder_angle']}°")
        print(f"  Hip angle to slope: {result.summary['avg_hip_angle']}°")
        print(f"  Body angulation: {result.summary['avg_angulation']}°")
        print(f"  Body inclination: {result.summary['avg_inclination']}°")

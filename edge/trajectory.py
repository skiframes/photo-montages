#!/usr/bin/env python3
"""
Trajectory Visualizer (TR) - Visualize skier trajectory on video frames.

Implements a simplified version of SkiTraVis (Dunnhofer et al., CVPRW 2023):
1. Track skier via bounding box (YOLOv8 or provided detections)
2. Estimate camera motion via keypoint matching (Shi-Tomasi + Lucas-Kanade)
3. Accumulate trajectory points with homography correction
4. Draw trajectory on each frame

For fixed cameras (our typical setup), the homography is near-identity,
so trajectory visualization is more accurate than the broadcast scenario
in the paper. Falls back to raw pixel accumulation when homography fails.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---- Configuration ----

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    # Tracker settings
    use_yolo: bool = True             # Use YOLOv8 for skier detection (vs manual bbox)
    yolo_model: str = 'yolov8l.pt'    # YOLOv8 model for detection
    yolo_conf: float = 0.3            # Detection confidence threshold
    person_class_id: int = 0          # COCO class ID for 'person'

    # Keypoint matching (camera motion estimation)
    shi_tomasi_max_corners: int = 200
    shi_tomasi_quality: float = 0.01
    shi_tomasi_min_distance: int = 10
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    ransac_reproj_threshold: float = 5.0
    min_keypoints_for_homography: int = 8

    # Trajectory visualization
    line_color: Tuple[int, int, int] = (0, 180, 255)    # Orange (BGR)
    line_thickness: int = 3
    dot_radius: int = 5
    dot_color: Tuple[int, int, int] = (0, 0, 255)       # Red (BGR)
    current_dot_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR)
    current_dot_radius: int = 7
    fade_alpha: float = 0.7           # Opacity of trajectory line
    show_timestamps: bool = True      # Show time labels along trajectory
    timestamp_interval: int = 10      # Show timestamp every N points

    # Skier bbox exclusion padding (pixels) for keypoint detection
    bbox_padding: int = 20


# ---- Core trajectory computation ----

@dataclass
class TrajectoryPoint:
    """Single point in the trajectory."""
    x: float
    y: float
    frame_idx: int
    timestamp_sec: float = 0.0


def detect_skier_bbox(frame: np.ndarray, model=None,
                      conf: float = 0.3, person_class: int = 0) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the largest person bounding box in the frame using YOLOv8.

    Returns (x, y, w, h) or None if no person detected.
    """
    if model is None:
        return None

    results = model(frame, verbose=False, conf=conf)
    if not results or len(results) == 0:
        return None

    # Find largest person detection by area
    best_bbox = None
    best_area = 0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            if cls != person_class:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    return best_bbox


def detect_skier_bbox_motion(prev_gray: np.ndarray, curr_gray: np.ndarray,
                              threshold: int = 25,
                              min_area: int = 500) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect skier bounding box using frame differencing (no ML model needed).

    Finds the largest moving region by absolute difference between frames.
    Works well when there's a single skier against a static snow background.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        threshold: Pixel difference threshold
        min_area: Minimum contour area to be considered a skier

    Returns:
        (x, y, w, h) of the largest moving region, or None
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


def extract_foot_position(bbox: Tuple[int, int, int, int], k: float = 0.9) -> Tuple[float, float]:
    """
    Extract approximate foot position from bounding box.

    Following the paper: foot position is at horizontal center,
    k * height from top of bbox (k=0.9 approximates snow contact).

    Args:
        bbox: (x, y, w, h)
        k: Vertical position factor (0=top, 1=bottom of bbox)

    Returns:
        (x, y) foot position
    """
    x, y, w, h = bbox
    foot_x = x + w * 0.5
    foot_y = y + h * k
    return (foot_x, foot_y)


def compute_homography(prev_gray: np.ndarray, curr_gray: np.ndarray,
                       skier_bbox: Optional[Tuple[int, int, int, int]] = None,
                       config: TrajectoryConfig = None) -> Optional[np.ndarray]:
    """
    Compute homography between two frames using Shi-Tomasi + Lucas-Kanade.

    Detects static keypoints, tracks them, then estimates homography via RANSAC.
    Excludes keypoints inside the skier bounding box (non-static object).

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        skier_bbox: Optional (x, y, w, h) of skier to exclude
        config: Trajectory configuration

    Returns:
        3x3 homography matrix or None if estimation fails
    """
    if config is None:
        config = TrajectoryConfig()

    h, w = prev_gray.shape[:2]

    # Create mask to exclude skier region from keypoint detection
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if skier_bbox is not None:
        bx, by, bw, bh = skier_bbox
        pad = config.bbox_padding
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)
        mask[y1:y2, x1:x2] = 0

    # Detect keypoints in previous frame (Shi-Tomasi corners)
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=config.shi_tomasi_max_corners,
        qualityLevel=config.shi_tomasi_quality,
        minDistance=config.shi_tomasi_min_distance,
        mask=mask,
    )

    if corners is None or len(corners) < config.min_keypoints_for_homography:
        return None

    # Track keypoints to current frame (Lucas-Kanade optical flow)
    lk_params = dict(
        winSize=config.lk_win_size,
        maxLevel=config.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, corners, None, **lk_params
    )

    if next_pts is None:
        return None

    # Filter by status (successfully tracked)
    good_mask = status.ravel() == 1
    src_pts = corners[good_mask]
    dst_pts = next_pts[good_mask]

    if len(src_pts) < config.min_keypoints_for_homography:
        return None

    # Compute homography via RANSAC
    H, inlier_mask = cv2.findHomography(
        src_pts, dst_pts,
        cv2.RANSAC,
        config.ransac_reproj_threshold,
    )

    if H is None:
        return None

    # Validate: check if homography is reasonable (not degenerate)
    # Determinant should be close to 1 for a valid perspective transform
    det = np.linalg.det(H)
    if det < 0.1 or det > 10.0:
        logger.debug(f"Degenerate homography (det={det:.3f}), skipping")
        return None

    return H


def transform_points(points: List[Tuple[float, float]],
                     H: np.ndarray) -> List[Tuple[float, float]]:
    """
    Transform 2D points using a homography matrix.

    Args:
        points: List of (x, y) tuples
        H: 3x3 homography matrix

    Returns:
        List of transformed (x, y) tuples
    """
    if not points or H is None:
        return points

    # Convert to homogeneous coordinates
    pts = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
    pts_h = np.column_stack([pts, np.ones(len(pts))])

    # Apply homography
    transformed = (H @ pts_h.T).T

    # Convert back from homogeneous
    result = []
    for p in transformed:
        if abs(p[2]) > 1e-8:
            result.append((p[0] / p[2], p[1] / p[2]))
        else:
            result.append((p[0], p[1]))

    return result


def draw_trajectory(frame: np.ndarray,
                    trajectory: List[TrajectoryPoint],
                    config: TrajectoryConfig = None,
                    source_fps: float = 30.0) -> np.ndarray:
    """
    Draw trajectory overlay on a frame.

    Args:
        frame: BGR frame to draw on
        trajectory: List of trajectory points in current frame coordinates
        config: Visualization configuration
        source_fps: Source video FPS for timestamp calculation

    Returns:
        Frame with trajectory overlay
    """
    if config is None:
        config = TrajectoryConfig()

    if len(trajectory) < 2:
        return frame

    result = frame.copy()
    h, w = result.shape[:2]

    # Create trajectory overlay on separate layer for alpha blending
    overlay = result.copy()

    # Draw trajectory line segments
    pts = [(int(p.x), int(p.y)) for p in trajectory]

    # Clip points to frame bounds
    pts_clipped = [(max(0, min(w-1, x)), max(0, min(h-1, y))) for x, y in pts]

    # Draw polyline
    if len(pts_clipped) >= 2:
        pts_array = np.array(pts_clipped, dtype=np.int32)
        cv2.polylines(overlay, [pts_array], isClosed=False,
                      color=config.line_color, thickness=config.line_thickness,
                      lineType=cv2.LINE_AA)

    # Draw dots at each point
    for i, pt in enumerate(pts_clipped):
        if i == len(pts_clipped) - 1:
            # Current position: larger green dot
            cv2.circle(overlay, pt, config.current_dot_radius,
                       config.current_dot_color, -1, cv2.LINE_AA)
        else:
            # Historical position: small red dot
            cv2.circle(overlay, pt, config.dot_radius,
                       config.dot_color, -1, cv2.LINE_AA)

    # Add time labels along trajectory
    if config.show_timestamps and len(trajectory) > 1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, w / 3000)
        thickness = max(1, int(w / 1500))

        for i, tp in enumerate(trajectory):
            if i == 0 or (i % config.timestamp_interval == 0) or i == len(trajectory) - 1:
                time_str = f"{tp.timestamp_sec:.1f}s"
                pt = pts_clipped[i]
                # Offset label slightly to avoid overlapping the line
                label_x = pt[0] + config.dot_radius + 3
                label_y = pt[1] - config.dot_radius - 2

                # Background for readability
                (tw, th), _ = cv2.getTextSize(time_str, font, font_scale, thickness)
                cv2.rectangle(overlay,
                              (label_x - 1, label_y - th - 1),
                              (label_x + tw + 1, label_y + 2),
                              (0, 0, 0), -1)
                cv2.putText(overlay, time_str, (label_x, label_y),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Alpha blend
    cv2.addWeighted(overlay, config.fade_alpha, result, 1 - config.fade_alpha, 0, result)

    return result


# ---- Main pipeline ----

def generate_trajectory_video(
    frames: List[np.ndarray],
    output_path: str,
    source_fps: float = 30.0,
    crop_region: Optional[Dict] = None,
    config: Optional[TrajectoryConfig] = None,
    sample_fps: float = 10.0,
) -> Optional[str]:
    """
    Generate a video with trajectory overlay from run frames.

    This is the main entry point. Takes raw frames from a detected run,
    tracks the skier, computes trajectory, and outputs an annotated video.

    Args:
        frames: List of BGR frames from the run (at source_fps)
        output_path: Where to write the output MP4
        source_fps: FPS of the source frames
        crop_region: Optional dict {x, y, w, h} to crop frames
        config: Trajectory configuration (uses defaults if None)
        sample_fps: FPS to sample for trajectory computation
                    (lower = faster processing, higher = smoother trajectory)

    Returns:
        Path to generated video, or None on failure
    """
    if not frames or len(frames) < 5:
        logger.warning("Too few frames for trajectory generation")
        return None

    if config is None:
        config = TrajectoryConfig()

    n_frames = len(frames)
    logger.info(f"Generating trajectory video: {n_frames} frames @ {source_fps:.0f}fps -> {output_path}")

    # Apply crop if specified
    if crop_region:
        cx = crop_region.get('x', 0)
        cy = crop_region.get('y', 0)
        cw = crop_region.get('w', frames[0].shape[1])
        ch = crop_region.get('h', frames[0].shape[0])
        cropped_frames = [f[cy:cy+ch, cx:cx+cw] for f in frames]
    else:
        cropped_frames = frames

    frame_h, frame_w = cropped_frames[0].shape[:2]

    # Ensure even dimensions for H.264
    out_w = frame_w if frame_w % 2 == 0 else frame_w - 1
    out_h = frame_h if frame_h % 2 == 0 else frame_h - 1

    # Load YOLO model if configured
    yolo_model = None
    if config.use_yolo:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(config.yolo_model)
            logger.info(f"YOLO model loaded: {config.yolo_model}")
        except ImportError:
            logger.warning("ultralytics not installed, falling back to no detection")
            config.use_yolo = False
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            config.use_yolo = False

    # Compute sampling interval for trajectory computation
    # Process every Nth frame for speed, but render trajectory on ALL frames
    sample_interval = max(1, int(source_fps / sample_fps))
    use_motion_fallback = not config.use_yolo or yolo_model is None

    if use_motion_fallback:
        print(f"  TR: Using motion-based detection (no YOLO)")
    else:
        print(f"  TR: Using YOLO detection ({config.yolo_model})")

    print(f"  TR: Computing trajectory ({n_frames} frames, sample every {sample_interval})...")

    # Single pass: compute trajectory and store per-frame snapshots for rendering
    trajectory_snapshots: Dict[int, List[TrajectoryPoint]] = {}
    trajectory_live: List[TrajectoryPoint] = []
    prev_gray = None
    prev_bbox = None

    for i in range(0, n_frames, sample_interval):
        frame = cropped_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = i / source_fps

        # Detect skier — YOLO if available, otherwise motion-based
        bbox = None
        if not use_motion_fallback:
            bbox = detect_skier_bbox(frame, yolo_model, config.yolo_conf, config.person_class_id)

        if bbox is None and prev_gray is not None:
            bbox = detect_skier_bbox_motion(prev_gray, gray)

        if bbox is None and prev_bbox is not None:
            # Last resort: reuse previous bbox (will drift)
            bbox = prev_bbox

        if bbox is None:
            # Store current trajectory state even when no detection
            trajectory_snapshots[i] = [TrajectoryPoint(x=tp.x, y=tp.y,
                                                        frame_idx=tp.frame_idx,
                                                        timestamp_sec=tp.timestamp_sec)
                                        for tp in trajectory_live]
            prev_gray = gray
            continue

        foot_x, foot_y = extract_foot_position(bbox)

        # Transform existing trajectory points to current frame coordinates
        if prev_gray is not None and len(trajectory_live) > 0:
            H = compute_homography(prev_gray, gray, skier_bbox=bbox, config=config)
            if H is not None:
                old_pts = [(tp.x, tp.y) for tp in trajectory_live]
                new_pts = transform_points(old_pts, H)
                for j, tp in enumerate(trajectory_live):
                    tp.x, tp.y = new_pts[j]

        trajectory_live.append(TrajectoryPoint(
            x=foot_x, y=foot_y, frame_idx=i, timestamp_sec=timestamp
        ))

        # Store snapshot (deep copy) for rendering
        trajectory_snapshots[i] = [TrajectoryPoint(x=tp.x, y=tp.y,
                                                    frame_idx=tp.frame_idx,
                                                    timestamp_sec=tp.timestamp_sec)
                                    for tp in trajectory_live]

        prev_gray = gray
        prev_bbox = bbox

    if len(trajectory_live) < 2:
        logger.warning("Insufficient trajectory points computed")
        return None

    print(f"  TR: {len(trajectory_live)} trajectory points computed")

    # ---- Phase 3: Render video ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use ffmpeg for encoding (same approach as video_clip.py)
    import subprocess

    # Detect encoder
    encoder = 'libx264'
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        if 'h264_nvenc' in result.stdout:
            test = subprocess.run(
                ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1',
                 '-c:v', 'h264_nvenc', '-f', 'null', '-'],
                capture_output=True, timeout=10
            )
            if test.returncode == 0:
                encoder = 'h264_nvenc'
    except Exception:
        pass

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(source_fps),
        '-i', '-',
        '-c:v', encoder,
    ]

    if encoder == 'libx264':
        cmd.extend(['-crf', '23', '-preset', 'fast'])
    elif encoder == 'h264_nvenc':
        cmd.extend(['-rc', 'vbr', '-cq', '23', '-preset', 'medium'])

    cmd.extend([
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path,
    ])

    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Get sorted snapshot frame indices
    snapshot_indices = sorted(trajectory_snapshots.keys())

    try:
        for i in range(n_frames):
            frame = cropped_frames[i]

            # Find the latest trajectory snapshot at or before this frame
            current_traj = None
            for si in snapshot_indices:
                if si <= i:
                    current_traj = trajectory_snapshots[si]
                else:
                    break

            # Draw trajectory if we have one
            if current_traj and len(current_traj) >= 2:
                frame = draw_trajectory(frame, current_traj, config, source_fps)

            # Ensure dimensions match
            frame = frame[:out_h, :out_w]
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait(timeout=120)

        if process.returncode != 0:
            stderr = process.stderr.read().decode('utf-8', errors='replace')
            logger.error(f"ffmpeg failed: {stderr[-500:]}")
            return None

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  TR: Video generated: {output_path} ({size_kb:.0f} KB)")
            return output_path
        else:
            logger.error(f"Output missing: {output_path}")
            return None

    except Exception as e:
        logger.error(f"Trajectory video generation failed: {e}")
        if process.poll() is None:
            process.kill()
        return None


def generate_trajectory_on_montage(
    frames: List[np.ndarray],
    montage_image: np.ndarray,
    source_fps: float = 30.0,
    crop_region: Optional[Dict] = None,
    config: Optional[TrajectoryConfig] = None,
    sample_fps: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Draw trajectory line on an existing montage (stop-motion composite).

    Takes the same frames used for montage generation and computes
    the trajectory, then overlays it on the composite image.

    Args:
        frames: Original run frames
        montage_image: The composite montage image to draw on
        source_fps: FPS of source frames
        crop_region: Crop region used for the montage
        config: Trajectory configuration
        sample_fps: Sampling rate for trajectory computation

    Returns:
        Montage image with trajectory overlay, or None on failure
    """
    if config is None:
        config = TrajectoryConfig()

    if not frames or len(frames) < 5:
        return None

    # Apply crop to get frames in montage coordinate space
    if crop_region:
        cx = crop_region.get('x', 0)
        cy = crop_region.get('y', 0)
        cw = crop_region.get('w', frames[0].shape[1])
        ch = crop_region.get('h', frames[0].shape[0])
        cropped_frames = [f[cy:cy+ch, cx:cx+cw] for f in frames]
    else:
        cropped_frames = frames

    sample_interval = max(1, int(source_fps / sample_fps))

    # Load YOLO model
    yolo_model = None
    if config.use_yolo:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(config.yolo_model)
        except Exception:
            pass

    # Compute trajectory (all points in LAST frame's coordinate system)
    trajectory: List[TrajectoryPoint] = []
    prev_gray = None
    prev_bbox = None

    for i in range(0, len(cropped_frames), sample_interval):
        frame = cropped_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = i / source_fps

        bbox = None
        if config.use_yolo and yolo_model is not None:
            bbox = detect_skier_bbox(frame, yolo_model, config.yolo_conf, config.person_class_id)
        if bbox is None and prev_bbox is not None:
            bbox = prev_bbox
        if bbox is None:
            prev_gray = gray
            continue

        foot_x, foot_y = extract_foot_position(bbox)

        if prev_gray is not None and len(trajectory) > 0:
            H = compute_homography(prev_gray, gray, skier_bbox=bbox, config=config)
            if H is not None:
                old_pts = [(tp.x, tp.y) for tp in trajectory]
                new_pts = transform_points(old_pts, H)
                for j, tp in enumerate(trajectory):
                    tp.x, tp.y = new_pts[j]

        trajectory.append(TrajectoryPoint(
            x=foot_x, y=foot_y, frame_idx=i, timestamp_sec=timestamp
        ))

        prev_gray = gray
        prev_bbox = bbox

    if len(trajectory) < 2:
        return None

    # Scale trajectory to montage dimensions if needed
    montage_h, montage_w = montage_image.shape[:2]
    frame_h, frame_w = cropped_frames[0].shape[:2]

    if montage_w != frame_w or montage_h != frame_h:
        scale_x = montage_w / frame_w
        scale_y = montage_h / frame_h
        for tp in trajectory:
            tp.x *= scale_x
            tp.y *= scale_y

    # Draw on montage
    result = draw_trajectory(montage_image, trajectory, config, source_fps)
    return result

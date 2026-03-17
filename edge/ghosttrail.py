#!/usr/bin/env python3
"""
GhostTrail Video Generator - Create stroboscopic slow-motion videos.

Creates a video that:
1. Plays at 0.25x speed (slow motion)
2. Every N frames, "freezes" an impression of the skier
3. Impressions accumulate on screen while the skier continues moving
4. Result is a video showing the progression through the turn with ghost trail

The "ghost trail" effect shows the athlete's movement path with accumulated
frozen impressions, creating a stroboscopic analysis visualization.
"""

import os
import logging
import subprocess
from typing import Optional, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_skier_mask(
    frame: np.ndarray,
    background: np.ndarray,
    threshold: int = 30,
    min_area: int = 500,
) -> np.ndarray:
    """
    Extract skier mask using frame differencing against background.

    Args:
        frame: Current frame (BGR)
        background: Background reference frame (BGR)
        threshold: Pixel difference threshold for detection
        min_area: Minimum contour area to keep (filters noise)

    Returns:
        Binary mask where skier pixels are 255, background is 0
    """
    # Convert to grayscale for differencing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray_frame, gray_bg)

    # Threshold to binary
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours and filter by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create clean mask with only large contours
    clean_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    return clean_mask


def add_impression(
    canvas: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.7,
) -> np.ndarray:
    """
    Add a skier impression to the accumulator canvas.

    Args:
        canvas: Accumulator image with previous impressions
        frame: Current frame with skier
        mask: Binary mask of skier region
        opacity: Opacity of the new impression (0-1)

    Returns:
        Updated canvas with new impression added
    """
    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Blend the new impression onto the canvas
    # Only blend where mask is active
    canvas_float = canvas.astype(np.float32)
    frame_float = frame.astype(np.float32)

    # Apply semi-transparent impression
    blended = canvas_float * (1 - mask_3ch * opacity) + frame_float * (mask_3ch * opacity)

    return blended.astype(np.uint8)


def generate_ghosttrail_video(
    frames: List[np.ndarray],
    output_path: str,
    source_fps: float = 30.0,
    slowmo_factor: float = 4.0,
    impression_interval: int = 4,
    impression_opacity: float = 1.0,
    crop_region: Optional[Dict] = None,
    diff_threshold: int = 25,
    min_skier_area: int = 800,
) -> Optional[str]:
    """
    Generate a GhostTrail stroboscopic slow-motion video.

    Args:
        frames: List of numpy arrays (BGR format from OpenCV).
        output_path: Where to write the .mp4 file.
        source_fps: FPS of the source video.
        slowmo_factor: Slow motion factor (4.0 = 0.25x speed).
        impression_interval: Capture impression every N frames.
        impression_opacity: Opacity of frozen impressions (0-1).
        crop_region: Optional dict with x, y, w, h to crop frames.
        diff_threshold: Threshold for frame differencing.
        min_skier_area: Minimum pixel area for skier detection.

    Returns:
        Path to the generated MP4 file, or None if generation fails.
    """
    if not frames or len(frames) < 2:
        logger.warning("Not enough frames for GhostTrail video")
        return None

    try:
        # Apply crop if specified
        if crop_region:
            x = crop_region.get('x', 0)
            y = crop_region.get('y', 0)
            w = crop_region.get('w', frames[0].shape[1])
            h = crop_region.get('h', frames[0].shape[0])
            frames = [f[y:y+h, x:x+w].copy() for f in frames]

        # Get frame dimensions
        sample = frames[0]
        frame_h, frame_w = sample.shape[:2]

        # Ensure even dimensions (required by H.264)
        out_w = frame_w if frame_w % 2 == 0 else frame_w - 1
        out_h = frame_h if frame_h % 2 == 0 else frame_h - 1

        if out_w < 2 or out_h < 2:
            logger.warning(f"Frame dimensions too small: {out_w}x{out_h}")
            return None

        # Use first frame as background reference (cropped to even dimensions)
        background = frames[0][:out_h, :out_w].copy()

        # Initialize accumulator canvas with background
        canvas = background.copy()

        # Calculate output FPS for slow motion
        output_fps = source_fps / slowmo_factor

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{out_w}x{out_h}',
            '-pix_fmt', 'bgr24',
            '-r', str(output_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-crf', '20',  # Higher quality for analysis videos
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]

        # Pipe frames to ffmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,  # Don't capture stdout to avoid blocking
            stderr=subprocess.PIPE,
        )

        impressions_added = 0
        frames_written = 0

        for i, frame in enumerate(frames):
            # Ensure even dimensions
            frame = frame[:out_h, :out_w]

            # Check if this is an impression frame
            if i > 0 and i % impression_interval == 0:
                # Extract skier mask
                mask = extract_skier_mask(
                    frame, background,
                    threshold=diff_threshold,
                    min_area=min_skier_area
                )

                # Add impression to canvas if mask has content
                if np.sum(mask) > 0:
                    canvas = add_impression(canvas, frame, mask, impression_opacity)
                    impressions_added += 1

            # Composite: canvas (with accumulated impressions) + current skier
            current_mask = extract_skier_mask(
                frame, background,
                threshold=diff_threshold,
                min_area=min_skier_area
            )

            # Create output frame: canvas + current moving skier overlaid
            output_frame = canvas.copy()

            # Overlay current skier at full opacity
            mask_3ch = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            output_float = output_frame.astype(np.float32)
            frame_float = frame.astype(np.float32)
            output_frame = (output_float * (1 - mask_3ch) + frame_float * mask_3ch).astype(np.uint8)

            # Write frame
            try:
                process.stdin.write(output_frame.tobytes())
                frames_written += 1
            except BrokenPipeError:
                logger.error(f"Broken pipe at frame {i}")
                break
            except Exception as e:
                logger.error(f"Error writing frame {i}: {e}")
                break

        process.stdin.close()
        process.wait(timeout=120)

        if process.returncode != 0:
            stderr = process.stderr.read().decode('utf-8', errors='replace')
            logger.error(f"ffmpeg failed (exit {process.returncode}): {stderr[-500:]}")
            return None

        # Verify output exists and has meaningful content (>1KB for a real video)
        if os.path.exists(output_path):
            size_bytes = os.path.getsize(output_path)
            if size_bytes > 1024:
                size_kb = size_bytes / 1024
                duration_sec = len(frames) / output_fps
                logger.info(f"GhostTrail video: {output_path} ({size_kb:.0f} KB, {duration_sec:.1f}s, {impressions_added} impressions)")
                print(f"  GhostTrail: {os.path.basename(output_path)} ({size_kb:.0f} KB, {impressions_added} impressions)")
                return output_path
            else:
                logger.error(f"Output too small ({size_bytes} bytes), ffmpeg may have failed")
                return None
        else:
            logger.error(f"Output file missing: {output_path}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        if process:
            process.kill()
        return None
    except Exception as e:
        logger.error(f"GhostTrail video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_ghosttrail_from_clip(
    video_path: str,
    output_path: str,
    slowmo_factor: float = 4.0,
    impression_interval: int = 4,
    impression_opacity: float = 1.0,
    crop_region: Optional[Dict] = None,
) -> Optional[str]:
    """
    Generate GhostTrail video from an existing video clip file.

    Args:
        video_path: Path to input video clip.
        output_path: Where to write the GhostTrail .mp4 file.
        slowmo_factor: Slow motion factor (4.0 = 0.25x speed).
        impression_interval: Capture impression every N frames.
        impression_opacity: Opacity of frozen impressions (0-1).
        crop_region: Optional dict with x, y, w, h to crop frames.

    Returns:
        Path to the generated MP4 file, or None if generation fails.
    """
    # Read all frames from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        logger.error(f"No frames read from video: {video_path}")
        return None

    return generate_ghosttrail_video(
        frames=frames,
        output_path=output_path,
        source_fps=fps,
        slowmo_factor=slowmo_factor,
        impression_interval=impression_interval,
        impression_opacity=impression_opacity,
        crop_region=crop_region,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate GhostTrail stroboscopic video")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output path (default: input_ghosttrail.mp4)")
    parser.add_argument("--slowmo", type=float, default=4.0, help="Slow motion factor (default: 4.0 = 0.25x)")
    parser.add_argument("--interval", type=int, default=4, help="Impression interval in frames (default: 4)")
    parser.add_argument("--opacity", type=float, default=0.75, help="Impression opacity 0-1 (default: 0.75)")

    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_ghosttrail.mp4"

    result = generate_ghosttrail_from_clip(
        video_path=args.input,
        output_path=args.output,
        slowmo_factor=args.slowmo,
        impression_interval=args.interval,
        impression_opacity=args.opacity,
    )

    if result:
        print(f"Generated: {result}")
    else:
        print("Failed to generate GhostTrail video")

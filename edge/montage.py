#!/usr/bin/env python3
"""
Montage Generator - Creates stop-motion photo composites from ski run frames.

Inspired by Ron LeMaster's technique photography in "Ultimate Skiing".
Uses OpenCV's minimum operation to blend frames, keeping the darkest pixels
(the skier) visible against the lighter snow background.

Outputs:
- Thumbnail: ~800px wide, JPEG 85%, ~200KB (for quick 5G review)
- Full resolution: Native crop zone resolution, JPEG 95% (aspect ratio preserved)
"""

import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# Default settings
DEFAULT_FPS = 4  # Frames per second to sample
DEFAULT_PADDING_LEFT_PCT = 30  # Left padding - keep wide for ski path
DEFAULT_PADDING_RIGHT_PCT = 50  # Right padding - lots needed for this view
DEFAULT_PADDING_Y_PCT = 10  # Vertical padding (top/bottom) - less needed


@dataclass
class MontageResult:
    """Result of montage generation."""
    run_number: int
    timestamp: datetime
    thumbnail_path: str
    fullres_path: str
    frame_count: int
    thumbnail_size: int  # bytes
    fullres_size: int    # bytes


@dataclass
class CropRegion:
    """Crop region computed from trigger zones."""
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_zones(cls, start_zone: Dict, end_zone: Dict,
                   frame_width: int, frame_height: int,
                   padding_left_pct: float = DEFAULT_PADDING_LEFT_PCT,
                   padding_right_pct: float = DEFAULT_PADDING_RIGHT_PCT,
                   padding_y_pct: float = DEFAULT_PADDING_Y_PCT) -> 'CropRegion':
        """
        Compute crop region that encompasses both trigger zones with padding.

        Args:
            start_zone: Dict with x, y, w, h for START zone
            end_zone: Dict with x, y, w, h for END zone
            frame_width: Original frame width
            frame_height: Original frame height
            padding_left_pct: Left padding percentage
            padding_right_pct: Right padding percentage
            padding_y_pct: Vertical padding percentage (top/bottom)
        """
        # Get bounding box of both zones
        x1 = min(start_zone['x'], end_zone['x'])
        y1 = min(start_zone['y'], end_zone['y'])
        x2 = max(start_zone['x'] + start_zone['w'], end_zone['x'] + end_zone['w'])
        y2 = max(start_zone['y'] + start_zone['h'], end_zone['y'] + end_zone['h'])

        # Calculate padding (separate left/right, same top/bottom)
        region_w = x2 - x1
        region_h = y2 - y1
        pad_left = int(region_w * padding_left_pct / 100)
        pad_right = int(region_w * padding_right_pct / 100)
        pad_y = int(region_h * padding_y_pct / 100)

        # Apply padding and clamp to frame bounds
        x1 = max(0, x1 - pad_left)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_width, x2 + pad_right)
        y2 = min(frame_height, y2 + pad_y)

        return cls(x=x1, y=y1, w=x2-x1, h=y2-y1)


def crop_frame(frame: np.ndarray, crop: CropRegion) -> np.ndarray:
    """Crop a frame to the specified region."""
    return frame[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w]


def select_frames_by_fps(frames: List[np.ndarray], source_fps: float,
                         target_fps: float = DEFAULT_FPS,
                         start_offset: int = 0) -> List[np.ndarray]:
    """
    Select frames at the target FPS rate from source frames.

    Args:
        frames: List of frames captured at source_fps
        source_fps: FPS at which frames were captured
        target_fps: Desired FPS for the montage
        start_offset: Number of frames to skip before starting selection

    Returns:
        List of frames sampled at target_fps
    """
    if not frames:
        return frames

    # Apply start offset
    if start_offset > 0 and start_offset < len(frames):
        frames = frames[start_offset:]

    if target_fps >= source_fps:
        return frames

    # Calculate interval
    interval = source_fps / target_fps

    # Select frames at interval
    selected = []
    next_frame = 0.0
    for i, frame in enumerate(frames):
        if i >= next_frame:
            selected.append(frame)
            next_frame += interval

    return selected


def create_composite_imagemagick(frames: List[np.ndarray], temp_dir: str = None) -> np.ndarray:
    """
    Create composite from frames using pixel-wise Min evaluation.

    Takes the minimum (darkest) pixel value across all frames,
    producing the classic stop-motion overlay effect. Uses NumPy
    for speed and reliability (no ImageMagick memory limits).

    Args:
        frames: List of frames (numpy arrays)
        temp_dir: Optional temp directory (unused, kept for API compat)

    Returns:
        Composite image as numpy array
    """
    if not frames:
        raise ValueError("No frames provided")

    # NumPy min across all frames — same result as ImageMagick
    # evaluate-sequence Min, but faster and no memory limits
    composite = frames[0].copy()
    for frame in frames[1:]:
        np.minimum(composite, frame, out=composite)

    return composite


def resize_for_thumbnail(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize image for thumbnail, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image

    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_overlay(image: np.ndarray, timestamp: datetime, fps: float, variant: str = "",
                duration_sec: Optional[float] = None, race_title: str = "",
                race_info: Optional[Dict] = None) -> np.ndarray:
    """Add overlay with race title, date, duration, FPS, and variant indicator.

    Title format: Western Division Ranking | SL | U14 | Run 1 | 2026-02-01 | 2.0s | 4fps (+2frames)
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from datetime import timezone, timedelta
        # Python 3.8 fallback: define US/Eastern as UTC-5 (no DST handling)
        class ZoneInfo:
            def __init__(self, name):
                self.name = name
                # Simple EST offset (UTC-5)
                self._offset = timezone(timedelta(hours=-5))
            def __repr__(self):
                return f"ZoneInfo('{self.name}')"

    img = image.copy()
    h, w = img.shape[:2]

    # Text settings - scale with image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, w / 2000)
    thickness = max(1, int(w / 1000))

    # Convert timestamp to Boston time (US/Eastern)
    boston_tz = ZoneInfo("America/New_York")
    if timestamp.tzinfo is None:
        # Assume local time already (from RTSP/detection), just use date as-is
        date_str = timestamp.strftime("%Y-%m-%d")
    else:
        boston_time = timestamp.astimezone(boston_tz)
        date_str = boston_time.strftime("%Y-%m-%d")

    # Build title from race_info if provided
    # Format: "Western Division Ranking | SL | U14 | Run 1 | 2026-02-01"
    if race_info:
        title_parts = []
        if race_info.get("event"):
            title_parts.append(race_info["event"])
        if race_info.get("discipline"):
            title_parts.append(race_info["discipline"])
        if race_info.get("age_group"):
            title_parts.append(race_info["age_group"])
        if race_info.get("run"):
            title_parts.append(race_info["run"])
        title_parts.append(date_str)
        title_part = " | ".join(title_parts)
    elif race_title:
        title_part = f"{race_title} | {date_str}"
    else:
        title_part = f"Ski Race | {date_str}"

    # Build duration part if provided
    duration_part = f"{duration_sec:.1f}s" if duration_sec else ""

    # Build FPS part with optional variant
    fps_part = f"{fps:.0f}fps"
    if variant:
        fps_part += f" ({variant})"

    # Combine parts, including duration if available
    if duration_part:
        text = f"{title_part} | {duration_part} | {fps_part}"
    else:
        text = f"{title_part} | {fps_part}"

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Position in bottom-right corner
    padding = int(8 * font_scale)
    x = w - text_w - padding * 2
    y = h - padding

    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (x - padding, y - text_h - padding),
                  (w, h),
                  (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, (50, 50, 50), thickness)

    # Add logos at bottom-left
    img = add_logos(img)

    return img


def add_logos(image: np.ndarray) -> np.ndarray:
    """Add sponsor logos at bottom-left corner."""
    img = image.copy()
    h, w = img.shape[:2]

    # Logo directory
    logo_dir = Path(__file__).parent.parent / 'logos'
    logo_files = ['NHARA_logo.png', 'RMST_logo.png', 'Ragged_logo.png']

    # Target logo height - scale with image size (roughly 6% of image height)
    logo_height = int(h * 0.06)
    padding = int(logo_height * 0.3)

    # Starting position - bottom left
    x_pos = padding
    y_bottom = h - padding

    for logo_file in logo_files:
        logo_path = logo_dir / logo_file
        if not logo_path.exists():
            continue

        # Load logo with alpha channel
        logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if logo is None:
            continue

        # Resize logo to target height while maintaining aspect ratio
        logo_h, logo_w = logo.shape[:2]
        scale = logo_height / logo_h
        new_w = int(logo_w * scale)
        new_h = logo_height
        logo = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Position for this logo
        y_pos = y_bottom - new_h

        # Check if logo has alpha channel
        if logo.shape[2] == 4:
            # Split into BGR and alpha
            bgr = logo[:, :, :3]
            alpha = logo[:, :, 3] / 255.0

            # Blend with background
            for c in range(3):
                img[y_pos:y_pos+new_h, x_pos:x_pos+new_w, c] = \
                    (alpha * bgr[:, :, c] + (1 - alpha) * img[y_pos:y_pos+new_h, x_pos:x_pos+new_w, c])
        else:
            # No alpha, just overlay
            img[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = logo

        # Move x position for next logo
        x_pos += new_w + padding

    return img


@dataclass
class MontageResultPair:
    """Result of Time1/Time2 montage generation."""
    run_number: int
    timestamp: datetime
    results: Dict[str, MontageResult]  # 'A' and 'B' keys


def generate_montage(frames: List[np.ndarray],
                     run_number: int,
                     timestamp: datetime,
                     output_dir: str,
                     session_id: str,
                     start_zone: Optional[Dict] = None,
                     end_zone: Optional[Dict] = None,
                     crop_zone: Optional[Dict] = None,
                     source_fps: float = 30.0,
                     montage_fps: float = DEFAULT_FPS,
                     padding_left_pct: float = DEFAULT_PADDING_LEFT_PCT,
                     padding_right_pct: float = DEFAULT_PADDING_RIGHT_PCT,
                     padding_y_pct: float = DEFAULT_PADDING_Y_PCT,
                     add_branding: bool = False,
                     b_offset_frames: int = 3,
                     custom_filename: Optional[str] = None,
                     run_view_folder: Optional[str] = None,
                     run_duration_sec: Optional[float] = None,
                     race_title: str = "",
                     race_info: Optional[Dict] = None) -> Optional[MontageResultPair]:
    """
    Generate A and B montage pairs from run frames.

    A starts at frame 0, B starts 3 frames later (offset).

    Args:
        frames: List of video frames from the run
        run_number: Run number for naming
        timestamp: Timestamp of run start
        output_dir: Directory to save outputs
        session_id: Session ID for naming
        start_zone: Dict with x, y, w, h for START zone (for cropping, if no crop_zone)
        end_zone: Dict with x, y, w, h for END zone (for cropping, if no crop_zone)
        crop_zone: Dict with x, y, w, h for exact crop area (overrides start/end zone calculation)
        source_fps: FPS at which frames were captured
        montage_fps: Target FPS for the montage
        padding_left_pct: Left padding percentage (only used if no crop_zone)
        padding_right_pct: Right padding percentage (only used if no crop_zone)
        padding_y_pct: Vertical padding percentage (only used if no crop_zone)
        b_offset_frames: Number of frames to offset B version (default 3)
        custom_filename: Optional custom base filename (e.g., "JohnDoe_106")
                        If provided, used instead of run_XXX_HHMMSS format
        run_view_folder: Optional folder name for organization (e.g., "Run1_View1")
                        If provided, outputs are placed in this subfolder
        run_duration_sec: Optional run duration in seconds for overlay display

    Returns:
        MontageResultPair with A and B results, or None on failure
    """
    if not frames:
        print(f"  ERROR: No frames for run {run_number}")
        return None

    print(f"  Generating montage A/B for run {run_number} ({len(frames)} frames @ {source_fps:.0f}fps)...")

    # Compute crop region - use crop_zone if provided, otherwise calculate from trigger zones
    crop = None
    if crop_zone and len(frames) > 0:
        # Use exact crop zone from config
        crop = CropRegion(x=crop_zone['x'], y=crop_zone['y'], w=crop_zone['w'], h=crop_zone['h'])
        print(f"    Crop region (from crop_zone): ({crop.x}, {crop.y}) {crop.w}x{crop.h}")
    elif start_zone and end_zone and len(frames) > 0:
        # Calculate from trigger zones with padding
        frame_h, frame_w = frames[0].shape[:2]
        crop = CropRegion.from_zones(start_zone, end_zone, frame_w, frame_h, padding_left_pct, padding_right_pct, padding_y_pct)
        print(f"    Crop region (from zones): ({crop.x}, {crop.y}) {crop.w}x{crop.h}")

    # Create output directories
    # If run_view_folder provided (e.g., "Run1_View1"), create subfolder structure
    if run_view_folder:
        thumb_dir = os.path.join(output_dir, run_view_folder, "thumbnails")
        full_dir = os.path.join(output_dir, run_view_folder, "fullres")
    else:
        thumb_dir = os.path.join(output_dir, "thumbnails")
        full_dir = os.path.join(output_dir, "fullres")
    os.makedirs(thumb_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)

    time_str = timestamp.strftime("%H%M%S")
    results = {}

    # Generate two versions: base (offset=0) and +2frames (offset by b_offset_frames)
    time2_offset = max(1, b_offset_frames)  # Use b_offset_frames parameter (default 3)
    # variant_label: shown in overlay, file_suffix: used in filename
    for variant_label, file_suffix, offset in [('', '', 0), ('2 frames later', '_2later', time2_offset)]:
        # Sample frames at target FPS with offset
        selected = select_frames_by_fps(frames, source_fps, montage_fps, start_offset=offset)
        variant_name = file_suffix if file_suffix else 'base'
        print(f"    {variant_name}: Sampled {len(selected)} frames (offset={offset})")

        # Crop selected frames
        if crop:
            selected = [crop_frame(f, crop) for f in selected]

        # Create composite using ImageMagick (best quality)
        try:
            composite = create_composite_imagemagick(selected)
        except Exception as e:
            print(f"    ERROR creating composite {variant_name}: {e}")
            continue

        # Add branding overlay with variant label and race title/info
        if add_branding:
            composite = add_overlay(composite, timestamp, montage_fps, variant_label, run_duration_sec, race_title, race_info)

        # Output paths - use file_suffix for naming
        # Use custom filename if provided (from Vola racer data), otherwise use run number
        if custom_filename:
            base_name = f"{custom_filename}{file_suffix}"
        else:
            base_name = f"run_{run_number:03d}_{time_str}{file_suffix}"
        thumb_path = os.path.join(thumb_dir, f"{base_name}_thumb.jpg")
        full_path = os.path.join(full_dir, f"{base_name}.jpg")

        # Generate thumbnail
        thumbnail = resize_for_thumbnail(composite, max_width=800)
        cv2.imwrite(thumb_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Save full resolution at native size (preserves crop zone aspect ratio)
        cv2.imwrite(full_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Get file sizes
        thumb_size = os.path.getsize(thumb_path)
        full_size = os.path.getsize(full_path)

        print(f"    {variant_name} Thumbnail: {thumb_path} ({thumb_size / 1024:.0f} KB)")
        print(f"    {variant_name} Full-res: {full_path} ({full_size / 1024 / 1024:.1f} MB)")

        # Use file_suffix as key ('' for base, '_2later' for 2 frames later)
        result_key = file_suffix if file_suffix else 'base'
        results[result_key] = MontageResult(
            run_number=run_number,
            timestamp=timestamp,
            thumbnail_path=thumb_path,
            fullres_path=full_path,
            frame_count=len(selected),
            thumbnail_size=thumb_size,
            fullres_size=full_size,
        )

    if not results:
        return None

    return MontageResultPair(
        run_number=run_number,
        timestamp=timestamp,
        results=results,
    )


def main():
    """CLI for testing montage generator."""
    import argparse
    from datetime import timedelta

    parser = argparse.ArgumentParser(description="Generate ski run montage")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("-s", "--start", type=float, required=True, help="Start time (seconds)")
    parser.add_argument("-e", "--end", type=float, required=True, help="End time (seconds)")
    parser.add_argument("-n", "--run-number", type=int, default=1, help="Run number")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--session", default="test", help="Session ID")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Montage FPS")

    args = parser.parse_args()

    # Extract frames from video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time > args.end:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        print("ERROR: No frames extracted")
        return

    result = generate_montage(
        frames=frames,
        run_number=args.run_number,
        timestamp=datetime.now(),
        output_dir=args.output,
        session_id=args.session,
        source_fps=fps,
        montage_fps=args.fps,
    )

    if result:
        print(f"\nMontage generated successfully!")
        print(f"  Thumbnail: {result.thumbnail_path}")
        print(f"  Full-res: {result.fullres_path}")


if __name__ == "__main__":
    main()

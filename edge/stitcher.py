#!/usr/bin/env python3
"""
Video stitcher for multi-camera ski race footage.
Creates broadcast-style sequential cuts from 4 camera angles into a single video per athlete.
"""

import os
import re
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Detect platform for hardware acceleration
IS_MACOS = sys.platform == 'darwin'
HAS_NVENC = False
NUM_GPUS = 0
_gpu_counter = 0
_gpu_lock = threading.Lock()

# Check for NVIDIA GPU with NVENC support on Linux
if not IS_MACOS:
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            # Verify ffmpeg has nvenc support
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
            if 'h264_nvenc' in result.stdout:
                HAS_NVENC = True
                # Count available GPUs
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
                NUM_GPUS = len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU ')])
                print(f"NVIDIA NVENC detected - using GPU encoding ({NUM_GPUS} GPUs available)")
    except Exception:
        pass

def _get_next_gpu():
    """Round-robin GPU selection for parallel encoding."""
    global _gpu_counter
    if NUM_GPUS <= 1:
        return 0
    with _gpu_lock:
        gpu_id = _gpu_counter % NUM_GPUS
        _gpu_counter += 1
        return gpu_id

def get_encoder_args(gpu_id: int = None):
    """Get the appropriate encoder arguments for the current platform."""
    if IS_MACOS:
        return ['-c:v', 'h264_videotoolbox', '-b:v', '8M']
    elif HAS_NVENC:
        # Use specified GPU or round-robin
        if gpu_id is None:
            gpu_id = _get_next_gpu()
        # NVENC with quality settings comparable to CRF 18
        return ['-gpu', str(gpu_id), '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '18', '-b:v', '0']
    else:
        return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '18']


@dataclass
class CameraCut:
    """Defines a camera segment in the stitched video."""
    camera: str
    start_pct: float  # Percentage of run time when this camera starts (can be negative)
    end_pct: float    # Percentage of run time when this camera ends (can be > 1.0)


@dataclass
class Racer:
    """Racer timing data from Vola."""
    bib: int
    name: str
    team: str
    gender: str
    start_time_sec: float  # Seconds since midnight
    finish_time_sec: float  # Seconds since midnight
    duration: float  # Run duration in seconds
    ussa_id: str = ""  # USSA member ID (e.g., "7225120")
    status: str = "finished"  # "finished", "DSQ", "DNF", "DNS"

    @property
    def start_time_str(self) -> str:
        """Format start time as HH:MM:SS."""
        h = int(self.start_time_sec // 3600)
        m = int((self.start_time_sec % 3600) // 60)
        s = int(self.start_time_sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def ussa_profile_url(self) -> Optional[str]:
        """Get USSA profile URL if ussa_id is set."""
        if self.ussa_id:
            # Strip leading 'E' prefix from USSA ID for the profile URL
            numeric_id = self.ussa_id.lstrip('E')
            return f"https://www.usskiandsnowboard.org/public-tools/members/{numeric_id}"
        return None


@dataclass
class VideoFile:
    """Represents a video file with its time range."""
    path: str
    start_sec: float  # Seconds since midnight when video starts
    end_sec: float    # Seconds since midnight when video ends (estimated)
    duration: float   # Actual duration in seconds


def parse_video_start_time(video_path: str) -> Optional[float]:
    """
    Parse start time from video filename.
    Supports:
    - Reolink format: RecM03_YYYYMMDD_HHMMSS_HHMMSS_XXXXX_XXXXX.mp4
    - Axis format: YYYYMMDD_HHMMSS_XXXX.mkv
    Returns seconds since midnight.
    """
    filename = os.path.basename(video_path)

    # Try Reolink format first (has end time in filename)
    match = re.match(r'RecM03_(\d{8})_(\d{6})_(\d{6})_', filename)
    if match:
        start_time_str = match.group(2)  # HHMMSS
        hours = int(start_time_str[0:2])
        minutes = int(start_time_str[2:4])
        seconds = int(start_time_str[4:6])
        return hours * 3600 + minutes * 60 + seconds

    # Try Axis format: YYYYMMDD_HHMMSS_XXXX.mkv
    match = re.match(r'(\d{8})_(\d{6})_', filename)
    if match:
        start_time_str = match.group(2)  # HHMMSS
        hours = int(start_time_str[0:2])
        minutes = int(start_time_str[2:4])
        seconds = int(start_time_str[4:6])
        return hours * 3600 + minutes * 60 + seconds

    return None


def parse_reolink_end_time(video_path: str) -> Optional[float]:
    """Parse end time from Reolink filename (has both start and end in name)."""
    filename = os.path.basename(video_path)
    match = re.match(r'RecM03_(\d{8})_(\d{6})_(\d{6})_', filename)
    if match:
        end_time_str = match.group(3)  # HHMMSS
        hours = int(end_time_str[0:2])
        minutes = int(end_time_str[2:4])
        seconds = int(end_time_str[4:6])
        return hours * 3600 + minutes * 60 + seconds
    return None


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def build_video_index(video_paths: List[str]) -> List[VideoFile]:
    """
    Build an index of video files with their time ranges.
    Returns list sorted by start time.
    """
    videos = []
    for path in video_paths:
        start_sec = parse_video_start_time(path)
        if start_sec is None:
            print(f"  Warning: Could not parse start time from {path}")
            continue

        # Try to get end time from filename (Reolink) or estimate
        end_sec = parse_reolink_end_time(path)
        if end_sec:
            duration = end_sec - start_sec
        else:
            # Get actual duration via ffprobe
            duration = get_video_duration(path)
            if duration is None:
                duration = 300  # Default to 5 minutes
            end_sec = start_sec + duration

        videos.append(VideoFile(
            path=path,
            start_sec=start_sec,
            end_sec=end_sec,
            duration=duration
        ))

    # Sort by start time
    videos.sort(key=lambda v: v.start_sec)
    return videos


def find_video_for_time(videos: List[VideoFile], target_time: float) -> Optional[VideoFile]:
    """Find the video file that contains the target time."""
    for video in videos:
        if video.start_sec <= target_time <= video.end_sec:
            return video
    return None


def find_next_video(videos: List[VideoFile], current_video: VideoFile) -> Optional[VideoFile]:
    """Find the next video file after the current one (by start time)."""
    # Videos should already be sorted by start time
    for i, video in enumerate(videos):
        if video.path == current_video.path and i + 1 < len(videos):
            return videos[i + 1]
    return None


class VideoStitcher:
    """
    Stitches multiple camera angles into a single video per racer.

    Uses ffmpeg to:
    1. Extract time-synced segments from each camera
    2. Normalize to 1080p/30fps
    3. Add title and logo overlay
    4. Concatenate with hard cuts
    """

    def __init__(
        self,
        racers: List[Racer],
        cuts: List[CameraCut],
        video_paths: Dict[str, Union[str, List[str]]],  # camera_id -> video path or list of paths
        output_dir: str,
        race_name: str = "race",
        logo_dir: Optional[str] = None,
        race_title: str = "",
        race_info: Optional[Dict] = None,
        selected_logos: Optional[List[str]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        pre_buffer_sec: float = 2.0,
        post_buffer_sec: float = 2.0
    ):
        """
        Initialize the stitcher.

        Args:
            racers: List of Racer objects with timing data
            cuts: List of CameraCut defining the edit sequence
            video_paths: Dict mapping camera ID to video path(s) - can be single path or list
            output_dir: Directory for output videos
            race_name: Name for output folder (e.g., "u12_run1")
            logo_dir: Directory containing logo PNG files
            race_title: Title to overlay on video (e.g., "Western Division U14 Ranking - SL")
            race_info: Optional dict with race metadata (event, discipline, age_group, run,
                      date, course, location, type, vertical_drop, length, gates, snow)
            selected_logos: Optional list of logo filenames in order (left to right)
            stop_flag: Optional callable that returns True if processing should stop
            pre_buffer_sec: Seconds before racer start to begin video (default: 2.0)
            post_buffer_sec: Seconds after racer finish to end video (default: 2.0)
        """
        self.racers = racers
        self.cuts = sorted(cuts, key=lambda c: c.start_pct)  # Ensure chronological order
        self.pre_buffer_sec = pre_buffer_sec
        self.post_buffer_sec = post_buffer_sec
        self.output_dir = Path(output_dir)
        self.race_name = race_name
        self.logo_dir = Path(logo_dir) if logo_dir else Path(__file__).parent.parent / 'logos'
        self.race_title = race_title
        self.race_info = race_info or {}
        self.selected_logos = selected_logos  # User-selected logos in order
        self.stop_flag = stop_flag
        self.event_date: Optional[str] = None  # Will be extracted from video filenames

        # Build video index for each camera
        self.camera_videos: Dict[str, List[VideoFile]] = {}
        for camera, paths in video_paths.items():
            # Normalize to list
            if isinstance(paths, str):
                paths = [paths]

            print(f"  Indexing {len(paths)} video(s) for {camera}...")
            self.camera_videos[camera] = build_video_index(paths)

            if self.camera_videos[camera]:
                first = self.camera_videos[camera][0]
                last = self.camera_videos[camera][-1]
                print(f"    Time range: {first.start_sec//3600:02.0f}:{(first.start_sec%3600)//60:02.0f} - {last.end_sec//3600:02.0f}:{(last.end_sec%3600)//60:02.0f}")

                # Extract event date from first video filename if not already set
                if not self.event_date:
                    self.event_date = self._extract_date_from_filename(first.path)

    def _should_stop(self) -> bool:
        """Check if processing should stop."""
        if self.stop_flag:
            return self.stop_flag()
        return False

    def _extract_date_from_filename(self, video_path: str) -> Optional[str]:
        """
        Extract date from video filename.
        Supports Reolink (RecM03_YYYYMMDD_...) and Axis (YYYYMMDD_...) formats.
        Returns date as YYYY-MM-DD string.
        """
        filename = os.path.basename(video_path)

        # Try Reolink format: RecM03_YYYYMMDD_...
        match = re.match(r'RecM03_(\d{4})(\d{2})(\d{2})_', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        # Try Axis format: YYYYMMDD_...
        match = re.match(r'(\d{4})(\d{2})(\d{2})_', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

        return None

    def _find_video_and_offset(self, camera: str, absolute_time: float) -> tuple:
        """
        Find the video file containing the absolute time and calculate offset.

        Returns:
            (video_path, offset_in_video, video_end_sec) or (None, None, None) if not found
        """
        videos = self.camera_videos.get(camera, [])
        video = find_video_for_time(videos, absolute_time)

        if video:
            offset = absolute_time - video.start_sec
            return video.path, offset, video.end_sec

        return None, None, None

    def _extract_segment(
        self,
        video_path: str,
        offset_sec: float,
        duration_sec: float,
        output_path: str
    ) -> bool:
        """
        Extract a segment from a video file using ffmpeg.
        Normalizes to 1080p/30fps H.264 using M1 VideoToolbox hardware acceleration.
        No audio for faster processing.

        Returns True on success.
        """
        # Use -ss BEFORE -i for fast seeking to nearest keyframe
        # Then use -ss after -i for fine-tuning (hybrid approach)
        # This gives fast seeking without black frames

        # Calculate coarse seek (2 seconds before target for keyframe alignment)
        coarse_seek = max(0, offset_sec - 2)
        fine_seek = offset_sec - coarse_seek

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(coarse_seek),  # Fast seek to keyframe BEFORE input
            '-i', video_path,
            '-ss', str(fine_seek),    # Fine seek AFTER input (accurate)
            '-t', str(duration_sec),
            '-an',  # No audio - faster processing
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30',
        ]
        # Use hardware encoding based on platform
        cmd.extend(get_encoder_args())
        cmd.extend([
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # Faster with hardware encoding
            )
            if result.returncode != 0:
                # Fallback to software encoding if VideoToolbox fails
                print(f"    VideoToolbox failed, falling back to software: {result.stderr[:100]}")
                return self._extract_segment_software(video_path, offset_sec, duration_sec, output_path)

            # Verify output file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # Less than 1KB is likely an error
                    print(f"    Warning: Extracted file is too small ({file_size} bytes), extraction may have failed")
                    return False
                return True
            else:
                print(f"    Warning: Output file was not created: {output_path}")
                return False
        except subprocess.TimeoutExpired:
            print(f"    Warning: Extraction timed out")
            return False

    def _extract_segment_software(
        self,
        video_path: str,
        offset_sec: float,
        duration_sec: float,
        output_path: str
    ) -> bool:
        """Software fallback for segment extraction."""
        coarse_seek = max(0, offset_sec - 2)
        fine_seek = offset_sec - coarse_seek

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(coarse_seek),
            '-i', video_path,
            '-ss', str(fine_seek),
            '-t', str(duration_sec),
            '-an',
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30',
        ]
        cmd.extend(get_encoder_args())
        cmd.extend(['-movflags', '+faststart', output_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                print(f"    ffmpeg error: {result.stderr[:200]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _get_font(self, size: int = 24):
        """Get a font for PIL drawing. Works on both Mac and Linux."""
        from PIL import ImageFont
        font_paths = [
            # Mac fonts
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/HelveticaNeue.ttc',
            '/Library/Fonts/Arial.ttf',
            # Linux fonts (common locations)
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            '/usr/share/fonts/TTF/DejaVuSans.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans.ttf',
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except:
                    continue
        # Last resort: try to use PIL's default with size parameter if available
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except:
            pass
        return ImageFont.load_default()

    def _create_title_overlay(self, title_text: str, racer_id: str = "", width: int = 1920, height: int = 1080) -> Optional[str]:
        """
        Create a full-frame transparent overlay with title at bottom right.
        Returns path to temporary PNG file, or None on failure.
        """
        try:
            from PIL import Image, ImageDraw
            import uuid

            # Use unique filename per racer to avoid conflicts
            unique_id = racer_id or str(uuid.uuid4())[:8]
            title_path = os.path.join(tempfile.gettempdir(), f"skiframes_title_{os.getpid()}_{unique_id}.png")

            # Create transparent image
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            font = self._get_font(28)

            # Get text size
            bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position at bottom right with padding
            padding = 20
            box_padding = 10
            x = width - text_width - padding - box_padding * 2
            y = height - text_height - padding - box_padding * 2

            # Draw semi-transparent background box
            draw.rectangle(
                [x - box_padding, y - box_padding, x + text_width + box_padding, y + text_height + box_padding],
                fill=(0, 0, 0, 153)
            )

            # Draw white text
            draw.text((x, y), title_text, font=font, fill=(255, 255, 255, 255))

            img.save(title_path, 'PNG')
            return title_path

        except Exception as e:
            print(f"    Title overlay creation failed: {e}")
            return None

    def _create_timer_frames(self, run_duration_sec: float, video_duration_sec: float, racer_id: str = "", fps: int = 30, width: int = 1920, height: int = 1080, pre_buffer_sec: float = 0.0) -> Optional[str]:
        """
        Create a video with timer overlay (top right) using PIL frames.
        Timer starts at -pre_buffer_sec (showing negative time before racer starts),
        shows 0:00.0 when racer actually starts, and stops at run_duration_sec.
        Returns path to temporary video file.
        """
        try:
            from PIL import Image, ImageDraw
            import uuid

            # Use unique filename per racer to avoid conflicts
            unique_id = racer_id or str(uuid.uuid4())[:8]

            # Create temp directory for frames
            frames_dir = os.path.join(tempfile.gettempdir(), f"skiframes_timer_{os.getpid()}_{unique_id}")
            os.makedirs(frames_dir, exist_ok=True)

            font = self._get_font(48)  # Bigger font
            total_frames = int(video_duration_sec * fps) + 1

            print(f"    Creating {total_frames} timer frames (pre-buffer: {pre_buffer_sec:.1f}s, run: {run_duration_sec:.1f}s)...")

            for frame_num in range(total_frames):
                # Calculate time for this frame
                # Video starts at -pre_buffer_sec, racer starts at 0, finishes at run_duration_sec
                raw_time_sec = (frame_num / fps) - pre_buffer_sec

                # Clamp: don't go below -pre_buffer_sec or above run_duration_sec
                time_sec = max(-pre_buffer_sec, min(raw_time_sec, run_duration_sec))

                # Format time - handle negative values
                is_negative = time_sec < 0
                abs_time = abs(time_sec)
                minutes = int(abs_time // 60)
                seconds = abs_time % 60

                # Format as M:SS.s - truncate (not round) the decimal
                whole_sec = int(seconds)
                tenths = int((seconds - whole_sec) * 10)  # Truncate, not round
                timer_text = f"{'-' if is_negative else ''}{minutes}:{whole_sec:02d}.{tenths}"

                # Create transparent image
                img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Get text size
                bbox = draw.textbbox((0, 0), timer_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Position at top right, a bit lower (y=60 instead of 20)
                padding = 30
                box_padding = 12
                x = width - text_width - padding - box_padding * 2
                y = 60  # Lower position

                # Draw semi-transparent background box
                draw.rectangle(
                    [x - box_padding, y - box_padding, x + text_width + box_padding, y + text_height + box_padding],
                    fill=(0, 0, 0, 180)
                )

                # Draw white text
                draw.text((x, y), timer_text, font=font, fill=(255, 255, 255, 255))

                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
                img.save(frame_path, 'PNG')

            # Convert frames to video using ffmpeg
            timer_video = os.path.join(tempfile.gettempdir(), f"skiframes_timer_{os.getpid()}_{unique_id}.mov")
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%05d.png'),
                '-c:v', 'png',  # Use PNG codec for transparency
                '-pix_fmt', 'rgba',
                timer_video
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Cleanup frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

            if result.returncode == 0:
                return timer_video
            else:
                print(f"    Timer video creation failed: {result.stderr[:200]}")
                return None

        except Exception as e:
            print(f"    Timer creation failed: {e}")
            return None

    def _create_comparison_timer_frames(
        self,
        run_duration_sec: float,
        video_duration_sec: float,
        fastest_duration_sec: float,
        fastest_bib: int,
        racer_id: str = "",
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        pre_buffer_sec: float = 0.0
    ) -> Optional[str]:
        """
        Create a video with two timer overlays:
        - Main timer at top right showing current run time
        - Comparison timer below showing time difference vs fastest racer

        Timer starts at -pre_buffer_sec (showing negative time before racer starts),
        shows 0:00.0 when racer actually starts, and stops at run_duration_sec.
        Returns path to temporary video file.
        """
        try:
            from PIL import Image, ImageDraw
            import uuid

            # Use unique filename per racer to avoid conflicts
            unique_id = racer_id or str(uuid.uuid4())[:8]

            # Create temp directory for frames
            frames_dir = os.path.join(tempfile.gettempdir(), f"skiframes_timer_cmp_{os.getpid()}_{unique_id}")
            os.makedirs(frames_dir, exist_ok=True)

            font_main = self._get_font(48)  # Main timer font
            font_diff = self._get_font(36)  # Difference timer font (smaller)
            font_label = self._get_font(20)  # Label font
            total_frames = int(video_duration_sec * fps) + 1

            print(f"    Creating {total_frames} comparison timer frames (vs bib {fastest_bib}, pre-buffer: {pre_buffer_sec:.1f}s)...")

            for frame_num in range(total_frames):
                # Calculate time for this frame
                # Video starts at -pre_buffer_sec, racer starts at 0, finishes at run_duration_sec
                raw_time_sec = (frame_num / fps) - pre_buffer_sec

                # Clamp: don't go below -pre_buffer_sec or above run_duration_sec
                time_sec = max(-pre_buffer_sec, min(raw_time_sec, run_duration_sec))

                # Format time - handle negative values
                is_negative = time_sec < 0
                abs_time = abs(time_sec)
                minutes = int(abs_time // 60)
                seconds = abs_time % 60

                # Format main timer as M:SS.s - truncate (not round) the decimal
                whole_sec = int(seconds)
                tenths = int((seconds - whole_sec) * 10)
                timer_text = f"{'-' if is_negative else ''}{minutes}:{whole_sec:02d}.{tenths}"

                # Calculate time difference vs fastest
                # At the same point in their run, how far ahead/behind are they?
                # Negative = behind (slower), Positive = ahead (faster)
                # We compare at same elapsed time percentage
                # During pre-buffer (time_sec < 0), show no difference yet
                if time_sec <= 0:
                    time_diff = 0
                elif fastest_duration_sec > 0 and run_duration_sec > 0:
                    # At this point in the run, fastest racer would be at:
                    pct_complete = time_sec / run_duration_sec if run_duration_sec > 0 else 0
                    fastest_time_at_pct = pct_complete * fastest_duration_sec
                    # Difference: negative means current racer is slower
                    time_diff = fastest_time_at_pct - time_sec
                else:
                    time_diff = 0

                # Format difference timer
                if time_diff >= 0:
                    diff_sign = "+"
                    diff_color = (0, 255, 0, 255)  # Green for ahead
                else:
                    diff_sign = "-"
                    diff_color = (255, 100, 100, 255)  # Red for behind
                    time_diff = abs(time_diff)

                diff_minutes = int(time_diff // 60)
                diff_seconds = time_diff % 60
                diff_whole_sec = int(diff_seconds)
                diff_tenths = int((diff_seconds - diff_whole_sec) * 10)

                if diff_minutes > 0:
                    diff_text = f"{diff_sign}{diff_minutes}:{diff_whole_sec:02d}.{diff_tenths}"
                else:
                    diff_text = f"{diff_sign}{diff_whole_sec}.{diff_tenths}"

                # Create transparent image
                img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Get text sizes
                bbox_main = draw.textbbox((0, 0), timer_text, font=font_main)
                main_width = bbox_main[2] - bbox_main[0]
                main_height = bbox_main[3] - bbox_main[1]

                label_text = f"vs Bib {fastest_bib}"
                bbox_label = draw.textbbox((0, 0), label_text, font=font_label)
                label_width = bbox_label[2] - bbox_label[0]
                label_height = bbox_label[3] - bbox_label[1]

                bbox_diff = draw.textbbox((0, 0), diff_text, font=font_diff)
                diff_width = bbox_diff[2] - bbox_diff[0]
                diff_height = bbox_diff[3] - bbox_diff[1]

                # Position at top right
                padding = 30
                box_padding = 12
                y_main = 60

                # Main timer box
                x_main = width - main_width - padding - box_padding * 2
                draw.rectangle(
                    [x_main - box_padding, y_main - box_padding,
                     x_main + main_width + box_padding, y_main + main_height + box_padding],
                    fill=(0, 0, 0, 180)
                )
                draw.text((x_main, y_main), timer_text, font=font_main, fill=(255, 255, 255, 255))

                # Comparison timer box (below main timer)
                y_diff = y_main + main_height + box_padding * 2 + 10
                # Make box wide enough for both label and diff
                total_diff_width = max(label_width, diff_width)
                x_diff = width - total_diff_width - padding - box_padding * 2

                # Draw box for label + diff
                box_height = label_height + diff_height + 8
                draw.rectangle(
                    [x_diff - box_padding, y_diff - box_padding,
                     x_diff + total_diff_width + box_padding, y_diff + box_height + box_padding],
                    fill=(0, 0, 0, 180)
                )

                # Draw label
                label_x = x_diff + (total_diff_width - label_width) // 2
                draw.text((label_x, y_diff), label_text, font=font_label, fill=(200, 200, 200, 255))

                # Draw difference time
                diff_x = x_diff + (total_diff_width - diff_width) // 2
                draw.text((diff_x, y_diff + label_height + 4), diff_text, font=font_diff, fill=diff_color)

                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
                img.save(frame_path, 'PNG')

            # Convert frames to video using ffmpeg
            timer_video = os.path.join(tempfile.gettempdir(), f"skiframes_timer_cmp_{os.getpid()}_{unique_id}.mov")
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%05d.png'),
                '-c:v', 'png',
                '-pix_fmt', 'rgba',
                timer_video
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Cleanup frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

            if result.returncode == 0:
                return timer_video
            else:
                print(f"    Comparison timer video creation failed: {result.stderr[:200]}")
                return None

        except Exception as e:
            print(f"    Comparison timer creation failed: {e}")
            return None

    def _add_overlay(self, input_path: str, output_path: str, racer: Racer, comparison_racer: Optional[Racer] = None) -> bool:
        """
        Add overlays to the video:
        - Title at bottom right
        - Timer at top right
        - Logos at bottom left

        Uses PIL for overlay images and M1 hardware encoding for speed.
        Returns True on success.
        """
        # Build title text from race_info
        # Format: "Western Division Ranking | SL | U14 | Run 1"
        event = self.race_info.get("event", "")
        discipline = self.race_info.get("discipline", "")
        age_group = self.race_info.get("age_group", "")
        run_str = self.race_info.get("run", "")

        # Build title parts (skip empty ones) - no date in title
        title_parts = []
        if event:
            title_parts.append(event)
        if discipline:
            title_parts.append(discipline)
        if age_group:
            title_parts.append(age_group)
        if run_str:
            title_parts.append(run_str)

        title_text = " | ".join(title_parts)

        # Fallback to old format if no race_info provided
        if not event and not discipline and not age_group:
            if self.race_title:
                title_text = self.race_title
            else:
                age_group_fallback = "U12" if "u12" in self.race_name.lower() else "U14" if "u14" in self.race_name.lower() else ""
                title_text = f"Western Divisional {age_group_fallback} Ranking - SL" if age_group_fallback else "Western Divisional Ranking - SL"

        # Unique ID for this racer's temp files (prevents conflicts between racers)
        racer_id = f"bib{racer.bib}" + ("_cmp" if comparison_racer else "")

        # Create title overlay (bottom right)
        title_overlay_path = self._create_title_overlay(title_text, racer_id=racer_id)

        # Create timer video overlay (top right)
        # Get video duration from input file, add 1 second buffer to ensure timer covers full video
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
        video_duration = (float(duration_result.stdout.strip()) if duration_result.returncode == 0 else racer.duration) + 1.0

        # Timer stops at racer's finish time, but video may be longer
        # Timer starts at -pre_buffer_sec and shows 0:00 when racer actually starts
        # Use comparison timer if comparing against fastest racer
        if comparison_racer:
            timer_video_path = self._create_comparison_timer_frames(
                racer.duration, video_duration,
                comparison_racer.duration, comparison_racer.bib,
                racer_id=racer_id,
                pre_buffer_sec=self.pre_buffer_sec
            )
        else:
            timer_video_path = self._create_timer_frames(
                racer.duration, video_duration, racer_id=racer_id,
                pre_buffer_sec=self.pre_buffer_sec
            )

        # Get logos to use - either user-selected or default
        existing_logos = []
        if self.selected_logos:
            # Use user-selected logos in specified order
            for logo_file in self.selected_logos:
                logo_path = self.logo_dir / logo_file
                if logo_path.exists():
                    existing_logos.append(str(logo_path))
                else:
                    print(f"    Warning: Logo not found: {logo_file}")
        else:
            # Default logos: US Ski & Snowboard first, Skiframes.com last
            default_logos = [
                'US-Ski-Snowboard.png',
                'NHARA_logo.png',
                'RMST_logo.png',
                'Ragged_logo.png',
                'Skiframes-com_logo.png'
            ]
            for logo_file in default_logos:
                logo_path = self.logo_dir / logo_file
                if logo_path.exists():
                    existing_logos.append(str(logo_path))

        print(f"    Overlays: title={title_overlay_path is not None}, timer={timer_video_path is not None}, logos={len(existing_logos)}")

        temp_files = []

        try:
            # Build ffmpeg command
            # IMPORTANT: Use -loop 1 BEFORE -i for static images to make them loop forever
            cmd = ['ffmpeg', '-y', '-i', input_path]
            input_idx = 1

            # Add timer video if created (it's already a video, no loop needed)
            if timer_video_path:
                cmd.extend(['-i', timer_video_path])
                temp_files.append(timer_video_path)

            # Add title overlay if created - use -loop 1 to repeat the static PNG
            if title_overlay_path:
                cmd.extend(['-loop', '1', '-i', title_overlay_path])
                temp_files.append(title_overlay_path)

            # Add logo inputs - use -loop 1 to repeat static PNGs
            for logo_path in existing_logos:
                cmd.extend(['-loop', '1', '-i', logo_path])

            # Build filter graph
            filter_parts = []
            prev_label = "0:v"
            current_idx = 1

            # Overlay timer video (top right - it's already positioned)
            # eof_action=pass: continue main video if timer ends (don't freeze)
            if timer_video_path:
                filter_parts.append(f"[{prev_label}][{current_idx}:v]overlay=x=0:y=0:eof_action=pass[v{current_idx}]")
                prev_label = f"v{current_idx}"
                current_idx += 1

            # Overlay title (bottom right - it's a full frame with positioned text)
            # shortest=1 ends when base video ends (image loops forever via -loop 1)
            if title_overlay_path:
                filter_parts.append(f"[{prev_label}][{current_idx}:v]overlay=x=0:y=0:shortest=1[v{current_idx}]")
                prev_label = f"v{current_idx}"
                current_idx += 1

            # Add logos at bottom left
            # shortest=1 ends when base video ends (images loop forever via -loop 1)
            # Scale logos to 60px height, calculate actual width for proper spacing
            x_offset = 20
            logo_height = 60
            logo_gap = 15  # Gap between logos
            for i, logo_path in enumerate(existing_logos):
                # Calculate actual scaled width for this logo
                try:
                    from PIL import Image
                    with Image.open(logo_path) as img:
                        orig_w, orig_h = img.size
                        scaled_w = int(orig_w * logo_height / orig_h)
                except:
                    scaled_w = 90  # Fallback

                scale_label = f"scaled{i}"
                out_label = f"v{current_idx}" if i < len(existing_logos) - 1 else "out"
                filter_parts.append(f"[{current_idx}:v]scale=-1:{logo_height}[{scale_label}]")
                filter_parts.append(f"[{prev_label}][{scale_label}]overlay=x={x_offset}:y=H-h-20:shortest=1[{out_label}]")
                prev_label = out_label
                current_idx += 1
                x_offset += scaled_w + logo_gap

            # Handle case where we have overlays but no logos
            if not existing_logos and (timer_video_path or title_overlay_path):
                filter_parts[-1] = filter_parts[-1].rsplit('[', 1)[0] + "[out]"
                prev_label = "out"

            if not filter_parts:
                return self._copy_without_overlay(input_path, output_path)

            filter_graph = ";".join(filter_parts)
            cmd.extend(['-filter_complex', filter_graph, '-map', '[out]' if prev_label == "out" else f'[{prev_label}]'])

            # Use hardware encoding based on platform
            cmd.extend(get_encoder_args())
            cmd.extend(['-movflags', '+faststart', output_path])

            print(f"    Running overlay composition...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                print(f"    Overlay error: {result.stderr[:300]}")
                # Fallback to simpler approach
                return self._add_overlay_simple(input_path, output_path, title_overlay_path, existing_logos)
            return True

        except subprocess.TimeoutExpired:
            print(f"    Overlay timeout")
            return False
        finally:
            # Cleanup temp files
            for tf in temp_files:
                if tf and os.path.exists(tf):
                    try:
                        os.unlink(tf)
                    except:
                        pass

    def _extract_ghost_segments(self, ghost_racer: Racer, current_racer: Racer, temp_dir: str) -> Optional[str]:
        """
        Extract video segments for the ghost racer at real speed.
        Ghost starts at same time as current racer but finishes earlier (since ghost is faster).
        Both racers use the SAME cut percentages applied to their OWN durations.

        Args:
            ghost_racer: The fastest racer to use as ghost
            current_racer: The current racer being processed
            temp_dir: Directory for intermediate files

        Returns:
            Path to the ghost video file, or None on failure
        """
        print(f"    Extracting ghost segments for Bib {ghost_racer.bib} (real speed, finishes earlier)...")

        # Extract each segment for the ghost racer using their actual timing
        ghost_segment_paths = []

        # Filter out skipped cuts (0% duration) to determine first/last active cut
        active_cuts = [(i, cut) for i, cut in enumerate(self.cuts) if cut.start_pct != cut.end_pct]

        for cut_idx, (i, cut) in enumerate(active_cuts):
            is_first_cut = (cut_idx == 0)
            is_last_cut = (cut_idx == len(active_cuts) - 1)

            try:
                # Calculate absolute time for ghost racer's segment
                # First cut: use pre_buffer_sec before ghost start
                # Last cut: use post_buffer_sec after ghost finish
                if is_first_cut:
                    segment_start_abs = ghost_racer.start_time_sec - self.pre_buffer_sec
                else:
                    segment_start_abs = ghost_racer.start_time_sec + (cut.start_pct * ghost_racer.duration)

                if is_last_cut:
                    segment_end_abs = ghost_racer.start_time_sec + ghost_racer.duration + self.post_buffer_sec
                else:
                    segment_end_abs = ghost_racer.start_time_sec + (cut.end_pct * ghost_racer.duration)

                segment_duration = segment_end_abs - segment_start_abs

                # Find the right video file for this time
                video_path, offset, video_end_sec = self._find_video_and_offset(cut.camera, segment_start_abs)

                if not video_path:
                    print(f"      Warning: No video found for ghost {cut.camera}")
                    continue

                # Handle offset adjustments
                if offset < 0:
                    segment_duration += offset
                    offset = 0
                    if segment_duration <= 0:
                        continue

                # Check if segment extends past video file
                if video_end_sec and segment_start_abs + segment_duration > video_end_sec:
                    available = video_end_sec - segment_start_abs
                    # Try to find continuation
                    next_video_path, _, _ = self._find_video_and_offset(cut.camera, video_end_sec + 0.1)
                    if next_video_path and next_video_path != video_path:
                        # Extract from both files - use both bib numbers for unique filenames in parallel
                        missing = segment_duration - available
                        first_path = os.path.join(temp_dir, f"ghost_bib{current_racer.bib}_g{ghost_racer.bib}_{i:02d}_{cut.camera}_p1.mp4")
                        second_path = os.path.join(temp_dir, f"ghost_bib{current_racer.bib}_g{ghost_racer.bib}_{i:02d}_{cut.camera}_p2.mp4")

                        self._extract_segment(video_path, offset, available, first_path)
                        self._extract_segment(next_video_path, 0.1, missing, second_path)

                        # Concatenate parts
                        segment_path = os.path.join(temp_dir, f"ghost_bib{current_racer.bib}_g{ghost_racer.bib}_{i:02d}_{cut.camera}.mp4")
                        if os.path.exists(first_path) and os.path.exists(second_path):
                            self._concatenate_segments([first_path, second_path], segment_path)
                            ghost_segment_paths.append(segment_path)
                        elif os.path.exists(first_path):
                            ghost_segment_paths.append(first_path)
                        continue
                    else:
                        segment_duration = available

                segment_path = os.path.join(temp_dir, f"ghost_bib{current_racer.bib}_g{ghost_racer.bib}_{i:02d}_{cut.camera}.mp4")
                if self._extract_segment(video_path, offset, segment_duration, segment_path):
                    ghost_segment_paths.append(segment_path)

            except Exception as e:
                print(f"      Error extracting ghost segment {cut.camera}: {e}")

        if not ghost_segment_paths:
            print(f"      No ghost segments extracted")
            return None

        # Concatenate ghost segments (no time scaling - ghost finishes at real speed)
        ghost_concat_path = os.path.join(temp_dir, f"ghost_concat_bib{current_racer.bib}_g{ghost_racer.bib}.mp4")
        if not self._concatenate_segments(ghost_segment_paths, ghost_concat_path):
            print(f"      Failed to concatenate ghost segments")
            return None

        time_diff = current_racer.duration - ghost_racer.duration
        print(f"      Ghost video ready - finishes {time_diff:.2f}s earlier than current racer")

        return ghost_concat_path

    def _composite_ghost_overlay(self, main_video: str, ghost_video: str, output_path: str, ghost_opacity: float = 0.4) -> bool:
        """
        Composite the ghost video behind the main video with reduced opacity.
        Ghost finishes earlier (since they're faster), then disappears while main video continues.

        Args:
            main_video: Path to the main racer's video
            ghost_video: Path to the ghost racer's video
            output_path: Path to save the composited video
            ghost_opacity: Opacity for the ghost (0.0-1.0), default 0.4 (40%)

        Returns:
            True on success
        """
        print(f"    Compositing ghost overlay (opacity={ghost_opacity})...")

        # Ghost video is shorter than main video (ghost finishes earlier)
        # Use eof_action=pass so when ghost ends, main video continues without overlay
        cmd = [
            'ffmpeg', '-y',
            '-i', main_video,  # Input 0: main video (longer)
            '-i', ghost_video,  # Input 1: ghost video (shorter - finishes earlier)
            '-filter_complex',
            # Tint ghost slightly blue and reduce opacity
            f'[1:v]colorbalance=bs=0.3:gs=-0.1:rs=-0.2,format=rgba,colorchannelmixer=aa={ghost_opacity}[ghost];'
            # Overlay ghost on main - eof_action=pass means continue main when ghost ends
            f'[0:v][ghost]overlay=x=0:y=0:eof_action=pass[out]',
            '-map', '[out]',
        ]
        # Use hardware encoding based on platform
        cmd.extend(get_encoder_args())
        cmd.extend(['-movflags', '+faststart', output_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                print(f"      Ghost composite failed: {result.stderr[:200]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print(f"      Ghost composite timeout")
            return False

    def _add_overlay_simple(self, input_path: str, output_path: str, title_overlay_path: Optional[str], logos: List[str]) -> bool:
        """Simpler overlay approach - one overlay at a time (fallback)."""
        current_input = input_path
        temp_files = []

        try:
            # Add title overlay first (full frame transparent PNG)
            # Use -loop 1 to repeat the static PNG forever, shortest=1 to end with video
            # Encoding args based on platform
            enc_args = get_encoder_args()

            if title_overlay_path and os.path.exists(title_overlay_path):
                temp_out = os.path.join(tempfile.gettempdir(), f"skiframes_t_{os.getpid()}.mp4")
                cmd = [
                    'ffmpeg', '-y', '-i', current_input, '-loop', '1', '-i', title_overlay_path,
                    '-filter_complex', '[0:v][1:v]overlay=x=0:y=0:shortest=1[out]',
                    '-map', '[out]',
                    *enc_args,
                    temp_out
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    temp_files.append(temp_out)
                    current_input = temp_out

            # Add logos one at a time
            # Use -loop 1 to repeat static PNGs forever, shortest=1 to end with video
            x_offset = 20
            logo_height = 60
            logo_gap = 15  # Gap between logos
            for i, logo_path in enumerate(logos):
                # Calculate actual scaled width for this logo
                try:
                    from PIL import Image
                    with Image.open(logo_path) as img:
                        orig_w, orig_h = img.size
                        scaled_w = int(orig_w * logo_height / orig_h)
                except:
                    scaled_w = 90  # Fallback

                temp_out = os.path.join(tempfile.gettempdir(), f"skiframes_l{i}_{os.getpid()}.mp4")
                cmd = [
                    'ffmpeg', '-y', '-i', current_input, '-loop', '1', '-i', logo_path,
                    '-filter_complex', f'[1:v]scale=-1:{logo_height}[logo];[0:v][logo]overlay=x={x_offset}:y=H-h-20:shortest=1[out]',
                    '-map', '[out]',
                    *enc_args,
                    temp_out
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    temp_files.append(temp_out)
                    current_input = temp_out
                x_offset += scaled_w + logo_gap

            # Copy final result to output
            if current_input != input_path:
                import shutil
                shutil.copy(current_input, output_path)
                return True
            else:
                return self._copy_without_overlay(input_path, output_path)

        finally:
            # Cleanup temp files
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.unlink(tf)
                except:
                    pass

    def _copy_without_overlay(self, input_path: str, output_path: str) -> bool:
        """Copy video without overlay (fallback)."""
        try:
            import shutil
            shutil.copy(input_path, output_path)
            return True
        except Exception:
            return False

    def _concatenate_segments(
        self,
        segment_paths: List[str],
        output_path: str
    ) -> bool:
        """
        Concatenate multiple video segments using ffmpeg concat demuxer.
        Re-encode to ensure smooth transitions without freezes at cut points.

        Returns True on success.
        """
        # Create concat file list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for path in segment_paths:
                # Escape single quotes in path
                escaped_path = path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
            concat_file = f.name

        try:
            # Re-encode during concat to fix timestamp issues that cause freezes
            # Use hardware encoding based on platform
            encoder_args = get_encoder_args()

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-vf', 'fps=30',  # Ensure consistent frame rate
                *encoder_args,
                '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for software encoding
            )
            return result.returncode == 0
        finally:
            os.unlink(concat_file)

    def _get_output_path(self, racer: Racer, comparison_racer: Optional[Racer] = None) -> Path:
        """
        Generate output path with proper folder structure and filename.

        Folder structure: {output_dir}/stitch_{date}_{race}/Team/Gender/U12_Run1/
        Filename: Name_Bib#_Run#_YYYY-MM-DD.mp4
        For comparison videos: Name_Bib#_Run#_YYYY-MM-DD_vs_Bib3.mp4
        """
        # Parse age group and run from race_name or race_info
        race_upper = self.race_name.upper()
        age_group = self.race_info.get("age_group", "")
        run_str = self.race_info.get("run", "")  # e.g., "Run 1" or "Run 2"

        # Fallback: parse from race_name if not in race_info
        if not age_group:
            if "U12" in race_upper:
                age_group = "U12"
            elif "U14" in race_upper:
                age_group = "U14"
            elif "U16" in race_upper:
                age_group = "U16"

        # Parse run number
        run_num = "Run1"
        if run_str:
            # Extract number from "Run 1", "Run 2", etc.
            run_match = re.search(r'(\d+)', run_str)
            if run_match:
                run_num = f"Run{run_match.group(1)}"
        elif "RUN_2" in race_upper or "RUN2" in race_upper or "RUN 2" in race_upper:
            run_num = "Run2"

        age_run = f"{age_group}_{run_num}" if age_group else run_num

        # Get date in condensed format YYYY-MM-DD
        # Try race_info date first, then event_date, then today
        date_str = self.race_info.get("date", "")
        if date_str:
            # Parse various date formats and convert to YYYY-MM-DD
            # Handle "Sunday, 2026/02/01" or "2026-02-01" or "02-01-2026"
            date_match = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', date_str)
            if date_match:
                date_condensed = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            else:
                date_match = re.search(r'(\d{2})[-/](\d{2})[-/](\d{4})', date_str)
                if date_match:
                    date_condensed = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
                else:
                    date_condensed = self.event_date or datetime.now().strftime("%Y-%m-%d")
        else:
            date_condensed = self.event_date or datetime.now().strftime("%Y-%m-%d")

        # Build folder path: Team/Gender/U12_Run1 (or U12_Run1_vs_Fastest for comparison)
        session_base = self.output_dir / f"stitch_{date_condensed}_{self.race_name}"

        folder_age_run = age_run
        if comparison_racer:
            folder_age_run = f"{age_run}_vs_Bib{comparison_racer.bib}"

        if racer.team and racer.gender:
            folder_path = session_base / racer.team / racer.gender / folder_age_run
        elif racer.team:
            folder_path = session_base / racer.team / folder_age_run
        else:
            folder_path = session_base / folder_age_run

        folder_path.mkdir(parents=True, exist_ok=True)

        # Build filename: Name_Bib#_Run#_YYYY-MM-DD.mp4
        # Replace spaces with underscores, remove other non-alphanumeric characters
        clean_name = re.sub(r'[^\w\s]', '', racer.name) if racer.name else "Racer"
        clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores

        if comparison_racer:
            filename = f"{clean_name}_Bib{racer.bib}_{run_num}_{date_condensed}_vs_Bib{comparison_racer.bib}.mp4"
        else:
            filename = f"{clean_name}_Bib{racer.bib}_{run_num}_{date_condensed}.mp4"

        return folder_path / filename

    def stitch_racer(self, racer: Racer, temp_dir: str, comparison_racer: Optional[Racer] = None) -> Optional[Path]:
        """
        Generate stitched video for one racer.

        Args:
            racer: Racer timing data
            temp_dir: Directory for intermediate files
            comparison_racer: Optional fastest racer to compare against

        Returns:
            Path to output video, or None on failure
        """
        if self._should_stop():
            return None

        output_path = self._get_output_path(racer, comparison_racer)

        # Extract each segment
        segment_paths = []
        # Filter out skipped cuts (0% duration) to determine first/last active cut
        active_cuts = [(i, cut) for i, cut in enumerate(self.cuts) if cut.start_pct != cut.end_pct]

        for cut_idx, (i, cut) in enumerate(active_cuts):
            if self._should_stop():
                return None

            is_first_cut = (cut_idx == 0)
            is_last_cut = (cut_idx == len(active_cuts) - 1)

            try:
                # Calculate absolute time for this segment
                # First cut: use pre_buffer_sec before racer start
                # Last cut: use post_buffer_sec after racer finish
                # Middle cuts: use percentage-based timing
                if is_first_cut:
                    segment_start_abs = racer.start_time_sec - self.pre_buffer_sec
                else:
                    segment_start_abs = racer.start_time_sec + (cut.start_pct * racer.duration)

                if is_last_cut:
                    segment_end_abs = racer.start_time_sec + racer.duration + self.post_buffer_sec
                else:
                    segment_end_abs = racer.start_time_sec + (cut.end_pct * racer.duration)

                segment_duration = segment_end_abs - segment_start_abs

                # Skip if duration is zero or negative
                if segment_duration <= 0:
                    continue

                # Find the right video file for this time
                video_path, offset, video_end_sec = self._find_video_and_offset(cut.camera, segment_start_abs)

                if not video_path:
                    print(f"  Warning: No video found for {cut.camera} at time {segment_start_abs:.0f}s")
                    continue

                # Handle case where segment starts before video
                if offset < 0:
                    print(f"  Warning: Segment {cut.camera} starts before video (offset={offset:.1f}s), adjusting...")
                    segment_duration += offset  # Reduce duration
                    offset = 0
                    if segment_duration <= 0:
                        print(f"  Skipping segment {cut.camera} - entirely before video start")
                        continue

                # Check if segment extends past end of video file
                segment_end_abs = segment_start_abs + segment_duration
                next_video_path = None
                next_video_duration = 0

                if video_end_sec and segment_end_abs > video_end_sec:
                    # Segment extends past this video file - check for next video
                    available_in_this_video = video_end_sec - segment_start_abs
                    missing_duration = segment_end_abs - video_end_sec
                    print(f"  Warning: {cut.camera} segment extends {missing_duration:.1f}s past video file end (video ends at {video_end_sec:.0f}s)")

                    # Try to find the next video file - first by time lookup, then by sequence
                    next_video_path, next_offset, _ = self._find_video_and_offset(cut.camera, video_end_sec + 0.1)

                    # If time-based lookup fails, try finding next video in sequence
                    if not next_video_path or next_video_path == video_path:
                        videos = self.camera_videos.get(cut.camera, [])
                        current_video = find_video_for_time(videos, segment_start_abs)
                        if current_video:
                            next_video = find_next_video(videos, current_video)
                            if next_video:
                                next_video_path = next_video.path
                                # Calculate offset: segment continues from where first video ended
                                next_offset = 0.0
                                print(f"    Found next video in sequence: {os.path.basename(next_video_path)} (starts at {next_video.start_sec:.0f}s)")

                    if next_video_path and next_video_path != video_path:
                        print(f"    Found continuation in next video file: {os.path.basename(next_video_path)}")
                        # Extract from first video only what's available
                        first_video_duration = available_in_this_video
                        next_video_duration = missing_duration
                        segment_duration = first_video_duration  # Will extract first part
                    else:
                        print(f"    No continuation video found - truncating segment to {available_in_this_video:.1f}s")
                        segment_duration = available_in_this_video

                # Use bib number in temp file names to avoid collisions in parallel processing
                segment_path = os.path.join(temp_dir, f"segment_bib{racer.bib}_{i:02d}_{cut.camera}.mp4")

                # Convert absolute time to HH:MM:SS for debugging
                abs_start_h = int(segment_start_abs // 3600)
                abs_start_m = int((segment_start_abs % 3600) // 60)
                abs_start_s = segment_start_abs % 60
                print(f"  Extracting {cut.camera}: abs_time={abs_start_h:02d}:{abs_start_m:02d}:{abs_start_s:04.1f}, offset={offset:.1f}s, duration={segment_duration:.1f}s from {os.path.basename(video_path)}")

                if next_video_path and next_video_duration > 0:
                    # Need to extract from two video files and concatenate them
                    first_segment_path = os.path.join(temp_dir, f"segment_bib{racer.bib}_{i:02d}_{cut.camera}_part1.mp4")
                    second_segment_path = os.path.join(temp_dir, f"segment_bib{racer.bib}_{i:02d}_{cut.camera}_part2.mp4")

                    print(f"    Extracting part 1: {segment_duration:.1f}s from {os.path.basename(video_path)}")
                    first_ok = self._extract_segment(video_path, offset, segment_duration, first_segment_path)

                    # Calculate correct offset for second video
                    # The second part should start where the first video ended
                    videos = self.camera_videos.get(cut.camera, [])
                    next_video_obj = find_video_for_time(videos, video_end_sec + 1)
                    if not next_video_obj:
                        # Try to find by path
                        for v in videos:
                            if v.path == next_video_path:
                                next_video_obj = v
                                break

                    if next_video_obj:
                        # Calculate where in the next video we should start
                        # This is where the first video ended, relative to the next video's start
                        second_offset = max(0, video_end_sec - next_video_obj.start_sec)
                        print(f"    Extracting part 2: {next_video_duration:.1f}s from {os.path.basename(next_video_path)} at offset {second_offset:.1f}s")
                    else:
                        second_offset = 0.1
                        print(f"    Extracting part 2: {next_video_duration:.1f}s from {os.path.basename(next_video_path)} at offset {second_offset:.1f}s (fallback)")

                    second_ok = self._extract_segment(next_video_path, second_offset, next_video_duration, second_segment_path)

                    if first_ok and second_ok:
                        # Concatenate the two parts
                        print(f"    Concatenating {cut.camera} parts...")
                        if self._concatenate_segments([first_segment_path, second_segment_path], segment_path):
                            segment_paths.append(segment_path)
                        else:
                            print(f"  Warning: Failed to concatenate {cut.camera} parts")
                    elif first_ok:
                        # Use just the first part
                        import shutil
                        shutil.copy(first_segment_path, segment_path)
                        segment_paths.append(segment_path)
                    else:
                        print(f"  Warning: Failed to extract segment from {cut.camera}")
                else:
                    # Normal single-file extraction
                    if self._extract_segment(video_path, offset, segment_duration, segment_path):
                        segment_paths.append(segment_path)
                    else:
                        print(f"  Warning: Failed to extract segment from {cut.camera}")

            except Exception as e:
                print(f"  Error processing {cut.camera}: {e}")

        if not segment_paths:
            print(f"  No segments extracted for {racer.name}")
            return None

        if self._should_stop():
            return None

        # Concatenate all segments
        print(f"  Concatenating {len(segment_paths)} segments...")
        concat_path = os.path.join(temp_dir, f"concat_{racer.bib}.mp4")
        if not self._concatenate_segments(segment_paths, concat_path):
            print(f"  Failed to concatenate segments for {racer.name}")
            return None

        if self._should_stop():
            return None

        # If comparison mode, extract ghost and composite it with main video
        video_for_overlay = concat_path
        if comparison_racer:
            # Extract ghost segments at real speed (ghost finishes earlier since they're faster)
            ghost_video_path = self._extract_ghost_segments(comparison_racer, racer, temp_dir)

            if ghost_video_path and os.path.exists(ghost_video_path):
                # Composite ghost behind main video
                ghost_composite_path = os.path.join(temp_dir, f"ghost_composite_{racer.bib}.mp4")
                if self._composite_ghost_overlay(concat_path, ghost_video_path, ghost_composite_path, ghost_opacity=0.35):
                    video_for_overlay = ghost_composite_path
                    print(f"  Ghost overlay applied successfully")
                else:
                    print(f"  Warning: Ghost overlay failed, continuing without ghost")
            else:
                print(f"  Warning: Could not extract ghost segments, continuing without ghost")

        if self._should_stop():
            return None

        # Add overlay (title + logos + comparison timer if applicable)
        if comparison_racer:
            print(f"  Adding overlay (vs Bib {comparison_racer.bib})...")
        else:
            print(f"  Adding overlay...")
        if self._add_overlay(video_for_overlay, str(output_path), racer, comparison_racer):
            return output_path
        else:
            # Fallback: copy without overlay
            print(f"  Warning: Overlay failed, copying without overlay")
            if self._copy_without_overlay(video_for_overlay, str(output_path)):
                return output_path
            return None

    def _get_condensed_date(self) -> str:
        """Get date in condensed YYYY-MM-DD format from race_info or event_date."""
        date_str = self.race_info.get("date", "")
        if date_str:
            # Parse various date formats and convert to YYYY-MM-DD
            date_match = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', date_str)
            if date_match:
                return f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            date_match = re.search(r'(\d{2})[-/](\d{2})[-/](\d{4})', date_str)
            if date_match:
                return f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
        return self.event_date or datetime.now().strftime("%Y-%m-%d")

    def _generate_manifest(self, output_paths: List[Path], fastest_by_gender: Optional[Dict[str, 'Racer']] = None) -> Path:
        """
        Generate a manifest.json file for the stitched videos.

        Args:
            output_paths: List of generated video paths
            fastest_by_gender: Optional dict of fastest racers by gender (for comparison info)

        Returns:
            Path to the manifest file
        """
        date_condensed = self._get_condensed_date()

        # Find the session directory (parent of Team folders or direct parent)
        if output_paths:
            # Walk up to find the stitch_* directory
            session_dir = output_paths[0].parent
            while session_dir.name and not session_dir.name.startswith('stitch_'):
                session_dir = session_dir.parent
        else:
            session_dir = self.output_dir / f"stitch_{date_condensed}_{self.race_name}"

        session_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute rankings per gender (bib -> rank)
        bib_to_rank: Dict[int, int] = {}
        for gender in set(r.gender for r in self.racers if r.gender):
            gender_racers = [r for r in self.racers if r.gender == gender and r.status == 'finished']
            gender_racers.sort(key=lambda r: r.duration)
            for i, r in enumerate(gender_racers):
                bib_to_rank[r.bib] = i + 1

        # Build video entries
        videos = []
        for path in output_paths:
            # Find matching racer by parsing filename
            filename = path.stem  # e.g., "JohnDoe_Bib24" or "JohnDoe_Bib24_vs_Bib3"
            is_comparison = "_vs_Bib" in filename

            # Extract bib number
            bib_match = re.search(r'_Bib(\d+)', filename)
            bib = int(bib_match.group(1)) if bib_match else 0

            # Find racer
            racer = next((r for r in self.racers if r.bib == bib), None)

            # Extract comparison bib if present
            comparison_bib = None
            if is_comparison:
                cmp_match = re.search(r'_vs_Bib(\d+)', filename)
                comparison_bib = int(cmp_match.group(1)) if cmp_match else None

            video_entry = {
                "filename": path.name,
                "path": str(path.relative_to(session_dir)) if path.is_relative_to(session_dir) else str(path),
                "bib": bib,
                "name": racer.name if racer else "",
                "team": racer.team if racer else "",
                "gender": racer.gender if racer else "",
                "rank": bib_to_rank.get(bib),  # None for DSQ/DNF/DNS
                "duration": round(racer.duration, 2) if racer else 0,
                "ussa_id": racer.ussa_id if racer else "",
                "ussa_profile_url": racer.ussa_profile_url if racer else None,
                "status": racer.status if racer else "finished",
                "is_comparison": is_comparison,
                "comparison_bib": comparison_bib,
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0,
            }
            videos.append(video_entry)

        # Build fastest racer info
        fastest_info = {}
        if fastest_by_gender:
            for gender, racer in fastest_by_gender.items():
                fastest_info[gender] = {
                    "bib": racer.bib,
                    "name": racer.name,
                    "ussa_id": racer.ussa_id,
                    "ussa_profile_url": racer.ussa_profile_url,
                    "duration": round(racer.duration, 2),
                }

        # Compute rankings per gender (only for finished racers)
        rankings_by_gender = {}
        for gender in set(r.gender for r in self.racers if r.gender):
            gender_racers = [r for r in self.racers if r.gender == gender and r.status == 'finished']
            gender_racers.sort(key=lambda r: r.duration)
            rankings_by_gender[gender] = [
                {
                    "rank": i + 1,
                    "bib": r.bib,
                    "name": r.name,
                    "team": r.team,
                    "ussa_id": r.ussa_id,
                    "ussa_profile_url": r.ussa_profile_url,
                    "duration": round(r.duration, 2),
                    "gap": round(r.duration - gender_racers[0].duration, 2) if i > 0 else 0,
                }
                for i, r in enumerate(gender_racers)
            ]

        # Add DSQ/DNF racers to rankings (unranked) - DNS excluded as not relevant
        dsq_dnf_by_gender = {}
        for gender in set(r.gender for r in self.racers if r.gender):
            dsq_dnf = [r for r in self.racers if r.gender == gender and r.status in ('DSQ', 'DNF')]
            if dsq_dnf:
                dsq_dnf_by_gender[gender] = [
                    {
                        "bib": r.bib,
                        "name": r.name,
                        "team": r.team,
                        "ussa_id": r.ussa_id,
                        "ussa_profile_url": r.ussa_profile_url,
                        "status": r.status,
                    }
                    for r in dsq_dnf
                ]

        # Build structured manifest
        manifest = {
            "session_id": f"stitch_{date_condensed}_{self.race_name}",
            "created_at": datetime.now().isoformat(),

            # Race information
            "race": {
                "event": self.race_info.get("event", self.race_title or ""),
                "discipline": self.race_info.get("discipline", ""),
                "age_group": self.race_info.get("age_group", ""),
                "run": self.race_info.get("run", ""),
                "date": date_condensed,
                "date_display": self.race_info.get("date", date_condensed),  # Original format for display
                "type": self.race_info.get("type", ""),
            },

            # Course information
            "course": {
                "name": self.race_info.get("course", ""),
                "location": self.race_info.get("location", ""),
                "vertical_drop": self.race_info.get("vertical_drop", ""),
                "length": self.race_info.get("length", ""),
                "gates": self.race_info.get("gates", ""),
                "snow": self.race_info.get("snow", ""),
            },

            # Processing summary
            "summary": {
                "total_racers": len([r for r in self.racers if r.status != 'DNS']),  # Exclude DNS
                "total_videos": len(videos),
                "finished": len([r for r in self.racers if r.status == 'finished']),
                "dsq": len([r for r in self.racers if r.status == 'DSQ']),
                "dnf": len([r for r in self.racers if r.status == 'DNF']),
                "fastest_by_gender": fastest_info,
            },

            # Rankings per gender (searchable)
            "rankings": {
                "by_gender": rankings_by_gender,
                "dsq_dnf": dsq_dnf_by_gender,
            },

            # Technical details
            "processing": {
                "cuts": [{"camera": c.camera, "start_pct": c.start_pct, "end_pct": c.end_pct} for c in self.cuts],
            },

            # Video list
            "videos": videos,
        }

        manifest_path = session_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\nManifest created: {manifest_path}")
        return manifest_path

    def process_all(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Path]:
        """
        Process all racers.

        Args:
            progress_callback: Called with (current_index, total, racer_name)

        Returns:
            List of output video paths
        """
        output_paths = []
        total = len(self.racers)

        # Create a persistent temp directory for the session
        with tempfile.TemporaryDirectory(prefix='skiframes_stitch_') as temp_dir:
            for i, racer in enumerate(self.racers):
                if self._should_stop():
                    print("\nProcessing stopped by user")
                    break

                if progress_callback:
                    progress_callback(i + 1, total, racer.name)

                print(f"\nProcessing {i+1}/{total}: {racer.name} (bib {racer.bib})")
                print(f"  Start: {racer.start_time_str}, Duration: {racer.duration:.2f}s, Finish: {racer.finish_time_sec:.1f}s")
                print(f"  Team: {racer.team}, Gender: {racer.gender}")

                # Show cut points for debugging timing issues
                for cut in self.cuts:
                    cut_start = racer.start_time_sec + (cut.start_pct * racer.duration)
                    cut_end = racer.start_time_sec + (cut.end_pct * racer.duration)
                    cut_dur = cut_end - cut_start
                    print(f"    {cut.camera}: {cut.start_pct*100:.0f}%-{cut.end_pct*100:.0f}% -> {cut_dur:.1f}s")

                output_path = self.stitch_racer(racer, temp_dir)
                if output_path:
                    output_paths.append(output_path)
                    print(f"  Output: {output_path}")

        # Generate manifest
        if output_paths:
            self._generate_manifest(output_paths)

        return output_paths

    def process_all_with_comparison(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        generate_comparison: bool = True
    ) -> List[Path]:
        """
        Process all racers, generating both regular videos and comparison videos
        against the fastest racer in each gender category.

        Args:
            progress_callback: Called with (current_index, total, racer_name)
            generate_comparison: If True, also generate comparison videos vs fastest

        Returns:
            List of output video paths
        """
        output_paths = []

        # Find fastest racer per gender
        fastest_by_gender: Dict[str, Racer] = {}
        for racer in self.racers:
            gender = racer.gender or "Unknown"
            if gender not in fastest_by_gender or racer.duration < fastest_by_gender[gender].duration:
                fastest_by_gender[gender] = racer

        print(f"\nFastest racers by gender:")
        for gender, fastest in fastest_by_gender.items():
            print(f"  {gender}: Bib {fastest.bib} ({fastest.name}) - {fastest.duration:.2f}s")

        # Calculate total videos to process
        total_regular = len(self.racers)
        total_comparison = len(self.racers) if generate_comparison else 0
        # Don't generate comparison for the fastest racer themselves
        if generate_comparison:
            total_comparison -= len(fastest_by_gender)
        total = total_regular + total_comparison

        current = 0

        # Create a persistent temp directory for the session
        with tempfile.TemporaryDirectory(prefix='skiframes_stitch_') as temp_dir:
            # First pass: generate regular videos
            print(f"\n=== Generating regular videos ({total_regular} racers) ===")
            for i, racer in enumerate(self.racers):
                if self._should_stop():
                    print("\nProcessing stopped by user")
                    break

                current += 1
                if progress_callback:
                    progress_callback(current, total, f"{racer.name} (regular)")

                print(f"\nProcessing {i+1}/{total_regular}: {racer.name} (bib {racer.bib})")
                print(f"  Start: {racer.start_time_str}, Duration: {racer.duration:.2f}s, Finish: {racer.finish_time_sec:.1f}s")
                print(f"  Team: {racer.team}, Gender: {racer.gender}")

                # Show cut points for debugging timing issues
                for cut in self.cuts:
                    cut_start = racer.start_time_sec + (cut.start_pct * racer.duration)
                    cut_end = racer.start_time_sec + (cut.end_pct * racer.duration)
                    cut_dur = cut_end - cut_start
                    print(f"    {cut.camera}: {cut.start_pct*100:.0f}%-{cut.end_pct*100:.0f}% -> {cut_dur:.1f}s")

                output_path = self.stitch_racer(racer, temp_dir)
                if output_path:
                    output_paths.append(output_path)
                    print(f"  Output: {output_path}")

            # Second pass: generate comparison videos
            if generate_comparison and not self._should_stop():
                print(f"\n=== Generating comparison videos ===")
                comparison_count = 0
                for i, racer in enumerate(self.racers):
                    if self._should_stop():
                        print("\nProcessing stopped by user")
                        break

                    gender = racer.gender or "Unknown"
                    fastest = fastest_by_gender.get(gender)

                    # Skip if this IS the fastest racer (no point comparing to self)
                    if fastest and racer.bib == fastest.bib:
                        print(f"\nSkipping Bib {racer.bib} ({racer.name}) - is fastest in {gender}")
                        continue

                    comparison_count += 1
                    current += 1
                    if progress_callback:
                        progress_callback(current, total, f"{racer.name} (vs Bib {fastest.bib})")

                    print(f"\nProcessing comparison {comparison_count}: {racer.name} (bib {racer.bib}) vs Bib {fastest.bib}")
                    print(f"  Time diff: {racer.duration - fastest.duration:+.2f}s")

                    output_path = self.stitch_racer(racer, temp_dir, comparison_racer=fastest)
                    if output_path:
                        output_paths.append(output_path)
                        print(f"  Output: {output_path}")

        # Generate manifest with fastest racer info
        if output_paths:
            self._generate_manifest(output_paths, fastest_by_gender)

        return output_paths

    def _process_single_racer(
        self,
        racer: Racer,
        temp_dir: str,
        comparison_racer: Optional[Racer] = None
    ) -> Tuple[Optional[Path], str]:
        """
        Process a single racer (thread-safe helper for parallel processing).

        Returns:
            Tuple of (output_path or None, status_message)
        """
        try:
            label = f"{racer.name} (bib {racer.bib})"
            if comparison_racer:
                label += f" vs Bib {comparison_racer.bib}"

            output_path = self.stitch_racer(racer, temp_dir, comparison_racer)
            if output_path:
                return (output_path, f"✓ {label}")
            else:
                return (None, f"✗ {label} - failed")
        except Exception as e:
            return (None, f"✗ {racer.name} (bib {racer.bib}) - error: {e}")

    def process_all_parallel(
        self,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Path]:
        """
        Process all racers in parallel for faster throughput on M1 Mac.

        Uses ThreadPoolExecutor to run multiple ffmpeg processes concurrently.
        M1 Max can handle 4-6 concurrent video processing tasks efficiently.

        Args:
            max_workers: Number of concurrent workers (default 4 for M1 Max)
            progress_callback: Called with (current_index, total, racer_name)

        Returns:
            List of output video paths
        """
        output_paths = []
        total = len(self.racers)
        completed = 0
        lock = threading.Lock()

        print(f"\n=== Parallel processing with {max_workers} workers ===")
        print(f"Total racers: {total}")

        # Create a persistent temp directory for the session
        with tempfile.TemporaryDirectory(prefix='skiframes_stitch_parallel_') as temp_dir:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_racer = {
                    executor.submit(self._process_single_racer, racer, temp_dir, None): racer
                    for racer in self.racers
                }

                # Process results as they complete
                for future in as_completed(future_to_racer):
                    if self._should_stop():
                        print("\nProcessing stopped by user - cancelling remaining tasks")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    racer = future_to_racer[future]
                    result_path, status = future.result()

                    with lock:
                        completed += 1
                        if result_path:
                            output_paths.append(result_path)

                        if progress_callback:
                            progress_callback(completed, total, racer.name)

                        print(f"[{completed}/{total}] {status}")

        # Generate manifest
        if output_paths:
            self._generate_manifest(output_paths)

        print(f"\n=== Completed: {len(output_paths)}/{total} videos ===")
        return output_paths

    def process_all_with_comparison_parallel(
        self,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        generate_comparison: bool = True
    ) -> List[Path]:
        """
        Process all racers in parallel, generating both regular and comparison videos.

        Two-phase approach:
        1. First process all regular videos in parallel
        2. Then process all comparison videos in parallel

        This ensures fastest racers are identified before comparisons.

        Args:
            max_workers: Number of concurrent workers (default 4 for M1 Max)
            progress_callback: Called with (current_index, total, racer_name)
            generate_comparison: If True, also generate comparison videos vs fastest

        Returns:
            List of output video paths
        """
        output_paths = []
        lock = threading.Lock()

        # Find fastest racer per gender
        fastest_by_gender: Dict[str, Racer] = {}
        for racer in self.racers:
            if racer.status != 'finished':
                continue  # Skip DSQ/DNF/DNS for fastest calculation
            gender = racer.gender or "Unknown"
            if gender not in fastest_by_gender or racer.duration < fastest_by_gender[gender].duration:
                fastest_by_gender[gender] = racer

        print(f"\nFastest racers by gender:")
        for gender, fastest in fastest_by_gender.items():
            print(f"  {gender}: Bib {fastest.bib} ({fastest.name}) - {fastest.duration:.2f}s")

        # Calculate total videos
        total_regular = len(self.racers)
        total_comparison = 0
        if generate_comparison:
            for racer in self.racers:
                gender = racer.gender or "Unknown"
                fastest = fastest_by_gender.get(gender)
                if fastest and racer.bib != fastest.bib:
                    total_comparison += 1

        total = total_regular + total_comparison
        completed = 0

        print(f"\n=== Parallel processing with {max_workers} workers ===")
        print(f"Regular videos: {total_regular}")
        print(f"Comparison videos: {total_comparison}")
        print(f"Total: {total}")

        # Create a persistent temp directory for the session
        with tempfile.TemporaryDirectory(prefix='skiframes_stitch_parallel_') as temp_dir:

            # Phase 1: Process regular videos
            print(f"\n--- Phase 1: Regular videos ---")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_racer = {
                    executor.submit(self._process_single_racer, racer, temp_dir, None): racer
                    for racer in self.racers
                }

                for future in as_completed(future_to_racer):
                    if self._should_stop():
                        print("\nProcessing stopped by user")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    racer = future_to_racer[future]
                    result_path, status = future.result()

                    with lock:
                        completed += 1
                        if result_path:
                            output_paths.append(result_path)

                        if progress_callback:
                            progress_callback(completed, total, f"{racer.name} (regular)")

                        print(f"[{completed}/{total}] {status}")

            # Phase 2: Process comparison videos
            if generate_comparison and not self._should_stop():
                print(f"\n--- Phase 2: Comparison videos ---")

                # Build list of comparison tasks
                comparison_tasks = []
                for racer in self.racers:
                    gender = racer.gender or "Unknown"
                    fastest = fastest_by_gender.get(gender)
                    if fastest and racer.bib != fastest.bib:
                        comparison_tasks.append((racer, fastest))

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(self._process_single_racer, racer, temp_dir, fastest): (racer, fastest)
                        for racer, fastest in comparison_tasks
                    }

                    for future in as_completed(future_to_task):
                        if self._should_stop():
                            print("\nProcessing stopped by user")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                        racer, fastest = future_to_task[future]
                        result_path, status = future.result()

                        with lock:
                            completed += 1
                            if result_path:
                                output_paths.append(result_path)

                            if progress_callback:
                                progress_callback(completed, total, f"{racer.name} (vs Bib {fastest.bib})")

                            print(f"[{completed}/{total}] {status}")

        # Generate manifest with fastest racer info
        if output_paths:
            self._generate_manifest(output_paths, fastest_by_gender)

        print(f"\n=== Completed: {len(output_paths)}/{total} videos ===")
        return output_paths


def load_stitch_config(config_path: str) -> Dict:
    """Load a stitch configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_stitch_config(config: Dict, config_path: str):
    """Save a stitch configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def racers_from_vola_data(vola_racers: List[Dict]) -> List[Racer]:
    """
    Convert Vola racer dicts to Racer objects.

    Expected dict format from existing Vola parsing:
    {
        'bib': 24,
        'name': 'JohnDoe',
        'team': 'SUN',
        'gender': 'Women',
        'start_time_sec': 36072.0,  # seconds since midnight
        'run_duration': 43.27,
        'finished': True
    }
    """
    racers = []
    for r in vola_racers:
        if not r.get('finished', True):
            continue  # Skip DNF/DNS

        start_sec = r.get('start_time_sec', 0)
        duration = r.get('run_duration', 0)

        if duration <= 0:
            continue  # Skip invalid entries

        racers.append(Racer(
            bib=r.get('bib', 0),
            name=r.get('name', ''),
            team=r.get('team', ''),
            gender=r.get('gender', ''),
            start_time_sec=start_sec,
            finish_time_sec=start_sec + duration,
            duration=duration
        ))

    return racers


# CLI entry point for testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stitcher.py <config.json>")
        print("\nConfig format:")
        print(json.dumps({
            "race_name": "u12_run1",
            "output_dir": "./output",
            "race_title": "Western Division U12 Ranking - SL",
            "cuts": [
                {"camera": "R1", "start_pct": -0.05, "end_pct": 0.05},
                {"camera": "Axis", "start_pct": 0.05, "end_pct": 0.425},
                {"camera": "R2", "start_pct": 0.425, "end_pct": 0.575},
                {"camera": "R3", "start_pct": 0.575, "end_pct": 1.05}
            ],
            "video_paths": {
                "R1": ["/path/to/r1_video1.mp4", "/path/to/r1_video2.mp4"],
                "R2": ["/path/to/r2_video.mp4"],
                "Axis": ["/path/to/axis_video.mkv"],
                "R3": ["/path/to/r3_video.mp4"]
            },
            "racers": [
                {
                    "bib": 24,
                    "name": "JohnDoe",
                    "team": "SUN",
                    "gender": "Women",
                    "start_time_sec": 36072.0,
                    "run_duration": 43.27,
                    "finished": True
                }
            ]
        }, indent=2))
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_stitch_config(config_path)

    # Convert config to objects
    cuts = [CameraCut(**c) for c in config['cuts']]
    racers = racers_from_vola_data(config['racers'])

    print(f"Loaded {len(racers)} racers, {len(cuts)} camera cuts")

    stitcher = VideoStitcher(
        racers=racers,
        cuts=cuts,
        video_paths=config['video_paths'],
        output_dir=config.get('output_dir', './output'),
        race_name=config.get('race_name', 'race'),
        race_title=config.get('race_title', '')
    )

    def progress(current, total, name):
        print(f"[{current}/{total}] Processing: {name}")

    outputs = stitcher.process_all(progress_callback=progress)

    print(f"\nCompleted! Generated {len(outputs)} videos:")
    for p in outputs:
        print(f"  {p}")

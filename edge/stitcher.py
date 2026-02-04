#!/usr/bin/env python3
"""
Video stitcher for multi-camera ski race footage.
Creates broadcast-style sequential cuts from 4 camera angles into a single video per athlete.
"""

import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass


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

    @property
    def start_time_str(self) -> str:
        """Format start time as HH:MM:SS."""
        h = int(self.start_time_sec // 3600)
        m = int((self.start_time_sec % 3600) // 60)
        s = int(self.start_time_sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


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
        stop_flag: Optional[Callable[[], bool]] = None
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
            stop_flag: Optional callable that returns True if processing should stop
        """
        self.racers = racers
        self.cuts = sorted(cuts, key=lambda c: c.start_pct)  # Ensure chronological order
        self.output_dir = Path(output_dir)
        self.race_name = race_name
        self.logo_dir = Path(logo_dir) if logo_dir else Path(__file__).parent.parent / 'logos'
        self.race_title = race_title
        self.stop_flag = stop_flag

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

    def _should_stop(self) -> bool:
        """Check if processing should stop."""
        if self.stop_flag:
            return self.stop_flag()
        return False

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
            '-c:v', 'h264_videotoolbox',  # M1 hardware acceleration
            '-b:v', '8M',  # Bitrate for hardware encoder (no CRF support)
            '-movflags', '+faststart',
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

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
            return True
        except subprocess.TimeoutExpired:
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
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-movflags', '+faststart',
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                print(f"    ffmpeg error: {result.stderr[:200]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _get_font(self, size: int = 24):
        """Get a font for PIL drawing."""
        from PIL import ImageFont
        font_paths = [
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/HelveticaNeue.ttc',
            '/Library/Fonts/Arial.ttf',
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except:
                    continue
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

    def _create_timer_frames(self, run_duration_sec: float, video_duration_sec: float, racer_id: str = "", fps: int = 30, width: int = 1920, height: int = 1080) -> Optional[str]:
        """
        Create a video with timer overlay (top right) using PIL frames.
        Timer stops at run_duration_sec (finish time) even if video continues.
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

            print(f"    Creating {total_frames} timer frames (stops at {run_duration_sec:.1f}s)...")

            for frame_num in range(total_frames):
                # Calculate time for this frame - stop at run_duration
                time_sec = min(frame_num / fps, run_duration_sec)
                minutes = int(time_sec // 60)
                seconds = time_sec % 60

                # Format as M:SS.s - truncate (not round) the decimal
                # Use int() on tenths to truncate instead of round
                whole_sec = int(seconds)
                tenths = int((seconds - whole_sec) * 10)  # Truncate, not round
                timer_text = f"{minutes}:{whole_sec:02d}.{tenths}"

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

    def _add_overlay(self, input_path: str, output_path: str, racer: Racer) -> bool:
        """
        Add overlays to the video:
        - Title at bottom right
        - Timer at top right
        - Logos at bottom left

        Uses PIL for overlay images and M1 hardware encoding for speed.
        Returns True on success.
        """
        # Build title text
        date_str = datetime.now().strftime("%Y-%m-%d")

        if self.race_title:
            title_text = f"Skiframes.com | {self.race_title} | {date_str}"
        else:
            age_group = "U12" if "u12" in self.race_name.lower() else "U14" if "u14" in self.race_name.lower() else ""
            default_title = f"Western Divisional {age_group} Ranking - SL" if age_group else "Western Divisional Ranking - SL"
            title_text = f"Skiframes.com | {default_title} | {date_str}"

        # Unique ID for this racer's temp files (prevents conflicts between racers)
        racer_id = f"bib{racer.bib}"

        # Create title overlay (bottom right)
        title_overlay_path = self._create_title_overlay(title_text, racer_id=racer_id)

        # Create timer video overlay (top right)
        # Get video duration from input file, add 1 second buffer to ensure timer covers full video
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
        video_duration = (float(duration_result.stdout.strip()) if duration_result.returncode == 0 else racer.duration) + 1.0

        # Timer stops at racer's finish time, but video may be longer
        timer_video_path = self._create_timer_frames(racer.duration, video_duration, racer_id=racer_id)

        # Check which logos exist
        logo_files = ['NHARA_logo.png', 'RMST_logo.png', 'Ragged_logo.png']
        existing_logos = []
        for logo_file in logo_files:
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
            x_offset = 20
            for i, logo_path in enumerate(existing_logos):
                scale_label = f"scaled{i}"
                out_label = f"v{current_idx}" if i < len(existing_logos) - 1 else "out"
                filter_parts.append(f"[{current_idx}:v]scale=-1:80[{scale_label}]")  # 120px height (doubled from 60)
                filter_parts.append(f"[{prev_label}][{scale_label}]overlay=x={x_offset}:y=H-h-20:shortest=1[{out_label}]")
                prev_label = out_label
                current_idx += 1
                x_offset += 100  # Increased spacing for larger logos

            # Handle case where we have overlays but no logos
            if not existing_logos and (timer_video_path or title_overlay_path):
                filter_parts[-1] = filter_parts[-1].rsplit('[', 1)[0] + "[out]"
                prev_label = "out"

            if not filter_parts:
                return self._copy_without_overlay(input_path, output_path)

            filter_graph = ";".join(filter_parts)
            cmd.extend(['-filter_complex', filter_graph, '-map', '[out]' if prev_label == "out" else f'[{prev_label}]'])

            # Use hardware encoding
            cmd.extend([
                '-c:v', 'h264_videotoolbox',
                '-b:v', '8M',
                '-movflags', '+faststart',
                output_path
            ])

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

    def _add_overlay_simple(self, input_path: str, output_path: str, title_overlay_path: Optional[str], logos: List[str]) -> bool:
        """Simpler overlay approach - one overlay at a time (fallback)."""
        current_input = input_path
        temp_files = []

        try:
            # Add title overlay first (full frame transparent PNG)
            # Use -loop 1 to repeat the static PNG forever, shortest=1 to end with video
            if title_overlay_path and os.path.exists(title_overlay_path):
                temp_out = os.path.join(tempfile.gettempdir(), f"skiframes_t_{os.getpid()}.mp4")
                cmd = [
                    'ffmpeg', '-y', '-i', current_input, '-loop', '1', '-i', title_overlay_path,
                    '-filter_complex', '[0:v][1:v]overlay=x=0:y=0:shortest=1[out]',
                    '-map', '[out]',
                    '-c:v', 'h264_videotoolbox', '-b:v', '8M',
                    temp_out
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    temp_files.append(temp_out)
                    current_input = temp_out

            # Add logos one at a time
            # Use -loop 1 to repeat static PNGs forever, shortest=1 to end with video
            x_offset = 20
            for i, logo_path in enumerate(logos):
                temp_out = os.path.join(tempfile.gettempdir(), f"skiframes_l{i}_{os.getpid()}.mp4")
                cmd = [
                    'ffmpeg', '-y', '-i', current_input, '-loop', '1', '-i', logo_path,
                    '-filter_complex', f'[1:v]scale=-1:80[logo];[0:v][logo]overlay=x={x_offset}:y=H-h-20:shortest=1[out]',
                    '-map', '[out]',
                    '-c:v', 'h264_videotoolbox', '-b:v', '8M',
                    temp_out
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    temp_files.append(temp_out)
                    current_input = temp_out
                x_offset += 80

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
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-vf', 'fps=30',  # Ensure consistent frame rate
                '-c:v', 'h264_videotoolbox',  # Hardware encoding
                '-b:v', '8M',
                '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        finally:
            os.unlink(concat_file)

    def _get_output_path(self, racer: Racer) -> Path:
        """
        Generate output path with proper folder structure and filename.

        Folder structure: {output_dir}/stitch_{date}_{race}/Team/Gender/U12_Run1/
        Filename: Name_bib#_U12_Run1.mp4
        """
        # Parse age group and run from race_name (e.g., "u12_run_1" -> "U12", "Run1")
        race_upper = self.race_name.upper()
        age_group = ""
        run_num = "Run1"

        if "U12" in race_upper:
            age_group = "U12"
        elif "U14" in race_upper:
            age_group = "U14"
        elif "U16" in race_upper:
            age_group = "U16"

        if "RUN_2" in race_upper or "RUN2" in race_upper:
            run_num = "Run2"

        age_run = f"{age_group}_{run_num}" if age_group else run_num

        # Build folder path: Team/Gender/U12_Run1
        session_base = self.output_dir / f"stitch_{datetime.now().strftime('%Y-%m-%d')}_{self.race_name}"

        if racer.team and racer.gender:
            folder_path = session_base / racer.team / racer.gender / age_run
        elif racer.team:
            folder_path = session_base / racer.team / age_run
        else:
            folder_path = session_base / age_run

        folder_path.mkdir(parents=True, exist_ok=True)

        # Build filename: Name_Bib#_U12.mp4
        # Clean name - remove any non-alphanumeric characters
        clean_name = re.sub(r'[^\w]', '', racer.name) if racer.name else f"Racer"
        filename = f"{clean_name}_Bib{racer.bib}_{age_group}.mp4"

        return folder_path / filename

    def stitch_racer(self, racer: Racer, temp_dir: str) -> Optional[Path]:
        """
        Generate stitched video for one racer.

        Args:
            racer: Racer timing data
            temp_dir: Directory for intermediate files

        Returns:
            Path to output video, or None on failure
        """
        if self._should_stop():
            return None

        output_path = self._get_output_path(racer)

        # Extract each segment
        segment_paths = []
        for i, cut in enumerate(self.cuts):
            if self._should_stop():
                return None

            try:
                # Calculate absolute time for this segment
                segment_start_abs = racer.start_time_sec + (cut.start_pct * racer.duration)
                segment_end_abs = racer.start_time_sec + (cut.end_pct * racer.duration)
                segment_duration = segment_end_abs - segment_start_abs

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

                    # Try to find the next video file for the remaining time
                    next_video_path, next_offset, _ = self._find_video_and_offset(cut.camera, video_end_sec + 0.1)
                    if next_video_path and next_video_path != video_path:
                        print(f"    Found continuation in next video file: {os.path.basename(next_video_path)}")
                        # Extract from first video only what's available
                        first_video_duration = available_in_this_video
                        next_video_duration = missing_duration
                        segment_duration = first_video_duration  # Will extract first part
                    else:
                        print(f"    No continuation video found - truncating segment to {available_in_this_video:.1f}s")
                        segment_duration = available_in_this_video

                segment_path = os.path.join(temp_dir, f"segment_{i:02d}_{cut.camera}.mp4")

                # Convert absolute time to HH:MM:SS for debugging
                abs_start_h = int(segment_start_abs // 3600)
                abs_start_m = int((segment_start_abs % 3600) // 60)
                abs_start_s = segment_start_abs % 60
                print(f"  Extracting {cut.camera}: abs_time={abs_start_h:02d}:{abs_start_m:02d}:{abs_start_s:04.1f}, offset={offset:.1f}s, duration={segment_duration:.1f}s from {os.path.basename(video_path)}")

                if next_video_path and next_video_duration > 0:
                    # Need to extract from two video files and concatenate them
                    first_segment_path = os.path.join(temp_dir, f"segment_{i:02d}_{cut.camera}_part1.mp4")
                    second_segment_path = os.path.join(temp_dir, f"segment_{i:02d}_{cut.camera}_part2.mp4")

                    print(f"    Extracting part 1: {segment_duration:.1f}s from {os.path.basename(video_path)}")
                    first_ok = self._extract_segment(video_path, offset, segment_duration, first_segment_path)

                    print(f"    Extracting part 2: {next_video_duration:.1f}s from {os.path.basename(next_video_path)}")
                    # Next video starts at offset 0 (or very small offset from video_end_sec)
                    second_ok = self._extract_segment(next_video_path, 0.1, next_video_duration, second_segment_path)

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

        # Add overlay (title + logos)
        print(f"  Adding overlay...")
        if self._add_overlay(concat_path, str(output_path), racer):
            return output_path
        else:
            # Fallback: copy without overlay
            print(f"  Warning: Overlay failed, copying without overlay")
            if self._copy_without_overlay(concat_path, str(output_path)):
                return output_path
            return None

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

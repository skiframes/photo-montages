#!/usr/bin/env python3
"""
Video Clip Generator - Create MP4 video clips from detected run frames.

Pipes raw BGR frames to ffmpeg via stdin to encode H.264 video clips.
Uses platform-specific hardware encoders when available:
- h264_nvenc (Jetson/NVIDIA GPU)
- h264_videotoolbox (macOS)
- libx264 (CPU fallback)

Output is web-optimized with -movflags +faststart for streaming.
"""

import os
import sys
import logging
import subprocess
import platform
from typing import Optional, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def _detect_encoder() -> str:
    """Detect the best available H.264 encoder for this platform."""
    # Check for NVIDIA GPU (Jetson or discrete GPU)
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        if 'h264_nvenc' in result.stdout:
            # Verify it actually works (some systems list it but can't use it)
            test = subprocess.run(
                ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.1',
                 '-c:v', 'h264_nvenc', '-f', 'null', '-'],
                capture_output=True, timeout=10
            )
            if test.returncode == 0:
                logger.info("Using h264_nvenc encoder")
                return 'h264_nvenc'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for macOS VideoToolbox
    if platform.system() == 'Darwin':
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True, text=True, timeout=5
            )
            if 'h264_videotoolbox' in result.stdout:
                logger.info("Using h264_videotoolbox encoder")
                return 'h264_videotoolbox'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Fallback to CPU encoder
    logger.info("Using libx264 encoder (CPU)")
    return 'libx264'


# Cache the encoder choice
_encoder = None


def _get_encoder() -> str:
    """Get cached encoder choice."""
    global _encoder
    if _encoder is None:
        _encoder = _detect_encoder()
    return _encoder


def generate_video_clip(
    frames: List[np.ndarray],
    output_path: str,
    source_fps: float = 30.0,
    crop_region: Optional[Dict] = None,
) -> Optional[str]:
    """
    Generate an MP4 video clip from a list of BGR frames.

    Args:
        frames: List of numpy arrays (BGR format from OpenCV).
        output_path: Where to write the .mp4 file.
        source_fps: FPS of the source video (for correct playback speed).
        crop_region: Optional dict with x, y, w, h to crop frames.

    Returns:
        Path to the generated MP4 file, or None if generation fails.
    """
    if not frames or len(frames) == 0:
        logger.warning("No frames provided for video clip")
        return None

    try:
        # Get frame dimensions (after potential crop)
        sample = frames[0]
        if crop_region:
            x = crop_region.get('x', 0)
            y = crop_region.get('y', 0)
            w = crop_region.get('w', sample.shape[1])
            h = crop_region.get('h', sample.shape[0])
        else:
            h, w = sample.shape[:2]
            x, y = 0, 0

        # Ensure even dimensions (required by H.264)
        out_w = w if w % 2 == 0 else w - 1
        out_h = h if h % 2 == 0 else h - 1

        if out_w < 2 or out_h < 2:
            logger.warning(f"Frame dimensions too small: {out_w}x{out_h}")
            return None

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        encoder = _get_encoder()

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{out_w}x{out_h}',
            '-pix_fmt', 'bgr24',
            '-r', str(source_fps),
            '-i', '-',  # Read from stdin
            '-c:v', encoder,
        ]

        # Encoder-specific options
        if encoder == 'libx264':
            cmd.extend(['-crf', '23', '-preset', 'fast'])
        elif encoder == 'h264_nvenc':
            cmd.extend(['-rc', 'vbr', '-cq', '23', '-preset', 'medium'])
        elif encoder == 'h264_videotoolbox':
            cmd.extend(['-q:v', '65'])

        cmd.extend([
            '-pix_fmt', 'yuv420p',  # Web-compatible pixel format
            '-movflags', '+faststart',  # Enable streaming
            output_path
        ])

        # Pipe frames to ffmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        for frame in frames:
            # Apply crop if specified
            if crop_region:
                cropped = frame[y:y+h, x:x+w]
            else:
                cropped = frame

            # Ensure even dimensions
            cropped = cropped[:out_h, :out_w]

            process.stdin.write(cropped.tobytes())

        process.stdin.close()
        process.wait(timeout=60)

        if process.returncode != 0:
            stderr = process.stderr.read().decode('utf-8', errors='replace')
            logger.error(f"ffmpeg failed (exit {process.returncode}): {stderr[-500:]}")
            return None

        # Verify output exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"Video clip generated: {output_path} ({size_kb:.0f} KB, {len(frames)} frames)")
            return output_path
        else:
            logger.error(f"Output file missing or empty: {output_path}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out")
        if process:
            process.kill()
        return None
    except Exception as e:
        logger.error(f"Video clip generation failed: {e}")
        return None

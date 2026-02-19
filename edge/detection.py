#!/usr/bin/env python3
"""
Detection Engine - Monitors video streams for ski runs using trigger zones.

Detects skiers entering START zone, tracks until they exit END zone,
then passes the run to the montage generator.

Supports:
- Live RTSP streams (real-time monitoring)
- Video files (batch processing)
"""

import cv2
import numpy as np
import os
import json
import threading
import queue
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict
from collections import deque


@dataclass
class Zone:
    """Trigger zone definition."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @classmethod
    def from_dict(cls, d: dict) -> 'Zone':
        return cls(x=d['x'], y=d['y'], w=d['w'], h=d['h'])


@dataclass
class DetectionConfig:
    """Configuration for detection engine."""
    session_id: str
    camera_url: str
    start_zone: Zone
    end_zone: Optional[Zone]  # Can be None if using duration mode
    session_end_time: datetime
    detection_threshold: int = 25
    min_pixel_change_pct: float = 5.0
    pre_buffer_seconds: float = 2.0
    post_buffer_seconds: float = 2.0
    cooldown_seconds: float = 3.0
    # Duration mode: if set, run ends after this many seconds instead of END zone trigger
    run_duration_seconds: Optional[float] = None
    # Shadow filtering: minimum brightness of changed pixels (0-255)
    # Shadows are dark (low brightness), skiers are brighter
    # Set to ~100 to filter shadows on snow
    min_brightness: int = 100
    # Start offset: fixed delay in seconds after trigger before capturing frames
    # Useful when camera is behind skier - they trigger while close, but you want
    # to capture after they've moved away to a better framing distance
    start_offset_sec: float = 0.0

    # Path to config file for hot-reloading
    config_path: Optional[str] = None
    _last_reload_check: float = 0.0
    _last_config_mtime: float = 0.0

    @classmethod
    def from_json(cls, path: str) -> 'DetectionConfig':
        with open(path) as f:
            data = json.load(f)

        # Parse session end time, handling various formats
        # Convert to naive local time so it can be compared with datetime.now()
        end_time_str = data['session_end_time']
        if end_time_str.endswith('Z'):
            end_time_str = end_time_str[:-1] + '+00:00'
        session_end = datetime.fromisoformat(end_time_str)
        # If timezone-aware, convert to local time then strip tzinfo
        if session_end.tzinfo is not None:
            session_end = session_end.astimezone().replace(tzinfo=None)

        # Parse end_zone - may be None if using duration mode
        end_zone = None
        if data.get('end_zone'):
            end_zone = Zone.from_dict(data['end_zone'])

        # Get run duration (default 2 seconds if no end zone)
        run_duration = data.get('run_duration_seconds')
        if run_duration is None and end_zone is None:
            run_duration = 2.0  # Default to 2 seconds if no end zone

        config = cls(
            session_id=data['session_id'],
            camera_url=data.get('camera_url', ''),
            start_zone=Zone.from_dict(data['start_zone']),
            end_zone=end_zone,
            session_end_time=session_end,
            detection_threshold=data.get('detection_threshold', 25),
            min_pixel_change_pct=data.get('min_pixel_change_pct', 5.0),
            pre_buffer_seconds=data.get('pre_buffer_seconds', 2.0),
            post_buffer_seconds=data.get('post_buffer_seconds', 2.0),
            run_duration_seconds=run_duration,
            min_brightness=data.get('min_brightness', 100),
            start_offset_sec=data.get('start_offset_sec', 0.0),
        )
        config.config_path = path
        try:
            config._last_config_mtime = os.path.getmtime(path)
        except OSError:
            pass
        return config

    def check_for_updates(self):
        """
        Hot-reload detection settings from the config file if it has been modified.
        Called periodically from the detection loop. Only reloads the tunable
        parameters (threshold, min_pixel_change, min_brightness, session_end_time),
        not zones or session structure.
        """
        if not self.config_path:
            return

        now = time.time()
        # Only check file every 2 seconds to avoid disk thrashing
        if now - self._last_reload_check < 2.0:
            return
        self._last_reload_check = now

        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return

        if mtime <= self._last_config_mtime:
            return  # File hasn't changed

        self._last_config_mtime = mtime

        try:
            with open(self.config_path) as f:
                data = json.load(f)

            old_threshold = self.detection_threshold
            old_pct = self.min_pixel_change_pct
            old_brightness = self.min_brightness

            self.detection_threshold = data.get('detection_threshold', self.detection_threshold)
            self.min_pixel_change_pct = data.get('min_pixel_change_pct', self.min_pixel_change_pct)
            self.min_brightness = data.get('min_brightness', self.min_brightness)
            self.start_offset_sec = data.get('start_offset_sec', self.start_offset_sec)

            # Reload session_end_time (allows extending session live)
            end_time_str = data.get('session_end_time', '')
            if end_time_str:
                if end_time_str.endswith('Z'):
                    end_time_str = end_time_str[:-1] + '+00:00'
                new_end = datetime.fromisoformat(end_time_str)
                if new_end.tzinfo is not None:
                    new_end = new_end.astimezone().replace(tzinfo=None)
                self.session_end_time = new_end

            changed = (old_threshold != self.detection_threshold or
                       old_pct != self.min_pixel_change_pct or
                       old_brightness != self.min_brightness)
            if changed:
                print(f"  [LIVE] Settings updated: threshold={self.detection_threshold}, "
                      f"min_pct={self.min_pixel_change_pct}%, brightness={self.min_brightness}")

        except Exception as e:
            print(f"  [LIVE] Failed to reload config: {e}")


@dataclass
class Run:
    """Detected ski run."""
    run_number: int
    start_time: datetime
    end_time: Optional[datetime] = None
    start_frame_num: int = 0
    end_frame_num: int = 0
    frames: List[np.ndarray] = field(default_factory=list)
    # Track when capture actually starts (after delay)
    capture_start_time: Optional[datetime] = None

    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class FrameBuffer:
    """
    Ring buffer for frames to capture pre-trigger footage.
    Stores frames with timestamps for the last N seconds.
    """

    def __init__(self, max_seconds: float, fps: float):
        self.max_frames = int(max_seconds * fps) + 1
        self.buffer = deque(maxlen=self.max_frames)
        self.fps = fps

    def add(self, frame: np.ndarray, frame_num: int, timestamp: datetime):
        self.buffer.append((frame.copy(), frame_num, timestamp))

    def get_frames_since(self, since_frame: int) -> List[tuple]:
        """Get all frames from since_frame onwards."""
        return [(f, n, t) for f, n, t in self.buffer if n >= since_frame]

    def get_recent(self, seconds: float) -> List[tuple]:
        """Get frames from the last N seconds."""
        if not self.buffer or seconds <= 0:
            return []
        count = int(seconds * self.fps)
        if count <= 0:
            return []
        return list(self.buffer)[-count:]


class DetectionEngine:
    """
    Main detection engine that monitors video for ski runs.

    Flow:
    1. Motion in START zone → begin tracking run, start buffering frames
    2. Motion in END zone → run complete, pass frames to callback
    3. Timeout → abandon run (skier didn't complete)
    """

    def __init__(self, config: DetectionConfig, on_run_complete: Optional[Callable[[Run], None]] = None,
                 vola_racers: Optional[List[Dict]] = None):
        self.config = config
        self.on_run_complete = on_run_complete
        self.vola_racers = vola_racers or []

        self.running = False
        self.run_count = 0
        self.current_run: Optional[Run] = None
        self.last_start_trigger = 0  # Frame number of last start trigger

        # Detection state
        self.prev_frame = None
        self.frame_buffer: Optional[FrameBuffer] = None

        # Stats
        self.frames_processed = 0
        self.runs_detected = 0
        self.stream_fps = 30.0  # Updated when connected to RTSP/video

        # Live metrics for calibration chart
        self.metrics_path: Optional[str] = None  # Set by runner.py
        self._last_start_pct = 0.0
        self._last_end_pct = 0.0
        self._metrics_buffer: deque = deque(maxlen=120)  # Last 60s at 0.5s intervals
        self._last_metrics_write = 0.0  # time.time() of last write

    def _get_racer_run_duration(self, timestamp: datetime) -> Optional[float]:
        """
        Find racer by timestamp and return their run duration.
        Returns None if no match or racer has no finish time.
        """
        if not self.vola_racers:
            return None

        # Convert timestamp to seconds since midnight
        time_sec = (
            timestamp.hour * 3600 +
            timestamp.minute * 60 +
            timestamp.second +
            timestamp.microsecond / 1_000_000
        )

        # Find racer whose camera window contains this timestamp
        tolerance = 10.0
        for racer in self.vola_racers:
            cam_start = racer.get('camera_start_sec', 0)
            cam_end = racer.get('camera_end_sec', 0)
            if (cam_start - tolerance) <= time_sec <= (cam_end + tolerance):
                run_duration = racer.get('run_duration')
                if run_duration and run_duration > 0:
                    return run_duration
                break  # Found racer but no finish time

        return None

    def detect_motion_in_zone(self, prev_frame: np.ndarray, curr_frame: np.ndarray, zone: Zone) -> tuple:
        """
        Detect motion in a trigger zone using frame differencing.

        Filters out shadows by requiring changed pixels to be bright enough
        (shadows are dark, skiers in colorful suits are brighter).

        Returns:
            Tuple of (triggered: bool, pct_changed: float)
        """
        # Extract zone regions
        prev_zone = prev_frame[zone.y1:zone.y2, zone.x1:zone.x2]
        curr_zone = curr_frame[zone.y1:zone.y2, zone.x1:zone.x2]

        # Compute absolute difference
        diff = cv2.absdiff(prev_zone, curr_zone)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Create mask of pixels with significant motion
        motion_mask = gray_diff > self.config.detection_threshold

        # Shadow filtering: check brightness of changed pixels in current frame
        # Shadows cause motion but the area is dark; skiers are brighter
        if self.config.min_brightness > 0:
            # Convert current zone to grayscale for brightness check
            curr_gray = cv2.cvtColor(curr_zone, cv2.COLOR_BGR2GRAY)
            # Only count motion pixels that are also bright enough (not shadow)
            brightness_mask = curr_gray >= self.config.min_brightness
            motion_mask = motion_mask & brightness_mask

        # Count qualifying motion pixels
        motion_pixels = np.sum(motion_mask)
        total_pixels = gray_diff.size

        # Check if percentage exceeds minimum
        pct_changed = (motion_pixels / total_pixels) * 100
        return (pct_changed > self.config.min_pixel_change_pct, pct_changed)

    def _write_metrics(self, timestamp: datetime):
        """Write detection metrics to JSON file for live calibration chart."""
        if not self.metrics_path:
            return

        now = time.time()
        # Only write every 0.5 seconds to avoid disk thrashing
        if now - self._last_metrics_write < 0.5:
            return
        self._last_metrics_write = now

        # Add entry to buffer
        entry = {
            't': timestamp.isoformat(),
            'start_pct': round(self._last_start_pct, 2),
            'end_pct': round(self._last_end_pct, 2),
            'threshold': self.config.min_pixel_change_pct,
            'detection_threshold': self.config.detection_threshold,
            'min_brightness': self.config.min_brightness,
            'run_active': self.current_run is not None,
            'run_count': self.run_count,
        }
        self._metrics_buffer.append(entry)

        # Atomic write: tmp file + os.replace()
        try:
            metrics_data = {
                'entries': list(self._metrics_buffer),
                'updated_at': now,
            }
            tmp_path = self.metrics_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(metrics_data, f)
            os.replace(tmp_path, self.metrics_path)
        except Exception:
            pass  # Non-critical, skip silently

    def process_frame(self, frame: np.ndarray, frame_num: int, timestamp: datetime) -> Optional[Run]:
        """
        Process a single frame for run detection.
        Returns a completed Run if one was just finished, None otherwise.
        """
        completed_run = None

        # Add to buffer
        if self.frame_buffer:
            self.frame_buffer.add(frame, frame_num, timestamp)

        # Need previous frame for differencing
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return None

        # Check for motion in START zone
        start_motion, start_pct = self.detect_motion_in_zone(self.prev_frame, frame, self.config.start_zone)

        # Check END zone only if configured (not using duration mode)
        end_motion = False
        end_pct = 0.0
        if self.config.end_zone:
            end_motion, end_pct = self.detect_motion_in_zone(self.prev_frame, frame, self.config.end_zone)

        # Store latest metrics for live visualization
        self._last_start_pct = start_pct
        self._last_end_pct = end_pct
        self._write_metrics(timestamp)

        # State machine
        if self.current_run is None:
            # Not tracking - look for START trigger
            cooldown_frames = int(self.config.cooldown_seconds * (self.frame_buffer.fps if self.frame_buffer else 30))
            if start_motion and (frame_num - self.last_start_trigger) > cooldown_frames:
                # Start new run
                self.run_count += 1
                self.current_run = Run(
                    run_number=self.run_count,
                    start_time=timestamp,
                    start_frame_num=frame_num,
                )
                self.last_start_trigger = frame_num
                mode_str = f"duration={self.config.run_duration_seconds}s" if self.config.run_duration_seconds else "END zone"

                # Fixed delay in seconds after trigger (same for all racers)
                start_delay_seconds = self.config.start_offset_sec if self.config.start_offset_sec > 0 else 0
                self.current_run.start_delay_seconds = start_delay_seconds

                delay_str = f", offset={start_delay_seconds:.1f}s" if start_delay_seconds > 0 else ""
                print(f"  [RUN {self.run_count}] Triggered at {timestamp.strftime('%H:%M:%S')} ({mode_str}{delay_str})")

                # If no delay, start capturing immediately with pre-buffer
                if start_delay_seconds <= 0:
                    self.current_run.capture_start_time = timestamp
                    if self.frame_buffer:
                        pre_frames = self.frame_buffer.get_recent(self.config.pre_buffer_seconds)
                        self.current_run.frames = [f for f, n, t in pre_frames]
                # Otherwise, capture_start_time stays None until delay elapses
        else:
            # Currently tracking a run
            # Check if we're still in the delay period
            if self.current_run.capture_start_time is None:
                # Still waiting for delay to elapse
                elapsed_since_trigger = (timestamp - self.current_run.start_time).total_seconds()
                start_delay = getattr(self.current_run, 'start_delay_seconds', 0)
                if elapsed_since_trigger >= start_delay:
                    # Delay elapsed - start capturing now
                    self.current_run.capture_start_time = timestamp
                    print(f"  [RUN {self.current_run.run_number}] Capture started after {elapsed_since_trigger:.1f}s delay")
                    # Add current frame as first frame (no pre-buffer when using delay)
                    self.current_run.frames.append(frame.copy())
                # Still in delay - don't add frames yet
            else:
                # Actively capturing - add frame
                self.current_run.frames.append(frame.copy())

            # Calculate elapsed time since trigger (for timeout check)
            run_elapsed = (timestamp - self.current_run.start_time).total_seconds()

            # For duration calculation, use capture_start_time if capturing has started
            capture_elapsed = (timestamp - self.current_run.capture_start_time).total_seconds() if self.current_run.capture_start_time else 0

            # Check for run completion - either by END zone trigger or duration
            run_complete = False
            if self.config.run_duration_seconds:
                # Duration mode: complete after N seconds of actual capture
                # Only check if capture has started (delay elapsed)
                if self.current_run.capture_start_time and capture_elapsed >= self.config.run_duration_seconds:
                    run_complete = True
            elif end_motion:
                # Zone mode: complete when END zone triggered
                run_complete = True

            if run_complete:
                # Run complete!
                self.current_run.end_time = timestamp
                self.current_run.end_frame_num = frame_num
                self.runs_detected += 1

                print(f"  [RUN {self.current_run.run_number}] Completed at {timestamp.strftime('%H:%M:%S')} "
                      f"({self.current_run.duration:.1f}s, {len(self.current_run.frames)} frames)")

                completed_run = self.current_run
                self.current_run = None

                if self.on_run_complete:
                    self.on_run_complete(completed_run)

            # Timeout check (30 seconds max run time from trigger)
            elif run_elapsed > 30:
                print(f"  [RUN {self.current_run.run_number}] Timeout - abandoned")
                self.current_run = None

        self.prev_frame = frame.copy()
        self.frames_processed += 1

        return completed_run

    def run_on_video(self, video_path: str, output_callback: Optional[Callable[[Run], None]] = None,
                     video_start_time: Optional[datetime] = None) -> List[Run]:
        """
        Process a video file and detect all runs.

        Args:
            video_path: Path to video file
            output_callback: Optional callback for each completed run
            video_start_time: Optional datetime for when video recording started (Boston time).
                              If provided, run timestamps will be accurate clock times.

        Returns list of completed runs.
        """
        print(f"\nProcessing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ERROR: Cannot open video")
            return []

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}, Duration: {duration:.1f}s, Frames: {total_frames}")

        # Reset detection state for new video
        self.prev_frame = None
        self.current_run = None
        self.last_start_trigger = 0

        # Initialize frame buffer
        buffer_seconds = (self.config.pre_buffer_seconds or 0) + 5  # Extra buffer
        self.frame_buffer = FrameBuffer(buffer_seconds, fps)

        # Use provided video start time, or try to extract from filename, or fall back to mtime
        if video_start_time:
            video_start = video_start_time
            print(f"  Video start time (from config): {video_start.strftime('%H:%M:%S')}")
        else:
            video_start = self._get_video_start_time(video_path)
            if video_start:
                print(f"  Video start time (from filename): {video_start}")
            else:
                # Use file modification time as fallback
                import os
                mtime = os.path.getmtime(video_path)
                video_start = datetime.fromtimestamp(mtime)
                print(f"  Video start time (from file mtime): {video_start}")

        runs = []
        frame_num = 0
        check_interval = max(1, int(fps / 10))  # Check ~10 times per second

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Only process every Nth frame for efficiency
            if frame_num % check_interval != 0:
                continue

            current_time = frame_num / fps
            timestamp = video_start + timedelta(seconds=current_time)

            completed = self.process_frame(frame, frame_num, timestamp)
            if completed:
                runs.append(completed)
                # Note: on_run_complete callback is already called in process_frame

            # Progress update every minute
            if frame_num % int(fps * 60) == 0:
                print(f"  ... {current_time / 60:.1f} minutes processed")

        cap.release()
        print(f"  Finished: {len(runs)} runs detected")
        return runs

    def run_on_rtsp(self, rtsp_url: Optional[str] = None) -> None:
        """
        Monitor an RTSP stream in real-time.
        Runs until session_end_time or stop() is called.
        """
        url = rtsp_url or self.config.camera_url
        print(f"\nConnecting to: {url}")

        # Use ffmpeg backend with Reolink timestamp fixes:
        # - genpts: fix non-monotonic timestamps from Reolink cameras
        # - rtsp_transport tcp: reliable transport (no UDP packet loss)
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;+genpts"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  ERROR: Cannot connect to stream")
            return

        # Get stream info
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if not available
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.stream_fps = fps
        print(f"  Connected: {width}x{height} @ {fps:.1f}fps")
        print(f"  Session ends: {self.config.session_end_time.strftime('%H:%M:%S')}")

        # Initialize frame buffer
        buffer_seconds = self.config.pre_buffer_seconds + 5
        self.frame_buffer = FrameBuffer(buffer_seconds, fps)

        self.running = True
        frame_num = 0
        check_interval = max(1, int(fps / 10))

        try:
            while self.running:
                # Hot-reload settings if config file changed
                self.config.check_for_updates()

                # Check session end time
                now = datetime.now()
                if now >= self.config.session_end_time:
                    print(f"\n  Session ended at {now.strftime('%H:%M:%S')}")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("  WARNING: Frame read failed, reconnecting...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    continue

                frame_num += 1

                # When a run is active, capture every frame for sharp montages.
                # When idle, only check every Nth frame for detection efficiency.
                if self.current_run is not None:
                    # Active run: capture frame and check for completion
                    self.process_frame(frame, frame_num, now)
                elif frame_num % check_interval == 0:
                    # Idle: check for new run trigger at ~10fps
                    self.process_frame(frame, frame_num, now)

        except KeyboardInterrupt:
            print("\n  Interrupted by user")
        finally:
            cap.release()
            self.running = False
            print(f"\n  Total runs detected: {self.runs_detected}")

    def stop(self):
        """Stop the detection engine."""
        self.running = False

    def _get_video_start_time(self, filename: str) -> Optional[datetime]:
        """Extract start time from video filename."""
        basename = Path(filename).stem
        basename = basename.replace('_trimmed', '')
        parts = basename.split('_')

        date_str = None
        time_str = None

        for part in parts:
            if len(part) == 8 and part.isdigit():
                date_str = part
            elif len(part) == 6 and part.isdigit():
                time_str = part

        if date_str and time_str:
            try:
                return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        return None


def main():
    """CLI for testing detection engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Ski run detection engine")
    parser.add_argument("input", help="Video file or 'rtsp' for live stream")
    parser.add_argument("-c", "--config", required=True, help="Session config JSON file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")

    args = parser.parse_args()

    # Load config
    config = DetectionConfig.from_json(args.config)
    print(f"Session: {config.session_id}")
    print(f"Start zone: ({config.start_zone.x}, {config.start_zone.y}) {config.start_zone.w}x{config.start_zone.h}")
    if config.end_zone:
        print(f"End zone: ({config.end_zone.x}, {config.end_zone.y}) {config.end_zone.w}x{config.end_zone.h}")
    if config.run_duration_seconds:
        print(f"Run duration: {config.run_duration_seconds}s (duration mode)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Simple callback that saves run info
    def on_run(run: Run):
        print(f"    → Run {run.run_number}: {len(run.frames)} frames, {run.duration:.1f}s")

    # Create engine
    engine = DetectionEngine(config, on_run_complete=on_run)

    if args.input.lower() == 'rtsp':
        engine.run_on_rtsp()
    else:
        engine.run_on_video(args.input)


if __name__ == "__main__":
    main()

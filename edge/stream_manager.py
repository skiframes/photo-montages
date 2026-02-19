#!/usr/bin/env python3
"""
Stream Manager - Shares a single RTSP connection across multiple detection sessions.

When multiple sessions monitor the same camera, only one RTSP connection is opened.
Frames are decoded once and distributed to all registered DetectionEngine instances.
This saves ~50% CPU per additional session on the same camera.
"""

import cv2
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Optional

from detection import FrameBuffer


class CameraStream:
    """
    Manages a single RTSP camera connection shared by multiple sessions.

    Reads frames in a dedicated thread and distributes them to all
    registered DetectionEngine instances. Each engine maintains its own
    state (prev_frame, frame_buffer, current_run) independently.
    """

    def __init__(self, camera_id: str, rtsp_url: str, montage_pool: ThreadPoolExecutor):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.montage_pool = montage_pool

        # session_id -> {'runner': SkiFramesRunner, 'engine': DetectionEngine}
        self.sessions: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stream_fps = 30.0
        self.frame_num = 0

    def add_session(self, session_id: str, runner) -> None:
        """Register a detection session to receive frames from this camera."""
        engine = runner.engine

        with self.lock:
            self.sessions[session_id] = {
                'runner': runner,
                'engine': engine,
            }

            # Initialize the engine's frame buffer
            buffer_seconds = engine.config.pre_buffer_seconds + 5
            engine.frame_buffer = FrameBuffer(buffer_seconds, self.stream_fps)
            engine.running = True

            should_start = not self.running

        if should_start:
            self._start()

        print(f"[STREAM {self.camera_id}] Session {session_id} registered "
              f"({len(self.sessions)} total)")

    def remove_session(self, session_id: str) -> None:
        """Unregister a session. Stops stream if no sessions remain."""
        with self.lock:
            info = self.sessions.pop(session_id, None)
            remaining = len(self.sessions)
            should_stop = remaining == 0 and self.running

        if info:
            info['engine'].running = False
            print(f"[STREAM {self.camera_id}] Session {session_id} removed "
                  f"({remaining} remaining)")

        if should_stop:
            self.running = False
            print(f"[STREAM {self.camera_id}] No sessions left, stopping stream")

    def _start(self) -> None:
        """Start the RTSP reader thread."""
        self.running = True
        self.frame_num = 0
        self.thread = threading.Thread(
            target=self._read_loop, daemon=True,
            name=f"stream-{self.camera_id}"
        )
        self.thread.start()

    def _read_loop(self) -> None:
        """Main loop: read RTSP frames and distribute to all sessions."""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
            "rtsp_transport;tcp|fflags;+genpts"

        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[STREAM {self.camera_id}] ERROR: Cannot connect to "
                  f"{self.rtsp_url}")
            self.running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stream_fps = fps
        check_interval = max(1, int(fps / 10))  # ~10 checks/sec when idle

        print(f"[STREAM {self.camera_id}] Connected: {width}x{height} "
              f"@ {fps:.1f}fps")

        # Update FPS on all registered runners and reinit frame buffers
        with self.lock:
            for info in self.sessions.values():
                info['runner'].source_fps = fps
                engine = info['engine']
                buffer_seconds = engine.config.pre_buffer_seconds + 5
                engine.frame_buffer = FrameBuffer(buffer_seconds, fps)

        reconnect_attempts = 0
        max_reconnect = 10

        while self.running:
            ret, frame = cap.read()
            if not ret:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect:
                    print(f"[STREAM {self.camera_id}] Too many reconnect "
                          f"failures, stopping")
                    break
                print(f"[STREAM {self.camera_id}] Frame read failed, "
                      f"reconnecting ({reconnect_attempts}/{max_reconnect})...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue

            reconnect_attempts = 0
            self.frame_num += 1
            now = datetime.now()

            # Distribute frame to all sessions
            expired = []
            with self.lock:
                for session_id, info in self.sessions.items():
                    engine = info['engine']

                    # Hot-reload config settings
                    engine.config.check_for_updates()

                    # Check session expiry
                    if now >= engine.config.session_end_time:
                        expired.append(session_id)
                        continue

                    # Per-engine frame skipping:
                    # Active run → every frame (for sharp montages)
                    # Idle → every Nth frame (for detection efficiency)
                    try:
                        if engine.current_run is not None:
                            engine.process_frame(frame, self.frame_num, now)
                        elif self.frame_num % check_interval == 0:
                            engine.process_frame(frame, self.frame_num, now)
                    except Exception as e:
                        print(f"[STREAM {self.camera_id}] Error in session "
                              f"{session_id}: {e}")

            # Remove expired sessions outside the frame distribution lock
            for sid in expired:
                print(f"[STREAM {self.camera_id}] Session {sid} expired")
                self.remove_session(sid)

            # If all sessions were removed, stop
            with self.lock:
                if not self.sessions:
                    break

        cap.release()
        self.running = False
        print(f"[STREAM {self.camera_id}] Stream closed")


class StreamManager:
    """
    Manages shared RTSP streams across multiple detection sessions.

    Only one RTSP connection is opened per camera, regardless of how
    many sessions are monitoring it. Montage generation runs in a
    thread pool to avoid blocking frame reading.
    """

    def __init__(self, max_montage_workers: int = 4):
        self.streams: Dict[str, CameraStream] = {}  # camera_id -> CameraStream
        self.lock = threading.Lock()
        self.montage_pool = ThreadPoolExecutor(
            max_workers=max_montage_workers,
            thread_name_prefix="montage"
        )
        # Track session_id -> camera_id for stop_session lookup
        self._session_cameras: Dict[str, str] = {}

    def start_session(self, camera_id: str, rtsp_url: str,
                      session_id: str, runner) -> None:
        """Register a session to receive frames from a camera's RTSP stream.

        The runner's on_run_complete callback is wrapped to run montage
        generation asynchronously in a thread pool, so it doesn't block
        frame reading for other sessions.
        """
        # Wrap on_run_complete to run async in thread pool
        original_callback = runner._on_run_complete

        def async_on_run_complete(run):
            self.montage_pool.submit(original_callback, run)

        runner.engine.on_run_complete = async_on_run_complete

        with self.lock:
            if camera_id not in self.streams:
                self.streams[camera_id] = CameraStream(
                    camera_id, rtsp_url, self.montage_pool
                )
            self.streams[camera_id].add_session(session_id, runner)
            self._session_cameras[session_id] = camera_id

    def stop_session(self, session_id: str) -> bool:
        """Stop a session by session_id. Returns True if found and stopped."""
        with self.lock:
            camera_id = self._session_cameras.pop(session_id, None)
            stream = self.streams.get(camera_id) if camera_id else None

        if stream:
            stream.remove_session(session_id)
            # Clean up empty streams
            with self.lock:
                if camera_id in self.streams and not self.streams[camera_id].sessions:
                    del self.streams[camera_id]
            return True
        return False

    def stop_all(self) -> None:
        """Stop all streams and sessions."""
        with self.lock:
            for stream in self.streams.values():
                stream.running = False
            self.streams.clear()
            self._session_cameras.clear()
        self.montage_pool.shutdown(wait=False)

    def get_status(self) -> dict:
        """Get status of all streams and sessions."""
        with self.lock:
            status = {}
            for camera_id, stream in self.streams.items():
                status[camera_id] = {
                    'running': stream.running,
                    'fps': stream.stream_fps,
                    'frame_count': stream.frame_num,
                    'sessions': list(stream.sessions.keys()),
                }
            return status

    def get_session_camera(self, session_id: str) -> Optional[str]:
        """Get the camera_id for a session."""
        with self.lock:
            return self._session_cameras.get(session_id)

#!/usr/bin/env python3
"""
Runner - Main entry point that ties detection and montage generation together.

Monitors video source, detects runs, generates montages, and prepares for upload.
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from detection import DetectionEngine, DetectionConfig, Run
from montage import generate_montage, MontageResult, MontageResultPair, DEFAULT_FPS


def parse_reolink_video_start_time(video_path: str, race_date: str) -> Optional[datetime]:
    """
    Parse video start time from Reolink filename.
    Format: RecM03_YYYYMMDD_HHMMSS_HHMMSS_XXXXX_XXXXX.mp4

    Args:
        video_path: Path to video file
        race_date: Race date in YYYY-MM-DD format (for building datetime)

    Returns datetime or None if parse fails.
    """
    filename = Path(video_path).name
    # Example: RecM03_20260201_081538_082039_533C810_BEF5D90.mp4
    match = re.match(r'RecM03_(\d{8})_(\d{6})_(\d{6})_', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        start_str = match.group(2)  # HHMMSS

        try:
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(start_str[0:2])
            minute = int(start_str[2:4])
            second = int(start_str[4:6])

            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass

    return None


class SkiFramesRunner:
    """
    Main runner that orchestrates detection and montage generation.
    """

    def __init__(self, config_path: str, output_dir: str = "./output",
                 montage_fps: float = None):
        self.config = DetectionConfig.from_json(config_path)
        self.config_path = config_path
        self.output_dir = output_dir
        self.session_dir = os.path.join(output_dir, self.config.session_id)

        # Load raw config to get zone dicts and montage_fps
        with open(config_path) as f:
            self.raw_config = json.load(f)

        # Use montage_fps_list from config if not overridden, default to [DEFAULT_FPS]
        if montage_fps:
            self.montage_fps_list = [montage_fps]
        else:
            fps_list = self.raw_config.get('montage_fps_list')
            if fps_list and isinstance(fps_list, list):
                # Filter out None values (null in JSON)
                self.montage_fps_list = [float(f) for f in fps_list if f is not None]
                if not self.montage_fps_list:
                    self.montage_fps_list = [float(DEFAULT_FPS)]
            else:
                # Backward compatibility: single montage_fps value
                single = self.raw_config.get('montage_fps', DEFAULT_FPS)
                self.montage_fps_list = [float(single)]
        self.montage_fps = self.montage_fps_list[0]  # Keep for backward compat

        # Create output directories
        os.makedirs(self.session_dir, exist_ok=True)

        # Track generated montages (A/B pairs)
        self.montage_pairs: list[MontageResultPair] = []

        # Source FPS (will be set when video is opened)
        self.source_fps = 30.0

        # Vola racer data for naming montages
        self.vola_racers: List[Dict] = self.raw_config.get('vola_racers', [])
        self.vola_race: str = self.raw_config.get('vola_race', '')
        self.vola_camera: str = self.raw_config.get('vola_camera', '')
        self.race_date: str = self.raw_config.get('race_date', datetime.now().strftime('%Y-%m-%d'))

        # Race info for overlay (same format as video stitcher)
        self.race_info: Dict = self.raw_config.get('race_info', {})

        # Videos to process (from Vola API)
        self.vola_videos: List[Dict] = self.raw_config.get('vola_videos', [])

        # Track which racers have been matched (prevent duplicate matches)
        self.matched_racer_indices: set = set()

        # Create detection engine with callback and vola_racers for offset calculation
        self.engine = DetectionEngine(self.config, on_run_complete=self._on_run_complete,
                                       vola_racers=self.vola_racers)

        print(f"SkiFrames Runner initialized")
        print(f"  Session: {self.config.session_id}")
        print(f"  Output: {self.session_dir}")
        print(f"  Montage FPS list: {self.montage_fps_list}")
        if self.vola_racers:
            print(f"  Vola racers loaded: {len(self.vola_racers)} ({self.vola_race})")
            # Show timing coverage summary
            self._print_vola_timing_coverage()
        if self.vola_videos:
            print(f"  Vola videos: {len(self.vola_videos)}")
            for v in self.vola_videos[:3]:
                print(f"    {v.get('name', v.get('path', 'unknown'))}")

    def _print_vola_timing_coverage(self):
        """Print summary of Vola timing windows for debugging."""
        if not self.vola_racers:
            print("    No Vola racers loaded")
            return

        # Get timing range
        all_starts = [r.get('camera_start_sec', 0) for r in self.vola_racers]
        all_ends = [r.get('camera_end_sec', 0) for r in self.vola_racers]

        min_start = min(all_starts) if all_starts else 0
        max_end = max(all_ends) if all_ends else 0

        def sec_to_time(sec):
            h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        print(f"    Vola timing coverage: {sec_to_time(min_start)} - {sec_to_time(max_end)}")
        print(f"    First racer: Bib {self.vola_racers[0].get('bib')} @ {sec_to_time(all_starts[0])}")
        print(f"    Last racer:  Bib {self.vola_racers[-1].get('bib')} @ {sec_to_time(all_starts[-1])}")

        # Check for gaps in coverage (>60 seconds between racers)
        gaps = []
        sorted_starts = sorted(enumerate(all_starts), key=lambda x: x[1])
        for i in range(1, len(sorted_starts)):
            prev_idx, prev_start = sorted_starts[i-1]
            curr_idx, curr_start = sorted_starts[i]
            prev_end = all_ends[prev_idx]
            gap = curr_start - prev_end
            if gap > 60:  # Gap > 60 seconds
                gaps.append((prev_end, curr_start, gap))

        if gaps:
            print(f"    WARNING: {len(gaps)} timing gaps > 60s detected:")
            for prev_end, curr_start, gap in gaps[:3]:
                print(f"      Gap from {sec_to_time(prev_end)} to {sec_to_time(curr_start)} ({gap:.0f}s)")

        # Show all racers in compact format
        print(f"    All {len(self.vola_racers)} racers:")
        for r in self.vola_racers:
            cam_start = r.get('camera_start_sec', 0)
            cam_end = r.get('camera_end_sec', 0)
            print(f"      Bib {r.get('bib'):>3}: {sec_to_time(cam_start)} - {sec_to_time(cam_end)} ({r.get('name', 'N/A')})")

    def _find_matching_racer_by_timestamp(self, run_start_time: datetime) -> Optional[Dict]:
        """
        Match a detected run to a Vola racer by timestamp.

        Compares the run's start time (Boston clock time) against each racer's
        camera timing window (camera_start_sec, camera_end_sec) from Vola data.

        Returns the racer dict or None if no match found.
        """
        if not self.vola_racers:
            return None

        # Convert run start time to seconds since midnight (Boston time)
        run_time_sec = (
            run_start_time.hour * 3600 +
            run_start_time.minute * 60 +
            run_start_time.second +
            run_start_time.microsecond / 1_000_000
        )

        # Allow tolerance for timing drift (10 seconds before/after camera window)
        tolerance = 10.0

        for i, racer in enumerate(self.vola_racers):
            if i in self.matched_racer_indices:
                continue  # Already matched this racer

            cam_start = racer.get('camera_start_sec', 0)
            cam_end = racer.get('camera_end_sec', 0)

            # Check if run time falls within camera window (with tolerance)
            if (cam_start - tolerance) <= run_time_sec <= (cam_end + tolerance):
                self.matched_racer_indices.add(i)
                return racer

        # If no match by timestamp, log detailed debug info
        def sec_to_time(sec):
            h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        print(f"    ⚠️  NO MATCH for run at {run_start_time.strftime('%H:%M:%S')} ({run_time_sec:.0f}s since midnight)")

        if not self.vola_racers:
            print(f"       Reason: No Vola racers loaded")
            return None

        # Check if run is outside entire Vola timing range
        all_starts = [r.get('camera_start_sec', 0) for r in self.vola_racers]
        all_ends = [r.get('camera_end_sec', 0) for r in self.vola_racers]
        min_start = min(all_starts) - tolerance
        max_end = max(all_ends) + tolerance

        if run_time_sec < min_start:
            print(f"       Reason: Run is BEFORE first racer ({sec_to_time(min_start + tolerance)})")
            print(f"       Run time {sec_to_time(run_time_sec)} is {min_start + tolerance - run_time_sec:.0f}s too early")
        elif run_time_sec > max_end:
            print(f"       Reason: Run is AFTER last racer ({sec_to_time(max_end - tolerance)})")
            print(f"       Run time {sec_to_time(run_time_sec)} is {run_time_sec - max_end + tolerance:.0f}s too late")
        else:
            # Run is within overall range but no match - likely already matched or in gap
            print(f"       Reason: Run falls in a gap or racer already matched")

            # Show nearby unmatched racers
            nearby_unmatched = []
            nearby_matched = []
            for i, r in enumerate(self.vola_racers):
                cam_start = r.get('camera_start_sec', 0)
                diff = abs(run_time_sec - cam_start)
                if diff < 120:  # Within 2 minutes
                    if i in self.matched_racer_indices:
                        nearby_matched.append((r, diff, i))
                    else:
                        nearby_unmatched.append((r, diff, i))

            if nearby_matched:
                nearby_matched.sort(key=lambda x: x[1])
                print(f"       Already matched nearby racers:")
                for r, diff, i in nearby_matched[:3]:
                    cam_start = r.get('camera_start_sec', 0)
                    cam_end = r.get('camera_end_sec', 0)
                    print(f"         Bib {r.get('bib'):>3}: {sec_to_time(cam_start)}-{sec_to_time(cam_end)} (diff={diff:.0f}s) [MATCHED]")

            if nearby_unmatched:
                nearby_unmatched.sort(key=lambda x: x[1])
                print(f"       Unmatched nearby racers:")
                for r, diff, i in nearby_unmatched[:3]:
                    cam_start = r.get('camera_start_sec', 0)
                    cam_end = r.get('camera_end_sec', 0)
                    # Show why it didn't match
                    if run_time_sec < cam_start - tolerance:
                        reason = f"run is {cam_start - tolerance - run_time_sec:.0f}s too early"
                    elif run_time_sec > cam_end + tolerance:
                        reason = f"run is {run_time_sec - cam_end - tolerance:.0f}s too late"
                    else:
                        reason = "should match??"
                    print(f"         Bib {r.get('bib'):>3}: {sec_to_time(cam_start)}-{sec_to_time(cam_end)} ({reason})")

            if not nearby_matched and not nearby_unmatched:
                print(f"       No racers within 2 minutes of this run time")

        print(f"       Matched so far: {len(self.matched_racer_indices)}/{len(self.vola_racers)} racers")
        return None

    def _generate_filename(self, racer: Optional[Dict], run_number: int) -> Optional[str]:
        """
        Generate filename based on Vola racer data.
        Format: {Name}_{bib}
        Example: JohnDoe_175
        """
        if not racer:
            return None

        bib = racer.get('bib', run_number)
        name = racer.get('name', '')  # Name from results PDF (already formatted as "FirstnameLastname")

        if name:
            return f"{name}_{bib}"
        else:
            return f"{bib}"

    def _get_run_view_folder(self) -> str:
        """
        Generate folder name based on run number and view.
        Format: Run{X}_View{Y}
        Example: Run1_View1
        """
        # Parse run number from race name (e.g., "U12 run 1" -> "1")
        run_num = '1'
        if 'run 2' in self.vola_race.lower():
            run_num = '2'

        # View number from config (1-5), default to 1
        view_num = self.raw_config.get('vola_view', '1')

        return f"Run{run_num}_View{view_num}"

    def _get_output_folder(self, racer: Optional[Dict]) -> Optional[str]:
        """
        Generate output folder path based on team, gender, and run/view.
        Format: {Team}/{Gender}/Run{X}_View{Y}
        Example: SUN/Women/Run1_View1

        Falls back to simpler paths if team or gender not available.
        """
        if not self.vola_race:
            return None

        run_view = self._get_run_view_folder()

        # Get team and gender from racer data
        team = racer.get('team', '') if racer else ''
        gender = racer.get('gender', '') if racer else ''

        if team and gender:
            return f"{team}/{gender}/{run_view}"
        elif team:
            return f"{team}/{run_view}"
        else:
            return run_view

    def _on_run_complete(self, run: Run):
        """Callback when a run is detected - generate Time1/Time2 montage pair."""
        # Match this run to a Vola racer by timestamp (Boston clock time)
        racer = None
        if run.start_time and self.vola_racers:
            racer = self._find_matching_racer_by_timestamp(run.start_time)

        custom_filename = self._generate_filename(racer, run.run_number)

        if racer:
            print(f"  Matched to racer: Bib {racer.get('bib')} ({racer.get('team', 'no team')})")

        # Get run duration for overlay display
        run_duration = run.duration if run.end_time else self.raw_config.get('run_duration_seconds')

        # Get folder path for Team/Run_View organization
        output_folder = self._get_output_folder(racer)

        # Build race title for overlay (fallback if no race_info)
        race_title = ""
        if not self.race_info and self.vola_race:
            # Extract age group from vola_race (e.g., "U12 run 1" -> "U12")
            age_group = ""
            if "U12" in self.vola_race.upper():
                age_group = "U12"
            elif "U14" in self.vola_race.upper():
                age_group = "U14"
            elif "U16" in self.vola_race.upper():
                age_group = "U16"
            elif "U19" in self.vola_race.upper():
                age_group = "U19"
            if age_group:
                race_title = f"Western Division {age_group} Ranking - SL"

        # Generate montages at each selected FPS value, merge into single run entry
        merged_results = {}
        for fps_val in self.montage_fps_list:
            # Append FPS to filename to distinguish variants
            fps_suffix = f"_{fps_val:.1f}fps"
            fps_filename = f"{custom_filename}{fps_suffix}" if custom_filename else None

            # Compute elapsed time for overlay
            elapsed_time = round(run.duration, 2) if run.end_time else None

            result = generate_montage(
                frames=run.frames,
                run_number=run.run_number,
                timestamp=run.start_time,
                output_dir=self.session_dir,
                session_id=self.config.session_id,
                start_zone=self.raw_config.get('start_zone'),
                end_zone=self.raw_config.get('end_zone'),
                crop_zone=self.raw_config.get('crop_zone'),
                source_fps=self.source_fps,
                montage_fps=fps_val,
                custom_filename=fps_filename,  # Pass custom filename with FPS suffix
                run_view_folder=output_folder,  # Pass folder path (Team/Run1_View1)
                run_duration_sec=run_duration,  # Pass duration for overlay display
                race_title=race_title,  # Pass race title for overlay (fallback)
                race_info=self.race_info,  # Pass race info for overlay (preferred)
                elapsed_time=elapsed_time,  # Pass elapsed time for overlay
            )

            if result:
                # Use FPS value as variant key (e.g., "4.0fps")
                fps_key = f"{fps_val:.1f}fps"
                for variant_key, montage_result in result.results.items():
                    merged_results[fps_key] = montage_result

        if merged_results:
            # Compute elapsed time (seconds) between start and end trigger zones
            elapsed_time = round(run.duration, 2) if run.end_time else None

            merged_pair = MontageResultPair(
                run_number=run.run_number,
                timestamp=run.start_time,
                results=merged_results,
                elapsed_time=elapsed_time,
            )
            if racer:
                merged_pair.racer_bib = racer.get('bib')
            self.montage_pairs.append(merged_pair)

        self._update_manifest()

    def _update_manifest(self):
        """Update the session manifest file."""
        runs = []
        for pair in self.montage_pairs:
            run_entry = {
                "run_number": pair.run_number,
                "timestamp": pair.timestamp.isoformat(),
                "elapsed_time": pair.elapsed_time,
                "variants": {}
            }
            for variant, m in pair.results.items():
                # Use relative paths from session dir (e.g., "thumbnails/run_001_thumb.jpg")
                # to preserve subdirectory structure for S3 upload
                try:
                    thumb_rel = os.path.relpath(m.thumbnail_path, self.session_dir)
                    full_rel = os.path.relpath(m.fullres_path, self.session_dir)
                except ValueError:
                    thumb_rel = os.path.basename(m.thumbnail_path)
                    full_rel = os.path.basename(m.fullres_path)
                run_entry["variants"][variant] = {
                    "thumbnail": thumb_rel,
                    "fullres": full_rel,
                    "frame_count": m.frame_count,
                }
            runs.append(run_entry)

        manifest = {
            "session_id": self.config.session_id,
            "event_type": self.raw_config.get('session_type', 'training'),
            "discipline": self.raw_config.get('discipline', 'freeski'),
            "group": self.raw_config.get('group', ''),
            "camera_id": self.raw_config.get('camera_id', ''),
            "device_id": self.raw_config.get('device_id', ''),
            "event_date": self.race_date,
            "generated_at": datetime.now().isoformat(),
            "montage_fps_list": self.montage_fps_list,
            "runs": runs,
        }

        manifest_path = os.path.join(self.session_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def run_on_video(self, video_path: str):
        """Process a video file."""
        print(f"\nProcessing video: {video_path}")

        # Get video FPS for montage generation
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

        # Parse video start time from filename (Reolink format)
        video_start_time = parse_reolink_video_start_time(video_path, self.race_date)
        if video_start_time:
            print(f"  Video start time (from filename): {video_start_time.strftime('%H:%M:%S')}")

        # Pass video_start_time for accurate timestamp matching with Vola data
        self.engine.run_on_video(video_path, video_start_time=video_start_time)
        self._print_summary()

    def run_on_videos(self, video_paths: List[str]):
        """Process multiple video files in sequence."""
        print(f"\nProcessing {len(video_paths)} videos...")

        for video_path in video_paths:
            print(f"\nProcessing video: {video_path}")

            # Get video FPS for montage generation
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

            # Parse video start time from filename (Reolink format)
            video_start_time = parse_reolink_video_start_time(video_path, self.race_date)
            if video_start_time:
                print(f"  Video start time (from filename): {video_start_time.strftime('%H:%M:%S')}")

            # Pass video_start_time for accurate timestamp matching with Vola data
            self.engine.run_on_video(video_path, video_start_time=video_start_time)

        self._print_summary()

    def run_on_rtsp(self, rtsp_url: Optional[str] = None):
        """Monitor RTSP stream."""
        url = rtsp_url or self.config.camera_url
        print(f"\nMonitoring RTSP stream: {url}")
        # Probe stream FPS before starting detection so _on_run_complete
        # uses the correct source_fps for frame sampling
        import cv2
        probe = cv2.VideoCapture(url)
        if probe.isOpened():
            self.source_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
            probe.release()
            print(f"  Source FPS: {self.source_fps}")
        self.engine.run_on_rtsp(url)
        self._print_summary()

    def _print_summary(self):
        """Print session summary."""
        print(f"\n{'='*60}")
        print(f"SESSION SUMMARY: {self.config.session_id}")
        print(f"{'='*60}")

        # Matching statistics
        total_runs = self.engine.run_count if hasattr(self.engine, 'run_count') else len(self.montage_pairs)
        matched_count = len(self.matched_racer_indices)
        unmatched_count = total_runs - matched_count if total_runs > matched_count else 0

        print(f"  Runs detected: {total_runs}")
        if self.vola_racers:
            print(f"  Vola racers loaded: {len(self.vola_racers)}")
            print(f"  Matched to racers: {matched_count}")
            print(f"  Unmatched runs: {unmatched_count}")

            # Show which racers were NOT matched (missing runs)
            unmatched_racers = [r for i, r in enumerate(self.vola_racers) if i not in self.matched_racer_indices]
            if unmatched_racers:
                # Separate by finished status
                dnf_racers = [r for r in unmatched_racers if not r.get('finished', True)]
                finished_racers = [r for r in unmatched_racers if r.get('finished', True)]

                print(f"\n  ⚠️  Racers with NO detected run ({len(unmatched_racers)} total):")

                if finished_racers:
                    print(f"\n     FINISHED but not detected ({len(finished_racers)}):")
                    for r in finished_racers[:10]:
                        cam_start = r.get('camera_start_sec', 0)
                        cam_end = r.get('camera_end_sec', 0)
                        h1, m1, s1 = int(cam_start // 3600), int((cam_start % 3600) // 60), int(cam_start % 60)
                        h2, m2, s2 = int(cam_end // 3600), int((cam_end % 3600) // 60), int(cam_end % 60)
                        duration = r.get('run_duration')
                        dur_str = f"{duration:.1f}s" if duration else "N/A"
                        print(f"       Bib {r.get('bib'):>3} ({r.get('name', 'N/A'):18}) window {h1:02d}:{m1:02d}:{s1:02d}-{h2:02d}:{m2:02d}:{s2:02d} ({dur_str})")
                    if len(finished_racers) > 10:
                        print(f"       ... and {len(finished_racers) - 10} more")

                if dnf_racers:
                    print(f"\n     DNF/DNS (estimated timing, {len(dnf_racers)}):")
                    for r in dnf_racers[:10]:
                        cam_start = r.get('camera_start_sec', 0)
                        cam_end = r.get('camera_end_sec', 0)
                        h1, m1, s1 = int(cam_start // 3600), int((cam_start % 3600) // 60), int(cam_start % 60)
                        h2, m2, s2 = int(cam_end // 3600), int((cam_end % 3600) // 60), int(cam_end % 60)
                        print(f"       Bib {r.get('bib'):>3} ({r.get('name', 'N/A'):18}) window {h1:02d}:{m1:02d}:{s1:02d}-{h2:02d}:{m2:02d}:{s2:02d} (estimated)")
                    if len(dnf_racers) > 10:
                        print(f"       ... and {len(dnf_racers) - 10} more")

                # Check video coverage for unmatched racers
                if self.vola_videos and finished_racers:
                    # Get combined video coverage range
                    video_start = min(v.get('start_time_sec', float('inf')) for v in self.vola_videos)
                    video_end = max(v.get('end_time_sec', 0) for v in self.vola_videos)

                    outside_video = []
                    for r in finished_racers:
                        cam_start = r.get('camera_start_sec', 0)
                        cam_end = r.get('camera_end_sec', 0)
                        if cam_end < video_start or cam_start > video_end:
                            outside_video.append(r)

                    if outside_video:
                        def sec_to_time(sec):
                            h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
                            return f"{h:02d}:{m:02d}:{s:02d}"
                        print(f"\n     ⚠️  {len(outside_video)} racers fall OUTSIDE video coverage ({sec_to_time(video_start)}-{sec_to_time(video_end)}):")
                        for r in outside_video[:5]:
                            cam_start = r.get('camera_start_sec', 0)
                            print(f"       Bib {r.get('bib'):>3} ({r.get('name', 'N/A'):18}) @ {sec_to_time(cam_start)}")

        if self.montage_pairs:
            total_thumb = sum(
                m.thumbnail_size
                for pair in self.montage_pairs
                for m in pair.results.values()
            )
            total_full = sum(
                m.fullres_size
                for pair in self.montage_pairs
                for m in pair.results.values()
            )
            print(f"\n  Montages generated: {len(self.montage_pairs)}")
            print(f"  Total thumbnail size: {total_thumb / 1024:.0f} KB")
            print(f"  Total full-res size: {total_full / 1024 / 1024:.1f} MB")
            print(f"\n  Output directory: {self.session_dir}")


def main():
    parser = argparse.ArgumentParser(description="SkiFrames run detection and montage generation")
    parser.add_argument("input", nargs='*', help="Video file(s), directory, or 'rtsp' for live stream")
    parser.add_argument("-c", "--config", required=True, help="Session config JSON file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help=f"Montage FPS (overrides config, default: from config or {DEFAULT_FPS})")
    parser.add_argument("--use-config-videos", action="store_true", help="Use video list from config file (vola_videos)")

    args = parser.parse_args()

    runner = SkiFramesRunner(args.config, args.output, montage_fps=args.fps)

    # If --use-config-videos, process videos from config
    if args.use_config_videos:
        if runner.vola_videos:
            video_paths = [v['path'] for v in runner.vola_videos]
            print(f"Processing {len(video_paths)} videos from config")
            runner.run_on_videos(video_paths)
        else:
            print("No vola_videos in config")
    elif not args.input:
        print("Error: No input specified")
    elif args.input[0].lower() == 'rtsp':
        runner.run_on_rtsp()
    elif len(args.input) == 1 and os.path.isdir(args.input[0]):
        # Process all videos in directory
        video_files = sorted(
            list(Path(args.input[0]).glob("*.mp4")) +
            list(Path(args.input[0]).glob("*.mkv")) +
            list(Path(args.input[0]).glob("*.avi"))
        )
        print(f"Found {len(video_files)} video files")
        runner.run_on_videos([str(v) for v in video_files])
    elif len(args.input) == 1:
        runner.run_on_video(args.input[0])
    else:
        # Multiple video files specified
        runner.run_on_videos(args.input)


if __name__ == "__main__":
    main()

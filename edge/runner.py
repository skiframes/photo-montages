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

from detection import DetectionEngine, DetectionConfig, Run, RUN_CAPTURE_FPS
from montage import generate_montage, MontageResult, MontageResultPair, DEFAULT_FPS
from video_clip import generate_video_clip
from trajectory import generate_trajectory_video
from ghosttrail import generate_ghosttrail_video


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

        # Track generated montages (A/B pairs)
        self.montage_pairs: list[MontageResultPair] = []

        # Source FPS - can be overridden in config, otherwise detected from video
        self.source_fps_override = self.raw_config.get('source_fps')  # e.g., 30.0 or 60.0
        self.source_fps = self.source_fps_override or 30.0  # Default, updated when video opens

        # Vola racer data for naming montages
        self.vola_racers: List[Dict] = self.raw_config.get('vola_racers', [])

        # Limit number of athletes to process (0 = all)
        num_athletes = self.raw_config.get('num_athletes', 0)
        if num_athletes and num_athletes > 0 and self.vola_racers:
            print(f"  Limiting to first {num_athletes} athletes (of {len(self.vola_racers)})")
            self.vola_racers = self.vola_racers[:num_athletes]
        self.vola_race: str = self.raw_config.get('vola_race', '')
        self.vola_camera: str = self.raw_config.get('vola_camera', '')
        self.race_date: str = self.raw_config.get('race_date', datetime.now().strftime('%Y-%m-%d'))

        # Race info for overlay (same format as video stitcher)
        self.race_info: Dict = self.raw_config.get('race_info', {})

        # Section/staging config for batch processing recorded videos
        self.section_id: str = self.raw_config.get('section_id', '')     # e.g., "Cam1"
        self.run_number: str = self.raw_config.get('run_number', '')     # e.g., "run1"
        self.staging_dir: str = self.raw_config.get('staging_dir', '')   # e.g., "/tmp/montages/western-q-2026-02-22"
        self.race_slug: str = self.raw_config.get('race_slug', '')       # e.g., "western-q-2026-02-22"

        # Videos to process (from Vola API)
        self.vola_videos: List[Dict] = self.raw_config.get('vola_videos', [])

        # Selected logos for overlay (from UI logo selection)
        self.selected_logos: Optional[List[str]] = self.raw_config.get('selected_logos') or None
        self.logo_corner: str = self.raw_config.get('logo_corner', 'bottom-right')

        # Gate info for overlay (gate number, slope, distance, etc.)
        self.gate_info: Optional[Dict] = self.raw_config.get('gate_info') or None
        self.gate_info_corner: str = self.raw_config.get('gate_info_corner', 'top-right')

        # Video clip generation (can disable to save memory on J40)
        self.generate_videos: bool = self.raw_config.get('generate_videos', True)

        # Track which racers have been matched (prevent duplicate matches)
        self.matched_racer_indices: set = set()

        # Track detection count per base filename key for multi-detection support
        # Key: (gender_prefix, bib) for matched runs, ('unmatched', ts) for unmatched
        # Value: count of detections for that key
        self.detection_counter: Dict[tuple, int] = {}

        # If staging mode (batch processing), redirect ALL output to staging dir
        # so everything is in one location (important for 4T server batch processing)
        if self.section_id and self.staging_dir and self.run_number:
            self.staging_output_dir = os.path.join(self.staging_dir, self.section_id, self.run_number)
            os.makedirs(self.staging_output_dir, exist_ok=True)
            # Redirect session_dir to staging dir so metrics/manifest go there too
            self.session_dir = self.staging_output_dir
            # Write metadata for populate-manifest to know gender context
            gender = 'girls' if 'girl' in self.vola_race.lower() else 'boys' if 'boy' in self.vola_race.lower() else ''
            meta = {
                'vola_file': self.vola_race,
                'gender': gender,
                'section_id': self.section_id,
                'run_number': self.run_number,
                'race_slug': self.race_slug,
            }
            meta_path = os.path.join(self.staging_output_dir, '_meta.json')
            with open(meta_path, 'w') as mf:
                import json as _json
                _json.dump(meta, mf, indent=2)
        else:
            self.staging_output_dir = None
            # Normal mode: create session output directory
            os.makedirs(self.session_dir, exist_ok=True)

        # Create detection engine with callback and vola_racers for offset calculation
        self.engine = DetectionEngine(self.config, on_run_complete=self._on_run_complete,
                                       vola_racers=self.vola_racers)
        # Set metrics path for live detection chart
        self.engine.metrics_path = os.path.join(self.session_dir, 'detection_metrics.json')

        print(f"SkiFrames Runner initialized")
        print(f"  Session: {self.config.session_id}")
        print(f"  Output: {self.session_dir}")
        if self.staging_output_dir:
            print(f"  Staging: {self.staging_output_dir}")
            print(f"  Section: {self.section_id}, Run: {self.run_number}")
        print(f"  Montage FPS list: {self.montage_fps_list}")
        if self.source_fps_override:
            print(f"  Source FPS override: {self.source_fps_override}")
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
        # Filter out short runs (false positives)
        min_run_duration = self.raw_config.get('min_run_duration_seconds', 3.0)
        if run.duration < min_run_duration:
            print(f"  [RUN {run.run_number}] Skipped: duration {run.duration:.2f}s < min {min_run_duration}s")
            # Clean up temp frame files
            if hasattr(run, 'cleanup'):
                run.cleanup()
            return

        # Filter out long runs (coaches side-slipping, not racing)
        max_run_duration = self.raw_config.get('max_run_duration_seconds', 0)  # 0 = disabled
        if max_run_duration > 0 and run.duration > max_run_duration:
            print(f"  [RUN {run.run_number}] Skipped: duration {run.duration:.2f}s > max {max_run_duration}s (likely coach)")
            # Clean up temp frame files
            if hasattr(run, 'cleanup'):
                run.cleanup()
            return

        # Match this run to a Vola racer by timestamp (Boston clock time)
        racer = None
        if run.start_time and self.vola_racers:
            racer = self._find_matching_racer_by_timestamp(run.start_time)

        if racer:
            print(f"  Matched to racer: Bib {racer.get('bib')} ({racer.get('team', 'no team')})")
            print(f"  Matched so far: {len(self.matched_racer_indices)}/{len(self.vola_racers)} racers")

        # Get run duration for overlay display
        run_duration = run.duration if run.end_time else self.raw_config.get('run_duration_seconds')

        # Compute elapsed time for overlay
        elapsed_time = round(run.duration, 2) if run.end_time else None

        # Build race title for overlay (fallback if no race_info)
        race_title = ""
        if not self.race_info and self.vola_race:
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

        # Staging mode: output to {staging_dir}/{CamX}/{runN}/ with bib-based filenames
        # Include gender prefix to avoid collisions (boys/girls share bib numbers)
        # Always generate montages even for unmatched runs (coaches, timing mismatches)
        # — better to have too many and delete than miss athletes
        merged_results = {}
        if self.staging_output_dir:
            if racer:
                bib = racer.get('bib', run.run_number)
                # Determine gender prefix from racer data (preferred) or vola_race filename (legacy)
                racer_gender = racer.get('gender', '').lower()
                if 'women' in racer_gender or 'girl' in racer_gender or 'female' in racer_gender:
                    gender_prefix = 'g'
                elif 'men' in racer_gender or 'boy' in racer_gender or 'male' in racer_gender:
                    gender_prefix = 'b'
                elif self.vola_race and 'girl' in self.vola_race.lower():
                    gender_prefix = 'g'
                elif self.vola_race and 'boy' in self.vola_race.lower():
                    gender_prefix = 'b'
                else:
                    gender_prefix = ''
                staging_filename = f"{gender_prefix}{bib}" if gender_prefix else str(bib)
                det_key = (gender_prefix, bib)
            else:
                # Unmatched run — use timestamp-based filename so it's still generated
                ts = run.start_time.strftime('%H%M%S') if run.start_time else f"{run.run_number:03d}"
                staging_filename = f"unmatched_{ts}"
                det_key = ('unmatched', ts)
                print(f"  ⚠️  Generating montage for unmatched run (filename: {staging_filename})")

            # Track multiple detections per key (same bib can trigger multiple times)
            self.detection_counter[det_key] = self.detection_counter.get(det_key, 0) + 1
            det_count = self.detection_counter[det_key]
            det_id = f"d{det_count:03d}"

            for fps_val in self.montage_fps_list:
                fps_suffix = f"_{fps_val:.1f}fps"
                fps_filename = f"{staging_filename}_{det_id}{fps_suffix}"

                result = generate_montage(
                    frames=run.frames,
                    run_number=run.run_number,
                    timestamp=run.start_time,
                    output_dir=self.staging_output_dir,
                    session_id=self.config.session_id,
                    start_zone=self.raw_config.get('start_zone'),
                    end_zone=self.raw_config.get('end_zone'),
                    crop_zone=self.raw_config.get('crop_zone'),
                    source_fps=self.source_fps,
                    montage_fps=fps_val,
                    custom_filename=fps_filename,
                    run_view_folder=None,  # Flat output, no subfolders
                    run_duration_sec=run_duration,
                    race_title=race_title,
                    race_info=self.race_info,
                    elapsed_time=elapsed_time,
                    selected_logos=self.selected_logos,
                    gate_info=self.gate_info,
                    gate_info_corner=self.gate_info_corner,
                )

                if result:
                    fps_key = f"{fps_val:.1f}fps"
                    for variant_key, montage_result in result.results.items():
                        merged_results[fps_key] = montage_result

            # Save per-bib timing data for section time estimation
            import json as _json2
            timing_filename = f"{staging_filename}_{det_id}_timing.json"
            timing_path = os.path.join(self.staging_output_dir, timing_filename)
            timing_data = {
                'bib': racer.get('bib', None) if racer else None,
                'det_id': det_id,
                'gender': ('F' if gender_prefix == 'g' else 'M' if gender_prefix == 'b' else '') if racer else '',
                'matched': racer is not None,
                'section_elapsed_sec': round(run.duration, 2) if run.end_time else None,
                'start_trigger_time': run.start_time.isoformat() if run.start_time else None,
                'end_trigger_time': run.end_time.isoformat() if run.end_time else None,
            }
            with open(timing_path, 'w') as tf:
                _json2.dump(timing_data, tf, indent=2)

            # Generate video clip in staging directory
            try:
                video_filename = f"{staging_filename}_{det_id}.mp4"
                video_out_path = os.path.join(self.staging_output_dir, video_filename)

                # Build crop region (same as montage)
                vid_crop_region = self.raw_config.get('crop_zone')
                if not vid_crop_region and self.raw_config.get('start_zone') and self.raw_config.get('end_zone'):
                    try:
                        from montage import CropRegion
                        frame_h, frame_w = run.frames[0].shape[:2]
                        cr = CropRegion.from_zones(
                            self.raw_config['start_zone'],
                            self.raw_config['end_zone'],
                            frame_w, frame_h
                        )
                        vid_crop_region = {'x': cr.x, 'y': cr.y, 'w': cr.w, 'h': cr.h}
                    except Exception:
                        pass

                # Encode at RUN_CAPTURE_FPS (actual capture rate) for real-time playback
                staging_video_path = generate_video_clip(
                    frames=run.frames,
                    output_path=video_out_path,
                    source_fps=RUN_CAPTURE_FPS,
                    crop_region=vid_crop_region,
                    selected_logos=self.selected_logos,
                    logo_corner=self.logo_corner,
                    gate_info=self.gate_info,
                    gate_info_corner=self.gate_info_corner,
                )
                if staging_video_path:
                    size_kb = os.path.getsize(staging_video_path) / 1024
                    print(f"  Video clip: {video_filename} ({size_kb:.0f} KB)")
            except Exception as e:
                print(f"  Video clip skipped: {e}")

            # Generate GhostTrail stroboscopic slow-motion video
            try:
                ghosttrail_filename = f"{staging_filename}_{det_id}_ghosttrail.mp4"
                ghosttrail_out_path = os.path.join(self.staging_output_dir, ghosttrail_filename)
                ghosttrail_interval = self.raw_config.get('ghosttrail_interval', 4)
                ghosttrail_opacity = self.raw_config.get('ghosttrail_opacity', 1.0)
                ghosttrail_path = generate_ghosttrail_video(
                    frames=run.frames,
                    output_path=ghosttrail_out_path,
                    source_fps=RUN_CAPTURE_FPS,
                    slowmo_factor=4.0,  # 0.25x speed
                    impression_interval=ghosttrail_interval,
                    impression_opacity=ghosttrail_opacity,
                    crop_region=vid_crop_region,
                    selected_logos=self.selected_logos,
                    logo_corner=self.logo_corner,
                )
                if ghosttrail_path:
                    size_kb = os.path.getsize(ghosttrail_path) / 1024
                    print(f"  GhostTrail: {ghosttrail_filename} ({size_kb:.0f} KB)")
            except Exception as e:
                print(f"  GhostTrail skipped: {e}")

            # Trajectory generation disabled for now
            # try:
            #     tr_filename = f"{staging_filename}_TR.mp4"
            #     tr_out_path = os.path.join(self.staging_output_dir, tr_filename)
            #     tr_path = generate_trajectory_video(
            #         frames=run.frames, output_path=tr_out_path,
            #         source_fps=self.source_fps, crop_region=vid_crop_region,
            #     )
            #     if tr_path:
            #         size_kb = os.path.getsize(tr_path) / 1024
            #         print(f"  Trajectory: {tr_filename} ({size_kb:.0f} KB)")
            # except Exception as e:
            #     print(f"  Trajectory skipped: {e}")

            print(f"  Staged: {self.staging_output_dir}/{staging_filename}_{det_id}*.jpg")
        else:
            # Normal mode: output to session dir with name/bib filenames
            custom_filename = self._generate_filename(racer, run.run_number)
            output_folder = self._get_output_folder(racer)

            for fps_val in self.montage_fps_list:
                fps_suffix = f"_{fps_val:.1f}fps"
                fps_filename = f"{custom_filename}{fps_suffix}" if custom_filename else None

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
                    custom_filename=fps_filename,
                    run_view_folder=output_folder,
                    run_duration_sec=run_duration,
                    race_title=race_title,
                    race_info=self.race_info,
                    elapsed_time=elapsed_time,
                    selected_logos=self.selected_logos,
                    gate_info=self.gate_info,
                    gate_info_corner=self.gate_info_corner,
                )

                if result:
                    # Use FPS value as variant key (e.g., "4.0fps")
                    fps_key = f"{fps_val:.1f}fps"
                    for variant_key, montage_result in result.results.items():
                        merged_results[fps_key] = montage_result

        if merged_results:
            # Compute elapsed time (seconds) between start and end trigger zones
            elapsed_time = round(run.duration, 2) if run.end_time else None

            # Compute CLIP embedding from a single frame of the run
            # for athlete re-identification (clothing/appearance matching)
            # Only for training sessions — races already have named athletes
            # Uses raw frame (not composite) for better athlete distinction
            embedding = None
            session_type = self.raw_config.get('session_type', 'training')
            if session_type != 'race':
                try:
                    import embedder
                    if embedder.is_available():
                        # Build crop region from config (same as montage uses)
                        crop_region = self.raw_config.get('crop_zone')
                        if not crop_region and self.raw_config.get('start_zone') and self.raw_config.get('end_zone'):
                            from montage import CropRegion
                            frame_h, frame_w = run.frames[0].shape[:2]
                            cr = CropRegion.from_zones(
                                self.raw_config['start_zone'],
                                self.raw_config['end_zone'],
                                frame_w, frame_h
                            )
                            crop_region = {'x': cr.x, 'y': cr.y, 'w': cr.w, 'h': cr.h}
                        embedding = embedder.embed_frames(run.frames, crop_region)
                        if embedding:
                            print(f"  Embedding computed ({len(embedding)} dims)")
                except Exception as e:
                    print(f"  Embedding skipped: {e}")

            # Generate video clip from raw frames (can be disabled via config to save memory)
            # Skip in staging mode - videos already created with bib naming above
            video_clip_path = None
            if self.generate_videos and not self.staging_output_dir:
                try:
                    video_dir = os.path.join(self.session_dir, 'videos')
                    video_filename = f"run_{run.run_number:03d}_{run.start_time.strftime('%H%M%S')}.mp4"
                    video_out_path = os.path.join(video_dir, video_filename)

                    # Build crop region (reuse same logic as embedding/montage)
                    vid_crop_region = self.raw_config.get('crop_zone')
                    if not vid_crop_region and self.raw_config.get('start_zone') and self.raw_config.get('end_zone'):
                        try:
                            from montage import CropRegion
                            frame_h, frame_w = run.frames[0].shape[:2]
                            cr = CropRegion.from_zones(
                                self.raw_config['start_zone'],
                                self.raw_config['end_zone'],
                                frame_w, frame_h
                            )
                            vid_crop_region = {'x': cr.x, 'y': cr.y, 'w': cr.w, 'h': cr.h}
                        except Exception:
                            pass

                    # Encode at RUN_CAPTURE_FPS (actual capture rate) for real-time playback
                    video_clip_path = generate_video_clip(
                        frames=run.frames,
                        output_path=video_out_path,
                        source_fps=RUN_CAPTURE_FPS,
                        crop_region=vid_crop_region,
                        selected_logos=self.selected_logos,
                        logo_corner=self.logo_corner,
                        gate_info=self.gate_info,
                        gate_info_corner=self.gate_info_corner,
                    )
                    if video_clip_path:
                        size_kb = os.path.getsize(video_clip_path) / 1024
                        print(f"  Video clip: {video_filename} ({size_kb:.0f} KB)")
                except Exception as e:
                    print(f"  Video clip skipped: {e}")

            # Generate GhostTrail stroboscopic slow-motion video
            # Skip in staging mode - ghosttrail already created with bib naming above
            ghosttrail_path = None
            if self.generate_videos and not self.staging_output_dir:
                try:
                    ghosttrail_dir = os.path.join(self.session_dir, 'ghosttrail')
                    ghosttrail_filename = f"run_{run.run_number:03d}_{run.start_time.strftime('%H%M%S')}_ghosttrail.mp4"
                    ghosttrail_out_path = os.path.join(ghosttrail_dir, ghosttrail_filename)
                    ghosttrail_interval = self.raw_config.get('ghosttrail_interval', 4)
                    ghosttrail_opacity = self.raw_config.get('ghosttrail_opacity', 1.0)
                    ghosttrail_path = generate_ghosttrail_video(
                        frames=run.frames,
                        output_path=ghosttrail_out_path,
                        source_fps=RUN_CAPTURE_FPS,
                        slowmo_factor=4.0,  # 0.25x speed
                        impression_interval=ghosttrail_interval,
                        impression_opacity=ghosttrail_opacity,
                        crop_region=vid_crop_region,
                        selected_logos=self.selected_logos,
                        logo_corner=self.logo_corner,
                    )
                    if ghosttrail_path:
                        size_kb = os.path.getsize(ghosttrail_path) / 1024
                        print(f"  GhostTrail: {ghosttrail_filename} ({size_kb:.0f} KB)")
                except Exception as e:
                    print(f"  GhostTrail skipped: {e}")

            # Trajectory generation disabled for now
            trajectory_path = None
            # try:
            #     tr_dir = os.path.join(self.session_dir, 'trajectories')
            #     tr_filename = f"run_{run.run_number:03d}_{run.start_time.strftime('%H%M%S')}_TR.mp4"
            #     tr_out_path = os.path.join(tr_dir, tr_filename)
            #     trajectory_path = generate_trajectory_video(
            #         frames=run.frames, output_path=tr_out_path,
            #         source_fps=self.source_fps, crop_region=vid_crop_region,
            #     )
            #     if trajectory_path:
            #         size_kb = os.path.getsize(trajectory_path) / 1024
            #         print(f"  Trajectory: {tr_filename} ({size_kb:.0f} KB)")
            # except Exception as e:
            #     print(f"  Trajectory skipped: {e}")

            merged_pair = MontageResultPair(
                run_number=run.run_number,
                timestamp=run.start_time,
                results=merged_results,
                elapsed_time=elapsed_time,
                embedding=embedding,
                video_clip_path=video_clip_path,
                trajectory_path=trajectory_path,
                ghosttrail_path=ghosttrail_path,
            )
            if racer:
                merged_pair.racer_bib = racer.get('bib')
            self.montage_pairs.append(merged_pair)
            print(f"  Added run {run.run_number} to montage_pairs (total: {len(self.montage_pairs)})")

        print(f"  Updating manifest (merged_results: {bool(merged_results)}, pairs: {len(self.montage_pairs)})")
        self._update_manifest()

        # Clean up temp frame files from disk to prevent disk fill
        if hasattr(run, 'cleanup'):
            run.cleanup()
        import gc
        gc.collect()

    def _update_manifest(self):
        """Update the session manifest file."""
        print(f"  [MANIFEST] Writing manifest with {len(self.montage_pairs)} runs to {self.session_dir}")
        runs = []
        for pair in self.montage_pairs:
            run_entry = {
                "run_number": pair.run_number,
                "timestamp": pair.timestamp.isoformat(),
                "elapsed_time": pair.elapsed_time,
                "embedding": pair.embedding,
                "variants": {}
            }
            # Add video clip URL if available
            if pair.video_clip_path:
                try:
                    run_entry["video_url"] = os.path.relpath(pair.video_clip_path, self.session_dir)
                except ValueError:
                    run_entry["video_url"] = os.path.basename(pair.video_clip_path)
            # Add trajectory video URL if available
            if pair.trajectory_path:
                try:
                    run_entry["trajectory_url"] = os.path.relpath(pair.trajectory_path, self.session_dir)
                except ValueError:
                    run_entry["trajectory_url"] = os.path.basename(pair.trajectory_path)
            # Add GhostTrail video URL if available
            if pair.ghosttrail_path:
                try:
                    run_entry["ghosttrail_url"] = os.path.relpath(pair.ghosttrail_path, self.session_dir)
                except ValueError:
                    run_entry["ghosttrail_url"] = os.path.basename(pair.ghosttrail_path)
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
            "min_run_duration_seconds": self.raw_config.get('min_run_duration_seconds', 3.0),
            "runs": runs,
        }

        manifest_path = os.path.join(self.session_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def run_on_video(self, video_path: str):
        """Process a video file."""
        print(f"\nProcessing video: {video_path}")

        # Get video FPS for montage generation (config override takes precedence)
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if self.source_fps_override:
                print(f"  Source FPS: {self.source_fps_override} (config override, detected {detected_fps:.1f})")
                self.source_fps = self.source_fps_override
            else:
                print(f"  Source FPS: {detected_fps:.1f} (auto-detected)")
                self.source_fps = detected_fps

        # Parse video start time from filename (Reolink format)
        video_start_time = parse_reolink_video_start_time(video_path, self.race_date)
        if video_start_time:
            print(f"  Video start time (from filename): {video_start_time.strftime('%H:%M:%S')}")

        # Pass video_start_time for accurate timestamp matching with Vola data
        self.engine.run_on_video(video_path, video_start_time=video_start_time)
        self._print_summary()

    def all_athletes_matched(self) -> bool:
        """Check if all expected athletes have been matched."""
        if not self.vola_racers:
            return False
        return len(self.matched_racer_indices) >= len(self.vola_racers)

    def _get_last_racer_end_sec(self) -> float:
        """Get the latest camera_end_sec across all Vola racers."""
        if not self.vola_racers:
            return 0
        return max(r.get('camera_end_sec', 0) for r in self.vola_racers)

    def run_on_videos(self, video_paths: List[str]):
        """Process multiple video files in sequence."""
        print(f"\nProcessing {len(video_paths)} videos...")

        # Compute cutoff: stop processing videos that start after the last racer's window + buffer
        last_racer_end_sec = self._get_last_racer_end_sec()
        # 2-minute buffer past last racer to catch late triggers / timing drift
        past_racers_cutoff_sec = last_racer_end_sec + 120 if last_racer_end_sec > 0 else 0

        for i, video_path in enumerate(video_paths):
            # Stop early if all athletes have been matched
            if self.all_athletes_matched():
                print(f"\n✅ All {len(self.vola_racers)} athletes matched — stopping early "
                      f"(skipped {len(video_paths) - i} remaining videos)")
                break

            print(f"\nProcessing video {i+1}/{len(video_paths)}: {video_path}")

            # Get video FPS for montage generation (config override takes precedence)
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                if self.source_fps_override:
                    self.source_fps = self.source_fps_override
                else:
                    self.source_fps = detected_fps

            # Parse video start time from filename (Reolink format)
            video_start_time = parse_reolink_video_start_time(video_path, self.race_date)
            if video_start_time:
                print(f"  Video start time (from filename): {video_start_time.strftime('%H:%M:%S')}")

            # Skip videos that start well past the last racer's time window
            if past_racers_cutoff_sec > 0 and video_start_time:
                video_start_sec = (video_start_time.hour * 3600 +
                                   video_start_time.minute * 60 +
                                   video_start_time.second)
                if video_start_sec > past_racers_cutoff_sec and self.engine.current_run is None:
                    def _sec_to_time(s):
                        return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"
                    print(f"\n⏭️  Stopping — video starts at {_sec_to_time(video_start_sec)} which is past "
                          f"last racer's window ({_sec_to_time(last_racer_end_sec)}) + 2min buffer. "
                          f"Matched {len(self.matched_racer_indices)}/{len(self.vola_racers)} racers. "
                          f"(skipped {len(video_paths) - i} remaining videos)")
                    break

            # Pass video_start_time for accurate timestamp matching with Vola data
            self.engine.run_on_video(video_path, video_start_time=video_start_time)

            # Check again after each video in case all matched mid-video
            if self.all_athletes_matched():
                print(f"\n✅ All {len(self.vola_racers)} athletes matched after video {i+1}")

        # Warn if there's still an incomplete run after all videos processed
        if self.engine.current_run is not None:
            incomplete = self.engine.current_run
            print(f"\n  ⚠️  Run {incomplete.run_number} still incomplete after all videos "
                  f"(started {incomplete.start_time.strftime('%H:%M:%S')}, {len(incomplete.frames)} frames) "
                  f"— no END zone trigger found. Discarding.")
            self.engine.current_run = None

        self._print_summary()

    def run_on_rtsp(self, rtsp_url: Optional[str] = None):
        """Monitor RTSP stream."""
        url = rtsp_url or self.config.camera_url
        print(f"\nMonitoring RTSP stream: {url}")
        # Probe stream FPS before starting detection (config override takes precedence)
        import cv2
        probe = cv2.VideoCapture(url)
        if probe.isOpened():
            detected_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
            probe.release()
            if self.source_fps_override:
                print(f"  Source FPS: {self.source_fps_override} (config override, detected {detected_fps:.1f})")
                self.source_fps = self.source_fps_override
            else:
                print(f"  Source FPS: {detected_fps:.1f} (auto-detected)")
                self.source_fps = detected_fps
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

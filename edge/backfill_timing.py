#!/usr/bin/env python3
"""
Backfill _timing.json files for existing staged montages.

Reads runner session manifests, matches runs to bibs using Vola timing data,
and writes _timing.json files in the staging directory so the race manifest
populator can include section_time in the web gallery.
"""

import json
import os
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path

# Camera offset percentages (how far into the run the camera covers)
CAMERAS = {
    'R1': {'offset_pct': 45},   # Cam3, gates 17-21
    'R2': {'offset_pct': 10},   # Cam1, gates 6-7 (early in course)
    'R3': {'offset_pct': 30},   # Cam2, gates 7-14
}


def parse_vola_time(time_str):
    """Parse Vola time string like '9h07:51.73224' to seconds since midnight."""
    time_str = time_str.strip()
    m = re.match(r'(\d+)h(\d+):(\d+(?:\.\d+)?)', time_str)
    if not m:
        return None
    hours = int(m.group(1))
    minutes = int(m.group(2))
    seconds = float(m.group(3))
    return hours * 3600 + minutes * 60 + seconds


def parse_vola_csv(filepath):
    """Parse Vola CSV file. Returns dict of {bib: start_seconds}."""
    results = {}
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')
    if len(lines) < 2:
        return results
    sep = ',' if ',' in lines[1] else '\t'

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(sep)
        if len(parts) < 2:
            continue

        bib_str = parts[0].strip()
        time_str = parts[1].strip()

        if bib_str == 'Num' or not bib_str:
            continue
        if 'Did Not Start' in time_str or 'Did Not Finish' in time_str:
            continue
        if not bib_str.isdigit():
            continue

        bib = int(bib_str)
        start_seconds = parse_vola_time(time_str)
        if start_seconds is not None:
            results[bib] = start_seconds

    return results


def match_runs_to_bibs(manifest_runs, vola_start_times, camera_id, estimated_duration=40.0):
    """
    Match detected runs to Vola bibs by timestamp.

    Returns list of (bib, elapsed_time) tuples for matched runs.
    """
    camera_offset_pct = CAMERAS.get(camera_id, {}).get('offset_pct', 0) / 100.0

    # Build camera timing windows for each bib
    bib_windows = {}
    for bib, start_sec in vola_start_times.items():
        camera_start_sec = start_sec + (estimated_duration * camera_offset_pct)
        camera_duration = estimated_duration * 0.2  # Camera covers ~20% of run
        camera_end_sec = camera_start_sec + camera_duration
        bib_windows[bib] = (camera_start_sec, camera_end_sec)

    matched = []
    used_bibs = set()

    for run in manifest_runs:
        # Parse run timestamp to seconds since midnight
        ts_str = run.get('timestamp', '')
        try:
            ts = datetime.fromisoformat(ts_str)
            run_sec = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1_000_000
        except (ValueError, AttributeError):
            continue

        elapsed = run.get('elapsed_time')
        if elapsed is None:
            continue

        # Try to extract bib from filename in variants
        bib_from_name = None
        for var_key, var_data in run.get('variants', {}).items():
            filename = var_data.get('thumbnail', '') or var_data.get('fullres', '')
            # Match patterns like "Ken Stewart_1_6.0fps" -> bib 1
            name_match = re.search(r'_(\d+)_[\d.]+fps', filename)
            if name_match:
                bib_from_name = int(name_match.group(1))
                break

        if bib_from_name and bib_from_name not in used_bibs:
            matched.append((bib_from_name, elapsed))
            used_bibs.add(bib_from_name)
            continue

        # Fall back to timestamp matching
        tolerance = 15.0  # seconds
        best_bib = None
        best_diff = float('inf')

        for bib, (cam_start, cam_end) in bib_windows.items():
            if bib in used_bibs:
                continue
            cam_mid = (cam_start + cam_end) / 2
            diff = abs(run_sec - cam_mid)
            if diff < tolerance and diff < best_diff:
                best_bib = bib
                best_diff = diff

        if best_bib is not None:
            matched.append((best_bib, elapsed))
            used_bibs.add(best_bib)

    return matched


def main():
    # Configuration for this race
    race_slug = 'western-q-2026-02-22'
    staging_base = Path('/Volumes/OWC_48/data/montages') / race_slug
    vola_dir = Path('/Volumes/OWC_48/data/vola/U12_U14_02-22-2026')
    output_base = Path('/Users/paul2/skiframes/photo-montages/edge/output')

    # Camera R2 = Cam1 for this race
    camera_id = 'R2'
    cam_id = 'Cam1'
    run_key = 'run1'

    staging_dir = staging_base / cam_id / run_key

    if not staging_dir.exists():
        print(f"Error: staging dir {staging_dir} does not exist")
        sys.exit(1)

    # Parse Vola data for both genders
    girls_vola = parse_vola_csv(str(vola_dir / 'girls-run1.csv'))
    boys_vola = parse_vola_csv(str(vola_dir / 'boys-run1.csv'))

    print(f"Vola: {len(girls_vola)} girls, {len(boys_vola)} boys")

    # Find runner session manifests
    session_dirs = sorted(output_base.glob('*_R2_*'))
    print(f"Found {len(session_dirs)} session dirs for camera R2")

    all_girl_matches = {}  # {bib: elapsed_time}
    all_boy_matches = {}

    for session_dir in session_dirs:
        manifest_path = session_dir / 'manifest.json'
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        runs = manifest.get('runs', [])
        print(f"\nSession {session_dir.name}: {len(runs)} runs")

        # Try matching against girls first (they race first at ~09:07)
        girl_matches = match_runs_to_bibs(runs, girls_vola, camera_id)
        boy_matches = match_runs_to_bibs(runs, boys_vola, camera_id)

        print(f"  Girl matches: {[(b, t) for b, t in girl_matches]}")
        print(f"  Boy matches: {[(b, t) for b, t in boy_matches]}")

        # Check which gender has more matches — the vola file that was loaded
        # determines which bibs were actually matched during processing
        for bib, elapsed in girl_matches:
            if bib not in all_girl_matches:
                all_girl_matches[bib] = elapsed

        for bib, elapsed in boy_matches:
            if bib not in all_boy_matches:
                all_boy_matches[bib] = elapsed

    # Now check which bibs have staged montages and write timing files
    existing_bibs = set()
    for f in staging_dir.iterdir():
        m = re.match(r'^([gb])(\d+)_[\d.]+fps\.jpg$', f.name)
        if m and '_thumb' not in f.name:
            gender_char = m.group(1)
            bib = int(m.group(2))
            existing_bibs.add((gender_char, bib))

    unique_bibs = set()
    for gc, bib in existing_bibs:
        unique_bibs.add((gc, bib))

    print(f"\nStaged montage bibs: {sorted(unique_bibs)}")

    written = 0
    for gender_char, bib in sorted(unique_bibs):
        if gender_char == 'g':
            elapsed = all_girl_matches.get(bib)
            gender_code = 'F'
        else:
            elapsed = all_boy_matches.get(bib)
            gender_code = 'M'

        if elapsed is None:
            print(f"  WARNING: No timing data for {gender_char}{bib}")
            continue

        # Find the fps variants for this bib to get exact staging filename prefix
        fps_files = list(staging_dir.glob(f'{gender_char}{bib}_*fps.jpg'))
        if not fps_files:
            continue

        # Use the first FPS variant's prefix for the timing filename
        # e.g., g1_4.0fps.jpg -> staging_filename = g1_4.0fps
        sample_name = fps_files[0].stem  # e.g., "g1_4.0fps"
        staging_filename = sample_name  # Use the full fps filename as prefix

        timing_path = staging_dir / f'{staging_filename}_timing.json'
        timing_data = {
            'bib': bib,
            'gender': gender_code,
            'section_elapsed_sec': round(elapsed, 2),
            'start_trigger_time': None,  # Not available from manifest
            'end_trigger_time': None,
            'source': 'backfilled from runner manifest'
        }

        with open(timing_path, 'w') as tf:
            json.dump(timing_data, tf, indent=2)

        print(f"  Wrote {timing_path.name}: {gender_char}{bib} = {elapsed:.2f}s")
        written += 1

    print(f"\nDone: wrote {written} timing files to {staging_dir}")


if __name__ == '__main__':
    main()

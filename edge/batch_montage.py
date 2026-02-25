#!/usr/bin/env python3
"""
Batch montage processing script for server-side processing.
Run this on the Tesla GPU server without needing a web UI.

Usage:
    python3 batch_montage.py --race-date 2026-02-22 --run run1 --camera R1

Example on server:
    cd /home/pa91/skiframes/photo-montages
    source venv/bin/activate
    export SKIFRAMES_DATA_DIR=/home/pa91/data
    python3 edge/batch_montage.py --race-date 2026-02-22 --run run1 --camera R1 \\
        --config edge/config/montage_config_20260222_run1_R1.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Use larger temp directory if available
_alt_tmp = os.path.expanduser('~/tmp')
if os.path.isdir(_alt_tmp) and not os.environ.get('TMPDIR'):
    os.environ['TMPDIR'] = _alt_tmp
    os.environ['TEMP'] = _alt_tmp
    os.environ['TMP'] = _alt_tmp
    import tempfile
    tempfile.tempdir = _alt_tmp
    print(f"[batch] Using temp directory: {_alt_tmp}")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from runner import SkiFramesRunner

# Data directory from environment
DATA_BASE_DIR = Path(os.environ.get('SKIFRAMES_DATA_DIR', '/Volumes/OWC_48/data'))
VOLA_DIR = DATA_BASE_DIR / 'vola'
RECORDINGS_DIR = DATA_BASE_DIR / 'recordings'

# Project root for finding race manifests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_RACES_DIR = PROJECT_ROOT / 'web' / 'races'

# Default montage staging directory (same as app.py)
MONTAGES_DIR = Path(os.environ.get('SKIFRAMES_MONTAGES_DIR', '/tmp/montages'))


def find_race_manifest(race_date: str):
    """
    Find the race manifest for a given date.
    Scans web/races/ directories for a race_manifest.json with matching date.
    Returns (race_slug, manifest_data) or (None, None).
    """
    if not WEB_RACES_DIR.exists():
        return None, None

    for race_dir in WEB_RACES_DIR.iterdir():
        if not race_dir.is_dir():
            continue
        manifest_path = race_dir / 'race_manifest.json'
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                event = manifest.get('event', {})
                if event.get('date') == race_date:
                    return race_dir.name, manifest
            except (json.JSONDecodeError, IOError):
                continue

    return None, None


def get_camera_mapping(manifest, edge_camera_id: str):
    """
    Map edge camera ID (R1, R2, R3, Axis) to web section ID (Cam1, Cam2, Cam3).
    Returns section_id or None.
    """
    if not manifest:
        return None
    for cam in manifest.get('cameras', []):
        if cam.get('edge_camera') == edge_camera_id:
            return cam['id']
    return None


def find_vola_dir_for_date(race_date: str) -> Path:
    """
    Find the Vola directory for a given race date.
    race_date: YYYY-MM-DD format
    Vola dirs: U12_U14_MM-DD-YYYY/
    """
    # Convert YYYY-MM-DD to MM-DD-YYYY for matching
    parts = race_date.split('-')
    if len(parts) == 3:
        date_suffix = f"{parts[1]}-{parts[2]}-{parts[0]}"
    else:
        date_suffix = race_date

    for d in VOLA_DIR.iterdir():
        if d.is_dir() and date_suffix in d.name:
            return d

    raise FileNotFoundError(f"No Vola directory found for date {race_date} (looking for *{date_suffix}* in {VOLA_DIR})")


def parse_vola_csv(csv_path: str) -> dict:
    """Parse Vola CSV to get start times. Returns {bib: start_time_seconds}."""
    import csv

    start_times = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                bib = int(row[0])
                # Parse time - could be HH:MM:SS.mmm or seconds
                time_str = row[2].strip() if len(row) > 2 else ''
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        h, m, s = parts
                        seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    elif len(parts) == 2:
                        m, s = parts
                        seconds = int(m) * 60 + float(s)
                    else:
                        continue
                elif 'h' in time_str:
                    h_part, rest = time_str.split('h')
                    m_part, s_part = rest.split(':')
                    seconds = int(h_part) * 3600 + int(m_part) * 60 + float(s_part)
                else:
                    seconds = float(time_str)

                start_times[bib] = seconds
            except (ValueError, IndexError):
                continue

    return start_times


def find_recordings(camera_id: str, race_date: str) -> list:
    """
    Find video recordings for a specific camera on a given date.

    Args:
        camera_id: Camera identifier (R1, R2, R3, Axis)
        race_date: Date in YYYY-MM-DD format

    Returns:
        List of video file paths
    """
    date_compact = race_date.replace('-', '')  # YYYYMMDD

    camera_folders = {
        'R1': 'sd_R1',
        'R2': 'sd_R2',
        'R3': 'sd_R3',
        'Axis': 'sd_Axis'
    }

    folder = camera_folders.get(camera_id)
    if not folder:
        print(f"Warning: Unknown camera {camera_id}")
        return []

    videos = []

    if camera_id == 'Axis':
        # Try server structure: /sd_Axis/YYYYMMDD/**/*.mkv
        axis_path = RECORDINGS_DIR / folder / date_compact
        if axis_path.exists():
            videos = sorted(axis_path.glob('**/*.mkv'))
        else:
            # Mac structure: /sd_Axis/sdcard/YYYYMMDD/*.mkv
            axis_path = RECORDINGS_DIR / folder / 'sdcard' / date_compact
            if axis_path.exists():
                videos = sorted(axis_path.glob('*.mkv'))
    else:
        # Reolink cameras
        # Server: /sd_R1/YYYY-MM-DD/*.mp4
        reolink_path = RECORDINGS_DIR / folder / race_date
        if reolink_path.exists():
            videos = sorted(reolink_path.glob('*.mp4'))
        else:
            # Mac: /sd_R1/sdcard/Mp4Record/YYYYMMDD/*.mp4
            reolink_path = RECORDINGS_DIR / folder / 'sdcard' / 'Mp4Record' / date_compact
            if reolink_path.exists():
                videos = sorted(reolink_path.glob('*.mp4'))

    return [str(v) for v in videos]


def main():
    parser = argparse.ArgumentParser(description='Batch montage generation for ski race footage')
    parser.add_argument('--race-date', type=str, required=True, help='Race date in YYYY-MM-DD format')
    parser.add_argument('--run', type=str, default='run1', help='Run number (run1 or run2)')
    parser.add_argument('--camera', type=str, default='R1', help='Edge camera ID (R1, R2, R3, Axis)')
    parser.add_argument('--config', type=str, help='Path to JSON config file (from UI)')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: MONTAGES_DIR/race_slug)')
    parser.add_argument('--logos', type=str, help='Comma-separated list of logo filenames')
    parser.add_argument('--fps-list', type=str, default='4.0', help='Comma-separated list of montage FPS values')
    parser.add_argument('--test', type=int, default=0, help='Only process first N athletes (0 = all, default: 0)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (unused, for compat)')

    args = parser.parse_args()

    print("=" * 60)
    print("Skiframes Batch Montage Generator")
    print("=" * 60)
    print(f"Data directory: {DATA_BASE_DIR}")
    print(f"Montages directory: {MONTAGES_DIR}")
    print(f"Race date: {args.race_date}")
    print(f"Run: {args.run}")
    print(f"Camera: {args.camera}")
    print()

    # Load config file if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
        else:
            print(f"Warning: Config file not found: {config_path}")

    # Look up race manifest to get proper race_slug and camera mapping
    race_slug, manifest = find_race_manifest(args.race_date)
    if race_slug:
        print(f"Race manifest found: {race_slug}")
    else:
        race_slug = f"batch-{args.race_date}"
        print(f"No race manifest found, using: {race_slug}")

    # Map edge camera ID (R1) to web section ID (Cam1)
    section_id = get_camera_mapping(manifest, args.camera) if manifest else None
    if section_id:
        print(f"Camera mapping: {args.camera} -> {section_id}")
    else:
        # Fallback: use camera ID as section ID
        section_id = args.camera
        print(f"No camera mapping found, using: {section_id}")

    # All output goes to one directory: MONTAGES_DIR/race_slug
    output_dir = args.output or str(MONTAGES_DIR / race_slug)
    staging_dir = output_dir

    print(f"Output directory: {output_dir}")
    print()

    # Find Vola data
    try:
        vola_dir = find_vola_dir_for_date(args.race_date)
        print(f"Vola directory: {vola_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse Vola CSVs — boys and girls separately (bibs overlap between genders!)
    boys_csv = list(vola_dir.glob(f'boys-{args.run}.csv'))
    girls_csv = list(vola_dir.glob(f'girls-{args.run}.csv'))

    racers = []

    if boys_csv:
        print(f"Boys CSV: {boys_csv[0]}")
        boys_times = parse_vola_csv(str(boys_csv[0]))
        print(f"  Boys: {len(boys_times)} racers")
        for bib in sorted(boys_times.keys()):
            start_sec = boys_times[bib]
            racers.append({
                'bib': bib,
                'name': f'Bib{bib}',
                'team': '',
                'gender': 'Men',
                'camera_start_sec': start_sec,
                'camera_end_sec': start_sec + 30.0,
                'finished': True,
            })

    if girls_csv:
        print(f"Girls CSV: {girls_csv[0]}")
        girls_times = parse_vola_csv(str(girls_csv[0]))
        print(f"  Girls: {len(girls_times)} racers")
        for bib in sorted(girls_times.keys()):
            start_sec = girls_times[bib]
            racers.append({
                'bib': bib,
                'name': f'Bib{bib}',
                'team': '',
                'gender': 'Women',
                'camera_start_sec': start_sec,
                'camera_end_sec': start_sec + 30.0,
                'finished': True,
            })

    if not racers:
        print(f"Error: No Vola timing data found for {args.run}")
        sys.exit(1)

    # Sort by start time (interleaved boys + girls)
    racers.sort(key=lambda r: r['camera_start_sec'])
    print(f"Found {len(racers)} total racers with timing data")

    # Find video recordings
    videos = find_recordings(args.camera, args.race_date)
    if not videos:
        print(f"Error: No video recordings found for camera {args.camera} on {args.race_date}")
        sys.exit(1)

    print(f"Found {len(videos)} video files for camera {args.camera}")
    for v in videos[:3]:
        print(f"  {Path(v).name}")
    if len(videos) > 3:
        print(f"  ... and {len(videos) - 3} more")

    # Get logos
    selected_logos = config.get('selected_logos', [])
    if args.logos:
        selected_logos = [l.strip() for l in args.logos.split(',')]

    # Get FPS list
    fps_list = config.get('montage_fps_list', [4.0])
    if args.fps_list:
        fps_list = [float(f.strip()) for f in args.fps_list.split(',')]

    runner_config = {
        'session_id': f'batch_{args.race_date}_{args.run}_{args.camera}',
        'session_type': 'race',
        'camera_id': args.camera,
        'camera_url': '',
        'start_zone': config.get('start_zone', {'x': 0, 'y': 0, 'w': 100, 'h': 50}),
        'end_zone': config.get('end_zone'),
        'crop_zone': config.get('crop_zone'),
        'detection_threshold': config.get('detection_threshold', 25),
        'min_pixel_change_pct': config.get('min_pixel_change_pct', 5.0),
        'min_brightness': config.get('min_brightness', 94),
        'montage_fps_list': fps_list,
        'selected_logos': selected_logos if selected_logos else None,
        'vola_racers': racers,
        'vola_race': args.race_date,
        'vola_camera': args.camera,
        'vola_videos': [{'path': v, 'name': Path(v).name} for v in videos],
        'race_date': args.race_date,
        'race_info': config.get('race_info', {}),
        # Limit athletes (0 = all)
        'num_athletes': args.test,
        # Staging config — uses mapped section_id (Cam1) not edge camera (R1)
        'section_id': section_id,
        'run_number': args.run,
        'staging_dir': staging_dir,
        'race_slug': race_slug,
    }

    # Write temp config for runner
    import tempfile
    config_fd, config_path = tempfile.mkstemp(suffix='.json', prefix='montage_batch_')
    with os.fdopen(config_fd, 'w') as f:
        json.dump(runner_config, f, indent=2)

    print(f"\nStarting batch montage processing...")
    print(f"Output: {staging_dir}/{section_id}/{args.run}/")
    if selected_logos:
        print(f"Logos: {', '.join(selected_logos)}")
    print(f"FPS: {fps_list}")
    print()

    try:
        # Create runner and process all videos
        # output_dir passed here is only used for the default session_dir,
        # but staging mode overrides session_dir to staging_output_dir
        runner = SkiFramesRunner(config_path, output_dir)
        runner.run_on_videos(videos)

        print()
        print("=" * 60)
        print(f"Complete! Output: {staging_dir}/{section_id}/{args.run}/")
        print("=" * 60)
    finally:
        # Clean up temp config
        try:
            os.unlink(config_path)
        except OSError:
            pass


if __name__ == '__main__':
    main()

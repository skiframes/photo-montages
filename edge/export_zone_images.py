#!/usr/bin/env python3
"""
Export trigger zone calibration images for race documentation.

Reads edge config JSON files (with zone coordinates), grabs a representative
frame from the corresponding video file, draws START/END/CROP zone rectangles,
and saves annotated images to the race docs directory.

Usage:
    python3 export_zone_images.py --race-date 2026-02-22

Or for a specific config:
    python3 export_zone_images.py --config path/to/config.json --output path/to/output.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# Zone colors (BGR) — matching calibration UI CSS colors
ZONE_COLORS = {
    'crop':  {'border': (245, 158, 59),   'fill': (245, 158, 59),   'label': 'CROP'},   # blue
    'start': {'border': (94, 197, 34),    'fill': (94, 197, 34),    'label': 'START'},   # green
    'end':   {'border': (68, 68, 239),    'fill': (68, 68, 239),    'label': 'END'},     # red
}

# Section ID to section number mapping
SECTION_MAP = {
    'Cam1': 1,
    'Cam2': 2,
    'Cam3': 3,
}


def grab_frame_from_video(video_path, offset_sec=2.0):
    """Grab a single frame from a video file at the given offset."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_frame = int(offset_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Cannot read frame at {offset_sec}s from: {video_path}")

    return frame


def draw_zone_rect(frame, zone, zone_type, thickness=3):
    """Draw a zone rectangle with semi-transparent fill and label."""
    if zone is None:
        return frame

    colors = ZONE_COLORS[zone_type]
    x, y, w, h = zone['x'], zone['y'], zone['w'], zone['h']

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colors['fill'], -1)
    alpha = 0.25 if zone_type == 'crop' else 0.30
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), colors['border'], thickness)

    # Label background + text
    label = colors['label']
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2 if zone_type == 'crop' else 1.0
    label_thickness = 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
    padding = 6

    # Position label at top-left of zone
    lx = x + 4
    ly = y + th + padding + 2

    # Label background
    cv2.rectangle(frame,
                  (lx - 2, ly - th - padding),
                  (lx + tw + padding, ly + baseline + 2),
                  colors['border'], -1)

    # Label text (white on colored background)
    cv2.putText(frame, label, (lx + 2, ly - 2),
                font, font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)

    return frame


def generate_zone_image(config_path, output_path=None):
    """Generate an annotated zone image from a config file."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    camera_id = cfg.get('camera_id', '?')
    section_id = cfg.get('section_id', '?')
    section_num = SECTION_MAP.get(section_id, '?')

    # Find a video file to grab a frame from
    video_path = None
    vola_videos = cfg.get('vola_videos', [])
    if vola_videos:
        # Use first video
        video_path = vola_videos[0].get('path')

    if not video_path or not Path(video_path).exists():
        print(f"  No video file found for {camera_id} ({section_id})")
        return None

    print(f"  Grabbing frame from: {Path(video_path).name}")
    frame = grab_frame_from_video(video_path)

    h, w = frame.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # Draw zones in order: CROP first (background), then START and END on top
    crop_zone = cfg.get('crop_zone')
    start_zone = cfg.get('start_zone')
    end_zone = cfg.get('end_zone')

    frame = draw_zone_rect(frame, crop_zone, 'crop')
    frame = draw_zone_rect(frame, start_zone, 'start')
    frame = draw_zone_rect(frame, end_zone, 'end')

    # Add info text at bottom
    info = f"Section {section_num} | Camera {camera_id} ({section_id})"
    if cfg.get('race_date'):
        info += f" | {cfg['race_date']}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(info, font, 0.7, 2)
    # Black background bar at bottom
    cv2.rectangle(frame, (0, h - th - 16), (tw + 20, h), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, h - 8), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Save
    if output_path is None:
        output_path = config_path.parent / f"section_{section_num}_zones.jpg"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({size_kb:.0f} KB)")

    return str(output_path)


def find_configs_for_race(race_date, zones_dir=None):
    """Find the latest config files for each camera for a given race date."""
    if zones_dir is None:
        zones_dir = Path(__file__).parent / 'config' / 'zones'

    configs = {}  # camera_id -> config_path

    for config_file in sorted(zones_dir.glob('*.json')):
        # Skip corrected_offset variants
        if '_corrected_offset' in config_file.name:
            continue
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            if cfg.get('race_date') == race_date:
                cam = cfg.get('camera_id')
                if cam:
                    # Latest file wins (sorted order)
                    configs[cam] = config_file
        except (json.JSONDecodeError, KeyError):
            continue

    return configs


def main():
    parser = argparse.ArgumentParser(description="Export trigger zone calibration images")
    parser.add_argument("--race-date", help="Race date (YYYY-MM-DD) to export all cameras")
    parser.add_argument("--config", help="Specific config JSON file to export")
    parser.add_argument("--output", help="Output image path (default: auto-generate)")
    parser.add_argument("--output-dir", help="Output directory (for --race-date mode)")

    args = parser.parse_args()

    if args.config:
        output = generate_zone_image(args.config, args.output)
        if output:
            print(f"\nDone: {output}")
        else:
            print("\nFailed to generate image")
            sys.exit(1)

    elif args.race_date:
        configs = find_configs_for_race(args.race_date)
        if not configs:
            print(f"No configs found for race date {args.race_date}")
            sys.exit(1)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default: web/races/<race-slug>/docs/
            project_root = Path(__file__).parent.parent
            web_races = project_root / 'web' / 'races'
            race_dir = None
            for d in web_races.iterdir():
                if args.race_date in d.name:
                    race_dir = d
                    break
            if race_dir:
                output_dir = race_dir / 'docs'
            else:
                output_dir = Path('.') / 'zone_images'

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting zone images for race {args.race_date}")
        print(f"Output: {output_dir}\n")

        for cam_id, config_path in sorted(configs.items()):
            with open(config_path) as f:
                cfg = json.load(f)
            section_id = cfg.get('section_id', cam_id)
            section_num = SECTION_MAP.get(section_id, cam_id)
            output_path = output_dir / f"section_{section_num}_zones.jpg"

            print(f"Camera {cam_id} ({section_id}, Section {section_num}):")
            print(f"  Config: {config_path.name}")
            generate_zone_image(str(config_path), str(output_path))
            print()

        print("Done!")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

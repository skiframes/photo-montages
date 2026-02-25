#!/usr/bin/env python3
"""
Edge device Flask application for ski run detection and calibration.
Serves mobile-first calibration UI for coaches to configure trigger zones.
"""

import os
import sys
import json
import uuid
import signal
import socket
import logging
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Optional

import cv2
import numpy as np
import re
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# --- Timestamped logging setup ---
# Override print() so every line in /tmp/flask.log gets a timestamp
_original_print = print

def print(*args, **kwargs):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _original_print(f"[{ts}]", *args, **kwargs)

# Also configure Flask/werkzeug logging with timestamps
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

from calibration import (
    detect_gates, compute_calibration, draw_verification_grid,
    draw_gates_on_frame, GATE_SPECS,
)
from stream_manager import StreamManager
from runner import SkiFramesRunner

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

app = Flask(__name__, static_folder='static', template_folder='static')


@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses for local development (port 8888 → port 5000)."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


# Track active processing jobs (subprocess-based: video files)
# Key: job_id, Value: dict with 'process', 'status', 'output', 'config_path', 'video_path'
active_jobs = {}
jobs_lock = threading.Lock()

# Shared RTSP stream manager: one connection per camera, multiple sessions
stream_manager = StreamManager()

# Track in-process RTSP sessions managed by stream_manager
# Key: job_id, Value: dict with 'runner', 'session_id', 'camera_id', 'status', etc.
rtsp_sessions = {}
rtsp_sessions_lock = threading.Lock()

# Configuration
# Use arm64 homebrew python if available, then venv, then system python3
_arm64_python = Path('/opt/homebrew/bin/python3.11')
_venv_python = Path.home() / 'venv' / 'bin' / 'python3'
if _arm64_python.exists():
    PYTHON = str(_arm64_python)
elif _venv_python.exists():
    PYTHON = str(_venv_python)
else:
    PYTHON = 'python3'

PYTHON_AI = PYTHON  # Same arm64 python for AI tasks

CONFIG_DIR = Path(__file__).parent / 'config' / 'zones'
STITCH_CONFIG_DIR = Path(__file__).parent / 'config' / 'stitch'
CALIBRATION_FRAMES_DIR = Path(tempfile.gettempdir()) / 'skiframes_calibration'
VIDEO_DIR = Path(__file__).parent.parent  # Parent of edge/ for finding test videos
LOGOS_DIR = Path(__file__).parent.parent / 'logos'  # Logo images for overlays
CALIBRATION_CONFIG_DIR = Path(__file__).parent / 'config' / 'calibrations'
UPLOAD_DIR = Path(tempfile.gettempdir()) / 'skiframes_uploads'
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
STITCH_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Track pending gate calibrations (not yet saved to disk)
pending_calibrations = {}
pending_calibrations_lock = threading.Lock()


def _expire_session_config(config_path: str):
    """Set session_end_time to now so the session no longer shows as active in the UI."""
    try:
        cp = Path(config_path)
        if not cp.exists():
            return
        with open(cp) as f:
            config = json.load(f)
        config['session_end_time'] = datetime.now().isoformat()
        with open(cp, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[SESSION] Expired session_end_time in {cp.name}")
    except Exception as e:
        print(f"[SESSION] Failed to expire config {config_path}: {e}")

# Camera definitions with time offset percentages for race timing
# offset_pct: what percentage of run time has elapsed when skier enters this camera's view
CAMERAS = {
    'R1': {
        'name': 'R1',
        'rtsp_url': os.environ.get('R1_RTSP_URL', 'rtsp://j40:J40j40j40@192.168.0.101/h264Preview_01_main'),
        'offset_pct': 0,  # 0% - covers start gate
    },
    'R2': {
        'name': 'R2',
        'rtsp_url': os.environ.get('R2_RTSP_URL', 'rtsp://j40:J40j40j40@192.168.0.102/h264Preview_01_main'),
        'offset_pct': 10,  # 10% into the run
    },
    'Axis': {
        'name': 'Axis',
        'rtsp_url': os.environ.get('AXIS_RTSP_URL', 'rtsp://j40:j40@192.168.1.100/axis-media/media.amp'),
        'offset_pct': 20,  # 20% into the run
    },
    'R3': {
        'name': 'R3',
        'rtsp_url': os.environ.get('R3_RTSP_URL', 'rtsp://j40:J40j40j40@192.168.0.103/h265Preview_01_main'),
        'offset_pct': 50,  # 50% - covers last half of run
    },
}

# Data directory - configurable via environment variable
# Mac: /Volumes/OWC_48/data, Server: /home/pa91/data
DATA_BASE_DIR = Path(os.environ.get('SKIFRAMES_DATA_DIR', '/Volumes/OWC_48/data'))

# Device identity - used in session IDs and for coach page coordination
DEVICE_ID = os.environ.get('SKIFRAMES_DEVICE_ID', socket.gethostname().split('.')[0].lower())

# Admin API URL for device heartbeat
ADMIN_API_URL = os.environ.get('SKIFRAMES_ADMIN_API', 'https://skiframes-admin-api.avillach.workers.dev')

# Vola timing file location
VOLA_DIR = DATA_BASE_DIR / 'vola'

# Project root and web races directory (for race_manifest.json)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_RACES_DIR = PROJECT_ROOT / 'web' / 'races'

# Montage staging directory (organized by race/camera/run for website integration)
# Default: <project_root>/montages/  (override with SKIFRAMES_MONTAGES_DIR env var)
# On 4T server: SKIFRAMES_MONTAGES_DIR=/home/pa91/data/montages
MONTAGES_DIR = Path(os.environ.get('SKIFRAMES_MONTAGES_DIR', str(Path(__file__).parent.parent / 'montages')))

# Session types and groups
SESSION_TYPES = ['test', 'race', 'gate_training', 'free_skiing']
GROUPS = ['U10', 'U12', 'U14', 'Scored', 'Masters', 'Free Ski']


@app.route('/')
def index():
    """Serve the calibration UI."""
    return send_from_directory('static', 'calibration.html')


@app.route('/montages/<path:filepath>')
def serve_montage(filepath):
    """Serve montage images from the staging directory (with CORS for file:// access)."""
    resp = send_from_directory(str(MONTAGES_DIR), filepath)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/race/<path:filepath>')
def serve_race_web(filepath):
    """Serve the web gallery files (race page) for local preview."""
    race_dir = WEB_RACES_DIR / 'western-q-2026-02-22'
    resp = send_from_directory(str(race_dir), filepath)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/api/cameras')
def list_cameras():
    """List available cameras with their time offset percentages."""
    cameras = [
        {
            'id': cid,
            'name': cam['name'],
            'offset_pct': cam.get('offset_pct', 0)
        }
        for cid, cam in CAMERAS.items()
    ]
    return jsonify(cameras)


@app.route('/api/sessions')
def list_sessions():
    """List session types and groups."""
    return jsonify({
        'session_types': SESSION_TYPES,
        'groups': GROUPS
    })


@app.route('/api/vola/files')
def list_vola_files():
    """List available Vola CSV timing files (searches subdirectories)."""
    files = []
    if VOLA_DIR.exists():
        # Search recursively for csv files
        for f in VOLA_DIR.glob('**/*.csv'):
            rel_path = f.relative_to(VOLA_DIR)
            files.append({
                'name': str(rel_path),
                'path': str(f),
            })
    return jsonify(sorted(files, key=lambda x: x['name']))


@app.route('/api/vola/dates')
def list_vola_dates():
    """List available race dates from Vola subdirectories.

    Scans VOLA_DIR for subdirectories with date patterns (MM-DD-YYYY)
    and reports which runs have CSV files available.

    Returns: [{"date": "2026-02-22", "folder": "U12_U14_02-22-2026", "runs": ["run1", "run2"]}, ...]
    """
    dates = []
    if VOLA_DIR.exists():
        for d in sorted(VOLA_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            # Extract date from folder name like "U12_U14_02-22-2026"
            date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', d.name)
            if not date_match:
                continue
            iso_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
            # Check which runs have CSV files
            runs = []
            for run_num in ['run1', 'run2']:
                # Look for any CSV matching *-runN.csv (e.g., boys-run1.csv, girls-run1.csv)
                csvs = list(d.glob(f'*-{run_num}.csv'))
                if csvs:
                    runs.append(run_num)
            if runs:
                dates.append({
                    'date': iso_date,
                    'folder': d.name,
                    'path': str(d),
                    'runs': runs,
                })
    return jsonify(dates)


def find_vola_dir_for_date(race_date: str) -> 'Path | None':
    """Find the Vola subdirectory for a given ISO date (YYYY-MM-DD).

    Returns the Path to the subdirectory, or None if not found.
    """
    if not VOLA_DIR.exists():
        return None
    # Convert ISO date to MM-DD-YYYY for folder matching
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', race_date)
    if not match:
        return None
    folder_date = f"{match.group(2)}-{match.group(3)}-{match.group(1)}"
    for d in VOLA_DIR.iterdir():
        if d.is_dir() and folder_date in d.name:
            return d
    return None


@app.route('/api/race-manifest/cameras')
def get_race_manifest_cameras():
    """
    Get camera→section mapping from race_manifest.json for a given Vola file or race date.
    Query params:
        vola_file - path to Vola CSV (used to find matching race manifest)
        race_date - ISO date (YYYY-MM-DD) as alternative to vola_file

    Returns camera mapping with section info, gate coverage, and edge camera IDs.
    """
    vola_file = request.args.get('vola_file', '')
    race_date_param = request.args.get('race_date', '')

    if not vola_file and not race_date_param:
        return jsonify({'error': 'vola_file or race_date required'}), 400

    # Find race manifest by matching date from Vola path or race_date param
    manifest_path = None

    if vola_file:
        vola_path = Path(vola_file)
        vola_dir = vola_path.parent
    elif race_date_param:
        vola_dir = find_vola_dir_for_date(race_date_param)
    else:
        vola_dir = None

    if vola_dir and Path(vola_dir).is_dir():
        # Check same directory as Vola CSV
        candidate = Path(vola_dir) / 'race_manifest.json'
        if candidate.exists():
            manifest_path = candidate
        else:
            # Extract date from Vola directory name and search web/races/
            date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', Path(vola_dir).name)
            if date_match:
                race_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
                if WEB_RACES_DIR.exists():
                    for race_dir in WEB_RACES_DIR.iterdir():
                        if race_date in race_dir.name:
                            candidate = race_dir / 'race_manifest.json'
                            if candidate.exists():
                                manifest_path = candidate
                                break
    elif race_date_param:
        # Direct search in web/races/ by date
        if WEB_RACES_DIR.exists():
            for race_dir in WEB_RACES_DIR.iterdir():
                if race_date_param in race_dir.name:
                    candidate = race_dir / 'race_manifest.json'
                    if candidate.exists():
                        manifest_path = candidate
                        break

    if not manifest_path:
        return jsonify({'cameras': [], 'race_slug': None})

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Extract race slug from manifest path (e.g., "western-q-2026-02-22")
        race_slug = manifest_path.parent.name

        cameras = []
        for i, cam in enumerate(manifest.get('cameras', [])):
            cameras.append({
                'id': cam.get('id'),                    # "Cam1", "Cam2", "Cam3"
                'edge_camera': cam.get('edge_camera'),  # "R1", "R2", "R3"
                'section_num': i + 1,                   # 1, 2, 3
                'gates_covered_run1': cam.get('gates_covered_run1', []),
                'gates_covered_run2': cam.get('gates_covered_run2', []),
                'has_run1': cam.get('has_run1', False),
                'has_run2': cam.get('has_run2', False),
                'note': cam.get('note', ''),
            })

        # Derive staging path
        staging_dir = str(MONTAGES_DIR / race_slug)

        return jsonify({
            'cameras': cameras,
            'race_slug': race_slug,
            'staging_dir': staging_dir,
            'event': manifest.get('event', {}),
        })
    except Exception as e:
        print(f"Error reading race manifest: {e}")
        return jsonify({'error': str(e)}), 500


def get_gender_from_bib(bib: int) -> str:
    """Determine gender from bib number based on standard ranges."""
    # U12: Women 1-60, Men 61-99
    # U14: Women 111-170, Men 171-220
    if 1 <= bib <= 60 or 111 <= bib <= 170:
        return 'Women'
    elif 61 <= bib <= 110 or 171 <= bib <= 220:
        return 'Men'
    return ''


def parse_results_pdf(pdf_path: str) -> dict:
    """
    Parse a results PDF to extract USSA IDs, rankings, and status (DSQ/DNF/DNS).

    Results format: "1 132 E7031024 Nelson Coulee 2012 PROC 26.89 31.13 58.02"
    (rank bib ussa_id lastname firstname year club run1 run2 total)

    Also parses DSQ/DNF/DNS sections.

    Returns dict mapping bib number to:
    {
        'ussa_id': 'E7031024',
        'name': 'Coulee Nelson',
        'team': 'PROC',
        'gender': 'Women',
        'rank': 1,  # None for DSQ/DNF/DNS
        'status': 'finished',  # or 'DSQ', 'DNF', 'DNS'
        'run1_time': 26.89,  # None for DNS
        'run2_time': 31.13,  # None for DNF in run2
        'total_time': 58.02,  # None for DSQ/DNF/DNS
    }
    """
    if not HAS_PDFPLUMBER:
        return {}

    bib_to_racer = {}
    current_gender = None
    current_status = 'finished'

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                for line in text.split('\n'):
                    # Detect gender sections
                    if 'Women' in line and len(line.strip()) < 20:
                        current_gender = 'Women'
                        continue
                    if 'Men' in line and len(line.strip()) < 20:
                        current_gender = 'Men'
                        continue

                    # Detect status sections
                    if 'Did Not Start' in line:
                        current_status = 'DNS'
                        continue
                    if 'Did Not Finish' in line:
                        current_status = 'DNF'
                        continue
                    if 'Disqualified' in line:
                        current_status = 'DSQ'
                        continue

                    # Parse ranked finishers
                    # Pattern: rank bib [ussa_id] lastname firstname year club run1 run2 total [gap]
                    # Example: "1 132 E7031024 Nelson Coulee 2012 PROC 26.89 31.13 58.02"
                    ranked_match = re.match(
                        r'(\d+)\s+(\d+)\s+(?:(E\d{7})\s+)?((?:[A-Za-z\'\-]+\s+)+)(\d{4})\s+([A-Z]{2,4})\s+([\d:.]+)\s+([\d:.]+)\s+([\d:.]+)',
                        line
                    )
                    if ranked_match:
                        rank = int(ranked_match.group(1))
                        bib = int(ranked_match.group(2))
                        ussa_id = ranked_match.group(3) or ''
                        name_parts = ranked_match.group(4).strip().split()
                        year = ranked_match.group(5)
                        team = ranked_match.group(6)
                        run1 = ranked_match.group(7)
                        run2 = ranked_match.group(8)
                        total = ranked_match.group(9)

                        # Name: last word is firstname
                        if len(name_parts) >= 2:
                            firstname = name_parts[-1]
                            lastname = ' '.join(name_parts[:-1])
                        else:
                            firstname = name_parts[0] if name_parts else ''
                            lastname = ''

                        bib_to_racer[bib] = {
                            'ussa_id': ussa_id,
                            'name': f"{firstname} {lastname}".strip(),
                            'team': team,
                            'gender': current_gender or get_gender_from_bib(bib),
                            'rank': rank,
                            'status': 'finished',
                            'run1_time': parse_race_time(run1),
                            'run2_time': parse_race_time(run2),
                            'total_time': parse_race_time(total),
                        }
                        continue

                    # Parse DSQ/DNF/DNS entries (no rank, may have partial times)
                    # Pattern: bib [ussa_id] lastname firstname year club [times...]
                    # Example: "133 E7090134 Morton Julia 2012 SUN"
                    # Example: "129 E6997139 O'Brien Madison 2012 PROC 32.18"
                    unranked_match = re.match(
                        r'(\d+)\s+(?:(E\d{7})\s+)?((?:[A-Za-z\'\-]+\s+)+)(\d{4})\s+([A-Z]{2,4})(?:\s+([\d:.]+))?',
                        line
                    )
                    if unranked_match and current_status != 'finished':
                        bib = int(unranked_match.group(1))
                        ussa_id = unranked_match.group(2) or ''
                        name_parts = unranked_match.group(3).strip().split()
                        year = unranked_match.group(4)
                        team = unranked_match.group(5)
                        partial_time = unranked_match.group(6)

                        if len(name_parts) >= 2:
                            firstname = name_parts[-1]
                            lastname = ' '.join(name_parts[:-1])
                        else:
                            firstname = name_parts[0] if name_parts else ''
                            lastname = ''

                        bib_to_racer[bib] = {
                            'ussa_id': ussa_id,
                            'name': f"{firstname} {lastname}".strip(),
                            'team': team,
                            'gender': current_gender or get_gender_from_bib(bib),
                            'rank': None,
                            'status': current_status,
                            'run1_time': parse_race_time(partial_time) if partial_time else None,
                            'run2_time': None,
                            'total_time': None,
                        }

        # Fallback: try table extraction for bibs missed by text parsing
        # Some PDF rows overlap with page headers, garbling text extraction
        # but table extraction can still find the cell contents
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if not row:
                            continue
                        bib = None
                        ussa_id = ''
                        name_cell = ''
                        team = ''
                        for cell in row:
                            if cell is None:
                                continue
                            cell = str(cell).strip()
                            if not bib and re.match(r'^\d{3}$', cell):
                                candidate = int(cell)
                                if 100 <= candidate <= 220 and candidate not in bib_to_racer:
                                    bib = candidate
                            if not bib:
                                bib_in_cell = re.search(r'\b(1[1-9]\d|20\d|21\d|220)\b', cell)
                                if bib_in_cell:
                                    candidate = int(bib_in_cell.group(1))
                                    if candidate not in bib_to_racer:
                                        bib = candidate
                            ussa_match = re.search(r'(E\d{7})', cell)
                            if ussa_match:
                                ussa_id = ussa_match.group(1)
                            name_match = re.search(r'([A-Z][a-z]+(?:[\'\\-][A-Z][a-z]+)?\s+[A-Z][a-z]+)', cell)
                            if name_match and not name_cell:
                                name_cell = name_match.group(1)
                            team_match = re.search(r'\b([A-Z]{2,4})\b', cell)
                            if team_match and team_match.group(1) not in ('None', 'THE', 'AND', 'FOR'):
                                candidate_team = team_match.group(1)
                                if candidate_team in ('SUN', 'PROC', 'RMS', 'FS', 'CMS'):
                                    team = candidate_team

                        if bib and bib not in bib_to_racer and (ussa_id or name_cell):
                            name = ''
                            if name_cell:
                                parts = name_cell.split()
                                if len(parts) >= 2:
                                    name = f"{parts[-1]} {' '.join(parts[:-1])}"
                                else:
                                    name = name_cell
                            bib_to_racer[bib] = {
                                'ussa_id': ussa_id,
                                'name': name,  # Leave empty if not found; race_manifest can fill it
                                'team': team,
                                'gender': get_gender_from_bib(bib),
                                'rank': None,
                                'status': 'finished',
                                'run1_time': None,
                                'run2_time': None,
                                'total_time': None,
                            }
                            print(f"  Table fallback: Bib {bib} -> {name or '(no name)'} ({team}) ussa={ussa_id}")

    except Exception as e:
        print(f"Error parsing results PDF {pdf_path}: {e}")

    return bib_to_racer


def parse_race_time(time_str: str) -> Optional[float]:
    """Parse race time string to seconds (e.g., '26.89' -> 26.89, '1:03.02' -> 63.02)."""
    if not time_str:
        return None
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(time_str)
    except (ValueError, IndexError):
        return None


@app.route('/api/vola/results-files')
def list_results_files():
    """List available results PDF files (for USSA IDs, rankings, DSQ/DNF status)."""
    files = []
    if VOLA_DIR.exists():
        # Search recursively for PDF files containing 'results'
        for f in VOLA_DIR.glob('**/*results*.pdf'):
            rel_path = f.relative_to(VOLA_DIR)
            files.append({
                'name': str(rel_path),
                'path': str(f),
            })
    return jsonify(sorted(files, key=lambda x: x['name']))


@app.route('/api/vola/parse-results', methods=['POST'])
def parse_results():
    """
    Parse a results PDF and return bib-to-name mapping.

    Request body:
    {
        "file_path": "/path/to/results.pdf"
    }
    """
    if not HAS_PDFPLUMBER:
        return jsonify({'error': 'pdfplumber not installed'}), 500

    data = request.get_json() or {}
    file_path = data.get('file_path')

    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'Invalid file_path'}), 400

    bib_to_name = parse_results_pdf(file_path)

    return jsonify({
        'file': file_path,
        'racers': len(bib_to_name),
        'bib_to_name': bib_to_name,
    })


def parse_vola_time(time_str):
    """
    Parse Vola time string like '10h41:23.9049' or ' 9h52:28.4396' to seconds since midnight.
    Returns None for 'Did Not Start', 'Did Not Finish', 'Disqualified', etc.
    """
    if not time_str or not isinstance(time_str, str):
        return None

    # Strip leading/trailing whitespace
    time_str = time_str.strip()

    # Check for DNF/DNS/DQ
    lower = time_str.lower()
    if 'did not' in lower or 'disqualified' in lower or 'dnf' in lower or 'dns' in lower or 'dq' in lower:
        return None

    # Parse format: 10h41:23.9049 or 9h52:28.4396 (with optional leading space already stripped)
    match = re.match(r'(\d+)h(\d+):(\d+(?:\.\d+)?)', time_str)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    return None


def seconds_to_time_str(seconds):
    """Convert seconds since midnight to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_vola_csv(filepath):
    """
    Parse Vola timing CSV file. Returns dict of {bib: start_seconds}.

    CSV format (comma or tab separated):
        Num,Hour Cell.
        86,10h10:11.58575
        85,10h09:46.64594
        80,Did Not Start

    Handles:
    - Auto-detects comma vs tab separator
    - Skips header row
    - Skips DNS/DNF entries
    - Skips non-numeric bibs (forerunners like 901-A)
    """
    results = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Detect separator from second line (first data line)
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

        # Skip header
        if bib_str == 'Num' or not bib_str:
            continue

        # Skip DNS/DNF
        if 'Did Not Start' in time_str or 'Did Not Finish' in time_str:
            continue

        # Skip non-numeric bibs (forerunners like 901-A)
        if not bib_str.isdigit():
            continue

        bib = int(bib_str)

        # Parse time using existing parse_vola_time()
        start_seconds = parse_vola_time(time_str)
        if start_seconds is not None:
            results[bib] = start_seconds

    return results


def build_racers_from_start_times(start_times, camera_id, estimated_duration=40.0):
    """
    Build racer list with camera timing windows from start times dict.

    Args:
        start_times: dict of {bib: start_seconds}
        camera_id: camera ID for offset calculation
        estimated_duration: estimated run duration in seconds (CSV has no end times)

    Returns:
        list of racer dicts with timing windows
    """
    camera_offset_pct = CAMERAS.get(camera_id, {}).get('offset_pct', 0) / 100.0

    racers = []
    for bib in sorted(start_times.keys()):
        start_sec = start_times[bib]

        # CSV files only have start times, so estimate duration
        camera_start_sec = start_sec + (estimated_duration * camera_offset_pct)
        camera_duration = estimated_duration * 0.2
        camera_end_sec = camera_start_sec + camera_duration

        racer = {
            'bib': bib,
            'start_time_sec': start_sec,
            'start_time_str': seconds_to_time_str(start_sec),
            'run_duration': None,  # Not available from CSV
            'finished': True,  # Assume finished if they have a start time
            'camera_start_sec': camera_start_sec,
            'camera_start_str': seconds_to_time_str(camera_start_sec),
            'camera_end_sec': camera_end_sec,
            'camera_end_str': seconds_to_time_str(camera_end_sec),
        }
        racers.append(racer)

    # Sort by start time
    racers.sort(key=lambda r: r['start_time_sec'])
    return racers


def load_race_manifest(vola_file_path: str) -> dict:
    """
    Load race_manifest.json and return a bib_to_racer dict.

    Searches for the manifest in:
    1. Same directory as the Vola CSV file
    2. web/races/*/race_manifest.json matching the race date

    Gender-aware: When bibs overlap between categories (e.g., U12_Girls bib 4
    and U12_Boys bib 4), the gender from the Vola CSV filename (girls-run1.csv
    or boys-run1.csv) determines which athlete gets priority.

    Returns dict mapping bib number to:
        {'name': 'Firstname Lastname', 'team': 'PROC', 'gender': 'Girls',
         'category': 'U12', 'rank': 1, 'run1_time': 39.61, ...}
    """
    bib_to_racer = {}
    manifest_path = None

    vola_path = Path(vola_file_path)
    vola_dir = vola_path.parent

    # Detect gender from Vola CSV filename (e.g., "girls-run1.csv" or "boys-run2.csv")
    vola_gender = None
    fname_lower = vola_path.stem.lower()
    if 'girl' in fname_lower:
        vola_gender = 'Girls'
    elif 'boy' in fname_lower:
        vola_gender = 'Boys'

    # 1. Check same directory as Vola CSV
    candidate = vola_dir / 'race_manifest.json'
    if candidate.exists():
        manifest_path = candidate
    else:
        # 2. Extract date from Vola directory name and search web/races/
        race_date = None
        date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', vola_dir.name)
        if date_match:
            race_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"

        if race_date and WEB_RACES_DIR.exists():
            for race_dir in WEB_RACES_DIR.iterdir():
                if race_date in race_dir.name:
                    candidate = race_dir / 'race_manifest.json'
                    if candidate.exists():
                        manifest_path = candidate
                        break

    if not manifest_path:
        print(f"No race_manifest.json found for {vola_file_path}")
        return bib_to_racer

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Two-pass approach: first load matching gender, then fill gaps with others.
        # This ensures that when bibs overlap, the correct gender gets priority.
        matching_entries = {}    # bib -> racer dict (matching gender)
        other_entries = {}       # bib -> racer dict (other gender)

        for category in manifest.get('categories', []):
            cat_id = category.get('id', '')  # e.g. "U12_Girls"
            parts = cat_id.split('_')
            cat_label = parts[0] if parts else ''  # "U12"
            gender = parts[1] if len(parts) > 1 else ''  # "Girls"

            for athlete in category.get('athletes', []):
                bib = athlete.get('bib')
                if bib is None:
                    continue
                entry = {
                    'name': f"{athlete.get('first', '')} {athlete.get('last', '')}".strip(),
                    'team': athlete.get('club', ''),
                    'gender': gender,
                    'category': cat_label,
                    'rank': athlete.get('rank'),
                    'run1_time': athlete.get('run1_time'),
                    'run2_time': athlete.get('run2_time'),
                    'total': athlete.get('total'),
                    'run1_status': athlete.get('run1_status', 'finished'),
                    'run2_status': athlete.get('run2_status', 'finished'),
                }
                if vola_gender and gender == vola_gender:
                    matching_entries[bib] = entry
                else:
                    other_entries[bib] = entry

        # Start with other-gender entries, then overwrite with matching gender
        # This ensures matching gender always wins for overlapping bibs
        bib_to_racer = {**other_entries, **matching_entries}

        if vola_gender:
            print(f"Loaded {len(bib_to_racer)} athletes from {manifest_path} "
                  f"(prioritizing {vola_gender}, {len(matching_entries)} matched)")
        else:
            print(f"Loaded {len(bib_to_racer)} athletes from {manifest_path}")
    except Exception as e:
        print(f"Error loading race manifest {manifest_path}: {e}")

    return bib_to_racer


@app.route('/api/vola/parse', methods=['POST'])
def parse_vola_file():
    """
    Parse a Vola CSV timing file and return timing data.

    Request body:
    {
        "file_path": "/path/to/girls-run1.csv",
        "camera_id": "R1",  // Camera to use for offset calculation
    }

    Returns list of racers with their timing windows for the specified camera.
    """
    data = request.get_json() or {}
    file_path = data.get('file_path')
    camera_id = data.get('camera_id', 'R1')

    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'Invalid file_path'}), 400

    try:
        start_times = parse_vola_csv(file_path)
        camera_offset_pct = CAMERAS.get(camera_id, {}).get('offset_pct', 0) / 100.0
        racers = build_racers_from_start_times(start_times, camera_id)

        return jsonify({
            'file': file_path,
            'camera_id': camera_id,
            'camera_offset_pct': camera_offset_pct * 100,
            'racers': racers,
            'total_racers': len(racers),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/vola/generate-filename', methods=['POST'])
def generate_vola_filename():
    """
    Generate a filename for a photo montage based on Vola data.

    Format: Name_bib#_RunX_CamXX_Ragged_YYYY-Mon-DD.jpg
    Example: JohnDoe_106_R1_CamR1_Ragged_2026-Feb-01.jpg

    Request body:
    {
        "bib": 106,
        "name": "John Doe",  // Optional - will use bib if not provided
        "race": "U12 run 1",
        "camera_id": "R1",
        "date": "2026-02-01"  // Optional - defaults to today
    }
    """
    data = request.get_json() or {}
    bib = data.get('bib')
    name = data.get('name', '').strip()
    race = data.get('race', '').lower()
    camera_id = data.get('camera_id', 'R1')
    date_str = data.get('date')

    if not bib:
        return jsonify({'error': 'bib is required'}), 400

    # Parse race to get run number
    run_num = '1'
    if 'run 2' in race:
        run_num = '2'

    # Format name (remove spaces, use bib if no name)
    if name:
        name_part = name.replace(' ', '')
    else:
        name_part = f"Bib{bib}"

    # Format date
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str)
        except:
            dt = datetime.now()
    else:
        dt = datetime.now()
    date_part = dt.strftime('%Y%b%d')  # e.g., 2026Feb01

    filename = f"{name_part}_{bib}_R{run_num}_Cam{camera_id}_Ragged_{date_part}.jpg"

    return jsonify({
        'filename': filename,
        'bib': bib,
        'name': name_part,
        'run': run_num,
        'camera': camera_id,
        'date': date_part,
    })


@app.route('/api/vola/racers-for-video', methods=['POST'])
def get_racers_for_video():
    """
    Get list of racers whose timing windows fall within a video's timespan.

    This helps identify which racers should be in the montages generated
    from a specific video file.

    Request body:
    {
        "file_path": "/path/to/girls-run1.csv",
        "camera_id": "R1",
        "video_start_time": "10:41:00",  // HH:MM:SS format (Boston time)
        "video_duration_sec": 120  // Video length in seconds
    }

    Returns list of racers with their camera timing windows.
    """
    data = request.get_json() or {}
    file_path = data.get('file_path')
    camera_id = data.get('camera_id', 'R1')
    video_start = data.get('video_start_time', '')
    video_duration = data.get('video_duration_sec', 0)

    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'Invalid file_path'}), 400

    # Parse video start time
    try:
        parts = video_start.split(':')
        video_start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except:
        return jsonify({'error': 'Invalid video_start_time format (use HH:MM:SS)'}), 400

    video_end_sec = video_start_sec + video_duration

    try:
        start_times = parse_vola_csv(file_path)
        all_racers = build_racers_from_start_times(start_times, camera_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Filter to racers whose camera window overlaps with video timespan
    matching_racers = []
    for racer in all_racers:
        cam_start = racer['camera_start_sec']
        cam_end = racer['camera_end_sec']

        # Check if camera window overlaps with video
        if cam_start <= video_end_sec and cam_end >= video_start_sec:
            # Calculate video offset (when racer appears in video)
            video_offset_sec = max(0, cam_start - video_start_sec)
            racer['video_offset_sec'] = video_offset_sec
            racer['video_offset_str'] = f"{int(video_offset_sec // 60)}:{video_offset_sec % 60:05.2f}"
            matching_racers.append(racer)

    return jsonify({
        'file': file_path,
        'camera_id': camera_id,
        'video_start': video_start,
        'video_duration_sec': video_duration,
        'matching_racers': matching_racers,
        'total_matching': len(matching_racers),
    })


@app.route('/api/frame/grab', methods=['POST'])
def grab_frame():
    """
    Grab a frame from RTSP stream or video file.
    Returns the frame as a JPEG with a unique ID for reference.
    """
    data = request.get_json() or {}
    source_type = data.get('source_type', 'rtsp')  # 'rtsp' or 'video'
    camera_id = data.get('camera_id')
    video_path = data.get('video_path')
    seek_seconds = data.get('seek_seconds', 0)

    if source_type == 'rtsp':
        if not camera_id or camera_id not in CAMERAS:
            return jsonify({'error': 'Invalid camera_id'}), 400
        source = CAMERAS[camera_id]['rtsp_url']
    else:
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Invalid video_path'}), 400
        source = video_path

    try:
        if source_type == 'rtsp':
            # Use same ffmpeg options as detection engine:
            # - TCP transport for reliable delivery
            # - genpts to fix non-monotonic timestamps (Reolink)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;+genpts|stimeout;5000000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return jsonify({'error': f'Cannot connect to camera stream'}), 500

        if source_type == 'video' and seek_seconds > 0:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_seconds * fps))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'error': 'Connected but failed to read frame'}), 500

        # Generate unique ID for this frame
        frame_id = str(uuid.uuid4())[:8]
        frame_path = CALIBRATION_FRAMES_DIR / f'{frame_id}.jpg'

        # Save frame
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        height, width = frame.shape[:2]

        return jsonify({
            'frame_id': frame_id,
            'width': width,
            'height': height,
            'url': f'/api/frame/{frame_id}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/frame/<frame_id>')
def get_frame(frame_id):
    """Serve a grabbed frame."""
    frame_path = CALIBRATION_FRAMES_DIR / f'{frame_id}.jpg'
    if not frame_path.exists():
        return jsonify({'error': 'Frame not found'}), 404
    return send_file(frame_path, mimetype='image/jpeg')


@app.route('/api/video/info', methods=['POST'])
def video_info():
    """Get video file info for scrubbing."""
    data = request.get_json() or {}
    video_path = data.get('video_path')

    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video_path'}), 400

    try:
        cap = cv2.VideoCapture(video_path)
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info['duration'] = info['frames'] / info['fps'] if info['fps'] > 0 else 0
        cap.release()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/save', methods=['POST'])
def save_config():
    """Save session configuration with trigger zones."""
    data = request.get_json()

    # start_zone is always required
    # end_zone is required unless run_duration_seconds is set (duration mode)
    required = ['camera_id', 'session_type', 'group', 'start_zone']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    # Validate: need either end_zone or run_duration_seconds
    if not data.get('end_zone') and not data.get('run_duration_seconds'):
        return jsonify({'error': 'Either end_zone or run_duration_seconds is required'}), 400

    # Generate session ID: date_time_group_type_camera_device
    # Uses seconds precision to allow multiple simultaneous sessions
    now = datetime.now()
    camera_id = data['camera_id']
    session_id = f"{now.strftime('%Y-%m-%d_%H%M%S')}_{data['group'].lower()}_{data['session_type']}_{camera_id}_{DEVICE_ID}"

    # Default session end time: 90 minutes from now
    session_end = data.get('session_end_time')
    if not session_end:
        session_end = (now + timedelta(minutes=90)).isoformat()

    config = {
        'session_id': session_id,
        'device_id': DEVICE_ID,
        'session_type': data['session_type'],
        'group': data['group'],
        'camera_id': data['camera_id'],
        'camera_url': CAMERAS.get(data['camera_id'], {}).get('rtsp_url', ''),
        'calibration_frame': data.get('calibration_frame_id', ''),
        'start_zone': data['start_zone'],  # {x, y, w, h}
        'end_zone': data.get('end_zone'),  # {x, y, w, h} - optional if using duration mode
        'crop_zone': data.get('crop_zone'),  # {x, y, w, h} - optional, overrides zone-based crop
        'run_duration_seconds': data.get('run_duration_seconds'),  # if set, use duration mode instead of END zone
        'session_start_time': now.isoformat(),
        'session_end_time': session_end,
        'pre_buffer_seconds': data.get('pre_buffer_seconds', 2),
        'post_buffer_seconds': data.get('post_buffer_seconds', 2),
        'detection_threshold': data.get('detection_threshold', 25),
        'min_pixel_change_pct': data.get('min_pixel_change_pct', 5.0),
        'min_brightness': data.get('min_brightness', 100),  # Shadow filter: min brightness of motion pixels
        'discipline': data.get('discipline', 'freeski'),  # freeski, sl_youth, sl_adult, gs_panel, sg_panel
        'montage_fps_list': data.get('montage_fps_list', [4.0]),
        'start_offset_sec': data.get('start_offset_sec', 0),  # Fixed delay in seconds after trigger
        # Vola racer data for naming montages
        'vola_racers': data.get('vola_racers', []),
        'vola_videos': data.get('vola_videos', []),  # Videos to process
        'vola_race': data.get('vola_race', ''),
        'vola_camera': data.get('vola_camera', ''),
        'vola_view': data.get('vola_view', '1'),  # View number 1-5 for filename
        'race_date': data.get('race_date', now.strftime('%Y-%m-%d')),
        'calibration_id': data.get('calibration_id'),  # Links to gate calibration if present
        # Staging mode fields for batch processing recorded videos
        'num_athletes': data.get('num_athletes', 0),  # 0 = all
        'section_id': data.get('section_id', ''),      # e.g., "Cam1"
        'run_number': data.get('run_number', ''),       # e.g., "run1"
        'race_slug': data.get('race_slug', ''),         # e.g., "western-q-2026-02-22"
        'staging_dir': data.get('staging_dir', ''),     # e.g., "/path/to/montages/western-q-2026-02-22"
    }

    # Save config
    config_path = CONFIG_DIR / f'{session_id}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'config_path': str(config_path)
    })


def parse_iso_datetime(s):
    """Parse ISO datetime string, handling 'Z' suffix for UTC."""
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    return datetime.fromisoformat(s)


@app.route('/api/config/active')
def get_active_config():
    """Get the currently active session config (most recent)."""
    configs = sorted(CONFIG_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        return jsonify({'active': False})

    with open(configs[0]) as f:
        config = json.load(f)

    # Check if session is still active
    try:
        end_time = parse_iso_datetime(config['session_end_time'])
        # Compare as naive datetime (ignore timezone for simplicity)
        if end_time.replace(tzinfo=None) < datetime.now():
            return jsonify({'active': False, 'last_session': config})
    except (ValueError, KeyError):
        return jsonify({'active': False, 'last_session': config})

    return jsonify({'active': True, 'config': config})


@app.route('/api/config/all_active')
def get_all_active_configs():
    """Get ALL sessions that are still active (end_time in future) or have running processes."""
    configs = sorted(CONFIG_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    now = datetime.now()
    sessions = []

    # Collect sessions with running jobs (even if config says expired)
    running_session_ids = {}  # session_id -> job_id mapping

    # Check subprocess jobs
    with jobs_lock:
        for job_id, job in active_jobs.items():
            if job['status'] == 'running':
                # Extract session_id from config_path
                cp = job.get('config_path', '')
                if cp:
                    sid = Path(cp).stem
                    running_session_ids[sid] = job_id

    # Check in-process RTSP sessions (stream manager)
    with rtsp_sessions_lock:
        for job_id, session in rtsp_sessions.items():
            if session['status'] == 'running':
                running_session_ids[session['session_id']] = job_id

    for config_path in configs:
        try:
            with open(config_path) as f:
                config = json.load(f)

            session_id = config_path.stem
            is_running = session_id in running_session_ids

            # Check if session end_time is in the future
            # Only use session_end_time for RTSP sessions — video file sessions
            # are active only while their process is running
            is_active = False
            source_type = config.get('source_type', '')
            has_videos = bool(config.get('vola_videos') or config.get('video_files'))
            if not has_videos:
                # RTSP session: active until session_end_time
                try:
                    end_time = parse_iso_datetime(config['session_end_time'])
                    if end_time.replace(tzinfo=None) > now:
                        is_active = True
                except (ValueError, KeyError):
                    pass

            if is_active or is_running:
                session_info = {
                    'session_id': session_id,
                    'config': config,
                    'is_running': is_running,
                    'is_active': is_active,
                }
                # Include job_id so frontend can poll metrics/status
                if session_id in running_session_ids:
                    session_info['job_id'] = running_session_ids[session_id]
                sessions.append(session_info)
        except Exception:
            continue

    return jsonify({'sessions': sessions})


@app.route('/api/detection/metrics/<job_id>')
def get_detection_metrics(job_id):
    """Get live detection metrics for a running job (for calibration chart)."""
    metrics_path = None

    # Check in-process RTSP sessions first
    with rtsp_sessions_lock:
        rtsp_session = rtsp_sessions.get(job_id)
    if rtsp_session:
        runner = rtsp_session.get('runner')
        if runner and hasattr(runner, 'engine') and runner.engine.metrics_path:
            metrics_path = runner.engine.metrics_path

    # Fall back to subprocess jobs
    if not metrics_path:
        with jobs_lock:
            job = active_jobs.get(job_id)
            if not job:
                return jsonify({'error': 'Job not found'}), 404

            config_path = job.get('config_path', '')

        if not config_path:
            return jsonify({'error': 'No config path for job'}), 404

        # Derive session output dir from config
        try:
            with open(config_path) as f:
                config = json.load(f)
            session_id = config.get('session_id', Path(config_path).stem)
        except Exception:
            session_id = Path(config_path).stem

        output_dir = str(Path(__file__).resolve().parent.parent / 'output')
        metrics_path = os.path.join(output_dir, session_id, 'detection_metrics.json')

    if not os.path.exists(metrics_path):
        return jsonify({'entries': [], 'message': 'No metrics yet'})

    try:
        with open(metrics_path) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/<session_id>')
def get_config(session_id):
    """Get a specific session config."""
    config_path = CONFIG_DIR / f'{session_id}.json'
    if not config_path.exists():
        return jsonify({'error': 'Config not found'}), 404

    with open(config_path) as f:
        config = json.load(f)
    return jsonify(config)


@app.route('/api/config/stop/<session_id>', methods=['POST'])
def stop_session(session_id):
    """Stop a session: set end time to now AND kill any running detection process."""
    config_path = CONFIG_DIR / f'{session_id}.json'
    if not config_path.exists():
        return jsonify({'error': 'Config not found'}), 404

    with open(config_path) as f:
        config = json.load(f)

    # Set end time to now to mark session as ended
    config['session_end_time'] = datetime.now().isoformat()

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Also kill any running process/session for this session
    process_killed = False

    # Check in-process RTSP sessions first
    with rtsp_sessions_lock:
        for job_id, session in rtsp_sessions.items():
            if session['session_id'] == session_id and session['status'] == 'running':
                stream_manager.stop_session(session_id)
                session['status'] = 'stopped'
                process_killed = True
                print(f"[STOP] Stopped in-process RTSP session {session_id} (job {job_id})")
                break

    # Check subprocess jobs
    if not process_killed:
        with jobs_lock:
            for job_id, job in active_jobs.items():
                if job.get('config_path', '').endswith(f'{session_id}.json') and job['status'] == 'running':
                    process = job['process']
                    try:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except:
                            process.kill()
                        job['status'] = 'stopped'
                        process_killed = True
                        print(f"[STOP] Killed process for session {session_id} (job {job_id})")
                    except Exception as e:
                        print(f"[STOP] Failed to kill process for {session_id}: {e}")
                    break

    return jsonify({'success': True, 'session_id': session_id, 'process_killed': process_killed})


@app.route('/api/config/update_live/<session_id>', methods=['POST'])
def update_live_settings(session_id):
    """
    Update detection settings on a running session.
    The detection engine hot-reloads from the config file every ~2 seconds,
    so changes take effect within a few seconds without restarting.
    """
    config_path = CONFIG_DIR / f'{session_id}.json'
    if not config_path.exists():
        return jsonify({'error': 'Config not found'}), 404

    data = request.get_json()

    with open(config_path) as f:
        config = json.load(f)

    # Only allow updating tunable parameters (not zones or session structure)
    updatable = ['detection_threshold', 'min_pixel_change_pct', 'min_brightness',
                 'start_offset_sec', 'session_end_time']
    changed = []
    for key in updatable:
        if key in data:
            old_val = config.get(key)
            config[key] = data[key]
            if old_val != data[key]:
                changed.append(f"{key}: {old_val} -> {data[key]}")

    if changed:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[LIVE] Updated settings for {session_id}: {', '.join(changed)}")

    return jsonify({
        'success': True,
        'session_id': session_id,
        'changed': changed,
        'detection_threshold': config.get('detection_threshold'),
        'min_pixel_change_pct': config.get('min_pixel_change_pct'),
        'min_brightness': config.get('min_brightness'),
    })


@app.route('/api/videos')
def list_videos():
    """List available video files for processing."""
    videos = []

    # Directories to scan for videos
    video_dirs = [
        VIDEO_DIR,  # Parent of edge/ for finding test videos
        VIDEO_DIR / 'videos-to-process',  # Dedicated folder for videos to process
    ]

    for video_dir in video_dirs:
        if not video_dir.exists():
            continue
        for ext in ['*.mp4', '*.mkv', '*.avi']:
            for video_path in video_dir.glob(ext):
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        cap.release()
                        continue
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    videos.append({
                        'name': video_path.name,
                        'path': str(video_path),
                        'duration': round(duration, 1),
                        'resolution': f"{width}x{height}",
                        'size_mb': round(video_path.stat().st_size / 1024 / 1024, 1),
                    })
                except:
                    pass

    # Remove duplicates (by path) and sort by name
    seen_paths = set()
    unique_videos = []
    for v in videos:
        if v['path'] not in seen_paths:
            seen_paths.add(v['path'])
            unique_videos.append(v)

    return jsonify(sorted(unique_videos, key=lambda v: v['name']))


# External recordings drive path (uses DATA_BASE_DIR from environment)
RECORDINGS_DIR = DATA_BASE_DIR / 'recordings'

# J40 SD card data directory (flat structure: R1/R1_YYYYMMDD_HHMMSS.mp4)
J40_SD_DATA_DIR = Path(os.environ.get('J40_SD_DATA_DIR', '/Users/paul2/j40/data/sd_data'))


def parse_reolink_video_times(filename: str) -> tuple:
    """
    Parse start and end times from Reolink video filename.
    Format: RecM03_YYYYMMDD_HHMMSS_HHMMSS_XXXXX_XXXXX.mp4

    Returns (date_str, start_time_sec, end_time_sec) or (None, None, None) if parse fails.
    """
    # Example: RecM03_20260201_081538_082039_533C810_BEF5D90.mp4
    match = re.match(r'RecM03_(\d{8})_(\d{6})_(\d{6})_', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        start_str = match.group(2)  # HHMMSS
        end_str = match.group(3)  # HHMMSS

        # Convert to seconds since midnight
        start_sec = int(start_str[0:2]) * 3600 + int(start_str[2:4]) * 60 + int(start_str[4:6])
        end_sec = int(end_str[0:2]) * 3600 + int(end_str[2:4]) * 60 + int(end_str[4:6])

        # Format date for display
        formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"

        return formatted_date, start_sec, end_sec

    return None, None, None


def parse_j40_video_times(filename: str) -> tuple:
    """
    Parse start time from J40 SD card video filename.
    Format: R1_YYYYMMDD_HHMMSS.mp4 (each file is ~5 minutes / 300 seconds)

    Returns (date_str, start_time_sec, end_time_sec) or (None, None, None) if parse fails.
    """
    match = re.match(r'(?:R[123]|Axis)_(\d{8})_(\d{6})\.mp4$', filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        start_str = match.group(2)  # HHMMSS

        start_sec = int(start_str[0:2]) * 3600 + int(start_str[2:4]) * 60 + int(start_str[4:6])
        end_sec = start_sec + 300  # Each file is ~5 minutes

        formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return formatted_date, start_sec, end_sec

    return None, None, None


@app.route('/api/recordings/dates')
def list_recording_dates():
    """List available recording dates across all cameras."""
    dates = set()

    if not RECORDINGS_DIR.exists():
        return jsonify({'error': 'Recordings drive not mounted', 'dates': []})

    # Check each camera folder (Reolink cameras: sd_R1, sd_R2, sd_R3)
    for camera_folder in ['sd_R1', 'sd_R2', 'sd_R3']:
        mp4_path = RECORDINGS_DIR / camera_folder / 'sdcard' / 'Mp4Record'
        if mp4_path.exists():
            for date_folder in mp4_path.iterdir():
                if date_folder.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', date_folder.name):
                    dates.add(date_folder.name)

    # Check exported SD card copies (flat structure: export_sd_R1/R1_YYYYMMDD_HHMMSS.mp4)
    for camera_folder in ['export_sd_R1', 'export_sd_R2', 'export_sd_R3']:
        cam_path = RECORDINGS_DIR / camera_folder
        if cam_path.exists():
            for video_file in cam_path.glob('*.mp4'):
                parsed_date, _, _ = parse_j40_video_times(video_file.name)
                if parsed_date:
                    dates.add(parsed_date)

    # Also check J40 SD card data (flat structure: R1/R1_YYYYMMDD_HHMMSS.mp4)
    if J40_SD_DATA_DIR.exists():
        for camera_folder in ['R1', 'R2', 'R3', 'Axis']:
            cam_path = J40_SD_DATA_DIR / camera_folder
            if cam_path.exists():
                for video_file in cam_path.glob('*.mp4'):
                    parsed_date, _, _ = parse_j40_video_times(video_file.name)
                    if parsed_date:
                        dates.add(parsed_date)

    return jsonify({'dates': sorted(dates, reverse=True)})


@app.route('/api/recordings/videos', methods=['POST'])
def list_recording_videos():
    """
    List videos from recordings drive for a specific camera and date.
    Optionally filter by time range based on Vola race timing.

    Request body:
    {
        "camera_id": "R1",
        "date": "2026-02-01",
        "start_time_sec": 36900,  // Optional: filter videos covering this time range
        "end_time_sec": 39600,    // Optional: filter videos covering this time range
    }
    """
    data = request.get_json() or {}
    camera_id = data.get('camera_id', 'R1')
    date = data.get('date')
    filter_start = data.get('start_time_sec')
    filter_end = data.get('end_time_sec')

    if not date:
        return jsonify({'error': 'date is required'}), 400

    # Map camera_id to folder name for Reolink recordings
    camera_folders = {
        'R1': 'sd_R1',
        'R2': 'sd_R2',
        'R3': 'sd_R3',
        'Axis': 'sd_Axis',
    }

    folder_name = camera_folders.get(camera_id)
    if not folder_name:
        return jsonify({'error': f'Unknown camera_id: {camera_id}'}), 400

    videos = []

    # Source 1: Reolink recordings drive (sd_R1/sdcard/Mp4Record/<date>/)
    videos_path = RECORDINGS_DIR / folder_name / 'sdcard' / 'Mp4Record' / date
    if videos_path.exists():
        for video_file in sorted(videos_path.glob('*.mp4')):
            parsed_date, start_sec, end_sec = parse_reolink_video_times(video_file.name)
            if start_sec is None:
                continue
            if filter_start is not None and filter_end is not None:
                if end_sec < filter_start or start_sec > filter_end:
                    continue
            start_time_str = f"{start_sec // 3600:02d}:{(start_sec % 3600) // 60:02d}:{start_sec % 60:02d}"
            end_time_str = f"{end_sec // 3600:02d}:{(end_sec % 3600) // 60:02d}:{end_sec % 60:02d}"
            videos.append({
                'name': video_file.name,
                'path': str(video_file),
                'start_time_sec': start_sec,
                'end_time_sec': end_sec,
                'start_time_str': start_time_str,
                'end_time_str': end_time_str,
                'duration_sec': end_sec - start_sec,
            })

    # Source 2: Exported SD card copies (export_sd_R1/R1_YYYYMMDD_HHMMSS.mp4)
    export_cam_path = RECORDINGS_DIR / f'export_{folder_name}'
    if export_cam_path.exists():
        for video_file in sorted(export_cam_path.glob('*.mp4')):
            parsed_date, start_sec, end_sec = parse_j40_video_times(video_file.name)
            if start_sec is None or parsed_date != date:
                continue
            if filter_start is not None and filter_end is not None:
                if end_sec < filter_start or start_sec > filter_end:
                    continue
            start_time_str = f"{start_sec // 3600:02d}:{(start_sec % 3600) // 60:02d}:{start_sec % 60:02d}"
            end_time_str = f"{end_sec // 3600:02d}:{(end_sec % 3600) // 60:02d}:{end_sec % 60:02d}"
            videos.append({
                'name': video_file.name,
                'path': str(video_file),
                'start_time_sec': start_sec,
                'end_time_sec': end_sec,
                'start_time_str': start_time_str,
                'end_time_str': end_time_str,
                'duration_sec': end_sec - start_sec,
            })

    # Source 3: J40 SD card data (R1/R1_YYYYMMDD_HHMMSS.mp4)
    j40_cam_path = J40_SD_DATA_DIR / camera_id
    if j40_cam_path.exists():
        for video_file in sorted(j40_cam_path.glob('*.mp4')):
            parsed_date, start_sec, end_sec = parse_j40_video_times(video_file.name)
            if start_sec is None or parsed_date != date:
                continue
            if filter_start is not None and filter_end is not None:
                if end_sec < filter_start or start_sec > filter_end:
                    continue
            start_time_str = f"{start_sec // 3600:02d}:{(start_sec % 3600) // 60:02d}:{start_sec % 60:02d}"
            end_time_str = f"{end_sec // 3600:02d}:{(end_sec % 3600) // 60:02d}:{end_sec % 60:02d}"
            videos.append({
                'name': video_file.name,
                'path': str(video_file),
                'start_time_sec': start_sec,
                'end_time_sec': end_sec,
                'start_time_str': start_time_str,
                'end_time_str': end_time_str,
                'duration_sec': end_sec - start_sec,
            })

    if not videos:
        return jsonify({'error': f'No recordings found for {camera_id} on {date}', 'videos': []})

    # Sort all videos by start time
    videos.sort(key=lambda v: v['start_time_sec'])

    return jsonify({
        'camera_id': camera_id,
        'date': date,
        'videos': videos,
        'total': len(videos),
    })


@app.route('/api/recordings/videos-for-race', methods=['POST'])
def get_videos_for_race():
    """
    Get videos that cover a specific race's timing window.

    Request body (new simplified mode — race_date + run_number):
    {
        "race_date": "2026-02-22",
        "run_number": "run1",
        "camera_id": "R1"
    }

    Request body (legacy mode — single vola_file):
    {
        "vola_file": "/path/to/girls-run1.csv",
        "camera_id": "R1",
        "num_athletes": 5
    }

    Returns list of videos needed and the racers they cover (with names/teams from race_manifest.json).
    """
    data = request.get_json() or {}
    vola_file = data.get('vola_file')
    camera_id = data.get('camera_id', 'R1')
    num_athletes = data.get('num_athletes', 0)  # 0 = all (legacy)
    race_date_override = data.get('race_date')  # YYYY-MM-DD
    run_number = data.get('run_number')  # "run1" or "run2"

    try:
        # === New simplified mode: race_date + run_number (merges boys + girls) ===
        if race_date_override and run_number and not vola_file:
            vola_dir = find_vola_dir_for_date(race_date_override)
            if not vola_dir:
                return jsonify({'error': f'No Vola data found for date {race_date_override}'}), 400

            # Find boys and girls CSVs for this run
            boys_csv = list(vola_dir.glob(f'boys-{run_number}.csv'))
            girls_csv = list(vola_dir.glob(f'girls-{run_number}.csv'))

            if not boys_csv and not girls_csv:
                return jsonify({'error': f'No CSV files found for {run_number} in {vola_dir.name}'}), 400

            # Parse boys and girls separately — bibs overlap between genders!
            # (e.g., boys bib 10 and girls bib 10 are different athletes)
            racers = []

            if boys_csv:
                boys_times = parse_vola_csv(str(boys_csv[0]))
                bib_to_racer_boys = load_race_manifest(str(boys_csv[0]))
                boys_racers = build_racers_from_start_times(boys_times, camera_id)
                for racer in boys_racers:
                    racer_info = bib_to_racer_boys.get(racer['bib'], {})
                    racer['name'] = racer_info.get('name', '')
                    racer['team'] = racer_info.get('team', '')
                    racer['gender'] = 'Men'
                racers.extend(boys_racers)

            if girls_csv:
                girls_times = parse_vola_csv(str(girls_csv[0]))
                bib_to_racer_girls = load_race_manifest(str(girls_csv[0]))
                girls_racers = build_racers_from_start_times(girls_times, camera_id)
                for racer in girls_racers:
                    racer_info = bib_to_racer_girls.get(racer['bib'], {})
                    racer['name'] = racer_info.get('name', '')
                    racer['team'] = racer_info.get('team', '')
                    racer['gender'] = 'Women'
                racers.extend(girls_racers)

            # Sort all racers by start time (interleaved boys + girls)
            racers.sort(key=lambda r: r['start_time_sec'])

            race_date = race_date_override

        # === Legacy mode: single vola_file ===
        else:
            if not vola_file or not Path(vola_file).exists():
                return jsonify({'error': 'Invalid vola_file'}), 400

            bib_to_racer = load_race_manifest(vola_file)
            start_times = parse_vola_csv(vola_file)
            racers = build_racers_from_start_times(start_times, camera_id)

            # Enrich racers with name/team/gender from start list
            for racer in racers:
                racer_info = bib_to_racer.get(racer['bib'], {})
                racer['name'] = racer_info.get('name', '')
                racer['team'] = racer_info.get('team', '')
                racer['gender'] = racer_info.get('gender', '')

            # Limit to num_athletes if specified (legacy)
            if num_athletes and num_athletes > 0:
                racers = racers[:num_athletes]

            # Determine race date
            race_date = None
            if race_date_override and re.match(r'\d{4}-\d{2}-\d{2}$', race_date_override):
                race_date = race_date_override
            else:
                vola_path = Path(vola_file)
                for name_to_check in [vola_path.parent.name, vola_path.stem]:
                    date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', name_to_check)
                    if date_match:
                        race_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
                        break

            if not race_date:
                return jsonify({'error': 'Could not determine race date. Please select a date in the Race Date picker.'}), 400

        if not racers:
            return jsonify({'error': 'No racers found', 'videos': [], 'racers': []})

        # Find the time range we need videos for
        first_camera_time = min(r['camera_start_sec'] for r in racers)
        last_camera_time = max(r['camera_end_sec'] for r in racers)

        # Add some buffer (30 seconds before and after)
        filter_start = first_camera_time - 30
        filter_end = last_camera_time + 30

        # Build path to videos - try multiple directory structures
        camera_folders = {'R1': 'sd_R1', 'R2': 'sd_R2', 'R3': 'sd_R3', 'Axis': 'sd_Axis'}
        folder_name = camera_folders.get(camera_id, 'sd_R1')

        # Try multiple directory structures: exported copies, Reolink SD card, flat J40
        videos_paths = [
            RECORDINGS_DIR / f'export_{folder_name}',  # Exported SD card copies
            RECORDINGS_DIR / folder_name / 'sdcard' / 'Mp4Record' / race_date,  # Reolink SD card
            RECORDINGS_DIR / folder_name,  # Flat directory (J40 recordings)
        ]

        videos = []
        race_date_compact = race_date.replace('-', '')  # YYYYMMDD for filename matching

        for videos_path in videos_paths:
            if not videos_path.exists():
                continue
            for video_file in sorted(videos_path.glob('*.mp4')):
                # Try Reolink RecM03 format
                parsed_date, start_sec, end_sec = parse_reolink_video_times(video_file.name)
                if start_sec is None:
                    # Try J40 format (R1_YYYYMMDD_HHMMSS.mp4)
                    parsed_date, start_sec, end_sec = parse_j40_video_times(video_file.name)
                if start_sec is None:
                    continue

                # Filter by date
                if parsed_date and parsed_date != race_date:
                    continue

                # Check if video overlaps with the time range we need
                if end_sec < filter_start or start_sec > filter_end:
                    continue

                start_time_str = f"{start_sec // 3600:02d}:{(start_sec % 3600) // 60:02d}:{start_sec % 60:02d}"
                end_time_str = f"{end_sec // 3600:02d}:{(end_sec % 3600) // 60:02d}:{end_sec % 60:02d}"

                videos.append({
                    'name': video_file.name,
                    'path': str(video_file),
                    'start_time_sec': start_sec,
                    'end_time_sec': end_sec,
                    'start_time_str': start_time_str,
                    'end_time_str': end_time_str,
                })

            if videos:  # Stop after first directory that has matching videos
                break

        return jsonify({
            'file': vola_file,
            'run_number': run_number,
            'camera_id': camera_id,
            'date': race_date,
            'num_athletes': len(racers),
            'time_range': {
                'start_sec': filter_start,
                'end_sec': filter_end,
                'start_str': f"{int(filter_start) // 3600:02d}:{(int(filter_start) % 3600) // 60:02d}:{int(filter_start) % 60:02d}",
                'end_str': f"{int(filter_end) // 3600:02d}:{(int(filter_end) % 3600) // 60:02d}:{int(filter_end) % 60:02d}",
            },
            'videos': videos,
            'racers': racers,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/video', methods=['POST'])
def process_video():
    """Start processing video file(s) - returns job_id for tracking."""
    data = request.json
    config_path = data.get('config_path')
    video_path = data.get('video_path')  # Optional - can use videos from config instead
    use_config_videos = data.get('use_config_videos', False)

    print(f"[DEBUG] process_video called: config_path={config_path}, video_path={video_path}, use_config_videos={use_config_videos}")

    if not config_path:
        return jsonify({'error': 'Missing config_path'}), 400

    if not Path(config_path).exists():
        return jsonify({'error': f'Config not found: {config_path}'}), 404

    # If not using config videos, need a video_path
    if not use_config_videos and not video_path:
        return jsonify({'error': 'Missing video_path or use_config_videos'}), 400

    if video_path and not Path(video_path).exists():
        return jsonify({'error': f'Video not found: {video_path}'}), 404

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Run the runner.py script as a background process
    runner_path = Path(__file__).resolve().parent / 'runner.py'

    try:
        # Log job start with config details
        try:
            with open(config_path) as _cf:
                _cfg = json.load(_cf)
            _log_msg = (
                f"\n{'='*70}\n"
                f"[JOB START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  Job ID:        {job_id}\n"
                f"  Config:        {Path(config_path).name}\n"
                f"  Camera:        {_cfg.get('camera_id', '?')}\n"
                f"  Group:         {_cfg.get('group', '?')} ({_cfg.get('session_type', '?')})\n"
                f"  Section:       {_cfg.get('section_id', 'N/A')} / {_cfg.get('run_number', 'N/A')}\n"
                f"  Staging dir:   {_cfg.get('staging_dir', 'N/A')}\n"
                f"  Num athletes:  {_cfg.get('num_athletes', 0)} (of {len(_cfg.get('vola_racers', []))} total)\n"
                f"  End zone:      {'SET' if _cfg.get('end_zone') else 'NULL (duration mode: ' + str(_cfg.get('run_duration_seconds')) + 's)'}\n"
                f"  Detection:     threshold={_cfg.get('detection_threshold')}, min_pix_pct={_cfg.get('min_pixel_change_pct')}, min_bright={_cfg.get('min_brightness')}\n"
                f"  Montage FPS:   {_cfg.get('montage_fps_list')}\n"
                f"  Videos:        {len(_cfg.get('vola_videos', []))}\n"
                f"{'='*70}"
            )
            print(_log_msg)
        except Exception as _e:
            print(f"[JOB START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} job_id={job_id} config={config_path} (could not parse config: {_e})")

        if use_config_videos:
            # Use --use-config-videos flag to process videos from config's vola_videos list
            cmd = [PYTHON, '-u', str(runner_path), '--config', config_path, '--use-config-videos']
        else:
            cmd = [PYTHON, '-u', str(runner_path), '--config', config_path, video_path]
        print(f"[DEBUG] Running command: {' '.join(cmd)}")

        process = Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        with jobs_lock:
            active_jobs[job_id] = {
                'process': process,
                'status': 'running',
                'output': '',
                'config_path': config_path,
                'video_path': video_path,
                'started_at': datetime.now().isoformat(),
            }

        # Start a thread to collect output
        def collect_output():
            try:
                for line in process.stdout:
                    print(f"[DEBUG] stdout: {line.rstrip()}")
                    with jobs_lock:
                        if job_id in active_jobs:
                            active_jobs[job_id]['output'] += line
                process.wait()
                # Collect any stderr
                stderr_output = process.stderr.read() if process.stderr else ''
                if stderr_output:
                    print(f"[DEBUG] stderr: {stderr_output}")
                with jobs_lock:
                    if job_id in active_jobs:
                        if process.returncode == 0:
                            active_jobs[job_id]['status'] = 'completed'
                        elif process.returncode == -signal.SIGTERM or process.returncode == -signal.SIGKILL:
                            active_jobs[job_id]['status'] = 'stopped'
                        else:
                            active_jobs[job_id]['status'] = 'failed'
                            active_jobs[job_id]['error'] = stderr_output or f'Exit code: {process.returncode}'
                        _end_status = active_jobs[job_id]['status']
                        _started = active_jobs[job_id].get('started_at', '')
                        _config_path = active_jobs[job_id].get('config_path', '')
                        print(f"\n{'='*70}")
                        print(f"[JOB END] Job {job_id} — {_end_status}")
                        print(f"  Started:  {_started}")
                        print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        if _end_status == 'failed':
                            print(f"  Error:    {active_jobs[job_id].get('error', 'unknown')}")
                        print(f"{'='*70}")

                # Expire session config so it stops showing as active in the UI
                if _config_path:
                    _expire_session_config(_config_path)
            except Exception as e:
                print(f"[DEBUG] Exception in collect_output: {e}")
                with jobs_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id]['status'] = 'failed'
                        active_jobs[job_id]['error'] = str(e)

        thread = threading.Thread(target=collect_output, daemon=True)
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'running',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/rtsp', methods=['POST'])
def process_rtsp():
    """Start live RTSP monitoring using shared stream manager.

    Multiple sessions on the same camera share a single RTSP connection.
    Frames are decoded once and distributed to all detection engines.
    """
    data = request.json
    config_path = data.get('config_path')

    if not config_path:
        return jsonify({'error': 'Missing config_path'}), 400

    if not Path(config_path).exists():
        return jsonify({'error': f'Config not found: {config_path}'}), 404

    # Load config to get camera_id and session_id
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        return jsonify({'error': f'Failed to read config: {e}'}), 500

    camera_id = config.get('camera_id', '')
    session_id = config.get('session_id', Path(config_path).stem)
    rtsp_url = config.get('camera_url', '')

    if not rtsp_url:
        # Fall back to CAMERAS dict
        rtsp_url = CAMERAS.get(camera_id, {}).get('rtsp_url', '')
    if not rtsp_url:
        return jsonify({'error': f'No RTSP URL for camera {camera_id}'}), 400

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    output_dir = str(Path(__file__).resolve().parent.parent / 'output')

    try:
        # Create runner in-process (no subprocess needed)
        runner = SkiFramesRunner(config_path, output_dir)

        # Register with stream manager (shares RTSP connection per camera)
        stream_manager.start_session(camera_id, rtsp_url, session_id, runner)

        # Track session
        with rtsp_sessions_lock:
            rtsp_sessions[job_id] = {
                'runner': runner,
                'session_id': session_id,
                'camera_id': camera_id,
                'config_path': config_path,
                'status': 'running',
                'started_at': datetime.now().isoformat(),
            }

        print(f"[RTSP] Session {session_id} started on camera {camera_id} "
              f"(job {job_id})")

        return jsonify({
            'job_id': job_id,
            'status': 'running',
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/status/<job_id>')
def process_status(job_id):
    """Get the status of a processing job (subprocess or in-process RTSP)."""
    # Check in-process RTSP sessions first
    with rtsp_sessions_lock:
        rtsp_session = rtsp_sessions.get(job_id)

    if rtsp_session:
        runner = rtsp_session['runner']
        session_id = rtsp_session['session_id']
        camera_id = rtsp_session['camera_id']

        # Check if session is still active in stream manager
        runs_detected = runner.engine.run_count if hasattr(runner.engine, 'run_count') else 0
        actual_status = rtsp_session['status']

        # Check if stream is still running for this session
        if actual_status == 'running':
            sm_camera = stream_manager.get_session_camera(session_id)
            if sm_camera is None:
                # Session was removed (expired or stopped)
                actual_status = 'completed'
                rtsp_session['status'] = actual_status

        response = {
            'job_id': job_id,
            'status': actual_status,
            'runs_detected': runs_detected,
            'output_dir': runner.session_dir if hasattr(runner, 'session_dir') else '',
            'output': f'[In-process RTSP session on {camera_id}]\nRuns detected: {runs_detected}',
            'error': rtsp_session.get('error', ''),
        }
        return jsonify(response)

    # Fall back to subprocess jobs
    with jobs_lock:
        if job_id not in active_jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = active_jobs[job_id]
        output = job['output']

        # Parse output for progress info
        runs_detected = 0
        matched_count = 0
        total_athletes = 0
        output_dir = ''
        for line in output.split('\n'):
            # Count completed runs in real-time from "[RUN X] Completed" messages
            if '] Completed at' in line:
                runs_detected += 1
            # Also check final summary line "Runs detected: X"
            if 'Runs detected:' in line:
                try:
                    runs_detected = int(line.split(':')[1].strip())
                except:
                    pass
            # Parse "Matched so far: X/Y racers"
            if 'Matched so far:' in line:
                try:
                    parts = line.split('Matched so far:')[1].strip().split('/')
                    matched_count = int(parts[0])
                    total_athletes = int(parts[1].split()[0])
                except:
                    pass
            if 'Output directory:' in line:
                output_dir = line.split(':', 1)[1].strip()

        # Health check: verify subprocess is actually alive if status says 'running'
        actual_status = job['status']
        if actual_status == 'running' and 'process' in job:
            poll = job['process'].poll()
            if poll is not None:
                # Process has exited but status wasn't updated
                if poll == 0:
                    actual_status = 'completed'
                elif poll == -15 or poll == -9:  # SIGTERM or SIGKILL
                    actual_status = 'stopped'
                else:
                    actual_status = 'failed'
                    job['error'] = f'Process exited with code {poll}'
                job['status'] = actual_status
                print(f"[HEALTH] Job {job_id} process died (exit={poll}), status corrected to '{actual_status}'")

        response = {
            'job_id': job_id,
            'status': actual_status,
            'runs_detected': runs_detected,
            'matched_count': matched_count,
            'total_athletes': total_athletes,
            'output_dir': output_dir,
            'output': output,
            'error': job.get('error', ''),
        }
        print(f"[DEBUG] Status for {job_id}: status={actual_status}, runs={runs_detected}, matched={matched_count}/{total_athletes}")
        return jsonify(response)


@app.route('/api/process/stop/<job_id>', methods=['POST'])
def stop_process(job_id):
    """Stop a running processing job (subprocess or in-process RTSP)."""
    # Check in-process RTSP sessions first
    with rtsp_sessions_lock:
        rtsp_session = rtsp_sessions.get(job_id)

    if rtsp_session:
        session_id = rtsp_session['session_id']
        stream_manager.stop_session(session_id)
        with rtsp_sessions_lock:
            rtsp_sessions[job_id]['status'] = 'stopped'
        print(f"[STOP] In-process RTSP session {session_id} stopped (job {job_id})")
        # Expire session config
        config_path = CONFIG_DIR / f'{session_id}.json'
        _expire_session_config(str(config_path))
        return jsonify({'success': True, 'status': 'stopped'})

    # Fall back to subprocess jobs
    with jobs_lock:
        if job_id not in active_jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = active_jobs[job_id]
        if job['status'] != 'running':
            return jsonify({'error': f"Job is not running (status: {job['status']})"}), 400

        process = job['process']
        config_path = job.get('config_path', '')

    # Send SIGTERM to gracefully stop
    try:
        process.terminate()
        # Give it a moment to terminate gracefully
        try:
            process.wait(timeout=3)
        except:
            # Force kill if it doesn't stop
            process.kill()

        with jobs_lock:
            if job_id in active_jobs:
                active_jobs[job_id]['status'] = 'stopped'

        # Expire session config so it stops showing as active in the UI
        if config_path:
            _expire_session_config(config_path)

        return jsonify({'success': True, 'status': 'stopped'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/streams/status')
def streams_status():
    """Get status of shared RTSP streams and their sessions."""
    return jsonify(stream_manager.get_status())


# =============================================================================
# MONTAGE GALLERY ENDPOINTS
# =============================================================================

@app.route('/api/montages/latest')
def montages_latest():
    """List recent full-res montage images across all sessions, newest first."""
    output_dir = Path(__file__).resolve().parent.parent / 'output'
    if not output_dir.exists():
        return jsonify({'montages': []})

    montages = []
    # Walk all session directories looking for fullres images
    for session_dir in output_dir.iterdir():
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        # Find all fullres jpg files recursively
        for fullres_file in session_dir.rglob('fullres/*.jpg'):
            # Skip thumbnails
            if '_thumb' in fullres_file.name:
                continue
            stat = fullres_file.stat()
            # Build relative path from output dir for serving
            rel_path = fullres_file.relative_to(output_dir)
            # Extract FPS from filename (e.g., "run_001_093523_4.0fps.jpg" -> 4.0)
            import re as _re
            fps_match = _re.search(r'_(\d+\.?\d*)fps', fullres_file.name)
            fps_val = float(fps_match.group(1)) if fps_match else None

            # Check for matching trajectory video
            # Match by run number pattern (e.g., run_001_093523 -> run_001_093523_TR.mp4)
            base_stem = fullres_file.stem  # e.g., "run_001_093523_4.0fps"
            # Extract run identifier (before fps tag)
            import re as _re2
            run_id_match = _re2.match(r'(.+?)_[\d.]+fps', base_stem)
            trajectory_url = None
            if run_id_match:
                run_id = run_id_match.group(1)
                # Look in trajectories/ dir
                tr_dir = session_dir / 'trajectories'
                if tr_dir.exists():
                    tr_candidates = list(tr_dir.glob(f'{run_id}*_TR.mp4'))
                    if tr_candidates:
                        trajectory_url = str(tr_candidates[0].relative_to(output_dir))

            montages.append({
                'filename': fullres_file.name,
                'path': str(rel_path),
                'session_id': session_id,
                'size_kb': int(stat.st_size / 1024),
                'modified': stat.st_mtime,
                'fps': fps_val,
                'trajectory_url': trajectory_url,
            })

    # Sort by modification time, newest first
    montages.sort(key=lambda m: m['modified'], reverse=True)

    # Limit to most recent 10
    montages = montages[:10]

    return jsonify({'montages': montages})


@app.route('/api/montages/image/<path:image_path>')
def montages_image(image_path):
    """Serve a montage image from the output directory."""
    output_dir = Path(__file__).resolve().parent.parent / 'output'
    full_path = output_dir / image_path

    # Security: ensure path doesn't escape output directory
    try:
        full_path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 403

    if not full_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    return send_file(str(full_path), mimetype='image/jpeg')


# =============================================================================
# TRAJECTORY (TR) ENDPOINTS
# =============================================================================

@app.route('/api/trajectory/latest')
def trajectory_latest():
    """List recent trajectory videos across all sessions, newest first."""
    output_dir = Path(__file__).resolve().parent.parent / 'output'
    if not output_dir.exists():
        return jsonify({'trajectories': []})

    trajectories = []
    for session_dir in output_dir.iterdir():
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        # Find trajectory videos in trajectories/ subdirs
        for tr_file in session_dir.rglob('trajectories/*_TR.mp4'):
            stat = tr_file.stat()
            rel_path = tr_file.relative_to(output_dir)
            trajectories.append({
                'filename': tr_file.name,
                'path': str(rel_path),
                'session_id': session_id,
                'size_kb': int(stat.st_size / 1024),
                'modified': stat.st_mtime,
            })
        # Also check staging dirs for TR videos
        for tr_file in session_dir.rglob('*_TR.mp4'):
            if 'trajectories' in str(tr_file):
                continue  # Already found above
            stat = tr_file.stat()
            rel_path = tr_file.relative_to(output_dir)
            trajectories.append({
                'filename': tr_file.name,
                'path': str(rel_path),
                'session_id': session_id,
                'size_kb': int(stat.st_size / 1024),
                'modified': stat.st_mtime,
            })

    trajectories.sort(key=lambda t: t['modified'], reverse=True)
    trajectories = trajectories[:10]

    return jsonify({'trajectories': trajectories})


@app.route('/api/trajectory/video/<path:video_path>')
def trajectory_video(video_path):
    """Serve a trajectory video from the output directory."""
    output_dir = Path(__file__).resolve().parent.parent / 'output'
    full_path = output_dir / video_path

    # Security: ensure path doesn't escape output directory
    try:
        full_path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 403

    if not full_path.exists():
        return jsonify({'error': 'Video not found'}), 404

    return send_file(str(full_path), mimetype='video/mp4')


@app.route('/api/trajectory/generate', methods=['POST'])
def trajectory_generate():
    """
    Generate a trajectory video for a specific montage/run.

    Request body:
    {
        "session_id": "2026-02-02_0900_u14_training",
        "run_number": 1,
        "video_path": "/path/to/source/video.mp4",  (optional, for re-generation)
    }

    Looks for frames from existing video clip or cached run data.
    """
    data = request.get_json() or {}
    session_id = data.get('session_id')
    run_number = data.get('run_number')

    if not session_id:
        return jsonify({'error': 'session_id required'}), 400

    output_dir = Path(__file__).resolve().parent.parent / 'output'
    session_dir = output_dir / session_id

    if not session_dir.exists():
        return jsonify({'error': f'Session not found: {session_id}'}), 404

    # Look for existing video clip to generate trajectory from
    video_path = data.get('video_path')
    if not video_path:
        # Try to find video clip in session
        video_dir = session_dir / 'videos'
        if video_dir.exists():
            pattern = f"run_{run_number:03d}_*" if run_number else "*.mp4"
            videos = list(video_dir.glob(pattern))
            if videos:
                video_path = str(videos[0])

    if not video_path or not Path(video_path).exists():
        return jsonify({'error': 'No source video found for trajectory generation'}), 404

    # Generate trajectory video from the source video frames
    try:
        import cv2
        from trajectory import generate_trajectory_video

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': f'Cannot open video: {video_path}'}), 500

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 5:
            return jsonify({'error': 'Too few frames in video'}), 400

        # Output path
        tr_dir = session_dir / 'trajectories'
        tr_dir.mkdir(exist_ok=True)
        src_name = Path(video_path).stem
        tr_output = str(tr_dir / f"{src_name}_TR.mp4")

        result_path = generate_trajectory_video(
            frames=frames,
            output_path=tr_output,
            source_fps=fps,
        )

        if result_path:
            rel_path = Path(result_path).relative_to(output_dir)
            return jsonify({
                'success': True,
                'trajectory_url': str(rel_path),
                'size_kb': int(os.path.getsize(result_path) / 1024),
            })
        else:
            return jsonify({'error': 'Trajectory generation failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/montages/populate-manifest', methods=['POST'])
def populate_manifest_with_montages():
    """
    Scan staging directory and populate race_manifest.json with montage paths.

    Request body:
    {
        "race_slug": "western-q-2026-02-22"  // Race directory name
    }

    Scans /Volumes/OWC_48/data/montages/{race_slug}/{CamX}/{runN}/
    for montage images and updates the corresponding race_manifest.json.
    """
    data = request.get_json() or {}
    race_slug = data.get('race_slug')

    if not race_slug:
        return jsonify({'error': 'race_slug required'}), 400

    staging_dir = MONTAGES_DIR / race_slug
    if not staging_dir.exists():
        return jsonify({'error': f'Staging directory not found: {staging_dir}'}), 404

    # Find race manifest
    manifest_path = None
    if WEB_RACES_DIR.exists():
        candidate = WEB_RACES_DIR / race_slug / 'race_manifest.json'
        if candidate.exists():
            manifest_path = candidate

    if not manifest_path:
        return jsonify({'error': f'No race_manifest.json found for {race_slug}'}), 404

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Clear all existing montage entries before repopulating
        # This ensures stale entries (from renamed/deleted files) are removed
        for category in manifest.get('categories', []):
            for athlete in category.get('athletes', []):
                athlete['montages'] = {}

        # Build bib → list of (cat_idx, athlete_idx, gender_key) for all matches
        # Boys and girls share bib numbers, so we need gender-aware lookup
        bib_lookup = {}  # {bib: [(cat_idx, athlete_idx, gender_key), ...]}
        for cat_idx, category in enumerate(manifest.get('categories', [])):
            cat_id = category.get('id', '')
            # Determine gender from category id: "U12_Girls" → "girls", "U14_Boys" → "boys"
            if 'Girls' in cat_id or 'girls' in cat_id:
                gender_key = 'girls'
            elif 'Boys' in cat_id or 'boys' in cat_id:
                gender_key = 'boys'
            else:
                gender_key = ''
            for ath_idx, athlete in enumerate(category.get('athletes', [])):
                bib = athlete['bib']
                if bib not in bib_lookup:
                    bib_lookup[bib] = []
                bib_lookup[bib].append((cat_idx, ath_idx, gender_key))

        # Auto-create Forerunners category if any f-prefix files exist on disk
        # Check all cameras for f{N}_*.jpg or f{N}_*.mp4 files
        has_forerunner_files = False
        for cam_dir in sorted(staging_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('Cam'):
                continue
            for run_dir in cam_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith('run'):
                    continue
                if list(run_dir.glob('f[0-9]*.*'))[:1]:
                    has_forerunner_files = True
                    break
                frd = run_dir / 'fullres'
                if frd.is_dir() and list(frd.glob('f[0-9]*.*'))[:1]:
                    has_forerunner_files = True
                    break
            if has_forerunner_files:
                break

        if has_forerunner_files:
            # Find or create Forerunners category
            fr_cat_idx = None
            for ci, c in enumerate(manifest.get('categories', [])):
                if c.get('id') == 'Forerunners':
                    fr_cat_idx = ci
                    break
            if fr_cat_idx is None:
                manifest.setdefault('categories', []).append({
                    'id': 'Forerunners',
                    'athletes': [],
                })
                fr_cat_idx = len(manifest['categories']) - 1

            fr_cat = manifest['categories'][fr_cat_idx]
            existing_fr_bibs = {a['bib'] for a in fr_cat['athletes']}
            for fnum in range(1, 5):  # F1-F4
                if fnum not in existing_fr_bibs:
                    fr_cat['athletes'].append({
                        'bib': fnum,
                        'first': f'Forerunner',
                        'last': f'F{fnum}',
                        'club': '',
                        'is_forerunner': True,
                        'montages': {},
                    })
            # Ensure all forerunners have montages cleared
            for a in fr_cat['athletes']:
                a['montages'] = {}

            # Add forerunners to bib_lookup with gender_key 'forerunners'
            for ai, a in enumerate(fr_cat['athletes']):
                fb = a['bib']
                if fb not in bib_lookup:
                    bib_lookup[fb] = []
                bib_lookup[fb].append((fr_cat_idx, ai, 'forerunners'))

        # Scan staging directory: {CamX}/{runN}/{bib}[_thumb|_fps].jpg
        updated_count = 0
        cam_ids = set()
        for cam_dir in sorted(staging_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('Cam'):
                continue
            cam_id = cam_dir.name  # e.g., "Cam1"
            cam_ids.add(cam_id)

            for run_dir in sorted(cam_dir.iterdir()):
                if not run_dir.is_dir() or not run_dir.name.startswith('run'):
                    continue
                run_key = run_dir.name  # e.g., "run1"

                # Read _meta.json if present to determine gender context
                run_gender = ''
                meta_file = run_dir / '_meta.json'
                if meta_file.exists():
                    try:
                        with open(meta_file) as mf:
                            meta = json.load(mf)
                        run_gender = meta.get('gender', '')
                        print(f"  {cam_id}/{run_key}: gender={run_gender} (from _meta.json)")
                    except Exception:
                        pass

                # Load per-bib timing data from {prefix}{bib}_timing.json files
                timing_data = {}  # {(gender_char, bib): section_elapsed_sec}
                for tf in run_dir.glob('*_timing.json'):
                    try:
                        with open(tf) as tfh:
                            td = json.load(tfh)
                        timing_data[(td.get('gender', ''), td['bib'])] = td.get('section_elapsed_sec')
                    except Exception:
                        pass

                # Load per-detection timing data (new format with det_id)
                # Key: (gender_char, bib, det_id) -> timing dict
                det_timing = {}
                for tf in run_dir.glob('*_timing.json'):
                    try:
                        with open(tf) as tfh:
                            td = json.load(tfh)
                        bib_val = td.get('bib')
                        det_id_val = td.get('det_id', 'd000')
                        gender_val = td.get('gender', '')
                        # Map gender code to file prefix char
                        # Handles both auto-detect format ('F'/'M') and manual assign format ('g'/'b'/'f')
                        if gender_val in ('F', 'g'):
                            gc = 'g'
                        elif gender_val in ('M', 'b'):
                            gc = 'b'
                        elif gender_val == 'f':
                            gc = 'f'
                        else:
                            gc = ''
                        if bib_val is not None:
                            det_timing[(gc, bib_val, det_id_val)] = td
                        # Also populate legacy timing_data
                        timing_data[(gender_val, bib_val or 0)] = td.get('section_elapsed_sec')
                    except Exception:
                        pass

                # Collect all FPS variants per (gender_char, bib, det_id)
                # Filename formats:
                #   Multi-detect: {g|b}{bib}_{det_id}_{fps}fps.jpg  (e.g., g10_d001_5.0fps.jpg)
                #   Legacy:       {g|b}{bib}_{fps}fps.jpg           (e.g., g10_5.0fps.jpg)
                #   Old:          {bib}_{fps}fps.jpg                (no prefix)
                #   Unmatched:    unmatched_{ts}[_{det_id}]_{fps}fps.jpg
                detection_variants = {}  # (gender_char, bib, det_id) -> list of {fps, filename}
                # Scan for full-res images: check both run_dir/*.jpg and run_dir/fullres/*.jpg
                fullres_dir = run_dir / 'fullres'
                scan_dir = fullres_dir if fullres_dir.is_dir() else run_dir
                for img_file in sorted(scan_dir.glob('*.jpg')):
                    filename = img_file.stem  # e.g., "g10_d001_5.0fps" or "g4_4.0fps"

                    # Skip thumbnails and unmatched
                    if '_thumb' in filename:
                        continue
                    if filename.startswith('unmatched_'):
                        continue

                    # Parse filename parts
                    parts = filename.split('_')
                    first_part = parts[0]  # "g10" or "4" or "b12"

                    # Extract gender and bib
                    gender_char = ''
                    if first_part and first_part[0] in ('g', 'b', 'f') and first_part[1:].isdigit():
                        gender_char = first_part[0]
                        bib = int(first_part[1:])
                    elif first_part.isdigit():
                        bib = int(first_part)
                    else:
                        continue

                    # Extract det_id if present (d001, d002, etc.)
                    det_id = None
                    for p in parts[1:]:
                        if re.match(r'^d\d{3}$', p):
                            det_id = p
                            break

                    # Extract fps value
                    fps_match = re.search(r'_([\d.]+)fps$', filename)
                    fps_val = float(fps_match.group(1)) if fps_match else 0

                    # Use 'd000' for legacy files without det_id
                    key = (gender_char, bib, det_id or 'd000')
                    if key not in detection_variants:
                        detection_variants[key] = []
                    detection_variants[key].append({'fps': fps_val, 'filename': filename})

                # Build manifest entries — array of detections per bib
                bib_detections = {}  # (gender_char, bib) -> list of detection entries
                for (gender_char, bib, det_id), variants in detection_variants.items():
                    if bib not in bib_lookup:
                        print(f"  Warning: bib {bib} not found in manifest")
                        continue

                    # Sort variants by fps
                    variants.sort(key=lambda v: v['fps'])

                    # Use middle fps as default
                    default_variant = variants[len(variants) // 2]
                    default_filename = default_variant['filename']

                    # Build paths (handle both flat and fullres/thumbnails subdirectory layouts)
                    has_subdirs = fullres_dir.is_dir()
                    thumb_dir = run_dir / 'thumbnails'
                    if has_subdirs:
                        full_path = f"{cam_id}/{run_key}/fullres/{default_filename}.jpg"
                        thumb_name = f"{default_filename}_thumb.jpg"
                        thumb_file = thumb_dir / thumb_name
                        thumb_path = f"{cam_id}/{run_key}/thumbnails/{thumb_name}" if thumb_file.exists() else full_path
                    else:
                        full_path = f"{cam_id}/{run_key}/{default_filename}.jpg"
                        thumb_name = f"{default_filename}_thumb.jpg"
                        thumb_file = run_dir / thumb_name
                        thumb_path = f"{cam_id}/{run_key}/{thumb_name}" if thumb_file.exists() else full_path

                    # Get timing from det_timing or legacy
                    gender_code = 'F' if gender_char == 'g' else 'M' if gender_char == 'b' else ''
                    dt = det_timing.get((gender_char, bib, det_id), {})
                    section_time = dt.get('section_elapsed_sec') or timing_data.get((gender_code, bib))
                    trigger_time = dt.get('start_trigger_time', '')
                    # Format trigger_time as HH:MM:SS if it's an ISO timestamp
                    if trigger_time and 'T' in trigger_time:
                        try:
                            trigger_time = trigger_time.split('T')[1][:8]
                        except Exception:
                            pass

                    # Build fps_variants array
                    fps_variants = []
                    for v in variants:
                        vf = v['filename']
                        if has_subdirs:
                            v_full = f"{cam_id}/{run_key}/fullres/{vf}.jpg"
                            v_thumb_name = f"{vf}_thumb.jpg"
                            v_thumb_file = thumb_dir / v_thumb_name
                            v_thumb = f"{cam_id}/{run_key}/thumbnails/{v_thumb_name}" if v_thumb_file.exists() else v_full
                        else:
                            v_full = f"{cam_id}/{run_key}/{vf}.jpg"
                            v_thumb_name = f"{vf}_thumb.jpg"
                            v_thumb_file = run_dir / v_thumb_name
                            v_thumb = f"{cam_id}/{run_key}/{v_thumb_name}" if v_thumb_file.exists() else v_full
                        fps_variants.append({
                            'fps': v['fps'],
                            'thumb': v_thumb,
                            'full': v_full,
                        })

                    montage_entry = {
                        'det_id': det_id,
                        'thumb': thumb_path,
                        'full': full_path,
                        'fps_variants': fps_variants,
                    }
                    if section_time is not None:
                        montage_entry['section_time'] = section_time
                    if trigger_time:
                        montage_entry['trigger_time'] = trigger_time

                    # Check for video clip (with or without det_id)
                    base_prefix = f"{gender_char}{bib}" if gender_char else str(bib)
                    video_prefix = f"{base_prefix}_{det_id}" if det_id and det_id != 'd000' else base_prefix
                    video_file = run_dir / f"{video_prefix}.mp4"
                    if video_file.exists():
                        montage_entry['video'] = f"{cam_id}/{run_key}/{video_prefix}.mp4"
                    else:
                        # Try legacy video without det_id
                        legacy_video = run_dir / f"{base_prefix}.mp4"
                        if legacy_video.exists():
                            montage_entry['video'] = f"{cam_id}/{run_key}/{base_prefix}.mp4"

                    bib_key = (gender_char, bib)
                    if bib_key not in bib_detections:
                        bib_detections[bib_key] = []
                    bib_detections[bib_key].append(montage_entry)

                # Write detection arrays to manifest
                for (gender_char, bib), detections in bib_detections.items():
                    if bib not in bib_lookup:
                        continue

                    # Find correct athlete entry
                    matches = bib_lookup[bib]
                    file_gender = 'girls' if gender_char == 'g' else 'boys' if gender_char == 'b' else 'forerunners' if gender_char == 'f' else run_gender
                    if file_gender and len(matches) > 1:
                        gender_matches = [m for m in matches if m[2] == file_gender]
                        if gender_matches:
                            matches = gender_matches
                    cat_idx, ath_idx, _ = matches[0]

                    athlete = manifest['categories'][cat_idx]['athletes'][ath_idx]
                    if 'montages' not in athlete:
                        athlete['montages'] = {}
                    if cam_id not in athlete['montages']:
                        athlete['montages'][cam_id] = {}

                    # Sort detections by det_id
                    detections.sort(key=lambda d: d.get('det_id', 'd000'))

                    # Store as array (new format)
                    athlete['montages'][cam_id][run_key] = detections
                    updated_count += len(detections)

        # Save updated manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'cameras_found': sorted(cam_ids),
            'manifest_path': str(manifest_path),
        })

    except Exception as e:
        print(f"Error populating manifest: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# MONTAGE DELETE DETECTION ENDPOINT
# =============================================================================


# Delete password (simple shared secret for coach access)
DELETE_PASSWORD = os.environ.get('SKIFRAMES_DELETE_PW', 'skiframes2026')


@app.route('/api/montages/delete-detection', methods=['POST'])
def delete_detection():
    """
    Delete a single detection entry from the manifest and its files from disk.
    Requires delete password.

    Request body:
    {
        "race_slug": "western-q-2026-02-22",
        "cam_id": "Cam1",
        "run_key": "run1",
        "bib": 10,
        "gender": "g",
        "det_id": "d001",
        "password": "..."
    }
    """
    data = request.get_json() or {}

    # Password authorization check
    password = data.get('password', '')
    if password != DELETE_PASSWORD:
        return jsonify({'error': 'Wrong password.'}), 403

    race_slug = data.get('race_slug')
    cam_id = data.get('cam_id')
    run_key = data.get('run_key')
    bib = data.get('bib')
    gender = data.get('gender', '')
    det_id = data.get('det_id')

    if not all([race_slug, cam_id, run_key, bib is not None, det_id]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find and update manifest
    manifest_path = WEB_RACES_DIR / race_slug / 'race_manifest.json'
    if not manifest_path.exists():
        return jsonify({'error': f'Manifest not found for {race_slug}'}), 404

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Find athlete by bib and gender
        found = False
        for category in manifest.get('categories', []):
            cat_id = category.get('id', '')
            cat_gender = 'g' if 'girl' in cat_id.lower() else 'b' if 'boy' in cat_id.lower() else ''
            for athlete in category.get('athletes', []):
                if athlete['bib'] != bib:
                    continue
                if gender and cat_gender and gender != cat_gender:
                    continue

                montages = athlete.get('montages', {})
                cam_data = montages.get(cam_id, {})
                run_data = cam_data.get(run_key)
                if not run_data:
                    continue

                # Handle both array format and legacy single-object format
                if isinstance(run_data, list):
                    # Find and remove the detection with matching det_id
                    new_list = [d for d in run_data if d.get('det_id') != det_id]
                    if len(new_list) < len(run_data):
                        found = True
                        if new_list:
                            athlete['montages'][cam_id][run_key] = new_list
                        else:
                            del athlete['montages'][cam_id][run_key]
                elif isinstance(run_data, dict):
                    if run_data.get('det_id') == det_id or det_id == 'd000':
                        found = True
                        del athlete['montages'][cam_id][run_key]

                if found:
                    break
            if found:
                break

        if not found:
            return jsonify({'error': f'Detection {det_id} not found for bib {bib}'}), 404

        # Delete files from staging dir
        staging_dir = MONTAGES_DIR / race_slug / cam_id / run_key
        if staging_dir.exists():
            prefix = f"{gender}{bib}" if gender else str(bib)
            file_prefix = f"{prefix}_{det_id}" if det_id != 'd000' else prefix
            deleted_files = []
            for f in staging_dir.iterdir():
                if f.name.startswith(file_prefix) or (det_id == 'd000' and f.name.startswith(prefix + '_') and not re.match(r'.*_d\d{3}', f.stem)):
                    f.unlink()
                    deleted_files.append(f.name)
            # Also delete video
            video_file = staging_dir / f"{file_prefix}.mp4"
            if video_file.exists():
                video_file.unlink()
                deleted_files.append(video_file.name)

            print(f"  Deleted {len(deleted_files)} files for {file_prefix}")

        # Save updated manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return jsonify({'success': True, 'deleted_det_id': det_id, 'files_deleted': len(deleted_files) if staging_dir.exists() else 0})

    except Exception as e:
        print(f"Error deleting detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# MONTAGE BATCH PROCESSING ENDPOINTS
# =============================================================================


@app.route('/api/montage/server-command', methods=['POST'])
def generate_montage_server_command():
    """
    Generate a command to run batch montage processing on the remote server (4T).
    Returns the command string that can be copied and run via SSH.
    """
    data = request.get_json() or {}
    race_date = data.get('race_date', '')
    run_number = data.get('run_number', 'run1')
    camera_id = data.get('camera_id', 'R1')
    selected_logos = data.get('selected_logos', [])
    montage_fps_list = data.get('montage_fps_list', [4.0])
    start_zone = data.get('start_zone')
    end_zone = data.get('end_zone')
    crop_zone = data.get('crop_zone')
    detection_threshold = data.get('detection_threshold', 25)
    min_pixel_change_pct = data.get('min_pixel_change_pct', 5.0)
    min_brightness = data.get('min_brightness', 94)
    race_info = data.get('race_info', {})
    num_athletes = data.get('num_athletes', 0)

    # Convert race date from YYYY-MM-DD to YYYYMMDD for CLI
    date_compact = race_date.replace('-', '') if race_date else '20260222'

    # Build config JSON to pass to batch_montage.py
    config = {
        'race_date': race_date,
        'run_number': run_number,
        'camera_id': camera_id,
        'start_zone': start_zone,
        'end_zone': end_zone,
        'crop_zone': crop_zone,
        'detection_threshold': detection_threshold,
        'min_pixel_change_pct': min_pixel_change_pct,
        'min_brightness': min_brightness,
        'montage_fps_list': montage_fps_list,
        'selected_logos': selected_logos,
        'race_info': race_info,
    }

    # Save config to a temp file on the server
    config_filename = f"montage_config_{date_compact}_{run_number}_{camera_id}.json"
    montage_config_dir = CONFIG_DIR.parent  # edge/config/
    montage_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = montage_config_dir / config_filename
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Build the CLI command
    cmd_parts = [
        'cd /home/pa91/skiframes/photo-montages',
        'source venv/bin/activate',
        'export SKIFRAMES_DATA_DIR=/home/pa91/data',
        'export SKIFRAMES_MONTAGES_DIR=/home/pa91/data/montages',
        f'python3 edge/batch_montage.py --race-date {race_date} --run {run_number} --camera {camera_id}'
    ]

    # Add config path (convert local path to server path)
    server_config_path = str(config_path).replace('/Volumes/OWC_48/data', '/home/pa91/data')
    # If config is in local edge dir, use a relative path
    if str(config_path).startswith(str(montage_config_dir)):
        server_config_path = f'/home/pa91/skiframes/photo-montages/edge/config/{config_filename}'
    cmd_parts[-1] += f' --config "{server_config_path}"'

    # Add logos
    if selected_logos:
        cmd_parts[-1] += f' --logos "{",".join(selected_logos)}"'

    # Add FPS list
    if montage_fps_list:
        fps_str = ','.join(str(f) for f in montage_fps_list)
        cmd_parts[-1] += f' --fps-list "{fps_str}"'

    # Add test count (limit athletes)
    if num_athletes and num_athletes > 0:
        cmd_parts[-1] += f' --test {num_athletes}'

    full_command = ' && '.join(cmd_parts)
    ssh_command = f"ssh 4t '{full_command}'"

    return jsonify({
        'command': full_command,
        'ssh_command': ssh_command,
        'server': '4t',
        'race_date': race_date,
        'run_number': run_number,
        'camera_id': camera_id,
        'config_path': str(config_path),
    })


# =============================================================================
# VIDEO STITCH ENDPOINTS
# =============================================================================

# Track active stitch jobs
stitch_jobs = {}
stitch_jobs_lock = threading.Lock()


@app.route('/api/stitch/logos')
def list_available_logos():
    """
    List all available logo images for overlay selection.

    Returns list of logos with filename, name, and preview URL.
    User can select which logos to include and in what order (left to right).
    """
    logos = []
    image_extensions = {'*.png', '*.jpg', '*.jpeg'}
    if LOGOS_DIR.exists():
        all_files = []
        for ext in image_extensions:
            all_files.extend(LOGOS_DIR.glob(ext))
        for f in sorted(all_files, key=lambda x: x.name):
            # Skip hidden files
            if f.name.startswith('.'):
                continue

            # Create a friendly display name from filename
            # e.g., "US-Ski-Snowboard.png" -> "US Ski Snowboard"
            display_name = f.stem.replace('_', ' ').replace('-', ' ')

            logos.append({
                'filename': f.name,
                'name': display_name,
                'path': str(f),
                'preview_url': f'/api/stitch/logo/{f.name}'
            })

    return jsonify({
        'logos': logos,
        'default_order': [
            'US-Ski-Snowboard.png',
            'NHARA_logo.png',
            'ProctorLogo.jpg',
            'Skiframes-com_logo.png'
        ],
        'excluded_by_default': [
            'skieast_logo.png',
            'Ragged_logo.png',
            'RMST_logo.png'
        ]
    })


@app.route('/api/stitch/logo/<filename>')
def get_logo_preview(filename):
    """Serve a logo image for preview."""
    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    logo_path = LOGOS_DIR / safe_filename

    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    if not logo_path.exists() or logo_path.suffix.lower() not in allowed_extensions:
        return jsonify({'error': 'Logo not found'}), 404

    mime_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
    mimetype = mime_types.get(logo_path.suffix.lower(), 'image/png')
    return send_file(logo_path, mimetype=mimetype)


@app.route('/api/stitch/configs')
def list_stitch_configs():
    """List saved stitch cut point configurations."""
    configs = []
    for f in STITCH_CONFIG_DIR.glob('*.json'):
        try:
            with open(f, 'r') as fp:
                config = json.load(fp)
                configs.append({
                    'filename': f.name,
                    'name': config.get('name', f.stem),
                    'created': config.get('created', ''),
                    'cuts': config.get('cuts', [])
                })
        except Exception as e:
            print(f"Error reading stitch config {f}: {e}")
    return jsonify(sorted(configs, key=lambda x: x['name']))


@app.route('/api/stitch/config/<name>')
def get_stitch_config(name):
    """Get a specific stitch configuration."""
    # Sanitize name to prevent path traversal
    safe_name = re.sub(r'[^\w\-]', '_', name.lower())
    config_path = STITCH_CONFIG_DIR / f"{safe_name}.json"

    if not config_path.exists():
        return jsonify({'error': 'Config not found'}), 404

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stitch/config', methods=['POST'])
def save_stitch_config():
    """
    Save a stitch cut point configuration.

    Request body:
    {
        "name": "U12 run 1",
        "cuts": [
            {"camera": "R1", "start_pct": -0.05, "end_pct": 0.05},
            {"camera": "Axis", "start_pct": 0.05, "end_pct": 0.425},
            {"camera": "R2", "start_pct": 0.425, "end_pct": 0.575},
            {"camera": "R3", "start_pct": 0.575, "end_pct": 1.05}
        ]
    }
    """
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    cuts = data.get('cuts', [])

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    if not cuts or len(cuts) == 0:
        return jsonify({'error': 'At least one cut is required'}), 400

    # Validate cuts
    for cut in cuts:
        if 'camera' not in cut or 'start_pct' not in cut or 'end_pct' not in cut:
            return jsonify({'error': 'Each cut must have camera, start_pct, and end_pct'}), 400

    # Create config
    config = {
        'name': name,
        'created': datetime.now().isoformat(),
        'cuts': cuts
    }

    # Save to file
    safe_name = re.sub(r'[^\w\-]', '_', name.lower())
    config_path = STITCH_CONFIG_DIR / f"{safe_name}.json"

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return jsonify({'success': True, 'path': str(config_path)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stitch/parse-racers', methods=['POST'])
def parse_racers_for_stitch():
    """
    Parse Vola CSV file and results PDF to get racer data for stitching.
    Athlete names/teams loaded automatically from race_manifest.json.

    Request body:
    {
        "vola_file": "/path/to/girls-run1.csv",
        "results_file": "/path/to/results.pdf",  // Optional - for USSA IDs, rankings, DSQ/DNF
        "include_dsq_dnf": true  // Optional - include DSQ/DNF racers with 60s replacement time
    }

    Returns list of racers with timing, name, USSA ID, ranking, and status data.
    """
    data = request.get_json() or {}
    vola_file = data.get('vola_file')
    results_file = data.get('results_file')
    include_dsq_dnf = data.get('include_dsq_dnf', True)

    if not vola_file or not Path(vola_file).exists():
        return jsonify({'error': 'Invalid vola_file'}), 400

    # Load athlete names/teams from race_manifest.json
    bib_to_manifest = load_race_manifest(vola_file)

    # Parse results PDF for USSA IDs, rankings, and status
    bib_to_results = {}
    if results_file and Path(results_file).exists():
        bib_to_results = parse_results_pdf(results_file)

    # Default replacement time for DSQ/DNF (60 seconds)
    DSQ_DNF_REPLACEMENT_TIME = 60.0

    try:
        start_times = parse_vola_csv(vola_file)

        # Build racer list
        racers = []
        for bib in sorted(start_times.keys()):
            start_sec = start_times[bib]

            # Get data from results PDF (has USSA ID, rank, status)
            results_info = bib_to_results.get(bib, {})
            manifest_info = bib_to_manifest.get(bib, {})

            # Determine status from results PDF, fallback to manifest
            status = results_info.get('status') or manifest_info.get('run1_status', 'finished')
            rank = results_info.get('rank') or manifest_info.get('rank')

            # Use actual run time from manifest if available, then results PDF, then estimate
            run_duration = manifest_info.get('run1_time') or results_info.get('run_duration') or 40.0

            # Handle DSQ/DNF/DNS - use replacement time
            if status in ('DSQ', 'DNF', 'DNS'):
                if include_dsq_dnf:
                    run_duration = DSQ_DNF_REPLACEMENT_TIME
                else:
                    continue

            # Get name/team - prefer manifest (has clean data), fallback to results
            name = manifest_info.get('name') or results_info.get('name', f'Bib{bib}')
            team = manifest_info.get('team') or results_info.get('team', '')
            gender = manifest_info.get('gender') or results_info.get('gender', '') or get_gender_from_bib(bib)

            # Get USSA ID from results
            ussa_id = results_info.get('ussa_id', '')

            racer = {
                'bib': bib,
                'name': name,
                'team': team,
                'gender': gender,
                'ussa_id': ussa_id,
                'ussa_profile_url': f"https://www.usskiandsnowboard.org/public-tools/members/{ussa_id.lstrip('E')}" if ussa_id else None,
                'start_time_sec': start_sec,
                'start_time_str': seconds_to_time_str(start_sec),
                'finish_time_sec': start_sec + run_duration,
                'finish_time_str': seconds_to_time_str(start_sec + run_duration),
                'run_duration': run_duration,
                'status': status,  # 'finished', 'DSQ', 'DNF', 'DNS'
                'rank': rank,  # None for DSQ/DNF/DNS
            }
            racers.append(racer)

        # Sort by start time
        racers.sort(key=lambda r: r['start_time_sec'])

        # Extract date from Vola file path (parent dir or filename)
        vola_path = Path(vola_file)
        race_date = None
        for name_to_check in [vola_path.parent.name, vola_path.stem]:
            date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', name_to_check)
            if date_match:
                race_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
                break

        if not race_date:
            return jsonify({'error': 'Could not extract date from Vola file path'}), 400

        # Calculate time range needed (earliest start - 60s buffer, latest finish + 60s buffer)
        if racers:
            earliest_start = min(r['start_time_sec'] for r in racers) - 60
            latest_finish = max(r['finish_time_sec'] for r in racers) + 60
        else:
            earliest_start = 0
            latest_finish = 86400

        # Find videos for each camera
        camera_folders = {'R1': 'sd_R1', 'R2': 'sd_R2', 'R3': 'sd_R3', 'Axis': 'sd_Axis'}
        video_paths = {}

        for camera_id, folder_name in camera_folders.items():
            camera_videos = []

            if camera_id == 'Axis':
                # Axis has different folder structure: sd_Axis/sdcard/YYYYMMDD/HH/.../*.mkv
                axis_date_str = race_date.replace('-', '')
                axis_base = RECORDINGS_DIR / folder_name / 'sdcard' / axis_date_str
                if axis_base.exists():
                    for hour_folder in sorted(axis_base.glob('*')):
                        if not hour_folder.is_dir():
                            continue
                        for recording_folder in sorted(hour_folder.glob('*')):
                            if not recording_folder.is_dir():
                                continue
                            for mkv_file in recording_folder.glob('**/*.mkv'):
                                fname = mkv_file.stem
                                match = re.match(r'(\d{8})_(\d{6})_', fname)
                                if match:
                                    start_str = match.group(2)
                                    start_sec = int(start_str[0:2]) * 3600 + int(start_str[2:4]) * 60 + int(start_str[4:6])
                                    end_sec = start_sec + 300
                                    if end_sec >= earliest_start and start_sec <= latest_finish:
                                        camera_videos.append({
                                            'path': str(mkv_file),
                                            'start_sec': start_sec,
                                            'end_sec': end_sec,
                                        })
            else:
                # Try multiple directory structures for Reolink cameras
                search_paths = [
                    ('export', RECORDINGS_DIR / f'export_{folder_name}'),  # Exported SD copies
                    ('reolink', RECORDINGS_DIR / folder_name / 'sdcard' / 'Mp4Record' / race_date),  # Reolink SD
                    ('flat', RECORDINGS_DIR / folder_name),  # Flat directory
                ]

                for source_type, search_path in search_paths:
                    if not search_path.exists():
                        continue
                    for video_file in sorted(search_path.glob('*.mp4')):
                        # Try Reolink RecM03 format
                        parsed_date, start_sec, end_sec = parse_reolink_video_times(video_file.name)
                        if start_sec is None:
                            # Try J40 format (R1_YYYYMMDD_HHMMSS.mp4)
                            parsed_date, start_sec, end_sec = parse_j40_video_times(video_file.name)
                        if start_sec is None:
                            continue
                        if parsed_date and parsed_date != race_date:
                            continue
                        if end_sec >= earliest_start and start_sec <= latest_finish:
                            camera_videos.append({
                                'path': str(video_file),
                                'start_sec': start_sec,
                                'end_sec': end_sec,
                            })
                    if camera_videos:
                        break  # Stop after first directory with matches

            if camera_videos:
                camera_videos.sort(key=lambda v: v['start_sec'])
                video_paths[camera_id] = [v['path'] for v in camera_videos]

        return jsonify({
            'file': vola_file,
            'race_date': race_date,
            'racers': racers,
            'total_racers': len(racers),
            'video_paths': video_paths,
            'time_range': {
                'start_sec': earliest_start,
                'end_sec': latest_finish,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_stitch_job(job_id: str, config: dict):
    """Background thread function to run the stitch job."""
    from stitcher import VideoStitcher, CameraCut, Racer

    try:
        with stitch_jobs_lock:
            stitch_jobs[job_id]['status'] = 'running'
            stitch_jobs[job_id]['progress'] = {'current': 0, 'total': 0, 'current_racer': ''}

        # Extract config
        racers_data = config['racers']
        cuts_data = config['cuts']
        video_paths = config['video_paths']
        output_dir = config.get('output_dir', str(Path(__file__).parent.parent / 'output'))
        race_name = config.get('race_name', 'race')
        race_title = config.get('race_title', '')
        race_info = config.get('race_info', {})
        selected_logos = config.get('selected_logos')
        generate_comparison = config.get('generate_comparison', False)

        # Convert to objects
        cuts = [CameraCut(**c) for c in cuts_data]
        racers = []
        for r in racers_data:
            if r.get('run_duration', 0) <= 0:
                continue
            racers.append(Racer(
                bib=r['bib'],
                name=r.get('name', f"Bib{r['bib']}"),
                team=r.get('team', ''),
                gender=r.get('gender', ''),
                start_time_sec=r['start_time_sec'],
                finish_time_sec=r.get('finish_time_sec', r['start_time_sec'] + r['run_duration']),
                duration=r['run_duration'],
                ussa_id=r.get('ussa_id', ''),
                status=r.get('status', 'finished')
            ))

        with stitch_jobs_lock:
            stitch_jobs[job_id]['progress']['total'] = len(racers)

        # Stop flag function - checks if job should stop
        def should_stop():
            with stitch_jobs_lock:
                return stitch_jobs.get(job_id, {}).get('status') == 'stopping'

        # Create stitcher and run
        stitcher = VideoStitcher(
            racers=racers,
            cuts=cuts,
            video_paths=video_paths,
            output_dir=output_dir,
            race_name=race_name,
            race_title=race_title,
            race_info=race_info,
            selected_logos=selected_logos,
            stop_flag=should_stop
        )

        def progress_callback(current, total, name):
            with stitch_jobs_lock:
                stitch_jobs[job_id]['progress'] = {
                    'current': current,
                    'total': total,
                    'current_racer': name
                }

        # Get parallel processing options
        # Default workers: 8 for M1 Max, override via env SKIFRAMES_MAX_WORKERS for server
        default_workers = int(os.environ.get('SKIFRAMES_MAX_WORKERS', 8))
        parallel = config.get('parallel', True)
        max_workers = config.get('max_workers', default_workers)

        if parallel:
            print(f"Using parallel processing with {max_workers} workers")
            if generate_comparison:
                outputs = stitcher.process_all_with_comparison_parallel(
                    max_workers=max_workers,
                    progress_callback=progress_callback,
                    generate_comparison=True
                )
            else:
                outputs = stitcher.process_all_parallel(
                    max_workers=max_workers,
                    progress_callback=progress_callback
                )
        else:
            print("Using sequential processing")
            if generate_comparison:
                outputs = stitcher.process_all_with_comparison(progress_callback=progress_callback, generate_comparison=True)
            else:
                outputs = stitcher.process_all(progress_callback=progress_callback)

        with stitch_jobs_lock:
            # Check if we were stopped
            if stitch_jobs[job_id]['status'] == 'stopping':
                stitch_jobs[job_id]['status'] = 'stopped'
            else:
                stitch_jobs[job_id]['status'] = 'completed'
            stitch_jobs[job_id]['outputs'] = [str(p) for p in outputs]
            stitch_jobs[job_id]['output_dir'] = output_dir

    except Exception as e:
        with stitch_jobs_lock:
            stitch_jobs[job_id]['status'] = 'error'
            stitch_jobs[job_id]['error'] = str(e)


@app.route('/api/stitch/process', methods=['POST'])
def start_stitch_process():
    """
    Start video stitching process.

    Request body:
    {
        "racers": [...],  // From parse-racers endpoint
        "cuts": [...],    // Cut point configuration
        "video_paths": {  // Camera ID to video file path
            "R1": "/path/to/r1.mp4",
            "R2": "/path/to/r2.mp4",
            "Axis": "/path/to/axis.mp4",
            "R3": "/path/to/r3.mp4"
        },
        "race_name": "u12_run1",  // Optional
        "race_title": "Western Division Ranking - SL",  // Optional, for video overlay (deprecated, use race_info)
        "race_info": {  // Optional, for manifest metadata and title overlay
            "event": "Western Division Ranking",
            "discipline": "SL",
            "age_group": "U14",
            "run": "Run 1",
            "date": "Sunday, 2026/02/01",
            "course": "Flying Yankee",
            "location": "Ragged Mountain, NH",
            "type": "USSA/NHARA",
            "vertical_drop": "85m",
            "length": "305m",
            "gates": "35",
            "snow": "Packed Powder"
        },
        "selected_logos": [  // Optional, logo filenames in order (left to right)
            "US-Ski-Snowboard.png",
            "NHARA_logo.png",
            "RMST_logo.png",
            "Ragged_logo.png",
            "Skiframes-com_logo.png"
        ],
        "output_dir": "/path/to/output",  // Optional
        "generate_comparison": false,  // Optional, generate vs fastest videos
        "parallel": true,  // Optional, use parallel processing (default: true for M1 Mac)
        "max_workers": 4   // Optional, number of concurrent workers (default: 4 for M1 Max)
    }
    """
    data = request.get_json() or {}

    racers = data.get('racers', [])
    cuts = data.get('cuts', [])
    video_paths = data.get('video_paths', {})

    if not racers:
        return jsonify({'error': 'No racers provided'}), 400
    if not cuts:
        return jsonify({'error': 'No cuts provided'}), 400
    if not video_paths:
        return jsonify({'error': 'No video_paths provided'}), 400

    # Validate video paths exist (video_paths values can be single path or list of paths)
    for camera, paths in video_paths.items():
        # Normalize to list
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            return jsonify({'error': f'No video files provided for {camera}'}), 400
        for path in paths:
            if not Path(path).exists():
                return jsonify({'error': f'Video file not found for {camera}: {path}'}), 400

    # Create job
    job_id = str(uuid.uuid4())[:8]

    with stitch_jobs_lock:
        stitch_jobs[job_id] = {
            'status': 'starting',
            'progress': {'current': 0, 'total': len(racers), 'current_racer': ''},
            'outputs': [],
            'error': None
        }

    # Start background thread
    config = {
        'racers': racers,
        'cuts': cuts,
        'video_paths': video_paths,
        'race_name': data.get('race_name', 'race'),
        'race_title': data.get('race_title', ''),  # e.g., "Western Division U14 Ranking - SL"
        'race_info': data.get('race_info', {}),  # Structured race metadata for manifest
        'selected_logos': data.get('selected_logos'),  # Logo filenames in order (left to right)
        'output_dir': data.get('output_dir', str(Path(__file__).parent.parent / 'output')),
        'generate_comparison': data.get('generate_comparison', False)  # Generate videos vs fastest racer
    }

    thread = threading.Thread(target=run_stitch_job, args=(job_id, config))
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'starting'})


@app.route('/api/stitch/job/<job_id>')
def get_stitch_job_status(job_id):
    """Get status of a stitch job."""
    with stitch_jobs_lock:
        if job_id not in stitch_jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = stitch_jobs[job_id].copy()

    return jsonify(job)


@app.route('/api/stitch/job/<job_id>/stop', methods=['POST'])
def stop_stitch_job(job_id):
    """Stop a running stitch job (marks as stopped, thread will check)."""
    with stitch_jobs_lock:
        if job_id not in stitch_jobs:
            return jsonify({'error': 'Job not found'}), 404

        stitch_jobs[job_id]['status'] = 'stopping'

    return jsonify({'success': True})


@app.route('/api/stitch/server-command', methods=['POST'])
def generate_server_command():
    """
    Generate a command to run batch processing on the remote server (4T).
    Returns the command string that can be copied and run via SSH.

    Request body includes all UI settings: cuts, race_info, logos, etc.
    """
    data = request.get_json() or {}
    race = data.get('race', 'U14 run 1')
    date = data.get('date', '20260201')
    workers = data.get('workers', 16)
    test_count = data.get('test_count', 0)
    generate_comparison = data.get('generate_comparison', False)
    vola_file = data.get('vola_file')
    results_file = data.get('results_file')
    cuts = data.get('cuts', [])
    race_info = data.get('race_info', {})
    selected_logos = data.get('selected_logos', [])
    pre_buffer = data.get('pre_buffer', 2.0)
    post_buffer = data.get('post_buffer', 2.0)

    # Build command
    cmd_parts = [
        'cd /home/pa91/skiframes/photo-montages',
        'source venv/bin/activate',
        'export PATH=/home/pa91/bin:$PATH',  # Use NVENC-enabled ffmpeg
        'export SKIFRAMES_DATA_DIR=/home/pa91/data',
        f'python3 edge/batch_stitch.py --race "{race}" --date {date} --workers {workers}'
    ]

    # Add pre/post buffer settings
    cmd_parts[-1] += f' --pre-buffer {pre_buffer} --post-buffer {post_buffer}'

    if test_count > 0:
        cmd_parts[-1] += f' --test {test_count}'

    if generate_comparison:
        cmd_parts[-1] += ' --comparison'

    if vola_file:
        # Convert local path to server path
        server_vola = vola_file.replace('/Volumes/OWC_48/data', '/home/pa91/data')
        cmd_parts[-1] += f' --vola "{server_vola}"'

    if results_file:
        # Convert local path to server path
        server_results = results_file.replace('/Volumes/OWC_48/data', '/home/pa91/data')
        cmd_parts[-1] += f' --results "{server_results}"'

    # Add cut percentages if provided (for middle cameras - first/last use pre/post buffer)
    if cuts:
        for cut in cuts:
            camera = cut.get('camera', '').lower()
            start = cut.get('start_pct', 0) * 100
            end = cut.get('end_pct', 0) * 100
            if camera == 'r1':
                cmd_parts[-1] += f' --r1-start {start:.0f} --r1-end {end:.0f}'
            elif camera == 'axis':
                cmd_parts[-1] += f' --axis-start {start:.0f} --axis-end {end:.0f}'
            elif camera == 'r2':
                cmd_parts[-1] += f' --r2-start {start:.0f} --r2-end {end:.0f}'
            elif camera == 'r3':
                cmd_parts[-1] += f' --r3-start {start:.0f} --r3-end {end:.0f}'

    # Add race info
    if race_info.get('event'):
        cmd_parts[-1] += f' --event "{race_info["event"]}"'
    if race_info.get('discipline'):
        cmd_parts[-1] += f' --discipline "{race_info["discipline"]}"'

    # Add logos
    if selected_logos:
        cmd_parts[-1] += f' --logos "{",".join(selected_logos)}"'

    full_command = ' && '.join(cmd_parts)

    # Also provide SSH command for easy copy-paste
    # Use single quotes around the whole command to preserve inner double quotes
    ssh_command = f"ssh 4t '{full_command}'"

    return jsonify({
        'command': full_command,
        'ssh_command': ssh_command,
        'server': '4t',
        'race': race,
        'date': date,
        'workers': workers
    })


# ============================================
# GATE CALIBRATION ENDPOINTS
# ============================================

@app.route('/api/calibrate/gate-specs')
def get_gate_specs():
    """Return available gate/discipline presets."""
    return jsonify(GATE_SPECS)


@app.route('/api/calibrate/detect', methods=['POST'])
def detect_gates_endpoint():
    """Run gate detection on an already-grabbed frame."""
    data = request.json or {}
    frame_id = data.get('frame_id')
    discipline = data.get('discipline', 'sl_adult')
    min_height = int(data.get('min_height', 30))
    max_height = int(data.get('max_height', 800))
    sat_thresh = int(data.get('sat_thresh', 120))
    val_thresh = int(data.get('val_thresh', 80))

    if not frame_id:
        return jsonify({'error': 'frame_id is required'}), 400

    frame_path = CALIBRATION_FRAMES_DIR / f'{frame_id}.jpg'
    if not frame_path.exists():
        return jsonify({'error': f'Frame {frame_id} not found'}), 404

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return jsonify({'error': 'Failed to read frame'}), 500

    gates = detect_gates(frame, discipline=discipline, min_height=min_height,
                         max_height=max_height, sat_thresh=sat_thresh,
                         val_thresh=val_thresh)

    # Save annotated frame
    annotated = draw_gates_on_frame(frame, gates)
    annotated_path = CALIBRATION_FRAMES_DIR / f'{frame_id}_gates.jpg'
    cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return jsonify({
        'success': True,
        'gates': gates,
        'gate_count': len(gates),
        'discipline': discipline,
        'annotated_frame_url': f'/api/calibrate/frame/{frame_id}/annotated',
    })


@app.route('/api/calibrate/frame/<frame_id>/annotated')
def get_annotated_frame(frame_id):
    """Serve annotated frame with detected gates drawn."""
    # Sanitize frame_id
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', frame_id)
    path = CALIBRATION_FRAMES_DIR / f'{safe_id}_gates.jpg'
    if not path.exists():
        return jsonify({'error': 'Annotated frame not found'}), 404
    return send_file(str(path), mimetype='image/jpeg')


@app.route('/api/calibrate/compute', methods=['POST'])
def compute_calibration_endpoint():
    """Compute calibration from coach-adjusted gates."""
    data = request.json or {}
    frame_id = data.get('frame_id')
    camera_id = data.get('camera_id')
    discipline = data.get('discipline', 'sl_adult')
    gates = data.get('gates', [])
    slope_only = data.get('slope_only', False)

    if not frame_id:
        return jsonify({'error': 'frame_id is required'}), 400
    if not camera_id:
        return jsonify({'error': 'camera_id is required'}), 400

    # Minimum gates: 2 for slope-only, 4 for full calibration
    min_gates = 2 if slope_only else 4
    if len(gates) < min_gates:
        return jsonify({'error': f'Need at least {min_gates} gates, got {len(gates)}'}), 400

    frame_path = CALIBRATION_FRAMES_DIR / f'{frame_id}.jpg'
    if not frame_path.exists():
        return jsonify({'error': f'Frame {frame_id} not found'}), 404

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return jsonify({'error': 'Failed to read frame'}), 500

    gate_spec = GATE_SPECS.get(discipline, GATE_SPECS['sl_adult'])

    # GPS world coordinates (Mode B) — None falls back to Mode A
    world_coords = data.get('world_coords')

    # Generate calibration ID
    ts = datetime.now().strftime('%Y-%m-%d_%H%M')
    cal_id = f'{camera_id}_{ts}'

    if slope_only:
        # Slope-only mode: save gates without computing homography
        calibration = {
            'camera_id': camera_id,
            'discipline': discipline,
            'frame_id': frame_id,
            'gates': gates,
            'frame_shape': list(frame.shape[:2]),
            'calibration_id': cal_id,
            'calibration_mode': 'slope_only',
            'timestamp': datetime.now().isoformat(),
        }

        # Compute slope angle from gate pixel positions
        slope_angle_deg = None
        try:
            from slope_calculator import compute_slope_from_pixels_only
            import tempfile
            import json as json_mod

            # Write temp calibration file for slope calculation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json_mod.dump({'gates': gates}, f)
                temp_path = f.name

            result = compute_slope_from_pixels_only(temp_path)
            slope_angle_deg = result.get('slope_angle_deg')
            calibration['slope_angle_deg'] = slope_angle_deg
            calibration['slope_direction'] = result.get('direction_label')

            import os
            os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Could not compute slope angle: {e}")

        # Draw gates on verification frame (no grid)
        verify_frame = frame.copy()
        for gate in gates:
            top = tuple(gate['top'])
            base = tuple(gate['base'])
            color = (0, 0, 255) if gate.get('color') == 'red' else (255, 0, 0)
            cv2.line(verify_frame, top, base, color, 3)
            cv2.circle(verify_frame, top, 8, color, -1)
            cv2.circle(verify_frame, base, 8, color, -1)
            # Draw gate ID
            cv2.putText(verify_frame, str(gate['id']), (base[0] + 10, base[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        verify_path = CALIBRATION_FRAMES_DIR / f'{frame_id}_verify.jpg'
        cv2.imwrite(str(verify_path), verify_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Store pending calibration
        with pending_calibrations_lock:
            pending_calibrations[cal_id] = calibration

        return jsonify({
            'success': True,
            'calibration_id': cal_id,
            'slope_only': True,
            'slope_angle_deg': slope_angle_deg,
            'verification_frame_url': f'/api/calibrate/frame/{frame_id}/verification',
            'gate_count': len(gates),
        })

    # Full calibration mode (4+ gates)
    try:
        calibration = compute_calibration(gates, gate_spec, frame.shape, world_coords=world_coords)
    except Exception as e:
        return jsonify({'error': f'Calibration failed: {str(e)}'}), 500

    calibration['camera_id'] = camera_id
    calibration['discipline'] = discipline
    calibration['frame_id'] = frame_id

    # Store GPS metadata if provided
    if data.get('gps_data'):
        calibration['gps_data'] = data['gps_data']
        calibration['calibration_mode'] = 'gps'
    else:
        calibration['calibration_mode'] = 'pixel'

    calibration['calibration_id'] = cal_id

    # Draw verification grid
    try:
        verify_frame = draw_verification_grid(frame, calibration)
        verify_path = CALIBRATION_FRAMES_DIR / f'{frame_id}_verify.jpg'
        cv2.imwrite(str(verify_path), verify_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:
        return jsonify({'error': f'Verification grid failed: {str(e)}'}), 500

    # Store pending calibration
    with pending_calibrations_lock:
        pending_calibrations[cal_id] = calibration

    return jsonify({
        'success': True,
        'calibration_id': cal_id,
        'reprojection_error': calibration['reprojection_error'],
        'per_gate_errors': calibration.get('per_gate_errors', []),
        'focal_length': calibration.get('focal_length'),
        'verification_frame_url': f'/api/calibrate/frame/{frame_id}/verification',
        'gate_count': len(gates),
    })


@app.route('/api/calibrate/frame/<frame_id>/verification')
def get_verification_frame(frame_id):
    """Serve verification grid frame."""
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', frame_id)
    path = CALIBRATION_FRAMES_DIR / f'{safe_id}_verify.jpg'
    if not path.exists():
        return jsonify({'error': 'Verification frame not found'}), 404
    return send_file(str(path), mimetype='image/jpeg')


@app.route('/api/calibrate/accept', methods=['POST'])
def accept_calibration():
    """Save a computed calibration to disk."""
    data = request.json or {}
    cal_id = data.get('calibration_id')

    if not cal_id:
        return jsonify({'error': 'calibration_id is required'}), 400

    with pending_calibrations_lock:
        calibration = pending_calibrations.pop(cal_id, None)

    if calibration is None:
        return jsonify({'error': f'No pending calibration {cal_id}'}), 404

    # Save calibration JSON
    config_path = CALIBRATION_CONFIG_DIR / f'{cal_id}.json'
    with open(config_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    # Copy reference frame
    frame_id = calibration.get('frame_id', '')
    src_frame = CALIBRATION_FRAMES_DIR / f'{frame_id}.jpg'
    if src_frame.exists():
        import shutil
        dst_frame = CALIBRATION_CONFIG_DIR / f'{cal_id}_frame.jpg'
        shutil.copy2(str(src_frame), str(dst_frame))

    return jsonify({
        'success': True,
        'config_path': str(config_path),
        'calibration_id': cal_id,
    })


@app.route('/api/calibrate/status')
def calibration_status():
    """Return current calibration status for all cameras."""
    cameras = {}
    for cam_id in CAMERAS:
        # Find most recent calibration file for this camera
        pattern = f'{cam_id}_*.json'
        cal_files = sorted(CALIBRATION_CONFIG_DIR.glob(pattern), reverse=True)
        if cal_files:
            with open(cal_files[0]) as f:
                cal = json.load(f)
            cameras[cam_id] = {
                'calibrated': True,
                'calibration_id': cal.get('calibration_id', cal_files[0].stem),
                'timestamp': cal.get('timestamp', ''),
                'reprojection_error': cal.get('reprojection_error', None),
                'discipline': cal.get('discipline', ''),
            }
        else:
            cameras[cam_id] = {'calibrated': False}

    return jsonify({'cameras': cameras})


@app.route('/api/video/upload', methods=['POST'])
def upload_video():
    """Accept a video file upload for testing."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in request'}), 400

    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    # Size check (500MB limit)
    content_length = request.content_length
    if content_length and content_length > 500 * 1024 * 1024:
        return jsonify({'error': 'File too large (max 500MB)'}), 413

    # Save with UUID prefix to avoid collisions
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', video_file.filename)
    dest_name = f'{uuid.uuid4().hex[:8]}_{safe_name}'
    dest_path = UPLOAD_DIR / dest_name
    video_file.save(str(dest_path))

    # Get video metadata
    cap = cv2.VideoCapture(str(dest_path))
    info = {}
    if cap.isOpened():
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        else:
            info['duration'] = 0
        cap.release()

    size_mb = dest_path.stat().st_size / (1024 * 1024)

    return jsonify({
        'success': True,
        'video_path': str(dest_path),
        'filename': video_file.filename,
        'size_mb': round(size_mb, 1),
        **info,
    })


# =============================================================================
# DEVICE INFO & HEARTBEAT
# =============================================================================

@app.route('/api/device/info')
def device_info():
    """Return device identity and status for coach page discovery."""
    active_sessions = []
    with jobs_lock:
        for job_id, job in active_jobs.items():
            if job.get('status') == 'running':
                active_sessions.append({
                    'job_id': job_id,
                    'config_path': os.path.basename(job.get('config_path', '')),
                    'started_at': job.get('started_at', ''),
                })

    return jsonify({
        'device_id': DEVICE_ID,
        'cameras': [{'id': cid, 'name': cam.get('name', cid)}
                     for cid, cam in CAMERAS.items()],
        'active_sessions': active_sessions,
        'hostname': socket.gethostname(),
        'port': 5000,
        'timestamp': datetime.now().isoformat(),
    })


def _device_heartbeat_loop():
    """Background thread: send periodic heartbeats to the admin API."""
    import time
    try:
        import requests as http_req
    except ImportError:
        print(f"[HEARTBEAT] requests not installed, heartbeat disabled")
        return

    while True:
        try:
            active = []
            with jobs_lock:
                for job_id, job in active_jobs.items():
                    if job.get('status') == 'running':
                        # Get session_id from runner config if available
                        sid = ''
                        cfg_path = job.get('config_path', '')
                        if cfg_path:
                            sid = os.path.basename(cfg_path).replace('.json', '')
                        active.append({
                            'job_id': job_id,
                            'session_id': sid,
                            'started_at': job.get('started_at', ''),
                        })
            payload = {
                'device_id': DEVICE_ID,
                'cameras': list(CAMERAS.keys()),
                'active_sessions': active,
                'hostname': socket.gethostname(),
                'timestamp': datetime.now().isoformat(),
            }
            http_req.post(f'{ADMIN_API_URL}/device/heartbeat', json=payload, timeout=5)
        except Exception as e:
            pass  # Silent fail — heartbeat is best-effort
        time.sleep(30)


# Start heartbeat thread
_heartbeat_thread = threading.Thread(target=_device_heartbeat_loop, daemon=True)
_heartbeat_thread.start()


# ═══════════════════════════════════════════════════════════════════════════
# POSE ANALYSIS API
# ═══════════════════════════════════════════════════════════════════════════

# =============================================================================
# AI POSE ANALYSIS (async with progress)
# =============================================================================

ai_jobs = {}  # job_id -> {status, progress, total_frames, analyzed_frames, results, error}


@app.route('/api/ai/analyze', methods=['POST'])
def ai_analyze():
    """
    Launch async AI pose analysis on an athlete's video clip.
    Request body: {race_slug, cam_id, run_key, bib, gender, det_id}
    Returns: {job_id}
    """
    data = request.get_json() or {}
    race_slug = data.get('race_slug')
    cam_id = data.get('cam_id')
    run_key = data.get('run_key')
    bib = data.get('bib')
    gender = data.get('gender', '')
    det_id = data.get('det_id', 'd000')

    if not all([race_slug, cam_id, run_key, bib is not None]):
        return jsonify({'error': 'Missing required fields'}), 400

    # Find video file in montages dir
    staging_dir = MONTAGES_DIR / race_slug / cam_id / run_key
    prefix = f"{gender}{bib}" if gender else str(bib)
    video_prefix = f"{prefix}_{det_id}" if det_id and det_id != 'd000' else prefix
    video_path = staging_dir / f"{video_prefix}.mp4"

    if not video_path.exists():
        return jsonify({'error': f'Video not found: {video_path.name}'}), 404

    job_id = str(uuid.uuid4())[:8]
    ai_jobs[job_id] = {
        'status': 'running',
        'progress': 0,
        'total_frames': 0,
        'analyzed_frames': 0,
        'results': None,
        'error': None,
        'bib': bib,
        'cam_id': cam_id,
        'video_path': str(video_path),
    }

    def run_analysis():
        try:
            import subprocess as _sp
            import re

            ai_jobs[job_id]['status_msg'] = 'Starting AI analysis with SAM3...'

            # Get total frames first
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            ai_jobs[job_id]['total_frames'] = total_frames

            # Compute slope angle from race manifest using camera+gate GPS
            slope_angle_deg = 0.0
            try:
                from slope_calculator import SlopeCalculator
                manifest_path = WEB_RACES_DIR / race_slug / 'race_manifest.json'
                if manifest_path.exists():
                    calc = SlopeCalculator.from_race_manifest(
                        str(manifest_path), camera_id=cam_id, run=run_key)
                    result = calc.compute_apparent_slope_angle()
                    slope_angle_deg = result.get('slope_angle_deg', 0.0)
                    print(f"[AI] Slope angle for {cam_id}/{run_key}: {slope_angle_deg}° "
                          f"(method={result.get('method')}, pitch={result.get('pitch_deg')}°)")
            except Exception as e:
                print(f"[AI] Warning: Could not compute slope angle: {e}")

            # Output annotated video alongside original: g10.mp4 → g10_ai.mp4
            ai_video_name = video_path.stem + '_ai_raw.mp4'
            ai_video_path = video_path.parent / ai_video_name

            # Run pose_analyzer_yolo.py as subprocess with ARM64 Python for SAM3 support
            ai_jobs[job_id]['status_msg'] = 'Loading SAM3 model...'
            edge_dir = Path(__file__).parent
            cmd = [
                PYTHON_AI,
                str(edge_dir / 'pose_analyzer_yolo.py'),
                str(video_path),
                str(slope_angle_deg),
                '--output-video', str(ai_video_path),
                '--model', 'l',
            ]
            print(f"[AI] Running: {' '.join(cmd)}")

            # Run subprocess and capture output for progress tracking
            process = _sp.Popen(
                cmd,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                bufsize=0,  # Unbuffered for real-time progress
                cwd=str(edge_dir),
            )

            ai_jobs[job_id]['status_msg'] = 'Analyzing frames with SAM3...'
            poses_detected = 0
            progress_pattern = re.compile(r'Progress:\s*([\d.]+)%.*Poses:\s*(\d+)')

            # Read output character by character to handle \r progress updates
            buffer = ''
            while True:
                char = process.stdout.read(1)
                if not char:
                    # Process finished
                    break

                if char == '\r' or char == '\n':
                    # End of a progress line - parse it
                    if buffer:
                        match = progress_pattern.search(buffer)
                        if match:
                            pct = float(match.group(1))
                            poses_detected = int(match.group(2))
                            ai_jobs[job_id]['progress'] = int(pct)
                            ai_jobs[job_id]['analyzed_frames'] = int(pct * total_frames / 100)

                        if 'Done!' in buffer:
                            ai_jobs[job_id]['progress'] = 95
                            ai_jobs[job_id]['status_msg'] = 'Encoding for browser...'

                        buffer = ''
                else:
                    buffer += char

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"pose_analyzer_yolo.py failed with code {process.returncode}")

            if not ai_video_path.exists():
                raise RuntimeError(f"Output video not created: {ai_video_path}")

            # Re-encode to H.264 for browser compatibility (OpenCV mp4v = mpeg4, not playable in Safari/Chrome)
            ai_jobs[job_id]['status_msg'] = 'Encoding video for browser...'
            ai_video_final = video_path.parent / (video_path.stem + '_ai.mp4')
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(ai_video_path),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Required for Safari
                '-movflags', '+faststart',  # Enables streaming
                str(ai_video_final),
            ]
            _sp.run(ffmpeg_cmd, capture_output=True, check=True)

            # Remove the raw mp4v version
            ai_video_path.unlink()

            # Build relative path for serving via /montages/ route
            # e.g. Cam1/run1/g10_ai.mp4
            rel_path = str(ai_video_final.relative_to(MONTAGES_DIR / race_slug))

            ai_jobs[job_id]['status'] = 'complete'
            ai_jobs[job_id]['progress'] = 100
            ai_jobs[job_id]['results'] = {
                'ai_video': rel_path,
                'frames_analyzed': poses_detected,
                'total_frames': total_frames,
                'slope_angle_deg': slope_angle_deg,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            ai_jobs[job_id]['status'] = 'error'
            ai_jobs[job_id]['error'] = str(e)

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/api/ai/status/<job_id>')
def ai_status(job_id):
    """Get progress/results for an AI analysis job."""
    job = ai_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    # Return only fields needed by the frontend
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'total_frames': job['total_frames'],
        'analyzed_frames': job['analyzed_frames'],
        'results': job['results'],
        'error': job['error'],
        'status_msg': job.get('status_msg', ''),
    })


@app.route('/api/analyze/pose', methods=['POST'])
def analyze_pose_endpoint():
    """Run AI pose analysis on a racer's video segment."""
    data = request.json or {}
    bib = data.get('bib')
    name = data.get('name', f'Bib{bib}')
    vola_file = data.get('vola_file')
    video_path = data.get('video_path')
    race = data.get('race')
    calibration_id = data.get('calibration_id')

    if not video_path:
        return jsonify({'error': 'video_path is required'}), 400

    # Find calibration file
    calibration_path = None
    if calibration_id:
        cal_path = CALIBRATIONS_DIR / f'{calibration_id}.json'
        if cal_path.exists():
            calibration_path = str(cal_path)
    else:
        # Try to find the most recent calibration
        cals = sorted(CALIBRATIONS_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if cals:
            calibration_path = str(cals[0])

    # Generate output path
    output_dir = Path('/tmp/pose_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    output_video = output_dir / f'{safe_name}_{bib}_pose.mp4'

    try:
        import subprocess

        cmd = [
            'python3', 'pose_analyzer_yolo.py',
            video_path,
            '--output-video', str(output_video),
        ]

        if calibration_path:
            cmd.extend(['--calibration', calibration_path])

        # Run pose analysis (this can take a while)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            return jsonify({
                'error': f'Analysis failed: {result.stderr}',
                'stdout': result.stdout,
            }), 500

        # Try to open the output video
        try:
            subprocess.Popen(['open', str(output_video)])
        except Exception:
            pass  # Ignore if open fails

        return jsonify({
            'success': True,
            'output_video': str(output_video),
            'stdout': result.stdout,
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis timed out (5 min limit)'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Unmatched Video Reconciliation ──────────────────────────────────────────

@app.route('/api/unmatched/list', methods=['GET'])
def list_unmatched():
    """
    List unmatched video clips for a race.

    Query params:
        race  - race slug (e.g. "western-q-2026-02-22")
        cam   - optional camera filter (e.g. "Cam1")
        run   - optional run filter (e.g. "run1")

    Returns JSON with unmatched clips grouped by camera, plus counts.
    """
    race_slug = request.args.get('race')
    cam_filter = request.args.get('cam', '')
    run_filter = request.args.get('run', '')

    if not race_slug:
        return jsonify({'error': 'race parameter required'}), 400

    staging_dir = MONTAGES_DIR / race_slug
    if not staging_dir.exists():
        return jsonify({'error': f'Staging directory not found: {staging_dir}'}), 404

    unmatched = []
    counts = {}

    # Scan each camera directory
    for cam_dir in sorted(staging_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith('Cam'):
            continue
        cam_id = cam_dir.name
        if cam_filter and cam_id != cam_filter:
            continue

        for run_dir in sorted(cam_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('run'):
                continue
            run_id = run_dir.name
            if run_filter and run_id != run_filter:
                continue

            # Find unmatched .mp4 files
            for mp4 in sorted(run_dir.glob('unmatched_*.mp4')):
                stem = mp4.stem  # e.g. "unmatched_090540_d001"

                # Read companion timing JSON
                timing_path = run_dir / f'{stem}_timing.json'
                timing = {}
                if timing_path.exists():
                    try:
                        with open(timing_path) as f:
                            timing = json.load(f)
                    except Exception:
                        pass

                # Find thumbnail (check thumbnails/ subdir)
                thumb = ''
                thumb_dir = run_dir / 'thumbnails'
                if thumb_dir.is_dir():
                    thumbs = sorted(thumb_dir.glob(f'{stem}_*_thumb.jpg'))
                    if thumbs:
                        # Use middle FPS variant as representative thumb
                        thumb = f'{cam_id}/{run_id}/thumbnails/{thumbs[len(thumbs)//2].name}'

                # Find fullres montage images (for preview)
                fullres_dir = run_dir / 'fullres'
                montage = ''
                if fullres_dir.is_dir():
                    montages = sorted(fullres_dir.glob(f'{stem}_*fps.jpg'))
                    if montages:
                        montage = f'{cam_id}/{run_id}/fullres/{montages[len(montages)//2].name}'

                # Extract timestamp from filename: unmatched_HHMMSS_d001
                parts = stem.split('_')
                ts_str = parts[1] if len(parts) >= 2 else '?'
                timestamp = f'{ts_str[:2]}:{ts_str[2:4]}:{ts_str[4:6]}' if len(ts_str) == 6 else ts_str

                entry = {
                    'id': stem,
                    'cam': cam_id,
                    'run': run_id,
                    'timestamp': timestamp,
                    'trigger_time': timing.get('start_trigger_time', ''),
                    'duration_sec': timing.get('section_elapsed_sec', 0),
                    'video': f'{cam_id}/{run_id}/{mp4.name}',
                    'thumb': thumb,
                    'montage': montage,
                }
                unmatched.append(entry)

                counts[cam_id] = counts.get(cam_id, 0) + 1

    cameras = sorted(counts.keys())
    return jsonify({
        'unmatched': unmatched,
        'cameras': cameras,
        'counts': counts,
    })


@app.route('/api/unmatched/assign', methods=['POST'])
def assign_unmatched():
    """
    Assign a bib number to an unmatched detection.
    Renames all associated files from unmatched_{ts}_{det_id}* to {gender}{bib}_{det_id}*
    and rebuilds the race manifest.

    Request body:
    {
        "race": "western-q-2026-02-22",
        "id": "unmatched_090540_d001",
        "bib": 7,
        "gender": "g",
        "cam": "Cam1",
        "run": "run1"
    }
    """
    data = request.get_json() or {}
    race_slug = data.get('race')
    unmatched_id = data.get('id')        # e.g. "unmatched_090540_d001"
    bib = data.get('bib')
    gender = data.get('gender', '')       # "g" or "b"
    cam = data.get('cam')
    run = data.get('run')

    if not all([race_slug, unmatched_id, bib, cam, run]):
        return jsonify({'error': 'Missing required fields: race, id, bib, cam, run'}), 400

    try:
        bib = int(bib)
    except (ValueError, TypeError):
        return jsonify({'error': 'bib must be an integer'}), 400

    staging_dir = MONTAGES_DIR / race_slug
    run_dir = staging_dir / cam / run
    if not run_dir.exists():
        return jsonify({'error': f'Directory not found: {run_dir}'}), 404

    # Build the new prefix: e.g. "g7_d001" or "b12_d001"
    # Extract det_id from unmatched_id: "unmatched_090540_d001" → "d001"
    id_parts = unmatched_id.split('_')
    det_id = id_parts[2] if len(id_parts) >= 3 else 'd001'
    new_prefix = f'{gender}{bib}_{det_id}'

    renamed_files = 0

    # Rename in run_dir (mp4, timing json)
    for f in run_dir.glob(f'{unmatched_id}*'):
        suffix = f.name[len(unmatched_id):]  # e.g. ".mp4", "_timing.json"
        new_name = new_prefix + suffix
        new_path = f.parent / new_name
        if new_path.exists():
            print(f'[unmatched] WARNING: target exists, skipping: {new_path}')
            continue
        f.rename(new_path)
        renamed_files += 1

    # Rename in thumbnails/ subdir
    thumb_dir = run_dir / 'thumbnails'
    if thumb_dir.is_dir():
        for f in thumb_dir.glob(f'{unmatched_id}*'):
            suffix = f.name[len(unmatched_id):]
            new_name = new_prefix + suffix
            new_path = f.parent / new_name
            if new_path.exists():
                print(f'[unmatched] WARNING: target exists, skipping: {new_path}')
                continue
            f.rename(new_path)
            renamed_files += 1

    # Rename in fullres/ subdir
    fullres_dir = run_dir / 'fullres'
    if fullres_dir.is_dir():
        for f in fullres_dir.glob(f'{unmatched_id}*'):
            suffix = f.name[len(unmatched_id):]
            new_name = new_prefix + suffix
            new_path = f.parent / new_name
            if new_path.exists():
                print(f'[unmatched] WARNING: target exists, skipping: {new_path}')
                continue
            f.rename(new_path)
            renamed_files += 1

    # Update the timing JSON if it was renamed
    timing_path = run_dir / f'{new_prefix}_timing.json'
    if timing_path.exists():
        try:
            with open(timing_path) as f:
                timing = json.load(f)
            timing['bib'] = bib
            # Map file prefix to canonical gender code used by detection pipeline
            gender_code_map = {'g': 'F', 'b': 'M', 'f': 'f'}
            timing['gender'] = gender_code_map.get(gender, gender)
            timing['matched'] = True
            with open(timing_path, 'w') as f:
                json.dump(timing, f, indent=2)
        except Exception as e:
            print(f'[unmatched] Warning: failed to update timing JSON: {e}')

    # Rebuild the race manifest
    manifest_rebuilt = False
    try:
        with app.test_request_context(json={'race_slug': race_slug}):
            resp = populate_manifest_with_montages()
            if hasattr(resp, 'status_code') and resp.status_code == 200:
                manifest_rebuilt = True
            elif isinstance(resp, tuple) and resp[1] == 200:
                manifest_rebuilt = True
            elif not isinstance(resp, tuple):
                manifest_rebuilt = True
    except Exception as e:
        print(f'[unmatched] Warning: manifest rebuild failed: {e}')

    return jsonify({
        'ok': True,
        'renamed_files': renamed_files,
        'new_prefix': new_prefix,
        'manifest_rebuilt': manifest_rebuilt,
    })


@app.route('/api/unmatched/delete', methods=['POST'])
def delete_unmatched():
    """
    Delete all files for an unmatched detection (video, montages, thumbnails, timing JSON).

    Request body:
    {
        "race": "western-q-2026-02-22",
        "id": "unmatched_090540_d001",
        "cam": "Cam1",
        "run": "run1"
    }
    """
    data = request.get_json() or {}
    race_slug = data.get('race')
    unmatched_id = data.get('id')
    cam = data.get('cam')
    run = data.get('run')

    if not all([race_slug, unmatched_id, cam, run]):
        return jsonify({'error': 'Missing required fields: race, id, cam, run'}), 400

    staging_dir = MONTAGES_DIR / race_slug
    run_dir = staging_dir / cam / run
    if not run_dir.exists():
        return jsonify({'error': f'Directory not found: {run_dir}'}), 404

    deleted_files = 0

    # Delete from run_dir (mp4, timing json)
    for f in list(run_dir.glob(f'{unmatched_id}*')):
        try:
            f.unlink()
            deleted_files += 1
        except Exception as e:
            print(f'[unmatched] Warning: failed to delete {f}: {e}')

    # Delete from thumbnails/ subdir
    thumb_dir = run_dir / 'thumbnails'
    if thumb_dir.is_dir():
        for f in list(thumb_dir.glob(f'{unmatched_id}*')):
            try:
                f.unlink()
                deleted_files += 1
            except Exception as e:
                print(f'[unmatched] Warning: failed to delete {f}: {e}')

    # Delete from fullres/ subdir
    fullres_dir = run_dir / 'fullres'
    if fullres_dir.is_dir():
        for f in list(fullres_dir.glob(f'{unmatched_id}*')):
            try:
                f.unlink()
                deleted_files += 1
            except Exception as e:
                print(f'[unmatched] Warning: failed to delete {f}: {e}')

    print(f'[unmatched] Deleted {deleted_files} files for {unmatched_id} in {cam}/{run}')

    return jsonify({
        'ok': True,
        'deleted_files': deleted_files,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=os.environ.get('FLASK_DEBUG', '0') == '1')

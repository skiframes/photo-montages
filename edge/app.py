#!/usr/bin/env python3
"""
Edge device Flask application for ski run detection and calibration.
Serves mobile-first calibration UI for coaches to configure trigger zones.
"""

import os
import json
import uuid
import signal
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import Popen, PIPE

import cv2
import re
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

app = Flask(__name__, static_folder='static', template_folder='static')

# Track active processing jobs
# Key: job_id, Value: dict with 'process', 'status', 'output', 'config_path', 'video_path'
active_jobs = {}
jobs_lock = threading.Lock()

# Configuration
CONFIG_DIR = Path(__file__).parent / 'config' / 'zones'
STITCH_CONFIG_DIR = Path(__file__).parent / 'config' / 'stitch'
CALIBRATION_FRAMES_DIR = Path(tempfile.gettempdir()) / 'skiframes_calibration'
VIDEO_DIR = Path(__file__).parent.parent  # Parent of edge/ for finding test videos
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
STITCH_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Camera definitions with time offset percentages for race timing
# offset_pct: what percentage of run time has elapsed when skier enters this camera's view
CAMERAS = {
    'R1': {
        'name': 'R1 - Start Gate',
        'rtsp_url': os.environ.get('R1_RTSP_URL', 'rtsp://192.168.1.101/h264Preview_01_main'),
        'offset_pct': 0,  # 0% - covers start gate
    },
    'R2': {
        'name': 'R2 - Zoom Down',
        'rtsp_url': os.environ.get('R2_RTSP_URL', 'rtsp://192.168.1.102/h264Preview_01_main'),
        'offset_pct': 10,  # 10% into the run
    },
    'Axis': {
        'name': 'Axis PTZ - Mid Course',
        'rtsp_url': os.environ.get('AXIS_RTSP_URL', 'rtsp://192.168.1.100/axis-media/media.amp'),
        'offset_pct': 20,  # 20% into the run
    },
    'R3': {
        'name': 'R3 - Finish',
        'rtsp_url': os.environ.get('R3_RTSP_URL', 'rtsp://192.168.1.103/h264Preview_01_main'),
        'offset_pct': 50,  # 50% - covers last half of run
    },
}

# Vola Excel file location (external drive, outside git repo)
VOLA_DIR = Path('/Volumes/OWC_48/data/vola')

# Session types and groups
SESSION_TYPES = ['training', 'race']
GROUPS = ['U10', 'U12', 'U14', 'Scored', 'Masters']


@app.route('/')
def index():
    """Serve the calibration UI."""
    return send_from_directory('static', 'calibration.html')


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
    """List available Vola Excel files (searches subdirectories)."""
    if not HAS_OPENPYXL:
        return jsonify({'error': 'openpyxl not installed'}), 500

    files = []
    if VOLA_DIR.exists():
        # Search recursively for xlsx files
        for f in VOLA_DIR.glob('**/*.xlsx'):
            # Show relative path from VOLA_DIR for clarity
            rel_path = f.relative_to(VOLA_DIR)
            files.append({
                'name': str(rel_path),
                'path': str(f),
            })
    return jsonify(sorted(files, key=lambda x: x['name']))


@app.route('/api/vola/startlist-files')
def list_startlist_files():
    """List available start list PDF files (for racer names and teams)."""
    files = []
    if VOLA_DIR.exists():
        # Search recursively for PDF files containing 'start-list'
        for f in VOLA_DIR.glob('**/*start-list*.pdf'):
            rel_path = f.relative_to(VOLA_DIR)
            files.append({
                'name': str(rel_path),
                'path': str(f),
            })
    return jsonify(sorted(files, key=lambda x: x['name']))


def parse_startlist_pdf(pdf_path: str) -> dict:
    """
    Parse a start list PDF to extract bib-to-racer mapping.

    Start list format: "111 Stellato Brigid 2013 SUN"
    (bib lastname firstname year team)

    PDF has two columns, so a line might contain two racers.
    Gender is determined by bib number ranges:
    - U12 Women: 1-60, U12 Men: 61-99
    - U14 Women: 111-170, U14 Men: 171-220

    Returns dict mapping bib number to {'name': 'FirstnameLastname', 'team': 'SUN', 'gender': 'Women'}
    """
    if not HAS_PDFPLUMBER:
        return {}

    bib_to_racer = {}

    def get_gender_from_bib(bib: int) -> str:
        """Determine gender from bib number based on standard ranges."""
        # U12: Women 1-60, Men 61-99
        # U14: Women 111-170, Men 171-220
        if 1 <= bib <= 60 or 111 <= bib <= 170:
            return 'Women'
        elif 61 <= bib <= 110 or 171 <= bib <= 220:
            return 'Men'
        return ''

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                for line in text.split('\n'):
                    # Find all racers in the line (may have 2 due to two-column layout)
                    # Pattern: bib (name words) year team
                    # Name can be 2-4 words (e.g., "Van Der Berg John", "Doe Jane", "O'Brien Mary")
                    # Match: bib, then name words, then 4-digit year, then 2-4 letter team code
                    matches = re.findall(r'(\d+)\s+((?:[A-Za-z\'\-]+\s+){1,3}[A-Za-z\'\-]+)\s+(\d{4})\s+([A-Z]{2,4})', line)
                    for match in matches:
                        bib = int(match[0])
                        name_parts = match[1].strip().split()
                        year = match[2]
                        team = match[3]

                        # Name format: last word is firstname, rest is lastname
                        # "Stellato Brigid" -> firstname=Brigid, lastname=Stellato
                        # "Van Der Berg John" -> firstname=John, lastname=VanDerBerg
                        if len(name_parts) >= 2:
                            firstname = name_parts[-1]
                            lastname = ''.join(name_parts[:-1])  # Join multi-word last names
                        else:
                            firstname = name_parts[0] if name_parts else ''
                            lastname = ''

                        bib_to_racer[bib] = {
                            'name': f"{firstname}{lastname}",
                            'team': team,
                            'gender': get_gender_from_bib(bib)
                        }
    except Exception as e:
        print(f"Error parsing start list PDF {pdf_path}: {e}")

    return bib_to_racer


@app.route('/api/vola/results-files')
def list_results_files():
    """List available results PDF files (for racer names) - DEPRECATED, use startlist-files."""
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


def parse_results_pdf(pdf_path: str) -> dict:
    """
    Parse a results PDF to extract bib-to-name mapping.
    DEPRECATED: Use parse_startlist_pdf instead for better data (includes team).

    Returns dict mapping bib number to name (e.g., {132: "Nelson Coulee", 131: "Cooper Josie"})
    """
    if not HAS_PDFPLUMBER:
        return {}

    bib_to_name = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                # Look for lines with bib numbers and names
                # Format: "1 132 E7031024 Nelson Coulee 2012 PROC 26.89 ..."
                # Also handles DNS/DNF: "132 E7031024 Nelson Coulee 2012 PROC"
                for line in text.split('\n'):
                    # Pattern 1: with rank - "1 132 E7031024 Lastname Firstname 2012"
                    match = re.match(r'^\d+\s+(\d+)\s+[A-Z]\d+\s+([A-Za-z]+)\s+([A-Za-z]+)\s+\d{4}\s', line)
                    if match:
                        bib = int(match.group(1))
                        lastname = match.group(2)
                        firstname = match.group(3)
                        bib_to_name[bib] = f"{firstname}{lastname}"
                        continue

                    # Pattern 2: without rank (DNS/DNF) - "132 E7031024 Lastname Firstname 2012"
                    match = re.match(r'^(\d+)\s+[A-Z]\d+\s+([A-Za-z]+)\s+([A-Za-z]+)\s+\d{4}\s', line)
                    if match:
                        bib = int(match.group(1))
                        lastname = match.group(2)
                        firstname = match.group(3)
                        if bib not in bib_to_name:  # Don't overwrite existing entry
                            bib_to_name[bib] = f"{firstname}{lastname}"
    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")

    return bib_to_name


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


@app.route('/api/vola/parse', methods=['POST'])
def parse_vola_file():
    """
    Parse a Vola Excel file and return timing data for a specific race.

    Request body:
    {
        "file_path": "/path/to/vola.xlsx",
        "race": "U12 run 1",  // One of: "U12 run 1", "U12 run 2", "U14 run 1", "U14 run 2"
        "camera_id": "R1",  // Camera to use for offset calculation
    }

    Returns list of racers with their timing windows for the specified camera.
    """
    if not HAS_OPENPYXL:
        return jsonify({'error': 'openpyxl not installed'}), 500

    data = request.get_json() or {}
    file_path = data.get('file_path')
    race = data.get('race', '').lower()
    camera_id = data.get('camera_id', 'R1')

    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'Invalid file_path'}), 400

    # Map race selection to sheet names
    race_map = {
        'u12 run 1': ('u12 start run 1', 'u12 end run 1'),
        'u12 run 2': ('u12 start run 2', 'u12 end run 2'),
        'u14 run 1': ('u14 start run 1', 'u14 end run 1'),
        'u14 run 2': ('u14 start run 2', 'u14 end run 2'),
    }

    if race not in race_map:
        return jsonify({'error': f'Invalid race. Must be one of: {list(race_map.keys())}'}), 400

    start_sheet, end_sheet = race_map[race]
    camera_offset_pct = CAMERAS.get(camera_id, {}).get('offset_pct', 0) / 100.0

    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)

        # Parse start times (Column B = bib, Column C = time)
        start_times = {}
        ws_start = wb[start_sheet]
        for row in ws_start.iter_rows(min_row=2, values_only=True):
            bib_raw = row[1]  # Column B (index 1)
            time_str = row[2]  # Column C (index 2)
            if bib_raw and time_str:
                try:
                    bib = int(bib_raw)  # Convert to int for consistent sorting
                except (ValueError, TypeError):
                    continue
                start_seconds = parse_vola_time(str(time_str))
                if start_seconds is not None:
                    start_times[bib] = start_seconds

        # Parse end times and run durations (Column B = bib, Column C = time, Column D = duration)
        end_times = {}
        run_durations = {}
        ws_end = wb[end_sheet]
        for row in ws_end.iter_rows(min_row=2, values_only=True):
            bib_raw = row[1]  # Column B
            time_str = row[2]  # Column C
            duration = row[3] if len(row) > 3 else None  # Column D
            if bib_raw:
                try:
                    bib = int(bib_raw)  # Convert to int for consistent sorting
                except (ValueError, TypeError):
                    continue
                if time_str:
                    end_seconds = parse_vola_time(str(time_str))
                    if end_seconds is not None:
                        end_times[bib] = end_seconds
                if duration is not None and isinstance(duration, (int, float)):
                    run_durations[bib] = float(duration)

        wb.close()

        # Build racer list with timing windows for the camera
        racers = []
        for bib in sorted(start_times.keys()):
            start_sec = start_times[bib]

            # Calculate run duration from end time or direct duration
            run_duration = None
            if bib in end_times:
                run_duration = end_times[bib] - start_sec
            elif bib in run_durations:
                run_duration = run_durations[bib]

            # Calculate camera-specific timing window
            # Camera offset: when the skier enters this camera's view
            if run_duration and run_duration > 0:
                camera_start_sec = start_sec + (run_duration * camera_offset_pct)
                # Assume camera covers ~20% of the run (configurable)
                camera_duration = run_duration * 0.2
                camera_end_sec = camera_start_sec + camera_duration
            else:
                # No finish time - estimate based on average run duration (~40 seconds)
                estimated_duration = 40.0
                camera_start_sec = start_sec + (estimated_duration * camera_offset_pct)
                camera_duration = estimated_duration * 0.2
                camera_end_sec = camera_start_sec + camera_duration
                run_duration = None  # Mark as estimated

            racer = {
                'bib': bib,
                'start_time_sec': start_sec,
                'start_time_str': seconds_to_time_str(start_sec),
                'run_duration': run_duration,
                'finished': bib in end_times,
                'camera_start_sec': camera_start_sec,
                'camera_start_str': seconds_to_time_str(camera_start_sec),
                'camera_end_sec': camera_end_sec,
                'camera_end_str': seconds_to_time_str(camera_end_sec),
            }
            racers.append(racer)

        # Sort by start time
        racers.sort(key=lambda r: r['start_time_sec'])

        return jsonify({
            'race': race,
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
        "file_path": "/path/to/vola.xlsx",
        "race": "U12 run 1",
        "camera_id": "R1",
        "video_start_time": "10:41:00",  // HH:MM:SS format (Boston time)
        "video_duration_sec": 120  // Video length in seconds
    }

    Returns list of racers with their camera timing windows.
    """
    if not HAS_OPENPYXL:
        return jsonify({'error': 'openpyxl not installed'}), 500

    data = request.get_json() or {}
    file_path = data.get('file_path')
    race = data.get('race', '').lower()
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

    # Get all racers for this race
    from flask import url_for
    response = parse_vola_file()
    if response.status_code != 200:
        return response

    all_racers = response.get_json().get('racers', [])

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
        'race': race,
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
        cap = cv2.VideoCapture(source)

        if source_type == 'video' and seek_seconds > 0:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_seconds * fps))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'error': 'Failed to grab frame'}), 500

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

    # Generate session ID
    now = datetime.now()
    session_id = f"{now.strftime('%Y-%m-%d_%H%M')}_{data['group'].lower()}_{data['session_type']}"

    # Default session end time: 90 minutes from now
    session_end = data.get('session_end_time')
    if not session_end:
        session_end = (now + timedelta(minutes=90)).isoformat()

    config = {
        'session_id': session_id,
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
        'montage_fps': data.get('montage_fps', 4),
        'start_offset_sec': data.get('start_offset_sec', 0),  # Fixed delay in seconds after trigger
        # Vola racer data for naming montages
        'vola_racers': data.get('vola_racers', []),
        'vola_videos': data.get('vola_videos', []),  # Videos to process
        'vola_race': data.get('vola_race', ''),
        'vola_camera': data.get('vola_camera', ''),
        'vola_view': data.get('vola_view', '1'),  # View number 1-5 for filename
        'race_date': data.get('race_date', now.strftime('%Y-%m-%d')),
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
    """Stop a session by setting its end time to now."""
    config_path = CONFIG_DIR / f'{session_id}.json'
    if not config_path.exists():
        return jsonify({'error': 'Config not found'}), 404

    with open(config_path) as f:
        config = json.load(f)

    # Set end time to now to mark session as ended
    config['session_end_time'] = datetime.now().isoformat()

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return jsonify({'success': True, 'session_id': session_id})


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


# External recordings drive path
RECORDINGS_DIR = Path('/Volumes/OWC_48/data/recordings')


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

    if not RECORDINGS_DIR.exists():
        return jsonify({'error': 'Recordings drive not mounted', 'videos': []})

    # Map camera_id to folder name
    camera_folders = {
        'R1': 'sd_R1',
        'R2': 'sd_R2',
        'R3': 'sd_R3',
        'Axis': 'sd_Axis',
    }

    folder_name = camera_folders.get(camera_id)
    if not folder_name:
        return jsonify({'error': f'Unknown camera_id: {camera_id}'}), 400

    # Build path to videos
    videos_path = RECORDINGS_DIR / folder_name / 'sdcard' / 'Mp4Record' / date

    if not videos_path.exists():
        return jsonify({'error': f'No recordings found for {camera_id} on {date}', 'videos': []})

    videos = []
    for video_file in sorted(videos_path.glob('*.mp4')):
        parsed_date, start_sec, end_sec = parse_reolink_video_times(video_file.name)

        if start_sec is None:
            continue  # Skip files that don't match expected format

        # Filter by time range if specified
        if filter_start is not None and filter_end is not None:
            # Check if video overlaps with the filter range
            if end_sec < filter_start or start_sec > filter_end:
                continue  # Video doesn't overlap with filter range

        # Format times for display
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

    Request body:
    {
        "vola_file": "/path/to/vola.xlsx",
        "race": "U12 run 1",
        "camera_id": "R1",
        "num_athletes": 5,  // Number of athletes to process (0 or null = all)
        "startlist_file": "/path/to/start-list.pdf"  // Optional: PDF with racer names and teams
    }

    Returns list of videos needed and the racers they cover (with names/teams if startlist_file provided).
    """
    if not HAS_OPENPYXL:
        return jsonify({'error': 'openpyxl not installed'}), 500

    data = request.get_json() or {}
    vola_file = data.get('vola_file')
    race = data.get('race', '').lower()
    camera_id = data.get('camera_id', 'R1')
    num_athletes = data.get('num_athletes', 0)  # 0 = all
    startlist_file = data.get('startlist_file')  # Optional PDF with names and teams

    # Parse start list PDF for names and teams if provided
    bib_to_racer = {}
    if startlist_file and Path(startlist_file).exists():
        bib_to_racer = parse_startlist_pdf(startlist_file)

    if not vola_file or not Path(vola_file).exists():
        return jsonify({'error': 'Invalid vola_file'}), 400

    # First, parse the Vola file to get racer timing data
    # Re-use the parse_vola_file logic
    race_map = {
        'u12 run 1': ('u12 start run 1', 'u12 end run 1'),
        'u12 run 2': ('u12 start run 2', 'u12 end run 2'),
        'u14 run 1': ('u14 start run 1', 'u14 end run 1'),
        'u14 run 2': ('u14 start run 2', 'u14 end run 2'),
    }

    if race not in race_map:
        return jsonify({'error': f'Invalid race: {race}'}), 400

    start_sheet, end_sheet = race_map[race]
    camera_offset_pct = CAMERAS.get(camera_id, {}).get('offset_pct', 0) / 100.0

    try:
        wb = openpyxl.load_workbook(vola_file, data_only=True)

        # Parse start times
        start_times = {}
        ws_start = wb[start_sheet]
        for row in ws_start.iter_rows(min_row=2, values_only=True):
            bib_raw = row[1]
            time_str = row[2]
            if bib_raw and time_str:
                try:
                    bib = int(bib_raw)
                except (ValueError, TypeError):
                    continue
                start_seconds = parse_vola_time(str(time_str))
                if start_seconds is not None:
                    start_times[bib] = start_seconds

        # Parse end times
        end_times = {}
        run_durations = {}
        ws_end = wb[end_sheet]
        for row in ws_end.iter_rows(min_row=2, values_only=True):
            bib_raw = row[1]
            time_str = row[2]
            duration = row[3] if len(row) > 3 else None
            if bib_raw:
                try:
                    bib = int(bib_raw)
                except (ValueError, TypeError):
                    continue
                if time_str:
                    end_seconds = parse_vola_time(str(time_str))
                    if end_seconds is not None:
                        end_times[bib] = end_seconds
                if duration is not None and isinstance(duration, (int, float)):
                    run_durations[bib] = float(duration)

        wb.close()

        # Build racer list with camera timing windows
        racers = []
        for bib in sorted(start_times.keys()):
            start_sec = start_times[bib]

            run_duration = None
            if bib in end_times:
                run_duration = end_times[bib] - start_sec
            elif bib in run_durations:
                run_duration = run_durations[bib]

            if run_duration and run_duration > 0:
                camera_start_sec = start_sec + (run_duration * camera_offset_pct)
                camera_duration = run_duration * 0.2
                camera_end_sec = camera_start_sec + camera_duration
            else:
                estimated_duration = 40.0
                camera_start_sec = start_sec + (estimated_duration * camera_offset_pct)
                camera_duration = estimated_duration * 0.2
                camera_end_sec = camera_start_sec + camera_duration

            # Get name, team, and gender from start list
            racer_info = bib_to_racer.get(bib, {})
            racers.append({
                'bib': bib,
                'name': racer_info.get('name', ''),  # Name from start list PDF
                'team': racer_info.get('team', ''),  # Team from start list PDF
                'gender': racer_info.get('gender', ''),  # Gender from start list PDF
                'start_time_sec': start_sec,
                'run_duration': run_duration,  # Actual run time from Vola (for start offset calculation)
                'camera_start_sec': camera_start_sec,
                'camera_end_sec': camera_end_sec,
                'finished': bib in end_times,
            })

        # Sort by start time
        racers.sort(key=lambda r: r['start_time_sec'])

        # Limit to num_athletes if specified
        if num_athletes and num_athletes > 0:
            racers = racers[:num_athletes]

        if not racers:
            return jsonify({'error': 'No racers found', 'videos': [], 'racers': []})

        # Find the time range we need videos for
        first_camera_time = min(r['camera_start_sec'] for r in racers)
        last_camera_time = max(r['camera_end_sec'] for r in racers)

        # Add some buffer (30 seconds before and after)
        filter_start = first_camera_time - 30
        filter_end = last_camera_time + 30

        # Extract date from Vola filename (format: Vola_export_U12_U14_02-01-2026.xlsx)
        vola_name = Path(vola_file).stem
        date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', vola_name)
        if date_match:
            race_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
        else:
            return jsonify({'error': 'Could not extract date from Vola filename'}), 400

        # Get videos that cover this time range
        videos_response = list_recording_videos()
        # Actually call the function with proper request context
        from flask import request as flask_request

        # Build path to videos
        camera_folders = {'R1': 'sd_R1', 'R2': 'sd_R2', 'R3': 'sd_R3', 'Axis': 'sd_Axis'}
        folder_name = camera_folders.get(camera_id, 'sd_R1')
        videos_path = RECORDINGS_DIR / folder_name / 'sdcard' / 'Mp4Record' / race_date

        videos = []
        if videos_path.exists():
            for video_file in sorted(videos_path.glob('*.mp4')):
                parsed_date, start_sec, end_sec = parse_reolink_video_times(video_file.name)
                if start_sec is None:
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

        return jsonify({
            'race': race,
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
    runner_path = Path(__file__).parent / 'runner.py'

    try:
        if use_config_videos:
            # Use --use-config-videos flag to process videos from config's vola_videos list
            cmd = ['python3', '-u', str(runner_path), '--config', config_path, '--use-config-videos']
        else:
            cmd = ['python3', '-u', str(runner_path), '--config', config_path, video_path]
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
                        print(f"[DEBUG] Job {job_id} finished with status: {active_jobs[job_id]['status']}")
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


@app.route('/api/process/status/<job_id>')
def process_status(job_id):
    """Get the status of a processing job."""
    with jobs_lock:
        if job_id not in active_jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = active_jobs[job_id]
        output = job['output']

        # Parse output for progress info
        runs_detected = 0
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
            if 'Output directory:' in line:
                output_dir = line.split(':', 1)[1].strip()

        response = {
            'job_id': job_id,
            'status': job['status'],
            'runs_detected': runs_detected,
            'output_dir': output_dir,
            'output': output,
            'error': job.get('error', ''),
        }
        print(f"[DEBUG] Status for {job_id}: status={job['status']}, runs={runs_detected}")
        return jsonify(response)


@app.route('/api/process/stop/<job_id>', methods=['POST'])
def stop_process(job_id):
    """Stop a running processing job."""
    with jobs_lock:
        if job_id not in active_jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = active_jobs[job_id]
        if job['status'] != 'running':
            return jsonify({'error': f"Job is not running (status: {job['status']})"}), 400

        process = job['process']

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

        return jsonify({'success': True, 'status': 'stopped'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# VIDEO STITCH ENDPOINTS
# =============================================================================

# Track active stitch jobs
stitch_jobs = {}
stitch_jobs_lock = threading.Lock()


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
    Parse Vola file and start list PDF to get racer data for stitching.

    Request body:
    {
        "vola_file": "/path/to/vola.xlsx",
        "race": "U12 run 1",
        "startlist_file": "/path/to/startlist.pdf"  // Optional
    }

    Returns list of racers with timing and name data.
    """
    if not HAS_OPENPYXL:
        return jsonify({'error': 'openpyxl not installed'}), 500

    data = request.get_json() or {}
    vola_file = data.get('vola_file')
    race = data.get('race', '').lower()
    startlist_file = data.get('startlist_file')

    if not vola_file or not Path(vola_file).exists():
        return jsonify({'error': 'Invalid vola_file'}), 400

    # Map race selection to sheet names
    race_map = {
        'u12 run 1': ('u12 start run 1', 'u12 end run 1'),
        'u12 run 2': ('u12 start run 2', 'u12 end run 2'),
        'u14 run 1': ('u14 start run 1', 'u14 end run 1'),
        'u14 run 2': ('u14 start run 2', 'u14 end run 2'),
    }

    if race not in race_map:
        return jsonify({'error': f'Invalid race. Must be one of: {list(race_map.keys())}'}), 400

    # Parse start list for names if provided
    bib_to_racer = {}
    if startlist_file and Path(startlist_file).exists():
        bib_to_racer = parse_startlist_pdf(startlist_file)

    start_sheet, end_sheet = race_map[race]

    try:
        wb = openpyxl.load_workbook(vola_file, data_only=True)

        # Parse start times
        start_times = {}
        ws_start = wb[start_sheet]
        for row in ws_start.iter_rows(min_row=2, values_only=True):
            bib_raw = row[1]
            time_str = row[2]
            if bib_raw and time_str:
                try:
                    bib = int(bib_raw)
                except (ValueError, TypeError):
                    continue
                start_seconds = parse_vola_time(str(time_str))
                if start_seconds is not None:
                    start_times[bib] = start_seconds

        # Parse end times and durations
        end_times = {}
        run_durations = {}
        ws_end = wb[end_sheet]
        for row in ws_end.iter_rows(min_row=2, values_only=True):
            bib_raw = row[1]
            time_str = row[2]
            duration = row[3] if len(row) > 3 else None
            if bib_raw:
                try:
                    bib = int(bib_raw)
                except (ValueError, TypeError):
                    continue
                if time_str:
                    end_seconds = parse_vola_time(str(time_str))
                    if end_seconds is not None:
                        end_times[bib] = end_seconds
                if duration is not None and isinstance(duration, (int, float)):
                    run_durations[bib] = float(duration)

        wb.close()

        # Build racer list
        racers = []
        for bib in sorted(start_times.keys()):
            start_sec = start_times[bib]

            # Calculate run duration
            run_duration = None
            if bib in end_times:
                run_duration = end_times[bib] - start_sec
            elif bib in run_durations:
                run_duration = run_durations[bib]

            if not run_duration or run_duration <= 0:
                continue  # Skip racers without valid finish

            # Get name/team from start list
            racer_info = bib_to_racer.get(bib, {})
            name = racer_info.get('name', f'Bib{bib}')
            team = racer_info.get('team', '')
            gender = racer_info.get('gender', '')

            racer = {
                'bib': bib,
                'name': name,
                'team': team,
                'gender': gender,
                'start_time_sec': start_sec,
                'start_time_str': seconds_to_time_str(start_sec),
                'finish_time_sec': start_sec + run_duration,
                'finish_time_str': seconds_to_time_str(start_sec + run_duration),
                'run_duration': run_duration,
                'finished': bib in end_times
            }
            racers.append(racer)

        # Sort by start time
        racers.sort(key=lambda r: r['start_time_sec'])

        # Find videos for all 4 cameras
        # Extract date from Vola filename (format: Vola_export_U12_U14_02-01-2026.xlsx)
        vola_name = Path(vola_file).stem
        date_match = re.search(r'(\d{2})-(\d{2})-(\d{4})', vola_name)
        if not date_match:
            return jsonify({'error': 'Could not extract date from Vola filename'}), 400

        race_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"

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
                # Axis has different folder structure: sd_Axis/sdcard/YYYYMMDD/HH/YYYYMMDD_HHMMSS_XXXX_XXX/YYYYMMDD_HH/*.mkv
                axis_date_str = race_date.replace('-', '')  # 2026-02-01 -> 20260201
                axis_base = RECORDINGS_DIR / folder_name / 'sdcard' / axis_date_str
                if axis_base.exists():
                    # Search through hour folders
                    for hour_folder in sorted(axis_base.glob('*')):
                        if not hour_folder.is_dir():
                            continue
                        for recording_folder in sorted(hour_folder.glob('*')):
                            if not recording_folder.is_dir():
                                continue
                            # Find mkv files in subdirectories
                            for mkv_file in recording_folder.glob('**/*.mkv'):
                                # Parse Axis filename: YYYYMMDD_HHMMSS_XXXX.mkv
                                fname = mkv_file.stem  # e.g., 20260201_081402_A4CC
                                match = re.match(r'(\d{8})_(\d{6})_', fname)
                                if match:
                                    start_str = match.group(2)
                                    start_sec = int(start_str[0:2]) * 3600 + int(start_str[2:4]) * 60 + int(start_str[4:6])
                                    # Estimate end time (Axis videos are typically ~5 min segments)
                                    end_sec = start_sec + 300

                                    if end_sec >= earliest_start and start_sec <= latest_finish:
                                        camera_videos.append({
                                            'path': str(mkv_file),
                                            'start_sec': start_sec,
                                            'end_sec': end_sec,
                                        })
            else:
                # Reolink cameras: sd_XX/sdcard/Mp4Record/YYYY-MM-DD/*.mp4
                videos_path = RECORDINGS_DIR / folder_name / 'sdcard' / 'Mp4Record' / race_date
                if videos_path.exists():
                    for video_file in sorted(videos_path.glob('*.mp4')):
                        parsed_date, start_sec, end_sec = parse_reolink_video_times(video_file.name)
                        if start_sec is None:
                            continue

                        if end_sec >= earliest_start and start_sec <= latest_finish:
                            camera_videos.append({
                                'path': str(video_file),
                                'start_sec': start_sec,
                                'end_sec': end_sec,
                            })

            # Return ALL videos that overlap with the race time range, sorted by start time
            if camera_videos:
                camera_videos.sort(key=lambda v: v['start_sec'])
                video_paths[camera_id] = [v['path'] for v in camera_videos]

        return jsonify({
            'race': race,
            'race_date': race_date,
            'racers': racers,
            'total_racers': len(racers),
            'video_paths': video_paths,  # Now a dict of camera_id -> list of video paths
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
                duration=r['run_duration']
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
            stop_flag=should_stop
        )

        def progress_callback(current, total, name):
            with stitch_jobs_lock:
                stitch_jobs[job_id]['progress'] = {
                    'current': current,
                    'total': total,
                    'current_racer': name
                }

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
        "output_dir": "/path/to/output"  // Optional
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

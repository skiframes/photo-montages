# photo-montages

Web application for ski race photo montage generation with AWS-hosted review interface at skiframes.com.

## Project Context

This is part of an open source video analytics platform for alpine ski racing at Ragged Mountain, NH. The system processes RTSP camera streams or recorded video files, detects skier runs via frame differencing in configurable trigger zones, generates stop-motion photo montages (inspired by Ron LeMaster's technique photography), uploads them to AWS for review at skiframes.com, and supports printing selected montages.

**Existing code to reference:** [github.com/skiframes/run-detection](https://github.com/skiframes/run-detection) — contains proven `axis_detect.py` (frame differencing detection), trigger zone configs, video clip extraction, and stop-motion photo montage generation using OpenCV/NumPy/ffmpeg/ImageMagick. Port and adapt this logic rather than rewriting from scratch.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  EDGE (J40/MS-01 on mountain)                                       │
│                                                                     │
│  1. Calibration UI: grab frame from RTSP/video, draw START & END   │
│     trigger zones. Gates are between the two zones.                │
│                                                                     │
│  2. Detection: monitor stream, detect skier entering START zone,   │
│     track until they exit END zone = one run                       │
│                                                                     │
│  3. Photo montage generation: create stop-motion overlay image     │
│                                                                     │
│  4. Upload: push montage to AWS (S3 + CloudFront)                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AWS (skiframes.com)                                                │
│                                                                     │
│  • S3 bucket for photo montages                                    │
│  • CloudFront for fast delivery                                    │
│  • Static site or Lambda-backed API                                │
│                                                                     │
│  Two views:                                                        │
│  - PUBLIC (athletes/parents): browse by date, filter training vs   │
│    race, no login required                                         │
│  - COACH VIEW: password protected, can delete montages, print      │
└─────────────────────────────────────────────────────────────────────┘
```

## Trigger Zone Concept

The calibration UI lets you:
1. Select a frame from live RTSP stream or a video recording
2. Draw a **START zone** (rectangle near top of frame where skiers enter)
3. Draw an **END zone** (rectangle near bottom of frame where skiers exit)

The gates being filmed are **between** these two zones. When a skier:
- Enters the START zone → begin tracking this run
- Exits the END zone → run complete → generate composite → upload

This captures the full gate sequence in between.

## Hardware Environment

### Cameras
- **R1, R2:** Reolink RLC-823S2 — 2K/30fps, 5Mbps RTSP streams. Known for non-monotonic RTSP timestamps (use `-fflags +genpts` when recording).
- **R3:** Reolink RLC-823S2 — same model, finish line position.
- **Axis Q6135-LE:** PTZ camera, 1080p/60fps RTSP stream.

### Edge Compute
- **Prototype:** Seeed Studio reComputer J40 (Jetson Orin NX). Single camera.
- **Production:** Minisforum MS-01 (i9-13900H, 14 cores, Quick Sync). Three cameras.
- Both run Ubuntu.

**Video processing approach:** Start with pure CPU ffmpeg decode — GPU encoding has shown quality issues with fast-moving skiers (blur). For this pipeline, only decoding is needed (composites are JPEG images, not re-encoded video), so CPU decode is acceptable and simpler to debug. GPU acceleration can be added later if CPU becomes a bottleneck.

### Network
- Edge device has 5G uplink via Teltonika RUTM50 router.
- Composites upload to AWS over 5G (~200KB thumbnail + ~3MB full-res per run).
- Coaches access calibration UI on phones via 5G.
- Mac mini in race lodge on local network + internet.

### Race Lodge Display Station (Mac mini)
- Mac mini with TV screen + video projector
- Brother MFC-J5855DW printer connected to Mac mini (tabloid 11x17)
- Displays gallery on TV/projector for group review
- Handles all printing locally
- Accesses same AWS review site as coaches/athletes on phones

### Server Processing Mode (Tesla GPU)

For batch processing recorded race videos much faster than real-time. Same codebase runs on a server with 4× Tesla GPUs.

**Use case:** Process hours of race footage overnight, or re-process with different composite settings.

**Hardware:** Server with 4× Tesla GPUs (V100 or T4)

**Parallel processing:**
- Each GPU processes a different video file or camera stream
- GPU-accelerated decode (NVDEC) + frame differencing (CUDA)
- Multiple composite generations in parallel
- 10-50× faster than real-time processing

**Same pipeline, different scale:**
- Same detection logic, same composite generation
- Config specifies input folder of video files
- Outputs upload to same S3/AWS structure
- Results appear in same review website

## Application Components

### 1. Calibration UI (Edge, accessed via mobile phone)

Web interface served from edge device. Coaches access from their phone (iPhone/Android) over 5G — no iPad needed on the slope.

**Mobile-first design:**
- Touch-friendly on small screens (iPhone-sized)
- Large tap targets for zone drawing
- Pinch-to-zoom on calibration frame
- Simple dropdown for group selection
- Works on any modern mobile browser

**Workflow:**
1. Coach opens edge device URL on phone (e.g., via Tailscale or public URL)
2. Select camera from list
3. Select input source: live RTSP stream OR video file
4. Grab/scrub to select a representative frame
5. Pinch-zoom and draw START zone rectangle (touch-drag)
6. Pinch-zoom and draw END zone rectangle
7. Select session type: Training / Race
8. Select group: U10 / U12 / U14 / Scored / Masters
9. Set session end time (default: 90 minutes from now)
10. Adjust detection sensitivity (threshold, min pixel change %)
11. Save config → starts new training session

Any coach can update trigger zones mid-session from their phone if camera position changes.

Detection automatically stops at session end time to avoid false positives from grooming equipment, coaches, lift shadows, etc.

**Config format:**
```json
{
  "session_id": "2026-02-02_0900_u14_training",
  "session_type": "training",
  "group": "U14",
  "camera_id": "axis_ptz_mid",
  "camera_url": "rtsp://...",
  "calibration_frame": "calibration_axis_ptz_mid.jpg",
  "start_zone": {"x": 400, "y": 100, "w": 200, "h": 80},
  "end_zone": {"x": 350, "y": 900, "w": 250, "h": 80},
  "session_start_time": "2026-02-02T09:00:00-05:00",
  "session_end_time": "2026-02-02T10:30:00-05:00",
  "pre_buffer_seconds": 2,
  "post_buffer_seconds": 2,
  "detection_threshold": 25,
  "min_pixel_change_pct": 5.0
}
```

**Session type:** `training` or `race`

**Groups:** U10, U12, U14, Scored, Masters

**Session end time:** Detection automatically stops at this time. Calibration UI defaults to 90 minutes from now, adjustable via time picker. Prevents false positives after training ends (e.g., grooming equipment, coaches skiing through, lift shadows).

### 2. Detection Engine (Edge or Server)

Runs on edge device (real-time) or Tesla GPU server (batch processing).

**Three modes:**

**Live mode (edge):** Connect to RTSP stream, maintain frame ring buffer (~30s), run frame differencing on START and END zones continuously. CPU decode.

**Replay mode (edge):** Process a folder of recorded video files. Same detection logic, different input. CPU decode.

**Server mode (Tesla GPU):** Batch process video files with GPU acceleration. NVDEC decode, CUDA frame differencing, parallel composite generation across 4 GPUs. 10-50× faster than real-time.

**Detection logic (same for all modes):**
- Frame differencing in START zone fires → mark run start, begin buffering
- Frame differencing in END zone fires → mark run end
- Extract frames from start to end (plus pre/post buffer)
- Pass to composite generator

### 3. Photo Montage Generator (Edge or Server)

Takes the extracted run frames and generates a stop-motion overlay image (photo montage) showing the skier's progression through the gates.

**Output:**
- Thumbnail: ~800px wide, JPEG 85%, ~200KB (for quick review)
- Full resolution: 3300×5100 pixels at 300 DPI (tabloid), JPEG 95%, ~3-5MB

**Processing:**
- Edge (J40/MS-01): CPU-based OpenCV/ImageMagick
- Server (Tesla GPU): CUDA-accelerated image processing, parallel generation

Port existing logic from run-detection repo.

### 4. AWS Uploader (Edge)

After composite generation:
1. Upload thumbnail to S3
2. Upload full-res to S3
3. Record metadata (timestamp, camera_id, run_number) to DynamoDB or JSON manifest

**S3 structure:**
```
s3://skiframes/
  sessions/
    2026-02-02_0900_u14_training/
      manifest.json
      run_001_axis_ptz_thumb.jpg
      run_001_axis_ptz_full.jpg
      run_002_axis_ptz_thumb.jpg
      ...
    2026-02-02_1030_scored_race/
      manifest.json
      run_001_axis_ptz_thumb.jpg
      ...
```

**Manifest format:**
```json
{
  "session_id": "2026-02-02_0900_u14_training",
  "session_type": "training",
  "group": "U14",
  "start_time": "2026-02-02T09:00:00-05:00",
  "end_time": "2026-02-02T10:30:00-05:00",
  "runs": [
    {"id": "run_001", "camera": "axis_ptz", "timestamp": "2026-02-02T09:05:23-05:00", "deleted": false},
    {"id": "run_002", "camera": "axis_ptz", "timestamp": "2026-02-02T09:06:45-05:00", "deleted": false}
  ]
}
```

### 5. Review Website (skiframes.com)

Static site on S3 + CloudFront, or simple Lambda-backed API.

**Organization:** Site organized by date and session type. Filter by:
- **Training** vs **Race**
- Date picker (today's sessions by default, access to past dates)
- Group: U10 / U12 / U14 / Scored / Masters

When calibrating trigger zones on phone, coach selects the group and session type (training or race).

**Public View (athletes, parents — no login):**
- Landing page: "Today's Sessions" button front and center
- Browse by date, filter by training/race, filter by group
- Grid of all photo montage thumbnails
- Click thumbnail → view full size
- No delete capability

**Coach View (password protected):**
- Same browsing/filtering as public
- Delete button (removes from S3, coach curation)
- "Print" button → downloads full-res for local printing
- Bulk select/delete

**Print Queue:**
- List of montages marked for printing
- "Print" button downloads full-res image
- Mac mini (or any computer with printer) prints via standard browser print or downloads and prints locally

### 6. Lodge Display Mode (Mac mini)

The Mac mini in the race lodge runs a browser showing the review website on TV/projector:
- **Display mode:** Auto-refreshing gallery view, new composites appear as they're uploaded
- **Review mode:** Same coach interface for delete/print decisions
- Printing: click print → browser downloads full-res → print dialog → Brother printer connected to Mac mini

No special edge-to-printer bridge needed. The website is the interface; printing happens from whatever device has the printer (Mac mini).

## Tech Stack

### Edge Device
- Python (Flask or FastAPI)
- OpenCV for frame capture and differencing
- ffmpeg for video file processing
- ImageMagick for composite assembly
- boto3 for S3 uploads

### AWS
- S3 for image storage
- CloudFront for CDN
- DynamoDB or S3 JSON manifests for metadata
- Static site (S3 hosted) or Lambda + API Gateway for dynamic features
- Simple auth: HTTP Basic or query param tokens (not high security, just access control)

## File Organization

```
photo-montages/
├── CLAUDE.md
├── README.md
├── requirements.txt
│
├── edge/                      # Runs on J40/MS-01 (real-time)
│   ├── app.py                 # Flask entry point
│   ├── calibration.py         # Frame grab, zone drawing UI backend
│   ├── detection.py           # Frame differencing, run detection
│   ├── montage.py             # Photo montage generation
│   ├── uploader.py            # S3 upload logic
│   ├── static/
│   │   └── calibration.html   # Trigger zone calibration UI (mobile-first)
│   └── config/
│       └── zones/             # Per-camera JSON configs
│
├── server/                    # Runs on Tesla GPU server (batch)
│   ├── batch.py               # Batch processing entry point
│   ├── gpu_detection.py       # CUDA-accelerated detection
│   ├── gpu_montage.py         # CUDA-accelerated montage generation
│   └── parallel.py            # Multi-GPU job distribution
│
├── shared/                    # Common code used by edge and server
│   ├── detection_core.py      # Detection logic (CPU/GPU agnostic)
│   ├── montage_core.py        # Montage logic (CPU/GPU agnostic)
│   ├── uploader.py            # S3 upload (shared)
│   └── config.py              # Config parsing
│
├── web/                       # Deploys to skiframes.com
│   ├── index.html             # Landing: "Today's Sessions" button
│   ├── browse.html            # Browse by date, filter training/race
│   ├── coach.html             # Coach view (password protected)
│   ├── js/
│   │   └── app.js             # Gallery logic, filters, delete, print
│   └── css/
│       └── style.css
│
├── infrastructure/            # Deployment scripts
│   ├── deploy-edge.sh
│   ├── deploy-server.sh
│   └── deploy-aws.sh
│
└── credentials.local          # Gitignored: AWS keys, camera passwords
```

## Security

- `credentials.local` excluded from git — contains AWS credentials, camera RTSP URLs with passwords
- Public site: no login needed for athletes/parents to view montages
- Coach password: simple shared password for delete/print access
- No PII stored — just photo montages and run timestamps
- S3 bucket public read via CloudFront for montage images

## Development Sequence

**Phase 1: Edge prototype (J40 + single camera)**
1. **Calibration UI** — Mobile-first frame selection and zone drawing. Critical for race day.
2. **Detection engine** — Port frame differencing from run-detection, CPU decode.
3. **Montage generator** — Port from run-detection, verify output quality.
4. **AWS uploader** — S3 upload with proper structure.

**Phase 2: skiframes.com**
5. **Public browse** — Date picker, training/race filter, group filter, thumbnail grid.
6. **Coach view** — Password protected, delete and print buttons.
7. **Display mode** — Auto-refresh view for TV/projector in lodge.

**Phase 3: Server batch processing**
8. **GPU detection** — CUDA-accelerated frame differencing.
9. **GPU montages** — CUDA-accelerated image processing.
10. **Multi-GPU parallel** — Distribute jobs across 4 Tesla GPUs.

**Phase 4: Scale edge to 3 cameras**
11. Test J40 with 3 cameras (may need GPU decode).
12. Or deploy MS-01 for 3-camera production.

## Key Design Decisions

- **Trigger zones bracket the gates** — START at top, END at bottom, gates in between. Simpler than trying to detect individual gate passages.
- **Upload everything, delete bad ones** — Easier than trying to identify "good" runs automatically. Coach curation is fast.
- **Edge generates, AWS hosts** — Edge device does heavy lifting (detection, montage), AWS just serves static files.
- **Public access for athletes** — No login friction. Anyone can browse skiframes.com to see today's sessions.
- **Training vs Race filter** — Easy way to find relevant sessions.
- **Thumbnails for 5G, full-res for print** — Bandwidth-conscious design for mountain cellular.
- **Ron LeMaster-inspired** — Photo montages follow the style of his technique photography in "Ultimate Skiing".

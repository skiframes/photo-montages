# photo-montages — Claude Code Session Brief

## What I'm Building

A system for ski race video analysis hosted at skiframes.com that:
1. Grabs a frame from an RTSP stream or video file
2. Lets me draw START and END trigger zones on that frame (gates are between them)
3. Detects all skiers passing through both zones
4. Generates stop-motion photo montages for each run (Ron LeMaster style)
5. Uploads montages to AWS automatically
6. Public website at skiframes.com — athletes browse by date, filter training vs race
7. Coach view with delete/print capabilities (password protected)
8. Supports printing selected montages to a tabloid printer

## What Already Exists

I have a working repo at `github.com/skiframes/run-detection` with:
- `axis_detect.py` — frame differencing detection on trigger zones (proven reliable)
- Trigger zone JSON configs
- Video clip extraction
- Stop-motion photo montage generation using OpenCV/NumPy/ffmpeg/ImageMagick

Clone this repo first and study the existing code. Port the detection and montage logic.

## New Repo: skiframes/photo-montages

I've committed a CLAUDE.md with full architecture. Read it first.

## The Components

### Edge Device (J40 on mountain)

1. **Calibration UI** — Mobile-first web page (iPhone-sized). Coach accesses from phone over 5G. Pinch-zoom frame, draw START/END zones. Select session type (Training/Race). Select group (U10/U12/U14/Scored/Masters). Set session end time (default 90min). Any coach can update zones mid-session.

2. **Detection Engine** — Monitor RTSP stream or process video files. Frame differencing on trigger zones. START fires → track run. END fires → complete. Auto-stops at session end time.

3. **Photo Montage Generator** — Create stop-motion overlay image from run frames. Output: thumbnail (~200KB) + full-res (3-5MB for tabloid print).

4. **AWS Uploader** — Push both versions to S3 after each run. Update manifest.

### Server (Tesla GPU batch processing)

Same codebase, GPU-accelerated for processing recorded race videos much faster than real-time.

- 4× Tesla GPUs process video files in parallel
- NVDEC decode + CUDA frame differencing + parallel montage generation
- 10-50× faster than real-time
- Outputs to same S3/AWS structure, appears on skiframes.com

### skiframes.com (AWS)

**Public access — no login for athletes/parents:**
- Landing page: "Today's Sessions" button
- Browse by date
- Filter: Training vs Race
- Filter: U10 / U12 / U14 / Scored / Masters
- Thumbnail grid, click for full size

**Coach View (password protected):**
- Same browsing + delete button + print button

**Display Mode:**
- Auto-refreshing gallery for TV/projector in race lodge

### Race Lodge (Mac mini)

- Mac mini with TV screen + video projector
- Brother printer connected to Mac mini
- Runs website in display mode on big screen
- Printing: click print → download full-res → print locally (no special bridge needed)

## Tech Stack

- **Edge:** Python (Flask/FastAPI), OpenCV, ffmpeg, ImageMagick, boto3
- **AWS:** S3, CloudFront, static site or Lambda for dynamic bits
- **Lodge:** Mac mini + Brother printer, just runs website in browser
- **Auth:** Public for browsing, coach password only for delete/print

## Hardware

- Cameras: Reolink RLC-823S2 (2K/30fps), Axis Q6135-LE PTZ (1080p/60fps)
- Edge: Seeed Studio J40 (Jetson Orin NX) for prototype, MS-01 for production
- Server: 4× Tesla GPUs (V100/T4) for batch processing
- Lodge: Mac mini + TV + projector + Brother MFC-J5855DW printer (tabloid 11x17)
- Clients: Coach phones for calibration (5G), all phones for review site

**Note:** Use pure CPU ffmpeg decode on edge to start — GPU encoding has quality issues with fast skiers. Server mode uses GPU decode/processing since it's batch (quality verified after).

## Build Order

**Phase 1: Edge prototype (J40 + 1 camera)**
1. Calibration UI — mobile-first, critical for race day
2. Detection engine — port from run-detection, CPU decode
3. Montage generator — port from run-detection
4. AWS uploader

**Phase 2: skiframes.com**
5. Public browse — date picker, training/race filter, thumbnail grid
6. Coach view — password protected, delete/print
7. Display mode (TV/projector)

**Phase 3: Server batch processing**
8. GPU-accelerated detection + montages
9. Multi-GPU parallel processing

## Key Points

- START zone at top of frame, END zone at bottom, gates in between
- Upload everything automatically, coaches delete the bad ones
- Public site — no login for athletes/parents, just browse and view
- Filter by Training vs Race, and by group
- Thumbnails for fast review over 5G, full-res only for printing
- Same codebase runs on edge (real-time) and server (batch GPU)
- Ron LeMaster-inspired photo montages

## Security

Camera creds and AWS keys in `credentials.local` (gitignored). Never commit.

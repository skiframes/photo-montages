#!/usr/bin/env python3
"""
Extract frames from race videos for ski detection training data.
Samples frames where skiers are likely visible.
"""

import cv2
import os
from pathlib import Path
import random

def extract_frames(video_path: str, output_dir: str, frames_per_video: int = 5,
                   start_pct: float = 0.6, end_pct: float = 0.95):
    """Extract frames from video, focusing on portion where skier is visible."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        return 0

    # Calculate frame range (skier usually visible in latter portion)
    start_frame = int(total_frames * start_pct)
    end_frame = int(total_frames * end_pct)

    # Sample evenly across the range
    frame_indices = []
    step = (end_frame - start_frame) // (frames_per_video + 1)
    for i in range(frames_per_video):
        frame_indices.append(start_frame + step * (i + 1))

    video_name = Path(video_path).stem
    extracted = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"{video_name}_frame{idx:04d}.jpg")
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1

    cap.release()
    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract frames for ski detection training')
    parser.add_argument('--input-dir', default='output', help='Directory with video subdirs')
    parser.add_argument('--output-dir', default='training_data/ski_frames', help='Output directory')
    parser.add_argument('--max-videos', type=int, default=50, help='Max videos to process')
    parser.add_argument('--frames-per-video', type=int, default=4, help='Frames per video')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.mp4'):
                full_path = os.path.join(root, f)
                # Check file size > 500KB
                if os.path.getsize(full_path) > 500000:
                    video_files.append(full_path)

    print(f"Found {len(video_files)} valid video files")

    # Shuffle and limit
    random.shuffle(video_files)
    video_files = video_files[:args.max_videos]

    total_frames = 0
    for i, video_path in enumerate(video_files):
        print(f"[{i+1}/{len(video_files)}] Processing: {Path(video_path).name}")
        count = extract_frames(video_path, str(output_dir), args.frames_per_video)
        total_frames += count

    print(f"\nDone! Extracted {total_frames} frames to {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Upload frames to Roboflow (https://roboflow.com) or CVAT")
    print(f"2. Draw bounding boxes around each ski (label: 'ski')")
    print(f"3. Export in YOLO format")
    print(f"4. Train with: yolo detect train data=dataset.yaml model=yolov8m.pt epochs=100")


if __name__ == "__main__":
    main()

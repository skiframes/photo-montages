#!/usr/bin/env python3
"""
AWS Uploader - Uploads montages to S3 for skiframes.com.

Uploads thumbnails and full-res images to S3, updates manifest,
and makes them available via CloudFront.

S3 Structure:
    s3://skiframes/
      sessions/
        2026-02-02_0900_u14_training/
          manifest.json
          thumbnails/
            run_001_093023_thumb.jpg
          fullres/
            run_001_093023_full.jpg
"""

import os
import json
import boto3
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict
from botocore.exceptions import ClientError


# Default S3 bucket
DEFAULT_BUCKET = "skiframes"
DEFAULT_REGION = "us-east-1"


@dataclass
class UploadedRun:
    """Metadata for an uploaded run."""
    id: str
    run_number: int
    timestamp: str
    camera: str
    thumbnail_url: str
    fullres_url: str
    deleted: bool = False


@dataclass
class SessionManifest:
    """Session manifest for skiframes.com."""
    session_id: str
    session_type: str  # 'training' or 'race'
    group: str         # 'U10', 'U12', 'U14', 'Scored', 'Masters'
    start_time: str
    end_time: str
    runs: List[UploadedRun]


class S3Uploader:
    """
    Handles uploading montages to S3 and managing manifests.
    """

    def __init__(self, bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION,
                 cloudfront_domain: Optional[str] = None):
        self.bucket = bucket
        self.region = region
        self.cloudfront_domain = cloudfront_domain or f"{bucket}.s3.{region}.amazonaws.com"

        # Initialize S3 client
        self.s3 = boto3.client('s3', region_name=region)

        print(f"S3 Uploader initialized")
        print(f"  Bucket: {bucket}")
        print(f"  CloudFront: {self.cloudfront_domain}")

    def upload_file(self, local_path: str, s3_key: str, content_type: str = "image/jpeg") -> str:
        """
        Upload a file to S3.

        Returns the public URL.
        """
        try:
            self.s3.upload_file(
                local_path,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'CacheControl': 'max-age=31536000',  # 1 year cache
                }
            )
            url = f"https://{self.cloudfront_domain}/{s3_key}"
            return url
        except ClientError as e:
            print(f"  ERROR uploading {local_path}: {e}")
            raise

    def upload_montage(self, session_id: str, run_number: int, timestamp: datetime,
                       camera_id: str, thumbnail_path: str, fullres_path: str) -> UploadedRun:
        """
        Upload a montage (thumbnail + full-res) to S3.

        Returns UploadedRun metadata.
        """
        time_str = timestamp.strftime("%H%M%S")
        run_id = f"run_{run_number:03d}_{time_str}"

        # S3 keys
        thumb_key = f"sessions/{session_id}/thumbnails/{run_id}_thumb.jpg"
        full_key = f"sessions/{session_id}/fullres/{run_id}_full.jpg"

        print(f"  Uploading run {run_number}...")

        # Upload files
        thumb_url = self.upload_file(thumbnail_path, thumb_key)
        full_url = self.upload_file(fullres_path, full_key)

        thumb_size = os.path.getsize(thumbnail_path)
        full_size = os.path.getsize(fullres_path)
        print(f"    Thumbnail: {thumb_size/1024:.0f} KB → {thumb_key}")
        print(f"    Full-res: {full_size/1024/1024:.1f} MB → {full_key}")

        return UploadedRun(
            id=run_id,
            run_number=run_number,
            timestamp=timestamp.isoformat(),
            camera=camera_id,
            thumbnail_url=thumb_url,
            fullres_url=full_url,
        )

    def get_manifest(self, session_id: str) -> Optional[SessionManifest]:
        """Get existing manifest from S3."""
        key = f"sessions/{session_id}/manifest.json"

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))

            runs = [UploadedRun(**r) for r in data.get('runs', [])]
            return SessionManifest(
                session_id=data['session_id'],
                session_type=data['session_type'],
                group=data['group'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                runs=runs,
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def update_manifest(self, manifest: SessionManifest):
        """Upload/update manifest to S3."""
        key = f"sessions/{manifest.session_id}/manifest.json"

        data = {
            'session_id': manifest.session_id,
            'session_type': manifest.session_type,
            'group': manifest.group,
            'start_time': manifest.start_time,
            'end_time': manifest.end_time,
            'runs': [asdict(r) for r in manifest.runs],
        }

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType='application/json',
            CacheControl='max-age=60',  # Short cache for manifest
        )
        print(f"  Updated manifest: {key}")

    def upload_session(self, session_dir: str, session_config: dict) -> SessionManifest:
        """
        Upload all montages from a session directory.

        Args:
            session_dir: Local directory containing thumbnails/ and fullres/
            session_config: Session config dict with metadata

        Returns:
            SessionManifest with all uploaded runs
        """
        session_id = session_config['session_id']
        print(f"\nUploading session: {session_id}")

        # Get or create manifest
        manifest = self.get_manifest(session_id)
        if manifest is None:
            manifest = SessionManifest(
                session_id=session_id,
                session_type=session_config.get('session_type', 'training'),
                group=session_config.get('group', 'Unknown'),
                start_time=session_config.get('session_start_time', datetime.now().isoformat()),
                end_time=session_config.get('session_end_time', datetime.now().isoformat()),
                runs=[],
            )

        # Find all thumbnails
        thumb_dir = Path(session_dir) / "thumbnails"
        if not thumb_dir.exists():
            print(f"  No thumbnails directory found")
            return manifest

        existing_ids = {r.id for r in manifest.runs}

        for thumb_path in sorted(thumb_dir.glob("*_thumb.jpg")):
            # Parse run info from filename: run_001_093023_thumb.jpg
            parts = thumb_path.stem.replace("_thumb", "").split("_")
            if len(parts) >= 3:
                run_number = int(parts[1])
                time_str = parts[2]

                run_id = f"run_{run_number:03d}_{time_str}"
                if run_id in existing_ids:
                    print(f"  Skipping {run_id} (already uploaded)")
                    continue

                # Find corresponding full-res
                full_path = Path(session_dir) / "fullres" / f"{run_id}_full.jpg"
                if not full_path.exists():
                    print(f"  WARNING: No full-res for {run_id}")
                    continue

                # Parse timestamp
                try:
                    # Combine with session date
                    session_date = session_id.split("_")[0]  # "2026-02-02"
                    timestamp = datetime.strptime(f"{session_date}_{time_str}", "%Y-%m-%d_%H%M%S")
                except:
                    timestamp = datetime.now()

                # Upload
                uploaded = self.upload_montage(
                    session_id=session_id,
                    run_number=run_number,
                    timestamp=timestamp,
                    camera_id=session_config.get('camera_id', 'unknown'),
                    thumbnail_path=str(thumb_path),
                    fullres_path=str(full_path),
                )
                manifest.runs.append(uploaded)

        # Sort runs by number
        manifest.runs.sort(key=lambda r: r.run_number)

        # Update manifest
        self.update_manifest(manifest)

        print(f"  Total runs uploaded: {len(manifest.runs)}")
        return manifest

    def list_sessions(self, date_filter: Optional[str] = None) -> List[str]:
        """List all sessions, optionally filtered by date (YYYY-MM-DD)."""
        prefix = "sessions/"
        if date_filter:
            prefix = f"sessions/{date_filter}"

        sessions = set()
        paginator = self.s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
            for prefix_obj in page.get('CommonPrefixes', []):
                # Extract session ID from prefix
                session_id = prefix_obj['Prefix'].replace('sessions/', '').rstrip('/')
                sessions.add(session_id)

        return sorted(sessions)

    def delete_run(self, session_id: str, run_id: str, soft_delete: bool = True):
        """
        Delete a run from a session.

        Args:
            session_id: Session ID
            run_id: Run ID to delete
            soft_delete: If True, just mark as deleted in manifest. If False, remove from S3.
        """
        manifest = self.get_manifest(session_id)
        if not manifest:
            print(f"  Session not found: {session_id}")
            return

        # Find run
        for run in manifest.runs:
            if run.id == run_id:
                if soft_delete:
                    run.deleted = True
                    print(f"  Marked as deleted: {run_id}")
                else:
                    # Actually delete from S3
                    thumb_key = f"sessions/{session_id}/thumbnails/{run_id}_thumb.jpg"
                    full_key = f"sessions/{session_id}/fullres/{run_id}_full.jpg"

                    try:
                        self.s3.delete_object(Bucket=self.bucket, Key=thumb_key)
                        self.s3.delete_object(Bucket=self.bucket, Key=full_key)
                        manifest.runs.remove(run)
                        print(f"  Deleted from S3: {run_id}")
                    except ClientError as e:
                        print(f"  ERROR deleting {run_id}: {e}")
                        return

                self.update_manifest(manifest)
                return

        print(f"  Run not found: {run_id}")


def main():
    """CLI for S3 uploader."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload montages to S3")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a session')
    upload_parser.add_argument('session_dir', help='Local session directory')
    upload_parser.add_argument('-c', '--config', required=True, help='Session config JSON')
    upload_parser.add_argument('--bucket', default=DEFAULT_BUCKET, help='S3 bucket')

    # List command
    list_parser = subparsers.add_parser('list', help='List sessions')
    list_parser.add_argument('--date', help='Filter by date (YYYY-MM-DD)')
    list_parser.add_argument('--bucket', default=DEFAULT_BUCKET, help='S3 bucket')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a run')
    delete_parser.add_argument('session_id', help='Session ID')
    delete_parser.add_argument('run_id', help='Run ID')
    delete_parser.add_argument('--hard', action='store_true', help='Hard delete (remove from S3)')
    delete_parser.add_argument('--bucket', default=DEFAULT_BUCKET, help='S3 bucket')

    args = parser.parse_args()

    if args.command == 'upload':
        with open(args.config) as f:
            config = json.load(f)
        uploader = S3Uploader(bucket=args.bucket)
        uploader.upload_session(args.session_dir, config)

    elif args.command == 'list':
        uploader = S3Uploader(bucket=args.bucket)
        sessions = uploader.list_sessions(args.date)
        print(f"Sessions ({len(sessions)}):")
        for s in sessions:
            print(f"  {s}")

    elif args.command == 'delete':
        uploader = S3Uploader(bucket=args.bucket)
        uploader.delete_run(args.session_id, args.run_id, soft_delete=not args.hard)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

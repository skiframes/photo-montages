#!/usr/bin/env python3
"""
Backfill CLIP embeddings for existing montages.

Downloads montage images from S3/CloudFront, computes CLIP embeddings,
and updates the manifest with embedding vectors.

Usage:
    python backfill_embeddings.py <event_id>
    python backfill_embeddings.py 2026-02-17_masters_gate_training

Requires:
    pip install sentence-transformers Pillow boto3 requests
"""

import argparse
import json
import os
import sys
import tempfile
import requests
import boto3

# Add edge directory to path so we can import embedder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'edge'))
import embedder

MEDIA_BASE = "https://media.skiframes.com"
S3_BUCKET = "avillachlab-netm"


def fetch_manifest(event_id):
    """Download manifest.json from CloudFront."""
    url = f"{MEDIA_BASE}/events/{event_id}/manifest.json"
    print(f"Fetching manifest: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def download_image(event_id, relative_path, tmp_dir):
    """Download a montage image from CloudFront to a temp file."""
    url = f"{MEDIA_BASE}/events/{event_id}/{relative_path}"
    local_path = os.path.join(tmp_dir, os.path.basename(relative_path))
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path


def upload_manifest(event_id, manifest):
    """Upload updated manifest.json to S3."""
    s3 = boto3.client('s3')
    key = f"events/{event_id}/manifest.json"
    print(f"Uploading manifest to s3://{S3_BUCKET}/{key}")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(manifest, indent=2),
        ContentType='application/json',
    )
    print("Manifest uploaded successfully.")


def backfill_event(event_id, dry_run=False, force=False):
    """Compute embeddings for all runs in an event."""
    manifest = fetch_manifest(event_id)
    runs = manifest.get('runs', [])

    if not runs:
        print("No runs found in manifest.")
        return

    print(f"Found {len(runs)} runs in event '{event_id}'")

    # Check how many already have embeddings
    with_embeddings = sum(1 for r in runs if r.get('embedding'))
    without_embeddings = len(runs) - with_embeddings
    print(f"  With embeddings: {with_embeddings}")
    print(f"  Without embeddings: {without_embeddings}")

    if without_embeddings == 0 and not force:
        print("All runs already have embeddings. Use --force to recompute.")
        return

    # Ensure model is loaded before processing
    print("\nLoading CLIP model...")
    embedder._get_model()
    print("Model ready.\n")

    updated = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for run in runs:
            run_num = run['run_number']
            has_embedding = run.get('embedding') is not None

            if has_embedding and not force:
                print(f"  Run {run_num}: already has embedding, skipping")
                continue

            # Get the first variant's fullres image (highest detail)
            variants = run.get('variants', {})
            if not variants:
                print(f"  Run {run_num}: no variants, skipping")
                continue

            # Use highest FPS variant for most detail
            variant_key = sorted(variants.keys(), key=lambda k: float(k.replace('fps', '')))[-1]
            fullres_path = variants[variant_key].get('fullres')

            if not fullres_path:
                print(f"  Run {run_num}: no fullres path in variant {variant_key}, skipping")
                continue

            print(f"  Run {run_num}: downloading {fullres_path}...", end=' ')
            try:
                local_path = download_image(event_id, fullres_path, tmp_dir)
                embedding = embedder.embed_image(local_path)

                if embedding:
                    run['embedding'] = embedding
                    updated += 1
                    print(f"OK ({len(embedding)} dims)")
                else:
                    print("FAILED (no embedding returned)")

                # Clean up downloaded image
                os.remove(local_path)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    print(f"\nComputed {updated} embeddings out of {len(runs)} runs.")

    if updated > 0:
        # Always save a local copy
        local_path = f"manifest_{event_id}.json"
        with open(local_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved local copy: {local_path}")

        if not dry_run:
            try:
                upload_manifest(event_id, manifest)
                print(f"\nDone! Embeddings backfilled for event '{event_id}'.")
                print(f"View at: https://skiframes.com/event.html?event={event_id}")
            except Exception as e:
                print(f"\nS3 upload failed: {e}")
                print(f"Local manifest saved to: {local_path}")
                print(f"Upload manually with:")
                print(f"  aws s3 cp {local_path} s3://avillachlab-netm/events/{event_id}/manifest.json --content-type application/json")
        else:
            print("\n[DRY RUN] No changes uploaded.")
    else:
        print("\nNo new embeddings computed.")


def main():
    parser = argparse.ArgumentParser(description='Backfill CLIP embeddings for existing montages')
    parser.add_argument('event_id', help='Event ID (e.g., 2026-02-17_masters_gate_training)')
    parser.add_argument('--dry-run', action='store_true', help='Compute embeddings but don\'t upload')
    parser.add_argument('--force', action='store_true', help='Recompute even if embeddings exist')
    args = parser.parse_args()

    backfill_event(args.event_id, dry_run=args.dry_run, force=args.force)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Embedder - Compute CLIP image embeddings for athlete re-identification.

Uses sentence-transformers with the clip-ViT-B-32 model to generate 512-dimensional
embedding vectors. These embeddings capture visual features (clothing color, body
position, equipment) that enable grouping runs by the same athlete on the web gallery.

Key insight: CLIP works best on tight crops of the athlete, not the full montage.
Full montages are 90% snow/mountain background which dominates the embedding and
makes all runs look similar. Instead, we extract a single frame from the middle of
the run and crop tightly around the athlete's position for distinctive features.

The model is loaded lazily on first use (~5s on CPU) and cached for subsequent calls.
Each embedding computation takes ~300ms on CPU (i9) or ~700ms on Jetson.
"""

import os
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# Global model instance (lazy loaded)
_model = None
_available = None  # None = not checked yet, True/False after check


def is_available() -> bool:
    """Check if sentence-transformers and PIL are available."""
    global _available
    if _available is None:
        try:
            import sentence_transformers  # noqa: F401
            from PIL import Image  # noqa: F401
            _available = True
        except ImportError:
            _available = False
            logger.info("Embedder not available: sentence-transformers or Pillow not installed")
    return _available


def _get_model():
    """Lazy-load the CLIP vision model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading CLIP model (clip-ViT-B-32)...")
        _model = SentenceTransformer('clip-ViT-B-32')
        logger.info("CLIP model loaded")
    return _model


def _encode_pil_image(img) -> Optional[list]:
    """Encode a PIL Image to a 512-dim embedding vector."""
    try:
        model = _get_model()
        embedding = model.encode(img)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        return None


def embed_frames(frames, crop_region=None) -> Optional[list]:
    """
    Compute a 512-dim CLIP embedding from a single frame of the run.

    Takes the middle frame (where athlete is most visible/centered) and crops
    to the athlete area for maximum clothing detail. This gives much better
    re-identification than the full montage composite.

    Args:
        frames: List of numpy arrays (raw video frames from the run).
        crop_region: Optional dict with x, y, w, h for the montage crop zone.
                     If provided, crops the frame to this region first.

    Returns:
        List of 512 floats (the embedding vector), or None if computation fails.
    """
    if not is_available():
        return None

    if not frames or len(frames) == 0:
        return None

    try:
        import numpy as np
        from PIL import Image

        # Take the middle frame — athlete is typically most centered/visible
        mid_idx = len(frames) // 2
        frame = frames[mid_idx].copy()

        # Apply crop region if provided (same crop as montage uses)
        if crop_region:
            x = crop_region.get('x', 0)
            y = crop_region.get('y', 0)
            w = crop_region.get('w', frame.shape[1])
            h = crop_region.get('h', frame.shape[0])
            frame = frame[y:y+h, x:x+w]

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = frame[:, :, ::-1] if len(frame.shape) == 3 else frame
        img = Image.fromarray(frame_rgb)

        return _encode_pil_image(img)

    except Exception as e:
        logger.error(f"Frame embedding failed: {e}")
        return None


def embed_image(image_path: str) -> Optional[list]:
    """
    Compute a 512-dimensional CLIP embedding from a montage composite image.

    For backfill use when raw frames aren't available. Finds the largest
    dark region (the athlete figures) in the image and crops tightly around
    it to minimize snow background that would make all embeddings look similar.

    Args:
        image_path: Path to the montage image file (JPEG).

    Returns:
        List of 512 floats (the embedding vector), or None if computation fails.
    """
    if not is_available():
        return None

    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return None

    try:
        from PIL import Image
        import numpy as np
        import cv2

        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)
        iw, ih = img.size

        # Find the athlete: dark pixels against white snow background
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        dark_mask = (gray < 110).astype(np.uint8) * 255

        # Clean up noise with morphological operations
        kernel = np.ones((7, 7), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        # Find the largest dark contour (the main athlete blob)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            # Add 25% padding around the athlete
            pad = max(w, h) // 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(iw, x + w + pad)
            y2 = min(ih, y + h + pad)
            img = img.crop((x1, y1, x2, y2))

        return _encode_pil_image(img)

    except Exception as e:
        logger.error(f"Embedding computation failed for {image_path}: {e}")
        return None

#!/usr/bin/env python3
"""
Embedder - Compute CLIP image embeddings for athlete re-identification.

Uses sentence-transformers with the clip-ViT-B-32 model to generate 512-dimensional
embedding vectors from montage images. These embeddings capture visual features
(clothing color, body position, equipment) that enable grouping runs by the same
athlete on the web gallery.

The model is loaded lazily on first use (~5s on CPU) and cached for subsequent calls.
Each embedding computation takes ~300ms on CPU (i9) or ~700ms on Jetson.
"""

import os
import logging
from typing import Optional

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


def embed_image(image_path: str) -> Optional[list]:
    """
    Compute a 512-dimensional CLIP embedding from a montage image.

    Uses the full-resolution montage image for maximum clothing detail.
    CLIP internally resizes to 224x224 but benefits from higher-res input
    for better detail extraction.

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

        model = _get_model()
        img = Image.open(image_path).convert('RGB')
        embedding = model.encode(img)

        # Convert numpy array to Python list for JSON serialization
        return embedding.tolist()

    except Exception as e:
        logger.error(f"Embedding computation failed for {image_path}: {e}")
        return None

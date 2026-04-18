"""
Stage 1 – Multimodal Ingestion: image_processor.py

Processes a standalone image file into a single Chunk object.

Pipeline:
  1. Read image dimensions with Pillow.
  2. Caption the image via a pluggable captioner (Ollama or Anthropic).
  3. Produce one Chunk with modality="image".

Reuses the same `captioning` config section as slide_processor.py:
    captioning:
      provider: ollama   # or: anthropic
      ollama_model: llava:13b
      ...

Public API:
    chunks = process_image(image_path, config)
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.ingestion.captioners import build_captioner
from backend.schemas import Chunk

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dimensions(image_path: Path) -> tuple[int, int]:
    """Return (width, height) for the image using Pillow."""
    from PIL import Image

    with Image.open(image_path) as img:
        return img.width, img.height


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_image(image_path: str | Path, config: dict) -> list[Chunk]:
    """
    Process a single image file into a list containing one Chunk.

    Args:
        image_path: Path to the source image file.
        config:     Parsed contents of config/llm.yaml.

    Returns:
        A list with one Chunk object with modality="image".
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{image_path.suffix}'. "
            f"image_processor handles: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    source_id = image_path.stem
    captioner = build_captioner(config)

    width, height = _get_dimensions(image_path)

    caption = ""
    try:
        caption = captioner.caption(image_path)
    except Exception as exc:
        logger.warning(f"[{source_id}] Captioning failed: {exc}")

    text = caption if caption else f"[Image: {image_path.name}]"

    chunk = Chunk(
        id=f"{source_id}_image_0",
        text=text.strip(),
        source_id=source_id,
        modality="image",
        metadata={
            "source_file": str(image_path),
            "width": width,
            "height": height,
        },
    )

    logger.info(f"[{source_id}] Done — 1 image chunk produced")
    return [chunk]

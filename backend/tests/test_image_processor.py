"""
Tests for backend/ingestion/image_processor.py

Strategy:
  - Uses real synthetic images created with Pillow (no external files needed).
  - The captioner is always mocked — no Ollama/Anthropic calls during tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from backend.ingestion.image_processor import _get_dimensions, process_image
from backend.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(
    tmp_path: Path,
    width: int = 200,
    height: int = 100,
    name: str = "test.jpg",
) -> Path:
    """Create a minimal JPEG image using Pillow."""
    img_path = tmp_path / name
    img = PILImage.new("RGB", (width, height), color=(128, 64, 32))
    img.save(str(img_path), "JPEG")
    return img_path


@pytest.fixture
def stub_captioner() -> MagicMock:
    captioner = MagicMock()
    captioner.caption.return_value = "A diagram showing cellular respiration."
    return captioner


@pytest.fixture
def base_config() -> dict:
    return {
        "captioning": {
            "provider": "ollama",
            "ollama_model": "llava:13b",
            "ollama_base_url": "http://localhost:11434",
            "anthropic_model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
        },
    }


# ---------------------------------------------------------------------------
# _get_dimensions
# ---------------------------------------------------------------------------

class TestGetDimensions:
    def test_returns_correct_width_height(self, tmp_path):
        img_path = _make_image(tmp_path, width=320, height=240)
        w, h = _get_dimensions(img_path)
        assert w == 320
        assert h == 240

    def test_portrait_image(self, tmp_path):
        img_path = _make_image(tmp_path, width=100, height=300)
        w, h = _get_dimensions(img_path)
        assert w == 100
        assert h == 300

    def test_square_image(self, tmp_path):
        img_path = _make_image(tmp_path, width=64, height=64)
        w, h = _get_dimensions(img_path)
        assert w == h == 64


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------

class TestProcessImage:
    def test_returns_list_with_one_chunk(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)

    def test_modality_is_image(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert chunks[0].modality == "image"

    def test_caption_used_as_text(self, tmp_path, base_config, stub_captioner):
        stub_captioner.caption.return_value = "A cell diagram."
        img_path = _make_image(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert chunks[0].text == "A cell diagram."

    def test_metadata_contains_dimensions(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path, width=640, height=480)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert chunks[0].metadata["width"] == 640
        assert chunks[0].metadata["height"] == 480

    def test_metadata_contains_source_file(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path, name="lecture_diagram.jpg")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert "source_file" in chunks[0].metadata

    def test_chunk_id_format(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path, name="diagram.jpg")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert chunks[0].id == "diagram_image_0"

    def test_source_id_matches_stem(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path, name="my_figure.jpg")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert chunks[0].source_id == "my_figure"

    def test_captioner_failure_uses_fallback_text(self, tmp_path, base_config):
        failing_captioner = MagicMock()
        failing_captioner.caption.side_effect = RuntimeError("Ollama down")

        img_path = _make_image(tmp_path, name="figure.jpg")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: failing_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert len(chunks) == 1
        assert "figure.jpg" in chunks[0].text

    def test_captioner_called_once(self, tmp_path, base_config, stub_captioner):
        img_path = _make_image(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            process_image(img_path, base_config)

        stub_captioner.caption.assert_called_once()

    def test_file_not_found_raises(self, tmp_path, base_config):
        with pytest.raises(FileNotFoundError):
            process_image(tmp_path / "nonexistent.jpg", base_config)

    def test_unsupported_extension_raises(self, tmp_path, base_config):
        bad_file = tmp_path / "image.bmp"
        bad_file.write_bytes(b"fake bmp data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            process_image(bad_file, base_config)

    def test_png_supported(self, tmp_path, base_config, stub_captioner):
        img = PILImage.new("RGB", (100, 100), color=(0, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(str(img_path), "PNG")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert len(chunks) == 1
        assert chunks[0].modality == "image"

    def test_jpeg_extension_accepted(self, tmp_path, base_config, stub_captioner):
        img = PILImage.new("RGB", (50, 50), color=(255, 0, 0))
        img_path = tmp_path / "photo.jpeg"
        img.save(str(img_path), "JPEG")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "backend.ingestion.image_processor.build_captioner",
                lambda _: stub_captioner,
            )
            chunks = process_image(img_path, base_config)

        assert len(chunks) == 1

"""
Tests for backend/ingestion/video_transcriber.py

Strategy:
  - Schemas and pure logic (chunking) are tested directly.
  - External I/O (ffmpeg, Whisper, Ollama/Anthropic) is mocked.
  - Keyframe extraction is tested against a real synthetic video built with OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.ingestion.captioners import (
    AnthropicCaptioner,
    OllamaCaptioner,
    build_captioner,
)
from backend.ingestion.video_transcriber import (
    _build_chunks,
    _extract_keyframes,
)
from backend.schemas import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> dict:
    return {
        "whisper": {"model_size": "tiny", "device": "cpu", "compute_type": "int8"},
        "captioning": {
            "provider": "ollama",
            "ollama_model": "llama3.2-vision",
            "ollama_base_url": "http://localhost:11434",
            "anthropic_model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
        },
        "keyframe": {"ssim_threshold": 0.85, "sample_fps": 1},
    }


@pytest.fixture
def stub_captioner() -> MagicMock:
    captioner = MagicMock()
    captioner.caption.return_value = "A lecture slide showing enzyme kinetics."
    return captioner


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """
    30 frames of black followed by 30 frames of white at 10 fps.
    Guarantees at least one keyframe when SSIM threshold < 1.0.
    """
    video_path = tmp_path / "lecture.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10, (160, 120))

    black = np.zeros((120, 160, 3), dtype=np.uint8)
    white = np.full((120, 160, 3), 255, dtype=np.uint8)

    for _ in range(30):
        writer.write(black)
    for _ in range(30):
        writer.write(white)

    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# Captioner factory
# ---------------------------------------------------------------------------

class TestBuildCaptioner:
    def test_returns_ollama_captioner(self, base_config):
        captioner = build_captioner(base_config)
        assert isinstance(captioner, OllamaCaptioner)

    def test_returns_anthropic_captioner(self, base_config):
        base_config["captioning"]["provider"] = "anthropic"
        captioner = build_captioner(base_config)
        assert isinstance(captioner, AnthropicCaptioner)

    def test_unknown_provider_raises(self, base_config):
        base_config["captioning"]["provider"] = "openai"
        with pytest.raises(ValueError, match="Unknown captioning provider"):
            build_captioner(base_config)


# ---------------------------------------------------------------------------
# Keyframe extraction (real OpenCV, synthetic video)
# ---------------------------------------------------------------------------

class TestExtractKeyframes:
    def test_detects_scene_change(self, synthetic_video, tmp_path):
        kf_dir = tmp_path / "keyframes"
        kf_dir.mkdir()
        keyframes = _extract_keyframes(synthetic_video, kf_dir, threshold=0.85, sample_fps=1)
        assert len(keyframes) >= 1

    def test_keyframe_files_exist(self, synthetic_video, tmp_path):
        kf_dir = tmp_path / "keyframes"
        kf_dir.mkdir()
        keyframes = _extract_keyframes(synthetic_video, kf_dir, threshold=0.85, sample_fps=1)
        for kf in keyframes:
            assert Path(kf["path"]).exists()

    def test_keyframe_has_timestamp(self, synthetic_video, tmp_path):
        kf_dir = tmp_path / "keyframes"
        kf_dir.mkdir()
        keyframes = _extract_keyframes(synthetic_video, kf_dir, threshold=0.85, sample_fps=1)
        for kf in keyframes:
            assert isinstance(kf["timestamp"], float)
            assert kf["timestamp"] >= 0.0

    def test_no_keyframe_on_static_video(self, tmp_path):
        """A perfectly static video should yield no keyframes."""
        video_path = tmp_path / "static.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (160, 120))
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        for _ in range(20):
            writer.write(frame)
        writer.release()

        kf_dir = tmp_path / "keyframes"
        kf_dir.mkdir()
        keyframes = _extract_keyframes(video_path, kf_dir, threshold=0.85, sample_fps=1)
        assert len(keyframes) == 0


# ---------------------------------------------------------------------------
# Chunk assembly (no I/O — captioner is mocked)
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_returns_list_of_chunks(self, stub_captioner):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Enzymes are biological catalysts."},
            {"start": 5.0, "end": 10.0, "text": "They lower activation energy."},
        ]
        chunks = _build_chunks("lecture_01", segments, [], stub_captioner)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_modality_is_video(self, stub_captioner):
        segments = [{"start": 0.0, "end": 3.0, "text": "Hello world."}]
        chunks = _build_chunks("vid", segments, [], stub_captioner)
        assert all(c.modality == "video" for c in chunks)

    def test_chunk_ids_are_unique(self, stub_captioner):
        segments = [
            {"start": 0.0, "end": 4.0, "text": "Part one."},
            {"start": 4.0, "end": 8.0, "text": "Part two."},
        ]
        chunks = _build_chunks("vid", segments, [], stub_captioner)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_include_source_id(self, stub_captioner):
        segments = [{"start": 0.0, "end": 5.0, "text": "content"}]
        chunks = _build_chunks("lecture_42", segments, [], stub_captioner)
        assert all("lecture_42" in c.id for c in chunks)

    def test_metadata_keys_present(self, stub_captioner):
        segments = [{"start": 0.0, "end": 5.0, "text": "content"}]
        chunks = _build_chunks("src", segments, [], stub_captioner)
        for c in chunks:
            assert "start_time" in c.metadata
            assert "end_time" in c.metadata
            assert "keyframe_path" in c.metadata

    def test_keyframe_caption_injected(self, tmp_path, stub_captioner):
        """Caption from keyframe should appear in chunk text."""
        # Create a dummy keyframe image
        kf_path = tmp_path / "frame_0000030.jpg"
        cv2.imwrite(str(kf_path), np.zeros((10, 10, 3), dtype=np.uint8))

        segments = [
            {"start": 0.0, "end": 3.0, "text": "Before slide change."},
            {"start": 3.5, "end": 6.0, "text": "After slide change."},
        ]
        keyframes = [{"timestamp": 3.0, "path": kf_path}]
        stub_captioner.caption.return_value = "A diagram of enzyme kinetics."

        chunks = _build_chunks("src", segments, keyframes, stub_captioner)
        captions_in_text = [c for c in chunks if "[Visual:" in c.text]
        assert len(captions_in_text) >= 1

    def test_empty_segments_no_chunks(self, stub_captioner):
        chunks = _build_chunks("src", [], [], stub_captioner)
        assert chunks == []

    def test_no_keyframes_produces_single_chunk(self, stub_captioner):
        """Without keyframes, all segments collapse into one chunk."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "First."},
            {"start": 5.0, "end": 10.0, "text": "Second."},
            {"start": 10.0, "end": 15.0, "text": "Third."},
        ]
        chunks = _build_chunks("src", segments, [], stub_captioner)
        assert len(chunks) == 1

    def test_captioner_failure_is_handled(self, stub_captioner, tmp_path):
        """A captioner error should be swallowed — chunk still created without caption."""
        stub_captioner.caption.side_effect = RuntimeError("Ollama unavailable")
        kf_path = tmp_path / "frame.jpg"
        cv2.imwrite(str(kf_path), np.zeros((10, 10, 3), dtype=np.uint8))

        segments = [
            {"start": 0.0, "end": 2.0, "text": "Before."},
            {"start": 3.0, "end": 5.0, "text": "After."},
        ]
        keyframes = [{"timestamp": 2.5, "path": kf_path}]
        # Should not raise
        chunks = _build_chunks("src", segments, keyframes, stub_captioner)
        assert len(chunks) >= 1
        # No [Visual:] prefix since caption failed
        for c in chunks:
            if c.metadata.get("keyframe_path"):
                assert "[Visual:" not in c.text

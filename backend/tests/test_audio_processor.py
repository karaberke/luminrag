"""
Tests for backend/ingestion/audio_processor.py

Strategy:
  - _build_chunks is pure logic — tested directly with no mocks.
  - _transcribe and _to_wav rely on external tools (faster-whisper, ffmpeg) — mocked.
  - process_audio is tested end-to-end by patching _to_wav and _transcribe.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.ingestion.audio_processor import (
    _build_chunks,
    process_audio,
)
from backend.schemas import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> dict:
    return {
        "whisper": {"model_size": "tiny", "device": "cpu", "compute_type": "int8"},
        "audio_processor": {"chunk_window_sec": 60.0},
    }


def _make_segments(*ranges: tuple[float, float, str]) -> list[dict]:
    """Helper to build segment dicts from (start, end, text) tuples."""
    return [{"start": s, "end": e, "text": t} for s, e, t in ranges]


# ---------------------------------------------------------------------------
# _build_chunks — pure logic tests
# ---------------------------------------------------------------------------

class TestBuildChunks:
    def test_empty_segments_returns_empty(self):
        assert _build_chunks("src", [], 60.0) == []

    def test_single_segment_single_chunk(self):
        segs = _make_segments((0.0, 5.0, "Hello world."))
        chunks = _build_chunks("lecture", segs, 60.0)
        assert len(chunks) == 1

    def test_modality_is_audio(self):
        segs = _make_segments((0.0, 5.0, "Some speech."))
        chunks = _build_chunks("lecture", segs, 60.0)
        assert chunks[0].modality == "audio"

    def test_source_id_propagated(self):
        segs = _make_segments((0.0, 5.0, "Content."))
        chunks = _build_chunks("my_lecture", segs, 60.0)
        assert chunks[0].source_id == "my_lecture"

    def test_chunk_id_format(self):
        segs = _make_segments((0.0, 5.0, "Content."))
        chunks = _build_chunks("audio01", segs, 60.0)
        assert chunks[0].id == "audio01_chunk_0"

    def test_chunk_ids_are_sequential(self):
        segs = _make_segments(
            (0.0, 30.0, "First window."),
            (60.0, 90.0, "Second window."),
        )
        chunks = _build_chunks("src", segs, 60.0)
        assert len(chunks) == 2
        assert chunks[0].id == "src_chunk_0"
        assert chunks[1].id == "src_chunk_1"

    def test_all_segments_in_window_merged(self):
        segs = _make_segments(
            (0.0, 10.0, "First sentence."),
            (10.0, 20.0, "Second sentence."),
            (20.0, 30.0, "Third sentence."),
        )
        chunks = _build_chunks("src", segs, 60.0)
        assert len(chunks) == 1
        assert "First sentence." in chunks[0].text
        assert "Second sentence." in chunks[0].text
        assert "Third sentence." in chunks[0].text

    def test_segments_split_across_windows(self):
        segs = _make_segments(
            (0.0, 30.0, "Window one content."),
            (65.0, 95.0, "Window two content."),
        )
        chunks = _build_chunks("src", segs, 60.0)
        assert len(chunks) == 2

    def test_metadata_start_time(self):
        segs = _make_segments((0.0, 30.0, "Content."))
        chunks = _build_chunks("src", segs, 60.0)
        assert chunks[0].metadata["start_time"] == 0.0

    def test_metadata_end_time_is_last_segment_end(self):
        segs = _make_segments(
            (0.0, 10.0, "A."),
            (10.0, 25.0, "B."),
        )
        chunks = _build_chunks("src", segs, 60.0)
        assert chunks[0].metadata["end_time"] == 25.0

    def test_second_window_start_time(self):
        segs = _make_segments(
            (0.0, 30.0, "First."),
            (61.0, 80.0, "Second."),
        )
        chunks = _build_chunks("src", segs, 60.0)
        assert chunks[1].metadata["start_time"] == 60.0

    def test_three_windows(self):
        segs = _make_segments(
            (0.0, 10.0, "W1"),
            (65.0, 70.0, "W2"),
            (125.0, 130.0, "W3"),
        )
        chunks = _build_chunks("src", segs, 60.0)
        assert len(chunks) == 3

    def test_text_is_stripped(self):
        segs = _make_segments((0.0, 5.0, "  Spaced text.  "))
        chunks = _build_chunks("src", segs, 60.0)
        assert chunks[0].text == "Spaced text."

    def test_small_window_groups_correctly(self):
        segs = _make_segments(
            (0.0, 5.0, "A"),
            (8.0, 13.0, "B"),
            (15.0, 19.0, "C"),
        )
        chunks = _build_chunks("src", segs, 10.0)
        # 0–10: segments at 0 and 8
        # 10–20: segments at 15
        assert len(chunks) == 2
        assert "A" in chunks[0].text
        assert "B" in chunks[0].text
        assert "C" in chunks[1].text


# ---------------------------------------------------------------------------
# process_audio — integration (external I/O mocked)
# ---------------------------------------------------------------------------

class TestProcessAudio:
    def test_file_not_found_raises(self, tmp_path, base_config):
        with pytest.raises(FileNotFoundError):
            process_audio(tmp_path / "missing.mp3", base_config)

    def test_unsupported_extension_raises(self, tmp_path, base_config):
        bad = tmp_path / "audio.xyz"
        bad.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported file type"):
            process_audio(bad, base_config)

    def test_returns_chunks_list(self, tmp_path, base_config):
        audio = tmp_path / "lecture.mp3"
        audio.write_bytes(b"fake audio")

        fake_segments = [
            {"start": 0.0, "end": 30.0, "text": "Introduction to enzymes."},
            {"start": 30.0, "end": 55.0, "text": "Enzyme kinetics overview."},
        ]

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=fake_segments),
        ):
            chunks = process_audio(audio, base_config)

        assert isinstance(chunks, list)
        assert len(chunks) == 1  # both segments fit in a single 60s window
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_all_chunks_have_audio_modality(self, tmp_path, base_config):
        audio = tmp_path / "podcast.mp3"
        audio.write_bytes(b"fake audio")

        fake_segments = [
            {"start": 0.0, "end": 50.0, "text": "Part one."},
            {"start": 65.0, "end": 115.0, "text": "Part two."},
        ]

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=fake_segments),
        ):
            chunks = process_audio(audio, base_config)

        assert all(c.modality == "audio" for c in chunks)

    def test_source_id_is_stem(self, tmp_path, base_config):
        audio = tmp_path / "my_lecture.mp3"
        audio.write_bytes(b"fake audio")

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=[
                {"start": 0.0, "end": 5.0, "text": "Hello."}
            ]),
        ):
            chunks = process_audio(audio, base_config)

        assert all(c.source_id == "my_lecture" for c in chunks)

    def test_empty_transcription_returns_empty_list(self, tmp_path, base_config):
        audio = tmp_path / "silent.mp3"
        audio.write_bytes(b"fake audio")

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=[]),
        ):
            chunks = process_audio(audio, base_config)

        assert chunks == []

    def test_custom_window_size_respected(self, tmp_path):
        config = {
            "whisper": {"model_size": "tiny", "device": "cpu", "compute_type": "int8"},
            "audio_processor": {"chunk_window_sec": 30.0},
        }
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")

        fake_segments = [
            {"start": 0.0, "end": 20.0, "text": "Window one."},
            {"start": 35.0, "end": 55.0, "text": "Window two."},
        ]

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=fake_segments),
        ):
            chunks = process_audio(audio, config)

        # 30s window: segment at 0s → chunk 0; segment at 35s → chunk 1
        assert len(chunks) == 2

    def test_wav_extension_accepted(self, tmp_path, base_config):
        audio = tmp_path / "recording.wav"
        audio.write_bytes(b"fake wav")

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=[
                {"start": 0.0, "end": 10.0, "text": "Some speech."}
            ]),
        ):
            chunks = process_audio(audio, base_config)

        assert len(chunks) == 1

    def test_m4a_extension_accepted(self, tmp_path, base_config):
        audio = tmp_path / "voice.m4a"
        audio.write_bytes(b"fake m4a")

        with (
            patch("backend.ingestion.audio_processor._to_wav", return_value=tmp_path / "audio.wav"),
            patch("backend.ingestion.audio_processor._transcribe", return_value=[
                {"start": 0.0, "end": 5.0, "text": "Testing."}
            ]),
        ):
            chunks = process_audio(audio, base_config)

        assert len(chunks) == 1

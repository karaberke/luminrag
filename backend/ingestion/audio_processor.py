"""
Stage 1 – Multimodal Ingestion: audio_processor.py

Processes a raw audio file into a list of Chunk objects.

Pipeline:
  1. Convert to 16 kHz mono WAV with ffmpeg.
  2. Transcribe with faster-whisper → timestamped segments.
  3. Group segments into fixed-length time windows → one Chunk per window.

Reuses the same `whisper` config section as video_transcriber.py:
    whisper:
      model_size: small
      device: cpu
      compute_type: int8

Optional config section for window size:
    audio_processor:
      chunk_window_sec: 60   # seconds per chunk (default: 60)

Public API:
    chunks = process_audio(audio_path, config)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from backend.schemas import Chunk

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}


# ---------------------------------------------------------------------------
# Step 1: Convert to 16 kHz mono WAV
# ---------------------------------------------------------------------------

def _to_wav(audio_path: Path, tmp_dir: str) -> Path:
    """Convert any supported audio format to a 16 kHz mono WAV using ffmpeg."""
    wav_path = Path(tmp_dir) / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {audio_path}:\n{result.stderr.decode()}"
        )
    return wav_path


# ---------------------------------------------------------------------------
# Step 2: Whisper transcription
# ---------------------------------------------------------------------------

def _transcribe(audio_path: Path, cfg: dict) -> list[dict]:
    """
    Transcribe audio with faster-whisper.
    Returns list of {start: float, end: float, text: str}.
    """
    from faster_whisper import WhisperModel

    w = cfg["whisper"]
    model = WhisperModel(
        w["model_size"],
        device=w.get("device", "cpu"),
        compute_type=w.get("compute_type", "int8"),
    )
    segments, _ = model.transcribe(str(audio_path), beam_size=5)
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]


# ---------------------------------------------------------------------------
# Step 3: Group segments into fixed time windows
# ---------------------------------------------------------------------------

def _build_chunks(
    source_id: str,
    segments: list[dict],
    window_sec: float,
) -> list[Chunk]:
    """
    Group transcript segments into fixed time windows of `window_sec` seconds.
    Each window that contains at least one segment becomes one Chunk.

    Args:
        source_id:  Identifier for the source audio file (used in chunk IDs).
        segments:   List of {start, end, text} dicts from Whisper.
        window_sec: Duration of each time window in seconds.

    Returns:
        Ordered list of Chunk objects.
    """
    if not segments:
        return []

    chunks: list[Chunk] = []
    window_start = 0.0
    window_end = window_sec
    buffer: list[dict] = []

    for seg in segments:
        # Advance the window until this segment fits inside it
        while seg["start"] >= window_end:
            if buffer:
                text = " ".join(s["text"] for s in buffer)
                chunks.append(
                    Chunk(
                        id=f"{source_id}_chunk_{len(chunks)}",
                        text=text.strip(),
                        source_id=source_id,
                        modality="audio",
                        metadata={
                            "start_time": window_start,
                            "end_time": buffer[-1]["end"],
                        },
                    )
                )
                buffer = []
            window_start = window_end
            window_end += window_sec

        buffer.append(seg)

    # Flush the final (possibly partial) window
    if buffer:
        text = " ".join(s["text"] for s in buffer)
        chunks.append(
            Chunk(
                id=f"{source_id}_chunk_{len(chunks)}",
                text=text.strip(),
                source_id=source_id,
                modality="audio",
                metadata={
                    "start_time": window_start,
                    "end_time": buffer[-1]["end"],
                },
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_audio(audio_path: str | Path, config: dict) -> list[Chunk]:
    """
    Process a single audio file into a list of Chunk objects.

    Args:
        audio_path: Path to the source audio file.
        config:     Parsed contents of config/llm.yaml.

    Returns:
        Ordered list of Chunk objects with modality="audio".
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{audio_path.suffix}'. "
            f"audio_processor handles: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    source_id = audio_path.stem
    window_sec = float(config.get("audio_processor", {}).get("chunk_window_sec", 60.0))

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"[{source_id}] Converting to WAV…")
        wav_path = _to_wav(audio_path, tmp_dir)

        logger.info(f"[{source_id}] Transcribing with Whisper ({config['whisper']['model_size']})…")
        segments = _transcribe(wav_path, config)
        logger.info(f"[{source_id}] {len(segments)} transcript segments")

        logger.info(f"[{source_id}] Building chunks (window={window_sec}s)…")
        chunks = _build_chunks(source_id, segments, window_sec)

    logger.info(f"[{source_id}] Done — {len(chunks)} chunks produced")
    return chunks

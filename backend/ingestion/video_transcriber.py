"""
Stage 1 – Multimodal Ingestion: video_transcriber.py

Pipeline for a single lecture video:
  1. Extract audio with ffmpeg → temp WAV
  2. Transcribe with faster-whisper → timestamped segments
  3. Extract keyframes with OpenCV (SSIM-based) → JPEG files
  4. Caption each keyframe via a pluggable captioner (Ollama or Anthropic)
  5. Merge transcript + captions into Chunk objects aligned to keyframe boundaries

Public API:
    chunks = transcribe_video(video_path, config)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from backend.ingestion.captioners import BaseCaptioner, build_captioner
from backend.schemas import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Audio extraction
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path, tmp_dir: str) -> Path:
    """Strip audio track to a 16 kHz mono WAV using ffmpeg."""
    audio_path = Path(tmp_dir) / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr.decode()}"
        )
    return audio_path


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
# Step 3: Keyframe extraction
# ---------------------------------------------------------------------------

def _extract_keyframes(
    video_path: Path,
    keyframe_dir: Path,
    threshold: float,
    sample_fps: int = 1,
) -> list[dict]:
    """
    Sample the video at sample_fps frames/sec and emit a keyframe whenever
    SSIM between consecutive samples drops below threshold.

    Returns list of {timestamp: float, path: Path}.
    """
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(native_fps / sample_fps))

    keyframes: list[dict] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                score, _ = ssim(prev_gray, gray, full=True)
                if score < threshold:
                    timestamp = frame_idx / native_fps
                    kf_path = keyframe_dir / f"frame_{frame_idx:07d}.jpg"
                    cv2.imwrite(str(kf_path), frame)
                    keyframes.append({"timestamp": timestamp, "path": kf_path})
                    logger.debug(f"  Keyframe at {timestamp:.1f}s (SSIM={score:.3f})")

            prev_gray = gray

        frame_idx += 1

    cap.release()
    return keyframes


# ---------------------------------------------------------------------------
# Step 4: Chunk assembly
# ---------------------------------------------------------------------------

def _build_chunks(
    source_id: str,
    segments: list[dict],
    keyframes: list[dict],
    captioner: BaseCaptioner,
) -> list[Chunk]:
    """
    Align transcript segments to keyframe-defined intervals.

    Boundaries: [0, kf_0, kf_1, ..., kf_n, ∞]
    Each interval becomes one Chunk whose text is:
        "[Visual: <caption>] <transcript for that interval>"
    The caption comes from the keyframe that *opens* the interval.
    """
    kf_timestamps = sorted(kf["timestamp"] for kf in keyframes)
    kf_by_ts: dict[float, Path] = {kf["timestamp"]: kf["path"] for kf in keyframes}

    boundaries = [0.0] + kf_timestamps + [float("inf")]
    chunks: list[Chunk] = []

    for i in range(len(boundaries) - 1):
        interval_start = boundaries[i]
        interval_end = boundaries[i + 1]

        interval_segs = [
            s for s in segments
            if interval_start <= s["start"] < interval_end
        ]

        transcript_text = " ".join(s["text"] for s in interval_segs)

        kf_path = kf_by_ts.get(interval_start) if interval_start > 0.0 else None
        caption = ""
        if kf_path:
            try:
                caption = captioner.caption(kf_path)
            except Exception as exc:
                logger.warning(f"Captioning failed for {kf_path}: {exc}")

        full_text = f"[Visual: {caption}] {transcript_text}" if caption else transcript_text

        if not full_text.strip():
            continue

        actual_end = interval_segs[-1]["end"] if interval_segs else interval_start

        chunks.append(
            Chunk(
                id=f"{source_id}_chunk_{len(chunks)}",
                text=full_text.strip(),
                source_id=source_id,
                modality="video",
                metadata={
                    "start_time": interval_start,
                    "end_time": actual_end,
                    "keyframe_path": str(kf_path) if kf_path else None,
                },
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_video(video_path: str | Path, config: dict) -> list[Chunk]:
    """
    Process a single lecture video into a list of Chunk objects.

    Args:
        video_path: Path to the source video file.
        config:     Parsed contents of config/llm.yaml.

    Returns:
        Ordered list of Chunk objects with modality="video".
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    source_id = video_path.stem
    kf_cfg = config.get("keyframe", {})
    threshold = kf_cfg.get("ssim_threshold", 0.85)
    sample_fps = kf_cfg.get("sample_fps", 1)

    keyframe_dir = Path("backend/data/processed/keyframes") / source_id
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    captioner = build_captioner(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"[{source_id}] Extracting audio…")
        audio_path = _extract_audio(video_path, tmp_dir)

        logger.info(f"[{source_id}] Transcribing with Whisper ({config['whisper']['model_size']})…")
        segments = _transcribe(audio_path, config)
        logger.info(f"[{source_id}] {len(segments)} transcript segments")

        logger.info(f"[{source_id}] Extracting keyframes (threshold={threshold})…")
        keyframes = _extract_keyframes(video_path, keyframe_dir, threshold, sample_fps)
        logger.info(f"[{source_id}] {len(keyframes)} keyframes extracted")

        logger.info(f"[{source_id}] Building chunks…")
        chunks = _build_chunks(source_id, segments, keyframes, captioner)

    logger.info(f"[{source_id}] Done — {len(chunks)} chunks produced")
    return chunks

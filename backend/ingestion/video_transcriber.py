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
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from backend.ingestion.captioners import BaseCaptioner, build_captioner, safe_caption
from backend.ingestion.media_limits import decode_audio, transcribe_bounded
from backend.schemas import Chunk
from backend.utils.paths import processed_data_dir
from backend.utils.processes import (
    MAX_KEYFRAME_BYTES, MAX_KEYFRAMES, MAX_MEDIA_SECONDS, MAX_VIDEO_FRAMES,
    MAX_VIDEO_PIXELS, MAX_VIDEO_SAMPLES, run_parser,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Audio extraction
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path, tmp_dir: str) -> Path:
    """Strip audio track to a 16 kHz mono WAV using ffmpeg."""
    return decode_audio(video_path, tmp_dir, video=True)


# ---------------------------------------------------------------------------
# Step 2: Whisper transcription
# ---------------------------------------------------------------------------

def _transcribe(audio_path: Path, cfg: dict) -> list[dict]:
    """
    Transcribe audio with faster-whisper.
    Returns list of {start: float, end: float, text: str}.
    """
    return transcribe_bounded(audio_path, cfg)


# ---------------------------------------------------------------------------
# Step 3: Keyframe extraction
# ---------------------------------------------------------------------------

def _extract_keyframes(
    video_path: Path,
    keyframe_dir: Path,
    threshold: float,
    sample_fps: int = 1,
) -> list[dict]:
    """Keep native video decoders and their temporary assets outside the API."""
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=keyframe_dir) as temporary:
        frames = run_parser("keyframes", video_path, options={"keyframe_dir": temporary,
                            "threshold": threshold, "sample_fps": sample_fps})
        if len(frames) > MAX_KEYFRAMES:
            raise ValueError("Video keyframe count exceeds the processing limit.")
        root = Path(temporary).resolve()
        result = []
        for frame in frames:
            source = Path(frame["path"]).resolve()
            if source.parent != root or source.suffix != ".jpg" or not source.is_file():
                raise ValueError("Invalid keyframe output.")
            target = keyframe_dir / source.name
            os.replace(source, target)
            result.append({"timestamp": frame["timestamp"], "path": target})
        return result


def _extract_keyframes_local(
    video_path: Path, keyframe_dir: Path, threshold: float, sample_fps: float = 1,
) -> list[dict]:
    """
    Sample the video at sample_fps frames/sec and emit a keyframe whenever
    SSIM between consecutive samples drops below threshold.

    Returns list of {timestamp: float, path: Path}.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Video similarity threshold must be between zero and one.")
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 2:
        raise ValueError("Video sampling rate must be between zero and two frames per second.")
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if (not math.isfinite(native_fps) or not 0 < native_fps <= 120
            or not math.isfinite(frame_count) or frame_count > MAX_VIDEO_FRAMES
            or frame_count / native_fps > MAX_MEDIA_SECONDS
            or not all(math.isfinite(value) for value in (width, height))
            or width * height > MAX_VIDEO_PIXELS):
        cap.release()
        raise ValueError("Video dimensions, frames or duration exceed the processing limit.")
    step = max(1, int(native_fps / sample_fps))

    keyframes: list[dict] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    samples = output_bytes = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= MAX_VIDEO_FRAMES or frame_idx / native_fps > MAX_MEDIA_SECONDS:
                raise ValueError("Video frames or duration exceed the processing limit.")
            if frame.shape[0] * frame.shape[1] > MAX_VIDEO_PIXELS:
                raise ValueError("Video pixel count exceeds the processing limit.")
            if frame_idx % step == 0:
                samples += 1
                if samples > MAX_VIDEO_SAMPLES:
                    raise ValueError("Video sample count exceeds the processing limit.")
                ratio = min(1.0, 512 / max(frame.shape[:2]))
                sample = cv2.resize(frame, None, fx=ratio, fy=ratio) if ratio < 1 else frame
                gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    if gray.shape != prev_gray.shape:
                        raise ValueError("Video frame dimensions changed during processing.")
                    score = ssim(prev_gray, gray)
                    if score < threshold:
                        if len(keyframes) >= MAX_KEYFRAMES:
                            raise ValueError("Video keyframe count exceeds the processing limit.")
                        timestamp = frame_idx / native_fps
                        kf_path = keyframe_dir / f"frame_{frame_idx:07d}.jpg"
                        ratio = min(1.0, 1920 / max(frame.shape[:2]))
                        image = cv2.resize(frame, None, fx=ratio, fy=ratio) if ratio < 1 else frame
                        written, encoded = cv2.imencode(".jpg", image)
                        if not written:
                            raise ValueError("Video keyframe could not be written.")
                        output_bytes += encoded.nbytes
                        if output_bytes > MAX_KEYFRAME_BYTES:
                            raise ValueError("Video keyframe output exceeds the processing size limit.")
                        kf_path.write_bytes(encoded.tobytes())
                        keyframes.append({"timestamp": timestamp, "path": kf_path})
                prev_gray = gray
            frame_idx += 1
    finally:
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
        caption = safe_caption(captioner, kf_path, str(kf_path)) if kf_path else ""

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

    keyframe_dir = processed_data_dir(config) / "keyframes" / source_id
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

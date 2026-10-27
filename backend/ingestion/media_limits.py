"""Shared local-media conversion/transcription budgets."""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import wave

from backend.utils.processes import (
    FFMPEG_LIMITS, MAX_MEDIA_SECONDS, MAX_TRANSCRIPT_CHARS,
    MAX_TRANSCRIPT_SEGMENTS, MAX_WAV_BYTES, run_bounded, run_parser,
)

MAX_MEDIA_INPUT_BYTES = 50 * 1024 * 1024


def validate_wav(path: Path) -> None:
    if path.stat().st_size > MAX_WAV_BYTES:
        raise ValueError("Decoded audio exceeds the processing size limit.")
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16_000 or audio.getsampwidth() != 2:
            raise ValueError("Decoded audio must be 16 kHz mono PCM.")
        if audio.getnframes() / audio.getframerate() > MAX_MEDIA_SECONDS:
            raise ValueError("Media duration exceeds the processing limit.")


def decode_audio(source: Path, tmp_dir: str, *, video: bool = False) -> Path:
    if source.stat().st_size > MAX_MEDIA_INPUT_BYTES:
        raise ValueError("Media input exceeds the processing size limit.")
    # A fixed local basename avoids FFmpeg URL-special filename semantics.
    local_input = Path(tmp_dir) / f"input{source.suffix.lower()}"
    shutil.copyfile(source, local_input)
    output = Path(tmp_dir) / "audio.wav"
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-y",
        "-max_alloc", str(64 * 1024 * 1024), "-threads", "1",
        "-protocol_whitelist", "file", "-i", str(local_input),
    ]
    if video:
        command.append("-vn")
    command += [
        "-t", str(MAX_MEDIA_SECONDS + 1), "-fs", str(MAX_WAV_BYTES),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-threads", "1", str(output),
    ]
    try:
        run_bounded(command, FFMPEG_LIMITS)
        validate_wav(output)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return output


def transcribe_bounded(audio_path: Path, config: dict) -> list[dict]:
    settings = config["whisper"]
    # Deliberately pass only model settings, never provider credentials or the
    # rest of the application's request configuration, to the parser worker.
    options = {"model_size": settings["model_size"], "device": settings.get("device", "cpu"),
               "compute_type": settings.get("compute_type", "int8")}
    return run_parser("whisper", audio_path, options=options)


def transcribe_local(audio_path: Path, options: dict) -> list[dict]:
    """Worker-only Whisper iteration; parent enforces native-call deadlines."""
    validate_wav(audio_path)
    import numpy as np
    from faster_whisper import WhisperModel
    # FFmpeg already produced bounded 16 kHz mono PCM. Whisper accepts an
    # ndarray directly, avoiding a redundant PyAV decode and its version-
    # dependent container options. Normalize in place to bound peak memory.
    with wave.open(str(audio_path), "rb") as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)
    samples /= 32768.0
    model = WhisperModel(options["model_size"], device=options.get("device", "cpu"),
                         compute_type=options.get("compute_type", "int8"), cpu_threads=1, num_workers=1)
    segments, _ = model.transcribe(samples, beam_size=5)
    result = []
    characters = count = 0
    previous_start = 0.0
    for segment in segments:
        count += 1
        start, end, text = segment.start, segment.end, segment.text.strip()
        if count > MAX_TRANSCRIPT_SEGMENTS:
            raise ValueError("Transcript segment count exceeds the processing limit.")
        if not all(math.isfinite(value) for value in (start, end)) or not 0 <= previous_start <= start <= end <= MAX_MEDIA_SECONDS:
            raise ValueError("Transcript timestamps exceed the processing limit.")
        previous_start = start
        characters += len(text)
        if len(text) > 100_000 or characters > MAX_TRANSCRIPT_CHARS:
            raise ValueError("Transcript text exceeds the processing limit.")
        if text:
            result.append({"start": start, "end": end, "text": text})
    return result

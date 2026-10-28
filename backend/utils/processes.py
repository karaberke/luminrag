"""Bound untrusted media/document work outside the API process."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import psutil

from backend.utils.environment import hosted_mode

MIB = 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_DOCUMENT_TEXT_BYTES = 8 * MIB
MAX_MEDIA_SECONDS = 3600
MAX_WAV_BYTES = MAX_MEDIA_SECONDS * 16_000 * 2 + 64 * 1024
MAX_TRANSCRIPT_SEGMENTS = 20_000
MAX_TRANSCRIPT_CHARS = 1_000_000
MAX_VIDEO_PIXELS = 4096 * 4096
MAX_VIDEO_FRAMES = 360_000
MAX_VIDEO_SAMPLES = 7200
MAX_KEYFRAMES = 128
MAX_KEYFRAME_BYTES = 64 * MIB


class ProcessingLimitError(ValueError):
    """A parser exceeded its bounded processing allowance."""


@dataclass(frozen=True)
class ProcessLimits:
    seconds: float
    memory_bytes: int
    stdout_bytes: int = 16 * MIB
    stderr_bytes: int = 64 * 1024
    file_bytes: int = 128 * MIB


DOCUMENT_LIMITS = ProcessLimits(seconds=60, memory_bytes=512 * MIB)
FFMPEG_LIMITS = ProcessLimits(seconds=120, memory_bytes=512 * MIB, stdout_bytes=64 * 1024, file_bytes=MAX_WAV_BYTES)
WHISPER_LIMITS = ProcessLimits(seconds=1200, memory_bytes=2 * 1024 * MIB, file_bytes=1024 * MIB)
KEYFRAME_LIMITS = ProcessLimits(seconds=120, memory_bytes=1024 * MIB)
PDF_LIMITS = ProcessLimits(seconds=60, memory_bytes=512 * MIB, stdout_bytes=32 * MIB)


def _child_environment() -> dict[str, str]:
    # Parser processes do not need provider keys, session tokens or arbitrary
    # inherited Python startup hooks. Keep only OS/model-cache necessities.
    allowed = {
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
        "LANG", "LC_ALL", "APPDATA", "LOCALAPPDATA", "HF_HOME", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", LUMINRAG_SKIP_DOTENV="1",
               LUMIN_HOSTED="1" if hosted_mode() else "0",
               ORT_DISABLE_TELEMETRY="1",
               OPENCV_FFMPEG_CAPTURE_OPTIONS="protocol_whitelist;file")
    return env


def _kill_process_tree(process: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    for child in reversed(children):
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        process.kill()
    except OSError:
        pass


def run_bounded(command: list[str], limits: ProcessLimits, *, input_bytes: bytes = b"") -> bytes:
    """Run an argument vector with bounded pipes, wall time and process-tree RSS.

    Python workers additionally install hard OS limits before parsing. This
    watchdog also protects Windows and non-Python tools such as FFmpeg.
    """
    output = [bytearray(), bytearray()]
    exceeded = threading.Event()
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=REPO_ROOT, env=_child_environment(), start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    def drain(stream, index: int, maximum: int) -> None:
        try:
            while block := stream.read(16 * 1024):
                if len(output[index]) + len(block) > maximum:
                    exceeded.set()
                    _kill_process_tree(process)
                    return
                output[index].extend(block)
        finally:
            stream.close()

    readers = [threading.Thread(target=drain, args=(stream, index, maximum), daemon=True)
               for index, (stream, maximum) in enumerate(zip(
                   (process.stdout, process.stderr), (limits.stdout_bytes, limits.stderr_bytes)))]
    for reader in readers:
        reader.start()
    started = time.monotonic()
    failure = None
    try:
        try:
            if input_bytes:
                process.stdin.write(input_bytes)
                process.stdin.flush()
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            tracked = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            tracked = None
        while process.poll() is None:
            if exceeded.is_set():
                failure = "Parser output exceeds the processing limit."
                break
            if time.monotonic() - started > limits.seconds:
                failure = "Parser exceeded the processing time limit."
                break
            try:
                processes = [tracked, *tracked.children(recursive=True)] if tracked else []
                rss = sum(child.memory_info().rss for child in processes if child.is_running())
                if rss > limits.memory_bytes:
                    failure = "Parser exceeded the memory limit."
                    break
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.02)
        if failure or exceeded.is_set():
            _kill_process_tree(process)
        process.wait(timeout=5)
    except BaseException:
        _kill_process_tree(process)
        process.wait(timeout=5)
        raise
    finally:
        _kill_process_tree(process)
        for reader in readers:
            reader.join(timeout=5)
    if failure or exceeded.is_set():
        raise ProcessingLimitError(failure or "Parser output exceeds the processing limit.")
    if process.returncode:
        raise ProcessingLimitError("Parser failed or exceeded its processing allowance.")
    return bytes(output[0])


def apply_worker_limits(limits: ProcessLimits) -> None:
    """Apply hard Linux/Unix limits before importing native parser libraries."""
    if os.name == "nt":
        return  # Parent RSS/time/output watchdog applies on Windows.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
    cpu = max(1, math.ceil(limits.seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_bytes, limits.file_bytes))


def run_parser(kind: str, path: Path, *, options: dict | None = None) -> object:
    """Use the fixed worker entry point; never pass caller credentials/config."""
    limits = {"office": DOCUMENT_LIMITS, "whisper": WHISPER_LIMITS, "keyframes": KEYFRAME_LIMITS,
              "pdf": PDF_LIMITS, "slides": PDF_LIMITS}[kind]
    payload = json.dumps({"kind": kind, "path": str(path.resolve()), "options": options or {}}).encode("utf-8")
    if len(payload) > 64 * 1024:
        raise ValueError("Parser options exceed the processing limit.")
    raw = run_bounded([sys.executable, "-m", "backend.ingestion.parser_worker"], limits, input_bytes=payload)
    result = json.loads(raw)
    if "error" in result:
        raise ValueError(result["error"])
    return result["result"]


def publish_parser_assets(paths: list[str], source_root: Path, destination_root: Path, *, max_bytes: int) -> dict[str, str]:
    """Validate a worker's private output completely before moving any assets."""
    source_root, destination_root = source_root.resolve(), destination_root.resolve()
    transfers = []
    total = 0
    for name in dict.fromkeys(paths):
        source = Path(name)
        if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
               for path in (source, *source.parents)):
            raise ValueError("Linked parser output is not allowed.")
        resolved = source.resolve(strict=True)
        relative = resolved.relative_to(source_root)
        if not resolved.is_file() or relative == Path("."):
            raise ValueError("Invalid parser asset output.")
        target = destination_root / relative
        if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
               for path in (target, *target.parents)) or destination_root not in target.resolve().parents:
            raise ValueError("Unsafe parser asset destination.")
        total += resolved.stat().st_size
        if total > max_bytes:
            raise ValueError("Parser asset output exceeds the processing limit.")
        transfers.append((name, resolved, target))
    result = {}
    for name, source, target in transfers:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        result[name] = str(target)
    return result

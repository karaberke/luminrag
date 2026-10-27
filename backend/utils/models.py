"""Pinned public ONNX snapshots shared by runtime inference and image builds."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

# Must precede importing FastEmbed/ONNX Runtime: its API opt-out can be too late
# for initialization telemetry in official cross-platform builds.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

MODEL_MANIFEST = json.loads(Path(__file__).with_name("model_manifest.json").read_text(encoding="utf-8"))["models"]


def pinned_model_path(model_name: str, cache_dir: str | Path | None = None) -> Path:
    """Reuse an exact cached revision, verify every file, and fail closed on mismatch."""
    if model_name not in MODEL_MANIFEST:
        raise ValueError("The retrieval model is not in the verified model manifest.")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    description = MODEL_MANIFEST[model_name]
    cache = Path(cache_dir or os.getenv("FASTEMBED_CACHE_PATH") or Path(tempfile.gettempdir()) / "fastembed_cache").resolve()
    options = dict(repo_id=description["repo_id"], revision=description["revision"],
                   allow_patterns=list(description["files"]), cache_dir=str(cache),
                   endpoint="https://huggingface.co", token=False)
    try:
        snapshot = Path(snapshot_download(**options, local_files_only=True))
    except LocalEntryNotFoundError:
        if os.getenv("HF_HUB_OFFLINE", "").upper() in {"1", "TRUE", "YES", "ON"}:
            raise ValueError("The pinned retrieval model is missing from the offline cache.") from None
        snapshot = Path(snapshot_download(**options, local_files_only=False))
    if not snapshot.resolve().is_relative_to(cache):
        raise ValueError("The model snapshot escaped its cache directory.")
    for filename, expected in description["files"].items():
        path = snapshot / filename
        if not path.resolve().is_relative_to(cache) or not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError("A pinned retrieval model file is missing or has an incorrect size.")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError("A pinned retrieval model file failed SHA-256 verification.")
    return snapshot

"""Deployment copies and model pins must enforce their boundaries at real use sites."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.utils import models
from scripts.prepare_space import prepare_space


def test_model_bootstrap_disables_telemetry_before_lazy_onnx_import():
    result = subprocess.run([sys.executable, "-c", "import os,sys; os.environ.pop('ORT_DISABLE_TELEMETRY',None); import backend.retrieval.embedder; assert os.environ['ORT_DISABLE_TELEMETRY']=='1'; assert 'onnxruntime' not in sys.modules"],
                            cwd=Path(__file__).resolve().parents[2], env={**os.environ, "LUMINRAG_SKIP_DOTENV": "1"},
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


@pytest.fixture
def deployment_root(tmp_path):
    root = tmp_path / "repo"
    for name in ("backend/main.py", "backend/utils/model_manifest.json", "config/llm.yaml", "scripts/cache_models.py",
                 "requirements-hosted.txt", "requirements-hosted.lock", "requirements-bootstrap.txt", "requirements-bootstrap.lock",
                 "LICENSE", "README.md", "Dockerfile.hosted", "backend/data/private.py", "backend/tests/test_private.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    return root


def test_space_copy_contains_verified_build_inputs_and_excludes_local_data(deployment_root):
    output = prepare_space(deployment_root, deployment_root / ".tmp" / "space")
    assert (output / "backend/utils/model_manifest.json").is_file()
    assert (output / "requirements-hosted.lock").is_file()
    assert (output / "requirements-bootstrap.lock").is_file()
    assert (output / "Dockerfile").read_text() == "Dockerfile.hosted"
    assert not (output / "backend/data").exists()
    assert not (output / "backend/tests").exists()


@pytest.mark.parametrize("target", ["backend/main.py", "README.md", "config/linked"])
def test_space_copy_rejects_source_symlinks_before_creating_output(deployment_root, tmp_path, target):
    outside = tmp_path / "outside"
    outside.mkdir()
    file = outside / "secret.py"
    file.write_text("must not be copied")
    source = deployment_root / target
    if source.exists():
        source.unlink()
    try:
        source.symlink_to(outside if target.endswith("linked") else file, target_is_directory=target.endswith("linked"))
    except OSError:
        pytest.skip("The OS does not permit test symlink creation.")
    output = deployment_root / ".tmp" / "space"
    with pytest.raises(ValueError, match="symbolic links"):
        prepare_space(deployment_root, output)
    assert not output.exists()


def test_space_copy_rejects_junctions_and_destination_escape(deployment_root, tmp_path, monkeypatch):
    linked = deployment_root / "config" / "junction"
    linked.mkdir()
    monkeypatch.setattr(Path, "is_junction", lambda path: path == linked, raising=False)
    with pytest.raises(ValueError, match="junctions"):
        prepare_space(deployment_root, deployment_root / ".tmp" / "space")
    with pytest.raises(ValueError, match="inside the repository"):
        prepare_space(deployment_root, tmp_path / "outside")


@pytest.fixture
def pinned_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    snapshot = cache / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    content = b"verified model bytes"
    (snapshot / "model.onnx").write_bytes(content)
    monkeypatch.setattr(models, "MODEL_MANIFEST", {"test": {"repo_id": "trusted/test", "revision": "a" * 40,
        "files": {"model.onnx": {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}}}})
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return SimpleNamespace(cache=cache, snapshot=snapshot, calls=calls)


def test_pinned_model_reuses_cache_without_network_and_passes_exact_revision(pinned_snapshot):
    result = models.pinned_model_path("test", pinned_snapshot.cache)
    assert result == pinned_snapshot.snapshot
    assert len(pinned_snapshot.calls) == 1
    options = pinned_snapshot.calls[0]
    assert options["local_files_only"] and options["revision"] == "a" * 40
    assert options["token"] is False and options["endpoint"] == "https://huggingface.co"
    assert options["allow_patterns"] == ["model.onnx"]


def test_pinned_model_rejects_same_size_tampering(pinned_snapshot):
    path = pinned_snapshot.snapshot / "model.onnx"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256"):
        models.pinned_model_path("test", pinned_snapshot.cache)


def test_pinned_model_rejects_cache_escape(pinned_snapshot, tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: str(tmp_path))
    with pytest.raises(ValueError, match="escaped"):
        models.pinned_model_path("test", pinned_snapshot.cache)


def test_pinned_model_fails_offline_without_fetching(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    calls = []
    def missing(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("missing")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ValueError, match="offline cache"):
        models.pinned_model_path("BAAI/bge-small-en-v1.5", tmp_path)
    assert len(calls) == 1 and calls[0]["local_files_only"]

"""Main ingestion retires old assets in the same recoverable publication."""

import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.tests.test_ingest_safety import api_state  # noqa: F401
from backend.utils.state import current_epoch
from backend.utils.uploads import StagedFile


@pytest.mark.parametrize("store_fails", [False, True])
def test_asset_retirement_commits_or_rolls_back_with_replaced_source(api_state, monkeypatch, store_fails):
    api = api_state
    old_dir = api.processed / "pdf_images" / f"{api.sid}_{uuid.uuid4()}"
    old_dir.mkdir(parents=True)
    old_image = old_dir / "p0001_img0.png"
    old_unused = old_dir / "p0002_img0.png"
    unrelated = old_dir / "keep.txt"
    for path in (old_image, old_unused, unrelated):
        path.write_bytes(b"old owned material")
    metadata = {**api.old.metadata, "image_paths": [str(old_image)],
                "derived_asset_dirs": [old_dir.relative_to(api.processed).as_posix()]}
    api.store.save_chunks([api.old.model_copy(update={"metadata": metadata})])
    figure = api.processed / "illustrations" / f"{uuid.uuid4().hex}.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"old illustration")
    old_node = next(iter(api.graph.graph))
    api.graph.graph.nodes[old_node]["illustration_path"] = f"/static/illustrations/{figure.name}"
    api.graph.save()

    job_id = str(uuid.uuid4())
    job_dir = api.staging / job_id
    staged_path = job_dir / "00" / f"{api.sid}_{job_id}.pdf"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_bytes(b"new raw")
    staged = [StagedFile(staged_path, "notes.pdf", "notes.pdf", api.sid)]
    job = api.main._IngestJob(job_id)
    if store_fails:
        monkeypatch.setattr(api.store, "replace_sources", Mock(side_effect=RuntimeError("disk full")))
        with pytest.raises(RuntimeError, match="disk full"):
            api.main._ingest(job, staged, current_epoch(), job_dir)
        assert api.source.read_bytes() == b"old raw"
        assert all(path.read_bytes() == b"old owned material" for path in (old_image, old_unused))
        assert figure.read_bytes() == b"old illustration"
        assert api.store.get_chunk(api.old.id).text == "previous material"
    else:
        api.main._ingest(job, staged, current_epoch(), job_dir)
        assert job.status == "done"
        assert not old_image.exists()
        assert not old_unused.exists()
        assert not figure.exists()
        stored = api.store.get_chunk(api.old.id)
        assert stored.text == "new material"
        assert stored.metadata["derived_asset_dirs"] == [f"pdf_images/{api.sid}_{job_id}"]
        assert Path(stored.metadata["image_paths"][0]).read_bytes() == b"new derived image"
    assert unrelated.read_bytes() == b"old owned material"

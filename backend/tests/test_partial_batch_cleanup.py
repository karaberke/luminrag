"""A failed file in a successful batch must not retain its attempt assets."""

import asyncio
import uuid

import pytest

from backend.tests.test_ingest_safety import api_state  # noqa: F401
from backend.utils.state import current_epoch
from backend.utils.uploads import DERIVED_ASSET_DIRS, StagedFile, cleanup_attempt_assets, source_identity


@pytest.mark.parametrize("failure", ["raises", "empty"])
def test_partial_batch_removes_only_failed_attempt_outputs(api_state, monkeypatch, failure):
    api = api_state
    job_id = str(uuid.uuid4())
    job_dir = api.staging / job_id
    good_id = source_identity("new.pdf")
    staged = []
    for index, (source_id, filename) in enumerate(((api.sid, "notes.pdf"), (good_id, "new.pdf"))):
        path = job_dir / f"{index:02d}" / f"{source_id}_{job_id}.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"new upload")
        staged.append(StagedFile(path, filename, filename, source_id))
    original_parse = api.main._ingest_single_file
    failed_dirs = [api.processed / kind / staged[0].path.stem for kind in DERIVED_ASSET_DIRS]

    def parse(path, config, slides):
        if path == staged[0].path:
            for directory in failed_dirs:
                directory.mkdir(parents=True)
                (directory / "rendered-output.jpg").write_bytes(b"partial processor output")
            if failure == "raises":
                raise RuntimeError("caption extraction failed after rendering")
            return []
        return original_parse(path, config, slides)

    monkeypatch.setattr(api.main, "_ingest_single_file", parse)
    job = api.main._IngestJob(job_id)
    asyncio.run(api.main._run_ingest_job(job, staged, job_dir, current_epoch()))
    assert job.status == "done"
    assert job.result.files_processed == 1
    assert job.result.failed == ["notes.pdf"]
    assert job.result.ingested_source_ids == [good_id]
    api.main._jobs[job_id] = job
    response = api.client.get(f"/api/ingest/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["result"]["ingested_source_ids"] == [good_id]
    assert all(not directory.exists() for directory in failed_dirs)
    assert not job_dir.exists()
    assert api.source.read_bytes() == b"old raw"
    assert api.store.get_chunk(api.old.id).text == "previous material"
    from pathlib import Path
    assert Path(api.old.metadata["image_paths"][0]).read_bytes() == b"previous derived image"
    incoming = api.store.get_chunks_by_source(good_id)
    assert len(incoming) == 1
    assert Path(incoming[0].metadata["image_paths"][0]).read_bytes() == b"new derived image"


def test_cleanup_rejects_forged_attempt_path_that_names_committed_asset_directory(api_state):
    api = api_state
    committed = api.processed / "slides" / api.sid
    committed.mkdir(parents=True)
    image = committed / "slide_0001.jpg"
    image.write_bytes(b"committed source")
    job_id = str(uuid.uuid4())
    forged = StagedFile(api.staging / job_id / "00" / f"{api.sid}.pdf", "notes.pdf", "notes.pdf", api.sid)
    cleanup_attempt_assets(forged, api.staging, api.processed)
    assert image.read_bytes() == b"committed source"

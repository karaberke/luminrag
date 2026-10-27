"""Source retirement plans use synthetic files only and never perform deletion."""

from __future__ import annotations

import uuid
from pathlib import Path

import networkx as nx
import pytest

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk
from backend.utils.assets import LegacyReplacement, plan_retired_assets, record_derived_asset_dirs
from backend.utils.recovery import RecoveryManager
from backend.utils.uploads import source_identity


@pytest.fixture
def roots(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    return raw, processed


def _write(path: Path, value: bytes = b"synthetic") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path.resolve()


def _chunk(source: str, **metadata) -> Chunk:
    return Chunk(id=f"{source}-chunk", source_id=source, modality="pdf", text="Synthetic text", metadata=metadata)


def _plan(roots, old, new=(), replaced=None, before=None, after=None, legacy=()):
    raw, processed = roots
    return plan_retired_assets(
        old, new, set(replaced if replaced is not None else [chunk.source_id for chunk in old]),
        nx.DiGraph() if before is None else before, nx.DiGraph() if after is None else after,
        raw_root=raw, processed_root=processed, legacy_replacements=legacy,
    )


@pytest.mark.parametrize("kind,field,first,second", [
    ("slides", "slide_image_path", "slide_0001.jpg", "slide_0009.jpg"),
    ("pdf_images", "image_paths", "p0001_img0.png", "p0009_img2.jpeg"),
    ("keyframes", "keyframe_path", "frame_0000001.jpg", "frame_0000099.jpg"),
])
def test_replaced_unique_job_assets_are_planned_without_touching_other_files(roots, kind, field, first, second):
    _, processed = roots
    source = source_identity("notes.pdf")
    directory = processed / kind / f"{source}_{uuid.uuid4()}"
    old_asset = _write(directory / first)
    unused_asset = _write(directory / second)
    backup = _write(directory / f".{first}.luminrag-{uuid.uuid4().hex}.bak")
    unknown = _write(directory / "notes.txt")
    nested = _write(directory / "nested" / first)
    other_job = _write(processed / kind / f"{source}_{uuid.uuid4()}" / first)
    metadata = {field: [str(old_asset)] if field == "image_paths" else str(old_asset)}

    assert set(_plan(roots, [_chunk(source, **metadata)])) == {old_asset, unused_asset}
    assert all(path.exists() for path in (old_asset, unused_asset, backup, unknown, nested, other_job))


def test_recorded_directory_covers_unused_frames_without_a_chunk_image_reference(roots):
    _, processed = roots
    source = source_identity("lecture.mp4")
    staged = processed.parent / "staging" / f"{source}_{uuid.uuid4()}.mp4"
    unused = _write(processed / "keyframes" / staged.stem / "frame_0000010.jpg")
    original = _chunk(source, start_time=0)

    recorded = record_derived_asset_dirs([original], staged, processed)

    assert original.metadata == {"start_time": 0}
    assert recorded[0].metadata["derived_asset_dirs"] == [f"keyframes/{staged.stem}"]
    assert _plan(roots, recorded) == [unused]


def test_candidate_directory_and_shared_metadata_protect_every_file(roots):
    _, processed = roots
    source = source_identity("notes.pdf")
    old_directory = processed / "slides" / f"{source}_{uuid.uuid4()}"
    old_asset = _write(old_directory / "slide_0001.jpg")
    unused = _write(old_directory / "slide_0009.jpg")
    old = _chunk(source, slide_image_path=str(old_asset))
    shared = _chunk("another-source", slide_image_path=str(old_asset))
    assert _plan(roots, [old], [shared]) == []

    new_staged = processed.parent / "staging" / f"{source}_{uuid.uuid4()}.pdf"
    new_asset = _write(processed / "slides" / new_staged.stem / "slide_0001.jpg")
    new = record_derived_asset_dirs([_chunk(source)], new_staged, processed)
    assert set(_plan(roots, [old], new)) == {old_asset, unused}
    assert new_asset.exists()


def test_unsuccessful_other_source_in_a_batch_is_not_retired(roots):
    _, processed = roots
    first, second = source_identity("a.pdf"), source_identity("b.pdf")
    a = _write(processed / "slides" / f"{first}_{uuid.uuid4()}" / "slide_0001.jpg")
    b = _write(processed / "slides" / f"{second}_{uuid.uuid4()}" / "slide_0001.jpg")
    old = [_chunk(first, slide_image_path=str(a)), _chunk(second, slide_image_path=str(b))]
    assert _plan(roots, old, [old[1]], replaced={first}) == [a]


def test_ambiguous_historical_and_wrong_owner_directories_are_left_alone(roots):
    _, processed = roots
    source = source_identity("notes.pdf")
    historical = _write(processed / "slides" / source / "slide_0001.jpg")
    other = _write(processed / "slides" / f"different-source_{uuid.uuid4()}" / "slide_0001.jpg")
    old = _chunk(source, image_paths=[str(historical), str(other)])
    assert _plan(roots, [old]) == []


def test_arbitrary_metadata_paths_never_establish_deletion_ownership(roots, tmp_path):
    raw, processed = roots
    source = source_identity("notes.pdf")
    outside = _write(tmp_path / "outside" / "slide_0001.jpg")
    database = _write(processed / "chunks.db")
    journal = _write(processed / "recovery" / "journal.json")
    other_raw = _write(raw / "other-source" / "notes.pdf")
    traversal = processed / "slides" / ".." / "chunks.db"
    old = _chunk(source, original_filename="notes.pdf", image_paths=[str(outside), str(database), str(journal), str(traversal)], source_file=str(other_raw), derived_asset_dirs=["recovery", "slides/../../outside"])
    assert _plan(roots, [old]) == []


def test_modern_raw_requires_filename_identity_and_preserves_current_source(roots):
    raw, _ = roots
    source = source_identity("notes.pdf")
    original = _write(raw / source / "notes.pdf")
    old = _chunk(source, original_filename="notes.pdf", source_path=f"{source}/notes.pdf")
    assert _plan(roots, [old]) == [original]
    assert _plan(roots, [old], [old]) == []
    forged = old.model_copy(update={"metadata": {**old.metadata, "original_filename": "different.pdf"}})
    assert _plan(roots, [forged]) == []


def test_flat_legacy_raw_requires_explicit_replacement_and_compatible_modality(roots):
    raw, _ = roots
    flat = _write(raw / "lecture.pdf")
    old = _chunk("lecture")
    validated = LegacyReplacement("lecture", "pdf", flat)
    assert _plan(roots, [old]) == []
    assert _plan(roots, [old], legacy=[validated]) == [flat]
    assert _plan(roots, [old], legacy=[LegacyReplacement("lecture", "video", flat)]) == []
    other_modality = old.model_copy(update={"id": "video-chunk", "modality": "video"})
    assert _plan(roots, [old, other_modality], legacy=[validated]) == []
    assert _plan(roots, [old], replaced=set(), legacy=[validated]) == []
    assert _plan(roots, [old], [old], legacy=[validated]) == []


def test_only_dropped_generated_illustrations_are_retired(roots):
    _, processed = roots
    dropped, retained = uuid.uuid4().hex + ".png", uuid.uuid4().hex + ".jpg"
    removed_file = _write(processed / "illustrations" / dropped)
    _write(processed / "illustrations" / retained)
    _write(processed / "illustrations" / "legacy-unknown.png")
    before, after = nx.DiGraph(), nx.DiGraph()
    before.add_node("removed", illustration_path=f"/static/illustrations/{dropped}")
    before.add_node("retained", illustration_path=f"/static/illustrations/{retained}")
    before.add_node("unknown", illustration_path="/static/illustrations/legacy-unknown.png")
    before.add_node("foreign", illustration_path=f"https://attacker.test/static/illustrations/{retained}")
    before.add_node("traversal", illustration_path=f"/static/illustrations/../{retained}")
    after.add_node("retained", illustration_path=f"/static/illustrations/{retained}")
    assert _plan(roots, [], before=before, after=after) == [removed_file]
    after.add_node("another-reference", illustration_path=f"/static/illustrations/{dropped}")
    assert _plan(roots, [], before=before, after=after) == []


def test_linked_generated_file_cannot_target_an_outside_file(roots, tmp_path):
    _, processed = roots
    source = source_identity("notes.pdf")
    directory = processed / "slides" / f"{source}_{uuid.uuid4()}"
    directory.mkdir(parents=True)
    outside = _write(tmp_path / "outside.jpg")
    linked = directory / "slide_0001.jpg"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not permit symlink creation")
    old = _chunk(source, derived_asset_dirs=[directory.relative_to(processed).as_posix()])
    assert _plan(roots, [old]) == []
    assert outside.read_bytes() == b"synthetic"


def test_junction_guard_rejects_owned_directory_without_traversing_it(roots, monkeypatch):
    _, processed = roots
    source = source_identity("notes.pdf")
    directory = processed / "slides" / f"{source}_{uuid.uuid4()}"
    asset = _write(directory / "slide_0001.jpg")
    monkeypatch.setattr(Path, "is_junction", lambda self: self == directory, raising=False)
    old = _chunk(source, slide_image_path=str(asset))
    assert _plan(roots, [old]) == []


def test_retirement_plan_uses_the_same_rollback_boundary_as_new_content(roots):
    raw, processed = roots
    source = source_identity("notes.pdf")
    old_file = _write(processed / "slides" / f"{source}_{uuid.uuid4()}" / "slide_0001.jpg", b"old image")
    old = _chunk(source, slide_image_path=str(old_file))
    new = old.model_copy(update={"text": "Replacement", "metadata": {}})
    with DocumentStore(processed / "chunks.db") as store:
        store.save_chunks([old])
        manager = RecoveryManager(processed / "recovery", [raw, processed], store)
        plan = _plan(roots, [old], [new])

        def fail_commit(_transaction_id):
            raise RuntimeError("Synthetic SQL failure")

        with pytest.raises(RuntimeError, match="Synthetic SQL failure"):
            manager.publish([(None, path) for path in plan], fail_commit)
        assert old_file.read_bytes() == b"old image"
        assert store.get_chunk(old.id).text == old.text

        manager.publish([(None, path) for path in plan], lambda transaction_id: store.replace_sources([new], transaction_id=transaction_id))
        assert not old_file.exists()
        assert store.get_chunk(new.id).text == "Replacement"
        assert not list(processed.rglob("*.luminrag-*"))


def test_shared_metadata_paths_are_checked_once_per_plan_and_revalidated_next_time(roots, monkeypatch):
    from backend.utils import assets

    _, processed = roots
    source = source_identity("notes.pdf")
    asset = _write(processed / "slides" / f"{source}_{uuid.uuid4()}" / "slide_0001.jpg")
    repeated = [_chunk(source, slide_image_path=str(asset)) for _ in range(1000)]
    original_check = assets._checked_path
    calls = []

    def checked(path, root, **kwargs):
        calls.append((path, root))
        return original_check(path, root, **kwargs)

    monkeypatch.setattr(assets, "_checked_path", checked)
    assert _plan(roots, [], repeated) == []
    assert len(calls) == 2  # One candidate file, checked under the two allowed roots.
    assert _plan(roots, [], repeated) == []
    assert len(calls) == 4  # A fresh plan must not retain cached filesystem results.

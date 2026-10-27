"""
Regression tests for graph safety fixes:
  - prune_to_ratio is non-destructive (1-1-1 graph survives)
  - node text uses the current `summary` field
  - semantic-linker candidate search is bounded and keeps the best pairs
  - illustration source lookup via source_ids, image-only
  - IllustrationScheduler.reset() discards in-flight work for the old graph

All embedders are fakes — no model downloads.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from backend.graph import illustration_worker as iw
from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import (
    ContentAttrs,
    ContentProposal,
    ProposalBundle,
    SubtopicAttrs,
    SubtopicProposal,
    TopicAttrs,
    TopicProposal,
)
from backend.graph.semantic_linker import _candidate_pairs, build_linkables_from_graph
from backend.retrieval.graph_retriever import _node_text


def _types(builder: GraphBuilder) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, a in builder.graph.nodes(data=True):
        counts[a["node_type"]] = counts.get(a["node_type"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# prune_to_ratio
# ---------------------------------------------------------------------------

class TestPruneToRatioNonDestructive:
    def test_one_one_one_graph_survives(self, tmp_path):
        b = GraphBuilder(tmp_path / "g.json")
        b.apply_proposals(ProposalBundle(
            topics=[TopicProposal(name="NLP")],
            subtopics=[SubtopicProposal(name="Transformers", parent_topic_names=["NLP"])],
            contents=[ContentProposal(
                title="Self-Attention", parent_subtopic_names=["Transformers"],
            )],
        ))
        # Same sequence as the ingest pipeline.
        b.prune_disconnected()
        assert b.prune_to_ratio() == {"topic": 0, "subtopic": 0}
        b.prune_disconnected()
        assert _types(b) == {"topic": 1, "subtopic": 1, "content": 1}
        assert b.edge_count() == 2

    def test_connected_nodes_kept_when_ratio_violated(self, tmp_path):
        b = GraphBuilder(tmp_path / "g.json")
        topics = [b.add_node(TopicAttrs(name=f"T{i}")) for i in range(3)]
        subs = [b.add_node(SubtopicAttrs(name=f"S{i}")) for i in range(3)]
        for t, s in zip(topics, subs):
            b.add_structural_edge(t, "HAS_SUBTOPIC", s)
            c = b.add_node(ContentAttrs(name=f"C-{s}"))
            b.add_structural_edge(s, "HAS_CONTENT", c)
        n = b.node_count()
        assert b.prune_to_ratio() == {"topic": 0, "subtopic": 0}
        assert b.node_count() == n


# ---------------------------------------------------------------------------
# Node text uses `summary`
# ---------------------------------------------------------------------------

class TestSummaryText:
    def test_content_summary_used_for_linking_and_retrieval(self):
        g = nx.MultiDiGraph()
        g.add_node("content:a", name="C", node_type="content", summary="current text")
        assert "current text" in build_linkables_from_graph(g)[0]["text"]
        assert "current text" in _node_text(g.nodes["content:a"], "content:a")

    def test_legacy_field_still_read(self):
        attrs = {"name": "C", "node_type": "content", "summary_intermediate": "old"}
        assert "old" in _node_text(attrs, "content:a")


# ---------------------------------------------------------------------------
# Bounded candidate search
# ---------------------------------------------------------------------------

def _fake_embedder(vectors: np.ndarray) -> MagicMock:
    by_text = {f"n{i}": v for i, v in enumerate(vectors)}
    mock = MagicMock()
    mock.embed.side_effect = lambda texts: np.stack([by_text[t] for t in texts])
    return mock


def _nodes(n: int) -> list[dict]:
    return [{"key": f"k{i}", "name": f"n{i}", "text": f"n{i}", "node_type": "content"}
            for i in range(n)]


@pytest.fixture
def knn_backend():
    return pytest.importorskip("faiss")


class TestCandidatePairs:
    def test_large_graph_uses_bounded_neighbor_search(self, knn_backend, monkeypatch):
        rng = np.random.default_rng(3)
        vecs = rng.normal(size=(400, 32)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        monkeypatch.setattr(knn_backend, "IndexFlatIP", lambda *_: pytest.fail("large graph used exhaustive search"))
        pairs = _candidate_pairs(_nodes(400), _fake_embedder(vecs), 0.1, 0.95, 20)
        assert len(pairs) == 20
        assert all(0.1 <= score <= 0.95 for _, _, _, score in pairs)

    def test_capped_and_highest_in_range_pairs_kept(self, knn_backend):
        rng = np.random.default_rng(0)
        n, dim = 150, 32
        common = rng.normal(size=dim)
        vecs = common + 0.9 * rng.normal(size=(n, dim))
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
        sim_lo, sim_hi, max_pairs = 0.3, 0.99, 25
        emb = _fake_embedder(vecs)

        # neighbors_per_node >= max_pairs guarantees the global top-k is reachable
        # even through a hub node, while still searching far fewer than N.
        pairs = _candidate_pairs(
            _nodes(n), emb, sim_lo, sim_hi, max_pairs, neighbors_per_node=max_pairs
        )

        assert emb.embed.call_count == 1
        assert len(pairs) == max_pairs
        assert [p[0] for p in pairs] == list(range(max_pairs))
        sims = [p[3] for p in pairs]
        assert sims == sorted(sims, reverse=True)
        assert all(sim_lo <= s <= sim_hi for s in sims)

        # Reference: exact global top-k of in-range pairs (test-only brute force).
        full = vecs @ vecs.T
        iu, ju = np.triu_indices(n, 1)
        ref = [(float(full[i, j]), i, j) for i, j in zip(iu, ju)
               if sim_lo <= full[i, j] <= sim_hi]
        ref.sort(reverse=True)
        expected = {(f"k{i}", f"k{j}") for _, i, j in ref[:max_pairs]}
        got = {(a["key"], b["key"]) for _, a, b, _ in pairs}
        assert got == expected
        # Stable orientation: a precedes b in input order, no self/duplicate pairs.
        assert all(int(a[1:]) < int(b[1:]) for a, b in got)

    def test_near_duplicates_excluded_but_neighbours_kept(self, knn_backend):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([0.7, 0.7, 0.0], dtype=np.float32)
        c /= np.linalg.norm(c)
        vecs = np.stack([a, a, c, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
        pairs = _candidate_pairs(_nodes(4), _fake_embedder(vecs), 0.55, 0.9, 10)
        got = sorted((x["key"], y["key"]) for _, x, y, _ in pairs)
        assert got == [("k0", "k2"), ("k1", "k2")]

    def test_zero_cap_returns_nothing(self, knn_backend):
        vecs = np.eye(3, dtype=np.float32)
        assert _candidate_pairs(_nodes(3), _fake_embedder(vecs), 0.0, 1.0, 0) == []


# ---------------------------------------------------------------------------
# Source-image resolution
# ---------------------------------------------------------------------------

def test_candidate_graph_and_failed_save_preserve_original(tmp_path, monkeypatch):
    from backend.utils import files

    path = tmp_path / "graph.json"
    original = GraphBuilder(path)
    key = original.add_node(ContentAttrs(name="A", source_ids=["old"]))
    original.save()
    previous = path.read_bytes()
    candidate = original.clone(tmp_path / "candidate.json")
    candidate.graph.nodes[key]["source_ids"].append("new")
    assert original.get_node(key)["source_ids"] == ["old"]

    def fail_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(files.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        original.save()
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def _store(chunks: dict[str, dict]) -> SimpleNamespace:
    return SimpleNamespace(
        get_chunk=lambda cid: SimpleNamespace(metadata=chunks[cid]) if cid in chunks else None
    )


class TestFindSourceImage:
    def test_resolves_via_source_ids_and_skips_pdf(self, tmp_path):
        pdf = tmp_path / "deck.pdf"
        pdf.write_bytes(b"%PDF")
        png = tmp_path / "img.png"
        png.write_bytes(b"png")
        b = GraphBuilder(tmp_path / "g.json")
        key = b.add_node(ContentAttrs(name="Attn", source_ids=["c1", "c2"]))
        store = _store({
            "c1": {"source_file": str(pdf), "slide_image_path": str(tmp_path / "gone.jpg")},
            "c2": {"image_paths": [str(pdf), str(png)]},
        })
        assert iw._find_source_image(key, b, store) == str(png)

    def test_pdf_only_returns_none(self, tmp_path):
        pdf = tmp_path / "deck.pdf"
        pdf.write_bytes(b"%PDF")
        b = GraphBuilder(tmp_path / "g.json")
        key = b.add_node(ContentAttrs(name="Attn", source_ids=["c1"]))
        assert iw._find_source_image(key, b, _store({"c1": {"source_file": str(pdf)}})) is None

    def test_keyframe_path_used(self, tmp_path):
        frame = tmp_path / "frame.JPG"
        frame.write_bytes(b"jpg")
        b = GraphBuilder(tmp_path / "g.json")
        key = b.add_node(ContentAttrs(name="Attn", source_ids=["v1"]))
        store = _store({"v1": {"keyframe_path": str(frame)}})
        assert iw._find_source_image(key, b, store) == str(frame)


# ---------------------------------------------------------------------------
# IllustrationScheduler.reset
# ---------------------------------------------------------------------------

def test_reset_discards_in_flight_and_queued_work(tmp_path, monkeypatch):
    async def scenario() -> None:
        old = GraphBuilder(tmp_path / "old.json")
        k1 = old.add_node(ContentAttrs(name="A"))
        k2 = old.add_node(ContentAttrs(name="B"))
        started, release = asyncio.Event(), asyncio.Event()

        async def fake_resolve(node_key, *args, **kwargs):
            started.set()
            await release.wait()
            out = tmp_path / f"{node_key.replace(':', '_')}.png"
            out.write_bytes(b"png")
            return out

        monkeypatch.setattr(iw, "resolve_illustration", fake_resolve)
        sched = iw.IllustrationScheduler(old, doc_store=None, output_dir=tmp_path, config={})
        sched.start()
        await sched.enqueue(k1, "hint", "image")
        await sched.enqueue(k2, "hint", "image")
        await asyncio.wait_for(started.wait(), 5)

        new = GraphBuilder(tmp_path / "new.json")
        new.add_node(ContentAttrs(name="A"))
        sched.reset(new)
        release.set()
        await asyncio.wait_for(sched._queue.join(), 5)
        await sched.stop()

        for builder in (old, new):
            for k in (k1, k2):
                node = builder.get_node(k)
                assert node is None or not node.get("illustration_path")
        # A stale save must not resurrect the deleted graph file.
        assert not (tmp_path / "old.json").exists()
        assert not (tmp_path / "new.json").exists()
        assert not list(tmp_path.glob("*.png"))

    asyncio.run(scenario())


def test_queue_drain_preserves_work_added_after_reset(tmp_path):
    async def scenario():
        builder = GraphBuilder(tmp_path / "g.json")
        sched = iw.IllustrationScheduler(builder, None, tmp_path, {})
        await sched.enqueue("old", "hint", "image")
        sched.reset(builder)
        await sched.enqueue("new", "hint", "image")
        sched._drain_queue()
        assert sched._queue.get_nowait()[1] == "new"
        sched._queue.task_done()
        await asyncio.wait_for(sched._queue.join(), 1)

    asyncio.run(scenario())


def test_reset_between_snapshot_and_enqueue_cannot_relabel_old_work(tmp_path, monkeypatch):
    async def scenario():
        builder = GraphBuilder(tmp_path / "g.json")
        sched = iw.IllustrationScheduler(builder, None, tmp_path, {})
        monkeypatch.setattr(sched, "_pending_snapshot", lambda: (0, [("node", "old hint", "image")]))
        original_enqueue = sched.enqueue

        async def resetting_enqueue(*args, **kwargs):
            sched.reset(builder)
            await original_enqueue(*args, **kwargs)

        monkeypatch.setattr(sched, "enqueue", resetting_enqueue)
        await sched.scan_and_enqueue_pending()
        assert sched._queue.empty()
        assert not sched._seen

    asyncio.run(scenario())


def test_worker_writes_when_not_reset(tmp_path, monkeypatch):
    async def scenario() -> None:
        b = GraphBuilder(tmp_path / "g.json")
        key = b.add_node(ContentAttrs(name="A"))
        out = tmp_path / "a.png"
        out.write_bytes(b"png")

        async def fake_resolve(*args, **kwargs) -> Path:
            return out

        monkeypatch.setattr(iw, "resolve_illustration", fake_resolve)
        sched = iw.IllustrationScheduler(b, doc_store=None, output_dir=tmp_path, config={})
        sched.start()
        await sched.enqueue(key, "hint", "image")
        await asyncio.wait_for(sched._queue.join(), 5)
        await sched.stop()

        assert b.get_node(key)["illustration_path"] == "/static/illustrations/a.png"
        assert (tmp_path / "g.json").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["render", "commit"])
def test_stop_waits_for_threaded_work_before_clear(tmp_path, monkeypatch, phase):
    async def scenario() -> None:
        builder = GraphBuilder(tmp_path / "graph.json")
        key = builder.add_node(ContentAttrs(name="Active"))
        queued_key = builder.add_node(ContentAttrs(name="Queued"))
        output_dir = tmp_path / "illustrations"
        output_dir.mkdir()
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def delay() -> None:
            started.set()
            assert release.wait(5), "test did not release background work"

        if phase == "render":
            def delayed_copy(_source, out_path):
                delay()
                out_path.write_bytes(b"private image")
                finished.set()
                return True

            monkeypatch.setattr(iw, "_find_source_image", lambda *_: "source.png")
            monkeypatch.setattr(iw, "_copy_if_missing", delayed_copy)
        else:
            async def resolved(*_args, **_kwargs):
                out = output_dir / "image.png"
                out.write_bytes(b"private image")
                return out

            save = builder.save

            def delayed_save():
                delay()
                save()
                finished.set()

            monkeypatch.setattr(iw, "resolve_illustration", resolved)
            monkeypatch.setattr(builder, "save", delayed_save)

        scheduler = iw.IllustrationScheduler(builder, None, output_dir, {})
        stopping = None
        try:
            await scheduler.enqueue(key, "source image", "image")
            await scheduler.enqueue(queued_key, "unused", "image")
            scheduler.start()
            assert await asyncio.to_thread(started.wait, 5)
            stopping = asyncio.create_task(scheduler.stop())
            # stop cannot finish while the renderer or persistence thread is
            # still able to recreate data after its caller clears the files.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopping), 0.05)
            release.set()
            await asyncio.wait_for(stopping, 5)
            assert finished.is_set()
            assert scheduler._queue.empty()
            await asyncio.wait_for(scheduler._queue.join(), 1)
            assert not builder.get_node(queued_key).get("illustration_path")

            for image in output_dir.iterdir():
                image.unlink()
            (tmp_path / "graph.json").unlink()
            assert not list(output_dir.iterdir())
            assert not (tmp_path / "graph.json").exists()
        finally:
            release.set()
            if stopping is not None:
                await asyncio.wait_for(stopping, 5)
            else:
                await scheduler.stop()

    asyncio.run(scenario())


def test_stop_wakes_idle_worker_and_balances_queue(tmp_path):
    async def scenario() -> None:
        scheduler = iw.IllustrationScheduler(GraphBuilder(tmp_path / "graph.json"), None, tmp_path, {})
        scheduler.start()
        await asyncio.wait_for(scheduler.stop(), 2)
        assert scheduler._queue.empty()
        await asyncio.wait_for(scheduler._queue.join(), 1)

    asyncio.run(scenario())


def test_suspend_fences_inflight_writer_until_reset_to_committed_builder(tmp_path, monkeypatch):
    async def scenario():
        graph_path = tmp_path / "graph.json"
        old = GraphBuilder(graph_path)
        old_key = old.add_node(ContentAttrs(name="Old content", illustration={"hint": "old hint", "kind": "image"}))
        old.save()
        started, release = asyncio.Event(), asyncio.Event()
        old_image, new_image = tmp_path / "old-render.png", tmp_path / "new-render.png"

        async def resolve(node_key, *_args, **_kwargs):
            if node_key == old_key:
                started.set()
                await release.wait()
                out = old_image
            else:
                out = new_image
            out.write_bytes(b"synthetic image")
            return out

        monkeypatch.setattr(iw, "resolve_illustration", resolve)
        scheduler = iw.IllustrationScheduler(old, None, tmp_path, {})
        scheduler.start()
        try:
            await scheduler.enqueue(old_key, "old hint", "image")
            await asyncio.wait_for(started.wait(), 5)

            # A storage transaction publishes a new graph while the old render
            # is outside the lock, then its adoption/bookkeeping fails closed.
            candidate = GraphBuilder(tmp_path / "candidate.json")
            new_key = candidate.add_node(ContentAttrs(name="Committed content", illustration={"hint": "new hint", "kind": "image"}))
            candidate = candidate.clone(graph_path)
            candidate.save()
            committed_bytes = graph_path.read_bytes()
            scheduler.suspend()
            assert graph_path.read_bytes() == committed_bytes
            assert not scheduler._needs_illustration(scheduler._generation, old_key)
            assert await scheduler.scan_and_enqueue_pending() == 0
            await scheduler.enqueue(old_key, "another hint", "image")
            assert scheduler._queue.empty()

            release.set()
            await asyncio.wait_for(scheduler._queue.join(), 5)
            assert not old_image.exists()
            assert not old.get_node(old_key).get("illustration_path")
            assert graph_path.read_bytes() == committed_bytes, "A suspended writer overwrote the committed graph"

            scheduler.reset(candidate)
            assert await scheduler.scan_and_enqueue_pending() == 1
            await asyncio.wait_for(scheduler._queue.join(), 5)
            assert candidate.get_node(new_key)["illustration_path"] == "/static/illustrations/new-render.png"
            assert new_image.exists()
            restored = GraphBuilder(graph_path)
            assert restored.get_node(old_key) is None
            assert restored.get_node(new_key)["illustration_path"] == "/static/illustrations/new-render.png"
        finally:
            release.set()
            await asyncio.wait_for(scheduler.stop(), 5)

    asyncio.run(scenario())


def test_suspended_commit_rejects_even_current_generation_without_saving(tmp_path):
    builder = GraphBuilder(tmp_path / "graph.json")
    key = builder.add_node(ContentAttrs(name="Uncertain state"))
    builder.save()
    before = (tmp_path / "graph.json").read_bytes()
    output = tmp_path / "discard.png"
    output.write_bytes(b"synthetic image")
    scheduler = iw.IllustrationScheduler(builder, None, tmp_path, {})

    scheduler.suspend()
    scheduler._commit(scheduler._generation, builder, key, output)

    assert not output.exists()
    assert not builder.get_node(key).get("illustration_path")
    assert (tmp_path / "graph.json").read_bytes() == before

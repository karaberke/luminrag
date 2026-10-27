"""
Tests for DocumentStore.replace_sources, graph source reconciliation, and
the mutation-lock helpers in backend/utils/state.py.

All storage is under tmp_path; graphs are in-memory networkx objects.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import networkx as nx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk
from backend.utils.associations import add_association, association_id
from backend.utils import state as mstate
from backend.utils.state import (
    MUTATION_LOCK,
    bump_epoch,
    current_epoch,
    locked,
    reconcile_graph_sources,
    run_locked,
)


def _chunk(cid: str, source: str, text: str = "t") -> Chunk:
    return Chunk(id=cid, source_id=source, modality="pdf", text=text)


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(tmp_path / "chunks.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# replace_sources
# ---------------------------------------------------------------------------

class TestReplaceSources:
    def test_removes_stale_chunks_and_returns_old_ids(self, store):
        store.save_chunks([_chunk("a_0", "a"), _chunk("a_1", "a"), _chunk("a_2", "a")])
        old = store.replace_sources([_chunk("a_0", "a", "new")])
        assert sorted(old) == ["a_0", "a_1", "a_2"]
        remaining = store.get_chunks_by_source("a")
        assert [c.id for c in remaining] == ["a_0"]
        assert remaining[0].text == "new"

    def test_other_sources_untouched(self, store):
        store.save_chunks([_chunk("a_0", "a"), _chunk("b_0", "b")])
        store.replace_sources([_chunk("a_9", "a")])
        assert store.get_chunk("b_0") is not None
        assert store.count() == 2

    def test_new_source_returns_empty(self, store):
        assert store.replace_sources([_chunk("n_0", "n")]) == []
        assert store.count() == 1

    def test_multiple_sources_in_one_call(self, store):
        store.save_chunks([_chunk("a_0", "a"), _chunk("b_0", "b")])
        old = store.replace_sources([_chunk("a_1", "a"), _chunk("b_1", "b")])
        assert sorted(old) == ["a_0", "b_0"]
        assert sorted(c.id for c in store.get_all_chunks()) == ["a_1", "b_1"]

    def test_failure_rolls_back(self, store):
        store.save_chunks([_chunk("a_0", "a"), _chunk("a_1", "a")])

        class _FailingInsert:
            def __init__(self, conn):
                self._conn = conn

            def executemany(self, *_a, **_k):
                raise sqlite3.OperationalError("disk full")

            def __getattr__(self, name):
                return getattr(self._conn, name)

        real = store._conn
        store._conn = _FailingInsert(real)
        try:
            with pytest.raises(sqlite3.OperationalError):
                store.replace_sources([_chunk("a_9", "a")])
        finally:
            store._conn = real
        assert sorted(c.id for c in store.get_chunks_by_source("a")) == ["a_0", "a_1"]
        # Connection is usable afterwards (no dangling transaction).
        store.save_chunks([_chunk("c_0", "c")])
        assert store.count() == 3

    def test_save_chunks_still_upserts(self, store):
        store.save_chunks([_chunk("a_0", "a", "v1")])
        store.save_chunks([_chunk("a_0", "a", "v2")])
        assert store.get_chunk("a_0").text == "v2"

    def test_first_chunk_by_source(self, store):
        assert store.get_first_chunk_by_source("missing") is None
        store.save_chunks([_chunk("a_0", "a")])
        assert store.get_first_chunk_by_source("a").id == "a_0"

    def test_concurrent_writers_are_serialised(self, store):
        errors: list[BaseException] = []

        def worker(n: int) -> None:
            try:
                for i in range(20):
                    store.replace_sources([_chunk(f"s{n}_{i}", f"s{n}")])
                    store.count()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert store.count() == 4


# ---------------------------------------------------------------------------
# Graph reconciliation
# ---------------------------------------------------------------------------

def _graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("topic:t", node_type="topic", origin="ingested", source_ids=["old_0", "other_0"])
    g.add_node("subtopic:s", node_type="subtopic", origin="ingested",
               source_ids=["old_0", "old_1"], parent_topic_keys=["topic:t"])
    g.add_node("content:c", node_type="content", origin="ingested",
               source_ids=["old_1"], parent_subtopic_keys=["subtopic:s"])
    g.add_node("content:keep", node_type="content", origin="ingested",
               source_ids=[], parent_subtopic_keys=["subtopic:s"])
    g.add_node("topic:manual", node_type="topic", origin="manual", source_ids=["old_0"])
    g.add_node("old_0", node_type="chunk_ref")
    g.add_edge("topic:t", "subtopic:s", relation="HAS_SUBTOPIC", source_chunk_ids=[])
    g.add_edge("subtopic:s", "content:c", relation="HAS_CONTENT", source_chunk_ids=["old_1"])
    g.add_edge("subtopic:s", "content:keep", relation="HAS_CONTENT", source_chunk_ids=[])
    g.add_edge("topic:t", "topic:manual", relation="RELATED_TO", label="x",
               source_chunk_ids=["old_0", "other_0"])
    return g


class TestReconcile:
    def test_cleans_evidence_and_removes_orphaned_ingested_nodes(self):
        g = _graph()
        stats = reconcile_graph_sources(g, ["old_0", "old_1"])

        # node with other-source evidence survives with filtered ids
        assert g.nodes["topic:t"]["source_ids"] == ["other_0"]
        # ingested nodes whose evidence became empty are removed
        assert "subtopic:s" not in g and "content:c" not in g
        # manual node preserved even with no evidence left
        assert g.nodes["topic:manual"]["source_ids"] == []
        # node that never had evidence untouched; dangling parent key pruned
        assert "content:keep" in g
        assert g.nodes["content:keep"]["parent_subtopic_keys"] == []
        # legacy chunk_ref removed
        assert "old_0" not in g
        # edge evidence filtered
        data = next(iter(g.get_edge_data(*sorted(("topic:t", "topic:manual"))).values()))
        assert data["source_chunk_ids"] == ["other_0"]
        assert stats["nodes_removed"] == 3

    def test_noop_for_empty_removed_set(self):
        g = _graph()
        assert reconcile_graph_sources(g, []) == {
            "nodes_removed": 0, "nodes_updated": 0, "edges_updated": 0,
        }
        assert g.number_of_nodes() == 6

    def test_manual_connection_survives_last_source_removal(self):
        g = nx.MultiDiGraph()
        for key in ("a", "b", "unsupported"):
            g.add_node(key, node_type="content", origin="ingested", source_ids=["old"], name=key)
        add_association(g, "a", "b", kind="automatic", verified=True, source_chunk_ids=["old"])
        add_association(g, "b", "a", kind="manual")
        stats = reconcile_graph_sources(g, ["old"])
        assert set(g) == {"a", "b"}
        assert all(attrs["source_ids"] == [] and attrs["origin"] == "ingested" for _, attrs in g.nodes(data=True))
        support = g["a"]["b"][association_id("a", "b")]["supports"]
        assert len(support) == 1 and support[0]["kind"] == "manual" and support[0]["verified"] is False
        assert stats["nodes_removed"] == 1

    @pytest.mark.parametrize("with_edges", [True, False])
    def test_note_connection_preserves_endpoint_and_all_hierarchy_ancestors(self, with_edges):
        g = nx.MultiDiGraph()
        for key, node_type, extra in (
            ("topic", "topic", {}),
            ("subtopic", "subtopic", {"parent_topic_keys": ["topic"]}),
            ("content", "content", {"parent_subtopic_keys": ["subtopic"]}),
            ("orphan", "content", {}),
        ):
            g.add_node(key, node_type=node_type, origin="ingested", source_ids=["old"], name=key, **extra)
        g.add_node("note", node_type="note", origin="manual", source_ids=["note:note"])
        if with_edges:
            g.add_edge("topic", "subtopic", key="parent", relation="HAS_SUBTOPIC", source_chunk_ids=["old"])
            g.add_edge("subtopic", "content", key="child", relation="HAS_CONTENT", source_chunk_ids=["old"])
        add_association(g, "content", "note", kind="wikilink", owner="note", source_chunk_ids=["note:note"])
        reconcile_graph_sources(g, ["old"])
        assert set(g) == {"topic", "subtopic", "content", "note"}
        assert g.nodes["content"]["parent_subtopic_keys"] == ["subtopic"]
        assert g.nodes["subtopic"]["parent_topic_keys"] == ["topic"]
        if with_edges:
            assert g.has_edge("topic", "subtopic", "parent") and g.has_edge("subtopic", "content", "child")
            assert g["subtopic"]["content"]["child"]["source_chunk_ids"] == []
        assert g["content"]["note"][association_id("content", "note")]["supports"][0]["kind"] == "wikilink"
        assert all(g.nodes[key]["origin"] == "ingested" for key in ("topic", "subtopic", "content"))

    def test_unknown_legacy_backing_survives_and_unbacked_generated_nodes_are_removed(self):
        g = nx.MultiDiGraph()
        for key in ("a", "b", "c", "d"):
            g.add_node(key, node_type="content", origin="ingested", source_ids=["old"])
        g.add_edge("b", "a", relation="RELATED_TO", label="used_by")
        add_association(g, "c", "d", kind="automatic", verified=True, source_chunk_ids=["old"])
        reconcile_graph_sources(g, ["old"])
        assert set(g) == {"a", "b"}
        attrs = g["a"]["b"][association_id("a", "b")]
        assert attrs["label"] is None
        assert attrs["supports"][0]["kind"] == "legacy" and attrs["supports"][0]["verified"] is False


# ---------------------------------------------------------------------------
# Lock + epoch helpers
# ---------------------------------------------------------------------------

class TestMutationLock:
    def test_run_locked_holds_lock_in_calling_thread(self):
        observed: list[bool] = []

        def probe() -> str:
            t = threading.Thread(
                target=lambda: observed.append(MUTATION_LOCK.acquire(blocking=False))
            )
            t.start()
            t.join()
            return "done"

        assert run_locked(probe) == "done"
        assert observed == [False]
        # released afterwards
        assert MUTATION_LOCK.acquire(blocking=False)
        MUTATION_LOCK.release()

    def test_reentrant(self):
        assert run_locked(lambda: run_locked(lambda: 42)) == 42

    def test_released_on_exception(self):
        with pytest.raises(ValueError):
            run_locked(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert MUTATION_LOCK.acquire(blocking=False)
        MUTATION_LOCK.release()

    def test_epoch_bump(self):
        before = current_epoch()
        assert bump_epoch() == before + 1
        assert current_epoch() == before + 1


class _Payload(BaseModel):
    name: str


def test_locked_endpoint_resolves_string_annotations():
    """@locked must keep FastAPI body parsing working under postponed annotations."""
    app = FastAPI()
    seen: list[bool] = []

    @app.post("/thing")
    @locked
    def make_thing(payload: _Payload) -> dict[str, str]:
        seen.append(MUTATION_LOCK._is_owned())  # type: ignore[attr-defined]
        return {"name": payload.name}

    r = TestClient(app).post("/thing", json={"name": "x"})
    assert r.status_code == 200 and r.json() == {"name": "x"}
    assert seen == [True]
    assert mstate.MUTATION_LOCK is MUTATION_LOCK

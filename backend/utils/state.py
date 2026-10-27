"""
Process-wide mutation coordination.

Every mutation of the shared knowledge state (document store + vector index +
concept graph + raw sources) runs while holding MUTATION_LOCK. The lock is a
threading.RLock, so it is owned by the thread that acquired it:

  - Synchronous code wraps the whole critical section in run_locked() or the
    @locked decorator (FastAPI runs sync endpoints in a worker thread).
  - Async code must NOT acquire the lock on the event loop. It calls
    ``await asyncio.to_thread(run_locked, fn, ...)`` so acquisition, the full
    synchronous body, and release all happen in one worker thread.

A data epoch guards against resurrecting old work after a clear: work
captures current_epoch() when it is admitted, clear calls bump_epoch() under
the lock, and the work re-checks the epoch under the lock before publishing.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import shutil
import threading
import typing
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from backend.utils.resources import nonblocking_model_resources

T = TypeVar("T")
logger = logging.getLogger(__name__)

MUTATION_LOCK = threading.RLock()

_epoch = 0
_revision = 0


def run_locked(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run fn(*args, **kwargs) synchronously while holding MUTATION_LOCK."""
    from backend.workspace import current_workspace
    workspace = current_workspace.get()
    mutation_lock = workspace.mutation_lock if workspace is not None else MUTATION_LOCK
    with mutation_lock, nonblocking_model_resources():
        return fn(*args, **kwargs)


def locked(fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator for synchronous functions/endpoints that mutate shared state."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_locked(fn, *args, **kwargs)

    # Frameworks (FastAPI) resolve string annotations against the wrapper's
    # module globals; publish a signature with annotations already resolved
    # in the wrapped function's own module.
    try:
        hints = typing.get_type_hints(fn)
        sig = inspect.signature(fn)
        wrapper.__signature__ = sig.replace(
            parameters=[
                p.replace(annotation=hints.get(name, p.annotation))
                for name, p in sig.parameters.items()
            ],
            return_annotation=hints.get("return", sig.return_annotation),
        )
    except Exception:  # noqa: BLE001 — unresolved hints: keep wraps() metadata
        pass
    return wrapper


def current_epoch() -> int:
    """Data generation counter; changes every time all data is cleared."""
    from backend.workspace import current_workspace
    workspace = current_workspace.get()
    return workspace.epoch if workspace is not None else _epoch


def bump_epoch() -> int:
    """Invalidate all previously admitted work. Call while holding the lock."""
    global _epoch
    from backend.workspace import current_workspace
    workspace = current_workspace.get()
    if workspace is not None:
        with workspace.mutation_lock:
            workspace.epoch += 1
            return workspace.epoch
    with MUTATION_LOCK:
        _epoch += 1
        return _epoch


def current_revision() -> int:
    """Knowledge content version; read while holding MUTATION_LOCK."""
    from backend.workspace import current_workspace
    workspace = current_workspace.get()
    return workspace.version if workspace is not None else _revision


def bump_revision() -> int:
    """Invalidate detached generation after an upload, clear, or graph edit."""
    global _revision
    from backend.workspace import current_workspace
    workspace = current_workspace.get()
    if workspace is not None:
        with workspace.mutation_lock:
            workspace.version += 1
            return workspace.version
    with MUTATION_LOCK:
        _revision += 1
        return _revision


@contextmanager
def publish_files(files: Iterable[tuple[Path, Path]]):
    """Publish prepared files, restoring their previous versions if the body fails.

    New files and backups live beside each destination, so each replacement is
    atomic on its own filesystem. Call under MUTATION_LOCK and commit SQLite
    inside the context; preparation failures never overwrite an existing file.
    """
    prepared: list[dict] = []
    failed = False
    try:
        for source, destination in files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            suffix = uuid.uuid4().hex
            temporary = destination.with_name(f".{destination.name}.{suffix}.new")
            backup = destination.with_name(f".{destination.name}.{suffix}.bak")
            record = {
                "destination": destination,
                "temporary": temporary,
                "backup": backup,
                "existed": destination.exists(),
                "published": False,
                "restored": False,
            }
            prepared.append(record)
            if record["existed"]:
                shutil.copyfile(destination, backup)
            shutil.copyfile(source, temporary)
        for record in prepared:
            os.replace(record["temporary"], record["destination"])
            record["published"] = True
        yield
    except BaseException as original_error:
        failed = True
        rollback_errors = []
        for record in reversed(prepared):
            if record["published"]:
                try:
                    if record["existed"]:
                        os.replace(record["backup"], record["destination"])
                    else:
                        record["destination"].unlink(missing_ok=True)
                    record["restored"] = True
                except OSError as error:
                    rollback_errors.append(error)
                    logger.exception("Could not restore %s; backup retained at %s", record["destination"], record["backup"])
        if rollback_errors:
            raise RuntimeError("File publication failed and some backups could not be restored; see server logs.") from original_error
        raise
    finally:
        for record in prepared:
            for path in (
                record["temporary"],
                *([record["backup"]] if not failed or not record["published"] or record["restored"] else []),
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove publication temporary file %s", path)


# ---------------------------------------------------------------------------
# Graph source reconciliation (operates on the raw networkx graph so the
# graph modules themselves stay untouched)
# ---------------------------------------------------------------------------

def reconcile_graph_sources(graph, removed_chunk_ids: Iterable[str]) -> dict[str, int]:
    """
    Drop references to chunks that no longer exist after a source replacement.

      - node ``source_ids`` / edge ``source_chunk_ids`` lose the removed IDs
      - a non-manual node whose evidence becomes empty (it had evidence before)
        is removed unless a surviving association needs it or its hierarchy
        branch; manual nodes and nodes with other evidence are preserved
      - legacy chunk_ref nodes keyed by a removed chunk ID are removed
      - parent_*_keys pointing at removed nodes are pruned

    Returns counts: {"nodes_removed", "nodes_updated", "edges_updated"}.
    Caller must hold MUTATION_LOCK and persist the graph afterwards.
    """
    removed = set(removed_chunk_ids)
    stats = {"nodes_removed": 0, "nodes_updated": 0, "edges_updated": 0}
    if not removed:
        return stats

    from backend.utils.associations import normalize_associations, prune_association_supports

    normalize_associations(graph)
    stats["edges_updated"] = prune_association_supports(graph, removed)

    doomed: set[str] = set()
    removed_refs: set[str] = set()
    parents_by_child: dict[str, set[str]] = {}
    for key, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "chunk_ref":
            if key in removed:
                removed_refs.add(key)
            continue
        for field in ("parent_topic_keys", "parent_subtopic_keys"):
            for parent in attrs.get(field, []):
                if parent in graph:
                    parents_by_child.setdefault(key, set()).add(parent)
        ids = attrs.get("source_ids") or []
        kept = [i for i in ids if i not in removed]
        if len(kept) == len(ids):
            continue
        attrs["source_ids"] = kept
        stats["nodes_updated"] += 1
        if not kept and attrs.get("origin") != "manual":
            doomed.add(key)

    retained: set[str] = set()
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("relation") == "RELATED_TO":
            # Source pruning already removed unsupported backings. Every
            # remaining canonical association needs both of its concept nodes.
            retained.update((source, target))
            continue  # Independent association backings are reconciled above.
        if attrs.get("relation") in {"HAS_SUBTOPIC", "HAS_CONTENT"}:
            parents_by_child.setdefault(target, set()).add(source)
        ids = attrs.get("source_chunk_ids") or []
        kept = [i for i in ids if i not in removed]
        if len(kept) != len(ids):
            attrs["source_chunk_ids"] = kept
            stats["edges_updated"] += 1

    # Traverse reverse hierarchy once, including polyhierarchy and cycles.
    # Keeping one endpoint must not leave its visible branch without ancestors.
    retained.difference_update(removed_refs)
    pending = list(retained)
    while pending:
        child = pending.pop()
        for parent in parents_by_child.get(child, ()):
            if parent not in retained and parent not in removed_refs:
                retained.add(parent)
                pending.append(parent)
    doomed.difference_update(retained)
    doomed.update(removed_refs)

    if doomed:
        graph.remove_nodes_from(doomed)
        gone = set(doomed)
        for _key, attrs in graph.nodes(data=True):
            for field in ("parent_topic_keys", "parent_subtopic_keys"):
                parents = attrs.get(field)
                if parents and gone.intersection(parents):
                    attrs[field] = [p for p in parents if p not in gone]
    stats["nodes_removed"] = len(doomed)
    return stats

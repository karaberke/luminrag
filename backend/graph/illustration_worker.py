"""
Illustration resolver — deterministic, source-grounded asset pipeline.

Replaces the previous ComfyUI / A1111 text-to-image approach, which could not
reliably render equations, formulas, or precise technical diagrams.

Routing by illustration kind
-----------------------------
  equation   → retain the LaTeX hint for browser KaTeX rendering
  diagram /
  image      → find an image already extracted from the source document
               (slide JPEG rendered during ingestion, or embedded PDF image)
  code /
  fallback   → illustration_path left null; no generation attempted

Invoked in two places (unchanged from prior version):
  - backend/main.py lifespan — scan for pending nodes on startup
  - backend/main.py /api/ingest — enqueue freshly synthesised content nodes
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from backend.utils.state import MUTATION_LOCK

if TYPE_CHECKING:
    from backend.db.document_store import DocumentStore
    from backend.graph.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

_ILLUSTRATIONS_SUBDIR = "illustrations"

def illustrations_dir(processed_dir: Path) -> Path:
    path = processed_dir / _ILLUSTRATIONS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Source-image resolver
# ---------------------------------------------------------------------------

# Formats browsers render in <img>. PDFs and other documents are never returned.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


def _is_image_file(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    p = Path(path)
    return p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file()


def _evidence_chunk_ids(node_key: str, builder: "GraphBuilder") -> list[str]:
    """
    Chunk IDs backing *node_key*: its `source_ids` first, then (legacy graphs
    only) the keys of chunk_ref predecessor nodes, which were chunk IDs.
    """
    with MUTATION_LOCK:
        g = builder.graph
        if node_key not in g:
            return []
        ids = list(g.nodes[node_key].get("source_ids") or [])
        for pred_key in g.predecessors(node_key):
            if g.nodes[pred_key].get("node_type") == "chunk_ref" and pred_key not in ids:
                ids.append(pred_key)
    return ids


def _find_source_image(
    node_key: str,
    builder: "GraphBuilder",
    doc_store: "DocumentStore",
) -> str | None:
    """
    Return the first existing, browser-renderable image associated with the
    chunks that back *node_key*. Blocking (DB + filesystem) — call via
    asyncio.to_thread from async code.

    Checks chunk metadata keys in order:
      keyframe_path     — video keyframe
      slide_image_path  — JPEG rendered by slide_processor (one per page)
      image_paths       — images extracted from embedded PDF images
      source_file       — only when the uploaded file itself is an image
    """
    for chunk_id in _evidence_chunk_ids(node_key, builder):
        chunk = doc_store.get_chunk(chunk_id)
        if chunk is None:
            continue
        meta = chunk.metadata or {}
        candidates = [meta.get("keyframe_path"), meta.get("slide_image_path")]
        candidates += list(meta.get("image_paths") or [])
        candidates.append(meta.get("source_file"))
        for path in candidates:
            if _is_image_file(path):
                return path
    return None


# ---------------------------------------------------------------------------
# Top-level resolver (replaces generate_illustration)
# ---------------------------------------------------------------------------

async def resolve_illustration(
    node_key: str,
    hint: str,
    kind: str,
    output_dir: Path,
    config: dict,
    builder: "GraphBuilder",
    doc_store: "DocumentStore",
) -> Path | None:
    """
    Resolve the illustration for a content node deterministically.
    Returns the Path to an image on disk, or None if no asset is available.
    """
    cfg = config.get("illustration_worker", {})
    if not cfg.get("enabled", True):
        return None

    # All rendering, DB and filesystem work runs off the event loop.
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    # A render from before a reset must never overwrite a newer node's image.
    safe = uuid.uuid4().hex

    if kind == "equation":
        return None  # Rendered from its original hint by RichText/KaTeX.

    if kind in ("diagram", "image"):
        src = await asyncio.to_thread(_find_source_image, node_key, builder, doc_store)
        if not src:
            return None
        src_path = Path(src)
        out_path = output_dir / f"{safe}{src_path.suffix.lower()}"
        await asyncio.to_thread(_copy_if_missing, src_path, out_path)
        return out_path

    # kind == "code" or any unrecognised kind — no illustration
    return None


def _copy_if_missing(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Background queue + consumer
# ---------------------------------------------------------------------------

class IllustrationScheduler:
    """
    Owns a single asyncio.Queue of (generation, node_key, hint, kind) tuples
    and a single consumer task that processes them one at a time.  The
    consumer writes each resolved path back onto the graph via
    set_illustration_path and persists the graph.

    Every graph read/write/save happens in a worker thread while holding the
    shared MUTATION_LOCK, so the event loop never blocks on the lock and
    ingestion/clear cannot interleave with a write.

    After the graph is cleared or the builder replaced, call reset(): it
    bumps the generation token so queued and in-flight work for the old graph
    is discarded instead of being written (or saved) onto it.
    """

    def __init__(
        self,
        builder: "GraphBuilder",
        doc_store: "DocumentStore",
        output_dir: Path,
        config: dict,
        static_url_prefix: str = "/static/illustrations",
    ) -> None:
        self._builder = builder
        self._doc_store = doc_store
        self._output_dir = output_dir
        self._config = config
        self._url_prefix = static_url_prefix.rstrip("/")
        self._queue: asyncio.Queue[tuple[int, str, str, str] | None] = asyncio.Queue()
        self._seen: set[str] = set()
        # Nodes that already failed this process — blacklisted until reset/restart
        # so a single bad hint cannot re-queue endlessly via scan_and_enqueue_pending.
        self._failed: set[str] = set()
        self._generation = 0
        self._suspended = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._loop = asyncio.get_running_loop()
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="illustration-worker")

    def reset(
        self,
        builder: "GraphBuilder",
        doc_store: "DocumentStore | None" = None,
    ) -> None:
        """
        Rebind to *builder* (may be the same object after an in-place clear)
        and invalidate all pending work. Synchronous and callable from any
        thread (e.g. a sync FastAPI endpoint); safe to call while already
        holding MUTATION_LOCK since it is re-entrant.
        """
        with MUTATION_LOCK:
            self._generation += 1
            self._suspended = False
            self._builder = builder
            if doc_store is not None:
                self._doc_store = doc_store
            self._seen.clear()
            self._failed.clear()

        # asyncio.Queue is not thread-safe: drain on the loop thread. Stale
        # items that slip through are dropped by the generation check anyway.
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._drain_queue()
        else:
            loop.call_soon_threadsafe(self._drain_queue)

    def suspend(self) -> None:
        """Fence background writes while storage's committed state is uncertain.

        No graph is saved or rebound here. Only an explicit reset to a verified
        builder can resume work; already-running renders are discarded at commit.
        """
        with MUTATION_LOCK:
            self._suspended = True
            self._generation += 1

    def _drain_queue(self, keep_current: bool = True) -> None:
        current = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if keep_current and (item is None or item[0] == self._generation):
                current.append(item)
        for item in current:
            self._queue.put_nowait(item)

    async def stop(self) -> None:
        """Finish active render/persistence before storage can be cleared or closed."""
        self._stop.set()
        task = self._task
        if task is not None:
            # Wake an idle consumer. Cancelling it would abandon to_thread()
            # work, which can keep writing files after stop() returns.
            self._queue.put_nowait(None)
            try:
                await asyncio.shield(task)
            except (asyncio.CancelledError, Exception):
                if not task.done():
                    await task
            self._task = None
        await asyncio.to_thread(self._invalidate_stopped_work)
        self._drain_queue(keep_current=False)

    def _invalidate_stopped_work(self) -> None:
        with MUTATION_LOCK:
            self._generation += 1
            self._seen.clear()
            self._failed.clear()

    def _reserve(self, node_key: str, hint: str, generation: int | None) -> int | None:
        with MUTATION_LOCK:
            generation = self._generation if generation is None else generation
            if self._suspended or self._stop.is_set() or generation != self._generation or not hint or node_key in self._seen or node_key in self._failed:
                return None
            self._seen.add(node_key)
            return generation

    async def enqueue(self, node_key: str, hint: str, kind: str, *, generation: int | None = None) -> None:
        reserved = await asyncio.to_thread(self._reserve, node_key, hint, generation)
        if reserved is not None and not self._stop.is_set() and reserved == self._generation:
            self._queue.put_nowait((reserved, node_key, hint, kind))

    async def scan_and_enqueue_pending(self) -> int:
        """Find content nodes that have an illustration hint but no path yet."""
        pending = 0
        generation, items = await asyncio.to_thread(self._pending_snapshot)
        for key, hint, kind in items:
            if generation != self._generation:
                break
            await self.enqueue(key, hint, kind, generation=generation)
            pending += 1
        return pending

    def _pending_snapshot(self) -> tuple[int, list[tuple[str, str, str]]]:
        with MUTATION_LOCK:
            if self._suspended:
                return self._generation, []
            pending: list[tuple[str, str, str]] = []
            for key, attrs in self._builder.graph.nodes(data=True):
                if attrs.get("node_type") != "content":
                    continue
                if attrs.get("illustration_path"):
                    continue
                if key in self._failed:
                    continue
                illus = attrs.get("illustration")
                if not illus or not illus.get("hint"):
                    continue
                pending.append((key, illus["hint"], illus.get("kind", "image")))
            return self._generation, pending

    def _needs_illustration(self, generation: int, node_key: str) -> bool:
        with MUTATION_LOCK:
            if self._suspended or generation != self._generation:
                return False
            g = self._builder.graph
            return g.has_node(node_key) and not g.nodes[node_key].get("illustration_path")

    def _commit(self, generation: int, builder: "GraphBuilder", node_key: str, out: Path) -> None:
        """Write + persist the path unless the graph was reset since dequeue."""
        with MUTATION_LOCK:
            if self._suspended or generation != self._generation or builder is not self._builder:
                logger.info(f"Discarding stale illustration for {node_key}")
                out.unlink(missing_ok=True)
                return
            url = f"{self._url_prefix}/{out.name}"
            if builder.set_illustration_path(node_key, url):
                try:
                    builder.save()
                except OSError as exc:
                    logger.warning(f"Failed to persist illustration path: {exc}")

    def _mark_failed(self, generation: int, node_key: str) -> None:
        with MUTATION_LOCK:
            if generation == self._generation:
                self._failed.add(node_key)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break
            if item is None or self._stop.is_set():
                self._queue.task_done()
                break
            generation, node_key, hint, kind = item
            try:
                if not await asyncio.to_thread(self._needs_illustration, generation, node_key):
                    continue
                builder = self._builder
                out = await resolve_illustration(
                    node_key,
                    hint,
                    kind,
                    self._output_dir,
                    self._config,
                    builder,
                    self._doc_store,
                )
                if out is None:
                    await asyncio.to_thread(self._mark_failed, generation, node_key)
                    continue
                await asyncio.to_thread(self._commit, generation, builder, node_key, out)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Illustration worker error on {node_key}: {exc}")
                await asyncio.to_thread(self._mark_failed, generation, node_key)
            finally:
                self._queue.task_done()

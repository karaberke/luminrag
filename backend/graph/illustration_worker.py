"""
Illustration resolver — deterministic, source-grounded asset pipeline.

Replaces the previous ComfyUI / A1111 text-to-image approach, which could not
reliably render equations, formulas, or precise technical diagrams.

Routing by illustration kind
-----------------------------
  equation   → render the LaTeX hint to a PNG using matplotlib.mathtext
               (no external TeX installation required)
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
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

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
# Equation renderer
# ---------------------------------------------------------------------------

def _clean_latex_hint(hint: str) -> str:
    """
    Normalise an LLM-emitted LaTeX hint before passing it to matplotlib mathtext.

    Problems addressed:
    - JSON-mangled backslashes: json.loads() turns \\theta → tab+"heta" (chr 9),
      \\frac → form-feed+"rac" (chr 12), \\boldsymbol → backspace+"oldsymbol" (chr 8).
    - Double-backslash commands: \\cos → \cos (LLM double-escaped in JSON).
    - Python-style Unicode escapes: Λ → actual character (e.g. Λ).
    - \textbf → \mathbf (matplotlib uses math-mode bold, not text-mode bold).
    - Outer delimiter mismatch: strip \\[...\\], \[...\], \(...\), $$...$$.
    """
    s = hint.strip()
    # Restore control characters introduced by JSON parsing of bare-backslash LaTeX
    s = (s
         .replace('\x08', r'\b')   # \b (backspace)  → e.g. \boldsymbol
         .replace('\x09', r'\t')   # \t (tab)          → e.g. \theta, \text
         .replace('\x0c', r'\f')   # \f (form feed)   → e.g. \frac
         .replace('\x0d', r'\r')   # \r (CR)           → e.g. \rho
    )
    # Collapse double-backslash commands (\\cos → \cos) from JSON double-escaping
    s = re.sub(r'\\\\([a-zA-Z])', r'\\\1', s)
    # Decode Python-style Unicode escapes left as literal text (Λ → Λ)
    s = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)
    # matplotlib uses \mathbf, not \textbf
    s = s.replace(r'\textbf', r'\mathbf')
    # Strip outer display/inline-math delimiters not understood by mathtext
    for pat in (
        r'^\s*\\\\\[(.*)\\\\\]\s*$',  # \\[...\\]
        r'^\s*\\\[(.*)\\\]\s*$',       # \[...\]
        r'^\s*\\\((.*)\\\)\s*$',       # \(...\)
        r'^\s*\$\$(.*)\$\$\s*$',       # $$...$$
    ):
        m = re.match(pat, s, re.DOTALL)
        if m:
            return m.group(1).strip()
    return s.strip('$')


def _render_equation(hint: str, out_path: Path, dpi: int = 150) -> bool:
    """
    Render a LaTeX hint to a PNG using matplotlib.mathtext.

    Uses matplotlib's built-in math parser — no TeX installation required.
    Returns False if mathtext cannot parse the expression (unsupported syntax).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latex = _clean_latex_hint(hint)
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.text(
        0.5, 0.5,
        f"${latex}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="#1a1a2e",
    )
    ax.axis("off")
    fig.patch.set_facecolor("white")
    try:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        return True
    except ValueError as exc:
        # matplotlib mathtext parse failure — unsupported LaTeX constructs
        logger.warning(f"mathtext could not render '{latex[:60]}': {exc}")
        return False
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Source-image resolver
# ---------------------------------------------------------------------------

def _find_source_image(
    node_key: str,
    builder: "GraphBuilder",
    doc_store: "DocumentStore",
) -> str | None:
    """
    Walk EVIDENCE_OF predecessor (ChunkRef) nodes for *node_key* and return
    the path of the first real on-disk image found in their chunk metadata.

    Checks two metadata keys:
      slide_image_path  — JPEG rendered by slide_processor (one per page)
      image_paths       — list of PNGs extracted from embedded PDF images
    """
    g = builder.graph
    for pred_key in g.predecessors(node_key):
        if g.nodes[pred_key].get("node_type") != "chunk_ref":
            continue
        chunk = doc_store.get_chunk(pred_key)  # ChunkRef key IS the chunk_id
        if chunk is None:
            continue
        slide_path = chunk.metadata.get("slide_image_path")
        if slide_path and Path(slide_path).exists():
            return slide_path
        for p in chunk.metadata.get("image_paths", []):
            if Path(p).exists():
                return p
        source_file = chunk.metadata.get("source_file")
        if source_file and Path(source_file).exists():
            return source_file
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

    output_dir.mkdir(parents=True, exist_ok=True)

    if kind == "equation":
        safe = node_key.replace(":", "_").replace("/", "_")
        out_path = output_dir / f"{safe}.png"
        dpi = int(cfg.get("latex_dpi", 150))
        ok = _render_equation(hint, out_path, dpi)
        return out_path if ok and out_path.exists() else None

    if kind in ("diagram", "image"):
        src = _find_source_image(node_key, builder, doc_store)
        if not src:
            return None
        src_path = Path(src)
        safe = node_key.replace(":", "_").replace("/", "_")
        out_path = output_dir / f"{safe}{src_path.suffix}"
        if not out_path.exists():
            shutil.copy2(src_path, out_path)
        return out_path

    # kind == "code" or any unrecognised kind — no illustration
    return None


# ---------------------------------------------------------------------------
# Background queue + consumer
# ---------------------------------------------------------------------------

class IllustrationScheduler:
    """
    Owns a single asyncio.Queue of (node_key, hint, kind) tuples and a
    single consumer task that processes them one at a time.  The consumer
    writes each resolved path back onto the graph via set_illustration_path
    and persists the graph.
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
        self._queue: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()
        self._seen: set[str] = set()
        # Nodes that already failed this process — blacklisted until restart
        # so a single bad hint cannot re-queue endlessly via scan_and_enqueue_pending.
        self._failed: set[str] = set()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="illustration-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def enqueue(self, node_key: str, hint: str, kind: str) -> None:
        if not hint or node_key in self._seen or node_key in self._failed:
            return
        self._seen.add(node_key)
        await self._queue.put((node_key, hint, kind))

    async def scan_and_enqueue_pending(self) -> int:
        """Find content nodes that have an illustration hint but no path yet."""
        pending = 0
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
            await self.enqueue(key, illus["hint"], illus.get("kind", "image"))
            pending += 1
        return pending

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                node_key, hint, kind = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                if self._builder.graph.has_node(node_key) and self._builder.graph.nodes[
                    node_key
                ].get("illustration_path"):
                    continue
                out = await resolve_illustration(
                    node_key,
                    hint,
                    kind,
                    self._output_dir,
                    self._config,
                    self._builder,
                    self._doc_store,
                )
                if out is None:
                    self._failed.add(node_key)
                    continue
                url = f"{self._url_prefix}/{out.name}"
                if self._builder.set_illustration_path(node_key, url):
                    try:
                        self._builder.save()
                    except OSError as exc:
                        logger.warning(f"Failed to persist illustration path: {exc}")
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Illustration worker error on {node_key}: {exc}")
                self._failed.add(node_key)
            finally:
                self._queue.task_done()

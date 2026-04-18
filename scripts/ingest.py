"""
Unified ingestion CLI for LuminRAG.

Usage:
    python scripts/ingest.py                          # uses backend/data/raw/ by default
    python scripts/ingest.py path/to/other/folder
    python scripts/ingest.py --dry-run                # preview without running

Walks the given directory recursively, dispatches each file to the
correct processor by extension, then runs the full downstream pipeline:
  1. Ingest files  → list[Chunk]
  2. Save chunks   → DocumentStore (SQLite)
  3. Build index   → VectorIndex (FAISS)
  4. Build graph   → GraphBuilder + export_graph

Prints a summary when done.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `backend.*` imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# Load .env from repo root (no python-dotenv needed)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → processor dispatch table
# ---------------------------------------------------------------------------

_VIDEO_EXT   = {".mp4", ".mkv", ".mov", ".avi"}
_AUDIO_EXT   = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
_IMAGE_EXT   = {".jpg", ".jpeg", ".png", ".webp"}
_PDF_EXT     = {".pdf"}


def _load_config() -> dict:
    root = Path(__file__).parent.parent
    with open(root / "config" / "llm.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(root / "config" / "db.yaml") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def _collect_files(directory: Path) -> list[Path]:
    all_ext = _VIDEO_EXT | _AUDIO_EXT | _IMAGE_EXT | _PDF_EXT
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in all_ext
    )


def _ingest_file(
    path: Path,
    config: dict,
    slide_names: set[str],
) -> list:
    """Return chunks for one file, dispatching by extension."""
    ext = path.suffix.lower()

    if ext in _VIDEO_EXT:
        from backend.ingestion.video_transcriber import transcribe_video
        logger.info(f"  [video]  {path.name}")
        return transcribe_video(path, config)

    if ext in _AUDIO_EXT:
        from backend.ingestion.audio_processor import process_audio
        logger.info(f"  [audio]  {path.name}")
        return process_audio(path, config)

    if ext in _IMAGE_EXT:
        from backend.ingestion.image_processor import process_image
        logger.info(f"  [image]  {path.name}")
        return process_image(path, config)

    if ext in _PDF_EXT:
        if path.name in slide_names:
            from backend.ingestion.slide_processor import process_slides
            logger.info(f"  [slides] {path.name}")
            return process_slides(path, config)
        else:
            from backend.ingestion.pdf_processor import process_pdf
            logger.info(f"  [pdf]    {path.name}")
            return process_pdf(path, config)

    return []  # unreachable — only known extensions collected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest course files into LuminRAG.")
    default_dir = Path(__file__).parent.parent / "backend" / "data" / "raw"
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=default_dir,
        help=f"Directory to scan for course files (default: {default_dir})",
    )
    parser.add_argument(
        "--slides",
        nargs="*",
        metavar="FILENAME",
        default=[],
        help="PDF file names (basename only) to treat as slide decks instead of text PDFs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dispatch plan without running any processors.",
    )
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        sys.exit(f"Error: '{directory}' is not a directory.")

    slide_names: set[str] = set(args.slides)

    # ── Collect files ────────────────────────────────────────────────────────
    files = _collect_files(directory)
    if not files:
        logger.warning(f"No supported files found under {directory}")
        return

    logger.info(f"Found {len(files)} file(s) under {directory}")

    if args.dry_run:
        all_ext = _VIDEO_EXT | _AUDIO_EXT | _IMAGE_EXT | _PDF_EXT
        for p in files:
            ext = p.suffix.lower()
            if ext in _VIDEO_EXT:     kind = "video"
            elif ext in _AUDIO_EXT:   kind = "audio"
            elif ext in _IMAGE_EXT:   kind = "image"
            elif p.name in slide_names: kind = "slides (pdf)"
            else:                      kind = "pdf"
            print(f"  [{kind:10s}] {p.relative_to(directory)}")
        return

    config = _load_config()

    # ── Wipe existing databases so each ingest run starts clean ─────────────
    root = Path(__file__).parent.parent
    store_path   = Path(config["document_store"]["path"])
    index_path   = Path(config["vector_db"]["path"])
    graph_path   = Path(config["graph_db"]["path"])
    export_path  = graph_path.with_name(graph_path.stem + "_export.json")

    for p in (
        store_path,
        store_path.with_suffix(".db-wal"),
        store_path.with_suffix(".db-shm"),
        index_path,
        index_path.with_suffix(".json"),
        graph_path,
        export_path,
    ):
        if p.exists():
            p.unlink()
            logger.info(f"Removed existing {p.relative_to(root)}")

    # Ensure the output directory exists
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Ingest all files ────────────────────────────────────────────
    all_chunks: list = []
    failed: list[str] = []

    for path in files:
        try:
            chunks = _ingest_file(path, config, slide_names)
            all_chunks.extend(chunks)
            logger.info(f"             → {len(chunks)} chunk(s)")
        except Exception as exc:
            logger.error(f"  FAILED {path.name}: {exc}")
            failed.append(path.name)

    if not all_chunks:
        sys.exit("No chunks produced — nothing to index.")

    logger.info(f"\nTotal chunks: {len(all_chunks)}")

    # ── Stage 2: Save to DocumentStore ──────────────────────────────────────
    from backend.db.document_store import DocumentStore
    logger.info(f"\nSaving chunks to document store ({store_path})…")
    store = DocumentStore(store_path)
    store.save_chunks(all_chunks)
    logger.info(f"Document store now has {store.count()} chunk(s).")

    # ── Stage 3: Build FAISS vector index ───────────────────────────────────
    from backend.retrieval.embedder import Embedder
    from backend.retrieval.vector_retriever import VectorIndex
    logger.info(f"\nBuilding vector index ({config['vector_retriever']['model']})…")
    embedder = Embedder(config["vector_retriever"]["model"])
    index = VectorIndex(embedder)
    index.build(all_chunks)
    index.save(index_path)
    logger.info(f"Vector index saved to {index_path}.")

    # ── Stage 4: Build concept graph ─────────────────────────────────────────
    from backend.graph.entity_extractor import extract_triples
    from backend.graph.graph_builder import GraphBuilder
    from backend.graph.graph_export import export_graph
    logger.info("\nExtracting concept triples…")
    triples = extract_triples(all_chunks, config)
    logger.info(f"{len(triples)} triple(s) extracted.")

    logger.info("Building concept graph…")
    builder = GraphBuilder(graph_path)
    builder.add_triples(triples)
    builder.save()

    logger.info(f"Exporting graph to {export_path}…")
    export_graph(builder.graph, export_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Ingestion complete")
    print("=" * 50)
    print(f"  Files processed : {len(files) - len(failed)}/{len(files)}")
    if failed:
        print(f"  Failed files    : {', '.join(failed)}")
    print(f"  Chunks produced : {len(all_chunks)}")
    print(f"  Triples         : {len(triples)}")
    print(f"  Graph nodes     : {builder.graph.number_of_nodes()}")
    print(f"  Graph edges     : {builder.graph.number_of_edges()}")
    print(f"  Document store  : {store_path}")
    print(f"  Vector index    : {index_path}")
    print(f"  Graph JSON      : {export_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

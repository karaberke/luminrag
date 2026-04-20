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

    # ── Stage 4: Build hierarchical concept graph ──────────────────────────
    from backend.graph.graph_builder import GraphBuilder
    from backend.graph.graph_export import export_graph
    from backend.graph.schema import ChunkLocator, ChunkRefAttrs, ProposalBundle
    from backend.graph.topic_extractor import extract_topics
    from backend.graph.subtopic_extractor import extract_subtopics
    from backend.graph.content_synthesizer import synthesize_contents
    from backend.graph.semantic_linker import link_nodes, build_linkables_from_graph

    builder = GraphBuilder(graph_path)

    # Build a short preview for topic inference
    preview_chars = 3000
    preview = "Source files: " + ", ".join(p.name for p in files) + "\n\n"
    preview += "\n\n".join(c.text for c in all_chunks if c.text.strip())
    preview = preview[:preview_chars]

    # Build ChunkRef nodes so Content → ChunkRef edges resolve
    def _chunk_ref(c) -> ChunkRefAttrs:
        meta = c.metadata or {}
        src = Path(c.source_id).name if c.source_id else "source"
        if c.modality in ("slide", "pdf") and (meta.get("page_number") or meta.get("page_start")):
            name = f"Page {meta.get('page_number') or meta.get('page_start')} of {src}"
        elif c.modality in ("video", "audio") and meta.get("start_time") is not None:
            s = int(meta['start_time'])
            name = f"{s // 60}:{s % 60:02d} in {src}"
        else:
            name = src
        return ChunkRefAttrs(
            chunk_id=c.id,
            name=name,
            modality=c.modality,
            source_id=c.source_id,
            locator=ChunkLocator(
                page=meta.get("page_number") or meta.get("page_start"),
                start_time=meta.get("start_time"),
                end_time=meta.get("end_time"),
            ),
        )

    logger.info("\nStage 1.2 — Topic inference …")
    topic_proposals = extract_topics(
        preview, [c.id for c in all_chunks], config,
        existing_names=[], embedder=embedder,
    )
    topic_names = [tp.name for tp in topic_proposals]
    logger.info(f"  {len(topic_proposals)} topic(s): {topic_names}")

    logger.info("Stage 1.3 — Subtopic mapping …")
    subtopic_proposals = extract_subtopics(
        all_chunks, topic_names, config, existing_names=[], embedder=embedder,
    )
    logger.info(f"  {len(subtopic_proposals)} subtopic(s)")

    subs_by_chunk: dict[str, list[str]] = {}
    for sp in subtopic_proposals:
        for cid in sp.source_chunk_ids:
            subs_by_chunk.setdefault(cid, []).append(sp.name)

    logger.info("Stage 1.4 — Content synthesis …")
    content_proposals = []
    for chunk in all_chunks:
        content_proposals.extend(
            synthesize_contents(
                chunk, subs_by_chunk.get(chunk.id, []), topic_names,
                config, existing_titles=[], embedder=embedder,
            )
        )
    logger.info(f"  {len(content_proposals)} content unit(s)")

    first_bundle = ProposalBundle(
        chunk_refs=[_chunk_ref(c) for c in all_chunks],
        topics=topic_proposals,
        subtopics=subtopic_proposals,
        contents=content_proposals,
    )
    apply_result = builder.apply_proposals(first_bundle)

    logger.info("Stage 1.5 — Semantic linking …")
    linkables = build_linkables_from_graph(builder.graph)
    related_proposals = link_nodes(linkables, embedder, config)
    logger.info(f"  {len(related_proposals)} RELATED_TO edge(s)")

    if related_proposals:
        related_result = builder.apply_proposals(ProposalBundle(related=related_proposals))
        apply_result.related_edges_added += related_result.related_edges_added

    builder.save()
    logger.info(f"Exporting graph to {export_path}…")
    export_graph(builder.graph, export_path)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Ingestion complete")
    print("=" * 50)
    print(f"  Files processed     : {len(files) - len(failed)}/{len(files)}")
    if failed:
        print(f"  Failed files        : {', '.join(failed)}")
    print(f"  Chunks produced     : {len(all_chunks)}")
    print(f"  Topics added        : {apply_result.topics_added}")
    print(f"  Subtopics added     : {apply_result.subtopics_added}")
    print(f"  Contents added      : {apply_result.contents_added}")
    print(f"  RELATED_TO edges    : {apply_result.related_edges_added}")
    print(f"  Graph nodes         : {builder.graph.number_of_nodes()}")
    print(f"  Graph edges         : {builder.graph.number_of_edges()}")
    print(f"  Document store      : {store_path}")
    print(f"  Vector index        : {index_path}")
    print(f"  Graph JSON          : {export_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

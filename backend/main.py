"""
LuminRAG Backend — FastAPI application.

Wires together all pipeline stages and exposes three endpoints:
  POST /api/query  — main QA pipeline
  GET  /api/graph  — concept graph for frontend visualization
  GET  /api/health — liveness check

Run with:
  python -m backend.main
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

# Load .env from repo root before any SDK imports so API keys are available
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import shutil

import boto3
import yaml
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.db.document_store import DocumentStore
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_retriever import VectorIndex
from backend.retrieval.graph_retriever import warm_up as warm_graph_retriever
from backend.graph.graph_builder import GraphBuilder
from backend.graph.graph_export import export_graph
from backend.graph.illustration_worker import IllustrationScheduler, illustrations_dir
from backend.graph.schema import (
    STRUCTURAL_RELATIONS,
    ContentAttrs,
    Illustration,
    ProposalBundle,
    SubtopicAttrs,
    TopicAttrs,
    make_key,
    normalise_relation_label,
)
from backend.self_rag.multi_hop_reasoner import reason
from backend.generation.generator import generate
from backend.schemas import Chunk, GenerationResult, RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Merge llm.yaml and db.yaml into a single config dict."""
    root = Path(__file__).parent.parent
    with open(root / "config" / "llm.yaml", encoding="utf-8") as f:
        llm = yaml.safe_load(f) or {}
    with open(root / "config" / "db.yaml", encoding="utf-8") as f:
        db = yaml.safe_load(f) or {}
    return {**llm, **db}


# ---------------------------------------------------------------------------
# Application state (singletons initialized at startup)
# ---------------------------------------------------------------------------

class _AppState:
    config: dict = {}
    store: DocumentStore | None = None
    embedder: Embedder | None = None
    vector_index: VectorIndex | None = None
    graph_builder: GraphBuilder | None = None
    illustration_worker: IllustrationScheduler | None = None
    index_loaded: bool = False
    graph_loaded: bool = False


state = _AppState()


def _warm_ollama(config: dict) -> None:
    """
    Send a minimal prompt to Ollama so the model is loaded into RAM before
    the first real user query. Skipped silently if provider is not ollama or
    if Ollama is unreachable.
    """
    ollama_sections = [
        config.get("generator", {}),
        config.get("query_router", {}),
        config.get("self_rag", {}),
        config.get("multi_hop_reasoner", {}),
    ]
    ollama_models = {
        s.get("ollama_model") or s.get("llm_ollama_model")
        for s in ollama_sections
        if (s.get("provider") == "ollama" or s.get("llm_provider") == "ollama")
    }
    ollama_models.discard(None)

    if not ollama_models:
        return  # no section uses ollama

    import httpx
    # Use the base URL from the first ollama section found
    base_url = next(
        (
            s.get("ollama_base_url") or s.get("llm_ollama_base_url", "http://localhost:11434")
            for s in ollama_sections
            if (s.get("provider") == "ollama" or s.get("llm_provider") == "ollama")
        ),
        "http://localhost:11434",
    )

    for model in ollama_models:
        try:
            logger.info(f"Warming up Ollama model '{model}'…")
            httpx.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}},
                timeout=60,
            ).raise_for_status()
            logger.info(f"Ollama model '{model}' ready.")
        except Exception as exc:
            logger.warning(f"Ollama warmup failed for '{model}': {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all singletons on startup; clean up on shutdown."""
    logger.info("LuminRAG starting up …")
    state.config = _load_config()

    # Document store
    db_path = state.config.get("document_store", {}).get("path", "backend/data/processed/chunks.db")
    state.store = DocumentStore(db_path)
    logger.info(f"DocumentStore: {db_path} ({state.store.count()} chunks)")

    # Embedder (shared across retrieval modules)
    model_name = state.config.get("vector_retriever", {}).get("model", "all-MiniLM-L6-v2")
    state.embedder = Embedder(model_name)

    # Vector index — load if exists, otherwise mark unavailable
    index_path = Path(
        state.config.get("vector_db", {}).get("path", "backend/data/processed/faiss.index")
    )
    state.vector_index = VectorIndex(state.embedder)
    if index_path.exists():
        state.vector_index.load(index_path)
        state.index_loaded = True
        logger.info(f"VectorIndex loaded from {index_path}")
    else:
        logger.warning(f"VectorIndex not found at {index_path} — dense retrieval unavailable")

    # Graph builder — constructor already loads from disk if the file exists
    graph_path = state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    state.graph_builder = GraphBuilder(graph_path)
    if state.graph_builder.node_count() > 0:
        state.graph_loaded = True
        logger.info(
            f"GraphBuilder loaded from {graph_path} "
            f"({state.graph_builder.node_count()} nodes, "
            f"{state.graph_builder.edge_count()} edges)"
        )
        # Pre-compute node name embeddings so graph queries skip per-call embedding
        warm_graph_retriever(state.graph_builder.graph, state.embedder)
    else:
        logger.warning(f"Graph not found at {graph_path} — graph retrieval unavailable")

    # Ollama warmup — load the model into memory before the first real query
    _warm_ollama(state.config)

    # Illustration worker — background queue + consumer for text-to-image generation.
    # Directory must match the StaticFiles mount above so generated images are servable.
    scheduler = IllustrationScheduler(
        builder=state.graph_builder,
        doc_store=state.store,
        output_dir=_ILLUS_DIR,
        config=state.config,
    )
    scheduler.start()
    state.illustration_worker = scheduler
    # Re-queue any content nodes that have a hint but no generated image yet
    # (covers crashes or restarts mid-generation).
    try:
        pending = await scheduler.scan_and_enqueue_pending()
        if pending:
            logger.info(f"Illustration worker: {pending} pending node(s) re-queued")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Illustration startup scan failed: {exc}")

    logger.info("LuminRAG ready.")
    yield

    # Shutdown
    if state.illustration_worker:
        await state.illustration_worker.stop()
    if state.store:
        state.store.close()
    logger.info("LuminRAG shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="LuminRAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://luminrag-frontend-bucket.s3-website.us-east-2.amazonaws.com",
        "http://luminrag-frontend-bucket.s3-website.us-east-2.amazonaws.com/",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated illustration images as static files. The directory is
# created eagerly so StaticFiles can mount cleanly even on a cold install.
_PROCESSED_DIR = (Path(__file__).parent / "data" / "processed").resolve()
_ILLUS_DIR = illustrations_dir(_PROCESSED_DIR)
app.mount(
    "/static/illustrations",
    StaticFiles(directory=str(_ILLUS_DIR)),
    name="illustrations",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    routing_mode: Literal["dense", "graph", "hybrid"] | None = None
    max_sources: int | None = None      # None = no cap on returned chunks
    min_relevancy: float | None = None  # None = 0.0 (no cutoff)


class EvidenceChunkResponse(BaseModel):
    id: str
    text: str
    source: str
    modality: str
    page: int | None = None
    timestamp: str | None = None
    retrieval_source: str | None = None  # "dense" | "graph" | "both" (hybrid mode only)
    relevancy_score: float | None = None  # cosine similarity vs. query, set by reranker


class ReflectionVerdictResponse(BaseModel):
    needs_retrieval: bool
    is_relevant: bool
    is_supported: bool
    is_useful: bool
    reasoning: str


class QueryResponse(BaseModel):
    role: str = "assistant"
    content: str
    routing_mode: str
    evidence: list[EvidenceChunkResponse]
    reflection: ReflectionVerdictResponse
    hops: list[str]


class IllustrationPayload(BaseModel):
    kind: Literal["diagram", "equation", "code", "image"]
    hint: str


class GraphNodeResponse(BaseModel):
    """
    Rich node payload for the frontend. Different fields are populated
    depending on node_type — the frontend reads the ones it needs.
    """

    id: str
    name: str
    node_type: Literal["topic", "subtopic", "content"]
    source_ids: list[str] = Field(default_factory=list)
    # Topic / Subtopic
    summary: str | None = None
    scope: Literal["broad", "narrow"] | None = None
    illustration: IllustrationPayload | None = None
    parent_topic_keys: list[str] = Field(default_factory=list)
    # Content
    content_type: str | None = None
    parent_subtopic_keys: list[str] = Field(default_factory=list)
    raw_excerpt: str | None = None
    key_terms: list[str] = Field(default_factory=list)
    illustration_path: str | None = None
    # Frontend selection state — returned as `false` from the API; the
    # frontend mutates client-side.
    highlighted: bool = False


class GraphEdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    label: str | None = None
    confidence: float | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    highlighted: bool = False


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


class GraphNodeDetailResponse(GraphNodeResponse):
    """
    Extended node payload served by GET /api/graph/node/{key} with the
    resolved evidence trail (chunk text joined from SQLite via source_ids).
    """

    evidence: list[EvidenceChunkResponse] = Field(default_factory=list)


class IngestResponse(BaseModel):
    files_processed: int
    chunks_produced: int
    topics_added: int
    subtopics_added: int
    contents_added: int
    related_edges_added: int
    graph_nodes: int
    graph_edges: int
    failed: list[str]
    warnings: list[str] = Field(default_factory=list)


class IngestJobResponse(BaseModel):
    job_id: str
    status: str


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_stage: str
    result: IngestResponse | None = None
    error: str | None = None


@dataclasses.dataclass
class _IngestJob:
    job_id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    progress_stage: str = "Queued"
    result: IngestResponse | None = None
    error: str | None = None


_jobs: dict[str, _IngestJob] = {}


_CONTENT_TYPES = {
    "definition", "theorem", "technique", "example", "question", "figure", "other"
}


# ---------------------------------------------------------------------------
# S3 presigned upload helpers
# ---------------------------------------------------------------------------

_s3_client: boto3.client | None = None


_UNSAFE_FILENAME_CHARS = re.compile(r"[#?%&]")


def _sanitize_filename(name: str) -> str:
    """Replace URL-special characters in a filename with underscores."""
    return _UNSAFE_FILENAME_CHARS.sub("_", name)


def _get_s3() -> boto3.client:
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _s3_client


class PresignRequest(BaseModel):
    filename: str
    content_type: str


class PresignedFile(BaseModel):
    key: str
    url: str
    filename: str


class IngestFromS3Request(BaseModel):
    keys: list[str]
    slides: list[str] = Field(default_factory=list)


class AddNodeRequest(BaseModel):
    name: str
    node_type: Literal["topic", "subtopic", "content"] = "topic"
    summary: str | None = None
    scope: Literal["broad", "narrow"] = "broad"
    content_type: str | None = None
    illustration: IllustrationPayload | None = None
    # Content-only enrichment
    raw_excerpt: str | None = None
    key_terms: list[str] = Field(default_factory=list)
    # Optional auto-linking on insert. Must be existing node keys.
    parent_topic_keys: list[str] = Field(default_factory=list)
    parent_subtopic_keys: list[str] = Field(default_factory=list)


class PatchNodeRequest(BaseModel):
    name: str | None = None
    summary: str | None = None
    scope: Literal["broad", "narrow"] | None = None
    content_type: str | None = None
    raw_excerpt: str | None = None
    key_terms: list[str] | None = None
    source_ids: list[str] | None = None  # full replacement list; omit to leave unchanged


class AddEdgeRequest(BaseModel):
    source: str
    target: str
    relation: str                         # HAS_SUBTOPIC | HAS_CONTENT | RELATED_TO
    label: str | None = None              # required when relation == RELATED_TO
    confidence: float = 0.5


# Supported file-type sets (mirrors scripts/ingest.py)
_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_PDF_EXT   = {".pdf"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks_to_evidence(result: RetrievalResult) -> list[EvidenceChunkResponse]:
    """Convert retrieved Chunks into frontend-compatible EvidenceChunk objects."""
    evidence = []
    for chunk in result.chunks:
        meta = chunk.metadata or {}
        page = meta.get("page_number") or meta.get("page_start")
        ts_raw = meta.get("start_time")
        timestamp: str | None = None
        if ts_raw is not None:
            secs = int(float(ts_raw))
            timestamp = f"{secs // 60}:{secs % 60:02d}"
        evidence.append(
            EvidenceChunkResponse(
                id=chunk.id,
                text=chunk.text[:500],  # cap for network payload
                source=chunk.source_id,
                modality=chunk.modality,
                page=int(page) if page is not None else None,
                timestamp=timestamp,
                retrieval_source=meta.get("retrieval_source"),
                relevancy_score=meta.get("relevancy_score"),
            )
        )
    return evidence


def _subgraph_to_hops(result: RetrievalResult) -> list[str]:
    """Stable node keys for the retrieved subgraph — frontend highlights these."""
    return list(result.subgraph_node_keys)


def _node_to_response(node_id: str, attrs: dict) -> GraphNodeResponse:
    illustration = attrs.get("illustration")
    illustration_payload = (
        IllustrationPayload(**illustration)
        if isinstance(illustration, dict)
        else None
    )
    return GraphNodeResponse(
        id=node_id,
        name=attrs.get("name", node_id),
        node_type=attrs.get("node_type", "content"),
        source_ids=list(attrs.get("source_ids", [])),
        summary=attrs.get("summary"),
        scope=attrs.get("scope"),
        illustration=illustration_payload,
        parent_topic_keys=list(attrs.get("parent_topic_keys", [])),
        content_type=attrs.get("content_type"),
        parent_subtopic_keys=list(attrs.get("parent_subtopic_keys", [])),
        raw_excerpt=attrs.get("raw_excerpt") or None,
        key_terms=list(attrs.get("key_terms", [])),
        illustration_path=attrs.get("illustration_path"),
    )


def _edge_to_response(source: str, target: str, attrs: dict) -> GraphEdgeResponse:
    relation = attrs.get("relation", "")
    label = attrs.get("label") if relation == "RELATED_TO" else None
    confidence = attrs.get("confidence") if relation == "RELATED_TO" else None
    suffix = label or relation
    return GraphEdgeResponse(
        id=f"{source}__{suffix}__{target}",
        source=source,
        target=target,
        relation=relation,
        label=label,
        confidence=confidence,
        source_chunk_ids=list(attrs.get("source_chunk_ids", [])),
    )


def _export_graph_response() -> GraphResponse:
    """Build a GraphResponse from the loaded GraphBuilder — empty if not loaded."""
    if not state.graph_loaded or state.graph_builder is None:
        return GraphResponse(nodes=[], edges=[])

    graph = state.graph_builder.graph
    nodes = [
        _node_to_response(n, a)
        for n, a in graph.nodes(data=True)
        if a.get("node_type") != "chunk_ref"
    ]
    node_ids = {node.id for node in nodes}
    edges = [
        _edge_to_response(u, v, a)
        for u, v, a in graph.edges(data=True)
        if u in node_ids and v in node_ids
    ]
    return GraphResponse(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_DOC_PREVIEW_CHARS = 3000


def _make_doc_preview(chunks: list[Chunk], filenames: list[str]) -> str:
    """Filename hint + concatenated chunk text, capped for the topic-extractor prompt."""
    header = "Source files: " + ", ".join(filenames) + "\n\n"
    body = "\n\n".join(c.text for c in chunks if c.text.strip())
    return (header + body)[:_DOC_PREVIEW_CHARS]


def _names_by_type(graph, node_type: str) -> list[str]:
    return [
        a.get("name", k) for k, a in graph.nodes(data=True)
        if a.get("node_type") == node_type
    ]


def _run_graph_pipeline(
    all_chunks: list,
    filenames: list[str],
    graph_path: Path,
    config: dict,
    graph_builder: GraphBuilder,
    embedder: Embedder,
) -> "ApplyResult":
    """
    Stages 1.2–1.5 + Stage 2 of the ingest pipeline. All synchronous LLM
    calls and CPU-bound work. Must be called via asyncio.to_thread — never
    directly from an async handler.
    """
    from backend.graph.topic_extractor import extract_topics
    from backend.graph.subtopic_extractor import extract_subtopics
    from backend.graph.content_synthesizer import synthesize_contents
    from backend.graph.semantic_linker import link_nodes, build_linkables_from_graph

    graph = graph_builder.graph

    preview = _make_doc_preview(all_chunks, filenames)
    topic_proposals = extract_topics(
        preview,
        [c.id for c in all_chunks],
        config,
        existing_names=_names_by_type(graph, "topic"),
        embedder=embedder,
    )
    topic_names = [tp.name for tp in topic_proposals]
    logger.info(f"Topics: {len(topic_proposals)} — {topic_names}")

    subtopic_proposals = extract_subtopics(
        all_chunks,
        topic_names,
        config,
        existing_names=_names_by_type(graph, "subtopic"),
        embedder=embedder,
    )
    logger.info(f"Subtopics: {len(subtopic_proposals)}")

    subs_by_chunk: dict[str, list[str]] = {}
    for sp in subtopic_proposals:
        for cid in sp.source_chunk_ids:
            subs_by_chunk.setdefault(cid, []).append(sp.name)

    existing_titles = _names_by_type(graph, "content")
    content_proposals = []
    for chunk in all_chunks:
        content_proposals.extend(
            synthesize_contents(
                chunk,
                subs_by_chunk.get(chunk.id, []),
                topic_names,
                config,
                existing_titles=existing_titles,
                embedder=embedder,
            )
        )
    logger.info(f"Contents: {len(content_proposals)}")

    apply_result = graph_builder.apply_proposals(
        ProposalBundle(
            topics=topic_proposals,
            subtopics=subtopic_proposals,
            contents=content_proposals,
        )
    )

    linkables = build_linkables_from_graph(graph_builder.graph)
    related_proposals = link_nodes(linkables, embedder, config)
    logger.info(f"RELATED_TO proposals: {len(related_proposals)}")

    if related_proposals:
        rr = graph_builder.apply_proposals(ProposalBundle(related=related_proposals))
        apply_result.related_edges_added += rr.related_edges_added
        apply_result.related_edges_merged += rr.related_edges_merged
        apply_result.warnings.extend(rr.warnings)

    pruned = graph_builder.prune_disconnected()
    if pruned:
        logger.info(f"Pruned {pruned} disconnected node(s) after ingestion")
    ratio_removed = graph_builder.prune_to_ratio()
    if any(ratio_removed.values()):
        graph_builder.prune_disconnected()

    graph_builder.save()
    export_graph(graph_builder.graph, graph_path)
    warm_graph_retriever(graph_builder.graph, embedder)
    logger.info("Graph retriever re-warmed.")

    return apply_result


async def _run_ingest_job(
    job_id: str,
    saved_paths: list[Path],
    slide_names: set[str],
    graph_path: Path,
) -> None:
    """
    Background task: runs the full ingest pipeline and updates job progress.
    All blocking stages are wrapped in asyncio.to_thread so the event loop
    stays responsive for health checks and other requests during ingestion.
    """
    job = _jobs[job_id]
    try:
        job.status = "running"

        # Stage 1 — per-file chunking (Ollama vision calls for slides; CPU for PDFs)
        all_chunks: list = []
        failed: list[str] = []
        for path in saved_paths:
            job.progress_stage = f"Chunking {path.name}…"
            try:
                chunks = await asyncio.to_thread(
                    _ingest_single_file, path, state.config, slide_names
                )
                all_chunks.extend(chunks)
                logger.info(f"  {path.name} → {len(chunks)} chunk(s)")
            except Exception as exc:
                logger.error(f"Failed to ingest {path.name}: {exc}")
                failed.append(path.name)

        if not all_chunks:
            raise ValueError(
                f"No chunks produced. Failed: {', '.join(failed) or 'none'}"
            )

        # Stage 2 — persist to SQLite (fast, OK on event loop)
        state.store.save_chunks(all_chunks)
        logger.info(f"DocumentStore now has {state.store.count()} chunk(s).")

        # Stage 3 — rebuild FAISS index (CPU-bound sentence-transformers)
        job.progress_stage = "Building vector index…"
        all_stored = state.store.get_all_chunks()
        index_path = Path(
            state.config.get("vector_db", {}).get(
                "path", "backend/data/processed/faiss.index"
            )
        )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        new_index = VectorIndex(state.embedder)
        await asyncio.to_thread(new_index.build, all_stored)
        new_index.save(index_path)
        state.vector_index = new_index
        state.index_loaded = True
        logger.info(f"Vector index rebuilt ({len(all_stored)} chunks).")

        # Stages 1.2-1.5 + Stage 2 — LLM extraction + graph build
        if state.graph_builder is None:
            state.graph_builder = GraphBuilder(graph_path)

        job.progress_stage = "Extracting topics & subtopics…"
        filenames = [p.name for p in saved_paths]
        apply_result = await asyncio.to_thread(
            _run_graph_pipeline,
            all_chunks,
            filenames,
            graph_path,
            state.config,
            state.graph_builder,
            state.embedder,
        )
        state.graph_loaded = True

        if state.illustration_worker is not None:
            try:
                await state.illustration_worker.scan_and_enqueue_pending()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Illustration enqueue after ingest failed: {exc}")

        job.result = IngestResponse(
            files_processed=len(saved_paths) - len(failed),
            chunks_produced=len(all_chunks),
            topics_added=apply_result.topics_added,
            subtopics_added=apply_result.subtopics_added,
            contents_added=apply_result.contents_added,
            related_edges_added=apply_result.related_edges_added,
            graph_nodes=state.graph_builder.node_count(),
            graph_edges=state.graph_builder.edge_count(),
            failed=failed,
            warnings=apply_result.warnings[:20],
        )
        job.progress_stage = "Done"
        job.status = "done"

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Ingest job {job_id} failed: {exc}")
        job.error = str(exc)
        job.progress_stage = "Failed"
        job.status = "failed"


def _ingest_single_file(path: Path, config: dict, slide_names: set[str]) -> list:
    """Dispatch a single file to the correct ingestion processor."""
    ext = path.suffix.lower()

    if ext in _VIDEO_EXT:
        from backend.ingestion.video_transcriber import transcribe_video
        return transcribe_video(path, config)

    if ext in _AUDIO_EXT:
        from backend.ingestion.audio_processor import process_audio
        return process_audio(path, config)

    if ext in _IMAGE_EXT:
        from backend.ingestion.image_processor import process_image
        return process_image(path, config)

    if ext in _PDF_EXT:
        if path.name in slide_names:
            from backend.ingestion.slide_processor import process_slides
            return process_slides(path, config)
        else:
            from backend.ingestion.pdf_processor import process_pdf
            return process_pdf(path, config)

    return []


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_loaded": state.index_loaded,
        "graph_loaded": state.graph_loaded,
        "chunk_count": state.store.count() if state.store else 0,
    }


@app.get("/api/suggestions")
def get_suggestions(n: int = 6) -> list[str]:
    """
    Return up to *n* example questions derived from the concept graph.

    Questions are templated from topic/subtopic/content node names so they
    reflect what is actually in the database. Falls back to an empty list
    when no graph is loaded.
    """
    import random

    if state.graph_builder is None:
        return []

    g = state.graph_builder.graph
    if g.number_of_nodes() == 0:
        return []

    _TOPIC_TEMPLATES = [
        "Give me an overview of {name}.",
        "What are the key ideas in {name}?",
        "How does {name} work?",
    ]
    _SUBTOPIC_TEMPLATES = [
        "Explain {name}.",
        "What is {name}?",
        "Why is {name} important?",
    ]
    _CONTENT_TEMPLATES: dict[str, list[str]] = {
        "definition":  ["What is the definition of {name}?", "Define {name}."],
        "theorem":     ["State and explain {name}.", "What does {name} say?"],
        "technique":   ["How does {name} work?", "Walk me through {name}."],
        "example":     ["Can you give me an example of {name}?"],
        "question":    ["How would you solve {name}?"],
        "figure":      ["What does {name} illustrate?"],
        "_default":    ["Explain {name}.", "What is {name}?"],
    }

    candidates: list[str] = []
    nodes = list(g.nodes(data=True))
    random.shuffle(nodes)

    for node_id, attrs in nodes:
        ntype = attrs.get("node_type")
        name = attrs.get("name", "").strip()
        if not name or ntype == "chunk_ref":
            continue

        if ntype == "topic":
            tmpl = random.choice(_TOPIC_TEMPLATES)
        elif ntype == "subtopic":
            tmpl = random.choice(_SUBTOPIC_TEMPLATES)
        elif ntype == "content":
            ct = attrs.get("content_type") or "_default"
            bucket = _CONTENT_TEMPLATES.get(ct, _CONTENT_TEMPLATES["_default"])
            tmpl = random.choice(bucket)
        else:
            continue

        candidates.append(tmpl.format(name=name))
        if len(candidates) >= n:
            break

    return candidates[:n]


@app.get("/api/graph", response_model=GraphResponse)
def get_graph() -> GraphResponse:
    """Return the concept knowledge graph for the frontend visualization."""
    return _export_graph_response()


_RAW_DIR = (Path(__file__).parent / "data" / "raw").resolve()


@app.get("/api/source/{source_id}")
def get_source_file(source_id: str):
    """
    Serve the original raw file behind a chunk's ``source_id``.

    The ``source_id`` on each chunk is the file *stem* (e.g. "lecture_01"),
    so we scan ``backend/data/raw/`` for any file whose stem matches and
    stream it back. Browsers render PDFs / images inline and stream video /
    audio via HTTP Range (FileResponse handles that automatically).

    Path-traversal safety: the candidate path is resolved and checked to
    stay inside ``_RAW_DIR``.
    """
    if not source_id or "/" in source_id or "\\" in source_id or ".." in source_id:
        raise HTTPException(status_code=400, detail="Invalid source_id.")

    if not _RAW_DIR.exists():
        raise HTTPException(status_code=404, detail="Raw file directory missing.")

    # Content-Disposition: inline tells the browser to render the file
    # (PDF viewer / image viewer / native video/audio player) instead of
    # triggering a download. We still pass `filename` so if the user does
    # choose "Save as…", they get a sensible name.

    # First, try treating source_id as the literal filename (with extension).
    direct = (_RAW_DIR / source_id).resolve()
    if direct.is_file() and _RAW_DIR in direct.parents:
        return FileResponse(direct, filename=direct.name, content_disposition_type="inline")

    # Otherwise, resolve by stem — find any file whose stem matches.
    for entry in _RAW_DIR.iterdir():
        if entry.is_file() and entry.stem == source_id:
            resolved = entry.resolve()
            if _RAW_DIR in resolved.parents:
                return FileResponse(resolved, filename=entry.name, content_disposition_type="inline")

    raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Main QA pipeline:
      1. Multi-hop reasoner (decompose → route → retrieve → reflect per sub-question)
      2. Generate answer with Self-RAG post-reflection
      3. Return answer + evidence + graph hops
    """
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    # Run multi-hop reasoning (handles simple + complex queries uniformly)
    retrieval_result = reason(
        question=q,
        vector_index=state.vector_index,
        graph_builder=state.graph_builder,
        store=state.store,
        embedder=state.embedder,
        config=state.config,
        force_routing_mode=request.routing_mode,
    )

    # Rerank: re-score all merged chunks against the original question with a
    # cross-encoder, apply min_relevancy cutoff, sort, and cap at max_sources.
    from backend.retrieval.reranker import rerank as _rerank
    if retrieval_result.chunks:
        reranked_chunks = _rerank(
            query=q,
            chunks=retrieval_result.chunks,
            config=state.config,
            min_relevancy=request.min_relevancy if request.min_relevancy is not None else 0.0,
            max_sources=request.max_sources,
        )
        retrieval_result = retrieval_result.model_copy(update={"chunks": reranked_chunks})

    # Generate answer
    gen: GenerationResult = generate(
        question=q,
        retrieval_result=retrieval_result,
        config=state.config,
    )

    evidence = _chunks_to_evidence(retrieval_result)
    hops = _subgraph_to_hops(retrieval_result)

    # If the user explicitly forced a routing mode, echo it back so the frontend
    # label always matches what was requested (even when the pipeline falls back
    # to an empty result).
    response_routing_mode = request.routing_mode or gen.routing_mode

    return QueryResponse(
        content=gen.answer,
        routing_mode=response_routing_mode,
        evidence=evidence,
        reflection=ReflectionVerdictResponse(**gen.reflection.model_dump()),
        hops=hops,
    )


@app.delete("/api/data")
def clear_data() -> dict[str, Any]:
    """
    Wipe all stored data:
      - DocumentStore (all chunks)
      - FAISS vector index (file + in-memory)
      - Concept graph (file + in-memory)
      - Raw uploaded files
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    # Clear document store
    deleted_chunks = state.store.clear_all()

    # Reset in-memory vector index and delete files
    index_path = Path(
        state.config.get("vector_db", {}).get("path", "backend/data/processed/faiss.index")
    )
    state.vector_index = VectorIndex(state.embedder)
    state.index_loaded = False
    for p in [index_path, index_path.with_suffix(".index.json")]:
        if p.exists():
            p.unlink()

    # Reset in-memory graph and delete file
    graph_path = Path(
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    )
    from backend.retrieval.graph_retriever import _node_emb_cache
    _node_emb_cache.clear()
    if graph_path.exists():
        graph_path.unlink()
    state.graph_builder = GraphBuilder(graph_path)  # starts empty since file is gone
    state.graph_loaded = False

    # Delete raw uploaded files
    raw_dir = Path(__file__).parent.parent / "backend" / "data" / "raw"
    deleted_files = 0
    if raw_dir.exists():
        for f in raw_dir.iterdir():
            if f.is_file():
                f.unlink()
                deleted_files += 1

    logger.info(
        f"Data cleared: {deleted_chunks} chunks, {deleted_files} raw files, "
        "index + graph reset."
    )
    return {
        "deleted_chunks": deleted_chunks,
        "deleted_raw_files": deleted_files,
    }


@app.post("/api/upload/presign", response_model=list[PresignedFile])
def presign_uploads(files: list[PresignRequest]) -> list[PresignedFile]:
    """
    Generate short-lived S3 presigned PUT URLs so the frontend can upload
    files directly to S3, bypassing the backend (and Cloudflare) entirely.

    Each file gets a unique key: uploads/<uuid>/<filename>
    """
    bucket = os.getenv("S3_BUCKET", "luminrag-frontend-bucket")
    result: list[PresignedFile] = []
    try:
        s3 = _get_s3()
        for f in files:
            key = f"uploads/{uuid.uuid4()}/{Path(f.filename).name}"
            url = s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket, "Key": key, "ContentType": f.content_type},
                ExpiresIn=3600,
            )
            result.append(PresignedFile(key=key, url=url, filename=f.filename))
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=f"S3 presign failed: {exc}") from exc
    return result


@app.post("/api/ingest/from-s3", response_model=IngestJobResponse, status_code=202)
async def ingest_from_s3(req: IngestFromS3Request) -> IngestJobResponse:
    """
    Download files from S3 then kick off the ingestion pipeline as a background
    job. Returns a job_id immediately; poll GET /api/ingest/jobs/{job_id} for
    status and results.
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    bucket = os.getenv("S3_BUCKET", "luminrag-frontend-bucket")
    slide_names: set[str] = set(req.slides)

    raw_dir = Path(__file__).parent.parent / "backend" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download from S3 (sync boto3 IO — wrapped in thread)
    saved_paths: list[Path] = []
    try:
        s3 = _get_s3()
        for key in req.keys:
            filename = _sanitize_filename(Path(key).name)
            dest = raw_dir / filename
            await asyncio.to_thread(s3.download_file, bucket, key, str(dest))
            saved_paths.append(dest)
            logger.info(f"Downloaded from S3: {key} → {dest}")
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=f"S3 download failed: {exc}") from exc

    graph_path = Path(
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    )
    job_id = str(uuid.uuid4())
    _jobs[job_id] = _IngestJob(job_id=job_id)
    asyncio.create_task(_run_ingest_job(job_id, saved_paths, slide_names, graph_path))
    return IngestJobResponse(job_id=job_id, status="queued")


@app.get("/api/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
def get_ingest_job(job_id: str) -> IngestJobStatusResponse:
    """Poll the status of a background ingest job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_stage=job.progress_stage,
        result=job.result,
        error=job.error,
    )


@app.post("/api/ingest", response_model=IngestJobResponse, status_code=202)
async def ingest_files(
    files: list[UploadFile] = File(...),
    slides: str = Form(default=""),
) -> IngestJobResponse:
    """
    Upload course files and kick off the ingestion pipeline as a background job.
    Returns a job_id immediately; poll GET /api/ingest/jobs/{job_id} for status.

    `slides` is a comma-separated list of PDF filenames that should be
    treated as slide decks (captioned per-page) instead of text PDFs.
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    slide_names: set[str] = {s.strip() for s in slides.split(",") if s.strip()}

    raw_dir = Path(__file__).parent.parent / "backend" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save uploads to disk (fast in-process copy — fine on event loop)
    saved_paths: list[Path] = []
    for upload in files:
        filename = Path(upload.filename).name
        dest = raw_dir / filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(dest)
        logger.info(f"Saved uploaded file: {dest}")

    graph_path = Path(
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    )
    job_id = str(uuid.uuid4())
    _jobs[job_id] = _IngestJob(job_id=job_id)
    asyncio.create_task(_run_ingest_job(job_id, saved_paths, slide_names, graph_path))
    return IngestJobResponse(job_id=job_id, status="queued")


# ---------------------------------------------------------------------------
# Graph edit endpoints
# ---------------------------------------------------------------------------

def _require_graph_builder() -> GraphBuilder:
    """Return the live GraphBuilder, initialising a fresh one if needed."""
    if state.graph_builder is None:
        graph_path = Path(
            state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
        )
        state.graph_builder = GraphBuilder(graph_path)
    return state.graph_builder


def _persist_graph(builder: GraphBuilder) -> None:
    """Save graph to disk, re-export JSON, and re-warm the embedding cache."""
    graph_path = Path(
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    )
    builder.save()
    export_graph(builder.graph, graph_path)
    state.graph_loaded = True
    if state.embedder:
        warm_graph_retriever(builder.graph, state.embedder)


@app.post("/api/graph/node", response_model=GraphNodeResponse, status_code=201)
def add_graph_node(request: AddNodeRequest) -> GraphNodeResponse:
    """
    Add a manually-created Topic / Subtopic / Content node. Rich fields are
    optional; the frontend form decides which to send. If parent_*_keys are
    provided, structural edges are auto-created.
    """
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Node name must not be empty.")

    builder = _require_graph_builder()
    key = make_key(request.node_type, name)

    if key in builder.graph:
        raise HTTPException(status_code=409, detail=f"Node '{key}' already exists.")

    illustration = (
        Illustration(kind=request.illustration.kind, hint=request.illustration.hint)
        if request.illustration
        else None
    )

    if request.node_type == "topic":
        builder.add_node(TopicAttrs(
            name=name,
            summary=request.summary or "",
            scope=request.scope,
            illustration=illustration,
            origin="manual",
        ))
    elif request.node_type == "subtopic":
        # Validate any declared parent topic keys
        for pkey in request.parent_topic_keys:
            if pkey not in builder.graph:
                raise HTTPException(
                    status_code=404, detail=f"Parent topic '{pkey}' not found."
                )
        builder.add_node(SubtopicAttrs(
            name=name,
            summary=request.summary or "",
            illustration=illustration,
            parent_topic_keys=list(request.parent_topic_keys),
            origin="manual",
        ))
        for pkey in request.parent_topic_keys:
            builder.add_structural_edge(pkey, "HAS_SUBTOPIC", key)
    else:  # content
        content_type = request.content_type or "other"
        if content_type not in _CONTENT_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"content_type must be one of: {sorted(_CONTENT_TYPES)}",
            )
        for pkey in request.parent_subtopic_keys:
            if pkey not in builder.graph:
                raise HTTPException(
                    status_code=404, detail=f"Parent subtopic '{pkey}' not found."
                )
        builder.add_node(ContentAttrs(
            name=name,
            content_type=content_type,
            summary=request.summary or "",
            raw_excerpt=(request.raw_excerpt or "")[:300],
            key_terms=list(request.key_terms),
            illustration=illustration,
            parent_subtopic_keys=list(request.parent_subtopic_keys),
            origin="manual",
        ))
        for pkey in request.parent_subtopic_keys:
            builder.add_structural_edge(pkey, "HAS_CONTENT", key)

    _persist_graph(builder)
    return _node_to_response(key, builder.graph.nodes[key])


@app.get("/api/graph/node/{node_id}", response_model=GraphNodeDetailResponse)
def get_graph_node_detail(node_id: str) -> GraphNodeDetailResponse:
    """
    Return the rich node payload plus the resolved evidence trail from
    the node's source_ids (joined with chunk text from SQLite).
    """
    builder = _require_graph_builder()
    if node_id not in builder.graph:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    base = _node_to_response(node_id, builder.graph.nodes[node_id])

    # Collect chunk_ids from the node's source_ids attribute.
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for cid in builder.graph.nodes[node_id].get("source_ids", []):
        if cid not in seen:
            seen.add(cid)
            chunk_ids.append(cid)

    evidence: list[EvidenceChunkResponse] = []
    if state.store and chunk_ids:
        for cid in chunk_ids:
            chunk = state.store.get_chunk(cid)
            if not chunk:
                continue
            meta = chunk.metadata or {}
            page = meta.get("page_number") or meta.get("page_start")
            ts_raw = meta.get("start_time")
            timestamp = None
            if ts_raw is not None:
                secs = int(float(ts_raw))
                timestamp = f"{secs // 60}:{secs % 60:02d}"
            evidence.append(EvidenceChunkResponse(
                id=chunk.id,
                text=chunk.text[:500],
                source=chunk.source_id,
                modality=chunk.modality,
                page=int(page) if page is not None else None,
                timestamp=timestamp,
            ))

    return GraphNodeDetailResponse(**base.model_dump(), evidence=evidence)


@app.delete("/api/graph/node/{node_id}")
def delete_graph_node(node_id: str) -> dict[str, Any]:
    builder = _require_graph_builder()
    if not builder.remove_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    _persist_graph(builder)
    return {"deleted": node_id}


@app.patch("/api/graph/node/{node_id}")
def patch_graph_node(node_id: str, request: PatchNodeRequest) -> dict[str, Any]:
    """
    Partially update mutable attributes of an existing node.
    Immutable fields (node_type, key, created_at) are silently ignored.
    Provide source_ids as a full replacement list to remove entries.
    """
    builder = _require_graph_builder()
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update.")
    if not builder.patch_node(node_id, updates):
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    _persist_graph(builder)
    return {"updated": node_id}


@app.post("/api/graph/edge", response_model=GraphEdgeResponse, status_code=201)
def add_graph_edge(request: AddEdgeRequest) -> GraphEdgeResponse:
    """
    Add a structural (HAS_SUBTOPIC / HAS_CONTENT / EVIDENCE_OF) or semantic
    (RELATED_TO) edge between two existing nodes. RELATED_TO requires a label.
    """
    relation = request.relation.strip().upper()
    if not relation:
        raise HTTPException(status_code=422, detail="Relation must not be empty.")

    builder = _require_graph_builder()
    src = request.source.strip()
    tgt = request.target.strip()

    if src not in builder.graph:
        raise HTTPException(status_code=404, detail=f"Source node '{src}' not found.")
    if tgt not in builder.graph:
        raise HTTPException(status_code=404, detail=f"Target node '{tgt}' not found.")

    if relation in STRUCTURAL_RELATIONS:
        existing = builder.graph.get_edge_data(src, tgt) or {}
        for ed in existing.values():
            if ed.get("relation") == relation:
                raise HTTPException(status_code=409, detail="Edge already exists.")
        builder.add_structural_edge(src, relation, tgt)
        _persist_graph(builder)
        edge_id = f"{src}__{relation}__{tgt}"
        return GraphEdgeResponse(id=edge_id, source=src, target=tgt, relation=relation)

    if relation == "RELATED_TO":
        if not request.label or not request.label.strip():
            raise HTTPException(
                status_code=422,
                detail="RELATED_TO edges require a `label`.",
            )
        norm = normalise_relation_label(request.label)
        existing = builder.graph.get_edge_data(src, tgt) or {}
        for ed in existing.values():
            if ed.get("relation") == "RELATED_TO" and ed.get("label") == norm:
                raise HTTPException(
                    status_code=409,
                    detail=f"RELATED_TO edge with label '{norm}' already exists.",
                )
        builder.add_related_to(
            src, tgt, norm, confidence=request.confidence, source_chunk_ids=[]
        )
        _persist_graph(builder)
        return GraphEdgeResponse(
            id=f"{src}__{norm}__{tgt}",
            source=src,
            target=tgt,
            relation="RELATED_TO",
            label=norm,
            confidence=request.confidence,
        )

    raise HTTPException(
        status_code=422,
        detail=f"Unknown relation '{relation}'. Use one of: "
        f"{sorted(STRUCTURAL_RELATIONS | {'RELATED_TO'})}",
    )


@app.delete("/api/graph/edge")
def delete_graph_edge(
    source: str, relation: str, target: str, label: str | None = None
) -> dict[str, Any]:
    """
    Delete a specific edge. For RELATED_TO, pass `label` to disambiguate
    when multiple labelled edges exist between the same pair.
    """
    builder = _require_graph_builder()
    if not builder.remove_edge(source, relation, target, label=label):
        raise HTTPException(status_code=404, detail="Edge not found.")

    _persist_graph(builder)
    return {"deleted": f"{source}__{label or relation}__{target}"}

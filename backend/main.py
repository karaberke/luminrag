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

import logging
import os
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

import re
import shutil
from collections import defaultdict

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.db.document_store import DocumentStore
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_retriever import VectorIndex
from backend.retrieval.graph_retriever import warm_up as warm_graph_retriever
from backend.graph.graph_builder import GraphBuilder
from backend.graph.graph_export import export_graph
from backend.self_rag.multi_hop_reasoner import reason
from backend.generation.generator import generate
from backend.schemas import GenerationResult, RetrievalResult

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

    logger.info("LuminRAG ready.")
    yield

    # Shutdown
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


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    routing_mode: Literal["dense", "graph"] | None = None


class EvidenceChunkResponse(BaseModel):
    id: str
    text: str
    source: str
    modality: str
    page: int | None = None
    timestamp: str | None = None


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


class GraphNodeResponse(BaseModel):
    id: str
    label: str
    type: str
    highlighted: bool = False


class GraphEdgeResponse(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    highlighted: bool = False


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


class IngestResponse(BaseModel):
    files_processed: int
    chunks_produced: int
    triples_extracted: int
    graph_nodes: int
    graph_edges: int
    failed: list[str]


VALID_RELATIONS = {
    "PART_OF", "PREREQUISITE", "CAUSES", "EXAMPLE_OF", "EXPLAINS", "BELONGS_TO_TOPIC"
}


class AddNodeRequest(BaseModel):
    name: str
    node_type: Literal["concept", "topic"] = "concept"


class AddEdgeRequest(BaseModel):
    source: str   # lowercased node key
    target: str   # lowercased node key
    relation: str


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
            )
        )
    return evidence


def _subgraph_to_hops(result: RetrievalResult) -> list[str]:
    """
    Extract unique node IDs from the retrieved subgraph.
    Node IDs are lowercase names (matching GraphBuilder's key convention).
    """
    seen: set[str] = set()
    hops: list[str] = []
    for triple in result.subgraph:
        for name in (triple.head, triple.tail):
            key = name.strip().lower()
            if key not in seen:
                seen.add(key)
                hops.append(key)
    return hops


def _export_graph_response() -> GraphResponse:
    """
    Build a GraphResponse from the loaded GraphBuilder.
    Returns an empty graph if the builder is not loaded.
    """
    if not state.graph_loaded or state.graph_builder is None:
        return GraphResponse(nodes=[], edges=[])

    raw = export_graph(
        state.graph_builder.graph,
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json"),
    )

    nodes = [
        GraphNodeResponse(
            id=n["id"],
            label=n.get("name", n["id"]),
            type="topic" if n.get("node_type") == "cluster" else "concept",
            highlighted=False,
        )
        for n in raw.get("nodes", [])
    ]

    edges = [
        GraphEdgeResponse(
            id=f"{e['source']}__{e['relation']}__{e['target']}",
            source=e["source"],
            target=e["target"],
            relation=e.get("relation", ""),
            highlighted=False,
        )
        for e in raw.get("edges", [])
    ]

    return GraphResponse(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _derive_topic_name(filename: str) -> str:
    """
    Convert a raw filename into a readable topic cluster name.
    e.g. "nlp_lecture_week3.pdf" → "Nlp Lecture Week"
    """
    stem = Path(filename).stem
    clean = re.sub(r"[_\-]+", " ", stem)           # underscores/hyphens → spaces
    clean = re.sub(r"(^|\s)\d+(\s|$)", " ", clean) # strip standalone numbers
    clean = " ".join(w.capitalize() for w in clean.split() if w)
    return clean or stem


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


@app.get("/api/graph", response_model=GraphResponse)
def get_graph() -> GraphResponse:
    """Return the concept knowledge graph for the frontend visualization."""
    return _export_graph_response()


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
    state.graph_builder = GraphBuilder.__new__(GraphBuilder)
    state.graph_builder._path = graph_path
    import networkx as nx
    state.graph_builder._graph = nx.MultiDiGraph()
    state.graph_loaded = False
    if graph_path.exists():
        graph_path.unlink()

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


@app.post("/api/ingest", response_model=IngestResponse)
def ingest_files(
    files: list[UploadFile] = File(...),
    slides: str = Form(default=""),
) -> IngestResponse:
    """
    Upload one or more course files and run the full ingestion pipeline:
      1. Save uploaded files to backend/data/raw/
      2. Ingest → Chunks
      3. Save chunks to DocumentStore
      4. Rebuild FAISS vector index (from all stored chunks)
      5. Extract triples + update concept graph
      6. Reload in-memory singletons so queries reflect new content

    `slides` is a comma-separated list of PDF filenames that should be
    treated as slide decks (captioned per-page) instead of text PDFs.
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    slide_names: set[str] = {s.strip() for s in slides.split(",") if s.strip()}

    raw_dir = Path(__file__).parent.parent / "backend" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Save uploaded files to disk ─────────────────────────────────────────
    saved_paths: list[Path] = []
    for upload in files:
        filename = Path(upload.filename).name  # strip any path components
        dest = raw_dir / filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(dest)
        logger.info(f"Saved uploaded file: {dest}")

    # ── Stage 1: Ingest files → Chunks ──────────────────────────────────────
    all_chunks: list = []
    failed: list[str] = []

    for path in saved_paths:
        try:
            chunks = _ingest_single_file(path, state.config, slide_names)
            all_chunks.extend(chunks)
            logger.info(f"  {path.name} → {len(chunks)} chunk(s)")
        except Exception as exc:
            logger.error(f"Failed to ingest {path.name}: {exc}")
            failed.append(path.name)

    if not all_chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No chunks produced from uploaded files. Failed: {', '.join(failed) or 'none'}",
        )

    # ── Stage 2: Persist chunks ──────────────────────────────────────────────
    state.store.save_chunks(all_chunks)
    logger.info(f"DocumentStore now has {state.store.count()} chunk(s).")

    # ── Stage 3: Rebuild FAISS index from all stored chunks ─────────────────
    all_stored_chunks = state.store.get_all_chunks()
    index_path = Path(
        state.config.get("vector_db", {}).get("path", "backend/data/processed/faiss.index")
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new_index = VectorIndex(state.embedder)
    new_index.build(all_stored_chunks)
    new_index.save(index_path)
    state.vector_index = new_index
    state.index_loaded = True
    logger.info(f"Vector index rebuilt ({len(all_stored_chunks)} chunks).")

    # ── Stage 4: Extract triples + update concept graph ─────────────────────
    from backend.graph.entity_extractor import extract_triples

    graph_path = Path(
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    )
    triples = extract_triples(all_chunks, state.config)
    logger.info(f"Extracted {len(triples)} triple(s).")

    if state.graph_builder is None:
        state.graph_builder = GraphBuilder(graph_path)

    state.graph_builder.add_triples(triples)

    # Create topic cluster nodes — one per uploaded source file.
    # Maps each entity that appears in triples back to its source file so
    # the cluster node links to the right concepts.
    chunk_id_to_source: dict[str, str] = {c.id: c.source_id for c in all_chunks}
    source_to_entity_keys: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        for cid in triple.source_chunk_ids:
            sid = chunk_id_to_source.get(cid)
            if sid:
                source_to_entity_keys[sid].add(triple.head.strip().lower())
                source_to_entity_keys[sid].add(triple.tail.strip().lower())

    for source_id, entity_keys in source_to_entity_keys.items():
        cluster_name = _derive_topic_name(Path(source_id).name)
        state.graph_builder.add_source_cluster(cluster_name, list(entity_keys))
        logger.info(f"Cluster '{cluster_name}': {len(entity_keys)} entity nodes")

    state.graph_builder.save()
    export_graph(state.graph_builder.graph, graph_path)
    state.graph_loaded = True

    # Re-warm graph retriever so new node embeddings are pre-computed
    warm_graph_retriever(state.graph_builder.graph, state.embedder)
    logger.info("Graph retriever re-warmed.")

    return IngestResponse(
        files_processed=len(saved_paths) - len(failed),
        chunks_produced=len(all_chunks),
        triples_extracted=len(triples),
        graph_nodes=state.graph_builder.node_count(),
        graph_edges=state.graph_builder.edge_count(),
        failed=failed,
    )


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
    """Add a manually-created concept or topic node to the graph."""
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Node name must not be empty.")

    builder = _require_graph_builder()
    key = name.lower()

    if key in builder.graph:
        raise HTTPException(status_code=409, detail=f"Node '{key}' already exists.")

    attrs: dict[str, Any] = {"name": name, "source_ids": []}
    if request.node_type == "topic":
        attrs["node_type"] = "cluster"
    builder.graph.add_node(key, **attrs)

    _persist_graph(builder)

    node_data = builder.graph.nodes[key]
    return GraphNodeResponse(
        id=key,
        label=node_data.get("name", key),
        type="topic" if node_data.get("node_type") == "cluster" else "concept",
    )


@app.delete("/api/graph/node/{node_id}")
def delete_graph_node(node_id: str) -> dict[str, Any]:
    """Delete a node and all its edges."""
    builder = _require_graph_builder()
    if not builder.remove_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    _persist_graph(builder)
    return {"deleted": node_id}


@app.post("/api/graph/edge", response_model=GraphEdgeResponse, status_code=201)
def add_graph_edge(request: AddEdgeRequest) -> GraphEdgeResponse:
    """Add a directed edge between two existing nodes."""
    relation = request.relation.strip().upper()
    if not relation:
        raise HTTPException(status_code=422, detail="Relation must not be empty.")

    builder = _require_graph_builder()
    src = request.source.strip().lower()
    tgt = request.target.strip().lower()

    if src not in builder.graph:
        raise HTTPException(status_code=404, detail=f"Source node '{src}' not found.")
    if tgt not in builder.graph:
        raise HTTPException(status_code=404, detail=f"Target node '{tgt}' not found.")

    # Reject duplicate (same head, relation, tail)
    existing = builder.graph.get_edge_data(src, tgt) or {}
    for ed in existing.values():
        if ed.get("relation") == relation:
            raise HTTPException(status_code=409, detail="Edge already exists.")

    builder.graph.add_edge(src, tgt, relation=relation, source_chunk_ids=[])
    _persist_graph(builder)

    edge_id = f"{src}__{relation}__{tgt}"
    return GraphEdgeResponse(id=edge_id, source=src, target=tgt, relation=relation)


@app.delete("/api/graph/edge")
def delete_graph_edge(source: str, relation: str, target: str) -> dict[str, Any]:
    """Delete a specific directed edge identified by source, relation, and target."""
    builder = _require_graph_builder()
    if not builder.remove_edge(source, relation, target):
        raise HTTPException(status_code=404, detail="Edge not found.")

    _persist_graph(builder)
    return {"deleted": f"{source}__{relation}__{target}"}

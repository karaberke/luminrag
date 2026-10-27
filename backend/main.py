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
import io
import dataclasses
import json
import logging
import os
import time
from types import SimpleNamespace
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

# Load .env from repo root before any SDK imports so API keys are available
_env_file = Path(__file__).parent.parent / ".env"
if os.getenv("LUMIN_HOSTED") != "1" and os.getenv("LUMINRAG_SKIP_DOTENV") != "1" and _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.db.document_store import DocumentStore
from backend.utils.associations import AssociationSupport
from backend.utils.auth import (
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    AuthMiddleware,
    evaluate_request,
    get_api_token,
    load_allowed_origins,
    make_session_value,
    tokens_match,
)
from backend.utils.state import (
    bump_epoch,
    bump_revision,
    current_epoch,
    current_revision,
    locked,
    reconcile_graph_sources,
    run_locked,
)
from backend.utils.uploads import (
    AUDIO_EXT,
    DERIVED_ASSET_DIRS,
    IMAGE_EXT,
    PDF_EXT,
    VIDEO_EXT,
    IssuedUploadRegistry,
    StagedFile,
    UploadBodyLimitMiddleware,
    UploadRejected,
    cleanup_attempt_assets,
    copy_bounded,
    download_issued_object,
    find_legacy_raw_source,
    is_safe_source_id,
    max_upload_bytes,
    media_type_for,
    new_job_staging_dir,
    remap_chunks,
    remove_staging,
    resolve_raw_source,
    s3_upload_bucket,
    source_identity,
    staged_path,
    validate_batch_size,
    validate_declared_size,
    validate_file,
)
from backend.utils.recovery import PublicationOutcome, RecoveryError, RecoveryManager
from backend.utils.resources import ModelResourcesBusy
from backend.utils.inference import MAX_QUERY_CHARACTERS, inference_limited
from backend.utils.assets import LegacyReplacement, plan_retired_assets, record_derived_asset_dirs
from backend.utils.provenance import evidence_metadata
from backend.utils.jobs import (
    JobCapacityError,
    REGISTRY_LOCK,
    prune_terminal_jobs,
    register_job,
)
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_retriever import VectorIndex
from backend.retrieval.graph_retriever import reserve_node_embeddings, warm_up as warm_graph_retriever
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
from backend.schemas import Chunk, CitationSpan, GenerationResult, RetrievalResult
from backend.workspace import (
    MAX_BUNDLE_BYTES, MAX_CHUNKS, StateProxy, Workspace, WorkspaceManager,
    WorkspaceMapping, current_workspace, export_bundle, hosted_mode,
    import_bundle, request_config, validate_workspace_data, workspace_id,
    copy_workspace_upload,
)
from backend.credentials import apply_credentials, get_credentials
from backend.utils.hosted import HostedSecurityMiddleware

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
    config = {**llm, **db}
    # An isolated data directory must also replace paths explicitly present in
    # the bundled YAML, so browser tests cannot touch the user's existing data.
    if os.getenv("LUMINRAG_DATA_DIR"):
        for section, name in (
            ("document_store", "chunks.db"), ("vector_db", "faiss.index"), ("graph_db", "graph.json")
        ):
            config[section] = {**config.get(section, {}), "path": str(_workspace_dir("processed") / name)}
    config["data"] = {**config.get("data", {}), "processed_dir": str(_workspace_dir("processed"))}
    return config


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
    storage_error: str | None = None
    startup_complete: bool = False


state = StateProxy(_AppState())
_workspace_manager: WorkspaceManager | None = None
_STORAGE_UNAVAILABLE_DETAIL = "Storage recovery is incomplete. Restart the server after resolving the storage error."


def _require_available_storage() -> None:
    if state.storage_error:
        raise HTTPException(status_code=503, detail=state.storage_error)


def _mark_storage_unavailable(reason: str) -> None:
    """Fence API reads and background graph writes until startup reloads state."""
    state.storage_error = reason
    if state.illustration_worker is not None:
        try:
            state.illustration_worker.suspend()
        except Exception:
            logger.exception("Failed to suspend illustration work after a storage failure")


def _publication_adoption_failure(outcome: PublicationOutcome, committed_message: str) -> RecoveryError | None:
    if outcome.status == "committed":
        message = f"{committed_message}, but the server needs to restart before using it."
    elif outcome.status == "pending":
        message = "Publication needs recovery; restart the server before using the materials."
    else:
        return None
    _mark_storage_unavailable(message)
    return RecoveryError(message)


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
    state.startup_complete = False
    logger.info("LuminRAG starting up …")
    state.config = _load_config()

    if hosted_mode():
        global _workspace_manager
        state.embedder = Embedder(state.config.get("vector_retriever", {}).get("model", "BAAI/bge-small-en-v1.5"))
        _workspace_manager = WorkspaceManager(state.config, state.embedder)
        eviction_task = asyncio.create_task(_workspace_manager.expire_idle())
        state.startup_complete = True
        try:
            yield
        finally:
            state.startup_complete = False
            eviction_task.cancel()
            await asyncio.gather(eviction_task, return_exceptions=True)
            if _background_tasks:
                for task in _background_tasks:
                    task.cancel()
                await asyncio.gather(*_background_tasks, return_exceptions=True)
            _workspace_manager.close()
            _workspace_manager = None
        return

    # Document store
    db_path = state.config.get("document_store", {}).get("path", "backend/data/processed/chunks.db")
    state.store = DocumentStore(db_path)
    scheduler = None
    try:
        # Resolve incomplete publications before loading any derived state or
        # removing staging files that a recovery operation might still require.
        await asyncio.to_thread(run_locked, _recover_storage, _recovery_manager())
        logger.info(f"DocumentStore: {db_path} ({state.store.count()} chunks)")
        state.storage_error = None

        # Embedder (shared across retrieval modules)
        model_name = state.config.get("vector_retriever", {}).get("model", "BAAI/bge-small-en-v1.5")
        state.embedder = Embedder(model_name)

        # SQLite is authoritative after journal recovery. A missing or unusable
        # derived index can be rebuilt in memory without overwriting its disk files.
        index_path = Path(
            state.config.get("vector_db", {}).get("path", "backend/data/processed/faiss.index")
        )
        state.vector_index = VectorIndex(state.embedder)
        state.index_loaded = False
        try:
            state.vector_index.load(index_path)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Derived vector index unavailable (%s); rebuilding from stored documents in memory", exc)
            stored = state.store.get_all_chunks()
            # load() must never leave a partly initialized instance in use.
            state.vector_index = VectorIndex(state.embedder)
            if stored:
                state.vector_index.build(stored)
                state.index_loaded = True
        else:
            state.index_loaded = True
        if state.index_loaded:
            logger.info(f"VectorIndex ready from {index_path}")

        # Graph builder — constructor already loads from disk if the file exists
        graph_path = state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
        state.graph_builder = GraphBuilder(graph_path)
        state.graph_loaded = False
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

        # Staging dirs only hold uploads of jobs from a previous process — drop them.
        if _workspace_dir("staging").is_dir():
            for leftover in _workspace_dir("staging").iterdir():
                remove_staging(leftover, _workspace_dir("staging"))

        scheduler = _start_illustration_scheduler()
        # Re-queue any content nodes that have a hint but no generated image yet
        # (covers crashes or restarts mid-generation).
        try:
            pending = await scheduler.scan_and_enqueue_pending()
            if pending:
                logger.info(f"Illustration worker: {pending} pending node(s) re-queued")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Illustration startup scan failed: {exc}")

        logger.info("LuminRAG ready.")
        state.startup_complete = True
        yield
    finally:
        state.startup_complete = False
        try:
            if scheduler is not None:
                await scheduler.stop()
        finally:
            state.store.close()
        logger.info("LuminRAG shut down.")


def _start_illustration_scheduler() -> IllustrationScheduler:
    """
    Create, start, and publish an illustration scheduler bound to the current
    graph builder. Directory must match the StaticFiles mount so generated
    images are servable.
    """
    scheduler = IllustrationScheduler(
        builder=state.graph_builder,
        doc_store=state.store,
        output_dir=_workspace_dir("illustrations"),
        config=state.config,
    )
    scheduler.start()
    state.illustration_worker = scheduler
    return scheduler


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def _api_docs_enabled() -> bool:
    return os.getenv("LUMINRAG_ENABLE_DOCS") == "1" or (not hosted_mode() and not get_api_token())


app = FastAPI(title="LuminRAG", version="0.1.0", lifespan=lifespan,
              docs_url="/docs" if _api_docs_enabled() else None,
              redoc_url="/redoc" if _api_docs_enabled() else None,
              openapi_url="/openapi.json" if _api_docs_enabled() else None)


@app.exception_handler(ModelResourcesBusy)
async def _model_resources_busy(_request: Request, exc: ModelResourcesBusy):
    return JSONResponse({"detail": str(exc)}, status_code=503, headers={"Retry-After": "2"})


class _StorageAvailabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        data_route = (path == "/api" or path.startswith("/api/")) and path not in {
            "/api/health", "/api/ready", "/api/auth/status", "/api/auth/session"
        }
        if scope["type"] == "http" and state.storage_error and (
            data_route or path.startswith("/static/illustrations")
        ):
            response = JSONResponse(
                {"detail": state.storage_error},
                status_code=503,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _WorkspaceMiddleware:
    """Select a workspace before auth/storage middleware and thread dispatch."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path == "/api/quiz" or path.startswith("/api/quiz/"):
            await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
            return
        req = Request(scope)
        ws = None
        config_token = None
        workspace_token = None
        try:
            if hosted_mode():
                if path.startswith(("/static/", "/api/source/", "/api/upload/", "/api/ingest/from-s3")) or path == "/api/auth/session":
                    raise HTTPException(404, "Not available on the hosted service.")
                if path.startswith("/api/") and path not in {"/api/health", "/api/ready", "/api/auth/status"}:
                    identifier = workspace_id(req.headers.get("X-Workspace-Id"))
                    if _workspace_manager is None:
                        raise HTTPException(503, "Server is starting. Please retry shortly.")
                    ws = _workspace_manager.acquire(identifier)
                    if ws is None and path not in {"/api/workspace/status", "/api/workspace/import"}:
                        response = JSONResponse({"code": "workspace_not_loaded", "detail": "Restore this knowledge base from your browser."}, status_code=409)
                        await response(scope, receive, send)
                        return
                    if ws:
                        workspace_token = current_workspace.set(ws)
                        scope.setdefault("state", {})["workspace_selected"] = True
                    scope.setdefault("state", {})["hosted_workspace_authorized"] = True
            # Keys are read only for operations that call a provider. Browser
            # backup, graph editing and cache hydration need no LLM credentials.
            calls_llm = path in {"/api/query", "/api/ingest"}
            if calls_llm:
                credentials = get_credentials(req)
                config_token = request_config.set(apply_credentials(state.config, credentials))
            await self.app(scope, receive, send)
        except HTTPException as exc:
            await JSONResponse({"detail": exc.detail}, status_code=exc.status_code)(scope, receive, send)
        finally:
            if config_token is not None:
                request_config.reset(config_token)
            if workspace_token is not None:
                current_workspace.reset(workspace_token)
            if ws is not None:
                _workspace_manager.release(ws)

# Middleware order: the last one added is outermost. CORS wraps auth so that
# 401/403 responses still carry CORS headers; the body cap runs innermost,
# i.e. only for authenticated requests.
_ALLOWED_ORIGINS = load_allowed_origins()
app.add_middleware(_StorageAvailabilityMiddleware)
app.add_middleware(UploadBodyLimitMiddleware, paths=("/api/ingest",))
app.add_middleware(AuthMiddleware, allowed_origins=_ALLOWED_ORIGINS)
app.add_middleware(_WorkspaceMiddleware)
app.add_middleware(HostedSecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated illustration images as static files. The directory is
# created eagerly so StaticFiles can mount cleanly even on a cold install.
_DATA_DIR = Path(os.getenv("LUMINRAG_DATA_DIR") or (Path(__file__).parent / "data")).expanduser().resolve()
_PROCESSED_DIR = _DATA_DIR / "processed"
_RAW_DIR = _DATA_DIR / "raw"
_STAGING_DIR = _DATA_DIR / "staging"
_ILLUS_DIR = _PROCESSED_DIR / "illustrations"
if not hosted_mode():
    _ILLUS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static/illustrations",
        StaticFiles(directory=str(_ILLUS_DIR)),
        name="illustrations",
    )


def _workspace_dir(name: str) -> Path:
    workspace = current_workspace.get()
    if workspace is not None:
        return getattr(workspace, name + "_dir")
    return {"raw": _RAW_DIR, "processed": _PROCESSED_DIR, "staging": _STAGING_DIR, "illustrations": _ILLUS_DIR}[name]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARACTERS)
    routing_mode: Literal["dense", "graph", "hybrid"] | None = None
    # None = no cap on returned chunks
    max_sources: int | None = Field(default=None, ge=1, le=1000)
    # None = 0.0 (no cutoff)
    min_relevancy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    query_mode: Literal["fast", "thorough"] = "thorough"


class EvidenceChunkResponse(BaseModel):
    id: str
    text: str
    source: str
    modality: str
    page: int | None = None
    page_end: int | None = None
    slide_number: int | None = None
    source_filename: str | None = None
    title: str | None = None
    section_title: str | None = None
    quote: str | None = None
    quote_start: int | None = None
    quote_end: int | None = None
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
    hint: str = Field(max_length=10_000)


class GraphNodeResponse(BaseModel):
    """
    Rich node payload for the frontend. Different fields are populated
    depending on node_type — the frontend reads the ones it needs.
    """

    id: str
    name: str
    node_type: Literal["topic", "subtopic", "content", "note"]
    source_ids: list[str] = Field(default_factory=list)
    title: str | None = None
    body: str | None = None
    x: float | None = None
    y: float | None = None
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
    supports: list[AssociationSupport] = Field(default_factory=list)
    highlighted: bool = False


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
    association_version: Literal[1] = 1


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
    ingested_source_ids: list[str] = Field(default_factory=list)
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
    status: Literal["queued", "running", "done", "failed", "cancelled"] = "queued"
    progress_stage: str = "Queued"
    result: IngestResponse | None = None
    error: str | None = None
    publication: PublicationOutcome = dataclasses.field(default_factory=PublicationOutcome)


_TERMINAL_JOB_STATUSES = {"done", "failed", "cancelled"}
_jobs: dict[str, _IngestJob] = WorkspaceMapping("ingest_jobs")

# Ingest jobs queue here (one waiting thread at most) before taking the
# process-wide mutation lock inside a single worker thread.
_ingest_queue = asyncio.Lock()
# Serialises clear_data's stop/clear/restart of the illustration scheduler.
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Start a background task and keep a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


_CONTENT_TYPES = {
    "definition", "theorem", "technique", "example", "question", "figure", "other"
}


# ---------------------------------------------------------------------------
# S3 presigned upload helpers
# ---------------------------------------------------------------------------

_s3_client: Any = None

_PRESIGN_EXPIRES_SECONDS = 900
# Issued keys stay claimable a bit longer than the PUT URL is valid.
_issued_uploads = IssuedUploadRegistry(ttl_seconds=3600)


def _get_s3() -> Any:
    global _s3_client
    if _s3_client is None:
        try:
            import boto3
        except ImportError as exc:
            raise UploadRejected(
                "S3 uploads require the optional boto3 package in your self-hosted installation.", status=503,
            ) from exc
        _s3_client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _s3_client


def _s3_error_types() -> tuple[type[Exception], ...]:
    # Evaluated only while handling an SDK failure, after request/bucket/key
    # validation. Local use and test clients need no AWS packages installed.
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return ()
    return BotoCoreError, ClientError


class PresignRequest(BaseModel):
    filename: str = Field(max_length=1024)
    content_type: str = Field(max_length=255)
    size: int = Field(gt=0)  # bytes; also capped by LUMINRAG_MAX_UPLOAD_BYTES


class PresignedFile(BaseModel):
    key: str
    url: str
    filename: str


class IngestFromS3Request(BaseModel):
    keys: list[str]
    slides: list[str] = Field(default_factory=list)


class SessionRequest(BaseModel):
    token: str = Field(max_length=4096)


GraphText = Annotated[str, Field(max_length=500)]


class AddNodeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=500)
    node_type: Literal["topic", "subtopic", "content", "note"] = "topic"
    summary: str | None = Field(default=None, max_length=100_000)
    scope: Literal["broad", "narrow"] = "broad"
    content_type: str | None = Field(default=None, max_length=100)
    illustration: IllustrationPayload | None = None
    # Content-only enrichment
    raw_excerpt: str | None = Field(default=None, max_length=100_000)
    key_terms: list[GraphText] = Field(default_factory=list, max_length=200)
    # Optional auto-linking on insert. Must be existing node keys.
    parent_topic_keys: list[GraphText] = Field(default_factory=list, max_length=200)
    parent_subtopic_keys: list[GraphText] = Field(default_factory=list, max_length=200)
    title: str | None = Field(default=None, max_length=500)
    body: str | None = Field(default=None, max_length=100_000)
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class PatchNodeRequest(BaseModel):
    name: str | None = Field(default=None, max_length=500)
    summary: str | None = Field(default=None, max_length=100_000)
    scope: Literal["broad", "narrow"] | None = None
    content_type: str | None = Field(default=None, max_length=100)
    raw_excerpt: str | None = Field(default=None, max_length=100_000)
    key_terms: list[GraphText] | None = Field(default=None, max_length=200)
    source_ids: list[GraphText] | None = Field(default=None, max_length=20_000)  # full replacement list
    title: str | None = Field(default=None, max_length=500)
    body: str | None = Field(default=None, max_length=100_000)
    node_type: Literal["topic", "subtopic", "content", "note"] | None = None


class AddEdgeRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    target: str = Field(min_length=1, max_length=500)
    relation: str = Field(min_length=1, max_length=50)
    label: str | None = Field(default=None, max_length=200)
    confidence: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks_to_evidence(result: RetrievalResult, citations: list[CitationSpan] | None = None) -> list[EvidenceChunkResponse]:
    """Convert retrieved Chunks into frontend-compatible EvidenceChunk objects."""
    evidence = []
    by_chunk = {}
    for citation in citations or []:
        by_chunk.setdefault(citation.chunk_id, citation)
    for chunk in result.chunks:
        evidence.append(
            EvidenceChunkResponse(
                id=chunk.id,
                source=chunk.source_id,
                modality=chunk.modality,
                **evidence_metadata(chunk, by_chunk.get(chunk.id)),
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
        title=attrs.get("title"), body=attrs.get("body"), x=attrs.get("x"), y=attrs.get("y"),
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
    from backend.utils.associations import association_id

    relation = attrs.get("relation", "")
    confidence = attrs.get("confidence") if relation == "RELATED_TO" else None
    return GraphEdgeResponse(
        id=association_id(source, target) if relation == "RELATED_TO" else f"{source}__{relation}__{target}",
        source=source,
        target=target,
        relation=relation,
        label=None,
        confidence=confidence,
        source_chunk_ids=list(attrs.get("source_chunk_ids", [])),
        supports=list(attrs.get("supports", [])) if relation == "RELATED_TO" else [],
    )


def _export_graph_response() -> GraphResponse:
    """Build a GraphResponse from the loaded GraphBuilder — empty if not loaded."""
    if not state.graph_loaded or state.graph_builder is None:
        return GraphResponse(nodes=[], edges=[])

    from backend.utils.associations import normalize_associations

    graph = state.graph_builder.graph
    normalize_associations(graph)
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
    parts = [header[:_DOC_PREVIEW_CHARS]]
    remaining = _DOC_PREVIEW_CHARS - len(parts[0])
    separator = ""
    for chunk in chunks:
        if remaining <= 0:
            break
        if not chunk.text or chunk.text.isspace():
            continue
        piece = (separator + chunk.text[:remaining])[:remaining]
        parts.append(piece)
        remaining -= len(piece)
        separator = "\n\n"
    return "".join(parts)


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
    *,
    warm_cache: bool = True,
    context_chunks: list[Chunk] | None = None,
) -> "ApplyResult":
    """
    Stages 1.2–1.5 + Stage 2 of the ingest pipeline. All synchronous LLM
    calls and CPU-bound work. Must be called via asyncio.to_thread — never
    directly from an async handler.
    """
    from backend.graph.topic_extractor import extract_topics
    from backend.graph.subtopic_extractor import extract_subtopics
    from backend.graph.content_synthesizer import synthesize_contents
    from backend.graph.semantic_linker import link_graph_changes, snapshot_linkable_texts

    graph = graph_builder.graph
    previous_linkable_texts = snapshot_linkable_texts(graph)

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

    related_proposals = link_graph_changes(
        graph_builder.graph, previous_linkable_texts, embedder, config,
        chunks=context_chunks if context_chunks is not None else all_chunks,
    )
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
    if warm_cache:
        warm_graph_retriever(graph_builder.graph, embedder)
        logger.info("Graph retriever re-warmed.")

    return apply_result


def _index_path() -> Path:
    return Path(
        state.config.get("vector_db", {}).get("path", "backend/data/processed/faiss.index")
    )


def _graph_path() -> Path:
    return Path(
        state.config.get("graph_db", {}).get("path", "backend/data/processed/graph.json")
    )


def _recovery_manager() -> RecoveryManager:
    if state.store is None:
        raise RecoveryError("Document store is not initialized.")
    return RecoveryManager(
        _workspace_dir("processed") / "recovery",
        (_workspace_dir("raw"), _workspace_dir("processed"), _workspace_dir("staging"), _index_path().resolve().parent, _graph_path().resolve().parent),
        state.store,
    )


def _recover_storage(manager: RecoveryManager) -> int:
    try:
        return manager.recover()
    except BaseException:
        _mark_storage_unavailable("Storage recovery did not finish safely.")
        logger.exception("Storage is unavailable until recovery succeeds")
        raise


def _publish_storage(manager: RecoveryManager, files, commit, *, outcome: PublicationOutcome | None = None, expected_sql_bytes: int = 0):
    workspace = current_workspace.get()
    if workspace is not None:
        copies = sum(source.stat().st_size + (destination.stat().st_size if destination.is_file() else 0) for source, destination in files if source is not None)
        workspace.check_disk_budget(copies + expected_sql_bytes)
    outcome = outcome if outcome is not None else PublicationOutcome()
    # Even a failure before publish() returns can follow an actual SQL commit.
    # Cleanup remains unsafe until this token positively proves rollback.
    outcome.status = "pending"
    try:
        value = manager.publish(files, commit, outcome=outcome)
    except BaseException as exc:
        _recover_storage(manager)
        if outcome.status == "committed" and isinstance(exc, Exception):
            logger.warning("Recovered a confirmed storage commit after publication raised: %s", exc)
            return None
        if outcome.status != "rolled_back":
            _mark_storage_unavailable("The storage publication outcome could not be safely adopted.")
        raise
    if outcome.status != "committed":
        _mark_storage_unavailable("Storage publication returned without a confirmed commit.")
        raise RecoveryError("Storage publication returned without a confirmed commit.")
    return value


def _parse_staged_file(staged: StagedFile, config: dict | None = None) -> list[Chunk]:
    """
    Parse without changing a committed source. Raw files are published only
    after the complete candidate index and graph have been prepared.
    """
    chunks = _ingest_single_file(staged.path, config if config is not None else state.config, staged.is_slides)
    if not chunks:
        raise ValueError("no chunks produced")
    dest = (_workspace_dir("raw") / staged.source_id / staged.filename).resolve()
    if _workspace_dir("raw").resolve() not in dest.parents:
        raise UploadRejected("Unsafe source path.")
    rel = dest.relative_to(_workspace_dir("raw").resolve()).as_posix()
    chunks = remap_chunks(chunks, staged.source_id, staged.original_filename, rel)
    chunks = record_derived_asset_dirs(chunks, staged.path, _workspace_dir("processed"))
    for chunk in chunks:
        # Processors that point at the input file must not reference staging.
        if "source_file" in chunk.metadata:
            chunk.metadata["source_file"] = str(dest)
    return chunks


def _legacy_sources_replaced(staged: list[StagedFile], chunks: list[Chunk], stored: list[Chunk]) -> set[str]:
    """Migrate identifiable flat legacy sources without deleting another modality."""
    incoming = {chunk.source_id: chunk.modality for chunk in chunks}
    replaced: set[str] = set()
    modalities_by_source: dict[str, set[str]] = {}
    for previous in stored:
        modalities_by_source.setdefault(previous.source_id, set()).add(previous.modality)
    for item in staged:
        if item.source_id not in incoming:
            continue
        legacy = Path(item.original_filename).stem
        if legacy == item.source_id:
            continue
        original = resolve_raw_source(_workspace_dir("raw"), item.original_filename)
        if original is None or original.parent != _workspace_dir("raw").resolve():
            continue
        previous = modalities_by_source.get(legacy, set())
        compatible = {"pdf", "slide"} if incoming[item.source_id] in {"pdf", "slide"} else {incoming[item.source_id]}
        if previous and previous <= compatible:
            replaced.add(legacy)
    return replaced


@dataclasses.dataclass(frozen=True)
class _IngestSnapshot:
    stored: list[Chunk]
    vector_index: VectorIndex | None
    embedder: Embedder
    config: dict
    builder: GraphBuilder
    index_path: Path
    graph_path: Path
    revision: int


def _snapshot_ingest(job: _IngestJob, epoch: int, job_dir: Path) -> _IngestSnapshot | None:
    _require_available_storage()
    if job.status == "cancelled":
        return None
    if current_epoch() != epoch:
        job.status = "cancelled"
        job.progress_stage = "Cancelled"
        job.error = "Data was cleared before this job started."
        return None
    job.status = "running"
    candidate_dir = job_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return _IngestSnapshot(
        stored=state.store.get_all_chunks(), vector_index=state.vector_index,
        embedder=state.embedder, config=state.config,
        builder=_require_graph_builder().clone(candidate_dir / "graph.json"),
        index_path=_index_path(), graph_path=_graph_path(), revision=current_revision(),
    )


class IngestConflictError(RuntimeError):
    """A known retryable edit conflict with a constant public message."""


def _ingest_snapshot_current(job: _IngestJob, snapshot: _IngestSnapshot, epoch: int) -> bool:
    _require_available_storage()
    if job.status == "cancelled":
        job.progress_stage = "Cancelled"
        return False
    if current_epoch() != epoch:
        job.status = "cancelled"
        job.progress_stage = "Cancelled"
        job.error = "Data was cleared while this job was running."
        return False
    if current_revision() != snapshot.revision:
        raise IngestConflictError("The knowledge base changed during ingestion. Please retry with the updated materials.")
    return True


def _merge_concurrent_illustrations(candidate: GraphBuilder) -> bool:
    """Keep images completed after the snapshot when their content is unchanged."""
    live = state.graph_builder.graph if state.graph_builder is not None else None
    if live is None:
        return False
    changed = False
    excluded = {"illustration_path", "updated_at"}
    for key, attrs in candidate.graph.nodes(data=True):
        current = live.nodes.get(key)
        if current is None or not current.get("illustration_path"):
            continue
        if {k: v for k, v in attrs.items() if k not in excluded} == {k: v for k, v in current.items() if k not in excluded}:
            changed = changed or attrs.get("illustration_path") != current["illustration_path"]
            attrs["illustration_path"] = current["illustration_path"]
            if "updated_at" in current:
                attrs["updated_at"] = current["updated_at"]
    return changed


def _ingest(job: _IngestJob, staged: list[StagedFile], epoch: int, job_dir: Path) -> None:
    """Finish cleanup after a cancelled producer has stopped creating files."""
    try:
        _build_ingest(job, staged, epoch, job_dir)
    finally:
        if job.status == "cancelled":
            run_locked(_cleanup_ingest_job, job, staged, job_dir)


def _build_ingest(job: _IngestJob, staged: list[StagedFile], epoch: int, job_dir: Path) -> None:
    """Build an independent candidate without holding the shared mutation lock."""
    snapshot = run_locked(_snapshot_ingest, job, epoch, job_dir)
    if snapshot is None:
        return
    # Stage 1 — per-file chunking (Ollama vision calls for slides; CPU for PDFs)
    all_chunks: list[Chunk] = []
    failed: list[str] = []
    for item in staged:
        job.progress_stage = f"Chunking {item.filename}…"
        try:
            chunks = _parse_staged_file(item, snapshot.config)
            all_chunks.extend(chunks)
            logger.info(f"  {item.filename} → {len(chunks)} chunk(s)")
        except Exception as exc:  # noqa: BLE001
            logger.error("File ingestion failed (%s)", type(exc).__name__)
            cleanup_attempt_assets(item, _workspace_dir("staging"), _workspace_dir("processed"))
            failed.append(item.original_filename)

    if not all_chunks:
        raise ValueError(f"No chunks produced. Failed: {', '.join(failed) or 'none'}")

    if not run_locked(_ingest_snapshot_current, job, snapshot, epoch):
        return

    # Expensive or fallible work uses a separate candidate until commit.
    replaced_sources = {chunk.source_id for chunk in all_chunks}
    replaced_sources.update(_legacy_sources_replaced(staged, all_chunks, snapshot.stored))
    stored = snapshot.stored
    old_chunk_ids = [chunk.id for chunk in stored if chunk.source_id in replaced_sources]
    candidate_chunks = [chunk for chunk in stored if chunk.source_id not in replaced_sources] + all_chunks
    if hosted_mode() and len(candidate_chunks) > MAX_CHUNKS:
        raise ValueError(f"A workspace supports at most {MAX_CHUNKS} chunks.")
    candidate_dir = job_dir / "candidate"
    candidate_graph_path = candidate_dir / "graph.json"
    builder = snapshot.builder
    if old_chunk_ids:
        stats = reconcile_graph_sources(builder.graph, old_chunk_ids)
        if any(stats.values()):
            logger.info(f"Graph reconciled after source replacement: {stats}")

    job.progress_stage = "Building vector index…"
    candidate_index_path = candidate_dir / "faiss.index"
    new_index = VectorIndex(snapshot.embedder)
    new_index.build(candidate_chunks, reuse_from=snapshot.vector_index, previous_chunks=stored)
    new_index.save(candidate_index_path)

    job.progress_stage = "Extracting topics & subtopics…"
    apply_result = _run_graph_pipeline(
        all_chunks,
        [item.original_filename for item in staged],
        candidate_graph_path,
        snapshot.config,
        builder,
        snapshot.embedder,
        warm_cache=False,
        context_chunks=candidate_chunks,
    )

    if hosted_mode():
        from backend.utils.assets import attach_source_illustrations
        if attach_source_illustrations(builder.graph, candidate_chunks, _workspace_dir("processed")):
            builder.save()
            export_graph(builder.graph, candidate_graph_path)
        validate_workspace_data(candidate_chunks, builder.graph)

    try:
        warm_graph_retriever(builder.graph, snapshot.embedder)
    except Exception as exc:
        apply_result.warnings.append(f"Graph cache warmup failed: {exc}")

    run_locked(
        _publish_ingest, job, staged, epoch, snapshot, all_chunks, failed,
        replaced_sources, candidate_chunks, new_index, candidate_graph_path, candidate_index_path, apply_result,
    )


def _publish_ingest(
    job, staged, epoch, snapshot, all_chunks, failed, replaced_sources, candidate_chunks,
    new_index, candidate_graph_path, candidate_index_path, apply_result,
) -> None:
    if not _ingest_snapshot_current(job, snapshot, epoch):
        return
    builder = snapshot.builder
    index_path, graph_path = snapshot.index_path, snapshot.graph_path
    files: list[tuple[Path | None, Path]] = []
    new_sources = {chunk.source_id for chunk in all_chunks}
    for item in staged:
        if item.source_id in new_sources and not hosted_mode():
            destination = (_workspace_dir("raw") / item.source_id / item.filename).resolve()
            if _workspace_dir("raw").resolve() not in destination.parents:
                raise UploadRejected("Unsafe source path.")
            files.append((item.path, destination))
    files.extend([
        (candidate_index_path, index_path),
        (candidate_index_path.with_suffix(".json"), index_path.with_suffix(".json")),
        (candidate_graph_path, graph_path),
    ])
    if _merge_concurrent_illustrations(builder):
        builder.save()
        export_graph(builder.graph, candidate_graph_path)
    modalities = {chunk.source_id: chunk.modality for chunk in all_chunks}
    legacy_replacements = []
    for item in staged:
        legacy = Path(item.original_filename).stem
        if legacy in replaced_sources and item.source_id in modalities:
            path = resolve_raw_source(_workspace_dir("raw"), item.original_filename)
            if path is not None and path.parent == _workspace_dir("raw").resolve():
                legacy_replacements.append(LegacyReplacement(legacy, modalities[item.source_id], path))
    retired = plan_retired_assets(
        snapshot.stored, candidate_chunks, replaced_sources,
        state.graph_builder.graph, builder.graph,
        raw_root=_workspace_dir("raw"), processed_root=_workspace_dir("processed"),
        legacy_replacements=legacy_replacements,
    )
    destinations = {destination.resolve() for _, destination in files}
    files.extend((None, path) for path in retired if path.resolve() not in destinations)
    final_builder = builder.clone(graph_path)
    prepared_result = IngestResponse(
        files_processed=len(staged) - len(failed),
        chunks_produced=len(all_chunks),
        topics_added=apply_result.topics_added,
        subtopics_added=apply_result.subtopics_added,
        contents_added=apply_result.contents_added,
        related_edges_added=apply_result.related_edges_added,
        graph_nodes=builder.node_count(),
        graph_edges=builder.edge_count(),
        failed=failed,
        ingested_source_ids=sorted({chunk.source_id for chunk in all_chunks}),
        warnings=apply_result.warnings[:20],
    )
    try:
        job.progress_stage = "Publishing documents…"
        _publish_storage(
            _recovery_manager(),
            files,
            lambda transaction_id: state.store.replace_sources(
                all_chunks, replaced_source_ids=replaced_sources, transaction_id=transaction_id
            ),
            outcome=job.publication,
            expected_sql_bytes=2 * sum(len(chunk.model_dump_json().encode("utf-8")) for chunk in all_chunks),
        )

        state.vector_index = new_index
        state.index_loaded = True
        state.graph_builder = final_builder
        state.graph_loaded = final_builder.node_count() > 0
        if state.illustration_worker is not None:
            state.illustration_worker.reset(final_builder, state.store)
        bump_revision()
        job.result = prepared_result
        job.progress_stage = "Done"
        job.status = "done"
    except BaseException as exc:
        failure = _publication_adoption_failure(job.publication, "Upload was saved")
        if failure is not None and isinstance(exc, Exception):
            raise failure from exc
        raise



def _ingest_locked(job: _IngestJob, staged: list[StagedFile], epoch: int, job_dir: Path) -> None:
    """Compatibility entry point for callers already holding the mutation lock."""
    _ingest(job, staged, epoch, job_dir)


async def _run_ingest_job(
    job: _IngestJob,
    staged: list[StagedFile],
    job_dir: Path,
    epoch: int,
) -> None:
    """
    Background task: serializes builds while only snapshot/publication hold
    the mutation lock. Parsing and model calls run in a worker thread.
    """
    try:
        async with _ingest_queue:
            await asyncio.to_thread(_ingest, job, staged, epoch, job_dir)
        if job.status == "done" and state.illustration_worker is not None:
            try:
                await state.illustration_worker.scan_and_enqueue_pending()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Illustration enqueue after ingest failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        from backend.llm import LLMError
        from backend.utils.processes import ProcessingLimitError
        logger.error("Ingest job %s failed (%s)", job.job_id, type(exc).__name__)
        if isinstance(exc, UploadRejected):
            job.error = exc.detail
        elif isinstance(exc, (IngestConflictError, LLMError, ProcessingLimitError)):
            job.error = str(exc)
        else:
            job.error = "Ingestion failed. Please check the file and try again."
        job.progress_stage = "Failed"
        job.status = "failed"
    finally:
        await asyncio.to_thread(run_locked, _cleanup_ingest_job, job, staged, job_dir)


def _cleanup_ingest_job(job: _IngestJob, staged: list[StagedFile], job_dir: Path) -> None:
    """Decide and clean under the same lock used to begin publication.

    Cancellation of the awaiting task does not stop its ingestion thread. The
    decision must remain locked through deletion so that a not-started worker
    cannot begin committing while cleanup removes its derived source images.
    """
    cleanup_safe = job.publication.status in {"not_started", "rolled_back"}
    if cleanup_safe and job.status in {"queued", "running"}:
        # Staging deletion can fail on Windows. Fence a surviving worker even
        # when its raw candidate remains after the derived images are removed.
        job.status = "cancelled"
        job.progress_stage = "Cancelled"
        job.error = "Ingestion was cancelled before publication."
    if job.status != "done" and cleanup_safe:
        for item in staged:
            cleanup_attempt_assets(item, _workspace_dir("staging"), _workspace_dir("processed"))
    if job.status == "done" or cleanup_safe:
        remove_staging(job_dir, _workspace_dir("staging"))
    prune_terminal_jobs(_jobs)


def _admit_ingest_job(job_id: str, staged: list[StagedFile], job_dir: Path, epoch: int) -> IngestJobResponse:
    job = _IngestJob(job_id=job_id)
    try:
        register_job(_jobs, job_id, job)
    except JobCapacityError as exc:
        remove_staging(job_dir, _workspace_dir("staging"))
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    _spawn(_run_ingest_job(job, staged, job_dir, epoch))
    return IngestJobResponse(job_id=job_id, status="queued")


@dataclasses.dataclass(frozen=True)
class _BatchItem:
    filename: str           # sanitised
    original_filename: str
    source_id: str
    is_slides: bool


def _validate_batch(
    items: list[tuple[str | None, str | None]], slide_names: set[str]
) -> list[_BatchItem]:
    """Validate (filename, content_type) pairs; reject duplicates of the same identity."""
    validate_batch_size(len(items))
    out: list[_BatchItem] = []
    seen: set[str] = set()
    for name, content_type in items:
        safe = validate_file(name, content_type)
        sid = source_identity(name or safe)
        if sid in seen:
            raise UploadRejected(f"Duplicate file '{safe}' in batch.")
        seen.add(sid)
        original = (name or safe).replace("\\", "/").rsplit("/", 1)[-1][:255]
        out.append(_BatchItem(safe, original, sid, False))
    originals = {item.original_filename for item in out}
    counts: dict[str, int] = {}
    for item in out:
        counts[item.filename] = counts.get(item.filename, 0) + 1
    return [
        dataclasses.replace(
            item,
            is_slides=item.original_filename in slide_names or (
                item.filename in slide_names
                and item.filename not in originals
                and counts[item.filename] == 1
            ),
        )
        for item in out
    ]


def _chunk_ids_to_evidence(chunk_ids: list[str]) -> list[EvidenceChunkResponse]:
    """Resolve a list of chunk IDs to EvidenceChunkResponse objects via the document store."""
    if not state.store:
        return []
    return _chunks_to_evidence(RetrievalResult(
        chunks=state.store.get_chunks_by_ids(chunk_ids), subgraph=[], routing_mode="none"
    ))


def _ingest_single_file(path: Path, config: dict, is_slides: bool) -> list:
    """Dispatch a single file to the correct ingestion processor."""
    ext = path.suffix.lower()
    from backend.utils.uploads import MARKDOWN_EXT, OFFICE_EXT
    if ext in MARKDOWN_EXT:
        from backend.ingestion.markdown_processor import process_markdown
        return process_markdown(path, config)
    if ext in OFFICE_EXT:
        from backend.ingestion.office_processor import process_office
        return process_office(path, config)

    if ext in VIDEO_EXT:
        from backend.ingestion.video_transcriber import transcribe_video
        return transcribe_video(path, config)

    if ext in AUDIO_EXT:
        from backend.ingestion.audio_processor import process_audio
        return process_audio(path, config)

    if ext in IMAGE_EXT:
        from backend.ingestion.image_processor import process_image
        return process_image(path, config)

    if ext in PDF_EXT:
        if is_slides:
            from backend.ingestion.slide_processor import process_slides
            return process_slides(path, config)
        else:
            from backend.ingestion.pdf_processor import process_pdf
            return process_pdf(path, config)

    return []


@app.get("/api/health")
def health() -> dict[str, Any]:
    # Public endpoint — deliberately reveals nothing about stored data.
    from backend.utils.uploads import SUPPORTED_EXT
    extensions = sorted(SUPPORTED_EXT - (VIDEO_EXT | AUDIO_EXT) if hosted_mode() else SUPPORTED_EXT)
    return {"status": "ok", "features": {"hosted": hosted_mode(), "file_types": extensions, "accepted_extensions": extensions, "max_file_bytes": max_upload_bytes(), "query_modes": ["fast", "thorough"]}}


@app.get("/api/workspace/status")
def workspace_status(request: Request) -> dict:
    ws = current_workspace.get()
    if hosted_mode():
        return {"loaded": ws is not None, "version": ws.version if ws else 0}
    return {"loaded": state.store is not None, "version": current_revision()}


def _bundle_workspace():
    ws = current_workspace.get()
    if ws is not None:
        return ws
    if state.store is None or state.graph_builder is None or state.embedder is None:
        raise HTTPException(503, "Knowledge base is not initialized.")
    return SimpleNamespace(store=state.store, graph_builder=state.graph_builder, vector_index=state.vector_index, embedder=state.embedder, version=current_revision(), processed_dir=_workspace_dir("processed"))


@app.get("/api/workspace/export")
@locked
def workspace_export():
    content = export_bundle(_bundle_workspace())
    return StreamingResponse(io.BytesIO(content), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="knowledge-base.lumin"', "X-Workspace-Version": str(current_revision()), "Access-Control-Expose-Headers": "X-Workspace-Version"})


@app.post("/api/workspace/import")
async def workspace_import(request: Request, file: UploadFile | None = File(default=None), empty: bool = Form(default=False)):
    if file is None and not empty:
        raise HTTPException(422, "Supply a .lumin bundle or empty=true.")
    content = b""
    if file is not None:
        # Bound compressed input too; expanded bytes are checked in read_bundle.
        content = await file.read(MAX_BUNDLE_BYTES + 1)
        if len(content) > MAX_BUNDLE_BYTES:
            raise HTTPException(413, "Bundle exceeds the size limit.")
    elif empty:
        import zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("manifest.json", '{"format":"lumin","version":1}')
            archive.writestr("chunks.jsonl", "")
            archive.writestr("graph.json", '{"nodes":[],"edges":[]}')
        content = buffer.getvalue()

    def restore():
        if hosted_mode():
            identifier = workspace_id(request.headers.get("X-Workspace-Id"))
            previous = current_workspace.get()
            if previous is not None and (previous.active > 1 or any(job.status in {"queued", "running"} for job in previous.ingest_jobs.values())):
                raise HTTPException(409, "Wait for the active request or job before restoring this workspace.")
            candidate = Workspace(identifier, _workspace_manager.config, _workspace_manager.embedder)
            try:
                import_bundle(candidate, content)
                _workspace_manager.adopt(identifier, candidate, previous)
                return {"loaded": True, "version": candidate.version}
            except BaseException:
                candidate.close()
                raise
        # Prepare embeddings and graph in a temporary candidate, then publish
        # using the same crash-safe transaction mechanism as normal ingestion.
        candidate = Workspace("local", state.config, state.embedder, temp_parent=_workspace_dir("staging"))
        try:
            import_bundle(candidate, content)
            chunks = candidate.store.get_all_chunks()
            candidate.graph_builder.save()
            files = [(candidate.graph_builder._path, _graph_path())]
            final_processed = _workspace_dir("processed")
            for asset in candidate.processed_dir.rglob("*"):
                if asset.is_file() and asset.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    relative = asset.relative_to(candidate.processed_dir)
                    if relative.parts[0] in {"illustrations", "pdf_images", "slides", "slide_images", "keyframes"}:
                        files.append((asset, final_processed / relative))
            for chunk in chunks:
                for field in ("source_file", "keyframe_path", "slide_image_path"):
                    value = chunk.metadata.get(field)
                    if isinstance(value, str):
                        try:
                            chunk.metadata[field] = str(final_processed / Path(value).relative_to(candidate.processed_dir))
                        except ValueError:
                            chunk.metadata.pop(field, None)
                chunk.metadata["image_paths"] = [str(final_processed / Path(value).relative_to(candidate.processed_dir)) for value in chunk.metadata.get("image_paths", [])]
            if chunks:
                candidate_path = candidate.processed_dir / "embeddings.npy"
                candidate.vector_index.save(candidate_path)
                files.extend([(candidate_path, _index_path()), (candidate_path.with_suffix(".json"), _index_path().with_suffix(".json"))])
            else:
                files.extend([(None, _index_path()), (None, _index_path().with_suffix(".json"))])
            _publish_storage(_recovery_manager(), files, lambda transaction_id: state.store.replace_sources(chunks, replaced_source_ids={c.source_id for c in state.store.get_all_chunks()}, transaction_id=transaction_id))
            state.vector_index = candidate.vector_index
            state.index_loaded = bool(chunks)
            state.graph_builder = candidate.graph_builder.clone(_graph_path())
            state.graph_loaded = bool(state.graph_builder.node_count())
            if state.illustration_worker is not None:
                state.illustration_worker.reset(state.graph_builder, state.store)
            bump_epoch()
            bump_revision()
            return {"loaded": True, "version": current_revision()}
        finally:
            candidate.close()
    try:
        return await asyncio.to_thread(run_locked, restore)
    except UploadRejected as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    except (ValueError, KeyError, TypeError, OSError, __import__("zipfile").BadZipFile) as exc:
        raise HTTPException(422, "Invalid .lumin bundle.") from exc


@app.get("/api/ready")
async def readiness() -> JSONResponse:
    # An empty knowledge base is ready for uploads. Do not probe the model,
    # acquire the mutation lock, or expose corpus/storage details to monitors.
    ready = state.startup_complete and not state.storage_error
    return JSONResponse(
        {"status": "ready" if ready else "unavailable"},
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Auth endpoints (public; origin-checked by AuthMiddleware)
# ---------------------------------------------------------------------------

@app.get("/api/auth/status")
def auth_status(request: Request) -> dict[str, bool]:
    if hosted_mode():
        # Hosted mode has no application login. Data endpoints still require
        # their validated workspace capability; provider calls require BYOK.
        return {"authenticated": True, "token_required": False}
    return {
        "authenticated": evaluate_request(request).authenticated,
        "token_required": get_api_token() is not None,
    }


@app.post("/api/auth/session")
def create_session(body: SessionRequest, request: Request, response: Response) -> dict[str, bool]:
    """Exchange the API token for a signed HttpOnly session cookie (8h)."""
    token = get_api_token()
    if token is None:
        raise HTTPException(
            status_code=400,
            detail="No API token is configured; the API only accepts loopback clients.",
        )
    if not tokens_match(body.token, token):
        raise HTTPException(status_code=401, detail="Invalid API token.")
    response.set_cookie(
        SESSION_COOKIE,
        make_session_value(token),
        max_age=SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https" or request.headers.get("origin", "").startswith("https://"),
    )
    return {"authenticated": True}


@app.delete("/api/auth/session")
def delete_session(request: Request, response: Response) -> dict[str, bool]:
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https" or request.headers.get("origin", "").startswith("https://"),
    )
    return {"authenticated": False}


@app.get("/api/suggestions")
@locked
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
@locked
def get_graph() -> GraphResponse:
    """Return the concept knowledge graph for the frontend visualization."""
    return _export_graph_response()


@app.get("/api/source/{source_id}")
@locked
def get_source_file(source_id: str):
    """
    Serve the original raw file behind a chunk's ``source_id``.

    New ingests record ``source_path`` (raw/<source_id>/<filename>) in chunk
    metadata; that path is resolved strictly inside ``_workspace_dir("raw")`` and must have
    an allowlisted extension. Sources ingested before that (flat raw/<file>,
    source_id == stem) are still served via a restricted legacy lookup.
    Browsers render PDFs / images inline and stream video / audio via HTTP
    Range (FileResponse handles that automatically).
    """
    _require_available_storage()
    if not is_safe_source_id(source_id):
        raise HTTPException(status_code=400, detail="Invalid source_id.")

    path: Path | None = None
    chunk = state.store.get_first_chunk_by_source(source_id) if state.store else None
    recorded = (chunk.metadata or {}).get("source_path") if chunk else None
    if recorded:
        if Path(recorded).parts[:1] == (source_id,):
            path = resolve_raw_source(_workspace_dir("raw"), recorded)
    elif chunk:
        path = find_legacy_raw_source(_workspace_dir("raw"), source_id)

    if path is None:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")

    # Content-Disposition: inline renders in the browser's viewer; the media
    # type comes from the allowlist, never from sniffing the file.
    return FileResponse(
        path,
        media_type=media_type_for(path),
        filename=path.name,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


def _no_documents_response() -> QueryResponse:
    return QueryResponse(
        content=(
            "No documents have been ingested yet, so there is nothing to search. "
            "Upload course materials first, then ask again."
        ),
        routing_mode="none",
        evidence=[],
        reflection=ReflectionVerdictResponse(
            needs_retrieval=True,
            is_relevant=False,
            is_supported=False,
            is_useful=False,
            reasoning="No vector index or concept graph is available.",
        ),
        hops=[],
    )


@dataclasses.dataclass(frozen=True)
class _PreparedQuery:
    question: str
    retrieval_result: RetrievalResult
    config: dict
    revision: int


@dataclasses.dataclass(frozen=True)
class _QuerySnapshot:
    question: str
    config: dict
    revision: int
    routing_mode: str | None
    vector_index: VectorIndex | None
    graph_builder: GraphBuilder | None
    store: DocumentStore
    embedder: Embedder


def _prepare_query(request: QueryRequest) -> _QuerySnapshot | QueryResponse:
    """Validate and capture references to one knowledge version under the lock."""
    _require_available_storage()
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    # Never let retrieval hit an unbuilt index (it would surface as a 500).
    has_index = state.index_loaded and state.vector_index is not None
    has_graph = (
        state.graph_loaded
        and state.graph_builder is not None
        and state.graph_builder.node_count() > 0
    )
    routing_mode = request.routing_mode
    if routing_mode in ("dense", "hybrid") and not has_index:
        hint = " or switch to graph mode" if has_graph else ""
        raise HTTPException(
            status_code=409,
            detail=f"'{routing_mode}' retrieval needs the vector index, which has not "
            f"been built yet. Ingest documents first{hint}.",
        )
    if routing_mode == "graph" and not has_graph:
        raise HTTPException(
            status_code=409,
            detail="Graph retrieval needs a concept graph, which has not been built yet. "
            "Ingest documents first.",
        )
    if routing_mode is None and not has_index:
        if not has_graph:
            return _no_documents_response()
        routing_mode = "graph"

    return _QuerySnapshot(
        q, {**state.config, "query_mode": request.query_mode}, current_revision(), routing_mode, state.vector_index,
        state.graph_builder, state.store, state.embedder,
    )


def _retrieve_query(snapshot: _QuerySnapshot, question: str, mode: str) -> RetrievalResult:
    from backend.self_rag.multi_hop_reasoner import retrieve_for_mode
    _require_current_revision(snapshot.revision)
    return retrieve_for_mode(
        question, mode, snapshot.vector_index, snapshot.graph_builder,
        snapshot.store, snapshot.embedder, snapshot.config,
    )


@app.post("/api/query", response_model=QueryResponse)
@inference_limited
def query(request: QueryRequest) -> QueryResponse:
    snapshot = run_locked(_prepare_query, request)
    if isinstance(snapshot, QueryResponse):
        return snapshot

    # Decomposition, routing, and reflection may call a provider. Only the
    # synchronous retrieval callback touches shared knowledge under the lock.
    retrieval_result = reason(
        question=snapshot.question,
        vector_index=snapshot.vector_index,
        graph_builder=snapshot.graph_builder,
        store=snapshot.store,
        embedder=snapshot.embedder,
        config=snapshot.config,
        force_routing_mode=snapshot.routing_mode,
        retrieve_callback=lambda subquestion, mode: run_locked(_retrieve_query, snapshot, subquestion, mode),
    )
    run_locked(_require_current_revision, snapshot.revision)

    # Rerank: re-score all merged chunks against the original question with a
    # cross-encoder, apply min_relevancy cutoff, sort, and cap at max_sources.
    from backend.retrieval.reranker import rerank as _rerank
    if retrieval_result.chunks:
        reranked_chunks = _rerank(
            query=snapshot.question,
            chunks=retrieval_result.chunks,
            config=snapshot.config,
            min_relevancy=request.min_relevancy if request.min_relevancy is not None else 0.0,
            max_sources=request.max_sources,
        )
        retrieval_result = retrieval_result.model_copy(update={"chunks": reranked_chunks})

    prepared = _PreparedQuery(snapshot.question, retrieval_result, snapshot.config, snapshot.revision)

    # Generation and answer reflection consume detached retrieval models, so
    # their network waits need not block graph reads or another question.
    gen: GenerationResult = generate(
        question=prepared.question,
        retrieval_result=prepared.retrieval_result,
        config=prepared.config,
    )

    return run_locked(_finish_query, request, prepared, gen)


def _require_current_revision(revision: int) -> None:
    _require_available_storage()
    if revision != current_revision():
        raise HTTPException(
            status_code=409,
            detail="The knowledge base changed during generation. Please retry with the updated materials.",
        )


def _finish_query(request: QueryRequest, prepared: _PreparedQuery, gen: GenerationResult) -> QueryResponse:
    _require_current_revision(prepared.revision)

    evidence = _chunks_to_evidence(prepared.retrieval_result, gen.citations)
    hops = _subgraph_to_hops(prepared.retrieval_result)

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


def _clear_data_locked() -> dict[str, Any]:
    """Wipe all derived + uploaded data. Must run inside run_locked()."""
    # Reserve before touching files/SQLite so a busy background cache never
    # causes a failure after a clear has already committed.
    with reserve_node_embeddings():
        return _clear_data_reserved()


def _clear_data_reserved() -> dict[str, Any]:
    _require_available_storage()
    manager = _recovery_manager()
    # An older journal must be resolved before enumerating files for this clear.
    _recover_storage(manager)
    index_path = _index_path()
    graph_path = _graph_path()
    paths = {index_path.resolve(), index_path.with_suffix(".json").resolve(), graph_path.resolve()}

    def regular_files(root: Path) -> list[Path]:
        if not root.exists():
            return []
        found: list[Path] = []
        for directory, folders, names in os.walk(root, followlinks=False):
            for path in (Path(directory), *(Path(directory) / name for name in folders + names)):
                if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                    raise RecoveryError("Clear cannot traverse a linked storage directory or file.")
            found.extend(Path(directory) / name for name in names)
        return found

    raw_files = regular_files(_workspace_dir("raw"))
    illustration_files = regular_files(_workspace_dir("illustrations"))
    paths.update(path.resolve() for path in raw_files + illustration_files)
    for derived_dir in DERIVED_ASSET_DIRS:
        paths.update(path.resolve() for path in regular_files(_workspace_dir("processed") / derived_dir))
    before_count = state.store.count()
    outcome = PublicationOutcome()
    try:
        deleted_chunks = _publish_storage(
            manager,
            [(None, path) for path in sorted(paths)],
            lambda transaction_id: state.store.clear_all(transaction_id=transaction_id),
            outcome=outcome,
        )
        if deleted_chunks is None:
            deleted_chunks = before_count  # callback may raise after committing

        # Only committed clears invalidate admitted jobs and the in-memory state.
        bump_epoch()
        bump_revision()
        state.vector_index = VectorIndex(state.embedder)
        state.index_loaded = False
        from backend.retrieval.graph_retriever import clear_node_embeddings
        if current_workspace.get() is not None and state.graph_builder is not None:
            clear_node_embeddings(state.graph_builder.graph)
        else:
            clear_node_embeddings()
        state.graph_builder = GraphBuilder(graph_path)  # starts empty since file is gone
        state.graph_loaded = False
        if state.illustration_worker is not None:
            state.illustration_worker.reset(state.graph_builder, state.store)
        deleted_files = len(raw_files)
        deleted_illustrations = len(illustration_files)

        with REGISTRY_LOCK:
            # Active jobs retain their admission slots until the worker observes
            # the changed epoch and exits; repeated clears cannot bypass the cap.
            prune_terminal_jobs(_jobs, max_completed=0)
        _issued_uploads.clear()

        logger.info(
            f"Data cleared: {deleted_chunks} chunks, {deleted_files} raw files, "
            f"{deleted_illustrations} illustrations, index + graph reset."
        )
        return {
            "deleted_chunks": deleted_chunks,
            "deleted_raw_files": deleted_files,
        }
    except BaseException as exc:
        failure = _publication_adoption_failure(outcome, "Data was cleared")
        if failure is not None and isinstance(exc, Exception):
            raise failure from exc
        raise


@app.delete("/api/data")
async def clear_data() -> dict[str, Any]:
    """
    Wipe all stored data:
      - DocumentStore (all chunks)
      - FAISS vector index (file + in-memory)
      - Concept graph (file + in-memory)
      - Raw uploaded files and generated illustrations
      - Finished ingest jobs
    The illustration scheduler is rebound to the empty graph under the same
    lock and invalidates old work before it can write stale results.
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")

    try:
        return await asyncio.to_thread(run_locked, _clear_data_locked)
    except RecoveryError as exc:
        if state.storage_error:
            raise HTTPException(status_code=503, detail=state.storage_error) from exc
        raise


@app.post("/api/upload/presign", response_model=list[PresignedFile])
def presign_uploads(files: list[PresignRequest]) -> list[PresignedFile]:
    """
    Generate short-lived S3 presigned PUT URLs so the frontend can upload
    files directly to S3, bypassing the backend (and Cloudflare) entirely.

    Each file gets a unique key: uploads/<uuid>/<filename>. Only keys issued
    here (with their declared size) are later accepted by /api/ingest/from-s3.
    """
    try:
        bucket = s3_upload_bucket()
        checked = _validate_batch([(f.filename, f.content_type) for f in files], set())
        limit = max_upload_bytes()
        for f in files:
            validate_declared_size(f.size, limit)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

    result: list[PresignedFile] = []
    try:
        s3 = _get_s3()
        for item, f in zip(checked, files):
            record = _issued_uploads.issue(item.filename, f.size, original_filename=item.original_filename)
            url = s3.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": bucket,
                    "Key": record.key,
                    "ContentType": f.content_type,
                    "ContentLength": f.size,
                },
                ExpiresIn=_PRESIGN_EXPIRES_SECONDS,
            )
            result.append(PresignedFile(key=record.key, url=url, filename=item.filename))
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    except _s3_error_types() as exc:
        logger.error(f"S3 presign failed: {exc}")
        raise HTTPException(status_code=502, detail="S3 presign failed.") from exc
    return result


@app.post("/api/ingest/from-s3", response_model=IngestJobResponse, status_code=202)
async def ingest_from_s3(req: IngestFromS3Request) -> IngestJobResponse:
    """
    Download server-issued uploads from the dedicated S3 bucket into a job
    staging dir, then kick off the ingestion pipeline as a background job.
    Returns a job_id immediately; poll GET /api/ingest/jobs/{job_id} for
    status and results.
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")
    epoch = current_epoch()

    # All key validation happens before any network call.
    try:
        bucket = s3_upload_bucket()
        records = _issued_uploads.claim(req.keys)
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    try:
        items = _validate_batch([(r.original_filename or r.filename, None) for r in records], set(req.slides))
    except UploadRejected as exc:
        _issued_uploads.release(records)
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc

    job_id = str(uuid.uuid4())
    job_dir = new_job_staging_dir(_workspace_dir("staging"), job_id)
    limit = max_upload_bytes()
    staged: list[StagedFile] = []
    try:
        s3 = _get_s3()
        for i, (record, item) in enumerate(zip(records, items)):
            dest = staged_path(job_dir, i, f"{item.source_id}_{job_id}{Path(item.filename).suffix.lower()}")
            await asyncio.to_thread(download_issued_object, s3, bucket, record, dest, limit)
            staged.append(StagedFile(dest, item.filename, item.original_filename, item.source_id, item.is_slides))
            logger.info(f"Downloaded from S3: {record.key} → {dest}")
    except UploadRejected as exc:
        _issued_uploads.release(records)
        remove_staging(job_dir, _workspace_dir("staging"))
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    except _s3_error_types() as exc:
        _issued_uploads.release(records)
        remove_staging(job_dir, _workspace_dir("staging"))
        logger.error(f"S3 download failed: {exc}")
        raise HTTPException(status_code=502, detail="S3 download failed.") from exc
    except BaseException:
        _issued_uploads.release(records)
        remove_staging(job_dir, _workspace_dir("staging"))
        raise

    try:
        return _admit_ingest_job(job_id, staged, job_dir, epoch)
    except HTTPException:
        # Capacity rejection did not admit work; allow the already uploaded
        # objects to be retried without uploading them to S3 a second time.
        _issued_uploads.release(records)
        raise


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

    `slides` is a JSON array (or legacy comma-separated list) of PDF filenames to be
    treated as slide decks (captioned per-page) instead of text PDFs.
    """
    if not state.store:
        raise HTTPException(status_code=503, detail="Document store not initialized.")
    epoch = current_epoch()

    try:
        if slides.lstrip().startswith("["):
            slide_list = json.loads(slides)
            if not isinstance(slide_list, list) or not all(isinstance(name, str) for name in slide_list):
                raise ValueError("slides must be a JSON array of filenames")
            slide_names = set(slide_list)
        else:
            slide_names = {name.strip() for name in slides.split(",") if name.strip()}
        items = _validate_batch([(u.filename, u.content_type) for u in files], slide_names)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, UploadRejected):
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        raise HTTPException(status_code=422, detail="slides must be a JSON array of filenames.") from exc

    # Stage each upload in its own slot with a bounded streaming copy; nothing
    # reaches raw/ until the file has been parsed successfully.
    job_id = str(uuid.uuid4())
    job_dir = new_job_staging_dir(_workspace_dir("staging"), job_id)
    limit = max_upload_bytes()
    staged: list[StagedFile] = []
    try:
        for i, (upload, item) in enumerate(zip(files, items)):
            dest = staged_path(job_dir, i, f"{item.source_id}_{job_id}{Path(item.filename).suffix.lower()}")
            await asyncio.to_thread(copy_workspace_upload, upload.file, dest, limit)
            staged.append(StagedFile(dest, item.filename, item.original_filename, item.source_id, item.is_slides))
            logger.info(f"Staged uploaded file: {dest}")
    except UploadRejected as exc:
        remove_staging(job_dir, _workspace_dir("staging"))
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    except BaseException:
        remove_staging(job_dir, _workspace_dir("staging"))
        raise

    return _admit_ingest_job(job_id, staged, job_dir, epoch)


# ---------------------------------------------------------------------------
# Graph edit endpoints
# ---------------------------------------------------------------------------

def _require_graph_builder() -> GraphBuilder:
    """Return the live GraphBuilder, initialising a fresh one if needed."""
    _require_available_storage()
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
    # Callers have already changed the in-memory graph. Invalidate detached
    # answers even if persistence fails, since subsequent reads see that edit.
    bump_revision()
    builder.save()
    export_graph(builder.graph, graph_path)
    state.graph_loaded = True
    if state.embedder:
        try:
            warm_graph_retriever(builder.graph, state.embedder)
        except ModelResourcesBusy:
            logger.info("Graph saved; embedding cache refresh deferred until the model is available")


class GraphOpsRequest(BaseModel):
    ops: list[dict[str, Any]] = Field(min_length=1, max_length=500)


@app.post("/api/graph/ops")
@locked
def graph_ops(request: GraphOpsRequest) -> dict:
    """Apply an ordered batch atomically and return sparse inverse operations."""
    from backend.utils.graph_ops import apply_op, inverse_budget, note_chunks
    from backend.workspace import MAX_NODES, MAX_EDGES
    builder = _require_graph_builder()
    directory = _workspace_dir("staging") / ("ops-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = builder.clone(directory / "graph.json")
    inverses = []
    budget = inverse_budget()
    outcome = PublicationOutcome()
    try:
        for operation in request.ops:
            inverses.append(apply_op(candidate, operation, budget))
        if candidate.node_count() > MAX_NODES or candidate.edge_count() > MAX_EDGES:
            raise ValueError("Workspace graph size limit exceeded.")
        previous = state.store.get_all_chunks() if state.store is not None else []
        notes = note_chunks(candidate.graph)
        chunks = [chunk for chunk in previous if chunk.modality != "note"] + notes
        if len(chunks) > MAX_CHUNKS:
            raise ValueError("Workspace chunk limit exceeded.")
        validate_workspace_data(chunks, candidate.graph)
        chunk_ids = {chunk.id for chunk in chunks}
        for _, attrs in candidate.graph.nodes(data=True):
            attrs["source_ids"] = [cid for cid in attrs.get("source_ids", []) if not cid.startswith("note:") or cid in chunk_ids]
        changed_notes = [chunk for chunk in previous if chunk.modality == "note"] != notes
        new_index = state.vector_index
        candidate.save()
        files = [(candidate._path, _graph_path())]
        if changed_notes:
            new_index = VectorIndex(state.embedder)
            if chunks:
                new_index.build(chunks, reuse_from=state.vector_index, previous_chunks=previous)
                index_file = directory / "embeddings.npy"
                new_index.save(index_file)
                files.extend([(index_file, _index_path()), (index_file.with_suffix(".json"), _index_path().with_suffix(".json"))])
            else:
                files.extend([(None, _index_path()), (None, _index_path().with_suffix(".json"))])
        if state.store is None:
            raise HTTPException(503, "Document store is not initialized.")
        if changed_notes:
            replaced = {chunk.source_id for chunk in previous if chunk.modality == "note"}
            commit = lambda transaction_id: state.store.replace_sources(notes, replaced_source_ids=replaced, transaction_id=transaction_id)
        else:
            commit = lambda transaction_id: state.store.replace_sources([], transaction_id=transaction_id)
        _publish_storage(_recovery_manager(), files, commit, outcome=outcome, expected_sql_bytes=2 * sum(len(chunk.model_dump_json().encode("utf-8")) for chunk in notes) if changed_notes else 0)
        state.graph_builder = candidate.clone(_graph_path())
        state.graph_loaded = bool(candidate.node_count())
        state.vector_index = new_index
        if changed_notes:
            state.index_loaded = bool(chunks)
        bump_revision()
        # Lazy graph-owned embedding caches see changed signatures on next read.
        # Position-only edits never need a model inference or cache warm-up.
        inverse = list(reversed(inverses))
        return {"version": current_revision(), "inverses": inverse, "inverse": inverse, "results": [{"inverse": item} for item in inverses], "graph": _export_graph_response().model_dump()}
    except UploadRejected as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if outcome.status != "pending":
            remove_staging(directory, _workspace_dir("staging"))


@app.get("/api/graph/duplicates")
@inference_limited
def graph_duplicates(threshold: float = 0.9, limit: int = 100) -> dict:
    import numpy as np
    from backend.retrieval.graph_retriever import get_node_embeddings
    if not 0.5 <= threshold <= 1.0 or not 1 <= limit <= 500:
        raise HTTPException(422, "Invalid duplicate threshold or limit.")
    def snapshot():
        graph = _require_graph_builder().graph
        # The snapshot owns mutable graph attributes while the matrix below is
        # computed. Its original graph key keeps embedding-cache reuse local.
        nodes = [(key, attrs.get("node_type")) for key, attrs in graph.nodes(data=True) if attrs.get("node_type") != "chunk_ref"]
        vectors = get_node_embeddings([item[0] for item in nodes], graph, state.embedder)
        return nodes, vectors
    nodes, all_vectors = run_locked(snapshot)
    pairs = []
    for node_type in ("topic", "subtopic", "content", "note"):
        indices = [index for index, node in enumerate(nodes) if node[1] == node_type]
        selected = [nodes[index] for index in indices]
        if len(selected) < 2:
            continue
        vectors = all_vectors[indices]
        # Bounded blocks retain exact cosine comparisons without allocating an
        # N x N matrix for large knowledge bases.
        for start in range(0, len(selected), 128):
            similarities = vectors[start:start + 128] @ vectors.T
            for row, scores in enumerate(similarities):
                index = start + row
                columns = np.flatnonzero(scores[index + 1:] >= threshold)[:limit - len(pairs)] + index + 1
                for column in columns:
                    pairs.append({"keep": selected[index][0], "drop": selected[int(column)][0], "similarity": float(scores[column])})
                if len(pairs) >= limit:
                    break
            if len(pairs) >= limit:
                break
        if len(pairs) >= limit:
            break
    return {"pairs": sorted(pairs, key=lambda pair: pair["similarity"], reverse=True)[:limit]}


@app.post("/api/graph/node", response_model=GraphNodeResponse, status_code=201)
@locked
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
    for parent in [*request.parent_topic_keys, *request.parent_subtopic_keys]:
        if parent not in builder.graph:
            raise HTTPException(status_code=404, detail=f"Parent node '{parent}' not found.")
    node = request.model_dump(exclude_none=True)
    node.update(name=name, id=key)
    graph_ops(GraphOpsRequest(ops=[{"op": "add_node", "node": node}]))
    return _node_to_response(key, _require_graph_builder().graph.nodes[key])


@app.get("/api/graph/node/{node_id}", response_model=GraphNodeDetailResponse)
@locked
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

    return GraphNodeDetailResponse(**base.model_dump(), evidence=_chunk_ids_to_evidence(chunk_ids))


@app.delete("/api/graph/node/{node_id}")
@locked
def delete_graph_node(node_id: str) -> dict[str, Any]:
    if node_id not in _require_graph_builder().graph:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    graph_ops(GraphOpsRequest(ops=[{"op": "delete_node", "id": node_id}]))
    return {"deleted": node_id}


@app.patch("/api/graph/node/{node_id}")
@locked
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
    if node_id not in builder.graph:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    graph_ops(GraphOpsRequest(ops=[{"op": "update_node", "id": node_id, "updates": updates}]))
    return {"updated": node_id}


@app.post("/api/graph/edge", response_model=GraphEdgeResponse, status_code=201)
@locked
def add_graph_edge(request: AddEdgeRequest) -> GraphEdgeResponse:
    """
    Add a structural (HAS_SUBTOPIC / HAS_CONTENT / EVIDENCE_OF) or semantic
    (RELATED_TO) connection between two existing nodes. Legacy labels are ignored.
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
        graph_ops(GraphOpsRequest(ops=[{"op": "add_edge", "edge": {"source": src, "target": tgt, "relation": relation}}]))
        edge_id = f"{src}__{relation}__{tgt}"
        return GraphEdgeResponse(id=edge_id, source=src, target=tgt, relation=relation)

    if relation == "RELATED_TO":
        from backend.utils.associations import association_id, association_pair
        graph_ops(GraphOpsRequest(ops=[{"op": "add_edge", "edge": {"source": src, "target": tgt,
                             "relation": relation, "confidence": request.confidence}}]))
        source, target = association_pair(src, tgt)
        attrs = _require_graph_builder().graph[source][target][association_id(src, tgt)]
        return _edge_to_response(source, target, attrs)

    raise HTTPException(
        status_code=422,
        detail=f"Unknown relation '{relation}'. Use one of: "
        f"{sorted(STRUCTURAL_RELATIONS | {'RELATED_TO'})}",
    )


@app.delete("/api/graph/edge")
@locked
def delete_graph_edge(
    source: Annotated[str, Field(max_length=500)], relation: Annotated[str, Field(max_length=50)],
    target: Annotated[str, Field(max_length=500)], label: Annotated[str | None, Field(max_length=200)] = None,
) -> dict[str, Any]:
    """
    Delete a hierarchy edge or the complete unordered concept connection.
    Legacy labels remain accepted but do not select individual backing records.
    """
    builder = _require_graph_builder()
    relation = relation.strip().upper()
    from backend.utils.associations import association_edges, association_id
    found = bool(association_edges(builder.graph, source, target)) if relation == "RELATED_TO" else any(
        attrs.get("relation") == relation and (label is None or attrs.get("label") == normalise_relation_label(label))
        for attrs in (builder.graph.get_edge_data(source, target) or {}).values())
    if not found:
        raise HTTPException(status_code=404, detail="Edge not found.")
    graph_ops(GraphOpsRequest(ops=[{"op": "delete_edge", "edge": {"source": source, "target": target,
                         "relation": relation, "label": label}}]))
    return {"deleted": association_id(source, target) if relation == "RELATED_TO" else f"{source}__{label or relation}__{target}"}

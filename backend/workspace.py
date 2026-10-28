"""Ephemeral, capability-addressed knowledge workspaces and portable bundles.

Only model weights are shared. Documents, graph, jobs and versions belong to
one UUID workspace. ContextVars propagate through asyncio.to_thread and tasks.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from collections.abc import MutableMapping
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import networkx as nx
import numpy as np
from fastapi import HTTPException, Request

from backend.db.document_store import DocumentStore
from backend.graph.graph_builder import GraphBuilder
from backend.graph.schema import (
    EdgeInput, MAX_GRAPH_NAME, NODE_SCHEMA_VERSION, legacy_node_schema,
    normalize_legacy_edge_attrs, normalize_legacy_node_attrs, validate_node_attrs,
)
from backend.retrieval.vector_retriever import VectorIndex
from backend.schemas import Chunk, MAX_CHUNK_TEXT_CHARACTERS
from backend.utils.associations import normalize_associations
from backend.utils.environment import hosted_mode

MAX_CHUNKS = 20_000
MAX_NODES = 30_000
MAX_EDGES = 100_000
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_MEMBERS = 20_000
ASSET_DIRECTORIES = frozenset({"illustrations", "pdf_images", "slides", "slide_images", "keyframes"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
MAX_MEMBER_BYTES = {
    "manifest.json": 64 * 1024,
    "chunks.jsonl": 64 * 1024 * 1024,
    "graph.json": 32 * 1024 * 1024,
    "embedding_ids.json": 4 * 1024 * 1024,
    "embeddings.npy": 40 * 1024 * 1024,
}
current_workspace: ContextVar[Workspace | None] = ContextVar("workspace", default=None)
request_config: ContextVar[dict | None] = ContextVar("workspace_config", default=None)


def validate_workspace_data(chunks: list[Chunk], graph: nx.MultiDiGraph) -> None:
    """Keep each cache within the same bounds as its portable bundle."""
    if len(chunks) > MAX_CHUNKS or graph.number_of_nodes() > MAX_NODES or graph.number_of_edges() > MAX_EDGES:
        raise ValueError("Workspace size limit exceeded.")
    for key, attrs in graph.nodes(data=True):
        if not isinstance(key, str) or not key or len(key) > MAX_GRAPH_NAME:
            raise ValueError("Invalid graph node identifier.")
        validate_node_attrs(attrs)
    for _, _, attrs in graph.edges(data=True):
        EdgeInput.model_validate(attrs)
    normalize_associations(graph)
    chunk_ids = {chunk.id for chunk in chunks}
    for source, target, attrs in graph.edges(data=True):
        if attrs.get("relation") != "RELATED_TO":
            continue
        for support in attrs["supports"]:
            if not support["verified"]:
                continue
            supplied = set(support["source_chunk_ids"])
            if (not supplied <= chunk_ids
                    or not supplied.intersection(graph.nodes[source].get("source_ids", []))
                    or not supplied.intersection(graph.nodes[target].get("source_ids", []))):
                raise ValueError("Verified connection passages must exist and ground both concepts.")
    text_bytes = 0
    for chunk in chunks:
        if len(chunk.text) > MAX_CHUNK_TEXT_CHARACTERS:
            raise ValueError("Chunk text exceeds the size limit.")
        text_bytes += len(chunk.model_dump_json().encode("utf-8")) + 1
        if text_bytes > MAX_MEMBER_BYTES["chunks.jsonl"]:
            raise ValueError("Workspace text exceeds the size limit.")
    graph_bytes = len(json.dumps(nx.node_link_data(graph), allow_nan=False).encode("utf-8"))
    if graph_bytes > MAX_MEMBER_BYTES["graph.json"]:
        raise ValueError("Workspace graph content exceeds the size limit.")


def max_workspace_bytes() -> int:
    try:
        configured = int(os.getenv("LUMIN_MAX_WORKSPACE_BYTES", str(256 * 1024 * 1024)))
    except ValueError:
        configured = 256 * 1024 * 1024
    return max(1, min(1024 * 1024 * 1024, configured))


def workspace_id(value: str | None) -> str:
    try:
        parsed = uuid.UUID(value or "")
    except (ValueError, AttributeError):
        raise HTTPException(400, "X-Workspace-Id must be a UUID4.") from None
    if parsed.version != 4 or str(parsed) != value:
        raise HTTPException(400, "X-Workspace-Id must be a canonical UUID4.")
    return str(parsed)


class Workspace:
    def __init__(self, identifier: str, config: dict, embedder, *, temp_parent: Path | None = None) -> None:
        self.id = identifier
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="lumin-workspace-", dir=temp_parent)
        self.root = Path(self._temporary.name)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.staging_dir = self.root / "staging"
        self.illustrations_dir = self.processed_dir / "illustrations"
        for directory in (self.raw_dir, self.processed_dir, self.staging_dir, self.illustrations_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.config = deepcopy(config)
        for section, filename in (("document_store", "chunks.db"), ("vector_db", "embeddings.npy"), ("graph_db", "graph.json")):
            self.config[section] = {**self.config.get(section, {}), "path": str(self.processed_dir / filename)}
        self.config["data"] = {**self.config.get("data", {}), "processed_dir": str(self.processed_dir)}
        self.store = DocumentStore(self.processed_dir / "chunks.db")
        self.embedder = embedder
        self.vector_index = VectorIndex(embedder)
        self.graph_builder = GraphBuilder(self.processed_dir / "graph.json")
        self.node_embedding_cache: dict = {}
        self.ingest_jobs: dict = {}
        self.lock = asyncio.Lock()
        self.mutation_lock = threading.RLock()
        self.illustration_worker = None
        self.index_loaded = False
        self.graph_loaded = False
        self.storage_error = None
        self.startup_complete = True
        self.version = 0
        self.epoch = 0
        self.last_used = time.monotonic()
        self.active = 0
        self.retired = False

    @property
    def busy(self) -> bool:
        return self.active > 0 or any(job.status in {"queued", "running"} for job in self.ingest_jobs.values())

    def disk_bytes(self) -> int:
        total = 0
        for directory, folders, names in os.walk(self.root, followlinks=False):
            for name in (*folders, *names):
                path = Path(directory) / name
                if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                    raise ValueError("Workspace contains a linked storage path.")
            total += sum((Path(directory) / name).stat().st_size for name in names)
        return total

    def check_disk_budget(self, additional_bytes: int = 0) -> None:
        if self.disk_bytes() + additional_bytes > max_workspace_bytes():
            from backend.utils.uploads import UploadRejected
            raise UploadRejected("This knowledge base exceeds the workspace storage limit. Remove materials or use another knowledge base.", status=413)

    def close(self) -> None:
        self.store.close()
        self._temporary.cleanup()


def copy_workspace_upload(source, destination: Path, limit: int) -> int:
    from backend.utils.uploads import copy_bounded, UploadRejected
    from backend.utils.resources import reserve_model_resource
    ws = current_workspace.get()
    if ws is None:
        return copy_bounded(source, destination, limit)
    with reserve_model_resource(ws.mutation_lock):
        remaining = max_workspace_bytes() - ws.disk_bytes()
        if remaining <= 0:
            raise UploadRejected("Workspace storage limit exceeded.", status=413)
        return copy_bounded(source, destination, min(limit, remaining))


class WorkspaceManager:
    def __init__(self, config: dict, embedder, *, max_workspaces: int | None = None, idle_seconds: float = 1800) -> None:
        self.config = config
        self.embedder = embedder
        self.max_workspaces = max(1, min(64, max_workspaces if max_workspaces is not None else int(os.getenv("LUMIN_MAX_WORKSPACES", "8"))))
        self.idle_seconds = idle_seconds
        self._workspaces: OrderedDict[str, Workspace] = OrderedDict()
        self._lock = threading.RLock()

    def evict(self) -> None:
        with self._lock:
            now = time.monotonic()
            for identifier, ws in list(self._workspaces.items()):
                if not ws.busy and now - ws.last_used >= self.idle_seconds:
                    self._workspaces.pop(identifier)
                    ws.close()

    async def expire_idle(self) -> None:
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(self.evict)

    def get(self, identifier: str) -> Workspace | None:
        with self._lock:
            self.evict()
            ws = self._workspaces.get(identifier)
            if ws:
                ws.last_used = time.monotonic()
                self._workspaces.move_to_end(identifier)
            return ws

    def create(self, identifier: str) -> Workspace:
        with self._lock:
            existing = self.get(identifier)
            if existing:
                return existing
            if len(self._workspaces) >= self.max_workspaces:
                disposable = next((key for key, ws in self._workspaces.items() if not ws.busy), None)
                if disposable is None:
                    raise HTTPException(503, "All workspaces are busy. Please retry shortly.")
                self._workspaces.pop(disposable).close()
            ws = Workspace(identifier, self.config, self.embedder)
            self._workspaces[identifier] = ws
            return ws

    def acquire(self, identifier: str) -> Workspace | None:
        """Lookup and pin atomically so eviction cannot close an active cache."""
        with self._lock:
            ws = self.get(identifier)
            if ws is not None:
                ws.active += 1
            return ws

    def release(self, ws: Workspace) -> None:
        with self._lock:
            ws.active -= 1
            ws.last_used = time.monotonic()
            if ws.retired and not ws.busy:
                ws.close()

    def adopt(self, identifier: str, candidate: Workspace, previous: Workspace | None) -> None:
        """Publish a fully validated import; a failed candidate never evicts data."""
        with self._lock:
            live = self._workspaces.get(identifier)
            if live is not previous:
                raise HTTPException(409, "The workspace changed while it was being restored. Please retry.")
            if previous is not None:
                if previous.active > 1 or any(job.status in {"queued", "running"} for job in previous.ingest_jobs.values()):
                    raise HTTPException(409, "Wait for the active request or job before restoring this workspace.")
                candidate.version = max(candidate.version, previous.version + 1)
                previous.retired = True
                if not previous.busy:
                    previous.close()
            elif len(self._workspaces) >= self.max_workspaces:
                disposable = next((key for key, ws in self._workspaces.items() if not ws.busy), None)
                if disposable is None:
                    raise HTTPException(503, "All workspaces are busy. Please retry shortly.")
                self._workspaces.pop(disposable).close()
            self._workspaces[identifier] = candidate
            self._workspaces.move_to_end(identifier)

    def close(self) -> None:
        with self._lock:
            for ws in self._workspaces.values():
                ws.close()
            self._workspaces.clear()


class StateProxy:
    """Keep self-host APIs/tests compatible while selecting hosted state."""
    def __init__(self, local) -> None:
        object.__setattr__(self, "local", local)

    def __getattr__(self, name):
        if name == "config" and request_config.get() is not None:
            return request_config.get()
        return getattr(current_workspace.get() or self.local, name)

    def __setattr__(self, name, value):
        setattr(current_workspace.get() or self.local, name, value)


class WorkspaceMapping(MutableMapping):
    def __init__(self, field: str) -> None:
        self.field = field
        self.local: dict = {}

    def _mapping(self):
        ws = current_workspace.get()
        return getattr(ws, self.field) if ws is not None else self.local

    def __getitem__(self, key): return self._mapping()[key]
    def __setitem__(self, key, value): self._mapping()[key] = value
    def __delitem__(self, key): del self._mapping()[key]
    def __iter__(self): return iter(self._mapping())
    def __len__(self): return len(self._mapping())


def get_workspace(request: Request) -> Workspace:
    ws = current_workspace.get()
    if ws is None:
        raise HTTPException(409, {"code": "workspace_not_loaded"})
    return ws


def _portable_asset(value: object, processed_dir: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("/static/illustrations/"):
        value = "assets/illustrations/" + value.rsplit("/", 1)[-1]
    if value.startswith("assets/"):
        relative = PurePosixPath(value).relative_to("assets")
        if ".." not in relative.parts and "\\" not in value and ":" not in value and relative.parts and relative.parts[0] in ASSET_DIRECTORIES and relative.suffix.lower() in IMAGE_SUFFIXES:
            return "assets/" + relative.as_posix()
        return None
    try:
        path = Path(value).resolve()
        relative = path.relative_to(processed_dir.resolve())
        if relative.parts and relative.parts[0] in ASSET_DIRECTORIES and path.suffix.lower() in IMAGE_SUFFIXES:
            return "assets/" + relative.as_posix()
    except (OSError, ValueError):
        pass
    return None


def _portable_metadata(metadata: dict, processed_dir: Path) -> dict:
    result = deepcopy(metadata)
    for field in ("source_file", "keyframe_path", "slide_image_path"):
        value = _portable_asset(result.pop(field, None), processed_dir)
        if value:
            result[field] = value
    result["image_paths"] = [asset for value in result.pop("image_paths", []) if (asset := _portable_asset(value, processed_dir))] if isinstance(metadata.get("image_paths", []), list) else []
    # Absolute raw/derived directories are never portable or trusted on import.
    result.pop("derived_asset_dirs", None)
    source = result.get("source_path")
    if source is not None and (not isinstance(source, str) or "\\" in source or ":" in source or ".." in PurePosixPath(source).parts or PurePosixPath(source).is_absolute()):
        result.pop("source_path", None)
    return result


def export_bundle(ws: Workspace) -> bytes:
    """Export only data formats, never pickle or SQLite database files."""
    buffer = io.BytesIO()
    chunks = ws.store.get_all_chunks()
    validate_workspace_data(chunks, ws.graph_builder.graph)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps({"format": "lumin", "version": 1, "workspace_version": ws.version, "embedding_model": ws.embedder.model_name, "chunk_ids": [c.id for c in chunks]}))
        archive.writestr("chunks.jsonl", "\n".join(c.model_copy(update={"metadata": _portable_metadata(c.metadata, ws.processed_dir)}).model_dump_json() for c in chunks))
        graph = {"association_version": 1, "node_schema_version": NODE_SCHEMA_VERSION, "nodes": [{"id": key, **attrs, "illustration_path": _portable_asset(attrs.get("illustration_path"), ws.processed_dir)} for key, attrs in ws.graph_builder.graph.nodes(data=True)], "edges": [{"source": u, "target": v, "key": k, **attrs} for u, v, k, attrs in ws.graph_builder.graph.edges(keys=True, data=True)]}
        archive.writestr("graph.json", json.dumps(graph))
        vectors = getattr(ws.vector_index, "embeddings", None)
        ids = list(getattr(ws.vector_index, "chunk_ids", []))
        if vectors is not None and len(ids):
            out = io.BytesIO()
            np.save(out, vectors, allow_pickle=False)
            archive.writestr("embeddings.npy", out.getvalue())
            archive.writestr("embedding_ids.json", json.dumps(ids))
        for path in ws.processed_dir.rglob("*"):
            if path.is_file() and not path.is_symlink() and ws.processed_dir in path.resolve().parents:
                relative = path.relative_to(ws.processed_dir)
                if relative.parts[0] in ASSET_DIRECTORIES and path.suffix.lower() in IMAGE_SUFFIXES:
                    archive.write(path, "assets/" + relative.as_posix())
    return buffer.getvalue()


def read_bundle(data: bytes) -> tuple[dict, list[Chunk], nx.MultiDiGraph, dict[str, bytes]]:
    """Validate every member before extraction, including zip bombs and paths."""
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Bundle exceeds the size limit.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > MAX_BUNDLE_MEMBERS or sum(i.file_size for i in members) > MAX_BUNDLE_BYTES:
            raise ValueError("Expanded bundle exceeds the size limit.")
        names = set()
        for item in members:
            name = item.filename
            raw_name = item.orig_filename
            path = PurePosixPath(name)
            if name in names or "\\" in raw_name or "\x00" in raw_name or path.is_absolute() or ".." in path.parts or ":" in name or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Bundle contains an unsafe or duplicate path.")
            names.add(name)
            if item.file_size > MAX_MEMBER_BYTES.get(name, 50 * 1024 * 1024):
                raise ValueError("Bundle member exceeds the size limit.")
        if not {"manifest.json", "chunks.jsonl", "graph.json"} <= names:
            raise ValueError("Bundle is missing its manifest, chunks or graph.")
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict) or manifest.get("format", "lumin") != "lumin" or manifest.get("version", 1) != 1:
            raise ValueError("Unsupported bundle version.")
        version = manifest.get("workspace_version", 0)
        if not isinstance(version, int) or not 0 <= version <= 2**53 - 1:
            raise ValueError("Invalid workspace version.")
        lines = archive.read("chunks.jsonl").decode("utf-8").splitlines()
        if len(lines) > MAX_CHUNKS:
            raise ValueError("Workspace chunk limit exceeded.")
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        if any(len(chunk.text) > MAX_CHUNK_TEXT_CHARACTERS or len(chunk.id) > 1024 or len(chunk.source_id) > 1024 for chunk in chunks):
            raise ValueError("Chunk text or identifier exceeds the size limit.")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs.")
        payload = json.loads(archive.read("graph.json"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid graph payload.")
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", payload.get("links", []))
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise ValueError("Invalid graph payload.")
        if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
            raise ValueError("Workspace graph limit exceeded.")
        legacy = legacy_node_schema(payload.get("node_schema_version"))
        graph = nx.MultiDiGraph()
        graph.graph["node_schema_version"] = NODE_SCHEMA_VERSION
        for raw in nodes:
            attrs = dict(raw)
            key = attrs.pop("id")
            if not isinstance(key, str) or not key or len(key) > 500 or key in graph or attrs.get("node_type") not in {"topic", "subtopic", "content", "note", "chunk_ref"}:
                raise ValueError("Invalid graph node.")
            if legacy:
                attrs = normalize_legacy_node_attrs(attrs)
            validated = validate_node_attrs(attrs)
            attrs.update(validated.model_dump(exclude_none=True, exclude_unset=True))
            graph.add_node(key, **attrs)
        for raw in edges:
            attrs = dict(raw)
            source, target = attrs.pop("source"), attrs.pop("target")
            key = attrs.pop("key", None)
            attrs.pop("id", None)
            if (not isinstance(source, str) or not isinstance(target, str)
                    or not isinstance(key, (str, int, type(None)))
                    or source not in graph or target not in graph
                    or attrs.get("relation") not in {"HAS_SUBTOPIC", "HAS_CONTENT", "RELATED_TO", "EVIDENCE_OF"}):
                raise ValueError("Invalid graph edge.")
            if legacy:
                attrs = normalize_legacy_edge_attrs(attrs)
            else:
                EdgeInput.model_validate(attrs)
            graph.add_edge(source, target, key=key, **attrs)
        normalize_associations(graph)
        files = {name: archive.read(name) for name in names if name.startswith("assets/") and not name.endswith("/")}
        for name in ("embeddings.npy", "embedding_ids.json"):
            if name in names:
                files[name] = archive.read(name)
        return manifest, chunks, graph, files


def import_bundle(ws: Workspace, data: bytes) -> None:
    manifest, chunks, graph, files = read_bundle(data)
    validate_workspace_data(chunks, graph)
    candidate = VectorIndex(ws.embedder)
    # Reuse portable vectors only through the index's validating load API.
    if chunks and manifest.get("embedding_model") == ws.embedder.model_name and "embeddings.npy" in files:
        candidate_path = ws.staging_dir / f"import-{uuid.uuid4().hex}.npy"
        try:
            candidate_path.write_bytes(files["embeddings.npy"])
            ids = json.loads(files.get("embedding_ids.json", json.dumps(manifest.get("chunk_ids", [c.id for c in chunks])).encode()))
            if set(ids) != {c.id for c in chunks}:
                raise ValueError("Embedding IDs do not match chunks.")
            candidate_path.with_suffix(".json").write_text(json.dumps({"version": 1, "model": ws.embedder.model_name, "chunk_ids": ids}), encoding="utf-8")
            candidate.load(candidate_path)
        finally:
            candidate_path.unlink(missing_ok=True)
            candidate_path.with_suffix(".json").unlink(missing_ok=True)
    elif chunks:
        candidate.build(chunks)
    # Metadata is data, never a capability to read a host file. Accept only
    # bundle-owned images, then remap them into this workspace's temp folder.
    for chunk in chunks:
        metadata = _portable_metadata(chunk.metadata, ws.processed_dir)
        for field in ("source_file", "keyframe_path", "slide_image_path"):
            asset = metadata.pop(field, None)
            if asset in files:
                metadata[field] = str(ws.processed_dir.joinpath(*PurePosixPath(asset).parts[1:]))
        metadata["image_paths"] = [str(ws.processed_dir.joinpath(*PurePosixPath(asset).parts[1:])) for asset in metadata.get("image_paths", []) if asset in files]
        chunk.metadata = metadata
    for _, attrs in graph.nodes(data=True):
        asset = _portable_asset(attrs.get("illustration_path"), ws.processed_dir)
        attrs["illustration_path"] = asset if asset in files else None
    # Finish all fallible filesystem writes before publishing SQL/graph state.
    for name, content in files.items():
        if name.startswith("assets/"):
            relative = PurePosixPath(name).relative_to("assets")
            if relative.parts and relative.parts[0] in ASSET_DIRECTORIES and relative.suffix.lower() in IMAGE_SUFFIXES:
                target = ws.processed_dir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    old_sources = {c.source_id for c in ws.store.get_all_chunks()}
    ws.store.replace_sources(chunks, replaced_source_ids=old_sources)
    ws.vector_index = candidate
    ws.index_loaded = bool(chunks)
    ws.graph_builder._graph = graph
    ws.graph_loaded = bool(graph.number_of_nodes())
    ws.epoch += 1
    ws.version = max(ws.version + 1, int(manifest.get("workspace_version", 0)))
    ws.check_disk_budget()

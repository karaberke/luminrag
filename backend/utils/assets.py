"""Plan retirement of provably owned source assets without deleting files.

Call the planner under MUTATION_LOCK after the final candidate graph is known,
then publish its paths as journalled deletion entries alongside the new state.
Historical assets without a unique job directory or explicit legacy ownership
are intentionally retained rather than guessed from a filename or swept by age.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import networkx as nx

from backend.schemas import Chunk
from backend.utils.uploads import AUDIO_EXT, IMAGE_EXT, PDF_EXT, VIDEO_EXT, is_safe_source_id, sanitize_filename, source_identity

_DERIVED_NAMES = {
    "slides": re.compile(r"slide_\d{4,}\.jpg\Z"),
    "pdf_images": re.compile(r"p\d{4,}_img\d+\.(?:png|jpe?g|jpx|jp2|tiff?|bmp|gif|webp)\Z"),
    "keyframes": re.compile(r"frame_\d{7,}\.jpg\Z"),
}
_ILLUSTRATION_URL = re.compile(r"/static/illustrations/([0-9a-f]{32}\.(?:png|jpe?g|gif|webp))\Z")
_SINGLE_ASSET_FIELDS = ("slide_image_path", "keyframe_path", "source_file")


def attach_source_illustrations(graph: nx.Graph, chunks: list[Chunk], processed_root: Path) -> bool:
    """Attach workspace-owned extracted images without a background worker."""
    by_id = {chunk.id: chunk for chunk in chunks}
    changed = False
    for _, attrs in graph.nodes(data=True):
        illustration = attrs.get("illustration") or {}
        if attrs.get("illustration_path") or illustration.get("kind") == "equation":
            continue
        if illustration.get("kind") not in {"diagram", "image"} and attrs.get("content_type") != "figure":
            continue
        candidates = []
        for chunk_id in attrs.get("source_ids", []):
            chunk = by_id.get(chunk_id)
            if chunk is not None:
                metadata = chunk.metadata
                candidates.extend([metadata.get("slide_image_path"), metadata.get("keyframe_path"), *metadata.get("image_paths", [])])
        for value in candidates:
            if not isinstance(value, str):
                continue
            path = _checked_path(Path(value), processed_root)
            if path is not None and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                attrs["illustration_path"] = "assets/" + path.relative_to(processed_root.resolve()).as_posix()
                changed = True
                break
    return changed


@dataclass(frozen=True)
class LegacyReplacement:
    """An explicitly validated flat source retired by the caller's migration."""

    source_id: str
    modality: str
    path: Path


def _checked_path(path: Path, root: Path, *, directory: bool = False) -> Path | None:
    """Resolve an existing regular asset without following links or junctions."""
    try:
        path = path.absolute()
        root = root.absolute()
        if ".." in path.parts or ".." in root.parts:
            return None
        for ancestor in (path, *path.parents):
            if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
                return None
        resolved, resolved_root = path.resolve(strict=True), root.resolve(strict=True)
        if resolved_root not in resolved.parents:
            return None
        if directory:
            return resolved if resolved.is_dir() else None
        return resolved if resolved.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _unique_job_directory(name: str, source_id: str) -> bool:
    if not is_safe_source_id(source_id) or not name.startswith(source_id + "_"):
        return False
    job_id = name[len(source_id) + 1:]
    try:
        parsed = uuid.UUID(job_id)
        return parsed.version == 4 and str(parsed) == job_id
    except (ValueError, AttributeError):
        return False


def _recorded_directories(chunk: Chunk, processed_root: Path, checked=_checked_path) -> set[Path]:
    recorded = chunk.metadata.get("derived_asset_dirs")
    if not isinstance(recorded, list):
        return set()
    directories = set()
    for value in recorded:
        if not isinstance(value, str):
            continue
        relative = Path(value)
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] not in _DERIVED_NAMES:
            continue
        directory = checked(processed_root / relative, processed_root, directory=True)
        if directory is not None:
            directories.add(directory)
    return directories


def _metadata_files(chunk: Chunk, raw_root: Path, processed_root: Path, checked=_checked_path) -> set[Path]:
    values = [chunk.metadata.get(field) for field in _SINGLE_ASSET_FIELDS]
    images = chunk.metadata.get("image_paths")
    if isinstance(images, list):
        values.extend(images)
    files = set()
    for value in values:
        if isinstance(value, str):
            for root in (raw_root, processed_root):
                path = checked(Path(value), root)
                if path is not None:
                    files.add(path)
    source_path = chunk.metadata.get("source_path")
    if isinstance(source_path, str) and not Path(source_path).is_absolute():
        path = checked(raw_root / source_path, raw_root)
        if path is not None:
            files.add(path)
    return files


def _derived_directory(path: Path, processed_root: Path) -> tuple[str, Path] | None:
    try:
        relative = path.relative_to(processed_root)
    except ValueError:
        return None
    if len(relative.parts) != 3 or relative.parts[0] not in _DERIVED_NAMES:
        return None
    return relative.parts[0], path.parent


def _illustration_files(graph: nx.Graph, processed_root: Path, checked=_checked_path) -> set[Path]:
    files = set()
    for _, attributes in graph.nodes(data=True):
        value = attributes.get("illustration_path")
        matched = _ILLUSTRATION_URL.fullmatch(value) if isinstance(value, str) else None
        if matched:
            path = checked(processed_root / "illustrations" / matched[1], processed_root)
            if path is not None:
                files.add(path)
    return files


def record_derived_asset_dirs(chunks: list[Chunk], staged_path: Path, processed_root: Path) -> list[Chunk]:
    """Record existing job-owned directories, including rendered but unreferenced pages."""
    result = []
    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.source_id not in by_source:
            directories = []
            if _unique_job_directory(staged_path.stem, chunk.source_id):
                for kind in _DERIVED_NAMES:
                    directory = _checked_path(processed_root / kind / staged_path.stem, processed_root, directory=True)
                    if directory is not None:
                        directories.append(directory.relative_to(processed_root.resolve()).as_posix())
            by_source[chunk.source_id] = directories
        result.append(chunk.model_copy(update={"metadata": {**chunk.metadata, "derived_asset_dirs": list(by_source[chunk.source_id])}}))
    return result


def plan_retired_assets(
    previous_chunks: Iterable[Chunk],
    candidate_chunks: Iterable[Chunk],
    replaced_source_ids: set[str],
    previous_graph: nx.Graph,
    candidate_graph: nx.Graph,
    *,
    raw_root: Path,
    processed_root: Path,
    legacy_replacements: Iterable[LegacyReplacement] = (),
) -> list[Path]:
    """Return only old, unreferenced regular files eligible for journal deletion.

    Current chunk references protect their entire derived directory, including
    unused frames or blank pages. Only replaced source metadata establishes
    ownership; a graph URL establishes ownership of a generated illustration.
    The helper never scans whole roots, traverses directories recursively, or
    includes recovery companions, staging files, or unknown filenames.
    """
    previous_chunks, candidate_chunks = list(previous_chunks), list(candidate_chunks)
    raw_root, processed_root = Path(raw_root), Path(processed_root)
    try:
        resolved_raw, resolved_processed = raw_root.resolve(), processed_root.resolve()
    except (OSError, RuntimeError):
        return []
    # Scope filesystem memoization to this plan; the next transaction revalidates.
    checked = cache(_checked_path)
    protected_files = _illustration_files(candidate_graph, processed_root, checked)
    protected_directories = set()
    for chunk in candidate_chunks:
        files = _metadata_files(chunk, raw_root, processed_root, checked)
        protected_files.update(files)
        protected_directories.update(_recorded_directories(chunk, processed_root, checked))
        for path in files:
            derived = _derived_directory(path, resolved_processed)
            if derived is not None:
                protected_directories.add(derived[1])

    retired = _illustration_files(previous_graph, processed_root, checked) - protected_files
    old_directories: set[tuple[str, Path]] = set()
    for chunk in previous_chunks:
        if chunk.source_id not in replaced_source_ids:
            continue
        for directory in _recorded_directories(chunk, processed_root, checked):
            if _unique_job_directory(directory.name, chunk.source_id):
                old_directories.add((directory.parent.name, directory))
        for path in _metadata_files(chunk, raw_root, processed_root, checked):
            derived = _derived_directory(path, resolved_processed)
            if derived is not None and _unique_job_directory(derived[1].name, chunk.source_id):
                old_directories.add(derived)
            try:
                relative = path.relative_to(resolved_raw)
            except ValueError:
                continue
            original = chunk.metadata.get("original_filename")
            if isinstance(original, str) and len(relative.parts) == 2 and relative.parts[0] == chunk.source_id:
                try:
                    if source_identity(original) == chunk.source_id and path.name == sanitize_filename(original):
                        retired.add(path)
                except ValueError:
                    continue

    for kind, directory in old_directories:
        if directory in protected_directories:
            continue
        try:
            for entry in directory.iterdir():
                if _DERIVED_NAMES[kind].fullmatch(entry.name):
                    path = checked(entry, processed_root)
                    if path is not None:
                        retired.add(path)
        except OSError:
            continue

    old_modalities: dict[str, set[str]] = {}
    for chunk in previous_chunks:
        old_modalities.setdefault(chunk.source_id, set()).add(chunk.modality)
    allowed_extensions = {"pdf": PDF_EXT, "slide": PDF_EXT, "image": IMAGE_EXT, "audio": AUDIO_EXT, "video": VIDEO_EXT}
    current_source_ids = {chunk.source_id for chunk in candidate_chunks}
    for legacy in legacy_replacements:
        if legacy.source_id in current_source_ids or legacy.source_id not in replaced_source_ids or not is_safe_source_id(legacy.source_id):
            continue
        compatible = {"pdf", "slide"} if legacy.modality in {"pdf", "slide"} else {legacy.modality}
        if not old_modalities.get(legacy.source_id) or not old_modalities[legacy.source_id] <= compatible:
            continue
        path = checked(Path(legacy.path), raw_root)
        if path is not None and path.parent == resolved_raw and path.stem == legacy.source_id and path.suffix.lower() in allowed_extensions.get(legacy.modality, ()):
            retired.add(path)
    return sorted(retired - protected_files, key=str)

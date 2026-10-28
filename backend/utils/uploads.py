"""
Upload safety helpers shared by the multipart and S3 ingest paths.

  - one allowlist of supported extensions + content types (PDF/image/audio/video)
  - filename sanitisation and a deterministic per-filename source identity
  - bounded streaming copies (never read a whole upload into memory)
  - per-job staging directories and atomic commit into raw/<source_id>/<file>
  - safe lookup of persisted raw sources for serving
  - an in-process registry of server-issued S3 upload keys
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from backend.utils.environment import hosted_mode

MAX_UPLOAD_BYTES_ENV = "LUMINRAG_MAX_UPLOAD_BYTES"
DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_BATCH_FILES = 20
COPY_CHUNK_BYTES = 1024 * 1024
_MAX_FILENAME_CHARS = 180

PDF_EXT = frozenset({".pdf"})
IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".webp"})
AUDIO_EXT = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"})
VIDEO_EXT = frozenset({".mp4", ".mkv", ".mov", ".avi"})
MARKDOWN_EXT = frozenset({".md", ".markdown", ".txt"})
OFFICE_EXT = frozenset({".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub"})
SUPPORTED_EXT = PDF_EXT | IMAGE_EXT | AUDIO_EXT | VIDEO_EXT | MARKDOWN_EXT | OFFICE_EXT

# Content type served back for each extension (never sniffed from the file).
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".html": "text/html",
    ".htm": "text/html",
    ".epub": "application/epub+zip",
}

# Generic types browsers send when they do not know the file; the extension decides.
_GENERIC_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})
_ACTIVE_TYPE_MARKERS = ("html", "svg", "xml", "javascript", "ecmascript")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.\- ()\[\]]", re.UNICODE)


class UploadRejected(ValueError):
    """Raised for uploads that fail validation. ``status`` is the HTTP code to use."""

    def __init__(self, detail: str, status: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


def max_upload_bytes() -> int:
    hosted = hosted_mode()
    default = 50 * 1024 * 1024 if hosted else DEFAULT_MAX_UPLOAD_BYTES
    raw = os.getenv(MAX_UPLOAD_BYTES_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    value = value if value > 0 else default
    return min(value, default) if hosted else value


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def sanitize_filename(name: str | None) -> str:
    """
    Reduce a client-supplied name to a safe basename (no directories, no
    URL-special or control characters). Raises UploadRejected if nothing
    usable remains.
    """
    if not name:
        raise UploadRejected("Missing filename.")
    name = unicodedata.normalize("NFC", name)
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(ch for ch in base if unicodedata.category(ch)[0] != "C")
    base = _UNSAFE_FILENAME_CHARS.sub("_", base).strip(" .")
    if not base or base in {".", ".."}:
        raise UploadRejected(f"Invalid filename: {name!r}.")
    if len(base) > _MAX_FILENAME_CHARS:
        stem, ext = os.path.splitext(base)
        base = stem[: _MAX_FILENAME_CHARS - len(ext)] + ext
    return base


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_file(filename: str | None, content_type: str | None) -> str:
    """
    Check extension + declared content type against the allowlist.
    Returns the sanitised filename. Active content (HTML/SVG/XML/JS) is
    rejected regardless of extension.
    """
    safe = sanitize_filename(filename)
    ext = file_extension(safe)
    if hosted_mode() and ext in VIDEO_EXT | AUDIO_EXT:
        raise UploadRejected("Video and audio are available only when self hosted.", status=415)
    if ext not in SUPPORTED_EXT:
        raise UploadRejected(
            f"Unsupported file type '{ext or safe}'. Allowed: {', '.join(sorted(SUPPORTED_EXT))}.",
            status=415,
        )
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    recognised_document = ext in OFFICE_EXT | MARKDOWN_EXT and ctype in {MEDIA_TYPES.get(ext), "text/plain", "text/markdown", "application/zip", "application/x-zip-compressed"}
    if any(marker in ctype for marker in _ACTIVE_TYPE_MARKERS) and not recognised_document:
        raise UploadRejected(f"Content type '{ctype}' is not allowed.", status=415)
    if ctype in _GENERIC_TYPES:
        return safe
    if ext in PDF_EXT:
        ok = ctype in {"application/pdf", "application/x-pdf"}
    elif ext in IMAGE_EXT:
        ok = ctype in {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp"}
    elif ext in AUDIO_EXT:
        ok = ctype.startswith("audio/") or ctype == "application/ogg"
    elif ext in MARKDOWN_EXT | OFFICE_EXT:
        ok = recognised_document
    else:
        ok = ctype.startswith("video/")
    if not ok:
        raise UploadRejected(
            f"Content type '{ctype}' does not match extension '{ext}'.", status=415
        )
    return safe


def validate_batch_size(count: int) -> None:
    if count < 1:
        raise UploadRejected("No files supplied.")
    if count > MAX_BATCH_FILES:
        raise UploadRejected(
            f"Too many files ({count}); at most {MAX_BATCH_FILES} per batch.", status=413
        )


def validate_declared_size(size: int, limit: int | None = None) -> None:
    limit = limit or max_upload_bytes()
    if size <= 0:
        raise UploadRejected("File size must be positive.")
    if size > limit:
        raise UploadRejected(f"File exceeds the {limit}-byte upload limit.", status=413)


# ---------------------------------------------------------------------------
# Source identity
# ---------------------------------------------------------------------------

def source_identity(filename: str) -> str:
    """
    Deterministic source_id for a sanitised filename. Includes the extension
    and a digest of the full filename, so ``lecture.pdf`` and ``lecture.mp4``
    never collide while re-uploading the same filename maps to the same ID.
    """
    safe = sanitize_filename(filename)
    stem, ext = os.path.splitext(safe)
    original = unicodedata.normalize("NFC", filename).replace("\\", "/").rsplit("/", 1)[-1]
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "file"
    ext_tag = ext.lstrip(".").lower() or "bin"
    return f"{slug}_{ext_tag}_{digest}"


_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\- ()\[\]]{0,199}$")


def is_safe_source_id(source_id: str) -> bool:
    return bool(_SOURCE_ID_RE.match(source_id)) and ".." not in source_id


def remap_chunks(chunks: list, source_id: str, original_filename: str, source_path: str) -> list:
    """
    Rewrite chunk IDs/source_id to the stable identity and record provenance.
    ``source_path`` is relative to the raw directory. Returns new Chunk objects.
    """
    remapped = []
    seen: set[str] = set()
    for idx, chunk in enumerate(chunks):
        old_source = chunk.source_id
        if old_source and chunk.id.startswith(old_source):
            new_id = source_id + chunk.id[len(old_source):]
        else:
            new_id = f"{source_id}_{chunk.id}"
        if new_id in seen:
            new_id = f"{source_id}_{idx}"
        seen.add(new_id)
        meta = dict(chunk.metadata or {})
        meta["original_filename"] = original_filename
        meta["source_path"] = source_path
        remapped.append(chunk.model_copy(update={"id": new_id, "source_id": source_id, "metadata": meta}))
    return remapped


# ---------------------------------------------------------------------------
# Bounded copies + staging
# ---------------------------------------------------------------------------

def copy_bounded(src: BinaryIO, dest: Path, limit: int, expected_size: int | None = None) -> int:
    """
    Stream src into dest in fixed-size chunks, aborting once more than
    ``limit`` bytes arrive. If ``expected_size`` is given the total must match
    exactly. A partially written dest is removed on failure.
    """
    cap = min(limit, expected_size) if expected_size is not None else limit
    total = 0
    out = dest.open("xb")  # never clobber an existing file
    try:
        with out:
            while True:
                block = src.read(COPY_CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                if total > cap:
                    raise UploadRejected(
                        f"File exceeds the {cap}-byte limit.", status=413
                    )
                out.write(block)
        if expected_size is not None and total != expected_size:
            raise UploadRejected(
                f"Size mismatch: expected {expected_size} bytes, received {total}."
            )
        if total == 0:
            raise UploadRejected("Empty file.")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return total


@dataclass(frozen=True)
class StagedFile:
    path: Path               # staging/<job>/<n>/<filename>
    filename: str            # sanitised filename
    original_filename: str   # as supplied by the client
    source_id: str
    is_slides: bool = False


def new_job_staging_dir(staging_root: Path, job_id: str) -> Path:
    path = staging_root / job_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def staged_path(job_dir: Path, index: int, filename: str) -> Path:
    slot = job_dir / f"{index:02d}"
    slot.mkdir(parents=True, exist_ok=False)
    return slot / filename


DERIVED_ASSET_DIRS = ("slides", "pdf_images", "keyframes")


def cleanup_attempt_assets(staged: StagedFile, staging_root: Path, processed_root: Path) -> None:
    """Remove only this staging attempt's derived directories, never old sources."""
    try:
        relative = staged.path.absolute().relative_to(staging_root.absolute())
        if (
            len(relative.parts) != 3
            or any(part in {".", ".."} for part in relative.parts)
            or not relative.parts[1].isdigit()
            or not is_safe_source_id(staged.source_id)
            or staged.path.stem != f"{staged.source_id}_{relative.parts[0]}"
        ):
            return
        if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
               for path in (staged.path, *staged.path.parents)):
            return
        for derived in DERIVED_ASSET_DIRS:
            root = processed_root / derived
            target = root / staged.path.stem
            # Check the unresolved path as well as the final root boundary;
            # a linked parent must not redirect cleanup to another source.
            if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                   for path in (target, *target.parents)):
                continue
            remove_staging(target, root)
    except (OSError, RuntimeError, ValueError):
        return


def remove_staging(job_dir: Path, staging_root: Path) -> None:
    """Delete a job staging dir, refusing anything outside staging_root."""
    try:
        root = staging_root.resolve()
        target = job_dir.resolve()
    except OSError:
        return
    if target == root or root not in target.parents or job_dir.is_symlink():
        return
    shutil.rmtree(target, ignore_errors=True)


def commit_raw_source(staged: Path, raw_dir: Path, source_id: str, filename: str) -> Path:
    """
    Atomically publish a parsed staged file as raw/<source_id>/<filename>.
    Returns the persisted absolute path.
    """
    if not is_safe_source_id(source_id) or sanitize_filename(filename) != filename:
        raise UploadRejected("Unsafe source identity.")
    raw_root = raw_dir.resolve()
    dest_dir = raw_root / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if raw_root not in dest.resolve().parents:
        raise UploadRejected("Unsafe source path.")
    tmp = dest_dir / f".{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(staged, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def resolve_raw_source(raw_dir: Path, relative_path: str | None) -> Path | None:
    """
    Resolve a persisted source path (relative to raw_dir) only if it stays
    inside raw_dir, is a regular non-symlink file, and has an allowed extension.
    """
    if not relative_path:
        return None
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    root = raw_dir.resolve()
    candidate = root / rel
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if root not in resolved.parents or not resolved.is_file():
        return None
    # HTML is accepted for text conversion, but never served as an executable
    # page on the API origin. The browser keeps its own original copy.
    if resolved.suffix.lower() not in SUPPORTED_EXT - {".html", ".htm"}:
        return None
    return resolved


def find_legacy_raw_source(raw_dir: Path, source_id: str) -> Path | None:
    """
    Pre-identity uploads were stored flat as raw/<filename> with source_id ==
    stem. Match a top-level allowlisted file by exact name or stem.
    """
    if not is_safe_source_id(source_id) or not raw_dir.is_dir():
        return None
    direct = resolve_raw_source(raw_dir, source_id)
    if direct is not None and direct.parent == raw_dir.resolve():
        return direct
    for entry in sorted(raw_dir.iterdir()):
        if entry.stem == source_id and entry.suffix.lower() in SUPPORTED_EXT:
            found = resolve_raw_source(raw_dir, entry.name)
            if found is not None:
                return found
    return None


def media_type_for(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class UploadBodyLimitMiddleware:
    """
    Reject multipart upload requests whose Content-Length exceeds the batch
    budget *before* the framework spools the body to disk. Requests without
    a Content-Length (chunked) are refused.
    """

    def __init__(self, app, paths: Iterable[str] = ("/api/ingest",)) -> None:
        self.app = app
        self.paths = frozenset(paths)

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in self.paths
        ):
            from starlette.responses import JSONResponse

            cap = MAX_BATCH_FILES * max_upload_bytes() + _MULTIPART_OVERHEAD_BYTES
            if hosted_mode():
                cap = min(cap, 100 * 1024 * 1024 + _MULTIPART_OVERHEAD_BYTES)
            raw = dict(scope.get("headers") or []).get(b"content-length")
            status, detail = None, None
            if raw is None:
                status, detail = 411, "Content-Length required for uploads."
            elif not raw.isdigit():
                status, detail = 400, "Invalid Content-Length."
            elif int(raw) > cap:
                status, detail = 413, f"Upload batch exceeds {cap} bytes."
            if status is not None:
                await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def download_issued_object(s3, bucket: str, record: "IssuedUpload", dest: Path, limit: int) -> int:
    """
    Fetch a server-issued S3 object into dest. The object's size is checked
    with head_object before any download, the body is pinned to that ETag,
    and the stream is copied with a hard byte bound.
    """
    head = s3.head_object(Bucket=bucket, Key=record.key)
    length = int(head.get("ContentLength", -1))
    if length != record.size:
        raise UploadRejected(
            f"Uploaded object size {length} does not match declared size {record.size}."
        )
    validate_declared_size(length, limit)
    validate_file(record.filename, head.get("ContentType"))
    params = {"Bucket": bucket, "Key": record.key}
    if head.get("ETag"):
        params["IfMatch"] = head["ETag"]
    body = s3.get_object(**params)["Body"]
    try:
        return copy_bounded(body, dest, limit, expected_size=record.size)
    finally:
        close = getattr(body, "close", None)
        if close:
            close()


# ---------------------------------------------------------------------------
# S3 configuration + issued-key registry
# ---------------------------------------------------------------------------

S3_UPLOAD_BUCKET_ENV = "S3_UPLOAD_BUCKET"
S3_FRONTEND_BUCKET_ENV = "S3_FRONTEND_BUCKET"
LEGACY_FRONTEND_BUCKET = "luminrag-frontend-bucket"


def s3_upload_bucket() -> str:
    """
    Dedicated upload bucket. Never falls back to the frontend bucket and
    refuses to run if configured to it.
    """
    bucket = os.getenv(S3_UPLOAD_BUCKET_ENV, "").strip()
    if not bucket:
        raise UploadRejected(
            f"S3 uploads are disabled: {S3_UPLOAD_BUCKET_ENV} is not configured.", status=503
        )
    frontend = os.getenv(S3_FRONTEND_BUCKET_ENV, LEGACY_FRONTEND_BUCKET).strip()
    if bucket in {frontend, LEGACY_FRONTEND_BUCKET}:
        raise UploadRejected(
            f"{S3_UPLOAD_BUCKET_ENV} must be a dedicated bucket, not the frontend bucket.",
            status=503,
        )
    return bucket


@dataclass(frozen=True)
class IssuedUpload:
    key: str
    filename: str
    size: int
    expires_at: float
    original_filename: str | None = None


class IssuedUploadRegistry:
    """Thread-safe record of presigned keys this process issued."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 2000) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._items: dict[str, IssuedUpload] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        for key in [k for k, v in self._items.items() if v.expires_at <= now]:
            del self._items[key]

    def issue(
        self, filename: str, size: int, now: float | None = None, *, original_filename: str | None = None
    ) -> IssuedUpload:
        now = time.time() if now is None else now
        record = IssuedUpload(
            key=f"uploads/{uuid.uuid4()}/{filename}",
            filename=filename,
            size=size,
            expires_at=now + self._ttl,
            original_filename=original_filename,
        )
        with self._lock:
            self._prune(now)
            if len(self._items) >= self._max:
                raise UploadRejected("Too many pending uploads; try again later.", status=429)
            self._items[record.key] = record
        return record

    def claim(self, keys: Iterable[str], now: float | None = None) -> list[IssuedUpload]:
        """
        Atomically validate and remove a batch of keys. Unknown, expired, or
        duplicated keys reject the whole batch and nothing is consumed.
        """
        now = time.time() if now is None else now
        keys = list(keys)
        validate_batch_size(len(keys))
        if len(set(keys)) != len(keys):
            raise UploadRejected("Duplicate upload keys in request.")
        with self._lock:
            self._prune(now)
            missing = [k for k in keys if k not in self._items]
            if missing:
                raise UploadRejected("Unknown or expired upload key.", status=404)
            return [self._items.pop(k) for k in keys]

    def release(self, records: Iterable[IssuedUpload], now: float | None = None) -> None:
        """Return claimed keys after a failed download so the client can retry."""
        now = time.time() if now is None else now
        with self._lock:
            for record in records:
                if record.expires_at > now:
                    self._items.setdefault(record.key, record)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

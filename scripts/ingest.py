"""
Batch ingestion CLI for LuminRAG.

Uploads course files to a *running* LuminRAG backend through its ingest API,
so the document store, vector index and concept graph are extended in place.
This script never opens, deletes or rewrites any local database, index or
graph file.

Usage:
    python scripts/ingest.py                          # uses backend/data/raw/ by default
    python scripts/ingest.py path/to/other/folder
    python scripts/ingest.py --dry-run                # validate + preview, no network
    python scripts/ingest.py --slides week1.pdf week2.pdf
    python scripts/ingest.py --api-url http://127.0.0.1:8000

An optional access key is read only from the LUMINRAG_API_TOKEN environment
variable and sent as ``Authorization: Bearer <key>``. ``.env`` is not loaded.

Flow:
  1. Walk the directory and validate every supported file up front (type,
     size limit, names that would map to the same source and replace one
     another). Nothing is uploaded if any file fails validation.
  2. Submit files in batches of at most MAX_BATCH_FILES to POST /api/ingest.
  3. Poll GET /api/ingest/jobs/{job_id} until the job finishes before
     submitting the next batch.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

# Ensure repo root is on sys.path so `backend.*` imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from backend.utils.uploads import (
    MAX_BATCH_FILES,
    MEDIA_TYPES,
    PDF_EXT,
    AUDIO_EXT,
    IMAGE_EXT,
    VIDEO_EXT,
    MARKDOWN_EXT,
    OFFICE_EXT,
    SUPPORTED_EXT,
    UploadRejected,
    file_extension,
    max_upload_bytes,
    source_identity,
    validate_declared_size,
    validate_file,
)

TOKEN_ENV = "LUMINRAG_API_TOKEN"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_DIR = Path(__file__).parent.parent / "backend" / "data" / "raw"

INGEST_PATH = "/api/ingest"
JOB_PATH = "/api/ingest/jobs/{job_id}"

DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_POLL_TIMEOUT = 3600.0
MIN_POLL_INTERVAL, MAX_POLL_INTERVAL = 0.2, 60.0
MAX_POLL_TIMEOUT = 24 * 3600.0
# Polling is a GET and safe to repeat; uploads are never retried.
MAX_POLL_NETWORK_ERRORS = 3

UPLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)
POLL_TIMEOUT = httpx.Timeout(10.0)

_TERMINAL_FAILURES = {"failed", "cancelled"}


class CLIError(Exception):
    """User-facing failure; the message is printed as-is."""


# ---------------------------------------------------------------------------
# Planning + validation (local only, no network)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedFile:
    path: Path
    relative: str     # path relative to the scanned directory, for messages
    filename: str     # original basename, used by the server's source identity
    source_id: str    # server-side identity; equal IDs replace each other
    size: int
    is_slides: bool

    @property
    def kind(self) -> str:
        ext = file_extension(self.filename)
        if ext in VIDEO_EXT:
            return "video"
        if ext in AUDIO_EXT:
            return "audio"
        if ext in IMAGE_EXT:
            return "image"
        if ext in MARKDOWN_EXT:
            return "text"
        if ext in OFFICE_EXT:
            return "document"
        return "slides (pdf)" if self.is_slides else "pdf"


@dataclass
class Plan:
    files: list[PlannedFile]
    problems: list[str]
    skipped: list[str]
    warnings: list[str]


def _walk_files(directory: Path) -> Iterator[tuple[Path, bool]]:
    """Yield (path, is_symlink) for every non-directory entry, never following links."""
    for root, dirs, names in os.walk(directory, followlinks=False):
        # Windows junctions are not symlinks, and os.walk can follow them even
        # with followlinks=False. Do not scan aliases or paths outside the root.
        dirs[:] = sorted(
            name for name in dirs
            if not (Path(root) / name).is_symlink()
            and not getattr(Path(root) / name, "is_junction", lambda: False)()
            and (Path(root) / name).resolve() == (Path(root) / name).absolute()
            and directory in (Path(root) / name).resolve().parents
        )
        for name in sorted(names):
            path = Path(root) / name
            yield path, path.is_symlink()


def plan_directory(directory: Path, slide_names: set[str], limit: int | None = None) -> Plan:
    directory = directory.resolve()
    limit = limit or max_upload_bytes()
    plan = Plan(files=[], problems=[], skipped=[], warnings=[])
    by_source: dict[str, list[PlannedFile]] = {}
    matched_slides: set[str] = set()

    for path, is_link in _walk_files(directory):
        rel = path.relative_to(directory).as_posix()
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXT:
            plan.skipped.append(f"{rel} (unsupported type)")
            continue
        if is_link:
            plan.skipped.append(f"{rel} (symlink)")
            continue
        if not path.is_file():
            continue
        try:
            safe = validate_file(path.name, MEDIA_TYPES[ext])
            size = path.stat().st_size
            validate_declared_size(size, limit)
        except UploadRejected as exc:
            plan.problems.append(f"{rel}: {exc.detail}")
            continue
        except OSError as exc:
            plan.problems.append(f"{rel}: cannot read file ({exc.strerror or exc}).")
            continue

        is_slides = path.name in slide_names
        if is_slides:
            matched_slides.update({path.name, safe} & slide_names)
            if ext not in PDF_EXT:
                plan.warnings.append(f"{rel}: --slides only applies to PDFs; ignored.")
                is_slides = False

        item = PlannedFile(path, rel, path.name, source_identity(path.name), size, is_slides)
        by_source.setdefault(item.source_id, []).append(item)
        plan.files.append(item)

    for group in by_source.values():
        if len(group) > 1:
            names = ", ".join(f.relative for f in group)
            plan.problems.append(
                f"Files would replace one another on the server (same name "
                f"'{group[0].filename}'): {names}. Rename all but one so each "
                f"file name is unique."
            )

    for name in sorted(slide_names - matched_slides):
        plan.warnings.append(f"--slides name '{name}' does not match any file.")
    return plan


def make_batches(files: list[PlannedFile], size: int = MAX_BATCH_FILES) -> list[list[PlannedFile]]:
    return [files[i:i + size] for i in range(0, len(files), size)]


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def normalize_api_url(raw: str) -> str:
    parts = urlsplit(raw.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CLIError(f"--api-url must be an http(s) URL with a host, got {raw!r}.")
    if parts.username or parts.password:
        raise CLIError(f"--api-url must not contain credentials; set {TOKEN_ENV} instead.")
    if parts.query or parts.fragment:
        raise CLIError("--api-url must not contain a query string or fragment.")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_client(api_url: str, token: str | None, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    headers = {"Accept": "application/json", "User-Agent": "luminrag-ingest-cli"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Redirects are never followed: a redirected request would carry the
    # bearer token (and the upload) to wherever the Location points.
    return httpx.Client(
        base_url=api_url,
        headers=headers,
        follow_redirects=False,
        timeout=POLL_TIMEOUT,
        transport=transport,
    )


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])[:500]
    return (resp.text or resp.reason_phrase or "").strip()[:500]


def _raise_for_response(resp: httpx.Response, action: str, has_token: bool) -> None:
    if resp.is_redirect:
        where = resp.headers.get("location", "?")
        raise CLIError(
            f"{action}: server answered {resp.status_code} redirect to {where}. "
            f"Redirects are not followed; pass the final backend URL with --api-url."
        )
    if resp.status_code in (401, 403):
        hint = (
            f"the key in {TOKEN_ENV} was rejected"
            if has_token
            else f"set {TOKEN_ENV} to the backend access key"
        )
        raise CLIError(f"{action}: HTTP {resp.status_code} ({_detail(resp)}); {hint}.")
    if resp.status_code >= 400:
        raise CLIError(f"{action}: HTTP {resp.status_code}: {_detail(resp)}")


def _open_for_upload(path: Path):
    return path.open("rb")


def submit_batch(client: httpx.Client, batch: list[PlannedFile], has_token: bool) -> str:
    """POST one batch exactly once and return the job id."""
    action = "Upload failed"
    with ExitStack() as stack:
        parts = [
            (
                "files",
                (f.filename, stack.enter_context(_open_for_upload(f.path)), MEDIA_TYPES[file_extension(f.filename)]),
            )
            for f in batch
        ]
        slides = json.dumps([f.filename for f in batch if f.is_slides])
        try:
            resp = client.post(INGEST_PATH, files=parts, data={"slides": slides}, timeout=UPLOAD_TIMEOUT)
        except httpx.ConnectError as exc:
            raise CLIError(
                f"{action}: cannot connect to {client.base_url} ({exc}). "
                f"Is the backend running? Nothing was submitted for this batch."
            ) from exc
        except httpx.TransportError as exc:
            raise CLIError(
                f"{action}: network error during upload ({type(exc).__name__}: {exc}). "
                f"The server may still have accepted this batch; check it before re-running "
                f"(re-uploading a file replaces its earlier copy rather than duplicating it)."
            ) from exc
    _raise_for_response(resp, action, has_token)
    try:
        job_id = resp.json().get("job_id")
    except (ValueError, AttributeError):
        job_id = None
    if not isinstance(job_id, str) or not job_id or not all(c.isalnum() or c == "-" for c in job_id):
        raise CLIError(f"{action}: unexpected response from server: {resp.text[:200]!r}")
    return job_id


def poll_job(
    client: httpx.Client,
    job_id: str,
    *,
    interval: float,
    timeout: float,
    has_token: bool,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Poll until the job is done; return its result dict. Raises CLIError otherwise."""
    deadline = clock() + timeout
    url = JOB_PATH.format(job_id=job_id)
    errors = 0
    last_stage = None
    while True:
        try:
            resp = client.get(url)
        except httpx.TransportError as exc:
            errors += 1
            if errors >= MAX_POLL_NETWORK_ERRORS:
                raise CLIError(
                    f"Lost contact with the backend while polling job {job_id} "
                    f"({type(exc).__name__}: {exc}). The job may still be running."
                ) from exc
        else:
            errors = 0
            if resp.status_code == 404:
                raise CLIError(
                    f"Job {job_id} is unknown to the server (was the backend restarted?). "
                    f"Its outcome cannot be determined."
                )
            _raise_for_response(resp, f"Polling job {job_id} failed", has_token)
            try:
                body = resp.json()
            except ValueError:
                body = None
            if not isinstance(body, dict):
                raise CLIError(f"Polling job {job_id}: unexpected response {resp.text[:200]!r}")
            status = body.get("status")
            if status == "done":
                result = body.get("result")
                if not isinstance(result, dict):
                    raise CLIError(f"Job {job_id} completed without a valid result.")
                return result
            if status in _TERMINAL_FAILURES:
                reason = body.get("error") or body.get("progress_stage") or "no details"
                raise CLIError(f"Job {job_id} {status}: {reason}")
            if status not in {"queued", "running"}:
                raise CLIError(f"Job {job_id} returned an unknown status: {status!r}.")
            stage = body.get("progress_stage")
            if on_progress and stage and stage != last_stage:
                on_progress(stage)
            last_stage = stage

        remaining = deadline - clock()
        if remaining <= 0:
            raise CLIError(
                f"Gave up waiting for job {job_id} after {timeout:g}s; it is still running "
                f"on the server. Check GET {JOB_PATH.format(job_id=job_id)} later."
            )
        sleep(min(interval, remaining))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _bounded_float(lo: float, hi: float) -> Callable[[str], float]:
    def parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {raw!r}")
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(f"must be between {lo:g} and {hi:g}")
        return value
    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload course files to a running LuminRAG backend (existing data is kept)."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=DEFAULT_DIR,
        help=f"Directory to scan for course files (default: {DEFAULT_DIR})",
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
        help="Validate files and print the upload plan without contacting the backend.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Base URL of the running backend (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=_bounded_float(MIN_POLL_INTERVAL, MAX_POLL_INTERVAL),
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between job status checks (default: {DEFAULT_POLL_INTERVAL:g}).",
    )
    parser.add_argument(
        "--timeout",
        type=_bounded_float(1.0, MAX_POLL_TIMEOUT),
        default=DEFAULT_POLL_TIMEOUT,
        help=f"Max seconds to wait for each batch's job (default: {DEFAULT_POLL_TIMEOUT:g}).",
    )
    return parser


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(
    argv: list[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    args = _parser().parse_args(argv)

    directory: Path = args.directory
    if not directory.is_dir():
        _err(f"Error: '{directory}' is not a directory.")
        return 2

    try:
        api_url = normalize_api_url(args.api_url)
    except CLIError as exc:
        _err(f"Error: {exc}")
        return 2

    # ── Validate the whole directory before uploading anything ──────────────
    plan = plan_directory(directory, set(args.slides))
    for msg in plan.warnings:
        _err(f"Warning: {msg}")
    if plan.problems:
        _err(f"Error: {len(plan.problems)} problem(s) found; nothing was uploaded.")
        for msg in plan.problems:
            _err(f"  - {msg}")
        return 2
    if not plan.files:
        _err(f"No supported files found under {directory}")
        return 0

    batches = make_batches(plan.files)
    print(f"Found {len(plan.files)} file(s) under {directory} → {len(batches)} batch(es)")
    if plan.skipped:
        print(f"Skipping {len(plan.skipped)} file(s): {', '.join(plan.skipped[:10])}"
              + (" …" if len(plan.skipped) > 10 else ""))

    if args.dry_run:
        for n, batch in enumerate(batches, 1):
            print(f"Batch {n}:")
            for f in batch:
                print(f"  [{f.kind:12s}] {f.relative}")
        print(f"Dry run: nothing sent to {api_url}.")
        return 0

    token = os.environ.get(TOKEN_ENV, "").strip() or None
    host = urlsplit(api_url).hostname or ""
    if token and urlsplit(api_url).scheme == "http" and not _is_loopback(host):
        _err(f"Warning: sending {TOKEN_ENV} over unencrypted HTTP to {host}; use https://.")

    totals = {"files_processed": 0, "chunks_produced": 0, "topics_added": 0,
              "subtopics_added": 0, "contents_added": 0, "related_edges_added": 0}
    failed_files: list[str] = []
    last_result: dict = {}
    done_batches = 0

    with build_client(api_url, token, transport) as client:
        for n, batch in enumerate(batches, 1):
            print(f"\nBatch {n}/{len(batches)}: uploading {len(batch)} file(s)…")
            try:
                job_id = submit_batch(client, batch, bool(token))
                print(f"  job {job_id} accepted; waiting for completion")
                result = poll_job(
                    client, job_id,
                    interval=args.poll_interval, timeout=args.timeout,
                    has_token=bool(token), sleep=sleep, clock=clock,
                    on_progress=lambda stage: print(f"  … {stage}"),
                )
            except (CLIError, OSError) as exc:
                _err(f"Error: {exc}")
                if done_batches:
                    _err(f"Batches 1-{done_batches} completed and were kept.")
                not_sent = sum(len(b) for b in batches[n:])
                if not_sent:
                    _err(f"{not_sent} file(s) in later batches were not submitted.")
                return 1
            done_batches += 1
            for key in totals:
                totals[key] += int(result.get(key) or 0)
            failed_files.extend(result.get("failed") or [])
            last_result = result
            for w in result.get("warnings") or []:
                _err(f"  warning: {w}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Ingestion complete")
    print("=" * 50)
    print(f"  Files processed     : {totals['files_processed']}/{len(plan.files)}")
    if failed_files:
        print(f"  Failed files        : {', '.join(failed_files)}")
    print(f"  Chunks produced     : {totals['chunks_produced']}")
    print(f"  Topics added        : {totals['topics_added']}")
    print(f"  Subtopics added     : {totals['subtopics_added']}")
    print(f"  Contents added      : {totals['contents_added']}")
    print(f"  RELATED_TO edges    : {totals['related_edges_added']}")
    print(f"  Graph nodes         : {last_result.get('graph_nodes', '?')}")
    print(f"  Graph edges         : {last_result.get('graph_edges', '?')}")
    print("=" * 50)
    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())

"""Fixed subprocess entry point for bounded, credential-free parser work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from backend.utils.processes import (
    DOCUMENT_LIMITS, KEYFRAME_LIMITS, PDF_LIMITS, WHISPER_LIMITS, apply_worker_limits,
)


def main() -> None:
    # Preserve the response pipe, then send Python and native-library stdout
    # diagnostics to the parent's separately bounded stderr pipe.
    response_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    payload = json.loads(sys.stdin.buffer.read(64 * 1024))
    kind = payload["kind"]
    limits = {"office": DOCUMENT_LIMITS, "whisper": WHISPER_LIMITS, "keyframes": KEYFRAME_LIMITS,
              "pdf": PDF_LIMITS, "slides": PDF_LIMITS}[kind]
    apply_worker_limits(limits)
    path = Path(payload["path"])
    options = payload.get("options", {})
    try:
        if kind == "office":
            from backend.ingestion.office_processor import _convert_office
            result = _convert_office(path)
        elif kind == "whisper":
            from backend.ingestion.media_limits import transcribe_local
            result = transcribe_local(path, options)
        elif kind == "pdf":
            from backend.ingestion.pdf_processor import _process_pdf_local
            chunks = _process_pdf_local(path, options)
            result = [chunk.model_dump(mode="json") for chunk in chunks]
        elif kind == "slides":
            from backend.ingestion.slide_processor import _extract_slides_local
            result = _extract_slides_local(path, options)
        else:
            from backend.ingestion.video_transcriber import _extract_keyframes_local
            result = _extract_keyframes_local(path, Path(options["keyframe_dir"]), options["threshold"], options["sample_fps"])
            result = [{**frame, "path": str(frame["path"])} for frame in result]
        response = {"result": result}
    except (Exception, MemoryError) as exc:
        # Return a short validation message without a traceback or environment.
        response = {"error": str(exc)[:512] or "Parser failed or exceeded its processing allowance."}
    encoded = json.dumps(response, ensure_ascii=False).encode("utf-8")
    if len(encoded) > limits.stdout_bytes:
        encoded = b'{"error":"Parser output exceeds the processing limit."}'
    with os.fdopen(response_fd, "wb") as response:
        response.write(encoded)


if __name__ == "__main__":
    main()

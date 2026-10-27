"""Atomic replacement of persisted text files without truncating prior data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Persist directory entries where directory fsync is supported.

    Windows does not expose portable directory flushing through os.open;
    file contents are still flushed before atomic replacement there.
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory_durable(path: Path) -> None:
    """Create missing directory entries and flush each newly created parent."""
    missing = []
    current = Path(path)
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_directory(directory.parent)


def copy_file_durable(source: Path, destination: Path) -> None:
    """Copy a regular file, flushing its contents before it can be published."""
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while block := incoming.read(1024 * 1024):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    fsync_directory(destination.parent)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    ensure_directory_durable(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

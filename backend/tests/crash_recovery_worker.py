"""Child-process harness: crash without unwinding transactions or file contexts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from backend.db.document_store import DocumentStore
from backend.schemas import Chunk


CRASH_EXIT = 91
DATABASE = Path("processed/chunks.db")
JOURNAL = Path("journal")
OLD_FILES: dict[str, bytes | None] = {
    "raw/notes.pdf": b"OLD PDF",
    "processed/faiss.index": b"OLD INDEX",
    "processed/faiss.json": b'["old-chunk"]',
    "processed/graph.json": b'{"revision":"old"}',
    "raw/new.pdf": None,
    "processed/empty.dat": b"",
}
NEW_FILES: dict[str, bytes] = {
    "raw/notes.pdf": b"NEW PDF",
    "processed/faiss.index": b"NEW INDEX",
    "processed/faiss.json": b'["new-chunk","additional-chunk"]',
    "processed/graph.json": b'{"revision":"new"}',
    "raw/new.pdf": b"NEW SOURCE",
    "processed/empty.dat": b"",
}
MIXED_FILES: dict[str, bytes | None] = {
    relative: content if index % 2 == 0 else None
    for index, (relative, content) in enumerate(NEW_FILES.items())
}


def old_chunks() -> list[Chunk]:
    return [
        Chunk(id="old-chunk", source_id="notes", modality="pdf", text="OLD TEXT"),
        Chunk(id="unchanged", source_id="other", modality="pdf", text="UNRELATED TEXT"),
    ]


def new_chunks() -> list[Chunk]:
    return [
        Chunk(id="new-chunk", source_id="notes", modality="pdf", text="NEW TEXT"),
        Chunk(id="additional-chunk", source_id="additional", modality="pdf", text="NEW SOURCE TEXT"),
    ]


def seed(root: Path) -> None:
    for relative, content in OLD_FILES.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if content is not None:
            destination.write_bytes(content)
    for relative, content in NEW_FILES.items():
        candidate = root / "incoming" / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(content)
    with DocumentStore(root / DATABASE) as store:
        store.save_chunks(old_chunks())


def main(arguments: list[str]) -> int:
    # Import here so collection can read fixtures while the implementation is
    # being developed, without manufacturing a fake recovery module.
    from backend.utils import recovery
    from backend.utils.recovery import RecoveryManager

    root = Path(arguments[0]).resolve()
    mode, crash_phase = arguments[1:3]

    def crash(phase: str) -> None:
        if phase == crash_phase:
            (root / "crashed-at.txt").write_text(phase, encoding="utf-8")
            os._exit(CRASH_EXIT)

    if crash_phase in {"during_backup", "during_copy", "during_restore"}:
        original_copy = recovery.copy_file_durable
        suffixes = {"during_backup": ".bak", "during_copy": ".new", "during_restore": ".restore"}

        def partial_copy(source: Path, destination: Path):
            if Path(destination).suffix == suffixes[crash_phase]:
                # Leave a real partially written companion behind, without
                # context cleanup or a normal return from the file helper.
                with Path(source).open("rb") as reader, Path(destination).open("wb") as writer:
                    writer.write(reader.read(1))
                    writer.flush()
                    os.fsync(writer.fileno())
                    crash(crash_phase)
            return original_copy(source, destination)

        recovery.copy_file_durable = partial_copy

    if crash_phase in {"delete_move_applied", "delete_restore_applied"}:
        original_replace = recovery.os.replace

        def abrupt_replace(source, destination):
            result = original_replace(source, destination)
            if Path(destination).suffix == ".bak":
                crash("delete_move_applied")
            elif Path(source).suffix == ".bak":
                crash("delete_restore_applied")
            return result

        recovery.os.replace = abrupt_replace

    with DocumentStore(root / DATABASE) as store:
        manager = RecoveryManager(root / JOURNAL, [root], store, failpoint=crash)
        if mode == "recover":
            manager.recover()
            return 0

        if crash_phase in {"sql_rows_uncommitted", "sql_committed_before_manager_resumes"}:
            connection = store._conn

            class CrashAtCommit:
                def __getattr__(self, name):
                    return getattr(connection, name)

                def commit(self):
                    crash("sql_rows_uncommitted")
                    result = connection.commit()
                    crash("sql_committed_before_manager_resumes")
                    return result

            store._conn = CrashAtCommit()

        if mode in {"clear", "clear_rollback"}:
            files = [(None, root / relative) for relative in OLD_FILES]

            def commit(transaction_id: str):
                if mode == "clear_rollback":
                    raise RuntimeError("intentional failure before SQLite clear commit")
                return store.clear_all(transaction_id=transaction_id)
        else:
            files = [
                (None if mode.startswith("mixed") and content is None else root / "incoming" / relative, root / relative)
                for relative, content in (MIXED_FILES if mode.startswith("mixed") else NEW_FILES).items()
            ]

            def commit(transaction_id: str):
                if mode in {"rollback", "mixed_rollback"}:
                    raise RuntimeError("intentional failure before SQLite commit")
                return store.replace_sources(new_chunks(), transaction_id=transaction_id)

        manager.publish(files, commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

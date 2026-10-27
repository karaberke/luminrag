"""Prepare a clean HF Space checkout without local data, keys, or tests.

Run from the repository root:
    python scripts/prepare_space.py --output .tmp/hf-space
Then initialize/push that directory to the desired Space repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


_EXCLUDED_DIRECTORIES = {"data", "tests", "__pycache__"}
_MANIFEST = Path("backend/utils/model_manifest.json")


def _checked_source(root: Path, source: Path) -> Path:
    for path in (source, *source.parents):
        if path == root.parent:
            break
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Deployment sources and output paths must not contain symbolic links or junctions.")
    if not source.resolve().is_relative_to(root):
        raise ValueError("Deployment sources must remain inside the repository.")
    return source


def prepare_space(root: Path, output: Path) -> Path:
    root = root.resolve()
    output = output.absolute()
    _checked_source(root, output)
    if output.resolve() == root or not output.resolve().is_relative_to(root):
        raise ValueError("Choose a new output directory inside the repository.")
    sources: list[tuple[Path, Path]] = []
    for directory in ("backend", "config"):
        start = _checked_source(root, root / directory)
        for directory_path, folders, names in os.walk(start, followlinks=False):
            folders[:] = [name for name in folders if name not in _EXCLUDED_DIRECTORIES]
            for folder in folders:
                _checked_source(root, Path(directory_path) / folder)
            for name in names:
                source = Path(directory_path) / name
                relative = source.relative_to(root)
                if source.suffix in {".py", ".yaml"} or relative == _MANIFEST:
                    _checked_source(root, source)
                    if not source.is_file():
                        raise ValueError("Deployment source is not a regular file.")
                    sources.append((source, relative))
    for name in ("scripts/cache_models.py", "requirements-hosted.txt", "requirements-hosted.lock",
                 "requirements-bootstrap.txt", "requirements-bootstrap.lock", "LICENSE", "README.md"):
        source = _checked_source(root, root / name)
        if not source.is_file():
            raise ValueError("Deployment source is not a regular file.")
        sources.append((source, Path(name)))
    sources.append((_checked_source(root, root / "Dockerfile.hosted"), Path("Dockerfile")))
    # Validate every source before creating a destination or reading source bytes.
    output.mkdir(parents=True, exist_ok=False)
    for source, relative in sources:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        output = prepare_space(root, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Prepared HF Docker Space source: {output}")


if __name__ == "__main__":
    main()

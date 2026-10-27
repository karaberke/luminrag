"""Shared data locations for processors and isolated application instances."""

from pathlib import Path


def processed_data_dir(config: dict) -> Path:
    """Use the application-provided location, preserving direct processor defaults."""
    return Path(config.get("data", {}).get("processed_dir", "backend/data/processed"))

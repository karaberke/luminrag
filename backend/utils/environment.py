"""Lightweight environment boundaries shared by entry points and workers."""
from __future__ import annotations

import os


def hosted_mode() -> bool:
    """Use the same explicit hosted flag before dotenv, SDKs, and middleware."""
    return os.getenv("LUMIN_HOSTED", "").strip() == "1"

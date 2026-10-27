"""Format optional source timestamps without failing an evidence response."""

from __future__ import annotations

import math


def format_timestamp(value: object) -> str | None:
    """Return elapsed ``minutes:seconds`` for finite, nonnegative seconds.

    Persisted source metadata can contain malformed or non-finite values.
    Such values have no usable media position and are omitted from evidence.
    Fractional seconds are truncated consistently with the media link format.
    """
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    whole_seconds = int(seconds)
    return f"{whole_seconds // 60}:{whole_seconds % 60:02d}"

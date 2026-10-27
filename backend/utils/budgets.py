"""Bound retained JSON-compatible work before copying potentially large data."""
from __future__ import annotations

import json
from dataclasses import dataclass

_JSON_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))

@dataclass
class JsonWorkBudget:
    max_items: int
    max_bytes: int
    copies: int = 1
    items: int = 0
    bytes: int = 0

    def reserve(self, value, *, items: int = 1) -> None:
        """Walk bounded serialized pieces; never allocate a full JSON payload."""
        self.items += items * self.copies
        if self.items > self.max_items:
            raise ValueError("Graph edit exceeds the undo processing limit. Use a smaller batch.")
        for piece in _JSON_ENCODER.iterencode(value):
            self.bytes += len(piece.encode("utf-8")) * self.copies
            if self.bytes > self.max_bytes:
                raise ValueError("Graph edit exceeds the undo processing limit. Use a smaller batch.")

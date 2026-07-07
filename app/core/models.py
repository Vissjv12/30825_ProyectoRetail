"""Shared data models for frames and pipeline payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class FramePayload:
    """Represents a captured video frame and its metadata."""

    frame_id: str
    timestamp: str
    source: str
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC timestamp in ISO 8601 format."""

        return datetime.now(timezone.utc).isoformat()


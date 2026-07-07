"""Interfaces that decouple system modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.models import FramePayload


class IFrameSource(ABC):
    """Abstraction for any component capable of capturing frames."""

    @abstractmethod
    def read(self) -> FramePayload:
        """Capture one frame and return it as a payload."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying capture resource."""


"""Adaptive transport driver. The app owns the time loop."""

from __future__ import annotations

from typing import Any

from adaptdg.flow import FlowState


class AdaptiveTransport:
    """Owns mesh adaptation and field transfer; the app owns time loop and sources."""

    def __init__(self, mesh: Any, flow: FlowState | None = None, **options: Any) -> None:
        raise NotImplementedError

    def register_field(self, name: str, kind: str = "dg1", initial: Any | None = None) -> None:
        """Register a field to be transferred whenever the mesh changes."""
        raise NotImplementedError

    def advance(self, flow: FlowState) -> None:
        """Transport step plus estimate, mark, adapt and transfer."""
        raise NotImplementedError

    def water_volume(self) -> float:
        """Integral of theta over the current mesh (conservation diagnostic)."""
        raise NotImplementedError

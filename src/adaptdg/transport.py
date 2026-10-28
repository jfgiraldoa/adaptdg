"""DG1 advection-diffusion transport step."""

from __future__ import annotations

from typing import Any


def transport_step(c_old: Any, flow: Any, dt: float, **options: Any) -> Any:
    """Advance a DG1 concentration field by one step with the given flow."""
    raise NotImplementedError

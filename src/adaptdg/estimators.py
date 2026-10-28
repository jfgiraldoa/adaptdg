"""A-posteriori error estimators."""

from __future__ import annotations

from typing import Any


def cell_indicators(c: Any, flow: Any) -> Any:
    """Return per-cell error indicators for a DG1 field."""
    raise NotImplementedError

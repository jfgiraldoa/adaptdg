"""Mesh refinement and coarsening."""

from __future__ import annotations

from typing import Any


def adapt_mesh(mesh: Any, marked_refine: Any, marked_coarsen: Any | None = None) -> Any:
    """Return an adapted mesh from refinement/coarsening markers."""
    raise NotImplementedError

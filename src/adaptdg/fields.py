"""Registry of fields transferred when the mesh changes."""

from __future__ import annotations

from typing import Any

FIELD_KINDS = ("dg1", "cell_average")


def register_field(name: str, kind: str = "dg1", initial: Any | None = None) -> Any:
    """Register a field by name; kind is 'dg1' or 'cell_average'."""
    raise NotImplementedError

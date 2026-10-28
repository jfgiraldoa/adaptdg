"""Conservative transfer of fields between meshes."""

from __future__ import annotations

from typing import Any


def transfer_field(field: Any, old_mesh: Any, new_mesh: Any, kind: str = "dg1") -> Any:
    """Transfer a field to a new mesh, preserving its integral."""
    raise NotImplementedError

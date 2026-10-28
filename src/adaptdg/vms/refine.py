"""Dimension-agnostic (d = 2, 3) refinement of marked cells by marking all their edges."""

from __future__ import annotations

from typing import Any

import numpy as np


def refine_marked(domain: Any, marked: np.ndarray) -> tuple[Any, np.ndarray]:
    """Refine the cells flagged in ``marked``; return the new mesh and each new cell's parent.

    All edges of a marked cell are refined (plus the conforming closure added by DOLFINx), so
    the same code serves triangles and tetrahedra. The parent map is permuted to the final local
    cell ordering through ``original_cell_index``.
    """
    from dolfinx import mesh

    d = domain.topology.dim
    ncells = domain.topology.index_map(d).size_local
    flags = np.asarray(marked, dtype=bool)
    if flags.shape != (ncells,):
        raise ValueError("There must be one refinement marker per local cell")
    if not flags.any():
        return domain, np.arange(ncells, dtype=np.int32)
    domain.topology.create_connectivity(d, 1)
    c2e = domain.topology.connectivity(d, 1)
    edges = np.unique(np.concatenate([c2e.links(int(c)) for c in np.flatnonzero(flags)]))
    refined, parent, _ = mesh.refine(
        domain, edges.astype(np.int32), option=mesh.RefinementOption.parent_cell
    )
    order = np.asarray(refined.topology.original_cell_index, dtype=np.int64)
    return refined, np.asarray(parent, dtype=np.int32)[order]


def refine_uniform(domain: Any) -> Any:
    """Refine every cell."""
    from dolfinx import mesh

    return mesh.refine(domain)[0]

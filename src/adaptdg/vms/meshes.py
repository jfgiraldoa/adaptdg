"""Start meshes (crossed triangles / tetrahedra) and boundary tagging; dolfinx is imported lazily."""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptdg.vms.problems import Marker


def _comm(comm: Any) -> Any:
    if comm is not None:
        return comm
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _remove_cells(domain: Any, inside: Marker) -> Any:
    """Return the submesh of cells whose midpoint does not satisfy ``inside`` (midpoints.T)."""
    from dolfinx import mesh

    d = domain.topology.dim
    cells = np.arange(domain.topology.index_map(d).size_local, dtype=np.int32)
    mid = mesh.compute_midpoints(domain, d, cells)
    keep = np.flatnonzero(~inside(mid.T)).astype(np.int32)
    return mesh.create_submesh(domain, d, keep)[0]


def crossed_square(n: int, lo: float = -1.0, hi: float = 1.0, comm: Any = None) -> Any:
    """``n x n`` squares, each split into four triangles (crossed diagonals)."""
    from dolfinx import mesh

    return mesh.create_rectangle(
        _comm(comm), [(lo, lo), (hi, hi)], (n, n), mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.crossed,
    )


def crossed_rectangle(n: int, lo: tuple[float, float], hi: tuple[float, float],
                      comm: Any = None) -> Any:
    """``n x n`` rectangles on ``[lo, hi]``, each split into four triangles (crossed diagonals)."""
    from dolfinx import mesh

    return mesh.create_rectangle(
        _comm(comm), [lo, hi], (n, n), mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed
    )


def l_shape(n: int, comm: Any = None) -> Any:
    """(-1,1)^2 minus the quadrant x>0, y<0; ``n`` (even) cells per side before the cut."""
    if n % 2:
        raise ValueError("n must be even so that the cut follows cell edges")
    return _remove_cells(crossed_square(n, comm=comm), lambda m: (m[0] > 0) & (m[1] < 0))


def fichera(n: int, comm: Any = None) -> Any:
    """(-1,1)^3 minus [0,1)^3 (Fichera corner at the origin); ``n`` even cells per side."""
    from dolfinx import mesh

    if n % 2:
        raise ValueError("n must be even so that the cut follows cell edges")
    box = mesh.create_box(
        _comm(comm), [(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)], (n, n, n), mesh.CellType.tetrahedron
    )
    return _remove_cells(box, lambda m: (m > 0).all(axis=0))


def tag_boundary(domain: Any, markers: dict[int, Marker]) -> Any:
    """Facet ``MeshTags`` with the given tags on the exterior boundary (later tags win)."""
    from dolfinx import mesh

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    tagged: dict[int, int] = {}
    for tag, marker in markers.items():
        for facet in mesh.locate_entities_boundary(domain, fdim, marker):
            tagged[int(facet)] = tag
    facets = np.array(sorted(tagged), dtype=np.int32)
    values = np.array([tagged[int(f)] for f in facets], dtype=np.int32)
    return mesh.meshtags(domain, fdim, facets, values)

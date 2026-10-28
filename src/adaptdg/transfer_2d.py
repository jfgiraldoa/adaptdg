"""Conservative transfers between adaptive triangles and a Cartesian chemistry grid.

Extracted unchanged from ASRTM-PHREEQC (src/asrtm_phreeqc/transfer_2d.py, commit a152b91).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
from numpy.typing import ArrayLike, NDArray


_PLAN_CACHE_MAX_ENTRIES = 16
_PLAN_CACHE_MAX_BYTES = 512 * 1024**2
_PLAN_CACHE: OrderedDict[tuple[object, ...], "ConservativeRemapPlan2D"] = (
    OrderedDict()
)
_PLAN_CACHE_BYTES = 0
_PLAN_CACHE_HITS = 0
_PLAN_CACHE_MISSES = 0


def _edges(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 2 or not np.all(np.diff(result) > 0.0):
        raise ValueError(f"{name} must be a strictly increasing one-dimensional array")
    return result


def _triangles(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (3, 2):
        raise ValueError("Triangles must have shape (number_of_cells, 3, 2)")
    if not np.all(np.isfinite(result)):
        raise ValueError("Triangle coordinates must be finite")
    return result


def triangle_areas(triangles: ArrayLike) -> NDArray[np.float64]:
    """Return the unsigned area of each physical triangle."""

    tri = _triangles(triangles)
    first = tri[:, 1, :] - tri[:, 0, :]
    second = tri[:, 2, :] - tri[:, 0, :]
    areas = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    if np.any(areas <= 0.0):
        raise ValueError("Degenerate triangles are not supported")
    return areas


def _clip_axis(
    polygon: list[tuple[np.float64, np.float64]],
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[np.float64, np.float64]]:
    """Clip a tiny polygon without allocating an array at every edge.

    The transfer builder calls this routine hundreds of thousands of times.
    Keeping the at-most-seven vertices as Python tuples avoids four temporary
    NumPy arrays per candidate rectangle.  The intersection arithmetic and the
    final shoelace evaluation retain the legacy operation order, so the
    resulting conservative weights remain bitwise identical.
    """

    if not polygon:
        return polygon

    def inside(point: tuple[np.float64, np.float64]) -> bool:
        if keep_greater:
            return bool(point[axis] >= bound)
        return bool(point[axis] <= bound)

    output: list[tuple[np.float64, np.float64]] = []
    start = polygon[-1]
    start_inside = inside(start)
    for end in polygon:
        end_inside = inside(end)
        if end_inside != start_inside:
            denominator = end[axis] - start[axis]
            if denominator != 0.0:
                fraction = (bound - start[axis]) / denominator
                output.append(
                    (
                        start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]),
                    )
                )
        if end_inside:
            output.append(end)
        start = end
        start_inside = end_inside
    return output


def _intersection_area_with_rectangle(
    triangle: NDArray[np.float64],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> float:
    polygon = [tuple(vertex) for vertex in triangle]
    polygon = _clip_axis(polygon, axis=0, bound=xmin, keep_greater=True)
    polygon = _clip_axis(polygon, axis=0, bound=xmax, keep_greater=False)
    polygon = _clip_axis(polygon, axis=1, bound=ymin, keep_greater=True)
    polygon = _clip_axis(polygon, axis=1, bound=ymax, keep_greater=False)
    if len(polygon) < 3:
        return 0.0
    polygon_array = np.asarray(polygon, dtype=np.float64)
    x = polygon_array[:, 0]
    y = polygon_array[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _candidate_range(edges: NDArray[np.float64], low: float, high: float) -> range:
    first = max(0, int(np.searchsorted(edges, low, side="right") - 1))
    last = min(edges.size - 2, int(np.searchsorted(edges, high, side="left")))
    return range(first, last + 1)


def _array_digest(values: NDArray[np.float64]) -> bytes:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return blake2b(contiguous.view(np.uint8), digest_size=16).digest()


@dataclass(frozen=True)
class ConservativeRemapPlan2D:
    """Reusable exact-overlap geometry for both conservative transfer directions."""

    triangle_indices: NDArray[np.int64]
    rectangle_indices: NDArray[np.int64]
    overlap_areas: NDArray[np.float64]
    triangle_areas: NDArray[np.float64]
    rectangle_areas: NDArray[np.float64]
    nx: int
    ny: int

    @property
    def memory_bytes(self) -> int:
        return int(
            self.triangle_indices.nbytes
            + self.rectangle_indices.nbytes
            + self.overlap_areas.nbytes
            + self.triangle_areas.nbytes
            + self.rectangle_areas.nbytes
        )

    @classmethod
    def create(
        cls,
        triangles: ArrayLike,
        x_edges: ArrayLike,
        y_edges: ArrayLike,
        *,
        require_rectangle_coverage: bool = True,
    ) -> "ConservativeRemapPlan2D":
        tri = _triangles(triangles)
        x = _edges(x_edges, "x_edges")
        y = _edges(y_edges, "y_edges")
        tri_areas = triangle_areas(tri)
        rectangle_areas = np.outer(np.diff(y), np.diff(x))
        tri_indices: list[int] = []
        rectangle_indices: list[int] = []
        overlaps: list[float] = []
        covered_triangles = np.zeros(tri.shape[0], dtype=np.float64)
        covered_rectangles = np.zeros(rectangle_areas.size, dtype=np.float64)
        nx = x.size - 1
        ny = y.size - 1
        for cell, polygon in enumerate(tri):
            xmin, ymin = np.min(polygon, axis=0)
            xmax, ymax = np.max(polygon, axis=0)
            for iy in _candidate_range(y, ymin, ymax):
                for ix in _candidate_range(x, xmin, xmax):
                    overlap = _intersection_area_with_rectangle(
                        polygon, x[ix], x[ix + 1], y[iy], y[iy + 1]
                    )
                    if overlap <= 0.0:
                        continue
                    rectangle = iy * nx + ix
                    tri_indices.append(cell)
                    rectangle_indices.append(rectangle)
                    overlaps.append(overlap)
                    covered_triangles[cell] += overlap
                    covered_rectangles[rectangle] += overlap
        if not np.allclose(
            covered_triangles, tri_areas, rtol=1.0e-10, atol=1.0e-14
        ):
            raise ValueError("The chemistry grid does not fully cover a transport triangle")
        if require_rectangle_coverage and not np.allclose(
            covered_rectangles, rectangle_areas.ravel(), rtol=1.0e-9, atol=1.0e-14
        ):
            raise ValueError("The transport triangles do not fully cover the chemistry grid")
        return cls(
            triangle_indices=np.asarray(tri_indices, dtype=np.int64),
            rectangle_indices=np.asarray(rectangle_indices, dtype=np.int64),
            overlap_areas=np.asarray(overlaps, dtype=np.float64),
            triangle_areas=tri_areas,
            rectangle_areas=rectangle_areas.ravel(),
            nx=nx,
            ny=ny,
        )

    def rectangles_to_triangles(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        expected = (self.ny, self.nx)
        if array.shape[-2:] != expected or not np.all(np.isfinite(array)):
            raise ValueError(f"Rectangle values must be finite and end with shape {expected}")
        flat = array.reshape((-1, self.rectangle_areas.size))
        target = np.stack(
            [
                np.bincount(
                    self.triangle_indices,
                    weights=field[self.rectangle_indices] * self.overlap_areas,
                    minlength=self.triangle_areas.size,
                )
                for field in flat
            ],
            axis=0,
        )
        target /= self.triangle_areas[None, :]
        return target.reshape(array.shape[:-2] + (self.triangle_areas.size,))

    def triangles_to_rectangles(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim == 0
            or array.shape[-1] != self.triangle_areas.size
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("The final value axis must contain finite triangle values")
        flat = array.reshape((-1, self.triangle_areas.size))
        target = np.stack(
            [
                np.bincount(
                    self.rectangle_indices,
                    weights=field[self.triangle_indices] * self.overlap_areas,
                    minlength=self.rectangle_areas.size,
                )
                for field in flat
            ],
            axis=0,
        )
        target /= self.rectangle_areas[None, :]
        return target.reshape(array.shape[:-1] + (self.ny, self.nx))


def _plan_key(
    triangles: NDArray[np.float64],
    x_edges: NDArray[np.float64],
    y_edges: NDArray[np.float64],
    require_rectangle_coverage: bool,
) -> tuple[object, ...]:
    return (
        triangles.shape,
        x_edges.shape,
        y_edges.shape,
        _array_digest(triangles),
        _array_digest(x_edges),
        _array_digest(y_edges),
        require_rectangle_coverage,
    )


def conservative_remap_plan_2d(
    triangles: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    *,
    require_rectangle_coverage: bool = True,
) -> ConservativeRemapPlan2D:
    """Return a bounded-cache exact-overlap plan for the specified grids."""

    global _PLAN_CACHE_BYTES, _PLAN_CACHE_HITS, _PLAN_CACHE_MISSES
    tri = _triangles(triangles)
    x = _edges(x_edges, "x_edges")
    y = _edges(y_edges, "y_edges")
    key = _plan_key(tri, x, y, require_rectangle_coverage)
    cached = _PLAN_CACHE.pop(key, None)
    if cached is not None:
        _PLAN_CACHE[key] = cached
        _PLAN_CACHE_HITS += 1
        return cached
    _PLAN_CACHE_MISSES += 1
    plan = ConservativeRemapPlan2D.create(
        tri,
        x,
        y,
        require_rectangle_coverage=require_rectangle_coverage,
    )
    _PLAN_CACHE[key] = plan
    _PLAN_CACHE_BYTES += plan.memory_bytes
    while (
        len(_PLAN_CACHE) > _PLAN_CACHE_MAX_ENTRIES
        or _PLAN_CACHE_BYTES > _PLAN_CACHE_MAX_BYTES
    ):
        _, evicted = _PLAN_CACHE.popitem(last=False)
        _PLAN_CACHE_BYTES -= evicted.memory_bytes
    return plan


def refine_conservative_remap_plan_2d(
    parent_plan: ConservativeRemapPlan2D,
    parent_cell: ArrayLike,
    child_triangles: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
) -> ConservativeRemapPlan2D:
    """Update a complete overlap plan after local triangle refinement.

    Overlaps belonging to unchanged parent cells are retained exactly.  Only
    children of split parents require new polygon clipping.  The assembled
    complete plan is inserted into the normal cache, so the subsequent
    transport-to-chemistry transfer reuses it without rebuilding geometry.
    """

    global _PLAN_CACHE_BYTES, _PLAN_CACHE_HITS, _PLAN_CACHE_MISSES
    children = _triangles(child_triangles)
    x = _edges(x_edges, "x_edges")
    y = _edges(y_edges, "y_edges")
    parents = np.asarray(parent_cell, dtype=np.int64)
    if parents.shape != (children.shape[0],):
        raise ValueError("There must be one parent index per child triangle")
    number_of_parents = parent_plan.triangle_areas.size
    if np.any(parents < 0) or np.any(parents >= number_of_parents):
        raise ValueError("Parent triangle indices are outside the parent plan")
    if parent_plan.nx != x.size - 1 or parent_plan.ny != y.size - 1:
        raise ValueError("Parent plan and Cartesian grid are incompatible")

    key = _plan_key(children, x, y, True)
    cached = _PLAN_CACHE.pop(key, None)
    if cached is not None:
        _PLAN_CACHE[key] = cached
        _PLAN_CACHE_HITS += 1
        return cached
    _PLAN_CACHE_MISSES += 1

    parent_counts = np.bincount(parents, minlength=number_of_parents)
    unchanged_children = parent_counts[parents] == 1
    refined_children = ~unchanged_children

    parent_to_child = np.full(number_of_parents, -1, dtype=np.int64)
    unchanged_indices = np.flatnonzero(unchanged_children)
    parent_to_child[parents[unchanged_indices]] = unchanged_indices
    retained_children = parent_to_child[parent_plan.triangle_indices]
    retained = retained_children >= 0

    refined_indices = np.flatnonzero(refined_children)
    partial = conservative_remap_plan_2d(
        children[refined_indices],
        x,
        y,
        require_rectangle_coverage=False,
    )
    triangle_indices = np.concatenate(
        (retained_children[retained], refined_indices[partial.triangle_indices])
    )
    rectangle_indices = np.concatenate(
        (
            parent_plan.rectangle_indices[retained],
            partial.rectangle_indices,
        )
    )
    overlap_areas = np.concatenate(
        (parent_plan.overlap_areas[retained], partial.overlap_areas)
    )
    areas = triangle_areas(children)
    covered_triangles = np.bincount(
        triangle_indices,
        weights=overlap_areas,
        minlength=children.shape[0],
    )
    covered_rectangles = np.bincount(
        rectangle_indices,
        weights=overlap_areas,
        minlength=parent_plan.rectangle_areas.size,
    )
    if not np.allclose(covered_triangles, areas, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("Incremental plan does not cover every child triangle")
    if not np.allclose(
        covered_rectangles,
        parent_plan.rectangle_areas,
        rtol=1.0e-9,
        atol=1.0e-14,
    ):
        raise ValueError("Incremental plan does not cover the Cartesian grid")

    plan = ConservativeRemapPlan2D(
        triangle_indices=triangle_indices,
        rectangle_indices=rectangle_indices,
        overlap_areas=overlap_areas,
        triangle_areas=areas,
        rectangle_areas=parent_plan.rectangle_areas,
        nx=parent_plan.nx,
        ny=parent_plan.ny,
    )
    _PLAN_CACHE[key] = plan
    _PLAN_CACHE_BYTES += plan.memory_bytes
    while (
        len(_PLAN_CACHE) > _PLAN_CACHE_MAX_ENTRIES
        or _PLAN_CACHE_BYTES > _PLAN_CACHE_MAX_BYTES
    ):
        _, evicted = _PLAN_CACHE.popitem(last=False)
        _PLAN_CACHE_BYTES -= evicted.memory_bytes
    return plan


def transfer_plan_cache_info() -> dict[str, int]:
    """Return lightweight cache diagnostics for timing manifests."""

    return {
        "entries": len(_PLAN_CACHE),
        "bytes": _PLAN_CACHE_BYTES,
        "hits": _PLAN_CACHE_HITS,
        "misses": _PLAN_CACHE_MISSES,
    }


def clear_transfer_plan_cache() -> None:
    """Clear cached geometry, primarily for controlled performance experiments."""

    global _PLAN_CACHE_BYTES, _PLAN_CACHE_HITS, _PLAN_CACHE_MISSES
    _PLAN_CACHE.clear()
    _PLAN_CACHE_BYTES = 0
    _PLAN_CACHE_HITS = 0
    _PLAN_CACHE_MISSES = 0


def rectangles_to_triangles(
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    rectangle_cell_averages: ArrayLike,
    triangles: ArrayLike,
) -> NDArray[np.float64]:
    """Conservatively average Cartesian chemistry cells onto triangles.

    The final two value axes are ordered ``(y_cell, x_cell)``. Leading axes,
    if present, are treated as independent fields.
    """

    x = _edges(x_edges, "x_edges")
    y = _edges(y_edges, "y_edges")
    tri = _triangles(triangles)
    values = np.asarray(rectangle_cell_averages, dtype=np.float64)
    expected = (y.size - 1, x.size - 1)
    if values.shape[-2:] != expected:
        raise ValueError(f"Rectangle values must end with shape {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rectangle values must be finite")

    return conservative_remap_plan_2d(tri, x, y).rectangles_to_triangles(values)


def rectangles_to_triangles_partial(
    x_edges: ArrayLike,
    y_edges: ArrayLike,
    rectangle_cell_averages: ArrayLike,
    triangles: ArrayLike,
) -> NDArray[np.float64]:
    """Average a Cartesian field onto a subset of transport triangles.

    Unlike :func:`rectangles_to_triangles`, the triangles need not cover the
    complete Cartesian domain.  Every supplied triangle must still be fully
    covered by the Cartesian grid.  This is used to update only newly created
    children after local refinement while retaining exact fixed-grid history.
    """

    x = _edges(x_edges, "x_edges")
    y = _edges(y_edges, "y_edges")
    tri = _triangles(triangles)
    values = np.asarray(rectangle_cell_averages, dtype=np.float64)
    expected = (y.size - 1, x.size - 1)
    if values.shape[-2:] != expected:
        raise ValueError(f"Rectangle values must end with shape {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rectangle values must be finite")
    return conservative_remap_plan_2d(
        tri, x, y, require_rectangle_coverage=False
    ).rectangles_to_triangles(values)


def triangles_to_rectangles(
    triangles: ArrayLike,
    triangle_cell_averages: ArrayLike,
    x_edges: ArrayLike,
    y_edges: ArrayLike,
) -> NDArray[np.float64]:
    """Conservatively average adaptive triangle values onto Cartesian cells."""

    tri = _triangles(triangles)
    x = _edges(x_edges, "x_edges")
    y = _edges(y_edges, "y_edges")
    values = np.asarray(triangle_cell_averages, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != tri.shape[0]:
        raise ValueError("The final value axis must contain one average per triangle")
    if not np.all(np.isfinite(values)):
        raise ValueError("Triangle values must be finite")

    return conservative_remap_plan_2d(tri, x, y).triangles_to_rectangles(values)

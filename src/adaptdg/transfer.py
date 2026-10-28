"""Conservative transfer operators between transport and chemistry meshes.

Extracted unchanged from ASRTM-PHREEQC (src/asrtm_phreeqc/transfer.py).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_edges(edges: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(edges, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two entries")
    if not np.all(np.diff(array) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def conservative_remap_1d(
    source_edges: ArrayLike,
    source_cell_averages: ArrayLike,
    target_edges: ArrayLike,
    *,
    domain_tolerance: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Remap one or more cell-average fields while preserving their integrals.

    The final axis of ``source_cell_averages`` indexes source cells. Every
    target value is the overlap-weighted average of the source cells covering
    it. Source and target meshes must cover the same one-dimensional domain.
    """

    source = _validated_edges(source_edges, "source_edges")
    target = _validated_edges(target_edges, "target_edges")
    values = np.asarray(source_cell_averages, dtype=np.float64)

    if values.ndim == 0 or values.shape[-1] != source.size - 1:
        raise ValueError("The final value axis must contain one average per source cell")
    if not np.all(np.isfinite(values)):
        raise ValueError("Source cell averages must be finite")
    if not (
        np.isclose(source[0], target[0], rtol=0.0, atol=domain_tolerance)
        and np.isclose(source[-1], target[-1], rtol=0.0, atol=domain_tolerance)
    ):
        raise ValueError("Source and target meshes must cover the same domain")

    target_values = np.zeros(values.shape[:-1] + (target.size - 1,), dtype=np.float64)
    source_cell = 0

    for target_cell, (left, right) in enumerate(zip(target[:-1], target[1:], strict=True)):
        while source_cell + 1 < source.size and source[source_cell + 1] <= left:
            source_cell += 1

        candidate = source_cell
        while candidate < source.size - 1 and source[candidate] < right:
            overlap_left = max(left, source[candidate])
            overlap_right = min(right, source[candidate + 1])
            overlap = overlap_right - overlap_left
            if overlap > 0:
                target_values[..., target_cell] += values[..., candidate] * overlap
            candidate += 1

        target_values[..., target_cell] /= right - left

    return target_values


@dataclass(frozen=True)
class ConservativeRemapPlan1D:
    """Sparse overlap plan reusable for any number of cell-average fields."""

    source_edges: NDArray[np.float64]
    target_edges: NDArray[np.float64]
    source_indices: NDArray[np.int64]
    target_offsets: NDArray[np.int64]
    weights: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        source_edges: ArrayLike,
        target_edges: ArrayLike,
        *,
        domain_tolerance: float = 1.0e-12,
    ) -> "ConservativeRemapPlan1D":
        source = _validated_edges(source_edges, "source_edges").copy()
        target = _validated_edges(target_edges, "target_edges").copy()
        if not (
            np.isclose(source[0], target[0], rtol=0.0, atol=domain_tolerance)
            and np.isclose(source[-1], target[-1], rtol=0.0, atol=domain_tolerance)
        ):
            raise ValueError("Source and target meshes must cover the same domain")

        source_indices: list[int] = []
        weights: list[float] = []
        offsets = [0]
        source_cell = 0
        for left, right in zip(target[:-1], target[1:], strict=True):
            while source_cell + 1 < source.size and source[source_cell + 1] <= left:
                source_cell += 1
            candidate = source_cell
            while candidate < source.size - 1 and source[candidate] < right:
                overlap = min(right, source[candidate + 1]) - max(
                    left, source[candidate]
                )
                if overlap > 0.0:
                    source_indices.append(candidate)
                    weights.append(overlap / (right - left))
                candidate += 1
            offsets.append(len(source_indices))

        if any(right == left for left, right in zip(offsets[:-1], offsets[1:])):
            raise RuntimeError("Every target cell must overlap at least one source cell")
        return cls(
            source_edges=source,
            target_edges=target,
            source_indices=np.asarray(source_indices, dtype=np.int64),
            target_offsets=np.asarray(offsets, dtype=np.int64),
            weights=np.asarray(weights, dtype=np.float64),
        )

    def apply(self, source_cell_averages: ArrayLike) -> NDArray[np.float64]:
        """Apply the stored overlap weights to one or more fields."""

        values = np.asarray(source_cell_averages, dtype=np.float64)
        if values.ndim == 0 or values.shape[-1] != self.source_edges.size - 1:
            raise ValueError("The final value axis must contain one average per source cell")
        if not np.all(np.isfinite(values)):
            raise ValueError("Source cell averages must be finite")
        contributions = values[..., self.source_indices] * self.weights
        return np.add.reduceat(contributions, self.target_offsets[:-1], axis=-1)


class ConservativeRemapCache1D:
    """Small LRU cache of sparse conservative overlap plans."""

    def __init__(self, *, maximum_plans: int = 64) -> None:
        if maximum_plans < 1:
            raise ValueError("maximum_plans must be positive")
        self.maximum_plans = maximum_plans
        self._plans: OrderedDict[
            tuple[bytes, bytes], ConservativeRemapPlan1D
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(source_edges: ArrayLike, target_edges: ArrayLike) -> tuple[bytes, bytes]:
        source = np.ascontiguousarray(source_edges, dtype=np.float64)
        target = np.ascontiguousarray(target_edges, dtype=np.float64)
        return source.tobytes(), target.tobytes()

    def remap(
        self,
        source_edges: ArrayLike,
        source_cell_averages: ArrayLike,
        target_edges: ArrayLike,
    ) -> NDArray[np.float64]:
        """Reuse or create a plan, then conservatively transfer the fields."""

        key = self._key(source_edges, target_edges)
        plan = self._plans.get(key)
        if plan is None:
            self.misses += 1
            plan = ConservativeRemapPlan1D.create(source_edges, target_edges)
            self._plans[key] = plan
            if len(self._plans) > self.maximum_plans:
                self._plans.popitem(last=False)
        else:
            self.hits += 1
            self._plans.move_to_end(key)
        return plan.apply(source_cell_averages)


def cell_averages_to_p1_nodes(cell_averages: ArrayLike) -> NDArray[np.float64]:
    """Reconstruct continuous P1 nodal values from uniform-cell averages.

    Boundary nodes inherit the adjacent cell value and each interior node is
    the arithmetic mean of its two neighbouring cells. This reconstruction
    preserves the global integral on a uniform one-dimensional mesh. It does
    not claim to preserve every individual cell average; that limitation is
    recorded explicitly until the DG transport port is available.
    """

    values = np.asarray(cell_averages, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cell averages must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cell averages must be finite")

    nodes = np.empty(values.size + 1, dtype=np.float64)
    nodes[0] = values[0]
    nodes[-1] = values[-1]
    if values.size > 1:
        nodes[1:-1] = 0.5 * (values[:-1] + values[1:])
    return nodes


def transfer_field(field: Any, old_mesh: Any, new_mesh: Any, kind: str = "dg1") -> Any:
    """Transfer a field to a new mesh, preserving its integral (not yet implemented)."""
    raise NotImplementedError

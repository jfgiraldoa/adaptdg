"""Mesh refinement, persistent-mesh policy and predictive reactive mesh (2D).

Provenance: ``adapt_mesh`` is a skeleton stub. The rest is copied from ASRTM-PHREEQC
(source repo asrtm-phreeqc-public, commit a152b91):
- ``src/asrtm_phreeqc/reactive_transport_policy_2d.py`` (``ReactiveTransportPolicy2D``,
  ``reactive_front_seed_regions``, ``reactive_front_seed_points``,
  ``create_predictive_reactive_mesh_2d``), only the import lines changed;
- ``src/asrtm_phreeqc/adaptive_calcite_dolomite_2d.py`` (``_triangle_restart_permutation``,
  ``persistent_mesh_rebuild_required``), verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from adaptdg.transport_2d import (
    create_rectangle_mesh_2d,
    mesh_triangles,
    refine_marked_triangles,
    triangle_characteristic_sizes,
)


def adapt_mesh(mesh: Any, marked_refine: Any, marked_coarsen: Any | None = None) -> Any:
    """Return an adapted mesh from refinement/coarsening markers."""
    raise NotImplementedError


def _triangle_restart_permutation(
    saved_triangles: np.ndarray, reconstructed_triangles: np.ndarray
) -> np.ndarray:
    """Map reconstructed-cell order to saved-cell order by physical geometry."""

    saved = np.asarray(saved_triangles, dtype=np.float64)
    reconstructed = np.asarray(reconstructed_triangles, dtype=np.float64)
    if (
        saved.shape != reconstructed.shape
        or saved.ndim != 3
        or saved.shape[1:] != (3, 2)
    ):
        raise ValueError(
            "Restart triangle arrays must have identical (cells, 3, 2) shape"
        )

    def key(triangle: np.ndarray) -> tuple[float, ...]:
        order = np.lexsort((triangle[:, 1], triangle[:, 0]))
        return tuple(np.round(triangle[order], decimals=14).ravel())

    saved_index = {key(triangle): index for index, triangle in enumerate(saved)}
    if len(saved_index) != saved.shape[0]:
        raise RuntimeError("Saved restart mesh contains duplicate triangle geometry")
    try:
        permutation = np.asarray(
            [saved_index[key(triangle)] for triangle in reconstructed],
            dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(
            "Reconstructed restart mesh does not match saved geometry"
        ) from error
    if np.unique(permutation).size != saved.shape[0]:
        raise RuntimeError("Restart triangle permutation is not one-to-one")
    return permutation


def persistent_mesh_rebuild_required(
    step_index: int, rebuild_interval: int, *, has_persistent_mesh: bool
) -> bool:
    """Return whether this physical step must construct a new predictive mesh."""

    if step_index < 0:
        raise ValueError("step_index cannot be negative")
    if rebuild_interval < 1:
        raise ValueError("rebuild_interval must be positive")
    return not has_persistent_mesh or step_index % rebuild_interval == 0



@dataclass(frozen=True)
class ReactiveTransportPolicy2D:
    """Numerical policy validated by the final transport-only experiment."""

    maximum_solves: int = 2
    corridor_width_in_chemistry_cells: float = 2.0
    target_size_in_chemistry_cells: float = 2.5
    core_corridor_width_in_chemistry_cells: float | None = None
    core_target_size_in_chemistry_cells: float | None = None
    maximum_predictive_levels: int = 8
    gradient_fraction: float = 0.15
    transition_gradient_fraction: float = 0.05
    mesh_rebuild_interval: int = 1
    reused_mesh_maximum_solves: int = 1
    incremental_exact_transfer: bool = True
    final_transfer_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.maximum_solves < 1:
            raise ValueError("maximum_solves must be positive")
        if self.corridor_width_in_chemistry_cells <= 0.0:
            raise ValueError("corridor width must be positive")
        if self.target_size_in_chemistry_cells <= 0.0:
            raise ValueError("target size must be positive")
        if (self.core_corridor_width_in_chemistry_cells is None) != (
            self.core_target_size_in_chemistry_cells is None
        ):
            raise ValueError(
                "core corridor and core target size must be enabled together"
            )
        if (
            self.core_corridor_width_in_chemistry_cells is not None
            and self.core_corridor_width_in_chemistry_cells <= 0.0
        ):
            raise ValueError("core corridor width must be positive")
        if (
            self.core_target_size_in_chemistry_cells is not None
            and self.core_target_size_in_chemistry_cells <= 0.0
        ):
            raise ValueError("core target size must be positive")
        if (
            self.core_target_size_in_chemistry_cells is not None
            and self.core_target_size_in_chemistry_cells
            > self.target_size_in_chemistry_cells
        ):
            raise ValueError("core target size cannot exceed the halo target size")
        if self.maximum_predictive_levels < 0:
            raise ValueError("maximum predictive levels cannot be negative")
        if not 0.0 < self.gradient_fraction <= 1.0:
            raise ValueError("gradient fraction must lie in (0, 1]")
        if not 0.0 < self.transition_gradient_fraction <= 1.0:
            raise ValueError("transition gradient fraction must lie in (0, 1]")
        if self.mesh_rebuild_interval < 1:
            raise ValueError("mesh rebuild interval must be positive")
        if not 1 <= self.reused_mesh_maximum_solves <= self.maximum_solves:
            raise ValueError(
                "reused mesh solves must lie between one and maximum_solves"
            )
        if self.final_transfer_tolerance <= 0.0:
            raise ValueError("final transfer tolerance must be positive")

    @property
    def core_halo_enabled(self) -> bool:
        """Whether a finer strong-gradient core is active inside the halo."""

        return self.core_target_size_in_chemistry_cells is not None

    @property
    def persistent_mesh_enabled(self) -> bool:
        """Whether an accepted mesh is reused between scheduled rebuilds."""

        return self.mesh_rebuild_interval > 1


def _inlet_seed_points(
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    left_inlet_y_min: float,
    top_inlet_x_max: float,
) -> np.ndarray:
    left_y = np.linspace(left_inlet_y_min, upper[1], 81)
    top_x = np.linspace(lower[0], top_inlet_x_max, 81)
    return np.vstack(
        (
            np.column_stack((np.full(left_y.size, lower[0]), left_y)),
            np.column_stack((top_x, np.full(top_x.size, upper[1]))),
        )
    )


def reactive_front_seed_regions(
    fields: Mapping[str, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    left_inlet_y_min: float,
    top_inlet_x_max: float,
    policy: ReactiveTransportPolicy2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strong-gradient core and wider transition-halo seed points."""

    x = np.asarray(x_edges, dtype=float)
    y = np.asarray(y_edges, dtype=float)
    centres_x = 0.5 * (x[:-1] + x[1:])
    centres_y = 0.5 * (y[:-1] + y[1:])
    xx, yy = np.meshgrid(centres_x, centres_y)
    core_active = np.zeros(xx.shape, dtype=bool)
    halo_active = np.zeros(xx.shape, dtype=bool)
    for name, raw in fields.items():
        values = np.asarray(raw, dtype=float)
        if values.shape != core_active.shape:
            raise ValueError(
                f"Predictive field {name} has shape {values.shape}; "
                f"expected {core_active.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Predictive field {name} contains non-finite values")
        span = float(np.max(values) - np.min(values))
        scale = max(span, float(np.max(np.abs(values))), np.finfo(float).tiny)
        normalized = (values - float(np.min(values))) / scale
        dy, dx = np.gradient(
            normalized,
            float(np.mean(np.diff(y))),
            float(np.mean(np.diff(x))),
        )
        gradient = np.hypot(dx, dy)
        maximum_gradient = float(np.max(gradient))
        if maximum_gradient <= 1.0e-14:
            continue
        transition = (normalized > 0.02) & (normalized < 0.98)
        strong = gradient >= policy.gradient_fraction * maximum_gradient
        transition_halo = transition & (
            gradient >= policy.transition_gradient_fraction * maximum_gradient
        )
        core_active |= strong
        halo_active |= strong | transition_halo

    inlet = _inlet_seed_points(
        lower=lower,
        upper=upper,
        left_inlet_y_min=left_inlet_y_min,
        top_inlet_x_max=top_inlet_x_max,
    )
    core = np.column_stack((xx[core_active], yy[core_active]))
    halo = np.column_stack((xx[halo_active], yy[halo_active]))
    core = inlet if core.size == 0 else np.vstack((core, inlet))
    halo = inlet if halo.size == 0 else np.vstack((halo, inlet))
    return core, halo


def reactive_front_seed_points(
    fields: Mapping[str, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    left_inlet_y_min: float,
    top_inlet_x_max: float,
    policy: ReactiveTransportPolicy2D,
) -> np.ndarray:
    """Return the legacy single-band union of reactive front locations."""

    _core, halo = reactive_front_seed_regions(
        fields,
        x_edges,
        y_edges,
        lower=lower,
        upper=upper,
        left_inlet_y_min=left_inlet_y_min,
        top_inlet_x_max=top_inlet_x_max,
        policy=policy,
    )
    return halo


def _swept_seed_points(
    seeds: np.ndarray,
    *,
    displacement: np.ndarray,
    lower: tuple[float, float],
    upper: tuple[float, float],
) -> np.ndarray:
    swept = np.vstack(
        [seeds + fraction * displacement for fraction in np.linspace(0.0, 1.0, 6)]
    )
    inside = np.logical_and.reduce(
        (
            swept[:, 0] >= lower[0],
            swept[:, 0] <= upper[0],
            swept[:, 1] >= lower[1],
            swept[:, 1] <= upper[1],
        )
    )
    return swept[inside]


def create_predictive_reactive_mesh_2d(
    *,
    fields: Mapping[str, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    lower: tuple[float, float],
    upper: tuple[float, float],
    initial_cells: tuple[int, int],
    velocity: tuple[float, float],
    time_step: float,
    left_inlet_y_min: float,
    top_inlet_x_max: float,
    policy: ReactiveTransportPolicy2D,
):
    """Build a coarse-to-local predictive mesh from all reactive fronts."""

    domain = create_rectangle_mesh_2d(
        lower=lower,
        upper=upper,
        cells=initial_cells,
    )
    core_seeds, halo_seeds = reactive_front_seed_regions(
        fields,
        x_edges,
        y_edges,
        lower=lower,
        upper=upper,
        left_inlet_y_min=left_inlet_y_min,
        top_inlet_x_max=top_inlet_x_max,
        policy=policy,
    )
    displacement = np.asarray(velocity, dtype=float) * float(time_step)
    halo_swept = _swept_seed_points(
        halo_seeds,
        displacement=displacement,
        lower=lower,
        upper=upper,
    )
    core_swept = _swept_seed_points(
        core_seeds,
        displacement=displacement,
        lower=lower,
        upper=upper,
    )
    if halo_swept.size == 0:
        return domain
    halo_tree = cKDTree(halo_swept)
    core_tree = cKDTree(core_swept) if core_swept.size else None
    chemistry_width = min(
        float(np.min(np.diff(x_edges))),
        float(np.min(np.diff(y_edges))),
    )
    corridor = policy.corridor_width_in_chemistry_cells * chemistry_width
    target_size = policy.target_size_in_chemistry_cells * chemistry_width
    core_corridor = (
        None
        if policy.core_corridor_width_in_chemistry_cells is None
        else policy.core_corridor_width_in_chemistry_cells * chemistry_width
    )
    core_target_size = (
        None
        if policy.core_target_size_in_chemistry_cells is None
        else policy.core_target_size_in_chemistry_cells * chemistry_width
    )
    for _level in range(policy.maximum_predictive_levels):
        triangles = mesh_triangles(domain)
        centres = np.mean(triangles, axis=1)
        sizes = triangle_characteristic_sizes(triangles)
        halo_distance, _ = halo_tree.query(centres, workers=1)
        marked = (halo_distance <= corridor + 0.75 * sizes) & (
            sizes > 1.01 * target_size
        )
        if (
            core_tree is not None
            and core_corridor is not None
            and core_target_size is not None
        ):
            core_distance, _ = core_tree.query(centres, workers=1)
            marked |= (core_distance <= core_corridor + 0.75 * sizes) & (
                sizes > 1.01 * core_target_size
            )
        if not np.any(marked):
            break
        refined = refine_marked_triangles(domain, marked)
        if mesh_triangles(refined).shape[0] == triangles.shape[0]:
            break
        domain = refined
    return domain

"""Adaptive loop (2D) with chemistry supplied by callbacks.

Provenance: ``AdaptiveTransport`` is a skeleton stub. The rest is copied from ASRTM-PHREEQC
(source: ASRTM-PHREEQC),
``src/asrtm_phreeqc/adaptive_calcite_dolomite_2d.py``: ``NegativeTransportCorrection2D``,
``AdaptiveRefinementSnapshot2D``, ``_prepare_chemistry_component_2d``,
``_solve_one_adaptive_transport_step_2d``, ``_rectangle_inventory``, ``_triangle_inventory``,
``_process_peak_rss_bytes`` and ``run_calcite_dolomite_adaptive_2d``. Control flow, order of
operations and numerics are unchanged. The chemistry and config touchpoints became:
``LoopSettings2D`` (replaces ``load_case``), ``ChemistryCallbacks2D`` (replaces the PHREEQC engine,
the mineral fields and the solid inventory), the ``components`` argument (replaces
``COMPONENTS_2D``) and ``model_floor`` (replaces ``PHREEQC_MODEL_FLOOR_MOL_L``).
The Dörfler union is ``adaptdg.marking.dorfler_union_2d``. See docs/DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
import sys
import warnings

import numpy as np

from adaptdg.f2a import F2AHistory, F2AOptions
from adaptdg.fastadap import (
    FastOptions,
    FastPath,
    fast_mesh_triangles,
    fast_refine_marked_triangles_with_parent,
)
from adaptdg.flow import FlowState
from adaptdg.marking import dorfler_union_2d
from adaptdg.mesh import (
    ReactiveTransportPolicy2D,
    _triangle_restart_permutation,
    create_predictive_reactive_mesh_2d,
    persistent_mesh_rebuild_required,
)
from adaptdg.transfer_2d import (
    conservative_remap_plan_2d,
    refine_conservative_remap_plan_2d,
    rectangles_to_triangles,
    rectangles_to_triangles_partial,
    transfer_plan_cache_info,
    triangle_areas,
    triangles_to_rectangles,
    use_plan_builder,
)
from adaptdg.transport_2d import (
    ResidualMinimizationBackwardEuler2D,
    StabilizedTransportStep2D,
    create_mesh_from_triangles_2d,
    create_rectangle_mesh_2d,
    mesh_triangles,
    refine_marked_triangles,
    refine_marked_triangles_with_parent,
    triangle_characteristic_sizes,
)

NEGATIVE_ROUNDOFF_TOLERANCE = 1.0e-10


class AdaptiveTransport:
    """Owns mesh adaptation and field transfer; the app owns time loop and sources."""

    def __init__(self, mesh: Any, flow: FlowState | None = None, **options: Any) -> None:
        raise NotImplementedError

    def register_field(self, name: str, kind: str = "dg1", initial: Any | None = None) -> None:
        """Register a field to be transferred whenever the mesh changes."""
        raise NotImplementedError

    def advance(self, flow: FlowState) -> None:
        """Transport step plus estimate, mark, adapt and transfer."""
        raise NotImplementedError

    def water_volume(self) -> float:
        """Integral of theta over the current mesh (conservation diagnostic)."""
        raise NotImplementedError


@dataclass(frozen=True)
class LoopSettings2D:
    """Numerical settings the loop reads (replaces the source ``load_case`` object).

    ``adaptivity`` is any object with ``theta``, ``epsilon_chem``, ``eligibility_factor``,
    ``tie_tolerance``, ``initial_cells_2d`` and ``maximum_refinements`` (as the source
    ``AdaptivityConfig``). ``left_inlet_y_min`` / ``top_inlet_x_max`` may be None (defaults
    0.4 / 0.1 as in the source). ``time_scheme`` is ``backward_euler`` or ``bdf2``.
    """

    lower: tuple[float, float]
    upper: tuple[float, float]
    chemistry_cells: tuple[int, int]
    time_step: float
    steps: int
    diffusion: float
    velocity: tuple[float, float]
    degree: int
    adaptivity: Any
    negative_concentration_policy: str = "legacy_clip"
    time_scheme: str = "backward_euler"
    left_inlet_y_min: float | None = None
    top_inlet_x_max: float | None = None


@dataclass(frozen=True)
class ChemistryCallbacks2D:
    """Everything the loop needs from the chemistry engine (the loop imports no chemistry).

    ``snapshot()`` returns the current chemistry state, an object with ``concentrations``
    (dict name -> flat array over chemistry cells, C order of the (ny, nx) grid).
    ``react(inputs, time_step)`` takes ``inputs`` (dict name -> (ny, nx) array) and returns an
    object with ``concentrations`` (flat arrays); it advances the engine state.
    ``boundary`` is the inflow concentration per component.
    ``seed_fields(snapshot, shape)`` returns extra (ny, nx) fields (e.g. minerals) used only to
    seed the predictive mesh (None: no extra fields).
    ``solid_inventory(name, state, shape, areas)`` returns the solid inventory of component
    ``name`` for a snapshot/reaction result, for the mass diagnostics (None: zero).
    """

    snapshot: Callable[[], Any]
    react: Callable[..., Any]
    boundary: Mapping[str, float]
    seed_fields: Callable[[Any, tuple[int, int]], Mapping[str, np.ndarray]] | None = None
    solid_inventory: Callable[[str, Any, tuple[int, int], np.ndarray], float] | None = None


@dataclass(frozen=True)
class NegativeTransportCorrection2D:
    """Diagnostics for the legacy non-negative concentration clipping."""

    minimum_before_clip: float
    corrected_cells: int
    added_integral_mol_L_m2: float
    model_floor_mol_L: float = 0.0
    floor_corrected_cells: int = 0
    floor_added_integral_mol_L_m2: float = 0.0




@dataclass(frozen=True)
class AdaptiveRefinementSnapshot2D:
    """One selected solve-estimate-mark state inside a physical timestep."""

    refinement_level: int
    solve_number: int
    triangles: np.ndarray
    node_coordinates: np.ndarray
    transported_cell_averages: dict[str, np.ndarray]
    transported_nodal_values: dict[str, np.ndarray]
    cell_indicators: dict[str, np.ndarray]
    marked_cells: np.ndarray



@dataclass(frozen=True)
class AdaptiveStep2D:
    """Final adaptive state and chemistry-grid fields at one physical step.

    ``chemistry_result`` is the object returned by ``ChemistryCallbacks2D.react``.
    """

    step: int
    time: float
    refinement_solves: int
    mesh_updates: int
    triangles: np.ndarray
    transported_cell_averages: dict[str, np.ndarray]
    transported_node_coordinates: dict[str, np.ndarray]
    transported_nodal_values: dict[str, np.ndarray]
    cell_indicators: dict[str, np.ndarray]
    chemistry_result: Any
    negative_transport_corrections: dict[str, NegativeTransportCorrection2D]
    chemistry_input_concentrations: dict[str, np.ndarray]
    chemistry_output_concentrations: dict[str, np.ndarray]
    refinement_history: tuple[AdaptiveRefinementSnapshot2D, ...]
    timings_seconds: dict[str, float]
    work_counts: dict[str, int]
    mass_balance_diagnostics: dict[str, dict[str, float]]
    reaction_history_cell_values: dict[str, np.ndarray] | None = None
    older_reaction_history_cell_values: dict[str, np.ndarray] | None = None



@dataclass(frozen=True)
class AdaptiveRun2D:
    """Time sequence of the adaptive 2D loop."""

    steps: tuple[AdaptiveStep2D, ...]
    chemistry_x_edges: np.ndarray
    chemistry_y_edges: np.ndarray
    boundary_concentrations: dict[str, float]
    # Statistics of the history-preserving rebuild (F2A) and of the fast path, if used.
    statistics: dict[str, dict] | None = None


def _dorfler_union_2d(
    indicators: dict[str, np.ndarray],
    cell_sizes: np.ndarray,
    *,
    chemistry_width: float,
    adaptivity,
    components: Sequence[str],
) -> np.ndarray:
    """Species-wise union of the legacy chemistry-aware Dörfler marking."""

    return dorfler_union_2d(
        indicators,
        cell_sizes,
        chemistry_width=chemistry_width,
        eligibility_factor=adaptivity.eligibility_factor,
        theta=adaptivity.theta,
        epsilon_chem=adaptivity.epsilon_chem,
        tie_tolerance=adaptivity.tie_tolerance,
        names=components,
    )


def prepare_chemistry_component_2d(
    name: str,
    values: np.ndarray,
    chemistry_cell_areas: np.ndarray,
    *,
    negative_concentration_policy: str,
    model_floor: float,
) -> tuple[np.ndarray, NegativeTransportCorrection2D]:
    """Apply an explicit positivity policy and the component model floor."""

    array = np.asarray(values, dtype=np.float64)
    areas = np.asarray(chemistry_cell_areas, dtype=np.float64)
    if array.shape != areas.shape:
        raise ValueError("Chemistry values and cell areas must have the same shape")
    minimum = float(np.min(array))
    negative_mask = array < 0.0
    if (
        negative_concentration_policy == "reject"
        and minimum < -NEGATIVE_ROUNDOFF_TOLERANCE
    ):
        raise RuntimeError(
            f"Adaptive 2D transport returned negative {name}: "
            f"minimum={minimum:.6e} mol/L"
        )

    corrected = np.maximum(array, 0.0)
    added_integral = float(np.sum(np.where(negative_mask, -array, 0.0) * areas))
    floor_mask = corrected < model_floor
    floor_added_integral = float(
        np.sum(np.where(floor_mask, model_floor - corrected, 0.0) * areas)
    )
    if model_floor > 0.0:
        corrected = np.maximum(corrected, model_floor)

    return corrected, NegativeTransportCorrection2D(
        minimum_before_clip=minimum,
        corrected_cells=int(np.count_nonzero(negative_mask)),
        added_integral_mol_L_m2=added_integral,
        model_floor_mol_L=model_floor,
        floor_corrected_cells=int(np.count_nonzero(floor_mask)),
        floor_added_integral_mol_L_m2=floor_added_integral,
    )



def solve_one_adaptive_transport_step_2d(
    *,
    components: Sequence[str],
    previous_chemistry: dict[str, np.ndarray],
    older_chemistry: dict[str, np.ndarray] | None = None,
    chemistry_x_edges: np.ndarray,
    chemistry_y_edges: np.ndarray,
    boundary: dict[str, float],
    lower: tuple[float, float],
    upper: tuple[float, float],
    time_step: float,
    diffusion: float,
    velocity: tuple[float, float],
    left_inlet_y_min: float,
    top_inlet_x_max: float,
    degree: int,
    adaptivity,
    maximum_solves: int,
    snapshot_levels: frozenset[int] = frozenset(),
    initial_domain: object | None = None,
    minimum_solves: int = 1,
    stagnation_ratio: float | None = None,
    stagnation_patience: int = 2,
    nested_previous_transfer: bool = False,
    incremental_exact_previous_transfer: bool = False,
    incremental_exact_transfer_plan: bool = False,
    full_exact_previous_transfer: bool = False,
    marking_filter: Callable[
        [dict[str, StabilizedTransportStep2D], np.ndarray, np.ndarray], np.ndarray
    ]
    | None = None,
    history_marking_filter: Callable[
        [
            dict[str, StabilizedTransportStep2D],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        np.ndarray,
    ]
    | None = None,
    nested_transfer_tolerance: float = 1.0e-10,
    time_scheme: str = "backward_euler",
    history_degree: int = 0,
    previous_history_cell_values: dict[str, np.ndarray] | None = None,
    older_history_cell_values: dict[str, np.ndarray] | None = None,
    f2a_history: F2AHistory | None = None,
    fast: FastPath | None = None,
) -> tuple[
    object,
    dict[str, StabilizedTransportStep2D],
    int,
    int,
    tuple[AdaptiveRefinementSnapshot2D, ...],
    np.ndarray,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, float],
    dict[str, int],
]:
    if adaptivity.initial_cells_2d is None:
        raise ValueError("The 2D case must declare adaptivity.initial_cells_2d")
    if maximum_solves < 1:
        raise ValueError("At least one transport solve is required")
    if not 1 <= minimum_solves <= maximum_solves:
        raise ValueError("minimum_solves must be within 1..maximum_solves")
    if stagnation_ratio is not None and not 0.0 < stagnation_ratio <= 1.0:
        raise ValueError("stagnation_ratio must be in (0, 1]")
    if stagnation_patience < 1:
        raise ValueError("stagnation_patience must be positive")
    if nested_transfer_tolerance <= 0.0:
        raise ValueError("nested_transfer_tolerance must be positive")
    if sum(
        bool(value)
        for value in (
            nested_previous_transfer,
            incremental_exact_previous_transfer,
            full_exact_previous_transfer,
        )
        ) > 1:
        raise ValueError(
            "Choose only one adaptive history-transfer strategy"
        )
    if incremental_exact_transfer_plan and not incremental_exact_previous_transfer:
        raise ValueError(
            "Incremental plan updates require incremental exact history transfer"
        )
    if time_scheme not in {"backward_euler", "bdf2"}:
        raise ValueError("time_scheme must be backward_euler or bdf2")
    if history_degree not in {0, 1}:
        raise ValueError("history_degree must be zero or one")
    if previous_history_cell_values is not None and history_degree != 1:
        raise ValueError("Complete history values require DG1 history")
    if older_history_cell_values is not None and previous_history_cell_values is None:
        raise ValueError("Older complete history requires current complete history")
    if time_scheme == "bdf2" and older_chemistry is None:
        raise ValueError("BDF2 requires older_chemistry")
    if time_scheme == "backward_euler" and older_chemistry is not None:
        raise ValueError("Backward Euler accepts no older_chemistry")

    # F2A (P2): children of an in-step refinement inherit the exact restriction of the parent
    # DG1 history. Fast path: solver cache, vectorised mesh helpers (see adaptdg.fastadap).
    make_solver = (
        ResidualMinimizationBackwardEuler2D if fast is None else fast.solver_factory
    )
    triangles_of = mesh_triangles if fast is None else fast_mesh_triangles
    refine_with_parent = (
        refine_marked_triangles_with_parent
        if fast is None
        else fast_refine_marked_triangles_with_parent
    )

    timings = {
        "transport_mesh_creation": 0.0,
        "transport_solver_setup": 0.0,
        "chemistry_to_transport_transfer": 0.0,
        "transport_linear_solves": 0.0,
        "error_estimate_and_mark": 0.0,
        "snapshot_copy": 0.0,
        "mesh_refinement": 0.0,
    }
    work_counts = {
        "linear_system_setups": 0,
        "transport_component_solves": 0,
        "accumulated_trial_dofs": 0,
        "accumulated_test_dofs": 0,
        "accumulated_mixed_dofs": 0,
        "accumulated_component_mixed_dof_solves": 0,
    }
    if initial_domain is None:
        started = perf_counter()
        domain = create_rectangle_mesh_2d(
            lower=lower,
            upper=upper,
            cells=adaptivity.initial_cells_2d,
        )
        timings["transport_mesh_creation"] += perf_counter() - started
    else:
        domain = initial_domain
    chemistry_width = min(
        float(np.min(np.diff(chemistry_x_edges))),
        float(np.min(np.diff(chemistry_y_edges))),
    )
    transport_results: dict[str, StabilizedTransportStep2D] = {}
    mesh_updates = 0
    snapshots: list[AdaptiveRefinementSnapshot2D] = []
    previous_normalised_estimator: float | None = None
    stagnation_count = 0
    previous_on_transport: np.ndarray | None = None
    older_on_transport: np.ndarray | None = None
    current_complete_history = previous_history_cell_values
    older_complete_history = older_history_cell_values
    previous_history_used: dict[str, np.ndarray] | None = None
    older_history_used: dict[str, np.ndarray] | None = None
    previous_stack = np.stack(
        [previous_chemistry[name] for name in components]
    )
    older_stack = (
        np.stack([older_chemistry[name] for name in components])
        if older_chemistry is not None
        else None
    )

    def retain_snapshot(
        refinement_level: int,
        refinement_solves: int,
        triangles: np.ndarray,
        marked: np.ndarray,
    ) -> None:
        """Copy one requested or terminal adaptive state for later output."""

        snapshots.append(
            AdaptiveRefinementSnapshot2D(
                refinement_level=refinement_level,
                solve_number=refinement_solves,
                triangles=triangles.copy(),
                node_coordinates=transport_results[
                    components[0]
                ].node_coordinates.copy(),
                transported_cell_averages={
                    name: transport_results[name].cell_averages.copy()
                    for name in components
                },
                transported_nodal_values={
                    name: transport_results[name].nodal_values.copy()
                    for name in components
                },
                cell_indicators={
                    name: transport_results[name].cell_indicators.copy()
                    for name in components
                },
                marked_cells=marked.copy(),
            )
        )

    for refinement_level in range(maximum_solves):
        started = perf_counter()
        solver = make_solver(
            domain=domain,
            lower=lower,
            upper=upper,
            time_step=time_step,
            diffusion=diffusion,
            velocity=velocity,
            left_inlet_y_min=left_inlet_y_min,
            top_inlet_x_max=top_inlet_x_max,
            degree=degree,
            name="calcite_dolomite_adaptive",
            time_scheme=time_scheme,
            history_degree=history_degree,
        )
        timings["transport_solver_setup"] += perf_counter() - started
        triangles = solver.triangles
        expected_history_shape = (
            triangles.shape[0],
            1 if history_degree == 0 else 3,
        )
        if current_complete_history is not None and any(
            current_complete_history[name].shape != expected_history_shape
            for name in components
        ):
            current_complete_history = None
            older_complete_history = None
        if older_complete_history is not None and any(
            older_complete_history[name].shape != expected_history_shape
            for name in components
        ):
            older_complete_history = None

        # A persistent DG1 mesh already carries the complete BDF2 histories.
        # Their cell averages are exact arithmetic means of the local DG1
        # coefficients, so remapping the same state from the chemistry grid is
        # redundant.  These averages are still returned for mass diagnostics
        # and for the nested-refinement fallback.  Rebuilt meshes have no
        # compatible complete history and continue to use the exact overlap
        # transfer below.
        if current_complete_history is not None and previous_on_transport is None:
            previous_on_transport = np.stack(
                [
                    np.mean(current_complete_history[name], axis=1)
                    for name in components
                ]
            )
            work_counts["history_average_transfers_avoided"] = (
                work_counts.get("history_average_transfers_avoided", 0) + 1
            )
        if older_complete_history is not None and older_on_transport is None:
            older_on_transport = np.stack(
                [
                    np.mean(older_complete_history[name], axis=1)
                    for name in components
                ]
            )
            work_counts["history_average_transfers_avoided"] = (
                work_counts.get("history_average_transfers_avoided", 0) + 1
            )
        if previous_on_transport is None:
            started = perf_counter()
            previous_on_transport = rectangles_to_triangles(
                chemistry_x_edges,
                chemistry_y_edges,
                previous_stack,
                triangles,
            )
            timings["chemistry_to_transport_transfer"] += perf_counter() - started
        if older_stack is not None and older_on_transport is None:
            started = perf_counter()
            older_on_transport = rectangles_to_triangles(
                chemistry_x_edges,
                chemistry_y_edges,
                older_stack,
                triangles,
            )
            timings["chemistry_to_transport_transfer"] += perf_counter() - started
        transport_results = {}
        started = perf_counter()
        begin_batch = getattr(solver, "begin_batch", None)
        if f2a_history is not None and begin_batch is not None:
            begin_batch(current_complete_history, older_complete_history, boundary)
        for component_index, name in enumerate(components):
            complete_current = (
                None
                if current_complete_history is None
                else current_complete_history[name]
            )
            complete_older = (
                None
                if older_complete_history is None
                else older_complete_history[name]
            )
            if complete_current is not None:
                solver.set_history_cell_values(complete_current, complete_older)
            elif older_on_transport is None:
                # Keep the legacy single-history entry point available to
                # external wrappers and the existing BE regression tests.
                solver.set_cell_averages(previous_on_transport[component_index])
            else:
                solver.set_history_cell_averages(
                    previous_on_transport[component_index],
                    older_on_transport[component_index],
                )
            transport_results[name] = solver.step(inflow_value=boundary[name])
        if history_degree == 1:
            previous_history_used = {}
            older_history_used = {} if older_on_transport is not None else None
            for component_index, name in enumerate(components):
                if current_complete_history is not None:
                    previous_history_used[name] = current_complete_history[name].copy()
                else:
                    previous_history_used[name] = np.repeat(
                        previous_on_transport[component_index, :, None], 3, axis=1
                    )
                if older_history_used is not None:
                    if older_complete_history is not None:
                        older_history_used[name] = older_complete_history[name].copy()
                    else:
                        older_history_used[name] = np.repeat(
                            older_on_transport[component_index, :, None], 3, axis=1
                        )
        timings["transport_linear_solves"] += perf_counter() - started

        trial_dofs = int(
            transport_results[components[0]].node_coordinates.shape[0]
        )
        test_dofs = 3 * int(triangles.shape[0])
        mixed_dofs = trial_dofs + test_dofs
        work_counts["linear_system_setups"] += 1
        work_counts["transport_component_solves"] += len(components)
        work_counts["accumulated_trial_dofs"] += trial_dofs
        work_counts["accumulated_test_dofs"] += test_dofs
        work_counts["accumulated_mixed_dofs"] += mixed_dofs
        work_counts["accumulated_component_mixed_dof_solves"] += (
            len(components) * mixed_dofs
        )

        refinement_solves = refinement_level + 1
        started = perf_counter()
        marked = _dorfler_union_2d(
            {name: transport_results[name].cell_indicators for name in components},
            triangle_characteristic_sizes(triangles),
            chemistry_width=chemistry_width,
            adaptivity=adaptivity,
            components=components,
        )
        if marking_filter is not None:
            filtered = np.asarray(
                marking_filter(transport_results, triangles, marked), dtype=bool
            )
            if filtered.shape != marked.shape:
                raise ValueError("marking_filter returned an incompatible shape")
            marked = filtered
        if history_marking_filter is not None:
            filtered = np.asarray(
                history_marking_filter(
                    transport_results,
                    triangles,
                    marked,
                    previous_on_transport,
                ),
                dtype=bool,
            )
            if filtered.shape != marked.shape:
                raise ValueError(
                    "history_marking_filter returned an incompatible shape"
                )
            marked = filtered
        # Optional refinement-stagnation guard.  It is disabled by default so
        # legacy/paper reproduction remains bit-for-bit on its original path.
        # The normalisation prevents a larger number of cells from appearing
        # artificially worse solely because the indicator vector grew.
        areas = triangle_areas(triangles)
        normalised_estimators = []
        for name in components:
            indicator_norm = float(
                np.sqrt(
                    max(
                        float(
                            np.sum(
                                np.abs(transport_results[name].cell_indicators)
                            )
                        ),
                        0.0,
                    )
                )
            )
            solution_norm = float(
                np.sqrt(
                    np.sum(
                        areas * np.square(transport_results[name].cell_averages)
                    )
                )
            )
            normalised_estimators.append(
                indicator_norm / max(solution_norm, np.finfo(float).tiny)
            )
        normalised_estimator = max(normalised_estimators, default=0.0)
        if (
            stagnation_ratio is not None
            and previous_normalised_estimator is not None
        ):
            stagnated = (
                normalised_estimator
                >= stagnation_ratio * previous_normalised_estimator
            )
            stagnation_count = stagnation_count + 1 if stagnated else 0
        previous_normalised_estimator = normalised_estimator
        timings["error_estimate_and_mark"] += perf_counter() - started
        reached_solve_limit = refinement_solves >= maximum_solves
        no_cells_marked = not np.any(marked)
        refinement_stagnated = (
            stagnation_ratio is not None
            and refinement_solves >= minimum_solves
            and stagnation_count >= stagnation_patience
        )
        # A requested physical step must retain its genuine terminal adaptive
        # state even when the stopping criterion is reached before one of the
        # requested levels.  Snapshot selection controls output only; it must
        # never turn a valid physical solve into a failed timestep.
        retain_terminal = bool(snapshot_levels) and (
            reached_solve_limit or no_cells_marked or refinement_stagnated
        )
        snapshot_retained = refinement_level in snapshot_levels or retain_terminal
        if snapshot_retained:
            started = perf_counter()
            retain_snapshot(
                refinement_level,
                refinement_solves,
                triangles,
                marked,
            )
            timings["snapshot_copy"] += perf_counter() - started
        if reached_solve_limit:
            break
        if no_cells_marked:
            break
        if refinement_stagnated:
            break
        started = perf_counter()
        if nested_previous_transfer or incremental_exact_previous_transfer:
            refined_domain, parent_cell = refine_with_parent(domain, marked)
        else:
            refined_domain = refine_marked_triangles(domain, marked)
            parent_cell = None
        timings["mesh_refinement"] += perf_counter() - started
        if triangles_of(refined_domain).shape[0] == triangles.shape[0]:
            if snapshot_levels and not snapshot_retained:
                started = perf_counter()
                retain_snapshot(
                    refinement_level,
                    refinement_solves,
                    triangles,
                    marked,
                )
                timings["snapshot_copy"] += perf_counter() - started
            break
        f2a_restricted = False
        if full_exact_previous_transfer:
            child_triangles = triangles_of(refined_domain)
            started = perf_counter()
            final_plan = conservative_remap_plan_2d(
                child_triangles,
                chemistry_x_edges,
                chemistry_y_edges,
            )
            child_previous = final_plan.rectangles_to_triangles(previous_stack)
            child_older = (
                None
                if older_stack is None
                else final_plan.rectangles_to_triangles(older_stack)
            )
            timings["chemistry_to_transport_transfer"] += perf_counter() - started
            previous_on_transport = child_previous
            older_on_transport = child_older
        elif (
            f2a_history is not None
            and current_complete_history is not None
            and parent_cell is not None
            and (nested_previous_transfer or incremental_exact_previous_transfer)
        ):
            # F2A (P2): exact restriction of the parent DG1 history to the children.
            (
                current_complete_history,
                older_complete_history,
                previous_on_transport,
                older_on_transport,
            ) = f2a_history.restrict(
                current_complete_history,
                older_complete_history,
                domain,
                refined_domain,
                parent_cell,
            )
            f2a_restricted = True
        elif nested_previous_transfer or incremental_exact_previous_transfer:
            if parent_cell is None or np.any(parent_cell < 0):
                raise RuntimeError("Missing parent-cell map for nested transfer")
            parent_inventory = previous_on_transport @ triangle_areas(triangles)
            older_parent_inventory = (
                None
                if older_on_transport is None
                else older_on_transport @ triangle_areas(triangles)
            )
            child_triangles = triangles_of(refined_domain)
            child_previous = previous_on_transport[:, parent_cell]
            child_older = (
                None
                if older_on_transport is None
                else older_on_transport[:, parent_cell]
            )
            if incremental_exact_previous_transfer:
                parent_plan = (
                    conservative_remap_plan_2d(
                        triangles,
                        chemistry_x_edges,
                        chemistry_y_edges,
                    )
                    if incremental_exact_transfer_plan
                    else None
                )
                parent_counts = np.bincount(
                    parent_cell, minlength=triangles.shape[0]
                )
                new_children = parent_counts[parent_cell] > 1
                started = perf_counter()
                child_previous[:, new_children] = rectangles_to_triangles_partial(
                    chemistry_x_edges,
                    chemistry_y_edges,
                    previous_stack,
                    child_triangles[new_children],
                )
                if child_older is not None and older_stack is not None:
                    child_older[:, new_children] = rectangles_to_triangles_partial(
                        chemistry_x_edges,
                        chemistry_y_edges,
                        older_stack,
                        child_triangles[new_children],
                    )
                timings["chemistry_to_transport_transfer"] += (
                    perf_counter() - started
                )
                if parent_plan is not None:
                    started = perf_counter()
                    refine_conservative_remap_plan_2d(
                        parent_plan,
                        parent_cell,
                        child_triangles,
                        chemistry_x_edges,
                        chemistry_y_edges,
                    )
                    timings["chemistry_to_transport_transfer"] += (
                        perf_counter() - started
                    )
            child_inventory = child_previous @ triangle_areas(child_triangles)
            absolute_error = np.abs(child_inventory - parent_inventory)
            inventory_scale = np.maximum.reduce(
                (
                    np.abs(parent_inventory),
                    np.abs(child_inventory),
                    np.full(parent_inventory.shape, 1.0e-14),
                )
            )
            relative_error = float(np.max(absolute_error / inventory_scale))
            if relative_error > nested_transfer_tolerance:
                raise RuntimeError(
                    "Parent-to-child history transfer is not conservative: "
                    f"relative error={relative_error:.6e}, "
                    f"absolute error={float(np.max(absolute_error)):.6e}"
                )
            if child_older is not None and older_parent_inventory is not None:
                older_child_inventory = child_older @ triangle_areas(child_triangles)
                older_absolute_error = np.abs(
                    older_child_inventory - older_parent_inventory
                )
                older_inventory_scale = np.maximum.reduce(
                    (
                        np.abs(older_parent_inventory),
                        np.abs(older_child_inventory),
                        np.full(older_parent_inventory.shape, 1.0e-14),
                    )
                )
                older_relative_error = float(
                    np.max(older_absolute_error / older_inventory_scale)
                )
                if older_relative_error > nested_transfer_tolerance:
                    raise RuntimeError(
                        "Older parent-to-child history transfer is not conservative: "
                        f"relative error={older_relative_error:.6e}, "
                        f"absolute error={float(np.max(older_absolute_error)):.6e}"
                    )
            previous_on_transport = child_previous
            older_on_transport = child_older
        else:
            previous_on_transport = None
            older_on_transport = None
        if not f2a_restricted:
            current_complete_history = None
            older_complete_history = None
        domain = refined_domain
        mesh_updates += 1

    if not transport_results:
        raise RuntimeError("No 2D adaptive transport solve was completed")
    return (
        domain,
        transport_results,
        refinement_solves,
        mesh_updates,
        tuple(snapshots),
        previous_on_transport.copy(),
        previous_history_used,
        older_history_used,
        timings,
        work_counts,
    )



def _rectangle_inventory(values: np.ndarray, areas: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64) * areas))


def _triangle_inventory(values: np.ndarray, triangles: np.ndarray) -> float:
    return float(
        np.sum(
            np.asarray(values, dtype=np.float64)
            * triangle_areas(np.asarray(triangles, dtype=np.float64))
        )
    )


def _process_peak_rss_bytes() -> int:
    """Return the process high-water RSS with platform units normalized."""

    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError):
        return -1
    # macOS reports bytes; Linux and the BSD-derived Python documentation use
    # KiB on the platforms where this project may also be run.
    return int(value if sys.platform == "darwin" else value * 1024.0)


def _run_adaptive_loop_2d(
    settings: LoopSettings2D,
    chemistry: ChemistryCallbacks2D,
    *,
    components: Sequence[str],
    model_floor: float,
    steps: int | None = None,
    maximum_refinement_solves: int | None = None,
    start_step: int = 0,
    chemistry_replay_inputs: Sequence[Mapping[str, np.ndarray]] = (),
    snapshot_selection: Mapping[int, Sequence[int]] | None = None,
    step_callback: Callable[[AdaptiveStep2D], None] | None = None,
    retain_all_steps: bool = True,
    progress: Callable[[AdaptiveStep2D], None] | None = None,
    transport_policy: ReactiveTransportPolicy2D | None = None,
    restart_transport_triangles: np.ndarray | None = None,
    reaction_increment_history: bool = False,
    restart_reaction_history: dict[str, np.ndarray] | None = None,
    restart_older_reaction_history: dict[str, np.ndarray] | None = None,
    f2a_options: F2AOptions | None = None,
    f2a_history: F2AHistory | None = None,
    fast_path: FastPath | None = None,
) -> AdaptiveRun2D:
    """Implementation of :func:`run_adaptive_loop_2d` (see there)."""

    case = settings
    configured_time_scheme = case.time_scheme
    requested_steps = case.steps if steps is None else int(steps)
    if not 1 <= requested_steps <= case.steps:
        raise ValueError(
            f"Steps must lie between 1 and the configured maximum {case.steps}"
        )
    if not 0 <= start_step <= requested_steps:
        raise ValueError("start_step must lie between zero and the requested steps")
    if len(chemistry_replay_inputs) != start_step:
        raise ValueError(
            "chemistry_replay_inputs must contain one entry for each completed step"
        )
    if reaction_increment_history and start_step > 0:
        if restart_reaction_history is None or restart_older_reaction_history is None:
            raise ValueError(
                "A resumed DG1 reaction-history run requires both saved BDF2 histories"
            )
    elif restart_reaction_history is not None or restart_older_reaction_history is not None:
        raise ValueError("Saved DG1 histories require reaction_increment_history")
    if case.adaptivity.maximum_refinements < 1:
        raise ValueError("Adaptivity must be enabled with at least one solve level")

    maximum_solves = (
        (
            transport_policy.maximum_solves
            if transport_policy is not None
            else case.adaptivity.maximum_refinements
        )
        if maximum_refinement_solves is None
        else int(maximum_refinement_solves)
    )
    if not 1 <= maximum_solves <= case.adaptivity.maximum_refinements:
        raise ValueError(
            "maximum_refinement_solves must lie between 1 and the configured "
            f"maximum {case.adaptivity.maximum_refinements}"
        )

    normalized_snapshots: dict[int, frozenset[int]] = {}
    for physical_step, levels in (snapshot_selection or {}).items():
        physical_step = int(physical_step)
        if not 1 <= physical_step <= requested_steps:
            raise ValueError(
                f"Snapshot physical step {physical_step} is outside the run"
            )
        level_set = frozenset(int(level) for level in levels)
        invalid = sorted(
            level for level in level_set if not 0 <= level < maximum_solves
        )
        if invalid:
            raise ValueError(
                f"Snapshot levels {invalid} exceed the solve range 0..{maximum_solves - 1}"
            )
        normalized_snapshots[physical_step] = level_set

    nx, ny = case.chemistry_cells
    # Transport and chemistry resolutions are intentionally independent.  The
    # Cartesian PHREEQC grid is defined by [chemistry].cells, while the initial
    # transport mesh is defined by [adaptivity].initial_cells_2d.  Historical
    # reproduction cases happen to use equal counts, but validation studies do
    # not need to spend extra PHREEQC work when only transport is refined.
    x_edges = np.linspace(case.lower[0], case.upper[0], nx + 1)
    y_edges = np.linspace(case.lower[1], case.upper[1], ny + 1)
    chemistry_shape = (ny, nx)
    current = {
        name: chemistry.snapshot().concentrations[name].reshape(chemistry_shape).copy()
        for name in components
    }
    older_current: dict[str, np.ndarray] | None = None
    boundary = {name: float(chemistry.boundary[name]) for name in components}
    records: list[AdaptiveStep2D] = []
    last_record: AdaptiveStep2D | None = None

    for replay_step, replay_input in enumerate(chemistry_replay_inputs, start=1):
        older_current = {name: current[name].copy() for name in components}
        replayed = chemistry.react(replay_input, time_step=case.time_step)
        current = {
            name: replayed.concentrations[name].reshape(chemistry_shape).copy()
            for name in components
        }
        if progress is not None:
            print(
                f"Restored PHREEQC checkpoint state {replay_step}/{start_step}",
                flush=True,
            )
    if configured_time_scheme == "bdf2" and start_step > 0 and older_current is None:
        raise RuntimeError("BDF2 restart did not reconstruct the older history")

    persistent_domain = None
    current_transport_history = (
        None
        if restart_reaction_history is None
        else {
            name: np.asarray(restart_reaction_history[name], dtype=np.float64).copy()
            for name in components
        }
    )
    older_transport_history = (
        None
        if restart_older_reaction_history is None
        else {
            name: np.asarray(
                restart_older_reaction_history[name], dtype=np.float64
            ).copy()
            for name in components
        }
    )
    triangles_of = mesh_triangles if fast_path is None else fast_mesh_triangles
    rebuild_interval = (
        1 if transport_policy is None else transport_policy.mesh_rebuild_interval
    )
    if (
        transport_policy is not None
        and transport_policy.persistent_mesh_enabled
        and start_step > 0
        and (
            start_step % rebuild_interval != 0
            # F2A (P3): a resume at a rebuild step also needs the last committed mesh,
            # so that the DG1 histories can be projected onto the rebuilt mesh.
            or (
                f2a_options is not None
                and f2a_options.rebuild_mesh_on_restart
                and reaction_increment_history
            )
        )
    ):
        if restart_transport_triangles is None:
            raise RuntimeError(
                "A mid-block persistent-mesh restart requires the last committed "
                "transport triangles"
            )
        persistent_domain = create_mesh_from_triangles_2d(
            restart_transport_triangles
        )
        if reaction_increment_history:
            if current_transport_history is None or older_transport_history is None:
                raise RuntimeError("Restarted R76 mesh is missing its DG1 histories")
            restart_permutation = _triangle_restart_permutation(
                restart_transport_triangles,
                mesh_triangles(persistent_domain),
            )
            current_transport_history = {
                name: current_transport_history[name][restart_permutation].copy()
                for name in components
            }
            older_transport_history = {
                name: older_transport_history[name][restart_permutation].copy()
                for name in components
            }

    for step_index in range(start_step, requested_steps):
        physical_step_started = perf_counter()
        transfer_cache_before = transfer_plan_cache_info()
        chemistry_before_reaction = chemistry.snapshot()
        previous_chemistry = {
            name: current[name].copy() for name in components
        }
        step_scheme = (
            "backward_euler"
            if configured_time_scheme == "bdf2" and step_index == 0
            else configured_time_scheme
        )
        if step_scheme == "bdf2" and older_current is None:
            raise RuntimeError("BDF2 requires the previous reacted chemistry state")
        older_chemistry = (
            None
            if step_scheme == "backward_euler"
            else {
                name: older_current[name].copy()  # type: ignore[index]
                for name in components
            }
        )
        predictive_seconds = 0.0
        initial_domain = None
        predictive_mesh_rebuilt = False
        mesh_age_steps = 0
        step_maximum_solves = maximum_solves
        if transport_policy is not None:
            predictive_mesh_rebuilt = persistent_mesh_rebuild_required(
                step_index,
                rebuild_interval,
                has_persistent_mesh=persistent_domain is not None,
            )
            if predictive_mesh_rebuilt:
                started = perf_counter()
                initial_domain = create_predictive_reactive_mesh_2d(
                    fields={
                        **previous_chemistry,
                        **(
                            {}
                            if chemistry.seed_fields is None
                            else chemistry.seed_fields(
                                chemistry_before_reaction, chemistry_shape
                            )
                        ),
                    },
                    x_edges=x_edges,
                    y_edges=y_edges,
                    lower=(case.lower[0], case.lower[1]),
                    upper=(case.upper[0], case.upper[1]),
                    initial_cells=case.adaptivity.initial_cells_2d,
                    velocity=(
                        case.velocity[0],
                        case.velocity[1],
                    ),
                    time_step=case.time_step,
                    left_inlet_y_min=(
                        0.4
                        if case.left_inlet_y_min is None
                        else case.left_inlet_y_min
                    ),
                    top_inlet_x_max=(
                        0.1
                        if case.top_inlet_x_max is None
                        else case.top_inlet_x_max
                    ),
                    policy=transport_policy,
                )
                predictive_seconds = perf_counter() - started
            else:
                initial_domain = persistent_domain
                step_maximum_solves = min(
                    maximum_solves,
                    transport_policy.reused_mesh_maximum_solves,
                )
                mesh_age_steps = step_index % rebuild_interval
        if not reaction_increment_history:
            step_histories = (None, None)
        elif f2a_history is not None and f2a_options.project_on_rebuild:
            # F2A (P1): project the DG1 histories onto the rebuilt mesh.
            step_histories = f2a_history.histories(
                predictive_mesh_rebuilt,
                current_transport_history,
                older_transport_history,
                persistent_domain,
                initial_domain,
            )
        elif predictive_mesh_rebuilt:
            step_histories = (None, None)
        else:
            step_histories = (current_transport_history, older_transport_history)
        (
            domain,
            transport,
            refinement_solves,
            mesh_updates,
            refinement_history,
            previous_on_final_transport_mesh,
            previous_history_used,
            older_history_used,
            timings,
            work_counts,
        ) = solve_one_adaptive_transport_step_2d(
                components=components,
                previous_chemistry=previous_chemistry,
                older_chemistry=older_chemistry,
                chemistry_x_edges=x_edges,
                chemistry_y_edges=y_edges,
                boundary=boundary,
                lower=(case.lower[0], case.lower[1]),
                upper=(case.upper[0], case.upper[1]),
                time_step=case.time_step,
                diffusion=case.diffusion,
                velocity=(case.velocity[0], case.velocity[1]),
                left_inlet_y_min=(
                    0.4
                    if case.left_inlet_y_min is None
                    else case.left_inlet_y_min
                ),
                top_inlet_x_max=(
                    0.1
                    if case.top_inlet_x_max is None
                    else case.top_inlet_x_max
                ),
                degree=case.degree,
                adaptivity=case.adaptivity,
                maximum_solves=step_maximum_solves,
                initial_domain=initial_domain,
                incremental_exact_previous_transfer=(
                    transport_policy is not None
                    and transport_policy.incremental_exact_transfer
                ),
                incremental_exact_transfer_plan=(
                    transport_policy is not None
                    and transport_policy.incremental_exact_transfer
                ),
                time_scheme=step_scheme,
                history_degree=1 if reaction_increment_history else 0,
                previous_history_cell_values=step_histories[0],
                older_history_cell_values=step_histories[1],
                snapshot_levels=normalized_snapshots.get(
                    step_index + 1, frozenset()
                ),
                f2a_history=(
                    f2a_history
                    if f2a_options is not None and f2a_options.restrict_on_refinement
                    else None
                ),
                fast=fast_path,
            )
        if transport_policy is not None and transport_policy.persistent_mesh_enabled:
            persistent_domain = domain
        timings["predictive_mesh"] = predictive_seconds
        work_counts.update(
            {
                "predictive_mesh_rebuilt": int(predictive_mesh_rebuilt),
                "persistent_mesh_reused": int(
                    transport_policy is not None
                    and transport_policy.persistent_mesh_enabled
                    and not predictive_mesh_rebuilt
                ),
                "persistent_mesh_age_steps": int(mesh_age_steps),
                "configured_mesh_rebuild_interval": int(rebuild_interval),
            }
        )
        requested_levels = normalized_snapshots.get(step_index + 1, frozenset())
        saved_levels = {
            snapshot.refinement_level for snapshot in refinement_history
        }
        missing_levels = sorted(requested_levels.difference(saved_levels))
        if missing_levels:
            terminal_level = refinement_solves - 1
            warnings.warn(
                f"Physical step {step_index + 1} stopped after "
                f"{refinement_solves} adaptive solves at refinement level "
                f"{terminal_level}, before requested snapshot levels "
                f"{missing_levels}. The genuine terminal level was retained and "
                "the valid physical step will be committed.",
                RuntimeWarning,
                stacklevel=2,
            )
        triangles = triangles_of(domain)
        transported_stack = np.stack(
            [transport[name].cell_averages for name in components]
        )
        started = perf_counter()
        transported_on_chemistry = triangles_to_rectangles(
            triangles,
            transported_stack,
            x_edges,
            y_edges,
        )
        timings["transport_to_chemistry_transfer"] = perf_counter() - started
        chemistry_cell_areas = np.outer(np.diff(y_edges), np.diff(x_edges))
        if transport_policy is not None:
            source_inventory = transported_stack @ triangle_areas(triangles)
            target_inventory = np.asarray(
                [
                    _rectangle_inventory(values, chemistry_cell_areas)
                    for values in transported_on_chemistry
                ]
            )
            scale = np.maximum(
                np.abs(source_inventory), np.full(source_inventory.shape, 1.0e-14)
            )
            relative_transfer_error = float(
                np.max(np.abs(target_inventory - source_inventory) / scale)
            )
            if relative_transfer_error > transport_policy.final_transfer_tolerance:
                raise RuntimeError(
                    "Final transport-to-chemistry transfer is not conservative: "
                    f"relative error={relative_transfer_error:.6e}"
                )
        chemistry_input: dict[str, np.ndarray] = {}
        negative_corrections: dict[str, NegativeTransportCorrection2D] = {}
        started = perf_counter()
        for component_index, name in enumerate(components):
            values = transported_on_chemistry[component_index]
            chemistry_input[name], negative_corrections[name] = (
                prepare_chemistry_component_2d(
                    name,
                    values,
                    chemistry_cell_areas,
                    model_floor=model_floor,
                    negative_concentration_policy=(
                        case.negative_concentration_policy
                    ),
                )
            )
        timings["positivity_correction"] = perf_counter() - started

        started = perf_counter()
        reacted = chemistry.react(chemistry_input, time_step=case.time_step)
        timings["phreeqc_reaction"] = perf_counter() - started
        older_current = previous_chemistry
        current = {
            name: reacted.concentrations[name].reshape(chemistry_shape).copy()
            for name in components
        }
        reaction_history_linf = {name: 0.0 for name in components}
        reaction_history_inventory_residual = {
            name: 0.0 for name in components
        }
        if reaction_increment_history:
            if previous_history_used is None:
                raise RuntimeError("DG1 history was not returned by the transport solve")
            started = perf_counter()
            plan = conservative_remap_plan_2d(triangles, x_edges, y_edges)
            reaction_increment_rectangles = np.stack(
                [
                    current[name] - transported_on_chemistry[component_index]
                    for component_index, name in enumerate(components)
                ]
            )
            reaction_increment_triangles = plan.rectangles_to_triangles(
                reaction_increment_rectangles
            )
            lifted_history: dict[str, np.ndarray] = {}
            lifted_averages = np.empty_like(transported_stack)
            for component_index, name in enumerate(components):
                base = transport[name].history_cell_values
                if base is None:
                    raise RuntimeError("Transport did not retain its DG1 history")
                lifted = (
                    base
                    + reaction_increment_triangles[component_index, :, None]
                )
                lifted_history[name] = lifted
                lifted_averages[component_index] = np.mean(lifted, axis=1)
            lifted_on_chemistry = plan.triangles_to_rectangles(lifted_averages)
            for component_index, name in enumerate(components):
                difference = lifted_on_chemistry[component_index] - current[name]
                reaction_history_linf[name] = float(np.max(np.abs(difference)))
                reaction_history_inventory_residual[name] = float(
                    np.sum(difference * chemistry_cell_areas)
                )
            older_transport_history = {
                name: previous_history_used[name].copy() for name in components
            }
            current_transport_history = lifted_history
            timings["reaction_increment_history_lift"] = perf_counter() - started
        else:
            timings["reaction_increment_history_lift"] = 0.0
        started = perf_counter()
        mass_balance_diagnostics: dict[str, dict[str, float]] = {}
        for component_index, name in enumerate(components):
            chemistry_before = _rectangle_inventory(
                previous_chemistry[name], chemistry_cell_areas
            )
            transport_initial = _triangle_inventory(
                previous_on_final_transport_mesh[component_index], triangles
            )
            transport_after = _triangle_inventory(
                transported_stack[component_index], triangles
            )
            chemistry_raw = _rectangle_inventory(
                transported_on_chemistry[component_index], chemistry_cell_areas
            )
            chemistry_corrected = _rectangle_inventory(
                chemistry_input[name], chemistry_cell_areas
            )
            chemistry_after = _rectangle_inventory(
                current[name], chemistry_cell_areas
            )
            mass_balance_diagnostics[name] = {
                "chemistry_before_transport_mol_L_m2": chemistry_before,
                "transport_initial_mol_L_m2": transport_initial,
                "chemistry_to_transport_transfer_residual_mol_L_m2": (
                    transport_initial - chemistry_before
                ),
                "transport_after_solve_mol_L_m2": transport_after,
                "transport_inventory_change_mol_L_m2": (
                    transport_after - transport_initial
                ),
                "chemistry_before_positivity_mol_L_m2": chemistry_raw,
                "transport_to_chemistry_transfer_residual_mol_L_m2": (
                    chemistry_raw - transport_after
                ),
                "chemistry_after_positivity_mol_L_m2": chemistry_corrected,
                "positivity_added_mol_L_m2": chemistry_corrected - chemistry_raw,
                "chemistry_after_reaction_aqueous_mol_L_m2": chemistry_after,
                "aqueous_reaction_change_mol_L_m2": (
                    chemistry_after - chemistry_corrected
                ),
                "reaction_history_lift_linf_mol_L": reaction_history_linf[name],
                "reaction_history_lift_inventory_residual_mol_L_m2": (
                    reaction_history_inventory_residual[name]
                ),
                "solid_before_reaction_mol_per_kgw_m2": (
                    0.0
                    if chemistry.solid_inventory is None
                    else chemistry.solid_inventory(
                        name,
                        chemistry_before_reaction,
                        chemistry_shape,
                        chemistry_cell_areas,
                    )
                ),
                "solid_after_reaction_mol_per_kgw_m2": (
                    0.0
                    if chemistry.solid_inventory is None
                    else chemistry.solid_inventory(
                        name,
                        reacted,
                        chemistry_shape,
                        chemistry_cell_areas,
                    )
                ),
            }
        timings["mass_diagnostics"] = perf_counter() - started

        timings["physical_step_total"] = perf_counter() - physical_step_started
        work_counts.update(
            {
                "final_triangles": int(triangles.shape[0]),
                "final_trial_dofs": int(
                    transport[components[0]].node_coordinates.shape[0]
                ),
                "final_test_dofs": 3 * int(triangles.shape[0]),
                "chemistry_cells": int(nx * ny),
                "phreeqc_module_calls": 1,
                "phreeqc_cell_reactions": int(nx * ny),
                "refinement_solves": int(refinement_solves),
                "mesh_updates": int(mesh_updates),
                "process_peak_rss_bytes": _process_peak_rss_bytes(),
            }
        )
        transfer_cache_after = transfer_plan_cache_info()
        work_counts.update(
            {
                "transfer_plan_cache_hits": (
                    transfer_cache_after["hits"] - transfer_cache_before["hits"]
                ),
                "transfer_plan_cache_misses": (
                    transfer_cache_after["misses"]
                    - transfer_cache_before["misses"]
                ),
                "transfer_plan_cache_entries": transfer_cache_after["entries"],
                "transfer_plan_cache_bytes": transfer_cache_after["bytes"],
            }
        )
        record = AdaptiveStep2D(
            step=step_index,
            time=(step_index + 1) * case.time_step,
            refinement_solves=refinement_solves,
            mesh_updates=mesh_updates,
            triangles=triangles.copy(),
            transported_cell_averages={
                name: transport[name].cell_averages.copy() for name in components
            },
            transported_node_coordinates={
                name: transport[name].node_coordinates.copy() for name in components
            },
            transported_nodal_values={
                name: transport[name].nodal_values.copy() for name in components
            },
            cell_indicators={
                name: transport[name].cell_indicators.copy() for name in components
            },
            chemistry_result=reacted,
            negative_transport_corrections=negative_corrections,
            chemistry_input_concentrations={
                name: chemistry_input[name].copy() for name in components
            },
            chemistry_output_concentrations={
                name: reacted.concentrations[name].reshape(chemistry_shape).copy()
                for name in components
            },
            refinement_history=refinement_history,
            timings_seconds=timings,
            work_counts=work_counts,
            mass_balance_diagnostics=mass_balance_diagnostics,
            reaction_history_cell_values=(
                None
                if current_transport_history is None
                else {
                    name: current_transport_history[name].copy()
                    for name in components
                }
            ),
            older_reaction_history_cell_values=(
                None
                if older_transport_history is None
                else {
                    name: older_transport_history[name].copy()
                    for name in components
                }
            ),
        )
        last_record = record
        if retain_all_steps:
            records.append(record)
        if step_callback is not None:
            step_callback(record)
        if progress is not None:
            progress(record)

    if not retain_all_steps and last_record is not None:
        records.append(last_record)

    statistics = {}
    if f2a_history is not None:
        statistics["f2a"] = dict(f2a_history.stats)
    if fast_path is not None:
        statistics["fast"] = dict(fast_path.stats)
    return AdaptiveRun2D(
        steps=tuple(records),
        chemistry_x_edges=x_edges,
        chemistry_y_edges=y_edges,
        boundary_concentrations=boundary,
        statistics=statistics or None,
    )


def run_adaptive_loop_2d(
    settings: LoopSettings2D,
    chemistry: ChemistryCallbacks2D,
    *,
    components: Sequence[str],
    model_floor: float,
    steps: int | None = None,
    maximum_refinement_solves: int | None = None,
    start_step: int = 0,
    chemistry_replay_inputs: Sequence[Mapping[str, np.ndarray]] = (),
    snapshot_selection: Mapping[int, Sequence[int]] | None = None,
    step_callback: Callable[[AdaptiveStep2D], None] | None = None,
    retain_all_steps: bool = True,
    progress: Callable[[AdaptiveStep2D], None] | None = None,
    transport_policy: ReactiveTransportPolicy2D | None = None,
    restart_transport_triangles: np.ndarray | None = None,
    reaction_increment_history: bool = False,
    restart_reaction_history: dict[str, np.ndarray] | None = None,
    restart_older_reaction_history: dict[str, np.ndarray] | None = None,
    history_policy: str = "rebuild",
    f2a_options: F2AOptions | None = None,
    fast: bool | FastOptions = False,
) -> AdaptiveRun2D:
    """Run the adaptive 2D transport/chemistry loop.

    By default the transport mesh is restarted from 5x5 at every physical
    time step, matching the legacy control flow. A persistent transport policy
    may instead reuse an accepted mesh inside deterministic rebuild blocks.
    The chemistry grid always remains fixed.

    ``history_policy`` ("rebuild", the default, or "f2a") selects what happens to the
    DG1 reaction history (``reaction_increment_history=True``) when the persistent mesh
    is rebuilt or refined: "rebuild" restarts it from chemistry-grid averages; "f2a" is the
    history-preserving mesh rebuild of :mod:`adaptdg.f2a` (``f2a_options`` switches its three
    changes P1, P2 and P3 individually; all on by default). ``fast`` (True or a
    :class:`adaptdg.fastadap.FastOptions`) selects the cost-only fast path of
    :mod:`adaptdg.fastadap`; results agree with the plain path to round-off.
    ``AdaptiveRun2D.statistics`` reports the counters of both when used.
    """

    if history_policy not in {"rebuild", "f2a"}:
        raise ValueError('history_policy must be "rebuild" or "f2a"')
    if history_policy == "rebuild" and f2a_options is not None:
        raise ValueError('f2a_options requires history_policy="f2a"')
    fast_path = None
    if fast:
        fast_path = FastPath(fast if isinstance(fast, FastOptions) else None)
    f2a_history = None
    if history_policy == "f2a":
        f2a_options = F2AOptions() if f2a_options is None else f2a_options
        f2a_history = F2AHistory(
            components,
            optimised=fast_path is not None and fast_path.options.plan_reuse,
        )
    arguments = dict(
        components=components,
        model_floor=model_floor,
        steps=steps,
        maximum_refinement_solves=maximum_refinement_solves,
        start_step=start_step,
        chemistry_replay_inputs=chemistry_replay_inputs,
        snapshot_selection=snapshot_selection,
        step_callback=step_callback,
        retain_all_steps=retain_all_steps,
        progress=progress,
        transport_policy=transport_policy,
        restart_transport_triangles=restart_transport_triangles,
        reaction_increment_history=reaction_increment_history,
        restart_reaction_history=restart_reaction_history,
        restart_older_reaction_history=restart_older_reaction_history,
        f2a_options=f2a_options,
        f2a_history=f2a_history,
        fast_path=fast_path,
    )
    if fast_path is None:
        return _run_adaptive_loop_2d(settings, chemistry, **arguments)
    with use_plan_builder(fast_path.plan_create):
        return _run_adaptive_loop_2d(settings, chemistry, **arguments)

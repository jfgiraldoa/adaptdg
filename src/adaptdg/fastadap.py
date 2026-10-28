"""Fast path of the 2D adaptive loop: cost-only changes, the method and its numbers are unchanged.

Extracted from ASRTM-PHREEQC (``methods/front2d/fastadap.py``). There the changes were
installed at run time by replacing functions of the repository; here they are selected by
``run_adaptive_loop_2d(..., fast=True)`` (a :class:`FastPath` object per run, no global
patching).

1. Solver reuse. :func:`FastPath.solver_factory` returns the previous solver when the mesh
   object, time scheme, history degree and physical parameters are the same. On a fixed mesh
   the operator is identical, so the MUMPS factorisation is reused on the persistent-mesh
   steps (7 of 8 with rebuild interval 8). The cache holds one solver.
2. Vectorised per-cell work. History injection/extraction, cell averages and indicator
   extraction use ``dofmap.list`` instead of Python loops over cells.
3. Forms compiled once per solver. The residual and estimator forms act on persistent
   coefficient functions filled through the collapse maps, and the DG0 indicator space is
   created once.
4. Vectorised transfer plans. ``ConservativeRemapPlan2D.create`` is replaced (through
   ``adaptdg.transfer_2d.use_plan_builder``) by a builder that clips every candidate
   (triangle, rectangle) pair with the same Sutherland-Hodgman arithmetic, in arrays; same
   pairs in the same order, overlap areas equal to round-off (the shoelace sum order may
   differ in the last bits). With ``plan_reuse`` the rows of the previous plan are copied
   for geometrically identical triangles.
5. Multi-RHS back-solve (``batch_rhs``, used with F2A): the back-solves of all components
   on a mesh are one MUMPS call.
6. Vectorised ``mesh_triangles`` and a refinement without the MPI-communicator leak of
   dolfinx 0.11 (``partitioner=None``, serial; same mesh and parent map).

Results agree with the plain path to round-off, not bit for bit (1 and 4).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from adaptdg.transport_2d import (
    ResidualMinimizationBackwardEuler2D,
    StabilizedTransportStep2D,
)

_CHUNK = 200_000  # candidate pairs clipped per vectorised batch
_SLOTS = 7  # a triangle clipped by four half-planes has <= 7 vertices


@dataclass(frozen=True)
class FastOptions:
    """Optional parts of the fast path (all on, as in the paper driver)."""

    plan_reuse: bool = True  # copy plan rows / history values of identical triangles
    batch_rhs: bool = True  # one multi-RHS back-solve per mesh (active with F2A)


class FastResidualMinimization2D(ResidualMinimizationBackwardEuler2D):
    """Same forms and solver; vectorised bookkeeping, persistent coefficients."""

    batch_rhs = False  # set by FastPath.solver_factory
    stats: dict = {}  # replaced by the FastPath statistics

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fem, ufl = self._fem, self._ufl
        self._prev_dofs = np.asarray(self._previous_space.dofmap.list, dtype=np.int64)
        if self._prev_dofs.shape[1] == 3:
            xy = self._previous_space.tabulate_dof_coordinates()[:, :2]
            order = np.lexsort((xy[self._prev_dofs, 1], xy[self._prev_dofs, 0]), axis=-1)
            self._canon = np.take_along_axis(self._prev_dofs, order, axis=1)
        else:
            self._canon = self._prev_dofs
        space0, map0 = self._mixed_space.sub(0).collapse()
        space1, map1 = self._mixed_space.sub(1).collapse()
        self._map0 = np.asarray(map0, dtype=np.int64).ravel()
        self._map1 = np.asarray(map1, dtype=np.int64).ravel()
        self._residual = fem.Function(space0)
        self._concentration = fem.Function(space1)
        if self._map0.size != self._residual.x.array.size or \
                self._map1.size != self._concentration.x.array.size:
            raise RuntimeError("Unexpected collapse-map layout")
        self._conc_dofs = np.asarray(space1.dofmap.list, dtype=np.int64)
        self._node_xy = space1.tabulate_dof_coordinates()[:, :2].copy()
        self._history_function = (fem.Function(self._previous_space)
                                  if self._history_degree == 1 else None)
        indicator_space = fem.functionspace(self._domain, ("DG", 0))
        self._dg0_dofs = np.asarray(indicator_space.dofmap.list, dtype=np.int64)[:, 0]
        residual = self._residual
        self._residual_l2_form = fem.form(residual * residual * self._dx)
        # Same expression, term order and measures as the plain step().
        indicator_test = ufl.TestFunction(indicator_space)
        beta_normal = ufl.dot(self._beta, self._normal)
        beta_normal_plus = ufl.dot(self._beta("+"), self._normal("+"))
        indicator_form = residual * residual * indicator_test * self._dx + self._dt * (
            ufl.inner(self._diffusivity * ufl.grad(residual), ufl.grad(residual))
            * indicator_test * self._dx
            + 0.5 * self._diffusivity * ufl.avg(self._eta)
            * ufl.inner(ufl.jump(residual, self._normal), ufl.jump(residual, self._normal))
            * ufl.avg(indicator_test) * self._dS
            + self._diffusivity * self._eta * residual * residual * indicator_test * self._ds
            + 0.5 * abs(beta_normal_plus) * ufl.jump(residual) * ufl.jump(residual)
            * ufl.avg(indicator_test) * self._dS
            + 0.5 * abs(beta_normal) * residual * residual * indicator_test * self._ds
            + self._cell_diameter * ufl.dot(self._beta, ufl.grad(residual))
            * ufl.dot(self._beta, ufl.grad(residual)) * indicator_test * self._dx
            + residual * residual * indicator_test * self._dx
        )
        self._indicator_form = fem.form(indicator_form)

    # ---- history bookkeeping (vectorised; DG dofs belong to one cell each)
    def set_history_cell_averages(self, values, older_values=None):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.cells,):
            raise ValueError(f"Expected {(self.cells,)} cell values, received {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("Cell averages must be finite")
        self._previous.x.array[self._prev_dofs] = array[:, None]
        self._previous.x.scatter_forward()
        if self._older is None:
            if older_values is not None:
                raise ValueError("Backward Euler accepts only one history state")
            return
        if older_values is None:
            raise ValueError("BDF2 requires the two preceding history states")
        older = np.asarray(older_values, dtype=np.float64)
        if older.shape != (self.cells,):
            raise ValueError(f"Expected {(self.cells,)} older values, received {older.shape}")
        if not np.all(np.isfinite(older)):
            raise ValueError("Older cell averages must be finite")
        self._older.x.array[self._prev_dofs] = older[:, None]
        self._older.x.scatter_forward()

    def set_history_cell_values(self, values, older_values=None):
        expected = (self.cells, 1 if self._history_degree == 0 else 3)

        def flatten(cell_values):
            array = np.asarray(cell_values, dtype=np.float64)
            if array.shape != expected or not np.all(np.isfinite(array)):
                raise ValueError(f"Expected finite cell history with shape {expected}")
            result = np.empty_like(self._previous.x.array)
            result[self._canon] = array
            return result

        current = flatten(values)
        older = None if older_values is None else flatten(older_values)
        self.set_history_dof_values(current, older)

    def history_from_cell_averages(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.cells,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Expected finite cell values with shape {(self.cells,)}")
        history = np.empty_like(self._previous.x.array)
        history[self._prev_dofs] = array[:, None]
        return history

    def cell_averages_from_history(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self._previous.x.array.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"Expected finite history values with shape "
                             f"{self._previous.x.array.shape}")
        return array[self._prev_dofs].mean(axis=1)

    def add_cell_average_increments(self, values, increments):
        result = np.asarray(values, dtype=np.float64).copy()
        delta = np.asarray(increments, dtype=np.float64)
        if result.shape != self._previous.x.array.shape or not np.all(np.isfinite(result)):
            raise ValueError("History values have an incompatible shape")
        if delta.shape != (self.cells,) or not np.all(np.isfinite(delta)):
            raise ValueError("Reaction increments have an incompatible shape")
        result[self._prev_dofs] += delta[:, None]
        return result

    def history_cell_values(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self._previous.x.array.shape or not np.all(np.isfinite(array)):
            raise ValueError("History values have an incompatible shape")
        return array[self._canon]

    # ---- multi-RHS back-solve of the component solves on one mesh
    def begin_batch(self, current, older, boundary):
        """Announce the histories and inflow values of the components that the driver is about
        to solve one after the other on this mesh (no effect unless ``batch_rhs``)."""
        self._batch = None
        if self.batch_rhs and current is not None and self._history_degree == 1:
            self._batch = {"cur": current, "old": older, "boundary": boundary, "calls": 0,
                           "cache": [], "filled": False}

    def _batched_solution(self, inflow_value):
        """Solution vector of the current component from the multi-RHS solve, or None.
        The cached RHS were assembled with the histories/inflow the driver injects; a hit
        requires the very same history arrays and inflow now, otherwise nothing is reused."""
        batch = self._batch
        prev = self._previous.x.array
        old = None if self._older is None else self._older.x.array
        for entry in batch["cache"]:
            if not entry["used"] and entry["inflow"] == float(inflow_value) \
                    and np.array_equal(entry["prev"], prev) \
                    and (old is None and entry["older"] is None
                         or old is not None and entry["older"] is not None
                         and np.array_equal(entry["older"], old)):
                entry["used"] = True
                return entry["x"]
        if batch["filled"]:
            return None
        batch["filled"] = True
        names = list(batch["cur"])[batch["calls"] - 1:]
        saved = (prev.copy(), None if old is None else old.copy(), float(self._inflow.value))
        columns = []
        for name in names:
            self.set_history_cell_values(batch["cur"][name],
                                         None if batch["old"] is None else batch["old"][name])
            self._inflow.value = batch["boundary"][name]
            self._problem.b.set(0.0)
            self._fem.petsc.assemble_vector(self._problem.b, self._problem.L)
            self._problem.b.ghostUpdate(addv=self._PETSc.InsertMode.ADD,
                                        mode=self._PETSc.ScatterMode.REVERSE)
            columns.append({"prev": self._previous.x.array.copy(),
                            "older": None if self._older is None else self._older.x.array.copy(),
                            "inflow": float(batch["boundary"][name]),
                            "b": np.array(self._problem.b.array_r), "used": False})
        self._previous.x.array[:] = saved[0]
        self._previous.x.scatter_forward()
        if saved[1] is not None:
            self._older.x.array[:] = saved[1]
            self._older.x.scatter_forward()
        self._inflow.value = saved[2]
        n, k = columns[0]["b"].size, len(columns)
        rhs = np.empty((n, k), order="F")
        sol = np.empty((n, k), order="F")
        for j, column in enumerate(columns):
            rhs[:, j] = column["b"]
        B = self._PETSc.Mat().createDense((n, k), array=rhs, comm=self._PETSc.COMM_SELF)
        X = self._PETSc.Mat().createDense((n, k), array=sol, comm=self._PETSc.COMM_SELF)
        self._problem.solver.matSolve(B, X)
        X.assemble()
        B.destroy()
        X.destroy()
        for j, column in enumerate(columns):
            column["x"] = sol[:, j].copy()
            del column["b"]
        batch["cache"] = columns
        self.stats["batched_systems"] = self.stats.get("batched_systems", 0) + 1
        self.stats["batched_rhs"] = self.stats.get("batched_rhs", 0) + k
        return self._batched_solution(inflow_value)

    # ---- one component solve (same solve path as the plain step)
    def step(self, *, inflow_value):
        if not np.isfinite(inflow_value):
            raise ValueError("The inlet concentration must be finite")
        self._inflow.value = inflow_value
        batch = getattr(self, "_batch", None)
        if batch is not None:
            batch["calls"] += 1
        if not self._matrix_assembled:
            mixed_solution = self._problem.solve()
            self._matrix_assembled = True
        else:
            cached = None if batch is None else self._batched_solution(inflow_value)
            if cached is None:
                self._problem.b.set(0.0)
                self._fem.petsc.assemble_vector(self._problem.b, self._problem.L)
                self._problem.b.ghostUpdate(addv=self._PETSc.InsertMode.ADD,
                                            mode=self._PETSc.ScatterMode.REVERSE)
                self._problem.solver.solve(self._problem.b, self._problem.x)
            else:
                self._problem.x.array[:] = cached
            self._problem.x.ghostUpdate(addv=self._PETSc.InsertMode.INSERT,
                                        mode=self._PETSc.ScatterMode.FORWARD)
            self._fem.petsc.assign(self._problem.x, self._problem.u)
            mixed_solution = self._problem.u
        mixed_solution.x.scatter_forward()
        mixed = mixed_solution.x.array
        self._residual.x.array[:] = mixed[self._map0]
        self._concentration.x.array[:] = mixed[self._map1]
        self._residual.x.scatter_forward()
        self._concentration.x.scatter_forward()
        concentration = self._concentration

        history_dof_values = None
        history_cell_values = None
        if self._history_degree == 1:
            self._history_function.interpolate(concentration)
            self._history_function.x.scatter_forward()
            history_dof_values = self._history_function.x.array.copy()
            history_cell_values = history_dof_values[self._canon]

        values = concentration.x.array
        cell_averages = values[self._conc_dofs].mean(axis=1)
        residual_squared = self._fem.assemble_scalar(self._residual_l2_form)
        residual_l2 = float(np.sqrt(self._comm.allreduce(residual_squared)))
        indicator_vector = self._fem.petsc.assemble_vector(self._indicator_form)
        cell_indicators = np.abs(np.asarray(indicator_vector.array)[self._dg0_dofs])
        indicator_vector.destroy()
        return StabilizedTransportStep2D(
            cell_averages=cell_averages,
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            integral=float(np.dot(cell_averages, self.cell_areas)),
            residual_l2=residual_l2,
            cell_indicators=cell_indicators,
            node_coordinates=self._node_xy.copy(),
            nodal_values=values.copy(),
            history_dof_values=history_dof_values,
            history_cell_values=history_cell_values,
        )


def _freeze(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_freeze(v) for v in value)
    return value


# ---------------------------------------------------------------- transfer plans
def _clip_axis(px, py, count, axis, bound, keep_greater):
    """Vectorised _clip_axis of transfer_2d on padded polygons (same arithmetic)."""
    m, k = px.shape
    pa = px if axis == 0 else py
    bound = np.broadcast_to(np.asarray(bound, dtype=np.float64), (m,))
    inside = (pa >= bound[:, None]) if keep_greater else (pa <= bound[:, None])
    last = count - 1
    rows = np.arange(m)
    ox = np.zeros((m, 2 * k))
    oy = np.zeros((m, 2 * k))
    keep = np.zeros((m, 2 * k), dtype=bool)
    for j in range(k):
        valid = j < count
        if j == 0:
            sx, sy, s_in = px[rows, np.maximum(last, 0)], py[rows, np.maximum(last, 0)], \
                inside[rows, np.maximum(last, 0)]
        else:
            sx, sy, s_in = px[:, j - 1], py[:, j - 1], inside[:, j - 1]
        ex, ey, e_in = px[:, j], py[:, j], inside[:, j]
        sa = sx if axis == 0 else sy
        ea = ex if axis == 0 else ey
        denominator = ea - sa
        cross = valid & (e_in != s_in) & (denominator != 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = (bound - sa) / np.where(cross, denominator, 1.0)
        ox[:, 2 * j] = sx + fraction * (ex - sx)
        oy[:, 2 * j] = sy + fraction * (ey - sy)
        keep[:, 2 * j] = cross
        ox[:, 2 * j + 1] = ex
        oy[:, 2 * j + 1] = ey
        keep[:, 2 * j + 1] = valid & e_in
    new_count = keep.sum(axis=1)
    order = np.argsort(~keep, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(ox, order, axis=1), np.take_along_axis(oy, order, axis=1), new_count


def _clip_areas(tri, x0, x1, y0, y1):
    m = tri.shape[0]
    px = np.zeros((m, _SLOTS))
    py = np.zeros((m, _SLOTS))
    px[:, :3] = tri[:, :, 0]
    py[:, :3] = tri[:, :, 1]
    count = np.full(m, 3)
    for axis, bound, greater in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        px, py, count = _clip_axis(px, py, count, axis, bound, greater)
    j = np.arange(_SLOTS)
    valid = j[None, :] < count[:, None]
    nxt = np.where(j[None, :] + 1 < count[:, None], j[None, :] + 1, 0)
    x = np.where(valid, px, 0.0)
    y = np.where(valid, py, 0.0)
    xn = np.where(valid, np.take_along_axis(px, nxt, axis=1), 0.0)
    yn = np.where(valid, np.take_along_axis(py, nxt, axis=1), 0.0)
    # Batched matmul reproduces np.dot(x, roll(y, -1)) bit for bit (zero padding
    # adds exact zeros), unlike sum(), whose order differs in the last bits.
    s1 = np.matmul(x[:, None, :], yn[:, :, None])[:, 0, 0]
    s2 = np.matmul(y[:, None, :], xn[:, :, None])[:, 0, 0]
    return np.where(count >= 3, 0.5 * np.abs(s1 - s2), 0.0)


def _plan_rows(tri, x, y):
    """Overlap rows (triangle, rectangle, area) of the triangles ``tri`` with the grid, in the
    repository order (triangle, then iy, then ix); also the number of candidate pairs clipped."""
    nx = x.size - 1
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    ix0 = np.maximum(0, np.searchsorted(x, lo[:, 0], side="right") - 1)
    ix1 = np.minimum(x.size - 2, np.searchsorted(x, hi[:, 0], side="left"))
    iy0 = np.maximum(0, np.searchsorted(y, lo[:, 1], side="right") - 1)
    iy1 = np.minimum(y.size - 2, np.searchsorted(y, hi[:, 1], side="left"))
    ncx = np.maximum(ix1 - ix0 + 1, 0)
    ncy = np.maximum(iy1 - iy0 + 1, 0)
    counts = ncx * ncy
    total = int(counts.sum())
    starts = np.cumsum(counts) - counts
    tri_parts, rect_parts, area_parts = [], [], []
    # Pairs in the repository order: triangle, then iy, then ix.
    for first in range(0, total, _CHUNK):
        idx = np.arange(first, min(first + _CHUNK, total))
        cell = np.searchsorted(starts, idx, side="right") - 1
        local = idx - starts[cell]
        iy = iy0[cell] + local // ncx[cell]
        ix = ix0[cell] + local % ncx[cell]
        area = _clip_areas(tri[cell], x[ix], x[ix + 1], y[iy], y[iy + 1])
        keep = area > 0.0
        tri_parts.append(cell[keep])
        rect_parts.append((iy * nx + ix)[keep])
        area_parts.append(area[keep])
    tri_idx = np.concatenate(tri_parts) if tri_parts else np.zeros(0, dtype=np.int64)
    rect_idx = np.concatenate(rect_parts) if rect_parts else np.zeros(0, dtype=np.int64)
    overlaps = np.concatenate(area_parts) if area_parts else np.zeros(0)
    return tri_idx, rect_idx, overlaps, total


# ---------------------------------------------------------------- meshes
def fast_mesh_triangles(domain):
    """Vectorised mesh_triangles (same values, same local-cell order)."""
    if domain.topology.dim != 2:
        raise ValueError("A two-dimensional mesh is required")
    local_cells = domain.topology.index_map(2).size_local
    if domain.comm.size != 1:
        raise RuntimeError("The provisional 2D runner currently requires serial execution")
    dofmap = np.asarray(domain.geometry.dofmap)
    if dofmap.ndim != 2 or dofmap.shape[1] != 3:
        raise RuntimeError("Only affine triangular transport meshes are supported")
    return np.ascontiguousarray(domain.geometry.x[:, :2][dofmap[:local_cells]], dtype=np.float64)


def fast_refine_marked_triangles_with_parent(domain, marked):
    """Same refinement and parent map as ``refine_marked_triangles_with_parent``, without the MPI leak.

    dolfinx 0.11 ``mesh.refine`` with its default partitioner (ghost-layer
    update) leaves one MPI communicator behind per call; MPICH allows ~2048,
    which aborted F4 at step 1784. In serial, ``partitioner=None`` builds the
    bitwise-identical mesh and parent map and frees everything. The edge
    selection is vectorised (same sorted unique edges).
    """
    from dolfinx import mesh
    local_cells = domain.topology.index_map(2).size_local
    markers = np.asarray(marked, dtype=bool)
    if markers.shape != (local_cells,):
        raise ValueError("There must be one refinement marker per local triangle")
    if not np.any(markers):
        return domain, np.arange(local_cells, dtype=np.int32)
    if domain.comm.size != 1:
        raise RuntimeError("The leak-free refinement path is serial only")
    domain.topology.create_connectivity(2, 1)
    cell_to_edge = domain.topology.connectivity(2, 1)
    if cell_to_edge is None:
        raise RuntimeError("DOLFINx did not construct cell-to-edge connectivity")
    offsets = np.asarray(cell_to_edge.offsets)
    if not np.all(np.diff(offsets) == 3):
        raise RuntimeError("Unexpected cell-to-edge layout")
    selected_edges = np.unique(
        np.asarray(cell_to_edge.array).reshape(-1, 3)[np.flatnonzero(markers)].ravel()).astype(np.int32)
    refined, parent_cell, _ = mesh.refine(domain, selected_edges, partitioner=None,
                                          option=mesh.RefinementOption.parent_cell)
    parent_cell = np.asarray(parent_cell, dtype=np.int32)
    input_cell_index = np.asarray(refined.topology.original_cell_index, dtype=np.int64)
    if input_cell_index.shape != parent_cell.shape:
        raise RuntimeError("Refined-cell permutation has an incompatible shape")
    if np.any(input_cell_index < 0) or np.any(input_cell_index >= parent_cell.size):
        raise RuntimeError("Refined-cell permutation contains invalid indices")
    return refined, parent_cell[input_cell_index]


class FastPath:
    """Per-run state of the fast path: options, one-entry solver cache, last transfer plan and
    statistics. Pass ``fast=True`` (or a :class:`FastOptions`) to ``run_adaptive_loop_2d``."""

    def __init__(self, options: FastOptions | None = None) -> None:
        self.options = FastOptions() if options is None else options
        self.stats: dict = {"solvers_built": 0, "solvers_reused": 0, "plans_built": 0,
                            "plan_seconds": 0.0, "plan_pairs_clipped": 0}
        self._cache = {"key": None, "solver": None, "domain": None}
        self._last_plan = {"tri": None, "x": None, "y": None, "plan": None}

    def solver_factory(self, **kwargs):
        """Drop-in for ``ResidualMinimizationBackwardEuler2D(...)`` with a one-entry cache."""
        domain = kwargs["domain"]
        key = (id(domain), tuple(sorted((k, _freeze(v)) for k, v in kwargs.items() if k != "domain")))
        cache = self._cache
        if cache["key"] == key and cache["domain"] is domain:
            self.stats["solvers_reused"] += 1
            return cache["solver"]
        cache.update(key=None, solver=None, domain=None)  # free the old factorisation first
        gc.collect()
        solver = FastResidualMinimization2D(**kwargs)
        solver.batch_rhs = self.options.batch_rhs
        solver.stats = self.stats
        cache.update(key=key, solver=solver, domain=domain)
        self.stats["solvers_built"] += 1
        return solver

    def plan_create(self, cls, triangles, x_edges, y_edges, *, require_rectangle_coverage=True):
        """Plan builder for ``adaptdg.transfer_2d.use_plan_builder`` (same plan as ``create``)."""
        from adaptdg import transfer_2d as tr
        started = perf_counter()
        tri = tr._triangles(triangles)
        x = tr._edges(x_edges, "x_edges")
        y = tr._edges(y_edges, "y_edges")
        tri_areas = tr.triangle_areas(tri)
        rectangle_areas = np.outer(np.diff(y), np.diff(x))
        nx, ny = x.size - 1, y.size - 1
        last = self._last_plan
        rows = None
        if self.options.plan_reuse and require_rectangle_coverage and last["plan"] is not None \
                and np.array_equal(last["x"], x) and np.array_equal(last["y"], y):
            from adaptdg.history_projection import canonical_vertices
            from adaptdg.history_projection_fast import match_identical
            old_plan = last["plan"]
            new_idx, old_idx, _ = match_identical(canonical_vertices(last["tri"]), canonical_vertices(tri))
            if new_idx.size:
                # rows of the old plan, grouped by old triangle, copied to the matching new triangle
                new_of_old = np.full(old_plan.triangle_areas.size, -1, dtype=np.int64)
                new_of_old[old_idx] = new_idx
                keep = new_of_old[old_plan.triangle_indices] >= 0
                rest = np.ones(tri.shape[0], dtype=bool)
                rest[new_idx] = False
                rest_idx = np.flatnonzero(rest)
                t2, r2, o2, total = _plan_rows(tri[rest_idx], x, y) if rest_idx.size else (
                    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0), 0)
                tri_idx = np.concatenate((new_of_old[old_plan.triangle_indices[keep]], rest_idx[t2]))
                rect_idx = np.concatenate((old_plan.rectangle_indices[keep], r2))
                overlaps = np.concatenate((old_plan.overlap_areas[keep], o2))
                order = np.argsort(tri_idx, kind="stable")      # rows of one triangle keep their (iy, ix) order
                tri_idx, rect_idx, overlaps = tri_idx[order], rect_idx[order], overlaps[order]
                rows = (tri_idx, rect_idx, overlaps, total)
                self.stats["plan_triangles_reused"] = self.stats.get("plan_triangles_reused", 0) + int(new_idx.size)
        if rows is None:
            rows = _plan_rows(tri, x, y)
        tri_idx, rect_idx, overlaps, total = rows
        covered_triangles = np.bincount(tri_idx, weights=overlaps, minlength=tri.shape[0])
        covered_rectangles = np.bincount(rect_idx, weights=overlaps, minlength=rectangle_areas.size)
        if not np.allclose(covered_triangles, tri_areas, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError("The chemistry grid does not fully cover a transport triangle")
        if require_rectangle_coverage and not np.allclose(
                covered_rectangles, rectangle_areas.ravel(), rtol=1.0e-9, atol=1.0e-14):
            raise ValueError("The transport triangles do not fully cover the chemistry grid")
        self.stats["plans_built"] += 1
        self.stats["plan_pairs_clipped"] += total
        self.stats["plan_seconds"] += perf_counter() - started
        plan = cls(triangle_indices=tri_idx.astype(np.int64), rectangle_indices=rect_idx.astype(np.int64),
                   overlap_areas=overlaps, triangle_areas=tri_areas, rectangle_areas=rectangle_areas.ravel(),
                   nx=nx, ny=ny)
        if self.options.plan_reuse and require_rectangle_coverage:
            self._last_plan.update(tri=tri.copy(), x=x.copy(), y=y.copy(), plan=plan)
        return plan

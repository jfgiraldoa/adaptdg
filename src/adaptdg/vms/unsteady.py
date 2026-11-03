"""Unsteady residual minimisation with adaptive refinement in space at every time step.

``u_t + L u = f`` is discretised in time with backward Euler (BE) or BDF2 and each step is the stationary saddle-point problem of
:mod:`adaptdg.vms.saddle` for the operator ``c1 I + c2 L`` (legacy ``TimeIntegratorMethod``): ``c1 = 1`` and ``c2 = dt`` (BE) or
``2 dt/3`` (BDF2); the right-hand side is ``c2 l + sum_j a_j (u^{n-j}, v)`` with ``a = (1)`` (BE) or ``(4/3, -1/3)`` (BDF2) and the Gram
form is ``c1 (a, c) + c2 g`` (:class:`~adaptdg.vms.dgforms.DGForms`, ``time_coeffs``). The paper's unsteady text (Giraldo & Calo,
Math. Comput. Appl. 28 (1), 7, 2023) was not available: the discretisation follows the legacy implementation. BDF2 starts with one BE step.

Every step runs the loop solve -> estimate -> mark -> refine of :mod:`adaptdg.vms.loop` (stopping rule per step: ``E <= tol``, at most
``max_levels`` solves, ``ndofs >= dofmax``). Presets (:mod:`adaptdg.vms.preset`): ``paper`` starts each step from the final mesh of
the previous one and carries ``u = ubar + utilde``; ``legacy`` restarts from the initial mesh at every step (the legacy re-creates
the mesh), carries ``phi`` and projects it onto CG(p) in the Gram norm before use, with the other legacy choices of the preset.

History transfer between meshes: the field kept after a step lives on that step's final mesh and is moved to the current mesh of every
level by an L2 projection onto DG(p). The old function is sampled at the quadrature points of the new mesh with
``interpolate_nonmatching`` (a quadrature-element space) and the projection solves ``(w, v) = (f_old, v)``. This is the identity for
nested meshes (``paper``: each level refines the previous mesh), preserves the constants, hence the integral of ``u`` up to quadrature
error, and is a least-squares (non-conservative in the pointwise sense) coarsening when the new mesh is not a refinement (``legacy``).
Serial runs only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from adaptdg.vms.loop import LoopSettings, mark_cells

INTEGRATORS = ("be", "bdf2")


@dataclass
class UnsteadySettings(LoopSettings):
    """``LoopSettings`` of the per-step loop (``max_levels``: solves per step, ``tol``: per-step estimator tolerance,
    ``refinement="uniform"``: fixed uniform mesh, one solve per step) plus the time stepping."""

    integrator: str = "bdf2"  # "be" | "bdf2"
    dt: float = 0.005
    t_start: float = 0.0
    t_final: float = 0.1
    initial_refinements: int = 0  # uniform refinements of the start mesh before the first step
    reconstruct: bool = True


def step_coefficients(scheme: str, dt: float) -> tuple[float, float, tuple[float, ...]]:
    """``(c1, c2, history)``: ``c1 u + c2 L u = c2 f + sum_j history[j] u^{n-1-j}`` (legacy ``TimeIntegratorMethod``)."""
    if scheme == "be":
        return 1.0, dt, (1.0,)
    if scheme == "bdf2":
        return 1.0, 2.0 * dt / 3.0, (4.0 / 3.0, -1.0 / 3.0)
    raise ValueError(f"Unknown time integrator {scheme!r}; choose from {INTEGRATORS}")


def build_unsteady_settings(spec: dict[str, Any], problem_spec: dict[str, Any] | None = None) -> UnsteadySettings:
    """``UnsteadySettings`` from numerics.json keys plus ``initial_mesh_size`` of problem.json."""
    names = {f.name for f in fields(UnsteadySettings)}
    unknown = set(spec) - names - {"eta_0"}
    if unknown:
        raise KeyError(f"Unknown numerics keys: {sorted(unknown)}")
    merged = dict(spec)
    if "eta_0" in merged:
        merged["eta0"] = merged.pop("eta_0")
    if problem_spec and "initial_mesh_size" in problem_spec:
        merged.setdefault("initial_mesh_size", int(problem_spec["initial_mesh_size"]))
    return UnsteadySettings(**merged)


def l2_project(expression: Any, space: Any, degree: int) -> Any:
    """L2 projection of a UFL expression onto ``space`` with quadrature ``degree`` (MUMPS LU)."""
    import ufl
    from dolfinx.fem import petsc

    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.dx(domain=space.mesh, metadata={"quadrature_degree": degree})
    opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    return petsc.LinearProblem(u * v * dx, expression * v * dx, petsc_options=opts,
                               petsc_options_prefix="vms_hist_").solve()


def transfer(source: Any, space: Any, degree: int | None = None) -> Any:
    """L2 projection onto ``space`` (DG(p)) of ``source``, a function on the same or on another mesh (see module docstring)."""
    import basix.ufl
    import ufl
    from dolfinx import fem

    mesh = space.mesh
    if source.function_space.mesh is mesh:
        out = fem.Function(space)
        out.interpolate(source)
        return out
    q = degree if degree is not None else 2 * space.element.basix_element.degree + 2
    cdim = mesh.topology.dim
    ncells = mesh.topology.index_map(cdim).size_local + mesh.topology.index_map(cdim).num_ghosts
    cells = np.arange(ncells, dtype=np.int32)
    qspace = fem.functionspace(mesh, basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(), degree=q))
    sampled = fem.Function(qspace)
    data = fem.create_interpolation_data(qspace, source.function_space, cells, padding=1e-8)
    sampled.interpolate_nonmatching(source, cells, data)
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": q, "quadrature_scheme": "default"})
    from dolfinx.fem import petsc

    opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    return petsc.LinearProblem(u * v * dx, sampled * v * dx, petsc_options=opts,
                               petsc_options_prefix="vms_hist_").solve()


def gram_project_cg(forms: Any, source: Any, p: int) -> Any:
    """Gram-norm projection of the DG(p) function ``source`` onto CG(p), returned as a DG(p) function (legacy ``Vh`` projection)."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc

    mesh = source.function_space.mesh
    cg = fem.functionspace(mesh, ("Lagrange", p))
    u, v = ufl.TrialFunction(cg), ufl.TestFunction(cg)
    opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    proj = petsc.LinearProblem(forms.gram(u, v), forms.gram(source, v), petsc_options=opts,
                               petsc_options_prefix="vms_gramproj_").solve()
    out = fem.Function(source.function_space)
    out.interpolate(proj)
    return out


def time_integrated(rows: list[dict[str, float]], dt: float) -> dict[str, float]:
    """``int_<key>`` = ``sqrt(sum_n dt e_n^2)`` (discrete ``L2(0, T)`` norm over the steps) and ``max_<key>`` of every error column."""
    keys = [k for k in rows[0] if (k.startswith("L2") or k.startswith("Vh")) and np.isfinite(rows[-1][k])]
    out: dict[str, float] = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[f"int_{k}"] = float(np.sqrt(dt * np.sum(v * v)))
        out[f"max_{k}"] = float(np.max(v))
    return out


class UnsteadyLoop:
    """Time stepping with adaptive refinement in space for a time dependent :class:`~adaptdg.vms.problems.Problem`
    (``bind``, ``set_time``, ``initial``). Rows of ``history`` (one per step) and ``levels`` (one per solve) are filled by :meth:`run`."""

    def __init__(self, problem: Any, settings: UnsteadySettings, domain: Any = None) -> None:
        from adaptdg.vms.preset import resolve_preset

        s = settings
        if s.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if s.refinement not in ("adaptive", "uniform"):
            raise ValueError("refinement must be 'adaptive' or 'uniform'")
        if s.error_targets not in ("both", "exact"):
            raise ValueError("error_targets must be 'both' or 'exact'")
        self.preset = resolve_preset(s.preset, **(s.preset_overrides or {}))
        if self.preset.history_field != "ubar" and not s.reconstruct:
            raise ValueError("history_field 'u' and 'phi' need reconstruct = True")
        self.problem, self.settings = problem, s
        self._start = domain
        self.nsteps = int(round((s.t_final - s.t_start) / s.dt))
        if self.nsteps < 1 or abs(self.nsteps * s.dt - (s.t_final - s.t_start)) > 1e-9 * max(1.0, s.t_final):
            raise ValueError("t_final - t_start must be a positive multiple of dt")
        self.domain: Any = None  # mesh of the last step
        self.history: list[dict[str, float]] = []
        self.levels: list[dict[str, float]] = []
        self.vms: Any = None
        self.result: Any = None
        self._past: list[Any] = [None]  # newest last: None = initial condition, else a function on a previous mesh

    def _start_mesh(self) -> Any:
        from adaptdg.vms.refine import refine_uniform

        domain = self._start if self._start is not None else self.problem.initial_mesh(self.settings.initial_mesh_size)
        for _ in range(self.settings.initial_refinements):
            domain = refine_uniform(domain)
        return domain

    def _history_function(self, entry: Any, space: Any, forms: Any, qdeg: int, t: float) -> Any:
        """The history field ``entry`` on the current mesh as a DG(p) function (initial condition: L2 projection)."""
        s = self.settings
        if entry is None:
            self.problem.set_time(s.t_start)
            fn = l2_project(self.problem.initial(forms.x), space, qdeg)
            self.problem.set_time(t)
        else:
            fn = transfer(entry, space)
        if self.preset.history_projection == "gram_cg":
            fn = gram_project_cg(forms, fn, s.p)
        return fn

    def run(self, progress: Callable[[str], None] | None = print,
            on_step: Callable[[dict[str, float], Any], None] | None = None) -> list[dict[str, float]]:
        """Run all steps; ``on_step(row, loop)`` is called after each accepted step (e.g. to save the mesh)."""
        from dolfinx import fem

        from adaptdg.marking import fixed_fraction_bands
        from adaptdg.vms.dgforms import DGForms
        from adaptdg.vms.norms import error_targets, history_columns
        from adaptdg.vms.reconstruction import reconstruct
        from adaptdg.vms.refine import refine_marked
        from adaptdg.vms.saddle import RMSolver

        s, preset, problem = self.settings, self.preset, self.problem
        theta = s.theta_ref if s.theta_ref is not None else 0.25
        qdeg = s.quadrature_degree if preset.quadrature_degree is None else preset.quadrature_degree
        ordering = s.adjoint_ordering if preset.adjoint_ordering is None else preset.adjoint_ordering
        ratio = theta if preset.marking_ratio is None else preset.marking_ratio
        targets = ("exact", "projection") if s.error_targets == "both" else ("exact",)
        domain = self._start_mesh()
        work = 0
        t_run = time.time()
        for k in range(1, self.nsteps + 1):
            t0 = time.time()
            t = s.t_start + k * s.dt
            scheme = "be" if (s.integrator == "be" or k == 1) else "bdf2"
            c1, c2, coeffs = step_coefficients(scheme, s.dt)
            entries = self._past[::-1][:len(coeffs)]  # newest first, as `coeffs`
            if preset.step_mesh == "initial":
                domain = self._start_mesh()
            for level in range(s.max_levels):
                tl = time.time()
                problem.bind(domain, t)
                space = fem.functionspace(domain, ("DG", s.p))
                gforms = DGForms(domain, problem, s.p, eta0=s.eta0, rhs_degree=qdeg, preset=preset,
                                 time_coeffs=tuple(fem.Constant(domain, c) for c in (c1, c2)))
                hist = [(a, self._history_function(e, space, gforms, qdeg, t)) for a, e in zip(coeffs, entries)]
                solver = RMSolver(problem, s.p, eta0=s.eta0, quadrature_degree=qdeg, solver=s.solver,
                                  preset=preset, time_coeffs=(c1, c2))
                self.result = res = solver.solve(domain, hist)
                ncells = int(domain.topology.index_map(domain.topology.dim).size_local)
                work += res.ndofs
                self.levels.append({"step": k, "level": level, "t": t, "ndofs": res.ndofs, "ncells": ncells,
                                    "E": res.E, "time": time.time() - tl})
                done = (level + 1 == s.max_levels or res.E <= s.tol or s.refinement == "uniform"
                        or (s.dofmax is not None and res.ndofs >= s.dofmax))
                if done:
                    break
                if preset.marking == "fixed_fraction":
                    marked = fixed_fraction_bands(res.indicators, ratio)
                else:
                    marked = mark_cells(res.indicators, theta, s.tolref)
                domain = refine_marked(domain, marked)[0]
            self.vms = reconstruct(res, adjoint_ordering=ordering, solver=s.solver) if s.reconstruct else None
            errs = error_targets(res, problem, s.p, degree=qdeg, vms=self.vms, targets=targets)
            cols = history_columns(errs)
            field = {"u": getattr(self.vms, "u_full", None), "phi": getattr(self.vms, "phi", None),
                     "ubar": res.ubar}[preset.history_field]
            self._past = (self._past + [field])[-2:]
            first = next(r for r in self.levels if r["step"] == k and r["level"] == 0)
            row = {"step": k, "t": t, "scheme": 1.0 if scheme == "be" else 2.0, "levels": level + 1,
                   "ndofs": res.ndofs, "ncells": ncells, "E": res.E, "E_level0": first["E"], "ndofs_level0": first["ndofs"],
                   **cols, "work": work, "time": time.time() - t0}
            self.history.append(row)
            self.domain = domain
            if progress:
                extra = f" Vh_phi={row['Vh_phi']:.3e}" if s.reconstruct else ""
                progress(f"step {k}/{self.nsteps} t={t:.4f} {scheme} levels={level + 1} ndofs={res.ndofs} E={res.E:.3e} "
                         f"L2={row['L2']:.3e} Vh={row['Vh']:.3e}{extra} t={row['time']:.1f}s (total {time.time() - t_run:.0f}s)")
            if on_step is not None:
                on_step(row, self)
        return self.history

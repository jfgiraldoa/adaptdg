"""Nonlinear residual minimisation for ``div(f(u)) - div(kappa grad u) = s`` (Giraldo & Calo 2023, CMAME 417,
116285, section 6): dG forms, saddle point, Newton solvers, multiscale reconstruction and the adaptive loop.

* eqs. (44)-(48): :class:`NonlinearForms` -- SWIP diffusion plus the conservative form
  ``-(grad v, f(u)) + ([[v]], Phi)`` with the Lax-Friedrichs flux ``Phi`` (``eta_f = |f'(u).n|``, interior: maximum
  over both sides) and the weakly imposed Dirichlet flux; natural outflow faces (``Problem.outflow_tags``) get
  ``v f(u).n``. With ``preset.nonlinear_flux == "wind"`` the legacy form is used instead: the wind ``f'(u)`` goes
  through the generic upwind form with a non-conservative volume term (legacy ``bh_dar_NL``).
* eq. (51): :class:`NonlinearRMSolver` -- saddle point ``g(w, eps) + n(w; ubar) = l(w)``, ``n'(eps, v; ubar) = 0`` on
  ``DG(p) x CG(p)`` in the unknowns ``(eps, ubar)``; the Jacobian is the exact ``ufl.derivative`` of the residual.
* eqs. (53)-(54): :func:`newton_solve` -- ``method="hybrid"`` (default of the paper preset): Gauss-Newton Jacobian
  (the second derivative ``n''`` of the second equation is dropped) until ``|R| < switch``, full Jacobian afterwards,
  with Armijo backtracking on ``|R|`` (``damping="backtrack"``, ``t`` = 1, 1/2, ... as the relaxation ``k`` of eq. 54;
  the full-Jacobian step falls back to the Gauss-Newton one when it needs ``t < 1/4``); ``method="damped"`` (legacy preset): Gauss-Newton Jacobian with the legacy damping
  ``t = 1/(1 + K |R|)``, ``K`` multiplied by 10 on failure and divided by 10 on success; ``method="full"``.
* eqs. (61)-(62): :func:`reconstruct_nonlinear` -- ``utilde`` from ``n'(v, utilde; ubar) = g(v, eps)``, the adjoint
  reconstruction ``n'(v, ucheck; ubar) = g(v, eps) + n'(eps, v; ubar)`` and the nonlinear dG solution ``theta`` (Newton).
* :class:`NonlinearAdaptiveLoop` -- solve, estimate, mark, refine with a warm start between levels; a level where
  Newton fails is flagged in the history (and optionally retried with the hybrid solver) but never stops the loop.

Several roots can exist on one mesh: Newton may land on different ones depending on the path, so the L2 error of
``ubar`` is path dependent while the estimator ``E`` is stable (see ``docs/VMS.md``). Serial runs only.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from adaptdg.vms.dgforms import DGForms
from adaptdg.vms.saddle import RMResult, _dg0_test, cell_values


# --------------------------------------------------------------------------------------------------
# Forms
# --------------------------------------------------------------------------------------------------
class NonlinearForms(DGForms):
    """dG forms of a :class:`~adaptdg.vms.problems.NonlinearProblem`; ``degree`` defaults to ``2p + 4``."""

    def __init__(self, domain: Any, problem: Any, p: int, *, eta0: float = 1.0, degree: int | None = None,
                 rhs_degree: int = 12, preset: Any = None) -> None:
        super().__init__(domain, problem, p, eta0=eta0, degree=2 * p + 4 if degree is None else degree,
                         rhs_degree=rhs_degree, preset=preset)
        self.wind = self.preset.nonlinear_flux == "wind"

    def operator(self, v: Any, u: Any) -> Any:
        """``N(v; u) = n_h(v; u) - l_h(v)``: the residual of the dG problem (44) (Dirichlet and source data inside)."""
        import ufl

        n, eta = self.n, self.eta
        uD = self.problem.dirichlet(self.x)
        f = self.problem.source(self.x)
        gam_b = _gam_b(self)
        data = (eta * gam_b * v * uD - ufl.dot(_kflux(self, ufl.grad(v)), n) * uD) * self.ds_r  # eq. (11)
        form = self.diffusion(v, u) - data
        form += self._wind_convection(v, u, uD) if self.wind else self._lf_convection(v, u, uD)
        if f is not None:
            form -= v * f * self.dx_r
        return form

    def gram(self, a: Any, c: Any, weight: Any = None) -> Any:
        """Gram form of the norm ``V_h``. ``problem.norm_form == "isotropic"``: eqs. (49)-(50), SWIP part plus
        ``(a, c) + sum_F ([[a]], [[c]])_F + sum_K h_K (grad a, grad c)_K`` (all faces, no wind); otherwise the
        streamline form of the linear problems (:meth:`DGForms.gram` with the norm wind ``problem.beta``)."""
        if getattr(self.problem, "norm_form", "streamline") != "isotropic":
            return super().gram(a, c, weight)
        import ufl
        from adaptdg.vms.dgforms import _flux, _normal_kappa

        n, eta = self.n, self.eta
        wc = 1.0 if weight is None else weight
        wf = 1.0 if weight is None else ufl.avg(weight)
        gam = self._interior_weights()[2] * self.preset.gamma_factor_gram
        gam_b = _normal_kappa(self.kappa, n)
        h = ufl.CellDiameter(self.domain)
        ds_all = ufl.Measure("ds", domain=self.domain, metadata=self.dx.metadata())
        return (ufl.inner(ufl.grad(a), _flux(self.kappa, ufl.grad(c))) * wc * self.dx
                + ufl.avg(eta) * gam * ufl.inner(ufl.jump(a, n), ufl.jump(c, n)) * wf * self.dS
                + eta * gam_b * a * c * wc * self.ds
                + a * c * wc * self.dx + ufl.jump(a) * ufl.jump(c) * wf * self.dS + a * c * wc * ds_all
                + h * ufl.inner(ufl.grad(a), ufl.grad(c)) * wc * self.dx)

    def _lf_convection(self, v: Any, u: Any, uD: Any) -> Any:
        """Eqs. (45)-(48): ``-(grad v, f(u)) + ([[v]], Phi)`` with Lax-Friedrichs ``Phi`` and the outflow term."""
        import ufl

        n, flux, dflux = self.n, self.problem.flux, self.problem.dflux
        eta_i = ufl.max_value(abs(ufl.dot(dflux(u("+")), n("+"))), abs(ufl.dot(dflux(u("-")), n("+"))))
        phi = 0.5 * (ufl.dot(flux(u("+")), n("+")) + ufl.dot(flux(u("-")), n("+")) + eta_i * (u("+") - u("-")))
        eta_b = abs(ufl.dot(dflux(u), n))
        phi_b = (0.5 * (ufl.dot(flux(u), n) + eta_b * u) + 0.5 * (ufl.dot(flux(uD), n) - eta_b * uD))  # eq. (46)
        form = (-ufl.inner(ufl.grad(v), flux(u)) * self.dx + ufl.jump(v) * phi * self.dS
                + v * phi_b * self.ds)
        if self.ds_out is not None:
            form += v * ufl.dot(flux(u), n) * self.ds_out  # natural outflow: f(u).n leaves, no diffusive flux
        return form

    def _wind_convection(self, v: Any, u: Any, uD: Any) -> Any:
        """Legacy ``bh_dar_NL``: wind ``beta = f'(u)`` in the generic upwind form (non-conservative volume term)."""
        import ufl

        n = self.n
        beta = self.problem.dflux(u)
        bn, bnp = ufl.dot(beta, n), ufl.dot(beta("+"), n("+"))
        return (v * ufl.dot(beta, ufl.grad(u)) * self.dx
                + (0.5 * abs(bnp) * ufl.jump(v) * ufl.jump(u) - ufl.avg(v) * bnp * ufl.jump(u)) * self.dS
                - ufl.min_value(bn, 0.0) * v * (u - uD) * self.ds)

    def wind_linearisation(self, t: Any, d: Any, ub: Any) -> Any:
        """Legacy ``bh_prima``: ``n'(t; d, ub)`` with the upwind faces frozen at the wind ``f'(ub)`` (test ``t``,
        direction ``d``)."""
        import ufl

        n = self.n
        beta, dbeta = self.problem.dflux(ub), self.problem.d2flux(ub)
        bn, bnp = ufl.dot(beta, n), ufl.dot(beta("+"), n("+"))
        return (self.diffusion(t, d)
                + t * (ufl.dot(beta, ufl.grad(d)) + d * ufl.dot(dbeta, ufl.grad(ub))) * self.dx
                + (0.5 * abs(bnp) * ufl.jump(t) * ufl.jump(d) - ufl.avg(t) * bnp * ufl.jump(d)) * self.dS
                - ufl.min_value(bn, 0.0) * t * d * self.ds)

    def linearised(self, t: Any, d: Any, ub: Any) -> Any:
        """``n'(t; d, ub)`` for a coefficient ``ub``: exact Gateaux derivative (paper) or the legacy frozen wind."""
        import ufl

        if self.wind:
            return self.wind_linearisation(t, d, ub)
        return ufl.derivative(self.operator(t, ub), ub, d)


def _kflux(forms: NonlinearForms, g: Any) -> Any:
    from adaptdg.vms.dgforms import _flux

    return _flux(forms.kappa, g)


def _gam_b(forms: NonlinearForms) -> Any:
    from adaptdg.vms.dgforms import _normal_kappa

    return _normal_kappa(forms.kappa, forms.n)


# --------------------------------------------------------------------------------------------------
# Newton
# --------------------------------------------------------------------------------------------------
@dataclass
class NewtonOptions:
    """Options of :func:`newton_solve`. ``method``: ``"hybrid"`` | ``"damped"`` | ``"full"``."""

    method: str = "hybrid"
    #: stop when the Vh norm of the increment of ``ubar`` is below ``tol_du`` or ``|R|`` below ``tol_res``
    #: (the paper says 1e-6 for the increment, the hybrid default is tighter).
    tol_du: float = 1e-9
    tol_res: float = 1e-11
    maxit: int = 100
    #: hybrid: Gauss-Newton Jacobian while ``|R| >= switch``, full Jacobian afterwards.
    switch: float = 1e-3
    #: ``None`` -> ``"legacy"`` for method ``"damped"``, ``"backtrack"`` for ``"hybrid"``, ``"none"`` for ``"full"``;
    #: ``"legacy"``: ``t = 1/(1 + K |R|)``; ``"backtrack"``: ``t`` = 1, 1/2, 1/4, ... until ``|R_new| < (1 - c t) |R|``.
    damping: str | None = None
    omega: float = 0.5  # legacy acceptance threshold on om = (1 - |R_new|/|R|)/t
    armijo: float = 1e-4  # backtracking: sufficient decrease constant c
    t_min: float = 1e-4  # backtracking: give up (status "stalled") when no direction accepts a step above this
    #: hybrid + backtracking, ``|R| < switch``: the full-Jacobian step is accepted only if ``t >= t_full``, otherwise the
    #: Gauss-Newton step is tried (the full Jacobian misses the kink of ``|f'(u).n|`` at the shock: exact near the root only).
    t_full: float = 0.25
    k_max: float = 1e14  # legacy: give up when K exceeds this (the original loop would continue)
    diverge: float = 1e8  # undamped iterations stop when |R| exceeds this
    #: hybrid only: if the undamped iteration fails, restart from the same initial state with the legacy damping
    #: (``omega_fallback``); needed for the single-component case at small kappa (see docs/VMS.md).
    damping_fallback: bool = True
    omega_fallback: float = 0.1
    #: a failed solve from a warm start is retried once from the initial guess (``eps = 0``, ``problem.initial_guess``);
    #: if that fails too (and the system has at most ``continuation_max_dofs`` unknowns) it is retried by continuation in the
    #: diffusivity (``problem.diffusivity``): solve at
    #: ``continuation_start`` and decrease ``kappa`` by ``continuation_ratio`` per stage (the ratio is reduced when a stage
    #: fails); needed for the cold start of the first mesh at ``kappa = 1e-4``.
    continuation: bool = True
    continuation_start: float = 1e-2
    continuation_ratio: float = 3.16
    continuation_max_dofs: int = 20000
    cold_retry: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("hybrid", "damped", "full"):
            raise ValueError("Newton method must be 'hybrid', 'damped' or 'full'")
        if self.damping not in (None, "none", "legacy", "backtrack"):
            raise ValueError("damping must be None, 'none', 'legacy' or 'backtrack'")

    @property
    def damping_kind(self) -> str:
        return self.damping or {"damped": "legacy", "hybrid": "backtrack"}.get(self.method, "none")

    @property
    def legacy_damping(self) -> bool:
        return self.damping_kind == "legacy"


def newton_options(preset: Any, overrides: dict[str, Any] | None = None) -> NewtonOptions:
    """Defaults of the preset (paper: hybrid; legacy: damped, maxit 20000, tolerance 1e-12) plus ``overrides``."""
    from adaptdg.vms.preset import resolve_preset

    preset = resolve_preset(preset)
    if preset.newton_method == "damped":
        base = NewtonOptions(method="damped", tol_du=1e-12, tol_res=1e-12, maxit=20000, continuation=False, cold_retry=False)
    else:
        base = NewtonOptions(method=preset.newton_method)
    return replace(base, **overrides) if overrides else base


@dataclass
class NewtonInfo:
    """Outcome of one Newton solve (``status``: conv | already | maxit | failK | stalled | diverged | singular)."""

    status: str = "none"
    converged: bool = False
    iterations: int = 0
    failed_steps: int = 0  # rejected damping trials (legacy damping)
    residual0: float = 0.0
    residual: float = 0.0
    increment: float = 0.0
    attempts: int = 1  # 2 when the damping fallback of the hybrid solver was used

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def newton_solve(sol: Any, residual: Any, jacobian: Callable[[str], Any], increment_norm: Callable[[np.ndarray], float],
                 opts: NewtonOptions, solver: str = "mumps") -> NewtonInfo:
    """:func:`_newton` plus, for ``method == "hybrid"`` with ``damping_fallback``, a damped restart after a failure."""
    start = sol.x.array.copy()
    info = _newton(sol, residual, jacobian, increment_norm, opts, solver)
    if info.converged or opts.method != "hybrid" or not opts.damping_fallback or opts.damping_kind == "backtrack":
        return info
    sol.x.array[:] = start
    sol.x.scatter_forward()
    retry = replace(opts, damping="legacy", omega=opts.omega_fallback, damping_fallback=False)
    second = _newton(sol, residual, jacobian, increment_norm, retry, solver)
    second.iterations += info.iterations
    second.failed_steps += info.failed_steps
    second.attempts, second.residual0 = 2, info.residual0
    return second


def _newton(sol: Any, residual: Any, jacobian: Callable[[str], Any], increment_norm: Callable[[np.ndarray], float],
            opts: NewtonOptions, solver: str = "mumps") -> NewtonInfo:
    """Newton iteration for ``R(sol) = 0`` (serial); ``sol`` is updated in place.

    ``residual`` is a compiled linear form, ``jacobian("gn" | "full")`` returns the compiled Jacobian form and
    ``increment_norm(delta)`` the Vh norm of an increment. On failure ``sol`` is restored to the iterate with the
    smallest residual (the failure stays flagged in the returned info).
    """
    if opts.damping_kind == "backtrack":
        return _newton_backtrack(sol, residual, jacobian, increment_norm, opts, solver)
    from dolfinx.fem import petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    def res_vec() -> Any:
        b = petsc.assemble_vector(residual)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return b

    def res_norm() -> float:
        b = res_vec()
        val = float(b.norm())
        b.destroy()
        return val

    info = NewtonInfo()
    info.residual0 = info.residual = nres = res_norm()
    if nres < opts.tol_res:
        info.status, info.converged = "already", True
        return info
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(solver)
    legacy = opts.legacy_damping
    best, best_x = nres, sol.x.array.copy()
    K, status, need_solve = 0.0, "maxit", True
    x0 = delta = None
    nres_k = nres
    while info.iterations < opts.maxit:
        if need_solve:
            kind = "full" if opts.method == "full" or (opts.method == "hybrid" and nres < opts.switch) else "gn"
            A = petsc.assemble_matrix(jacobian(kind))
            A.assemble()
            rhs = res_vec()
            rhs.scale(-1.0)
            ksp.setOperators(A)
            dvec = A.createVecRight()
            try:
                ksp.solve(rhs, dvec)
                delta = dvec.array.copy()
            except PETSc.Error:
                delta = np.full(sol.x.array.shape, np.nan)
            for obj in (A, rhs, dvec):
                obj.destroy()
            if not np.isfinite(delta).all():
                status = "singular"
                break
            x0, nres_k, need_solve = sol.x.array.copy(), nres, False
        t = 1.0 / (1.0 + K * nres_k) if legacy else 1.0
        sol.x.array[:] = x0 + t * delta
        sol.x.scatter_forward()
        nres1 = res_norm()
        om = (1.0 - nres1 / nres_k) / t
        if legacy and om < opts.omega and abs(om) > 1e-12:  # damping trial rejected: smaller step, same direction
            info.failed_steps += 1
            K = 1.0 if K == 0 else K * 10
            if K > opts.k_max:
                status = "failK"
                break
            continue
        info.iterations += 1
        info.increment = increment_norm(t * delta)
        info.residual = nres = nres1
        if nres < best:
            best, best_x = nres, sol.x.array.copy()
        if info.increment < opts.tol_du or nres < opts.tol_res:
            status = "conv"
            break
        if not np.isfinite(nres) or (not legacy and nres > opts.diverge):
            status = "diverged"
            break
        K = K / 10
        need_solve = True
    ksp.destroy()
    info.status, info.converged = status, status == "conv"
    if not info.converged:  # keep the best iterate; the flag records the failure
        sol.x.array[:] = best_x
        sol.x.scatter_forward()
        info.residual = best
    return info


def _newton_backtrack(sol: Any, residual: Any, jacobian: Callable[[str], Any],
                      increment_norm: Callable[[np.ndarray], float], opts: NewtonOptions,
                      solver: str = "mumps") -> NewtonInfo:
    """Newton iteration with Armijo backtracking on ``|R|`` (see :func:`_newton` for the arguments).

    Far from the root (``|R| >= switch``) the Gauss-Newton direction is used; closer, the full-Jacobian direction is tried
    first and accepted if ``t >= t_full``, otherwise the Gauss-Newton direction is used (and the full one is not tried
    again until ``|R|`` has dropped by a factor 10). The iteration stops with ``stalled`` when no step above ``t_min`` gives
    a sufficient decrease.
    """
    from dolfinx.fem import petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    def res_norm() -> float:
        b = petsc.assemble_vector(residual)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        val = float(b.norm())
        b.destroy()
        return val

    def direction(kind: str) -> np.ndarray | None:
        A = petsc.assemble_matrix(jacobian(kind))
        A.assemble()
        rhs = petsc.assemble_vector(residual)
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        rhs.scale(-1.0)
        ksp.setOperators(A)
        dvec = A.createVecRight()
        try:
            ksp.solve(rhs, dvec)
            delta = dvec.array.copy()
        except PETSc.Error:
            delta = None
        for obj in (A, rhs, dvec):
            obj.destroy()
        return delta if delta is not None and np.isfinite(delta).all() else None

    info = NewtonInfo()
    info.residual0 = info.residual = nres = res_norm()
    if nres < opts.tol_res:
        info.status, info.converged = "already", True
        return info
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(solver)
    best, best_x, status, gn_until = nres, sol.x.array.copy(), "maxit", np.inf
    while info.iterations < opts.maxit:
        x0 = sol.x.array.copy()
        if opts.method == "full":
            kinds = ["full"]
        elif opts.method == "hybrid" and nres < min(opts.switch, gn_until):
            kinds = ["full", "gn"]
        else:
            kinds = ["gn"]
        accepted = None
        for i, kind in enumerate(kinds):
            sol.x.array[:] = x0
            sol.x.scatter_forward()
            delta = direction(kind)
            if delta is None:
                continue
            floor = opts.t_full if i < len(kinds) - 1 else opts.t_min
            t = 1.0
            while t >= floor:
                sol.x.array[:] = x0 + t * delta
                sol.x.scatter_forward()
                nres1 = res_norm()
                if nres1 < (1.0 - opts.armijo * t) * nres:
                    accepted = (kind, t, delta, nres1)
                    break
                info.failed_steps += 1
                t /= 2.0
            if accepted:
                break
            if kind == "full" and len(kinds) > 1:
                gn_until = nres / 10.0
        if accepted is None:
            sol.x.array[:] = x0
            sol.x.scatter_forward()
            status = "stalled"
            break
        _, t, delta, nres = accepted
        info.iterations += 1
        info.increment = increment_norm(t * delta)
        info.residual = nres
        if nres < best:
            best, best_x = nres, sol.x.array.copy()
        if info.increment < opts.tol_du or nres < opts.tol_res:
            status = "conv"
            break
        if not np.isfinite(nres) or nres > opts.diverge:
            status = "diverged"
            break
    ksp.destroy()
    info.status, info.converged = status, status == "conv"
    if not info.converged:  # keep the best iterate; the flag records the failure
        sol.x.array[:] = best_x
        sol.x.scatter_forward()
        info.residual = best
    return info


# --------------------------------------------------------------------------------------------------
# Saddle point
# --------------------------------------------------------------------------------------------------
@dataclass
class NonlinearRMResult(RMResult):
    """:class:`~adaptdg.vms.saddle.RMResult` plus the Newton diagnostics (``newton`` = :meth:`NewtonInfo.as_dict`)."""

    newton: dict[str, Any] = field(default_factory=dict)


def _transfer(old: Any, new: Any) -> None:
    """Interpolate ``old`` (a Function on another mesh) into ``new`` (nonmatching meshes)."""
    from dolfinx import fem

    dom = new.function_space.mesh
    tdim = dom.topology.dim
    cells = np.arange(dom.topology.index_map(tdim).size_local + dom.topology.index_map(tdim).num_ghosts,
                      dtype=np.int32)
    data = fem.create_interpolation_data(new.function_space, old.function_space, cells, padding=1e-8)
    new.interpolate_nonmatching(old, cells, interpolation_data=data)


def _dofmap(sub: Any) -> tuple[Any, np.ndarray]:
    """``(collapsed space, parent dof map)`` of a sub space (dolfinx 0.11 wraps the map in a list)."""
    space, mapping = sub.collapse()
    return space, np.asarray(mapping).ravel()


class NonlinearRMSolver:
    """Nonlinear stationary residual-minimisation solver (any p, 2D)."""

    def __init__(self, problem: Any, p: int, *, eta0: float = 1.0, quadrature_degree: int = 12,
                 solver: str = "mumps", preset: Any = None, newton: dict[str, Any] | NewtonOptions | None = None) -> None:
        from adaptdg.vms.preset import resolve_preset

        self.problem, self.p, self.eta0 = problem, p, eta0
        self.preset = resolve_preset(preset)
        self.quadrature_degree, self.solver = quadrature_degree, solver
        self.newton = newton if isinstance(newton, NewtonOptions) else newton_options(self.preset, newton)

    def system(self, domain: Any, guess: tuple[Any, Any] | None = None, problem: Any = None) -> SimpleNamespace:
        """Forms and initial state of the saddle point on ``domain``: ``sol`` (mixed Function with the start state),
        ``residual`` (UFL form ``F1 + F2``), ``jacobian(kind)`` (compiled, ``"gn"`` or ``"full"``), the dof maps
        ``map_r``/``map_c``, ``increment_norm`` and the spaces. ``guess = (eps, ubar)`` (Functions on another or the
        same mesh) warm-starts, otherwise ``eps = 0`` and ``ubar = problem.initial_guess``; ``problem`` replaces
        ``self.problem`` (continuation in the diffusivity)."""
        import basix.ufl
        import ufl
        from dolfinx import fem

        p, problem = self.p, problem or self.problem
        forms = NonlinearForms(domain, problem, p, eta0=self.eta0, rhs_degree=self.quadrature_degree,
                               preset=self.preset)
        cell = domain.basix_cell()
        W = fem.functionspace(domain, basix.ufl.mixed_element(
            [basix.ufl.element("DG", cell, p), basix.ufl.element("Lagrange", cell, p)]))
        sol = fem.Function(W)
        (Vr, map_r), (Vc, map_c) = _dofmap(W.sub(0)), _dofmap(W.sub(1))
        eps0, ub0 = fem.Function(Vr), fem.Function(Vc)
        if guess is None:
            ub0.interpolate(problem.initial_guess)
        else:
            for old, new in zip(guess, (eps0, ub0), strict=True):
                if old.function_space.mesh is domain:  # same mesh: copy (interpolation would mix DG values at facets)
                    new.x.array[:] = old.x.array
                else:
                    _transfer(old, new)
        sol.x.array[map_r], sol.x.array[map_c] = eps0.x.array, ub0.x.array

        r, u = ufl.split(sol)
        w, v = ufl.TestFunctions(W)
        du = ufl.TrialFunction(W)
        dr, dub = ufl.split(du)
        f1 = forms.gram(r, w) + forms.operator(w, u)
        if forms.wind:  # legacy residual: F2 = bh_prima(u_k, v, e_k)
            f2 = forms.wind_linearisation(r, v, u)
            gn = forms.gram(dr, w) + forms.wind_linearisation(w, dub, u) + forms.wind_linearisation(dr, v, u)
        else:
            f2 = ufl.derivative(forms.operator(r, u), sol, ufl.as_vector([0 * v, v]))  # n'(eps, v; ubar)
            gn = (ufl.derivative(f1, sol, du)
                  + ufl.derivative(f2, sol, ufl.as_vector([dr, 0 * dr])))  # second derivative n'' dropped
        total = f1 + f2
        cache: dict[str, Any] = {}

        def jacobian(kind: str) -> Any:
            if kind not in cache:
                cache[kind] = fem.form(ufl.derivative(total, sol, du) if kind == "full" else gn)
            return cache[kind]

        dd = fem.Function(Vc)
        gram_du = fem.form(forms.gram(dd, dd))

        def increment_norm(delta: np.ndarray) -> float:
            dd.x.array[:] = delta[map_c]
            return float(np.sqrt(abs(fem.assemble_scalar(gram_du))))

        return SimpleNamespace(forms=forms, space=W, sol=sol, residual=total, jacobian=jacobian,
                               increment_norm=increment_norm, map_r=map_r, map_c=map_c, Vr=Vr, Vc=Vc)

    def _continuation(self, domain: Any, guess: Any, opts: NewtonOptions) -> tuple[Any, NewtonInfo | None]:
        """Solve at ``kappa = continuation_start`` and lower ``kappa`` stage by stage down to the one of the problem;
        returns the system and the info of the last stage (``(None, None)`` if a stage fails with the smallest ratio).
        The iterations of all stages are added up in ``info.iterations``."""
        from dataclasses import replace as dc_replace

        from dolfinx import fem

        target = float(self.problem.diffusivity)
        kappa, ratio, total, stages, state, prev = max(opts.continuation_start, target), opts.continuation_ratio, 0, 0, guess, None
        while True:
            s = self.system(domain, state, dc_replace(self.problem, diffusivity=kappa))
            info = newton_solve(s.sol, fem.form(s.residual), s.jacobian, s.increment_norm, opts, self.solver)
            total, stages = total + info.iterations, stages + 1
            if not info.converged:
                if prev is None or ratio < 1.2:
                    return None, None
                ratio = ratio ** 0.5  # retry from the last converged stage with a smaller decrease of kappa
                kappa, state = max(target, prev[0] / ratio), prev[1]
                continue
            state = self._state(s)
            prev = (kappa, state)
            if kappa <= target:
                break
            kappa = max(target, kappa / ratio)
        info.iterations, info.attempts = total, stages
        return s, info

    def _state(self, s: Any) -> tuple[Any, Any]:
        """``(eps, ubar)`` Functions holding the current iterate of the system ``s``."""
        from dolfinx import fem

        r_fun, ub_fun = fem.Function(s.Vr), fem.Function(s.Vc)
        r_fun.x.array[:], ub_fun.x.array[:] = s.sol.x.array[s.map_r], s.sol.x.array[s.map_c]
        return r_fun, ub_fun

    def solve(self, domain: Any, guess: tuple[Any, Any] | None = None,
              newton: NewtonOptions | None = None) -> NonlinearRMResult:
        """Solve on ``domain`` (see :meth:`system` for ``guess``)."""
        from dolfinx import fem

        opts = newton or self.newton
        s = self.system(domain, guess)
        info = newton_solve(s.sol, fem.form(s.residual), s.jacobian, s.increment_norm, opts, self.solver)
        total, attempts = info.iterations, 1
        if not info.converged and guess is not None and opts.cold_retry:  # a bad warm start: retry from the initial guess
            s2 = self.system(domain, None)
            info2 = newton_solve(s2.sol, fem.form(s2.residual), s2.jacobian, s2.increment_norm, opts, self.solver)
            total, attempts = total + info2.iterations, attempts + 1
            if info2.converged or info2.residual < info.residual:
                s, info = s2, info2
        ndofs = s.space.dofmap.index_map.size_global * s.space.dofmap.index_map_bs
        if (not info.converged and opts.continuation and hasattr(self.problem, "diffusivity")
                and ndofs <= opts.continuation_max_dofs):
            s3, info3 = self._continuation(domain, None, opts)
            if info3 is not None:
                total, attempts = total + info3.iterations, attempts + info3.attempts
                if info3.converged:
                    s, info = s3, info3
        info.iterations, info.attempts = total, max(info.attempts, attempts)
        r_fun, ub_fun = fem.Function(s.Vr, name="eps"), fem.Function(s.Vc, name="ubar")
        r_fun.x.array[:], ub_fun.x.array[:] = s.sol.x.array[s.map_r], s.sol.x.array[s.map_c]
        forms = s.forms
        ind = np.abs(cell_values(domain, forms.gram(r_fun, r_fun, _dg0_test(domain))))
        return NonlinearRMResult(r_fun, ub_fun, int(ndofs), ind, float(np.sqrt(ind.sum())), forms, info.as_dict())


# --------------------------------------------------------------------------------------------------
# Multiscale reconstruction (eqs. 61-62) and the dG solution
# --------------------------------------------------------------------------------------------------
def solve_dg(forms: NonlinearForms, u0: Any, opts: NewtonOptions, solver: str = "mumps") -> tuple[Any, NewtonInfo]:
    """Nonlinear dG solution ``theta`` (eq. 44) by Newton from ``u0`` (a DG(p) Function); not necessarily the same
    root as ``ubar + utilde`` (Several roots can coexist)."""
    import ufl
    from dolfinx import fem

    theta = fem.Function(u0.function_space, name="theta")
    theta.x.array[:] = u0.x.array
    w, du = ufl.TestFunction(theta.function_space), ufl.TrialFunction(theta.function_space)
    res = forms.operator(w, theta)
    cache: dict[str, Any] = {}

    def jacobian(kind: str) -> Any:
        if kind not in cache:
            form = forms.wind_linearisation(w, du, theta) if (kind == "gn" and forms.wind) else ufl.derivative(res, theta, du)
            cache[kind] = fem.form(form)
        return cache[kind]

    dd = fem.Function(theta.function_space)
    gram_du = fem.form(forms.gram(dd, dd))

    def increment_norm(delta: np.ndarray) -> float:
        dd.x.array[:] = delta
        return float(np.sqrt(abs(fem.assemble_scalar(gram_du))))

    info = newton_solve(theta, fem.form(res), jacobian, increment_norm, opts, solver)
    return theta, info


def reconstruct_nonlinear(result: NonlinearRMResult, *, adjoint_ordering: str = "paper", solver: str = "mumps",
                          newton: NewtonOptions | None = None) -> Any:
    """``utilde`` (eq. 61), ``ucheck`` (eq. 62), ``phi = ubar + ucheck`` and the dG solution ``theta`` (Newton from
    ``ubar + utilde`` with the undamped exact-Jacobian Newton in both presets (the same iteration as the hybrid one
    for the paper preset), because the legacy damped dG iteration fails on most levels;
    ``theta`` is ``None`` when that Newton iteration fails). Returns a
    :class:`~adaptdg.vms.reconstruction.VMSResult` (``u_full = ubar + utilde``, an approximation of ``theta``)."""
    if adjoint_ordering not in ("paper", "legacy"):
        raise ValueError("adjoint_ordering must be 'paper' or 'legacy'")
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc
    from petsc4py import PETSc

    from adaptdg.vms.reconstruction import VMSResult

    forms, eps = result.forms, result.r
    space = eps.function_space
    ub = fem.Function(space)
    ub.interpolate(result.ubar)
    v, u = ufl.TestFunction(space), ufl.TrialFunction(space)
    g = forms.gram(v, eps)
    adj = forms.linearised(eps, v, ub) if adjoint_ordering == "paper" else forms.linearised(v, eps, ub)
    A = petsc.assemble_matrix(fem.form(forms.linearised(v, u, ub)))
    A.assemble()
    ksp = PETSc.KSP().create(space.mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(solver)
    out: dict[str, Any] = {}
    for name, form in {"utilde": g, "ucheck": g + adj}.items():
        bvec = petsc.assemble_vector(fem.form(form))
        bvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fun = fem.Function(space, name=name)
        ksp.solve(bvec, fun.x.petsc_vec)
        fun.x.scatter_forward()
        bvec.destroy()
        out[name] = fun
    ksp.destroy()
    A.destroy()
    full = fem.Function(space, name="u_full")
    full.x.array[:] = ub.x.array + out["utilde"].x.array
    phi = fem.Function(space, name="phi")
    phi.x.array[:] = ub.x.array + out["ucheck"].x.array
    theta, info = solve_dg(forms, full, newton or NewtonOptions(method="full"), solver)
    return VMSResult(result.ubar, out["utilde"], out["ucheck"], phi, theta if info.converged else None, full,
                     adjoint_ordering)


# --------------------------------------------------------------------------------------------------
# Diagnostics along a line (shock position and width of the single-component Burgers case)
# --------------------------------------------------------------------------------------------------
def line_values(func: Any, xs: np.ndarray, y: float) -> np.ndarray:
    """Values of a 2D Function on the horizontal line ``y`` at the abscissae ``xs`` (NaN outside the mesh)."""
    from dolfinx import geometry

    domain = func.function_space.mesh
    pts = np.zeros((len(xs), 3))
    pts[:, 0], pts[:, 1] = xs, y
    tree = geometry.bb_tree(domain, domain.topology.dim)
    coll = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(tree, pts), pts)
    vals = np.full(len(xs), np.nan)
    for i in range(len(xs)):
        cells = coll.links(i)
        if len(cells):
            vals[i] = func.eval(pts[i:i + 1], cells[:1])[0]
    return vals


def shock_diagnostics(ubar: Any, y: float = 0.75, samples: int = 2001, window: float = 0.05) -> dict[str, float]:
    """Position (zero crossing of ``u`` nearest to ``x = 1/2``), steepest slope ``min u_x`` and full width at half
    maximum of ``-u_x`` of the shock of ``ubar`` along the line ``y``. The steepest point is first found on
    ``samples`` points of ``[0, 1]``; the slope is then taken on ``2 samples`` points of a narrow window around it
    and the width on ``2 samples`` points of ``[x - window, x + window]`` (a coarse line underestimates a layer
    thinner than its spacing, e.g. ``kappa = 1e-4`` against ``dx = 5e-4``)."""
    xs = np.linspace(0.0, 1.0, samples)
    u = line_values(ubar, xs, y)
    k = int(np.nanargmin(np.gradient(u, xs)))

    def profile(half: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xf = np.linspace(max(0.0, xs[k] - half), min(1.0, xs[k] + half), 2 * samples)
        uf = line_values(ubar, xf, y)
        return xf, uf, np.gradient(uf, xf)

    xn, un, uxn = profile(0.1 * window)
    xf, uf, uxf = profile(window)
    kn = int(np.nanargmin(uxn))
    kf = int(np.nanargmin(uxf))
    cross = xf[:-1][np.diff(np.sign(uf)) != 0]
    if not len(cross):
        cross = xs[:-1][np.diff(np.sign(u)) != 0]
    pos = float(cross[np.argmin(abs(cross - 0.5))]) if len(cross) else float("nan")
    half = -uxf < 0.5 * -uxn[kn]
    left = xf[:kf][half[:kf]]
    right = xf[kf:][half[kf:]]
    width = float(right[0] - left[-1]) if len(left) and len(right) else float("nan")
    ncross = int(np.count_nonzero(np.diff(np.sign(u)) != 0))
    return {"position": pos, "max_slope": float(uxn[kn]), "width_fwhm": width, "crossings": ncross,
            "u_min": float(np.nanmin(u)), "u_max": float(np.nanmax(u))}


# --------------------------------------------------------------------------------------------------
# Adaptive loop
# --------------------------------------------------------------------------------------------------
class NonlinearAdaptiveLoop:
    """Adaptive (or uniform) loop for a :class:`~adaptdg.vms.problems.NonlinearProblem`; results in ``history``.

    Uses ``LoopSettings`` (``newton`` = dict of :class:`NewtonOptions` overrides, ``newton_fallback``: when the
    preset's Newton fails, retry the level with the undamped hybrid solver). Rows carry ``newton_iterations``,
    ``newton_failed_steps``, ``newton_residual``, ``newton_increment`` (last ``|du|_Vh``), ``newton_converged`` (0/1), ``newton_status``, the nodal
    ``umin``/``umax`` of ``ubar``, ``theta_converged`` (with ``reconstruct``: the dG Newton iteration for ``theta``
    converged; otherwise the ``*_theta`` columns are NaN) and, for problems with an exact solution, the error columns of the linear loop.
    """

    def __init__(self, problem: Any, settings: Any, domain: Any = None) -> None:
        if settings.refinement not in ("adaptive", "uniform"):
            raise ValueError("refinement must be 'adaptive' or 'uniform'")
        if settings.method != "rm":
            raise ValueError("the nonlinear loop only supports method 'rm'")
        self.problem, self.settings = problem, settings
        self.domain = domain if domain is not None else problem.initial_mesh(settings.initial_mesh_size)
        self.history: list[dict[str, Any]] = []
        self.result: Any = None
        self.vms: Any = None

    def run(self, progress: Callable[[str], None] | None = print) -> list[dict[str, Any]]:
        from adaptdg.marking import fixed_fraction_bands
        from adaptdg.vms.loop import mark_cells
        from adaptdg.vms.norms import error_targets, history_columns
        from adaptdg.vms.preset import resolve_preset
        from adaptdg.vms.refine import refine_marked, refine_uniform

        s, problem = self.settings, self.problem
        if s.error_targets not in ("both", "exact"):
            raise ValueError("error_targets must be 'both' or 'exact'")
        theta = s.theta_ref if s.theta_ref is not None else 0.25
        preset = resolve_preset(s.preset, **(s.preset_overrides or {}))
        qdeg = s.quadrature_degree if preset.quadrature_degree is None else preset.quadrature_degree
        ordering = s.adjoint_ordering if preset.adjoint_ordering is None else preset.adjoint_ordering
        ratio = theta if preset.marking_ratio is None else preset.marking_ratio
        targets = ("exact", "projection") if s.error_targets == "both" else ("exact",)
        solver = NonlinearRMSolver(problem, s.p, eta0=s.eta0, quadrature_degree=qdeg, solver=s.solver, preset=preset,
                                   newton=s.newton)
        guess = None
        for level in range(s.max_levels):
            t0 = time.time()
            ncells = self.domain.topology.index_map(self.domain.topology.dim).size_local
            self.result = res = solver.solve(self.domain, guess)
            used_fallback = 0
            if not res.newton["converged"] and s.newton_fallback and solver.newton.method != "hybrid":
                alt = solver.solve(self.domain, guess, newton=newton_options("paper"))
                if alt.newton["converged"]:
                    self.result = res = alt
                    used_fallback = 1
            guess = (res.r, res.ubar)
            cols: dict[str, Any] = {"E": res.E}
            self.vms = None
            if s.reconstruct:
                self.vms = reconstruct_nonlinear(res, adjoint_ordering=ordering, solver=s.solver)
            if problem.has_exact:
                errs = error_targets(res, problem, s.p, degree=qdeg, vms=self.vms, targets=targets)
                cols.update(history_columns(errs))
            nodal = res.ubar.x.array
            info = res.newton
            row = {"level": level, "ndofs": res.ndofs, "ncells": int(ncells), **cols,
                   "newton_iterations": info["iterations"], "newton_failed_steps": info["failed_steps"],
                   "newton_residual": info["residual"], "newton_increment": info["increment"],
                   "newton_converged": int(info["converged"]),
                   "newton_status": info["status"], "newton_fallback": used_fallback,
                   "umin": float(nodal.min()), "umax": float(nodal.max())}
            if s.reconstruct:
                row["theta_converged"] = int(self.vms.theta is not None)
            self.history.append({**row, "time": time.time() - t0})
            if progress:
                err = f" L2={cols['L2']:.3e} Vh={cols['Vh']:.3e}" if "L2" in cols else ""
                flag = "" if info["converged"] else f" NEWTON {info['status'].upper()}"
                progress(f"level {level + 1}/{s.max_levels} p={s.p} ndofs={res.ndofs} E={res.E:.3e}{err} "
                         f"newton={info['iterations']}it |R|={info['residual']:.1e}{flag} "
                         f"ubar in [{row['umin']:.3f}, {row['umax']:.3f}] t={time.time() - t0:.1f}s")
            if (level + 1 == s.max_levels or res.E <= s.tol or (s.dofmax is not None and res.ndofs >= s.dofmax)):
                break
            if s.refinement == "uniform":
                self.domain = refine_uniform(self.domain)
            else:
                marked = (fixed_fraction_bands(res.indicators, ratio) if preset.marking == "fixed_fraction"
                          else mark_cells(res.indicators, theta, s.tolref))
                self.domain = refine_marked(self.domain, marked)[0]
        return self.history

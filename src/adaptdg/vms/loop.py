"""Adaptive (or uniform) stationary loop: solve -> estimate -> mark -> refine.

Marking is the Doerfler rule of the paper (``adaptdg.marking.dorfler_union`` with all cells
eligible) on the indicators ``g_K`` of :mod:`adaptdg.vms.saddle`; theta_ref defaults to 0.25
(2D) and 0.125 (3D), the tie tolerance to 0.2. ``settings.preset = "legacy"`` switches to the choices
of the original implementation (:mod:`adaptdg.vms.preset`), among them fixed-fraction marking.
Errors are reported against the exact solution and against its DG(p) L2 projection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LoopSettings:
    """Numerical settings of the stationary loop (``None`` theta_ref -> dimension default)."""

    p: int = 1
    theta_ref: float | None = None
    tolref: float = 0.2
    eta0: float = 1.0
    max_levels: int = 8
    dofmax: int | None = None
    tol: float = 0.0
    solver: str = "mumps"
    quadrature_degree: int = 12
    refinement: str = "adaptive"  # "adaptive" | "uniform"
    initial_mesh_size: int = 4
    reconstruct: bool = False  # add the errors of theta, u = ubar + utilde and phi to each row
    adjoint_ordering: str = "paper"  # "paper" b(eps', v) | "legacy" b(v, eps')
    preset: Any = "paper"  # "paper" | "legacy" | Preset (see adaptdg.vms.preset)
    preset_overrides: dict[str, Any] | None = None  # Preset fields to change, e.g. {"gram_l2": True}
    error_targets: str = "both"  # "both" | "exact": "both" adds the *_proj columns (L2 projection of u_exact)
    method: str = "rm"  # "rm" residual minimisation | "galerkin" plain CG(p) baseline (uniform refinement only)
    compare_galerkin: bool = False  # "rm" only: add L2_cg, Vh_cg (and *_proj) of CG(p) Galerkin on the same meshes


def mark_cells(indicators: np.ndarray, theta: float, tolref: float) -> np.ndarray:
    """Doerfler marking of the paper (legacy REF_TYPE 1) on squared per-cell indicators."""
    from adaptdg.marking import dorfler_union

    return dorfler_union({"e": indicators}, None, 0.0, theta, 0.0, tolref,
                         eligibility=np.ones(indicators.size, dtype=bool))


class AdaptiveStationaryLoop:
    """Run the loop for a :class:`~adaptdg.vms.problems.Problem`; results in ``history``."""

    def __init__(self, problem: Any, settings: LoopSettings, domain: Any = None) -> None:
        if settings.refinement not in ("adaptive", "uniform"):
            raise ValueError("refinement must be 'adaptive' or 'uniform'")
        if settings.method not in ("rm", "galerkin"):
            raise ValueError("method must be 'rm' or 'galerkin'")
        if settings.method == "galerkin" and settings.refinement != "uniform":
            raise ValueError("method 'galerkin' has no error indicator: use refinement 'uniform'")
        self.problem, self.settings = problem, settings
        self.domain = domain if domain is not None else problem.initial_mesh(settings.initial_mesh_size)
        self.history: list[dict[str, float]] = []
        self.result: Any = None
        self.vms: Any = None  # VMSResult of the last level when ``settings.reconstruct``

    def run(self, progress: Callable[[str], None] | None = print) -> list[dict[str, float]]:
        from adaptdg.marking import fixed_fraction_bands
        from adaptdg.vms.galerkin import galerkin_errors, solve_galerkin
        from adaptdg.vms.norms import error_targets, history_columns
        from adaptdg.vms.preset import resolve_preset
        from adaptdg.vms.reconstruction import reconstruct
        from adaptdg.vms.refine import refine_marked, refine_uniform
        from adaptdg.vms.saddle import RMSolver

        s = self.settings
        theta = s.theta_ref if s.theta_ref is not None else (0.25 if self.problem.dim == 2 else 0.125)
        if s.error_targets not in ("both", "exact"):
            raise ValueError("error_targets must be 'both' or 'exact'")
        preset = resolve_preset(s.preset, **(s.preset_overrides or {}))
        qdeg = s.quadrature_degree if preset.quadrature_degree is None else preset.quadrature_degree
        ordering = s.adjoint_ordering if preset.adjoint_ordering is None else preset.adjoint_ordering
        ratio = theta if preset.marking_ratio is None else preset.marking_ratio
        targets = ("exact", "projection") if s.error_targets == "both" else ("exact",)
        rm = s.method == "rm"
        solver = RMSolver(self.problem, s.p, eta0=s.eta0, quadrature_degree=qdeg,
                          solver=s.solver, preset=preset)
        gopts = {"eta0": s.eta0, "quadrature_degree": qdeg, "solver": s.solver, "preset": preset}
        for level in range(s.max_levels):
            t0 = time.time()
            ncells = self.domain.topology.index_map(self.domain.topology.dim).size_local
            if rm:
                self.result = res = solver.solve(self.domain)
                if s.reconstruct:
                    self.vms = reconstruct(res, adjoint_ordering=ordering, solver=s.solver)
                errs = error_targets(res, self.problem, s.p, degree=qdeg,
                                     vms=self.vms if s.reconstruct else None, targets=targets)
                cols = {"E": res.E, **history_columns(errs)}
            else:  # plain Galerkin: no estimator, hence no "E" column
                self.result = res = solve_galerkin(self.problem, self.domain, s.p, **gopts)
                cols = history_columns(galerkin_errors(res, self.problem, s.p, degree=qdeg, targets=targets))
            if rm and s.compare_galerkin:
                cg = solve_galerkin(self.problem, self.domain, s.p, **gopts)
                cg_errs = galerkin_errors(cg, self.problem, s.p, degree=qdeg, targets=targets)
                cols.update(history_columns({t: {"cg": v["ubar"]} for t, v in cg_errs.items()}))
            l2, vh = cols["L2"], cols["Vh"]
            row = {"level": level, "ndofs": res.ndofs, "ncells": int(ncells), **cols}
            self.history.append({**row, "time": time.time() - t0})
            if progress:
                extra = f" Vh_phi={row['Vh_phi']:.3e}" if rm and s.reconstruct else ""
                extra += f" Vh_cg={row['Vh_cg']:.3e}" if rm and s.compare_galerkin else ""
                est = f"E={res.E:.3e} " if rm else "galerkin "
                progress(f"level {level + 1}/{s.max_levels} p={s.p} ndofs={res.ndofs} "
                         f"{est}L2={l2:.3e} Vh={vh:.3e}{extra} t={time.time() - t0:.1f}s")
            if (level + 1 == s.max_levels or (rm and res.E <= s.tol)
                    or (s.dofmax is not None and res.ndofs >= s.dofmax)):
                break
            if s.refinement == "uniform":
                self.domain = refine_uniform(self.domain)
            else:
                if preset.marking == "fixed_fraction":
                    marked = fixed_fraction_bands(res.indicators, ratio)
                else:
                    marked = mark_cells(res.indicators, theta, s.tolref)
                self.domain = refine_marked(self.domain, marked)[0]
        return self.history


def fitted_slope(history: list[dict[str, float]], key: str, dim: int, span: float = 4.0) -> float:
    """Slope of ``log(key)`` versus ``log(ndofs**(1/dim))`` over the levels whose ndofs is within
    a factor ``span`` of the last one (at least three levels): adaptive histories are stepped."""
    final = history[-1]["ndofs"]
    h = [e for e in history if e["ndofs"] * span >= final]
    h = h if len(h) >= 3 else history[-3:]
    x = np.log([e["ndofs"] ** (1.0 / dim) for e in h])
    return float(np.polyfit(x, np.log([e[key] for e in h]), 1)[0])

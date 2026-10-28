"""Adaptive (or uniform) stationary loop: solve -> estimate -> mark -> refine.

Marking is the Doerfler rule of the paper (``adaptdg.marking.dorfler_union`` with all cells
eligible) on the indicators ``g_K`` of :mod:`adaptdg.vms.saddle`; theta_ref defaults to 0.25
(2D) and 0.125 (3D), the tie tolerance to 0.2.
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
        self.problem, self.settings = problem, settings
        self.domain = domain if domain is not None else problem.initial_mesh(settings.initial_mesh_size)
        self.history: list[dict[str, float]] = []
        self.result: Any = None
        self.vms: Any = None  # VMSResult of the last level when ``settings.reconstruct``

    def run(self, progress: Callable[[str], None] | None = print) -> list[dict[str, float]]:
        from adaptdg.vms.norms import error_norms, error_norms_vms
        from adaptdg.vms.reconstruction import reconstruct
        from adaptdg.vms.refine import refine_marked, refine_uniform
        from adaptdg.vms.saddle import RMSolver

        s = self.settings
        theta = s.theta_ref if s.theta_ref is not None else (0.25 if self.problem.dim == 2 else 0.125)
        solver = RMSolver(self.problem, s.p, eta0=s.eta0, quadrature_degree=s.quadrature_degree,
                          solver=s.solver)
        for level in range(s.max_levels):
            t0 = time.time()
            self.result = res = solver.solve(self.domain)
            ncells = self.domain.topology.index_map(self.domain.topology.dim).size_local
            l2, vh = error_norms(res, self.problem, s.p, degree=s.quadrature_degree)
            row = {"level": level, "ndofs": res.ndofs, "ncells": int(ncells),
                   "E": res.E, "L2": l2, "Vh": vh}
            if s.reconstruct:
                self.vms = reconstruct(res, adjoint_ordering=s.adjoint_ordering, solver=s.solver)
                errs = error_norms_vms(res, self.vms, self.problem, s.p, degree=s.quadrature_degree)
                for name in ("theta", "u", "phi"):
                    row[f"L2_{name}"], row[f"Vh_{name}"] = errs[name]
            self.history.append({**row, "time": time.time() - t0})
            if progress:
                extra = f" Vh_phi={row['Vh_phi']:.3e}" if s.reconstruct else ""
                progress(f"level {level + 1}/{s.max_levels} p={s.p} ndofs={res.ndofs} "
                         f"E={res.E:.3e} L2={l2:.3e} Vh={vh:.3e}{extra} t={time.time() - t0:.1f}s")
            if (level + 1 == s.max_levels or res.E <= s.tol
                    or (s.dofmax is not None and res.ndofs >= s.dofmax)):
                break
            if s.refinement == "uniform":
                self.domain = refine_uniform(self.domain)
            else:
                self.domain = refine_marked(self.domain, mark_cells(res.indicators, theta, s.tolref))[0]
        return self.history


def fitted_slope(history: list[dict[str, float]], key: str, dim: int, span: float = 4.0) -> float:
    """Slope of ``log(key)`` versus ``log(ndofs**(1/dim))`` over the levels whose ndofs is within
    a factor ``span`` of the last one (at least three levels): adaptive histories are stepped."""
    final = history[-1]["ndofs"]
    h = [e for e in history if e["ndofs"] * span >= final]
    h = h if len(h) >= 3 else history[-3:]
    x = np.log([e["ndofs"] ** (1.0 / dim) for e in h])
    return float(np.polyfit(x, np.log([e[key] for e in h]), 1)[0])

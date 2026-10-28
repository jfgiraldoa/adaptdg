"""Standard conforming Galerkin baseline ``CG(p)`` (the legacy ``test_poisson_FEM_solution``).

Plain Galerkin for ``-div(kappa grad u) + beta . grad u = f`` with strongly imposed Dirichlet data on the tagged
boundary, on the same :class:`~adaptdg.vms.problems.Problem` classes, meshes and error norms as the
residual-minimisation solver (the Vh norm is the Gram form of :mod:`adaptdg.vms.dgforms`, so the numbers are
comparable). For pure diffusion this is the classical Galerkin method; with advection it is **not stabilised** (no
SUPG), so it oscillates when the Peclet number is large and is only meant as a baseline. There is no a posteriori
estimator: the legacy baseline used none, so the loop refines uniformly. Serial runs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptdg.vms.dgforms import DGForms, _flux
from adaptdg.vms.norms import TARGETS, _norms, _quadrature_forms, project_exact


@dataclass
class GalerkinResult:
    """Output of one Galerkin solve (``u`` lives in ``CG(p)``; ``forms`` carries the quadrature and the penalty)."""

    u: Any
    ndofs: int
    forms: DGForms


def solve_galerkin(problem: Any, domain: Any, p: int, *, eta0: float = 1.0, quadrature_degree: int = 12,
                   solver: str = "mumps", preset: Any = None) -> GalerkinResult:
    """Solve the Galerkin problem on ``CG(p)`` (``eta0`` and ``preset`` only set the forms used for the norms)."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc

    from adaptdg.vms.meshes import tag_boundary

    forms = DGForms(domain, problem, p, eta0=eta0, rhs_degree=quadrature_degree, preset=preset)
    space = fem.functionspace(domain, ("Lagrange", p))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    a = ufl.inner(_flux(forms.kappa, ufl.grad(u)), ufl.grad(v)) * forms.dx
    if forms.beta is not None:
        a += v * ufl.dot(forms.beta, ufl.grad(u)) * forms.dx  # not stabilised
    f = problem.source(forms.x)
    rhs = v * (fem.Constant(domain, 0.0) if f is None else f) * forms.dx_r
    fdim = domain.topology.dim - 1
    tags = tag_boundary(domain, problem.boundary_markers())
    facets = tags.indices[np.isin(tags.values, problem.dirichlet_tags)]
    uD = fem.Function(space)
    uD.interpolate(fem.Expression(problem.dirichlet(forms.x), space.element.interpolation_points))
    bc = fem.dirichletbc(uD, fem.locate_dofs_topological(space, fdim, facets))
    opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": solver}
    sol = petsc.LinearProblem(a, rhs, bcs=[bc], petsc_options=opts, petsc_options_prefix="vms_cg_").solve()
    return GalerkinResult(sol, int(space.dofmap.index_map.size_global), forms)


def galerkin_errors(result: GalerkinResult, problem: Any, p: int, *, degree: int = 8,
                    targets: tuple[str, ...] = TARGETS) -> dict[str, dict[str, tuple[float, float]]]:
    """``{target: {"ubar": (L2, Vh)}}`` as :func:`adaptdg.vms.norms.error_targets` (same targets and quadrature)."""
    from dolfinx import fem

    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown error targets {sorted(unknown)}; choose from {TARGETS}")
    hq = _quadrature_forms(result, problem, p, degree)
    out = {}
    for target in targets:
        if target == "exact":
            ref = problem.exact(hq.x)
        else:
            ref = project_exact(hq, problem, fem.functionspace(result.u.function_space.mesh, ("DG", p)))
        out[target] = {"ubar": _norms(hq, ref, result.u)}
    return out

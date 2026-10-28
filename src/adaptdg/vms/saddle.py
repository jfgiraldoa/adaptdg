"""Residual-minimisation saddle-point solver (Giraldo & Calo 2023, CMAME 417, 116285, eq. 18).

Unknowns ``(r, ubar)`` in ``DG(p) x CG(p)``: ``g(w, r) + b(w, ubar) + b(r, v) = l(w)`` for all
``(w, v)``, with ``r`` the residual representation (eq. 17). The element indicator is
``g_K = ||r||^2_{Vh,K}`` and the global estimator ``E = sqrt(sum_K g_K)``. Serial runs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptdg.vms.dgforms import DGForms


@dataclass
class RMResult:
    """Output of one saddle-point solve."""

    r: Any
    ubar: Any
    ndofs: int
    indicators: np.ndarray
    E: float
    forms: DGForms


class RMSolver:
    """Linear stationary residual-minimisation solver (any p, 2D and 3D)."""

    def __init__(self, problem: Any, p: int, *, eta0: float = 1.0,
                 quadrature_degree: int = 12, solver: str = "mumps", preset: Any = None) -> None:
        self.problem, self.p, self.eta0 = problem, p, eta0
        self.preset = preset
        self.quadrature_degree, self.solver = quadrature_degree, solver

    def system(self, domain: Any) -> tuple[Any, Any, Any, DGForms]:
        """Return ``(a, L, space, forms)``: eq. (18) as UFL forms on ``DG(p) x CG(p)``."""
        import basix.ufl
        import ufl
        from dolfinx import fem

        p = self.p
        forms = DGForms(domain, self.problem, p, eta0=self.eta0, rhs_degree=self.quadrature_degree,
                         preset=self.preset)
        cell = domain.basix_cell()
        element = basix.ufl.mixed_element(
            [basix.ufl.element("DG", cell, p), basix.ufl.element("Lagrange", cell, p)]
        )
        space = fem.functionspace(domain, element)
        r, u = ufl.TrialFunctions(space)
        w, v = ufl.TestFunctions(space)
        a = forms.gram(w, r) + forms.b(w, u) + forms.b(r, v)
        return a, forms.rhs(w), space, forms

    def solve(self, domain: Any) -> RMResult:
        from dolfinx.fem import petsc

        a, rhs, space, forms = self.system(domain)
        opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": self.solver}
        sol = petsc.LinearProblem(a, rhs, petsc_options=opts, petsc_options_prefix="vms_rm_").solve()
        ind = cell_values(domain, forms.gram(sol.sub(0), sol.sub(0), _dg0_test(domain)))
        ind = np.abs(ind)
        ndofs = space.dofmap.index_map.size_global * space.dofmap.index_map_bs
        return RMResult(sol.sub(0).collapse(), sol.sub(1).collapse(), int(ndofs), ind,
                        float(np.sqrt(ind.sum())), forms)


def _dg0_test(domain: Any) -> Any:
    import ufl
    from dolfinx import fem

    return ufl.TestFunction(fem.functionspace(domain, ("DG", 0)))


def cell_values(domain: Any, form: Any) -> np.ndarray:
    """Assemble a form linear in a DG0 test function into an array indexed by local cell."""
    from dolfinx import fem

    vec = fem.assemble_vector(fem.form(form))
    v0 = form.arguments()[0].ufl_function_space()
    ncells = domain.topology.index_map(domain.topology.dim).size_local
    return vec.array[np.asarray(v0.dofmap.list)[:ncells, 0]]

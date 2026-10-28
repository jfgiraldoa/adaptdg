"""Variational multiscale quantities of the paper (Giraldo & Calo 2023, CMAME 417, 116285, section 4).

From the saddle-point solution ``(eps', ubar)`` of :mod:`adaptdg.vms.saddle` (eq. 18, ``eps'`` is the
residual representation) every quantity is the solution of one dG problem on ``DG(p)`` with the same
matrix ``b_h(v, .)``, so a single factorisation serves all right-hand sides:

  * fine scale ``utilde``  (eq. 30):  ``b(v, utilde) = g(v, eps')``; ``u = ubar + utilde`` is the full-scale
    solution and equals the dG solution ``theta`` (Remark 1; checked numerically);
  * adjoint reconstruction ``ucheck`` (eq. 32): ``b(v, ucheck) = g(v, eps') + b(eps', v)``,
    ``phi = ubar + ucheck`` (``ucheck = utilde + uacute`` with ``b(v, uacute) = b(eps', v)``, eq. 35);
  * pure dG baseline ``theta`` (eq. 6): ``b(v, theta) = l(v)``.

``adjoint_ordering="paper"`` uses ``b(eps', v)`` exactly as in eq. (32)/(35) (``eps'`` in the test slot,
``v`` in the trial slot, i.e. the adjoint operator); ``"legacy"`` uses ``b(v, eps')``, which the original
implementation used and which differs only when advection is present (``b`` is symmetric otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptdg.vms.saddle import RMResult


@dataclass
class VMSResult:
    """Multiscale quantities; ``ubar`` is in ``CG(p)``, all others live in ``DG(p)``."""

    ubar: Any
    utilde: Any
    ucheck: Any
    phi: Any
    theta: Any
    u_full: Any
    adjoint_ordering: str = "paper"


def reconstruct(result: RMResult, *, adjoint_ordering: str = "paper",
                solver: str = "mumps") -> VMSResult:
    """Build ``utilde``, ``ucheck``, ``phi``, ``theta`` and ``u_full`` from an :class:`RMResult`."""
    if adjoint_ordering not in ("paper", "legacy"):
        raise ValueError("adjoint_ordering must be 'paper' or 'legacy'")
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc
    from petsc4py import PETSc

    forms, eps = result.forms, result.r
    space = eps.function_space
    v, u = ufl.TestFunction(space), ufl.TrialFunction(space)
    g = forms.gram(v, eps)
    adj = forms.b(eps, v) if adjoint_ordering == "paper" else forms.b(v, eps)
    rhs = {"utilde": g, "ucheck": g + adj, "theta": forms.rhs(v)}

    A = petsc.assemble_matrix(fem.form(forms.b(v, u)))
    A.assemble()
    ksp = PETSc.KSP().create(space.mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType(solver)
    out: dict[str, Any] = {}
    for name, form in rhs.items():
        bvec = petsc.assemble_vector(fem.form(form))
        bvec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fun = fem.Function(space, name=name)
        ksp.solve(bvec, fun.x.petsc_vec)
        fun.x.scatter_forward()
        bvec.destroy()
        out[name] = fun
    ksp.destroy()
    A.destroy()

    ubar_dg = fem.Function(space)
    ubar_dg.interpolate(result.ubar)
    full = fem.Function(space, name="u_full")
    full.x.array[:] = ubar_dg.x.array + out["utilde"].x.array
    phi = fem.Function(space, name="phi")
    phi.x.array[:] = ubar_dg.x.array + out["ucheck"].x.array
    return VMSResult(result.ubar, out["utilde"], out["ucheck"], phi, out["theta"], full,
                     adjoint_ordering)

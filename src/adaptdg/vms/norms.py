"""Errors of ``ubar`` and, with the reconstruction of :mod:`adaptdg.vms.reconstruction`, of ``theta``,
``u`` and ``phi`` (L2 and the dG norm of eq. 12), against two targets:

* ``"exact"``: the exact solution (the true error);
* ``"projection"``: its L2 projection onto DG(p), which is what the original implementation measured.

The exact solution is evaluated symbolically with a high quadrature degree, so singular data
do not create an artificial error floor at high polynomial degree; the projection uses the same
quadrature degree (MUMPS LU, serial).
"""

from __future__ import annotations

from typing import Any

import numpy as np

TARGETS = ("exact", "projection")
FIELDS = ("ubar", "theta", "u", "phi")


def _norms(hq: Any, target: Any, field: Any) -> tuple[float, float]:
    from dolfinx import fem

    e = target - field
    l2 = fem.assemble_scalar(fem.form(e * e * hq.dx))
    vh = fem.assemble_scalar(fem.form(hq.gram(e, e)))
    return float(np.sqrt(abs(l2))), float(np.sqrt(abs(vh)))


def _quadrature_forms(result: Any, problem: Any, p: int, degree: int) -> Any:
    from adaptdg.vms.dgforms import DGForms

    hq = DGForms(result.forms.domain, problem, p, degree=degree, preset=result.forms.preset)
    hq.eta = result.forms.eta  # same penalty as the solver
    return hq


def project_exact(hq: Any, problem: Any, space: Any) -> Any:
    """L2 projection of ``problem.exact`` onto ``space`` (DG(p)) with the quadrature of ``hq``."""
    import ufl
    from dolfinx.fem import petsc

    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    return petsc.LinearProblem(u * v * hq.dx, problem.exact(hq.x) * v * hq.dx, petsc_options=opts,
                               petsc_options_prefix="vms_proj_").solve()


def error_targets(result: Any, problem: Any, p: int, *, degree: int = 8, vms: Any = None,
                  targets: tuple[str, ...] = TARGETS) -> dict[str, dict[str, tuple[float, float]]]:
    """``{target: {field: (L2, Vh)}}`` for ``ubar`` and, when ``vms`` is given, ``theta``, ``u``, ``phi``."""
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown error targets {sorted(unknown)}; choose from {TARGETS}")
    hq = _quadrature_forms(result, problem, p, degree)
    fields = {"ubar": result.ubar}
    if vms is not None:
        fields = {"ubar": vms.ubar, "theta": vms.theta, "u": vms.u_full, "phi": vms.phi}
    out = {}
    for target in targets:
        ref = problem.exact(hq.x) if target == "exact" else project_exact(hq, problem, result.r.function_space)
        out[target] = {name: _norms(hq, ref, fun) if fun is not None else (float("nan"), float("nan"))
                       for name, fun in fields.items()}  # None: field not available (e.g. dG Newton failed)
    return out


def error_norms(result: Any, problem: Any, p: int, *, degree: int = 8,
                target: str = "exact") -> tuple[float, float]:
    """Return ``(||e||_0, ||e||_Vh)`` for ``e = target - ubar`` (needs ``problem.exact``)."""
    return error_targets(result, problem, p, degree=degree, targets=(target,))[target]["ubar"]


def error_norms_vms(result: Any, vms: Any, problem: Any, p: int, *, degree: int = 8,
                    target: str = "exact") -> dict[str, tuple[float, float]]:
    """``(L2, Vh)`` errors of ``ubar``, ``theta`` (dG), ``u`` (= ubar + utilde) and ``phi`` (adjoint
    reconstruction, ``vms`` from :func:`adaptdg.vms.reconstruction.reconstruct`)."""
    return error_targets(result, problem, p, degree=degree, vms=vms, targets=(target,))[target]


def history_columns(errors: dict[str, dict[str, tuple[float, float]]]) -> dict[str, float]:
    """Flatten :func:`error_targets` into history columns.

    Against the exact solution the columns are ``L2, Vh`` (``ubar``) and ``L2_<f>, Vh_<f>``
    (``f`` = theta, u, phi); against the projection the same names with the suffix ``_proj``.
    """
    row: dict[str, float] = {}
    for target, suffix in (("exact", ""), ("projection", "_proj")):
        for name, (l2, vh) in errors.get(target, {}).items():
            tag = "" if name == "ubar" else f"_{name}"
            row[f"L2{tag}{suffix}"], row[f"Vh{tag}{suffix}"] = l2, vh
    return row

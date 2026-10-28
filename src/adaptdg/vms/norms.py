"""Errors against the exact solution (L2 and the dG norm of eq. 12): coarse solution ``ubar`` and,
with the reconstruction of :mod:`adaptdg.vms.reconstruction`, also ``theta``, ``u`` and ``phi``.

The exact solution is evaluated symbolically with a high quadrature degree, so singular data
do not create an artificial error floor at high polynomial degree.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _norms(hq: Any, problem: Any, field: Any) -> tuple[float, float]:
    from dolfinx import fem

    e = problem.exact(hq.x) - field
    l2 = fem.assemble_scalar(fem.form(e * e * hq.dx))
    vh = fem.assemble_scalar(fem.form(hq.gram(e, e)))
    return float(np.sqrt(abs(l2))), float(np.sqrt(abs(vh)))


def _quadrature_forms(result: Any, problem: Any, p: int, degree: int) -> Any:
    from adaptdg.vms.dgforms import DGForms

    hq = DGForms(result.forms.domain, problem, p, degree=degree)
    hq.eta = result.forms.eta  # same penalty as the solver
    return hq


def error_norms(result: Any, problem: Any, p: int, *, degree: int = 8) -> tuple[float, float]:
    """Return ``(||e||_0, ||e||_Vh)`` for ``e = u_exact - ubar`` (needs ``problem.exact``)."""
    return _norms(_quadrature_forms(result, problem, p, degree), problem, result.ubar)


def error_norms_vms(result: Any, vms: Any, problem: Any, p: int, *,
                    degree: int = 8) -> dict[str, tuple[float, float]]:
    """``(L2, Vh)`` errors of ``ubar``, ``theta`` (dG), ``u`` (= ubar + utilde) and ``phi`` (adjoint
    reconstruction, ``vms`` from :func:`adaptdg.vms.reconstruction.reconstruct`)."""
    hq = _quadrature_forms(result, problem, p, degree)
    fields = {"ubar": vms.ubar, "theta": vms.theta, "u": vms.u_full, "phi": vms.phi}
    return {name: _norms(hq, problem, fun) for name, fun in fields.items()}

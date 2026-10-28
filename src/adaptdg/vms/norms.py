"""Errors of the coarse solution against the exact solution (L2 and the dG norm of eq. 12).

The exact solution is evaluated symbolically with a high quadrature degree, so singular data
do not create an artificial error floor at high polynomial degree.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def error_norms(result: Any, problem: Any, p: int, *, degree: int = 8) -> tuple[float, float]:
    """Return ``(||e||_0, ||e||_Vh)`` for ``e = u_exact - ubar`` (needs ``problem.exact``)."""
    from dolfinx import fem

    from adaptdg.vms.dgforms import DGForms

    forms = result.forms
    hq = DGForms(forms.domain, problem, p, degree=degree)
    hq.eta = forms.eta  # same penalty as the solver
    e = problem.exact(hq.x) - result.ubar
    l2 = fem.assemble_scalar(fem.form(e * e * hq.dx))
    vh = fem.assemble_scalar(fem.form(hq.gram(e, e)))
    return float(np.sqrt(abs(l2))), float(np.sqrt(abs(vh)))

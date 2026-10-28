# ruff: noqa: F401  (verbatim legacy code)
"""Variable macro time step for long horizons (opt-in; the fixed path is untouched).

``next_dt`` is the pure scheduling rule; ``VariableDtHistoryDGTransport`` is the
DG1 history backend with the substep dt held in a ``fem.Constant`` so that it can
change between macro steps (the original solver bakes dt into the form as a float).

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/variable_dt.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

import numpy as np

from .dg_history import HistoryDGTransport

TIME_TOL_S = 1.0e-6


def next_dt(rule, t_s, dt_nominal_prev_s, q_m_s, q0_m_s, step_s, dt_max_s, growth, targets_s):
    """Return (dt actually taken, nominal dt, landed target or None).

    courant: nominal = min(dt_max, step_s q0/q, prev_nominal*growth), >= step_s.
    fixed:   nominal = step_s.
    The step is shortened to land exactly on the next target time (> t).
    """
    if rule == "courant":
        nominal = min(dt_max_s, step_s * q0_m_s / q_m_s, dt_nominal_prev_s * growth)
        nominal = max(nominal, step_s)
    else:
        nominal = step_s
    for target in targets_s:
        if target > t_s + TIME_TOL_S:
            if t_s + nominal >= target - TIME_TOL_S:
                return target - t_s, nominal, target
            break
    return nominal, nominal, None


class VariableDtHistoryDGTransport(HistoryDGTransport):
    def __init__(self, chemistry, time_step_s, history="dg1", positivity="local", tvb_m=0.0,
                 reaction_update="additive"):
        super().__init__(chemistry, time_step_s, history, positivity, tvb_m, reaction_update)
        import ufl
        from dolfinx import fem
        from dolfinx.fem.petsc import LinearProblem
        from petsc4py import PETSc

        s = self.solver
        s.dt_const = fem.Constant(s.domain, PETSc.ScalarType(s.dt))
        n = ufl.FacetNormal(s.domain)
        qn, qn_plus = s.q * n[0], s.q("+") * n("+")[0]
        incoming = 0.5 * (abs(qn) - qn)
        trial, test = ufl.TrialFunction(s.space), ufl.TestFunction(s.space)
        advection = -trial * s.q * test.dx(0) * s.dx
        advection += (qn_plus * ufl.avg(trial) * ufl.jump(test)
                      + 0.5 * abs(qn_plus) * ufl.jump(trial) * ufl.jump(test)) * s.dS
        advection += (qn + incoming) * trial * test * s.ds
        rhs = (s.u_previous / 1000.0 * test * s.dx
               + s.dt_const * incoming * s.inflow * test * s.ds)
        lhs = s.phi * trial * test * s.dx + s.dt_const * advection
        s._problem = LinearProblem(lhs, rhs, petsc_options_prefix="xieb2case4vdt_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                           "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True})

    def set_substep_dt(self, dt_s):
        self.solver.dt = float(dt_s)
        self.solver.dt_const.value = float(dt_s)

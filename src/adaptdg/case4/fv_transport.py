"""First-order upwind finite-volume transport (= DG0) for the Case 4 runner (opt-in).

``--transport dg --history fv0`` of run_case4_dg.py. Same interface as
dg_history.HistoryDGTransport, so the coupling (advance_case4_step: Darcy flux of
the step start with series/harmonic cell permeabilities, Strang-type substeps
with the chemistry at the middle substep, aqueous rescaling at porosity change),
the chemistry and the Courant time-step rule are exactly those of the DG1 and
ADAP runs; only the transport discretization changes.

Each substep is one implicit-Euler step of ``phi dC/dt + d(qC)/dx = 0`` with
upwind fluxes, fixed inlet value and free outflow (fv_upwind_step). Implicit upwind is monotone, so no
positivity correction is needed (none is applied; the minimum is logged).

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/fv_transport.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

import numpy as np
from scipy.linalg import solve_banded


def fv_upwind_step(eps_new, eps_old, conc, q, inlet, dt, dx):
    """One implicit-Euler step of ddt(eps, Y) + div(phi, Y) = 0, upwind, 1D, q > 0.

    Row i (cell volume / cross-section = dx):
        eps_new_i dx/dt Y_i + q (Y_i - Y_(i-1)) = eps_old_i dx/dt Y_old_i,   Y_(-1) = inlet
    (fixedValue inlet, zeroGradient outlet: the outlet face carries q Y_(N-1)).
    ``conc``: (ncomp, N). Returns (ncomp, N).
    """
    if q < 0.0:
        raise ValueError("upwind transport implemented for q >= 0")
    n = conc.shape[1]
    rdt = dx / dt
    ab = np.zeros((2, n))
    ab[0] = np.asarray(eps_new) * rdt + q
    ab[1, :-1] = -q
    rhs = (np.asarray(eps_old) * rdt * conc).T.copy()
    rhs[0] += q * np.asarray(inlet)
    return solve_banded((1, 0), ab, rhs).T



class FVUpwindTransport:
    """DG0 / first-order upwind FV backend (uniform chemistry grid)."""

    def __init__(self, chemistry, time_step_s):
        from adaptdg.case4.constants import LENGTH_M

        self.chemistry = chemistry
        self.dt = float(time_step_s)
        self.dx = LENGTH_M / chemistry.cells
        self.substep_log = []

    def set_substep_dt(self, dt_s):
        self.dt = float(dt_s)

    def step(self, state, flux_m_s):
        phi = np.asarray(state.porosity, dtype=float)
        conc = np.asarray(state.concentrations_mol_l, dtype=float)
        inlet = np.asarray(self.chemistry.inlet, dtype=float)
        q = float(flux_m_s)
        new = fv_upwind_step(phi, phi, conc, q, inlet, self.dt, self.dx)
        weights = 1000.0 * phi * self.dx
        names = self.chemistry.components
        balance, positivity = {}, {}
        for c, name in enumerate(names):
            before = float(weights @ conc[c])
            after = float(weights @ new[c])
            outlet = float(new[c, -1])
            net = self.dt * 1000.0 * q * (inlet[c] - outlet)
            residual = after - before - net
            balance[name] = {"relative": abs(residual) / max(abs(before), abs(net), 1e-30),
                             "absolute_mol_m2": residual, "inventory_before_mol_m2": before,
                             "net_boundary_moles_per_m2": net, "outlet_concentration_mol_l": outlet}
            positivity[name] = {"min_mean_raw": float(new[c].min()), "cells_fixed": 0,
                                "moved_inventory_mol_m2": 0.0, "max_mean_correction_mol_l": 0.0,
                                "cells_scaled": 0}
        self.substep_log.append({"positivity": positivity, "balance": balance})
        return new, {
            "balance_relative": {k: v["relative"] for k, v in balance.items()},
            "last_substep_transport_balance": balance,
            "projection_correction": {k: 0.0 for k in names},
            "minimum_nodal": {k: v["min_mean_raw"] for k, v in positivity.items()},
            "positivity": positivity,
        }

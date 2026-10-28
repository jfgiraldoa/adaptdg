"""Stage 2b: backward Euler or BDF2 transport substeps, for DG and ADAP alike.

Both methods get the same integrator so that the comparison changes only the
spatial adaptivity. With a constant substep dt, BDF2

    phi (3/2 u - 2 u^n + 1/2 u^(n-1)) / dt + A u = 0

is the backward-Euler problem  phi u + dt' A u = phi h  with dt' = 2 dt / 3
and history h = (4 u^n - u^(n-1)) / 3, so the existing operators, positivity
treatments and conservative projection are reused unchanged. The first
substep of a run is backward Euler.

Operator splitting: chemistry/geometry change the chemistry-cell means once
per macro step. The current history receives that change as a constant
increment (exactly as the backward-Euler path). The older history receives
the same geometry rescaling as the state (c * phi_old / phi_new, which keeps
phi*c) plus the chemical part of the increment, so both histories keep the
same inventory change and BDF2 differentiates transport only. (The 2D uniform DG path
instead keeps the previous reacted state as the older history; there the
reaction acts every step.)

Conservation: BDF2 conserves J^n = I^n + (I^n - I^(n-1)) / 2 per substep,
J^(n+1) - J^n = dt * (inflow - outflow). ``bdf2_offset`` returns
(I^n - I^(n-1)) / 2 per component and ``bdf2_startup`` the constant term
introduced by the backward-Euler start, so the runner can check the
cumulative balance exactly.

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/transport_bdf2.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

from __future__ import annotations

import numpy as np

from .positivity import local_mean_redistribution, scale_slopes_nonnegative

NOT_SIGN_CONSTRAINED = frozenset({"H2O", "H", "O", "Charge"})


def _coefficients(older_available, dt, scheme):
    """Return (dt_eff, weight_current, weight_older) of the BE-form solve."""
    if scheme == "bdf2" and older_available:
        return 2.0 * dt / 3.0, 4.0 / 3.0, -1.0 / 3.0
    return dt, 1.0, 0.0


class _TimeHistory:
    """Current and older nodal histories (ncomp, cells, 2) + increments."""

    def __init__(self, scheme):
        if scheme not in {"be", "bdf2"}:
            raise ValueError("time scheme must be be or bdf2")
        self.scheme = scheme
        self.current = None
        self.older = None
        self.means_seen = None
        self.startup = None        # per component, mol/m2 (bdf2 only)

        self.phi_seen = None       # chemistry-cell porosity of the last substep

    def apply_increment(self, old_means, new_means, ratio):
        """Per history cell: old/new means (ncomp, cells), ratio phi_old/phi_new."""
        delta = new_means - old_means
        if self.older is not None:
            chemical = new_means - old_means * ratio[None, :]
            self.older = self.older * ratio[None, :, None] + chemical[:, :, None]
        self.current = self.current + delta[:, :, None]


# --------------------------------------------------------------------- DG
class BDF2DGTransport:
    """DG1 on the chemistry grid (history dg1, positivity local) with BE/BDF2."""

    def __init__(self, chemistry, time_step_s, scheme="bdf2"):
        self.chemistry, self.dt = chemistry, float(time_step_s)
        names = chemistry.components
        self.constrained = {i for i, n in enumerate(names) if n not in NOT_SIGN_CONSTRAINED}
        self.time = _TimeHistory(scheme)
        self._solvers = {}
        self.substep_log = []

    def _solver(self, dt_eff):
        if dt_eff not in self._solvers:
            from adaptdg.case4.dg_1d import PorosityDG1D

            class _DG(PorosityDG1D):
                def __init__(self, cells, dt):
                    super().__init__(cells, dt)
                    dofs = self.cell_dofs[self.cell_order]
                    coords = self.dof_coordinates[dofs]
                    swap = coords[:, 0] > coords[:, 1]
                    self.left_dof = np.where(swap, dofs[:, 1], dofs[:, 0])
                    self.right_dof = np.where(swap, dofs[:, 0], dofs[:, 1])
                    self.phi_dof_phys = self.phi_dofs[self.cell_order]

                def solve_nodal(self, porosity, flux, left, right, inlet):
                    self.phi.x.array[self.phi_dof_phys] = porosity
                    self.phi.x.scatter_forward()
                    self.q.value = np.asarray([flux], dtype=float)
                    arr = self.u_previous.x.array
                    arr[self.left_dof] = 1000.0 * porosity * left
                    arr[self.right_dof] = 1000.0 * porosity * right
                    self.u_previous.x.scatter_forward()
                    self.inflow.x.array[:] = inlet
                    self.inflow.x.scatter_forward()
                    u = self._problem.solve()
                    u.x.scatter_forward()
                    return u.x.array[self.left_dof].copy(), u.x.array[self.right_dof].copy()

            self._solvers[dt_eff] = _DG(self.chemistry.cells, dt_eff)
        return self._solvers[dt_eff]

    @property
    def widths(self):
        from adaptdg.case4.constants import LENGTH_M
        # Same widths (bit for bit) as PorosityDG1D.widths.
        return np.diff(np.linspace(0.0, LENGTH_M, self.chemistry.cells + 1))

    def bdf2_offset(self, porosity):
        if self.time.older is None:
            return np.zeros(len(self.chemistry.components))
        w = 1000.0 * np.asarray(porosity) * self.widths
        return 0.5 * (self.time.current.mean(axis=2) - self.time.older.mean(axis=2)) @ w

    def bdf2_startup(self):
        return np.zeros(len(self.chemistry.components)) if self.time.startup is None else self.time.startup

    def checkpoint_arrays(self):
        out = {"tb_current": self.time.current, "tb_means_seen": self.time.means_seen}
        if self.time.older is not None:
            out["tb_older"] = self.time.older
        if self.time.startup is not None:
            out["tb_startup"] = self.time.startup
        if self.time.phi_seen is not None:
            out["tb_phi_seen"] = self.time.phi_seen
        return out

    def restore(self, arrays):
        self.time.current = arrays["tb_current"].copy()
        self.time.means_seen = arrays["tb_means_seen"].copy()
        self.time.older = arrays["tb_older"].copy() if "tb_older" in arrays else None
        self.time.startup = arrays["tb_startup"].copy() if "tb_startup" in arrays else None
        self.time.phi_seen = arrays["tb_phi_seen"].copy() if "tb_phi_seen" in arrays else None

    def step(self, state, flux_m_s):
        conc = np.asarray(state.concentrations_mol_l, dtype=float)
        porosity = np.asarray(state.porosity, dtype=float)
        names, inlet = self.chemistry.components, self.chemistry.inlet
        t = self.time
        if t.current is None:
            t.current = np.repeat(conc[:, :, None], 2, axis=2)
        else:
            t.apply_increment(t.means_seen, conc, t.phi_seen / porosity)
        t.phi_seen = porosity.copy()
        dt_eff, a, b = _coefficients(t.older is not None, self.dt, t.scheme)
        hist = a * t.current + (b * t.older if b else 0.0)
        solver = self._solver(dt_eff)
        w = 1000.0 * porosity * self.widths
        new = np.empty_like(t.current)
        balance, positivity = {}, {}
        for c in range(conc.shape[0]):
            left, right = solver.solve_nodal(porosity, flux_m_s, hist[c, :, 0], hist[c, :, 1], inlet[c])
            outlet = float(right[-1])
            net_eff = dt_eff * 1000.0 * flux_m_s * (inlet[c] - outlet)
            means = 0.5 * (left + right)
            before = float(w @ hist[c].mean(axis=1))
            residual = float(w @ means) - before - net_eff
            info = {"min_mean_raw": float(means.min()), "cells_fixed": 0,
                    "moved_inventory_mol_m2": 0.0, "max_mean_correction_mol_l": 0.0, "cells_scaled": 0}
            if c in self.constrained:
                if means.min() < 0.0:
                    fixed, red = local_mean_redistribution(means, w)
                    shift = fixed - means
                    left, right = left + shift, right + shift
                    info.update(cells_fixed=red["cells_fixed"], moved_inventory_mol_m2=red["moved_inventory"],
                                max_mean_correction_mol_l=float(np.max(np.abs(shift))))
                left, right, sc = scale_slopes_nonnegative(left, right)
                info["cells_scaled"] = sc["cells_scaled"]
            new[c, :, 0], new[c, :, 1] = left, right
            # Physical boundary exchange over the substep (dt, not dt_eff).
            balance[names[c]] = {
                "relative": abs(residual) / max(abs(before), abs(net_eff), 1e-30),
                "absolute_mol_m2": residual,
                "net_boundary_moles_per_m2": self.dt * 1000.0 * flux_m_s * (inlet[c] - outlet),
                "outlet_concentration_mol_l": outlet}
            positivity[names[c]] = info
        if t.scheme == "bdf2" and t.older is None:
            # BE start: J^1 = I^1 + (I^1 - I^0)/2 exceeds I^0 + dt*net_1 by dt*net_1/2.
            t.startup = 0.5 * np.array([balance[n]["net_boundary_moles_per_m2"] for n in names])
        if t.scheme == "bdf2":
            t.older = t.current
        t.current = new
        t.means_seen = new.mean(axis=2)
        self.substep_log.append({"positivity": positivity, "balance": balance})
        return t.means_seen.copy(), {
            "balance_relative": {k: v["relative"] for k, v in balance.items()},
            "last_substep_transport_balance": balance,
        }


# ------------------------------------------------------------------- ADAP
class BDF2AdaptiveTransport:
    """Persistent ADAP (adap_fast operator/projection, nested mesh) with BE/BDF2."""

    def __init__(self, chemistry, time_step_s, settings, scheme="bdf2", reference_porosity=0.35):
        from adaptdg.case4.constants import LENGTH_M
        from .adap_persistent import NestedMesh1D
        self.chemistry, self.dt, self.settings = chemistry, float(time_step_s), settings
        self.chem_edges = np.linspace(0.0, LENGTH_M, chemistry.cells + 1)
        self.mesh = NestedMesh1D(self.chem_edges)
        self.phi0 = reference_porosity
        names = chemistry.components
        self.constrained = [i for i, n in enumerate(names) if n not in NOT_SIGN_CONSTRAINED]
        self.time = _TimeHistory(scheme)
        self._op = None
        self._op_edges = None
        self.substep_log = []

    # mesh-level helpers ----------------------------------------------------
    def _operator(self):
        from .adap_fast import NumpyOperator
        edges = self.mesh.edges()
        if self._op is None or self._op_edges.shape != edges.shape or not np.array_equal(self._op_edges, edges):
            self._op, self._op_edges = NumpyOperator(edges, self.phi0), edges
        return self._op

    def chem_means(self, nodal):
        idx = self.mesh.chem_index()
        w = np.diff(self.mesh.edges())
        sums = np.zeros((nodal.shape[0], self.chem_edges.size - 1))
        for c in range(nodal.shape[0]):
            np.add.at(sums[c], idx, w * nodal[c].mean(axis=1))
        return sums / np.diff(self.chem_edges)

    def _both(self):
        t = self.time
        return t.current if t.older is None else np.concatenate([t.current, t.older], axis=0)

    def _split(self, stacked):
        n = self.time.current.shape[0]
        self.time.current = stacked[:n]
        if self.time.older is not None:
            self.time.older = stacked[n:]

    @property
    def nodal(self):
        return self.time.current

    def bdf2_offset(self, porosity):
        if self.time.older is None:
            return np.zeros(len(self.chemistry.components))
        w = 1000.0 * np.asarray(porosity)[self.mesh.chem_index()] * np.diff(self.mesh.edges())
        return 0.5 * (self.time.current.mean(axis=2) - self.time.older.mean(axis=2)) @ w

    def bdf2_startup(self):
        return np.zeros(len(self.chemistry.components)) if self.time.startup is None else self.time.startup

    def checkpoint_arrays(self):
        out = {"adap_leaves": np.asarray(self.mesh.leaves, dtype=float),
               "tb_current": self.time.current, "tb_means_seen": self.time.means_seen}
        if self.time.older is not None:
            out["tb_older"] = self.time.older
        if self.time.startup is not None:
            out["tb_startup"] = self.time.startup
        if self.time.phi_seen is not None:
            out["tb_phi_seen"] = self.time.phi_seen
        return out

    def restore(self, arrays):
        self.mesh.leaves = [tuple(int(v) for v in row) for row in arrays["adap_leaves"]]
        self.time.current = arrays["tb_current"].copy()
        self.time.means_seen = arrays["tb_means_seen"].copy()
        self.time.older = arrays["tb_older"].copy() if "tb_older" in arrays else None
        self.time.startup = arrays["tb_startup"].copy() if "tb_startup" in arrays else None
        self.time.phi_seen = arrays["tb_phi_seen"].copy() if "tb_phi_seen" in arrays else None

    # one substep -------------------------------------------------------------
    def step(self, state, flux_m_s):
        from adaptdg.marking import dorfler_mark_1d
        from adaptdg.transport import gradient_jump_indicator_1d
        from .adap_fast import positive_projection

        conc = np.asarray(state.concentrations_mol_l, dtype=float)
        ncomp = conc.shape[0]
        names, inlet = self.chemistry.components, self.chemistry.inlet
        s, t = self.settings, self.time
        idx = self.mesh.chem_index()
        porosity = np.asarray(state.porosity, dtype=float)
        if t.current is None:
            t.current = np.repeat(conc[:, idx, None], 2, axis=2)
        else:
            t.apply_increment(t.means_seen[:, idx], conc[:, idx], (t.phi_seen / porosity)[idx])
        t.phi_seen = porosity.copy()
        dt_eff, a, b = _coefficients(t.older is not None, self.dt, t.scheme)
        chem_w = float(np.min(np.diff(self.chem_edges)))
        solves = 0
        while True:
            solves += 1
            edges = self.mesh.edges()
            widths = np.diff(edges)
            phi_t = np.asarray(state.porosity)[self.mesh.chem_index()]
            hist = a * t.current + (b * t.older if b else 0.0)
            op = self._operator()
            op.set_coefficients(phi_t, flux_m_s, dt_eff)
            raw, indicators, grads = [], [], []
            for c in range(ncomp):
                values, ind = op.solve(hist[c, :, 0], hist[c, :, 1], float(inlet[c]))
                raw.append(values)
                indicators.append(ind)
                grads.append(gradient_jump_indicator_1d(edges, values))
            if solves >= s.max_solves:
                break
            marked = np.zeros(widths.size, dtype=bool)
            for ind in indicators + grads:
                marked |= dorfler_mark_1d(ind, widths, theta=s.theta, minimum_width=chem_w / 2 ** s.max_level,
                                          epsilon_eligible=s.epsilon_eligible)
            if not marked.any():
                break
            self._split(self.mesh.refine(marked, self._both()))

        new = np.empty((ncomp, widths.size, 2))
        balance, positivity = {}, {}
        weights = 1000.0 * phi_t * widths
        for c in range(ncomp):
            hist_means = hist[c].mean(axis=1)
            before = float(weights @ hist_means)
            values = raw[c]
            if c in self.constrained:
                values, correction = positive_projection(
                    edges, values, phi_t, hist_means, time_step_s=dt_eff,
                    darcy_flux_m_s=flux_m_s, inlet_concentration=float(inlet[c]))
            else:
                after_raw = float(weights @ (0.5 * (values[:-1] + values[1:])))
                net_raw = dt_eff * 1000.0 * flux_m_s * (inlet[c] - values[-1])
                correction = (before + net_raw - after_raw) / (
                    1000.0 * (float(phi_t @ widths) + dt_eff * flux_m_s))
                values = values + correction
            outlet = float(values[-1])
            net_eff = dt_eff * 1000.0 * flux_m_s * (inlet[c] - outlet)
            residual = float(weights @ (0.5 * (values[:-1] + values[1:]))) - before - net_eff
            balance[names[c]] = {
                "relative": abs(residual) / max(abs(before), abs(net_eff), 1e-30),
                "absolute_mol_m2": residual,
                "net_boundary_moles_per_m2": self.dt * 1000.0 * flux_m_s * (inlet[c] - outlet),
                "outlet_concentration_mol_l": outlet}
            positivity[names[c]] = {"min_mean_raw": float(raw[c].min()), "cells_fixed": 0,
                                    "moved_inventory_mol_m2": 0.0,
                                    "max_mean_correction_mol_l": float(abs(correction)), "cells_scaled": 0}
            new[c, :, 0], new[c, :, 1] = values[:-1], values[1:]

        if t.scheme == "bdf2" and t.older is None:
            t.startup = 0.5 * np.array([balance[n]["net_boundary_moles_per_m2"] for n in names])
        if t.scheme == "bdf2":
            t.older = t.current
        t.current = new
        # Coarsen with both histories (same merges, means preserved).
        stacked = self._both()
        scales = np.maximum(np.ptp(stacked.reshape(stacked.shape[0], -1), axis=1), 1e-30)
        constrained = self.constrained + ([ncomp + c for c in self.constrained] if t.older is not None else [])
        stacked, merged = self.mesh.coarsen(stacked, s.coarsen_tolerance, scales, constrained)
        self._split(stacked)
        t.means_seen = self.chem_means(t.current)
        levels = self.mesh.levels()
        self.substep_log.append({"positivity": positivity, "balance": balance,
                                 "transport_cells": int(levels.size), "solves": solves,
                                 "merged": merged, "max_level": int(levels.max())})
        return t.means_seen.copy(), {
            "balance_relative": {k: v["relative"] for k, v in balance.items()},
            "last_substep_transport_balance": balance,
        }

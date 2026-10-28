# ruff: noqa: E402, E702  (verbatim legacy code)
"""Opt-in DG1 history and admissibility for the Case 4 DG transport.

Reuses the exact weak form and solver of asrtm_phreeqc.porosity.dg_1d.PorosityDG1D
(read-only import). Options:

history="mean"  -> reproduces PorosityDG1D: slopes reset every substep.
history="dg1"   -> nodal DG1 values are kept between substeps. Changes made to
                   the cell means outside transport (chemistry, porosity
                   rescaling) are applied as a uniform increment to both nodal
                   values (same treatment as the 2D R76 DG1 history).
positivity="none"  -> no change (as the historical run).
positivity="local" -> after each substep: local conservative redistribution
                   of negative means, then Zhang-Shu slope scaling (dg1 only).
reaction_update="additive" -> chemistry/porosity changes of a cell mean are added
                   uniformly to both nodes (historical behaviour).
reaction_update="proportional" -> for sign-constrained components whose mean
                   DECREASED outside transport (consumed by the reaction), the nodal
                   deviation from the mean is scaled by m_new/m_old instead of shifted:
                   the reaction removes more where there is more, nodes that were >= 0
                   stay >= 0 and the positivity clipping is not triggered (the clipping
                   of the limiting reagent, Ca, every substep was found to create the
                   odd-even pattern of the minerals, 2026-10-18). Means are unchanged.
                   Increases keep the additive update.
positivity="local_minmod" -> as "local", plus a minmod slope limiter (dg1 only,
                   every component, before the Zhang-Shu scaling): half-jump
                   s_i = (right - left)/2 replaced by minmod(s_i, m_{i+1} - m_i,
                   m_i - m_{i-1}); inlet value as left neighbour of cell 0, the last
                   cell uses its upwind difference only. Means are unchanged.
                   Applied only to element totals: PhreeqcRM's H, O (excess
                   over H2O), H2O and Charge are not sign-constrained.

Provenance: copied from ASRTM-PHREEQC, methods/case4dev/dg_history.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

import numpy as np

NOT_SIGN_CONSTRAINED = frozenset({"H2O", "H", "O", "Charge"})

from .positivity import local_mean_redistribution, scale_slopes_nonnegative


def minmod_limit(left, right, inlet_value, tvb_threshold=0.0):
    """Minmod slope limiter on nodal DG1 values (cell means unchanged).

    tvb_threshold > 0: TVB variant (Cockburn & Shu), half-jumps with |s| <= tvb_threshold
    (= M h^2 times the component scale, set by the caller) are left untouched."""
    m = 0.5 * (left + right)
    s = 0.5 * (right - left)
    back = m - np.r_[inlet_value, m[:-1]]
    fwd = np.r_[m[1:] - m[:-1], back[-1]]
    same = (np.sign(s) == np.sign(back)) & (np.sign(s) == np.sign(fwd))
    lim = np.where(same, np.sign(s) * np.minimum(np.abs(s), np.minimum(np.abs(back), np.abs(fwd))), 0.0)
    if tvb_threshold > 0.0:
        lim = np.where(np.abs(s) <= tvb_threshold, s, lim)
    changed = int(np.sum(np.abs(lim - s) > 1e-14 * np.maximum(np.abs(m), 1e-30)))
    return m - lim, m + lim, changed


def make_history_solver(cells, time_step_s, history="dg1", positivity="local",
                        constrained=None, tvb_m=0.0, reaction_update="additive"):
    """``constrained``: component indices to keep nonnegative (None -> all)."""
    from adaptdg.case4.dg_1d import PorosityDG1D

    if history not in {"mean", "dg1"} or positivity not in {"none", "local", "local_minmod"}:
        raise ValueError("history must be mean|dg1 and positivity none|local|local_minmod")
    minmod = positivity == "local_minmod"
    if reaction_update not in {"additive", "proportional"}:
        raise ValueError("reaction_update must be additive|proportional")

    class PorosityDG1DHistory(PorosityDG1D):
        def __init__(self):
            super().__init__(cells, time_step_s)
            self.history, self.positivity = history, positivity
            dofs = self.cell_dofs[self.cell_order]            # physical order
            coords = self.dof_coordinates[dofs]
            swap = coords[:, 0] > coords[:, 1]
            self.left_dof = np.where(swap, dofs[:, 1], dofs[:, 0])
            self.right_dof = np.where(swap, dofs[:, 0], dofs[:, 1])
            self.phi_dof_phys = self.phi_dofs[self.cell_order]
            self.nodal = None        # (ncomp, cells, 2) mol/L, physical order
            self.means_seen = None   # means at the end of the last substep

        def set_porosity(self, porosity):
            self.phi.x.array[self.phi_dof_phys] = porosity
            self.phi.x.scatter_forward()

        def step(self, porosity, concentration, flux_m_s, inlet):
            conc = np.asarray(concentration, dtype=float)
            ncomp = conc.shape[0]
            if self.history == "mean" or self.nodal is None:
                nodal = np.repeat(conc[:, :, None], 2, axis=2)
            elif reaction_update == "additive":
                nodal = self.nodal + (conc - self.means_seen)[:, :, None]
            else:
                dev = self.nodal - self.means_seen[:, :, None]
                old, new = self.means_seen, conc
                ratio = np.ones_like(old)
                mask = (new < old) & (old > 0.0)
                if constrained is not None:
                    keep = np.zeros(ncomp, bool); keep[list(constrained)] = True
                    mask &= keep[:, None]
                ratio[mask] = np.maximum(new[mask], 0.0) / old[mask]
                nodal = new[:, :, None] + dev * ratio[:, :, None]
            self.set_porosity(porosity)
            self.q.value = np.asarray([flux_m_s], dtype=float)
            weights = 1000.0 * porosity * self.widths
            new = np.empty_like(nodal)
            balance, positivity_info = {}, {}
            for c in range(ncomp):
                inventory_before = float(np.dot(weights, conc[c]))
                arr = self.u_previous.x.array
                arr[self.left_dof] = 1000.0 * porosity * nodal[c, :, 0]
                arr[self.right_dof] = 1000.0 * porosity * nodal[c, :, 1]
                self.u_previous.x.scatter_forward()
                self.inflow.x.array[:] = inlet[c]
                self.inflow.x.scatter_forward()
                u = self._problem.solve()
                u.x.scatter_forward()
                left = u.x.array[self.left_dof].copy()
                right = u.x.array[self.right_dof].copy()
                outlet = float(right[-1])
                net = self.dt * 1000.0 * flux_m_s * (inlet[c] - outlet)
                means = 0.5 * (left + right)
                inventory_after = float(np.dot(weights, means))
                residual = inventory_after - inventory_before - net
                balance[str(c)] = {
                    "relative": abs(residual) / max(abs(inventory_before), abs(net), 1e-30),
                    "absolute_mol_m2": residual, "inventory_before_mol_m2": inventory_before,
                    "net_boundary_moles_per_m2": net, "outlet_concentration_mol_l": outlet,
                }
                info = {"min_mean_raw": float(means.min()), "cells_fixed": 0,
                        "moved_inventory_mol_m2": 0.0, "max_mean_correction_mol_l": 0.0,
                        "cells_scaled": 0}
                apply = self.positivity in ("local", "local_minmod") and (constrained is None or c in constrained)
                if apply and means.min() < 0.0:
                    fixed, red = local_mean_redistribution(means, weights)
                    shift = fixed - means
                    left, right = left + shift, right + shift
                    info.update(cells_fixed=red["cells_fixed"],
                                moved_inventory_mol_m2=red["moved_inventory"],
                                max_mean_correction_mol_l=float(np.max(np.abs(shift))),
                                redistribution_defect_mol_m2=red["inventory_defect"],
                                max_radius=red["max_radius"])
                    means = fixed
                if minmod and self.history == "dg1":
                    thr = 0.0
                    if tvb_m > 0.0:   # M h^2 relative to the component scale (inlet and current means)
                        scale = max(abs(float(inlet[c])), float(np.max(np.abs(0.5 * (left + right)))), 1e-30)
                        thr = tvb_m * float(np.max(self.widths)) ** 2 * scale
                    left, right, info["cells_limited"] = minmod_limit(left, right, inlet[c], thr)
                if apply and self.history == "dg1":
                    left, right, sc = scale_slopes_nonnegative(left, right)
                    info["cells_scaled"] = sc["cells_scaled"]
                new[c, :, 0], new[c, :, 1] = left, right
                positivity_info[str(c)] = info
            self.nodal = new
            self.means_seen = new.mean(axis=2)
            return self.means_seen.copy(), balance, positivity_info

    return PorosityDG1DHistory()


class HistoryDGTransport:
    """Backend with the same interface as coupling.UniformDGTransport."""

    def __init__(self, chemistry, time_step_s, history="dg1", positivity="local", tvb_m=0.0,
                 reaction_update="additive"):
        self.chemistry = chemistry
        constrained = {i for i, name in enumerate(chemistry.components)
                       if name not in NOT_SIGN_CONSTRAINED}
        self.solver = make_history_solver(chemistry.cells, time_step_s, history, positivity,
                                          constrained, tvb_m, reaction_update)
        # advance_case4_step only keeps minima/corrections for adaptive
        # backends and the balance of the last substep, so every substep's
        # positivity and boundary record is kept here; the caller reads and
        # clears it after each macro step.
        self.substep_log = []

    def step(self, state, flux_m_s):
        means, balance, pos = self.solver.step(
            state.porosity, state.concentrations_mol_l, flux_m_s, self.chemistry.inlet)
        names = self.chemistry.components
        named = {names[int(i)]: v for i, v in balance.items()}
        pos_named = {names[int(i)]: v for i, v in pos.items()}
        self.substep_log.append({"positivity": pos_named, "balance": named})
        return means, {
            "balance_relative": {k: v["relative"] for k, v in named.items()},
            "last_substep_transport_balance": named,
            "projection_correction": {k: v["max_mean_correction_mol_l"] for k, v in pos_named.items()},
            "minimum_nodal": {k: v["min_mean_raw"] for k, v in pos_named.items()},
            "positivity": pos_named,
        }

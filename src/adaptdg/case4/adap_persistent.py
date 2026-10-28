# ruff: noqa: E702  (verbatim legacy code)
"""Stage 2a: persistent adaptive transport for Case 4 (opt-in, 1D).

Same operator as asrtm_phreeqc.porosity.transport_1d (CG1 trial / DG1 test
residual minimization, backward Euler, the same Riesz norm, Dörfler marking
on residual and gradient-jump indicators, the same conservative positive
projection). What changes is what the 2D R76 policy changed:

* the transport mesh and its solution persist between substeps;
* the history is the complete DG1 field on the transport mesh (it holds
  the accepted CG1 solution exactly), not the 80-cell averages;
* chemistry/geometry changes of a chemistry-cell mean are added as a
  constant increment to every transport cell inside that chemistry cell;
* refinement bisects cells and transfers the history exactly;
* coarsening merges two sibling cells when the L2 projection of every
  component onto the parent changes it by less than ``coarsen_tolerance``
  (relative to the component range); mean preserved exactly.

The transport mesh is nested in the chemistry grid: every transport cell
lies inside one chemistry cell, so porosity is exact per transport cell and
both transfers are exact and conservative. The coarsest transport cell is
one chemistry cell.

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/adap_persistent.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .positivity import scale_slopes_nonnegative

NOT_SIGN_CONSTRAINED = frozenset({"H2O", "H", "O", "Charge"})


@dataclass(frozen=True)
class PersistentSettings:
    max_level: int = 3             # transport cell >= chemistry width / 2**max_level
    theta: float = 0.5             # Dörfler fraction (as the historical ADAP)
    epsilon_eligible: float = 0.2  # as the historical ADAP default
    max_solves: int = 2            # solves per substep (refine between them)
    coarsen_tolerance: float = 1e-3


class NestedMesh1D:
    """Leaves (chem_cell, level, index) ordered left to right."""

    def __init__(self, chem_edges):
        self.chem_edges = np.asarray(chem_edges, dtype=float)
        n = self.chem_edges.size - 1
        self.leaves = [(j, 0, 0) for j in range(n)]

    def edges(self):
        out = []
        for j, level, k in self.leaves:
            a, b = self.chem_edges[j], self.chem_edges[j + 1]
            out.append(a + (b - a) * k / 2 ** level)
        out.append(self.chem_edges[-1])
        return np.asarray(out)

    def chem_index(self):
        return np.array([leaf[0] for leaf in self.leaves])

    def levels(self):
        return np.array([leaf[1] for leaf in self.leaves])

    def refine(self, marked, nodal):
        """Bisect marked leaves; nodal (ncomp, cells, 2) transferred exactly."""
        leaves, out = [], []
        for i, leaf in enumerate(self.leaves):
            left, right = nodal[:, i, 0], nodal[:, i, 1]
            if marked[i]:
                j, level, k = leaf
                mid = 0.5 * (left + right)
                leaves += [(j, level + 1, 2 * k), (j, level + 1, 2 * k + 1)]
                out += [np.stack([left, mid], axis=1), np.stack([mid, right], axis=1)]
            else:
                leaves.append(leaf)
                out.append(np.stack([left, right], axis=1))
        self.leaves = leaves
        return np.stack(out, axis=1)

    def coarsen(self, nodal, tolerance, scales, constrained):
        """Merge sibling pairs whose parent L2 projection is within tolerance."""
        leaves, out, merged = [], [], 0
        i, n = 0, len(self.leaves)
        while i < n:
            j, level, k = self.leaves[i]
            if (level > 0 and k % 2 == 0 and i + 1 < n
                    and self.leaves[i + 1] == (j, level, k + 1)):
                l1, r1 = nodal[:, i, 0], nodal[:, i, 1]
                l2, r2 = nodal[:, i + 1, 0], nodal[:, i + 1, 1]
                m1, s1 = 0.5 * (l1 + r1), 0.5 * (r1 - l1)
                m2, s2 = 0.5 * (l2 + r2), 0.5 * (r2 - l2)
                mean = 0.5 * (m1 + m2)
                slope = 0.75 * (m2 - m1) + 0.25 * (s1 + s2)
                # RMS of (u - Pu) over the parent, from 2-point Gauss rules
                # on each child (exact for the piecewise-linear difference).
                g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
                err2 = np.zeros_like(mean)
                for (m, s, centre) in ((m1, s1, -0.5), (m2, s2, 0.5)):
                    for eta in g:
                        xi = centre + 0.5 * eta
                        child = m[:, None] + s[:, None] * eta
                        parent = mean[:, None] + slope[:, None] * xi
                        err2 += 0.25 * ((child - parent) ** 2)[:, 0]
                if np.all(np.sqrt(err2) <= tolerance * scales):
                    left, right = mean - slope, mean + slope
                    pair = np.stack([left, right], axis=1)
                    for c in constrained:
                        a, b, _ = scale_slopes_nonnegative(pair[c:c + 1, 0], pair[c:c + 1, 1])
                        pair[c, 0], pair[c, 1] = a[0], b[0]
                    leaves.append((j, level - 1, k // 2))
                    out.append(pair)
                    merged += 1
                    i += 2
                    continue
            leaves.append(self.leaves[i])
            out.append(nodal[:, i, :])
            i += 1
        self.leaves = leaves
        return np.stack(out, axis=1), merged


class _Operator:
    """CG1/DG1 residual minimization on one transport mesh (forms of transport_1d)."""

    def __init__(self, edges, reference_porosity):
        from mpi4py import MPI
        from petsc4py import PETSc
        import basix.ufl
        import ufl
        from dolfinx import fem, mesh

        n = edges.size - 1
        cells = np.stack([np.arange(n), np.arange(1, n + 1)], axis=1).astype(np.int64)
        coord = basix.ufl.element("Lagrange", "interval", 1, shape=(1,))
        domain = mesh.create_mesh(MPI.COMM_SELF, cells, coord, edges.reshape(-1, 1))
        self.domain, self.n, self.phi0 = domain, n, float(reference_porosity)
        geom = domain.geometry
        centres = np.array([geom.x[geom.dofmap[c], 0].mean() for c in range(n)])
        self.order = np.argsort(centres)        # physical i -> dolfinx cell
        mixed = fem.functionspace(domain, basix.ufl.mixed_element([
            basix.ufl.element("DG", "interval", 1), basix.ufl.element("Lagrange", "interval", 1)]))
        self.mixed = mixed
        hist_space = fem.functionspace(domain, ("DG", 1))
        dg0 = fem.functionspace(domain, ("DG", 0))
        self.previous = fem.Function(hist_space)
        self.storage = fem.Function(dg0)
        self.dg0_dofs = np.array([dg0.dofmap.cell_dofs(int(c))[0] for c in self.order])
        hx = hist_space.tabulate_dof_coordinates()[:, 0]
        hd = np.stack([hist_space.dofmap.cell_dofs(int(c)) for c in self.order])
        swap = hx[hd[:, 0]] > hx[hd[:, 1]]
        self.h_left = np.where(swap, hd[:, 1], hd[:, 0])
        self.h_right = np.where(swap, hd[:, 0], hd[:, 1])

        dx, dS = ufl.Measure("dx", domain=domain), ufl.Measure("dS", domain=domain)
        ds = ufl.Measure("ds", domain=domain)
        normal, h = ufl.FacetNormal(domain), ufl.CellDiameter(domain)
        self.beta = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.dt = fem.Constant(domain, PETSc.ScalarType(1.0))
        self.inflow = fem.Constant(domain, PETSc.ScalarType(0.0))
        beta, dt = self.beta, self.dt
        bn, bnp = beta * normal[0], beta("+") * normal("+")[0]
        incoming = 0.5 * (abs(bn) - bn)
        inlet_v = mesh.locate_entities_boundary(domain, 0, lambda x: np.isclose(x[0], edges[0]))
        tags = mesh.meshtags(domain, 0, inlet_v.astype(np.int32), np.ones(inlet_v.size, np.int32))
        ds_in = ufl.Measure("ds", domain=domain, subdomain_data=tags)(1)
        r, c = ufl.TrialFunctions(mixed)
        rt, ct = ufl.TestFunctions(mixed)
        # Mass weighted by s = phi/phi0 like the storage term (stability with strong phi contrasts).
        riesz = self.storage * r * rt * dx + dt * (h ** 2 * beta ** 2 * r.dx(0) * rt.dx(0) * dx
                                    + 0.5 * abs(bnp) * ufl.jump(r) * ufl.jump(rt) * dS
                                    + 0.5 * abs(bn) * r * rt * ds)
        block = self.storage * rt * c * dx + dt * (rt * beta * c.dx(0) * dx + incoming * rt * c * ds)
        self.a = fem.form(riesz + block + ufl.adjoint(block))
        self.L = fem.form(self.storage * self.previous * rt * dx + dt * incoming * rt * self.inflow * ds_in)
        self.residual_test_form = None
        self._fem, self._ufl, self._PETSc = fem, ufl, PETSc
        self._dx = dx
        self.lu = None
        self.key = None
        V1, map1 = mixed.sub(1).collapse()
        V0, map0 = mixed.sub(0).collapse()
        # dolfinx 0.11 returns the sub-space map as a one-element list.
        self.map1 = np.asarray(map1[0] if isinstance(map1, list) else map1, dtype=np.int32)
        self.map0 = np.asarray(map0[0] if isinstance(map0, list) else map0, dtype=np.int32)
        x1 = V1.tabulate_dof_coordinates()[:, 0]
        self.node_order = np.argsort(x1)        # physical node -> V1 dof
        self.residual_fn = fem.Function(V0)
        ind_test = ufl.TestFunction(dg0)
        self.ind_form = fem.form(self.residual_fn * self.residual_fn * ind_test * dx)

    def set_coefficients(self, porosity_cells, flux, dt):
        key = (porosity_cells.tobytes(), float(flux), float(dt))
        if key == self.key:
            return
        from dolfinx.fem.petsc import assemble_matrix
        self.storage.x.array[self.dg0_dofs] = porosity_cells / self.phi0
        self.storage.x.scatter_forward()
        self.beta.value = flux / self.phi0
        self.dt.value = dt
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import splu
        A = assemble_matrix(self.a)
        A.assemble()
        indptr, indices, data = A.getValuesCSR()
        size = A.getSize()
        A.destroy()
        # Small 1D saddle-point systems: SuperLU (partial pivoting) is much
        # cheaper than MUMPS' fixed overhead; factorized once per operator.
        self.lu = splu(csr_matrix((data, indices, indptr), shape=size).tocsc())
        self.key = key

    def solve(self, left, right, inlet):
        """Return physical-order CG1 nodal values and cell residual indicators."""
        from dolfinx.fem.petsc import assemble_vector
        self.previous.x.array[self.h_left] = left
        self.previous.x.array[self.h_right] = right
        self.previous.x.scatter_forward()
        self.inflow.value = inlet
        b = assemble_vector(self.L)
        arr = self.lu.solve(b.array.copy())
        b.destroy()
        values = arr[self.map1][self.node_order].copy()
        self.residual_fn.x.array[:] = arr[self.map0]
        ind = assemble_vector(self.ind_form)
        indicators = ind.array[self.dg0_dofs].copy()
        ind.destroy()
        return values, indicators

    def destroy(self):
        self.lu = None


class PersistentAdaptiveTransport:
    """Backend with the interface of coupling.UniformDGTransport."""

    def __init__(self, chemistry, time_step_s, settings=PersistentSettings(),
                 reference_porosity=0.35):
        from adaptdg.case4.constants import LENGTH_M
        self.chemistry, self.dt, self.settings = chemistry, float(time_step_s), settings
        self.chem_edges = np.linspace(0.0, LENGTH_M, chemistry.cells + 1)
        self.mesh = NestedMesh1D(self.chem_edges)
        self.phi0 = reference_porosity
        names = chemistry.components
        self.constrained = [i for i, n in enumerate(names) if n not in NOT_SIGN_CONSTRAINED]
        self.nodal = None          # (ncomp, transport cells, 2), mol/L
        self.means_seen = None     # (ncomp, chem cells)
        self._op = None
        self._op_edges = None
        self.substep_log = []

    # -- helpers ------------------------------------------------------------
    def _operator(self):
        edges = self.mesh.edges()
        if self._op is None or self._op_edges.shape != edges.shape or not np.array_equal(self._op_edges, edges):
            if self._op is not None:
                self._op.destroy()
            self._op, self._op_edges = _Operator(edges, self.phi0), edges
        return self._op

    def chem_means(self, nodal):
        idx = self.mesh.chem_index()
        w = np.diff(self.mesh.edges())
        sums = np.zeros((nodal.shape[0], self.chem_edges.size - 1))
        for c in range(nodal.shape[0]):
            np.add.at(sums[c], idx, w * nodal[c].mean(axis=1))
        return sums / np.diff(self.chem_edges)

    def checkpoint_arrays(self):
        return {"adap_leaves": np.asarray(self.mesh.leaves, dtype=float),
                "adap_nodal": self.nodal, "adap_means_seen": self.means_seen}

    def restore(self, arrays):
        self.mesh.leaves = [tuple(int(v) for v in row) for row in arrays["adap_leaves"]]
        self.nodal = arrays["adap_nodal"].copy()
        self.means_seen = arrays["adap_means_seen"].copy()

    # -- one transport substep ----------------------------------------------
    def step(self, state, flux_m_s):
        from adaptdg.marking import dorfler_mark_1d
        from adaptdg.transport import (
            conservative_positive_projection_1d, gradient_jump_indicator_1d)

        conc = np.asarray(state.concentrations_mol_l, dtype=float)
        ncomp = conc.shape[0]
        inlet = self.chemistry.inlet
        names = self.chemistry.components
        if self.nodal is None:
            self.nodal = np.repeat(conc[:, self.mesh.chem_index(), None], 2, axis=2)
        else:
            # Chemistry / geometry changes of the chemistry-cell means.
            delta = conc - self.means_seen
            idx = self.mesh.chem_index()
            if getattr(self, "reaction_update", "additive") == "proportional":
                # Shape-preserving update for consumed, sign-constrained components: every
                # transport sub-cell of the chemistry cell is scaled by m_new/m_old (the mean of
                # the chemistry cell is linear in the sub-cells, so it is reproduced exactly and
                # nonnegative nodes stay nonnegative); increases and unconstrained components
                # keep the uniform shift. Same rule as dg_history.py (2026-10-18).
                old_m, new_m = self.means_seen, conc
                ratio = np.ones_like(old_m)
                mask = (new_m < old_m) & (old_m > 0.0)
                keep = np.zeros(ncomp, bool); keep[list(self.constrained)] = True
                mask &= keep[:, None]
                ratio[mask] = np.maximum(new_m[mask], 0.0) / old_m[mask]
                r_cells, d_cells = ratio[:, idx, None], delta[:, idx, None]
                self.nodal = np.where(r_cells < 1.0, self.nodal * r_cells, self.nodal + d_cells)
            else:
                self.nodal = self.nodal + delta[:, idx, None]
        history = self.nodal
        s = self.settings
        chem_w = float(np.min(np.diff(self.chem_edges)))
        solves = 0
        while True:
            solves += 1
            edges = self.mesh.edges()
            widths = np.diff(edges)
            phi_t = np.asarray(state.porosity)[self.mesh.chem_index()]
            op = self._operator()
            op.set_coefficients(phi_t, flux_m_s, self.dt)
            raw, indicators, grads = [], [], []
            for c in range(ncomp):
                values, ind = op.solve(history[c, :, 0], history[c, :, 1], float(inlet[c]))
                raw.append(values)
                indicators.append(ind)
                grads.append(gradient_jump_indicator_1d(edges, values))
            if solves >= s.max_solves:
                break
            min_width = chem_w / 2 ** s.max_level
            marked = np.zeros(widths.size, dtype=bool)
            for ind in indicators + grads:
                marked |= dorfler_mark_1d(ind, widths, theta=s.theta, minimum_width=min_width,
                                          epsilon_eligible=s.epsilon_eligible)
            if not marked.any():
                break
            history = self.mesh.refine(marked, history)

        # Conservative projection (as the historical ADAP), per component.
        new = np.empty((ncomp, widths.size, 2))
        balance, positivity = {}, {}
        weights = 1000.0 * phi_t * widths
        for c in range(ncomp):
            previous_means = history[c].mean(axis=1)
            before = float(weights @ previous_means)
            values = raw[c]
            if c in self.constrained:
                values, correction = conservative_positive_projection_1d(
                    edges, values, edges, phi_t, previous_means, time_step_s=self.dt,
                    darcy_flux_m_s=flux_m_s, inlet_concentration=float(inlet[c]))
            else:
                after_raw = float(weights @ (0.5 * (values[:-1] + values[1:])))
                net_raw = self.dt * 1000.0 * flux_m_s * (inlet[c] - values[-1])
                correction = (before + net_raw - after_raw) / (
                    1000.0 * (float(phi_t @ widths) + self.dt * flux_m_s))
                values = values + correction
            outlet = float(values[-1])
            net = self.dt * 1000.0 * flux_m_s * (inlet[c] - outlet)
            after = float(weights @ (0.5 * (values[:-1] + values[1:])))
            residual = after - before - net
            balance[names[c]] = {
                "relative": abs(residual) / max(abs(before), abs(net), 1e-30),
                "absolute_mol_m2": residual, "net_boundary_moles_per_m2": net,
                "outlet_concentration_mol_l": outlet}
            positivity[names[c]] = {"min_mean_raw": float(raw[c].min()), "cells_fixed": 0,
                                    "moved_inventory_mol_m2": 0.0,
                                    "max_mean_correction_mol_l": float(abs(correction)),
                                    "cells_scaled": 0}
            new[c, :, 0], new[c, :, 1] = values[:-1], values[1:]

        # Coarsen where every component is represented by the parent.
        scales = np.maximum(np.ptp(new.reshape(ncomp, -1), axis=1), 1e-30)
        new, merged = self.mesh.coarsen(new, s.coarsen_tolerance, scales, self.constrained)
        means = self.chem_means(new)
        self.nodal, self.means_seen = new, means
        levels = self.mesh.levels()
        self.substep_log.append({"positivity": positivity, "balance": balance,
                                 "transport_cells": int(levels.size), "solves": solves,
                                 "merged": merged, "max_level": int(levels.max())})
        return means.copy(), {
            "balance_relative": {k: v["relative"] for k, v in balance.items()},
            "last_substep_transport_balance": balance,
            "projection_correction": {k: v["max_mean_correction_mol_l"] for k, v in positivity.items()},
            "minimum_nodal": {k: v["min_mean_raw"] for k, v in positivity.items()},
            "cells": int(levels.size), "solves": solves,
        }

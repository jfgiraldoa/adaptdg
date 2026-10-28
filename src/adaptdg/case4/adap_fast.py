# ruff: noqa: E702  (verbatim legacy code)
"""Stage 2a (fast path): the same persistent adaptive transport, assembled in NumPy.

``adap_persistent`` rebuilds a DOLFINx mesh, function spaces and forms at
every mesh change (twice per substep) and calls the repository's positivity
projection, which loops over cells in Python. Both are implementation
costs, not part of the method. This module keeps the method identical:

* ``NumpyOperator`` assembles exactly the forms of ``adap_persistent._Operator``
  (= ``transport_1d.ResidualMinimizationCase4Transport1D`` with a DG1
  history) with the closed-form 1D element matrices, and factorizes once per
  (mesh, porosity, flux);
* ``positive_projection`` is a vectorized copy of
  ``transport_1d.conservative_positive_projection_1d`` for meshes whose nodes
  are the cell edges (same weights, same bracketing and 100 bisections).

Both are checked against the originals by tests/stage2a_fast_tests.py.

Element matrices on a cell of width h (nodal linear basis):
  M = h/6 [[2,1],[1,2]],  K = 1/h [[1,-1],[-1,1]],  C_ij = int N_i N_j' = 1/2 [[-1,1],[-1,1]].
Saddle point [[G, B], [B^T, 0]] [r; c] = [L; 0] with
  G = s_K M + dt (h^2 beta^2 K + interior jumps 0.5|beta| + boundary 0.5|beta|),
  B = s_K M + dt beta C, plus dt*beta at the inlet (r_L0, c_0),
  L = s_K M u_prev + dt*beta*c_in at r_L0;  s_K = phi_K/phi0, beta = q/phi0.

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/adap_fast.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .adap_persistent import PersistentAdaptiveTransport


class NumpyOperator:
    def __init__(self, edges, reference_porosity):
        self.edges = np.asarray(edges, dtype=float)
        self.n = self.edges.size - 1
        self.h = np.diff(self.edges)
        self.phi0 = float(reference_porosity)
        self.key = None

    def set_coefficients(self, porosity_cells, flux, dt):
        key = (np.asarray(porosity_cells).tobytes(), float(flux), float(dt))
        if key == self.key:
            return
        n, h = self.n, self.h
        s = np.asarray(porosity_cells, dtype=float) / self.phi0
        beta = float(flux) / self.phi0
        ab = abs(beta)
        nr = 2 * n                       # residual dofs: (L_k, R_k) = (2k, 2k+1)
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r).ravel()), cols.append(np.asarray(c).ravel())
            vals.append(np.broadcast_to(v, np.broadcast(np.asarray(r), np.asarray(c)).shape).ravel())

        k = np.arange(n)
        rl, rr = 2 * k, 2 * k + 1
        cl, cr = nr + k, nr + k + 1      # CG1 nodes k, k+1
        # Riesz block G: mass + dt*h^2 beta^2 * stiffness = (h/6)[2,1;1,2] + dt*h*beta^2[1,-1;-1,1]
        # The Riesz mass is weighted by s = phi/phi0 like the storage term: with an
        # unweighted mass the residual-minimising step can be unstable (spectral
        # radius > 1) when phi varies strongly, e.g. next to a clogged cell.
        g_d = s * h / 3.0 + dt * h * beta ** 2
        g_o = s * h / 6.0 - dt * h * beta ** 2
        add(rl, rl, g_d), add(rr, rr, g_d), add(rl, rr, g_o), add(rr, rl, g_o)
        # interior jumps between R_k and L_{k+1}
        j = np.arange(n - 1)
        w = 0.5 * ab * dt
        add(rr[j], rr[j], w), add(rl[j + 1], rl[j + 1], w)
        add(rr[j], rl[j + 1], -w), add(rl[j + 1], rr[j], -w)
        # boundary terms
        add(rl[0], rl[0], w), add(rr[-1], rr[-1], w)
        # transport block B (rows residual, cols concentration) and its transpose
        m_d, m_o = s * h / 3.0, s * h / 6.0
        a = 0.5 * dt * beta              # dt*beta*C, C = 1/2 [[-1,1],[-1,1]]
        blocks = ((rl, cl, m_d - a), (rl, cr, m_o + a), (rr, cl, m_o - a), (rr, cr, m_d + a))
        for r_, c_, v in blocks:
            add(r_, c_, v), add(c_, r_, v)
        inlet = dt * max(beta, 0.0)      # incoming flux at x = 0 (normal -1)
        add(rl[0], cl[0], inlet), add(cl[0], rl[0], inlet)
        size = 3 * n + 1
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(size, size)).tocsc()
        self.lu = splu(A)
        self.s, self.dt, self.beta = s, float(dt), beta
        self.key = key

    def solve(self, left, right, inlet):
        n, h, s = self.n, self.h, self.s
        b = np.zeros(3 * n + 1)
        b[0::2][:n] = s * h / 6.0 * (2.0 * left + right)
        b[1:2 * n:2] = s * h / 6.0 * (left + 2.0 * right)
        b[0] += self.dt * max(self.beta, 0.0) * inlet
        x = self.lu.solve(b)
        rL, rR = x[0:2 * n:2], x[1:2 * n:2]
        values = x[2 * n:].copy()
        indicators = h / 3.0 * (rL ** 2 + rL * rR + rR ** 2)
        return values, indicators

    def destroy(self):
        self.lu = None


def positive_projection(edges, raw_values, porosity_cells, previous_cell_values, *,
                        time_step_s, darcy_flux_m_s, inlet_concentration):
    """Vectorized conservative_positive_projection_1d for nodes == edges."""
    widths = np.diff(edges)
    phi = np.asarray(porosity_cells, dtype=float)
    values = np.asarray(raw_values, dtype=float)
    q = float(darcy_flux_m_s)
    if q < 0.0:
        raise ValueError("This benchmark projection currently assumes left-to-right flow")
    lumped = np.zeros(values.size)
    # Same summation order as the original loop: left then right, cell by cell.
    half = 0.5 * phi * widths
    for i in range(widths.size):            # cheap: one float add per node
        lumped[i] += half[i]
        lumped[i + 1] += half[i]
    constraint = lumped.copy()
    constraint[-1] += time_step_s * q
    target = float(np.dot(phi * widths, previous_cell_values)) + time_step_s * q * inlet_concentration
    if target < 0.0:
        raise ValueError("A nonnegative concentration cannot satisfy a negative total inventory")
    ratio = constraint / lumped

    def inventory(lam):
        return float(np.dot(constraint, np.maximum(0.0, values + lam * ratio)))

    low, high = -1.0, 1.0
    for _ in range(100):
        if inventory(low) <= target:
            break
        low *= 2.0
    else:
        raise RuntimeError("Could not bracket lower positivity projection multiplier")
    for _ in range(100):
        if inventory(high) >= target:
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket upper positivity projection multiplier")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if inventory(middle) < target:
            low = middle
        else:
            high = middle
    projected = np.maximum(0.0, values + 0.5 * (low + high) * ratio)
    return projected, float(np.max(np.abs(projected - values)))


def local_conservative_positivity(edges, raw_values, porosity_cells, previous_cell_values, *,
                                  time_step_s, darcy_flux_m_s, inlet_concentration):
    """Local alternative to ``positive_projection`` (same balance equation).

    The global projection subtracts one multiplier from every node before
    clipping, so a deficit created anywhere is taken uniformly from the whole
    domain (it emptied the inlet carbon in Case 4 once a reactor clogged).
    Here, on the DG1 pairs of the CG1 solution:
      1. negative cell means are raised to zero with the deficit taken from
         the nearest positive cells (``local_mean_redistribution``);
      2. Zhang-Shu scaling makes both nodal values nonnegative (means kept);
      3. one multiplicative factor f restores the same inventory equation as
         the projection, w.m + dt q c_out = w.m_prev + dt q c_in, with c_out
         the right value of the last cell. f >= 0 keeps nonnegativity and
         changes every value by the same relative amount (|f - 1| is the
         solver's global balance defect, logged).
    Returns (left, right, info).
    """
    from .positivity import local_mean_redistribution, scale_slopes_nonnegative

    widths = np.diff(edges)
    w = np.asarray(porosity_cells, dtype=float) * widths
    values = np.asarray(raw_values, dtype=float)
    left, right = values[:-1].copy(), values[1:].copy()
    means = 0.5 * (left + right)
    fixed, red = local_mean_redistribution(means, w)
    shift = fixed - means
    left += shift
    right += shift
    left, right, scaled = scale_slopes_nonnegative(left, right)
    q, dt = float(darcy_flux_m_s), float(time_step_s)
    if q < 0.0:
        raise ValueError("This benchmark positivity currently assumes left-to-right flow")
    target = float(w @ np.asarray(previous_cell_values, dtype=float)) + dt * q * float(inlet_concentration)
    if target < 0.0:
        raise ValueError("A nonnegative concentration cannot satisfy a negative total inventory")
    denom = float(w @ (0.5 * (left + right))) + dt * q * float(right[-1])
    factor = target / denom if denom > 0.0 else 1.0
    left *= factor
    right *= factor
    raw_means = 0.5 * (values[:-1] + values[1:])
    info = {"cells_fixed": int(red["cells_fixed"]), "moved_inventory": float(red["moved_inventory"]),
            "max_radius": int(red["max_radius"]),
            "cells_scaled": int(scaled["cells_scaled"]),
            "balance_factor_minus_one": float(factor - 1.0),
            "max_mean_correction": float(np.max(np.abs(0.5 * (left + right) - raw_means)))}
    return left, right, info


def minmod_chemistry_scale(nodal, edges, chem_edges, chem_index, inlet):
    """Minmod limiter at the scale of the chemistry cells (``--positivity local_minmod_chem``).

    The leading-edge zigzag of ADAP lives in the chemistry-cell means (the transport-leaf
    minmod of ``limiter == "minmod"`` cannot see it). For every component and chemistry cell:
    L2-project the piecewise-linear transport leaves to one linear function (mean M, half-jump
    S), limit S with dg_history.minmod_limit against the neighbouring chemistry means (inlet
    value on the left), and where S changes replace the cell's leaves by the limited linear
    function. Chemistry-cell means are unchanged; porosity is constant inside a chemistry cell,
    so every inventory is unchanged. Unlimited cells keep their sub-cell (adaptive) profile.
    Returns (nodal, cells limited per component)."""
    from .dg_history import minmod_limit

    edges = np.asarray(edges, float); chem_edges = np.asarray(chem_edges, float)
    w = np.diff(edges); xc = 0.5 * (edges[:-1] + edges[1:])
    W = np.diff(chem_edges); Xc = 0.5 * (chem_edges[:-1] + chem_edges[1:])
    idx = np.asarray(chem_index); dx = xc - Xc[idx]
    out = nodal.copy(); counts = []
    for c in range(nodal.shape[0]):
        left, right = nodal[c, :, 0], nodal[c, :, 1]
        m_leaf, g_leaf = 0.5 * (left + right), (right - left) / w
        M = np.zeros(W.size); np.add.at(M, idx, w * m_leaf); M /= W
        # int (u - M)(x - Xc) dx over the chemistry cell, exact for piecewise-linear u
        mom = np.zeros(W.size); np.add.at(mom, idx, w * m_leaf * dx + g_leaf * w ** 3 / 12.0)
        a = mom / (W ** 3 / 12.0); S = 0.5 * a * W
        lo, hi, _ = minmod_limit(M - S, M + S, float(inlet[c]))
        S_lim = 0.5 * (hi - lo)
        changed = np.abs(S_lim - S) > 1e-14 * np.maximum(np.abs(M), 1e-30)
        counts.append(int(changed.sum()))
        if changed.any():
            a_lim = 2.0 * S_lim / W
            sel = changed[idx]
            j = idx[sel]
            out[c, sel, 0] = M[j] + a_lim[j] * (edges[:-1][sel] - Xc[j])
            out[c, sel, 1] = M[j] + a_lim[j] * (edges[1:][sel] - Xc[j])
    return out, counts


def fct_chemistry_scale(nodal, chem_means_fn, chem_index, m_old, phi, dx, q, inlet, dt, constrained):
    """Flux-corrected transport at the chemistry-cell scale (``--positivity local_fct``, Zalesak 1979 / Kuzmin).

    The leading-edge zigzag of ADAP lives in the chemistry-cell MEANS; slope limiters (minmod on the leaves or
    at the chemistry scale) leave the means untouched. Here the means themselves are made local-extremum-
    diminishing, conservatively:
      * high-order means m_H = chemistry means of ``nodal`` (the ADAP transport result);
      * low-order means m_L = one implicit upwind (DG0) step on the chemistry grid from the same start m_old;
      * interface fluxes of both schemes are recovered by telescoping from the inlet flux q*c_in
        (F_{j+1/2} = F_{j-1/2} - phi_j dx_j (m_j - m_old_j)/dt); the antidiffusive flux is A = F_H - F_L;
      * Zalesak limiting: alpha in [0,1] per interface so that every limited mean stays within
        [min, max] of {m_L, m_old} over the cell and its neighbours;
      * the limited mean replaces the high-order one inside each chemistry cell by scaling its leaves
        (sign-constrained components, both means > 0) or by a uniform shift otherwise (means are linear in the
        leaves, so the new mean is exact; nonnegative leaves stay nonnegative under scaling).
    The outlet interface is limited one-sidedly (bounds of the last cell), so the outflow flux of the substep
    changes by (alpha_out - 1) A_out / dt; the caller updates its boundary record from the new inventory.
    Returns (nodal, info per component: interfaces with alpha < 1, min alpha, max |mean change|)."""
    from .fv_transport import fv_upwind_step

    idx = np.asarray(chem_index); phi = np.asarray(phi, float); dx = np.asarray(dx, float)
    m_old = np.asarray(m_old, float); ncomp, n = m_old.shape
    m_high = chem_means_fn(nodal)
    m_low = fv_upwind_step(phi, phi, m_old, float(q), np.asarray(inlet, float), float(dt), float(dx[0]))
    cap = phi * dx                                               # inventory per unit concentration
    out = nodal.copy(); info = []
    for c in range(ncomp):
        f_in = float(q) * float(inlet[c])
        F_H = f_in - np.cumsum(cap * (m_high[c] - m_old[c]) / dt)   # fluxes at interfaces 1/2 .. N-1/2
        F_L = f_in - np.cumsum(cap * (m_low[c] - m_old[c]) / dt)
        A = (F_H - F_L) * dt                                     # antidiffusive mass per unit area, interfaces j+1/2
        lo_b = np.minimum(m_low[c], m_old[c]); hi_b = np.maximum(m_low[c], m_old[c])
        m_min = np.minimum.reduce([lo_b, np.r_[lo_b[0], lo_b[:-1]], np.r_[lo_b[1:], lo_b[-1]]])
        m_max = np.maximum.reduce([hi_b, np.r_[hi_b[0], hi_b[:-1]], np.r_[hi_b[1:], hi_b[-1]]])
        Ap = np.maximum(A, 0.0); Am = np.maximum(-A, 0.0)        # A_{j+1/2} > 0: mass leaves cell j, enters j+1
        P_plus = np.r_[0.0, Ap[:-1]] + Am                        # incoming to cell j (inlet interface: A = 0)
        P_minus = np.r_[0.0, Am[:-1]] + Ap                       # outgoing from cell j
        Q_plus = (m_max - m_low[c]) * cap; Q_minus = (m_low[c] - m_min) * cap
        with np.errstate(divide="ignore", invalid="ignore"):
            R_plus = np.where(P_plus > 0, np.minimum(1.0, Q_plus / P_plus), 1.0)
            R_minus = np.where(P_minus > 0, np.minimum(1.0, Q_minus / P_minus), 1.0)
        A_int = A[:-1]
        alpha = np.where(A_int > 0, np.minimum(R_plus[1:], R_minus[:-1]), np.minimum(R_plus[:-1], R_minus[1:]))
        # outlet interface: one-sided (bounds of the last cell only); the outflow flux changes accordingly
        alpha_out = R_minus[-1] if A[-1] > 0 else R_plus[-1]
        alpha = np.clip(np.r_[alpha, alpha_out], 0.0, 1.0)
        A_lim = alpha * A
        m_lim = m_low[c] - (A_lim - np.r_[0.0, A_lim[:-1]]) / cap
        change = m_lim - m_high[c]
        info.append({"fct_limited_interfaces": int(np.sum(alpha < 1.0 - 1e-12)),
                     "fct_min_alpha": float(alpha.min()) if alpha.size else 1.0,
                     "fct_max_mean_change_mol_l": float(np.abs(change).max())})
        if not np.any(np.abs(change) > 0.0):
            continue
        scale_ok = (c in constrained) & (m_high[c] > 0.0) & (m_lim > 0.0)
        ratio = np.where(scale_ok, m_lim / np.where(m_high[c] > 0.0, m_high[c], 1.0), 1.0)
        shift = np.where(scale_ok, 0.0, change)
        out[c] = out[c] * ratio[idx, None] + shift[idx, None]
    return out, info


class FastPersistentAdaptiveTransport(PersistentAdaptiveTransport):
    """PersistentAdaptiveTransport with the NumPy operator and projection.

    ``step`` is a copy of PersistentAdaptiveTransport.step whose only change
    is the projection call (adap_persistent stays frozen while its 10-year
    campaign runs; it is kept as the DOLFINx reference implementation).
    ``positivity_mode``: "global_projection" (historical ADAP, default) or
    "local" (``local_conservative_positivity``).
    """

    positivity_mode = "global_projection"

    def _operator(self):
        edges = self.mesh.edges()
        if self._op is None or self._op_edges.shape != edges.shape or not np.array_equal(self._op_edges, edges):
            self._op, self._op_edges = NumpyOperator(edges, self.phi0), edges
        return self._op

    def step(self, state, flux_m_s):
        from adaptdg.marking import dorfler_mark_1d
        from adaptdg.transport import gradient_jump_indicator_1d

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
        befores = []
        for c in range(ncomp):
            previous_means = history[c].mean(axis=1)
            before = float(weights @ previous_means); befores.append(before)
            values = raw[c]
            local = None
            if c in self.constrained and self.positivity_mode == "local":
                left_c, right_c, local = local_conservative_positivity(
                    edges, values, phi_t, previous_means, time_step_s=self.dt,
                    darcy_flux_m_s=flux_m_s, inlet_concentration=float(inlet[c]))
                correction = local["max_mean_correction"]
            elif c in self.constrained:
                values, correction = positive_projection(
                    edges, values, phi_t, previous_means, time_step_s=self.dt,
                    darcy_flux_m_s=flux_m_s, inlet_concentration=float(inlet[c]))
            else:
                after_raw = float(weights @ (0.5 * (values[:-1] + values[1:])))
                net_raw = self.dt * 1000.0 * flux_m_s * (inlet[c] - values[-1])
                correction = (before + net_raw - after_raw) / (
                    1000.0 * (float(phi_t @ widths) + self.dt * flux_m_s))
                values = values + correction
            if local is None:
                left_c, right_c = values[:-1], values[1:]
            outlet = float(right_c[-1])
            net = self.dt * 1000.0 * flux_m_s * (inlet[c] - outlet)
            after = float(weights @ (0.5 * (left_c + right_c)))
            residual = after - before - net
            balance[names[c]] = {
                "relative": abs(residual) / max(abs(before), abs(net), 1e-30),
                "absolute_mol_m2": residual, "net_boundary_moles_per_m2": net,
                "outlet_concentration_mol_l": outlet}
            positivity[names[c]] = {"min_mean_raw": float(raw[c].min()),
                                    "cells_fixed": local["cells_fixed"] if local else 0,
                                    "moved_inventory_mol_m2": 1000.0 * local["moved_inventory"] if local else 0.0,
                                    "max_mean_correction_mol_l": float(abs(correction)),
                                    "cells_scaled": local["cells_scaled"] if local else 0}
            if local:
                positivity[names[c]]["balance_factor_minus_one"] = local["balance_factor_minus_one"]
            if getattr(self, "limiter", None) == "minmod":
                # Same minmod as dg_history.minmod_limit (all components, cell means unchanged, so
                # the inventory above is exact); applied after the balance record of the solve.
                from .dg_history import minmod_limit
                left_c, right_c, n_lim = minmod_limit(np.asarray(left_c, float), np.asarray(right_c, float), float(inlet[c]))
                positivity[names[c]]["cells_limited"] = n_lim
            new[c, :, 0], new[c, :, 1] = left_c, right_c

        if getattr(self, "limiter", None) == "fct":
            # Chemistry-scale FCT on the means (after the balance record; total inventory unchanged).
            new, fct_info = fct_chemistry_scale(new, self.chem_means, self.mesh.chem_index(), conc,
                                                np.asarray(state.porosity), np.diff(self.chem_edges),
                                                flux_m_s, inlet, self.dt, set(self.constrained))
            for c in range(ncomp):
                positivity[names[c]].update(fct_info[c])
                # FCT changes the inventory only through the outlet interface: move that change to the
                # boundary record (net = inflow - outflow), keep the high-order residual as the defect.
                b = balance[names[c]]; after_fct = float(weights @ new[c].mean(axis=1))
                old_after = befores[c] + b["net_boundary_moles_per_m2"] + b["absolute_mol_m2"]
                b["net_boundary_moles_per_m2"] += after_fct - old_after
                b["outlet_concentration_mol_l"] = float(inlet[c]) - b["net_boundary_moles_per_m2"] / (self.dt * 1000.0 * flux_m_s) if flux_m_s else b["outlet_concentration_mol_l"]
        if getattr(self, "limiter", None) == "minmod_chem":
            # Chemistry-scale minmod (applied after the balance record; means unchanged).
            new, n_lim = minmod_chemistry_scale(new, edges, self.chem_edges, self.mesh.chem_index(), inlet)
            for c in range(ncomp):
                positivity[names[c]]["chem_cells_limited"] = n_lim[c]

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


class _FrozenNestedMesh:
    """Mixin: a nested mesh that never coarsens."""

    def coarsen(self, nodal, tolerance, scales, constrained):
        return nodal, 0


class UniformFineTransport(FastPersistentAdaptiveTransport):
    """Reference transport on the same chemistry grid: every chemistry cell is
    split uniformly into 2**level transport cells, with the same operator,
    history, increments and projection as the adaptive path, but no marking
    and no coarsening. Converging it in ``level`` isolates the transport error
    at fixed chemistry resolution."""

    def __init__(self, chemistry, time_step_s, level, reference_porosity=0.35):
        from dataclasses import replace
        from .adap_persistent import NestedMesh1D, PersistentSettings
        super().__init__(chemistry, time_step_s,
                         replace(PersistentSettings(), max_solves=1, max_level=level),
                         reference_porosity)
        mesh_cls = type("UniformNestedMesh1D", (_FrozenNestedMesh, NestedMesh1D), {})
        self.mesh = mesh_cls(self.chem_edges)
        self.mesh.leaves = [(j, level, k) for j in range(chemistry.cells) for k in range(2 ** level)]

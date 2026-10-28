"""Stage 2c: ASFEM adaptivity ingredients for the persistent Case 4 ADAP (1D).

Ported from ~/Documents/Claude/Projects/ASFEM (residual_minimization.py,
adaptive_native.py), specialised to Case 4 (pure advection: D = 0, no
reaction weight, backward Euler, constant substep dt):

* Riesz norm ``paper`` (Giraldo & Calo 2023): streamline weight h_K/|beta|,
  i.e. tau*h*|beta|*(y', v') in 1D, instead of the Case 4 tau*h^2*beta^2.
  ``norm="case4"`` keeps the historical weight for attribution.
* Local indicator = the full Riesz norm of the residual per cell (cell terms,
  half of every interior jump to each neighbour, boundary terms), instead of
  only int r^2.
* Dörfler bulk chasing with ties on the squared indicators (fraction 0.0625,
  tie 0.2, as the paper).
* Stopping by tolerance: per marking component, estimator <= tol * ||c||_R
  (scale-free; the components differ by four orders of magnitude). Only the
  element totals (C, Ca, Na, S) mark and stop; H, O, H2O and Charge follow.
* Optional feature markers: resolution h|c'| > theta*scale and bounds (vertex
  outside the previous/inlet range by more than delta*scale), with a fixed
  per-component scale = |inlet - initial resident| (not the current range).

Not ported on purpose: Courant time-step control (DG and ADAP keep the same
dt) and Dirichlet-data oscillation (a point boundary in 1D).
The mesh, history, increments, projection and coarsening are those of
stage 2a/2b (transport_bdf2.BDF2AdaptiveTransport, adap_fast).

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/adap_asfem.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
``dorfler_with_ties`` is imported from ``adaptdg.marking`` (verbatim identical body).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from adaptdg.marking import dorfler_with_ties  # noqa: F401 (public name kept)
from .transport_bdf2 import BDF2AdaptiveTransport, _coefficients


class RieszOperator:
    """NumPy CG1/DG1 residual minimization with a selectable Riesz norm."""

    def __init__(self, edges, reference_porosity, norm="paper"):
        if norm not in {"paper", "case4", "paper_cfloor", "paper_ccap"}:
            raise ValueError("norm must be paper, case4, paper_cfloor or paper_ccap")
        self.edges = np.asarray(edges, dtype=float)
        self.n = self.edges.size - 1
        self.h = np.diff(self.edges)
        self.phi0 = float(reference_porosity)
        self.norm = norm
        self.key = None

    def set_coefficients(self, porosity_cells, flux, dt):
        key = (np.asarray(porosity_cells).tobytes(), float(flux), float(dt))
        if key == self.key:
            return
        n, h = self.n, self.h
        s = np.asarray(porosity_cells, dtype=float) / self.phi0
        beta = float(flux) / self.phi0
        ab = abs(beta)
        # Time scale of the norm per cell: tau (paper, case4) or, for
        # paper_cfloor, max(tau, h_K/|beta|) so that the advective terms do not
        # vanish against the L2 term when the cell Courant number is small.
        scale = np.full(n, dt) if self.norm != "paper_cfloor" else np.maximum(dt, h / max(ab, 1e-300))
        # streamline coefficient multiplying int r' v' (= stiffness/h per cell)
        stream = scale * ab if self.norm != "case4" else dt * h * beta ** 2
        if self.norm == "paper_ccap":
            # Streamline weight capped by the cell Courant number:
            # tau*h*|beta|*min(1, tau*|beta|/h) -> tau^2 beta^2 for small Courant
            # (dimensionally consistent; equals the paper weight for Courant >= 1).
            stream = stream * np.minimum(1.0, dt * ab / h)
        jw_int = 0.5 * ab * np.maximum(scale[:-1], scale[1:])
        self.stream, self.jw_int = stream, jw_int
        self.jw_left, self.jw_right = 0.5 * ab * scale[0], 0.5 * ab * scale[-1]
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r).ravel()), cols.append(np.asarray(c).ravel())
            vals.append(np.broadcast_to(v, np.broadcast(np.asarray(r), np.asarray(c)).shape).ravel())

        k = np.arange(n)
        rl, rr = 2 * k, 2 * k + 1
        nr = 2 * n
        cl, cr = nr + k, nr + k + 1
        # The Riesz mass is weighted by s = phi/phi0 like the storage term: with an
        # unweighted mass the residual-minimising step can be unstable (spectral
        # radius > 1) when phi varies strongly, e.g. next to a clogged cell.
        g_d, g_o = s * h / 3.0 + stream, s * h / 6.0 - stream
        add(rl, rl, g_d), add(rr, rr, g_d), add(rl, rr, g_o), add(rr, rl, g_o)
        j = np.arange(n - 1)
        w = jw_int
        add(rr[j], rr[j], w), add(rl[j + 1], rl[j + 1], w)
        add(rr[j], rl[j + 1], -w), add(rl[j + 1], rr[j], -w)
        add(rl[0], rl[0], self.jw_left), add(rr[-1], rr[-1], self.jw_right)
        G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(nr, nr)).tocsr()
        m_d, m_o = s * h / 3.0, s * h / 6.0
        a = 0.5 * dt * beta
        for r_, c_, v in ((rl, cl, m_d - a), (rl, cr, m_o + a), (rr, cl, m_o - a), (rr, cr, m_d + a)):
            add(r_, c_, v), add(c_, r_, v)
        inlet = dt * max(beta, 0.0)
        add(rl[0], cl[0], inlet), add(cl[0], rl[0], inlet)
        size = 3 * n + 1
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(size, size)).tocsc()
        self.lu, self.G = splu(A), G
        self.s, self.dt, self.beta = s, float(dt), beta
        self.key = key

    def _cell_norm2(self, L, R):
        """Per-cell Riesz-norm contributions of a DG1 field (L, R nodal)."""
        # mass (h/6)[2,1;1,2] and streamline stream*[1,-1;-1,1] (stream already
        # contains the 1/h of int r'^2: dt|beta| paper, dt*h*beta^2 case4)
        out = self.s * self.h / 3.0 * (L ** 2 + L * R + R ** 2) + self.stream * (R - L) ** 2
        jump2 = (L[1:] - R[:-1]) ** 2 * self.jw_int
        out[:-1] += 0.5 * jump2
        out[1:] += 0.5 * jump2
        out[0] += self.jw_left * L[0] ** 2
        out[-1] += self.jw_right * R[-1] ** 2
        return out

    def solve(self, left, right, inlet):
        n, h, s = self.n, self.h, self.s
        b = np.zeros(3 * n + 1)
        b[0:2 * n:2] = s * h / 6.0 * (2.0 * left + right)
        b[1:2 * n:2] = s * h / 6.0 * (left + 2.0 * right)
        b[0] += self.dt * max(self.beta, 0.0) * inlet
        x = self.lu.solve(b)
        rL, rR = x[0:2 * n:2], x[1:2 * n:2]
        values = x[2 * n:].copy()
        return values, self._cell_norm2(rL, rR)

    def solution_norm(self, values):
        """||c||_R of the CG1 field (as a continuous DG1 field)."""
        return float(np.sqrt(max(np.sum(self._cell_norm2(values[:-1], values[1:])), 0.0)))


@dataclass(frozen=True)
class ASFEMSettings:
    tolerance: float = 1e-2          # relative: estimator <= tol * ||c||_R per marking component
    norm: str = "paper"
    fraction: float = 0.0625
    tie: float = 0.2
    max_solves: int = 6
    max_level: int = 4               # finest cell = chemistry width / 2**max_level
    resolution_threshold: float | None = None
    bound_tolerance: float | None = None
    coarsen_tolerance: float = 1e-3


class ASFEMAdaptiveTransport(BDF2AdaptiveTransport):
    """Persistent ADAP with the paper norm, full indicators and tolerance stopping."""

    MARKING = ("C", "Ca", "Na", "S")

    def __init__(self, chemistry, time_step_s, settings=ASFEMSettings(), scheme="be",
                 reference_porosity=0.35):
        super().__init__(chemistry, time_step_s, settings, scheme, reference_porosity)
        names = list(chemistry.components)
        self.marking = [names.index(n) for n in self.MARKING if n in names]
        self.scales = None

    def _operator(self):
        edges = self.mesh.edges()
        if self._op is None or self._op_edges.shape != edges.shape or not np.array_equal(self._op_edges, edges):
            self._op, self._op_edges = RieszOperator(edges, self.phi0, self.settings.norm), edges
        return self._op

    def checkpoint_arrays(self):
        out = super().checkpoint_arrays()
        if self.scales is not None:
            out["asfem_scales"] = self.scales
        return out

    def restore(self, arrays):
        super().restore(arrays)
        if "asfem_scales" in arrays:
            self.scales = arrays["asfem_scales"].copy()

    def step(self, state, flux_m_s):
        from .adap_fast import positive_projection

        conc = np.asarray(state.concentrations_mol_l, dtype=float)
        ncomp = conc.shape[0]
        names, inlet = self.chemistry.components, self.chemistry.inlet
        s, t = self.settings, self.time
        idx = self.mesh.chem_index()
        porosity = np.asarray(state.porosity, dtype=float)
        if self.scales is None:
            self.scales = np.maximum(np.abs(inlet - conc.mean(axis=1)), 1e-30)
        if t.current is None:
            t.current = np.repeat(conc[:, idx, None], 2, axis=2)
        else:
            t.apply_increment(t.means_seen[:, idx], conc[:, idx], (t.phi_seen / porosity)[idx])
        t.phi_seen = porosity.copy()
        dt_eff, a, b = _coefficients(t.older is not None, self.dt, t.scheme)
        chem_w = float(np.min(np.diff(self.chem_edges)))
        min_width = chem_w / 2 ** s.max_level
        solves, stopped, estimators = 0, "", []
        while True:
            solves += 1
            edges = self.mesh.edges()
            widths = np.diff(edges)
            phi_t = porosity[self.mesh.chem_index()]
            hist = a * t.current + (b * t.older if b else 0.0)
            op = self._operator()
            op.set_coefficients(phi_t, flux_m_s, dt_eff)
            raw, indicators = [], []
            for c in range(ncomp):
                values, ind = op.solve(hist[c, :, 0], hist[c, :, 1], float(inlet[c]))
                raw.append(values)
                indicators.append(ind)
            failing, features = [], np.zeros(widths.size, dtype=bool)
            rel, absolute = {}, {}
            for c in self.marking:
                est = float(np.sqrt(np.sum(indicators[c])))
                norm = op.solution_norm(raw[c])
                rel[names[c]] = est / max(norm, 1e-300)
                absolute[names[c]] = est
                if est > s.tolerance * norm:
                    failing.append(c)
                scale = self.scales[c]
                if s.resolution_threshold:
                    features |= np.abs(np.diff(raw[c])) > s.resolution_threshold * scale
                if s.bound_tolerance is not None:
                    lo = min(float(hist[c].min()), float(inlet[c]))
                    hi = max(float(hist[c].max()), float(inlet[c]))
                    slack = s.bound_tolerance * scale
                    vmax = np.maximum(raw[c][:-1], raw[c][1:])
                    vmin = np.minimum(raw[c][:-1], raw[c][1:])
                    features |= (vmax > hi + slack) | (vmin < lo - slack)
            eligible = widths > min_width * (1.0 + 1e-12)
            features &= eligible
            estimators.append((rel, absolute))
            if not failing and not features.any():
                stopped = "tolerance"
                break
            if solves >= s.max_solves:
                stopped = "maximum_solves"
                break
            marked = features.copy()
            for c in failing:
                marked |= dorfler_with_ties(indicators[c], fraction=s.fraction, tie=s.tie, eligible=eligible)
            if not marked.any():
                stopped = "nothing_marked"
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
        stacked = self._both()
        scales = np.maximum(np.ptp(stacked.reshape(stacked.shape[0], -1), axis=1), 1e-30)
        constrained = self.constrained + ([ncomp + c for c in self.constrained] if t.older is not None else [])
        stacked, merged = self.mesh.coarsen(stacked, s.coarsen_tolerance, scales, constrained)
        self._split(stacked)
        t.means_seen = self.chem_means(t.current)
        levels = self.mesh.levels()
        self.substep_log.append({"positivity": positivity, "balance": balance,
                                 "transport_cells": int(levels.size), "solves": solves,
                                 "merged": merged, "max_level": int(levels.max()),
                                 "stopped_by": stopped, "relative_estimators": estimators[-1][0],
                                 "estimators": estimators[-1][1]})
        return t.means_seen.copy(), {
            "balance_relative": {k: v["relative"] for k, v in balance.items()},
            "last_substep_transport_balance": balance,
        }

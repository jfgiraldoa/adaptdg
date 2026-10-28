"""New 1D ADAP driver (paper method, current code), generic in the chemistry.

Extracted from ASRTM-PHREEQC (asrtm-phreeqc-public @ a152b91, methods/onedim/adap1d.py): numerics,
class and method names unchanged. Deviations: the relative import of the operator points to
``adaptdg.rm_1d``, ``Adap1DRun._marking`` calls ``adaptdg.marking.dorfler_union_1d_adap`` (the
exact adapter of the former body) and the duck-typed chemistry interface is documented by the
``Chemistry1D`` protocol (documentation only, never enforced).

Same ingredients as the 2D code that closes the paper:
  * residual-minimisation transport with diffusion (onedim.rm1d, verified against the
    repository operator to 1e-14), one factorisation per mesh;
  * persistent dyadic transport mesh: refined by the repository's Dörfler union
    (theta, eligibility, epsilon_chem, tie tolerance) and coarsened by sibling pairs whose
    L2 projection changes no component by more than ``coarsen_tol`` of its range; the
    mesh may be coarser than the chemistry grid where the solution is smooth;
  * DG1 transport history carried across steps (exact restriction on refinement,
    L2 projection on coarsening) and the chemistry entering as the reaction increment of
    each chemistry cell, L2-projected onto the transport cells (conservative);
  * exact conservative transfers between the transport mesh and the uniform chemistry
    grid; ``legacy_clip`` of negative chemistry inputs (as in the 2D comparison), with
    the added mass recorded.
Backward Euler, flow left to right, inlet at x = lower.

Chemistry: any object with ``snapshot.concentrations`` (dict of chemistry-grid arrays),
``boundary`` (dict) and ``react(dict, time_step=dt)`` returning a snapshot (see ``Chemistry1D``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Protocol

import numpy as np

from adaptdg.marking import dorfler_union_1d_adap
from adaptdg.rm_1d import RMOperator1D


class Chemistry1D(Protocol):
    """Documentation of the duck-typed chemistry object used by :class:`Adap1DRun`.

    Used attributes (nothing else is touched; the protocol is never enforced):
    ``snapshot.concentrations``: mapping component -> (chem_cells,) array, read once at start;
    ``boundary``: mapping component -> inflow concentration (``float(boundary[c])``);
    ``react(inputs, time_step=dt)``: ``inputs`` maps component -> (chem_cells,) array (negative
    values already fixed per ``Adap1DSettings.positivity``); returns an object whose
    ``concentrations`` maps component -> (chem_cells,) array (the reacted state).
    """

    snapshot: object
    boundary: Mapping[str, float]

    def react(self, inputs: Mapping[str, np.ndarray], *, time_step: float) -> object: ...


@dataclass
class Adap1DSettings:
    theta: float = 0.3
    epsilon_chem: float = 0.1
    eligibility_factor: float = 0.24
    tie_tolerance: float = 0.35
    initial_cells: int = 4
    first_step_solves: int = 30
    step_solves: int = 3
    coarsen_tol: float = 1e-3
    coarsen_interval: int = 1         # coarsen every k steps (k = 1: every step)
    penalty: float = 8.0
    positivity: str = "clip"          # "clip" (legacy_clip), "conservative" (per component) or "mixing"


# ---------------------------------------------------------------- exact 1D transfers
def _segments(a_edges, b_edges):
    """Union partition of two meshes of the same interval: (left, right, ia, ib)."""
    e = np.unique(np.concatenate([a_edges, b_edges]))
    left, right = e[:-1], e[1:]
    keep = right > left
    left, right = left[keep], right[keep]
    mid = 0.5 * (left + right)
    ia = np.clip(np.searchsorted(a_edges, mid) - 1, 0, a_edges.size - 2)
    ib = np.clip(np.searchsorted(b_edges, mid) - 1, 0, b_edges.size - 2)
    return left, right, ia, ib


def dg1_to_cells(edges, dg1, target_edges):
    """Exact cell means on ``target_edges`` of a DG1 field (n, 2) on ``edges``."""
    left, right, ia, ib = _segments(edges, target_edges)
    x0, h = edges[ia], np.diff(edges)[ia]
    mid = 0.5 * (left + right)
    val = dg1[ia, 0] + (dg1[ia, 1] - dg1[ia, 0]) * (mid - x0) / h      # exact for linear
    out = np.bincount(ib, weights=val * (right - left), minlength=target_edges.size - 1)
    return out / np.diff(target_edges)


def cells_to_dg1(src_edges, values, edges):
    """L2 projection onto DG1 on ``edges`` of a piecewise-constant field on ``src_edges``."""
    left, right, ia, ib = _segments(src_edges, edges)
    v = np.asarray(values)[ia]
    h = np.diff(edges)
    xc = 0.5 * (edges[:-1] + edges[1:])
    mean = np.bincount(ib, weights=v * (right - left), minlength=h.size) / h
    mom = np.bincount(ib, weights=v * 0.5 * ((right - xc[ib]) ** 2 - (left - xc[ib]) ** 2), minlength=h.size)
    slope = 12.0 * mom / h ** 3
    return np.column_stack((mean - 0.5 * slope * h, mean + 0.5 * slope * h))


def positivity_fix(values, mode):
    """Negative chemistry inputs on a uniform grid. Returns (fixed, mass_added_per_width).
    "clip": set to zero (adds mass). "conservative": set to zero and take the same amount
    from the nearest positive cells (window grown symmetrically until it suffices,
    proportionally to their values): mass conserved exactly, positivity kept."""
    u = np.array(values, dtype=float)
    neg = np.flatnonzero(u < 0.0)
    if neg.size == 0:
        return u, 0.0
    if mode == "clip":
        added = float(-u[neg].sum())
        u[neg] = 0.0
        return u, added
    if mode != "conservative":
        raise ValueError(mode)
    deficit = -u[neg]
    u[neg] = 0.0
    n = u.size
    for i, d in zip(neg, deficit):
        r = 1
        while True:
            lo, hi = max(i - r, 0), min(i + r + 1, n)
            avail = float(u[lo:hi].sum())
            if avail >= d or (lo == 0 and hi == n):
                break
            r += 1
        if avail <= 0.0:
            continue                       # nothing to take from: the total is non-positive
        u[lo:hi] *= 1.0 - min(d, avail) / avail
    return u, 0.0


def positivity_mixing(fields, rel_tol=1e-3):
    """Charge-preserving conservative positivity fix (same linear operation for every
    component, like PHREEQC's dispersion mixing). For each cell where some component is
    negative, a symmetric window is grown until every component's window sum is >= 0;
    the window is then blended with its mean, u <- (1 - theta) u + theta mean(u), with the
    smallest theta in [0, 1] that leaves every component >= 0. The window sum of every
    component is unchanged (mass conserved) and, the operation being identical for all
    components, each cell stays electroneutral (charge is zero after linear transport of
    neutral data). Only significant negatives (below -rel_tol*max|u_c|, default 1e-3) are
    mixed; smaller ones are clipped and their added mass is returned (it is recorded and
    must stay negligible), so the mixing does not smear fronts for noise-level undershoots. Returns (fixed dict, number of windows, added mass dict)."""
    names = list(fields)
    U = np.array([fields[c] for c in names], dtype=float)
    n = U.shape[1]
    tol = rel_tol * np.maximum(np.max(np.abs(U), axis=1), 1e-300)
    windows, i = 0, 0
    while i < n:
        if not np.any(U[:, i] < -tol):
            i += 1
            continue
        lo, hi = i, i + 1
        while np.any(U[:, lo:hi].sum(axis=1) < 0.0) and (lo > 0 or hi < n):
            lo, hi = max(lo - 1, 0), min(hi + 1, n)
        W = U[:, lo:hi]
        m = W.mean(axis=1, keepdims=True)
        neg = W < 0.0
        theta = np.max(np.where(neg, -W / np.where(neg, m - W, 1.0), 0.0))
        U[:, lo:hi] = (1.0 - min(theta, 1.0)) * W + min(theta, 1.0) * m
        windows += 1
        i = lo                       # recheck the window (round-off) and move on
        if not np.any(U[:, lo:hi] < -tol[:, None]):
            i = hi
    neg = np.minimum(U, 0.0)
    added = {c: float(-neg[k].sum()) for k, c in enumerate(names)}
    U = np.maximum(U, 0.0)
    return {c: U[k] for k, c in enumerate(names)}, windows, added


# ---------------------------------------------------------------- driver
@dataclass
class Adap1DRun:
    chemistry: object
    components: tuple
    lower: float
    upper: float
    chem_cells: int
    time_step: float
    diffusion: float
    velocity: float
    settings: Adap1DSettings = field(default_factory=Adap1DSettings)

    def __post_init__(self):
        s = self.settings
        self.chem_edges = np.linspace(self.lower, self.upper, self.chem_cells + 1)
        self.chem_width = (self.upper - self.lower) / self.chem_cells
        self.edges = np.linspace(self.lower, self.upper, s.initial_cells + 1)
        self.level = np.zeros(s.initial_cells, dtype=int)
        snap = self.chemistry.snapshot.concentrations
        self.history = {c: cells_to_dg1(self.chem_edges, snap[c], self.edges) for c in self.components}
        self.boundary = {c: float(self.chemistry.boundary[c]) for c in self.components}
        self._op, self._op_key = None, None
        self.step_index = 0
        self.timings = {"transport": 0.0, "chemistry": 0.0, "adaptivity": 0.0, "transfer": 0.0}
        self.counts = {"factorisations": 0, "component_solves": 0, "refinements": 0, "coarsenings": 0}
        self.clip_added = {c: 0.0 for c in self.components}

    # ---- mesh operations
    def _operator(self):
        key = self.edges.tobytes()
        if key != self._op_key:
            self._op = RMOperator1D(self.edges, time_step=self.time_step, diffusion=self.diffusion,
                                    velocity=self.velocity, penalty=self.settings.penalty)
            self._op_key = key
            self.counts["factorisations"] += 1
        return self._op

    def _refine(self, marked, fields):
        """Bisect marked cells; DG1 fields restricted exactly (children of a linear)."""
        idx = np.flatnonzero(marked)
        if idx.size == 0:
            return fields
        mids = 0.5 * (self.edges[idx] + self.edges[idx + 1])
        rep = np.where(marked, 2, 1)
        new_level = np.repeat(self.level, rep)
        first = np.cumsum(rep) - rep
        new_level[first[idx]] += 1
        new_level[first[idx] + 1] += 1
        out = {}
        for c, f in fields.items():
            g = np.repeat(f, rep, axis=0)
            m = 0.5 * (f[idx, 0] + f[idx, 1])
            g[first[idx], 1] = m
            g[first[idx] + 1, 0] = m
            out[c] = g
        self.edges = np.sort(np.concatenate([self.edges, mids]))
        self.level = new_level
        self.counts["refinements"] += idx.size
        return out

    def _coarsen(self, fields):
        """Merge dyadic sibling pairs whose L2 projection to the parent changes no field
        by more than coarsen_tol of its range (one pass)."""
        e, lev = self.edges, self.level
        n = lev.size
        if n < 2:
            return fields
        i = np.arange(n - 1)
        span0 = (self.upper - self.lower) / self.settings.initial_cells
        width = span0 / 2.0 ** lev[i]
        pos = np.rint((e[i] - self.lower) / width).astype(np.int64)
        cand = (lev[i] > 0) & (lev[i] == lev[i + 1]) & (pos % 2 == 0)
        # parent L2 projection of two linear pieces (children of width w, parent 2w)
        w = np.diff(e)[i]
        ok = cand.copy()
        proj = {}
        for c, f in fields.items():
            a, b = f[i], f[i + 1]
            mean = 0.25 * (a[:, 0] + a[:, 1] + b[:, 0] + b[:, 1])
            # first moment about the parent centre (children on [-w,0] and [0,w])
            ma = w ** 2 * (-a[:, 0] / 3 - a[:, 1] / 6)
            mb = w ** 2 * (b[:, 0] / 6 + b[:, 1] / 3)
            slope = 12.0 * (ma + mb) / (2 * w) ** 3
            pl, pr = mean - slope * w, mean + slope * w
            err = np.maximum.reduce([np.abs(a[:, 0] - pl), np.abs(a[:, 1] - 0.5 * (pl + pr)),
                                     np.abs(b[:, 0] - 0.5 * (pl + pr)), np.abs(b[:, 1] - pr)])
            rng = max(float(np.ptp(f)), 1e-300)
            ok &= err <= self.settings.coarsen_tol * rng
            proj[c] = (pl, pr)
        # no overlapping pairs: keep a pair only if the previous cell is not its partner
        sel = np.zeros(n - 1, dtype=bool)
        k = 0
        while k < n - 1:
            if ok[k]:
                sel[k] = True
                k += 2
            else:
                k += 1
        if not sel.any():
            return fields
        drop = np.zeros(n, dtype=bool)
        drop[np.flatnonzero(sel) + 1] = True
        idx = np.flatnonzero(sel)
        out = {}
        for c, f in fields.items():
            g = f.copy()
            pl, pr = proj[c]
            g[idx, 0], g[idx, 1] = pl[idx], pr[idx]
            out[c] = g[~drop]
        keep_edge = np.ones(e.size, dtype=bool)
        keep_edge[idx + 1] = False
        self.edges = e[keep_edge]
        lev2 = lev.copy()
        lev2[idx] -= 1
        self.level = lev2[~drop]
        self.counts["coarsenings"] += int(sel.sum())
        return out

    def _marking(self, indicators):
        s = self.settings
        return dorfler_union_1d_adap(
            indicators, np.diff(self.edges), self.chem_width, theta=s.theta,
            eligibility_factor=s.eligibility_factor, epsilon_chem=s.epsilon_chem,
            tie_tolerance=s.tie_tolerance, components=self.components)

    # ---- one physical step
    def step(self):
        s = self.settings
        solves = s.first_step_solves if self.step_index == 0 else s.step_solves
        hist = self.history
        for k in range(solves):
            t0 = perf_counter()
            op = self._operator()
            res = {c: op.solve(hist[c], self.boundary[c]) for c in self.components}
            self.counts["component_solves"] += len(self.components)
            self.timings["transport"] += perf_counter() - t0
            if k == solves - 1:
                break
            t0 = perf_counter()
            marked = self._marking({c: res[c]["cell_indicators"] for c in self.components})
            if not marked.any():
                self.timings["adaptivity"] += perf_counter() - t0
                break
            hist = self._refine(marked, hist)
            self.timings["adaptivity"] += perf_counter() - t0
        transported = {c: res[c]["history_dg1"] for c in self.components}
        t0 = perf_counter()
        on_chem = {c: dg1_to_cells(self.edges, transported[c], self.chem_edges) for c in self.components}
        if s.positivity == "mixing":
            clipped, sweeps, added = positivity_mixing(on_chem)
            self.mixing_sweeps = getattr(self, "mixing_sweeps", 0) + sweeps
            for c in self.components:
                self.clip_added[c] += added[c] * self.chem_width
        else:
            clipped = {}
            for c in self.components:
                clipped[c], added = positivity_fix(on_chem[c], s.positivity)
                self.clip_added[c] += added * self.chem_width
        self.timings["transfer"] += perf_counter() - t0
        t0 = perf_counter()
        reacted = self.chemistry.react(clipped, time_step=self.time_step).concentrations
        self.timings["chemistry"] += perf_counter() - t0
        t0 = perf_counter()
        lifted = {}
        for c in self.components:
            inc = np.asarray(reacted[c]) - on_chem[c]          # raw transported, as in the 2D lift
            lifted[c] = transported[c] + cells_to_dg1(self.chem_edges, inc, self.edges)
        self.timings["transfer"] += perf_counter() - t0
        t0 = perf_counter()
        if self.step_index % s.coarsen_interval == 0:
            self.history = self._coarsen(lifted)
        else:
            self.history = lifted
        self.timings["adaptivity"] += perf_counter() - t0
        self.step_index += 1
        self.last = {"transported": transported, "transported_on_chemistry": on_chem,
                     "reacted": {c: np.asarray(reacted[c]).copy() for c in self.components},
                     "edges_solved": self._op.edges.copy(), "cells": int(self.edges.size - 1)}
        return self.last

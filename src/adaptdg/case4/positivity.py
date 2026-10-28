"""Conservative admissibility fixes for DG1 cell data (pure NumPy).

Two separate, logged operations:
1. ``local_mean_redistribution``: negative cell means are raised to zero and
   the deficit is taken from the nearest positive neighbours (growing radius),
   proportionally to their inventory. The weighted inventory sum(w*m) is
   conserved exactly (to round-off); no global shift is applied.
2. ``scale_slopes_nonnegative``: Zhang-Shu scaling of the two nodal values
   around a nonnegative mean so both endpoint values are >= 0. Means are not
   changed.

Provenance: copied from ASRTM-PHREEQC, methods/case4dev/positivity.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

import numpy as np


def local_mean_redistribution(means, weights, floor=0.0, max_radius=None):
    """Return (fixed_means, info). ``weights`` are phi*width per cell."""
    m = np.array(means, dtype=float)
    w = np.asarray(weights, dtype=float)
    if m.shape != w.shape or m.ndim != 1:
        raise ValueError("means and weights must be 1D arrays of equal length")
    if np.any(w <= 0) or not np.all(np.isfinite(m)):
        raise ValueError("weights must be positive and means finite")
    n = m.size
    max_radius = n if max_radius is None else int(max_radius)
    total_before = float(np.dot(w, m))
    negative = np.flatnonzero(m < floor)
    moved = 0.0
    max_radius_used = 0
    for i in negative:
        deficit = (floor - m[i]) * w[i]
        m[i] = floor
        moved += deficit
        radius = 1
        while deficit > 0.0:
            if radius > max_radius:
                raise RuntimeError("cannot restore admissibility: not enough positive inventory")
            lo, hi = max(0, i - radius), min(n, i + radius + 1)
            idx = np.arange(lo, hi)
            idx = idx[idx != i]
            avail = np.maximum(m[idx] - floor, 0.0) * w[idx]
            supply = float(avail.sum())
            if supply > 0.0:
                if supply <= deficit:
                    # Exhaust the neighbours exactly (no round-off below floor).
                    m[idx[avail > 0.0]] = floor
                    deficit -= supply
                else:
                    m[idx] -= (deficit / supply) * avail / w[idx]
                    deficit = 0.0
                max_radius_used = max(max_radius_used, radius)
            radius += 1
    total_after = float(np.dot(w, m))
    info = {
        "cells_fixed": int(negative.size),
        "moved_inventory": moved,
        "min_mean_before": float(np.min(means)),
        "max_radius": int(max_radius_used),
        "inventory_defect": total_after - total_before,
    }
    return m, info


def scale_slopes_nonnegative(left, right, floor=0.0):
    """Zhang-Shu scaling of nodal pairs around their (admissible) mean."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    mean = 0.5 * (a + b)
    low = np.minimum(a, b)
    need = low < floor
    # Means are admissible on entry; only round-off (|mean| ~ 1e-20) can
    # place 0.5*(a+b) below the floor after the mean redistribution shift.
    mean[need] = np.maximum(mean[need], floor)
    theta = np.ones_like(mean)
    denom = mean[need] - low[need]
    theta[need] = np.where(denom > 0, (mean[need] - floor) / denom, 0.0)
    theta = np.clip(theta, 0.0, 1.0)
    a_new = mean + theta * (a - mean)
    b_new = mean + theta * (b - mean)
    # Remove round-off below the floor in scaled cells: the low node is set to
    # the floor and the other node to 2*mean-floor, so the mean is unchanged.
    a_low = need & (a <= b)
    b_low = need & ~(a <= b)
    a_new[a_low], b_new[a_low] = floor, 2.0 * mean[a_low] - floor
    b_new[b_low], a_new[b_low] = floor, 2.0 * mean[b_low] - floor
    info = {"cells_scaled": int(need.sum()),
            "max_slope_change": float(np.max(np.abs(a_new - a))) if a.size else 0.0}
    return a_new, b_new, info

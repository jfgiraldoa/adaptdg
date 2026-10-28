"""Extracted from ASRTM-PHREEQC (methods/front2d/history_projection_fast.py); numerics unchanged.

Vectorised (NumPy) build of the exact DG1 -> DG1 projection plan between two 2D
triangulations. Same algorithm and same output as ``history_projection.build_plan``:
candidate pairs from a uniform bucket grid with a bounding-box test, Sutherland–Hodgman
clipping of every new triangle by every overlapping old triangle, fan triangulation of
the overlap polygon and the 3-point edge-midpoint rule (exact for the quadratic
integrand). The per-pair Python loops of the original are replaced by batched array
operations over all pairs; results agree with the original to round-off
(checked in tests/test_history_projection.py).
"""

from __future__ import annotations

import numpy as np

from .history_projection import ProjectionPlan, canonical_vertices

MAXV = 9          # a triangle clipped by 3 half-planes has at most 6 vertices; 9 is safe


def _pairs(old_lo, old_hi, new_lo, new_hi, cell, origin, tol):
    """Unique (new, old) pairs whose bounding boxes overlap, via a bucket sort-join."""
    def expand(lo, hi):
        i_lo = np.floor((lo - origin) / cell).astype(np.int64)
        i_hi = np.floor((hi - origin) / cell).astype(np.int64)
        nx, ny = i_hi[:, 0] - i_lo[:, 0] + 1, i_hi[:, 1] - i_lo[:, 1] + 1
        cnt = nx * ny
        owner = np.repeat(np.arange(lo.shape[0]), cnt)
        start = np.cumsum(cnt) - cnt
        local = np.arange(owner.size) - np.repeat(start, cnt)
        bx = i_lo[owner, 0] + local % nx[owner]
        by = i_lo[owner, 1] + local // nx[owner]
        return owner, bx, by
    o_own, o_bx, o_by = expand(old_lo, old_hi)
    n_own, n_bx, n_by = expand(new_lo, new_hi)
    shift = np.int64(1) << 31
    o_key = o_bx * shift + o_by
    n_key = n_bx * shift + n_by
    order = np.argsort(o_key, kind="stable")
    o_key, o_own = o_key[order], o_own[order]
    left = np.searchsorted(o_key, n_key, side="left")
    right = np.searchsorted(o_key, n_key, side="right")
    cnt = right - left
    pn = np.repeat(n_own, cnt)
    start = np.repeat(left, cnt)
    offs = np.arange(pn.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    po = o_own[start + offs]
    pair = np.unique(pn.astype(np.int64) * (old_lo.shape[0] + 1) + po)
    pn, po = pair // (old_lo.shape[0] + 1), pair % (old_lo.shape[0] + 1)
    keep = ~((old_lo[po, 0] > new_hi[pn, 0] + tol) | (old_hi[po, 0] < new_lo[pn, 0] - tol)
             | (old_lo[po, 1] > new_hi[pn, 1] + tol) | (old_hi[po, 1] < new_lo[pn, 1] - tol))
    return pn[keep], po[keep]


def _ccw(tri):
    """(m, 3, 2) -> counter-clockwise vertex order (as the original ``_ccw``)."""
    (x0, y0), (x1, y1), (x2, y2) = tri[:, 0].T, tri[:, 1].T, tri[:, 2].T
    flip = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0.0
    out = tri.copy()
    out[flip, 1], out[flip, 2] = tri[flip, 2], tri[flip, 1]
    return out


def _clip_batch(subject, clipper):
    """Batched Sutherland–Hodgman: subject (P, 3, 2) CCW clipped by clipper (P, 3, 2) CCW.
    Returns polygons (P, MAXV, 2) and vertex counts (P,), same predicates as ``_clip``."""
    P = subject.shape[0]
    poly = np.zeros((P, MAXV, 2))
    poly[:, :3] = subject
    n = np.full(P, 3)
    ar = np.arange(P)
    for e in range(3):
        a, b = clipper[:, e], clipper[:, (e + 1) % 3]
        ex, ey = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
        new = np.zeros((P, MAXV, 2))
        m = np.zeros(P, dtype=int)
        alive = n > 0
        prev = poly[ar, np.maximum(n - 1, 0)]
        prev_in = ex * (prev[:, 1] - a[:, 1]) - ey * (prev[:, 0] - a[:, 0]) >= -1e-15
        for j in range(MAXV):
            act = alive & (j < n)
            if not act.any():
                break
            cur = poly[:, j]
            cur_in = ex * (cur[:, 1] - a[:, 1]) - ey * (cur[:, 0] - a[:, 0]) >= -1e-15
            cross = act & (cur_in != prev_in)
            if cross.any():
                dx, dy = cur[:, 0] - prev[:, 0], cur[:, 1] - prev[:, 1]
                den = ex * dy - ey * dx
                safe = np.where(den == 0.0, 1.0, den)
                t = np.where(den == 0.0, 0.0, (ey * (prev[:, 0] - a[:, 0]) - ex * (prev[:, 1] - a[:, 1])) / safe)
                ix = np.column_stack((prev[:, 0] + t * dx, prev[:, 1] + t * dy))
                idx = ar[cross]
                new[idx, np.minimum(m[idx], MAXV - 1)] = ix[idx]
                m[idx] += 1
            put = act & cur_in
            idx = ar[put]
            new[idx, np.minimum(m[idx], MAXV - 1)] = cur[idx]
            m[idx] += 1
            prev = np.where(act[:, None], cur, prev)
            prev_in = np.where(act, cur_in, prev_in)
        if np.any(m > MAXV):
            raise RuntimeError("clipped polygon exceeds MAXV vertices")
        poly, n = new, np.where(alive, m, 0)
    return poly, n


def _affine_inverse(tri):
    """For barycentric coordinates: (m, 3, 2) -> inverse 2x2 maps and origin."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    M = np.stack([np.column_stack((b[:, 0] - a[:, 0], c[:, 0] - a[:, 0])),
                  np.column_stack((b[:, 1] - a[:, 1], c[:, 1] - a[:, 1]))], axis=1)
    return np.linalg.inv(M), a


def _bary(inv, a, pts):
    lam = np.einsum("pij,pj->pi", inv, pts - a)
    return np.column_stack((1.0 - lam.sum(axis=1), lam))


def build_plan(old_triangles, new_triangles, chunk=400_000):
    old_c = canonical_vertices(old_triangles)
    new_c = canonical_vertices(new_triangles)
    old_lo, old_hi = old_c.min(axis=1), old_c.max(axis=1)
    new_lo, new_hi = new_c.min(axis=1), new_c.max(axis=1)
    cell = float(np.median(np.max(old_hi - old_lo, axis=1)))
    origin = np.minimum(old_lo.min(axis=0), new_lo.min(axis=0))
    tol = 1e-12 * cell
    new_areas = 0.5 * np.abs(
        (new_c[:, 1, 0] - new_c[:, 0, 0]) * (new_c[:, 2, 1] - new_c[:, 0, 1])
        - (new_c[:, 2, 0] - new_c[:, 0, 0]) * (new_c[:, 1, 1] - new_c[:, 0, 1]))
    pn_all, po_all = _pairs(old_lo, old_hi, new_lo, new_hi, cell, origin, tol)
    inv_new, a_new = _affine_inverse(new_c)
    inv_old, a_old = _affine_inverse(old_c)
    new_ccw, old_ccw = _ccw(new_c), _ccw(old_c)
    keep_n, keep_o, keep_w = [], [], []
    covered = np.zeros(new_c.shape[0])
    for s in range(0, pn_all.size, chunk):
        pn, po = pn_all[s:s + chunk], po_all[s:s + chunk]
        poly, nv = _clip_batch(new_ccw[pn], old_ccw[po])
        P = pn.size
        w = np.zeros((P, 3, 3))
        area_sum = np.zeros(P)
        v0 = poly[:, 0]
        for f in range(1, MAXV - 1):              # fan triangles (v0, v_f, v_f+1)
            act = nv >= f + 2
            if not act.any():
                break
            A, B = poly[:, f], poly[:, f + 1]
            area = 0.5 * np.abs((A[:, 0] - v0[:, 0]) * (B[:, 1] - v0[:, 1]) - (B[:, 0] - v0[:, 0]) * (A[:, 1] - v0[:, 1]))
            act &= area > 0.0
            if not act.any():
                continue
            idx = np.flatnonzero(act)
            mids = (0.5 * (v0[idx] + A[idx]), 0.5 * (A[idx] + B[idx]), 0.5 * (B[idx] + v0[idx]))
            phi = np.stack([_bary(inv_new[pn[idx]], a_new[pn[idx]], q) for q in mids], axis=1)   # (k, 3 pts, 3)
            psi = np.stack([_bary(inv_old[po[idx]], a_old[po[idx]], q) for q in mids], axis=1)
            w[idx] += (area[idx] / 3.0)[:, None, None] * np.einsum("kqi,kqj->kij", phi, psi)
            area_sum[idx] += area[idx]
        ok = area_sum > 0.0
        keep_n.append(pn[ok])
        keep_o.append(po[ok])
        keep_w.append(w[ok])
        np.add.at(covered, pn[ok], area_sum[ok])
    pairs_new = np.concatenate(keep_n)
    pairs_old = np.concatenate(keep_o)
    weights = np.concatenate(keep_w)
    order = np.lexsort((pairs_old, pairs_new))
    coverage = float(np.max(np.abs(covered - new_areas) / new_areas))
    return ProjectionPlan(new_c, pairs_new[order], pairs_old[order], weights[order], new_areas, coverage)


# ------------------------------------------------------------------ identical-triangle shortcut
def _tri_areas(canonical):
    return 0.5 * np.abs(
        (canonical[:, 1, 0] - canonical[:, 0, 0]) * (canonical[:, 2, 1] - canonical[:, 0, 1])
        - (canonical[:, 2, 0] - canonical[:, 0, 0]) * (canonical[:, 1, 1] - canonical[:, 0, 1]))


def _triangle_hash(c, origin, q):
    """Order-independent integer hash of each triangle (vertices rounded to a grid of size q).
    Only a proposal: matches are verified on the true coordinates."""
    k = np.rint((c - origin) / q).astype(np.int64)
    v = k[..., 0] * np.int64(1000003) + k[..., 1] * np.int64(7919)      # (n, 3) vertex keys
    v = np.sort(v, axis=1)
    return (v[:, 0] * np.int64(0x9E3779B1) ^ v[:, 1] * np.int64(0x85EBCA77)
            ^ v[:, 2] * np.int64(0xC2B2AE3D))


def match_identical(old_c, new_c, tol_fraction=1e-13):
    """Pairs of geometrically identical triangles between two canonical-vertex arrays.

    Returns (new_idx, old_idx, perm) with ``perm[m, j]`` = old vertex slot holding the vertex
    that sits in new slot j (the per-cell vertex order of each mesh is the solver order, which
    differs between meshes when ties in x are decided by ~1e-20 dof-coordinate noise).
    Every returned pair has all three vertices equal within ``tol_fraction * span``."""
    span = float(max(np.ptp(old_c[..., 0]), np.ptp(old_c[..., 1]), 1e-300))
    origin = np.minimum(old_c.reshape(-1, 2).min(axis=0), new_c.reshape(-1, 2).min(axis=0))
    q = 1e-8 * span
    h_old, h_new = _triangle_hash(old_c, origin, q), _triangle_hash(new_c, origin, q)
    order = np.argsort(h_old, kind="stable")
    pos = np.clip(np.searchsorted(h_old[order], h_new), 0, h_old.size - 1)
    cand = order[pos]
    hit = h_old[cand] == h_new
    new_idx = np.flatnonzero(hit)
    old_idx = cand[new_idx]
    if new_idx.size == 0:
        return new_idx, old_idx, np.zeros((0, 3), dtype=np.int64)
    dist = np.linalg.norm(new_c[new_idx][:, :, None, :] - old_c[old_idx][:, None, :, :], axis=-1)  # (m, new j, old i)
    perm = np.argmin(dist, axis=2)
    ok = (np.take_along_axis(dist, perm[:, :, None], axis=2)[..., 0].max(axis=1) <= tol_fraction * span) \
        & (np.sort(perm, axis=1) == np.arange(3)).all(axis=1)
    return new_idx[ok], old_idx[ok], perm[ok]


def _apply_plan(plan, flat):
    """Same result as ``ProjectionPlan.apply`` for all fields at once (flat: (F, n_old, 3))."""
    n_new = plan.new_canonical.shape[0]
    contrib = np.einsum("pij,fpj->fpi", plan.weights, flat[:, plan.pairs_old])
    pn = plan.pairs_new
    if pn.size and np.all(np.diff(pn) >= 0) and np.unique(pn).size == n_new:
        starts = np.concatenate(([0], np.flatnonzero(np.diff(pn)) + 1))
        rhs = np.add.reduceat(contrib, starts, axis=1)
    else:
        rhs = np.zeros((flat.shape[0], n_new, 3))
        for k in range(flat.shape[0]):
            np.add.at(rhs[k], pn, contrib[k])
    inverse = np.array([[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]])
    return (rhs @ inverse.T) * (3.0 / plan.new_areas)[None, :, None]


def project_stack(old_triangles, new_triangles, stack, identity=True, coverage_tolerance=1e-9):
    """Exact DG1 -> DG1 L2 projection of ``stack`` (F, n_old, 3) onto the new mesh.

    Both meshes are conforming partitions of the same domain, so a new triangle that is
    geometrically identical to an old one overlaps no other old triangle and its exact
    projection is the identity (values copied, per-cell vertex order mapped). The exact plan
    is built only between the remaining triangles; if its coverage check fails the full plan
    is used (the unoptimised path). Returns (projected, new_areas, coverage_error, n_identical)."""
    old_c, new_c = canonical_vertices(old_triangles), canonical_vertices(new_triangles)
    values = np.asarray(stack, dtype=float)
    new_areas = _tri_areas(new_c)
    if identity:
        new_id, old_id, perm = match_identical(old_c, new_c)
        old_rest = np.ones(old_c.shape[0], dtype=bool)
        old_rest[old_id] = False
        new_rest = np.ones(new_c.shape[0], dtype=bool)
        new_rest[new_id] = False
        o_sub, n_sub = np.flatnonzero(old_rest), np.flatnonzero(new_rest)
        out = np.empty((values.shape[0], new_c.shape[0], 3))
        out[:, new_id] = np.take_along_axis(values[:, old_id], perm[None], axis=2)
        coverage = 0.0
        ok = True
        if n_sub.size:
            if o_sub.size:
                plan = build_plan(old_c[o_sub], new_c[n_sub])
                coverage = plan.coverage_error
                ok = coverage <= coverage_tolerance
                if ok:
                    out[:, n_sub] = _apply_plan(plan, values[:, o_sub])
            else:
                ok = False
        if ok:
            return out, new_areas, coverage, int(new_id.size)
    plan = build_plan(old_c, new_c)
    return _apply_plan(plan, values), new_areas, plan.coverage_error, 0

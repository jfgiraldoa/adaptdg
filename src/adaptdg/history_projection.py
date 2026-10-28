"""Exact L2 projection of DG1 fields between two triangulations (pure NumPy).

Extracted from ASRTM-PHREEQC (methods/front2d/history_projection.py); numerics unchanged.

Used by the history-preserving mesh rebuild (F2A, ``adaptdg.f2a``) to keep the complete
DG1 transport history when the 2D ADAP mesh is rebuilt or refined, instead of resetting
it to chemistry-grid cell averages. For every new triangle T and every old triangle K overlapping it,
the convex polygon T ∩ K is computed (Sutherland–Hodgman); the product of the
old linear field on K and the new P1 basis on T is quadratic, so a 3-point
edge-midpoint rule on a fan triangulation of the polygon integrates it
exactly. The local DG1 mass system on T is then solved.

Properties (checked in tests/test_history_projection.py):
* a field that is linear on the whole domain is reproduced exactly;
* if T lies inside one K (nested refinement) the result is the exact
  restriction;
* the integral over every new triangle equals the integral of the old field
  over it, hence global conservation;
* the overlap areas of each new triangle add up to its area (coverage).

Vertex ordering: DG1 "cell values" follow the repository convention of
``set_history_cell_values``: the three vertices of each triangle sorted by x,
then y (``np.lexsort((y, x))``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dof_triangles(domain):
    """Vertices of each cell as the DG1 dof coordinates (n, 3, 2), cell order of the mesh.

    The RM solver orders the three DG1 history values of a cell by lexsort of these dof
    coordinates (repository ``history_cell_values`` and fastadap ``_canon``). They equal the
    geometric vertices up to round-off (~1e-20), which decides ties on right triangles; using
    them here makes ``canonical_vertices`` reproduce the solver order exactly (with the
    geometric vertices of ``mesh_triangles`` the order differed in ~50 % of the cells).
    """
    from dolfinx import fem
    space = fem.functionspace(domain, ("DG", 1))
    xy = space.tabulate_dof_coordinates()[:, :2]
    return xy[np.asarray(space.dofmap.list)]


def canonical_vertices(triangles):
    """Vertices of each triangle in the history-value order (x, then y)."""
    tri = np.asarray(triangles, dtype=float)
    order = np.lexsort((tri[:, :, 1], tri[:, :, 0]), axis=-1)
    return np.take_along_axis(tri, order[:, :, None], axis=1)


def _clip(subject, clipper):
    """Sutherland–Hodgman: convex polygon ``subject`` clipped by convex ``clipper`` (CCW)."""
    output = subject
    n = len(clipper)
    for i in range(n):
        if not output:
            break
        a, b = clipper[i], clipper[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        inp, output = output, []
        prev = inp[-1]
        prev_in = ex * (prev[1] - a[1]) - ey * (prev[0] - a[0]) >= -1e-15
        for cur in inp:
            cur_in = ex * (cur[1] - a[1]) - ey * (cur[0] - a[0]) >= -1e-15
            if cur_in != prev_in:
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                den = ex * dy - ey * dx
                t = 0.0 if den == 0.0 else (ey * (prev[0] - a[0]) - ex * (prev[1] - a[1])) / den
                output.append((prev[0] + t * dx, prev[1] + t * dy))
            if cur_in:
                output.append(cur)
            prev, prev_in = cur, cur_in
    return output


def _ccw(t):
    (x0, y0), (x1, y1), (x2, y2) = t
    if (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0) < 0.0:
        return [tuple(t[0]), tuple(t[2]), tuple(t[1])]
    return [tuple(p) for p in t]


def _barycentric(tri, points):
    """Barycentric coordinates (m, 3) of points (m, 2) in triangle (3, 2)."""
    a, b, c = tri
    m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
    lam12 = np.linalg.solve(m, (points - a).T).T
    return np.column_stack([1.0 - lam12.sum(axis=1), lam12])


@dataclass
class ProjectionPlan:
    """Geometric weights W[t] (3 new x 3 old) per overlap, reusable for many fields."""

    new_canonical: np.ndarray
    pairs_new: np.ndarray
    pairs_old: np.ndarray
    weights: np.ndarray          # (pairs, 3, 3): int_P phi_new_i * psi_old_j
    new_areas: np.ndarray
    coverage_error: float        # max |sum overlap areas - area| / area

    def apply(self, old_values):
        """old_values (..., n_old, 3) canonical -> (..., n_new, 3) canonical."""
        values = np.asarray(old_values, dtype=float)
        lead = values.shape[:-2]
        flat = values.reshape((-1,) + values.shape[-2:])
        n_new = self.new_canonical.shape[0]
        # P1 mass matrix A/12 [[2,1,1],[1,2,1],[1,1,2]]; inverse (3/A)[[3,-1,-1],[-1,3,-1],[-1,-1,3]]
        inverse = np.array([[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]])
        out = np.empty((flat.shape[0], n_new, 3))
        for k in range(flat.shape[0]):
            contrib = np.einsum("pij,pj->pi", self.weights, flat[k][self.pairs_old])
            rhs = np.zeros((n_new, 3))
            np.add.at(rhs, self.pairs_new, contrib)
            out[k] = (rhs @ inverse.T) * (3.0 / self.new_areas)[:, None]
        return out.reshape(lead + (n_new, 3))


def build_plan(old_triangles, new_triangles):
    old_c = canonical_vertices(old_triangles)
    new_c = canonical_vertices(new_triangles)
    n_old, n_new = old_c.shape[0], new_c.shape[0]
    old_lo, old_hi = old_c.min(axis=1), old_c.max(axis=1)
    new_lo, new_hi = new_c.min(axis=1), new_c.max(axis=1)
    # uniform bucket grid over the old triangles' bounding boxes
    cell = float(np.median(np.max(old_hi - old_lo, axis=1)))
    origin = np.minimum(old_lo.min(axis=0), new_lo.min(axis=0))
    buckets: dict[tuple[int, int], list[int]] = {}
    i_lo = np.floor((old_lo - origin) / cell).astype(int)
    i_hi = np.floor((old_hi - origin) / cell).astype(int)
    for k in range(n_old):
        for ix in range(i_lo[k, 0], i_hi[k, 0] + 1):
            for iy in range(i_lo[k, 1], i_hi[k, 1] + 1):
                buckets.setdefault((ix, iy), []).append(k)
    pairs_new, pairs_old, weights = [], [], []
    covered = np.zeros(n_new)
    new_areas = 0.5 * np.abs(
        (new_c[:, 1, 0] - new_c[:, 0, 0]) * (new_c[:, 2, 1] - new_c[:, 0, 1])
        - (new_c[:, 2, 0] - new_c[:, 0, 0]) * (new_c[:, 1, 1] - new_c[:, 0, 1]))
    q_lo = np.floor((new_lo - origin) / cell).astype(int)
    q_hi = np.floor((new_hi - origin) / cell).astype(int)
    tol = 1e-12 * cell
    for t in range(n_new):
        cand = set()
        for ix in range(q_lo[t, 0], q_hi[t, 0] + 1):
            for iy in range(q_lo[t, 1], q_hi[t, 1] + 1):
                cand.update(buckets.get((ix, iy), ()))
        tri_t = new_c[t]
        poly_t = _ccw(tri_t)
        for k in cand:
            if (old_lo[k, 0] > new_hi[t, 0] + tol or old_hi[k, 0] < new_lo[t, 0] - tol
                    or old_lo[k, 1] > new_hi[t, 1] + tol or old_hi[k, 1] < new_lo[t, 1] - tol):
                continue
            poly = _clip(poly_t, _ccw(old_c[k]))
            if len(poly) < 3:
                continue
            p = np.asarray(poly)
            # fan triangulation, 3-point edge-midpoint rule (exact for quadratics)
            v0 = p[0]
            w = np.zeros((3, 3))
            area_sum = 0.0
            for a_, b_ in zip(p[1:-1], p[2:]):
                area = 0.5 * abs((a_[0] - v0[0]) * (b_[1] - v0[1]) - (b_[0] - v0[0]) * (a_[1] - v0[1]))
                if area <= 0.0:
                    continue
                mids = np.array([(v0 + a_) / 2, (a_ + b_) / 2, (b_ + v0) / 2])
                phi = _barycentric(tri_t, mids)
                psi = _barycentric(old_c[k], mids)
                w += (area / 3.0) * phi.T @ psi
                area_sum += area
            if area_sum > 0.0:
                pairs_new.append(t), pairs_old.append(k), weights.append(w)
                covered[t] += area_sum
    coverage = float(np.max(np.abs(covered - new_areas) / new_areas))
    return ProjectionPlan(new_c, np.asarray(pairs_new), np.asarray(pairs_old), np.asarray(weights),
                          new_areas, coverage)


def scale_nonnegative(values):
    """Zhang–Shu slope scaling per triangle (mean = vertex average is kept)."""
    v = np.array(values, dtype=float)
    mean = v.mean(axis=-1, keepdims=True)
    low = v.min(axis=-1, keepdims=True)
    need = (low < 0.0) & (mean >= 0.0)
    theta = np.where(need, mean / np.where(need, mean - low, 1.0), 1.0)
    scaled = mean + theta * (v - mean)
    return np.where(need, scaled, v), int(need.sum())

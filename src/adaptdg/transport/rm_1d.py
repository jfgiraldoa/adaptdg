"""NumPy residual-minimisation operator in 1D (paper method, with diffusion).

Extracted unchanged from ASRTM-PHREEQC (methods/onedim/rm1d.py);
the numerics, class and method names are those of the source.

Exact NumPy transcription of ``asrtm_phreeqc.stabilized_transport.ResidualMinimizationBackwardEuler1D``
(DG1 residual test/trial, CG1 concentration trial; SIP diffusion with penalty
``penalty/h``, upwind advection, the same Riesz norm and estimator), assembled once per
mesh and factorised once (SuperLU), so every component and every step on the same mesh
costs one back-substitution. Unlike the repository class it also accepts a DG1 history
(two values per cell), which the DG0 history is the special case of.

Unknowns: residual (L_k, R_k) = (2k, 2k+1), k < n; concentration nodes 2n + i, i <= n.
Flow left to right (velocity > 0); inlet at x = edges[0].
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class RMOperator1D:
    def __init__(self, edges, *, time_step, diffusion, velocity, penalty=8.0):
        self.edges = np.asarray(edges, dtype=float)
        if self.edges.ndim != 1 or self.edges.size < 2 or np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        if velocity <= 0:
            raise ValueError("Flow must go from left to right")
        self.n = n = self.edges.size - 1
        self.h = h = np.diff(self.edges)
        self.dt, self.D, self.beta = dt, D, b = float(time_step), float(diffusion), float(velocity)
        self.pen = pen = penalty / h
        rows, cols, vals = [], [], []

        def add(r, c, v):
            r, c = np.atleast_1d(r), np.atleast_1d(c)
            rows.append(r), cols.append(c), vals.append(np.broadcast_to(v, r.shape).astype(float))

        k = np.arange(n)
        L, R = 2 * k, 2 * k + 1
        c0, c1 = 2 * n + k, 2 * n + k + 1
        # ---- Riesz block (residual x residual)
        m_d, m_o = h / 3.0, h / 6.0
        kd = dt * D / h + dt * b * b               # dt D K + dt h beta^2 K  (K = [1,-1;-1,1]/h)
        add(L, L, (1 + dt) * m_d + kd), add(R, R, (1 + dt) * m_d + kd)
        add(L, R, (1 + dt) * m_o - kd), add(R, L, (1 + dt) * m_o - kd)
        j = np.arange(n - 1)
        wf = dt * 0.5 * D * 0.5 * (pen[j] + pen[j + 1]) + dt * 0.5 * abs(b)   # interior jumps
        add(R[j], R[j], wf), add(L[j + 1], L[j + 1], wf), add(R[j], L[j + 1], -wf), add(L[j + 1], R[j], -wf)
        add(L[0], L[0], dt * D * pen[0] + dt * 0.5 * abs(b))                # both boundaries (ds)
        add(R[-1], R[-1], dt * D * pen[-1] + dt * 0.5 * abs(b))
        # ---- transport block B (residual rows, concentration columns) and its transpose
        Br, Bc, Bv = [], [], []

        def addB(r, c, v):
            r, c = np.atleast_1d(r), np.atleast_1d(c)
            Br.append(r), Bc.append(c), Bv.append(np.broadcast_to(v, r.shape).astype(float))

        dk = dt * D / h
        a = 0.5 * dt * b
        addB(L, c0, m_d + dk - a), addB(L, c1, m_o - dk + a)
        addB(R, c0, m_o - dk - a), addB(R, c1, m_d + dk + a)
        # interior facets: -dt jump(test, n) avg(D c'),  jump(test,n) = R_k - L_{k+1}
        s = 0.5 * dt * D
        for (row, sign) in ((R[j], -1.0), (L[j + 1], 1.0)):
            addB(row, c1[j], sign * s / h[j]), addB(row, c0[j], -sign * s / h[j])           # slope of cell k
            addB(row, c1[j + 1], sign * s / h[j + 1]), addB(row, c0[j + 1], -sign * s / h[j + 1])  # cell k+1
        # inlet (x0, n = -1): +D test c' + D test' c + D pen test c + beta test c
        addB(L[0], c1[0], dt * D / h[0]), addB(L[0], c0[0], -dt * D / h[0])
        addB(L[0], c0[0], -dt * D / h[0]), addB(R[0], c0[0], dt * D / h[0])
        addB(L[0], c0[0], dt * D * pen[0] + dt * b)
        Br, Bc, Bv = np.concatenate(Br), np.concatenate(Bc), np.concatenate(Bv)
        rows += [Br, Bc]
        cols += [Bc, Br]
        vals += [Bv, Bv]
        size = 3 * n + 1
        self.A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(size, size)).tocsc()
        self.lu = splu(self.A)

    # ------------------------------------------------------------------
    def solve(self, history, inflow):
        """history: (n,) DG0 means or (n, 2) DG1 (left, right) values. Returns a dict."""
        n, h, dt, D, b = self.n, self.h, self.dt, self.D, self.beta
        p = np.asarray(history, dtype=float)
        if p.shape == (n,):
            pl = pr = p
        elif p.shape == (n, 2):
            pl, pr = p[:, 0], p[:, 1]
        else:
            raise ValueError("history must have shape (n,) or (n, 2)")
        rhs = np.zeros(3 * n + 1)
        rhs[0:2 * n:2] = h / 6.0 * (2 * pl + pr)
        rhs[1:2 * n:2] = h / 6.0 * (pl + 2 * pr)
        g = float(inflow)
        rhs[0] += dt * (D * self.pen[0] + b) * g - dt * D / h[0] * g
        rhs[1] += dt * D / h[0] * g
        x = self.lu.solve(rhs)
        rL, rR = x[0:2 * n:2], x[1:2 * n:2]
        c = x[2 * n:]
        # estimator (same expression as the repository's _indicator_expression)
        r2 = h / 3.0 * (rL ** 2 + rL * rR + rR ** 2)
        d2 = (rR - rL) ** 2 / h
        ind = (1 + dt) * r2 + dt * D * d2 + dt * b * b * h * d2
        jumps = (rR[:-1] - rL[1:]) ** 2
        wf = dt * 0.5 * D * 0.5 * (self.pen[:-1] + self.pen[1:]) + dt * 0.5 * abs(b)
        ind[:-1] += 0.5 * wf * jumps
        ind[1:] += 0.5 * wf * jumps
        ind[0] += (dt * D * self.pen[0] + dt * 0.5 * abs(b)) * rL[0] ** 2
        ind[-1] += (dt * D * self.pen[-1] + dt * 0.5 * abs(b)) * rR[-1] ** 2
        return {"nodal_values": c.copy(), "cell_averages": 0.5 * (c[:-1] + c[1:]),
                "history_dg1": np.column_stack((c[:-1], c[1:])),
                "cell_indicators": np.abs(ind), "residual_l2": float(np.sqrt(np.sum(r2))),
                "residual": np.column_stack((rL, rR))}

"""UFL building blocks of the dG discretisation (Giraldo & Calo 2023, CMAME 417, 116285).

Implements, for scalar problems of any polynomial degree p in d = 2, 3:
  * eq. (10): penalty ``eta_K = (p+1)(p+d)/d * A(dK)/V(K) * eta_0`` as a DG0 field;
  * eq. (8):  SWIP diffusion form (symmetric sign) with weights and harmonic mean ``gamma_kappa``
              for discontinuous, possibly tensor, ``kappa``;
  * eq. (9):  upwind advection form;
  * eq. (11): right-hand side with weak Nitsche Dirichlet data (all Dirichlet faces);
  * eqs. (12)-(14): Gram form of the discrete dual norm (L2 + jump/streamline upwind norm +
              SWIP norm), optionally weighted by a DG0 test function to get per-cell values.
"""

from __future__ import annotations

from typing import Any


def _side(value: Any, sign: str) -> Any:
    return value(sign) if hasattr(value, "ufl_shape") else value


def _flux(kappa: Any, g: Any) -> Any:
    import ufl

    return ufl.dot(kappa, g) if getattr(kappa, "ufl_shape", ()) else kappa * g


def _normal_kappa(kappa: Any, n: Any) -> Any:
    import ufl

    return ufl.dot(n, _flux(kappa, n))


def penalty_field(domain: Any, p: int, eta0: float = 1.0) -> Any:
    """DG0 function ``eta_K`` of eq. (10) (boundary faces use the same cell ratio)."""
    import ufl
    from dolfinx import fem

    d = domain.topology.dim
    v0 = fem.functionspace(domain, ("DG", 0))
    c0 = ufl.TestFunction(v0)
    area = fem.assemble_vector(fem.form(2 * ufl.avg(c0) * ufl.dS + c0 * ufl.ds)).array
    vol = fem.assemble_vector(fem.form(c0 * ufl.dx)).array
    eta = fem.Function(v0)
    eta.x.array[:] = eta0 * (p + 1) * (p + d) / d * area / vol
    return eta


class DGForms:
    """dG forms on one mesh for a given problem; ``degree`` is the quadrature degree.

    ``b(v, u)`` has the test function first, as in the paper, so the saddle-point system of
    eq. (18) reads ``gram(w, r) + b(w, u) + b(r, v)``. ``rhs_degree`` is the quadrature degree of
    the (possibly non-polynomial) data in :meth:`rhs`.
    """

    def __init__(self, domain: Any, problem: Any, p: int, *, eta0: float = 1.0,
                 degree: int | None = None, rhs_degree: int = 8) -> None:
        import ufl
        from adaptdg.vms.meshes import tag_boundary

        self.domain, self.problem, self.p = domain, problem, p
        self.n = ufl.FacetNormal(domain)
        self.x = ufl.SpatialCoordinate(domain)
        self.eta = penalty_field(domain, p, eta0)
        self.kappa = problem.kappa(self.x)
        self.beta = problem.beta(self.x)
        tags = tag_boundary(domain, problem.boundary_markers())
        self.dx, self.dS, self.ds = self._measures(2 * p + 2 if degree is None else degree, tags)
        self.dx_r, _, self.ds_r = self._measures(rhs_degree, tags)

    def _measures(self, degree: int, tags: Any) -> tuple[Any, Any, Any]:
        import ufl

        md = {"quadrature_degree": degree}
        dx = ufl.Measure("dx", domain=self.domain, metadata=md)
        dS = ufl.Measure("dS", domain=self.domain, metadata=md)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=tags, metadata=md)
        dsd = None
        for tag in self.problem.dirichlet_tags:
            dsd = ds(tag) if dsd is None else dsd + ds(tag)
        return dx, dS, dsd

    # -- diffusion weights (SWIP) -------------------------------------------------------
    def _interior_weights(self) -> tuple[Any, Any, Any]:
        n, k = self.n, self.kappa
        kp = _normal_kappa(_side(k, "+"), n("+"))
        km = _normal_kappa(_side(k, "-"), n("-"))
        return km / (kp + km), kp / (kp + km), 2 * kp * km / (kp + km)

    def _avg_flux(self, w: Any) -> Any:
        wp, wm, _ = self._interior_weights()
        k = self.kappa
        return wp * _flux(_side(k, "+"), w("+")) + wm * _flux(_side(k, "-"), w("-"))

    # -- bilinear form (eqs. 7-9) ---------------------------------------------------------
    def b(self, v: Any, u: Any) -> Any:
        """``b_h(v, u)``: SWIP diffusion plus upwind advection (v test, u trial)."""
        import ufl

        n, eta = self.n, self.eta
        gv, gu = ufl.grad(v), ufl.grad(u)
        _, _, gam = self._interior_weights()
        gam_b = _normal_kappa(self.kappa, n)
        jv, ju = ufl.jump(v, n), ufl.jump(u, n)
        form = (
            ufl.inner(gv, _flux(self.kappa, gu)) * self.dx
            - (ufl.inner(jv, self._avg_flux(gu)) + ufl.inner(self._avg_flux(gv), ju)) * self.dS
            + ufl.avg(eta) * gam * ufl.inner(jv, ju) * self.dS
            - (v * ufl.dot(_flux(self.kappa, gu), n) + ufl.dot(_flux(self.kappa, gv), n) * u) * self.ds
            + eta * gam_b * v * u * self.ds
        )
        if self.beta is not None:
            form += self._upwind(v, u)
        return form

    def _upwind(self, v: Any, u: Any) -> Any:
        import ufl

        n, beta = self.n, self.beta
        bn = ufl.dot(beta, n)
        bnp = ufl.dot(beta("+"), n("+"))
        return (
            v * ufl.dot(beta, ufl.grad(u)) * self.dx
            + ufl.min_value(bn, 0.0) * v * u * self.ds
            + (0.5 * abs(bnp) * ufl.jump(v) * ufl.jump(u) - ufl.avg(v) * bnp * ufl.jump(u)) * self.dS
        )

    # -- right-hand side (eq. 11) ---------------------------------------------------------
    def rhs(self, v: Any) -> Any:
        """``l_h(v)``: source plus weak Dirichlet data."""
        import ufl

        n, eta = self.n, self.eta
        uD = self.problem.dirichlet(self.x)
        f = self.problem.source(self.x)
        gam_b = _normal_kappa(self.kappa, n)
        form = (eta * gam_b * v * uD - ufl.dot(_flux(self.kappa, ufl.grad(v)), n) * uD) * self.ds_r
        if self.beta is not None:
            form -= ufl.min_value(ufl.dot(self.beta, n), 0.0) * v * uD * self.ds_r
        if f is not None:
            form += v * f * self.dx_r
        return form

    # -- Gram form of the discrete dual norm (eqs. 12-14) -----------------------------------
    def gram(self, a: Any, c: Any, weight: Any = None) -> Any:
        """``g(a, c)``; with a DG0 test function as ``weight`` it yields per-cell values."""
        import ufl

        n, eta = self.n, self.eta
        wc = 1.0 if weight is None else weight
        wf = 1.0 if weight is None else ufl.avg(weight)
        _, _, gam = self._interior_weights()
        gam_b = _normal_kappa(self.kappa, n)
        form = (
            ufl.inner(ufl.grad(a), _flux(self.kappa, ufl.grad(c))) * wc * self.dx
            + ufl.avg(eta) * gam * ufl.inner(ufl.jump(a, n), ufl.jump(c, n)) * wf * self.dS
            + eta * gam_b * a * c * wc * self.ds
            + a * c * wc * self.dx
        )
        if self.beta is not None:
            beta, bn = self.beta, ufl.dot(self.beta, n)
            bnp = ufl.dot(beta("+"), n("+"))
            h = ufl.CellDiameter(self.domain)
            form += (
                0.5 * abs(bnp) * ufl.jump(a) * ufl.jump(c) * wf * self.dS
                + 0.5 * abs(bn) * a * c * wc * self.ds
                + h * ufl.dot(beta, ufl.grad(a)) * ufl.dot(beta, ufl.grad(c)) * wc * self.dx
            )
        return form

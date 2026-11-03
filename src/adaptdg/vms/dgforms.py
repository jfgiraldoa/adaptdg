"""UFL building blocks of the dG discretisation (Giraldo & Calo 2023, CMAME 417, 116285).

Implements, for scalar problems of any polynomial degree p in d = 2, 3:
  * eq. (10): penalty ``eta_K = (p+1)(p+d)/d * A(dK)/V(K) * eta_0`` as a DG0 field;
  * eq. (8):  SWIP diffusion form (symmetric sign) with weights and harmonic mean ``gamma_kappa``
              for discontinuous, possibly tensor, ``kappa``;
  * eq. (9):  upwind advection form;
  * eq. (11): right-hand side with weak Nitsche Dirichlet data (all Dirichlet faces);
  * eqs. (12)-(14): Gram form of the discrete dual norm (L2 + jump/streamline upwind norm +
              SWIP norm), optionally weighted by a DG0 test function to get per-cell values.

Optional extensions, both off by default (the stationary forms are then unchanged): Neumann data
``kappa grad(u) . n = g_N`` on ``Problem.neumann_tags`` and one time step of a BDF scheme through
``time_coeffs = (c1, c2)`` (:mod:`adaptdg.vms.unsteady`): ``b -> c1 (v, u) + c2 b``, ``g -> c1 (a, c) + c2 g`` and
``l -> c2 l + history``, with the legacy factors ``c1 = 1`` and ``c2 = dt`` (BE) or ``2 dt/3`` (BDF2).
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
    the (possibly non-polynomial) data in :meth:`rhs`. ``preset`` (:class:`~adaptdg.vms.preset.Preset`,
    its name or ``None`` = paper) switches the Gram L2 term and the interior penalty factors.
    """

    def __init__(self, domain: Any, problem: Any, p: int, *, eta0: float = 1.0,
                 degree: int | None = None, rhs_degree: int = 8, preset: Any = None,
                 time_coeffs: tuple[Any, Any] | None = None) -> None:
        import ufl
        from adaptdg.vms.meshes import tag_boundary
        from adaptdg.vms.preset import resolve_preset

        self.domain, self.problem, self.p = domain, problem, p
        self.preset = resolve_preset(preset)
        self.n = ufl.FacetNormal(domain)
        self.x = ufl.SpatialCoordinate(domain)
        self.eta = penalty_field(domain, p, eta0)
        self.kappa = problem.kappa(self.x)
        self.beta = problem.beta(self.x)
        tags = tag_boundary(domain, problem.boundary_markers())
        self.dx, self.dS, self.ds = self._measures(2 * p + 2 if degree is None else degree, tags)
        self.dx_r, _, self.ds_r = self._measures(rhs_degree, tags)
        self.ds_out = self._outflow_measure(2 * p + 2 if degree is None else degree, tags)
        # Neumann faces: advective inflow data and flux datum (rhs), boundary terms of the Gram form
        self.ds_n = self._tagged_measure("neumann_tags", 2 * p + 2 if degree is None else degree, tags)
        self.ds_n_r = self._tagged_measure("neumann_tags", rhs_degree, tags)
        self.ds_g = self.ds if self.ds_n is None else (self.ds_n if self.ds is None else self.ds + self.ds_n)
        self.time_coeffs = time_coeffs
        #: ``[(coefficient, function), ...]``: the ``(coefficient * function, v)`` history terms of the right-hand side
        self.history: list[tuple[Any, Any]] = []

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

    def _tagged_measure(self, attribute: str, degree: int, tags: Any) -> Any:
        """Measure on the faces tagged ``getattr(problem, attribute, ())``, or ``None`` when there are none."""
        import ufl

        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=tags, metadata={"quadrature_degree": degree})
        out = None
        for tag in getattr(self.problem, attribute, ()):
            out = ds(tag) if out is None else out + ds(tag)
        return out

    def _outflow_measure(self, degree: int, tags: Any) -> Any:
        """Measure on the faces tagged ``Problem.outflow_tags`` (natural / outflow boundary), or ``None``."""
        return self._tagged_measure("outflow_tags", degree, tags)

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
        form = self.diffusion(v, u)
        if self.beta is not None:
            form += self._upwind(v, u)
        if self.time_coeffs is not None:
            c1, c2 = self.time_coeffs
            form = c1 * v * u * self.dx + c2 * form
        return form

    def diffusion(self, v: Any, u: Any) -> Any:
        """SWIP diffusion part of ``b_h`` (eq. 8) on the Dirichlet faces; other boundary faces are natural."""
        import ufl

        n, eta = self.n, self.eta
        gv, gu = ufl.grad(v), ufl.grad(u)
        _, _, gam = self._interior_weights()
        gam = gam * self.preset.gamma_factor_b
        gam_b = _normal_kappa(self.kappa, n)
        jv, ju = ufl.jump(v, n), ufl.jump(u, n)
        form = (
            ufl.inner(gv, _flux(self.kappa, gu)) * self.dx
            - (ufl.inner(jv, self._avg_flux(gu)) + ufl.inner(self._avg_flux(gv), ju)) * self.dS
            + ufl.avg(eta) * gam * ufl.inner(jv, ju) * self.dS
            - (v * ufl.dot(_flux(self.kappa, gu), n) + ufl.dot(_flux(self.kappa, gv), n) * u) * self.ds
            + eta * gam_b * v * u * self.ds
        )
        return form

    def _upwind(self, v: Any, u: Any) -> Any:
        import ufl

        n, beta = self.n, self.beta
        bn = ufl.dot(beta, n)
        bnp = ufl.dot(beta("+"), n("+"))
        form = (
            v * ufl.dot(beta, ufl.grad(u)) * self.dx
            - ufl.min_value(bn, 0.0) * v * u * self.ds  # inflow term bn (u_D - u) v; u_D is in rhs
            + (0.5 * abs(bnp) * ufl.jump(v) * ufl.jump(u) - ufl.avg(v) * bnp * ufl.jump(u)) * self.dS
        )
        if self.ds_n is not None:  # legacy: the advective inflow is also imposed on the Neumann faces
            form -= ufl.min_value(bn, 0.0) * v * u * self.ds_n
        return form

    # -- right-hand side (eq. 11) ---------------------------------------------------------
    def rhs(self, v: Any) -> Any:
        """``l_h(v)``: source plus weak Dirichlet data (and Neumann data, time-step history if configured)."""
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
        if self.ds_n_r is not None:  # Neumann faces: flux datum g_N and (legacy) the advective inflow datum
            form += v * self.problem.neumann(self.x, n) * self.ds_n_r
            if self.beta is not None:
                form -= ufl.min_value(ufl.dot(self.beta, n), 0.0) * v * uD * self.ds_n_r
        if self.time_coeffs is not None:  # one step of a BDF scheme: c2 l + sum_j a_j (u^{n-j}, v)
            form = self.time_coeffs[1] * form
            for coefficient, function in self.history:
                form += coefficient * function * v * self.dx
        return form

    # -- Gram form of the discrete dual norm (eqs. 12-14) -----------------------------------
    def gram(self, a: Any, c: Any, weight: Any = None) -> Any:
        """``g(a, c)``; with a DG0 test function as ``weight`` it yields per-cell values."""
        import ufl

        n, eta = self.n, self.eta
        wc = 1.0 if weight is None else weight
        wf = 1.0 if weight is None else ufl.avg(weight)
        _, _, gam = self._interior_weights()
        gam = gam * self.preset.gamma_factor_gram
        gam_b = _normal_kappa(self.kappa, n)
        form = (
            ufl.inner(ufl.grad(a), _flux(self.kappa, ufl.grad(c))) * wc * self.dx
            + ufl.avg(eta) * gam * ufl.inner(ufl.jump(a, n), ufl.jump(c, n)) * wf * self.dS
            + eta * gam_b * a * c * wc * self.ds_g
        )
        if self.preset.gram_l2 or self.beta is not None:  # legacy beta_inf = 1 gives the L2 term
            form += a * c * wc * self.dx
        if self.beta is not None:
            beta, bn = self.beta, ufl.dot(self.beta, n)
            bnp = ufl.dot(beta("+"), n("+"))
            h = ufl.CellDiameter(self.domain)
            form += (
                0.5 * abs(bnp) * ufl.jump(a) * ufl.jump(c) * wf * self.dS
                + 0.5 * abs(bn) * a * c * wc * self.ds_g
                + h * ufl.dot(beta, ufl.grad(a)) * ufl.dot(beta, ufl.grad(c)) * wc * self.dx
            )
        if self.time_coeffs is not None:  # mass term of the time step: c1 (a, c) + c2 g_steady
            c1, c2 = self.time_coeffs
            form = c1 * a * c * wc * self.dx + c2 * form
        return form

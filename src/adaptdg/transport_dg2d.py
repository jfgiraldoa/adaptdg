"""MoMaS Easy 2D, step D2: conservative DG1 transport with variable q and tensor D.

    d_t(phi c) + div(q c - D grad c) = s,   D = a_T |q| I + (a_L - a_T) q q^T / |q|

The 2D analogue of asrtm_phreeqc.dg_transport_1d.ConservativeDG1D (MoMaS 1D), with the
same choices so that the 1D-equivalence test (D3) is clean:
- DG1 on triangles, backward Euler, one matrix shared by all components;
- advection: conservative form, upwind flux with q.n on facets (single-valued for
  an RT0 q; the scheme also accepts any UFL vector field, e.g. an exact q for MMS);
- diffusion: SIP with the weighted average of Ern-Stephansen-Zunino for a
  discontinuous tensor: delta = n.D n on each side, weights delta_-/(delta_+ + delta_-),
  penalty with the harmonic mean of delta (reduces to the 1D harmonic SIP);
  D is evaluated pointwise from q (no cell averaging);
- inlets: weak Dirichlet (upwind q.n c_in plus SIP terms with c_in), as in 1D and as
  the MoMaS contract (Dirichlet T at the inlet);
- outflow wherever q.n > 0 on the boundary: q.n c, zero diffusive flux (natural);
  walls (q.n = 0): no flux;
- positivity (optional per component): least-change projection of cell means with the
  same inventory (nonnegative_means from the 1D code) and Zhang-Shu scaling of the
  vertex deviations so that every vertex value is >= 0 (in 1D this is the same
  limiter as |slope| <= mean).

The step reports inventory, the consistent boundary fluxes of the unlimited solve and
the balance defect, like the 1D operator.
"""

from __future__ import annotations

import numpy as np

EPS_Q = 1.0e-14  # regularises |q| at stagnation points
EPS_DELTA = 1.0e-30  # avoids 0/0 in the weights where n.D n = 0 on both sides


def dispersion_tensor(q, alpha_L, alpha_T):
    import ufl

    speed = ufl.sqrt(ufl.dot(q, q) + EPS_Q**2)
    return alpha_T * speed * ufl.Identity(2) + (alpha_L - alpha_T) * ufl.outer(q, q) / speed


class ConservativeDG2D:
    def __init__(self, domain, *, q, porosity, alpha_L, alpha_T, facet_tags, inlet_tags,
                 time_step, components=1, positive_components=(), penalty=20.0,
                 inflow=None, source=None):
        """q: UFL vector (RT0 Function or expression); porosity, alpha_L, alpha_T: DG0
        Functions or UFL scalars; inflow: UFL scalar for c_in on the inlets (default: a
        Constant set per component in step); source: UFL scalar s (default none)."""

        import ufl
        from dolfinx import fem
        from petsc4py import PETSc

        if domain.comm.size != 1:
            raise ValueError("Serial mesh required")
        if time_step <= 0 or penalty <= 0:
            raise ValueError("Positive time step and penalty required")
        self.domain = domain
        self.space = fem.functionspace(domain, ("DG", 1))
        tdim = domain.topology.dim
        self.cells = domain.topology.index_map(tdim).size_local
        self.cell_dofs = self.space.dofmap.list[: self.cells].copy()  # (cells, 3), vertex dofs
        self.state = np.zeros((components, self.space.dofmap.index_map.size_local))
        self.positive_components = tuple(positive_components)
        self.previous = fem.Function(self.space)
        self.solution = fem.Function(self.space)
        self.inflow_constant = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.dt = fem.Constant(domain, PETSc.ScalarType(time_step))
        self.time_step = float(time_step)
        c_in = self.inflow_constant if inflow is None else inflow

        V0 = fem.functionspace(domain, ("DG", 0))
        w0 = ufl.TestFunction(V0)
        self.areas = fem.assemble_vector(fem.form(w0 * ufl.dx)).array[: self.cells].copy()
        phi_h = fem.Function(V0)
        phi_h.interpolate(fem.Expression(porosity, V0.element.interpolation_points))
        self.cell_porosity = phi_h.x.array[V0.dofmap.list[: self.cells, 0]].copy()
        self.weights = self.cell_porosity * self.areas  # inventory = weights @ means

        u, v = ufl.TrialFunction(self.space), ufl.TestFunction(self.space)
        n = ufl.FacetNormal(domain)
        h = ufl.CellDiameter(domain)
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        ds_in = sum((ds(int(t)) for t in inlet_tags[1:]), ds(int(inlet_tags[0])))
        dx, dS = ufl.dx, ufl.dS
        D = dispersion_tensor(q, alpha_L, alpha_T)

        # Advection.
        qn = ufl.dot(q("+"), n("+"))
        qn_b = ufl.dot(q, n)
        adv = (-ufl.dot(q * u, ufl.grad(v)) * dx
               + (qn * ufl.avg(u) * ufl.jump(v) + 0.5 * abs(qn) * ufl.jump(u) * ufl.jump(v)) * dS
               + ufl.max_value(qn_b, 0.0) * u * v * ds)

        # Weighted SIP (normal n = n('+'); jump(w) = w+ - w-).
        dp = ufl.dot(n("+"), ufl.dot(D("+"), n("+"))) + EPS_DELTA
        dm = ufl.dot(n("+"), ufl.dot(D("-"), n("+"))) + EPS_DELTA
        wp, wm = dm / (dp + dm), dp / (dp + dm)
        gamma = 2.0 * dp * dm / (dp + dm)
        sigma = penalty * gamma * ufl.avg(1.0 / h)

        def wflux(f):
            return wp * ufl.dot(ufl.dot(D("+"), ufl.grad(f)("+")), n("+")) + \
                wm * ufl.dot(ufl.dot(D("-"), ufl.grad(f)("-")), n("+"))

        diff = (ufl.inner(ufl.dot(D, ufl.grad(u)), ufl.grad(v)) * dx
                + (-wflux(u) * ufl.jump(v) - wflux(v) * ufl.jump(u)
                   + sigma * ufl.jump(u) * ufl.jump(v)) * dS)
        Dnn = ufl.dot(n, ufl.dot(D, n))
        sigma_b = penalty * Dnn / h
        Dgrad_n = lambda f: ufl.dot(ufl.dot(D, ufl.grad(f)), n)  # noqa: E731
        diff += (-Dgrad_n(u) * v - Dgrad_n(v) * u + sigma_b * u * v) * ds_in

        phi = porosity
        lhs = phi * u * v * dx + self.dt * (adv + diff)
        rhs = phi * self.previous * v * dx + self.dt * (
            -qn_b * c_in * v - Dgrad_n(v) * c_in + sigma_b * c_in * v) * ds_in
        if source is not None:
            rhs += self.dt * source * v * dx
        self._a, self._L = fem.form(lhs), fem.form(rhs)

        # Consistent boundary fluxes of the (unlimited) solution, per unit time.
        c = self.solution
        self._in_flux = fem.form((-qn_b * c_in + Dgrad_n(c) + sigma_b * (c_in - c)) * ds_in)
        self._out_flux = fem.form(ufl.max_value(qn_b, 0.0) * c * ds)
        self._source_integral = None if source is None else fem.form(source * dx)
        self._build_solver()

    def _build_solver(self):
        from dolfinx.fem import petsc
        from petsc4py import PETSc

        A = petsc.assemble_matrix(self._a)
        A.assemble()
        self._A = A
        self._b = petsc.create_vector(self.space)
        ksp = PETSc.KSP().create(self.domain.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setErrorIfNotConverged(True)
        self._ksp = ksp

    def set_time_step(self, dt):
        if dt != self.time_step:
            self.dt.value = dt
            self.time_step = float(dt)
            self._build_solver()

    # ------------------------------------------------------------ cell means

    def cell_means(self):
        """(cells, components); for P1 on a triangle the mean is the vertex average."""
        return self.state[:, self.cell_dofs].mean(axis=2).T

    def initialize(self, means):
        means = np.asarray(means, dtype=float)
        self.state[:] = 0.0
        self.state[:, self.cell_dofs] = means.T[:, :, None]

    def set_dofs(self, component, values):
        self.state[component] = values

    def replace_cell_means(self, means):
        target = np.asarray(means, dtype=float).T  # (components, cells)
        local = self.state[:, self.cell_dofs]
        self.state[:, self.cell_dofs] = local - local.mean(axis=2, keepdims=True) + target[:, :, None]
        for comp in self.positive_components:
            self._limit_vertices(comp)

    def _limit_vertices(self, comp):
        local = self.state[comp, self.cell_dofs]
        mean = local.mean(axis=1)
        if np.any(mean < -1e-13 * max(1.0, float(np.max(np.abs(mean))))):
            raise ValueError("Cannot limit around a negative cell mean")
        mean = np.maximum(mean, 0.0)  # round-off of the recomputed mean only
        low = local.min(axis=1)
        theta = np.ones_like(mean)
        bad = low < 0
        theta[bad] = mean[bad] / (mean[bad] - low[bad])
        self.state[comp, self.cell_dofs] = mean[:, None] + theta[:, None] * (local - mean[:, None])

    # ------------------------------------------------------------ step

    def step(self, inlet=None):
        """One BE step for every component; inlet: per-component c_in (Constant inflow)."""

        from dolfinx import fem
        from dolfinx.fem import petsc
        from adaptdg.transport import nonnegative_means

        ncomp = self.state.shape[0]
        inlet = np.zeros(ncomp) if inlet is None else np.asarray(inlet, dtype=float)
        dt = self.time_step
        diagnostics = []
        for comp in range(ncomp):
            before = float(self.weights @ self.cell_means()[:, comp])
            self.previous.x.array[:] = self.state[comp]
            self.inflow_constant.value = float(inlet[comp])
            with self._b.localForm() as b:
                b.set(0.0)
            petsc.assemble_vector(self._b, self._L)
            self._ksp.solve(self._b, self.solution.x.petsc_vec)
            self.solution.x.scatter_forward()
            self.state[comp] = self.solution.x.array
            inward = dt * float(fem.assemble_scalar(self._in_flux))
            outward = dt * float(fem.assemble_scalar(self._out_flux))
            generated = 0.0 if self._source_integral is None else dt * float(
                fem.assemble_scalar(self._source_integral))
            net = inward - outward + generated
            correction = 0.0
            if comp in self.positive_components:
                raw = self.cell_means()[:, comp]
                repaired, correction = nonnegative_means(raw, self.weights)
                local = self.state[comp, self.cell_dofs]
                self.state[comp, self.cell_dofs] = local - local.mean(axis=1, keepdims=True) + repaired[:, None]
                self._limit_vertices(comp)
            after = float(self.weights @ self.cell_means()[:, comp])
            residual = after - before - net
            diagnostics.append({
                "inventory_before": before, "inventory_after": after, "net_boundary_input": net,
                "inward_flux_integral": inward, "outward_flux_integral": outward,
                "source_integral": generated, "absolute_defect": residual,
                "scaled_defect": abs(residual) / max(1.0, abs(before), abs(net)),
                "maximum_mean_positivity_correction": correction,
            })
        return self.cell_means(), diagnostics

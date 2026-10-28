"""MoMaS 2D ADAP transport: residual minimization (DG1 test / CG1 trial), backward Euler.

Generalisation of asrtm_phreeqc.stabilized_transport_2d.ResidualMinimizationBackwardEuler2D
(main's ADAP transport) to the MoMaS operator

    d_t(phi c) + q . grad c - div(D grad c) = 0,   div q = 0,
    D = a_T |q| I + (a_L - a_T) q q^T / |q|,

with q an RT0 Function (or any UFL vector), phi, a_L, a_T cellwise, and inlets given by
facet tags. Same structure as main:
- SIP with DG test functions, interior penalty 0.5 avg(eta) {n.D n} [v][u], eta =
  3 * perimeter / area (main's legacy opt_sta), weak Dirichlet (Nitsche) on the inlets;
- upwind terms written with q.n on facets; inflow term on the inlets only (walls have
  q.n = 0 exactly with the RT0 Darcy flux, so no inflow boundary is left unconstrained);
- the same Riesz norm and cell indicator, with the main's extra dt r^2 term, the scalar
  diffusivity replaced by the tensor D (and n.D n in the penalty terms) and beta by q;
- one assembled operator per mesh, reused for all components (LU reused).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptdg.transport_dg2d import dispersion_tensor


@dataclass
class RMStep:
    cell_averages: np.ndarray
    cell_indicators: np.ndarray
    minimum: float
    inflow: float  # integral of the consistent inlet flux (per unit time)
    outflow: float  # integral of q.n c over the outflow boundary (per unit time)


class RMTransportMoMaS2D:
    def __init__(self, domain, *, q, porosity, alpha_L, alpha_T, facet_tags, inlet_tags, time_step,
                 history_degree=0, norm_phi=True, swip=True):
        import basix.ufl
        import ufl
        from dolfinx import fem
        from petsc4py import PETSc

        if domain.comm.size != 1:
            raise RuntimeError("serial only")
        self.domain = domain
        tdim = domain.topology.dim
        self.cells = domain.topology.index_map(tdim).size_local
        x = domain.geometry.x[:, :2]
        geo = domain.geometry.dofmap[: self.cells]
        tri = x[geo]
        self.triangles = tri
        e = [np.linalg.norm(tri[:, (i + 1) % 3] - tri[:, i], axis=1) for i in range(3)]
        d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        self.areas = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])

        cell = domain.basix_cell()
        W = fem.functionspace(domain, basix.ufl.mixed_element(
            [basix.ufl.element("DG", cell, 1), basix.ufl.element("Lagrange", cell, 1)]))
        self._W = W
        V0 = fem.functionspace(domain, ("DG", 0))
        eta = fem.Function(V0)
        eta.x.array[V0.dofmap.list[: self.cells, 0]] = 3.0 * (e[0] + e[1] + e[2]) / self.areas
        self.history_degree = history_degree
        Vh = fem.functionspace(domain, ("DG", history_degree))
        self.previous = fem.Function(Vh)
        self._prev_dofs = V0.dofmap.list[: self.cells, 0]
        self.inflow = fem.Constant(domain, PETSc.ScalarType(0.0))
        dt = fem.Constant(domain, PETSc.ScalarType(time_step))

        dx, dS = ufl.dx, ufl.dS
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        ds_in = sum((ds(int(t)) for t in inlet_tags[1:]), ds(int(inlet_tags[0])))
        n = ufl.FacetNormal(domain)
        hK = ufl.CellDiameter(domain)
        D = dispersion_tensor(q, alpha_L, alpha_T)
        Dnn = ufl.dot(n, ufl.dot(D, n))
        Dnn_avg = 0.5 * (ufl.dot(n("+"), ufl.dot(D("+"), n("+"))) + ufl.dot(n("+"), ufl.dot(D("-"), n("+"))))
        qn = ufl.dot(q, n)
        qn_p = ufl.dot(q("+"), n("+"))
        phi = porosity

        r, c = ufl.TrialFunctions(W)
        w, v = ufl.TestFunctions(W)

        # SWIP (Ern-Stephansen-Zunino; Cier, Rojas, Calo 2021 eqs. 5, 9-11): averages weighted by the normal
        # diffusivities delta = n.D n of both sides, penalty with their harmonic mean. Reduces to SIP
        # (main's form: plain average, penalty D) when D is scalar and continuous.
        dpl = ufl.dot(n("+"), ufl.dot(D("+"), n("+"))) + 1e-30
        dmi = ufl.dot(n("+"), ufl.dot(D("-"), n("+"))) + 1e-30
        if swip:
            wpl, wmi = dmi / (dpl + dmi), dpl / (dpl + dmi)
            gamma = 2.0 * dpl * dmi / (dpl + dmi)
        else:
            wpl, wmi, gamma = 0.5, 0.5, Dnn_avg

        def wavg(f):
            return wpl * ufl.dot(D("+"), ufl.grad(f)("+")) + wmi * ufl.dot(D("-"), ufl.grad(f)("-"))

        self._gamma = gamma

        def sip(test, trial):
            return (ufl.inner(ufl.dot(D, ufl.grad(test)), ufl.grad(trial)) * dx
                    - ufl.inner(ufl.jump(test, n), wavg(trial)) * dS
                    - ufl.inner(wavg(test), ufl.jump(trial, n)) * dS
                    + 0.5 * gamma * ufl.avg(eta) * ufl.inner(ufl.jump(test, n), ufl.jump(trial, n)) * dS
                    - test * ufl.dot(ufl.dot(D, ufl.grad(trial)), n) * ds_in
                    - ufl.dot(ufl.dot(D, ufl.grad(test)), n) * trial * ds_in
                    + Dnn * eta * test * trial * ds_in)

        def upwind(test, trial):
            return (test * ufl.dot(q, ufl.grad(trial)) * dx
                    + 0.5 * (abs(qn) - qn) * test * trial * ds_in
                    + 0.5 * abs(qn_p) * ufl.jump(test) * ufl.jump(trial) * dS
                    - qn_p * ufl.avg(test) * ufl.jump(trial) * dS)

        def transport(test, trial):
            return phi * test * trial * dx + dt * (sip(test, trial) + upwind(test, trial))

        def riesz(a, b, wgt=None):
            k = (lambda f: f) if wgt is None else (lambda f: f * wgt)
            kS = (lambda f: f) if wgt is None else (lambda f: f * ufl.avg(wgt))
            mw = phi if norm_phi else 1.0  # L2 part of the test norm weighted by phi (as the storage term)
            return (k(mw * a * b) * dx + dt * (
                k(ufl.inner(ufl.dot(D, ufl.grad(a)), ufl.grad(b))) * dx
                + kS(0.5 * gamma * ufl.avg(eta) * ufl.inner(ufl.jump(a, n), ufl.jump(b, n))) * dS
                + k(Dnn * eta * a * b) * ds
                + kS(0.5 * abs(qn_p) * ufl.jump(a) * ufl.jump(b)) * dS
                + k(0.5 * abs(qn) * a * b) * ds
                + k(hK * ufl.dot(q, ufl.grad(a)) * ufl.dot(q, ufl.grad(b))) * dx
                + k(mw * a * b) * dx))

        self._transport, self._riesz = transport, riesz
        block = transport(w, c)
        a = riesz(r, w) + block + ufl.adjoint(block)
        L = phi * self.previous * w * dx + dt * (
            Dnn * eta * w * self.inflow - ufl.dot(ufl.dot(D, ufl.grad(w)), n) * self.inflow
            - w * qn * self.inflow) * ds_in
        self._a, self._L = fem.form(a), fem.form(L)
        self._test_w, self._dt, self._ds_in, self._eta, self._D, self._n = w, dt, ds_in, eta, D, n

        def _L_with(source, inflow_expr, history=False):  # test hook: MMS with a source and a variable inflow
            import ufl as _u
            Dn = _u.dot(n, _u.dot(D, n))
            hist = phi * self.previous * w * dx if history else 0
            return fem.form(hist + dt * source * w * dx + dt * (
                Dn * eta * w * inflow_expr - _u.dot(_u.dot(D, _u.grad(w)), n) * inflow_expr
                - w * _u.dot(q, n) * inflow_expr) * ds_in)
        self._L_with = _L_with
        self._sol = fem.Function(W)
        self._ind_test = ufl.TestFunction(V0)
        self._V0 = V0
        rh = self._sol.sub(0)
        ch = self._sol.sub(1)
        self._ind_form = fem.form(riesz(rh, rh, self._ind_test))
        self._in_form = fem.form((-qn * self.inflow + ufl.dot(ufl.dot(D, ufl.grad(ch)), n)
                                  + Dnn * eta * (self.inflow - ch)) * ds_in)
        self._out_form = fem.form(ufl.max_value(qn, 0.0) * ch * ds)
        Wc, cmap = W.sub(1).collapse()
        self._c_map = np.asarray(cmap[0] if isinstance(cmap, list) else cmap)
        self._c_cell_dofs = Wc.dofmap.list[: self.cells]
        self._c_fun = fem.Function(Wc)
        self._build()

    def _build(self):
        from dolfinx.fem import petsc
        from petsc4py import PETSc

        A = petsc.assemble_matrix(self._a)
        A.assemble()
        self._A = A
        self._b = petsc.create_vector(self._W)
        ksp = PETSc.KSP().create(self.domain.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setErrorIfNotConverged(True)
        self._ksp = ksp

    def step(self, previous_cell_averages, inflow_value) -> RMStep:
        from dolfinx import fem
        from dolfinx.fem import petsc

        if self.history_degree == 0:
            self.previous.x.array[self._prev_dofs] = previous_cell_averages
        elif previous_cell_averages.size == self.previous.x.array.size:  # DG1 dof values on this mesh
            self.previous.x.array[:] = previous_cell_averages
        else:  # DG1 history: CG1 nodal values on this mesh, interpolated into DG1
            self._c_fun.x.array[:] = previous_cell_averages
            self.previous.interpolate(self._c_fun)
        self.inflow.value = float(inflow_value)
        with self._b.localForm() as b:
            b.set(0.0)
        petsc.assemble_vector(self._b, self._L)
        self._ksp.solve(self._b, self._sol.x.petsc_vec)
        self._sol.x.scatter_forward()
        nodal = self._sol.x.array[self._c_map]
        avg = nodal[self._c_cell_dofs].mean(axis=1)
        ind = fem.assemble_vector(self._ind_form).array[self._V0.dofmap.list[: self.cells, 0]]
        self.last_nodal = nodal
        return RMStep(cell_averages=avg, cell_indicators=np.abs(ind), minimum=float(nodal.min()),
                      inflow=float(fem.assemble_scalar(self._in_form)),
                      outflow=float(fem.assemble_scalar(self._out_form)))

    def fine_scale(self):
        """VMS fine scale (Giraldo & Calo, arXiv:2305.12454): u~ in DG1 with
        b_h(v, u~) = g(v, r) for all v in DG1, r the residual representative of the last
        solve. Returns the full-scale DG1 function u = u_bar + u~ (as a DG1 Function)."""
        import ufl
        from dolfinx import fem
        from dolfinx.fem import petsc
        from petsc4py import PETSc

        if not hasattr(self, "_fs_ksp"):
            Vd = fem.functionspace(self.domain, ("DG", 1))
            u, v = ufl.TrialFunction(Vd), ufl.TestFunction(Vd)
            self._Vd = Vd
            self._r_fun = fem.Function(Vd)
            A = petsc.assemble_matrix(fem.form(self._transport(v, u)))
            A.assemble()
            ksp = PETSc.KSP().create(self.domain.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            self._fs_A, self._fs_ksp = A, ksp
            self._fs_L = fem.form(self._riesz(self._r_fun, v))
            self._fs_b = petsc.create_vector(Vd)
            self._u_tilde = fem.Function(Vd)
            self._u_full = fem.Function(Vd)
            self._c_dg = fem.Function(Vd)
        r_sub = self._sol.sub(0).collapse()
        self._r_fun.interpolate(r_sub)
        with self._fs_b.localForm() as b:
            b.set(0.0)
        petsc.assemble_vector(self._fs_b, self._fs_L)
        self._fs_ksp.solve(self._fs_b, self._u_tilde.x.petsc_vec)
        self._u_tilde.x.scatter_forward()
        self._c_dg.interpolate(self._sol.sub(1).collapse())
        self._u_full.x.array[:] = self._c_dg.x.array + self._u_tilde.x.array
        return self._u_full

    def destroy(self):
        """Release the PETSc objects (KSP/Mat/Vec) explicitly: thousands of meshes per run otherwise exhaust
        the MPI communicators (MUMPS INFOG(1) = -3 after ~1000 steps on the station)."""
        for name in ("_ksp", "_A", "_b", "_fs_ksp", "_fs_A", "_fs_b"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)


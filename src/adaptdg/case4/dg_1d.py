"""Conservative DG1 control for the evolving-porosity 1D benchmark.

Provenance: copied from ASRTM-PHREEQC, src/asrtm_phreeqc/porosity/dg_1d.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

import numpy as np

from .constants import LENGTH_M


class PorosityDG1D:
    """DG1 upwind transport for d_t(phi c)+d_x(q c)=0, with evolving phi."""

    def __init__(self, cells: int, time_step_s: float):
        from mpi4py import MPI
        from petsc4py import PETSc
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem

        self.cells, self.dt = cells, float(time_step_s)
        self.edges = np.linspace(0.0, LENGTH_M, cells + 1)
        self.widths = np.diff(self.edges)
        self.domain = mesh.create_interval(MPI.COMM_SELF, cells, np.asarray([0.0, LENGTH_M]))
        self.space = fem.functionspace(self.domain, ("DG", 1))
        self.phi_space = fem.functionspace(self.domain, ("DG", 0))
        self.phi = fem.Function(self.phi_space, name="porosity")
        self.u_previous = fem.Function(self.space, name="previous_bulk_inventory")
        self.inflow = fem.Function(self.space, name="inlet_concentration")
        self.cell_dofs = np.stack([self.space.dofmap.cell_dofs(i) for i in range(cells)])
        self.dof_coordinates = self.space.tabulate_dof_coordinates()[:, 0].copy()
        self.phi_dofs = np.array([self.phi_space.dofmap.cell_dofs(i)[0] for i in range(cells)])
        geom = self.domain.geometry
        centers = [np.mean(geom.x[geom.dofmaps[0][i], 0]) for i in range(cells)]
        self.cell_order = np.argsort(centers)
        self.dx = ufl.Measure("dx", domain=self.domain)
        self.ds = ufl.Measure("ds", domain=self.domain)
        self.dS = ufl.Measure("dS", domain=self.domain)
        n = ufl.FacetNormal(self.domain)
        q = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        self.q = q
        trial, test = ufl.TrialFunction(self.space), ufl.TestFunction(self.space)
        qn, qn_plus = q * n[0], q("+") * n("+")[0]
        incoming = 0.5 * (abs(qn) - qn)
        # Conservative weak-form DG upwinding. For a constant test function,
        # interior terms vanish and only the physical inlet/outlet fluxes remain.
        advection = -trial * q * test.dx(0) * self.dx
        advection += (qn_plus * ufl.avg(trial) * ufl.jump(test)
                      + 0.5 * abs(qn_plus) * ufl.jump(trial) * ufl.jump(test)) * self.dS
        advection += (qn + incoming) * trial * test * self.ds
        rhs = (self.u_previous / 1000.0 * test * self.dx
               + self.dt * incoming * self.inflow * test * self.ds)
        lhs = self.phi * trial * test * self.dx + self.dt * advection
        self._problem = LinearProblem(lhs, rhs, petsc_options_prefix="xieb2case4_",
            petsc_options={"ksp_type":"preonly", "pc_type":"lu",
                           "pc_factor_mat_solver_type":"mumps", "ksp_error_if_not_converged":True})

    def set_porosity(self, porosity: np.ndarray):
        self.phi.x.array[self.phi_dofs] = porosity[self.cell_order]
        self.phi.x.scatter_forward()

    def step(self, porosity: np.ndarray, concentration: np.ndarray, flux_m_s: float,
             inlet: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        self.set_porosity(porosity)
        self.q.value = np.asarray([flux_m_s], dtype=float)
        # solve every transported component with a common DG operator
        result = np.empty_like(concentration)
        balance = {}
        for component in range(concentration.shape[0]):
            bulk = 1000 * porosity[self.cell_order] * concentration[component, self.cell_order]
            inventory_before = float(np.dot(bulk, self.widths))
            self.u_previous.x.array[self.cell_dofs] = np.repeat(bulk[:, None], 2, axis=1)
            self.u_previous.x.scatter_forward()
            self.inflow.x.array[:] = inlet[component]
            self.inflow.x.scatter_forward()
            u = self._problem.solve()
            u.x.scatter_forward()
            means_ordered = np.mean(u.x.array[self.cell_dofs], axis=1)
            result[component, self.cell_order] = means_ordered
            inventory_after = float(np.dot(1000 * porosity * result[component], self.widths))
            outlet_cell = int(self.cell_order[-1])
            outlet_dof = self.cell_dofs[outlet_cell][
                int(np.argmax(self.dof_coordinates[self.cell_dofs[outlet_cell]]))
            ]
            outlet_concentration = float(u.x.array[outlet_dof])
            net_boundary_moles_per_m2 = self.dt * 1000 * flux_m_s * (
                inlet[component] - outlet_concentration
            )
            residual = inventory_after - inventory_before - net_boundary_moles_per_m2
            relative = abs(residual) / max(
                abs(inventory_before), abs(net_boundary_moles_per_m2), 1e-30
            )
            balance[str(component)] = {
                "relative":relative, "absolute_mol_m2":residual,
                "inventory_before_mol_m2":inventory_before,
                "net_boundary_moles_per_m2":net_boundary_moles_per_m2,
            }
        return result, balance



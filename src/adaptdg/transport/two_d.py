"""Two-dimensional residual-minimization transport on triangular FEniCSx meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilizedTransportStep2D:
    """CG1 transport solution and DG residual diagnostics on each triangle."""

    cell_averages: np.ndarray
    minimum: float
    maximum: float
    integral: float
    residual_l2: float
    cell_indicators: np.ndarray
    node_coordinates: np.ndarray
    nodal_values: np.ndarray
    history_dof_values: np.ndarray | None = None
    history_cell_values: np.ndarray | None = None


def create_rectangle_mesh_2d(
    *,
    lower: tuple[float, float],
    upper: tuple[float, float],
    cells: tuple[int, int],
    comm=None,
):
    """Create the coarse triangular transport mesh used by the 2D case."""

    if len(cells) != 2 or any(value <= 0 for value in cells):
        raise ValueError("2D mesh cells must contain two positive values")
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError("Every lower mesh bound must be smaller than its upper bound")

    from mpi4py import MPI
    from dolfinx import mesh

    communicator = MPI.COMM_SELF if comm is None else comm
    return mesh.create_rectangle(
        communicator,
        np.asarray([lower, upper], dtype=np.float64),
        cells,
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.right,
    )


def mesh_triangles(domain) -> np.ndarray:
    """Return physical triangle vertices in DOLFINx local-cell order."""

    topological_dimension = domain.topology.dim
    if topological_dimension != 2:
        raise ValueError("A two-dimensional mesh is required")
    local_cells = domain.topology.index_map(topological_dimension).size_local
    if domain.comm.size != 1:
        raise RuntimeError(
            "The provisional 2D runner currently requires serial execution"
        )
    coordinates = domain.geometry.x[:, :2]
    geometry_dofmap = domain.geometry.dofmap
    triangles = np.empty((local_cells, 3, 2), dtype=np.float64)
    for cell in range(local_cells):
        geometry_dofs = geometry_dofmap[cell]
        if len(geometry_dofs) != 3:
            raise RuntimeError("Only affine triangular transport meshes are supported")
        triangles[cell, :, :] = coordinates[geometry_dofs, :]
    return triangles


def create_mesh_from_triangles_2d(triangles: np.ndarray):
    """Reconstruct a serial affine DOLFINx mesh from saved triangle geometry."""

    from basix.ufl import element
    from dolfinx import mesh
    from mpi4py import MPI

    values = np.asarray(triangles, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 2):
        raise ValueError("Saved triangles must have shape (cells, 3, 2)")
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Saved triangles must be finite and non-empty")
    coordinates, inverse = np.unique(
        values.reshape((-1, 2)), axis=0, return_inverse=True
    )
    connectivity = inverse.reshape((-1, 3)).astype(np.int64, copy=False)
    coordinate_element = element(
        "Lagrange", "triangle", 1, shape=(2,), dtype=np.float64
    )
    return mesh.create_mesh(
        MPI.COMM_SELF,
        connectivity,
        coordinate_element,
        coordinates,
    )


def triangle_characteristic_sizes(triangles: np.ndarray) -> np.ndarray:
    """Return the longest edge of each triangle, as in legacy ``Cell.h()``."""

    values = np.asarray(triangles, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 2):
        raise ValueError("Triangles must have shape (number_of_cells, 3, 2)")
    edges = np.stack(
        [
            values[:, 1] - values[:, 0],
            values[:, 2] - values[:, 1],
            values[:, 0] - values[:, 2],
        ],
        axis=1,
    )
    return np.max(np.linalg.norm(edges, axis=2), axis=1)


def refine_marked_triangles(domain, marked: np.ndarray):
    """Refine every edge incident to a marked cell and return the new mesh."""

    refined, _ = refine_marked_triangles_with_parent(domain, marked)
    return refined


def refine_marked_triangles_with_parent(
    domain, marked: np.ndarray
) -> tuple[object, np.ndarray]:
    """Refine marked cells and return each new cell's parent-cell index."""

    from dolfinx import mesh

    topological_dimension = domain.topology.dim
    local_cells = domain.topology.index_map(topological_dimension).size_local
    markers = np.asarray(marked, dtype=bool)
    if markers.shape != (local_cells,):
        raise ValueError("There must be one refinement marker per local triangle")
    if not np.any(markers):
        return domain, np.arange(local_cells, dtype=np.int32)

    edge_dimension = 1
    domain.topology.create_connectivity(topological_dimension, edge_dimension)
    cell_to_edge = domain.topology.connectivity(topological_dimension, edge_dimension)
    if cell_to_edge is None:
        raise RuntimeError("DOLFINx did not construct cell-to-edge connectivity")
    selected_edges = np.unique(
        np.concatenate(
            [cell_to_edge.links(int(cell)) for cell in np.flatnonzero(markers)]
        )
    ).astype(np.int32)
    refined, parent_cell, _ = mesh.refine(
        domain,
        selected_edges,
        option=mesh.RefinementOption.parent_cell,
    )
    # DOLFINx returns ``parent_cell`` in the input ordering used to build the
    # refined mesh.  Mesh creation may reorder those cells; solution arrays and
    # ``geometry.dofmap`` use the final local-cell ordering.  Apply the recorded
    # permutation before using the map for parent-to-child field transfer.
    parent_cell = np.asarray(parent_cell, dtype=np.int32)
    input_cell_index = np.asarray(
        refined.topology.original_cell_index, dtype=np.int64
    )
    if input_cell_index.shape != parent_cell.shape:
        raise RuntimeError("Refined-cell permutation has an incompatible shape")
    if (
        np.any(input_cell_index < 0)
        or np.any(input_cell_index >= parent_cell.size)
    ):
        raise RuntimeError("Refined-cell permutation contains invalid indices")
    return refined, parent_cell[input_cell_index]


class ResidualMinimizationBackwardEuler2D:
    """DG1-test/CG1-trial residual minimization on one triangular mesh.

    The forms are the two-dimensional transcription of ``legacy/src/
    DA_transport.py``. The inlet geometry intentionally follows the legacy
    corner segments and is configurable so it can be audited when the original
    figure scripts are supplied.
    """

    def __init__(
        self,
        *,
        domain,
        lower: tuple[float, float],
        upper: tuple[float, float],
        time_step: float,
        diffusion: float,
        velocity: tuple[float, float],
        name: str,
        degree: int = 1,
        left_inlet_y_min: float = 0.4,
        top_inlet_x_max: float = 0.1,
        time_scheme: str = "backward_euler",
        history_degree: int = 0,
    ) -> None:
        if domain.topology.dim != 2:
            raise ValueError("The 2D stabilized solver requires a 2D mesh")
        if domain.comm.size != 1:
            raise RuntimeError(
                "The provisional 2D runner currently requires serial execution"
            )
        if time_step <= 0.0 or diffusion <= 0.0:
            raise ValueError("Time step and diffusion must be positive")
        if degree != 1:
            raise ValueError("The paper residual-minimization case requires degree one")
        if time_scheme not in {"backward_euler", "bdf2"}:
            raise ValueError("time_scheme must be backward_euler or bdf2")
        if history_degree not in {0, 1}:
            raise ValueError("history_degree must be zero or one")
        if not lower[1] <= left_inlet_y_min <= upper[1]:
            raise ValueError("The left inlet limit must lie on the left boundary")
        if not lower[0] <= top_inlet_x_max <= upper[0]:
            raise ValueError("The top inlet limit must lie on the top boundary")

        from basix.ufl import element, mixed_element
        from petsc4py import PETSc
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem

        self._domain = domain
        self._comm = domain.comm
        self.triangles = mesh_triangles(domain)
        self.cells = self.triangles.shape[0]
        self.cell_sizes = triangle_characteristic_sizes(self.triangles)
        side_a = np.linalg.norm(self.triangles[:, 1] - self.triangles[:, 0], axis=1)
        side_b = np.linalg.norm(self.triangles[:, 2] - self.triangles[:, 1], axis=1)
        side_c = np.linalg.norm(self.triangles[:, 0] - self.triangles[:, 2], axis=1)
        first = self.triangles[:, 1] - self.triangles[:, 0]
        second = self.triangles[:, 2] - self.triangles[:, 0]
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        self.cell_areas = 0.5 * np.abs(determinant)
        if np.any(self.cell_areas <= 0.0):
            raise ValueError("Degenerate transport triangles are not supported")

        cell = domain.basix_cell()
        dg1_element = element("DG", cell, degree)
        cg1_element = element("Lagrange", cell, degree)
        self._mixed_space = fem.functionspace(
            domain, mixed_element([dg1_element, cg1_element])
        )
        self._history_degree = history_degree
        self._previous_space = fem.functionspace(domain, ("DG", history_degree))
        self._previous = fem.Function(self._previous_space, name=f"{name}_previous")
        self._older = (
            fem.Function(self._previous_space, name=f"{name}_older")
            if time_scheme == "bdf2"
            else None
        )
        self.time_scheme = time_scheme

        facet_dimension = 1
        left_facets = mesh.locate_entities_boundary(
            domain,
            facet_dimension,
            lambda x: np.logical_and(
                np.isclose(x[0], lower[0]),
                x[1] >= left_inlet_y_min - 1.0e-12,
            ),
        )
        top_facets = mesh.locate_entities_boundary(
            domain,
            facet_dimension,
            lambda x: np.logical_and(
                np.isclose(x[1], upper[1]),
                x[0] <= top_inlet_x_max + 1.0e-12,
            ),
        )
        inlet_facets = np.unique(np.concatenate([left_facets, top_facets])).astype(
            np.int32
        )
        if inlet_facets.size == 0:
            raise RuntimeError("The configured 2D inlet contains no mesh facets")
        facet_tags = mesh.meshtags(
            domain,
            facet_dimension,
            inlet_facets,
            np.ones(inlet_facets.size, dtype=np.int32),
        )

        dx = ufl.Measure("dx", domain=domain)
        dS = ufl.Measure("dS", domain=domain)
        ds = ufl.Measure("ds", domain=domain)
        ds_tagged = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        ds_inlet = ds_tagged(1)
        normal = ufl.FacetNormal(domain)
        cell_diameter = ufl.CellDiameter(domain)

        diffusivity = fem.Constant(domain, PETSc.ScalarType(diffusion))
        beta = fem.Constant(domain, np.asarray(velocity, dtype=PETSc.ScalarType))
        dt = fem.Constant(domain, PETSc.ScalarType(time_step))
        self._inflow = fem.Constant(domain, PETSc.ScalarType(0.0))

        # legacy opt_sta = cell perimeter / cell area, multiplied by
        # (p+1)(p+d)/d = 3 for p=1 and d=2.
        eta_space = fem.functionspace(domain, ("DG", 0))
        eta = fem.Function(eta_space, name="sip_penalty_density")
        eta_values = 3.0 * (side_a + side_b + side_c) / self.cell_areas
        for cell_index, value in enumerate(eta_values):
            dof = eta_space.dofmap.cell_dofs(cell_index)[0]
            eta.x.array[dof] = value
        eta.x.scatter_forward()

        residual_trial, concentration_trial = ufl.TrialFunctions(self._mixed_space)
        residual_test, concentration_test = ufl.TestFunctions(self._mixed_space)
        beta_normal = ufl.dot(beta, normal)
        beta_normal_plus = ufl.dot(beta("+"), normal("+"))

        def sip(test, trial):
            return (
                ufl.inner(diffusivity * ufl.grad(test), ufl.grad(trial)) * dx
                - ufl.inner(
                    ufl.jump(test, normal), ufl.avg(diffusivity * ufl.grad(trial))
                )
                * dS
                - ufl.inner(
                    ufl.avg(diffusivity * ufl.grad(test)), ufl.jump(trial, normal)
                )
                * dS
                + 0.5
                * diffusivity
                * ufl.avg(eta)
                * ufl.inner(ufl.jump(test, normal), ufl.jump(trial, normal))
                * dS
                - test * ufl.dot(diffusivity * ufl.grad(trial), normal) * ds_inlet
                - ufl.dot(diffusivity * ufl.grad(test), normal) * trial * ds_inlet
                + diffusivity * eta * test * trial * ds_inlet
            )

        def upwind(test, trial):
            return (
                test * ufl.dot(beta, ufl.grad(trial)) * dx
                + 0.5 * (abs(beta_normal) - beta_normal) * test * trial * ds_inlet
                + 0.5 * abs(beta_normal_plus) * ufl.jump(test) * ufl.jump(trial) * dS
                - beta_normal_plus * ufl.avg(test) * ufl.jump(trial) * dS
            )

        mass_coefficient = 1.5 if time_scheme == "bdf2" else 1.0
        previous_coefficient = 2.0 if time_scheme == "bdf2" else 1.0
        older_coefficient = -0.5 if time_scheme == "bdf2" else 0.0

        def transport_form(test, trial):
            return mass_coefficient * test * trial * dx + dt * (
                sip(test, trial) + upwind(test, trial)
            )

        riesz = residual_trial * residual_test * dx + dt * (
            ufl.inner(diffusivity * ufl.grad(residual_trial), ufl.grad(residual_test))
            * dx
            + 0.5
            * diffusivity
            * ufl.avg(eta)
            * ufl.inner(
                ufl.jump(residual_trial, normal), ufl.jump(residual_test, normal)
            )
            * dS
            + diffusivity * eta * residual_trial * residual_test * ds
            + 0.5
            * abs(beta_normal_plus)
            * ufl.jump(residual_trial)
            * ufl.jump(residual_test)
            * dS
            + 0.5 * abs(beta_normal) * residual_trial * residual_test * ds
            + cell_diameter
            * ufl.dot(beta, ufl.grad(residual_trial))
            * ufl.dot(beta, ufl.grad(residual_test))
            * dx
            + residual_trial * residual_test * dx
        )
        transport_block = transport_form(residual_test, concentration_trial)
        bilinear = riesz + transport_block + ufl.adjoint(transport_block)
        inlet_rhs = (
            diffusivity * eta * residual_test * self._inflow
            - ufl.dot(diffusivity * ufl.grad(residual_test), normal) * self._inflow
            - residual_test * beta_normal * self._inflow
        ) * ds_inlet
        history = previous_coefficient * self._previous
        if self._older is not None:
            history += older_coefficient * self._older
        linear = history * residual_test * dx + dt * inlet_rhs

        safe_name = "".join(
            character if character.isalnum() else "_" for character in name
        )
        self._problem = LinearProblem(
            bilinear,
            linear,
            petsc_options_prefix=f"phreeqnics_{safe_name}_2d_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "ksp_error_if_not_converged": True,
            },
        )
        self._fem = fem
        self._ufl = ufl
        self._dx = dx
        self._dS = dS
        self._ds = ds
        self._normal = normal
        self._cell_diameter = cell_diameter
        self._diffusivity = diffusivity
        self._beta = beta
        self._dt = dt
        self._eta = eta
        self._PETSc = PETSc
        self._matrix_assembled = False

    @property
    def domain(self):
        return self._domain

    def set_cell_averages(self, values: np.ndarray) -> None:
        """Inject one previous concentration average per triangle."""

        self.set_history_cell_averages(values)

    def set_history_cell_averages(
        self, values: np.ndarray, older_values: np.ndarray | None = None
    ) -> None:
        """Inject cellwise constants into the one or two history functions."""

        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.cells,):
            raise ValueError(
                f"Expected {(self.cells,)} cell values, received {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Cell averages must be finite")
        for cell_index, value in enumerate(array):
            dofs = self._previous_space.dofmap.cell_dofs(cell_index)
            self._previous.x.array[dofs] = value
        self._previous.x.scatter_forward()
        if self._older is None:
            if older_values is not None:
                raise ValueError("Backward Euler accepts only one history state")
            return
        if older_values is None:
            raise ValueError("BDF2 requires the two preceding history states")
        older = np.asarray(older_values, dtype=np.float64)
        if older.shape != (self.cells,):
            raise ValueError(
                f"Expected {(self.cells,)} older values, received {older.shape}"
            )
        if not np.all(np.isfinite(older)):
            raise ValueError("Older cell averages must be finite")
        for cell_index, value in enumerate(older):
            dofs = self._previous_space.dofmap.cell_dofs(cell_index)
            self._older.x.array[dofs] = value
        self._older.x.scatter_forward()

    def set_history_dof_values(
        self, values: np.ndarray, older_values: np.ndarray | None = None
    ) -> None:
        """Inject complete DG history fields without discarding linear modes."""

        array = np.asarray(values, dtype=np.float64)
        expected = self._previous.x.array.shape
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError(f"Expected finite history values with shape {expected}")
        self._previous.x.array[:] = array
        self._previous.x.scatter_forward()
        if self._older is None:
            if older_values is not None:
                raise ValueError("Backward Euler accepts only one history state")
            return
        if older_values is None:
            raise ValueError("BDF2 requires the two preceding history states")
        older = np.asarray(older_values, dtype=np.float64)
        if older.shape != self._older.x.array.shape or not np.all(np.isfinite(older)):
            raise ValueError(
                f"Expected finite older history values with shape "
                f"{self._older.x.array.shape}"
            )
        self._older.x.array[:] = older
        self._older.x.scatter_forward()

    def set_history_cell_values(
        self, values: np.ndarray, older_values: np.ndarray | None = None
    ) -> None:
        """Inject DG histories stored in deterministic cell-local ordering."""

        # A degree-one triangle has three scalar basis functions.
        expected = (self.cells, 1 if self._history_degree == 0 else 3)

        def flatten(cell_values: np.ndarray) -> np.ndarray:
            array = np.asarray(cell_values, dtype=np.float64)
            if array.shape != expected or not np.all(np.isfinite(array)):
                raise ValueError(f"Expected finite cell history with shape {expected}")
            result = np.empty_like(self._previous.x.array)
            coordinates = self._previous_space.tabulate_dof_coordinates()[:, :2]
            for cell_index in range(self.cells):
                dofs = self._previous_space.dofmap.cell_dofs(cell_index)
                order = np.lexsort((coordinates[dofs, 1], coordinates[dofs, 0]))
                result[dofs[order]] = array[cell_index]
            return result

        current = flatten(values)
        older = None if older_values is None else flatten(older_values)
        self.set_history_dof_values(current, older)

    def history_from_cell_averages(self, values: np.ndarray) -> np.ndarray:
        """Return a DG history whose value is constant on every triangle."""

        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.cells,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Expected finite cell values with shape {(self.cells,)}")
        history = np.empty_like(self._previous.x.array)
        for cell_index, value in enumerate(array):
            history[self._previous_space.dofmap.cell_dofs(cell_index)] = value
        return history

    def cell_averages_from_history(self, values: np.ndarray) -> np.ndarray:
        """Recover exact affine-cell averages from a DG0/DG1 history."""

        array = np.asarray(values, dtype=np.float64)
        if array.shape != self._previous.x.array.shape or not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                f"Expected finite history values with shape "
                f"{self._previous.x.array.shape}"
            )
        averages = np.empty(self.cells, dtype=np.float64)
        for cell_index in range(self.cells):
            dofs = self._previous_space.dofmap.cell_dofs(cell_index)
            averages[cell_index] = float(np.mean(array[dofs]))
        return averages

    def add_cell_average_increments(
        self, values: np.ndarray, increments: np.ndarray
    ) -> np.ndarray:
        """Add one constant reaction increment per triangle to a DG history."""

        result = np.asarray(values, dtype=np.float64).copy()
        delta = np.asarray(increments, dtype=np.float64)
        if result.shape != self._previous.x.array.shape or not np.all(
            np.isfinite(result)
        ):
            raise ValueError("History values have an incompatible shape")
        if delta.shape != (self.cells,) or not np.all(np.isfinite(delta)):
            raise ValueError("Reaction increments have an incompatible shape")
        for cell_index, value in enumerate(delta):
            result[self._previous_space.dofmap.cell_dofs(cell_index)] += value
        return result

    def history_cell_values(self, values: np.ndarray) -> np.ndarray:
        """Return a DG history ordered canonically by physical DOF coordinates."""

        array = np.asarray(values, dtype=np.float64)
        if array.shape != self._previous.x.array.shape or not np.all(
            np.isfinite(array)
        ):
            raise ValueError("History values have an incompatible shape")
        width = 1 if self._history_degree == 0 else 3
        result = np.empty((self.cells, width), dtype=np.float64)
        coordinates = self._previous_space.tabulate_dof_coordinates()[:, :2]
        for cell_index in range(self.cells):
            dofs = self._previous_space.dofmap.cell_dofs(cell_index)
            order = np.lexsort((coordinates[dofs, 1], coordinates[dofs, 0]))
            result[cell_index] = array[dofs[order]]
        return result

    def step(self, *, inflow_value: float) -> StabilizedTransportStep2D:
        """Advance one stabilized transport step on the current mesh."""

        if not np.isfinite(inflow_value):
            raise ValueError("The inlet concentration must be finite")
        self._inflow.value = inflow_value
        if not self._matrix_assembled:
            mixed_solution = self._problem.solve()
            self._matrix_assembled = True
        else:
            # The operator is identical for every transported component on a
            # fixed mesh. Reassemble only the right-hand side and reuse the LU
            # factorization created by the first component solve.
            self._problem.b.set(0.0)
            self._fem.petsc.assemble_vector(
                self._problem.b, self._problem.L
            )
            self._problem.b.ghostUpdate(
                addv=self._PETSc.InsertMode.ADD,
                mode=self._PETSc.ScatterMode.REVERSE,
            )
            self._problem.solver.solve(
                self._problem.b, self._problem.x
            )
            self._problem.x.ghostUpdate(
                addv=self._PETSc.InsertMode.INSERT,
                mode=self._PETSc.ScatterMode.FORWARD,
            )
            self._fem.petsc.assign(
                self._problem.x, self._problem.u
            )
            mixed_solution = self._problem.u
        mixed_solution.x.scatter_forward()
        residual = mixed_solution.sub(0).collapse()
        concentration = mixed_solution.sub(1).collapse()
        residual.x.scatter_forward()
        concentration.x.scatter_forward()

        history_dof_values = None
        history_cell_values = None
        if self._history_degree == 1:
            history_function = self._fem.Function(self._previous_space)
            history_function.interpolate(concentration)
            history_function.x.scatter_forward()
            history_dof_values = history_function.x.array.copy()
            history_cell_values = self.history_cell_values(history_dof_values)

        cell_averages = np.empty(self.cells, dtype=np.float64)
        for cell_index in range(self.cells):
            dofs = concentration.function_space.dofmap.cell_dofs(cell_index)
            cell_averages[cell_index] = float(np.mean(concentration.x.array[dofs]))

        residual_squared = self._fem.assemble_scalar(
            self._fem.form(residual * residual * self._dx)
        )
        residual_l2 = float(np.sqrt(self._comm.allreduce(residual_squared)))

        indicator_space = self._fem.functionspace(self._domain, ("DG", 0))
        indicator_test = self._ufl.TestFunction(indicator_space)
        beta_normal = self._ufl.dot(self._beta, self._normal)
        beta_normal_plus = self._ufl.dot(self._beta("+"), self._normal("+"))
        indicator_form = residual * residual * indicator_test * self._dx + self._dt * (
            self._ufl.inner(
                self._diffusivity * self._ufl.grad(residual),
                self._ufl.grad(residual),
            )
            * indicator_test
            * self._dx
            + 0.5
            * self._diffusivity
            * self._ufl.avg(self._eta)
            * self._ufl.inner(
                self._ufl.jump(residual, self._normal),
                self._ufl.jump(residual, self._normal),
            )
            * self._ufl.avg(indicator_test)
            * self._dS
            + self._diffusivity
            * self._eta
            * residual
            * residual
            * indicator_test
            * self._ds
            + 0.5
            * abs(beta_normal_plus)
            * self._ufl.jump(residual)
            * self._ufl.jump(residual)
            * self._ufl.avg(indicator_test)
            * self._dS
            + 0.5 * abs(beta_normal) * residual * residual * indicator_test * self._ds
            + self._cell_diameter
            * self._ufl.dot(self._beta, self._ufl.grad(residual))
            * self._ufl.dot(self._beta, self._ufl.grad(residual))
            * indicator_test
            * self._dx
            + residual * residual * indicator_test * self._dx
        )
        indicator_vector = self._fem.petsc.assemble_vector(
            self._fem.form(indicator_form)
        )
        cell_indicators = np.empty(self.cells, dtype=np.float64)
        for cell_index in range(self.cells):
            dof = indicator_space.dofmap.cell_dofs(cell_index)[0]
            cell_indicators[cell_index] = abs(float(indicator_vector.array[dof]))

        return StabilizedTransportStep2D(
            cell_averages=cell_averages,
            minimum=float(np.min(concentration.x.array)),
            maximum=float(np.max(concentration.x.array)),
            integral=float(np.dot(cell_averages, self.cell_areas)),
            residual_l2=residual_l2,
            cell_indicators=cell_indicators,
            node_coordinates=concentration.function_space.tabulate_dof_coordinates()[
                :, :2
            ].copy(),
            nodal_values=concentration.x.array.copy(),
            history_dof_values=history_dof_values,
            history_cell_values=history_cell_values,
        )

"""DG1 / CG1 one-dimensional transport solvers and a step stub.

Extracted from ASRTM-PHREEQC without changing numerics or
public names. Two source modules live here, section by section; each keeps the
unit convention of its origin (NOT harmonised, see docs/DESIGN.md):

* Section A, ``src/asrtm_phreeqc/dg_transport_1d.py``: ``nonnegative_means``,
  ``ConservativeDG1D``. Dimensionless/consistent SI-like units chosen by the
  caller (``porosity`` multiplies storage, ``darcy_flux`` is a flux, ``dispersion``
  a diffusivity); no unit conversion inside.
* Section B, ``src/asrtm_phreeqc/porosity/transport_1d.py``: projection helpers,
  ``ResidualMinimizationCase4Transport1D``, ``solve_adaptive_transport_step``.
  Lengths in m, time in s, Darcy flux in m/s, concentrations in mol/L and bulk
  inventories in mol/m3 (factor 1000 inside); porosity per cell, scaled by a
  ``reference_porosity``.

Deviations from the source: imports point to ``adaptdg`` (``ReusableLinearProblem``
from ``adaptdg._linear``, ``conservative_remap_1d`` from ``adaptdg.transfer``,
``dorfler_mark_1d`` from ``adaptdg.marking``, the adapter proven equal to the
source function). dolfinx/ufl/basix/petsc4py/mpi4py were already imported lazily
inside functions in the source and stay so.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any

import numpy as np

from adaptdg._linear import ReusableLinearProblem
from adaptdg.marking import dorfler_mark_1d
from adaptdg.transfer import conservative_remap_1d


def transport_step(c_old: Any, flow: Any, dt: float, **options: Any) -> Any:
    """Advance a DG1 concentration field by one step with the given flow."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Section A: ASRTM-PHREEQC src/asrtm_phreeqc/dg_transport_1d.py
# ---------------------------------------------------------------------------

def nonnegative_means(values, weights):
    """Least-change positivity projection with unchanged weighted inventory.

Only apply to nonnegative physical totals, never a signed basis component.
The correction is diagnosed: it must vanish under benchmark refinement.
"""
    v,w = np.asarray(values,dtype=float),np.asarray(weights,dtype=float)
    if v.ndim!=1 or v.shape != w.shape or not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)) or np.any(w<=0):
        raise ValueError('Finite matching means and positive weights required')
    if np.min(v)>=0:
        return v.copy(),0.
    mass = float(w@v)
    if mass<0:
        raise ValueError('Negative total inventory cannot admit a positive projection')
    if mass == 0:
        return np.zeros_like(v),float(np.max(np.abs(v)))
    low,high = 0.,float(np.max(v))
    for _ in range(100):
        midpoint=(low+high)/2
        if w@np.maximum(v-midpoint,0)>mass:
            low=midpoint
        else:
            high=midpoint
    out=np.maximum(v-(low+high)/2,0)
    # Remove the final summation roundoff without changing the active set.
    positive=out>0
    out[positive] *= mass/float(w@out)
    return out,float(np.max(np.abs(out-v)))


class ConservativeDG1D:
    def __init__(self, domain, *, porosity, dispersion, darcy_flux,
                 time_step, components=4, positive_components=(0,2,3), penalty=20., source=None):
        from dolfinx import fem,mesh
        from dolfinx.fem.petsc import LinearProblem
        from petsc4py import PETSc
        import ufl
        if domain.topology.dim!=1 or domain.comm.size!=1:
            raise ValueError('This operator requires a serial interval mesh')
        if not np.all(np.isfinite([darcy_flux,time_step,penalty])) or darcy_flux<=0 or time_step<=0 or penalty<=0:
            raise ValueError('Require positive left-to-right flux, time step and SIP penalty')
        self.domain=domain
        self.cells=domain.topology.index_map(1).size_local
        self.space=fem.functionspace(domain,('DG',1))
        coeff_space=fem.functionspace(domain,('DG',0))
        self.cell_dofs=np.array([self.space.dofmap.cell_dofs(i) for i in range(self.cells)])
        coordinates=self.space.tabulate_dof_coordinates()[:,0]
        intervals=np.sort(coordinates[self.cell_dofs],axis=1)
        self.order=np.argsort(intervals.mean(axis=1))
        self.sorted_dofs=self.cell_dofs[self.order]
        self.centers=intervals.mean(axis=1)[self.order]
        self.widths=np.diff(intervals,axis=1)[:,0][self.order]
        self.porosity=np.asarray(porosity,dtype=float)
        dispersion=np.asarray(dispersion,dtype=float)
        if (self.porosity.shape!=(self.cells,) or dispersion.shape!=(self.cells,)
                or not np.all(np.isfinite(self.porosity)) or not np.all(np.isfinite(dispersion))
                or np.any(self.porosity<=0) or np.any(dispersion<=0)):
            raise ValueError('Require one finite positive phi/D per physical cell')
        phi,D,h=(fem.Function(coeff_space) for _ in range(3))
        coeff_dofs=np.array([coeff_space.dofmap.cell_dofs(i)[0] for i in self.order])
        for field,values in ((phi,self.porosity),(D,dispersion),(h,self.widths)):
            field.x.array[coeff_dofs]=values
            field.x.scatter_forward()
        self.coefficients=(phi,D,h)
        self.state=np.zeros((components,len(coordinates)))
        self.positive_components=tuple(positive_components)
        self.previous=fem.Function(self.space)
        self.inflow=fem.Constant(domain,PETSc.ScalarType(0.))
        self.dt=fem.Constant(domain,PETSc.ScalarType(time_step))
        self.time_step=float(time_step)
        q=fem.Constant(domain,PETSc.ScalarType(darcy_flux))
        left,right=float(intervals[:,0].min()),float(intervals[:,1].max())
        inlet=mesh.locate_entities_boundary(domain,0,lambda x:np.isclose(x[0],left))
        outlet=mesh.locate_entities_boundary(domain,0,lambda x:np.isclose(x[0],right))
        facets=np.concatenate((inlet,outlet)).astype(np.int32)
        tags_values=np.concatenate((np.ones(len(inlet)),2*np.ones(len(outlet)))).astype(np.int32)
        permutation=np.argsort(facets)
        tags=mesh.meshtags(domain,0,facets[permutation],tags_values[permutation])
        dx=ufl.Measure('dx',domain=domain)
        ds=ufl.Measure('ds',domain=domain,subdomain_data=tags)
        dS=ufl.Measure('dS',domain=domain)
        n=ufl.FacetNormal(domain)
        u,v=ufl.TrialFunction(self.space),ufl.TestFunction(self.space)
        qn=q('+')*n('+')[0]
        adv=-q*u*v.dx(0)*dx+(qn*ufl.avg(u)*ufl.jump(v)
             +.5*abs(qn)*ufl.jump(u)*ufl.jump(v))*dS+q*u*v*ds(2)
        harmonic=2*D('+')*D('-')/(D('+')+D('-'))
        weighted_u_flux=.5*harmonic*(u.dx(0)('+')+u.dx(0)('-'))
        weighted_v_flux=.5*harmonic*(v.dx(0)('+')+v.dx(0)('-'))
        sigma=penalty*harmonic*ufl.avg(1/h)
        jump_u=ufl.jump(u,n)[0]; jump_v=ufl.jump(v,n)[0]  # noqa: E702
        diffusion=D*u.dx(0)*v.dx(0)*dx+(-weighted_u_flux*jump_v
            -weighted_v_flux*jump_u+sigma*ufl.jump(u)*ufl.jump(v))*dS
        boundary_penalty=penalty*D/h
        diffusion+=(-D*u.dx(0)*n[0]*v-D*v.dx(0)*n[0]*u+boundary_penalty*u*v)*ds(1)
        rhs=phi*self.previous*v*dx+self.dt*(q*self.inflow*v
             -D*v.dx(0)*n[0]*self.inflow+boundary_penalty*self.inflow*v)*ds(1)
        if source is not None:
            rhs+=self.dt*source*v*dx
        lhs=phi*u*v*dx+self.dt*(adv+diffusion)
        base=LinearProblem(lhs,rhs,petsc_options_prefix='phreeqnics_dg1d_',
                           petsc_options={'ksp_type':'preonly','pc_type':'lu',
                                          'ksp_error_if_not_converged':True})
        self.problem=ReusableLinearProblem(base)
        self.matrix_assemblies=1
        # Flux of the unmodified transport solve; limiter preserves its global
        # inventory, but the post-limiter boundary trace is a different value.
        self._in_flux=fem.form((q*self.inflow-D*base.u.dx(0)
                         +boundary_penalty*(self.inflow-base.u))*ds(1))
        self._out_flux=fem.form(q*base.u*ds(2))
        self._source_integral=None if source is None else fem.form(source*dx)

    def initialize(self, mobile):
        means=np.asarray(mobile,dtype=float)
        if means.shape!=(self.cells,self.state.shape[0]) or not np.all(np.isfinite(means)):
            raise ValueError('Require (physical cells, components) means')
        self.state[:,self.sorted_dofs]=means.T[:,:,None]

    def cell_means(self):
        return self.state[:,self.sorted_dofs].mean(axis=2).T

    def _limit_slopes(self,component):
        local=self.state[component,self.sorted_dofs].copy()
        means=local.mean(axis=1)
        if np.any(means<0):
            raise ValueError('Cannot limit slopes around a negative cell mean')
        slope=.5*(local[:,1]-local[:,0])
        slope=np.sign(slope)*np.minimum(np.abs(slope),means)
        self.state[component,self.sorted_dofs]=np.column_stack((means-slope,means+slope))

    def replace_cell_means(self, mobile):
        target=np.asarray(mobile,dtype=float)
        if target.shape!=(self.cells,self.state.shape[0]) or not np.all(np.isfinite(target)):
            raise ValueError('Require finite (physical cells, components) means')
        local=self.state[:,self.sorted_dofs]
        slope=.5*(local[:,:,1]-local[:,:,0])
        self.state[:,self.sorted_dofs]=np.stack((target.T-slope,target.T+slope),axis=2)
        for component in self.positive_components:
            self._limit_slopes(component)

    def step(self,inlet,*,time_step=None):
        from dolfinx import fem
        dt=self.time_step if time_step is None else float(time_step)
        inlet=np.asarray(inlet,dtype=float)
        if inlet.shape!=(self.state.shape[0],) or not np.all(np.isfinite(inlet)):
            raise ValueError('Require one finite inlet value per component')
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('Positive time step required')
        if dt!=self.time_step:
            self.dt.value=dt
            self.time_step=dt
            self.problem=ReusableLinearProblem(self.problem.problem)
            self.matrix_assemblies+=1
        weights=self.porosity*self.widths
        diagnostics=[]
        for component,value in enumerate(inlet):
            before=float(weights@self.cell_means()[:,component])
            self.previous.x.array[:]=self.state[component]
            self.previous.x.scatter_forward()
            self.inflow.value=float(value)
            solution=self.problem.solve()
            solution.x.scatter_forward()
            self.state[component]=solution.x.array.copy()
            inward=dt*float(fem.assemble_scalar(self._in_flux))
            outward=dt*float(fem.assemble_scalar(self._out_flux))
            generated=0. if self._source_integral is None else dt*float(fem.assemble_scalar(self._source_integral))
            net=inward-outward+generated
            raw=self.cell_means()[:,component]
            correction=0.
            if component in self.positive_components:
                repaired,correction=nonnegative_means(raw,weights)
                local=self.state[component,self.sorted_dofs]
                slope=.5*(local[:,1]-local[:,0])
                self.state[component,self.sorted_dofs]=np.column_stack((repaired-slope,repaired+slope))
                self._limit_slopes(component)
            after=float(weights@self.cell_means()[:,component])
            residual=after-before-net
            diagnostics.append({'inventory_before':before,'inventory_after':after,
                                'net_boundary_input':net,'absolute_defect':residual,
                                'inward_flux_integral':inward,'outward_flux_integral':outward,
                                'source_integral':generated,
                                'scaled_defect':abs(residual)/max(1.,abs(before),abs(net)),
                                'maximum_mean_positivity_correction':correction})
        return self.cell_means(),diagnostics


# ---------------------------------------------------------------------------
# Section B: ASRTM-PHREEQC src/asrtm_phreeqc/porosity/transport_1d.py
# (``dorfler_mark_1d`` is imported from adaptdg.marking, see module docstring)
# ---------------------------------------------------------------------------

def _validate_edges(edges: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional edge array")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def project_piecewise_constant(
    source_edges: np.ndarray, source_values: np.ndarray, target_edges: np.ndarray
) -> np.ndarray:
    """Project cell averages by exact 1D overlap, preserving the integral."""

    source = _validate_edges(source_edges, "source_edges")
    target = _validate_edges(target_edges, "target_edges")
    values = np.asarray(source_values, dtype=float)
    if values.shape != (source.size - 1,) or not np.all(np.isfinite(values)):
        raise ValueError("source_values must have one finite value per source cell")
    if not np.isclose(source[[0, -1]], target[[0, -1]], rtol=0.0, atol=1e-12).all():
        raise ValueError("Source and target grids must cover the same interval")
    return conservative_remap_1d(source, values, target)


def project_linear_product_to_cells(
    source_nodes: np.ndarray,
    source_values: np.ndarray,
    coefficient_edges: np.ndarray,
    coefficient_values: np.ndarray,
    target_edges: np.ndarray,
) -> np.ndarray:
    """Average ``a(x)*u_h(x)`` on target cells exactly for CG1 x DG0 data."""

    nodes = _validate_edges(source_nodes, "source_nodes")
    coeff_edges = _validate_edges(coefficient_edges, "coefficient_edges")
    target = _validate_edges(target_edges, "target_edges")
    u = np.asarray(source_values, dtype=float)
    a = np.asarray(coefficient_values, dtype=float)
    if u.shape != nodes.shape or not np.all(np.isfinite(u)):
        raise ValueError("source_values must have one finite value per source node")
    if a.shape != (coeff_edges.size - 1,) or not np.all(np.isfinite(a)):
        raise ValueError("coefficient_values must have one finite value per coefficient cell")
    for edges, name in ((nodes, "source"), (coeff_edges, "coefficient")):
        if not np.isclose(edges[[0, -1]], target[[0, -1]], rtol=0.0, atol=1e-12).all():
            raise ValueError(f"{name} and target grids must cover the same interval")

    result = np.empty(target.size - 1, dtype=float)
    for cell, (left, right) in enumerate(zip(target[:-1], target[1:], strict=True)):
        internal_nodes = nodes[(nodes > left) & (nodes < right)]
        internal_coeff = coeff_edges[(coeff_edges > left) & (coeff_edges < right)]
        cuts = np.unique(np.concatenate(([left], internal_nodes, internal_coeff, [right])))
        integral = 0.0
        for a0, a1 in zip(cuts[:-1], cuts[1:], strict=True):
            midpoint = 0.5 * (a0 + a1)
            coeff_cell = int(np.searchsorted(coeff_edges, midpoint, side="right") - 1)
            u0, u1 = np.interp([a0, a1], nodes, u)
            integral += a[coeff_cell] * 0.5 * (u0 + u1) * (a1 - a0)
        result[cell] = integral / (right - left)
    return result


def gradient_jump_indicator_1d(cell_edges: np.ndarray, nodal_values: np.ndarray) -> np.ndarray:
    """Return a cell indicator for unresolved CG1 gradient changes/fronts."""

    edges = _validate_edges(cell_edges, "cell_edges")
    values = np.asarray(nodal_values, dtype=float)
    if values.shape != edges.shape or not np.all(np.isfinite(values)):
        raise ValueError("nodal_values must be finite and match the interval nodes")
    widths = np.diff(edges)
    slopes = np.diff(values) / widths
    indicator = np.zeros(widths.size, dtype=float)
    indicator[0] += abs(float(slopes[0])) * widths[0]
    indicator[-1] += abs(float(slopes[-1])) * widths[-1]
    for vertex in range(1, widths.size):
        jump = abs(float(slopes[vertex] - slopes[vertex - 1]))
        indicator[vertex - 1] += 0.5 * widths[vertex - 1] * jump
        indicator[vertex] += 0.5 * widths[vertex] * jump
    return indicator


def conservative_positive_projection_1d(
    nodes: np.ndarray,
    raw_values: np.ndarray,
    cell_edges: np.ndarray,
    porosity_cells: np.ndarray,
    previous_cell_values: np.ndarray,
    *,
    time_step_s: float,
    darcy_flux_m_s: float,
    inlet_concentration: float,
) -> tuple[np.ndarray, float]:
    """Project CG1 nodal values to nonnegative values with exact mass balance.

    The weighted least-change projection enforces the linear global inventory
    equation including the upwind outlet trace. It reports, and never hides,
    its maximum pointwise correction.
    """

    x = _validate_edges(nodes, "nodes")
    edges = _validate_edges(cell_edges, "cell_edges")
    values = np.asarray(raw_values, dtype=float)
    phi = np.asarray(porosity_cells, dtype=float)
    previous = np.asarray(previous_cell_values, dtype=float)
    if values.shape != x.shape or not np.all(np.isfinite(values)):
        raise ValueError("raw_values must be finite and defined at every CG1 node")
    if phi.shape != (edges.size - 1,) or previous.shape != phi.shape:
        raise ValueError("Porosity and history must match the transport cells")
    if np.any(phi <= 0.0) or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(previous)):
        raise ValueError("Porosity must be positive and history finite")
    if time_step_s <= 0.0 or not np.isfinite(darcy_flux_m_s):
        raise ValueError("Time step must be positive and Darcy flux finite")
    if not np.isfinite(inlet_concentration):
        raise ValueError("Inlet concentration must be finite")
    if not np.isclose([x[0], x[-1]], [edges[0], edges[-1]], rtol=0.0, atol=1e-12).all():
        raise ValueError("CG1 and coefficient meshes must cover the same interval")
    q = float(darcy_flux_m_s)
    if q < 0.0:
        raise ValueError("This benchmark projection currently assumes left-to-right flow")

    lumped_weights = np.zeros(x.size, dtype=float)
    for cell, width in enumerate(np.diff(edges)):
        left = int(np.argmin(np.abs(x - edges[cell])))
        right = int(np.argmin(np.abs(x - edges[cell + 1])))
        if not np.isclose(x[left], edges[cell], atol=1e-12, rtol=0.0) or not np.isclose(
            x[right], edges[cell + 1], atol=1e-12, rtol=0.0
        ):
            raise ValueError("Every CG1 transport cell must be represented by two mesh nodes")
        lumped_weights[left] += 0.5 * phi[cell] * width
        lumped_weights[right] += 0.5 * phi[cell] * width
    constraint_weights = lumped_weights.copy()
    constraint_weights[-1] += time_step_s * q
    target = float(np.dot(phi * np.diff(edges), previous))
    target += time_step_s * q * inlet_concentration
    if target < 0.0:
        raise ValueError("A nonnegative concentration cannot satisfy a negative total inventory")

    def inventory(lam: float) -> float:
        adjusted = np.maximum(0.0, values + lam * constraint_weights / lumped_weights)
        return float(np.dot(constraint_weights, adjusted))

    low, high = -1.0, 1.0
    for _ in range(100):
        if inventory(low) <= target:
            break
        low *= 2.0
    else:
        raise RuntimeError("Could not bracket lower positivity projection multiplier")
    for _ in range(100):
        if inventory(high) >= target:
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket upper positivity projection multiplier")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if inventory(middle) < target:
            low = middle
        else:
            high = middle
    projected = np.maximum(
        0.0, values + 0.5 * (low + high) * constraint_weights / lumped_weights
    )
    correction = float(np.max(np.abs(projected - values)))
    return projected, correction


@dataclass(frozen=True)
class PorosityTransportSettings:
    """Transport-only adaptive transport settings for the fixed-reactor Case 4 prototype."""

    initial_cells: int = 5
    maximum_solves: int = 6
    theta: float = 0.5
    eligibility_factor: float = 1.0
    epsilon_eligible: float = 0.2

    def __post_init__(self) -> None:
        if self.initial_cells < 2 or self.maximum_solves < 1:
            raise ValueError("adaptive transport needs at least two initial cells and one solve")
        if not 0.0 < self.theta <= 1.0 or self.eligibility_factor <= 0.0:
            raise ValueError("Invalid Dörfler marking settings")
        if not 0.0 <= self.epsilon_eligible <= 1.0:
            raise ValueError("epsilon_eligible must lie in [0, 1]")


@dataclass(frozen=True)
class AdaptiveTransportState1D:
    concentrations_mol_l: np.ndarray
    cell_edges_m: np.ndarray
    nodal_concentrations_mol_l: np.ndarray
    cell_indicators: np.ndarray
    gradient_jump_indicators: np.ndarray
    mass_balance_relative: float
    mass_balance_absolute_mol_m2: float
    minimum_concentration_mol_l: float
    maximum_projection_correction_mol_l: float


class ResidualMinimizationCase4Transport1D:
    """CG1 trial/DG1 residual minimization for 1D conservative advection.

    ``porosity_cells`` is projected from the fixed chemistry grid to this
    transport mesh. The physical benchmark has zero dispersion, so the
    Riesz map uses its L2, streamline, jump, and boundary contributions.
    """

    def __init__(
        self,
        *,
        domain,
        cell_edges_m: np.ndarray,
        porosity_cells: np.ndarray,
        time_step_s: float,
        darcy_flux_m_s: float,
        reference_porosity: float,
        length_m: float,
    ) -> None:
        from basix.ufl import element, mixed_element
        from petsc4py import PETSc
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem.petsc import LinearProblem

        self.domain = domain
        self.length_m = float(length_m)
        self.time_step_s = float(time_step_s)
        self.reference_porosity = float(reference_porosity)
        if self.time_step_s <= 0.0 or self.reference_porosity <= 0.0:
            raise ValueError("Time step and reference porosity must be positive")

        topology = domain.topology
        tdim = topology.dim
        if tdim != 1 or domain.comm.size != 1:
            raise ValueError("Case 4 adaptive transport currently requires serial 1D meshes")
        num_cells = topology.index_map(tdim).size_local
        topology.create_connectivity(tdim, 0)
        geometry_dofmap = domain.geometry.dofmap
        geometry = domain.geometry.x
        cell_intervals = np.empty((num_cells, 2), dtype=float)
        for cell in range(num_cells):
            dofs = geometry_dofmap[cell]
            coords = geometry[dofs, 0]
            cell_intervals[cell] = [float(np.min(coords)), float(np.max(coords))]
        centers = np.mean(cell_intervals, axis=1)
        self.cell_order = np.argsort(centers)
        self.cell_edges_m = np.concatenate(
            (cell_intervals[self.cell_order, 0], [cell_intervals[self.cell_order[-1], 1]])
        )
        supplied_edges = _validate_edges(cell_edges_m, "cell_edges_m")
        if (not np.all(np.diff(self.cell_edges_m) > 0.0)
                or supplied_edges.shape != self.cell_edges_m.shape
                or not np.allclose(supplied_edges, self.cell_edges_m, rtol=0.0, atol=1e-12)):
            raise ValueError("Adaptive transport cells must tile the interval without overlap")
        self.cell_widths = np.diff(self.cell_edges_m)
        phi = np.asarray(porosity_cells, dtype=float)
        if phi.shape != self.cell_widths.shape or np.any(phi <= 0.0) or not np.all(np.isfinite(phi)):
            raise ValueError("Porosity must be finite, positive, and defined on each transport cell")
        self.porosity_cells = phi.copy()

        interval = domain.basix_cell()
        dg1 = element("DG", interval, degree=1)
        cg1 = element("Lagrange", interval, degree=1)
        self.mixed_space = fem.functionspace(domain, mixed_element([dg1, cg1]))
        self.concentration_space = fem.functionspace(domain, ("Lagrange", 1))
        self.history_space = fem.functionspace(domain, ("DG", 0))
        self.porosity_space = fem.functionspace(domain, ("DG", 0))
        self.previous = fem.Function(self.history_space, name="case4_bulk_history_concentration")
        self.storage = fem.Function(self.porosity_space, name="case4_phi_over_phi0")
        self.cell_dofs = np.stack(
            [self.concentration_space.dofmap.cell_dofs(int(i)) for i in self.cell_order]
        )
        coordinates = self.concentration_space.tabulate_dof_coordinates()[:, 0]
        self.cell_dofs = np.asarray([
            dofs[np.argsort(coordinates[dofs])] for dofs in self.cell_dofs
        ], dtype=np.int32)
        self.node_coordinates = np.unique(np.asarray(coordinates, dtype=float))

        storage_values = phi / self.reference_porosity
        storage_dofs = np.asarray([
            self.porosity_space.dofmap.cell_dofs(int(i))[0] for i in self.cell_order
        ], dtype=np.int32)
        self.storage.x.array[storage_dofs] = storage_values
        self.storage.x.scatter_forward()

        dx = ufl.Measure("dx", domain=domain)
        dS = ufl.Measure("dS", domain=domain)
        ds = ufl.Measure("ds", domain=domain)
        normal = ufl.FacetNormal(domain)
        h = ufl.CellDiameter(domain)
        beta_value = float(darcy_flux_m_s) / self.reference_porosity
        beta = fem.Constant(domain, PETSc.ScalarType(beta_value))
        beta_normal = beta * normal[0]
        beta_normal_plus = beta("+") * normal("+")[0]
        incoming = 0.5 * (abs(beta_normal) - beta_normal)
        self.inflow = fem.Constant(domain, PETSc.ScalarType(0.0))
        boundary_vertices = mesh.locate_entities_boundary(
            domain, 0, lambda x: np.isclose(x[0], 0.0, rtol=0.0, atol=1e-12)
        )
        if boundary_vertices.size != 1:
            raise RuntimeError(f"Expected a single Case 4 inlet vertex, got {boundary_vertices}")
        inlet_tags = mesh.meshtags(
            domain, 0, boundary_vertices.astype(np.int32),
            np.ones(boundary_vertices.size, dtype=np.int32),
        )
        ds_inlet = ufl.Measure("ds", domain=domain, subdomain_data=inlet_tags)(1)

        residual_trial, concentration_trial = ufl.TrialFunctions(self.mixed_space)
        residual_test, concentration_test = ufl.TestFunctions(self.mixed_space)
        riesz = residual_trial * residual_test * dx
        riesz += self.time_step_s * (
            h**2 * beta**2 * residual_trial.dx(0) * residual_test.dx(0) * dx
            + 0.5 * abs(beta_normal_plus)
            * ufl.jump(residual_trial) * ufl.jump(residual_test) * dS
            + 0.5 * abs(beta_normal)
            * residual_trial * residual_test * ds
        )
        transport_block = self.storage * residual_test * concentration_trial * dx
        transport_block += self.time_step_s * (
            residual_test * beta * concentration_trial.dx(0) * dx
            + incoming * residual_test * concentration_trial * ds
        )
        bilinear = riesz + transport_block + ufl.adjoint(transport_block)
        linear = self.storage * self.previous * residual_test * dx
        linear += self.time_step_s * incoming * residual_test * self.inflow * ds_inlet
        self.problem = LinearProblem(
            bilinear,
            linear,
            petsc_options_prefix="phreeqnics_porosity_1d_",
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
        self._beta = beta
        self._scalar_type = PETSc.ScalarType
        self._normal = normal
        self._linear_solver = ReusableLinearProblem(self.problem)

    def step(
        self,
        previous_concentration: np.ndarray,
        inlet_concentration: float,
        *,
        enforce_nonnegative: bool = True,
    ) -> AdaptiveTransportState1D:
        previous = np.asarray(previous_concentration, dtype=float)
        if previous.shape != self.cell_widths.shape or not np.all(np.isfinite(previous)):
            raise ValueError("Previous concentration must be finite and defined on transport cells")
        if not np.isfinite(inlet_concentration):
            raise ValueError("Inlet concentration must be finite")
        history_dofs = np.asarray([
            self.history_space.dofmap.cell_dofs(int(i))[0] for i in self.cell_order
        ], dtype=np.int32)
        self.previous.x.array[history_dofs] = previous
        self.previous.x.scatter_forward()
        self.inflow.value = self._scalar_type(inlet_concentration)

        mixed_solution = self._linear_solver.solve()
        mixed_solution.x.scatter_forward()
        residual = mixed_solution.sub(0).collapse()
        concentration = mixed_solution.sub(1).collapse()
        residual.x.scatter_forward()
        concentration.x.scatter_forward()
        dof_coordinates = concentration.function_space.tabulate_dof_coordinates()[:, 0]
        order = np.argsort(dof_coordinates)
        nodes = np.asarray(dof_coordinates[order], dtype=float)
        values = np.asarray(concentration.x.array[order], dtype=float)
        if nodes.shape != self.node_coordinates.shape or not np.allclose(nodes, self.node_coordinates):
            raise RuntimeError("CG1 node map changed unexpectedly")

        indicator_space = self._fem.functionspace(self.domain, ("DG", 0))
        test = self._ufl.TestFunction(indicator_space)
        indicator_form = residual * residual * test * self._dx
        vector = self._fem.petsc.assemble_vector(self._fem.form(indicator_form))
        cell_indicators_natural = np.asarray(vector.array, dtype=float)
        indicators = cell_indicators_natural[self.cell_order].copy()
        vector.destroy()
        # CG1 averages are exactly the mean of the endpoint values. Evaluate
        # from the collapsed solution's own coordinates to avoid relying on a
        # mixed-space collapse permutation when mapping back to PHREEQC cells.
        cell_values = 0.5 * (
            np.interp(self.cell_edges_m[:-1], nodes, values)
            + np.interp(self.cell_edges_m[1:], nodes, values)
        )
        outlet = float(np.interp(self.length_m, nodes, values))
        before = float(np.dot(1000.0 * self.porosity_cells * self.cell_widths, previous))
        after = float(np.dot(1000.0 * self.porosity_cells * self.cell_widths, cell_values))
        q = float(self._beta.value) * self.reference_porosity
        boundary_moles = self.time_step_s * 1000.0 * q * (inlet_concentration - outlet)
        if enforce_nonnegative:
            values, correction = conservative_positive_projection_1d(
                nodes,
                values,
                self.cell_edges_m,
                self.porosity_cells,
                previous,
                time_step_s=self.time_step_s,
                darcy_flux_m_s=q,
                inlet_concentration=inlet_concentration,
            )
        else:
            # Charge is a signed aggregate, so it uses only the global
            # conservative constant-mode reconciliation.
            denominator = 1000.0 * (
                float(np.dot(self.porosity_cells, self.cell_widths))
                + self.time_step_s * q
            )
            correction = (before + boundary_moles - after) / denominator
            values = values + correction
        cell_values = 0.5 * (
            np.interp(self.cell_edges_m[:-1], nodes, values)
            + np.interp(self.cell_edges_m[1:], nodes, values)
        )
        outlet = float(np.interp(self.length_m, nodes, values))
        after = float(np.dot(1000.0 * self.porosity_cells * self.cell_widths, cell_values))
        boundary_moles = self.time_step_s * 1000.0 * q * (inlet_concentration - outlet)
        balance = after - before - boundary_moles
        relative = abs(balance) / max(abs(before), abs(boundary_moles), 1e-30)
        return AdaptiveTransportState1D(
            concentrations_mol_l=cell_values,
            cell_edges_m=self.cell_edges_m.copy(),
            nodal_concentrations_mol_l=values.copy(),
            cell_indicators=indicators,
            gradient_jump_indicators=gradient_jump_indicator_1d(self.cell_edges_m, values),
            mass_balance_relative=float(relative),
            mass_balance_absolute_mol_m2=float(balance),
            minimum_concentration_mol_l=float(np.min(values)),
            maximum_projection_correction_mol_l=float(correction),
        )


@dataclass(frozen=True)
class Case4AdaptiveTransportResult:
    concentrations_mol_l: np.ndarray
    cell_edges_m: np.ndarray
    node_coordinates_m: np.ndarray
    nodal_concentrations_mol_l: np.ndarray
    cell_indicators: np.ndarray
    refinement_history: tuple[dict[str, float | int], ...]
    component_mass_balance_relative: dict[str, float]
    component_mass_balance_offset_mol_l: dict[str, float]
    component_minimum_nodal_concentration_mol_l: dict[str, float]
    minimum_concentration_mol_l: float


def _domain_edges_1d(domain) -> tuple[np.ndarray, np.ndarray]:
    topology = domain.topology
    tdim = topology.dim
    if tdim != 1:
        raise ValueError("Expected a one-dimensional interval mesh")
    num_cells = topology.index_map(tdim).size_local
    geometry_dofmap = domain.geometry.dofmap
    coordinates = domain.geometry.x
    intervals = np.empty((num_cells, 2), dtype=float)
    for cell in range(num_cells):
        cell_coordinates = coordinates[geometry_dofmap[cell], 0]
        intervals[cell] = [np.min(cell_coordinates), np.max(cell_coordinates)]
    order = np.argsort(np.mean(intervals, axis=1))
    sorted_intervals = intervals[order]
    edges = np.concatenate((sorted_intervals[:, 0], sorted_intervals[-1:, 1]))
    if not np.allclose(sorted_intervals[:-1, 1], sorted_intervals[1:, 0], rtol=0.0, atol=1e-12):
        raise ValueError("Interval cells must form a contiguous partition")
    return edges, order


def solve_adaptive_transport_step(
    *,
    bulk_components_mol_m3: np.ndarray,
    component_names: tuple[str, ...],
    porosity_chemistry: np.ndarray,
    chemistry_edges_m: np.ndarray,
    inlet_concentrations_mol_l: np.ndarray,
    time_step_s: float,
    darcy_flux_m_s: float,
    reference_porosity: float,
    settings: PorosityTransportSettings,
) -> Case4AdaptiveTransportResult:
    """Solve one BE transport substep with Dörfler adaptive mesh refinement.

    A fixed PHREEQC chemistry grid bounds the transport refinement. Before and
    after every solve, component bulk inventories are conservatively remapped
    between that grid and the current CG1 mesh.
    """

    from mpi4py import MPI
    from dolfinx import mesh

    chem_edges = _validate_edges(chemistry_edges_m, "chemistry_edges_m")
    phi_chem = np.asarray(porosity_chemistry, dtype=float)
    bulk_chem = np.asarray(bulk_components_mol_m3, dtype=float)
    inlet = np.asarray(inlet_concentrations_mol_l, dtype=float)
    nchem = chem_edges.size - 1
    ncomponents = len(component_names)
    if phi_chem.shape != (nchem,) or np.any(phi_chem <= 0.0):
        raise ValueError("Chemistry porosity must be positive on every chemistry cell")
    if bulk_chem.shape != (ncomponents, nchem) or inlet.shape != (ncomponents,):
        raise ValueError("Component inventory/inlet shapes do not match names and chemistry grid")
    if not np.all(np.isfinite(bulk_chem)) or not np.all(np.isfinite(inlet)):
        raise ValueError("Component inventories and inlet values must be finite")

    domain = mesh.create_interval(
        MPI.COMM_SELF, settings.initial_cells, np.asarray([chem_edges[0], chem_edges[-1]])
    )
    chemistry_width = float(np.min(np.diff(chem_edges)))
    history: list[dict[str, float | int]] = []
    final_concentrations = None
    final_mesh_edges = None
    final_nodes = None
    final_nodal_profile = None
    final_indicators = None
    final_balance: dict[str, float] = {}
    final_offsets: dict[str, float] = {}
    final_minima: dict[str, float] = {}
    minimum_value = float("inf")

    for solve_number in range(1, settings.maximum_solves + 1):
        transport_edges, cell_order = _domain_edges_1d(domain)
        phi_transport = project_piecewise_constant(chem_edges, phi_chem, transport_edges)
        solver = ResidualMinimizationCase4Transport1D(
            domain=domain,
            cell_edges_m=transport_edges,
            porosity_cells=phi_transport,
            time_step_s=time_step_s,
            darcy_flux_m_s=darcy_flux_m_s,
            reference_porosity=reference_porosity,
            length_m=float(chem_edges[-1] - chem_edges[0]),
        )
        indicators_by_component = []
        gradient_indicators_by_component = []
        states_by_component = []
        output_concentrations = np.empty((ncomponents, nchem), dtype=float)
        balances: dict[str, float] = {}
        offsets: dict[str, float] = {}
        minima: dict[str, float] = {}
        minimum_value = float("inf")
        for index, name in enumerate(component_names):
            bulk_transport = project_piecewise_constant(
                chem_edges, bulk_chem[index], transport_edges
            )
            concentration_previous = bulk_transport / (1000.0 * phi_transport)
            step = solver.step(
                concentration_previous,
                float(inlet[index]),
                enforce_nonnegative=name != "Charge",
            )
            bulk_after = 1000.0 * project_linear_product_to_cells(
                step.cell_edges_m,
                step.nodal_concentrations_mol_l,
                chem_edges,
                phi_chem,
                chem_edges,
            )
            output_concentrations[index] = bulk_after / (1000.0 * phi_chem)
            balances[name] = step.mass_balance_relative
            offsets[name] = step.maximum_projection_correction_mol_l
            minima[name] = step.minimum_concentration_mol_l
            indicators_by_component.append(step.cell_indicators)
            gradient_indicators_by_component.append(step.gradient_jump_indicators)
            states_by_component.append(step)
            minimum_value = min(minimum_value, step.minimum_concentration_mol_l)

        indicators_by_component_array = np.asarray(indicators_by_component)
        union = np.zeros(transport_edges.size - 1, dtype=bool)
        for indicators in indicators_by_component_array:
            union |= dorfler_mark_1d(
                indicators,
                np.diff(transport_edges),
                theta=settings.theta,
                minimum_width=settings.eligibility_factor * chemistry_width,
                epsilon_eligible=settings.epsilon_eligible,
            )
        for indicators in gradient_indicators_by_component:
            union |= dorfler_mark_1d(
                indicators,
                np.diff(transport_edges),
                theta=settings.theta,
                minimum_width=settings.eligibility_factor * chemistry_width,
                epsilon_eligible=settings.epsilon_eligible,
            )
        marked_count = int(np.count_nonzero(union))
        history.append({
            "solve": solve_number,
            "cells": int(transport_edges.size - 1),
            "marked_cells": marked_count,
            "minimum_cell_width_m": float(np.min(np.diff(transport_edges))),
            "maximum_cell_width_m": float(np.max(np.diff(transport_edges))),
            "indicator_sum": float(np.sum(indicators_by_component_array)),
        })
        final_concentrations = output_concentrations
        final_mesh_edges = transport_edges
        final_indicators = np.max(indicators_by_component_array, axis=0)
        final_balance = balances
        final_offsets = offsets
        final_minima = minima
        # The mesh and chemistry grids intentionally share the same endpoints.
        # Save one representative component's nodal profile for diagnostics.
        diagnostic_component = int(np.argmax(np.sum(indicators_by_component_array, axis=1)))
        final_nodal_profile = states_by_component[diagnostic_component].nodal_concentrations_mol_l.copy()
        if solve_number >= settings.maximum_solves or marked_count == 0:
            # The next refinement solve is deliberately avoided when no cell
            # above the chemistry resolution carries enough residual.
            final_nodes = solver.node_coordinates.copy()
            break
        marked_entities = cell_order[np.flatnonzero(union)].astype(np.int32)
        # DOLFINx interval refinement creates MPI graph communicators. The
        # mixed UFL/PETSc objects can form Python reference cycles, so release
        # the completed solve before replacing its mesh and collect cycles to
        # avoid exhausting MPI's finite communicator context IDs in long runs.
        del solver
        gc.collect()
        # This solver is serial on COMM_SELF. Skip repartitioning/ghost updates;
        # the default identity partitioner creates a fresh MPI graph
        # communicator at every refinement and can exhaust MPI context IDs.
        domain, _, _ = mesh.refine(domain, edges=marked_entities, partitioner=None)

    if (final_concentrations is None or final_mesh_edges is None or final_nodes is None
            or final_nodal_profile is None):
        raise RuntimeError("Adaptive transport did not produce a completed solve")
    return Case4AdaptiveTransportResult(
        concentrations_mol_l=final_concentrations,
        cell_edges_m=final_mesh_edges,
        node_coordinates_m=final_nodes,
        nodal_concentrations_mol_l=final_nodal_profile,
        cell_indicators=np.asarray(final_indicators, dtype=float),
        refinement_history=tuple(history),
        component_mass_balance_relative=final_balance,
        component_mass_balance_offset_mol_l=final_offsets,
        component_minimum_nodal_concentration_mol_l=final_minima,
        minimum_concentration_mol_l=minimum_value,
    )

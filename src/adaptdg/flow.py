"""Flow state handed from the application to the transport solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FlowState:
    """Water content and Darcy flux over one time step.

    Attributes
    ----------
    mesh : dolfinx mesh.
    theta_old, theta_new : water content (volumetric) at the step start/end.
    q : Darcy flux, Raviart-Thomas RT0 function.
    """

    mesh: Any
    theta_old: Any
    theta_new: Any
    q: Any


# ---------------------------------------------------------------------------
# Steady Darcy flow solver (mixed RT0-DG0), copied unchanged from ASRTM-PHREEQC
# methods/momas2d/darcy.py. Its original module docstring follows.
# ---------------------------------------------------------------------------
# Steady Darcy flow for MoMaS Easy 2D: mixed RT0-DG0 on triangles.
#
# Problem (contract: examples/momas2d/spec.json, geometry_2d):
#     q = -K grad h,  div q = f   in Omega = [0, 2.1] x [0, 1]
#     q.n = g          on the flux boundary (inlets -0.0225, walls 0), essential
#     h   = h_D        on the outlet (x = 2.1, y in [0.2, 0.4]), natural
#
# Mixed weak form, (q, h) in RT0 x DG0 with q.n = g strongly:
#     (K^-1 q, v) - (h, div v) = -<h_D, v.n>_out
#     -(div q, w)              = -(f, w)
#
# RT0 gives one normal flux per facet, continuous between neighbours, and with
# f = 0 the cell balance div q = 0 holds to round-off. That facet flux is what
# the conservative DG upwind needs. RT0 spaces are nested under refinement, so a
# coarse q interpolates exactly (and conservatively) onto a refined mesh.
#
# Triangles, as in the existing 2D transport/ADAP code
# (asrtm_phreeqc.stabilized_transport_2d.create_rectangle_mesh_2d).


# MoMaS Easy 2D geometry and materials (contract values).
LX, LY = 2.1, 1.0
BLOCK_B = ((1.0, 1.1), (0.0, 0.9))
INLET_Q_DOT_N = -0.0225
INLET_TOP = (0.3, 0.4)  # x-range on y = 1
INLET_LEFT = (0.3, 0.4)  # y-range on x = 0
OUTLET = (0.2, 0.4)  # y-range on x = 2.1
OUTLET_HEAD = 1.0
MATERIALS = {  # tag: (name, porosity, K, total sites)
    0: ("A", 0.25, 1.0e-2, 1.0),
    1: ("B", 0.50, 1.0e-5, 10.0),
}

# Facet tags.
WALL, INLET1, INLET2, OUTLET_TAG = 1, 2, 3, 4
TOL = 1.0e-10


def create_momas_mesh(k: int, comm=None):
    """Triangular mesh with 21k x 10k squares; aligned with B and the boundary segments."""

    from mpi4py import MPI
    from dolfinx import mesh

    return mesh.create_rectangle(
        MPI.COMM_SELF if comm is None else comm,
        np.asarray([[0.0, 0.0], [LX, LY]]),
        (21 * k, 10 * k),
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.right,
    )


def material_tags(domain) -> np.ndarray:
    """Per local cell: 1 inside block B, 0 in A (by cell midpoint; mesh must be aligned)."""

    from dolfinx import mesh

    tdim = domain.topology.dim
    n = domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts
    mid = mesh.compute_midpoints(domain, tdim, np.arange(n, dtype=np.int32))
    (x0, x1), (y0, y1) = BLOCK_B
    inside = (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)
    return inside.astype(np.int32)


def cell_field(domain, tags: np.ndarray, index: int, name: str):
    """DG0 function with MATERIALS[tag][index] on each cell."""

    from dolfinx import fem

    V = fem.functionspace(domain, ("DG", 0))
    f = fem.Function(V, name=name)
    values = np.array([MATERIALS[t][index] for t in sorted(MATERIALS)])
    f.x.array[V.dofmap.list[: tags.size, 0]] = values[tags]
    return f


def momas_facet_tags(domain):
    """MeshTags on boundary facets: WALL, INLET1 (top), INLET2 (left), OUTLET_TAG (right)."""

    from dolfinx import mesh

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, fdim + 1)
    facets = mesh.exterior_facet_indices(domain.topology)
    mid = mesh.compute_midpoints(domain, fdim, facets)
    x, y = mid[:, 0], mid[:, 1]
    tag = np.full(facets.size, WALL, dtype=np.int32)
    tag[(np.abs(y - LY) < TOL) & (x > INLET_TOP[0]) & (x < INLET_TOP[1])] = INLET1
    tag[(np.abs(x) < TOL) & (y > INLET_LEFT[0]) & (y < INLET_LEFT[1])] = INLET2
    tag[(np.abs(x - LX) < TOL) & (y > OUTLET[0]) & (y < OUTLET[1])] = OUTLET_TAG
    order = np.argsort(facets)
    return mesh.meshtags(domain, fdim, facets[order], tag[order])


@dataclass
class DarcySolution:
    q: object  # dolfinx Function, RT0
    h: object  # dolfinx Function, DG0
    facet_tags: object
    info: dict = field(default_factory=dict)


def solve_darcy(
    domain,
    K,
    facet_tags,
    *,
    flux_bc,
    head_tags: dict,
    source=None,
):
    """Solve the mixed RT0-DG0 Darcy problem.

    K: UFL expression or DG0 Function (conductivity, scalar).
    flux_bc: (tags, q_bc) with q_bc a UFL vector expression or callable x -> (2, N);
        its normal component is imposed strongly on facets with those tags.
    head_tags: {tag: h_D (float or UFL expression)} weak Dirichlet head.
    source: f in div q = f (None for 0).
    """

    import basix.ufl
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem

    cell = domain.basix_cell()
    W = fem.functionspace(
        domain,
        basix.ufl.mixed_element([basix.ufl.element("RT", cell, 1), basix.ufl.element("DG", cell, 0)]),
    )
    (q, h), (v, w) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    a = (ufl.inner(q, v) / K - h * ufl.div(v) - ufl.div(q) * w) * ufl.dx
    L = sum(-hD * ufl.dot(v, n) * ds(tag) for tag, hD in head_tags.items())
    L = L + (-(source if source is not None else fem.Constant(domain, 0.0)) * w * ufl.dx)

    tags, q_bc_expr = flux_bc
    V_q, _ = W.sub(0).collapse()
    q_bc = fem.Function(V_q)
    if isinstance(q_bc_expr, ufl.core.expr.Expr):
        q_bc.interpolate(fem.Expression(q_bc_expr, V_q.element.interpolation_points))
    else:
        q_bc.interpolate(q_bc_expr)
    fdim = domain.topology.dim - 1
    facets = np.concatenate([facet_tags.find(t) for t in tags]).astype(np.int32)
    dofs = fem.locate_dofs_topological((W.sub(0), V_q), fdim, facets)
    bc = fem.dirichletbc(q_bc, dofs, W.sub(0))

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="darcy_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    wh = problem.solve()
    q_h = wh.sub(0).collapse()
    h_h = wh.sub(1).collapse()
    for obj in (problem.solver, problem.A, problem.b, problem.x):  # free PETSc objects (many meshes per run)
        obj.destroy()
    q_h.name, h_h.name = "q", "h"
    return DarcySolution(q=q_h, h=h_h, facet_tags=facet_tags,
                         info={"num_dofs": W.dofmap.index_map.size_global * W.dofmap.index_map_bs})


def momas_flux_bc(x: np.ndarray) -> np.ndarray:
    """Vector field whose normal flux is the MoMaS inflow on the inlets, zero elsewhere."""

    out = np.zeros((2, x.shape[1]))
    top = (np.abs(x[1] - LY) < TOL) & (x[0] > INLET_TOP[0] - TOL) & (x[0] < INLET_TOP[1] + TOL)
    left = (np.abs(x[0]) < TOL) & (x[1] > INLET_LEFT[0] - TOL) & (x[1] < INLET_LEFT[1] + TOL)
    out[1, top] = INLET_Q_DOT_N  # n = (0, 1)
    out[0, left] = -INLET_Q_DOT_N  # n = (-1, 0)
    return out


def solve_momas_darcy(k: int, comm=None):
    """MoMaS Easy 2D flow on the 21k x 10k triangular mesh."""

    domain = create_momas_mesh(k, comm)
    tags = material_tags(domain)
    K = cell_field(domain, tags, 2, "K")
    ft = momas_facet_tags(domain)
    sol = solve_darcy(domain, K, ft, flux_bc=((WALL, INLET1, INLET2), momas_flux_bc),
                      head_tags={OUTLET_TAG: OUTLET_HEAD})
    sol.info.update(k=k, material_tags=tags, K=K)
    return sol


# ---------------------------------------------------------------- diagnostics


def cell_divergence(q) -> np.ndarray:
    """Per cell: integral of div q over the cell (= net outflow), local cells."""

    import ufl
    from dolfinx import fem

    domain = q.function_space.mesh
    V0 = fem.functionspace(domain, ("DG", 0))
    w = ufl.TestFunction(V0)
    vec = fem.assemble_vector(fem.form(ufl.div(q) * w * ufl.dx))
    return vec.array[: V0.dofmap.index_map.size_local].copy()


def boundary_fluxes(q, facet_tags) -> dict:
    """{tag: integral of q.n} over each tagged boundary piece."""

    import ufl
    from dolfinx import fem

    domain = q.function_space.mesh
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    return {int(t): domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.dot(q, n) * ds(int(t)))))
            for t in np.unique(facet_tags.values)}


def interior_facet_flux_mismatch(q) -> float:
    """max over interior facets of |q.n(+) + q.n(-)| (facet-averaged); RT0 => round-off."""

    import ufl
    from dolfinx import fem

    domain = q.function_space.mesh
    n = ufl.FacetNormal(domain)
    V = fem.functionspace(domain, ("DG", 0))  # cell-wise accumulator of jumps
    w = ufl.TestFunction(V)
    jump = (ufl.dot(q("+"), n("+")) + ufl.dot(q("-"), n("-")))
    vec = fem.assemble_vector(fem.form(abs(jump) * ufl.avg(w) * ufl.dS))
    return float(np.max(np.abs(vec.array)))


def interface_head_jump(sol) -> float:
    """max |h(+) - h(-)| over interior facets between materials A and B."""

    domain = sol.h.function_space.mesh
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    f2c = domain.topology.connectivity(tdim - 1, tdim)
    tags = sol.info["material_tags"]
    dof = sol.h.function_space.dofmap.list[:, 0]
    worst = 0.0
    for f in range(domain.topology.index_map(tdim - 1).size_local):
        c = f2c.links(f)
        if c.size == 2 and tags[c[0]] != tags[c[1]]:
            worst = max(worst, abs(sol.h.x.array[dof[c[0]]] - sol.h.x.array[dof[c[1]]]))
    return worst


def section_flux(q, x0: float, y_range: tuple[float, float], npts: int = 2001) -> float:
    """Integral of q_x along the vertical line x = x0 (midpoint rule, point evaluation)."""

    from dolfinx import geometry

    domain = q.function_space.mesh
    ys = np.linspace(*y_range, npts + 1)
    ym = 0.5 * (ys[1:] + ys[:-1])
    pts = np.column_stack([np.full(npts, x0), ym, np.zeros(npts)])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    cells = np.array([coll.links(i)[0] for i in range(npts)], dtype=np.int32)
    vals = q.eval(pts, cells)
    return float(np.sum(vals[:, 0]) * (ys[1] - ys[0]))

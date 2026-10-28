"""Darcy RT0-DG0 solver (copied from ASRTM-PHREEQC methods/momas2d/darcy.py).

Adapted from the source script ``tests/momas2d_rt0_nesting.py`` (RT0 nesting under
local refinement) as a pytest test; asserts replace the printed diagnostics.
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from adaptdg import flow as darcy  # noqa: E402


def test_flow_state_stub_still_importable():
    assert darcy.FlowState.__name__ == "FlowState"


def test_momas_darcy_is_divergence_free_and_balanced():
    sol = darcy.solve_momas_darcy(1)
    div = darcy.cell_divergence(sol.q)
    assert np.abs(div).max() < 1e-10
    flux = darcy.boundary_fluxes(sol.q, sol.facet_tags)
    assert flux[darcy.INLET1] + flux[darcy.INLET2] + flux[darcy.OUTLET_TAG] == pytest.approx(0.0, abs=1e-10)
    assert abs(flux[darcy.WALL]) < 1e-10


def test_rt0_flux_nests_under_local_refinement():
    from dolfinx import fem, geometry, mesh

    sol = darcy.solve_momas_darcy(2)
    coarse = sol.q.function_space.mesh
    q_c = sol.q
    msh = coarse
    for _level in range(1, 3):
        msh.topology.create_entities(1)
        msh.topology.create_connectivity(1, 2)  # needed by compute_midpoints on dolfinx 0.11 (not in the source script)
        mid = mesh.compute_midpoints(msh, 1, np.arange(msh.topology.index_map(1).size_local, dtype=np.int32))
        x, y = mid[:, 0], mid[:, 1]
        marked = np.flatnonzero(((x > 0.8) & (x < 1.3) & (y < 1.0)) | ((x < 0.6) & (y > 0.2) & (y < 0.5))
                                | ((x > 0.2) & (x < 0.5) & (y > 0.8))).astype(np.int32)
        msh, _, _ = mesh.refine(msh, marked)
        V = fem.functionspace(msh, ("RT", 1))
        q_f = fem.Function(V)
        ncell = msh.topology.index_map(2).size_local
        cells = np.arange(ncell, dtype=np.int32)
        data = fem.create_interpolation_data(V, q_c.function_space, cells, padding=1e-12)
        q_f.interpolate_nonmatching(q_c, cells, data)
        verts = msh.geometry.x[mesh.entities_to_geometry(msh, 2, cells)][:, :, :2]
        lam = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
        pts = np.einsum("pv,cvd->cpd", lam, verts).reshape(-1, 2)
        pts3 = np.column_stack([pts, np.zeros(len(pts))])
        fine_cells = np.repeat(cells, 3)
        tree = geometry.bb_tree(coarse, 2)
        coll = geometry.compute_colliding_cells(coarse, geometry.compute_collisions_points(tree, pts3), pts3)
        coarse_cells = np.array([coll.links(i)[0] for i in range(len(pts3))], dtype=np.int32)
        vf, vc = q_f.eval(pts3, fine_cells), q_c.eval(pts3, coarse_cells)
        assert np.abs(vf - vc).max() / np.abs(vc).max() < 1e-10
        assert np.abs(darcy.cell_divergence(q_f)).max() < 1e-10

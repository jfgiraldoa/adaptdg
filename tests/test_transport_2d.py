"""2D DG1 / residual-minimisation transport (piece 5b): smoke and conservation checks.

New tests (the source has none for these classes besides the parent-map test kept in
``test_transfer_2d.py``); bitwise equality with the source was shown separately.
"""

import subprocess
import sys

import numpy as np
import pytest


def test_import_adaptdg_does_not_load_dolfinx():
    code = "import sys, adaptdg; sys.exit(int('dolfinx' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_transport_2d_modules_import_without_dolfinx():
    code = (
        "import sys, adaptdg.transport_2d, adaptdg.transport_dg2d, adaptdg.rm_transport, adaptdg.flow;"
        "sys.exit(int('dolfinx' in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_refine_marked_triangles_with_parent_counts():
    pytest.importorskip("dolfinx")
    from adaptdg.transport_2d import (
        create_rectangle_mesh_2d,
        mesh_triangles,
        refine_marked_triangles_with_parent,
    )

    domain = create_rectangle_mesh_2d(lower=(0.0, 0.0), upper=(1.0, 1.0), cells=(6, 6))
    none = np.zeros(72, dtype=bool)
    same, parent = refine_marked_triangles_with_parent(domain, none)
    assert same is domain and np.array_equal(parent, np.arange(72))
    marked = none.copy()
    marked[::5] = True
    refined, parent = refine_marked_triangles_with_parent(domain, marked)
    assert mesh_triangles(refined).shape[0] == parent.size > 72


def _momas_setup():
    from dolfinx import fem

    from adaptdg import flow

    sol = flow.solve_momas_darcy(1)
    dom = sol.q.function_space.mesh
    tags = flow.material_tags(dom)
    phi = flow.cell_field(dom, tags, 1, "phi")

    def dg0(value):
        f = fem.Function(fem.functionspace(dom, ("DG", 0)))
        f.x.array[:] = value
        return f

    return flow, sol, dom, phi, dg0


def test_conservative_dg2d_balance_and_positivity():
    pytest.importorskip("dolfinx")
    from adaptdg.transport_dg2d import ConservativeDG2D

    flow, sol, dom, phi, dg0 = _momas_setup()
    g = ConservativeDG2D(
        dom, q=sol.q, porosity=phi, alpha_L=dg0(1e-2), alpha_T=dg0(1e-3), facet_tags=sol.facet_tags,
        inlet_tags=(flow.INLET1, flow.INLET2), time_step=0.1, components=2, positive_components=(0, 1),
    )
    g.initialize(np.zeros((g.cells, 2)))
    for _ in range(3):
        means, diag = g.step(inlet=[1.0, 0.5])
    assert means.min() >= -1e-12
    assert all(d["scaled_defect"] < 1e-10 for d in diag)
    assert means[:, 0].max() > 0.0


def test_rm_transport_momas_runs_and_is_nonnegative_ish():
    pytest.importorskip("dolfinx")
    from adaptdg.rm_transport import RMTransportMoMaS2D

    flow, sol, dom, phi, dg0 = _momas_setup()
    r = RMTransportMoMaS2D(
        dom, q=sol.q, porosity=phi, alpha_L=dg0(1e-2), alpha_T=dg0(1e-3), facet_tags=sol.facet_tags,
        inlet_tags=(flow.INLET1, flow.INLET2), time_step=0.1,
    )
    prev = np.zeros(r.cells)
    for _ in range(3):
        out = r.step(prev, 1.0)
        prev = out.cell_averages
    r.destroy()
    assert np.all(np.isfinite(out.cell_averages)) and np.all(out.cell_indicators >= 0.0)
    assert out.inflow > 0.0

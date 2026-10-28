"""Heterogeneous (section 5.4) and anisotropic (section 5.5) cases of the VMS paper."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from dolfinx import fem  # noqa: E402

from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings, fitted_slope, mark_cells  # noqa: E402
from adaptdg.vms.problems import Anisotropic, Heterogeneous, heterogeneous_exact  # noqa: E402


def _cells(domain):
    xyz = domain.geometry.x[domain.geometry.dofmaps[0]]
    return xyz, xyz.mean(axis=1)


def _run(problem, p, **kw):
    loop = AdaptiveStationaryLoop(problem, LoopSettings(p=p, reconstruct=True, **kw))
    return loop, loop.run(progress=None)


def test_ufl_exact_solution_matches_numpy_and_pde():
    import ufl

    problem = Heterogeneous()
    domain = problem.initial_mesh(8)
    x = ufl.SpatialCoordinate(domain)
    space = fem.functionspace(domain, ("DG", 3))
    ex = fem.Function(space)
    ex.interpolate(fem.Expression(problem.exact(x), space.element.interpolation_points))
    pts = space.tabulate_dof_coordinates()
    assert np.allclose(ex.x.array, heterogeneous_exact(pts[:, 0]), atol=1e-12)
    # strong PDE residual, cell by cell (interface facets are mesh facets, so no cell straddles it)
    u = problem.exact(x)
    kap = problem.kappa(x)
    res = -ufl.div(ufl.dot(kap, ufl.grad(u))) + ufl.dot(problem.beta(x), ufl.grad(u))
    err = fem.assemble_scalar(fem.form(res**2 * ufl.dx(metadata={"quadrature_degree": 8})))
    assert np.sqrt(err) < 1e-6  # scale of the solution gradient is ~10


@pytest.mark.parametrize("refinement", ["uniform", "adaptive"])
def test_interface_stays_on_mesh_facets(refinement):
    loop = AdaptiveStationaryLoop(Heterogeneous(), LoopSettings(p=1, refinement=refinement, max_levels=4))
    loop.run(progress=None)
    xyz, _ = _cells(loop.domain)
    x = xyz[:, :, 0]
    assert ((x <= 0.5 + 1e-12).all(axis=1) | (x >= 0.5 - 1e-12).all(axis=1)).all()  # no straddling cell
    assert np.isclose(loop.domain.geometry.x[:, 0], 0.5).sum() > 5  # vertices lie on x = 1/2
    with pytest.raises(ValueError):
        Heterogeneous().initial_mesh(3)


@pytest.mark.parametrize("p", [1, 2])
def test_heterogeneous_uniform_rates_and_theta_identity(p):
    _, h = _run(Heterogeneous(), p, refinement="uniform", max_levels=5)
    kw = {"span": 20.0}  # last three levels
    for key in ("Vh", "Vh_u", "Vh_phi"):
        assert fitted_slope(h, key, 2, **kw) == pytest.approx(-p, abs=0.4)
    for key in ("L2_u", "L2_phi"):  # dG optimality recovered, one-sided (pre-asymptotic is faster/slower)
        assert fitted_slope(h, key, 2, **kw) < -(p + 1) + 0.5
    assert all(r["L2_u"] == pytest.approx(r["L2_theta"], rel=1e-8) for r in h)
    assert h[-1]["L2_phi"] < h[-1]["L2_u"] < h[-1]["L2"]  # phi most accurate in L2 on the same mesh


def test_heterogeneous_coarse_l2_loses_rate_for_even_degree_only():
    ratio = {}
    for p in (2, 3):
        _, h = _run(Heterogeneous(), p, refinement="uniform", max_levels=5 if p == 2 else 4)
        ratio[p] = [r["L2"] / r["L2_u"] for r in h]
    assert ratio[2][-1] > 1.5 and ratio[2][-1] > ratio[2][-2] > ratio[2][-3]  # ubar degrades (p = 2)
    assert ratio[3][-1] < 1.3  # odd degree: ubar is as good as u (measured ~1.07)


def test_heterogeneous_adaptive_runs_and_decreases_estimator():
    _, h = _run(Heterogeneous(), 1, max_levels=40, dofmax=15000)
    assert h[-1]["ndofs"] > 10 * h[0]["ndofs"]
    assert h[-1]["E"] < 0.2 * h[0]["E"]
    assert fitted_slope(h, "Vh", 2) < -0.6 and fitted_slope(h, "L2_u", 2) < -1.2


def test_anisotropic_source_is_derived_and_run_converges():
    import ufl

    problem = Anisotropic(r=1e4)
    assert problem.source(ufl.SpatialCoordinate(problem.initial_mesh(2))) is not None
    _, h = _run(problem, 2, max_levels=80, dofmax=12000)
    assert h[-1]["ndofs"] > 10 * h[0]["ndofs"]
    # measured about -2.2 (Vh) and -2.9 (L2) at 20000 DoF
    for key in ("Vh", "Vh_u", "Vh_phi"):
        assert fitted_slope(h, key, 2) < -1.5
    for key in ("L2", "L2_u", "L2_phi"):
        assert fitted_slope(h, key, 2) < -2.0


def test_anisotropic_refinement_clusters_at_the_inner_layer():
    r = 1e6
    problem = Anisotropic(r=r)
    loop, h = _run(problem, 1, max_levels=80, dofmax=15000)
    _, mid = _cells(loop.domain)
    band = np.abs(mid[:, 1]) < 2 / np.sqrt(r * problem.tau)  # 2/sqrt(r tau) = 0.063, 13 % of the area
    area = 2 * 2 / np.sqrt(r * problem.tau)
    marked = mark_cells(loop.result.indicators, 0.25, 0.2)
    assert band.mean() > 2 * area  # measured ~2.8x (cells)
    assert band[marked].mean() > 2 * area  # measured ~3.3x (marked cells)
    assert fitted_slope(h, "Vh", 2) < -0.4 and fitted_slope(h, "L2_phi", 2) < -0.6

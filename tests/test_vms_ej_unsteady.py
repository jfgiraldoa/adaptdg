"""Unsteady Eriksson-Johnson problem: exact solution, BE / BDF2 time orders, adaptive loop, history transfer, presets."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptdg.vms.preset import PRESETS, Preset
from adaptdg.vms.problems import PROBLEMS, ErikssonJohnsonUnsteady
from adaptdg.vms.run import build_problem
from adaptdg.vms.unsteady import build_unsteady_settings, step_coefficients, time_integrated

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vms" / "eriksson_johnson_unsteady"


@pytest.fixture(scope="module")
def dolfinx_env():
    return pytest.importorskip("dolfinx")


def _loop(p=1, integrator="bdf2", steps=3, kappa=1e-2, **kw):
    from adaptdg.vms.unsteady import UnsteadyLoop, UnsteadySettings

    dt = kw.pop("dt", 0.005)
    st = UnsteadySettings(p=p, integrator=integrator, dt=dt, t_final=steps * dt, **kw)
    loop = UnsteadyLoop(ErikssonJohnsonUnsteady(diffusivity=kappa), st)
    loop.run(progress=None)
    return loop


def test_step_coefficients_and_presets():
    assert step_coefficients("be", 0.1) == (1.0, 0.1, (1.0,))
    c1, c2, h = step_coefficients("bdf2", 0.3)
    assert (c1, h) == (1.0, (4 / 3, -1 / 3)) and c2 == pytest.approx(0.2) and sum(h) == pytest.approx(1.0)  # consistency
    with pytest.raises(ValueError):
        step_coefficients("cn", 0.1)
    paper, legacy = PRESETS["paper"], PRESETS["legacy"]
    assert (paper.step_mesh, paper.history_field, paper.history_projection) == ("previous", "u", "none")
    assert (legacy.step_mesh, legacy.history_field, legacy.history_projection) == ("initial", "phi", "gram_cg")
    with pytest.raises(ValueError):
        Preset(history_field="x")


def test_roots_match_the_legacy_formulas():
    k, lc = 1e-2, 2.0
    lamb1 = (-1 + np.sqrt(1 - 4 * k * lc)) / (-2 * k)  # legacy ErikssonJhonsson
    lamb2 = (-1 - np.sqrt(1 - 4 * k * lc)) / (-2 * k)
    r1 = (1 + np.sqrt(1 + 4 * np.pi**2 * k**2)) / (2 * k)
    s1 = (1 - np.sqrt(1 + 4 * np.pi**2 * k**2)) / (2 * k)
    np.testing.assert_allclose(ErikssonJohnsonUnsteady(diffusivity=k).roots(), (lamb1, lamb2, r1, s1), rtol=1e-12)


def test_registry_json_and_lazy_import():
    assert PROBLEMS["eriksson_johnson_unsteady"] is ErikssonJohnsonUnsteady
    pspec = json.loads((EXAMPLE / "problem.json").read_text())
    prob = build_problem(pspec)
    assert (prob.diffusivity, prob.decay, prob.dirichlet_tags, prob.neumann_tags) == (1e-2, 2.0, (2, 3, 4), (1,))
    numerics = json.loads((EXAMPLE / "numerics.json").read_text())
    for preset in ("paper", "legacy"):
        nspec = {k: v for k, v in numerics.items() if k != "example"}
        nspec.update(numerics["example"]["presets"][preset], preset=preset)
        nspec["tol"] = 0.0
        s = build_unsteady_settings(nspec, pspec)
        assert s.preset == preset and s.dt == 0.005 and s.t_final == 0.1 and s.initial_mesh_size == 4
    with pytest.raises(KeyError):
        build_unsteady_settings({"bogus": 1})
    rows = [{"L2": 1.0, "Vh": 2.0}, {"L2": 3.0, "Vh": 2.0}]
    out = time_integrated(rows, 0.5)
    assert out["int_L2"] == pytest.approx(np.sqrt(0.5 * 10)) and out["max_L2"] == 3.0
    code = ("import sys, adaptdg.vms.unsteady;"
            "bad = [m for m in ('dolfinx', 'petsc4py', 'ufl', 'basix') if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))


def test_exact_solution_pde_data_and_legacy_neumann(dolfinx_env):
    import ufl
    from dolfinx import fem

    kappa, h = 1e-2, 1e-5
    prob = ErikssonJohnsonUnsteady(diffusivity=kappa)
    dom = prob.initial_mesh(4)
    prob.bind(dom, 0.03)
    x = ufl.SpatialCoordinate(dom)
    u = prob.exact(x)
    cells = np.arange(dom.topology.index_map(2).size_local, dtype=np.int32)
    pts = np.array([[1 / 3, 1 / 3], [0.2, 0.4]])  # reference points inside every cell
    res = ufl.dot(prob.beta(x), ufl.grad(u)) - kappa * ufl.div(ufl.grad(u))
    r = fem.Expression(res, pts).eval(dom, cells)
    vals = []
    for t in (0.03 - h, 0.03 + h):
        prob.set_time(t)
        vals.append(fem.Expression(u, pts).eval(dom, cells))
    prob.set_time(0.03)
    ut = (vals[1] - vals[0]) / (2 * h)
    scale = np.abs(fem.Expression(ufl.dot(prob.beta(x), ufl.grad(u)), pts).eval(dom, cells)).max()
    assert np.abs(ut + r).max() < 1e-5 * scale  # u_t + b.grad u - kappa Lap u = 0
    # initial condition / Dirichlet data against the legacy uDexp
    l1, l2, r1, s1 = prob.roots()
    t = 0.03
    legacy_u = lambda xx, yy: (np.exp(-2 * t) * (np.exp(l1 * xx) - np.exp(l2 * xx))  # noqa: E731
                               + np.cos(np.pi * yy) * (np.exp(s1 * xx) - np.exp(r1 * xx)) / (np.exp(-s1) - np.exp(-r1)))
    assert legacy_u(0.0, 0.3) == pytest.approx(0.0, abs=1e-12)  # both layers vanish at the outflow
    space = fem.functionspace(dom, ("DG", 1))
    f = fem.Function(space)
    f.interpolate(fem.Expression(u, space.element.interpolation_points))
    xs = space.tabulate_dof_coordinates()
    np.testing.assert_allclose(f.x.array, legacy_u(xs[:, 0], xs[:, 1]), rtol=1e-10, atol=1e-12)


def test_neumann_datum_equals_legacy_formula_on_the_inflow_face(dolfinx_env):
    import ufl
    from dolfinx import fem

    kappa, t = 1e-2, 0.03
    prob = ErikssonJohnsonUnsteady(diffusivity=kappa)
    dom = prob.initial_mesh(4)
    prob.bind(dom, t)
    x, n = ufl.SpatialCoordinate(dom), ufl.FacetNormal(dom)
    l1, l2, r1, s1 = prob.roots()
    y = ufl.SpatialCoordinate(dom)[1]
    legacy = -kappa * ((l1 * np.exp(-l1) - l2 * np.exp(-l2)) * np.exp(-2 * t)
                       + (-r1 * np.exp(-r1) + s1 * np.exp(-s1)) * ufl.cos(ufl.pi * y) / (np.exp(-s1) - np.exp(-r1)))
    from adaptdg.vms.meshes import tag_boundary

    ds1 = ufl.Measure("ds", domain=dom, subdomain_data=tag_boundary(dom, prob.boundary_markers()),
                      metadata={"quadrature_degree": 8})(1)
    diff = fem.assemble_scalar(fem.form((prob.neumann(x, n) - legacy) ** 2 * ds1))
    assert abs(diff) < 1e-20


def _order(integrator, p, n, kappa=0.1, T=0.1):
    errs = []
    for m in (4, 8, 16):
        loop = _loop(p=p, integrator=integrator, steps=m, dt=T / m, kappa=kappa, refinement="uniform", initial_mesh_size=n,
                     error_targets="exact", quadrature_degree=12)
        errs.append(loop.history[-1]["L2_theta"])
    return errs, [np.log2(errs[i] / errs[i + 1]) for i in range(2)]


def test_backward_euler_is_first_order_in_time(dolfinx_env):
    errs, orders = _order("be", 4, 6)
    assert errs[0] > errs[1] > errs[2] and all(0.9 < o < 1.1 for o in orders), (errs, orders)


def test_bdf2_is_second_order_in_time(dolfinx_env):
    errs, orders = _order("bdf2", 4, 6)
    assert errs[0] > errs[1] > errs[2] and all(1.85 < o < 2.2 for o in orders), (errs, orders)


def test_adaptive_run_beats_uniform_refines_to_the_layer_and_reports_both_targets(dolfinx_env):
    ada = _loop(p=2, steps=3, max_levels=14, tol=1e-3, theta_ref=0.25)
    last = ada.history[-1]
    for key in ("L2", "Vh", "L2_proj", "Vh_proj", "L2_theta", "Vh_phi", "L2_u", "Vh_phi_proj"):
        assert np.isfinite(last[key]) and last[key] > 0, key
    assert ada.history[0]["levels"] > 1 and last["levels"] <= 14
    n = ada.domain.topology.index_map(2).size_local
    xyz = ada.domain.geometry.x[ada.domain.geometry.dofmaps[0][:n]]
    h = np.sqrt(np.abs(np.linalg.det(xyz[:, 1:, :2] - xyz[:, :1, :2])) / 2)
    xm = xyz[:, :, 0].mean(axis=1)
    assert h[xm < -0.5].mean() / h[xm > -0.05].mean() > 4.0
    side = int(np.sqrt(last["ndofs"] / 8 / 4) + 1)  # a uniform mesh with about the same number of DoF (p = 2: 8 per cell)
    uni = _loop(p=2, steps=3, refinement="uniform", initial_mesh_size=side)
    assert uni.history[-1]["ndofs"] > 0.7 * last["ndofs"]
    assert last["L2"] < 0.2 * uni.history[-1]["L2"] and last["L2_proj"] < 0.2 * uni.history[-1]["L2_proj"]


@pytest.mark.parametrize("integrator", ["be", "bdf2"])
def test_legacy_preset_restarts_from_the_initial_mesh(dolfinx_env, integrator):
    loop = _loop(p=1, integrator=integrator, steps=2, max_levels=4, preset="legacy")
    first = [r for r in loop.levels if r["level"] == 0]
    assert [r["ndofs"] for r in first] == [first[0]["ndofs"]] * 2  # every step starts from the 4 x 4 mesh
    assert loop.history[-1]["levels"] == 4 and np.isfinite(loop.history[-1]["L2_phi_proj"])


def test_history_transfer_is_the_identity_on_nested_meshes_and_keeps_the_integral(dolfinx_env):
    import ufl
    from dolfinx import fem

    from adaptdg.vms.refine import refine_marked
    from adaptdg.vms.unsteady import transfer

    prob = ErikssonJohnsonUnsteady()
    d0 = prob.initial_mesh(4)
    marked = np.zeros(d0.topology.index_map(2).size_local, dtype=bool)
    marked[:12] = True
    d1 = refine_marked(d0, marked)[0]
    v0, v1 = fem.functionspace(d0, ("DG", 2)), fem.functionspace(d1, ("DG", 2))
    f0 = fem.Function(v0)
    f0.interpolate(lambda x: np.sin(3 * x[0]) * np.cos(2 * x[1]) + x[0] ** 3)
    f1 = transfer(f0, v1)
    m0 = fem.assemble_scalar(fem.form(f0 * ufl.dx))
    m1 = fem.assemble_scalar(fem.form(f1 * ufl.dx))
    assert m1 == pytest.approx(m0, abs=1e-12)
    xx = ufl.SpatialCoordinate(d1)
    err = fem.assemble_scalar(fem.form((f1 - (ufl.sin(3 * xx[0]) * ufl.cos(2 * xx[1]) + xx[0] ** 3)) ** 2 * ufl.dx(degree=10)))
    assert err < 1e-5  # only the DG(2) interpolation error of the sampled function
    f2 = transfer(f1, v0)  # coarsening: L2 projection, the integral is kept
    assert fem.assemble_scalar(fem.form(f2 * ufl.dx)) == pytest.approx(m0, abs=1e-6)


def test_stationary_forms_are_unchanged_without_neumann_tags_or_time_coefficients(dolfinx_env):
    from adaptdg.vms.dgforms import DGForms
    from adaptdg.vms.problems import ErikssonJohnson

    prob = ErikssonJohnson()
    forms = DGForms(prob.initial_mesh(4), prob, 1)
    assert forms.ds_n is None and forms.ds_g is forms.ds and forms.time_coeffs is None and forms.history == []


def test_adaptive_refinement_does_not_leak_mpi_communicators(dolfinx_env):
    """DOLFINx 0.11 ``mesh.refine`` with the default partitioner leaks one communicator per call; MPICH has only 2048,
    so the full legacy run died with "Too many communicators". ``refine_marked``/``refine_uniform`` avoid it in serial."""
    import gc

    from mpi4py import MPI

    from adaptdg.vms.refine import refine_marked, refine_uniform

    if "MPICH" not in MPI.Get_library_version() or MPI.COMM_WORLD.size != 1:
        pytest.skip("communicator footprint is probed through MPICH handle indices, serial only")

    def footprint():  # highest MPICH handle index of 200 fresh duplicates ~ number of live communicators
        comms = [MPI.COMM_WORLD.Dup() for _ in range(200)]
        top = max(MPI._handleof(c) & 0x3FFFFFF for c in comms)
        for c in comms:
            c.Free()
        return top

    problem = ErikssonJohnsonUnsteady(diffusivity=1e-2)
    gc.collect()
    before = footprint()
    for _ in range(20):  # every call refines a new parent mesh, as the adaptive loop does
        domain = problem.initial_mesh(4)
        for _ in range(3):
            marked = np.zeros(domain.topology.index_map(2).size_local, dtype=bool)
            marked[:3] = True
            domain = refine_marked(domain, marked)[0]
        domain.topology.create_entities(1)
        refine_uniform(domain)
    gc.collect()
    assert footprint() - before <= 10  # unfixed: +80, fixed: +5 (live meshes)

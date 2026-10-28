"""Steady Eriksson-Johnson problem (section 5.6 of the VMS paper): exact solution, tags, loop in both presets."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptdg.vms.problems import ErikssonJohnson, ErikssonJohnson3D, Problem
from adaptdg.vms.run import build_problem, build_settings

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vms" / "eriksson_johnson"


@pytest.fixture(scope="module")
def dolfinx_env():
    return pytest.importorskip("dolfinx")


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("ej_example_run", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(preset, levels, dofmax=10**9, refinement="adaptive", **extra):
    problem = json.loads((EXAMPLE / "problem.json").read_text())
    numerics = json.loads((EXAMPLE / "numerics.json").read_text())
    nspec = {k: v for k, v in numerics.items() if k != "example"}
    nspec.update(numerics["example"]["presets"][preset], preset=preset, p=1, max_levels=levels,
                 dofmax=dofmax, refinement=refinement, **extra)
    return problem, nspec


def test_import_adaptdg_vms_is_lazy():
    code = (
        "import sys, adaptdg.vms;"
        "from adaptdg.vms import ErikssonJohnson, ErikssonJohnson3D;"
        "from adaptdg.vms.problems import PROBLEMS;"
        "bad = [m for m in ('dolfinx', 'petsc4py', 'ufl', 'basix', 'mpi4py') if m in sys.modules];"
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_roots_solve_the_characteristic_equation():
    for kappa in (1e-1, 1e-2, 1e-3):
        r1, r2 = ErikssonJohnson(diffusivity=kappa).roots()
        for r in (r1, r2):  # u = exp(r x) sin(pi y) in -kappa Delta u + u_x = 0: kappa (r^2 - pi^2) - r = 0
            assert abs(kappa * (r**2 - np.pi**2) - r) < 1e-9 * max(1.0, abs(kappa * r**2))
        assert r1 > 1 / kappa - 1 and r2 < 0


def test_example_json_files_are_consistent():
    problem = json.loads((EXAMPLE / "problem.json").read_text())
    numerics = json.loads((EXAMPLE / "numerics.json").read_text())
    prob = build_problem(problem)
    assert isinstance(prob, ErikssonJohnson) and prob.dim == 2 and prob.diffusivity == pytest.approx(1e-2)  # legacy
    assert prob.velocity == (1.0, 0.0) and 1e-3 in problem["kappa_full"]
    assert build_problem({"name": "eriksson_johnson_3d"}).dim == 3
    assert isinstance(build_problem({"name": "eriksson_johnson_3d", "kappa": 0.1}), ErikssonJohnson3D)
    ex = numerics["example"]
    assert ex["presets_3d"]["legacy"]["preset_overrides"] == {"marking_ratio": 0.125}
    for preset in ("paper", "legacy"):
        nspec = {k: v for k, v in numerics.items() if k != "example"}
        nspec.update(ex["presets"][preset], preset=preset)
        s = build_settings(nspec, problem)
        assert s.preset == preset and s.initial_mesh_size == 4 and s.error_targets == "both" and s.reconstruct


@pytest.mark.parametrize("cls,n", [(ErikssonJohnson, 6), (ErikssonJohnson3D, 2)])
@pytest.mark.parametrize("kappa", [1e-2, 1e-1])
def test_exact_solution_satisfies_pde_and_boundary_data(dolfinx_env, cls, n, kappa):
    import ufl
    from dolfinx import fem

    prob = cls(diffusivity=kappa)
    dom = prob.initial_mesh(n)
    space = fem.functionspace(dom, ("DG", 4))
    x = ufl.SpatialCoordinate(dom)
    assert prob.source(x) is None  # f = 0 by definition
    pts = space.tabulate_dof_coordinates()
    res, u = fem.Function(space), fem.Function(space)
    res.interpolate(fem.Expression(Problem.source(prob, x), space.element.interpolation_points))  # symbolic residual
    u.interpolate(fem.Expression(prob.exact(x), space.element.interpolation_points))
    scale = kappa * max(abs(r) for r in prob.roots()) ** 2  # size of the individual terms of the PDE
    assert np.abs(res.x.array).max() < 1e-10 * max(scale, 1.0)
    r1, r2 = prob.roots()
    ref = (np.exp(r1 * (pts[:, 0] - 1)) - np.exp(r2 * (pts[:, 0] - 1))) / (np.exp(-r1) - np.exp(-r2)) \
        * np.sin(np.pi * pts[:, 1])
    assert np.allclose(u.x.array, ref, atol=1e-12)
    inflow, outflow = np.isclose(pts[:, 0], 0.0), np.isclose(pts[:, 0], 1.0)
    assert np.allclose(u.x.array[inflow], np.sin(np.pi * pts[inflow, 1]), atol=1e-12)  # u(0, y) = sin(pi y)
    assert np.allclose(u.x.array[outflow], 0.0, atol=1e-12)  # u(1, y) = 0
    side = np.isclose(pts[:, 1], 0.0) | np.isclose(pts[:, 1], 1.0)
    assert np.allclose(u.x.array[side], 0.0, atol=1e-12)
    if cls is ErikssonJohnson3D:  # independent of z
        assert np.allclose(u.x.array, ref)


@pytest.mark.parametrize("cls,measure", [(ErikssonJohnson, 4.0), (ErikssonJohnson3D, 6.0)])
def test_boundary_tags_cover_the_unit_box(dolfinx_env, cls, measure):
    import ufl
    from dolfinx import fem

    from adaptdg.vms import meshes

    prob = cls()
    dom = prob.initial_mesh(4)
    tags = meshes.tag_boundary(dom, prob.boundary_markers())
    ds = ufl.Measure("ds", domain=dom, subdomain_data=tags)
    assert sorted(set(tags.values)) == list(prob.dirichlet_tags)
    assert sum(fem.assemble_scalar(fem.form(1.0 * ds(t))) for t in prob.dirichlet_tags) == pytest.approx(measure)
    assert fem.assemble_scalar(fem.form(1.0 * ufl.Measure("dx", domain=dom))) == pytest.approx(1.0)
    assert prob.beta(ufl.SpatialCoordinate(dom)).ufl_shape == (prob.dim,)


def test_adaptive_loop_decreases_error_and_refines_towards_the_layer(dolfinx_env, example):
    problem, nspec = _run("paper", levels=14, dofmax=4000)
    history, domain = example.run_one(problem, nspec)
    assert len(history) >= 6 and history[-1]["ndofs"] > 5 * history[0]["ndofs"]
    assert history[-1]["E"] < 0.5 * history[0]["E"] and history[-1]["Vh"] < 0.15 * history[0]["Vh"]
    assert history[-1]["L2"] < 0.2 * history[0]["L2"]
    stats = example.layer_clustering(domain)
    assert stats["h_near"] < 0.3 * stats["h_far"]  # much smaller cells in the outflow layer x > 0.9
    assert stats["h_far_over_near"] > 3.0


@pytest.mark.parametrize("preset", ["paper", "legacy"])
def test_both_presets_and_both_targets_run(dolfinx_env, example, preset):
    problem, nspec = _run(preset, levels=3)
    history, _ = example.run_one(problem, nspec)
    assert len(history) == 3
    for key in ("L2", "Vh", "L2_proj", "Vh_proj", "L2_theta", "Vh_theta", "L2_phi", "Vh_phi_proj"):
        assert all(np.isfinite(row[key]) and row[key] > 0 for row in history), key
    assert history[-1]["E"] < history[0]["E"] * 1.2 and history[-1]["ndofs"] > history[0]["ndofs"]


def test_presets_differ(dolfinx_env, example):
    paper, _ = example.run_one(*_run("paper", levels=2))
    legacy, _ = example.run_one(*_run("legacy", levels=2))
    assert paper[0]["ndofs"] == legacy[0]["ndofs"]  # same start mesh
    assert paper[0]["E"] != pytest.approx(legacy[0]["E"], rel=1e-3)  # Gram form and penalty differ


def test_3d_variant_runs(dolfinx_env, example):
    problem, nspec = _run("paper", levels=2)
    problem.update(name="eriksson_johnson_3d", beta=[1.0, 0.0, 0.0], initial_mesh_size=2)
    nspec.update(reconstruct=False, theta_ref=0.125)
    history, domain = example.run_one(problem, nspec)
    assert domain.topology.dim == 3 and len(history) == 2
    assert all(np.isfinite(row["Vh"]) and np.isfinite(row["L2_proj"]) for row in history)
    assert history[1]["ndofs"] > history[0]["ndofs"]

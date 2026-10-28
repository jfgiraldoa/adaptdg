"""Fichera corner (section 5.3 of the VMS paper): mesh, data, adaptive loop in both presets and targets."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptdg.vms.problems import Fichera3D
from adaptdg.vms.run import build_problem, build_settings

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vms" / "fichera_3d"


@pytest.fixture(scope="module")
def dolfinx_env():
    return pytest.importorskip("dolfinx")


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("fichera_example_run", EXAMPLE / "run.py")
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
        "from adaptdg.vms import Fichera3D;"
        "bad = [m for m in ('dolfinx', 'petsc4py', 'ufl', 'basix', 'mpi4py') if m in sys.modules];"
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_example_json_files_are_consistent():
    problem = json.loads((EXAMPLE / "problem.json").read_text())
    numerics = json.loads((EXAMPLE / "numerics.json").read_text())
    prob = build_problem(problem)
    assert isinstance(prob, Fichera3D) and prob.dim == 3 and prob.q == pytest.approx(1 / 3)
    assert problem["initial_mesh_size"] == 2 and 0.1 in problem["q_full"]
    ex = numerics["example"]
    assert ex["presets"]["legacy"]["preset_overrides"] == {"marking_ratio": 0.125}  # 3D value of the legacy tests
    for preset in ("paper", "legacy"):
        nspec = {k: v for k, v in numerics.items() if k != "example"}
        nspec.update(ex["presets"][preset], preset=preset)
        s = build_settings(nspec, problem)
        assert s.preset == preset and s.initial_mesh_size == 2 and s.error_targets == "both" and s.reconstruct
    assert build_problem({"name": "fichera", "q": 0.1}).q == 0.1


def test_fichera_volume_and_boundary(dolfinx_env):
    import ufl
    from dolfinx import fem

    from adaptdg.vms import meshes
    from adaptdg.vms.refine import refine_marked

    dom = meshes.fichera(4)
    assert dom.topology.index_map(3).size_local == 7 * 8 * 6
    marked = np.zeros(dom.topology.index_map(3).size_local, bool)
    marked[:20] = True
    for d in (dom, refine_marked(dom, marked)[0]):
        tags = meshes.tag_boundary(d, {1: lambda x: np.full(x.shape[1], True)})
        ds = ufl.Measure("ds", domain=d, subdomain_data=tags)
        assert fem.assemble_scalar(fem.form(1.0 * ufl.Measure("dx", domain=d))) == pytest.approx(7.0)
        assert fem.assemble_scalar(fem.form(1.0 * ds(1))) == pytest.approx(24.0)  # 6 faces of area 4 (the cut adds none)
    x = d.geometry.x
    assert x.min() == -1.0 and x.max() == 1.0 and not ((x > 0).all(axis=1) & (x < 1).all(axis=1)).any()


@pytest.mark.parametrize("q", [0.1, 1 / 3])
def test_exact_solution_and_forcing_match_the_closed_form(dolfinx_env, q):
    import ufl
    from dolfinx import fem

    prob = Fichera3D(q=q)
    dom = prob.initial_mesh(2)
    space = fem.functionspace(dom, ("DG", 3))
    x = ufl.SpatialCoordinate(dom)
    f, u = fem.Function(space), fem.Function(space)
    f.interpolate(fem.Expression(prob.source(x), space.element.interpolation_points))
    u.interpolate(fem.Expression(prob.exact(x), space.element.interpolation_points))
    pts = space.tabulate_dof_coordinates()
    r = np.linalg.norm(pts, axis=1)
    keep = r > 0.3  # f is singular at the origin (not in L2 for q <= 1/2)
    assert keep.sum() > 100
    away = r > 1e-8  # the origin is a node: r = 0 up to round-off there
    assert np.allclose(u.x.array[away], r[away] ** q, atol=1e-12)
    assert np.allclose(f.x.array[keep], -q * (q + 1) * r[keep] ** (q - 2), rtol=1e-10)


def test_adaptive_loop_decreases_error_and_clusters_at_the_corner(dolfinx_env, example):
    problem, nspec = _run("paper", levels=5, dofmax=6000)
    history, domain = example.run_one(problem, nspec)
    assert len(history) >= 4 and history[-1]["ndofs"] > history[0]["ndofs"]
    assert history[-1]["E"] < 0.8 * history[0]["E"] and history[-1]["Vh"] < 0.7 * history[0]["Vh"]
    assert history[-1]["L2"] < 0.3 * history[0]["L2"]
    r, h = example.cell_sizes(domain)
    assert h[r < 0.3].mean() < 0.6 * h[r > 0.7].mean()  # cells shrink towards the corner
    stats = example.clustering(r, h)
    assert 0 < stats["h_min"] < stats["h_max"] and stats["h_far"] > 0


@pytest.mark.parametrize("preset", ["paper", "legacy"])
def test_both_presets_and_both_targets_run(dolfinx_env, example, preset):
    problem, nspec = _run(preset, levels=3)
    history, _ = example.run_one(problem, nspec)
    assert len(history) == 3
    for key in ("L2", "Vh", "L2_proj", "Vh_proj", "L2_theta", "Vh_theta", "L2_phi", "Vh_phi_proj"):
        assert all(np.isfinite(row[key]) and row[key] > 0 for row in history), key
    assert history[-1]["E"] < history[0]["E"]


def test_presets_differ(dolfinx_env, example):
    paper, _ = example.run_one(*_run("paper", levels=2))
    legacy, _ = example.run_one(*_run("legacy", levels=2))
    assert paper[0]["ndofs"] == legacy[0]["ndofs"]  # same start mesh
    assert paper[0]["E"] != pytest.approx(legacy[0]["E"], rel=1e-3)  # Gram form and penalty differ


def test_uniform_sequence_uses_structured_meshes(dolfinx_env, example):
    problem, nspec = _run("paper", levels=4, dofmax=1500, refinement="uniform", reconstruct=False)
    history, _ = example.run_one(problem, nspec, step=2)
    assert [row["level"] for row in history] == list(range(len(history))) and history[-1]["ndofs"] >= 1500
    assert [row["ncells"] for row in history[:2]] == [7 * 6, 7 * 8 * 6]  # n = 2, 4: 7 of 8 cubes, 6 tetrahedra each
    assert "L2_phi" not in history[0]

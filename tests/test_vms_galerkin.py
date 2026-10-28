"""CG(p) Galerkin baseline: convergence rates, legacy Poisson check, agreement with residual minimisation, loop."""

import json
from pathlib import Path

import numpy as np
import pytest

from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings
from adaptdg.vms.problems import PoissonSine, ReentrantCorner

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vms" / "poisson_fem"


@pytest.fixture(scope="module")
def dolfinx_env():
    return pytest.importorskip("dolfinx")


def test_example_json_files_are_consistent():
    problem = json.loads((EXAMPLE / "problem.json").read_text())
    numerics = json.loads((EXAMPLE / "numerics.json").read_text())
    from adaptdg.vms.run import build_problem, build_settings

    assert isinstance(build_problem(problem), ReentrantCorner)
    s = build_settings({k: v for k, v in numerics.items() if k != "example"}, problem)
    assert s.method == "rm" and s.compare_galerkin and s.error_targets == "both"
    assert build_problem({"name": "poisson_sine"}).amplitude == 10.0


def test_galerkin_converges_at_optimal_rates_on_a_smooth_solution(dolfinx_env):
    from adaptdg.vms.galerkin import galerkin_errors, solve_galerkin

    prob = PoissonSine()
    for p in (1, 2):
        errs = []
        for n in (4, 8, 16):
            dom = prob.initial_mesh(n)
            e = galerkin_errors(solve_galerkin(prob, dom, p), prob, p)
            errs.append(e["exact"]["ubar"])
        (l2a, vha), (l2b, vhb) = errs[1], errs[2]
        assert np.log2(l2a / l2b) == pytest.approx(p + 1, abs=0.15)  # L2: p + 1
        assert np.log2(vha / vhb) == pytest.approx(p, abs=0.1)  # Vh (H1-like): p
        assert set(e) == {"exact", "projection"}


def test_legacy_poisson_fem_norm(dolfinx_env):
    """Legacy ``test_poisson_FEM_solution``: P1 on UnitSquareMesh(10, 10); reference L2 norm 0.247140 (delta 0.01)."""
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    from adaptdg.vms.galerkin import solve_galerkin

    dom = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    u = solve_galerkin(PoissonSine(), dom, 1).u
    norm = np.sqrt(fem.assemble_scalar(fem.form(u * u * ufl.dx)))
    assert norm == pytest.approx(0.247140, abs=0.01)


def test_galerkin_and_rm_agree_on_the_smooth_case(dolfinx_env):
    from adaptdg.vms.galerkin import galerkin_errors, solve_galerkin
    from adaptdg.vms.norms import error_targets
    from adaptdg.vms.saddle import RMSolver

    prob, p = PoissonSine(), 2
    dom = prob.initial_mesh(8)
    g = galerkin_errors(solve_galerkin(prob, dom, p), prob, p, degree=8)["exact"]["ubar"]
    rm = error_targets(RMSolver(prob, p, quadrature_degree=8).solve(dom), prob, p, degree=8, targets=("exact",))
    r = rm["exact"]["ubar"]
    assert 0.2 < g[0] / r[0] < 5 and 0.2 < g[1] / r[1] < 5  # same order of discretisation error
    assert g[0] < 1e-3 and r[0] < 1e-3


def test_galerkin_with_advection_and_both_targets_in_the_loop(dolfinx_env):
    s = LoopSettings(p=1, method="galerkin", refinement="uniform", max_levels=3, initial_mesh_size=4)
    hist = AdaptiveStationaryLoop(PoissonSine(), s).run(progress=None)
    assert len(hist) == 3 and "E" not in hist[0]
    for key in ("L2", "Vh", "L2_proj", "Vh_proj"):
        assert all(np.isfinite(r[key]) and r[key] > 0 for r in hist)
    assert hist[2]["L2"] < 0.3 * hist[1]["L2"]
    with pytest.raises(ValueError):
        AdaptiveStationaryLoop(PoissonSine(), LoopSettings(method="galerkin", refinement="adaptive"))
    with pytest.raises(ValueError):
        AdaptiveStationaryLoop(PoissonSine(), LoopSettings(method="fem"))


@pytest.mark.parametrize("preset", ["paper", "legacy"])
def test_reentrant_corner_compare_galerkin_columns(dolfinx_env, preset):
    s = LoopSettings(p=1, preset=preset, max_levels=4, compare_galerkin=True, initial_mesh_size=4, reconstruct=True)
    hist = AdaptiveStationaryLoop(ReentrantCorner(), s).run(progress=None)
    assert len(hist) == 4
    for key in ("L2_cg", "Vh_cg", "L2_cg_proj", "Vh_cg_proj", "Vh_phi", "E"):
        assert all(np.isfinite(r[key]) and r[key] > 0 for r in hist), key
    assert hist[-1]["Vh_cg"] < hist[0]["Vh_cg"]


def test_galerkin_runs_with_advection_unstabilised(dolfinx_env):
    from adaptdg.vms.galerkin import galerkin_errors, solve_galerkin
    from adaptdg.vms.problems import ErikssonJohnson

    prob = ErikssonJohnson(diffusivity=0.1)  # moderate Peclet number: plain Galerkin is accurate on a fine mesh
    dom = prob.initial_mesh(16)
    l2, _ = galerkin_errors(solve_galerkin(prob, dom, 1), prob, 1, targets=("exact",))["exact"]["ubar"]
    assert l2 < 5e-3

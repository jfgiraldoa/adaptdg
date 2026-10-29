"""Nonlinear VMS cases (paper section 6.5): Newton, forms, presets, shock of the single-component Burgers equation."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from adaptdg.vms.problems import PROBLEMS, BurgersSingle, NonlinearProblem
from adaptdg.vms.run import build_problem, build_settings

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "vms"
SINGLE = EXAMPLES / "burgers_single"


@pytest.fixture(scope="module")
def dolfinx_env():
    return pytest.importorskip("dolfinx")


def _settings(preset, levels, p=3, example=SINGLE, **extra):
    pspec = json.loads((example / "problem.json").read_text())
    numerics = json.loads((example / "numerics.json").read_text())
    nspec = {k: v for k, v in numerics.items() if k != "example"}
    nspec.update(numerics["example"]["presets"][preset], preset=preset, p=p, max_levels=levels, dofmax=10**9, **extra)
    return pspec, nspec


def _loop(pspec, nspec, **kappa):
    from adaptdg.vms.nonlinear import NonlinearAdaptiveLoop

    loop = NonlinearAdaptiveLoop(build_problem({**pspec, **kappa}), build_settings(nspec, pspec))
    loop.run(None)
    return loop


def test_import_nonlinear_is_lazy():
    code = (
        "import sys, adaptdg.vms, adaptdg.vms.nonlinear;"
        "from adaptdg.vms import BurgersSingle, NonlinearProblem;"
        "bad = [m for m in ('dolfinx', 'petsc4py', 'ufl', 'basix', 'mpi4py') if m in sys.modules];"
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_problem_registry_and_example_files():
    assert PROBLEMS["burgers_single"] is BurgersSingle and issubclass(BurgersSingle, NonlinearProblem)
    pspec = json.loads((SINGLE / "problem.json").read_text())
    numerics = json.loads((SINGLE / "numerics.json").read_text())
    prob = build_problem(pspec)
    assert isinstance(prob, BurgersSingle) and prob.diffusivity == pytest.approx(1e-2) and not prob.has_exact
    assert prob.dirichlet_tags == (1, 2, 3) and prob.outflow_tags == (4,)
    assert pspec["kappa_full"] == [1e-2, 1e-3, 1e-4] and pspec["initial_mesh_size"] == 4 and numerics["p"] == 3
    for preset in ("paper", "legacy"):
        nspec = {k: v for k, v in numerics.items() if k != "example"}
        nspec.update(numerics["example"]["presets"][preset], preset=preset)
        s = build_settings(nspec, pspec)
        assert s.preset == preset and s.initial_mesh_size == 4
    with pytest.raises(NotImplementedError):
        prob.exact(None)


def test_preset_fields_of_the_nonlinear_choices():
    from adaptdg.vms.nonlinear import newton_options
    from adaptdg.vms.preset import resolve_preset

    paper, legacy = resolve_preset("paper"), resolve_preset("legacy")
    assert (paper.nonlinear_flux, paper.newton_method) == ("lax_friedrichs", "hybrid")
    assert (legacy.nonlinear_flux, legacy.newton_method) == ("wind", "damped")
    op, ol = newton_options("paper"), newton_options("legacy")
    assert op.method == "hybrid" and not op.legacy_damping and op.damping_fallback
    assert ol.method == "damped" and ol.legacy_damping and ol.maxit == 20000 and ol.tol_du == 1e-12
    assert ol.omega == 0.5 and ol.k_max == 1e14
    with pytest.raises(ValueError):
        resolve_preset("paper", nonlinear_flux="x")


@pytest.mark.parametrize("preset,p", [("paper", 1), ("paper", 2), ("legacy", 1)])
def test_jacobian_matches_finite_differences(dolfinx_env, preset, p):
    from dolfinx.fem import petsc
    from petsc4py import PETSc

    import dolfinx.fem as fem
    from adaptdg.vms.nonlinear import NonlinearRMSolver

    prob = BurgersSingle(diffusivity=1e-2)
    s = NonlinearRMSolver(prob, p, preset=preset).system(prob.initial_mesh(2))
    rng = np.random.default_rng(0)
    n = s.sol.x.array.size
    s.sol.x.array[s.map_c] = 0.3 + 0.4 * np.sin(np.arange(len(s.map_c)))  # an arbitrary state
    s.sol.x.array[s.map_r] = 0.02 * rng.standard_normal(len(s.map_r))
    x0 = s.sol.x.array.copy()
    res = fem.form(s.residual)

    def residual() -> np.ndarray:
        vec = petsc.assemble_vector(res)
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return vec.array.copy()

    f0 = residual()
    kinds = ("full",) if preset == "legacy" else ("full", "gn")
    for kind in kinds:
        mat = petsc.assemble_matrix(s.jacobian(kind))
        mat.assemble()
        for name, rows_cols in (("eps", s.map_r), ("u", s.map_c), ("both", np.arange(n))):
            d = np.zeros(n)
            d[rows_cols] = rng.standard_normal(len(rows_cols))
            xv, jd = mat.createVecRight(), mat.createVecLeft()
            xv.array[:] = d
            mat.mult(xv, jd)
            h = 1e-6
            s.sol.x.array[:] = x0 + h * d
            fd = (residual() - f0) / h
            s.sol.x.array[:] = x0
            err = lambda rows: np.linalg.norm((fd - jd.array)[rows]) / np.linalg.norm(fd[rows])  # noqa: E731
            if kind == "full":
                assert err(np.arange(n)) < 1e-4, (kind, name)
            elif name != "u":  # Gauss-Newton: the eps-equation and the d_eps direction are exact
                assert err(s.map_r) < 1e-4, (kind, name)
            else:  # the second derivative n'' of the u-equation is dropped
                assert err(s.map_c) > 1e-3


def test_hybrid_newton_converges_on_adaptive_levels_p1(dolfinx_env):
    pspec, nspec = _settings("paper", 4, p=1, error_targets="exact")
    loop = _loop(pspec, nspec)
    assert len(loop.history) == 4
    for row in loop.history:
        assert row["newton_converged"] == 1 and row["newton_residual"] < 1e-9 and row["newton_iterations"] <= 40
        assert "L2" not in row  # no exact solution: the error-target columns do not exist
    assert loop.history[-1]["ndofs"] > loop.history[0]["ndofs"]


def test_legacy_preset_records_failures_without_raising(dolfinx_env):
    pspec, nspec = _settings("legacy", 3, p=3, error_targets="exact")
    nspec["newton"] = {"maxit": 40}
    loop = _loop(pspec, nspec, kappa=1e-3)  # the legacy damped Newton is known to fail on some levels here
    assert len(loop.history) == 3
    for row in loop.history:
        assert row["newton_status"] in ("conv", "failK", "maxit", "diverged", "singular")
        assert (row["newton_status"] == "conv") == bool(row["newton_converged"])
        assert np.isfinite(row["E"]) and row["newton_fallback"] == 0
    # the failures are explicit: either every level converged or the status says why not
    assert any(r["newton_status"] != "conv" for r in loop.history) or all(r["newton_converged"] for r in loop.history)


def test_newton_fallback_retries_with_the_hybrid_solver(dolfinx_env):
    pspec, nspec = _settings("legacy", 2, p=3, error_targets="exact")
    nspec.update(newton={"maxit": 5}, newton_fallback=True)  # too few damped iterations: the fallback must act
    loop = _loop(pspec, nspec, kappa=1e-2)
    assert all(r["newton_converged"] and r["newton_fallback"] for r in loop.history)


def test_outflow_term_is_necessary(dolfinx_env):
    from adaptdg.vms.nonlinear import NonlinearRMSolver

    @dataclass
    class NoOutflow(BurgersSingle):
        outflow_tags: tuple = ()

    with_out = NonlinearRMSolver(BurgersSingle(), 3).solve(BurgersSingle().initial_mesh(4))
    assert with_out.newton["converged"] and with_out.newton["residual"] < 1e-9
    without = NonlinearRMSolver(NoOutflow(), 3).solve(NoOutflow().initial_mesh(4))  # the natural flux v f(u).n is missing
    assert not without.newton["converged"] and without.newton["residual"] > 1e-6


def test_shock_at_one_half_with_the_viscous_slope(dolfinx_env):
    from adaptdg.vms.nonlinear import shock_diagnostics

    pspec, nspec = _settings("paper", 9, p=3, error_targets="exact")
    loop = _loop(pspec, nspec)  # kappa = 1e-2
    shock = shock_diagnostics(loop.result.ubar, y=0.75)
    assert abs(shock["position"] - 0.5) < 0.02 and shock["crossings"] == 1
    assert abs(shock["max_slope"] / (-1 / (2 * 1e-2)) - 1) < 0.15
    last = loop.history[-1]
    assert last["umin"] > -1.02 and last["umax"] < 1.02  # no oscillation beyond the boundary data


def test_adaptive_mesh_clusters_along_the_shock(dolfinx_env):
    pspec, nspec = _settings("paper", 9, p=3, error_targets="exact")
    loop = _loop(pspec, nspec)
    d = loop.domain
    n = d.topology.index_map(2).size_local
    xyz = d.geometry.x[d.geometry.dofmaps[0][:n]]
    h = np.sqrt(np.abs(np.linalg.det(xyz[:, 1:, :2] - xyz[:, :1, :2])) / 2)
    dist = np.abs(xyz[:, :, 0].mean(axis=1) - 0.5)
    assert h[dist > 0.25].mean() / h[dist < 0.05].mean() > 3.0


def test_both_presets_run_and_report_the_estimator(dolfinx_env):
    for preset in ("paper", "legacy"):
        pspec, nspec = _settings(preset, 2, p=3, error_targets="both")
        loop = _loop(pspec, nspec)
        row = loop.history[-1]
        assert row["E"] > 0 and row["ndofs"] > 0 and {"newton_iterations", "umin", "umax"} <= set(row)
        assert "Vh" not in row and "L2_proj" not in row  # n/a without an exact solution


def test_example_run_script_summarises_the_newton_statistics():
    spec = importlib.util.spec_from_file_location("burgers_single_run", SINGLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hist = [{"newton_converged": 1, "newton_iterations": 5, "newton_fallback": 0, "newton_status": "conv"},
            {"newton_converged": 0, "newton_iterations": 100, "newton_fallback": 1, "newton_status": "maxit"}]
    out = module.newton_summary(hist)
    assert out["converged"] == 1 and out["failed_levels"] == [2] and out["fallback_levels"] == [2]

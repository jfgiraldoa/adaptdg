"""VMS module: checks that need neither dolfinx nor ufl."""

import os
import subprocess
import sys

import numpy as np
import pytest

from adaptdg.marking import dorfler_union
from adaptdg.vms.loop import LoopSettings, fitted_slope, mark_cells
from adaptdg.vms.problems import Anisotropic, Heterogeneous, heterogeneous_exact
from adaptdg.vms.run import build_problem, build_settings


def test_import_does_not_load_heavy_dependencies():
    code = (
        "import sys, adaptdg, adaptdg.vms, adaptdg.vms.run, adaptdg.vms.loop,"
        "adaptdg.vms.reconstruction, adaptdg.vms.norms, adaptdg.vms.problems;"
        "bad = [m for m in ('dolfinx', 'petsc4py', 'ufl', 'basix') if m in sys.modules];"
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_marking_maps_legacy_ref_type_1():
    g = np.array([0.01, 0.5, 0.2, 0.3, 0.001, 0.25])
    expected = dorfler_union({"e": g}, None, 0.0, 0.25, 0.0, 0.2, eligibility=np.ones(6, bool))
    marked = mark_cells(g, 0.25, 0.2)
    assert np.array_equal(marked, expected)
    assert marked[1]  # the largest indicator is always marked
    assert g[marked].sum() >= 0.25**2 * g.sum()
    assert not marked[4]


def test_fitted_slope_recovers_power_law():
    hist = [{"ndofs": 4.0**k, "E": (2.0**k) ** -1.5} for k in range(2, 9)]
    assert fitted_slope(hist, "E", 2) == pytest.approx(-1.5)


def test_loader_defaults_and_unknown_keys():
    problem = build_problem({"name": "fichera", "alpha": 0.1})
    assert problem.dim == 3 and problem.q == 0.1
    assert build_problem({}).dim == 2
    settings = build_settings({"p": 3}, {"initial_mesh_size": 6})
    assert settings == LoopSettings(p=3, initial_mesh_size=6)
    with pytest.raises(KeyError):
        build_settings({"degree": 3})


def test_loader_new_cases_and_reconstruction_flags():
    het = build_problem({"name": "heterogeneous", "eps1": 0.2, "beta": [1.0, 0.0]})
    assert isinstance(het, Heterogeneous) and het.eps1 == 0.2 and het.eps2 == 1.0
    assert het.velocity == (1.0, 0.0)
    ani = build_problem({"name": "anisotropic", "r": 1e6, "tau": 1e-3})
    assert isinstance(ani, Anisotropic) and ani.r == 1e6
    s = build_settings({"reconstruct": True, "adjoint_ordering": "legacy"})
    assert s.reconstruct and s.adjoint_ordering == "legacy"
    assert not LoopSettings().reconstruct


@pytest.mark.parametrize("eps", [(0.1, 1.0), (0.5, 2.0), (1.0, 1.0)])
def test_heterogeneous_exact_solves_the_1d_problem(eps):
    """Paper eqs. (39)-(40): -(eps u')' + u' = 0, u(0) = 1, u(1) = 0, u and eps u' continuous at 1/2."""
    e1, e2 = eps

    def u(x):
        return heterogeneous_exact(np.asarray(x, dtype=float), e1, e2)

    assert u(0.0) == pytest.approx(1.0) and u(1.0) == pytest.approx(0.0, abs=1e-12)
    d = 1e-6
    assert u(0.5 - d) == pytest.approx(u(0.5 + d), abs=1e-5)  # continuous value
    def eps_du(x, e):
        return e * (u(x + 1e-7) - u(x - 1e-7)) / 2e-7

    assert eps_du(0.5 - 2e-6, e1) == pytest.approx(eps_du(0.5 + 2e-6, e2), rel=1e-3)  # flux
    h = 1e-3  # ODE residual by finite differences away from the interface
    for x0, e in ((0.2, e1), (0.8, e2)):
        upp = (u(x0 + h) - 2 * u(x0) + u(x0 - h)) / h**2
        up = (u(x0 + h) - u(x0 - h)) / (2 * h)
        assert -e * upp + up == pytest.approx(0.0, abs=1e-4 * max(1.0, abs(up) / e))

"""VMS module: checks that need neither dolfinx nor ufl."""

import os
import subprocess
import sys

import numpy as np
import pytest

from adaptdg.marking import dorfler_union
from adaptdg.vms.loop import LoopSettings, fitted_slope, mark_cells
from adaptdg.vms.run import build_problem, build_settings


def test_import_does_not_load_heavy_dependencies():
    code = (
        "import sys, adaptdg, adaptdg.vms, adaptdg.vms.run, adaptdg.vms.loop;"
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

"""Presets and the legacy fixed-fraction marking: checks that need neither dolfinx nor ufl."""

import os
import subprocess
import sys

import numpy as np
import pytest

from adaptdg.marking import fixed_fraction_bands
from adaptdg.vms.preset import PRESETS, Preset, resolve_preset
from adaptdg.vms.run import build_settings


def test_fixed_fraction_hand_computed_example():
    g = np.array([10.0, 9.6, 5.2, 3.9, 1.0, 0.4])  # total 30.1
    # ratio 0.25 -> 7.525: the first band (> 0.95 * 10) already holds 19.6
    assert fixed_fraction_bands(g, 0.25).tolist() == [True, True, False, False, False, False]
    # ratio 0.9 -> 27.09: bands add 5.2 (frac 0.5) and 3.9 (frac 0.35), then 28.7 >= 27.09
    marked = fixed_fraction_bands(g, 0.9)
    assert marked.tolist() == [True, True, True, True, False, False]
    assert g[marked].sum() == pytest.approx(28.7)


def test_fixed_fraction_sum_property_and_minimality():
    rng = np.random.default_rng(3)
    for ratio in (0.1, 0.25, 0.5, 0.99, 1.0):
        g = rng.random(200) ** 3
        marked = fixed_fraction_bands(g, ratio)
        assert g[marked].sum() >= ratio * g.sum() - 1e-12  # marked sum reaches the target
        assert marked.any() and g[marked].min() >= g[~marked].max() if (~marked).any() else True


def test_fixed_fraction_edge_cases():
    assert not fixed_fraction_bands(np.zeros(4)).any()
    assert fixed_fraction_bands(np.array([-3.0, 1.0]), 0.5).tolist() == [True, False]  # absolute values
    assert fixed_fraction_bands(np.ones(5), 1.0).all()  # ties: one band marks everything
    with pytest.raises(ValueError):
        fixed_fraction_bands(np.ones(3), 0.0)


def test_presets_and_overrides():
    assert resolve_preset() is PRESETS["paper"] and resolve_preset(None) is PRESETS["paper"]
    paper, legacy = PRESETS["paper"], PRESETS["legacy"]
    assert (paper.gram_l2, paper.gamma_factor_b, paper.marking) == (True, 1.0, "dorfler")
    assert paper.quadrature_degree is None and paper.adjoint_ordering is None
    assert (legacy.gram_l2, legacy.gamma_factor_b, legacy.gamma_factor_gram) == (False, 0.5, 0.5)
    assert (legacy.quadrature_degree, legacy.marking, legacy.marking_ratio) == (8, "fixed_fraction", 0.25)
    assert legacy.adjoint_ordering == "legacy"
    mixed = resolve_preset("legacy", gram_l2=True, marking="dorfler")
    assert mixed.gram_l2 and mixed.marking == "dorfler" and mixed.gamma_factor_b == 0.5
    assert resolve_preset(mixed) is mixed
    with pytest.raises(KeyError):
        resolve_preset("nope")
    with pytest.raises(KeyError):
        resolve_preset("legacy", bogus=1)
    with pytest.raises(ValueError):
        Preset(marking="x")


def test_numerics_json_selects_preset():
    s = build_settings({"preset": "legacy", "preset_overrides": {"gram_l2": True}})
    assert s.preset == "legacy" and s.preset_overrides == {"gram_l2": True}
    assert build_settings({}).preset == "paper" and build_settings({}).error_targets == "both"


def test_preset_and_norms_imports_do_not_load_heavy_dependencies():
    code = (
        "import sys, adaptdg.marking, adaptdg.vms.preset, adaptdg.vms.norms, adaptdg.vms.loop;"
        "bad = [m for m in ('dolfinx', 'petsc4py', 'ufl', 'basix') if m in sys.modules];"
        "assert not bad, bad"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

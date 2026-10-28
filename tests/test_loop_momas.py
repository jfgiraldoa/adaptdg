"""MoMaS 2D ADAP loop and uniform coupled run with a fake chemistry (no PHREEQC).

The dolfinx tests are skipped without dolfinx."""

import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest

SPEC = {
    "materials": {
        "A": {"phi": 0.25, "alpha_advective": [0.01, 0.001], "alpha_dispersive": [10.0, 1.0]},
        "B": {"phi": 0.5, "alpha_advective": [0.06, 0.006], "alpha_dispersive": [60.0, 6.0]},
    },
    "time": {"snapshots": [0.01], "injection_ends": 0.015},
}


class Snap:
    def __init__(self, mobile, fixed, sites, water):
        self.mobile, self.fixed, self.sites, self.water_mass_kg = mobile, fixed, sites, water
        self.primary = mobile.copy()
        self.aqueous = mobile.copy()
        self.surface = fixed[:, :2].copy()

    @property
    def total(self):
        return self.mobile + self.fixed


class FakeEngine:
    """Conserving toy reaction: 10 % of the mobile totals goes to the fixed totals."""

    def __init__(self, materials, volume):
        n = len(materials)
        self.water_volume_l = np.full(n, 1.0)
        self.site_total = np.zeros(n)
        self.snapshot = Snap(np.zeros((n, 4)), np.full((n, 4), 0.01), np.zeros(n), self.water_volume_l.copy())
        self.calls = []

    def react(self, mobile):
        self.calls.append(np.array(mobile, copy=True))
        fixed = self.snapshot.fixed + 0.1 * mobile
        self.snapshot = Snap(0.9 * mobile, fixed, np.zeros(len(mobile)), self.water_volume_l.copy())
        return self.snapshot


class FakeChemistry:
    spec = SPEC

    def __init__(self):
        self.engines = []
        self.opened = []

    @contextmanager
    def open_chemistry(self, materials, representative_volume_l, *, concentration_scale,
                       zero_tolerance, anchor_mode):
        self.opened.append((len(materials), concentration_scale, zero_tolerance, anchor_mode))
        engine = FakeEngine(materials, representative_volume_l)
        self.engines.append(engine)
        yield engine

    def inlet_mobile(self, *, concentration_scale):
        return np.array([[1.0, 0.5, 0.2, 0.1], [0.0, 0.5, 0.0, 0.0]])


def _dorfler_union_inline(indicators, sizes, h_chem, theta, eligibility_factor, epsilon_chem, tie_tolerance):
    """Body of ``methods/momas2d/adap.py::dorfler_union`` (asrtm-phreeqc-public @ a152b91), verbatim."""
    eligible = sizes > eligibility_factor * h_chem
    marked = np.zeros(sizes.size, dtype=bool)
    for values in indicators:
        values = np.abs(values)
        total = float(values.sum())
        if total <= 0.0 or float(values[eligible].sum()) / total < epsilon_chem:
            continue
        target = theta**2 * total
        order = np.argsort(values)[::-1]
        acc, cutoff, last = 0.0, None, -1
        for pos, cell in enumerate(order):
            if not eligible[cell]:
                continue
            marked[cell] = True
            acc += float(values[cell])
            cutoff = float(values[cell])
            last = pos
            if acc >= target:
                break
        if cutoff is None:
            continue
        for cell in order[last + 1:]:
            if not eligible[cell]:
                continue
            if cutoff - (1.0 + tie_tolerance) * values[cell] < 0.0:
                marked[cell] = True
            else:
                break
    return marked


def test_loop_uses_exact_dorfler_adapter():
    import adaptdg.loop_momas as loop_momas

    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(5, 40))
        inds = [np.round(rng.random(n), 1) * rng.integers(0, 2) for _ in range(int(rng.integers(1, 4)))]
        sizes = rng.choice([0.05, 0.1, 0.2], size=n)
        args = (0.1, float(rng.choice([0.24, 1.0])), float(rng.choice([0.1, 0.5])), 0.35)
        theta = float(rng.choice([0.3, 0.6]))
        assert np.array_equal(
            loop_momas.dorfler_union(inds, sizes, args[0], theta, *args[1:]),
            _dorfler_union_inline(inds, sizes, args[0], theta, *args[1:]),
        )


def test_import_isolation():
    code = (
        "import sys, adaptdg, adaptdg.loop_momas, adaptdg.coupled_momas;"
        "bad = [m for m in ('dolfinx', 'phreeqcrm', 'asrtm_dg', 'asrtm_phreeqc') if m in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_protocols_document_the_engine_interface():
    import adaptdg.loop_momas as loop_momas

    assert hasattr(loop_momas, "ChemistryMomas") and hasattr(loop_momas, "EngineMomas")
    assert "react" in loop_momas.EngineMomas.__dict__ and "open_chemistry" in loop_momas.ChemistryMomas.__dict__


def test_adaptive_run_with_fake_chemistry(capsys):
    pytest.importorskip("dolfinx")
    import adaptdg.loop_momas as loop_momas

    def go():
        chem = FakeChemistry()
        res = loop_momas.run(1, "dispersive", 0.01, 0.02, [0.02], solves=2, chemistry=chem)
        return chem, res

    chem, res = go()
    assert chem.opened == [(210, 1e-6, 1e-16, "one_end")]
    assert res["stats"]["steps"] == 3  # the event at 0.015 splits the second step
    assert set(res["saved"]) == {0.02}
    assert res["saved"][0.02]["mobile"].shape == (210, 4)
    assert res["stats"]["max_reaction_defect"] < 1e-8
    assert len(chem.engines[0].calls) == 3
    chem2, res2 = go()  # deterministic
    assert np.array_equal(res["saved"][0.02]["mobile"], res2["saved"][0.02]["mobile"])
    assert "[progress]" in capsys.readouterr().out


def test_uniform_run_with_fake_chemistry():
    pytest.importorskip("dolfinx")
    import adaptdg.coupled_momas as coupled_momas
    from adaptdg import flow as darcy

    flow = darcy.solve_momas_darcy(1)
    domain = flow.q.function_space.mesh
    materials = np.where(flow.info["material_tags"] == 1, "B", "A")
    groups = coupled_momas.square_groups(domain, 0.1, 0.1, 10)
    assert groups.max() + 1 == 210
    chem = FakeChemistry()
    res = coupled_momas.run_coupled(
        domain, flow.facet_tags, (darcy.INLET1, darcy.INLET2), flow.q, materials,
        regime="dispersive", dt=0.01, end_time=0.02, groups=groups, save_times=[0.02],
        chem_volume_l=10.0, chemistry=chem,
    )
    assert chem.opened == [(210, 1e-6, 1e-16, "one_end")]
    assert res["stats"]["steps"] == 3 and res["stats"]["chemistry_cells"] == 210
    assert 0.02 in res["saved"] and res["stats"]["max_reaction_defect"] < 1e-8

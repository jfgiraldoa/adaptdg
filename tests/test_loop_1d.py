"""1D RM operator and ADAP loop with a fake decay chemistry (no PHREEQC, numpy/scipy only)."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adaptdg.loop_1d import (
    Adap1DRun,
    Adap1DSettings,
    cells_to_dg1,
    dg1_to_cells,
    positivity_fix,
    positivity_mixing,
)
from adaptdg.rm_1d import RMOperator1D

COMPONENTS = ("a", "b")
L, N, DT, D, V = 1.0, 20, 0.05, 1e-4, 1.0


class DecayChemistry:
    """First-order decay of every component; exposes the documented Chemistry1D interface."""

    def __init__(self, cells, rate=0.5):
        self.rate = rate
        self.boundary = {"a": 1.0, "b": 0.5}
        self.snapshot = SimpleNamespace(concentrations={c: np.zeros(cells) for c in COMPONENTS})
        self.calls = []

    def react(self, inputs, *, time_step):
        self.calls.append((dict(inputs), time_step))
        out = {c: np.asarray(v) * np.exp(-self.rate * time_step) for c, v in inputs.items()}
        self.snapshot = SimpleNamespace(concentrations=out)
        return self.snapshot


def _run(settings, chemistry=None, cells=N):
    chem = chemistry or DecayChemistry(cells)
    return Adap1DRun(chem, COMPONENTS, 0.0, L, cells, DT, D, V, settings), chem


def test_import_is_light():
    code = (
        "import sys, adaptdg.loop_1d, adaptdg.rm_1d;"
        "assert not {'dolfinx', 'asrtm_dg', 'asrtm_phreeqc', 'phreeqcrm'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_operator_validation_and_dg0_equals_constant_dg1():
    edges = np.linspace(0.0, 1.0, 9)
    with pytest.raises(ValueError):
        RMOperator1D(edges[::-1], time_step=DT, diffusion=D, velocity=V)
    with pytest.raises(ValueError):
        RMOperator1D(edges, time_step=DT, diffusion=D, velocity=-1.0)
    op = RMOperator1D(edges, time_step=DT, diffusion=D, velocity=V)
    p = np.linspace(0.0, 1.0, 8)
    a = op.solve(p, 1.0)
    b = op.solve(np.column_stack((p, p)), 1.0)
    assert np.array_equal(a["nodal_values"], b["nodal_values"])
    with pytest.raises(ValueError):
        op.solve(np.zeros((8, 3)), 1.0)


def test_exact_transfers_roundtrip_and_conserve_mass():
    rng = np.random.default_rng(1)
    edges = np.sort(np.concatenate(([0.0, 1.0], rng.uniform(0.05, 0.95, 6))))
    chem_edges = np.linspace(0.0, 1.0, 11)
    dg1 = rng.uniform(size=(edges.size - 1, 2))
    cells = dg1_to_cells(edges, dg1, chem_edges)
    mass_dg1 = np.sum(0.5 * (dg1[:, 0] + dg1[:, 1]) * np.diff(edges))
    assert np.sum(cells * np.diff(chem_edges)) == pytest.approx(mass_dg1, rel=1e-13)
    back = cells_to_dg1(chem_edges, cells, edges)
    mass_back = np.sum(0.5 * (back[:, 0] + back[:, 1]) * np.diff(edges))
    assert mass_back == pytest.approx(mass_dg1, rel=1e-13)


def test_positivity_fixes():
    u = np.array([1.0, -0.2, 0.5, 2.0])
    fixed, added = positivity_fix(u, "clip")
    assert added == pytest.approx(0.2) and fixed[1] == 0.0
    fixed, added = positivity_fix(u, "conservative")
    assert added == 0.0 and np.all(fixed >= 0.0) and fixed.sum() == pytest.approx(u.sum())
    with pytest.raises(ValueError):
        positivity_fix(u, "nope")
    fields = {"a": np.array([1.0, -0.5, 1.0]), "b": np.array([2.0, -1.0, 2.0])}
    mixed, windows, added = positivity_mixing(fields)
    assert windows == 1 and all(np.all(v >= 0.0) for v in mixed.values())
    assert mixed["a"].sum() == pytest.approx(1.5) and added["a"] == 0.0


def test_marking_matches_verbatim_source_body():
    """``_marking`` (adapter) equals the original inline implementation."""
    run, _ = _run(Adap1DSettings(initial_cells=7))
    rng = np.random.default_rng(3)
    s = run.settings

    def source_marking(indicators):
        widths = np.diff(run.edges)
        eligible = widths > s.eligibility_factor * run.chem_width
        marked = np.zeros(widths.size, dtype=bool)
        for c in run.components:
            values = np.abs(indicators[c])
            total = float(values.sum())
            if total <= 0.0 or float(values[eligible].sum()) / total < s.epsilon_chem:
                continue
            target = s.theta ** 2 * total
            order = np.argsort(values)[::-1]
            acc, cutoff, last = 0.0, None, -1
            for p, cell in enumerate(order):
                if not eligible[cell]:
                    continue
                marked[cell] = True
                acc += float(values[cell])
                cutoff, last = float(values[cell]), p
                if acc >= target:
                    break
            if cutoff is None:
                continue
            for cell in order[last + 1:]:
                if not eligible[cell]:
                    continue
                if cutoff - (1.0 + s.tie_tolerance) * values[cell] < 0.0:
                    marked[cell] = True
                else:
                    break
        return marked

    for trial in range(30):
        run.edges = np.linspace(0.0, L, 8)
        if trial % 3 == 0:
            run.edges = np.sort(np.concatenate((run.edges, rng.uniform(0.0, L, 3))))
        n = run.edges.size - 1
        ind = {c: rng.uniform(size=n) ** (1 + trial % 4) for c in COMPONENTS}
        if trial % 5 == 0:
            ind["a"][:] = 0.3  # ties
        assert np.array_equal(run._marking(ind), source_marking(ind))


def test_adaptive_run_refines_and_is_consistent():
    settings = Adap1DSettings(initial_cells=4, first_step_solves=6, step_solves=2)
    run, chem = _run(settings)
    for _ in range(6):
        info = run.step()
    assert run.step_index == 6 and len(chem.calls) == 6
    assert chem.calls[0][1] == DT
    assert np.all(np.diff(run.edges) > 0) and run.edges[0] == 0.0 and run.edges[-1] == L
    assert info["cells"] == run.edges.size - 1 > settings.initial_cells
    assert run.counts["refinements"] > 0 and run.counts["factorisations"] >= 1
    for c in COMPONENTS:
        assert np.all(np.isfinite(info["reacted"][c]))
        assert info["reacted"][c].shape == (N,)
        assert run.history[c].shape == (info["cells"], 2)


def test_uniform_configuration_keeps_mesh_and_one_factorisation():
    """Adaptivity off (as in the Engesgaard uniform driver): fixed mesh, one solve per step."""
    settings = Adap1DSettings(initial_cells=N, first_step_solves=1, step_solves=1,
                              coarsen_tol=-1.0, positivity="mixing")
    run, chem = _run(settings)
    for _ in range(5):
        run.step()
    assert np.array_equal(run.edges, np.linspace(0.0, L, N + 1))
    assert run.counts["factorisations"] == 1 and run.counts["refinements"] == 0
    assert run.counts["coarsenings"] == 0
    assert run.counts["component_solves"] == 5 * len(COMPONENTS)
    # decay only lowers the inlet-driven profile: bounded by the inlet value
    assert np.all(chem.snapshot.concentrations["a"] <= 1.0 + 1e-12)


def test_decay_is_applied_through_the_reaction_increment():
    """With rate 0 the loop is pure transport; with decay the total mass is smaller."""
    s = Adap1DSettings(initial_cells=N, first_step_solves=1, step_solves=1, coarsen_tol=-1.0)
    still, c0 = _run(s, DecayChemistry(N, rate=0.0))
    decay, c1 = _run(s, DecayChemistry(N, rate=2.0))
    for _ in range(4):
        still.step()
        decay.step()
    m0 = c0.snapshot.concentrations["a"].sum()
    m1 = c1.snapshot.concentrations["a"].sum()
    assert 0.0 < m1 < m0

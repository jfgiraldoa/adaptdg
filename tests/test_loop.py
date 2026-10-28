"""Mechanics of the 2D adaptive loop with a trivial fake chemistry (no PHREEQC)."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adaptdg.loop import (
    ChemistryCallbacks2D,
    LoopSettings2D,
    prepare_chemistry_component_2d,
    run_adaptive_loop_2d,
)
from adaptdg.mesh import (
    ReactiveTransportPolicy2D,
    _triangle_restart_permutation,
    persistent_mesh_rebuild_required,
)

COMPONENTS = ("a", "b")
SHAPE = (6, 6)  # (ny, nx)


def test_import_keeps_dolfinx_lazy():
    code = (
        "import sys, adaptdg, adaptdg.loop, adaptdg.mesh;"
        "assert 'dolfinx' not in sys.modules and 'asrtm_phreeqc' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_persistent_mesh_rebuild_schedule():
    flags = [
        persistent_mesh_rebuild_required(k, 3, has_persistent_mesh=k > 0) for k in range(7)
    ]
    assert flags == [True, False, False, True, False, False, True]
    assert persistent_mesh_rebuild_required(5, 3, has_persistent_mesh=False)
    with pytest.raises(ValueError):
        persistent_mesh_rebuild_required(-1, 3, has_persistent_mesh=True)
    with pytest.raises(ValueError):
        persistent_mesh_rebuild_required(0, 0, has_persistent_mesh=True)


def test_restart_permutation_maps_by_geometry():
    tri = np.array(
        [[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]], [[1, 0], [2, 0], [1, 1]]], float
    )
    perm = _triangle_restart_permutation(tri, tri[[2, 0, 1]][:, ::-1])
    assert perm.tolist() == [2, 0, 1]
    with pytest.raises(RuntimeError):
        _triangle_restart_permutation(tri, tri + 0.5)


def test_policy_validation():
    assert not ReactiveTransportPolicy2D().persistent_mesh_enabled
    assert ReactiveTransportPolicy2D(mesh_rebuild_interval=3).persistent_mesh_enabled
    with pytest.raises(ValueError):
        ReactiveTransportPolicy2D(reused_mesh_maximum_solves=5)


def test_prepare_component_positivity_and_floor():
    values = np.array([[1.0, -0.5], [0.0, 2.0]])
    areas = np.full((2, 2), 0.25)
    out, info = prepare_chemistry_component_2d(
        "a", values, areas, negative_concentration_policy="legacy_clip", model_floor=1e-3
    )
    assert np.array_equal(out, [[1.0, 1e-3], [1e-3, 2.0]])
    assert info.corrected_cells == 1 and info.floor_corrected_cells == 2
    assert info.added_integral_mol_L_m2 == pytest.approx(0.125)
    with pytest.raises(RuntimeError):
        prepare_chemistry_component_2d(
            "a", values, areas, negative_concentration_policy="reject", model_floor=0.0
        )


class FakeChemistry:
    """Concentrations decay by 10% per reaction; counts calls."""

    def __init__(self):
        n = SHAPE[0] * SHAPE[1]
        self.state = SimpleNamespace(
            concentrations={name: np.full(n, 1.0e-12) for name in COMPONENTS}
        )
        self.react_calls = 0
        self.seed_calls = 0

    def snapshot(self):
        return self.state

    def react(self, inputs, time_step):
        self.react_calls += 1
        self.state = SimpleNamespace(
            concentrations={n: (0.9 * np.asarray(inputs[n])).reshape(-1) for n in COMPONENTS}
        )
        return self.state

    def seed_fields(self, snapshot, shape):
        self.seed_calls += 1
        return {"extra": snapshot.concentrations["a"].reshape(shape)}

    def callbacks(self):
        return ChemistryCallbacks2D(
            snapshot=self.snapshot,
            react=self.react,
            boundary={"a": 1.0, "b": 0.5},
            seed_fields=self.seed_fields,
            solid_inventory=lambda name, state, shape, areas: 7.0,
        )


def _settings(scheme="bdf2"):
    adaptivity = SimpleNamespace(
        theta=0.3,
        epsilon_chem=0.5,
        eligibility_factor=0.24,
        tie_tolerance=0.35,
        initial_cells_2d=(5, 5),
        maximum_refinements=2,
    )
    return LoopSettings2D(
        lower=(0.0, 0.0),
        upper=(0.5, 0.5),
        chemistry_cells=(SHAPE[1], SHAPE[0]),
        time_step=1500.0,
        steps=8,
        diffusion=7.07107e-10,
        velocity=(1.0e-5, -1.0e-5),
        degree=1,
        adaptivity=adaptivity,
        time_scheme=scheme,
    )


def _run(steps, policy=None, **kw):
    fake = FakeChemistry()
    run = run_adaptive_loop_2d(
        _settings(),
        fake.callbacks(),
        components=COMPONENTS,
        model_floor=1e-20,
        steps=steps,
        transport_policy=policy,
        **kw,
    )
    return fake, run


def test_loop_is_deterministic_and_calls_chemistry_once_per_step():
    pytest.importorskip("dolfinx")
    fake1, run1 = _run(4)
    fake2, run2 = _run(4)
    assert fake1.react_calls == 4 and len(run1.steps) == 4
    assert fake1.seed_calls == 0  # no predictive policy, no seeding
    for s1, s2 in zip(run1.steps, run2.steps):
        assert np.array_equal(s1.triangles, s2.triangles)
        for name in COMPONENTS:
            assert np.array_equal(s1.transported_cell_averages[name], s2.transported_cell_averages[name])
            assert np.array_equal(
                s1.chemistry_output_concentrations[name], s2.chemistry_output_concentrations[name]
            )
        # the fake reaction is 0.9 * (corrected transport output)
        for name in COMPONENTS:
            assert np.array_equal(
                s1.chemistry_output_concentrations[name].reshape(-1),
                0.9 * s1.chemistry_input_concentrations[name].reshape(-1),
            )
    assert run1.steps[0].work_counts["phreeqc_cell_reactions"] == SHAPE[0] * SHAPE[1]
    assert run1.steps[0].mass_balance_diagnostics["a"]["solid_before_reaction_mol_per_kgw_m2"] == 7.0
    assert run1.boundary_concentrations == {"a": 1.0, "b": 0.5}


def test_persistent_mesh_reuse_and_seed_callback():
    pytest.importorskip("dolfinx")
    policy = ReactiveTransportPolicy2D(
        maximum_solves=2, mesh_rebuild_interval=3, reused_mesh_maximum_solves=1
    )
    fake, run = _run(4, policy)
    rebuilt = [s.work_counts["predictive_mesh_rebuilt"] for s in run.steps]
    reused = [s.work_counts["persistent_mesh_reused"] for s in run.steps]
    assert rebuilt == [1, 0, 0, 1] and reused == [0, 1, 1, 0]
    assert fake.seed_calls == 2
    assert run.steps[1].triangles.shape == run.steps[2].triangles.shape


def test_snapshot_selection_and_step_validation():
    pytest.importorskip("dolfinx")
    _, run = _run(3, snapshot_selection={2: [0]})
    assert len(run.steps[0].refinement_history) == 0 and len(run.steps[2].refinement_history) == 0
    # the requested level 0 is kept; the genuine terminal level may be added
    assert [h.refinement_level for h in run.steps[1].refinement_history][0] == 0
    with pytest.raises(ValueError):
        _run(99)
    with pytest.raises(ValueError):
        _run(3, start_step=1)  # replay inputs must match start_step


def test_restart_by_replay_matches_uninterrupted_run():
    pytest.importorskip("dolfinx")
    _, full = _run(4)
    inputs = [s.chemistry_input_concentrations for s in full.steps[:2]]
    _, resumed = _run(4, start_step=2, chemistry_replay_inputs=inputs)
    assert [s.step for s in resumed.steps] == [2, 3]
    for a, b in zip(full.steps[2:], resumed.steps):
        assert np.array_equal(a.triangles, b.triangles)
        for name in COMPONENTS:
            assert np.array_equal(a.transported_cell_averages[name], b.transported_cell_averages[name])
            assert np.array_equal(
                a.chemistry_output_concentrations[name], b.chemistry_output_concentrations[name]
            )

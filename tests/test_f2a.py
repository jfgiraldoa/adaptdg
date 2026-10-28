"""History-preserving mesh rebuild (F2A) and fast path: projection, restriction, loop options."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adaptdg.f2a import F2AHistory, F2AOptions, restrict_to_children
from adaptdg.history_projection import build_plan, canonical_vertices, scale_nonnegative
from adaptdg.history_projection_fast import build_plan as build_plan_fast
from adaptdg.history_projection_fast import project_stack
from adaptdg.loop import ChemistryCallbacks2D, LoopSettings2D, run_adaptive_loop_2d
from adaptdg.mesh import ReactiveTransportPolicy2D

COMPONENTS = ("a", "b")
SHAPE = (6, 6)


def grid_triangles(n, nested=False):
    """Triangulation of the unit square: n x n squares cut by a diagonal; ``nested`` splits every
    triangle of the n x n grid into four (midpoint refinement)."""
    out = []
    h = 1.0 / n
    for j in range(n):
        for i in range(n):
            p = np.array([[i * h, j * h], [(i + 1) * h, j * h], [(i + 1) * h, (j + 1) * h], [i * h, (j + 1) * h]])
            out += [p[[0, 1, 2]], p[[0, 2, 3]]]
    out = np.array(out)
    if nested:
        kids = []
        for t in out:
            m = [(t[0] + t[1]) / 2, (t[1] + t[2]) / 2, (t[2] + t[0]) / 2]
            kids += [[t[0], m[0], m[2]], [m[0], t[1], m[1]], [m[2], m[1], t[2]], [m[0], m[1], m[2]]]
        out = np.array(kids)
    return out


def areas(c):
    return 0.5 * np.abs((c[:, 1, 0] - c[:, 0, 0]) * (c[:, 2, 1] - c[:, 0, 1])
                        - (c[:, 2, 0] - c[:, 0, 0]) * (c[:, 1, 1] - c[:, 0, 1]))


def discontinuous_dg1(canonical, seed=0):
    """Random linear field on every triangle (discontinuous), evaluated at its canonical vertices."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(size=(canonical.shape[0], 3))
    return coef[:, :1] + coef[:, 1:2] * canonical[..., 0] + coef[:, 2:3] * canonical[..., 1]


def test_import_keeps_dolfinx_lazy():
    code = (
        "import sys, adaptdg.f2a, adaptdg.fastadap, adaptdg.history_projection,"
        "adaptdg.history_projection_fast, adaptdg.loop;"
        "assert 'dolfinx' not in sys.modules and 'asrtm_phreeqc' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("pair", [(4, 3), (3, 4), (3, 6)])
def test_projection_is_conservative_per_new_triangle_and_exact_for_linear_fields(pair):
    old, new = grid_triangles(pair[0]), grid_triangles(pair[1])
    plan = build_plan(old, new)
    oc, nc = canonical_vertices(old), canonical_vertices(new)
    assert plan.coverage_error < 1e-12
    # a globally linear field is reproduced exactly
    linear = lambda c: 2.0 + 3.0 * c[..., 0] - 1.5 * c[..., 1]  # noqa: E731
    np.testing.assert_allclose(plan.apply(linear(oc)), linear(nc), atol=1e-13)
    # a discontinuous field: global inventory is conserved
    values = discontinuous_dg1(oc)
    projected = plan.apply(values)
    before = float(np.sum(values.mean(axis=1) * areas(oc)))
    after = float(np.sum(projected.mean(axis=1) * areas(nc)))
    assert after == pytest.approx(before, rel=1e-12)


def test_projection_onto_nested_children_is_the_exact_restriction():
    old, new = grid_triangles(3), grid_triangles(3, nested=True)
    oc, nc = canonical_vertices(old), canonical_vertices(new)
    values = discontinuous_dg1(oc)
    projected = build_plan(old, new).apply(values)
    # each child lies in one parent: the parent's linear field evaluated at the child vertices
    parent = np.repeat(np.arange(old.shape[0]), 4)
    restricted = restrict_to_children(values, old, new, parent)
    np.testing.assert_allclose(projected, restricted, atol=1e-12)
    assert restricted.shape == nc.shape[:2]
    # the children reproduce the parent inventory
    parent_inventory = values.mean(axis=1) * areas(oc)
    child_inventory = np.bincount(parent, weights=restricted.mean(axis=1) * areas(nc))
    np.testing.assert_allclose(child_inventory, parent_inventory, rtol=1e-12)


def test_fast_projection_matches_plain_projection():
    old, new = grid_triangles(5), grid_triangles(4)
    oc = canonical_vertices(old)
    values = np.stack([discontinuous_dg1(oc, seed) for seed in range(3)])
    plain = build_plan(old, new).apply(values)
    fast_plan = build_plan_fast(old, new)
    assert fast_plan.coverage_error < 1e-12
    np.testing.assert_allclose(fast_plan.apply(values), plain, atol=1e-12)
    stacked, _, coverage, identical = project_stack(old, new, values)
    np.testing.assert_allclose(stacked, plain, atol=1e-12)
    assert coverage < 1e-12 and identical == 0
    # identical triangles are copied: projecting a mesh onto itself is the identity
    same, _, _, identical = project_stack(old, old, values)
    assert identical == old.shape[0]
    np.testing.assert_array_equal(same, values)


def test_nonnegative_scaling_keeps_the_mean():
    values = np.array([[-0.2, 0.5, 1.0], [0.1, 0.2, 0.3], [-1.0, -0.5, -0.2]])
    scaled, count = scale_nonnegative(values)
    assert count == 1  # only the first triangle has a negative vertex and a nonnegative mean
    np.testing.assert_allclose(scaled.mean(axis=1)[:2], values.mean(axis=1)[:2], atol=1e-15)
    assert scaled[0].min() >= 0.0
    np.testing.assert_array_equal(scaled[1:], values[1:])  # untouched (nonnegative / negative mean)


def test_f2a_history_projection_and_restriction_on_dolfinx_meshes():
    pytest.importorskip("dolfinx")
    from adaptdg.transport_2d import create_rectangle_mesh_2d, refine_marked_triangles_with_parent

    coarse = create_rectangle_mesh_2d(lower=(0.0, 0.0), upper=(1.0, 1.0), cells=(3, 3))
    other = create_rectangle_mesh_2d(lower=(0.0, 0.0), upper=(1.0, 1.0), cells=(4, 4))
    linear = lambda c: 1.0 + 2.0 * c[..., 0] - 0.5 * c[..., 1]  # noqa: E731
    for optimised in (False, True):
        f2a = F2AHistory(COMPONENTS, optimised=optimised)
        cc = canonical_vertices(f2a.dof_triangles(coarse))
        current = {"a": linear(cc), "b": 2.0 * linear(cc)}
        older = {"a": 0.5 * linear(cc), "b": 0.0 * linear(cc) + 1.0}
        # reused mesh: the histories pass through
        assert f2a.histories(False, current, older, coarse, coarse) == (current, older)
        # nothing to project from
        assert f2a.histories(True, None, None, None, other) == (None, None)
        new_current, new_older = f2a.histories(True, current, older, coarse, other)
        nc = canonical_vertices(f2a.dof_triangles(other))
        np.testing.assert_allclose(new_current["a"], linear(nc), atol=1e-12)
        np.testing.assert_allclose(new_current["b"], 2.0 * linear(nc), atol=1e-12)
        np.testing.assert_allclose(new_older["a"], 0.5 * linear(nc), atol=1e-12)
        assert f2a.stats["rebuild_projections"] == 1 and f2a.stats["max_rebuild_inventory_rel"] < 1e-12
        # refinement: children carry the restriction of the parent field
        refined, parent = refine_marked_triangles_with_parent(coarse, np.arange(18) % 3 == 0)
        child_current, child_older, averages, older_averages = f2a.restrict(
            current, older, coarse, refined, parent
        )
        rc = canonical_vertices(f2a.dof_triangles(refined))
        np.testing.assert_allclose(child_current["a"], linear(rc), atol=1e-12)
        np.testing.assert_allclose(child_older["a"], 0.5 * linear(rc), atol=1e-12)
        np.testing.assert_allclose(averages[0], child_current["a"].mean(axis=1))
        assert older_averages.shape == averages.shape
        assert f2a.stats["max_refinement_inventory_rel"] < 1e-12


# ---------------------------------------------------------------- loop options (fake chemistry)
class FakeChemistry:
    def __init__(self):
        n = SHAPE[0] * SHAPE[1]
        self.state = SimpleNamespace(concentrations={k: np.full(n, 1.0e-12) for k in COMPONENTS})

    def snapshot(self):
        return self.state

    def react(self, inputs, time_step):
        self.state = SimpleNamespace(
            concentrations={k: (0.9 * np.asarray(inputs[k])).reshape(-1) for k in COMPONENTS}
        )
        return self.state

    def callbacks(self):
        return ChemistryCallbacks2D(
            snapshot=self.snapshot, react=self.react, boundary={"a": 1.0, "b": 0.5},
            seed_fields=lambda snapshot, shape: {"extra": snapshot.concentrations["a"].reshape(shape)},
        )


def _run(steps, **kw):
    adaptivity = SimpleNamespace(theta=0.3, epsilon_chem=0.5, eligibility_factor=0.24,
                                 tie_tolerance=0.35, initial_cells_2d=(5, 5), maximum_refinements=2)
    settings = LoopSettings2D(
        lower=(0.0, 0.0), upper=(0.5, 0.5), chemistry_cells=(SHAPE[1], SHAPE[0]), time_step=1500.0,
        steps=8, diffusion=7.07107e-10, velocity=(1.0e-5, -1.0e-5), degree=1, adaptivity=adaptivity,
        time_scheme="bdf2")
    policy = ReactiveTransportPolicy2D(maximum_solves=2, mesh_rebuild_interval=2, reused_mesh_maximum_solves=1)
    return run_adaptive_loop_2d(
        settings, FakeChemistry().callbacks(), components=COMPONENTS, model_floor=1e-20, steps=steps,
        transport_policy=policy, reaction_increment_history=True, maximum_refinement_solves=2, **kw)


def test_history_policy_validation():
    with pytest.raises(ValueError):
        _run(1, history_policy="other")
    with pytest.raises(ValueError):
        _run(1, f2a_options=F2AOptions())  # options need history_policy="f2a"


def test_f2a_and_fast_loop_options():
    pytest.importorskip("dolfinx")
    plain = _run(5)
    f2a = _run(5, history_policy="f2a")
    both = _run(5, history_policy="f2a", fast=True)
    assert plain.statistics is None
    assert f2a.statistics["f2a"]["rebuild_projections"] == 2  # rebuilds at steps 3 and 5
    assert both.statistics["fast"]["solvers_reused"] > 0 and both.statistics["f2a"]["rebuild_projections"] == 2
    # the fast path changes round-off only
    for s1, s2 in zip(f2a.steps, both.steps):
        assert np.array_equal(s1.triangles, s2.triangles)
        for name in COMPONENTS:
            np.testing.assert_allclose(s1.transported_cell_averages[name],
                                       s2.transported_cell_averages[name], rtol=1e-8, atol=1e-18)
    # the first rebuilt block (steps 1-2) is untouched by F2A: identical to the plain policy
    for s1, s2 in zip(plain.steps[:2], f2a.steps[:2]):
        for name in COMPONENTS:
            assert np.array_equal(s1.transported_cell_averages[name], s2.transported_cell_averages[name])
    # the DG1 history of the plain policy is rebuilt from averages at step 3, F2A projects it
    assert not np.array_equal(plain.steps[2].reaction_history_cell_values["a"],
                              f2a.steps[2].reaction_history_cell_values["a"])


@pytest.mark.parametrize("start", [3, 4])  # interval 2: 3 is mid-block, 4 a rebuild boundary
def test_f2a_restart_matches_uninterrupted_run(start):
    pytest.importorskip("dolfinx")
    full = _run(6, history_policy="f2a")
    done = full.steps[start - 1]
    inputs = [s.chemistry_input_concentrations for s in full.steps[:start]]
    resumed = _run(
        6, history_policy="f2a", start_step=start, chemistry_replay_inputs=inputs,
        restart_transport_triangles=done.triangles,
        restart_reaction_history=done.reaction_history_cell_values,
        restart_older_reaction_history=done.older_reaction_history_cell_values,
    )
    for a, b in zip(full.steps[start:], resumed.steps):
        assert a.triangles.shape == b.triangles.shape
        for name in COMPONENTS:
            np.testing.assert_allclose(a.transported_cell_averages[name],
                                       b.transported_cell_averages[name], rtol=1e-9, atol=1e-18)

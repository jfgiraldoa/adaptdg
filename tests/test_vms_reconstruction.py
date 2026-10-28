"""VMS reconstruction: fine scale, adjoint multiscale reconstruction and the dG baseline."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings  # noqa: E402
from adaptdg.vms.norms import error_norms, error_norms_vms  # noqa: E402
from adaptdg.vms.problems import ReentrantCorner  # noqa: E402
from adaptdg.vms.reconstruction import reconstruct  # noqa: E402
from adaptdg.vms.saddle import RMSolver  # noqa: E402


def _solve(p, velocity=()):
    problem = ReentrantCorner(velocity=velocity)
    return problem, RMSolver(problem, p).solve(problem.initial_mesh(4))


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("velocity", [(), (1.0, 0.5)])
def test_dg_solution_equals_ubar_plus_utilde(p, velocity):
    """Paper Remark 1: the partitioned full-scale solution is the classical dG solution."""
    _, res = _solve(p, velocity)
    vms = reconstruct(res)
    theta = vms.theta.x.array
    assert np.abs(vms.u_full.x.array - theta).max() < 1e-10 * np.abs(theta).max()
    ubar = vms.u_full.x.array - vms.utilde.x.array  # ubar interpolated into DG(p)
    assert np.abs(ubar).max() > 0 and np.abs(vms.utilde.x.array).max() > 0
    assert vms.phi.function_space.dofmap.index_map.size_global == theta.size  # all in DG(p)
    assert vms.ubar.function_space.dofmap.index_map.size_global < theta.size  # ubar in CG(p)


def test_adjoint_ordering_matters_only_with_advection():
    _, res = _solve(1)  # pure diffusion: b is symmetric
    a, b = reconstruct(res, adjoint_ordering="paper"), reconstruct(res, adjoint_ordering="legacy")
    scale = np.abs(a.ucheck.x.array).max()
    assert np.abs(a.ucheck.x.array - b.ucheck.x.array).max() < 1e-10 * scale
    _, res = _solve(1, (1.0, 0.5))
    a, b = reconstruct(res, adjoint_ordering="paper"), reconstruct(res, adjoint_ordering="legacy")
    assert np.abs(a.ucheck.x.array - b.ucheck.x.array).max() > 1e-4 * np.abs(a.ucheck.x.array).max()
    assert np.array_equal(a.theta.x.array, b.theta.x.array)  # same baseline
    with pytest.raises(ValueError):
        reconstruct(res, adjoint_ordering="other")


def test_error_norms_vms_are_consistent():
    problem, res = _solve(1)
    vms = reconstruct(res)
    errs = error_norms_vms(res, vms, problem, 1)
    assert set(errs) == {"ubar", "theta", "u", "phi"}
    assert errs["ubar"] == pytest.approx(error_norms(res, problem, 1))
    assert errs["u"] == pytest.approx(errs["theta"], rel=1e-8)


def test_phi_improves_l2_and_keeps_vh_close_to_dg():
    """Measured on the re-entrant corner (p = 1, uniform): phi/theta is ~0.5 in L2 and ~1.2 in Vh."""
    s = LoopSettings(p=1, refinement="uniform", max_levels=4, reconstruct=True)
    hist = AdaptiveStationaryLoop(ReentrantCorner(), s).run(progress=None)
    last = hist[-1]
    assert last["L2_phi"] < 0.8 * last["L2_theta"]
    assert last["Vh_phi"] < 1.5 * last["Vh_theta"]
    assert last["L2_u"] == pytest.approx(last["L2_theta"], rel=1e-8)
    assert last["Vh_u"] == pytest.approx(last["Vh_theta"], rel=1e-8)


def test_history_columns_depend_on_reconstruct_flag():
    off = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(max_levels=2, error_targets="exact")).run(progress=None)
    assert list(off[0]) == ["level", "ndofs", "ncells", "E", "L2", "Vh", "time"]
    default = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(max_levels=1)).run(progress=None)
    assert list(default[0]) == ["level", "ndofs", "ncells", "E", "L2", "Vh", "L2_proj", "Vh_proj", "time"]
    on = AdaptiveStationaryLoop(ReentrantCorner(), LoopSettings(max_levels=2, reconstruct=True, error_targets="exact"))
    rows = on.run(progress=None)
    extra = ["L2_theta", "Vh_theta", "L2_u", "Vh_u", "L2_phi", "Vh_phi"]
    assert list(rows[0]) == ["level", "ndofs", "ncells", "E", "L2", "Vh", *extra, "time"]
    for a, b in zip(off, rows):  # the pieces 1-3 numbers do not change
        assert all(a[k] == pytest.approx(b[k]) for k in ("ndofs", "ncells", "E", "L2", "Vh"))
    assert on.vms is not None

"""Adaptive loop on the re-entrant corner (2D) and a 3D smoke test."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings, fitted_slope  # noqa: E402
from adaptdg.vms.problems import Fichera3D, ReentrantCorner  # noqa: E402
from adaptdg.vms.run import run_from_json  # noqa: E402

DOFMAX = {1: 8000, 2: 8000}


def _run(p, **kw):
    s = LoopSettings(p=p, max_levels=60, dofmax=DOFMAX[p], **kw)
    loop = AdaptiveStationaryLoop(ReentrantCorner(), s)
    return loop, loop.run(progress=None)


@pytest.mark.parametrize("p", [1, 2])
def test_reentrant_corner_rates_and_clustering(p):
    loop, hist = _run(p)
    ndofs = [h["ndofs"] for h in hist]
    assert ndofs == sorted(ndofs) and ndofs[-1] > 5 * ndofs[0]
    assert hist[-1]["E"] < 0.1 * hist[0]["E"]
    assert fitted_slope(hist, "E", 2) == pytest.approx(-p, abs=0.4)
    assert fitted_slope(hist, "Vh", 2) == pytest.approx(-p, abs=0.35)
    # L2 is pre-asymptotically a little faster than p+1 on the singular problem: one-sided check
    l2 = fitted_slope(hist, "L2", 2)
    assert -(p + 1) - 1.0 < l2 < -(p + 1) + 0.45
    # the mesh is refined towards the corner at the origin
    dom = loop.domain
    xyz = dom.geometry.x[dom.geometry.dofmaps[0]]  # (ncells, 3 vertices, 3)
    mid = xyz.mean(axis=1)
    e1, e2 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    h = np.sqrt(0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]))  # sqrt(area)
    near = np.hypot(mid[:, 0], mid[:, 1]) < 0.2
    far = np.median(h[~near])
    assert near.any() and np.median(h[near]) < 0.5 * far  # measured 0.35 (p=1), < 0.25 (p=2)
    assert h[near].min() < 0.1 * far


def test_adaptive_beats_uniform_for_the_singular_solution():
    uni = AdaptiveStationaryLoop(
        ReentrantCorner(), LoopSettings(p=1, refinement="uniform", max_levels=5)
    ).run(progress=None)
    slope = fitted_slope(uni, "Vh", 2, span=1e9)
    assert slope > -0.8  # limited by the singularity (about -2/3)
    _, ada = _run(1)
    assert fitted_slope(ada, "Vh", 2) < slope - 0.2


def test_fichera_3d_smoke():
    s = LoopSettings(p=1, max_levels=3, initial_mesh_size=2)
    hist = AdaptiveStationaryLoop(Fichera3D(q=1 / 3), s).run(progress=None)
    assert [h["ndofs"] for h in hist] == sorted(h["ndofs"] for h in hist)
    assert hist[-1]["ndofs"] > hist[0]["ndofs"]
    assert all(np.isfinite(h["E"]) and h["E"] > 0 for h in hist)
    assert hist[-1]["E"] < hist[0]["E"]


def test_run_from_json_writes_csv_and_mesh(tmp_path):
    out = run_from_json({"name": "reentrant_corner"}, {"p": 1, "max_levels": 5}, tmp_path,
                        progress=None)
    assert out["csv"].exists() and out["mesh"].exists()
    header = out["csv"].read_text().splitlines()[0]
    assert header == "level,ndofs,ncells,E,L2,Vh,time"
    assert len(out["history"]) == 5 and "E" in out["slopes"]

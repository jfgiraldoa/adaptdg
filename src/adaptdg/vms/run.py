"""JSON loader: ``run_from_json(problem_json, numerics_json)`` -> history CSV + result dict.

problem.json keys (defaults in brackets): name ["reentrant_corner", "fichera", "heterogeneous",
"anisotropic"], kappa [1.0], beta [none; (1, 0) for heterogeneous], alpha [2/3, exponent of the exact
solution; ``q`` for fichera, also accepted under the key ``q``], eps1/eps2 [0.1/1.0, heterogeneous], r/tau [1e4/1e-3, anisotropic],
initial_mesh_size [4, even number of cells per side]. numerics.json keys: see :class:`~adaptdg.vms.loop.LoopSettings` (``eta_0`` is accepted for ``eta0``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings, fitted_slope
from adaptdg.vms.problems import PROBLEMS, Problem


def _load(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        return dict(spec)
    return json.loads(Path(spec).read_text())


def build_problem(spec: dict[str, Any]) -> Problem:
    """Instantiate the problem named in ``spec`` (unknown names raise ``KeyError``)."""
    cls = PROBLEMS[spec.get("name", "reentrant_corner")]
    names = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    if "kappa" in spec and "diffusivity" in names:
        kwargs["diffusivity"] = float(spec["kappa"])
    if "alpha" in spec:
        kwargs["alpha" if "alpha" in names else "q"] = float(spec["alpha"])
    if "q" in spec and "q" in names:  # fichera: ``q`` is accepted as well as ``alpha``
        kwargs["q"] = float(spec["q"])
    if spec.get("beta") and "velocity" in names:
        kwargs["velocity"] = tuple(float(v) for v in spec["beta"])
    for key in ("eps1", "eps2", "r", "tau"):  # Heterogeneous / Anisotropic parameters
        if key in spec and key in names:
            kwargs[key] = float(spec[key])
    return cls(**kwargs)


def build_settings(spec: dict[str, Any], problem_spec: dict[str, Any] | None = None) -> LoopSettings:
    """``LoopSettings`` from numerics.json plus ``initial_mesh_size`` of problem.json."""
    names = {f.name for f in fields(LoopSettings)}
    unknown = set(spec) - names - {"eta_0"}
    if unknown:
        raise KeyError(f"Unknown numerics keys: {sorted(unknown)}")
    merged = dict(spec)
    if "eta_0" in merged:
        merged["eta0"] = merged.pop("eta_0")
    if problem_spec and "initial_mesh_size" in problem_spec:
        merged.setdefault("initial_mesh_size", int(problem_spec["initial_mesh_size"]))
    return LoopSettings(**merged)


def run_from_json(problem_json: Any, numerics_json: Any, out_dir: Any = None,
                  progress: Any = print, **overrides: Any) -> dict[str, Any]:
    """Run one adaptive/uniform loop; ``overrides`` replace numerics keys (e.g. ``p=2``).

    Returns ``{"history", "settings", "slopes", "csv", "mesh"}``; when ``out_dir`` is given the
    history is written to ``history_p<p>_<refinement>.csv`` and the final mesh to ``mesh_*.npz``.
    """
    pspec, nspec = _load(problem_json), _load(numerics_json)
    nspec.update(overrides)
    problem, settings = build_problem(pspec), build_settings(nspec, pspec)
    loop = AdaptiveStationaryLoop(problem, settings)
    history = loop.run(progress)
    d = problem.dim
    keys = [k for k in history[0] if k not in ("level", "ndofs", "ncells", "time")]
    slopes = {k: fitted_slope(history, k, d) for k in keys} if len(history) > 3 else {}
    out: dict[str, Any] = {"history": history, "settings": asdict(settings), "slopes": slopes,
                           "csv": None, "mesh": None}
    if out_dir is not None:
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"p{settings.p}_{settings.refinement}"
        out["csv"] = folder / f"history_{stem}.csv"
        with open(out["csv"], "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        out["mesh"] = folder / f"mesh_{stem}.npz"
        import numpy as np

        geom = loop.domain.geometry
        np.savez(out["mesh"], x=geom.x, cells=geom.dofmaps[0])
    return out

"""Fichera corner (3D Poisson, paper section 5.3): adaptive versus uniform residual minimisation.

Usage: python run.py [--full] [--preset paper|legacy] [--degrees 1 2 ...] [--dofmax N]
The short run (p = 1, q = 1/3, adaptive, about 40000 DoF, well under a minute) is the default;
--full runs q in {0.1, 1/3} x p in {1, 2} x {adaptive, uniform} up to the caps of numerics.json
(about 4 min per preset, up to about 3 GB). Results go to ./out/<preset>/: one CSV per run
(history_q<q>_p<p>_<refinement>.csv, both error targets), cells_*.npz (cell centroid distance to
the origin and cell size of the last mesh) and summary.json (slopes over the last 2x window of DoF,
time, peak memory and clustering statistics).
"""

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from adaptdg.vms.loop import AdaptiveStationaryLoop, fitted_slope
from adaptdg.vms.run import build_problem, build_settings

HERE = Path(__file__).parent
SLOPE_SPAN = 2.0  # slopes use the levels whose DoF is within a factor 2 of the last one
BASES = ("L2", "Vh")
FIELDS = ("", "_theta", "_u", "_phi")  # ubar, theta, u = ubar + utilde, phi
SUFFIXES = ("", "_proj")  # error against the exact solution / its DG(p) L2 projection


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def cell_sizes(domain) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each cell midpoint to the origin and cell size ``h = volume^(1/3)``."""
    from dolfinx import mesh

    n = domain.topology.index_map(3).size_local
    mid = mesh.compute_midpoints(domain, 3, np.arange(n, dtype=np.int32))
    xyz = domain.geometry.x[domain.geometry.dofmaps[0][:n]]
    vol = np.abs(np.linalg.det(xyz[:, 1:, :] - xyz[:, :1, :])) / 6.0
    return np.linalg.norm(mid, axis=1), vol ** (1.0 / 3.0)


def clustering(r: np.ndarray, h: np.ndarray) -> dict[str, float]:
    """Cell-size statistics of the last mesh: near the corner (r < 0.1) versus far (r > 0.7)."""
    near, far = r < 0.1, r > 0.7
    return {"h_min": float(h.min()), "h_max": float(h.max()),
            "h_near": float(h[near].mean()) if near.any() else float("nan"),
            "h_far": float(h[far].mean()) if far.any() else float("nan"),
            "h_far_over_near": float(h[far].mean() / h[near].mean()) if near.any() and far.any() else float("nan"),
            "frac_cells_near": float(near.mean())}


def run_one(pspec: dict, nspec: dict, step: int = 2, progress=None):
    """One adaptive loop, or (``refinement == "uniform"``) the structured meshes n0, n0 + step, ...
    solved one after the other and stopped at ``dofmax`` (more points for the slope than halving h)."""
    problem = build_problem(pspec)
    settings = build_settings(nspec, pspec)
    if settings.refinement == "adaptive":
        loop = AdaptiveStationaryLoop(problem, settings)
        return loop.run(progress), loop.domain
    history, domain = [], None
    for level in range(settings.max_levels):
        domain = problem.initial_mesh(settings.initial_mesh_size + step * level)
        row = AdaptiveStationaryLoop(problem, build_settings({**nspec, "max_levels": 1}, pspec),
                                     domain=domain).run(progress)[0]
        history.append({**row, "level": level})
        if row["ndofs"] >= settings.dofmax:
            break
    return history, domain


def summarise(history: list[dict]) -> dict:
    keys = [f"{b}{f}{s}" for s in SUFFIXES for b in BASES for f in FIELDS if f"{b}{f}{s}" in history[0]]
    return {k: fitted_slope(history, k, 3, span=SLOPE_SPAN) for k in keys}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="q in {0.1, 1/3} x p in {1, 2} x {adaptive, uniform}")
    ap.add_argument("--preset", choices=("paper", "legacy"), default="paper")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default 1; 1 2 with --full)")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap of every run")
    args = ap.parse_args(argv)

    pspec = json.loads((HERE / "problem.json").read_text())
    nspec = json.loads((HERE / "numerics.json").read_text())
    extra = nspec.pop("example")  # keys of this example that are not LoopSettings fields
    nspec.update(extra["presets"][args.preset], preset=args.preset)
    qs = pspec["q_full"] if args.full else [pspec["q"]]
    degrees = args.degrees or ([1, 2] if args.full else [1])
    refinements = ("adaptive", "uniform") if args.full else ("adaptive",)
    jobs = [(q, p, ref) for q in qs for p in degrees for ref in refinements]
    out = HERE / "out" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for i, (q, p, ref) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: preset={args.preset} q={q:.3f} p={p} {ref}  (t={time.time() - t0:.0f}s)", flush=True)
        dofmax = args.dofmax or (extra["dofmax_full"][str(p)] if args.full else nspec["dofmax"])
        t1 = time.time()
        history, domain = run_one({**pspec, "q": q}, {**nspec, "p": p, "refinement": ref, "dofmax": dofmax},
                                  extra["uniform_n_step"])
        elapsed = time.time() - t1
        tag = f"q{q:.2f}_p{p}_{ref}"
        with open(out / f"history_{tag}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        r, h = cell_sizes(domain)
        np.savez_compressed(out / f"cells_{tag}.npz", r=r.astype(np.float32), h=h.astype(np.float32))
        last = history[-1]
        slopes = summarise(history) if len(history) > 3 else {}
        summary.append({"preset": args.preset, "q": q, "p": p, "refinement": ref, "levels": len(history),
                        "ndofs": last["ndofs"], "E": last["E"], "time_s": round(elapsed, 1),
                        "peak_rss_mb": round(peak_rss_mb()), "slope_span": SLOPE_SPAN,
                        "slopes": {k: round(v, 3) for k, v in slopes.items()}, "clustering": clustering(r, h),
                        "last_row": last})
        print(f"  ndofs={last['ndofs']} levels={len(history)} E={last['E']:.2e} L2={last['L2']:.2e} "
              f"Vh={last['Vh']:.2e} time={elapsed:.0f}s peak={peak_rss_mb():.0f}MB", flush=True)
        print("  slopes vs DoF^(1/3), last 2x window: "
              + " ".join(f"{k}={v:.2f}" for k, v in summary[-1]["slopes"].items() if k in ("L2", "Vh", "Vh_proj", "L2_proj")),
              flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main()

"""Single-component Burgers equation (paper section 6.5.2): shock layer at x = 1/2, adaptive refinement.

Usage: python run.py [--full] [--preset paper|legacy] [--degrees 3 ...] [--kappa K ...] [--dofmax N]
The short run (p = 3, kappa in {1e-2, 1e-3}, about 5000 DoF, under a minute) is the default; --full runs kappa in
{1e-2, 1e-3, 1e-4} up to 20000 DoF (overnight batch). Results go to ./out/<preset>/: one CSV per run
(history_k<kappa>_p<p>.csv: level, ndofs, E, Newton iterations / residual / status, min and max of ubar, time),
mesh_*.npz (last mesh), field_*.npz (nodal values of the last ubar and its profiles along y = 0.25, 0.75) and summary.json (Newton statistics, shock position,
steepest slope and width on the line y = 0.75, cell-size clustering, time, peak memory). There is no exact solution:
the error-target columns do not exist.
"""

import argparse
import csv
import json
import resource
import sys
import time
from math import factorial
from pathlib import Path

import numpy as np

from adaptdg.vms.loop import fitted_slope
from adaptdg.vms.nonlinear import NonlinearAdaptiveLoop, line_values, shock_diagnostics
from adaptdg.vms.run import build_problem, build_settings

HERE = Path(__file__).parent


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def shock_clustering(domain) -> dict[str, float]:
    """Mean cell size (volume^(1/2)) within 0.05 of the shock line x = 1/2 versus farther than 0.25."""
    n = domain.topology.index_map(2).size_local
    xyz = domain.geometry.x[domain.geometry.dofmaps[0][:n]]
    h = np.abs(np.linalg.det(xyz[:, 1:, :2] - xyz[:, :1, :2])) / factorial(2)
    dist = np.abs(xyz[:, :, 0].mean(axis=1) - 0.5)
    near, far = dist < 0.05, dist > 0.25
    return {"h_near": float(np.sqrt(h[near]).mean()), "h_far": float(np.sqrt(h[far]).mean()),
            "h_far_over_near": float(np.sqrt(h[far]).mean() / np.sqrt(h[near]).mean())}


def newton_summary(history: list[dict]) -> dict:
    ok = [bool(r["newton_converged"]) for r in history]
    its = [int(r["newton_iterations"]) for r in history]
    return {"levels": len(ok), "converged": int(sum(ok)), "failed_levels": [i + 1 for i, c in enumerate(ok) if not c],
            "iterations_min": min(its), "iterations_max": max(its), "iterations_median": float(np.median(its)),
            "fallback_levels": [i + 1 for i, r in enumerate(history) if r["newton_fallback"]],
            "statuses": sorted({r["newton_status"] for r in history})}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="kappa in {1e-2, 1e-3, 1e-4}, larger DoF cap")
    ap.add_argument("--preset", choices=("paper", "legacy"), default="paper")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default: numerics.json)")
    ap.add_argument("--kappa", type=float, nargs="+", help="diffusivities (default: numerics.json)")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap of every run")
    args = ap.parse_args(argv)

    pspec = json.loads((HERE / "problem.json").read_text())
    nspec = json.loads((HERE / "numerics.json").read_text())
    extra = nspec.pop("example")  # keys of this example that are not LoopSettings fields
    mode = "full" if args.full else "short"
    nspec.update(extra["presets"][args.preset], preset=args.preset)
    kappas = args.kappa or extra["kappa"][mode]
    degrees = args.degrees or extra["degrees"][mode]
    jobs = [(k, p) for k in kappas for p in degrees]
    out = HERE / "out" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for i, (kappa, p) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: preset={args.preset} kappa={kappa:g} p={p}  (t={time.time() - t0:.0f}s)", flush=True)
        t1 = time.time()
        settings = build_settings({**nspec, "p": p, "dofmax": args.dofmax or extra["dofmax"][mode][str(p)]}, pspec)
        loop = NonlinearAdaptiveLoop(build_problem({**pspec, "kappa": kappa}), settings)
        history = loop.run(print)
        elapsed = time.time() - t1
        tag = f"k{kappa:g}_p{p}"
        with open(out / f"history_{tag}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        geom = loop.domain.geometry
        np.savez_compressed(out / f"mesh_{tag}.npz", x=geom.x, cells=geom.dofmaps[0])
        ub = loop.result.ubar
        xs = np.linspace(0.0, 1.0, 2001)
        np.savez_compressed(out / f"field_{tag}.npz", x=ub.function_space.tabulate_dof_coordinates()[:, :2],
                            u=ub.x.array, line_x=xs, line_y25=line_values(ub, xs, 0.25),
                            line_y75=line_values(ub, xs, 0.75))
        last = history[-1]
        shock = shock_diagnostics(ub, y=0.75)
        slope = fitted_slope(history, "E", 2, span=2.0) if len(history) > 3 else float("nan")
        summary.append({"preset": args.preset, "kappa": kappa, "p": p, "levels": len(history), "ndofs": last["ndofs"],
                        "E": last["E"], "E_slope_last_2x": slope, "time_s": round(elapsed, 1),
                        "peak_rss_mb": round(peak_rss_mb()), "newton": newton_summary(history),
                        "shock_y0.75": shock, "analytic_max_slope": -1.0 / (2.0 * kappa),
                        "clustering": shock_clustering(loop.domain), "last_row": last})
        nw = summary[-1]["newton"]
        print(f"  ndofs={last['ndofs']} levels={len(history)} E={last['E']:.2e} newton {nw['converged']}/{nw['levels']} "
              f"converged (iterations {nw['iterations_min']}-{nw['iterations_max']})  shock x={shock['position']:.3f} "
              f"slope={shock['max_slope']:.1f} (analytic {-1 / (2 * kappa):.1f}) fwhm={shock['width_fwhm']:.4f}  "
              f"time={elapsed:.0f}s peak={peak_rss_mb():.0f}MB", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main()

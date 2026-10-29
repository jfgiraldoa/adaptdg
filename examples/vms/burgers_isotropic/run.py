"""Isotropic Burgers equation (paper section 6.5.1): manufactured tanh layer, adaptive refinement, both error targets.

Usage: python run.py [--full] [--preset paper|legacy] [--degrees 1 2 ...] [--kappa K] [--dofmax N]
The short run (kappa = 1e-3, p = 1, 2, about 10000 / 12000 DoF, under a minute) is the default; --full runs p = 1, 2, 3
up to the caps of numerics.json (overnight batch). Results go to ./out/<preset>/: one CSV per run
(history_k<kappa>_p<p>.csv: level, ndofs, ncells, E, L2/Vh of ubar, theta, u, phi against the exact solution and against its
DG(p) L2 projection (suffix _proj), Newton columns, min/max of ubar, time), mesh_*.npz (last mesh), field_*.npz (nodal
ubar) and summary.json (slopes over the last 2x window of DoF, Newton statistics, ratio Vh(phi)/Vh(theta), time, memory).
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
from adaptdg.vms.nonlinear import NonlinearAdaptiveLoop
from adaptdg.vms.run import build_problem, build_settings

HERE = Path(__file__).parent
SLOPE_SPAN = 2.0  # slopes use the levels whose DoF is within a factor 2 of the last one
BASES = ("L2", "Vh")
FIELDS = ("", "_theta", "_u", "_phi")  # ubar, theta (nonlinear dG), u = ubar + utilde, phi
SUFFIXES = ("", "_proj")  # error against the exact solution / its DG(p) L2 projection


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def layer_clustering(domain, kappa: float) -> dict[str, float]:
    """Mean cell size (volume^(1/2)) within 0.02 of the layer ``2x - y = 1/4`` versus farther than 0.2."""
    n = domain.topology.index_map(2).size_local
    xyz = domain.geometry.x[domain.geometry.dofmaps[0][:n]]
    h = np.sqrt(np.abs(np.linalg.det(xyz[:, 1:, :2] - xyz[:, :1, :2])) / factorial(2))
    mid = xyz[:, :, :2].mean(axis=1)
    dist = np.abs(2 * mid[:, 0] - mid[:, 1] - 0.25) / np.sqrt(5.0)
    near, far = dist < 0.02, dist > 0.2
    return {"h_near": float(h[near].mean()), "h_far": float(h[far].mean()),
            "h_far_over_near": float(h[far].mean() / h[near].mean()), "layer_width": float(np.sqrt(5 * kappa))}


def newton_summary(history: list[dict]) -> dict:
    ok = [bool(r["newton_converged"]) for r in history]
    its = [int(r["newton_iterations"]) for r in history]
    return {"levels": len(ok), "converged": int(sum(ok)), "failed_levels": [i + 1 for i, c in enumerate(ok) if not c],
            "iterations_min": min(its), "iterations_max": max(its), "iterations_median": float(np.median(its)),
            "fallback_levels": [i + 1 for i, r in enumerate(history) if r["newton_fallback"]],
            "statuses": sorted({r["newton_status"] for r in history})}


def summarise(history: list[dict]) -> dict:
    """Slopes against DoF^(1/2) over the last 2x window (finite columns only)."""
    keys = [f"{b}{f}{s}" for s in SUFFIXES for b in BASES for f in FIELDS
            if f"{b}{f}{s}" in history[0] and np.isfinite(history[-1][f"{b}{f}{s}"])]
    keys.append("E")
    return {k: fitted_slope(history, k, 2, span=SLOPE_SPAN) for k in keys}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="p = 1, 2, 3 and larger DoF caps")
    ap.add_argument("--preset", choices=("paper", "legacy"), default="paper")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default: numerics.json)")
    ap.add_argument("--kappa", type=float, help="diffusivity (default: problem.json)")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap of every run")
    args = ap.parse_args(argv)

    pspec = json.loads((HERE / "problem.json").read_text())
    nspec = json.loads((HERE / "numerics.json").read_text())
    extra = nspec.pop("example")  # keys of this example that are not LoopSettings fields
    mode = "full" if args.full else "short"
    nspec.update(extra["presets"][args.preset], preset=args.preset)
    kappa = args.kappa or pspec["kappa"]
    degrees = args.degrees or extra["degrees"][mode]
    out = HERE / "out" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for i, p in enumerate(degrees, start=1):
        print(f"run {i}/{len(degrees)}: preset={args.preset} kappa={kappa:g} p={p}  (t={time.time() - t0:.0f}s)", flush=True)
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
        np.savez_compressed(out / f"field_{tag}.npz", x=ub.function_space.tabulate_dof_coordinates()[:, :2], u=ub.x.array)
        last = history[-1]
        slopes = summarise(history) if len(history) > 3 else {}
        ratio = {t: last[f"Vh_phi{t}"] / last[f"Vh_theta{t}"] for t in SUFFIXES}
        summary.append({"preset": args.preset, "kappa": kappa, "p": p, "levels": len(history), "ndofs": last["ndofs"],
                        "E": last["E"], "time_s": round(elapsed, 1), "peak_rss_mb": round(peak_rss_mb()),
                        "slope_span": SLOPE_SPAN, "slopes": {k: round(v, 3) for k, v in slopes.items()},
                        "Vh_phi_over_theta": {"exact": ratio[""], "proj": ratio["_proj"]},
                        "newton": newton_summary(history), "umin": min(r["umin"] for r in history[-5:]),
                        "umax": max(r["umax"] for r in history[-5:]),
                        "clustering": layer_clustering(loop.domain, kappa), "last_row": last})
        nw = summary[-1]["newton"]
        print(f"  ndofs={last['ndofs']} levels={len(history)} E={last['E']:.2e} L2={last['L2']:.2e} Vh={last['Vh']:.2e} "
              f"newton {nw['converged']}/{nw['levels']} converged (iterations {nw['iterations_min']}-"
              f"{nw['iterations_max']}) time={elapsed:.0f}s peak={peak_rss_mb():.0f}MB", flush=True)
        print("  slopes vs DoF^(1/2), last 2x window: "
              + " ".join(f"{k}={v:.2f}" for k, v in summary[-1]["slopes"].items()
                         if k in ("E", "L2", "Vh", "L2_proj", "Vh_proj", "Vh_phi", "Vh_theta")), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main()

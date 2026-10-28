"""Poisson FEM baseline on the re-entrant corner: CG(p) Galerkin versus residual minimisation (ubar, theta, phi).

Usage: python run.py [--full] [--preset paper|legacy] [--degrees 1 2 ...] [--dofmax N]
Per polynomial degree three runs: residual minimisation with adaptive refinement and with uniform refinement (each
row also holds the plain CG(p) Galerkin errors on the same mesh, columns *_cg) and plain Galerkin on a uniform
sequence (``method: galerkin``, cheaper, so it reaches larger DoF). The short run (p = 1, 2, about 6000 DoF) takes
under a minute; --full goes to 1e5 DoF and p = 1, 2, 3. Results go to ./out/<preset>/ (history_p<p>_<tag>.csv,
summary.json with slopes over the last 2x window of DoF, ratios Galerkin / residual minimisation, time, memory).
"""

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

from adaptdg.vms.loop import AdaptiveStationaryLoop, fitted_slope
from adaptdg.vms.run import build_problem, build_settings

HERE = Path(__file__).parent
SLOPE_SPAN = 2.0  # slopes use the levels whose DoF is within a factor 2 of the last one
FIELDS = ("", "_theta", "_u", "_phi", "_cg")  # ubar, theta, u = ubar + utilde, phi, CG(p) Galerkin
SUFFIXES = ("", "_proj")  # error against the exact solution / its DG(p) L2 projection
RUNS = (("rm_adaptive", "rm", "adaptive"), ("rm_uniform", "rm", "uniform"), ("galerkin_uniform", "galerkin", "uniform"))


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def run_one(pspec: dict, nspec: dict, progress=None):
    loop = AdaptiveStationaryLoop(build_problem(pspec), build_settings(nspec, pspec))
    return loop.run(progress), loop.domain


def summarise(history: list[dict]) -> dict:
    keys = [f"{b}{f}{s}" for s in SUFFIXES for b in ("L2", "Vh") for f in FIELDS if f"{b}{f}{s}" in history[0]]
    return {k: fitted_slope(history, k, 2, span=SLOPE_SPAN) for k in keys}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="p = 1, 2, 3 up to 1e5 DoF")
    ap.add_argument("--preset", choices=("paper", "legacy"), default="paper")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default: numerics.json)")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap of every run")
    args = ap.parse_args(argv)

    pspec = json.loads((HERE / "problem.json").read_text())
    nspec = json.loads((HERE / "numerics.json").read_text())
    extra = nspec.pop("example")  # keys of this example that are not LoopSettings fields
    mode = "full" if args.full else "short"
    nspec.update(extra["presets"][args.preset], preset=args.preset)
    degrees = args.degrees or extra["degrees"][mode]
    jobs = [(p, *r) for p in degrees for r in RUNS]
    out = HERE / "out" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for i, (p, tag, method, ref) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: preset={args.preset} p={p} {tag}  (t={time.time() - t0:.0f}s)", flush=True)
        dofmax = args.dofmax or extra["dofmax"][mode][str(p)]
        t1 = time.time()
        history, _ = run_one(pspec, {**nspec, "p": p, "method": method, "refinement": ref, "dofmax": dofmax,
                                     "compare_galerkin": method == "rm"})
        elapsed = time.time() - t1
        with open(out / f"history_p{p}_{tag}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        last = history[-1]
        slopes = summarise(history) if len(history) > 3 else {}
        ratios = {}
        if method == "rm":  # Galerkin / residual minimisation on the same (last) mesh
            for s in SUFFIXES:
                for b in ("L2", "Vh"):
                    ratios[f"{b}_cg/{b}_ubar{s}"] = last[f"{b}_cg{s}"] / last[f"{b}{s}"]
                    ratios[f"{b}_cg/{b}_phi{s}"] = last[f"{b}_cg{s}"] / last[f"{b}_phi{s}"]
        summary.append({"preset": args.preset, "p": p, "run": tag, "levels": len(history), "ndofs": last["ndofs"],
                        "time_s": round(elapsed, 1), "peak_rss_mb": round(peak_rss_mb()), "slope_span": SLOPE_SPAN,
                        "slopes": {k: round(v, 3) for k, v in slopes.items()},
                        "ratios_last_mesh": {k: round(v, 3) for k, v in ratios.items()}, "last_row": last})
        print(f"  ndofs={last['ndofs']} levels={len(history)} L2={last['L2']:.2e} Vh={last['Vh']:.2e} "
              f"time={elapsed:.0f}s peak={peak_rss_mb():.0f}MB", flush=True)
        print("  slopes vs DoF^(1/2), last 2x window: "
              + " ".join(f"{k}={v:.2f}" for k, v in summary[-1]["slopes"].items()
                         if k in ("L2", "Vh", "Vh_phi", "L2_cg", "Vh_cg")), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main()

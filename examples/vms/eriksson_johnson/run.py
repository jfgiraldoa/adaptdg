"""Steady Eriksson-Johnson problem (paper section 5.6): boundary layer at x = 1, adaptive versus uniform.

Usage: python run.py [--full] [--preset paper|legacy] [--degrees 1 2 ...] [--kappa K ...] [--dim 3] [--dofmax N]
The short run (kappa = 1e-2, p = 1, 2, adaptive and uniform, about 6000 / 12000 DoF, under a minute) is the
default; --full runs kappa in {1e-2, 1e-3} and p = 1, 2, 3 up to the caps of numerics.json (overnight batch).
--dim 3 runs the 3D variant on the unit cube (p = 1 only by default). Results go to ./out/<preset>/ (dimension 3:
./out/<preset>_3d/): one CSV per run (history_k<kappa>_p<p>_<refinement>.csv with both error targets), mesh_*.npz
(last mesh) and summary.json (slopes over the last 2x window of DoF, time, peak memory, layer clustering).
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


def layer_clustering(domain) -> dict[str, float]:
    """Cell size (volume^(1/d)) in the outflow layer ``x > 0.9`` versus the inflow half ``x < 0.5``."""
    d = domain.topology.dim
    n = domain.topology.index_map(d).size_local
    xyz = domain.geometry.x[domain.geometry.dofmaps[0][:n]]
    vol = np.abs(np.linalg.det(xyz[:, 1:, :d] - xyz[:, :1, :d])) / factorial(d)
    h, xm = vol ** (1.0 / d), xyz[:, :, 0].mean(axis=1)
    near, far = xm > 0.9, xm < 0.5
    return {"ncells": int(n), "h_min": float(h.min()), "h_max": float(h.max()),
            "h_near": float(h[near].mean()), "h_far": float(h[far].mean()),
            "h_far_over_near": float(h[far].mean() / h[near].mean()),
            "frac_cells_near": float(near.mean())}


def summarise(history: list[dict], dim: int) -> dict:
    keys = [f"{b}{f}{s}" for s in SUFFIXES for b in BASES for f in FIELDS if f"{b}{f}{s}" in history[0]]
    return {k: fitted_slope(history, k, dim, span=SLOPE_SPAN) for k in keys}


def run_one(pspec: dict, nspec: dict, progress=None):
    """One adaptive or uniform loop; returns ``(history, final mesh)``."""
    loop = AdaptiveStationaryLoop(build_problem(pspec), build_settings(nspec, pspec))
    return loop.run(progress), loop.domain


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="kappa in {1e-2, 1e-3}, p = 1, 2, 3, larger DoF caps")
    ap.add_argument("--preset", choices=("paper", "legacy"), default="paper")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default: numerics.json)")
    ap.add_argument("--kappa", type=float, nargs="+", help="diffusivities (default: problem.json)")
    ap.add_argument("--dim", type=int, choices=(2, 3), default=2, help="3: unit cube variant of the paper")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap of every run")
    args = ap.parse_args(argv)

    pspec = json.loads((HERE / "problem.json").read_text())
    nspec = json.loads((HERE / "numerics.json").read_text())
    extra = nspec.pop("example")  # keys of this example that are not LoopSettings fields
    dim, mode = args.dim, "full" if args.full else "short"
    if dim == 3:
        pspec["name"] = "eriksson_johnson_3d"
        pspec["beta"] = [1.0, 0.0, 0.0]
        pspec["initial_mesh_size"] = 2
    nspec.update(extra["presets_3d" if dim == 3 else "presets"][args.preset], preset=args.preset)
    caps = extra["dofmax_3d" if dim == 3 else "dofmax"][mode]
    kappas = args.kappa or (pspec["kappa_full"] if args.full else [pspec["kappa"]])
    degrees = args.degrees or ([1] if dim == 3 else extra["degrees"][mode])
    jobs = [(k, p, ref) for k in kappas for p in degrees for ref in ("adaptive", "uniform")]
    out = HERE / "out" / (args.preset + ("_3d" if dim == 3 else ""))
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for i, (kappa, p, ref) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: preset={args.preset} dim={dim} kappa={kappa:g} p={p} {ref}  "
              f"(t={time.time() - t0:.0f}s)", flush=True)
        dofmax = args.dofmax or caps[str(p)]
        t1 = time.time()
        history, domain = run_one({**pspec, "kappa": kappa}, {**nspec, "p": p, "refinement": ref, "dofmax": dofmax})
        elapsed = time.time() - t1
        tag = f"k{kappa:g}_p{p}_{ref}"
        with open(out / f"history_{tag}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        np.savez_compressed(out / f"mesh_{tag}.npz", x=domain.geometry.x, cells=domain.geometry.dofmaps[0])
        last = history[-1]
        slopes = summarise(history, dim) if len(history) > 3 else {}
        ratio = {t: last[f"Vh_phi{t}"] / last[f"Vh_theta{t}"] for t in SUFFIXES}
        summary.append({"preset": args.preset, "dim": dim, "kappa": kappa, "p": p, "refinement": ref,
                        "levels": len(history), "ndofs": last["ndofs"], "E": last["E"], "time_s": round(elapsed, 1),
                        "peak_rss_mb": round(peak_rss_mb()), "slope_span": SLOPE_SPAN,
                        "slopes": {k: round(v, 3) for k, v in slopes.items()},
                        "Vh_phi_over_theta": {"exact": ratio[""], "proj": ratio["_proj"]},
                        "layer": layer_clustering(domain), "last_row": last})
        print(f"  ndofs={last['ndofs']} levels={len(history)} E={last['E']:.2e} L2={last['L2']:.2e} "
              f"Vh={last['Vh']:.2e} time={elapsed:.0f}s peak={peak_rss_mb():.0f}MB", flush=True)
        print(f"  slopes vs DoF^(1/{dim}), last 2x window: "
              + " ".join(f"{k}={v:.2f}" for k, v in summary[-1]["slopes"].items()
                         if k in ("L2", "Vh", "L2_proj", "Vh_proj", "Vh_phi", "Vh_theta")), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main()

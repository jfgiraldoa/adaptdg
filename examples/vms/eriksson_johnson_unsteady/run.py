"""Unsteady Eriksson-Johnson problem: BE and BDF2 with adaptive refinement in space at every time step.

Usage: python run.py [--full] [--preset paper|legacy] [--integrator be|bdf2|both] [--degrees 1 2 ...] [--kappa K]
                     [--dt DT] [--t-final T] [--steps N] [--dofmax N] [--no-uniform]
The short run (p = 2, 8 steps of dt = 0.005, both integrators, adaptive and two uniform meshes, under a minute) is the default;
--full runs p = 1, 2, 3 up to t = 0.1 (the legacy test: 20 steps) with the tolerances and caps of numerics.json.
Results go to ./out/<preset>/: per run steps_<tag>.csv (one row per time step: DoF, levels, E, errors of ubar, theta, u, phi in
L2 and Vh against the exact solution and its DG(p) L2 projection), levels_<tag>.csv (one row per solve), mesh_<tag>_s<k>.npz (meshes
at the first, middle and last step) and summary.json (final-time errors, time-integrated errors, DoF per step, time, memory).
"""

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from adaptdg.vms.run import build_problem
from adaptdg.vms.unsteady import UnsteadyLoop, build_unsteady_settings, time_integrated

HERE = Path(__file__).parent
SUFFIXES = ("", "_proj")  # error against the exact solution / its DG(p) L2 projection


def peak_rss_mb() -> float:
    """Peak resident memory of this process (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    scale = 1e6 if sys.platform == "darwin" else 1e3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def layer_clustering(domain) -> dict[str, float]:
    """Cell size in the outflow layer ``x > -0.05`` versus the inflow half ``x < -0.5``."""
    n = domain.topology.index_map(2).size_local
    xyz = domain.geometry.x[domain.geometry.dofmaps[0][:n]]
    h = np.sqrt(np.abs(np.linalg.det(xyz[:, 1:, :2] - xyz[:, :1, :2])) / 2)
    xm = xyz[:, :, 0].mean(axis=1)
    near, far = xm > -0.05, xm < -0.5
    return {"ncells": int(n), "h_min": float(h.min()), "h_max": float(h.max()), "h_near": float(h[near].mean()),
            "h_far": float(h[far].mean()), "h_far_over_near": float(h[far].mean() / h[near].mean())}


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(pspec: dict, nspec: dict, out: Path, tag: str, progress=None) -> dict:
    """One time-stepping run; writes its CSV and mesh files and returns the ``summary.json`` entry."""
    settings = build_unsteady_settings(nspec, pspec)
    loop = UnsteadyLoop(build_problem(pspec), settings)
    keep = {1, max(1, loop.nsteps // 2), loop.nsteps}

    def save_mesh(row, lp):
        if settings.refinement == "adaptive" and int(row["step"]) in keep:
            g = lp.domain.geometry
            np.savez_compressed(out / f"mesh_{tag}_s{int(row['step'])}.npz", x=g.x, cells=g.dofmaps[0], t=row["t"])

    t0 = time.time()
    history = loop.run(progress, on_step=save_mesh)
    elapsed = time.time() - t0
    write_csv(out / f"steps_{tag}.csv", history)
    last_step = [r for r in loop.levels if r["step"] == loop.nsteps]
    write_csv(out / f"levels_{tag}.csv", [{k: r.get(k, "") for k in last_step[0]} for r in loop.levels])
    last, ndofs = history[-1], [r["ndofs"] for r in history]
    ratio = {t: last[f"Vh_phi{t}"] / last[f"Vh_theta{t}"] for t in SUFFIXES if f"Vh_phi{t}" in last}
    return {"preset": settings.preset, "p": settings.p, "kappa": pspec["kappa"], "integrator": settings.integrator,
            "refinement": settings.refinement, "dt": settings.dt, "t_final": settings.t_final, "steps": len(history),
            "ndofs": last["ndofs"], "ndofs_min": min(ndofs), "ndofs_max": max(ndofs), "levels_per_step": [r["levels"] for r in history],
            "total_solves": len(loop.levels), "E": last["E"], "time_s": round(elapsed, 1), "peak_rss_mb": round(peak_rss_mb()),
            "Vh_phi_over_theta": {"exact": ratio.get("", None), "proj": ratio.get("_proj", None)},
            "integrated": time_integrated(history, settings.dt), "layer": layer_clustering(loop.domain), "last_row": last}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="p = 1, 2, 3, t_final = 0.1 and the caps of numerics.json")
    ap.add_argument("--preset", choices=("legacy", "paper"), default="legacy")
    ap.add_argument("--integrator", choices=("be", "bdf2", "both"), default="both")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default: numerics.json)")
    ap.add_argument("--kappa", type=float, help="diffusivity (default: problem.json)")
    ap.add_argument("--dt", type=float, help="time step (default: numerics.json)")
    ap.add_argument("--t-final", type=float, help="final time (default: numerics.json)")
    ap.add_argument("--steps", type=int, help="number of time steps (overrides --t-final)")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap per time step of every adaptive run")
    ap.add_argument("--no-uniform", action="store_true", help="skip the uniform-mesh reference runs")
    ap.add_argument("--max-levels", type=int, help="override the solves per time step of the adaptive runs")
    ap.add_argument("--level-errors", action="store_true", help="measure the errors on every level of the last step")
    args = ap.parse_args(argv)

    pspec = json.loads((HERE / "problem.json").read_text())
    nspec = json.loads((HERE / "numerics.json").read_text())
    extra = nspec.pop("example")  # keys of this example that are not UnsteadySettings fields
    mode = "full" if args.full else "short"
    nspec.update(extra["presets"][args.preset], preset=args.preset)
    if args.kappa:
        pspec["kappa"] = args.kappa
    dt = args.dt or nspec["dt"]
    t_final = nspec["t_start"] + args.steps * dt if args.steps else (args.t_final or extra["t_final"][mode])
    degrees = args.degrees or extra["degrees"][mode]
    integrators = ("be", "bdf2") if args.integrator == "both" else (args.integrator,)
    meshes = [] if args.no_uniform else extra["uniform_meshes"][mode]
    jobs = [(p, integ, ref, n) for p in degrees for integ in integrators for ref, n in [("adaptive", 0)] + [("uniform", n) for n in meshes]]
    out = HERE / "out" / args.preset
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = [], time.time()
    for i, (p, integ, ref, n) in enumerate(jobs, start=1):
        tag = f"p{p}_{integ}_" + (ref if ref == "adaptive" else f"uniform{n}")
        print(f"run {i}/{len(jobs)}: preset={args.preset} p={p} {integ} {ref}{n or ''} dt={dt:g} T={t_final:g} (t={time.time() - t0:.0f}s)",
              flush=True)
        spec = {**nspec, "p": p, "integrator": integ, "refinement": ref, "dt": dt, "t_final": t_final,
                "tol": nspec["tol"] if not isinstance(nspec["tol"], dict) else nspec["tol"][str(p)]}
        if ref == "adaptive":
            spec["dofmax"] = args.dofmax or extra["dofmax"][args.preset][mode][str(p)]
            if args.max_levels:
                spec["max_levels"] = args.max_levels
            spec["level_errors"] = args.level_errors
        else:
            spec.update(initial_mesh_size=n, max_levels=1, dofmax=None, tol=0.0)
        summary.append(run_one(pspec, spec, out, tag, progress=lambda s: print("  " + s, flush=True)))
        e = summary[-1]
        last = e["last_row"]
        print(f"  steps={e['steps']} ndofs={e['ndofs_min']}-{e['ndofs_max']} solves={e['total_solves']} E={e['E']:.2e} "
              f"L2_proj={last.get('L2_proj', float('nan')):.2e} L2_exact={last['L2']:.2e} Vh_proj={last.get('Vh_proj', float('nan')):.2e} time={e['time_s']:.0f}s "
              f"peak={e['peak_rss_mb']:.0f}MB", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main()

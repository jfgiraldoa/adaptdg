"""Paper preset versus legacy preset on the three VMS cases, both error targets.

Usage: python run.py [--full]. The short run (p = 1, 2, 12 runs, about 1-2 min) is the default;
--full uses larger problem sizes. Results go to ./out/<case>/<preset>/history_p*_*.csv and the
summary (slopes and the ratio Vh(phi)/Vh(theta) at the finest level) to ./out/summary.csv.
"""

import csv
import json
import sys
import time
from pathlib import Path

from adaptdg.vms.loop import fitted_slope
from adaptdg.vms.run import run_from_json

HERE = Path(__file__).parent
EXAMPLES = HERE.parent
PRESETS = ("paper", "legacy")
# case: (example folder, refinement, dofmax short/full, max levels short/full, slope span)
CASES = {
    "reentrant_corner": ("reentrant_corner", "adaptive", (8000, 40000), (80, 80), 4.0),
    "heterogeneous": ("heterogeneous", "uniform", (60000, 400000), (5, 6), 20.0),
    "anisotropic_r1e4": ("anisotropic", "adaptive", (8000, 40000), (80, 80), 4.0),
}
BASES = ("L2", "Vh")
FIELDS = ("", "_theta", "_phi")  # ubar, theta, phi


def summarise(case: str, p: int, preset: str, history: list[dict], span: float) -> list[dict]:
    """One row per (norm, field, target): slope and finest-level error; ratio columns for phi."""
    rows = []
    last = history[-1]
    for target, suffix in (("exact", ""), ("projection", "_proj")):
        for base in BASES:
            for field in FIELDS:
                key = f"{base}{field}{suffix}"
                rows.append({"case": case, "p": p, "preset": preset, "target": target,
                             "norm": base, "field": field.lstrip("_") or "ubar",
                             "slope": fitted_slope(history, key, 2, span=span),
                             "final_error": last[key], "ndofs": last["ndofs"],
                             "ratio_Vh_phi_over_theta": last[f"Vh_phi{suffix}"] / last[f"Vh_theta{suffix}"]})
    return rows


def main(full: bool = False) -> None:
    degrees = (1, 2)
    jobs = [(c, p, pr) for c in CASES for p in degrees for pr in PRESETS]
    summary: list[dict] = []
    t0 = time.time()
    for i, (case, p, preset) in enumerate(jobs, start=1):
        folder, refinement, dofmax, levels, span = CASES[case]
        print(f"run {i}/{len(jobs)}: {case} p={p} preset={preset}  (t={time.time() - t0:.0f}s)", flush=True)
        problem = json.loads((EXAMPLES / folder / "problem.json").read_text())
        if folder == "anisotropic":
            problem["r"] = 1e4
        out = run_from_json(problem, EXAMPLES / folder / "numerics.json", HERE / "out" / case / preset,
                            progress=None, p=p, refinement=refinement, reconstruct=True, preset=preset,
                            dofmax=dofmax[full], max_levels=levels[full])
        last = out["history"][-1]
        print(f"  ndofs={last['ndofs']} levels={len(out['history'])} "
              f"Vh_phi/Vh_theta exact={last['Vh_phi'] / last['Vh_theta']:.3f} "
              f"projection={last['Vh_phi_proj'] / last['Vh_theta_proj']:.3f}", flush=True)
        summary += summarise(case, p, preset, out["history"], span)
    with open(HERE / "out" / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main("--full" in sys.argv[1:])

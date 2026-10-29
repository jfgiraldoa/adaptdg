"""Heterogeneous diffusion-advection: ubar, u = ubar + utilde (= dG) and the adjoint reconstruction phi.

Usage: python run.py [--full] [--degrees 1 2 ...] [--dofmax N]. The short run (about 30 s) is the default: uniform refinement at
p = 1, 2, 3 (as in the original study, 4x4 start mesh) plus adaptive p = 1, 2. --full goes up to
about 1e6 DoF (uniform) and 2e5 DoF (adaptive), p = 1..3. Results go to ./out/.
"""

import argparse
import time
from pathlib import Path

from adaptdg.vms.run import run_from_json

HERE = Path(__file__).parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="larger problems")
    ap.add_argument("--degrees", type=int, nargs="+", help="polynomial degrees (default: see above)")
    ap.add_argument("--dofmax", type=int, help="override the DoF cap of every run")
    return ap.parse_args(argv)


def main(full: bool = False, degrees=None, dofmax=None) -> None:
    levels = {1: 7, 2: 6, 3: 6} if full else {1: 6, 2: 6, 3: 6}  # about 9e5 DoF at most (full)
    uniform = [(p, "uniform", dofmax or (1_000_000 if full else 150_000), levels[p]) for p in degrees or (1, 2, 3)]
    ada_p = degrees or ((1, 2, 3) if full else (1, 2))
    adaptive = [(p, "adaptive", dofmax or (200_000 if full else 20_000), 80) for p in ada_p]
    jobs = uniform + adaptive
    t0 = time.time()
    for i, (p, refinement, dofmax, levels) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: p={p} {refinement}  (t={time.time() - t0:.0f}s)", flush=True)
        out = run_from_json(
            HERE / "problem.json", HERE / "numerics.json", HERE / "out", progress=None,
            p=p, refinement=refinement, dofmax=dofmax, max_levels=levels,
        )
        last = out["history"][-1]
        s = {k: round(v, 2) for k, v in out["slopes"].items() if k != "E"}
        print(f"  ndofs={last['ndofs']} L2 ubar/u/phi={last['L2']:.2e}/{last['L2_u']:.2e}/"
              f"{last['L2_phi']:.2e} Vh={last['Vh']:.2e}/{last['Vh_u']:.2e}/{last['Vh_phi']:.2e}\n"
              f"  slopes vs DoF^(1/2): {s}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    a = parse_args()
    main(a.full, a.degrees, a.dofmax)

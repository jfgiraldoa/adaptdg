"""Re-entrant corner: adaptive residual minimisation at several polynomial degrees.

Usage: python run.py [--full]. The short run (p = 1, 2, about 10 s) is the default; --full runs
p = 1..4 up to 40000 DoF plus uniform refinement for comparison. Results go to ./out/.
"""

import sys
import time
from pathlib import Path

from adaptdg.vms.run import run_from_json

HERE = Path(__file__).parent


def main(full: bool = False) -> None:
    degrees = [1, 2, 3, 4] if full else [1, 2]
    dofmax = 40000 if full else 10000
    jobs = [(p, "adaptive") for p in degrees] + [(p, "uniform") for p in degrees[:2]]
    t0 = time.time()
    for i, (p, refinement) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: p={p} {refinement}  (t={time.time() - t0:.0f}s)", flush=True)
        out = run_from_json(
            HERE / "problem.json", HERE / "numerics.json", HERE / "out",
            progress=None, p=p, refinement=refinement, dofmax=dofmax,
            max_levels=60 if refinement == "adaptive" else 6,
        )
        last = out["history"][-1]
        slopes = {k: round(v, 2) for k, v in out["slopes"].items()}
        print(f"  ndofs={last['ndofs']} E={last['E']:.2e} L2={last['L2']:.2e} "
              f"Vh={last['Vh']:.2e} slopes vs DoF^(1/2): {slopes}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main("--full" in sys.argv[1:])

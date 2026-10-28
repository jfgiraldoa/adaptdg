"""Strongly anisotropic diffusion (r = 1e4 and 1e6): adaptive ubar, u (= dG) and phi.

Usage: python run.py [--full]. The short run (about 1 min) is the default: p = 1, 2 up to 20000 DoF;
--full runs p = 1..3 up to 150000 DoF. Results go to ./out/r1e04/ and ./out/r1e06/ (one folder per anisotropy ratio).
"""

import json
import sys
import time
from pathlib import Path

from adaptdg.vms.run import run_from_json

HERE = Path(__file__).parent


def main(full: bool = False) -> None:
    degrees = (1, 2, 3) if full else (1, 2)
    dofmax = 150_000 if full else 20_000
    jobs = [(r, p) for r in (1e4, 1e6) for p in degrees]
    t0 = time.time()
    for i, (r, p) in enumerate(jobs, start=1):
        print(f"run {i}/{len(jobs)}: r={r:.0e} p={p}  (t={time.time() - t0:.0f}s)", flush=True)
        out = run_from_json(
            {**json.loads((HERE / "problem.json").read_text()), "r": r},
            HERE / "numerics.json", HERE / "out" / f"r{r:.0e}".replace("+", ""), progress=None,
            p=p, dofmax=dofmax, max_levels=80,
        )
        last = out["history"][-1]
        s = {k: round(v, 2) for k, v in out["slopes"].items() if k != "E"}
        print(f"  ndofs={last['ndofs']} L2 ubar/u/phi={last['L2']:.2e}/{last['L2_u']:.2e}/"
              f"{last['L2_phi']:.2e} Vh={last['Vh']:.2e}/{last['Vh_u']:.2e}/{last['Vh_phi']:.2e}\n"
              f"  slopes vs DoF^(1/2): {s}", flush=True)
    print(f"done in {time.time() - t0:.0f}s; now run: python plot.py")


if __name__ == "__main__":
    main("--full" in sys.argv[1:])

# Paper preset versus legacy preset

Runs the three VMS cases (re-entrant corner and anisotropic r = 1e4 with adaptive refinement,
heterogeneous diffusion-advection with uniform refinement) for p = 1, 2 under both presets of
`adaptdg.vms.preset` and measures every error against two targets:

* `exact`: the exact solution (the true error);
* `projection`: its DG(p) L2 projection (what the original implementation measured).

The `paper` preset is the package default; `legacy` reproduces the choices of the original code
(no L2 term in the diffusion Gram form, interior penalty `k1 k2/(k1+k2)`, quadrature degree 8,
fixed-fraction marking with ratio 0.25, adjoint ordering `b(v, eps')`); see "Legacy and paper presets"
in `docs/VMS.md`. Individual differences can be switched with `preset_overrides` in `numerics.json`.

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py          # short: 12 runs, about 1-2 min
    python run.py --full   # larger problems (a few minutes)
    python plot.py         # figures in out/

`out/summary.csv` has, per (case, p, preset, target, norm, field), the fitted slope against DoF^(1/2)
and the finest-level error, plus the ratio Vh(phi)/Vh(theta) at the finest level. Per-run histories are in
`out/<case>/<preset>/history_p*.csv` (columns `L2, Vh` against the exact solution, `*_proj` against the
projection, for ubar and `_theta`, `_u`, `_phi`). Figures: `errors_<case>.png` (rows Vh/L2, columns target;
solid = paper, dashed = legacy) and `ratio_Vh_phi_theta.png`.

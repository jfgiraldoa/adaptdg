# Isotropic Burgers equation (manufactured tanh layer)

Nonlinear conservation law with diffusion on the unit square (paper section 6.5.1, eq. 63)

    div(b u^2/2) - kappa Delta u = f,   b = (1, 1),   kappa = 1e-3,   Omega = (0, 1)^2,

with the exact solution `u = (1 - tanh((2x - y - 1/4)/sqrt(5 kappa)))/2` (an inner layer of width `sqrt(5 kappa)` = 0.07 along
the line `2x - y = 1/4`, `0 < u < 1`); the source `f` and the Dirichlet data on the four sides are derived from it
symbolically, the initial guess is `u0 = 0.5`, the start mesh is 6 x 6 (crossed triangles) and the Gram form is the isotropic norm
of the paper (eqs. 49-50: SWIP part, `(w, w)`, face jumps and `h_K |grad w|^2`, no wind; `"norm_form": "streamline"` in `problem.json` gives
the earlier norm with the wind `(1, 1)`). Both error targets are reported: against the exact solution and against its DG(p)
L2 projection (suffix `_proj`, what the original code measured).

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: p = 1, 2, about 10000 / 12000 DoF, about 30 s, < 0.5 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # p = 1, 2, 3, 4 up to 60000 / 80000 / 100000 / 150000 DoF (see "Time and memory")
    python run.py --kappa 1e-2     # another diffusivity
    python plot.py                 # errors_k*.png and fields_k*.png in out/<preset>/

Every run prints `run i/N` and one line per level (DoF, `E`, `L2`, `Vh`, Newton iterations and residual, `ubar` min/max, time).
`out/<preset>/` (git-ignored) holds `history_k<kappa>_p<p>.csv` (level, ndofs, ncells, `E`, the `L2`/`Vh` errors of `ubar`,
`theta`, `u` and `phi` in both targets, `newton_*`, `theta_converged`, `umin`, `umax`, time), `mesh_*.npz`, `field_*.npz` and
`summary.json` (slopes over the last 2x window of DoF, Newton statistics, ratio `Vh(phi)/Vh(theta)`, cell-size clustering around
the layer, time, peak memory). Settings are in `problem.json` and `numerics.json` (the usual `LoopSettings` keys plus the block
`example`).

## Presets

* `paper` (default): Lax-Friedrichs flux with `eta_f = |f'(u).n|` (eqs. 45-47, conservative form), hybrid Newton (Gauss-Newton
  Jacobian until `|R| < 1e-3`, full Jacobian afterwards, undamped, stops at `|du|_Vh < 1e-9` or `|R| < 1e-11`, damped restart
  on failure), Doerfler marking, `reconstruct` on: `theta` is the nonlinear dG solution (Newton from `ubar + utilde`), `u =
  ubar + utilde` (eq. 61, one Newton-type correction) and `phi` the adjoint reconstruction (eq. 62).
* `legacy`: the original choices: the wind `u b` through the generic upwind form (`bh_dar_NL`), damped Newton
  (`Damping_solve`, `t = 1/(1 + K |R|)`, `omega = 0.5`, tolerance 1e-12, Gauss-Newton `bh_prima` Jacobian), no L2 term in the Gram
  form of pure diffusion, SWIP factors 0.5, quadrature degree 8, fixed-fraction marking, adjoint ordering `b(v, eps')`. The
  original loop has no bound on `K`: a level is given up when `K > 1e14` (`failK`) or after `maxit` steps (example value 400,
  original 20000), flagged in the history, the best iterate is kept and the run continues; `"newton_fallback": true` would retry
  with the hybrid solver (off by default). `theta` is solved with the undamped exact-Jacobian Newton in both presets
  (`theta_converged = 0` and NaN `*_theta` columns when it fails; the legacy dG iteration fails on most levels).

The paper preset uses the library default `maxit = 100`: with the isotropic norm no level fails by `maxit` (with the earlier streamline
norm, 4 of 24 levels at `p = 1` and 4 of 16 at `p = 2` did, and `maxit = 400` was needed); the legacy preset keeps `"newton": {"maxit": 400}`.

## Measured behaviour (Mac, kappa = 1e-3, isotropic norm, `python run.py --full`, 2 min in total)

| p | DoF (levels) | Newton converged | iterations | E last | L2 / Vh last | slopes vs DoF^(1/2): E, L2, Vh | Vh(phi)/Vh(theta) exact / proj | time, memory |
|---|---|---|---|---|---|---|---|---|
| 1 | 63095 (29) | 27/29 (levels 3 and 7 `stalled`) | 3-53 (median 5) | 1.3e-3 | 2.0e-3 / 9.5e-3 | -1.0, -0.8, -1.3 | 0.97 / 0.83 | 26 s, 0.8 GB |
| 2 | 93096 (32) | 32/32 | 2-138 (median 4) | 7.3e-5 | 1.3e-4 / 6.4e-4 | -2.0, -2.5, -2.8 | 0.97 / 0.88 | 37 s, 1.4 GB |
| 3 | 109641 (41) | 41/41 | 1-94 (median 3) | 4.4e-6 | 1.1e-6 / 5.4e-5 | -3.2, -5.9, -3.9 | 0.99 / 0.94 | 56 s, 1.7 GB |

Before (streamline norm with the wind `(1, 1)`, `maxit = 400`, 30000-50000 DoF): 28/28, 27/28 and 28/28 levels, up to 421 iterations,
`L2` of `ubar` irregular at `p = 2, 3` and a plateau near 5e-5 at `p = 3` while `E` and `Vh` kept decreasing. With the isotropic norm the
library default `maxit = 100` is enough (no level fails by `maxit`; the largest iteration counts are 53, 138 and 94 for `p = 1, 2, 3`
against 220, 421 and 36), the `p = 3` plateau is gone (`L2` 1.1e-6 at 110000 DoF, 2 increases in the last 20 levels) and `E`, `Vh` follow
the optimal rate. The price is two `stalled` levels at `p = 1` (3 and 7, 676 and 1219 DoF, final `|R|` 5e-6 and 1e-6, the same with
`maxit = 400`); the run continues from the best iterate and the later levels are not affected.
(slopes over the last 2x window of DoF; the adaptive histories are stepped, so they fluctuate by a few tenths and the pre-asymptotic
range is long: the layer needs cells of 0.003-0.004 at the end.)

* Both targets give the same `L2`; `Vh_proj/Vh` is 0.79-0.87 at `p = 1`, 0.88-0.96 at `p = 2` and 0.95-1.14 at `p = 3` over the second half of the levels.
* `E` follows the optimal rate `-p` (against DoF^(1/2)) at `p = 1, 2` and is faster than `-p` at `p = 3` in the last window (`-3.2`, `Vh`
  `-3.9`). `L2` of `ubar` is still irregular at `p = 1` (jumps of up to 3x between consecutive levels, slope not meaningful) and
  mildly at `p = 2` (one increase in the last 15 levels).
* `Vh(phi)/Vh(theta)` is 0.97-0.99 against the exact solution (phi is not better there) and 0.83-0.94 against the projection; the
  fine-scale update `u = ubar + utilde` coincides with `theta` in `Vh` to 0.2%. `L2(phi)/L2(theta)` at the last level is 1.4, 0.9 and 0.97 for `p = 1, 2, 3`.
  `theta` converges on every level. `Vh(ubar)/Vh(theta)` is 1.17, 1.22 and 1.09 at the last level: `ubar` is the coarse-scale (conforming) part.
* `ubar` stays within `[-8.1e-4, 1.0001]` over the last five levels at `p = 1` (`[-7.8e-4, 1.00001]` at `p = 2`, `[-8e-7, 1.000002]` at `p = 3`). Adaptive
  cells near the layer are 29, 22 and 16 times smaller than far from it (`p = 1, 2, 3`).

Legacy preset (fixed-fraction marking: fewer, larger refinement steps; `maxit = 400`, up to about 100000 DoF):

| p | DoF (levels) | Newton converged | iterations (median) | E last | L2 / Vh last | slopes: E, L2, Vh | time, memory |
|---|---|---|---|---|---|---|---|
| 1 | 81299 (15) | 5/15 | 8-145 (22) | 1.3e-3 | 1.2e-3 / 7.8e-3 | -0.9, -0.7, -1.4 | 30 s, 0.8 GB |
| 2 | 91933 (16) | 13/16 | 2-257 (14) | 8.4e-5 | 1.9e-4 / 7.0e-4 | -1.9, -2.5, -2.4 | 34 s, 1.3 GB |
| 3 | 101063 (20) | 16/20 | 2-205 (9) | 5.5e-6 | 1.2e-6 / 5.8e-5 | -3.2, -7.0, -4.3 | 44 s, 1.8 GB |

The legacy Newton fails (`failK`) on 10 of 15, 3 of 16 and 4 of 20 levels, nearly all of them on the coarse levels (below 3500 DoF at `p = 2, 3`;
also at 7839, 11032 and 25015 DoF at `p = 1`; `theta` fails on 5 levels at `p = 1`). The Gauss-Newton Jacobian drops `n''` and
converges linearly (or stalls where `|R|` is a local minimum), which is why the hybrid solver switches to the full Jacobian. The original code
would loop on these levels. (With the earlier streamline norm the failed levels looked like false failures, `E` and `Vh` at equal DoF as close
to the paper preset's as those of the converged levels; this was not rechecked with the isotropic norm.)

## Several roots and path dependence

On one mesh the discrete problem has more than one solution; which one Newton reaches depends on the path (warm start from the
previous level, damping, the Jacobian choice), and `L2` of `ubar` changes by up to 50% between them (the L2 columns jump from
level to level, see the L2 panels). The estimator `E` agrees to 3-4 digits between them and is the number to quote. Failed Newton
levels are drawn with a red cross in `errors_k*.png`.

## Degree 4 and the `theta` plateau (paper Fig. 15, station, 2026-11-04)

`p = 4` (paper preset, 150000 DoF cap): 44/44 levels converge (1-10 iterations), last level 156665 DoF, `E` 2.9e-7, `L2` of `ubar` 1.3e-7,
`Vh` 4.4e-6; slopes against DoF^(1/2) over the last 2x window: `L2` -5.3 (optimal -5), `Vh` -4.5 (optimal -4), `E` -3.9; 129 s, 1.8 GB.
At `p = 3, 4` the `L2` of `theta`, `u` and `phi` flattens (2.2e-6 at 110000 DoF, 1.6e-7 at `p = 4`) while that of `ubar` and `Vh` keep decreasing.
This is not a measurement artifact: on the last `p = 3` mesh (frozen) raising the error quadrature from 12 to 24 and tightening the Newton
tolerances of `ubar` and `theta` (`|du|` 1e-14, `|R|` 1e-16) change `L2(theta)` in no digit, and one uniform refinement lowers it
from 2.24e-6 to 1.5e-7 (rate 3.9 = p + 1). The error of `theta` sits in the tails of the layer (72% within 6 layer widths, 28% beyond), where
the estimator (driven by `ubar`) no longer refines; the L2 projection of the exact solution is 3.8e-7 on that mesh.

## Time and memory of `--full`

Measured with `--full` (about linear in DoF for the time, MUMPS memory about DoF^1.2): paper preset
p = 1 (60000 DoF) 26 s, p = 2 (80000) 37 s, p = 3 (100000) 56 s, in total 2 min and 1.7 GB at p = 3 (measured); legacy
preset 108 s (the failed levels spend up to 400 steps each). The figures of the paper show 20+ levels,
which the short run already reaches at `p = 1`.

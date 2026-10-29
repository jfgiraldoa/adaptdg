# Isotropic Burgers equation (manufactured tanh layer)

Nonlinear conservation law with diffusion on the unit square (paper section 6.5.1, eq. 63)

    div(b u^2/2) - kappa Delta u = f,   b = (1, 1),   kappa = 1e-3,   Omega = (0, 1)^2,

with the exact solution `u = (1 - tanh((2x - y - 1/4)/sqrt(5 kappa)))/2` (an inner layer of width `sqrt(5 kappa)` = 0.07 along
the line `2x - y = 1/4`, `0 < u < 1`); the source `f` and the Dirichlet data on the four sides are derived from it
symbolically, the initial guess is `u0 = 0.5`, the start mesh is 6 x 6 (crossed triangles) and the Gram form uses the constant
norm wind `(1, 1)` (legacy `beta_inf = 1`). Both error targets are reported: against the exact solution and against its DG(p)
L2 projection (suffix `_proj`, what the original code measured).

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: p = 1, 2, about 10000 / 12000 DoF, about 30 s, < 0.5 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # p = 1, 2, 3 up to 60000 / 80000 / 100000 DoF (see "Time and memory")
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

The example sets `"newton": {"maxit": 400}` in both presets: with the default 100 undamped iterations 4 of 24 levels at `p = 1`
and 4 of 16 at `p = 2` fail, while 400 converges them (one level needs 220 iterations); the library default is unchanged.

## Measured behaviour (Mac, kappa = 1e-3, paper preset)

| p | DoF (levels) | Newton converged | iterations | E last | L2 / Vh last | slopes vs DoF^(1/2): E, L2, Vh | Vh(phi)/Vh(theta) exact / proj | time, memory |
|---|---|---|---|---|---|---|---|---|
| 1 | 32079 (28) | 28/28 | 3-220 (median 4) | 2.0e-3 | 3.4e-3 / 1.9e-2 | -1.1, -1.6, -1.6 | 0.98 / 0.86 | 17 s, 0.5 GB |
| 2 | 49453 (28) | 27/28 (level 4 `failK`) | 2-421 | 1.7e-4 | 1.2e-4 / 1.8e-3 | -2.0, -6.1, -2.5 | 0.97 / 0.85 | 25 s, 1.1 GB |
| 3 | 40901 (28) | 28/28 | 2-36 | 2.5e-5 | 5.0e-5 / 5.5e-4 | -3.2, -0.9, -4.0 | 0.98 / 0.94 | 28 s, 1.3 GB |

(slopes over the last 2x window of DoF; the adaptive histories are stepped, so they fluctuate by a few tenths and the
pre-asymptotic range is long: the layer needs cells of 0.005 at the end.)

* Both targets give the same `L2` and nearly the same `Vh` (`Vh_proj` is 14-17% smaller than `Vh` at `p = 1, 2` and 1.5% at `p = 3`).
* `E` follows the optimal rate `-p` (against DoF^(1/2)) at `p = 2, 3` and `Vh` of `ubar` is at or above it (`-2.5`, `-4.0`); at `p = 1` all
  slopes are between `-1.1` and `-1.6` in the last window. The `L2` slope is not meaningful at `p = 2, 3`: `L2` of `ubar` is path
  dependent (see below) and plateaus near `5e-5` at `p = 3` (`L2` of `theta` too, `3e-5`), while `E` and `Vh` keep decreasing; the
  cause of the plateau was not investigated.
* `Vh(phi)/Vh(theta)` is 0.97-0.98 against the exact solution (phi is not better there) and 0.85-0.94 against the projection; the
  fine-scale update `u = ubar + utilde` is within 1% of `theta` in `Vh` for `p >= 1` (`Vh_u/Vh_theta` 1.00-1.01 at the last level).
  `L2(phi)/L2(theta)` at the last level is 2.4, 0.8 and 0.9 for `p = 1, 2, 3` and fluctuates from level to level. As `theta` is the
  nonlinear dG solution on the same mesh, `Vh(theta)` is 1.4-1.55 times smaller than `Vh(ubar)` at the last level: `ubar` is the
  coarse-scale (conforming) part.
* `ubar` stays within `[-0.022, 1.0003]` over the last five levels at `p = 1` (`[-0.002, 1.00001]` at `p = 2`, `[-2e-4, 1.0000]` at `p = 3`): the
  oscillation is confined to the coarse levels. Adaptive cells within 0.02 of the layer are 17 times smaller than farther than 0.2
  (`p = 2`).

Legacy preset (fixed-fraction marking: fewer, larger refinement steps; `maxit = 400`, up to about 50000 DoF):

| p | DoF (levels) | Newton converged | theta converged | E last | L2 / Vh last | slopes: E, L2, Vh |
|---|---|---|---|---|---|---|
| 1 | 53411 (16) | 7/16 | 12/16 | 1.9e-3 | 1.7e-3 / 1.4e-2 | -1.1, -3.0, -1.6 |
| 2 | 52774 (15) | 6/15 | 15/15 | 2.0e-4 | 1.3e-4 / 1.7e-3 | -2.3, -10.1, -3.9 |
| 3 | 46903 (17) | 11/17 | 17/17 | 2.4e-5 | 1.5e-5 / 4.2e-4 | -3.6, -4.9, -4.7 |

The legacy Newton fails (`failK`) on 9 of 16, 9 of 15 and 6 of 17 levels, with final `|R|` between `7.6e-12` and `2.4e-5`. They look like "false
failures": at equal DoF the `E` and `Vh` of the failed levels are as close to the paper preset's as those of the converged levels
(median legacy/paper ratio of `E`: 1.19-1.28 failed against 1.19-1.32 converged for `p = 1, 2, 3`; of `Vh`: 1.05-1.55 against
1.04-1.37), but they do not meet the
original criterion (`|du|_Vh < 1e-12`); the Gauss-Newton Jacobian drops `n''` and converges linearly (or stalls where `|R|` is a local minimum), which is
why the hybrid solver switches to the full Jacobian. The original code would loop on these levels.

## Several roots and path dependence

On one mesh the discrete problem has more than one solution; which one Newton reaches depends on the path (warm start from the
previous level, damping, the Jacobian choice), and `L2` of `ubar` changes by up to 50% between them (the L2 columns jump from
level to level, see the L2 panels). The estimator `E` agrees to 3-4 digits between them and is the number to quote. Failed Newton
levels are drawn with a red cross in `errors_k*.png`.

## Time and memory of `--full`

Estimated from the 30000-50000 DoF runs above (about linear in DoF for the time, MUMPS memory about DoF^1.2): paper preset
p = 1 (60000 DoF) about 1 min, p = 2 (80000) about 1 min, p = 3 (100000) about 1.5-2 min, in total 4-5 min and about 3.5 GB at
p = 3; legacy preset similar (the failed levels spend up to 400 steps each: add 1-3 min). The figures of the paper show 20+ levels,
which the short run already reaches at `p = 1`.

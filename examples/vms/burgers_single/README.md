# Single-component Burgers equation (shock layer)

Nonlinear conservation law with diffusion on the unit square (paper section 6.5.2, eq. 64)

    div f(u) - kappa Delta u = 0,   f(u) = (u^2/2, u),   i.e.   u u_x + u_y = kappa (u_xx + u_yy),

Dirichlet data `u_D = u0 = 1 - 2x` on `y = 0`, `x = 0` and `x = 1`, zero diffusive flux at the outflow `y = 1`,
initial guess `u0 = 1 - 2x`, `kappa = 1e-2, 1e-3, 1e-4`, `p = 3` and a uniform 4 x 4 start mesh (crossed triangles), as in
the paper. There is no exact solution: the runs report the estimator `E`, the Newton convergence, `min`/`max` of `ubar`
(no overshoot beyond `[-1, 1]` means no oscillation), the shock position and its width. The error-target columns
(`L2`, `Vh`, `*_proj`) do not exist for this case.

Two things differ from the text of the paper:

* **The shock is at `x = 1/2`, not at `x = 0`.** `u0 = 1 - 2x` changes sign at `x = 1/2` and characteristics
  (`dx/dy = u`) converge there; measured on the line `y = 0.75`: zero crossing of `ubar` at `x = 0.500`, steepest slope
  about `-52` against the analytic `-1/(2 kappa) = -50` at `kappa = 1e-2`.
* **The outflow boundary `y = 1` needs the natural flux term `v f(u).n`.** In the conservative form
  `-(grad v, f(u)) + ...` the boundary integral `v f(u).n` has to be put back on the faces without Dirichlet data
  (`Problem.outflow_tags`); without it Newton fails on every level at `p = 3`.
  The SWIP/Nitsche terms already skip faces without Dirichlet data, which gives the zero diffusive flux.

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: kappa = 1e-2, 1e-3, p = 3, about 5000 DoF, about 10 s, < 0.5 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # kappa in {1e-2, 1e-3, 1e-4}, up to 20000 DoF (see "Time and memory")
    python plot.py                 # burgers_single_k*_p*.png in out/<preset>/

Every run prints `run i/N` and one line per level (DoF, `E`, Newton iterations and residual, `ubar` min/max, time).
`out/<preset>/` (git-ignored) holds `history_k<kappa>_p<p>.csv` (level, ndofs, ncells, E, `newton_*` columns, umin, umax,
time), `mesh_*.npz` (last mesh), `field_*.npz` (nodal `ubar` and its profiles along `y = 0.25`, `0.75`) and
`summary.json` (Newton statistics, shock position / steepest slope / width at half maximum on `y = 0.75`, cell-size
clustering around the shock, time, peak memory). Settings live in `problem.json` and `numerics.json` (the usual
`LoopSettings` keys plus the block `example`).

## Presets

* `paper` (default): Lax-Friedrichs flux of eqs. (45)-(47) in conservative form, hybrid Newton (Gauss-Newton Jacobian
  until `|R| < 1e-3`, full Jacobian afterwards, undamped; stops at `|du|_Vh < 1e-9` or `|R| < 1e-11`; if the undamped
  iteration fails the level is restarted once with the legacy damping, `omega = 0.1`), Doerfler marking
  (`theta_ref = 0.25`), quadrature degree 12.
* `legacy`: the original choices: the wind `f'(u) = (u, 1)` through the generic upwind form (non-conservative volume term,
  `bh_dar_NL`), damped Newton (`Damping_solve`: `t = 1/(1 + K |R|)`, `omega = 0.5`, `K` x10 on failure, tolerance 1e-12) with
  the Gauss-Newton `bh_prima` Jacobian, no L2 term, SWIP factors 0.5, quadrature degree 8, fixed-fraction marking. The
  original loop has no cap on `K` and would not terminate on a failed level; here a level is given up when `K > 1e14`
  (status `failK`) or after `maxit = 500` accepted steps (the example's cap, the original value is 20000) and is flagged
  in the history; the best iterate is kept and the run continues. `"newton_fallback": true` retries such a level with the
  hybrid solver (off by default so the comparison is honest). The original code never ran this case.

## Measured behaviour (Mac, `p = 3`)

Paper preset, up to about 20000 DoF (`--full` caps):

| kappa | levels | Newton converged | iterations per level | shock x (y = 0.75) | steepest slope (analytic `-1/(2 kappa)`) | width (FWHM of `-u_x`) | `ubar` range | E last / slope vs DoF^(1/2) | time, peak memory |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | 26 | 26/26 | 2-8 | 0.499 | -48.7 (-50) | 0.036 | [-1.002, 1.002] | 2.9e-4 / -3.4 | 6 s, 0.8 GB |
| 1e-3 | 16 | 16/16 | 4-7, and 110-116 on levels 1-5 (undamped fallback) | 0.499 | -495 (-500) | 0.0040 | [-1.011, 1.016] | 1.0e-2 / -1.8 | 14 s, 0.8 GB |
| 1e-4 | 15 | **5/15** | 113-200 | 0.483 | -1126 (-5000) | 0.0020 | [-1.86, 1.33] | 9.8e-2 / -0.6 | 141 s, 0.7 GB |

* `kappa = 1e-2`: the solution is oscillation free (`ubar` within 0.2% of `[-1, 1]`), the shock is at `x = 0.500` (zero crossing)
  and the slope matches the viscous profile to 4-5%; `E` decays with a slope of about -3.4 against DoF^(1/2) (the optimal
  rate is `-p = -3`). The mean cell size away from the shock (`|x - 0.5| > 0.25`) is 7.5 times that within 0.05 of it (16 times at `kappa = 1e-3`).
* `kappa = 1e-3`: the damped restart is needed on the first five (coarse) levels, afterwards the undamped iteration converges in
  4-7 steps; the slope of `E` (-1.8) is still pre-asymptotic at 24000 DoF.
* `kappa = 1e-4` is **not reproduced**: Newton converges on 5 of 15 levels, `ubar` overshoots to `[-1.9, 1.3]`, the shock is
  not resolved (the layer is about 2e-3 wide, the finest cells are about 0.01 at 21000 DoF) and `E` stays near 0.1. The
  paper's Fig. 16 presumably needs finer meshes than the cap of `--full` (not tested beyond 21000 DoF) or a better nonlinear
  strategy (not found: the damped restart with `omega = 0.1` is the only fallback tried that converges on some levels).

Legacy preset (damped Newton, wind form, fixed-fraction marking): `kappa = 1e-2`: 15/15 levels converge (2-27 iterations),
22000 DoF, shock `x = 0.499`, slope -48.7, `E` slope -3.4, 7 s; `kappa = 1e-3`: **7/13** levels converge, the others stop with `failK`
after 20-131 iterations at `|R|` between 9e-11 and 4e-5 (levels 1, 2, 4, 6, 9, 10); the last level converged, with `E` 1.06e-2 against 1.02e-2 of the paper preset and slope -502 (28 s, 0.9 GB).
Failed legacy levels are real non-convergence in the sense of the original criterion (the original loop would not terminate), but
the residual of level 2 is 9e-11, so some are close to converged.

### Time and memory of `--full`

Paper preset: 6 + 14 + 141 s = about 3 minutes and at most 0.9 GB (measured per kappa, one run each); legacy preset:
7 + 28 s for kappa = 1e-2, 1e-3 and the `kappa = 1e-4` run is estimated at 3-8 minutes (failed levels spend up to `maxit = 500`
steps, 1-3 s per 100 steps at 10000 DoF); at most about 1 GB.


## Caveats

* Newton is the hard part at small `kappa`: the undamped hybrid iteration stagnates on several levels (`kappa = 1e-3`
  levels 1-5 and `kappa = 1e-4` almost every level), the damped restart then converges in 10-20 iterations at `kappa = 1e-3`
  and fails on some levels at `kappa = 1e-4` (status `failK`; the iterate with the smallest residual is kept and the
  adaptive run goes on, so `E` of a failed level is not a converged value). The `newton_iterations` column of a level that
  used the fallback counts both attempts (100 failed undamped steps + the damped ones).
* Several roots exist on a given mesh (the problem is nonlinear); the root reached depends on the warm start from the
  previous level. `E` is stable to 3-4 digits between roots, other functionals are not.
* The legacy preset differs from the original in the points listed above (`K` cap, `maxit`, no FEniCS 2019 `refine`).

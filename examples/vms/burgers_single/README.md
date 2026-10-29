# Single-component Burgers equation (shock layer)

Nonlinear conservation law with diffusion on the unit square (paper section 6.5.2, eq. 64)

    div f(u) - kappa Delta u = 0,   f(u) = (u^2/2, u),   i.e.   u u_x + u_y = kappa (u_xx + u_yy),

Dirichlet data `u_D = u0 = 1 - 2x` on `y = 0`, `x = 0` and `x = 1`, zero diffusive flux at the outflow `y = 1`,
initial guess `u0 = 1 - 2x`, `kappa = 1e-2, 1e-3, 1e-4`, `p = 3` and a uniform 4 x 4 start mesh (crossed triangles), as in
the paper. There is no exact solution: the runs report the estimator `E`, the Newton convergence, `min`/`max` of `ubar`
(no overshoot beyond `[-1, 1]` means no oscillation), the shock position and its width. The error-target columns
(`L2`, `Vh`, `*_proj`) do not exist for this case.

Three things differ from the text of the paper:

* **The shock is at `x = 1/2`, not at `x = 0`.** `u0 = 1 - 2x` changes sign at `x = 1/2` and characteristics
  (`dx/dy = u`) converge there; all of them meet at `(1/2, 1/2)`, so for `y > 1/2` the solution is the stationary viscous
  profile `u = -tanh((x - 1/2)/(2 kappa))` between the inflow states `+1` and `-1`: steepest slope `-1/(2 kappa)`, full width
  at half maximum of `-u_x` equal to `3.5 kappa`. Measured on the line `y = 0.75`.
* **The outflow boundary `y = 1` needs the natural flux term `v f(u).n`.** In the conservative form
  `-(grad v, f(u)) + ...` the boundary integral `v f(u).n` has to be put back on the faces without Dirichlet data
  (`Problem.outflow_tags`); without it the discrete problem has a different (wrong) solution: at `kappa = 1e-2`, 4 x 4 mesh,
  Newton needs 38 instead of 6 iterations, `E` is 6 times larger and `ubar` differs by 1.2 from the correct one.
  The SWIP/Nitsche terms already skip faces without Dirichlet data, which gives the zero diffusive flux.
* **The norm is the one of eqs. (49)-(50), not the streamline norm of the linear problems.** The Gram form is the SWIP part plus
  `(w, w) + sum_F ([[w]], [[w]])_F + sum_K h_K (grad w, grad w)_K` (all faces, isotropic gradient, no wind;
  `norm_form = "isotropic"`, default of `BurgersSingle`). The streamline form with a norm wind `(1, 1)` (`"streamline"`, what
  `burgers_isotropic` uses) controls `grad w` only along `(1, 1)`; with it Newton fails on most levels at `kappa = 1e-4`
  (5 of 15 levels) and the line sampling used to measure the slope was coarser than the layer, which made the shock look
  unresolved.

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: kappa = 1e-2, 1e-3, p = 3, about 5000 DoF, about 10 s, < 0.5 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # kappa in {1e-2, 1e-3, 1e-4}, up to 50000 DoF (see "Time and memory")
    python plot.py                 # burgers_single_k*_p*.png in out/<preset>/

Every run prints `run i/N` and one line per level (DoF, `E`, Newton iterations and residual, `ubar` min/max, time).
`out/<preset>/` (git-ignored) holds `history_k<kappa>_p<p>.csv` (level, ndofs, ncells, E, `newton_*` columns, umin, umax,
time), `mesh_*.npz` (last mesh), `field_*.npz` (nodal `ubar` and its profiles along `y = 0.25`, `0.75`) and
`summary.json` (Newton statistics, shock position / steepest slope / width at half maximum on `y = 0.75`, cell-size
clustering around the shock, time, peak memory). Settings live in `problem.json` and `numerics.json` (the usual
`LoopSettings` keys plus the block `example`).

## Presets

* `paper` (default): Lax-Friedrichs flux of eqs. (45)-(47) in conservative form, hybrid Newton (Gauss-Newton Jacobian
  until `|R| < 1e-3`, full Jacobian afterwards, with Armijo backtracking on `|R|`, steps `t = 1, 1/2, ...`, the relaxation `k` of
  eq. 54; a full step that needs `t < 1/4` is replaced by the Gauss-Newton step), Doerfler marking (`theta_ref = 0.25`), quadrature
  degree 12, the isotropic norm of eqs. (49)-(50). A level whose Newton iteration `stalls` (no step above `t = 1e-4` decreases
  `|R|`) is retried once from the initial guess (`eps = 0`, `u0 = 1 - 2x`, `cold_retry`) and, if that fails too on at most
  20000 unknowns, by continuation in `kappa` (solve at `1e-2` and divide `kappa` by 3.16 per stage; `continuation`). Stops at
  `|du|_Vh < 1e-9` or `|R| < 1e-11` (the paper: `|du|_Vh < 1e-6`; the full-Jacobian steps converge quadratically, so the
  stricter test costs about one more iteration; the last increment is in the column `newton_increment`).
* `legacy`: the original choices: the wind `f'(u) = (u, 1)` through the generic upwind form (non-conservative volume term,
  `bh_dar_NL`), damped Newton (`Damping_solve`: `t = 1/(1 + K |R|)`, `omega = 0.5`, `K` x10 on failure, tolerance 1e-12) with
  the Gauss-Newton `bh_prima` Jacobian, no L2 term, SWIP factors 0.5, quadrature degree 8, fixed-fraction marking, no cold retry
  or continuation. The original loop has no cap on `K` and would not terminate on a failed level; here a level is given up when
  `K > 1e14` (status `failK`) or after `maxit = 500` accepted steps (the example's cap, the original value is 20000) and is flagged
  in the history; the best iterate is kept and the run continues. `"newton_fallback": true` retries such a level with the
  hybrid solver (off by default so the comparison is honest). The original code never ran this case; the norm is the
  isotropic one of the paper in both presets (`"norm_form": "streamline"` in `problem.json` gives the old one).

## Parameters of the paper (section 6.5.2, 6.2) and this example

| Item | Paper | This example |
|---|---|---|
| Flux, local dissipation | `f(u) = (u^2/2, u)`, `eta_f = |f'(u).n|`, `f'(w) = [u, 1]` (eqs. 46-47) | same |
| Degree, start mesh | `p = 3`, uniform 4 x 4 | same (crossed triangles) |
| Initial guess, boundary data | `u0 = 1 - 2x`; `u_D = u0` at `y = 0`, `x = 0`, `x = 1`; zero Neumann at `y = 1` | same (outflow term `v f(u).n` at `y = 1`) |
| Kappa | `1e-2`, `1e-3`, `1e-4` (Fig. 16: coarse and fine-scale solutions); refinement levels shown only for `1e-2` (Fig. 17) | same (the figures were not compared pixel by pixel) |
| Norm | `V_h`: SWIP + `||w||_0^2 + sum ||[[w]]||_F^2 + sum h_K ||grad w||_K^2` (eqs. 49-50) | same (before: streamline form with wind `(1, 1)`) |
| Refinement, `theta_ref`, number of levels | not stated for this case (adaptive, estimator `E` of eq. 51) | Doerfler `theta_ref = 0.25`, stops at the DoF cap (18-32 levels) |
| Newton | damped Newton of ref. [73], relaxation `k` (eq. 54); stop at `||ubar_{i+1} - ubar_i||_Vh < 1e-6`; warm start `(eps, ubar)` from the previous level | hybrid Newton with backtracking relaxation, warm start, stop at `1e-9` / `1e-11` (stricter) |

## Measured behaviour (Mac, `p = 3`, cap 50000 DoF, `python run.py --full`)

Paper preset (all levels, `|R|` and `|du|_Vh` of the last iteration in the CSV):

| kappa | levels | Newton converged | iterations per level | shock x (y = 0.75) | steepest slope (analytic `-1/(2 kappa)`) | width (FWHM of `-u_x`, analytic `3.5 kappa`) | `ubar` range on the line / nodal | E last / slope vs DoF^(1/2) | time, peak memory |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | 32 | 32/32 | 2-6 | 0.500 | -48.7 (-50) | 0.036 (0.035) | [-1.000, 1.000] / [-1.000, 1.000] | 7.6e-5 / -2.7 | 15 s, 1.4 GB |
| 1e-3 | 20 | 20/20 | 3-61 (the 61 is the first mesh, by continuation) | 0.500 | -515 (-500) | 0.0035 (0.0035) | [-1.000, 1.000] / [-1.003, 1.003] | 2.6e-3 / -2.4 | 20 s, 1.4 GB |
| 1e-4 | 18 | 18/18 | 5-81 (first mesh and level 3) | 0.500 | -4442 (-5000) | 0.0004 (0.00035) | [-1.09, 1.17] / [-1.06, 1.10] | 3.4e-2 / -0.2 (pre-asymptotic) | 22 s, 1.5 GB |

`kappa = 1e-4`, per level (paper preset): 953 DoF: slope -18 (width 0.08); 2173: -195 (0.010); 3391: -384 (0.0048); 5772: -734 (0.0025);
10241: -1346 (0.0013); 14075: -2442 (0.0007); 32087: -4647 (0.0004); 41367: -4986 (0.0004); 53334: -4442 (0.0004). The layer is
resolved from about 32000 DoF on (width within 15% of `3.5 kappa`, slope within 10% of the analytic one); the steepest
slope of a `p = 3` field oscillates by about 10% between levels because the root reached depends on the path. Extending the cap
(`--dofmax 60000`) adds a level with 63774 DoF that needs the cold retry (108 iterations, 97 s) and gives -3957 (21% below).

* `kappa = 1e-2`: oscillation free (`ubar` within 0.001% of `[-1, 1]`), slope within 3% of the viscous profile; the mean cell
  size away from the shock (`|x - 0.5| > 0.25`) is 5.9 times the one within 0.05 of it (10.6 times at `kappa = 1e-3`, 30 times at `1e-4`).
* `kappa = 1e-4`: the first two levels (953 and 1191 DoF) are unresolved and reach `+-1.7` nodally; from level 3 on the nodal
  range stays within `[-1.18, 1.20]` (6-20% overshoot, `[-1.06, 1.10]` at the last level); the zero crossing is at `x = 0.5000`
  (levels 3-8 sit at 0.4915-0.4977, the root reached after the cold retry is slightly asymmetric).

Legacy preset (damped Newton, wind form, fixed-fraction marking, strict tolerance 1e-12), same cap:

| kappa | levels | converged (`failK`) | levels with `R < 1e-6` / `|du|_Vh < 1e-6` | steepest slope | time |
|---|---|---|---|---|---|
| 1e-2 | 21 | 18/21 | 21 / 18 | -48.7 | 10 s |
| 1e-3 | 18 | 12/18 | 15 / 16 | -498 | 130 s |
| 1e-4 | 14 | 3/14 | 9 / 10 | -4227 (width 0.00037, 59070 DoF, x = 0.5001) | 136 s |

The failed legacy levels are real non-convergence of the original criterion (the original loop would not terminate), but most
end at residuals of 1e-7 to 1e-10 with increments below the paper's `1e-6`; the legacy preset reaches the same resolved shock.

### Time and memory of `--full`

Paper preset: 15 + 20 + 22 s = about 1 minute (57 s in one process, one run each on a loaded machine), peak memory about 1.5 GB
(`ru_maxrss`, whole process). Legacy preset: 10 + 130 + 136 s = about 4.5 minutes, at most 1.7 GB. All runs are under the 8 minute
limit and none needs the workstation. A larger cap costs more for `kappa = 1e-4` (diagnostic driver, paper preset: 97000 DoF after
126 s, 124000 DoF after 180 s) and one level may need the cold retry (+97 s at 64000 DoF).

## Caveats

* Newton is the hard part at small `kappa` on the coarse meshes: the first mesh (4 x 4, 953 DoF) at `kappa <= 1e-3` and one or
  two early levels of `kappa = 1e-4` stall with the plain iteration and need the cold retry or the continuation (these levels
  take 37-81 iterations; the refined levels need 5-17). The `newton_iterations` column of a level that used a retry counts all
  attempts, and the `newton_fallback` column refers to the legacy preset's optional hybrid retry only.
* Several roots exist on a given mesh (the problem is nonlinear); the root reached depends on the warm start or the retry.
  `E` is stable to 3-4 digits between roots, other functionals (maximum of `ubar`, slope) are not: on the 3-level mesh at
  `kappa = 1e-4` the warm start and the cold start give different roots with `ubar` ranges of `[-1.6, 1.4]` and `[-1.4, 1.4]`.
* The steepest slope is measured on 4001 points in a window of 0.01 around the maximum of the 2001-point line, the width on 4001
  points in a window of 0.1 (the layer at `kappa = 1e-4` is thinner than the 5e-4 spacing of the coarse line: measured there
  the slope saturates near -2000 whatever the mesh).
* The legacy preset differs from the original in the points listed above (`K` cap, `maxit`, no FEniCS 2019 `refine`).

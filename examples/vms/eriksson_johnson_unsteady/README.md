# Unsteady Eriksson-Johnson problem: BE and BDF2 with adaptive refinement in space

`u_t + b . grad u - kappa Delta u = 0` on `(-1, 0) x (-1/2, 1/2)`, `b = (1, 0)`, `kappa = 1e-2` (the legacy tests
`test_unsteady_ErikssonJhonssonBE` / `BDF2`: `dt = 0.005`, `T = 0.1`, 4 x 4 crossed start mesh, `MAX_REF = 23`). The exact solution is the steady
layer plus a transient `exp(-2 t)(exp(l1 x) - exp(l2 x))`; both layers (width about `kappa`) sit at the outflow `x = 0`. Dirichlet data on
the top, bottom and outflow, Neumann data `kappa grad u . n` on the inflow `x = -1`; initial condition = exact solution. Every time step
solves the residual-minimisation saddle-point problem of `adaptdg.vms` for `c1 I + c2 L` and refines the mesh in space (solve, estimate, mark, refine)
until `E <= tol`, `max_levels` solves or the DoF cap. The paper behind the legacy tests (Giraldo & Calo, Math. Comput. Appl. 28 (1), 7, 2023) was
not available; the discretisation follows the legacy code (see `docs/VMS.md`, "Unsteady Eriksson-Johnson"). Both error targets are reported
(exact solution, and its DG(p) L2 projection with the suffix `_proj`) for `ubar`, `theta` (dG solution of the step), `u = ubar + utilde` and `phi`.

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                          # short: p = 2, 8 steps, BE and BDF2, adaptive + 2 uniform meshes, about 45 s, < 1 GB
    python run.py --preset legacy          # the same with the choices of the original implementation
    python run.py --full                   # p = 1, 2, 3, 20 steps (T = 0.1), 3 uniform meshes: 7 min, 1.2 GB (paper preset)
    python run.py --integrator bdf2 --degrees 3 --steps 10 --no-uniform
    python plot.py                         # time_p*.png, final_p*.png, meshes_p*_*.png in out/<preset>/

Every run prints `run i/N` and one line per step (`step k/M`, scheme, solves, DoF, `E`, errors, time). `out/<preset>/` (git-ignored) holds
`steps_p<p>_<integrator>_<adaptive|uniform<n>>.csv` (one row per step: `ndofs`, `levels`, `E`, `E_level0`, the errors `L2`/`Vh` of the four fields in both
targets, cumulative `work`, time), `levels_*.csv` (one row per solve), `mesh_*_s<k>.npz` (first, middle and last step) and `summary.json`
(final-time errors, time-integrated errors `int_*` and maxima `max_*`, DoF range, solves per step, clustering, time, peak memory). Settings are in
`problem.json` and `numerics.json` (`UnsteadySettings`: `p`, `integrator`, `dt`, `t_final`, `tol`, `max_levels`, `dofmax`, `quadrature_degree`,
`preset`, ...; the block `example` holds the degrees, uniform meshes, caps and the per-degree tolerances).

## Presets

* `paper` (default): each step starts from the final mesh of the previous one (so the mesh only refines), Doerfler marking, per-step tolerance
  `E <= tol` (`2e-3`, `3e-4`, `3e-5` for `p = 1, 2, 3`) with at most 40 solves, history `u = ubar + utilde`, quadrature degree 12.
* `legacy`: every step restarts from the 4 x 4 mesh and refines `max_levels = 23` times (no tolerance, as `MAX_REF = 23` in the legacy tests, capped by
  the DoF limit 40000 instead of `dofmax = 2e6`), fixed-fraction marking, history `phi` projected onto CG(p) in the Gram norm, quadrature degree 8,
  SWIP factors 0.5. This is much more expensive (every step repeats the whole refinement): about 4 s per step at `p = 2` for 12000 DoF.

## Measured behaviour (Mac, `--full`, paper preset, `dt = 0.005`, `T = 0.1`)

| p | integrator | DoF (solves in step 1 / total) | L2 of ubar at T: exact / proj | Vh of ubar | Vh(phi)/Vh(theta) exact / proj | L2 in time | time |
|---|---|---|---|---|---|---|---|
| 1 | BE | 45327 (29 / 48) | 9.8e-4 / 8.2e-4 | 7.2e-2 | 1.08 / 0.72 | 3.0e-4 | 25 s |
| 1 | BDF2 | 45327 (29 / 48) | 9.1e-4 / 7.3e-4 | 7.2e-2 | 1.08 / 0.72 | 2.9e-4 | 25 s |
| 2 | BE | 28833 (24 / 43) | 4.0e-4 / 4.0e-4 | 1.1e-2 | 1.08 / 0.80 | 8.9e-5 | 21 s |
| 2 | BDF2 | 28833 (24 / 43) | 1.5e-4 / 1.3e-4 | 1.1e-2 | 1.09 / 0.80 | 4.8e-5 | 21 s |
| 3 | BE | 28446 (28 / 47) | 3.7e-4 / 3.7e-4 | 1.9e-3 | 1.03 / 0.91 | 7.5e-5 | 24 s |
| 3 | BDF2 | 28446 (28 / 47) | 3.5e-5 / 3.2e-5 | 1.7e-3 | 1.04 / 0.90 | 1.2e-5 | 25 s |

("L2 in time" = `sqrt(sum_n dt e_n^2)` of the `ubar` error against the exact solution; the whole run takes 417 s including the uniform references.)

* The refinement happens in the first step (24-29 solves until `E <= tol`), where the history is the initial condition; the later steps need one solve
  each (a few in step 2) because the layers do not move, so the DoF per step is constant (28000-45000). Cells at the outflow layer are 18-23
  times smaller than in the inflow half.
* Error against uniform meshes at the same final time (BDF2; DoF of the uniform meshes 3600-60000): `p = 2` adaptive 1.5e-4 with 28833 DoF, uniform 8.4e-3 with
  18529 and 4.6e-3 with 32897; `p = 3` adaptive 3.5e-5 with 28446 DoF, uniform 1.7e-3 with 33553 and 7.0e-4 with 59585 (`final_p*.png`).
* BE is first order in time and BDF2 second order: at `dt = 0.005` the BE error is the time error (3.7e-4 at `p = 3`, 4.0e-4 at `p = 2`) and the BDF2 error is
  limited by the space error at `p = 1, 2`; the orders measured on a fixed fine mesh are in `tests/test_vms_ej_unsteady.py` (0.97 / 0.98 for BE, 2.02 / 1.90 for BDF2).
* `Vh(phi)/Vh(theta)` is 1.03-1.09 against the exact solution (phi is not better there) and 0.72-0.91 against the projection, as in the stationary cases.
  `Vh` is the steady Gram norm of the error (independent of `dt`).
* The `L2` error of `ubar` grows slowly in time for BE (time error accumulating) and saturates for BDF2 (`time_p*.png`).

## Memory, time and caveats

* Short run 45 s and 0.7 GB; `--full` (paper) 7 min and 1.2 GB (peak, at `p = 3` with 60000 DoF uniform); `--full --preset legacy` was not run in full: the 6-step
  `p = 2` run with a 12000 DoF cap took 24 s (BE) and 30 s (BDF2); with the cap 40000 and 20 steps it should take about 4 min per run, about 30 min for
  the whole batch (estimate, to be run on the workstation).
* The first step's accuracy is carried forever by the history (the layers do not leave the domain within `T = 0.1`), so `tol` must be met in step 1; a
  step 1 stopped after 8 levels (2800 DoF at `p = 2`) left an `L2` error of 6.7e-3 that is still 5e-4 at step 6.
* `theta` is the dG solution of each step from the same history; the legacy marched a separate dG sequence. BDF2 starts with one adaptive BE step; the legacy recomputed
  that BE step on every level of step 2.
* Serial runs only; the quadrature degree 12 of the data is needed for the layers.

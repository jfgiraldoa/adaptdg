# Stationary residual-minimisation solver (`adaptdg.vms`)

`adaptdg.vms` solves stationary scalar diffusion(-advection) problems

    -div(kappa grad u) + beta . grad u = f   in Omega,   u = u_D on the Dirichlet boundary

in 2D and 3D, with a discontinuous Galerkin (dG) discrete dual norm and an H1-conforming
trial space, following Giraldo & Calo (2023), *Comput. Methods Appl. Mech. Eng.* 417, 116285.
The estimator drives an adaptive loop (solve, estimate, mark, refine). Importing the package does
not load dolfinx; the solver needs dolfinx 0.11 with PETSc and MUMPS and runs in serial.

## Equations implemented

| Paper | Where | Content |
|---|---|---|
| (8) | `dgforms.DGForms.b` | SWIP diffusion form, weights and harmonic mean `gamma_kappa` (scalar or tensor `kappa`) |
| (9) | `dgforms.DGForms.b` | upwind advection form (inflow boundary term `bn (u_D - u) v`) |
| (10) | `dgforms.penalty_field` | penalty `eta_K = (p+1)(p+d)/d A(dK)/V(K)` as a DG0 field |
| (11) | `dgforms.DGForms.rhs` | source and weak (Nitsche) Dirichlet data |
| (12)-(14) | `dgforms.DGForms.gram` | Gram form of the dG norm (L2 + jump/streamline upwind norm + SWIP norm) |
| (18) | `saddle.RMSolver` | saddle point `g(w,r) + b(w,u) + b(r,v) = l(w)` on `DG(p) x CG(p)` |

Per-cell indicators are `g_K = ||r||^2_{Vh,K}` (a DG0-weighted Gram form) and the global
estimator is `E = sqrt(sum g_K)`. Marking is the Doerfler rule of the paper (`theta_ref` 0.25 in
2D, 0.125 in 3D, tie tolerance 0.2) through `adaptdg.marking.dorfler_union`.

Notes on the implementation: the SWIP norm uses `kappa |grad w|^2` and `eta gamma_kappa` for the
jump term (energy form); Dirichlet conditions are imposed on the tagged boundary faces only
(`Problem.dirichlet_tags`), other boundary faces are natural; errors use a high-degree
quadrature of the exact solution (default 12) so singular data do not create an error floor.

## Multiscale reconstruction (section 4)

`reconstruction.reconstruct(result, adjoint_ordering="paper")` returns a `VMSResult(ubar, utilde, ucheck, phi,
theta, u_full)`; all functions except `ubar` (CG(p)) live in the dG space DG(p) of the saddle point, and one
factorisation of `b(v, .)` serves all right-hand sides.

| Paper | Where | Content |
|---|---|---|
| (30) | `utilde` | fine scale, `b(v, utilde) = g(v, eps')`; `u_full = ubar + utilde` equals the dG solution `theta` (Remark 1, checked to round-off) |
| (32) | `ucheck`, `phi` | adjoint reconstruction `b(v, ucheck) = g(v, eps') + b(eps', v)`, `phi = ubar + ucheck` |
| (35) | | equivalent form `ucheck = utilde + uacute`, `b(v, uacute) = b(eps', v)` |
| (6) | `theta` | dG baseline `b(v, theta) = l(v)` |

`adjoint_ordering="paper"` puts `eps'` in the test slot of `b` (the adjoint operator, as written in eqs. 32 and 35);
`"legacy"` uses `b(v, eps')`, as the original implementation did. They differ only when advection is present.
`norms.error_norms_vms` returns the (L2, Vh) errors of `ubar`, `theta`, `u` and `phi`; setting `reconstruct`
(and `adjoint_ordering`) in `numerics.json` adds `L2_theta, Vh_theta, L2_u, Vh_u, L2_phi, Vh_phi` to the loop history
(and their slopes to `run_from_json`). Without it the history is unchanged.

Measured behaviour: `phi` has a smaller L2 error than the dG solution (ratio 0.4-0.95 depending on the case). In
the Vh norm `phi` is smaller than `theta` only when the error is measured against the DG(p) L2 projection of the
exact solution, as the original implementation did; against the exact solution `phi` is 1.1-1.26 times worse
than `theta` with the same slope. Both targets are reported by default (see "Error targets").

## Error targets

`norms.error_targets` returns, for `ubar` and (with `vms`) `theta`, `u`, `phi`, the L2 and Vh errors against
`exact` (the exact solution) and `projection` (its L2 projection onto DG(p), solved with MUMPS using the quadrature
degree of the error). History columns: `L2, Vh, L2_theta, ...` against the exact solution and the same names with the
suffix `_proj` against the projection (`LoopSettings.error_targets = "exact"` skips the projection).

## Other cases

`Heterogeneous` (section 5.4, eqs. 39-40) and `Anisotropic` (section 5.5, eq. 42), see
`examples/vms/heterogeneous/` and `examples/vms/anisotropic/`; the 3D Fichera corner has its own section below. Note on eq. (38): the paper text prints
`beta = (0, 1)` but the exact solution (39)-(40) is a function of `x` solving `-(eps u')' + u' = 0`, so the
implementation uses `beta = (1, 0)`. The diffusivity of `Heterogeneous` is a DG0 tensor field (the SWIP weights need
different values on the two sides of a facet) and its start mesh needs an even number of cells per side so that
`x = 1/2` consists of mesh facets (preserved by bisection). `refinement = "uniform"` reproduces the original
uniform runs. Linear problems: Dirichlet boundaries only (the nonlinear forms add outflow faces, see the Burgers section).

## Fichera 3D

`Fichera3D` (section 5.3): `-Delta u = f` on (-1,1)^3 minus [0,1)^3 with `u = r^q` (default `q = 1/3`, also 0.1), `f` derived
symbolically (`-q(q+1) r^(q-2)`), `meshes.fichera(n)` (even `n`), `theta_ref = 0.125`. See `examples/vms/fichera_3d/`
(short and `--full` runs, both presets, both error targets, uniform sequence of structured meshes). With adaptive refinement the
errors approach the optimal slopes (-p for Vh, -(p+1) for L2 against DoF^(1/3)); at p = 1 and 5e4 DoF the measured slopes
are about -1.1 and -2.0, at p = 2 and 1e5 DoF about -2.3 and -3.7 (pre-asymptotic: the DoF cap is small in 3D), and uniform
refinement is suboptimal (Vh about -(q + 1/2)). `reconstruct` (theta, u, phi) works in 3D. Memory: about 1 GB at 5e4 DoF
(p = 1) and 2.5 GB at 1e5 DoF (p = 2). The legacy test used marking ratio 0.125 in 3D, whereas the `legacy` preset has the 2D
value 0.25: the example sets `"preset_overrides": {"marking_ratio": 0.125}`.

## Eriksson-Johnson (steady)

`ErikssonJohnson` (section 5.6, legacy `ErikssonJhonsson`, steady branch): `-kappa Delta u + (1, 0) . grad u = 0` on the unit
square, `kappa = 1e-2` by default, Dirichlet data on the four sides (tags 1-4) from
`u = (exp(r1 (x-1)) - exp(r2 (x-1)))/(exp(-r1) - exp(-r2)) sin(pi y)`, `r1,2 = (1 +- sqrt(1 + 4 pi^2 kappa^2))/(2 kappa)`
(`f = 0`; boundary layer of width about `kappa` at `x = 1`). `ErikssonJohnson3D` is the 3D version of the paper on the unit
cube (`beta = (1, 0, 0)`, same solution, six faces); the original code only ran the 2D case. Registry names
`"eriksson_johnson"` and `"eriksson_johnson_3d"`, `meshes.unit_cube(n)`. See `examples/vms/eriksson_johnson/` (short and
`--full` runs, both presets, both error targets, adaptive and uniform; `--dim 3`). Measured (about 22k DoF, kappa = 1e-2,
paper preset): the mean cell size away from the layer is 12 times that in `x > 0.9`; the last-window Vh slope is about
-1.3 (p = 1) and -2.2 (p = 2) against DoF^(1/2); at kappa = 1e-3 the uniform sequences do not resolve the layer.

## Nonlinear problems: Burgers equation (section 6)

`adaptdg.vms.nonlinear` solves `div(f(u)) - div(kappa grad u) = s` (eq. 43) with the same adaptive loop idea as the linear
solver; `problems.NonlinearProblem` is the interface (`flux(u)`, `dflux(u)`, `d2flux(u)`, `initial_guess`, a source
derived from the exact solution when there is one, `outflow_tags`, `has_exact`). The norm wind of the Gram form (eq. 50,
legacy `beta_inf = 1`) is `Problem.beta`, a constant `(1, 1)`; it is not a convection velocity.
`numerics.json` accepts `newton` (overrides of `nonlinear.NewtonOptions`) and `newton_fallback`.

| Paper | Where | Content |
|---|---|---|
| (44)-(46) | `nonlinear.NonlinearForms.operator` | residual `N(v; u) = n_h(v; u) - l_h(v)`: SWIP diffusion, `-(grad v, f(u)) + ([[v]], Phi)` with the Lax-Friedrichs flux (interior `eta_f` = maximum of `|f'(u_i).n|` over both sides, Dirichlet faces `|f'(u).n|`) |
| (48) | same | weak Dirichlet data and source; the part `-(v, f(u_D).n - eta_f u_D)/2` of the boundary flux is evaluated with the same `eta_f(u)`, so it is differentiated together with `n_h` |
| -- | same | faces with no Dirichlet data (`outflow_tags`): `v f(u).n` (natural outflow, zero diffusive flux) |
| (49)-(50) | `dgforms.DGForms.gram` | norm: SWIP norm plus `||w||_0^2 + sum ||[[w]]||_F^2 + sum h ||grad w||_K^2` (norm wind `(1, 1)`) |
| (51) | `nonlinear.NonlinearRMSolver` | saddle point `g(w, eps) + N(w; ubar) = 0`, `n'(eps, v; ubar) = 0` on `DG(p) x CG(p)`; Jacobian by `ufl.derivative` (checked by finite differences in `tests/test_vms_burgers.py`) |
| (53)-(54) | `nonlinear.newton_solve` | Newton iteration, see below; stopping at `|du|_Vh < 1e-9` or `|R| < 1e-11` (the paper says `1e-6`) |
| (61)-(62) | `nonlinear.reconstruct_nonlinear` | `utilde`: `n'(v, utilde; ubar) = g(v, eps)`; `ucheck`: `... = g(v, eps) + n'(eps, v; ubar)`; `phi = ubar + ucheck`; `theta` = nonlinear dG solution by Newton from `ubar + utilde` |

Warm start: `(eps, ubar)` are interpolated to the next mesh (`create_interpolation_data` / `interpolate_nonmatching`).
A level where Newton fails is flagged in the history (`newton_status`, `newton_converged`) and the loop continues with the
iterate of smallest residual; with `newton_fallback` such a level is retried with the hybrid solver.

**Newton choices and why.** Unknowns `(eps, ubar)`; the residual is `F1 = g(w, eps) + N(w; ubar)` (rows of the `DG(p)` test
functions) and `F2 = n'(eps, v; ubar)` (rows of the `CG(p)` ones); `|R|` is the Euclidean norm of both.
* *Hybrid* (paper preset): Gauss-Newton Jacobian (the second derivative `n''` of `F2` is dropped) while `|R| >= 1e-3`, the full
  Jacobian afterwards, undamped. The legacy damping together with the Gauss-Newton Jacobian fails on most levels of the
  isotropic case because the dropped term is not small away from the solution, and the damped iteration then stops
  at local minima of `|R|` where the Jacobian is singular; the undamped full Newton step escapes them. At small kappa the
  single-component case needs the damping (undamped iterations stagnate): `damping_fallback` (default on) restarts a failed
  level once from the same state with the legacy damping and `omega = 0.1`.
* *Damped* (legacy preset): `Damping_solve` of the original code, `t = 1/(1 + K |R|)`, `K` x10 when `om = (1 - |R_new|/|R|)/t < 0.5`
  and /10 after a success, tolerance 1e-12, `maxit` 20000; the original has no bound on `K`, here `K > 1e14` ends the level
  (`failK`).

**Presets.** `Preset.nonlinear_flux`: `"lax_friedrichs"` (paper, conservative) or `"wind"` (legacy `bh_dar_NL`: the wind
`beta = f'(u)` through the generic upwind form with the non-conservative volume term `v beta.grad u`; its Jacobian is the legacy
`bh_prima`, i.e. `v (beta.grad du + du f''(u).grad u)` with upwind faces frozen at `beta(u)`). `Preset.newton_method`:
`"hybrid"` or `"damped"`. The adjoint ordering and the other fields behave as in the linear case. The legacy code's
automatic-differentiation branch (`matrix_assembler.py`) builds the second block of the Newton matrix as a derivative of `n(v; u_k)`
with the test and trial slots exchanged; it is not reproduced (both presets use `n'(eps, v; ubar)` of eq. 51).

**Several roots.** On one mesh the discrete problem has several solutions. The root reached depends on the path (warm start,
damping), and `||u - ubar||_0` changes by up to 50% between them while the estimator `E` agrees to 3-4 digits. Report `E` first;
read single-level errors with care.

### Single-component Burgers (6.5.2)

`BurgersSingle` (`"burgers_single"`): `f(u) = (u^2/2, u)` on (0,1)^2, `u_D = 1 - 2x` on `y = 0, x = 0, 1` (tags 3, 1, 2),
outflow `y = 1` (tag 4), initial guess `1 - 2x`, no exact solution, `kappa = 1e-2` (also `1e-3`, `1e-4`), `p = 3`, 4 x 4 start mesh.
The shock is at `x = 1/2` (the paper text says `x = 0`: `u0 = 1 - 2x` changes sign at `1/2`), steepest slope about `-1/(2 kappa)`.
Without the outflow term `v f(u).n` Newton fails at `p = 3`. See `examples/vms/burgers_single/` (README: measured Newton
statistics, shock position and width, time and memory).

## Galerkin baseline (Poisson FEM)

`galerkin.solve_galerkin(problem, mesh, p)` solves the standard conforming `CG(p)` Galerkin problem on the same `Problem`
classes (strong Dirichlet data on the tagged boundary; for advection it is not stabilised, no SUPG) and
`galerkin.galerkin_errors` returns its (L2, Vh) errors with the same two targets and the same Gram-form Vh norm as the
residual-minimisation solver, so the numbers are comparable. `LoopSettings.method = "galerkin"` runs the loop with it
(uniform refinement only: the legacy baseline had no indicator, hence no `E` column) and
`compare_galerkin = true` adds `L2_cg, Vh_cg` (and `_proj`) columns to a residual-minimisation loop (Galerkin on the same
meshes). `PoissonSine` (`"poisson_sine"`) is the smooth legacy test (`-Delta u = 10 sin(pi x) sin(pi y)`); on it Galerkin
converges at p + 1 (L2) and p (Vh), and P1 on the 10 x 10 mesh gives ||u_h|| = 0.2472 (legacy reference 0.247140).
See `examples/vms/poisson_fem/`. Measured on the re-entrant corner (30k-100k DoF): on adaptive meshes CG and `ubar` have the
same Vh error (ratio 1.00 +- 0.02) and slopes (-1.0 at p = 1, -2.2 at p = 2 against DoF^(1/2)); `phi` has the smallest
L2 error and a Vh error 1.1-1.5 times larger than CG against the exact solution.

## Modules

`problems` (`Problem`, `NonlinearProblem`, `ReentrantCorner`, `Fichera3D`, `Heterogeneous`, `Anisotropic`, `ErikssonJohnson`,
`ErikssonJohnson3D`, `PoissonSine`, `BurgersSingle`), `nonlinear` (`NonlinearForms`, `NonlinearRMSolver`, `newton_solve`, `reconstruct_nonlinear`, `NonlinearAdaptiveLoop`), `galerkin` (`solve_galerkin`, `galerkin_errors`), `meshes` (crossed-triangle L-shape, Fichera cube, unit cube, boundary tags), `dgforms`, `saddle` (`RMSolver`, `RMResult`), `refine`
(`refine_marked`, `refine_uniform`, any dimension), `norms` (`error_targets`, `error_norms`, `error_norms_vms`), `preset` (`Preset`, `resolve_preset`), `reconstruction` (`reconstruct`, `VMSResult`), `loop`
(`AdaptiveStationaryLoop`, `LoopSettings`, `fitted_slope`) and `run` (`run_from_json`).

## Example

`examples/vms/reentrant_corner/` runs the 2D re-entrant corner (singular solution
`r^(2/3) sin(2 theta/3)`): `python run.py` (short) or `python run.py --full`, then `python plot.py`.
With adaptive refinement the dG-norm error decays like DoF^(-p/2) and the L2 error like
DoF^(-(p+1)/2) for p = 1..4, while uniform refinement stalls at DoF^(-1/3).

## Programmatic use

    from adaptdg.vms.run import run_from_json
    out = run_from_json("problem.json", "numerics.json", "out", p=2)
    out["history"], out["slopes"]

## Legacy and paper presets

`numerics.json` key `"preset": "paper" | "legacy"` (default `"paper"`, the behaviour of all releases so far).
`legacy` reproduces the choices of the original implementation; every row is one field of
`adaptdg.vms.preset.Preset`, so mixed variants are possible (`"preset_overrides": {"gram_l2": true}`).
Legacy references are file:line in the `asfem-legacy` tag.

| Item (`Preset` field) | Legacy value | Paper value | Package |
|---|---|---|---|
| L2 term in the Gram form of pure diffusion (`gram_l2`) | absent: `gamma_norm = 0`, `src/dg_forms.py:98`, `Benchmark/TestCases.py:85,130,187,233` | `(a, c)` term of eq. (13) | paper: included; legacy: omitted (with advection always included, `beta_inf = 1`, `dg_forms.py:113`) |
| Interior penalty harmonic mean in `b` (`gamma_factor_b`) | `k1 k2/(k1+k2)`, `dg_forms.py:70,196` | `2 k1 k2/(k1+k2)`, eq. (8) | paper: 1.0; legacy: 0.5 |
| Same in the Gram jump term (`gamma_factor_gram`) | `k1 k2/(k1+k2)`, `dg_forms.py:119` | `2 k1 k2/(k1+k2)`, eq. (14) | paper: 1.0; legacy: 0.5 |
| Quadrature degree of data and error norms (`quadrature_degree`) | 8 (`Degree`, `add_default_parameters.py:179`) | not specified | paper: `LoopSettings.quadrature_degree` (12); legacy: 8 |
| Marking (`marking`, `marking_ratio`) | `REF_TYPE 2` fixed-fraction bands of the maximum indicator (frac 0.95, step 0.05) until the marked sum reaches `0.25 * total` (`refinement.py:58-74`, `add_default_parameters.py:88`) | Doerfler, `theta^2 * total` with `theta_ref` 0.25 (2D), tie tolerance 0.2 | paper: `dorfler_union`; legacy: `marking.fixed_fraction_bands` (`REF_TYPE 0`, uniform, is `refinement = "uniform"` in both) |
| Adjoint term of the reconstruction (`adjoint_ordering`) | `b(v, eps')`, `ASFEM_main.py:524`, `matrix_assembler.py:104` | `b(eps', v)`, eqs. (32), (35) | paper: `"paper"`; legacy: `"legacy"` (differs only with advection) |
| Error target | `project(u_exact, DG(p))` for all errors, `postprocess.py:387-392` | exact solution | both targets reported (previous section) |
| Not switchable (equal or not reproducible) | `eta_0 = 1`, `tolref = 0.2`, `eta_adv = 1`, `epsilon = -1` (`add_default_parameters.py`); marking refines with FEniCS 2019 `refine` (not all edges of the marked cells); `MAX_REF 10`, `dofmax 1e6` | `eta_0 = 1`, `tolref = 0.2` | the same values in both presets; the mesh refinement routine and stopping rules are the package ones |

## Known differences from the paper

* `beta = (1, 0)` is used in the heterogeneous case because the exact solution needs it; the paper prints `(0, 1)`.
* The Gram form uses the energy form `kappa |grad w|^2` in eq. (14), as the legacy code did.
* In the Vh norm the reconstruction `phi` is 1.1-1.26 times worse than `theta` against the exact solution, and
  better only against the DG(p) L2 projection of the exact solution (the target of the legacy postprocessing);
  see "Error targets" and "Legacy and paper presets".

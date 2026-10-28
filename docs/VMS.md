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
uniform runs. First release: Dirichlet boundaries only (no Neumann).

## Fichera 3D

`Fichera3D` (section 5.3): `-Delta u = f` on (-1,1)^3 minus [0,1)^3 with `u = r^q` (default `q = 1/3`, also 0.1), `f` derived
symbolically (`-q(q+1) r^(q-2)`), `meshes.fichera(n)` (even `n`), `theta_ref = 0.125`. See `examples/vms/fichera_3d/`
(short and `--full` runs, both presets, both error targets, uniform sequence of structured meshes). With adaptive refinement the
errors approach the optimal slopes (-p for Vh, -(p+1) for L2 against DoF^(1/3)); at p = 1 and 5e4 DoF the measured slopes
are about -1.1 and -2.0, at p = 2 and 1e5 DoF about -2.3 and -3.7 (pre-asymptotic: the DoF cap is small in 3D), and uniform
refinement is suboptimal (Vh about -(q + 1/2)). `reconstruct` (theta, u, phi) works in 3D. Memory: about 1 GB at 5e4 DoF
(p = 1) and 2.5 GB at 1e5 DoF (p = 2). The legacy test used marking ratio 0.125 in 3D, whereas the `legacy` preset has the 2D
value 0.25: the example sets `"preset_overrides": {"marking_ratio": 0.125}`.

## Modules

`problems` (`Problem`, `ReentrantCorner`, `Fichera3D`, `Heterogeneous`, `Anisotropic`), `meshes` (crossed-triangle L-shape,
Fichera cube, boundary tags), `dgforms`, `saddle` (`RMSolver`, `RMResult`), `refine`
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

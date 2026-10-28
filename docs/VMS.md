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
| (9) | `dgforms.DGForms.b` | upwind advection form |
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

## Modules

`problems` (`Problem`, `ReentrantCorner`, `Fichera3D`), `meshes` (crossed-triangle L-shape,
Fichera cube, boundary tags), `dgforms`, `saddle` (`RMSolver`, `RMResult`), `refine`
(`refine_marked`, `refine_uniform`, any dimension), `norms` (`error_norms`), `loop`
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

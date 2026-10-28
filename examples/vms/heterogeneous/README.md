# Heterogeneous diffusion-advection (2D)

`-div(kappa grad u) + beta . grad u = 0` on (0,1)^2 with `kappa = diag(eps_i, 1)`, `eps1 = 0.1` for
`x < 1/2` and `eps2 = 1` for `x > 1/2`, `beta = (1, 0)` and Dirichlet data from the 1D exact solution
(Giraldo & Calo 2023, CMAME 417, 116285, section 5.4, eqs. 39-40). `kappa` is a cell-wise constant
(DG0) tensor, so the SWIP weights and the harmonic mean see the jump; the start mesh has an even number
of cells per side and bisection keeps `x = 1/2` a set of mesh facets.

Note on the data: the paper text prints `beta = (0, 1)`, but the solution of eqs. (39)-(40) depends on
`x` only and solves `-(eps u')' + u' = 0`, which needs `beta = (1, 0)`. This example uses `beta = (1, 0)`.

Besides the coarse solution `ubar` (CG(p)) the run reconstructs, on DG(p), the full-scale solution
`u = ubar + utilde` (eq. 30; identical to the dG solution `theta`) and the adjoint multiscale
reconstruction `phi = ubar + ucheck` (eqs. 32 and 35, `adjoint_ordering = "paper"`; `"legacy"` uses
`b(v, eps')` instead of `b(eps', v)`).

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py          # short: uniform p = 1, 2, 3 (4x4 start) + adaptive p = 1, 2, about 30 s
    python run.py --full   # uniform up to ~9e5 DoF, adaptive p = 1..3 up to 2e5 DoF, about 3 min
    python plot.py         # out/errors_uniform.png, out/errors_adaptive.png, out/final_mesh.png

`out/` holds one CSV per run (level, ndofs, ncells, E, L2, Vh, then L2/Vh of theta, u and phi, time).

## Expected behaviour and measured slopes

Slopes of the error against DoF^(1/2) over the last levels of the short run (uniform refinement):

| p | Vh (ubar / u / phi) | L2 u / phi | L2 ubar |
|---|---|---|---|
| 1 | -1.04 / -1.03 / -1.04 | -1.95 / -1.94 | -1.97 |
| 2 | -2.09 / -2.04 / -2.04 | -3.09 / -3.08 | -2.65 (loses the optimal -3) |
| 3 | -3.17 / -3.16 / -3.15 | -4.09 / -4.07 | -4.12 |

* For even p the coarse L2 error of `ubar` loses the optimal rate; `u` and `phi` recover it.
* `phi` has the smallest L2 error on the same mesh (about 0.75-0.8 of the dG error); its Vh error is
  about 1.1-1.2 times the dG one (the paper reports an improvement; not reproduced in this norm).
* `u = ubar + utilde` and the dG solution `theta` have the same errors (equal to round-off).

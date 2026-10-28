# Strongly anisotropic diffusion (2D)

`-div(kappa grad u) = f` on (-1,1) x (-1/2,1/2) with `kappa = diag(1, 1/r)`, `r = 1e4` and `1e6`,
`tau = 1e-3` and the manufactured Gaussian `u = exp(-(x^2 + r tau y^2)) sqrt(r tau)/(4 pi)` (paper section 5.5,
eq. 42; the normalisation constant is the one of the original implementation and is irrelevant). `f` and
the Dirichlet data are derived from `u` symbolically. The solution has an inner layer of width
`1/sqrt(r tau)` around `y = 0`. The residual-minimisation loop is run with Doerfler marking
(theta_ref = 0.25) from a 4x4 start mesh, and `ubar`, `u = ubar + utilde` (= dG) and the adjoint
reconstruction `phi` are compared (see `../heterogeneous/README.md`).

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py          # short: r = 1e4, 1e6 with p = 1, 2 up to 20000 DoF, about 15 s
    python run.py --full   # p = 1..3 up to 150000 DoF, about 2 min
    python plot.py         # out/r1e04/ and out/r1e06/: errors.png and final_mesh.png

## Expected behaviour and measured results

Slopes against DoF^(1/2) over the last levels (20000 DoF; they approach p and p+1 as the layer is
resolved, and are still pre-asymptotic for r = 1e6):

| r | p | Vh (ubar / u / phi) | L2 (ubar / u / phi) |
|---|---|---|---|
| 1e4 | 1 | -0.8 / -0.8 / -0.8 | -1.05 / -1.0 / -1.0 |
| 1e4 | 2 | -2.2 / -2.2 / -2.1 | -2.9 / -2.8 / -2.6 |
| 1e6 | 1 | -0.6 / -0.65 / -0.6 | -0.9 / -0.9 / -1.0 |
| 1e6 | 2 | -1.6 / -1.6 / -1.6 | -1.8 / -1.8 / -1.8 |

* With 150000 DoF (`--full`) the Vh slopes of `u` are about -0.9, -1.85, -2.9 (r = 1e4) and -0.8, -1.8, -3.9
  (r = 1e6) for p = 1, 2, 3 and the L2 slopes about -1.1, -1.9, -3.8 (r = 1e4); r = 1e6 is still pre-asymptotic.
* The mesh concentrates around `y = 0`: for r = 1e6 (p = 1) about 40 % of the cells and of the marked
  cells lie in `|y| < 2/sqrt(r tau)`, a band of 13 % of the area. For r = 1e4 the layer is wide
  (that band is the whole domain) and the concentration is mild.
* `phi` is a little better than the dG solution in L2 (about 0.85-0.95) and a little worse in Vh (1.07-1.17).

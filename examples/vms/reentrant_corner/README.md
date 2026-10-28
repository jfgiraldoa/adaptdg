# Re-entrant corner (2D Laplace)

Laplace equation on the L-shaped domain (-1,1)^2 minus the quadrant x>0, y<0 with the exact
solution u = r^(2/3) sin(2 theta/3) (theta = atan2(y, x) mod 2 pi), Dirichlet data on the whole
boundary. The solution is singular at the re-entrant corner (the origin), so uniform refinement
converges slowly while adaptive refinement recovers the optimal rates.

The discrete problem is the residual-minimisation saddle point on DG(p) x CG(p)
(Giraldo & Calo 2023, CMAME 417, 116285, eq. 18); the error indicator is the norm of the
residual representation and cells are marked with the Doerfler rule (theta_ref = 0.25).

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py          # short: p = 1, 2 adaptive + p = 1, 2 uniform, about 5 s
    python run.py --full   # p = 1..4 adaptive up to 40000 DoF + uniform, about 20 s
    python plot.py         # writes out/errors.png and out/final_mesh.png

Settings live in `problem.json` and `numerics.json` (see `adaptdg.vms.run`); `out/` holds one CSV
per run (level, ndofs, ncells, E, L2, Vh, time) and the final meshes.

## Expected behaviour

* The mesh clusters at the origin and stays coarse elsewhere.
* Adaptive: the dG-norm error (Vh) and the estimator E decay like DoF^(-p/2) and the L2 error
  like DoF^(-(p+1)/2) (slopes -p and -(p+1) against DoF^(1/2)); in the full run the measured Vh
  slopes are about -1.0, -2.1, -3.1, -4.4 for p = 1..4, and L2 is a little faster than p+1.
* Uniform: both errors are limited by the singularity (Vh slope about -2/3 for every p).

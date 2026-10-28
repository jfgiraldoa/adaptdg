# Poisson FEM baseline versus residual minimisation (re-entrant corner)

Standard conforming Galerkin CG(p) (the original `test_poisson_FEM_solution`, there P1 on a 10 x 10 mesh with
`f = 10 sin(pi x) sin(pi y)`, reproduced by the problem `poisson_sine`) used as the baseline for the residual-minimisation
solution on the singular problem of `examples/vms/reentrant_corner` (Laplace on the L-shaped domain, exact solution
`r^(2/3) sin(2 theta/3)`, Dirichlet data imposed strongly for Galerkin and weakly, as always, in the saddle point).
Both methods run on the same meshes and are measured with the same L2 and Vh (Gram-form) norms, against the exact
solution and against its DG(p) L2 projection. The question: what does the residual-minimisation machinery add to plain
Galerkin on this problem? The Galerkin baseline has no error estimator (the original used none), so it only runs on
the meshes produced by the residual-minimisation loop (adaptive or uniform) or on uniform refinement.
For advection problems `solve_galerkin` is the unstabilised Galerkin method (no SUPG) and oscillates at high Peclet number.

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: p = 1, 2, about 6000-9000 DoF, ~15 s, < 0.5 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # p = 1, 2, 3 up to 1e5 DoF
    python plot.py                 # compare_p<p>.png in out/<preset>/

Per degree three runs (printed as `run i/N`): `rm_adaptive` and `rm_uniform` (residual minimisation; the columns
`L2_cg, Vh_cg, ..._proj` are CG(p) Galerkin on the same mesh) and `galerkin_uniform` (`"method": "galerkin"`, plain
Galerkin only, uniform refinement). `out/<preset>/` holds `history_p<p>_<run>.csv` and `summary.json` (slopes over the
last 2x window of DoF, Galerkin / residual-minimisation ratios on the last mesh, time, memory).
`--full` is estimated from the 30k-DoF runs (1-9 s, 0.3-1.2 GB per run) at about 6 minutes per preset and at most 5 GB.

## Measured (mid-size run, 30k-100k DoF, Mac)

* Adaptive, slopes vs DoF^(1/2) (paper preset): p = 1 Vh -1.03 for both CG Galerkin and ubar, L2 about -2.1 (CG) and
  -1.8 (ubar); p = 2 Vh -2.2, L2 -3.5 for both; the optimal rates are recovered by both methods because the adaptive mesh
  resolves the corner. Uniform: Vh -0.67 and L2 -1.4 for CG and ubar (the singularity limits both).
* On the same adaptive mesh the CG and ubar errors are nearly equal (Vh_cg/Vh_ubar 1.00 +- 0.02, L2_cg/L2_ubar 0.77 at
  p = 1, 1.02 at p = 2); on uniform meshes CG is slightly better in Vh (ratio 0.93-0.96, 1.04-1.09 against the projection) and worse in L2
  (L2_cg/L2_ubar 1.3 at p = 1, 1.8 at p = 2).
* phi (adjoint reconstruction) has the smallest L2 error (L2_cg/L2_phi 1.2-1.3 adaptive, 3-8 uniform); in Vh against the
  exact solution phi is 1.1-1.5 times worse than CG (Vh_cg/Vh_phi 0.75-0.88), against the DG(p) projection it is
  better (ratios 1.5-4.5).
* So on this problem plain CG(p) on the same meshes is as accurate as ubar; the benefit of the method is the estimator
  that produces the mesh (and the dG-based reconstructions), not a better CG-conforming solution.

## Caveats

* Single window slopes of the stepped adaptive histories fluctuate by a few tenths.
* The Galerkin baseline has no residual indicator: there is no "Galerkin adaptive" run.
* The `legacy` preset only changes the residual-minimisation forms, marking and quadrature; the Galerkin solve is the same.

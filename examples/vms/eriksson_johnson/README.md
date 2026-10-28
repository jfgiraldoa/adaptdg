# Steady Eriksson-Johnson problem (boundary layer)

Advection-diffusion problem -kappa Delta u + beta . grad u = 0 on the unit square, beta = (1, 0), f = 0,
Dirichlet data on the four sides (tags 1-4) from the exact solution

    u = (exp(r1 (x-1)) - exp(r2 (x-1))) / (exp(-r1) - exp(-r2)) * sin(pi y),
    r1,2 = (1 +- sqrt(1 + 4 pi^2 kappa^2)) / (2 kappa)

(paper section 5.6; the original code `ErikssonJhonsson`, steady branch). The solution has a boundary layer of
width about kappa at the outflow x = 1, so uniform meshes need h << kappa there. The default kappa = 1e-2 is the
value of the paper and of the original test (start mesh 4 x 4 crossed triangles, 5 refinements, fixed-fraction
marking); kappa = 1e-3 is added to show the layer resolution (adaptive meshes put cells of size about kappa at x = 1).
The paper also defines a 3D version on the unit cube with beta = (1, 0, 0), the same exact solution (independent of
z) and Dirichlet data on the six faces; it is available as `--dim 3` (`name: "eriksson_johnson_3d"`). The original
code only ran the 2D problem.

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: kappa = 1e-2, p = 1, 2, adaptive and uniform, about 20 s, < 1 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # kappa in {1e-2, 1e-3} x p in {1, 2, 3} x {adaptive, uniform}
    python run.py --dim 3          # 3D variant, p = 1 (about 15 s, 1.4 GB; --full: 100000 DoF)
    python plot.py                 # errors_k*.png and mesh_k*.png in out/<preset>/

Every run prints `run i/N` with the elapsed time. `out/<preset>/` holds one CSV per run
(`history_k<kappa>_p<p>_<refinement>.csv`: level, ndofs, ncells, E, time and the L2/Vh errors of ubar, theta, u and
phi, both targets), `mesh_*.npz` (last mesh) and `summary.json` (slopes over the last 2x window of DoF, time, peak
memory, ratio Vh(phi)/Vh(theta), cell-size statistics in the layer x > 0.9 versus x < 0.5). `--full` caps
(`numerics.json`, `example.dofmax`): p = 1: 60000 DoF, p = 2: 120000, p = 3: 150000; measured on the Mac at 20-45k DoF:
5-18 s and 0.3-1.3 GB per run, so `--full` takes about 8 minutes per preset and at most about 4 GB.
Settings live in `problem.json` and `numerics.json` (the usual `LoopSettings` keys plus the block `example`).

## Presets and error targets

* `paper` (default): Gram form with the L2 term, SWIP penalty as in eqs. (8) and (14), Doerfler marking
  (theta_ref = 0.25 in 2D, 0.125 in 3D), quadrature degree 12.
* `legacy`: the choices of the original implementation (see "Legacy and paper presets" in `docs/VMS.md`); the
  fixed-fraction ratio is the original 0.25 in 2D and 0.125 in 3D (`example.presets_3d`).
* Every row has two targets: the error against the exact solution (`L2`, `Vh`, `L2_phi`, ...) and against its DG(p)
  L2 projection (suffix `_proj`), which is what the original implementation measured.

## Measured behaviour (Mac, paper preset unless stated)

* Adaptive refinement clusters cells at the outflow: the mean cell size at x < 0.5 is 12-13 times that at x > 0.9 for
  kappa = 1e-2 and 21-34 times for kappa = 1e-3 (20-25k DoF); the uniform sequences are still in the pre-asymptotic
  range at kappa = 1e-3 (the layer is not resolved, errors barely decrease).
* Slopes against DoF^(1/2), last 2x window (adaptive, kappa = 1e-2, about 22k DoF): p = 1 Vh about -1.3 and L2 about
  -2.4 against -1 and -2 expected; p = 2 Vh about -2.2 (-2 expected), L2 about -1.4 to -2.0; the adaptive histories are
  stepped, so single-window slopes fluctuate by a few tenths.
* Vh(phi)/Vh(theta) against the exact solution is 1.0-1.15 (phi is not better than theta there) and 0.6-0.9 against the
  DG(p) projection, as for the other cases (see `docs/VMS.md`, "Error targets").

## Caveats

* The 3D variant is strongly pre-asymptotic at kappa = 1e-2 with 2e4 DoF (the layer needs h of about 0.01): slopes of
  about -0.8 are trends, not rates.
* The uniform runs refine by halving, so their last levels overshoot `dofmax` (4-5 levels).
* The legacy preset reproduces the forms, marking rule and quadrature, not the mesh refinement or the solver of the
  original test (FEniCS 2019 `refine`), so level-by-level numbers differ from the original.

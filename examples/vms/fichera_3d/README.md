# Fichera corner (3D Poisson)

Poisson problem -Delta u = f on the Fichera domain (-1,1)^3 minus [0,1)^3 with the exact solution
u = r^q (r = distance to the origin, q = 1/3 by default, q = 0.1 also available), Dirichlet data on the
whole boundary and f = -q(q+1) r^(q-2) (paper section 5.3). The origin is a re-entrant corner where u is
singular and, for q <= 1/2, f is not in L2, so the quadrature degree of the data matters (the error moves
by about 2% between degrees 4 and 16).

The claim of the paper: with the residual-minimisation indicator and Doerfler marking
(theta_ref = 0.125 in 3D) the mesh clusters at the corner and the optimal rates are recovered,
Vh error ~ DoF^(-p/3) and L2 error ~ DoF^(-(p+1)/3) (slopes -p and -(p+1) against DoF^(1/3)), whereas
uniform refinement is limited by the singularity (slopes about -(q + 1/2) in the dG norm).

## Run (needs dolfinx 0.11 with PETSc/MUMPS; matplotlib for the plots)

    python run.py                  # short: p = 1, q = 1/3, adaptive, about 40000 DoF, ~10 s, ~1 GB
    python run.py --preset legacy  # the same with the choices of the original implementation
    python run.py --full           # q in {0.1, 1/3} x p in {1, 2} x {adaptive, uniform}, about 4 min per preset
    python run.py --full --degrees 1 2 3 --dofmax 110000   # p = 3 is optional (about 1 min more, ~4 GB)
    python plot.py                 # errors_q*.png and cell_size_q*.png in out/<preset>/

Every run prints `run i/N` with the elapsed time. `out/<preset>/` holds one CSV per run
(`history_q<q>_p<p>_<refinement>.csv`: level, ndofs, ncells, E, time and the L2/Vh errors of ubar, theta, u
and phi), `cells_*.npz` (distance to the origin and size of each cell of the last mesh) and `summary.json`
(slopes over the last 2x window of DoF, time, peak memory of the process, clustering statistics).
`--full` caps (numerics.json, `example.dofmax_full`): p = 1: 150000 DoF (about 20 s, 2 GB); p = 2: 100000 DoF
(about 30 s, 2.5 GB); runs are sequential in one process, so the peak is that of the largest one (about 3 GB).

Settings live in `problem.json` (`q`, `q_full`, `initial_mesh_size` = 2) and `numerics.json`
(the usual `LoopSettings` keys plus the block `example` read by `run.py`: DoF caps per p, marking parameters
per preset, step of the uniform mesh sequence). `"reconstruct": true` adds the errors of theta (dG baseline),
u = ubar + utilde and phi (adjoint reconstruction); it works in 3D at p = 1, 2 and costs about 10% in time and memory.

## Presets and error targets

* `paper` (default): Gram form with the L2 term, SWIP penalty as in eqs. (8) and (14), Doerfler marking
  (theta_ref = 0.125, tie tolerance 0.2), quadrature degree 12.
* `legacy`: the choices of the original implementation (see "Legacy and paper presets" in `docs/VMS.md`):
  no L2 term in the Gram form, halved harmonic mean in the penalties, quadrature degree 8, fixed-fraction
  bands marking. The original Fichera test used ratio 0.125 (`REFINE_RATIO`, `test_steady_Fichera_vms`),
  so `numerics.json` overrides the preset's 2D ratio 0.25 with `marking_ratio` 0.125.
* Every row has two targets: the error against the exact solution (columns `L2`, `Vh`, `L2_phi`, ...) and
  against its DG(p) L2 projection (suffix `_proj`), which is what the original implementation measured.
  The Vh error against the projection is slightly smaller and its slope somewhat lower than against the
  exact solution.

## Expected behaviour

* Adaptive: the mesh clusters at the corner (mean cell size near the origin about 8 times smaller than far
  away at 5e4 DoF, p = 1) and the errors recover the optimal rates, but in 3D the asymptotic range is reached
  late: at p = 1 the last-2x-window slopes are about -1.1 (Vh) and -2.0 (L2) at 5e4 DoF (paper preset, q = 1/3);
  at p = 2 about -2.3 and -3.7 at 1e5 DoF against -2 and -3 expected.
* Uniform: Vh about -(q + 1/2) vs DoF^(1/3) and L2 about -1.7 for q = 1/3, independent of p.

## Caveats

* The DoF cap is pre-asymptotic for p >= 2 (the 3D meshes are small): slopes are reported over the last 2x
  window of DoF and the p = 2 and 3 values should be read as trends, not as asymptotic rates.
* Adaptive histories are stepped (marking adds cells in bursts), so slopes of single windows fluctuate by a few
  tenths, more so with the legacy preset, which reaches the cap in fewer, larger steps.
* The uniform runs solve the structured meshes n = 2, 4, 6, ... (more points than halving h at 8x cost per level).
* The legacy preset reproduces the forms, marking rule and quadrature, not the mesh refinement or the solver of
  the original test (unstructured start mesh with characteristic length 0.2, FEniCS 2019 `refine`, an iterative
  solver and a quadrature-element forcing), so level-by-level numbers differ from the original.

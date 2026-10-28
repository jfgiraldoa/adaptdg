# adaptdg design notes

## Dependency rule
Applications depend on adaptdg, never the reverse. adaptdg never imports chemistry or any
application. dolfinx is imported lazily, so `import adaptdg` works without it.

## Provenance
The transport, transfer, marking and loop modules were extracted unchanged from the ASRTM-PHREEQC
code (numerically identical, verified against the originals). The VMS module (`adaptdg.vms`) is a
new FEniCSx implementation of the AS-FEM / variational multiscale method of the cited papers
(see `docs/VMS.md`). The AS-FEM code on legacy FEniCS 2019 is kept in the git tag `asfem-legacy`.

## Modules

- `adaptdg.vms`: stationary adaptive residual-minimisation solver and VMS reconstruction
  (`docs/VMS.md`).
- `adaptdg.transport`: 1D DG1/CG1 transport (`ConservativeDG1D`, residual-minimisation transport
  with adaptive cells). Unit conventions are those of the original solvers and are not harmonised:
  each solver documents which units its arguments use, and the caller supplies them consistently.
- `adaptdg.transport_2d`, `adaptdg.transport_dg2d`, `adaptdg.rm_transport`: 2D residual-minimisation
  transport, general 2D conservative DG1 with tensor dispersion, and the 2D RM step. The error
  indicator (residual representation in the dual DG norm) is built into the step.
- `adaptdg.rm_1d`, `adaptdg.loop_1d`: 1D residual-minimisation operator and adaptive loop
  (numpy/scipy only).
- `adaptdg.flow`: Darcy RT0-DG0 solver and `FlowState`.
- `adaptdg.transfer`, `adaptdg.transfer_2d`: conservative remap of DG1 fields between meshes.
- `adaptdg.marking`: Dörfler marking core `dorfler_union` (minimum size, epsilon and tie hooks,
  optional post-mark filter) and adapters for the 1D, 2D, MoMaS and 1D-loop variants. The variants
  differ in tie rules, eligibility and normalisation; they are kept as they are, and defaults
  (epsilon, eligibility factor, tie tolerance) are passed by the caller.
- `adaptdg.mesh`, `adaptdg.loop`: persistent-mesh policy and the 2D adaptive loop
  (`run_adaptive_loop_2d`). Chemistry enters only through callbacks (`ChemistryCallbacks2D`:
  `snapshot` and `react` required, `seed_fields` and `solid_inventory` optional, plus a `boundary`
  dict); numerical settings live in `LoopSettings2D`.
- `adaptdg.f2a`, `adaptdg.history_projection`, `adaptdg.history_projection_fast`: history-preserving mesh
  rebuild (F2A), `run_adaptive_loop_2d(..., history_policy="f2a")`. With the DG1 reaction history, the
  default policy restarts the history from chemistry-grid averages at every predictive-mesh rebuild and
  after every in-step refinement. F2A (P1) projects the current and older DG1 histories exactly (L2,
  DG1 to DG1, conservative per new triangle) from the old persistent mesh to the rebuilt one and applies a
  nonnegative scaling that keeps the mean; (P2) gives the children of an in-step refinement the exact
  restriction of the parent history; (P3) rebuilds the last committed mesh when a run is resumed exactly at
  a rebuild step, so that P1 has an old mesh to project from. `F2AOptions` switches each change.
- `adaptdg.fastadap`: cost-only fast path of the 2D loop (`fast=True`): solver and factorisation reuse on the
  persistent mesh, vectorised history bookkeeping and transfer plans, multi-RHS back-solve (with F2A),
  vectorised `mesh_triangles` and a refinement without the MPI-communicator leak. The plans are built through
  `adaptdg.transfer_2d.use_plan_builder`. Results agree with the plain path to round-off, not bit for bit.
- `adaptdg.loop_momas`, `adaptdg.coupled_momas`: adaptive and uniform coupled 2D drivers; the
  chemistry is a duck-typed protocol (`ChemistryMomas`, `EngineMomas`) that is documented, not enforced.
- `adaptdg.case4`: evolving-porosity 1D transport (Xie et al. benchmark B2): ADAP variants, BDF2,
  DG history, positivity fixes, variable time step, FV baseline and the step-level coupling
  `advance_case4_step`. The backends read only `cells`, `components` and `inlet` from the chemistry
  object, and `advance_case4_step` calls `chemistry.react` once per macro step; the porosity model
  enters through the `Case4Model` hook. The command-line driver, history storage and checkpointing
  are not part of the package.

## Piece-by-piece notes

- The 1D chemistry protocol (`Chemistry1D`) needs `snapshot.concentrations`, `boundary[c]` and
  `react(inputs, time_step=dt)` returning an object with `concentrations`. A uniform 1D mesh
  without adaptivity is `Adap1DSettings(initial_cells=N, first_step_solves=1, step_solves=1,
  coarsen_tol=-1.0, positivity="mixing")`.
- The residual-minimisation estimator of the transport solvers is embedded in their `step`, so
  `adaptdg.estimators` is still a stub; the same holds for `AdaptiveTransport`, `adapt_mesh`,
  `transport_step` and `transfer_field`.
- Planned interface (not implemented): `FlowState(mesh, theta_old, theta_new, q)` and
  `AdaptiveTransport`, which owns mesh adaptation (estimate, mark, refine/coarsen, transfer) while
  the application owns the time loop and any chemistry or sources.

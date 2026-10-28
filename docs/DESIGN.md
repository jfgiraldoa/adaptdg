# adaptdg design notes

## Dependency rule
Applications depend on adaptdg (apps -> adaptdg), never the reverse. adaptdg
has no chemistry or application physics. dolfinx is imported lazily so the
package imports without it.

## Planned interface
- `FlowState`: `mesh`, `theta_old`, `theta_new`, `q` (Darcy flux, RT0).
- `AdaptiveTransport`: owns mesh adaptation (estimate, mark, refine/coarsen,
  transfer). The app owns the time loop and any chemistry/sources.
- Registered fields (`dg1`, `cell_average`) are transferred conservatively
  whenever the mesh changes. `water_volume()` is the conservation diagnostic.

## Extraction order
1. Conservative mesh transfer. **Done** (`transfer`, `transfer_2d` extracted from
   ASRTM-PHREEQC at commit a152b91; `transfer_field` stays a stub).
2. DG1 transport. 1D part (piece 5a) **done**, see below; 2D part (5b) pending.
3. Estimators (pending) and Dörfler marking (**marking done**, see below).
4. ADAP loop.

## Piece 2: Dörfler marking (done)
Extracted from ASRTM-PHREEQC at commit a152b91 into `adaptdg.marking`. The
stub `dorfler_mark` was removed (replaced by `dorfler_union`; api test updated).
Generic core `dorfler_union(indicators, sizes, min_size, theta, epsilon, tie_tol,
*, eligibility=None, post_mark_filter=None)`; `post_mark_filter(marked)` generalises
`history_marking_filter` (the app closes over its extra arguments).
- Adapters, `np.array_equal` to the source on 4000 seeded random cases each
  (1-3 species, scalar/array sizes, exact ties, empty eligible sets):
  `dorfler_union_2d` (`_dorfler_union_2d`), `dorfler_union_momas`
  (`momas2d/adap.py::dorfler_union`), `dorfler_union_1d_adap`
  (`Adap1DRun._marking`), `dorfler_mark_1d` (`porosity/transport_1d.py`).
- Verbatim copy: `dorfler_with_ties` (`case4dev/adap_asfem.py`); cannot be
  expressed by the core.
- Differences between variants (kept, not harmonised): `dorfler_mark_1d` has
  no tie extension (`tie_tol=None`), eligibility `width > minimum_width*(1+1e-12)`,
  validates inputs and does not take abs (inputs must be nonnegative);
  2D/momas/adap1d use strict `size > factor*chem_width`, abs values and tie rule
  `cutoff - (1+tau)*v < 0`, target `theta**2*total`. `dorfler_with_ties`:
  target `fraction*total` (no square), cumulative-sum `searchsorted`, tie rule
  `v >= (1-tie)**2 * v_stop` (inclusive, also before the stop), no epsilon rule,
  eligibility as an explicit mask, total taken over all cells.
- Defaults (epsilon 0.1 / 0.5 / 0.2, eligibility factor, tie tolerance) stay
  per-case and are passed by the caller; none is defined in adaptdg.

## Piece 5a: DG1 transport, 1D (done)
Extracted from ASRTM-PHREEQC at commit a152b91, numerics and public names unchanged.
- `adaptdg._linear.ReusableLinearProblem` = `reusable_linear_problem.py` (verbatim; dolfinx/petsc4py already lazy).
- `adaptdg.transport` holds two source modules as sections: A = `dg_transport_1d.py`
  (`nonnegative_means`, `ConservativeDG1D`), B = `porosity/transport_1d.py` (projection helpers,
  `gradient_jump_indicator_1d`, `conservative_positive_projection_1d`, `PorosityTransportSettings`,
  `AdaptiveTransportState1D`, `ResidualMinimizationCase4Transport1D`, `Case4AdaptiveTransportResult`,
  `solve_adaptive_transport_step`). The `transport_step` stub stays (not implemented; the two
  solvers have no common interface and none was invented). `adaptdg/__init__` and `__all__` unchanged.
- Unit conventions (NOT harmonised): A has no units inside; `porosity` is a per-cell storage
  coefficient multiplying the time derivative (phi), `dispersion` a per-cell diffusivity, `darcy_flux`
  a flux, all supplied by the caller consistently (physical-order arrays, left-to-right flow). B uses
  m, s, m/s, mol/L for concentrations and mol/m3 bulk (factor 1000 inside); porosity per cell with
  storage `phi/reference_porosity` and velocity `q/reference_porosity`, so the solver's
  concentration unknown is per pore volume scaled by `phi0`; zero dispersion; the 1000 factor and the
  mol/L vs mol/m3 convention are hard-coded in `ResidualMinimizationCase4Transport1D.step`.
- Porosity treatment: A multiplies storage by `phi` on the old and new state (heterogeneous phi
  in space, constant in time); B projects `phi` from the chemistry grid to the transport cells
  (`project_piecewise_constant`) and keeps it constant in time (time-varying porosity is not here).
- `dorfler_mark_1d` is imported from `adaptdg.marking` (the adapter proven `np.array_equal` to the
  source on 4000 cases; re-checked below); `conservative_remap_1d` from `adaptdg.transfer`.
- Estimator: the residual-minimisation estimator is embedded in `ResidualMinimizationCase4Transport1D.step`
  (the mixed-formulation residual is the indicator), not a separable function, so it stays in
  `transport` and `adaptdg.estimators.cell_indicators` remains a stub. `gradient_jump_indicator_1d`
  (numpy only) is in `transport` too, not moved.
- Equality: throwaway script, source vs adaptdg, `np.array_equal` on 1197 arrays, 0 differences;
  source run twice also identical (deterministic). Cases: 30 random `nonnegative_means` inputs, 10 sets
  of random projection/indicator/Dörfler/positive-projection inputs; 8 `ConservativeDG1D` setups (uniform and
  refined non-uniform meshes 10-60 cells, D 1e-7..0.5, porosity scalar/array, positive components
  none/(0,2)/(0,1,2), with/without source, 4 steps incl. a time-step change); 3 adaptive
  `solve_adaptive_transport_step` runs (uniform and non-uniform chemistry grids, 2-3 components incl. Charge).
  Limit: the positivity correction inside `ConservativeDG1D.step` was never triggered (0 in all cases);
  `nonnegative_means` is covered by direct calls only.
- Deviation list: only the import lines (to `adaptdg`), `transport_step` stub kept above the copied
  code, and `# noqa: E702` on one source line for ruff.

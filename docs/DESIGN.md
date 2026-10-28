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
2. DG1 transport. 1D (piece 5a) and 2D/Darcy (piece 5b) **done**, see below. Piece 5 is complete.
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

## Piece 5b: 2D DG1 / residual-minimisation transport and Darcy flow (done; piece 5 complete)
Extracted from ASRTM-PHREEQC at commit a152b91 (source repo `asrtm-phreeqc-public`), copied, numerics and public names unchanged.
- `adaptdg.transport_2d` = `src/asrtm_phreeqc/stabilized_transport_2d.py` (mesh helpers
  `create_rectangle_mesh_2d`, `mesh_triangles`, `create_mesh_from_triangles_2d`, `triangle_characteristic_sizes`,
  `refine_marked_triangles(_with_parent)`, and `ResidualMinimizationBackwardEuler2D`, `StabilizedTransportStep2D`).
  Whole file verbatim.
- `adaptdg.transport_dg2d` = `methods/momas2d/transport.py` (`dispersion_tensor`, `ConservativeDG2D`, the general
  2D conservative DG1 with tensor dispersion, weighted SIP, vertex limiter).
- `adaptdg.rm_transport` = `methods/momas2d/rm_transport.py` (`RMStep`, `RMTransportMoMaS2D`). It is a complete
  solver (the estimator, the Riesz-norm residual indicator, is built into `step`, as in 5a), so it is not in
  `estimators` (still a stub); it is a sibling of `transport_2d` and generalises `ResidualMinimizationBackwardEuler2D`.
- `adaptdg.flow`: the real Darcy RT0-DG0 code of `methods/momas2d/darcy.py` (`solve_darcy`, `solve_momas_darcy`,
  `DarcySolution`, `create_momas_mesh`, `material_tags`, `cell_field`, `momas_facet_tags`, `momas_flux_bc`,
  `cell_divergence`, `boundary_fluxes`, `interior_facet_flux_mismatch`, `interface_head_jump`, `section_flux`, MoMaS
  constants) is placed after the unchanged `FlowState` stub; the original module docstring is kept as a comment block.
- Not copied: front2d monkey patches (fastadap, rm_class_patch, wall_bc, wall_seeds), case drivers, chemistry.
  Nothing extracted depends on chemistry or on `porosity`. `adaptdg/__init__` and `__all__` unchanged;
  `import adaptdg` does not load dolfinx (checked: `'dolfinx' in sys.modules` is False); all dolfinx/ufl/basix/petsc4py/mpi4py
  imports in the sources were already inside functions, so no import line was moved.
- Duplicates (kept, not merged): no function name occurs in two of the new modules. Conceptual duplicates:
  `ResidualMinimizationBackwardEuler2D` (transport_2d, scalar D, constant velocity, fixed inlet corners) vs
  `RMTransportMoMaS2D` (rm_transport, tensor D, RT0 q, inlet facet tags; the latter declares itself a generalisation of the
  former, with an extra `dt r^2` term in the Riesz norm); `ConservativeDG2D` vs `ConservativeDG1D` (5a) and the
  case-4 1D classes. `refine_marked_triangles*` and the mesh helpers live only in `transport_2d`; `flow.create_momas_mesh`
  builds its own mesh with `mesh.create_rectangle`. `nonnegative_means` is imported from `adaptdg.transport` (5a), not copied.
- Unit/porosity conventions (NOT harmonised). `transport_2d`: no porosity (phi = 1), scalar `diffusion`, constant
  `velocity` tuple, time in the caller's unit, serial only, inlets on the left (`y >= left_inlet_y_min`) and top
  (`x <= top_inlet_x_max`) edges, CG1 trial / DG1 test, history degree 0 or 1, BE or BDF2.
  `transport_dg2d` and `rm_transport`: equation `d_t(phi c) + q.grad c - div(D grad c) = 0`, `q` is the Darcy flux (not divided by
  phi), `D = a_T|q| I + (a_L - a_T) q q^T/|q|` with `q` the Darcy flux, `phi`, `a_L`, `a_T` per cell (DG0 Function or UFL scalar),
  concentrations per unit pore volume, no unit factor inside, serial only; `ConservativeDG2D` reports per-step inventory defects.
  `flow`: `K` is a hydraulic conductivity (q = -K grad h), head `h`, flux boundary `q.n = g` imposed strongly (MoMaS inlets
  -0.0225, walls 0), Dirichlet head on the outlet; the MoMaS geometry, materials (porosity, K, sites) and tags are module
  constants (`LX, LY, MATERIALS, WALL, INLET1, ...`), i.e. `flow.py` holds the MoMaS-specific helpers beside the generic `solve_darcy`.
  No canonical convention is decided here.
- Deviations from verbatim (all in `transport_dg2d`/`rm_transport`/`flow`/tests): `transport_dg2d` lost the source
  `SRC = ... parents[2] / "src"; sys.path.insert(...)` block (a path hack tied to the source repo layout) and its `Path`/`sys` imports; the
  import of `nonnegative_means` now points to `adaptdg.transport`; `rm_transport` imports `dispersion_tensor` from `adaptdg.transport_dg2d`;
  `flow.py` gained `field` and `numpy` imports at top (FlowState unchanged). Docstrings still mention `asrtm_phreeqc` names in prose.
- Tests: `test_transfer_2d.py` gets back `test_refined_parent_map_matches_final_local_cell_order` (needs `transport_2d`, guarded with
  `importorskip("dolfinx")`); new `test_flow_darcy.py` (adapted from `tests/momas2d_rt0_nesting.py`, assertions instead of prints; adds
  `create_connectivity(1, 2)` so `compute_midpoints` works on dolfinx 0.11) and `test_transport_2d.py` (import laziness, refine, `ConservativeDG2D` balance,
  `RMTransportMoMaS2D` smoke). Not ported: `tests/test_transport_history_2d.py` (depends on `transport_comparison_2d`, not extracted),
  `tests/momas2d_anchor_tests.py` (chemistry); `momas2d_rm_mms.py` and `test_efficient_adaptivity_2d.py` do not exist in the source repo.
- Equality: throwaway script outside the repo, source vs adaptdg, `np.array_equal`, 0 differences on 1053 arrays; the source run twice is also
  identical (deterministic). Setups: 8 `ResidualMinimizationBackwardEuler2D` configs (rect 8/16/24 and a refined mesh, D 1e-4..1,
  BE/BDF2, history degree 0/1, 3 steps); mesh/refine on rect 8-24 and two locally refined levels (random marks, parent maps, no-mark case);
  Darcy on MoMaS meshes k=1,2 (+ k=3 geometry), heterogeneous random K on a k=2 mesh and on a locally refined k=1 mesh;
  6 `ConservativeDG2D` setups (3 components, positive (0,2)/none, DG0 and constant porosity, aL 1e-4..1e-1, 3 steps with a dt change, `replace_cell_means`)
  and 12 `RMTransportMoMaS2D` runs (history 0/1, SWIP on/off, 3 steps, `fine_scale`). Limit: serial, dolfinx 0.11 only; the positivity limiter was not forced.

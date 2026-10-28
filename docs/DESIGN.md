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
4. ADAP loop and mesh persistence (piece 6). **Done**, see below.

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

## Piece 6: ADAP loop and mesh persistence with chemistry callbacks (done)
Extracted from ASRTM-PHREEQC at commit a152b91 (`adaptive_calcite_dolomite_2d.py`,
`reactive_transport_policy_2d.py`). Guard result: NOT triggered. The loop touches chemistry in only a few
places, none of them inside a numerical operation, so no operation or order changed.
Chemistry/config touchpoints found in `run_calcite_dolomite_adaptive_2d` and how they became parameters:
| Touchpoint in the source | Replacement |
| --- | --- |
| `load_case(config_path)` (mesh, time, transport, chemistry cells, adaptivity) | `LoopSettings2D` dataclass (`adaptivity` stays a duck-typed object with `theta, epsilon_chem, eligibility_factor, tie_tolerance, initial_cells_2d, maximum_refinements`) |
| `EquilibriumCalciteDolomite2DPhreeqcRM(...)` construction, `chemistry.snapshot` | `ChemistryCallbacks2D.snapshot()` |
| `chemistry.react(inputs, time_step=)` (also in checkpoint replay) | `ChemistryCallbacks2D.react` |
| `chemistry.boundary` | `ChemistryCallbacks2D.boundary` (dict) |
| `Calcite`/`Dolomite` fields seeding the predictive mesh | `ChemistryCallbacks2D.seed_fields(snapshot, shape)` (optional) |
| `_solid_component_inventory` (mass diagnostics) | `ChemistryCallbacks2D.solid_inventory(name, state, shape, areas)` (optional, default 0.0) |
| `COMPONENTS_2D` | `components` argument |
| `PHREEQC_MODEL_FLOOR_MOL_L` | `model_floor` argument of the loop and of `prepare_chemistry_component_2d` |
| record fields `chemistry_molalities, calcite, dolomite, carbonate, ph`; result `porosity` | record keeps `chemistry_result` (the object `react` returned); porosity dropped (engine property) |
Four callbacks (snapshot, react required; seed_fields, solid_inventory optional) plus the `boundary` dict.
Layout: `adaptdg.loop` = `run_adaptive_loop_2d`, `solve_one_adaptive_transport_step_2d`,
`prepare_chemistry_component_2d`, `AdaptiveStep2D`, `AdaptiveRun2D`, `AdaptiveRefinementSnapshot2D`,
`NegativeTransportCorrection2D` (+ the `AdaptiveTransport` stub, unchanged). `adaptdg.mesh` =
`ReactiveTransportPolicy2D`, `reactive_front_seed_regions/points`, `create_predictive_reactive_mesh_2d`,
`persistent_mesh_rebuild_required`, `_triangle_restart_permutation` (+ the `adapt_mesh` stub, unchanged).
`adaptdg.fields` unchanged (stub): the source has no field registry. Dörfler: the loop calls
`adaptdg.marking.dorfler_union_2d(names=components)` instead of a second copy of `_dorfler_union_2d`.
Kept verbatim: all control flow, solve/estimate/mark/refine order, history transfer strategies, restart
(`start_step`, replay inputs, mid-block triangles, DG1 histories), snapshots, timings, mass diagnostics.
Left out: `front2d/fastadap.py`, `rm_class_patch.py`, `wall_bc.py`, `wall_seeds.py`, `checkpoint_2d`,
`history_projection*.py` (only used by those patches), `load_case`/config validation (dimension, method,
`adaptivity.enabled` checks), `COMPONENTS_2D`, the chemistry engine, `_solid_component_inventory`,
`PHREEQC_MG_MODEL_FLOOR_MOL_L` alias.
Deviations (all outside numerics): names (`run_calcite_dolomite_adaptive_2d` -> `run_adaptive_loop_2d`,
`_solve_one_adaptive_transport_step_2d` -> `solve_one_adaptive_transport_step_2d`,
`_prepare_chemistry_component_2d` -> `prepare_chemistry_component_2d`, step/result dataclasses renamed);
config object -> `LoopSettings2D`; the record stores `chemistry_result`; `Path`/`load_case` imports gone.
Equality (throwaway script outside the repo, fenicsx env, dolfinx 0.11, phreeqcrm; case of
`examples/front2d_calcite_dolomite` with chemistry grid 24x24, dt 1500 s, 5x5 initial mesh): source
`run_calcite_dolomite_adaptive_2d` vs adaptdg loop with the chemistry callbacks wrapping (i) the
`asrtm_dg.engines.equilibrium_2d` engine and (ii) the source engine; per step `np.array_equal` on triangles, transported
cell averages/nodal values/indicators, chemistry input/output concentrations, molalities, calcite, dolomite, carbonate,
pH, negative-correction diagnostics, snapshot marked sets, mass diagnostics and work counts (timings and RSS excluded).
Setups (BDF2, 12-14 steps): A legacy 5x5 restart every step (12 steps, mesh updates 60..90 triangles); B persistent mesh
interval 3 + core/halo policy + DG1 reaction-increment history (14 steps, 472..3000 triangles); C persistent policy with
snapshots at step 2 levels 0,1 and step 4 level 0 (marked sets 18, 38, 40 cells; 12 steps); D mid-block restart of B at step 8
with replay inputs, restart triangles and both DG1 histories (6 steps; also identical to the uninterrupted source). A also
with backward Euler (12 steps). 0 differences over 12253 compared arrays/scalars in the BDF2 batch (+3490 for BE A). The
plan cache counters are compared after resetting both caches before every run (they are module state shared by consecutive
runs in one process). Limits: BE + DG1 history cannot run in the source either (ValueError); only the serial 24x24 chemistry
case; the stagnation guard and nested/full transfer strategies of `solve_one_adaptive_transport_step_2d` were not exercised
(the loop never enables them); CG1 degree 1 only.
Tests: `tests/test_loop.py` (fake chemistry, no PHREEQC; dolfinx tests guarded with `importorskip`).

## Piece 7: 1D RM operator and ADAP loop (done)
Extracted from ASRTM-PHREEQC at commit a152b91 (`methods/onedim/rm1d.py` -> `adaptdg.rm_1d`,
`methods/onedim/adap1d.py` -> `adaptdg.loop_1d`). Class and method names, numerics and operation order unchanged
(`RMOperator1D`, `Adap1DRun`, `Adap1DSettings`, `dg1_to_cells`, `cells_to_dg1`, `positivity_fix`, `positivity_mixing`).
Chemistry stays duck-typed exactly as in the source; the `Chemistry1D` protocol class only documents it:
`snapshot.concentrations` (component -> chemistry-grid array, read once), `boundary[c]` (inflow) and
`react(inputs, time_step=dt)` returning an object with `concentrations`. Nothing is enforced and adaptdg imports no
chemistry. Deviations (outside numerics): the operator import points to `adaptdg.rm_1d`; `Adap1DRun._marking` calls
`adaptdg.marking.dorfler_union_1d_adap` (the adapter built from this very method; verified equal against the inline
original in `tests/test_loop_1d.py` and by full runs). A uniform mesh without adaptivity (Engesgaard driver) is
`Adap1DSettings(initial_cells=N, first_step_solves=1, step_solves=1, coarsen_tol=-1.0, positivity="mixing")`.
Both modules need only numpy/scipy; `import adaptdg` does not import them.
Equality (throwaway script outside the repo): source `Adap1DRun` vs `adaptdg.loop_1d.Adap1DRun` with the source cation
exchange chemistry, 60 steps, `np.array_equal` on 1020 arrays (solved edges, cell counts, transported, transported on
the chemistry grid, reacted): 0 differences. Full-case equality through the asrtm-dg glue is documented in asrtm-dg.
Tests: `tests/test_loop_1d.py` (fake decay chemistry, no PHREEQC).

## Piece 8: MoMaS 2D adaptive and coupled loops (done)
Extracted from ASRTM-PHREEQC at commit a152b91 (`methods/momas2d/adap.py` -> `adaptdg.loop_momas`,
`methods/momas2d/coupled.py` -> `adaptdg.coupled_momas`). Two modules because the sources are two independent drivers
(ADAP on the chemistry grid vs the uniform DG1 run); numerics, names and operation order unchanged.
- `loop_momas`: `run` (the driver, name kept), `singularity_cap`, `MeshFields`, `dg1_transfer`, `full_dg1`, `linear_remap`,
  `remap_history`, `POSITIVE`, `SINGULAR_POINTS`; plus `fast_refine_marked_triangles_with_parent` copied verbatim from
  `methods/front2d/fastadap.py` (the leak-free refinement `adap.py` imports; the `fastadap.apply()` patches are not used by
  `adap.py` and not copied). `dorfler_union` is `adaptdg.marking.dorfler_union_momas` (exact adapter; re-verified against the
  inline original on 300 random cases in `tests/test_loop_momas.py` and by full runs). Not extracted: `main` (argparse CLI),
  the `sys.path` hacks. `run`/`run_coupled` gain a required keyword-only `chemistry`.
- `chemistry_fast.py` IS used by both drivers (`FastAnchorMomasPhreeqcRM`, site re-anchoring with one END); it is chemistry, so it
  goes to asrtm-dg (`engines/momas_fast`), not here.
- Chemistry enters through two documented Protocols (documentation only, never enforced; no chemistry import):
  `ChemistryMomas` (`spec` dict with `materials` phi/alpha_* and `time` snapshots/injection_ends; `open_chemistry(materials,
  representative_volume_l, *, concentration_scale, zero_tolerance, anchor_mode)` -> context manager yielding the engine;
  `inlet_mobile(*, concentration_scale)` -> (2, 4)) and `EngineMomas` (`snapshot.total/mobile/fixed`, `water_volume_l`,
  `site_total`, `react(mobile)` -> `total, mobile, fixed, primary, sites, water_mass_kg` (+ `aqueous`, `surface` in the uniform
  run), `dump_state`/`restore_state` for checkpoints). The engine is built inside the loop because the chemistry grid
  (cells, A/B materials, volume) depends on `kc`; the order of operations of the source (engine, initial inventory, inlet
  reservoir) is kept. The constants of the source construction are passed in `run` (1e-6, 1e-16, "one_end").
- Guard result: NOT triggered; adap.py's `run` is separable from its CLI, and the chemistry touchpoints (spec read, engine and
  reservoir construction) sit outside every numerical operation.
- Deviations: imports only (`adaptdg.flow` as `darcy`, `adaptdg.rm_transport`, `adaptdg.transport.nonnegative_means`,
  `adaptdg.transport_2d.triangle_characteristic_sizes`, `adaptdg.transfer_2d.conservative_remap_plan_2d`,
  `adaptdg.transport_dg2d.ConservativeDG2D`); the `model`/`WS` lines replaced as above; a file-level
  `# ruff: noqa: E702, F821, F841` in `loop_momas` (verbatim semicolons and names bound inside the step loop).
- Equality (throwaway script outside the repo, fenicsx python, dolfinx 0.11; source `adap.run` / `run_coupled` called from Python vs
  the asrtm-dg glue on the adaptdg backend; per step `np.array_equal` on RM-step triangles, history, cell averages, indicators,
  inflow/outflow, DG states, Dorfler indicators and marked sets, chemistry inputs/outputs, and on every saved array and stat except
  timings): ADAP kc=2 dispersive dt 0.0125 t=0.1 (the example; 580 items), ADAP kc=2 advective dt 0.025 t=0.1 with rebuild
  every 2, singularity cap, theta 0.6 (585), ADAP kc=2 remap history + mean remap + reactive marking + VMS + kb=1 + cap-before (311),
  uniform kc=2 dispersive square chemistry (172), uniform kc=2 advective triangle chemistry (112): 0 differences; the source run twice
  is identical. Limits: serial, short runs (t <= 0.1), checkpoint/resume path not exercised.
- Tests: `tests/test_loop_momas.py` (fake conserving chemistry; dolfinx tests guarded with `importorskip`).

## Piece 9 (Case 4): evolving-porosity transport and ADAP modules (done, partial by design)
Extracted from ASRTM-PHREEQC at commit a152b91 into the subpackage `adaptdg/case4/` (copy; numerics, public names and operation
order unchanged; provenance in each docstring). A subpackage (not flat modules) because the source modules share short names with
other families (`positivity`, `dg_history`, `adap_*`) and a common `constants`.

### Module map and classification
| source | role | class | adaptdg |
|---|---|---|---|
| `case4dev/adap_persistent.py` | persistent nested-mesh ADAP, DOLFINx operator (`--transport adap`) | TRANSPORT-ADAPTIVE | `case4.adap_persistent` |
| `case4dev/adap_fast.py` | same method, NumPy/SuperLU assembly; positivity (global/local), minmod/FCT chemistry-scale limiters, `UniformFineTransport` | TRANSPORT-ADAPTIVE | `case4.adap_fast` |
| `case4dev/adap_asfem.py` | ASFEM Riesz-norm estimator + `ASFEMAdaptiveTransport`; its `dorfler_with_ties` is the body already in `adaptdg.marking` | TRANSPORT-ADAPTIVE | `case4.adap_asfem` (imports `marking.dorfler_with_ties`, name kept) |
| `case4dev/transport_bdf2.py` | BE/BDF2 substeps for DG (`BDF2DGTransport`) and ADAP (`BDF2AdaptiveTransport`) | TRANSPORT-ADAPTIVE | `case4.transport_bdf2` |
| `case4dev/dg_history.py` | DG1 history (`dg1`/`mean`), minmod, local positivity, reaction update (`HistoryDGTransport`) | TRANSPORT | `case4.dg_history` |
| `case4dev/positivity.py` | local mean redistribution, slope scaling (numpy) | TRANSPORT | `case4.positivity` |
| `case4dev/variable_dt.py` | `next_dt` rule + variable-dt DG1 backend | TRANSPORT | `case4.variable_dt` |
| `case4dev/coarse_transport.py` | DG1 on a coarser grid than the chemistry grid | TRANSPORT | `case4.coarse_transport` |
| `case4dev/fv_transport.py` | first-order upwind FV (DG0), `--history fv0` | TRANSPORT | `case4.fv_transport` |
| `porosity/dg_1d.py` | `PorosityDG1D`, uniform DG1 weak form + MUMPS | TRANSPORT | `case4.dg_1d` |
| `porosity/coupling.py` | `Case4State`, `advance_case4_step` (Strang splitting, midpoint kinetics call, geometry timing), `update_geometry`, `UniformDGTransport`, `AdaptivePorosityTransport` | COUPLING loop (transport + chemistry hook) | `case4.coupling` (`run_uniform_case4` omitted: builds the engine) |
| `case4dev/chemistry_b1.py`, `chemistry_n.py`, `porosity/case4.py` | PhreeqcRM engines (`Case4Chemistry`, B1, N-cell mapping) | CHEMISTRY | asrtm-dg (`engines.case4`), not extracted here |
| `porosity/geometry.py` | `ReactorVolumeBasis`, `evolve_porosity` (numpy) | GEOMETRY | already in asrtm-dg; passed to the loop through the `model` hook |
| `porosity/checkpoint.py` | atomic checkpoint write/read (JSON + npz + gz RAW dump) | DRIVER/IO | not extracted |
| `case4dev/run_case4_dg.py` | argparse, SQLite history, gzip blobs, flock, checkpoint/resume, progress, code hashes | DRIVER/CLI | stays in the paper repo / glue |

Imports re-pointed: `LENGTH_M`/`INITIAL_POROSITY` -> `adaptdg.case4.constants` (2.0, 0.35, benchmark values); `dorfler_mark_1d`
-> `adaptdg.marking`, `gradient_jump_indicator_1d`, `conservative_positive_projection_1d` -> `adaptdg.transport` (adapters/copies
proven equal in pieces 4-5 and re-verified through every ADAP run below); `PorosityDG1D` -> `case4.dg_1d`. File-level
`# ruff: noqa` only in `adap_fast`, `adap_persistent` (E702), `dg_history` (E402, E702), `variable_dt` (F401).

### Chemistry touchpoints (defines the asrtm-dg glue, step 3b)
All transport backends read only three attributes of the chemistry object: `cells` (int), `components` (tuple of names, order =
rows of the concentration array, units mol/L water) and `inlet` (array per component, mol/L). Only
`coupling.advance_case4_step` calls the chemistry: `chemistry.react(concentrations, porosity, minerals_mol_per_reactor, time_s,
dt_s) -> (aqueous, minerals_after, balance)` once per macro step, after substep `strang_substeps // 2`, at `time_s = (step - 0.5) dt`;
`balance[element]` has `relative` and `absolute_mol_per_reactor` arrays. The initial state needs `chemistry.concentrations`
(components x cells) and `chemistry.minerals_mol` (`Calcite`, `Gypsum`, mol per 1 L reactor). The Case 4 model enters through the
documented `Case4Model` hook (a module or namespace): `VM_M3_MOL`, `darcy_flux_m_s(porosity)`, `ReactorVolumeBasis`,
`evolve_porosity`; asrtm-dg already owns all four (`engines.case4`, `geometry`). The glue must also: apply `TimedChemistry`/Na
balance and `time_override_s` (variable dt) if it wants them, and `_install_clog_threshold` (monkeypatches
`_set_kinetic_area`; engine side). The restart state of a backend is `checkpoint_arrays()` / `restore(arrays)` (kept as is).

### Guard result and what stays in the driver
The guard was PARTIALLY triggered. The numerical state of a run is produced by `advance_case4_step` plus the backend; the rest of
`run_case4_dg.py::main` interleaves that loop with argparse validation, the code-hash configuration, flock, SQLite append/truncate,
checkpoint writes with PHREEQC RAW dumps and resume reconstruction, and the per-step diagnostics (element inventories, cumulative
boundary balances, BDF2 offsets, positivity aggregation) read from `backend.substep_log` and written to the history rows. Cutting
a `run` function out of that would mean re-ordering the history/checkpoint I/O around the loop and re-implementing the resume
branch (backend restore, chemistry restore, history truncation), which is not a copy. Extracted: every separable numerical module
and the step-level coupling (`advance_case4_step`). Options for the driver: (a) keep it in the paper repo importing `adaptdg.case4`
(minimal); (b) in the asrtm-dg glue, write a thin step loop on `advance_case4_step` plus the diagnostics (`inventories`,
cumulative balance), leaving history/checkpoint to the caller; neither changes a numerical operation.

### Equality (throwaway script outside the repo, fenicsx python, dolfinx 0.11, serial)
Source classes (`case4dev.*`, `asrtm_phreeqc.porosity.coupling`) vs `adaptdg.case4.*`, each side with its own fresh instance of the
source chemistry (`case4dev.chemistry_n.make_chemistry`, deterministic) and the source `darcy_flux_m_s`/geometry passed as the
`model`; per step `np.array_equal` on porosity, concentrations, minerals, every `record` entry, every `substep_log` entry
(positivity, balances, estimators), nodal/means_seen, mesh edges, `checkpoint_arrays`, BDF2 offsets. 16 setups, 0 differences in
all (items compared in parentheses): DG1 local N=80, 20 steps (the example, 11048); DG1 minmod+TVB N=40 dt 10800 4 substeps (2596);
history `mean` + no positivity + proportional update (2200); fv0 N=80 20 steps (10960); fv0 N=40 dt 43200 reaction_midpoint
(3288); variable-dt DG1 with dts 21600/32400/43200 (2810); coarse transport 40->20 (2822); BDF2 DG (1996); ADAP fast N=40 10 steps
(max 63 transport cells; 5390), with minmod+proportional (6110), FCT (6830), minmod_chem (5870); uniform_fine N=20 (3234);
ADAP persistent DOLFINx N=20 (2156); BDF2 ADAP N=40 (4208); ASFEM N=40 BDF2 (4648). Limits: serial, short runs (<= 20 steps),
checkpoint/resume and the driver not exercised; `Case4Chemistry` itself not compared (not extracted).
Tests: `tests/test_case4_transport.py` (numpy/scipy, fake chemistry and model, lazy-import check in a subprocess),
`tests/test_case4_dolfinx.py` (guarded with `importorskip`), shared fakes in `tests/case4_fakes.py`.

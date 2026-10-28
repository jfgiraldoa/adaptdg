# Extraction inventory: asrtm-phreeqc-public -> adaptdg / asrtm-dg

Source: `/Users/juanfe360/Documents/Shared/asrtm-phreeqc-public` (read-only). Facts from AST import tracing, `git grep`; nothing executed.
Legend: dolfinx = imports dolfinx/ufl/petsc4py (ALL are lazy, inside functions; none at top level). RM = imports phreeqcrm (lazy).
Class: T = TRANSPORT-ADAPTIVE (adaptdg), C = CHEMISTRY (asrtm-dg), A = COUPLING/APP (stays), S = CASE-SPECIFIC (stays).
Internal imports abbreviated (src = asrtm_phreeqc.).

## 1. Module map (src/asrtm_phreeqc, 28 modules)
| path | lines | dolfinx | RM | internal imports | class |
|---|---|---|---|---|---|
| __init__ | 15 | no | no | config, mass_balance, transfer | A |
| adaptive_calcite_dolomite_2d | 1496 | no (via st2d) | no | config, equilibrium_chemistry_2d, reactive_transport_policy_2d, stabilized_transport_2d, transfer_2d | A (holds T piece `_dorfler_union_2d`:172) |
| checkpoint_2d | 748 | no | no | adaptive_calcite_dolomite_2d, equilibrium_chemistry_2d | A |
| chemistry | 317 | no | lazy | - | C (Engesgaard) |
| config | 238 | no | no | - | A |
| dg_transport_1d | 196 | lazy | no | reusable_linear_problem | T |
| equilibrium_chemistry_2d | 327 | no | lazy | native_output, porosity | C (calcite/dolomite) |
| exchange_chemistry | 253 | no | lazy | - | C |
| mass_balance | 37 | no | no | - | A (generic diagnostic, numpy) |
| native_output | 56 | no | no | - | A |
| reactive_diagnostics_2d | 269 | no | no | equilibrium_chemistry_2d, porosity | A |
| reactive_transport_policy_2d | 306 | no | no | stabilized_transport_2d | A (chemistry-guided seeds; geometry part T) |
| reusable_linear_problem | 74 | lazy | no | - | T |
| stabilized_transport_2d | 686 | lazy | no | - | T |
| transfer | 198 | no | no | - | T (1D conservative remap; numpy only) |
| transfer_2d | 500 | no | no | - | T (rect<->tri remap; numpy only; also chemistry<->transport, see 2) |
| transport_comparison_2d | 340 | lazy | no | reusable_linear_problem, stabilized_transport_2d | T (reference/uniform baseline) |
| uniform_reactive_dg_2d | 437 | no | no | adaptive_cd_2d, config, eq_chem_2d, reactive_diagnostics_2d, st2d, transfer_2d, transport_comparison_2d | A |
| momas/__init__ | 1 | no | no | - | S |
| momas/chemistry | 244 | no | lazy | .model, native_output, porosity.geometry | C |
| momas/model | 120 | no | no | - | S |
| porosity/__init__ | 16 | no | no | .geometry | S |
| porosity/case4 | 219 | no | lazy | (.coupling lazy) | C (Case 4 kinetics) |
| porosity/checkpoint | 123 | no | no | - | S |
| porosity/coupling | 209 | no | no | .case4, .geometry, .dg_1d, .transport_1d (lazy) | A |
| porosity/dg_1d | 97 | lazy | no | .case4 (top-level!) | T (1D DG, couples to case4) |
| porosity/geometry | 128 | no | no | - | C (`ReactorVolumeBasis`, numpy; water-volume basis) |
| porosity/transport_1d | 695 | lazy | no | reusable_linear_problem, transfer | T (ADAP 1D + `dorfler_mark_1d`:81) |

## 1b. methods/ (35 modules)
| path | lines | dolfinx | RM | internal imports | class |
|---|---|---|---|---|---|
| onedim/rm1d | 112 | no (numpy+scipy) | no | - | T |
| onedim/adap1d | 355 | no (numpy) | no | .rm1d | T (marking inline :270-300 uses chem_width) |
| onedim/run_adap_exchange, run_engesgaard_rm_uniform | 25, 38 | no | via exchange_chemistry/chemistry (top) | onedim.adap1d | A |
| onedim/__init__ | 0 | - | - | - | - |
| case4dev/positivity | 88 | no | no | - | T |
| case4dev/adap_fast (NumpyOperator:35, positive_projection:106) | 486 | lazy | no | .adap_persistent, porosity.transport_1d | T |
| case4dev/adap_persistent | 385 | lazy | via porosity.case4 (lazy) | .positivity, porosity.case4/transport_1d | T |
| case4dev/adap_asfem (`dorfler_with_ties`:137) | 317 | no | no | .transport_bdf2, .adap_fast | T |
| case4dev/transport_bdf2 | 367 | lazy | no | .positivity, .adap_* , porosity.* | T |
| case4dev/dg_history, variable_dt | 190, 63 | lazy | no | .positivity / .dg_history, porosity.dg_1d | T |
| case4dev/fv_transport, coarse_transport | 82, 74 | no | no | porosity.case4 (lazy) / .dg_history | T |
| case4dev/chemistry_b1, chemistry_n | 107, 82 | no | lazy | porosity.case4 | C |
| case4dev/run_case4_dg | 751 | no | no (via lazies) | porosity.*, case4dev.* | A |
| front2d/f2a | 336 | no | no | .history_projection(_fast), st2d, adaptive_cd_2d, eq_chem_2d | A |
| front2d/fastadap | 796 | lazy | no | st2d, adaptive_cd_2d, checkpoint_2d, policy_2d, eq_chem_2d; monkey-patches src | A |
| front2d/history_projection, history_projection_fast | 189, 256 | lazy / no | no | - / .history_projection | T (mesh-to-mesh history transfer) |
| front2d/rm_class_patch, rm_norm, wall_bc, wall_seeds | 99, 36, 189, 98 | lazy (wall_bc) | no | st2d, adaptive_cd_2d, policy_2d, uniform_reactive_dg_2d (runtime patching) | S |
| front2d/run_front2d_adap | 150 | no | no | front2d | S |
| momas2d/adap | 540 | lazy | via momas.chemistry (lazy) | momas2d.rm_transport, front2d.fastadap, st2d, transfer_2d, dg_transport_1d, momas.* | A (holds `dorfler_union`:58) |
| momas2d/rm_transport, transport | 254, 235 | lazy | no | momas2d.transport, dg_transport_1d (lazy) | T |
| momas2d/darcy | 272 | lazy | no | - | T (RT0 Darcy; flow) |
| momas2d/chemistry_fast, coupled, run_momas2d | 54, 211, 75 | lazy | via momas.chemistry | momas.*, momas2d.* | A |
Also: scripts/run_calcite_dolomite_2d_provisional.py (746, A). Counts: src T 8 / C 6 / A 10 / S 4 (porosity/__init__, checkpoint, momas/__init__, momas/model); methods T 17 / C 2 / A 9 / S 5 (counting each file; empty __init__ excluded).

## 2. Transport <-> chemistry interface today
- 2D driver `adaptive_calcite_dolomite_2d.py`: transport fields on triangles (`cell_averages`, one array per component in `COMPONENTS_2D`) are stacked (shape [ncomp, ntri]) and sent to the chemistry rectangle grid with `transfer_2d.triangles_to_rectangles(triangles, stack, x_edges, y_edges)` (:1234); `chemistry.react(dict name->array(ny,nx) or (cells,), time_step=)` (:1279; def `equilibrium_chemistry_2d.py:178`); the reaction increment goes back with `plan.rectangles_to_triangles` (:1301, :1316 lifts averages). Units: totals in mol/kgw (PHREEQC reporting basis, `equilibrium_chemistry_2d.py:291,301`); constant porosity `POROSITY_2D` is passed to `SetPorosity` and representative volume (:96-117); transport is fed the same mol/kgw numbers (no explicit water volume per cell).
- 1D Case 4: `porosity/case4.py:164 react(concentrations, porosity, ...)` with `SetUnitsKinetics(1)` mol/L-water (:89), `SetRepresentativeVolume(ones)` (:92), porosity array per cell; coupling in `porosity/coupling.py:146` (conserves aqueous inventory; mineral volume update changes porosity). Transport cell mapping: `transfer.conservative_remap_1d` (edges only).
- Conservative transfer lives in `transfer.py` (1D, `ConservativeRemapPlan1D`, cache) and `transfer_2d.py` (`ConservativeRemapPlan2D`, plan cache, `refine_conservative_remap_plan_2d`:303). Needs only: cell edges / triangle vertex coordinates (areas are derived inside), source->target mapping built from geometry. No volumes are passed as arguments; no dolfinx. They are mesh-to-mesh AND transport-to-chemistry; asrtm-dg DESIGN wants a `transfer` taking cell volumes + mapping arrays (not yet matching: current API takes geometry).
- Chemistry code (chemistry, exchange_chemistry, equilibrium_chemistry_2d, momas/chemistry, porosity/case4) does NOT import dolfinx or meshes (verified by grep). Transport modules do NOT import PHREEQC (grep clean in st2d, dg_transport_1d, transport_1d, momas2d/transport, adap1d, fastadap).
- Separation violations:
  - `adaptive_calcite_dolomite_2d.py:16,185,406-440,1229-1236`: transport driver hard-codes `COMPONENTS_2D` (from `equilibrium_chemistry_2d`), mixes transport, transfer and `chemistry.react` (:998,:1279) in one 1496-line function.
  - `adaptive_calcite_dolomite_2d.py:172-220`: marking takes `chemistry_width` (:176,:588) and species list from chemistry.
  - `adaptive_calcite_dolomite_2d.py:1098-1108`: chemistry fields (incl. Calcite/Dolomite) passed to `create_predictive_reactive_mesh_2d`; `reactive_transport_policy_2d.py:267-281` derives sizes from chemistry grid edges.
  - `porosity/transport_1d.py:3,634,642`: docstring names the 80 PhreeqcRM cells; `minimum_width = eligibility_factor * chemistry_width`. `onedim/adap1d.py:34-35,273` (`chem_width`, `epsilon_chem`). `momas2d/adap.py:58-80,387`.
  - `porosity/dg_1d.py` imports `.case4` at top level (case4 is chemistry); `porosity/coupling.py:8,9` imports case4 + dg_1d.
  - `equilibrium_chemistry_2d.py:12`, `momas/chemistry.py` import `porosity` (ReactorVolumeBasis; numpy-only, but chemistry depends on a case package; should move to asrtm-dg `units`/`state`).
  - `methods/front2d/fastadap.py:719-752,794`, `rm_class_patch.py`, `wall_bc.py`, `wall_seeds.py`: runtime monkey-patching of `asrtm_phreeqc` functions (`_replace_everywhere`).
  - `methods/momas2d/adap.py:10` and `front2d/fastadap.py` import `src` internals; src imports nothing from methods (good, no reverse dependency).

## 3. Does marking depend on chemistry? YES, in four ways
1. Eligibility: a cell is markable only if `size > eligibility_factor * chemistry_width` (chemistry cell size). Implemented in `adaptive_calcite_dolomite_2d._dorfler_union_2d` (:172-220, eligible :182), `onedim/adap1d.py:273`, `porosity/transport_1d.dorfler_mark_1d` (:81-115, arg `minimum_width`), `momas2d/adap.py:dorfler_union`:58-80, `case4dev/adap_asfem.dorfler_with_ties`:137 (eligible mask passed in).
2. `epsilon_chem`: skip species if eligible share of the indicator < epsilon_chem (same functions; config `config.py:94-106,215`; `epsilon_eligible` in 1D Case 4).
3. Union over chemistry species (`COMPONENTS_2D`), theta, tie tolerance: loop over species names owned by chemistry.
4. Initial/predictive 2D mesh: `reactive_transport_policy_2d.reactive_front_seed_regions`:110 and `create_predictive_reactive_mesh_2d`:219 (call site `adaptive_calcite_dolomite_2d.py:1098`) take chemistry fields (components + Calcite/Dolomite on the chemistry grid) and edges; gradients on the chemistry grid give core/halo seed points; mesh refined to `target_size_in_chemistry_cells * chemistry_width`. Plus a `history_marking_filter` callback already exists (:296, applied :598).
Hook needed in adaptdg (no chemistry import): `marking.mark(indicators: dict[name, ndarray], *, min_size: float | ndarray, theta, epsilon, tie_tol)` where the app supplies `min_size` (scalar or per-cell, = factor * chemistry cell size) and optionally the species weights; the indicator dict keys are opaque strings. For the initial mesh: `mesh.refine_to_seeds(seed_points_or_size_field_callable)` / a `size_field(x, y) -> target h` callback computed by the app from chemistry gradients; plus a `post_mark_filter(marked, indicators) -> marked` callback (generalizes `history_marking_filter`).

## 4. Duplicates
- Same-name functions across src/methods (AST name match; not all identical): `_clip_axis` (transfer_2d vs front2d/fastadap), `history_from_cell_averages`, `cell_averages_from_history`, `add_cell_average_increments`, `set_history_cell_averages` (transport_comparison_2d, stabilized_transport_2d, fastadap), `sip`/`upwind` (st2d vs momas2d/rm_transport), `initialize`/`cell_means`/`replace_cell_means` (dg_transport_1d vs momas2d/transport), `_anchor_resident_sites` (momas/chemistry vs momas2d/chemistry_fast), `_read_minerals`/`_set_kinetic_area` (porosity/case4 vs case4dev/chemistry_b1), `set_porosity` (porosity/dg_1d vs case4dev/dg_history), `inventory` (porosity/transport_1d vs adap_fast).
- Five Dörfler variants: `_dorfler_union_2d`, `dorfler_union` (momas2d), inline in `adap1d.py:270-300`, `dorfler_mark_1d`, `dorfler_with_ties`: candidates for one `marking` module.
- `case4dev/adap_fast.py` (NumpyOperator, positive_projection, local_conservative_positivity), `adap_persistent`, `adap_asfem`, `transport_bdf2`: repeated `_operator`, `set_coefficients`, `coarsen`, `solve`.
- richards copy `/Users/juanfe360/Documents/Shared/richards/transport/` (ORIGEN.md; copied at MAIN 5dffa3a, MOMAS 71077d1, POROS 83224ed, so older than this repo): rm2d.py = st2d (+`l2dt_weight` arg replacing rm_norm patch); rm2d_fast.py = class extracted from fastadap `_fast_solver_class` (VMS/batching/wall-BC removed); rm2d_general.py = momas2d/rm_transport; dg2d_general.py = momas2d/transport (+`nonnegative_means` copied in); darcy_rt0.py = momas2d/darcy; reference2d.py = transport_comparison_2d; supg2d.py = supg_transport_2d (not in this repo; present in Phreeqnics-fenicsx); linear_problem.py = reusable_linear_problem; mesh_transfer(_fast).py = front2d/history_projection(_fast); remap_rect_tri.py = transfer_2d; predictive_mesh.py = reactive_transport_policy_2d; oned/rm1d.py = onedim/rm1d; oned/uniform1d.py (no counterpart here); oned/positivity.py = case4dev/positivity; oned/porosity_rm1d.py = adap_fast.py lines 1-201. Not in richards: `transfer.py` (1D remap), `adap1d`, `porosity/transport_1d`, `dg_transport_1d`, `case4dev/*` adaptive classes, all chemistry. It has no tests of numerical equivalence run (py_compile only per ORIGEN.md) and `rm2d` differs from `st2d` by the `l2dt_weight` keyword.

## 5. Proposed extraction order
1. `transfer.py` + `transfer_2d.py` (numpy-only, 198 + 500 lines) -> adaptdg `transfer` (mesh-to-mesh) and, with a volumes+mapping API, asrtm-dg `transfer`. Tests: `tests/test_conservative_remap.py` (79), `tests/test_transfer_2d.py` (200), `tests/test_transport_history_2d.py` (102). Difficulty low. Risk: the asrtm-dg design wants arrays-only API (volumes + mapping) while current API takes geometry; keep geometry plans in adaptdg and add a thin mapping-matrix layer.
2. Marking: unify the five Dörfler variants into `adaptdg.marking` with `min_size`/`epsilon`/tie params (see 3). Files: `porosity/transport_1d.py:81-115`, `adaptive_calcite_dolomite_2d.py:172-220`, `momas2d/adap.py:58`, `onedim/adap1d.py:270-300`, `adap_asfem.py:137`. Tests: `tests/test_porosity_transport_1d.py`, `tests/adap1d_smoke.py`, `tests/test_reactive_transport_policy_2d.py`, bitwise comparison of marked sets old vs new on recorded indicators. Difficulty medium; risk: subtle tie/eligibility differences between variants (different default epsilon 0.1/0.5/0.2).
3. Chemistry units + state: `porosity/geometry.py` (128) + `mineral_moles_per_kgw_to_bulk_mol_m3*` (chemistry.py:17, eq_chem_2d:44) + snapshot dataclasses -> asrtm-dg `units`, `state`. Tests: `test_porosity_geometry.py`, `test_engesgaard_units.py`, `test_porosity.py`. Low-medium; risk: historical factor 1/(1-phi) in 2D (eq_chem_2d:48) vs the "water volume only" rule.
4. Chemistry engines: `chemistry.py`, `exchange_chemistry.py`, `equilibrium_chemistry_2d.py`, `momas/chemistry.py`, `porosity/case4.py` -> asrtm-dg `engine`/`mask`. Tests: `test_momas_site_anchor.py`, `test_native_output.py`, `test_reactive_diagnostics_2d.py`, `tests/adap1d_smoke.py`. Medium-high; risk: PhreeqcRM state not bitwise reproducible across versions; replacing porosity args by water volume changes the call contract of each engine.
5. DG1/RM transport: `reusable_linear_problem`, `dg_transport_1d` (+`nonnegative_means`), `stabilized_transport_2d`, `porosity/transport_1d`, `momas2d/{transport,rm_transport,darcy}` -> adaptdg `flow`, `transport`, `estimators`. Tests: `test_porosity_transport_1d.py`, `momas2d_rt0_nesting.py`, `momas2d_anchor_tests.py`, `test_fct_chemistry_scale.py`, `test_minmod_chemistry_scale.py`, `case4_local_positivity_tests.py`. High; risk: dolfinx version drift, the three near-duplicate DG implementations (1D, 2D momas, case4 dg_1d), and the richards copy diverging.
6. ADAP loop + mesh persistence: split `adaptive_calcite_dolomite_2d.run_calcite_dolomite_adaptive_2d` (:887-1496), `reactive_transport_policy_2d`, `front2d/history_projection*` into `adaptdg.loop`/`mesh`/`fields` with chemistry via callbacks. Tests: `tests/test_checkpoint_2d.py`, `test_transport_history_2d.py`, `test_reactive_transport_policy_2d.py`; end-to-end 2D front reference runs (long, not short checks). High; risk: fastadap monkey-patches and the `history_marking_filter` coupling; 1500-line function.

## 6. Open questions for Juan
1. Should adaptdg keep the 1D (adap1d/transport_1d/case4dev) code or only 2D DG1 plus the common 1D remap?
2. Is `transfer` meant to be one module in adaptdg (mesh-to-mesh) re-exported in asrtm-dg, or duplicated in both (asrtm-dg DESIGN forbids importing adaptdg)?
3. Are the epsilon/eligibility defaults (0.1 1D, 0.5 MoMaS, 0.2 Case 4, config-driven 2D) per-case values to keep, or to be unified (changes published results)?
4. Who owns the `fastadap` monkey-patch variants (wall_bc, wall_seeds, VMS, batching): extracted as options of adaptdg or left in the paper repo?
5. Is the richards `transport/` copy to be replaced by adaptdg (it is older than this repo, missing `l2dt_weight` upstream), and should its tests (verificar.sh) be the contract?

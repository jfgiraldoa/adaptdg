# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: Semantic Versioning.

## [Unreleased]

### Added
- `adaptdg.vms` (pieces 1-3): stationary adaptive residual-minimisation solver for scalar diffusion-advection problems
  (Giraldo & Calo 2023, CMAME 417, 116285): `problems`, `meshes` (L-shape, Fichera), `dgforms` (SWIP/upwind forms, Gram norm),
  `saddle` (`RMSolver`, DG(p) x CG(p), any p, 2D/3D), `refine`, `norms`, `loop` and the JSON loader `run`; example
  `examples/vms/reentrant_corner/` and `docs/VMS.md`. dolfinx imports are lazy.
- `adaptdg.case4` subpackage (piece 9): Case 4 (evolving porosity, Xie B2) transport and ADAP modules extracted from
  ASRTM-PHREEQC (asrtm-phreeqc-public @ a152b91, `methods/case4dev/` and `porosity/{dg_1d,coupling}.py`): `adap_persistent`,
  `adap_fast`, `adap_asfem`, `transport_bdf2`, `dg_history`, `positivity`, `variable_dt`, `coarse_transport`, `fv_transport`
  (`--history fv0`), `dg_1d`, `coupling` (`Case4State`, `advance_case4_step`, `Case4Model` protocol) and `constants`. No
  chemistry import; dolfinx lazy. Not extracted: the driver `run_case4_dg.py` (CLI, SQLite/gzip history, checkpoints), the
  checkpoint module and the chemistry modules (see docs/DESIGN.md).
- `adaptdg.loop_momas` (MoMaS 2D ADAP loop `run` and helpers) and `adaptdg.coupled_momas` (uniform DG1 coupled run
  `run_coupled`, `square_groups`), extracted from ASRTM-PHREEQC (asrtm-phreeqc-public @ a152b91, `methods/momas2d/adap.py`,
  `coupled.py`) with the chemistry behind the `ChemistryMomas`/`EngineMomas` protocols (piece 8); no chemistry import, dolfinx lazy.
- `adaptdg.rm_1d` (`RMOperator1D`) and `adaptdg.loop_1d` (`Adap1DRun`, `Adap1DSettings`, 1D transfers and positivity
  fixes, `Chemistry1D` protocol documenting the duck-typed chemistry): the 1D residual-minimisation operator and the 1D
  ADAP loop of the paper, extracted from ASRTM-PHREEQC (asrtm-phreeqc-public @ a152b91, `methods/onedim/rm1d.py`,
  `adap1d.py`); numpy/scipy only, no dolfinx and no chemistry import (piece 7). `Adap1DRun._marking` uses
  `adaptdg.marking.dorfler_union_1d_adap` (exact adapter).
- `adaptdg.transfer` (1D) and `adaptdg.transfer_2d`: conservative remap plans,
  extracted unchanged from ASRTM-PHREEQC (commit a152b91). `transfer_field` is still a stub.
- `adaptdg.marking`: Dörfler marking core `dorfler_union` (min_size/epsilon/tie hooks,
  `post_mark_filter`), exact adapters for the 1D, 2D, MoMaS and adap1d variants and a verbatim
  `dorfler_with_ties`, extracted from ASRTM-PHREEQC (commit a152b91). The `dorfler_mark` stub was removed.
- `adaptdg.transport`: 1D DG1/CG1 transport extracted unchanged from ASRTM-PHREEQC (commit a152b91):
  `ConservativeDG1D`, `nonnegative_means` (`dg_transport_1d.py`) and the porosity/Case 4
  residual-minimisation transport (`porosity/transport_1d.py`); private `adaptdg._linear.ReusableLinearProblem`.
  `transport_step` is still a stub (piece 5a).
- `adaptdg.transport_2d` (`stabilized_transport_2d`: mesh/refinement helpers and the residual-minimisation
  solver), `adaptdg.transport_dg2d` (`momas2d/transport`: general 2D conservative DG1), `adaptdg.rm_transport`
  (`momas2d/rm_transport`) and the Darcy RT0-DG0 solver (`momas2d/darcy`) in `adaptdg.flow`, extracted from
  ASRTM-PHREEQC (commit a152b91); piece 5 is complete.
- `adaptdg.loop`: the 2D ADAP loop `run_adaptive_loop_2d` and `solve_one_adaptive_transport_step_2d` extracted from
  ASRTM-PHREEQC (commit a152b91) with chemistry as callbacks (`ChemistryCallbacks2D`) and numerical settings in
  `LoopSettings2D`; `adaptdg.mesh`: persistent-mesh policy and predictive reactive mesh. `AdaptiveTransport` and
  `adapt_mesh` remain stubs (piece 6).

## [0.0.1] - unreleased

### Added
- Package skeleton with stub modules (flow, transport, estimators, marking,
  mesh, transfer, loop, fields), tests and CI/release workflows.

# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: Semantic Versioning.

## [Unreleased]

## [0.1.1] - 2026-11-04

### Added
- Zenodo DOI in `README.md` and `CITATION.cff`.

### Changed
- `examples/vms`: the cases use the `legacy` preset by default and lead with the errors against the DG(p) projection, which match the
  published figures. The library defaults (`paper` preset, columns `L2`, `Vh` against the exact solution) are unchanged.

### Removed
- Unimplemented stubs: `adaptdg.AdaptiveTransport`, `adaptdg.register_field`, `transport_step`, `transfer_field`, `adapt_mesh`, `cell_indicators`,
  the modules `adaptdg.fields` and `adaptdg.estimators`, and the two skipped placeholder tests of `tests/test_conservation_contract.py`.

## [0.1.0] - 2026-11-04

First public release (alpha).

### Changed
- Modules moved into subpackages, without changes to the numerics: `adaptdg.transport` (`one_d`, `two_d`, `dg_2d`, `rm_1d`, `rm_2d`),
  `adaptdg.transfer` (`one_d`, `two_d`, `f2a`, `history_projection`, `history_projection_fast`), `adaptdg.loops` (`one_d`, `two_d`, `fastadap`,
  `mesh_policy`) and `adaptdg.momas` (`loop`, `coupled`). The old paths `adaptdg.loop`, `adaptdg.loop_1d`, `adaptdg.mesh` and the others are gone.

### Added
- `examples/vms/burgers_isotropic`: `p = 4` in `--full` (150000 DoF cap; paper Fig. 15): `L2`/`Vh` slopes -5.3/-4.5. The `L2` plateau of `theta` at `p = 3, 4`
  was checked on a frozen mesh (quadrature, Newton tolerances) and is a mesh-resolution effect, not an error-measurement artifact (README).
- `adaptdg.vms.unsteady` (`UnsteadyLoop`, `UnsteadySettings`) and the problem `ErikssonJohnsonUnsteady`: backward Euler and BDF2 for the
  residual-minimisation scheme with adaptive refinement in space at every time step (per-step tolerance, level and DoF caps), L2-projection history
  transfer between meshes, per-step errors in both targets for `ubar`, `theta`, `u`, `phi`; optional `time_coeffs` and Neumann data
  (`Problem.neumann_tags`, `neumann`) in `DGForms` (stationary forms unchanged); `Preset` fields `step_mesh`, `history_field`, `history_projection`.
  Orders measured on a fixed mesh: 0.97 / 0.98 (BE), 2.02 / 1.90 (BDF2). Example `examples/vms/eriksson_johnson_unsteady` (also in `run_all.py`).
- `examples/vms/run_all.py`: batch runner for all VMS examples and presets (parallel workers, per-job copies of the example folder,
  resume, timeouts, `status.json`, `REPORT.md` and `summary.csv` with slopes, ratios, Newton and clustering numbers, `--smoke`
  plumbing mode) and `examples/vms/README.md`. `--degrees` / `--dofmax` added to the `run.py` of `reentrant_corner`,
  `heterogeneous`, `anisotropic` and `compare_presets`.
- `examples/vms/burgers_isotropic` (paper section 6.5.1) and the problem `BurgersIsotropic`: manufactured tanh layer, both error
  targets, `ubar`/`theta`/`u`/`phi` errors with the nonlinear reconstruction, both presets; `theta_converged` column.
- `adaptdg.vms.nonlinear`: nonlinear residual minimisation for `div f(u) - div(kappa grad u) = s` (paper section 6): conservative
  dG forms with the Lax-Friedrichs flux (or the legacy wind form), the saddle point on `DG(p) x CG(p)` with the exact UFL
  Jacobian, hybrid Newton (Gauss-Newton then full Jacobian, optional damped restart) and the legacy damped Newton, warm start
  between adaptive levels, the nonlinear multiscale reconstruction and `NonlinearAdaptiveLoop` (a failed Newton level is
  flagged, never fatal). New `Preset` fields `nonlinear_flux` and `newton_method`, `LoopSettings.newton` / `newton_fallback`,
  `Problem.outflow_tags` / `has_exact`, `NonlinearProblem`, the problem `BurgersSingle` (section 6.5.2) and the example
  `examples/vms/burgers_single` (shock at `x = 1/2`, both presets). Linear cases are unchanged.
- History-preserving mesh rebuild (F2A) and fast path of the 2D adaptive loop: `run_adaptive_loop_2d(...,
  history_policy="f2a", fast=True)` keeps the DG1 reaction history across predictive-mesh rebuilds and in-step
  refinements (new modules `adaptdg.transfer.f2a`, `adaptdg.transfer.history_projection`, `adaptdg.transfer.history_projection_fast`,
  `adaptdg.loops.fastadap`; `adaptdg.transfer.two_d.use_plan_builder`). Defaults are unchanged; `AdaptiveRun2D.statistics`
  reports the counters.
- `adaptdg.vms.galerkin`: standard CG(p) Galerkin baseline (`solve_galerkin`, `galerkin_errors`) with the same problems, meshes
  and error targets; `LoopSettings.method` (`"rm"` | `"galerkin"`) and `compare_galerkin`; the problem `PoissonSine` and the
  example `examples/vms/poisson_fem` (Galerkin versus residual minimisation on the re-entrant corner).
- `adaptdg.vms`: the steady Eriksson-Johnson problem (`ErikssonJohnson`, `ErikssonJohnson3D`, `meshes.unit_cube`) and the example
  `examples/vms/eriksson_johnson` (paper section 5.6) with `paper` and `legacy` presets, both error targets, adaptive and
  uniform refinement, two diffusivities and a 3D option.
- `adaptdg.vms`: the Fichera 3D example `examples/vms/fichera_3d` (paper section 5.3) with `paper` and `legacy` presets, both
  error targets, adaptive and uniform refinement, cell-size statistics and plots; `problem.json` accepts the key `q`.
- `adaptdg.vms`: every VMS run can use the choices of the original implementation (`"preset": "legacy"` in
  `numerics.json`; default `"paper"` is unchanged) and reports the errors of `ubar`, `theta` and `phi` against both the
  exact solution and its DG(p) L2 projection (new `*_proj` history columns). Individual differences are switchable
  (`"preset_overrides"`), `adaptdg.marking.fixed_fraction_bands` implements the legacy marking, and the example
  `examples/vms/compare_presets` compares both presets. See "Legacy and paper presets" in `docs/VMS.md`.
- `adaptdg.vms`: stationary adaptive residual-minimisation solver for scalar diffusion-advection problems in 2D and 3D
  (Giraldo & Calo 2023, CMAME 417, 116285): problems, meshes (L-shape, Fichera), SWIP/upwind DG forms with the Gram
  norm, the DG(p) x CG(p) saddle-point solver (any p), refinement, error norms, the adaptive loop and a JSON-driven
  runner. See `docs/VMS.md`.
- VMS reconstruction (fine scale, adjoint multiscale reconstruction and dG baseline) with the corresponding error norms,
  switched on by the `reconstruct` key of `numerics.json`.
- `Heterogeneous` and `Anisotropic` cases and the examples `examples/vms/reentrant_corner`, `heterogeneous` and
  `anisotropic`.
- 1D transport (`adaptdg.transport.one_d`, `adaptdg.transport.rm_1d`) and 2D transport (`adaptdg.transport.two_d`,
  `adaptdg.transport.dg_2d`, `adaptdg.transport.rm_2d`) with the Darcy RT0-DG0 solver in `adaptdg.flow`.
- Conservative mesh transfer in 1D and 2D (`adaptdg.transfer.one_d`, `adaptdg.transfer.two_d`).
- Dörfler marking (`adaptdg.marking`): a common core and adapters for the 1D, 2D, MoMaS and 1D-loop variants.
- Adaptive loops that take the chemistry through callbacks or protocols: `adaptdg.loops.two_d` (2D), `adaptdg.loops.one_d`,
  `adaptdg.momas.loop` and `adaptdg.momas.coupled`; persistent-mesh policy in `adaptdg.loops.mesh_policy`.
- `adaptdg.case4`: Case 4 (evolving porosity) transport and adaptive modules, with no chemistry import.
- `CITATION.cff`, project URLs and keywords in `pyproject.toml`.

### Changed
- `BurgersIsotropic` (6.5.1) now uses the paper's isotropic Gram norm (eqs. 49-50, `norm_form = "isotropic"`, as `BurgersSingle`);
  `norm_form = "streamline"` restores the old norm wind (1, 1). Paper preset, `--full` (63000-110000 DoF), p = 1, 2, 3: Newton converges on
  27/29, 32/32 and 41/41 levels (before 28/28, 27/28, 28/28 at 30000-50000 DoF) with at most 53, 138 and 94 iterations per level (before 220, 421
  and 36); the library default `maxit = 100` is now enough, so the example no longer overrides it for the paper preset (the legacy
  preset keeps 400). The two non-converged levels at `p = 1` are `stalled` at `|R| < 6e-6`. The `L2` plateau of `ubar` at `p = 3` is gone
  (1.1e-6 against 5e-5).
- `adaptdg.vms` single-component Burgers (6.5.2) at small kappa: `BurgersSingle` now uses the norm of eqs. (49)-(50)
  (`norm_form = "isotropic"`: `(w, w) + sum ||[[w]]||_F^2 + sum h_K ||grad w||_K^2`, no wind; `"streamline"` restores the old
  Gram form with the norm wind (1, 1), kept for comparison). With the old form Newton failed on 10 of 15 levels
  at `kappa = 1e-4`; now `python run.py --full` converges on every level at `kappa = 1e-2, 1e-3, 1e-4` (32, 20 and 18 levels, up to
  50000 DoF, about 1 minute) and resolves the shock (slope -4442 against the analytic -5000 at 53000 DoF). The hybrid Newton
  solver has Armijo backtracking (`NewtonOptions.damping = "backtrack"`, default of the hybrid method; `damping = "none"` gives the
  old undamped iteration), a cold retry from the initial guess and a continuation in kappa for failed levels
  (`cold_retry`, `continuation`; not in the legacy preset), and the new status `stalled`. The history has the column
  `newton_increment`. `shock_diagnostics` measures the slope and width on a refined window around the shock (the 2001-point line
  saturated near -2000 at `kappa = 1e-4`); `kappa = 1e-2, 1e-3` values are unchanged. The example's `--full` cap is 50000 DoF
  (was 20000); the Newton tolerances are unchanged (`|du|_Vh < 1e-9` or `|R| < 1e-11`, stricter than the 1e-6 of the paper).
- `adaptdg.vms`: the loop history has two extra columns, `L2_proj` and `Vh_proj` (and `*_proj` for `theta`, `u`, `phi`
  with `reconstruct`); set `error_targets = "exact"` to keep the previous columns only. Existing values are unchanged.

### Fixed
- `adaptdg.vms.dgforms`: the weak inflow boundary term of the upwind advection form had the wrong sign (inconsistent
  for nonzero Dirichlet data on the inflow boundary, rates degraded to about DoF^(-1/2) in Vh). The re-entrant
  corner (no advection) and zero-data cases were not affected.

## [0.0.1] - unreleased

### Added
- Package skeleton with stub modules, tests and CI/release workflows.

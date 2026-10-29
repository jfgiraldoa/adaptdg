# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: Semantic Versioning.

## [Unreleased]

### Added
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
  refinements (new modules `adaptdg.f2a`, `adaptdg.history_projection`, `adaptdg.history_projection_fast`,
  `adaptdg.fastadap`; `adaptdg.transfer_2d.use_plan_builder`). Defaults are unchanged; `AdaptiveRun2D.statistics`
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
- 1D transport (`adaptdg.transport`, `adaptdg.rm_1d`) and 2D transport (`adaptdg.transport_2d`,
  `adaptdg.transport_dg2d`, `adaptdg.rm_transport`) with the Darcy RT0-DG0 solver in `adaptdg.flow`.
- Conservative mesh transfer in 1D and 2D (`adaptdg.transfer`, `adaptdg.transfer_2d`).
- Dörfler marking (`adaptdg.marking`): a common core and adapters for the 1D, 2D, MoMaS and 1D-loop variants.
- Adaptive loops that take the chemistry through callbacks or protocols: `adaptdg.loop` (2D), `adaptdg.loop_1d`,
  `adaptdg.loop_momas` and `adaptdg.coupled_momas`; persistent-mesh policy in `adaptdg.mesh`.
- `adaptdg.case4`: Case 4 (evolving porosity) transport and adaptive modules, with no chemistry import.
- `CITATION.cff`, project URLs and keywords in `pyproject.toml`.

### Changed
- `adaptdg.vms` single-component Burgers (6.5.2) at small kappa: `BurgersSingle` now uses the norm of eqs. (49)-(50)
  (`norm_form = "isotropic"`: `(w, w) + sum ||[[w]]||_F^2 + sum h_K ||grad w||_K^2`, no wind; `"streamline"` restores the old
  Gram form with the norm wind (1, 1), still used by `BurgersIsotropic`). With the old form Newton failed on 10 of 15 levels
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

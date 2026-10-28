# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: Semantic Versioning.

## [Unreleased]

### Added
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
- `adaptdg.vms`: the loop history has two extra columns, `L2_proj` and `Vh_proj` (and `*_proj` for `theta`, `u`, `phi`
  with `reconstruct`); set `error_targets = "exact"` to keep the previous columns only. Existing values are unchanged.

### Fixed
- `adaptdg.vms.dgforms`: the weak inflow boundary term of the upwind advection form had the wrong sign (inconsistent
  for nonzero Dirichlet data on the inflow boundary, rates degraded to about DoF^(-1/2) in Vh). The re-entrant
  corner (no advection) and zero-data cases were not affected.

## [0.0.1] - unreleased

### Added
- Package skeleton with stub modules, tests and CI/release workflows.

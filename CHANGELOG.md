# Changelog

All notable changes are documented here. Format: Keep a Changelog;
versioning: Semantic Versioning.

## [Unreleased]

### Added
- `adaptdg.transfer` (1D) and `adaptdg.transfer_2d`: conservative remap plans,
  extracted unchanged from ASRTM-PHREEQC (commit a152b91). `transfer_field` is still a stub.
- `adaptdg.marking`: Dörfler marking core `dorfler_union` (min_size/epsilon/tie hooks,
  `post_mark_filter`), exact adapters for the 1D, 2D, MoMaS and adap1d variants and a verbatim
  `dorfler_with_ties`, extracted from ASRTM-PHREEQC (commit a152b91). The `dorfler_mark` stub was removed.
- `adaptdg.transport`: 1D DG1/CG1 transport extracted unchanged from ASRTM-PHREEQC (commit a152b91):
  `ConservativeDG1D`, `nonnegative_means` (`dg_transport_1d.py`) and the porosity/Case 4
  residual-minimisation transport (`porosity/transport_1d.py`); private `adaptdg._linear.ReusableLinearProblem`.
  `transport_step` is still a stub (piece 5a).

## [0.0.1] - unreleased

### Added
- Package skeleton with stub modules (flow, transport, estimators, marking,
  mesh, transfer, loop, fields), tests and CI/release workflows.

# Backport list: adaptdg <-> Richards and Fractura

Policy (Juan, 2026-10-28): Richards (`transrich`) and Fractura keep their own independent transport modules and do not
depend on adaptdg. adaptdg is the testbed: strong transport changes are tested here first (exact-equality checks and
tests), and the best of adaptdg is ported by hand, selectively, to those projects, with their own tests
(`verificar.sh` in Richards, `dev/tests/fractura_*` in Fractura). Improvements also flow the other way.

## To absorb into adaptdg (present in Richards/Fractura, absent here)
- Richards `transport/rm2d_general.py`: `l2dt_weight` keyword in the 2D residual-minimisation transport (the paper
  code, hence adaptdg `transport_2d`, lacks it). Absorb as an option; the default behaviour must not change.
- Fractura `dev/fractura/`: `tensor_adapt.py` / `tensor_transfer.py` (adaptive tensor mesh and exact DG1 projection),
  `dg_md.py` / `dg0_md.py` (mixed dimension), `darcy_md.py` (hybrid RT0-P0 Darcy), `estimators.py`,
  `estimator_steperror.py`. Evaluate before absorbing; mixed-dimension code may stay in Fractura as an extension.

## To evaluate for porting from adaptdg (nothing ported yet)
- `adaptdg.transfer`, `adaptdg.transfer_2d`: conservative remaps proven exactly equal to the ASRTM-PHREEQC originals
  (Richards' `mesh_transfer*` and Fractura's `tensor_transfer` are related; compare before porting).
- `adaptdg.marking`: unified Dörfler core with `min_size`, `epsilon`, `tie_tol` and `post_mark_filter` hooks.

## Rule for each port
One piece at a time; keep a note here with the adaptdg commit, the target project commit, the test that guards it and
the date. Record anything that changes numerical results.

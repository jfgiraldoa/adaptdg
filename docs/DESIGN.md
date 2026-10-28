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
2. DG1 transport.
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

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
1. Conservative mesh transfer.
2. DG1 transport.
3. Estimators and Dörfler marking.
4. ADAP loop.

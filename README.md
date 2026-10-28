# adaptdg

Adaptive discontinuous-Galerkin (DG) transport on FEniCSx (dolfinx 0.11):
DG1 advection-diffusion, a-posteriori error estimators with Dörfler marking,
conservative mesh transfer, and an adaptive-mesh loop.

**Status: pre-alpha, interface not stable.** Everything is a stub for now.

## Install

dolfinx, petsc4py and mpi4py are not pip-installable. Create a conda-forge
environment first, then install the package:

```bash
conda create -n adaptdg -c conda-forge python=3.12 fenics-dolfinx=0.11
conda activate adaptdg
pip install adaptdg
```

`import adaptdg` works without dolfinx; FEM functionality imports it lazily.

## Dependency rule

Applications (reactive transport, fractures, Richards, ...) depend on
`adaptdg`, never the reverse. `adaptdg` contains no application-specific
physics or chemistry and must not import or mention any application package.

## Planned API (sketch)

```python
from adaptdg import FlowState, AdaptiveTransport

flow = FlowState(mesh=mesh, theta_old=th0, theta_new=th1, q=q_rt0)
at = AdaptiveTransport(mesh, flow, dt=dt)
at.register_field("c", kind="dg1")             # transferred conservatively
at.register_field("mineral", kind="cell_average")

for step in range(n_steps):
    at.advance(flow)          # transport + mesh adaptation
    # the app owns the time loop and any source terms / reactions
    print(at.water_volume())  # conservation diagnostic
```

## License

MIT

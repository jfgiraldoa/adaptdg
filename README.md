# adaptdg: adaptive discontinuous-Galerkin transport

`adaptdg` is a framework for **adaptive transport on FEniCSx** (dolfinx 0.11). It solves
linear and nonlinear **advection–diffusion–reaction** problems, particularly in
**advection-dominated regimes** where standard methods lose stability and accuracy, and it
adapts the mesh at every time step from an on-the-fly error estimator.

The method couples a **semi-discrete discontinuous Galerkin (DG) formulation in space** with
a **time-marching scheme** (method of lines) and a **residual-minimisation strategy**. The
residual minimisation yields a stable, continuous solution and, at the same time, an
**error estimator** that drives **adaptive mesh refinement** at each discrete time step.

> **Status: pre-alpha, interface not stable.** The package is being extracted from
> working research codes; most modules are stubs until that migration is finished.

## History

`adaptdg` is the continuation of **AS-FEM** (Adaptive Stabilized Finite Element Method),
first released in 2025 as a research code on legacy FEniCS 2019 for steady and unsteady
advection–diffusion–reaction problems. Its benchmarks included the Eriksson–Johnson problem
(steady and transient), the nonlinear Burgers equation, the 3D Fichera corner, highly
heterogeneous media and highly anisotropic diffusion.

The method then moved to **FEniCSx** and became the transport engine of reactive-transport
work (coupled with PHREEQC through PhreeqcRM), where it was extended with:

- DG1 transport with conservative mesh-to-mesh transfer of the history fields;
- a-posteriori residual estimators with Dörfler marking;
- adaptive refinement per time step, tested on reaction fronts in porous media.

Because the same machinery is needed by other applications (fractured media with mixed
dimensions, Richards flow in variably saturated soils, reactive transport), it is being
extracted into this standalone, application-independent package. The original AS-FEM
code (FEniCS 2019) remains in the git history, tagged `asfem-legacy`.

## Dependency rule

Applications (reactive transport, fractures, Richards, …) depend on `adaptdg`, never the
reverse. `adaptdg` contains no application-specific physics or chemistry and must not
import or mention any application package.

## Features (target)

- Adaptive refinement driven by residual-based error estimators
- Stabilised DG formulations for convection-dominated problems
- DG spatial discretisation coupled with time-marching schemes
- Conservative transfer of fields between meshes after adaptation
- Modular structure: the application owns the time loop, sources and reactions

## Install

dolfinx, petsc4py and mpi4py are not pip-installable. Create a conda-forge environment
first, then install the package:

```bash
conda create -n adaptdg -c conda-forge python=3.12 fenics-dolfinx=0.11
conda activate adaptdg
pip install adaptdg
```

`import adaptdg` works without dolfinx; FEM functionality imports it lazily.
Until the first PyPI release, install from the repository:

```bash
pip install "git+https://github.com/jfgiraldoa/adaptdg.git"
```

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

## Folder structure

```
adaptdg/
├── src/adaptdg/     # package source (flow, transport, estimators, marking, mesh, transfer, loop, fields)
├── tests/           # unit and contract tests
├── docs/DESIGN.md   # architecture, interface and extraction plan
└── .github/         # CI and release workflows
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## How to cite

If you use this software, please cite it (see `CITATION.cff`; a DOI will be added with the
first release) together with the methodology papers below.

## References

1. Giraldo, J. F., & Calo, V. M. (2023). An Adaptive in Space, Stabilized Finite Element
   Method via Residual Minimization for Linear and Nonlinear Unsteady
   Advection–Diffusion–Reaction Equations. *Mathematical and Computational Applications*,
   28(1), 7. [https://doi.org/10.3390/mca28010007](https://doi.org/10.3390/mca28010007)

2. Giraldo, J. F., & Calo, V. M. (2023). A variational multiscale method derived from an
   adaptive stabilized conforming finite element method via residual minimization on dual
   norms. *Computer Methods in Applied Mechanics and Engineering*, 417, 116285.
   [https://doi.org/10.1016/j.cma.2023.116285](https://doi.org/10.1016/j.cma.2023.116285)

3. Giraldo, J. F., & Calo, V. M. (2025). An unfitted domain approach using a stabilized
   residual minimization method for Dirichlet problems. *Advances in Computational Science
   and Engineering*, 4.

4. Liu, C., Giraldo, J. F., Calo, V. M., Regenauer-Lieb, K., & Hu, M. (2026). An adaptive
   stabilized finite element method for coupled nonlinear reaction-diffusion systems.
   *Journal of Computational Physics*, 548, 114598.
   [https://doi.org/10.1016/j.jcp.2025.114598](https://doi.org/10.1016/j.jcp.2025.114598)

5. Giraldo, J. F., Poulet, T., Vialle, S., & Calo, V. M. (under review). Adaptive Stabilised
   Reactive Transport in Porous Media with Reaction Fronts. *Computers & Geosciences*.

## License

This project is distributed under the MIT License, a permissive open-source license that
allows reuse, modification and distribution with proper attribution.
© 2025–2026 Juan Felipe Giraldo Avila. See the `LICENSE` file for details.

## Contributing

Contributions are welcome. Feel free to open issues, submit pull requests or suggest new
features.

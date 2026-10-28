# adaptdg

Adaptive stabilised discontinuous-Galerkin (DG) / residual-minimisation transport and PDE solvers
on FEniCSx. It is the successor of AS-FEM and covers transport only, with no chemistry.

Status: alpha.

## Summary

adaptdg discretises advection-diffusion-reaction problems with DG and minimises the residual in a
dual DG norm. This gives a stable solution and, at the same time, an error indicator that drives
adaptive mesh refinement. A variational multiscale reconstruction is available for stationary
problems, and time-dependent transport runs with adaptive meshes in 1D and 2D.

## What is inside

- `adaptdg.vms`: stationary adaptive solver (2D/3D, any polynomial degree) and VMS reconstruction.
- 1D and 2D transient transport with adaptive meshes.
- Conservative mesh-to-mesh transfer of the history fields.
- Dörfler marking.
- Adaptive loops that take the chemistry through callbacks.
- Case 4 transport (evolving porosity).

## Examples

- `examples/vms/reentrant_corner`: Laplace problem with a singular solution at a re-entrant corner;
  `python run.py` takes about 5 s.
- `examples/vms/heterogeneous`: diffusion-advection across a jump in diffusivity;
  `python run.py` takes about 30 s.
- `examples/vms/anisotropic`: strongly anisotropic diffusion with an inner layer;
  `python run.py` takes about 15 s.
- `examples/vms/fichera_3d`: 3D Poisson problem with a singular solution at the Fichera corner;
  `python run.py` takes about 10 s (`--full`: about 4 min per preset, up to 3 GB).
- `examples/vms/eriksson_johnson`: steady Eriksson-Johnson problem with a boundary layer (2D, and the 3D variant);
  `python run.py` takes about 20 s (`--full`: about 8 min per preset, up to 4 GB).
- `examples/vms/poisson_fem`: plain CG(p) Galerkin baseline versus residual minimisation on the re-entrant corner;
  `python run.py` takes about 15 s.
- The Burgers cases are not ported yet.

Each example reads `problem.json` and `numerics.json`.

## Install

dolfinx, petsc4py and mpi4py are not pip-installable, so create a conda environment first:

```bash
conda create -n adaptdg -c conda-forge python=3.12 fenics-dolfinx=0.11
conda activate adaptdg
pip install "git+https://github.com/jfgiraldoa/adaptdg.git"
```

A PyPI release will follow. `import adaptdg` works without dolfinx.

## Run an example

```bash
git clone https://github.com/jfgiraldoa/adaptdg.git
cd adaptdg/examples/vms/reentrant_corner
python run.py && python plot.py
```

`problem.json` defines the problem (domain, data, boundary conditions) and `numerics.json` the
discretisation and the adaptive loop (degree, marking, refinement).

## Used by

[asrtm-phreeqc](https://github.com/jfgiraldoa/asrtm-phreeqc) (ASRTM-PHREEQC) uses adaptdg as its transport
backend and couples the PHREEQC chemistry; adaptdg itself has no chemistry.

## History

adaptdg continues AS-FEM. The original FEniCS 2019 code and its benchmarks are kept in the git tag
`asfem-legacy`. Port status: Poisson / re-entrant corner, heterogeneous, anisotropic, Fichera, the Poisson FEM baseline and the steady
Eriksson-Johnson problem (2D and 3D) are ported; Burgers and the unsteady Eriksson-Johnson cases are pending.

## How to cite

Cite the software through `CITATION.cff` (a DOI will be added with the first release) and the VMS
paper (Giraldo & Calo 2023, CMAME, below).

## Previous work

- Giraldo, J. F., & Calo, V. M. (2023). An Adaptive in Space, Stabilized Finite Element Method via
  Residual Minimization for Linear and Nonlinear Unsteady Advection-Diffusion-Reaction Equations.
  *Mathematical and Computational Applications*, 28(1), 7.
  [doi:10.3390/mca28010007](https://doi.org/10.3390/mca28010007)
- Giraldo, J. F., & Calo, V. M. (2023). A variational multiscale method derived from an adaptive
  stabilized conforming finite element method via residual minimization on dual norms. *Computer
  Methods in Applied Mechanics and Engineering*, 417, 116285.
  [doi:10.1016/j.cma.2023.116285](https://doi.org/10.1016/j.cma.2023.116285)
- Giraldo, J. F., & Calo, V. M. (2025). An unfitted domain approach using a stabilized residual
  minimization method for Dirichlet problems. *Advances in Computational Science and Engineering*, 4.
- Liu, C., Giraldo, J. F., Calo, V. M., Regenauer-Lieb, K., & Hu, M. (2026). An adaptive stabilized
  finite element method for coupled nonlinear reaction-diffusion systems. *Journal of Computational
  Physics*, 548, 114598. [doi:10.1016/j.jcp.2025.114598](https://doi.org/10.1016/j.jcp.2025.114598)

## License

MIT, (c) 2025-2026 Juan Felipe Giraldo Avila.

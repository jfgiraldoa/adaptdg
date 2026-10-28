"""VMS dG forms and the saddle-point system."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("scipy")

import ufl  # noqa: E402
from dolfinx import fem  # noqa: E402

from adaptdg.vms import meshes  # noqa: E402
from adaptdg.vms.dgforms import penalty_field  # noqa: E402
from adaptdg.vms.loop import AdaptiveStationaryLoop, LoopSettings, fitted_slope  # noqa: E402
from adaptdg.vms.problems import Problem, ReentrantCorner  # noqa: E402
from adaptdg.vms.saddle import RMSolver  # noqa: E402


class Smooth(Problem):
    """Manufactured diffusion-advection problem on the unit square (source derived)."""

    name = "smooth"
    dim = 2

    def __init__(self, beta=None):
        self._beta = beta

    def exact(self, x):
        return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    def beta(self, x):
        return None if self._beta is None else ufl.as_vector(self._beta)

    def initial_mesh(self, n, comm=None):
        return meshes.crossed_square(n, 0.0, 1.0, comm)


def test_penalty_positive_and_scales_with_degree():
    dom = meshes.l_shape(4)
    e1, e2 = penalty_field(dom, 1), penalty_field(dom, 2)
    assert (e1.x.array > 0).all()
    assert np.allclose(e2.x.array / e1.x.array, (3 * 4) / (2 * 3))  # (p+1)(p+d)/d ratio, d=2
    e3 = penalty_field(meshes.fichera(2), 1)
    assert (e3.x.array > 0).all()


def test_saddle_matrix_is_symmetric_with_zero_coarse_block():
    solver = RMSolver(ReentrantCorner(), 1)
    a, _, space, _ = solver.system(meshes.l_shape(2))
    mat = fem.assemble_matrix(fem.form(a))
    dense = mat.to_scipy().toarray()
    assert np.abs(dense - dense.T).max() < 1e-10 * np.abs(dense).max()
    cg = np.asarray(space.sub(1).collapse()[1]).ravel()
    dg = np.setdiff1d(np.arange(dense.shape[0]), cg)
    assert np.abs(dense[np.ix_(cg, cg)]).max() < 1e-12  # zero (u,u) block: saddle point
    assert np.linalg.eigvalsh(dense[np.ix_(dg, dg)]).min() > 0  # Gram block is SPD
    eig = np.linalg.eigvalsh(dense)
    assert eig.min() < 0 < eig.max()  # indefinite


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("beta", [None, (1.0, 0.5)])
def test_uniform_refinement_rates_on_smooth_solution(p, beta):
    settings = LoopSettings(p=p, refinement="uniform", max_levels=4, initial_mesh_size=2)
    hist = AdaptiveStationaryLoop(Smooth(beta), settings).run(progress=None)
    assert hist[-1]["ndofs"] > hist[0]["ndofs"]
    kw = {"span": 1e9}
    assert fitted_slope(hist, "Vh", 2, **kw) == pytest.approx(-p, abs=0.35)
    assert fitted_slope(hist, "L2", 2, **kw) == pytest.approx(-(p + 1), abs=0.45)


@pytest.mark.parametrize("p", [1, 2])
def test_inflow_dirichlet_data_is_consistent_with_advection(p):
    """Regression: the weak inflow boundary term bn (u_D - u) v must make the exact solution consistent."""

    class Shifted(Smooth):
        def exact(self, x):
            return 1.0 + ufl.sin(2 * x[0]) * ufl.cos(x[1])

    settings = LoopSettings(p=p, refinement="uniform", max_levels=4, initial_mesh_size=2)
    hist = AdaptiveStationaryLoop(Shifted((1.0, 0.5)), settings).run(progress=None)
    assert fitted_slope(hist, "Vh", 2, span=1e9) == pytest.approx(-p, abs=0.35)  # was about -0.5
    assert fitted_slope(hist, "L2", 2, span=1e9) == pytest.approx(-(p + 1), abs=0.45)  # was about -1.5

"""Problem definitions for the stationary scalar solver (Giraldo & Calo 2023, CMAME 417, 116285).

A problem is ``-div(kappa grad u) + beta . grad u = f`` with Dirichlet data on the tagged
boundary. Data callbacks receive a UFL ``SpatialCoordinate`` so they can return UFL expressions
(scalars, vectors or tensors, possibly discontinuous); ufl is imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Marker = Callable[[np.ndarray], np.ndarray]


def _whole_boundary(x: np.ndarray) -> np.ndarray:
    return np.full(x.shape[1], True)


class Problem:
    """Interface of a stationary diffusion-advection problem (override what differs)."""

    name = "problem"
    dim = 2
    dirichlet_tags: tuple[int, ...] = (1,)
    #: tags of natural (outflow) boundary faces: no Dirichlet data, only used by the nonlinear forms.
    outflow_tags: tuple[int, ...] = ()
    #: False when the problem has no exact solution (the error targets are then not available).
    has_exact = True

    def kappa(self, x: Any) -> Any:
        """Diffusivity: number, scalar UFL expression or (dim x dim) UFL tensor."""
        return 1.0

    def beta(self, x: Any) -> Any | None:
        """Advection velocity (UFL vector) or ``None`` for pure diffusion."""
        return None

    def exact(self, x: Any) -> Any:
        """Exact solution as a UFL expression."""
        raise NotImplementedError

    def dirichlet(self, x: Any) -> Any:
        """Dirichlet data (default: the exact solution)."""
        return self.exact(x)

    def source(self, x: Any) -> Any | None:
        """Source ``f``; by default derived symbolically from the exact solution."""
        import ufl

        u, k, b = self.exact(x), self.kappa(x), self.beta(x)
        flux = k * ufl.grad(u) if not getattr(k, "ufl_shape", ()) else ufl.dot(k, ufl.grad(u))
        f = -ufl.div(flux)
        return f if b is None else f + ufl.dot(b, ufl.grad(u))

    def boundary_markers(self) -> dict[int, Marker]:
        """Boundary tags: ``{tag: marker(x) -> bool array}`` (default: one tag, whole boundary)."""
        return {1: _whole_boundary}

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        """Start mesh with ``n`` cells per direction."""
        raise NotImplementedError


@dataclass
class ReentrantCorner(Problem):
    """Laplace on (-1,1)^2 minus the quadrant x>0, y<0 with u = r^a sin(a theta), a = 2/3.

    theta = atan2(y, x) mod 2 pi lies in [0, 3 pi/2] on the domain, the singular corner is the
    origin and u is harmonic, so f = 0 (paper section 5.2; the removed quadrant is the mirror
    image of the paper's by symmetry).
    """

    alpha: float = 2.0 / 3.0
    diffusivity: float = 1.0
    velocity: tuple[float, ...] = field(default_factory=tuple)
    name = "reentrant_corner"
    dim = 2

    def kappa(self, x: Any) -> Any:
        return float(self.diffusivity)

    def beta(self, x: Any) -> Any | None:
        import ufl

        return ufl.as_vector([float(v) for v in self.velocity]) if any(self.velocity) else None

    def exact(self, x: Any) -> Any:
        import ufl

        r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
        th = ufl.atan2(x[1], x[0])
        th = ufl.conditional(th < 0, th + 2 * ufl.pi, th)
        return r**self.alpha * ufl.sin(self.alpha * th)

    def source(self, x: Any) -> Any | None:
        if self.beta(x) is None:
            return None  # u is harmonic
        return super().source(x)

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import l_shape

        return l_shape(n, comm)


@dataclass
class Fichera3D(Problem):
    """-Delta u = f on (-1,1)^3 minus [0,1)^3 with u = r^q (paper section 5.3), f derived."""

    q: float = 1.0 / 3.0
    diffusivity: float = 1.0
    name = "fichera"
    dim = 3

    def kappa(self, x: Any) -> Any:
        return float(self.diffusivity)

    def exact(self, x: Any) -> Any:
        return (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (self.q / 2)

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import fichera

        return fichera(n, comm)


def heterogeneous_exact(x: Any, eps1: float = 0.1, eps2: float = 1.0) -> Any:
    """1D exact solution of ``-(eps u')' + u' = 0`` on (0,1) with ``u(0) = 1``, ``u(1) = 0``, ``eps = eps1``
    for ``x < 1/2`` and ``eps2`` after, continuous in ``u`` and in ``eps u'`` (paper eqs. 39-40).
    Works on numpy arrays and on UFL expressions (the latter through ``ufl.conditional``)."""
    e1, e2 = np.exp(1 / (2 * eps1)), np.exp(1 / (2 * eps2))
    a = e1 / (1 - e1)
    uh = a / (a + 1 / (1 - e2))  # eq. (40)
    if hasattr(x, "ufl_shape"):
        import ufl

        exp, where = ufl.exp, ufl.conditional
        left = (uh - e1 + (1 - uh) * exp(x / eps1)) / (1 - e1)
        right = (-e2 * uh + uh * exp((x - 0.5) / eps2)) / (1 - e2)
        return where(x < 0.5, left, right)
    left = (uh - e1 + (1 - uh) * np.exp(x / eps1)) / (1 - e1)
    right = (-e2 * uh + uh * np.exp((x - 0.5) / eps2)) / (1 - e2)
    return np.where(x < 0.5, left, right)


@dataclass
class Heterogeneous(Problem):
    """-div(kappa grad u) + beta . grad u = 0 on (0,1)^2, ``kappa = diag(eps_i, 1)`` (paper section 5.4).

    ``eps = eps1`` for ``x < 1/2`` and ``eps2`` for ``x > 1/2``; ``u`` depends on ``x`` only, so the
    exact solution (eqs. 39-40) needs ``beta = (1, 0)``. The paper text prints ``beta = (0, 1)``, which
    would not be solved by ``exp(x/eps)``; ``beta = (1, 0)`` is used here (the legacy code did too).
    ``kappa`` is a DG0 tensor field (cell-wise constant, so the SWIP weights see the jump); the start
    mesh must have an even number of cells per side so that ``x = 1/2`` stays a set of mesh facets
    (bisection keeps it so).
    """

    eps1: float = 0.1
    eps2: float = 1.0
    velocity: tuple[float, ...] = (1.0, 0.0)
    name = "heterogeneous"
    dim = 2

    def kappa(self, x: Any) -> Any:
        import ufl
        from dolfinx import fem, mesh
        from ufl.domain import extract_unique_domain

        ufl_domain = extract_unique_domain(x)
        domain = mesh.Mesh(ufl_domain.ufl_cargo(), ufl_domain)  # python wrapper of the same mesh
        k = fem.Function(fem.functionspace(domain, ("DG", 0)))
        k.interpolate(lambda y: np.where(y[0] < 0.5, self.eps1, self.eps2))  # cell midpoints
        return ufl.as_matrix([[k, 0.0], [0.0, 1.0]])

    def beta(self, x: Any) -> Any | None:
        import ufl

        return ufl.as_vector([float(v) for v in self.velocity])

    def exact(self, x: Any) -> Any:
        return heterogeneous_exact(x[0], self.eps1, self.eps2)

    def source(self, x: Any) -> Any | None:
        return None

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import crossed_rectangle

        if n % 2:
            raise ValueError("n must be even so that x = 1/2 follows cell edges")
        return crossed_rectangle(n, (0.0, 0.0), (1.0, 1.0), comm)


@dataclass
class Anisotropic(Problem):
    """-div(kappa grad u) = f on (-1,1) x (-1/2,1/2), ``kappa = diag(1, 1/r)`` (paper section 5.5).

    Manufactured solution ``u = exp(-(x^2 + r tau y^2)) sqrt(r tau)/(4 pi)`` (the constant of the original
    implementation; eq. (42) prints ``1/(4 pi sqrt(r tau))``, the normalisation is irrelevant); ``f`` and
    the Dirichlet data are derived from it. The inner layer has width ``1/sqrt(r tau)`` around ``y = 0``.
    """

    r: float = 1.0e4
    tau: float = 1.0e-3
    name = "anisotropic"
    dim = 2

    def kappa(self, x: Any) -> Any:
        import ufl

        return ufl.as_matrix([[1.0, 0.0], [0.0, 1.0 / self.r]])

    def exact(self, x: Any) -> Any:
        import ufl

        rt = self.r * self.tau
        return ufl.exp(-(x[0] ** 2 + rt * x[1] ** 2)) * float(np.sqrt(rt) / (4 * np.pi))

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import crossed_rectangle

        return crossed_rectangle(n, (-1.0, -0.5), (1.0, 0.5), comm)


def _box_markers(dim: int) -> dict[int, Marker]:
    """Tags 1..2 dim on the faces x_i = 0, x_i = 1 (i = 0, 1, (2)) of the unit square / cube."""
    markers: dict[int, Marker] = {}
    for i in range(dim):
        for j, val in enumerate((0.0, 1.0)):
            markers[2 * i + j + 1] = lambda x, i=i, val=val: np.isclose(x[i], val)
    return markers


@dataclass
class ErikssonJohnson(Problem):
    """Steady Eriksson-Johnson problem ``-kappa Delta u + beta . grad u = 0`` on the unit square.

    ``beta = (1, 0)``, ``f = 0`` and Dirichlet data on the four sides (tags 1-4: ``x = 0, 1``, ``y = 0, 1``) from
    the exact solution ``u = (exp(r1 (x-1)) - exp(r2 (x-1)))/(exp(-r1) - exp(-r2)) sin(pi y)`` with
    ``r1,2 = (1 +- sqrt(1 + 4 pi^2 kappa^2))/(2 kappa)``: a boundary layer of width ``1/r1 ~ kappa`` at the outflow
    ``x = 1`` (paper section 5.6; legacy ``ErikssonJhonsson``, default ``kappa = 1e-2``).
    """

    diffusivity: float = 1.0e-2
    velocity: tuple[float, ...] = (1.0, 0.0)
    name = "eriksson_johnson"
    dim = 2
    dirichlet_tags: tuple[int, ...] = (1, 2, 3, 4)

    def roots(self) -> tuple[float, float]:
        """``(r1, r2)``; ``r2`` uses the cancellation-free form ``-2 pi^2 kappa/(1 + sqrt(...))``."""
        k = float(self.diffusivity)
        root = float(np.sqrt(1 + 4 * np.pi**2 * k**2))
        return (1 + root) / (2 * k), -2 * np.pi**2 * k / (1 + root)

    def kappa(self, x: Any) -> Any:
        return float(self.diffusivity)

    def beta(self, x: Any) -> Any | None:
        import ufl

        return ufl.as_vector([float(v) for v in self.velocity])

    def exact(self, x: Any) -> Any:
        import ufl

        r1, r2 = self.roots()
        layer = (ufl.exp(r1 * (x[0] - 1)) - ufl.exp(r2 * (x[0] - 1))) / (np.exp(-r1) - np.exp(-r2))
        return layer * ufl.sin(ufl.pi * x[1])

    def source(self, x: Any) -> Any | None:
        return None  # f = 0

    def boundary_markers(self) -> dict[int, Marker]:
        return _box_markers(self.dim)

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import crossed_rectangle

        return crossed_rectangle(n, (0.0, 0.0), (1.0, 1.0), comm)


@dataclass
class ErikssonJohnson3D(ErikssonJohnson):
    """3D version on the unit cube (paper section 5.6): ``beta = (1, 0, 0)``, same exact solution (independent
    of ``z``), Dirichlet data on the six faces (tags 1-6). The legacy code only ran the 2D problem."""

    velocity: tuple[float, ...] = (1.0, 0.0, 0.0)
    name = "eriksson_johnson_3d"
    dim = 3
    dirichlet_tags: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import unit_cube

        return unit_cube(n, comm)


@dataclass
class PoissonSine(Problem):
    """-Delta u = 10 sin(pi x) sin(pi y) on the unit square, ``u = 0`` on the boundary (smooth case of the legacy
    ``test_poisson_FEM_solution``); exact solution ``10/(2 pi^2) sin(pi x) sin(pi y)``."""

    amplitude: float = 10.0
    name = "poisson_sine"
    dim = 2

    def exact(self, x: Any) -> Any:
        import ufl

        return self.amplitude / (2 * np.pi**2) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import crossed_rectangle

        return crossed_rectangle(n, (0.0, 0.0), (1.0, 1.0), comm)


class NonlinearProblem(Problem):
    """Interface of ``div(f(u)) - div(kappa grad u) = s`` (conservation law with diffusion, paper section 6).

    Override ``flux``/``dflux`` (UFL vectors; ``d2flux`` for the legacy linearisation). ``beta`` returns the constant
    *norm wind* of the Gram form (eq. 50, ``beta_inf = 1`` in the legacy code), not a convection velocity. The source
    defaults to ``div f(u_D) - div(kappa grad u_D)`` for the exact solution ``u_D``; ``initial_guess`` is a numpy
    callable on points of shape ``(3, N)``.
    """

    name = "nonlinear"
    dim = 2

    def flux(self, u: Any) -> Any:
        """Convective flux ``f(u)`` (UFL vector)."""
        raise NotImplementedError

    def dflux(self, u: Any) -> Any:
        """``f'(u)`` (UFL vector): the wind of the linearisation and of the Lax-Friedrichs dissipation."""
        raise NotImplementedError

    def d2flux(self, u: Any) -> Any:
        """``f''(u)`` (UFL vector), used by the legacy (frozen-wind) linearisation."""
        raise NotImplementedError

    def source(self, x: Any) -> Any | None:
        import ufl

        u = self.exact(x)
        return ufl.div(self.flux(u)) - ufl.div(self.kappa(x) * ufl.grad(u))

    def initial_guess(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[1], 0.5)


@dataclass
class BurgersIsotropic(NonlinearProblem):
    """Isotropic Burgers equation ``div(b u^2/2) - kappa Delta u = f`` on (0,1)^2 (paper section 6.5.1, eq. 63).

    ``b = (1, 1)``, ``kappa = 1e-3``, exact solution ``u = (1 - tanh((2x - y - 1/4)/sqrt(5 kappa)))/2`` (an inner layer
    along the line ``2x - y = 1/4``), ``f`` and the Dirichlet data on the four sides derived from it, initial guess 0.5.
    The norm wind of the Gram form is ``norm_velocity = (1, 1)`` (legacy ``beta_inf = 1``). Start mesh: 6 x 6.
    """

    diffusivity: float = 1.0e-3
    velocity: tuple[float, ...] = (1.0, 1.0)
    norm_velocity: tuple[float, ...] = (1.0, 1.0)
    name = "burgers_isotropic"
    dim = 2
    dirichlet_tags: tuple[int, ...] = (1, 2, 3, 4)

    def kappa(self, x: Any) -> Any:
        return float(self.diffusivity)

    def beta(self, x: Any) -> Any | None:
        import ufl

        return ufl.as_vector([float(v) for v in self.norm_velocity])

    def flux(self, u: Any) -> Any:
        import ufl

        return ufl.as_vector([float(v) for v in self.velocity]) * (u * u / 2)

    def dflux(self, u: Any) -> Any:
        import ufl

        return ufl.as_vector([float(v) for v in self.velocity]) * u

    def d2flux(self, u: Any) -> Any:
        import ufl

        return ufl.as_vector([float(v) for v in self.velocity])

    def exact(self, x: Any) -> Any:
        import ufl

        return 0.5 * (1 - ufl.tanh((2 * x[0] - x[1] - 0.25) / float(np.sqrt(5 * self.diffusivity))))

    def boundary_markers(self) -> dict[int, Marker]:
        return _box_markers(self.dim)

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import crossed_rectangle

        return crossed_rectangle(n, (0.0, 0.0), (1.0, 1.0), comm)


@dataclass
class BurgersSingle(NonlinearProblem):
    """Single-component Burgers equation ``f(u) = (u^2/2, u)``, ``div f(u) = kappa Delta u`` on (0,1)^2 (section 6.5.2).

    Dirichlet data ``u_D = 1 - 2x`` on ``y = 0``, ``x = 0``, ``x = 1`` (tags 3, 1, 2) and the zero-diffusive-flux
    outflow boundary ``y = 1`` (tag 4, ``outflow_tags``); initial guess ``u0 = 1 - 2x``; no exact solution. The
    shock sits at ``x = 1/2`` where ``u0`` changes sign (the paper text says ``x = 0``) with ``max |u_x| = 1/(2 kappa)``.
    Defaults: ``kappa = 1e-2``, 4 x 4 start mesh, norm wind (1, 1).
    """

    diffusivity: float = 1.0e-2
    norm_velocity: tuple[float, ...] = (1.0, 1.0)
    name = "burgers_single"
    dim = 2
    dirichlet_tags: tuple[int, ...] = (1, 2, 3)
    outflow_tags: tuple[int, ...] = (4,)
    has_exact = False

    def kappa(self, x: Any) -> Any:
        return float(self.diffusivity)

    def beta(self, x: Any) -> Any | None:
        import ufl

        return ufl.as_vector([float(v) for v in self.norm_velocity])

    def flux(self, u: Any) -> Any:
        import ufl

        return ufl.as_vector([u * u / 2, u])

    def dflux(self, u: Any) -> Any:
        import ufl

        return ufl.as_vector([u, 1.0])

    def d2flux(self, u: Any) -> Any:
        import ufl

        return ufl.as_vector([1.0, 0.0])

    def dirichlet(self, x: Any) -> Any:
        return 1 - 2 * x[0]

    def source(self, x: Any) -> Any | None:
        return None

    def initial_guess(self, x: np.ndarray) -> np.ndarray:
        return 1 - 2 * x[0]

    def boundary_markers(self) -> dict[int, Marker]:
        return _box_markers(self.dim)

    def initial_mesh(self, n: int, comm: Any = None) -> Any:
        from adaptdg.vms.meshes import crossed_rectangle

        return crossed_rectangle(n, (0.0, 0.0), (1.0, 1.0), comm)


PROBLEMS: dict[str, type[Problem]] = {
    "reentrant_corner": ReentrantCorner,
    "fichera": Fichera3D,
    "heterogeneous": Heterogeneous,
    "anisotropic": Anisotropic,
    "eriksson_johnson": ErikssonJohnson,
    "eriksson_johnson_3d": ErikssonJohnson3D,
    "poisson_sine": PoissonSine,
    "burgers_isotropic": BurgersIsotropic,
    "burgers_single": BurgersSingle,
}

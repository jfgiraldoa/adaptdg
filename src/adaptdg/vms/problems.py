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


PROBLEMS: dict[str, type[Problem]] = {
    "reentrant_corner": ReentrantCorner,
    "fichera": Fichera3D,
    "heterogeneous": Heterogeneous,
    "anisotropic": Anisotropic,
}

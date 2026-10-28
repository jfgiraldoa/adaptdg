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


PROBLEMS: dict[str, type[Problem]] = {
    "reentrant_corner": ReentrantCorner,
    "fichera": Fichera3D,
}

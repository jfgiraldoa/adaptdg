"""Presets: the choices of the paper versus those of the original (legacy) implementation.

Each difference is one field of :class:`Preset`, so mixed variants are obtained with
``resolve_preset("legacy", gram_l2=True)``. ``"paper"`` is the package default and reproduces the
behaviour of every release so far; ``"legacy"`` reproduces the original code (see the table
"Legacy and paper presets" in ``docs/VMS.md``). Pure Python: no dolfinx import.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Preset:
    """Switchable choices; ``None`` means "use the value of ``LoopSettings``"."""

    name: str = "paper"
    #: L2 term ``(a, c)`` in the Gram form of pure diffusion problems (with advection it is always present).
    gram_l2: bool = True
    #: factor on the interior harmonic mean ``2 k1 k2/(k1+k2)`` in the SWIP penalty of ``b`` (legacy 0.5).
    gamma_factor_b: float = 1.0
    #: same factor in the jump term of the Gram form.
    gamma_factor_gram: float = 1.0
    #: quadrature degree of the data and of the error norms (legacy 8).
    quadrature_degree: int | None = None
    #: "dorfler" (paper) or "fixed_fraction" (legacy REF_TYPE 2, bands of the maximum indicator).
    marking: str = "dorfler"
    #: ratio of the fixed-fraction marking (legacy REFINE_RATIO, 0.25 in 2D); ``None``: ``theta_ref``.
    marking_ratio: float | None = None
    #: "paper" ``b(eps', v)`` or "legacy" ``b(v, eps')`` in the reconstruction.
    adjoint_ordering: str | None = None
    #: convective flux of nonlinear problems: "lax_friedrichs" (paper eqs. 45-47, conservative form) or "wind"
    #: (legacy ``bh_dar_NL``: the wind ``f'(u)`` through the generic upwind form, non-conservative volume term).
    nonlinear_flux: str = "lax_friedrichs"
    #: Newton variant: "hybrid" (Gauss-Newton Jacobian far from the solution, full Jacobian near it, undamped) or
    #: "damped" (legacy ``Damping_solve``: Gauss-Newton Jacobian, damping ``1/(1 + K |R|)``).
    newton_method: str = "hybrid"

    def __post_init__(self) -> None:
        if self.marking not in ("dorfler", "fixed_fraction"):
            raise ValueError("marking must be 'dorfler' or 'fixed_fraction'")
        if self.adjoint_ordering not in (None, "paper", "legacy"):
            raise ValueError("adjoint_ordering must be None, 'paper' or 'legacy'")
        if self.nonlinear_flux not in ("lax_friedrichs", "wind"):
            raise ValueError("nonlinear_flux must be 'lax_friedrichs' or 'wind'")
        if self.newton_method not in ("hybrid", "damped", "full"):
            raise ValueError("newton_method must be 'hybrid', 'damped' or 'full'")


PRESETS: dict[str, Preset] = {
    "paper": Preset(),
    "legacy": Preset(name="legacy", gram_l2=False, gamma_factor_b=0.5, gamma_factor_gram=0.5,
                     quadrature_degree=8, marking="fixed_fraction", marking_ratio=0.25,
                     adjoint_ordering="legacy", nonlinear_flux="wind", newton_method="damped"),
}


def resolve_preset(preset: str | Preset | None = "paper", **overrides: object) -> Preset:
    """Return the :class:`Preset` named ``preset`` (or the instance) with ``overrides`` applied."""
    if preset is None:
        preset = "paper"
    if isinstance(preset, str):
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        preset = PRESETS[preset]
    unknown = set(overrides) - {f.name for f in fields(Preset)}
    if unknown:
        raise KeyError(f"Unknown preset fields: {sorted(unknown)}")
    return replace(preset, **overrides) if overrides else preset

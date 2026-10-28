"""Cell marking strategies (Dörfler / bulk marking).

Extracted from ASRTM-PHREEQC without changing the numerics.
Two layers:

* ``dorfler_union``: generic, numpy-only core (species union, eligibility by
  cell size, species skip by ``epsilon``, tie extension, optional post-mark
  filter). It knows nothing about chemistry: the application supplies
  ``min_size`` (e.g. factor * chemistry cell size) and the species keys are
  opaque strings.
* Thin adapters ``dorfler_mark_1d``, ``dorfler_union_2d``, ``dorfler_union_momas``
  that reproduce the source variants exactly (``np.array_equal``), and
  ``dorfler_with_ties``, a verbatim copy because its algorithm differs
  (see docs/DESIGN.md). Defaults are deliberately NOT unified.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


def dorfler_union(
    indicators: Mapping[str, np.ndarray],
    sizes: np.ndarray | None,
    min_size: float | np.ndarray,
    theta: float,
    epsilon: float,
    tie_tol: float | None,
    *,
    eligibility: np.ndarray | None = None,
    post_mark_filter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Species-wise union of chemistry-aware Dörfler markings.

    ``indicators`` maps a species key to per-cell squared indicators (their
    absolute value is used). A cell is eligible if ``sizes > min_size``
    (``min_size`` scalar or per-cell array; strict comparison) or, when given,
    ``eligibility`` (boolean mask, overrides ``sizes``/``min_size``). For each
    species the marking is skipped if its total is <= 0 or the eligible share
    of the total is < ``epsilon``; otherwise eligible cells are marked in
    decreasing indicator order until the accumulated sum reaches
    ``theta**2 * total``. If ``tie_tol`` is not None, following eligible cells
    are also marked while ``cutoff - (1 + tie_tol) * value < 0`` (cutoff = last
    marked value). ``post_mark_filter(marked) -> marked`` is applied last and
    must return the same shape.
    """

    if eligibility is not None:
        eligible = np.asarray(eligibility, dtype=bool)
        n = eligible.size
    else:
        sizes_arr = np.asarray(sizes, dtype=np.float64)
        eligible = sizes_arr > min_size
        n = sizes_arr.size
    marked = np.zeros(n, dtype=bool)

    for name, raw in indicators.items():
        values = np.abs(np.asarray(raw, dtype=np.float64))
        if values.shape != marked.shape:
            raise ValueError(f"Indicator shape mismatch for {name}")
        total = float(np.sum(values))
        if total <= 0.0:
            continue
        if float(np.sum(values[eligible])) / total < epsilon:
            continue

        target = theta**2 * total
        order = np.argsort(values)[::-1]
        accumulated = 0.0
        cutoff: float | None = None
        last_position = -1
        for position, cell in enumerate(order):
            if not eligible[cell]:
                continue
            marked[cell] = True
            accumulated += float(values[cell])
            cutoff = float(values[cell])
            last_position = position
            if accumulated >= target:
                break

        if cutoff is None or tie_tol is None:
            continue
        for cell in order[last_position + 1 :]:
            if not eligible[cell]:
                continue
            if cutoff - (1.0 + tie_tol) * values[cell] < 0.0:
                marked[cell] = True
            else:
                break

    if post_mark_filter is not None:
        filtered = np.asarray(post_mark_filter(marked), dtype=bool)
        if filtered.shape != marked.shape:
            raise ValueError("post_mark_filter returned an incompatible shape")
        marked = filtered
    return marked


def dorfler_union_2d(
    indicators: Mapping[str, np.ndarray],
    cell_sizes: np.ndarray,
    *,
    chemistry_width: float,
    eligibility_factor: float,
    theta: float,
    epsilon_chem: float,
    tie_tolerance: float,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """Adapter equal to ``_dorfler_union_2d`` (adaptive_calcite_dolomite_2d.py).

    ``names`` selects the species (default: all keys of ``indicators``).
    """

    keys = list(indicators) if names is None else list(names)
    return dorfler_union(
        {k: indicators[k] for k in keys},
        cell_sizes,
        eligibility_factor * chemistry_width,
        theta,
        epsilon_chem,
        tie_tolerance,
    )


def dorfler_union_momas(
    indicators: Sequence[np.ndarray],
    sizes: np.ndarray,
    h_chem: float,
    theta: float,
    eligibility_factor: float,
    epsilon_chem: float,
    tie_tolerance: float,
) -> np.ndarray:
    """Adapter equal to ``methods/momas2d/adap.py::dorfler_union`` (list of arrays)."""

    return dorfler_union(
        {i: v for i, v in enumerate(indicators)},
        sizes,
        eligibility_factor * h_chem,
        theta,
        epsilon_chem,
        tie_tolerance,
    )


def dorfler_union_1d_adap(
    indicators: Mapping[str, np.ndarray],
    widths: np.ndarray,
    chem_width: float,
    *,
    theta: float,
    eligibility_factor: float,
    epsilon_chem: float,
    tie_tolerance: float,
    components: Sequence[str] | None = None,
) -> np.ndarray:
    """Adapter equal to ``Adap1DRun._marking`` (methods/onedim/adap1d.py)."""

    keys = list(indicators) if components is None else list(components)
    return dorfler_union(
        {k: indicators[k] for k in keys},
        widths,
        eligibility_factor * chem_width,
        theta,
        epsilon_chem,
        tie_tolerance,
    )


def dorfler_mark_1d(
    indicators: np.ndarray,
    cell_widths: np.ndarray,
    *,
    theta: float,
    minimum_width: float,
    epsilon_eligible: float = 0.0,
) -> np.ndarray:
    """Dörfler-mark eligible interval cells using residual indicators.

    Adapter equal to ``asrtm_phreeqc.porosity.transport_1d.dorfler_mark_1d``:
    same validation, eligibility ``width > minimum_width * (1 + 1e-12)`` and no
    tie extension.
    """

    eta = np.asarray(indicators, dtype=float)
    widths = np.asarray(cell_widths, dtype=float)
    if eta.ndim != 1 or widths.shape != eta.shape:
        raise ValueError("indicators and cell_widths must be matching vectors")
    if not np.all(np.isfinite(eta)) or np.any(eta < 0.0) or np.any(widths <= 0.0):
        raise ValueError("Indicators must be finite/nonnegative and widths positive")
    if not 0.0 < theta <= 1.0 or minimum_width <= 0.0:
        raise ValueError("theta and minimum_width must be positive (theta <= 1)")
    if not 0.0 <= epsilon_eligible <= 1.0:
        raise ValueError("epsilon_eligible must lie in [0, 1]")
    return dorfler_union(
        {"eta": eta},
        widths,
        minimum_width * (1.0 + 1e-12),
        theta,
        epsilon_eligible,
        None,
    )


def dorfler_with_ties(squared, *, fraction, tie, eligible):
    """Verbatim copy of ``case4dev/adap_asfem.py::dorfler_with_ties``.

    Kept verbatim (not an adapter): different algorithm (target ``fraction *
    total`` without squaring, cumulative-sum search, tie rule
    ``values >= (1 - tie)**2 * values[stop]``, no epsilon, ``eligible`` mask).
    """
    values = np.abs(np.asarray(squared, dtype=float))
    marked = np.zeros(values.size, dtype=bool)
    total = float(values.sum())
    if total <= 0.0:
        return marked
    order = np.argsort(values)[::-1]
    order = order[np.asarray(eligible, bool)[order]]
    if order.size == 0:
        return marked
    cumulative = np.cumsum(values[order])
    stop = min(int(np.searchsorted(cumulative, fraction * total, side="left")), order.size - 1)
    marked[order[: stop + 1]] = True
    threshold = (1.0 - tie) ** 2 * values[order[stop]]
    marked[order[values[order] >= threshold]] = True
    return marked


def fixed_fraction_bands(
    indicators: np.ndarray,
    ratio: float = 0.25,
    *,
    frac: float = 0.95,
    delfrac: float = 0.05,
) -> np.ndarray:
    """Fixed-fraction marking by bands of the maximum indicator (legacy ASFEM ``REF_TYPE 2``).

    Starting at ``frac`` and lowering it by ``delfrac`` each step, the not yet marked cells with
    ``|indicator| > frac * max`` are marked, until the marked indicators add up to at least
    ``ratio * total`` (``ratio`` is applied to the sum of the squared indicators, not squared
    itself, unlike the Dörfler parameter). Returns a boolean mask; all False if the total is zero.
    """
    values = np.abs(np.asarray(indicators, dtype=np.float64))
    marked = np.zeros(values.shape, dtype=bool)
    total = float(values.sum())
    if total <= 0.0:
        return marked
    if not 0.0 < ratio <= 1.0 or delfrac <= 0.0:
        raise ValueError("ratio must lie in (0, 1] and delfrac must be positive")
    top = float(values.max())
    accumulated = 0.0
    while accumulated < ratio * total and not marked.all():
        new = ~marked & (values > frac * top)
        accumulated += float(values[new].sum())
        marked |= new
        frac -= delfrac
    return marked

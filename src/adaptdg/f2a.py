"""History-preserving mesh rebuild (F2A) of the 2D adaptive loop.

Extracted from ASRTM-PHREEQC (``methods/front2d/f2a.py``). There the three changes below were
applied to the repository functions as source-text replacements at run time; here they are
native options of :func:`adaptdg.loop.run_adaptive_loop_2d` (``history_policy="f2a"``), with
the same operations in the same order. Without F2A, the DG1 transport history of the
persistent core/halo mesh is lost in two places:

P1 (predictive-mesh rebuild, every ``mesh_rebuild_interval`` steps): the solver receives no
   history and rebuilds it from chemistry-grid cell averages (DG0 at chemistry resolution).
   With F2A the current and older DG1 histories are projected exactly (L2, DG1 to DG1,
   conservative per new triangle; ``adaptdg.history_projection``) from the old persistent
   mesh to the rebuilt mesh, then nonnegative-scaled (Zhang-Shu, mean kept).
P2 (in-step refinement): the complete history is discarded and the children get
   chemistry-grid averages. With F2A the children receive the exact restriction of their
   parent's DG1 history (the chemistry-based child transfer is used only when there is no
   complete history).
P3 (restart): a run resumed exactly at a rebuild step also rebuilds its last committed mesh,
   so that P1 has an old mesh to project the histories from.

``F2AOptions`` switches each change on or off. ``F2AHistory`` holds the per-run state
(statistics and a one-entry cache of the DG1 dof-coordinate vertices). With ``optimised=True``
(used together with the fast path, ``adaptdg.fastadap``) identical triangles are copied
instead of projected and the restriction weights are computed once per refinement; results
agree with the plain path to round-off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from adaptdg.history_projection import (
    build_plan,
    canonical_vertices,
    dof_triangles,
    scale_nonnegative,
)


@dataclass(frozen=True)
class F2AOptions:
    """Which of the three F2A changes are active (all on by default)."""

    project_on_rebuild: bool = True  # P1
    restrict_on_refinement: bool = True  # P2
    rebuild_mesh_on_restart: bool = True  # P3


def _triangle_areas(t: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(
        (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
    )


def restrict_to_children(values, parent_triangles, child_triangles, parent_index):
    """Exact restriction of parent DG1 values (..., n_parent, 3) to children."""
    pc = canonical_vertices(parent_triangles)[parent_index]
    cc = canonical_vertices(child_triangles)
    # barycentric coordinates of each child vertex in its parent (vectorized)
    a = pc[:, 0]
    m = np.stack([pc[:, 1] - a, pc[:, 2] - a], axis=-1)  # (n, 2, 2)
    rhs = (cc - a[:, None, :]).transpose(0, 2, 1)  # (n, 2, 3)
    lam12 = np.linalg.solve(m, rhs).transpose(0, 2, 1)  # (n, 3, 2)
    lam = np.concatenate(
        [1.0 - lam12.sum(axis=-1, keepdims=True), lam12], axis=-1
    )  # (n, 3v, 3p)
    parent_vals = np.asarray(values)[..., parent_index, :]  # (..., n, 3p)
    return np.einsum("nvj,...nj->...nv", lam, parent_vals)


def _restriction_weights(pc, cc, parent_index):
    """Barycentric weights (n, 3 child vertices, 3 parent vertices) of the canonical child
    vertices cc in their canonical parent pc[parent_index] (same arithmetic as
    restrict_to_children)."""
    pc = pc[parent_index]
    a = pc[:, 0]
    m = np.stack([pc[:, 1] - a, pc[:, 2] - a], axis=-1)  # (n, 2, 2)
    rhs = (cc - a[:, None, :]).transpose(0, 2, 1)  # (n, 2, 3)
    lam12 = np.linalg.solve(m, rhs).transpose(0, 2, 1)  # (n, 3, 2)
    return np.concatenate([1.0 - lam12.sum(axis=-1, keepdims=True), lam12], axis=-1)


class F2AHistory:
    """Per-run state of the F2A history operations (P1 and P2)."""

    def __init__(self, components: Sequence[str], *, optimised: bool = False) -> None:
        self.components = tuple(components)
        self.optimised = bool(optimised)
        self.stats: dict[str, float] = {
            "rebuild_projections": 0,
            "rebuild_seconds": 0.0,
            "rebuild_scaled_cells": 0,
            "refinement_restrictions": 0,
            "refinement_seconds": 0.0,
            "max_coverage_error": 0.0,
            "max_rebuild_inventory_rel": 0.0,
        }
        self._dof_tri = {"domain": None, "tri": None}

    def dof_triangles(self, domain) -> np.ndarray:
        """``dof_triangles`` with a one-entry cache: a mesh is asked for twice in a row (as the
        new mesh of a projection or refinement and as the old mesh of the next one)."""
        if not self.optimised:
            return dof_triangles(domain)
        if self._dof_tri["domain"] is not domain:
            self._dof_tri.update(domain=domain, tri=dof_triangles(domain))
        return self._dof_tri["tri"]

    def histories(
        self,
        rebuilt: bool,
        current: Mapping[str, np.ndarray] | None,
        older: Mapping[str, np.ndarray] | None,
        old_domain,
        new_domain,
    ):
        """Histories to pass to the adaptive step (P1): the current ones on a reused mesh, the
        projection onto the rebuilt mesh at a rebuild, ``(None, None)`` when there is nothing
        to project from."""
        if not rebuilt:
            return current, older
        if current is None or old_domain is None or new_domain is None:
            return None, None
        stats = self.stats
        started = time.perf_counter()
        # dof-coordinate vertices: same per-cell order as the solver's DG1 history
        old_tri, new_tri = self.dof_triangles(old_domain), self.dof_triangles(new_domain)
        names = list(self.components)
        have_older = older is not None
        stack = np.stack(
            [current[n] for n in names] + ([older[n] for n in names] if have_older else [])
        )
        if self.optimised:
            from adaptdg.history_projection_fast import project_stack

            projected, new_areas, coverage_error, n_identical = project_stack(
                old_tri, new_tri, stack
            )
            stats["rebuild_identical_triangles"] = (
                stats.get("rebuild_identical_triangles", 0) + n_identical
            )
            if n_identical == 0:  # project_stack fell back to the full plan
                stats["rebuild_full_plan_fallbacks"] = (
                    stats.get("rebuild_full_plan_fallbacks", 0) + 1
                )
        else:
            plan = build_plan(old_tri, new_tri)
            projected = plan.apply(stack)
            new_areas, coverage_error = plan.new_areas, plan.coverage_error
        projected, scaled = scale_nonnegative(projected)
        oc = canonical_vertices(old_tri)
        e1, e2 = oc[:, 1] - oc[:, 0], oc[:, 2] - oc[:, 0]
        old_areas = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        before = (stack.mean(axis=-1) * old_areas).sum(axis=-1)
        after = (projected.mean(axis=-1) * new_areas).sum(axis=-1)
        rel = float(np.max(np.abs(after - before) / np.maximum(np.abs(before), 1e-300)))
        new_current = {n: projected[i] for i, n in enumerate(names)}
        new_older = (
            {n: projected[len(names) + i] for i, n in enumerate(names)} if have_older else None
        )
        stats["rebuild_projections"] += 1
        stats["rebuild_seconds"] += time.perf_counter() - started
        stats["rebuild_scaled_cells"] += scaled
        stats["max_coverage_error"] = max(stats["max_coverage_error"], coverage_error)
        stats["max_rebuild_inventory_rel"] = max(stats["max_rebuild_inventory_rel"], rel)
        return new_current, new_older

    def restrict(
        self,
        current: Mapping[str, np.ndarray],
        older: Mapping[str, np.ndarray] | None,
        domain,
        refined_domain,
        parent_index: np.ndarray,
    ):
        """Children inherit the exact restriction of the parent DG1 histories (P2).

        Returns ``(current, older, current_averages, older_averages)`` on the refined mesh
        (the averages are the per-triangle means, shape ``(components, n_children)``)."""
        stats = self.stats
        started = time.perf_counter()
        parent_triangles = self.dof_triangles(domain)
        child_triangles = self.dof_triangles(refined_domain)
        names = list(self.components)
        pc = canonical_vertices(parent_triangles)
        cc = canonical_vertices(child_triangles)
        if self.optimised:
            # one weight computation and one gather for all 4 (8 with the older state) fields
            lam = _restriction_weights(pc, cc, parent_index)
            fields = [current[n] for n in names] + (
                [] if older is None else [older[n] for n in names]
            )
            restricted = np.einsum(
                "nvj,fnj->fnv", lam, np.stack(fields)[:, parent_index, :]
            )
            new_current = {n: restricted[i] for i, n in enumerate(names)}
            new_older = (
                None
                if older is None
                else {n: restricted[len(names) + i] for i, n in enumerate(names)}
            )
        else:
            new_current = {
                n: restrict_to_children(current[n], parent_triangles, child_triangles, parent_index)
                for n in names
            }
            new_older = (
                None
                if older is None
                else {
                    n: restrict_to_children(older[n], parent_triangles, child_triangles, parent_index)
                    for n in names
                }
            )
        previous = np.stack([np.mean(new_current[n], axis=1) for n in names])
        # Exact restriction: parent integrals are reproduced by their children.
        before = np.stack([np.mean(current[n], axis=1) for n in names]) @ _triangle_areas(pc)
        after = previous @ _triangle_areas(cc)
        stats["max_refinement_inventory_rel"] = max(
            stats.get("max_refinement_inventory_rel", 0.0),
            float(np.max(np.abs(after - before) / np.maximum(np.abs(before), 1e-300))),
        )
        older_avg = (
            None
            if new_older is None
            else np.stack([np.mean(new_older[n], axis=1) for n in names])
        )
        stats["refinement_restrictions"] += 1
        stats["refinement_seconds"] += time.perf_counter() - started
        return new_current, new_older, previous, older_avg

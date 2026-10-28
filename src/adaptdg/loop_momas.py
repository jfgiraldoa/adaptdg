"""MoMaS Easy 2D ADAP loop (adaptive RM transport on a fixed chemistry grid), generic in the chemistry.

Extracted from ASRTM-PHREEQC (asrtm-phreeqc-public @ a152b91, ``methods/momas2d/adap.py``): the driver
function ``run`` (kept under its name) and its helpers ``singularity_cap``, ``MeshFields``,
``dg1_transfer``, ``full_dg1``, ``linear_remap``, ``remap_history`` and the constants ``POSITIVE`` and
``SINGULAR_POINTS``; numerics, names and operation order unchanged. ``fast_refine_marked_triangles_with_parent``
is copied verbatim from ``methods/front2d/fastadap.py`` (the leak-free refinement ``adap.py`` imports).
The argparse CLI (``main``) and the ``sys.path`` hacks are not part of the library.

State lives on the chemistry grid (squares h_c = 0.1/kc, one PHREEQC cell per square). Every step:
the transport mesh is the triangulated chemistry grid, up to ``solves`` RM solves with chemistry-aware
Dorfler marking, conservative transfer to the chemistry squares, positivity of the means of components
0, 2, 3, chemistry ``react`` with the conservation gates (see ``adap.py`` for the full description).

Deviations from the source (outside numerics):
- ``dorfler_union`` is ``adaptdg.marking.dorfler_union_momas`` (the exact adapter of the inline original).
- Imports point to adaptdg: ``adaptdg.flow`` (``darcy``), ``adaptdg.rm_transport``, ``adaptdg.transport``
  (``nonnegative_means``), ``adaptdg.transport_2d`` (``triangle_characteristic_sizes``),
  ``adaptdg.transfer_2d`` (``conservative_remap_plan_2d``).
- The chemistry enters through the ``chemistry`` keyword-only argument of ``run`` (protocol
  :class:`ChemistryMomas`) instead of ``EasyModel.read(.../spec.json)``, ``FastAnchorMomasPhreeqcRM`` and
  ``MomasEasyPhreeqcRM``; adaptdg imports no chemistry. The constants of the source construction
  (``concentration_scale=1e-6``, ``zero_tolerance=1e-16``, ``anchor_mode="one_end"``) are passed to it.
"""

# ruff: noqa: E702, F821, F841  (copied code kept verbatim: semicolons, names bound in the step loop)
from __future__ import annotations

from pathlib import Path
from typing import Any, ContextManager, Mapping, Protocol
import time as clock

import numpy as np

from adaptdg import flow as darcy
from adaptdg.marking import dorfler_union_momas as dorfler_union
from adaptdg.rm_transport import RMTransportMoMaS2D


class EngineMomas(Protocol):
    """Documentation of the duck-typed chemistry engine used by :func:`run` (never enforced).

    Source type: ``FastAnchorMomasPhreeqcRM`` (anchor_mode ``"one_end"``), a ``MomasEasyPhreeqcRM``.

    Attributes: ``snapshot`` (``.total`` (cells, 4), ``.mobile`` (cells, 4), ``.fixed`` (cells, 4)),
    ``water_volume_l`` and ``site_total`` (arrays over cells).
    ``react(mobile)`` takes the transported mobile totals (cells, 4) and returns an object with
    ``total``, ``mobile``, ``fixed``, ``primary``, ``sites`` (arrays over cells) and ``water_mass_kg``.
    ``dump_state(path)`` / ``restore_state(path)`` write / read the native chemistry state (checkpoints only).
    """

    snapshot: Any
    water_volume_l: np.ndarray
    site_total: np.ndarray

    def react(self, mobile: np.ndarray) -> Any: ...
    def dump_state(self, path: Any) -> None: ...
    def restore_state(self, path: Any) -> Any: ...


class ChemistryMomas(Protocol):
    """What :func:`run` needs from the chemistry set-up (replaces ``EasyModel`` + engine construction).

    ``spec``: the benchmark spec (dict as in ``examples/momas2d/spec.json``); only
    ``spec["materials"][m]["phi"]``, ``["alpha_advective"]``, ``["alpha_dispersive"]`` (m in "A", "B") and
    ``spec["time"]["snapshots"]`` and ``["injection_ends"]`` are read.
    ``open_chemistry(materials, representative_volume_l, ...)`` returns a context manager yielding an
    :class:`EngineMomas` with one cell per entry of ``materials`` ("A"/"B", flat chemistry index
    ``iy * nx + ix``) and the given representative volume per cell [L].
    ``inlet_mobile(concentration_scale=...)`` returns the (2, 4) mobile totals of the injection (row 0)
    and leaching (row 1) inlet solutions.
    """

    spec: Mapping[str, Any]

    def open_chemistry(
        self,
        materials: np.ndarray,
        representative_volume_l: np.ndarray,
        *,
        concentration_scale: float,
        zero_tolerance: float,
        anchor_mode: str,
    ) -> ContextManager[EngineMomas]: ...

    def inlet_mobile(self, *, concentration_scale: float) -> np.ndarray: ...

POSITIVE = (0, 2, 3)
# Geometric singularities of the data: ends of the inlet/outlet segments (jump of q.n) and corners of B.
SINGULAR_POINTS = np.array([[0.3, 1.0], [0.4, 1.0], [0.0, 0.3], [0.0, 0.4], [2.1, 0.2], [2.1, 0.4],
                            [1.0, 0.9], [1.1, 0.9], [1.0, 0.0], [1.1, 0.0]])


def singularity_cap(triangles, sizes, radius, min_size):
    """True where refinement is allowed: cells within `radius` of a singular point are not refined below
    `min_size` (maximum-level cap at singularities, declared in the paper)."""
    mid = triangles.mean(axis=1)
    d = np.min(np.linalg.norm(mid[:, None, :] - SINGULAR_POINTS[None, :, :], axis=2), axis=1)
    return ~((d < radius) & (sizes <= min_size))
class MeshFields:
    """Material fields, facet tags and the nested RT0 flux on one transport mesh."""

    def __init__(self, domain, q_base, regime, mat):
        from dolfinx import fem

        self.tags = darcy.material_tags(domain)
        n = self.tags.size
        V0 = fem.functionspace(domain, ("DG", 0))
        dofs = V0.dofmap.list[:n, 0]
        names = np.where(self.tags == 1, "B", "A")

        def cellf(vals):
            f = fem.Function(V0)
            f.x.array[dofs] = vals
            return f

        self.phi = cellf(np.array([mat[m]["phi"] for m in names]))
        self.aL = cellf(np.array([mat[m]["alpha_" + regime][0] for m in names]))
        self.aT = cellf(np.array([mat[m]["alpha_" + regime][1] for m in names]))
        self.facet_tags = darcy.momas_facet_tags(domain)
        V = fem.functionspace(domain, ("RT", 1))
        self.q = fem.Function(V)
        if q_base is None:  # Darcy on this transport mesh (divergence-free on it)
            K = cellf(np.array([darcy.MATERIALS[t][2] for t in self.tags]))
            self.q = darcy.solve_darcy(domain, K, self.facet_tags,
                                       flux_bc=((darcy.WALL, darcy.INLET1, darcy.INLET2), darcy.momas_flux_bc),
                                       head_tags={darcy.OUTLET_TAG: darcy.OUTLET_HEAD}).q
        elif q_base.function_space.mesh is domain:
            self.q.x.array[:] = q_base.x.array
        else:
            cells = np.arange(n, dtype=np.int32)
            data = fem.create_interpolation_data(V, q_base.function_space, cells, padding=1e-12)
            self.q.interpolate_nonmatching(q_base, cells, data)


def dg1_transfer(old_fun, new_space):
    """Interpolate a DG1 field onto a nested refinement (exact: the coarse DG1 space is contained)."""
    from dolfinx import fem

    n = new_space.mesh.topology.index_map(2).size_local
    cells = np.arange(n, dtype=np.int32)
    new = fem.Function(new_space)
    new.interpolate_nonmatching(old_fun, cells, fem.create_interpolation_data(new_space, old_fun.function_space,
                                                                               cells, padding=1e-12))
    return new


def full_dg1(op, full):
    """The 4 transported DG1 fields of the last solve (u_bar interpolated, or u_bar + u_tilde) as Functions."""
    from dolfinx import fem

    out = []
    for j in range(full.shape[0]):
        f = fem.Function(op.previous.function_space)
        f.x.array[:] = full[j]
        out.append(f)
    return out


def linear_remap(op, fields, mean_chem, X, Y, h_c, w_chem, nx, ny, kb):
    """Transport -> chemistry with the DG1 field instead of the triangle means.

    Squares lying in a triangle larger than themselves get the DG1 value at their centre (exact average of a
    linear field over a square inside one triangle); squares in the refined zone keep the conservative overlap
    mean. Then a constant per base square (0.1/kb, union of triangles and of squares) restores the exact
    inventory of the overlap means, so the transfer stays conservative."""
    from dolfinx import geometry

    dom = op.domain
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    tree = geometry.bb_tree(dom, 2)
    coll = geometry.compute_colliding_cells(dom, geometry.compute_collisions_points(tree, pts), pts)
    cells = np.array([coll.links(i)[0] for i in range(len(pts))], dtype=np.int32)
    big = op.areas[cells] > 1.001 * h_c * h_c
    out = mean_chem.copy()
    for j, f in enumerate(fields):
        vals = f.eval(pts, cells)[:, 0]
        out[big, j] = vals[big]
    h_b = 0.1 / kb
    grp = (np.floor(Y.ravel() / h_b).astype(int) * (21 * kb) + np.floor(X.ravel() / h_b).astype(int))
    ng = grp.max() + 1
    wsum = np.bincount(grp, weights=w_chem, minlength=ng)
    for j in range(out.shape[1]):
        target = np.bincount(grp, weights=w_chem * mean_chem[:, j], minlength=ng)
        now = np.bincount(grp, weights=w_chem * out[:, j], minlength=ng)
        out[:, j] += ((target - now) / wsum)[grp]
    return out


def remap_history(old_hist, old_space, old_tri, old_sq, new_space, new_tri, new_sq, w_sq_density):
    """Old DG1 history -> new (non-nested) mesh: dolfinx interpolation, then a constant per triangle so that
    every chemistry square keeps its inventory exactly (squares are unions of triangles in both meshes).
    Squares where that makes a mean negative fall back to a constant equal to the square's old mean."""
    from dolfinx import fem

    def areas(t):
        d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        return 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])

    a_old, a_new = areas(old_tri), areas(new_tri)
    nsq = int(max(old_sq.max(), new_sq.max())) + 1
    od, nd = old_space.dofmap.list[: len(old_tri)], new_space.dofmap.list[: len(new_tri)]
    n = len(new_tri)
    cells = np.arange(n, dtype=np.int32)
    # interpolate into DG2 on the new mesh, then cellwise L2 projection onto DG1 (keeps the DG2 cell means,
    # much closer to the true overlap averages than pointwise DG1 interpolation)
    import ufl
    from dolfinx.fem import petsc
    from petsc4py import PETSc
    mesh_new = new_space.mesh
    V2 = fem.functionspace(mesh_new, ("DG", 2))
    data = fem.create_interpolation_data(V2, old_space, cells, padding=1e-12)
    u1, v1 = ufl.TrialFunction(new_space), ufl.TestFunction(new_space)
    M = petsc.assemble_matrix(fem.form(u1 * v1 * ufl.dx)); M.assemble()
    ksp = PETSc.KSP().create(mesh_new.comm); ksp.setOperators(M); ksp.setType("preonly"); ksp.getPC().setType("lu")
    f2 = fem.Function(V2)
    rhs = fem.form(f2 * v1 * ufl.dx)
    bvec = petsc.create_vector(new_space)
    f_old, f_new = fem.Function(old_space), fem.Function(new_space)
    out = np.zeros((old_hist.shape[0], new_space.dofmap.index_map.size_local))
    sq_area = np.bincount(new_sq, weights=a_new, minlength=nsq)
    for j in range(old_hist.shape[0]):
        f_old.x.array[:] = old_hist[j]
        f2.interpolate_nonmatching(f_old, cells, data)
        with bvec.localForm() as bl:
            bl.set(0.0)
        petsc.assemble_vector(bvec, rhs)
        ksp.solve(bvec, f_new.x.petsc_vec)
        f_new.x.scatter_forward()
        target = np.bincount(old_sq, weights=a_old * old_hist[j][od].mean(axis=1), minlength=nsq)
        now = np.bincount(new_sq, weights=a_new * f_new.x.array[nd].mean(axis=1), minlength=nsq)
        shift = np.divide(target - now, sq_area, out=np.zeros(nsq), where=sq_area > 0)
        vals = f_new.x.array.copy()
        vals[nd] += shift[new_sq][:, None]
        if j in POSITIVE:
            bad = np.unique(new_sq[vals[nd].mean(axis=1) < 0.0])
            if bad.size:
                sel = np.isin(new_sq, bad)
                vals[nd[sel]] = (target / np.where(sq_area > 0, sq_area, 1.0))[new_sq[sel]][:, None]
        out[j] = vals
    ksp.destroy(); M.destroy(); bvec.destroy()
    return out


def fast_refine_marked_triangles_with_parent(domain, marked):
    """Same refinement and parent map as the repository, without the MPI leak.

    dolfinx 0.11 ``mesh.refine`` with its default partitioner (ghost-layer
    update) leaves one MPI communicator behind per call; MPICH allows ~2048,
    which aborted F4 at step 1784. In serial, ``partitioner=None`` builds the
    bitwise-identical mesh and parent map and frees everything. The edge
    selection is vectorised (same sorted unique edges).
    """
    from dolfinx import mesh
    local_cells = domain.topology.index_map(2).size_local
    markers = np.asarray(marked, dtype=bool)
    if markers.shape != (local_cells,):
        raise ValueError("There must be one refinement marker per local triangle")
    if not np.any(markers):
        return domain, np.arange(local_cells, dtype=np.int32)
    if domain.comm.size != 1:
        raise RuntimeError("The leak-free refinement path is serial only")
    domain.topology.create_connectivity(2, 1)
    cell_to_edge = domain.topology.connectivity(2, 1)
    if cell_to_edge is None:
        raise RuntimeError("DOLFINx did not construct cell-to-edge connectivity")
    offsets = np.asarray(cell_to_edge.offsets)
    if not np.all(np.diff(offsets) == 3):
        raise RuntimeError("Unexpected cell-to-edge layout")
    selected_edges = np.unique(
        np.asarray(cell_to_edge.array).reshape(-1, 3)[np.flatnonzero(markers)].ravel()).astype(np.int32)
    refined, parent_cell, _ = mesh.refine(domain, selected_edges, partitioner=None,
                                          option=mesh.RefinementOption.parent_cell)
    parent_cell = np.asarray(parent_cell, dtype=np.int32)
    input_cell_index = np.asarray(refined.topology.original_cell_index, dtype=np.int64)
    if input_cell_index.shape != parent_cell.shape:
        raise RuntimeError("Refined-cell permutation has an incompatible shape")
    if np.any(input_cell_index < 0) or np.any(input_cell_index >= parent_cell.size):
        raise RuntimeError("Refined-cell permutation contains invalid indices")
    return refined, parent_cell[input_cell_index]


def run(kc, regime, dt, end_time, save_times=(), solves=3, theta=0.5, eligibility_factor=0.24,
        epsilon_chem=0.5, tie_tolerance=0.35, vms=False, chem_volume_l=10.0, label="", history="persistent", rebuild_every=8, kb=None,
        remap="linear", marking="union", norm_phi=True, cap_before=False,
        sing_radius=0.0, sing_min_size=0.0, checkpoint_dir=None, checkpoint_every=0, *, chemistry):
    from dolfinx import fem
    from dolfinx import mesh as dmesh
    from adaptdg.transport import nonnegative_means
    from adaptdg.transport_2d import triangle_characteristic_sizes
    from adaptdg.transfer_2d import conservative_remap_plan_2d

    spec = chemistry.spec
    mat = spec["materials"]
    h_c = 0.1 / kc
    nx, ny = 21 * kc, 10 * kc
    xe, ye = np.linspace(0.0, darcy.LX, nx + 1), np.linspace(0.0, darcy.LY, ny + 1)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    X, Y = np.meshgrid(xc, yc)  # (ny, nx), the transfer_2d layout; chemistry index = iy * nx + ix
    (b0, b1), (c0, c1) = darcy.BLOCK_B
    chem_mat = np.where((X > b0) & (X < b1) & (Y > c0) & (Y < c1), "B", "A").ravel()
    phi_c = np.array([mat[m]["phi"] for m in chem_mat])
    w_chem = phi_c * h_c * h_c

    kb = kc if kb is None else kb  # transport base mesh: 21kb x 10kb squares (kb <= kc: coarse transport base)
    h_b = 0.1 / kb
    if kb == kc:
        flow = darcy.solve_momas_darcy(kc)
        q_src = flow.q
        base = flow.q.function_space.mesh
    else:
        q_src = None
        base = darcy.create_momas_mesh(kb)
    base_fields = MeshFields(base, q_src, regime, mat)
    persistent = history == "persistent"
    species_weights = np.full(4, 0.25)  # reactivity weights for marking == "reactive" (updated every step)

    def square_of(tri):
        mid = tri.mean(axis=1)
        return (np.clip(np.floor(mid[:, 1] / h_b), 0, 10 * kb - 1).astype(int) * (21 * kb)
                + np.clip(np.floor(mid[:, 0] / h_b), 0, 21 * kb - 1).astype(int))
    cur_domain, cur_fields, hist = base, base_fields, None  # hist: (4, DG1 dofs) on cur_domain

    stats = {"max_remap_shift": 0.0, "max_reaction_defect": 0.0, "max_site_defect": 0.0, "max_water_change": 0.0,
             "max_positivity_correction": 0.0, "max_transport_balance_defect": 0.0,
             "triangles_final_mean": 0.0, "triangles_final_max": 0, "min_transported": 0.0,
             "seconds_transport": 0.0, "seconds_chemistry": 0.0}
    saved = {}
    t0 = clock.perf_counter()
    with chemistry.open_chemistry(chem_mat, np.full(chem_mat.size, chem_volume_l),
                                  concentration_scale=1e-6, zero_tolerance=1e-16, anchor_mode="one_end") as chem:
        initial = w_chem @ chem.snapshot.total
        inlet = chemistry.inlet_mobile(concentration_scale=1e-6)
        events = sorted(set(spec["time"]["snapshots"] + [spec["time"]["injection_ends"], end_time]
                            + [t for t in save_times if t <= end_time]))
        nsteps = int(np.ceil(end_time / dt - 1e-9))
        time, step = 0.0, 0
        ckpt = None if checkpoint_dir is None else Path(checkpoint_dir)
        if ckpt is not None and (ckpt / "state.npz").exists():
            # Resume: chemistry state is exact (PhreeqcRM dump); the transport history restarts from the
            # chemistry means on the base mesh (one-time loss of the DG1 slopes; conservative, order-safe).
            zc = np.load(ckpt / "state.npz", allow_pickle=True)
            time, step = float(zc["time"]), int(zc["step"])
            initial = zc["initial"]
            stats.update(zc["stats"].item())
            saved = zc["saved"].item()
            chem.restore_state(ckpt / "chem.dump")
            print(f"[progress] {label} resumed at step {step}, t = {time:g} (history restarted from chemistry)",
                  flush=True)
        while time < end_time - 1e-12:
            next_event = min(e for e in events if e > time + 1e-9)
            dtc = min(dt, next_event - time)
            values = inlet[int(time >= spec["time"]["injection_ends"])]
            resident = chem.snapshot.fixed.copy()
            prev_rect = chem.snapshot.mobile.T.reshape(4, ny, nx)
            ts = clock.perf_counter()
            t_next = time + dtc
            if abs(t_next - next_event) < 1e-9:  # same snapping as the time update below (round-off over many steps)
                t_next = float(next_event)
            save_now = any(abs(t_next - st) < 1e-10 for st in list(save_times) + [end_time])
            first = persistent and hist is None
            # saved steps always rebuild, so their full refinement sequence is available
            rebuild = persistent and not first and (step % rebuild_every == 0 or save_now)
            if persistent and not rebuild and hist is not None:
                domain, fields, nsolves = cur_domain, cur_fields, 1
            else:
                domain, fields, nsolves = base, base_fields, solves
            if rebuild:  # keep the old state to remap onto every new level mesh
                old_state = (hist, op_last_space, op_last_tri, square_of(op_last_tri))
            levels_store = []
            for level in range(nsolves):
                if level > 0:
                    op.destroy()
                op = RMTransportMoMaS2D(domain, q=fields.q, porosity=fields.phi, alpha_L=fields.aL,
                                        alpha_T=fields.aT, facet_tags=fields.facet_tags,
                                        inlet_tags=(darcy.INLET1, darcy.INLET2), time_step=dtc,
                                        history_degree=1 if persistent else 0, norm_phi=norm_phi)
                Vh = op.previous.function_space
                plan = conservative_remap_plan_2d(op.triangles, xe, ye)
                prev_tri = plan.rectangles_to_triangles(prev_rect)
                Vh = op.previous.function_space
                hdofs = Vh.dofmap.list[: op.cells]
                if first:  # first step: constant per triangle from the chemistry, on every level mesh
                    hist = np.zeros((4, Vh.dofmap.index_map.size_local))
                    for j in range(4):
                        hist[j][hdofs] = prev_tri[j][:, None]
                elif rebuild:  # rebuild: remap the old state onto this level's mesh (conservative per square)
                    hist = remap_history(old_state[0], old_state[1], old_state[2], old_state[3],
                                         Vh, op.triangles, square_of(op.triangles), None)
                w_tri = fields.phi.x.array[fields.phi.function_space.dofmap.list[: op.cells, 0]] * op.areas
                results, transported = [], np.zeros((4, op.cells))
                full = np.zeros((4, Vh.dofmap.index_map.size_local)) if persistent else None
                for j in range(4):
                    r = op.step(hist[j] if persistent else prev_tri[j], values[j])
                    results.append(r)
                    if vms:  # extension (not the paper's method): transport u_bar + u_tilde = u_dG
                        u = op.fine_scale()
                        transported[j] = u.x.array[op._Vd.dofmap.list[: op.cells]].mean(axis=1)
                        if persistent:
                            full[j] = u.x.array  # same DG1 layout as the history space
                    else:  # Cier, Rojas, Calo 2021: transport the conforming CG1 solution u_bar
                        transported[j] = r.cell_averages
                        if persistent:
                            ub = fem.Function(Vh)
                            ub.interpolate(op._sol.sub(1).collapse())
                            full[j] = ub.x.array
                    base_mass = w_tri @ (hist[j][hdofs].mean(axis=1) if persistent else prev_tri[j])
                    defect = abs(w_tri @ transported[j] - base_mass - dtc * (r.inflow - r.outflow))
                    stats["max_transport_balance_defect"] = max(stats["max_transport_balance_defect"],
                                                                defect / max(1.0, abs(base_mass)))
                if save_now:  # every refinement level of a saved step (for the ParaView sequence)
                    levels_store.append({"triangles": op.triangles.copy(), "transported": transported.copy(),
                                         "indicator": np.sum([r.cell_indicators / max(r.cell_indicators.sum(),
                                                                                       1e-300) for r in results],
                                                             axis=0) / op.areas})
                if level == nsolves - 1:
                    break
                sizes = triangle_characteristic_sizes(op.triangles)
                inds = [r.cell_indicators for r in results]
                if marking == "reactive":  # one indicator: species weighted by their reactivity (last step)
                    wts = species_weights
                    inds = [sum(wts[j] * inds[j] / max(inds[j].sum(), 1e-300) for j in range(4))]
                if sing_radius > 0.0 and cap_before:  # capped cells leave the Dorfler budget entirely
                    keep = singularity_cap(op.triangles, sizes, sing_radius, sing_min_size)
                    inds = [np.where(keep, ind, 0.0) for ind in inds]
                marked = dorfler_union(inds, sizes, h_c, theta,
                                       eligibility_factor, epsilon_chem, tie_tolerance)
                if sing_radius > 0.0:
                    marked &= singularity_cap(op.triangles, sizes, sing_radius, sing_min_size)
                if not marked.any():
                    break
                # main's leak-free refinement (dolfinx 0.11 refine with the default partitioner leaves one
                # MPI communicator per call; same mesh with partitioner=None): methods/front2d/fastadap.py
                new_domain, _ = fast_refine_marked_triangles_with_parent(domain, marked)
                domain = new_domain
                fields = MeshFields(domain, q_src, regime, mat)
            stats["min_transported"] = min(stats["min_transported"], float(transported[[0, 2, 3]].min()))
            rect = plan.triangles_to_rectangles(transported)  # (4, ny, nx), overlap means (conservative)
            transported_chem = rect.reshape(4, -1).T.copy()
            if remap == "linear" and persistent:
                transported_chem = linear_remap(op, full_dg1(op, full), transported_chem, X, Y, h_c, w_chem,
                                                nx, ny, kb)
            raw_chem = transported_chem.copy()
            for c in POSITIVE:
                transported_chem[:, c], corr = nonnegative_means(transported_chem[:, c], w_chem)
                stats["max_positivity_correction"] = max(stats["max_positivity_correction"], corr)
            stats["seconds_transport"] += clock.perf_counter() - ts
            ntri = op.cells
            stats["triangles_final_mean"] += ntri
            stats["triangles_final_max"] = max(stats["triangles_final_max"], ntri)
            tc = clock.perf_counter()
            expected = transported_chem + resident
            reacted = chem.react(transported_chem)
            stats["seconds_chemistry"] += clock.perf_counter() - tc
            water = float(np.max(np.abs(reacted.water_mass_kg / chem.water_volume_l - 1)))
            rerr = float(np.max(np.abs(reacted.total - expected) / np.maximum(1.0, np.abs(expected))))
            serr = float(np.max(np.abs(reacted.sites - chem.site_total)))
            if water > 1e-6 or rerr > 1e-8 or serr > 1e-8:
                if checkpoint_dir is not None:  # diagnostic dump of the worst cells before stopping
                    cell_err = np.max(np.abs(reacted.total - expected) / np.maximum(1.0, np.abs(expected)), axis=1)
                    worst = np.argsort(cell_err)[-20:]
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    np.savez(Path(checkpoint_dir) / f"gate_failure_step{step + 1}.npz", cells=worst,
                             cell_err=cell_err[worst], transported=transported_chem[worst], raw=raw_chem[worst],
                             resident=resident[worst], reacted_total=reacted.total[worst],
                             reacted_primary=reacted.primary[worst], sites=reacted.sites[worst],
                             site_total=chem.site_total[worst], chem_mid=np.column_stack([X.ravel(), Y.ravel()])[worst],
                             materials=chem_mat[worst], time=time + dtc)
                raise RuntimeError(f"conservation gate failed at step {step + 1}: water {water:.3g}, "
                                   f"components {rerr:.3g}, sites {serr:.3g}")
            stats["max_reaction_defect"] = max(stats["max_reaction_defect"], rerr)
            stats["max_site_defect"] = max(stats["max_site_defect"], serr)
            stats["max_water_change"] = max(stats["max_water_change"], water)
            react_mass = np.abs(reacted.mobile - transported_chem).T @ w_chem
            species_weights = react_mass / react_mass.sum() if react_mass.sum() > 0 else np.full(4, 0.25)
            if persistent:  # DG1 history = full-scale solution + reaction increment of its square
                inc_rect = (reacted.mobile - raw_chem).T.reshape(4, ny, nx)
                inc = plan.rectangles_to_triangles(inc_rect)  # (4, cells), conservative overlap average
                grp = square_of(op.triangles)  # base squares: unions of triangles
                hist = full.copy()
                for j in range(4):
                    hist[j][hdofs] += inc[j][:, None]
                for j in POSITIVE:
                    means = hist[j][hdofs].mean(axis=1)
                    bad = np.unique(grp[means < 0.0])
                    if bad.size:  # scale those base squares instead (same inventory)
                        stats["scaled_lift_squares"] = stats.get("scaled_lift_squares", 0) + int(bad.size)
                        sel = np.isin(grp, bad)
                        target = np.bincount(grp[sel], weights=(w_tri * means)[sel], minlength=grp.max() + 1)
                        fm = full[j][hdofs].mean(axis=1)
                        have = np.bincount(grp[sel], weights=(w_tri * fm)[sel], minlength=grp.max() + 1)
                        ratio = np.divide(target, have, out=np.zeros_like(target), where=have > 0)
                        hist[j][hdofs[sel]] = full[j][hdofs[sel]] * np.maximum(ratio[grp[sel]], 0.0)[:, None]
                cur_domain, cur_fields = domain, fields
                op_last_space, op_last_tri = Vh, op.triangles.copy()
            op.destroy()
            if step % rebuild_every == 0:
                import gc
                gc.collect()
            time = float(time + dtc)
            step += 1
            if abs(time - next_event) < 1e-9:
                time = float(next_event)
            for st in list(save_times) + [end_time]:
                if abs(time - st) < 1e-10 and st not in saved:
                    saved[st] = {"primary": reacted.primary.copy(), "mobile": reacted.mobile.copy(),
                                 "fixed": reacted.fixed.copy(), "sites": reacted.sites.copy(),
                                 "triangles": op.triangles.copy(), "tri_transported": transported.copy(),
                                 "tri_indicator": np.sum([r.cell_indicators / max(r.cell_indicators.sum(), 1e-300)
                                                          for r in results], axis=0) / op.areas,
                                 "levels": np.array(levels_store, dtype=object)}
            if ckpt is not None and checkpoint_every and step % checkpoint_every == 0:
                ckpt.mkdir(parents=True, exist_ok=True)
                chem.dump_state(ckpt / "chem.dump.tmp")
                np.savez(ckpt / "state.tmp.npz", time=time, step=step, initial=initial,
                         stats=np.array(stats, dtype=object), saved=np.array(saved, dtype=object))
                (ckpt / "chem.dump.tmp").replace(ckpt / "chem.dump")
                (ckpt / "state.tmp.npz").replace(ckpt / "state.npz")
            if step % max(1, nsteps // 20) == 0 or time >= end_time - 1e-12:
                el = clock.perf_counter() - t0
                print(f"[progress] {label} step {step}/{nsteps}, t = {time:.4f}/{end_time:g}, triangles {ntri}, "
                      f"{el:.1f} s (transport {stats['seconds_transport']:.0f} s, chemistry "
                      f"{stats['seconds_chemistry']:.0f} s)", flush=True)
        stats["global_inventory_change"] = (w_chem @ reacted.total - initial).tolist()
        stats["steps"] = step
        stats["triangles_final_mean"] /= max(step, 1)
    stats["seconds"] = clock.perf_counter() - t0
    geometry = {"chem_mid": np.column_stack([X.ravel(), Y.ravel()]), "chem_area": np.full(chem_mat.size, h_c * h_c),
                "chem_materials": chem_mat}
    _ = dmesh
    return {"geometry": geometry, "saved": saved, "stats": stats}

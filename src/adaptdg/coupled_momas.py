"""MoMaS 2D uniform coupled run: conservative DG1 transport + chemistry on a triangular mesh.

Extracted from ASRTM-PHREEQC (asrtm-phreeqc-public @ a152b91, ``methods/momas2d/coupled.py``):
``run_coupled`` and ``square_groups``, numerics, names and operation order unchanged; ``POSITIVE`` as in
the source. Same loop and conservation gates as the 1D MoMaS driver (transport -> chemistry -> replace DG means;
components 1e-8, sites 1e-8, water 1e-6).

Chemistry cells: one per triangle (groups=None) or one per group of triangles
(e.g. the two triangles of a square). With groups:
- down: porosity-weighted mean of the triangle means (conservative);
- up: the reaction increment of the group is added to every triangle of the group
  (conservative). If that would make a triangle mean of a positive component negative,
  that group is scaled instead (c_i *= reacted / transported), also conservative.

Deviations from the source (outside numerics): ``ConservativeDG2D`` comes from
``adaptdg.transport_dg2d``; the chemistry enters through the keyword-only argument ``chemistry`` of
``run_coupled`` (protocol :class:`adaptdg.loop_momas.ChemistryMomas`) instead of
``EasyModel.read(.../spec.json)``, ``FastAnchorMomasPhreeqcRM`` and ``MomasEasyPhreeqcRM``; the ``sys.path`` hack and
``WS``/``REPO`` are gone. adaptdg imports no chemistry.
"""

from __future__ import annotations

from pathlib import Path
import time as clock

import numpy as np

from adaptdg.transport_dg2d import ConservativeDG2D

POSITIVE = (0, 2, 3)


def run_coupled(domain, facet_tags, inlet_tags, q, materials, *, regime, dt, end_time,
                groups=None, save_times=(), label="", penalty=20.0, chem_volume_l=None,
                anchor_mode="one_end", checkpoint_dir=None, checkpoint_every=0,
                history="dg1",
                concentration_scale=1e-6, zero_tolerance=1e-16, chemistry):
    """chem_volume_l: PHREEQC representative volume per chemistry cell. None -> 1000 * area
    (litres per metre of depth). A constant keeps PHREEQC's absolute amounts O(1) on fine
    meshes: chemistry is intensive, so the volume only sets the numerical scale (the site
    drift in concentration units grows like 1 / volume).
    anchor_mode: site re-anchoring ("one_end": one END for all cells, linear cost and
    bit-identical to "repository"; see tests/momas2d_anchor_tests.py).
    checkpoint_dir / checkpoint_every: every n steps write the DG state, counters, saved
    snapshots and the PhreeqcRM dump (atomic replace); if a checkpoint exists the run
    resumes from it."""
    from dolfinx import fem, mesh

    spec = chemistry.spec
    mat = spec["materials"]
    ncell = domain.topology.index_map(2).size_local
    V0 = fem.functionspace(domain, ("DG", 0))
    dof0 = V0.dofmap.list[:ncell, 0]

    def cellf(values):
        f = fem.Function(V0)
        f.x.array[dof0] = values
        return f

    phi = cellf(np.array([mat[m]["phi"] for m in materials]))
    aL = cellf(np.array([mat[m]["alpha_" + regime][0] for m in materials]))
    aT = cellf(np.array([mat[m]["alpha_" + regime][1] for m in materials]))
    op = ConservativeDG2D(domain, q=q, porosity=phi, alpha_L=aL, alpha_T=aT, facet_tags=facet_tags,
                          inlet_tags=tuple(inlet_tags), time_step=dt, components=4,
                          positive_components=POSITIVE, penalty=penalty)
    w_tri = op.weights
    if groups is None:
        groups = np.arange(ncell)
    _, groups = np.unique(groups, return_inverse=True)
    nchem = int(groups.max()) + 1
    w_chem = np.bincount(groups, weights=w_tri, minlength=nchem)
    area_chem = np.bincount(groups, weights=op.areas, minlength=nchem)
    first = np.full(nchem, -1)
    first[groups[::-1]] = np.arange(ncell)[::-1]
    chem_materials = materials[first]
    if np.any(chem_materials[groups] != materials):
        raise ValueError("a chemistry group mixes materials")
    tri_mid = mesh.compute_midpoints(domain, 2, np.arange(ncell, dtype=np.int32))[:, :2]
    chem_mid = np.column_stack([np.bincount(groups, weights=op.areas * tri_mid[:, j], minlength=nchem)
                                for j in (0, 1)]) / area_chem[:, None]

    def down(tri_means):
        if nchem == ncell:  # one triangle per group: identity, keep bits
            return tri_means[np.argsort(groups)]
        out = np.zeros((nchem, tri_means.shape[1]))
        for j in range(tri_means.shape[1]):
            out[:, j] = np.bincount(groups, weights=w_tri * tri_means[:, j], minlength=nchem)
        return out / w_chem[:, None]

    stats = {"max_transport_defect": 0.0, "max_reaction_defect": 0.0, "max_site_defect": 0.0,
             "max_water_change": 0.0, "max_positivity_correction": 0.0, "scaled_lift_groups": 0,
             "chemistry_cells": nchem, "transport_cells": ncell}
    saved = {}
    t0 = clock.perf_counter()
    volume = 1000.0 * area_chem if chem_volume_l is None else np.full(nchem, float(chem_volume_l))
    stats["chem_volume_l_range"] = [float(volume.min()), float(volume.max())]
    with chemistry.open_chemistry(chem_materials, volume,
                                  concentration_scale=concentration_scale, zero_tolerance=zero_tolerance,
                                  anchor_mode=anchor_mode) as chem:
        op.initialize(chem.snapshot.mobile[groups])
        initial = w_chem @ chem.snapshot.total
        inlet = chemistry.inlet_mobile(concentration_scale=concentration_scale)
        events = sorted(set(spec["time"]["snapshots"] + [spec["time"]["injection_ends"], end_time]
                            + [t for t in save_times if t <= end_time]))
        nsteps = int(np.ceil(end_time / dt - 1e-9))
        time, step = 0.0, 0
        inward, outward = np.zeros(4), np.zeros(4)
        ckpt = None if checkpoint_dir is None else Path(checkpoint_dir)
        if ckpt is not None and (ckpt / "state.npz").exists():
            z = np.load(ckpt / "state.npz", allow_pickle=True)
            op.state[:] = z["dg_state"]
            time, step = float(z["time"]), int(z["step"])
            inward, outward, initial = z["inward"], z["outward"], z["initial"]
            stats.update(z["stats"].item())
            saved = z["saved"].item()
            chem.restore_state(ckpt / "chem.dump")
            print(f"[progress] {label} resumed at step {step}, t = {time:g}", flush=True)
        while time < end_time - 1e-12:
            next_event = min(e for e in events if e > time + 1e-9)
            dtc = min(dt, next_event - time)
            op.set_time_step(dtc)
            resident = chem.snapshot.fixed.copy()
            values = inlet[int(time >= spec["time"]["injection_ends"])]
            tri_transported, diagnostics = op.step(values)
            for j, d in enumerate(diagnostics):
                inward[j] += d["inward_flux_integral"]
                outward[j] += d["outward_flux_integral"]
                stats["max_transport_defect"] = max(stats["max_transport_defect"], d["scaled_defect"])
                stats["max_positivity_correction"] = max(stats["max_positivity_correction"],
                                                         d["maximum_mean_positivity_correction"])
            transported = down(tri_transported)
            expected = transported + resident
            reacted = chem.react(transported)
            water = float(np.max(np.abs(reacted.water_mass_kg / chem.water_volume_l - 1)))
            rerr = float(np.max(np.abs(reacted.total - expected) / np.maximum(1.0, np.abs(expected))))
            serr = float(np.max(np.abs(reacted.sites - chem.site_total)))
            if water > 1e-6 or rerr > 1e-8 or serr > 1e-8:
                raise RuntimeError(f"conservation gate failed at step {step + 1}: water {water:.3g}, "
                                   f"components {rerr:.3g}, sites {serr:.3g}")
            stats["max_reaction_defect"] = max(stats["max_reaction_defect"], rerr)
            stats["max_site_defect"] = max(stats["max_site_defect"], serr)
            stats["max_water_change"] = max(stats["max_water_change"], water)
            if nchem == ncell:
                new = reacted.mobile[groups]
            else:
                new = tri_transported + (reacted.mobile - transported)[groups]
                for c in POSITIVE:
                    bad_groups = np.unique(groups[new[:, c] < 0])
                    if bad_groups.size:
                        stats["scaled_lift_groups"] += int(bad_groups.size)
                        ratio = np.divide(reacted.mobile[:, c], transported[:, c],
                                          out=np.zeros(nchem), where=transported[:, c] > 0)
                        sel = np.isin(groups, bad_groups)
                        new[sel, c] = np.where(transported[groups[sel], c] > 0,
                                               tri_transported[sel, c] * ratio[groups[sel]],
                                               reacted.mobile[groups[sel], c])
            if history == "dg0":  # diagnostic: drop the DG1 slopes every step (as ADAP's chemistry remap)
                op.initialize(new)
            else:
                op.replace_cell_means(new)
            time = float(time + dtc)
            step += 1
            if abs(time - next_event) < 1e-9:
                time = float(next_event)
            for ts in save_times:
                if abs(time - ts) < 1e-10:
                    saved[ts] = {"primary": reacted.primary.copy(), "mobile": reacted.mobile.copy(),
                                 "fixed": reacted.fixed.copy(), "sites": reacted.sites.copy(),
                                 "aqueous": reacted.aqueous.copy(), "surface": reacted.surface.copy(),
                                 "tri_mobile": op.cell_means().copy()}
            if ckpt is not None and checkpoint_every and step % checkpoint_every == 0:
                ckpt.mkdir(parents=True, exist_ok=True)
                chem.dump_state(ckpt / "chem.dump.tmp")
                np.savez(ckpt / "state.tmp.npz", dg_state=op.state, time=time, step=step, inward=inward,
                         outward=outward, initial=initial, stats=np.array(stats, dtype=object),
                         saved=np.array(saved, dtype=object))
                (ckpt / "chem.dump.tmp").replace(ckpt / "chem.dump")
                (ckpt / "state.tmp.npz").replace(ckpt / "state.npz")
            if step % max(1, nsteps // 20) == 0 or time >= end_time - 1e-12:
                print(f"[progress] {label} step {step}/{nsteps}, t = {time:.4f}/{end_time:g}, "
                      f"{clock.perf_counter() - t0:.1f} s", flush=True)
        final = w_chem @ reacted.total
        defect = final - initial - inward + outward
        stats["global_balance_scaled"] = (np.abs(defect) / np.maximum(1.0, np.abs(initial))).tolist()
        stats["steps"] = step
        if end_time not in saved:
            saved[end_time] = {"primary": reacted.primary.copy(), "mobile": reacted.mobile.copy(),
                               "fixed": reacted.fixed.copy(), "sites": reacted.sites.copy(),
                               "aqueous": reacted.aqueous.copy(), "surface": reacted.surface.copy(),
                               "tri_mobile": op.cell_means().copy()}
    stats["seconds"] = clock.perf_counter() - t0
    geometry = {"tri_mid": tri_mid, "tri_area": op.areas, "groups": groups, "chem_mid": chem_mid,
                "chem_area": area_chem, "chem_materials": chem_materials,
                "tri_vertices": domain.geometry.x[mesh.entities_to_geometry(
                    domain, 2, np.arange(ncell, dtype=np.int32))][:, :, :2]}
    return {"geometry": geometry, "saved": saved, "stats": stats}

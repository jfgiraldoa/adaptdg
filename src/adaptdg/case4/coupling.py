"""Shared kinetic/geometry controller for the two Case 4 transport backends.

Both backends transport every RM component on a fixed chemistry grid. The
geometry timing is explicit because the original DG and adaptive prototypes
used different splitting sequences. Their defaults are retained for regression.

Provenance: copied from ASRTM-PHREEQC, src/asrtm_phreeqc/porosity/coupling.py.
Deviations: ``run_uniform_case4`` (builds ``Case4Chemistry``) is omitted; ``update_geometry`` and
``advance_case4_step`` take a required keyword-only ``model`` supplying
what the source imported from ``.case4`` / ``.geometry`` (``Case4Model`` Protocol): ``VM_M3_MOL``,
``darcy_flux_m_s``, ``ReactorVolumeBasis``, ``evolve_porosity``. ``INITIAL_POROSITY`` and ``LENGTH_M``
come from ``adaptdg.case4.constants``. Everything else is unchanged.
"""

from dataclasses import dataclass
import gc
from typing import Any, Protocol

import numpy as np

from .constants import INITIAL_POROSITY, LENGTH_M


@dataclass
class Case4State:
    porosity: np.ndarray
    concentrations_mol_l: np.ndarray
    minerals_mol_per_reactor: dict[str, np.ndarray]

    @classmethod
    def initial(cls, chemistry):
        return cls(
            np.full(chemistry.cells, INITIAL_POROSITY),
            chemistry.concentrations.copy(),
            {name: values.copy() for name, values in chemistry.minerals_mol.items()},
        )


class Case4Model(Protocol):
    """Documentation only: the model hooks (a module or namespace) of the Case 4 coupling.

    ``VM_M3_MOL`` (mineral molar volumes, m3/mol), ``darcy_flux_m_s(porosity) -> float``,
    ``ReactorVolumeBasis`` and ``evolve_porosity`` (as in asrtm-dg ``geometry``).
    """

    VM_M3_MOL: Any
    darcy_flux_m_s: Any
    ReactorVolumeBasis: Any
    evolve_porosity: Any


class TransportBackend(Protocol):
    def step(self, state: Case4State, flux_m_s: float) -> tuple[np.ndarray, dict]: ...


class UniformDGTransport:
    def __init__(self, chemistry, time_step_s):
        from .dg_1d import PorosityDG1D

        self.chemistry = chemistry
        self.solver = PorosityDG1D(chemistry.cells, time_step_s)

    def step(self, state, flux_m_s):
        concentrations, balance = self.solver.step(
            state.porosity, state.concentrations_mol_l, flux_m_s, self.chemistry.inlet
        )
        named = {self.chemistry.components[int(index)]: value for index, value in balance.items()}
        return concentrations, {
            "balance_relative": {name: values["relative"] for name, values in named.items()},
            "last_substep_transport_balance": named,
        }


class AdaptivePorosityTransport:
    def __init__(self, chemistry, time_step_s, settings):
        self.chemistry, self.dt, self.settings = chemistry, time_step_s, settings
        self.edges = np.linspace(0.0, LENGTH_M, chemistry.cells + 1)

    def step(self, state, flux_m_s):
        from adaptdg.transport import solve_adaptive_transport_step

        result = solve_adaptive_transport_step(
            bulk_components_mol_m3=1000.0 * state.porosity[None, :] * state.concentrations_mol_l,
            component_names=self.chemistry.components,
            porosity_chemistry=state.porosity,
            chemistry_edges_m=self.edges,
            inlet_concentrations_mol_l=self.chemistry.inlet,
            time_step_s=self.dt,
            darcy_flux_m_s=flux_m_s,
            reference_porosity=INITIAL_POROSITY,
            settings=self.settings,
        )
        diagnostics = {
            "balance_relative": result.component_mass_balance_relative,
            "projection_correction": result.component_mass_balance_offset_mol_l,
            "minimum_nodal": result.component_minimum_nodal_concentration_mol_l,
            "cells": result.cell_edges_m.size - 1,
            "solves": len(result.refinement_history),
        }
        concentrations = result.concentrations_mol_l
        del result
        gc.collect()
        return concentrations, diagnostics


def update_geometry(state, concentrations, minerals_after, *, model):
    """Apply mineral volume changes once, preserving every aqueous inventory.

    The Case 4 representative bulk volume is a fixed litre per reactor. This
    rescaling is the prototype's declared volume convention; it does not supply
    a physical solvent mass balance or the omitted Darcy volume source.
    """
    if not np.all(np.isfinite(concentrations)):
        raise RuntimeError("Non-finite transported or reacted component concentrations")
    if any(np.any(~np.isfinite(values)) or np.any(values < 0.0)
           for values in minerals_after.values()):
        raise RuntimeError("PhreeqcRM produced invalid remaining mineral moles")
    basis = model.ReactorVolumeBasis(state.porosity, 1.0)
    updated = model.evolve_porosity(
        state.porosity,
        {name: basis.mineral_bulk_density(values)
         for name, values in state.minerals_mol_per_reactor.items()},
        {name: basis.mineral_bulk_density(values) for name, values in minerals_after.items()},
        molar_volumes_m3_mol=model.VM_M3_MOL,
    )
    concentrations = concentrations * state.porosity[None, :] / updated.porosity[None, :]
    return Case4State(updated.porosity, concentrations, minerals_after)


def advance_case4_step(chemistry, state, transport: TransportBackend, *, step,
                       time_step_s, strang_substeps=6, geometry_update="after_transport", model):
    """Advance one macro step with a single full-dt kinetic call at its midpoint.

    ``after_transport`` retains the uniform DG prototype; ``reaction_midpoint``
    retains the adaptive prototype. Darcy flux is frozen for the macro step in
    both. On failure the caller must restore a committed chemistry checkpoint;
    no in-memory step rejection/rollback is claimed here.
    """
    if (step < 1 or not np.isfinite(time_step_s) or time_step_s <= 0
            or strang_substeps < 2 or strang_substeps % 2):
        raise ValueError("Require positive step/dt and an even number of transport substeps")
    if geometry_update not in {"after_transport", "reaction_midpoint"}:
        raise ValueError("Unknown porosity geometry timing")
    q = model.darcy_flux_m_s(state.porosity)
    current = Case4State(state.porosity, state.concentrations_mol_l.copy(),
                         state.minerals_mol_per_reactor)
    balances = {name: 0.0 for name in chemistry.components}
    corrections, minima = {}, {}
    max_cells, solves = 0, 0
    last_details = {}
    for substep in range(strang_substeps):
        current.concentrations_mol_l, diagnostics = transport.step(current, q)
        if not np.all(np.isfinite(current.concentrations_mol_l)):
            raise RuntimeError("Transport produced non-finite component concentrations")
        for name, value in diagnostics["balance_relative"].items():
            balances[name] = max(balances[name], float(value))
        for name, value in diagnostics.get("projection_correction", {}).items():
            corrections[name] = max(corrections.get(name, 0.0), abs(float(value)))
        for name, value in diagnostics.get("minimum_nodal", {}).items():
            minima[name] = min(minima.get(name, float("inf")), float(value))
        max_cells = max(max_cells, int(diagnostics.get("cells", 0)))
        solves += int(diagnostics.get("solves", 0))
        last_details = diagnostics.get("last_substep_transport_balance", {})
        if substep + 1 == strang_substeps // 2:
            concentrations, minerals_after, reaction_balance = chemistry.react(
                current.concentrations_mol_l, current.porosity,
                current.minerals_mol_per_reactor,
                (step - 0.5) * time_step_s, time_step_s,
            )
            if geometry_update == "reaction_midpoint":
                current = update_geometry(current, concentrations, minerals_after, model=model)
            else:
                current.concentrations_mol_l = concentrations
    if geometry_update == "after_transport":
        current = update_geometry(current, current.concentrations_mol_l, minerals_after, model=model)
    record = {
        "step": step, "time_s": step * time_step_s, "darcy_flux_m_s": q,
        "geometry_update": geometry_update,
        "porosity_min": float(current.porosity.min()),
        "porosity_max": float(current.porosity.max()),
        "reaction_element_balance_max_relative": {
            name: float(np.max(values["relative"])) for name, values in reaction_balance.items()
        },
        "reaction_element_balance_max_absolute_mol_per_reactor": {
            name: float(np.max(values["absolute_mol_per_reactor"]))
            for name, values in reaction_balance.items()
        },
        "transport_component_balance_max_relative": balances,
    }
    for name in ("Calcite", "Gypsum"):
        vf = current.minerals_mol_per_reactor[name] * model.VM_M3_MOL[name] * 1000.0
        record[f"{name.lower()}_vf_min"] = float(vf.min())
        record[f"{name.lower()}_vf_max"] = float(vf.max())
    if last_details:
        record["last_substep_transport_balance"] = last_details
    if max_cells:
        record.update({
            "max_transport_cells": max_cells,
            "adaptive_solves_over_all_substeps": solves,
            "minimum_transport_nodal_concentration_by_component_mol_l": minima,
            "maximum_conservative_positivity_projection_by_component_mol_l": corrections,
        })
    return current, record

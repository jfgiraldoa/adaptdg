"""Fakes shared by the Case 4 tests (documented chemistry/model interfaces only)."""

from types import SimpleNamespace

import numpy as np

from adaptdg.case4 import constants
from adaptdg.case4.coupling import Case4State

COMPONENTS = ("C", "Ca", "S", "Na")
CELLS = 20
INLET = np.array([1.0, 0.5, 0.25, 2.0])


class FakeChemistry:
    """Documented interface only: cells, components, inlet, minerals, react (identity)."""

    def __init__(self, cells=CELLS):
        self.cells, self.components, self.inlet = cells, COMPONENTS, INLET.copy()
        self.component_index = {n: i for i, n in enumerate(COMPONENTS)}
        self.concentrations = np.zeros((len(COMPONENTS), cells))
        self.minerals_mol = {"Calcite": np.full(cells, 0.01), "Gypsum": np.zeros(cells)}

    def react(self, concentrations, porosity, minerals, time_s, dt_s):
        balance = {e: {"absolute_mol_per_reactor": np.zeros(self.cells),
                       "relative": np.zeros(self.cells)} for e in ("C", "Ca", "S")}
        return concentrations.copy(), {k: v.copy() for k, v in minerals.items()}, balance


class FakeBasis:
    def __init__(self, porosity, volume):
        self.volume = volume

    def mineral_bulk_density(self, moles):
        return 1000.0 * np.asarray(moles) / self.volume


def fake_evolve(porosity, before, after, *, molar_volumes_m3_mol):
    return SimpleNamespace(porosity=np.asarray(porosity, dtype=float).copy())


MODEL = SimpleNamespace(VM_M3_MOL={"Calcite": 36.9e-6, "Gypsum": 74.2e-6},
                        darcy_flux_m_s=lambda porosity: 1.0e-6,
                        ReactorVolumeBasis=FakeBasis, evolve_porosity=fake_evolve)


def _state(chemistry, value=None):
    c = np.zeros((4, chemistry.cells)) if value is None else np.tile(value[:, None], (1, chemistry.cells))
    return Case4State(np.full(chemistry.cells, constants.INITIAL_POROSITY), c, chemistry.minerals_mol)

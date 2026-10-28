"""Case 4 DG1 transport (dolfinx) with a fake chemistry; skipped without dolfinx."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

from adaptdg.case4.dg_history import HistoryDGTransport  # noqa: E402
from adaptdg.case4.dg_1d import PorosityDG1D  # noqa: E402
from adaptdg.case4.coupling import Case4State  # noqa: E402
from adaptdg.case4.fv_transport import FVUpwindTransport  # noqa: E402
from case4_fakes import INLET, FakeChemistry, _state  # noqa: E402


def test_dg1_inlet_state_is_stationary():
    chem = FakeChemistry(10)
    solver = PorosityDG1D(10, 3600.0)
    conc = np.tile(INLET[:, None], (1, 10))
    out, balance = solver.step(np.full(10, 0.35), conc, 1e-6, INLET)
    assert np.allclose(out, conc, rtol=1e-9)
    assert all(abs(v["relative"]) < 1e-9 for v in balance.values())
    assert chem.cells == 10


@pytest.mark.parametrize("history", ["dg1", "mean"])
def test_history_dg_vs_fv_first_order_agreement(history):
    chem = FakeChemistry(40)
    dg = HistoryDGTransport(chem, 3600.0, history=history, positivity="local")
    fv = FVUpwindTransport(chem, 3600.0)
    s_dg, s_fv = _state(chem), _state(chem)
    for _ in range(4):
        s_dg = Case4State(s_dg.porosity, dg.step(s_dg, 1e-5)[0], s_dg.minerals_mol_per_reactor)
        s_fv = Case4State(s_fv.porosity, fv.step(s_fv, 1e-5)[0], s_fv.minerals_mol_per_reactor)
    assert np.all(s_dg.concentrations_mol_l >= 0.0)
    # Both transport the same inventory from the inlet (total mass within a few percent).
    assert s_dg.concentrations_mol_l.sum() == pytest.approx(s_fv.concentrations_mol_l.sum(), rel=0.1)

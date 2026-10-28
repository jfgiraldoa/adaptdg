"""Case 4 transport modules with a fake chemistry (no PHREEQC, no benchmark data)."""

import subprocess
import sys
import numpy as np
import pytest

from adaptdg import marking
from adaptdg.case4 import adap_asfem, constants, positivity
from adaptdg.case4.adap_fast import FastPersistentAdaptiveTransport, UniformFineTransport
from adaptdg.case4.adap_persistent import PersistentSettings
from adaptdg.case4.coupling import advance_case4_step
from adaptdg.case4.fv_transport import FVUpwindTransport, fv_upwind_step
from adaptdg.case4.variable_dt import next_dt
from case4_fakes import CELLS, INLET, MODEL, FakeChemistry, _state

def test_constants_and_marking_reuse():
    assert constants.LENGTH_M == 2.0 and constants.INITIAL_POROSITY == 0.35
    assert adap_asfem.dorfler_with_ties is marking.dorfler_with_ties


def test_positivity_conserves_inventory():
    means = np.array([1.0, -0.2, 0.5, 0.0, 2.0])
    weights = np.array([0.3, 0.2, 0.3, 0.2, 0.4])
    fixed, _ = positivity.local_mean_redistribution(means, weights)
    assert np.all(fixed >= 0.0)
    assert np.dot(weights, fixed) == pytest.approx(np.dot(weights, means), rel=1e-12)


def test_fv_upwind_uniform_inlet_state_is_stationary():
    n, dx = 10, 0.2
    phi = np.full(n, 0.35)
    conc = np.tile(INLET[:, None], (1, n))
    new = fv_upwind_step(phi, phi, conc, 1e-6, INLET, 3600.0, dx)
    assert np.allclose(new, conc, rtol=1e-12, atol=0.0)


def test_fv_backend_and_coupling_step():
    chem = FakeChemistry()
    backend = FVUpwindTransport(chem, 21600.0 / 6)
    state, record = advance_case4_step(chem, _state(chem), backend, step=1, time_step_s=21600.0,
                                       model=MODEL)
    assert state.concentrations_mol_l.shape == (4, CELLS)
    assert np.all(state.concentrations_mol_l >= 0.0)
    assert state.concentrations_mol_l[:, 0].min() > 0.0          # inlet reached the first cell
    assert len(backend.substep_log) == 6
    assert record["darcy_flux_m_s"] == 1.0e-6 and record["geometry_update"] == "after_transport"
    with pytest.raises(ValueError):
        advance_case4_step(chem, _state(chem), backend, step=1, time_step_s=21600.0,
                           strang_substeps=3, model=MODEL)


def test_next_dt_courant_and_target():
    dt, nominal, landed = next_dt("fixed", 0.0, 100.0, 1.0, 1.0, 100.0, 1000.0, 1.5, [])
    assert (dt, landed) == (100.0, None)
    dt, nominal, landed = next_dt("fixed", 0.0, 100.0, 1.0, 1.0, 100.0, 1000.0, 1.5, [100.0])
    assert landed == 100.0 and dt == 100.0


def test_adap_fast_refines_front_and_conserves_inlet_state():
    chem = FakeChemistry()
    settings = PersistentSettings(max_level=2, theta=0.5, max_solves=2, coarsen_tolerance=1e-3)
    backend = FastPersistentAdaptiveTransport(chem, 3600.0, settings)
    state, record = advance_case4_step(chem, _state(chem), backend, step=1, time_step_s=21600.0,
                                       geometry_update="reaction_midpoint", model=MODEL)
    assert record["max_transport_cells"] > CELLS                  # refinement near the front
    assert np.all(state.concentrations_mol_l >= 0.0)
    # An inlet-composition state is a stationary solution of the transport step.
    full = _state(chem, INLET)
    out, _ = FastPersistentAdaptiveTransport(chem, 3600.0, settings).step(full, 1e-6)
    assert np.allclose(out, full.concentrations_mol_l, rtol=1e-9, atol=1e-12)


def test_uniform_fine_keeps_fixed_mesh():
    chem = FakeChemistry(10)
    backend = UniformFineTransport(chem, 3600.0, 2)
    backend.step(_state(chem), 1e-6)
    assert backend.mesh.edges().size - 1 == 10 * 4


def test_import_is_lazy():
    code = ("import sys, adaptdg.case4, adaptdg.case4.constants\n"
            "import adaptdg.case4.adap_fast, adaptdg.case4.adap_asfem, adaptdg.case4.transport_bdf2\n"
            "import adaptdg.case4.dg_history, adaptdg.case4.variable_dt, adaptdg.case4.coarse_transport\n"
            "import adaptdg.case4.fv_transport, adaptdg.case4.dg_1d, adaptdg.case4.coupling\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in\n"
            "       ('dolfinx', 'ufl', 'basix', 'petsc4py', 'mpi4py', 'phreeqcrm', 'asrtm_dg')]\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True)

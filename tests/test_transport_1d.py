"""Tests for the 1D DG1/CG1 transport extracted from ASRTM-PHREEQC (piece 5a).

The numpy-only tests are adapted from the source ``tests/test_porosity_transport_1d.py``.
The dolfinx tests are new smoke/conservation checks (the source has none for
``ConservativeDG1D``); equality with the source was shown separately.
"""

import subprocess
import sys
import unittest

import numpy as np
import pytest

from adaptdg.transport import (
    PorosityTransportSettings,
    conservative_positive_projection_1d,
    dorfler_mark_1d,
    gradient_jump_indicator_1d,
    nonnegative_means,
    project_linear_product_to_cells,
    project_piecewise_constant,
)


class PorosityTransferTests(unittest.TestCase):
    def test_piecewise_constant_projection_preserves_integral(self):
        source_edges = np.array([0.0, 0.5, 1.0])
        source = np.array([1.0, 3.0])
        target_edges = np.array([0.0, 0.25, 0.75, 1.0])
        projected = project_piecewise_constant(source_edges, source, target_edges)
        np.testing.assert_allclose(projected, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(np.dot(projected, np.diff(target_edges))), 2.0)

    def test_cg1_times_dg0_projection_preserves_bulk_inventory(self):
        nodes = np.array([0.0, 0.25, 0.75, 1.0])
        concentration = 1.0 + nodes
        coefficient_edges = np.array([0.0, 0.5, 1.0])
        porosity = np.array([0.2, 0.4])
        target_edges = np.array([0.0, 0.3, 0.7, 1.0])
        projected = project_linear_product_to_cells(
            nodes, concentration, coefficient_edges, porosity, target_edges
        )
        self.assertAlmostEqual(float(np.dot(projected, np.diff(target_edges))), 0.475)

    def test_dorfler_marks_only_cells_above_chemistry_resolution(self):
        marked = dorfler_mark_1d(
            np.array([10.0, 4.0, 2.0]),
            np.array([0.5, 0.25, 0.125]),
            theta=0.5,
            minimum_width=0.125,
        )
        np.testing.assert_array_equal(marked, [True, False, False])

    def test_gradient_indicator_detects_a_front(self):
        edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        values = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        indicator = gradient_jump_indicator_1d(edges, values)
        self.assertGreater(indicator[2], 0.0)
        self.assertGreater(indicator[3], 0.0)
        self.assertEqual(indicator[0], 0.0)

    def test_settings_reject_invalid_marking(self):
        with self.assertRaises(ValueError):
            PorosityTransportSettings(theta=0.0)

    def test_positive_projection_preserves_boundary_flux_inventory(self):
        edges = np.array([0.0, 0.5, 1.0])
        nodes = edges.copy()
        phi = np.array([0.3, 0.4])
        previous = np.array([0.2, 0.1])
        raw = np.array([0.15, -0.08, 0.12])
        dt, q, inlet = 10.0, 0.01, 0.2
        corrected, magnitude = conservative_positive_projection_1d(
            nodes, raw, edges, phi, previous,
            time_step_s=dt, darcy_flux_m_s=q, inlet_concentration=inlet,
        )
        self.assertGreaterEqual(float(corrected.min()), 0.0)
        weights = np.array([phi[0] * 0.25, phi[0] * 0.25 + phi[1] * 0.25, phi[1] * 0.25])
        target = float(np.dot(phi * np.diff(edges), previous) + dt * q * inlet)
        self.assertAlmostEqual(float(np.dot(weights, corrected) + dt * q * corrected[-1]), target)
        self.assertGreater(magnitude, 0.0)


def test_nonnegative_means_keeps_weighted_inventory():
    values = np.array([1.0, -0.2, 0.5, 0.0])
    weights = np.array([0.5, 1.0, 2.0, 1.0])
    repaired, correction = nonnegative_means(values, weights)
    assert repaired.min() >= 0.0
    assert correction > 0.0
    assert float(weights @ repaired) == pytest.approx(float(weights @ values), rel=1e-14)
    same, zero = nonnegative_means(np.abs(values), weights)
    assert zero == 0.0 and np.array_equal(same, np.abs(values))
    with pytest.raises(ValueError):
        nonnegative_means(np.array([-1.0, 0.1]), np.ones(2))


def test_import_adaptdg_does_not_load_dolfinx():
    code = (
        "import sys, adaptdg, adaptdg.transport; "
        "assert not any(m.split('.')[0] in ('dolfinx', 'ufl', 'basix', 'petsc4py', 'mpi4py') "
        "for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _interval(n):
    pytest.importorskip("dolfinx")
    from dolfinx import mesh
    from mpi4py import MPI

    return mesh.create_interval(MPI.COMM_SELF, n, np.array([0.0, 1.0]))


def test_conservative_dg1_inventory_balance():
    domain = _interval(20)
    from adaptdg.transport import ConservativeDG1D

    cells = 20
    solver = ConservativeDG1D(
        domain, porosity=np.full(cells, 0.4), dispersion=np.full(cells, 1e-2),
        darcy_flux=0.5, time_step=0.02, components=2, positive_components=(0, 1),
    )
    solver.initialize(np.zeros((cells, 2)))
    for _ in range(3):
        means, diag = solver.step(np.array([1.0, 2.0]))
    assert means.shape == (cells, 2)
    assert means.min() >= -1e-12
    assert all(d["scaled_defect"] < 1e-10 for d in diag)
    assert solver.problem.matrix_assembly_count == 1
    # Changing the time step reassembles the operator exactly once.
    solver.step(np.array([1.0, 2.0]), time_step=0.03)
    assert solver.matrix_assemblies == 2


def test_adaptive_transport_step_runs_and_balances():
    pytest.importorskip("dolfinx")
    from adaptdg.transport import solve_adaptive_transport_step

    n = 16
    edges = np.linspace(0.0, 1.0, n + 1)
    phi = np.full(n, 0.3)
    bulk = np.zeros((1, n))
    bulk[0, :4] = 1000.0 * 0.3 * 0.5
    settings = PorosityTransportSettings(initial_cells=5, maximum_solves=3)
    result = solve_adaptive_transport_step(
        bulk_components_mol_m3=bulk, component_names=("C",), porosity_chemistry=phi,
        chemistry_edges_m=edges, inlet_concentrations_mol_l=np.array([0.5]),
        time_step_s=50.0, darcy_flux_m_s=0.003, reference_porosity=0.3, settings=settings,
    )
    assert result.concentrations_mol_l.shape == (1, n)
    assert result.minimum_concentration_mol_l >= -1e-12
    assert result.component_mass_balance_relative["C"] < 1e-8
    assert len(result.refinement_history) >= 1

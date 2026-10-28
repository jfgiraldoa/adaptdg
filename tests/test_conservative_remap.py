import unittest

import numpy as np

from adaptdg.transfer import (
    ConservativeRemapCache1D,
    ConservativeRemapPlan1D,
    cell_averages_to_p1_nodes,
    conservative_remap_1d,
)


def integrate_cell_averages(edges, values):
    """Integrate cell-average fields (local copy of the ASRTM-PHREEQC helper)."""
    return np.sum(np.asarray(values) * np.diff(np.asarray(edges)), axis=-1)


class ConservativeRemapTests(unittest.TestCase):
    def test_constant_field_is_exact(self):
        source_edges = np.array([0.0, 0.2, 0.5, 1.0])
        target_edges = np.linspace(0.0, 1.0, 9)

        remapped = conservative_remap_1d(source_edges, np.full(3, 2.5), target_edges)

        np.testing.assert_allclose(remapped, 2.5, rtol=0.0, atol=1.0e-14)

    def test_integral_is_preserved_for_multiple_components(self):
        source_edges = np.array([0.0, 0.1, 0.4, 1.0])
        target_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        source = np.array([[1.0, 3.0, 2.0], [0.5, 0.0, 4.0]])

        target = conservative_remap_1d(source_edges, source, target_edges)

        np.testing.assert_allclose(
            integrate_cell_averages(target_edges, target),
            integrate_cell_averages(source_edges, source),
            rtol=1.0e-14,
            atol=1.0e-14,
        )

    def test_domain_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same domain"):
            conservative_remap_1d([0.0, 1.0], [1.0], [0.0, 0.5])

    def test_uniform_cell_to_p1_reconstruction_preserves_integral(self):
        cell_values = np.array([1.0, 3.0, 2.0, 5.0])

        nodes = cell_averages_to_p1_nodes(cell_values)

        p1_integral = np.sum(0.5 * (nodes[:-1] + nodes[1:]))
        self.assertAlmostEqual(p1_integral, float(np.sum(cell_values)), places=14)

    def test_sparse_plan_matches_direct_multifield_remap(self):
        source_edges = np.array([0.0, 0.2, 0.55, 1.0])
        target_edges = np.array([0.0, 0.1, 0.4, 0.8, 1.0])
        values = np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 0.0]])
        expected = conservative_remap_1d(source_edges, values, target_edges)

        plan = ConservativeRemapPlan1D.create(source_edges, target_edges)

        np.testing.assert_allclose(
            plan.apply(values), expected, rtol=0.0, atol=1.0e-15
        )

    def test_remap_cache_reuses_geometry_and_preserves_integral(self):
        source_edges = np.linspace(0.0, 1.0, 8)
        target_edges = np.array([0.0, 0.15, 0.4, 0.7, 1.0])
        values = np.linspace(1.0, 2.0, source_edges.size - 1)
        cache = ConservativeRemapCache1D(maximum_plans=2)

        first = cache.remap(source_edges, values, target_edges)
        second = cache.remap(source_edges, values, target_edges)

        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        source_integral = np.dot(values, np.diff(source_edges))
        target_integral = np.dot(first, np.diff(target_edges))
        self.assertAlmostEqual(target_integral, source_integral, places=14)


if __name__ == "__main__":
    unittest.main()

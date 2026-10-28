import unittest

import numpy as np
import pytest

from adaptdg.transfer_2d import (
    clear_transfer_plan_cache,
    conservative_remap_plan_2d,
    refine_conservative_remap_plan_2d,
    rectangles_to_triangles,
    rectangles_to_triangles_partial,
    transfer_plan_cache_info,
    triangle_areas,
    triangles_to_rectangles,
)


class ConservativeTransfer2DTests(unittest.TestCase):
    def setUp(self):
        self.x_edges = np.asarray([0.0, 0.5, 1.0])
        self.y_edges = np.asarray([0.0, 0.5, 1.0])
        self.triangles = np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ]
        )

    def test_constant_field_is_exact_in_both_directions(self):
        rectangles = np.full((2, 2), 7.25)
        on_triangles = rectangles_to_triangles(
            self.x_edges, self.y_edges, rectangles, self.triangles
        )
        np.testing.assert_allclose(on_triangles, 7.25, atol=1.0e-14)

        back = triangles_to_rectangles(
            self.triangles, on_triangles, self.x_edges, self.y_edges
        )
        np.testing.assert_allclose(back, rectangles, atol=1.0e-14)

    def test_triangle_to_rectangle_transfer_conserves_integral(self):
        triangle_values = np.asarray([1.0, 3.0])
        rectangles = triangles_to_rectangles(
            self.triangles,
            triangle_values,
            self.x_edges,
            self.y_edges,
        )
        source_integral = np.dot(triangle_values, triangle_areas(self.triangles))
        rectangle_areas = np.outer(np.diff(self.y_edges), np.diff(self.x_edges))
        target_integral = np.sum(rectangles * rectangle_areas)
        self.assertAlmostEqual(
            float(source_integral), float(target_integral), places=13
        )

    def test_partial_rectangle_to_triangle_transfer_needs_no_global_cover(self):
        subset = self.triangles[:1]
        rectangle_values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        transferred = rectangles_to_triangles_partial(
            self.x_edges,
            self.y_edges,
            rectangle_values,
            subset,
        )
        self.assertEqual(transferred.shape, (1,))
        self.assertTrue(np.isfinite(transferred[0]))

    def test_multiple_fields_preserve_leading_axis(self):
        rectangles = np.stack(
            [np.ones((2, 2)), np.arange(4, dtype=float).reshape((2, 2))]
        )
        transferred = rectangles_to_triangles(
            self.x_edges,
            self.y_edges,
            rectangles,
            self.triangles,
        )
        self.assertEqual(transferred.shape, (2, 2))
        np.testing.assert_allclose(transferred[0], 1.0)

    def test_incomplete_domain_is_rejected(self):
        outside = self.triangles.copy()
        outside[0, 1, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "does not fully cover"):
            rectangles_to_triangles(
                self.x_edges,
                self.y_edges,
                np.ones((2, 2)),
                outside,
            )

    def test_one_cached_plan_serves_both_transfer_directions(self):
        rectangle_values = np.asarray([[2.0, 5.0], [3.0, 7.0]])
        clear_transfer_plan_cache()
        plan = conservative_remap_plan_2d(
            self.triangles, self.x_edges, self.y_edges
        )
        on_triangles = plan.rectangles_to_triangles(rectangle_values)
        back = plan.triangles_to_rectangles(on_triangles)
        source_integral = np.sum(rectangle_values) * 0.25
        target_integral = np.sum(
            on_triangles * triangle_areas(self.triangles)
        )
        self.assertAlmostEqual(float(source_integral), float(target_integral))
        self.assertEqual(back.shape, rectangle_values.shape)
        self.assertEqual(transfer_plan_cache_info()["misses"], 1)

        rectangles_to_triangles(
            self.x_edges, self.y_edges, rectangle_values, self.triangles
        )
        self.assertEqual(transfer_plan_cache_info()["hits"], 1)

    def test_incremental_refined_plan_matches_complete_rebuild(self):
        child_triangles = np.asarray(
            [
                [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
                [[0.5, 0.0], [1.0, 0.0], [1.0, 0.5]],
                [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]],
                [[0.5, 0.0], [1.0, 0.5], [0.5, 0.5]],
                [[0.0, 0.0], [0.5, 0.5], [0.0, 0.5]],
                [[0.5, 0.5], [1.0, 1.0], [0.5, 1.0]],
                [[0.0, 0.5], [0.5, 1.0], [0.0, 1.0]],
                [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0]],
            ]
        )
        parent_cell = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        rectangle_values = np.asarray([[2.0, 5.0], [3.0, 7.0]])
        triangle_values = np.linspace(0.2, 1.6, child_triangles.shape[0])

        clear_transfer_plan_cache()
        parent_plan = conservative_remap_plan_2d(
            self.triangles, self.x_edges, self.y_edges
        )
        incremental = refine_conservative_remap_plan_2d(
            parent_plan,
            parent_cell,
            child_triangles,
            self.x_edges,
            self.y_edges,
        )
        clear_transfer_plan_cache()
        rebuilt = conservative_remap_plan_2d(
            child_triangles, self.x_edges, self.y_edges
        )

        np.testing.assert_allclose(
            incremental.rectangles_to_triangles(rectangle_values),
            rebuilt.rectangles_to_triangles(rectangle_values),
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            incremental.triangles_to_rectangles(triangle_values),
            rebuilt.triangles_to_rectangles(triangle_values),
            atol=1.0e-14,
        )

    def test_refined_parent_map_matches_final_local_cell_order(self):
        pytest.importorskip("dolfinx")
        from adaptdg.transport_2d import (
            create_rectangle_mesh_2d,
            mesh_triangles,
            refine_marked_triangles_with_parent,
        )

        domain = create_rectangle_mesh_2d(
            lower=(0.0, 0.0), upper=(0.5, 0.5), cells=(5, 5)
        )
        parent_triangles = mesh_triangles(domain)
        marked = np.zeros(parent_triangles.shape[0], dtype=bool)
        marked[[0, 3, 7]] = True
        refined, parent_cell = refine_marked_triangles_with_parent(
            domain, marked
        )
        child_triangles = mesh_triangles(refined)

        parent_values = np.linspace(0.1, 2.0, parent_triangles.shape[0])
        parent_inventory = parent_values @ triangle_areas(parent_triangles)
        child_inventory = (
            parent_values[parent_cell] @ triangle_areas(child_triangles)
        )
        self.assertAlmostEqual(
            float(parent_inventory), float(child_inventory), places=14
        )


if __name__ == "__main__":
    unittest.main()

import numpy as np
import pytest

from adaptdg.marking import (
    dorfler_mark_1d,
    dorfler_union,
    dorfler_union_2d,
    dorfler_union_momas,
    dorfler_with_ties,
)


def test_dorfler_marks_only_cells_above_chemistry_resolution():
    marked = dorfler_mark_1d(
        np.array([10.0, 4.0, 2.0]),
        np.array([0.5, 0.25, 0.125]),
        theta=0.5,
        minimum_width=0.125,
    )
    np.testing.assert_array_equal(marked, [True, False, False])


def test_mark_1d_validation():
    with pytest.raises(ValueError):
        dorfler_mark_1d(np.array([1.0]), np.array([1.0]), theta=0.0, minimum_width=0.1)
    with pytest.raises(ValueError):
        dorfler_mark_1d(np.array([-1.0]), np.array([1.0]), theta=0.5, minimum_width=0.1)


def test_core_bulk_and_ties():
    ind = {"a": np.array([4.0, 3.0, 2.9, 0.1])}
    sizes = np.ones(4)
    # theta**2 * total = 0.5 * 10 = 5 -> cells 0 and 1; no ties
    m = dorfler_union(ind, sizes, 0.5, np.sqrt(0.5), 0.0, None)
    np.testing.assert_array_equal(m, [True, True, False, False])
    # tie_tol=0.1: 3.0 - 1.1*2.9 < 0 -> cell 2 joins, cell 3 does not
    m = dorfler_union(ind, sizes, 0.5, np.sqrt(0.5), 0.0, 0.1)
    np.testing.assert_array_equal(m, [True, True, True, False])


def test_core_eligibility_is_strict_and_array_min_size():
    ind = {"a": np.array([5.0, 4.0, 3.0])}
    sizes = np.array([1.0, 1.0, 1.0])
    assert not dorfler_union(ind, sizes, 1.0, 1.0, 0.0, None).any()
    m = dorfler_union(ind, sizes, np.array([1.0, 0.5, 1.0]), 1.0, 0.0, None)
    np.testing.assert_array_equal(m, [False, True, False])


def test_core_epsilon_skips_species_and_union():
    ind = {"a": np.array([9.0, 1.0]), "b": np.array([1.0, 9.0])}
    sizes = np.array([2.0, 1.0])
    m = dorfler_union(ind, sizes, 1.5, 0.5, 0.0, None)
    np.testing.assert_array_equal(m, [True, False])  # b has eligible share 0.1
    m = dorfler_union(ind, sizes, 1.5, 0.5, 0.05, None)
    np.testing.assert_array_equal(m, [True, False])  # b marks its only eligible cell
    m = dorfler_union(ind, sizes, 1.5, 0.5, 0.5, None)
    np.testing.assert_array_equal(m, [True, False])


def test_core_explicit_eligibility_and_post_filter():
    ind = {"a": np.array([5.0, 4.0, 3.0])}
    el = np.array([False, True, True])
    m = dorfler_union(ind, None, 0.0, 1.0, 0.0, None, eligibility=el)
    np.testing.assert_array_equal(m, [False, True, True])
    seen = []

    def keep_first(marked):
        seen.append(marked.copy())
        out = marked.copy()
        out[0] = True
        return out

    m = dorfler_union(ind, None, 0.0, 0.1, 0.0, None, eligibility=el, post_mark_filter=keep_first)
    np.testing.assert_array_equal(seen[0], [False, True, False])
    np.testing.assert_array_equal(m, [True, True, False])
    with pytest.raises(ValueError):
        dorfler_union(ind, None, 0.0, 1.0, 0.0, None, eligibility=el,
                      post_mark_filter=lambda m: m[:1])


def test_core_empty_and_zero_indicators():
    ind = {"a": np.zeros(3)}
    assert not dorfler_union(ind, np.ones(3), 0.5, 0.5, 0.0, 0.1).any()
    ind = {"a": np.array([1.0, 2.0])}
    assert not dorfler_union(ind, np.ones(2), 5.0, 0.5, 0.0, 0.1).any()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dorfler_union({"a": np.ones(3)}, np.ones(2), 0.5, 0.5, 0.0, None)


def test_adapters_match_core():
    ind = {"x": np.array([4.0, 3.0, 2.9, 0.1]), "y": np.array([0.1, 0.2, 5.0, 1.0])}
    sizes = np.array([1.0, 0.2, 1.0, 1.0])
    core = dorfler_union(ind, sizes, 0.5, 0.6, 0.1, 0.2)
    a2 = dorfler_union_2d(ind, sizes, chemistry_width=1.0, eligibility_factor=0.5,
                          theta=0.6, epsilon_chem=0.1, tie_tolerance=0.2)
    am = dorfler_union_momas(list(ind.values()), sizes, 1.0, 0.6, 0.5, 0.1, 0.2)
    np.testing.assert_array_equal(core, a2)
    np.testing.assert_array_equal(core, am)


def test_with_ties_verbatim_behaviour():
    v = np.array([4.0, 3.0, 2.9, 0.1])
    el = np.ones(4, dtype=bool)
    m = dorfler_with_ties(v, fraction=0.5, tie=0.2, eligible=el)
    # cumulative [4, 7, ...] >= 5 -> stop=1; threshold 0.64*3 = 1.92 -> cell 2 joins
    np.testing.assert_array_equal(m, [True, True, True, False])
    assert not dorfler_with_ties(v, fraction=0.5, tie=0.2, eligible=np.zeros(4, bool)).any()

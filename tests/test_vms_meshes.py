"""VMS meshes and refinement (2D and 3D)."""

import numpy as np
import pytest

pytest.importorskip("dolfinx")

import ufl  # noqa: E402
from dolfinx import fem  # noqa: E402

from adaptdg.vms import meshes  # noqa: E402
from adaptdg.vms.refine import refine_marked, refine_uniform  # noqa: E402


def _measure(domain):
    return fem.assemble_scalar(fem.form(1.0 * ufl.Measure("dx", domain=domain)))


def _ncells(domain):
    d = domain.topology.dim
    return domain.topology.index_map(d).size_local


def test_l_shape_cells_area_and_tags():
    dom = meshes.l_shape(4)
    assert _ncells(dom) == 48  # 4*4 squares x 4 triangles, minus one quadrant
    assert _measure(dom) == pytest.approx(3.0)
    tags = meshes.tag_boundary(dom, {1: lambda x: np.full(x.shape[1], True)})
    ds = ufl.Measure("ds", domain=dom, subdomain_data=tags)
    assert fem.assemble_scalar(fem.form(1.0 * ds(1))) == pytest.approx(8.0)
    assert len(tags.indices) == 16  # 8 boundary edges of length 1, each split in two
    with pytest.raises(ValueError):
        meshes.l_shape(3)


def test_fichera_volume_and_surface():
    dom = meshes.fichera(2)
    assert dom.topology.dim == 3
    assert _ncells(dom) == 7 * 6  # 7 of 8 cubes, 6 tetrahedra each
    assert _measure(dom) == pytest.approx(7.0)
    tags = meshes.tag_boundary(dom, {1: lambda x: np.full(x.shape[1], True)})
    ds = ufl.Measure("ds", domain=dom, subdomain_data=tags)
    assert fem.assemble_scalar(fem.form(1.0 * ds(1))) == pytest.approx(24.0)


def test_refine_marked_2d_and_parent_map():
    dom = meshes.l_shape(4)
    n0 = _ncells(dom)
    marked = np.zeros(n0, bool)
    marked[:5] = True
    new, parent = refine_marked(dom, marked)
    assert _ncells(new) > n0 and parent.shape == (_ncells(new),)
    assert set(parent) == set(range(n0))  # every old cell has at least one child
    assert np.bincount(parent)[:5].min() >= 2  # marked cells are split
    assert _measure(new) == pytest.approx(3.0)
    same, ident = refine_marked(dom, np.zeros(n0, bool))
    assert same is dom and np.array_equal(ident, np.arange(n0))


def test_refine_marked_and_uniform_3d():
    dom = meshes.fichera(2)
    n0 = _ncells(dom)
    marked = np.zeros(n0, bool)
    marked[:6] = True
    new, parent = refine_marked(dom, marked)
    assert _ncells(new) > n0 and parent.shape == (_ncells(new),)
    assert _measure(new) == pytest.approx(7.0)
    uni = refine_uniform(dom)
    assert _ncells(uni) == 8 * n0
    assert _measure(uni) == pytest.approx(7.0)

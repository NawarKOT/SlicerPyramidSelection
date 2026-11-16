import numpy as np
import pytest

from PyramidSelectionLib import geometry


@pytest.mark.parametrize("normal", [
    [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [1, 2, 3], [0.01, 0, 1],
])
def test_tangent_frame_is_orthonormal_right_handed(normal):
    n = geometry.normalize(normal)
    u, v = geometry.tangent_frame(n)
    assert np.isclose(np.linalg.norm(u), 1)
    assert np.isclose(np.linalg.norm(v), 1)
    assert np.isclose(np.dot(u, v), 0)
    assert np.isclose(np.dot(u, n), 0)
    assert np.allclose(np.cross(u, v), n)


def test_tangent_rectangle_size_center_and_orientation():
    center = np.array([10.0, -5.0, 3.0])
    normal = geometry.normalize([1, 1, 0])
    corners = geometry.tangent_rectangle(center, normal, 40, 20)
    assert np.allclose(corners.mean(axis=0), center)
    width, height = geometry.rectangle_size(corners)
    assert np.isclose(width, 40) and np.isclose(height, 20)
    _, _, _, n, _, _ = geometry.rectangle_frame(corners)
    assert np.allclose(n, normal)  # counter-clockwise around the outward normal


def test_edge_midpoints():
    corners = geometry.tangent_rectangle([0, 0, 0], [0, 0, 1], 4, 2)
    mids = geometry.edge_midpoints(corners)
    assert np.allclose(mids, (corners + np.roll(corners, -1, axis=0)) / 2)


@pytest.mark.parametrize("edge", [0, 1, 2, 3])
def test_scale_edge_moves_edge_to_handle_and_keeps_center(edge):
    corners = geometry.tangent_rectangle([0, 0, 0], [0, 0, 1], 40, 20)
    mid = geometry.edge_midpoints(corners)[edge]
    outward = geometry.normalize(mid)
    new = geometry.scale_edge(corners, edge, mid + 5 * outward)
    assert np.allclose(new.mean(axis=0), 0)
    assert np.allclose(geometry.edge_midpoints(new)[edge], mid + 5 * outward)
    w, h = geometry.rectangle_size(new)
    if edge in (0, 2):
        assert np.isclose(w, 40) and np.isclose(h, 30)
    else:
        assert np.isclose(w, 50) and np.isclose(h, 20)


def test_scale_edge_cannot_collapse_or_flip():
    corners = geometry.tangent_rectangle([0, 0, 0], [0, 0, 1], 40, 20)
    # Drag edge 0 far past the opposite edge
    mid = geometry.edge_midpoints(corners)[0]
    new = geometry.scale_edge(corners, 0, -10 * mid, min_size=2.0)
    w, h = geometry.rectangle_size(new)
    assert np.isclose(w, 40)
    assert np.isclose(h, 2.0)
    _, _, _, n, _, _ = geometry.rectangle_frame(new)
    assert np.allclose(n, [0, 0, 1])


def test_footprint_mask_central_projection():
    apex = np.array([0.0, 0.0, 0.0])
    corners = geometry.tangent_rectangle([0, 0, 100], [0, 0, 1], 20, 20)
    points = np.array([
        [0, 0, 100],      # centre of the base
        [0, 0, 200],      # further along the axis
        [15, 0, 200],     # inside the widening pyramid (15/200 < 10/100)
        [15, 0, 100],     # outside the base
        [0, 0, -100],     # behind the apex
        [0, 0, 0],        # the apex itself
    ])
    mask = geometry.footprint_mask(points, corners, apex)
    assert mask.tolist() == [True, True, True, False, False, False]


def test_safety_colors_endpoints():
    rgb = geometry.safety_colors([0.0, 0.5, 1.0, -1, 2])
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[0], np.round(geometry.SAFETY_RED * 255))
    assert np.array_equal(rgb[1], np.round(geometry.SAFETY_ORANGE * 255))
    assert np.array_equal(rgb[2], np.round(geometry.SAFETY_GREEN * 255))
    assert np.array_equal(rgb[3], rgb[0]) and np.array_equal(rgb[4], rgb[2])

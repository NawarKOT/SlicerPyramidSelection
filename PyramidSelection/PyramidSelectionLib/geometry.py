"""Pure NumPy geometry for the pyramid selection.

Nothing in this file depends on Slicer or VTK, so it can be unit-tested with
plain ``pytest``.

Conventions
-----------
A footprint rectangle is a ``(4, 3)`` array of corners ordered
counter-clockwise around the outward surface normal::

    c3 ---- c2
    |        |
    c0 ---- c1

Edge ``i`` joins corner ``i`` and corner ``(i + 1) % 4``.
"""

import numpy as np

_EPS = 1e-9


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < _EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def tangent_frame(normal):
    """Return two unit vectors ``(u, v)`` such that ``(u, v, normal)`` is a
    right-handed orthonormal frame.

    An auxiliary reference axis is switched when the normal is nearly aligned
    with it, to avoid a degenerate cross product.
    """
    n = normalize(normal)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(n, ref))
    v = normalize(np.cross(n, u))
    # (u, v, n) right-handed requires u x v == n
    if np.dot(np.cross(u, v), n) < 0:
        v = -v
    return u, v


def tangent_rectangle(center, normal, width, height):
    """Rectangle of size ``width`` x ``height`` centred on ``center`` and lying
    in the plane orthogonal to ``normal``."""
    if width <= 0 or height <= 0:
        raise ValueError("Rectangle width and height must be positive")
    center = np.asarray(center, dtype=float)
    u, v = tangent_frame(normal)
    hw, hh = width / 2.0, height / 2.0
    return np.array([
        center - u * hw - v * hh,
        center + u * hw - v * hh,
        center + u * hw + v * hh,
        center - u * hw + v * hh,
    ])


def rectangle_frame(corners):
    """Return ``(origin, u, v, n, width, height)`` of a rectangle."""
    c = np.asarray(corners, dtype=float)
    edge_u = c[1] - c[0]
    edge_v = c[3] - c[0]
    width = np.linalg.norm(edge_u)
    height = np.linalg.norm(edge_v)
    u = normalize(edge_u)
    n = normalize(np.cross(edge_u, c[2] - c[0]))
    v = np.cross(n, u)
    return c[0], u, v, n, width, height


def rectangle_size(corners):
    _, _, _, _, width, height = rectangle_frame(corners)
    return width, height


def edge_midpoints(corners):
    c = np.asarray(corners, dtype=float)
    return (c + np.roll(c, -1, axis=0)) / 2.0


def scale_edge(corners, edge_index, new_midpoint, min_size=2.0):
    """Resize the rectangle symmetrically about its centre so that edge
    ``edge_index`` passes through the projection of ``new_midpoint``.

    The extent orthogonal to the dragged edge is clamped to ``min_size`` so
    the rectangle can never collapse or flip.
    """
    c = np.array(corners, dtype=float)
    i0 = edge_index % 4
    i1 = (i0 + 1) % 4
    j0 = (i0 + 2) % 4
    j1 = (i0 + 3) % 4

    center = c.mean(axis=0)
    _, _, _, n, _, _ = rectangle_frame(c)
    edge_dir = normalize(c[i1] - c[i0])
    # In-plane direction pointing from the centre towards the dragged edge
    outward = np.cross(edge_dir, n)
    edge_mid = (c[i0] + c[i1]) / 2.0
    if np.dot(edge_mid - center, outward) < 0:
        outward = -outward

    half_extent = np.dot(np.asarray(new_midpoint, dtype=float) - center, outward)
    half_extent = max(half_extent, min_size / 2.0)
    current_half = np.dot(edge_mid - center, outward)
    delta = half_extent - current_half

    c[i0] += outward * delta
    c[i1] += outward * delta
    c[j0] -= outward * delta
    c[j1] -= outward * delta
    return c


def footprint_mask(points, corners, apex):
    """Central-projection footprint test.

    For each point ``p`` the ray from ``apex`` (target centre) through ``p``
    is intersected with the rectangle plane. A point is selected when that
    intersection lies inside the rectangle, i.e. when ``p`` lies inside the
    infinite pyramid defined by ``apex`` and the rectangle.

    Parameters
    ----------
    points : (N, 3) array
    corners : (4, 3) array
    apex : (3,) array

    Returns
    -------
    (N,) boolean array
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    apex = np.asarray(apex, dtype=float)
    origin, u, v, n, width, height = rectangle_frame(corners)

    direction = points - apex
    norms = np.linalg.norm(direction, axis=1)
    valid = norms > 1e-6
    direction[valid] /= norms[valid, None]

    denom = direction @ n
    valid &= np.abs(denom) > 1e-8
    t = np.full(len(points), -1.0)
    t[valid] = np.dot(n, origin - apex) / denom[valid]
    valid &= t >= 0

    hit = apex + t[:, None] * direction
    w = hit - origin
    u_coord = w @ u
    v_coord = w @ v
    inside = (u_coord >= 0) & (u_coord <= width) & (v_coord >= 0) & (v_coord <= height)
    return valid & inside


# Safety colour scale: red (0, unsafe) -> orange (0.5) -> green (1, safe)
SAFETY_RED = np.array([0.863, 0.196, 0.196])
SAFETY_ORANGE = np.array([0.941, 0.627, 0.157])
SAFETY_GREEN = np.array([0.196, 0.706, 0.314])


def safety_colors(scores):
    """Map safety scores in ``[0, 1]`` to RGB uint8 colours, shape ``(N, 3)``."""
    s = np.clip(np.asarray(scores, dtype=float).reshape(-1), 0.0, 1.0)
    low = s < 0.5
    t = np.where(low, s / 0.5, (s - 0.5) / 0.5)[:, None]
    rgb = np.where(
        low[:, None],
        SAFETY_RED + (SAFETY_ORANGE - SAFETY_RED) * t,
        SAFETY_ORANGE + (SAFETY_GREEN - SAFETY_ORANGE) * t,
    )
    return np.round(rgb * 255).astype(np.uint8)

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from PyramidSelectionLib import geometry, risk, surface  # noqa: E402


def sphere(radius, center=(0, 0, 0), resolution=80):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetCenter(*center)
    source.SetThetaResolution(resolution)
    source.SetPhiResolution(resolution)
    source.Update()
    return source.GetOutput()


@pytest.fixture(scope="module")
def mesh():
    return surface.SurfaceMesh(sphere(100))


def test_snap_and_outward_normal(mesh):
    vertex, _ = mesh.closestVertex([0, 0, 150])
    assert np.isclose(np.linalg.norm(vertex), 100, atol=1e-3)
    normal = mesh.normalAt(vertex)
    assert np.dot(normal, vertex / np.linalg.norm(vertex)) > 0.99


def test_patch_is_local_and_inside_footprint(mesh):
    apex = np.zeros(3)
    center, _ = mesh.closestVertex([0, 0, 100])
    corners = geometry.tangent_rectangle(center, mesh.normalAt(center), 40, 40)
    ids = mesh.footprintCellIds(corners, apex)
    assert len(ids) > 0
    centers = mesh.cellCenters[ids]
    assert np.all(centers[:, 2] > 0)  # never the far side of the surface
    assert np.all(np.abs(centers[:, :2]) <= 20 + 1e-6)

    patch = mesh.extractCells(ids)
    assert patch.GetNumberOfCells() == len(ids)
    # Independent copy
    assert patch.GetPoints() is not mesh.polydata.GetPoints()


def test_obstacle_clearance_scores():
    obstacle = sphere(10, center=(0, 0, 50), resolution=40)
    provider = risk.ObstacleClearanceRisk([obstacle], safeDistance=10, samples=30)
    target = np.zeros(3)
    entries = np.array([
        [0, 0, 100],    # straight through the obstacle
        [0, 60, 80],    # passes well away from it
    ])
    scores = provider.compute(entries, target)
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(1.0)


def test_obstacle_clearance_requires_obstacles():
    with pytest.raises(ValueError):
        risk.ObstacleClearanceRisk([])

"""Safety scoring of candidate entry points.

A *risk provider* maps candidate surface entry points to a safety score in
``[0, 1]`` (0 = unsafe, 1 = safe) for a straight path going to ``target``.
Any object with a matching ``compute`` method can be plugged into the module,
so other scoring methods can be added without touching the selection code.
"""

import numpy as np
import vtk
from vtk.util import numpy_support


class RiskProvider:
    name = "None"

    def compute(self, entryPoints, target):
        """Return an ``(N,)`` array of safety scores in ``[0, 1]``."""
        raise NotImplementedError


class ObstacleClearanceRisk(RiskProvider):
    """Straight-path clearance to user-selected obstacle surfaces.

    - Paths that cross an obstacle surface get a score of 0.
    - Otherwise the minimum distance between the path and the obstacles is
      estimated on ``samples`` points and mapped linearly to ``[0, 1]``,
      reaching 1 at ``safeDistance`` millimetres.

    Distances are evaluated in a single VTK call over all sample points.
    """

    name = "Obstacle clearance"

    def __init__(self, obstaclePolyDatas, safeDistance=10.0, samples=20):
        if safeDistance <= 0:
            raise ValueError("safeDistance must be positive")
        if samples < 2:
            raise ValueError("samples must be >= 2")
        self.safeDistance = float(safeDistance)
        self.samples = int(samples)

        append = vtk.vtkAppendPolyData()
        for poly in obstaclePolyDatas:
            if poly is not None and poly.GetNumberOfCells() > 0:
                append.AddInputData(poly)
        if append.GetNumberOfInputConnections(0) == 0:
            raise ValueError("No non-empty obstacle mesh was given")
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(append.GetOutputPort())
        triangulate.Update()
        self._obstacles = triangulate.GetOutput()

        self._obbTree = vtk.vtkOBBTree()
        self._obbTree.SetDataSet(self._obstacles)
        self._obbTree.BuildLocator()

        self._distance = vtk.vtkImplicitPolyDataDistance()
        self._distance.SetInput(self._obstacles)

    def compute(self, entryPoints, target):
        entries = np.asarray(entryPoints, dtype=float).reshape(-1, 3)
        target = np.asarray(target, dtype=float)
        n = len(entries)
        if n == 0:
            return np.zeros(0)

        # 1. Direct intersection test
        hits = np.zeros(n, dtype=bool)
        scratch = vtk.vtkPoints()
        for i in range(n):
            hits[i] = self._obbTree.IntersectWithLine(entries[i], target, scratch, None) != 0

        # 2. Clearance along the path, all samples evaluated at once
        t = np.linspace(0.0, 1.0, self.samples)
        samplePoints = entries[:, None, :] + t[None, :, None] * (target - entries)[:, None, :]
        distances = self._evaluateDistance(samplePoints.reshape(-1, 3)).reshape(n, self.samples)
        clearance = np.abs(distances).min(axis=1)

        scores = np.clip(clearance / self.safeDistance, 0.0, 1.0)
        scores[hits] = 0.0
        return scores

    def _evaluateDistance(self, points):
        vtkPoints = vtk.vtkPoints()
        vtkPoints.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points), deep=True))
        cloud = vtk.vtkPolyData()
        cloud.SetPoints(vtkPoints)

        sampler = vtk.vtkSampleImplicitFunctionFilter()
        sampler.SetImplicitFunction(self._distance)
        sampler.ComputeGradientsOff()
        sampler.SetInputData(cloud)
        sampler.Update()
        return numpy_support.vtk_to_numpy(sampler.GetOutput().GetPointData().GetScalars()).copy()

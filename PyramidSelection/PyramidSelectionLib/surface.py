"""VTK helpers around the surface mesh.

Locators, normals and cell centres are built once per mesh and reused for
every interaction, instead of being rebuilt on each drag event.
"""

import numpy as np
import vtk
from vtk.util import numpy_support

from . import geometry


class SurfaceMesh:
    """Cached spatial queries on a surface ``vtkPolyData``."""

    def __init__(self, polydata):
        if polydata is None or polydata.GetNumberOfCells() == 0:
            raise ValueError("Surface mesh is empty")
        self.polydata = polydata
        self._mtime = polydata.GetMTime()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.Update()
        self._withNormals = normals.GetOutput()
        self._pointNormals = numpy_support.vtk_to_numpy(
            self._withNormals.GetPointData().GetNormals())

        self._pointLocator = vtk.vtkStaticPointLocator()
        self._pointLocator.SetDataSet(self._withNormals)
        self._pointLocator.BuildLocator()

        self._cellLocator = vtk.vtkStaticCellLocator()
        self._cellLocator.SetDataSet(polydata)
        self._cellLocator.BuildLocator()

        self.cellCenters = cellCenters(polydata)

    def isUpToDate(self, polydata):
        return polydata is self.polydata and polydata.GetMTime() == self._mtime

    def closestVertex(self, point):
        """Return ``(vertex_position, vertex_id)`` of the mesh vertex closest
        to ``point``."""
        pid = self._pointLocator.FindClosestPoint(np.asarray(point, dtype=float))
        return np.array(self._withNormals.GetPoint(pid)), pid

    def normalAt(self, point):
        """Unit outward normal at the vertex closest to ``point``."""
        _, pid = self.closestVertex(point)
        return geometry.normalize(self._pointNormals[pid])

    def cellsInBox(self, center, radius):
        center = np.asarray(center, dtype=float)
        bounds = [center[0] - radius, center[0] + radius,
                  center[1] - radius, center[1] + radius,
                  center[2] - radius, center[2] + radius]
        ids = vtk.vtkIdList()
        self._cellLocator.FindCellsWithinBounds(bounds, ids)
        return np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)

    def footprintCellIds(self, corners, apex):
        """Ids of surface cells whose centre lies inside the pyramid
        ``(apex, corners)``, restricted to the neighbourhood of the footprint
        so that the far side of the surface is never selected."""
        corners = np.asarray(corners, dtype=float)
        width, height = geometry.rectangle_size(corners)
        candidates = self.cellsInBox(corners.mean(axis=0), 1.2 * max(width, height))
        if len(candidates) == 0:
            return candidates
        mask = geometry.footprint_mask(self.cellCenters[candidates], corners, apex)
        return candidates[mask]

    def extractCells(self, cellIds):
        """Copy the given cells into a new, independent ``vtkPolyData``."""
        idArray = numpy_support.numpy_to_vtkIdTypeArray(
            np.asarray(cellIds, dtype=np.int64), deep=True)

        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(idArray)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, self.polydata)
        extract.SetInputData(1, selection)

        toPolyData = vtk.vtkGeometryFilter()
        toPolyData.SetInputConnection(extract.GetOutputPort())
        toPolyData.Update()

        patch = vtk.vtkPolyData()
        patch.DeepCopy(toPolyData.GetOutput())
        return patch


def centerOfMass(polydata):
    com = vtk.vtkCenterOfMass()
    com.SetInputData(polydata)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    return np.array(com.GetCenter())


def cellCenters(polydata):
    centers = vtk.vtkCellCenters()
    centers.SetInputData(polydata)
    centers.VertexCellsOff()
    centers.Update()
    points = centers.GetOutput().GetPoints()
    if points is None:
        return np.zeros((0, 3))
    return numpy_support.vtk_to_numpy(points.GetData()).copy()


def setCellColors(polydata, rgb, name="SafetyColor"):
    """Attach an ``(N, 3)`` uint8 array as cell scalars."""
    array = numpy_support.numpy_to_vtk(np.ascontiguousarray(rgb, dtype=np.uint8), deep=True,
                                       array_type=vtk.VTK_UNSIGNED_CHAR)
    array.SetName(name)
    polydata.GetCellData().AddArray(array)
    polydata.GetCellData().SetActiveScalars(name)
    return array

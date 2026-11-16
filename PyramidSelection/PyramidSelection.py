import logging
import time
from typing import Annotated, Optional

import numpy as np
import qt
import vtk

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.parameterNodeWrapper import WithinRange, parameterNodeWrapper
from slicer.util import VTKObservationMixin

from PyramidSelectionLib import geometry
from PyramidSelectionLib.risk import ObstacleClearanceRisk
from PyramidSelectionLib.surface import SurfaceMesh, cellCenters, centerOfMass, setCellColors


#
# Module
#


class PyramidSelection(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Pyramid Selection"
        self.parent.categories = ["Surface Models"]
        self.parent.dependencies = []
        self.parent.contributors = ["Nawar Koteish"]
        self.parent.helpText = """
Interactive selection of a region on a surface model, linked to a target by a pyramid-shaped corridor.
<ol>
<li>Choose the surface and target models.</li>
<li>Click <b>Place pyramid</b>, then click on the surface in the 3D view.</li>
<li>Drag the yellow handle to move the footprint on the surface, and the purple handles to resize it.</li>
</ol>
The surface cells inside the pyramid (apex at the target center of mass) are extracted as a patch.
If obstacles are selected, each cell is colored by the clearance of the straight path from the cell to the target
(red: crosses or touches an obstacle, green: farther than the safe distance).
"""
        self.parent.acknowledgementText = ""


#
# Parameter node
#


@parameterNodeWrapper
class PyramidSelectionParameterNode:
    surfaceModel: slicer.vtkMRMLModelNode
    targetModel: slicer.vtkMRMLModelNode
    width: Annotated[float, WithinRange(2.0, 300.0)] = 40.0
    height: Annotated[float, WithinRange(2.0, 300.0)] = 40.0
    debounceMs: Annotated[int, WithinRange(0, 2000)] = 150
    liveUpdate: bool = True
    showSurfaceMesh: bool = True
    obstacles: list[slicer.vtkMRMLModelNode]
    safeDistance: Annotated[float, WithinRange(0.1, 100.0)] = 10.0
    pathSamples: Annotated[int, WithinRange(2, 200)] = 20


#
# Widget
#


class PyramidSelectionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._placementNode = None
        self._handleObservations = []
        self._inputObservations = []
        self._updatingHandles = False
        self._interactionTimer = None
        self._lastLiveUpdate = 0.0
        self._lastPatchSeconds = 0.0

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PyramidSelection.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = PyramidSelectionLogic()

        self._interactionTimer = qt.QTimer()
        self._interactionTimer.setSingleShot(True)
        self._interactionTimer.timeout.connect(self.onInteractionFinished)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.placeButton.connect("clicked(bool)", self.onPlaceButton)
        self.ui.resetButton.connect("clicked(bool)", self.onResetButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)
        self.ui.obstaclesSelector.connect("checkedNodesChanged()", self.onObstaclesChanged)
        self.ui.showSurfaceMeshCheckBox.connect("toggled(bool)", self.onShowSurfaceMeshToggled)
        self.ui.surfaceSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda node: self._applySurfaceMeshDisplay())

        self.initializeParameterNode()
        self._updateButtonStates()

    def cleanup(self) -> None:
        self._cancelPlacement()
        self._removeHandleObservers()
        self._removeInputObservers()
        if self._interactionTimer:
            self._interactionTimer.stop()
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        self._cancelPlacement()
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)

    def onSceneStartClose(self, caller, event) -> None:
        self._cancelPlacement()
        self._removeHandleObservers()
        self._removeInputObservers()
        self._interactionTimer.stop()
        self.logic.reset()
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()
        self._updateButtonStates()

    # --- Parameter node -----------------------------------------------------

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.surfaceModel:
            self._parameterNode.surfaceModel = _findModelByName(["surface"])
        if not self._parameterNode.targetModel:
            self._parameterNode.targetModel = _findModelByName(["target"])

    def setParameterNode(self, inputParameterNode: Optional[PyramidSelectionParameterNode]) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._onParameterNodeModified)
            self._obstaclesToGui()
        self._updateButtonStates()

    def _obstaclesToGui(self) -> None:
        """Check the obstacles stored in the parameter node (saved with the scene)."""
        combo = self.ui.obstaclesSelector
        saved = [n for n in self._parameterNode.obstacles if n is not None]
        wasBlocked = combo.blockSignals(True)
        try:
            for i in range(combo.nodeCount()):
                node = combo.nodeFromIndex(i)
                combo.setCheckState(node, qt.Qt.Checked if node in saved else qt.Qt.Unchecked)
        finally:
            combo.blockSignals(wasBlocked)

    def _checkedObstacles(self):
        p = self._parameterNode
        return [n for n in p.obstacles
                if n is not None and n.GetScene() is not None and n not in (p.surfaceModel, p.targetModel)]

    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        self._updateButtonStates()

    def _updateButtonStates(self) -> None:
        p = self._parameterNode
        ready = bool(p and p.surfaceModel and p.targetModel and p.surfaceModel != p.targetModel)
        self.ui.placeButton.enabled = ready
        self.ui.placeButton.toolTip = (
            "Click on the surface in the 3D view to place the footprint center." if ready
            else "Select two different models for the surface and the target.")
        hasSelection = self.logic is not None and self.logic.hasSelection()
        self.ui.resetButton.enabled = hasSelection
        self.ui.exportButton.enabled = hasSelection

    # --- Placement ------------------------------------------------------------

    def onPlaceButton(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to start placement.", waitCursor=True):
            self._cancelPlacement()
            self._resetSelection()
            p = self._parameterNode
            self.logic.setInputs(p.surfaceModel, p.targetModel)
            self._applySurfaceMeshDisplay()

            self._placementNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", slicer.mrmlScene.GenerateUniqueName("PyramidPlacement"))
            self._placementNode.SetAttribute(PyramidSelectionLogic.ROLE_ATTRIBUTE, "placement")
            self.addObserver(self._placementNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                             self.onPlacementPointDefined)

            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
            selectionNode.SetActivePlaceNodeID(self._placementNode.GetID())
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetPlaceModePersistence(False)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
            self.ui.statusLabel.text = "Click on the surface in the 3D view."

    def onPlacementPointDefined(self, caller, event) -> None:
        if caller.GetNumberOfControlPoints() < 1:
            return
        point = _controlPointWorld(caller)
        # Remove the temporary node outside of its own callback
        qt.QTimer.singleShot(0, lambda: self._finishPlacement(point))

    def _finishPlacement(self, point) -> None:
        self._cancelPlacement()
        with slicer.util.tryWithErrorDisplay("Failed to create the pyramid.", waitCursor=True):
            p = self._parameterNode
            self.logic.createSelection(point, p.width, p.height)
            self._addHandleObservers()
            self._addInputObservers()
            self._refreshPatch()
        self._updateButtonStates()

    def _cancelPlacement(self) -> None:
        if self._placementNode is None:
            return
        self.removeObserver(self._placementNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                            self.onPlacementPointDefined)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if interactionNode.GetCurrentInteractionMode() == interactionNode.Place:
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        if self._placementNode.GetScene():
            slicer.mrmlScene.RemoveNode(self._placementNode)
        self._placementNode = None

    # --- Handle interaction -------------------------------------------------

    def _addHandleObservers(self) -> None:
        self._removeHandleObservers()
        event = slicer.vtkMRMLMarkupsNode.PointModifiedEvent
        center = self.logic.centerHandle
        self._handleObservations.append(
            (center, center.AddObserver(event, lambda caller, ev: self.onCenterHandleMoved(caller))))
        for index, handle in enumerate(self.logic.edgeHandles):
            self._handleObservations.append(
                (handle, handle.AddObserver(event, lambda caller, ev, i=index: self.onEdgeHandleMoved(caller, i))))

    def _removeHandleObservers(self) -> None:
        for node, tag in self._handleObservations:
            node.RemoveObserver(tag)
        self._handleObservations = []

    def onCenterHandleMoved(self, caller) -> None:
        if self._updatingHandles:
            return
        self._updatingHandles = True
        try:
            self.logic.moveCenter(_controlPointWorld(caller))
        finally:
            self._updatingHandles = False
        self._onHandleDragged()

    def onEdgeHandleMoved(self, caller, edgeIndex) -> None:
        if self._updatingHandles:
            return
        self._updatingHandles = True
        try:
            self.logic.resizeEdge(edgeIndex, _controlPointWorld(caller),
                                  draggedHandle=caller)
        finally:
            self._updatingHandles = False
        self._onHandleDragged()

    def _onHandleDragged(self) -> None:
        """Live mode: recompute the patch during the drag, throttled so that
        rendering stays smooth even for large patches. Otherwise hide it until
        the drag stops. In both modes a final update runs after the delay."""
        if self._parameterNode.liveUpdate:
            now = time.perf_counter()
            minInterval = max(0.03, 2.0 * self._lastPatchSeconds)
            if now - self._lastLiveUpdate >= minInterval:
                try:
                    self._refreshPatch()
                except Exception as e:
                    logging.error(f"Live patch update failed: {e}")
                self._lastLiveUpdate = time.perf_counter()
        else:
            self.logic.setPatchVisible(False)
        self._interactionTimer.start(self._parameterNode.debounceMs)

    def onInteractionFinished(self) -> None:
        if not self.logic.hasSelection():
            return
        with slicer.util.tryWithErrorDisplay("Failed to update the patch.", waitCursor=True):
            # Re-read moved/edited inputs, then snap the handles back onto the
            # rectangle (guarded so that moving handles does not re-trigger us)
            self._updatingHandles = True
            try:
                self.logic.refreshInputs()
                self.logic.updateHandles()
            finally:
                self._updatingHandles = False
            self._refreshPatch()

    # --- Patch ---------------------------------------------------------------

    def onObstaclesChanged(self) -> None:
        if self._parameterNode is None:
            return
        self._parameterNode.obstacles = list(self.ui.obstaclesSelector.checkedNodes())
        if self.logic.hasSelection():
            self._addInputObservers()
            self._interactionTimer.start(0)

    # --- Input change tracking ----------------------------------------------------

    def _addInputObservers(self) -> None:
        """Recompute when the surface, the target or an obstacle mesh is edited or
        moved by a transform after the pyramid was placed."""
        self._removeInputObservers()
        p = self._parameterNode
        nodes = [p.surfaceModel, p.targetModel] + self._checkedObstacles()
        events = (slicer.vtkMRMLModelNode.MeshModifiedEvent, slicer.vtkMRMLTransformableNode.TransformModifiedEvent)
        for node in nodes:
            if node is None:
                continue
            for event in events:
                self._inputObservations.append((node, node.AddObserver(event, self._onInputModified)))

    def _removeInputObservers(self) -> None:
        for node, tag in self._inputObservations:
            node.RemoveObserver(tag)
        self._inputObservations = []

    def _onInputModified(self, caller, event) -> None:
        if self.logic.hasSelection() and self._parameterNode is not None:
            self._interactionTimer.start(self._parameterNode.debounceMs)

    def _refreshPatch(self) -> None:
        p = self._parameterNode
        self.logic.setObstacles(self._checkedObstacles(), p.safeDistance, p.pathSamples)
        self.logic.showMeshEdges = p.showSurfaceMesh
        count, elapsed = self.logic.updatePatch()
        self._lastPatchSeconds = elapsed
        width, height = geometry.rectangle_size(self.logic.corners)
        coloring = "safety colored" if self.logic.riskProvider else "no obstacles selected"
        self.ui.statusLabel.text = (
            f"Footprint {width:.0f} x {height:.0f} mm, patch of {count} triangles ({coloring}), "
            f"computed in {elapsed * 1000:.0f} ms.")

    def onShowSurfaceMeshToggled(self, checked) -> None:
        self._applySurfaceMeshDisplay()
        self.logic.setPatchEdgesVisible(checked)

    def _applySurfaceMeshDisplay(self) -> None:
        p = self._parameterNode
        if p is None or p.surfaceModel is None:
            return
        self.logic.setSurfaceMeshVisible(p.surfaceModel, p.showSurfaceMesh)

    def onExportButton(self) -> None:
        with slicer.util.tryWithErrorDisplay("Failed to export the patch.", waitCursor=True):
            node = self.logic.exportPatch()
            self.ui.statusLabel.text = f"Patch exported as '{node.GetName()}'."

    def onResetButton(self) -> None:
        self._resetSelection()
        self.ui.statusLabel.text = ""

    def _resetSelection(self) -> None:
        self._interactionTimer.stop()
        self._removeHandleObservers()
        self._removeInputObservers()
        self.logic.removeSelection()
        self._updateButtonStates()


def worldPolyData(modelNode):
    """Mesh of ``modelNode`` in world coordinates (parent transforms applied)."""
    poly = modelNode.GetPolyData()
    transformNode = modelNode.GetParentTransformNode()
    if poly is None or transformNode is None:
        return poly
    toWorld = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transformNode, None, toWorld)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(toWorld)
    transformFilter.SetInputData(poly)
    transformFilter.Update()
    return transformFilter.GetOutput()


def _inputKey(modelNode):
    """Changes whenever the mesh or any parent transform of ``modelNode`` changes."""
    poly = modelNode.GetPolyData()
    key = [modelNode.GetID(), poly.GetMTime() if poly is not None else 0]
    transformNode = modelNode.GetParentTransformNode()
    while transformNode is not None:
        toParent = transformNode.GetTransformToParent()
        key.append((transformNode.GetID(), transformNode.GetMTime(),
                    toParent.GetMTime() if toParent is not None else 0))
        transformNode = transformNode.GetParentTransformNode()
    return tuple(key)


def _controlPointWorld(markupsNode, index=0):
    position = [0.0, 0.0, 0.0]
    markupsNode.GetNthControlPointPositionWorld(index, position)
    return np.array(position)


def _findModelByName(keywords):
    for node in slicer.util.getNodesByClass("vtkMRMLModelNode"):
        if node.GetAttribute(PyramidSelectionLogic.ROLE_ATTRIBUTE):
            continue
        name = node.GetName().lower()
        if any(k in name for k in keywords):
            return node
    return None


#
# Logic
#


class PyramidSelectionLogic(ScriptedLoadableModuleLogic):
    """Owns the selection state and the nodes it creates.

    Can be used without the GUI, e.g. from the Python console::

        logic = slicer.modules.pyramidselection.widgetRepresentation().self().logic
        # or: logic = PyramidSelectionLogic()
        logic.setInputs(getNode("MySurface"), getNode("MyTarget"))
        logic.createSelection([x, y, z], 40, 40)
        logic.updatePatch()
    """

    ROLE_ATTRIBUTE = "PyramidSelection.Role"
    CENTER_COLOR = (1.0, 1.0, 0.0)
    EDGE_COLOR = (1.0, 0.0, 1.0)
    PYRAMID_COLOR = (1.0, 0.0, 0.0)
    PATCH_COLOR = (0.2, 0.5, 1.0)

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self._surface = None
        self.surfaceModel = None
        self.targetModel = None
        self._surfaceKey = None
        self._targetKey = None
        self.apex = None
        self.normal = None
        self.corners = None
        self.riskProvider = None
        self._riskKey = None
        self.rectangleNode = None
        self.pyramidNode = None
        self.patchNode = None
        self.centerHandle = None
        self.edgeHandles = []
        self.showMeshEdges = True

    def getParameterNode(self):
        return PyramidSelectionParameterNode(super().getParameterNode())

    # --- Inputs -----------------------------------------------------------------

    def setInputs(self, surfaceModel, targetModel) -> None:
        if surfaceModel is None or targetModel is None:
            raise ValueError("Surface and target models are required")
        if surfaceModel is not self.surfaceModel:
            self._surfaceKey = None
        if targetModel is not self.targetModel:
            self._targetKey = None
        self.surfaceModel = surfaceModel
        self.targetModel = targetModel
        self.refreshInputs()

    def refreshInputs(self) -> bool:
        """Re-read the surface and target if their mesh or parent transforms changed.

        All computations are done in world coordinates, so models placed under
        a transform are handled correctly. Returns True if anything changed.
        """
        if self.surfaceModel is None or self.targetModel is None:
            raise RuntimeError("Call setInputs() first")
        for node in (self.surfaceModel, self.targetModel):
            if node.GetScene() is None:
                raise RuntimeError(f"Input model '{node.GetName()}' was removed from the scene")
        changed = False

        key = _inputKey(self.surfaceModel)
        if key != self._surfaceKey:
            self._surface = SurfaceMesh(worldPolyData(self.surfaceModel))
            self._surfaceKey = key
            changed = True
            if self.hasSelection():
                # Keep the footprint on the (moved or edited) surface
                center, _ = self._surface.closestVertex(self.corners.mean(axis=0))
                width, height = geometry.rectangle_size(self.corners)
                self.normal = self._surface.normalAt(center)
                self.corners = geometry.tangent_rectangle(center, self.normal, width, height)

        key = _inputKey(self.targetModel)
        if key != self._targetKey:
            targetPoly = worldPolyData(self.targetModel)
            if targetPoly is None or targetPoly.GetNumberOfPoints() == 0:
                raise ValueError("Target mesh is empty")
            self.apex = centerOfMass(targetPoly)
            self._targetKey = key
            changed = True

        if changed and self.hasSelection():
            self._updateGeometry()
        return changed

    def setObstacles(self, obstacleModels, safeDistance=10.0, samples=20) -> None:
        """Select the safety scoring. An empty list disables coloring."""
        models = [m for m in obstacleModels
                  if m is not None and m.GetScene() is not None and m.GetPolyData() is not None]
        key = (tuple(_inputKey(m) for m in models), float(safeDistance), int(samples))
        if key == self._riskKey:
            return
        self._riskKey = key
        self.riskProvider = (ObstacleClearanceRisk([worldPolyData(m) for m in models], safeDistance, samples)
                             if models else None)

    # --- Selection --------------------------------------------------------------

    def hasSelection(self) -> bool:
        return self.corners is not None and self.rectangleNode is not None

    def createSelection(self, point, width=40.0, height=40.0) -> None:
        if self._surface is None or self.apex is None:
            raise RuntimeError("Call setInputs() first")
        self.removeSelection()
        center, _ = self._surface.closestVertex(point)
        self.normal = self._surface.normalAt(center)
        self.corners = geometry.tangent_rectangle(center, self.normal, width, height)

        self.rectangleNode = self._newModelNode("PyramidBaseRectangle", "rectangle", self._rectanglePolyData())
        dn = self.rectangleNode.GetDisplayNode()
        dn.SetLineWidth(3)
        dn.SetColor(*self.PYRAMID_COLOR)

        self.pyramidNode = self._newModelNode("PyramidToTarget", "pyramid", self._pyramidPolyData())
        dn = self.pyramidNode.GetDisplayNode()
        dn.SetColor(*self.PYRAMID_COLOR)
        dn.SetOpacity(0.4)

        self.centerHandle = self._newHandle("PyramidCenterHandle", "centerHandle", center, self.CENTER_COLOR, 5.0)
        self.edgeHandles = [
            self._newHandle(f"PyramidEdgeHandle{i}", "edgeHandle", mid, self.EDGE_COLOR, 4.0)
            for i, mid in enumerate(geometry.edge_midpoints(self.corners))
        ]

    def moveCenter(self, point) -> np.ndarray:
        """Move the footprint to the surface vertex closest to ``point``, keeping
        its size, and re-orient it in the new tangent plane."""
        center, _ = self._surface.closestVertex(point)
        width, height = geometry.rectangle_size(self.corners)
        self.normal = self._surface.normalAt(center)
        self.corners = geometry.tangent_rectangle(center, self.normal, width, height)
        self._updateGeometry()
        self.updateHandles()
        return center

    def resizeEdge(self, edgeIndex, point, draggedHandle=None) -> None:
        self.corners = geometry.scale_edge(self.corners, edgeIndex, point)
        self._updateGeometry()
        self.updateHandles(skip=draggedHandle)

    def updateHandles(self, skip=None) -> None:
        if not self.hasSelection():
            return
        if self.centerHandle is not skip:
            self.centerHandle.SetNthControlPointPositionWorld(0, *self.corners.mean(axis=0))
        for handle, mid in zip(self.edgeHandles, geometry.edge_midpoints(self.corners)):
            if handle is not skip:
                handle.SetNthControlPointPositionWorld(0, *mid)

    # --- Patch ------------------------------------------------------------------

    def updatePatch(self):
        """Extract (and color) the surface patch. Returns ``(numberOfCells, seconds)``."""
        if not self.hasSelection():
            raise RuntimeError("No selection")
        start = time.perf_counter()
        if self.refreshInputs():
            self.updateHandles()
        cellIds = self._surface.footprintCellIds(self.corners, self.apex)
        patch = self._surface.extractCells(cellIds)

        if self.patchNode is None or self.patchNode.GetScene() is None:
            self.patchNode = self._newModelNode("SelectedSurfacePatch", "patch", patch)
        else:
            self.patchNode.SetAndObservePolyData(patch)
        dn = self.patchNode.GetDisplayNode()
        dn.SetOpacity(0.9)
        dn.SetBackfaceCulling(False)
        dn.SetEdgeVisibility(self.showMeshEdges)

        if self.riskProvider is not None and patch.GetNumberOfCells() > 0:
            scores = self.riskProvider.compute(cellCenters(patch), self.apex)
            setCellColors(patch, geometry.safety_colors(scores))
            dn.SetActiveScalar("SafetyColor", vtk.vtkAssignAttribute.CELL_DATA)
            dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
            dn.SetScalarVisibility(True)
        else:
            dn.SetScalarVisibility(False)
            dn.SetColor(*self.PATCH_COLOR)
        dn.SetVisibility(True)
        return patch.GetNumberOfCells(), time.perf_counter() - start

    SURFACE_MESH_OPACITY = 0.3

    def setSurfaceMeshVisible(self, surfaceModel, showMesh: bool) -> None:
        """Show the surface in 3D; with ``showMesh`` it is semi-transparent with
        its triangle edges drawn, otherwise opaque without edges."""
        if surfaceModel is None:
            return
        surfaceModel.CreateDefaultDisplayNodes()
        dn = surfaceModel.GetDisplayNode()
        dn.SetVisibility(True)
        dn.SetVisibility3D(True)
        dn.SetEdgeVisibility(showMesh)
        dn.SetOpacity(self.SURFACE_MESH_OPACITY if showMesh else 1.0)

    def setPatchEdgesVisible(self, visible: bool) -> None:
        self.showMeshEdges = visible
        if self.patchNode and self.patchNode.GetDisplayNode():
            self.patchNode.GetDisplayNode().SetEdgeVisibility(visible)

    def setPatchVisible(self, visible: bool) -> None:
        if self.patchNode and self.patchNode.GetDisplayNode():
            self.patchNode.GetDisplayNode().SetVisibility(visible)

    def exportPatch(self, name="ExportedSurfacePatch"):
        if self.patchNode is None or self.patchNode.GetPolyData() is None:
            raise RuntimeError("No patch to export")
        poly = vtk.vtkPolyData()
        poly.DeepCopy(self.patchNode.GetPolyData())
        node = slicer.modules.models.logic().AddModel(poly)
        node.SetName(slicer.mrmlScene.GenerateUniqueName(name))
        src, dst = self.patchNode.GetDisplayNode(), node.GetDisplayNode()
        if src and dst:
            dst.CopyContent(src)
            dst.SetVisibility(True)
        return node

    # --- Node management ----------------------------------------------------------

    def _nodes(self):
        return [self.rectangleNode, self.pyramidNode, self.patchNode, self.centerHandle] + list(self.edgeHandles)

    def removeSelection(self) -> None:
        """Remove every node created by this logic."""
        for node in self._nodes():
            if node is not None and node.GetScene():
                slicer.mrmlScene.RemoveNode(node)
        self.forgetNodes()

    def forgetNodes(self) -> None:
        """Drop node references without touching the scene."""
        self.rectangleNode = self.pyramidNode = self.patchNode = self.centerHandle = None
        self.edgeHandles = []
        self.corners = None

    def reset(self) -> None:
        """Forget nodes and cached inputs (e.g. when the scene is closed)."""
        self.forgetNodes()
        self._surface = None
        self.apex = None
        self.surfaceModel = self.targetModel = None
        self._surfaceKey = self._targetKey = None
        self.riskProvider = None
        self._riskKey = None

    def _newModelNode(self, name, role, polydata):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", slicer.mrmlScene.GenerateUniqueName(name))
        node.SetAttribute(self.ROLE_ATTRIBUTE, role)
        node.SetAndObservePolyData(polydata)
        node.CreateDefaultDisplayNodes()
        return node

    def _newHandle(self, name, role, position, color, glyphScale):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                  slicer.mrmlScene.GenerateUniqueName(name))
        node.SetAttribute(self.ROLE_ATTRIBUTE, role)
        node.AddControlPointWorld(*position)
        node.SetFixedNumberOfControlPoints(True)
        node.SetLocked(False)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        dn.SetSelectedColor(*color)
        dn.SetColor(*color)
        dn.SetGlyphScale(glyphScale)
        dn.SetPointLabelsVisibility(False)
        dn.SetPropertiesLabelVisibility(False)
        return node

    def _rectanglePolyData(self):
        points = vtk.vtkPoints()
        for corner in self.corners:
            points.InsertNextPoint(corner)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(5)
        for i in range(5):
            line.GetPointIds().SetId(i, i % 4)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        return poly

    def _pyramidPolyData(self):
        points = vtk.vtkPoints()
        for p in list(self.corners) + [self.apex]:
            points.InsertNextPoint(p)
        faces = vtk.vtkCellArray()
        for i in range(4):
            faces.InsertNextCell(3, [i, (i + 1) % 4, 4])
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(faces)
        return poly

    def _updateGeometry(self) -> None:
        """Lightweight update used while dragging: only moves existing points."""
        for node, pts in ((self.rectangleNode, self.corners),
                          (self.pyramidNode, list(self.corners) + [self.apex])):
            poly = node.GetPolyData()
            vtkPts = poly.GetPoints()
            for i, p in enumerate(pts):
                vtkPts.SetPoint(i, p)
            vtkPts.Modified()
            poly.Modified()


#
# Test
#


class PyramidSelectionTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_SelectionOnSyntheticScene()
        self.setUp()
        self.test_TransformsAndInputChanges()
        self.setUp()
        self.test_ObstaclesSavedInParameterNode()

    @staticmethod
    def _sphereModel(name, radius, center):
        source = vtk.vtkSphereSource()
        source.SetRadius(radius)
        source.SetCenter(*center)
        source.SetThetaResolution(80)
        source.SetPhiResolution(80)
        source.Update()
        node = slicer.modules.models.logic().AddModel(source.GetOutput())
        node.SetName(name)
        return node

    def test_SelectionOnSyntheticScene(self):
        self.delayDisplay("Building synthetic scene")
        surface = self._sphereModel("surface", 100.0, (0, 0, 0))
        target = self._sphereModel("target", 5.0, (0, 0, 0))
        obstacle = self._sphereModel("obstacle", 10.0, (0, 0, 50))

        logic = PyramidSelectionLogic()
        logic.setInputs(surface, target)
        np.testing.assert_allclose(logic.apex, [0, 0, 0], atol=1e-6)

        self.delayDisplay("Creating selection above the obstacle")
        logic.setObstacles([obstacle], safeDistance=10.0, samples=30)
        logic.createSelection([0, 0, 120], 40.0, 40.0)
        self.assertTrue(logic.hasSelection())
        width, height = geometry.rectangle_size(logic.corners)
        self.assertAlmostEqual(width, 40.0, places=6)
        self.assertAlmostEqual(height, 40.0, places=6)
        count, _ = logic.updatePatch()
        self.assertGreater(count, 0)
        colors = logic.patchNode.GetPolyData().GetCellData().GetArray("SafetyColor")
        self.assertIsNotNone(colors)
        red = tuple(geometry.safety_colors([0.0])[0])
        self.assertIn(red, {tuple(int(c) for c in colors.GetTuple3(i)) for i in range(colors.GetNumberOfTuples())})

        self.delayDisplay("Moving footprint away from the obstacle")
        newCenter = logic.moveCenter([0, 120, 0])
        np.testing.assert_allclose(np.linalg.norm(newCenter), 100.0, atol=1e-3)
        logic.updatePatch()
        colors = logic.patchNode.GetPolyData().GetCellData().GetArray("SafetyColor")
        green = tuple(geometry.safety_colors([1.0])[0])
        self.assertEqual({tuple(int(c) for c in colors.GetTuple3(i)) for i in range(colors.GetNumberOfTuples())},
                         {green})

        self.delayDisplay("Resizing an edge")
        mid = geometry.edge_midpoints(logic.corners)[0]
        outward = geometry.normalize(mid - logic.corners.mean(axis=0))
        logic.resizeEdge(0, mid + 10.0 * outward)
        sizes = sorted(geometry.rectangle_size(logic.corners))
        np.testing.assert_allclose(sizes, [40.0, 60.0], atol=1e-6)

        self.delayDisplay("Export and cleanup")
        exported = logic.exportPatch()
        self.assertGreater(exported.GetPolyData().GetNumberOfCells(), 0)
        logic.removeSelection()
        self.assertFalse(logic.hasSelection())
        remaining = [n for n in slicer.util.getNodesByClass("vtkMRMLNode")
                     if n.GetAttribute(PyramidSelectionLogic.ROLE_ATTRIBUTE)]
        self.assertEqual(remaining, [])
        self.delayDisplay("Test passed")

    def test_TransformsAndInputChanges(self):
        self.delayDisplay("Models under a transform")
        surface = self._sphereModel("surface", 100.0, (0, 0, 0))
        target = self._sphereModel("target", 5.0, (0, 0, 0))
        transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        matrix = vtk.vtkMatrix4x4()
        matrix.SetElement(0, 3, 200.0)
        transformNode.SetMatrixTransformToParent(matrix)
        surface.SetAndObserveTransformNodeID(transformNode.GetID())
        target.SetAndObserveTransformNodeID(transformNode.GetID())

        logic = PyramidSelectionLogic()
        logic.setInputs(surface, target)
        np.testing.assert_allclose(logic.apex, [200, 0, 0], atol=1e-6)
        logic.createSelection([200, 0, 130], 40.0, 40.0)
        center = logic.corners.mean(axis=0)
        np.testing.assert_allclose(np.linalg.norm(center - [200, 0, 0]), 100.0, atol=1e-3)
        logic.updatePatch()
        patchCenters = cellCenters(logic.patchNode.GetPolyData())
        np.testing.assert_allclose(patchCenters.mean(axis=0)[:2], [200, 0], atol=2.0)

        self.delayDisplay("Moving the transform: pyramid follows")
        matrix.SetElement(0, 3, 250.0)
        transformNode.SetMatrixTransformToParent(matrix)
        logic.updatePatch()
        np.testing.assert_allclose(logic.apex, [250, 0, 0], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(logic.corners.mean(axis=0) - [250, 0, 0]), 100.0, atol=1e-3)
        tip = logic.pyramidNode.GetPolyData().GetPoint(4)
        np.testing.assert_allclose(tip, [250, 0, 0], atol=1e-6)
        handle = [0.0, 0.0, 0.0]
        logic.centerHandle.GetNthControlPointPositionWorld(0, handle)
        np.testing.assert_allclose(handle, logic.corners.mean(axis=0), atol=1e-6)

        self.delayDisplay("Editing the target mesh: apex follows")
        source = vtk.vtkSphereSource()
        source.SetRadius(5.0)
        source.SetCenter(0, 0, 20)
        source.Update()
        target.SetAndObservePolyData(source.GetOutput())
        logic.updatePatch()
        np.testing.assert_allclose(logic.apex, [250, 0, 20], atol=1e-3)
        logic.removeSelection()

    def test_ObstaclesSavedInParameterNode(self):
        self.delayDisplay("Obstacle list round trip")
        a = self._sphereModel("obstacleA", 5.0, (0, 0, 0))
        b = self._sphereModel("obstacleB", 5.0, (10, 0, 0))
        p = PyramidSelectionLogic().getParameterNode()
        p.obstacles = [a, b]
        again = PyramidSelectionLogic().getParameterNode()
        self.assertEqual([n.GetID() for n in again.obstacles], [a.GetID(), b.GetID()])
        p.obstacles = []
        self.assertEqual(len(again.obstacles), 0)
        self.delayDisplay("Test passed")

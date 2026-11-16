"""Slicer-independent building blocks of the PyramidSelection module.

- ``geometry``: pure NumPy geometry (tangent frame, footprint rectangle, edge
  resizing, central-projection footprint test).
- ``surface``: VTK helpers around the surface mesh (snapping, normals, patch
  extraction).
- ``risk``: pluggable safety scoring of candidate entry points.
"""

cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(PyramidSelection)

#-----------------------------------------------------------------------------
# Extension meta-information
# TODO: update the URLs once the repository is public on GitHub.
set(EXTENSION_HOMEPAGE "https://github.com/NawarKOT/SlicerPyramidSelection")
set(EXTENSION_CATEGORY "Surface Models")
set(EXTENSION_CONTRIBUTORS "Nawar Koteish")
set(EXTENSION_DESCRIPTION "Interactive selection of a region on a surface model, linked to a target by a pyramid-shaped corridor, with optional obstacle-clearance coloring.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/NawarKOT/SlicerPyramidSelection/main/PyramidSelection.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/NawarKOT/SlicerPyramidSelection/main/Screenshots/demo.gif")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PyramidSelection)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

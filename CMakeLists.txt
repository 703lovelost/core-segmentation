cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(CoreSeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/CoreSeg")
set(EXTENSION_CONTRIBUTORS "Aleksey Spirkin (Novosibirsk State University), Lev Moryakin (Novosibirsk State University)")
set(EXTENSION_DESCRIPTION "Segformer-based core image segmentation module for 3D Slicer.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/CoreSeg.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/CoreSeg/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(segment_core)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

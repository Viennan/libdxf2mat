cmake_minimum_required (VERSION 3.8)

project ("libdxf2mat")

# You might need to set "OpenCV_DIR" manually.
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")

include_directories("./3rd/dxflib")
include_directories("./3rd/tinyspline")
include_directories("./include")

add_subdirectory("./3rd")
add_subdirectory("./src")

set(Dxf2Mat_LIBS "libdxf2mat")

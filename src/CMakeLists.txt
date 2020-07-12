add_library(libdxf2mat STATIC libdxf2mat.cpp)
target_link_libraries(libdxf2mat dxflib tinyspline ${OpenCV_LIBS})
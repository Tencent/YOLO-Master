# Standalone tempo bench - independent of cpp/ so nothing clashes with the
# main runner build. Point NCNN_INSTALL at an ncnn install prefix:
#   x86 dev box:  third_party/ncnn-20260526-ubuntu-2204-shared
#   Jetson:       third_party/ncnn-aarch64-20260526   (from jetson/24_build_ncnn.sh)
cmake_minimum_required(VERSION 3.16)
project(ncnn_tempo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if(NOT NCNN_INSTALL)
  message(FATAL_ERROR "pass -DNCNN_INSTALL=<ncnn install prefix>")
endif()

add_executable(ncnn_tempo bench.cpp)
target_include_directories(ncnn_tempo PRIVATE ${NCNN_INSTALL}/include/ncnn)
find_library(NCNN_LIB ncnn PATHS ${NCNN_INSTALL}/lib NO_DEFAULT_PATH)
target_link_libraries(ncnn_tempo ${NCNN_LIB})
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ncnn_tempo OpenMP::OpenMP_CXX)
endif()

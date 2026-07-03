# CMake build for YOLO-Master-EsMoE edge inference (C++ / ONNX Runtime).
# Arch-agnostic: builds identically on Linux x86_64, Linux ARM64 (Jetson), and Windows.
#
# Typical build (Linux, after running scripts/setup_ort.sh):
#   cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=$(pwd)/third_party/onnxruntime
#   cmake --build build -j
#
# ARM64 cross / native (provide aarch64 OpenCV + ORT builds, or use Dockerfile.arm64):
#   cmake -S . -B build-arm64 -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake \
#         -DONNXRUNTIME_ROOT_DIR=/path/to/aarch64/onnxruntime
cmake_minimum_required(VERSION 3.18)
project(yolo_edge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(USE_CUDA "Append CUDA execution provider (needs ORT built with CUDA)" OFF)
option(BUILD_OBJECTS_ONLY "Compile objects only (skip link — used to prove cross-arch portability)" OFF)

# ---- OpenCV (config mode first — works with conda-forge / vcpkg / system) ----
find_package(OpenCV CONFIG QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif()
message(STATUS "OpenCV ${OpenCV_VERSION} -> ${OpenCV_INCLUDE_DIRS}")

# ---- ONNX Runtime ----
if(NOT DEFINED ONNXRUNTIME_ROOT_DIR)
  set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
endif()
set(ORT_INCLUDE "${ONNXRUNTIME_ROOT_DIR}/include")
if(MSVC)
  set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")
  set(ORT_LIB onnxruntime)
else()
  set(ORT_LIB_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")
  set(ORT_LIB onnxruntime)
endif()
if(NOT EXISTS "${ORT_INCLUDE}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime headers not found at ${ORT_INCLUDE}. "
    "Run cpp/scripts/setup_ort.sh or pass -DONNXRUNTIME_ROOT_DIR=/path/to/ort")
endif()

add_executable(yolo_edge src/main.cpp)
target_include_directories(yolo_edge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${ORT_INCLUDE} ${OpenCV_INCLUDE_DIRS})

if(USE_CUDA)
  target_compile_definitions(yolo_edge PRIVATE USE_CUDA=1)
endif()

if(BUILD_OBJECTS_ONLY)
  # Compile to objects without linking — proves the sources are portable to the
  # target arch (used by the aarch64 cross-compile check in scripts/cross_compile_check.sh).
  set_target_properties(yolo_edge PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_custom_target(yolo_objects ALL
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
            -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${ORT_INCLUDE} -I${OpenCV_INCLUDE_DIRS}
            -o ${CMAKE_CURRENT_BINARY_DIR}/main.o
    COMMENT "Cross-compiling main.cpp -> ${CMAKE_CURRENT_BINARY_DIR}/main.o")
else()
  target_link_directories(yolo_edge PRIVATE ${ORT_LIB_DIR})
  if(MSVC)
    target_link_libraries(yolo_edge PRIVATE ${OpenCV_LIBS} ${ORT_LIB})
  else()
    target_link_libraries(yolo_edge PRIVATE ${OpenCV_LIBS} ${ORT_LIB} pthread)
  endif()
  # make the shared ORT lib discoverable at runtime
  set_target_properties(yolo_edge PROPERTIES BUILD_RPATH "${ORT_LIB_DIR}")
endif()

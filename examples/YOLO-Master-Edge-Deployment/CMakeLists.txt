cmake_minimum_required(VERSION 3.12)
project(yolo_master_edge_deployment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

option(WITH_ONNXRUNTIME "Build ONNX Runtime backend" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime package root")
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime include directory")
set(ONNXRUNTIME_LIB "" CACHE FILEPATH "Path to ONNX Runtime library")

add_executable(
    yolo_master_edge_benchmark
    cpp/edge_benchmark.cpp
    cpp/preprocess.cpp
    cpp/postprocess.cpp
    cpp/backends/backend_factory.cpp
    cpp/backends/onnx_backend.cpp
    cpp/backends/ncnn_backend.cpp
    cpp/backends/mnn_backend.cpp
)

target_include_directories(
    yolo_master_edge_benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/backends
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    yolo_master_edge_benchmark
    PRIVATE
    ${OpenCV_LIBS}
)

if(WITH_ONNXRUNTIME)
    if(NOT ONNXRUNTIME_ROOT AND (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB))
        message(FATAL_ERROR "WITH_ONNXRUNTIME=ON requires either -DONNXRUNTIME_ROOT=/path/to/onnxruntime or both -DONNXRUNTIME_INCLUDE_DIR=... and -DONNXRUNTIME_LIB=...")
    endif()
    target_compile_definitions(yolo_master_edge_benchmark PRIVATE WITH_ONNXRUNTIME=1)
    if(ONNXRUNTIME_INCLUDE_DIR)
        set(ONNXRUNTIME_INCLUDE_DIR_RESOLVED ${ONNXRUNTIME_INCLUDE_DIR})
    else()
        find_path(
            ONNXRUNTIME_INCLUDE_DIR_RESOLVED
            onnxruntime_cxx_api.h
            HINTS
                ${ONNXRUNTIME_ROOT}/include
                ${ONNXRUNTIME_ROOT}/include/onnxruntime
                ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
                /opt/homebrew/include
                /opt/homebrew/include/onnxruntime
                /opt/homebrew/include/onnxruntime/core/session
                /usr/local/include
                /usr/local/include/onnxruntime
                /usr/local/include/onnxruntime/core/session
            REQUIRED
        )
    endif()
    if(ONNXRUNTIME_LIB)
        set(ONNXRUNTIME_LIB_RESOLVED ${ONNXRUNTIME_LIB})
    else()
        find_library(
            ONNXRUNTIME_LIB_RESOLVED
            NAMES onnxruntime libonnxruntime
            HINTS
                ${ONNXRUNTIME_ROOT}/lib
                ${ONNXRUNTIME_ROOT}/lib64
                /opt/homebrew/lib
                /usr/local/lib
            REQUIRED
        )
    endif()
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR_RESOLVED}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB_RESOLVED}")
    target_include_directories(
        yolo_master_edge_benchmark
        PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR_RESOLVED}
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
    )
    target_link_libraries(yolo_master_edge_benchmark PRIVATE ${ONNXRUNTIME_LIB_RESOLVED})
endif()

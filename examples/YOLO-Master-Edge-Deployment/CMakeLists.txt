cmake_minimum_required(VERSION 3.12)
project(yolo_master_edge_deployment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

option(WITH_ONNXRUNTIME "Build ONNX Runtime backend" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime package root")
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime include directory")
set(ONNXRUNTIME_LIB "" CACHE FILEPATH "Path to ONNX Runtime library")
option(WITH_NCNN "Build NCNN backend" OFF)
set(NCNN_ROOT "" CACHE PATH "Path to NCNN package root")
set(NCNN_INCLUDE_DIR "" CACHE PATH "Path to NCNN include directory")
set(NCNN_LIB "" CACHE FILEPATH "Path to NCNN library")
option(WITH_MNN "Build MNN backend" OFF)
set(MNN_ROOT "" CACHE PATH "Path to MNN package root")
set(MNN_INCLUDE_DIR "" CACHE PATH "Path to MNN include directory")
set(MNN_LIB "" CACHE FILEPATH "Path to MNN library")

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

if(WITH_NCNN)
    if(NOT NCNN_ROOT AND (NOT NCNN_INCLUDE_DIR OR NOT NCNN_LIB))
        message(FATAL_ERROR "WITH_NCNN=ON requires either -DNCNN_ROOT=/path/to/ncnn or both -DNCNN_INCLUDE_DIR=... and -DNCNN_LIB=...")
    endif()
    target_compile_definitions(yolo_master_edge_benchmark PRIVATE WITH_NCNN=1)
    if(NCNN_INCLUDE_DIR)
        set(NCNN_INCLUDE_DIR_RESOLVED ${NCNN_INCLUDE_DIR})
    else()
        find_path(
            NCNN_INCLUDE_DIR_RESOLVED
            net.h
            HINTS
                ${NCNN_ROOT}/include
                ${NCNN_ROOT}/include/ncnn
                /opt/homebrew/include
                /opt/homebrew/include/ncnn
                /usr/local/include
                /usr/local/include/ncnn
            REQUIRED
        )
    endif()
    if(NCNN_LIB)
        set(NCNN_LIB_RESOLVED ${NCNN_LIB})
    else()
        find_library(
            NCNN_LIB_RESOLVED
            NAMES ncnn libncnn
            HINTS
                ${NCNN_ROOT}/lib
                ${NCNN_ROOT}/lib64
                /opt/homebrew/lib
                /usr/local/lib
            REQUIRED
        )
    endif()
    message(STATUS "NCNN include: ${NCNN_INCLUDE_DIR_RESOLVED}")
    message(STATUS "NCNN library: ${NCNN_LIB_RESOLVED}")
    target_include_directories(
        yolo_master_edge_benchmark
        PRIVATE
        ${NCNN_INCLUDE_DIR_RESOLVED}
        ${NCNN_ROOT}/include
        ${NCNN_ROOT}/include/ncnn
    )
    target_link_libraries(yolo_master_edge_benchmark PRIVATE ${NCNN_LIB_RESOLVED})
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(yolo_master_edge_benchmark PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if(WITH_MNN)
    target_compile_definitions(yolo_master_edge_benchmark PRIVATE WITH_MNN=1)
    if(MNN_INCLUDE_DIR)
        set(MNN_INCLUDE_DIR_RESOLVED ${MNN_INCLUDE_DIR})
    else()
        find_path(
            MNN_INCLUDE_DIR_RESOLVED
            Interpreter.hpp
            HINTS
                ${MNN_ROOT}/include
                /opt/homebrew/include
                /usr/local/include
            REQUIRED
        )
    endif()
    if(MNN_LIB)
        set(MNN_LIB_RESOLVED ${MNN_LIB})
    else()
        find_library(
            MNN_LIB_RESOLVED
            NAMES MNN libMNN
            HINTS
                ${MNN_ROOT}/lib
                ${MNN_ROOT}/lib64
                /opt/homebrew/lib
                /usr/local/lib
            REQUIRED
        )
    endif()
    message(STATUS "MNN include: ${MNN_INCLUDE_DIR_RESOLVED}")
    message(STATUS "MNN library: ${MNN_LIB_RESOLVED}")
    target_include_directories(yolo_master_edge_benchmark PRIVATE ${MNN_INCLUDE_DIR_RESOLVED})
    target_link_libraries(yolo_master_edge_benchmark PRIVATE ${MNN_LIB_RESOLVED})
endif()

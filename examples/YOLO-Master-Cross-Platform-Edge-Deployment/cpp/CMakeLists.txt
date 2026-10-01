cmake_minimum_required(VERSION 3.16)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" YM_VERSION LIMIT_COUNT 1)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/server/SERVER_VERSION" YM_SERVER_VERSION LIMIT_COUNT 1)
project(yolomaster_edge VERSION ${YM_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# PORTABLE=ON -> slim relocatable bundle: link only the OpenCV modules headless
# image inference needs (drops videoio/highgui -> no FFmpeg/GStreamer/X11 closure).
option(PORTABLE "Slim relocatable build: image inference only, minimal OpenCV deps" OFF)
# BoT-SORT camera motion compensation needs OpenCV video (LK optical flow) + calib3d
# (estimateAffinePartial2D); OFF keeps ncnn-only mobile builds lean (tracking then runs without GMC).
option(USE_GMC "BoT-SORT camera motion compensation (OpenCV video + calib3d)" ON)
# GPU preprocessing (letterbox + normalize + NCHW as one CUDA kernel feeding the TensorRT / ORT-CUDA
# input tensor) and TensorRT CUDA graphs. Needs nvcc; CPU-only builds leave it OFF.
option(USE_CUDA_PREPROC "CUDA preprocessing kernel + IoBinding (needs nvcc)" OFF)
# image I/O via stb + hand-written NMS -> only need core+imgproc (no imgcodecs->GDAL,
# no dnn->protobuf). Full build adds videoio for cv::VideoCapture (--source video).
if(PORTABLE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)   # -DOpenCV_DIR=... on Windows
  if(USE_GMC)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio video calib3d)
  endif()
endif()

# SDK roots - override per platform:
#   Linux : cmake -DONNXRUNTIME_ROOT=/path -DNCNN_ROOT=/path ..
#   Windows: cmake -DONNXRUNTIME_ROOT=C:/dev/onnxruntime -DNCNN_ROOT=C:/dev/ncnn -DOpenCV_DIR=C:/dev/opencv/build ..
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime" CACHE PATH "ONNXRuntime root")
set(NCNN_ROOT        "${CMAKE_SOURCE_DIR}/../third_party/ncnn"        CACHE PATH "ncnn root")
set(MNN_ROOT         "${CMAKE_SOURCE_DIR}/../third_party/mnn-src"     CACHE PATH "MNN root (include/ + lib/ or build/)")

option(USE_ORT  "Build ONNXRuntime backend" ON)
option(USE_NCNN "Build ncnn backend"        ON)
option(USE_MNN  "Build MNN backend"         ON)
option(USE_TRT  "Build TensorRT backend (GPU .engine inference; Jetson/CUDA)" OFF)
option(BUILD_SERVER "Build the HTTP/WebSocket inference API server (cpp/server)" OFF)

# The runtime core is one static library (yolomaster_core); the CLI, the API server and
# the GUI/mobile builds are thin consumers of it. Keep main.cpp out of the core.
set(SRC  src/common.cpp src/slicing.cpp src/annotate_writers.cpp src/annotate_export.cpp
         src/stb_impl.cpp src/map_metrics.cpp src/bench.cpp src/tracker.cpp)
set(INCS include third_party ${OpenCV_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS})

# The portable core (in-process mAP, BoT-SORT / ByteTrack, bench statistics, C shim) has no
# OpenCV or JSON dependency and is shared verbatim with the Swift package: it lives under
# mac/Sources/YOLOMasterCore because SwiftPM needs its targets inside the package, and CMake
# reaches across. yolomaster_core links it PUBLIC; yolomaster_score links it alone, which keeps
# the extraction honest (an OpenCV include there fails that link). The Android runtime CMake
# adds the same directory when it next picks up these features.
set(YM_CCORE_DIR "${CMAKE_SOURCE_DIR}/../mac/Sources/YOLOMasterCore")
file(GLOB YM_CCORE_SRC "${YM_CCORE_DIR}/*.cpp")
add_library(yolomaster_ccore STATIC ${YM_CCORE_SRC})
target_include_directories(yolomaster_ccore PUBLIC "${YM_CCORE_DIR}" "${YM_CCORE_DIR}/include")
set_target_properties(yolomaster_ccore PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(DEFS "")
if(USE_GMC AND NOT PORTABLE)
  list(APPEND DEFS HAVE_GMC)
endif()
# version / provenance stamps (bench JSON, --version, GET /)
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE YM_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT YM_GIT_COMMIT)
  set(YM_GIT_COMMIT "unknown")
endif()
list(APPEND DEFS "YM_VERSION=\"${YM_VERSION}\"" "YM_GIT_COMMIT=\"${YM_GIT_COMMIT}\"")
if(NOT PORTABLE)
  list(APPEND DEFS HAVE_VIDEOIO)   # cv::VideoCapture (--source video) needs opencv_videoio
endif()
set(RPATH "")
set(RUNTIME_DLLS "")   # Windows: DLLs to copy next to the .exe

if(USE_ORT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  list(APPEND SRC  src/ort_backend.cpp)
  list(APPEND INCS ${ONNXRUNTIME_ROOT}/include)
  if(WIN32)
    list(APPEND LIBS ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll" "${ONNXRUNTIME_ROOT}/bin/*.dll")
    list(APPEND RUNTIME_DLLS ${ORT_DLLS})
  elseif(APPLE)
    # macOS: ORT ships a .dylib with the CoreML EP built in; link the frameworks it pulls
    find_library(ORT_DYLIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
    list(APPEND LIBS ${ORT_DYLIB} "-framework Foundation" "-framework CoreML")
    list(APPEND RPATH ${ONNXRUNTIME_ROOT}/lib)
  else()
    list(APPEND LIBS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    list(APPEND RPATH ${ONNXRUNTIME_ROOT}/lib)
  endif()
  list(APPEND DEFS USE_ORT)
  message(STATUS "ONNXRuntime backend: ON (${ONNXRUNTIME_ROOT})")
else()
  message(WARNING "ONNXRuntime backend: OFF (not found at ${ONNXRUNTIME_ROOT})")
endif()

if(USE_NCNN AND EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
  list(APPEND SRC  src/ncnn_backend.cpp)
  list(APPEND INCS ${NCNN_ROOT}/include ${NCNN_ROOT}/include/ncnn)
  find_library(NCNN_LIB ncnn PATHS ${NCNN_ROOT}/lib NO_DEFAULT_PATH)
  list(APPEND LIBS ${NCNN_LIB})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    list(APPEND LIBS OpenMP::OpenMP_CXX)
  endif()
  if(WIN32)
    file(GLOB NCNN_DLLS "${NCNN_ROOT}/bin/*.dll" "${NCNN_ROOT}/lib/*.dll")
    list(APPEND RUNTIME_DLLS ${NCNN_DLLS})
  else()
    list(APPEND RPATH ${NCNN_ROOT}/lib)
  endif()
  list(APPEND DEFS USE_NCNN)
  message(STATUS "ncnn backend: ON (${NCNN_ROOT})")
else()
  message(WARNING "ncnn backend: OFF (not found at ${NCNN_ROOT})")
endif()

if(USE_MNN AND EXISTS "${MNN_ROOT}/include/MNN/Interpreter.hpp")
  list(APPEND SRC  src/mnn_backend.cpp)
  list(APPEND INCS ${MNN_ROOT}/include)
  find_library(MNN_LIB MNN PATHS ${MNN_ROOT}/lib ${MNN_ROOT}/build ${MNN_ROOT}/build/Release NO_DEFAULT_PATH)
  if(NOT MNN_LIB)
    message(FATAL_ERROR "USE_MNN=ON but libMNN not found under ${MNN_ROOT}/{lib,build}")
  endif()
  list(APPEND LIBS ${MNN_LIB})
  if(WIN32)
    file(GLOB MNN_DLLS "${MNN_ROOT}/lib/*.dll" "${MNN_ROOT}/bin/*.dll" "${MNN_ROOT}/build/*.dll" "${MNN_ROOT}/build/Release/*.dll")
    list(APPEND RUNTIME_DLLS ${MNN_DLLS})
  else()
    list(APPEND RPATH ${MNN_ROOT}/lib ${MNN_ROOT}/build)
  endif()
  list(APPEND DEFS USE_MNN)
  message(STATUS "MNN backend: ON (${MNN_ROOT})")
else()
  message(WARNING "MNN backend: OFF (not found at ${MNN_ROOT})")
endif()

if(USE_TRT OR USE_CUDA_PREPROC)
  # CUDA runtime (JetPack: /usr/local/cuda; x86: /usr/local/cuda-*)
  file(GLOB _CUDA_DIRS /usr/local/cuda /usr/local/cuda-*)
  find_path(CUDA_INC cuda_runtime_api.h PATHS ${_CUDA_DIRS} PATH_SUFFIXES include)
  find_library(CUDART_LIB cudart PATHS ${_CUDA_DIRS} PATH_SUFFIXES lib64 lib/aarch64-linux-gnu)
endif()
if(USE_CUDA_PREPROC)
  if(NOT CUDA_INC OR NOT CUDART_LIB)
    message(FATAL_ERROR "USE_CUDA_PREPROC=ON but the CUDA runtime was not found under /usr/local/cuda*")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 120)   # L4 / L40S, RTX PRO (Blackwell); Jetson passes its own
  endif()
  list(APPEND SRC  src/cuda/preprocess.cu)
  list(APPEND INCS ${CUDA_INC})
  list(APPEND LIBS ${CUDART_LIB})
  list(APPEND DEFS HAVE_CUDA_PREPROC)
  message(STATUS "CUDA preprocessing: ON (arch ${CMAKE_CUDA_ARCHITECTURES})")
endif()

if(USE_TRT)
  # TensorRT + CUDA from JetPack (headers under /usr/include/aarch64-linux-gnu, CUDA under /usr/local/cuda)
  find_path(TRT_INC NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include ${TENSORRT_ROOT}/include)
  find_library(TRT_LIB nvinfer PATHS /usr/lib/aarch64-linux-gnu ${TENSORRT_ROOT}/lib)
    find_library(TRT_ONNX_LIB nvonnxparser PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu ${TENSORRT_ROOT}/lib)
  if(TRT_INC AND TRT_LIB AND CUDA_INC AND CUDART_LIB)
    list(APPEND SRC  src/trt_backend.cpp)
    list(APPEND INCS ${TRT_INC} ${CUDA_INC})
    list(APPEND LIBS ${TRT_LIB} ${CUDART_LIB})
    list(APPEND DEFS USE_TRT)
    if(TRT_ONNX_LIB)
      # nvonnxparser enables building engines from .onnx at load time (fp32/fp16, cached)
      list(APPEND LIBS ${TRT_ONNX_LIB})
      list(APPEND DEFS USE_TRT_ONNXPARSER)
      message(STATUS "TensorRT onnx parser: ON (${TRT_ONNX_LIB})")
    endif()
    message(STATUS "TensorRT backend: ON (${TRT_LIB} + ${CUDART_LIB})")
  else()
    message(FATAL_ERROR "USE_TRT=ON but TensorRT/CUDA not found (TRT_INC=${TRT_INC} TRT_LIB=${TRT_LIB} CUDA_INC=${CUDA_INC} CUDART_LIB=${CUDART_LIB})")
  endif()
endif()

add_library(yolomaster_core STATIC ${SRC})
target_include_directories(yolomaster_core PUBLIC ${INCS})
target_link_libraries(yolomaster_core PUBLIC ${LIBS} yolomaster_ccore)
string(REPLACE ";" "," _YM_FLAGS "${DEFS};${CMAKE_BUILD_TYPE}")
string(REGEX REPLACE "YM_[A-Z_]+=\"[^\"]*\",?" "" _YM_FLAGS "${_YM_FLAGS}")
list(APPEND DEFS "YM_BUILD_FLAGS=\"${_YM_FLAGS}\"")
target_compile_definitions(yolomaster_core PUBLIC ${DEFS})
set_target_properties(yolomaster_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(yolomaster_score src/score_main.cpp src/stb_impl.cpp)   # portable core only, no OpenCV
target_include_directories(yolomaster_score PRIVATE third_party)
target_link_libraries(yolomaster_score PRIVATE yolomaster_ccore)
if(USE_CUDA_PREPROC)
  add_executable(preproc_parity tests/preproc_parity.cpp)
  target_link_libraries(preproc_parity PRIVATE yolomaster_core)
endif()
add_executable(yolomaster_edge src/main.cpp)
target_link_libraries(yolomaster_edge PRIVATE yolomaster_core)

if(BUILD_SERVER)
  add_subdirectory(server)
endif()

install(TARGETS yolomaster_edge RUNTIME DESTINATION bin)

if(WIN32)
  # bundle the MSVC runtime (VCRUNTIME140.dll etc.) so the exe needs no VC++ Redist on targets
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
  include(InstallRequiredSystemLibraries)
  list(APPEND RUNTIME_DLLS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
  # copy backend DLLs next to the exe; OpenCV's opencv_world DLL: add its bin dir to PATH
  # (C:/dev/opencv/build/x64/vc16/bin) or it is auto-copied below if found.
  # OpenCV DLL location depends on whether OpenCV_DIR = build/ or build/x64/vcNN/lib
  file(GLOB OCV_DLLS
    "${OpenCV_DIR}/x64/*/bin/opencv_world*.dll"   # OpenCV_DIR = <opencv>/build
    "${OpenCV_DIR}/../bin/opencv_world*.dll"       # OpenCV_DIR = <opencv>/build/x64/vcNN/lib
    "${OpenCV_DIR}/bin/opencv_world*.dll")
  list(FILTER OCV_DLLS EXCLUDE REGEX "[0-9]d\\.dll$")   # drop the ~60MB debug opencv_world*d.dll
  foreach(dll ${RUNTIME_DLLS} ${OCV_DLLS})
    add_custom_command(TARGET yolomaster_edge POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:yolomaster_edge>"
      COMMENT "copying ${dll}")
  endforeach()
else()
  set_target_properties(yolomaster_edge PROPERTIES BUILD_RPATH "${RPATH}")
  if(BUILD_SERVER)
    set_target_properties(yolomaster_server PROPERTIES BUILD_RPATH "${RPATH}")
  endif()
endif()

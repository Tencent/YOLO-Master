cmake_minimum_required(VERSION 3.16)
project(yolomaster_edge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# PORTABLE=ON -> slim relocatable bundle: link only the OpenCV modules headless
# image inference needs (drops videoio/highgui -> no FFmpeg/GStreamer/X11 closure).
option(PORTABLE "Slim relocatable build: image inference only, minimal OpenCV deps" OFF)
# image I/O via stb + hand-written NMS -> only need core+imgproc (no imgcodecs->GDAL,
# no dnn->protobuf). Full build adds videoio for cv::VideoCapture (--source video).
if(PORTABLE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)   # -DOpenCV_DIR=... on Windows
endif()

# SDK roots - override per platform:
#   Linux : cmake -DONNXRUNTIME_ROOT=/path -DNCNN_ROOT=/path ..
#   Windows: cmake -DONNXRUNTIME_ROOT=C:/dev/onnxruntime -DNCNN_ROOT=C:/dev/ncnn -DOpenCV_DIR=C:/dev/opencv/build ..
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime" CACHE PATH "ONNXRuntime root")
set(NCNN_ROOT        "${CMAKE_SOURCE_DIR}/../third_party/ncnn"        CACHE PATH "ncnn root")
set(MNN_ROOT         "${CMAKE_SOURCE_DIR}/../third_party/mnn-src"     CACHE PATH "MNN root (include/ + lib/ or build/)")
# Native TensorRT is normally supplied by JetPack under /usr, but keeping the
# roots explicit makes the same CMake invocation usable with an unpacked SDK
# or an aarch64 sysroot.  Empty values preserve the platform defaults below.
set(TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root (include/ and lib/)")
set(CUDA_ROOT     "" CACHE PATH "CUDA SDK root (include/ and lib64/)")

option(USE_ORT  "Build ONNXRuntime backend" ON)
option(USE_NCNN "Build ncnn backend"        ON)
option(USE_MNN  "Build MNN backend"         ON)
option(USE_TRT  "Build native TensorRT 10 backend (GPU .engine; Jetson/CUDA)" OFF)
# A dependency-light CLI build is useful for checking the argument/CSV
# contract, so it remains allowed by default.  Acceptance and release builds
# should set the corresponding REQUIRE_* option(s) to make missing SDKs a
# configuration error instead of silently producing a partial binary.
option(ALLOW_NO_BACKENDS "Allow a diagnostic build without inference SDKs" ON)
option(REQUIRE_ORT  "Fail configuration when ONNX Runtime is unavailable" OFF)
option(REQUIRE_NCNN "Fail configuration when NCNN is unavailable" OFF)
option(REQUIRE_MNN  "Fail configuration when MNN is unavailable" OFF)

set(SRC  src/common.cpp src/slicing.cpp src/annotate_writers.cpp src/annotate_export.cpp
         src/main.cpp src/stb_impl.cpp)
set(INCS include third_party ${OpenCV_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS})
set(DEFS "")
if(NOT PORTABLE)
  list(APPEND DEFS HAVE_VIDEOIO)   # cv::VideoCapture (--source video) needs opencv_videoio
endif()
if(WIN32)
  # Keep Win32's min/max macros from colliding with std::min/std::max in the
  # shared C++ runtime and third-party headers.
  list(APPEND DEFS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
set(RPATH "")
set(RUNTIME_DLLS "")   # Windows: DLLs to copy next to the .exe
set(ORT_ENABLED OFF)
set(NCNN_ENABLED OFF)
set(MNN_ENABLED OFF)

if(USE_ORT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  set(ORT_LINK_LIB "")
  if(WIN32)
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
      set(ORT_LINK_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
      file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll" "${ONNXRUNTIME_ROOT}/bin/*.dll")
      list(APPEND RUNTIME_DLLS ${ORT_DLLS})
    endif()
  elseif(APPLE)
    # macOS: ORT ships a .dylib with the CoreML EP built in; link the frameworks it pulls.
    find_library(ORT_DYLIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
    set(ORT_LINK_LIB "${ORT_DYLIB}")
  else()
    # Release archives commonly ship only a versioned SONAME
    # (libonnxruntime.so.1.18.1), without an unversioned development symlink.
    file(GLOB ORT_SOS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
    list(SORT ORT_SOS)
    list(LENGTH ORT_SOS ORT_SO_COUNT)
    if(ORT_SO_COUNT GREATER 0)
      list(GET ORT_SOS 0 ORT_LINK_LIB)
    endif()
  endif()
  if(ORT_LINK_LIB)
    list(APPEND SRC src/ort_backend.cpp)
    list(APPEND INCS ${ONNXRUNTIME_ROOT}/include)
    list(APPEND LIBS ${ORT_LINK_LIB})
    if(APPLE)
      list(APPEND LIBS "-framework Foundation" "-framework CoreML")
    endif()
    list(APPEND RPATH ${ONNXRUNTIME_ROOT}/lib)
    list(APPEND DEFS USE_ORT)
    set(ORT_ENABLED ON)
    message(STATUS "ONNXRuntime backend: ON (${ONNXRUNTIME_ROOT})")
  elseif(REQUIRE_ORT)
    message(FATAL_ERROR "REQUIRE_ORT=ON but ONNXRuntime library was not found under ${ONNXRUNTIME_ROOT}")
  else()
    message(WARNING "ONNXRuntime backend: OFF (headers found but library is missing under ${ONNXRUNTIME_ROOT})")
  endif()
else()
  if(REQUIRE_ORT)
    message(FATAL_ERROR "REQUIRE_ORT=ON but ONNXRuntime was not found at ${ONNXRUNTIME_ROOT}")
  endif()
  message(WARNING "ONNXRuntime backend: OFF (not found at ${ONNXRUNTIME_ROOT})")
endif()

if(USE_NCNN AND EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
  # Accept both an installed SDK (lib/) and an uninstalled source build
  # (build/src/).  The latter is the layout produced by the standard NCNN
  # CMake instructions and is useful on an offline Ubuntu host.
  find_library(NCNN_LIB ncnn PATHS
    ${NCNN_ROOT}/lib
    ${NCNN_ROOT}/lib64
    ${NCNN_ROOT}/build/src
    ${NCNN_ROOT}/build
    NO_DEFAULT_PATH)
  if(NOT NCNN_LIB)
    if(REQUIRE_NCNN)
      message(FATAL_ERROR "REQUIRE_NCNN=ON but libncnn was not found under ${NCNN_ROOT}/lib")
    endif()
    message(WARNING "ncnn headers found but libncnn is missing; backend disabled")
  else()
    list(APPEND SRC src/ncnn_backend.cpp)
    list(APPEND INCS ${NCNN_ROOT}/include ${NCNN_ROOT}/include/ncnn)
    list(APPEND LIBS ${NCNN_LIB})
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
      list(APPEND LIBS OpenMP::OpenMP_CXX)
    endif()
    if(UNIX AND NOT APPLE)
      # Static NCNN archives do not carry their POSIX runtime dependencies.
      # These are harmless for shared builds and avoid platform-specific
      # undefined references (pthread/dlopen/libm) at link time.
      find_package(Threads REQUIRED)
      list(APPEND LIBS Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
    if(WIN32)
      file(GLOB NCNN_DLLS "${NCNN_ROOT}/bin/*.dll" "${NCNN_ROOT}/lib/*.dll")
      list(APPEND RUNTIME_DLLS ${NCNN_DLLS})
    else()
      get_filename_component(NCNN_LIB_DIR "${NCNN_LIB}" DIRECTORY)
      list(APPEND RPATH ${NCNN_LIB_DIR} ${NCNN_ROOT}/lib ${NCNN_ROOT}/lib64
           ${NCNN_ROOT}/build/src ${NCNN_ROOT}/build)
    endif()
    list(APPEND DEFS USE_NCNN)
    set(NCNN_ENABLED ON)
    message(STATUS "ncnn backend: ON (${NCNN_ROOT})")
  endif()
else()
  if(REQUIRE_NCNN)
    message(FATAL_ERROR "REQUIRE_NCNN=ON but ncnn was not found at ${NCNN_ROOT}")
  endif()
  message(WARNING "ncnn backend: OFF (not found at ${NCNN_ROOT})")
endif()

if(USE_MNN AND EXISTS "${MNN_ROOT}/include/MNN/Interpreter.hpp")
  find_library(MNN_LIB MNN PATHS
    ${MNN_ROOT}/lib
    ${MNN_ROOT}/lib64
    ${MNN_ROOT}/build
    ${MNN_ROOT}/build/Release
    ${MNN_ROOT}/build/Debug
    NO_DEFAULT_PATH)
  if(NOT MNN_LIB)
    if(REQUIRE_MNN)
      message(FATAL_ERROR "REQUIRE_MNN=ON but libMNN not found under ${MNN_ROOT}/{lib,build}")
    endif()
    message(WARNING "MNN headers found but libMNN is missing; backend disabled")
  else()
    list(APPEND SRC src/mnn_backend.cpp)
    list(APPEND INCS ${MNN_ROOT}/include)
    list(APPEND LIBS ${MNN_LIB})
    if(UNIX AND NOT APPLE)
      # Static MNN builds do not propagate their POSIX runtime dependencies.
      # Keep shared-library builds unchanged while making libMNN.a linkable.
      find_package(Threads REQUIRED)
      list(APPEND LIBS Threads::Threads ${CMAKE_DL_LIBS} m)
    endif()
    if(WIN32)
      file(GLOB MNN_DLLS "${MNN_ROOT}/lib/*.dll" "${MNN_ROOT}/bin/*.dll" "${MNN_ROOT}/build/*.dll" "${MNN_ROOT}/build/Release/*.dll")
      list(APPEND RUNTIME_DLLS ${MNN_DLLS})
    else()
      get_filename_component(MNN_LIB_DIR "${MNN_LIB}" DIRECTORY)
      list(APPEND RPATH ${MNN_LIB_DIR} ${MNN_ROOT}/lib ${MNN_ROOT}/lib64
           ${MNN_ROOT}/build ${MNN_ROOT}/build/Release ${MNN_ROOT}/build/Debug)
    endif()
    list(APPEND DEFS USE_MNN)
    set(MNN_ENABLED ON)
    message(STATUS "MNN backend: ON (${MNN_ROOT})")
  endif()
else()
  if(REQUIRE_MNN)
    message(FATAL_ERROR "REQUIRE_MNN=ON but MNN was not found at ${MNN_ROOT}")
  endif()
  message(WARNING "MNN backend: OFF (not found at ${MNN_ROOT})")
endif()

if(USE_TRT)
  # The native runner uses TensorRT 10's named-I/O API.  TensorRT 8 has a
  # separate binding API and is intentionally routed through ORT + TRT-EP
  # instead of being accepted here and failing later during compilation.
  # HINTS are searched first so a caller-provided SDK/sysroot wins over a
  # host installation.  The explicit PATHS retain the JetPack defaults when
  # both cache variables are left empty.
  find_path(TRT_INC NvInfer.h
    HINTS "${TENSORRT_ROOT}/include"
    PATHS /usr/include/aarch64-linux-gnu /usr/include
    PATH_SUFFIXES include)
  find_library(TRT_LIB nvinfer
    HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/lib64
    PATH_SUFFIXES lib lib64)
  file(GLOB _CUDA_DIRS /usr/local/cuda /usr/local/cuda-* "${CUDA_ROOT}")
  find_path(CUDA_INC cuda_runtime_api.h
    HINTS "${CUDA_ROOT}/include"
    PATHS ${_CUDA_DIRS}
    PATH_SUFFIXES include)
  find_library(CUDART_LIB cudart
    HINTS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib"
    PATHS ${_CUDA_DIRS}
    PATH_SUFFIXES lib64 lib lib/aarch64-linux-gnu)
  if(TRT_INC AND TRT_LIB AND CUDA_INC AND CUDART_LIB)
    set(TRT_MAJOR "")
    if(EXISTS "${TRT_INC}/NvInferVersion.h")
      file(STRINGS "${TRT_INC}/NvInferVersion.h" _TRT_MAJOR_LINE
           REGEX "^[ \t]*#define[ \t]+NV_TENSORRT_MAJOR[ \t]+[0-9]+")
      if(_TRT_MAJOR_LINE)
        string(REGEX REPLACE ".*NV_TENSORRT_MAJOR[ \t]+([0-9]+).*" "\\1"
               TRT_MAJOR "${_TRT_MAJOR_LINE}")
      endif()
    endif()
    if(NOT TRT_MAJOR)
      message(FATAL_ERROR
        "USE_TRT=ON requires NvInferVersion.h with NV_TENSORRT_MAJOR; "
        "the native backend targets TensorRT 10.x. Set USE_TRT=OFF and use "
        "the ORT TensorRT EP for older SDKs.")
    endif()
    if(TRT_MAJOR LESS 10)
      message(FATAL_ERROR
        "USE_TRT=ON found TensorRT ${TRT_MAJOR}; the native backend requires "
        "TensorRT 10.x named-I/O APIs. Set USE_TRT=OFF and use ORT + TRT-EP "
        "for TensorRT 8.x targets.")
    endif()
    list(APPEND SRC  src/trt_backend.cpp)
    list(APPEND INCS ${TRT_INC} ${CUDA_INC})
    list(APPEND LIBS ${TRT_LIB} ${CUDART_LIB})
    list(APPEND DEFS USE_TRT)
    message(STATUS "TensorRT backend: ON (v${TRT_MAJOR}; ${TRT_LIB} + ${CUDART_LIB})")
  else()
    message(FATAL_ERROR "USE_TRT=ON but TensorRT/CUDA not found (TRT_INC=${TRT_INC} TRT_LIB=${TRT_LIB} CUDA_INC=${CUDA_INC} CUDART_LIB=${CUDART_LIB})")
  endif()
endif()

if(NOT ORT_ENABLED AND NOT NCNN_ENABLED AND NOT MNN_ENABLED AND NOT USE_TRT AND NOT ALLOW_NO_BACKENDS)
  message(FATAL_ERROR
    "No inference backend was configured. Install an SDK, enable one backend, or set "
    "-DALLOW_NO_BACKENDS=ON for a diagnostic CLI build.")
endif()

add_executable(yolomaster_edge ${SRC})
target_include_directories(yolomaster_edge PRIVATE ${INCS})
target_link_libraries(yolomaster_edge PRIVATE ${LIBS})
target_compile_definitions(yolomaster_edge PRIVATE ${DEFS})

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
endif()

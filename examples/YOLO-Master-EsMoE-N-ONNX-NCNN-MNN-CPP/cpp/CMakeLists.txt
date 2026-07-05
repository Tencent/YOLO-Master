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

# SDK roots — override per platform:
#   Linux : cmake -DONNXRUNTIME_ROOT=/path -DNCNN_ROOT=/path ..
#   Windows: cmake -DONNXRUNTIME_ROOT=C:/dev/onnxruntime -DNCNN_ROOT=C:/dev/ncnn -DOpenCV_DIR=C:/dev/opencv/build ..
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime" CACHE PATH "ONNXRuntime root")
set(NCNN_ROOT        "${CMAKE_SOURCE_DIR}/../third_party/ncnn"        CACHE PATH "ncnn root")

option(USE_ORT  "Build ONNXRuntime backend" ON)
option(USE_NCNN "Build ncnn backend"        ON)

set(SRC  src/common.cpp src/main.cpp src/stb_impl.cpp)
set(INCS include third_party ${OpenCV_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS})
set(DEFS "")
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

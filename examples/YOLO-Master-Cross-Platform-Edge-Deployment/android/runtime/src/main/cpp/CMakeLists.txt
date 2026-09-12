cmake_minimum_required(VERSION 3.22.1)
project(yolomaster_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The C++ inference core is reused verbatim from the repo's cpp/ (the same sources
# the Linux/Windows/Jetson runners compile). This CMakeLists lives at
# android/runtime/src/main/cpp/ -> the repo root is five levels up.
# Precedence: -DYOLOMASTER_CORE_DIR=<path>, then a vendored snapshot at ./core (used when
# the module is shipped on its own, e.g. inside the upstream example), then the repo's cpp/.
if(YOLOMASTER_CORE_DIR)
  set(CORE ${YOLOMASTER_CORE_DIR})
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/src/common.cpp)
  set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/core)
else()
  set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp)
endif()
message(STATUS "yolomaster core: ${CORE}")

# ---- ncnn (official Android prebuilt, per-ABI; tag 20260526 for .param/.bin parity) ----
if(NOT NCNN_ANDROID_ROOT)
  message(FATAL_ERROR "NCNN_ANDROID_ROOT is not set. Copy android/sdk-paths.example.properties "
                      "to android/sdk-paths.properties and point it at the ncnn Android SDK.")
endif()
set(ncnn_DIR "${NCNN_ANDROID_ROOT}/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
# The official prebuilt is built with NCNN_DISABLE_EXCEPTION/RTTI and exports
# "-fno-rtti;-fno-exceptions" to every consumer. Our core (NcnnBackend throws on a bad
# model, the JNI bridge catches and reports it as lastError) needs both, and ncnn itself
# never throws, so dropping the exported flags is safe. A PRIVATE target_compile_options
# would be emitted BEFORE the interface flags and lose; clearing the property is the
# deterministic fix.
set_property(TARGET ncnn PROPERTY INTERFACE_COMPILE_OPTIONS "")

# ---- OpenCV (opencv-mobile Android: core + imgproc only) ----
if(NOT OPENCV_ANDROID_ROOT)
  message(FATAL_ERROR "OPENCV_ANDROID_ROOT is not set. See android/sdk-paths.example.properties.")
endif()
set(OpenCV_DIR "${OPENCV_ANDROID_ROOT}/sdk/native/jni")
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(yolomaster_ncnn SHARED
  jni_bridge.cpp
  ${CORE}/src/common.cpp
  ${CORE}/src/ncnn_backend.cpp
  ${CORE}/src/stb_impl.cpp
)

target_include_directories(yolomaster_ncnn PRIVATE
  ${CORE}/include
  ${CORE}/third_party
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(yolomaster_ncnn PRIVATE USE_NCNN)

# ---- ONNX Runtime (optional second runtime; QNN EP = Hexagon NPU on arm64) ----
# ORT_ANDROID_ROOT is build/ort, filled by the Gradle `extractOrt` task from the
# onnxruntime-android(-qnn) AARs: headers/ + jni/<abi>/libonnxruntime.so. Present for both ABIs
# (arm64-v8a: the QNN build; x86_64: the plain build for ort-CPU on the emulator); when the
# library for this ABI is missing the module builds ncnn-only, exactly as before.
set(ORT_LIB "${ORT_ANDROID_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so")
if(ORT_ANDROID_ROOT AND EXISTS "${ORT_LIB}")
  target_sources(yolomaster_ncnn PRIVATE ${CORE}/src/ort_backend.cpp)
  target_include_directories(yolomaster_ncnn PRIVATE ${ORT_ANDROID_ROOT}/headers)
  target_compile_definitions(yolomaster_ncnn PRIVATE USE_ORT)
  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_LIB}")
  target_link_libraries(yolomaster_ncnn onnxruntime)
  message(STATUS "ONNX Runtime backend: ON (${ORT_LIB})")
else()
  message(STATUS "ONNX Runtime backend: OFF (no ${ORT_LIB})")
endif()

find_library(log-lib log)

# Linking the `ncnn` package target propagates its include dirs (net.h/cpu.h/gpu.h)
# and Vulkan/glslang deps. jnigraphics is for AndroidBitmap_* pixel access.
target_link_libraries(yolomaster_ncnn
  ncnn
  ${OpenCV_LIBS}
  jnigraphics
  android
  ${log-lib}
)

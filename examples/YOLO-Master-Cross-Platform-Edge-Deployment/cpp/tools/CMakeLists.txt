# ncnn_bench - standalone ncnn latency bench (Workstream B4 CPU matrix + R5 Vulkan variants).
# Independent of cpp/ so nothing clashes with the runner build: links ncnn (+OpenMP), no OpenCV,
# C++17. Point NCNN_INSTALL at an ncnn install prefix (include/ncnn/net.h + lib/libncnn.*).
# The `vulkan` / `vulkan-fp32` variants exist only when that install was built with NCNN_VULKAN=ON
# (platform.h `#define NCNN_VULKAN 1`: the Android SDK at /data/android/ncnn-*-android-vulkan and
# the Jetson build from jetson/24_build_ncnn.sh); ncnn-x86-* is NCNN_VULKAN=OFF and prints a skip
# line. ncnn's simplevk dlopens libvulkan.so at runtime, so no Vulkan link dependency is needed.
#
# x86 dev box (load smoke ONLY - fp16 flags are inert without asimdhp and AVX int8 says nothing
# about ARM sdot/i8mm; never report x86 timings as speed). Host cmake 3.16 is too old:
#   /data/tmp/venv-v2608/bin/cmake -S cpp/tools -B cpp/tools/build-x86 -DCMAKE_BUILD_TYPE=Release \
#       -DNCNN_INSTALL=$PWD/third_party/ncnn-x86-20260526
#   /data/tmp/venv-v2608/bin/cmake --build cpp/tools/build-x86 -j8
#   (never third_party/ncnn: that is the 22.04 Android/parity SDK and does not run on this host)
#
# Jetson / aarch64 Linux (jetson/26_ncnn_int8_bench.sh does this):
#   cmake -S cpp/tools -B cpp/tools/build-aarch64 -DCMAKE_BUILD_TYPE=Release \
#       -DNCNN_INSTALL=$PWD/third_party/ncnn-aarch64-20260526      # from jetson/24_build_ncnn.sh
#
# Android arm64 via the NDK toolchain (the device that yields the real verdict):
#   cmake -S cpp/tools -B cpp/tools/build-android \
#       -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
#       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
#       -DNCNN_INSTALL=$NCNN_ANDROID_ROOT/arm64-v8a
#   cmake --build cpp/tools/build-android -j8
#   adb shell mkdir -p /data/local/tmp/ymbench
#   adb push cpp/tools/build-android/ncnn_bench probe_640.f32 \
#            models/p03_v01n_ncnn models/p03_v01n-int8_ncnn /data/local/tmp/ymbench/
#   adb shell "cd /data/local/tmp/ymbench && chmod +x ncnn_bench && \
#       ./ncnn_bench p03_v01n_ncnn --input probe_640.f32 --threads 1,2,4,big --iters 100 \
#       --warmup 20 --rounds 3 --variants fp32,fp16,int8+fp32,int8+fp16,vulkan --powersave 2 \
#       --label \$(getprop ro.product.model | tr ' ' _) --json p03_v01n.json"
#   (the Android SDK ships a static libncnn.a whose OpenMP/glslang deps come from its
#    ncnnConfig.cmake - hence the config-file route below is preferred over find_library;
#    the resulting binary NEEDs only libc/libm/libdl/liblog/libandroid/libjnigraphics)
cmake_minimum_required(VERSION 3.16)
project(ncnn_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT NCNN_INSTALL)
  message(FATAL_ERROR "pass -DNCNN_INSTALL=<ncnn install prefix> (include/ncnn + lib/)")
endif()
get_filename_component(NCNN_INSTALL "${NCNN_INSTALL}" ABSOLUTE)
if(NOT EXISTS "${NCNN_INSTALL}/include/ncnn/net.h")
  message(FATAL_ERROR "no ncnn headers under ${NCNN_INSTALL}/include/ncnn")
endif()

add_executable(ncnn_bench ncnn_bench.cpp)
target_include_directories(ncnn_bench PRIVATE ${NCNN_INSTALL}/include/ncnn)

# Prefer the install's own ncnnConfig.cmake (carries OpenMP / Threads / Vulkan deps, needed for
# the static Android SDK); fall back to the bare library the way the prior art did.
set(ncnn_DIR "${NCNN_INSTALL}/lib/cmake/ncnn")
find_package(ncnn QUIET CONFIG PATHS "${NCNN_INSTALL}/lib/cmake/ncnn" NO_DEFAULT_PATH)
if(TARGET ncnn)
  message(STATUS "ncnn: imported target from ${ncnn_DIR}")
  target_link_libraries(ncnn_bench PRIVATE ncnn)
else()
  find_library(NCNN_LIB ncnn PATHS ${NCNN_INSTALL}/lib NO_DEFAULT_PATH)
  if(NOT NCNN_LIB)
    message(FATAL_ERROR "libncnn not found under ${NCNN_INSTALL}/lib")
  endif()
  message(STATUS "ncnn: ${NCNN_LIB}")
  target_link_libraries(ncnn_bench PRIVATE ${NCNN_LIB})
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(ncnn_bench PRIVATE OpenMP::OpenMP_CXX)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(ncnn_bench PRIVATE Threads::Threads)
endif()

# Shared-lib installs: keep the lib dir on the runtime path of the build-tree binary so it runs
# from cpp/tools/build-*/ without LD_LIBRARY_PATH.
if(NOT ANDROID)
  set_target_properties(ncnn_bench PROPERTIES BUILD_RPATH "${NCNN_INSTALL}/lib")
endif()

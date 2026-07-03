# CMake toolchain for cross-compiling to Linux ARM64 (aarch64).
# Targets Jetson / ARM servers. Use with aarch64 builds of OpenCV + ONNX Runtime
# (installable natively on the ARM box, or via Dockerfile.arm64).
#
#   cmake -S . -B build-arm64 -DCMAKE_TOOLCHAIN_FILE=../aarch64-toolchain.cmake \
#         -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime-aarch64
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Prefer the conda cross-compiler if present, else the Debian/Ubuntu system cross-compiler.
find_program(AARCH64_CXX NAMES aarch64-conda-linux-gnu-g++ aarch64-linux-gnu-g++)
if(AARCH64_CXX)
  set(CMAKE_CXX_COMPILER ${AARCH64_CXX})
  find_program(AARCH64_CC NAMES aarch64-conda-linux-gnu-gcc aarch64-linux-gnu-gcc)
  set(CMAKE_C_COMPILER ${AARCH64_CC})
else()
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
  set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
endif()
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)

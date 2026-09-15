# Cross-compile the image-only runner for Linux ARM64/Jetson.  Set the
# compiler/sysroot paths in the configure command when they are not on PATH.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc CACHE FILEPATH "ARM64 C compiler")
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++ CACHE FILEPATH "ARM64 C++ compiler")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

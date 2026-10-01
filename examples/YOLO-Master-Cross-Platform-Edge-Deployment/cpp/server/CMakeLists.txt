# yolomaster_server: HTTP/1.1 + WebSocket inference API on top of yolomaster_core.
# HTTP engine: vendored uWebSockets (third_party/uWebSockets, Apache-2.0) + uSockets (epoll, no SSL).
cmake_minimum_required(VERSION 3.16)
enable_language(C)
set(UWS_ROOT "${CMAKE_SOURCE_DIR}/../third_party/uWebSockets" CACHE PATH "uWebSockets source root (scripts/server/fetch_uws.sh)")
if(NOT EXISTS "${UWS_ROOT}/src/App.h")
  message(FATAL_ERROR "uWebSockets not found at ${UWS_ROOT}: run scripts/server/fetch_uws.sh")
endif()
set(USOCKETS_SRC
  ${UWS_ROOT}/uSockets/src/bsd.c
  ${UWS_ROOT}/uSockets/src/context.c
  ${UWS_ROOT}/uSockets/src/loop.c
  ${UWS_ROOT}/uSockets/src/socket.c
  ${UWS_ROOT}/uSockets/src/udp.c
  ${UWS_ROOT}/uSockets/src/eventing/epoll_kqueue.c)
add_library(usockets STATIC ${USOCKETS_SRC})
target_include_directories(usockets PUBLIC ${UWS_ROOT}/uSockets/src)
target_compile_definitions(usockets PUBLIC LIBUS_NO_SSL LIBUS_USE_EPOLL)
set_target_properties(usockets PROPERTIES C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)

add_executable(yolomaster_server
  server_main.cpp http_api.cpp worker_pool.cpp serialize.cpp openapi.cpp)
target_include_directories(yolomaster_server PRIVATE ${UWS_ROOT}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(yolomaster_server PRIVATE UWS_NO_ZLIB UWS_HTTPRESPONSE_NO_WRITEMARK
                           "YM_SERVER_VERSION=\"${YM_SERVER_VERSION}\"")
target_link_libraries(yolomaster_server PRIVATE yolomaster_core usockets pthread)
# CUDA runtime for /metrics GPU memory when the TRT backend is in (CUDART_LIB is set by the parent)
if((USE_TRT OR USE_CUDA_PREPROC) AND CUDART_LIB)
  target_compile_definitions(yolomaster_server PRIVATE HAVE_CUDART)
endif()
install(TARGETS yolomaster_server RUNTIME DESTINATION bin)

cmake_minimum_required(VERSION 3.12)
project(yolo_master_edge_deployment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(yolo_master_edge_benchmark cpp/edge_benchmark_stub.cpp)

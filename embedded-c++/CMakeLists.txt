cmake_minimum_required(VERSION 2.8.12...3.29)
project(example-c++)

set(CMAKE_CXX_STANDARD 11)

include_directories(../duckdb/src/include include)
link_directories(../build/release/src)

set(BENCHMARK_LIB_PATH "${CMAKE_SOURCE_DIR}/include/benchmark")
find_library(BENCHMARK_LIB NAMES benchmark PATHS ${BENCHMARK_LIB_PATH})
if(NOT BENCHMARK_LIB)
    message(FATAL_ERROR "Benchmark library not found.")
else()
    message(STATUS "Benchmark library found: ${BENCHMARK_LIB}")
endif()

add_executable(example main.cpp)
target_link_libraries(example duckdb ${BENCHMARK_LIB})

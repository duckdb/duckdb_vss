cmake_minimum_required(VERSION 2.8.12)

# Set extension name here
set(TARGET_NAME vss)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

option(USE_SIMSIMD
       "Use SIMSIMD library to sacrifice portability for vectorized search" OFF)
if(USE_SIMSIMD)
  add_definitions(-DDUCKDB_USEARCH_USE_SIMSIMD=1)
else()
  add_definitions(-DDUCKDB_USEARCH_USE_SIMSIMD=0)
endif()

include_directories(src/include)

set(EXTENSION_SOURCES src/vss_extension.cpp)
add_subdirectory(src/hnsw)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

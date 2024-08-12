
set(EXTENSION_SOURCES
        ${EXTENSION_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_logical_create.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_macros.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_physical_create.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_pragmas.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_plan_index_create.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_plan_index_scan.cpp
        PARENT_SCOPE
)

set(EXTENSION_SOURCES
        ${EXTENSION_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_macros.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_physical_create.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_pragmas.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_index_plan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_topk_operator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_optimize_expr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_optimize_join.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_optimize_topk.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_optimize_scan.cpp
        PARENT_SCOPE
)
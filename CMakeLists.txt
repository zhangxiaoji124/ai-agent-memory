cmake_minimum_required(VERSION 3.20)
project(agent_memory_io_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AMIO_ENABLE_URING "Enable liburing backend on Linux" ON)

add_library(amio
  src/policy/agent_io_policy.cpp
  src/index/storage_layout.cpp
  src/index/nsw.cpp
  src/index/hnsw.cpp
  src/cache/graph_aware_cache.cpp
  src/prefetch/io_uring_backend.cpp
  src/prefetch/topology_prefetcher.cpp
  src/write/memtable.cpp
  src/write/wal.cpp
  src/write/compaction.cpp
  src/dataset/fvecs.cpp
  src/dataset/vector_dataset.cpp
  src/dataset/index_builder.cpp
  src/dataset/index_build_common.cpp
  src/index/external_vectors.cpp
  src/runtime/memory_partition.cpp
  src/runtime/static_subgraph.cpp
  src/runtime/graph_reorder.cpp
  src/vector_store.cpp
        include/cache/graph_aware_cache.h
        include/util/io_metrics.h
)

target_include_directories(amio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(amio PUBLIC AMIO_VERSION_STR="0.1.0")

if(UNIX AND NOT APPLE)
  target_compile_definitions(amio PUBLIC AMIO_LINUX=1)
endif()

if(AMIO_ENABLE_URING AND UNIX AND NOT APPLE)
  find_path(LIBURING_INCLUDE_DIR liburing.h)
  find_library(LIBURING_LIBRARY uring)
  if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
    target_compile_definitions(amio PUBLIC AMIO_HAS_URING=1)
    target_include_directories(amio PRIVATE ${LIBURING_INCLUDE_DIR})
    target_link_libraries(amio PRIVATE ${LIBURING_LIBRARY})
  else()
    message(STATUS "liburing not found; building with noop prefetch backend")
    target_compile_definitions(amio PUBLIC AMIO_HAS_URING=0)
  endif()
else()
  target_compile_definitions(amio PUBLIC AMIO_HAS_URING=0)
endif()

add_executable(run_tests
  tests/main.cpp
  tests/recall_test.cpp
  tests/disk_roundtrip_test.cpp
  tests/concurrent_test.cpp
  tests/memory_partition_test.cpp
  tests/high_dim_index_test.cpp
)
target_link_libraries(run_tests PRIVATE amio)

add_executable(dataset_loader tools/dataset_loader.cpp)
target_link_libraries(dataset_loader PRIVATE amio)

add_executable(build_index tools/build_index.cpp)
target_link_libraries(build_index PRIVATE amio)

add_executable(eval_disk tools/eval_disk.cpp)
target_link_libraries(eval_disk PRIVATE amio)

add_executable(reorder_index tools/reorder_index.cpp)
target_link_libraries(reorder_index PRIVATE amio)

add_executable(bench_search benches/bench_search.cpp)
target_link_libraries(bench_search PRIVATE amio)

add_executable(bench_write benches/bench_write.cpp)
target_link_libraries(bench_write PRIVATE amio)

# ── Python ctypes bridge ───────────────────────────────────────────────────
option(AMIO_BUILD_PYTHON "Build amio_py shared library for Python bindings" OFF)
if(AMIO_BUILD_PYTHON)
  add_library(amio_py SHARED src/python_bridge.cpp)
  target_link_libraries(amio_py PRIVATE amio)
  target_include_directories(amio_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(amio_py PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  if(WIN32)
    # Windows: DLL goes to build root so Python can find it easily
    set_target_properties(amio_py PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
  endif()
  message(STATUS "amio_py: Python ctypes bridge enabled")
endif()
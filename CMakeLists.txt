# =============================================================================
# Cardinal - Root CMakeLists.txt (v1.2.0)
# Neurosymbolic AGI Architecture
# =============================================================================
# Offline-first: uses vendored sources if present, falls back to FetchContent
# 
# For offline builds (default, recommended):
#   - Populate vendor/ with required repos (see README.md)
#   - cmake -B build -DCMAKE_BUILD_TYPE=Release
#   - cmake --build build -j$(nproc)
#
# For online builds (if vendor/ missing, or -DFORCE_FETCH=ON):
#   - cmake -B build -DCMAKE_BUILD_TYPE=Release -DFORCE_FETCH=ON
#   - cmake --build build -j$(nproc)
# =============================================================================

cmake_minimum_required(VERSION 3.28)

project(Cardinal
    VERSION 1.2.0
    DESCRIPTION "Neurosymbolic AGI Architecture"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# User options
# -----------------------------------------------------------------------------
option(CARDINAL_ENABLE_TENSORRT
    "Build TensorRT-LLM backend"
    OFF)

option(FORCE_FETCH
    "Force FetchContent even if vendored sources exist"
    OFF)

message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT backend:   ${CARDINAL_ENABLE_TENSORRT}")
message(STATUS "Force fetch:        ${FORCE_FETCH}")
message(STATUS "Version:            ${PROJECT_VERSION}")

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------
add_compile_options(
    -Wall -Wextra -Wno-unused-parameter
    "$<$<CONFIG:Release>:-O2>"
    "$<$<CONFIG:Debug>:-g;-O0>"
)

# -----------------------------------------------------------------------------
# Helper macro: use vendored or fetch
# -----------------------------------------------------------------------------
macro(add_vendored_or_fetched name vendor_path git_url git_tag)
    if(NOT FORCE_FETCH AND EXISTS "${CMAKE_SOURCE_DIR}/${vendor_path}/CMakeLists.txt")
        message(STATUS "${name}: using vendored source from ${vendor_path}")
        add_subdirectory(${vendor_path} ${CMAKE_BINARY_DIR}/${name})
    else()
        message(STATUS "${name}: vendored source not found, fetching from ${git_url}")
        FetchContent_Declare(${name}
            GIT_REPOSITORY ${git_url}
            GIT_TAG        ${git_tag}
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(${name})
    endif()
endmacro()

# -----------------------------------------------------------------------------
# FetchContent (only used as fallback)
# -----------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# nlohmann/json
add_vendored_or_fetched(nlohmann_json
    "vendor/nlohmann_json"
    "https://github.com/nlohmann/json.git"
    "v3.11.3")

# cpp-httplib
add_vendored_or_fetched(httplib
    "vendor/cpp-httplib"
    "https://github.com/yhirose/cpp-httplib.git"
    "v0.18.3")

# muparser (vendored fallback)
if(NOT FORCE_FETCH AND EXISTS "${CMAKE_SOURCE_DIR}/vendor/muparser/CMakeLists.txt")
    message(STATUS "muparser: using vendored source from vendor/muparser")
    set(ENABLE_SAMPLES OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENMP  OFF CACHE BOOL "" FORCE)
    add_subdirectory(vendor/muparser ${CMAKE_BINARY_DIR}/muparser)
else()
    message(STATUS "muparser: vendored not found, fetching")
    FetchContent_Declare(muparser
        GIT_REPOSITORY https://github.com/beltoforion/muparser.git
        GIT_TAG        v2.3.4
        GIT_SHALLOW    TRUE)
    set(ENABLE_SAMPLES OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENMP  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(muparser)
endif()

# tokenizers-cpp (TRT backend only)
set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(CARDINAL_ENABLE_TENSORRT)
    add_vendored_or_fetched(tokenizers_cpp
        "vendor/tokenizers-cpp"
        "https://github.com/mlc-ai/tokenizers-cpp.git"
        "main")
endif()

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------

# OpenSSL
find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL: ${OPENSSL_VERSION}")
add_compile_definitions(CPPHTTPLIB_OPENSSL_SUPPORT)

# SQLite3
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SQLITE3 REQUIRED sqlite3)
    message(STATUS "SQLite3: ${SQLITE3_VERSION}")
    set(SQLITE3_SYSTEM ON)
else()
    find_library(SQLITE3_LIB  NAMES sqlite3 REQUIRED)
    find_path(SQLITE3_INCLUDE  NAMES sqlite3.h REQUIRED)
    set(SQLITE3_MANUAL ON)
endif()

# llama.cpp (vendored only — no fetch fallback)
set(LLAMA_ROOT  ${CMAKE_SOURCE_DIR}/vendor/llama.cpp)
if(NOT EXISTS "${LLAMA_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_ROOT}\n"
        "Please clone it: git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_ROOT}")
endif()

set(LLAMA_BUILD ${LLAMA_ROOT}/build)
set(LLAMA_INCLUDE_DIRS
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
    ${LLAMA_ROOT}/common)

set(LLAMA_SEARCH ${LLAMA_BUILD}/bin ${LLAMA_BUILD}/lib
                  ${LLAMA_BUILD}/src ${LLAMA_BUILD}/common
                  ${LLAMA_BUILD}/ggml/src ${LLAMA_BUILD}/ggml/src/ggml-cuda)

find_library(LLAMA_LIB        NAMES llama     PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB         NAMES ggml      PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB    NAMES ggml-base PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB    NAMES ggml-cuda PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)
find_library(LLAMA_COMMON_LIB NAMES common    PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)

set(LLAMA_LIBRARIES ${LLAMA_LIB} ${GGML_LIB})
foreach(L ${GGML_BASE_LIB} ${GGML_CUDA_LIB} ${LLAMA_COMMON_LIB})
    if(L)
        list(APPEND LLAMA_LIBRARIES ${L})
    endif()
endforeach()

# CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")

# SWI-Prolog
if(PkgConfig_FOUND)
    pkg_check_modules(SWIPL QUIET swipl)
endif()
if(SWIPL_FOUND)
    set(SWIPL_INCLUDE_DIR ${SWIPL_INCLUDE_DIRS})
    set(SWIPL_LIB         ${SWIPL_LIBRARIES})
else()
    find_path(SWIPL_INCLUDE_DIR NAMES SWI-Prolog.h
        PATHS /usr/lib/swi-prolog/include
              /usr/local/lib/swi-prolog/include
              /usr/include/swipl REQUIRED)
    find_library(SWIPL_LIB NAMES swipl
        PATHS /usr/lib/swi-prolog/lib/x86_64-linux
              /usr/lib/x86_64-linux-gnu
              /usr/local/lib REQUIRED)
endif()
message(STATUS "SWI-Prolog: ${SWIPL_INCLUDE_DIR}")
add_compile_definitions(CARDINAL_PROLOG_ENABLED)

# TensorRT-LLM (optional)
if(CARDINAL_ENABLE_TENSORRT)
    if(NOT DEFINED TRT_LLM_INCLUDE_DIR)
        set(TRT_LLM_INCLUDE_DIR /usr/local/include)
    endif()
    if(NOT DEFINED TRT_LLM_LIB_DIR)
        set(TRT_LLM_LIB_DIR /usr/local/lib)
    endif()
    find_path(TENSORRT_LLM_INCLUDE
        NAMES tensorrt_llm/executor/executor.h
        PATHS ${TRT_LLM_INCLUDE_DIR} REQUIRED)
    find_library(TENSORRT_LLM_LIB NAMES tensorrt_llm
        PATHS ${TRT_LLM_LIB_DIR} REQUIRED)
    find_library(NVINFER_LIB NAMES nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib REQUIRED)
    find_library(CUDNN_LIB NAMES cudnn
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 REQUIRED)
    message(STATUS "TRT-LLM: ${TENSORRT_LLM_LIB}")
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
set(CARDINAL_UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/config_loader.cpp
    src/utils/json_parser.cpp
)

set(CARDINAL_CORE_SOURCES
    src/core/feeling_output.cpp
    src/core/inference.cpp
    src/core/backend_factory.cpp
    src/core/backends/llama_cpp_backend.cpp
)
if(CARDINAL_ENABLE_TENSORRT)
    list(APPEND CARDINAL_CORE_SOURCES
        src/core/backends/tensorrt_backend.cpp)
endif()

set(CARDINAL_MEMORY_SOURCES
    src/memory/rule_store.cpp
    src/memory/knowledge_graph.cpp
    src/memory/episodic.cpp
    src/memory/episodic_storage.cpp
    src/memory/episodic_retriever.cpp
)

set(CARDINAL_VERIFIER_SOURCES
    src/verifier/symbolic_engine.cpp
    src/verifier/rule_extractor.cpp
    src/verifier/consistency_check.cpp
    src/verifier/neural_verifier.cpp
)

set(CARDINAL_TOOLS_SOURCES
    src/tools/tool_registry.cpp
    src/tools/tool_executor.cpp
)

set(CARDINAL_AGENT_SOURCES
    src/agent/agent_context.cpp
    src/agent/agent_planner.cpp
    src/agent/agent_executor.cpp
)

set(CARDINAL_EXPLAINABILITY_SOURCES
    src/explainability/trace_builder.cpp
    src/explainability/audit_log.cpp
    src/explainability/explainability_exporter.cpp
)

set(CARDINAL_LEARNING_SOURCES
    src/learning/training_exporter.cpp
)

set(CARDINAL_API_SOURCES
    src/api/cardinal_settings.cpp
    src/api/session.cpp
    src/api/cardinal_api.cpp
    src/api/http_server.cpp
)

set(CARDINAL_ALL_SOURCES
    ${CARDINAL_UTILS_SOURCES}
    ${CARDINAL_CORE_SOURCES}
    ${CARDINAL_MEMORY_SOURCES}
    ${CARDINAL_VERIFIER_SOURCES}
    ${CARDINAL_TOOLS_SOURCES}
    ${CARDINAL_AGENT_SOURCES}
    ${CARDINAL_EXPLAINABILITY_SOURCES}
    ${CARDINAL_LEARNING_SOURCES}
    ${CARDINAL_API_SOURCES}
    src/main.cpp
)

# -----------------------------------------------------------------------------
# Cardinal executable
# -----------------------------------------------------------------------------
add_executable(cardinal ${CARDINAL_ALL_SOURCES})

target_include_directories(cardinal PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LLAMA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${SWIPL_INCLUDE_DIR}
)

if(DEFINED SQLITE3_SYSTEM AND SQLITE3_SYSTEM)
    target_include_directories(cardinal PRIVATE ${SQLITE3_INCLUDE_DIRS})
elseif(DEFINED SQLITE3_MANUAL AND SQLITE3_MANUAL)
    target_include_directories(cardinal PRIVATE ${SQLITE3_INCLUDE})
endif()

if(CARDINAL_ENABLE_TENSORRT)
    target_include_directories(cardinal PRIVATE ${TENSORRT_LLM_INCLUDE})
    target_link_libraries(cardinal PRIVATE tokenizers_cpp)
endif()

target_compile_definitions(cardinal PRIVATE
    CARDINAL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    CARDINAL_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    CARDINAL_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    GGML_USE_CUDA
)

if(CARDINAL_ENABLE_TENSORRT)
    target_compile_definitions(cardinal PRIVATE CARDINAL_ENABLE_TENSORRT)
endif()

target_link_libraries(cardinal PRIVATE
    ${LLAMA_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
    httplib::httplib
    muparser
    OpenSSL::SSL
    OpenSSL::Crypto
    ${SWIPL_LIB}
    pthread
    dl
)

if(DEFINED SQLITE3_SYSTEM AND SQLITE3_SYSTEM)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_LIBRARIES})
elseif(DEFINED SQLITE3_MANUAL AND SQLITE3_MANUAL)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_LIB})
endif()

if(CARDINAL_ENABLE_TENSORRT)
    target_link_libraries(cardinal PRIVATE
        ${TENSORRT_LLM_LIB} ${NVINFER_LIB} ${CUDNN_LIB}
        CUDA::cuda_driver)
endif()

# RPATH
set(CARDINAL_RPATH "${LLAMA_BUILD}/lib;${LLAMA_BUILD}/bin")
if(CARDINAL_ENABLE_TENSORRT)
    list(APPEND CARDINAL_RPATH "${TRT_LLM_LIB_DIR}")
endif()
set_target_properties(cardinal PROPERTIES
    BUILD_RPATH   "${CARDINAL_RPATH}"
    INSTALL_RPATH "${CARDINAL_RPATH}")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "=============================================================")
message(STATUS "Cardinal v${PROJECT_VERSION} Configuration")
message(STATUS "=============================================================")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch:          ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT backend:   ${CARDINAL_ENABLE_TENSORRT}")
message(STATUS "Features:           tools + agent + explainability")
message(STATUS "Output dir:         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "=============================================================")

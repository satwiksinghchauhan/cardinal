# =============================================================================
# Cardinal - Root CMakeLists.txt
# Neurosymbolic AGI Architecture
# =============================================================================
# Dependencies (always):
#   - llama.cpp         (vendor/llama.cpp, CUDA-built)
#   - nlohmann/json     (vendor/nlohmann_json)
#   - cpp-httplib       (vendor/cpp-httplib)
#   - OpenSSL           (system: apt install libssl-dev)
#   - SQLite3           (system: apt install libsqlite3-dev)
#   - SWI-Prolog        (system: apt install swi-prolog, or pkg-config)
#
# Dependencies (TensorRT backend only, -DCARDINAL_ENABLE_TENSORRT=ON):
#   - TensorRT 10.x     (system install, /usr/include/tensorrt-llm)
#   - CUDA 12.x         (already required by llama.cpp)
#   - cuDNN 9.x         (system: apt install cudnn9-cuda-12)
#   - tokenizers-cpp    (vendor/tokenizers-cpp)
#
# Switching backends at runtime:
#   Edit config.json -> "backend": "llama_cpp" | "tensorrt"
#
# Build (llama.cpp only, default):
#   cd vendor/llama.cpp && mkdir build && cd build
#   cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#
#   Then from project root:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#
# Build (with TensorRT backend):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARDINAL_ENABLE_TENSORRT=ON
#   cmake --build build -j$(nproc)
# =============================================================================

cmake_minimum_required(VERSION 3.28)

# -----------------------------------------------------------------------------
# Project definition
# -----------------------------------------------------------------------------
project(Cardinal
    VERSION 1.1.0
    DESCRIPTION "Neurosymbolic AGI Architecture"
    LANGUAGES CXX CUDA
)

# -----------------------------------------------------------------------------
# C++ / CUDA standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)

# RTX 3050 Laptop = sm_86 (Ampere)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# -----------------------------------------------------------------------------
# Build type default
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# User options
# -----------------------------------------------------------------------------
option(CARDINAL_ENABLE_TENSORRT
    "Build TensorRT-LLM backend (requires TRT 10.x + cuDNN 9.x installed)"
    OFF)

message(STATUS "Build type:             ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures:     ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT backend:       ${CARDINAL_ENABLE_TENSORRT}")

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)

# -----------------------------------------------------------------------------
# Compiler flags (GCC / Clang)
# -----------------------------------------------------------------------------
add_compile_options(
    -Wall
    -Wextra
    -Wno-unused-parameter
    "$<$<CONFIG:Release>:-O2>"
    "$<$<CONFIG:Debug>:-g>"
    "$<$<CONFIG:Debug>:-O0>"
)

# -----------------------------------------------------------------------------
# Vendor submodules (LOCAL, no FetchContent)
# -----------------------------------------------------------------------------
set(NLOHMANN_JSON_ROOT ${CMAKE_SOURCE_DIR}/vendor/nlohmann_json)
set(CPP_HTTPLIB_ROOT    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
set(TOKENIZERS_CPP_ROOT ${CMAKE_SOURCE_DIR}/vendor/tokenizers-cpp)

# -----------------------------------------------------------------------------
# nlohmann/json (header-only, just need include path)
# -----------------------------------------------------------------------------
if(NOT EXISTS ${NLOHMANN_JSON_ROOT}/include/nlohmann/json.hpp)
    message(FATAL_ERROR 
        "nlohmann/json not found at ${NLOHMANN_JSON_ROOT}\n"
        "Run: git submodule update --init --recursive")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${NLOHMANN_JSON_ROOT}/include)

# -----------------------------------------------------------------------------
# cpp-httplib (header-only, needs OpenSSL)
# -----------------------------------------------------------------------------
if(NOT EXISTS ${CPP_HTTPLIB_ROOT}/httplib.h)
    message(FATAL_ERROR 
        "cpp-httplib not found at ${CPP_HTTPLIB_ROOT}\n"
        "Run: git submodule update --init --recursive")
endif()
add_library(cpp_httplib INTERFACE)
target_include_directories(cpp_httplib INTERFACE ${CPP_HTTPLIB_ROOT})
target_link_libraries(cpp_httplib INTERFACE OpenSSL::SSL OpenSSL::Crypto)
target_compile_definitions(cpp_httplib INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT)

# -----------------------------------------------------------------------------
# OpenSSL (system)
# sudo apt install libssl-dev
# -----------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
else()
    message(FATAL_ERROR "OpenSSL not found. Run: sudo apt install libssl-dev")
endif()

# -----------------------------------------------------------------------------
# SQLite3 (system)
# sudo apt install libsqlite3-dev
# -----------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SQLITE3 REQUIRED sqlite3)
    message(STATUS "SQLite3 found via pkg-config: ${SQLITE3_VERSION}")
    set(SQLITE3_SYSTEM ON)
else()
    find_library(SQLITE3_LIB NAMES sqlite3 REQUIRED)
    find_path(SQLITE3_INCLUDE NAMES sqlite3.h REQUIRED)
    message(STATUS "SQLite3 found: ${SQLITE3_LIB}")
    set(SQLITE3_MANUAL ON)
endif()

# -----------------------------------------------------------------------------
# llama.cpp (vendored)
# Build separately first:
#   cd vendor/llama.cpp && mkdir build && cd build
#   cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
# -----------------------------------------------------------------------------
set(LLAMA_ROOT  ${CMAKE_SOURCE_DIR}/vendor/llama.cpp)
set(LLAMA_BUILD ${LLAMA_ROOT}/build)

set(LLAMA_INCLUDE_DIRS
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
    ${LLAMA_ROOT}/common
)

set(LLAMA_LIB_SEARCH_PATHS
    ${LLAMA_BUILD}/bin
    ${LLAMA_BUILD}/lib
    ${LLAMA_BUILD}/src
)
set(GGML_LIB_SEARCH_PATHS
    ${LLAMA_BUILD}/bin
    ${LLAMA_BUILD}/lib
    ${LLAMA_BUILD}/ggml/src
)
set(GGML_CUDA_SEARCH_PATHS
    ${LLAMA_BUILD}/bin
    ${LLAMA_BUILD}/lib
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/ggml/src/ggml-cuda
)
set(LLAMA_COMMON_SEARCH_PATHS
    ${LLAMA_BUILD}/bin
    ${LLAMA_BUILD}/lib
    ${LLAMA_BUILD}/common
)

find_library(LLAMA_LIB        NAMES llama      PATHS ${LLAMA_LIB_SEARCH_PATHS}    NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB         NAMES ggml       PATHS ${GGML_LIB_SEARCH_PATHS}     NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB    NAMES ggml-base  PATHS ${GGML_LIB_SEARCH_PATHS}     NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB    NAMES ggml-cuda  PATHS ${GGML_CUDA_SEARCH_PATHS}    NO_DEFAULT_PATH)
find_library(LLAMA_COMMON_LIB NAMES common     PATHS ${LLAMA_COMMON_SEARCH_PATHS} NO_DEFAULT_PATH)

message(STATUS "llama lib:          ${LLAMA_LIB}")
message(STATUS "ggml lib:           ${GGML_LIB}")
message(STATUS "ggml-cuda lib:      ${GGML_CUDA_LIB}")
message(STATUS "llama common lib:   ${LLAMA_COMMON_LIB}")

set(LLAMA_LIBRARIES ${LLAMA_LIB} ${GGML_LIB})
if(GGML_BASE_LIB)
    list(APPEND LLAMA_LIBRARIES ${GGML_BASE_LIB})
endif()
if(GGML_CUDA_LIB)
    list(APPEND LLAMA_LIBRARIES ${GGML_CUDA_LIB})
endif()
if(LLAMA_COMMON_LIB)
    list(APPEND LLAMA_LIBRARIES ${LLAMA_COMMON_LIB})
endif()

# -----------------------------------------------------------------------------
# CUDA Toolkit
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# -----------------------------------------------------------------------------
# TensorRT-LLM (optional)
# -----------------------------------------------------------------------------
if(CARDINAL_ENABLE_TENSORRT)
    message(STATUS "TensorRT backend enabled — searching for TRT-LLM...")

    if(NOT DEFINED TRT_LLM_INCLUDE_DIR)
        set(TRT_LLM_INCLUDE_DIR /usr/local/include)
    endif()
    if(NOT DEFINED TRT_LLM_LIB_DIR)
        set(TRT_LLM_LIB_DIR /usr/local/lib)
    endif()

    find_path(TENSORRT_LLM_INCLUDE
        NAMES tensorrt_llm/executor/executor.h
        PATHS ${TRT_LLM_INCLUDE_DIR}
        REQUIRED
    )
    find_library(TENSORRT_LLM_LIB
        NAMES tensorrt_llm
        PATHS ${TRT_LLM_LIB_DIR}
        REQUIRED
    )
    find_library(NVINFER_LIB
        NAMES nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
        REQUIRED
    )
    find_library(CUDNN_LIB
        NAMES cudnn
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
        REQUIRED
    )

    message(STATUS "TRT-LLM include:    ${TENSORRT_LLM_INCLUDE}")
    message(STATUS "TRT-LLM lib:        ${TENSORRT_LLM_LIB}")
    message(STATUS "nvinfer lib:        ${NVINFER_LIB}")
    message(STATUS "cuDNN lib:          ${CUDNN_LIB}")

    # tokenizers-cpp (vendored submodule)
    if(NOT EXISTS ${TOKENIZERS_CPP_ROOT}/CMakeLists.txt)
        message(FATAL_ERROR
            "tokenizers-cpp not found at ${TOKENIZERS_CPP_ROOT}\n"
            "Run: git submodule update --init --recursive")
    endif()
    add_subdirectory(${TOKENIZERS_CPP_ROOT} ${CMAKE_BINARY_DIR}/tokenizers-cpp)
endif()

# -----------------------------------------------------------------------------
# SWI-Prolog (system)
# sudo apt install swi-prolog
# -----------------------------------------------------------------------------
if(PkgConfig_FOUND)
    pkg_check_modules(SWIPL QUIET swipl)
endif()

if(SWIPL_FOUND)
    message(STATUS "SWI-Prolog found via pkg-config")
    set(SWIPL_INCLUDE_DIR ${SWIPL_INCLUDE_DIRS})
    set(SWIPL_LIB         ${SWIPL_LIBRARIES})
else()
    find_path(SWIPL_INCLUDE_DIR
        NAMES SWI-Prolog.h
        PATHS
            /usr/lib/swi-prolog/include
            /usr/local/lib/swi-prolog/include
            /usr/include/swipl
            /usr/include/swi-prolog
            /opt/swi-prolog/include
        REQUIRED
    )
    find_library(SWIPL_LIB
        NAMES swipl
        PATHS
            /usr/lib/swi-prolog/lib/x86_64-linux
            /usr/lib/swi-prolog/lib/aarch64-linux
            /usr/local/lib/swi-prolog/lib/x86_64-linux
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /opt/swi-prolog/lib
        REQUIRED
    )
endif()

if(SWIPL_INCLUDE_DIR AND SWIPL_LIB)
    message(STATUS "SWI-Prolog include: ${SWIPL_INCLUDE_DIR}")
    message(STATUS "SWI-Prolog lib:     ${SWIPL_LIB}")
    set(CARDINAL_HAS_PROLOG ON)
    add_compile_definitions(CARDINAL_PROLOG_ENABLED)
else()
    message(FATAL_ERROR
        "SWI-Prolog not found.\n"
        "Install with: sudo apt install swi-prolog")
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
        src/core/backends/tensorrt_backend.cpp
    )
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
    ${CARDINAL_LEARNING_SOURCES}
    ${CARDINAL_API_SOURCES}
    src/main.cpp
)

# -----------------------------------------------------------------------------
# Cardinal executable
# -----------------------------------------------------------------------------
add_executable(cardinal ${CARDINAL_ALL_SOURCES})

# -----------------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------------
target_include_directories(cardinal PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LLAMA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${SWIPL_INCLUDE_DIR}
)

# SQLite3
if(DEFINED SQLITE3_SYSTEM AND SQLITE3_SYSTEM)
    target_include_directories(cardinal PRIVATE ${SQLITE3_INCLUDE_DIRS})
elseif(DEFINED SQLITE3_MANUAL AND SQLITE3_MANUAL)
    target_include_directories(cardinal PRIVATE ${SQLITE3_INCLUDE})
endif()

# TensorRT-LLM headers + tokenizers-cpp
if(CARDINAL_ENABLE_TENSORRT)
    target_include_directories(cardinal PRIVATE ${TENSORRT_LLM_INCLUDE})
    target_link_libraries(cardinal PRIVATE tokenizers_cpp)
endif()

# nlohmann/json + cpp-httplib (vendor submodules)
target_link_libraries(cardinal PRIVATE nlohmann_json cpp_httplib)

# -----------------------------------------------------------------------------
# Compile definitions
# -----------------------------------------------------------------------------
target_compile_definitions(cardinal PRIVATE
    CARDINAL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    CARDINAL_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    CARDINAL_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    GGML_USE_CUDA
)

if(CARDINAL_ENABLE_TENSORRT)
    target_compile_definitions(cardinal PRIVATE CARDINAL_ENABLE_TENSORRT)
endif()

# -----------------------------------------------------------------------------
# Link libraries
# -----------------------------------------------------------------------------
target_link_libraries(cardinal PRIVATE
    ${LLAMA_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    OpenSSL::SSL
    OpenSSL::Crypto
    ${SWIPL_LIB}
    pthread
    dl
)

# SQLite3
if(DEFINED SQLITE3_TARGET)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_TARGET})
elseif(DEFINED SQLITE3_SYSTEM AND SQLITE3_SYSTEM)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_LIBRARIES})
elseif(DEFINED SQLITE3_MANUAL AND SQLITE3_MANUAL)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_LIB})
endif()

# TensorRT-LLM stack (optional)
if(CARDINAL_ENABLE_TENSORRT)
    target_link_libraries(cardinal PRIVATE
        ${TENSORRT_LLM_LIB}
        ${NVINFER_LIB}
        ${CUDNN_LIB}
        CUDA::cuda_driver
    )
endif()

# -----------------------------------------------------------------------------
# RPATH (Linux)
# -----------------------------------------------------------------------------
set(CARDINAL_RPATH "${LLAMA_BUILD}/lib;${LLAMA_BUILD}/bin")
if(CARDINAL_ENABLE_TENSORRT)
    list(APPEND CARDINAL_RPATH "${TRT_LLM_LIB_DIR}")
endif()

set_target_properties(cardinal PROPERTIES
    BUILD_RPATH   "${CARDINAL_RPATH}"
    INSTALL_RPATH "${CARDINAL_RPATH}"
)

# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------
message(STATUS "=============================================================")
message(STATUS "Cardinal Configuration Summary")
message(STATUS "=============================================================")
message(STATUS "Version:            ${PROJECT_VERSION}")
message(STATUS "Platform:           Linux (only)")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA arch:          ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA version:       ${CUDAToolkit_VERSION}")
message(STATUS "OpenSSL version:    ${OPENSSL_VERSION}")
message(STATUS "Prolog support:     ${CARDINAL_HAS_PROLOG}")
message(STATUS "TensorRT backend:   ${CARDINAL_ENABLE_TENSORRT}")
message(STATUS "Output dir:         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "=============================================================")
message(STATUS "To switch backends at runtime:")
message(STATUS "  Edit config.json -> backend: \"llama_cpp\" | \"tensorrt\"")
message(STATUS "=============================================================")

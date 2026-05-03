# =============================================================================
# Cardinal - Root CMakeLists.txt
# Neurosymbolic AGI Architecture
# =============================================================================
# Dependencies:
#   - llama.cpp         (vendor/llama.cpp, CUDA-built)
#   - nlohmann/json     (FetchContent)
#   - cpp-httplib       (FetchContent, HTTPS via OpenSSL)
#   - OpenSSL           (vcpkg or system)
#   - SQLite3           (vcpkg or system)
#   - SWI-Prolog        (Windows: C:/Program Files/swipl,
#                        Linux:   /usr/lib/swi-prolog or pkg-config)
# =============================================================================

cmake_minimum_required(VERSION 3.28)

# -----------------------------------------------------------------------------
# vcpkg toolchain -- must be set before project()
# Vendored vcpkg: cardinal/vendor/vcpkg/
# Override with -DVCPKG_ROOT=path or set environment variable
# -----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Check for vendored vcpkg first (relative path)
    set(VENDORED_VCPKG "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vcpkg/scripts/buildsystems/vcpkg.cmake")
    if(EXISTS "${VENDORED_VCPKG}")
        set(CMAKE_TOOLCHAIN_FILE "${VENDORED_VCPKG}"
            CACHE STRING "vcpkg toolchain file (vendored)")
        message(STATUS "Using vendored vcpkg at: vendor/vcpkg/")
    elseif(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "vcpkg toolchain file")
        message(STATUS "Using vcpkg from VCPKG_ROOT: $ENV{VCPKG_ROOT}")
    elseif(WIN32)
        # Fallback to common install locations (optional)
        set(FALLBACK_VCPKG "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        if(EXISTS "${FALLBACK_VCPKG}")
            set(CMAKE_TOOLCHAIN_FILE "${FALLBACK_VCPKG}"
                CACHE STRING "vcpkg toolchain file")
            message(STATUS "Using fallback vcpkg at: D:/vcpkg/")
        else()
            message(WARNING "vcpkg not found. Install dependencies manually or set CMAKE_TOOLCHAIN_FILE")
        endif()
    endif()
endif()

# -----------------------------------------------------------------------------
# Project definition
# -----------------------------------------------------------------------------
project(Cardinal
    VERSION 0.1.0
    DESCRIPTION "Neurosymbolic AGI Architecture"
    LANGUAGES CXX CUDA
)

# -----------------------------------------------------------------------------
# C++ standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# CUDA configuration
# RTX 3050 Laptop = compute capability 8.6
# Override on the cmake command line: -DCMAKE_CUDA_ARCHITECTURES=75
# -----------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# -----------------------------------------------------------------------------
# Build type default
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures:${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Platform:          ${CMAKE_SYSTEM_NAME}")

# -----------------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(
        /W4
        /WX-
        /MP
        /utf-8
        /EHsc
        "$<$<CONFIG:Release>:/O2>"
        "$<$<CONFIG:Release>:/GL>"
        "$<$<CONFIG:Debug>:/Od>"
        "$<$<CONFIG:Debug>:/Zi>"
    )
    add_link_options(
        "$<$<CONFIG:Release>:/LTCG>"
    )
else()
    # GCC / Clang (Linux)
    add_compile_options(
        -Wall
        -Wextra
        -Wno-unused-parameter
        "$<$<CONFIG:Release>:-O2>"
        "$<$<CONFIG:Debug>:-g>"
        "$<$<CONFIG:Debug>:-O0>"
    )
endif()

# -----------------------------------------------------------------------------
# FetchContent setup
# -----------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# -----------------------------------------------------------------------------
# nlohmann/json
# -----------------------------------------------------------------------------
message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# -----------------------------------------------------------------------------
# cpp-httplib
# -----------------------------------------------------------------------------
message(STATUS "Fetching cpp-httplib...")
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# -----------------------------------------------------------------------------
# OpenSSL
# Windows: via vcpkg
# Linux:   via system (apt install libssl-dev) or vcpkg
# -----------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
    add_compile_definitions(CPPHTTPLIB_OPENSSL_SUPPORT)
else()
    if(WIN32)
        message(FATAL_ERROR "OpenSSL not found. Run: vcpkg install openssl:x64-windows")
    else()
        message(FATAL_ERROR "OpenSSL not found. Run: sudo apt install libssl-dev")
    endif()
endif()

# -----------------------------------------------------------------------------
# SQLite3
# Windows: via vcpkg (unofficial-sqlite3)
# Linux:   via vcpkg or system (apt install libsqlite3-dev)
# -----------------------------------------------------------------------------
if(WIN32 OR DEFINED CMAKE_TOOLCHAIN_FILE)
    # vcpkg path
    find_package(unofficial-sqlite3 CONFIG QUIET)
    if(TARGET unofficial::sqlite3::sqlite3)
        message(STATUS "SQLite3 found via vcpkg")
        set(SQLITE3_TARGET unofficial::sqlite3::sqlite3)
    endif()
endif()

if(NOT DEFINED SQLITE3_TARGET)
    # System path (Linux apt / brew)
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
endif()

# -----------------------------------------------------------------------------
# llama.cpp (vendor)
# On Linux the build output goes to build/lib or build/bin (no Release subdir)
# -----------------------------------------------------------------------------
set(LLAMA_ROOT ${CMAKE_SOURCE_DIR}/vendor/llama.cpp)
set(LLAMA_BUILD ${LLAMA_ROOT}/build)

set(LLAMA_INCLUDE_DIRS
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
)

if(WIN32)
    set(LLAMA_LIB_SEARCH_PATHS
        ${LLAMA_BUILD}/bin/Release
        ${LLAMA_BUILD}/lib/Release
        ${LLAMA_BUILD}/src/Release
    )
    set(GGML_LIB_SEARCH_PATHS
        ${LLAMA_BUILD}/bin/Release
        ${LLAMA_BUILD}/lib/Release
        ${LLAMA_BUILD}/ggml/src/Release
    )
    set(GGML_CUDA_SEARCH_PATHS
        ${LLAMA_BUILD}/bin/Release
        ${LLAMA_BUILD}/lib/Release
        ${LLAMA_BUILD}/ggml/src/Release
        ${LLAMA_BUILD}/ggml/src/ggml-cuda/Release
    )
    set(LLAMA_COMMON_SEARCH_PATHS
        ${LLAMA_BUILD}/bin/Release
        ${LLAMA_BUILD}/lib/Release
        ${LLAMA_BUILD}/common/Release
    )
else()
    # Linux -- no Release subdirectory
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
endif()

find_library(LLAMA_LIB      NAMES llama    PATHS ${LLAMA_LIB_SEARCH_PATHS}    NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB       NAMES ggml     PATHS ${GGML_LIB_SEARCH_PATHS}     NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB  NAMES ggml-base PATHS ${GGML_LIB_SEARCH_PATHS}   NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB  NAMES ggml-cuda PATHS ${GGML_CUDA_SEARCH_PATHS}  NO_DEFAULT_PATH)
find_library(LLAMA_COMMON_LIB NAMES common PATHS ${LLAMA_COMMON_SEARCH_PATHS} NO_DEFAULT_PATH)

message(STATUS "llama lib:         ${LLAMA_LIB}")
message(STATUS "ggml lib:          ${GGML_LIB}")
message(STATUS "ggml-cuda lib:     ${GGML_CUDA_LIB}")
message(STATUS "llama common lib:  ${LLAMA_COMMON_LIB}")

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
# CUDA toolkit
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# -----------------------------------------------------------------------------
# SWI-Prolog 9.2.9
# Windows: fixed install path at C:/Program Files/swipl
# Linux:   search standard system paths, then pkg-config
# -----------------------------------------------------------------------------
if(WIN32)
    set(SWIPL_ROOT "C:/Program Files/swipl")

    find_path(SWIPL_INCLUDE_DIR
        NAMES SWI-Prolog.h
        PATHS "${SWIPL_ROOT}/include"
        NO_DEFAULT_PATH
        REQUIRED
    )
    find_library(SWIPL_LIB
        NAMES libswipl
        PATHS "${SWIPL_ROOT}/bin"
        NO_DEFAULT_PATH
        REQUIRED
    )
    set(SWIPL_DLL "${SWIPL_ROOT}/bin/libswipl.dll")

else()
    # Linux -- try pkg-config first, then common install paths
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SWIPL QUIET swipl)
    endif()

    if(SWIPL_FOUND)
        message(STATUS "SWI-Prolog found via pkg-config")
        set(SWIPL_INCLUDE_DIR ${SWIPL_INCLUDE_DIRS})
        set(SWIPL_LIB ${SWIPL_LIBRARIES})
    else()
        # Manual search across common Linux install locations
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
endif()

if(SWIPL_INCLUDE_DIR AND SWIPL_LIB)
    message(STATUS "SWI-Prolog include: ${SWIPL_INCLUDE_DIR}")
    message(STATUS "SWI-Prolog lib:     ${SWIPL_LIB}")
    set(CARDINAL_HAS_PROLOG ON)
    set(SWI_PROLOG_INCLUDE_DIRS ${SWIPL_INCLUDE_DIR})
    set(SWI_PROLOG_LIBRARIES ${SWIPL_LIB})
    add_compile_definitions(CARDINAL_PROLOG_ENABLED)
else()
    if(WIN32)
        message(FATAL_ERROR
            "SWI-Prolog not found.\n"
            "Install from https://www.swi-prolog.org/download/stable\n"
            "Expected at: C:/Program Files/swipl")
    else()
        message(FATAL_ERROR
            "SWI-Prolog not found.\n"
            "Install with: sudo apt install swi-prolog\n"
            "Or download from https://www.swi-prolog.org/download/stable")
    endif()
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
    src/core/llm_engine.cpp
    src/core/feeling_output.cpp
    src/core/inference.cpp
)

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
    ${LLAMA_ROOT}/common
    ${CUDAToolkit_INCLUDE_DIRS}
    nlohmann_json::nlohmann_json
)

if(DEFINED SQLITE3_SYSTEM AND SQLITE3_SYSTEM)
    target_include_directories(cardinal PRIVATE ${SQLITE3_INCLUDE_DIRS})
elseif(DEFINED SQLITE3_MANUAL AND SQLITE3_MANUAL)
    target_include_directories(cardinal PRIVATE ${SQLITE3_INCLUDE})
endif()

# -----------------------------------------------------------------------------
# Link libraries
# -----------------------------------------------------------------------------
target_link_libraries(cardinal PRIVATE
    ${LLAMA_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
    httplib::httplib
    OpenSSL::SSL
    OpenSSL::Crypto

    # Windows system libs
    $<$<PLATFORM_ID:Windows>:ws2_32>
    $<$<PLATFORM_ID:Windows>:psapi>

    # Linux system libs
    $<$<PLATFORM_ID:Linux>:pthread>
    $<$<PLATFORM_ID:Linux>:dl>
)

# SQLite3 -- link whichever way it was found
if(DEFINED SQLITE3_TARGET)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_TARGET})
elseif(DEFINED SQLITE3_SYSTEM AND SQLITE3_SYSTEM)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_LIBRARIES})
elseif(DEFINED SQLITE3_MANUAL AND SQLITE3_MANUAL)
    target_link_libraries(cardinal PRIVATE ${SQLITE3_LIB})
endif()

# SWI-Prolog
target_include_directories(cardinal PRIVATE ${SWI_PROLOG_INCLUDE_DIRS})
target_link_libraries(cardinal PRIVATE ${SWI_PROLOG_LIBRARIES})

# -----------------------------------------------------------------------------
# Compile definitions
# -----------------------------------------------------------------------------
target_compile_definitions(cardinal PRIVATE
    CARDINAL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    CARDINAL_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    CARDINAL_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS>
    GGML_USE_CUDA
)

# -----------------------------------------------------------------------------
# Post-build: copy runtime files
# Windows only -- on Linux .so files are found via rpath or ldconfig
# -----------------------------------------------------------------------------
if(WIN32)
    # Copy libswipl.dll
    add_custom_command(TARGET cardinal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SWIPL_DLL}"
            $<TARGET_FILE_DIR:cardinal>
        COMMENT "Copying libswipl.dll"
    )

    # Ensure output directory exists
    add_custom_command(TARGET cardinal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:cardinal>
        COMMENT "Ensuring output directory exists"
    )

    # Copy llama.cpp DLLs
    set(LLAMA_DLL_NAMES
        ggml.dll
        ggml-base.dll
        ggml-cpu.dll
        ggml-cuda.dll
        llama.dll
    )
    foreach(DLL_NAME ${LLAMA_DLL_NAMES})
        set(DLL_PATH "${LLAMA_BUILD}/bin/Release/${DLL_NAME}")
        if(EXISTS "${DLL_PATH}")
            add_custom_command(TARGET cardinal POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL_PATH}"
                    $<TARGET_FILE_DIR:cardinal>
                COMMENT "Copying ${DLL_NAME}"
            )
        endif()
    endforeach()

    # Copy vcpkg OpenSSL DLLs
    # Copy vcpkg OpenSSL DLLs (from vendored vcpkg)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vcpkg/installed/x64-windows/bin")
        set(VCPKG_BIN "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vcpkg/installed/x64-windows/bin")
    elseif(DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_BIN "$ENV{VCPKG_ROOT}/installed/x64-windows/bin")
    else()
        set(VCPKG_BIN "D:/vcpkg/installed/x64-windows/bin")  # fallback
    endif()
    set(OPENSSL_DLLS
        "${VCPKG_BIN}/libssl-3-x64.dll"
        "${VCPKG_BIN}/libcrypto-3-x64.dll"
    )
    foreach(DLL_PATH ${OPENSSL_DLLS})
        if(EXISTS "${DLL_PATH}")
            add_custom_command(TARGET cardinal POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL_PATH}"
                    $<TARGET_FILE_DIR:cardinal>
                COMMENT "Copying OpenSSL DLL: ${DLL_PATH}"
            )
        endif()
    endforeach()

else()
    # Linux: set RPATH so the binary finds llama.cpp .so files at runtime
    # without needing ldconfig or LD_LIBRARY_PATH
    set_target_properties(cardinal PROPERTIES
        BUILD_RPATH "${LLAMA_BUILD}/lib;${LLAMA_BUILD}/bin"
        INSTALL_RPATH "${LLAMA_BUILD}/lib;${LLAMA_BUILD}/bin"
    )
endif()

# -----------------------------------------------------------------------------
# Status summary
# -----------------------------------------------------------------------------
message(STATUS "=============================================================")
message(STATUS "Cardinal Configuration Summary")
message(STATUS "=============================================================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Platform:         ${CMAKE_SYSTEM_NAME}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA arch:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA version:     ${CUDAToolkit_VERSION}")
message(STATUS "OpenSSL version:  ${OPENSSL_VERSION}")
message(STATUS "Prolog support:   ${CARDINAL_HAS_PROLOG}")
message(STATUS "Prolog include:   ${SWI_PROLOG_INCLUDE_DIRS}")
message(STATUS "Output dir:       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "=============================================================")
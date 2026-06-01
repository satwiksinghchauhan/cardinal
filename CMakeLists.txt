# =============================================================================
# Cardinal - Root CMakeLists.txt (v1.6.0)
# Neurosymbolic AGI Architecture
# =============================================================================

cmake_minimum_required(VERSION 3.28)

project(Cardinal
    VERSION 1.6.0
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
    "Build TensorRT-LLM backend" OFF)

option(CARDINAL_ENABLE_VOICE
    "Build voice subsystem (whisper.cpp + piper + portaudio + pocketsphinx)" ON)

# Piper install prefix — flat layout (libs + espeak-ng-data are in this dir directly)
set(PIPER_INSTALL_DIR "${CMAKE_SOURCE_DIR}/vendor/piper/install"
    CACHE PATH "Path to pre-built piper install prefix")

# PocketSphinx install prefix — pre-built separately
set(POCKETSPHINX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/vendor/pocketsphinx/install"
    CACHE PATH "Path to pre-built pocketsphinx install prefix")

message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT backend:   ${CARDINAL_ENABLE_TENSORRT}")
message(STATUS "Voice subsystem:    ${CARDINAL_ENABLE_VOICE}")
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
# FetchContent (offline first)
# -----------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

macro(fetch_if_missing name repo tag)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/vendor/${name}")
        message(STATUS "Fetching ${name} (missing from vendor/${name})")
        FetchContent_Declare(${name}
            GIT_REPOSITORY ${repo}
            GIT_TAG        ${tag}
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(${name})
    else()
        message(STATUS "Using vendored ${name} (vendor/${name})")
        if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/${name}/CMakeLists.txt")
            add_subdirectory("${CMAKE_SOURCE_DIR}/vendor/${name}"
                             "${CMAKE_CURRENT_BINARY_DIR}/vendor-${name}")
        endif()
    endif()
endmacro()

fetch_if_missing(nlohmann_json
    https://github.com/nlohmann/json.git v3.11.3)

fetch_if_missing(cpp-httplib
    https://github.com/yhirose/cpp-httplib.git v0.18.3)

set(ENABLE_SAMPLES OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENMP  OFF CACHE BOOL "" FORCE)
fetch_if_missing(muparser
    https://github.com/beltoforion/muparser.git v2.3.4)

if(CARDINAL_ENABLE_TENSORRT)
    fetch_if_missing(tokenizers_cpp
        https://github.com/mlc-ai/tokenizers-cpp.git main)
    set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
endif()

# -----------------------------------------------------------------------------
# Voice vendor deps (v1.6.0)
# -----------------------------------------------------------------------------
if(CARDINAL_ENABLE_VOICE)

    # -------------------------------------------------------------------------
    # whisper.cpp — USE PRE-BUILT LIBRARY (NO REBUILD)
    # -------------------------------------------------------------------------
    set(WHISPER_ROOT ${CMAKE_SOURCE_DIR}/vendor/whisper.cpp)
    if(NOT EXISTS "${WHISPER_ROOT}")
        message(FATAL_ERROR
            "whisper.cpp not found at ${WHISPER_ROOT}\n"
            "Run: cd vendor && git clone https://github.com/ggerganov/whisper.cpp.git")
    endif()
    
    set(WHISPER_BUILD_DIR "${WHISPER_ROOT}/build")
    set(WHISPER_LIB_DIR "${WHISPER_BUILD_DIR}/src")
    
    if(NOT EXISTS "${WHISPER_LIB_DIR}/libwhisper.so")
        message(FATAL_ERROR
            "Pre-built whisper.cpp not found at ${WHISPER_LIB_DIR}/libwhisper.so\n"
            "Build it first:\n"
            "  cd ${WHISPER_ROOT}\n"
            "  mkdir -p build && cd build\n"
            "  cmake .. -DGGML_CUDA=ON -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF\n"
            "  make -j$(nproc)")
    endif()

    add_library(whisper SHARED IMPORTED)
    set_target_properties(whisper PROPERTIES
        IMPORTED_LOCATION "${WHISPER_LIB_DIR}/libwhisper.so"
        INTERFACE_INCLUDE_DIRECTORIES "${WHISPER_ROOT}"
    )

    # -------------------------------------------------------------------------
    # piper — pre-built
    # -------------------------------------------------------------------------
    set(PIPER_ROOT ${CMAKE_SOURCE_DIR}/vendor/piper)
    if(NOT EXISTS "${PIPER_ROOT}")
        message(FATAL_ERROR
            "piper not found at ${PIPER_ROOT}\n"
            "Run: cd vendor && git clone https://github.com/rhasspy/piper.git")
    endif()

    set(PIPER_SRC_CPP "${PIPER_ROOT}/src/cpp")

    if(NOT EXISTS "${PIPER_INSTALL_DIR}")
        message(FATAL_ERROR
            "Piper install not found at ${PIPER_INSTALL_DIR}\n"
            "Pre-build piper:\n"
            "  cd ${PIPER_ROOT}\n"
            "  cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build --target install -j\$(nproc)")
    endif()

    find_library(PIPER_PHONEMIZE_LIB
        NAMES piper_phonemize
        PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_library(ESPEAK_NG_LIB
        NAMES espeak-ng espeak
        PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH REQUIRED)

    find_library(FMT_LIB    NAMES fmt    PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH)
    find_library(SPDLOG_LIB NAMES spdlog PATHS "${PIPER_INSTALL_DIR}" NO_DEFAULT_PATH)

    # Paths for piper-phonemize headers (from build directory)
    set(PIPER_PHONEMIZE_BUILD_INCLUDE "${CMAKE_SOURCE_DIR}/vendor/piper/build/pi/include")

    message(STATUS "Piper install dir:      ${PIPER_INSTALL_DIR}")
    message(STATUS "piper_phonemize lib:    ${PIPER_PHONEMIZE_LIB}")
    message(STATUS "onnxruntime lib:        ${ONNXRUNTIME_LIB}")
    message(STATUS "espeak-ng lib:          ${ESPEAK_NG_LIB}")

    # -------------------------------------------------------------------------
    # portaudio
    # -------------------------------------------------------------------------
    set(PORTAUDIO_ROOT ${CMAKE_SOURCE_DIR}/vendor/portaudio)
    if(NOT EXISTS "${PORTAUDIO_ROOT}")
        message(FATAL_ERROR
            "portaudio not found at ${PORTAUDIO_ROOT}\n"
            "Run: cd vendor && git clone https://github.com/PortAudio/portaudio.git")
    endif()
    message(STATUS "Using vendored portaudio (${PORTAUDIO_ROOT})")
    set(PA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${PORTAUDIO_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/vendor-portaudio")

    # -------------------------------------------------------------------------
    # pocketsphinx — pre-built
    # -------------------------------------------------------------------------
    set(POCKETSPHINX_ROOT ${CMAKE_SOURCE_DIR}/vendor/pocketsphinx)
    if(NOT EXISTS "${POCKETSPHINX_ROOT}")
        message(FATAL_ERROR
            "pocketsphinx not found at ${POCKETSPHINX_ROOT}\n"
            "Run: cd vendor && git clone https://github.com/cmusphinx/pocketsphinx.git")
    endif()

    if(NOT EXISTS "${POCKETSPHINX_INSTALL_DIR}")
        message(FATAL_ERROR
            "PocketSphinx install not found at ${POCKETSPHINX_INSTALL_DIR}\n"
            "Pre-build pocketsphinx:\n"
            "  cd ${POCKETSPHINX_ROOT}\n"
            "  cmake -B build -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build --target install -j\$(nproc)")
    endif()

    find_library(POCKETSPHINX_LIB
        NAMES pocketsphinx
        PATHS "${POCKETSPHINX_INSTALL_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

    message(STATUS "PocketSphinx install:   ${POCKETSPHINX_INSTALL_DIR}")
    message(STATUS "PocketSphinx lib:       ${POCKETSPHINX_LIB}")

endif()

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL: ${OPENSSL_VERSION}")
add_compile_definitions(CPPHTTPLIB_OPENSSL_SUPPORT)

find_package(PkgConfig QUIET)
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

if(CARDINAL_ENABLE_VOICE)
    find_package(ALSA REQUIRED)
    message(STATUS "ALSA: ${ALSA_VERSION_STRING}")
endif()

# llama.cpp (vendored)
set(LLAMA_ROOT  ${CMAKE_SOURCE_DIR}/vendor/llama.cpp)
if(NOT EXISTS "${LLAMA_ROOT}")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_ROOT}")
endif()

set(LLAMA_BUILD ${LLAMA_ROOT}/build)
set(LLAMA_INCLUDE_DIRS
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
    ${LLAMA_ROOT}/common)
set(LLAMA_SEARCH
    ${LLAMA_BUILD}/bin ${LLAMA_BUILD}/lib
    ${LLAMA_BUILD}/src ${LLAMA_BUILD}/common
    ${LLAMA_BUILD}/ggml/src ${LLAMA_BUILD}/ggml/src/ggml-cuda)

find_library(LLAMA_LIB        NAMES llama     PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB         NAMES ggml      PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB    NAMES ggml-base PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB    NAMES ggml-cuda PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)
find_library(LLAMA_COMMON_LIB NAMES common    PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)
find_library(LLAVA_LIB        NAMES llava     PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)

if(EXISTS "${LLAMA_ROOT}/tools/mtmd/mtmd.h")
    set(CARDINAL_MTMD_AVAILABLE ON)
    find_library(MTMD_LIB NAMES mtmd PATHS ${LLAMA_SEARCH} NO_DEFAULT_PATH)
else()
    set(CARDINAL_MTMD_AVAILABLE OFF)
endif()

set(LLAMA_LIBRARIES ${LLAMA_LIB} ${GGML_LIB})
foreach(L ${GGML_BASE_LIB} ${GGML_CUDA_LIB} ${LLAMA_COMMON_LIB} ${LLAVA_LIB})
    if(L)
        list(APPEND LLAMA_LIBRARIES ${L})
    endif()
endforeach()

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")

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
              /usr/lib/x86_64-linux-gnu /usr/local/lib REQUIRED)
endif()
add_compile_definitions(CARDINAL_PROLOG_ENABLED)

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
    list(APPEND CARDINAL_CORE_SOURCES src/core/backends/tensorrt_backend.cpp)
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
    src/tools/builtin/analyze_image.cpp
)

set(CARDINAL_VISION_SOURCES
    src/vision/vision_cache.cpp
    src/vision/vision_encoder.cpp
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

set(CARDINAL_SELF_MODEL_SOURCES
    src/self_model/self_model.cpp
    src/self_model/meta_cognition.cpp
)

set(CARDINAL_TRAINING_SOURCES
    src/training/training_factory.cpp
    src/training/llama_cpp_trainer.cpp
    src/training/tensorrt_trainer.cpp
    src/training/curriculum_builder.cpp
    src/training/dataset_curator.cpp
    src/training/adapter_evaluator.cpp
    src/training/self_improvement_loop.cpp
)

set(CARDINAL_API_SOURCES
    src/api/cardinal_settings.cpp
    src/api/session.cpp
    src/api/cardinal_api.cpp
    src/api/http_server.cpp
)

set(CARDINAL_SCHEDULER_SOURCES
    src/scheduler/scheduler_store.cpp
    src/scheduler/scheduler_parser.cpp
    src/scheduler/scheduler_engine.cpp
)

set(CARDINAL_COMPUTER_SOURCES
    src/computer/display_detector.cpp
    src/computer/screen_reader.cpp
    src/computer/input_controller.cpp
    src/computer/app_controller.cpp
    src/computer/browser_controller.cpp
    src/computer/shell_executor.cpp
    src/computer/file_manager.cpp
    src/computer/system_controller.cpp
    src/computer/email_controller.cpp
    src/computer/atspi_reader.cpp
)

set(CARDINAL_WATCH_SOURCES
    src/watch/file_watcher.cpp
    src/watch/screen_watcher.cpp
    src/watch/process_watcher.cpp
)

set(CARDINAL_COMPUTER_TOOLS_SOURCES
    src/tools/builtin/computer/tool_screenshot.cpp
    src/tools/builtin/computer/tool_click.cpp
    src/tools/builtin/computer/tool_type_text.cpp
    src/tools/builtin/computer/tool_open_app.cpp
    src/tools/builtin/computer/tool_close_app.cpp
    src/tools/builtin/computer/tool_browser.cpp
    src/tools/builtin/computer/tool_shell_run.cpp
    src/tools/builtin/computer/tool_file_ops.cpp
    src/tools/builtin/computer/tool_system_control.cpp
    src/tools/builtin/computer/tool_email.cpp
    src/tools/builtin/computer/tool_watch_screen.cpp
    src/tools/builtin/computer/tool_schedule_task.cpp
)

set(CARDINAL_VOICE_SOURCES
    src/voice/audio_device.cpp
    src/voice/vad_detector.cpp
    src/voice/stt_engine.cpp
    src/voice/tts_engine.cpp
    src/voice/wake_word_detector.cpp
    src/voice/voice_loop.cpp
    src/tools/builtin/voice/tool_voice_control.cpp
)

if(CARDINAL_ENABLE_VOICE)
    list(APPEND CARDINAL_VOICE_SOURCES
        "${CMAKE_SOURCE_DIR}/vendor/piper/src/cpp/piper.cpp")
endif()

set(CARDINAL_ALL_SOURCES
    ${CARDINAL_UTILS_SOURCES}
    ${CARDINAL_CORE_SOURCES}
    ${CARDINAL_MEMORY_SOURCES}
    ${CARDINAL_VERIFIER_SOURCES}
    ${CARDINAL_TOOLS_SOURCES}
    ${CARDINAL_VISION_SOURCES}
    ${CARDINAL_AGENT_SOURCES}
    ${CARDINAL_EXPLAINABILITY_SOURCES}
    ${CARDINAL_LEARNING_SOURCES}
    ${CARDINAL_SELF_MODEL_SOURCES}
    ${CARDINAL_TRAINING_SOURCES}
    ${CARDINAL_API_SOURCES}
    ${CARDINAL_SCHEDULER_SOURCES}
    ${CARDINAL_COMPUTER_SOURCES}
    ${CARDINAL_WATCH_SOURCES}
    ${CARDINAL_COMPUTER_TOOLS_SOURCES}
    src/main.cpp
)

if(CARDINAL_ENABLE_VOICE)
    list(APPEND CARDINAL_ALL_SOURCES ${CARDINAL_VOICE_SOURCES})
endif()

# -----------------------------------------------------------------------------
# Cardinal executable
# -----------------------------------------------------------------------------
add_executable(cardinal ${CARDINAL_ALL_SOURCES})

target_include_directories(cardinal PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LLAMA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${SWIPL_INCLUDE_DIR}
    src/scheduler
    src/computer
    src/watch
)

if(CARDINAL_ENABLE_TENSORRT)
    target_include_directories(cardinal PRIVATE ${TENSORRT_LLM_INCLUDE})
    target_link_libraries(cardinal PRIVATE tokenizers_cpp)
endif()

if(CARDINAL_MTMD_AVAILABLE)
    target_compile_definitions(cardinal PRIVATE CARDINAL_MTMD_AVAILABLE)
    target_include_directories(cardinal PRIVATE ${LLAMA_ROOT}/tools/mtmd)
    if(MTMD_LIB)
        list(APPEND LLAMA_LIBRARIES ${MTMD_LIB})
    endif()
endif()

# v1.6.0 — voice include dirs (FIXED)
# v1.6.0 — voice include dirs (FIXED)
# v1.6.0 — voice include dirs (FIXED)
# v1.6.0 — voice include dirs (FIXED)
if(CARDINAL_ENABLE_VOICE)
    target_include_directories(cardinal PRIVATE
        # whisper
        "${CMAKE_SOURCE_DIR}/vendor/whisper.cpp/include"
        "${CMAKE_SOURCE_DIR}/vendor/whisper.cpp"
        # piper core
        "${PIPER_SRC_CPP}"
        # piper-phonemize headers (from build directory)
        "${PIPER_PHONEMIZE_BUILD_INCLUDE}"
        # espeak-ng headers
        "${PIPER_INSTALL_DIR}/include"
        # spdlog headers (from piper build)
        "${CMAKE_SOURCE_DIR}/vendor/piper/build/si/include"
        # pocketsphinx
        "${POCKETSPHINX_INSTALL_DIR}/include"
        # ONNX Runtime
        "${CMAKE_SOURCE_DIR}/vendor/onnxruntime/include"
    )
    target_compile_definitions(cardinal PRIVATE
        CARDINAL_VOICE_ENABLED
        PIPER_ESPEAK_DATA_PATH="${PIPER_INSTALL_DIR}/espeak-ng-data"
    )
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
    SQLite::SQLite3
    Threads::Threads
    pthread
    dl
)

if(CARDINAL_ENABLE_TENSORRT)
    target_link_libraries(cardinal PRIVATE
        ${TENSORRT_LLM_LIB} ${NVINFER_LIB} ${CUDNN_LIB}
        CUDA::cuda_driver)
endif()

if(CARDINAL_ENABLE_VOICE)
    target_link_libraries(cardinal PRIVATE
        whisper
        portaudio
        ${POCKETSPHINX_LIB}
        ALSA::ALSA
        ${PIPER_PHONEMIZE_LIB}
        ${ONNXRUNTIME_LIB}
        ${ESPEAK_NG_LIB}
    )
    if(FMT_LIB)
        target_link_libraries(cardinal PRIVATE ${FMT_LIB})
    endif()
    if(SPDLOG_LIB)
        target_link_libraries(cardinal PRIVATE ${SPDLOG_LIB})
    endif()

    set_target_properties(cardinal PROPERTIES
        BUILD_RPATH   "${PIPER_INSTALL_DIR};${POCKETSPHINX_INSTALL_DIR}/lib"
        INSTALL_RPATH "${PIPER_INSTALL_DIR};${POCKETSPHINX_INSTALL_DIR}/lib"
    )
endif()

set(CARDINAL_RPATH "${LLAMA_BUILD}/lib;${LLAMA_BUILD}/bin")
if(CARDINAL_ENABLE_TENSORRT)
    list(APPEND CARDINAL_RPATH "${TRT_LLM_LIB_DIR}")
endif()
set_target_properties(cardinal PROPERTIES
    BUILD_RPATH   "${CARDINAL_RPATH}"
    INSTALL_RPATH "${CARDINAL_RPATH}")

# -----------------------------------------------------------------------------
# Data / model directory creation
# -----------------------------------------------------------------------------
add_custom_command(TARGET cardinal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/data/scheduler"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/data/screenshots"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/data/browser_profile"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/models/voice"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/models/voice/pocketsphinx"
    COMMENT "Creating v1.6.0 data and model directories..."
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "=============================================================")
message(STATUS "Cardinal v${PROJECT_VERSION} Configuration")
message(STATUS "=============================================================")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch:          ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT backend:   ${CARDINAL_ENABLE_TENSORRT}")
message(STATUS "Voice subsystem:    ${CARDINAL_ENABLE_VOICE}")
if(CARDINAL_ENABLE_VOICE)
message(STATUS "Piper install:      ${PIPER_INSTALL_DIR}")
message(STATUS "PocketSphinx:       ${POCKETSPHINX_INSTALL_DIR}")
endif()
message(STATUS "Output dir:         ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "=============================================================")

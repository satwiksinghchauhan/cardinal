// =============================================================================
// Cardinal - Backend Factory Implementation
// File: src/core/backend_factory.cpp
// =============================================================================

#include "core/backend_factory.h"
#include "core/backends/llama_cpp_backend.h"
#include "utils/logger.h"

#ifdef CARDINAL_ENABLE_TENSORRT
#include "core/backends/tensorrt_backend.h"
#endif

#include <stdexcept>

namespace cardinal {

BackendType BackendFactory::resolve_type(const CardinalConfig& config) {
    auto t = backend_type_from_string(config.backend.type);
    if (t == BackendType::UNKNOWN) {
        throw std::runtime_error(
            "Unknown backend type: \"" + config.backend.type + "\". "
            "Valid values: \"llama_cpp\", \"tensorrt\"");
    }
    return t;
}

std::unique_ptr<ILLMBackend> BackendFactory::create(const CardinalConfig& config) {
    BackendType type = resolve_type(config);

    LOG_INFO("BackendFactory: creating backend: " + config.backend.type);

    switch (type) {
        case BackendType::LLAMA_CPP:
            return std::make_unique<LlamaCppBackend>(config);

        case BackendType::TENSORRT:
#ifdef CARDINAL_ENABLE_TENSORRT
            return std::make_unique<TensorRTBackend>(config);
#else
            throw std::runtime_error(
                "Backend \"tensorrt\" requested in config.json but Cardinal was "
                "built without TensorRT support.\n"
                "Rebuild with: cmake -DCARDINAL_ENABLE_TENSORRT=ON ...");
#endif

        default:
            throw std::runtime_error(
                "BackendFactory: unhandled backend type: " + config.backend.type);
    }
}

} // namespace cardinal

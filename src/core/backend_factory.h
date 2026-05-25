#pragma once
// =============================================================================
// Cardinal - Backend Factory
// File: src/core/backend_factory.h
//
// Single point of construction for all ILLMBackend implementations.
// Reads config.backend.type, constructs the right backend, returns it.
//
// This is the ONLY place in the codebase that #includes concrete backend
// headers. Everything else depends only on ILLMBackend.
//
// Usage (in CardinalAPI::init()):
//   backend_ = BackendFactory::create(config);
//   backend_->load_model();
// =============================================================================

#include "core/llm_backend.h"
#include "utils/config_loader.h"

#include <memory>
#include <string>

namespace cardinal {

class BackendFactory {
public:
    // Create and return the backend specified by config.backend.type.
    // Throws std::runtime_error if:
    //   - backend type string is unrecognised
    //   - TENSORRT backend requested but CARDINAL_ENABLE_TENSORRT not defined
    //
    // Does NOT call load_model() — caller is responsible.
    static std::unique_ptr<ILLMBackend> create(const CardinalConfig& config);

    // Convenience: return the type enum for the configured backend
    // without constructing it. Used for early validation / logging.
    static BackendType resolve_type(const CardinalConfig& config);

private:
    BackendFactory() = delete;
};

} // namespace cardinal

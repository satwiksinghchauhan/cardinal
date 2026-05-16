// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Training Factory (Layer 3)
// File: src/training/training_factory.h
//
// Creates the correct ITrainingBackend implementation based on the active
// backend type in CardinalConfig.
//
// Selection logic:
//   config.backend.type == "llama_cpp"  →  LlamaCppTrainer
//   config.backend.type == "tensorrt"   →  TensorRTTrainer
//   anything else                       →  throws TrainingFactoryError
//
// The factory does NOT own the ILLMBackend — it is passed in from CardinalAPI
// which owns all subsystems. LlamaCppTrainer needs the live backend reference
// to call llama_adapter_lora_init() on the running context. TensorRTTrainer
// ignores the backend reference (script-only mode).
//
// Usage:
//   auto trainer = TrainingFactory::create(config, backend);
//   // trainer is std::unique_ptr<ITrainingBackend>
// =============================================================================

#include "training/i_training_backend.h"
#include "utils/config_loader.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace cardinal {

    class ILLMBackend;

    // -------------------------------------------------------------------------
    // TrainingFactory
    // -------------------------------------------------------------------------
    class TrainingFactory {
    public:
        // Creates and returns the appropriate ITrainingBackend.
        // backend is required for LlamaCppTrainer; TensorRTTrainer ignores it
        // but we pass it uniformly so the call site doesn't need to branch.
        // Throws TrainingFactoryError if the backend type is unrecognised or
        // if self_improvement.training.enabled is false.
        static std::unique_ptr<ITrainingBackend> create(
            const CardinalConfig& config,
            ILLMBackend&          backend);

        // Returns the backend type string that would be selected for config,
        // without constructing the object. Useful for logging and health checks.
        static std::string backend_type_for(const CardinalConfig& config);

    private:
        TrainingFactory() = delete;
    };

    // -------------------------------------------------------------------------
    // TrainingFactoryError
    // -------------------------------------------------------------------------
    class TrainingFactoryError : public std::runtime_error {
    public:
        explicit TrainingFactoryError(const std::string& msg)
            : std::runtime_error("TrainingFactoryError: " + msg) {}
    };

} // namespace cardinal

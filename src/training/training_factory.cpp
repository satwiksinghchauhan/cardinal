// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Training Factory — Implementation
// File: src/training/training_factory.cpp
// =============================================================================

#include "training/training_factory.h"
#include "training/llama_cpp_trainer.h"
#include "training/tensorrt_trainer.h"
#include "utils/logger.h"

namespace cardinal {

std::unique_ptr<ITrainingBackend> TrainingFactory::create(
        const CardinalConfig& config,
        ILLMBackend&          backend) {

    if (!config.self_improvement.training.enabled) {
        throw TrainingFactoryError(
            "training is disabled in config (self_improvement.training.enabled=false)");
    }

    const std::string& type = config.backend.type;

    LOG_INFO("TrainingFactory: creating trainer for backend type '" + type + "'");

    if (type == "llama_cpp") {
        return std::make_unique<LlamaCppTrainer>(config, backend);
    }

    if (type == "tensorrt") {
        return std::make_unique<TensorRTTrainer>(config);
    }

    throw TrainingFactoryError(
        "unrecognised backend type '" + type +
        "' — expected 'llama_cpp' or 'tensorrt'");
}

std::string TrainingFactory::backend_type_for(const CardinalConfig& config) {
    return config.backend.type;
}

} // namespace cardinal

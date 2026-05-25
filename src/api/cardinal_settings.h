#pragma once
// =============================================================================
// Cardinal - Settings Manager
// File: src/api/cardinal_settings.h
//
// Manages runtime-mutable settings without requiring restart.
// Changes propagate immediately to the relevant core components.
//
// Relationship to CardinalConfig:
//   CardinalConfig  -- loaded once at startup, never modified at runtime
//   CardinalSettings -- subset of config fields that CAN change at runtime
//
// Settings are validated before applying. Invalid values are rejected
// with a CardinalVoidResult::failure() -- current settings unchanged.
//
// Thread safety:
//   All public methods are protected by a shared_mutex.
//   get() uses shared lock, update() uses exclusive lock.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "api/cardinal_types.h"
#include "utils/config_loader.h"
#include "memory/episodic_retriever.h"
#include "core/inference.h"

#include <string>
#include <shared_mutex>

namespace cardinal {

    // =========================================================================
    // CardinalSettings
    // The runtime-mutable settings snapshot.
    // Returned by get() and accepted by update().
    // All fields are simple value types -- safe for pybind11.
    // =========================================================================
    struct CardinalSettings {
        // Retriever
        std::string retriever_mode = "hybrid";  // keyword|semantic|hybrid
        float       keyword_weight = 0.7f;
        float       semantic_weight = 0.3f;
        int         max_retrieval_results = 5;
        float       min_retrieval_score = 0.1f;

        // Verifier
        std::string verifier_mode = "symbolic"; // symbolic|neural|hybrid
        float       min_rule_confidence = 0.3f;
        float       contradiction_threshold = 0.75f;

        // Inference
        float       temperature = 0.7f;
        float       top_p = 0.9f;
        bool        stream_responses = true;

        // Logging
        std::string log_level = "info"; // trace|debug|info|warn|error|fatal
    };

    // =========================================================================
    // SettingsManager
    //
    // Usage:
    //   SettingsManager settings(config, retriever, pipeline);
    //   auto current = settings.get();
    //   current.retriever_mode = "keyword";
    //   auto result = settings.update(current);
    //   if (!result.ok()) { handle_error(result.error_message); }
    // =========================================================================
    class SettingsManager {
    public:
        // Constructor -- initializes current settings from config
        SettingsManager(const CardinalConfig& config,
            EpisodicRetriever& retriever,
            InferencePipeline& pipeline);

        // -- Read --

        // Get a snapshot of current settings.
        // Thread-safe read -- shared lock.
        CardinalSettings get() const;

        // -- Write --

        // Apply a new settings snapshot.
        // Validates all fields before applying any changes.
        // If validation fails, no changes are made.
        // Thread-safe write -- exclusive lock.
        CardinalVoidResult update(const CardinalSettings& settings);

        // Update a single field by name and string value.
        // Useful for HTTP API and Python bindings where full struct
        // round-trips are cumbersome.
        // Examples:
        //   set("retriever_mode", "keyword")
        //   set("temperature", "0.8")
        //   set("log_level", "debug")
        CardinalVoidResult set(const std::string& key,
            const std::string& value);

        // -- Reset --

        // Reset all settings to the values from the original config.
        CardinalVoidResult reset();

        // -- Serialization --

        // Serialize current settings to JSON string.
        // Used by the HTTP /api/settings GET endpoint.
        std::string to_json() const;

        // Deserialize settings from JSON string.
        // Used by the HTTP /api/settings POST endpoint.
        CardinalResult<CardinalSettings> from_json(const std::string& json_str) const;

    private:
        // -- Validation --
        CardinalVoidResult validate(const CardinalSettings& settings) const;

        // -- Propagation --
        // Apply validated settings to core components.
        // Called inside update() after validation passes.
        void propagate(const CardinalSettings& settings);

        // -- Members --
        const CardinalConfig& config_;       // Original config -- for reset()
        EpisodicRetriever& retriever_;
        InferencePipeline& pipeline_;

        CardinalSettings      current_;
        mutable std::shared_mutex mutex_;
    };

} // namespace cardinal
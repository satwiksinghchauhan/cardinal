// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Neural Verifier
// File: src/verifier/neural_verifier.h
// A dedicated GGUF model used as a neural verification layer.
// Runs independently of the main LLM — separate model, separate context.
// Defaults to CPU inference (neural_gpu_layers=0) so it doesn't compete
// with the main model for VRAM on constrained hardware.
//
// The neural verifier takes a structured prompt describing a rule candidate
// and outputs a JSON assessment: contradiction score, quality score, and
// reasoning. This complements the symbolic engine with pattern-based
// verification that works on fuzzy/ambiguous domains.
//
// Enabled when:
//   config.verifier.mode == "neural" || config.verifier.mode == "hybrid"
//   AND config.verifier.neural_model_path is a valid GGUF file
//
// If neural_model_path is empty or missing, NeuralVerifier reports
// is_available() == false and ConsistencyChecker falls back to symbolic.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "utils/json_parser.h"

#include "llama.h"

#include <string>
#include <mutex>
#include <optional>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // NeuralVerificationResult
    // Output of a single neural verification call.
    // -----------------------------------------------------------------------------
    struct NeuralVerificationResult {
        bool        available;              // Was the neural verifier able to run?
        float       contradiction_score;    // 0.0 = no contradiction, 1.0 = definite
        float       rule_quality_score;     // 0.0 = poor rule, 1.0 = excellent rule
        bool        contradiction_detected; // contradiction_score >= threshold
        std::string reasoning;             // Neural model's explanation
        std::string raw_output;            // Raw JSON from model (for debugging)
        std::string error_message;         // Set if available == false
    };

    // -----------------------------------------------------------------------------
    // NeuralVerifier
    // Wraps a GGUF model for neural verification.
    // Uses a dedicated llama_model + llama_context, separate from the main engine.
    // Thread-safe via internal mutex.
    // -----------------------------------------------------------------------------
    class NeuralVerifier {
    public:
        explicit NeuralVerifier(const CardinalConfig& config);
        ~NeuralVerifier();

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        // Load the neural verifier model
        // Safe to call even if neural_model_path is empty — sets available=false
        void load();

        // Unload and free resources
        void unload();

        // Check if neural verifier is loaded and ready
        bool is_available() const { return available_; }

        // -------------------------------------------------------------------------
        // Verification
        // -------------------------------------------------------------------------

        // Check if a rule candidate contradicts existing rules
        // Prompt is built from domain + condition + consequence + existing_rules_summary
        NeuralVerificationResult verify_rule_candidate(
            const std::string& domain,
            const std::string& condition,
            const std::string& consequence,
            const std::vector<Rule>& existing_rules) const;

        // Check if a specific claim is consistent with a set of rules
        NeuralVerificationResult verify_claim(
            const std::string& domain,
            const std::string& claim,
            const std::vector<Rule>& existing_rules) const;

        // -------------------------------------------------------------------------
        // Model info
        // -------------------------------------------------------------------------
        std::string model_name() const;
        bool        is_cpu_only() const {
            return config_.verifier.neural_gpu_layers == 0;
        }

        // Disable copy/move
        NeuralVerifier(const NeuralVerifier&) = delete;
        NeuralVerifier& operator=(const NeuralVerifier&) = delete;
        NeuralVerifier(NeuralVerifier&&) = delete;
        NeuralVerifier& operator=(NeuralVerifier&&) = delete;

    private:
        // -------------------------------------------------------------------------
        // Internal generation
        // -------------------------------------------------------------------------

        // Run inference with the verifier model
        std::string generate(const std::string& prompt,
            int max_tokens = 256) const;

        // Build verification prompt
        std::string build_verification_prompt(
            const std::string& domain,
            const std::string& condition,
            const std::string& consequence,
            const std::vector<Rule>& existing_rules) const;

        // Build claim check prompt
        std::string build_claim_prompt(
            const std::string& domain,
            const std::string& claim,
            const std::vector<Rule>& existing_rules) const;

        // Parse JSON output from model into NeuralVerificationResult
        NeuralVerificationResult parse_output(const std::string& raw) const;

        // Format existing rules for prompt injection (truncated)
        std::string format_rules_for_prompt(const std::vector<Rule>& rules,
            int max_rules = 5) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        llama_model* model_ = nullptr;
        llama_context* ctx_ = nullptr;
        bool                   available_ = false;
        mutable std::mutex     mutex_;
    };

    // -----------------------------------------------------------------------------
    // NeuralVerifierError
    // -----------------------------------------------------------------------------
    class NeuralVerifierError : public std::runtime_error {
    public:
        explicit NeuralVerifierError(const std::string& message)
            : std::runtime_error("NeuralVerifierError: " + message) {}
    };

} // namespace cardinal
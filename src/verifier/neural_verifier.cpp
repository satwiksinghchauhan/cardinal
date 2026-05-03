// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Neural Verifier Implementation
// File: src/verifier/neural_verifier.cpp
// =============================================================================

#include "neural_verifier.h"
#include "utils/logger.h"

#include <filesystem>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace cardinal {

    // =============================================================================
    // System prompt for the neural verifier model
    // Instructs any GGUF model to act as a logical consistency checker.
    // Output MUST be JSON — this is enforced by the prompt, not by grammar
    // (grammar sampler is not used here to keep it model-agnostic).
    // =============================================================================

    static constexpr const char* VERIFIER_SYSTEM_PROMPT =
        "You are a logical consistency checker. "
        "You analyze rule candidates and existing rules for contradictions. "
        "You MUST respond with a valid JSON object and nothing else. "
        "The JSON must have exactly these fields: "
        "\"contradiction_score\" (float 0.0-1.0, where 1.0 = definite contradiction), "
        "\"rule_quality_score\" (float 0.0-1.0, where 1.0 = excellent rule), "
        "\"reasoning\" (string, brief explanation under 100 words). "
        "Do not include any text outside the JSON object.";

    // =============================================================================
    // Constructor / Destructor
    // =============================================================================

    NeuralVerifier::NeuralVerifier(const CardinalConfig& config)
        : config_(config)
    {
        if (config_.verifier.neural_model_path.empty()) {
            LOG_INFO("NeuralVerifier: no model path configured — disabled");
        }
        else {
            LOG_INFO("NeuralVerifier: model path: " +
                config_.verifier.neural_model_path +
                " (gpu_layers=" +
                std::to_string(config_.verifier.neural_gpu_layers) + ")");
        }
    }

    NeuralVerifier::~NeuralVerifier() {
        unload();
    }

    // =============================================================================
    // load
    // =============================================================================

    void NeuralVerifier::load() {
        std::lock_guard<std::mutex> lock(mutex_);

        const std::string& path = config_.verifier.neural_model_path;

        // Silent no-op if not configured
        if (path.empty()) {
            LOG_INFO("NeuralVerifier: neural_model_path empty — verifier disabled");
            available_ = false;
            return;
        }

        if (!std::filesystem::exists(path)) {
            LOG_WARN("NeuralVerifier: model file not found: " + path +
                " — neural verifier disabled");
            available_ = false;
            return;
        }

        LOG_INFO("NeuralVerifier: loading model: " + path);

        // Model parameters
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.verifier.neural_gpu_layers;

        model_ = llama_model_load_from_file(path.c_str(), model_params);
        if (!model_) {
            LOG_WARN("NeuralVerifier: failed to load model — neural verifier disabled");
            available_ = false;
            return;
        }

        // Context parameters — small context, CPU-friendly
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;  // Small context — verification prompts are short
        ctx_params.n_threads = config_.backend.llama_cpp.threads;
        ctx_params.n_threads_batch = config_.backend.llama_cpp.threads;

        ctx_ = llama_init_from_model(model_, ctx_params);
        if (!ctx_) {
            llama_model_free(model_);
            model_ = nullptr;
            available_ = false;
            LOG_WARN("NeuralVerifier: failed to create context — neural verifier disabled");
            return;
        }

        available_ = true;
        LOG_INFO("NeuralVerifier: ready — " + model_name() +
            (is_cpu_only() ? " (CPU)" : " (GPU)"));
    }

    // =============================================================================
    // unload
    // =============================================================================

    void NeuralVerifier::unload() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!available_) return;

        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }

        available_ = false;
        LOG_INFO("NeuralVerifier: unloaded");
    }

    // =============================================================================
    // verify_rule_candidate
    // =============================================================================

    NeuralVerificationResult NeuralVerifier::verify_rule_candidate(
        const std::string& domain,
        const std::string& condition,
        const std::string& consequence,
        const std::vector<Rule>& existing_rules) const
    {
        if (!available_) {
            NeuralVerificationResult r;
            r.available = false;
            r.error_message = "Neural verifier not available";
            return r;
        }

        std::string prompt = build_verification_prompt(
            domain, condition, consequence, existing_rules);

        std::string raw = generate(prompt, config_.verifier.neural_max_tokens);
        auto result = parse_output(raw);
        result.available = true;

        LOG_DEBUG("NeuralVerifier: contradiction_score=" +
            std::to_string(result.contradiction_score) +
            " quality_score=" +
            std::to_string(result.rule_quality_score));

        return result;
    }

    // =============================================================================
    // verify_claim
    // =============================================================================

    NeuralVerificationResult NeuralVerifier::verify_claim(
        const std::string& domain,
        const std::string& claim,
        const std::vector<Rule>& existing_rules) const
    {
        if (!available_) {
            NeuralVerificationResult r;
            r.available = false;
            r.error_message = "Neural verifier not available";
            return r;
        }

        std::string prompt = build_claim_prompt(domain, claim, existing_rules);
        std::string raw = generate(prompt, config_.verifier.neural_max_tokens);
        auto result = parse_output(raw);
        result.available = true;

        return result;
    }

    // =============================================================================
    // generate — internal inference
    // =============================================================================

    std::string NeuralVerifier::generate(const std::string& prompt,
        int max_tokens) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!available_) return "";

        const llama_vocab* vocab = llama_model_get_vocab(model_);

        // Tokenize
        int n_prompt = llama_tokenize(
            vocab, prompt.c_str(), static_cast<int>(prompt.size()),
            nullptr, 0, true, true);
        if (n_prompt < 0) n_prompt = -n_prompt;

        std::vector<llama_token> tokens(n_prompt);
        llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()),
            tokens.data(), n_prompt, true, true);

        // Clear KV cache for clean verification context
        llama_memory_clear(llama_get_memory(ctx_), false);

        // Decode prompt
        llama_batch batch = llama_batch_get_one(
            tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(ctx_, batch) != 0) {
            LOG_WARN("NeuralVerifier: prompt decode failed");
            return "";
        }

        // Build greedy sampler (deterministic for verification)
        llama_sampler* sampler = llama_sampler_chain_init(
            llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler,
            llama_sampler_init_temp(0.1f)); // Near-greedy
        llama_sampler_chain_add(sampler,
            llama_sampler_init_dist(42));

        // Generate
        std::string output;
        output.reserve(max_tokens * 4);

        for (int i = 0; i < max_tokens; ++i) {
            llama_token token_id = llama_sampler_sample(sampler, ctx_, -1);

            if (llama_vocab_is_eog(vocab, token_id)) break;

            char buf[256] = {};
            int len = llama_token_to_piece(vocab, token_id, buf,
                sizeof(buf) - 1, 0, true);
            if (len > 0) output.append(buf, len);

            // Stop at closing brace — we only need the JSON object
            if (!output.empty() && output.back() == '}') break;

            // Decode next
            llama_batch next = llama_batch_get_one(&token_id, 1);
            if (llama_decode(ctx_, next) != 0) break;
        }

        llama_sampler_free(sampler);
        return output;
    }

    // =============================================================================
    // build_verification_prompt
    // =============================================================================

    std::string NeuralVerifier::build_verification_prompt(
        const std::string& domain,
        const std::string& condition,
        const std::string& consequence,
        const std::vector<Rule>& existing_rules) const
    {
        std::ostringstream oss;

        oss << VERIFIER_SYSTEM_PROMPT << "\n\n";
        oss << "DOMAIN: " << domain << "\n";
        oss << "PROPOSED RULE:\n";
        oss << "  Condition: " << condition << "\n";
        oss << "  Consequence: " << consequence << "\n\n";

        if (!existing_rules.empty()) {
            oss << "EXISTING RULES IN THIS DOMAIN:\n";
            oss << format_rules_for_prompt(existing_rules);
            oss << "\n";
        }
        else {
            oss << "EXISTING RULES: none\n\n";
        }

        oss << "Does the proposed rule contradict any existing rules? "
            "Assess its logical quality. Respond with JSON only:";

        return oss.str();
    }

    // =============================================================================
    // build_claim_prompt
    // =============================================================================

    std::string NeuralVerifier::build_claim_prompt(
        const std::string& domain,
        const std::string& claim,
        const std::vector<Rule>& existing_rules) const
    {
        std::ostringstream oss;

        oss << VERIFIER_SYSTEM_PROMPT << "\n\n";
        oss << "DOMAIN: " << domain << "\n";
        oss << "CLAIM TO CHECK: " << claim << "\n\n";

        if (!existing_rules.empty()) {
            oss << "EXISTING RULES:\n";
            oss << format_rules_for_prompt(existing_rules);
            oss << "\n";
        }

        oss << "Is this claim consistent with the existing rules? "
            "Respond with JSON only:";

        return oss.str();
    }

    // =============================================================================
    // parse_output
    // =============================================================================

    NeuralVerificationResult NeuralVerifier::parse_output(
        const std::string& raw) const
    {
        NeuralVerificationResult result;
        result.available = true;
        result.contradiction_score = 0.0f;
        result.rule_quality_score = 0.5f;
        result.contradiction_detected = false;
        result.raw_output = raw;

        if (raw.empty()) {
            result.error_message = "Empty output from neural verifier";
            result.available = false;
            return result;
        }

        // Find JSON object in output
        size_t start = raw.find('{');
        size_t end = raw.rfind('}');

        if (start == std::string::npos || end == std::string::npos) {
            result.error_message = "No JSON object in neural verifier output";
            LOG_WARN("NeuralVerifier: failed to find JSON in output: " + raw);
            return result;
        }

        std::string json_str = raw.substr(start, end - start + 1);

        try {
            auto j = json::parse(json_str);

            if (j.contains("contradiction_score") &&
                j["contradiction_score"].is_number()) {
                result.contradiction_score =
                    std::clamp(j["contradiction_score"].get<float>(), 0.0f, 1.0f);
            }

            if (j.contains("rule_quality_score") &&
                j["rule_quality_score"].is_number()) {
                result.rule_quality_score =
                    std::clamp(j["rule_quality_score"].get<float>(), 0.0f, 1.0f);
            }

            if (j.contains("reasoning") && j["reasoning"].is_string()) {
                result.reasoning = j["reasoning"].get<std::string>();
            }

            result.contradiction_detected =
                result.contradiction_score >=
                config_.verifier.contradiction_threshold;

        }
        catch (const json::exception& e) {
            result.error_message = "JSON parse error: " + std::string(e.what());
            LOG_WARN("NeuralVerifier: JSON parse failed: " + result.error_message);
        }

        return result;
    }

    // =============================================================================
    // format_rules_for_prompt
    // =============================================================================

    std::string NeuralVerifier::format_rules_for_prompt(
        const std::vector<Rule>& rules, int max_rules) const
    {
        std::ostringstream oss;
        int count = 0;

        for (const auto& rule : rules) {
            if (count >= max_rules) break;
            oss << "  - IF \"" << rule.condition
                << "\" THEN \"" << rule.consequence
                << "\" (confidence: "
                << static_cast<int>(rule.confidence * 100) << "%)\n";
            ++count;
        }

        return oss.str();
    }

    // =============================================================================
    // model_name
    // =============================================================================

    std::string NeuralVerifier::model_name() const {
        if (!model_) return "none";
        std::string desc(256, '\0');
        llama_model_desc(model_, desc.data(), 256);
        return desc.c_str();
    }

} // namespace cardinal
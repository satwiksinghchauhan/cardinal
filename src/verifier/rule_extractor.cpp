// =============================================================================
// Cardinal - Rule Extractor Implementation
// File: src/verifier/rule_extractor.cpp
// =============================================================================

#include "rule_extractor.h"
#include "utils/logger.h"

#include <algorithm>
#include <sstream>
#include <regex>
#include <cctype>

namespace cardinal {

    // =============================================================================
    // Constructor
    // =============================================================================

    RuleExtractor::RuleExtractor(const CardinalConfig& config,
        RuleStore& rule_store,
        SymbolicEngine& symbolic_engine)
        : config_(config)
        , rule_store_(rule_store)
        , symbolic_engine_(symbolic_engine)
    {
        LOG_INFO("RuleExtractor initialized");
    }

    // =============================================================================
    // sync_rules_to_prolog
    // Load all rules from rule_store into the Prolog engine at startup.
    // =============================================================================

    void RuleExtractor::sync_rules_to_prolog() {
        if (!symbolic_engine_.is_ready()) {
            LOG_WARN("RuleExtractor: symbolic engine not ready, skipping sync");
            return;
        }

        symbolic_engine_.retract_all_rules();
        auto rules = rule_store_.get_all();
        int  n = symbolic_engine_.assert_rules(rules);
        LOG_INFO("RuleExtractor: synced " + std::to_string(n) +
            " rules to Prolog engine");
    }

    // =============================================================================
    // extract Ã¢â‚¬â€ main entry point
    // =============================================================================

    ExtractionResult RuleExtractor::extract(const ExtractionInput& input) {
        std::lock_guard<std::mutex> lock(mutex_);

        ExtractionResult result;
        result.extracted = false;
        result.committed = false;
        result.contradiction_found = false;

        if (!input.feeling.rule_candidate_signal) {
            return result;
        }

        ++total_extracted_;

        // -------------------------------------------------------------------------
        // Step 1: Extract candidate
        // -------------------------------------------------------------------------
        std::optional<CandidateRule> candidate;
        try {
            candidate = extract_candidate(input);
        }
        catch (const std::exception& e) {
            LOG_WARN("RuleExtractor: extract_candidate threw: " +
                std::string(e.what()));
            result.rejection_reason = "extraction exception: " + std::string(e.what());
            ++total_rejected_;
            return result;
        }
        catch (...) {
            LOG_WARN("RuleExtractor: extract_candidate threw unknown exception");
            result.rejection_reason = "extraction unknown exception";
            ++total_rejected_;
            return result;
        }

        if (!candidate.has_value()) {
            result.rejection_reason = "No extractable rule pattern found in response";
            ++total_rejected_;
            LOG_DEBUG("RuleExtractor: no candidate found");
            return result;
        }

        result.extracted = true;
        result.candidate = candidate;

        LOG_DEBUG("RuleExtractor: candidate ["
            + candidate->domain + "] "
            + candidate->condition.substr(0, std::min((size_t)60,
                candidate->condition.size())));

        // -------------------------------------------------------------------------
        // Step 2: Validate
        // -------------------------------------------------------------------------
        std::string rejection_reason;
        if (!validate_candidate(*candidate, rejection_reason)) {
            result.rejection_reason = rejection_reason;
            ++total_rejected_;
            LOG_DEBUG("RuleExtractor: rejected Ã¢â‚¬â€ " + rejection_reason);
            return result;
        }

        // -------------------------------------------------------------------------
        // Step 3: Contradiction check
        // -------------------------------------------------------------------------
        if (symbolic_engine_.is_ready()) {
            try {
                auto contradiction = check_contradictions(*candidate);
                if (contradiction.has_contradiction) {
                    result.contradiction_found = true;
                    result.contradiction = contradiction;
                    ++total_contradicted_;
                    result.rejection_reason = "Contradicts rule: " +
                        contradiction.rule_id_a;
                    LOG_WARN("RuleExtractor: contradiction Ã¢â‚¬â€ " +
                        result.rejection_reason);
                    return result;
                }
            }
            catch (const std::exception& e) {
                LOG_WARN("RuleExtractor: contradiction check threw: " +
                    std::string(e.what()));
                // Don't abort Ã¢â‚¬â€ proceed without contradiction check
            }
        }

        // -------------------------------------------------------------------------
        // Step 4: Commit
        // -------------------------------------------------------------------------
        try {
            std::string rule_id = commit_rule(*candidate);
            if (!rule_id.empty()) {
                result.committed = true;
                result.committed_rule_id = rule_id;
                ++total_committed_;
                LOG_INFO("RuleExtractor: committed [" +
                    candidate->domain + "] id=" + rule_id);
            }
            else {
                result.rejection_reason = "Rule store rejected (capacity or similarity)";
                ++total_rejected_;
            }
        }
        catch (const std::exception& e) {
            LOG_WARN("RuleExtractor: commit threw: " + std::string(e.what()));
            result.rejection_reason = "commit exception: " + std::string(e.what());
            ++total_rejected_;
        }

        return result;
    }

    // =============================================================================
    // extract_candidate
    // Tries multiple extraction strategies in priority order.
    // =============================================================================

    std::optional<CandidateRule> RuleExtractor::extract_candidate(
        const ExtractionInput& input) const
    {
        const std::string& domain = input.feeling.reasoning_domain;
        const std::string& text = input.response_text;

        std::optional<CandidateRule> candidate;

        // Strategy 1: Causal patterns
        if (input.feeling.reasoning_type == "causal" ||
            input.feeling.reasoning_type == "deductive") {
            candidate = extract_causal(text, domain);
            if (candidate.has_value()) {
                candidate->episode_id = input.episode_id;
                candidate->reasoning_type = input.feeling.reasoning_type;
                return candidate;
            }
        }

        // Strategy 2: Deductive patterns
        if (input.feeling.reasoning_type == "deductive" ||
            input.feeling.reasoning_type == "inductive") {
            candidate = extract_deductive(text, domain);
            if (candidate.has_value()) {
                candidate->episode_id = input.episode_id;
                candidate->reasoning_type = input.feeling.reasoning_type;
                return candidate;
            }
        }

        // Strategy 3: Declarative fallback
        candidate = extract_declarative(text, input.user_message, domain);
        if (candidate.has_value()) {
            candidate->episode_id = input.episode_id;
            candidate->reasoning_type = input.feeling.reasoning_type;
        }
        return candidate;
    }

    // =============================================================================
    // extract_causal
    // Matches: "if X then Y", "X causes Y", "X leads to Y", "X results in Y",
    //          "when X, Y", "X produces Y", "X implies Y"
    // =============================================================================

    std::optional<CandidateRule> RuleExtractor::extract_causal(
        const std::string& text,
        const std::string& domain) const
    {
        auto sentences = split_sentences(text);

        // Causal connective patterns
        const std::vector<std::pair<std::string, std::string>> patterns = {
            {"if ",        " then "},
            {"when ",      ", "},
            {"whenever ",  ", "},
            {"because ",   ", "},
            {" causes ",   ""},
            {" leads to ", ""},
            {" results in ", ""},
            {" produces ", ""},
            {" implies ",  ""},
            {" therefore ", ""},
            {" thus ",     ""},
        };

        for (const auto& sentence : sentences) {
            std::string lower = sentence;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            for (const auto& [ante, cons] : patterns) {
                size_t ante_pos = lower.find(ante);
                if (ante_pos == std::string::npos) continue;

                std::string condition;
                std::string consequence;

                if (!cons.empty()) {
                    size_t cons_pos = lower.find(cons, ante_pos + ante.size());
                    if (cons_pos == std::string::npos) continue;

                    // Guard against wraparound
                    size_t cond_start = ante_pos + ante.size();
                    if (cons_pos <= cond_start) continue;

                    condition = sentence.substr(cond_start, cons_pos - cond_start);
                    size_t cons_end = cons_pos + cons.size();
                    if (cons_end >= sentence.size()) continue;
                    consequence = sentence.substr(cons_end);
                }
                else {
                    // Split around the connective itself
                    if (ante_pos + ante.size() >= sentence.size()) continue;
                    condition = sentence.substr(0, ante_pos);
                    consequence = sentence.substr(ante_pos + ante.size());
                }

                condition = clean_text(condition);
                consequence = clean_text(consequence);

                if (condition.size() > 10 && consequence.size() > 10) {
                    CandidateRule candidate;
                    candidate.domain = domain;
                    candidate.condition = condition;
                    candidate.consequence = consequence;
                    candidate.confidence = 0.6f;  // Moderate initial confidence
                    candidate.extraction_method = "causal_pattern:" + ante;
                    candidate.source_sentence = clean_text(sentence, 300);
                    return candidate;
                }
            }
        }

        return std::nullopt;
    }

    // =============================================================================
    // extract_deductive
    // Matches conclusion sentences: "therefore", "thus", "it follows that",
    // "we can conclude", "this means", "this shows"
    // =============================================================================

    std::optional<CandidateRule> RuleExtractor::extract_deductive(
        const std::string& text,
        const std::string& domain) const
    {
        auto sentences = split_sentences(text);
        if (sentences.size() < 2) return std::nullopt;

        const std::vector<std::string> conclusion_markers = {
            "therefore", "thus", "hence", "consequently",
            "it follows that", "we can conclude",
            "this means", "this shows", "this demonstrates",
            "in conclusion", "as a result"
        };

        for (size_t i = 1; i < sentences.size(); ++i) {
            std::string lower = sentences[i];
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            for (const auto& marker : conclusion_markers) {
                if (lower.find(marker) != std::string::npos) {
                    // Condition = previous sentence, Consequence = this sentence
                    std::string condition = clean_text(sentences[i - 1]);
                    std::string consequence = clean_text(sentences[i]);

                    // Strip the marker from the consequence
                    size_t marker_pos = consequence.find(marker);
                    if (marker_pos != std::string::npos) {
                        consequence = clean_text(
                            consequence.substr(marker_pos + marker.size()));
                    }

                    if (condition.size() > 10 && consequence.size() > 10) {
                        CandidateRule candidate;
                        candidate.domain = domain;
                        candidate.condition = condition;
                        candidate.consequence = consequence;
                        candidate.confidence = 0.65f;
                        candidate.extraction_method = "deductive:" + marker;
                        candidate.source_sentence = condition + " -> " + consequence;
                        return candidate;
                    }
                }
            }
        }

        return std::nullopt;
    }

    // =============================================================================
    // extract_declarative
    // Fallback: uses user_message as condition context,
    // extracts the main factual claim from the response.
    // =============================================================================

    std::optional<CandidateRule> RuleExtractor::extract_declarative(
        const std::string& text,
        const std::string& user_message,
        const std::string& domain) const
    {
        auto sentences = split_sentences(text);
        if (sentences.empty()) return std::nullopt;

        // Find the most substantive sentence (longest, not a question, not a heading)
        std::string best_sentence;
        size_t      best_len = 0;

        for (const auto& s : sentences) {
            if (s.empty() || s.back() == '?' || s.size() < 20) continue;
            // Skip markdown headers
            if (s.front() == '#') continue;
            // Skip list items
            if (s.front() == '-' || s.front() == '*') continue;

            if (s.size() > best_len) {
                best_len = s.size();
                best_sentence = s;
            }
        }

        if (best_sentence.empty()) return std::nullopt;

        // Use cleaned user message as condition, best sentence as consequence
        std::string condition = clean_text(user_message, 150);
        std::string consequence = clean_text(best_sentence, 200);

        if (condition.size() < 5 || consequence.size() < 10) return std::nullopt;

        CandidateRule candidate;
        candidate.domain = domain;
        candidate.condition = condition;
        candidate.consequence = consequence;
        candidate.confidence = 0.5f;  // Lower confidence for fallback
        candidate.extraction_method = "declarative_fallback";
        candidate.source_sentence = best_sentence;
        return candidate;
    }

    // =============================================================================
    // validate_candidate
    // =============================================================================

    bool RuleExtractor::validate_candidate(const CandidateRule& candidate,
        std::string& rejection_reason) const {
        // Non-empty fields
        if (candidate.condition.empty()) {
            rejection_reason = "Empty condition";
            return false;
        }
        if (candidate.consequence.empty()) {
            rejection_reason = "Empty consequence";
            return false;
        }
        if (candidate.domain.empty()) {
            rejection_reason = "Empty domain";
            return false;
        }

        // Minimum length
        if (candidate.condition.size() < 5) {
            rejection_reason = "Condition too short: " + candidate.condition;
            return false;
        }
        if (candidate.consequence.size() < 5) {
            rejection_reason = "Consequence too short: " + candidate.consequence;
            return false;
        }

        // Not identical
        if (candidate.condition == candidate.consequence) {
            rejection_reason = "Condition and consequence are identical";
            return false;
        }

        // Confidence above minimum threshold
        if (candidate.confidence < config_.verifier.min_rule_confidence) {
            rejection_reason = "Confidence " +
                std::to_string(candidate.confidence) +
                " below minimum " +
                std::to_string(config_.verifier.min_rule_confidence);
            return false;
        }

        return true;
    }

    // =============================================================================
    // check_contradictions
    // =============================================================================

    ContradictionResult RuleExtractor::check_contradictions(
        const CandidateRule& candidate) const
    {
        return symbolic_engine_.check_contradiction(
            candidate.domain,
            candidate.condition,
            candidate.consequence);
    }

    // =============================================================================
    // commit_rule
    // Writes confirmed rule to rule_store and asserts into Prolog.
    // =============================================================================

    std::string RuleExtractor::commit_rule(const CandidateRule& candidate) {
        // Write to rule store -- pass provenance fields
        std::string rule_id = rule_store_.add_rule(
            candidate.domain,
            candidate.condition,
            candidate.consequence,
            candidate.confidence,
            candidate.episode_id,       // Phase 6: provenance
            candidate.reasoning_type);  // Phase 6: provenance

        if (rule_id.empty()) return "";

        // Assert into Prolog if engine is ready
        if (symbolic_engine_.is_ready()) {
            auto rule_opt = rule_store_.get_rule(rule_id);
            if (rule_opt.has_value()) {
                auto result = symbolic_engine_.assert_rule(rule_opt.value());
                if (!result.success) {
                    LOG_WARN("RuleExtractor: rule committed to store but "
                        "Prolog assertion failed: " + result.error_message);
                }
            }
        }

        return rule_id;
    }

    // =============================================================================
    // Text processing utilities
    // =============================================================================

    std::vector<std::string> RuleExtractor::split_sentences(
        const std::string& text) const
    {
        std::vector<std::string> sentences;
        std::string current;
        current.reserve(256);

        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            current += c;

            // Sentence boundary detection
            if (c == '.' || c == '!' || c == '?') {
                // Make sure it's not a decimal or abbreviation
                bool is_boundary = true;
                if (c == '.' && i + 1 < text.size()) {
                    char next = text[i + 1];
                    if (std::isdigit(next)) is_boundary = false; // decimal
                    if (std::isupper(next) && i > 0 &&
                        std::isupper(text[i - 1])) is_boundary = false; // abbreviation
                }

                if (is_boundary) {
                    std::string trimmed = current;
                    // Trim leading whitespace
                    size_t start = trimmed.find_first_not_of(" \t\r\n");
                    if (start == std::string::npos) {
                        current.clear();
                        continue;
                    }
                    trimmed = trimmed.substr(start);
                    size_t end = trimmed.find_last_not_of(" \t\r\n");
                    if (end != std::string::npos) trimmed = trimmed.substr(0, end + 1);

                    if (!trimmed.empty()) {
                        sentences.push_back(trimmed);
                    }
                    current.clear();
                }
            }
            else if (c == '\n') {
                // Newlines can also be sentence boundaries in structured text
                std::string trimmed = current;
                size_t start = trimmed.find_first_not_of(" \t\r\n");
                if (start != std::string::npos) {
                    trimmed = trimmed.substr(start);
                    size_t end = trimmed.find_last_not_of(" \t\r\n");
                    if (end != std::string::npos) trimmed = trimmed.substr(0, end + 1);
                    if (trimmed.size() > 20) {
                        sentences.push_back(trimmed);
                        current.clear();
                    }
                }
            }
        }

        // Remainder
        if (!current.empty()) {
            std::string trimmed = current;
            size_t start = trimmed.find_first_not_of(" \t\r\n");
            if (start != std::string::npos) {
                trimmed = trimmed.substr(start);
                size_t end = trimmed.find_last_not_of(" \t\r\n");
                if (end != std::string::npos) trimmed = trimmed.substr(0, end + 1);
                if (!trimmed.empty()) sentences.push_back(trimmed);
            }
        }

        return sentences;
    }

    std::string RuleExtractor::clean_text(const std::string& text,
        int max_length) const
    {
        std::string result;
        result.reserve(text.size());

        bool skip_next = false;

        for (size_t i = 0; i < text.size(); ++i) {
            char c = text[i];

            if (skip_next) { skip_next = false; continue; }

            // Skip markdown bold/italic markers
            if (c == '*' || c == '_') {
                if (i + 1 < text.size() && text[i + 1] == c) skip_next = true;
                continue;
            }
            // Skip markdown code markers
            if (c == '`') continue;
            // Skip markdown headers
            if (c == '#') continue;
            // Normalize whitespace
            if (std::isspace(c)) {
                if (!result.empty() && result.back() != ' ') result += ' ';
                continue;
            }

            result += c;
        }

        // Trim
        while (!result.empty() && result.front() == ' ') result.erase(result.begin());
        while (!result.empty() && result.back() == ' ') result.pop_back();

        // Truncate
        if (max_length > 0 && static_cast<int>(result.size()) > max_length) {
            result = result.substr(0, max_length);
            // Try to break at a word boundary
            size_t last_space = result.rfind(' ');
            if (last_space != std::string::npos && last_space > max_length / 2) {
                result = result.substr(0, last_space);
            }
            result += "...";
        }

        return result;
    }

    bool RuleExtractor::contains_any(
        const std::string& text,
        const std::vector<std::string>& keywords) const
    {
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        for (const auto& kw : keywords) {
            if (lower.find(kw) != std::string::npos) return true;
        }
        return false;
    }

    std::string RuleExtractor::find_pattern_sentence(
        const std::vector<std::string>& sentences,
        const std::vector<std::string>& patterns) const
    {
        for (const auto& sentence : sentences) {
            if (contains_any(sentence, patterns)) return sentence;
        }
        return "";
    }

} // namespace cardinal
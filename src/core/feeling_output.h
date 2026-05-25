#pragma once
// =============================================================================
// Cardinal - Feeling Output Pipeline Context
// File: src/core/feeling_output.h
// Owns the state that flows through the two-pass inference pipeline.
// Pass 1 produces a FeelingOutput - this context carries it into Pass 2
// and manages grammar loading, retry state, and synthetic turn injection.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/json_parser.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <optional>
#include <chrono>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // PassState - tracks which inference pass we are in
    // -----------------------------------------------------------------------------
    enum class PassState {
        IDLE,           // No inference in progress
        PASS1_FEELING,  // Running constrained decoding for feeling output
        PASS2_RESPONSE, // Running free decoding for final response
        COMPLETE,       // Both passes done
        FAILED          // Unrecoverable failure
    };

    std::string pass_state_to_string(PassState state);

    // -----------------------------------------------------------------------------
    // GrammarBuffer
    // Holds the loaded GBNF grammar for constrained decoding in Pass 1.
    // Loaded once at startup from config.feeling_schema.grammar_path.
    // -----------------------------------------------------------------------------
    struct GrammarBuffer {
        std::string  path;      // Path it was loaded from
        std::string  content;   // Raw GBNF grammar text
        bool         loaded = false;

        // Load grammar from file - throws if file not found
        void load(const std::string& grammar_path);

        // Check if loaded
        bool is_ready() const { return loaded && !content.empty(); }
    };

    // -----------------------------------------------------------------------------
    // SyntheticTurn
    // The injected assistant turn between Pass 1 and Pass 2.
    // Contains the serialized FeelingOutput JSON - the model treats this
    // as its own prior thought when generating the final response.
    // -----------------------------------------------------------------------------
    struct SyntheticTurn {
        std::string role = "assistant";  // Always assistant
        std::string content;                // Serialized FeelingOutput JSON
        bool        injected = false;       // Whether it's been added to context

        // Build synthetic turn content from a FeelingOutput
        void build(const FeelingOutput& feeling);

        // Format for system prompt injection
        std::string format() const;
    };

    // -----------------------------------------------------------------------------
    // InferenceMetrics
    // Timing and quality metrics for a single inference cycle.
    // Accumulated across passes for benchmarking.
    // -----------------------------------------------------------------------------
    struct InferenceMetrics {
        // Timing
        std::chrono::milliseconds pass1_duration{ 0 };
        std::chrono::milliseconds pass2_duration{ 0 };
        std::chrono::milliseconds total_duration{ 0 };

        // Token counts
        int pass1_tokens_generated = 0;
        int pass2_tokens_generated = 0;
        int prompt_tokens = 0;

        // Quality
        int   retry_count = 0;
        bool  feeling_valid = false;
        bool  grammar_constrained = true;

        // Tokens per second
        float pass1_tps() const;
        float pass2_tps() const;

        // Summary string for logging
        std::string to_string() const;
    };

    // -----------------------------------------------------------------------------
    // FeelingContext
    // The central pipeline state object for one complete inference cycle.
    // Created at the start of each inference call, destroyed when complete.
    //
    // Lifecycle:
    //   1. reset() - clear state from previous cycle
    //   2. Pass 1 runs - feeling_output populated, synthetic_turn built
    //   3. inject() - synthetic turn added to context
    //   4. Pass 2 runs - final_response populated
    //   5. metrics recorded
    // -----------------------------------------------------------------------------
    class FeelingContext {
    public:
        explicit FeelingContext(const CardinalConfig& config);

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------

        // Reset for a new inference cycle - clears output, keeps grammar loaded
        void reset();

        // -------------------------------------------------------------------------
        // Pass 1 - feeling output
        // -------------------------------------------------------------------------

        // Store the raw JSON string from constrained decoding
        void set_raw_feeling(const std::string& raw_json);

        // Parse and validate the raw feeling JSON
        // Returns true on success, false on parse failure (triggers retry logic)
        bool parse_feeling();

        // Whether Pass 1 produced a valid feeling output
        bool has_valid_feeling() const { return feeling_output_.has_value(); }

        // Get the parsed feeling output - must check has_valid_feeling() first
        const FeelingOutput& feeling() const;

        // -------------------------------------------------------------------------
        // Synthetic turn injection
        // -------------------------------------------------------------------------

        // Build and mark synthetic turn ready for injection
        void prepare_synthetic_turn();

        // Get the synthetic turn content for context injection
        const SyntheticTurn& synthetic_turn() const { return synthetic_turn_; }

        // -------------------------------------------------------------------------
        // Pass 2 - final response
        // -------------------------------------------------------------------------

        // Store the final response from Pass 2
        void set_final_response(const std::string& response);

        // Get the final response
        const std::string& final_response() const { return final_response_; }

        // Whether Pass 2 produced a response
        bool has_response() const { return !final_response_.empty(); }

        // -------------------------------------------------------------------------
        // State management
        // -------------------------------------------------------------------------

        PassState   state()         const { return state_; }
        void        set_state(PassState s) { state_ = s; }

        int         retry_count()   const { return retry_count_; }
        bool        should_retry()  const;
        void        increment_retry();

        // -------------------------------------------------------------------------
        // Metrics
        // -------------------------------------------------------------------------

        InferenceMetrics& metrics() { return metrics_; }
        const InferenceMetrics& metrics() const { return metrics_; }

        void start_pass1_timer();
        void stop_pass1_timer();
        void start_pass2_timer();
        void stop_pass2_timer();

        // -------------------------------------------------------------------------
        // Grammar
        // -------------------------------------------------------------------------

        // Loaded once - shared across all inference cycles
        const GrammarBuffer& grammar() const { return grammar_; }
        bool grammar_ready() const { return grammar_.is_ready(); }

        // -------------------------------------------------------------------------
        // Debug
        // -------------------------------------------------------------------------
        std::string to_string() const;

    private:
        const CardinalConfig& config_;
        PassState                        state_ = PassState::IDLE;
        int                              retry_count_ = 0;

        // Pass 1
        std::string                      raw_feeling_;
        std::optional<FeelingOutput>     feeling_output_;

        // Synthetic turn
        SyntheticTurn                    synthetic_turn_;

        // Pass 2
        std::string                      final_response_;

        // Grammar - loaded at construction
        GrammarBuffer                    grammar_;

        // Metrics
        InferenceMetrics                 metrics_;
        std::chrono::steady_clock::time_point pass1_start_;
        std::chrono::steady_clock::time_point pass2_start_;
    };

} // namespace cardinal
#pragma once
// =============================================================================
// Cardinal - Explainability Exporter
// File: src/explainability/explainability_exporter.h
//
// Serializes ReasoningTrace to JSON.
// Used by:
//   - AuditLog::append() for canonical serialization before hashing
//   - HTTP API /trace endpoint for on-demand export
//   - CardinalAPI::export_trace() for file export
// =============================================================================

#include "explainability/reasoning_trace.h"
#include "utils/config_loader.h"

#include <string>

namespace cardinal {

    class ExplainabilityExporter {
    public:
        explicit ExplainabilityExporter(const CardinalConfig& config);

        // ------------------------------------------------------------------
        // Export a trace to JSON string (pretty-printed)
        // ------------------------------------------------------------------
        std::string export_json(const ReasoningTrace& trace) const;

        // ------------------------------------------------------------------
        // Export a trace to a JSON file
        // Returns the output file path on success.
        // ------------------------------------------------------------------
        std::string export_to_file(const ReasoningTrace& trace,
                                   const std::string& filename = "") const;

        // ------------------------------------------------------------------
        // Static: canonical JSON for hashing (compact, deterministic)
        // Used by AuditLog — no pretty printing, keys sorted.
        // ------------------------------------------------------------------
        static std::string trace_to_json(const ReasoningTrace& trace);

    private:
        const CardinalConfig& config_;
    };

} // namespace cardinal

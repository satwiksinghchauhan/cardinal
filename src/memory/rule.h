// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Rule
// File: src/memory/rule.h
// The Rule struct -- a single entry in the persistent rule base.
// Moved here from json_parser.h (Phase 6 refactor) to give Rule its own
// translation unit and allow provenance fields to be added cleanly.
//
// Provenance fields (added Phase 6):
//   episode_id      -- which inference episode created this rule
//   reasoning_type  -- carried from FeelingOutput at extraction time
// =============================================================================

#pragma once

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include <string>

namespace cardinal {

    // -------------------------------------------------------------------------
    // Rule
    // A single entry in the rule store.
    // Created by the verifier when rule_candidate_signal is true in Pass 1.
    //
    // Lifecycle:
    //   1. RuleExtractor creates a Rule from response text + FeelingOutput
    //   2. RuleStore deduplicates via Jaccard similarity before inserting
    //   3. ConsistencyChecker verifies against Prolog KB
    //   4. Rule persists to rules.json with full provenance
    //   5. Confidence decays over time; rule is pruned if below threshold
    // -------------------------------------------------------------------------
    struct Rule {
        // -- Identity --
        std::string id;               // Unique rule ID (timestamp + counter)

        // -- Content --
        std::string domain;           // Reasoning domain: factual | ethical |
        //   spatial | temporal | social | mathematical
        std::string condition;        // "if" part -- natural language or structured
        std::string consequence;      // "then" part

        // -- Confidence --
        float       confidence;       // How confident we are in this rule (0.0 - 1.0)
        int         trigger_count;    // How many times this rule has been triggered

        // -- Timestamps --
        std::string created_at;       // ISO 8601 timestamp of creation
        std::string updated_at;       // ISO 8601 timestamp of last update

        // -- Provenance (Phase 6) --
        std::string episode_id;       // ID of the episode that generated this rule
        //   Empty string if rule predates Phase 6
        //   or was loaded from legacy rules.json
        std::string reasoning_type;   // Reasoning type at extraction time:
        //   analogical | causal | deductive |
        //   inductive | abductive | associative
        //   Carried from FeelingOutput.reasoning_type

// -- Convenience --
        bool is_active(float min_confidence = 0.1f) const {
            return confidence >= min_confidence;
        }

        bool has_provenance() const {
            return !episode_id.empty();
        }
    };

} // namespace cardinal
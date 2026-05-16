// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Self Model (Layer 1)
// File: src/self_model/self_model.h
//
// Builds and maintains Cardinal's symbolic self-knowledge from episodic data.
// Updated after every inference. Read at inference time for prompt injection.
// Persisted to SQLite.
// =============================================================================

#include "self_model/self_model_types.h"
#include "utils/config_loader.h"

#include <string>
#include <mutex>

struct sqlite3;

namespace cardinal {

    class SelfModel {
    public:
        explicit SelfModel(const CardinalConfig& config);
        ~SelfModel();

        void open();
        void close();
        bool is_open() const { return db_ != nullptr; }

        // Called after every inference to update stats
        void record_inference(const std::string& domain,
                              const std::string& reasoning_type,
                              float              confidence,
                              bool               contradiction,
                              bool               uncertainty,
                              bool               rule_committed);

        // Get current snapshot for prompt injection
        SelfModelSnapshot get_snapshot() const;

        // Get weakest domains for curriculum building (Layer 3)
        std::vector<DomainStats> get_weakest_domains(int n = 3) const;

        // Get all domain stats
        std::vector<DomainStats> get_all_domain_stats() const;

        // Formatted string for system prompt injection
        std::string format_for_prompt() const;

        // Stats
        int total_records() const;

    private:
        void create_schema();
        void update_domain_stats(const std::string& domain,
                                 float confidence, bool contradiction,
                                 bool uncertainty, bool rule_committed);
        void update_reasoning_stats(const std::string& reasoning_type,
                                    float confidence, bool contradiction);

        const CardinalConfig& config_;
        sqlite3*              db_    = nullptr;
        mutable std::mutex    mutex_;
    };

} // namespace cardinal

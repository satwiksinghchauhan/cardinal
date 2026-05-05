// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Audit Log
// File: src/explainability/audit_log.h
//
// Append-only SQLite audit log for all inference traces.
// Every entry is:
//   - SHA256 hashed (always)
//   - Ed25519 signed (when signing_enabled = true in config)
//
// The audit log is tamper-evident: any modification to a stored trace
// invalidates its hash and signature.
//
// Key management:
//   - auto_generate_keys=true: generates Ed25519 PEM key pair on first run
//   - Keys stored at config.explainability.private_key_path / public_key_path
//   - Public key can be distributed to verifiers
//
// Thread-safe — append() is serialized via mutex.
// =============================================================================

#include "explainability/reasoning_trace.h"
#include "utils/config_loader.h"

#include <string>
#include <memory>
#include <mutex>

// Forward declare sqlite3 to avoid pulling in the full header everywhere
struct sqlite3;

namespace cardinal {

    class AuditLog {
    public:
        explicit AuditLog(const CardinalConfig& config);
        ~AuditLog();

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------

        // Open/create the audit database. Generates keys if needed.
        void open();

        // Close gracefully
        void close();

        bool is_open() const { return db_ != nullptr; }

        // ------------------------------------------------------------------
        // Write
        // Append a finalized trace to the audit log.
        // Computes SHA256 hash, signs with Ed25519 if enabled,
        // stores in traces table. Returns the inference_id on success.
        // ------------------------------------------------------------------
        std::string append(ReasoningTrace& trace);

        // ------------------------------------------------------------------
        // Verification
        // Verify the signature of a stored trace by inference_id.
        // Returns true if signature is valid (or signing was disabled).
        // ------------------------------------------------------------------
        bool verify(const std::string& inference_id) const;

        // ------------------------------------------------------------------
        // Query
        // ------------------------------------------------------------------

        // Get a trace by inference_id. Returns empty trace if not found.
        ReasoningTrace get(const std::string& inference_id) const;

        // Count total entries
        int count() const;

        // ------------------------------------------------------------------
        // Key management
        // ------------------------------------------------------------------

        // Returns PEM-encoded public key (for distribution to verifiers)
        std::string public_key_pem() const;

        // Returns fingerprint (SHA256 of DER public key, hex)
        std::string public_key_fingerprint() const;

    private:
        // ------------------------------------------------------------------
        // Internal
        // ------------------------------------------------------------------

        // Create tables if they don't exist
        void create_schema();

        // Generate Ed25519 key pair, save to config paths
        void generate_keys();

        // Load existing keys from config paths
        void load_keys();

        // Compute SHA256 of a string, return hex
        std::string sha256_hex(const std::string& data) const;

        // Ed25519 sign, return base64
        std::string ed25519_sign(const std::string& data) const;

        // Ed25519 verify
        bool ed25519_verify(const std::string& data,
                            const std::string& signature_b64) const;

        // Serialize a ReasoningTrace to canonical JSON for hashing
        std::string serialize_for_hash(const ReasoningTrace& trace) const;

        // ------------------------------------------------------------------
        // Members
        // ------------------------------------------------------------------
        const CardinalConfig& config_;

        sqlite3*              db_          = nullptr;
        mutable std::mutex    mutex_;

        // Key material (loaded from PEM files)
        std::string           private_key_pem_;
        std::string           public_key_pem_;
        std::string           key_fingerprint_;
        bool                  keys_loaded_ = false;
    };

} // namespace cardinal

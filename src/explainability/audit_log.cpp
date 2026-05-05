// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Audit Log Implementation
// File: src/explainability/audit_log.cpp
// =============================================================================

#include "explainability/audit_log.h"
#include "explainability/explainability_exporter.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <nlohmann/json.hpp>
#include <sqlite3.h>

// OpenSSL for Ed25519 + SHA256
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/err.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iomanip>

using json = nlohmann::json;

namespace cardinal {

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    AuditLog::AuditLog(const CardinalConfig& config)
        : config_(config)
    {}

    AuditLog::~AuditLog() {
        close();
    }

    // =========================================================================
    // open
    // =========================================================================

    void AuditLog::open() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!config_.explainability.enabled) {
            LOG_INFO("AuditLog: explainability disabled in config");
            return;
        }

        // Create parent directory
        auto db_path = config_.explainability.audit_log_path;
        auto parent  = std::filesystem::path(db_path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        // Open SQLite database
        int rc = sqlite3_open(db_path.c_str(), &db_);
        if (rc != SQLITE_OK) {
            throw std::runtime_error(
                "AuditLog: failed to open database: " + db_path +
                " — " + sqlite3_errmsg(db_));
        }

        // WAL mode for better concurrent access
        sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);

        create_schema();

        // Key management
        if (config_.explainability.signing_enabled) {
            bool priv_exists = std::filesystem::exists(
                config_.explainability.private_key_path);
            bool pub_exists  = std::filesystem::exists(
                config_.explainability.public_key_path);

            if (!priv_exists || !pub_exists) {
                if (config_.explainability.auto_generate_keys) {
                    generate_keys();
                } else {
                    LOG_WARN("AuditLog: signing enabled but key files not found "
                             "and auto_generate_keys=false. Signing disabled.");
                }
            } else {
                load_keys();
            }
        }

        LOG_INFO("AuditLog: opened — " + db_path +
                 (keys_loaded_ ? " [Ed25519 signing enabled]" : " [unsigned]"));
    }

    void AuditLog::close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    // =========================================================================
    // create_schema
    // =========================================================================

    void AuditLog::create_schema() {
        const char* sql = R"(
            CREATE TABLE IF NOT EXISTS traces (
                inference_id    TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                backend_type    TEXT,
                model_name      TEXT,
                agent_mode      INTEGER DEFAULT 0,
                query_preview   TEXT,
                total_tokens    INTEGER DEFAULT 0,
                total_ms        INTEGER DEFAULT 0,
                tool_calls      INTEGER DEFAULT 0,
                rule_committed  INTEGER DEFAULT 0,
                sha256_hash     TEXT NOT NULL,
                signature       TEXT,
                public_key_id   TEXT,
                trace_json      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_traces_session
                ON traces(session_id);
            CREATE INDEX IF NOT EXISTS idx_traces_timestamp
                ON traces(timestamp);
        )";

        char* err = nullptr;
        int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err);
        if (rc != SQLITE_OK) {
            std::string msg = err ? err : "unknown";
            sqlite3_free(err);
            throw std::runtime_error("AuditLog: schema creation failed: " + msg);
        }
    }

    // =========================================================================
    // append
    // =========================================================================

    std::string AuditLog::append(ReasoningTrace& trace) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!db_) return trace.inference_id;

        // Serialize to canonical JSON
        std::string trace_json = ExplainabilityExporter::trace_to_json(trace);

        // Compute SHA256
        std::string hash = sha256_hex(trace_json);
        trace.integrity.sha256_hash = hash;

        // Sign if enabled
        if (config_.explainability.signing_enabled && keys_loaded_) {
            trace.integrity.signature    = ed25519_sign(hash);
            trace.integrity.public_key_id = key_fingerprint_;
            trace.integrity.signed_      = true;
            trace.integrity.signed_at    = JsonParser::current_timestamp();
        }

        // Re-serialize with integrity fields populated
        trace_json = ExplainabilityExporter::trace_to_json(trace);

        // Insert into SQLite
        const char* sql =
            "INSERT OR REPLACE INTO traces "
            "(inference_id, session_id, timestamp, backend_type, model_name, "
            " agent_mode, query_preview, total_tokens, total_ms, tool_calls, "
            " rule_committed, sha256_hash, signature, public_key_id, trace_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

        std::string query_preview = trace.query.substr(
            0, std::min(trace.query.size(), size_t(200)));

        sqlite3_bind_text(stmt, 1,  trace.inference_id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2,  trace.session_id.c_str(),   -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3,  trace.timestamp.c_str(),    -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4,  trace.backend_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5,  trace.model_name.c_str(),   -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 6,  trace.agent_mode ? 1 : 0);
        sqlite3_bind_text(stmt, 7,  query_preview.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (stmt, 8,  trace.total_tokens);
        sqlite3_bind_int (stmt, 9,  trace.total_duration_ms);
        sqlite3_bind_int (stmt, 10, static_cast<int>(trace.tool_calls.size()));
        sqlite3_bind_int (stmt, 11, trace.rule_committed ? 1 : 0);
        sqlite3_bind_text(stmt, 12, hash.c_str(),                -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 13, trace.integrity.signature.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 14, key_fingerprint_.c_str(),    -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 15, trace_json.c_str(),          -1, SQLITE_TRANSIENT);

        int rc = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if (rc != SQLITE_DONE) {
            LOG_WARN("AuditLog: insert failed: " + std::string(sqlite3_errmsg(db_)));
        } else {
            LOG_DEBUG("AuditLog: appended " + trace.inference_id +
                      (trace.integrity.signed_ ? " [signed]" : " [unsigned]"));
        }

        return trace.inference_id;
    }

    // =========================================================================
    // verify
    // =========================================================================

    // =========================================================================
    // get — retrieve a trace by inference_id
    // =========================================================================

    ReasoningTrace AuditLog::get(const std::string& inference_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        ReasoningTrace trace;
        if (!db_) return trace;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT trace_json FROM traces WHERE inference_id = ?;",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, inference_id.c_str(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string trace_json =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            sqlite3_finalize(stmt);

            // Parse the stored JSON back into a ReasoningTrace
            try {
                auto j = json::parse(trace_json);

                trace.inference_id   = j.value("inference_id", "");
                trace.session_id     = j.value("session_id", "");
                trace.episode_id     = j.value("episode_id", "");
                trace.timestamp      = j.value("timestamp", "");
                trace.backend_type   = j.value("backend_type", "");
                trace.model_name     = j.value("model_name", "");
                trace.agent_mode     = j.value("agent_mode", false);
                trace.query          = j.value("query", "");
                trace.final_response = j.value("final_response", "");
                trace.total_tokens   = j.value("total_tokens", 0);
                trace.total_duration_ms = j.value("total_duration_ms", 0);

                // Integrity
                if (j.contains("integrity")) {
                    const auto& i = j["integrity"];
                    trace.integrity.sha256_hash   = i.value("sha256_hash", "");
                    trace.integrity.signed_        = i.value("signed", false);
                    trace.integrity.signature      = i.value("signature", "");
                    trace.integrity.public_key_id  = i.value("public_key_id", "");
                    trace.integrity.signed_at      = i.value("signed_at", "");
                }

                // Pass 1 feeling
                if (j.contains("pass1")) {
                    const auto& p1 = j["pass1"];
                    trace.feeling_valid     = p1.value("valid", false);
                    trace.pass1_retries     = p1.value("retries", 0);
                    trace.pass1_tokens      = p1.value("tokens", 0);
                    trace.pass1_duration_ms = p1.value("duration_ms", 0);
                    if (p1.contains("feeling")) {
                        const auto& f = p1["feeling"];
                        trace.feeling.confidence            = f.value("confidence", 0.0f);
                        trace.feeling.reasoning_type        = f.value("reasoning_type", "");
                        trace.feeling.reasoning_domain      = f.value("reasoning_domain", "");
                        trace.feeling.uncertainty_flag      = f.value("uncertainty_flag", false);
                        trace.feeling.contradiction_flag    = f.value("contradiction_flag", false);
                        trace.feeling.rule_candidate_signal = f.value("rule_candidate_signal", false);
                    }
                }

                // Pass 2
                if (j.contains("pass2")) {
                    const auto& p2 = j["pass2"];
                    trace.pass2_tokens      = p2.value("tokens", 0);
                    trace.pass2_duration_ms = p2.value("duration_ms", 0);
                }

                // Rule extraction
                if (j.contains("rule_extraction")) {
                    const auto& re = j["rule_extraction"];
                    trace.rule_committed    = re.value("committed", false);
                    trace.committed_rule_id = re.value("rule_id", "");
                }

            } catch (const json::exception& e) {
                LOG_WARN("AuditLog::get: JSON parse error for " +
                         inference_id + ": " + e.what());
            }
        } else {
            sqlite3_finalize(stmt);
        }

        return trace;
    }

    // =========================================================================
    // verify
    // =========================================================================

    bool AuditLog::verify(const std::string& inference_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_ || !keys_loaded_) return true; // can't verify, assume ok

        const char* sql =
            "SELECT sha256_hash, signature, trace_json FROM traces "
            "WHERE inference_id = ?;";

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, inference_id.c_str(), -1, SQLITE_TRANSIENT);

        bool valid = false;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            std::string stored_hash =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            std::string stored_sig  =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            std::string trace_json  =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

            // Recompute hash and compare
            // Note: hash was computed on trace_json *before* integrity fields
            // were added. We store separately for this reason.
            valid = ed25519_verify(stored_hash, stored_sig);
        }

        sqlite3_finalize(stmt);
        return valid;
    }

    // =========================================================================
    // count
    // =========================================================================

    int AuditLog::count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!db_) return 0;

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM traces;", -1, &stmt, nullptr);

        int count = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW)
            count = sqlite3_column_int(stmt, 0);

        sqlite3_finalize(stmt);
        return count;
    }

    // =========================================================================
    // Key management
    // =========================================================================

    void AuditLog::generate_keys() {
        LOG_INFO("AuditLog: generating Ed25519 key pair...");

        // Create key pair
        EVP_PKEY* pkey = nullptr;
        EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
        if (!ctx) throw std::runtime_error("AuditLog: EVP_PKEY_CTX_new_id failed");

        EVP_PKEY_keygen_init(ctx);
        EVP_PKEY_keygen(ctx, &pkey);
        EVP_PKEY_CTX_free(ctx);

        if (!pkey) throw std::runtime_error("AuditLog: key generation failed");

        // Save private key
        {
            auto priv_path = config_.explainability.private_key_path;
            auto parent    = std::filesystem::path(priv_path).parent_path();
            if (!parent.empty())
                std::filesystem::create_directories(parent);

            FILE* f = fopen(priv_path.c_str(), "wb");
            if (!f) {
                EVP_PKEY_free(pkey);
                throw std::runtime_error(
                    "AuditLog: cannot write private key: " + priv_path);
            }
            PEM_write_PrivateKey(f, pkey, nullptr, nullptr, 0, nullptr, nullptr);
            fclose(f);
            // Restrict permissions on private key
            std::filesystem::permissions(priv_path,
                std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write);
        }

        // Save public key
        {
            auto pub_path = config_.explainability.public_key_path;
            FILE* f = fopen(pub_path.c_str(), "wb");
            if (!f) {
                EVP_PKEY_free(pkey);
                throw std::runtime_error(
                    "AuditLog: cannot write public key: " + pub_path);
            }
            PEM_write_PUBKEY(f, pkey);
            fclose(f);
        }

        EVP_PKEY_free(pkey);

        LOG_INFO("AuditLog: key pair generated");
        load_keys();
    }

    void AuditLog::load_keys() {
        const auto& priv_path = config_.explainability.private_key_path;
        const auto& pub_path  = config_.explainability.public_key_path;

        // Read PEM files
        {
            std::ifstream f(priv_path);
            if (!f.is_open()) {
                LOG_WARN("AuditLog: cannot read private key: " + priv_path);
                return;
            }
            private_key_pem_ = std::string(
                (std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());
        }
        {
            std::ifstream f(pub_path);
            if (!f.is_open()) {
                LOG_WARN("AuditLog: cannot read public key: " + pub_path);
                return;
            }
            public_key_pem_ = std::string(
                (std::istreambuf_iterator<char>(f)),
                 std::istreambuf_iterator<char>());
        }

        // Compute fingerprint (SHA256 of DER-encoded public key)
        BIO* bio = BIO_new_mem_buf(public_key_pem_.data(),
                                    static_cast<int>(public_key_pem_.size()));
        EVP_PKEY* pkey = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);

        if (pkey) {
            // Get DER encoding
            unsigned char* der = nullptr;
            int der_len = i2d_PublicKey(pkey, &der);
            if (der_len > 0) {
                unsigned char hash[SHA256_DIGEST_LENGTH];
                SHA256(der, der_len, hash);
                OPENSSL_free(der);

                std::ostringstream oss;
                for (int i = 0; i < SHA256_DIGEST_LENGTH; i += 2) {
                    if (i > 0) oss << ":";
                    oss << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(hash[i]);
                }
                key_fingerprint_ = oss.str();
            }
            EVP_PKEY_free(pkey);
        }

        keys_loaded_ = true;
        LOG_INFO("AuditLog: keys loaded, fingerprint: " + key_fingerprint_);
    }

    // =========================================================================
    // SHA256
    // =========================================================================

    std::string AuditLog::sha256_hex(const std::string& data) const {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(data.data()),
               data.size(), hash);

        std::ostringstream oss;
        for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
            oss << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(hash[i]);
        return oss.str();
    }

    // =========================================================================
    // Ed25519 sign
    // =========================================================================

    std::string AuditLog::ed25519_sign(const std::string& data) const {
        if (!keys_loaded_ || private_key_pem_.empty()) return "";

        BIO* bio = BIO_new_mem_buf(private_key_pem_.data(),
                                    static_cast<int>(private_key_pem_.size()));
        EVP_PKEY* pkey = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);

        if (!pkey) return "";

        EVP_MD_CTX* md_ctx = EVP_MD_CTX_new();
        EVP_DigestSignInit(md_ctx, nullptr, nullptr, nullptr, pkey);

        size_t sig_len = 0;
        EVP_DigestSign(md_ctx,
            nullptr, &sig_len,
            reinterpret_cast<const unsigned char*>(data.data()), data.size());

        std::vector<unsigned char> sig(sig_len);
        EVP_DigestSign(md_ctx,
            sig.data(), &sig_len,
            reinterpret_cast<const unsigned char*>(data.data()), data.size());

        EVP_MD_CTX_free(md_ctx);
        EVP_PKEY_free(pkey);

        // Base64 encode
        BIO* b64  = BIO_new(BIO_f_base64());
        BIO* bmem = BIO_new(BIO_s_mem());
        b64 = BIO_push(b64, bmem);
        BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
        BIO_write(b64, sig.data(), static_cast<int>(sig_len));
        BIO_flush(b64);

        BUF_MEM* bptr = nullptr;
        BIO_get_mem_ptr(b64, &bptr);
        std::string result(bptr->data, bptr->length);
        BIO_free_all(b64);

        return result;
    }

    // =========================================================================
    // Ed25519 verify
    // =========================================================================

    bool AuditLog::ed25519_verify(const std::string& data,
                                   const std::string& signature_b64) const
    {
        if (!keys_loaded_ || public_key_pem_.empty() || signature_b64.empty())
            return false;

        // Base64 decode signature
        BIO* b64  = BIO_new(BIO_f_base64());
        BIO* bmem = BIO_new_mem_buf(signature_b64.data(),
                                     static_cast<int>(signature_b64.size()));
        b64 = BIO_push(b64, bmem);
        BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

        std::vector<unsigned char> sig(signature_b64.size());
        int sig_len = BIO_read(b64, sig.data(), static_cast<int>(sig.size()));
        BIO_free_all(b64);

        if (sig_len <= 0) return false;

        // Load public key
        BIO* bio = BIO_new_mem_buf(public_key_pem_.data(),
                                    static_cast<int>(public_key_pem_.size()));
        EVP_PKEY* pkey = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);

        if (!pkey) return false;

        EVP_MD_CTX* md_ctx = EVP_MD_CTX_new();
        EVP_DigestVerifyInit(md_ctx, nullptr, nullptr, nullptr, pkey);

        int rc = EVP_DigestVerify(
            md_ctx,
            sig.data(), static_cast<size_t>(sig_len),
            reinterpret_cast<const unsigned char*>(data.data()), data.size());

        EVP_MD_CTX_free(md_ctx);
        EVP_PKEY_free(pkey);

        return rc == 1;
    }

    // =========================================================================
    // Public key accessors
    // =========================================================================

    std::string AuditLog::public_key_pem() const {
        return public_key_pem_;
    }

    std::string AuditLog::public_key_fingerprint() const {
        return key_fingerprint_;
    }

} // namespace cardinal

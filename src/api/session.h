// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Conversation Session
// File: src/api/session.h
//
// Manages a single conversation session's state.
// CardinalAPI owns a map of these -- one per active session.
//
// Responsibilities:
//   - Conversation history (accumulated across turns)
//   - Turn counter
//   - Session ID (generated at creation)
//   - Timestamps (created_at, last_active_at)
//   - Session-scoped context (future: per-session system prompt override)
//
// Thread safety:
//   Individual sessions are NOT thread-safe internally.
//   Thread safety is provided by CardinalAPI's session map mutex.
//   One session is never accessed concurrently -- API serializes per session.
//
// Lifecycle:
//   Created by CardinalAPI::create_session() or on first chat() call
//   Destroyed by CardinalAPI::destroy_session() or on shutdown
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "api/cardinal_types.h"
#include "utils/json_parser.h"

#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <mutex>

namespace cardinal {

    // =========================================================================
    // ConversationSession
    // =========================================================================
    class ConversationSession {
    public:
        // Create a new session with a generated ID
        ConversationSession();

        // Create a session with a specific ID (for restore/testing)
        explicit ConversationSession(const std::string& session_id);

        // -- Identity --
        const std::string& id()         const { return session_id_; }
        int                turn_count() const { return turn_count_; }

        // -- Timestamps --
        const std::string& created_at()     const { return created_at_; }
        const std::string& last_active_at() const { return last_active_at_; }

        // -- History management --

        // Append a user turn to history
        void add_user_turn(const std::string& content);

        // Append an assistant turn to history
        void add_assistant_turn(const std::string& content);

        // Get the full history as ChatMessage vector for InferenceRequest
        // Returns the raw core type -- used internally by CardinalAPI
        std::vector<ChatMessage> get_history() const;

        // Get the full history as API-boundary ChatTurn vector
        // Used for SessionInfo and HTTP responses
        std::vector<ChatTurn> get_chat_turns() const;

        // Clear all history and reset turn counter
        void reset();

        // -- Context window management --

        // How many total tokens the history approximately occupies
        // Rough estimate: 1 token per 4 chars
        int estimated_token_count() const;

        // Trim history to keep it within a token budget.
        // Removes oldest turns first, always keeps the most recent N turns.
        // Returns number of turns removed.
        int trim_to_token_budget(int max_tokens, int min_turns_to_keep = 2);

        // -- Snapshot --

        // Build a SessionInfo for API responses
        SessionInfo to_session_info() const;

    private:
        std::string              session_id_;
        int                      turn_count_ = 0;
        std::string              created_at_;
        std::string              last_active_at_;

        // Internal history -- core type for direct pipeline use
        std::vector<ChatMessage> history_;

        // Parallel timestamp list -- one entry per history_ entry
        // Used to populate ChatTurn.timestamp in get_chat_turns()
        std::vector<std::string> timestamps_;

        // Update last_active_at to now
        void touch();
    };

    // =========================================================================
    // SessionManager
    // Owns all active sessions. Used by CardinalAPI internally.
    // Thread-safe -- all methods protected by mutex.
    // =========================================================================
    class SessionManager {
    public:
        SessionManager() = default;

        // Not copyable -- owns session state
        SessionManager(const SessionManager&) = delete;
        SessionManager& operator=(const SessionManager&) = delete;

        // -- Session lifecycle --

        // Create a new session. Returns the new session ID.
        std::string create();

        // Check if a session exists
        bool exists(const std::string& session_id) const;

        // Destroy a session. Returns false if not found.
        bool destroy(const std::string& session_id);

        // Destroy all sessions
        void destroy_all();

        // -- Session access --
        // These return raw pointers into the map -- caller must hold
        // the session lock for the duration of use.
        // CardinalAPI uses these with its own session_mutex_.

        // Get session pointer -- nullptr if not found
        ConversationSession* get(const std::string& session_id);
        const ConversationSession* get(const std::string& session_id) const;

        // -- Stats --
        int  count() const;
        bool empty() const;

        // Get all session IDs
        std::vector<std::string> all_ids() const;

    private:
        std::unordered_map<std::string, ConversationSession> sessions_;
        mutable std::mutex mutex_;
    };

} // namespace cardinal
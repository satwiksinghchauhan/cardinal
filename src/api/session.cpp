// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Conversation Session Implementation
// File: src/api/session.cpp
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "api/session.h"
#include "utils/logger.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <mutex>

namespace cardinal {

    // =========================================================================
    // ConversationSession
    // =========================================================================

    ConversationSession::ConversationSession()
        : session_id_(JsonParser::generate_id())
        , created_at_(JsonParser::current_timestamp())
        , last_active_at_(created_at_)
    {
    }

    ConversationSession::ConversationSession(const std::string& session_id)
        : session_id_(session_id)
        , created_at_(JsonParser::current_timestamp())
        , last_active_at_(created_at_)
    {
    }

    // =========================================================================
    // History management
    // =========================================================================

    void ConversationSession::add_user_turn(const std::string& content) {
        history_.push_back({ "user", content });
        timestamps_.push_back(JsonParser::current_timestamp());
        ++turn_count_;
        touch();
    }

    void ConversationSession::add_assistant_turn(const std::string& content) {
        history_.push_back({ "assistant", content });
        timestamps_.push_back(JsonParser::current_timestamp());
        // Don't increment turn_count_ -- one turn = one user+assistant pair
        touch();
    }

    std::vector<ChatMessage> ConversationSession::get_history() const {
        return history_;
    }

    std::vector<ChatTurn> ConversationSession::get_chat_turns() const {
        std::vector<ChatTurn> turns;
        turns.reserve(history_.size());

        for (size_t i = 0; i < history_.size(); ++i) {
            ChatTurn t;
            t.role = history_[i].role;
            t.content = history_[i].content;
            t.timestamp = (i < timestamps_.size()) ? timestamps_[i] : "";
            turns.push_back(std::move(t));
        }

        return turns;
    }

    void ConversationSession::reset() {
        history_.clear();
        timestamps_.clear();
        turn_count_ = 0;
        touch();
        LOG_DEBUG("Session " + session_id_ + " reset");
    }

    // =========================================================================
    // Context window management
    // =========================================================================

    int ConversationSession::estimated_token_count() const {
        int total_chars = 0;
        for (const auto& msg : history_) {
            total_chars += static_cast<int>(msg.content.size());
        }
        // Rough estimate: 1 token per 4 characters
        return total_chars / 4;
    }

    int ConversationSession::trim_to_token_budget(int max_tokens,
        int min_turns_to_keep)
    {
        if (history_.empty()) return 0;

        int removed = 0;

        // Each logical turn is a user+assistant pair = 2 history entries
        // We remove from the front, keeping at least min_turns_to_keep pairs
        int min_entries_to_keep = min_turns_to_keep * 2;

        while (estimated_token_count() > max_tokens &&
            static_cast<int>(history_.size()) > min_entries_to_keep) {
            // Remove oldest entry (front)
            history_.erase(history_.begin());
            if (!timestamps_.empty()) {
                timestamps_.erase(timestamps_.begin());
            }
            ++removed;
        }

        if (removed > 0) {
            LOG_DEBUG("Session " + session_id_ + ": trimmed " +
                std::to_string(removed) + " history entries");
        }

        return removed;
    }

    // =========================================================================
    // Snapshot
    // =========================================================================

    SessionInfo ConversationSession::to_session_info() const {
        SessionInfo info;
        info.session_id = session_id_;
        info.turn_count = turn_count_;
        info.history = get_chat_turns();
        info.created_at = created_at_;
        info.last_active_at = last_active_at_;
        return info;
    }

    // =========================================================================
    // Private
    // =========================================================================

    void ConversationSession::touch() {
        last_active_at_ = JsonParser::current_timestamp();
    }

    // =========================================================================
    // SessionManager
    // =========================================================================

    std::string SessionManager::create() {
        std::lock_guard<std::mutex> lock(mutex_);

        ConversationSession session;
        std::string id = session.id();
        sessions_.emplace(id, std::move(session));

        LOG_DEBUG("SessionManager: created session " + id);
        return id;
    }

    bool SessionManager::exists(const std::string& session_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sessions_.count(session_id) > 0;
    }

    bool SessionManager::destroy(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return false;
        sessions_.erase(it);
        LOG_DEBUG("SessionManager: destroyed session " + session_id);
        return true;
    }

    void SessionManager::destroy_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        int n = static_cast<int>(sessions_.size());
        sessions_.clear();
        LOG_DEBUG("SessionManager: destroyed " + std::to_string(n) + " sessions");
    }

    ConversationSession*
        SessionManager::get(const std::string& session_id) {
        // No lock here -- caller (CardinalAPI) holds its own session mutex
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return nullptr;
        return &it->second;
    }

    const ConversationSession*
        SessionManager::get(const std::string& session_id) const {
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return nullptr;
        return &it->second;
    }

    int SessionManager::count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(sessions_.size());
    }

    bool SessionManager::empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sessions_.empty();
    }

    std::vector<std::string> SessionManager::all_ids() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> ids;
        ids.reserve(sessions_.size());
        for (const auto& [id, _] : sessions_) {
            ids.push_back(id);
        }
        return ids;
    }

} // namespace cardinal
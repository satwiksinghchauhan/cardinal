// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Tool Executor
// File: src/tools/tool_executor.h
//
// Execution runtime for all tool calls.
// Responsibilities:
//   - Detect tool calls in model output (model-agnostic JSON parsing)
//   - Validate against ToolRegistry
//   - Route to correct built-in implementation
//   - Handle confirmation pausing
//   - Return ToolResult for context injection
//
// Thread-safe — multiple inference threads can call execute() concurrently.
// =============================================================================

#include "tools/tool_result.h"
#include "tools/tool_registry.h"
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>

namespace cardinal {

    // Forward declarations — full definitions in their own headers
    class KnowledgeGraph;
    class EpisodicRetriever;
    class VisionEncoder;
    class VisionCache;

    // -------------------------------------------------------------------------
    // ConfirmationCallback
    // Called when a tool requires human confirmation.
    // The callback must return true to approve, false to deny.
    // Used by the HTTP API to pause and wait for user input.
    // -------------------------------------------------------------------------
    using ConfirmationCallback =
        std::function<bool(const ConfirmationRequest&)>;

    // -------------------------------------------------------------------------
    // ToolExecutor
    // -------------------------------------------------------------------------
    class ToolExecutor {
    public:
        ToolExecutor(const CardinalConfig&  config,
                     const ToolRegistry&    registry);

        // ------------------------------------------------------------------
        // Dependency injection for tools that need Cardinal subsystems
        // ------------------------------------------------------------------
        void set_knowledge_graph(KnowledgeGraph* kg)      { kg_ = kg; }
        void set_retriever(EpisodicRetriever* retriever)  { retriever_ = retriever; }
        void set_vision_encoder(VisionEncoder* encoder) { vision_encoder_ = encoder; }
        void set_vision_cache(VisionCache* cache)         { vision_cache_   = cache; }

        void set_confirmation_callback(ConfirmationCallback cb) {
            confirmation_cb_ = std::move(cb);
        }

        // ------------------------------------------------------------------
        // Detection
        // Scan model output for tool call blocks.
        // Looks for <tool_call>...</tool_call> pattern (model-agnostic).
        // Also handles raw JSON objects that match tool call schema.
        // ------------------------------------------------------------------
        ToolCallDetectionResult detect_tool_calls(
            const std::string& model_output) const;

        // ------------------------------------------------------------------
        // Execution
        // Execute a single validated tool call.
        // Returns ToolResult — caller injects result.output into context.
        // ------------------------------------------------------------------
        ToolResult execute(const ToolCall& call) const;

        // Execute multiple tool calls (parallel where safe)
        std::vector<ToolResult> execute_all(
            const std::vector<ToolCall>& calls) const;

        // ------------------------------------------------------------------
        // Context injection
        // Format tool results for injection back into the model context.
        // Called by InferencePipeline after tool execution.
        // ------------------------------------------------------------------
        std::string format_results_for_context(
            const std::vector<ToolResult>& results) const;

    private:
        // ------------------------------------------------------------------
        // Built-in dispatch
        // ------------------------------------------------------------------
        ToolResult dispatch(const ToolCall& call) const;

        ToolResult execute_web_search(const ToolCall& call)   const;
        ToolResult execute_web_fetch(const ToolCall& call)    const;
        ToolResult execute_calculator(const ToolCall& call)   const;
        ToolResult execute_run_python(const ToolCall& call)   const;
        ToolResult execute_file_read(const ToolCall& call)    const;
        ToolResult execute_file_write(const ToolCall& call)   const;
        ToolResult execute_kg_query(const ToolCall& call)     const;
        ToolResult execute_episodic_search(const ToolCall& call) const;

        // run_python sandbox modes
        ToolResult run_python_subprocess(const std::string& code,
                                          int timeout_seconds) const;
        ToolResult run_python_docker(const std::string& code,
                                      int timeout_seconds) const;

        // Path safety check for file tools
        bool is_path_allowed(const std::string& path,
                              const std::vector<std::string>& allowed_paths) const;

        // Get argument value with fallback
        std::string get_arg(const ToolCall& call,
                            const std::string& key,
                            const std::string& default_val = "") const;

        // Build a ToolResult for a failed call
        ToolResult make_error(const ToolCall& call,
                              ToolStatus status,
                              const std::string& message) const;

        // Build a ToolResult for a successful call
        ToolResult make_success(const ToolCall& call,
                                const std::string& output,
                                int duration_ms) const;

        // ------------------------------------------------------------------
        // Members
        // ------------------------------------------------------------------
        const CardinalConfig&  config_;
        const ToolRegistry&    registry_;

        // Optional subsystem pointers (set after construction)
        KnowledgeGraph*        kg_        = nullptr;
        EpisodicRetriever*     retriever_ = nullptr;
        VisionEncoder*         vision_encoder_ = nullptr;
        VisionCache*           vision_cache_   = nullptr;
        ConfirmationCallback   confirmation_cb_;

        mutable std::mutex     exec_mutex_;
    };

} // namespace cardinal

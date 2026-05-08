// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - analyze_image Tool Implementation
// File: src/tools/builtin/analyze_image.cpp
// =============================================================================

#include "tools/builtin/analyze_image.h"
#include "utils/logger.h"
#include "utils/json_parser.h"

#include <chrono>

namespace cardinal {

    ToolResult execute_analyze_image(const ToolCall&       call,
                                      const CardinalConfig& config,
                                      VisionEncoder&        encoder,
                                      VisionCache&          cache)
    {
        auto start = std::chrono::steady_clock::now();

        auto elapsed_ms = [&start]() {
            return static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count());
        };

        auto make_err = [&](ToolStatus status, const std::string& msg) {
            ToolResult r;
            r.tool_name     = "analyze_image";
            r.call          = call;
            r.status        = status;
            r.error_message = msg;
            r.duration_ms   = elapsed_ms();
            r.timestamp     = JsonParser::current_timestamp();
            LOG_WARN("analyze_image: " + msg);
            return r;
        };

        // ------------------------------------------------------------------
        // Validate vision encoder
        // ------------------------------------------------------------------
        if (!encoder.is_ready()) {
            return make_err(ToolStatus::FAILURE,
                "Vision encoder not available. Configure vision.model_path "
                "and vision.mmproj_path in config.json and restart Cardinal.");
        }

        // ------------------------------------------------------------------
        // Extract arguments
        // ------------------------------------------------------------------
        auto get_arg = [&](const std::string& key,
                            const std::string& def = "") -> std::string {
            auto it = call.arguments.find(key);
            if (it == call.arguments.end()) return def;
            std::string val = it->second;
            // Strip surrounding quotes from JSON string values
            if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
                val = val.substr(1, val.size() - 2);
            return val;
        };

        std::string image_input = get_arg("image");
        std::string prompt      = get_arg("prompt"); // empty = use default

        if (image_input.empty()) {
            return make_err(ToolStatus::INVALID_ARGS,
                "analyze_image: 'image' argument is required");
        }

        // ------------------------------------------------------------------
        // Resolve image (file or URL → local path)
        // ------------------------------------------------------------------
        ImageMetadata metadata;
        std::string   cache_error;

        std::string local_path = cache.resolve(image_input, metadata, cache_error);

        if (local_path.empty()) {
            return make_err(ToolStatus::FAILURE,
                "analyze_image: could not resolve image: " + cache_error);
        }

        // ------------------------------------------------------------------
        // Encode image
        // ------------------------------------------------------------------
        VisionResult vision = encoder.encode(local_path, prompt, metadata);

        if (!vision.success) {
            return make_err(ToolStatus::FAILURE,
                "analyze_image: encoding failed: " + vision.error_message);
        }

        // ------------------------------------------------------------------
        // Format output (Option C: metadata header + description)
        // ------------------------------------------------------------------
        std::string output = vision.format_for_context();

        ToolResult result;
        result.tool_name   = "analyze_image";
        result.call        = call;
        result.status      = ToolStatus::SUCCESS;
        result.output      = output;
        result.duration_ms = elapsed_ms();
        result.timestamp   = JsonParser::current_timestamp();

        LOG_INFO("analyze_image: success in " +
                 std::to_string(result.duration_ms) + "ms");

        return result;
    }

} // namespace cardinal

#pragma once
// =============================================================================
// Cardinal - analyze_image Tool
// File: src/tools/builtin/analyze_image.h
//
// Tool that takes an image (file path or URL) and returns a text description
// via moondream2. Output format (Option C):
//
//   [Image: 1920x1080 JPEG, 245KB, source: url]
//   Description: <moondream2 output>
//
// Registered in ToolRegistry at startup if vision.model_path is configured.
// =============================================================================

#include "tools/tool_result.h"
#include "vision/vision_encoder.h"
#include "vision/vision_cache.h"
#include "utils/config_loader.h"

namespace cardinal {

    // Execute the analyze_image tool call.
    // Called by ToolExecutor::dispatch() when tool_name == "analyze_image".
    ToolResult execute_analyze_image(const ToolCall&     call,
                                      const CardinalConfig& config,
                                      VisionEncoder&       encoder,
                                      VisionCache&         cache);

} // namespace cardinal

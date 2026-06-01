// =============================================================================
// Cardinal - Tool: screenshot Implementation
// File: src/tools/builtin/computer/tool_screenshot.cpp
// =============================================================================

#include "tools/builtin/computer/tool_screenshot.h"
#include "computer/screen_reader.h"
#include "utils/config_loader.h"
#include "utils/logger.h"

#include <chrono>
#include <sstream>

namespace cardinal {

ToolDefinition make_screenshot_tool_def(const CardinalConfig& config) {
    ToolDefinition def;
    def.name        = "screenshot";
    def.description = "Capture the current screen and return a description of what is visible. "
                      "Optionally capture a specific region. "
                      "Returns the screenshot file path and a vision-based description.";
    def.confirmation_required = false;

    def.parameters.push_back({
        "analyze", ToolParameterType::BOOLEAN,
        "If true, run vision analysis and return a text description of the screen. "
        "Default: true.",
        false, "true"
    });
    def.parameters.push_back({
        "prompt", ToolParameterType::STRING,
        "Optional specific question about the screen content, e.g. 'What text is visible?'",
        false, ""
    });
    def.parameters.push_back({
        "region_x", ToolParameterType::NUMBER,
        "X coordinate of region to capture (leave unset for full screen)",
        false, ""
    });
    def.parameters.push_back({
        "region_y", ToolParameterType::NUMBER,
        "Y coordinate of region to capture",
        false, ""
    });
    def.parameters.push_back({
        "region_w", ToolParameterType::NUMBER,
        "Width of region to capture",
        false, ""
    });
    def.parameters.push_back({
        "region_h", ToolParameterType::NUMBER,
        "Height of region to capture",
        false, ""
    });

    (void)config;
    return def;
}

ToolResult execute_screenshot(const ToolCall& call, ScreenReader& reader) {
    ToolResult result;
    result.tool_name = "screenshot";
    result.call      = call;

    auto t0 = std::chrono::steady_clock::now();

    try {
        auto get = [&](const std::string& k, const std::string& def = "") -> std::string {
            auto it = call.arguments.find(k);
            return it != call.arguments.end() ? it->second : def;
        };

        bool analyze = get("analyze", "true") != "false";
        std::string prompt = get("prompt");

        std::string rx = get("region_x"), ry = get("region_y");
        std::string rw = get("region_w"), rh = get("region_h");

        Screenshot s;
        if (!rx.empty() && !ry.empty() && !rw.empty() && !rh.empty()) {
            ScreenRegion region;
            region.x      = std::stoi(rx);
            region.y      = std::stoi(ry);
            region.width  = std::stoi(rw);
            region.height = std::stoi(rh);
            s = reader.capture_region(region, analyze);
        } else {
            s = reader.capture(analyze);
        }

        // If analyze requested but description empty, run manually with prompt
        if (analyze && s.description.empty() && !prompt.empty()) {
            s.description = reader.analyze(s.path, prompt);
        }

        std::ostringstream oss;
        oss << "Screenshot captured: " << s.path << "\n";
        oss << "Resolution: " << s.width << "x" << s.height << "\n";
        if (!s.description.empty())
            oss << "Screen contents:\n" << s.description;
        else
            oss << "(Vision analysis not available — use the path to reference this screenshot)";

        result.status  = ToolStatus::SUCCESS;
        result.output  = oss.str();
    } catch (const std::exception& e) {
        result.status        = ToolStatus::FAILURE;
        result.error_message = e.what();
        result.output        = "Screenshot failed: " + std::string(e.what());
    }

    result.duration_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
    return result;
}

} // namespace cardinal

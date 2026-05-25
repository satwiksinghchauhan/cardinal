#pragma once
// =============================================================================
// Cardinal - Screen Reader
// File: src/computer/screen_reader.h
//
// Captures screenshots using scrot (X11) or grim (Wayland), then optionally
// runs moondream2 vision analysis via VisionEncoder to produce a text
// description of the screen state.
//
// Full screen or region captures are supported.
// Results are cached by region hash to avoid redundant vision calls.
// =============================================================================

#include "computer/computer_types.h"
#include "computer/display_detector.h"
#include "utils/config_loader.h"

#include <string>
#include <memory>
#include <optional>

namespace cardinal {

    class VisionEncoder;

    class ScreenReader {
    public:
        explicit ScreenReader(const DisplayDetector&       display,
                              const CardinalConfig&        config,
                              VisionEncoder*               vision = nullptr);
        ~ScreenReader() = default;

        ScreenReader(const ScreenReader&)            = delete;
        ScreenReader& operator=(const ScreenReader&) = delete;

        // ------------------------------------------------------------------
        // Screenshot
        // ------------------------------------------------------------------

        // Capture full screen. Returns Screenshot with path set.
        // If analyze=true and VisionEncoder is available, also sets description.
        Screenshot capture(bool analyze = false);

        // Capture a specific region.
        Screenshot capture_region(const ScreenRegion& region, bool analyze = false);

        // ------------------------------------------------------------------
        // Analysis
        // ------------------------------------------------------------------

        // Run vision analysis on an existing screenshot path.
        // Returns description string or empty on failure.
        std::string analyze(const std::string& image_path,
                            const std::string& prompt = "");

        // Describe what is currently on screen (capture + analyze).
        std::string describe_screen(const std::string& prompt = "");

        // Find the coordinates of a UI element described in natural language.
        // Returns the center point of the matched element, or nullopt.
        std::optional<Point> find_element(const std::string& description);

    private:
        std::string capture_x11(const std::optional<ScreenRegion>& region) const;
        std::string capture_wayland(const std::optional<ScreenRegion>& region) const;
        std::string output_path(const std::string& suffix = "") const;

        const DisplayDetector& display_;
        const CardinalConfig&  config_;
        VisionEncoder*         vision_;
        int                    capture_count_ = 0;
    };

} // namespace cardinal

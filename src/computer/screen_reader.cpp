// =============================================================================
// Cardinal - Screen Reader Implementation (v1.5.0)
// File: src/computer/screen_reader.cpp
// =============================================================================

#include "computer/screen_reader.h"
#include "vision/vision_encoder.h"
#include "vision/vision_types.h"
#include "utils/logger.h"

#include <filesystem>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <stdexcept>

namespace fs = std::filesystem;

namespace cardinal {

static std::string iso_now_compact() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    return oss.str();
}

ScreenReader::ScreenReader(const DisplayDetector& display,
                           const CardinalConfig&  config,
                           VisionEncoder*         vision)
    : display_(display), config_(config), vision_(vision)
{}

std::string ScreenReader::output_path(const std::string& suffix) const {
    fs::create_directories("data/screenshots");
    std::string name = "data/screenshots/screen_" + iso_now_compact();
    if (!suffix.empty()) name += "_" + suffix;
    name += ".png";
    return fs::absolute(name).string();
}

std::string ScreenReader::capture_x11(
        const std::optional<ScreenRegion>& region) const {
    if (!display_.has_scrot())
        throw std::runtime_error("scrot not available");

    std::string path = output_path();
    std::string cmd;
    if (region) {
        cmd = "scrot -a " + std::to_string(region->x) + "," +
              std::to_string(region->y) + "," +
              std::to_string(region->width) + "," +
              std::to_string(region->height) +
              " " + path + " 2>/dev/null";
    } else {
        cmd = "scrot " + path + " 2>/dev/null";
    }
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(path))
        throw std::runtime_error("scrot failed (exit=" + std::to_string(rc) + ")");
    return path;
}

std::string ScreenReader::capture_wayland(
        const std::optional<ScreenRegion>& region) const {
    if (!display_.has_grim())
        throw std::runtime_error("grim not available");

    std::string path = output_path();
    std::string cmd;
    if (region) {
        cmd = "grim -g \"" +
              std::to_string(region->x) + "," +
              std::to_string(region->y) + " " +
              std::to_string(region->width) + "x" +
              std::to_string(region->height) +
              "\" " + path + " 2>/dev/null";
    } else {
        cmd = "grim " + path + " 2>/dev/null";
    }
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(path))
        throw std::runtime_error("grim failed (exit=" + std::to_string(rc) + ")");
    return path;
}

Screenshot ScreenReader::capture(bool analyze) {
    Screenshot s;
    s.timestamp = iso_now_compact();

    if (display_.is_headless())
        throw std::runtime_error("Cannot capture screenshot in headless mode");

    if (display_.is_x11())
        s.path = capture_x11(std::nullopt);
    else
        s.path = capture_wayland(std::nullopt);

    s.width  = display_.info().width;
    s.height = display_.info().height;
    ++capture_count_;
    LOG_DEBUG("ScreenReader: captured " + s.path);

    if (analyze && vision_) {
        s.description = this->analyze(s.path);
        s.analyzed    = !s.description.empty();
    }
    return s;
}

Screenshot ScreenReader::capture_region(const ScreenRegion& region, bool analyze) {
    Screenshot s;
    s.timestamp = iso_now_compact();
    s.region    = region;

    if (display_.is_headless())
        throw std::runtime_error("Cannot capture screenshot in headless mode");

    if (display_.is_x11())
        s.path = capture_x11(region);
    else
        s.path = capture_wayland(region);

    s.width  = region.width;
    s.height = region.height;
    ++capture_count_;

    if (analyze && vision_) {
        s.description = this->analyze(s.path);
        s.analyzed    = !s.description.empty();
    }
    return s;
}

std::string ScreenReader::analyze(const std::string& image_path,
                                   const std::string& prompt) {
    if (!vision_) {
        LOG_WARN("ScreenReader: vision encoder not available");
        return "";
    }
    if (!fs::exists(image_path)) {
        LOG_WARN("ScreenReader: image not found: " + image_path);
        return "";
    }
    try {
        std::string p = prompt.empty()
            ? "Describe everything visible on this screen in detail. "
              "Include visible text, UI elements, open applications, "
              "and any notable content."
            : prompt;

        // VisionEncoder::encode() takes image_path, prompt, ImageMetadata
        ImageMetadata meta;
        meta.origin     = image_path;
        meta.cache_path = image_path;
        meta.source     = ImageSource::FILE;
        auto result = vision_->encode(image_path, p, meta);
        return result.description;
    } catch (const std::exception& e) {
        LOG_ERROR("ScreenReader: vision analysis failed: " + std::string(e.what()));
        return "";
    }
}

std::string ScreenReader::describe_screen(const std::string& prompt) {
    auto s = capture(false);
    return analyze(s.path, prompt);
}

std::optional<Point> ScreenReader::find_element(const std::string& description) {
    if (!vision_ || display_.is_headless()) return std::nullopt;

    auto s = capture(false);
    std::string prompt =
        "Look at this screenshot. Find the UI element described as: '" +
        description + "'. "
        "Respond ONLY with the x,y pixel coordinates of its center, like: "
        "'x=123 y=456'. If not found, respond: 'not found'.";

    std::string result = analyze(s.path, prompt);
    if (result.find("not found") != std::string::npos) return std::nullopt;

    int x = -1, y = -1;
    if (sscanf(result.c_str(), "x=%d y=%d", &x, &y) == 2 && x >= 0 && y >= 0)
        return Point{x, y};
    if (sscanf(result.c_str(), "%d, %d", &x, &y) == 2 && x >= 0 && y >= 0)
        return Point{x, y};

    LOG_DEBUG("ScreenReader: could not parse element coordinates: " + result);
    return std::nullopt;
}

} // namespace cardinal

#pragma once
// =============================================================================
// Cardinal - Vision Types
// File: src/vision/vision_types.h
//
// Shared types for the vision subsystem.
// No dependencies on other Cardinal headers — safe to include anywhere.
// =============================================================================

#include <string>
#include <cstdint>

namespace cardinal {

    // -------------------------------------------------------------------------
    // ImageFormat
    // -------------------------------------------------------------------------
    enum class ImageFormat {
        JPEG,
        PNG,
        BMP,
        GIF,
        WEBP,
        UNKNOWN
    };

    inline std::string image_format_to_string(ImageFormat f) {
        switch (f) {
            case ImageFormat::JPEG:  return "JPEG";
            case ImageFormat::PNG:   return "PNG";
            case ImageFormat::BMP:   return "BMP";
            case ImageFormat::GIF:   return "GIF";
            case ImageFormat::WEBP:  return "WEBP";
            default:                 return "UNKNOWN";
        }
    }

    inline ImageFormat image_format_from_extension(const std::string& ext) {
        if (ext == ".jpg" || ext == ".jpeg") return ImageFormat::JPEG;
        if (ext == ".png")                   return ImageFormat::PNG;
        if (ext == ".bmp")                   return ImageFormat::BMP;
        if (ext == ".gif")                   return ImageFormat::GIF;
        if (ext == ".webp")                  return ImageFormat::WEBP;
        return ImageFormat::UNKNOWN;
    }

    // -------------------------------------------------------------------------
    // ImageSource
    // -------------------------------------------------------------------------
    enum class ImageSource {
        FILE,       // Local file path
        URL,        // Downloaded from URL (cached)
        UNKNOWN
    };

    inline std::string image_source_to_string(ImageSource s) {
        switch (s) {
            case ImageSource::FILE: return "file";
            case ImageSource::URL:  return "url";
            default:                return "unknown";
        }
    }

    // -------------------------------------------------------------------------
    // ImageMetadata
    // Information about the image extracted before encoding.
    // -------------------------------------------------------------------------
    struct ImageMetadata {
        int         width      = 0;
        int         height     = 0;
        int64_t     file_size  = 0;       // bytes
        ImageFormat format     = ImageFormat::UNKNOWN;
        ImageSource source     = ImageSource::UNKNOWN;
        std::string origin;               // original path or URL
        std::string cache_path;           // local path (may equal origin for files)

        std::string format_string() const {
            // e.g. "1920x1080 JPEG, 245KB, source: url"
            std::string s;
            if (width > 0 && height > 0)
                s += std::to_string(width) + "x" + std::to_string(height) + " ";
            s += image_format_to_string(format);
            if (file_size > 0) {
                if (file_size >= 1024 * 1024)
                    s += ", " + std::to_string(file_size / (1024*1024)) + "MB";
                else
                    s += ", " + std::to_string(file_size / 1024) + "KB";
            }
            s += ", source: " + image_source_to_string(source);
            return s;
        }
    };

    // -------------------------------------------------------------------------
    // VisionResult
    // Output of a single analyze_image call.
    // -------------------------------------------------------------------------
    struct VisionResult {
        bool          success       = false;
        std::string   description;          // moondream2 output text
        ImageMetadata metadata;
        std::string   error_message;
        int           duration_ms   = 0;

        // Formatted output for injection into model context (Option C format)
        std::string format_for_context() const {
            if (!success) return "Vision analysis failed: " + error_message;
            std::string out = "[Image: " + metadata.format_string() + "]\n";
            out += "Description: " + description;
            return out;
        }
    };

} // namespace cardinal

// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Vision Cache
// File: src/vision/vision_cache.h
//
// Download cache for URL images.
// - Downloads image from URL, stores locally with SHA256-based filename
// - TTL eviction: files older than ttl_hours are deleted on next access
// - TTL = 0: keep forever (never evict)
// - Thread-safe
// =============================================================================

#include "vision/vision_types.h"
#include "utils/config_loader.h"

#include <string>
#include <mutex>

namespace cardinal {

    class VisionCache {
    public:
        explicit VisionCache(const CardinalConfig& config);

        // ------------------------------------------------------------------
        // Initialization — creates cache directory
        // ------------------------------------------------------------------
        void init();

        // ------------------------------------------------------------------
        // Resolve an image input to a local file path.
        // - If input is a file path: validates it exists, returns as-is
        // - If input is a URL: downloads to cache, returns local path
        // Returns empty string on failure, sets error_out.
        // ------------------------------------------------------------------
        std::string resolve(const std::string& input,
                            ImageMetadata&      metadata_out,
                            std::string&        error_out) const;

        // ------------------------------------------------------------------
        // Evict entries older than TTL. Called on each resolve() if TTL > 0.
        // ------------------------------------------------------------------
        void evict_expired() const;

        // ------------------------------------------------------------------
        // Clear entire cache
        // ------------------------------------------------------------------
        void clear() const;

        int cached_count() const;

    private:
        // Download a URL to a local cache file. Returns local path or "".
        std::string download(const std::string& url,
                             std::string&        error_out) const;

        // Compute SHA256 of a string, return first 16 hex chars
        static std::string url_hash(const std::string& url);

        // Probe image dimensions and format from file header
        static void probe_image(const std::string& path,
                                 ImageMetadata&     meta);

        const CardinalConfig& config_;
        mutable std::mutex    mutex_;
    };

} // namespace cardinal

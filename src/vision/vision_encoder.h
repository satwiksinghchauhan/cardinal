
#pragma once

// =============================================================================
// Cardinal - Vision Encoder
// File: src/vision/vision_encoder.h
//
// Wraps moondream2 via llama.cpp's libmtmd multimodal API.
// mtmd.h lives in vendor/llama.cpp/tools/mtmd/
// Requires CARDINAL_MTMD_AVAILABLE compile flag (set by CMake when found).
// If not available, encode() returns a clear error — no crash.
// =============================================================================

#include "vision/vision_types.h"
#include "utils/config_loader.h"

#ifdef CARDINAL_MTMD_AVAILABLE
#include "mtmd.h"
#include "mtmd-helper.h"
#endif
#include "llama.h"

#include <string>
#include <mutex>

namespace cardinal {

    class VisionEncoder {
    public:
        explicit VisionEncoder(const CardinalConfig& config);
        ~VisionEncoder();

        void load();
        bool is_ready() const { return ready_; }
        void unload();

        VisionResult encode(const std::string&   image_path,
                            const std::string&   prompt,
                            const ImageMetadata& metadata) const;

        VisionEncoder(const VisionEncoder&)            = delete;
        VisionEncoder& operator=(const VisionEncoder&) = delete;

    private:
        int ctx_params_n_batch() const;

        const CardinalConfig& config_;

        llama_model*   model_  = nullptr;
        llama_context* ctx_    = nullptr;
#ifdef CARDINAL_MTMD_AVAILABLE
        mtmd_context*  mtmd_   = nullptr;
#else
        void*          mtmd_   = nullptr;  // stub
#endif
        bool           ready_  = false;
        mutable std::mutex mutex_;

        static constexpr const char* DEFAULT_PROMPT =
            "Describe this image in detail, including all visible objects, "
            "text, colors, spatial relationships, and any notable features.";
    };

} // namespace cardinal

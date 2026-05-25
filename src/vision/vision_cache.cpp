// =============================================================================
// Cardinal - Vision Cache Implementation
// File: src/vision/vision_cache.cpp
// =============================================================================

#include "vision/vision_cache.h"
#include "utils/logger.h"

#include <httplib.h>
#include <openssl/sha.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstring>

namespace cardinal {

    namespace {
        // Detect image format from file magic bytes
        ImageFormat probe_format(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) return ImageFormat::UNKNOWN;

            unsigned char buf[12] = {};
            f.read(reinterpret_cast<char*>(buf), sizeof(buf));

            // JPEG: FF D8 FF
            if (buf[0] == 0xFF && buf[1] == 0xD8 && buf[2] == 0xFF)
                return ImageFormat::JPEG;
            // PNG: 89 50 4E 47 0D 0A 1A 0A
            if (buf[0] == 0x89 && buf[1] == 0x50 && buf[2] == 0x4E && buf[3] == 0x47)
                return ImageFormat::PNG;
            // BMP: 42 4D
            if (buf[0] == 0x42 && buf[1] == 0x4D)
                return ImageFormat::BMP;
            // GIF: 47 49 46
            if (buf[0] == 0x47 && buf[1] == 0x49 && buf[2] == 0x46)
                return ImageFormat::GIF;
            // WEBP: RIFF....WEBP
            if (buf[0] == 0x52 && buf[1] == 0x49 && buf[2] == 0x46 && buf[3] == 0x46 &&
                buf[8] == 0x57 && buf[9] == 0x45 && buf[10] == 0x42 && buf[11] == 0x50)
                return ImageFormat::WEBP;

            return ImageFormat::UNKNOWN;
        }

        // Read PNG dimensions from IHDR chunk (bytes 16-23)
        // Read JPEG dimensions by scanning for SOF marker
        std::pair<int,int> probe_dimensions(const std::string& path,
                                             ImageFormat fmt)
        {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) return {0, 0};

            if (fmt == ImageFormat::PNG) {
                f.seekg(16);
                unsigned char buf[8] = {};
                f.read(reinterpret_cast<char*>(buf), 8);
                int w = (buf[0]<<24)|(buf[1]<<16)|(buf[2]<<8)|buf[3];
                int h = (buf[4]<<24)|(buf[5]<<16)|(buf[6]<<8)|buf[7];
                return {w, h};
            }

            if (fmt == ImageFormat::JPEG) {
                f.seekg(0);
                std::vector<unsigned char> data(
                    (std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
                for (size_t i = 0; i + 8 < data.size(); ++i) {
                    if (data[i] == 0xFF &&
                        (data[i+1] == 0xC0 || data[i+1] == 0xC2)) {
                        int h = (data[i+5]<<8)|data[i+6];
                        int w = (data[i+7]<<8)|data[i+8];
                        return {w, h};
                    }
                }
            }

            return {0, 0};
        }
    }

    // =========================================================================
    // Constructor / init
    // =========================================================================

    VisionCache::VisionCache(const CardinalConfig& config)
        : config_(config)
    {}

    void VisionCache::init() {
        std::filesystem::create_directories(config_.vision.cache_path);
        LOG_INFO("VisionCache: initialized at " + config_.vision.cache_path +
                 " (TTL=" + std::to_string(config_.vision.cache_ttl_hours) + "h)");
    }

    // =========================================================================
    // resolve
    // =========================================================================

    std::string VisionCache::resolve(const std::string& input,
                                      ImageMetadata&      meta_out,
                                      std::string&        error_out) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (input.empty()) {
            error_out = "Empty image input";
            return "";
        }

        // Determine if URL or file path
        bool is_url = (input.substr(0, 7) == "http://" ||
                       input.substr(0, 8) == "https://");

        std::string local_path;

        if (is_url) {
            // Check cache first
            std::string hash     = url_hash(input);
            std::string cache_dir = config_.vision.cache_path;
            if (cache_dir.back() != '/') cache_dir += '/';

            // Try common extensions
            for (const auto& ext : {".jpg", ".png", ".webp", ".gif", ".bmp"}) {
                std::string candidate = cache_dir + hash + ext;
                if (std::filesystem::exists(candidate)) {
                    LOG_DEBUG("VisionCache: cache hit for " + input);
                    local_path = candidate;
                    meta_out.source    = ImageSource::URL;
                    meta_out.origin    = input;
                    meta_out.cache_path = local_path;
                    break;
                }
            }

            // Not in cache — download
            if (local_path.empty()) {
                // Evict expired before downloading
                if (config_.vision.cache_ttl_hours > 0) {
                    // Evict without lock (we hold it already) — inline eviction
                    auto now = std::filesystem::file_time_type::clock::now();
                    auto ttl = std::chrono::hours(config_.vision.cache_ttl_hours);
                    try {
                        for (const auto& entry :
                             std::filesystem::directory_iterator(config_.vision.cache_path)) {
                            if (!entry.is_regular_file()) continue;
                            auto age = now - entry.last_write_time();
                            if (age > ttl) {
                                std::filesystem::remove(entry.path());
                                LOG_DEBUG("VisionCache: evicted " +
                                          entry.path().string());
                            }
                        }
                    } catch (...) {}
                }

                local_path = download(input, error_out);
                if (local_path.empty()) return "";
            }

            meta_out.source     = ImageSource::URL;
            meta_out.origin     = input;
            meta_out.cache_path = local_path;

        } else {
            // File path — resolve ~ and validate
            std::string resolved = input;
            if (!resolved.empty() && resolved[0] == '~') {
                const char* home = std::getenv("HOME");
                if (home) resolved = std::string(home) + resolved.substr(1);
            }

            if (!std::filesystem::exists(resolved)) {
                error_out = "Image file not found: " + resolved;
                return "";
            }

            // Check allowed paths
            bool allowed = false;
            for (const auto& ap : config_.vision.allowed_paths) {
                std::string rap = ap;
                if (!rap.empty() && rap[0] == '~') {
                    const char* home = std::getenv("HOME");
                    if (home) rap = std::string(home) + rap.substr(1);
                }
                try {
                    auto ca = std::filesystem::weakly_canonical(rap).string();
                    auto cp = std::filesystem::weakly_canonical(resolved).string();
                    if (ca.back() != '/') ca += '/';
                    if (cp.substr(0, ca.size()) == ca ||
                        cp == ca.substr(0, ca.size()-1)) {
                        allowed = true;
                        break;
                    }
                } catch (...) {}
            }

            // Also allow if home_access is set
            if (!allowed && config_.tools.home_access) {
                const char* home = std::getenv("HOME");
                if (home) {
                    std::string hp = std::string(home);
                    if (resolved.substr(0, hp.size()) == hp) allowed = true;
                }
            }

            if (!allowed) {
                error_out = "Image path not in vision.allowed_paths: " + resolved;
                return "";
            }

            local_path          = resolved;
            meta_out.source     = ImageSource::FILE;
            meta_out.origin     = input;
            meta_out.cache_path = local_path;
        }

        // Probe metadata
        probe_image(local_path, meta_out);
        return local_path;
    }

    // =========================================================================
    // download
    // =========================================================================

    std::string VisionCache::download(const std::string& url,
                                       std::string&        error_out) const
    {
        LOG_INFO("VisionCache: downloading " + url);

        try {
            // Parse URL
            bool https = url.substr(0, 8) == "https://";
            std::string stripped = url.substr(https ? 8 : 7);
            auto slash = stripped.find('/');
            std::string host = (slash == std::string::npos)
                               ? stripped : stripped.substr(0, slash);
            std::string path_str = (slash == std::string::npos)
                               ? "/" : stripped.substr(slash);

            httplib::Client cli((https ? "https://" : "http://") + host);
            cli.set_connection_timeout(config_.vision.download_timeout_seconds);
            cli.set_read_timeout(config_.vision.download_timeout_seconds);
            cli.set_follow_location(true);

            auto res = cli.Get(path_str);
            if (!res || res->status != 200) {
                error_out = "HTTP " + (res ? std::to_string(res->status) : "failed") +
                            " downloading " + url;
                return "";
            }

            // Determine extension from Content-Type
            std::string ext = ".jpg"; // default
            auto ct = res->get_header_value("Content-Type");
            if (ct.find("png")  != std::string::npos) ext = ".png";
            else if (ct.find("webp") != std::string::npos) ext = ".webp";
            else if (ct.find("gif")  != std::string::npos) ext = ".gif";
            else if (ct.find("bmp")  != std::string::npos) ext = ".bmp";

            // Write to cache
            std::string hash       = url_hash(url);
            std::string cache_dir  = config_.vision.cache_path;
            if (cache_dir.back() != '/') cache_dir += '/';
            std::string local_path = cache_dir + hash + ext;

            std::ofstream f(local_path, std::ios::binary);
            if (!f.is_open()) {
                error_out = "Cannot write to cache: " + local_path;
                return "";
            }
            f.write(res->body.data(), static_cast<std::streamsize>(res->body.size()));

            LOG_INFO("VisionCache: downloaded " + std::to_string(res->body.size()) +
                     " bytes → " + local_path);
            return local_path;

        } catch (const std::exception& e) {
            error_out = std::string("Download error: ") + e.what();
            return "";
        }
    }

    // =========================================================================
    // url_hash — first 16 hex chars of SHA256
    // =========================================================================

    std::string VisionCache::url_hash(const std::string& url) {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(url.data()),
               url.size(), hash);
        std::ostringstream oss;
        for (int i = 0; i < 8; ++i)
            oss << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(hash[i]);
        return oss.str();
    }

    // =========================================================================
    // probe_image
    // =========================================================================

    void VisionCache::probe_image(const std::string& path,
                                   ImageMetadata&     meta)
    {
        try {
            meta.file_size = static_cast<int64_t>(
                std::filesystem::file_size(path));
        } catch (...) {}

        meta.format = probe_format(path);
        auto [w, h] = probe_dimensions(path, meta.format);
        meta.width  = w;
        meta.height = h;

        // Fill cache_path if not already set
        if (meta.cache_path.empty()) meta.cache_path = path;
    }

    // =========================================================================
    // evict_expired
    // =========================================================================

    void VisionCache::evict_expired() const {
        if (config_.vision.cache_ttl_hours == 0) return; // keep forever

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::filesystem::file_time_type::clock::now();
        auto ttl = std::chrono::hours(config_.vision.cache_ttl_hours);

        int evicted = 0;
        try {
            for (const auto& entry :
                 std::filesystem::directory_iterator(config_.vision.cache_path)) {
                if (!entry.is_regular_file()) continue;
                if (now - entry.last_write_time() > ttl) {
                    std::filesystem::remove(entry.path());
                    ++evicted;
                }
            }
        } catch (...) {}

        if (evicted > 0)
            LOG_DEBUG("VisionCache: evicted " + std::to_string(evicted) +
                      " expired entries");
    }

    // =========================================================================
    // clear / count
    // =========================================================================

    void VisionCache::clear() const {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            for (const auto& entry :
                 std::filesystem::directory_iterator(config_.vision.cache_path))
                if (entry.is_regular_file())
                    std::filesystem::remove(entry.path());
        } catch (...) {}
        LOG_INFO("VisionCache: cleared");
    }

    int VisionCache::cached_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int count = 0;
        try {
            for (const auto& entry :
                 std::filesystem::directory_iterator(config_.vision.cache_path))
                if (entry.is_regular_file()) ++count;
        } catch (...) {}
        return count;
    }

} // namespace cardinal

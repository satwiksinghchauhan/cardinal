#pragma once
// =============================================================================
// Cardinal - AT-SPI2 Accessibility Tree Reader
// File: src/computer/atspi_reader.h
//
// Reads the AT-SPI2 accessibility tree of any running GTK/Qt application
// via a Python subprocess (pyatspi). Returns structured AtSpiNode trees.
//
// Primary use: find UI elements without needing vision/screenshots.
// Falls back to ScreenReader+vision when AT-SPI is unavailable or fails.
// =============================================================================

#include "computer/computer_types.h"
#include <nlohmann/json.hpp>
#include "utils/config_loader.h"

#include <string>
#include <vector>
#include <optional>

namespace cardinal {

    class AtSpiReader {
    public:
        explicit AtSpiReader(const CardinalConfig& config);
        ~AtSpiReader() = default;

        AtSpiReader(const AtSpiReader&)            = delete;
        AtSpiReader& operator=(const AtSpiReader&) = delete;

        bool is_available() const;

        // Get the full accessibility tree for a running application.
        // app_name: executable name or window title (fuzzy match)
        std::optional<AtSpiNode> get_tree(const std::string& app_name) const;

        // Find all nodes matching a role and/or name (fuzzy).
        std::vector<AtSpiNode> find_nodes(const std::string& app_name,
                                          const std::string& role = "",
                                          const std::string& name = "") const;

        // Find the first focusable element matching a description.
        std::optional<AtSpiNode> find_element(const std::string& app_name,
                                              const std::string& description) const;

        // Get the bounds of a named element (for click targeting).
        std::optional<ScreenRegion> get_bounds(const std::string& app_name,
                                               const std::string& element_name) const;

    private:
        std::string run_python(const std::string& script) const;
        static AtSpiNode parse_node_json(const nlohmann::json& j, int depth = 0);
        static std::vector<AtSpiNode> collect_nodes(const AtSpiNode& root,
                                                     const std::string& role,
                                                     const std::string& name);
        static bool fuzzy_match(const std::string& a, const std::string& b);

        const CardinalConfig& config_;
    };

} // namespace cardinal

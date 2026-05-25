// =============================================================================
// Cardinal - AT-SPI2 Reader Implementation
// File: src/computer/atspi_reader.cpp
// =============================================================================

#include "computer/atspi_reader.h"
#include "utils/logger.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <cctype>
#include <fstream>

using json = nlohmann::json;

namespace cardinal {

// ---------------------------------------------------------------------------
// Helper: run a Python script string
// ---------------------------------------------------------------------------

AtSpiReader::AtSpiReader(const CardinalConfig& config)
    : config_(config)
{}

bool AtSpiReader::is_available() const {
    return config_.computer_use.atspi.enabled &&
           std::system("python3 -c 'import pyatspi' >/dev/null 2>&1") == 0;
}

bool AtSpiReader::fuzzy_match(const std::string& a, const std::string& b) {
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    return lower(a).find(lower(b)) != std::string::npos ||
           lower(b).find(lower(a)) != std::string::npos;
}

std::string AtSpiReader::run_python(const std::string& script) const {
    std::string path = "/tmp/cardinal_atspi_" + std::to_string(getpid()) + ".py";
    {
        std::ofstream f(path);
        if (!f) return "";
        f << script;
    }
    std::string cmd = "python3 " + path + " 2>/dev/null";
    std::array<char, 32768> buf{};
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { remove(path.c_str()); return ""; }
    while (fgets(buf.data(), buf.size(), pipe)) output += buf.data();
    pclose(pipe);
    remove(path.c_str());
    return output;
}

// ---------------------------------------------------------------------------
// Python AT-SPI tree dump script
// ---------------------------------------------------------------------------

static std::string build_tree_script(const std::string& app_name) {
    std::string script;
    script += "import pyatspi, json, sys\n\n";
    script += "def node_to_dict(acc, depth=0):\n";
    script += "    if depth > 8: return None\n";
    script += "    try:\n";
    script += "        name   = acc.name or ''\n";
    script += "        role   = acc.getRoleName() or ''\n";
    script += "        state  = [s.valueToString(s) for s in acc.getState().getStates()]\n";
    script += "        bounds = acc.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)\n";
    script += "        rect   = {'x': bounds.x, 'y': bounds.y, 'width': bounds.width, 'height': bounds.height}\n";
    script += "    except:\n";
    script += "        return None\n";
    script += "    children = []\n";
    script += "    for i in range(acc.childCount):\n";
    script += "        try:\n";
    script += "            child = acc.getChildAtIndex(i)\n";
    script += "            child_dict = node_to_dict(child, depth+1)\n";
    script += "            if child_dict: children.append(child_dict)\n";
    script += "        except: pass\n";
    script += "    return {'role': role, 'name': name, 'bounds': rect, 'states': state, 'children': children}\n\n";
    script += "desktop = pyatspi.Registry.getDesktop(0)\n";
    script += "app_name = '" + app_name + "'\n";
    script += "for app in desktop:\n";
    script += "    try:\n";
    script += "        if not app: continue\n";
    script += "        if app_name.lower() in (app.name or '').lower():\n";
    script += "            tree = node_to_dict(app)\n";
    script += "            if tree:\n";
    script += "                print(json.dumps(tree))\n";
    script += "                import sys; sys.exit(0)\n";
    script += "    except: pass\n";
    script += "print('{}')\n";
    return script;
}

// ---------------------------------------------------------------------------
// parse_node_json
// ---------------------------------------------------------------------------

AtSpiNode AtSpiReader::parse_node_json(const json& j, int depth) {
    AtSpiNode node;
    if (!j.is_object()) return node;

    node.role        = j.value("role", "");
    node.name        = j.value("name", "");
    node.description = j.value("description", "");
    node.value       = j.value("value", "");

    if (j.contains("bounds") && j["bounds"].is_object()) {
        auto& b = j["bounds"];
        node.bounds = {
            b.value("x", 0), b.value("y", 0),
            b.value("width", 0), b.value("height", 0)
        };
    }

    if (j.contains("states") && j["states"].is_array()) {
        for (const auto& s : j["states"]) {
            std::string sv = s.get<std::string>();
            node.states.push_back(sv);
            if (sv == "focusable") node.focusable = true;
            if (sv == "focused")   node.focused   = true;
            if (sv == "enabled")   node.enabled   = true;
        }
    }

    if (depth < 8 && j.contains("children") && j["children"].is_array()) {
        for (const auto& child : j["children"]) {
            node.children.push_back(parse_node_json(child, depth + 1));
        }
    }
    return node;
}

// ---------------------------------------------------------------------------
// get_tree
// ---------------------------------------------------------------------------

std::optional<AtSpiNode> AtSpiReader::get_tree(const std::string& app_name) const {
    if (!is_available()) return std::nullopt;

    std::string output = run_python(build_tree_script(app_name));
    if (output.empty() || output == "{}") return std::nullopt;

    try {
        auto j = json::parse(output);
        if (j.is_object() && !j.empty())
            return parse_node_json(j);
    } catch (...) {}
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// collect_nodes (recursive flattener)
// ---------------------------------------------------------------------------

std::vector<AtSpiNode> AtSpiReader::collect_nodes(const AtSpiNode& root,
                                                    const std::string& role,
                                                    const std::string& name) {
    std::vector<AtSpiNode> result;
    bool role_ok = role.empty() || fuzzy_match(root.role, role);
    bool name_ok = name.empty() || fuzzy_match(root.name, name);
    if (role_ok && name_ok && !root.role.empty())
        result.push_back(root);
    for (const auto& child : root.children) {
        auto sub = collect_nodes(child, role, name);
        result.insert(result.end(), sub.begin(), sub.end());
    }
    return result;
}

// ---------------------------------------------------------------------------
// find_nodes
// ---------------------------------------------------------------------------

std::vector<AtSpiNode> AtSpiReader::find_nodes(const std::string& app_name,
                                                const std::string& role,
                                                const std::string& name) const {
    auto tree = get_tree(app_name);
    if (!tree) return {};
    return collect_nodes(*tree, role, name);
}

// ---------------------------------------------------------------------------
// find_element
// ---------------------------------------------------------------------------

std::optional<AtSpiNode> AtSpiReader::find_element(
        const std::string& app_name,
        const std::string& description) const {
    // Try to find by name first
    auto nodes = find_nodes(app_name, "", description);
    if (!nodes.empty()) return nodes[0];

    // Try role-based search (e.g. "submit button" → role=button, name=submit)
    auto words = [](const std::string& s) {
        std::vector<std::string> v;
        std::istringstream ss(s);
        std::string w;
        while (ss >> w) v.push_back(w);
        return v;
    }(description);

    static const std::vector<std::string> roles = {
        "button", "text", "entry", "link", "checkbox", "menu", "menuitem",
        "combobox", "listitem", "tab", "image", "label"
    };

    for (const auto& role : roles) {
        if (description.find(role) != std::string::npos) {
            std::string name_hint;
            for (const auto& w : words) {
                if (w != role) { name_hint = w; break; }
            }
            auto found = find_nodes(app_name, role, name_hint);
            if (!found.empty()) return found[0];
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// get_bounds
// ---------------------------------------------------------------------------

std::optional<ScreenRegion> AtSpiReader::get_bounds(
        const std::string& app_name,
        const std::string& element_name) const {
    auto node = find_element(app_name, element_name);
    if (!node) return std::nullopt;
    return node->bounds;
}

} // namespace cardinal

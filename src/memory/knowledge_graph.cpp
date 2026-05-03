// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
// =============================================================================
// Cardinal - Knowledge Graph Implementation
// File: src/memory/knowledge_graph.cpp
// =============================================================================

#include "knowledge_graph.h"
#include "utils/logger.h"

#include <algorithm>
#include <sstream>
#include <queue>
#include <set>

namespace cardinal {

    // =============================================================================
    // NodeType helpers
    // =============================================================================

    std::string node_type_to_string(NodeType t) {
        switch (t) {
        case NodeType::CONCEPT:  return "concept";
        case NodeType::FACT:     return "fact";
        case NodeType::ENTITY:   return "entity";
        case NodeType::RELATION: return "relation";
        default:                 return "unknown";
        }
    }

    NodeType node_type_from_string(const std::string& s) {
        if (s == "concept")  return NodeType::CONCEPT;
        if (s == "fact")     return NodeType::FACT;
        if (s == "entity")   return NodeType::ENTITY;
        if (s == "relation") return NodeType::RELATION;
        return NodeType::UNKNOWN;
    }

    // =============================================================================
    // Constructor
    // =============================================================================

    KnowledgeGraph::KnowledgeGraph(const CardinalConfig& config)
        : config_(config)
    {
        LOG_INFO("KnowledgeGraph created - path: " +
            config_.memory.knowledge_graph_path);
    }

    // =============================================================================
    // load
    // =============================================================================

    void KnowledgeGraph::load() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto nodes_vec = JsonParser::load_knowledge(
            config_.memory.knowledge_graph_path);

        nodes_.clear();
        edges_.clear();
        label_index_.clear();

        for (const auto& knode : nodes_vec) {
            GraphNode gnode;
            gnode.data = knode;
            gnode.type = node_type_from_string(knode.type);
            gnode.outgoing = knode.related_ids;
            nodes_[knode.id] = gnode;
            label_index_[knode.label] = knode.id;
        }

        // Rebuild incoming edges from outgoing
        for (auto& [id, gnode] : nodes_) {
            for (const auto& related_id : gnode.outgoing) {
                auto it = nodes_.find(related_id);
                if (it != nodes_.end()) {
                    it->second.incoming.push_back(id);
                }
            }
        }

        loaded_ = true;
        dirty_ = false;

        LOG_INFO("KnowledgeGraph loaded - " +
            std::to_string(nodes_.size()) + " nodes");
    }

    // =============================================================================
    // save
    // =============================================================================

    void KnowledgeGraph::save() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!dirty_) {
            LOG_DEBUG("KnowledgeGraph: no changes, skipping save");
            return;
        }

        std::vector<KnowledgeNode> nodes_vec;
        nodes_vec.reserve(nodes_.size());

        for (const auto& [id, gnode] : nodes_) {
            KnowledgeNode knode = gnode.data;
            knode.type = node_type_to_string(gnode.type);
            knode.related_ids = gnode.outgoing;
            nodes_vec.push_back(knode);
        }

        JsonParser::save_knowledge(config_.memory.knowledge_graph_path, nodes_vec);
        dirty_ = false;

        LOG_INFO("KnowledgeGraph saved - " +
            std::to_string(nodes_vec.size()) + " nodes");
    }

    // =============================================================================
    // add_node
    // =============================================================================

    std::string KnowledgeGraph::add_node(const std::string& label,
        NodeType           type,
        const std::string& content,
        float              confidence,
        const std::string& source) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check for existing node with same label
        auto label_it = label_index_.find(label);
        if (label_it != label_index_.end()) {
            // Update existing node
            auto& gnode = nodes_[label_it->second];
            gnode.data.content = content.size() > gnode.data.content.size()
                ? content : gnode.data.content;
            gnode.data.confidence = std::clamp(
                std::max(gnode.data.confidence, confidence) + 0.05f, 0.0f, 1.0f);
            gnode.data.updated_at = JsonParser::current_timestamp();
            dirty_ = true;
            LOG_DEBUG("KnowledgeGraph: updated node: " + label);
            return label_it->second;
        }

        // Check for similar labels
        for (const auto& [existing_label, existing_id] : label_index_) {
            if (is_similar_label(label, existing_label)) {
                LOG_DEBUG("KnowledgeGraph: similar node exists for: " + label);
                auto& gnode = nodes_[existing_id];
                gnode.data.confidence = std::clamp(
                    gnode.data.confidence + 0.02f, 0.0f, 1.0f);
                dirty_ = true;
                return existing_id;
            }
        }

        // Create new node
        KnowledgeNode knode;
        knode.id = JsonParser::generate_id();
        knode.label = label;
        knode.type = node_type_to_string(type);
        knode.content = content;
        knode.confidence = std::clamp(confidence, 0.0f, 1.0f);
        knode.source = source;
        knode.created_at = JsonParser::current_timestamp();

        GraphNode gnode;
        gnode.data = knode;
        gnode.type = type;

        nodes_[knode.id] = gnode;
        label_index_[label] = knode.id;
        dirty_ = true;

        LOG_DEBUG("KnowledgeGraph: added node [" +
            node_type_to_string(type) + "] " + label);

        return knode.id;
    }

    // =============================================================================
    // update_node_confidence
    // =============================================================================

    bool KnowledgeGraph::update_node_confidence(const std::string& node_id,
        float delta) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) return false;

        it->second.data.confidence = std::clamp(
            it->second.data.confidence + delta, 0.0f, 1.0f);
        it->second.data.updated_at = JsonParser::current_timestamp();
        dirty_ = true;
        return true;
    }

    // =============================================================================
    // get_node / get_node_by_label
    // =============================================================================

    std::optional<GraphNode> KnowledgeGraph::get_node(
        const std::string& node_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) return std::nullopt;
        return it->second;
    }

    std::optional<GraphNode> KnowledgeGraph::get_node_by_label(
        const std::string& label) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = label_index_.find(label);
        if (it == label_index_.end()) return std::nullopt;
        auto node_it = nodes_.find(it->second);
        if (node_it == nodes_.end()) return std::nullopt;
        return node_it->second;
    }

    // =============================================================================
    // remove_node
    // =============================================================================

    bool KnowledgeGraph::remove_node(const std::string& node_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) return false;

        // Remove from label index
        label_index_.erase(it->second.data.label);

        // Remove edges involving this node
        auto edge_it = edges_.begin();
        while (edge_it != edges_.end()) {
            if (edge_it->second.from_id == node_id ||
                edge_it->second.to_id == node_id) {
                edge_it = edges_.erase(edge_it);
            }
            else {
                ++edge_it;
            }
        }

        // Remove from neighbor lists of other nodes
        for (auto& [id, gnode] : nodes_) {
            gnode.outgoing.erase(
                std::remove(gnode.outgoing.begin(), gnode.outgoing.end(), node_id),
                gnode.outgoing.end());
            gnode.incoming.erase(
                std::remove(gnode.incoming.begin(), gnode.incoming.end(), node_id),
                gnode.incoming.end());
        }

        nodes_.erase(it);
        dirty_ = true;
        return true;
    }

    // =============================================================================
    // add_edge
    // =============================================================================

    bool KnowledgeGraph::add_edge(const std::string& from_id,
        const std::string& to_id,
        const std::string& relation,
        float              confidence) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Both nodes must exist
        if (nodes_.find(from_id) == nodes_.end() ||
            nodes_.find(to_id) == nodes_.end()) {
            LOG_WARN("KnowledgeGraph: add_edge failed - node not found");
            return false;
        }

        std::string key = make_edge_key(from_id, to_id, relation);

        // Update if exists
        if (edges_.count(key)) {
            edges_[key].confidence = std::clamp(
                edges_[key].confidence + 0.05f, 0.0f, 1.0f);
            dirty_ = true;
            return true;
        }

        Edge edge;
        edge.from_id = from_id;
        edge.to_id = to_id;
        edge.relation = relation;
        edge.confidence = std::clamp(confidence, 0.0f, 1.0f);
        edge.created_at = JsonParser::current_timestamp();

        edges_[key] = edge;

        // Update adjacency lists
        nodes_[from_id].outgoing.push_back(to_id);
        nodes_[to_id].incoming.push_back(from_id);

        // Sync data.related_ids
        nodes_[from_id].data.related_ids = nodes_[from_id].outgoing;

        dirty_ = true;
        return true;
    }

    // =============================================================================
    // remove_edge
    // =============================================================================

    bool KnowledgeGraph::remove_edge(const std::string& from_id,
        const std::string& to_id,
        const std::string& relation) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string key = make_edge_key(from_id, to_id, relation);
        if (!edges_.count(key)) return false;

        edges_.erase(key);

        // Update adjacency
        auto& from_node = nodes_[from_id];
        from_node.outgoing.erase(
            std::remove(from_node.outgoing.begin(),
                from_node.outgoing.end(), to_id),
            from_node.outgoing.end());
        from_node.data.related_ids = from_node.outgoing;

        auto& to_node = nodes_[to_id];
        to_node.incoming.erase(
            std::remove(to_node.incoming.begin(),
                to_node.incoming.end(), from_id),
            to_node.incoming.end());

        dirty_ = true;
        return true;
    }

    // =============================================================================
    // get_edges_from / get_edges_to
    // =============================================================================

    std::vector<Edge> KnowledgeGraph::get_edges_from(
        const std::string& node_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Edge> result;
        for (const auto& [key, edge] : edges_) {
            if (edge.from_id == node_id) result.push_back(edge);
        }
        return result;
    }

    std::vector<Edge> KnowledgeGraph::get_edges_to(
        const std::string& node_id) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Edge> result;
        for (const auto& [key, edge] : edges_) {
            if (edge.to_id == node_id) result.push_back(edge);
        }
        return result;
    }

    // =============================================================================
    // query
    // =============================================================================

    std::vector<GraphNode> KnowledgeGraph::query(const GraphQuery& q) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<GraphNode> results;

        for (const auto& [id, gnode] : nodes_) {
            // Type filter
            if (q.type_filter != NodeType::UNKNOWN &&
                gnode.type != q.type_filter) continue;

            // Confidence filter
            if (gnode.data.confidence < q.min_confidence) continue;

            // Label hint
            if (!q.label_hint.empty()) {
                if (gnode.data.label.find(q.label_hint) == std::string::npos)
                    continue;
            }

            // Content hint
            if (!q.content_hint.empty()) {
                if (gnode.data.content.find(q.content_hint) == std::string::npos &&
                    gnode.data.label.find(q.content_hint) == std::string::npos)
                    continue;
            }

            results.push_back(gnode);
        }

        // Sort by confidence descending
        std::sort(results.begin(), results.end(),
            [](const GraphNode& a, const GraphNode& b) {
                return a.data.confidence > b.data.confidence;
            });

        if (q.max_results > 0 &&
            static_cast<int>(results.size()) > q.max_results) {
            results.resize(q.max_results);
        }

        return results;
    }

    // =============================================================================
    // get_neighbors
    // BFS to depth N from a starting node
    // =============================================================================

    std::vector<GraphNode> KnowledgeGraph::get_neighbors(
        const std::string& node_id, int depth) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<GraphNode> result;
        std::unordered_set<std::string> visited;
        std::queue<std::pair<std::string, int>> bfs_queue;

        bfs_queue.push({ node_id, 0 });
        visited.insert(node_id);

        while (!bfs_queue.empty()) {
            auto [current_id, current_depth] = bfs_queue.front();
            bfs_queue.pop();

            if (current_depth >= depth) continue;

            auto it = nodes_.find(current_id);
            if (it == nodes_.end()) continue;

            for (const auto& neighbor_id : it->second.outgoing) {
                if (visited.count(neighbor_id)) continue;
                visited.insert(neighbor_id);

                auto neighbor_it = nodes_.find(neighbor_id);
                if (neighbor_it != nodes_.end()) {
                    result.push_back(neighbor_it->second);
                    bfs_queue.push({ neighbor_id, current_depth + 1 });
                }
            }
        }

        return result;
    }

    // =============================================================================
    // find_path - BFS shortest path
    // =============================================================================

    std::vector<std::string> KnowledgeGraph::find_path(
        const std::string& from_id,
        const std::string& to_id,
        int max_depth) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return bfs(from_id, to_id, max_depth);
    }

    std::vector<std::string> KnowledgeGraph::bfs(
        const std::string& from,
        const std::string& to,
        int max_depth) const
    {
        if (from == to) return { from };

        std::unordered_map<std::string, std::string> parent;
        std::queue<std::pair<std::string, int>> q;

        q.push({ from, 0 });
        parent[from] = "";

        while (!q.empty()) {
            auto [current, depth] = q.front();
            q.pop();

            if (depth >= max_depth) continue;

            auto it = nodes_.find(current);
            if (it == nodes_.end()) continue;

            for (const auto& neighbor : it->second.outgoing) {
                if (parent.count(neighbor)) continue;
                parent[neighbor] = current;

                if (neighbor == to) {
                    // Reconstruct path
                    std::vector<std::string> path;
                    std::string node = to;
                    while (!node.empty()) {
                        path.push_back(node);
                        node = parent[node];
                    }
                    std::reverse(path.begin(), path.end());
                    return path;
                }

                q.push({ neighbor, depth + 1 });
            }
        }

        return {}; // No path found
    }

    // =============================================================================
    // get_hub_nodes - nodes with most connections
    // =============================================================================

    std::vector<GraphNode> KnowledgeGraph::get_hub_nodes(int n) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::pair<int, std::string>> degree_list;
        for (const auto& [id, gnode] : nodes_) {
            int degree = static_cast<int>(gnode.outgoing.size() +
                gnode.incoming.size());
            degree_list.push_back({ degree, id });
        }

        std::sort(degree_list.begin(), degree_list.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            });

        std::vector<GraphNode> result;
        for (int i = 0; i < n && i < static_cast<int>(degree_list.size()); ++i) {
            auto it = nodes_.find(degree_list[i].second);
            if (it != nodes_.end()) result.push_back(it->second);
        }

        return result;
    }

    // =============================================================================
    // format_for_prompt
    // =============================================================================

    std::string KnowledgeGraph::format_for_prompt(
        const std::vector<GraphNode>& nodes, int max_chars) const
    {
        if (nodes.empty()) return "";

        std::ostringstream oss;
        oss << "## Relevant Knowledge\n";

        int chars = 0;
        for (const auto& gnode : nodes) {
            std::string entry = "- [" + node_type_to_string(gnode.type) + "] " +
                gnode.data.label + ": " +
                gnode.data.content + "\n";
            if (chars + static_cast<int>(entry.size()) > max_chars) break;
            oss << entry;
            chars += static_cast<int>(entry.size());
        }

        return oss.str();
    }

    // =============================================================================
    // Stats
    // =============================================================================

    int KnowledgeGraph::node_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(nodes_.size());
    }

    int KnowledgeGraph::edge_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(edges_.size());
    }

    bool KnowledgeGraph::empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.empty();
    }

    // =============================================================================
    // Internal helpers
    // =============================================================================

    bool KnowledgeGraph::is_similar_label(const std::string& a,
        const std::string& b) const {
        if (a == b) return true;
        // Case-insensitive check
        std::string la = a, lb = b;
        std::transform(la.begin(), la.end(), la.begin(), ::tolower);
        std::transform(lb.begin(), lb.end(), lb.begin(), ::tolower);
        return la == lb;
    }

    std::string KnowledgeGraph::make_edge_key(const std::string& from,
        const std::string& to,
        const std::string& rel) const {
        return from + "|" + to + "|" + rel;
    }

} // namespace cardinal
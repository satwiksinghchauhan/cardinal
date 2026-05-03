// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: Copyright (C) 2026 Satwik Singh (Cardinal AGI)
#pragma once
// =============================================================================
// Cardinal - Knowledge Graph
// File: src/memory/knowledge_graph.h
// Manages Cardinal's factual knowledge as a graph of nodes and edges.
// Nodes represent concepts, facts, entities. Edges represent relationships.
// Grows over time as Cardinal browses the internet and reasons about new info.
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
#ifdef ERROR
#undef ERROR
#endif

#include "utils/config_loader.h"
#include "utils/json_parser.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <mutex>

namespace cardinal {

    // -----------------------------------------------------------------------------
    // NodeType - semantic type of a knowledge node
    // -----------------------------------------------------------------------------
    enum class NodeType {
        CONCEPT,    // Abstract concept (e.g. "gravity", "democracy")
        FACT,       // Concrete factual claim (e.g. "water boils at 100C")
        ENTITY,     // Named entity (person, place, organization)
        RELATION,   // Relationship descriptor (e.g. "causes", "is part of")
        UNKNOWN
    };

    std::string node_type_to_string(NodeType t);
    NodeType    node_type_from_string(const std::string& s);

    // -----------------------------------------------------------------------------
    // Edge - directed relationship between two nodes
    // -----------------------------------------------------------------------------
    struct Edge {
        std::string from_id;        // Source node ID
        std::string to_id;          // Target node ID
        std::string relation;       // Relationship label (e.g. "causes", "is_a")
        float       confidence;     // How confident we are in this relationship
        std::string created_at;
    };

    // -----------------------------------------------------------------------------
    // GraphNode - extends KnowledgeNode with typed edges
    // -----------------------------------------------------------------------------
    struct GraphNode {
        KnowledgeNode            data;        // Core node data
        NodeType                 type;        // Semantic type
        std::vector<std::string> outgoing;    // IDs of nodes this node points to
        std::vector<std::string> incoming;    // IDs of nodes pointing to this node
    };

    // -----------------------------------------------------------------------------
    // GraphQuery
    // Parameters for searching the knowledge graph.
    // -----------------------------------------------------------------------------
    struct GraphQuery {
        std::string  label_hint;        // Fuzzy match on node label
        std::string  content_hint;      // Fuzzy match on node content
        NodeType     type_filter;       // Filter by node type (UNKNOWN = all)
        float        min_confidence;    // Minimum confidence
        int          max_results;       // 0 = all
        int          max_depth;         // For traversal queries (0 = no traversal)

        GraphQuery()
            : type_filter(NodeType::UNKNOWN)
            , min_confidence(0.0f)
            , max_results(10)
            , max_depth(0) {
        }
    };

    // -----------------------------------------------------------------------------
    // KnowledgeGraph
    // Thread-safe persistent knowledge graph.
    //
    // Lifecycle:
    //   1. load() at startup
    //   2. add_node() / add_edge() as Cardinal learns new facts
    //   3. query() to retrieve relevant knowledge for reasoning
    //   4. save() periodically
    // -----------------------------------------------------------------------------
    class KnowledgeGraph {
    public:
        explicit KnowledgeGraph(const CardinalConfig& config);

        // -------------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------------
        void load();
        void save();

        // -------------------------------------------------------------------------
        // Node management
        // -------------------------------------------------------------------------

        // Add a new node - returns node ID
        // If a node with similar label exists, updates it instead
        std::string add_node(const std::string& label,
            NodeType           type,
            const std::string& content,
            float              confidence = 0.5f,
            const std::string& source = "");

        // Update node confidence
        bool update_node_confidence(const std::string& node_id, float delta);

        // Get node by ID
        std::optional<GraphNode> get_node(const std::string& node_id) const;

        // Get node by label (exact match)
        std::optional<GraphNode> get_node_by_label(const std::string& label) const;

        // Remove node and all its edges
        bool remove_node(const std::string& node_id);

        // -------------------------------------------------------------------------
        // Edge management
        // -------------------------------------------------------------------------

        // Add a directed edge between two nodes
        bool add_edge(const std::string& from_id,
            const std::string& to_id,
            const std::string& relation,
            float              confidence = 0.5f);

        // Remove edge between two nodes
        bool remove_edge(const std::string& from_id,
            const std::string& to_id,
            const std::string& relation);

        // Get all edges from a node
        std::vector<Edge> get_edges_from(const std::string& node_id) const;

        // Get all edges to a node
        std::vector<Edge> get_edges_to(const std::string& node_id) const;

        // -------------------------------------------------------------------------
        // Query
        // -------------------------------------------------------------------------

        // Search nodes by query parameters
        std::vector<GraphNode> query(const GraphQuery& q) const;

        // Get all neighbors of a node (depth 1)
        std::vector<GraphNode> get_neighbors(const std::string& node_id,
            int depth = 1) const;

        // Find path between two nodes (BFS, max_depth hops)
        std::vector<std::string> find_path(const std::string& from_id,
            const std::string& to_id,
            int max_depth = 5) const;

        // Get most connected nodes (highest degree)
        std::vector<GraphNode> get_hub_nodes(int n = 10) const;

        // -------------------------------------------------------------------------
        // Serialization helpers for inference injection
        // -------------------------------------------------------------------------

        // Format relevant nodes as natural language for system prompt injection
        std::string format_for_prompt(const std::vector<GraphNode>& nodes,
            int max_chars = 500) const;

        // -------------------------------------------------------------------------
        // Stats
        // -------------------------------------------------------------------------
        int  node_count() const;
        int  edge_count() const;
        bool empty()      const;
        bool is_dirty()   const { return dirty_; }

    private:
        // -------------------------------------------------------------------------
        // Internal helpers
        // -------------------------------------------------------------------------
        bool        is_similar_label(const std::string& a,
            const std::string& b) const;
        std::string make_edge_key(const std::string& from,
            const std::string& to,
            const std::string& rel) const;

        // BFS helper for find_path
        std::vector<std::string> bfs(const std::string& from,
            const std::string& to,
            int max_depth) const;

        // -------------------------------------------------------------------------
        // Members
        // -------------------------------------------------------------------------
        const CardinalConfig& config_;
        std::unordered_map<std::string, GraphNode>  nodes_;       // ID -> GraphNode
        std::unordered_map<std::string, Edge>       edges_;       // key -> Edge
        std::unordered_map<std::string, std::string> label_index_; // label -> ID
        mutable std::mutex                           mutex_;
        bool                                         dirty_ = false;
        bool                                         loaded_ = false;
    };

    // -----------------------------------------------------------------------------
    // KnowledgeGraphError
    // -----------------------------------------------------------------------------
    class KnowledgeGraphError : public std::runtime_error {
    public:
        explicit KnowledgeGraphError(const std::string& message)
            : std::runtime_error("KnowledgeGraphError: " + message) {}
    };

} // namespace cardinal
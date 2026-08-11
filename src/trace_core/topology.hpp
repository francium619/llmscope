// topology.hpp - turns a flat stream of ggml node names into a browsable tree.
//
// ggml hands us ~300 unrelated tensor names per forward pass. llama.cpp names
// them consistently ("attn_norm-3", "ffn_up-12", "l_out-0"), so the transformer
// structure can be recovered by parsing, with no model-specific knowledge and no
// changes to llama.cpp. The tree is discovered once on the first forward pass and
// is stable thereafter, because the graph is identical on every token.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "record.hpp"

namespace llmscope {

// Result of classifying one tensor name.
struct ParsedName {
    NodeKind kind = NodeKind::Unknown;
    int32_t  layer = -1;  // -1 when the node is not inside a transformer block
};

// Pure function, unit-tested directly. Recognises the trailing "-N" suffix that
// llama.cpp appends per layer, and the "_lN" form used by KV cache views.
ParsedName parse_node_name(const std::string& name);

// A row in the topology tree. Group rows (the model root, "layers", "layers.3",
// "attn") have no name_ids of their own; leaf rows map to real graph nodes.
struct TopoNode {
    std::string label;
    NodeKind kind = NodeKind::Unknown;
    int32_t  layer = -1;
    int      parent = -1;
    int      depth = 0;
    bool     expanded = false;
    bool     is_group = false;
    std::vector<int> children;
    std::vector<uint16_t> name_ids;  // graph nodes represented by this row
};

class Topology {
public:
    Topology();

    // Called by the tracer the first time each distinct node name is seen.
    // Idempotent: observing the same name twice changes nothing.
    void observe(uint16_t name_id, const std::string& name, const ParsedName& parsed);

    const std::vector<TopoNode>& nodes() const { return nodes_; }
    const TopoNode& node(int index) const { return nodes_[static_cast<size_t>(index)]; }
    int size() const { return static_cast<int>(nodes_.size()); }

    // Depth-first list of row indices honouring the expanded flags. This is what
    // the TUI actually renders and navigates with j/k.
    std::vector<int> visible_rows() const;

    void toggle(int index);
    void set_expanded(int index, bool expanded);
    void expand_all();
    void collapse_all();

    // Expand the root, "layers", and the first block, so the tree looks alive on
    // first paint instead of showing a single collapsed line.
    void apply_default_expansion();

    // Every graph node id under this row, including descendants. Used to decide
    // which records belong to the current capture target.
    std::vector<uint16_t> collect_name_ids(int index) const;

    // Serialised for trace files so replay reproduces the same tree.
    struct Entry {
        uint16_t name_id;
        std::string name;
        uint8_t kind;
        int32_t layer;
    };
    std::vector<Entry> entries() const;

private:
    int ensure_child(int parent, const std::string& label, bool is_group, NodeKind kind,
                     int32_t layer);
    void flatten(int index, std::vector<int>& out) const;

    std::vector<TopoNode> nodes_;
    int root_ = 0;
    int embeddings_group_ = -1;
    int layers_group_ = -1;
    int output_group_ = -1;

    // (parent index, label) -> child index
    std::map<std::pair<int, std::string>, int> child_lookup_;
    std::unordered_map<uint16_t, int> name_to_node_;
    std::vector<Entry> entries_;
};

}  // namespace llmscope

#include "topology.hpp"

#include <algorithm>
#include <cctype>

namespace llmscope {

const char* to_string(NodeKind k) noexcept {
    switch (k) {
        case NodeKind::Embedding:  return "Embedding";
        case NodeKind::Norm:       return "LayerNorm";
        case NodeKind::Attention:  return "Attn";
        case NodeKind::AttnScores: return "Attn (Scores)";
        case NodeKind::Rope:       return "RoPE";
        case NodeKind::Cache:      return "KV Cache";
        case NodeKind::Ffn:        return "MLP";
        case NodeKind::LayerOut:   return "Residual";
        case NodeKind::Output:     return "Output";
        case NodeKind::Unknown:
        default:                   return "Node";
    }
}

const char* to_string(BackendId b) noexcept {
    switch (b) {
        case BackendId::CPU:    return "CPU";
        case BackendId::CUDA:   return "CUDA";
        case BackendId::Metal:  return "Metal";
        case BackendId::Vulkan: return "Vulkan";
        case BackendId::SYCL:   return "SYCL";
        case BackendId::BLAS:   return "BLAS";
        case BackendId::Other:  return "Other";
        case BackendId::Unknown:
        default:                return "?";
    }
}

const char* to_string(AnomalyKind k) noexcept {
    switch (k) {
        case AnomalyKind::NaN:              return "NaN";
        case AnomalyKind::Inf:              return "Inf";
        case AnomalyKind::OutlierMagnitude: return "Outlier";
        case AnomalyKind::HighSparsity:     return "Dead";
        case AnomalyKind::LatencySpike:     return "Slow";
        default:                            return "?";
    }
}

namespace {

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

// Pulls the layer index out of "attn_norm-3" or "cache_k_l12".
int32_t extract_layer(const std::string& name) {
    // Trailing "-N" is llama.cpp's per-layer suffix.
    const size_t dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size()) {
        bool all_digits = true;
        for (size_t i = dash + 1; i < name.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) {
            return static_cast<int32_t>(std::stol(name.substr(dash + 1)));
        }
    }

    // KV cache views use "_lN" instead.
    const size_t l_pos = name.rfind("_l");
    if (l_pos != std::string::npos && l_pos + 2 < name.size()) {
        bool all_digits = true;
        for (size_t i = l_pos + 2; i < name.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) {
            return static_cast<int32_t>(std::stol(name.substr(l_pos + 2)));
        }
    }

    return -1;
}

}  // namespace

ParsedName parse_node_name(const std::string& name) {
    ParsedName out;
    out.layer = extract_layer(name);

    // Order matters. "attn_norm" and "ffn_norm" contain "attn"/"ffn" but are
    // normalisation nodes, so the norm test must come before those.
    if (contains(name, "kq_soft_max") || contains(name, "kq_softmax")) {
        out.kind = NodeKind::AttnScores;
    } else if (contains(name, "cache_")) {
        out.kind = NodeKind::Cache;
    } else if (contains(name, "norm")) {
        out.kind = NodeKind::Norm;
    } else if (contains(name, "ffn")) {
        out.kind = NodeKind::Ffn;
    } else if (contains(name, "rope")) {
        out.kind = NodeKind::Rope;
    } else if (contains(name, "attn") || contains(name, "kqv") || contains(name, "kq") ||
               name.rfind("Qcur", 0) == 0 || name.rfind("Kcur", 0) == 0 ||
               name.rfind("Vcur", 0) == 0 || name.rfind("q_", 0) == 0 ||
               name.rfind("k_", 0) == 0 || name.rfind("v_", 0) == 0) {
        out.kind = NodeKind::Attention;
    } else if (contains(name, "embd") || contains(name, "embed") || contains(name, "tok_emb")) {
        out.kind = NodeKind::Embedding;
    } else if (contains(name, "l_out")) {
        out.kind = NodeKind::LayerOut;
    } else if (contains(name, "result") || contains(name, "logits") || contains(name, "output")) {
        out.kind = NodeKind::Output;
    } else {
        out.kind = NodeKind::Unknown;
    }

    return out;
}

Topology::Topology() {
    TopoNode root;
    root.label = "model";
    root.is_group = true;
    root.depth = 0;
    root.expanded = true;
    nodes_.push_back(root);
    root_ = 0;
}

int Topology::ensure_child(int parent, const std::string& label, bool is_group, NodeKind kind,
                           int32_t layer) {
    const auto key = std::make_pair(parent, label);
    auto it = child_lookup_.find(key);
    if (it != child_lookup_.end()) {
        return it->second;
    }

    TopoNode n;
    n.label = label;
    n.is_group = is_group;
    n.kind = kind;
    n.layer = layer;
    n.parent = parent;
    n.depth = nodes_[static_cast<size_t>(parent)].depth + 1;
    n.expanded = false;

    const int index = static_cast<int>(nodes_.size());
    nodes_.push_back(n);
    nodes_[static_cast<size_t>(parent)].children.push_back(index);
    child_lookup_.emplace(key, index);
    return index;
}

void Topology::observe(uint16_t name_id, const std::string& name, const ParsedName& parsed) {
    if (name_to_node_.count(name_id) != 0) {
        return;
    }

    int parent = root_;

    if (parsed.layer >= 0) {
        if (layers_group_ < 0) {
            layers_group_ = ensure_child(root_, "layers", true, NodeKind::Unknown, -1);
        }
        // Zero-pad so "layers.2" sorts before "layers.10" in the child_lookup_ map
        // and, more importantly, so the rendered column stays aligned.
        std::string layer_label = "layers." + std::to_string(parsed.layer);
        const int layer_node = ensure_child(layers_group_, layer_label, true, NodeKind::Unknown,
                                            parsed.layer);

        const char* sub = nullptr;
        switch (parsed.kind) {
            case NodeKind::Attention:
            case NodeKind::AttnScores:
            case NodeKind::Rope:
            case NodeKind::Cache:
                sub = "attn";
                break;
            case NodeKind::Ffn:
                sub = "mlp";
                break;
            case NodeKind::Norm:
                sub = "norm";
                break;
            default:
                sub = nullptr;  // l_out and friends hang directly off the block
                break;
        }
        parent = sub ? ensure_child(layer_node, sub, true, parsed.kind, parsed.layer) : layer_node;
    } else if (parsed.kind == NodeKind::Embedding) {
        if (embeddings_group_ < 0) {
            embeddings_group_ = ensure_child(root_, "embeddings", true, NodeKind::Embedding, -1);
        }
        parent = embeddings_group_;
    } else {
        if (output_group_ < 0) {
            output_group_ = ensure_child(root_, "output", true, NodeKind::Output, -1);
        }
        parent = output_group_;
    }

    const int leaf = ensure_child(parent, name, false, parsed.kind, parsed.layer);
    nodes_[static_cast<size_t>(leaf)].name_ids.push_back(name_id);
    name_to_node_.emplace(name_id, leaf);

    Entry e;
    e.name_id = name_id;
    e.name = name;
    e.kind = static_cast<uint8_t>(parsed.kind);
    e.layer = parsed.layer;
    entries_.push_back(e);
}

void Topology::flatten(int index, std::vector<int>& out) const {
    out.push_back(index);
    const TopoNode& n = nodes_[static_cast<size_t>(index)];
    if (!n.expanded) {
        return;
    }
    for (int child : n.children) {
        flatten(child, out);
    }
}

std::vector<int> Topology::visible_rows() const {
    std::vector<int> out;
    out.reserve(nodes_.size());
    flatten(root_, out);
    return out;
}

void Topology::toggle(int index) {
    if (index < 0 || index >= size()) {
        return;
    }
    TopoNode& n = nodes_[static_cast<size_t>(index)];
    n.expanded = !n.expanded;
}

void Topology::set_expanded(int index, bool expanded) {
    if (index < 0 || index >= size()) {
        return;
    }
    nodes_[static_cast<size_t>(index)].expanded = expanded;
}

void Topology::expand_all() {
    for (auto& n : nodes_) {
        n.expanded = true;
    }
}

void Topology::collapse_all() {
    // Including the root: "collapse all" should leave exactly one visible row.
    for (auto& n : nodes_) {
        n.expanded = false;
    }
}

void Topology::apply_default_expansion() {
    nodes_[static_cast<size_t>(root_)].expanded = true;
    if (embeddings_group_ >= 0) {
        nodes_[static_cast<size_t>(embeddings_group_)].expanded = false;
    }
    if (layers_group_ >= 0) {
        TopoNode& layers = nodes_[static_cast<size_t>(layers_group_)];
        layers.expanded = true;
        if (!layers.children.empty()) {
            const int first_block = layers.children.front();
            nodes_[static_cast<size_t>(first_block)].expanded = true;
        }
    }
}

std::vector<uint16_t> Topology::collect_name_ids(int index) const {
    std::vector<uint16_t> out;
    if (index < 0 || index >= size()) {
        return out;
    }
    std::vector<int> stack{index};
    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        const TopoNode& n = nodes_[static_cast<size_t>(cur)];
        out.insert(out.end(), n.name_ids.begin(), n.name_ids.end());
        for (int c : n.children) {
            stack.push_back(c);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<Topology::Entry> Topology::entries() const {
    return entries_;
}

}  // namespace llmscope

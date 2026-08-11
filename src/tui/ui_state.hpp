// ui_state.hpp - everything the panes read, owned entirely by the TUI thread.
//
// UiState is the only place that talks to a TraceSource. It drains records once
// per frame and folds them into the shapes the panes actually need: a recent
// stream, per-layer latency totals, the latest record for each node, and the
// current attention matrix. Panes are then pure functions of this state.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "trace_core/record.hpp"
#include "trace_core/topology.hpp"
#include "trace_core/trace_source.hpp"

namespace llmscope {

struct AnomalyEntry {
    AnomalyRecord rec;
    std::string node_name;
};

// Rolling latency totals for one transformer block, used to answer the
// project's headline question: which block dominates compute?
struct LayerAggregate {
    uint64_t total_ns = 0;
    uint64_t count = 0;
    uint64_t attn_ns = 0;
    uint64_t ffn_ns = 0;
    uint64_t norm_ns = 0;
    uint64_t other_ns = 0;
};

class UiState {
public:
    explicit UiState(TraceSource& source);

    // Drain the source. Called once per frame from the TUI thread.
    void pump();

    TraceSource& source() { return source_; }

    const Topology& topo() const { return topo_; }
    Topology& topo() { return topo_; }

    std::string name_of(uint16_t id) const;

    const std::deque<NodeRecord>& stream() const { return stream_; }
    const std::vector<AnomalyEntry>& anomalies() const { return anomalies_; }
    const std::vector<TokenInfo>& tokens() const { return tokens_; }
    const std::vector<LayerAggregate>& per_layer() const { return per_layer_; }
    const ModelInfo& model() const { return model_; }
    const SourceStats& stats() const { return stats_; }

    // Most recent record for a given graph node, if one has arrived.
    const NodeRecord* last_record(uint16_t name_id) const;

    // Aggregate over every node under a topology row (a whole block, say).
    LayerAggregate aggregate_for_rows(const std::vector<uint16_t>& name_ids) const;

    // Current attention plane: row-major [rows x cols], rows = query positions.
    const std::vector<float>& attention() const { return attention_; }
    int attention_rows() const { return attention_rows_; }
    int attention_cols() const { return attention_cols_; }
    uint16_t attention_name_id() const { return attention_name_id_; }
    bool has_attention() const { return attention_rows_ > 0 && attention_cols_ > 0; }

    // Token index that row 0 of the accumulated matrix corresponds to.
    uint32_t attention_first_token() const { return attention_first_token_; }

    // Rows accumulated across decode steps are capped so a long generation
    // cannot grow the heatmap without bound.
    static constexpr int kMaxAttentionRows = 128;

    uint64_t heaviest_layer_ns() const { return heaviest_layer_ns_; }
    int heaviest_layer() const { return heaviest_layer_; }

    static constexpr size_t kStreamCapacity = 512;
    static constexpr size_t kAnomalyCapacity = 256;

private:
    void ingest_names();
    void ingest_records();
    void ingest_anomalies();
    void ingest_tokens();
    void ingest_payload();

    TraceSource& source_;
    Topology topo_;
    std::vector<std::string> names_;
    bool default_expansion_applied_ = false;

    std::deque<NodeRecord> stream_;
    std::vector<AnomalyEntry> anomalies_;
    std::vector<TokenInfo> tokens_;
    std::vector<LayerAggregate> per_layer_;
    std::unordered_map<uint16_t, NodeRecord> last_by_name_;

    std::vector<float> attention_;
    int attention_rows_ = 0;
    int attention_cols_ = 0;
    uint16_t attention_name_id_ = kInvalidNameId;
    uint64_t attention_version_ = 0;
    int32_t  attention_head_ = 0;
    uint32_t attention_first_token_ = 0;
    uint32_t attention_last_token_ = 0;
    bool     attention_seeded_ = false;

    ModelInfo model_;
    SourceStats stats_;

    uint64_t heaviest_layer_ns_ = 0;
    int heaviest_layer_ = -1;

    // Scratch buffers reused every frame so pump() does not churn the heap.
    std::vector<Topology::Entry> name_scratch_;
    std::vector<NodeRecord> record_scratch_;
    std::vector<AnomalyRecord> anomaly_scratch_;
    std::vector<TokenInfo> token_scratch_;
    std::vector<float> payload_scratch_;
};

}  // namespace llmscope

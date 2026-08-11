// fake_source.hpp - a synthetic model, so the TUI is developable and testable
// with no GGUF file, no llama.cpp, and no inference at all.
//
// It emits a structurally realistic graph (llama-style node names, causal
// attention matrices, plausible latency and sparsity distributions) at a
// controllable rate, including deliberate anomalies so the ledger pane has
// something to show. `llmscope --demo` runs against this.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "name_table.hpp"  // kInvalidNameId
#include "trace_source.hpp"

namespace llmscope {

class FakeSource : public TraceSource {
public:
    struct Options {
        int32_t n_layer = 16;
        int32_t n_head = 32;
        int32_t n_embd = 2048;
        int32_t n_ctx = 4096;
        double  tokens_per_sec = 6.0;
        uint32_t max_tokens = 512;
        uint32_t seed = 1234;
    };

    // Two constructors rather than a defaulted argument: Options' default member
    // initializers are not usable inside the enclosing class definition.
    FakeSource();
    explicit FakeSource(const Options& opts);

    ModelInfo model_info() const override { return model_; }
    size_t poll_names(std::vector<Topology::Entry>& out) override;
    size_t poll_records(std::vector<NodeRecord>& out, size_t max_items) override;
    size_t poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) override;
    size_t poll_tokens(std::vector<TokenInfo>& out) override;

    uint64_t payload_version() const override { return payload_version_; }
    bool read_payload(PayloadHeader& header, std::vector<float>& data) const override;

    void set_capture_target(uint16_t name_id, int32_t attention_layer) override;
    void set_attention_head(int32_t head) override;
    int32_t attention_head() const override { return capture_head_; }

    bool running() const override;
    SourceStats stats() const override;
    std::string status_line() const override;
    void request_stop() override { stopped_ = true; }

private:
    // One node in the synthetic graph.
    struct Node {
        uint16_t name_id;
        std::string name;
        NodeKind kind;
        int32_t layer;
        int64_t ne[4];
        uint32_t base_ns;  // typical latency for this node type
    };

    void build_graph(const Options& opts);
    void advance(uint64_t now);            // generate whatever the clock has earned
    void emit_token(uint64_t token_start_ns);
    void rebuild_attention_payload();

    ModelInfo model_;
    Options opts_;
    std::vector<Node> graph_;
    std::vector<Topology::Entry> names_;
    size_t names_cursor_ = 0;

    std::vector<NodeRecord> pending_;
    std::vector<AnomalyRecord> pending_anomalies_;
    std::vector<TokenInfo> pending_tokens_;

    PayloadHeader payload_header_{};
    std::vector<float> payload_data_;
    uint64_t payload_version_ = 0;

    uint16_t capture_name_id_ = kInvalidNameId;
    int32_t  capture_layer_ = 0;
    int32_t  capture_head_ = 0;

    uint64_t epoch_ns_ = 0;
    uint64_t next_token_ns_ = 0;
    uint64_t seq_ = 0;
    uint32_t token_index_ = 0;
    uint64_t records_total_ = 0;
    bool stopped_ = false;

    std::mt19937 rng_;
    std::vector<std::string> vocab_;
};

}  // namespace llmscope

// trace_source.hpp - the seam between "where records come from" and the TUI.
//
// Three implementations satisfy this interface:
//   LiveSource      - drains the ring fed by a running llama.cpp model
//   TraceFileSource - replays a .trace file recorded earlier
//   FakeSource      - synthesises a plausible model, so the TUI can be built and
//                     tested with no model, no GPU, and no llama.cpp at all
//
// Note what is deliberately absent: the TUI never shares a Topology or NameTable
// with the tracer thread. Instead, newly discovered names are polled as plain
// values and the TUI builds its *own* tree from them. That removes the whole
// class of data races that a shared mutable tree would introduce.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "record.hpp"
#include "topology.hpp"

namespace llmscope {

struct ModelInfo {
    std::string path;
    std::string name = "(no model)";
    std::string arch = "unknown";
    std::string quant = "unknown";
    int32_t n_layer = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_embd = 0;
    int32_t n_ctx = 0;
    int32_t n_vocab = 0;
    BackendId backend = BackendId::Unknown;
    bool flash_attn_disabled = true;  // required for attention scores to exist
};

struct TokenInfo {
    uint32_t index = 0;
    int32_t  id = 0;
    std::string piece;
    bool is_prompt = false;
};

struct SourceStats {
    uint64_t records_total = 0;
    uint64_t records_dropped = 0;
    uint64_t tokens_done = 0;
    double   tokens_per_sec = 0.0;
    uint64_t bytes_observed = 0;
    uint64_t payload_bytes = 0;
};

class TraceSource {
public:
    virtual ~TraceSource() = default;

    virtual ModelInfo model_info() const = 0;

    // Newly discovered node names since the last call. Non-empty essentially
    // only during the first forward pass.
    virtual size_t poll_names(std::vector<Topology::Entry>& out) = 0;

    virtual size_t poll_records(std::vector<NodeRecord>& out, size_t max_items) = 0;
    virtual size_t poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) = 0;
    virtual size_t poll_tokens(std::vector<TokenInfo>& out) = 0;

    // Bumps on every publish; lets the TUI avoid re-copying an unchanged tensor.
    virtual uint64_t payload_version() const = 0;
    virtual bool read_payload(PayloadHeader& header, std::vector<float>& data) const = 0;

    // Which graph node should get a full tensor snapshot, and which layer's
    // attention scores to mirror into the heatmap. Driven by the TUI selection.
    //
    // Only one tensor is ever in flight, so this is two scalars rather than a
    // set - which means the tracer can read it with two relaxed atomic loads
    // instead of taking a lock 300 times per token on the inference hot path.
    // Pass kInvalidNameId / -1 to capture nothing.
    virtual void set_capture_target(uint16_t name_id, int32_t attention_layer) = 0;

    // Which attention head's plane to mirror into the heatmap.
    virtual void set_attention_head(int32_t head) { (void)head; }
    virtual int32_t attention_head() const { return 0; }

    virtual bool running() const = 0;
    virtual SourceStats stats() const = 0;
    virtual std::string status_line() const = 0;
    virtual void request_stop() = 0;

    // Replay-only controls; live sources ignore them.
    virtual bool supports_seek() const { return false; }
    virtual void set_paused(bool) {}
    virtual bool paused() const { return false; }
    virtual void set_speed(double) {}
    virtual double speed() const { return 1.0; }
    virtual void restart() {}
    virtual double progress() const { return 0.0; }
};

}  // namespace llmscope

// tracer.hpp - the non-invasive hook into llama.cpp.
//
// llama.cpp is used entirely as-is: no patched source, no forked headers. The
// whole mechanism is one field on llama_context_params:
//
//     params.cb_eval           = &Tracer::cb_eval_thunk;
//     params.cb_eval_user_data = this;
//
// ggml's scheduler then invokes that callback twice per graph node - once to ask
// whether we want the node's data, once after computing it. Returning true on
// the ask means the scheduler computes that node as a graph view of exactly one
// node and synchronises the backend before handing it to us, which is what makes
// per-node latency measurable at all.
//
// That isolation is not free: always answering true defeats ggml's node
// batching, so traced inference runs slower than untraced. The latency figures
// are therefore *instrumented* latency. They are accurate relative to each other
// and correctly identify which block dominates, which is the point.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "trace_core/live_source.hpp"
#include "trace_core/record.hpp"
#include "trace_core/trace_file.hpp"

struct ggml_tensor;
struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_sampler;

namespace llmscope {

struct TracerConfig {
    std::string model_path;
    std::string prompt = "The capital of France is";
    int32_t n_predict = 96;
    int32_t n_ctx = 2048;
    int32_t n_threads = 0;      // 0 => hardware_concurrency
    int32_t n_gpu_layers = 0;   // CPU backend by default
    std::string trace_out;      // empty => do not record
    uint64_t sample_budget = 8192;
    // Which block's attention scores to snapshot when no TUI is driving the
    // selection. Without this a --headless recording would contain no attention
    // matrices at all, making the resulting trace far less useful to replay.
    int32_t capture_layer = 0;
    AnomalyConfig anomaly;
    bool quiet_llama = true;    // silence llama.cpp's own logging; it fights the TUI
};

class Tracer {
public:
    Tracer(LiveSource& sink, TracerConfig cfg);
    ~Tracer();

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;

    // Blocking. Runs load -> prompt -> generation on the calling thread.
    // Returns false and fills error_out if the model could not be run.
    bool run(std::string& error_out);

private:
    static bool cb_eval_thunk(ggml_tensor* t, bool ask, void* user_data);
    bool on_node(ggml_tensor* t, bool ask);

    bool load_model(std::string& error_out);
    void publish_model_info();
    bool maybe_capture_payload(const ggml_tensor* t, const void* data, const NodeRecord& rec,
                               NodeKind kind);
    void check_anomalies(const NodeRecord& rec);
    void unload();

    LiveSource& sink_;
    TracerConfig cfg_;

    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    llama_sampler* sampler_ = nullptr;

    TraceWriter writer_;

    // Hot-path scratch, reused across nodes so the callback never allocates
    // after warmup.
    std::vector<uint8_t> device_buf_;   // staging for non-host tensors
    std::vector<float> payload_buf_;

    uint64_t epoch_ns_ = 0;
    uint64_t node_start_ns_ = 0;
    uint64_t seq_ = 0;
    uint32_t token_index_ = 0;

    // Per-node exponential moving average of duration, for latency-spike
    // detection. An EMA rather than a true median: it is O(1) per node with no
    // allocation, which a median over a window would not be.
    std::unordered_map<uint16_t, float> latency_ema_;

    // Names already emitted to the trace file. Without this the file would
    // contain records referencing name ids that replay cannot resolve.
    std::unordered_set<uint16_t> written_names_;

    // Last token index at which each (node, rule) pair fired. Repeating the same
    // finding 300 times a second turns the ledger into noise, so each pair
    // reports at most once per token.
    std::unordered_map<uint32_t, uint32_t> anomaly_last_token_;
};

}  // namespace llmscope

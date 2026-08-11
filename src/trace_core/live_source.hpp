// live_source.hpp - the shared object between the inference thread and the TUI.
//
// Producer methods are called only from the tracer thread; TraceSource methods
// only from the TUI thread. The split is deliberate and load-bearing:
//
//   hot path  (300x per token)  -> SpscRing::push, two relaxed atomic loads
//   warm path (1x per token)    -> payload publish, token push
//   cold path (once at startup) -> name interning, model info
//
// Only the cold and warm paths take a lock. Nothing on the hot path does.
#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "name_table.hpp"
#include "payload_slot.hpp"
#include "record.hpp"
#include "spsc_ring.hpp"
#include "trace_source.hpp"

namespace llmscope {

class LiveSource : public TraceSource {
public:
    explicit LiveSource(size_t ring_capacity = 1u << 16);

    // ----- producer side: tracer thread only -----

    void set_model_info(const ModelInfo& info);

    // Interns a name and queues it for the TUI's topology. Returns the id.
    // Cheap after the first forward pass, when every name is already known.
    uint16_t observe_name(const std::string& name, NodeKind kind, int32_t layer);

    bool push_record(const NodeRecord& rec);
    void push_anomaly(const AnomalyRecord& rec);
    void push_token(const TokenInfo& tok);
    void publish_payload(const PayloadHeader& header, const float* data, uint32_t n_floats);

    // Hot-path reads of the TUI's current selection. Relaxed loads: a capture
    // target that lands one node late is harmless.
    uint16_t capture_name_id() const noexcept {
        return capture_name_id_.load(std::memory_order_relaxed);
    }
    int32_t capture_layer() const noexcept {
        return capture_layer_.load(std::memory_order_relaxed);
    }
    int32_t capture_head() const noexcept {
        return capture_head_.load(std::memory_order_relaxed);
    }

    void set_running(bool r) noexcept { running_.store(r, std::memory_order_release); }
    void set_status(const std::string& s);
    void add_observed_bytes(uint64_t n) noexcept {
        bytes_observed_.fetch_add(n, std::memory_order_relaxed);
    }
    bool stop_requested() const noexcept { return stop_requested_.load(std::memory_order_acquire); }

    // Resolve an id back to its name; used by the tracer for trace-file writing.
    std::string name_of(uint16_t id) const { return names_.name(id); }

    // ----- consumer side: TUI thread only (TraceSource) -----

    ModelInfo model_info() const override;
    size_t poll_names(std::vector<Topology::Entry>& out) override;
    size_t poll_records(std::vector<NodeRecord>& out, size_t max_items) override;
    size_t poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) override;
    size_t poll_tokens(std::vector<TokenInfo>& out) override;

    uint64_t payload_version() const override { return payload_.version(); }
    bool read_payload(PayloadHeader& header, std::vector<float>& data) const override {
        return payload_.try_read(header, data);
    }

    void set_capture_target(uint16_t name_id, int32_t attention_layer) override {
        capture_name_id_.store(name_id, std::memory_order_relaxed);
        capture_layer_.store(attention_layer, std::memory_order_relaxed);
    }
    void set_attention_head(int32_t head) override {
        capture_head_.store(head < 0 ? 0 : head, std::memory_order_relaxed);
    }
    int32_t attention_head() const override {
        return capture_head_.load(std::memory_order_relaxed);
    }

    bool running() const override { return running_.load(std::memory_order_acquire); }
    SourceStats stats() const override;
    std::string status_line() const override;
    void request_stop() override { stop_requested_.store(true, std::memory_order_release); }

private:
    SpscRing<NodeRecord> ring_;
    PayloadSlot payload_;
    NameTable names_;

    mutable std::mutex meta_mutex_;  // guards model_, pending_names_, tokens_, anomalies_, status_
    ModelInfo model_;
    std::vector<Topology::Entry> pending_names_;
    std::vector<TokenInfo> pending_tokens_;
    std::vector<AnomalyRecord> pending_anomalies_;
    std::string status_ = "starting";

    std::atomic<uint16_t> capture_name_id_{kInvalidNameId};
    // Defaults to block 0 rather than "nothing", so attention is captured even
    // when no TUI has selected anything yet.
    std::atomic<int32_t>  capture_layer_{0};
    std::atomic<int32_t>  capture_head_{0};
    std::atomic<bool>     running_{false};
    std::atomic<bool>     stop_requested_{false};
    std::atomic<uint64_t> bytes_observed_{0};
    std::atomic<uint64_t> payload_bytes_{0};
    std::atomic<uint64_t> tokens_done_{0};
    std::atomic<uint64_t> first_token_ns_{0};
    std::atomic<uint64_t> last_token_ns_{0};
};

}  // namespace llmscope

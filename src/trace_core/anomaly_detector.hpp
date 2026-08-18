// anomaly_detector.hpp - the anomaly rules, as a reusable stateful object.
//
// These rules started life inline in Tracer, but they are not tracer-specific:
// replaying a trace file and the synthetic FakeSource want the same verdicts,
// and rules embedded in a class that needs llama.cpp to construct cannot be
// tested at all. They live in trace_core so they stay dependency-free.
//
// The detector is stateful on purpose. Two of the rules are only meaningful
// across time:
//
//   - Deduplication. A stuck node trips the same rule on every one of the ~300
//     nodes per second, and a ledger repeating one finding 300 times a second
//     is a ledger nobody reads. Each (node, rule) pair reports at most once per
//     token, so a genuinely stuck node still appears every token while a merely
//     noisy one stops shouting.
//
//   - Latency. "Slow" only means something relative to a node's own history: a
//     matmul is legitimately slower than a layernorm, so comparing nodes to each
//     other flags the entire model. Each node is compared against an exponential
//     moving average of itself. An EMA rather than a true median because it is
//     O(1) with no allocation, which a windowed median is not.
//
// Not thread-safe: it is owned by whoever produces records and evaluated on
// that producer's thread.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "record.hpp"

namespace llmscope {

class AnomalyDetector {
public:
    AnomalyDetector() = default;
    explicit AnomalyDetector(const AnomalyConfig& cfg) : cfg_(cfg) {}

    const AnomalyConfig& config() const noexcept { return cfg_; }

    // Applies every rule to one record, appending any findings to `out`.
    // Appends rather than assigns so a caller can accumulate across a batch.
    void evaluate(const NodeRecord& rec, std::vector<AnomalyRecord>& out) {
        if (!cfg_.enabled) {
            return;
        }

        // NaN and Inf first: they are unambiguous faults, and when a node trips
        // several rules at once the ledger should lead with the real one.
        if (rec.flags & kFlagHasNaN) {
            emit(rec, AnomalyKind::NaN, 0.0f, 0.0f, Severity::Error, out);
        }
        if (rec.flags & kFlagHasInf) {
            emit(rec, AnomalyKind::Inf, 0.0f, 0.0f, Severity::Error, out);
        }

        if (rec.absmax > cfg_.magnitude_ceiling) {
            emit(rec, AnomalyKind::OutlierMagnitude, rec.absmax, cfg_.magnitude_ceiling,
                 Severity::Warn, out);
        }

        // A KV cache is allocated for the whole context and filled one position
        // per token, so early in a run it is ~99.6% zeros *by construction*.
        // That is the allocation working, not a dead layer, and on a real
        // 24-layer model it drowned the ledger: 144 of 147 anomalies on a
        // healthy 9-token run were cache_k/cache_v views. Excluding caches from
        // this rule only - they remain subject to every rule above, which flag
        // faults that are faults wherever they occur.
        if (static_cast<NodeKind>(rec.kind) != NodeKind::Cache &&
            rec.sparsity > cfg_.sparsity_ceiling &&
            element_count(rec) >= cfg_.sparsity_min_elements) {
            emit(rec, AnomalyKind::HighSparsity, rec.sparsity, cfg_.sparsity_ceiling,
                 Severity::Info, out);
        }

        evaluate_latency(rec, out);
    }

    // Forgets all history. Used when replay seeks, where carrying an EMA across
    // a discontinuity would manufacture a spike that never happened.
    void reset() {
        last_token_.clear();
        latency_ema_.clear();
    }

private:
    void evaluate_latency(const NodeRecord& rec, std::vector<AnomalyRecord>& out) {
        const float dur = static_cast<float>(rec.dur_ns);
        auto it = latency_ema_.find(rec.name_id);
        if (it == latency_ema_.end()) {
            // First sighting seeds the average. It cannot be a spike relative to
            // a baseline that does not exist yet.
            latency_ema_.emplace(rec.name_id, dur);
            return;
        }

        const float ema = it->second;
        if (rec.dur_ns > cfg_.latency_min_ns && ema > 0.0f &&
            dur > ema * cfg_.latency_factor) {
            emit(rec, AnomalyKind::LatencySpike, dur / 1e6f,
                 ema * cfg_.latency_factor / 1e6f, Severity::Warn, out);
        }
        it->second = ema * 0.9f + dur * 0.1f;
    }

    void emit(const NodeRecord& rec, AnomalyKind kind, float value, float threshold,
              Severity sev, std::vector<AnomalyRecord>& out) {
        const uint32_t key = (static_cast<uint32_t>(rec.name_id) << 8) |
                             static_cast<uint32_t>(kind);
        auto slot = last_token_.find(key);
        if (slot != last_token_.end() && slot->second == rec.token_index) {
            return;
        }
        last_token_[key] = rec.token_index;

        AnomalyRecord a{};
        a.seq = rec.seq;
        a.t_start_ns = rec.t_start_ns;
        a.value = value;
        a.threshold = threshold;
        a.layer = rec.layer;
        a.name_id = rec.name_id;
        a.kind = static_cast<uint8_t>(kind);
        a.severity = static_cast<uint8_t>(sev);
        out.push_back(a);
    }

    AnomalyConfig cfg_{};

    // Last token index at which each (node, rule) pair fired.
    std::unordered_map<uint32_t, uint32_t> last_token_;

    // Per-node exponential moving average of duration.
    std::unordered_map<uint16_t, float> latency_ema_;
};

}  // namespace llmscope

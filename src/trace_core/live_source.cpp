#include "live_source.hpp"

#include <chrono>

namespace llmscope {

namespace {
uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count());
}
}  // namespace

LiveSource::LiveSource(size_t ring_capacity) : ring_(ring_capacity) {}

void LiveSource::set_model_info(const ModelInfo& info) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    model_ = info;
}

uint16_t LiveSource::observe_name(const std::string& name, NodeKind kind, int32_t layer) {
    const size_t before = names_.size();
    const uint16_t id = names_.intern(name);
    if (names_.size() == before) {
        return id;  // already known - the common case after the first pass
    }

    Topology::Entry e;
    e.name_id = id;
    e.name = name;
    e.kind = static_cast<uint8_t>(kind);
    e.layer = layer;

    std::lock_guard<std::mutex> lock(meta_mutex_);
    pending_names_.push_back(std::move(e));
    return id;
}

bool LiveSource::push_record(const NodeRecord& rec) {
    return ring_.push(rec);
}

void LiveSource::push_anomaly(const AnomalyRecord& rec) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    // Bound the backlog: if the TUI is not draining, old anomalies are the ones
    // worth losing.
    if (pending_anomalies_.size() > 4096) {
        pending_anomalies_.erase(pending_anomalies_.begin(),
                                 pending_anomalies_.begin() + 2048);
    }
    pending_anomalies_.push_back(rec);
}

void LiveSource::push_token(const TokenInfo& tok) {
    const uint64_t t = now_ns();
    uint64_t expected = 0;
    first_token_ns_.compare_exchange_strong(expected, t, std::memory_order_relaxed);
    last_token_ns_.store(t, std::memory_order_relaxed);
    tokens_done_.fetch_add(1, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(meta_mutex_);
    pending_tokens_.push_back(tok);
}

void LiveSource::publish_payload(const PayloadHeader& header, const float* data,
                                 uint32_t n_floats) {
    payload_.publish(header, data, n_floats);
    payload_bytes_.fetch_add(static_cast<uint64_t>(n_floats) * sizeof(float),
                             std::memory_order_relaxed);
}

void LiveSource::set_status(const std::string& s) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    status_ = s;
}

ModelInfo LiveSource::model_info() const {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    return model_;
}

size_t LiveSource::poll_names(std::vector<Topology::Entry>& out) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    const size_t n = pending_names_.size();
    for (auto& e : pending_names_) {
        out.push_back(std::move(e));
    }
    pending_names_.clear();
    return n;
}

size_t LiveSource::poll_records(std::vector<NodeRecord>& out, size_t max_items) {
    return ring_.pop_batch(out, max_items);
}

size_t LiveSource::poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    const size_t n = pending_anomalies_.size() < max_items ? pending_anomalies_.size() : max_items;
    out.insert(out.end(), pending_anomalies_.begin(), pending_anomalies_.begin() + n);
    pending_anomalies_.erase(pending_anomalies_.begin(), pending_anomalies_.begin() + n);
    return n;
}

size_t LiveSource::poll_tokens(std::vector<TokenInfo>& out) {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    const size_t n = pending_tokens_.size();
    for (auto& t : pending_tokens_) {
        out.push_back(std::move(t));
    }
    pending_tokens_.clear();
    return n;
}

SourceStats LiveSource::stats() const {
    SourceStats s;
    s.records_total = ring_.total_pushed();
    s.records_dropped = ring_.dropped();
    s.tokens_done = tokens_done_.load(std::memory_order_relaxed);
    s.bytes_observed = bytes_observed_.load(std::memory_order_relaxed);
    s.payload_bytes = payload_bytes_.load(std::memory_order_relaxed);

    const uint64_t first = first_token_ns_.load(std::memory_order_relaxed);
    const uint64_t last = last_token_ns_.load(std::memory_order_relaxed);
    if (first > 0 && last > first && s.tokens_done > 1) {
        const double secs = static_cast<double>(last - first) / 1e9;
        s.tokens_per_sec = static_cast<double>(s.tokens_done - 1) / secs;
    }
    return s;
}

std::string LiveSource::status_line() const {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    return status_;
}

}  // namespace llmscope

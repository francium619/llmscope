#include "ui_state.hpp"

#include <algorithm>

namespace llmscope {

UiState::UiState(TraceSource& source) : source_(source) {
    model_ = source_.model_info();
}

void UiState::pump() {
    ingest_names();
    ingest_records();
    ingest_anomalies();
    ingest_tokens();
    ingest_payload();

    model_ = source_.model_info();
    stats_ = source_.stats();
}

void UiState::ingest_names() {
    name_scratch_.clear();
    if (source_.poll_names(name_scratch_) == 0) {
        return;
    }
    for (auto& e : name_scratch_) {
        if (e.name_id >= names_.size()) {
            names_.resize(static_cast<size_t>(e.name_id) + 1);
        }
        names_[e.name_id] = e.name;

        ParsedName p;
        p.kind = static_cast<NodeKind>(e.kind);
        p.layer = e.layer;
        topo_.observe(e.name_id, e.name, p);
    }

    // Open the tree once, after the first batch of names lands. Doing it here
    // rather than at construction means the default view is meaningful instead
    // of an empty root.
    if (!default_expansion_applied_ && topo_.size() > 1) {
        topo_.apply_default_expansion();
        default_expansion_applied_ = true;
    }
}

void UiState::ingest_records() {
    record_scratch_.clear();
    source_.poll_records(record_scratch_, 8192);
    if (record_scratch_.empty()) {
        return;
    }

    for (const NodeRecord& r : record_scratch_) {
        stream_.push_back(r);
        last_by_name_[r.name_id] = r;

        if (r.layer >= 0) {
            if (static_cast<size_t>(r.layer) >= per_layer_.size()) {
                per_layer_.resize(static_cast<size_t>(r.layer) + 1);
            }
            LayerAggregate& agg = per_layer_[static_cast<size_t>(r.layer)];
            agg.total_ns += r.dur_ns;
            ++agg.count;
            switch (static_cast<NodeKind>(r.kind)) {
                case NodeKind::Attention:
                case NodeKind::AttnScores:
                case NodeKind::Rope:
                case NodeKind::Cache:
                    agg.attn_ns += r.dur_ns;
                    break;
                case NodeKind::Ffn:
                    agg.ffn_ns += r.dur_ns;
                    break;
                case NodeKind::Norm:
                    agg.norm_ns += r.dur_ns;
                    break;
                default:
                    agg.other_ns += r.dur_ns;
                    break;
            }
        }
    }

    while (stream_.size() > kStreamCapacity) {
        stream_.pop_front();
    }

    heaviest_layer_ns_ = 0;
    heaviest_layer_ = -1;
    for (size_t i = 0; i < per_layer_.size(); ++i) {
        if (per_layer_[i].total_ns > heaviest_layer_ns_) {
            heaviest_layer_ns_ = per_layer_[i].total_ns;
            heaviest_layer_ = static_cast<int>(i);
        }
    }
}

void UiState::ingest_anomalies() {
    anomaly_scratch_.clear();
    source_.poll_anomalies(anomaly_scratch_, 128);
    for (const AnomalyRecord& a : anomaly_scratch_) {
        AnomalyEntry e;
        e.rec = a;
        e.node_name = name_of(a.name_id);
        anomalies_.push_back(std::move(e));
    }
    if (anomalies_.size() > kAnomalyCapacity) {
        anomalies_.erase(anomalies_.begin(),
                         anomalies_.begin() +
                             static_cast<long>(anomalies_.size() - kAnomalyCapacity));
    }
}

void UiState::ingest_tokens() {
    token_scratch_.clear();
    source_.poll_tokens(token_scratch_);
    for (auto& t : token_scratch_) {
        tokens_.push_back(std::move(t));
    }
    // The heatmap only ever shows a window of recent tokens; keeping thousands
    // of pieces alive would be pure waste.
    if (tokens_.size() > 4096) {
        tokens_.erase(tokens_.begin(), tokens_.begin() + 2048);
    }
}

void UiState::ingest_payload() {
    const uint64_t v = source_.payload_version();
    if (v == attention_version_) {
        return;
    }
    PayloadHeader header{};
    payload_scratch_.clear();
    if (!source_.read_payload(header, payload_scratch_)) {
        return;
    }
    attention_version_ = v;

    if (static_cast<NodeKind>(header.kind) != NodeKind::AttnScores) {
        return;  // some other tensor was captured; the heatmap keeps its last frame
    }

    // ne[0] is the key/context axis (columns), ne[1] the query axis (rows).
    const int cols = static_cast<int>(header.ne[0] > 0 ? header.ne[0] : 0);
    int rows = static_cast<int>(header.ne[1] > 0 ? header.ne[1] : 0);
    if (cols <= 0 || rows <= 0) {
        return;
    }
    // A truncated payload may not contain every row.
    const int available_rows = static_cast<int>(payload_scratch_.size()) / cols;
    rows = std::min(rows, available_rows);
    if (rows <= 0) {
        return;
    }

    const int32_t head = source_.attention_head();

    // Switching layer or head invalidates everything accumulated so far.
    const bool subject_changed =
        !attention_seeded_ || header.name_id != attention_name_id_ || head != attention_head_;

    // The prompt pass yields a real [n_tokens x n_kv] matrix; each decode step
    // afterwards yields a single row. Appending those rows reconstructs the full
    // causal picture instead of letting one row overwrite the matrix.
    const bool append = !subject_changed && rows == 1 && attention_rows_ > 0 &&
                        header.token_index > attention_last_token_;

    if (append) {
        const int new_cols = std::max(attention_cols_, cols);

        if (new_cols != attention_cols_) {
            // The KV axis grew: widen existing rows, zero-filling the tail.
            std::vector<float> widened(static_cast<size_t>(attention_rows_) *
                                           static_cast<size_t>(new_cols),
                                       0.0f);
            for (int r = 0; r < attention_rows_; ++r) {
                std::copy_n(attention_.begin() + static_cast<size_t>(r) * attention_cols_,
                            attention_cols_,
                            widened.begin() + static_cast<size_t>(r) * new_cols);
            }
            attention_ = std::move(widened);
            attention_cols_ = new_cols;
        }

        attention_.resize(static_cast<size_t>(attention_rows_ + 1) *
                              static_cast<size_t>(attention_cols_),
                          0.0f);
        std::copy_n(payload_scratch_.begin(), std::min(cols, attention_cols_),
                    attention_.begin() + static_cast<size_t>(attention_rows_) * attention_cols_);
        ++attention_rows_;

        if (attention_rows_ > kMaxAttentionRows) {
            // Drop the oldest row and advance the label origin with it.
            attention_.erase(attention_.begin(),
                             attention_.begin() + attention_cols_);
            --attention_rows_;
            ++attention_first_token_;
        }
    } else {
        attention_ = payload_scratch_;
        attention_.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        attention_rows_ = rows;
        attention_cols_ = cols;
        // A multi-row payload is the prompt pass, whose first row is token 0.
        attention_first_token_ =
            rows > 1 ? 0 : header.token_index;
    }

    attention_name_id_ = header.name_id;
    attention_head_ = head;
    attention_last_token_ = header.token_index;
    attention_seeded_ = true;
}

std::string UiState::name_of(uint16_t id) const {
    if (id < names_.size() && !names_[id].empty()) {
        return names_[id];
    }
    return "<unknown>";
}

const NodeRecord* UiState::last_record(uint16_t name_id) const {
    auto it = last_by_name_.find(name_id);
    return it == last_by_name_.end() ? nullptr : &it->second;
}

LayerAggregate UiState::aggregate_for_rows(const std::vector<uint16_t>& name_ids) const {
    LayerAggregate agg;
    for (uint16_t id : name_ids) {
        auto it = last_by_name_.find(id);
        if (it == last_by_name_.end()) {
            continue;
        }
        const NodeRecord& r = it->second;
        agg.total_ns += r.dur_ns;
        ++agg.count;
        switch (static_cast<NodeKind>(r.kind)) {
            case NodeKind::Attention:
            case NodeKind::AttnScores:
            case NodeKind::Rope:
            case NodeKind::Cache:
                agg.attn_ns += r.dur_ns;
                break;
            case NodeKind::Ffn:
                agg.ffn_ns += r.dur_ns;
                break;
            case NodeKind::Norm:
                agg.norm_ns += r.dur_ns;
                break;
            default:
                agg.other_ns += r.dur_ns;
                break;
        }
    }
    return agg;
}

}  // namespace llmscope

#include "fake_source.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace llmscope {

namespace {

uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count());
}

}  // namespace

FakeSource::FakeSource() : FakeSource(Options()) {}

FakeSource::FakeSource(const Options& opts) : opts_(opts), rng_(opts.seed) {
    model_.path = "(synthetic)";
    model_.name = "demo-1b-synthetic";
    model_.arch = "llama";
    model_.quant = "Q4_K_M";
    model_.n_layer = opts.n_layer;
    model_.n_head = opts.n_head;
    model_.n_head_kv = opts.n_head / 4;
    model_.n_embd = opts.n_embd;
    model_.n_ctx = opts.n_ctx;
    model_.n_vocab = 32000;
    model_.backend = BackendId::CPU;
    model_.flash_attn_disabled = true;

    vocab_ = {"The",  " quick", " brown", " fox",  " jumps", " over", " the",  " lazy",
              " dog", " and",   " then",  " runs", " far",   " away", " into", " night"};

    build_graph(opts);
    epoch_ns_ = now_ns();
    next_token_ns_ = epoch_ns_;
    capture_layer_ = 0;
    rebuild_attention_payload();
}

void FakeSource::build_graph(const Options& opts) {
    uint16_t next_id = 0;
    const int64_t n_embd = opts.n_embd;

    auto add = [&](const std::string& name, NodeKind kind, int32_t layer, int64_t d0, int64_t d1,
                   int64_t d2, uint32_t base_ns) {
        Node n;
        n.name_id = next_id++;
        n.name = name;
        n.kind = kind;
        n.layer = layer;
        n.ne[0] = d0;
        n.ne[1] = d1;
        n.ne[2] = d2;
        n.ne[3] = 1;
        n.base_ns = base_ns;
        graph_.push_back(n);

        Topology::Entry e;
        e.name_id = n.name_id;
        e.name = name;
        e.kind = static_cast<uint8_t>(kind);
        e.layer = layer;
        names_.push_back(e);
    };

    add("inp_embd", NodeKind::Embedding, -1, n_embd, 1, 1, 22'000);

    const int64_t head_dim = n_embd / opts.n_head;
    for (int32_t l = 0; l < opts.n_layer; ++l) {
        const std::string s = "-" + std::to_string(l);
        add("attn_norm" + s, NodeKind::Norm, l, n_embd, 1, 1, 9'000);
        add("Qcur" + s, NodeKind::Attention, l, head_dim, opts.n_head, 1, 120'000);
        add("Kcur" + s, NodeKind::Attention, l, head_dim, model_.n_head_kv, 1, 40'000);
        add("Vcur" + s, NodeKind::Attention, l, head_dim, model_.n_head_kv, 1, 40'000);
        add("cache_k_l" + std::to_string(l), NodeKind::Cache, l, head_dim, 1, 1, 6'000);
        add("kq_soft_max_ext" + s, NodeKind::AttnScores, l, 1, 1, opts.n_head, 55'000);
        add("kqv_out" + s, NodeKind::Attention, l, n_embd, 1, 1, 130'000);
        add("ffn_norm" + s, NodeKind::Norm, l, n_embd, 1, 1, 9'000);
        add("ffn_up" + s, NodeKind::Ffn, l, n_embd * 3, 1, 1, 310'000);
        add("ffn_gate" + s, NodeKind::Ffn, l, n_embd * 3, 1, 1, 305'000);
        add("ffn_down" + s, NodeKind::Ffn, l, n_embd, 1, 1, 300'000);
        add("l_out" + s, NodeKind::LayerOut, l, n_embd, 1, 1, 7'000);
    }

    add("result_norm", NodeKind::Norm, -1, n_embd, 1, 1, 9'000);
    add("result_output", NodeKind::Output, -1, model_.n_vocab, 1, 1, 480'000);
}

void FakeSource::set_capture_target(uint16_t name_id, int32_t attention_layer) {
    capture_name_id_ = name_id;
    if (attention_layer != capture_layer_ && attention_layer >= 0) {
        capture_layer_ = attention_layer;
        rebuild_attention_payload();
    }
}

void FakeSource::set_attention_head(int32_t head) {
    if (head < 0) head = 0;
    if (head >= opts_.n_head) head = opts_.n_head - 1;
    if (head != capture_head_) {
        capture_head_ = head;
        // Reseed so each head shows a visibly different pattern, the way real
        // heads specialise.
        rng_.seed(opts_.seed + static_cast<uint32_t>(head) * 7919u);
        rebuild_attention_payload();
    }
}

// Builds a causal attention matrix that looks like a real one: strong diagonal,
// an attention sink on the first token, and mild locality decay.
void FakeSource::rebuild_attention_payload() {
    const uint32_t n = std::min<uint32_t>(token_index_ + 1, 64);
    const uint32_t n_tok = n < 4 ? 4 : n;

    payload_data_.assign(static_cast<size_t>(n_tok) * n_tok, 0.0f);
    std::uniform_real_distribution<float> jitter(0.0f, 0.35f);

    for (uint32_t i = 0; i < n_tok; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j <= i; ++j) {
            const float dist = static_cast<float>(i - j);
            float w = std::exp(-dist * 0.45f);       // locality
            if (j == 0) w += 0.6f;                    // attention sink
            if (j == i) w += 1.2f;                    // self
            w += jitter(rng_);
            payload_data_[static_cast<size_t>(i) * n_tok + j] = w;
            sum += w;
        }
        if (sum > 0.0f) {
            for (uint32_t j = 0; j <= i; ++j) {
                payload_data_[static_cast<size_t>(i) * n_tok + j] /= sum;
            }
        }
    }

    payload_header_ = PayloadHeader{};
    payload_header_.seq = seq_;
    payload_header_.ne[0] = n_tok;
    payload_header_.ne[1] = n_tok;
    payload_header_.ne[2] = 1;
    payload_header_.ne[3] = 1;
    payload_header_.n_floats = static_cast<uint32_t>(payload_data_.size());
    payload_header_.token_index = token_index_;
    payload_header_.kind = static_cast<uint8_t>(NodeKind::AttnScores);

    // Point the header at the kq_soft_max node of the captured layer if we have one.
    for (const auto& n2 : graph_) {
        if (n2.kind == NodeKind::AttnScores && n2.layer == capture_layer_) {
            payload_header_.name_id = n2.name_id;
            break;
        }
    }
    ++payload_version_;
}

void FakeSource::emit_token(uint64_t token_start_ns) {
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    uint64_t t = token_start_ns;

    for (const Node& n : graph_) {
        NodeRecord r{};
        r.seq = seq_++;
        r.t_start_ns = t - epoch_ns_;

        // Latency wobbles around the node's baseline, with an occasional spike so
        // the anomaly ledger has real input rather than a hardcoded demo string.
        float scale = 1.0f + 0.18f * noise(rng_);
        if (scale < 0.4f) scale = 0.4f;
        const bool spike = unit(rng_) < 0.0015f;
        if (spike) scale *= 9.0f;
        r.dur_ns = static_cast<uint32_t>(static_cast<float>(n.base_ns) * scale);
        t += r.dur_ns;

        r.ne[0] = n.ne[0];
        r.ne[1] = n.ne[1];
        r.ne[2] = n.kind == NodeKind::AttnScores ? n.ne[2] : n.ne[2];
        r.ne[3] = n.ne[3];
        r.name_id = n.name_id;
        r.kind = static_cast<uint8_t>(n.kind);
        r.layer = n.layer;
        r.dtype = 0;  // F32
        r.op_id = 0;
        r.backend = static_cast<uint8_t>(BackendId::CPU);
        r.token_index = token_index_;
        r.flags = kFlagHostBuf;

        // Plausible activation statistics per node type.
        switch (n.kind) {
            case NodeKind::AttnScores:
                r.mean = 0.02f + 0.01f * unit(rng_);
                r.absmax = 0.85f + 0.15f * unit(rng_);
                r.minval = 0.0f;
                r.sparsity = 0.55f + 0.3f * unit(rng_);
                break;
            case NodeKind::Ffn:
                r.mean = 0.01f * noise(rng_);
                r.absmax = 4.0f + 3.0f * unit(rng_);
                r.minval = -r.absmax * 0.8f;
                r.sparsity = 0.35f + 0.25f * unit(rng_);
                break;
            case NodeKind::Norm:
                r.mean = 0.002f * noise(rng_);
                r.absmax = 1.6f + 0.4f * unit(rng_);
                r.minval = -r.absmax;
                r.sparsity = 0.02f * unit(rng_);
                break;
            default:
                r.mean = 0.05f * noise(rng_);
                r.absmax = 2.5f + 2.0f * unit(rng_);
                r.minval = -r.absmax * 0.9f;
                r.sparsity = 0.1f + 0.2f * unit(rng_);
                break;
        }

        // A rare genuine outlier, the kind that signals fp16 clipping risk.
        const bool outlier = unit(rng_) < 0.0008f;
        if (outlier) {
            r.absmax *= 40.0f;
        }

        pending_.push_back(r);
        ++records_total_;

        if (outlier) {
            AnomalyRecord a{};
            a.seq = r.seq;
            a.t_start_ns = r.t_start_ns;
            a.value = r.absmax;
            a.threshold = 100.0f;
            a.layer = n.layer;
            a.name_id = n.name_id;
            a.kind = static_cast<uint8_t>(AnomalyKind::OutlierMagnitude);
            a.severity = static_cast<uint8_t>(Severity::Warn);
            pending_anomalies_.push_back(a);
        }
        if (spike) {
            AnomalyRecord a{};
            a.seq = r.seq;
            a.t_start_ns = r.t_start_ns;
            a.value = static_cast<float>(r.dur_ns) / 1e6f;
            a.threshold = static_cast<float>(n.base_ns) * 8.0f / 1e6f;
            a.layer = n.layer;
            a.name_id = n.name_id;
            a.kind = static_cast<uint8_t>(AnomalyKind::LatencySpike);
            a.severity = static_cast<uint8_t>(Severity::Info);
            pending_anomalies_.push_back(a);
        }
    }

    TokenInfo tok;
    tok.index = token_index_;
    tok.id = 1000 + static_cast<int32_t>(token_index_ % vocab_.size());
    tok.piece = vocab_[token_index_ % vocab_.size()];
    tok.is_prompt = token_index_ < 6;
    pending_tokens_.push_back(tok);

    ++token_index_;
    rebuild_attention_payload();
}

void FakeSource::advance(uint64_t now) {
    if (stopped_ || token_index_ >= opts_.max_tokens) {
        return;
    }
    const uint64_t period_ns = static_cast<uint64_t>(1e9 / opts_.tokens_per_sec);
    // Cap catch-up so a long stall does not dump thousands of tokens at once.
    int budget = 4;
    while (now >= next_token_ns_ && budget-- > 0 && token_index_ < opts_.max_tokens) {
        emit_token(next_token_ns_);
        next_token_ns_ += period_ns;
    }
    if (now > next_token_ns_ + 5 * period_ns) {
        next_token_ns_ = now;  // we fell far behind; resync rather than sprint
    }
}

size_t FakeSource::poll_names(std::vector<Topology::Entry>& out) {
    size_t n = 0;
    while (names_cursor_ < names_.size()) {
        out.push_back(names_[names_cursor_++]);
        ++n;
    }
    return n;
}

size_t FakeSource::poll_records(std::vector<NodeRecord>& out, size_t max_items) {
    advance(now_ns());
    const size_t n = std::min(pending_.size(), max_items);
    out.insert(out.end(), pending_.begin(), pending_.begin() + n);
    pending_.erase(pending_.begin(), pending_.begin() + n);
    return n;
}

size_t FakeSource::poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) {
    const size_t n = std::min(pending_anomalies_.size(), max_items);
    out.insert(out.end(), pending_anomalies_.begin(), pending_anomalies_.begin() + n);
    pending_anomalies_.erase(pending_anomalies_.begin(), pending_anomalies_.begin() + n);
    return n;
}

size_t FakeSource::poll_tokens(std::vector<TokenInfo>& out) {
    const size_t n = pending_tokens_.size();
    out.insert(out.end(), pending_tokens_.begin(), pending_tokens_.end());
    pending_tokens_.clear();
    return n;
}

bool FakeSource::read_payload(PayloadHeader& header, std::vector<float>& data) const {
    if (payload_data_.empty()) {
        return false;
    }
    header = payload_header_;
    data = payload_data_;
    return true;
}

bool FakeSource::running() const {
    return !stopped_ && token_index_ < opts_.max_tokens;
}

SourceStats FakeSource::stats() const {
    SourceStats s;
    s.records_total = records_total_;
    s.records_dropped = 0;
    s.tokens_done = token_index_;
    s.tokens_per_sec = opts_.tokens_per_sec;
    s.bytes_observed = records_total_ * 4096;
    s.payload_bytes = payload_data_.size() * sizeof(float);
    return s;
}

std::string FakeSource::status_line() const {
    return std::string("DEMO (synthetic model, no inference) ") + std::to_string(token_index_) +
           "/" + std::to_string(opts_.max_tokens) + " tokens";
}

}  // namespace llmscope

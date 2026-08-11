#include "tracer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "tensor_stats.hpp"
#include "trace_core/topology.hpp"

namespace llmscope {

namespace {

uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count());
}

BackendId backend_from_buffer(const ggml_tensor* t) {
    if (t == nullptr || t->buffer == nullptr) {
        return BackendId::Unknown;
    }
    const char* name = ggml_backend_buffer_name(t->buffer);
    if (name == nullptr) {
        return BackendId::Unknown;
    }
    if (std::strncmp(name, "CPU", 3) == 0)     return BackendId::CPU;
    if (std::strncmp(name, "CUDA", 4) == 0)    return BackendId::CUDA;
    if (std::strncmp(name, "Metal", 5) == 0)   return BackendId::Metal;
    if (std::strncmp(name, "Vulkan", 6) == 0)  return BackendId::Vulkan;
    if (std::strncmp(name, "SYCL", 4) == 0)    return BackendId::SYCL;
    if (std::strncmp(name, "BLAS", 4) == 0)    return BackendId::BLAS;
    return BackendId::Other;
}

void silent_log(ggml_log_level, const char*, void*) {}

}  // namespace

Tracer::Tracer(LiveSource& sink, TracerConfig cfg) : sink_(sink), cfg_(std::move(cfg)) {
    device_buf_.reserve(1u << 20);
    payload_buf_.resize(kMaxPayloadFloats);
}

Tracer::~Tracer() {
    unload();
}

void Tracer::unload() {
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    writer_.close();
}

bool Tracer::cb_eval_thunk(ggml_tensor* t, bool ask, void* user_data) {
    return static_cast<Tracer*>(user_data)->on_node(t, ask);
}

bool Tracer::load_model(std::string& error_out) {
    if (cfg_.quiet_llama) {
        // llama.cpp writes progress to stderr, which would shred the TUI.
        llama_log_set(silent_log, nullptr);
        ggml_log_set(silent_log, nullptr);
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg_.n_gpu_layers;

    model_ = llama_model_load_from_file(cfg_.model_path.c_str(), mparams);
    if (model_ == nullptr) {
        error_out = "failed to load model: " + cfg_.model_path;
        return false;
    }

    vocab_ = llama_model_get_vocab(model_);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(cfg_.n_ctx);
    cparams.n_batch = static_cast<uint32_t>(cfg_.n_ctx);
    cparams.n_threads = cfg_.n_threads > 0
                            ? cfg_.n_threads
                            : static_cast<int32_t>(std::thread::hardware_concurrency());
    cparams.n_threads_batch = cparams.n_threads;

    // Flash attention fuses the softmax into a single kernel, so the attention
    // score matrix never materialises as a tensor and cannot be observed. The
    // heatmap requires it off - this is a hard requirement, not a preference.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    // The hook itself. This is the entire non-invasive mechanism.
    cparams.cb_eval = &Tracer::cb_eval_thunk;
    cparams.cb_eval_user_data = this;

    ctx_ = llama_init_from_model(model_, cparams);
    if (ctx_ == nullptr) {
        error_out = "failed to create llama context";
        return false;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    // Greedy: a deterministic run makes traces comparable between sessions.
    llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

    return true;
}

void Tracer::publish_model_info() {
    ModelInfo info;
    info.path = cfg_.model_path;

    char desc[256] = {0};
    if (llama_model_desc(model_, desc, sizeof(desc)) > 0) {
        info.name = desc;
    } else {
        const size_t slash = cfg_.model_path.find_last_of("/\\");
        info.name = slash == std::string::npos ? cfg_.model_path
                                               : cfg_.model_path.substr(slash + 1);
    }
    info.arch = "llama.cpp";
    info.quant = desc;  // llama_model_desc already encodes the quantisation type
    info.n_layer = llama_model_n_layer(model_);
    info.n_head = llama_model_n_head(model_);
    info.n_head_kv = llama_model_n_head_kv(model_);
    info.n_embd = llama_model_n_embd(model_);
    info.n_ctx = static_cast<int32_t>(llama_n_ctx(ctx_));
    info.n_vocab = llama_vocab_n_tokens(vocab_);
    info.backend = cfg_.n_gpu_layers > 0 ? BackendId::CUDA : BackendId::CPU;
    info.flash_attn_disabled = true;

    sink_.set_model_info(info);
    if (writer_.is_open()) {
        writer_.write_model(info);
    }
}

// ---------------------------------------------------------------------------
// The hot path
// ---------------------------------------------------------------------------

bool Tracer::on_node(ggml_tensor* t, bool ask) {
    if (ask) {
        // Yes, we want every node. This forces one-node graph views plus a
        // backend sync, which is what isolates the timing.
        node_start_ns_ = now_ns();
        return true;
    }

    const uint64_t end_ns = now_ns();

    if (t == nullptr) {
        return !sink_.stop_requested();
    }

    const std::string name = t->name[0] != '\0' ? t->name : "(unnamed)";
    const ParsedName parsed = parse_node_name(name);
    const uint16_t name_id = sink_.observe_name(name, parsed.kind, parsed.layer);

    // The name chunk must reach the file before any record that references it,
    // otherwise replay resolves every node to "<unknown>" and builds no tree.
    if (writer_.is_open() && written_names_.insert(name_id).second) {
        writer_.write_name(name_id, name, static_cast<uint8_t>(parsed.kind), parsed.layer);
    }

    // Reach the tensor's bytes. Host-visible buffers (the CPU backend) need no
    // copy at all, which is most of why CPU tracing is cheap.
    const void* data = nullptr;
    bool is_host = false;
    if (t->buffer != nullptr) {
        is_host = ggml_backend_buffer_is_host(t->buffer);
    }
    if (is_host) {
        data = t->data;
    } else if (t->buffer != nullptr && is_readable_type(t)) {
        const size_t nbytes = ggml_nbytes(t);
        if (nbytes <= (64u << 20)) {  // refuse absurd copies
            device_buf_.resize(nbytes);
            ggml_backend_tensor_get(t, device_buf_.data(), 0, nbytes);
            data = device_buf_.data();
        }
    }

    TensorStats st;
    if (data != nullptr) {
        st = compute_stats(t, data, cfg_.sample_budget);
    }

    NodeRecord rec{};
    rec.seq = seq_++;
    rec.t_start_ns = node_start_ns_ - epoch_ns_;
    rec.dur_ns = static_cast<uint32_t>(std::min<uint64_t>(end_ns - node_start_ns_, 0xFFFFFFFFull));
    rec.ne[0] = t->ne[0];
    rec.ne[1] = t->ne[1];
    rec.ne[2] = t->ne[2];
    rec.ne[3] = t->ne[3];
    rec.mean = st.mean;
    rec.absmax = st.absmax;
    rec.minval = st.minval;
    rec.sparsity = st.sparsity;
    rec.token_index = token_index_;
    rec.layer = parsed.layer;
    rec.name_id = name_id;
    rec.op_id = static_cast<uint16_t>(t->op);
    rec.kind = static_cast<uint8_t>(parsed.kind);
    rec.dtype = static_cast<uint8_t>(t->type);
    rec.backend = static_cast<uint8_t>(backend_from_buffer(t));
    rec.flags = kFlagNone;
    if (st.has_nan) rec.flags |= kFlagHasNaN;
    if (st.has_inf) rec.flags |= kFlagHasInf;
    if (is_host)    rec.flags |= kFlagHostBuf;
    if (st.sampled) rec.flags |= kFlagStatsSampled;

    if (data != nullptr && maybe_capture_payload(t, data, rec, parsed.kind)) {
        rec.flags |= kFlagPayload;
    }

    sink_.push_record(rec);
    sink_.add_observed_bytes(ggml_nbytes(t));
    if (writer_.is_open()) {
        writer_.write_record(rec);
    }

    check_anomalies(rec);

    // Returning false aborts the graph, which is how the TUI's quit key stops
    // inference promptly instead of waiting for the token to finish.
    return !sink_.stop_requested();
}

bool Tracer::maybe_capture_payload(const ggml_tensor* t, const void* data, const NodeRecord& rec,
                                   NodeKind kind) {
    const bool is_attention_target =
        kind == NodeKind::AttnScores && rec.layer == sink_.capture_layer();
    const bool is_explicit_target = rec.name_id == sink_.capture_name_id();

    if (!is_attention_target && !is_explicit_target) {
        return false;
    }

    uint32_t n = 0;
    PayloadHeader header{};
    header.seq = rec.seq;
    header.token_index = rec.token_index;
    header.name_id = rec.name_id;
    header.kind = static_cast<uint8_t>(kind);

    if (is_attention_target && t->ne[2] > 1) {
        // Attention scores are [n_kv, n_tokens, n_head]; take one head's plane.
        int64_t head = sink_.capture_head();
        if (head >= t->ne[2]) head = t->ne[2] - 1;
        if (head < 0) head = 0;
        n = extract_plane(t, data, head, payload_buf_.data(), kMaxPayloadFloats);
        header.ne[0] = t->ne[0];
        header.ne[1] = t->ne[1];
        header.ne[2] = 1;
        header.ne[3] = 1;
    } else {
        n = extract_floats(t, data, payload_buf_.data(), kMaxPayloadFloats);
        header.ne[0] = t->ne[0];
        header.ne[1] = t->ne[1];
        header.ne[2] = t->ne[2];
        header.ne[3] = t->ne[3];
    }

    if (n == 0) {
        return false;
    }
    header.n_floats = n;
    sink_.publish_payload(header, payload_buf_.data(), n);
    if (writer_.is_open()) {
        writer_.write_payload(header, payload_buf_.data());
    }
    return true;
}

void Tracer::check_anomalies(const NodeRecord& rec) {
    if (!cfg_.anomaly.enabled) {
        return;
    }

    auto emit = [&](AnomalyKind kind, float value, float threshold, Severity sev) {
        // One report per (node, rule) per token. A genuinely stuck node still
        // shows up every token; a merely noisy one no longer floods the ledger.
        const uint32_t key = (static_cast<uint32_t>(rec.name_id) << 8) |
                             static_cast<uint32_t>(kind);
        auto slot = anomaly_last_token_.find(key);
        if (slot != anomaly_last_token_.end() && slot->second == rec.token_index) {
            return;
        }
        anomaly_last_token_[key] = rec.token_index;

        AnomalyRecord a{};
        a.seq = rec.seq;
        a.t_start_ns = rec.t_start_ns;
        a.value = value;
        a.threshold = threshold;
        a.layer = rec.layer;
        a.name_id = rec.name_id;
        a.kind = static_cast<uint8_t>(kind);
        a.severity = static_cast<uint8_t>(sev);
        sink_.push_anomaly(a);
        if (writer_.is_open()) {
            writer_.write_anomaly(a);
        }
    };

    if (rec.flags & kFlagHasNaN) {
        emit(AnomalyKind::NaN, 0.0f, 0.0f, Severity::Error);
    }
    if (rec.flags & kFlagHasInf) {
        emit(AnomalyKind::Inf, 0.0f, 0.0f, Severity::Error);
    }
    if (rec.absmax > cfg_.anomaly.magnitude_ceiling) {
        emit(AnomalyKind::OutlierMagnitude, rec.absmax, cfg_.anomaly.magnitude_ceiling,
             Severity::Warn);
    }
    if (rec.sparsity > cfg_.anomaly.sparsity_ceiling &&
        element_count(rec) >= cfg_.anomaly.sparsity_min_elements) {
        emit(AnomalyKind::HighSparsity, rec.sparsity, cfg_.anomaly.sparsity_ceiling,
             Severity::Info);
    }

    // Latency spike vs this node's own moving average. Comparing a node only
    // against itself avoids flagging every matmul just for being slower than a
    // layernorm.
    auto it = latency_ema_.find(rec.name_id);
    const float dur = static_cast<float>(rec.dur_ns);
    if (it == latency_ema_.end()) {
        latency_ema_.emplace(rec.name_id, dur);
    } else {
        const float ema = it->second;
        if (rec.dur_ns > cfg_.anomaly.latency_min_ns && ema > 0.0f &&
            dur > ema * cfg_.anomaly.latency_factor) {
            emit(AnomalyKind::LatencySpike, dur / 1e6f,
                 ema * cfg_.anomaly.latency_factor / 1e6f, Severity::Warn);
        }
        it->second = ema * 0.9f + dur * 0.1f;
    }
}

// ---------------------------------------------------------------------------
// Run loop
// ---------------------------------------------------------------------------

bool Tracer::run(std::string& error_out) {
    epoch_ns_ = now_ns();

    if (!cfg_.trace_out.empty()) {
        std::string werr;
        if (!writer_.open(cfg_.trace_out, werr)) {
            error_out = werr;
            return false;
        }
    }

    sink_.set_status("loading model...");
    if (!load_model(error_out)) {
        sink_.set_running(false);
        return false;
    }
    publish_model_info();

    // Seed the capture target so a headless recording still contains attention
    // matrices. When the TUI is attached it re-asserts its own selection every
    // frame, so this only ever acts as the default.
    sink_.set_capture_target(kInvalidNameId, cfg_.capture_layer);

    // Tokenize the prompt.
    const bool add_bos = llama_vocab_get_add_bos(vocab_);
    std::vector<llama_token> tokens(static_cast<size_t>(cfg_.n_ctx));
    const int32_t n_prompt = llama_tokenize(vocab_, cfg_.prompt.c_str(),
                                            static_cast<int32_t>(cfg_.prompt.size()),
                                            tokens.data(), static_cast<int32_t>(tokens.size()),
                                            add_bos, true);
    if (n_prompt <= 0) {
        error_out = "failed to tokenize prompt";
        sink_.set_running(false);
        return false;
    }
    tokens.resize(static_cast<size_t>(n_prompt));

    auto piece_of = [&](llama_token tok) {
        char buf[256];
        const int32_t n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
    };

    for (int32_t i = 0; i < n_prompt; ++i) {
        TokenInfo ti;
        ti.index = static_cast<uint32_t>(i);
        ti.id = tokens[static_cast<size_t>(i)];
        ti.piece = piece_of(tokens[static_cast<size_t>(i)]);
        ti.is_prompt = true;
        sink_.push_token(ti);
        if (writer_.is_open()) {
            writer_.write_token(ti);
        }
    }

    sink_.set_running(true);
    sink_.set_status("processing prompt...");

    // Prompt pass. This is the interesting one for attention: n_tokens > 1 means
    // the score tensor is a real 2D matrix rather than a single row.
    token_index_ = 0;
    if (llama_decode(ctx_, llama_batch_get_one(tokens.data(), n_prompt)) != 0) {
        error_out = "llama_decode failed on the prompt";
        sink_.set_running(false);
        return false;
    }

    sink_.set_status("generating...");
    token_index_ = static_cast<uint32_t>(n_prompt);

    for (int32_t i = 0; i < cfg_.n_predict; ++i) {
        if (sink_.stop_requested()) {
            break;
        }

        llama_token next = llama_sampler_sample(sampler_, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, next)) {
            sink_.set_status("done (end of generation)");
            break;
        }

        TokenInfo ti;
        ti.index = token_index_;
        ti.id = next;
        ti.piece = piece_of(next);
        ti.is_prompt = false;
        sink_.push_token(ti);
        if (writer_.is_open()) {
            writer_.write_token(ti);
        }

        if (llama_decode(ctx_, llama_batch_get_one(&next, 1)) != 0) {
            sink_.set_status("decode failed");
            break;
        }
        ++token_index_;
    }

    if (sink_.stop_requested()) {
        sink_.set_status("stopped by user");
    } else {
        sink_.set_status("done");
    }

    sink_.set_running(false);
    writer_.close();
    return true;
}

}  // namespace llmscope

#include "trace_file.hpp"

#include <chrono>
#include <cstring>

namespace llmscope {

namespace {

uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
            .count());
}

}  // namespace

// ---------------------------------------------------------------------------
// TraceWriter
// ---------------------------------------------------------------------------

TraceWriter::~TraceWriter() {
    close();
}

bool TraceWriter::open(const std::string& path, std::string& error_out) {
    out_.open(path, std::ios::binary | std::ios::trunc);
    if (!out_) {
        error_out = "cannot open trace file for writing: " + path;
        return false;
    }
    path_ = path;
    bytes_ = 0;
    put_u64(kTraceMagic);
    put_u32(kTraceFormatVersion);
    put_u32(0);  // reserved
    return true;
}

void TraceWriter::close() {
    if (out_.is_open()) {
        out_.flush();
        out_.close();
    }
}

void TraceWriter::put_raw(const void* data, size_t n) {
    out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    bytes_ += n;
}

void TraceWriter::put_u8(uint8_t v)   { put_raw(&v, sizeof(v)); }
void TraceWriter::put_u16(uint16_t v) { put_raw(&v, sizeof(v)); }
void TraceWriter::put_u32(uint32_t v) { put_raw(&v, sizeof(v)); }
void TraceWriter::put_i32(int32_t v)  { put_raw(&v, sizeof(v)); }
void TraceWriter::put_u64(uint64_t v) { put_raw(&v, sizeof(v)); }

void TraceWriter::put_str(const std::string& s) {
    const uint32_t len = static_cast<uint32_t>(s.size());
    put_u32(len);
    if (len > 0) {
        put_raw(s.data(), len);
    }
}

void TraceWriter::write_model(const ModelInfo& info) {
    if (!out_) return;
    put_u8(static_cast<uint8_t>(ChunkTag::Model));
    put_str(info.path);
    put_str(info.name);
    put_str(info.arch);
    put_str(info.quant);
    put_i32(info.n_layer);
    put_i32(info.n_head);
    put_i32(info.n_head_kv);
    put_i32(info.n_embd);
    put_i32(info.n_ctx);
    put_i32(info.n_vocab);
    put_u8(static_cast<uint8_t>(info.backend));
    put_u8(info.flash_attn_disabled ? 1 : 0);
}

void TraceWriter::write_name(uint16_t name_id, const std::string& name, uint8_t kind,
                             int32_t layer) {
    if (!out_) return;
    put_u8(static_cast<uint8_t>(ChunkTag::Name));
    put_u16(name_id);
    put_u8(kind);
    put_i32(layer);
    put_str(name);
}

void TraceWriter::write_record(const NodeRecord& rec) {
    if (!out_) return;
    put_u8(static_cast<uint8_t>(ChunkTag::Record));
    put_raw(&rec, sizeof(rec));
}

void TraceWriter::write_anomaly(const AnomalyRecord& rec) {
    if (!out_) return;
    put_u8(static_cast<uint8_t>(ChunkTag::Anomaly));
    put_raw(&rec, sizeof(rec));
}

void TraceWriter::write_token(const TokenInfo& tok) {
    if (!out_) return;
    put_u8(static_cast<uint8_t>(ChunkTag::Token));
    put_u32(tok.index);
    put_i32(tok.id);
    put_u8(tok.is_prompt ? 1 : 0);
    put_str(tok.piece);
}

void TraceWriter::write_payload(const PayloadHeader& header, const float* data) {
    if (!out_) return;
    put_u8(static_cast<uint8_t>(ChunkTag::Payload));
    put_raw(&header, sizeof(header));
    if (header.n_floats > 0 && data != nullptr) {
        put_raw(data, static_cast<size_t>(header.n_floats) * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// TraceFileSource
// ---------------------------------------------------------------------------

namespace {

// Bounds-checked cursor over the loaded file. Any short read marks the reader
// bad and stops parsing, which is how a truncated trace replays cleanly rather
// than crashing.
class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    bool ok() const { return ok_; }
    bool done() const { return !ok_ || pos_ >= size_; }
    size_t pos() const { return pos_; }

    bool take(void* dst, size_t n) {
        if (!ok_ || pos_ + n > size_) {
            ok_ = false;
            return false;
        }
        std::memcpy(dst, data_ + pos_, n);
        pos_ += n;
        return true;
    }

    uint8_t  u8()  { uint8_t v = 0;  take(&v, sizeof(v)); return v; }
    uint16_t u16() { uint16_t v = 0; take(&v, sizeof(v)); return v; }
    uint32_t u32() { uint32_t v = 0; take(&v, sizeof(v)); return v; }
    int32_t  i32() { int32_t v = 0;  take(&v, sizeof(v)); return v; }
    uint64_t u64() { uint64_t v = 0; take(&v, sizeof(v)); return v; }

    std::string str() {
        const uint32_t len = u32();
        if (!ok_ || pos_ + len > size_) {
            ok_ = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    bool ok_ = true;
};

}  // namespace

std::unique_ptr<TraceFileSource> TraceFileSource::load(const std::string& path,
                                                       std::string& error_out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        error_out = "cannot open trace file: " + path;
        return nullptr;
    }
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        error_out = "failed to read trace file: " + path;
        return nullptr;
    }

    Reader r(bytes.data(), bytes.size());
    if (r.u64() != kTraceMagic) {
        error_out = "not an llmscope trace file: " + path;
        return nullptr;
    }
    const uint32_t version = r.u32();
    if (version != kTraceFormatVersion) {
        error_out = "trace format v" + std::to_string(version) + " but this build expects v" +
                    std::to_string(kTraceFormatVersion);
        return nullptr;
    }
    r.u32();  // reserved

    auto src = std::unique_ptr<TraceFileSource>(new TraceFileSource());
    src->path_ = path;

    while (!r.done()) {
        const auto tag = static_cast<ChunkTag>(r.u8());
        if (!r.ok()) break;

        switch (tag) {
            case ChunkTag::Model: {
                ModelInfo m;
                m.path = r.str();
                m.name = r.str();
                m.arch = r.str();
                m.quant = r.str();
                m.n_layer = r.i32();
                m.n_head = r.i32();
                m.n_head_kv = r.i32();
                m.n_embd = r.i32();
                m.n_ctx = r.i32();
                m.n_vocab = r.i32();
                m.backend = static_cast<BackendId>(r.u8());
                m.flash_attn_disabled = r.u8() != 0;
                if (r.ok()) src->model_ = m;
                break;
            }
            case ChunkTag::Name: {
                Topology::Entry e;
                e.name_id = r.u16();
                e.kind = r.u8();
                e.layer = r.i32();
                e.name = r.str();
                if (r.ok()) src->names_.push_back(std::move(e));
                break;
            }
            case ChunkTag::Record: {
                NodeRecord rec{};
                if (r.take(&rec, sizeof(rec))) src->records_.push_back(rec);
                break;
            }
            case ChunkTag::Anomaly: {
                AnomalyRecord rec{};
                if (r.take(&rec, sizeof(rec))) src->anomalies_.push_back(rec);
                break;
            }
            case ChunkTag::Token: {
                TokenInfo t;
                t.index = r.u32();
                t.id = r.i32();
                t.is_prompt = r.u8() != 0;
                t.piece = r.str();
                if (r.ok()) src->tokens_.push_back(std::move(t));
                break;
            }
            case ChunkTag::Payload: {
                StoredPayload p;
                if (!r.take(&p.header, sizeof(p.header))) break;
                const uint32_t n = p.header.n_floats > kMaxPayloadFloats ? kMaxPayloadFloats
                                                                        : p.header.n_floats;
                p.data.resize(n);
                if (n > 0 && !r.take(p.data.data(), static_cast<size_t>(n) * sizeof(float))) {
                    break;
                }
                src->payloads_.push_back(std::move(p));
                break;
            }
            default:
                // Unknown tag means the stream is no longer trustworthy; keep
                // whatever was parsed so far and stop.
                error_out.clear();
                goto done_parsing;
        }
    }

done_parsing:
    if (src->records_.empty() && src->names_.empty()) {
        error_out = "trace file contains no records: " + path;
        return nullptr;
    }

    if (!src->records_.empty()) {
        src->trace_duration_ns_ = src->records_.back().t_start_ns +
                                  src->records_.back().dur_ns;
    }
    src->epoch_ns_ = now_ns();
    error_out.clear();
    return src;
}

uint64_t TraceFileSource::current_trace_ns() const {
    if (paused_) {
        return offset_ns_;
    }
    const uint64_t elapsed = now_ns() - epoch_ns_;
    return offset_ns_ + static_cast<uint64_t>(static_cast<double>(elapsed) * speed_);
}

size_t TraceFileSource::poll_names(std::vector<Topology::Entry>& out) {
    // Names are handed over immediately and in full: the tree must exist before
    // the first record referencing it arrives.
    size_t n = 0;
    while (names_cursor_ < names_.size()) {
        out.push_back(names_[names_cursor_++]);
        ++n;
    }
    return n;
}

size_t TraceFileSource::poll_records(std::vector<NodeRecord>& out, size_t max_items) {
    const uint64_t cutoff = current_trace_ns();
    size_t n = 0;
    while (record_cursor_ < records_.size() && n < max_items) {
        const NodeRecord& rec = records_[record_cursor_];
        if (rec.t_start_ns > cutoff) {
            break;
        }
        out.push_back(rec);
        ++record_cursor_;
        ++n;
    }

    // Keep the active payload aligned with playback position.
    while (payload_cursor_ < payloads_.size() &&
           payloads_[payload_cursor_].header.seq <=
               (record_cursor_ > 0 ? records_[record_cursor_ - 1].seq : 0)) {
        active_payload_ = payload_cursor_;
        has_active_payload_ = true;
        ++payload_cursor_;
        ++payload_version_;
    }
    return n;
}

size_t TraceFileSource::poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) {
    const uint64_t cutoff = current_trace_ns();
    size_t n = 0;
    while (anomaly_cursor_ < anomalies_.size() && n < max_items) {
        if (anomalies_[anomaly_cursor_].t_start_ns > cutoff) {
            break;
        }
        out.push_back(anomalies_[anomaly_cursor_++]);
        ++n;
    }
    return n;
}

size_t TraceFileSource::poll_tokens(std::vector<TokenInfo>& out) {
    // Tokens are paced by how far the record stream has advanced, so the token
    // strip stays in step with the layer activity being shown.
    const uint32_t upto = record_cursor_ > 0 ? records_[record_cursor_ - 1].token_index : 0;
    size_t n = 0;
    while (token_cursor_ < tokens_.size() && tokens_[token_cursor_].index <= upto) {
        out.push_back(tokens_[token_cursor_++]);
        ++n;
    }
    return n;
}

bool TraceFileSource::read_payload(PayloadHeader& header, std::vector<float>& data) const {
    if (!has_active_payload_ || active_payload_ >= payloads_.size()) {
        return false;
    }
    const StoredPayload& p = payloads_[active_payload_];
    header = p.header;
    data = p.data;
    return true;
}

bool TraceFileSource::running() const {
    return !stopped_ && record_cursor_ < records_.size();
}

SourceStats TraceFileSource::stats() const {
    SourceStats s;
    s.records_total = record_cursor_;
    s.records_dropped = 0;
    s.tokens_done = token_cursor_;
    const uint64_t t = current_trace_ns();
    if (t > 0 && token_cursor_ > 0) {
        s.tokens_per_sec = static_cast<double>(token_cursor_) / (static_cast<double>(t) / 1e9);
    }
    return s;
}

std::string TraceFileSource::status_line() const {
    std::string s = "REPLAY ";
    s += paused_ ? "[paused] " : "[playing] ";
    s += std::to_string(record_cursor_) + "/" + std::to_string(records_.size());
    return s;
}

void TraceFileSource::set_paused(bool p) {
    if (p == paused_) {
        return;
    }
    if (p) {
        offset_ns_ = current_trace_ns();  // bank progress before stopping the clock
        paused_ = true;
    } else {
        paused_ = false;
        epoch_ns_ = now_ns();
    }
}

void TraceFileSource::set_speed(double s) {
    if (s < 0.05) s = 0.05;
    if (s > 64.0) s = 64.0;
    offset_ns_ = current_trace_ns();  // rebase so the change is not retroactive
    epoch_ns_ = now_ns();
    speed_ = s;
}

void TraceFileSource::restart() {
    record_cursor_ = 0;
    anomaly_cursor_ = 0;
    token_cursor_ = 0;
    payload_cursor_ = 0;
    names_cursor_ = 0;
    has_active_payload_ = false;
    offset_ns_ = 0;
    epoch_ns_ = now_ns();
    stopped_ = false;
}

double TraceFileSource::progress() const {
    if (records_.empty()) {
        return 0.0;
    }
    return static_cast<double>(record_cursor_) / static_cast<double>(records_.size());
}

}  // namespace llmscope

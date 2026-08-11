// trace_file.hpp - persistence and replay.
//
// The file is a flat stream of tagged chunks rather than a header-plus-body
// layout, because node names are discovered *during* the run, not before it.
// A chunk stream also degrades gracefully: if the process is killed mid-trace,
// everything written up to that point still replays.
#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "record.hpp"
#include "trace_source.hpp"

namespace llmscope {

enum class ChunkTag : uint8_t {
    Model = 1,
    Name = 2,
    Record = 3,
    Anomaly = 4,
    Token = 5,
    Payload = 6,
};

class TraceWriter {
public:
    TraceWriter() = default;
    ~TraceWriter();

    bool open(const std::string& path, std::string& error_out);
    bool is_open() const { return out_.is_open(); }
    void close();

    void write_model(const ModelInfo& info);
    void write_name(uint16_t name_id, const std::string& name, uint8_t kind, int32_t layer);
    void write_record(const NodeRecord& rec);
    void write_anomaly(const AnomalyRecord& rec);
    void write_token(const TokenInfo& tok);
    void write_payload(const PayloadHeader& header, const float* data);

    uint64_t bytes_written() const { return bytes_; }
    const std::string& path() const { return path_; }

private:
    void put_u8(uint8_t v);
    void put_u16(uint16_t v);
    void put_u32(uint32_t v);
    void put_i32(int32_t v);
    void put_u64(uint64_t v);
    void put_str(const std::string& s);
    void put_raw(const void* data, size_t n);

    std::ofstream out_;
    std::string path_;
    uint64_t bytes_ = 0;
};

// Replays a recorded trace. Everything is loaded up front - a trace of a few
// thousand tokens is a handful of megabytes, and having it all in memory is what
// makes instant restart and scrubbing possible.
class TraceFileSource : public TraceSource {
public:
    static std::unique_ptr<TraceFileSource> load(const std::string& path, std::string& error_out);

    ModelInfo model_info() const override { return model_; }

    size_t poll_names(std::vector<Topology::Entry>& out) override;
    size_t poll_records(std::vector<NodeRecord>& out, size_t max_items) override;
    size_t poll_anomalies(std::vector<AnomalyRecord>& out, size_t max_items) override;
    size_t poll_tokens(std::vector<TokenInfo>& out) override;

    uint64_t payload_version() const override { return payload_version_; }
    bool read_payload(PayloadHeader& header, std::vector<float>& data) const override;

    void set_capture_target(uint16_t, int32_t) override {}  // fixed at record time

    bool running() const override;
    SourceStats stats() const override;
    std::string status_line() const override;
    void request_stop() override { stopped_ = true; }

    bool supports_seek() const override { return true; }
    void set_paused(bool p) override;
    bool paused() const override { return paused_; }
    void set_speed(double s) override;
    double speed() const override { return speed_; }
    void restart() override;
    double progress() const override;

private:
    // Wall-clock ns of "now" in trace time, honouring pause and speed.
    uint64_t current_trace_ns() const;

    ModelInfo model_;
    std::vector<Topology::Entry> names_;
    std::vector<NodeRecord> records_;
    std::vector<AnomalyRecord> anomalies_;
    std::vector<TokenInfo> tokens_;

    struct StoredPayload {
        PayloadHeader header;
        std::vector<float> data;
    };
    std::vector<StoredPayload> payloads_;

    size_t names_cursor_ = 0;
    size_t record_cursor_ = 0;
    size_t anomaly_cursor_ = 0;
    size_t token_cursor_ = 0;
    size_t payload_cursor_ = 0;
    mutable uint64_t payload_version_ = 0;
    mutable size_t active_payload_ = 0;
    bool has_active_payload_ = false;

    uint64_t trace_duration_ns_ = 0;
    uint64_t epoch_ns_ = 0;      // steady-clock ns when playback (re)started
    uint64_t offset_ns_ = 0;     // trace-time position banked before a pause
    double speed_ = 1.0;
    bool paused_ = false;
    bool stopped_ = false;
    std::string path_;
};

}  // namespace llmscope

// payload_slot.hpp - lock-free handoff of one large tensor snapshot.
//
// NodeRecord carries only scalar stats. When the user selects a node in the TUI
// (or an attention layer becomes the capture target), the tracer additionally
// publishes the tensor's full float32 contents through here.
//
// Exactly one tensor is in flight at a time. That is the whole point: streaming
// every activation of a 1B model is gigabytes per second, and we would be
// profiling the tracer rather than the model.
//
// Mechanism: double buffer plus a generation counter. The writer fills the
// inactive slot and then publishes it atomically, so a reader never reads the
// buffer being written unless the writer laps it - which the generation check
// detects, causing the reader to retry or skip. Both buffers are preallocated so
// no reallocation can move memory out from under a reader.
#pragma once

#include <atomic>
#include <cstring>
#include <vector>

#include "record.hpp"

namespace llmscope {

class PayloadSlot {
public:
    PayloadSlot() {
        for (auto& buf : buffers_) {
            buf.data.resize(kMaxPayloadFloats);
        }
    }

    // Producer (tracer thread). n_floats is clamped to kMaxPayloadFloats.
    void publish(const PayloadHeader& header, const float* src, uint32_t n_floats) {
        const int write_idx = 1 - active_.load(std::memory_order_relaxed);
        Buffer& buf = buffers_[write_idx];

        const uint32_t n = n_floats > kMaxPayloadFloats ? kMaxPayloadFloats : n_floats;

        buf.generation.store(buf.generation.load(std::memory_order_relaxed) + 1,
                             std::memory_order_relaxed);

        buf.header = header;
        buf.header.n_floats = n;
        buf.header.truncated = (n_floats > kMaxPayloadFloats) ? 1u : 0u;
        if (src != nullptr && n > 0) {
            std::memcpy(buf.data.data(), src, static_cast<size_t>(n) * sizeof(float));
        }

        const uint64_t gen = buf.generation.load(std::memory_order_relaxed);
        buf.published.store(gen, std::memory_order_release);
        active_.store(write_idx, std::memory_order_release);
        version_.fetch_add(1, std::memory_order_release);
    }

    // Monotonic counter of publishes; lets the TUI skip redundant copies.
    uint64_t version() const noexcept { return version_.load(std::memory_order_acquire); }

    // Consumer (TUI thread). Returns false if nothing has been published yet or
    // the writer lapped us mid-read.
    bool try_read(PayloadHeader& header_out, std::vector<float>& data_out) const {
        const int idx = active_.load(std::memory_order_acquire);
        const Buffer& buf = buffers_[idx];

        const uint64_t before = buf.published.load(std::memory_order_acquire);
        if (before == 0) {
            return false;  // never written
        }

        header_out = buf.header;
        const uint32_t n = header_out.n_floats > kMaxPayloadFloats ? kMaxPayloadFloats
                                                                  : header_out.n_floats;
        data_out.resize(n);
        if (n > 0) {
            std::memcpy(data_out.data(), buf.data.data(), static_cast<size_t>(n) * sizeof(float));
        }

        const uint64_t after = buf.generation.load(std::memory_order_acquire);
        return before == after;  // false => writer lapped us, caller retries next frame
    }

private:
    struct Buffer {
        std::vector<float> data;
        PayloadHeader header{};
        std::atomic<uint64_t> generation{0};
        std::atomic<uint64_t> published{0};
    };

    Buffer buffers_[2];
    std::atomic<int> active_{0};
    std::atomic<uint64_t> version_{0};
};

}  // namespace llmscope

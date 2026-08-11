// spsc_ring.hpp - single-producer / single-consumer overwriting ring buffer.
//
// The inference thread must never block on the UI. When the consumer falls
// behind, the producer overwrites the oldest records and counts the loss so the
// TUI can display it honestly rather than silently showing a partial picture.
//
// Correctness rests on one rule: only the tracer thread calls push(), and only
// the TUI thread calls pop_batch(). Capacity is a power of two so the modulo
// reduces to a mask.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llmscope {

template <typename T>
class SpscRing {
public:
    // capacity is rounded up to the next power of two, minimum 2.
    explicit SpscRing(size_t capacity) {
        size_t cap = 2;
        while (cap < capacity) {
            cap <<= 1;
        }
        buffer_.resize(cap);
        mask_ = cap - 1;
    }

    size_t capacity() const noexcept { return buffer_.size(); }

    // Producer side. Never blocks. Returns false if the record was dropped.
    //
    // When full we drop the *incoming* record rather than overwriting the oldest.
    // Overwriting would require the producer to advance tail_, giving that
    // variable two writers - a lost update there can hand the consumer a slot
    // that is being rewritten underneath it. Keeping tail_ single-writer is what
    // makes this ring actually lock-free-correct instead of merely lock-free.
    //
    // In practice this path is close to unreachable: the TUI drains every frame
    // and the ring holds 65536 records, so overflow needs the model to emit
    // 65536 nodes inside 16 ms. It still has to be correct when it happens.
    bool push(const T& item) {
        const uint64_t head = head_.load(std::memory_order_relaxed);
        const uint64_t tail = tail_.load(std::memory_order_acquire);

        if (head - tail >= buffer_.size()) {
            dropped_.store(dropped_.load(std::memory_order_relaxed) + 1,
                           std::memory_order_relaxed);
            return false;
        }

        buffer_[head & mask_] = item;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Consumer side. Appends up to max_items into out, returns how many it took.
    size_t pop_batch(std::vector<T>& out, size_t max_items) {
        const uint64_t head = head_.load(std::memory_order_acquire);
        uint64_t tail = tail_.load(std::memory_order_relaxed);

        size_t taken = 0;
        while (tail < head && taken < max_items) {
            out.push_back(buffer_[tail & mask_]);
            ++tail;
            ++taken;
        }
        if (taken > 0) {
            tail_.store(tail, std::memory_order_release);
        }
        return taken;
    }

    // Approximate; both ends move underneath this call.
    size_t size_approx() const noexcept {
        const uint64_t head = head_.load(std::memory_order_acquire);
        const uint64_t tail = tail_.load(std::memory_order_acquire);
        return head >= tail ? static_cast<size_t>(head - tail) : 0;
    }

    uint64_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }
    uint64_t total_pushed() const noexcept { return head_.load(std::memory_order_acquire); }

private:
    std::vector<T> buffer_;
    size_t mask_ = 0;

    // Separate cache lines: false sharing between head and tail would cost more
    // than the ring saves.
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) std::atomic<uint64_t> dropped_{0};
};

}  // namespace llmscope

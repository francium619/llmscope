// name_table.hpp - interns ggml tensor names to uint16 ids.
//
// A transformer's compute graph is structurally identical on every token, so the
// same ~300 names repeat for the entire run. Interning them once keeps NodeRecord
// a fixed-size POD and shrinks a trace file by roughly an order of magnitude.
#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "record.hpp"  // kInvalidNameId

namespace llmscope {

class NameTable {
public:
    // Called from the tracer thread. Interning happens on the first forward pass
    // and then never again, so lock contention is effectively zero after warmup.
    uint16_t intern(const std::string& name);

    // Called from the TUI thread, a few dozen times per frame for visible rows.
    std::string name(uint16_t id) const;

    size_t size() const;

    // Whole-table snapshot, used by the trace file writer and by replay.
    std::vector<std::string> snapshot() const;
    void load(std::vector<std::string> names);

private:
    mutable std::mutex mutex_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, uint16_t> ids_;
};

}  // namespace llmscope

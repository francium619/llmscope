#include "name_table.hpp"

namespace llmscope {

uint16_t NameTable::intern(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ids_.find(name);
    if (it != ids_.end()) {
        return it->second;
    }
    if (names_.size() >= kInvalidNameId) {
        // 65535 distinct node names would mean the graph is not what we think it
        // is. Degrade instead of corrupting ids.
        return kInvalidNameId;
    }
    const uint16_t id = static_cast<uint16_t>(names_.size());
    names_.push_back(name);
    ids_.emplace(name, id);
    return id;
}

std::string NameTable::name(uint16_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (id >= names_.size()) {
        return "<unknown>";
    }
    return names_[id];
}

size_t NameTable::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return names_.size();
}

std::vector<std::string> NameTable::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return names_;
}

void NameTable::load(std::vector<std::string> names) {
    std::lock_guard<std::mutex> lock(mutex_);
    names_ = std::move(names);
    ids_.clear();
    for (size_t i = 0; i < names_.size(); ++i) {
        ids_.emplace(names_[i], static_cast<uint16_t>(i));
    }
}

}  // namespace llmscope

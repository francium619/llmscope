// tensor_stats.hpp - reduce a ggml tensor to scalars, in place.
//
// This is the single most performance-critical piece of llmscope. It runs inside
// the graph callback, once per node, ~300 times per token. It must not allocate,
// must not copy the tensor, and must respect ggml's strides rather than assuming
// contiguity.
//
// Large tensors are sampled on a stride instead of fully scanned. Visiting every
// element of a 1B model's activations costs more than the inference itself; a
// few thousand samples estimate mean and sparsity to well within the precision
// anyone reads off a dashboard. absmax is the one metric sampling can understate,
// which is called out in the UI via the "~" marker.
#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

namespace llmscope {

struct TensorStats {
    float mean = 0.0f;
    float absmax = 0.0f;
    float minval = 0.0f;
    float sparsity = 0.0f;
    bool has_nan = false;
    bool has_inf = false;
    bool sampled = false;    // true if computed from a stride sample
    bool supported = false;  // false for quantized types we cannot read elementwise
    uint64_t visited = 0;
};

// Default sample budget. ~8k elements is a fraction of a millisecond and keeps
// the tracer's overhead well under the cost of the node being measured.
inline constexpr uint64_t kDefaultSampleBudget = 8192;

// `data` must point at host-readable memory laid out per the tensor's ne/nb.
TensorStats compute_stats(const ggml_tensor* t, const void* data,
                          uint64_t sample_budget = kDefaultSampleBudget);

// True if compute_stats can interpret this tensor's element type.
bool is_readable_type(const ggml_tensor* t);

// Converts up to max_floats elements into out (row-major, strides respected).
// Returns how many were written. Used only for the single selected tensor.
uint32_t extract_floats(const ggml_tensor* t, const void* data, float* out, uint32_t max_floats);

// Extracts the [ne0 x ne1] plane at index i2, row-major with ne0 contiguous.
//
// Attention score tensors are [n_kv, n_tokens, n_head]. At full context that is
// millions of floats across all heads, but a single head's plane is a few
// hundred thousand - so the heatmap copies one head rather than the whole
// tensor. Returns elements written; 0 if i2 is out of range.
uint32_t extract_plane(const ggml_tensor* t, const void* data, int64_t i2, float* out,
                       uint32_t max_floats);

}  // namespace llmscope

#include "tensor_stats.hpp"

#include <cmath>
#include <cstring>

#include "ggml.h"
#include "trace_core/record.hpp"

namespace llmscope {

namespace {

// Reads one element as float, honouring the tensor's byte strides.
inline float read_element(const uint8_t* base, ggml_type type, const size_t nb[4], int64_t i0,
                          int64_t i1, int64_t i2, int64_t i3) {
    const size_t offset = static_cast<size_t>(i3) * nb[3] + static_cast<size_t>(i2) * nb[2] +
                          static_cast<size_t>(i1) * nb[1] + static_cast<size_t>(i0) * nb[0];
    const uint8_t* p = base + offset;

    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float*>(p);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(p));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(p));
        case GGML_TYPE_I8:
            return static_cast<float>(*reinterpret_cast<const int8_t*>(p));
        case GGML_TYPE_I16:
            return static_cast<float>(*reinterpret_cast<const int16_t*>(p));
        case GGML_TYPE_I32:
            return static_cast<float>(*reinterpret_cast<const int32_t*>(p));
        case GGML_TYPE_I64:
            return static_cast<float>(*reinterpret_cast<const int64_t*>(p));
        default:
            return 0.0f;
    }
}

}  // namespace

bool is_readable_type(const ggml_tensor* t) {
    if (t == nullptr) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
            return true;
        default:
            // Quantised block formats cannot be indexed elementwise without
            // dequantising, which is far too expensive for the hot path.
            return false;
    }
}

TensorStats compute_stats(const ggml_tensor* t, const void* data, uint64_t sample_budget) {
    TensorStats st;
    if (t == nullptr || data == nullptr || !is_readable_type(t)) {
        return st;
    }
    st.supported = true;

    const int64_t* ne = t->ne;
    const size_t* nb = t->nb;
    const auto* base = static_cast<const uint8_t*>(data);

    const int64_t total = ne[0] * ne[1] * ne[2] * ne[3];
    if (total <= 0) {
        return st;
    }

    // Stride so we touch at most sample_budget elements, spread across the whole
    // tensor rather than clustered at the front.
    int64_t step = 1;
    if (sample_budget > 0 && static_cast<uint64_t>(total) > sample_budget) {
        step = total / static_cast<int64_t>(sample_budget);
        if (step < 1) step = 1;
        st.sampled = true;
    }

    double sum = 0.0;
    float absmax = 0.0f;
    float minval = 0.0f;
    uint64_t zeros = 0;
    uint64_t visited = 0;
    bool first = true;

    for (int64_t flat = 0; flat < total; flat += step) {
        // Decompose the flat index using ne, then apply nb strides. This is what
        // makes non-contiguous views (which llama.cpp uses heavily for KV cache)
        // read correctly instead of returning garbage.
        int64_t rem = flat;
        const int64_t i0 = rem % ne[0]; rem /= ne[0];
        const int64_t i1 = rem % ne[1]; rem /= ne[1];
        const int64_t i2 = rem % ne[2]; rem /= ne[2];
        const int64_t i3 = rem % ne[3];

        const float v = read_element(base, t->type, nb, i0, i1, i2, i3);

        if (std::isnan(v)) {
            st.has_nan = true;
            continue;  // NaN would poison every other statistic
        }
        if (std::isinf(v)) {
            st.has_inf = true;
            continue;
        }

        const float a = std::fabs(v);
        if (a > absmax) absmax = a;
        if (first || v < minval) {
            minval = v;
            first = false;
        }
        if (a < kSparsityEps) ++zeros;
        sum += static_cast<double>(v);
        ++visited;
    }

    st.visited = visited;
    if (visited > 0) {
        st.mean = static_cast<float>(sum / static_cast<double>(visited));
        st.sparsity = static_cast<float>(zeros) / static_cast<float>(visited);
    }
    st.absmax = absmax;
    st.minval = minval;
    return st;
}

uint32_t extract_floats(const ggml_tensor* t, const void* data, float* out, uint32_t max_floats) {
    if (t == nullptr || data == nullptr || out == nullptr || !is_readable_type(t)) {
        return 0;
    }

    const int64_t* ne = t->ne;
    const size_t* nb = t->nb;
    const auto* base = static_cast<const uint8_t*>(data);

    const int64_t total = ne[0] * ne[1] * ne[2] * ne[3];
    const int64_t limit = total < static_cast<int64_t>(max_floats) ? total
                                                                  : static_cast<int64_t>(max_floats);

    // Fast path: fully contiguous float32 is just a memcpy.
    if (t->type == GGML_TYPE_F32 && nb[0] == sizeof(float) &&
        nb[1] == nb[0] * static_cast<size_t>(ne[0]) &&
        nb[2] == nb[1] * static_cast<size_t>(ne[1]) &&
        nb[3] == nb[2] * static_cast<size_t>(ne[2])) {
        std::memcpy(out, base, static_cast<size_t>(limit) * sizeof(float));
        return static_cast<uint32_t>(limit);
    }

    for (int64_t flat = 0; flat < limit; ++flat) {
        int64_t rem = flat;
        const int64_t i0 = rem % ne[0]; rem /= ne[0];
        const int64_t i1 = rem % ne[1]; rem /= ne[1];
        const int64_t i2 = rem % ne[2]; rem /= ne[2];
        const int64_t i3 = rem % ne[3];
        out[flat] = read_element(base, t->type, nb, i0, i1, i2, i3);
    }
    return static_cast<uint32_t>(limit);
}

uint32_t extract_plane(const ggml_tensor* t, const void* data, int64_t i2, float* out,
                       uint32_t max_floats) {
    if (t == nullptr || data == nullptr || out == nullptr || !is_readable_type(t)) {
        return 0;
    }
    const int64_t* ne = t->ne;
    const size_t* nb = t->nb;
    if (i2 < 0 || i2 >= ne[2]) {
        return 0;
    }
    const auto* base = static_cast<const uint8_t*>(data);

    const int64_t plane = ne[0] * ne[1];
    const int64_t limit = plane < static_cast<int64_t>(max_floats)
                              ? plane
                              : static_cast<int64_t>(max_floats);

    int64_t written = 0;
    for (int64_t i1 = 0; i1 < ne[1] && written < limit; ++i1) {
        for (int64_t i0 = 0; i0 < ne[0] && written < limit; ++i0) {
            out[written++] = read_element(base, t->type, nb, i0, i1, i2, 0);
        }
    }
    return static_cast<uint32_t>(written);
}

}  // namespace llmscope

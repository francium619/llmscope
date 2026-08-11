// record.hpp - the wire format shared by tracer, TUI, and trace files.
//
// NodeRecord is a fixed-size POD deliberately: it is written from the inference
// thread into a lock-free ring, and memcpy'd verbatim into .trace files. Adding
// a std::string here would break both. Names live in NameTable, referenced by id.
#pragma once

#include <cstdint>
#include <type_traits>

namespace llmscope {

// Sentinel for "no node". Lives here rather than in name_table.hpp because
// every module that handles a name_id needs it, including those that never
// touch the interning table.
inline constexpr uint16_t kInvalidNameId = 0xFFFF;

// Bumped whenever NodeRecord or the trace file layout changes.
inline constexpr uint32_t kTraceFormatVersion = 3;
inline constexpr uint64_t kTraceMagic = 0x45504f43534d4c4dULL;  // "LLMSCOPE"

// Coarse classification of a graph node, derived from its ggml tensor name.
// This is what lets the TUI group ~300 flat nodes into a readable tree.
enum class NodeKind : uint8_t {
    Unknown = 0,
    Embedding,   // inp_embd, token_embd
    Norm,        // attn_norm, ffn_norm, result_norm
    Attention,   // q/k/v projections, kqv, attn_out
    AttnScores,  // kq_soft_max* - the attention matrix itself
    Rope,        // positional rotation
    Cache,       // k_cache / v_cache views
    Ffn,         // ffn_up / ffn_gate / ffn_down
    LayerOut,    // l_out-N, the residual stream leaving a block
    Output,      // result_output / logits
    kCount
};

const char* to_string(NodeKind k) noexcept;

// Which device actually executed the node.
enum class BackendId : uint8_t {
    Unknown = 0,
    CPU,
    CUDA,
    Metal,
    Vulkan,
    SYCL,
    BLAS,
    Other,
    kCount
};

const char* to_string(BackendId b) noexcept;

// Bit flags on NodeRecord::flags.
enum RecordFlags : uint8_t {
    kFlagNone = 0,
    kFlagHasNaN = 1 << 0,
    kFlagHasInf = 1 << 1,
    kFlagPayload = 1 << 2,  // a full tensor snapshot was published for this node
    kFlagHostBuf = 1 << 3,  // tensor lived in host-visible memory (no copy needed)
    kFlagStatsSampled = 1 << 4,  // stats came from a strided sample, not every element
};

// One observed graph node. Keep POD and keep it 88 bytes.
struct NodeRecord {
    uint64_t seq;         // monotonic, gap => records were dropped
    uint64_t t_start_ns;  // steady-clock ns since trace start
    int64_t  ne[4];       // ggml shape, ne[0] fastest-varying
    float    mean;
    float    absmax;
    float    minval;
    float    sparsity;    // fraction of elements with |x| < kSparsityEps
    uint32_t dur_ns;      // isolated execution time for this node
    uint32_t token_index; // which llama_decode step produced this
    int32_t  layer;       // transformer block index, -1 if not layer-scoped
    uint16_t name_id;     // index into NameTable
    uint16_t op_id;       // ggml_op enum value
    uint8_t  kind;        // NodeKind
    uint8_t  dtype;       // ggml_type
    uint8_t  backend;     // BackendId
    uint8_t  flags;       // RecordFlags
    uint32_t reserved;
};

static_assert(sizeof(NodeRecord) == 88, "NodeRecord layout is part of the trace file format");
static_assert(std::is_trivially_copyable<NodeRecord>::value, "NodeRecord must be memcpy-safe");

// Elements with |x| below this count as zero for the sparsity metric.
inline constexpr float kSparsityEps = 1e-6f;

// Total element count of a record's tensor.
inline int64_t element_count(const NodeRecord& r) noexcept {
    return r.ne[0] * r.ne[1] * r.ne[2] * r.ne[3];
}

// ---------------------------------------------------------------------------
// Payload: a full float32 snapshot of one selected tensor.
//
// Streaming every activation would be gigabytes per second, so exactly one
// tensor at a time is copied - the node the user selected in the TUI, plus
// attention score matrices when an attention layer is the active target.
// ---------------------------------------------------------------------------

inline constexpr uint32_t kMaxPayloadFloats = 1u << 20;  // 4 MiB of float32

struct PayloadHeader {
    uint64_t seq;          // matching NodeRecord::seq
    int64_t  ne[4];
    uint32_t n_floats;     // how many floats are actually valid
    uint32_t token_index;
    uint16_t name_id;
    uint8_t  kind;
    uint8_t  truncated;    // 1 if the tensor was larger than kMaxPayloadFloats
    uint32_t reserved;
};

static_assert(std::is_trivially_copyable<PayloadHeader>::value, "PayloadHeader must be memcpy-safe");

// ---------------------------------------------------------------------------
// Anomalies
// ---------------------------------------------------------------------------

enum class AnomalyKind : uint8_t {
    NaN = 0,
    Inf,
    OutlierMagnitude,  // |x| exceeded the configured ceiling
    HighSparsity,      // layer output went mostly dead
    LatencySpike,      // node took k times its rolling median
    kCount
};

const char* to_string(AnomalyKind k) noexcept;

// Severity drives the colour used in the anomaly ledger pane.
enum class Severity : uint8_t { Info = 0, Warn, Error };

struct AnomalyRecord {
    uint64_t seq;
    uint64_t t_start_ns;
    float    value;      // the measurement that tripped the rule
    float    threshold;  // what it was compared against
    int32_t  layer;
    uint16_t name_id;
    uint8_t  kind;       // AnomalyKind
    uint8_t  severity;   // Severity
};

static_assert(std::is_trivially_copyable<AnomalyRecord>::value, "AnomalyRecord must be memcpy-safe");

// Thresholds for the anomaly rules. Defaults are deliberately loose so a
// healthy model stays quiet; the ledger is worthless if it cries wolf.
struct AnomalyConfig {
    // Tuned against a real Qwen2.5-0.5B trace. Large language models genuinely
    // produce activations in the hundreds - "outlier features" are a documented
    // property, not a fault - so a ceiling of 100 fires constantly and teaches
    // the user to ignore the pane. 1e4 still leaves plenty of headroom below
    // fp16's 65504 limit, which is the value that actually matters for clipping.
    float magnitude_ceiling = 1e4f;
    float sparsity_ceiling  = 0.995f;  // effectively dead, not merely sparse
    float latency_factor    = 8.0f;    // vs this node's own moving average
    uint32_t latency_min_ns = 200'000; // ignore spikes on nodes this fast
    int64_t sparsity_min_elements = 1024;  // tiny tensors are trivially sparse
    bool enabled = true;
};

}  // namespace llmscope

# llmscope — Design

**Date:** 2026-08-12
**Status:** Implemented and verified end-to-end against Qwen2.5-0.5B-Instruct (Q8_0)

---

## 1. Problem

Build a lightweight telemetry and diagnostic tool for local transformer models. It must:

- hook into a model's execution pipeline **without altering the model's code**
- capture real-time intermediate state — shapes, activation statistics, per-layer latency — as tokens flow
  through the architecture
- present that as an interactive, keyboard-driven TUI
- identify which transformer block dominates compute, visualise attention matrices, and flag numerical anomalies

Constraints: C++, two people, roughly two weeks.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Inference runtime | **llama.cpp / ggml** | Pure C/C++, builds on Windows, and exposes a first-class per-tensor eval callback. LibTorch requires a TorchScript export step; ONNX Runtime cannot surface intermediate activations without rewriting the graph, which would violate the non-invasive requirement. |
| Compute backend | **CPU, ~0.5–1B model** | 4 GB of laptop VRAM cannot hold an 8B model. More importantly the CPU backend runs synchronously, so wall-clock timing around a node measures real execution rather than kernel enqueue. |
| Process model | **One binary, two threads, plus a trace file** | A lock-free ring between inference and UI is simpler than IPC, and serialising the same record struct delivers the "Replay" requirement almost for free. |
| TUI library | **FTXUI 7.0.3** | CMake-native, works on Windows, provides flexbox layout and focus handling. Hand-rolling a layout engine is weeks of work on the least interesting part of the project. |
| Capture strategy | **Generic node capture + parsed topology + name interning** | Uniform per-node capture works on any GGUF architecture; parsing llama.cpp's consistent node names recovers the tree; interning keeps records fixed-size POD. |

## 3. The hook

The entire non-invasive mechanism:

```cpp
params.cb_eval           = &Tracer::cb_eval_thunk;
params.cb_eval_user_data = this;
params.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
```

ggml's scheduler calls the callback twice per node — `ask=true` to request data, `ask=false` with the computed
tensor. Reading `ggml_backend_sched_compute_splits` confirms that returning `true` on the ask causes the node to
be evaluated as a single-node graph view followed by `ggml_backend_synchronize`. **That synchronisation is what
makes per-node latency measurable.**

### Consequences, stated honestly

1. **The observer changes what it observes.** Answering `true` for every node defeats ggml's node batching.
   Reported latencies are *instrumented* latencies — internally consistent and correct for ranking blocks, but
   not free-running timings.
2. **Flash attention must be disabled**, or the softmax is fused and the attention matrix never exists as a
   tensor. This is a hard requirement of the feature, not a configuration preference.
3. **Statistics are sampled.** Full traversal of every activation costs more than inference. Tensors above the
   sample budget are strided and the UI marks the result `~`. `mean` and `sparsity` estimate well; `|max|` can
   be understated.

## 4. Architecture

```
app/          main(), CLI, thread orchestration
 ├── tracer/  cb_eval hook, stats kernels                     [llama.cpp + ggml]
 ├── tui/     panes, focus, keybindings, heatmap              [FTXUI]
 └── trace_core/  records, ring, payload slot, topology,      [no dependencies]
                  trace files, three TraceSource impls
```

Dependency rule: `tui/` never includes a llama.cpp header; `trace_core/` includes neither dependency. This is
enforced by the module boundaries in CMake and is what allows the UI to be developed, run, and tested with no
model present.

### TraceSource

The seam everything hangs off:

- `LiveSource` — drains the ring fed by the running model
- `TraceFileSource` — replays a recorded trace with pause, speed, and restart
- `FakeSource` — a synthetic 16-layer model with causal attention and injected anomalies

The TUI never shares a `Topology` or `NameTable` with the tracer. Newly discovered names are polled as plain
values and the TUI builds its **own** tree, which removes an entire class of data races rather than guarding
against them.

### Concurrency contract

| Tier | Frequency | Cost |
|---|---|---|
| Hot | ~300× per token | `SpscRing::push`, two relaxed atomic loads — no locks |
| Warm | ~1× per token | attention payload via double-buffered `PayloadSlot` |
| Cold | once | name interning, model metadata — mutex |

**The ring drops incoming records when full rather than overwriting the oldest.** Overwriting requires the
producer to advance `tail_`, giving that variable two writers; a lost update there can hand the consumer a slot
being rewritten beneath it. Single-writer `tail_` is what makes the ring correct, not merely lock-free. Drops are
counted and shown in the header.

The capture target is two atomics (`name_id`, `layer`) rather than a set, so the hot path reads it without a
lock 300 times per token.

## 5. Data model

`NodeRecord` is a fixed-size **88-byte POD** — deliberately, so the same struct is pushed through a lock-free
ring and `memcpy`'d verbatim into a trace file. A `std::string` member would break both. Names live in a
`NameTable`, referenced by `uint16` id, discovered once on the first forward pass.

Payloads (full tensor snapshots) travel separately, one at a time, through a double-buffered slot. Streaming
every activation would be gigabytes per second. Attention tensors are `[n_kv, n_tokens, n_head]`; only the
selected head's plane is copied.

## 6. Topology recovery

llama.cpp names nodes consistently, so the tree is parsed rather than configured:

| Pattern | Classification |
|---|---|
| `kq_soft_max-N` | attention scores — the matrix the heatmap draws |
| `attn_norm-N`, `ffn_norm-N`, `result_norm` | normalisation |
| `Qcur-N`, `Kcur-N`, `kqv_out-N`, `kq-N` | attention |
| `ffn_up-N`, `ffn_gate-N`, `ffn_down-N` | MLP |
| `cache_k_lN`, `cache_v_lN` | KV cache |
| `l_out-N` | residual stream |
| `inp_embd` | embeddings |

Test ordering matters: `attn_norm` and `ffn_norm` contain `attn`/`ffn` but are normalisation nodes, so the norm
test must precede those. This is covered by unit tests.

## 7. Attention accumulation

The prompt pass produces a genuine `[n_tokens × n_kv]` matrix; every decode step afterwards produces a single
row. Naively publishing each would let one row overwrite the matrix, so the UI **appends** decode rows,
reconstructing the full causal picture as generation proceeds. The KV axis grows with context, so existing rows
are widened and zero-filled; accumulation is capped at 128 rows.

## 8. Anomaly rules

| Rule | Default | Rationale |
|---|---|---|
| Non-finite | always | NaN/Inf are unambiguous faults |
| `\|max\|` ceiling | `1e4` | LLMs genuinely produce activations in the thousands; fp16 clips at 65504 |
| Sparsity | `> 99.5%`, ≥1024 elements | below that, "sparse" is normal |
| Latency spike | `8×` the node's own moving average | comparing a node to itself avoids flagging every matmul for being slower than a layernorm |

Each `(node, rule)` pair reports at most once per token.

**Tuned against real data, not intuition.** An initial ceiling of 100 produced 174 findings in 747 records. A
ledger that cries wolf is worse than no ledger.

## 9. Verification

- **99 unit checks**, no framework, covering ring drop accounting and ordering under real thread contention,
  name classification, tree construction, payload clamping, and trace-file round-tripping including corrupt-file
  rejection.
- **`--snapshot`** renders one frame off-screen to stdout with no TTY, making the UI verifiable
  non-interactively and in CI.
- **End-to-end**: 8478 nodes across 13 tokens captured from Qwen2.5-0.5B with zero drops; recorded to a 0.75 MB
  trace and replayed with the tree, names, attention, and anomalies all reconstructed.

## 10. Deliberately out of scope

- GPU backends. The design carries a real `backend` field and would work, but honest GPU timing needs CUDA
  events, and 4 GB of VRAM does not fit an interesting model.
- Training, gradients, backward passes.
- Multi-model or multi-process attachment.
- Trace scrubbing to an arbitrary timestamp. Replay supports pause, speed, and restart; seeking to an arbitrary
  point would need a keyframe index.

## 11. Two-person split

`trace_core/` and `tracer/` on one side, `tui/` on the other, meeting at `NodeRecord` and `TraceSource`. Because
`FakeSource` satisfies the same interface as the real tracer, the UI side is never blocked on the hook side.
Agreeing those two types is the only true day-one dependency.

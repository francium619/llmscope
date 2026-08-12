# llmscope

[![ci](https://github.com/francium619/llmscope/actions/workflows/ci.yml/badge.svg)](https://github.com/francium619/llmscope/actions/workflows/ci.yml)
[![tracer](https://github.com/francium619/llmscope/actions/workflows/tracer.yml/badge.svg)](https://github.com/francium619/llmscope/actions/workflows/tracer.yml)

Live instrumentation, tracing, and replay for local transformer inference — in the terminal.

llmscope attaches to a running [llama.cpp](https://github.com/ggml-org/llama.cpp) model **without modifying a
single line of its source**, captures what every tensor in the compute graph does as tokens flow through the
architecture, and shows it as an interactive TUI.

```
 llmscope  qwen2 1B Q8_0 L24 H14 D896 | CPU | 1.39 tok/s | nodes 12246  no drops                    done
┏ 1 MODEL TOPOLOGY  [focus] ━━━━━━━━━━━┓╭ 2 LIVE PACKET STREAM ──────────────────────────────────────────╮
┃- model                               ┃│  SEQ  TIME       LAYER TYPE          SHAPE        LATENCY  DEV │
┃  + embeddings                        ┃│ 11736 00:12.773     11 Attn          [896, 1]     1.545 ms CPU │
┃  - layers                            ┃│ 11739 00:12.776     11 Attn          [64, 14]     937.00 us CPU │
┃    - layers.0                     96%┃│ 11740 00:12.777     11 Attn          [128, 1]     2.404 ms CPU │
┃      + norm                          ┃│ 11741 00:12.779     11 Attn          [128, 1]     738.00 us CPU │
┃      + attn                          ┃╰────────────────────────────────────────────────────────────────╯
┃      + mlp                           ┃╭ 3 ATTENTION MATRIX  kq_soft_max-3 ─────────────────────────────╮
┃        l_out-0                       ┃│ head 0/13  rows 0-7 of 17  cols 0-33 of 256  contrast 1.00     │
┃    + layers.1                     97%┃│      The ██                                                    │
┃    + layers.2                     95%┃│  capital ██                                                    │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛│       of ██                                                    │
╭ 4 RUNTIME METRICS ───────────────────╮│   France ██                                                    │
│ compute split                        ││       is ██                                                    │
│ attn  36.2% ███████───────────        │╰────────────────────────────────────────────────────────────────╯
│ mlp   34.8% ██████────────────        │╭ 5 ANOMALY LEDGER ──────────────────────────────────────────────╮
│ norm  15.3% ███───────────────        ││ 00:03.511 ! kq-0 |max| 1.24e+04 exceeds 1e+04                   │
╰──────────────────────────────────────╯╰────────────────────────────────────────────────────────────────╯
```

*(Generate your own with `llmscope --demo --snapshot`.)*

---

## How the hook works

The entire non-invasive mechanism is one field on `llama_context_params`:

```cpp
params.cb_eval           = &Tracer::cb_eval_thunk;
params.cb_eval_user_data = this;
```

ggml's scheduler then calls that function **twice for every node** in the compute graph — once to ask whether
we want the node's data, once after computing it. Returning `true` on the ask makes the scheduler evaluate that
node as a graph view of exactly one node and call `ggml_backend_synchronize` before handing it over, which is
what makes per-node latency measurable at all.

llama.cpp is consumed as an ordinary library. Nothing in its tree is patched, and it is pinned to an exact commit.

### The honest caveat

Always answering `true` **defeats ggml's node batching**. Traced inference is therefore slower than untraced
inference, and the latencies reported are *instrumented* latencies. They are accurate relative to each other and
correctly identify which block dominates compute — which is the question the tool exists to answer — but they are
not free-running wall-clock timings. This is inherent to per-node observation, not an implementation shortcut.

Two other constraints worth stating plainly:

- **Flash attention must be off.** With FA on, the softmax is fused into one kernel and the attention matrix
  never materialises as a tensor. llmscope sets `LLAMA_FLASH_ATTN_TYPE_DISABLED` — required, not preferred.
- **Statistics are sampled.** Visiting every element of every activation costs more than the inference. Tensors
  above the sample budget (default 8192 elements) are strided; the UI marks those values with `~`. `mean` and
  `sparsity` are excellent estimates this way; `|max|` can be understated.

---

## Quick start

```powershell
# 1. Restore dependencies, a portable compiler toolchain, and a sample model.
#    Requires no administrator rights.
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

# 2. Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. Look around with no model at all
.\build\bin\llmscope.exe --demo

# 4. Trace a real model
.\build\bin\llmscope.exe --model models\qwen2.5-0.5b-instruct-q8_0.gguf -p "The capital of France is" -n 32

# 5. Record, then replay offline
.\build\bin\llmscope.exe --model models\qwen2.5-0.5b-instruct-q8_0.gguf --headless --record traces\run.trace
.\build\bin\llmscope.exe --replay traces\run.trace
```

`--demo` needs no GGUF, no GPU, and no inference — it drives the whole UI from a synthetic model. That is also
how the TUI is developed and tested.

---

## Keys

| Key | Action |
|---|---|
| `Tab` / `Shift-Tab` | cycle pane focus |
| `1`–`5` | jump straight to a pane |
| `j` / `k`, `↓` / `↑` | move within the focused pane |
| `h` / `l` | collapse / expand a tree row |
| `Space` / `Enter` | toggle a row (or play/pause during replay) |
| `g` / `G` | jump to top / resume following the tail |
| `q` / `Esc` | quit |

Attention pane: `H J K L` pan, `+` / `-` contrast, `n` / `p` change head, `f` fullscreen.
Replay: `Space` play/pause, `[` / `]` speed, `r` restart.

---

## Architecture

Four modules with a strict dependency rule: **`tui/` never includes a llama.cpp header, and `trace_core/`
includes neither llama.cpp nor FTXUI.**

```
app/          main(), CLI, thread orchestration
 ├── tracer/  cb_eval hook, stats kernels, topology parsing   [llama.cpp + ggml]
 ├── tui/     five panes, focus, vim keys, attention heatmap  [FTXUI]
 └── trace_core/   records, SPSC ring, payload slot, trace files, sources   [no deps]
```

`TraceSource` is the seam that makes all of this work. Three implementations satisfy it:

| Implementation | Purpose |
|---|---|
| `LiveSource` | drains the lock-free ring fed by the running model |
| `TraceFileSource` | replays a recorded `.trace` file, with pause / speed / restart |
| `FakeSource` | synthesises a plausible model so the TUI needs no inference to develop or test |

Because the TUI only ever talks to a `TraceSource`, it can be built to completion before the tracer exists —
which is also what lets two people work in parallel.

### Concurrency

One producer (the inference thread) and one consumer (the UI thread), with three tiers of cost:

- **Hot path**, ~300× per token: `SpscRing::push` plus two relaxed atomic loads. No locks.
- **Warm path**, ~1× per token: publishing an attention matrix through a double-buffered `PayloadSlot`.
- **Cold path**, once at startup: interning node names and model metadata, behind a mutex.

The ring **drops incoming records when full rather than overwriting the oldest**. Overwriting would require the
producer to advance `tail_`, giving that variable two writers; a lost update there can hand the consumer a slot
being rewritten underneath it. Keeping `tail_` single-writer is what makes the ring correct rather than merely
lock-free. Drops are counted and displayed rather than hidden.

### How the tree is built

ggml hands over ~300 unrelated tensor names per forward pass. llama.cpp names them consistently
(`attn_norm-3`, `ffn_up-12`, `kq_soft_max-0`, `l_out-9`, `cache_k_l5`), so the transformer structure is
recovered by parsing — no model-specific knowledge and no llama.cpp changes. The tree is discovered once on the
first forward pass and is stable thereafter, because the graph is identical on every token.

Names are interned to `uint16` ids on first sight, which keeps `NodeRecord` a fixed-size 88-byte POD. That is
what allows the same struct to be pushed through a lock-free ring *and* `memcpy`'d straight into a trace file.

---

## Trace file format

A flat stream of tagged chunks rather than a header-plus-body layout, because node names are discovered *during*
the run, not before it. A chunk stream also degrades gracefully: kill the process mid-trace and everything
written so far still replays.

```
u64 magic "LLMSCOPE" | u32 version | u32 reserved
then repeating: u8 tag, payload
   1 Model    strings + dimensions
   2 Name     name_id, kind, layer, string      (must precede records that use it)
   3 Record   NodeRecord, 88 bytes verbatim
   4 Anomaly  AnomalyRecord
   5 Token    index, id, is_prompt, piece
   6 Payload  PayloadHeader + n_floats × f32
```

---

## Testing

```powershell
.\build\bin\llmscope_tests.exe      # 99 checks, no framework
```

`trace_core` has no dependencies and its tests keep it that way. They cover the places where a bug would be
silent rather than loud: ring-buffer drop accounting and ordering under real thread contention, node-name
classification (`attn_norm` must classify as a norm, not attention), tree construction, payload clamping, and
trace-file round-tripping including rejection of corrupt files.

The UI is verified with `--snapshot`, which renders one frame off-screen to stdout and needs no TTY — usable in
CI, and how the screenshot above was produced.

Two workflows, split by what they cost:

| Workflow | Runs | Covers |
|---|---|---|
| `ci.yml` | every push | `-DLLMSCOPE_WITH_LLAMA=OFF` on Linux (GCC, Clang) and Windows (MinGW): unit tests, a rendered frame, a synthetic record/replay round-trip, corrupt-file rejection |
| `tracer.yml` | tracer changes, weekly, on demand | `-DLLMSCOPE_WITH_LLAMA=ON`: builds llama.cpp, traces a real GGUF through `cb_eval`, and round-trips the result |

Dropping the llama.cpp backend is what keeps per-push CI free of a large dependency and a 675 MB download — the
dependency rule is what makes that possible. But it also means `src/tracer/` is never compiled there, so the
hook would rot unnoticed; `tracer.yml` exists to cover exactly that, without slowing down every push. Both read
their pinned dependency commits out of `scripts/bootstrap.ps1`, so CI and the documented local setup cannot
drift apart.

The Linux jobs are what actually test the "portable C++17" claim, since Windows is the platform development
already happens on.

---

## CLI

```
SOURCE
  -m, --model <path>        GGUF model to load and trace
      --replay <path>       replay a previously recorded trace
      --demo                synthetic model; no GGUF or inference required

INFERENCE
  -p, --prompt <text>       prompt to run
  -n, --n-predict <int>     tokens to generate (default 96)
  -c, --ctx <int>           context size (default 2048)
  -t, --threads <int>       inference threads (default: all cores)
      --gpu-layers <int>    layers to offload to GPU (default 0, CPU only)

TRACING
      --record <path>       also write a .trace file
      --sample-budget <n>   elements sampled per tensor for stats (default 8192)
      --capture-layer <n>   block whose attention scores to snapshot (default 0)
      --anomaly-max <f>     |activation| ceiling for the ledger (default 10000)
      --no-anomalies        disable anomaly detection
      --headless            no TUI; drain the source, trace to file, print a summary
                            (works with --model, --demo and --replay)

DISPLAY
      --ascii               ASCII-only glyphs
      --fps <int>           UI refresh rate (default 20)
      --snapshot            render one frame to stdout and exit
```

---

## Anomaly thresholds

Defaults were tuned against a real Qwen2.5-0.5B trace rather than picked from intuition. An earlier
`|max| > 100` ceiling produced 174 findings in 747 records — large language models genuinely produce
activations in the thousands ("outlier features" are a documented property, not a fault), so that threshold
taught the user to ignore the pane. The current ceiling of `1e4` leaves headroom below fp16's 65504 limit, which
is the value that actually matters for clipping risk.

Each `(node, rule)` pair also reports at most once per token. A genuinely stuck node still appears every token;
a merely noisy one no longer floods the ledger.

---

## Requirements

- Windows (developed on Windows 11); the code itself is portable C++17
- No administrator rights needed — `bootstrap.ps1` fetches a portable GCC, CMake, and Ninja
- No GPU required; the CPU backend is the default and gives more honest per-node timings
- ~2 GB disk for dependencies and a sample model

Built and verified with GCC 16.2.0, CMake 4.4.2, Ninja 1.13.2, llama.cpp `7b13a84`, FTXUI 7.0.3.

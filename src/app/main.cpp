// main.cpp - CLI parsing and thread orchestration.
//
// Three modes, all ending in the same TUI:
//   --model  <gguf>   run llama.cpp under the tracer (optionally --record)
//   --replay <trace>  play back a recorded trace
//   --demo            synthetic model, no inference, no model file
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include "trace_core/fake_source.hpp"
#include "trace_core/live_source.hpp"
#include "trace_core/trace_file.hpp"
#include "tracer/tracer.hpp"  // declarations only; pulls in no llama.cpp headers
#include "tui/app.hpp"

namespace {

void print_usage() {
    std::printf(
        "llmscope - live instrumentation, tracing and replay for local transformers\n"
        "\n"
        "USAGE\n"
        "  llmscope --model <model.gguf> [options]\n"
        "  llmscope --replay <trace.llmscope>\n"
        "  llmscope --demo\n"
        "\n"
        "SOURCE\n"
        "  -m, --model <path>     GGUF model to load and trace\n"
        "      --replay <path>    replay a previously recorded trace\n"
        "      --demo             synthetic model; no GGUF or inference required\n"
        "\n"
        "INFERENCE\n"
        "  -p, --prompt <text>    prompt to run (default: \"The capital of France is\")\n"
        "  -n, --n-predict <int>  tokens to generate (default 96)\n"
        "  -c, --ctx <int>        context size (default 2048)\n"
        "  -t, --threads <int>    inference threads (default: all cores)\n"
        "      --gpu-layers <int> layers to offload to GPU (default 0, CPU only)\n"
        "\n"
        "TRACING\n"
        "      --record <path>    also write a .llmscope trace file\n"
        "      --sample-budget <n> elements sampled per tensor for stats (default 8192)\n"
        "      --capture-layer <n> block whose attention scores to snapshot (default 0)\n"
        "      --anomaly-max <f>  |activation| ceiling for the ledger (default 10000)\n"
        "      --no-anomalies     disable anomaly detection entirely\n"
        "      --headless         no TUI; trace to file and print a summary\n"
        "\n"
        "DISPLAY\n"
        "      --ascii            ASCII-only glyphs for terminals without block chars\n"
        "      --fps <int>        UI refresh rate (default 20)\n"
        "      --snapshot         render one frame to stdout and exit (no TTY needed)\n"
        "      --snapshot-size <WxH>  snapshot dimensions (default 160x48)\n"
        "      --snapshot-warmup <ms> collect for this long first (default 1500)\n"
        "  -h, --help             this message\n"
        "\n"
        "KEYS\n"
        "  Tab/1-5 focus pane   j/k move   h/l collapse/expand   Space toggle\n"
        "  Attention pane: HJKL pan, +/- contrast, n/p head, f fullscreen\n"
        "  Replay: Space play/pause, [/] speed, r restart        q quit\n");
}

bool next_arg(int argc, char** argv, int& i, const char* flag, std::string& out) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "llmscope: %s requires a value\n", flag);
        return false;
    }
    out = argv[++i];
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    llmscope::TracerConfig cfg;
    llmscope::UiOptions ui;

    std::string model_path;
    std::string replay_path;
    bool demo = false;
    bool headless = false;
    bool snapshot = false;
    int snap_w = 160;
    int snap_h = 48;
    int snap_warmup_ms = 1500;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        std::string v;

        if (a == "-h" || a == "--help") {
            print_usage();
            return 0;
        } else if (a == "-m" || a == "--model") {
            if (!next_arg(argc, argv, i, "--model", model_path)) return 2;
        } else if (a == "--replay") {
            if (!next_arg(argc, argv, i, "--replay", replay_path)) return 2;
        } else if (a == "--demo") {
            demo = true;
        } else if (a == "-p" || a == "--prompt") {
            if (!next_arg(argc, argv, i, "--prompt", cfg.prompt)) return 2;
        } else if (a == "-n" || a == "--n-predict") {
            if (!next_arg(argc, argv, i, "--n-predict", v)) return 2;
            cfg.n_predict = std::atoi(v.c_str());
        } else if (a == "-c" || a == "--ctx") {
            if (!next_arg(argc, argv, i, "--ctx", v)) return 2;
            cfg.n_ctx = std::atoi(v.c_str());
        } else if (a == "-t" || a == "--threads") {
            if (!next_arg(argc, argv, i, "--threads", v)) return 2;
            cfg.n_threads = std::atoi(v.c_str());
        } else if (a == "--gpu-layers") {
            if (!next_arg(argc, argv, i, "--gpu-layers", v)) return 2;
            cfg.n_gpu_layers = std::atoi(v.c_str());
        } else if (a == "--record") {
            if (!next_arg(argc, argv, i, "--record", cfg.trace_out)) return 2;
        } else if (a == "--sample-budget") {
            if (!next_arg(argc, argv, i, "--sample-budget", v)) return 2;
            cfg.sample_budget = static_cast<uint64_t>(std::strtoull(v.c_str(), nullptr, 10));
        } else if (a == "--capture-layer") {
            if (!next_arg(argc, argv, i, "--capture-layer", v)) return 2;
            cfg.capture_layer = std::atoi(v.c_str());
        } else if (a == "--anomaly-max") {
            if (!next_arg(argc, argv, i, "--anomaly-max", v)) return 2;
            cfg.anomaly.magnitude_ceiling = static_cast<float>(std::atof(v.c_str()));
        } else if (a == "--no-anomalies") {
            cfg.anomaly.enabled = false;
        } else if (a == "--headless") {
            headless = true;
        } else if (a == "--ascii") {
            ui.ascii_only = true;
        } else if (a == "--fps") {
            if (!next_arg(argc, argv, i, "--fps", v)) return 2;
            ui.fps = std::atoi(v.c_str());
        } else if (a == "--snapshot") {
            snapshot = true;
        } else if (a == "--snapshot-size") {
            if (!next_arg(argc, argv, i, "--snapshot-size", v)) return 2;
            const size_t x = v.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr, "llmscope: --snapshot-size expects WxH, e.g. 160x48\n");
                return 2;
            }
            snap_w = std::atoi(v.substr(0, x).c_str());
            snap_h = std::atoi(v.substr(x + 1).c_str());
            if (snap_w < 40 || snap_h < 16) {
                std::fprintf(stderr, "llmscope: --snapshot-size must be at least 40x16\n");
                return 2;
            }
        } else if (a == "--snapshot-warmup") {
            if (!next_arg(argc, argv, i, "--snapshot-warmup", v)) return 2;
            snap_warmup_ms = std::atoi(v.c_str());
        } else {
            std::fprintf(stderr, "llmscope: unknown argument '%s' (try --help)\n", a.c_str());
            return 2;
        }
    }

    const int modes = (model_path.empty() ? 0 : 1) + (replay_path.empty() ? 0 : 1) + (demo ? 1 : 0);
    if (modes == 0) {
        std::fprintf(stderr, "llmscope: need one of --model, --replay or --demo\n\n");
        print_usage();
        return 2;
    }
    if (modes > 1) {
        std::fprintf(stderr, "llmscope: --model, --replay and --demo are mutually exclusive\n");
        return 2;
    }

    // ---------------- replay ----------------
    if (!replay_path.empty()) {
        std::string err;
        auto src = llmscope::TraceFileSource::load(replay_path, err);
        if (!src) {
            std::fprintf(stderr, "llmscope: %s\n", err.c_str());
            return 1;
        }
        llmscope::TuiApp app(*src, ui);
        if (snapshot) {
            std::fputs(app.snapshot(snap_w, snap_h, snap_warmup_ms).c_str(), stdout);
            return 0;
        }
        return app.run();
    }

    // ---------------- demo ----------------
    if (demo) {
        llmscope::FakeSource src;
        llmscope::TuiApp app(src, ui);
        if (snapshot) {
            std::fputs(app.snapshot(snap_w, snap_h, snap_warmup_ms).c_str(), stdout);
            return 0;
        }
        return app.run();
    }

    // ---------------- live tracing ----------------
#if !LLMSCOPE_WITH_LLAMA
    std::fprintf(stderr,
                 "llmscope: built without the llama.cpp backend; --model is unavailable.\n"
                 "          Rebuild with -DLLMSCOPE_WITH_LLAMA=ON, or use --demo/--replay.\n");
    return 1;
#else
    cfg.model_path = model_path;

    llmscope::LiveSource sink;
    llmscope::Tracer tracer(sink, cfg);

    std::string tracer_error;
    bool tracer_ok = true;

    std::thread worker([&] {
        tracer_ok = tracer.run(tracer_error);
        sink.set_running(false);
    });

    int rc = 0;
    if (headless) {
        worker.join();
        if (!tracer_ok) {
            std::fprintf(stderr, "llmscope: %s\n", tracer_error.c_str());
            return 1;
        }
        const llmscope::SourceStats s = sink.stats();
        std::printf("llmscope: captured %llu nodes across %llu tokens (%llu dropped)\n",
                    static_cast<unsigned long long>(s.records_total),
                    static_cast<unsigned long long>(s.tokens_done),
                    static_cast<unsigned long long>(s.records_dropped));
        if (!cfg.trace_out.empty()) {
            std::printf("llmscope: trace written to %s\n", cfg.trace_out.c_str());
        }
        return 0;
    }

    {
        llmscope::TuiApp app(sink, ui);
        if (snapshot) {
            std::fputs(app.snapshot(snap_w, snap_h, snap_warmup_ms).c_str(), stdout);
        } else {
            rc = app.run();
        }
    }

    // The TUI has exited; tell the tracer to stop at the next graph node and
    // wait for it, so llama.cpp tears down cleanly.
    sink.request_stop();
    worker.join();

    if (!tracer_ok) {
        std::fprintf(stderr, "llmscope: %s\n", tracer_error.c_str());
        return 1;
    }
    return rc;
#endif  // LLMSCOPE_WITH_LLAMA
}

#include "app.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/node.hpp"
#include "ftxui/screen/color.hpp"
#include "ftxui/screen/screen.hpp"

namespace llmscope {

using namespace ftxui;

namespace {

// --- palette -------------------------------------------------------------
// Chosen to stay legible on both dark and light terminals: mid-luminance hues
// rather than pure primaries.
const Color kAccent    = Color::RGB(122, 190, 255);
const Color kAccentDim = Color::RGB(70, 110, 150);
const Color kOk        = Color::RGB(126, 200, 140);
const Color kWarn      = Color::RGB(230, 190, 110);
const Color kErr       = Color::RGB(235, 120, 120);
const Color kMuted     = Color::RGB(130, 130, 140);
const Color kHeatLow   = Color::RGB(20, 30, 60);
const Color kHeatHigh  = Color::RGB(255, 220, 120);

std::string fmt_ns(uint64_t ns) {
    char buf[32];
    if (ns < 1000) {
        std::snprintf(buf, sizeof(buf), "%llu ns", static_cast<unsigned long long>(ns));
    } else if (ns < 1000000) {
        std::snprintf(buf, sizeof(buf), "%.2f us", static_cast<double>(ns) / 1e3);
    } else if (ns < 1000000000ull) {
        std::snprintf(buf, sizeof(buf), "%.3f ms", static_cast<double>(ns) / 1e6);
    } else {
        std::snprintf(buf, sizeof(buf), "%.3f s", static_cast<double>(ns) / 1e9);
    }
    return buf;
}

std::string fmt_time(uint64_t ns) {
    const uint64_t total_ms = ns / 1000000ull;
    const uint64_t ms = total_ms % 1000;
    const uint64_t s = (total_ms / 1000) % 60;
    const uint64_t m = (total_ms / 60000) % 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02llu:%02llu.%03llu", static_cast<unsigned long long>(m),
                  static_cast<unsigned long long>(s), static_cast<unsigned long long>(ms));
    return buf;
}

std::string fmt_shape(const NodeRecord& r) {
    char buf[64];
    int used = std::snprintf(buf, sizeof(buf), "[%lld", static_cast<long long>(r.ne[0]));
    for (int i = 1; i < 4; ++i) {
        if (r.ne[i] <= 1 && i >= 2) break;
        used += std::snprintf(buf + used, sizeof(buf) - static_cast<size_t>(used), ", %lld",
                              static_cast<long long>(r.ne[i]));
    }
    std::snprintf(buf + used, sizeof(buf) - static_cast<size_t>(used), "]");
    return buf;
}

std::string fmt_float(float v) {
    char buf[32];
    const float a = std::fabs(v);
    if (a != 0.0f && (a < 1e-3f || a >= 1e5f)) {
        std::snprintf(buf, sizeof(buf), "%.3e", static_cast<double>(v));
    } else {
        std::snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(v));
    }
    return buf;
}

std::string dtype_name(uint8_t dtype) {
    // Mirrors the first entries of ggml_type; kept local so trace_core and the
    // TUI never need to link ggml.
    switch (dtype) {
        case 0:  return "f32";
        case 1:  return "f16";
        case 2:  return "q4_0";
        case 3:  return "q4_1";
        case 6:  return "q5_0";
        case 7:  return "q5_1";
        case 8:  return "q8_0";
        case 9:  return "q8_1";
        case 12: return "q4_K";
        case 13: return "q5_K";
        case 14: return "q6_K";
        case 15: return "q8_K";
        case 24: return "i8";
        case 25: return "i16";
        case 26: return "i32";
        case 27: return "i64";
        case 30: return "bf16";
        default: return "t" + std::to_string(static_cast<int>(dtype));
    }
}

Color kind_color(NodeKind k) {
    switch (k) {
        case NodeKind::AttnScores: return Color::RGB(255, 180, 120);
        case NodeKind::Attention:  return Color::RGB(230, 150, 190);
        case NodeKind::Ffn:        return Color::RGB(150, 200, 255);
        case NodeKind::Norm:       return Color::RGB(160, 220, 200);
        case NodeKind::Embedding:  return Color::RGB(200, 180, 255);
        case NodeKind::LayerOut:   return Color::RGB(180, 200, 160);
        case NodeKind::Output:     return Color::RGB(255, 210, 140);
        case NodeKind::Cache:      return Color::RGB(150, 170, 190);
        case NodeKind::Rope:       return Color::RGB(190, 190, 140);
        default:                   return kMuted;
    }
}

// A five-step intensity ramp. Unicode blocks look far better, but plenty of
// Windows terminals still render them as boxes, hence the ASCII fallback.
const char* ramp_cell(float t, bool ascii_only) {
    static const char* kUnicode[] = {"  ", "░░", "▒▒", "▓▓", "██"};
    static const char* kAscii[]   = {"  ", "..", "::", "**", "##"};
    int idx = static_cast<int>(t * 5.0f);
    if (idx < 0) idx = 0;
    if (idx > 4) idx = 4;
    return ascii_only ? kAscii[idx] : kUnicode[idx];
}

// Horizontal meter built from text, so it colours per segment (a plain gauge()
// cannot show a warning threshold).
Element meter(float fraction, int width, const Color& fill, bool ascii_only) {
    if (fraction < 0.0f) fraction = 0.0f;
    if (fraction > 1.0f) fraction = 1.0f;
    const int filled = static_cast<int>(fraction * static_cast<float>(width) + 0.5f);
    std::string on;
    std::string off;
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            on += ascii_only ? "#" : "█";
        } else {
            off += ascii_only ? "-" : "─";
        }
    }
    return hbox({text(on) | color(fill), text(off) | color(kAccentDim)});
}

}  // namespace

// ---------------------------------------------------------------------------

TuiApp::TuiApp(TraceSource& source, UiOptions opts)
    : source_(source), state_(source), opts_(opts) {}

int TuiApp::selected_topo_row() const {
    const std::vector<int> rows = state_.topo().visible_rows();
    if (rows.empty()) {
        return -1;
    }
    int c = topo_cursor_;
    if (c < 0) c = 0;
    if (c >= static_cast<int>(rows.size())) c = static_cast<int>(rows.size()) - 1;
    return rows[static_cast<size_t>(c)];
}

int TuiApp::selected_layer() const {
    const int row = selected_topo_row();
    if (row < 0) {
        return -1;
    }
    return state_.topo().node(row).layer;
}

uint16_t TuiApp::selected_name_id() const {
    const int row = selected_topo_row();
    if (row < 0) {
        return kInvalidNameId;
    }
    const TopoNode& n = state_.topo().node(row);
    if (!n.name_ids.empty()) {
        return n.name_ids.front();
    }
    return kInvalidNameId;
}

void TuiApp::sync_capture_target() {
    const uint16_t name_id = selected_name_id();
    int32_t layer = selected_layer();
    if (layer < 0) {
        layer = 0;  // default to block 0 so the heatmap always has a subject
    }
    if (name_id != last_capture_name_ || layer != last_capture_layer_) {
        source_.set_capture_target(name_id, layer);
        last_capture_name_ = name_id;
        last_capture_layer_ = layer;
    }
    if (attn_head_ != last_capture_head_) {
        source_.set_attention_head(attn_head_);
        last_capture_head_ = attn_head_;
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

Element TuiApp::pane_frame(Pane pane, const std::string& title, Element content) {
    const bool focused = focus_ == pane;
    const std::string label = focused ? " " + title + "  [focus] " : " " + title + " ";
    Element titled = text(label);
    titled = focused ? (titled | bold | color(kAccent)) : (titled | color(kMuted));
    Element win = window(titled, std::move(content), focused ? HEAVY : ROUNDED);
    return focused ? (win | color(kAccent)) : (win | color(kAccentDim));
}

Element TuiApp::render_header() {
    const ModelInfo& m = state_.model();
    const SourceStats& s = state_.stats();

    char rate[64];
    std::snprintf(rate, sizeof(rate), "%.2f tok/s", s.tokens_per_sec);

    Element drops =
        s.records_dropped > 0
            ? text(" dropped " + std::to_string(s.records_dropped) + " ") | color(kErr) | bold
            : text(" no drops ") | color(kOk);

    return hbox({
               text(" llmscope ") | bold | color(Color::Black) | bgcolor(kAccent),
               text(" " + m.name + " ") | bold,
               text("L" + std::to_string(m.n_layer) + " H" + std::to_string(m.n_head) + " D" +
                    std::to_string(m.n_embd) + " ") |
                   color(kMuted),
               text("| " + std::string(to_string(m.backend)) + " ") | color(kOk),
               text("| " + std::string(rate) + " ") | color(kAccent),
               text("| nodes " + std::to_string(s.records_total) + " ") | color(kMuted),
               drops,
               filler(),
               text(source_.status_line() + " ") | color(kWarn),
           }) |
           bgcolor(Color::RGB(18, 20, 28));
}

Element TuiApp::render_topology() {
    const Topology& topo = state_.topo();
    const std::vector<int> rows = topo.visible_rows();

    Elements lines;
    if (rows.size() <= 1) {
        lines.push_back(text(" waiting for first forward pass...") | color(kMuted));
    }

    // Keep the cursor inside a window that follows it.
    const int view_height = 14;
    int start = topo_cursor_ - view_height / 2;
    if (start < 0) start = 0;
    if (start + view_height > static_cast<int>(rows.size())) {
        start = static_cast<int>(rows.size()) - view_height;
    }
    if (start < 0) start = 0;

    for (int i = start; i < static_cast<int>(rows.size()) && i < start + view_height; ++i) {
        const TopoNode& n = topo.node(rows[static_cast<size_t>(i)]);
        const bool is_cursor = (i == topo_cursor_);

        std::string prefix(static_cast<size_t>(n.depth) * 2, ' ');
        if (n.is_group && !n.children.empty()) {
            prefix += n.expanded ? (opts_.ascii_only ? "- " : "▼ ") : (opts_.ascii_only ? "+ " : "▶ ");
        } else {
            prefix += opts_.ascii_only ? "  " : "● ";
        }

        // Show each block's share of total compute right in the tree - this is
        // the "which block dominates" answer, without leaving the pane.
        std::string suffix;
        if (n.layer >= 0 && n.is_group && n.depth == 2) {
            const auto& per = state_.per_layer();
            if (static_cast<size_t>(n.layer) < per.size() && state_.heaviest_layer_ns() > 0) {
                const double share = static_cast<double>(per[static_cast<size_t>(n.layer)].total_ns) /
                                     static_cast<double>(state_.heaviest_layer_ns());
                char b[32];
                std::snprintf(b, sizeof(b), " %3.0f%%", share * 100.0);
                suffix = b;
            }
        }

        Element line = hbox({
            text(prefix) | color(kAccentDim),
            text(n.label) | color(n.is_group ? Color::White : kind_color(n.kind)),
            filler(),
            text(suffix) | color(kWarn),
        });
        if (is_cursor) {
            line = line | inverted;
        }
        lines.push_back(line);
    }

    return pane_frame(Pane::Topology, "1 MODEL TOPOLOGY", vbox(std::move(lines)) | flex);
}

Element TuiApp::render_stream() {
    const auto& stream = state_.stream();

    Elements lines;
    lines.push_back(hbox({
                        text("  SEQ  ") | bold,
                        text("TIME       ") | bold,
                        text("LAYER ") | bold,
                        text("TYPE          ") | bold,
                        text("SHAPE            ") | bold,
                        text("LATENCY   ") | bold,
                        text("DEV") | bold,
                    }) |
                    color(kMuted));

    const int view_height = 12;
    const int total = static_cast<int>(stream.size());
    int start = total - view_height;
    if (stream_cursor_ >= 0) {
        start = stream_cursor_ - view_height / 2;
    }
    if (start < 0) start = 0;

    for (int i = start; i < total && i < start + view_height; ++i) {
        const NodeRecord& r = stream[static_cast<size_t>(i)];
        const auto kind = static_cast<NodeKind>(r.kind);

        char seq[16];
        std::snprintf(seq, sizeof(seq), "%6llu", static_cast<unsigned long long>(r.seq));
        char layer[8];
        if (r.layer >= 0) {
            std::snprintf(layer, sizeof(layer), "%5d", r.layer);
        } else {
            std::snprintf(layer, sizeof(layer), "    -");
        }

        // 14 wide: "Attn (Scores)" is 13 characters, so 13 would leave no gap.
        std::string type_str = to_string(kind);
        type_str.resize(14, ' ');
        std::string shape = fmt_shape(r);
        shape.resize(17, ' ');
        std::string lat = fmt_ns(r.dur_ns);
        lat.resize(10, ' ');

        Element line = hbox({
            text(std::string(seq) + " ") | color(kMuted),
            text(fmt_time(r.t_start_ns) + "  ") | color(kMuted),
            text(std::string(layer) + " ") | color(kAccent),
            text(type_str) | color(kind_color(kind)),
            text(shape) | color(kMuted),
            text(lat) | color((r.flags & kFlagHasNaN) ? kErr : kOk),
            text(to_string(static_cast<BackendId>(r.backend))) | color(kMuted),
        });
        if (stream_cursor_ == i) {
            line = line | inverted;
        }
        lines.push_back(line);
    }

    if (stream.empty()) {
        lines.push_back(text(" no nodes captured yet") | color(kMuted));
    }

    const std::string title = std::string("2 LIVE PACKET STREAM") +
                              (stream_cursor_ < 0 ? "  (following)" : "  (paused)");
    return pane_frame(Pane::Stream, title, vbox(std::move(lines)) | flex);
}

Element TuiApp::render_attention() {
    if (!state_.has_attention()) {
        return pane_frame(Pane::Attention, "3 ATTENTION MATRIX",
                          vbox({
                              text(" no attention scores captured yet") | color(kMuted),
                              text(" select a transformer block in pane 1 (j/k), then wait for the "
                                   "next token") |
                                  color(kMuted),
                          }) |
                              flex);
    }

    const std::vector<float>& a = state_.attention();
    const int rows = state_.attention_rows();
    const int cols = state_.attention_cols();

    // Viewport size in matrix cells. Each cell is two characters wide.
    const int view_rows = fullscreen_attention_ ? 28 : 8;
    const int view_cols = fullscreen_attention_ ? 60 : 34;

    int r0 = std::min(attn_row_offset_, std::max(0, rows - 1));
    int c0 = std::min(attn_col_offset_, std::max(0, cols - 1));

    // Normalise against the visible window so panning never washes out.
    float vmax = 0.0f;
    for (int r = r0; r < std::min(rows, r0 + view_rows); ++r) {
        for (int c = c0; c < std::min(cols, c0 + view_cols); ++c) {
            vmax = std::max(vmax, a[static_cast<size_t>(r) * cols + c]);
        }
    }
    if (vmax <= 0.0f) vmax = 1.0f;

    const auto& toks = state_.tokens();

    Elements grid;
    for (int r = r0; r < std::min(rows, r0 + view_rows); ++r) {
        Elements row_cells;

        // Row label: the query token this row belongs to. Rows accumulate from
        // attention_first_token(), not necessarily from token 0.
        std::string label;
        const size_t tok_idx = static_cast<size_t>(state_.attention_first_token()) +
                               static_cast<size_t>(r);
        if (tok_idx < toks.size()) {
            label = toks[tok_idx].piece;
        }
        if (label.empty()) label = std::to_string(r);
        if (label.size() > 8) label = label.substr(0, 8);
        label.insert(label.begin(), 9 - static_cast<int>(label.size()), ' ');
        row_cells.push_back(text(label + " ") | color(kMuted));

        for (int c = c0; c < std::min(cols, c0 + view_cols); ++c) {
            const float raw = a[static_cast<size_t>(r) * cols + c] / vmax;
            // Gamma controls contrast; low weights dominate a softmax row, so
            // the default curve lifts them into visibility.
            const float t = std::pow(std::max(0.0f, raw), 1.0f / attn_contrast_);
            row_cells.push_back(text(ramp_cell(t, opts_.ascii_only)) |
                                color(Color::Interpolate(t, kHeatLow, kHeatHigh)));
        }
        grid.push_back(hbox(std::move(row_cells)));
    }

    char info[160];
    std::snprintf(info, sizeof(info),
                  " head %d/%d   rows %d-%d of %d   cols %d-%d of %d   contrast %.2f   max %.4f",
                  attn_head_, std::max(1, state_.model().n_head) - 1, r0,
                  std::min(rows, r0 + view_rows) - 1, rows, c0,
                  std::min(cols, c0 + view_cols) - 1, cols, attn_contrast_, vmax);

    Elements body;
    body.push_back(text(info) | color(kMuted));
    body.push_back(separator());
    for (auto& g : grid) {
        body.push_back(std::move(g));
    }

    const std::string title = "3 ATTENTION MATRIX  " + state_.name_of(state_.attention_name_id()) +
                              (fullscreen_attention_ ? "  [fullscreen]" : "");
    return pane_frame(Pane::Attention, title, vbox(std::move(body)) | flex);
}

Element TuiApp::render_inspector() {
    const int row = selected_topo_row();
    Elements body;

    if (row < 0) {
        body.push_back(text(" nothing selected") | color(kMuted));
        return pane_frame(Pane::Inspector, "4 RUNTIME METRICS", vbox(std::move(body)) | flex);
    }

    const TopoNode& n = state_.topo().node(row);
    const std::vector<uint16_t> ids = state_.topo().collect_name_ids(row);

    body.push_back(hbox({text(" target  ") | color(kMuted),
                         text(n.label) | bold | color(kind_color(n.kind))}));

    if (n.name_ids.empty()) {
        // A group row: summarise everything underneath it.
        const LayerAggregate agg = state_.aggregate_for_rows(ids);
        body.push_back(hbox({text(" nodes   ") | color(kMuted),
                             text(std::to_string(ids.size()))}));
        body.push_back(hbox({text(" latency ") | color(kMuted),
                             text(fmt_ns(agg.total_ns)) | color(kOk),
                             text("  (last pass, summed)") | color(kMuted)}));

        const uint64_t tot = agg.total_ns > 0 ? agg.total_ns : 1;
        body.push_back(separator());
        body.push_back(text(" compute split") | color(kMuted));
        auto split = [&](const char* label, uint64_t v, Color c) {
            char pct[16];
            std::snprintf(pct, sizeof(pct), "%5.1f%%",
                          100.0 * static_cast<double>(v) / static_cast<double>(tot));
            return hbox({text(std::string(" ") + label) | color(kMuted),
                         text(pct) | color(c), text(" "),
                         meter(static_cast<float>(v) / static_cast<float>(tot), 18, c,
                               opts_.ascii_only)});
        };
        body.push_back(split("attn ", agg.attn_ns, Color::RGB(230, 150, 190)));
        body.push_back(split("mlp  ", agg.ffn_ns, Color::RGB(150, 200, 255)));
        body.push_back(split("norm ", agg.norm_ns, Color::RGB(160, 220, 200)));
        body.push_back(split("other", agg.other_ns, kMuted));
    } else {
        const NodeRecord* r = state_.last_record(n.name_ids.front());
        if (r == nullptr) {
            body.push_back(text(" no record yet for this node") | color(kMuted));
        } else {
            body.push_back(hbox({text(" shape   ") | color(kMuted), text(fmt_shape(*r)),
                                 text("  dtype ") | color(kMuted),
                                 text(dtype_name(r->dtype)) | color(kAccent)}));
            body.push_back(hbox({text(" latency ") | color(kMuted),
                                 text(fmt_ns(r->dur_ns)) | color(kOk),
                                 text("   device ") | color(kMuted),
                                 text(to_string(static_cast<BackendId>(r->backend)))}));
            body.push_back(hbox({text(" mean    ") | color(kMuted), text(fmt_float(r->mean)),
                                 text("   |max| ") | color(kMuted),
                                 text(fmt_float(r->absmax)),
                                 text((r->flags & kFlagStatsSampled) ? " ~" : "  ") |
                                     color(kWarn)}));
            body.push_back(hbox({text(" min     ") | color(kMuted), text(fmt_float(r->minval))}));

            char pct[16];
            std::snprintf(pct, sizeof(pct), "%5.1f%%", static_cast<double>(r->sparsity) * 100.0);
            body.push_back(hbox({
                text(" sparse  ") | color(kMuted),
                meter(r->sparsity, 20, r->sparsity > 0.9f ? kWarn : kOk, opts_.ascii_only),
                text(" "),
                text(pct),
            }));

            if (r->flags & (kFlagHasNaN | kFlagHasInf)) {
                body.push_back(text(" NUMERICAL FAULT: non-finite values present") | color(kErr) |
                               bold);
            }
            if (r->flags & kFlagStatsSampled) {
                body.push_back(text(" ~ stats from a strided sample, not every element") |
                               color(kMuted));
            }
        }
    }

    return pane_frame(Pane::Inspector, "4 RUNTIME METRICS", vbox(std::move(body)) | flex);
}

Element TuiApp::render_ledger() {
    const auto& list = state_.anomalies();
    Elements lines;

    if (list.empty()) {
        lines.push_back(text(" no anomalies detected") | color(kOk));
    }

    const int view_height = 6;
    int start = static_cast<int>(list.size()) - view_height;
    if (ledger_cursor_ >= 0) {
        start = ledger_cursor_ - view_height / 2;
    }
    if (start < 0) start = 0;

    for (int i = start; i < static_cast<int>(list.size()) && i < start + view_height; ++i) {
        const AnomalyEntry& e = list[static_cast<size_t>(i)];
        const auto sev = static_cast<Severity>(e.rec.severity);
        const Color c = sev == Severity::Error ? kErr : (sev == Severity::Warn ? kWarn : kMuted);
        const char* glyph = opts_.ascii_only
                                ? (sev == Severity::Error ? "X" : (sev == Severity::Warn ? "!" : "i"))
                                : (sev == Severity::Error ? "✖" : (sev == Severity::Warn ? "⚠" : "·"));

        // Units differ per rule, so each formats itself rather than sharing one
        // meaningless "%g (limit %g)".
        char detail[160];
        const auto akind = static_cast<AnomalyKind>(e.rec.kind);
        switch (akind) {
            case AnomalyKind::LatencySpike:
                std::snprintf(detail, sizeof(detail), "slow %.3f ms (limit %.3f ms)",
                              static_cast<double>(e.rec.value),
                              static_cast<double>(e.rec.threshold));
                break;
            case AnomalyKind::OutlierMagnitude:
                std::snprintf(detail, sizeof(detail), "|max| %.4g exceeds %.4g",
                              static_cast<double>(e.rec.value),
                              static_cast<double>(e.rec.threshold));
                break;
            case AnomalyKind::HighSparsity:
                std::snprintf(detail, sizeof(detail), "%.1f%% zeros (limit %.1f%%)",
                              static_cast<double>(e.rec.value) * 100.0,
                              static_cast<double>(e.rec.threshold) * 100.0);
                break;
            case AnomalyKind::NaN:
                std::snprintf(detail, sizeof(detail), "NaN present in tensor");
                break;
            case AnomalyKind::Inf:
                std::snprintf(detail, sizeof(detail), "Inf present in tensor");
                break;
            default:
                std::snprintf(detail, sizeof(detail), "%s", to_string(akind));
                break;
        }

        Element line = hbox({
            text(" " + fmt_time(e.rec.t_start_ns) + " ") | color(kMuted),
            text(std::string(glyph) + " ") | color(c),
            text(e.node_name + " ") | color(kind_color(NodeKind::Unknown)),
            text(detail) | color(c),
        });
        if (ledger_cursor_ == i) {
            line = line | inverted;
        }
        lines.push_back(line);
    }

    const std::string title = "5 ANOMALY LEDGER (" + std::to_string(list.size()) + ")";
    return pane_frame(Pane::Ledger, title, vbox(std::move(lines)) | flex);
}

Element TuiApp::render_footer() {
    Elements keys;
    auto key = [&](const char* k, const char* what) {
        keys.push_back(text(std::string(" ") + k) | bold | color(kAccent));
        keys.push_back(text(std::string(":") + what) | color(kMuted));
    };

    key("Tab", "focus");
    key("j/k", "move");
    key("h/l", "collapse/expand");
    key("Space", "toggle");
    switch (focus_) {
        case Pane::Attention:
            key("HJKL", "pan");
            key("+/-", "contrast");
            key("n/p", "head");
            key("f", "fullscreen");
            break;
        case Pane::Stream:
            key("g/G", "top/tail");
            break;
        default:
            break;
    }
    if (source_.supports_seek()) {
        key("Space", "play/pause");
        key("[/]", "speed");
        key("r", "restart");
    }
    key("q", "quit");

    Element bar = hbox(std::move(keys));
    if (toast_frames_ > 0) {
        bar = hbox({std::move(bar), filler(), text(" " + toast_ + " ") | color(kWarn) | bold});
    }
    return bar | bgcolor(Color::RGB(18, 20, 28));
}

Element TuiApp::render_root() {
    if (fullscreen_attention_) {
        return vbox({
            render_header(),
            render_attention() | flex,
            render_footer(),
        });
    }

    return vbox({
        render_header(),
        hbox({
            render_topology() | size(WIDTH, EQUAL, 40),
            render_stream() | flex,
        }) | flex,
        render_attention() | flex,
        hbox({
            render_inspector() | flex,
            render_ledger() | flex,
        }),
        render_footer(),
    });
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

void TuiApp::cycle_focus(int delta) {
    int f = static_cast<int>(focus_) + delta;
    const int n = static_cast<int>(Pane::kCount);
    f = ((f % n) + n) % n;
    focus_ = static_cast<Pane>(f);
}

void TuiApp::move_cursor(int delta) {
    switch (focus_) {
        case Pane::Topology: {
            const int n = static_cast<int>(state_.topo().visible_rows().size());
            if (n == 0) break;
            topo_cursor_ = std::clamp(topo_cursor_ + delta, 0, n - 1);
            sync_capture_target();
            break;
        }
        case Pane::Stream: {
            const int n = static_cast<int>(state_.stream().size());
            if (n == 0) break;
            if (stream_cursor_ < 0) {
                stream_cursor_ = n - 1;  // leaving follow mode grabs the tail
            }
            stream_cursor_ = std::clamp(stream_cursor_ + delta, 0, n - 1);
            break;
        }
        case Pane::Ledger: {
            const int n = static_cast<int>(state_.anomalies().size());
            if (n == 0) break;
            if (ledger_cursor_ < 0) {
                ledger_cursor_ = n - 1;
            }
            ledger_cursor_ = std::clamp(ledger_cursor_ + delta, 0, n - 1);
            break;
        }
        case Pane::Attention:
            attn_row_offset_ = std::max(0, attn_row_offset_ + delta);
            break;
        default:
            break;
    }
}

void TuiApp::activate_selection() {
    if (focus_ == Pane::Topology) {
        const int row = selected_topo_row();
        if (row >= 0) {
            state_.topo().toggle(row);
            sync_capture_target();
        }
    } else if (source_.supports_seek()) {
        source_.set_paused(!source_.paused());
    }
}

bool TuiApp::handle_event(const Event& e) {
    // --- global ---
    if (e == Event::Character('q') || e == Event::Escape) {
        should_quit_ = true;
        return true;
    }
    if (e == Event::Tab) {
        cycle_focus(1);
        return true;
    }
    if (e == Event::TabReverse) {
        cycle_focus(-1);
        return true;
    }
    for (int i = 1; i <= 5; ++i) {
        if (e == Event::Character(static_cast<char>('0' + i))) {
            focus_ = static_cast<Pane>(i - 1);
            return true;
        }
    }

    // --- movement ---
    if (e == Event::Character('j') || e == Event::ArrowDown) {
        move_cursor(1);
        return true;
    }
    if (e == Event::Character('k') || e == Event::ArrowUp) {
        move_cursor(-1);
        return true;
    }
    if (e == Event::PageDown) {
        move_cursor(10);
        return true;
    }
    if (e == Event::PageUp) {
        move_cursor(-10);
        return true;
    }

    if (e == Event::Character('h') || e == Event::ArrowLeft) {
        if (focus_ == Pane::Topology) {
            const int row = selected_topo_row();
            if (row >= 0) {
                state_.topo().set_expanded(row, false);
            }
        } else if (focus_ == Pane::Attention) {
            attn_col_offset_ = std::max(0, attn_col_offset_ - 4);
        }
        return true;
    }
    if (e == Event::Character('l') || e == Event::ArrowRight) {
        if (focus_ == Pane::Topology) {
            const int row = selected_topo_row();
            if (row >= 0) {
                state_.topo().set_expanded(row, true);
            }
        } else if (focus_ == Pane::Attention) {
            attn_col_offset_ += 4;
        }
        return true;
    }

    if (e == Event::Return || e == Event::Character(' ')) {
        activate_selection();
        return true;
    }

    if (e == Event::Character('g')) {
        if (focus_ == Pane::Stream) stream_cursor_ = 0;
        if (focus_ == Pane::Topology) { topo_cursor_ = 0; sync_capture_target(); }
        if (focus_ == Pane::Attention) { attn_row_offset_ = 0; attn_col_offset_ = 0; }
        return true;
    }
    if (e == Event::Character('G')) {
        if (focus_ == Pane::Stream) stream_cursor_ = -1;  // resume following
        if (focus_ == Pane::Ledger) ledger_cursor_ = -1;
        return true;
    }

    // --- attention pane ---
    if (focus_ == Pane::Attention) {
        if (e == Event::Character('J')) { attn_row_offset_ += 4; return true; }
        if (e == Event::Character('K')) { attn_row_offset_ = std::max(0, attn_row_offset_ - 4); return true; }
        if (e == Event::Character('H')) { attn_col_offset_ = std::max(0, attn_col_offset_ - 4); return true; }
        if (e == Event::Character('L')) { attn_col_offset_ += 4; return true; }
        if (e == Event::Character('f')) { fullscreen_attention_ = !fullscreen_attention_; return true; }
        if (e == Event::Character('+') || e == Event::Character('=')) {
            attn_contrast_ = std::min(8.0f, attn_contrast_ * 1.25f);
            return true;
        }
        if (e == Event::Character('-') || e == Event::Character('_')) {
            attn_contrast_ = std::max(0.25f, attn_contrast_ / 1.25f);
            return true;
        }
        if (e == Event::Character('n')) {
            attn_head_ = std::min(std::max(0, state_.model().n_head - 1), attn_head_ + 1);
            sync_capture_target();
            return true;
        }
        if (e == Event::Character('p')) {
            attn_head_ = std::max(0, attn_head_ - 1);
            sync_capture_target();
            return true;
        }
    }

    // --- replay transport ---
    if (source_.supports_seek()) {
        if (e == Event::Character('r')) {
            source_.restart();
            toast_ = "replay restarted";
            toast_frames_ = 40;
            return true;
        }
        if (e == Event::Character(']')) {
            source_.set_speed(source_.speed() * 2.0);
            toast_ = "speed x" + std::to_string(source_.speed());
            toast_frames_ = 40;
            return true;
        }
        if (e == Event::Character('[')) {
            source_.set_speed(source_.speed() / 2.0);
            toast_ = "speed x" + std::to_string(source_.speed());
            toast_frames_ = 40;
            return true;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------

std::string TuiApp::snapshot(int width, int height, int warmup_ms) {
    // Let the source produce something worth showing. Pumping in slices rather
    // than one long sleep matters for FakeSource and replay, which release
    // records against a wall clock.
    const int slices = 20;
    for (int i = 0; i < slices; ++i) {
        state_.pump();
        sync_capture_target();
        std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, warmup_ms / slices)));
    }
    state_.pump();

    Element root = render_root();
    Screen screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
    Render(screen, root);
    return screen.ToString();
}

int TuiApp::run() {
    auto screen = ScreenInteractive::Fullscreen();
    screen.TrackMouse(false);

    // The tracer thread produces asynchronously, so the UI drives its own
    // repaint clock rather than waiting for input.
    std::atomic<bool> alive{true};
    std::thread ticker([&] {
        const auto period = std::chrono::milliseconds(1000 / std::max(1, opts_.fps));
        while (alive.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(period);
            screen.PostEvent(Event::Custom);
        }
    });

    auto component = Renderer([&] {
        state_.pump();
        sync_capture_target();
        if (toast_frames_ > 0) {
            --toast_frames_;
        }
        return render_root();
    });

    component = CatchEvent(component, [&](Event e) {
        const bool handled = handle_event(e);
        if (should_quit_) {
            screen.Exit();
        }
        return handled;
    });

    screen.Loop(component);

    alive.store(false, std::memory_order_release);
    ticker.join();

    source_.request_stop();
    return 0;
}

}  // namespace llmscope

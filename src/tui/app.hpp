// app.hpp - the terminal UI.
//
// Owns the FTXUI event loop and all view state (focus, cursors, viewport). Every
// pane is a pure function of UiState plus this view state, which keeps rendering
// free of side effects and makes the whole thing testable against FakeSource.
#pragma once

#include <cstdint>
#include <string>

#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ui_state.hpp"

namespace llmscope {

// Panes in Tab order.
enum class Pane : int {
    Topology = 0,
    Stream,
    Attention,
    Inspector,
    Ledger,
    kCount,
};

struct UiOptions {
    // Terminals without a NerdFont or good Unicode coverage get an ASCII ramp
    // instead of block-drawing characters.
    bool ascii_only = false;
    bool no_color = false;
    int  fps = 20;
};

class TuiApp {
public:
    TuiApp(TraceSource& source, UiOptions opts);

    // Blocking. Returns a process exit code.
    int run();

    // Renders a single frame off-screen and returns it as an ANSI string,
    // without touching the terminal or requiring a TTY.
    //
    // This exists so the UI can be verified in CI and in non-interactive shells,
    // where FTXUI's event loop has no stdin to read. It is also how the README
    // screenshots are produced, which keeps them honest.
    std::string snapshot(int width, int height, int warmup_ms);

private:
    // --- rendering: each pane is a pure function of state_ plus view state ---
    ftxui::Element render_root();
    ftxui::Element render_header();
    ftxui::Element render_topology();
    ftxui::Element render_stream();
    ftxui::Element render_attention();
    ftxui::Element render_inspector();
    ftxui::Element render_ledger();
    ftxui::Element render_footer();
    ftxui::Element pane_frame(Pane pane, const std::string& title, ftxui::Element content);

    // --- input ---
    bool handle_event(const ftxui::Event& event);
    void cycle_focus(int delta);
    void move_cursor(int delta);
    void activate_selection();
    void sync_capture_target();

    // --- helpers ---
    int selected_topo_row() const;
    int selected_layer() const;
    uint16_t selected_name_id() const;

    TraceSource& source_;
    UiState state_;
    UiOptions opts_;

    Pane focus_ = Pane::Topology;
    bool fullscreen_attention_ = false;
    bool should_quit_ = false;

    // Cursors, one per navigable pane.
    int topo_cursor_ = 0;
    int stream_cursor_ = 0;   // -1 => follow the tail
    int ledger_cursor_ = -1;

    // Attention viewport.
    int attn_row_offset_ = 0;
    int attn_col_offset_ = 0;
    float attn_contrast_ = 1.0f;
    int attn_head_ = 0;

    // Cached so we only push a capture target when it actually changes.
    uint16_t last_capture_name_ = 0xFFFF;
    int32_t last_capture_layer_ = -2;
    int32_t last_capture_head_ = -1;

    std::string toast_;
    int toast_frames_ = 0;
};

}  // namespace llmscope

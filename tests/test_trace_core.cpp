// Unit tests for trace_core. No test framework on purpose: trace_core has zero
// dependencies and that property is worth keeping, including in its tests.
//
// Covers the pieces where a bug would be silent rather than loud: the ring
// buffer's drop accounting, name-based node classification, tree construction,
// and trace file round-tripping.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "trace_core/fake_source.hpp"
#include "trace_core/live_source.hpp"
#include "trace_core/payload_slot.hpp"
#include "trace_core/spsc_ring.hpp"
#include "trace_core/topology.hpp"
#include "trace_core/trace_file.hpp"

using namespace llmscope;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("FAIL %s:%d: %s\n", file, line, expr);
    }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

void test_record_layout() {
    // The trace file format depends on this; a silent layout change would make
    // old traces decode into garbage rather than fail loudly.
    CHECK(sizeof(NodeRecord) == 88);
    NodeRecord r{};
    r.ne[0] = 4; r.ne[1] = 3; r.ne[2] = 2; r.ne[3] = 1;
    CHECK(element_count(r) == 24);
}

void test_ring_basic() {
    SpscRing<int> ring(8);
    CHECK(ring.capacity() == 8);

    for (int i = 0; i < 5; ++i) {
        CHECK(ring.push(i));
    }
    std::vector<int> out;
    CHECK(ring.pop_batch(out, 100) == 5);
    CHECK(out.size() == 5);
    for (int i = 0; i < 5; ++i) {
        CHECK(out[static_cast<size_t>(i)] == i);
    }
    CHECK(ring.dropped() == 0);
}

void test_ring_overflow_drops_and_stays_consistent() {
    SpscRing<int> ring(4);
    for (int i = 0; i < 4; ++i) {
        CHECK(ring.push(i));
    }
    // Full: further pushes must fail rather than corrupt the buffer.
    CHECK(!ring.push(99));
    CHECK(!ring.push(100));
    CHECK(ring.dropped() == 2);

    std::vector<int> out;
    ring.pop_batch(out, 100);
    CHECK(out.size() == 4);
    // The oldest entries survive - we drop the newest, which is what keeps
    // tail_ single-writer.
    CHECK(out[0] == 0 && out[3] == 3);
}

void test_ring_threaded() {
    // Exercises the actual producer/consumer split under contention.
    constexpr int kTotal = 200000;
    SpscRing<int> ring(1024);

    std::thread producer([&] {
        for (int i = 0; i < kTotal; ++i) {
            while (!ring.push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::vector<int> received;
    received.reserve(kTotal);
    std::vector<int> batch;
    while (static_cast<int>(received.size()) < kTotal) {
        batch.clear();
        ring.pop_batch(batch, 256);
        for (int v : batch) {
            received.push_back(v);
        }
    }
    producer.join();

    CHECK(static_cast<int>(received.size()) == kTotal);
    bool ordered = true;
    for (int i = 0; i < kTotal; ++i) {
        if (received[static_cast<size_t>(i)] != i) {
            ordered = false;
            break;
        }
    }
    CHECK(ordered);  // SPSC must preserve order exactly
}

void test_parse_node_name() {
    // Ordering matters here: "attn_norm" and "ffn_norm" must classify as norms,
    // not as attention or MLP nodes.
    CHECK(parse_node_name("attn_norm-3").kind == NodeKind::Norm);
    CHECK(parse_node_name("attn_norm-3").layer == 3);
    CHECK(parse_node_name("ffn_norm-11").kind == NodeKind::Norm);
    CHECK(parse_node_name("ffn_up-12").kind == NodeKind::Ffn);
    CHECK(parse_node_name("ffn_up-12").layer == 12);
    CHECK(parse_node_name("kq_soft_max_ext-0").kind == NodeKind::AttnScores);
    CHECK(parse_node_name("kq_soft_max_ext-0").layer == 0);
    CHECK(parse_node_name("Qcur-7").kind == NodeKind::Attention);
    CHECK(parse_node_name("kqv_out-2").kind == NodeKind::Attention);
    CHECK(parse_node_name("cache_k_l5").kind == NodeKind::Cache);
    CHECK(parse_node_name("cache_k_l5").layer == 5);
    CHECK(parse_node_name("l_out-9").kind == NodeKind::LayerOut);
    CHECK(parse_node_name("inp_embd").kind == NodeKind::Embedding);
    CHECK(parse_node_name("inp_embd").layer == -1);
    CHECK(parse_node_name("result_output").kind == NodeKind::Output);
    CHECK(parse_node_name("result_norm").kind == NodeKind::Norm);
    CHECK(parse_node_name("result_norm").layer == -1);

    // A name with a trailing dash that is not a layer index must not be parsed
    // as one.
    CHECK(parse_node_name("some-node").layer == -1);
}

void test_topology_tree() {
    Topology topo;
    const char* names[] = {"inp_embd",   "attn_norm-0", "Qcur-0",  "kq_soft_max_ext-0",
                           "ffn_up-0",   "l_out-0",     "attn_norm-1", "ffn_up-1",
                           "result_norm"};
    uint16_t id = 0;
    for (const char* n : names) {
        topo.observe(id++, n, parse_node_name(n));
    }

    topo.expand_all();
    const std::vector<int> rows = topo.visible_rows();
    CHECK(rows.size() > 5);

    // Root must be first and there must be exactly one of it.
    CHECK(topo.node(rows[0]).label == "model");

    // Two distinct blocks should have been created.
    int layer_groups = 0;
    for (int r : rows) {
        const TopoNode& n = topo.node(r);
        if (n.is_group && n.label.rfind("layers.", 0) == 0) {
            ++layer_groups;
        }
    }
    CHECK(layer_groups == 2);

    // Collapsing the root hides everything below it.
    topo.collapse_all();
    CHECK(topo.visible_rows().size() == 1);

    // Observing the same id twice must not duplicate nodes.
    const int before = topo.size();
    topo.observe(0, "inp_embd", parse_node_name("inp_embd"));
    CHECK(topo.size() == before);
}

void test_topology_collect_ids() {
    Topology topo;
    const char* names[] = {"attn_norm-0", "Qcur-0", "ffn_up-0", "attn_norm-1"};
    uint16_t id = 0;
    for (const char* n : names) {
        topo.observe(id++, n, parse_node_name(n));
    }
    topo.expand_all();

    // The row for block 0 must gather all three of its nodes and none of block 1's.
    for (int i = 0; i < topo.size(); ++i) {
        const TopoNode& n = topo.node(i);
        if (n.is_group && n.label == "layers.0") {
            const std::vector<uint16_t> ids = topo.collect_name_ids(i);
            CHECK(ids.size() == 3);
            CHECK(std::find(ids.begin(), ids.end(), uint16_t(3)) == ids.end());
        }
    }
}

void test_payload_slot() {
    PayloadSlot slot;
    PayloadHeader header{};
    std::vector<float> out;

    // Nothing published yet.
    CHECK(!slot.try_read(header, out));

    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
    PayloadHeader h{};
    h.seq = 42;
    h.ne[0] = 2;
    h.ne[1] = 2;
    h.kind = static_cast<uint8_t>(NodeKind::AttnScores);
    slot.publish(h, data.data(), 4);

    CHECK(slot.version() == 1);
    CHECK(slot.try_read(header, out));
    CHECK(header.seq == 42);
    CHECK(out.size() == 4);
    CHECK(out[2] == 3.0f);

    // Oversized publishes must clamp and flag rather than overrun. The buffer is
    // genuinely that large: publish() promises to read at most kMaxPayloadFloats
    // elements, but the caller still has to own the ones it declares.
    std::vector<float> big(kMaxPayloadFloats + 1000, 7.0f);
    PayloadHeader h2{};
    h2.seq = 43;
    slot.publish(h2, big.data(), static_cast<uint32_t>(big.size()));
    CHECK(slot.try_read(header, out));
    CHECK(header.truncated == 1);
    CHECK(header.n_floats == kMaxPayloadFloats);
    CHECK(out.size() == kMaxPayloadFloats);
    CHECK(out[0] == 7.0f && out[kMaxPayloadFloats - 1] == 7.0f);
}

void test_trace_file_roundtrip() {
    const std::string path = "llmscope_test_roundtrip.trace";

    ModelInfo model;
    model.name = "test-model";
    model.arch = "llama";
    model.n_layer = 2;
    model.n_head = 8;
    model.n_embd = 128;
    model.backend = BackendId::CPU;

    {
        TraceWriter w;
        std::string err;
        CHECK(w.open(path, err));

        w.write_model(model);
        w.write_name(0, "attn_norm-0", static_cast<uint8_t>(NodeKind::Norm), 0);
        w.write_name(1, "ffn_up-0", static_cast<uint8_t>(NodeKind::Ffn), 0);

        for (int i = 0; i < 10; ++i) {
            NodeRecord r{};
            r.seq = static_cast<uint64_t>(i);
            r.t_start_ns = static_cast<uint64_t>(i) * 1000;  // all in the past on replay
            r.dur_ns = 500;
            r.name_id = static_cast<uint16_t>(i % 2);
            r.layer = 0;
            r.ne[0] = 128; r.ne[1] = 1; r.ne[2] = 1; r.ne[3] = 1;
            r.mean = 0.5f;
            r.absmax = 2.0f;
            r.sparsity = 0.25f;
            r.kind = static_cast<uint8_t>(NodeKind::Ffn);
            w.write_record(r);
        }

        TokenInfo t;
        t.index = 0;
        t.id = 7;
        t.piece = "hi";
        w.write_token(t);

        AnomalyRecord a{};
        a.seq = 3;
        a.value = 500.0f;
        a.threshold = 100.0f;
        a.kind = static_cast<uint8_t>(AnomalyKind::OutlierMagnitude);
        a.severity = static_cast<uint8_t>(Severity::Warn);
        w.write_anomaly(a);

        w.close();
    }

    std::string err;
    auto src = TraceFileSource::load(path, err);
    CHECK(src != nullptr);
    if (src) {
        CHECK(src->model_info().name == "test-model");
        CHECK(src->model_info().n_layer == 2);

        std::vector<Topology::Entry> names;
        CHECK(src->poll_names(names) == 2);
        CHECK(names[0].name == "attn_norm-0");
        CHECK(names[1].layer == 0);

        // Replay is time-paced from the moment of load, and the records span the
        // trace's own 9 us, so wait past that before expecting the full set -
        // exactly as the restart() case below does. Polling immediately only
        // works where load() happens to take longer than 9 us, which is why this
        // passed on Windows and failed on faster Linux CI runners.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::vector<NodeRecord> recs;
        src->poll_records(recs, 100);
        CHECK(recs.size() == 10);
        if (recs.size() == 10) {  // guard: a short read must not index past the end
            CHECK(recs[0].seq == 0);
            CHECK(recs[9].dur_ns == 500);
            CHECK(recs[4].sparsity == 0.25f);
        }

        std::vector<AnomalyRecord> anomalies;
        src->poll_anomalies(anomalies, 100);
        CHECK(anomalies.size() == 1);
        CHECK(anomalies[0].kind == static_cast<uint8_t>(AnomalyKind::OutlierMagnitude));

        // restart() rewinds the cursor, but replay is time-paced: immediately
        // after a rewind almost nothing is due yet. Wait past the trace's own
        // 9 us span before expecting the full set back.
        src->restart();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::vector<NodeRecord> again;
        src->poll_records(again, 100);
        CHECK(again.size() == 10);
        if (!again.empty()) {
            CHECK(again[0].seq == 0);
        }
    }

    std::remove(path.c_str());
}

void test_trace_file_rejects_garbage() {
    const std::string path = "llmscope_test_garbage.trace";
    {
        std::ofstream f(path, std::ios::binary);
        const char junk[] = "this is definitely not a trace file";
        f.write(junk, sizeof(junk));
    }
    std::string err;
    auto src = TraceFileSource::load(path, err);
    CHECK(src == nullptr);
    CHECK(!err.empty());
    std::remove(path.c_str());
}

void test_live_source_flow() {
    LiveSource live(64);

    ModelInfo m;
    m.name = "live-test";
    live.set_model_info(m);
    CHECK(live.model_info().name == "live-test");

    const uint16_t a = live.observe_name("ffn_up-0", NodeKind::Ffn, 0);
    const uint16_t b = live.observe_name("ffn_up-0", NodeKind::Ffn, 0);
    CHECK(a == b);  // interning must be stable

    std::vector<Topology::Entry> names;
    CHECK(live.poll_names(names) == 1);  // and only announce the name once
    names.clear();
    CHECK(live.poll_names(names) == 0);

    NodeRecord r{};
    r.name_id = a;
    r.dur_ns = 1234;
    CHECK(live.push_record(r));

    std::vector<NodeRecord> out;
    CHECK(live.poll_records(out, 10) == 1);
    CHECK(out[0].dur_ns == 1234);

    // Capture target is a plain pair of atomics.
    live.set_capture_target(a, 5);
    CHECK(live.capture_name_id() == a);
    CHECK(live.capture_layer() == 5);

    live.request_stop();
    CHECK(live.stop_requested());
}

void test_fake_source_produces_a_coherent_model() {
    FakeSource::Options opts;
    opts.n_layer = 4;
    opts.n_head = 8;
    opts.tokens_per_sec = 1000.0;  // run fast so the test does not sleep
    FakeSource src(opts);

    CHECK(src.model_info().n_layer == 4);

    std::vector<Topology::Entry> names;
    CHECK(src.poll_names(names) > 0);

    Topology topo;
    for (const auto& e : names) {
        ParsedName p;
        p.kind = static_cast<NodeKind>(e.kind);
        p.layer = e.layer;
        topo.observe(e.name_id, e.name, p);
    }
    topo.expand_all();
    CHECK(topo.visible_rows().size() > 10);

    // Give the synthetic clock a moment, then confirm records flow.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::vector<NodeRecord> recs;
    src.poll_records(recs, 100000);
    CHECK(!recs.empty());

    PayloadHeader h{};
    std::vector<float> attn;
    CHECK(src.read_payload(h, attn));
    CHECK(h.kind == static_cast<uint8_t>(NodeKind::AttnScores));
    CHECK(h.ne[0] == h.ne[1]);  // attention planes are square

    // Causality: the upper triangle of a causal attention matrix must be zero.
    const int n = static_cast<int>(h.ne[0]);
    bool causal = true;
    for (int i = 0; i < n && causal; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (attn[static_cast<size_t>(i) * n + j] != 0.0f) {
                causal = false;
                break;
            }
        }
    }
    CHECK(causal);

    // And each row should sum to ~1 after softmax normalisation.
    bool normalised = true;
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
            sum += attn[static_cast<size_t>(i) * n + j];
        }
        if (sum < 0.99f || sum > 1.01f) {
            normalised = false;
            break;
        }
    }
    CHECK(normalised);
}

}  // namespace

int main() {
    test_record_layout();
    test_ring_basic();
    test_ring_overflow_drops_and_stays_consistent();
    test_ring_threaded();
    test_parse_node_name();
    test_topology_tree();
    test_topology_collect_ids();
    test_payload_slot();
    test_trace_file_roundtrip();
    test_trace_file_rejects_garbage();
    test_live_source_flow();
    test_fake_source_produces_a_coherent_model();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

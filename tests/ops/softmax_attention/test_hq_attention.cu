// Independent FP64 attention oracle over the stored signed lattice codes and FP16 norms.
// Production BF16 QK/PV staging is an implementation profile, not part of the oracle.
// Each output row must have cosine > .999 and relative L2 < .02. Standalone append also
// supplies an exact byte oracle for fused append; cached reads must leave every byte intact.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include "core/device.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
using namespace ninfer;
using namespace ninfer::ops;

namespace {
constexpr int D = 256, Page = 64;
constexpr auto Storage = KvCacheStorage::HqE8Rice2B;

template <typename T>
struct DeviceArray {
    T* data = nullptr;
    std::size_t count;

    explicit DeviceArray(std::size_t n) : count(n) {
        CUDA_CHECK(cudaMalloc(&data, std::max(n, std::size_t{1}) * sizeof(T)));
        CUDA_CHECK(cudaMemset(data, 0, std::max(n, std::size_t{1}) * sizeof(T)));
    }

    ~DeviceArray() { cudaFree(data); }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void put(const std::vector<T>& v) {
        if (v.size() != count) throw std::runtime_error("upload size mismatch");
        CUDA_CHECK(cudaMemcpy(data, v.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<T> get() const {
        std::vector<T> v(count);
        CUDA_CHECK(cudaMemcpy(v.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

int sign(int d) {
    std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    return (x & 1u) ? 1 : -1;
}

void hadamard(double* v) {
    for (int len = 1; len < D; len *= 2)
        for (int i = 0; i < D; i += 2 * len)
            for (int j = 0; j < len; ++j) {
                const double a = v[i + j], b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
}

// MSB-first stream within little-endian stored 32-bit words. Recover coset and mod-4 parity
// from the stripped symbols, then multiply the signed lattice coordinates by the stored norm and
// add back the subtractive dither (host mirror of the codec's counter hash).
std::uint64_t dither_row_seed(int kv_head, std::int64_t position, bool role_v) {
    std::uint64_t x = 0x5DEECE66Dull ^
                      (static_cast<std::uint64_t>(kv_head) * 0x2545F4914F6CDD1Dull) ^
                      (static_cast<std::uint64_t>(position) * 0x9E3779B97F4A7C15ull) ^
                      (role_v ? 0xA5A5A5A5A5A5A5A5ull : 0x1B873593B5244C61ull);
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    return x;
}

std::uint64_t dither_word_seed(std::uint64_t row_seed, int word) {
    std::uint64_t x = row_seed ^ (0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(word) + 1u));
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

double dither(std::uint64_t word_seed, int j) {
    const std::uint32_t bits = static_cast<std::uint32_t>((word_seed >> (8 * j)) & 0xFFull);
    return static_cast<double>(bits) * (1.0 / 255.0) - 0.5;
}

void decode(const std::uint8_t* codes, const std::uint8_t* meta, int kv_head, int position,
            bool role_v, double* out) {
    const unsigned used = meta[3] | ((meta[4] & 3u) << 8);
    if (!used) {
        std::fill(out, out + D, 0.0);
        return;
    }
    const int k               = meta[2] & 15;
    const unsigned short bits = meta[0] | (static_cast<unsigned short>(meta[1]) << 8);
    __half norm;
    std::memcpy(&norm, &bits, sizeof(bits));
    const double scale =
        double(__half2float(norm)) * (1 << ((meta[2] >> 4) & 3)) / (double(1.45f) * 16.0);
    const std::uint64_t row_seed = dither_row_seed(kv_head, position, role_v);
    int cursor = 0;
    auto bit   = [&]() {
        if (cursor >= int(used)) throw std::runtime_error("truncated HQ row");
        const int word = cursor / 32, shift = 31 - cursor++ % 32;
        return (codes[word * 4 + shift / 8] >> (shift % 8)) & 1;
    };
    auto signed_code = [](unsigned z) { return (z & 1) ? -int(z / 2) - 1 : int(z / 2); };
    for (int w = 0; w < D / 8; ++w) {
        const std::uint64_t wseed = dither_word_seed(row_seed, w);
        unsigned z[8];
        for (auto& value : z) {
            unsigned quotient = 0, remainder = 0;
            while (!bit()) ++quotient;
            for (int r = 0; r < k; ++r) remainder = 2 * remainder + bit();
            value = (quotient << k) + remainder;
        }
        const int coset = z[7] & 1;
        int sum         = 0;
        for (int j = 0; j < 7; ++j) {
            const int s = signed_code(z[j]);
            sum += s;
            out[w * 8 + j] = (2 * s + coset + dither(wseed, j)) * scale;
        }
        out[w * 8 + 7] = (4 * signed_code(z[7] / 2) + 2 * (sum & 1) + coset + dither(wseed, 7)) *
                         scale;
    }
    if (cursor != int(used)) throw std::runtime_error("HQ used-bit mismatch");
}

struct Scenario {
    const char* name;
    int heads, batch, width, window;
    bool masked = false, cached = false, graph = false;
    // Run with the residual window: side planes + validity words threaded through every view;
    // the oracle reads sink/recent keys from the exact rotated rows instead of codec rows.
    bool residual = false;
};

void run(const Scenario& sc, unsigned seed) {
    const int H = sc.heads, KV = H == 24 ? 4 : 2, B = sc.batch, W = sc.width;
    const int history = sc.window - W, pages_per_batch = (sc.window + Page - 1) / Page;
    const int pages              = pages_per_batch * B;
    const std::size_t plane_rows = std::size_t(pages) * KV * Page;
    const std::size_t input_rows = std::size_t(sc.window) * KV;
    DeviceArray<std::uint8_t> ck(plane_rows * 64), cv(plane_rows * 64), mk(plane_rows * 8),
        mv(plane_rows * 8);
    DeviceArray<std::int32_t> tables(pages), rows(B), valid(B), pos(W * B), all_pos(sc.window);
    DeviceArray<__nv_bfloat16> q(D * H * W * B), k(D * KV * W * B), v(D * KV * W * B);
    DeviceArray<__nv_bfloat16> output(D * H * W * B), raw_k(D * input_rows * B),
        raw_v(D * input_rows * B);
    std::mt19937 rng(seed);
    std::normal_distribution<float> gaussian;
    auto random_rows = [&](std::size_t n) {
        std::vector<__nv_bfloat16> result(n);
        for (auto& x : result) x = __float2bfloat16(gaussian(rng));
        return result;
    };
    const auto hq = random_rows(q.count), hk = random_rows(raw_k.count),
               hv = random_rows(raw_v.count);
    q.put(hq);
    raw_k.put(hk);
    raw_v.put(hv);
    std::vector<std::int32_t> ht(pages), hr(B), hp(W * B), valid_count(B, W), sequential(sc.window);
    for (int p = 0; p < sc.window; ++p) sequential[p] = p;
    all_pos.put(sequential);
    for (int b = 0; b < B; ++b) {
        hr[b] = B - 1 - b;
        for (int p = 0; p < pages_per_batch; ++p)
            ht[hr[b] * pages_per_batch + p] = b * pages_per_batch + pages_per_batch - 1 - p;
        if (sc.masked) valid_count[b] = b == B - 1 ? 0 : std::max(1, W - b - 2);
        for (int t = 0; t < W; ++t)
            hp[b * W + t] = valid_count[b] ? history + std::min(t, valid_count[b] - 1) : 0;
        const auto src = std::size_t(b) * D * input_rows + std::size_t(history) * D * KV;
        const auto dst = std::size_t(b) * W * D * KV;
        CUDA_CHECK(cudaMemcpy(k.data + dst, raw_k.data + src, W * D * KV * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(v.data + dst, raw_v.data + src, W * D * KV * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
    }
    tables.put(ht);
    rows.put(hr);
    pos.put(hp);
    valid.put(valid_count);
    constexpr int kSideRows = 32 + 512;
    constexpr int kSideWords = 512 / 32 + 1;
    DeviceArray<__nv_bfloat16> residual_k(sc.residual ? D * KV * kSideRows * B : 0);
    DeviceArray<__nv_bfloat16> residual_v(sc.residual ? D * KV * kSideRows * B : 0);
    DeviceArray<std::uint32_t> side_words(sc.residual ? kSideWords * B : 0);
    if (sc.residual) {
        CUDA_CHECK(cudaMemset(residual_k.data, 0, residual_k.count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(residual_v.data, 0, residual_v.count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(side_words.data, 0, side_words.count * sizeof(std::uint32_t)));
    }
    Tensor residual_k_view, residual_v_view, words_view;
    if (sc.residual) {
        residual_k_view = Tensor(residual_k.data, DType::BF16, {D, KV, kSideRows, B});
        residual_v_view = Tensor(residual_v.data, DType::BF16, {D, KV, kSideRows, B});
        words_view      = Tensor(side_words.data, DType::I32, {kSideWords, B});
    }
    PagedKVBatchLayerView cache{Tensor(ck.data, DType::U8, {64, Page, KV, pages}),
                                Tensor(cv.data, DType::U8, {64, Page, KV, pages}),
                                Tensor(mk.data, DType::U8, {8, Page, KV, pages}),
                                Tensor(mv.data, DType::U8, {8, Page, KV, pages}),
                                residual_k_view,
                                residual_v_view,
                                words_view,
                                Tensor(tables.data, DType::I32, {pages_per_batch, B}),
                                D,
                                KV,
                                Storage};
    auto single_cache = [&](int b) {
        return PagedKVLayerView{
            cache.k_pages,
            cache.v_pages,
            cache.k_scale_pages,
            cache.v_scale_pages,
            residual_k_view,
            residual_v_view,
            words_view,
            hr[b],
            Tensor(tables.data + hr[b] * pages_per_batch, DType::I32, {pages_per_batch}),
            D,
            KV,
            Storage};
    };
    auto append = [&](int b, int count) {
        if (!count) return;
        Tensor keys(raw_k.data + b * D * input_rows, DType::BF16, {D, KV, count});
        Tensor vals(raw_v.data + b * D * input_rows, DType::BF16, {D, KV, count});
        Tensor positions(all_pos.data, DType::I32, {count});
        kv_cache_append(keys, vals, positions, single_cache(b), nullptr);
    };
    // Expected final cache, followed by rebuilding only the initial history in the same pages.
    for (int b = 0; b < B; ++b) append(b, history + valid_count[b]);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto expected_ck = ck.get(), expected_cv = cv.get(), expected_mk = mk.get(),
               expected_mv = mv.get();
    if (!sc.cached) {
        CUDA_CHECK(cudaMemset(ck.data, 0, ck.count));
        CUDA_CHECK(cudaMemset(cv.data, 0, cv.count));
        CUDA_CHECK(cudaMemset(mk.data, 0, mk.count));
        CUDA_CHECK(cudaMemset(mv.data, 0, mv.count));
        for (int b = 0; b < B; ++b) append(b, history);
    }
    Tensor tq(q.data, DType::BF16, {D, H, W, B}), tk(k.data, DType::BF16, {D, KV, W, B});
    Tensor tv(v.data, DType::BF16, {D, KV, W, B}), tp(pos.data, DType::I32, {W, B});
    Tensor tr(rows.data, DType::I32, {B}), tvalid(valid.data, DType::I32, {B});
    Tensor out(output.data, DType::BF16, {D, H, W, B});
    const AttentionHeadGeometry geometry{D, H, KV};
    const CausalAttentionExecutionEnvelope envelope{std::uint32_t(sc.window),
                                                    std::uint32_t(sc.window)};
    const auto capacity =
        causal_softmax_attention_workspace_capacity_bytes(geometry, Storage, envelope, B, W, W);
    DeviceArray<std::uint8_t> scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.data, capacity});
    auto invoke = [&](cudaStream_t stream) {
        if (sc.cached)
            causal_softmax_attention_cached(tq, tp, geometry, .0625f, single_cache(0), envelope,
                                            workspace, out, stream);
        else
            causal_softmax_attention(tq, tk, tv, tp, sc.masked ? tvalid : Tensor{}, tr, geometry,
                                     .0625f, cache, envelope, workspace, out, stream);
    };
    CUDA_CHECK(cudaDeviceSynchronize());
    invoke(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (sc.graph) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        invoke(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaMemset(output.data, 0x7f, output.count * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    if (ck.get() != expected_ck || cv.get() != expected_cv || mk.get() != expected_mk ||
        mv.get() != expected_mv)
        throw std::runtime_error(
            "cache differs from standalone append (or cached read mutated it)");
    const auto got   = output.get();
    double worst_cos = 1, worst_rel = 0;
    // Residual-window oracle convention (mirrors the pre-rebase gate): a side-selected key's
    // expected value is the side-plane SLOT ROW as it stands at attention time - whatever the
    // last dual-write left there - not the key's input row. The current chunk's fill rewrites
    // the ring slots congruent to its own keys, and consumers read those rows by design; the
    // scratch/fresh staging writes identical rotated values, so the plane is self-consistent
    // for both roles. Fresh (prompt) coverage extends the side selection to the whole current
    // chunk; the ring serves the W keys before it. Cached and small-T reads use the window
    // tail [total - 512, total).
    const bool prompt_route =
        !sc.cached && W > (H == 24 ? 8 : 6) &&
        detail::causal_attention_resolve_route(H, W, B, Storage, envelope) ==
            detail::CausalAttentionRoute::Prompt;
    std::vector<__nv_bfloat16> plane_k, plane_v;
    if (sc.residual) {
        plane_k.resize(residual_k.count);
        plane_v.resize(residual_v.count);
        CUDA_CHECK(cudaMemcpy(plane_k.data(), residual_k.data, plane_k.size() * 2,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(plane_v.data(), residual_v.data, plane_v.size() * 2,
                              cudaMemcpyDeviceToHost));
    }
    const auto side_value = [&](int b, int h, int p, bool role_v, double* out_row) {
        const std::vector<__nv_bfloat16>& plane = role_v ? plane_v : plane_k;
        const int row = p < 32 ? p : 32 + (p & 511);
        const __nv_bfloat16* src =
            plane.data() +
            (static_cast<std::size_t>(hr[b]) * (32 + 512) + row) * KV * D + std::size_t(h) * D;
        for (int d = 0; d < D; ++d) out_row[d] = double(__bfloat162float(src[d]));
    };
    const auto side_selected = [&](int total, int p) {
        return p < 32 ||
               (prompt_route ? p >= history - 512 && p < total : p >= total - 512 && p < total);
    };
    std::vector<double> side_row(D);
    for (int b = 0; b < B; ++b) {
        std::vector<double> decoded_k(input_rows * D), decoded_v(input_rows * D);
        const int total = history + valid_count[b];
        for (int p = 0; p < total; ++p) {
            const bool side = sc.residual && side_selected(total, p);
            for (int h = 0; h < KV; ++h) {
                double* dk = decoded_k.data() + (p * KV + h) * D;
                double* dv = decoded_v.data() + (p * KV + h) * D;
                const int page = ht[hr[b] * pages_per_batch + p / Page];
                const auto row = std::size_t(page * KV + h) * Page + p % Page;
                decode(expected_ck.data() + row * 64, expected_mk.data() + row * 8, h, p, false,
                       dk);
                decode(expected_cv.data() + row * 64, expected_mv.data() + row * 8, h, p, true,
                       dv);
                if (side) {
                    side_value(b, h, p, false, side_row.data());
                    std::copy(side_row.begin(), side_row.end(), dk);
                    side_value(b, h, p, true, side_row.data());
                    std::copy(side_row.begin(), side_row.end(), dv);
                }
            }
        }
        for (int t = 0; t < W; ++t)
            for (int h = 0; h < H; ++h) {
                const auto index = std::size_t((b * W + t) * H + h) * D;
                if (t >= valid_count[b]) {
                    for (int d = 0; d < D; ++d)
                        if (__bfloat162float(got[index + d]) != 0)
                            throw std::runtime_error("masked output is not zero");
                    continue;
                }
                double query[D], expected[D]{};
                for (int d = 0; d < D; ++d)
                    query[d] = double(__bfloat162float(hq[index + d])) * sign(d);
                hadamard(query);
                for (auto& x : query) x /= 16;
                const int visible = history + t + 1, kvh = h / (H / KV);
                std::vector<double> scores(visible);
                double maximum = -1e300, denom = 0;
                for (int p = 0; p < visible; ++p) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) dot += query[d] * decoded_k[(p * KV + kvh) * D + d];
                    maximum = std::max(maximum, scores[p] = dot / 16);
                }
                for (auto& s : scores) {
                    s = std::exp(s - maximum);
                    denom += s;
                }
                for (int p = 0; p < visible; ++p)
                    for (int d = 0; d < D; ++d)
                        expected[d] += scores[p] / denom * decoded_v[(p * KV + kvh) * D + d];
                hadamard(expected);
                double dot = 0, na = 0, nb = 0, error = 0;
                for (int d = 0; d < D; ++d) {
                    const double ref    = expected[d] * sign(d) / 16,
                                 actual = __bfloat162float(got[index + d]);
                    dot += ref * actual;
                    na += ref * ref;
                    nb += actual * actual;
                    error += (ref - actual) * (ref - actual);
                }
                const double cosine = dot / std::sqrt(na * nb), rel = std::sqrt(error / na);
                worst_cos = std::min(worst_cos, cosine);
                worst_rel = std::max(worst_rel, rel);
                if (!(cosine > .999 && rel < .02)) {
                    std::fprintf(stderr, "%s b%d t%d h%d: cosine %.6f relative L2 %.6f\n", sc.name,
                                 b, t, h, cosine, rel);
                    throw std::runtime_error("attention oracle mismatch");
                }
            }
    }
    std::printf("  %-27s cosine %.6f relative L2 %.6f ok\n", sc.name, worst_cos, worst_rel);
}
} // namespace

int main() {
    int failures = 0;
    try {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("device: %s\n", prop.name);
        const Scenario scenarios[]{
            {"w32 t1", 24, 1, 1, 32},
            {"w64 t1", 24, 1, 1, 64},
            {"w128 t1", 24, 1, 1, 128},
            {"w200 t1", 24, 1, 1, 200},
            {"w200 t6", 24, 1, 6, 200},
            {"w2560 t8", 24, 1, 8, 2560},
            {"chunked t13", 24, 1, 13, 400},
            {"b2 t6", 24, 2, 6, 200},
            {"b2 t16 chunked graph", 24, 2, 16, 2560, false, false, true},
            {"b3 masked chunks graph", 24, 3, 13, 200, true, false, true},
            {"H16 small t6", 16, 1, 6, 200},
            {"H16 b2 chunks", 16, 2, 13, 400},
            {"cached small t6", 24, 1, 6, 2560, false, true},
            {"H16 cached small", 16, 1, 6, 200, false, true},
            {"prompt t96 graph", 24, 1, 96, 300, false, false, true},
            {"prompt t6 short window", 24, 1, 6, 96},
            {"H16 prompt t96", 16, 1, 96, 300},
            {"cached prompt t96", 24, 1, 96, 300, false, true},
            // Residual-window scenarios: side planes + validity words through every route.
            {"residual w2560 t8", 24, 1, 8, 2560, false, false, false, true},
            {"residual ring 700 t6", 24, 1, 6, 700, false, false, false, true},
            {"residual b3 masked graph", 24, 3, 13, 200, true, false, true, true},
            {"H16 residual chunks", 16, 2, 13, 400, false, false, false, true},
            {"residual cached t6", 24, 1, 6, 2560, false, true, false, true},
            {"residual prompt t96", 24, 1, 96, 300, false, false, false, true},
            {"residual prompt graph", 24, 1, 96, 300, false, false, true, true},
            {"H16 residual prompt 1k", 16, 1, 96, 1024, false, false, false, true},
        };
        unsigned seed = 7;
        for (const auto& sc : scenarios) {
            try {
                run(sc, seed++);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "FAIL %s: %s\n", sc.name, error.what());
                ++failures;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf("%s hq attention conformance (%d failures)\n", failures ? "FAIL" : "OK", failures);
    return failures ? 1 : 0;
}

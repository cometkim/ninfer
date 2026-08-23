// Standalone qualification for the hq-e8-2b small-T decode kernel (decode/
// verify route). The kernel's partials are combined on the host and compared
// against an FP64 attention oracle computed over the device-decoded K/V rows,
// so the check covers exactly the kernel's own contracts: batch/column
// addressing, whole-window split coverage, causal masking, online softmax,
// neutral partials for inactive splits, the fused append, and independence
// from inherited shared-memory contents.
//
// Scenarios:
//   A. tokens=1, batch=1, column_begin=0 (plain decode regression)
//   B. tokens=3, batch=1 (multi-token rows; every valid token appended)
//   C. batch=2 with distinct block tables, tokens=1 (batch offsets)
//   D. chunked shape: full_width=12, width=6, column_begin=6 (keys before
//      column_begin must be attended; q/partial column offsets)
//   E. partial valid_columns (width 3, 2 valid)
//   F. determinism: re-run A after a kernel that trashes ~97 KB of shared
//      memory; outputs must be bit-identical
//   G. fused append: last `width` positions supplied through GqaAppendInput
//      instead of a pre-filled cache; outputs must match the same oracle
// Residual-window scenarios (WI-8): exact bf16 side rows for sink + recent
// positions, produced by the same dual-write helpers the engine uses.
//   K. window 200 > sink+recent: sink [0,32) and ring [72,200) read exact,
//      middle [32,72) codec; every fetch path exercised
//   L. append width 3 at window 100 (< sink+recent): fused append dual-writes
//      the new rows into the ring
//   M. one recent key's ring bit cleared: that key falls back to the codec
//      row while its neighbors stay exact
//   N. batch=2 residual with swapped table rows: each batch reads its own
//      slot's side planes
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ops/kernel/gqa_attention_decode_hq.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"

using namespace ninfer::ops;

namespace {

int g_failed = 0;

void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", msg);
        ++g_failed;
    }
}

constexpr int kQHeads  = Gqa27Geometry::QHeads;   // 24
constexpr int kKVHeads = Gqa27Geometry::KVHeads;  // 4
constexpr int kDim     = kGqaHeadDim;             // 256
constexpr int kGroup   = Gqa27Geometry::GroupSize;
constexpr int kPagePos = 64;
constexpr float kScale = 0.0625f;
constexpr int kSplits  = 16;

double bf16_bits_to_double(unsigned short bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short*>(&v) = bits;
    return static_cast<double>(__bfloat162float(v));
}

// Host mirror of the engine RHT diagonal (hq_engine_sign).
int hq_sign_host(int d) {
    std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    return (x & 1u) ? 1 : -1;
}

void host_fwht(double* v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                const double a = v[i + j];
                const double b = v[i + j + len];
                v[i + j]      = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

// Forward rotation (engine frame): signs, unnormalized FWHT, 1/16.
void host_rotate(double* v) {
    for (int d = 0; d < kDim; ++d) { v[d] *= hq_sign_host(d); }
    host_fwht(v, kDim);
    for (int d = 0; d < kDim; ++d) { v[d] /= 16.0; }
}

// Inverse rotation: unnormalized FWHT, then signs and 1/16.
void host_unrotate(double* v) {
    host_fwht(v, kDim);
    for (int d = 0; d < kDim; ++d) { v[d] = v[d] * hq_sign_host(d) / 16.0; }
}

__global__ void gen_gauss(float* out, unsigned long long seed, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    out[i] = curand_normal(&st);
}

__global__ void to_bf16(const float* in, __nv_bfloat16* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2bfloat16(in[i]); }
}

// Decode every (position, kv_head, role) row of a batch's visible window via
// the sequential row decoder: out[batch][pos][head][role][dim].
__global__ void decode_cache_rows(const std::uint8_t* codes_k, const std::uint8_t* codes_v,
                                  const std::uint8_t* meta_k, const std::uint8_t* meta_v,
                                  const std::int32_t* block_tables,
                                  const std::int32_t* table_rows, std::int32_t table_stride,
                                  std::int32_t window, std::int32_t batch_count,
                                  __nv_bfloat16* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int64_t units = static_cast<std::int64_t>(window) * kKVHeads * 2 * batch_count;
    if (i >= units) { return; }
    const int b   = static_cast<int>(i / (window * kKVHeads * 2));
    const int rem = static_cast<int>(i - static_cast<std::int64_t>(b) * window * kKVHeads * 2);
    const int pos = rem / (kKVHeads * 2);
    const int r2  = rem - pos * kKVHeads * 2;
    const int head = r2 >> 1;
    const bool role_v = (r2 & 1) != 0;
    const int trow = table_rows == nullptr ? 0 : table_rows[b];
    const std::int32_t* table =
        block_tables + static_cast<std::int64_t>(trow) * table_stride;
    __nv_bfloat16* dst =
        out + (static_cast<std::int64_t>(b) * window * kKVHeads * 2 + rem) * kDim;
    hq_decode_row_thread(
        hq_row_codes<Gqa27Geometry>(role_v ? codes_v : codes_k, table, head, pos),
        hq_row_meta<Gqa27Geometry>(role_v ? meta_v : meta_k, table, head, pos), dst,
        hq_dither_row_seed(head, pos, role_v));
}

// Fill dynamic shared memory with a garbage pattern (0xFF bytes = NaN
// floats) so the next kernel's dynamic smem starts with NaN leftovers — the
// l_smem read-before-init bug class.
__global__ void trash_smem() {
    extern __shared__ std::uint8_t s[];
    for (int i = static_cast<int>(threadIdx.x); i < 96 * 1024; i += blockDim.x) {
        s[i] = 0xFF;
    }
}

struct Scenario {
    const char* name;
    int batch;
    int full_width;
    int width;         // tokens per batch (invocation width)
    int column_begin;  // chunk offset within full_width
    int window;        // visible keys per batch (pos of last token + 1)
    int valid_columns; // -1: none; else absolute valid count
    bool append;       // last `width` positions supplied through GqaAppendInput
    bool trash;        // run twice with an smem trasher in between
    int capacity;      // -1: window; else the logical capacity passed to the
                       // kernel (smaller than window exercises the
                       // out-of-range neutral guard)
    bool residual;     // populate + consume the exact bf16 side planes
    int ring_clear;    // -1: all ring slots valid; else clear this key's bit
};

// Dual-write the exact (rotated) bf16 rows a residual scenario needs: one warp
// per (batch, position, kv_head, role), source = that batch's cache content,
// destination = the side-plane row of the table row the kernel will select.
__global__ void fill_residual_rows(const __nv_bfloat16* cache, std::int64_t cache_rows,
                                   const std::int32_t* table_rows, std::int32_t batch_count,
                                   std::int32_t fill_window, std::int32_t window_hint,
                                   __nv_bfloat16* residual_k, __nv_bfloat16* residual_v) {
    extern __shared__ std::int8_t signs_raw[];
    hq_engine_signs_fill(reinterpret_cast<std::int8_t*>(signs_raw));
    __syncthreads();
    const int warp = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const std::int64_t units =
        static_cast<std::int64_t>(fill_window) * kKVHeads * 2 * batch_count;
    if (warp >= units) { return; }
    const std::int64_t per_batch = static_cast<std::int64_t>(fill_window) * kKVHeads * 2;
    const int b    = static_cast<int>(warp / per_batch);
    const int rem  = static_cast<int>(warp - static_cast<std::int64_t>(b) * per_batch);
    const int pos  = rem / (kKVHeads * 2);
    const int r2   = rem - pos * kKVHeads * 2;
    const int head = r2 >> 1;
    const bool role_v = (r2 & 1) != 0;
    // Only the rows the fetch consumes (sink + current recent window); filling
    // mid-window rows would race the ring slots of later side rows.
    const int recent_from = window_hint - static_cast<int>(kGqaHqRecentKeys);
    if (pos >= static_cast<int>(kGqaHqSinkKeys) && pos < recent_from) { return; }
    const std::int32_t slot = table_rows[b];
    const __nv_bfloat16* src =
        cache + (static_cast<std::int64_t>(b) * cache_rows + static_cast<std::int64_t>(pos) * kKVHeads +
                 head) * kDim;
    hq_store_rotated_row_warp(
        src, reinterpret_cast<std::int8_t*>(signs_raw),
        hq_residual_row<Gqa27Geometry>(role_v ? residual_v : residual_k, slot, head, pos));
}

}  // namespace

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const std::vector<Scenario> scenarios = {
        {"A tokens=1 b=1",      1, 1,  1, 0, 200,  -1, false, false, -1, false, -1},
        {"B tokens=3 b=1",      1, 3,  3, 0, 200,  -1, false, false, -1, false, -1},
        {"C tokens=1 b=2",      2, 1,  1, 0, 150,  -1, false, false, -1, false, -1},
        {"D chunk col_begin=6", 1, 12, 6, 6, 120,  -1, false, false, -1, false, -1},
        {"E partial valid",     1, 3,  3, 0, 100,  2,  false, false, -1, false, -1},
        {"F smem trash",        1, 1,  1, 0, 200,  -1, false, true,  -1, false, -1},
        {"G append width=3",    1, 3,  3, 0, 100,  -1, true,  false, -1, false, -1},
        // A window large enough that every split runs many chunks, exercising
        // the online-softmax rescale chain across chunk boundaries.
        {"H window=2560",       1, 1,  1, 0, 2560, -1, false, false, -1, false, -1},
        // Full MTP-verify tile (width 6 -> 12 append units over 8 warps, two
        // rounds with per-warp scratch reuse) across two batches with
        // disjoint tables, so the append's offsets run together with batch.
        {"G2 append w6 b2",     2, 6,  6, 0, 150,  -1, true,  false, -1, false, -1},
        // logical_capacity below the last position: every split must write
        // neutral partials instead of indexing the block table out of range.
        {"I oob guard",         1, 1,  1, 0, 200,  -1, false, false, 199, false, -1},
        // Beyond the former 262144-key envelope (4,098 pages > 4,096; 16,388
        // keys per split at the test's 16 splits): pins the U8-only absolute
        // ceiling and page-table indices past the old linear boundary.
        {"J window=262208",     1, 1,  1, 0, 262208, -1, false, false, -1, false, -1},
        // Residual window (WI-8): sink + recent rows exact, middle codec.
        {"K residual w200",     1, 1,  1, 0, 200,  -1, false, false, -1, true,  -1},
        // Fused append dual-writes the three new rows into the ring.
        {"L residual append",   1, 3,  3, 0, 100,  -1, true,  false, -1, true,  -1},
        // One recent key's ring bit cleared: codec fallback for that key only.
        {"M residual bit=90",   1, 1,  1, 0, 200,  -1, false, false, -1, true,  90},
        // Two slots with swapped table rows: each batch reads its own plane.
        {"N residual b=2",      2, 1,  1, 0, 150,  -1, false, false, -1, true,  -1},
    };

    for (const auto& sc : scenarios) {
        // Disjoint injective page maps per batch: batch b maps position p to
        // b*half + (p % half), half >= window. With batch 2 the engine table
        // rows are swapped (table_rows = {1, 0}) so the kernel's
        // table_rows[batch] lookup selects the other row's memory.
        const int half = ((sc.window + kPagePos - 1) / kPagePos) * kPagePos;
        const int pages = half * sc.batch;
        std::vector<std::int32_t> table_rows(sc.batch);
        for (int b = 0; b < sc.batch; ++b) {
            table_rows[b] = (sc.batch == 2) ? (1 - b) : 0;
        }
        std::vector<std::int32_t> tables(static_cast<std::size_t>(sc.batch) * (sc.window + 1));
        for (int b = 0; b < sc.batch; ++b) {
            for (int p = 0; p < sc.window; ++p) {
                tables[static_cast<std::size_t>(table_rows[b]) * (sc.window + 1) + p] =
                    b * half + (p % half);
            }
        }

        // q and (for append scenarios) the new K/V rows, both laid out
        // [batch][full_width][heads][dim] in the invocation's full-width frame.
        const std::int64_t q_n =
            static_cast<std::int64_t>(sc.batch) * sc.full_width * kQHeads * kDim;
        const std::int64_t kv_n =
            static_cast<std::int64_t>(sc.batch) * sc.full_width * kKVHeads * kDim;
        std::vector<std::uint16_t> hq(q_n), hknew(kv_n), hvnew(kv_n);

        // Cache K/V rows for every visible position: [window][kv_heads][dim]
        // per role, encoded by the fill kernel.
        const std::int64_t cache_rows = static_cast<std::int64_t>(sc.window) * kKVHeads;
        const std::int64_t cache_n = cache_rows * 2 * kDim;

        const size_t code_bytes = static_cast<size_t>(pages) * kPagePos * kKVHeads * 64;
        const size_t meta_bytes = static_cast<size_t>(pages) * kPagePos * kKVHeads * 8;
        std::uint8_t *d_ck, *d_cv, *d_mk, *d_mv;
        __nv_bfloat16 *d_q, *d_knew, *d_vnew, *d_cache;
        std::int32_t *d_table, *d_pos0, *d_pos, *d_vc = nullptr;
        float* d_tmp;
        cudaMalloc(&d_ck, code_bytes);
        cudaMalloc(&d_cv, code_bytes);
        cudaMalloc(&d_mk, meta_bytes);
        cudaMalloc(&d_mv, meta_bytes);
        cudaMalloc(&d_q, q_n * 2);
        cudaMalloc(&d_knew, kv_n * 2);
        cudaMalloc(&d_vnew, kv_n * 2);
        cudaMalloc(&d_cache, cache_n * 2 * sc.batch);
        cudaMalloc(&d_table, tables.size() * 4);
        cudaMalloc(&d_pos0, 4);
        cudaMalloc(&d_pos, static_cast<std::size_t>(sc.batch) * sc.full_width * 4);
        cudaMalloc(&d_tmp, static_cast<size_t>(std::max(std::max(q_n, kv_n), cache_n)) * 4);
        cudaMemset(d_ck, 0, code_bytes);
        cudaMemset(d_cv, 0, code_bytes);
        cudaMemset(d_mk, 0, meta_bytes);
        cudaMemset(d_mv, 0, meta_bytes);
        cudaMemcpy(d_table, tables.data(), tables.size() * 4, cudaMemcpyHostToDevice);

        gen_gauss<<<static_cast<int>((q_n + 255) / 256), 256>>>(d_tmp, 7ull + sc.column_begin,
                                                       static_cast<int>(q_n));
        to_bf16<<<static_cast<int>((q_n + 255) / 256), 256>>>(d_tmp, d_q, static_cast<int>(q_n));
        gen_gauss<<<static_cast<int>((kv_n + 255) / 256), 256>>>(d_tmp, 21ull + sc.width,
                                                       static_cast<int>(kv_n));
        to_bf16<<<static_cast<int>((kv_n + 255) / 256), 256>>>(d_tmp, d_knew,
                                                      static_cast<int>(kv_n));
        gen_gauss<<<static_cast<int>((kv_n + 255) / 256), 256>>>(d_tmp, 22ull + sc.width,
                                                       static_cast<int>(kv_n));
        to_bf16<<<static_cast<int>((kv_n + 255) / 256), 256>>>(d_tmp, d_vnew,
                                                      static_cast<int>(kv_n));
        for (int b = 0; b < sc.batch; ++b) {
            gen_gauss<<<static_cast<int>((cache_n + 255) / 256), 256>>>(
                d_tmp, 99ull + sc.window + 7ull * b, static_cast<int>(cache_n));
            to_bf16<<<static_cast<int>((cache_n + 255) / 256), 256>>>(
                d_tmp, d_cache + static_cast<std::int64_t>(b) * cache_n,
                static_cast<int>(cache_n));
        }
        cudaDeviceSynchronize();
        {
            cudaError_t e = cudaGetLastError();
            check(e == cudaSuccess, "data-generation kernels failed");
        }
        cudaMemcpy(hq.data(), d_q, q_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hknew.data(), d_knew, kv_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hvnew.data(), d_vnew, kv_n * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_tmp);
        const std::int32_t zero = 0;
        cudaMemcpy(d_pos0, &zero, 4, cudaMemcpyHostToDevice);

        // positions: [batch][full_width] in the invocation frame; token t of
        // batch b (column column_begin + t) sits at window - width + t.
        std::vector<std::int32_t> pos(static_cast<std::size_t>(sc.batch) * sc.full_width, -1);
        for (int b = 0; b < sc.batch; ++b) {
            for (int t = 0; t < sc.width; ++t) {
                pos[static_cast<std::size_t>(b) * sc.full_width + sc.column_begin + t] =
                    sc.window - sc.width + t;
            }
        }
        cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
        if (sc.valid_columns >= 0) {
            std::vector<std::int32_t> vc(sc.batch, sc.valid_columns);
            cudaMalloc(&d_vc, vc.size() * 4);
            cudaMemcpy(d_vc, vc.data(), vc.size() * 4, cudaMemcpyHostToDevice);
        }

        // Fill the cache for every position; append scenarios leave the last
        // `width` positions to the decode kernel's fused append (the fill's
        // tokens bound covers only the prefix). Each batch fills through the
        // engine table row the kernel will select for it and from its own
        // cache content, so a wrong table lookup cannot pass by coincidence.
        std::int32_t* d_trows;
        cudaMalloc(&d_trows, table_rows.size() * 4);
        cudaMemcpy(d_trows, table_rows.data(), table_rows.size() * 4, cudaMemcpyHostToDevice);
        for (int b = 0; b < sc.batch; ++b) {
            GqaPrefillDirectMetadata meta{
                d_table + static_cast<std::size_t>(table_rows[b]) * (sc.window + 1)};
            const int fill_tokens = sc.append ? (sc.window - sc.width) : sc.window;
            __nv_bfloat16* src = d_cache + static_cast<std::int64_t>(b) * cache_n;
            gqa_attention_prefill_fill_hq_kernel<Gqa27Geometry, GqaPrefillDirectMetadata>
                <<<div_up(fill_tokens * kKVHeads * 2, kGqaHqFillWarps), kGqaHqFillWarps * 32,
                   kGqaHqFillSmemBytes>>>(src, src, d_pos0, meta, d_ck, d_cv, d_mk, d_mv,
                                          fill_tokens);
        }
        cudaDeviceSynchronize();

        // Residual side planes: [slots][160][KVHeads][256] per role, plus four
        // validity words per slot. Pre-fill every position the codec fill wrote
        // (append scenarios leave the tail to the decode kernel's dual-write).
        __nv_bfloat16 *d_rk = nullptr, *d_rv = nullptr;
        std::uint32_t* d_ring = nullptr;
        const int slots       = sc.batch;
        const std::int64_t side_plane =
            static_cast<std::int64_t>(kGqaHqSinkKeys + kGqaHqRecentKeys) * kKVHeads * kDim * slots;
        if (sc.residual) {
            cudaMalloc(&d_rk, side_plane * 2);
            cudaMalloc(&d_rv, side_plane * 2);
            const int ring_words = static_cast<int>(kGqaHqRecentKeys) / 32;
            cudaMalloc(&d_ring, static_cast<std::size_t>(slots) * ring_words * 4);
            cudaMemset(d_rk, 0xCD, side_plane * 2);
            cudaMemset(d_rv, 0xCD, side_plane * 2);
            cudaMemset(d_ring, 0xFF, static_cast<std::size_t>(slots) * ring_words * 4);
            if (sc.ring_clear >= 0) {
                std::vector<std::uint32_t> words(ring_words, 0xFFFFFFFFu);
                const int r = sc.ring_clear & (static_cast<int>(kGqaHqRecentKeys) - 1);
                words[r >> 5] &= ~(1u << (r & 31));
                cudaMemcpy(d_ring, words.data(),
                           static_cast<std::size_t>(ring_words) * 4, cudaMemcpyHostToDevice);
            }
            const int fill_window = sc.append ? (sc.window - sc.width) : sc.window;
            const std::int64_t units =
                static_cast<std::int64_t>(fill_window) * kKVHeads * 2 * sc.batch;
            fill_residual_rows<<<static_cast<int>((units + 7) / 8), 8 * 32, kDim>>>(
                d_cache, cache_n / kDim, d_trows, sc.batch, fill_window, sc.window, d_rk, d_rv);
            cudaDeviceSynchronize();
            {
                cudaError_t e = cudaGetLastError();
                check(e == cudaSuccess, "residual fill kernel failed");
            }
        }

        // ---- run the decode kernel ----------------------------------------
        const std::int64_t pacc_n = static_cast<std::int64_t>(sc.batch) * kQHeads * kDim *
                                    sc.width * kSplits;
        const std::int64_t pstat_n =
            static_cast<std::int64_t>(sc.batch) * kQHeads * sc.width * kSplits;
        __nv_bfloat16* d_pacc;
        float *d_pm, *d_pl;
        cudaMalloc(&d_pacc, pacc_n * 2);
        cudaMalloc(&d_pm, pstat_n * 4);
        cudaMalloc(&d_pl, pstat_n * 4);
        const dim3 grid(kKVHeads, kSplits, sc.batch);
        const GqaTcKVHq hq_cache{d_ck, d_cv, d_mk, d_mv, d_rk, d_rv, d_ring};
        const auto launch = [&]() {
            if (sc.append) {
                GqaAppendInput input{d_knew, d_vnew};
                gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 6, 4, true, true, GqaAppendInput, GqaTcKVHq>
                    <<<grid, kGqaHqDecodeThreads, 0>>>(
                        d_q, input, d_pos, hq_cache, d_table, d_vc, d_trows, sc.window + 1,
                        sc.width, sc.full_width, sc.column_begin,
                        sc.capacity >= 0 ? sc.capacity : sc.window, kScale, d_pacc, d_pm, d_pl);
            } else {
                GqaCachedInput no_append{};
                gqa_attention_small_t_tc_partial_bf16_kernel<Gqa27Geometry, 6, 4, true, true, GqaCachedInput, GqaTcKVHq>
                    <<<grid, kGqaHqDecodeThreads, 0>>>(
                        d_q, no_append, d_pos, hq_cache, d_table, d_vc, d_trows, sc.window + 1,
                        sc.width, sc.full_width, sc.column_begin,
                        sc.capacity >= 0 ? sc.capacity : sc.window, kScale, d_pacc, d_pm, d_pl);
            }
            cudaDeviceSynchronize();
        };
        std::vector<std::uint16_t> hpacc(pacc_n);
        std::vector<float> hpm(pstat_n), hpl(pstat_n);
        launch();
        if (sc.trash) {
            cudaMemcpy(hpacc.data(), d_pacc, pacc_n * 2, cudaMemcpyDeviceToHost);
            cudaFuncSetAttribute(trash_smem, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 96 * 1024);
            trash_smem<<<1, 256, 96 * 1024>>>();
            cudaDeviceSynchronize();
            cudaMemset(d_pacc, 0xCD, pacc_n * 2);
            launch();
            std::vector<std::uint16_t> second(pacc_n);
            cudaMemcpy(second.data(), d_pacc, pacc_n * 2, cudaMemcpyDeviceToHost);
            std::int64_t diff = 0;
            for (std::int64_t i = 0; i < pacc_n; ++i) {
                if (hpacc[i] != second[i]) { ++diff; }
            }
            std::printf("[%s] partial words differing across smem trash: %lld\n", sc.name,
                        static_cast<long long>(diff));
            check(diff == 0, "decode output depends on inherited shared-memory contents");
        }
        cudaMemcpy(hpacc.data(), d_pacc, pacc_n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hpm.data(), d_pm, pstat_n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hpl.data(), d_pl, pstat_n * 4, cudaMemcpyDeviceToHost);

        if (sc.capacity >= 0 && sc.capacity < sc.window) {
            // Out-of-range guard: every (head, token, split) partial must be
            // neutral, not a block-table read at an out-of-range position.
            int nonneutral = 0;
            for (std::int64_t i = 0; i < pstat_n; ++i) {
                if (hpm[i] != -INFINITY || hpl[i] != 0.0f) { ++nonneutral; }
            }
            std::printf("[%s] non-neutral partials under out-of-range positions: %d\n",
                        sc.name, nonneutral);
            check(nonneutral == 0, "out-of-range positions did not produce neutral partials");
        } else {
            // ---- FP64 oracle over the device-decoded cache ---------------------
            const std::int64_t dec_n =
                static_cast<std::int64_t>(sc.batch) * sc.window * kKVHeads * 2 * kDim;
            __nv_bfloat16* d_dec;
            cudaMalloc(&d_dec, dec_n * 2);
            decode_cache_rows<<<static_cast<int>((dec_n / kDim + 255) / 256), 256>>>(
                d_ck, d_cv, d_mk, d_mv, d_table, d_trows, sc.window + 1, sc.window, sc.batch, d_dec);
            cudaDeviceSynchronize();
            std::vector<std::uint16_t> hdec(dec_n);
            cudaMemcpy(hdec.data(), d_dec, dec_n * 2, cudaMemcpyDeviceToHost);

            if (sc.residual) {
                // Substitute the exact side-plane rows (the bits the kernel's
                // fetch reads) for the decoded rows at sink/recent positions.
                std::vector<std::uint16_t> hrk(side_plane), hrv(side_plane);
                cudaMemcpy(hrk.data(), d_rk, side_plane * 2, cudaMemcpyDeviceToHost);
                cudaMemcpy(hrv.data(), d_rv, side_plane * 2, cudaMemcpyDeviceToHost);
                const int ring_words = static_cast<int>(kGqaHqRecentKeys) / 32;
                std::vector<std::uint32_t> hring(
                    static_cast<std::size_t>(slots) * ring_words);
                cudaMemcpy(hring.data(), d_ring,
                           static_cast<std::size_t>(slots) * ring_words * 4,
                           cudaMemcpyDeviceToHost);
                const int sink = static_cast<int>(kGqaHqSinkKeys);
                const int ring = static_cast<int>(kGqaHqRecentKeys);
                for (int b = 0; b < sc.batch; ++b) {
                    const int slot = table_rows[b];
                    for (int p = 0; p < sc.window; ++p) {
                        const bool recent = p >= sc.window - ring;
                        const bool bit =
                            recent && ((hring[static_cast<std::size_t>(slot) * ring_words +
                                               ((p & (ring - 1)) >> 5)] >>
                                        ((p & (ring - 1)) & 31)) &
                                       1u) != 0;
                        if (p >= sink && !bit) { continue; }
                        const int row = p < sink ? p : sink + (p & (ring - 1));
                        for (int h = 0; h < kKVHeads; ++h) {
                            for (int role = 0; role < 2; ++role) {
                                const std::size_t src_off =
                                    (static_cast<std::size_t>(slot) * (sink + ring) * kKVHeads +
                                     static_cast<std::size_t>(row) * kKVHeads + h) * kDim;
                                const std::uint16_t* srcp = role ? &hrv[src_off] : &hrk[src_off];
                                std::uint16_t* dst =
                                    &hdec[((static_cast<std::size_t>(b) * sc.window + p) *
                                               kKVHeads * 2 +
                                           h * 2 + role) * kDim];
                                std::memcpy(dst, srcp, kDim * 2);
                            }
                        }
                    }
                }
                // Rotation spot check: a side row must equal the FWHT-rotated
                // source row (fp64 oracle; the device stages in fp32, so one
                // cosine gate with margin covers the rounding difference).
                // Append tails are written from the append inputs, not the
                // pre-fill cache content, so only pre-filled rows are compared.
                std::vector<std::uint16_t> hcache(cache_n * sc.batch);
                cudaMemcpy(hcache.data(), d_cache, cache_n * 2 * sc.batch,
                           cudaMemcpyDeviceToHost);
                const int fill_window = sc.append ? (sc.window - sc.width) : sc.window;
                double min_side_cos = 1.0;
                for (int b = 0; b < sc.batch; ++b) {
                    const int slot = table_rows[b];
                    for (int p = 0; p < fill_window; p += 7) {
                        const int row = p < sink ? p : sink + (p & (ring - 1));
                        if (p >= sink && !(p >= sc.window - ring)) { continue; }
                        for (int h = 0; h < kKVHeads; ++h) {
                            std::vector<double> rot(kDim);
                            for (int d = 0; d < kDim; ++d) {
                                rot[d] = bf16_bits_to_double(
                                    hcache[(static_cast<std::size_t>(b) * cache_n +
                                            (static_cast<std::size_t>(p) * kKVHeads + h) * kDim +
                                            d)]);
                            }
                            host_rotate(rot.data());
                            double xy = 0, xx = 0, yy = 0;
                            for (int d = 0; d < kDim; ++d) {
                                const double s = bf16_bits_to_double(
                                    hrk[(static_cast<std::size_t>(slot) * (sink + ring) * kKVHeads +
                                         static_cast<std::size_t>(row) * kKVHeads + h) * kDim + d]);
                                xy += rot[d] * s;
                                xx += rot[d] * rot[d];
                                yy += s * s;
                            }
                            const double cos = xy / (std::sqrt(xx * yy) + 1e-300);
                            if (cos < 0.9 && g_failed < 40) {
                                std::printf("    [diag] slot=%d p=%d row=%d h=%d cos=%.4f "
                                            "src0=%04x plane0=%04x\n",
                                            slot, p, row, h, cos,
                                            hcache[(static_cast<std::size_t>(b) * cache_n +
                                                    (static_cast<std::size_t>(p) * kKVHeads + h) *
                                                        kDim)],
                                            hrk[(static_cast<std::size_t>(slot) * (sink + ring) *
                                                     kKVHeads +
                                                 static_cast<std::size_t>(row) * kKVHeads + h) *
                                                    kDim]);
                            }
                            min_side_cos = std::min(min_side_cos, cos);
                        }
                    }
                }
                std::printf("[%s] side-row rotation min cos %.6f\n", sc.name, min_side_cos);
                check(min_side_cos > 0.9999, "side rows do not match the rotated source rows");
            }

            double min_cos = 1.0, max_rel = 0.0;
            int rows_checked = 0;
            for (int b = 0; b < sc.batch; ++b) {
                const int valid_width =
                    sc.valid_columns >= 0 ? (sc.valid_columns - sc.column_begin) : sc.width;
                for (int t = 0; t < valid_width; ++t) {
                    const int pos_t = sc.window - sc.width + t;
                    for (int h = 0; h < kQHeads; ++h) {
                        // Oracle (rotated-frame profile, like the kernel): scores
                        // of the rotated q row against the decoded (rotated) K
                        // rows; FP64 softmax; one un-rotation of the output.
                        const std::uint16_t* qrow =
                            &hq[((static_cast<std::size_t>(b) * sc.full_width + sc.column_begin + t) *
                                     kQHeads + h) * kDim];
                        std::vector<double> qrot(kDim);
                        for (int d = 0; d < kDim; ++d) {
                            qrot[d] = bf16_bits_to_double(qrow[d]);
                        }
                        host_rotate(qrot.data());
                        const int kvh = h / kGroup;
                        std::vector<double> s(pos_t + 1);
                        double m = -1e300;
                        for (int p = 0; p <= pos_t; ++p) {
                            const std::uint16_t* krow =
                                &hdec[((static_cast<std::size_t>(b) * sc.window + p) * kKVHeads * 2 +
                                       static_cast<std::size_t>(kvh) * 2) * kDim];
                            double dot = 0;
                            for (int d = 0; d < kDim; ++d) {
                                dot += qrot[d] * bf16_bits_to_double(krow[d]);
                            }
                            s[p] = dot * kScale;
                            m = m > s[p] ? m : s[p];
                        }
                        double l = 0;
                        for (int p = 0; p <= pos_t; ++p) { l += std::exp(s[p] - m); }
                        std::vector<double> o(kDim, 0.0);
                        for (int p = 0; p <= pos_t; ++p) {
                            const double w = std::exp(s[p] - m) / l;
                            const std::uint16_t* vrow =
                                &hdec[((static_cast<std::size_t>(b) * sc.window + p) * kKVHeads * 2 +
                                       static_cast<std::size_t>(kvh) * 2 + 1) * kDim];
                            for (int d = 0; d < kDim; ++d) { o[d] += w * bf16_bits_to_double(vrow[d]); }
                        }
                        host_unrotate(o.data());
                        // Combine this row's partials (m/l/acc, batch-major).
                        // Layout: head + QHeads*(token + tokens*split) per batch.
                        const std::int64_t stat_base =
                            static_cast<std::int64_t>(b) * kQHeads * sc.width * kSplits +
                            h + kQHeads * t + kQHeads * sc.width * 0;
                        const std::int64_t acc_base =
                            static_cast<std::int64_t>(b) * kQHeads * kDim * sc.width * kSplits +
                            (h + kQHeads * t) * kDim;
                        double gm = -1e300;
                        for (int sp = 0; sp < kSplits; ++sp) {
                            const std::int64_t si = stat_base +
                                                    static_cast<std::int64_t>(kQHeads) * sc.width * sp;
                            gm = gm > hpm[si] ? gm : hpm[si];
                        }
                        double gl = 0;
                        std::vector<double> g(kDim, 0.0);
                        for (int sp = 0; sp < kSplits; ++sp) {
                            const std::int64_t si = stat_base +
                                                    static_cast<std::int64_t>(kQHeads) * sc.width * sp;
                            if (hpm[si] == -INFINITY) { continue; }
                            const double w = std::exp(hpm[si] - gm);
                            gl += hpl[si] * w;
                            const std::int64_t ai = acc_base +
                                                    static_cast<std::int64_t>(kQHeads) * sc.width *
                                                        kDim * sp;
                            for (int d = 0; d < kDim; ++d) {
                                g[d] += bf16_bits_to_double(hpacc[ai + d]) * w;
                            }
                        }
                        for (int d = 0; d < kDim; ++d) { g[d] /= gl; }
                        double xy = 0, xx = 0, yy = 0, rel = 0, on = 0;
                        for (int d = 0; d < kDim; ++d) {
                            xy += o[d] * g[d];
                            xx += o[d] * o[d];
                            yy += g[d] * g[d];
                            rel += (g[d] - o[d]) * (g[d] - o[d]);
                            on += o[d] * o[d];
                        }
                        const double cos = xy / (std::sqrt(xx * yy) + 1e-300);
                        min_cos = min_cos < cos ? min_cos : cos;
                        max_rel = max_rel > std::sqrt(rel / (on + 1e-300))
                                      ? max_rel
                                      : std::sqrt(rel / (on + 1e-300));
                        ++rows_checked;
                    }
                }
            }
            std::printf("[%s] rows %d, min cos %.6f, max row rel %.4f%%\n", sc.name, rows_checked,
                        min_cos, max_rel * 100.0);
            check(rows_checked > 0, "no rows checked");
            check(min_cos > 0.9999, "combined partials diverge from the FP64 oracle");
            check(max_rel < 0.03, "combined partial row error exceeds 3%");

            cudaFree(d_dec);
        }
        cudaFree(d_pacc);
        cudaFree(d_pm);
        cudaFree(d_pl);
        cudaFree(d_ck);
        cudaFree(d_cv);
        cudaFree(d_mk);
        cudaFree(d_mv);
        cudaFree(d_q);
        cudaFree(d_knew);
        cudaFree(d_vnew);
        cudaFree(d_cache);
        cudaFree(d_table);
        cudaFree(d_trows);
        if (d_rk) { cudaFree(d_rk); }
        if (d_rv) { cudaFree(d_rv); }
        if (d_ring) { cudaFree(d_ring); }
        cudaFree(d_pos0);
        cudaFree(d_pos);
        if (d_vc) { cudaFree(d_vc); }
    }

    if (g_failed != 0) {
        std::fprintf(stderr, "test_hq_decode: %d failures\n", g_failed);
        return EXIT_FAILURE;
    }
    std::printf("test_hq_decode: ALL PASSED\n");
    return 0;
}

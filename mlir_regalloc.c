// Shared, target-parameterized register allocator (global linear-scan +
// block-local). Extracted from mlir_llvm_to_aarch64.c; see mlir_regalloc.h.
//
// Target independence: the allocators operate on the `llvm` dialect's values +
// liveness and produce a value->physical-register map (`rm`). The register pool
// is supplied as the data-only RegTarget; the fold/fusion behaviour (which ops
// emit no code / read custom register operands) is supplied as plain SlotMap
// pointers in the target. The fold/const/cast hooks below are provided by the
// active backend (aarch64 today); an x64 caller passes NULL fusion maps so only
// the generic constant/cast checks fire. No function pointers are used, keeping
// this file parseable by the tinyC self-host.

#include "mlir_regalloc.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

// Fold/const/cast hooks supplied by the active backend (mlir_llvm_to_aarch64.c).
// x64 reuses const_int_val / cast_src (generic) and passes NULL fusion maps.
extern bool const_int_val(MLIR_Context *ctx, MLIR_OpHandle op, int64_t *val, uint8_t *is64);
extern bool cast_src(MLIR_OpHandle op, MLIR_ValueHandle *src);
extern bool a64_is_spine(MLIR_OpHandle op, SlotMap *memspine);
extern bool a64_memfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *memfuse,
                             MLIR_ValueHandle *base, MLIR_ValueHandle *idx);
extern bool a64_shiftfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *shiftfuse,
                               MLIR_ValueHandle *xv);

static bool name_eq(string s, const char *cstr) {
    size_t l = strlen(cstr);
    return s.size == l && memcmp(s.str, cstr, l) == 0;
}

// ---------------------------------------------------------------------------
// SlotMap (value handle -> int32 slot/register/count).
// ---------------------------------------------------------------------------

static size_t sm_hash(uintptr_t k) {
    k *= 0x9E3779B97F4A7C15ull;
    return (size_t)(k >> 32);
}
static void sm_grow(SlotMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    SlotEnt *nt = (SlotEnt *)arena_alloc(m->arena, ncap * sizeof(SlotEnt));
    memset(nt, 0, ncap * sizeof(SlotEnt));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->t[i].key == 0) continue;
        size_t j = sm_hash(m->t[i].key) & (ncap - 1);
        while (nt[j].key != 0) j = (j + 1) & (ncap - 1);
        nt[j] = m->t[i];
    }
    m->t = nt;
    m->cap = ncap;
}
void sm_put(SlotMap *m, MLIR_ValueHandle k, int32_t slot) {
    if ((m->n + 1) * 4 >= m->cap * 3) sm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) return;
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].slot = slot;
    m->n++;
}
bool sm_get(SlotMap *m, MLIR_ValueHandle k, int32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) { *out = m->t[i].slot; return true; }
        i = (i + 1) & mask;
    }
    return false;
}

// Increment the use count of value `k` (used to build a use-count map keyed by
// SSA value handle, reusing the SlotMap storage with `slot` = count).
void uc_inc(SlotMap *m, MLIR_ValueHandle k) {
    if (!k) return;
    if ((m->n + 1) * 4 >= m->cap * 3) sm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) { m->t[i].slot++; return; }
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].slot = 1;
    m->n++;
}

// ---------------------------------------------------------------------------
// Liveness bitset + open-addressing handle hash.
// ---------------------------------------------------------------------------
static inline size_t a64_bs_words(size_t n) { return (n + 63u) / 64u; }
static inline void a64_bs_set(uint64_t *bs, uint32_t i) { bs[i >> 6] |= (1ull << (i & 63u)); }
static inline void a64_bs_clear(uint64_t *bs, uint32_t i) { bs[i >> 6] &= ~(1ull << (i & 63u)); }
static inline bool a64_bs_test(const uint64_t *bs, uint32_t i) { return (bs[i >> 6] >> (i & 63u)) & 1u; }
static inline void a64_bs_zero(uint64_t *bs, size_t n) { memset(bs, 0, a64_bs_words(n) * 8u); }
static inline bool a64_bs_equal(const uint64_t *a, const uint64_t *b, size_t n) {
    return memcmp(a, b, a64_bs_words(n) * 8u) == 0;
}
static inline void a64_bs_or(uint64_t *d, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t w = a64_bs_words(n); for (size_t i = 0; i < w; i++) d[i] = a[i] | b[i];
}
static inline void a64_bs_diff(uint64_t *d, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t w = a64_bs_words(n); for (size_t i = 0; i < w; i++) d[i] = a[i] & ~b[i];
}

// Open-addressing handle -> uint32 hash (malloc-backed, pre-sized, no grow).
// Key 0 (MLIR_INVALID_HANDLE) marks empty; handles are non-zero. Probe order
// is irrelevant to output determinism (only get/set by key are used).
typedef struct { MLIR_ValueHandle *k; uint32_t *v; size_t cap; } A64HHash;
static uint32_t a64_hh_hash(MLIR_ValueHandle v) {
    uint32_t h = (uint32_t)v ^ (uint32_t)(v >> 16);
    h ^= h >> 15; h *= 2654435761u; h ^= h >> 13; return h;
}
static void a64_hh_init(A64HHash *m, size_t want) {
    size_t c = 16; while (c < want * 2u) c *= 2;
    m->cap = c;
    m->k = (MLIR_ValueHandle *)calloc(c, sizeof(*m->k));
    m->v = (uint32_t *)malloc(c * sizeof(*m->v));
}
static void a64_hh_put(A64HHash *m, MLIR_ValueHandle key, uint32_t val) {
    size_t mask = m->cap - 1, p = (size_t)a64_hh_hash(key) & mask;
    while (m->k[p] != 0) { if (m->k[p] == key) { m->v[p] = val; return; } p = (p + 1) & mask; }
    m->k[p] = key; m->v[p] = val;
}
static bool a64_hh_get(const A64HHash *m, MLIR_ValueHandle key, uint32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1, p = (size_t)a64_hh_hash(key) & mask;
    while (m->k[p] != 0) { if (m->k[p] == key) { *out = m->v[p]; return true; } p = (p + 1) & mask; }
    return false;
}
static void a64_hh_free(A64HHash *m) { free(m->k); free(m->v); m->k = NULL; m->v = NULL; m->cap = 0; }

// ---------------------------------------------------------------------------
// Fold/const/fusion queries (dispatch to the backend-supplied hooks).
// ---------------------------------------------------------------------------
static inline bool ra_is_const(const RegTarget *T, MLIR_Context *c, MLIR_OpHandle o) {
    (void)T; int64_t v; uint8_t b; return const_int_val(c, o, &v, &b);
}
static inline bool ra_is_folded(const RegTarget *T, MLIR_Context *c, MLIR_OpHandle o) {
    (void)c; return a64_is_spine(o, T->memspine);
}
static inline bool ra_memfuse(const RegTarget *T, MLIR_Context *c, MLIR_OpHandle o,
                              MLIR_ValueHandle *b, MLIR_ValueHandle *i) {
    return a64_memfuse_uses(c, o, T->memfuse, b, i);
}
static inline bool ra_shiftfuse(const RegTarget *T, MLIR_Context *c, MLIR_OpHandle o,
                                MLIR_ValueHandle *x) {
    return a64_shiftfuse_uses(c, o, T->shiftfuse, x);
}

// ===========================================================================
// Global (whole-function) linear-scan allocator + block-local fallback.
// ===========================================================================
uint32_t mlir_regalloc_global(MLIR_Context *ctx, MLIR_RegionHandle reg,
                                  size_t n_blocks, SlotMap *rm,
                                  SlotMap *sm, int32_t *out_nslots,
                                  const RegTarget *T) {
    *out_nslots = 0;
    if (n_blocks == 0) return RA_BAIL;

    // --- 0. Count register-candidate values; bail on nested regions. Entry
    //        block args (incoming params) ARE allocated like any other value:
    //        the prologue moves them from x0..x7 (or the caller's stack) into
    //        their assigned home register, so a param used throughout the
    //        function (e.g. a parser/lexer state pointer) stays resident instead
    //        of being reloaded from its slot at every use. Param homing is in
    //        the existing pool (x12..x28), none of which alias the incoming
    //        x0..x7, so the prologue move is a safe sequential copy (no parallel
    //        move needed). Disable via TINYC_NO_PARAM_HOME for A/B measurement.
    //        Constants and folded spine results get no value number
    //        (rematerialized / fused). Positions still advance per op. ---
    bool home_params = !getenv("TINYC_NO_PARAM_HOME");
    size_t nv_max = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        if (b > 0 || home_params) nv_max += MLIR_GetBlockNumArgs(bk);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (MLIR_GetOpNumRegions(op) > 0) return RA_BAIL;
            nv_max += MLIR_GetOpNumResults(op);
        }
    }
    if (nv_max == 0) return 0;
    // Memory gate: bound the liveness bitsets (n_blocks * bw words, four of
    // them) so a truly pathological function falls back to block-local rather
    // than exhausting the wasm32 heap during self-host (stage1 runs this backend
    // under wasmtime). The deleted WMIR allocator ran the SAME O(n_blocks*bw)
    // liveness with NO gate and self-hosted, so these thresholds are set high
    // enough that every real tinyC function (largest ~2628 blocks / ~60K values
    // / ~60MB peak, freed per function) gets global allocation + slot reuse.
    {
        size_t bwg = a64_bs_words(nv_max);
        if (nv_max > 100000u || (uint64_t)n_blocks * (uint64_t)bwg > 4000000ull)
            return RA_BAIL;
    }

    // Half-open live intervals. With one integer position per op, an operand's
    // last read and a result's def land on the SAME position, so the strict
    // active-list retirement (last_use < first_pos) cannot retire the dying
    // operand before the consumer is allocated -- they falsely overlap and
    // demand two registers for what is logically one value (the induction
    // `%next = %a + 1` / loop-carried case). Numbering operand reads at 2g+1,
    // result defs at 2g+2, and block args at 2g makes a value dying at op g
    // retire (2g+1 < 2g+2) before the same-op result, so the consumer reuses
    // its register; and a block arg (2g) sits strictly before its block's first
    // op read (2g+1) so a block arg consumed by the first op is no longer
    // falsely flagged dead-on-def. Safe because every aarch64 op this backend
    // emits reads all source registers before writing its result register.
    // TINYC_NO_HALFOPEN restores integer positions.
    bool halfopen = (getenv("TINYC_NO_HALFOPEN") == NULL);
    #define A64_USEP(g)  (halfopen ? (2u * (uint32_t)(g) + 1u) : (uint32_t)(g))
    #define A64_DEFP(g)  (halfopen ? (2u * (uint32_t)(g) + 2u) : (uint32_t)(g))
    #define A64_BARGP(g) (halfopen ? (2u * (uint32_t)(g))      : (uint32_t)(g))
    #define A64_COVL(g)  (halfopen ? (2u * (uint32_t)(g) + 1u) : (uint32_t)(g))
    #define A64_COVF(g)  (halfopen ? (2u * (uint32_t)(g))      : (uint32_t)(g))

    // --- 1. Linearise + number values. One position per op (incl. const/spine
    //        ops, so is_call[] and use positions stay consistent). ---
    uint32_t *block_first = (uint32_t *)malloc(n_blocks * sizeof(uint32_t));
    uint32_t *block_last  = (uint32_t *)malloc(n_blocks * sizeof(uint32_t));
    MLIR_ValueHandle *handle = (MLIR_ValueHandle *)malloc(nv_max * sizeof(MLIR_ValueHandle));
    uint32_t *def_pos   = (uint32_t *)malloc(nv_max * sizeof(uint32_t));
    uint32_t *first_pos = (uint32_t *)malloc(nv_max * sizeof(uint32_t));
    uint32_t *last_use  = (uint32_t *)malloc(nv_max * sizeof(uint32_t));
    uint64_t *weight    = (uint64_t *)malloc(nv_max * sizeof(uint64_t));
    uint8_t  *crosses   = (uint8_t  *)malloc(nv_max * sizeof(uint8_t));
    int8_t   *home_pk   = (int8_t   *)malloc(nv_max * sizeof(int8_t));
    A64HHash vmap; a64_hh_init(&vmap, nv_max);
    A64HHash bmap; a64_hh_init(&bmap, n_blocks);

    for (size_t b = 0; b < n_blocks; b++)
        a64_hh_put(&bmap, (MLIR_ValueHandle)MLIR_GetRegionBlock(reg, b), (uint32_t)b);

    uint32_t pos = 0, nv = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        block_first[b] = pos;
        if (b > 0 || home_params) {
            size_t na = MLIR_GetBlockNumArgs(bk);
            for (size_t a = 0; a < na; a++) {
                MLIR_ValueHandle arg = MLIR_GetBlockArg(bk, a);
                handle[nv] = arg; def_pos[nv] = A64_BARGP(pos); first_pos[nv] = A64_BARGP(pos);
                last_use[nv] = 0; weight[nv] = 0; crosses[nv] = 0; home_pk[nv] = -1;
                a64_hh_put(&vmap, arg, nv); nv++;
            }
        }
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            bool skip = ra_is_const(T, ctx, op) || ra_is_folded(T, ctx, op);
            if (!skip) {
                size_t nr = MLIR_GetOpNumResults(op);
                for (size_t r = 0; r < nr; r++) {
                    MLIR_ValueHandle rv = MLIR_GetOpResult(op, r);
                    handle[nv] = rv; def_pos[nv] = A64_DEFP(pos); first_pos[nv] = A64_DEFP(pos);
                    last_use[nv] = 0; weight[nv] = 0; crosses[nv] = 0; home_pk[nv] = -1;
                    a64_hh_put(&vmap, rv, nv); nv++;
                }
            }
            pos++;
        }
        block_last[b] = pos;
    }

    // --- 1b. Per-block loop depth via back-edge difference array. ---
    uint32_t *loop_depth = (uint32_t *)calloc(n_blocks, sizeof(uint32_t));
    {
        int64_t *delta = (int64_t *)calloc(n_blocks + 1, sizeof(int64_t));
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    uint32_t ti;
                    if (a64_hh_get(&bmap, (MLIR_ValueHandle)MLIR_GetOpSuccessor(op, s), &ti) &&
                        ti <= (uint32_t)b) { delta[ti] += 1; delta[b + 1] -= 1; }
                }
            }
        }
        int64_t acc = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            acc += delta[b]; loop_depth[b] = acc < 0 ? 0 : (uint32_t)acc;
        }
        free(delta);
    }
    #define A64_USE_WEIGHT_AT(d) (1ull << (((d) * 3u) < 60u ? ((d) * 3u) : 60u))

    // --- 2. Liveness dataflow: gen/kill then live_in/live_out fixpoint. ---
    size_t bw = a64_bs_words(nv ? nv : 1);
    uint64_t *live_in  = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *live_out = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *gen      = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *kill     = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *scratch  = (uint64_t *)calloc(bw, sizeof(uint64_t));

    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        uint64_t *G = gen + b * bw, *K = kill + b * bw;
        if (b > 0) {
            size_t na = MLIR_GetBlockNumArgs(bk);
            for (size_t a = 0; a < na; a++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetBlockArg(bk, a), &vi)) a64_bs_set(K, vi);
            }
        }
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (ra_is_folded(T, ctx, op)) continue;
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi) && !a64_bs_test(K, vi))
                    a64_bs_set(G, vi);
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi) &&
                        !a64_bs_test(K, vi)) a64_bs_set(G, vi);
                }
            }
            MLIR_ValueHandle mbase, midx;
            if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                MLIR_ValueHandle uu[2] = { mbase, midx };
                for (int u = 0; u < 2; u++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, uu[u], &vi) && !a64_bs_test(K, vi)) a64_bs_set(G, vi);
                }
            }
            MLIR_ValueHandle sfx;
            if (ra_shiftfuse(T, ctx, op, &sfx)) {
                uint32_t vi;
                if (a64_hh_get(&vmap, sfx, &vi) && !a64_bs_test(K, vi)) a64_bs_set(G, vi);
            }
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t r = 0; r < nr; r++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetOpResult(op, r), &vi)) a64_bs_set(K, vi);
            }
        }
    }

    bool changed = true; int iters = 0;
    while (changed && iters++ < 4096) {
        changed = false;
        for (size_t br = 0; br < n_blocks; br++) {
            size_t b = n_blocks - 1 - br;
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            uint64_t *LO = live_out + b * bw;
            a64_bs_zero(LO, nv);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    uint32_t si;
                    if (a64_hh_get(&bmap, (MLIR_ValueHandle)MLIR_GetOpSuccessor(op, s), &si))
                        a64_bs_or(LO, LO, live_in + si * bw, nv);
                }
            }
            uint64_t *LI = live_in + b * bw;
            a64_bs_diff(scratch, LO, kill + b * bw, nv);
            a64_bs_or(scratch, scratch, gen + b * bw, nv);
            if (!a64_bs_equal(scratch, LI, nv)) {
                memcpy(LI, scratch, bw * sizeof(uint64_t)); changed = true;
            }
        }
    }

    // --- 3. Per-value first_pos / last_use_pos / use_weight. ---
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        uint64_t *LO = live_out + b * bw, *LI = live_in + b * bw;
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (a64_bs_test(LO, vi) && A64_COVL(block_last[b]) > last_use[vi])  last_use[vi]  = A64_COVL(block_last[b]);
            if (a64_bs_test(LI, vi) && A64_COVF(block_first[b]) < first_pos[vi]) first_pos[vi] = A64_COVF(block_first[b]);
        }
        uint32_t op_pos = block_first[b];
        uint64_t w = A64_USE_WEIGHT_AT(loop_depth[b]);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (ra_is_folded(T, ctx, op)) { op_pos++; continue; }
            uint32_t up = A64_USEP(op_pos);
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += w;
                }
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                MLIR_BlockHandle succ = MLIR_GetOpSuccessor(op, s);
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi)) {
                        if (up > last_use[vi]) last_use[vi] = up;
                        if (up < first_pos[vi]) first_pos[vi] = up;
                        weight[vi] += w;
                    }
                    // The branch writes the k-th successor block arg's home at
                    // THIS position; extend that arg's interval start to cover
                    // the write (forward edges define the arg before its own
                    // block-entry position).
                    uint32_t avi;
                    if (a64_hh_get(&vmap, MLIR_GetBlockArg(succ, k), &avi) &&
                        up < first_pos[avi]) first_pos[avi] = up;
                }
            }
            MLIR_ValueHandle mbase, midx;
            if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                MLIR_ValueHandle uu[2] = { mbase, midx };
                for (int u = 0; u < 2; u++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, uu[u], &vi)) {
                        if (up > last_use[vi]) last_use[vi] = up;
                        if (up < first_pos[vi]) first_pos[vi] = up;
                        weight[vi] += w;
                    }
                }
            }
            MLIR_ValueHandle sfx;
            if (ra_shiftfuse(T, ctx, op, &sfx)) {
                uint32_t vi;
                if (a64_hh_get(&vmap, sfx, &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += w;
                }
            }
            op_pos++;
        }
    }

    // crosses_call: a value must live in a callee-saved register iff it is live
    // across a call -- i.e. live immediately AFTER some llvm.call and not that
    // call's own result (the result is born at the call, it does not cross it).
    // This is computed PRECISELY from the per-block live_out by a backward walk:
    // `live_out(call) \ {result}` is exactly the set of values live across the
    // call, which is sound (a caller-saved reg live across a call is clobbered)
    // and exact. The legacy linear first_pos..last_use span (kept under
    // TINYC_LINEAR_CROSSES for A/B) over-approximates across mutually-exclusive
    // CFG paths, falsely forcing phi/block-arg values into the 9 callee-saved
    // registers and inflating spill pressure.
    if (getenv("TINYC_LINEAR_CROSSES")) {
        uint8_t *is_call = (uint8_t *)calloc(pos ? pos : 1, sizeof(uint8_t));
        uint32_t op_pos = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                if (name_eq(MLIR_GetOpName(MLIR_GetBlockOp(bk, i)), "llvm.call"))
                    is_call[op_pos] = 1;
                op_pos++;
            }
        }
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (last_use[vi] < first_pos[vi]) continue;
            uint32_t lo = halfopen ? first_pos[vi] / 2u : first_pos[vi];
            uint32_t hi = halfopen ? last_use[vi]  / 2u : last_use[vi];
            for (uint32_t g = lo; g <= hi && g < pos; g++) {
                if (!is_call[g]) continue;
                uint32_t cs = A64_USEP(g);
                if (cs >= first_pos[vi] && cs < last_use[vi]) { crosses[vi] = 1; break; }
            }
        }
        free(is_call);
    } else {
        uint64_t *live = (uint64_t *)calloc(bw, sizeof(uint64_t));
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            memcpy(live, live_out + b * bw, bw * sizeof(uint64_t));
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t ii = no; ii-- > 0; ) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, ii);
                if (ra_is_folded(T, ctx, op)) continue;
                // `live` holds the live set immediately AFTER op.
                if (name_eq(MLIR_GetOpName(op), "llvm.call")) {
                    uint32_t rvi = 0; bool has_r = false;
                    if (MLIR_GetOpNumResults(op) &&
                        a64_hh_get(&vmap, MLIR_GetOpResult(op, 0), &rvi)) has_r = true;
                    for (uint32_t vi = 0; vi < nv; vi++)
                        if (a64_bs_test(live, vi) && !(has_r && vi == rvi))
                            crosses[vi] = 1;
                }
                // Transform to live-in(op): live = (live \ defs) U uses.
                size_t nr = MLIR_GetOpNumResults(op);
                for (size_t r = 0; r < nr; r++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpResult(op, r), &vi)) a64_bs_clear(live, vi);
                }
                size_t nop = MLIR_GetOpNumOperands(op);
                for (size_t k = 0; k < nop; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) a64_bs_set(live, vi);
                }
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vi;
                        if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi))
                            a64_bs_set(live, vi);
                    }
                }
                MLIR_ValueHandle mbase, midx;
                if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, mbase, &vi)) a64_bs_set(live, vi);
                    if (a64_hh_get(&vmap, midx, &vi))  a64_bs_set(live, vi);
                }
                MLIR_ValueHandle sfx;
                if (ra_shiftfuse(T, ctx, op, &sfx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, sfx, &vi)) a64_bs_set(live, vi);
                }
            }
        }
        free(live);
    }

    // --- 3c. Phi edge-copy coalescing hints. -----------------------------------
    // For each block-arg B, find a predecessor edge whose source value V (a) is
    // used exactly once (its sole use is that edge) and (b) dies exactly where B
    // is born (last_use[V] == first_pos[B], the defining edge). If the linear
    // scan then gives B the SAME register V occupies, the edge `mov R_B, R_V`
    // becomes an identity copy that emit_edge_copies elides -- removing a link
    // from the loop-carried dependency chain (the IPC lever). The copies for
    // each cf.cond_br successor run in their own isolated landing block, so the
    // only unsoundness is passing B's own value through the same terminator to
    // another arg (then V's def would clobber it before the move): guard by
    // requiring B not appear as any successor operand of the terminator.
    int32_t *hint_src = (int32_t *)malloc((nv ? nv : 1) * sizeof(int32_t));
    for (uint32_t i = 0; i < nv; i++) hint_src[i] = -1;
    if (!getenv("TINYC_NO_PHI_COALESCE")) {
        uint32_t *nuse = (uint32_t *)calloc(nv ? nv : 1, sizeof(uint32_t));
        // Pass A: count every use of every value (regular operands, successor
        // operands, and re-injected memfuse/shiftfuse operands).
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                if (ra_is_folded(T, ctx, op)) continue;
                size_t nop = MLIR_GetOpNumOperands(op);
                for (size_t k = 0; k < nop; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) nuse[vi]++;
                }
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vi;
                        if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi))
                            nuse[vi]++;
                    }
                }
                MLIR_ValueHandle mbase, midx;
                if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, mbase, &vi)) nuse[vi]++;
                    if (a64_hh_get(&vmap, midx, &vi))  nuse[vi]++;
                }
                MLIR_ValueHandle sfx;
                if (ra_shiftfuse(T, ctx, op, &sfx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, sfx, &vi)) nuse[vi]++;
                }
            }
        }
        // Pass B: build hints from sole-use edge sources.
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                if (ns == 0) continue;
                for (size_t s = 0; s < ns; s++) {
                    MLIR_BlockHandle succ = MLIR_GetOpSuccessor(op, s);
                    size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vV, vB;
                        MLIR_ValueHandle Vh = MLIR_GetOpSuccessorOperand(op, s, k);
                        MLIR_ValueHandle Bh = MLIR_GetBlockArg(succ, k);
                        if (!a64_hh_get(&vmap, Vh, &vV)) continue;
                        if (!a64_hh_get(&vmap, Bh, &vB)) continue;
                        if (nuse[vV] != 1) continue;            // sole use = this edge
                        if (last_use[vV] != first_pos[vB]) continue; // V dies into B
                        if (hint_src[vB] >= 0) continue;        // already hinted
                        // Guard: B's own value must not be passed through this
                        // terminator to another arg (would be clobbered by V's def).
                        bool through = false;
                        for (size_t s2 = 0; s2 < ns && !through; s2++) {
                            size_t n2 = MLIR_GetOpNumSuccessorOperands(op, s2);
                            for (size_t k2 = 0; k2 < n2; k2++)
                                if (MLIR_GetOpSuccessorOperand(op, s2, k2) == Bh) {
                                    through = true; break;
                                }
                        }
                        if (through) continue;
                        hint_src[vB] = (int32_t)vV;
                    }
                }
            }
        }
        free(nuse);
    }

    // --- 4. Linear scan over the pool, loop-weighted eviction. ---
    uint32_t *order = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < nv; i++) order[i] = i;
    {   // stable bottom-up merge sort by first_pos (ties keep numbering order).
        uint32_t *tmp = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        for (size_t width = 1; width < nv; width *= 2) {
            for (size_t lo = 0; lo < nv; lo += 2 * width) {
                size_t mid = lo + width;     if (mid > nv) mid = nv;
                size_t hi  = lo + 2 * width; if (hi  > nv) hi  = nv;
                size_t a = lo, bb = mid, o = lo;
                while (a < mid && bb < hi)
                    tmp[o++] = (first_pos[order[bb]] < first_pos[order[a]]) ? order[bb++] : order[a++];
                while (a < mid) tmp[o++] = order[a++];
                while (bb < hi) tmp[o++] = order[bb++];
            }
            uint32_t *sw = order; order = tmp; tmp = sw;
        }
        free(tmp);
    }

    // Active list: (value index, pool index). reg_busy[pk].
    uint32_t *act_vi = (uint32_t *)malloc(T->npool * sizeof(uint32_t));
    uint8_t  *act_pk = (uint8_t  *)malloc(T->npool * sizeof(uint8_t));
    size_t nact = 0;
    bool reg_busy[RA_MAXPOOL]; for (int i = 0; i < T->npool; i++) reg_busy[i] = false;

    for (uint32_t k = 0; k < nv; k++) {
        uint32_t vi = order[k];
        // Retire active intervals ending before this one starts (strict, so an
        // operand whose last use == this def keeps its register).
        for (size_t a = 0; a < nact; ) {
            if (last_use[act_vi[a]] < first_pos[vi]) {
                reg_busy[act_pk[a]] = false;
                act_vi[a] = act_vi[nact - 1]; act_pk[a] = act_pk[nact - 1]; nact--;
            } else a++;
        }
        // Dead-on-def / never-used values keep their slot.
        if (last_use[vi] <= first_pos[vi]) continue;
        bool callee_only = crosses[vi];
        // Phi coalescing: if vi (a block arg) has a hint source V that is still
        // active and dies exactly at vi's birth (last_use[V] == first_pos[vi],
        // and V has no other use), retire V now and let vi inherit V's register.
        // The edge copy then sees edge_src_loc(V) == edge_dst_loc(vi) and is
        // elided. Sound: V is truly dead at this point (sole use is this edge),
        // so transferring its register to vi clobbers nothing; the register was
        // exclusive to V's interval [V.def, edge] and becomes exclusive to vi
        // from the edge onward (contiguous, no gap).
        if (hint_src[vi] >= 0) {
            int32_t vV = hint_src[vi];
            for (size_t a = 0; a < nact; a++) {
                if (act_vi[a] != (uint32_t)vV) continue;
                if (last_use[vV] != first_pos[vi]) break;   // not a clean death
                if (callee_only && act_pk[a] < T->ncaller) break; // reg class
                uint8_t pk = act_pk[a];
                home_pk[vi] = (int8_t)pk;        // inherit V's physical register
                act_vi[a] = vi;                   // ownership transfers; reg stays busy
                break;
            }
            if (home_pk[vi] >= 0) continue;
        }
        int found = -1;
        for (int pk = 0; pk < T->npool; pk++) {
            if (reg_busy[pk]) continue;
            if (callee_only && pk < T->ncaller) continue;
            found = pk; break;
        }
        if (found >= 0) {
            reg_busy[found] = true;
            act_vi[nact] = vi; act_pk[nact] = (uint8_t)found; nact++;
            home_pk[vi] = (int8_t)found;
            continue;
        }
        // No free eligible reg: evict the active interval with the lowest use
        // weight (cheapest to reload), tie-break latest end — but only if it is
        // strictly cheaper than the current value, else spill the current one.
        size_t evict_a = (size_t)-1; uint64_t best_w = 0; uint32_t best_end = 0;
        for (size_t a = 0; a < nact; a++) {
            if (callee_only && act_pk[a] < T->ncaller) continue;
            uint64_t aw = weight[act_vi[a]]; uint32_t ae = last_use[act_vi[a]];
            if (evict_a == (size_t)-1 || aw < best_w || (aw == best_w && ae > best_end)) {
                evict_a = a; best_w = aw; best_end = ae;
            }
        }
        if (evict_a != (size_t)-1 && best_w >= weight[vi]) evict_a = (size_t)-1;
        if (evict_a != (size_t)-1) {
            home_pk[act_vi[evict_a]] = -1;          // evicted -> slot
            act_vi[evict_a] = vi;                   // reuse its register
            home_pk[vi] = (int8_t)act_pk[evict_a];
        }
        // else: current value stays in its slot (home_pk already -1).
    }

    // --- 4b. Assign frame slots to spilled values with WMIR-style reuse.
    //         Walking values in first_pos order (the existing `order[]`), a slot
    //         whose occupant's interval ended strictly before the current
    //         value's first_pos is reclaimed onto a free list and handed to the
    //         current value. This is the SAME non-overlap invariant the register
    //         active-list retires on (last_use < first_pos), reusing the very
    //         intervals the allocator already trusts, so sharing is sound. Each
    //         value is processed at ITS OWN first_pos (never at an eviction
    //         point), so reused slots are always free as of that position and no
    //         WMIR-style "fresh, no-reuse" special case is needed. The frame is
    //         then sized to PEAK simultaneous spills, not TOTAL spills, keeping
    //         most offsets inside aarch64's scaled imm12 load/store range. ---
    int32_t next_slot = 0;
    {
        bool no_reuse = getenv("TINYC_NO_SLOT_REUSE") != NULL;
        uint32_t *sl_slot = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        uint32_t *sl_end  = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        uint32_t *freelist= (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        size_t n_sact = 0, n_free = 0;
        for (uint32_t k = 0; k < nv; k++) {
            uint32_t vi = order[k];
            for (size_t s = 0; s < n_sact; ) {
                if (sl_end[s] < first_pos[vi]) {
                    freelist[n_free++] = sl_slot[s];
                    sl_slot[s] = sl_slot[n_sact - 1];
                    sl_end[s]  = sl_end[n_sact - 1];
                    n_sact--;
                } else s++;
            }
            if (home_pk[vi] >= 0) continue;           // register-homed: no slot
            uint32_t end = last_use[vi] < first_pos[vi] ? first_pos[vi]
                                                        : last_use[vi];
            uint32_t slot = (!no_reuse && n_free > 0) ? freelist[--n_free]
                                                      : (uint32_t)next_slot++;
            sl_slot[n_sact] = slot; sl_end[n_sact] = end; n_sact++;
            sm_put(sm, handle[vi], (int32_t)slot);
        }
        free(sl_slot); free(sl_end); free(freelist);
    }
    if (getenv("TINYC_SPILL_STATS")) {
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (home_pk[vi] >= 0) continue;
            MLIR_OpHandle dop = MLIR_GetValueDefiningOp(handle[vi]);
            const char *nm = "(blockarg)";
            int nl = 10;
            if (dop) { string s = MLIR_GetOpName(dop); nm = s.str; nl = (int)s.size; }
            const char *dead = (last_use[vi] <= first_pos[vi]) ? "deaddef" : "live";
            char ops[128]; int op_off = 0; ops[0] = '\0';
            if (dop) {
                size_t non = MLIR_GetOpNumOperands(dop);
                for (size_t k = 0; k < non && op_off < 100; k++) {
                    MLIR_ValueHandle ov = MLIR_GetOpOperand(dop, k);
                    MLIR_OpHandle odef = MLIR_GetValueDefiningOp(ov);
                    const char *on; int onl;
                    if (!odef) { on = "barg"; onl = 4; }
                    else if (ra_is_const(T, ctx, odef)) { on = "const"; onl = 5; }
                    else { string os = MLIR_GetOpName(odef); on = os.str; onl = (int)os.size; }
                    op_off += snprintf(ops + op_off, sizeof(ops) - op_off,
                                       "%s%.*s", k ? "," : "", onl, on);
                }
            }
            fprintf(stderr, "SPILL\t%s\t%s\t%.*s\t%s\n",
                    weight[vi] > 1 ? "hot" : "cold", dead, nl, nm, ops);
        }
    }
    *out_nslots = next_slot;

    // --- 5. Emit results: write rm for register homes; report used_mask. ---
    uint32_t used_mask = 0;
    for (uint32_t vi = 0; vi < nv; vi++) {
        if (home_pk[vi] < 0) continue;
        int pk = home_pk[vi];
        sm_put(rm, handle[vi], T->pool[pk]);
        if (pk >= T->ncaller) used_mask |= (1u << (pk - T->ncaller));
    }

    #undef A64_USE_WEIGHT_AT
    #undef A64_USEP
    #undef A64_DEFP
    #undef A64_BARGP
    #undef A64_COVL
    #undef A64_COVF
    free(order); free(act_vi); free(act_pk);
    free(live_in); free(live_out); free(gen); free(kill); free(scratch);
    free(block_first); free(block_last); free(handle);
    free(def_pos); free(first_pos); free(last_use);
    free(weight); free(crosses); free(home_pk); free(loop_depth);
    free(hint_src);
    a64_hh_free(&vmap); a64_hh_free(&bmap);
    return used_mask;
}

// Linear-scan register allocator for the CFG (from-wasm) lowering path. Assigns
// block-local SSA values (def and all uses in one block, never a block arg, and
// never consumed by a terminator or an llvm.call — those paths read frame slots
// directly) to callee-saved registers x19..x28. Everything else keeps its frame
// slot. Fills `rm` (value -> home reg) and returns the set of used callee-saved
// regs as a bitmask over x19..x28 (bit k = x(19+k)).
//
// Allocation is deterministic (purely structural block/op order, ascending
// register preference), which is required for the bit-identical self-host gate.
// Callee-saved homes survive `bl`, so a value live across a call needs no
// spill-around-call handling. Functions containing ops with nested regions are
// not allocated (returns 0) — the CFG path is flat in practice.
uint32_t mlir_regalloc_cfg(MLIR_Context *ctx, MLIR_RegionHandle reg,
                               size_t n_blocks, SlotMap *rm,
                               const RegTarget *T) {
    Arena *ar = MLIR_GetArenaAllocator(ctx);
    size_t cap = 0;
    size_t maxops = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        if (no > maxops) maxops = no;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (MLIR_GetOpNumRegions(op) > 0) return 0;   // bail: nested regions
            cap += MLIR_GetOpNumResults(op);
        }
    }
    if (cap == 0) return 0;

    int32_t *def_block = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    int32_t *def_pos   = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    int32_t *last_use  = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    uint8_t *disq      = (uint8_t *)arena_alloc(ar, cap * sizeof(uint8_t));
    uint8_t *usedv     = (uint8_t *)arena_alloc(ar, cap * sizeof(uint8_t));
    uint8_t *crosses   = (uint8_t *)arena_alloc(ar, cap * sizeof(uint8_t));
    int8_t  *home_pk   = (int8_t *)arena_alloc(ar, cap * sizeof(int8_t));
    // Phi edge-copy coalescing bookkeeping: total use count of each value, and
    // (when it is used exactly once as a branch successor operand) the
    // destination block-arg value + the terminator carrying that edge.
    int32_t *use_cnt   = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    MLIR_ValueHandle *edge_arg =
        (MLIR_ValueHandle *)arena_alloc(ar, cap * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle *edge_term =
        (MLIR_OpHandle *)arena_alloc(ar, cap * sizeof(MLIR_OpHandle));
    MLIR_ValueHandle *vals =
        (MLIR_ValueHandle *)arena_alloc(ar, cap * sizeof(MLIR_ValueHandle));
    SlotMap idx = {0}; idx.arena = ar;   // value -> array index
    size_t nv = 0;

    // Pass 1: record every op-result definition with its block and position.
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (ra_is_const(T, ctx, op)) continue; // remat'd, no home
            // Folded linmem spine ops (zext/add/inttoptr) emit no code and get
            // no home register; their result is consumed by the fused load/store.
            if (ra_is_folded(T, ctx, op)) continue;
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t r = 0; r < nr; r++) {
                vals[nv] = MLIR_GetOpResult(op, r);
                def_block[nv] = (int32_t)b; def_pos[nv] = (int32_t)i;
                last_use[nv] = -1; disq[nv] = 0; usedv[nv] = 0; crosses[nv] = 0;
                home_pk[nv] = -1;
                use_cnt[nv] = 0; edge_arg[nv] = MLIR_INVALID_HANDLE;
                edge_term[nv] = MLIR_INVALID_HANDLE;
                sm_put(&idx, vals[nv], (int32_t)nv);
                nv++;
            }
        }
    }

    // Pass 2: scan operand uses. Disqualify on cross-block use, or use by a
    // terminator (incl. successor/edge operands) or an llvm.call (those paths
    // read frame slots, not home registers). Track the last in-block use.
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            bool is_term = (i + 1 == no);
            bool is_call = name_eq(MLIR_GetOpName(op), "llvm.call");
            // A folded spine op (zext/add/inttoptr) emits no code: its operand
            // reads must NOT extend liveness. The fused load/store re-registers
            // base+idx as live uses at its own position (below).
            if (ra_is_folded(T, ctx, op)) continue;
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                int32_t vi;
                if (!sm_get(&idx, MLIR_GetOpOperand(op, k), &vi)) continue;
                usedv[vi] = 1; use_cnt[vi]++;
                if ((int32_t)b != def_block[vi] || is_term || is_call)
                    disq[vi] = 1;
                else if ((int32_t)i > last_use[vi])
                    last_use[vi] = (int32_t)i;
            }
            // Branch successor (block-arg) operands are a separate operand list
            // resolved via slot-to-slot edge copies (copy_slot reads the frame
            // slot), so any value passed across an edge must stay in memory.
            size_t nsucc = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < nsucc; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    int32_t vi;
                    if (!sm_get(&idx, MLIR_GetOpSuccessorOperand(op, s, k), &vi))
                        continue;
                    usedv[vi] = 1; disq[vi] = 1; use_cnt[vi]++;
                    // Record the destination block-arg + terminator so a sole-use
                    // edge operand can later be coalesced into the block-arg's
                    // home register (the mov vanishes in emit_edge_copies).
                    edge_arg[vi]  = MLIR_GetBlockArg(MLIR_GetOpSuccessor(op, s), k);
                    edge_term[vi] = op;
                }
            }
            // Fused linmem access: the rewritten load/store reads `base` and
            // `idx` directly (not the skipped inttoptr spine). Register those as
            // live uses at this position so the allocator keeps them in regs /
            // models their lifetimes correctly (base is cross-block -> spilled,
            // idx is block-local -> stays live until here, never aliased).
            if (true) {
                MLIR_ValueHandle mb, mi;
                if (ra_memfuse(T, ctx, op, &mb, &mi)) {
                    {
                        MLIR_ValueHandle uu[2] = { mb, mi };
                        for (int u = 0; u < 2; u++) {
                            int32_t vi;
                            if (!sm_get(&idx, uu[u], &vi)) continue;
                            usedv[vi] = 1; use_cnt[vi]++;
                            if ((int32_t)b != def_block[vi] || is_call)
                                disq[vi] = 1;
                            else if ((int32_t)i > last_use[vi])
                                last_use[vi] = (int32_t)i;
                        }
                    }
                }
            }
            // Shift-fused add: the shifted-register add reads the shift source x
            // directly (the skipped shl/mul result is not a real operand). Mark
            // x live here, mirroring the memfuse re-injection above.
            if (true) {
                MLIR_ValueHandle sfx;
                if (ra_shiftfuse(T, ctx, op, &sfx)) {
                    int32_t vi;
                    if (sm_get(&idx, sfx, &vi)) {
                        usedv[vi] = 1; use_cnt[vi]++;
                        if ((int32_t)b != def_block[vi] || is_call)
                            disq[vi] = 1;
                        else if ((int32_t)i > last_use[vi])
                            last_use[vi] = (int32_t)i;
                    }
                }
            }
        }
    }

    // Pass 3: mark block-local values whose live range [def, last_use] spans an
    // llvm.call. Such values are clobbered by the `bl`, so they may only be
    // homed in callee-saved registers (never the caller-saved x12..x15 pool).
    int32_t *callpos = (int32_t *)arena_alloc(ar, (maxops ? maxops : 1) *
                                              sizeof(int32_t));
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        size_t ncall = 0;
        for (size_t i = 0; i < no; i++)
            if (name_eq(MLIR_GetOpName(MLIR_GetBlockOp(bk, i)), "llvm.call"))
                callpos[ncall++] = (int32_t)i;
        if (ncall == 0) continue;
        for (size_t vi = 0; vi < nv; vi++) {
            if (def_block[vi] != (int32_t)b) continue;
            // Strict: a value used only as a call arg (last_use == the call) is
            // not crossing -- read into the param reg before the `bl`.
            for (size_t c = 0; c < ncall; c++)
                if (def_pos[vi] < callpos[c] && callpos[c] < last_use[vi]) {
                    crosses[vi] = 1; break;
                }
        }
    }

    // Per-block linear scan over the pool (caller-saved x12..x15 then callee-
    // saved x19..x28). Block-local values never overlap across blocks, so each
    // block independently reuses the whole pool. Expiry uses a strict
    // `end < start` test so an operand whose last use coincides with a result's
    // definition keeps its register — preventing a multi-instruction lowering
    // from clobbering an operand it still reads. `used_mask` reports only the
    // callee-saved registers actually used (bit k = x(19+k)), driving the
    // prologue/epilogue save/restore; caller-saved homes need none. State is
    // kept in pool-index space (0..T->npool-1), mapped to physical regs via
    // A64_POOL; allocation is deterministic for the bit-identical self-host.
    uint32_t used_mask = 0;

    // ---- Block-argument register homing ----
    // Keep loop-carried SSA values (mem2reg promotes them to block args) in
    // registers across blocks. Collect non-entry block args, score them by use
    // frequency, and give the top T->nhome a UNIQUE reserved callee-saved
    // register (x28 downward). Because each homed arg gets a distinct register,
    // no two homed args ever share a register and correctness needs no global
    // liveness analysis (so none of the prior global-allocator O(blocks*values)
    // bitset blowup). Reserved regs are excluded from the block-local pool
    // below and saved/restored via used_mask; operand reads/writes already
    // route through `rm` (use_val/def_val/fin_val), and edges into homed args
    // are resolved by the allocation-aware parallel mover in emit_edge_copies.
    // Selection is deterministic (score desc, then block/arg order) for the
    // bit-identical self-host gate.
    uint32_t reserved_mask = 0;
    if (!getenv("TINYC_NO_HOMING")) {
        size_t nba = 0;
        for (size_t b = 1; b < n_blocks; b++)
            nba += MLIR_GetBlockNumArgs(MLIR_GetRegionBlock(reg, b));
        if (nba) {
            MLIR_ValueHandle *bv =
                (MLIR_ValueHandle *)arena_alloc(ar, nba * sizeof(MLIR_ValueHandle));
            int32_t *bscore = (int32_t *)arena_alloc(ar, nba * sizeof(int32_t));
            SlotMap bidx = {0}; bidx.arena = ar;
            size_t m = 0;
            for (size_t b = 1; b < n_blocks; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
                size_t na = MLIR_GetBlockNumArgs(bk);
                for (size_t a = 0; a < na; a++) {
                    bv[m] = MLIR_GetBlockArg(bk, a); bscore[m] = 0;
                    sm_put(&bidx, bv[m], (int32_t)m); m++;
                }
            }
            for (size_t b = 0; b < n_blocks; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
                size_t no = MLIR_GetBlockNumOps(bk);
                for (size_t i = 0; i < no; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                    size_t nop = MLIR_GetOpNumOperands(op);
                    for (size_t k = 0; k < nop; k++) {
                        int32_t bi;
                        if (sm_get(&bidx, MLIR_GetOpOperand(op, k), &bi)) bscore[bi]++;
                    }
                    size_t nsucc = MLIR_GetOpNumSuccessors(op);
                    for (size_t sx = 0; sx < nsucc; sx++) {
                        size_t nso = MLIR_GetOpNumSuccessorOperands(op, sx);
                        for (size_t k = 0; k < nso; k++) {
                            int32_t bi;
                            if (sm_get(&bidx, MLIR_GetOpSuccessorOperand(op, sx, k), &bi))
                                bscore[bi]++;
                        }
                    }
                }
            }
            for (int h = 0; h < T->nhome; h++) {
                int best = -1;
                for (size_t j = 0; j < nba; j++) {
                    if (bscore[j] <= 0) continue;
                    if (best < 0 || bscore[j] > bscore[best]) best = (int)j;
                }
                if (best < 0) break;
                int pk = T->npool - 1 - h;          // 20 -> x26, 19 -> x25, ...
                sm_put(rm, bv[best], T->pool[pk]);
                reserved_mask |= (1u << pk);
                used_mask |= (1u << (pk - T->ncaller));
                bscore[best] = -1;                   // consumed
            }
        }
    }

    // ---- Phi edge-copy coalescing (ISEL-time direct-write) ----
    // A value whose SOLE use is one branch successor-operand feeding a
    // register-homed block-arg is normally forced to a frame slot (disq above),
    // so emit_edge_copies materialises a `mov R_arg, R_val` (or a slot reload)
    // at the edge -- a link on the loop-carried dependency chain. If instead we
    // home the value DIRECTLY in the block-arg's reserved register, the edge
    // copy becomes an identity move (edge_src_loc == edge_dst_loc) and vanishes,
    // shortening the chain (the IPC lever).
    //
    // Safe only under tight conditions: the value's def is the last op before an
    // UNCONDITIONAL (single-successor) terminator, it has exactly one use (that
    // edge), and the destination register is not read by any OTHER operand of
    // the same edge (i.e. the old block-arg value is not also passed through).
    // Under these the block-arg register is dead from the def point onward
    // except for this edge, so writing the new value into it early is sound; the
    // value's own def-op may freely read the register as an input (the isel
    // already tolerates result-into-home-reg aliasing for every homed value).
    if (!getenv("TINYC_NO_PHI_COALESCE")) {
        for (size_t vi = 0; vi < nv; vi++) {
            if (!disq[vi] || use_cnt[vi] != 1) continue;
            MLIR_OpHandle term = edge_term[vi];
            MLIR_ValueHandle ba = edge_arg[vi];
            if (term == MLIR_INVALID_HANDLE || ba == MLIR_INVALID_HANDLE) continue;
            // Single-successor (unconditional) terminator only.
            if (MLIR_GetOpNumSuccessors(term) != 1) continue;
            // Value must be defined immediately before its block's terminator.
            MLIR_BlockHandle db = MLIR_GetRegionBlock(reg, def_block[vi]);
            size_t dno = MLIR_GetBlockNumOps(db);
            if (dno < 2 || def_pos[vi] != (int32_t)dno - 2) continue;
            if (MLIR_GetBlockOp(db, dno - 1) != term) continue;
            // Destination block-arg must be register-homed.
            int32_t R;
            if (!sm_get(rm, ba, &R)) continue;
            // The block-arg's register must not be read by any OTHER operand of
            // this edge (no pass-through of the old phi value into R).
            bool conflict = false;
            size_t nso = MLIR_GetOpNumSuccessorOperands(term, 0);
            for (size_t k = 0; k < nso && !conflict; k++) {
                MLIR_ValueHandle o = MLIR_GetOpSuccessorOperand(term, 0, k);
                if (o == ba) { conflict = true; break; }   // old arg passed through
                if (o == vals[vi]) continue;                // the value itself
                int32_t or_;
                if (sm_get(rm, o, &or_) && or_ == R) conflict = true;
            }
            if (conflict) continue;
            // Home the value directly in the block-arg's register.
            sm_put(rm, vals[vi], R);
        }
    }

    for (size_t b = 0; b < n_blocks; b++) {
        uint32_t freemask = ((1u << T->npool) - 1u) & ~reserved_mask;
        int32_t act_end[RA_MAXPOOL]; uint8_t act_pi[RA_MAXPOOL]; size_t nact = 0;
        for (size_t vi = 0; vi < nv; vi++) {
            if (def_block[vi] != (int32_t)b) continue;
            if (disq[vi] || !usedv[vi] || last_use[vi] < def_pos[vi]) continue;
            int32_t s = def_pos[vi];
            for (size_t a = 0; a < nact; ) {
                if (act_end[a] < s) {
                    freemask |= (1u << act_pi[a]);
                    act_pi[a]  = act_pi[nact - 1];
                    act_end[a] = act_end[nact - 1];
                    nact--;
                } else a++;
            }
            // Copy coalescing: if this value is the result of a single-source
            // cast whose source is a pool-homed, block-local value dying exactly
            // at this op, reuse the source's register. For the pure-move casts
            // this turns emit_mov_reg into a no-op (the copy disappears); for the
            // extend casts it merely lowers register pressure. Provably safe:
            // the source's live range ends here and these handlers tolerate
            // rd == r0, so no value is clobbered before its last read.
            if (!getenv("TINYC_NO_COALESCE")) {
                MLIR_ValueHandle csrc;
                MLIR_OpHandle dop = MLIR_GetValueDefiningOp(vals[vi]);
                int32_t ui;
                if (dop != MLIR_INVALID_HANDLE && cast_src(dop, &csrc) &&
                    sm_get(&idx, csrc, &ui) && def_block[ui] == (int32_t)b &&
                    home_pk[ui] >= 0 && last_use[ui] == s &&
                    !(crosses[vi] && home_pk[ui] < T->ncaller)) {
                    int pk = home_pk[ui];
                    // The source still occupies pk (strict expiry keeps an entry
                    // whose end == s active); extend it to cover this value.
                    bool ext = false;
                    for (size_t a = 0; a < nact; a++)
                        if (act_pi[a] == (uint8_t)pk) {
                            act_end[a] = last_use[vi]; ext = true; break;
                        }
                    if (ext) {
                        home_pk[vi] = (int8_t)pk;
                        sm_put(rm, vals[vi], T->pool[pk]);
                        if (pk >= T->ncaller)
                            used_mask |= (1u << (pk - T->ncaller));
                        continue;
                    }
                }
            }
            // Values that cross a call may only use callee-saved pool entries.
            uint32_t avail = crosses[vi]
                ? (freemask & ~((1u << T->ncaller) - 1u))
                : freemask;
            if (avail == 0) continue;        // no eligible reg: stay in memory
            int pk = 0; while (!(avail & (1u << pk))) pk++;
            freemask &= ~(1u << pk);
            uint8_t regn = T->pool[pk];
            sm_put(rm, vals[vi], regn);
            home_pk[vi] = (int8_t)pk;
            if (pk >= T->ncaller) used_mask |= (1u << (pk - T->ncaller));
            act_end[nact] = last_use[vi]; act_pi[nact] = (uint8_t)pk; nact++;
        }
    }
    return used_mask;
}

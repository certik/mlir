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
#include <base/buddy.h>
#include <base/string.h>
#include "mlir_api.h"

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
static void sm_reserve(SlotMap *m, size_t want) {
    size_t need = want + want / 2 + 8;
    if (m->cap >= need) return;
    size_t ncap = m->cap ? m->cap * 2 : 64;
    while (ncap < need) ncap *= 2;
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
static void sm_init_dense_buf(SlotMap *m, int32_t *buf, uintptr_t base, size_t len) {
    if (m->cap != 0 || m->n != 0 || m->dense != NULL || len == 0) return;
    memset(buf, 0xff, len * sizeof(int32_t));
    m->dense = buf;
    m->dense_base = base;
    m->dense_len = len;
}
void sm_put(SlotMap *m, MLIR_ValueHandle k, int32_t slot) {
    if (m->dense) {
        uintptr_t idx = (uintptr_t)k - m->dense_base;
        if (idx < m->dense_len && slot >= 0) {
            if (m->dense[idx] < 0) {
                m->dense[idx] = slot;
                m->n++;
            }
            return;
        }
    }
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
    if (m->dense) {
        uintptr_t idx = (uintptr_t)k - m->dense_base;
        if (idx < m->dense_len) {
            int32_t v = m->dense[idx];
            if (v >= 0) { *out = v; return true; }
        }
        if (m->cap == 0) return false;
    }
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
    return T->memspine != NULL && a64_is_spine(c, o, T->memspine);
}
static inline bool ra_memfuse(const RegTarget *T, MLIR_Context *c, MLIR_OpHandle o,
                              MLIR_ValueHandle *b, MLIR_ValueHandle *i) {
    return T->memfuse != NULL && a64_memfuse_uses(c, o, T->memfuse, b, i);
}
static inline bool ra_shiftfuse(const RegTarget *T, MLIR_Context *c, MLIR_OpHandle o,
                                MLIR_ValueHandle *x) {
    return T->shiftfuse != NULL && a64_shiftfuse_uses(c, o, T->shiftfuse, x);
}

static inline uint32_t ra_ctz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctzll(x);
#else
    uint32_t n = 0;
    if ((x & 0xffffffffull) == 0) { n += 32; x >>= 32; }
    if ((x & 0xffffull) == 0)     { n += 16; x >>= 16; }
    if ((x & 0xffull) == 0)       { n += 8;  x >>= 8;  }
    if ((x & 0xfull) == 0)        { n += 4;  x >>= 4;  }
    if ((x & 0x3ull) == 0)        { n += 2;  x >>= 2;  }
    if ((x & 0x1ull) == 0)        { n += 1; }
    return n;
#endif
}

typedef struct {
    bool inited;
    bool home_params;
    bool halfopen;
    bool linear_crosses;
    bool phi_coalesce;
    bool no_slot_reuse;
    bool spill_stats;
} RaEnvFlags;

static RaEnvFlags g_ra_env;

static const RaEnvFlags *ra_get_env(void) {
    if (!g_ra_env.inited) {
        g_ra_env.home_params    = (getenv("TINYC_NO_PARAM_HOME") == NULL);
        g_ra_env.halfopen       = (getenv("TINYC_NO_HALFOPEN") == NULL);
        g_ra_env.linear_crosses = (getenv("TINYC_LINEAR_CROSSES") != NULL);
        g_ra_env.phi_coalesce   = (getenv("TINYC_NO_PHI_COALESCE") == NULL);
        g_ra_env.no_slot_reuse  = (getenv("TINYC_NO_SLOT_REUSE") != NULL);
        g_ra_env.spill_stats    = (getenv("TINYC_SPILL_STATS") != NULL);
        g_ra_env.inited = true;
    }
    return &g_ra_env;
}

typedef struct { uint32_t b; uint32_t gid; } RaGenEdge;

// Min-heap on sl_end for O(log N) spill-slot retirement.
static void sl_heap_push(uint32_t *h_end, uint32_t *h_slot, size_t *hn,
                         uint32_t end, uint32_t slot) {
    size_t i = (*hn)++;
    while (i > 0) {
        size_t p = (i - 1) >> 1;
        if (h_end[p] <= end) break;
        h_end[i] = h_end[p];
        h_slot[i] = h_slot[p];
        i = p;
    }
    h_end[i] = end;
    h_slot[i] = slot;
}
static void sl_heap_pop(uint32_t *h_end, uint32_t *h_slot, size_t *hn) {
    size_t n = --(*hn);
    if (n == 0) return;
    uint32_t e = h_end[n], s = h_slot[n];
    size_t i = 0;
    for (;;) {
        size_t l = (i << 1) + 1;
        if (l >= n) break;
        size_t r = l + 1;
        size_t best = (r < n && h_end[r] < h_end[l]) ? r : l;
        if (e <= h_end[best]) break;
        h_end[i] = h_end[best];
        h_slot[i] = h_slot[best];
        i = best;
    }
    h_end[i] = e;
    h_slot[i] = s;
}

// Reusable scratch buffers across mlir_regalloc_global calls (avoids 35+
// malloc/calloc/free calls per function).
typedef struct {
    uint8_t *buf;
    size_t   cap;
} RaPool;

static void *ra_pool_ensure(RaPool *p, size_t need) {
    if (need > p->cap) {
        size_t ncap = p->cap ? p->cap * 2 : 4096u;
        while (ncap < need) ncap *= 2;
        if (p->buf) buddy_free(p->buf);
        size_t actual = 0;
        p->buf = (uint8_t *)buddy_alloc(ncap, &actual);
        p->cap = actual;
    }
    return p->buf;
}

static inline void *ra_bump(uint8_t *base, size_t *off, size_t bytes) {
    size_t a = (*off + 7u) & ~(size_t)7u;
    *off = a + bytes;
    return base + a;
}

static RaPool g_ra_blk_pool  = { NULL, 0 };
static RaPool g_ra_main_pool = { NULL, 0 };
static RaPool g_ra_live_pool = { NULL, 0 };
static RaPool g_ra_vmap_pool = { NULL, 0 };
static RaPool g_ra_gen_pool  = { NULL, 0 };
static RaPool g_ra_succ_pool = { NULL, 0 };
static RaPool g_ra_rm_pool   = { NULL, 0 };
static RaPool g_ra_sm_pool   = { NULL, 0 };
static RaPool g_ra_call_pool = { NULL, 0 };

// ===========================================================================
// Fast block-local linear-scan allocator (-O0 / default mode).
// ===========================================================================
static MLIR_OpHandle *g_ra_fast_allocas = NULL;
static size_t         g_ra_fast_allocas_n = 0;
static size_t         g_ra_fast_allocas_cap = 0;

const MLIR_OpHandle *mlir_regalloc_fast_allocas(size_t *out_n) {
    if (out_n) *out_n = g_ra_fast_allocas_n;
    return g_ra_fast_allocas;
}

uint32_t mlir_regalloc_fast(MLIR_Context *ctx, MLIR_RegionHandle reg,
                            size_t n_blocks, SlotMap *rm,
                            SlotMap *sm, int32_t *out_nslots,
                            const RegTarget *T) {
    *out_nslots = 0;
    g_ra_fast_allocas_n = 0;
    if (n_blocks == 0) return RA_BAIL;

    MLIR_BlockHandle *blk_h = (MLIR_BlockHandle *)ra_pool_ensure(
        &g_ra_blk_pool, n_blocks * sizeof(MLIR_BlockHandle));
    size_t nv_max = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        blk_h[b] = bk;
        nv_max += MLIR_GetBlockNumArgs(ctx, bk) + MLIR_GetBlockNumOps(ctx, bk);
    }
    if (nv_max == 0) return 0;

    size_t nv_alloc = nv_max;
    size_t main_need =
        n_blocks * (sizeof(uint32_t) * 3 + sizeof(uint8_t) + 8) +
        (n_blocks + 1) * (sizeof(uint32_t) + 8) +
        nv_alloc * (sizeof(MLIR_ValueHandle) + sizeof(uint32_t) * 3 + sizeof(uint8_t) + 8) +
        256;
    uint8_t *mbuf = (uint8_t *)ra_pool_ensure(&g_ra_main_pool, main_need);
    size_t moff = 0;

    uint32_t *block_first  = (uint32_t *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint32_t));
    uint32_t *blk_nops     = (uint32_t *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint32_t));
    uint32_t *blk_vstart   = (uint32_t *)ra_bump(mbuf, &moff, (n_blocks + 1) * sizeof(uint32_t));
    uint8_t  *blk_has_call = (uint8_t  *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint8_t));
    MLIR_ValueHandle *handle = (MLIR_ValueHandle *)ra_bump(mbuf, &moff, nv_alloc * sizeof(MLIR_ValueHandle));
    uint32_t *def_pos   = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint32_t *def_blk   = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint32_t *last_use  = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint8_t  *is_global = (uint8_t  *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint8_t));

    // Pass 1: number values and detect calls per block.
    uint32_t pos = 0, nv = 0;
    uintptr_t min_vh = ~(uintptr_t)0, max_vh = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = blk_h[b];
        block_first[b] = pos;
        blk_vstart[b] = nv;
        uint8_t has_call = 0;
        size_t na = MLIR_GetBlockNumArgs(ctx, bk);
        for (size_t a = 0; a < na; a++) {
            MLIR_ValueHandle arg = MLIR_GetBlockArg(ctx, bk, a);
            uintptr_t vh = (uintptr_t)arg;
            if (vh < min_vh) min_vh = vh;
            if (vh > max_vh) max_vh = vh;
            handle[nv] = arg; def_pos[nv] = pos; def_blk[nv] = (uint32_t)b;
            last_use[nv] = 0; is_global[nv] = 1;
            nv++;
        }
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        blk_nops[b] = (uint32_t)no;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpNumRegions(ctx, op) > 0) return RA_BAIL;
            MLIR_OpType oty = MLIR_GetOpType(ctx, op);
            if (oty == OP_TYPE_LLVM_CALL) has_call = 1;
            else if (oty == OP_TYPE_LLVM_ALLOCA) {
                if (g_ra_fast_allocas_n == g_ra_fast_allocas_cap) {
                    size_t nc = g_ra_fast_allocas_cap ? g_ra_fast_allocas_cap * 2 : 32;
                    g_ra_fast_allocas = (MLIR_OpHandle *)realloc(
                        g_ra_fast_allocas, nc * sizeof(MLIR_OpHandle));
                    g_ra_fast_allocas_cap = nc;
                }
                g_ra_fast_allocas[g_ra_fast_allocas_n++] = op;
            }
            if (oty == OP_TYPE_LLVM_MLIR_CONSTANT ||
                oty == OP_TYPE_LLVM_MLIR_ADDRESSOF ||
                oty == OP_TYPE_LLVM_MLIR_UNDEF ||
                oty == OP_TYPE_LLVM_MLIR_ZERO ||
                oty == OP_TYPE_LLVM_MLIR_NULL) {
                pos++;
                continue;
            }
            size_t nr = MLIR_GetOpNumResults(ctx, op);
            if (nv + nr > nv_max) return RA_BAIL;
            for (size_t r = 0; r < nr; r++) {
                MLIR_ValueHandle rv = MLIR_GetOpResult(ctx, op, r);
                uintptr_t vh = (uintptr_t)rv;
                if (vh < min_vh) min_vh = vh;
                if (vh > max_vh) max_vh = vh;
                handle[nv] = rv; def_pos[nv] = pos; def_blk[nv] = (uint32_t)b;
                last_use[nv] = 0; is_global[nv] = 0;
                nv++;
            }
            pos++;
        }
        blk_has_call[b] = has_call;
    }
    blk_vstart[n_blocks] = nv;
    if (nv == 0) return 0;

    // Value handle -> vi dense table (or hash fallback).
    bool use_dense_vmap = (max_vh >= min_vh &&
                           (max_vh - min_vh) < (uintptr_t)nv * 4u + 256u);
    size_t v_span = use_dense_vmap ? (size_t)(max_vh - min_vh) + 1u : 0;
    uint32_t *v_dense = NULL;
    A64HHash vmap = { NULL, NULL, 0 };
    if (use_dense_vmap) {
        v_dense = (uint32_t *)ra_pool_ensure(&g_ra_vmap_pool, v_span * sizeof(uint32_t));
        memset(v_dense, 0xff, v_span * sizeof(uint32_t));
        for (uint32_t i = 0; i < nv; i++)
            v_dense[(uintptr_t)handle[i] - min_vh] = i;
    } else {
        a64_hh_init(&vmap, nv);
        for (uint32_t i = 0; i < nv; i++) a64_hh_put(&vmap, handle[i], i);
    }

    // Pass 2: record last_use and mark cross-block uses as global.
    if (use_dense_vmap) {
        if (n_blocks == 1) {
            MLIR_BlockHandle bk = blk_h[0];
            size_t no = blk_nops[0];
            uint32_t op_pos = 0;
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                size_t nop = MLIR_GetOpNumOperands(ctx, op);
                for (size_t k = 0; k < nop; k++) {
                    uintptr_t d = (uintptr_t)MLIR_GetOpOperand(ctx, op, k) - min_vh;
                    if (d < v_span) {
                        uint32_t vi = v_dense[d];
                        if (vi != UINT32_MAX && op_pos > last_use[vi]) last_use[vi] = op_pos;
                    }
                }
                op_pos++;
            }
        } else {
            for (size_t b = 0; b < n_blocks; b++) {
                MLIR_BlockHandle bk = blk_h[b];
                uint32_t op_pos = block_first[b];
                size_t no = blk_nops[b];
                for (size_t i = 0; i < no; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                    size_t nop = MLIR_GetOpNumOperands(ctx, op);
                    for (size_t k = 0; k < nop; k++) {
                        uintptr_t d = (uintptr_t)MLIR_GetOpOperand(ctx, op, k) - min_vh;
                        if (d < v_span) {
                            uint32_t vi = v_dense[d];
                            if (vi != UINT32_MAX) {
                                if (def_blk[vi] != (uint32_t)b) is_global[vi] = 1;
                                else if (op_pos > last_use[vi]) last_use[vi] = op_pos;
                            }
                        }
                    }
                    op_pos++;
                }
                if (no > 0) {
                    MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
                    uint32_t tpos = op_pos - 1;
                    size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
                    for (size_t s = 0; s < ns; s++) {
                        size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, s);
                        for (size_t k = 0; k < nso; k++) {
                            uintptr_t d = (uintptr_t)MLIR_GetOpSuccessorOperand(ctx, term, s, k) - min_vh;
                            if (d < v_span) {
                                uint32_t vi = v_dense[d];
                                if (vi != UINT32_MAX) {
                                    if (def_blk[vi] != (uint32_t)b) is_global[vi] = 1;
                                    else if (tpos > last_use[vi]) last_use[vi] = tpos;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = blk_h[b];
            uint32_t op_pos = block_first[b];
            size_t no = blk_nops[b];
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                size_t nop = MLIR_GetOpNumOperands(ctx, op);
                for (size_t k = 0; k < nop; k++) {
                    uint32_t vi;
                    if (vmap.cap != 0 && a64_hh_get(&vmap, MLIR_GetOpOperand(ctx, op, k), &vi)) {
                        if (def_blk[vi] != (uint32_t)b) is_global[vi] = 1;
                        else if (op_pos > last_use[vi]) last_use[vi] = op_pos;
                    }
                }
                op_pos++;
            }
            if (n_blocks > 1 && no > 0) {
                MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
                uint32_t tpos = op_pos - 1;
                size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vi;
                        if (vmap.cap != 0 && a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(ctx, term, s, k), &vi)) {
                            if (def_blk[vi] != (uint32_t)b) is_global[vi] = 1;
                            else if (tpos > last_use[vi]) last_use[vi] = tpos;
                        }
                    }
                }
            }
        }
    }
    if (vmap.cap) a64_hh_free(&vmap);

    // Prepare dense output table for rm; initialize sm lazily only on first spill.
    int32_t *rm_d = NULL, *sm_d = NULL;
    bool sm_inited = false;
    if (use_dense_vmap) {
        rm_d = (int32_t *)ra_pool_ensure(&g_ra_rm_pool, v_span * sizeof(int32_t));
        sm_init_dense_buf(rm, rm_d, min_vh, v_span);
    } else {
        sm_reserve(rm, nv);
    }

    #define RAF_SM_PUT(VH, SLOT) do { \
        if (!sm_inited) { \
            sm_inited = true; \
            if (use_dense_vmap) { \
                sm_d = (int32_t *)ra_pool_ensure(&g_ra_sm_pool, v_span * sizeof(int32_t)); \
                sm_init_dense_buf(sm, sm_d, min_vh, v_span); \
            } else { \
                sm_reserve(sm, nv); \
            } \
        } \
        if (sm_d) { \
            uintptr_t _idx = (uintptr_t)(VH) - min_vh; \
            if (sm_d[_idx] < 0) { sm_d[_idx] = (SLOT); sm_n++; } \
        } else { \
            sm_put(sm, (VH), (SLOT)); \
        } \
    } while (0)

    // Pass 3: assign dedicated slots to cross-block values, then per-block linear scan.
    int32_t glob_slots = 0;
    size_t sm_n = sm->n;
    for (uint32_t vi = 0; vi < nv; vi++) {
        if (!is_global[vi]) continue;
        int32_t slot = glob_slots++;
        RAF_SM_PUT(handle[vi], slot);
    }

    uint32_t pool_mask = (T->npool >= 32) ? 0xFFFFFFFFu : ((1u << T->npool) - 1u);
    uint32_t caller_mask = (T->ncaller >= 32) ? 0xFFFFFFFFu : ((1u << T->ncaller) - 1u);
    uint32_t callee_mask = pool_mask & ~caller_mask;
    uint32_t used_mask = 0;
    int32_t dead_slot = glob_slots;
    int32_t local_slot_base = glob_slots + 1;
    int32_t max_slot = glob_slots;
    size_t rm_n = rm->n;

    uint32_t act_end[RA_MAXPOOL];
    uint8_t  act_pk[RA_MAXPOOL];

    for (size_t b = 0; b < n_blocks; b++) {
        uint32_t v0 = blk_vstart[b], v1 = blk_vstart[b + 1];
        if (v0 == v1) continue;
        uint32_t avail_pool = blk_has_call[b] ? callee_mask : pool_mask;
        uint32_t busy_mask = 0;
        size_t nact = 0;
        int32_t local_slot = local_slot_base;
        for (uint32_t vi = v0; vi < v1; vi++) {
            if (is_global[vi]) continue;
            uint32_t fp = def_pos[vi];
            uint32_t lu = last_use[vi];
            if (lu <= fp) {
                // Dead-on-def: share a dedicated scratch slot at dead_slot (never
                // aliases any live global [0..glob_slots) or live local [local_slot_base..)).
                if (dead_slot + 1 > max_slot) max_slot = dead_slot + 1;
                RAF_SM_PUT(handle[vi], dead_slot);
                continue;
            }
            for (size_t a = 0; a < nact; ) {
                if (act_end[a] < fp) {
                    busy_mask &= ~(1u << act_pk[a]);
                    nact--;
                    act_end[a] = act_end[nact];
                    act_pk[a]  = act_pk[nact];
                } else a++;
            }
            uint32_t avail = (~busy_mask) & avail_pool;
            if (avail) {
                uint32_t pk = ra_ctz64((uint64_t)avail);
                busy_mask |= (1u << pk);
                act_end[nact] = lu;
                act_pk[nact]  = (uint8_t)pk;
                nact++;
                if (pk >= (uint32_t)T->ncaller) used_mask |= (1u << (pk - (uint32_t)T->ncaller));
                if (rm_d) {
                    uintptr_t idx = (uintptr_t)handle[vi] - min_vh;
                    if (rm_d[idx] < 0) { rm_d[idx] = T->pool[pk]; rm_n++; }
                } else {
                    sm_put(rm, handle[vi], T->pool[pk]);
                }
            } else {
                int32_t slot = local_slot++;
                if (local_slot > max_slot) max_slot = local_slot;
                RAF_SM_PUT(handle[vi], slot);
            }
        }
    }
    #undef RAF_SM_PUT
    if (rm_d) rm->n = rm_n;
    if (sm_d) sm->n = sm_n;
    *out_nslots = max_slot;
    return used_mask;
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

    const RaEnvFlags *env = ra_get_env();
    bool home_params = env->home_params;
    bool halfopen = env->halfopen;

    // --- 0. Cache block handles and upper-bound nv_max in O(n_blocks) (no op walk). ---
    MLIR_BlockHandle *blk_h = (MLIR_BlockHandle *)ra_pool_ensure(
        &g_ra_blk_pool, n_blocks * sizeof(MLIR_BlockHandle));
    size_t nv_max = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        blk_h[b] = bk;
        if (b > 0 || home_params) nv_max += MLIR_GetBlockNumArgs(ctx, bk);
        nv_max += MLIR_GetBlockNumOps(ctx, bk);
    }
    if (nv_max == 0) return 0;
    {
        size_t bwg = a64_bs_words(nv_max);
        if (nv_max > 100000u || (uint64_t)n_blocks * (uint64_t)bwg > 4000000ull) {
            return RA_BAIL;
        }
    }

    #define A64_USEP(g)  (halfopen ? (2u * (uint32_t)(g) + 1u) : (uint32_t)(g))
    #define A64_DEFP(g)  (halfopen ? (2u * (uint32_t)(g) + 2u) : (uint32_t)(g))
    #define A64_BARGP(g) (halfopen ? (2u * (uint32_t)(g))      : (uint32_t)(g))
    #define A64_COVL(g)  (halfopen ? (2u * (uint32_t)(g) + 1u) : (uint32_t)(g))
    #define A64_COVF(g)  (halfopen ? (2u * (uint32_t)(g))      : (uint32_t)(g))

    // --- 1. Linearise + number values + record def_blk / block_vrange / calls. ---
    size_t nv_alloc = nv_max ? nv_max : 1;
    size_t main_need =
        n_blocks * (sizeof(uint32_t) * 4 + 8) +
        (n_blocks + 1) * (sizeof(uint32_t) * 3 + sizeof(int64_t) + 8) +
        nv_alloc * (sizeof(MLIR_ValueHandle) + sizeof(uint32_t) * 12 +
                    sizeof(uint64_t) + sizeof(uint8_t) * 2 + sizeof(int32_t) + 8) +
        256;
    uint8_t *mbuf = (uint8_t *)ra_pool_ensure(&g_ra_main_pool, main_need);
    size_t moff = 0;

    uint32_t *block_first = (uint32_t *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint32_t));
    uint32_t *block_last  = (uint32_t *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint32_t));
    uint32_t *blk_vstart  = (uint32_t *)ra_bump(mbuf, &moff, (n_blocks + 1) * sizeof(uint32_t));
    uint32_t *call_off    = (uint32_t *)ra_bump(mbuf, &moff, (n_blocks + 1) * sizeof(uint32_t));
    uint32_t *first_call  = (uint32_t *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint32_t));
    memset(first_call, 0xff, n_blocks * sizeof(uint32_t));
    MLIR_ValueHandle *handle = (MLIR_ValueHandle *)ra_bump(mbuf, &moff, nv_alloc * sizeof(MLIR_ValueHandle));
    uint32_t *def_pos   = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint32_t *def_blk   = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint32_t *first_pos = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint32_t *last_use  = (uint32_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint32_t));
    uint64_t *weight    = (uint64_t *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint64_t));
    uint8_t  *crosses   = (uint8_t  *)ra_bump(mbuf, &moff, nv_alloc * sizeof(uint8_t));
    memset(crosses, 0, nv_alloc * sizeof(uint8_t));
    int8_t   *home_pk   = (int8_t   *)ra_bump(mbuf, &moff, nv_alloc * sizeof(int8_t));

    uint32_t *call_pos_arr = (uint32_t *)ra_pool_ensure(&g_ra_call_pool, 64 * sizeof(uint32_t));
    size_t call_pos_cap = g_ra_call_pool.cap / sizeof(uint32_t);
    size_t n_calls_tot = 0;

    uint32_t pos = 0, nv = 0, n_entry_args = 0;
    uintptr_t min_vh = ~(uintptr_t)0, max_vh = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = blk_h[b];
        block_first[b] = pos;
        blk_vstart[b] = nv;
        call_off[b] = (uint32_t)n_calls_tot;
        if (b > 0 || home_params) {
            size_t na = MLIR_GetBlockNumArgs(ctx, bk);
            for (size_t a = 0; a < na; a++) {
                MLIR_ValueHandle arg = MLIR_GetBlockArg(ctx, bk, a);
                uintptr_t vh = (uintptr_t)arg;
                if (vh < min_vh) min_vh = vh;
                if (vh > max_vh) max_vh = vh;
                handle[nv] = arg; def_pos[nv] = A64_BARGP(pos); def_blk[nv] = (uint32_t)b;
                first_pos[nv] = A64_BARGP(pos);
                last_use[nv] = 0; weight[nv] = 0; home_pk[nv] = -1;
                nv++;
            }
            if (b == 0) n_entry_args = nv;
        }
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpNumRegions(ctx, op) > 0) return RA_BAIL;
            MLIR_OpType oty = MLIR_GetOpType(ctx, op);
            if (oty == OP_TYPE_LLVM_CALL) {
                if (first_call[b] == UINT32_MAX) first_call[b] = (uint32_t)i;
                if (n_calls_tot == call_pos_cap) {
                    size_t ncap = call_pos_cap * 2;
                    size_t actual = 0;
                    uint32_t *nc = (uint32_t *)buddy_alloc(ncap * sizeof(uint32_t), &actual);
                    memcpy(nc, call_pos_arr, n_calls_tot * sizeof(uint32_t));
                    buddy_free(g_ra_call_pool.buf);
                    g_ra_call_pool.buf = (uint8_t *)nc;
                    g_ra_call_pool.cap = actual;
                    call_pos_arr = nc; call_pos_cap = actual / sizeof(uint32_t);
                }
                call_pos_arr[n_calls_tot++] = pos;
            }
            bool is_c = (oty == OP_TYPE_LLVM_MLIR_CONSTANT ||
                         oty == OP_TYPE_LLVM_MLIR_ZERO ||
                         oty == OP_TYPE_LLVM_MLIR_NULL) &&
                        ra_is_const(T, ctx, op);
            bool skip = is_c || ra_is_folded(T, ctx, op);
            if (!skip) {
                size_t nr = MLIR_GetOpNumResults(ctx, op);
                if (nv + nr > nv_max) return RA_BAIL;
                for (size_t r = 0; r < nr; r++) {
                    MLIR_ValueHandle rv = MLIR_GetOpResult(ctx, op, r);
                    uintptr_t vh = (uintptr_t)rv;
                    if (vh < min_vh) min_vh = vh;
                    if (vh > max_vh) max_vh = vh;
                    handle[nv] = rv; def_pos[nv] = A64_DEFP(pos); def_blk[nv] = (uint32_t)b;
                    first_pos[nv] = A64_DEFP(pos);
                    last_use[nv] = 0; weight[nv] = 0; home_pk[nv] = -1;
                    nv++;
                }
            }
            pos++;
        }
        block_last[b] = pos;
    }
    blk_vstart[n_blocks] = nv;
    call_off[n_blocks] = (uint32_t)n_calls_tot;

    // Build value handle -> vi lookup: O(1) direct array when handles are dense
    // (always true on native backend), else open-addressing hash fallback.
    bool use_dense_vmap = (nv > 0 && max_vh >= min_vh &&
                           (max_vh - min_vh) < (uintptr_t)nv * 4u + 256u);
    size_t v_span = use_dense_vmap ? (size_t)(max_vh - min_vh) + 1u : 0;
    uint32_t *v_dense = NULL;
    A64HHash vmap = { NULL, NULL, 0 };
    if (use_dense_vmap) {
        v_dense = (uint32_t *)ra_pool_ensure(&g_ra_vmap_pool, v_span * sizeof(uint32_t));
        memset(v_dense, 0xff, v_span * sizeof(uint32_t));
        for (uint32_t i = 0; i < nv; i++)
            v_dense[(uintptr_t)handle[i] - min_vh] = i;
    } else if (nv > 0) {
        a64_hh_init(&vmap, nv);
        for (uint32_t i = 0; i < nv; i++) a64_hh_put(&vmap, handle[i], i);
    }

    #define RA_VMAP_GET(VH, OUT) ( \
        use_dense_vmap \
            ? (((uintptr_t)(VH) - min_vh) < v_span && \
               ((*(OUT) = v_dense[(uintptr_t)(VH) - min_vh]) != UINT32_MAX)) \
            : (vmap.cap != 0 && a64_hh_get(&vmap, (VH), (OUT))) \
    )

    // --- 1b. Precompute CSR successors + per-block loop depth via terminator only. ---
    uint32_t *succ_off = NULL;
    uint32_t *succ_idx = NULL;
    uint32_t *loop_depth = NULL;
    bool has_backedge = false;
    if (n_blocks > 1) {
        succ_off = (uint32_t *)ra_bump(mbuf, &moff, (n_blocks + 1) * sizeof(uint32_t));
        size_t succ_cap = n_blocks * 2u + 8u;
        succ_idx = (uint32_t *)ra_pool_ensure(&g_ra_succ_pool, succ_cap * sizeof(uint32_t));
        succ_cap = g_ra_succ_pool.cap / sizeof(uint32_t);
        loop_depth = (uint32_t *)ra_bump(mbuf, &moff, n_blocks * sizeof(uint32_t));
        memset(loop_depth, 0, n_blocks * sizeof(uint32_t));
        int64_t *delta = (int64_t *)ra_bump(mbuf, &moff, (n_blocks + 1) * sizeof(int64_t));
        memset(delta, 0, (n_blocks + 1) * sizeof(int64_t));
        size_t sc = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            succ_off[b] = (uint32_t)sc;
            MLIR_OpHandle term = MLIR_GetBlockTerminator(ctx, blk_h[b]);
            if (term == MLIR_INVALID_HANDLE) continue;
            size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
            if (sc + ns > succ_cap) {
                size_t ncap = succ_cap * 2;
                while (ncap < sc + ns) ncap *= 2;
                size_t actual = 0;
                uint32_t *ns_buf = (uint32_t *)buddy_alloc(ncap * sizeof(uint32_t), &actual);
                memcpy(ns_buf, succ_idx, sc * sizeof(uint32_t));
                buddy_free(g_ra_succ_pool.buf);
                g_ra_succ_pool.buf = (uint8_t *)ns_buf;
                g_ra_succ_pool.cap = actual;
                succ_idx = ns_buf; succ_cap = actual / sizeof(uint32_t);
            }
            for (size_t s = 0; s < ns; s++) {
                size_t ti_s = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, term, s));
                if (ti_s < n_blocks) {
                    uint32_t ti = (uint32_t)ti_s;
                    succ_idx[sc++] = ti;
                    if (ti <= (uint32_t)b) {
                        has_backedge = true;
                        delta[ti] += 1; delta[b + 1] -= 1;
                    }
                }
            }
        }
        succ_off[n_blocks] = (uint32_t)sc;
        if (has_backedge) {
            int64_t acc = 0;
            for (size_t b = 0; b < n_blocks; b++) {
                acc += delta[b]; loop_depth[b] = acc < 0 ? 0 : (uint32_t)acc;
            }
        }
    }
    #define A64_USE_WEIGHT_AT(d) (1ull << (((d) * 3u) < 60u ? ((d) * 3u) : 60u))

    // --- 2. Single combined operand pass: first_pos / last_use / weight / nuse
    //        AND identify cross-block ("global") values for sparse liveness. ---
    size_t nv_sz = nv ? nv : 1;
    uint32_t *glob_id = NULL;
    uint32_t *glob_vi = NULL;
    uint32_t n_glob = 0;
    size_t gen_cap = 0, gen_n = 0;
    RaGenEdge *gen_edges = NULL;
    uint32_t *nuse = NULL;
    if (n_blocks == 1) {
        MLIR_BlockHandle bk = blk_h[0];
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        uint32_t op_pos = 0;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (ra_is_folded(T, ctx, op)) { op_pos++; continue; }
            uint32_t up = A64_USEP(op_pos);
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            for (size_t k = 0; k < nop; k++) {
                uint32_t vi;
                if (RA_VMAP_GET(MLIR_GetOpOperand(ctx, op, k), &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += 1ull;
                }
            }
            MLIR_ValueHandle mbase, midx;
            if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                uint32_t vi;
                if (RA_VMAP_GET(mbase, &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += 1ull;
                }
                if (RA_VMAP_GET(midx, &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += 1ull;
                }
            }
            MLIR_ValueHandle sfx;
            if (ra_shiftfuse(T, ctx, op, &sfx)) {
                uint32_t vi;
                if (RA_VMAP_GET(sfx, &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += 1ull;
                }
            }
            op_pos++;
        }
    } else {
        glob_id = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
        memset(glob_id, 0xff, nv_sz * sizeof(uint32_t));
        glob_vi = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
        gen_cap = 256;
        gen_edges = (RaGenEdge *)ra_pool_ensure(&g_ra_gen_pool, gen_cap * sizeof(RaGenEdge));
        gen_cap = g_ra_gen_pool.cap / sizeof(RaGenEdge);
        if (env->phi_coalesce) {
            nuse = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
            memset(nuse, 0, nv_sz * sizeof(uint32_t));
        }

    #define RA_RECORD_USE(VI, BLK_B, UP, W) do { \
        uint32_t _vi = (VI); \
        if ((UP) > last_use[_vi]) last_use[_vi] = (UP); \
        if ((UP) < first_pos[_vi]) first_pos[_vi] = (UP); \
        weight[_vi] += (W); \
        if (nuse) nuse[_vi]++; \
        if (def_blk[_vi] != (uint32_t)(BLK_B) || ((BLK_B) == 0 && _vi < n_entry_args)) { \
            uint32_t _gid = glob_id[_vi]; \
            if (_gid == UINT32_MAX) { \
                _gid = n_glob++; \
                glob_id[_vi] = _gid; \
                glob_vi[_gid] = _vi; \
            } \
            if (gen_n == gen_cap) { \
                size_t new_cap = gen_cap * 2; \
                size_t actual = 0; \
                RaGenEdge *ng = (RaGenEdge *)buddy_alloc(new_cap * sizeof(RaGenEdge), &actual); \
                memcpy(ng, gen_edges, gen_n * sizeof(RaGenEdge)); \
                buddy_free(g_ra_gen_pool.buf); \
                g_ra_gen_pool.buf = (uint8_t *)ng; \
                g_ra_gen_pool.cap = actual; \
                gen_edges = ng; gen_cap = actual / sizeof(RaGenEdge); \
            } \
            gen_edges[gen_n].b = (uint32_t)(BLK_B); \
            gen_edges[gen_n].gid = _gid; \
            gen_n++; \
        } \
    } while (0)

        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = blk_h[b];
            uint32_t op_pos = block_first[b];
            uint64_t w = loop_depth ? A64_USE_WEIGHT_AT(loop_depth[b]) : 1ull;
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                if (ra_is_folded(T, ctx, op)) { op_pos++; continue; }
                uint32_t up = A64_USEP(op_pos);
                size_t nop = MLIR_GetOpNumOperands(ctx, op);
                for (size_t k = 0; k < nop; k++) {
                    uint32_t vi;
                    if (RA_VMAP_GET(MLIR_GetOpOperand(ctx, op, k), &vi))
                        RA_RECORD_USE(vi, b, up, w);
                }
                MLIR_ValueHandle mbase, midx;
                if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                    uint32_t vi;
                    if (RA_VMAP_GET(mbase, &vi)) RA_RECORD_USE(vi, b, up, w);
                    if (RA_VMAP_GET(midx, &vi))  RA_RECORD_USE(vi, b, up, w);
                }
                MLIR_ValueHandle sfx;
                if (ra_shiftfuse(T, ctx, op, &sfx)) {
                    uint32_t vi;
                    if (RA_VMAP_GET(sfx, &vi)) RA_RECORD_USE(vi, b, up, w);
                }
                op_pos++;
            }
            if (no > 0) {
                MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
                uint32_t up = A64_USEP(op_pos - 1);
                size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
                for (size_t s = 0; s < ns; s++) {
                    MLIR_BlockHandle succ = MLIR_GetOpSuccessor(ctx, term, s);
                    size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vi;
                        if (RA_VMAP_GET(MLIR_GetOpSuccessorOperand(ctx, term, s, k), &vi))
                            RA_RECORD_USE(vi, b, up, w);
                        uint32_t avi;
                        if (RA_VMAP_GET(MLIR_GetBlockArg(ctx, succ, k), &avi) &&
                            up < first_pos[avi]) first_pos[avi] = up;
                    }
                }
            }
        }
    #undef RA_RECORD_USE
    }

    // --- 2b. Compact liveness dataflow over ONLY the `n_glob` cross-block values. ---
    size_t bw_g = n_glob ? a64_bs_words(n_glob) : 0;
    uint64_t *live_in = NULL, *live_out = NULL, *gen_g = NULL, *kill_g = NULL;
    if (bw_g > 0) {
        size_t live_words = n_blocks * bw_g;
        size_t live_bytes = (live_words * 4 + bw_g * 3) * sizeof(uint64_t);
        uint64_t *lbuf = (uint64_t *)ra_pool_ensure(&g_ra_live_pool, live_bytes);
        live_in  = lbuf;
        live_out = lbuf + live_words;
        gen_g    = lbuf + live_words * 2;
        kill_g   = lbuf + live_words * 3;
        uint64_t *scr_g  = lbuf + live_words * 4;
        memset(lbuf, 0, live_words * 4 * sizeof(uint64_t));
        for (uint32_t g = 0; g < n_glob; g++) {
            uint32_t vi = glob_vi[g];
            if (vi >= n_entry_args)
                a64_bs_set(kill_g + (size_t)def_blk[vi] * bw_g, g);
        }
        for (size_t e = 0; e < gen_n; e++) {
            uint32_t b = gen_edges[e].b, g = gen_edges[e].gid;
            if (!a64_bs_test(kill_g + (size_t)b * bw_g, g))
                a64_bs_set(gen_g + (size_t)b * bw_g, g);
        }
        if (bw_g == 1) {
            bool changed = true; int iters = 0;
            while (changed && iters++ < 4096) {
                changed = false;
                for (size_t br = 0; br < n_blocks; br++) {
                    size_t b = n_blocks - 1 - br;
                    uint32_t s0 = succ_off[b], s1 = succ_off[b + 1];
                    uint32_t ns_b = s1 - s0;
                    uint64_t lo = 0;
                    if (ns_b == 1) lo = live_in[succ_idx[s0]];
                    else if (ns_b == 2) lo = live_in[succ_idx[s0]] | live_in[succ_idx[s0 + 1]];
                    else if (ns_b > 2) {
                        for (uint32_t s = s0; s < s1; s++) lo |= live_in[succ_idx[s]];
                    }
                    live_out[b] = lo;
                    uint64_t nv_w = (lo & ~kill_g[b]) | gen_g[b];
                    if (nv_w != live_in[b]) {
                        live_in[b] = nv_w;
                        changed = true;
                    }
                }
                if (!has_backedge) break;
            }
            // Extend last_use in reverse block order (seen_lo) and first_pos in
            // forward block order (seen_li): each global bit is visited at most once.
            uint64_t seen_lo = 0, seen_li = 0;
            for (size_t br = 0; br < n_blocks; br++) {
                size_t b = n_blocks - 1 - br;
                uint64_t wo = live_out[b] & ~seen_lo;
                if (wo) {
                    seen_lo |= wo;
                    uint32_t covl = A64_COVL(block_last[b]);
                    do {
                        uint32_t g = ra_ctz64(wo);
                        wo &= wo - 1;
                        uint32_t vi = glob_vi[g];
                        if (covl > last_use[vi]) last_use[vi] = covl;
                    } while (wo);
                }
            }
            for (size_t b = 0; b < n_blocks; b++) {
                uint64_t wl = live_in[b] & ~seen_li;
                if (wl) {
                    seen_li |= wl;
                    uint32_t covf = A64_COVF(block_first[b]);
                    do {
                        uint32_t g = ra_ctz64(wl);
                        wl &= wl - 1;
                        uint32_t vi = glob_vi[g];
                        if (covf < first_pos[vi]) first_pos[vi] = covf;
                    } while (wl);
                }
            }
        } else {
            bool changed = true; int iters = 0;
            while (changed && iters++ < 4096) {
                changed = false;
                for (size_t br = 0; br < n_blocks; br++) {
                    size_t b = n_blocks - 1 - br;
                    uint64_t *LO = live_out + b * bw_g;
                    uint32_t s0 = succ_off[b], s1 = succ_off[b + 1];
                    uint32_t ns_b = s1 - s0;
                    if (ns_b == 1) {
                        const uint64_t *SLI0 = live_in + (size_t)succ_idx[s0] * bw_g;
                        memcpy(LO, SLI0, bw_g * sizeof(uint64_t));
                    } else if (ns_b == 2) {
                        const uint64_t *SLI0 = live_in + (size_t)succ_idx[s0] * bw_g;
                        const uint64_t *SLI1 = live_in + (size_t)succ_idx[s0 + 1] * bw_g;
                        for (size_t wi = 0; wi < bw_g; wi++) LO[wi] = SLI0[wi] | SLI1[wi];
                    } else if (ns_b > 2) {
                        const uint64_t *SLI0 = live_in + (size_t)succ_idx[s0] * bw_g;
                        memcpy(LO, SLI0, bw_g * sizeof(uint64_t));
                        for (uint32_t s = s0 + 1; s < s1; s++) {
                            const uint64_t *SLI = live_in + (size_t)succ_idx[s] * bw_g;
                            for (size_t wi = 0; wi < bw_g; wi++) LO[wi] |= SLI[wi];
                        }
                    }
                    uint64_t *LI = live_in + b * bw_g;
                    const uint64_t *K = kill_g + b * bw_g;
                    const uint64_t *G = gen_g + b * bw_g;
                    if (!has_backedge) {
                        for (size_t wi = 0; wi < bw_g; wi++)
                            LI[wi] = (LO[wi] & ~K[wi]) | G[wi];
                    } else {
                        bool diff = false;
                        for (size_t wi = 0; wi < bw_g; wi++) {
                            uint64_t nv_w = (LO[wi] & ~K[wi]) | G[wi];
                            scr_g[wi] = nv_w;
                            if (nv_w != LI[wi]) diff = true;
                        }
                        if (diff) {
                            memcpy(LI, scr_g, bw_g * sizeof(uint64_t));
                            changed = true;
                        }
                    }
                }
                if (!has_backedge) break;
            }

            // Extend last_use in reverse block order (seen_lo) and first_pos in
            // forward block order (seen_li): each global bit is visited at most once.
            uint64_t *seen_lo = scr_g;
            uint64_t *seen_li = scr_g + bw_g;
            memset(seen_lo, 0, bw_g * sizeof(uint64_t));
            memset(seen_li, 0, bw_g * sizeof(uint64_t));
            for (size_t br = 0; br < n_blocks; br++) {
                size_t b = n_blocks - 1 - br;
                uint32_t covl = A64_COVL(block_last[b]);
                const uint64_t *LO = live_out + b * bw_g;
                for (size_t wi = 0; wi < bw_g; wi++) {
                    uint64_t wo = LO[wi] & ~seen_lo[wi];
                    if (wo) {
                        seen_lo[wi] |= wo;
                        do {
                            uint32_t g = (uint32_t)(wi * 64u) + ra_ctz64(wo);
                            wo &= wo - 1;
                            uint32_t vi = glob_vi[g];
                            if (covl > last_use[vi]) last_use[vi] = covl;
                        } while (wo);
                    }
                }
            }
            for (size_t b = 0; b < n_blocks; b++) {
                uint32_t covf = A64_COVF(block_first[b]);
                const uint64_t *LI = live_in + b * bw_g;
                for (size_t wi = 0; wi < bw_g; wi++) {
                    uint64_t wl = LI[wi] & ~seen_li[wi];
                    if (wl) {
                        seen_li[wi] |= wl;
                        do {
                            uint32_t g = (uint32_t)(wi * 64u) + ra_ctz64(wl);
                            wl &= wl - 1;
                            uint32_t vi = glob_vi[g];
                            if (covf < first_pos[vi]) first_pos[vi] = covf;
                        } while (wl);
                    }
                }
            }
        }
    }

    // --- 3b. crosses_call (exact, skipping call-free blocks, O(bw_g) per call). ---
    if (env->linear_crosses) {
        uint8_t *is_call = (uint8_t *)calloc(pos ? pos : 1, sizeof(uint8_t));
        uint32_t op_pos = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = blk_h[b];
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            for (size_t i = 0; i < no; i++) {
                if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, bk, i)) == OP_TYPE_LLVM_CALL)
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
    } else if (n_calls_tot > 0) {
        uint64_t *live_g  = bw_g ? (live_out + n_blocks * bw_g * 3 + bw_g) : NULL;
        uint64_t *cross_g = bw_g ? (live_g + bw_g) : NULL;
        if (bw_g) memset(cross_g, 0, bw_g * sizeof(uint64_t));
        for (size_t b = 0; b < n_blocks; b++) {
            uint32_t c0 = call_off[b], c1 = call_off[b + 1];
            if (c0 == c1) continue;
            uint32_t bfirst = block_first[b];
            if (bw_g) {
                const uint64_t *LO = live_out + b * bw_g;
                const uint64_t *K  = kill_g + b * bw_g;
                const uint64_t *G  = gen_g + b * bw_g;
                // Fast path: if no global is defined in b and no global dies in b
                // (all gen_g bits are already in live_out), the live global set is
                // identically LO at every call in b — skip the backwards op walk.
                bool const_live = (b > 0 || n_entry_args == 0);
                if (const_live) {
                    for (size_t wi = 0; wi < bw_g; wi++) {
                        if (K[wi] != 0 || (G[wi] & ~LO[wi]) != 0) {
                            const_live = false;
                            break;
                        }
                    }
                }
                if (const_live) {
                    for (size_t wi = 0; wi < bw_g; wi++) cross_g[wi] |= LO[wi];
                } else {
                    MLIR_BlockHandle bk = blk_h[b];
                    size_t no = MLIR_GetBlockNumOps(ctx, bk);
                    uint32_t fci = first_call[b];
                    memcpy(live_g, LO, bw_g * sizeof(uint64_t));
                    if (no > 0) {
                        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
                        size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
                        for (size_t s = 0; s < ns; s++) {
                            size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, s);
                            for (size_t k = 0; k < nso; k++) {
                                uint32_t vi;
                                if (RA_VMAP_GET(MLIR_GetOpSuccessorOperand(ctx, term, s, k), &vi) &&
                                    glob_id[vi] != UINT32_MAX) a64_bs_set(live_g, glob_id[vi]);
                            }
                        }
                    }
                    uint32_t c_idx = c1;
                    uint32_t next_call_i = call_pos_arr[c_idx - 1] - bfirst;
                    for (size_t ii = no; ii-- > (size_t)fci; ) {
                        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, ii);
                        if (ra_is_folded(T, ctx, op)) continue;
                        if (ii == (size_t)next_call_i) {
                            uint32_t rg = UINT32_MAX;
                            if (MLIR_GetOpNumResults(ctx, op)) {
                                uint32_t rvi;
                                if (RA_VMAP_GET(MLIR_GetOpResult(ctx, op, 0), &rvi))
                                    rg = glob_id[rvi];
                            }
                            bool had_rg = (rg != UINT32_MAX && a64_bs_test(live_g, rg));
                            if (had_rg) a64_bs_clear(live_g, rg);
                            for (size_t wi = 0; wi < bw_g; wi++) cross_g[wi] |= live_g[wi];
                            if (had_rg) a64_bs_set(live_g, rg);
                            c_idx--;
                            if (c_idx > c0) next_call_i = call_pos_arr[c_idx - 1] - bfirst;
                            else next_call_i = UINT32_MAX;
                        }
                        size_t nr = MLIR_GetOpNumResults(ctx, op);
                        for (size_t r = 0; r < nr; r++) {
                            uint32_t vi;
                            if (RA_VMAP_GET(MLIR_GetOpResult(ctx, op, r), &vi) &&
                                glob_id[vi] != UINT32_MAX) a64_bs_clear(live_g, glob_id[vi]);
                        }
                        size_t nop = MLIR_GetOpNumOperands(ctx, op);
                        for (size_t k = 0; k < nop; k++) {
                            uint32_t vi;
                            if (RA_VMAP_GET(MLIR_GetOpOperand(ctx, op, k), &vi) &&
                                glob_id[vi] != UINT32_MAX) a64_bs_set(live_g, glob_id[vi]);
                        }
                        MLIR_ValueHandle mbase, midx;
                        if (ra_memfuse(T, ctx, op, &mbase, &midx)) {
                            uint32_t vi;
                            if (RA_VMAP_GET(mbase, &vi) && glob_id[vi] != UINT32_MAX)
                                a64_bs_set(live_g, glob_id[vi]);
                            if (RA_VMAP_GET(midx, &vi) && glob_id[vi] != UINT32_MAX)
                                a64_bs_set(live_g, glob_id[vi]);
                        }
                        MLIR_ValueHandle sfx;
                        if (ra_shiftfuse(T, ctx, op, &sfx)) {
                            uint32_t vi;
                            if (RA_VMAP_GET(sfx, &vi) && glob_id[vi] != UINT32_MAX)
                                a64_bs_set(live_g, glob_id[vi]);
                        }
                    }
                }
            }
            // Local values defined in block b: check against block b's call positions.
            {
                uint32_t min_cp = call_pos_arr[c0];
                uint32_t max_cp = call_pos_arr[c1 - 1];
                uint32_t min_use = A64_USEP(min_cp);
                uint32_t max_def = A64_DEFP(max_cp);
                uint32_t v0 = blk_vstart[b], v1 = blk_vstart[b + 1];
                for (uint32_t vi = v0; vi < v1; vi++) {
                    if (glob_id && glob_id[vi] != UINT32_MAX) continue;
                    if (last_use[vi] <= min_use || def_pos[vi] >= max_def) continue;
                    for (uint32_t c = c0; c < c1; c++) {
                        uint32_t cp = call_pos_arr[c];
                        if (def_pos[vi] < A64_DEFP(cp) && last_use[vi] > A64_USEP(cp)) {
                            crosses[vi] = 1;
                            break;
                        }
                    }
                }
            }
        }
        if (bw_g) {
            for (size_t wi = 0; wi < bw_g; wi++) {
                uint64_t wc = cross_g[wi];
                while (wc) {
                    uint32_t g = (uint32_t)(wi * 64u) + ra_ctz64(wc);
                    wc &= wc - 1;
                    crosses[glob_vi[g]] = 1;
                }
            }
        }
    }

    // --- 3c. Phi edge-copy coalescing hints (terminator-only scan). ---
    int32_t *hint_src = NULL;
    bool has_hints = false;
    if (env->phi_coalesce && n_blocks > 1) {
        hint_src = (int32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(int32_t));
        memset(hint_src, 0xff, nv_sz * sizeof(int32_t));
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_OpHandle op = MLIR_GetBlockTerminator(ctx, blk_h[b]);
            if (op == MLIR_INVALID_HANDLE) continue;
            size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
            if (ns == 0) continue;
            for (size_t s = 0; s < ns; s++) {
                MLIR_BlockHandle succ = MLIR_GetOpSuccessor(ctx, op, s);
                size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                for (size_t k = 0; k < nso; k++) {
                    uint32_t vV, vB;
                    MLIR_ValueHandle Vh = MLIR_GetOpSuccessorOperand(ctx, op, s, k);
                    MLIR_ValueHandle Bh = MLIR_GetBlockArg(ctx, succ, k);
                    if (!RA_VMAP_GET(Vh, &vV)) continue;
                    if (!RA_VMAP_GET(Bh, &vB)) continue;
                    if (nuse[vV] != 1) continue;
                    if (last_use[vV] != first_pos[vB]) continue;
                    if (hint_src[vB] >= 0) continue;
                    bool through = false;
                    for (size_t s2 = 0; s2 < ns && !through; s2++) {
                        size_t n2 = MLIR_GetOpNumSuccessorOperands(ctx, op, s2);
                        for (size_t k2 = 0; k2 < n2; k2++)
                            if (MLIR_GetOpSuccessorOperand(ctx, op, s2, k2) == Bh) {
                                through = true; break;
                            }
                    }
                    if (through) continue;
                    hint_src[vB] = (int32_t)vV;
                    has_hints = true;
                }
            }
        }
    }
    #undef RA_VMAP_GET

    // --- 4. Linear scan over the pool, loop-weighted eviction. ---
    uint32_t *order = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
    for (uint32_t i = 0; i < nv; i++) order[i] = i;
    bool sorted = true;
    for (uint32_t i = 1; i < nv; i++) {
        if (first_pos[i] < first_pos[i - 1]) { sorted = false; break; }
    }
    if (!sorted) {   // stable bottom-up merge sort by first_pos (ties keep numbering order).
        uint32_t *tmp = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
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
    }

    // Active list: (value index, pool index, end pos). busy_mask tracks occupied pool regs.
    uint32_t act_vi[RA_MAXPOOL];
    uint32_t act_end[RA_MAXPOOL];
    uint8_t  act_pk[RA_MAXPOOL];
    size_t nact = 0;
    uint32_t busy_mask = 0;
    uint32_t pool_mask = (T->npool >= 32) ? 0xFFFFFFFFu : ((1u << T->npool) - 1u);
    uint32_t caller_mask = (T->ncaller >= 32) ? 0xFFFFFFFFu : ((1u << T->ncaller) - 1u);
    uint32_t callee_mask = pool_mask & ~caller_mask;

    for (uint32_t k = 0; k < nv; k++) {
        uint32_t vi = order[k];
        uint32_t fp = first_pos[vi];
        // Retire active intervals ending before this one starts (strict, so an
        // operand whose last use == this def keeps its register).
        for (size_t a = 0; a < nact; ) {
            if (act_end[a] < fp) {
                busy_mask &= ~(1u << act_pk[a]);
                nact--;
                act_vi[a] = act_vi[nact]; act_end[a] = act_end[nact]; act_pk[a] = act_pk[nact];
            } else a++;
        }
        // Dead-on-def / never-used values keep their slot.
        uint32_t lu = last_use[vi];
        if (lu <= fp) continue;
        bool callee_only = crosses[vi];
        if (has_hints && hint_src[vi] >= 0) {
            int32_t vV = hint_src[vi];
            for (size_t a = 0; a < nact; a++) {
                if (act_vi[a] != (uint32_t)vV) continue;
                if (act_end[a] != fp) break;   // not a clean death
                if (callee_only && act_pk[a] < T->ncaller) break; // reg class
                uint8_t pk = act_pk[a];
                home_pk[vi] = (int8_t)pk;        // inherit V's physical register
                act_vi[a] = vi; act_end[a] = lu; // ownership transfers; reg stays busy
                break;
            }
            if (home_pk[vi] >= 0) continue;
        }
        uint32_t avail = (~busy_mask) & (callee_only ? callee_mask : pool_mask);
        if (avail) {
            uint32_t found = ra_ctz64((uint64_t)avail);
            busy_mask |= (1u << found);
            act_vi[nact] = vi; act_end[nact] = lu; act_pk[nact] = (uint8_t)found; nact++;
            home_pk[vi] = (int8_t)found;
            continue;
        }
        size_t evict_a = (size_t)-1; uint64_t best_w = 0; uint32_t best_end = 0;
        for (size_t a = 0; a < nact; a++) {
            if (callee_only && act_pk[a] < T->ncaller) continue;
            uint64_t aw = weight[act_vi[a]]; uint32_t ae = act_end[a];
            if (evict_a == (size_t)-1 || aw < best_w || (aw == best_w && ae > best_end)) {
                evict_a = a; best_w = aw; best_end = ae;
            }
        }
        if (evict_a != (size_t)-1 && best_w >= weight[vi]) evict_a = (size_t)-1;
        if (evict_a != (size_t)-1) {
            home_pk[act_vi[evict_a]] = -1;          // evicted -> slot
            act_vi[evict_a] = vi; act_end[evict_a] = lu; // reuse its register
            home_pk[vi] = (int8_t)act_pk[evict_a];
        }
    }

    // --- 4b. Assign frame slots to spilled values with min-heap reuse. ---
    int32_t next_slot = 0;
    {
        size_t n_spills = 0;
        for (uint32_t vi = 0; vi < nv; vi++) if (home_pk[vi] < 0) n_spills++;
        if (v_dense) {
            if (n_spills > 0) {
                if (v_span <= n_spills * 4u + 64u) {
                    int32_t *sb = (int32_t *)ra_pool_ensure(&g_ra_sm_pool, v_span * sizeof(int32_t));
                    sm_init_dense_buf(sm, sb, min_vh, v_span);
                } else {
                    sm_reserve(sm, n_spills);
                }
            }
            if (nv > n_spills) {
                int32_t *rb = (int32_t *)ra_pool_ensure(&g_ra_rm_pool, v_span * sizeof(int32_t));
                sm_init_dense_buf(rm, rb, min_vh, v_span);
            }
        } else {
            if (n_spills > 0) sm_reserve(sm, n_spills);
            if (nv > n_spills) sm_reserve(rm, nv - n_spills);
        }
        if (n_spills > 0) {
            bool no_reuse = env->no_slot_reuse;
            uint32_t *sl_slot = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
            uint32_t *sl_end  = (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
            uint32_t *freelist= (uint32_t *)ra_bump(mbuf, &moff, nv_sz * sizeof(uint32_t));
            size_t n_sact = 0, n_free = 0;
            int32_t *sm_d = sm->dense;
            for (uint32_t k = 0; k < nv; k++) {
                uint32_t vi = order[k];
                uint32_t fp = first_pos[vi];
                while (n_sact > 0 && sl_end[0] < fp) {
                    freelist[n_free++] = sl_slot[0];
                    sl_heap_pop(sl_end, sl_slot, &n_sact);
                }
                if (home_pk[vi] >= 0) continue;           // register-homed: no slot
                uint32_t slot = (!no_reuse && n_free > 0) ? freelist[--n_free]
                                                          : (uint32_t)next_slot++;
                if (last_use[vi] > fp) {
                    sl_heap_push(sl_end, sl_slot, &n_sact, last_use[vi], slot);
                } else if (!no_reuse) {
                    freelist[n_free++] = slot;
                }
                if (sm_d) {
                    uintptr_t idx = (uintptr_t)handle[vi] - min_vh;
                    if (sm_d[idx] < 0) { sm_d[idx] = (int32_t)slot; sm->n++; }
                } else {
                    sm_put(sm, handle[vi], (int32_t)slot);
                }
            }
        }
    }
    if (env->spill_stats) {
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (home_pk[vi] >= 0) continue;
            MLIR_OpHandle dop = MLIR_GetValueDefiningOp(ctx, handle[vi]);
            const char *nm = "(blockarg)";
            int nl = 10;
            if (dop) { string s = MLIR_GetOpTypeName(ctx, MLIR_GetOpType(ctx, dop)); nm = s.str; nl = (int)s.size; }
            const char *dead = (last_use[vi] <= first_pos[vi]) ? "deaddef" : "live";
            char ops[128]; int op_off = 0; ops[0] = '\0';
            if (dop) {
                size_t non = MLIR_GetOpNumOperands(ctx, dop);
                for (size_t k = 0; k < non && op_off < 100; k++) {
                    MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, dop, k);
                    MLIR_OpHandle odef = MLIR_GetValueDefiningOp(ctx, ov);
                    const char *on; int onl;
                    if (!odef) { on = "barg"; onl = 4; }
                    else if (ra_is_const(T, ctx, odef)) { on = "const"; onl = 5; }
                    else { string os = MLIR_GetOpTypeName(ctx, MLIR_GetOpType(ctx, odef)); on = os.str; onl = (int)os.size; }
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
    int32_t *rm_d = rm->dense;
    if (rm_d) {
        size_t rm_n = rm->n;
        for (uint32_t vi = 0; vi < nv; vi++) {
            int pk = home_pk[vi];
            if (pk < 0) continue;
            uintptr_t idx = (uintptr_t)handle[vi] - min_vh;
            if (rm_d[idx] < 0) { rm_d[idx] = T->pool[pk]; rm_n++; }
            if (pk >= T->ncaller) used_mask |= (1u << (pk - T->ncaller));
        }
        rm->n = rm_n;
    } else {
        for (uint32_t vi = 0; vi < nv; vi++) {
            int pk = home_pk[vi];
            if (pk < 0) continue;
            sm_put(rm, handle[vi], T->pool[pk]);
            if (pk >= T->ncaller) used_mask |= (1u << (pk - T->ncaller));
        }
    }

    #undef A64_USE_WEIGHT_AT
    #undef A64_USEP
    #undef A64_DEFP
    #undef A64_BARGP
    #undef A64_COVL
    #undef A64_COVF
    if (vmap.cap) a64_hh_free(&vmap);
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
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        if (no > maxops) maxops = no;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpNumRegions(ctx, op) > 0) return 0;   // bail: nested regions
            cap += MLIR_GetOpNumResults(ctx, op);
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
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (ra_is_const(T, ctx, op)) continue; // remat'd, no home
            // Folded linmem spine ops (zext/add/inttoptr) emit no code and get
            // no home register; their result is consumed by the fused load/store.
            if (ra_is_folded(T, ctx, op)) continue;
            size_t nr = MLIR_GetOpNumResults(ctx, op);
            for (size_t r = 0; r < nr; r++) {
                vals[nv] = MLIR_GetOpResult(ctx, op, r);
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
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            bool is_term = (i + 1 == no);
            bool is_call = MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_CALL;
            // A folded spine op (zext/add/inttoptr) emits no code: its operand
            // reads must NOT extend liveness. The fused load/store re-registers
            // base+idx as live uses at its own position (below).
            if (ra_is_folded(T, ctx, op)) continue;
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            for (size_t k = 0; k < nop; k++) {
                int32_t vi;
                if (!sm_get(&idx, MLIR_GetOpOperand(ctx, op, k), &vi)) continue;
                usedv[vi] = 1; use_cnt[vi]++;
                if ((int32_t)b != def_block[vi] || is_term || is_call)
                    disq[vi] = 1;
                else if ((int32_t)i > last_use[vi])
                    last_use[vi] = (int32_t)i;
            }
            // Branch successor (block-arg) operands are a separate operand list
            // resolved via slot-to-slot edge copies (copy_slot reads the frame
            // slot), so any value passed across an edge must stay in memory.
            size_t nsucc = MLIR_GetOpNumSuccessors(ctx, op);
            for (size_t s = 0; s < nsucc; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                for (size_t k = 0; k < nso; k++) {
                    int32_t vi;
                    if (!sm_get(&idx, MLIR_GetOpSuccessorOperand(ctx, op, s, k), &vi))
                        continue;
                    usedv[vi] = 1; disq[vi] = 1; use_cnt[vi]++;
                    // Record the destination block-arg + terminator so a sole-use
                    // edge operand can later be coalesced into the block-arg's
                    // home register (the mov vanishes in emit_edge_copies).
                    edge_arg[vi]  = MLIR_GetBlockArg(ctx, MLIR_GetOpSuccessor(ctx, op, s), k);
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
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        size_t ncall = 0;
        for (size_t i = 0; i < no; i++)
            if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, bk, i)) == OP_TYPE_LLVM_CALL)
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
            nba += MLIR_GetBlockNumArgs(ctx, MLIR_GetRegionBlock(ctx, reg, b));
        if (nba) {
            MLIR_ValueHandle *bv =
                (MLIR_ValueHandle *)arena_alloc(ar, nba * sizeof(MLIR_ValueHandle));
            int32_t *bscore = (int32_t *)arena_alloc(ar, nba * sizeof(int32_t));
            SlotMap bidx = {0}; bidx.arena = ar;
            size_t m = 0;
            for (size_t b = 1; b < n_blocks; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
                size_t na = MLIR_GetBlockNumArgs(ctx, bk);
                for (size_t a = 0; a < na; a++) {
                    bv[m] = MLIR_GetBlockArg(ctx, bk, a); bscore[m] = 0;
                    sm_put(&bidx, bv[m], (int32_t)m); m++;
                }
            }
            for (size_t b = 0; b < n_blocks; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
                size_t no = MLIR_GetBlockNumOps(ctx, bk);
                for (size_t i = 0; i < no; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                    size_t nop = MLIR_GetOpNumOperands(ctx, op);
                    for (size_t k = 0; k < nop; k++) {
                        int32_t bi;
                        if (sm_get(&bidx, MLIR_GetOpOperand(ctx, op, k), &bi)) bscore[bi]++;
                    }
                    size_t nsucc = MLIR_GetOpNumSuccessors(ctx, op);
                    for (size_t sx = 0; sx < nsucc; sx++) {
                        size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, sx);
                        for (size_t k = 0; k < nso; k++) {
                            int32_t bi;
                            if (sm_get(&bidx, MLIR_GetOpSuccessorOperand(ctx, op, sx, k), &bi))
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
            if (MLIR_GetOpNumSuccessors(ctx, term) != 1) continue;
            // Value must be defined immediately before its block's terminator.
            MLIR_BlockHandle db = MLIR_GetRegionBlock(ctx, reg, def_block[vi]);
            size_t dno = MLIR_GetBlockNumOps(ctx, db);
            if (dno < 2 || def_pos[vi] != (int32_t)dno - 2) continue;
            if (MLIR_GetBlockOp(ctx, db, dno - 1) != term) continue;
            // Destination block-arg must be register-homed.
            int32_t R;
            if (!sm_get(rm, ba, &R)) continue;
            // The block-arg's register must not be read by any OTHER operand of
            // this edge (no pass-through of the old phi value into R).
            bool conflict = false;
            size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, 0);
            for (size_t k = 0; k < nso && !conflict; k++) {
                MLIR_ValueHandle o = MLIR_GetOpSuccessorOperand(ctx, term, 0, k);
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
                MLIR_OpHandle dop = MLIR_GetValueDefiningOp(ctx, vals[vi]);
                int32_t ui;
                if (dop != MLIR_INVALID_HANDLE && cast_src(ctx, dop, &csrc) &&
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

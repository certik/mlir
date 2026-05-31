// wmir register allocator. See mlir_wmir_regalloc.h for the design.
//
// Implementation outline:
//
//   - Linearisation: a flat array `pos[i] = op_i` with one position
//     per op, plus per-block `[first_pos, last_pos)` ranges. Block
//     arguments are defined at the block's first position.
//
//   - Liveness: standard backward dataflow.
//
//       live_out(B) = ∪ live_in(succ(B))
//       live_in(B)  = (live_out(B) \ defs(B)) ∪ uses(B)
//
//     We iterate to fixpoint over the blocks in reverse-postorder
//     until no live_in set changes. For each value `v` we then
//     record:
//
//       def_pos       = position of the op that produces v
//                       (or block.first_pos if v is a block arg)
//       last_use_pos  = max over (ops using v in the block ending
//                       its live range, and successor edges that
//                       use it as a successor operand)
//       crosses_call  = true if any `wmir.call` lies between
//                       def_pos and last_use_pos
//
//   - Allocator: linear scan. Pool is x11..x18 (8 caller-saved
//     integer regs). Iterate intervals by start position; on each:
//
//       * Expire all active intervals with end < interval.start.
//       * If the value is float-typed or `crosses_call=true`, give
//         it a stack slot directly.
//       * Else if a pool reg is free, assign it. Otherwise pick the
//         active interval with the latest end and either evict that
//         one (spill it to a slot, free its reg, assign the reg to
//         the current value) or spill the current one — choose
//         whichever has the latest end.
//
//   - Output is consumed by mlir_wmir_to_aarch64.c.

#include "mlir_wmir_regalloc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/string.h>

// =============================================================================
// Local helpers.
// =============================================================================
static bool type_is_int(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size != 3) return false;
    if (s.str[0] != 'i') return false;
    return memcmp(s.str + 1, "32", 2) == 0 ||
           memcmp(s.str + 1, "64", 2) == 0;
}

static bool value_is_int(MLIR_Context *ctx, MLIR_ValueHandle v) {
    return type_is_int(ctx, MLIR_GetValueType(v));
}

static bool value_is_i64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    string s = MLIR_GetTypeString(ctx, MLIR_GetValueType(v));
    return s.size == 3 && s.str[0] == 'i' && memcmp(s.str + 1, "64", 2) == 0;
}

// =============================================================================
// Value -> internal index map. Built once per function.
// =============================================================================
// Open-addressing hash table (linear probing). Key 0 (MLIR_INVALID_HANDLE)
// marks an empty slot; SSA value handles are always non-zero. Only get/set by
// key are used (never iterated for output), so probe order is irrelevant to
// determinism. Uses only 32-bit-safe arithmetic for the wasm32 self-host.
typedef struct {
    MLIR_ValueHandle *keys;  // 0 == empty
    uint32_t         *idx;   // index into the value table
    size_t            n, cap;
} VMap;

static uint32_t vmap_hash(MLIR_ValueHandle v) {
    uint32_t h = (uint32_t)v ^ (uint32_t)(v >> 16);
    h ^= h >> 15; h *= 2654435761u; h ^= h >> 13;
    return h;
}
static void vmap_put_raw(MLIR_ValueHandle *keys, uint32_t *idx, size_t cap,
                         MLIR_ValueHandle v, uint32_t i) {
    size_t mask = cap - 1;
    size_t p = (size_t)vmap_hash(v) & mask;
    while (keys[p] != 0) p = (p + 1) & mask;
    keys[p] = v; idx[p] = i;
}
static void vmap_grow(VMap *m) {
    size_t nc = m->cap ? m->cap * 2 : 64;
    MLIR_ValueHandle *nk = calloc(nc, sizeof(*nk));
    uint32_t         *ni = malloc(nc * sizeof(*ni));
    for (size_t i = 0; i < m->cap; i++)
        if (m->keys[i] != 0) vmap_put_raw(nk, ni, nc, m->keys[i], m->idx[i]);
    free(m->keys); free(m->idx);
    m->keys = nk; m->idx = ni; m->cap = nc;
}
static void vmap_set(VMap *m, MLIR_ValueHandle v, uint32_t i) {
    if (m->cap == 0 || (m->n + 1) * 4 >= m->cap * 3) vmap_grow(m);
    size_t mask = m->cap - 1;
    size_t p = (size_t)vmap_hash(v) & mask;
    while (m->keys[p] != 0) {
        if (m->keys[p] == v) { m->idx[p] = i; return; }
        p = (p + 1) & mask;
    }
    m->keys[p] = v; m->idx[p] = i; m->n++;
}
static bool vmap_get(const VMap *m, MLIR_ValueHandle v, uint32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t p = (size_t)vmap_hash(v) & mask;
    while (m->keys[p] != 0) {
        if (m->keys[p] == v) { *out = m->idx[p]; return true; }
        p = (p + 1) & mask;
    }
    return false;
}
static void vmap_free(VMap *m) { free(m->keys); free(m->idx); memset(m, 0, sizeof(*m)); }

// =============================================================================
// Per-block index map.
// =============================================================================
typedef struct {
    MLIR_BlockHandle *bs;
    uint32_t         *idx;
    size_t            n, cap;
} BMap;
static void bmap_set(BMap *m, MLIR_BlockHandle b, uint32_t i) {
    if (m->n == m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 8;
        m->bs  = realloc(m->bs,  nc * sizeof(*m->bs));
        m->idx = realloc(m->idx, nc * sizeof(*m->idx));
        m->cap = nc;
    }
    m->bs[m->n] = b; m->idx[m->n] = i; m->n++;
}
static bool bmap_get(const BMap *m, MLIR_BlockHandle b, uint32_t *out) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->bs[i] == b) { *out = m->idx[i]; return true; }
    }
    return false;
}
static void bmap_free(BMap *m) { free(m->bs); free(m->idx); memset(m, 0, sizeof(*m)); }

// =============================================================================
// Bitset helpers (over n_values bits).
// =============================================================================
static inline size_t bs_words(size_t n) { return (n + 63u) / 64u; }
static inline void bs_set  (uint64_t *bs, uint32_t i) { bs[i >> 6] |=  (1ull << (i & 63u)); }
static inline void bs_clear(uint64_t *bs, uint32_t i) { bs[i >> 6] &= ~(1ull << (i & 63u)); }
static inline bool bs_test (const uint64_t *bs, uint32_t i) { return (bs[i >> 6] >> (i & 63u)) & 1u; }
static inline void bs_zero (uint64_t *bs, size_t n)         { memset(bs, 0, bs_words(n) * 8u); }
static inline bool bs_equal(const uint64_t *a, const uint64_t *b, size_t n) {
    return memcmp(a, b, bs_words(n) * 8u) == 0;
}
// dst = a ∪ b
static inline void bs_or(uint64_t *dst, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t w = bs_words(n);
    for (size_t i = 0; i < w; i++) dst[i] = a[i] | b[i];
}
// dst = a \ b
static inline void bs_diff(uint64_t *dst, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t w = bs_words(n);
    for (size_t i = 0; i < w; i++) dst[i] = a[i] & ~b[i];
}

// =============================================================================
// Internal per-value record.
// =============================================================================
typedef struct {
    MLIR_ValueHandle handle;
    bool             is_int;        // false for f32/f64/ptr/etc -> always slot
    uint32_t         def_pos;       // INT32_MAX if undefined yet
    uint32_t         first_pos;     // interval START: min(def_pos, earliest use/live-in).
                                    // For mem2reg block args a use can appear BEFORE the
                                    // def in linear (source) order (e.g. a loop-exit block
                                    // ordered before the header that defines the phi); the
                                    // live interval must cover that earlier position too.
    uint32_t         last_use_pos;  // 0 if no use seen yet
    bool             crosses_call;
    bool             is_const;      // result of a wmir.const (rematerializable)
    bool             c_is_i64;      // is_const: 1 if i64 else 0
    int64_t          cval;          // is_const: the constant value
} VInfo;

// One slot in the linear-scan active list.
typedef struct {
    uint32_t vi;
    uint8_t  reg;
    uint8_t  pool_idx;
} Active;

// =============================================================================
// Main entry.
// =============================================================================
WmirRegAlloc *wmir_regalloc_run(MLIR_Context *ctx, MLIR_OpHandle func) {
    if (MLIR_GetOpNumRegions(func) < 1) return NULL;
    MLIR_RegionHandle region = MLIR_GetOpRegion(func, 0);
    size_t n_blocks = MLIR_GetRegionNumBlocks(region);
    if (n_blocks == 0) return NULL;

    // --- 1. Linearise. Assign a position to every block-arg def and
    //        every op, in source order. Record per-block ranges. ---
    BMap bmap = {0};
    VMap vmap = {0};
    uint32_t pos = 0;

    // Per-block ranges. We allocate up-front using a flat array.
    uint32_t *block_first_pos = calloc(n_blocks, sizeof(uint32_t));
    uint32_t *block_last_pos  = calloc(n_blocks, sizeof(uint32_t));
    MLIR_BlockHandle *blocks  = calloc(n_blocks, sizeof(MLIR_BlockHandle));

    // First pass: assign each block an internal index + count values.
    size_t n_values = 0;
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        blocks[bi] = b;
        bmap_set(&bmap, b, (uint32_t)bi);
        n_values += MLIR_GetBlockNumArgs(b);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, i);
            n_values += MLIR_GetOpNumResults(op);
        }
    }

    if (n_values == 0) {
        // Empty function body — return a degenerate allocation.
        WmirRegAlloc *ra = calloc(1, sizeof(*ra));
        free(block_first_pos); free(block_last_pos); free(blocks);
        bmap_free(&bmap); vmap_free(&vmap);
        return ra;
    }

    VInfo *vinfo = calloc(n_values, sizeof(*vinfo));
    for (size_t i = 0; i < n_values; i++) {
        vinfo[i].def_pos = UINT32_MAX;
        vinfo[i].first_pos = UINT32_MAX;
    }
    uint32_t next_vidx = 0;

    // Second pass: walk blocks, assign positions, record defs.
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle b = blocks[bi];
        block_first_pos[bi] = pos;

        // Block args defined at block-entry position.
        size_t na = MLIR_GetBlockNumArgs(b);
        for (size_t i = 0; i < na; i++) {
            MLIR_ValueHandle a = MLIR_GetBlockArg(b, i);
            uint32_t vi = next_vidx++;
            vmap_set(&vmap, a, vi);
            vinfo[vi].handle  = a;
            vinfo[vi].is_int  = value_is_int(ctx, a);
            vinfo[vi].def_pos = pos;
            vinfo[vi].first_pos = pos;
        }

        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, i);
            bool is_const_op = (MLIR_GetOpType(op) == OP_TYPE_WMIR_CONST);
            int64_t cval = 0;
            if (is_const_op) {
                MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, "value");
                cval = a ? MLIR_GetAttributeInteger(a) : 0;
            }
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t k = 0; k < nr; k++) {
                MLIR_ValueHandle r = MLIR_GetOpResult(op, k);
                uint32_t vi = next_vidx++;
                vmap_set(&vmap, r, vi);
                vinfo[vi].handle  = r;
                vinfo[vi].is_int  = value_is_int(ctx, r);
                vinfo[vi].def_pos = pos;
                vinfo[vi].first_pos = pos;
                // A wmir.const result is rematerializable: it is re-emitted as
                // a mov-immediate at each use, so it never needs a register or
                // a spill slot and adds no register pressure.
                if (is_const_op && vinfo[vi].is_int) {
                    vinfo[vi].is_const = true;
                    vinfo[vi].c_is_i64 = value_is_i64(ctx, r);
                    vinfo[vi].cval     = cval;
                }
            }
            pos++;  // one position per op
        }
        block_last_pos[bi] = pos;  // exclusive
    }

    // --- 2. Liveness dataflow. ---
    // For each block we maintain live_in and live_out bitsets sized
    // n_values. We also precompute use(B) and def(B) bitsets.
    size_t bw = bs_words(n_values);
    uint64_t *live_in  = calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *live_out = calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *gen      = calloc(n_blocks * bw, sizeof(uint64_t));  // uses upward-exposed
    uint64_t *kill     = calloc(n_blocks * bw, sizeof(uint64_t));  // defs in block
    uint64_t *scratch  = calloc(bw, sizeof(uint64_t));

    // Mark every value's def in its block's kill set, and every use
    // in gen — but only if not already killed in the same block.
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle b = blocks[bi];
        uint64_t *G = gen  + bi * bw;
        uint64_t *K = kill + bi * bw;

        // Block arg defs kill the predecessors' equivalents (we
        // treat block args as defined at block entry).
        size_t na = MLIR_GetBlockNumArgs(b);
        for (size_t i = 0; i < na; i++) {
            uint32_t vi; vmap_get(&vmap, MLIR_GetBlockArg(b, i), &vi);
            bs_set(K, vi);
        }

        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, i);

            // Uses of regular operands.
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
                uint32_t vi;
                if (vmap_get(&vmap, v, &vi)) {
                    if (!bs_test(K, vi)) bs_set(G, vi);
                }
            }
            // Uses of successor operands (block-arg passers).
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nsop = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nsop; k++) {
                    MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(op, s, k);
                    uint32_t vi;
                    if (vmap_get(&vmap, v, &vi)) {
                        if (!bs_test(K, vi)) bs_set(G, vi);
                    }
                }
            }
            // Defs.
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t k = 0; k < nr; k++) {
                uint32_t vi; vmap_get(&vmap, MLIR_GetOpResult(op, k), &vi);
                bs_set(K, vi);
            }
        }
    }

    // Iterate to fixpoint. RPO is overkill; iterate until stable.
    bool changed = true;
    int iters = 0;
    while (changed && iters++ < 1024) {
        changed = false;
        // Walk blocks in reverse for faster convergence on natural loops.
        for (size_t bi_rev = 0; bi_rev < n_blocks; bi_rev++) {
            size_t bi = n_blocks - 1 - bi_rev;
            MLIR_BlockHandle b = blocks[bi];

            // live_out(B) = ∪ live_in(succ)
            uint64_t *LO = live_out + bi * bw;
            bs_zero(LO, n_values);
            // Successors are recorded on the block's terminator —
            // which is the last op of the block (any op may have
            // successors, but in practice only the terminator does).
            size_t no = MLIR_GetBlockNumOps(b);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(b, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    MLIR_BlockHandle succ = MLIR_GetOpSuccessor(op, s);
                    uint32_t si; if (!bmap_get(&bmap, succ, &si)) continue;
                    bs_or(LO, LO, live_in + si * bw, n_values);
                }
            }

            // live_in(B) = gen(B) ∪ (live_out(B) \ kill(B))
            uint64_t *LI = live_in + bi * bw;
            bs_diff(scratch, LO, kill + bi * bw, n_values);
            bs_or  (scratch, scratch, gen + bi * bw, n_values);
            if (!bs_equal(scratch, LI, n_values)) {
                memcpy(LI, scratch, bw * sizeof(uint64_t));
                changed = true;
            }
        }
    }

    // --- 3. Compute per-value last_use_pos + crosses_call flag. ---
    // For every value `v` in live_out(B), v is used at least up to
    // block_last_pos[B]. For every operand use inside the block,
    // last_use_pos = max(last_use_pos, position-of-the-op).
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle b = blocks[bi];
        uint64_t *LO = live_out + bi * bw;
        uint64_t *LI = live_in  + bi * bw;
        for (uint32_t vi = 0; vi < n_values; vi++) {
            if (bs_test(LO, vi)) {
                if (block_last_pos[bi] > vinfo[vi].last_use_pos)
                    vinfo[vi].last_use_pos = block_last_pos[bi];
            }
            // A value live-in to this block occupies its register from the
            // block's first position onward; lower the interval start so it
            // covers uses that precede the value's def in linear order
            // (mem2reg loop-carried block args).
            if (bs_test(LI, vi)) {
                if (block_first_pos[bi] < vinfo[vi].first_pos)
                    vinfo[vi].first_pos = block_first_pos[bi];
            }
        }
        // Walk ops in-order to refine intra-block last uses.
        uint32_t op_pos = block_first_pos[bi];
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, i);
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                uint32_t vi;
                if (vmap_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) {
                    if (op_pos > vinfo[vi].last_use_pos)
                        vinfo[vi].last_use_pos = op_pos;
                    if (op_pos < vinfo[vi].first_pos)
                        vinfo[vi].first_pos = op_pos;
                }
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                MLIR_BlockHandle succ = MLIR_GetOpSuccessor(op, s);
                size_t nsop = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nsop; k++) {
                    uint32_t vi;
                    if (vmap_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi)) {
                        if (op_pos > vinfo[vi].last_use_pos)
                            vinfo[vi].last_use_pos = op_pos;
                        if (op_pos < vinfo[vi].first_pos)
                            vinfo[vi].first_pos = op_pos;
                    }
                    // The branch STORES the k-th successor operand into the
                    // successor block's k-th block-arg home. That write
                    // happens at THIS branch position, which for an entry
                    // (non-back) edge precedes the block arg's own
                    // block-entry position. Extend the block arg's interval
                    // start to cover that write, so its spill slot is not
                    // recycled out from under the predecessor store.
                    {
                        MLIR_ValueHandle arg = MLIR_GetBlockArg(succ, k);
                        uint32_t avi;
                        if (vmap_get(&vmap, arg, &avi) &&
                            op_pos < vinfo[avi].first_pos)
                            vinfo[avi].first_pos = op_pos;
                    }
                }
            }
            op_pos++;
        }
    }

    // crosses_call: scan from def_pos to last_use_pos for any
    // wmir.call op. Build a flat "is_call[pos]" lookup.
    bool *is_call = calloc(pos + 1, sizeof(bool));
    {
        uint32_t op_pos = 0;
        for (size_t bi = 0; bi < n_blocks; bi++) {
            MLIR_BlockHandle b = blocks[bi];
            size_t no = MLIR_GetBlockNumOps(b);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(b, i);
                MLIR_OpType t = MLIR_GetOpType(op);
                if (t == OP_TYPE_WMIR_CALL) is_call[op_pos] = true;
                op_pos++;
            }
        }
    }
    for (uint32_t vi = 0; vi < n_values; vi++) {
        if (vinfo[vi].def_pos == UINT32_MAX) continue;
        if (vinfo[vi].last_use_pos < vinfo[vi].first_pos) continue;
        for (uint32_t p = vinfo[vi].first_pos; p <= vinfo[vi].last_use_pos && p < (uint32_t)pos; p++) {
            if (is_call[p]) { vinfo[vi].crosses_call = true; break; }
        }
    }
    free(is_call);

    // --- 4. Linear-scan allocation. ---
    // Pool of caller-saved integer registers we may freely use
    // without prologue/epilogue work. x9/x10/x11 stay reserved as
    // scratch in the lowering for spill fill/move and as a third
    // operand register needed by ops like wmir.srem (a - (a/b)*b),
    // wmir.store (address vs. value), and wmir.select (cond).
    // x18 is the "platform register" reserved by Darwin/iOS/macOS
    // (the kernel uses it); we must NOT allocate it.
    // x16/x17 are IP0/IP1 — the "intra-procedure-call scratch" regs
    // that linker veneers/stubs may freely clobber. Until our linker
    // is provably stub-free we exclude them too for safety.
    // x27/x28 hold the globals and linear-memory bases; x29/x30 are
    // fp/lr. The remaining callee-saved registers x19..x26 are added
    // to the pool: they survive calls (the AAPCS guarantees callees
    // preserve them) and are saved/restored once per function in the
    // prologue/epilogue. Caller-saved x12..x15 come first so they are
    // preferred and leaf/call-light functions reach for callee-saved
    // registers only under pressure.
    //
    // A value whose live range CROSSES A CALL (`crosses_call`) can only
    // be kept in a register that every possible callee is guaranteed to
    // preserve. External functions honour the AAPCS (x19..x28 saved),
    // but a handful of hand-written runtime helpers in the lowering
    // (the print/format/_start shims) clobber x19..x23 and x26 without
    // saving them. Only x24 and x25 are touched by no helper, so they
    // are the sole call-safe registers; call-crossing values may use
    // ONLY x24/x25 (else they fall back to a stack slot). They are
    // placed last in the pool so non-crossing values prefer the others
    // and leave x24/x25 available for the call-crossing case.
    //
    // x0..x8 are the AAPCS argument/result registers. They are caller-
    // saved (every call clobbers them), so a `crosses_call` value may not
    // live there — but the vast majority of values are short-lived and
    // call-free, so adding x0..x8 roughly doubles the usable pool and cuts
    // spill traffic dramatically. They are NOT scratch for ordinary op
    // lowering (which only uses x9/x10/x11 and the x16/x17 rebase pair),
    // and the call-arg / entry-param / call-result paths route through a
    // general parallel-move resolver, so an operand homed in an arg reg
    // never clobbers another live arg.
    //
    // Pool order: the 13 caller-saved registers (x0..x8, x12..x15) come
    // FIRST so non-crossing values prefer them; the 8 callee-saved / call-
    // safe registers (x19..x26) come LAST so they are reached only under
    // pressure, leaving them available for `crosses_call` values that can
    // live nowhere else. All of x19..x26 are call-safe: external callees
    // honour the AAPCS, and every hand-written runtime shim saves/restores
    // the callee-saved registers it touches (the widest, path_open, saves
    // x19..x23; none touches x24..x26). x27/x28 hold the globals/linmem
    // bases and x29/x30 are fp/lr, so none of those are in the pool.
    // Effective pool: x0..x8, x12..x15, then x19..x26 → 21 registers.
    //
    // To disable register allocation entirely (debugging aid that
    // forces every value to a stack slot), set WMIR_NO_REGALLOC=1
    // in the environment.
    static const uint8_t POOL[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8,
                                    12, 13, 14, 15,
                                    19, 20, 21, 22, 23, 24, 25, 26 };
    // Parallel marker: is POOL[i] safe to hold a value across a call?
    // The 13 caller-saved entries are NOT; the 8 callee-saved entries are.
    static const bool POOL_CALLSAFE[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1 };
    size_t POOL_N = sizeof(POOL) / sizeof(POOL[0]);
    if (getenv("WMIR_NO_REGALLOC")) POOL_N = 0;

    // Build the result table.
    WmirRegAlloc *ra = calloc(1, sizeof(*ra));
    ra->cap      = n_values;
    ra->values   = calloc(n_values, sizeof(*ra->values));
    ra->homes    = calloc(n_values, sizeof(*ra->homes));
    ra->n_values = n_values;

    // Order values by def_pos.
    uint32_t *order = calloc(n_values, sizeof(uint32_t));
    for (uint32_t i = 0; i < n_values; i++) order[i] = i;
    // Stable bottom-up merge sort by def_pos (ties keep original index
    // order, matching the previous stable selection sort). O(n log n) —
    // n_values can be large in self-host functions.
    {
        uint32_t *tmp = calloc(n_values, sizeof(uint32_t));
        for (size_t width = 1; width < n_values; width *= 2) {
            for (size_t lo = 0; lo < n_values; lo += 2 * width) {
                size_t mid = lo + width;       if (mid > n_values) mid = n_values;
                size_t hi  = lo + 2 * width;   if (hi  > n_values) hi  = n_values;
                size_t a = lo, b = mid, o = lo;
                while (a < mid && b < hi) {
                    if (vinfo[order[b]].first_pos < vinfo[order[a]].first_pos)
                        tmp[o++] = order[b++];
                    else
                        tmp[o++] = order[a++];
                }
                while (a < mid) tmp[o++] = order[a++];
                while (b < hi)  tmp[o++] = order[b++];
            }
            uint32_t *sw = order; order = tmp; tmp = sw;
        }
        free(tmp);
    }

    // Active list of intervals currently occupying a pool register.
    // We track the pool index (not just the reg number) so that the
    // pool can change without breaking reg_busy indexing.
    Active *active = calloc(POOL_N ? POOL_N : 1, sizeof(*active));
    size_t  n_active = 0;
    bool   *reg_busy = calloc(POOL_N ? POOL_N : 1, sizeof(bool));

    uint32_t next_slot = 0;
    const uint32_t MAX_SLOTS = 1u << 24;  // 128 MiB frame — way beyond
                                          // any reasonable function.

    // Spill-slot reuse. A slot occupied by a value whose last use is in
    // the past can be handed to a later spill: the two never overlap, so
    // sharing the slot is safe (same non-overlap invariant the register
    // active-list relies on). Without this, `next_slot` grows to the
    // TOTAL number of values ever spilled in the function rather than the
    // PEAK number live at once. Big frames push spill-slot byte offsets
    // past aarch64's scaled imm12 load/store range, forcing the lowering
    // onto the expensive rebase path (movz/movk + add to synthesise the
    // address — several extra instructions per spill access). Reuse keeps
    // the frame ≈ peak-simultaneous spills so most offsets stay in range.
    typedef struct { uint32_t slot; uint32_t end; } SlotLive;
    SlotLive *sactive   = malloc((n_values ? n_values : 1) * sizeof(SlotLive));
    size_t    n_sactive = 0;
    uint32_t *free_slots = malloc((n_values ? n_values : 1) * sizeof(uint32_t));
    size_t    n_free    = 0;
    bool      no_reuse  = getenv("WMIR_NO_SLOT_REUSE") != NULL;
    // Allocate a spill slot for a value whose last use is at `end_pos`,
    // preferring a retired slot from the free list. Registers it in the
    // active-slot list so it can be reclaimed once `end_pos` passes.
    // SAFE ONLY for values that start at the current position (their slot
    // live range is [now .. end]); reused slots are guaranteed free as of
    // `now`. Use TAKE_SLOT_FRESH for a value whose interval starts in the
    // past (an evicted value), which must not pull a slot reclaimed only
    // relative to the current position.
    #define TAKE_SLOT(DST, END) do {                                  \
            if (!no_reuse && n_free > 0) { (DST) = free_slots[--n_free]; } \
            else {                                                    \
                if (next_slot >= MAX_SLOTS) goto fail_slot_overflow;  \
                (DST) = next_slot++;                                  \
            }                                                         \
            sactive[n_sactive].slot = (DST);                          \
            sactive[n_sactive].end  = (END);                          \
            n_sactive++;                                              \
        } while (0)
    #define TAKE_SLOT_FRESH(DST, END) do {                            \
            if (next_slot >= MAX_SLOTS) goto fail_slot_overflow;      \
            (DST) = next_slot++;                                      \
            sactive[n_sactive].slot = (DST);                          \
            sactive[n_sactive].end  = (END);                          \
            n_sactive++;                                              \
        } while (0)

    for (uint32_t k = 0; k < n_values; k++) {
        uint32_t vi = order[k];
        VInfo *vp = &vinfo[vi];
        ra->values[vi] = vp->handle;

        // Skip "dead" values (no defining op location). Shouldn't
        // happen on a well-formed IR; assign a slot defensively.
        if (vp->def_pos == UINT32_MAX) {
            ra->homes[vi].kind = HOME_SLOT;
            if (next_slot >= MAX_SLOTS) goto fail_slot_overflow;
            ra->homes[vi].idx  = next_slot;
            next_slot++;
            continue;
        }

        // Retire active intervals that end before this interval starts.
        for (size_t a = 0; a < n_active; ) {
            uint32_t avi = active[a].vi;
            if (vinfo[avi].last_use_pos < vp->first_pos) {
                reg_busy[active[a].pool_idx] = false;
                active[a] = active[--n_active];
            } else {
                a++;
            }
        }

        // Reclaim spill slots whose occupant's last use is now in the
        // past, returning them to the free list for reuse (same invariant
        // as the register retire above).
        for (size_t s = 0; s < n_sactive; ) {
            if (sactive[s].end < vp->first_pos) {
                free_slots[n_free++] = sactive[s].slot;
                sactive[s] = sactive[--n_sactive];
            } else {
                s++;
            }
        }

        // Rematerializable constants never occupy a register or slot.
        if (vp->is_const) {
            ra->homes[vi].kind   = HOME_CONST;
            ra->homes[vi].is_i64 = vp->c_is_i64 ? 1 : 0;
            ra->homes[vi].cval   = vp->cval;
            continue;
        }

        // Pin to slot if FP or never used. Call-crossing values are NOT
        // pinned here — they may use the call-safe registers (x24/x25);
        // `callee_only` below restricts them to that subset.
        bool force_slot =
            !vp->is_int ||
            vp->last_use_pos <= vp->first_pos;  // dead-on-def: still needs a slot for correctness

        if (force_slot) {
            ra->homes[vi].kind = HOME_SLOT;
            TAKE_SLOT(ra->homes[vi].idx, vp->last_use_pos);
            continue;
        }

        // A value live across a call may only occupy a call-safe
        // register (x24/x25). A non-crossing value may use any pool reg.
        bool callee_only = vp->crosses_call;

        // Find a free, eligible pool reg.
        int found = -1;
        for (size_t i = 0; i < POOL_N; i++) {
            if (reg_busy[i]) continue;
            if (callee_only && !POOL_CALLSAFE[i]) continue;
            found = (int)i; break;
        }
        if (found >= 0) {
            uint8_t reg = POOL[found];
            reg_busy[found] = true;
            active[n_active++] = (Active){ vi, reg, (uint8_t)found };
            ra->homes[vi].kind = HOME_REG;
            ra->homes[vi].idx  = reg;
            continue;
        }

        // No free eligible reg. Among the active intervals occupying an
        // ELIGIBLE register, evict the one with the latest end — unless
        // the current value ends even later (then spill it to a slot).
        size_t   evict_a   = (size_t)-1;
        uint32_t evict_end = 0;
        for (size_t a = 0; a < n_active; a++) {
            if (callee_only && !POOL_CALLSAFE[active[a].pool_idx]) continue;
            uint32_t e = vinfo[active[a].vi].last_use_pos;
            if (evict_a == (size_t)-1 || e > evict_end) { evict_end = e; evict_a = a; }
        }
        if (evict_a != (size_t)-1 && evict_end > vp->last_use_pos) {
            // Spill the evicted one; take its register.
            uint32_t evict_vi  = active[evict_a].vi;
            uint8_t  reg       = active[evict_a].reg;
            uint8_t  pool_idx  = active[evict_a].pool_idx;
            ra->homes[evict_vi].kind = HOME_SLOT;
            TAKE_SLOT_FRESH(ra->homes[evict_vi].idx, vinfo[evict_vi].last_use_pos);
            active[evict_a] = (Active){ vi, reg, pool_idx };
            ra->homes[vi].kind = HOME_REG;
            ra->homes[vi].idx  = reg;
        } else {
            // Spill the current value.
            ra->homes[vi].kind = HOME_SLOT;
            TAKE_SLOT(ra->homes[vi].idx, vp->last_use_pos);
        }
    }

    ra->n_slots = next_slot;
    #undef TAKE_SLOT
    #undef TAKE_SLOT_FRESH

    if (getenv("WMIR_CHECK_SLOTS")) {
        // O(n^2) debug check: no two values sharing a spill slot may have
        // overlapping live intervals [first_pos, last_use_pos].
        for (uint32_t i = 0; i < n_values; i++) {
            if (ra->homes[i].kind != HOME_SLOT) continue;
            if (vinfo[i].first_pos == UINT32_MAX) continue;
            for (uint32_t j = i + 1; j < n_values; j++) {
                if (ra->homes[j].kind != HOME_SLOT) continue;
                if (ra->homes[j].idx != ra->homes[i].idx) continue;
                if (vinfo[j].first_pos == UINT32_MAX) continue;
                uint32_t af = vinfo[i].first_pos, al = vinfo[i].last_use_pos;
                uint32_t bf = vinfo[j].first_pos, bl = vinfo[j].last_use_pos;
                if (af <= bl && bf <= al) {
                    fprintf(stderr,
                        "WMIR_CHECK_SLOTS: slot %u shared by overlapping "
                        "values: [%u,%u] and [%u,%u]\n",
                        ra->homes[i].idx, af, al, bf, bl);
                }
            }
        }
    }

    // Record which callee-saved registers (x19..x26) were actually
    // assigned, so the lowering can save/restore exactly those.
    ra->used_callee_mask = 0;
    for (size_t i = 0; i < ra->n_values; i++) {
        if (ra->homes[i].kind == HOME_REG) {
            uint32_t reg = ra->homes[i].idx;
            if (reg >= 19 && reg <= 26)
                ra->used_callee_mask |= 1u << (reg - 19);
        }
    }

    // Build the O(1) lookup index over the now-populated values array.
    {
        size_t hc = 16;
        while (hc < ra->n_values * 2) hc *= 2;
        ra->hcap  = hc;
        ra->hkeys = calloc(hc, sizeof(*ra->hkeys));
        ra->hidx  = malloc(hc * sizeof(*ra->hidx));
        size_t mask = hc - 1;
        for (size_t i = 0; i < ra->n_values; i++) {
            MLIR_ValueHandle v = ra->values[i];
            size_t p = (size_t)vmap_hash(v) & mask;
            while (ra->hkeys[p] != 0) p = (p + 1) & mask;
            ra->hkeys[p] = v; ra->hidx[p] = (uint32_t)i;
        }
    }

    free(order); free(active); free(reg_busy);
    free(sactive); free(free_slots);
    free(live_in); free(live_out); free(gen); free(kill); free(scratch);
    free(block_first_pos); free(block_last_pos); free(blocks);
    free(vinfo);
    bmap_free(&bmap); vmap_free(&vmap);
    return ra;

fail_slot_overflow:
    fprintf(stderr, "wmir_regalloc: more than %u slots — frame too large\n",
        (unsigned)MAX_SLOTS);
    free(order); free(active); free(reg_busy);
    free(sactive); free(free_slots);
    free(live_in); free(live_out); free(gen); free(kill); free(scratch);
    free(block_first_pos); free(block_last_pos); free(blocks);
    free(vinfo);
    bmap_free(&bmap); vmap_free(&vmap);
    wmir_regalloc_free(ra);
    return NULL;
}

bool wmir_regalloc_lookup(const WmirRegAlloc *ra,
                          MLIR_ValueHandle v,
                          ValueHome *out) {
    if (ra->hcap == 0) return false;
    size_t mask = ra->hcap - 1;
    size_t p = (size_t)vmap_hash(v) & mask;
    while (ra->hkeys[p] != 0) {
        if (ra->hkeys[p] == v) { *out = ra->homes[ra->hidx[p]]; return true; }
        p = (p + 1) & mask;
    }
    return false;
}

void wmir_regalloc_free(WmirRegAlloc *ra) {
    if (!ra) return;
    free(ra->values);
    free(ra->homes);
    free(ra->hkeys);
    free(ra->hidx);
    free(ra);
}

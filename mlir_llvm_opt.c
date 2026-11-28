// LLVM-dialect optimization passes for -O1/-O2 (LICM).
// Plain C99 (no VLAs, no function pointers) so tinyC self-host compiles it.

#include "mlir_llvm_opt.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <base/buddy.h>

// ---------------------------------------------------------------------------
// Small open-addressed set of MLIR_ValueHandle (0 = empty, ~0 = tombstone).
// ---------------------------------------------------------------------------
typedef struct {
    uintptr_t *k;
    size_t cap, n;
} OptVSet;

static void vset_init(OptVSet *s, size_t want) {
    size_t cap = 32;
    while (cap < want * 2) cap *= 2;
    s->k = (uintptr_t *)calloc(cap, sizeof(uintptr_t));
    s->cap = cap;
    s->n = 0;
}

static void vset_free(OptVSet *s) {
    free(s->k);
    s->k = NULL;
    s->cap = 0;
    s->n = 0;
}

static void vset_grow(OptVSet *s) {
    size_t ncap = s->cap * 2;
    uintptr_t *nk = (uintptr_t *)calloc(ncap, sizeof(uintptr_t));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < s->cap; i++) {
        uintptr_t v = s->k[i];
        if (!v || v == ~(uintptr_t)0) continue;
        size_t j = (v * 1099511628211ull) & mask;
        while (nk[j]) j = (j + 1) & mask;
        nk[j] = v;
    }
    free(s->k);
    s->k = nk;
    s->cap = ncap;
}

static void vset_add(OptVSet *s, MLIR_ValueHandle vh) {
    uintptr_t v = (uintptr_t)vh;
    if (!v) return;
    if ((s->n + 1) * 2 >= s->cap) vset_grow(s);
    size_t mask = s->cap - 1;
    size_t j = (v * 1099511628211ull) & mask;
    size_t first_tomb = ~(size_t)0;
    while (s->k[j]) {
        if (s->k[j] == v) return;
        if (s->k[j] == ~(uintptr_t)0 && first_tomb == ~(size_t)0) first_tomb = j;
        j = (j + 1) & mask;
    }
    if (first_tomb != ~(size_t)0) j = first_tomb;
    s->k[j] = v;
    s->n++;
}

static bool vset_has(const OptVSet *s, MLIR_ValueHandle vh) {
    uintptr_t v = (uintptr_t)vh;
    if (!v || s->cap == 0) return false;
    size_t mask = s->cap - 1;
    size_t j = (v * 1099511628211ull) & mask;
    while (s->k[j]) {
        if (s->k[j] == v) return true;
        j = (j + 1) & mask;
    }
    return false;
}

static void vset_del(OptVSet *s, MLIR_ValueHandle vh) {
    uintptr_t v = (uintptr_t)vh;
    if (!v || s->cap == 0) return;
    size_t mask = s->cap - 1;
    size_t j = (v * 1099511628211ull) & mask;
    while (s->k[j]) {
        if (s->k[j] == v) { s->k[j] = ~(uintptr_t)0; return; }
        j = (j + 1) & mask;
    }
}

// Pure, non-trapping, single-result LLVM ops that are always safe to hoist
// into a dominating preheader even if the loop executes 0 times.
// Excludes rematerialized constants/addressof/zero/null/undef (x64 remats them
// for free, so hoisting them would only increase register pressure).
static bool licm_is_hoistable(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (MLIR_GetOpNumResults(ctx, op) != 1) return false;
    switch (MLIR_GetOpType(ctx, op)) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR:  case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_TRUNC:
    case OP_TYPE_LLVM_INTTOPTR: case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_BITCAST:
    case OP_TYPE_LLVM_GEP:
    case OP_TYPE_LLVM_ICMP: case OP_TYPE_LLVM_SELECT:
    case OP_TYPE_LLVM_FADD: case OP_TYPE_LLVM_FSUB: case OP_TYPE_LLVM_FMUL:
    case OP_TYPE_LLVM_FNEG: case OP_TYPE_LLVM_FPEXT: case OP_TYPE_LLVM_FPTRUNC:
    case OP_TYPE_LLVM_SITOFP: case OP_TYPE_LLVM_UITOFP: case OP_TYPE_LLVM_FCMP:
        return true;
    default:
        return false;
    }
}

static bool licm_is_remat_const(MLIR_Context *ctx, MLIR_OpHandle op) {
    switch (MLIR_GetOpType(ctx, op)) {
    case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ADDRESSOF:
    case OP_TYPE_LLVM_MLIR_ZERO:
    case OP_TYPE_LLVM_MLIR_NULL:
    case OP_TYPE_LLVM_MLIR_UNDEF:
        return true;
    default:
        return false;
    }
}

static void licm_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 2 || nb > 1024) return;

    MLIR_BlockHandle *blks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    for (size_t b = 0; b < nb; b++) blks[b] = MLIR_GetRegionBlock(ctx, reg, b);

    // 1. Build predecessor lists (CSR format).
    size_t *pred_cnt = (size_t *)calloc(nb, sizeof(size_t));
    size_t total_edges = 0;
    for (size_t b = 0; b < nb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
        if (no == 0) continue;
        MLIR_OpHandle t = MLIR_GetBlockOp(ctx, blks[b], no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
        for (size_t s = 0; s < ns; s++) {
            size_t ti = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, t, s));
            if (ti < nb) { pred_cnt[ti]++; total_edges++; }
        }
    }
    size_t *pred_off = (size_t *)malloc((nb + 1) * sizeof(size_t));
    pred_off[0] = 0;
    for (size_t b = 0; b < nb; b++) pred_off[b + 1] = pred_off[b] + pred_cnt[b];
    size_t *preds = (size_t *)malloc((total_edges ? total_edges : 1) * sizeof(size_t));
    memset(pred_cnt, 0, nb * sizeof(size_t));
    for (size_t b = 0; b < nb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
        if (no == 0) continue;
        MLIR_OpHandle t = MLIR_GetBlockOp(ctx, blks[b], no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
        for (size_t s = 0; s < ns; s++) {
            size_t ti = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, t, s));
            if (ti < nb) preds[pred_off[ti] + pred_cnt[ti]++] = b;
        }
    }

    // 2. Compute dominator bitsets.
    size_t words = (nb + 63) / 64;
    uint64_t *dom = (uint64_t *)malloc(nb * words * sizeof(uint64_t));
    uint64_t *tmp = (uint64_t *)malloc(words * sizeof(uint64_t));
    // Entry block 0: dom[0] = {0}; all others: all 1s.
    memset(dom, 0, words * sizeof(uint64_t));
    dom[0] = 1ull;
    for (size_t b = 1; b < nb; b++) {
        for (size_t w = 0; w < words; w++) dom[b * words + w] = ~0ull;
    }
    bool changed = true;
    int iter = 0;
    while (changed && iter < 64) {
        changed = false;
        iter++;
        for (size_t b = 1; b < nb; b++) {
            size_t p0 = pred_off[b], p1 = pred_off[b + 1];
            if (p0 == p1) continue; // unreachable
            for (size_t w = 0; w < words; w++) tmp[w] = ~0ull;
            for (size_t pi = p0; pi < p1; pi++) {
                size_t p = preds[pi];
                for (size_t w = 0; w < words; w++) tmp[w] &= dom[p * words + w];
            }
            tmp[b >> 6] |= (1ull << (b & 63));
            for (size_t w = 0; w < words; w++) {
                if (dom[b * words + w] != tmp[w]) {
                    dom[b * words + w] = tmp[w];
                    changed = true;
                }
            }
        }
    }

    // 3. Process natural loops (innermost-first by iterating headers in reverse).
    uint8_t *in_loop_blk = (uint8_t *)malloc(nb);
    size_t *stack = (size_t *)malloc(nb * sizeof(size_t));

    for (size_t h_rev = nb; h_rev > 0; h_rev--) {
        size_t hdr = h_rev - 1;
        // Check if hdr has any back-edge (pred p dominated by hdr).
        bool has_backedge = false;
        for (size_t pi = pred_off[hdr]; pi < pred_off[hdr + 1]; pi++) {
            size_t p = preds[pi];
            if (dom[p * words + (hdr >> 6)] & (1ull << (hdr & 63))) {
                has_backedge = true;
                break;
            }
        }
        if (!has_backedge) continue;

        // Collect all blocks in the natural loop of `hdr`.
        memset(in_loop_blk, 0, nb);
        in_loop_blk[hdr] = 1;
        size_t sp = 0;
        for (size_t pi = pred_off[hdr]; pi < pred_off[hdr + 1]; pi++) {
            size_t p = preds[pi];
            if ((dom[p * words + (hdr >> 6)] & (1ull << (hdr & 63))) && !in_loop_blk[p]) {
                in_loop_blk[p] = 1;
                stack[sp++] = p;
            }
        }
        while (sp > 0) {
            size_t cur = stack[--sp];
            for (size_t pi = pred_off[cur]; pi < pred_off[cur + 1]; pi++) {
                size_t p = preds[pi];
                if (!in_loop_blk[p]) {
                    in_loop_blk[p] = 1;
                    stack[sp++] = p;
                }
            }
        }

        // Find unique outside predecessor `pre` with unconditional llvm.br -> hdr.
        size_t pre = ~(size_t)0;
        bool pre_ok = true;
        for (size_t pi = pred_off[hdr]; pi < pred_off[hdr + 1]; pi++) {
            size_t p = preds[pi];
            if (in_loop_blk[p]) continue;
            if (pre == ~(size_t)0) pre = p;
            else if (pre != p) { pre_ok = false; break; }
        }
        if (!pre_ok || pre == ~(size_t)0) continue;
        size_t pre_no = MLIR_GetBlockNumOps(ctx, blks[pre]);
        if (pre_no == 0) continue;
        MLIR_OpHandle pre_term = MLIR_GetBlockOp(ctx, blks[pre], pre_no - 1);
        if (MLIR_GetOpType(ctx, pre_term) != OP_TYPE_LLVM_BR) continue;

        // Collect all values defined inside the loop.
        OptVSet in_loop;
        vset_init(&in_loop, 64);
        for (size_t b = 0; b < nb; b++) {
            if (!in_loop_blk[b]) continue;
            size_t na = MLIR_GetBlockNumArgs(ctx, blks[b]);
            for (size_t a = 0; a < na; a++)
                vset_add(&in_loop, MLIR_GetBlockArg(ctx, blks[b], a));
            size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blks[b], i);
                if (MLIR_GetOpNumResults(ctx, op) == 1 && !licm_is_remat_const(ctx, op))
                    vset_add(&in_loop, MLIR_GetOpResult(ctx, op, 0));
            }
        }

        // Iteratively hoist invariant ops (and any remat consts they directly depend on).
        for (int round = 0; round < 4; round++) {
            bool moved_any = false;
            for (size_t b = 0; b < nb; b++) {
                if (!in_loop_blk[b]) continue;
                size_t i = 0;
                while (i + 1 < MLIR_GetBlockNumOps(ctx, blks[b])) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blks[b], i);
                    if (!licm_is_hoistable(ctx, op)) { i++; continue; }
                    size_t nop = MLIR_GetOpNumOperands(ctx, op);
                    bool inv = true;
                    for (size_t k = 0; k < nop; k++) {
                        if (vset_has(&in_loop, MLIR_GetOpOperand(ctx, op, k))) {
                            inv = false;
                            break;
                        }
                    }
                    if (!inv) { i++; continue; }

                    // Also ensure any remat-const operand defined in a loop block
                    // is moved to `pre` before `op` so dominance is preserved!
                    for (size_t k = 0; k < nop; k++) {
                        MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, op, k);
                        MLIR_OpHandle odef = MLIR_GetValueDefiningOp(ctx, ov);
                        if (odef != MLIR_INVALID_HANDLE && licm_is_remat_const(ctx, odef)) {
                            MLIR_BlockHandle ob = MLIR_GetOpParentBlock(ctx, odef);
                            if (ob != MLIR_INVALID_HANDLE) {
                                size_t obi = MLIR_GetBlockIndex(ctx, ob);
                                if (obi < nb && in_loop_blk[obi]) {
                                    MLIR_DetachOp(ctx, odef);
                                    size_t ins_pos = MLIR_GetBlockNumOps(ctx, blks[pre]) - 1;
                                    MLIR_InsertBlockOpAtIndex(ctx, blks[pre], odef, ins_pos);
                                    if (obi == b && i > 0) i--;
                                }
                            }
                        }
                    }

                    MLIR_DetachOp(ctx, op);
                    size_t ins_pos = MLIR_GetBlockNumOps(ctx, blks[pre]) - 1;
                    MLIR_InsertBlockOpAtIndex(ctx, blks[pre], op, ins_pos);
                    vset_del(&in_loop, MLIR_GetOpResult(ctx, op, 0));
                    moved_any = true;
                    // Do not increment `i`: next op shifted into slot `i`.
                }
            }
            if (!moved_any) break;
        }
        vset_free(&in_loop);
    }

    free(stack);
    free(in_loop_blk);
    free(tmp);
    free(dom);
    free(preds);
    free(pred_off);
    free(pred_cnt);
    free(blks);
}

void mlir_llvm_licm(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_LICM")) return;
    if (MLIR_GetOpNumRegions(ctx, module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        MLIR_OpType ft = MLIR_GetOpType(ctx, fn);
        if (ft != OP_TYPE_FUNC_FUNC && ft != OP_TYPE_LLVM_FUNC) continue;
        licm_func(ctx, fn);
    }
}

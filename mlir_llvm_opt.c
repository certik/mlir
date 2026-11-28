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
        // Do NOT hoist constant-offset GEPs: on x86-64 `[base + disp]` folds
        // for free into load/store addressing modes, whereas hoisting creates a
        // long-lived pointer that increases register pressure and spills.
        return MLIR_GetOpNumOperands(ctx, op) > 1;
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

    // Block handle -> current 0..nb-1 index (MLIR_GetBlockIndex can be stale after MLIR_EraseBlock).
    size_t bhcap = 16;
    while (bhcap < nb * 2) bhcap *= 2;
    uintptr_t *bhk = (uintptr_t *)calloc(bhcap, sizeof(uintptr_t));
    size_t    *bhv = (size_t    *)malloc(bhcap * sizeof(size_t));
    size_t bhmask = bhcap - 1;
    for (size_t b = 0; b < nb; b++) {
        uintptr_t k = (uintptr_t)blks[b];
        size_t s = (k * 1099511628211ull) & bhmask;
        while (bhk[s]) s = (s + 1) & bhmask;
        bhk[s] = k; bhv[s] = b;
    }

    // 1. Build predecessor lists (CSR format).
    size_t *pred_cnt = (size_t *)calloc(nb, sizeof(size_t));
    size_t total_edges = 0;
    for (size_t b = 0; b < nb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
        if (no == 0) continue;
        MLIR_OpHandle t = MLIR_GetBlockOp(ctx, blks[b], no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
        for (size_t s = 0; s < ns; s++) {
            uintptr_t tk = (uintptr_t)MLIR_GetOpSuccessor(ctx, t, s);
            size_t slot = (tk * 1099511628211ull) & bhmask;
            while (bhk[slot]) {
                if (bhk[slot] == tk) { pred_cnt[bhv[slot]]++; total_edges++; break; }
                slot = (slot + 1) & bhmask;
            }
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
            uintptr_t tk = (uintptr_t)MLIR_GetOpSuccessor(ctx, t, s);
            size_t slot = (tk * 1099511628211ull) & bhmask;
            while (bhk[slot]) {
                if (bhk[slot] == tk) {
                    size_t ti = bhv[slot];
                    preds[pred_off[ti] + pred_cnt[ti]++] = b;
                    break;
                }
                slot = (slot + 1) & bhmask;
            }
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
                    bool is_hoist = licm_is_hoistable(ctx, op) ||
                                    (nb <= 64 && MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_MLIR_ADDRESSOF);
                    if (!is_hoist) { i++; continue; }
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
                                uintptr_t obk = (uintptr_t)ob;
                                size_t oslot = (obk * 1099511628211ull) & bhmask;
                                size_t obi = nb;
                                while (bhk[oslot]) {
                                    if (bhk[oslot] == obk) { obi = bhv[oslot]; break; }
                                    oslot = (oslot + 1) & bhmask;
                                }
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
        // Loop-invariant memory accumulator scalar promotion (e.g. `C[i*N+j] = 0; for (k) C[i*N+j] += A[...]*B[...];`):
        // Promotes read-modify-write accumulators in simple 2-block inner loops into a
        // register-resident SSA block arg, eliminating the 5-cycle store-forwarding chain.
        {
            size_t loop_cnt = 0, body = nb;
            for (size_t b = 0; b < nb; b++) {
                if (!in_loop_blk[b]) continue;
                loop_cnt++;
                if (b != hdr) body = b;
            }
            if (loop_cnt == 2 && body < nb) {
                size_t hdr_no = MLIR_GetBlockNumOps(ctx, blks[hdr]);
                size_t body_no = MLIR_GetBlockNumOps(ctx, blks[body]);
                MLIR_OpHandle hdr_term = hdr_no ? MLIR_GetBlockOp(ctx, blks[hdr], hdr_no - 1) : MLIR_INVALID_HANDLE;
                MLIR_OpHandle body_term = body_no ? MLIR_GetBlockOp(ctx, blks[body], body_no - 1) : MLIR_INVALID_HANDLE;
                if (hdr_term != MLIR_INVALID_HANDLE && body_term != MLIR_INVALID_HANDLE &&
                    MLIR_GetOpType(ctx, hdr_term) == OP_TYPE_LLVM_COND_BR &&
                    MLIR_GetOpType(ctx, body_term) == OP_TYPE_LLVM_BR &&
                    MLIR_GetOpSuccessor(ctx, body_term, 0) == blks[hdr] &&
                    MLIR_GetOpSuccessor(ctx, hdr_term, 0) == blks[body]) {
                    MLIR_BlockHandle exit_bh = MLIR_GetOpSuccessor(ctx, hdr_term, 1);
                    size_t exit_b = nb;
                    {
                        uintptr_t ek = (uintptr_t)exit_bh;
                        size_t es = (ek * 1099511628211ull) & bhmask;
                        while (bhk[es]) { if (bhk[es] == ek) { exit_b = bhv[es]; break; } es = (es + 1) & bhmask; }
                    }
                    // Require unique exit dominated by hdr (pred_cnt[exit_b] == 1).
                    if (exit_b < nb && !in_loop_blk[exit_b] &&
                        pred_off[exit_b + 1] - pred_off[exit_b] == 1 &&
                        preds[pred_off[exit_b]] == hdr) {
                        bool ok = true;
                        // Header must have no calls or stores.
                        for (size_t i = 0; i < hdr_no; i++) {
                            MLIR_OpType ot = MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blks[hdr], i));
                            if (ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL ||
                                ot == OP_TYPE_LLVM_STORE || ot == OP_TYPE_LLVM_LOAD) { ok = false; break; }
                        }
                        // Find the single store in `body` to a loop-invariant pointer.
                        MLIR_OpHandle st_op = MLIR_INVALID_HANDLE;
                        size_t st_idx = 0;
                        for (size_t i = 0; i < body_no && ok; i++) {
                            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blks[body], i);
                            MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                            if (ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL) { ok = false; break; }
                            if (ot == OP_TYPE_LLVM_STORE) {
                                if (st_op != MLIR_INVALID_HANDLE) { ok = false; break; }
                                st_op = op; st_idx = i;
                            }
                        }
                        if (ok && st_op != MLIR_INVALID_HANDLE) {
                            MLIR_ValueHandle st_val = MLIR_GetOpOperand(ctx, st_op, 0);
                            MLIR_ValueHandle acc_ptr = MLIR_GetOpOperand(ctx, st_op, 1);
                            MLIR_TypeHandle acc_ty = MLIR_GetValueType(ctx, st_val);
                            MLIR_TypeKind ak = MLIR_GetTypeKind(ctx, acc_ty);
                            bool acc_scalar = (ak == MLIR_TYPE_KIND_INTEGER && MLIR_GetTypeWidth(ctx, acc_ty) >= 16);
                            if (acc_scalar && !vset_has(&in_loop, acc_ptr)) {
                                // Check all loads in `body`: either matching `load(acc_ptr)` before `st_op`,
                                // or a non-byte load of a strictly different type (C TBAA non-aliasing).
                                MLIR_OpHandle acc_loads[8];
                                size_t n_acc_loads = 0;
                                for (size_t i = 0; i < body_no && ok; i++) {
                                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blks[body], i);
                                    if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_LOAD) continue;
                                    MLIR_ValueHandle lp = MLIR_GetOpOperand(ctx, op, 0);
                                    MLIR_TypeHandle lt = MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, op, 0));
                                    if (lp == acc_ptr) {
                                        if (lt != acc_ty || i >= st_idx || n_acc_loads >= 8) { ok = false; break; }
                                        acc_loads[n_acc_loads++] = op;
                                    } else {
                                        // Must be a different non-byte type (char* may alias anything).
                                        if (lt == acc_ty) { ok = false; break; }
                                        if (MLIR_GetTypeKind(ctx, lt) == MLIR_TYPE_KIND_INTEGER &&
                                            MLIR_GetTypeWidth(ctx, lt) <= 8) { ok = false; break; }
                                    }
                                }
                                // Helper: check if two pointers are the same SSA value or identical GEPs.
                                // Require a dominating store to `acc_ptr` in `pre` (with no calls/other stores after it)
                                // so `acc_ptr` is provably valid/writable even if the loop executes 0 times.
                                MLIR_OpHandle pre_st = MLIR_INVALID_HANDLE;
                                if (ok && n_acc_loads > 0) {
                                    MLIR_OpHandle g_acc = MLIR_GetValueDefiningOp(ctx, acc_ptr);
                                    size_t pno = MLIR_GetBlockNumOps(ctx, blks[pre]);
                                    for (size_t pi = pno; pi-- > 0; ) {
                                        MLIR_OpHandle pop = MLIR_GetBlockOp(ctx, blks[pre], pi);
                                        MLIR_OpType pot = MLIR_GetOpType(ctx, pop);
                                        if (pot == OP_TYPE_LLVM_STORE) {
                                            MLIR_ValueHandle pp = MLIR_GetOpOperand(ctx, pop, 1);
                                            bool same_p = (pp == acc_ptr);
                                            if (!same_p && g_acc != MLIR_INVALID_HANDLE &&
                                                MLIR_GetOpType(ctx, g_acc) == OP_TYPE_LLVM_GEP) {
                                                MLIR_OpHandle g_pp = MLIR_GetValueDefiningOp(ctx, pp);
                                                if (g_pp != MLIR_INVALID_HANDLE &&
                                                    MLIR_GetOpType(ctx, g_pp) == OP_TYPE_LLVM_GEP &&
                                                    MLIR_GetOpNumOperands(ctx, g_pp) == MLIR_GetOpNumOperands(ctx, g_acc)) {
                                                    same_p = true;
                                                    for (size_t k = 0; k < MLIR_GetOpNumOperands(ctx, g_acc); k++) {
                                                        if (MLIR_GetOpOperand(ctx, g_pp, k) != MLIR_GetOpOperand(ctx, g_acc, k)) {
                                                            same_p = false; break;
                                                        }
                                                    }
                                                    if (same_p) {
                                                        MLIR_AttributeHandle e1 = MLIR_GetOpAttr(ctx, g_pp, MLIR_KEY_ELEM_TYPE);
                                                        MLIR_AttributeHandle e2 = MLIR_GetOpAttr(ctx, g_acc, MLIR_KEY_ELEM_TYPE);
                                                        if (e1 == MLIR_INVALID_HANDLE || e2 == MLIR_INVALID_HANDLE ||
                                                            MLIR_GetAttributeTypeValue(ctx, e1) != MLIR_GetAttributeTypeValue(ctx, e2))
                                                            same_p = false;
                                                    }
                                                    if (same_p) {
                                                        MLIR_AttributeHandle r1 = MLIR_GetOpAttr(ctx, g_pp, MLIR_KEY_RAW_CONSTANT_INDICES);
                                                        MLIR_AttributeHandle r2 = MLIR_GetOpAttr(ctx, g_acc, MLIR_KEY_RAW_CONSTANT_INDICES);
                                                        if (r1 != MLIR_INVALID_HANDLE && r2 != MLIR_INVALID_HANDLE) {
                                                            int32_t c1[16], c2[16];
                                                            size_t n1 = MLIR_GetAttributeDenseI32(ctx, r1, c1, 16);
                                                            size_t n2 = MLIR_GetAttributeDenseI32(ctx, r2, c2, 16);
                                                            if (n1 == 0 || n1 != n2) same_p = false;
                                                            else {
                                                                for (size_t k = 0; k < n1; k++)
                                                                    if (c1[k] != c2[k]) { same_p = false; break; }
                                                            }
                                                        } else if (r1 != r2) {
                                                            same_p = false;
                                                        }
                                                    }
                                                }
                                            }
                                            if (same_p &&
                                                MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, pop, 0)) == acc_ty) {
                                                pre_st = pop;
                                            }
                                            break;
                                        }
                                        if (pot == OP_TYPE_LLVM_CALL || pot == OP_TYPE_FUNC_CALL) break;
                                    }
                                }
                                if (ok && pre_st != MLIR_INVALID_HANDLE) {
                                    MLIR_ValueHandle init_val = MLIR_GetOpOperand(ctx, pre_st, 0);
                                    MLIR_ValueHandle acc_phi = MLIR_AddBlockArgument(ctx, blks[hdr], acc_ty, MLIR_INVALID_HANDLE);
                                    // Append init_val to pre_term -> hdr
                                    {
                                        size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, pre_term, 0);
                                        MLIR_ValueHandle *vals = (MLIR_ValueHandle *)malloc((nso + 1) * sizeof(MLIR_ValueHandle));
                                        for (size_t k = 0; k < nso; k++) vals[k] = MLIR_GetOpSuccessorOperand(ctx, pre_term, 0, k);
                                        vals[nso] = init_val;
                                        MLIR_SetOpSuccessorOperands(ctx, pre_term, 0, vals, nso + 1);
                                        free(vals);
                                    }
                                    // Append st_val to body_term -> hdr
                                    {
                                        size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, body_term, 0);
                                        MLIR_ValueHandle *vals = (MLIR_ValueHandle *)malloc((nso + 1) * sizeof(MLIR_ValueHandle));
                                        for (size_t k = 0; k < nso; k++) vals[k] = MLIR_GetOpSuccessorOperand(ctx, body_term, 0, k);
                                        vals[nso] = st_val;
                                        MLIR_SetOpSuccessorOperands(ctx, body_term, 0, vals, nso + 1);
                                        free(vals);
                                    }
                                    // Replace all `load(acc_ptr)` in `body` with `acc_phi` and detach them.
                                    for (size_t li = 0; li < n_acc_loads; li++) {
                                        MLIR_ValueHandle lv = MLIR_GetOpResult(ctx, acc_loads[li], 0);
                                        MLIR_ReplaceAllUsesOfValue(ctx, lv, acc_phi);
                                        // Also sweep `body` directly in case def-use tracking is off.
                                        size_t bno_cur = MLIR_GetBlockNumOps(ctx, blks[body]);
                                        for (size_t bi = 0; bi < bno_cur; bi++) {
                                            MLIR_OpHandle bop = MLIR_GetBlockOp(ctx, blks[body], bi);
                                            size_t bnop = MLIR_GetOpNumOperands(ctx, bop);
                                            for (size_t k = 0; k < bnop; k++)
                                                if (MLIR_GetOpOperand(ctx, bop, k) == lv)
                                                    MLIR_SetOpOperand(ctx, bop, k, acc_phi);
                                        }
                                        MLIR_DetachOp(ctx, acc_loads[li]);
                                    }
                                    // Insert final `store acc_phi, acc_ptr` at entry of `exit_bh`.
                                    MLIR_ValueHandle st_ops[2] = { acc_phi, acc_ptr };
                                    MLIR_OpSpec sp;
                                    memset(&sp, 0, sizeof(sp));
                                    sp.type = OP_TYPE_LLVM_STORE;
                                    sp.operands = st_ops;
                                    sp.n_operands = 2;
                                    MLIR_OpHandle ex_st = MLIR_CreateOpFromSpec(ctx, &sp);
                                    MLIR_InsertBlockOpAtIndex(ctx, exit_bh, ex_st, 0);
                                    // Detach body store and preheader store.
                                    MLIR_DetachOp(ctx, st_op);
                                    MLIR_DetachOp(ctx, pre_st);
                                }
                            }
                        }
                    }
                }
            }
        }
        vset_free(&in_loop);
    }

    // 4. Lightweight Dominator-Tree CSE for pure unary/binary ops + addressof (when nb <= 64).
    //    Deduplicates identical pure ops in blocks dominated by an earlier identical op.
    if (nb <= 256) {
        typedef struct {
            MLIR_OpType ty;
            MLIR_TypeHandle rty;
            MLIR_ValueHandle a0, a1;
            int64_t extra; // predicate for icmp, or hash for addressof
            string gname;
            size_t def_blk;
            MLIR_ValueHandle res;
        } DomEnt;
        size_t cap = 256, cnt = 0;
        DomEnt *ents = (DomEnt *)malloc(cap * sizeof(DomEnt));
        if (ents) {
            for (size_t b = 0; b < nb; b++) {
                MLIR_BlockHandle bk = blks[b];
                size_t i = 0;
                while (i < MLIR_GetBlockNumOps(ctx, bk)) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                    MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                    bool ok = (ot == OP_TYPE_LLVM_SEXT || ot == OP_TYPE_LLVM_ZEXT || ot == OP_TYPE_LLVM_TRUNC ||
                               ot == OP_TYPE_LLVM_ADD  || ot == OP_TYPE_LLVM_SUB  || ot == OP_TYPE_LLVM_MUL  ||
                               ot == OP_TYPE_LLVM_AND  || ot == OP_TYPE_LLVM_OR   || ot == OP_TYPE_LLVM_XOR  ||
                               ot == OP_TYPE_LLVM_SHL  || ot == OP_TYPE_LLVM_LSHR || ot == OP_TYPE_LLVM_ASHR ||
                               (nb <= 64 && ot == OP_TYPE_LLVM_MLIR_ADDRESSOF));
                    if (!ok || MLIR_GetOpNumResults(ctx, op) != 1) { i++; continue; }
                    MLIR_ValueHandle r = MLIR_GetOpResult(ctx, op, 0);
                    MLIR_TypeHandle rty = MLIR_GetValueType(ctx, r);
                    size_t nop = MLIR_GetOpNumOperands(ctx, op);
                    MLIR_ValueHandle a0 = nop >= 1 ? MLIR_GetOpOperand(ctx, op, 0) : MLIR_INVALID_HANDLE;
                    MLIR_ValueHandle a1 = nop >= 2 ? MLIR_GetOpOperand(ctx, op, 1) : MLIR_INVALID_HANDLE;
                    if ((ot == OP_TYPE_LLVM_ADD || ot == OP_TYPE_LLVM_MUL ||
                         ot == OP_TYPE_LLVM_AND || ot == OP_TYPE_LLVM_OR || ot == OP_TYPE_LLVM_XOR) &&
                        (uintptr_t)a1 < (uintptr_t)a0) {
                        MLIR_ValueHandle t = a0; a0 = a1; a1 = t;
                    }
                    string gn = {0};
                    if (ot == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
                        MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, MLIR_KEY_GLOBAL_NAME);
                        if (a == MLIR_INVALID_HANDLE) a = MLIR_GetOpAttr(ctx, op, MLIR_KEY_VALUE);
                        if (a != MLIR_INVALID_HANDLE) gn = MLIR_GetAttributeString(ctx, a);
                        if (!gn.size) { i++; continue; }
                    }
                    MLIR_ValueHandle hit = MLIR_INVALID_HANDLE;
                    for (size_t e = 0; e < cnt; e++) {
                        if (ents[e].ty != ot || ents[e].rty != rty) continue;
                        if (ot == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
                            if (ents[e].gname.size != gn.size ||
                                memcmp(ents[e].gname.str, gn.str, gn.size) != 0) continue;
                        } else {
                            if (ents[e].a0 != a0 || ents[e].a1 != a1) continue;
                        }
                        size_t db = ents[e].def_blk;
                        if (db == b || (dom[b * words + (db >> 6)] & (1ull << (db & 63)))) {
                            hit = ents[e].res;
                            break;
                        }
                    }
                    if (hit != MLIR_INVALID_HANDLE) {
                        MLIR_ReplaceAllUsesOfValue(ctx, r, hit);
                        MLIR_DetachOp(ctx, op);
                        continue;
                    }
                    if (cnt < 2048) {
                        if (cnt == cap) {
                            cap *= 2;
                            ents = (DomEnt *)realloc(ents, cap * sizeof(DomEnt));
                        }
                        ents[cnt].ty = ot; ents[cnt].rty = rty;
                        ents[cnt].a0 = a0; ents[cnt].a1 = a1;
                        ents[cnt].extra = 0; ents[cnt].gname = gn;
                        ents[cnt].def_blk = b; ents[cnt].res = r;
                        cnt++;
                    }
                    i++;
                }
            }
            free(ents);
        }
    }

    free(bhk); free(bhv);
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

// ---------------------------------------------------------------------------
// Shared helpers for inliner & TRE.
// ---------------------------------------------------------------------------
static string opt_strip_at(string s) {
    if (s.size && s.str[0] == '@') { s.str++; s.size--; }
    return s;
}

static string opt_attr_str(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, key);
    return a ? MLIR_GetAttributeString(ctx, a) : (string){0};
}

static bool opt_str_eq(string a, string b) {
    return a.size == b.size && (a.size == 0 || memcmp(a.str, b.str, a.size) == 0);
}

static uint32_t opt_str_hash(string s) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < s.size; i++) {
        h ^= (uint32_t)(unsigned char)s.str[i];
        h *= 16777619u;
    }
    return h;
}

static size_t opt_count_uses_in_region(MLIR_Context *ctx, MLIR_RegionHandle reg, MLIR_ValueHandle v) {
    size_t uses = 0;
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            for (size_t k = 0; k < nop; k++)
                if (MLIR_GetOpOperand(ctx, op, k) == v) uses++;
            if (i + 1 == no) {
                size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                    for (size_t k = 0; k < nso; k++)
                        if (MLIR_GetOpSuccessorOperand(ctx, op, s, k) == v) uses++;
                }
            }
        }
    }
    return uses;
}

static void opt_replace_uses_in_region(MLIR_Context *ctx, MLIR_RegionHandle reg,
                                       MLIR_ValueHandle old_v, MLIR_ValueHandle new_v) {
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            for (size_t k = 0; k < nop; k++) {
                if (MLIR_GetOpOperand(ctx, op, k) == old_v)
                    MLIR_SetOpOperand(ctx, op, k, new_v);
            }
            if (i + 1 == no) {
                size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                    bool any = false;
                    for (size_t k = 0; k < nso; k++) {
                        if (MLIR_GetOpSuccessorOperand(ctx, op, s, k) == old_v) { any = true; break; }
                    }
                    if (any) {
                        MLIR_ValueHandle *vals = (MLIR_ValueHandle *)malloc(nso * sizeof(MLIR_ValueHandle));
                        for (size_t k = 0; k < nso; k++) {
                            MLIR_ValueHandle x = MLIR_GetOpSuccessorOperand(ctx, op, s, k);
                            vals[k] = (x == old_v) ? new_v : x;
                        }
                        MLIR_SetOpSuccessorOperands(ctx, op, s, vals, nso);
                        free(vals);
                    }
                }
            }
        }
    }
}

static MLIR_OpHandle opt_create_br(MLIR_Context *ctx, MLIR_BlockHandle target,
                                   const MLIR_ValueHandle *args, size_t n_args) {
    MLIR_OpSpec s;
    memset(&s, 0, sizeof(s));
    uint32_t cnt = (uint32_t)n_args;
    s.type = OP_TYPE_LLVM_BR;
    s.operands = n_args ? args : NULL;
    s.n_operands = cnt;
    s.successors = &target;
    s.n_successors = 1;
    s.successor_operand_counts = n_args ? &cnt : NULL;
    return MLIR_CreateOpFromSpec(ctx, &s);
}

// ---------------------------------------------------------------------------
// Value -> Value map for cloning during inlining.
// ---------------------------------------------------------------------------
typedef struct {
    uintptr_t *k;
    MLIR_ValueHandle *v;
    size_t cap, n;
} OptVMap;

static void opt_vmap_init(OptVMap *m, size_t want) {
    size_t cap = 32;
    while (cap < want * 2) cap *= 2;
    m->k = (uintptr_t *)calloc(cap, sizeof(uintptr_t));
    m->v = (MLIR_ValueHandle *)malloc(cap * sizeof(MLIR_ValueHandle));
    m->cap = cap;
    m->n = 0;
}

static void opt_vmap_free(OptVMap *m) {
    free(m->k); free(m->v);
    m->k = NULL; m->v = NULL; m->cap = 0; m->n = 0;
}

static void opt_vmap_put(OptVMap *m, MLIR_ValueHandle key, MLIR_ValueHandle val) {
    if ((m->n + 1) * 2 >= m->cap) {
        size_t ncap = m->cap * 2;
        uintptr_t *nk = (uintptr_t *)calloc(ncap, sizeof(uintptr_t));
        MLIR_ValueHandle *nv = (MLIR_ValueHandle *)malloc(ncap * sizeof(MLIR_ValueHandle));
        size_t mask = ncap - 1;
        for (size_t i = 0; i < m->cap; i++) {
            if (!m->k[i]) continue;
            size_t j = (m->k[i] * 1099511628211ull) & mask;
            while (nk[j]) j = (j + 1) & mask;
            nk[j] = m->k[i]; nv[j] = m->v[i];
        }
        free(m->k); free(m->v);
        m->k = nk; m->v = nv; m->cap = ncap;
    }
    uintptr_t k = (uintptr_t)key;
    size_t mask = m->cap - 1;
    size_t j = (k * 1099511628211ull) & mask;
    while (m->k[j]) {
        if (m->k[j] == k) { m->v[j] = val; return; }
        j = (j + 1) & mask;
    }
    m->k[j] = k; m->v[j] = val; m->n++;
}

static MLIR_ValueHandle opt_vmap_get(const OptVMap *m, MLIR_ValueHandle key) {
    uintptr_t k = (uintptr_t)key;
    if (!k || m->cap == 0) return key;
    size_t mask = m->cap - 1;
    size_t j = (k * 1099511628211ull) & mask;
    while (m->k[j]) {
        if (m->k[j] == k) return m->v[j];
        j = (j + 1) & mask;
    }
    return key;
}

// ---------------------------------------------------------------------------
// Small-function inliner on flat `llvm`-dialect CFGs.
// ---------------------------------------------------------------------------
typedef struct {
    string name;
    MLIR_OpHandle fn;
    size_t n_ops;
    size_t n_blocks;
    bool has_alloca;
    bool only_devirt; // larger callee with indirect call: inline only if call passes addressof!
    bool is_fast_ret; // larger callee with fast-path early return: defer to later rounds so small wrappers inline first!
} InlCand;

static bool inl_check_candidate(MLIR_Context *ctx, MLIR_OpHandle fn, InlCand *out) {
    if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) return false;
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return false;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb == 0 || nb > 28) return false;
    string name = opt_attr_str(ctx, fn, MLIR_KEY_SYM_NAME);
    if (name.size == 0) return false;
    if (name.size == 4 && memcmp(name.str, "main", 4) == 0) return false;
    // Skip vararg functions.
    MLIR_AttributeHandle fta = MLIR_GetOpAttr(ctx, fn, MLIR_KEY_FUNCTION_TYPE);
    if (fta != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(ctx, fta);
        if (ft != MLIR_INVALID_HANDLE && MLIR_GetTypeFunctionIsVarArg(ctx, ft)) return false;
    }
    size_t total_ops = 0;
    bool has_alloca = false;
    bool has_call = false;
    bool has_indirect_call = false;
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        total_ops += no;
        if (total_ops > 180) return false;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpNumRegions(ctx, op) > 0) return false;
            MLIR_OpType t = MLIR_GetOpType(ctx, op);
            if (t == OP_TYPE_LLVM_INTR_VASTART || t == OP_TYPE_LLVM_INTR_VAEND ||
                t == OP_TYPE_LLVM_INTR_VACOPY || t == OP_TYPE_LLVM_UNREACHABLE)
                return false;
            if (t == OP_TYPE_LLVM_ALLOCA) has_alloca = true;
            if (t == OP_TYPE_LLVM_CALL || t == OP_TYPE_FUNC_CALL) {
                has_call = true;
                string cn = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE));
                if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE_NAME));
                if (!cn.size) has_indirect_call = true;
                else if (opt_str_eq(cn, name)) return false; // skip directly recursive callee
            }
        }
    }
    bool only_devirt = false;
    bool is_fast_ret = false;
    if (has_call) {
        if (nb > 8 || total_ops > 50) {
            if (has_indirect_call && nb <= 28 && total_ops <= 160) {
                only_devirt = true;
            } else if (nb <= 24 && total_ops <= 180) {
                // Fast-path early-return callee (e.g. CoreMark calc_func, Lua luaH_getint):
                // entry block ends in cond_br where one successor is a small
                // call-free return block (<= 14 ops).
                MLIR_BlockHandle e0 = MLIR_GetRegionBlock(ctx, reg, 0);
                size_t e0n = MLIR_GetBlockNumOps(ctx, e0);
                bool has_fast_ret = false;
                if (e0n > 0) {
                    MLIR_OpHandle et = MLIR_GetBlockOp(ctx, e0, e0n - 1);
                    if (MLIR_GetOpType(ctx, et) == OP_TYPE_LLVM_COND_BR) {
                        for (size_t si = 0; si < 2; si++) {
                            MLIR_BlockHandle sb = MLIR_GetOpSuccessor(ctx, et, si);
                            size_t sno = MLIR_GetBlockNumOps(ctx, sb);
                            if (sno > 0 && sno <= 14) {
                                MLIR_OpType st = MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, sb, sno - 1));
                                if (st == OP_TYPE_LLVM_RETURN || st == OP_TYPE_FUNC_RETURN) {
                                    bool sc = false;
                                    for (size_t k = 0; k < sno; k++) {
                                        MLIR_OpType kt = MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, sb, k));
                                        if (kt == OP_TYPE_LLVM_CALL || kt == OP_TYPE_FUNC_CALL) { sc = true; break; }
                                    }
                                    if (!sc) { has_fast_ret = true; break; }
                                }
                            }
                        }
                    }
                }
                if (!has_fast_ret) return false;
                is_fast_ret = true;
            } else {
                return false;
            }
        }
    } else {
        if (nb > 20 || total_ops > 150) return false;
    }
    out->name = name;
    out->fn = fn;
    out->n_ops = total_ops;
    out->n_blocks = nb;
    out->has_alloca = has_alloca;
    out->only_devirt = only_devirt;
    out->is_fast_ret = is_fast_ret;
    return true;
}

// Devirtualize indirect `llvm.call %fnptr(args...)` where `%fnptr` is
// `llvm.mlir.addressof @foo` into direct `llvm.call @foo(args...)`.
static void inl_devirt_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_CALL) continue;
            if (MLIR_GetOpAttr(ctx, op, MLIR_KEY_CALLEE) != MLIR_INVALID_HANDLE ||
                MLIR_GetOpAttr(ctx, op, MLIR_KEY_CALLEE_NAME) != MLIR_INVALID_HANDLE)
                continue;
            MLIR_AttributeHandle vct = MLIR_GetOpAttr(ctx, op, MLIR_KEY_VAR_CALLEE_TYPE);
            if (vct != MLIR_INVALID_HANDLE) {
                MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(ctx, vct);
                if (ft != MLIR_INVALID_HANDLE && MLIR_GetTypeFunctionIsVarArg(ctx, ft))
                    continue;
            }
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            if (nop < 1 || nop > 16) continue;
            MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, op, 0));
            if (def == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, def) != OP_TYPE_LLVM_MLIR_ADDRESSOF)
                continue;
            MLIR_AttributeHandle gname = MLIR_GetOpAttr(ctx, def, MLIR_KEY_GLOBAL_NAME);
            if (gname == MLIR_INVALID_HANDLE) gname = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
            if (gname == MLIR_INVALID_HANDLE) continue;
            MLIR_ValueHandle args[16];
            for (size_t k = 1; k < nop; k++) args[k - 1] = MLIR_GetOpOperand(ctx, op, k);
            size_t nres = MLIR_GetOpNumResults(ctx, op);
            MLIR_TypeHandle rty = nres ? MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, op, 0)) : MLIR_INVALID_HANDLE;
            MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_CALLEE, gname);
            MLIR_OpSpec sp;
            memset(&sp, 0, sizeof(sp));
            sp.type = OP_TYPE_LLVM_CALL;
            sp.operands = (nop > 1) ? args : NULL;
            sp.n_operands = nop - 1;
            sp.result_types = nres ? &rty : NULL;
            sp.n_results = nres;
            sp.attrs = &attr;
            sp.n_attrs = 1;
            MLIR_OpHandle dcall = MLIR_CreateOpFromSpec(ctx, &sp);
            MLIR_InsertBlockOpAtIndex(ctx, bk, dcall, i);
            if (nres == 1)
                MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetOpResult(ctx, op, 0), MLIR_GetOpResult(ctx, dcall, 0));
            MLIR_EraseOp(ctx, op);
        }
    }
}

static bool inl_inline_call(MLIR_Context *ctx, MLIR_RegionHandle caller_reg,
                            MLIR_BlockHandle call_blk, size_t call_idx,
                            MLIR_OpHandle call_op, MLIR_OpHandle callee_fn) {
    MLIR_RegionHandle creg = MLIR_GetOpRegion(ctx, callee_fn, 0);
    size_t cnb = MLIR_GetRegionNumBlocks(ctx, creg);
    MLIR_BlockHandle centry = MLIR_GetRegionBlock(ctx, creg, 0);
    size_t c_na = MLIR_GetBlockNumArgs(ctx, centry);
    size_t call_nop = MLIR_GetOpNumOperands(ctx, call_op);
    if (call_nop != c_na) return false;

    // Verify return arity/type matches call_op.
    size_t call_nres = MLIR_GetOpNumResults(ctx, call_op);
    MLIR_TypeHandle call_rty = call_nres ? MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, call_op, 0))
                                         : MLIR_INVALID_HANDLE;
    for (size_t b = 0; b < cnb; b++) {
        MLIR_BlockHandle cb = MLIR_GetRegionBlock(ctx, creg, b);
        size_t cno = MLIR_GetBlockNumOps(ctx, cb);
        if (cno == 0) return false;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, cb, cno - 1);
        MLIR_OpType tt = MLIR_GetOpType(ctx, term);
        if (tt == OP_TYPE_LLVM_RETURN || tt == OP_TYPE_FUNC_RETURN) {
            size_t rno = MLIR_GetOpNumOperands(ctx, term);
            if (call_nres == 0 && rno != 0) {
                // OK: caller ignores return value.
            } else if (call_nres == 1) {
                if (rno != 1) return false;
                if (MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, term, 0)) != call_rty) return false;
            }
        }
    }
    // Verify argument types match callee entry block args.
    for (size_t a = 0; a < c_na; a++) {
        if (MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, call_op, a)) !=
            MLIR_GetValueType(ctx, MLIR_GetBlockArg(ctx, centry, a)))
            return false;
    }

    // 1. Split `call_blk` after `call_idx` into `cont_blk`.
    MLIR_BlockHandle cont_blk = MLIR_CreateBlock(ctx);
    MLIR_InsertRegionBlockAfter(ctx, caller_reg, cont_blk, call_blk);
    MLIR_ValueHandle ret_arg = MLIR_INVALID_HANDLE;
    if (call_nres == 1) {
        ret_arg = MLIR_AddBlockArgument(ctx, cont_blk, call_rty, MLIR_INVALID_HANDLE);
        opt_replace_uses_in_region(ctx, caller_reg, MLIR_GetOpResult(ctx, call_op, 0), ret_arg);
    }
    // Move all ops after `call_idx` from `call_blk` into `cont_blk`.
    while (MLIR_GetBlockNumOps(ctx, call_blk) > call_idx + 1) {
        MLIR_OpHandle mv = MLIR_GetBlockOp(ctx, call_blk, call_idx + 1);
        MLIR_MoveOpToBlockEnd(ctx, mv, cont_blk);
    }

    // 2. Create cloned blocks for callee and map block args.
    MLIR_BlockHandle *cblks = (MLIR_BlockHandle *)malloc(cnb * sizeof(MLIR_BlockHandle));
    OptVMap vmap;
    opt_vmap_init(&vmap, 64);
    for (size_t a = 0; a < c_na; a++) {
        opt_vmap_put(&vmap, MLIR_GetBlockArg(ctx, centry, a), MLIR_GetOpOperand(ctx, call_op, a));
    }
    for (size_t b = 0; b < cnb; b++) {
        cblks[b] = MLIR_CreateBlock(ctx);
        MLIR_InsertRegionBlockBefore(ctx, caller_reg, cblks[b], cont_blk);
        if (b > 0) {
            MLIR_BlockHandle orig_b = MLIR_GetRegionBlock(ctx, creg, b);
            size_t bna = MLIR_GetBlockNumArgs(ctx, orig_b);
            for (size_t a = 0; a < bna; a++) {
                MLIR_ValueHandle oa = MLIR_GetBlockArg(ctx, orig_b, a);
                MLIR_ValueHandle na = MLIR_AddBlockArgument(ctx, cblks[b], MLIR_GetValueType(ctx, oa), MLIR_INVALID_HANDLE);
                opt_vmap_put(&vmap, oa, na);
            }
        }
    }

    // 3. Clone ops of each callee block.
    MLIR_BlockHandle caller_entry = MLIR_GetRegionBlock(ctx, caller_reg, 0);
    for (size_t b = 0; b < cnb; b++) {
        MLIR_BlockHandle orig_b = MLIR_GetRegionBlock(ctx, creg, b);
        size_t cno = MLIR_GetBlockNumOps(ctx, orig_b);
        for (size_t i = 0; i < cno; i++) {
            MLIR_OpHandle orig = MLIR_GetBlockOp(ctx, orig_b, i);
            MLIR_OpType ty = MLIR_GetOpType(ctx, orig);
            if (ty == OP_TYPE_LLVM_RETURN || ty == OP_TYPE_FUNC_RETURN) {
                MLIR_ValueHandle rv = MLIR_INVALID_HANDLE;
                size_t rn = 0;
                if (call_nres == 1) {
                    rv = opt_vmap_get(&vmap, MLIR_GetOpOperand(ctx, orig, 0));
                    rn = 1;
                }
                MLIR_OpHandle br = opt_create_br(ctx, cont_blk, rn ? &rv : NULL, rn);
                MLIR_AppendBlockOp(ctx, cblks[b], br);
                continue;
            }
            size_t nop = MLIR_GetOpNumOperands(ctx, orig);
            size_t nres = MLIR_GetOpNumResults(ctx, orig);
            size_t nattr = MLIR_GetOpNumAttributes(ctx, orig);
            size_t nsucc = MLIR_GetOpNumSuccessors(ctx, orig);

            MLIR_BlockHandle succ_buf[4];
            uint32_t sops_cnt[4];
            size_t total_sops = 0;
            if (nsucc > 4) continue;
            for (size_t s = 0; s < nsucc; s++) {
                size_t si = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, orig, s));
                succ_buf[s] = cblks[si];
                size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, orig, s);
                sops_cnt[s] = (uint32_t)nso;
                total_sops += nso;
            }

            size_t total_ops = nop + total_sops;
            MLIR_ValueHandle ops_buf[32];
            MLIR_ValueHandle *ops = total_ops <= 32 ? ops_buf : (MLIR_ValueHandle *)malloc(total_ops * sizeof(MLIR_ValueHandle));
            for (size_t k = 0; k < nop; k++) ops[k] = opt_vmap_get(&vmap, MLIR_GetOpOperand(ctx, orig, k));
            size_t pos = nop;
            for (size_t s = 0; s < nsucc; s++) {
                for (size_t k = 0; k < sops_cnt[s]; k++)
                    ops[pos++] = opt_vmap_get(&vmap, MLIR_GetOpSuccessorOperand(ctx, orig, s, k));
            }

            MLIR_TypeHandle rty_buf[4];
            MLIR_TypeHandle *rtys = nres <= 4 ? rty_buf : (MLIR_TypeHandle *)malloc(nres * sizeof(MLIR_TypeHandle));
            for (size_t k = 0; k < nres; k++) rtys[k] = MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, orig, k));

            MLIR_NamedAttr attr_buf[16];
            MLIR_NamedAttr *attrs = nattr <= 16 ? attr_buf : (MLIR_NamedAttr *)malloc(nattr * sizeof(MLIR_NamedAttr));
            for (size_t k = 0; k < nattr; k++) {
                attrs[k].key = MLIR_GetOpAttributeKey(ctx, orig, k);
                attrs[k].value = MLIR_GetOpAttribute(ctx, orig, k);
            }

            MLIR_OpSpec sp;
            memset(&sp, 0, sizeof(sp));
            sp.type = ty;
            sp.operands = total_ops ? ops : NULL;
            sp.n_operands = (uint32_t)total_ops;
            sp.result_types = nres ? rtys : NULL;
            sp.n_results = (uint32_t)nres;
            sp.attrs = nattr ? attrs : NULL;
            sp.n_attrs = (uint32_t)nattr;
            sp.successors = nsucc ? succ_buf : NULL;
            sp.n_successors = (uint32_t)nsucc;
            sp.successor_operand_counts = (nsucc && total_sops) ? sops_cnt : NULL;

            MLIR_OpHandle cloned = MLIR_CreateOpFromSpec(ctx, &sp);
            for (size_t k = 0; k < nres; k++)
                opt_vmap_put(&vmap, MLIR_GetOpResult(ctx, orig, k), MLIR_GetOpResult(ctx, cloned, k));

            if (ty == OP_TYPE_LLVM_ALLOCA) {
                // Hoist cloned alloca to caller's entry block so stack doesn't grow in loops.
                MLIR_InsertBlockOpAtIndex(ctx, caller_entry, cloned, 0);
            } else {
                MLIR_AppendBlockOp(ctx, cblks[b], cloned);
            }

            if (ops != ops_buf) free(ops);
            if (rtys != rty_buf) free(rtys);
            if (attrs != attr_buf) free(attrs);
        }
    }

    // 3b. Fixup pass: rewrite any forward-referenced operand (when callee's
    //     blocks are not in dominance/RPO order) using the now-complete vmap.
    for (size_t b = 0; b < cnb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, cblks[b]);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, cblks[b], i);
            size_t nop = MLIR_GetOpNumOperands(ctx, op);
            for (size_t k = 0; k < nop; k++) {
                MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, op, k);
                MLIR_ValueHandle nv = opt_vmap_get(&vmap, ov);
                if (nv != ov) MLIR_SetOpOperand(ctx, op, k, nv);
            }
            if (i + 1 == no) {
                size_t ns = MLIR_GetOpNumSuccessors(ctx, op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, s);
                    bool any = false;
                    for (size_t k = 0; k < nso; k++) {
                        MLIR_ValueHandle ov = MLIR_GetOpSuccessorOperand(ctx, op, s, k);
                        if (opt_vmap_get(&vmap, ov) != ov) { any = true; break; }
                    }
                    if (any) {
                        MLIR_ValueHandle *vals = (MLIR_ValueHandle *)malloc(nso * sizeof(MLIR_ValueHandle));
                        for (size_t k = 0; k < nso; k++)
                            vals[k] = opt_vmap_get(&vmap, MLIR_GetOpSuccessorOperand(ctx, op, s, k));
                        MLIR_SetOpSuccessorOperands(ctx, op, s, vals, nso);
                        free(vals);
                    }
                }
            }
        }
    }

    // 4. Replace `call_op` in `call_blk` with `llvm.br() [^cblks[0]]`.
    MLIR_OpHandle entry_br = opt_create_br(ctx, cblks[0], NULL, 0);
    MLIR_AppendBlockOp(ctx, call_blk, entry_br);
    MLIR_EraseOp(ctx, call_op);

    opt_vmap_free(&vmap);
    free(cblks);
    return true;
}

void mlir_llvm_inline(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_INLINE")) return;
    if (MLIR_GetOpNumRegions(ctx, module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n_fns = MLIR_GetBlockNumOps(ctx, mb);
    if (n_fns == 0) return;

    for (int round = 0; round < 4; round++) {
        // Devirtualize any indirect calls to known function addresses first.
        for (size_t fi = 0; fi < n_fns; fi++) {
            MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, fi);
            if (MLIR_GetOpType(ctx, fn) == OP_TYPE_LLVM_FUNC) inl_devirt_func(ctx, fn);
        }
        // Collect candidate callees into a hash table.
        InlCand *cands = (InlCand *)malloc(n_fns * sizeof(InlCand));
        size_t nc = 0;
        for (size_t i = 0; i < n_fns; i++) {
            MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
            if (inl_check_candidate(ctx, fn, &cands[nc])) nc++;
        }
        if (nc == 0) { free(cands); break; }

        size_t hcap = 64;
        while (hcap < nc * 2) hcap *= 2;
        size_t *ht = (size_t *)calloc(hcap, sizeof(size_t));
        size_t hmask = hcap - 1;
        for (size_t i = 0; i < nc; i++) {
            size_t j = opt_str_hash(cands[i].name) & hmask;
            while (ht[j]) j = (j + 1) & hmask;
            ht[j] = i + 1;
        }

        bool inlined_any = false;
        for (size_t fi = 0; fi < n_fns; fi++) {
            MLIR_OpHandle caller = MLIR_GetBlockOp(ctx, mb, fi);
            if (MLIR_GetOpType(ctx, caller) != OP_TYPE_LLVM_FUNC) continue;
            if (MLIR_GetOpNumRegions(ctx, caller) == 0) continue;
            MLIR_RegionHandle creg = MLIR_GetOpRegion(ctx, caller, 0);
            size_t cnb = MLIR_GetRegionNumBlocks(ctx, creg);
            if (cnb == 0) continue;
            string caller_name = opt_attr_str(ctx, caller, MLIR_KEY_SYM_NAME);
            size_t caller_ops = 0;
            for (size_t b = 0; b < cnb; b++)
                caller_ops += MLIR_GetBlockNumOps(ctx, MLIR_GetRegionBlock(ctx, creg, b));
            bool is_large_caller = (caller_ops > 2000);
            if (caller_ops > 8000) continue;
            size_t caller_lim = is_large_caller ? 12000 : 2000;
            int max_inl = is_large_caller ? 64 : 20;

            int n_inlined = 0;
            for (size_t b = 0; b < MLIR_GetRegionNumBlocks(ctx, creg) && caller_ops <= caller_lim && n_inlined < max_inl; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, creg, b);
                for (size_t i = 0; i < MLIR_GetBlockNumOps(ctx, bk) && caller_ops <= caller_lim && n_inlined < max_inl; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                    MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                    if (ot != OP_TYPE_LLVM_CALL && ot != OP_TYPE_FUNC_CALL) continue;
                    string cname = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE));
                    if (!cname.size) cname = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE_NAME));
                    if (!cname.size || opt_str_eq(cname, caller_name)) continue;

                    // Lookup in candidate hash table.
                    size_t j = opt_str_hash(cname) & hmask;
                    InlCand *found = NULL;
                    while (ht[j]) {
                        InlCand *c = &cands[ht[j] - 1];
                        if (opt_str_eq(c->name, cname)) { found = c; break; }
                        j = (j + 1) & hmask;
                    }
                    if (!found) continue;
                    if (round < 2 && !found->only_devirt && (found->is_fast_ret || found->n_ops > 30)) continue;
                    if (is_large_caller && (found->n_ops > 180 || found->n_blocks > 24)) continue;
                    if (found->only_devirt) {
                        bool passes_fnptr = false;
                        size_t cnop = MLIR_GetOpNumOperands(ctx, op);
                        for (size_t k = 0; k < cnop; k++) {
                            MLIR_OpHandle ad = MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, op, k));
                            if (ad != MLIR_INVALID_HANDLE && MLIR_GetOpType(ctx, ad) == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
                                passes_fnptr = true; break;
                            }
                        }
                        if (!passes_fnptr) continue;
                    }
                    if (inl_inline_call(ctx, creg, bk, i, op, found->fn)) {
                        caller_ops += found->n_ops;
                        n_inlined++;
                        inlined_any = true;
                        break; // `bk` was split; remaining ops are now in `cont_blk` (visited later)
                    }
                }
            }
        }
        free(ht);
        free(cands);
        if (!inlined_any && round >= 2) break;
    }
}

// ---------------------------------------------------------------------------
// Tail-recursion elimination (direct + accumulator ADD/MUL) on flat `llvm` CFGs.
// Turns e.g. `fib(n-1) + fib(n-2)` into an accumulator loop (halving calls).
// ---------------------------------------------------------------------------
static void tre_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) return;
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb == 0) return;
    string sym = opt_attr_str(ctx, fn, MLIR_KEY_SYM_NAME);
    if (!sym.size) return;

    MLIR_BlockHandle bb0 = MLIR_GetRegionBlock(ctx, reg, 0);
    size_t na = MLIR_GetBlockNumArgs(ctx, bb0);

    // Reject functions with allocas or varargs (stack addresses could escape).
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpType t = MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, bk, i));
            if (t == OP_TYPE_LLVM_ALLOCA || t == OP_TYPE_LLVM_INTR_VASTART) return;
        }
    }

    // Scan return blocks for direct or accumulator (ADD/MUL) self-tail-calls.
    bool has_direct = false;
    MLIR_OpType acc_kind = (MLIR_OpType)0;
    MLIR_TypeHandle acc_ty = MLIR_INVALID_HANDLE;

    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        if (no < 2) continue;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
        MLIR_OpType tt = MLIR_GetOpType(ctx, term);
        if (tt != OP_TYPE_LLVM_RETURN && tt != OP_TYPE_FUNC_RETURN) continue;
        size_t rno = MLIR_GetOpNumOperands(ctx, term);

        // Direct tail call?
        MLIR_OpHandle prev = MLIR_GetBlockOp(ctx, bk, no - 2);
        MLIR_OpType pt = MLIR_GetOpType(ctx, prev);
        if ((pt == OP_TYPE_LLVM_CALL || pt == OP_TYPE_FUNC_CALL) &&
            MLIR_GetOpNumOperands(ctx, prev) == na) {
            string cn = opt_strip_at(opt_attr_str(ctx, prev, MLIR_KEY_CALLEE));
            if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, prev, MLIR_KEY_CALLEE_NAME));
            if (opt_str_eq(cn, sym)) {
                bool match = (rno == 0 && MLIR_GetOpNumResults(ctx, prev) == 0) ||
                             (rno == 1 && MLIR_GetOpNumResults(ctx, prev) == 1 &&
                              MLIR_GetOpOperand(ctx, term, 0) == MLIR_GetOpResult(ctx, prev, 0));
                if (match) { has_direct = true; continue; }
            }
        }
        // Accumulator tail call: `c = call @sym(...); acc = add/mul(other, c); return acc`
        if (rno == 1 && no >= 3 && (pt == OP_TYPE_LLVM_ADD || pt == OP_TYPE_LLVM_MUL)) {
            MLIR_ValueHandle rv = MLIR_GetOpOperand(ctx, term, 0);
            if (MLIR_GetOpResult(ctx, prev, 0) == rv) {
                MLIR_TypeHandle ty = MLIR_GetValueType(ctx, rv);
                if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_INTEGER) {
                    MLIR_OpHandle call_op = MLIR_GetBlockOp(ctx, bk, no - 3);
                    MLIR_OpType ct = MLIR_GetOpType(ctx, call_op);
                    if ((ct == OP_TYPE_LLVM_CALL || ct == OP_TYPE_FUNC_CALL) &&
                        MLIR_GetOpNumOperands(ctx, call_op) == na &&
                        MLIR_GetOpNumResults(ctx, call_op) == 1) {
                        string cn = opt_strip_at(opt_attr_str(ctx, call_op, MLIR_KEY_CALLEE));
                        if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, call_op, MLIR_KEY_CALLEE_NAME));
                        MLIR_ValueHandle cr = MLIR_GetOpResult(ctx, call_op, 0);
                        MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, prev, 0);
                        MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, prev, 1);
                        if (opt_str_eq(cn, sym) && (a0 == cr || a1 == cr) && a0 != a1 &&
                            opt_count_uses_in_region(ctx, reg, cr) == 1 &&
                            opt_count_uses_in_region(ctx, reg, rv) == 1) {
                            if (acc_kind == (MLIR_OpType)0) { acc_kind = pt; acc_ty = ty; }
                            else if (acc_kind != pt) return; // mixed ops
                        }
                    }
                }
            }
        }
    }
    if (!has_direct && acc_kind == (MLIR_OpType)0) return;
    if (has_direct && acc_kind != (MLIR_OpType)0) return; // keep simple

    // Create new entry block before old `bb0`.
    MLIR_BlockHandle new_entry = MLIR_CreateBlock(ctx);
    MLIR_InsertRegionBlockBefore(ctx, reg, new_entry, bb0);
    MLIR_ValueHandle *entry_args = (MLIR_ValueHandle *)malloc((na + 1) * sizeof(MLIR_ValueHandle));
    for (size_t a = 0; a < na; a++) {
        entry_args[a] = MLIR_AddBlockArgument(ctx, new_entry,
                                              MLIR_GetValueType(ctx, MLIR_GetBlockArg(ctx, bb0, a)),
                                              MLIR_INVALID_HANDLE);
    }

    MLIR_ValueHandle acc_arg = MLIR_INVALID_HANDLE;
    size_t br_na = na;
    if (acc_kind != (MLIR_OpType)0) {
        acc_arg = MLIR_AddBlockArgument(ctx, bb0, acc_ty, MLIR_INVALID_HANDLE);
        // Materialize identity constant (0 for ADD, 1 for MUL) in new_entry.
        MLIR_Builder bd = MLIR_BuilderAtEnd(ctx, new_entry);
        MLIR_ValueHandle init = MLIR_BuildNullaryAttr(&bd, OP_TYPE_LLVM_MLIR_CONSTANT, acc_ty,
            MLIR_KEY_VALUE, MLIR_CreateAttributeInteger(ctx, acc_kind == OP_TYPE_LLVM_ADD ? 0 : 1, acc_ty));
        entry_args[na] = init;
        br_na = na + 1;
    }
    MLIR_AppendBlockOp(ctx, new_entry, opt_create_br(ctx, bb0, br_na ? entry_args : NULL, br_na));
    free(entry_args);

    // Rewrite every return block (skipping new_entry at index 0).
    size_t nb2 = MLIR_GetRegionNumBlocks(ctx, reg);
    for (size_t b = 1; b < nb2; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        if (no == 0) continue;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
        MLIR_OpType tt = MLIR_GetOpType(ctx, term);
        if (tt != OP_TYPE_LLVM_RETURN && tt != OP_TYPE_FUNC_RETURN) continue;
        size_t rno = MLIR_GetOpNumOperands(ctx, term);

        if (has_direct && no >= 2) {
            MLIR_OpHandle prev = MLIR_GetBlockOp(ctx, bk, no - 2);
            MLIR_OpType pt = MLIR_GetOpType(ctx, prev);
            if ((pt == OP_TYPE_LLVM_CALL || pt == OP_TYPE_FUNC_CALL) &&
                MLIR_GetOpNumOperands(ctx, prev) == na) {
                string cn = opt_strip_at(opt_attr_str(ctx, prev, MLIR_KEY_CALLEE));
                if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, prev, MLIR_KEY_CALLEE_NAME));
                bool match = opt_str_eq(cn, sym) &&
                             ((rno == 0 && MLIR_GetOpNumResults(ctx, prev) == 0) ||
                              (rno == 1 && MLIR_GetOpNumResults(ctx, prev) == 1 &&
                               MLIR_GetOpOperand(ctx, term, 0) == MLIR_GetOpResult(ctx, prev, 0)));
                if (match) {
                    MLIR_ValueHandle *args = na ? (MLIR_ValueHandle *)malloc(na * sizeof(MLIR_ValueHandle)) : NULL;
                    for (size_t a = 0; a < na; a++) args[a] = MLIR_GetOpOperand(ctx, prev, a);
                    MLIR_OpHandle br = opt_create_br(ctx, bb0, args, na);
                    free(args);
                    MLIR_EraseOp(ctx, term);
                    MLIR_EraseOp(ctx, prev);
                    MLIR_AppendBlockOp(ctx, bk, br);
                    continue;
                }
            }
        }
        if (acc_kind != (MLIR_OpType)0) {
            bool is_acc_tail = false;
            if (rno == 1 && no >= 3) {
                MLIR_OpHandle prev = MLIR_GetBlockOp(ctx, bk, no - 2);
                MLIR_OpHandle call_op = MLIR_GetBlockOp(ctx, bk, no - 3);
                MLIR_OpType ct = MLIR_GetOpType(ctx, call_op);
                if (MLIR_GetOpType(ctx, prev) == acc_kind &&
                    MLIR_GetOpResult(ctx, prev, 0) == MLIR_GetOpOperand(ctx, term, 0) &&
                    (ct == OP_TYPE_LLVM_CALL || ct == OP_TYPE_FUNC_CALL) &&
                    MLIR_GetOpNumOperands(ctx, call_op) == na &&
                    MLIR_GetOpNumResults(ctx, call_op) == 1) {
                    string cn = opt_strip_at(opt_attr_str(ctx, call_op, MLIR_KEY_CALLEE));
                    if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, call_op, MLIR_KEY_CALLEE_NAME));
                    MLIR_ValueHandle cr = MLIR_GetOpResult(ctx, call_op, 0);
                    MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, prev, 0);
                    MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, prev, 1);
                    if (opt_str_eq(cn, sym) && (a0 == cr || a1 == cr) && a0 != a1) {
                        MLIR_ValueHandle other = (a0 == cr) ? a1 : a0;
                        MLIR_ValueHandle *args = (MLIR_ValueHandle *)malloc((na + 1) * sizeof(MLIR_ValueHandle));
                        for (size_t a = 0; a < na; a++) args[a] = MLIR_GetOpOperand(ctx, call_op, a);
                        MLIR_EraseOp(ctx, term);
                        MLIR_EraseOp(ctx, prev);
                        MLIR_EraseOp(ctx, call_op);
                        MLIR_Builder bd = MLIR_BuilderAtEnd(ctx, bk);
                        MLIR_ValueHandle next_acc = MLIR_BuildBinary(&bd, acc_kind, acc_ty, acc_arg, other);
                        args[na] = next_acc;
                        MLIR_AppendBlockOp(ctx, bk, opt_create_br(ctx, bb0, args, na + 1));
                        free(args);
                        is_acc_tail = true;
                    }
                }
            }
            if (!is_acc_tail && rno == 1) {
                // Non-tail return: combine with accumulator (`rv + acc_arg`).
                MLIR_ValueHandle rv = MLIR_GetOpOperand(ctx, term, 0);
                MLIR_OpSpec sp;
                memset(&sp, 0, sizeof(sp));
                MLIR_ValueHandle ops2[2] = { rv, acc_arg };
                sp.type = acc_kind;
                sp.operands = ops2;
                sp.n_operands = 2;
                sp.result_types = &acc_ty;
                sp.n_results = 1;
                MLIR_OpHandle comb = MLIR_CreateOpFromSpec(ctx, &sp);
                MLIR_InsertBlockOpAtIndex(ctx, bk, comb, no - 1);
                MLIR_SetOpOperand(ctx, term, 0, MLIR_GetOpResult(ctx, comb, 0));
            }
        }
    }
}

void mlir_llvm_tre(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_TRE")) return;
    if (MLIR_GetOpNumRegions(ctx, module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        tre_func(ctx, fn);
    }
}

// ---------------------------------------------------------------------------
// Simplify-CFG: thread branches through empty forwarding blocks & erase dead
// unreachable blocks.
// ---------------------------------------------------------------------------
static void simplify_cfg_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 3) return;

    MLIR_BlockHandle entry = MLIR_GetRegionBlock(ctx, reg, 0);

    for (int iter = 0; iter < 4; iter++) {
        bool changed = false;
        nb = MLIR_GetRegionNumBlocks(ctx, reg);
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            if (no == 0) continue;
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            MLIR_OpType tt = MLIR_GetOpType(ctx, term);
            if (tt != OP_TYPE_LLVM_BR && tt != OP_TYPE_LLVM_COND_BR) continue;
            size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
            bool need_rebuild = false;
            MLIR_BlockHandle new_succ[2];
            MLIR_ValueHandle new_sops[2][64];
            uint32_t new_nso[2];
            for (size_t si = 0; si < ns && si < 2; si++) {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(ctx, term, si);
                size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, si);
                new_succ[si] = tgt;
                new_nso[si] = (uint32_t)nso;
                if (nso <= 64) {
                    for (size_t k = 0; k < nso; k++)
                        new_sops[si][k] = MLIR_GetOpSuccessorOperand(ctx, term, si, k);
                }
                if (tgt == entry || tgt == bk || nso > 64) continue;
                if (MLIR_GetBlockNumOps(ctx, tgt) != 1) continue;
                MLIR_OpHandle tterm = MLIR_GetBlockOp(ctx, tgt, 0);
                if (MLIR_GetOpType(ctx, tterm) != OP_TYPE_LLVM_BR) continue;
                MLIR_BlockHandle dst = MLIR_GetOpSuccessor(ctx, tterm, 0);
                if (dst == tgt) continue;
                size_t tnso = MLIR_GetOpNumSuccessorOperands(ctx, tterm, 0);
                if (tnso > 64) continue;
                // Avoid creating a cond_br with both arms going to the same block.
                if (tt == OP_TYPE_LLVM_COND_BR) {
                    MLIR_BlockHandle other = MLIR_GetOpSuccessor(ctx, term, 1 - si);
                    if (other == dst) continue;
                }
                size_t tna = MLIR_GetBlockNumArgs(ctx, tgt);
                if (tna != nso) continue;
                // Ensure none of tgt's block args are referenced outside tgt's terminator
                // (otherwise bypassing tgt would break SSA dominance).
                bool arg_escapes = false;
                if (tna > 0) {
                    for (size_t ob = 0; ob < nb && !arg_escapes; ob++) {
                        MLIR_BlockHandle obk = MLIR_GetRegionBlock(ctx, reg, ob);
                        if (obk == tgt) continue;
                        size_t ono = MLIR_GetBlockNumOps(ctx, obk);
                        for (size_t oi = 0; oi < ono && !arg_escapes; oi++) {
                            MLIR_OpHandle oop = MLIR_GetBlockOp(ctx, obk, oi);
                            size_t onop = MLIR_GetOpNumOperands(ctx, oop);
                            for (size_t k = 0; k < onop && !arg_escapes; k++) {
                                MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, oop, k);
                                for (size_t a = 0; a < tna; a++)
                                    if (ov == MLIR_GetBlockArg(ctx, tgt, a)) { arg_escapes = true; break; }
                            }
                            if (oi + 1 == ono && !arg_escapes) {
                                size_t ons = MLIR_GetOpNumSuccessors(ctx, oop);
                                for (size_t osi = 0; osi < ons && !arg_escapes; osi++) {
                                    size_t onso = MLIR_GetOpNumSuccessorOperands(ctx, oop, osi);
                                    for (size_t k = 0; k < onso && !arg_escapes; k++) {
                                        MLIR_ValueHandle ov = MLIR_GetOpSuccessorOperand(ctx, oop, osi, k);
                                        for (size_t a = 0; a < tna; a++)
                                            if (ov == MLIR_GetBlockArg(ctx, tgt, a)) { arg_escapes = true; break; }
                                    }
                                }
                            }
                        }
                    }
                }
                if (arg_escapes) continue;
                for (size_t k = 0; k < tnso; k++) {
                    MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(ctx, tterm, 0, k);
                    for (size_t a = 0; a < tna; a++) {
                        if (v == MLIR_GetBlockArg(ctx, tgt, a)) {
                            v = MLIR_GetOpSuccessorOperand(ctx, term, si, a);
                            break;
                        }
                    }
                    new_sops[si][k] = v;
                }
                new_succ[si] = dst;
                new_nso[si] = (uint32_t)tnso;
                need_rebuild = true;
            }
            if (!need_rebuild) continue;
            if (tt == OP_TYPE_LLVM_COND_BR && new_succ[0] == new_succ[1]) continue;

            MLIR_ValueHandle flat_ops[129];
            size_t n_flat = 0;
            if (tt == OP_TYPE_LLVM_COND_BR) {
                flat_ops[n_flat++] = MLIR_GetOpOperand(ctx, term, 0);
            }
            for (size_t si = 0; si < ns; si++) {
                for (size_t k = 0; k < new_nso[si]; k++)
                    flat_ops[n_flat++] = new_sops[si][k];
            }
            MLIR_OpSpec sp;
            memset(&sp, 0, sizeof(sp));
            sp.type = tt;
            sp.operands = flat_ops;
            sp.n_operands = n_flat;
            sp.successors = new_succ;
            sp.n_successors = ns;
            sp.successor_operand_counts = new_nso;
            MLIR_OpHandle nt = MLIR_CreateOpFromSpec(ctx, &sp);
            MLIR_InsertBlockOpAtIndex(ctx, bk, nt, no - 1);
            MLIR_DetachOp(ctx, term);
            changed = true;
        }
        if (!changed) break;
    }

    // BFS from entry block and erase unreachable blocks.
    nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb <= 1) return;
    size_t hcap = 16;
    while (hcap < nb * 2) hcap *= 2;
    uintptr_t *hk = (uintptr_t *)calloc(hcap, sizeof(uintptr_t));
    int32_t   *hv = (int32_t   *)malloc(hcap * sizeof(int32_t));
    uint8_t   *vis = (uint8_t  *)calloc(nb, 1);
    size_t    *q   = (size_t   *)malloc(nb * sizeof(size_t));
    if (hk && hv && vis && q) {
        size_t hmask = hcap - 1;
        for (size_t b = 0; b < nb; b++) {
            uintptr_t k = (uintptr_t)MLIR_GetRegionBlock(ctx, reg, b);
            size_t s = (k * 1099511628211ull) & hmask;
            while (hk[s]) s = (s + 1) & hmask;
            hk[s] = k; hv[s] = (int32_t)b;
        }
        size_t qh = 0, qt = 0;
        vis[0] = 1; q[qt++] = 0;
        while (qh < qt) {
            size_t b = q[qh++];
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            if (no == 0) continue;
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
            for (size_t si = 0; si < ns; si++) {
                uintptr_t sk = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, si);
                size_t s = (sk * 1099511628211ull) & hmask;
                while (hk[s]) {
                    if (hk[s] == sk) {
                        size_t tb = (size_t)hv[s];
                        if (!vis[tb]) { vis[tb] = 1; q[qt++] = tb; }
                        break;
                    }
                    s = (s + 1) & hmask;
                }
            }
        }
        for (size_t b = nb; b-- > 1; ) {
            if (!vis[b]) {
                MLIR_EraseBlock(ctx, MLIR_GetRegionBlock(ctx, reg, b));
            }
        }
    }
    free(hk); free(hv); free(vis); free(q);

    // Merge single-predecessor unconditional branch chains (`B -> br(args) -> C`
    // where `pred_cnt[C] == 1 && C != entry && C != B`).
    for (int miter = 0; miter < 4; miter++) {
        nb = MLIR_GetRegionNumBlocks(ctx, reg);
        if (nb < 2) break;
        size_t mcap = 16;
        while (mcap < nb * 2) mcap *= 2;
        uintptr_t *mhk = (uintptr_t *)calloc(mcap, sizeof(uintptr_t));
        size_t    *mhv = (size_t    *)malloc(mcap * sizeof(size_t));
        uint32_t  *mpc = (uint32_t  *)calloc(nb, sizeof(uint32_t));
        if (!mhk || !mhv || !mpc) { free(mhk); free(mhv); free(mpc); break; }
        size_t mmask = mcap - 1;
        for (size_t b = 0; b < nb; b++) {
            uintptr_t k = (uintptr_t)MLIR_GetRegionBlock(ctx, reg, b);
            size_t s = (k * 1099511628211ull) & mmask;
            while (mhk[s]) s = (s + 1) & mmask;
            mhk[s] = k; mhv[s] = b;
        }
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            if (no == 0) continue;
            MLIR_OpHandle t = MLIR_GetBlockOp(ctx, bk, no - 1);
            size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
            for (size_t si = 0; si < ns; si++) {
                uintptr_t sk = (uintptr_t)MLIR_GetOpSuccessor(ctx, t, si);
                size_t s = (sk * 1099511628211ull) & mmask;
                while (mhk[s]) { if (mhk[s] == sk) { mpc[mhv[s]]++; break; } s = (s + 1) & mmask; }
            }
        }
        bool merged = false;
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, bk);
            if (no == 0) continue;
            MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
            if (MLIR_GetOpType(ctx, term) != OP_TYPE_LLVM_BR) continue;
            MLIR_BlockHandle dst = MLIR_GetOpSuccessor(ctx, term, 0);
            if (dst == entry || dst == bk) continue;
            uintptr_t dk = (uintptr_t)dst;
            size_t s = (dk * 1099511628211ull) & mmask;
            size_t di = nb;
            while (mhk[s]) { if (mhk[s] == dk) { di = mhv[s]; break; } s = (s + 1) & mmask; }
            if (di >= nb || mpc[di] != 1) continue;
            size_t dna = MLIR_GetBlockNumArgs(ctx, dst);
            size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, term, 0);
            if (dna != nso) continue;
            for (size_t a = 0; a < dna; a++) {
                MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetBlockArg(ctx, dst, a),
                                           MLIR_GetOpSuccessorOperand(ctx, term, 0, a));
            }
            MLIR_DetachOp(ctx, term);
            while (MLIR_GetBlockNumOps(ctx, dst) > 0) {
                MLIR_OpHandle mop = MLIR_GetBlockOp(ctx, dst, 0);
                MLIR_MoveOpToBlockEnd(ctx, mop, bk);
            }
            MLIR_EraseBlock(ctx, dst);
            merged = true;
            break;
        }
        free(mhk); free(mhv); free(mpc);
        if (!merged) break;
    }

    // Hot-fallthrough RPO layout for small/medium loop functions where an exit/return
    // block sits strictly before a loop backedge (e.g. CoreMark core_state_transition).
    nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb >= 4 && nb <= 64) {
        MLIR_BlockHandle blks[64];
        uint8_t is_ret[64];
        size_t max_latch = 0;
        bool has_ret_in_loop = false;
        size_t ret_preds = 0;
        for (size_t b = 0; b < nb; b++) {
            blks[b] = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
            is_ret[b] = 0;
            if (no == 0) continue;
            MLIR_OpHandle t = MLIR_GetBlockOp(ctx, blks[b], no - 1);
            MLIR_OpType tt = MLIR_GetOpType(ctx, t);
            if (tt == OP_TYPE_LLVM_RETURN || tt == OP_TYPE_FUNC_RETURN) {
                is_ret[b] = 1;
            } else {
                size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
                for (size_t si = 0; si < ns; si++) {
                    size_t ti = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, t, si));
                    if (ti <= b && b > max_latch) max_latch = b;
                }
            }
        }
        for (size_t b = 1; b < max_latch; b++) {
            if (is_ret[b]) { has_ret_in_loop = true; break; }
        }
        if (has_ret_in_loop) {
            for (size_t b = 0; b < nb; b++) {
                MLIR_OpHandle t = MLIR_GetBlockTerminator(ctx, blks[b]);
                if (t == MLIR_INVALID_HANDLE) continue;
                size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
                for (size_t si = 0; si < ns; si++) {
                    size_t ti = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, t, si));
                    if (ti < nb && is_ret[ti]) ret_preds++;
                }
            }
        }
        if (has_ret_in_loop && ret_preds >= 2) {
            uint8_t vis[64];
            memset(vis, 0, nb);
            size_t post[64], n_post = 0;
            size_t st_b[64], st_si[64], sp = 0;
            vis[0] = 1;
            st_b[0] = 0; st_si[0] = 0; sp = 1;
            while (sp > 0) {
                size_t cur = st_b[sp - 1];
                MLIR_OpHandle t = MLIR_GetBlockTerminator(ctx, blks[cur]);
                size_t ns = (t != MLIR_INVALID_HANDLE) ? MLIR_GetOpNumSuccessors(ctx, t) : 0;
                // Build ordered successor list: put is_ret successors FIRST so in post-order
                // they finish FIRST -> in RPO they are placed LAST (at the end of the function)!
                size_t ord[64];
                size_t nord = 0;
                for (size_t pass = 0; pass < 2; pass++) {
                    for (size_t si = 0; si < ns && nord < 64; si++) {
                        size_t ti = MLIR_GetBlockIndex(ctx, MLIR_GetOpSuccessor(ctx, t, si));
                        if (ti >= nb) continue;
                        if ((is_ret[ti] != 0) == (pass == 0)) ord[nord++] = ti;
                    }
                }
                bool pushed = false;
                while (st_si[sp - 1] < nord) {
                    size_t nxt = ord[st_si[sp - 1]++];
                    if (!vis[nxt]) {
                        vis[nxt] = 1;
                        st_b[sp] = nxt; st_si[sp] = 0; sp++;
                        pushed = true;
                        break;
                    }
                }
                if (!pushed) {
                    post[n_post++] = cur;
                    sp--;
                }
            }
            // Reverse post-order -> RPO (entry block 0 is post[n_post - 1]).
            MLIR_BlockHandle rpo[64];
            size_t n_rpo = 0;
            for (size_t i = n_post; i-- > 0; ) {
                if (post[i] != 0) rpo[n_rpo++] = blks[post[i]];
            }
            if (n_rpo > 0) {
                MLIR_MoveBlocksToRegionEnd(ctx, rpo, n_rpo, reg);
            }
        }
    }
}

static bool ifcvt_is_cheap_pure(MLIR_Context *ctx, MLIR_OpHandle op) {
    switch (MLIR_GetOpType(ctx, op)) {
    case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ZERO:
    case OP_TYPE_LLVM_MLIR_NULL:
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR:  case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_TRUNC:
        return true;
    default:
        return false;
    }
}

// If-conversion: convert small 1-arm triangle diamonds (`if (c) x = pure(x);`)
// into branchless `llvm.select` (-> `cmov`), eliminating unpredictable branches
// (e.g. CoreMark's `crcu8` 50/50 CRC polynomial branch).
static void if_convert_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 3) return;

    // Count predecessors of each block.
    size_t hcap = 16;
    while (hcap < nb * 2) hcap *= 2;
    uintptr_t *hk = (uintptr_t *)calloc(hcap, sizeof(uintptr_t));
    int32_t   *hv = (int32_t   *)malloc(hcap * sizeof(int32_t));
    uint32_t  *pc = (uint32_t  *)calloc(nb, sizeof(uint32_t));
    if (!hk || !hv || !pc) { free(hk); free(hv); free(pc); return; }
    size_t hmask = hcap - 1;
    for (size_t b = 0; b < nb; b++) {
        uintptr_t k = (uintptr_t)MLIR_GetRegionBlock(ctx, reg, b);
        size_t s = (k * 1099511628211ull) & hmask;
        while (hk[s]) s = (s + 1) & hmask;
        hk[s] = k; hv[s] = (int32_t)b;
    }
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        if (no == 0) continue;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
        for (size_t si = 0; si < ns; si++) {
            uintptr_t sk = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, si);
            size_t s = (sk * 1099511628211ull) & hmask;
            while (hk[s]) { if (hk[s] == sk) { pc[hv[s]]++; break; } s = (s + 1) & hmask; }
        }
    }

    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        if (no == 0) continue;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, bk, no - 1);
        if (MLIR_GetOpType(ctx, term) != OP_TYPE_LLVM_COND_BR) continue;
        MLIR_BlockHandle s0 = MLIR_GetOpSuccessor(ctx, term, 0);
        MLIR_BlockHandle s1 = MLIR_GetOpSuccessor(ctx, term, 1);
        size_t nso0 = MLIR_GetOpNumSuccessorOperands(ctx, term, 0);
        size_t nso1 = MLIR_GetOpNumSuccessorOperands(ctx, term, 1);

        // Try both orientations: arm=0 is the single-pred small block, merge=s1 (or vice versa).
        for (int arm = 0; arm < 2; arm++) {
            MLIR_BlockHandle arm_blk = (arm == 0) ? s0 : s1;
            MLIR_BlockHandle mrg_blk = (arm == 0) ? s1 : s0;
            size_t arm_nso = (arm == 0) ? nso0 : nso1;
            size_t mrg_nso = (arm == 0) ? nso1 : nso0;
            if (arm_nso != 0 || mrg_nso != 1 || arm_blk == bk || arm_blk == mrg_blk) continue;
            if (MLIR_GetBlockNumArgs(ctx, arm_blk) != 0) continue;
            if (MLIR_GetBlockNumArgs(ctx, mrg_blk) != 1) continue;
            // Lookup arm_blk pred count
            uintptr_t ak = (uintptr_t)arm_blk;
            size_t s = (ak * 1099511628211ull) & hmask;
            int32_t abi = -1;
            while (hk[s]) { if (hk[s] == ak) { abi = hv[s]; break; } s = (s + 1) & hmask; }
            if (abi <= 0 || pc[abi] != 1) continue;

            size_t ano = MLIR_GetBlockNumOps(ctx, arm_blk);
            if (ano < 2 || ano > 8) continue;
            MLIR_OpHandle aterm = MLIR_GetBlockOp(ctx, arm_blk, ano - 1);
            if (MLIR_GetOpType(ctx, aterm) != OP_TYPE_LLVM_BR ||
                MLIR_GetOpSuccessor(ctx, aterm, 0) != mrg_blk ||
                MLIR_GetOpNumSuccessorOperands(ctx, aterm, 0) != 1)
                continue;
            // Only convert integer/pointer selects (avoid float select).
            MLIR_ValueHandle mrg_arg = MLIR_GetBlockArg(ctx, mrg_blk, 0);
            MLIR_TypeHandle mrg_ty = MLIR_GetValueType(ctx, mrg_arg);
            MLIR_TypeKind mk = MLIR_GetTypeKind(ctx, mrg_ty);
            if (mk != MLIR_TYPE_KIND_INTEGER && mk != MLIR_TYPE_KIND_LLVM_POINTER) continue;

            bool all_cheap = true;
            int n_real = 0;
            for (size_t i = 0; i + 1 < ano; i++) {
                MLIR_OpHandle aop = MLIR_GetBlockOp(ctx, arm_blk, i);
                if (!ifcvt_is_cheap_pure(ctx, aop)) { all_cheap = false; break; }
                MLIR_OpType at = MLIR_GetOpType(ctx, aop);
                if (at != OP_TYPE_LLVM_MLIR_CONSTANT && at != OP_TYPE_LLVM_MLIR_ZERO &&
                    at != OP_TYPE_LLVM_MLIR_NULL)
                    n_real++;
            }
            if (!all_cheap || n_real == 0 || n_real > 3) continue;

            // Hoist arm_blk's ops into bk before term!
            size_t insert_pos = no - 1;
            for (size_t i = 0; i + 1 < ano; i++) {
                MLIR_OpHandle aop = MLIR_GetBlockOp(ctx, arm_blk, 0);
                MLIR_DetachOp(ctx, aop);
                MLIR_InsertBlockOpAtIndex(ctx, bk, aop, insert_pos++);
            }
            MLIR_ValueHandle cond    = MLIR_GetOpOperand(ctx, term, 0);
            MLIR_ValueHandle arm_val = MLIR_GetOpSuccessorOperand(ctx, aterm, 0, 0);
            MLIR_ValueHandle old_val = MLIR_GetOpSuccessorOperand(ctx, term, (size_t)(1 - arm), 0);
            MLIR_ValueHandle sel_ops[3];
            sel_ops[0] = cond;
            sel_ops[1] = (arm == 0) ? arm_val : old_val;
            sel_ops[2] = (arm == 0) ? old_val : arm_val;
            MLIR_OpSpec ssp;
            memset(&ssp, 0, sizeof(ssp));
            ssp.type = OP_TYPE_LLVM_SELECT;
            ssp.operands = sel_ops;
            ssp.n_operands = 3;
            ssp.result_types = &mrg_ty;
            ssp.n_results = 1;
            MLIR_OpHandle sel_op = MLIR_CreateOpFromSpec(ctx, &ssp);
            MLIR_InsertBlockOpAtIndex(ctx, bk, sel_op, insert_pos++);

            MLIR_ValueHandle sel_res = MLIR_GetOpResult(ctx, sel_op, 0);
            uint32_t one = 1;
            MLIR_OpSpec bsp;
            memset(&bsp, 0, sizeof(bsp));
            bsp.type = OP_TYPE_LLVM_BR;
            bsp.operands = &sel_res;
            bsp.n_operands = 1;
            bsp.successors = &mrg_blk;
            bsp.n_successors = 1;
            bsp.successor_operand_counts = &one;
            MLIR_OpHandle new_br = MLIR_CreateOpFromSpec(ctx, &bsp);
            MLIR_InsertBlockOpAtIndex(ctx, bk, new_br, insert_pos);
            MLIR_DetachOp(ctx, term);
            break;
        }

        // Also handle 2-arm symmetric diamonds (`if (c) { ... } else { ... }`)
        // merging into `mrg_blk` with 1 or 2 integer/pointer block args (e.g. crcu8!).
        if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, bk, MLIR_GetBlockNumOps(ctx, bk) - 1)) == OP_TYPE_LLVM_COND_BR &&
            nso0 == 0 && nso1 == 0 && s0 != bk && s1 != bk && s0 != s1 &&
            MLIR_GetBlockNumArgs(ctx, s0) == 0 && MLIR_GetBlockNumArgs(ctx, s1) == 0) {
            int32_t bi0 = -1, bi1 = -1;
            uintptr_t k0 = (uintptr_t)s0, k1 = (uintptr_t)s1;
            size_t st = (k0 * 1099511628211ull) & hmask;
            while (hk[st]) { if (hk[st] == k0) { bi0 = hv[st]; break; } st = (st + 1) & hmask; }
            st = (k1 * 1099511628211ull) & hmask;
            while (hk[st]) { if (hk[st] == k1) { bi1 = hv[st]; break; } st = (st + 1) & hmask; }
            if (bi0 > 0 && bi1 > 0 && pc[bi0] == 1 && pc[bi1] == 1) {
                size_t no0 = MLIR_GetBlockNumOps(ctx, s0), no1 = MLIR_GetBlockNumOps(ctx, s1);
                if (no0 >= 1 && no0 <= 8 && no1 >= 1 && no1 <= 8) {
                    MLIR_OpHandle t0 = MLIR_GetBlockOp(ctx, s0, no0 - 1);
                    MLIR_OpHandle t1 = MLIR_GetBlockOp(ctx, s1, no1 - 1);
                    if (MLIR_GetOpType(ctx, t0) == OP_TYPE_LLVM_BR &&
                        MLIR_GetOpType(ctx, t1) == OP_TYPE_LLVM_BR) {
                        MLIR_BlockHandle m0 = MLIR_GetOpSuccessor(ctx, t0, 0);
                        MLIR_BlockHandle m1 = MLIR_GetOpSuccessor(ctx, t1, 0);
                        size_t mn = MLIR_GetOpNumSuccessorOperands(ctx, t0, 0);
                        if (m0 == m1 && mn >= 1 && mn <= 2 &&
                            MLIR_GetOpNumSuccessorOperands(ctx, t1, 0) == mn &&
                            MLIR_GetBlockNumArgs(ctx, m0) == mn) {
                            bool ok_types = true;
                            for (size_t a = 0; a < mn; a++) {
                                MLIR_TypeKind mk = MLIR_GetTypeKind(ctx, MLIR_GetValueType(ctx, MLIR_GetBlockArg(ctx, m0, a)));
                                if (mk != MLIR_TYPE_KIND_INTEGER && mk != MLIR_TYPE_KIND_LLVM_POINTER) { ok_types = false; break; }
                            }
                            bool ok_ops = ok_types;
                            int n_real = 0;
                            for (size_t i = 0; i + 1 < no0 && ok_ops; i++) {
                                MLIR_OpHandle aop = MLIR_GetBlockOp(ctx, s0, i);
                                if (!ifcvt_is_cheap_pure(ctx, aop)) { ok_ops = false; break; }
                                MLIR_OpType at = MLIR_GetOpType(ctx, aop);
                                if (at != OP_TYPE_LLVM_MLIR_CONSTANT && at != OP_TYPE_LLVM_MLIR_ZERO && at != OP_TYPE_LLVM_MLIR_NULL) n_real++;
                            }
                            for (size_t i = 0; i + 1 < no1 && ok_ops; i++) {
                                MLIR_OpHandle aop = MLIR_GetBlockOp(ctx, s1, i);
                                if (!ifcvt_is_cheap_pure(ctx, aop)) { ok_ops = false; break; }
                                MLIR_OpType at = MLIR_GetOpType(ctx, aop);
                                if (at != OP_TYPE_LLVM_MLIR_CONSTANT && at != OP_TYPE_LLVM_MLIR_ZERO && at != OP_TYPE_LLVM_MLIR_NULL) n_real++;
                            }
                            if (ok_ops && n_real <= 5) {
                                size_t insert_pos = MLIR_GetBlockNumOps(ctx, bk) - 1;
                                for (size_t i = 0; i + 1 < no0; i++) {
                                    MLIR_OpHandle aop = MLIR_GetBlockOp(ctx, s0, 0);
                                    MLIR_DetachOp(ctx, aop);
                                    MLIR_InsertBlockOpAtIndex(ctx, bk, aop, insert_pos++);
                                }
                                for (size_t i = 0; i + 1 < no1; i++) {
                                    MLIR_OpHandle aop = MLIR_GetBlockOp(ctx, s1, 0);
                                    MLIR_DetachOp(ctx, aop);
                                    MLIR_InsertBlockOpAtIndex(ctx, bk, aop, insert_pos++);
                                }
                                MLIR_ValueHandle cond = MLIR_GetOpOperand(ctx, term, 0);
                                MLIR_ValueHandle br_args[2];
                                for (size_t a = 0; a < mn; a++) {
                                    MLIR_ValueHandle v0 = MLIR_GetOpSuccessorOperand(ctx, t0, 0, a);
                                    MLIR_ValueHandle v1 = MLIR_GetOpSuccessorOperand(ctx, t1, 0, a);
                                    if (v0 == v1) {
                                        br_args[a] = v0;
                                    } else {
                                        MLIR_TypeHandle rty = MLIR_GetValueType(ctx, MLIR_GetBlockArg(ctx, m0, a));
                                        MLIR_ValueHandle sops[3] = { cond, v0, v1 };
                                        MLIR_OpSpec ssp;
                                        memset(&ssp, 0, sizeof(ssp));
                                        ssp.type = OP_TYPE_LLVM_SELECT;
                                        ssp.operands = sops;
                                        ssp.n_operands = 3;
                                        ssp.result_types = &rty;
                                        ssp.n_results = 1;
                                        MLIR_OpHandle sop = MLIR_CreateOpFromSpec(ctx, &ssp);
                                        MLIR_InsertBlockOpAtIndex(ctx, bk, sop, insert_pos++);
                                        br_args[a] = MLIR_GetOpResult(ctx, sop, 0);
                                    }
                                }
                                uint32_t mn32 = (uint32_t)mn;
                                MLIR_OpSpec bsp;
                                memset(&bsp, 0, sizeof(bsp));
                                bsp.type = OP_TYPE_LLVM_BR;
                                bsp.operands = br_args;
                                bsp.n_operands = mn;
                                bsp.successors = &m0;
                                bsp.n_successors = 1;
                                bsp.successor_operand_counts = &mn32;
                                MLIR_OpHandle new_br = MLIR_CreateOpFromSpec(ctx, &bsp);
                                MLIR_InsertBlockOpAtIndex(ctx, bk, new_br, insert_pos);
                                MLIR_DetachOp(ctx, term);
                            }
                        }
                    }
                }
            }
        }
    }
    free(hk); free(hv); free(pc);
}

// Unwrap redundant `icmp ne(select(c1, 1, 0), 0)` / `icmp ne(zext(c1:i1), 0)` -> `c1`.
// emit.c wraps C comparisons in select(1,0) for int-typed expressions; unwrapping
// them lets if-conversion and select lowering use `c1` directly without extra cmovs.
static void unwrap_bool_selects_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_ICMP) continue;
            MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_PREDICATE);
            if (!pa || MLIR_GetAttributeInteger(ctx, pa) != 1) continue; // ne
            MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
            MLIR_OpHandle d1 = MLIR_GetValueDefiningOp(ctx, a1);
            if (d1 == MLIR_INVALID_HANDLE) continue;
            MLIR_OpType dt1 = MLIR_GetOpType(ctx, d1);
            int64_t c1_val = -1;
            if (dt1 == OP_TYPE_LLVM_MLIR_ZERO) c1_val = 0;
            else if (dt1 == OP_TYPE_LLVM_MLIR_CONSTANT) {
                MLIR_AttributeHandle va = MLIR_GetOpAttr(ctx, d1, MLIR_KEY_VALUE);
                if (va != MLIR_INVALID_HANDLE) c1_val = MLIR_GetAttributeInteger(ctx, va);
            }
            if (c1_val != 0) continue;
            MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
            MLIR_OpHandle d0 = MLIR_GetValueDefiningOp(ctx, a0);
            if (d0 == MLIR_INVALID_HANDLE) continue;
            MLIR_OpType dt0 = MLIR_GetOpType(ctx, d0);
            MLIR_ValueHandle inner = MLIR_INVALID_HANDLE;
            if (dt0 == OP_TYPE_LLVM_SELECT) {
                MLIR_ValueHandle st = MLIR_GetOpOperand(ctx, d0, 1);
                MLIR_ValueHandle sf = MLIR_GetOpOperand(ctx, d0, 2);
                MLIR_OpHandle dst = MLIR_GetValueDefiningOp(ctx, st);
                MLIR_OpHandle dsf = MLIR_GetValueDefiningOp(ctx, sf);
                if (dst != MLIR_INVALID_HANDLE && dsf != MLIR_INVALID_HANDLE &&
                    MLIR_GetOpType(ctx, dst) == OP_TYPE_LLVM_MLIR_CONSTANT &&
                    (MLIR_GetOpType(ctx, dsf) == OP_TYPE_LLVM_MLIR_CONSTANT ||
                     MLIR_GetOpType(ctx, dsf) == OP_TYPE_LLVM_MLIR_ZERO)) {
                    MLIR_AttributeHandle vat = MLIR_GetOpAttr(ctx, dst, MLIR_KEY_VALUE);
                    int64_t vt = vat ? MLIR_GetAttributeInteger(ctx, vat) : -1;
                    int64_t vf = 0;
                    if (MLIR_GetOpType(ctx, dsf) == OP_TYPE_LLVM_MLIR_CONSTANT) {
                        MLIR_AttributeHandle vaf = MLIR_GetOpAttr(ctx, dsf, MLIR_KEY_VALUE);
                        vf = vaf ? MLIR_GetAttributeInteger(ctx, vaf) : -1;
                    }
                    if (vt == 1 && vf == 0) inner = MLIR_GetOpOperand(ctx, d0, 0);
                }
            } else if (dt0 == OP_TYPE_LLVM_ZEXT) {
                MLIR_ValueHandle z0 = MLIR_GetOpOperand(ctx, d0, 0);
                MLIR_TypeHandle zt = MLIR_GetValueType(ctx, z0);
                if (MLIR_GetTypeKind(ctx, zt) == MLIR_TYPE_KIND_INTEGER &&
                    MLIR_GetTypeWidth(ctx, zt) == 1)
                    inner = z0;
            }
            if (inner != MLIR_INVALID_HANDLE) {
                MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetOpResult(ctx, op, 0), inner);
            }
        }
    }
}

void mlir_llvm_simplify_cfg(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_SIMPLIFY_CFG")) return;
    if (MLIR_GetOpNumRegions(ctx, module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        unwrap_bool_selects_func(ctx, fn);
        if_convert_func(ctx, fn);
        unwrap_bool_selects_func(ctx, fn);
        simplify_cfg_func(ctx, fn);
    }
}

// ---------------------------------------------------------------------------
// Constant folding, algebraic identities, and constant-branch folding.
// ---------------------------------------------------------------------------
static bool cf_get_int_const(MLIR_Context *ctx, MLIR_ValueHandle v, int64_t *out, uint32_t *out_w) {
    MLIR_TypeHandle ty = MLIR_GetValueType(ctx, v);
    if (MLIR_GetTypeKind(ctx, ty) != MLIR_TYPE_KIND_INTEGER) return false;
    uint32_t w = MLIR_GetTypeWidth(ctx, ty);
    if (w == 0 || w > 64) return false;
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def == MLIR_INVALID_HANDLE) return false;
    MLIR_OpType dt = MLIR_GetOpType(ctx, def);
    int64_t val = 0;
    if (dt == OP_TYPE_LLVM_MLIR_ZERO) {
        val = 0;
    } else if (dt == OP_TYPE_LLVM_MLIR_CONSTANT) {
        MLIR_AttributeHandle va = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
        if (va == MLIR_INVALID_HANDLE) return false;
        val = MLIR_GetAttributeInteger(ctx, va);
    } else {
        return false;
    }
    if (w < 64) val = (int64_t)((uint64_t)val & ((1ull << w) - 1ull));
    *out = val;
    if (out_w) *out_w = w;
    return true;
}

static inline int64_t cf_sext(int64_t v, uint32_t w) {
    if (w >= 64) return v;
    uint64_t mask = (1ull << w) - 1ull;
    uint64_t sign = 1ull << (w - 1);
    return (int64_t)((((uint64_t)v & mask) ^ sign) - sign);
}

static inline int64_t cf_mask(int64_t v, uint32_t w) {
    if (w >= 64) return v;
    return (int64_t)((uint64_t)v & ((1ull << w) - 1ull));
}

static MLIR_ValueHandle cf_make_int_const(MLIR_Context *ctx, MLIR_BlockHandle bk,
                                          size_t idx, MLIR_TypeHandle ty, int64_t val) {
    MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeInteger(ctx, val, ty));
    MLIR_OpSpec sp;
    memset(&sp, 0, sizeof(sp));
    sp.type = OP_TYPE_LLVM_MLIR_CONSTANT;
    sp.result_types = &ty;
    sp.n_results = 1;
    sp.attrs = &attr;
    sp.n_attrs = 1;
    MLIR_OpHandle c = MLIR_CreateOpFromSpec(ctx, &sp);
    MLIR_InsertBlockOpAtIndex(ctx, bk, c, idx);
    return MLIR_GetOpResult(ctx, c, 0);
}

static MLIR_ValueHandle cf_unwrap_bool_i1(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(ctx, v);
    if (d == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    MLIR_OpType dt = MLIR_GetOpType(ctx, d);
    if (dt == OP_TYPE_LLVM_ZEXT) {
        MLIR_ValueHandle bv = MLIR_GetOpOperand(ctx, d, 0);
        MLIR_TypeHandle bty = MLIR_GetValueType(ctx, bv);
        if (MLIR_GetTypeKind(ctx, bty) == MLIR_TYPE_KIND_INTEGER &&
            MLIR_GetTypeWidth(ctx, bty) == 1)
            return bv;
    } else if (dt == OP_TYPE_LLVM_SELECT && MLIR_GetOpNumOperands(ctx, d) == 3) {
        int64_t vt = 0, vf = 0;
        if (cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, d, 1), &vt, NULL) && vt == 1 &&
            cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, d, 2), &vf, NULL) && vf == 0) {
            MLIR_ValueHandle bv = MLIR_GetOpOperand(ctx, d, 0);
            MLIR_TypeHandle bty = MLIR_GetValueType(ctx, bv);
            if (MLIR_GetTypeKind(ctx, bty) == MLIR_TYPE_KIND_INTEGER &&
                MLIR_GetTypeWidth(ctx, bty) == 1)
                return bv;
        }
    }
    return MLIR_INVALID_HANDLE;
}

static void constfold_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    for (int round = 0; round < 4; round++) {
        bool changed = false;
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t i = 0;
            while (i < MLIR_GetBlockNumOps(ctx, bk)) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
                MLIR_OpType ot = MLIR_GetOpType(ctx, op);

                // Constant-condition branch folding: cond_br(const) -> br
                if (ot == OP_TYPE_LLVM_COND_BR) {
                    int64_t cc = 0;
                    if (cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, op, 0), &cc, NULL)) {
                        size_t arm = (cc != 0) ? 0 : 1;
                        MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(ctx, op, arm);
                        size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, arm);
                        if (nso <= 64) {
                            MLIR_ValueHandle sops[64];
                            for (size_t k = 0; k < nso; k++)
                                sops[k] = MLIR_GetOpSuccessorOperand(ctx, op, arm, k);
                            uint32_t nso32 = (uint32_t)nso;
                            MLIR_OpSpec bsp;
                            memset(&bsp, 0, sizeof(bsp));
                            bsp.type = OP_TYPE_LLVM_BR;
                            bsp.operands = sops;
                            bsp.n_operands = nso;
                            bsp.successors = &tgt;
                            bsp.n_successors = 1;
                            bsp.successor_operand_counts = &nso32;
                            MLIR_OpHandle nbr = MLIR_CreateOpFromSpec(ctx, &bsp);
                            MLIR_InsertBlockOpAtIndex(ctx, bk, nbr, i);
                            MLIR_DetachOp(ctx, op);
                            changed = true;
                            continue;
                        }
                    }
                    i++;
                    continue;
                }

                if (MLIR_GetOpNumResults(ctx, op) != 1) { i++; continue; }
                MLIR_ValueHandle res = MLIR_GetOpResult(ctx, op, 0);
                MLIR_TypeHandle rty = MLIR_GetValueType(ctx, res);
                MLIR_TypeKind rk = MLIR_GetTypeKind(ctx, rty);

                // Casts of integer constants: sext / zext / trunc / sitofp / uitofp
                if (ot == OP_TYPE_LLVM_SEXT || ot == OP_TYPE_LLVM_ZEXT || ot == OP_TYPE_LLVM_TRUNC) {
                    int64_t c0 = 0; uint32_t sw = 0;
                    MLIR_ValueHandle src = MLIR_GetOpOperand(ctx, op, 0);
                    if (rk == MLIR_TYPE_KIND_INTEGER &&
                        cf_get_int_const(ctx, src, &c0, &sw)) {
                        uint32_t rw = MLIR_GetTypeWidth(ctx, rty);
                        int64_t rv = (ot == OP_TYPE_LLVM_SEXT) ? cf_mask(cf_sext(c0, sw), rw)
                                                               : cf_mask(c0, rw);
                        MLIR_ValueHandle nv = cf_make_int_const(ctx, bk, i, rty, rv);
                        MLIR_ReplaceAllUsesOfValue(ctx, res, nv);
                        MLIR_DetachOp(ctx, op);
                        changed = true;
                        continue;
                    }
                    // sext/zext (bool -> iN) -> zext (i1 -> iM)
                    if ((ot == OP_TYPE_LLVM_SEXT || ot == OP_TYPE_LLVM_ZEXT) &&
                        rk == MLIR_TYPE_KIND_INTEGER) {
                        MLIR_ValueHandle b1 = cf_unwrap_bool_i1(ctx, src);
                        if (b1 != MLIR_INVALID_HANDLE) {
                            MLIR_OpSpec sp;
                            memset(&sp, 0, sizeof(sp));
                            sp.type = OP_TYPE_LLVM_ZEXT;
                            sp.operands = &b1;
                            sp.n_operands = 1;
                            sp.result_types = &rty;
                            sp.n_results = 1;
                            MLIR_OpHandle nz = MLIR_CreateOpFromSpec(ctx, &sp);
                            MLIR_InsertBlockOpAtIndex(ctx, bk, nz, i);
                            MLIR_ReplaceAllUsesOfValue(ctx, res, MLIR_GetOpResult(ctx, nz, 0));
                            MLIR_DetachOp(ctx, op);
                            changed = true;
                            continue;
                        }
                    }
                    i++; continue;
                }
                if (ot == OP_TYPE_LLVM_SITOFP || ot == OP_TYPE_LLVM_UITOFP) {
                    int64_t c0 = 0; uint32_t sw = 0;
                    if (rk == MLIR_TYPE_KIND_FLOAT &&
                        cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, op, 0), &c0, &sw)) {
                        double dv = (ot == OP_TYPE_LLVM_SITOFP)
                            ? (double)cf_sext(c0, sw)
                            : (double)(uint64_t)cf_mask(c0, sw);
                        MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeFloat(ctx, dv, rty));
                        MLIR_OpSpec sp;
                        memset(&sp, 0, sizeof(sp));
                        sp.type = OP_TYPE_LLVM_MLIR_CONSTANT;
                        sp.result_types = &rty;
                        sp.n_results = 1;
                        sp.attrs = &attr;
                        sp.n_attrs = 1;
                        MLIR_OpHandle c = MLIR_CreateOpFromSpec(ctx, &sp);
                        MLIR_InsertBlockOpAtIndex(ctx, bk, c, i);
                        MLIR_ReplaceAllUsesOfValue(ctx, res, MLIR_GetOpResult(ctx, c, 0));
                        MLIR_DetachOp(ctx, op);
                        changed = true;
                        continue;
                    }
                    i++; continue;
                }

                // Post-increment `gep(p, const) ; load(p)` -> `load(p) ; gep(p, const)`
                // so `p` dies at the gep and can coalesce with `p_next` (e.g. `*(pc++)`).
                if (ot == OP_TYPE_LLVM_GEP && i + 1 < MLIR_GetBlockNumOps(ctx, bk)) {
                    MLIR_OpHandle nxt = MLIR_GetBlockOp(ctx, bk, i + 1);
                    if (MLIR_GetOpType(ctx, nxt) == OP_TYPE_LLVM_LOAD &&
                        MLIR_GetOpOperand(ctx, nxt, 0) == MLIR_GetOpOperand(ctx, op, 0)) {
                        MLIR_DetachOp(ctx, nxt);
                        MLIR_InsertBlockOpAtIndex(ctx, bk, nxt, i);
                        i += 2;
                        continue;
                    }
                }

                // GEP `trunc i64 X to i32` -> `i64 X` when X is a non-negative i64
                // loop IV bounded by `<= INT32_MAX` (e.g. sieve inner loop).
                if (ot == OP_TYPE_LLVM_GEP && MLIR_GetOpNumOperands(ctx, op) == 2) {
                    MLIR_ValueHandle iv = MLIR_GetOpOperand(ctx, op, 1);
                    MLIR_OpHandle tr = MLIR_GetValueDefiningOp(ctx, iv);
                    if (tr != MLIR_INVALID_HANDLE && MLIR_GetOpType(ctx, tr) == OP_TYPE_LLVM_TRUNC &&
                        MLIR_GetTypeKind(ctx, MLIR_GetValueType(ctx, iv)) == MLIR_TYPE_KIND_INTEGER &&
                        MLIR_GetTypeWidth(ctx, MLIR_GetValueType(ctx, iv)) == 32) {
                        MLIR_ValueHandle xv = MLIR_GetOpOperand(ctx, tr, 0);
                        MLIR_TypeHandle xty = MLIR_GetValueType(ctx, xv);
                        if (MLIR_GetTypeKind(ctx, xty) == MLIR_TYPE_KIND_INTEGER &&
                            MLIR_GetTypeWidth(ctx, xty) == 64 &&
                            MLIR_GetValueDefiningOp(ctx, xv) == MLIR_INVALID_HANDLE) {
                            // Verify `xv` is a block arg of a single-predecessor header `hdr`
                            // that branches to `bk` on `icmp sle/slt xv, C` (0 <= C <= INT32_MAX)
                            // with non-negative init and non-negative step.
                            for (size_t hb = 0; hb < nb; hb++) {
                                MLIR_BlockHandle hbk = MLIR_GetRegionBlock(ctx, reg, hb);
                                size_t hna = MLIR_GetBlockNumArgs(ctx, hbk);
                                int arg_k = -1;
                                for (size_t a = 0; a < hna; a++)
                                    if (MLIR_GetBlockArg(ctx, hbk, a) == xv) { arg_k = (int)a; break; }
                                if (arg_k < 0) continue;
                                size_t hno = MLIR_GetBlockNumOps(ctx, hbk);
                                if (hno < 2) break;
                                MLIR_OpHandle hterm = MLIR_GetBlockOp(ctx, hbk, hno - 1);
                                if (MLIR_GetOpType(ctx, hterm) != OP_TYPE_LLVM_COND_BR ||
                                    MLIR_GetOpSuccessor(ctx, hterm, 0) != bk) break;
                                MLIR_OpHandle hicmp = MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, hterm, 0));
                                if (hicmp == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, hicmp) != OP_TYPE_LLVM_ICMP) break;
                                MLIR_AttributeHandle hpa = MLIR_GetOpAttr(ctx, hicmp, MLIR_KEY_PREDICATE);
                                int64_t hpred = hpa ? MLIR_GetAttributeInteger(ctx, hpa) : -1;
                                int64_t lim = 0;
                                if ((hpred != 2 && hpred != 3) ||
                                    MLIR_GetOpOperand(ctx, hicmp, 0) != xv ||
                                    !cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, hicmp, 1), &lim, NULL) ||
                                    lim < 0 || lim > 0x7fffffffll) break;
                                // Check all incoming values to (hbk, arg_k) are non-negative or `add(xv, nonneg)`.
                                bool all_ok = true;
                                for (size_t pb = 0; pb < nb && all_ok; pb++) {
                                    MLIR_BlockHandle pbk = MLIR_GetRegionBlock(ctx, reg, pb);
                                    size_t pno = MLIR_GetBlockNumOps(ctx, pbk);
                                    if (pno == 0) continue;
                                    MLIR_OpHandle pterm = MLIR_GetBlockOp(ctx, pbk, pno - 1);
                                    size_t pns = MLIR_GetOpNumSuccessors(ctx, pterm);
                                    for (size_t ps = 0; ps < pns && all_ok; ps++) {
                                        if (MLIR_GetOpSuccessor(ctx, pterm, ps) != hbk) continue;
                                        if ((size_t)arg_k >= MLIR_GetOpNumSuccessorOperands(ctx, pterm, ps)) { all_ok = false; break; }
                                        MLIR_ValueHandle inc = MLIR_GetOpSuccessorOperand(ctx, pterm, ps, (size_t)arg_k);
                                        int64_t ic = 0;
                                        if (cf_get_int_const(ctx, inc, &ic, NULL)) {
                                            if (ic < 0 || ic > 0x7fffffffll) all_ok = false;
                                            continue;
                                        }
                                        MLIR_OpHandle id = MLIR_GetValueDefiningOp(ctx, inc);
                                        if (id == MLIR_INVALID_HANDLE) { all_ok = false; break; }
                                        MLIR_OpType idt = MLIR_GetOpType(ctx, id);
                                        if (idt == OP_TYPE_LLVM_MUL &&
                                            MLIR_GetOpOperand(ctx, id, 0) == MLIR_GetOpOperand(ctx, id, 1)) {
                                            // x*x >= 0 (provided no i64 overflow; sext(i32)^2 <= 2^62,
                                            // or widened i64 loop IV).
                                            MLIR_OpHandle sd = MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, id, 0));
                                            if (sd != MLIR_INVALID_HANDLE &&
                                                (MLIR_GetOpType(ctx, sd) != OP_TYPE_LLVM_SEXT ||
                                                 MLIR_GetTypeWidth(ctx, MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, sd, 0))) > 32))
                                                all_ok = false;
                                        } else if (idt == OP_TYPE_LLVM_ADD && MLIR_GetOpOperand(ctx, id, 0) == xv) {
                                            MLIR_ValueHandle step = MLIR_GetOpOperand(ctx, id, 1);
                                            int64_t sc = 0;
                                            if (cf_get_int_const(ctx, step, &sc, NULL)) {
                                                if (sc < 0 || sc > 0x7fffffffll) all_ok = false;
                                            } else {
                                                MLIR_OpHandle sd = MLIR_GetValueDefiningOp(ctx, step);
                                                if (sd != MLIR_INVALID_HANDLE &&
                                                    (MLIR_GetOpType(ctx, sd) != OP_TYPE_LLVM_SEXT ||
                                                     MLIR_GetTypeWidth(ctx, MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, sd, 0))) > 32))
                                                    all_ok = false;
                                            }
                                        } else {
                                            all_ok = false;
                                        }
                                    }
                                }
                                if (all_ok) {
                                    MLIR_SetOpOperand(ctx, op, 1, xv);
                                    changed = true;
                                }
                                break;
                            }
                        }
                    }
                }

                // Select folding
                if (ot == OP_TYPE_LLVM_SELECT) {
                    MLIR_ValueHandle vc = MLIR_GetOpOperand(ctx, op, 0);
                    MLIR_ValueHandle vt = MLIR_GetOpOperand(ctx, op, 1);
                    MLIR_ValueHandle vf = MLIR_GetOpOperand(ctx, op, 2);
                    int64_t cc = 0;
                    if (vt == vf) {
                        MLIR_ReplaceAllUsesOfValue(ctx, res, vt);
                        MLIR_DetachOp(ctx, op);
                        changed = true; continue;
                    }
                    if (cf_get_int_const(ctx, vc, &cc, NULL)) {
                        MLIR_ReplaceAllUsesOfValue(ctx, res, cc != 0 ? vt : vf);
                        MLIR_DetachOp(ctx, op);
                        changed = true; continue;
                    }
                    i++; continue;
                }

                // Integer binary ops + icmp
                if (MLIR_GetOpNumOperands(ctx, op) == 2 && rk == MLIR_TYPE_KIND_INTEGER) {
                    MLIR_ValueHandle a0 = MLIR_GetOpOperand(ctx, op, 0);
                    MLIR_ValueHandle a1 = MLIR_GetOpOperand(ctx, op, 1);
                    int64_t c0 = 0, c1 = 0;
                    uint32_t w0 = 0, w1 = 0;
                    bool k0 = cf_get_int_const(ctx, a0, &c0, &w0);
                    bool k1 = cf_get_int_const(ctx, a1, &c1, &w1);
                    uint32_t rw = MLIR_GetTypeWidth(ctx, rty);

                    if (ot == OP_TYPE_LLVM_ICMP && k0 && k1) {
                        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_PREDICATE);
                        int64_t pred = pa ? MLIR_GetAttributeInteger(ctx, pa) : 0;
                        int64_t s0 = cf_sext(c0, w0), s1 = cf_sext(c1, w0);
                        uint64_t u0 = (uint64_t)cf_mask(c0, w0), u1 = (uint64_t)cf_mask(c1, w0);
                        int64_t rv = 0;
                        switch (pred) {
                        case 0: rv = (u0 == u1); break;
                        case 1: rv = (u0 != u1); break;
                        case 2: rv = (s0 <  s1); break;
                        case 3: rv = (s0 <= s1); break;
                        case 4: rv = (s0 >  s1); break;
                        case 5: rv = (s0 >= s1); break;
                        case 6: rv = (u0 <  u1); break;
                        case 7: rv = (u0 <= u1); break;
                        case 8: rv = (u0 >  u1); break;
                        case 9: rv = (u0 >= u1); break;
                        default: break;
                        }
                        MLIR_ValueHandle nv = cf_make_int_const(ctx, bk, i, rty, rv);
                        MLIR_ReplaceAllUsesOfValue(ctx, res, nv);
                        MLIR_DetachOp(ctx, op);
                        changed = true; continue;
                    }

                    // Both constant -> fold!
                    if (k0 && k1 && rw > 0 && rw <= 64) {
                        uint64_t u0 = (uint64_t)cf_mask(c0, rw), u1 = (uint64_t)cf_mask(c1, rw);
                        int64_t s0 = cf_sext(c0, rw), s1 = cf_sext(c1, rw);
                        bool folded = true;
                        uint64_t ur = 0;
                        switch (ot) {
                        case OP_TYPE_LLVM_ADD:  ur = u0 + u1; break;
                        case OP_TYPE_LLVM_SUB:  ur = u0 - u1; break;
                        case OP_TYPE_LLVM_MUL:  ur = u0 * u1; break;
                        case OP_TYPE_LLVM_AND:  ur = u0 & u1; break;
                        case OP_TYPE_LLVM_OR:   ur = u0 | u1; break;
                        case OP_TYPE_LLVM_XOR:  ur = u0 ^ u1; break;
                        case OP_TYPE_LLVM_SHL:  ur = (u1 < rw) ? (u0 << u1) : 0; break;
                        case OP_TYPE_LLVM_LSHR: ur = (u1 < rw) ? (u0 >> u1) : 0; break;
                        case OP_TYPE_LLVM_ASHR: ur = (u1 < rw) ? (uint64_t)(s0 >> u1) : (s0 < 0 ? ~(uint64_t)0 : 0); break;
                        case OP_TYPE_LLVM_UDIV: if (u1 != 0) ur = u0 / u1; else folded = false; break;
                        case OP_TYPE_LLVM_UREM: if (u1 != 0) ur = u0 % u1; else folded = false; break;
                        case OP_TYPE_LLVM_SDIV:
                            if (s1 != 0 && !(rw == 64 && s0 == INT64_MIN && s1 == -1))
                                ur = (uint64_t)(s0 / s1);
                            else folded = false;
                            break;
                        case OP_TYPE_LLVM_SREM:
                            if (s1 != 0 && !(rw == 64 && s0 == INT64_MIN && s1 == -1))
                                ur = (uint64_t)(s0 % s1);
                            else folded = false;
                            break;
                        default: folded = false; break;
                        }
                        if (folded) {
                            MLIR_ValueHandle nv = cf_make_int_const(ctx, bk, i, rty, cf_mask((int64_t)ur, rw));
                            MLIR_ReplaceAllUsesOfValue(ctx, res, nv);
                            MLIR_DetachOp(ctx, op);
                            changed = true; continue;
                        }
                    }

                    // Algebraic identities (one operand constant or a0 == a1)
                    MLIR_ValueHandle repl = MLIR_INVALID_HANDLE;
                    bool repl_zero = false;
                    if (k1 && c1 == 0) {
                        if (ot == OP_TYPE_LLVM_ADD || ot == OP_TYPE_LLVM_SUB ||
                            ot == OP_TYPE_LLVM_OR  || ot == OP_TYPE_LLVM_XOR ||
                            ot == OP_TYPE_LLVM_SHL || ot == OP_TYPE_LLVM_LSHR || ot == OP_TYPE_LLVM_ASHR)
                            repl = a0;
                        else if (ot == OP_TYPE_LLVM_MUL || ot == OP_TYPE_LLVM_AND)
                            repl_zero = true;
                    } else if (k0 && c0 == 0) {
                        if (ot == OP_TYPE_LLVM_ADD || ot == OP_TYPE_LLVM_OR || ot == OP_TYPE_LLVM_XOR)
                            repl = a1;
                        else if (ot == OP_TYPE_LLVM_MUL || ot == OP_TYPE_LLVM_AND)
                            repl_zero = true;
                    } else if (k1 && c1 == 1 && (ot == OP_TYPE_LLVM_MUL || ot == OP_TYPE_LLVM_SDIV || ot == OP_TYPE_LLVM_UDIV)) {
                        repl = a0;
                    } else if (k0 && c0 == 1 && ot == OP_TYPE_LLVM_MUL) {
                        repl = a1;
                    } else if (a0 == a1) {
                        if (ot == OP_TYPE_LLVM_SUB || ot == OP_TYPE_LLVM_XOR) repl_zero = true;
                        else if (ot == OP_TYPE_LLVM_AND || ot == OP_TYPE_LLVM_OR) repl = a0;
                    }
                    // Redundant mask elimination:
                    //   and(zext(iW), C) / and(select(i1, 1, 0), C) -> a0 when C covers all low bits
                    //   and(and(x, C1), C2) -> and(x, C1) when (C1 & C2) == C1
                    if (ot == OP_TYPE_LLVM_AND && k1 && rw > 0 && rw <= 64) {
                        uint64_t m1 = (uint64_t)cf_mask(c1, rw);
                        if ((m1 & 1ull) && cf_unwrap_bool_i1(ctx, a0) != MLIR_INVALID_HANDLE) {
                            repl = a0;
                        } else {
                            MLIR_OpHandle d0 = MLIR_GetValueDefiningOp(ctx, a0);
                            if (d0 != MLIR_INVALID_HANDLE) {
                                MLIR_OpType d0t = MLIR_GetOpType(ctx, d0);
                                if (d0t == OP_TYPE_LLVM_ZEXT) {
                                    MLIR_TypeHandle sty = MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, d0, 0));
                                    if (MLIR_GetTypeKind(ctx, sty) == MLIR_TYPE_KIND_INTEGER) {
                                        uint32_t sw = MLIR_GetTypeWidth(ctx, sty);
                                        if (sw > 0 && sw < rw) {
                                            uint64_t sm = (1ull << sw) - 1ull;
                                            if ((m1 & sm) == sm) repl = a0;
                                        }
                                    }
                                } else if (d0t == OP_TYPE_LLVM_AND && MLIR_GetOpNumOperands(ctx, d0) == 2) {
                                    int64_t ic = 0;
                                    if (cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, d0, 1), &ic, NULL)) {
                                        uint64_t m0 = (uint64_t)cf_mask(ic, rw);
                                        if ((m0 & m1) == m0) repl = a0;
                                    }
                                }
                            }
                        }
                    }

                    // Boolean round-trip elimination:
                    //   icmp ne (zext/select i1 %b to iW), 0 -> %b
                    //   icmp eq (zext/select i1 %b to iW), 1 -> %b
                    if (ot == OP_TYPE_LLVM_ICMP && k1 && rw == 1) {
                        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_PREDICATE);
                        int64_t pred = pa ? MLIR_GetAttributeInteger(ctx, pa) : -1;
                        if ((pred == 1 /*ne*/ && c1 == 0) || (pred == 0 /*eq*/ && c1 == 1)) {
                            MLIR_ValueHandle bv = cf_unwrap_bool_i1(ctx, a0);
                            if (bv != MLIR_INVALID_HANDLE) repl = bv;
                        }
                    }

                    // Range-check folding:
                    //   and(bool(icmp sge/uge x, LO), bool(icmp sle/ule x, HI))
                    //     -> zext(icmp ule (sub x, LO), (HI - LO))
                    //   (also handles direct i1 `and(icmp, icmp)` -> `icmp ule`).
                    // Turns e.g. CoreMark's `ee_isdigit(c)` from 15 insns into 3 insns!
                    if (ot == OP_TYPE_LLVM_AND && repl == MLIR_INVALID_HANDLE && !repl_zero &&
                        (rw == 1 || rw == 32 || rw == 64)) {
                        MLIR_OpHandle d0 = MLIR_INVALID_HANDLE, d1 = MLIR_INVALID_HANDLE;
                        bool wrap_zext = (rw > 1);
                        if (wrap_zext) {
                            MLIR_ValueHandle b0 = cf_unwrap_bool_i1(ctx, a0);
                            MLIR_ValueHandle b1 = cf_unwrap_bool_i1(ctx, a1);
                            if (b0 != MLIR_INVALID_HANDLE && b1 != MLIR_INVALID_HANDLE) {
                                d0 = MLIR_GetValueDefiningOp(ctx, b0);
                                d1 = MLIR_GetValueDefiningOp(ctx, b1);
                            }
                        } else {
                            d0 = MLIR_GetValueDefiningOp(ctx, a0);
                            d1 = MLIR_GetValueDefiningOp(ctx, a1);
                        }
                        if (d0 != MLIR_INVALID_HANDLE && d1 != MLIR_INVALID_HANDLE &&
                            MLIR_GetOpType(ctx, d0) == OP_TYPE_LLVM_ICMP &&
                            MLIR_GetOpType(ctx, d1) == OP_TYPE_LLVM_ICMP) {
                            MLIR_ValueHandle x0 = MLIR_GetOpOperand(ctx, d0, 0);
                            MLIR_ValueHandle x1 = MLIR_GetOpOperand(ctx, d1, 0);
                            int64_t k_lo = 0, k_hi = 0; uint32_t xw = 0;
                            if (x0 == x1 &&
                                cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, d0, 1), &k_lo, &xw) &&
                                cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, d1, 1), &k_hi, NULL) &&
                                (xw == 32 || xw == 64)) {
                                MLIR_AttributeHandle pa0 = MLIR_GetOpAttr(ctx, d0, MLIR_KEY_PREDICATE);
                                MLIR_AttributeHandle pa1 = MLIR_GetOpAttr(ctx, d1, MLIR_KEY_PREDICATE);
                                int64_t p0 = pa0 ? MLIR_GetAttributeInteger(ctx, pa0) : -1;
                                int64_t p1 = pa1 ? MLIR_GetAttributeInteger(ctx, pa1) : -1;
                                // Canonicalize so (p0, k_lo) is the lower bound and (p1, k_hi) is the upper bound.
                                if ((p0 == 3 || p0 == 7) && (p1 == 5 || p1 == 9)) {
                                    int64_t tp = p0; p0 = p1; p1 = tp;
                                    int64_t tk = k_lo; k_lo = k_hi; k_hi = tk;
                                }
                                bool is_signed = (p0 == 5 /*sge*/ && p1 == 3 /*sle*/);
                                bool is_unsigned = (p0 == 9 /*uge*/ && p1 == 7 /*ule*/);
                                bool range_ok = false;
                                uint64_t diff = 0;
                                if (is_signed) {
                                    int64_t slo = cf_sext(k_lo, xw), shi = cf_sext(k_hi, xw);
                                    if (slo <= shi) {
                                        diff = (uint64_t)shi - (uint64_t)slo;
                                        range_ok = (xw == 64) ? (diff < (1ull << 63)) : (diff < (1ull << 31));
                                    }
                                } else if (is_unsigned) {
                                    uint64_t ulo = (uint64_t)cf_mask(k_lo, xw), uhi = (uint64_t)cf_mask(k_hi, xw);
                                    if (ulo <= uhi) {
                                        diff = uhi - ulo;
                                        range_ok = true;
                                    }
                                }
                                if (range_ok) {
                                    MLIR_TypeHandle xty = MLIR_GetValueType(ctx, x0);
                                    MLIR_ValueHandle sub_v = x0;
                                    if (cf_mask(k_lo, xw) != 0) {
                                        MLIR_ValueHandle lo_c = cf_make_int_const(ctx, bk, i, xty, cf_mask(k_lo, xw));
                                        i++;
                                        MLIR_ValueHandle sops[2] = { x0, lo_c };
                                        MLIR_OpSpec ssp; memset(&ssp, 0, sizeof(ssp));
                                        ssp.type = OP_TYPE_LLVM_SUB;
                                        ssp.operands = sops; ssp.n_operands = 2;
                                        ssp.result_types = &xty; ssp.n_results = 1;
                                        MLIR_OpHandle sop = MLIR_CreateOpFromSpec(ctx, &ssp);
                                        MLIR_InsertBlockOpAtIndex(ctx, bk, sop, i);
                                        i++;
                                        sub_v = MLIR_GetOpResult(ctx, sop, 0);
                                    }
                                    MLIR_ValueHandle diff_c = cf_make_int_const(ctx, bk, i, xty, cf_mask((int64_t)diff, xw));
                                    i++;
                                    MLIR_TypeHandle i1ty = MLIR_CreateTypeInteger(ctx, 1, false);
                                    MLIR_TypeHandle i64ty = MLIR_CreateTypeInteger(ctx, 64, false);
                                    MLIR_NamedAttr pattr = MLIR_Named(MLIR_KEY_PREDICATE, MLIR_CreateAttributeInteger(ctx, 7 /*ule*/, i64ty));
                                    MLIR_ValueHandle cops[2] = { sub_v, diff_c };
                                    MLIR_OpSpec csp; memset(&csp, 0, sizeof(csp));
                                    csp.type = OP_TYPE_LLVM_ICMP;
                                    csp.operands = cops; csp.n_operands = 2;
                                    csp.result_types = &i1ty; csp.n_results = 1;
                                    csp.attrs = &pattr; csp.n_attrs = 1;
                                    MLIR_OpHandle cop = MLIR_CreateOpFromSpec(ctx, &csp);
                                    MLIR_InsertBlockOpAtIndex(ctx, bk, cop, i);
                                    i++;
                                    MLIR_ValueHandle cmp_v = MLIR_GetOpResult(ctx, cop, 0);
                                    MLIR_ValueHandle out_v = cmp_v;
                                    if (wrap_zext) {
                                        MLIR_OpSpec zsp; memset(&zsp, 0, sizeof(zsp));
                                        zsp.type = OP_TYPE_LLVM_ZEXT;
                                        zsp.operands = &cmp_v; zsp.n_operands = 1;
                                        zsp.result_types = &rty; zsp.n_results = 1;
                                        MLIR_OpHandle zop = MLIR_CreateOpFromSpec(ctx, &zsp);
                                        MLIR_InsertBlockOpAtIndex(ctx, bk, zop, i);
                                        i++;
                                        out_v = MLIR_GetOpResult(ctx, zop, 0);
                                    }
                                    MLIR_ReplaceAllUsesOfValue(ctx, res, out_v);
                                    MLIR_DetachOp(ctx, op);
                                    changed = true;
                                    continue;
                                }
                            }
                        }
                    }

                    if (repl_zero) {
                        MLIR_ValueHandle nv = cf_make_int_const(ctx, bk, i, rty, 0);
                        MLIR_ReplaceAllUsesOfValue(ctx, res, nv);
                        MLIR_DetachOp(ctx, op);
                        changed = true; continue;
                    }
                    if (repl != MLIR_INVALID_HANDLE) {
                        MLIR_ReplaceAllUsesOfValue(ctx, res, repl);
                        MLIR_DetachOp(ctx, op);
                        changed = true; continue;
                    }
                }
                i++;
            }
        }
        if (!changed) break;
    }
}

static MLIR_OpHandle unroll_clone_op(MLIR_Context *ctx, MLIR_OpHandle src,
                                     const MLIR_ValueHandle *old_v,
                                     const MLIR_ValueHandle *new_v, size_t n_map);

// Constant-arg exit threading (for small/medium functions, nb <= 64):
// If loop header H checks `cond_br(c0) -> (^chk1, ^exit(e_sops))`, `^chk1` checks
// `icmp eq/ne(H_arg_k, K) -> (^chk2, ^exit(e_sops))` with identical `^exit` + operands,
// `^exit` is a small call-free return block (<= 8 ops, no successors), and pure latch L
// (<= 4 pure ops, no calls/stores/loads/allocas) forwards `L_arg_j -> H_arg_k`, then
// any predecessor P of L passing constant C that triggers `^exit` can jump directly
// to a shared exit trampoline `^texit` that executes L's pure ops (e.g. `str + 1`)
// followed by `^exit`'s body with H's args mapped to L's outputs (so `str` -> `str + 1`
// and `state` -> `C`).
static void const_blockarg_thread_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 4 || nb > 64) return;

    MLIR_BlockHandle blks[64];
    for (size_t b = 0; b < nb; b++) blks[b] = MLIR_GetRegionBlock(ctx, reg, b);

    for (size_t h = 1; h < nb; h++) {
        MLIR_BlockHandle hblk = blks[h];
        size_t hna = MLIR_GetBlockNumArgs(ctx, hblk);
        if (hna == 0 || hna > 8) continue;
        MLIR_OpHandle hterm = MLIR_GetBlockTerminator(ctx, hblk);
        if (hterm == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, hterm) != OP_TYPE_LLVM_COND_BR) continue;

        for (size_t c1_arm = 0; c1_arm < 2; c1_arm++) {
            size_t ex_arm = 1 - c1_arm;
            MLIR_BlockHandle chk1 = MLIR_GetOpSuccessor(ctx, hterm, c1_arm);
            MLIR_BlockHandle exit_b = MLIR_GetOpSuccessor(ctx, hterm, ex_arm);
            if (chk1 == hblk || chk1 == exit_b || MLIR_GetBlockNumArgs(ctx, chk1) != 0) continue;
            if (MLIR_GetOpNumSuccessorOperands(ctx, hterm, c1_arm) != 0) continue;
            size_t n_ex = MLIR_GetOpNumSuccessorOperands(ctx, hterm, ex_arm);
            if (n_ex > 8 || n_ex != MLIR_GetBlockNumArgs(ctx, exit_b)) continue;

            // Verify `exit_b` is a small call-free return block (no successors).
            size_t ex_no = MLIR_GetBlockNumOps(ctx, exit_b);
            if (ex_no == 0 || ex_no > 8) continue;
            MLIR_OpHandle ex_term = MLIR_GetBlockOp(ctx, exit_b, ex_no - 1);
            MLIR_OpType ex_tt = MLIR_GetOpType(ctx, ex_term);
            if (ex_tt != OP_TYPE_LLVM_RETURN && ex_tt != OP_TYPE_FUNC_RETURN) continue;
            bool ex_ok = true;
            for (size_t i = 0; i < ex_no; i++) {
                MLIR_OpHandle eo = MLIR_GetBlockOp(ctx, exit_b, i);
                MLIR_OpType eot = MLIR_GetOpType(ctx, eo);
                if (MLIR_GetOpNumRegions(ctx, eo) > 0 || MLIR_GetOpNumOperands(ctx, eo) > 16 ||
                    MLIR_GetOpNumResults(ctx, eo) > 1 || MLIR_GetOpNumAttributes(ctx, eo) > 8 ||
                    eot == OP_TYPE_LLVM_CALL || eot == OP_TYPE_FUNC_CALL || eot == OP_TYPE_LLVM_ALLOCA) {
                    ex_ok = false; break;
                }
            }
            if (!ex_ok) continue;

            size_t c1no = MLIR_GetBlockNumOps(ctx, chk1);
            if (c1no < 2 || c1no > 4) continue;
            MLIR_OpHandle c1term = MLIR_GetBlockOp(ctx, chk1, c1no - 1);
            if (MLIR_GetOpType(ctx, c1term) != OP_TYPE_LLVM_COND_BR) continue;

            // Check if one arm of c1term goes to `exit_b` with identical operands.
            int c1_ex_arm = -1;
            for (size_t a = 0; a < 2; a++) {
                if (MLIR_GetOpSuccessor(ctx, c1term, a) == exit_b &&
                    MLIR_GetOpNumSuccessorOperands(ctx, c1term, a) == n_ex) {
                    bool same = true;
                    for (size_t k = 0; k < n_ex; k++) {
                        if (MLIR_GetOpSuccessorOperand(ctx, c1term, a, k) !=
                            MLIR_GetOpSuccessorOperand(ctx, hterm, ex_arm, k)) {
                            same = false; break;
                        }
                    }
                    if (same) { c1_ex_arm = (int)a; break; }
                }
            }
            if (c1_ex_arm < 0) continue;

            // Verify chk1 is pure icmp eq/ne(H_arg_k, K).
            MLIR_ValueHandle c1cond = MLIR_GetOpOperand(ctx, c1term, 0);
            MLIR_OpHandle icmp = MLIR_INVALID_HANDLE;
            bool c1_ok = true;
            for (size_t i = 0; i + 1 < c1no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, chk1, i);
                MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                if (ot == OP_TYPE_LLVM_MLIR_CONSTANT || ot == OP_TYPE_LLVM_MLIR_ZERO) continue;
                if (ot == OP_TYPE_LLVM_ICMP && icmp == MLIR_INVALID_HANDLE &&
                    MLIR_GetOpResult(ctx, op, 0) == c1cond)
                    icmp = op;
                else { c1_ok = false; break; }
            }
            if (!c1_ok || icmp == MLIR_INVALID_HANDLE) continue;
            MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, icmp, MLIR_KEY_PREDICATE);
            int64_t pred = pa ? MLIR_GetAttributeInteger(ctx, pa) : -1;
            if (pred != 0 && pred != 1) continue; // eq or ne

            MLIR_ValueHandle lhs = MLIR_GetOpOperand(ctx, icmp, 0);
            MLIR_ValueHandle rhs = MLIR_GetOpOperand(ctx, icmp, 1);
            int64_t K = 0; uint32_t Kw = 0;
            MLIR_ValueHandle arg_v = MLIR_INVALID_HANDLE;
            if (cf_get_int_const(ctx, rhs, &K, &Kw)) arg_v = lhs;
            else if (cf_get_int_const(ctx, lhs, &K, &Kw)) arg_v = rhs;
            else continue;
            size_t h_arg_k = hna;
            for (size_t a = 0; a < hna; a++)
                if (MLIR_GetBlockArg(ctx, hblk, a) == arg_v) { h_arg_k = a; break; }
            if (h_arg_k >= hna) continue;

            // `exit_b` is taken when `(arg_v == K)` if (pred==0 && c1_ex_arm==0) || (pred==1 && c1_ex_arm==1).
            bool exit_on_eq = ((pred == 0 && c1_ex_arm == 0) || (pred == 1 && c1_ex_arm == 1));
            if (!exit_on_eq) continue;

            // Ensure no op in `hblk` has side effects (only pure ops / non-volatile loads).
            bool h_pure = true;
            size_t hno = MLIR_GetBlockNumOps(ctx, hblk);
            for (size_t i = 0; i + 1 < hno; i++) {
                MLIR_OpType ot = MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, hblk, i));
                if (ot == OP_TYPE_LLVM_STORE || ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL ||
                    ot == OP_TYPE_LLVM_ALLOCA) { h_pure = false; break; }
            }
            if (!h_pure) continue;

            // Find pure latch block L (`br ^hblk(l_sops...)`, <= 4 pure ops) where L_arg_j -> H_arg_k.
            for (size_t li = 1; li < nb; li++) {
                if (li == h) continue;
                MLIR_BlockHandle lblk = blks[li];
                size_t lna = MLIR_GetBlockNumArgs(ctx, lblk);
                if (lna == 0 || lna > 4) continue;
                size_t lno = MLIR_GetBlockNumOps(ctx, lblk);
                if (lno == 0 || lno > 5) continue;
                MLIR_OpHandle lterm = MLIR_GetBlockOp(ctx, lblk, lno - 1);
                if (MLIR_GetOpType(ctx, lterm) != OP_TYPE_LLVM_BR ||
                    MLIR_GetOpSuccessor(ctx, lterm, 0) != hblk ||
                    MLIR_GetOpNumSuccessorOperands(ctx, lterm, 0) != hna) continue;

                // Check all non-terminator ops in L are pure & clonable.
                bool l_pure = true;
                for (size_t i = 0; i + 1 < lno; i++) {
                    MLIR_OpHandle lo = MLIR_GetBlockOp(ctx, lblk, i);
                    MLIR_OpType lot = MLIR_GetOpType(ctx, lo);
                    if (MLIR_GetOpNumRegions(ctx, lo) > 0 || MLIR_GetOpNumOperands(ctx, lo) > 16 ||
                        MLIR_GetOpNumResults(ctx, lo) > 1 || MLIR_GetOpNumAttributes(ctx, lo) > 8 ||
                        lot == OP_TYPE_LLVM_STORE || lot == OP_TYPE_LLVM_LOAD ||
                        lot == OP_TYPE_LLVM_CALL || lot == OP_TYPE_FUNC_CALL || lot == OP_TYPE_LLVM_ALLOCA) {
                        l_pure = false; break;
                    }
                }
                if (!l_pure) continue;

                MLIR_ValueHandle passed_k = MLIR_GetOpSuccessorOperand(ctx, lterm, 0, h_arg_k);
                size_t l_arg_j = lna;
                for (size_t a = 0; a < lna; a++)
                    if (MLIR_GetBlockArg(ctx, lblk, a) == passed_k) { l_arg_j = a; break; }
                if (l_arg_j >= lna) continue;

                // Only thread if lna == 1 (so the shared trampoline needs no extra block args).
                if (lna != 1) continue;

                // Check if any predecessor P of L passes constant K to L_arg_j.
                MLIR_BlockHandle texit = MLIR_INVALID_HANDLE;
                for (size_t p = 0; p < nb; p++) {
                    if (p == li) continue;
                    MLIR_BlockHandle pblk = blks[p];
                    MLIR_OpHandle pterm = MLIR_GetBlockTerminator(ctx, pblk);
                    if (pterm == MLIR_INVALID_HANDLE) continue;
                    if (MLIR_GetOpType(ctx, pterm) != OP_TYPE_LLVM_BR) continue;
                    if (MLIR_GetOpSuccessor(ctx, pterm, 0) != lblk ||
                        MLIR_GetOpNumSuccessorOperands(ctx, pterm, 0) != 1) continue;
                    int64_t cv = 0;
                    if (!cf_get_int_const(ctx, MLIR_GetOpSuccessorOperand(ctx, pterm, 0, 0), &cv, NULL))
                        continue;
                    if (cf_mask(cv, Kw) != cf_mask(K, Kw)) continue;

                    if (texit == MLIR_INVALID_HANDLE) {
                        // Build shared exit trampoline `^texit` at end of region:
                        // 1) Clone L's pure ops with L_arg_0 -> K_const
                        // 2) Clone exit_b's ops with H_arg_m -> L_sops'[m] and exit_b_arg_q -> mapped(e_sops[q])
                        texit = MLIR_CreateBlock(ctx);
                        MLIR_InsertRegionBlockBefore(ctx, reg, texit, exit_b);
                        size_t tpos = 0;
                        MLIR_ValueHandle old_v[32], new_v[32];
                        size_t n_map = 0;
                        MLIR_ValueHandle k_val = cf_make_int_const(ctx, texit, tpos++,
                            MLIR_GetValueType(ctx, arg_v), K);
                        old_v[n_map] = MLIR_GetBlockArg(ctx, lblk, 0);
                        new_v[n_map] = k_val;
                        n_map++;
                        for (size_t i = 0; i + 1 < lno; i++) {
                            MLIR_OpHandle lo = MLIR_GetBlockOp(ctx, lblk, i);
                            MLIR_OpHandle cl = unroll_clone_op(ctx, lo, old_v, new_v, n_map);
                            MLIR_InsertBlockOpAtIndex(ctx, texit, cl, tpos++);
                            if (MLIR_GetOpNumResults(ctx, lo) == 1) {
                                old_v[n_map] = MLIR_GetOpResult(ctx, lo, 0);
                                new_v[n_map] = MLIR_GetOpResult(ctx, cl, 0);
                                n_map++;
                            }
                        }
                        // Map H_arg_m -> cloned L_sops[m].
                        MLIR_ValueHandle h_mapped[8];
                        for (size_t m = 0; m < hna; m++) {
                            MLIR_ValueHandle lv = MLIR_GetOpSuccessorOperand(ctx, lterm, 0, m);
                            for (size_t mi = 0; mi < n_map; mi++)
                                if (old_v[mi] == lv) { lv = new_v[mi]; break; }
                            h_mapped[m] = lv;
                        }
                        MLIR_ValueHandle ex_old[32], ex_new[32];
                        size_t ex_map = 0;
                        for (size_t m = 0; m < hna; m++) {
                            ex_old[ex_map] = MLIR_GetBlockArg(ctx, hblk, m);
                            ex_new[ex_map] = h_mapped[m];
                            ex_map++;
                        }
                        for (size_t q = 0; q < n_ex; q++) {
                            MLIR_ValueHandle ev = MLIR_GetOpSuccessorOperand(ctx, hterm, ex_arm, q);
                            for (size_t m = 0; m < hna; m++)
                                if (MLIR_GetBlockArg(ctx, hblk, m) == ev) { ev = h_mapped[m]; break; }
                            ex_old[ex_map] = MLIR_GetBlockArg(ctx, exit_b, q);
                            ex_new[ex_map] = ev;
                            ex_map++;
                        }
                        for (size_t i = 0; i < ex_no; i++) {
                            MLIR_OpHandle eo = MLIR_GetBlockOp(ctx, exit_b, i);
                            MLIR_OpHandle cl = unroll_clone_op(ctx, eo, ex_old, ex_new, ex_map);
                            MLIR_InsertBlockOpAtIndex(ctx, texit, cl, tpos++);
                            if (MLIR_GetOpNumResults(ctx, eo) == 1) {
                                ex_old[ex_map] = MLIR_GetOpResult(ctx, eo, 0);
                                ex_new[ex_map] = MLIR_GetOpResult(ctx, cl, 0);
                                ex_map++;
                            }
                        }
                    }

                    // Rewrite P's `br ^lblk(K)` -> `br ^texit`.
                    MLIR_OpSpec bsp; memset(&bsp, 0, sizeof(bsp));
                    bsp.type = OP_TYPE_LLVM_BR;
                    bsp.successors = &texit; bsp.n_successors = 1;
                    uint32_t zero_cnt = 0;
                    bsp.successor_operand_counts = &zero_cnt;
                    MLIR_OpHandle nbr = MLIR_CreateOpFromSpec(ctx, &bsp);
                    MLIR_InsertBlockOpAtIndex(ctx, pblk, nbr, MLIR_GetBlockNumOps(ctx, pblk) - 1);
                    MLIR_DetachOp(ctx, pterm);
                }
            }
        }
    }
}

void mlir_llvm_constfold(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_CONSTFOLD")) return;
    if (MLIR_GetOpNumRegions(ctx, module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        constfold_func(ctx, fn);
        const_blockarg_thread_func(ctx, fn);
    }
}

// ---------------------------------------------------------------------------
// Exact-trip-count loop unrolling:
//   - Complete unroll for small constant trip counts (2..8, e.g. CoreMark crcu8)
//   - 4x unroll for constant trip counts >= 16 divisible by 4 (e.g. matmul 256)
// Matches a single-body-block (or body+latch) counted loop:
//   ^pre: br(init_iv, init_args...) [^H]
//   ^H(iv, accs...): cond = icmp slt/sle(iv, lim) ; cond_br(cond) [^B, ^E]
//   ^B: <pure/mem ops, no calls/allocas, <= 32 ops> ; iv_next = add(iv, step) ; br(iv_next, accs_next...) [^H]
// ---------------------------------------------------------------------------
static MLIR_OpHandle unroll_clone_op(MLIR_Context *ctx, MLIR_OpHandle src,
                                     const MLIR_ValueHandle *old_v,
                                     const MLIR_ValueHandle *new_v, size_t n_map) {
    MLIR_OpType ot = MLIR_GetOpType(ctx, src);
    size_t nop = MLIR_GetOpNumOperands(ctx, src);
    MLIR_ValueHandle ops[16];
    for (size_t k = 0; k < nop && k < 16; k++) {
        MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, src, k);
        for (size_t m = 0; m < n_map; m++) {
            if (old_v[m] == ov) { ov = new_v[m]; break; }
        }
        ops[k] = ov;
    }
    size_t nr = MLIR_GetOpNumResults(ctx, src);
    MLIR_TypeHandle rty[2];
    for (size_t r = 0; r < nr && r < 2; r++)
        rty[r] = MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, src, r));
    size_t na = MLIR_GetOpNumAttributes(ctx, src);
    MLIR_NamedAttr attrs[8];
    for (size_t a = 0; a < na && a < 8; a++) {
        attrs[a].key = MLIR_GetOpAttributeKey(ctx, src, a);
        attrs[a].value = MLIR_GetOpAttribute(ctx, src, a);
    }
    MLIR_OpSpec sp;
    memset(&sp, 0, sizeof(sp));
    sp.type = ot;
    sp.operands = ops;
    sp.n_operands = nop;
    sp.result_types = rty;
    sp.n_results = nr;
    sp.attrs = attrs;
    sp.n_attrs = na;
    return MLIR_CreateOpFromSpec(ctx, &sp);
}

static void unroll_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 3 || nb > 512) return;

    // Count predecessors for each block.
    MLIR_BlockHandle *blks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    uint32_t *pred_cnt = (uint32_t *)calloc(nb, sizeof(uint32_t));
    size_t *pred_a = (size_t *)malloc(nb * sizeof(size_t));
    size_t *pred_b = (size_t *)malloc(nb * sizeof(size_t));
    if (!blks || !pred_cnt || !pred_a || !pred_b) {
        free(blks); free(pred_cnt); free(pred_a); free(pred_b); return;
    }
    size_t bhcap = 16;
    while (bhcap < nb * 2) bhcap *= 2;
    uintptr_t *bhk = (uintptr_t *)calloc(bhcap, sizeof(uintptr_t));
    size_t    *bhv = (size_t    *)malloc(bhcap * sizeof(size_t));
    size_t bhmask = bhcap - 1;
    for (size_t b = 0; b < nb; b++) {
        blks[b] = MLIR_GetRegionBlock(ctx, reg, b);
        uintptr_t k = (uintptr_t)blks[b];
        size_t s = (k * 1099511628211ull) & bhmask;
        while (bhk[s]) s = (s + 1) & bhmask;
        bhk[s] = k; bhv[s] = b;
    }
    for (size_t b = 0; b < nb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
        if (no == 0) continue;
        MLIR_OpHandle t = MLIR_GetBlockOp(ctx, blks[b], no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, t);
        for (size_t si = 0; si < ns; si++) {
            uintptr_t tk = (uintptr_t)MLIR_GetOpSuccessor(ctx, t, si);
            size_t s = (tk * 1099511628211ull) & bhmask;
            while (bhk[s]) {
                if (bhk[s] == tk) {
                    size_t ti = bhv[s];
                    if (pred_cnt[ti] == 0) pred_a[ti] = b;
                    else if (pred_cnt[ti] == 1) pred_b[ti] = b;
                    pred_cnt[ti]++;
                    break;
                }
                s = (s + 1) & bhmask;
            }
        }
    }

    for (size_t h = 1; h < nb; h++) {
        if (pred_cnt[h] != 2) continue;
        MLIR_BlockHandle hblk = blks[h];
        size_t hna = MLIR_GetBlockNumArgs(ctx, hblk);
        if (hna == 0 || hna > 8) continue;
        size_t hno = MLIR_GetBlockNumOps(ctx, hblk);
        if (hno < 2 || hno > 4) continue;
        MLIR_OpHandle hterm = MLIR_GetBlockOp(ctx, hblk, hno - 1);
        if (MLIR_GetOpType(ctx, hterm) != OP_TYPE_LLVM_COND_BR) continue;
        if (MLIR_GetOpNumSuccessorOperands(ctx, hterm, 0) != 0) continue;

        // Header must have only constants + 1 icmp(iv, lim_const) + cond_br.
        MLIR_ValueHandle cond = MLIR_GetOpOperand(ctx, hterm, 0);
        MLIR_OpHandle icmp = MLIR_INVALID_HANDLE;
        bool h_ok = true;
        for (size_t i = 0; i + 1 < hno; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, hblk, i);
            MLIR_OpType ot = MLIR_GetOpType(ctx, op);
            if (ot == OP_TYPE_LLVM_MLIR_CONSTANT || ot == OP_TYPE_LLVM_MLIR_ZERO) continue;
            if (ot == OP_TYPE_LLVM_ICMP && icmp == MLIR_INVALID_HANDLE &&
                MLIR_GetOpResult(ctx, op, 0) == cond) {
                icmp = op;
            } else { h_ok = false; break; }
        }
        if (!h_ok || icmp == MLIR_INVALID_HANDLE) continue;
        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, icmp, MLIR_KEY_PREDICATE);
        int64_t pred = pa ? MLIR_GetAttributeInteger(ctx, pa) : -1;
        if (pred != 2 && pred != 3 && pred != 6) continue; // slt, sle, or ult

        MLIR_ValueHandle iv = MLIR_GetOpOperand(ctx, icmp, 0);
        MLIR_ValueHandle lim_v = MLIR_GetOpOperand(ctx, icmp, 1);
        MLIR_TypeHandle iv_ty = MLIR_GetValueType(ctx, iv);
        if (MLIR_GetTypeKind(ctx, iv_ty) != MLIR_TYPE_KIND_INTEGER) continue;
        uint32_t iv_w = MLIR_GetTypeWidth(ctx, iv_ty);
        if (iv_w == 0 || iv_w > 64) continue;
        int64_t lim_c = 0;
        bool has_const_lim = cf_get_int_const(ctx, lim_v, &lim_c, NULL);
        if (has_const_lim) lim_c = cf_sext(lim_c, iv_w);

        // Find which block arg of hblk is `iv`.
        size_t iv_arg = hna;
        for (size_t a = 0; a < hna; a++)
            if (MLIR_GetBlockArg(ctx, hblk, a) == iv) { iv_arg = a; break; }
        if (iv_arg >= hna) continue;

        MLIR_BlockHandle bblk = MLIR_GetOpSuccessor(ctx, hterm, 0);
        MLIR_BlockHandle eblk = MLIR_GetOpSuccessor(ctx, hterm, 1);
        if (bblk == hblk || bblk == eblk || MLIR_GetBlockNumArgs(ctx, bblk) != 0) continue;
        // Lookup body block index
        size_t bi = nb;
        uintptr_t bk_key = (uintptr_t)bblk;
        size_t s = (bk_key * 1099511628211ull) & bhmask;
        while (bhk[s]) { if (bhk[s] == bk_key) { bi = bhv[s]; break; } s = (s + 1) & bhmask; }
        if (bi >= nb || pred_cnt[bi] != 1) continue;

        // Identify latch block (either bblk itself or a single-pred latch block).
        size_t bno = MLIR_GetBlockNumOps(ctx, bblk);
        if (bno < 2 || bno > 48) continue;
        MLIR_OpHandle bterm = MLIR_GetBlockOp(ctx, bblk, bno - 1);
        if (MLIR_GetOpType(ctx, bterm) != OP_TYPE_LLVM_BR) continue;
        MLIR_BlockHandle lblk = MLIR_INVALID_HANDLE;
        size_t li = nb;
        MLIR_OpHandle lterm = bterm;
        if (MLIR_GetOpSuccessor(ctx, bterm, 0) != hblk) {
            if (MLIR_GetOpNumSuccessorOperands(ctx, bterm, 0) != 0) continue;
            lblk = MLIR_GetOpSuccessor(ctx, bterm, 0);
            if (lblk == bblk || lblk == eblk || MLIR_GetBlockNumArgs(ctx, lblk) != 0) continue;
            uintptr_t lk = (uintptr_t)lblk;
            size_t ls = (lk * 1099511628211ull) & bhmask;
            while (bhk[ls]) { if (bhk[ls] == lk) { li = bhv[ls]; break; } ls = (ls + 1) & bhmask; }
            if (li >= nb || pred_cnt[li] != 1) continue;
            size_t lno = MLIR_GetBlockNumOps(ctx, lblk);
            if (lno < 2 || lno > 16) continue;
            lterm = MLIR_GetBlockOp(ctx, lblk, lno - 1);
            if (MLIR_GetOpType(ctx, lterm) != OP_TYPE_LLVM_BR ||
                MLIR_GetOpSuccessor(ctx, lterm, 0) != hblk) continue;
        }
        if (MLIR_GetOpNumSuccessorOperands(ctx, lterm, 0) != hna) continue;

        // Identify preheader (the other predecessor of hblk).
        size_t latch_idx = (lblk != MLIR_INVALID_HANDLE) ? li : bi;
        size_t pre_idx = (pred_a[h] == latch_idx) ? pred_b[h] : (pred_b[h] == latch_idx ? pred_a[h] : nb);
        if (pre_idx >= nb) continue;
        MLIR_BlockHandle pre_blk = blks[pre_idx];
        size_t pno = MLIR_GetBlockNumOps(ctx, pre_blk);
        if (pno == 0) continue;
        MLIR_OpHandle pterm = MLIR_GetBlockOp(ctx, pre_blk, pno - 1);
        if (MLIR_GetOpType(ctx, pterm) != OP_TYPE_LLVM_BR ||
            MLIR_GetOpSuccessor(ctx, pterm, 0) != hblk ||
            MLIR_GetOpNumSuccessorOperands(ctx, pterm, 0) != hna) continue;

        int64_t init_c = 0;
        bool has_const_init = cf_get_int_const(ctx, MLIR_GetOpSuccessorOperand(ctx, pterm, 0, iv_arg), &init_c, NULL);
        if (has_const_init) init_c = cf_sext(init_c, iv_w);

        // Verify iv_next = add(iv, step) (or and(add(iv, step_const), mask) for narrow unsigned iv).
        MLIR_ValueHandle iv_next = MLIR_GetOpSuccessorOperand(ctx, lterm, 0, iv_arg);
        MLIR_OpHandle iv_add = MLIR_GetValueDefiningOp(ctx, iv_next);
        bool masked_iv = false;
        if (iv_add != MLIR_INVALID_HANDLE && MLIR_GetOpType(ctx, iv_add) == OP_TYPE_LLVM_AND) {
            int64_t mask_c = 0;
            if (has_const_lim &&
                cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, iv_add, 1), &mask_c, NULL) &&
                lim_c >= 0 && lim_c < mask_c && (mask_c & (mask_c + 1)) == 0) {
                iv_add = MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, iv_add, 0));
                masked_iv = true;
            }
        }
        if (iv_add == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, iv_add) != OP_TYPE_LLVM_ADD) continue;
        if (MLIR_GetOpOperand(ctx, iv_add, 0) != iv) continue;
        MLIR_ValueHandle step_v = MLIR_GetOpOperand(ctx, iv_add, 1);
        int64_t step_c = 0;
        bool has_const_step = cf_get_int_const(ctx, step_v, &step_c, NULL);
        if (has_const_step) step_c = cf_sext(step_c, iv_w);

        // Collect all body (+ latch) ops and check they are safe to clone (no calls/allocas/regions, <= 48 ops).
        MLIR_OpHandle body_ops[64];
        size_t n_body = 0;
        bool body_ok = true;
        MLIR_BlockHandle sblks[2] = { bblk, lblk };
        for (int sbi = 0; sbi < 2 && body_ok; sbi++) {
            if (sblks[sbi] == MLIR_INVALID_HANDLE) continue;
            size_t sno = MLIR_GetBlockNumOps(ctx, sblks[sbi]);
            for (size_t i = 0; i + 1 < sno; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, sblks[sbi], i);
                MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                if (MLIR_GetOpNumRegions(ctx, op) > 0 || MLIR_GetOpNumOperands(ctx, op) > 16 ||
                    MLIR_GetOpNumResults(ctx, op) > 1 || MLIR_GetOpNumAttributes(ctx, op) > 8 ||
                    ot == OP_TYPE_LLVM_CALL || ot == OP_TYPE_FUNC_CALL || ot == OP_TYPE_LLVM_ALLOCA) {
                    body_ok = false; break;
                }
                if (n_body >= 48) { body_ok = false; break; }
                body_ops[n_body++] = op;
            }
        }
        if (!body_ok || n_body == 0) continue;

        // Decide static unroll factor or runtime 2x/4x unroll:
        size_t factor = 0;
        bool complete = false;
        size_t runtime_uf = 0;
        if (has_const_init && has_const_lim && has_const_step && step_c > 0 && (pred == 2 || pred == 3)) {
            int64_t diff = lim_c - init_c + (pred == 3 ? 1 : 0);
            if (diff > 0 && (diff % step_c) == 0) {
                int64_t trips = diff / step_c;
                if (trips >= 2 && trips <= 8 && n_body * (size_t)trips <= 256) {
                    factor = (size_t)trips;
                    complete = true;
                } else if (trips >= 16 && (trips % 4) == 0 && n_body <= 32) {
                    factor = 4;
                }
            }
        }
        if (factor == 0 && !masked_iv && n_body <= 18 && hna <= 4) {
            bool ok_rt = false;
            if ((pred == 2 || pred == 6) && has_const_init && init_c == 0 &&
                has_const_step && step_c == 1 && iv_w >= 32 &&
                MLIR_GetOpNumSuccessorOperands(ctx, hterm, 1) == 0) {
                ok_rt = true;
            } else if (pred == 3 && iv_w == 64 && has_const_lim && lim_c > 0 && lim_c <= 0x3fffffffll &&
                       n_body <= 6 && hna == 1) {
                // Loop-invariant proven-positive step (e.g. sieve inner marking loop `j += i`).
                bool step_pos = (has_const_step && step_c > 0 && step_c <= 0x3fffffffll);
                if (!step_pos && MLIR_GetValueDefiningOp(ctx, step_v) == MLIR_INVALID_HANDLE) {
                    for (size_t oh = 1; oh < nb && !step_pos; oh++) {
                        if (oh == h || pred_cnt[oh] != 2) continue;
                        MLIR_BlockHandle ob = blks[oh];
                        size_t ona = MLIR_GetBlockNumArgs(ctx, ob);
                        for (size_t oa = 0; oa < ona; oa++) {
                            if (MLIR_GetBlockArg(ctx, ob, oa) != step_v) continue;
                            size_t opreds[2] = { pred_a[oh], pred_b[oh] };
                            bool saw_init = false, saw_inc = false;
                            for (int pi = 0; pi < 2; pi++) {
                                MLIR_OpHandle pt = MLIR_GetBlockTerminator(ctx, blks[opreds[pi]]);
                                if (pt == MLIR_INVALID_HANDLE) continue;
                                size_t pns = MLIR_GetOpNumSuccessors(ctx, pt);
                                for (size_t ps = 0; ps < pns; ps++) {
                                    if (MLIR_GetOpSuccessor(ctx, pt, ps) != ob) continue;
                                    if (MLIR_GetOpNumSuccessorOperands(ctx, pt, ps) != ona) continue;
                                    MLIR_ValueHandle pv = MLIR_GetOpSuccessorOperand(ctx, pt, ps, oa);
                                    int64_t ic = 0;
                                    if (cf_get_int_const(ctx, pv, &ic, NULL) && ic > 0 && ic <= 0x3fffffffll) {
                                        saw_init = true;
                                    } else {
                                        MLIR_OpHandle pd = MLIR_GetValueDefiningOp(ctx, pv);
                                        if (pd != MLIR_INVALID_HANDLE && MLIR_GetOpType(ctx, pd) == OP_TYPE_LLVM_ADD) {
                                            MLIR_ValueHandle o0 = MLIR_GetOpOperand(ctx, pd, 0);
                                            MLIR_ValueHandle o1 = MLIR_GetOpOperand(ctx, pd, 1);
                                            int64_t sc = 0;
                                            if ((o0 == step_v && cf_get_int_const(ctx, o1, &sc, NULL) && sc > 0 && sc <= 16) ||
                                                (o1 == step_v && cf_get_int_const(ctx, o0, &sc, NULL) && sc > 0 && sc <= 16))
                                                saw_inc = true;
                                        }
                                    }
                                }
                            }
                            if (saw_init && saw_inc) step_pos = true;
                        }
                    }
                }
                if (step_pos) ok_rt = true;
            }
            if (ok_rt) {
                MLIR_OpHandle ldef = MLIR_GetValueDefiningOp(ctx, lim_v);
                MLIR_BlockHandle lblk_def = (ldef != MLIR_INVALID_HANDLE) ? MLIR_GetOpParentBlock(ctx, ldef) : MLIR_INVALID_HANDLE;
                bool lim_ok = has_const_lim || (lblk_def != hblk && lblk_def != bblk && lblk_def != lblk);
                if (lim_ok) {
                    for (size_t a = 0; a < hna; a++)
                        if (MLIR_GetBlockArg(ctx, hblk, a) == lim_v) { lim_ok = false; break; }
                }
                if (lim_ok) runtime_uf = (pred == 3 && n_body <= 6 && hna == 1) ? 4 : 2;
            }
        }
        if (factor == 0 && runtime_uf == 0) continue;

        // Verify no body-defined value escapes outside {bblk, lblk} (only header block args may be live out!).
        bool body_escapes = false;
        for (size_t ob = 0; ob < nb && !body_escapes; ob++) {
            if (ob == bi || ob == li) continue;
            size_t ono = MLIR_GetBlockNumOps(ctx, blks[ob]);
            for (size_t oi = 0; oi < ono && !body_escapes; oi++) {
                MLIR_OpHandle oop = MLIR_GetBlockOp(ctx, blks[ob], oi);
                size_t onop = MLIR_GetOpNumOperands(ctx, oop);
                for (size_t k = 0; k < onop && !body_escapes; k++) {
                    MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, oop, k);
                    for (size_t j = 0; j < n_body; j++) {
                        if (MLIR_GetOpNumResults(ctx, body_ops[j]) == 1 &&
                            MLIR_GetOpResult(ctx, body_ops[j], 0) == ov) { body_escapes = true; break; }
                    }
                }
                if (oi + 1 == ono && !body_escapes) {
                    size_t ons = MLIR_GetOpNumSuccessors(ctx, oop);
                    for (size_t osi = 0; osi < ons && !body_escapes; osi++) {
                        size_t onso = MLIR_GetOpNumSuccessorOperands(ctx, oop, osi);
                        for (size_t k = 0; k < onso && !body_escapes; k++) {
                            MLIR_ValueHandle ov = MLIR_GetOpSuccessorOperand(ctx, oop, osi, k);
                            for (size_t j = 0; j < n_body; j++) {
                                if (MLIR_GetOpNumResults(ctx, body_ops[j]) == 1 &&
                                    MLIR_GetOpResult(ctx, body_ops[j], 0) == ov) { body_escapes = true; break; }
                            }
                        }
                    }
                }
            }
        }
        if (body_escapes) continue;

        MLIR_ValueHandle hargs[8], init_vals[8], back_vals[8], cur_vals[8];
        for (size_t a = 0; a < hna; a++) {
            hargs[a] = MLIR_GetBlockArg(ctx, hblk, a);
            init_vals[a] = MLIR_GetOpSuccessorOperand(ctx, pterm, 0, a);
            back_vals[a] = MLIR_GetOpSuccessorOperand(ctx, lterm, 0, a);
            cur_vals[a] = back_vals[a];
        }

        if (runtime_uf > 0) {
            // Runtime 2x/4x unroll: insert `h2blk(iv2, accs2...)` + `b2blk` before `hblk`.
            // For guarded 4x (`pred == 3 && iv_w == 64 && has_const_lim`), emit an entry
            // guard `if (init <= lim)` in `pre_blk` -> `h2pre` (computing `lim2 = lim - 3*step`)
            // else -> `eblk`, so zero-trip loops (like sieve's 282k primes > sqrt(N)) skip
            // all unroll setup!
            MLIR_BlockHandle h2pre = MLIR_INVALID_HANDLE;
            MLIR_BlockHandle h2blk = MLIR_CreateBlock(ctx);
            MLIR_BlockHandle b2blk = MLIR_CreateBlock(ctx);
            if (runtime_uf == 4) {
                h2pre = MLIR_CreateBlock(ctx);
                MLIR_InsertRegionBlockBefore(ctx, reg, h2pre, hblk);
            }
            MLIR_InsertRegionBlockBefore(ctx, reg, h2blk, hblk);
            MLIR_InsertRegionBlockBefore(ctx, reg, b2blk, hblk);
            MLIR_ValueHandle h2args[8];
            for (size_t a = 0; a < hna; a++)
                h2args[a] = MLIR_AddBlockArgument(ctx, h2blk, MLIR_GetValueType(ctx, hargs[a]), MLIR_INVALID_HANDLE);

            size_t h2pos = 0;
            MLIR_ValueHandle cmp_lhs = h2args[iv_arg];
            MLIR_ValueHandle cmp_rhs = lim_v;
            MLIR_TypeHandle i1_ty = MLIR_CreateTypeInteger(ctx, 1, false);
            if (runtime_uf == 4) {
                // Build h2pre: step3 = step + step + step; lim2 = lim_c - step3; br ^h2blk(init_vals)
                size_t p2 = 0;
                MLIR_ValueHandle step_k = step_v;
                if (has_const_step) {
                    step_k = cf_make_int_const(ctx, h2pre, p2++, iv_ty, step_c * 3ll);
                } else {
                    MLIR_ValueHandle s2ops[2] = { step_v, step_v };
                    MLIR_OpSpec s2sp; memset(&s2sp, 0, sizeof(s2sp));
                    s2sp.type = OP_TYPE_LLVM_ADD; s2sp.operands = s2ops; s2sp.n_operands = 2;
                    s2sp.result_types = &iv_ty; s2sp.n_results = 1;
                    MLIR_OpHandle s2op = MLIR_CreateOpFromSpec(ctx, &s2sp);
                    MLIR_InsertBlockOpAtIndex(ctx, h2pre, s2op, p2++);
                    MLIR_ValueHandle step2 = MLIR_GetOpResult(ctx, s2op, 0);
                    MLIR_ValueHandle s3ops[2] = { step2, step_v };
                    MLIR_OpSpec s3sp; memset(&s3sp, 0, sizeof(s3sp));
                    s3sp.type = OP_TYPE_LLVM_ADD; s3sp.operands = s3ops; s3sp.n_operands = 2;
                    s3sp.result_types = &iv_ty; s3sp.n_results = 1;
                    MLIR_OpHandle s3op = MLIR_CreateOpFromSpec(ctx, &s3sp);
                    MLIR_InsertBlockOpAtIndex(ctx, h2pre, s3op, p2++);
                    step_k = MLIR_GetOpResult(ctx, s3op, 0);
                }
                MLIR_ValueHandle lim_p = cf_make_int_const(ctx, h2pre, p2++, iv_ty, lim_c);
                MLIR_ValueHandle sub_ops[2] = { lim_p, step_k };
                MLIR_OpSpec sub_sp; memset(&sub_sp, 0, sizeof(sub_sp));
                sub_sp.type = OP_TYPE_LLVM_SUB; sub_sp.operands = sub_ops; sub_sp.n_operands = 2;
                sub_sp.result_types = &iv_ty; sub_sp.n_results = 1;
                MLIR_OpHandle sub_op = MLIR_CreateOpFromSpec(ctx, &sub_sp);
                MLIR_InsertBlockOpAtIndex(ctx, h2pre, sub_op, p2++);
                cmp_rhs = MLIR_GetOpResult(ctx, sub_op, 0);
                uint32_t hna32_p = (uint32_t)hna;
                MLIR_OpSpec p2sp; memset(&p2sp, 0, sizeof(p2sp));
                p2sp.type = OP_TYPE_LLVM_BR;
                p2sp.operands = init_vals; p2sp.n_operands = hna;
                p2sp.successors = &h2blk; p2sp.n_successors = 1;
                p2sp.successor_operand_counts = &hna32_p;
                MLIR_InsertBlockOpAtIndex(ctx, h2pre, MLIR_CreateOpFromSpec(ctx, &p2sp), p2);
            } else {
                MLIR_ValueHandle step_h2 = step_v;
                if (has_const_step) {
                    step_h2 = cf_make_int_const(ctx, h2blk, h2pos++, iv_ty, step_c);
                }
                if (has_const_lim) {
                    cmp_rhs = cf_make_int_const(ctx, h2blk, h2pos++, iv_ty, lim_c);
                }
                MLIR_ValueHandle aops[2] = { h2args[iv_arg], step_h2 };
                MLIR_OpSpec asp; memset(&asp, 0, sizeof(asp));
                asp.type = OP_TYPE_LLVM_ADD; asp.operands = aops; asp.n_operands = 2;
                asp.result_types = &iv_ty; asp.n_results = 1;
                MLIR_OpHandle aop = MLIR_CreateOpFromSpec(ctx, &asp);
                MLIR_InsertBlockOpAtIndex(ctx, h2blk, aop, h2pos++);
                cmp_lhs = MLIR_GetOpResult(ctx, aop, 0);
            }

            MLIR_ValueHandle cops[2] = { cmp_lhs, cmp_rhs };
            MLIR_NamedAttr pattr = MLIR_Named(MLIR_KEY_PREDICATE, pa);
            MLIR_OpSpec csp; memset(&csp, 0, sizeof(csp));
            csp.type = OP_TYPE_LLVM_ICMP; csp.operands = cops; csp.n_operands = 2;
            csp.result_types = &i1_ty; csp.n_results = 1;
            csp.attrs = &pattr; csp.n_attrs = 1;
            MLIR_OpHandle cop = MLIR_CreateOpFromSpec(ctx, &csp);
            MLIR_InsertBlockOpAtIndex(ctx, h2blk, cop, h2pos++);
            MLIR_ValueHandle cond2 = MLIR_GetOpResult(ctx, cop, 0);

            MLIR_ValueHandle cb_ops[9];
            cb_ops[0] = cond2;
            for (size_t a = 0; a < hna; a++) cb_ops[1 + a] = h2args[a];
            MLIR_BlockHandle cb_succs[2] = { b2blk, hblk };
            uint32_t cb_cnts[2] = { 0, (uint32_t)hna };
            MLIR_OpSpec cbsp; memset(&cbsp, 0, sizeof(cbsp));
            cbsp.type = OP_TYPE_LLVM_COND_BR;
            cbsp.operands = cb_ops; cbsp.n_operands = 1 + hna;
            cbsp.successors = cb_succs; cbsp.n_successors = 2;
            cbsp.successor_operand_counts = cb_cnts;
            MLIR_OpHandle cb2 = MLIR_CreateOpFromSpec(ctx, &cbsp);
            MLIR_InsertBlockOpAtIndex(ctx, h2blk, cb2, h2pos);

            // Build b2blk: `runtime_uf` cloned copies of body_ops, then br ^h2blk(cur_vals).
            size_t b2pos = 0;
            for (size_t a = 0; a < hna; a++) cur_vals[a] = h2args[a];
            for (size_t u = 0; u < runtime_uf; u++) {
                MLIR_ValueHandle old_v[64], new_v[64];
                size_t n_map = 0;
                for (size_t a = 0; a < hna; a++) {
                    old_v[n_map] = hargs[a];
                    new_v[n_map] = cur_vals[a];
                    n_map++;
                }
                for (size_t j = 0; j < n_body; j++) {
                    MLIR_OpHandle cl = unroll_clone_op(ctx, body_ops[j], old_v, new_v, n_map);
                    MLIR_InsertBlockOpAtIndex(ctx, b2blk, cl, b2pos++);
                    if (MLIR_GetOpNumResults(ctx, body_ops[j]) == 1) {
                        old_v[n_map] = MLIR_GetOpResult(ctx, body_ops[j], 0);
                        new_v[n_map] = MLIR_GetOpResult(ctx, cl, 0);
                        n_map++;
                    }
                }
                for (size_t a = 0; a < hna; a++) {
                    MLIR_ValueHandle nv = back_vals[a];
                    for (size_t m = 0; m < n_map; m++) {
                        if (old_v[m] == nv) { nv = new_v[m]; break; }
                    }
                    cur_vals[a] = nv;
                }
            }
            uint32_t hna32 = (uint32_t)hna;
            MLIR_OpSpec bsp; memset(&bsp, 0, sizeof(bsp));
            bsp.type = OP_TYPE_LLVM_BR;
            bsp.operands = cur_vals; bsp.n_operands = hna;
            bsp.successors = &h2blk; bsp.n_successors = 1;
            bsp.successor_operand_counts = &hna32;
            MLIR_OpHandle b2t = MLIR_CreateOpFromSpec(ctx, &bsp);
            MLIR_InsertBlockOpAtIndex(ctx, b2blk, b2t, b2pos);

            // Redirect pre_blk: either guarded `if (init <= lim) -> h2pre else -> eblk`, or direct `br ^h2blk(init_vals)`.
            if (runtime_uf == 4) {
                size_t pins = MLIR_GetBlockNumOps(ctx, pre_blk) - 1;
                MLIR_ValueHandle lim_g = cf_make_int_const(ctx, pre_blk, pins++, iv_ty, lim_c);
                MLIR_ValueHandle gops[2] = { init_vals[iv_arg], lim_g };
                MLIR_OpSpec gsp; memset(&gsp, 0, sizeof(gsp));
                gsp.type = OP_TYPE_LLVM_ICMP; gsp.operands = gops; gsp.n_operands = 2;
                gsp.result_types = &i1_ty; gsp.n_results = 1;
                gsp.attrs = &pattr; gsp.n_attrs = 1;
                MLIR_OpHandle gop = MLIR_CreateOpFromSpec(ctx, &gsp);
                MLIR_InsertBlockOpAtIndex(ctx, pre_blk, gop, pins++);
                MLIR_ValueHandle gcond = MLIR_GetOpResult(ctx, gop, 0);
                size_t nex = MLIR_GetOpNumSuccessorOperands(ctx, hterm, 1);
                MLIR_ValueHandle gcb_ops[9];
                gcb_ops[0] = gcond;
                for (size_t k = 0; k < nex && k < 8; k++) {
                    MLIR_ValueHandle ev = MLIR_GetOpSuccessorOperand(ctx, hterm, 1, k);
                    for (size_t a = 0; a < hna; a++)
                        if (hargs[a] == ev) { ev = init_vals[a]; break; }
                    gcb_ops[1 + k] = ev;
                }
                MLIR_BlockHandle g_succs[2] = { h2pre, eblk };
                uint32_t g_cnts[2] = { 0, (uint32_t)nex };
                MLIR_OpSpec gcb; memset(&gcb, 0, sizeof(gcb));
                gcb.type = OP_TYPE_LLVM_COND_BR;
                gcb.operands = gcb_ops; gcb.n_operands = 1 + nex;
                gcb.successors = g_succs; gcb.n_successors = 2;
                gcb.successor_operand_counts = g_cnts;
                MLIR_InsertBlockOpAtIndex(ctx, pre_blk, MLIR_CreateOpFromSpec(ctx, &gcb), pins);
                MLIR_DetachOp(ctx, pterm);
            } else {
                MLIR_OpSpec psp; memset(&psp, 0, sizeof(psp));
                psp.type = OP_TYPE_LLVM_BR;
                psp.operands = init_vals; psp.n_operands = hna;
                psp.successors = &h2blk; psp.n_successors = 1;
                psp.successor_operand_counts = &hna32;
                MLIR_OpHandle new_pt = MLIR_CreateOpFromSpec(ctx, &psp);
                MLIR_InsertBlockOpAtIndex(ctx, pre_blk, new_pt, MLIR_GetBlockNumOps(ctx, pre_blk) - 1);
                MLIR_DetachOp(ctx, pterm);
            }
            continue;
        }

        if (complete && MLIR_GetOpNumSuccessorOperands(ctx, hterm, 1) == 0) {
            // Complete unroll: clone all `factor` iterations into pre_blk, bypass loop completely!
            size_t ins_pos = MLIR_GetBlockNumOps(ctx, pre_blk) - 1;
            for (size_t a = 0; a < hna; a++) cur_vals[a] = init_vals[a];
            for (size_t u = 0; u < factor; u++) {
                MLIR_ValueHandle old_v[64], new_v[64];
                size_t n_map = 0;
                for (size_t a = 0; a < hna; a++) {
                    old_v[n_map] = hargs[a];
                    new_v[n_map] = cur_vals[a];
                    n_map++;
                }
                for (size_t j = 0; j < n_body; j++) {
                    MLIR_OpHandle cl = unroll_clone_op(ctx, body_ops[j], old_v, new_v, n_map);
                    MLIR_InsertBlockOpAtIndex(ctx, pre_blk, cl, ins_pos++);
                    if (MLIR_GetOpNumResults(ctx, body_ops[j]) == 1) {
                        old_v[n_map] = MLIR_GetOpResult(ctx, body_ops[j], 0);
                        new_v[n_map] = MLIR_GetOpResult(ctx, cl, 0);
                        n_map++;
                    }
                }
                for (size_t a = 0; a < hna; a++) {
                    MLIR_ValueHandle nv = back_vals[a];
                    for (size_t m = 0; m < n_map; m++) {
                        if (old_v[m] == nv) { nv = new_v[m]; break; }
                    }
                    cur_vals[a] = nv;
                }
            }
            for (size_t a = 0; a < hna; a++) {
                MLIR_ReplaceAllUsesOfValue(ctx, hargs[a], cur_vals[a]);
            }
            uint32_t zero = 0;
            MLIR_OpSpec nbsp;
            memset(&nbsp, 0, sizeof(nbsp));
            nbsp.type = OP_TYPE_LLVM_BR;
            nbsp.successors = &eblk;
            nbsp.n_successors = 1;
            nbsp.successor_operand_counts = &zero;
            MLIR_OpHandle new_pt = MLIR_CreateOpFromSpec(ctx, &nbsp);
            MLIR_InsertBlockOpAtIndex(ctx, pre_blk, new_pt, ins_pos);
            MLIR_DetachOp(ctx, pterm);
            continue;
        }

        // Static partial unroll: clone body_ops (factor - 1) additional times into latch_blk before lterm!
        MLIR_BlockHandle ins_blk = (lblk != MLIR_INVALID_HANDLE) ? lblk : bblk;
        size_t ins_pos = MLIR_GetBlockNumOps(ctx, ins_blk) - 1;

        for (size_t u = 1; u < factor; u++) {
            MLIR_ValueHandle old_v[64], new_v[64];
            size_t n_map = 0;
            for (size_t a = 0; a < hna; a++) {
                old_v[n_map] = hargs[a];
                new_v[n_map] = cur_vals[a];
                n_map++;
            }
            for (size_t j = 0; j < n_body; j++) {
                MLIR_OpHandle cl = unroll_clone_op(ctx, body_ops[j], old_v, new_v, n_map);
                MLIR_InsertBlockOpAtIndex(ctx, ins_blk, cl, ins_pos++);
                if (MLIR_GetOpNumResults(ctx, body_ops[j]) == 1) {
                    old_v[n_map] = MLIR_GetOpResult(ctx, body_ops[j], 0);
                    new_v[n_map] = MLIR_GetOpResult(ctx, cl, 0);
                    n_map++;
                }
            }
            for (size_t a = 0; a < hna; a++) {
                MLIR_ValueHandle nv = back_vals[a];
                for (size_t m = 0; m < n_map; m++) {
                    if (old_v[m] == nv) { nv = new_v[m]; break; }
                }
                cur_vals[a] = nv;
            }
        }

        // Update lterm's successor operands to the final unrolled values!
        uint32_t hna32 = (uint32_t)hna;
        MLIR_OpSpec nbsp;
        memset(&nbsp, 0, sizeof(nbsp));
        nbsp.type = OP_TYPE_LLVM_BR;
        nbsp.operands = cur_vals;
        nbsp.n_operands = hna;
        nbsp.successors = &hblk;
        nbsp.n_successors = 1;
        nbsp.successor_operand_counts = &hna32;
        MLIR_OpHandle new_lt = MLIR_CreateOpFromSpec(ctx, &nbsp);
        MLIR_InsertBlockOpAtIndex(ctx, ins_blk, new_lt, ins_pos);
        MLIR_DetachOp(ctx, lterm);
    }

    free(bhk); free(bhv);
    free(blks); free(pred_cnt); free(pred_a); free(pred_b);
}

// Loop induction-variable strength reduction:
// In a simple 2-block counted loop (`hblk` with preheader `pre_blk` and body
// `bblk`, `iv_next = add(iv, 1)`), replace `mul(iv, step)` (where `step` is
// loop-invariant) with a loop-carried accumulator `sr += step` (eliminates
// `imul` from 2D matrix inner loops like CoreMark `matrix_mul_*`).
static void iv_sr_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 3 || nb > 512) return;

    MLIR_BlockHandle *blks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    uint32_t *pred_cnt = (uint32_t *)calloc(nb, sizeof(uint32_t));
    size_t   *pred_a   = (size_t   *)malloc(nb * sizeof(size_t));
    size_t   *pred_b   = (size_t   *)malloc(nb * sizeof(size_t));
    size_t bhcap = 16;
    while (bhcap < nb * 2) bhcap *= 2;
    uintptr_t *bhk = (uintptr_t *)calloc(bhcap, sizeof(uintptr_t));
    size_t    *bhv = (size_t    *)malloc(bhcap * sizeof(size_t));
    size_t bhmask = bhcap - 1;
    for (size_t b = 0; b < nb; b++) {
        blks[b] = MLIR_GetRegionBlock(ctx, reg, b);
        uintptr_t k = (uintptr_t)blks[b];
        size_t s = (k * 1099511628211ull) & bhmask;
        while (bhk[s]) s = (s + 1) & bhmask;
        bhk[s] = k; bhv[s] = b;
    }
    for (size_t b = 0; b < nb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
        if (no == 0) continue;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, blks[b], no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
        for (size_t si = 0; si < ns; si++) {
            uintptr_t k = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, si);
            size_t s = (k * 1099511628211ull) & bhmask;
            while (bhk[s]) {
                if (bhk[s] == k) {
                    size_t ti = bhv[s];
                    if (pred_cnt[ti] == 0) pred_a[ti] = b;
                    else if (pred_cnt[ti] == 1) pred_b[ti] = b;
                    pred_cnt[ti]++;
                    break;
                }
                s = (s + 1) & bhmask;
            }
        }
    }

    for (size_t h = 1; h < nb; h++) {
        if (pred_cnt[h] != 2) continue;
        MLIR_BlockHandle hblk = blks[h];
        size_t hna = MLIR_GetBlockNumArgs(ctx, hblk);
        if (hna == 0 || hna >= 7) continue;
        size_t hno = MLIR_GetBlockNumOps(ctx, hblk);
        if (hno < 2) continue;
        MLIR_OpHandle hterm = MLIR_GetBlockOp(ctx, hblk, hno - 1);
        if (MLIR_GetOpType(ctx, hterm) != OP_TYPE_LLVM_COND_BR) continue;
        if (MLIR_GetOpNumSuccessorOperands(ctx, hterm, 0) != 0) continue;
        MLIR_BlockHandle bblk = MLIR_GetOpSuccessor(ctx, hterm, 0);
        if (bblk == hblk || MLIR_GetBlockNumArgs(ctx, bblk) != 0) continue;
        size_t bi = nb;
        uintptr_t bk_key = (uintptr_t)bblk;
        size_t s = (bk_key * 1099511628211ull) & bhmask;
        while (bhk[s]) { if (bhk[s] == bk_key) { bi = bhv[s]; break; } s = (s + 1) & bhmask; }
        if (bi >= nb || pred_cnt[bi] != 1) continue;
        size_t bno = MLIR_GetBlockNumOps(ctx, bblk);
        if (bno < 2) continue;
        MLIR_OpHandle bterm = MLIR_GetBlockOp(ctx, bblk, bno - 1);
        if (MLIR_GetOpType(ctx, bterm) != OP_TYPE_LLVM_BR ||
            MLIR_GetOpSuccessor(ctx, bterm, 0) != hblk ||
            MLIR_GetOpNumSuccessorOperands(ctx, bterm, 0) != hna) continue;

        size_t pre_idx = (pred_a[h] == bi) ? pred_b[h] : (pred_b[h] == bi ? pred_a[h] : nb);
        if (pre_idx >= nb) continue;
        MLIR_BlockHandle pre_blk = blks[pre_idx];
        size_t pno = MLIR_GetBlockNumOps(ctx, pre_blk);
        if (pno == 0) continue;
        MLIR_OpHandle pterm = MLIR_GetBlockOp(ctx, pre_blk, pno - 1);
        if (MLIR_GetOpType(ctx, pterm) != OP_TYPE_LLVM_BR ||
            MLIR_GetOpSuccessor(ctx, pterm, 0) != hblk ||
            MLIR_GetOpNumSuccessorOperands(ctx, pterm, 0) != hna) continue;

        // Find an IV block arg with `iv_next = add(iv, 1)`.
        for (size_t a = 0; a < hna; a++) {
            MLIR_ValueHandle iv = MLIR_GetBlockArg(ctx, hblk, a);
            MLIR_TypeHandle iv_ty = MLIR_GetValueType(ctx, iv);
            if (MLIR_GetTypeKind(ctx, iv_ty) != MLIR_TYPE_KIND_INTEGER) continue;
            MLIR_ValueHandle iv_next = MLIR_GetOpSuccessorOperand(ctx, bterm, 0, a);
            MLIR_OpHandle iv_add = MLIR_GetValueDefiningOp(ctx, iv_next);
            if (iv_add == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, iv_add) != OP_TYPE_LLVM_ADD) continue;
            int64_t step_c = 0;
            if (MLIR_GetOpOperand(ctx, iv_add, 0) != iv ||
                !cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, iv_add, 1), &step_c, NULL) ||
                step_c != 1) continue;

            // Scan bblk for `mul(iv, inv)` or `mul(inv, iv)` where `inv` is loop-invariant & non-const.
            for (size_t i = 0; i + 1 < bno; i++) {
                MLIR_OpHandle mop = MLIR_GetBlockOp(ctx, bblk, i);
                if (MLIR_GetOpType(ctx, mop) != OP_TYPE_LLVM_MUL) continue;
                MLIR_ValueHandle m0 = MLIR_GetOpOperand(ctx, mop, 0);
                MLIR_ValueHandle m1 = MLIR_GetOpOperand(ctx, mop, 1);
                MLIR_ValueHandle inv = (m0 == iv) ? m1 : (m1 == iv ? m0 : MLIR_INVALID_HANDLE);
                if (inv == MLIR_INVALID_HANDLE) continue;
                int64_t inv_c = 0;
                if (cf_get_int_const(ctx, inv, &inv_c, NULL)) continue;
                // Check `inv` is not defined in hblk or bblk.
                bool is_inv = true;
                MLIR_OpHandle idef = MLIR_GetValueDefiningOp(ctx, inv);
                if (idef == MLIR_INVALID_HANDLE) {
                    for (size_t ha = 0; ha < hna; ha++)
                        if (MLIR_GetBlockArg(ctx, hblk, ha) == inv) { is_inv = false; break; }
                } else {
                    for (size_t k = 0; k < hno && is_inv; k++)
                        if (MLIR_GetBlockOp(ctx, hblk, k) == idef) is_inv = false;
                    for (size_t k = 0; k < bno && is_inv; k++)
                        if (MLIR_GetBlockOp(ctx, bblk, k) == idef) is_inv = false;
                }
                if (!is_inv) continue;

                // Promote `mul(iv, inv)` to a new block arg `sr_arg`!
                MLIR_ValueHandle sr_arg = MLIR_AddBlockArgument(ctx, hblk, iv_ty, MLIR_INVALID_HANDLE);
                MLIR_ValueHandle iv_init = MLIR_GetOpSuccessorOperand(ctx, pterm, 0, a);
                int64_t init_c = 0;
                MLIR_ValueHandle sr_init;
                size_t p_ins = MLIR_GetBlockNumOps(ctx, pre_blk) - 1;
                if (cf_get_int_const(ctx, iv_init, &init_c, NULL) && init_c == 0) {
                    sr_init = cf_make_int_const(ctx, pre_blk, p_ins, iv_ty, 0);
                } else {
                    MLIR_ValueHandle mops[2] = { iv_init, inv };
                    MLIR_OpSpec msp;
                    memset(&msp, 0, sizeof(msp));
                    msp.type = OP_TYPE_LLVM_MUL;
                    msp.operands = mops; msp.n_operands = 2;
                    msp.result_types = &iv_ty; msp.n_results = 1;
                    MLIR_OpHandle im = MLIR_CreateOpFromSpec(ctx, &msp);
                    MLIR_InsertBlockOpAtIndex(ctx, pre_blk, im, p_ins);
                    sr_init = MLIR_GetOpResult(ctx, im, 0);
                }

                // Update pterm with appended sr_init.
                MLIR_ValueHandle new_pargs[8];
                for (size_t ha = 0; ha < hna; ha++)
                    new_pargs[ha] = MLIR_GetOpSuccessorOperand(ctx, pterm, 0, ha);
                new_pargs[hna] = sr_init;
                MLIR_SetOpSuccessorOperands(ctx, pterm, 0, new_pargs, hna + 1);

                // In bblk before bterm: emit `sr_next = add(sr_arg, inv)` and update bterm.
                MLIR_ValueHandle aops[2] = { sr_arg, inv };
                MLIR_OpSpec asp;
                memset(&asp, 0, sizeof(asp));
                asp.type = OP_TYPE_LLVM_ADD;
                asp.operands = aops; asp.n_operands = 2;
                asp.result_types = &iv_ty; asp.n_results = 1;
                MLIR_OpHandle sr_add = MLIR_CreateOpFromSpec(ctx, &asp);
                MLIR_InsertBlockOpAtIndex(ctx, bblk, sr_add, MLIR_GetBlockNumOps(ctx, bblk) - 1);
                MLIR_ValueHandle new_bargs[8];
                for (size_t ha = 0; ha < hna; ha++)
                    new_bargs[ha] = MLIR_GetOpSuccessorOperand(ctx, bterm, 0, ha);
                new_bargs[hna] = MLIR_GetOpResult(ctx, sr_add, 0);
                MLIR_SetOpSuccessorOperands(ctx, bterm, 0, new_bargs, hna + 1);

                MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetOpResult(ctx, mop, 0), sr_arg);
                MLIR_DetachOp(ctx, mop);
                hna++;
                bno = MLIR_GetBlockNumOps(ctx, bblk);
                break; // one SR per loop to keep register pressure clean
            }
            break;
        }
    }

    free(bhk); free(bhv);
    free(blks); free(pred_cnt); free(pred_a); free(pred_b);
}

// Widen 32-bit loop induction variables (`iv_next = add/sub(iv, +-1)`) used in
// `llvm.getelementptr` / `llvm.sext` to `i64`, eliminating `movsxd` inside hot
// inner loops (e.g. qsort partition loops, sieve outer loop, matrix loops).
static void iv_widen_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb < 3 || nb > 256) return;

    MLIR_BlockHandle *blks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    uint32_t *pred_cnt = (uint32_t *)calloc(nb, sizeof(uint32_t));
    size_t   *pred_a   = (size_t   *)malloc(nb * sizeof(size_t));
    size_t   *pred_b   = (size_t   *)malloc(nb * sizeof(size_t));
    size_t bhcap = 16;
    while (bhcap < nb * 2) bhcap *= 2;
    uintptr_t *bhk = (uintptr_t *)calloc(bhcap, sizeof(uintptr_t));
    size_t    *bhv = (size_t    *)malloc(bhcap * sizeof(size_t));
    size_t bhmask = bhcap - 1;
    for (size_t b = 0; b < nb; b++) {
        blks[b] = MLIR_GetRegionBlock(ctx, reg, b);
        uintptr_t k = (uintptr_t)blks[b];
        size_t s = (k * 1099511628211ull) & bhmask;
        while (bhk[s]) s = (s + 1) & bhmask;
        bhk[s] = k; bhv[s] = b;
    }
    for (size_t b = 0; b < nb; b++) {
        size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
        if (no == 0) continue;
        MLIR_OpHandle term = MLIR_GetBlockOp(ctx, blks[b], no - 1);
        size_t ns = MLIR_GetOpNumSuccessors(ctx, term);
        for (size_t si = 0; si < ns; si++) {
            uintptr_t k = (uintptr_t)MLIR_GetOpSuccessor(ctx, term, si);
            size_t s = (k * 1099511628211ull) & bhmask;
            while (bhk[s]) {
                if (bhk[s] == k) {
                    size_t ti = bhv[s];
                    if (pred_cnt[ti] == 0) pred_a[ti] = b;
                    else if (pred_cnt[ti] == 1) pred_b[ti] = b;
                    pred_cnt[ti]++;
                    break;
                }
                s = (s + 1) & bhmask;
            }
        }
    }

    MLIR_TypeHandle i64_ty = MLIR_CreateTypeInteger(ctx, 64, false);

    for (size_t h = 1; h < nb; h++) {
        if (pred_cnt[h] != 2) continue;
        MLIR_BlockHandle hblk = blks[h];
        size_t hna = MLIR_GetBlockNumArgs(ctx, hblk);
        if (hna == 0 || hna >= 7) continue;
        size_t hno = MLIR_GetBlockNumOps(ctx, hblk);
        if (hno == 0) continue;
        MLIR_OpHandle hterm = MLIR_GetBlockOp(ctx, hblk, hno - 1);
        if (MLIR_GetOpType(ctx, hterm) != OP_TYPE_LLVM_COND_BR) continue;

        // Identify latch (backedge > h, unconditional br to hblk) and preheader.
        size_t latch_idx = nb, pre_idx = nb;
        size_t p_cand[2] = { pred_a[h], pred_b[h] };
        for (int pi = 0; pi < 2; pi++) {
            size_t bi = p_cand[pi];
            size_t bno = MLIR_GetBlockNumOps(ctx, blks[bi]);
            if (bi > h && bno > 0) {
                MLIR_OpHandle bt = MLIR_GetBlockOp(ctx, blks[bi], bno - 1);
                if (MLIR_GetOpType(ctx, bt) == OP_TYPE_LLVM_BR &&
                    MLIR_GetOpSuccessor(ctx, bt, 0) == hblk &&
                    MLIR_GetOpNumSuccessorOperands(ctx, bt, 0) == hna) {
                    latch_idx = bi;
                    pre_idx = p_cand[1 - pi];
                    break;
                }
            }
        }
        if (latch_idx >= nb || pre_idx >= nb || pre_idx == latch_idx) continue;

        MLIR_BlockHandle pre_blk = blks[pre_idx];
        MLIR_BlockHandle latch_blk = blks[latch_idx];
        size_t pno = MLIR_GetBlockNumOps(ctx, pre_blk);
        size_t lno = MLIR_GetBlockNumOps(ctx, latch_blk);
        if (pno == 0 || lno == 0) continue;
        MLIR_OpHandle pterm = MLIR_GetBlockOp(ctx, pre_blk, pno - 1);
        MLIR_OpHandle bterm = MLIR_GetBlockOp(ctx, latch_blk, lno - 1);
        size_t pns = MLIR_GetOpNumSuccessors(ctx, pterm);
        size_t pre_si = pns;
        for (size_t si = 0; si < pns; si++) {
            if (MLIR_GetOpSuccessor(ctx, pterm, si) == hblk) {
                if (pre_si != pns) { pre_si = pns; break; } // duplicate edge
                pre_si = si;
            }
        }
        if (pre_si >= pns || MLIR_GetOpNumSuccessorOperands(ctx, pterm, pre_si) != hna) continue;

        // Check if this is a simple 2-block loop (hblk + latch_blk) with a single-pred exit.
        bool is_two_blk = false;
        size_t exit_idx = nb;
        {
            MLIR_BlockHandle s0 = MLIR_GetOpSuccessor(ctx, hterm, 0);
            MLIR_BlockHandle s1 = MLIR_GetOpSuccessor(ctx, hterm, 1);
            if (s0 == latch_blk && pred_cnt[latch_idx] == 1) {
                is_two_blk = true;
                uintptr_t k = (uintptr_t)s1;
                size_t s = (k * 1099511628211ull) & bhmask;
                while (bhk[s]) { if (bhk[s] == k) { exit_idx = bhv[s]; break; } s = (s + 1) & bhmask; }
            } else if (s1 == latch_blk && pred_cnt[latch_idx] == 1) {
                is_two_blk = true;
                uintptr_t k = (uintptr_t)s0;
                size_t s = (k * 1099511628211ull) & bhmask;
                while (bhk[s]) { if (bhk[s] == k) { exit_idx = bhv[s]; break; } s = (s + 1) & bhmask; }
            }
        }
        bool exit_dom = exit_idx < nb && exit_idx != h && exit_idx != latch_idx &&
                        (pred_cnt[exit_idx] == 1 ||
                         (pred_cnt[exit_idx] == 2 &&
                          ((pred_a[exit_idx] == h && pred_b[exit_idx] > exit_idx) ||
                           (pred_b[exit_idx] == h && pred_a[exit_idx] > exit_idx))));
        bool can_exit_trunc = is_two_blk && exit_dom;

        // Skip pure byte-store memset loops (`for (i=0; i<=N; i++) a[i] = C;`)
        // so the x64 backend's `rep stosb` memset recognizer still matches them.
        if (is_two_blk) {
            bool has_i8_store = false, has_load = false;
            for (size_t li = 0; li < lno; li++) {
                MLIR_OpHandle lop = MLIR_GetBlockOp(ctx, latch_blk, li);
                MLIR_OpType lot = MLIR_GetOpType(ctx, lop);
                if (lot == OP_TYPE_LLVM_LOAD) has_load = true;
                else if (lot == OP_TYPE_LLVM_STORE) {
                    MLIR_TypeHandle st = MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, lop, 0));
                    if (MLIR_GetTypeKind(ctx, st) == MLIR_TYPE_KIND_INTEGER &&
                        MLIR_GetTypeWidth(ctx, st) == 8) has_i8_store = true;
                }
            }
            for (size_t hi = 0; hi < hno; hi++)
                if (MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, hblk, hi)) == OP_TYPE_LLVM_LOAD) has_load = true;
            if (has_i8_store && !has_load) continue;
        }

        for (size_t a = 0; a < hna; a++) {
            MLIR_ValueHandle iv = MLIR_GetBlockArg(ctx, hblk, a);
            MLIR_TypeHandle iv_ty = MLIR_GetValueType(ctx, iv);
            if (MLIR_GetTypeKind(ctx, iv_ty) != MLIR_TYPE_KIND_INTEGER ||
                MLIR_GetTypeWidth(ctx, iv_ty) != 32) continue;

            MLIR_ValueHandle iv_next = MLIR_GetOpSuccessorOperand(ctx, bterm, 0, a);
            MLIR_OpHandle iv_step_op = MLIR_GetValueDefiningOp(ctx, iv_next);
            if (iv_step_op == MLIR_INVALID_HANDLE) continue;
            MLIR_OpType st_ty = MLIR_GetOpType(ctx, iv_step_op);
            if (st_ty != OP_TYPE_LLVM_ADD && st_ty != OP_TYPE_LLVM_SUB) continue;
            // Must be defined in latch_blk and compute iv +- 1.
            bool in_latch = false;
            for (size_t li = 0; li + 1 < lno; li++)
                if (MLIR_GetBlockOp(ctx, latch_blk, li) == iv_step_op) { in_latch = true; break; }
            if (!in_latch) continue;
            int64_t step_c = 0;
            if (MLIR_GetOpOperand(ctx, iv_step_op, 0) != iv ||
                !cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, iv_step_op, 1), &step_c, NULL))
                continue;
            int64_t step = (st_ty == OP_TYPE_LLVM_ADD) ? cf_sext(step_c, 32) : -cf_sext(step_c, 32);
            if (step != 1 && step != -1) continue;

            // Verify all uses of `iv` and `iv_next` across the function.
            bool ok = true;
            size_t gep_or_sext = 0;
            bool has_outside = false;
            for (size_t b = 0; b < nb && ok; b++) {
                size_t no = MLIR_GetBlockNumOps(ctx, blks[b]);
                bool in_2blk = is_two_blk && (b == h || b == latch_idx);
                for (size_t oi = 0; oi < no && ok; oi++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blks[b], oi);
                    MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                    size_t nop = MLIR_GetOpNumOperands(ctx, op);
                    bool uses_iv = false, uses_next = false;
                    for (size_t k = 0; k < nop; k++) {
                        MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, op, k);
                        if (ov == iv) uses_iv = true;
                        if (ov == iv_next) uses_next = true;
                    }
                    size_t nsu = MLIR_GetOpNumSuccessors(ctx, op);
                    bool succ_iv = false, succ_next = false;
                    for (size_t si = 0; si < nsu; si++) {
                        size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, si);
                        for (size_t k = 0; k < nso; k++) {
                            MLIR_ValueHandle ov = MLIR_GetOpSuccessorOperand(ctx, op, si, k);
                            if (ov == iv) succ_iv = true;
                            if (ov == iv_next && !(op == bterm && si == 0 && k == a)) succ_next = true;
                        }
                    }
                    if (uses_next || succ_next) { ok = false; break; }
                    if (!uses_iv && !succ_iv) continue;
                    if (op == iv_step_op) continue;
                    if (is_two_blk && !in_2blk) {
                        if (!can_exit_trunc) { ok = false; break; }
                        has_outside = true;
                        continue;
                    }
                    if (succ_iv) { ok = false; break; }
                    if (ot == OP_TYPE_LLVM_GEP) {
                        if (MLIR_GetOpOperand(ctx, op, 0) == iv) { ok = false; break; }
                        gep_or_sext++;
                    } else if (ot == OP_TYPE_LLVM_SEXT) {
                        MLIR_TypeHandle rt = MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, op, 0));
                        if (MLIR_GetTypeKind(ctx, rt) != MLIR_TYPE_KIND_INTEGER ||
                            MLIR_GetTypeWidth(ctx, rt) != 64) { ok = false; break; }
                        gep_or_sext++;
                    } else if (ot == OP_TYPE_LLVM_ICMP) {
                        MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_PREDICATE);
                        int64_t pred = pa ? MLIR_GetAttributeInteger(ctx, pa) : -1;
                        if (pred < 2 || pred > 5) { ok = false; break; } // signed slt/sle/sgt/sge only
                        MLIR_ValueHandle other = (MLIR_GetOpOperand(ctx, op, 0) == iv)
                            ? MLIR_GetOpOperand(ctx, op, 1) : MLIR_GetOpOperand(ctx, op, 0);
                        int64_t bc = 0;
                        if (!cf_get_int_const(ctx, other, &bc, NULL)) { ok = false; break; }
                        int64_t sbc = cf_sext(bc, 32);
                        if (sbc <= -2147483647ll || sbc >= 2147483647ll) { ok = false; break; }
                    } else {
                        ok = false; break;
                    }
                }
            }
            if (!ok || gep_or_sext == 0) continue;

            // Widen! Add `iv64` block arg to hblk.
            MLIR_ValueHandle iv64 = MLIR_AddBlockArgument(ctx, hblk, i64_ty, MLIR_INVALID_HANDLE);
            // Preheader: iv64_init = sext(iv_init) -> i64
            // If pre_blk is itself a loop header (pred_cnt == 2 with a backedge) and
            // iv_init is not defined in pre_blk, hoist the sext into pre_blk's own entry
            // predecessor so we don't inject instructions into an adjacent inner loop!
            MLIR_ValueHandle iv_init = MLIR_GetOpSuccessorOperand(ctx, pterm, pre_si, a);
            MLIR_BlockHandle sext_blk = pre_blk;
            if (pred_cnt[pre_idx] == 2) {
                size_t pp = nb;
                if (pred_a[pre_idx] < pre_idx && pred_b[pre_idx] > pre_idx) pp = pred_a[pre_idx];
                else if (pred_b[pre_idx] < pre_idx && pred_a[pre_idx] > pre_idx) pp = pred_b[pre_idx];
                if (pp < nb) {
                    bool def_in_pre = false;
                    MLIR_OpHandle idop = MLIR_GetValueDefiningOp(ctx, iv_init);
                    if (idop == MLIR_INVALID_HANDLE) {
                        for (size_t pa = 0; pa < MLIR_GetBlockNumArgs(ctx, pre_blk); pa++)
                            if (MLIR_GetBlockArg(ctx, pre_blk, pa) == iv_init) { def_in_pre = true; break; }
                    } else {
                        for (size_t pi = 0; pi < pno; pi++)
                            if (MLIR_GetBlockOp(ctx, pre_blk, pi) == idop) { def_in_pre = true; break; }
                    }
                    if (!def_in_pre) sext_blk = blks[pp];
                }
            }
            size_t p_ins = MLIR_GetBlockNumOps(ctx, sext_blk) - 1;
            MLIR_OpHandle sterm = MLIR_GetBlockOp(ctx, sext_blk, p_ins);
            if (MLIR_GetOpType(ctx, sterm) == OP_TYPE_LLVM_COND_BR && p_ins > 0) {
                MLIR_OpHandle prev = MLIR_GetBlockOp(ctx, sext_blk, p_ins - 1);
                if (MLIR_GetOpType(ctx, prev) == OP_TYPE_LLVM_ICMP &&
                    MLIR_GetOpResult(ctx, prev, 0) == MLIR_GetOpOperand(ctx, sterm, 0))
                    p_ins--;
            }
            MLIR_ValueHandle iv64_init;
            int64_t init_c = 0;
            if (cf_get_int_const(ctx, iv_init, &init_c, NULL)) {
                iv64_init = cf_make_int_const(ctx, sext_blk, p_ins, i64_ty, cf_sext(init_c, 32));
            } else {
                MLIR_OpSpec ssp;
                memset(&ssp, 0, sizeof(ssp));
                ssp.type = OP_TYPE_LLVM_SEXT;
                ssp.operands = &iv_init; ssp.n_operands = 1;
                ssp.result_types = &i64_ty; ssp.n_results = 1;
                MLIR_OpHandle sop = MLIR_CreateOpFromSpec(ctx, &ssp);
                MLIR_InsertBlockOpAtIndex(ctx, sext_blk, sop, p_ins);
                iv64_init = MLIR_GetOpResult(ctx, sop, 0);
            }
            {
                MLIR_ValueHandle new_pargs[8];
                for (size_t ha = 0; ha < hna; ha++)
                    new_pargs[ha] = MLIR_GetOpSuccessorOperand(ctx, pterm, pre_si, ha);
                new_pargs[hna] = iv64_init;
                MLIR_SetOpSuccessorOperands(ctx, pterm, pre_si, new_pargs, hna + 1);
            }
            // Latch: iv64_next = add(iv64, step) -> i64
            size_t l_ins = MLIR_GetBlockNumOps(ctx, latch_blk) - 1;
            MLIR_ValueHandle c_step = cf_make_int_const(ctx, latch_blk, l_ins, i64_ty, step);
            MLIR_ValueHandle aops[2] = { iv64, c_step };
            MLIR_OpSpec asp;
            memset(&asp, 0, sizeof(asp));
            asp.type = OP_TYPE_LLVM_ADD;
            asp.operands = aops; asp.n_operands = 2;
            asp.result_types = &i64_ty; asp.n_results = 1;
            MLIR_OpHandle add64 = MLIR_CreateOpFromSpec(ctx, &asp);
            MLIR_InsertBlockOpAtIndex(ctx, latch_blk, add64, l_ins + 1);
            {
                MLIR_ValueHandle new_bargs[8];
                for (size_t ha = 0; ha < hna; ha++)
                    new_bargs[ha] = (ha == a) ? iv : MLIR_GetOpSuccessorOperand(ctx, bterm, 0, ha);
                new_bargs[hna] = MLIR_GetOpResult(ctx, add64, 0);
                MLIR_SetOpSuccessorOperands(ctx, bterm, 0, new_bargs, hna + 1);
            }
            MLIR_DetachOp(ctx, iv_step_op);

            // If there are outside uses (2-block loop), emit `trunc i64 iv64 -> i32` in exit_blk
            // (or in exit_blk's single-pred exit if exit_blk is itself a loop header that doesn't use iv!).
            MLIR_ValueHandle iv_trunc = MLIR_INVALID_HANDLE;
            if (has_outside) {
                MLIR_BlockHandle trunc_blk = blks[exit_idx];
                if (pred_cnt[exit_idx] == 2) {
                    bool used_in_exit = false;
                    size_t eno = MLIR_GetBlockNumOps(ctx, trunc_blk);
                    for (size_t ei = 0; ei < eno && !used_in_exit; ei++) {
                        MLIR_OpHandle eop = MLIR_GetBlockOp(ctx, trunc_blk, ei);
                        for (size_t k = 0; k < MLIR_GetOpNumOperands(ctx, eop); k++)
                            if (MLIR_GetOpOperand(ctx, eop, k) == iv) used_in_exit = true;
                        for (size_t si = 0; si < MLIR_GetOpNumSuccessors(ctx, eop); si++)
                            for (size_t k = 0; k < MLIR_GetOpNumSuccessorOperands(ctx, eop, si); k++)
                                if (MLIR_GetOpSuccessorOperand(ctx, eop, si, k) == iv) used_in_exit = true;
                    }
                    if (!used_in_exit && eno > 0) {
                        MLIR_OpHandle eterm = MLIR_GetBlockOp(ctx, trunc_blk, eno - 1);
                        if (MLIR_GetOpType(ctx, eterm) == OP_TYPE_LLVM_COND_BR) {
                            for (size_t si = 0; si < 2; si++) {
                                MLIR_BlockHandle es = MLIR_GetOpSuccessor(ctx, eterm, si);
                                size_t esi = nb;
                                uintptr_t ek = (uintptr_t)es;
                                size_t eslot = (ek * 1099511628211ull) & bhmask;
                                while (bhk[eslot]) { if (bhk[eslot] == ek) { esi = bhv[eslot]; break; } eslot = (eslot + 1) & bhmask; }
                                if (esi < nb && pred_cnt[esi] == 1 && esi != pred_a[exit_idx] && esi != pred_b[exit_idx]) {
                                    trunc_blk = es;
                                    break;
                                }
                            }
                        }
                    }
                }
                MLIR_OpSpec tsp;
                memset(&tsp, 0, sizeof(tsp));
                tsp.type = OP_TYPE_LLVM_TRUNC;
                tsp.operands = &iv64; tsp.n_operands = 1;
                tsp.result_types = &iv_ty; tsp.n_results = 1;
                MLIR_OpHandle top = MLIR_CreateOpFromSpec(ctx, &tsp);
                MLIR_InsertBlockOpAtIndex(ctx, trunc_blk, top, 0);
                iv_trunc = MLIR_GetOpResult(ctx, top, 0);
            }

            // Rewrite all uses of `iv`.
            for (size_t b = 0; b < nb; b++) {
                bool in_2blk = is_two_blk && (b == h || b == latch_idx);
                for (size_t oi = 0; oi < MLIR_GetBlockNumOps(ctx, blks[b]); ) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blks[b], oi);
                    if (op == bterm) { oi++; continue; }
                    MLIR_OpType ot = MLIR_GetOpType(ctx, op);
                    if (is_two_blk && !in_2blk) {
                        size_t nop = MLIR_GetOpNumOperands(ctx, op);
                        for (size_t k = 0; k < nop; k++)
                            if (MLIR_GetOpOperand(ctx, op, k) == iv)
                                MLIR_SetOpOperand(ctx, op, k, iv_trunc);
                        size_t nsu = MLIR_GetOpNumSuccessors(ctx, op);
                        for (size_t si = 0; si < nsu; si++) {
                            size_t nso = MLIR_GetOpNumSuccessorOperands(ctx, op, si);
                            bool ch = false;
                            for (size_t k = 0; k < nso; k++)
                                if (MLIR_GetOpSuccessorOperand(ctx, op, si, k) == iv) ch = true;
                            if (ch) {
                                MLIR_ValueHandle *sv = (MLIR_ValueHandle *)malloc(nso * sizeof(MLIR_ValueHandle));
                                for (size_t k = 0; k < nso; k++) {
                                    MLIR_ValueHandle ov = MLIR_GetOpSuccessorOperand(ctx, op, si, k);
                                    sv[k] = (ov == iv) ? iv_trunc : ov;
                                }
                                MLIR_SetOpSuccessorOperands(ctx, op, si, sv, nso);
                                free(sv);
                            }
                        }
                        oi++; continue;
                    }
                    if (ot == OP_TYPE_LLVM_GEP) {
                        size_t nop = MLIR_GetOpNumOperands(ctx, op);
                        for (size_t k = 1; k < nop; k++)
                            if (MLIR_GetOpOperand(ctx, op, k) == iv)
                                MLIR_SetOpOperand(ctx, op, k, iv64);
                        oi++; continue;
                    }
                    if (ot == OP_TYPE_LLVM_SEXT && MLIR_GetOpOperand(ctx, op, 0) == iv) {
                        MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetOpResult(ctx, op, 0), iv64);
                        MLIR_DetachOp(ctx, op);
                        continue;
                    }
                    if (ot == OP_TYPE_LLVM_ICMP &&
                        (MLIR_GetOpOperand(ctx, op, 0) == iv || MLIR_GetOpOperand(ctx, op, 1) == iv)) {
                        size_t iv_pos = (MLIR_GetOpOperand(ctx, op, 0) == iv) ? 0 : 1;
                        int64_t bc = 0;
                        cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, op, 1 - iv_pos), &bc, NULL);
                        MLIR_ValueHandle c64 = cf_make_int_const(ctx, blks[b], oi, i64_ty, cf_sext(bc, 32));
                        MLIR_SetOpOperand(ctx, op, iv_pos, iv64);
                        MLIR_SetOpOperand(ctx, op, 1 - iv_pos, c64);
                        oi += 2; continue;
                    }
                    oi++;
                }
            }
            hna++;
            break;
        }
    }

    free(bhk); free(bhv);
    free(blks); free(pred_cnt); free(pred_a); free(pred_b);
}

void mlir_llvm_unroll(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_UNROLL")) return;
    if (MLIR_GetOpNumRegions(ctx, module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle fn = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) continue;
        iv_sr_func(ctx, fn);
        iv_widen_func(ctx, fn);
        constfold_func(ctx, fn);
        unroll_func(ctx, fn);
    }
}

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
} InlCand;

static bool inl_check_candidate(MLIR_Context *ctx, MLIR_OpHandle fn, InlCand *out) {
    if (MLIR_GetOpType(ctx, fn) != OP_TYPE_LLVM_FUNC) return false;
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return false;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    if (nb == 0 || nb > 12) return false;
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
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        total_ops += no;
        if (total_ops > 90) return false;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpNumRegions(ctx, op) > 0) return false;
            MLIR_OpType t = MLIR_GetOpType(ctx, op);
            if (t == OP_TYPE_LLVM_INTR_VASTART || t == OP_TYPE_LLVM_INTR_VAEND ||
                t == OP_TYPE_LLVM_INTR_VACOPY)
                return false;
            if (t == OP_TYPE_LLVM_ALLOCA) has_alloca = true;
            if (t == OP_TYPE_LLVM_CALL || t == OP_TYPE_FUNC_CALL) {
                has_call = true;
                string cn = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE));
                if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE_NAME));
                if (opt_str_eq(cn, name)) return false; // skip directly recursive callee
            }
        }
    }
    // Non-leaf functions use the tighter budget (<= 8 blocks, <= 50 ops);
    // leaf functions (like crcu8) get up to 12 blocks / 90 ops.
    if (has_call && (nb > 8 || total_ops > 50)) return false;
    out->name = name;
    out->fn = fn;
    out->n_ops = total_ops;
    out->n_blocks = nb;
    out->has_alloca = has_alloca;
    return true;
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

    for (int round = 0; round < 2; round++) {
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
            if (caller_ops > 1500) continue;

            int n_inlined = 0;
            for (size_t b = 0; b < MLIR_GetRegionNumBlocks(ctx, creg) && caller_ops <= 1500 && n_inlined < 16; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, creg, b);
                for (size_t i = 0; i < MLIR_GetBlockNumOps(ctx, bk) && caller_ops <= 1500 && n_inlined < 16; i++) {
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
        if (!inlined_any) break;
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

static void constfold_func(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return;
    MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
    for (int round = 0; round < 2; round++) {
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
                    if (rk == MLIR_TYPE_KIND_INTEGER &&
                        cf_get_int_const(ctx, MLIR_GetOpOperand(ctx, op, 0), &c0, &sw)) {
                        uint32_t rw = MLIR_GetTypeWidth(ctx, rty);
                        int64_t rv = (ot == OP_TYPE_LLVM_SEXT) ? cf_mask(cf_sext(c0, sw), rw)
                                                               : cf_mask(c0, rw);
                        MLIR_ValueHandle nv = cf_make_int_const(ctx, bk, i, rty, rv);
                        MLIR_ReplaceAllUsesOfValue(ctx, res, nv);
                        MLIR_DetachOp(ctx, op);
                        changed = true;
                        continue;
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
    }
}

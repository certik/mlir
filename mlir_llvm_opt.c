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
    if (nb == 0 || nb > 8) return false;
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
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(ctx, reg, b);
        size_t no = MLIR_GetBlockNumOps(ctx, bk);
        total_ops += no;
        if (total_ops > 50) return false;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bk, i);
            if (MLIR_GetOpNumRegions(ctx, op) > 0) return false;
            MLIR_OpType t = MLIR_GetOpType(ctx, op);
            if (t == OP_TYPE_LLVM_INTR_VASTART || t == OP_TYPE_LLVM_INTR_VAEND ||
                t == OP_TYPE_LLVM_INTR_VACOPY)
                return false;
            if (t == OP_TYPE_LLVM_ALLOCA) has_alloca = true;
            if (t == OP_TYPE_LLVM_CALL || t == OP_TYPE_FUNC_CALL) {
                string cn = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE));
                if (!cn.size) cn = opt_strip_at(opt_attr_str(ctx, op, MLIR_KEY_CALLEE_NAME));
                if (opt_str_eq(cn, name)) return false; // skip directly recursive callee
            }
        }
    }
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
        simplify_cfg_func(ctx, fn);
    }
}

// Port of LFortran's `llvm_reductions.cpp` (`recombine_horizontal_reductions`)
// onto the public MLIR C API (doc/mlir_api_design.md §4 & §7 step 15).
//
// Rewrites SLP-transposed 3-component sum-of-squares over four `<4 x double>`
// vectors into AVX `vhaddpd` horizontal reductions, using a greatest-fixed-point
// `LaneZero` analysis to omit lane-3 masking when lane 3 is provably zero.

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <base/arena.h>
#include <base/string.h>
#include "pass_llvm_reductions.h"

static bool is_four_doubles(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_GetTypeKind(ctx, ty) != MLIR_TYPE_KIND_VECTOR) return false;
    if (MLIR_GetTypeNumElements(ctx, ty) != 4) return false;
    MLIR_TypeHandle elem = MLIR_GetTypeElement(ctx, ty);
    return MLIR_GetTypeKind(ctx, elem) == MLIR_TYPE_KIND_FLOAT &&
           MLIR_GetTypeWidth(ctx, elem) == 64;
}

static bool is_fast(MLIR_Context *ctx, MLIR_OpHandle op) {
    return MLIR_GetOpFastMath(ctx, op) == MLIR_FMF_FAST;
}

static uint32_t vec_width(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_VECTOR)
        return (uint32_t)MLIR_GetTypeNumElements(ctx, ty);
    return 1;
}

static bool is_fp_type(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_VECTOR)
        ty = MLIR_GetTypeElement(ctx, ty);
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    return k == MLIR_TYPE_KIND_FLOAT || k == MLIR_TYPE_KIND_BF16;
}

static bool match_gather(MLIR_Context *ctx, MLIR_ValueHandle v, uint32_t lane,
                         MLIR_ValueHandle bases[4]) {
    if (MLIR_GetValueKind(ctx, v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle g3 = MLIR_GetValueDefiningOp(ctx, v);
    if (MLIR_GetOpType(ctx, g3) != OP_TYPE_LLVM_SHUFFLEVECTOR) return false;
    MLIR_TypeHandle vty = MLIR_GetValueType(ctx, v);
    if (!is_four_doubles(ctx, vty)) return false;

    MLIR_AttributeHandle m3 = MLIR_GetOpAttr(ctx, g3, MLIR_KEY_MASK);
    if (m3 == MLIR_INVALID_HANDLE || MLIR_GetAttributeDenseSize(ctx, m3) != 4) return false;
    if (MLIR_GetAttributeDenseElement(ctx, m3, 0) != 0 ||
        MLIR_GetAttributeDenseElement(ctx, m3, 1) != 1 ||
        MLIR_GetAttributeDenseElement(ctx, m3, 2) != 2 ||
        MLIR_GetAttributeDenseElement(ctx, m3, 3) != (int64_t)lane + 4)
        return false;

    MLIR_ValueHandle g2v = MLIR_GetOpOperand(ctx, g3, 0);
    if (MLIR_GetValueKind(ctx, g2v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle g2 = MLIR_GetValueDefiningOp(ctx, g2v);
    if (MLIR_GetOpType(ctx, g2) != OP_TYPE_LLVM_SHUFFLEVECTOR ||
        MLIR_GetValueType(ctx, g2v) != vty)
        return false;
    MLIR_AttributeHandle m2 = MLIR_GetOpAttr(ctx, g2, MLIR_KEY_MASK);
    if (m2 == MLIR_INVALID_HANDLE || MLIR_GetAttributeDenseSize(ctx, m2) != 4) return false;
    if (MLIR_GetAttributeDenseElement(ctx, m2, 0) != 0 ||
        MLIR_GetAttributeDenseElement(ctx, m2, 1) != 1 ||
        MLIR_GetAttributeDenseElement(ctx, m2, 2) != (int64_t)lane + 4 ||
        MLIR_GetAttributeDenseElement(ctx, m2, 3) != -1)
        return false;

    MLIR_ValueHandle g1v = MLIR_GetOpOperand(ctx, g2, 0);
    if (MLIR_GetValueKind(ctx, g1v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle g1 = MLIR_GetValueDefiningOp(ctx, g1v);
    if (MLIR_GetOpType(ctx, g1) != OP_TYPE_LLVM_SHUFFLEVECTOR ||
        MLIR_GetValueType(ctx, g1v) != vty)
        return false;
    MLIR_AttributeHandle m1 = MLIR_GetOpAttr(ctx, g1, MLIR_KEY_MASK);
    if (m1 == MLIR_INVALID_HANDLE || MLIR_GetAttributeDenseSize(ctx, m1) != 4) return false;
    if (MLIR_GetAttributeDenseElement(ctx, m1, 0) != (int64_t)lane ||
        MLIR_GetAttributeDenseElement(ctx, m1, 1) != (int64_t)lane + 4 ||
        MLIR_GetAttributeDenseElement(ctx, m1, 2) != -1 ||
        MLIR_GetAttributeDenseElement(ctx, m1, 3) != -1)
        return false;

    bases[0] = MLIR_GetOpOperand(ctx, g1, 0);
    bases[1] = MLIR_GetOpOperand(ctx, g1, 1);
    bases[2] = MLIR_GetOpOperand(ctx, g2, 1);
    bases[3] = MLIR_GetOpOperand(ctx, g3, 1);
    for (int i = 0; i < 4; i++) {
        if (!is_four_doubles(ctx, MLIR_GetValueType(ctx, bases[i]))) return false;
    }
    return true;
}

static bool match_square(MLIR_Context *ctx, MLIR_ValueHandle v, MLIR_ValueHandle *out_x) {
    if (MLIR_GetValueKind(ctx, v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle op = MLIR_GetValueDefiningOp(ctx, v);
    if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_FMUL || !is_fast(ctx, op)) return false;
    MLIR_ValueHandle a = MLIR_GetOpOperand(ctx, op, 0);
    MLIR_ValueHandle b = MLIR_GetOpOperand(ctx, op, 1);
    if (a != b) return false;
    *out_x = a;
    return true;
}

static bool collect_terms(MLIR_Context *ctx, MLIR_ValueHandle v,
                          MLIR_ValueHandle *terms, size_t *n_terms, size_t max_terms,
                          MLIR_ValueHandle *seen, size_t *n_seen, size_t max_seen) {
    for (size_t i = 0; i < *n_seen; i++) {
        if (seen[i] == v) return false;
    }
    if (*n_seen >= max_seen) return false;
    seen[(*n_seen)++] = v;

    if (MLIR_GetValueKind(ctx, v) == MLIR_VALUE_OP_RESULT) {
        MLIR_OpHandle op = MLIR_GetValueDefiningOp(ctx, v);
        if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FADD && is_fast(ctx, op)) {
            return collect_terms(ctx, MLIR_GetOpOperand(ctx, op, 0), terms, n_terms, max_terms, seen, n_seen, max_seen) &&
                   collect_terms(ctx, MLIR_GetOpOperand(ctx, op, 1), terms, n_terms, max_terms, seen, n_seen, max_seen);
        }
    }
    if (*n_terms >= max_terms) return false;
    terms[(*n_terms)++] = v;
    return true;
}

// -----------------------------------------------------------------------------
// LaneZero: greatest-fixed-point analysis of whether lane `lane` of `v` is 0.0
// -----------------------------------------------------------------------------

typedef struct {
    MLIR_ValueHandle v;
    uint32_t lane;
} Elem;

typedef struct {
    Elem key;
    Elem *operands;
    size_t n_operands;
    bool any;
    bool zero;
} LZNode;

typedef struct {
    MLIR_Context *ctx;
    Arena *arena;
    LZNode *nodes;
    size_t n_nodes;
    size_t cap_nodes;
} LaneZero;

static int lz_find(const LaneZero *lz, Elem e) {
    for (size_t i = 0; i < lz->n_nodes; i++) {
        if (lz->nodes[i].key.v == e.v && lz->nodes[i].key.lane == e.lane)
            return (int)i;
    }
    return -1;
}

static bool lz_nonfinite_const(MLIR_Context *ctx, MLIR_ValueHandle v, uint32_t lane) {
    double d = 0.0;
    if (!MLIR_MatchConstantFloat(ctx, v, (int64_t)lane, &d)) return false;
    return isnan(d) || isinf(d);
}

static bool lz_rule(MLIR_Context *ctx, Arena *arena, Elem e, LZNode *node) {
    node->operands = NULL;
    node->n_operands = 0;
    node->any = false;

    if (MLIR_GetValueKind(ctx, e.v) == MLIR_VALUE_BLOCK_ARG) {
        MLIR_BlockHandle blk = MLIR_GetValueParentBlock(ctx, e.v);
        size_t arg_idx = MLIR_GetValueResultIndex(ctx, e.v);
        size_t np = 0;
        for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, blk); p; p = MLIR_GetPredNext(ctx, p))
            np++;
        if (np == 0) return false;
        node->operands = arena_new_array(arena, Elem, np);
        size_t i = 0;
        for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, blk); p; p = MLIR_GetPredNext(ctx, p)) {
            node->operands[i].v = MLIR_GetIncomingValue(ctx, p, arg_idx);
            node->operands[i].lane = e.lane;
            i++;
        }
        node->n_operands = np;
        return true;
    }

    MLIR_OpHandle op = MLIR_GetValueDefiningOp(ctx, e.v);
    if (op == MLIR_INVALID_HANDLE) return false;
    MLIR_OpType t = MLIR_GetOpType(ctx, op);

    if (t == OP_TYPE_LLVM_FADD || t == OP_TYPE_LLVM_FSUB) {
        node->operands = arena_new_array(arena, Elem, 2);
        node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 0), e.lane };
        node->operands[1] = (Elem){ MLIR_GetOpOperand(ctx, op, 1), e.lane };
        node->n_operands = 2;
        return true;
    }
    if (t == OP_TYPE_LLVM_FMUL) {
        MLIR_ValueHandle a = MLIR_GetOpOperand(ctx, op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(ctx, op, 1);
        bool unsafe = lz_nonfinite_const(ctx, a, e.lane) || lz_nonfinite_const(ctx, b, e.lane);
        node->any = is_fast(ctx, op) && !unsafe;
        node->operands = arena_new_array(arena, Elem, 2);
        node->operands[0] = (Elem){ a, e.lane };
        node->operands[1] = (Elem){ b, e.lane };
        node->n_operands = 2;
        return true;
    }
    if (t == OP_TYPE_LLVM_FNEG) {
        node->operands = arena_new_array(arena, Elem, 1);
        node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 0), e.lane };
        node->n_operands = 1;
        return true;
    }
    if (t == OP_TYPE_LLVM_SELECT) {
        node->operands = arena_new_array(arena, Elem, 2);
        node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 1), e.lane };
        node->operands[1] = (Elem){ MLIR_GetOpOperand(ctx, op, 2), e.lane };
        node->n_operands = 2;
        return true;
    }
    if (t == OP_TYPE_LLVM_SHUFFLEVECTOR) {
        MLIR_AttributeHandle mask = MLIR_GetOpAttr(ctx, op, MLIR_KEY_MASK);
        if (mask == MLIR_INVALID_HANDLE || e.lane >= MLIR_GetAttributeDenseSize(ctx, mask))
            return false;
        int64_t m = MLIR_GetAttributeDenseElement(ctx, mask, e.lane);
        if (m < 0) return false;
        MLIR_ValueHandle op0 = MLIR_GetOpOperand(ctx, op, 0);
        uint32_t w = vec_width(ctx, MLIR_GetValueType(ctx, op0));
        node->operands = arena_new_array(arena, Elem, 1);
        node->operands[0].v = (m < (int64_t)w) ? op0 : MLIR_GetOpOperand(ctx, op, 1);
        node->operands[0].lane = (uint32_t)(m % w);
        node->n_operands = 1;
        return true;
    }
    if (t == OP_TYPE_LLVM_INSERTELEMENT) {
        int64_t pos = -1;
        if (!MLIR_MatchConstantInt(ctx, MLIR_GetOpOperand(ctx, op, 2), -1, &pos) || pos < 0)
            return false;
        node->operands = arena_new_array(arena, Elem, 1);
        if ((uint32_t)pos == e.lane) {
            node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 1), 0 };
        } else {
            node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 0), e.lane };
        }
        node->n_operands = 1;
        return true;
    }
    if (t == OP_TYPE_LLVM_EXTRACTELEMENT) {
        int64_t pos = -1;
        if (e.lane != 0 || !MLIR_MatchConstantInt(ctx, MLIR_GetOpOperand(ctx, op, 1), -1, &pos) || pos < 0)
            return false;
        node->operands = arena_new_array(arena, Elem, 1);
        node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 0), (uint32_t)pos };
        node->n_operands = 1;
        return true;
    }
    if (t == OP_TYPE_LLVM_BITCAST) {
        MLIR_TypeHandle dt = MLIR_GetValueType(ctx, e.v);
        MLIR_TypeHandle st = MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, op, 0));
        if (vec_width(ctx, dt) != vec_width(ctx, st) || !is_fp_type(ctx, dt) || !is_fp_type(ctx, st))
            return false;
        node->operands = arena_new_array(arena, Elem, 1);
        node->operands[0] = (Elem){ MLIR_GetOpOperand(ctx, op, 0), e.lane };
        node->n_operands = 1;
        return true;
    }
    return false;
}

static bool lz_value_of(const LaneZero *lz, Elem e) {
    int idx = lz_find(lz, e);
    if (idx < 0) return MLIR_IsConstantZero(lz->ctx, e.v, (int64_t)e.lane);
    return lz->nodes[idx].zero;
}

static bool lz_holds(const LaneZero *lz, const LZNode *node) {
    for (size_t i = 0; i < node->n_operands; i++) {
        if (lz_value_of(lz, node->operands[i]) == node->any)
            return node->any;
    }
    return !node->any && node->n_operands > 0;
}

static bool lz_is_zero(MLIR_Context *ctx, Arena *arena, MLIR_ValueHandle v, uint32_t lane) {
    if (MLIR_IsConstantZero(ctx, v, (int64_t)lane)) return true;

    LaneZero lz;
    lz.ctx = ctx;
    lz.arena = arena;
    lz.cap_nodes = 32;
    lz.n_nodes = 0;
    lz.nodes = arena_new_array(arena, LZNode, lz.cap_nodes);

    Elem work_stack[64];
    size_t wlen = 0;
    work_stack[wlen++] = (Elem){ v, lane };

    while (wlen > 0) {
        Elem cur = work_stack[--wlen];
        if (MLIR_IsConstantZero(ctx, cur.v, (int64_t)cur.lane)) continue;
        if (lz_find(&lz, cur) >= 0) continue;
        if (lz.n_nodes == lz.cap_nodes) {
            size_t nc = lz.cap_nodes * 2;
            LZNode *nn = arena_new_array(arena, LZNode, nc);
            memcpy(nn, lz.nodes, lz.n_nodes * sizeof(LZNode));
            lz.nodes = nn;
            lz.cap_nodes = nc;
        }
        LZNode node;
        node.key = cur;
        node.zero = true;
        if (!lz_rule(ctx, arena, cur, &node)) {
            node.operands = NULL;
            node.n_operands = 0;
            node.any = false;
        }
        lz.nodes[lz.n_nodes++] = node;
        for (size_t i = 0; i < node.n_operands && wlen < 64; i++) {
            work_stack[wlen++] = node.operands[i];
        }
    }

    int root_idx = lz_find(&lz, (Elem){ v, lane });
    if (root_idx < 0) return false;

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < lz.n_nodes; i++) {
            if (lz.nodes[i].zero && !lz_holds(&lz, &lz.nodes[i])) {
                lz.nodes[i].zero = false;
                changed = true;
            }
        }
    }
    return lz.nodes[root_idx].zero;
}

static MLIR_ValueHandle build_shuffle4(MLIR_Builder *b, MLIR_TypeHandle vty,
                                       MLIR_ValueHandle a, MLIR_ValueHandle c,
                                       int32_t m0, int32_t m1, int32_t m2, int32_t m3) {
    int32_t mask[4];
    mask[0] = m0; mask[1] = m1; mask[2] = m2; mask[3] = m3;
    MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_MASK,
        MLIR_CreateAttributeDenseI32Array(b->ctx, mask, 4));
    MLIR_ValueHandle ops[2];
    ops[0] = a;
    ops[1] = c;
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = OP_TYPE_LLVM_SHUFFLEVECTOR;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &vty;
    s.n_results = 1;
    s.attrs = &attr;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

bool mlir_pass_llvm_reductions_run(MLIR_PassContext *pc, MLIR_OpHandle fn, void *state) {
    (void)state;
    MLIR_Context *ctx = pc->ctx;
    if (!MLIR_TargetIsX86(pc->target) || MLIR_IsFunctionDeclaration(ctx, fn) ||
        !MLIR_TargetHasFeature(ctx, pc->target, fn, MLIR_FEATURE_AVX))
        return true;

    MLIR_RegionHandle region = MLIR_GetOpRegion(ctx, fn, 0);
    MLIR_OpHandle roots[64];
    size_t n_roots = 0;

    for (MLIR_BlockHandle blk = MLIR_GetRegionFirstBlock(ctx, region);
         blk != MLIR_INVALID_HANDLE; blk = MLIR_GetBlockNext(ctx, blk)) {
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, blk);
             op != MLIR_INVALID_HANDLE; op = MLIR_GetOpNext(ctx, op)) {
            if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_FADD || !is_fast(ctx, op))
                continue;
            MLIR_ValueHandle res = MLIR_GetOpResult(ctx, op, 0);
            if (!is_four_doubles(ctx, MLIR_GetValueType(ctx, res))) continue;
            MLIR_ValueHandle terms[4], seen[16];
            size_t nt = 0, ns = 0;
            if (collect_terms(ctx, res, terms, &nt, 4, seen, &ns, 16) && nt == 3) {
                if (n_roots < 64) roots[n_roots++] = op;
            }
        }
    }

    for (size_t ri = 0; ri < n_roots; ri++) {
        MLIR_OpHandle root = roots[ri];
        MLIR_ValueHandle root_val = MLIR_GetOpResult(ctx, root, 0);
        MLIR_ValueHandle terms[4], seen[16];
        size_t nt = 0, ns = 0;
        if (!collect_terms(ctx, root_val, terms, &nt, 4, seen, &ns, 16) || nt != 3)
            continue;

        MLIR_ValueHandle bases[4] = { MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
                                      MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE };
        bool used_lane[3] = { false, false, false };
        bool ok = true;
        for (uint32_t k = 0; k < 3 && ok; k++) {
            MLIR_ValueHandle x = MLIR_INVALID_HANDLE;
            if (!match_square(ctx, terms[k], &x)) { ok = false; break; }
            bool matched = false;
            for (uint32_t lane = 0; lane < 3 && !matched; lane++) {
                if (used_lane[lane]) continue;
                MLIR_ValueHandle cand[4];
                if (!match_gather(ctx, x, lane, cand)) continue;
                if (bases[0] == MLIR_INVALID_HANDLE) {
                    for (int j = 0; j < 4; j++) bases[j] = cand[j];
                    matched = true;
                } else {
                    bool same = true;
                    for (int j = 0; j < 4; j++)
                        if (bases[j] != cand[j]) same = false;
                    if (same) matched = true;
                }
                if (matched) used_lane[lane] = true;
            }
            if (!matched) ok = false;
        }
        if (!ok) continue;

        MLIR_Builder b = MLIR_BuilderBefore(ctx, root);
        MLIR_FastMath fmf = MLIR_GetOpFastMath(ctx, root);
        MLIR_TypeHandle vty = MLIR_GetValueType(ctx, bases[0]);

        MLIR_ValueHandle sq[4];
        for (int j = 0; j < 4; j++) {
            sq[j] = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, vty, bases[j], bases[j], fmf);
        }

        bool all_lane3_zero = true;
        for (int j = 0; j < 4; j++) {
            if (!lz_is_zero(ctx, pc->scratch, bases[j], 3)) {
                all_lane3_zero = false;
                break;
            }
        }
        if (!all_lane3_zero) {
            MLIR_ValueHandle zeros = MLIR_BuildZero(&b, vty);
            for (int j = 0; j < 4; j++) {
                sq[j] = build_shuffle4(&b, vty, sq[j], zeros, 0, 1, 2, 4);
            }
        }

        MLIR_ValueHandle ha0[2] = { sq[0], sq[1] };
        MLIR_ValueHandle ha1[2] = { sq[2], sq[3] };
        MLIR_ValueHandle t0 = MLIR_BuildCallIntrinsic(&b, str_lit("llvm.x86.avx.hadd.pd.256"),
                                                      vty, ha0, 2, fmf);
        MLIR_ValueHandle t1 = MLIR_BuildCallIntrinsic(&b, str_lit("llvm.x86.avx.hadd.pd.256"),
                                                      vty, ha1, 2, fmf);
        MLIR_ValueHandle y0 = build_shuffle4(&b, vty, t0, t1, 2, 3, 4, 5);
        MLIR_ValueHandle y1 = build_shuffle4(&b, vty, t0, t1, 0, 1, 6, 7);
        MLIR_ValueHandle sum = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, vty, y0, y1, fmf);
        MLIR_ReplaceAllUsesOfValue(ctx, root_val, sum);
    }
    return true;
}

MLIR_Pass MLIR_PassLLVMReductions(void) {
    MLIR_Pass p;
    p.name = str_lit("llvm-reductions");
    p.anchor = MLIR_ANCHOR_FUNCTION;
    p.run = mlir_pass_llvm_reductions_run;
    p.state = NULL;
    return p;
}

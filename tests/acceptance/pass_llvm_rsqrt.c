// Port of LFortran's `llvm_rsqrt.cpp` (`expand_reciprocal_sqrt`) onto the
// public MLIR C API (doc/mlir_api_design.md §4 & §7 step 15).
//
// Expands fast `a / sqrt(x)` into a Newton-refined reciprocal square root
// estimate with the constant factor (-0.5 for f32, 0.25 for AVX-512 f64)
// hoisted out of products, sums, and loop-carried reductions (block-arg phis).

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <base/arena.h>
#include <base/string.h>
#include "pass_llvm_rsqrt.h"

typedef struct {
    MLIR_ValueHandle v;
    double k;
} Scaled;

// Simple open-addressed handle -> Scaled hash map in the pass scratch arena.
typedef struct {
    MLIR_ValueHandle key;
    Scaled val;
} ScaledEntry;

typedef struct {
    Arena *arena;
    ScaledEntry *entries;
    size_t cap;
    size_t len;
} ScaledMap;

static uint64_t hmix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static void smap_init(ScaledMap *m, Arena *a, size_t cap) {
    m->arena = a;
    m->cap = cap;
    m->len = 0;
    m->entries = arena_new_array(a, ScaledEntry, cap);
    memset(m->entries, 0, cap * sizeof(ScaledEntry));
}

static void smap_grow(ScaledMap *m) {
    size_t old_cap = m->cap;
    ScaledEntry *old = m->entries;
    smap_init(m, m->arena, old_cap * 2);
    size_t mask = m->cap - 1;
    for (size_t i = 0; i < old_cap; i++) {
        if (old[i].key != MLIR_INVALID_HANDLE) {
            size_t slot = (size_t)(hmix((uint64_t)old[i].key) & mask);
            while (m->entries[slot].key != MLIR_INVALID_HANDLE) slot = (slot + 1) & mask;
            m->entries[slot] = old[i];
            m->len++;
        }
    }
}

static void smap_put(ScaledMap *m, MLIR_ValueHandle k, MLIR_ValueHandle v, double factor) {
    if (m->len * 2 >= m->cap) smap_grow(m);
    size_t mask = m->cap - 1;
    size_t slot = (size_t)(hmix((uint64_t)k) & mask);
    while (m->entries[slot].key != MLIR_INVALID_HANDLE && m->entries[slot].key != k)
        slot = (slot + 1) & mask;
    if (m->entries[slot].key == MLIR_INVALID_HANDLE) m->len++;
    m->entries[slot].key = k;
    m->entries[slot].val.v = v;
    m->entries[slot].val.k = factor;
}

static bool smap_get(const ScaledMap *m, MLIR_ValueHandle k, Scaled *out) {
    size_t mask = m->cap - 1;
    size_t slot = (size_t)(hmix((uint64_t)k) & mask);
    while (m->entries[slot].key != MLIR_INVALID_HANDLE) {
        if (m->entries[slot].key == k) {
            if (out) *out = m->entries[slot].val;
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

typedef struct {
    Arena *arena;
    uintptr_t *keys;
    size_t cap;
    size_t len;
} HandleSet;

static void hset_init(HandleSet *s, Arena *a, size_t cap) {
    s->arena = a;
    s->cap = cap;
    s->len = 0;
    s->keys = arena_new_array(a, uintptr_t, cap);
    memset(s->keys, 0, cap * sizeof(uintptr_t));
}

static void hset_grow(HandleSet *s) {
    size_t old_cap = s->cap;
    uintptr_t *old = s->keys;
    hset_init(s, s->arena, old_cap * 2);
    size_t mask = s->cap - 1;
    for (size_t i = 0; i < old_cap; i++) {
        if (old[i] != 0) {
            size_t slot = (size_t)(hmix((uint64_t)old[i]) & mask);
            while (s->keys[slot] != 0) slot = (slot + 1) & mask;
            s->keys[slot] = old[i];
            s->len++;
        }
    }
}

static bool hset_insert(HandleSet *s, uintptr_t k) {
    if (s->len * 2 >= s->cap) hset_grow(s);
    size_t mask = s->cap - 1;
    size_t slot = (size_t)(hmix((uint64_t)k) & mask);
    while (s->keys[slot] != 0) {
        if (s->keys[slot] == k) return false;
        slot = (slot + 1) & mask;
    }
    s->keys[slot] = k;
    s->len++;
    return true;
}

static bool hset_has(const HandleSet *s, uintptr_t k) {
    size_t mask = s->cap - 1;
    size_t slot = (size_t)(hmix((uint64_t)k) & mask);
    while (s->keys[slot] != 0) {
        if (s->keys[slot] == k) return true;
        slot = (slot + 1) & mask;
    }
    return false;
}

static MLIR_TypeHandle scalar_type(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_VECTOR)
        return MLIR_GetTypeElement(ctx, ty);
    return ty;
}

static bool is_f32_type(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    MLIR_TypeHandle sc = scalar_type(ctx, ty);
    return MLIR_GetTypeKind(ctx, sc) == MLIR_TYPE_KIND_FLOAT &&
           MLIR_GetTypeWidth(ctx, sc) == 32;
}

static bool is_f64_vector(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_GetTypeKind(ctx, ty) != MLIR_TYPE_KIND_VECTOR) return false;
    MLIR_TypeHandle sc = MLIR_GetTypeElement(ctx, ty);
    if (MLIR_GetTypeKind(ctx, sc) != MLIR_TYPE_KIND_FLOAT ||
        MLIR_GetTypeWidth(ctx, sc) != 64)
        return false;
    uint64_t n = MLIR_GetTypeNumElements(ctx, ty);
    return n == 2 || n == 4 || n == 8;
}

static bool is_fast_op(MLIR_Context *ctx, MLIR_OpHandle op) {
    return MLIR_GetOpFastMath(ctx, op) == MLIR_FMF_FAST;
}

static bool is_fdiv_by(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_ValueHandle divisor) {
    return MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FDIV &&
           MLIR_GetOpNumOperands(ctx, op) == 2 &&
           MLIR_GetOpOperand(ctx, op, 1) == divisor &&
           MLIR_GetOpOperand(ctx, op, 0) != divisor &&
           is_fast_op(ctx, op);
}

static bool is_power_of_two(double ratio) {
    if (ratio == 0.0 || !isfinite(ratio)) return false;
    int exp = 0;
    double m = frexp(ratio, &exp);
    return fabs(m) == 0.5;
}

static MLIR_ValueHandle materialize(MLIR_Builder *b, Scaled s, MLIR_TypeHandle ty) {
    if (s.k == 1.0) return s.v;
    MLIR_ValueHandle c = MLIR_BuildConstantFloat(b, OP_TYPE_LLVM_MLIR_CONSTANT, ty, s.k);
    return MLIR_BuildFBinary(b, OP_TYPE_LLVM_FMUL, ty, s.v, c, MLIR_FMF_FAST);
}

typedef struct {
    MLIR_Context *ctx;
    Arena *scratch;
    ScaledMap scaled;
    ScaledMap newton;
    HandleSet created_args;
    MLIR_OpHandle *replaced;
    size_t n_replaced;
    size_t cap_replaced;
    bool expanded;
} RsqrtCtx;

static void record_replaced(RsqrtCtx *E, MLIR_OpHandle op) {
    if (E->n_replaced == E->cap_replaced) {
        size_t nc = E->cap_replaced ? E->cap_replaced * 2 : 32;
        MLIR_OpHandle *arr = arena_new_array(E->scratch, MLIR_OpHandle, nc);
        if (E->n_replaced) memcpy(arr, E->replaced, E->n_replaced * sizeof(MLIR_OpHandle));
        E->replaced = arr;
        E->cap_replaced = nc;
    }
    E->replaced[E->n_replaced++] = op;
}

static Scaled get_scaled(const RsqrtCtx *E, MLIR_ValueHandle v) {
    Scaled s;
    if (smap_get(&E->scaled, v, &s)) return s;
    s.v = v;
    s.k = 1.0;
    return s;
}

static void expand_sqrt_op(RsqrtCtx *E, MLIR_OpHandle sqrt_op) {
    MLIR_ValueHandle s = MLIR_GetOpResult(E->ctx, sqrt_op, 0);
    MLIR_ValueHandle x = MLIR_GetOpOperand(E->ctx, sqrt_op, 0);
    MLIR_TypeHandle ty = MLIR_GetValueType(E->ctx, s);
    MLIR_Builder b = MLIR_BuilderAfter(E->ctx, sqrt_op);

    if (is_f32_type(E->ctx, ty)) {
        MLIR_ValueHandle one = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, ty, 1.0);
        MLIR_ValueHandle e = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FDIV, ty, one, s, MLIR_FMF_FAST);
        MLIR_ValueHandle ee = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, e, e, MLIR_FMF_FAST);
        MLIR_ValueHandle xee = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, x, ee, MLIR_FMF_FAST);
        MLIR_ValueHandle neg3 = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, ty, -3.0);
        MLIR_ValueHandle t = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, ty, xee, neg3, MLIR_FMF_FAST);
        MLIR_ValueHandle z = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, e, t, MLIR_FMF_FAST);
        smap_put(&E->newton, s, z, -0.5);
    } else {
        // AVX-512 double-precision estimate + 2 Newton steps, factor 0.25.
        uint64_t n = MLIR_GetTypeNumElements(E->ctx, ty);
        string intrin = (n == 8) ? str_lit("llvm.x86.avx512.rsqrt14.pd.512")
                      : (n == 4) ? str_lit("llvm.x86.avx512.rsqrt14.pd.256")
                                 : str_lit("llvm.x86.avx512.rsqrt14.pd.128");
        MLIR_TypeHandle i8_ty = MLIR_CreateTypeInteger(E->ctx, 8, true);
        MLIR_ValueHandle mask = MLIR_BuildConstantInt(&b, OP_TYPE_LLVM_MLIR_CONSTANT, i8_ty, -1);
        MLIR_ValueHandle undef = MLIR_BuildUndef(&b, ty);
        MLIR_ValueHandle iargs[3];
        iargs[0] = x;
        iargs[1] = undef;
        iargs[2] = mask;
        MLIR_ValueHandle e = MLIR_BuildCallIntrinsic(&b, intrin, ty, iargs, 3, MLIR_FMF_FAST);
        MLIR_ValueHandle neg3 = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, ty, -3.0);
        MLIR_ValueHandle ee = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, e, e, MLIR_FMF_FAST);
        MLIR_ValueHandle t1 = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, ty,
            MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, x, ee, MLIR_FMF_FAST),
            neg3, MLIR_FMF_FAST);
        MLIR_ValueHandle u = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, e, t1, MLIR_FMF_FAST);
        MLIR_ValueHandle qtr = MLIR_BuildConstantFloat(&b, OP_TYPE_LLVM_MLIR_CONSTANT, ty, 0.25);
        MLIR_ValueHandle qx = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, qtr, x, MLIR_FMF_FAST);
        MLIR_ValueHandle uu = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, u, u, MLIR_FMF_FAST);
        MLIR_ValueHandle t2 = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD, ty,
            MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, qx, uu, MLIR_FMF_FAST),
            neg3, MLIR_FMF_FAST);
        MLIR_ValueHandle z = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL, ty, u, t2, MLIR_FMF_FAST);
        smap_put(&E->newton, s, z, 0.25);
    }
    E->expanded = true;
}

static bool try_reduction(RsqrtCtx *E, MLIR_OpHandle sum_op, size_t which) {
    MLIR_ValueHandle phi = MLIR_GetOpOperand(E->ctx, sum_op, which);
    MLIR_ValueHandle v = MLIR_GetOpOperand(E->ctx, sum_op, 1 - which);
    MLIR_ValueHandle sum_val = MLIR_GetOpResult(E->ctx, sum_op, 0);
    if (MLIR_GetValueKind(E->ctx, phi) != MLIR_VALUE_BLOCK_ARG) return false;
    if (smap_get(&E->scaled, phi, NULL)) return false;
    Scaled sv;
    if (!smap_get(&E->scaled, v, &sv)) return false;
    if (!MLIR_ValueHasOneUse(E->ctx, phi)) return false;

    MLIR_BlockHandle blk = MLIR_GetValueParentBlock(E->ctx, phi);
    size_t arg_idx = MLIR_GetValueResultIndex(E->ctx, phi);
    MLIR_PredHandle preds[2];
    size_t n_preds = 0;
    for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(E->ctx, blk); p; p = MLIR_GetPredNext(E->ctx, p)) {
        if (n_preds >= 2) return false;
        preds[n_preds++] = p;
    }
    if (n_preds != 2) return false;

    size_t back = (MLIR_GetIncomingValue(E->ctx, preds[0], arg_idx) == sum_val) ? 0 : 1;
    if (MLIR_GetIncomingValue(E->ctx, preds[back], arg_idx) != sum_val) return false;

    MLIR_PredHandle entry_pred = preds[1 - back];
    MLIR_PredHandle back_pred = preds[back];
    MLIR_ValueHandle init = MLIR_GetIncomingValue(E->ctx, entry_pred, arg_idx);
    MLIR_TypeHandle phi_ty = MLIR_GetValueType(E->ctx, phi);

    MLIR_ValueHandle start;
    if (MLIR_IsConstantZero(E->ctx, init, -1)) {
        start = init;
    } else {
        Scaled si = get_scaled(E, init);
        double ratio = si.k / sv.k;
        if (!is_power_of_two(ratio)) return false;
        MLIR_Builder eb = MLIR_BuilderBefore(E->ctx, MLIR_GetPredTerminator(E->ctx, entry_pred));
        Scaled rsc = { si.v, ratio };
        start = materialize(&eb, rsc, phi_ty);
    }

    MLIR_ValueHandle acc = MLIR_AddBlockArgument(E->ctx, blk, phi_ty, MLIR_INVALID_HANDLE);
    hset_insert(&E->created_args, (uintptr_t)acc);
    MLIR_Builder sb = MLIR_BuilderBefore(E->ctx, sum_op);
    MLIR_ValueHandle next = MLIR_BuildFBinary(&sb, OP_TYPE_LLVM_FADD, phi_ty, acc, sv.v, MLIR_FMF_FAST);

    MLIR_AppendOpSuccessorOperand(E->ctx, MLIR_GetPredTerminator(E->ctx, entry_pred),
                                  MLIR_GetPredSuccessorIndex(E->ctx, entry_pred), start);
    MLIR_AppendOpSuccessorOperand(E->ctx, MLIR_GetPredTerminator(E->ctx, back_pred),
                                  MLIR_GetPredSuccessorIndex(E->ctx, back_pred), next);

    smap_put(&E->scaled, phi, acc, sv.k);
    smap_put(&E->scaled, sum_val, next, sv.k);
    record_replaced(E, sum_op);
    return true;
}

static void visit_block_args(RsqrtCtx *E, MLIR_BlockHandle blk) {
    size_t n_args = MLIR_GetBlockNumArgs(E->ctx, blk);
    for (size_t a = 0; a < n_args; a++) {
        MLIR_ValueHandle arg = MLIR_GetBlockArg(E->ctx, blk, a);
        if (hset_has(&E->created_args, (uintptr_t)arg) || smap_get(&E->scaled, arg, NULL))
            continue;
        MLIR_PredHandle p = MLIR_GetBlockFirstPred(E->ctx, blk);
        if (!p) continue;
        double k = 0.0;
        bool all_same = true;
        size_t n_in = 0;
        for (; p; p = MLIR_GetPredNext(E->ctx, p)) {
            MLIR_ValueHandle inc = MLIR_GetIncomingValue(E->ctx, p, a);
            Scaled sc;
            if (!smap_get(&E->scaled, inc, &sc) || (n_in > 0 && sc.k != k)) {
                all_same = false;
                break;
            }
            k = sc.k;
            n_in++;
        }
        if (!all_same || n_in == 0) continue;
        MLIR_ValueHandle new_arg = MLIR_AddBlockArgument(E->ctx, blk,
            MLIR_GetValueType(E->ctx, arg), MLIR_INVALID_HANDLE);
        hset_insert(&E->created_args, (uintptr_t)new_arg);
        for (p = MLIR_GetBlockFirstPred(E->ctx, blk); p; p = MLIR_GetPredNext(E->ctx, p)) {
            MLIR_ValueHandle inc = MLIR_GetIncomingValue(E->ctx, p, a);
            Scaled sc;
            smap_get(&E->scaled, inc, &sc);
            MLIR_AppendOpSuccessorOperand(E->ctx, MLIR_GetPredTerminator(E->ctx, p),
                                          MLIR_GetPredSuccessorIndex(E->ctx, p), sc.v);
        }
        smap_put(&E->scaled, arg, new_arg, k);
    }
}

static void visit_op(RsqrtCtx *E, MLIR_OpHandle op) {
    size_t n_ops = MLIR_GetOpNumOperands(E->ctx, op);
    bool uses_scaled = false;
    for (size_t i = 0; i < n_ops; i++) {
        MLIR_ValueHandle v = MLIR_GetOpOperand(E->ctx, op, i);
        if (smap_get(&E->scaled, v, NULL) || smap_get(&E->newton, v, NULL)) {
            uses_scaled = true;
            break;
        }
    }
    if (!uses_scaled) return;

    MLIR_OpType t = MLIR_GetOpType(E->ctx, op);
    if (t == OP_TYPE_LLVM_FDIV && n_ops == 2) {
        Scaled nsc;
        if (smap_get(&E->newton, MLIR_GetOpOperand(E->ctx, op, 1), &nsc)) {
            MLIR_ValueHandle res = MLIR_GetOpResult(E->ctx, op, 0);
            MLIR_ValueHandle num = MLIR_GetOpOperand(E->ctx, op, 0);
            double a = 0.0;
            if (MLIR_MatchConstantFloat(E->ctx, num, -1, &a)) {
                smap_put(&E->scaled, res, nsc.v, a * nsc.k);
            } else {
                Scaled s = get_scaled(E, num);
                MLIR_Builder b = MLIR_BuilderBefore(E->ctx, op);
                MLIR_ValueHandle prod = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL,
                    MLIR_GetValueType(E->ctx, res), s.v, nsc.v, MLIR_FMF_FAST);
                smap_put(&E->scaled, res, prod, s.k * nsc.k);
            }
            record_replaced(E, op);
            return;
        }
    }
    if (t == OP_TYPE_LLVM_FMUL && is_fast_op(E->ctx, op) && n_ops == 2) {
        Scaled l = get_scaled(E, MLIR_GetOpOperand(E->ctx, op, 0));
        Scaled r = get_scaled(E, MLIR_GetOpOperand(E->ctx, op, 1));
        MLIR_ValueHandle res = MLIR_GetOpResult(E->ctx, op, 0);
        MLIR_Builder b = MLIR_BuilderBefore(E->ctx, op);
        MLIR_ValueHandle prod = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FMUL,
            MLIR_GetValueType(E->ctx, res), l.v, r.v, MLIR_FMF_FAST);
        smap_put(&E->scaled, res, prod, l.k * r.k);
        record_replaced(E, op);
        return;
    }
    if (t == OP_TYPE_LLVM_FADD && is_fast_op(E->ctx, op) && n_ops == 2) {
        if (try_reduction(E, op, 0) || try_reduction(E, op, 1)) return;
        Scaled l, r;
        if (smap_get(&E->scaled, MLIR_GetOpOperand(E->ctx, op, 0), &l) &&
            smap_get(&E->scaled, MLIR_GetOpOperand(E->ctx, op, 1), &r) &&
            l.k == r.k) {
            MLIR_ValueHandle res = MLIR_GetOpResult(E->ctx, op, 0);
            MLIR_Builder b = MLIR_BuilderBefore(E->ctx, op);
            MLIR_ValueHandle sum = MLIR_BuildFBinary(&b, OP_TYPE_LLVM_FADD,
                MLIR_GetValueType(E->ctx, res), l.v, r.v, MLIR_FMF_FAST);
            smap_put(&E->scaled, res, sum, l.k);
            record_replaced(E, op);
            return;
        }
    }
    // Any other use takes the value with its factor applied.
    MLIR_Builder b = MLIR_BuilderBefore(E->ctx, op);
    for (size_t i = 0; i < n_ops; i++) {
        MLIR_ValueHandle v = MLIR_GetOpOperand(E->ctx, op, i);
        Scaled sc;
        if (smap_get(&E->scaled, v, &sc)) {
            MLIR_SetOpOperand(E->ctx, op, i,
                materialize(&b, sc, MLIR_GetValueType(E->ctx, v)));
        }
    }
}

bool mlir_pass_llvm_rsqrt_run(MLIR_PassContext *pc, MLIR_OpHandle fn, void *state) {
    (void)state;
    MLIR_Context *ctx = pc->ctx;
    if (!MLIR_TargetIsX86(pc->target) || MLIR_IsFunctionDeclaration(ctx, fn) ||
        MLIR_HasFunctionPassthrough(ctx, fn, str_lit("reciprocal-estimates")))
        return true;

    MLIR_RegionHandle region = MLIR_GetOpRegion(ctx, fn, 0);
    HandleSet roots;
    hset_init(&roots, pc->scratch, 32);
    bool has_float = false;

    for (MLIR_BlockHandle blk = MLIR_GetRegionFirstBlock(ctx, region);
         blk != MLIR_INVALID_HANDLE; blk = MLIR_GetBlockNext(ctx, blk)) {
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, blk);
             op != MLIR_INVALID_HANDLE; op = MLIR_GetOpNext(ctx, op)) {
            MLIR_OpType t = MLIR_GetOpType(ctx, op);
            if (t == OP_TYPE_LLVM_INTR_POW) return true;
            if (t == OP_TYPE_LLVM_CALL) {
                string callee = MLIR_GetCallee(ctx, op);
                if (str_eq(callee, str_lit("sqrtf")) || str_eq(callee, str_lit("powf")) ||
                    str_eq(callee, str_lit("sqrt")) || str_eq(callee, str_lit("pow")))
                    return true;
                continue;
            }
            if (t != OP_TYPE_LLVM_INTR_SQRT) continue;
            MLIR_ValueHandle res = MLIR_GetOpResult(ctx, op, 0);
            MLIR_TypeHandle ty = MLIR_GetValueType(ctx, res);
            bool f32 = is_f32_type(ctx, ty);
            bool f64v = is_f64_vector(ctx, ty) &&
                        MLIR_TargetHasFeature(ctx, pc->target, fn, MLIR_FEATURE_AVX512F);
            if (!f32 && !f64v) continue;
            if (!is_fast_op(ctx, op)) return true;
            for (MLIR_UseHandle u = MLIR_GetValueFirstUse(ctx, res); u; u = MLIR_GetUseNext(ctx, u)) {
                if (!is_fdiv_by(ctx, MLIR_GetUseOwner(ctx, u), res)) return true;
            }
            hset_insert(&roots, (uintptr_t)op);
            if (f32) has_float = true;
        }
    }
    if (roots.len == 0) return true;

    RsqrtCtx E;
    memset(&E, 0, sizeof E);
    E.ctx = ctx;
    E.scratch = pc->scratch;
    smap_init(&E.scaled, pc->scratch, 64);
    smap_init(&E.newton, pc->scratch, 32);
    hset_init(&E.created_args, pc->scratch, 32);

    MLIR_BlockHandle *rpo = NULL;
    size_t n_rpo = MLIR_GetBlockOrder(ctx, region, MLIR_ORDER_RPO, pc->scratch, &rpo);
    for (size_t bi = 0; bi < n_rpo; bi++) {
        MLIR_BlockHandle blk = rpo[bi];
        visit_block_args(&E, blk);
        // Snapshot ops in blk before rewriting.
        size_t n_ops = 0;
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, blk); op; op = MLIR_GetOpNext(ctx, op))
            n_ops++;
        MLIR_OpHandle *snap = arena_new_array(pc->scratch, MLIR_OpHandle, n_ops ? n_ops : 1);
        size_t k = 0;
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, blk); op; op = MLIR_GetOpNext(ctx, op))
            snap[k++] = op;
        for (size_t oi = 0; oi < n_ops; oi++) {
            if (hset_has(&roots, (uintptr_t)snap[oi]))
                expand_sqrt_op(&E, snap[oi]);
            else
                visit_op(&E, snap[oi]);
        }
    }

    // Materialize scaled values flowing into non-scaled block arguments.
    for (size_t bi = 0; bi < n_rpo; bi++) {
        MLIR_BlockHandle blk = rpo[bi];
        size_t n_args = MLIR_GetBlockNumArgs(ctx, blk);
        for (size_t a = 0; a < n_args; a++) {
            MLIR_ValueHandle arg = MLIR_GetBlockArg(ctx, blk, a);
            if (hset_has(&E.created_args, (uintptr_t)arg) || smap_get(&E.scaled, arg, NULL))
                continue;
            for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, blk); p; p = MLIR_GetPredNext(ctx, p)) {
                MLIR_ValueHandle inc = MLIR_GetIncomingValue(ctx, p, a);
                Scaled sc;
                if (!smap_get(&E.scaled, inc, &sc)) continue;
                MLIR_Builder pb = MLIR_BuilderBefore(ctx, MLIR_GetPredTerminator(ctx, p));
                MLIR_SetIncomingValue(ctx, p, a,
                    materialize(&pb, sc, MLIR_GetValueType(ctx, arg)));
            }
        }
    }

    // Break reduction cycles on replaced block args, then erase dead replaced ops.
    for (size_t bi = 0; bi < n_rpo; bi++) {
        MLIR_BlockHandle blk = rpo[bi];
        size_t n_args = MLIR_GetBlockNumArgs(ctx, blk);
        for (size_t a = 0; a < n_args; a++) {
            MLIR_ValueHandle arg = MLIR_GetBlockArg(ctx, blk, a);
            if (!smap_get(&E.scaled, arg, NULL)) continue;
            for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, blk); p; p = MLIR_GetPredNext(ctx, p)) {
                MLIR_Builder pb = MLIR_BuilderBefore(ctx, MLIR_GetPredTerminator(ctx, p));
                MLIR_ValueHandle u = MLIR_BuildUndef(&pb, MLIR_GetValueType(ctx, arg));
                MLIR_SetIncomingValue(ctx, p, a, u);
            }
        }
    }

    bool erased = true;
    while (erased) {
        erased = false;
        for (size_t i = 0; i < E.n_replaced; i++) {
            MLIR_OpHandle op = E.replaced[i];
            if (op == MLIR_INVALID_HANDLE) continue;
            if (MLIR_GetOpNumResults(ctx, op) == 0 ||
                MLIR_IsValueUnused(ctx, MLIR_GetOpResult(ctx, op, 0))) {
                MLIR_EraseOp(ctx, op);
                E.replaced[i] = MLIR_INVALID_HANDLE;
                erased = true;
            }
        }
    }

    if (E.expanded && has_float) {
        MLIR_SetFunctionPassthrough(ctx, fn, str_lit("reciprocal-estimates"),
                                    str_lit("sqrtf:0,vec-sqrtf:0"));
    }
    return true;
}

MLIR_Pass MLIR_PassLLVMRsqrt(void) {
    MLIR_Pass p;
    p.name = str_lit("llvm-rsqrt");
    p.anchor = MLIR_ANCHOR_FUNCTION;
    p.run = mlir_pass_llvm_rsqrt_run;
    p.state = NULL;
    return p;
}

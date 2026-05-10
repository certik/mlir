// Stage 1 of the native LLVM->WASM pipeline:
// LLVM-dialect builtin.module --(walk)--> wasmssa_module_t.
//
// This file produces a wasmssa-form intermediate (mlir_wasm_dialect.h):
// a flat per-function array of wasmssa ops, each carrying a MLIR_OpType
// discriminator and operand indices into the same array. SSA value
// indexing convention shared with stage 2 (mlir_stackify_wasmssa.c):
//
//   ssa_def_idx in [0, n_params)              -> function parameter i
//   ssa_def_idx in [n_params, n_params+n_ops) -> result of op (idx-n_params)
//
// Scope (initial scaffold, matches the simple.tc end-to-end test):
//   - llvm.func (def & decl) -> wasmssa.func / imported wasmssa.func
//   - llvm.return (with optional i32 value)
//   - llvm.mlir.constant (i32 / i64)
//   - llvm.alloca (lowered onto a shadow stack)
//   - llvm.store / llvm.load (i32 / i64 / f32 / f64, plus i8/i16 subword)
//   - llvm.sext (i32 -> i64)
//   - llvm.call (direct, fixed-arity)
// Anything else returns NULL (the test runner's wasm_native_run gates
// which tests are exercised on the native path).

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir_api.h"
#include "mlir_wasm_dialect.h"

wasmssa_module_t *mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx,
                                             MLIR_OpHandle module);

// =============================================================================
// String / type helpers (mirror the old single-stage translator).
// =============================================================================
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}
static char *xstrdupn(const char *s, size_t n) {
    char *r = (char *)malloc(n + 1);
    memcpy(r, s, n);
    r[n] = 0;
    return r;
}
static char *xstrdup_str(string s) { return xstrdupn(s.str, s.size); }

static MLIR_AttributeHandle find_attr(MLIR_OpHandle op, const char *name) {
    size_t n = MLIR_GetOpNumAttributes(op);
    for (size_t i = 0; i < n; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(a);
        if (name_eq(an, name)) return a;
    }
    return MLIR_INVALID_HANDLE;
}

// MLIR LLVM-dialect type -> WT_* value-type byte. Returns 0 if the type
// is not a primitive scalar.
static uint8_t wasm_vt(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return WT_I32;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return WT_I32;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return WT_F32;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return WT_F64;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8 || w == 16 || w == 32) return WT_I32;
        if (w == 64) return WT_I64;
    }
    return 0;
}

// Returns the integer bit width of `ty` (1/8/16/32/64) or 0 if `ty`
// is not an `iN` integer type.
static int int_bits(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else return 0;
        }
        return w;
    }
    return 0;
}

static unsigned type_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty);
static unsigned type_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty);

// Round x up to the nearest multiple of `align` (a power of two).
static unsigned align_up(unsigned x, unsigned align) {
    return (x + align - 1) & ~(align - 1);
}

// Sum of (padded) field sizes for an LLVM struct type, with the natural
// alignment that the LLVM data layout uses on wasm32 (each field aligned
// to its own alignment; trailing padding to the struct's max-field alignment).
static unsigned struct_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
    unsigned off = 0, max_align = 1;
    for (size_t i = 0; i < nf; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ty, i);
        unsigned fsz = type_size_bytes(ctx, ft);
        unsigned fal = type_align_bytes(ctx, ft);
        if (fsz == 0 || fal == 0) return 0;
        off = align_up(off, fal);
        off += fsz;
        if (fal > max_align) max_align = fal;
    }
    return align_up(off, max_align);
}
static unsigned struct_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
    unsigned ma = 1;
    for (size_t i = 0; i < nf; i++) {
        unsigned fa = type_align_bytes(ctx, MLIR_GetTypeLLVMStructField(ty, i));
        if (fa > ma) ma = fa;
    }
    return ma;
}
// Byte offset of the i-th field within its containing struct.
static unsigned struct_field_offset(MLIR_Context *ctx, MLIR_TypeHandle sty,
                                    size_t fld_idx) {
    unsigned off = 0;
    for (size_t i = 0; i <= fld_idx; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(sty, i);
        unsigned fal = type_align_bytes(ctx, ft);
        off = align_up(off, fal);
        if (i == fld_idx) return off;
        off += type_size_bytes(ctx, ft);
    }
    return off;
}

static unsigned type_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 8;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    if (MLIR_IsTypeLLVMArray(ty)) {
        unsigned esz = type_size_bytes(ctx, MLIR_GetTypeLLVMArrayElement(ty));
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeLLVMArrayNumElements(ty);
    }
    if (MLIR_IsTypeLLVMStruct(ty)) {
        return struct_size_bytes(ctx, ty);
    }
    return 0;
}

static unsigned type_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_IsTypeLLVMArray(ty)) {
        return type_align_bytes(ctx, MLIR_GetTypeLLVMArrayElement(ty));
    }
    if (MLIR_IsTypeLLVMStruct(ty)) {
        return struct_align_bytes(ctx, ty);
    }
    unsigned sz = type_size_bytes(ctx, ty);
    return sz ? sz : 1;
}

// =============================================================================
// Per-function lowering state.
// =============================================================================
typedef struct { uintptr_t key; int idx; } VMapEntry;
typedef struct { uintptr_t key; uint32_t off; } AMapEntry;

// One frame on the YieldStack: tells lower_op how to translate a
// `scf.yield` it encounters (which carrier ids receive each yielded
// operand). For scf.while.after-region, this is empty -- yield is
// translated to a br back to loop start by lower_scf_while directly.
typedef struct {
    uint32_t *carrier_ids;
    uint8_t  *carrier_vts;
    int       n;
    bool      yield_is_branch;  // true if yield should emit a br instead
    uint32_t  br_depth;         // depth for that br
} YieldFrame;

typedef struct {
    MLIR_Context   *ctx;
    wasmssa_func_t *f;
    size_t          n_params;

    // Map MLIR_ValueHandle -> ssa_def_idx (index into [0, n_params + f->n_ops)).
    VMapEntry *vmap; size_t n_vmap, c_vmap;

    // Map MLIR_ValueHandle (alloca result) -> shadow-stack frame offset.
    AMapEntry *amap; size_t n_amap, c_amap;

    uint32_t  frame_size;
    int       sp_def;       // ssa_def_idx of the post-decrement SP (-1 if no frame)

    uint32_t  next_carrier;
    // Carrier valtypes: index by carrier_id.
    uint8_t  *carrier_vts; size_t n_cvt, c_cvt;

    YieldFrame *yield_stack; size_t n_ys, c_ys;
} FnCtx;

static uint32_t alloc_carrier(FnCtx *F, uint8_t vt) {
    if (F->n_cvt == F->c_cvt) {
        F->c_cvt = F->c_cvt ? F->c_cvt * 2 : 8;
        F->carrier_vts = (uint8_t *)realloc(F->carrier_vts, F->c_cvt);
    }
    uint32_t id = (uint32_t)F->n_cvt;
    F->carrier_vts[F->n_cvt++] = vt;
    F->next_carrier = (uint32_t)F->n_cvt;
    return id;
}
static void push_yield(FnCtx *F, YieldFrame fr) {
    if (F->n_ys == F->c_ys) {
        F->c_ys = F->c_ys ? F->c_ys * 2 : 4;
        F->yield_stack = (YieldFrame *)realloc(F->yield_stack,
                                              F->c_ys * sizeof(YieldFrame));
    }
    F->yield_stack[F->n_ys++] = fr;
}
static void pop_yield(FnCtx *F) {
    YieldFrame *top = &F->yield_stack[--F->n_ys];
    free(top->carrier_ids);
    free(top->carrier_vts);
}

static void vmap_set(FnCtx *F, MLIR_ValueHandle v, int idx) {
    if (F->n_vmap == F->c_vmap) {
        F->c_vmap = F->c_vmap ? F->c_vmap * 2 : 16;
        F->vmap = (VMapEntry *)realloc(F->vmap, F->c_vmap * sizeof(VMapEntry));
    }
    F->vmap[F->n_vmap].key = (uintptr_t)v;
    F->vmap[F->n_vmap].idx = idx;
    F->n_vmap++;
}
static int vmap_get(FnCtx *F, MLIR_ValueHandle v, int *out) {
    for (size_t i = 0; i < F->n_vmap; i++) {
        if (F->vmap[i].key == (uintptr_t)v) { *out = F->vmap[i].idx; return 1; }
    }
    return 0;
}
static void amap_set(FnCtx *F, MLIR_ValueHandle v, uint32_t off) {
    if (F->n_amap == F->c_amap) {
        F->c_amap = F->c_amap ? F->c_amap * 2 : 8;
        F->amap = (AMapEntry *)realloc(F->amap, F->c_amap * sizeof(AMapEntry));
    }
    F->amap[F->n_amap].key = (uintptr_t)v;
    F->amap[F->n_amap].off = off;
    F->n_amap++;
}
static int amap_get(FnCtx *F, MLIR_ValueHandle v, uint32_t *out) {
    for (size_t i = 0; i < F->n_amap; i++) {
        if (F->amap[i].key == (uintptr_t)v) { *out = F->amap[i].off; return 1; }
    }
    return 0;
}

// Append an op and return its ssa_def_idx (n_params + op_index).
static int add_op(FnCtx *F, wasmssa_op_t **out) {
    size_t i = F->f->n_ops;
    *out = wasmssa_func_add_op(F->f);
    return (int)(F->n_params + i);
}

// Convenience: const-i32, returns ssa_def_idx.
static int emit_const_i32(FnCtx *F, int32_t v) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_CONST;
    o->valtype = WT_I32;
    o->i_const = v;
    o->has_result = true;
    return idx;
}
// Convenience: i32 add of two ssa-def operands.
static int emit_add_i32(FnCtx *F, int lhs, int rhs) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_ADD;
    o->valtype = WT_I32;
    o->n_operands = 2;
    o->operands = (int *)malloc(2 * sizeof(int));
    o->operands[0] = lhs;
    o->operands[1] = rhs;
    o->has_result = true;
    return idx;
}
// Convenience: i32 sub.
static int emit_sub_i32(FnCtx *F, int lhs, int rhs) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_SUB;
    o->valtype = WT_I32;
    o->n_operands = 2;
    o->operands = (int *)malloc(2 * sizeof(int));
    o->operands[0] = lhs;
    o->operands[1] = rhs;
    o->has_result = true;
    return idx;
}
// Convenience: global_get of an i32 global.
static int emit_global_get(FnCtx *F, uint32_t gidx) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_GLOBAL_GET;
    o->valtype = WT_I32;
    o->global_idx = gidx;
    o->has_result = true;
    return idx;
}
static void emit_global_set(FnCtx *F, uint32_t gidx, int valv) {
    wasmssa_op_t *o; (void)add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_GLOBAL_SET;
    o->valtype = WT_I32;
    o->global_idx = gidx;
    o->n_operands = 1;
    o->operands = (int *)malloc(sizeof(int));
    o->operands[0] = valv;
}

// Convenience: i32 mul.
static int emit_mul_i32(FnCtx *F, int lhs, int rhs) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_BINOP;
    o->valtype = WT_I32;
    o->wasm_opcode = 0x6c;  // i32.mul
    o->n_operands = 2;
    o->operands = (int *)malloc(2 * sizeof(int));
    o->operands[0] = lhs;
    o->operands[1] = rhs;
    o->has_result = true;
    return idx;
}
// Convenience: i32.wrap_i64 (0xa7) — narrows an i64 SSA value to i32.
static int emit_wrap_i64_to_i32(FnCtx *F, int v) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_UNOP;
    o->valtype = WT_I32;
    o->wasm_opcode = 0xa7;
    o->n_operands = 1;
    o->operands = (int *)malloc(sizeof(int));
    o->operands[0] = v;
    o->has_result = true;
    return idx;
}

// Parse a "array<i32: v0, v1, ...>" string into a heap-allocated int32_t
// vector; returns NULL on parse failure. Sets *n_out.
static int32_t *parse_dense_i32_array(string s, size_t *n_out) {
    *n_out = 0;
    const char *p = s.str;
    const char *end = s.str + s.size;
    const char *pref = "array<i32";
    size_t plen = strlen(pref);
    if (s.size < plen || memcmp(p, pref, plen) != 0) return NULL;
    p += plen;
    // Allow "array<i32>" (empty) or "array<i32: ...>".
    while (p < end && *p == ' ') p++;
    if (p < end && *p == '>') { return (int32_t *)malloc(sizeof(int32_t)); }
    if (p >= end || *p != ':') return NULL;
    p++;
    int32_t *vs = NULL; size_t n = 0, c = 0;
    while (p < end && *p != '>') {
        while (p < end && (*p == ' ' || *p == ',')) p++;
        if (p >= end || *p == '>') break;
        bool neg = false;
        if (*p == '-') { neg = true; p++; }
        if (p >= end || *p < '0' || *p > '9') { free(vs); return NULL; }
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') { v = v*10 + (*p - '0'); p++; }
        if (neg) v = -v;
        if (n == c) { c = c ? c*2 : 4; vs = (int32_t *)realloc(vs, c * sizeof(int32_t)); }
        vs[n++] = (int32_t)v;
    }
    *n_out = n;
    return vs ? vs : (int32_t *)malloc(sizeof(int32_t));
}

// =============================================================================
// Per-op lowering.
// =============================================================================
static bool lower_op(FnCtx *F, MLIR_OpHandle op);
static bool lower_block(FnCtx *F, MLIR_BlockHandle blk);

// Append a structured-CF marker / br / select op.
static int emit_marker(FnCtx *F, MLIR_OpType t, uint8_t blocktype,
                       int cond_operand, uint32_t depth) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = t;
    o->valtype = blocktype;
    o->br_depth = depth;
    if (cond_operand >= 0) {
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = cond_operand;
    }
    return idx;
}
static int emit_eqz(FnCtx *F, int v) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_EQZ;
    o->valtype = WT_I32;
    o->n_operands = 1;
    o->operands = (int *)malloc(sizeof(int));
    o->operands[0] = v;
    o->has_result = true;
    return idx;
}
static int emit_carrier_get(FnCtx *F, uint32_t cid, uint8_t vt) {
    wasmssa_op_t *o; int idx = add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_CARRIER_GET;
    o->valtype = vt;
    o->carrier_id = cid;
    o->has_result = true;
    return idx;
}
static void emit_carrier_set(FnCtx *F, uint32_t cid, uint8_t vt, int valv) {
    wasmssa_op_t *o; (void)add_op(F, &o);
    o->type = OP_TYPE_WASMSSA_CARRIER_SET;
    o->valtype = vt;
    o->carrier_id = cid;
    o->n_operands = 1;
    o->operands = (int *)malloc(sizeof(int));
    o->operands[0] = valv;
}

// ---- scf.if ----------------------------------------------------------------
static bool lower_scf_if(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(op, 0);
    int cond_idx;
    if (!vmap_get(F, cond_v, &cond_idx)) return false;

    size_t n_results = MLIR_GetOpNumResults(op);
    uint32_t *cids = NULL;
    uint8_t  *cvts = NULL;
    if (n_results) {
        cids = (uint32_t *)malloc(n_results * sizeof(uint32_t));
        cvts = (uint8_t *)malloc(n_results);
        for (size_t i = 0; i < n_results; i++) {
            MLIR_ValueHandle r = MLIR_GetOpResult(op, i);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            if (vt == 0) { free(cids); free(cvts); return false; }
            cvts[i] = vt;
            cids[i] = alloc_carrier(F, vt);
        }
    }

    emit_marker(F, OP_TYPE_WASMSSA_IF_BEGIN, 0x40, cond_idx, 0);

    YieldFrame yf = {0};
    if (n_results) {
        yf.carrier_ids = (uint32_t *)malloc(n_results * sizeof(uint32_t));
        yf.carrier_vts = (uint8_t *)malloc(n_results);
        memcpy(yf.carrier_ids, cids, n_results * sizeof(uint32_t));
        memcpy(yf.carrier_vts, cvts, n_results);
        yf.n = (int)n_results;
    }
    push_yield(F, yf);

    // Then region (region 0).
    if (MLIR_GetOpNumRegions(op) < 1) goto fail;
    MLIR_RegionHandle then_r = MLIR_GetOpRegion(op, 0);
    if (MLIR_GetRegionNumBlocks(then_r) != 1) goto fail;
    if (!lower_block(F, MLIR_GetRegionBlock(then_r, 0))) goto fail;

    // Else region (region 1) — may be empty.
    bool has_else = MLIR_GetOpNumRegions(op) >= 2 &&
                    MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 1)) > 0;
    if (has_else) {
        emit_marker(F, OP_TYPE_WASMSSA_IF_ELSE, 0, -1, 0);
        if (!lower_block(F, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0))) goto fail;
    } else if (n_results) {
        // scf.if with results MUST have an else region; if the source had
        // none we can't fabricate values. Reject.
        goto fail;
    }
    pop_yield(F);

    emit_marker(F, OP_TYPE_WASMSSA_END, 0, -1, 0);

    // Bind scf.if results to fresh CARRIER_GET ops.
    for (size_t i = 0; i < n_results; i++) {
        int gidx = emit_carrier_get(F, cids[i], cvts[i]);
        vmap_set(F, MLIR_GetOpResult(op, i), gidx);
    }
    free(cids); free(cvts);
    return true;
fail:
    if (F->n_ys && F->yield_stack[F->n_ys-1].carrier_ids == yf.carrier_ids) {
        // Fixed: avoid double free; pop properly.
        pop_yield(F);
    }
    free(cids); free(cvts);
    return false;
}

// ---- scf.while -------------------------------------------------------------
static bool lower_scf_while(FnCtx *F, MLIR_OpHandle op) {
    // Lifted IR shape: scf.while : () -> () {<before> ... scf.condition(%c)}
    // do {<after> ... scf.yield}. No iter args / no carries across the
    // before/after boundary.
    if (MLIR_GetOpNumOperands(op) != 0 || MLIR_GetOpNumResults(op) != 0) {
        fprintf(stderr,
                "wasmssa-lower: scf.while with iter args / results not supported\n");
        return false;
    }
    if (MLIR_GetOpNumRegions(op) < 2) return false;
    MLIR_RegionHandle before_r = MLIR_GetOpRegion(op, 0);
    MLIR_RegionHandle after_r  = MLIR_GetOpRegion(op, 1);
    if (MLIR_GetRegionNumBlocks(before_r) != 1) return false;
    if (MLIR_GetRegionNumBlocks(after_r)  != 1) return false;
    MLIR_BlockHandle before_b = MLIR_GetRegionBlock(before_r, 0);
    MLIR_BlockHandle after_b  = MLIR_GetRegionBlock(after_r, 0);

    emit_marker(F, OP_TYPE_WASMSSA_BLOCK_BEGIN, 0x40, -1, 0);  // $exit
    emit_marker(F, OP_TYPE_WASMSSA_LOOP_BEGIN,  0x40, -1, 0);  // $continue

    // Walk before-block ops. Terminator is scf.condition.
    size_t nb = MLIR_GetBlockNumOps(before_b);
    for (size_t i = 0; i < nb; i++) {
        MLIR_OpHandle bop = MLIR_GetBlockOp(before_b, i);
        string n = MLIR_GetOpName(bop);
        if (name_eq(n, "scf.condition")) {
            if (MLIR_GetOpNumOperands(bop) < 1) return false;
            int cidx;
            if (!vmap_get(F, MLIR_GetOpOperand(bop, 0), &cidx)) return false;
            int z = emit_eqz(F, cidx);
            // If !cond, branch to $exit (depth 1 above the loop).
            emit_marker(F, OP_TYPE_WASMSSA_BR_IF, 0, z, 1);
        } else {
            if (!lower_op(F, bop)) return false;
        }
    }
    // Walk after-block ops. Terminator scf.yield -> br $continue (depth 0).
    size_t na = MLIR_GetBlockNumOps(after_b);
    for (size_t i = 0; i < na; i++) {
        MLIR_OpHandle aop = MLIR_GetBlockOp(after_b, i);
        string n = MLIR_GetOpName(aop);
        if (name_eq(n, "scf.yield")) {
            emit_marker(F, OP_TYPE_WASMSSA_BR, 0, -1, 0);
        } else {
            if (!lower_op(F, aop)) return false;
        }
    }
    emit_marker(F, OP_TYPE_WASMSSA_END, 0, -1, 0);  // close loop
    emit_marker(F, OP_TYPE_WASMSSA_END, 0, -1, 0);  // close block
    return true;
}

static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op);
static bool lower_op(FnCtx *F, MLIR_OpHandle op) {
    bool ok = lower_op_inner(F, op);
    if (!ok) {
        string n = MLIR_GetOpName(op);
        fprintf(stderr, "wasmssa-lower: lower_op failed at '%.*s'\n",
                (int)n.size, n.str);
    }
    return ok;
}
static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op) {
    string name = MLIR_GetOpName(op);

    // ---- scf.if / scf.while ----------------------------------------------
    if (name_eq(name, "scf.if"))    return lower_scf_if(F, op);
    if (name_eq(name, "scf.while")) return lower_scf_while(F, op);

    // ---- scf.yield (operand-bearing): stash into the active yield ctx ----
    if (name_eq(name, "scf.yield")) {
        if (F->n_ys == 0) return false;
        YieldFrame *yf = &F->yield_stack[F->n_ys - 1];
        size_t no = MLIR_GetOpNumOperands(op);
        if ((int)no != yf->n) return false;
        for (size_t i = 0; i < no; i++) {
            int v;
            if (!vmap_get(F, MLIR_GetOpOperand(op, i), &v)) return false;
            emit_carrier_set(F, yf->carrier_ids[i], yf->carrier_vts[i], v);
        }
        return true;
    }

    // ---- llvm.select -----------------------------------------------------
    if (name_eq(name, "llvm.select")) {
        if (MLIR_GetOpNumOperands(op) != 3 || MLIR_GetOpNumResults(op) != 1)
            return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        int ci, ai, bi;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &ci)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 2), &bi)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_SELECT;
        o->valtype = vt;
        o->n_operands = 3;
        o->operands = (int *)malloc(3 * sizeof(int));
        // wasm select pops cond,b,a so we order operands as a,b,cond.
        o->operands[0] = ai; o->operands[1] = bi; o->operands[2] = ci;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.mlir.constant ------------------------------------------------
    if (name_eq(name, "llvm.mlir.constant")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        MLIR_AttributeHandle va = find_attr(op, "value");
        if (va == MLIR_INVALID_HANDLE) return false;
        MLIR_AttrKind ak = MLIR_GetAttributeKind(va);

        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_CONST;
        o->valtype = vt;
        if (ak == MLIR_ATTR_KIND_INTEGER) {
            o->i_const = MLIR_GetAttributeInteger(va);
        } else if (ak == MLIR_ATTR_KIND_FLOAT) {
            double dv = MLIR_GetAttributeFloat(va);
            // Pack the bit pattern through an integer of matching width;
            // stage 3 emits f32.const / f64.const using the same bytes.
            if (vt == WT_F32) {
                float fv = (float)dv;
                uint32_t bits;
                memcpy(&bits, &fv, 4);
                o->i_const = (int64_t)(uint64_t)bits;
            } else if (vt == WT_F64) {
                uint64_t bits;
                memcpy(&bits, &dv, 8);
                o->i_const = (int64_t)bits;
            } else {
                return false;
            }
        } else {
            return false;
        }
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- ub.poison ---------------------------------------------------------
    // Inserted by --lift-cf-to-scf for "doesn't-matter" yield values on
    // unreachable / pre-break paths. Lower to a typed zero constant.
    if (name_eq(name, "ub.poison")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_CONST;
        o->valtype = vt;
        o->i_const = 0;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.alloca: result = sp_def + frame_offset -----------------------
    if (name_eq(name, "llvm.alloca")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint32_t off;
        if (!amap_get(F, r, &off)) return false;
        if (F->sp_def < 0) return false;
        int koff = emit_const_i32(F, (int32_t)off);
        int addr = emit_add_i32(F, F->sp_def, koff);
        vmap_set(F, r, addr);
        return true;
    }

    // ---- llvm.store --------------------------------------------------------
    if (name_eq(name, "llvm.store")) {
        if (MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 1);
        MLIR_TypeHandle  vt_ty = MLIR_GetValueType(val);
        uint8_t vt = wasm_vt(F->ctx, vt_ty);
        unsigned sz = type_size_bytes(F->ctx, vt_ty);
        if (vt == 0 || sz == 0) return false;
        int va, pa;
        if (!vmap_get(F, val, &va)) return false;
        if (!vmap_get(F, ptr, &pa)) return false;

        unsigned align_log2;
        if (vt == WT_I32) {
            align_log2 = (sz == 4) ? 2 : (sz == 2) ? 1 : 0;
        } else if (vt == WT_I64) align_log2 = 3;
        else if (vt == WT_F32)   align_log2 = 2;
        else if (vt == WT_F64)   align_log2 = 3;
        else return false;

        wasmssa_op_t *o; (void)add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_STORE;
        o->valtype = vt;
        o->mem_size_bytes = sz;
        o->memory_align_log2 = align_log2;
        o->memory_offset = 0;
        o->n_operands = 2;
        o->operands = (int *)malloc(2 * sizeof(int));
        o->operands[0] = pa;   // address (stage 2 emits local.get for these in order)
        o->operands[1] = va;   // value
        return true;
    }

    // ---- llvm.load ---------------------------------------------------------
    if (name_eq(name, "llvm.load")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 0);
        MLIR_TypeHandle  rt = MLIR_GetValueType(r);
        uint8_t vt = wasm_vt(F->ctx, rt);
        unsigned sz = type_size_bytes(F->ctx, rt);
        if (vt == 0 || sz == 0) return false;
        int pa;
        if (!vmap_get(F, ptr, &pa)) return false;

        unsigned align_log2;
        if (vt == WT_I32) {
            align_log2 = (sz == 4) ? 2 : (sz == 2) ? 1 : 0;
        } else if (vt == WT_I64) align_log2 = 3;
        else if (vt == WT_F32)   align_log2 = 2;
        else if (vt == WT_F64)   align_log2 = 3;
        else return false;

        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_LOAD;
        o->valtype = vt;
        o->mem_size_bytes = sz;
        o->memory_align_log2 = align_log2;
        o->memory_offset = 0;
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = pa;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.sext (i8/i16/i32 -> i32; i32 -> i64; i8/i16 -> i64) ----------
    if (name_eq(name, "llvm.sext")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int in_w  = int_bits(F->ctx, MLIR_GetValueType(s));
        int out_w = int_bits(F->ctx, MLIR_GetValueType(r));
        if (in_w == 0 || out_w == 0 || in_w > out_w) return false;
        int sa;
        if (!vmap_get(F, s, &sa)) return false;

        // Step 1: if narrowing source is i8/i16 stored in i32, sign-extend
        // within i32 first (extend8_s=0xc0, extend16_s=0xc1).
        int cur_idx = sa;
        uint8_t cur_vt = wasm_vt(F->ctx, MLIR_GetValueType(s));
        if (in_w == 8 || in_w == 16) {
            wasmssa_op_t *o; int idx = add_op(F, &o);
            o->type = OP_TYPE_WASMSSA_UNOP;
            o->valtype = WT_I32;
            o->wasm_opcode = (in_w == 8) ? 0xc0 : 0xc1;
            o->n_operands = 1;
            o->operands = (int *)malloc(sizeof(int));
            o->operands[0] = cur_idx;
            o->has_result = true;
            cur_idx = idx; cur_vt = WT_I32;
        }
        // Step 2: if widening into i64, use i64.extend_i32_s (0xac).
        if (out_w == 64) {
            wasmssa_op_t *o; int idx = add_op(F, &o);
            o->type = OP_TYPE_WASMSSA_EXTEND_I32_S;
            o->valtype = WT_I64;
            o->n_operands = 1;
            o->operands = (int *)malloc(sizeof(int));
            o->operands[0] = cur_idx;
            o->has_result = true;
            cur_idx = idx;
        }
        vmap_set(F, r, cur_idx);
        return true;
    }

    // ---- llvm.ptrtoint / llvm.inttoptr -------------------------------------
    if (name_eq(name, "llvm.ptrtoint") || name_eq(name, "llvm.inttoptr")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        int sa;
        if (!vmap_get(F, s, &sa)) return false;
        // Pointers are i32. If matched, identity; if narrowing/widening, defer
        // to the existing trunc/zext-style lowering by emitting a wrap or
        // extend.
        if (in_vt == out_vt) {
            vmap_set(F, r, sa);
            return true;
        }
        if (in_vt == WT_I64 && out_vt == WT_I32) {
            int idx = emit_wrap_i64_to_i32(F, sa);
            vmap_set(F, r, idx);
            return true;
        }
        if (in_vt == WT_I32 && out_vt == WT_I64) {
            // unsigned extension: i64.extend_i32_u (0xad)
            wasmssa_op_t *o; int idx = add_op(F, &o);
            o->type = OP_TYPE_WASMSSA_UNOP;
            o->valtype = WT_I64;
            o->wasm_opcode = 0xad;
            o->n_operands = 1;
            o->operands = (int *)malloc(sizeof(int));
            o->operands[0] = sa;
            o->has_result = true;
            vmap_set(F, r, idx);
            return true;
        }
        return false;
    }


    // ---- llvm binary integer ops -------------------------------------------
    // Mapped to a generic wasmssa.binop carrying the actual wasm bytecode
    // byte. Op kind picks the row; valtype picks the column of the table.
    {
        // (op-name, i32-opcode, i64-opcode); 0 = unsupported
        struct { const char *nm; uint8_t i32op, i64op; } btab[] = {
            { "llvm.add",  0x6a, 0x7c },
            { "llvm.sub",  0x6b, 0x7d },
            { "llvm.mul",  0x6c, 0x7e },
            { "llvm.sdiv", 0x6d, 0x7f },
            { "llvm.udiv", 0x6e, 0x80 },
            { "llvm.srem", 0x6f, 0x81 },
            { "llvm.urem", 0x70, 0x82 },
            { "llvm.and",  0x71, 0x83 },
            { "llvm.or",   0x72, 0x84 },
            { "llvm.xor",  0x73, 0x85 },
            { "llvm.shl",  0x74, 0x86 },
            { "llvm.ashr", 0x75, 0x87 },
            { "llvm.lshr", 0x76, 0x88 },
        };
        for (size_t k = 0; k < sizeof(btab)/sizeof(btab[0]); k++) {
            if (!name_eq(name, btab[k].nm)) continue;
            if (MLIR_GetOpNumResults(op) != 1 ||
                MLIR_GetOpNumOperands(op) != 2) return false;
            MLIR_ValueHandle r  = MLIR_GetOpResult(op, 0);
            MLIR_ValueHandle a  = MLIR_GetOpOperand(op, 0);
            MLIR_ValueHandle b  = MLIR_GetOpOperand(op, 1);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            uint8_t opc = (vt == WT_I32) ? btab[k].i32op
                        : (vt == WT_I64) ? btab[k].i64op : 0;
            if (opc == 0) return false;
            int ai, bi;
            if (!vmap_get(F, a, &ai) || !vmap_get(F, b, &bi)) return false;
            wasmssa_op_t *o; int idx = add_op(F, &o);
            o->type = OP_TYPE_WASMSSA_BINOP;
            o->valtype = vt;
            o->wasm_opcode = opc;
            o->n_operands = 2;
            o->operands = (int *)malloc(2 * sizeof(int));
            o->operands[0] = ai; o->operands[1] = bi;
            o->has_result = true;
            vmap_set(F, r, idx);
            return true;
        }
    }

    // ---- llvm binary float ops ---------------------------------------------
    {
        struct { const char *nm; uint8_t f32op, f64op; } ftab[] = {
            { "llvm.fadd",  0x92, 0xa0 },
            { "llvm.fsub",  0x93, 0xa1 },
            { "llvm.fmul",  0x94, 0xa2 },
            { "llvm.fdiv",  0x95, 0xa3 },
        };
        for (size_t k = 0; k < sizeof(ftab)/sizeof(ftab[0]); k++) {
            if (!name_eq(name, ftab[k].nm)) continue;
            if (MLIR_GetOpNumResults(op) != 1 ||
                MLIR_GetOpNumOperands(op) != 2) return false;
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            uint8_t opc = (vt == WT_F32) ? ftab[k].f32op
                        : (vt == WT_F64) ? ftab[k].f64op : 0;
            if (opc == 0) return false;
            int ai, bi;
            if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &ai)) return false;
            if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &bi)) return false;
            wasmssa_op_t *o; int idx = add_op(F, &o);
            o->type = OP_TYPE_WASMSSA_BINOP;
            o->valtype = vt;
            o->wasm_opcode = opc;
            o->n_operands = 2;
            o->operands = (int *)malloc(2 * sizeof(int));
            o->operands[0] = ai; o->operands[1] = bi;
            o->has_result = true;
            vmap_set(F, r, idx);
            return true;
        }
    }

    // ---- llvm float compares -----------------------------------------------
    if (name_eq(name, "llvm.fcmp")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        uint8_t opvt = wasm_vt(F->ctx, MLIR_GetValueType(a));
        // FCmpPredicate enum: oeq=1, ogt=2, oge=3, olt=4, ole=5, one=6,
        // une=11, ueq=8 (unordered variants); we only support ordered.
        // wasm has eq, ne, lt, gt, le, ge (ordered). Map oeq->eq, etc.
        MLIR_AttributeHandle pa = find_attr(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE) return false;
        int64_t pred = MLIR_GetAttributeInteger(pa);
        // Map LLVM fcmp predicate -> wasm op offset (eq=0,ne=1,lt=2,gt=3,le=4,ge=5).
        int8_t off;
        switch (pred) {
        case 1: off = 0; break;  // oeq
        case 2: off = 3; break;  // ogt
        case 3: off = 5; break;  // oge
        case 4: off = 2; break;  // olt
        case 5: off = 4; break;  // ole
        case 6: off = 1; break;  // one
        default: return false;
        }
        uint8_t opc = (opvt == WT_F32) ? (uint8_t)(0x5b + off)
                    : (opvt == WT_F64) ? (uint8_t)(0x61 + off) : 0;
        if (opc == 0) return false;
        int ai, bi;
        if (!vmap_get(F, a, &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &bi)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_BINOP;
        o->valtype = WT_I32;
        o->wasm_opcode = opc;
        o->n_operands = 2;
        o->operands = (int *)malloc(2 * sizeof(int));
        o->operands[0] = ai; o->operands[1] = bi;
        o->has_result = true;
        vmap_set(F, MLIR_GetOpResult(op, 0), idx);
        return true;
    }

    // ---- llvm.fpext / llvm.fptrunc / llvm.fneg ----------------------------
    if (name_eq(name, "llvm.fpext") || name_eq(name, "llvm.fptrunc") ||
        name_eq(name, "llvm.fneg")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        uint8_t opc = 0;
        if (name_eq(name, "llvm.fpext") && in_vt == WT_F32 && out_vt == WT_F64) opc = 0xbb;
        else if (name_eq(name, "llvm.fptrunc") && in_vt == WT_F64 && out_vt == WT_F32) opc = 0xb6;
        else if (name_eq(name, "llvm.fneg") && in_vt == WT_F32 && out_vt == WT_F32) opc = 0x8c;
        else if (name_eq(name, "llvm.fneg") && in_vt == WT_F64 && out_vt == WT_F64) opc = 0x9a;
        else return false;
        int sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_UNOP;
        o->valtype = out_vt;
        o->wasm_opcode = opc;
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = sa;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.sitofp / llvm.uitofp / llvm.fptosi / llvm.fptoui ------------
    if (name_eq(name, "llvm.sitofp") || name_eq(name, "llvm.uitofp") ||
        name_eq(name, "llvm.fptosi") || name_eq(name, "llvm.fptoui")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        bool is_signed = name_eq(name, "llvm.sitofp") || name_eq(name, "llvm.fptosi");
        uint8_t opc = 0;
        if (name_eq(name, "llvm.sitofp") || name_eq(name, "llvm.uitofp")) {
            // f{32,64}.convert_i{32,64}_{s,u}
            if (out_vt == WT_F32 && in_vt == WT_I32) opc = is_signed ? 0xb2 : 0xb3;
            else if (out_vt == WT_F32 && in_vt == WT_I64) opc = is_signed ? 0xb4 : 0xb5;
            else if (out_vt == WT_F64 && in_vt == WT_I32) opc = is_signed ? 0xb7 : 0xb8;
            else if (out_vt == WT_F64 && in_vt == WT_I64) opc = is_signed ? 0xb9 : 0xba;
        } else {
            // i{32,64}.trunc_f{32,64}_{s,u}
            if (out_vt == WT_I32 && in_vt == WT_F32) opc = is_signed ? 0xa8 : 0xa9;
            else if (out_vt == WT_I32 && in_vt == WT_F64) opc = is_signed ? 0xaa : 0xab;
            else if (out_vt == WT_I64 && in_vt == WT_F32) opc = is_signed ? 0xae : 0xaf;
            else if (out_vt == WT_I64 && in_vt == WT_F64) opc = is_signed ? 0xb0 : 0xb1;
        }
        if (opc == 0) return false;
        int sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_UNOP;
        o->valtype = out_vt;
        o->wasm_opcode = opc;
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = sa;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.icmp ---------------------------------------------------------
    // Result is i1 (encoded as i32 in wasm). Operand valtype determines
    // the i32.* vs i64.* opcode family.
    if (name_eq(name, "llvm.icmp")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(op, 1);
        uint8_t opvt = wasm_vt(F->ctx, MLIR_GetValueType(a));
        // LLVM ICmpPredicate enum: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5,
        // ult=6, ule=7, ugt=8, uge=9.
        MLIR_AttributeHandle pa = find_attr(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE) return false;
        int64_t pred = MLIR_GetAttributeInteger(pa);
        // i32: 0x46..0x4f ; i64: 0x51..0x5a (skip eqz at 0x45/0x50).
        // Map predicate -> bytecode offset within (eq, ne, lt_s, lt_u,
        // gt_s, gt_u, le_s, le_u, ge_s, ge_u).
        // LLVM order:  eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge
        // wasm order:  eq, ne, lt_s, lt_u, gt_s, gt_u, le_s, le_u, ge_s, ge_u
        static const int8_t llvm_to_wasm[10] = {0, 1, 2, 6, 4, 8, 3, 7, 5, 9};
        if (pred < 0 || pred >= 10) return false;
        int wasm_idx = llvm_to_wasm[pred];
        uint8_t opc = (opvt == WT_I32) ? (uint8_t)(0x46 + wasm_idx)
                    : (opvt == WT_I64) ? (uint8_t)(0x51 + wasm_idx) : 0;
        if (opc == 0) return false;
        int ai, bi;
        if (!vmap_get(F, a, &ai) || !vmap_get(F, b, &bi)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_BINOP;
        o->valtype = WT_I32;  // result type
        o->wasm_opcode = opc;
        o->n_operands = 2;
        o->operands = (int *)malloc(2 * sizeof(int));
        o->operands[0] = ai; o->operands[1] = bi;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.trunc / llvm.zext -------------------------------------------
    // i64 -> i32 trunc:    i32.wrap_i64    (0xa7)
    // i32 -> i64 zext:     i64.extend_i32_u(0xad)
    // smaller-int trunc/zext within i32: no-op (wasm has no sub-i32 reg).
    if (name_eq(name, "llvm.trunc") || name_eq(name, "llvm.zext")) {
        bool is_zext = name_eq(name, "llvm.zext");
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt == 0 || out_vt == 0) return false;
        int sa;
        if (!vmap_get(F, s, &sa)) return false;
        if (in_vt == out_vt) {
            // No-op: result alias of operand.
            vmap_set(F, r, sa);
            return true;
        }
        uint8_t opc;
        if (!is_zext && in_vt == WT_I64 && out_vt == WT_I32) opc = 0xa7;
        else if (is_zext && in_vt == WT_I32 && out_vt == WT_I64) opc = 0xad;
        else return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_UNOP;
        o->valtype = out_vt;
        o->wasm_opcode = opc;
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = sa;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.return -------------------------------------------------------
    if (name_eq(name, "llvm.return")) {
        size_t no = MLIR_GetOpNumOperands(op);
        int retv = -1;
        if (no == 1) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, 0);
            if (!vmap_get(F, v, &retv)) return false;
        } else if (no != 0) {
            return false;
        }

        // Epilogue: restore __stack_pointer = sp_def + frame_size.
        if (F->frame_size > 0 && F->sp_def >= 0) {
            int kf = emit_const_i32(F, (int32_t)F->frame_size);
            int restored = emit_add_i32(F, F->sp_def, kf);
            emit_global_set(F, /*sp*/0, restored);
        }

        wasmssa_op_t *o; (void)add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_RETURN;
        if (no == 1) {
            o->n_operands = 1;
            o->operands = (int *)malloc(sizeof(int));
            o->operands[0] = retv;
        }
        return true;
    }

    // ---- llvm.call (direct, fixed-arity) -----------------------------------
    if (name_eq(name, "llvm.call")) {
        MLIR_AttributeHandle callee = find_attr(op, "callee");
        if (callee == MLIR_INVALID_HANDLE) return false;
        MLIR_AttributeHandle var_callee_type = find_attr(op, "var_callee_type");
        if (var_callee_type != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(var_callee_type);
            if (MLIR_GetTypeFunctionIsVarArg(fty)) return false;
        }
        string nm = MLIR_GetAttributeAsString(F->ctx, callee);
        // SymbolRefAttr prints as `@name`.
        const char *cname = nm.str; size_t cn = nm.size;
        if (cn > 0 && cname[0] == '@') { cname++; cn--; }
        char *cstr = xstrdupn(cname, cn);

        size_t no = MLIR_GetOpNumOperands(op);
        int *opnds = (int *)malloc((no ? no : 1) * sizeof(int));
        for (size_t i = 0; i < no; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(op, i), &opnds[i])) {
                free(opnds); free(cstr);
                return false;
            }
        }

        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_CALL;
        o->call_target = cstr;
        o->n_operands = (int)no;
        o->operands = opnds;

        size_t nr = MLIR_GetOpNumResults(op);
        if (nr == 1) {
            MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            if (vt == 0) return false;
            o->valtype = vt;
            o->has_result = true;
            vmap_set(F, r, idx);
        } else if (nr != 0) {
            return false;
        }
        return true;
    }

    // ---- llvm.getelementptr -----------------------------------------------
    if (name_eq(name, "llvm.getelementptr")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        if (MLIR_GetOpNumOperands(op) < 1) return false;
        MLIR_ValueHandle r    = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle base = MLIR_GetOpOperand(op, 0);
        int addr;
        if (!vmap_get(F, base, &addr)) return false;
        MLIR_AttributeHandle eta = find_attr(op, "elem_type");
        if (eta == MLIR_INVALID_HANDLE) return false;
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(eta);
        MLIR_AttributeHandle ria = find_attr(op, "rawConstantIndices");
        if (ria == MLIR_INVALID_HANDLE) return false;
        size_t n_idx = 0;
        int32_t *cidx = parse_dense_i32_array(
            MLIR_GetAttributeAsString(F->ctx, ria), &n_idx);
        if (!cidx || n_idx == 0) { free(cidx); return false; }

        size_t op_idx = 1;
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            int dyn_def = -1;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(op)) { free(cidx); return false; }
                MLIR_ValueHandle ov = MLIR_GetOpOperand(op, op_idx++);
                if (!vmap_get(F, ov, &dyn_def)) { free(cidx); return false; }
                uint8_t ovt = wasm_vt(F->ctx, MLIR_GetValueType(ov));
                if (ovt == WT_I64) dyn_def = emit_wrap_i64_to_i32(F, dyn_def);
                else if (ovt != WT_I32) { free(cidx); return false; }
            }

            if (i == 0) {
                unsigned esz = type_size_bytes(F->ctx, elem_ty);
                if (esz == 0) { free(cidx); return false; }
                if (is_dyn) {
                    int k = emit_const_i32(F, (int32_t)esz);
                    int m = emit_mul_i32(F, dyn_def, k);
                    addr = emit_add_i32(F, addr, m);
                } else if (cidx[i] != 0) {
                    int k = emit_const_i32(F, (int32_t)((int64_t)cidx[i] * (int32_t)esz));
                    addr = emit_add_i32(F, addr, k);
                }
            } else if (MLIR_IsTypeLLVMStruct(cur_ty)) {
                if (is_dyn) { free(cidx); return false; }
                unsigned off = struct_field_offset(F->ctx, cur_ty, (size_t)cidx[i]);
                if (off != 0) {
                    int k = emit_const_i32(F, (int32_t)off);
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = MLIR_GetTypeLLVMStructField(cur_ty, (size_t)cidx[i]);
            } else if (MLIR_IsTypeLLVMArray(cur_ty)) {
                MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(cur_ty);
                unsigned esz = type_size_bytes(F->ctx, et);
                if (esz == 0) { free(cidx); return false; }
                if (is_dyn) {
                    int k = emit_const_i32(F, (int32_t)esz);
                    int m = emit_mul_i32(F, dyn_def, k);
                    addr = emit_add_i32(F, addr, m);
                } else if (cidx[i] != 0) {
                    int k = emit_const_i32(F, (int32_t)((int64_t)cidx[i] * (int32_t)esz));
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = et;
            } else {
                free(cidx);
                return false;
            }
        }
        free(cidx);
        vmap_set(F, r, addr);
        return true;
    }

    // ---- llvm.mlir.addressof ----------------------------------------------
    // Result is a pointer (i32) to a data global. We emit an ADDRESSOF op
    // that stage 3 will lower as `i32.const <padded sleb>` plus a
    // R_WASM_MEMORY_ADDR_SLEB reloc to the named global symbol.
    if (name_eq(name, "llvm.mlir.addressof")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_AttributeHandle ta = find_attr(op, "global_name");
        if (ta == MLIR_INVALID_HANDLE)
            ta = find_attr(op, "value");
        if (ta == MLIR_INVALID_HANDLE) {
            // Fallback: any string attr.
            size_t na = MLIR_GetOpNumAttributes(op);
            for (size_t i = 0; i < na; i++) {
                MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
                if (MLIR_GetAttributeKind(a) == MLIR_ATTR_KIND_STRING) {
                    ta = a; break;
                }
            }
        }
        if (ta == MLIR_INVALID_HANDLE) return false;
        string ts = MLIR_GetAttributeString(ta);
        // Strip leading '@' if present (FlatSymbolRefAttr prints with '@').
        if (ts.size && ts.str[0] == '@') {
            ts.str++;
            ts.size--;
        }

        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_ADDRESSOF;
        o->valtype = WT_I32;
        o->call_target = xstrdup_str(ts);  // reuse field for symbol name
        o->has_result = true;
        vmap_set(F, MLIR_GetOpResult(op, 0), idx);
        return true;
    }

    // ---- llvm.mlir.zero ---------------------------------------------------
    // Pointer / integer zero. f32/f64 zero would also fall through here but
    // currently only ptr/int paths are exercised (the float test uses a
    // typed llvm.mlir.constant for 0.0).
    if (name_eq(name, "llvm.mlir.zero")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_CONST;
        o->valtype = vt;
        o->i_const = 0;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.sqrt ---------------------------------------------------
    if (name_eq(name, "llvm.intr.sqrt")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        uint8_t opc = (vt == WT_F32) ? 0x91 : (vt == WT_F64) ? 0x9f : 0;
        if (opc == 0) return false;
        int sa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &sa)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_UNOP;
        o->valtype = vt;
        o->wasm_opcode = opc;
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = sa;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
    }

    fprintf(stderr, "wasmssa-lower: unsupported op '%.*s'\n",
            (int)name.size, name.str);
    return false;
}

// =============================================================================
// Function lowering: pre-walk for alloca offsets, then op-by-op walk.
// =============================================================================
static bool prewalk_block(FnCtx *F, MLIR_BlockHandle blk);

static bool prewalk_op(FnCtx *F, MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    if (name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
        name_eq(n, "llvm.switch") ||
        name_eq(n, "cf.br") || name_eq(n, "cf.cond_br") ||
        name_eq(n, "cf.switch")) {
        fprintf(stderr,
                "wasmssa-lower: control flow not lifted to scf (%.*s)\n",
                (int)n.size, n.str);
        return false;
    }
    if (name_eq(n, "llvm.alloca")) {
        MLIR_TypeHandle et = MLIR_INVALID_HANDLE;
        MLIR_AttributeHandle ea = find_attr(op, "elem_type");
        if (ea != MLIR_INVALID_HANDLE) et = MLIR_GetAttributeTypeValue(ea);
        unsigned esz = et != MLIR_INVALID_HANDLE
                           ? type_size_bytes(F->ctx, et) : 0;
        if (esz == 0) {
            fprintf(stderr,
                    "wasmssa-lower: alloca of unsupported element type\n");
            return false;
        }
        int64_t cnt = 1;
        if (MLIR_GetOpNumOperands(op) >= 1) {
            MLIR_ValueHandle co = MLIR_GetOpOperand(op, 0);
            MLIR_OpHandle    cd = MLIR_GetValueDefiningOp(co);
            if (cd == MLIR_INVALID_HANDLE) return false;
            if (!name_eq(MLIR_GetOpName(cd), "llvm.mlir.constant")) {
                fprintf(stderr,
                        "wasmssa-lower: dynamic alloca size not supported\n");
                return false;
            }
            MLIR_AttributeHandle va = find_attr(cd, "value");
            cnt = MLIR_GetAttributeInteger(va);
        }
        unsigned ealign = type_align_bytes(F->ctx, et);
        if (ealign < 4) ealign = 4;
        F->frame_size = align_up(F->frame_size, ealign);
        uint32_t off = F->frame_size;
        F->frame_size += (uint32_t)(esz * cnt);
        amap_set(F, MLIR_GetOpResult(op, 0), off);
    }
    // Recurse into nested regions for nested allocas.
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle rg = MLIR_GetOpRegion(op, r);
        size_t nblk = MLIR_GetRegionNumBlocks(rg);
        for (size_t b = 0; b < nblk; b++) {
            if (!prewalk_block(F, MLIR_GetRegionBlock(rg, b))) return false;
        }
    }
    return true;
}

static bool prewalk_block(FnCtx *F, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < nops; i++) {
        if (!prewalk_op(F, MLIR_GetBlockOp(blk, i))) return false;
    }
    return true;
}

static bool prewalk_func(FnCtx *F, MLIR_BlockHandle entry) {
    if (!prewalk_block(F, entry)) return false;
    F->frame_size = (F->frame_size + 15) & ~15u;
    return true;
}

static bool lower_block(FnCtx *F, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < nops; i++) {
        if (!lower_op(F, MLIR_GetBlockOp(blk, i))) return false;
    }
    return true;
}

static bool lower_function(MLIR_Context *ctx, wasmssa_func_t *df,
                           MLIR_OpHandle fn,
                           const uint8_t *param_types, size_t n_params) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx;
    F.f = df;
    F.n_params = n_params;
    F.sp_def = -1;

    MLIR_RegionHandle body = MLIR_GetOpRegion(fn, 0);
    MLIR_BlockHandle  entry = MLIR_GetRegionBlock(body, 0);

    // Bind parameter SSA values to ssa_def_idx 0..n_params-1.
    for (size_t i = 0; i < n_params; i++) {
        vmap_set(&F, MLIR_GetBlockArg(entry, i), (int)i);
    }

    if (!prewalk_func(&F, entry)) goto fail;

    // Prologue: __stack_pointer -= frame_size; sp_def = result.
    if (F.frame_size > 0) {
        int sp_orig = emit_global_get(&F, /*sp*/0);
        int kf      = emit_const_i32(&F, (int32_t)F.frame_size);
        int sp_new  = emit_sub_i32(&F, sp_orig, kf);
        emit_global_set(&F, /*sp*/0, sp_new);
        F.sp_def = sp_new;
    }

    size_t nops = MLIR_GetBlockNumOps(entry);
    for (size_t i = 0; i < nops; i++) {
        if (!lower_op(&F, MLIR_GetBlockOp(entry, i))) goto fail;
    }

    free(F.vmap); free(F.amap);
    // Hand off carrier metadata to the func.
    df->n_carriers = F.n_cvt;
    if (F.n_cvt) {
        df->carrier_vts = (uint8_t *)malloc(F.n_cvt);
        memcpy(df->carrier_vts, F.carrier_vts, F.n_cvt);
    }
    free(F.carrier_vts); free(F.yield_stack);
    (void)param_types;
    return true;
fail:
    free(F.vmap); free(F.amap);
    free(F.carrier_vts); free(F.yield_stack);
    return false;
}

// =============================================================================
// Module walker.
// =============================================================================
static bool sig_for_func(MLIR_Context *ctx, MLIR_OpHandle fn,
                         uint8_t **out_p, size_t *out_np,
                         uint8_t **out_r, size_t *out_nr) {
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    if (ftya == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
    size_t ni = MLIR_GetTypeFunctionNumInputs(fty);
    size_t no = MLIR_GetTypeFunctionNumResults(fty);
    uint8_t *p = (uint8_t *)malloc(ni ? ni : 1);
    for (size_t i = 0; i < ni; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionInput(fty, i));
        if (v == 0) { free(p); return false; }
        p[i] = v;
    }
    uint8_t *r = (uint8_t *)malloc(no ? no : 1);
    for (size_t i = 0; i < no; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionResult(fty, i));
        if (v == 0) { free(p); free(r); return false; }
        r[i] = v;
    }
    *out_p = p; *out_np = ni;
    *out_r = r; *out_nr = no;
    return true;
}

// Return alignment requirement (power-of-2 exponent) for a global of
// the given LLVM type. 0 = byte-aligned.
static uint32_t align_pow_for_type(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    unsigned a = type_align_bytes(ctx, ty);
    uint32_t p = 0;
    while ((1u << p) < a) p++;
    return p;
}

// Lower one llvm.mlir.global op into a wasm_global_t entry. Returns
// false on failure.
static bool lower_global(MLIR_Context *ctx, wasmssa_module_t *out,
                         MLIR_OpHandle op) {
    MLIR_AttributeHandle sa = find_attr(op, "sym_name");
    if (sa == MLIR_INVALID_HANDLE) return false;
    string sym = MLIR_GetAttributeString(sa);

    MLIR_AttributeHandle ga = find_attr(op, "global_type");
    MLIR_TypeHandle gty = MLIR_INVALID_HANDLE;
    if (ga != MLIR_INVALID_HANDLE) gty = MLIR_GetAttributeTypeValue(ga);

    wasm_global_t *g = wasmssa_module_add_global(out);
    g->name = xstrdup_str(sym);
    if (gty != MLIR_INVALID_HANDLE) {
        g->size = type_size_bytes(ctx, gty);
        g->align_pow = align_pow_for_type(ctx, gty);
    }

    // Constness: presence of attr "constant" (any value).
    g->is_const = (find_attr(op, "constant") != MLIR_INVALID_HANDLE);

    MLIR_AttributeHandle va = find_attr(op, "value");
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind ak = MLIR_GetAttributeKind(va);
        if (ak == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(va);
            g->size = (uint32_t)s.size;
            g->data = (uint8_t *)malloc(g->size ? g->size : 1);
            memcpy(g->data, s.str, g->size);
            if (g->align_pow == 0) g->align_pow = 0;
            return true;
        }
        if (ak == MLIR_ATTR_KIND_INTEGER || ak == MLIR_ATTR_KIND_FLOAT) {
            if (g->size == 0) return false;
            g->data = (uint8_t *)calloc(1, g->size);
            uint64_t bits = 0;
            if (ak == MLIR_ATTR_KIND_INTEGER) {
                bits = (uint64_t)MLIR_GetAttributeInteger(va);
            } else {
                double d = MLIR_GetAttributeFloat(va);
                if (g->size == 4) {
                    float f = (float)d; uint32_t b32;
                    memcpy(&b32, &f, 4); bits = b32;
                } else if (g->size == 8) {
                    memcpy(&bits, &d, 8);
                } else return false;
            }
            for (uint32_t i = 0; i < g->size && i < 8; i++)
                g->data[i] = (uint8_t)(bits >> (8 * i));
            return true;
        }
        // Unknown attr kind -- fall through to zero-init.
    }

    // Region-init globals: walk for an llvm.mlir.addressof + llvm.return.
    if (MLIR_GetOpNumRegions(op) > 0) {
        MLIR_RegionHandle rgn = MLIR_GetOpRegion(op, 0);
        if (MLIR_GetRegionNumBlocks(rgn) > 0) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(rgn, 0);
            size_t nb = MLIR_GetBlockNumOps(blk);
            // Map block-internal ssa value -> tagged "addressof @target".
            // We only support the trivial pattern produced by tinyc:
            //   %0 = llvm.mlir.addressof @other : !llvm.ptr
            //   llvm.return %0 : !llvm.ptr
            char *pending_target = NULL;
            for (size_t bi = 0; bi < nb; bi++) {
                MLIR_OpHandle bop = MLIR_GetBlockOp(blk, bi);
                string bn = MLIR_GetOpName(bop);
                if (name_eq(bn, "llvm.mlir.addressof")) {
                    MLIR_AttributeHandle ta = find_attr(bop, "global_name");
                    if (ta == MLIR_INVALID_HANDLE)
                        ta = find_attr(bop, "value");  // upstream attr name
                    if (ta == MLIR_INVALID_HANDLE) {
                        // Fallback: any FlatSymbolRef-shaped attr.
                        size_t na = MLIR_GetOpNumAttributes(bop);
                        for (size_t i = 0; i < na; i++) {
                            MLIR_AttributeHandle a = MLIR_GetOpAttribute(bop, i);
                            if (MLIR_GetAttributeKind(a) == MLIR_ATTR_KIND_STRING) {
                                ta = a; break;
                            }
                        }
                    }
                    if (ta == MLIR_INVALID_HANDLE) return false;
                    string ts = MLIR_GetAttributeString(ta);
                    if (ts.size && ts.str[0] == '@') {
                        ts.str++; ts.size--;
                    }
                    free(pending_target);
                    pending_target = xstrdup_str(ts);
                } else if (name_eq(bn, "llvm.return")) {
                    if (!pending_target) break;
                    g->size = 4;
                    if (g->align_pow == 0) g->align_pow = 2;
                    g->data = (uint8_t *)calloc(1, g->size);
                    wasm_global_add_reloc(g, 0, pending_target, 0);
                    free(pending_target);
                    return true;
                }
            }
            free(pending_target);
        }
    }

    // No initializer found (e.g. uninitialized scalar global). Zero-init.
    if (g->size == 0) return false;
    g->data = (uint8_t *)calloc(1, g->size);
    return true;
}

wasmssa_module_t *mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx,
                                             MLIR_OpHandle module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    wasmssa_module_t *out = wasmssa_module_new();

    // Pass 0: globals (llvm.mlir.global). Must precede func lowering so
    // addressof can resolve symbol names to an existing global table
    // entry — though the lowering only stores the name and stage 3
    // matches by name during DATA emission.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
        if (!lower_global(ctx, out, op)) {
            fprintf(stderr, "wasmssa-lower: failed to lower global\n");
            wasmssa_module_free(out); return NULL;
        }
    }

    // Pass 1: imports (declared funcs) first so they get the low function
    // indices in the wasm function-index space.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, op, &p, &np, &r, &nr)) {
            wasmssa_module_free(out); return NULL;
        }
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        char *nm = xstrdup_str(MLIR_GetAttributeString(sa));

        wasmssa_func_t *df = wasmssa_module_add_func(out);
        df->name = nm;
        df->imported = true;
        df->n_params = np; df->param_types = p;
        df->n_results = nr; df->result_types = r;
    }

    // Pass 2: defined funcs.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (!has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, op, &p, &np, &r, &nr)) {
            wasmssa_module_free(out); return NULL;
        }
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        string sym = MLIR_GetAttributeString(sa);
        bool is_main = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
        char *nm = is_main ? xstrdupn("__original_main", 15)
                           : xstrdup_str(sym);

        wasmssa_func_t *df = wasmssa_module_add_func(out);
        df->name = nm;
        df->imported = false;
        df->exported = is_main;
        df->n_params = np; df->param_types = p;
        df->n_results = nr; df->result_types = r;

        if (!lower_function(ctx, df, op, p, np)) {
            wasmssa_module_free(out); return NULL;
        }
    }

    return out;
}

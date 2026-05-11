// Stage 1 of the native LLVM->WASM pipeline:
// LLVM-dialect builtin.module --(walk)--> wasmssa-form `builtin.module`.
//
// Each `llvm.func` / `llvm.mlir.global` is lowered directly to `wasmssa.*`
// MLIR ops appended into the output module body. Inside a function, each
// emit-helper / inline call site builds a stack-local `wasmssa_op_t`
// describing one op and hands it to `commit_op`, which materializes the
// matching MLIR op into the function's body block and records its result
// value in the per-function `ssa_vals` table. SSA value indexing inside
// that table (shared with stage 2 in `mlir_stackify_wasmssa.c`):
//
//   ssa_def_idx in [0, n_params)              -> function parameter i
//   ssa_def_idx in [n_params, n_params+n_ops) -> result of op (idx-n_params)
//
// No module-level or per-function scratch buffer is retained: ops flow
// straight from lower_op into the MLIR module under construction.
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
#include "mlir_op_names.h"
#include "mlir_wasm_pipeline.h"

#include <base/arena.h>
#include <base/string.h>

// =============================================================================
// Private working representation for this stage. `wasmssa_op_t` is a
// stack-local descriptor handed to `commit_op` once per emitted op. It
// never escapes this file.
// =============================================================================
typedef struct {
    MLIR_OpType type;
    uint8_t     valtype;

    int64_t  i_const;
    uint32_t global_idx;
    uint32_t memory_offset;
    uint32_t memory_align_log2;
    uint32_t mem_size_bytes;
    char    *call_target;
    uint8_t  wasm_opcode;
    uint32_t br_depth;
    uint32_t carrier_id;

    uint8_t *sig_params;
    uint8_t *sig_results;
    size_t   n_sig_params, n_sig_results;

    int   *operands;
    int    n_operands;

    bool has_result;
} wasmssa_op_t;

// Module-level emit context: the output MLIR module's body block plus
// a list of function names already emitted (so `is_function_symbol`
// can distinguish `llvm.mlir.addressof` references between funcs and
// data globals).
typedef struct {
    MLIR_Context    *ctx;
    Arena           *arena;
    MLIR_BlockHandle body;
    char           **func_names;
    size_t           n_func_names, c_func_names;
} ModCtx;

static void modctx_add_func_name(ModCtx *m, const char *name) {
    if (m->n_func_names == m->c_func_names) {
        m->c_func_names = m->c_func_names ? m->c_func_names * 2 : 8;
        m->func_names = (char **)realloc(m->func_names,
                                         m->c_func_names * sizeof(char *));
    }
    size_t n = strlen(name);
    char *copy = (char *)malloc(n + 1);
    memcpy(copy, name, n + 1);
    m->func_names[m->n_func_names++] = copy;
}
static void modctx_free_contents(ModCtx *m) {
    for (size_t i = 0; i < m->n_func_names; i++) free(m->func_names[i]);
    free(m->func_names);
}

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
    if (s.size == 5 && memcmp(s.str, "index", 5) == 0) return WT_I32;
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
    Arena          *arena;   // for hex-encoded attribute strings
    ModCtx         *mod;     // parent module emit context (for func-name lookups)
    size_t          n_params;

    // Direct-MLIR emission state. body_block accumulates wasmssa.* ops; the
    // unified ssa_vals[] is indexed by ssa_def_idx and holds the MLIR
    // SSA value (or MLIR_INVALID_HANDLE for ops without a result).
    MLIR_BlockHandle  body_block;
    MLIR_ValueHandle *ssa_vals; size_t n_ssa, c_ssa;

    // Map MLIR_ValueHandle -> ssa_def_idx (index into ssa_vals[]).
    VMapEntry *vmap; size_t n_vmap, c_vmap;

    // Map MLIR_ValueHandle (alloca result) -> shadow-stack frame offset.
    AMapEntry *amap; size_t n_amap, c_amap;

    uint32_t  frame_size;
    int       sp_def;       // ssa_def_idx of the post-decrement SP (-1 if no frame)

    // Variadic ABI support.
    int       va_list_param_idx;  // ssa_def_idx of the hidden trailing va_list i32 param, or -1
    uint32_t  va_buf_size;        // max bytes needed for variadic call buffer
    uint32_t  va_buf_offset;      // offset within shadow frame for the variadic buffer

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

// Forward decls used by commit_op (defined further down with the other
// MLIR-emit helpers).
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt);
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v);
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v);
static MLIR_AttributeHandle attr_s_cstr(MLIR_Context *ctx, const char *name, const char *v);
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v);
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen);
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n);
static MLIR_OpHandle make_op(MLIR_Context *ctx, MLIR_OpType type,
                             MLIR_AttributeHandle *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt,
                             MLIR_ValueHandle *out_result);

// Materialize a wasmssa op directly into the function's MLIR body block
// and record its result (if any) in F->ssa_vals. Returns the new
// ssa_def_idx (n_params + result_vals_index). The op's operand[] indices
// reference earlier ssa_def_idx values; this routine resolves them to
// MLIR_ValueHandles via F->ssa_vals.
static int commit_op(FnCtx *F, wasmssa_op_t *o) {
    MLIR_Context *ctx = F->ctx;
    MLIR_ValueHandle ops_buf[16];
    MLIR_ValueHandle *ops = ops_buf;
    size_t n_ops = (size_t)(o->n_operands < 0 ? 0 : o->n_operands);
    if (n_ops > 16) {
        ops = (MLIR_ValueHandle *)arena_alloc(F->arena, n_ops * sizeof(MLIR_ValueHandle));
    }
    for (size_t k = 0; k < n_ops; k++) {
        int idx = o->operands[k];
        ops[k] = (idx >= 0 && (size_t)idx < F->n_ssa) ? F->ssa_vals[idx]
                                                     : MLIR_INVALID_HANDLE;
    }
    MLIR_AttributeHandle as[8];
    size_t nas = 0;
    as[nas++] = attr_i32(ctx, "valtype", o->valtype);
    switch (o->type) {
    case OP_TYPE_WASMSSA_CONST:
        as[nas++] = attr_i64(ctx, "value", o->i_const);
        break;
    case OP_TYPE_WASMSSA_BINOP:
    case OP_TYPE_WASMSSA_UNOP:
        as[nas++] = attr_i32(ctx, "wasm_opcode", o->wasm_opcode);
        break;
    case OP_TYPE_WASMSSA_LOAD:
    case OP_TYPE_WASMSSA_STORE:
        as[nas++] = attr_i32(ctx, "memory_offset",     (int64_t)o->memory_offset);
        as[nas++] = attr_i32(ctx, "memory_align_log2", (int64_t)o->memory_align_log2);
        as[nas++] = attr_i32(ctx, "mem_size_bytes",    (int64_t)o->mem_size_bytes);
        break;
    case OP_TYPE_WASMSSA_GLOBAL_GET:
    case OP_TYPE_WASMSSA_GLOBAL_SET:
        as[nas++] = attr_i32(ctx, "global_idx", (int64_t)o->global_idx);
        break;
    case OP_TYPE_WASMSSA_CALL:
    case OP_TYPE_WASMSSA_ADDRESSOF:
    case OP_TYPE_WASMSSA_FUNC_ADDR:
        as[nas++] = attr_s_cstr(ctx, "target",
                                o->call_target ? o->call_target : "");
        break;
    case OP_TYPE_WASMSSA_CALL_INDIRECT:
        as[nas++] = attr_s_hex(ctx, F->arena, "sig_params",
                               o->sig_params, o->n_sig_params);
        as[nas++] = attr_s_hex(ctx, F->arena, "sig_results",
                               o->sig_results, o->n_sig_results);
        break;
    case OP_TYPE_WASMSSA_BR:
    case OP_TYPE_WASMSSA_BR_IF:
        as[nas++] = attr_i32(ctx, "depth", (int64_t)o->br_depth);
        break;
    case OP_TYPE_WASMSSA_CARRIER_SET:
    case OP_TYPE_WASMSSA_CARRIER_GET:
        as[nas++] = attr_i32(ctx, "carrier_id", (int64_t)o->carrier_id);
        break;
    default: break;
    }
    MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
    MLIR_OpHandle mop = make_op(ctx, o->type, as, nas, ops, n_ops, NULL, 0,
                                o->has_result ? o->valtype : 0, &res);
    MLIR_AppendBlockOp(ctx, F->body_block, mop);

    // Reserve next ssa_def_idx (regardless of has_result, matching the
    // historical add_op semantics where every op consumes a slot).
    int idx = (int)F->n_ssa;
    if (F->n_ssa == F->c_ssa) {
        F->c_ssa = F->c_ssa ? F->c_ssa * 2 : 16;
        F->ssa_vals = (MLIR_ValueHandle *)realloc(F->ssa_vals,
                                                  F->c_ssa * sizeof(MLIR_ValueHandle));
    }
    F->ssa_vals[F->n_ssa++] = o->has_result ? res : MLIR_INVALID_HANDLE;
    // The temporary wasmssa_op_t owns these malloc'd buffers; release them
    // now that the MLIR op has captured their contents (attribute strings
    // are copied / encoded into arena-backed storage by the helpers).
    free(o->operands);
    free(o->call_target);
    free(o->sig_params);
    free(o->sig_results);
    return idx;
}

// Convenience: const-i32, returns ssa_def_idx.
static int emit_const_i32(FnCtx *F, int32_t v) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_CONST;
    o.valtype = WT_I32;
    o.i_const = v;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
// Convenience: i32 add of two ssa-def operands.
static int emit_add_i32(FnCtx *F, int lhs, int rhs) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_ADD;
    o.valtype = WT_I32;
    o.n_operands = 2;
    o.operands = (int *)malloc(2 * sizeof(int));
    o.operands[0] = lhs;
    o.operands[1] = rhs;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
// Convenience: i32 sub.
static int emit_sub_i32(FnCtx *F, int lhs, int rhs) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_SUB;
    o.valtype = WT_I32;
    o.n_operands = 2;
    o.operands = (int *)malloc(2 * sizeof(int));
    o.operands[0] = lhs;
    o.operands[1] = rhs;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
// Convenience: global_get of an i32 global.
static int emit_global_get(FnCtx *F, uint32_t gidx) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_GLOBAL_GET;
    o.valtype = WT_I32;
    o.global_idx = gidx;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
static void emit_global_set(FnCtx *F, uint32_t gidx, int valv) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_GLOBAL_SET;
    o.valtype = WT_I32;
    o.global_idx = gidx;
    o.n_operands = 1;
    o.operands = (int *)malloc(sizeof(int));
    o.operands[0] = valv;
    (void)commit_op(F, &o);
}

// Convenience: i32 mul.
static int emit_mul_i32(FnCtx *F, int lhs, int rhs) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_BINOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0x6c;  // i32.mul
    o.n_operands = 2;
    o.operands = (int *)malloc(2 * sizeof(int));
    o.operands[0] = lhs;
    o.operands[1] = rhs;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
// Convenience: i32.wrap_i64 (0xa7) — narrows an i64 SSA value to i32.
static int emit_wrap_i64_to_i32(FnCtx *F, int v) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_UNOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0xa7;
    o.n_operands = 1;
    o.operands = (int *)malloc(sizeof(int));
    o.operands[0] = v;
    o.has_result = true;
    int idx = commit_op(F, &o);
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

// Look up a name as a function symbol in the module under construction.
// Returns true if `m->func_names` already contains that name.
// Used to disambiguate addressof @sym between data globals and functions.
// Note: pass 1 (imports) and pass 2 (defs) record func names in source
// order. The function being lowered itself is already added before its
// body is walked; recursive self-address-taking works.
static bool is_function_symbol(const ModCtx *m, const char *nm, size_t nlen) {
    for (size_t i = 0; i < m->n_func_names; i++) {
        if (m->func_names[i] &&
            strlen(m->func_names[i]) == nlen &&
            memcmp(m->func_names[i], nm, nlen) == 0) return true;
    }
    return false;
}

// =============================================================================
// Per-op lowering.
// =============================================================================
static bool lower_op(FnCtx *F, MLIR_OpHandle op);
static bool va_call_layout(FnCtx *F, MLIR_OpHandle op, uint32_t *total_size,
                           size_t *n_fixed, uint32_t *out_offsets);
static bool lower_block(FnCtx *F, MLIR_BlockHandle blk);

// Append a structured-CF marker / br / select op.
static int emit_marker(FnCtx *F, MLIR_OpType t, uint8_t blocktype,
                       int cond_operand, uint32_t depth) {
    wasmssa_op_t o = {0};
    o.type = t;
    o.valtype = blocktype;
    o.br_depth = depth;
    if (cond_operand >= 0) {
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = cond_operand;
    }
    return commit_op(F, &o);
}
static int emit_eqz(FnCtx *F, int v) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_EQZ;
    o.valtype = WT_I32;
    o.n_operands = 1;
    o.operands = (int *)malloc(sizeof(int));
    o.operands[0] = v;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
static int emit_carrier_get(FnCtx *F, uint32_t cid, uint8_t vt) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_CARRIER_GET;
    o.valtype = vt;
    o.carrier_id = cid;
    o.has_result = true;
    int idx = commit_op(F, &o);
    return idx;
}
static void emit_carrier_set(FnCtx *F, uint32_t cid, uint8_t vt, int valv) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_CARRIER_SET;
    o.valtype = vt;
    o.carrier_id = cid;
    o.n_operands = 1;
    o.operands = (int *)malloc(sizeof(int));
    o.operands[0] = valv;
    (void)commit_op(F, &o);
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
// General shape (with iter args / results):
//   %res:R = scf.while (%a:T = %init:T) : (T...) -> (R...) {
//     ^before(%a:T...): ...; scf.condition(%cond) %r:R...
//   } do {
//     ^after(%b:R...): ...; scf.yield %y:T...
//   }
//
// Lowered to:
//   (init: store %init[i] -> iter_c[i])
//   block $exit
//     loop $continue
//       (bind before-args[i] = CARRIER_GET iter_c[i])
//       <before-body>
//       (scf.condition: store %r[i] -> tr_c[i]; br_if $exit on !cond)
//       (bind after-args[i] = CARRIER_GET tr_c[i])
//       <after-body>
//       (scf.yield: store %y[i] -> iter_c[i]; br $continue)
//     end
//   end
//   (bind scf.while results[i] = CARRIER_GET tr_c[i])
static bool lower_scf_while(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumRegions(op) < 2) return false;
    MLIR_RegionHandle before_r = MLIR_GetOpRegion(op, 0);
    MLIR_RegionHandle after_r  = MLIR_GetOpRegion(op, 1);
    if (MLIR_GetRegionNumBlocks(before_r) != 1) return false;
    if (MLIR_GetRegionNumBlocks(after_r)  != 1) return false;
    MLIR_BlockHandle before_b = MLIR_GetRegionBlock(before_r, 0);
    MLIR_BlockHandle after_b  = MLIR_GetRegionBlock(after_r, 0);

    size_t n_iter = MLIR_GetOpNumOperands(op);   // = #before-args = #after-yield
    size_t n_res  = MLIR_GetOpNumResults(op);    // = #after-args  = #condition payload

    if (n_iter != MLIR_GetBlockNumArgs(before_b)) return false;
    if (n_res  != MLIR_GetBlockNumArgs(after_b))  return false;

    // Allocate iter and transit carriers.
    uint32_t *iter_c = NULL; uint8_t *iter_v = NULL;
    uint32_t *tr_c   = NULL; uint8_t *tr_v   = NULL;
    if (n_iter) {
        iter_c = (uint32_t *)malloc(n_iter * sizeof(uint32_t));
        iter_v = (uint8_t *)malloc(n_iter);
        for (size_t i = 0; i < n_iter; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpOperand(op, i)));
            if (vt == 0) goto fail;
            iter_v[i] = vt;
            iter_c[i] = alloc_carrier(F, vt);
            int v;
            if (!vmap_get(F, MLIR_GetOpOperand(op, i), &v)) goto fail;
            emit_carrier_set(F, iter_c[i], vt, v);
        }
    }
    if (n_res) {
        tr_c = (uint32_t *)malloc(n_res * sizeof(uint32_t));
        tr_v = (uint8_t *)malloc(n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpResult(op, i)));
            if (vt == 0) goto fail;
            tr_v[i] = vt;
            tr_c[i] = alloc_carrier(F, vt);
        }
    }

    emit_marker(F, OP_TYPE_WASMSSA_BLOCK_BEGIN, 0x40, -1, 0);  // $exit
    emit_marker(F, OP_TYPE_WASMSSA_LOOP_BEGIN,  0x40, -1, 0);  // $continue

    // Bind before-block args from iter carriers.
    for (size_t i = 0; i < n_iter; i++) {
        int g = emit_carrier_get(F, iter_c[i], iter_v[i]);
        vmap_set(F, MLIR_GetBlockArg(before_b, i), g);
    }

    // Walk before-block ops. Terminator is scf.condition.
    size_t nb = MLIR_GetBlockNumOps(before_b);
    for (size_t i = 0; i < nb; i++) {
        MLIR_OpHandle bop = MLIR_GetBlockOp(before_b, i);
        string n = MLIR_GetOpName(bop);
        if (name_eq(n, "scf.condition")) {
            if (MLIR_GetOpNumOperands(bop) < 1) goto fail;
            // Store condition payloads into transit carriers.
            if (MLIR_GetOpNumOperands(bop) - 1 != n_res) goto fail;
            for (size_t k = 0; k < n_res; k++) {
                int v;
                if (!vmap_get(F, MLIR_GetOpOperand(bop, k + 1), &v)) goto fail;
                emit_carrier_set(F, tr_c[k], tr_v[k], v);
            }
            int cidx;
            if (!vmap_get(F, MLIR_GetOpOperand(bop, 0), &cidx)) goto fail;
            int z = emit_eqz(F, cidx);
            // If !cond, branch to $exit (depth 1 above the loop).
            emit_marker(F, OP_TYPE_WASMSSA_BR_IF, 0, z, 1);
        } else {
            if (!lower_op(F, bop)) goto fail;
        }
    }

    // Bind after-block args from transit carriers.
    for (size_t i = 0; i < n_res; i++) {
        int g = emit_carrier_get(F, tr_c[i], tr_v[i]);
        vmap_set(F, MLIR_GetBlockArg(after_b, i), g);
    }

    // Walk after-block ops. Terminator scf.yield -> store iter + br $continue.
    size_t na = MLIR_GetBlockNumOps(after_b);
    for (size_t i = 0; i < na; i++) {
        MLIR_OpHandle aop = MLIR_GetBlockOp(after_b, i);
        string n = MLIR_GetOpName(aop);
        if (name_eq(n, "scf.yield")) {
            if (MLIR_GetOpNumOperands(aop) != n_iter) goto fail;
            for (size_t k = 0; k < n_iter; k++) {
                int v;
                if (!vmap_get(F, MLIR_GetOpOperand(aop, k), &v)) goto fail;
                emit_carrier_set(F, iter_c[k], iter_v[k], v);
            }
            emit_marker(F, OP_TYPE_WASMSSA_BR, 0, -1, 0);
        } else {
            if (!lower_op(F, aop)) goto fail;
        }
    }
    emit_marker(F, OP_TYPE_WASMSSA_END, 0, -1, 0);  // close loop
    emit_marker(F, OP_TYPE_WASMSSA_END, 0, -1, 0);  // close block

    // Bind scf.while results from transit carriers.
    for (size_t i = 0; i < n_res; i++) {
        int g = emit_carrier_get(F, tr_c[i], tr_v[i]);
        vmap_set(F, MLIR_GetOpResult(op, i), g);
    }

    free(iter_c); free(iter_v); free(tr_c); free(tr_v);
    return true;
fail:
    free(iter_c); free(iter_v); free(tr_c); free(tr_v);
    return false;
}

// ---- scf.index_switch ------------------------------------------------------
// Lowered as a chain of (cond == case_i) ifs; default goes in the innermost
// else. All cases / default share one set of result carriers.
static bool lower_scf_index_switch(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(op, 0);
    int cond_idx;
    if (!vmap_get(F, cond_v, &cond_idx)) return false;
    if (wasm_vt(F->ctx, MLIR_GetValueType(cond_v)) != WT_I32) return false;

    size_t n_regions = MLIR_GetOpNumRegions(op);
    if (n_regions < 1) return false;
    size_t n_cases = n_regions - 1;  // region 0 = default

    // Parse "cases" attribute: prints as "array<i64: 1, 2, 3>".
    MLIR_AttributeHandle ca = find_attr(op, "cases");
    if (ca == MLIR_INVALID_HANDLE) return false;
    string cs = MLIR_GetAttributeAsString(F->ctx, ca);
    int64_t *case_vals = NULL;
    size_t   n_parsed = 0;
    if (n_cases > 0) {
        case_vals = (int64_t *)malloc(n_cases * sizeof(int64_t));
        size_t p = 0;
        while (p < cs.size && cs.str[p] != ':') p++;
        if (p < cs.size) p++;
        while (p < cs.size && n_parsed < n_cases) {
            while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
            if (p >= cs.size || cs.str[p] == '>') break;
            int64_t sign = 1;
            if (cs.str[p] == '-') { sign = -1; p++; }
            int64_t v = 0;
            while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9') {
                v = v * 10 + (cs.str[p] - '0');
                p++;
            }
            case_vals[n_parsed++] = sign * v;
        }
        if (n_parsed != n_cases) { free(case_vals); return false; }
    }

    // Allocate result carriers.
    size_t n_results = MLIR_GetOpNumResults(op);
    uint32_t *cids = NULL; uint8_t *cvts = NULL;
    if (n_results) {
        cids = (uint32_t *)malloc(n_results * sizeof(uint32_t));
        cvts = (uint8_t *)malloc(n_results);
        for (size_t i = 0; i < n_results; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(MLIR_GetOpResult(op, i)));
            if (vt == 0) { free(cids); free(cvts); free(case_vals); return false; }
            cvts[i] = vt;
            cids[i] = alloc_carrier(F, vt);
        }
    }

    YieldFrame yf = {0};
    if (n_results) {
        yf.carrier_ids = (uint32_t *)malloc(n_results * sizeof(uint32_t));
        yf.carrier_vts = (uint8_t *)malloc(n_results);
        memcpy(yf.carrier_ids, cids, n_results * sizeof(uint32_t));
        memcpy(yf.carrier_vts, cvts, n_results);
        yf.n = (int)n_results;
    }
    push_yield(F, yf);

    // Emit nested IF/ELSE chain.
    for (size_t i = 0; i < n_cases; i++) {
        int kc = emit_const_i32(F, (int32_t)case_vals[i]);
        // i32.eq = 0x46
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;
        o.wasm_opcode = 0x46;
        o.n_operands = 2;
        o.operands = (int *)malloc(2 * sizeof(int));
        o.operands[0] = cond_idx;
        o.operands[1] = kc;
        o.has_result = true;
        int eq_idx = commit_op(F, &o);
        emit_marker(F, OP_TYPE_WASMSSA_IF_BEGIN, 0x40, eq_idx, 0);
        // Lower case-region body (region i+1).
        MLIR_RegionHandle cr = MLIR_GetOpRegion(op, i + 1);
        if (MLIR_GetRegionNumBlocks(cr) != 1) goto fail;
        if (!lower_block(F, MLIR_GetRegionBlock(cr, 0))) goto fail;
        emit_marker(F, OP_TYPE_WASMSSA_IF_ELSE, 0, -1, 0);
    }
    // Innermost else: default region.
    MLIR_RegionHandle dr = MLIR_GetOpRegion(op, 0);
    if (MLIR_GetRegionNumBlocks(dr) != 1) goto fail;
    if (!lower_block(F, MLIR_GetRegionBlock(dr, 0))) goto fail;
    for (size_t i = 0; i < n_cases; i++) {
        emit_marker(F, OP_TYPE_WASMSSA_END, 0, -1, 0);
    }
    pop_yield(F);

    for (size_t i = 0; i < n_results; i++) {
        int g = emit_carrier_get(F, cids[i], cvts[i]);
        vmap_set(F, MLIR_GetOpResult(op, i), g);
    }
    free(cids); free(cvts); free(case_vals);
    return true;
fail:
    free(cids); free(cvts); free(case_vals);
    return false;
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

    // ---- scf.if / scf.while / scf.index_switch ---------------------------
    if (name_eq(name, "scf.if"))    return lower_scf_if(F, op);
    if (name_eq(name, "scf.while")) return lower_scf_while(F, op);
    if (name_eq(name, "scf.index_switch")) return lower_scf_index_switch(F, op);

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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_SELECT;
        o.valtype = vt;
        o.n_operands = 3;
        o.operands = (int *)malloc(3 * sizeof(int));
        // wasm select pops cond,b,a so we order operands as a,b,cond.
        o.operands[0] = ai; o.operands[1] = bi; o.operands[2] = ci;
        o.has_result = true;
        int idx = commit_op(F, &o);
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

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        if (ak == MLIR_ATTR_KIND_INTEGER) {
            o.i_const = MLIR_GetAttributeInteger(va);
        } else if (ak == MLIR_ATTR_KIND_BOOL) {
            o.i_const = MLIR_GetAttributeBool(va) ? 1 : 0;
        } else if (ak == MLIR_ATTR_KIND_FLOAT) {
            double dv = MLIR_GetAttributeFloat(va);
            // Pack the bit pattern through an integer of matching width;
            // stage 3 emits f32.const / f64.const using the same bytes.
            if (vt == WT_F32) {
                float fv = (float)dv;
                uint32_t bits;
                memcpy(&bits, &fv, 4);
                o.i_const = (int64_t)(uint64_t)bits;
            } else if (vt == WT_F64) {
                uint64_t bits;
                memcpy(&bits, &dv, 8);
                o.i_const = (int64_t)bits;
            } else {
                return false;
            }
        } else {
            return false;
        }
        o.has_result = true;
        int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        o.i_const = 0;
        o.has_result = true;
        int idx = commit_op(F, &o);
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

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_STORE;
        o.valtype = vt;
        o.mem_size_bytes = sz;
        o.memory_align_log2 = align_log2;
        o.memory_offset = 0;
        o.n_operands = 2;
        o.operands = (int *)malloc(2 * sizeof(int));
        o.operands[0] = pa;   // address (stage 2 emits local.get for these in order)
        o.operands[1] = va;   // value
        (void)commit_op(F, &o);
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

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_LOAD;
        o.valtype = vt;
        o.mem_size_bytes = sz;
        o.memory_align_log2 = align_log2;
        o.memory_offset = 0;
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = pa;
        o.has_result = true;
        int idx = commit_op(F, &o);
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
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_UNOP;
            o.valtype = WT_I32;
            o.wasm_opcode = (in_w == 8) ? 0xc0 : 0xc1;
            o.n_operands = 1;
            o.operands = (int *)malloc(sizeof(int));
            o.operands[0] = cur_idx;
            o.has_result = true;
            int idx = commit_op(F, &o);
            cur_idx = idx; cur_vt = WT_I32;
        }
        // Step 2: if widening into i64, use i64.extend_i32_s (0xac).
        if (out_w == 64) {
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_EXTEND_I32_S;
            o.valtype = WT_I64;
            o.n_operands = 1;
            o.operands = (int *)malloc(sizeof(int));
            o.operands[0] = cur_idx;
            o.has_result = true;
            int idx = commit_op(F, &o);
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
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_UNOP;
            o.valtype = WT_I64;
            o.wasm_opcode = 0xad;
            o.n_operands = 1;
            o.operands = (int *)malloc(sizeof(int));
            o.operands[0] = sa;
            o.has_result = true;
            int idx = commit_op(F, &o);
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
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_BINOP;
            o.valtype = vt;
            o.wasm_opcode = opc;
            o.n_operands = 2;
            o.operands = (int *)malloc(2 * sizeof(int));
            o.operands[0] = ai; o.operands[1] = bi;
            o.has_result = true;
            int idx = commit_op(F, &o);
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
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_BINOP;
            o.valtype = vt;
            o.wasm_opcode = opc;
            o.n_operands = 2;
            o.operands = (int *)malloc(2 * sizeof(int));
            o.operands[0] = ai; o.operands[1] = bi;
            o.has_result = true;
            int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = (int *)malloc(2 * sizeof(int));
        o.operands[0] = ai; o.operands[1] = bi;
        o.has_result = true;
        int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = sa;
        o.has_result = true;
        int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = sa;
        o.has_result = true;
        int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;  // result type
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = (int *)malloc(2 * sizeof(int));
        o.operands[0] = ai; o.operands[1] = bi;
        o.has_result = true;
        int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = sa;
        o.has_result = true;
        int idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- builtin.unrealized_conversion_cast --------------------------------
    // Treat as identity when wasm valtypes match (e.g. ptr<->i32, index<->i32).
    // Otherwise emit a trunc/extend (i64<->i32 unsigned).
    if (name_eq(name, "builtin.unrealized_conversion_cast")) {
        if (MLIR_GetOpNumOperands(op) != 1 || MLIR_GetOpNumResults(op) != 1)
            return false;
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt == 0 || out_vt == 0) return false;
        int sa;
        if (!vmap_get(F, s, &sa)) return false;
        if (in_vt == out_vt) { vmap_set(F, r, sa); return true; }
        uint8_t opc;
        if (in_vt == WT_I64 && out_vt == WT_I32) opc = 0xa7;       // i32.wrap_i64
        else if (in_vt == WT_I32 && out_vt == WT_I64) opc = 0xad;  // i64.extend_i32_u
        else return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = sa;
        o.has_result = true;
        int idx = commit_op(F, &o);
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

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_RETURN;
        if (no == 1) {
            o.n_operands = 1;
            o.operands = (int *)malloc(sizeof(int));
            o.operands[0] = retv;
        }
        (void)commit_op(F, &o);
        return true;
    }

    // ---- llvm.call (direct or indirect) ------------------------------------
    if (name_eq(name, "llvm.call")) {
        MLIR_AttributeHandle callee = MLIR_GetOpAttributeByName(op, "callee");
        MLIR_AttributeHandle var_callee_type = MLIR_GetOpAttributeByName(op, "var_callee_type");

        // Indirect: no callee attr, operand[0] is the function pointer
        // (a !llvm.ptr SSA value), and var_callee_type carries the signature.
        if (callee == MLIR_INVALID_HANDLE) {
            // Indirect call: operand[0] is function pointer (!llvm.ptr).
            // Build the signature from operand types and result types,
            // since the upstream wasm pipeline doesn't carry var_callee_type
            // through after lowering.
            size_t no = MLIR_GetOpNumOperands(op);
            if (no < 1) return false;
            size_t snp = no - 1;
            uint8_t *sp = (uint8_t *)malloc(snp ? snp : 1);
            for (size_t i = 0; i < snp; i++) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpOperand(op, i + 1)));
                if (v == 0) { free(sp); return false; }
                sp[i] = v;
            }
            size_t nr = MLIR_GetOpNumResults(op);
            if (nr > 1) { free(sp); return false; }
            uint8_t *sr = (uint8_t *)malloc(1);
            size_t snr = 0;
            if (nr == 1) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpResult(op, 0)));
                if (v == 0) { free(sp); free(sr); return false; }
                sr[0] = v;
                snr = 1;
            }
            int *opnds = (int *)malloc(no * sizeof(int));
            // wasm call_indirect pops args first then table-index, so order
            // them in operand[] as [args..., funcptr].
            for (size_t i = 0; i < snp; i++) {
                if (!vmap_get(F, MLIR_GetOpOperand(op, i + 1), &opnds[i])) {
                    free(opnds); free(sp); free(sr); return false;
                }
            }
            if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &opnds[snp])) {
                free(opnds); free(sp); free(sr); return false;
            }
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_CALL_INDIRECT;
            o.n_operands = (int)no;
            o.operands = opnds;
            o.sig_params = sp; o.n_sig_params = snp;
            o.sig_results = sr; o.n_sig_results = snr;
            if (snr == 1) {
                o.valtype = sr[0];
                o.has_result = true;
            }
            int idx = commit_op(F, &o);
            if (snr == 1) {
                vmap_set(F, MLIR_GetOpResult(op, 0), idx);
            }
            return true;
        }

        if (var_callee_type != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(var_callee_type);
            if (MLIR_GetTypeFunctionIsVarArg(fty)) {
                // Variadic direct call: pack variadic args into the
                // function's variadic buffer (allocated in the shadow
                // stack frame) and pass its address as a hidden trailing
                // i32 arg in place of the `...` operands.
                size_t nfixed = MLIR_GetTypeFunctionNumInputs(fty);
                size_t no = MLIR_GetOpNumOperands(op);
                size_t nvar = no - nfixed;
                uint32_t total = 0; size_t nf2 = 0;
                uint32_t *offs = (uint32_t *)malloc((nvar ? nvar : 1) * sizeof(uint32_t));
                if (!va_call_layout(F, op, &total, &nf2, offs)) {
                    free(offs);
                    return false;
                }
                if (F->sp_def < 0) { free(offs); return false; }
                // buf_addr = sp_def + va_buf_offset
                int buf_addr;
                if (F->va_buf_offset == 0) {
                    buf_addr = F->sp_def;
                } else {
                    int koff = emit_const_i32(F, (int32_t)F->va_buf_offset);
                    buf_addr = emit_add_i32(F, F->sp_def, koff);
                }
                // Emit a store for each variadic arg, then the call.
                for (size_t i = 0; i < nvar; i++) {
                    MLIR_ValueHandle av = MLIR_GetOpOperand(op, nfixed + i);
                    MLIR_TypeHandle aty = MLIR_GetValueType(av);
                    uint8_t vt = wasm_vt(F->ctx, aty);
                    int va;
                    if (!vmap_get(F, av, &va)) { free(offs); return false; }
                    uint8_t st_vt = vt;
                    unsigned sz, align_log2;
                    if (vt == WT_I32)      { sz = 4; align_log2 = 2; }
                    else if (vt == WT_I64) { sz = 8; align_log2 = 3; }
                    else if (vt == WT_F32) {
                        // Promote f32 -> f64.
                        wasmssa_op_t po = {0};
                        po.type = OP_TYPE_WASMSSA_UNOP;
                        po.valtype = WT_F64;
                        po.wasm_opcode = 0xbb; // f64.promote_f32
                        po.n_operands = 1;
                        po.operands = (int *)malloc(sizeof(int));
                        po.operands[0] = va;
                        po.has_result = true;
                        int pidx = commit_op(F, &po);
                        va = pidx;
                        st_vt = WT_F64;
                        sz = 8; align_log2 = 3;
                    }
                    else if (vt == WT_F64) { sz = 8; align_log2 = 3; }
                    else { free(offs); return false; }
                    wasmssa_op_t o2 = {0};
                    o2.type = OP_TYPE_WASMSSA_STORE;
                    o2.valtype = st_vt;
                    o2.mem_size_bytes = sz;
                    o2.memory_align_log2 = align_log2;
                    o2.memory_offset = offs[i];
                    o2.n_operands = 2;
                    o2.operands = (int *)malloc(2 * sizeof(int));
                    o2.operands[0] = buf_addr;
                    o2.operands[1] = va;
                    (void)commit_op(F, &o2);
                }
                free(offs);

                // Now emit the call with fixed args + buf_addr.
                string nm = MLIR_GetAttributeAsString(F->ctx, callee);
                const char *cname = nm.str; size_t cn = nm.size;
                if (cn > 0 && cname[0] == '@') { cname++; cn--; }
                char *cstr = xstrdupn(cname, cn);
                size_t nout = nfixed + 1;
                int *opnds = (int *)malloc(nout * sizeof(int));
                for (size_t i = 0; i < nfixed; i++) {
                    if (!vmap_get(F, MLIR_GetOpOperand(op, i), &opnds[i])) {
                        free(opnds); free(cstr); return false;
                    }
                }
                opnds[nfixed] = buf_addr;

                wasmssa_op_t oc = {0};
                oc.type = OP_TYPE_WASMSSA_CALL;
                oc.call_target = cstr;
                oc.n_operands = (int)nout;
                oc.operands = opnds;
                size_t nr = MLIR_GetOpNumResults(op);
                MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
                if (nr == 1) {
                    r = MLIR_GetOpResult(op, 0);
                    uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
                    if (vt == 0) return false;
                    oc.valtype = vt;
                    oc.has_result = true;
                } else if (nr != 0) {
                    return false;
                }
                int idx = commit_op(F, &oc);
                if (nr == 1) vmap_set(F, r, idx);
                return true;
            }
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

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CALL;
        o.call_target = cstr;
        o.n_operands = (int)no;
        o.operands = opnds;

        size_t nr = MLIR_GetOpNumResults(op);
        MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
        if (nr == 1) {
            r = MLIR_GetOpResult(op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
            if (vt == 0) return false;
            o.valtype = vt;
            o.has_result = true;
        } else if (nr != 0) {
            return false;
        }
        int idx = commit_op(F, &o);
        if (nr == 1) vmap_set(F, r, idx);
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
                // `!llvm.array<0 x T>` (zero-length array) has size 0; it is
                // used as the element type for addressof-based GEP into a
                // global whose runtime size we don't model in the type. Any
                // contribution from this index multiplied by 0 is 0, so just
                // skip it.
                if (esz == 0) {
                    // no-op
                } else if (is_dyn) {
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

        // If this name refers to a function (not a data global), emit a
        // FUNC_ADDR (lowered as i32.const + R_WASM_TABLE_INDEX_SLEB) so the
        // wasm linker places it in __indirect_function_table.
        if (is_function_symbol(F->mod, ts.str, ts.size)) {
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_FUNC_ADDR;
            o.valtype = WT_I32;
            o.call_target = xstrdup_str(ts);
            o.has_result = true;
            int idx = commit_op(F, &o);
            vmap_set(F, MLIR_GetOpResult(op, 0), idx);
            return true;
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_ADDRESSOF;
        o.valtype = WT_I32;
        o.call_target = xstrdup_str(ts);  // reuse field for symbol name
        o.has_result = true;
        int idx = commit_op(F, &o);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        o.i_const = 0;
        o.has_result = true;
        int idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.vastart / llvm.intr.vaend -------------------------------
    if (name_eq(name, "llvm.intr.vastart")) {
        // Store the function's hidden va_list i32 param into *%ap.
        if (F->va_list_param_idx < 0) return false;
        if (MLIR_GetOpNumOperands(op) != 1) return false;
        int pa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &pa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_STORE;
        o.valtype = WT_I32;
        o.mem_size_bytes = 4;
        o.memory_align_log2 = 2;
        o.memory_offset = 0;
        o.n_operands = 2;
        o.operands = (int *)malloc(2 * sizeof(int));
        o.operands[0] = pa;
        o.operands[1] = F->va_list_param_idx;
        (void)commit_op(F, &o);
        return true;
    }
    if (name_eq(name, "llvm.intr.vaend")) {
        // No-op under the wasm32 ABI.
        return true;
    }
    if (name_eq(name, "llvm.intr.vacopy")) {
        // Copy 4 bytes (a char*) from src to dst.
        if (MLIR_GetOpNumOperands(op) != 2) return false;
        int da, sa;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 0), &da)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(op, 1), &sa)) return false;
        wasmssa_op_t ol = {0};
        ol.type = OP_TYPE_WASMSSA_LOAD;
        ol.valtype = WT_I32;
        ol.mem_size_bytes = 4;
        ol.memory_align_log2 = 2;
        ol.memory_offset = 0;
        ol.n_operands = 1;
        ol.operands = (int *)malloc(sizeof(int));
        ol.operands[0] = sa;
        ol.has_result = true;
        int lidx = commit_op(F, &ol);
        wasmssa_op_t os = {0};
        os.type = OP_TYPE_WASMSSA_STORE;
        os.valtype = WT_I32;
        os.mem_size_bytes = 4;
        os.memory_align_log2 = 2;
        os.memory_offset = 0;
        os.n_operands = 2;
        os.operands = (int *)malloc(2 * sizeof(int));
        os.operands[0] = da;
        os.operands[1] = lidx;
        (void)commit_op(F, &os);
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
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        o.operands = (int *)malloc(sizeof(int));
        o.operands[0] = sa;
        o.has_result = true;
        int idx = commit_op(F, &o);
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

// For a variadic llvm.call, compute the buffer size and per-arg offsets
// needed to pack the variadic args under the wasm32 clang ABI. Returns
// false if the call is not variadic or the variadic args have an
// unsupported wasm value type. On success, *total_size receives the total
// buffer size (in bytes), *n_fixed receives the number of fixed
// (non-variadic) operands. If out_offsets is non-NULL, it must point to a
// caller-provided array large enough for (n_variadic) entries and will be
// filled with the per-variadic-arg buffer offsets (in operand order).
static bool va_call_layout(FnCtx *F, MLIR_OpHandle op, uint32_t *total_size,
                           size_t *n_fixed, uint32_t *out_offsets) {
    MLIR_AttributeHandle vct = MLIR_GetOpAttributeByName(op, "var_callee_type");
    if (vct == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(vct);
    if (!MLIR_GetTypeFunctionIsVarArg(fty)) return false;
    size_t nf = MLIR_GetTypeFunctionNumInputs(fty);
    size_t no = MLIR_GetOpNumOperands(op);
    if (no < nf) return false;
    uint32_t cur = 0;
    for (size_t i = nf; i < no; i++) {
        MLIR_TypeHandle aty = MLIR_GetValueType(MLIR_GetOpOperand(op, i));
        uint8_t vt = wasm_vt(F->ctx, aty);
        unsigned sz = type_size_bytes(F->ctx, aty);
        unsigned al;
        if (vt == WT_I32) {
            // i8/i16 are widened to i32 in C variadic promotion; the
            // frontend currently emits all small ints as i32 already.
            sz = 4; al = 4;
        } else if (vt == WT_I64) {
            sz = 8; al = 8;
        } else if (vt == WT_F32) {
            // C variadic promotion: f32 -> f64. Pack as 8 bytes f64.
            sz = 8; al = 8;
        } else if (vt == WT_F64) {
            sz = 8; al = 8;
        } else {
            return false;
        }
        cur = align_up(cur, al);
        if (out_offsets) out_offsets[i - nf] = cur;
        cur += sz;
        (void)sz;
    }
    *total_size = cur;
    *n_fixed = nf;
    return true;
}

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
    if (name_eq(n, "llvm.call")) {
        uint32_t sz = 0; size_t nfixed = 0;
        if (va_call_layout(F, op, &sz, &nfixed, NULL)) {
            if (sz > F->va_buf_size) F->va_buf_size = sz;
        }
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
    if (F->va_buf_size > 0) {
        F->frame_size = align_up(F->frame_size, 8);
        F->va_buf_offset = F->frame_size;
        F->frame_size += F->va_buf_size;
    }
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

// Lower a defined `llvm.func` into a `wasmssa.func` op appended to the
// module body. The function's signature attrs are passed in (param_types
// includes the synthetic trailing i32 for vararg functions); the special
// "__original_main" rename + exported flag are picked by the caller.
static bool lower_function(MLIR_Context *ctx, Arena *arena, ModCtx *mod,
                           MLIR_BlockHandle mod_body,
                           const char *fn_name, bool exported,
                           MLIR_OpHandle fn,
                           const uint8_t *param_types, size_t n_params,
                           const uint8_t *result_types, size_t n_results) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx;
    F.arena = arena;
    F.mod = mod;
    F.n_params = n_params;
    F.sp_def = -1;
    F.va_list_param_idx = -1;

    // If this function is variadic, sig_for_func has appended a synthetic
    // i32 va_list param at index n_params-1. The MLIR entry block only has
    // the original (non-hidden) parameters as block args.
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    bool is_vararg = false;
    if (ftya != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
        is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
    }
    size_t orig_np = is_vararg ? n_params - 1 : n_params;

    MLIR_RegionHandle body = MLIR_GetOpRegion(fn, 0);
    MLIR_BlockHandle  entry = MLIR_GetRegionBlock(body, 0);

    // Build the wasmssa.func body block and seed ssa_vals[0..n_params)
    // with fresh block-arg values (one per declared wasm param, including
    // the hidden va_list i32 for varargs).
    F.body_block = MLIR_CreateBlock(ctx);
    F.c_ssa = n_params ? n_params : 16;
    F.ssa_vals = (MLIR_ValueHandle *)calloc(F.c_ssa, sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < n_params; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, param_types[i]);
        MLIR_ValueHandle bv = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                      MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, F.body_block, bv);
        F.ssa_vals[i] = bv;
    }
    F.n_ssa = n_params;

    // Bind parameter SSA values to ssa_def_idx 0..orig_np-1.
    for (size_t i = 0; i < orig_np; i++) {
        vmap_set(&F, MLIR_GetBlockArg(entry, i), (int)i);
    }
    if (is_vararg) F.va_list_param_idx = (int)(n_params - 1);

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

    {
        // Build the wasmssa.func wrapper op around F.body_block.
        MLIR_AttributeHandle attrs[8];
        size_t na = 0;
        attrs[na++] = attr_s_cstr(ctx, "sym_name", fn_name ? fn_name : "");
        attrs[na++] = attr_s_hex(ctx, arena, "param_types", param_types, n_params);
        attrs[na++] = attr_s_hex(ctx, arena, "result_types", result_types, n_results);
        attrs[na++] = attr_b(ctx, "exported", exported);
        attrs[na++] = attr_s_hex(ctx, arena, "carrier_types",
                                 F.carrier_vts, F.n_cvt);

        MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
        MLIR_AppendRegionBlock(ctx, region, F.body_block);
        MLIR_RegionHandle regs[1] = { region };
        MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_FUNC,
            op_type_to_string(OP_TYPE_WASMSSA_FUNC),
            attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
            MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, mod_body, op);
    }

    free(F.vmap); free(F.amap);
    free(F.carrier_vts); free(F.yield_stack);
    free(F.ssa_vals);
    return true;
fail:
    free(F.vmap); free(F.amap);
    free(F.carrier_vts); free(F.yield_stack);
    free(F.ssa_vals);
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
    bool is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
    size_t pn = ni + (is_vararg ? 1 : 0);
    uint8_t *p = (uint8_t *)malloc(pn ? pn : 1);
    for (size_t i = 0; i < ni; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionInput(fty, i));
        if (v == 0) { free(p); return false; }
        p[i] = v;
    }
    if (is_vararg) p[ni] = WT_I32;  // hidden va_list pointer
    uint8_t *r = (uint8_t *)malloc(no ? no : 1);
    for (size_t i = 0; i < no; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionResult(fty, i));
        if (v == 0) { free(p); free(r); return false; }
        r[i] = v;
    }
    *out_p = p; *out_np = pn;
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

// Helper: emit a wasmssa.import_global op into `body` with the given
// fields. `data` may be NULL when `size == 0`.
static void emit_import_global(MLIR_Context *ctx, Arena *arena,
                               MLIR_BlockHandle body,
                               const char *name, uint32_t size,
                               uint32_t align_pow, bool is_const,
                               const uint8_t *data, const char *relocs_str,
                               size_t relocs_len) {
    MLIR_AttributeHandle attrs[16];
    size_t na = 0;
    attrs[na++] = attr_s_cstr(ctx, "sym_name", name ? name : "");
    attrs[na++] = attr_i32(ctx, "size", size);
    attrs[na++] = attr_i32(ctx, "align_pow", align_pow);
    attrs[na++] = attr_b(ctx, "is_const", is_const);
    if (size) {
        attrs[na++] = attr_s(ctx, "init_data", (const char *)data, size);
    } else {
        attrs[na++] = attr_s(ctx, "init_data", "", 0);
    }
    if (relocs_str) {
        attrs[na++] = attr_s(ctx, "relocs", relocs_str, relocs_len);
    }
    MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSSA_IMPORT_GLOBAL,
                               attrs, na, NULL, 0, NULL, 0, 0, NULL);
    MLIR_AppendBlockOp(ctx, body, op);
    (void)arena;
}

// Lower one llvm.mlir.global op directly into a wasmssa.import_global op
// appended to `body`. Returns false on failure (no op is appended).
static bool lower_global(MLIR_Context *ctx, Arena *arena,
                         MLIR_BlockHandle body, MLIR_OpHandle op) {
    MLIR_AttributeHandle sa = find_attr(op, "sym_name");
    if (sa == MLIR_INVALID_HANDLE) return false;
    string sym = MLIR_GetAttributeString(sa);
    char *name = xstrdup_str(sym);

    uint32_t size = 0;
    uint32_t align_pow = 0;
    MLIR_AttributeHandle ga = find_attr(op, "global_type");
    if (ga != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle gty = MLIR_GetAttributeTypeValue(ga);
        size = type_size_bytes(ctx, gty);
        align_pow = align_pow_for_type(ctx, gty);
    }

    bool is_const = (find_attr(op, "constant") != MLIR_INVALID_HANDLE);

    uint8_t *data = NULL;
    bool data_owned = false;

    MLIR_AttributeHandle va = find_attr(op, "value");
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind ak = MLIR_GetAttributeKind(va);
        if (ak == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(va);
            size = (uint32_t)s.size;
            data = (uint8_t *)malloc(size ? size : 1);
            data_owned = true;
            memcpy(data, s.str, size);
            emit_import_global(ctx, arena, body, name, size, align_pow,
                               is_const, data, NULL, 0);
            free(name); if (data_owned) free(data);
            return true;
        }
        if (ak == MLIR_ATTR_KIND_INTEGER || ak == MLIR_ATTR_KIND_FLOAT) {
            if (size == 0) { free(name); return false; }
            data = (uint8_t *)calloc(1, size);
            data_owned = true;
            uint64_t bits = 0;
            if (ak == MLIR_ATTR_KIND_INTEGER) {
                bits = (uint64_t)MLIR_GetAttributeInteger(va);
            } else {
                double d = MLIR_GetAttributeFloat(va);
                if (size == 4) {
                    float f = (float)d; uint32_t b32;
                    memcpy(&b32, &f, 4); bits = b32;
                } else if (size == 8) {
                    memcpy(&bits, &d, 8);
                } else { free(name); free(data); return false; }
            }
            for (uint32_t i = 0; i < size && i < 8; i++)
                data[i] = (uint8_t)(bits >> (8 * i));
            emit_import_global(ctx, arena, body, name, size, align_pow,
                               is_const, data, NULL, 0);
            free(name); free(data);
            return true;
        }
        // Unknown attr kind -- fall through to zero-init.
    }

    // Region-init globals: walk for an llvm.mlir.addressof + llvm.return.
    // The only pattern produced by tinyc is:
    //   %0 = llvm.mlir.addressof @other : !llvm.ptr
    //   llvm.return %0 : !llvm.ptr
    // which becomes a 4-byte pointer slot with an R_WASM_MEMORY_ADDR reloc.
    if (MLIR_GetOpNumRegions(op) > 0) {
        MLIR_RegionHandle rgn = MLIR_GetOpRegion(op, 0);
        if (MLIR_GetRegionNumBlocks(rgn) > 0) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(rgn, 0);
            size_t nb = MLIR_GetBlockNumOps(blk);
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
                    if (ta == MLIR_INVALID_HANDLE) {
                        free(pending_target); free(name); return false;
                    }
                    string ts = MLIR_GetAttributeString(ta);
                    if (ts.size && ts.str[0] == '@') { ts.str++; ts.size--; }
                    free(pending_target);
                    pending_target = xstrdup_str(ts);
                } else if (name_eq(bn, "llvm.return")) {
                    if (!pending_target) break;
                    size = 4;
                    if (align_pow == 0) align_pow = 2;
                    data = (uint8_t *)calloc(1, size);
                    data_owned = true;
                    // Build the relocs string inline: "<off>:<target>:<addend>".
                    size_t cap = strlen(pending_target) + 32;
                    char *buf = (char *)arena_alloc(arena, cap);
                    int n = snprintf(buf, cap, "%u:%s:%d", 0u, pending_target, 0);
                    size_t off = (n > 0) ? (size_t)n : 0;
                    emit_import_global(ctx, arena, body, name, size, align_pow,
                                       is_const, data, buf, off);
                    free(pending_target); free(name); free(data);
                    return true;
                }
            }
            free(pending_target);
        }
    }

    // No initializer found (e.g. uninitialized scalar global). Zero-init.
    if (size == 0) { free(name); return false; }
    data = (uint8_t *)calloc(1, size);
    emit_import_global(ctx, arena, body, name, size, align_pow,
                       is_const, data, NULL, 0);
    free(name); free(data);
    return true;
}

// =============================================================================
// MLIR emit helpers for `wasmssa.*` ops. Used both by `commit_op` (per
// in-function op) and by the module walker (for import_func / globals).
// =============================================================================
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
        case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_s_cstr(MLIR_Context *ctx, const char *name,
                                        const char *v) {
    return attr_s(ctx, name, v, v ? strlen(v) : 0);
}
static char *hex_encode(Arena *arena, const uint8_t *p, size_t n) {
    char *out = (char *)arena_alloc(arena, n ? n * 2 : 1);
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i*2 + 0] = d[(p[i] >> 4) & 0xf];
        out[i*2 + 1] = d[p[i] & 0xf];
    }
    return out;
}
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n) {
    return attr_s(ctx, name, hex_encode(arena, p, n), n * 2);
}

static MLIR_OpHandle make_op(MLIR_Context *ctx, MLIR_OpType type,
                             MLIR_AttributeHandle *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt,
                             MLIR_ValueHandle *out_result) {
    MLIR_TypeHandle res_tys[1];
    MLIR_ValueHandle res_vals[1];
    size_t n_res = 0;
    if (result_vt) {
        res_tys[0] = vt_to_type(ctx, result_vt);
        res_vals[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                                               res_tys[0], (string){0},
                                               MLIR_CreateLocationUnknown(ctx, (string){0}));
        n_res = 1;
    }
    MLIR_OpHandle op = MLIR_CreateOp(ctx, type, op_type_to_string(type),
        attrs, n_attrs, res_tys, n_res, res_vals, n_res,
        operands, n_operands, regions, n_regions,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    if (out_result) *out_result = n_res ? res_vals[0] : MLIR_INVALID_HANDLE;
    return op;
}

// Emit an imported (body-less) wasmssa.func op directly into the module body.
static void emit_import_func(MLIR_Context *ctx, Arena *arena,
                             MLIR_BlockHandle body, const char *name,
                             const uint8_t *param_types, size_t n_params,
                             const uint8_t *result_types, size_t n_results) {
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s_cstr(ctx, "sym_name", name ? name : "");
    attrs[na++] = attr_s_hex(ctx, arena, "param_types", param_types, n_params);
    attrs[na++] = attr_s_hex(ctx, arena, "result_types", result_types, n_results);
    attrs[na++] = attr_b(ctx, "exported", false);
    MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
                               attrs, na, NULL, 0, NULL, 0, 0, NULL);
    MLIR_AppendBlockOp(ctx, body, op);
}

// =============================================================================
// Public stage 1 entry point: walk the LLVM-dialect module and emit a
// wasmssa-form `builtin.module` directly. Each function/global is emitted
// straight into the output module body — no module-level intermediate is
// retained. Output ordering (matches the prior two-pass implementation
// byte-for-byte): import_funcs, then defined funcs, then import_globals.
// =============================================================================
MLIR_OpHandle mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    Arena            *arena = MLIR_GetArenaAllocator(ctx);
    MLIR_BlockHandle  body  = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, body);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("builtin.module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    ModCtx mod = {0};
    mod.ctx = ctx;
    mod.arena = arena;
    mod.body = body;

    // Pass 1: imported (declared, body-less) funcs first so they take the
    // low function-index space in wasm.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, op, &p, &np, &r, &nr)) {
            modctx_free_contents(&mod); return MLIR_INVALID_HANDLE;
        }
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        char *nm = xstrdup_str(MLIR_GetAttributeString(sa));

        modctx_add_func_name(&mod, nm);
        emit_import_func(ctx, arena, body, nm, p, np, r, nr);
        free(nm); free(p); free(r);
    }

    // Pass 2: defined funcs in source order. The function's own name is
    // recorded *before* its body is walked so recursive self-address-
    // taking and references between earlier defs work.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        bool has_body = MLIR_GetOpNumRegions(op) > 0 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0;
        if (!has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, op, &p, &np, &r, &nr)) {
            modctx_free_contents(&mod); return MLIR_INVALID_HANDLE;
        }
        MLIR_AttributeHandle sa = find_attr(op, "sym_name");
        string sym = MLIR_GetAttributeString(sa);
        bool is_main = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
        char *nm = is_main ? xstrdupn("__original_main", 15)
                           : xstrdup_str(sym);

        modctx_add_func_name(&mod, nm);
        if (!lower_function(ctx, arena, &mod, body, nm, is_main,
                            op, p, np, r, nr)) {
            free(nm); free(p); free(r);
            modctx_free_contents(&mod);
            return MLIR_INVALID_HANDLE;
        }
        free(nm); free(p); free(r);
    }

    // Pass 3: globals last (matches the legacy emit ordering of
    // funcs-before-globals in the previously buffered module).
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
        if (!lower_global(ctx, arena, body, op)) {
            fprintf(stderr, "wasmssa-lower: failed to lower global\n");
            modctx_free_contents(&mod);
            return MLIR_INVALID_HANDLE;
        }
    }

    modctx_free_contents(&mod);
    return out_module;
}

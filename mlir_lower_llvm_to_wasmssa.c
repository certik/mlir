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
    return 0;
}

// =============================================================================
// Per-function lowering state.
// =============================================================================
typedef struct { uintptr_t key; int idx; } VMapEntry;
typedef struct { uintptr_t key; uint32_t off; } AMapEntry;

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
} FnCtx;

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

// =============================================================================
// Per-op lowering.
// =============================================================================
static bool lower_op(FnCtx *F, MLIR_OpHandle op);

static bool lower_op(FnCtx *F, MLIR_OpHandle op) {
    string name = MLIR_GetOpName(op);

    // ---- llvm.mlir.constant ------------------------------------------------
    if (name_eq(name, "llvm.mlir.constant")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        MLIR_AttributeHandle va = find_attr(op, "value");
        if (va == MLIR_INVALID_HANDLE) return false;
        if (MLIR_GetAttributeKind(va) != MLIR_ATTR_KIND_INTEGER) return false;
        int64_t iv = MLIR_GetAttributeInteger(va);

        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_CONST;
        o->valtype = vt;
        o->i_const = iv;
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

    // ---- llvm.sext (i32 -> i64) -------------------------------------------
    if (name_eq(name, "llvm.sext")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt != WT_I32 || out_vt != WT_I64) return false;
        int sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t *o; int idx = add_op(F, &o);
        o->type = OP_TYPE_WASMSSA_EXTEND_I32_S;
        o->valtype = WT_I64;
        o->n_operands = 1;
        o->operands = (int *)malloc(sizeof(int));
        o->operands[0] = sa;
        o->has_result = true;
        vmap_set(F, r, idx);
        return true;
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

    fprintf(stderr, "wasmssa-lower: unsupported op '%.*s'\n",
            (int)name.size, name.str);
    return false;
}

// =============================================================================
// Function lowering: pre-walk for alloca offsets, then op-by-op walk.
// =============================================================================
static bool prewalk_func(FnCtx *F, MLIR_BlockHandle entry) {
    size_t nops = MLIR_GetBlockNumOps(entry);
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(entry, i);
        string n = MLIR_GetOpName(op);
        if (name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
            name_eq(n, "llvm.switch")) {
            fprintf(stderr,
                    "wasmssa-lower: control flow not supported in scaffold\n");
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
            unsigned align = esz < 4 ? 4 : esz;
            F->frame_size = (F->frame_size + align - 1) & ~(align - 1);
            uint32_t off = F->frame_size;
            F->frame_size += (uint32_t)(esz * cnt);
            amap_set(F, MLIR_GetOpResult(op, 0), off);
        }
    }
    F->frame_size = (F->frame_size + 15) & ~15u;
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
    (void)param_types;
    return true;
fail:
    free(F.vmap); free(F.amap);
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

wasmssa_module_t *mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx,
                                             MLIR_OpHandle module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    wasmssa_module_t *out = wasmssa_module_new();

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

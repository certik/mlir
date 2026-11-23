// Stage 1 of the native LLVM->WASM pipeline:
// LLVM-dialect builtin.module --(walk)--> wasmssa-form `builtin.module`.
//
// Input contract: the module is LLVM dialect ONLY (the output of
// MLIR_LowerToLLVMDialect): `llvm.func` / `llvm.mlir.global` at module
// level, `llvm.*` ops inside function bodies, and control flow as a flat
// CFG of llvm.br / llvm.cond_br / llvm.switch terminators with block
// arguments. No cf.*, scf.*, arith.* or func.* op may be handed in.
//
// WebAssembly has no arbitrary branches, only structured block / loop / if,
// so this stage has to rebuild structured control flow from that CFG. It
// does so in two internal steps:
//
//   1. Structurization: MLIR_LiftCfToScf (mlir_lift_cf_to_scf.c, a port of
//      upstream's CFGToSCF) rewrites each function's llvm.br / llvm.cond_br
//      / llvm.switch CFG, in place, into scf.if / scf.while /
//      scf.index_switch regions terminated by scf.yield / scf.condition.
//   2. Lowering: the walk below maps llvm.* ops and those scf.* regions to
//      wasmssa.* ops.
//
// scf.* is therefore a PRIVATE INTERMEDIATE FORM of the wasm backend: it is
// produced only by step 1 above and consumed only by step 2 (lower_scf_if /
// lower_scf_while / lower_scf_index_switch). Nothing upstream of this file
// creates scf.* for it, and nothing downstream sees it. Note that the input
// module is consumed (step 1 mutates it).
//
// Each `llvm.func` / `llvm.mlir.global` is lowered directly to `wasmssa.*`
// MLIR ops appended into the output module body. Inside a function, each
// emit-helper / inline call site builds a stack-local `wasmssa_op_t`
// describing one op and hands it to `commit_op`, which materializes the
// matching MLIR op into the function's body block and returns the
// MLIR_ValueHandle of its result. The per-function `vmap` then keys
// LLVM-side operand values to the wasmssa-side ValueHandle that supplies
// them; subsequent lookups thread those handles through `o.operands` to
// stitch up the wasmssa IR.
//
// No module-level or per-function scratch buffer is retained: ops flow
// straight from lower_op into the MLIR module under construction.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir_api.h"
#include "mlir_passes.h"
#include "mlir_llvm_to_wasmssa.h"
#include "mlir_llvm_type_layout.h"
#include <base/vector.h>

#include <base/arena.h>
#include <base/string.h>

// =============================================================================
// Private working representation for this stage. `wasmssa_op_t` is a
// stack-local descriptor handed to `commit_op` once per emitted op. It
// never escapes this file and owns none of its pointer fields:
//   - `operands` points at a caller-provided buffer (stack array for
//     fixed-arity ops, arena allocation for variadic call/call_indirect);
//   - `call_target` is a non-owning view into either an MLIR attribute
//     string or a caller-owned buffer;
//   - `sig_params` / `sig_results` are non-owning views into a function
//     signature buffer.
// =============================================================================
typedef struct {
    MLIR_OpType type;
    uint8_t     valtype;

    int64_t  i_const;
    uint32_t global_idx;
    uint32_t memory_offset;
    uint32_t memory_align_log2;
    uint32_t mem_size_bytes;
    string   call_target;
    uint8_t  wasm_opcode;

    const uint8_t *sig_params;
    const uint8_t *sig_results;
    size_t         n_sig_params, n_sig_results;

    const MLIR_ValueHandle *operands;
    int                     n_operands;

    bool has_result;
} wasmssa_op_t;

// Module-level emit context: the output MLIR module's body block. The
// already-emitted `wasmssa.func` / `wasmssa.import_func` ops in that
// block are the source of truth for `is_function_symbol`, so addressof
// references can distinguish between data globals and functions.
typedef struct {
    MLIR_Context    *ctx;
    Arena           *arena;
    MLIR_BlockHandle body;
} ModCtx;

// =============================================================================
// String / type helpers (mirror the old single-stage translator).
// =============================================================================
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}
static MLIR_AttributeHandle find_attr(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    // First try by-name, which on the upstream backend walks both the
    // discardable attribute dictionary AND the typed property storage
    // (ODS-defined inherent attrs like scf.index_switch's `cases`,
    // arith.cmpi's `predicate`, llvm.func's `function_type`, ...).
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, key);
    if (a != MLIR_INVALID_HANDLE) return a;
    // Fall back to the indexed attribute list. The two implementations
    // agree on what's "user-set" for ops constructed via MLIR_CreateOp,
    // so this branch matters mainly for operations created by upstream
    // pass internals where the by-name lookup may also miss.
    size_t n = MLIR_GetOpNumAttributes(ctx, op);
    for (size_t i = 0; i < n; i++) {
        if (MLIR_GetOpAttributeKey(ctx, op, i) == key) return MLIR_GetOpAttribute(ctx, op, i);
    }
    return MLIR_INVALID_HANDLE;
}

// MLIR LLVM-dialect type -> WT_* value-type byte. Returns 0 if the type
// is not a primitive scalar.
static uint8_t wasm_vt(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    uint32_t w = MLIR_GetTypeWidth(ctx, ty);
    if (k == MLIR_TYPE_KIND_LLVM_POINTER || k == MLIR_TYPE_KIND_INDEX) return WT_I32;
    if (k == MLIR_TYPE_KIND_FLOAT) return w == 32 ? WT_F32 : w == 64 ? WT_F64 : 0;
    if (k == MLIR_TYPE_KIND_INTEGER) {
        if (w == 1 || w == 8 || w == 16 || w == 32) return WT_I32;
        if (w == 64) return WT_I64;
    }
    return 0;
}

// Returns the integer bit width of `ty` (1/8/16/32/64; 32 for `index`) or 0
// if `ty` is not an integer type.
static int int_bits(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    if (k == MLIR_TYPE_KIND_INDEX) return 32;
    return k == MLIR_TYPE_KIND_INTEGER ? (int)MLIR_GetTypeWidth(ctx, ty) : 0;
}

// The value of the integer constant `v` as it is materialized (an i8 / i16
// constant zero-extended, see OP_TYPE_LLVM_MLIR_CONSTANT), if `v` is one.
static bool known_const(MLIR_Context *ctx, MLIR_ValueHandle v, int64_t *out) {
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(ctx, v);
    if (d == MLIR_INVALID_HANDLE || MLIR_GetOpType(ctx, d) != OP_TYPE_LLVM_MLIR_CONSTANT)
        return false;
    MLIR_AttributeHandle va = find_attr(ctx, d, MLIR_KEY_VALUE);
    if (va == MLIR_INVALID_HANDLE || MLIR_GetAttributeKind(ctx, va) != MLIR_ATTR_KIND_INTEGER)
        return false;
    int64_t c = MLIR_GetAttributeInteger(ctx, va);
    int w = int_bits(ctx, MLIR_GetValueType(ctx, v));
    if (w == 8 || w == 16) c &= ((int64_t)1 << w) - 1;
    *out = c;
    return true;
}

// Whether the i32 that holds the integer `v` (of 32 bits or fewer) is known
// to be in [0, 2^w): a narrow load (i32.load8_u / load16_u), a zero
// extension, an `and` with a small mask, a comparison, ... Lets a zext /
// sext of a narrow value that is already in range skip its mask / extend.
static bool known_zext(MLIR_Context *ctx, MLIR_ValueHandle v, int w, int depth) {
    int64_t c;
    if (known_const(ctx, v, &c)) return c >= 0 && c < ((int64_t)1 << w);
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(ctx, v);
    if (d == MLIR_INVALID_HANDLE || depth > 4) return false;
    switch (MLIR_GetOpType(ctx, d)) {
    case OP_TYPE_LLVM_LOAD: case OP_TYPE_LLVM_ZEXT: {
        MLIR_ValueHandle src = MLIR_GetOpType(ctx, d) == OP_TYPE_LLVM_LOAD
                             ? v : MLIR_GetOpOperand(ctx, d, 0);
        int sw = int_bits(ctx, MLIR_GetValueType(ctx, src));
        return sw > 0 && sw <= w && sw < 32;
    }
    case OP_TYPE_LLVM_ICMP:
        return true;
    case OP_TYPE_LLVM_TRUNC:
        return known_zext(ctx, MLIR_GetOpOperand(ctx, d, 0), w, depth + 1);
    case OP_TYPE_LLVM_AND:
        return (known_const(ctx, MLIR_GetOpOperand(ctx, d, 1), &c) &&
                c >= 0 && c < ((int64_t)1 << w)) ||
               (known_const(ctx, MLIR_GetOpOperand(ctx, d, 0), &c) &&
                c >= 0 && c < ((int64_t)1 << w));
    case OP_TYPE_LLVM_SELECT:
        return known_zext(ctx, MLIR_GetOpOperand(ctx, d, 1), w, depth + 1) &&
               known_zext(ctx, MLIR_GetOpOperand(ctx, d, 2), w, depth + 1);
    default:
        return false;
    }
}

// Whether the i32 that holds the integer `v` is known to be a w-bit value
// sign-extended to 32 bits.
static bool known_sext(MLIR_Context *ctx, MLIR_ValueHandle v, int w, int depth) {
    int64_t c;
    if (known_const(ctx, v, &c))
        return c >= -((int64_t)1 << (w - 1)) && c < ((int64_t)1 << (w - 1));
    if (known_zext(ctx, v, w - 1, depth)) return true;
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(ctx, v);
    if (d == MLIR_INVALID_HANDLE || depth > 4) return false;
    switch (MLIR_GetOpType(ctx, d)) {
    case OP_TYPE_LLVM_SEXT: {
        int sw = int_bits(ctx, MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, d, 0)));
        return sw > 0 && sw <= w;
    }
    case OP_TYPE_LLVM_ASHR:
        // `ashr (x, k)` of an i32 with k >= 32 - w (the shl / ashr pair that
        // sign-extends from w bits).
        return int_bits(ctx, MLIR_GetValueType(ctx, v)) == 32 &&
               known_const(ctx, MLIR_GetOpOperand(ctx, d, 1), &c) && c >= 32 - w && c < 32;
    case OP_TYPE_LLVM_TRUNC:
        return known_sext(ctx, MLIR_GetOpOperand(ctx, d, 0), w, depth + 1);
    case OP_TYPE_LLVM_SELECT:
        return known_sext(ctx, MLIR_GetOpOperand(ctx, d, 1), w, depth + 1) &&
               known_sext(ctx, MLIR_GetOpOperand(ctx, d, 2), w, depth + 1);
    default:
        return false;
    }
}

// Type layout (mlir_llvm_type_layout.h) on wasm32: pointers are 4 bytes.
static unsigned type_size_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return llvm_type_size(ctx, ty, 4);
}
static unsigned type_align_bytes(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return llvm_type_align(ctx, ty, 4);
}
static unsigned struct_field_offset(MLIR_Context *ctx, MLIR_TypeHandle sty,
                                    size_t fld_idx) {
    return llvm_struct_field_offset(ctx, sty, fld_idx, 4);
}

// =============================================================================
// Per-function lowering state.
// =============================================================================
typedef struct { uintptr_t key; MLIR_ValueHandle val; } VMapEntry;
typedef struct { uintptr_t key; uint32_t off; } AMapEntry;

DEFINE_VECTOR_FOR_TYPE(uint8_t,    VecU8)

typedef struct {
    MLIR_Context   *ctx;
    Arena          *arena;   // for hex-encoded attribute strings
    ModCtx         *mod;     // parent module emit context (for func-name lookups)
    size_t          n_params;

    // Direct-MLIR emission state. body_block accumulates wasmssa.* ops.
    MLIR_BlockHandle  body_block;

    // Map LLVM-dialect MLIR_ValueHandle -> wasmssa MLIR_ValueHandle, as an
    // open-addressing hash table (key 0 == empty; source value handles are
    // always non-zero). Replaces an earlier linear-scan vector that made
    // lowering O(n^2) in the number of values per function.
    VMapEntry *vmap;
    size_t     vmap_cap, vmap_n;   // cap is 0 or a power of two

    // Map MLIR_ValueHandle (alloca result) -> shadow-stack frame offset.
    AMapEntry *amap;
    size_t     amap_cap, amap_n;

    uint32_t  frame_size;
    MLIR_ValueHandle  sp_value;        // post-decrement SP, or MLIR_INVALID_HANDLE

    // Variadic ABI support.
    MLIR_ValueHandle  va_list_value;   // hidden trailing va_list i32 param, or MLIR_INVALID_HANDLE
    uint32_t  va_buf_size;        // max bytes needed for variadic call buffer
    uint32_t  va_buf_offset;      // offset within shadow frame for the variadic buffer
} FnCtx;

// Open-addressing hash maps keyed by MLIR value handles. The handle is never
// zero, so key==0 marks an empty slot. Lookups are key-only (never iterated in
// an output-affecting way), and the 32-bit-safe mix below is identical on the
// wasm32 self-host (uintptr_t == 32-bit) and the 64-bit host, so the emitted
// module stays deterministic / bit-identical across the self-host cycle.
static size_t map_hash(uintptr_t k) {
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}

static void vmap_grow(FnCtx *F) {
    size_t ncap = F->vmap_cap ? F->vmap_cap * 2 : 32;
    VMapEntry *nt = (VMapEntry *)arena_alloc(F->arena, ncap * sizeof(VMapEntry));
    memset(nt, 0, ncap * sizeof(VMapEntry));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < F->vmap_cap; i++) {
        if (F->vmap[i].key == 0) continue;
        size_t j = map_hash(F->vmap[i].key) & mask;
        while (nt[j].key != 0) j = (j + 1) & mask;
        nt[j] = F->vmap[i];
    }
    F->vmap = nt;
    F->vmap_cap = ncap;
}
static void vmap_set(FnCtx *F, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    // Grow at 75% load to keep probe chains short.
    if ((F->vmap_n + 1) * 4 >= F->vmap_cap * 3) vmap_grow(F);
    size_t mask = F->vmap_cap - 1;
    size_t i = map_hash((uintptr_t)k) & mask;
    while (F->vmap[i].key != 0) {
        if (F->vmap[i].key == (uintptr_t)k) return;  // first insert wins
        i = (i + 1) & mask;
    }
    F->vmap[i].key = (uintptr_t)k;
    F->vmap[i].val = v;
    F->vmap_n++;
}
static int vmap_get(FnCtx *F, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    if (F->vmap_cap == 0) return 0;
    size_t mask = F->vmap_cap - 1;
    size_t i = map_hash((uintptr_t)k) & mask;
    while (F->vmap[i].key != 0) {
        if (F->vmap[i].key == (uintptr_t)k) { *out = F->vmap[i].val; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}
static void amap_grow(FnCtx *F) {
    size_t ncap = F->amap_cap ? F->amap_cap * 2 : 16;
    AMapEntry *nt = (AMapEntry *)arena_alloc(F->arena, ncap * sizeof(AMapEntry));
    memset(nt, 0, ncap * sizeof(AMapEntry));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < F->amap_cap; i++) {
        if (F->amap[i].key == 0) continue;
        size_t j = map_hash(F->amap[i].key) & mask;
        while (nt[j].key != 0) j = (j + 1) & mask;
        nt[j] = F->amap[i];
    }
    F->amap = nt;
    F->amap_cap = ncap;
}
static void amap_set(FnCtx *F, MLIR_ValueHandle v, uint32_t off) {
    if ((F->amap_n + 1) * 4 >= F->amap_cap * 3) amap_grow(F);
    size_t mask = F->amap_cap - 1;
    size_t i = map_hash((uintptr_t)v) & mask;
    while (F->amap[i].key != 0) {
        if (F->amap[i].key == (uintptr_t)v) return;  // first insert wins
        i = (i + 1) & mask;
    }
    F->amap[i].key = (uintptr_t)v;
    F->amap[i].off = off;
    F->amap_n++;
}
static int amap_get(FnCtx *F, MLIR_ValueHandle v, uint32_t *out) {
    if (F->amap_cap == 0) return 0;
    size_t mask = F->amap_cap - 1;
    size_t i = map_hash((uintptr_t)v) & mask;
    while (F->amap[i].key != 0) {
        if (F->amap[i].key == (uintptr_t)v) { *out = F->amap[i].off; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

// Forward decls used by commit_op (defined further down with the other
// MLIR-emit helpers).
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt);
static MLIR_NamedAttr attr_i32(MLIR_Context *ctx, MLIR_AttrKey key, int64_t v);
static MLIR_NamedAttr attr_i64(MLIR_Context *ctx, MLIR_AttrKey key, int64_t v);
static MLIR_NamedAttr attr_s_cstr(MLIR_Context *ctx, MLIR_AttrKey key, const char *v);
static MLIR_NamedAttr attr_b(MLIR_Context *ctx, MLIR_AttrKey key, bool v);
static MLIR_NamedAttr attr_s(MLIR_Context *ctx, MLIR_AttrKey key,
                                   const char *v, size_t vlen);
static MLIR_NamedAttr attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       MLIR_AttrKey key,
                                       const uint8_t *p, size_t n);
static MLIR_OpHandle make_op(MLIR_Context *ctx, MLIR_OpType type,
                             MLIR_NamedAttr *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt,
                             MLIR_ValueHandle *out_result);

// Materialize a wasmssa op directly into the function's MLIR body block.
// Returns the op's MLIR result value (MLIR_INVALID_HANDLE for ops with no
// result). The op's operands[] reference earlier-produced MLIR values.
static MLIR_ValueHandle commit_op(FnCtx *F, wasmssa_op_t *o) {
    MLIR_Context *ctx = F->ctx;
    size_t n_ops = (size_t)(o->n_operands < 0 ? 0 : o->n_operands);
    MLIR_NamedAttr as[8];
    size_t nas = 0;
    as[nas++] = attr_i32(ctx, MLIR_KEY_VALTYPE, o->valtype);
    switch (o->type) {
    case OP_TYPE_WASMSSA_CONST:
        as[nas++] = attr_i64(ctx, MLIR_KEY_VALUE, o->i_const);
        break;
    case OP_TYPE_WASMSSA_BINOP:
    case OP_TYPE_WASMSSA_UNOP:
        as[nas++] = attr_i32(ctx, MLIR_KEY_WASM_OPCODE, o->wasm_opcode);
        break;
    case OP_TYPE_WASMSSA_LOAD:
    case OP_TYPE_WASMSSA_STORE:
        as[nas++] = attr_i32(ctx, MLIR_KEY_MEMORY_OFFSET,     (int64_t)o->memory_offset);
        as[nas++] = attr_i32(ctx, MLIR_KEY_MEMORY_ALIGN_LOG2, (int64_t)o->memory_align_log2);
        as[nas++] = attr_i32(ctx, MLIR_KEY_MEM_SIZE_BYTES,    (int64_t)o->mem_size_bytes);
        break;
    case OP_TYPE_WASMSSA_GLOBAL_GET:
    case OP_TYPE_WASMSSA_GLOBAL_SET:
        as[nas++] = attr_i32(ctx, MLIR_KEY_GLOBAL_IDX, (int64_t)o->global_idx);
        break;
    case OP_TYPE_WASMSSA_CALL:
    case OP_TYPE_WASMSSA_ADDRESSOF:
    case OP_TYPE_WASMSSA_FUNC_ADDR:
        as[nas++] = attr_s(ctx, MLIR_KEY_TARGET, o->call_target.str, o->call_target.size);
        break;
    case OP_TYPE_WASMSSA_CALL_INDIRECT:
        as[nas++] = attr_s_hex(ctx, F->arena, MLIR_KEY_SIG_PARAMS,
                               o->sig_params, o->n_sig_params);
        as[nas++] = attr_s_hex(ctx, F->arena, MLIR_KEY_SIG_RESULTS,
                               o->sig_results, o->n_sig_results);
        break;
    default: break;
    }
    MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
    MLIR_OpHandle mop = make_op(ctx, o->type, as, nas,
                                (MLIR_ValueHandle *)o->operands, n_ops, NULL, 0,
                                o->has_result ? o->valtype : 0, &res);
    MLIR_AppendBlockOp(ctx, F->body_block, mop);
    return o->has_result ? res : MLIR_INVALID_HANDLE;
}

// Convenience: const-i32.
static MLIR_ValueHandle emit_const_i32(FnCtx *F, int32_t v) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_CONST;
    o.valtype = WT_I32;
    o.i_const = v;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32 add of two ssa-def operands.
static MLIR_ValueHandle emit_add_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_ADD;
    o.valtype = WT_I32;
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32 and.
static MLIR_ValueHandle emit_and_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_BINOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0x71;  // i32.and
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32 sub.
static MLIR_ValueHandle emit_sub_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_SUB;
    o.valtype = WT_I32;
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: global_get of an i32 global.
static MLIR_ValueHandle emit_global_get(FnCtx *F, uint32_t gidx) {
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_GLOBAL_GET;
    o.valtype = WT_I32;
    o.global_idx = gidx;
    o.has_result = true;
    return commit_op(F, &o);
}
static void emit_global_set(FnCtx *F, uint32_t gidx, MLIR_ValueHandle valv) {
    MLIR_ValueHandle ops[1] = { valv };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_GLOBAL_SET;
    o.valtype = WT_I32;
    o.global_idx = gidx;
    o.n_operands = 1;
    o.operands = ops;
    (void)commit_op(F, &o);
}

// Convenience: i32 mul.
static MLIR_ValueHandle emit_mul_i32(FnCtx *F, MLIR_ValueHandle lhs, MLIR_ValueHandle rhs) {
    MLIR_ValueHandle ops[2] = { lhs, rhs };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_BINOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0x6c;  // i32.mul
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}
// Convenience: i32.wrap_i64 (0xa7) — narrows an i64 SSA value to i32.
static MLIR_ValueHandle emit_wrap_i64_to_i32(FnCtx *F, MLIR_ValueHandle v) {
    MLIR_ValueHandle ops[1] = { v };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_UNOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0xa7;
    o.n_operands = 1;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}

// The elements of a dense i32 array attribute, arena-allocated; NULL for an
// empty array or another kind. Sets *n_out.
static int32_t *parse_dense_i32_array(MLIR_Context *ctx, Arena *arena, MLIR_AttributeHandle a,
                                      size_t *n_out) {
    size_t n = MLIR_GetAttributeDenseSize(ctx, a);
    *n_out = n;
    if (n == 0) return NULL;
    int32_t *vs = (int32_t *)arena_alloc(arena, n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) vs[i] = (int32_t)MLIR_GetAttributeDenseElement(ctx, a, i);
    return vs;
}

// Look up a name as a function symbol in the module under construction
// by walking m->body for `wasmssa.func` / `wasmssa.import_func` ops
// whose `sym_name` matches. Used to disambiguate addressof @sym between
// data globals and functions. The function currently being lowered is
// not in m->body yet (its wrapper op is appended after the body is
// walked), so recursive self-address-taking falls through to the
// addressof path; that's fine because main / defined funcs that take
// their own address are rare and not exercised by the test corpus.
static bool is_function_symbol(const ModCtx *m, const char *nm, size_t nlen) {
    size_t n = MLIR_GetBlockNumOps(m->ctx, m->body);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(m->ctx, m->body, i);
        MLIR_OpType ot = MLIR_GetOpType(m->ctx, op);
        if (ot != OP_TYPE_WASMSSA_FUNC && ot != OP_TYPE_WASMSSA_IMPORT_FUNC) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttr(m->ctx, op, MLIR_KEY_SYM_NAME);
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(m->ctx, sa);
        if (s.size == nlen && memcmp(s.str, nm, nlen) == 0) return true;
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

static MLIR_ValueHandle emit_eqz(FnCtx *F, MLIR_ValueHandle v) {
    MLIR_ValueHandle ops[1] = { v };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_EQZ;
    o.valtype = WT_I32;
    o.n_operands = 1;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}

// =============================================================================
// Block-arg-mode helpers (new path).
// =============================================================================

// Multi-result variant of make_op: supports ops with N>=0 results and N>=0
// regions. Does NOT auto-add a "valtype" attr (the region-bearing ops and
// the block_return / unreachable / br-with-args ops don't have one).
static MLIR_OpHandle make_op_n(MLIR_Context *ctx, MLIR_OpType type,
                               MLIR_NamedAttr *attrs, size_t n_attrs,
                               MLIR_ValueHandle *operands, size_t n_operands,
                               MLIR_RegionHandle *regions, size_t n_regions,
                               const uint8_t *result_vts, size_t n_results,
                               MLIR_ValueHandle *out_results /* size n_results, or NULL */) {
    MLIR_TypeHandle   res_tys_inline[16];
    MLIR_ValueHandle  res_vals_inline[16];
    MLIR_TypeHandle *res_tys = res_tys_inline;
    if (n_results > 16) {
        res_tys = (MLIR_TypeHandle *)arena_alloc(MLIR_GetArenaAllocator(ctx), n_results * sizeof(MLIR_TypeHandle));
    }
    for (size_t i = 0; i < n_results; i++) {
        res_tys[i] = vt_to_type(ctx, result_vts[i]);
    }
    MLIR_OpSpec spec = {0};
    spec.type = type;
    spec.operands = operands;
    spec.n_operands = (uint32_t)n_operands;
    spec.result_types = res_tys;
    spec.n_results = (uint32_t)n_results;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)n_attrs;
    spec.regions = regions;
    spec.n_regions = (uint32_t)n_regions;
    MLIR_OpHandle op = MLIR_CreateOpFromSpec(ctx, &spec);
    if (out_results) {
        for (size_t i = 0; i < n_results; i++) out_results[i] = MLIR_GetOpResult(ctx, op, i);
    }
    return op;
}

// Emit `wasmssa.block_return (vals...)` as the terminator of F->body_block.
static void emit_block_return(FnCtx *F, MLIR_ValueHandle *vals, size_t n) {
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BLOCK_RETURN, NULL, 0,
                                 vals, n, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Emit `wasmssa.unreachable` as the terminator of F->body_block.
static void emit_unreachable(FnCtx *F) {
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE, NULL, 0,
                                 NULL, 0, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Emit `wasmssa.br {depth=N} (vals...)`.
static void emit_br_args(FnCtx *F, uint32_t depth, MLIR_ValueHandle *vals, size_t n) {
    MLIR_NamedAttr as[1];
    as[0] = attr_i32(F->ctx, MLIR_KEY_DEPTH, (int64_t)depth);
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BR, as, 1,
                                 vals, n, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Emit a `wasmssa.br_if {depth=N} (cond)`. Block-arg design: br_if carries
// the cond only — value-carrying conditional branches use
// `wasmssa.if (cond) { br N (vals) }` instead.
static void emit_br_if(FnCtx *F, uint32_t depth, MLIR_ValueHandle cond) {
    MLIR_NamedAttr as[1];
    as[0] = attr_i32(F->ctx, MLIR_KEY_DEPTH, (int64_t)depth);
    MLIR_ValueHandle ops[1] = { cond };
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BR_IF, as, 1,
                                 ops, 1, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, op);
}

// Build a fresh detached block with `n_args` block args of the given types.
// out_args (size n_args, may be NULL) receives the new block-arg values.
static MLIR_BlockHandle make_block_with_args(MLIR_Context *ctx,
                                             const uint8_t *arg_vts, size_t n_args,
                                             MLIR_ValueHandle *out_args) {
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    for (size_t i = 0; i < n_args; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, arg_vts[i]);
        MLIR_ValueHandle a = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                     MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, blk, a);
        if (out_args) out_args[i] = a;
    }
    return blk;
}

// ---- scf.if ----------------------------------------------------------------
// Block-arg form: build a `wasmssa.if (%cond) : (R*)` op with two regions,
// each terminated by `wasmssa.block_return (yield_vals:R*)`. scf.if results
// map 1:1 to the wasmssa.if's results.
static bool lower_scf_if(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(F->ctx, op, 0);
    MLIR_ValueHandle cond_idx;
    if (!vmap_get(F, cond_v, &cond_idx)) return false;

    size_t n_res = MLIR_GetOpNumResults(F->ctx, op);
    uint8_t *res_vts = NULL;
    if (n_res) {
        res_vts = (uint8_t *)arena_alloc(F->arena, n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, MLIR_GetOpResult(F->ctx, op, i)));
            if (vt == 0) return false;
            res_vts[i] = vt;
        }
    }

    if (MLIR_GetOpNumRegions(F->ctx, op) < 1) return false;
    MLIR_RegionHandle src_then = MLIR_GetOpRegion(F->ctx, op, 0);
    if (MLIR_GetRegionNumBlocks(F->ctx, src_then) != 1) return false;
    bool has_else = MLIR_GetOpNumRegions(F->ctx, op) >= 2 &&
                    MLIR_GetRegionNumBlocks(F->ctx, MLIR_GetOpRegion(F->ctx, op, 1)) > 0;
    if (n_res && !has_else) return false;  // result-bearing scf.if needs else

    // Save outer body; walk each scf.if region into a fresh wasmssa block.
    MLIR_BlockHandle saved = F->body_block;

    // --- then-region ---
    MLIR_BlockHandle then_blk = MLIR_CreateBlock(F->ctx);
    F->body_block = then_blk;
    bool ok = lower_block(F, MLIR_GetRegionBlock(F->ctx, src_then, 0));
    if (!ok) { F->body_block = saved; return false; }

    // --- else-region (may be empty) ---
    MLIR_BlockHandle else_blk = MLIR_INVALID_HANDLE;
    if (has_else) {
        else_blk = MLIR_CreateBlock(F->ctx);
        F->body_block = else_blk;
        ok = lower_block(F, MLIR_GetRegionBlock(F->ctx, MLIR_GetOpRegion(F->ctx, op, 1), 0));
        if (!ok) { F->body_block = saved; return false; }
    }
    F->body_block = saved;

    // Wrap blocks in regions and build the wasmssa.if op.
    MLIR_RegionHandle then_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, then_r, then_blk);
    MLIR_RegionHandle regs[2];
    regs[0] = then_r;
    size_t n_regs = 1;
    if (has_else) {
        MLIR_RegionHandle else_r = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, else_r, else_blk);
        regs[1] = else_r;
        n_regs = 2;
    }

    MLIR_NamedAttr as[1];
    size_t nas = 0;
    if (n_res) {
        as[nas++] = attr_s_hex(F->ctx, F->arena, MLIR_KEY_RESULT_TYPES, res_vts, n_res);
    }

    MLIR_ValueHandle res_vals_inline[16];
    MLIR_ValueHandle *res_vals = n_res <= 16 ? res_vals_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    MLIR_ValueHandle cond_ops[1] = { cond_idx };
    MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, as, nas,
                                   cond_ops, 1, regs, n_regs,
                                   res_vts, n_res, res_vals);
    MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);

    for (size_t i = 0; i < n_res; i++) {
        vmap_set(F, MLIR_GetOpResult(F->ctx, op, i), res_vals[i]);
    }
    return true;
}

// ---- scf.while -------------------------------------------------------------
// General shape (with iter args / results):
//   %res:R = scf.while (%a:T = %init:T) : (T...) -> (R...) {
//     ^before(%a:T...): ...; scf.condition(%cond) %r:R...
//   } do {
//     ^after(%b:R...): ...; scf.yield %y:T...
//   }
//
// Lowered to (block-arg form):
//   %res:R* = wasmssa.block () : (R*) {
//     wasmssa.loop (%init:T*) : () {
//     ^entry(%t:T*):                                ; bound to %a in before
//       <before-body>
//       wasmssa.if (eqz_cond) : () {
//         wasmssa.br {depth=2} (r:R*)               ; exit outer block
//       }                                            ; (no else)
//       ; fall through if cond was true.
//       <after-body>                                 ; uses r as %b
//       wasmssa.br {depth=0} (y:T*)                  ; back-edge
//     }
//     wasmssa.unreachable
//   }
static bool lower_scf_while(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumRegions(F->ctx, op) < 2) return false;
    MLIR_RegionHandle before_r = MLIR_GetOpRegion(F->ctx, op, 0);
    MLIR_RegionHandle after_r  = MLIR_GetOpRegion(F->ctx, op, 1);
    if (MLIR_GetRegionNumBlocks(F->ctx, before_r) != 1) return false;
    if (MLIR_GetRegionNumBlocks(F->ctx, after_r)  != 1) return false;
    MLIR_BlockHandle before_b = MLIR_GetRegionBlock(F->ctx, before_r, 0);
    MLIR_BlockHandle after_b  = MLIR_GetRegionBlock(F->ctx, after_r, 0);

    size_t n_iter = MLIR_GetOpNumOperands(F->ctx, op);
    size_t n_res  = MLIR_GetOpNumResults(F->ctx, op);

    if (n_iter != MLIR_GetBlockNumArgs(F->ctx, before_b)) return false;
    if (n_res  != MLIR_GetBlockNumArgs(F->ctx, after_b))  return false;

    // Collect iter (T*) and result (R*) wasm valtypes.
    uint8_t *iter_vts = NULL;
    if (n_iter) {
        iter_vts = (uint8_t *)arena_alloc(F->arena, n_iter);
        for (size_t i = 0; i < n_iter; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, MLIR_GetOpOperand(F->ctx, op, i)));
            if (vt == 0) return false;
            iter_vts[i] = vt;
        }
    }
    uint8_t *res_vts = NULL;
    if (n_res) {
        res_vts = (uint8_t *)arena_alloc(F->arena, n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, MLIR_GetOpResult(F->ctx, op, i)));
            if (vt == 0) return false;
            res_vts[i] = vt;
        }
    }

    // Resolve init operands in the outer scope.
    MLIR_ValueHandle *init_vals = NULL;
    if (n_iter) {
        init_vals = (MLIR_ValueHandle *)arena_alloc(F->arena, n_iter * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < n_iter; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, i), &init_vals[i])) return false;
        }
    }

    // Build the loop body block with entry block args = T*; bind before-region's
    // block args to these so the before-body sees them as %a:T*.
    MLIR_BlockHandle loop_body = MLIR_CreateBlock(F->ctx);
    for (size_t i = 0; i < n_iter; i++) {
        MLIR_TypeHandle ty = vt_to_type(F->ctx, iter_vts[i]);
        MLIR_ValueHandle a = MLIR_CreateValueBlockArg(F->ctx, (string){0}, (uint32_t)i, ty,
                                                     MLIR_CreateLocationUnknown(F->ctx, (string){0}));
        MLIR_AppendBlockArg(F->ctx, loop_body, a);
        vmap_set(F, MLIR_GetBlockArg(F->ctx, before_b, i), a);
    }

    MLIR_BlockHandle saved = F->body_block;
    F->body_block = loop_body;

    // Walk before-block ops. scf.condition (which must be the terminator)
    // triggers emission of the exit-if + after-body walk in the same loop_body.
    size_t nb = MLIR_GetBlockNumOps(F->ctx, before_b);
    bool saw_cond = false;
    for (size_t i = 0; i < nb; i++) {
        MLIR_OpHandle bop = MLIR_GetBlockOp(F->ctx, before_b, i);
        if (MLIR_GetOpType(F->ctx, bop) == OP_TYPE_SCF_CONDITION) {
            saw_cond = true;
            if (MLIR_GetOpNumOperands(F->ctx, bop) < 1) { F->body_block = saved; return false; }
            if (MLIR_GetOpNumOperands(F->ctx, bop) - 1 != n_res) { F->body_block = saved; return false; }

            MLIR_ValueHandle cidx;
            if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, bop, 0), &cidx)) { F->body_block = saved; return false; }
            MLIR_ValueHandle z = emit_eqz(F, cidx);

            // Resolve the scf.condition payload values in the loop_body scope.
            MLIR_ValueHandle r_vals_inline[16];
            MLIR_ValueHandle *r_vals = n_res <= 16 ? r_vals_inline
                : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
            for (size_t k = 0; k < n_res; k++) {
                if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, bop, k + 1), &r_vals[k])) {
                    F->body_block = saved; return false;
                }
            }

            // Build the exit-if: wasmssa.if (z) : () { wasmssa.br {depth=2} (r_vals) }.
            MLIR_BlockHandle then_blk = MLIR_CreateBlock(F->ctx);
            F->body_block = then_blk;
            emit_br_args(F, /*depth*/2, r_vals, n_res);
            F->body_block = loop_body;

            MLIR_RegionHandle then_r2 = MLIR_CreateRegion(F->ctx);
            MLIR_AppendRegionBlock(F->ctx, then_r2, then_blk);
            MLIR_RegionHandle regs[1] = { then_r2 };
            MLIR_ValueHandle cond_ops[1] = { z };
            MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, NULL, 0,
                                           cond_ops, 1, regs, 1,
                                           NULL, 0, NULL);
            MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);

            // Bind after-region's block args to the scf.condition payload.
            for (size_t k = 0; k < n_res; k++) {
                vmap_set(F, MLIR_GetBlockArg(F->ctx, after_b, k), r_vals[k]);
            }

            // Walk after-region ops; scf.yield becomes the back-edge.
            size_t na = MLIR_GetBlockNumOps(F->ctx, after_b);
            bool saw_yield = false;
            for (size_t j = 0; j < na; j++) {
                MLIR_OpHandle aop = MLIR_GetBlockOp(F->ctx, after_b, j);
                if (MLIR_GetOpType(F->ctx, aop) == OP_TYPE_SCF_YIELD) {
                    saw_yield = true;
                    if (MLIR_GetOpNumOperands(F->ctx, aop) != n_iter) { F->body_block = saved; return false; }
                    MLIR_ValueHandle y_vals_inline[16];
                    MLIR_ValueHandle *y_vals = n_iter <= 16 ? y_vals_inline
                        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_iter * sizeof(MLIR_ValueHandle));
                    for (size_t k = 0; k < n_iter; k++) {
                        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, aop, k), &y_vals[k])) {
                            F->body_block = saved; return false;
                        }
                    }
                    emit_br_args(F, /*depth*/0, y_vals, n_iter);
                } else {
                    if (!lower_op(F, aop)) { F->body_block = saved; return false; }
                }
            }
            if (!saw_yield && n_iter) { F->body_block = saved; return false; }
            break;  // scf.condition is the terminator of before-region.
        } else {
            if (!lower_op(F, bop)) { F->body_block = saved; return false; }
        }
    }
    if (!saw_cond) { F->body_block = saved; return false; }
    F->body_block = saved;

    // Build the wasmssa.loop op (no results; never falls through).
    MLIR_RegionHandle loop_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, loop_r, loop_body);
    MLIR_OpHandle loop_op = make_op_n(F->ctx, OP_TYPE_WASMSSA_LOOP, NULL, 0,
                                      init_vals, n_iter, &loop_r, 1,
                                      NULL, 0, NULL);

    // Build the outer wasmssa.block containing [loop_op; wasmssa.unreachable].
    MLIR_BlockHandle outer_blk = MLIR_CreateBlock(F->ctx);
    MLIR_AppendBlockOp(F->ctx, outer_blk, loop_op);
    {
        MLIR_OpHandle u = make_op_n(F->ctx, OP_TYPE_WASMSSA_UNREACHABLE, NULL, 0,
                                    NULL, 0, NULL, 0, NULL, 0, NULL);
        MLIR_AppendBlockOp(F->ctx, outer_blk, u);
    }
    MLIR_RegionHandle block_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, block_r, outer_blk);
    MLIR_ValueHandle block_res_inline[16];
    MLIR_ValueHandle *block_res_vals = n_res <= 16 ? block_res_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle block_op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BLOCK, NULL, 0,
                                       NULL, 0, &block_r, 1,
                                       res_vts, n_res, block_res_vals);
    MLIR_AppendBlockOp(F->ctx, F->body_block, block_op);

    for (size_t i = 0; i < n_res; i++) {
        vmap_set(F, MLIR_GetOpResult(F->ctx, op, i), block_res_vals[i]);
    }
    return true;
}

// ---- scf.index_switch ------------------------------------------------------
// Lowered as a chain of `wasmssa.if (cond==case_i) : (R*) { case_body }
// else { recurse }` — innermost else holds the default region. Results
// propagate via block_return up the chain.
static MLIR_ValueHandle emit_i32_eq(FnCtx *F, MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle ops[2] = { a, b };
    wasmssa_op_t o = {0};
    o.type = OP_TYPE_WASMSSA_BINOP;
    o.valtype = WT_I32;
    o.wasm_opcode = 0x46;  // i32.eq
    o.n_operands = 2;
    o.operands = ops;
    o.has_result = true;
    return commit_op(F, &o);
}

// Recursively build the case chain. The wasmssa.if op for case `i` is
// appended to F->body_block; its R* results are copied into *out_res.
static bool isw_build_chain(FnCtx *F, MLIR_OpHandle op,
                            size_t i, size_t n_cases,
                            const int64_t *case_vals,
                            MLIR_ValueHandle cond_idx,
                            size_t n_res, const uint8_t *res_vts,
                            MLIR_ValueHandle *out_res /* size n_res */) {
    MLIR_ValueHandle kc = emit_const_i32(F, (int32_t)case_vals[i]);
    MLIR_ValueHandle cmp = emit_i32_eq(F, cond_idx, kc);

    MLIR_BlockHandle saved = F->body_block;

    // then = case i body.
    MLIR_BlockHandle then_blk = MLIR_CreateBlock(F->ctx);
    F->body_block = then_blk;
    bool ok = lower_block(F, MLIR_GetRegionBlock(F->ctx, MLIR_GetOpRegion(F->ctx, op, i + 1), 0));
    if (!ok) { F->body_block = saved; return false; }

    // else = inner chain or default region.
    MLIR_BlockHandle else_blk = MLIR_CreateBlock(F->ctx);
    F->body_block = else_blk;
    if (i + 1 == n_cases) {
        ok = lower_block(F, MLIR_GetRegionBlock(F->ctx, MLIR_GetOpRegion(F->ctx, op, 0), 0));
        if (!ok) { F->body_block = saved; return false; }
    } else {
        MLIR_ValueHandle inner_inline[16];
        MLIR_ValueHandle *inner_res = n_res <= 16 ? inner_inline
            : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
        if (!isw_build_chain(F, op, i + 1, n_cases, case_vals, cond_idx,
                             n_res, res_vts, inner_res)) {
            F->body_block = saved; return false;
        }
        emit_block_return(F, inner_res, n_res);
    }

    F->body_block = saved;

    MLIR_RegionHandle then_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, then_r, then_blk);
    MLIR_RegionHandle else_r = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, else_r, else_blk);
    MLIR_RegionHandle regs[2] = { then_r, else_r };

    MLIR_NamedAttr as[1];
    size_t nas = 0;
    if (n_res) as[nas++] = attr_s_hex(F->ctx, F->arena, MLIR_KEY_RESULT_TYPES, res_vts, n_res);
    MLIR_ValueHandle cond_ops[1] = { cmp };
    MLIR_ValueHandle res_inline[16];
    MLIR_ValueHandle *res_vals = n_res <= 16 ? res_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle ifop = make_op_n(F->ctx, OP_TYPE_WASMSSA_IF, as, nas,
                                   cond_ops, 1, regs, 2,
                                   res_vts, n_res, res_vals);
    MLIR_AppendBlockOp(F->ctx, F->body_block, ifop);
    for (size_t k = 0; k < n_res; k++) out_res[k] = res_vals[k];
    return true;
}

static bool lower_scf_index_switch(FnCtx *F, MLIR_OpHandle op) {
    if (MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
    MLIR_ValueHandle cond_v = MLIR_GetOpOperand(F->ctx, op, 0);
    MLIR_ValueHandle cond_idx;
    if (!vmap_get(F, cond_v, &cond_idx)) return false;
    uint8_t cond_vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, cond_v));
    if (cond_vt == WT_I64)
        cond_idx = emit_wrap_i64_to_i32(F, cond_idx);
    else if (cond_vt != WT_I32)
        return false;

    size_t n_regions = MLIR_GetOpNumRegions(F->ctx, op);
    if (n_regions < 1) return false;
    size_t n_cases = n_regions - 1;  // region 0 = default

    // The "cases" attribute: a dense i64 array.
    int64_t *case_vals = NULL;
    if (n_cases > 0) {
        MLIR_AttributeHandle ca = find_attr(F->ctx, op, MLIR_KEY_CASES);
        if (ca == MLIR_INVALID_HANDLE) return false;
        case_vals = (int64_t *)arena_alloc(F->arena, n_cases * sizeof(int64_t));
        size_t n_parsed = MLIR_GetAttributeDenseSize(F->ctx, ca);
        if (n_parsed > n_cases) n_parsed = n_cases;
        for (size_t k = 0; k < n_parsed; k++)
            case_vals[k] = MLIR_GetAttributeDenseElement(F->ctx, ca, k);
        if (n_parsed != n_cases) return false;
    }

    // Collect result types.
    size_t n_res = MLIR_GetOpNumResults(F->ctx, op);
    uint8_t *res_vts = NULL;
    if (n_res) {
        res_vts = (uint8_t *)arena_alloc(F->arena, n_res);
        for (size_t i = 0; i < n_res; i++) {
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, MLIR_GetOpResult(F->ctx, op, i)));
            if (vt == 0) return false;
            res_vts[i] = vt;
        }
    }

    // Degenerate: no cases. Wrap default in a wasmssa.block.
    if (n_cases == 0) {
        MLIR_BlockHandle saved = F->body_block;
        MLIR_BlockHandle inner = MLIR_CreateBlock(F->ctx);
        F->body_block = inner;
        bool ok = lower_block(F, MLIR_GetRegionBlock(F->ctx, MLIR_GetOpRegion(F->ctx, op, 0), 0));
        F->body_block = saved;
        if (!ok) return false;

        MLIR_RegionHandle reg = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, reg, inner);
        MLIR_NamedAttr as[1];
        size_t nas = 0;
        if (n_res) as[nas++] = attr_s_hex(F->ctx, F->arena, MLIR_KEY_RESULT_TYPES, res_vts, n_res);
        MLIR_ValueHandle res_inline[16];
        MLIR_ValueHandle *res_vals = n_res <= 16 ? res_inline
            : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
        MLIR_OpHandle bop = make_op_n(F->ctx, OP_TYPE_WASMSSA_BLOCK, as, nas,
                                      NULL, 0, &reg, 1, res_vts, n_res, res_vals);
        MLIR_AppendBlockOp(F->ctx, F->body_block, bop);
        for (size_t i = 0; i < n_res; i++) {
            vmap_set(F, MLIR_GetOpResult(F->ctx, op, i), res_vals[i]);
        }
        return true;
    }

    MLIR_ValueHandle outer_inline[16];
    MLIR_ValueHandle *res_vals = n_res <= 16 ? outer_inline
        : (MLIR_ValueHandle *)arena_alloc(F->arena, n_res * sizeof(MLIR_ValueHandle));
    if (!isw_build_chain(F, op, 0, n_cases, case_vals, cond_idx,
                         n_res, res_vts, res_vals)) return false;
    for (size_t i = 0; i < n_res; i++) {
        vmap_set(F, MLIR_GetOpResult(F->ctx, op, i), res_vals[i]);
    }
    return true;
}

// The wasm opcode of an LLVM binop whose result has valtype `vt`, or 0 if
// there is none. Integer ops have an i32 and an i64 form, float ops an f32
// and an f64 form.
static uint8_t binop_opcode(MLIR_OpType ty, uint8_t vt) {
    uint8_t op32, op64;
    bool fp = false;
    switch (ty) {
    case OP_TYPE_LLVM_ADD:  op32 = 0x6a; op64 = 0x7c; break;
    case OP_TYPE_LLVM_SUB:  op32 = 0x6b; op64 = 0x7d; break;
    case OP_TYPE_LLVM_MUL:  op32 = 0x6c; op64 = 0x7e; break;
    case OP_TYPE_LLVM_SDIV: op32 = 0x6d; op64 = 0x7f; break;
    case OP_TYPE_LLVM_UDIV: op32 = 0x6e; op64 = 0x80; break;
    case OP_TYPE_LLVM_SREM: op32 = 0x6f; op64 = 0x81; break;
    case OP_TYPE_LLVM_UREM: op32 = 0x70; op64 = 0x82; break;
    case OP_TYPE_LLVM_AND:  op32 = 0x71; op64 = 0x83; break;
    case OP_TYPE_LLVM_OR:   op32 = 0x72; op64 = 0x84; break;
    case OP_TYPE_LLVM_XOR:  op32 = 0x73; op64 = 0x85; break;
    case OP_TYPE_LLVM_SHL:  op32 = 0x74; op64 = 0x86; break;
    case OP_TYPE_LLVM_ASHR: op32 = 0x75; op64 = 0x87; break;
    case OP_TYPE_LLVM_LSHR: op32 = 0x76; op64 = 0x88; break;
    case OP_TYPE_LLVM_FADD: op32 = 0x92; op64 = 0xa0; fp = true; break;
    case OP_TYPE_LLVM_FSUB: op32 = 0x93; op64 = 0xa1; fp = true; break;
    case OP_TYPE_LLVM_FMUL: op32 = 0x94; op64 = 0xa2; fp = true; break;
    case OP_TYPE_LLVM_FDIV: op32 = 0x95; op64 = 0xa3; fp = true; break;
    default: return 0;
    }
    if (vt == (fp ? WT_F32 : WT_I32)) return op32;
    if (vt == (fp ? WT_F64 : WT_I64)) return op64;
    return 0;
}

static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op);
static bool lower_op(FnCtx *F, MLIR_OpHandle op) {
    bool ok = lower_op_inner(F, op);
    if (!ok) {
        string n = MLIR_GetOpTypeName(F->ctx, MLIR_GetOpType(F->ctx, op));
        fprintf(stderr, "wasmssa-lower: lower_op failed at '%.*s'\n",
                (int)n.size, n.str);
    }
    return ok;
}
static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op) {
    // One switch on the op type selects the lowering. Every arm returns.
    MLIR_OpType ty = MLIR_GetOpType(F->ctx, op);
    switch (ty) {

    // ---- scf.if / scf.while / scf.index_switch ---------------------------
    case OP_TYPE_SCF_IF: return lower_scf_if(F, op);
    case OP_TYPE_SCF_WHILE: return lower_scf_while(F, op);
    case OP_TYPE_SCF_INDEX_SWITCH: return lower_scf_index_switch(F, op);

    // ---- scf.yield: emit wasmssa.block_return into the enclosing region's
    //                 block. The enclosing scf.if / scf.index_switch / scf.while
    //                 lowering has already set F->body_block to the right block.
    case OP_TYPE_SCF_YIELD: {
        size_t no = MLIR_GetOpNumOperands(F->ctx, op);
        MLIR_ValueHandle vs_inline[16];
        MLIR_ValueHandle *vs = no <= 16 ? vs_inline
            : (MLIR_ValueHandle *)arena_alloc(F->arena, no * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < no; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, i), &vs[i])) return false;
        }
        emit_block_return(F, vs, no);
        return true;
    }

    // ---- llvm.select -----------------------------------------------------
    case OP_TYPE_LLVM_SELECT: {
        if (MLIR_GetOpNumOperands(F->ctx, op) != 3 || MLIR_GetOpNumResults(F->ctx, op) != 1)
            return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (vt == 0) return false;
        MLIR_ValueHandle ci, ai, bi;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &ci)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 1), &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 2), &bi)) return false;
        MLIR_ValueHandle ops[3] = { ai, bi, ci };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_SELECT;
        o.valtype = vt;
        o.n_operands = 3;
        // wasm select pops cond,b,a so we order operands as a,b,cond.
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.mlir.constant ------------------------------------------------
    case OP_TYPE_LLVM_MLIR_CONSTANT: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (vt == 0) return false;
        MLIR_AttributeHandle va = find_attr(F->ctx, op, MLIR_KEY_VALUE);
        if (va == MLIR_INVALID_HANDLE) return false;
        MLIR_AttrKind ak = MLIR_GetAttributeKind(F->ctx, va);

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        if (ak == MLIR_ATTR_KIND_INTEGER) {
            o.i_const = MLIR_GetAttributeInteger(F->ctx, va);
            // An i8 / i16 constant is materialized zero-extended, whether the
            // API returns it sign- or zero-extended (upstream: sign).
            int cw = int_bits(F->ctx, MLIR_GetValueType(F->ctx, r));
            if (cw == 8 || cw == 16) o.i_const &= ((int64_t)1 << cw) - 1;
        } else if (ak == MLIR_ATTR_KIND_BOOL) {
            o.i_const = MLIR_GetAttributeBool(F->ctx, va) ? 1 : 0;
        } else if (ak == MLIR_ATTR_KIND_FLOAT) {
            double dv = MLIR_GetAttributeFloat(F->ctx, va);
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
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.mlir.undef ----------------------------------------------------
    // Inserted by the structurization step for "doesn't-matter" yield values
    // on unreachable / pre-break paths. Lower it to a typed zero constant.
    case OP_TYPE_LLVM_MLIR_UNDEF: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (vt == 0) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        o.i_const = 0;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.alloca: result = sp_def + frame_offset -----------------------
    case OP_TYPE_LLVM_ALLOCA: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint32_t off;
        if (!amap_get(F, r, &off)) return false;
        if (F->sp_value == MLIR_INVALID_HANDLE) return false;
        MLIR_ValueHandle koff = emit_const_i32(F, (int32_t)off);
        MLIR_ValueHandle addr = emit_add_i32(F, F->sp_value, koff);
        vmap_set(F, r, addr);
        return true;
    }

    // ---- llvm.store --------------------------------------------------------
    case OP_TYPE_LLVM_STORE: {
        if (MLIR_GetOpNumOperands(F->ctx, op) != 2) return false;
        MLIR_ValueHandle val = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(F->ctx, op, 1);
        MLIR_TypeHandle  vt_ty = MLIR_GetValueType(F->ctx, val);
        uint8_t vt = wasm_vt(F->ctx, vt_ty);
        unsigned sz = type_size_bytes(F->ctx, vt_ty);
        if (vt == 0 || sz == 0) return false;
        MLIR_ValueHandle va, pa;
        if (!vmap_get(F, val, &va)) return false;
        if (!vmap_get(F, ptr, &pa)) return false;

        unsigned align_log2;
        if (vt == WT_I32) {
            align_log2 = (sz == 4) ? 2 : (sz == 2) ? 1 : 0;
        } else if (vt == WT_I64) align_log2 = 3;
        else if (vt == WT_F32)   align_log2 = 2;
        else if (vt == WT_F64)   align_log2 = 3;
        else return false;

        MLIR_ValueHandle ops[2] = { pa, va };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_STORE;
        o.valtype = vt;
        o.mem_size_bytes = sz;
        o.memory_align_log2 = align_log2;
        o.memory_offset = 0;
        o.n_operands = 2;
        o.operands = ops;
        (void)commit_op(F, &o);
        return true;
    }

    // ---- llvm.load ---------------------------------------------------------
    case OP_TYPE_LLVM_LOAD: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_TypeHandle  rt = MLIR_GetValueType(F->ctx, r);
        uint8_t vt = wasm_vt(F->ctx, rt);
        unsigned sz = type_size_bytes(F->ctx, rt);
        if (vt == 0 || sz == 0) return false;
        MLIR_ValueHandle pa;
        if (!vmap_get(F, ptr, &pa)) return false;

        unsigned align_log2;
        if (vt == WT_I32) {
            align_log2 = (sz == 4) ? 2 : (sz == 2) ? 1 : 0;
        } else if (vt == WT_I64) align_log2 = 3;
        else if (vt == WT_F32)   align_log2 = 2;
        else if (vt == WT_F64)   align_log2 = 3;
        else return false;

        MLIR_ValueHandle ops[1] = { pa };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_LOAD;
        o.valtype = vt;
        o.mem_size_bytes = sz;
        o.memory_align_log2 = align_log2;
        o.memory_offset = 0;
        o.n_operands = 1;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.sext (i8/i16/i32 -> i32; i32 -> i64; i8/i16 -> i64) ----------
    case OP_TYPE_LLVM_SEXT: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(F->ctx, op, 0);
        int in_w  = int_bits(F->ctx, MLIR_GetValueType(F->ctx, s));
        int out_w = int_bits(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (in_w == 0 || out_w == 0 || in_w > out_w) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;

        // Step 1: if narrowing source is i8/i16 stored in i32, sign-extend
        // within i32 first (extend8_s=0xc0, extend16_s=0xc1).
        MLIR_ValueHandle cur_idx = sa;
        uint8_t cur_vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, s));
        if ((in_w == 8 || in_w == 16) && !known_sext(F->ctx, s, in_w, 0)) {
            MLIR_ValueHandle ops[1] = { cur_idx };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_UNOP;
            o.valtype = WT_I32;
            o.wasm_opcode = (in_w == 8) ? 0xc0 : 0xc1;
            o.n_operands = 1;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            cur_idx = idx; cur_vt = WT_I32;
        }
        // Step 2: if widening into i64, use i64.extend_i32_s (0xac).
        if (out_w == 64) {
            MLIR_ValueHandle ops[1] = { cur_idx };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_EXTEND_I32_S;
            o.valtype = WT_I64;
            o.n_operands = 1;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            cur_idx = idx;
        }
        vmap_set(F, r, cur_idx);
        return true;
    }

    // ---- llvm.ptrtoint / llvm.inttoptr -------------------------------------
    case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_INTTOPTR: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(F->ctx, op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        // Pointers are i32. If matched, identity; if narrowing/widening, defer
        // to the existing trunc/zext-style lowering by emitting a wrap or
        // extend.
        if (in_vt == out_vt) {
            vmap_set(F, r, sa);
            return true;
        }
        if (in_vt == WT_I64 && out_vt == WT_I32) {
            MLIR_ValueHandle idx = emit_wrap_i64_to_i32(F, sa);
            vmap_set(F, r, idx);
            return true;
        }
        if (in_vt == WT_I32 && out_vt == WT_I64) {
            // unsigned extension: i64.extend_i32_u (0xad)
            MLIR_ValueHandle ops[1] = { sa };
            wasmssa_op_t o = {0};
            o.type = OP_TYPE_WASMSSA_UNOP;
            o.valtype = WT_I64;
            o.wasm_opcode = 0xad;
            o.n_operands = 1;
            o.operands = ops;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, r, idx);
            return true;
        }
        return false;
    }

    // ---- llvm float compares -----------------------------------------------
    case OP_TYPE_LLVM_FCMP: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 2) return false;
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 0);
        uint8_t opvt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, a));
        // FCmpPredicate enum: oeq=1, ogt=2, oge=3, olt=4, ole=5, one=6,
        // une=11, ueq=8 (unordered variants); we only support ordered.
        // wasm has eq, ne, lt, gt, le, ge (ordered). Map oeq->eq, etc.
        MLIR_AttributeHandle pa = find_attr(F->ctx, op, MLIR_KEY_PREDICATE);
        if (pa == MLIR_INVALID_HANDLE) return false;
        int64_t pred = MLIR_GetAttributeInteger(F->ctx, pa);
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
        MLIR_ValueHandle ai, bi;
        if (!vmap_get(F, a, &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 1), &bi)) return false;
        MLIR_ValueHandle ops[2] = { ai, bi };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, MLIR_GetOpResult(F->ctx, op, 0), idx);
        return true;
    }

    // ---- llvm.fpext / llvm.fptrunc / llvm.fneg ----------------------------
    case OP_TYPE_LLVM_FPEXT: case OP_TYPE_LLVM_FPTRUNC: case OP_TYPE_LLVM_FNEG: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(F->ctx, op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        uint8_t opc = 0;
        if (ty == OP_TYPE_LLVM_FPEXT && in_vt == WT_F32 && out_vt == WT_F64) opc = 0xbb;
        else if (ty == OP_TYPE_LLVM_FPTRUNC && in_vt == WT_F64 && out_vt == WT_F32) opc = 0xb6;
        else if (ty == OP_TYPE_LLVM_FNEG && in_vt == WT_F32 && out_vt == WT_F32) opc = 0x8c;
        else if (ty == OP_TYPE_LLVM_FNEG && in_vt == WT_F64 && out_vt == WT_F64) opc = 0x9a;
        else return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.sitofp / llvm.uitofp / llvm.fptosi / llvm.fptoui ------------
    case OP_TYPE_LLVM_SITOFP: case OP_TYPE_LLVM_UITOFP: case OP_TYPE_LLVM_FPTOSI: case OP_TYPE_LLVM_FPTOUI: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(F->ctx, op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        bool is_signed = ty == OP_TYPE_LLVM_SITOFP || ty == OP_TYPE_LLVM_FPTOSI;
        uint8_t opc = 0;
        if (ty == OP_TYPE_LLVM_SITOFP || ty == OP_TYPE_LLVM_UITOFP) {
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
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = out_vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.icmp ---------------------------------------------------------
    // Result is i1 (encoded as i32 in wasm). Operand valtype determines
    // the i32.* vs i64.* opcode family.
    case OP_TYPE_LLVM_ICMP: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 2) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(F->ctx, op, 1);
        uint8_t opvt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, a));
        // LLVM ICmpPredicate enum: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5,
        // ult=6, ule=7, ugt=8, uge=9.
        MLIR_AttributeHandle pa = find_attr(F->ctx, op, MLIR_KEY_PREDICATE);
        if (pa == MLIR_INVALID_HANDLE) return false;
        int64_t pred = MLIR_GetAttributeInteger(F->ctx, pa);
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
        MLIR_ValueHandle ai, bi;
        if (!vmap_get(F, a, &ai) || !vmap_get(F, b, &bi)) return false;
        MLIR_ValueHandle ops[2] = { ai, bi };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;  // result type
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.trunc / llvm.zext -------------------------------------------
    // i64 -> i32 trunc:    i32.wrap_i64    (0xa7)
    // i32 -> i64 zext:     i64.extend_i32_u(0xad)
    // i8  -> i32 zext:     i32.and %v, 255 (wasm has no unsigned extend8)
    // i32 -> i8 trunc:     no-op on the i32 register (low byte preserved)
    // smaller-int trunc/zext within i32: no-op (wasm has no sub-i32 reg).
    case OP_TYPE_LLVM_TRUNC: case OP_TYPE_LLVM_ZEXT: {
        bool is_zext = ty == OP_TYPE_LLVM_ZEXT;
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(F->ctx, op, 0);
        int in_w  = int_bits(F->ctx, MLIR_GetValueType(F->ctx, s));
        int out_w = int_bits(F->ctx, MLIR_GetValueType(F->ctx, r));
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (in_vt == 0 || out_vt == 0) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, s, &sa)) return false;
        // An i8 / i16 value is held in an i32 whose upper bits are
        // unspecified (a trunc is a no-op): zext masks them off, unless they
        // are known to be clear (e.g. i32.load8_u / load16_u).
        if (is_zext && (in_w == 8 || in_w == 16)) {
            MLIR_ValueHandle idx = sa;
            if (!known_zext(F->ctx, s, in_w, 0)) {
                MLIR_ValueHandle mask = emit_const_i32(F, in_w == 8 ? 0xff : 0xffff);
                idx = emit_and_i32(F, sa, mask);
            }
            if (out_w == 64) {
                wasmssa_op_t o = {0};
                o.type = OP_TYPE_WASMSSA_UNOP;
                o.valtype = WT_I64;
                o.wasm_opcode = 0xad;              // i64.extend_i32_u
                o.n_operands = 1;
                MLIR_ValueHandle o_ops[1] = { idx };
                o.operands = o_ops;
                o.has_result = true;
                idx = commit_op(F, &o);
            }
            vmap_set(F, r, idx);
            return true;
        }
        if (!is_zext && in_w == 32 && out_w == 8) {
            vmap_set(F, r, sa);
            return true;
        }
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
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.return -------------------------------------------------------
    case OP_TYPE_LLVM_RETURN: {
        size_t no = MLIR_GetOpNumOperands(F->ctx, op);
        MLIR_ValueHandle retv = MLIR_INVALID_HANDLE;
        if (no == 1) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(F->ctx, op, 0);
            if (!vmap_get(F, v, &retv)) return false;
        } else if (no != 0) {
            return false;
        }

        // Epilogue: restore __stack_pointer = sp_def + frame_size.
        if (F->frame_size > 0 && F->sp_value != MLIR_INVALID_HANDLE) {
            MLIR_ValueHandle kf = emit_const_i32(F, (int32_t)F->frame_size);
            MLIR_ValueHandle restored = emit_add_i32(F, F->sp_value, kf);
            emit_global_set(F, /*sp*/0, restored);
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_RETURN;
        MLIR_ValueHandle o_ops[1] = { retv };
        if (no == 1) {
            o.n_operands = 1;
            o.operands = o_ops;
        }
        (void)commit_op(F, &o);
        return true;
    }

    // ---- llvm.call (direct or indirect) ------------------------------------
    case OP_TYPE_LLVM_CALL: {
        MLIR_AttributeHandle callee = MLIR_GetOpAttr(F->ctx, op, MLIR_KEY_CALLEE);
        MLIR_AttributeHandle var_callee_type = MLIR_GetOpAttr(F->ctx, op, MLIR_KEY_VAR_CALLEE_TYPE);

        // Indirect: no callee attr, operand[0] is the function pointer
        // (a !llvm.ptr SSA value), and var_callee_type carries the signature.
        if (callee == MLIR_INVALID_HANDLE) {
            // Indirect call: operand[0] is function pointer (!llvm.ptr).
            // Build the signature from operand types and result types,
            // since the upstream wasm pipeline doesn't carry var_callee_type
            // through after lowering.
            size_t no = MLIR_GetOpNumOperands(F->ctx, op);
            if (no < 1) return false;
            size_t snp = no - 1;
            uint8_t *sp = (uint8_t *)arena_alloc(F->arena, snp ? snp : 1);
            for (size_t i = 0; i < snp; i++) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(F->ctx, MLIR_GetOpOperand(F->ctx, op, i + 1)));
                if (v == 0) return false;
                sp[i] = v;
            }
            size_t nr = MLIR_GetOpNumResults(F->ctx, op);
            if (nr > 1) return false;
            uint8_t *sr = (uint8_t *)arena_alloc(F->arena, 1);
            size_t snr = 0;
            if (nr == 1) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(F->ctx, MLIR_GetOpResult(F->ctx, op, 0)));
                if (v == 0) return false;
                sr[0] = v;
                snr = 1;
            }
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, no * sizeof(MLIR_ValueHandle));
            // wasm call_indirect pops args first then table-index, so order
            // them in operand[] as [args..., funcptr].
            for (size_t i = 0; i < snp; i++) {
                if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, i + 1), &opnds[i]))
                    return false;
            }
            if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &opnds[snp]))
                return false;
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
            MLIR_ValueHandle idx = commit_op(F, &o);
            if (snr == 1) {
                vmap_set(F, MLIR_GetOpResult(F->ctx, op, 0), idx);
            }
            return true;
        }

        if (var_callee_type != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(F->ctx, var_callee_type);
            if (MLIR_GetTypeFunctionIsVarArg(F->ctx, fty)) {
                // Variadic direct call: pack variadic args into the
                // function's variadic buffer (allocated in the shadow
                // stack frame) and pass its address as a hidden trailing
                // i32 arg in place of the `...` operands.
                size_t nfixed = MLIR_GetTypeFunctionNumInputs(F->ctx, fty);
                size_t no = MLIR_GetOpNumOperands(F->ctx, op);
                size_t nvar = no - nfixed;
                uint32_t total = 0; size_t nf2 = 0;
                uint32_t *offs = (uint32_t *)arena_alloc(F->arena, (nvar ? nvar : 1) * sizeof(uint32_t));
                if (!va_call_layout(F, op, &total, &nf2, offs)) {
                    return false;
                }
                // buf_addr = sp_def + va_buf_offset. When this call passes no
                // variadic args (total == 0) the buffer is empty and never
                // dereferenced, so it needs no shadow-stack frame — pass a null
                // pointer as the hidden va_list arg. (Without this, a variadic
                // call with zero variadic args in a function that has no other
                // stack frame — e.g. after mem2reg promotes all locals — would
                // fail for lack of sp_value.)
                MLIR_ValueHandle buf_addr;
                if (total == 0) {
                    buf_addr = emit_const_i32(F, 0);
                } else {
                    if (F->sp_value == MLIR_INVALID_HANDLE) { return false; }
                    if (F->va_buf_offset == 0) {
                        buf_addr = F->sp_value;
                    } else {
                        MLIR_ValueHandle koff = emit_const_i32(F, (int32_t)F->va_buf_offset);
                        buf_addr = emit_add_i32(F, F->sp_value, koff);
                    }
                }
                // Emit a store for each variadic arg, then the call.
                for (size_t i = 0; i < nvar; i++) {
                    MLIR_ValueHandle av = MLIR_GetOpOperand(F->ctx, op, nfixed + i);
                    MLIR_TypeHandle aty = MLIR_GetValueType(F->ctx, av);
                    uint8_t vt = wasm_vt(F->ctx, aty);
                    MLIR_ValueHandle va;
                    if (!vmap_get(F, av, &va)) { return false; }
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
                        MLIR_ValueHandle po_ops[1] = { va };
                        po.operands = po_ops;
                        po.has_result = true;
                        MLIR_ValueHandle pidx = commit_op(F, &po);
                        va = pidx;
                        st_vt = WT_F64;
                        sz = 8; align_log2 = 3;
                    }
                    else if (vt == WT_F64) { sz = 8; align_log2 = 3; }
                    else { return false; }
                    wasmssa_op_t o2 = {0};
                    o2.type = OP_TYPE_WASMSSA_STORE;
                    o2.valtype = st_vt;
                    o2.mem_size_bytes = sz;
                    o2.memory_align_log2 = align_log2;
                    o2.memory_offset = offs[i];
                    o2.n_operands = 2;
                    MLIR_ValueHandle o2_ops[2] = { buf_addr, va };
                    o2.operands = o2_ops;
                    (void)commit_op(F, &o2);
                }

                // Now emit the call with fixed args + buf_addr.
                string nm = MLIR_GetAttributeAsString(F->ctx, callee);
                string cstr = nm;
                if (cstr.size > 0 && cstr.str[0] == '@') { cstr.str++; cstr.size--; }
                size_t nout = nfixed + 1;
                MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, nout * sizeof(MLIR_ValueHandle));
                for (size_t i = 0; i < nfixed; i++) {
                    if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, i), &opnds[i]))
                        return false;
                }
                opnds[nfixed] = buf_addr;

                wasmssa_op_t oc = {0};
                oc.type = OP_TYPE_WASMSSA_CALL;
                oc.call_target = cstr;
                oc.n_operands = (int)nout;
                oc.operands = opnds;
                size_t nr = MLIR_GetOpNumResults(F->ctx, op);
                MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
                if (nr == 1) {
                    r = MLIR_GetOpResult(F->ctx, op, 0);
                    uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
                    if (vt == 0) return false;
                    oc.valtype = vt;
                    oc.has_result = true;
                } else if (nr != 0) {
                    return false;
                }
                MLIR_ValueHandle idx = commit_op(F, &oc);
                if (nr == 1) vmap_set(F, r, idx);
                return true;
            }
        }
        string nm = MLIR_GetAttributeAsString(F->ctx, callee);
        // SymbolRefAttr prints as `@name`.
        string cstr = nm;
        if (cstr.size > 0 && cstr.str[0] == '@') { cstr.str++; cstr.size--; }

        size_t no = MLIR_GetOpNumOperands(F->ctx, op);
        MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, (no ? no : 1) * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < no; i++) {
            if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, i), &opnds[i]))
                return false;
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CALL;
        o.call_target = cstr;
        o.n_operands = (int)no;
        o.operands = opnds;

        size_t nr = MLIR_GetOpNumResults(F->ctx, op);
        MLIR_ValueHandle r = MLIR_INVALID_HANDLE;
        if (nr == 1) {
            r = MLIR_GetOpResult(F->ctx, op, 0);
            uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
            if (vt == 0) return false;
            o.valtype = vt;
            o.has_result = true;
        } else if (nr != 0) {
            return false;
        }
        MLIR_ValueHandle idx = commit_op(F, &o);
        if (nr == 1) vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.getelementptr -----------------------------------------------
    case OP_TYPE_LLVM_GEP: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        if (MLIR_GetOpNumOperands(F->ctx, op) < 1) return false;
        MLIR_ValueHandle r    = MLIR_GetOpResult(F->ctx, op, 0);
        MLIR_ValueHandle base = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_ValueHandle addr;
        if (!vmap_get(F, base, &addr)) return false;
        MLIR_AttributeHandle eta = find_attr(F->ctx, op, MLIR_KEY_ELEM_TYPE);
        if (eta == MLIR_INVALID_HANDLE) return false;
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(F->ctx, eta);
        MLIR_AttributeHandle ria = find_attr(F->ctx, op, MLIR_KEY_RAW_CONSTANT_INDICES);
        if (ria == MLIR_INVALID_HANDLE) return false;
        size_t n_idx = 0;
        int32_t *cidx = parse_dense_i32_array(F->ctx, F->arena, ria, &n_idx);
        if (!cidx || n_idx == 0) return false;

        size_t op_idx = 1;
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            MLIR_ValueHandle dyn_def = MLIR_INVALID_HANDLE;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(F->ctx, op)) return false;
                MLIR_ValueHandle ov = MLIR_GetOpOperand(F->ctx, op, op_idx++);
                if (!vmap_get(F, ov, &dyn_def)) return false;
                uint8_t ovt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, ov));
                if (ovt == WT_I64) dyn_def = emit_wrap_i64_to_i32(F, dyn_def);
                else if (ovt != WT_I32) return false;
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
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)esz);
                    MLIR_ValueHandle m = emit_mul_i32(F, dyn_def, k);
                    addr = emit_add_i32(F, addr, m);
                } else if (cidx[i] != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)((int64_t)cidx[i] * (int32_t)esz));
                    addr = emit_add_i32(F, addr, k);
                }
            } else if (MLIR_GetTypeKind(F->ctx, cur_ty) == MLIR_TYPE_KIND_LLVM_STRUCT) {
                if (is_dyn) return false;
                unsigned off = struct_field_offset(F->ctx, cur_ty, (size_t)cidx[i]);
                if (off != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)off);
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = MLIR_GetTypeLLVMStructField(F->ctx, cur_ty, (size_t)cidx[i]);
            } else if (MLIR_GetTypeKind(F->ctx, cur_ty) == MLIR_TYPE_KIND_LLVM_ARRAY) {
                MLIR_TypeHandle et = MLIR_GetTypeElement(F->ctx, cur_ty);
                unsigned esz = type_size_bytes(F->ctx, et);
                if (esz == 0) return false;
                if (is_dyn) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)esz);
                    MLIR_ValueHandle m = emit_mul_i32(F, dyn_def, k);
                    addr = emit_add_i32(F, addr, m);
                } else if (cidx[i] != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)((int64_t)cidx[i] * (int32_t)esz));
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = et;
            } else {
                return false;
            }
        }
        vmap_set(F, r, addr);
        return true;
    }

    // ---- llvm.mlir.addressof ----------------------------------------------
    // Result is a pointer (i32) to a data global. We emit an ADDRESSOF op
    // that stage 3 will lower as `i32.const <padded sleb>` plus a
    // R_WASM_MEMORY_ADDR_SLEB reloc to the named global symbol.
    case OP_TYPE_LLVM_MLIR_ADDRESSOF: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_AttributeHandle ta = find_attr(F->ctx, op, MLIR_KEY_GLOBAL_NAME);
        if (ta == MLIR_INVALID_HANDLE)
            ta = find_attr(F->ctx, op, MLIR_KEY_VALUE);
        if (ta == MLIR_INVALID_HANDLE) {
            // Fallback: any string attr.
            size_t na = MLIR_GetOpNumAttributes(F->ctx, op);
            for (size_t i = 0; i < na; i++) {
                MLIR_AttributeHandle a = MLIR_GetOpAttribute(F->ctx, op, i);
                if (MLIR_GetAttributeKind(F->ctx, a) == MLIR_ATTR_KIND_STRING) {
                    ta = a; break;
                }
            }
        }
        if (ta == MLIR_INVALID_HANDLE) return false;
        string ts = MLIR_GetAttributeString(F->ctx, ta);
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
            o.call_target = ts;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, MLIR_GetOpResult(F->ctx, op, 0), idx);
            return true;
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_ADDRESSOF;
        o.valtype = WT_I32;
        o.call_target = ts;  // reuse field for symbol name
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, MLIR_GetOpResult(F->ctx, op, 0), idx);
        return true;
    }

    // ---- llvm.mlir.zero ---------------------------------------------------
    // Pointer / integer zero. f32/f64 zero would also fall through here but
    // currently only ptr/int paths are exercised (the float test uses a
    // typed llvm.mlir.constant for 0.0).
    case OP_TYPE_LLVM_MLIR_ZERO: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (vt == 0) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_CONST;
        o.valtype = vt;
        o.i_const = 0;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm binary integer / float ops -----------------------------------
    // Mapped to a generic wasmssa.binop carrying the actual wasm bytecode
    // byte (binop_opcode).
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_SDIV: case OP_TYPE_LLVM_UDIV:
    case OP_TYPE_LLVM_SREM: case OP_TYPE_LLVM_UREM:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR: case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_ASHR: case OP_TYPE_LLVM_LSHR:
    case OP_TYPE_LLVM_FADD: case OP_TYPE_LLVM_FSUB:
    case OP_TYPE_LLVM_FMUL: case OP_TYPE_LLVM_FDIV: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 2) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        uint8_t opc = binop_opcode(ty, vt);
        if (opc == 0) return false;
        MLIR_ValueHandle ai, bi;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &ai)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 1), &bi)) return false;
        MLIR_ValueHandle ops[2] = { ai, bi };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = vt;
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.vastart / llvm.intr.vaend / llvm.intr.vacopy ------------
    case OP_TYPE_LLVM_INTR_VASTART: {
        // Store the function's hidden va_list i32 param into *%ap.
        if (F->va_list_value == MLIR_INVALID_HANDLE) return false;
        if (MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle pa;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &pa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_STORE;
        o.valtype = WT_I32;
        o.mem_size_bytes = 4;
        o.memory_align_log2 = 2;
        o.memory_offset = 0;
        o.n_operands = 2;
        MLIR_ValueHandle o_ops[2] = { pa, F->va_list_value };
        o.operands = o_ops;
        (void)commit_op(F, &o);
        return true;
    }
    case OP_TYPE_LLVM_INTR_VAEND: {
        // No-op under the wasm32 ABI.
        return true;
    }
    case OP_TYPE_LLVM_INTR_VACOPY: {
        // Copy 4 bytes (a char*) from src to dst.
        if (MLIR_GetOpNumOperands(F->ctx, op) != 2) return false;
        MLIR_ValueHandle da, sa;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &da)) return false;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 1), &sa)) return false;
        wasmssa_op_t ol = {0};
        ol.type = OP_TYPE_WASMSSA_LOAD;
        ol.valtype = WT_I32;
        ol.mem_size_bytes = 4;
        ol.memory_align_log2 = 2;
        ol.memory_offset = 0;
        ol.n_operands = 1;
        MLIR_ValueHandle ol_ops[1] = { sa };
        ol.operands = ol_ops;
        ol.has_result = true;
        MLIR_ValueHandle lidx = commit_op(F, &ol);
        wasmssa_op_t os = {0};
        os.type = OP_TYPE_WASMSSA_STORE;
        os.valtype = WT_I32;
        os.mem_size_bytes = 4;
        os.memory_align_log2 = 2;
        os.memory_offset = 0;
        os.n_operands = 2;
        MLIR_ValueHandle os_ops[2] = { da, lidx };
        os.operands = os_ops;
        (void)commit_op(F, &os);
        return true;
    }

    // ---- llvm.intr.sqrt ---------------------------------------------------
    case OP_TYPE_LLVM_INTR_SQRT: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        uint8_t opc = (vt == WT_F32) ? 0x91 : (vt == WT_F64) ? 0x9f : 0;
        if (opc == 0) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_UNOP;
        o.valtype = vt;
        o.wasm_opcode = opc;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.wasm.memory.size ---------------------------------------
    // `__builtin_wasm_memory_size(0)` lowers to this. No operands (the
    // memory index byte is hard-coded to 0 in the binary encoder); the
    // single i32 result is the current memory size in 64KiB pages.
    case OP_TYPE_LLVM_INTR_WASM_MEMORY_SIZE: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (vt != WT_I32) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_MEMORY_SIZE;
        o.valtype = vt;
        o.n_operands = 0;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.wasm.memory.grow ---------------------------------------
    // `__builtin_wasm_memory_grow(0, n)` lowers to this. One i32 operand
    // (pages to grow by) and one i32 result (previous size in pages or -1).
    case OP_TYPE_LLVM_INTR_WASM_MEMORY_GROW: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1 ||
            MLIR_GetOpNumOperands(F->ctx, op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(F->ctx, r));
        if (vt != WT_I32) return false;
        MLIR_ValueHandle sa;
        if (!vmap_get(F, MLIR_GetOpOperand(F->ctx, op, 0), &sa)) return false;
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_MEMORY_GROW;
        o.valtype = vt;
        o.n_operands = 1;
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    default: break;
    }

    string name = MLIR_GetOpTypeName(F->ctx, ty);
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
    MLIR_AttributeHandle vct = MLIR_GetOpAttr(F->ctx, op, MLIR_KEY_VAR_CALLEE_TYPE);
    if (vct == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(F->ctx, vct);
    if (!MLIR_GetTypeFunctionIsVarArg(F->ctx, fty)) return false;
    size_t nf = MLIR_GetTypeFunctionNumInputs(F->ctx, fty);
    size_t no = MLIR_GetOpNumOperands(F->ctx, op);
    if (no < nf) return false;
    uint32_t cur = 0;
    for (size_t i = nf; i < no; i++) {
        MLIR_TypeHandle aty = MLIR_GetValueType(F->ctx, MLIR_GetOpOperand(F->ctx, op, i));
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
        cur = llvm_align_up(cur, al);
        if (out_offsets) out_offsets[i - nf] = cur;
        cur += sz;
        (void)sz;
    }
    *total_size = cur;
    *n_fixed = nf;
    return true;
}

static bool prewalk_op(FnCtx *F, MLIR_OpHandle op) {
    MLIR_OpType ty = MLIR_GetOpType(F->ctx, op);
    if (ty == OP_TYPE_LLVM_BR || ty == OP_TYPE_LLVM_COND_BR ||
        ty == OP_TYPE_LLVM_SWITCH) {
        string n = MLIR_GetOpTypeName(F->ctx, ty);
        // The structurization step could not lift this branch (see the
        // limitations listed in mlir_lift_cf_to_scf.h).
        fprintf(stderr,
                "wasmssa-lower: control flow could not be structured "
                "(%.*s left in the CFG)\n",
                (int)n.size, n.str);
        return false;
    }
    if (ty == OP_TYPE_LLVM_CALL) {
        uint32_t sz = 0; size_t nfixed = 0;
        if (va_call_layout(F, op, &sz, &nfixed, NULL)) {
            if (sz > F->va_buf_size) F->va_buf_size = sz;
        }
    }
    if (ty == OP_TYPE_LLVM_ALLOCA) {
        MLIR_TypeHandle et = MLIR_INVALID_HANDLE;
        MLIR_AttributeHandle ea = find_attr(F->ctx, op, MLIR_KEY_ELEM_TYPE);
        if (ea != MLIR_INVALID_HANDLE) et = MLIR_GetAttributeTypeValue(F->ctx, ea);
        unsigned esz = et != MLIR_INVALID_HANDLE
                           ? type_size_bytes(F->ctx, et) : 0;
        if (esz == 0) {
            fprintf(stderr,
                    "wasmssa-lower: alloca of unsupported element type\n");
            return false;
        }
        int64_t cnt = 1;
        if (MLIR_GetOpNumOperands(F->ctx, op) >= 1) {
            MLIR_ValueHandle co = MLIR_GetOpOperand(F->ctx, op, 0);
            MLIR_OpHandle    cd = MLIR_GetValueDefiningOp(F->ctx, co);
            if (cd == MLIR_INVALID_HANDLE) return false;
            if (MLIR_GetOpType(F->ctx, cd) != OP_TYPE_LLVM_MLIR_CONSTANT) {
                fprintf(stderr,
                        "wasmssa-lower: dynamic alloca size not supported\n");
                return false;
            }
            MLIR_AttributeHandle va = find_attr(F->ctx, cd, MLIR_KEY_VALUE);
            cnt = MLIR_GetAttributeInteger(F->ctx, va);
        }
        unsigned ealign = type_align_bytes(F->ctx, et);
        if (ealign < 4) ealign = 4;
        F->frame_size = llvm_align_up(F->frame_size, ealign);
        uint32_t off = F->frame_size;
        F->frame_size += (uint32_t)(esz * cnt);
        amap_set(F, MLIR_GetOpResult(F->ctx, op, 0), off);
    }
    // Recurse into nested regions for nested allocas.
    size_t nr = MLIR_GetOpNumRegions(F->ctx, op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle rg = MLIR_GetOpRegion(F->ctx, op, r);
        size_t nblk = MLIR_GetRegionNumBlocks(F->ctx, rg);
        for (size_t b = 0; b < nblk; b++) {
            if (!prewalk_block(F, MLIR_GetRegionBlock(F->ctx, rg, b))) return false;
        }
    }
    return true;
}

static bool prewalk_block(FnCtx *F, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(F->ctx, blk);
    for (size_t i = 0; i < nops; i++) {
        if (!prewalk_op(F, MLIR_GetBlockOp(F->ctx, blk, i))) return false;
    }
    return true;
}

static bool prewalk_func(FnCtx *F, MLIR_BlockHandle entry) {
    if (!prewalk_block(F, entry)) return false;
    if (F->va_buf_size > 0) {
        F->frame_size = llvm_align_up(F->frame_size, 8);
        F->va_buf_offset = F->frame_size;
        F->frame_size += F->va_buf_size;
    }
    F->frame_size = (F->frame_size + 15) & ~15u;
    return true;
}

static bool lower_block(FnCtx *F, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(F->ctx, blk);
    for (size_t i = 0; i < nops; i++) {
        if (!lower_op(F, MLIR_GetBlockOp(F->ctx, blk, i))) return false;
    }
    return true;
}

// Lower a defined `llvm.func` into a `wasmssa.func` op appended to the
// module body. The function's signature attrs are passed in (param_types
// includes the synthetic trailing i32 for vararg functions); the special
// "__original_main" rename + exported flag are picked by the caller.
static bool lower_function(MLIR_Context *ctx, Arena *arena, ModCtx *mod,
                           MLIR_BlockHandle mod_body,
                           string fn_name, bool exported,
                           MLIR_OpHandle fn,
                           const uint8_t *param_types, size_t n_params,
                           const uint8_t *result_types, size_t n_results) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx;
    F.arena = arena;
    F.mod = mod;
    F.n_params = n_params;
    F.sp_value = MLIR_INVALID_HANDLE;
    F.va_list_value = MLIR_INVALID_HANDLE;

    // If this function is variadic, sig_for_func has appended a synthetic
    // i32 va_list param at index n_params-1. The MLIR entry block only has
    // the original (non-hidden) parameters as block args.
    MLIR_AttributeHandle ftya = find_attr(ctx, fn, MLIR_KEY_FUNCTION_TYPE);
    bool is_vararg = false;
    if (ftya != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ctx, ftya);
        is_vararg = MLIR_GetTypeFunctionIsVarArg(ctx, fty);
    }
    size_t orig_np = is_vararg ? n_params - 1 : n_params;

    MLIR_RegionHandle body = MLIR_GetOpRegion(ctx, fn, 0);
    MLIR_BlockHandle  entry = MLIR_GetRegionBlock(ctx, body, 0);

    // Build the wasmssa.func body block. For each declared wasm param
    // (including the hidden va_list i32 for varargs), create a fresh
    // block-arg value and bind it via vmap to the corresponding LLVM
    // function entry block-arg.
    F.body_block = MLIR_CreateBlock(ctx);
    MLIR_ValueHandle last_bv = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < n_params; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, param_types[i]);
        MLIR_ValueHandle bv = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                      MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, F.body_block, bv);
        if (i < orig_np) {
            vmap_set(&F, MLIR_GetBlockArg(ctx, entry, i), bv);
        }
        last_bv = bv;
    }
    if (is_vararg) F.va_list_value = last_bv;

    if (!prewalk_func(&F, entry)) goto fail;

    // Prologue: __stack_pointer -= frame_size; sp_def = result.
    if (F.frame_size > 0) {
        MLIR_ValueHandle sp_orig = emit_global_get(&F, /*sp*/0);
        MLIR_ValueHandle kf      = emit_const_i32(&F, (int32_t)F.frame_size);
        MLIR_ValueHandle sp_new  = emit_sub_i32(&F, sp_orig, kf);
        emit_global_set(&F, /*sp*/0, sp_new);
        F.sp_value = sp_new;
    }

    size_t nops = MLIR_GetBlockNumOps(ctx, entry);
    for (size_t i = 0; i < nops; i++) {
        if (!lower_op(&F, MLIR_GetBlockOp(ctx, entry, i))) goto fail;
    }

    {
        // Build the wasmssa.func wrapper op around F.body_block.
        MLIR_NamedAttr attrs[8];
        size_t na = 0;
        attrs[na++] = attr_s(ctx, MLIR_KEY_SYM_NAME,
                             fn_name.str ? fn_name.str : "",
                             fn_name.str ? fn_name.size : 0);
        attrs[na++] = attr_s_hex(ctx, arena, MLIR_KEY_PARAM_TYPES, param_types, n_params);
        attrs[na++] = attr_s_hex(ctx, arena, MLIR_KEY_RESULT_TYPES, result_types, n_results);
        attrs[na++] = attr_b(ctx, MLIR_KEY_EXPORTED, exported);
        // `static` C functions arrive as `llvm.func` with a
        // `llvm.linkage = "#llvm.linkage<internal>"` attribute (set in
        // emit.c). Forward that as a boolean `internal` flag on the
        // wasmssa.func so the wasm binary emitter can mark the
        // function's symbol-table entry BINDING_LOCAL — otherwise two
        // `static`s with the same name in different TUs collide at
        // link time.
        bool internal = false;
        MLIR_AttributeHandle linka = find_attr(ctx, fn, MLIR_KEY_LLVM_LINKAGE);
        if (linka != MLIR_INVALID_HANDLE) {
            string ls = MLIR_GetAttributeString(ctx, linka);
            const char *want = "#llvm.linkage<internal>";
            size_t wantn = strlen(want);
            if (ls.size == wantn && memcmp(ls.str, want, wantn) == 0) {
                internal = true;
            }
        }
        attrs[na++] = attr_b(ctx, MLIR_KEY_INTERNAL, internal);
        // Forward `wasm.export_name` (from `__attribute__((__export_name__("...")))`)
        // so the binary emitter can publish the function under the
        // user-requested export name.
        MLIR_AttributeHandle exa = find_attr(ctx, fn, MLIR_KEY_WASM_EXPORT_NAME);
        if (exa != MLIR_INVALID_HANDLE) {
            string es = MLIR_GetAttributeString(ctx, exa);
            attrs[na++] = attr_s(ctx, MLIR_KEY_EXPORT_NAME, es.str, es.size);
        }

        MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
        MLIR_AppendRegionBlock(ctx, region, F.body_block);
        MLIR_RegionHandle regs[1] = { region };
        MLIR_OpSpec spec = {0};
        spec.type = OP_TYPE_WASMSSA_FUNC;
        spec.attrs = attrs;
        spec.n_attrs = (uint32_t)na;
        spec.regions = regs;
        spec.n_regions = 1;
        MLIR_OpHandle op = MLIR_CreateOpFromSpec(ctx, &spec);
        MLIR_AppendBlockOp(ctx, mod_body, op);
    }

    return true;
fail:
    return false;
}

// =============================================================================
// Module walker.
// =============================================================================
static bool sig_for_func(MLIR_Context *ctx, Arena *arena, MLIR_OpHandle fn,
                         uint8_t **out_p, size_t *out_np,
                         uint8_t **out_r, size_t *out_nr) {
    MLIR_AttributeHandle ftya = find_attr(ctx, fn, MLIR_KEY_FUNCTION_TYPE);
    if (ftya == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ctx, ftya);
    size_t ni = MLIR_GetTypeFunctionNumInputs(ctx, fty);
    size_t no = MLIR_GetTypeFunctionNumResults(ctx, fty);
    bool is_vararg = MLIR_GetTypeFunctionIsVarArg(ctx, fty);
    size_t pn = ni + (is_vararg ? 1 : 0);
    uint8_t *p = (uint8_t *)arena_alloc(arena, pn ? pn : 1);
    for (size_t i = 0; i < ni; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionInput(ctx, fty, i));
        if (v == 0) return false;
        p[i] = v;
    }
    if (is_vararg) p[ni] = WT_I32;  // hidden va_list pointer
    uint8_t *r = (uint8_t *)arena_alloc(arena, no ? no : 1);
    for (size_t i = 0; i < no; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionResult(ctx, fty, i));
        if (v == 0) return false;
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
                               string sym_name, uint32_t size,
                               uint32_t align_pow, bool is_const,
                               const uint8_t *data, const char *relocs_str,
                               size_t relocs_len) {
    MLIR_NamedAttr attrs[16];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, MLIR_KEY_SYM_NAME,
                         sym_name.str ? sym_name.str : "",
                         sym_name.str ? sym_name.size : 0);
    attrs[na++] = attr_i32(ctx, MLIR_KEY_SIZE, size);
    attrs[na++] = attr_i32(ctx, MLIR_KEY_ALIGN_POW, align_pow);
    attrs[na++] = attr_b(ctx, MLIR_KEY_IS_CONST, is_const);
    if (size) {
        attrs[na++] = attr_s(ctx, MLIR_KEY_INIT_DATA, (const char *)data, size);
    } else {
        attrs[na++] = attr_s(ctx, MLIR_KEY_INIT_DATA, "", 0);
    }
    if (relocs_str) {
        attrs[na++] = attr_s(ctx, MLIR_KEY_RELOCS, relocs_str, relocs_len);
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
    MLIR_AttributeHandle sa = find_attr(ctx, op, MLIR_KEY_SYM_NAME);
    if (sa == MLIR_INVALID_HANDLE) return false;
    string sym = MLIR_GetAttributeString(ctx, sa);

    uint32_t size = 0;
    uint32_t align_pow = 0;
    MLIR_AttributeHandle ga = find_attr(ctx, op, MLIR_KEY_GLOBAL_TYPE);
    if (ga != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle gty = MLIR_GetAttributeTypeValue(ctx, ga);
        size = type_size_bytes(ctx, gty);
        align_pow = align_pow_for_type(ctx, gty);
    }

    bool is_const = (find_attr(ctx, op, MLIR_KEY_CONSTANT) != MLIR_INVALID_HANDLE);

    uint8_t *data = NULL;

    MLIR_AttributeHandle va = find_attr(ctx, op, MLIR_KEY_VALUE);
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind ak = MLIR_GetAttributeKind(ctx, va);
        if (ak == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(ctx, va);
            size = (uint32_t)s.size;
            emit_import_global(ctx, arena, body, sym, size, align_pow,
                               is_const, (const uint8_t *)s.str, NULL, 0);
            return true;
        }
        if (ak == MLIR_ATTR_KIND_INTEGER || ak == MLIR_ATTR_KIND_FLOAT) {
            if (size == 0) return false;
            data = (uint8_t *)arena_alloc(arena, size);
            memset(data, 0, size);
            uint64_t bits = 0;
            if (ak == MLIR_ATTR_KIND_INTEGER) {
                bits = (uint64_t)MLIR_GetAttributeInteger(ctx, va);
            } else {
                double d = MLIR_GetAttributeFloat(ctx, va);
                if (size == 4) {
                    float f = (float)d; uint32_t b32;
                    memcpy(&b32, &f, 4); bits = b32;
                } else if (size == 8) {
                    memcpy(&bits, &d, 8);
                } else { return false; }
            }
            for (uint32_t i = 0; i < size && i < 8; i++)
                data[i] = (uint8_t)(bits >> (8 * i));
            emit_import_global(ctx, arena, body, sym, size, align_pow,
                               is_const, data, NULL, 0);
            return true;
        }
        // Unknown attr kind -- fall through to zero-init.
    }

    // Region-init globals with a constant initializer: an
    // llvm.mlir.zero / undef + llvm.insertvalue chain of constants.
    if (MLIR_GetOpNumRegions(ctx, op) > 0 && ga != MLIR_INVALID_HANDLE && size > 0) {
        data = (uint8_t *)arena_alloc(arena, size);
        memset(data, 0, size);
        if (llvm_global_init_bytes(ctx, op, MLIR_GetAttributeTypeValue(ctx, ga), 4,
                                   data, size)) {
            emit_import_global(ctx, arena, body, sym, size, align_pow,
                               is_const, data, NULL, 0);
            return true;
        }
    }

    // Region-init globals: walk for an llvm.mlir.addressof + llvm.return.
    // The only pattern produced by tinyc is:
    //   %0 = llvm.mlir.addressof @other : !llvm.ptr
    //   llvm.return %0 : !llvm.ptr
    // which becomes a 4-byte pointer slot with an R_WASM_MEMORY_ADDR reloc.
    if (MLIR_GetOpNumRegions(ctx, op) > 0) {
        MLIR_RegionHandle rgn = MLIR_GetOpRegion(ctx, op, 0);
        if (MLIR_GetRegionNumBlocks(ctx, rgn) > 0) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, rgn, 0);
            size_t nb = MLIR_GetBlockNumOps(ctx, blk);
            string pending_target = (string){0};
            for (size_t bi = 0; bi < nb; bi++) {
                MLIR_OpHandle bop = MLIR_GetBlockOp(ctx, blk, bi);
                MLIR_OpType bty = MLIR_GetOpType(ctx, bop);
                if (bty == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
                    MLIR_AttributeHandle ta = find_attr(ctx, bop, MLIR_KEY_GLOBAL_NAME);
                    if (ta == MLIR_INVALID_HANDLE)
                        ta = find_attr(ctx, bop, MLIR_KEY_VALUE);  // upstream attr name
                    if (ta == MLIR_INVALID_HANDLE) {
                        // Fallback: any FlatSymbolRef-shaped attr.
                        size_t na = MLIR_GetOpNumAttributes(ctx, bop);
                        for (size_t i = 0; i < na; i++) {
                            MLIR_AttributeHandle a = MLIR_GetOpAttribute(ctx, bop, i);
                            if (MLIR_GetAttributeKind(ctx, a) == MLIR_ATTR_KIND_STRING) {
                                ta = a; break;
                            }
                        }
                    }
                    if (ta == MLIR_INVALID_HANDLE) return false;
                    string ts = MLIR_GetAttributeString(ctx, ta);
                    if (ts.size && ts.str[0] == '@') { ts.str++; ts.size--; }
                    pending_target = ts;
                } else if (bty == OP_TYPE_LLVM_RETURN) {
                    if (!pending_target.str) break;
                    size = 4;
                    if (align_pow == 0) align_pow = 2;
                    data = (uint8_t *)arena_alloc(arena, size);
                    memset(data, 0, size);
                    // Build the relocs string inline: "<off>:<target>:<addend>".
                    size_t cap = pending_target.size + 32;
                    char *buf = (char *)arena_alloc(arena, cap);
                    int n = snprintf(buf, cap, "%u:%.*s:%d", 0u,
                                     (int)pending_target.size,
                                     pending_target.str, 0);
                    size_t off = (n > 0) ? (size_t)n : 0;
                    emit_import_global(ctx, arena, body, sym, size, align_pow,
                                       is_const, data, buf, off);
                    return true;
                }
            }
        }
    }

    // Any other initializer region (one llvm_global_init_bytes rejected) is
    // an error, not zero.
    if (MLIR_GetOpNumRegions(ctx, op) > 0 &&
        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, op, 0)) > 0) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, op, 0), 0);
        size_t nb = MLIR_GetBlockNumOps(ctx, blk);
        for (size_t bi = 0; bi < nb; bi++) {
            MLIR_OpType bty = MLIR_GetOpType(ctx, MLIR_GetBlockOp(ctx, blk, bi));
            if (bty != OP_TYPE_LLVM_MLIR_ZERO && bty != OP_TYPE_LLVM_MLIR_UNDEF &&
                bty != OP_TYPE_LLVM_MLIR_POISON && bty != OP_TYPE_LLVM_RETURN) {
                fprintf(stderr, "wasmssa-lower: unsupported initializer for global '%.*s'\n",
                        (int)sym.size, sym.str);
                return false;
            }
        }
    }

    // No initializer found (e.g. uninitialized scalar global). Zero-init.
    // An empty struct (GNU C) still gets one byte, as an object of its own.
    if (size == 0 && ga != MLIR_INVALID_HANDLE &&
        llvm_type_is_empty(ctx, MLIR_GetAttributeTypeValue(ctx, ga), 4))
        size = 1;
    if (size == 0) return false;
    data = (uint8_t *)arena_alloc(arena, size);
    memset(data, 0, size);
    emit_import_global(ctx, arena, body, sym, size, align_pow,
                       is_const, data, NULL, 0);
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
static MLIR_NamedAttr attr_i32(MLIR_Context *ctx, MLIR_AttrKey key, int64_t v) {
    return MLIR_Named(key, MLIR_CreateAttributeInteger(ctx, v,
                                       MLIR_CreateTypeInteger(ctx, 32, true)));
}
static MLIR_NamedAttr attr_i64(MLIR_Context *ctx, MLIR_AttrKey key, int64_t v) {
    return MLIR_Named(key, MLIR_CreateAttributeInteger(ctx, v,
                                       MLIR_CreateTypeInteger(ctx, 64, true)));
}
static MLIR_NamedAttr attr_b(MLIR_Context *ctx, MLIR_AttrKey key, bool v) {
    return MLIR_Named(key, MLIR_CreateAttributeBool(ctx, v));
}
static MLIR_NamedAttr attr_s(MLIR_Context *ctx, MLIR_AttrKey key,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_Named(key, MLIR_CreateAttributeString(ctx, sv));
}
static MLIR_NamedAttr attr_s_cstr(MLIR_Context *ctx, MLIR_AttrKey key,
                                        const char *v) {
    return attr_s(ctx, key, v, v ? strlen(v) : 0);
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
static MLIR_NamedAttr attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       MLIR_AttrKey key,
                                       const uint8_t *p, size_t n) {
    return attr_s(ctx, key, hex_encode(arena, p, n), n * 2);
}

static MLIR_OpHandle make_op(MLIR_Context *ctx, MLIR_OpType type,
                             MLIR_NamedAttr *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt,
                             MLIR_ValueHandle *out_result) {
    MLIR_TypeHandle res_tys[1];
    uint32_t n_res = 0;
    if (result_vt) {
        res_tys[0] = vt_to_type(ctx, result_vt);
        n_res = 1;
    }
    MLIR_OpSpec spec = {0};
    spec.type = type;
    spec.operands = operands;
    spec.n_operands = (uint32_t)n_operands;
    spec.result_types = res_tys;
    spec.n_results = n_res;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)n_attrs;
    spec.regions = regions;
    spec.n_regions = (uint32_t)n_regions;
    MLIR_OpHandle op = MLIR_CreateOpFromSpec(ctx, &spec);
    if (out_result) *out_result = n_res ? MLIR_GetOpResult(ctx, op, 0) : MLIR_INVALID_HANDLE;
    return op;
}

// Emit an imported (body-less) wasmssa.func op directly into the module body.
static void emit_import_func(MLIR_Context *ctx, Arena *arena,
                             MLIR_BlockHandle body, string sym_name,
                             string import_module, string import_name,
                             const uint8_t *param_types, size_t n_params,
                             const uint8_t *result_types, size_t n_results) {
    MLIR_NamedAttr attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, MLIR_KEY_SYM_NAME,
                         sym_name.str ? sym_name.str : "",
                         sym_name.str ? sym_name.size : 0);
    attrs[na++] = attr_s_hex(ctx, arena, MLIR_KEY_PARAM_TYPES, param_types, n_params);
    attrs[na++] = attr_s_hex(ctx, arena, MLIR_KEY_RESULT_TYPES, result_types, n_results);
    attrs[na++] = attr_b(ctx, MLIR_KEY_EXPORTED, false);
    if (import_module.size > 0) {
        attrs[na++] = attr_s(ctx, MLIR_KEY_IMPORT_MODULE, import_module.str, import_module.size);
    }
    if (import_name.size > 0) {
        attrs[na++] = attr_s(ctx, MLIR_KEY_IMPORT_NAME, import_name.str, import_name.size);
    }
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
// ---------------------------------------------------------------------------
// Input guard. The wasm backend takes LLVM-dialect input only (see the file
// header). Reject, naming the offending op, anything that is not an `llvm.*`
// op -- at module level or anywhere inside a function / global-initializer
// body -- instead of letting some later code path quietly cope with a
// higher-level dialect (cf.*, scf.*, arith.*, func.*, ...). This is what
// keeps a pre-backend lowering regression from going unnoticed.
// ---------------------------------------------------------------------------

// True if the guard accepts an op of type `t`: an op of the LLVM dialect of
// mlir_ops.def, or an op of a dynamic type named `llvm.*`. The latter are
// module-level ops the module walk skips (llvm.module_flags, llvm.comdat,
// llvm.mlir.global_ctors, ...); inside a function the lowering reports them
// as unsupported, naming the op. Only dynamic types reach the name compare.
static bool op_type_is_llvm(MLIR_Context *ctx, MLIR_OpType t) {
    if (MLIR_OP_TYPE_IN_DIALECT(t, LLVM)) return true;
    if ((unsigned)t < (unsigned)OP_TYPE_COUNT) return false;
    string n = MLIR_GetOpTypeName(ctx, t);
    return n.size > 5 && memcmp(n.str, "llvm.", 5) == 0;
}

// `owner` is the enclosing module-level op (for the diagnostic), or
// MLIR_INVALID_HANDLE while walking the module body itself.
static bool check_llvm_only_block(MLIR_Context *ctx, MLIR_BlockHandle block,
                                  MLIR_OpHandle owner) {
    size_t no = MLIR_GetBlockNumOps(ctx, block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, block, i);
        MLIR_OpType t = MLIR_GetOpType(ctx, op);
        if (!op_type_is_llvm(ctx, t)) {
            string n = MLIR_GetOpTypeName(ctx, t);
            string where = str_lit("<module>");
            if (owner != MLIR_INVALID_HANDLE) {
                MLIR_AttributeHandle sa =
                    MLIR_GetOpAttr(ctx, owner, MLIR_KEY_SYM_NAME);
                if (sa != MLIR_INVALID_HANDLE)
                    where = MLIR_GetAttributeString(ctx, sa);
            }
            fprintf(stderr,
                    "wasmssa-lower: op '%.*s' in '%.*s' is not in the LLVM "
                    "dialect; the wasm backend takes LLVM-dialect input only "
                    "(lower the module with MLIR_LowerToLLVMDialect first)\n",
                    (int)n.size, n.str, (int)where.size, where.str);
            return false;
        }
        MLIR_OpHandle inner_owner = owner != MLIR_INVALID_HANDLE ? owner : op;
        size_t nr = MLIR_GetOpNumRegions(ctx, op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(ctx, op, r);
            size_t nb = MLIR_GetRegionNumBlocks(ctx, rg);
            for (size_t b = 0; b < nb; b++) {
                if (!check_llvm_only_block(ctx, MLIR_GetRegionBlock(ctx, rg, b),
                                           inner_owner))
                    return false;
            }
        }
    }
    return true;
}

MLIR_OpHandle mlir_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module) {
    // Step 0: enforce the input contract before anything else looks at the
    // module (in particular before step 1 introduces the backend's own
    // scf.* ops).
    if (MLIR_GetOpNumRegions(ctx, module) == 0 ||
        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, module, 0)) == 0) {
        fprintf(stderr, "wasmssa-lower: module has no body\n");
        return MLIR_INVALID_HANDLE;
    }
    if (!check_llvm_only_block(
            ctx, MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, module, 0), 0),
            MLIR_INVALID_HANDLE))
        return MLIR_INVALID_HANDLE;

    // Step 1 (see the file header): structurize the LLVM-dialect CFG in
    // place. The scf.* ops this creates never leave this function.
    MLIR_LiftCfToScfOptions lift_options = {0};
    lift_options.llvm_cfg = true;
    if (!MLIR_LiftCfToScf(ctx, module, &lift_options)) {
        fprintf(stderr,
                "wasmssa-lower: control-flow structurization failed\n");
        return MLIR_INVALID_HANDLE;
    }

    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t nops = MLIR_GetBlockNumOps(ctx, mb);

    Arena            *arena = MLIR_GetArenaAllocator(ctx);
    MLIR_BlockHandle  body  = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, body);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpSpec mod_spec = {0};
    mod_spec.type = OP_TYPE_MODULE;
    mod_spec.regions = regs;
    mod_spec.n_regions = 1;
    MLIR_OpHandle out_module = MLIR_CreateOpFromSpec(ctx, &mod_spec);

    ModCtx mod = {0};
    mod.ctx = ctx;
    mod.arena = arena;
    mod.body = body;

    // Pass 1: imported (declared, body-less) funcs first so they take the
    // low function-index space in wasm.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_FUNC) continue;
        bool has_body = MLIR_GetOpNumRegions(ctx, op) > 0 &&
                        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, op, 0)) > 0;
        if (has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, arena, op, &p, &np, &r, &nr)) {
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AttributeHandle sa = find_attr(ctx, op, MLIR_KEY_SYM_NAME);
        string nm = MLIR_GetAttributeString(ctx, sa);

        // Forward `wasm.import_module` / `wasm.import_name` annotations
        // (from `__attribute__((__import_module__("...")))`) so the
        // binary emitter can place this import in the requested module
        // (e.g. WASI's `wasi_snapshot_preview1`) instead of the default
        // `env`.
        string imod = {0}, iname = {0};
        MLIR_AttributeHandle iam = find_attr(ctx, op, MLIR_KEY_WASM_IMPORT_MODULE);
        if (iam != MLIR_INVALID_HANDLE) imod = MLIR_GetAttributeString(ctx, iam);
        MLIR_AttributeHandle ian = find_attr(ctx, op, MLIR_KEY_WASM_IMPORT_NAME);
        if (ian != MLIR_INVALID_HANDLE) iname = MLIR_GetAttributeString(ctx, ian);

        emit_import_func(ctx, arena, body, nm, imod, iname, p, np, r, nr);
    }

    // Pass 2: defined funcs in source order. `is_function_symbol`
    // discovers each func via its already-emitted wasmssa.func wrapper
    // in `body`; refs between earlier defs therefore work naturally.
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_FUNC) continue;
        bool has_body = MLIR_GetOpNumRegions(ctx, op) > 0 &&
                        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, op, 0)) > 0;
        if (!has_body) continue;
        uint8_t *p, *r; size_t np, nr;
        if (!sig_for_func(ctx, arena, op, &p, &np, &r, &nr)) {
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AttributeHandle sa = find_attr(ctx, op, MLIR_KEY_SYM_NAME);
        string sym = MLIR_GetAttributeString(ctx, sa);
        bool is_main = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
        string nm = is_main ? str_lit("__original_main") : sym;

        if (!lower_function(ctx, arena, &mod, body, nm, is_main,
                            op, p, np, r, nr)) {
            return MLIR_INVALID_HANDLE;
        }
    }

    // Pass 3: globals last (matches the legacy emit ordering of
    // funcs-before-globals in the previously buffered module).
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_MLIR_GLOBAL) continue;
        if (!lower_global(ctx, arena, body, op)) {
            fprintf(stderr, "wasmssa-lower: failed to lower global\n");
            return MLIR_INVALID_HANDLE;
        }
    }

    return out_module;
}

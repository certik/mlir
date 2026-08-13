// Stage 1 of the native LLVM->WASM pipeline:
// LLVM-dialect builtin.module --(walk)--> wasmssa-form `builtin.module`.
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
#include "mlir_op_names.h"
#include "mlir_llvm_to_wasmssa.h"
#include <base/vector.h>

#include <base/arena.h>

// Set WASMSSA_LOWER_DEBUG=1 to print per-op reject reasons and module-level
// LLVM ops that are not lowered to wasmssa (func/global only today).
static bool wasmssa_lower_debug(void) {
    const char *v = getenv("WASMSSA_LOWER_DEBUG");
    return v && v[0] && v[0] != '0';
}
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

// Parse bit width from MLIR integer type string "iN". Returns 0 if not iN or invalid.
static int parse_i_type_width(string s) {
    if (s.size <= 1 || s.str[0] != 'i') return 0;
    int w = 0;
    for (size_t i = 1; i < s.size; i++) {
        if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
        else return 0;
    }
    return w;
}
static MLIR_AttributeHandle find_attr(MLIR_OpHandle op, const char *name) {
    // First try by-name, which on the upstream backend walks both the
    // discardable attribute dictionary AND the typed property storage
    // (ODS-defined inherent attrs like llvm.func's `function_type`,
    // arith.cmpi's `predicate`, ...).
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    if (a != MLIR_INVALID_HANDLE) return a;
    // Fall back to the indexed attribute list. The two implementations
    // agree on what's "user-set" for ops constructed via MLIR_CreateOp,
    // so this branch matters mainly for operations created by upstream
    // pass internals where the by-name lookup may also miss.
    size_t n = MLIR_GetOpNumAttributes(op);
    for (size_t i = 0; i < n; i++) {
        MLIR_AttributeHandle ai = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(ai);
        if (name_eq(an, name)) return ai;
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
    int w = parse_i_type_width(s);
    if (w == 1 || w == 8 || w == 16 || w == 32) return WT_I32;
    if (w == 64) return WT_I64;
    return 0;
}

// Returns the integer bit width of `ty` (1/8/16/32/64) or 0 if `ty`
// is not an `iN` integer type.
static int int_bits(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return parse_i_type_width(MLIR_GetTypeString(ctx, ty));
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
    int w = parse_i_type_width(s);
    if (w == 1 || w == 8) return 1;
    if (w == 16) return 2;
    if (w == 32) return 4;
    if (w == 64) return 8;
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
typedef struct { uintptr_t key; MLIR_ValueHandle val; } VMapEntry;
typedef struct { uintptr_t key; uint32_t off; } AMapEntry;

typedef struct M8EmissionMetadata M8EmissionMetadata;
typedef struct M8EmitCtx M8EmitCtx;
typedef struct CFGInfoScratch CFGInfoScratch;

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

    const M8EmissionMetadata *m8_plan;     // optional M8 emission metadata
    M8EmitCtx                 *m8_emit;     // active direct-CFG emission state
    MLIR_BlockHandle          current_block; // LLVM block being lowered
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
        if (F->vmap[i].key == (uintptr_t)k) {
            F->vmap[i].val = v;
            return;
        }
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

static int fn_vmap_get(FnCtx *F, MLIR_ValueHandle k, MLIR_ValueHandle *out);

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

// Materialize a wasmssa op directly into the function's MLIR body block.
// Returns the op's MLIR result value (MLIR_INVALID_HANDLE for ops with no
// result). The op's operands[] reference earlier-produced MLIR values.
static MLIR_ValueHandle commit_op(FnCtx *F, wasmssa_op_t *o) {
    MLIR_Context *ctx = F->ctx;
    size_t n_ops = (size_t)(o->n_operands < 0 ? 0 : o->n_operands);
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
        as[nas++] = attr_s(ctx, "target", o->call_target.str, o->call_target.size);
        break;
    case OP_TYPE_WASMSSA_CALL_INDIRECT:
        as[nas++] = attr_s_hex(ctx, F->arena, "sig_params",
                               o->sig_params, o->n_sig_params);
        as[nas++] = attr_s_hex(ctx, F->arena, "sig_results",
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

// Parse a "array<i32: v0, v1, ...>" string into an arena-allocated int32_t
// vector; returns NULL on parse failure. Sets *n_out.
static int32_t *parse_dense_i32_array(Arena *arena, string s, size_t *n_out) {
    *n_out = 0;
    const char *p = s.str;
    const char *end = s.str + s.size;
    const char *pref = "array<i32";
    size_t plen = strlen(pref);
    if (s.size < plen || memcmp(p, pref, plen) != 0) return NULL;
    p += plen;
    // Allow "array<i32>" (empty) or "array<i32: ...>".
    while (p < end && *p == ' ') p++;
    if (p < end && *p == '>') {
        return (int32_t *)arena_alloc(arena, sizeof(int32_t));
    }
    if (p >= end || *p != ':') return NULL;
    p++;
    // Upper bound the element count by counting commas + 1.
    size_t cap = 1;
    for (const char *q = p; q < end && *q != '>'; q++) if (*q == ',') cap++;
    int32_t *vs = (int32_t *)arena_alloc(arena, cap * sizeof(int32_t));
    size_t n = 0;
    while (p < end && *p != '>') {
        while (p < end && (*p == ' ' || *p == ',')) p++;
        if (p >= end || *p == '>') break;
        bool neg = false;
        if (*p == '-') { neg = true; p++; }
        if (p >= end || *p < '0' || *p > '9') return NULL;
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') { v = v*10 + (*p - '0'); p++; }
        if (neg) v = -v;
        vs[n++] = (int32_t)v;
    }
    *n_out = n;
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
    size_t n = MLIR_GetBlockNumOps(m->body);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(m->body, i);
        MLIR_OpType ot = MLIR_GetOpType(op);
        if (ot != OP_TYPE_WASMSSA_FUNC && ot != OP_TYPE_WASMSSA_IMPORT_FUNC) continue;
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) continue;
        string s = MLIR_GetAttributeString(sa);
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
                               MLIR_AttributeHandle *attrs, size_t n_attrs,
                               MLIR_ValueHandle *operands, size_t n_operands,
                               MLIR_RegionHandle *regions, size_t n_regions,
                               const uint8_t *result_vts, size_t n_results,
                               MLIR_ValueHandle *out_results /* size n_results, or NULL */) {
    MLIR_TypeHandle   res_tys_inline[16];
    MLIR_ValueHandle  res_vals_inline[16];
    MLIR_TypeHandle  *res_tys  = res_tys_inline;
    MLIR_ValueHandle *res_vals = res_vals_inline;
    if (n_results > 16) {
        res_tys  = (MLIR_TypeHandle  *)arena_alloc(ctx->arena, n_results * sizeof(MLIR_TypeHandle));
        res_vals = (MLIR_ValueHandle *)arena_alloc(ctx->arena, n_results * sizeof(MLIR_ValueHandle));
    }
    for (size_t i = 0; i < n_results; i++) {
        res_tys[i] = vt_to_type(ctx, result_vts[i]);
        res_vals[i] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, (uint32_t)i,
                                               res_tys[i], (string){0},
                                               MLIR_CreateLocationUnknown(ctx, (string){0}));
    }
    MLIR_OpHandle op = MLIR_CreateOp(ctx, type, op_type_to_string(type),
        attrs, n_attrs, res_tys, n_results, res_vals, n_results,
        operands, n_operands, regions, n_regions,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    if (out_results) {
        for (size_t i = 0; i < n_results; i++) out_results[i] = res_vals[i];
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
    MLIR_AttributeHandle as[1];
    as[0] = attr_i32(F->ctx, "depth", (int64_t)depth);
    MLIR_OpHandle op = make_op_n(F->ctx, OP_TYPE_WASMSSA_BR, as, 1,
                                 vals, n, NULL, 0, NULL, 0, NULL);
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

static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op);
static bool lower_op(FnCtx *F, MLIR_OpHandle op) {
    bool ok = lower_op_inner(F, op);
    if (!ok) {
        string n = MLIR_GetOpName(op);
        if (wasmssa_lower_debug()) {
            fprintf(stderr,
                    "wasmssa-lower: lower_op failed '%.*s' "
                    "(operands=%zu results=%zu regions=%zu)\n",
                    (int)n.size, n.str,
                    MLIR_GetOpNumOperands(op),
                    MLIR_GetOpNumResults(op),
                    MLIR_GetOpNumRegions(op));
        } else {
            fprintf(stderr, "wasmssa-lower: lower_op failed at '%.*s'\n",
                    (int)n.size, n.str);
        }
    }
    return ok;
}
static bool lower_op_inner(FnCtx *F, MLIR_OpHandle op) {
    string name = MLIR_GetOpName(op);

    // ---- llvm.select -----------------------------------------------------
    if (name_eq(name, "llvm.select")) {
        if (MLIR_GetOpNumOperands(op) != 3 || MLIR_GetOpNumResults(op) != 1)
            return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt == 0) return false;
        MLIR_ValueHandle ci, ai, bi;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &ci)) return false;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 1), &ai)) return false;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 2), &bi)) return false;
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
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- ub.poison / llvm.mlir.undef ---------------------------------------
    // Placeholder values on unreachable paths. Lower either name to a typed
    // zero constant. Upstream may use ub.poison; the native port uses
    // llvm.mlir.undef (no UB dialect in the agnostic op table).
    if (name_eq(name, "ub.poison") || name_eq(name, "llvm.mlir.undef")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
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
    if (name_eq(name, "llvm.alloca")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint32_t off;
        if (!amap_get(F, r, &off)) return false;
        if (F->sp_value == MLIR_INVALID_HANDLE) return false;
        MLIR_ValueHandle koff = emit_const_i32(F, (int32_t)off);
        MLIR_ValueHandle addr = emit_add_i32(F, F->sp_value, koff);
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
        MLIR_ValueHandle va, pa;
        if (!fn_vmap_get(F, val, &va)) return false;
        if (!fn_vmap_get(F, ptr, &pa)) return false;

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
    if (name_eq(name, "llvm.load")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 0);
        MLIR_TypeHandle  rt = MLIR_GetValueType(r);
        uint8_t vt = wasm_vt(F->ctx, rt);
        unsigned sz = type_size_bytes(F->ctx, rt);
        if (vt == 0 || sz == 0) return false;
        MLIR_ValueHandle pa;
        if (!fn_vmap_get(F, ptr, &pa)) return false;

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
    if (name_eq(name, "llvm.sext")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int in_w  = int_bits(F->ctx, MLIR_GetValueType(s));
        int out_w = int_bits(F->ctx, MLIR_GetValueType(r));
        if (in_w == 0 || out_w == 0 || in_w > out_w) return false;
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, s, &sa)) return false;

        // Step 1: if narrowing source is i8/i16 stored in i32, sign-extend
        // within i32 first (extend8_s=0xc0, extend16_s=0xc1).
        MLIR_ValueHandle cur_idx = sa;
        uint8_t cur_vt = wasm_vt(F->ctx, MLIR_GetValueType(s));
        if (in_w == 8 || in_w == 16) {
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
    if (name_eq(name, "llvm.ptrtoint") || name_eq(name, "llvm.inttoptr")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, s, &sa)) return false;
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
            MLIR_ValueHandle ai, bi;
            if (!fn_vmap_get(F, a, &ai) || !fn_vmap_get(F, b, &bi)) return false;
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
            MLIR_ValueHandle ai, bi;
            if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &ai)) return false;
            if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 1), &bi)) return false;
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
        MLIR_ValueHandle ai, bi;
        if (!fn_vmap_get(F, a, &ai)) return false;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 1), &bi)) return false;
        MLIR_ValueHandle ops[2] = { ai, bi };
        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_BINOP;
        o.valtype = WT_I32;
        o.wasm_opcode = opc;
        o.n_operands = 2;
        o.operands = ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
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
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, s, &sa)) return false;
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
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, s, &sa)) return false;
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
        MLIR_ValueHandle ai, bi;
        if (!fn_vmap_get(F, a, &ai) || !fn_vmap_get(F, b, &bi)) return false;
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
    if (name_eq(name, "llvm.trunc") || name_eq(name, "llvm.zext")) {
        bool is_zext = name_eq(name, "llvm.zext");
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle s = MLIR_GetOpOperand(op, 0);
        int in_w  = int_bits(F->ctx, MLIR_GetValueType(s));
        int out_w = int_bits(F->ctx, MLIR_GetValueType(r));
        uint8_t in_vt  = wasm_vt(F->ctx, MLIR_GetValueType(s));
        uint8_t out_vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (in_vt == 0 || out_vt == 0) return false;
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, s, &sa)) return false;
        if (is_zext && in_w == 8 && out_w == 32) {
            MLIR_ValueHandle mask = emit_const_i32(F, 255);
            MLIR_ValueHandle idx = emit_and_i32(F, sa, mask);
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
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, s, &sa)) return false;
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
        MLIR_ValueHandle o_ops[1] = { sa };
        o.operands = o_ops;
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.return -------------------------------------------------------
    if (name_eq(name, "llvm.return")) {
        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle retv = MLIR_INVALID_HANDLE;
        if (no == 1) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, 0);
            if (!fn_vmap_get(F, v, &retv)) return false;
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
            uint8_t *sp = (uint8_t *)arena_alloc(F->arena, snp ? snp : 1);
            for (size_t i = 0; i < snp; i++) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpOperand(op, i + 1)));
                if (v == 0) return false;
                sp[i] = v;
            }
            size_t nr = MLIR_GetOpNumResults(op);
            if (nr > 1) return false;
            uint8_t *sr = (uint8_t *)arena_alloc(F->arena, 1);
            size_t snr = 0;
            if (nr == 1) {
                uint8_t v = wasm_vt(F->ctx,
                    MLIR_GetValueType(MLIR_GetOpResult(op, 0)));
                if (v == 0) return false;
                sr[0] = v;
                snr = 1;
            }
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, no * sizeof(MLIR_ValueHandle));
            // wasm call_indirect pops args first then table-index, so order
            // them in operand[] as [args..., funcptr].
            for (size_t i = 0; i < snp; i++) {
                if (!fn_vmap_get(F, MLIR_GetOpOperand(op, i + 1), &opnds[i]))
                    return false;
            }
            if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &opnds[snp]))
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
                    MLIR_ValueHandle av = MLIR_GetOpOperand(op, nfixed + i);
                    MLIR_TypeHandle aty = MLIR_GetValueType(av);
                    uint8_t vt = wasm_vt(F->ctx, aty);
                    MLIR_ValueHandle va;
                    if (!fn_vmap_get(F, av, &va)) { return false; }
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
                    if (!fn_vmap_get(F, MLIR_GetOpOperand(op, i), &opnds[i]))
                        return false;
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
                MLIR_ValueHandle idx = commit_op(F, &oc);
                if (nr == 1) vmap_set(F, r, idx);
                return true;
            }
        }
        string nm = MLIR_GetAttributeAsString(F->ctx, callee);
        // SymbolRefAttr prints as `@name`.
        string cstr = nm;
        if (cstr.size > 0 && cstr.str[0] == '@') { cstr.str++; cstr.size--; }

        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(F->arena, (no ? no : 1) * sizeof(MLIR_ValueHandle));
        for (size_t i = 0; i < no; i++) {
            if (!fn_vmap_get(F, MLIR_GetOpOperand(op, i), &opnds[i]))
                return false;
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
        MLIR_ValueHandle idx = commit_op(F, &o);
        if (nr == 1) vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.getelementptr -----------------------------------------------
    if (name_eq(name, "llvm.getelementptr")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        if (MLIR_GetOpNumOperands(op) < 1) return false;
        MLIR_ValueHandle r    = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle base = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle addr;
        if (!fn_vmap_get(F, base, &addr)) return false;
        MLIR_AttributeHandle eta = find_attr(op, "elem_type");
        if (eta == MLIR_INVALID_HANDLE) return false;
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(eta);
        MLIR_AttributeHandle ria = find_attr(op, "rawConstantIndices");
        if (ria == MLIR_INVALID_HANDLE) return false;
        size_t n_idx = 0;
        int32_t *cidx = parse_dense_i32_array(F->arena,
            MLIR_GetAttributeAsString(F->ctx, ria), &n_idx);
        if (!cidx || n_idx == 0) return false;

        size_t op_idx = 1;
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            MLIR_ValueHandle dyn_def = MLIR_INVALID_HANDLE;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(op)) return false;
                MLIR_ValueHandle ov = MLIR_GetOpOperand(op, op_idx++);
                if (!fn_vmap_get(F, ov, &dyn_def)) return false;
                uint8_t ovt = wasm_vt(F->ctx, MLIR_GetValueType(ov));
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
            } else if (MLIR_IsTypeLLVMStruct(cur_ty)) {
                if (is_dyn) return false;
                unsigned off = struct_field_offset(F->ctx, cur_ty, (size_t)cidx[i]);
                if (off != 0) {
                    MLIR_ValueHandle k = emit_const_i32(F, (int32_t)off);
                    addr = emit_add_i32(F, addr, k);
                }
                cur_ty = MLIR_GetTypeLLVMStructField(cur_ty, (size_t)cidx[i]);
            } else if (MLIR_IsTypeLLVMArray(cur_ty)) {
                MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(cur_ty);
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
            o.call_target = ts;
            o.has_result = true;
            MLIR_ValueHandle idx = commit_op(F, &o);
            vmap_set(F, MLIR_GetOpResult(op, 0), idx);
            return true;
        }

        wasmssa_op_t o = {0};
        o.type = OP_TYPE_WASMSSA_ADDRESSOF;
        o.valtype = WT_I32;
        o.call_target = ts;  // reuse field for symbol name
        o.has_result = true;
        MLIR_ValueHandle idx = commit_op(F, &o);
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
        MLIR_ValueHandle idx = commit_op(F, &o);
        vmap_set(F, r, idx);
        return true;
    }

    // ---- llvm.intr.vastart / llvm.intr.vaend -------------------------------
    if (name_eq(name, "llvm.intr.vastart")) {
        // Store the function's hidden va_list i32 param into *%ap.
        if (F->va_list_value == MLIR_INVALID_HANDLE) return false;
        if (MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle pa;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &pa)) return false;
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
    if (name_eq(name, "llvm.intr.vaend")) {
        // No-op under the wasm32 ABI.
        return true;
    }
    if (name_eq(name, "llvm.intr.vacopy")) {
        // Copy 4 bytes (a char*) from src to dst.
        if (MLIR_GetOpNumOperands(op) != 2) return false;
        MLIR_ValueHandle da, sa;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &da)) return false;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 1), &sa)) return false;
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
    if (name_eq(name, "llvm.intr.sqrt")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        uint8_t opc = (vt == WT_F32) ? 0x91 : (vt == WT_F64) ? 0x9f : 0;
        if (opc == 0) return false;
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &sa)) return false;
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
    if (name_eq(name, "llvm.intr.wasm.memory.size")) {
        if (MLIR_GetOpNumResults(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
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
    if (name_eq(name, "llvm.intr.wasm.memory.grow")) {
        if (MLIR_GetOpNumResults(op) != 1 ||
            MLIR_GetOpNumOperands(op) != 1) return false;
        MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
        uint8_t vt = wasm_vt(F->ctx, MLIR_GetValueType(r));
        if (vt != WT_I32) return false;
        MLIR_ValueHandle sa;
        if (!fn_vmap_get(F, MLIR_GetOpOperand(op, 0), &sa)) return false;
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

    fprintf(stderr, "wasmssa-lower: unsupported op '%.*s'\n",
            (int)name.size, name.str);
    if (wasmssa_lower_debug()) {
        fprintf(stderr,
                "wasmssa-lower: unsupported op detail: operands=%zu "
                "results=%zu regions=%zu\n",
                MLIR_GetOpNumOperands(op), MLIR_GetOpNumResults(op),
                MLIR_GetOpNumRegions(op));
    }
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
    }
    *total_size = cur;
    *n_fixed = nf;
    return true;
}

static bool prewalk_op(FnCtx *F, MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    if (n.size >= 4 && memcmp(n.str, "scf.", 4) == 0) {
        fprintf(stderr,
                "wasmssa-lower: scf dialect ops are not supported (%.*s)\n",
                (int)n.size, n.str);
        return false;
    }
    if ((!F->m8_plan &&
         (name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
          name_eq(n, "llvm.switch"))) ||
        name_eq(n, "cf.br") || name_eq(n, "cf.cond_br") ||
        name_eq(n, "cf.switch")) {
        fprintf(stderr,
                "wasmssa-lower: llvm control flow requires M8 structurization (%.*s)\n",
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

static bool prewalk_region(FnCtx *F, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        if (!prewalk_block(F, MLIR_GetRegionBlock(region, bi))) return false;
    }
    return true;
}

static bool prewalk_func(FnCtx *F, MLIR_RegionHandle fn_region) {
    if (!prewalk_region(F, fn_region)) return false;
    if (F->va_buf_size > 0) {
        F->frame_size = align_up(F->frame_size, 8);
        F->va_buf_offset = F->frame_size;
        F->frame_size += F->va_buf_size;
    }
    F->frame_size = (F->frame_size + 15) & ~15u;
    return true;
}

// Lower a defined `llvm.func` into a `wasmssa.func` op (implemented after CFG types).
static bool lower_function_impl(MLIR_Context *ctx, Arena *arena, ModCtx *mod,
                                MLIR_BlockHandle mod_body,
                                string fn_name, bool exported,
                                MLIR_OpHandle fn,
                                const uint8_t *param_types, size_t n_params,
                                const uint8_t *result_types, size_t n_results,
                                CFGInfoScratch *scratch);
static bool m8_emit_function_body(FnCtx *F);

// =============================================================================
// LLVM CFG structurization — discovery, snapshot, normalization, analysis.
//
// Target pipeline (cf-lowering.md):
//   LLVM MLIR  ->  CFGInfo (immutable snapshot)
//             ->  NormalizedCFG (mutable graph for M5/M7/M8 rewrites)
//             ->  M8PlanNode tree (strict structured plan)
//             ->  wasmssa.if / loop / block emission (M9)
//
// Upstream reference: mlir/lib/Transforms/Utils/CFGToSCF.cpp and the
// mlir_lift_cf_to_scf.c port. Phase mapping:
//   M5 — return normalization (shared return block, unify exits)  [implemented]
//   M7 — cycle normalization (SCC decomposition, loop latches)    [implemented]
//   M8 — branch planning (post-dominance, explicit transfers)      [implemented]
//   M9 — direct structured WasmSSA emission                        [implemented]
//
// LLVM MLIR is never mutated during normalization; only NormalizedCFG changes.
// Discovery mirrors the cf->scf driver's post-order region walk.
// =============================================================================

// ---------------------------------------------------------------------------
// CFGInfo — immutable per-region CFG snapshot taken once from LLVM MLIR.
// Upstream: initial CFG view before CFGToSCF graph rewrites; successor_slot
// preserves llvm.cond_br / llvm.switch operand identity for edge values.
// ---------------------------------------------------------------------------

// Classifies the terminator on a CFG block (llvm.br, llvm.cond_br, etc.).
typedef enum {
    CFG_TERM_NONE,
    CFG_TERM_BR,
    CFG_TERM_COND_BR,
    CFG_TERM_SWITCH,
    CFG_TERM_RETURN,
    CFG_TERM_UNREACHABLE,
} CFGTermKind;

// Directed CFG edge in the immutable snapshot pool.
typedef struct {
    size_t from;            // dense source block index
    size_t to;              // dense destination block index
    size_t successor_slot;  // terminator successor slot on `from`
} CFGEdge;

// One basic block in the immutable snapshot with adjacency into the edge pool.
typedef struct {
    MLIR_BlockHandle block;
    MLIR_OpHandle    terminator;
    CFGTermKind      term_kind;
    bool             reachable;
    size_t           n_successor_slots;
    size_t           n_out_edges;
    size_t           n_in_edges;
} CFGBlock;

// Open-addressing hash map: MLIR_BlockHandle -> dense block index for O(1) lookup.
typedef struct {
    uintptr_t *keys;   // MLIR_BlockHandle, 0 = empty slot
    size_t    *vals;   // dense block index
    size_t     cap;    // power of two
    size_t     n;
} BlockIndexMap;

// Immutable CFG snapshot for one llvm.func region. Built once; never mutated.
typedef struct {
    CFGBlock         *blocks;
    size_t            n_blocks;
    CFGEdge          *edges;
    size_t            n_edges;
    BlockIndexMap     block_to_index;
} CFGInfo;

// ---------------------------------------------------------------------------
// NormalizedCFG — mutable working graph for M5/M7/M8 (CFGToSCF-style rewrites).
// Synthetic blocks, edge redirection, and mux operands live here; LLVM MLIR
// stays frozen.
// ---------------------------------------------------------------------------

// Growable per-block list of NormEdge pool indices (outgoing or incoming).
typedef struct {
    size_t *edge_ids;
    size_t  n;
    size_t  cap;
} NormAdjacency;

// Semantic role of a synthetic block (why it exists in the normalized graph).
// Upstream: mux blocks, unified return, loop latch in CFGToSCF.cpp.
typedef enum {
    NORM_BLOCK_ORIGINAL,
    NORM_BLOCK_SHARED_RETURN,
    NORM_BLOCK_SHARED_UNREACHABLE,
    NORM_BLOCK_ENTRY_MUX,
    NORM_BLOCK_EXIT_MUX,
    NORM_BLOCK_LOOP_LATCH,
    NORM_BLOCK_LOOP_EXIT_DISPATCH,
} NormBlockKind;

// ---------------------------------------------------------------------------
// Value model — NormOperand for normalization and structured emission.
// Upstream: CFGToSCF edge/block argument forwarding; discriminators and typed
// undef padding for multiplexer edges (CFGToSCF.cpp edge multiplexing).
// ---------------------------------------------------------------------------

// Operand tag for values flowing on normalized CFG edges and terminators.
typedef enum {
    NORM_OPERAND_INVALID = 0,
    NORM_OPERAND_MLIR,
    NORM_OPERAND_BLOCK_ARG,
    NORM_OPERAND_DISCRIMINATOR,
    NORM_OPERAND_UNDEF,
} NormOperandKind;

// Tagged union for a value on a normalized edge or in a synthetic terminator.
typedef struct {
    NormOperandKind kind;
    MLIR_TypeHandle type;
    union {
        MLIR_ValueHandle mlir_value;
        struct {
            size_t block_id;
            size_t arg_index;
        } block_arg;
        int32_t discriminator;
    } as;
} NormOperand;

// Stable identity for normalized values in replacement maps (MLIR defs and
// synthetic header/latch block arguments without MLIR handles).
typedef struct {
    NormOperandKind kind;
    MLIR_ValueHandle mlir_value;
    size_t           block_id;
    size_t           arg_index;
} NormValueKey;

// Type for one formal argument on a synthetic (or snapshot) block.
typedef struct {
    MLIR_TypeHandle type;
} NormBlockArg;

// Maps one mux destination to its argument slice and discriminator constant.
// Upstream: CFGToSCF multiplexer destination layout.
typedef struct {
    size_t  destination_block;
    size_t  arg_offset;
    size_t  n_args;
    int32_t discriminator;
} NormMuxEntry;

// Synthetic terminator metadata (switch cases, return values, loop latch).
// Upstream: private terminator state in CFGToSCF before scf emission.
typedef struct {
    CFGTermKind  kind;
    NormOperand  selector;
    int32_t     *case_values;
    size_t      *case_edge_ids;
    size_t       n_cases;
    size_t       default_edge_id;
    NormOperand *return_values;
    size_t       n_return_values;
} NormTerminator;

// Whether a block's effective terminator comes from the LLVM snapshot or an
// owned NormalizedCFG override (M5 virtual branches, synthetic mux/latch).
typedef enum {
    NORM_TERM_SNAPSHOT,
    NORM_TERM_OWNED,
} NormTermSource;

// One node in the mutable normalized graph (original LLVM block or synthetic).
typedef struct {
    NormBlockKind    kind;
    MLIR_BlockHandle mlir_block;
    MLIR_OpHandle    source_terminator;

    NormTermSource   term_source;
    NormTerminator   normalized_term;
    CFGTermKind      effective_term_kind;

    NormBlockArg    *args;
    size_t           n_args;
    NormMuxEntry    *mux_entries;
    size_t           n_mux_entries;
    NormAdjacency    outgoing;
    NormAdjacency    incoming;
    bool             active;
    bool             outgoing_finalized;
} NormBlock;

// Whether edge operands are lazy snapshot refs or owned synthetic copies.
typedef enum {
    NORM_EDGE_SNAPSHOT,
    NORM_EDGE_SYNTHETIC,
} NormEdgePayloadKind;

// SSA values forwarded across a normalized edge (snapshot or synthetic).
typedef struct {
    NormEdgePayloadKind kind;
    size_t              snapshot_edge_id;  // valid when NORM_EDGE_SNAPSHOT
    NormOperand        *operands;
    size_t              n_operands;
} NormEdgePayload;

// Mutable directed edge with adjacency positions for O(1) redirect/remove.
typedef struct {
    size_t from;
    size_t to;
    size_t successor_slot;
    size_t out_position;
    size_t in_position;
    NormEdgePayload payload;
    bool   active;
} NormEdge;

// ---------------------------------------------------------------------------
// CFG graph analyses — reverse postorder, dominance, SCC (view-local).
// Upstream: Cooper-Harvey-Kennedy dominance (mlir_lift_cf_to_scf.c DomInfo),
// reverse postorder for SCC (M7) and branch structuring (M8).
// ---------------------------------------------------------------------------

// Reverse postorder with block->position inverse map for dominance intersect.
typedef struct {
    size_t *order;
    size_t *position;
    size_t  n;
} RPOInfo;

// Immediate-dominator tree; post-dominance uses root for exits.
typedef struct {
    size_t *idom;
    size_t  n;
    size_t  root;
    size_t *dom_lo;   // dominance-tree interval start (view-local dense)
    size_t *dom_hi;   // dominance-tree interval end
} DominanceInfo;

// Tarjan SCC result with component member ranges (M7 cycle normalization).
typedef struct {
    size_t *component_id;       // indexed by normalized block id
    size_t *component_offsets;
    size_t *members;
    bool   *is_cycle;           // per component: size>1 or singleton self-edge
    size_t  n;
    size_t  n_components;
} SCCInfo;

// Logical subgraph for M7/M8 (nested loop bodies without SCF regions).
typedef struct {
    size_t  id;
    size_t  entry_block;
    size_t *blocks;
    size_t  n_blocks;
    size_t *hidden_edges;       // excluded from view traversal (e.g. latch→header)
    size_t  n_hidden_edges;
    size_t  parent_loop;        // SIZE_MAX for root
} NormGraphView;

// Epoch marks for O(1) view membership (avoids linear scans of view->blocks).
typedef struct {
    size_t *block_epoch;
    size_t *edge_epoch;
    size_t  n_blocks;
    size_t  n_edges;
    size_t  current_epoch;
} NormViewMembership;

// Mutable normalized graph + arena pointers.
typedef struct {
    const CFGInfo *snapshot;

    NormBlock *blocks;
    size_t     n_blocks;
    size_t     blocks_cap;

    NormEdge *edges;
    size_t    n_edges;
    size_t    edges_cap;

    size_t entry_block;

    Arena      *graph_arena;
    Arena      *phase_arena;
    arena_pos_t phase_base_pos;
} NormalizedCFG;

// Single-pass iterator over active outgoing/incoming edges of one block.
typedef struct {
    const NormalizedCFG *cfg;
    const NormAdjacency *adj;
    size_t               i;
} NormEdgeIter;

// DFS stack frame for iterative RPO computation.
typedef struct {
    size_t bid;
    size_t edge_i;
} NormRpoFrame;

// DFS stack frame for iterative Tarjan SCC on a NormGraphView.
typedef struct {
    size_t bid;
    size_t edge_i;
} NormSccFrame;

// One llvm.func region that still needs CFG->structured lowering (discovery
// is deferred to per-function lowering; no module-level target list).

// Graph and phase arenas use separate lifetimes:
//   graph_arena — CFGInfo snapshot, NormalizedCFG graph payloads, M7 workspace
//   phase_arena — temporary M5/M7/M8 plans and worklists (reset after commit)
typedef struct {
    Arena      *graph_arena;
    arena_pos_t graph_base_pos;
    Arena      *phase_arena;
    arena_pos_t phase_base_pos;
} CFGAnalysisArena;

// M5 — return normalization (shared exit blocks for llvm.return sites).
typedef enum {
    M5_EXIT_LLVM_RETURN,
    M5_EXIT_LLVM_UNREACHABLE,
} M5ExitFlavor;

typedef struct {
    M5ExitFlavor     flavor;
    MLIR_TypeHandle *operand_types;
    size_t           n_operand_types;
    size_t           shared_exit_block;
    size_t           n_sites;
} M5ExitClass;

typedef struct {
    size_t       source_block;
    size_t       exit_class;
    NormOperand *return_values;
    size_t       n_return_values;
} M5ExitSite;

typedef struct {
    M5ExitClass *classes;
    size_t       n_classes;
    size_t       classes_cap;

    M5ExitSite *sites;
    size_t      n_sites;
    size_t      sites_cap;
} M5Plan;

// Persistent exit combiner retained after M5 for M7 loop-exit normalization.
typedef struct {
    M5ExitClass *classes;
    size_t          n_classes;
} M5ExitCombiner;

// ---------------------------------------------------------------------------
// CFG value-use index (built once from CFGInfo; used by M7 reduce-form).
// ---------------------------------------------------------------------------

typedef struct {
    MLIR_ValueHandle value;
    MLIR_TypeHandle  type;
    size_t           defining_block;  // dense CFGInfo / original norm block id

    size_t           last_use_block;  // SIZE_MAX until first use in a block

    size_t *use_blocks;
    size_t  n_use_blocks;
    size_t  use_blocks_cap;
} CFGValueInfo;

typedef struct {
    uintptr_t *keys;  // MLIR_ValueHandle, 0 = empty
    size_t    *vals;  // index into values[]
    size_t     cap;
    size_t     n;
} ValueIndexMap;

typedef struct {
    CFGValueInfo *values;
    size_t        n_values;
    size_t        values_cap;
    ValueIndexMap value_to_index;

    size_t *defs_by_block_offsets;  // [n_blocks + 1]
    size_t *defs_by_block_indices;  // value indices grouped by defining block
    size_t  n_def_blocks;
} CFGValueIndex;

// ---------------------------------------------------------------------------
// M7 — cycle normalization (SCC → structured loop forest on NormalizedCFG).
// ---------------------------------------------------------------------------

typedef struct {
    MLIR_TypeHandle type;
    NormOperand     header_argument;
    NormOperand     next_value;
    NormOperand     exit_argument;
    MLIR_ValueHandle replacement_source;
} M7IterationValue;

typedef struct {
    MLIR_ValueHandle source_value;
    MLIR_TypeHandle  type;
    NormOperand      header_argument;
    NormOperand      latch_argument;
    NormOperand      exit_argument;
} M7AdditionalLiveOut;

typedef struct {
    NormValueKey     source_key;
    size_t           loop_id;
    NormOperand      replacement;
} M7LiveOutReplacement;

typedef struct {
    size_t  id;
    size_t  parent_loop;  // SIZE_MAX for root-level loops

    size_t  header;
    size_t  latch;
    size_t  exit_dispatch;

    size_t  back_edge;   // latch -> header
    size_t  exit_edge;   // latch -> exit_dispatch

    NormOperand condition;

    size_t *body_blocks;
    size_t  n_body_blocks;

    M7IterationValue *iteration_values;
    size_t            n_iteration_values;

    M7AdditionalLiveOut *additional_live_outs;
    size_t               n_additional_live_outs;

} M7Loop;

typedef struct {
    M7Loop *loops;
    size_t  n_loops;
    size_t  loops_cap;

    NormGraphView *views;
    size_t         n_views;
    size_t         views_cap;

    size_t *header_to_loop;  // [n_norm_blocks] -> loop id or SIZE_MAX

    M7LiveOutReplacement *live_out_replacements;
    size_t                n_live_out_replacements;
    size_t                live_out_replacements_cap;

    size_t *worklist;
    size_t  worklist_n;
    size_t  worklist_cap;
} M7Result;

typedef struct {
    size_t *members;
    size_t  n_members;

    size_t *entry_edges;
    size_t  n_entry_edges;

    size_t *exit_edges;
    size_t  n_exit_edges;

    size_t *back_edges;
    size_t  n_back_edges;

    size_t *entry_targets;
    size_t  n_entry_targets;
} M7CycleEdges;

typedef struct {
    size_t  mux_block;
    size_t  discriminator_arg;  // SIZE_MAX when single destination
    size_t  extra_arg_offset;
    size_t  n_extra_args;
    size_t  n_mux_args;
    NormMuxEntry *entries;
    size_t         n_entries;
} NormEdgeMux;

typedef struct {
    size_t *edge_ids;
    size_t  n_edges;
    size_t *dest_blocks;     // parallel to edge_ids (destination at prepare time)
    NormMuxEntry *entries;
    size_t         n_entries;
    size_t         discriminator_arg;
    size_t         extra_arg_offset;
    size_t         n_extra_args;
    MLIR_TypeHandle *extra_types;
    MLIR_TypeHandle  discriminator_type;
    size_t         n_mux_args;
} NormEdgeMuxPlan;

typedef struct {
    M7CycleEdges edges;

    bool   needs_entry_mux;
    size_t predicted_header;     // block id at prepare; SIZE_MAX if entry mux
    size_t base_header_nargs;    // header args before additional live-out extension

    NormEdgeMuxPlan entry_mux;
    NormEdgeMuxPlan latch_mux;

    size_t           n_base_iterations;
    MLIR_TypeHandle *base_iteration_types;
    MLIR_ValueHandle *base_iteration_values;
    bool             *base_exit_passes;  /* [n_base * n_exit_edges] */

    CFGValueInfo **additional_values;
    size_t         n_additional_values;
    bool           *additional_forward;

    size_t         n_latch_sources;
    size_t        *latch_source_edges;
    bool           *additional_latch_passes;  /* [n_additional * n_latch_sources] */

} M7CyclePlan;

typedef struct {
    size_t latch;
    size_t exit_dispatch;
    size_t back_edge;
    size_t exit_edge;
    NormOperand condition;
} M7LatchResult;

typedef struct {
    MLIR_TypeHandle flag_type;  // i32
} M7Config;

// Reusable per-view analysis workspace (owned by CFGInfoScratch / graph_arena).
typedef struct {
    Arena *persistent_arena;

    NormViewMembership membership;

    size_t *global_to_view;   // [n_blocks] view dense index or SIZE_MAX
    size_t *view_to_global;   // [view_dense_cap]
    size_t  view_dense_cap;

    bool   *view_mark;        // [view_dense_cap] DFS visited / Tarjan
    size_t *view_index_of;    // [view_dense_cap] Tarjan
    size_t *view_lowlink;     // [view_dense_cap]
    bool   *view_on_stack;    // [view_dense_cap]
    size_t *view_tarjan_stk;  // [view_dense_cap]
    NormSccFrame *view_frames; // [view_dense_cap]

    size_t *target_mark;      // [n_blocks] entry-target dedup epochs
    size_t  target_mark_epoch;

    size_t *mux_dest_map;     // [n_blocks] distinct mux dest index or SIZE_MAX

    size_t  dense_map_view_id; // view id global_to_view was built for
} M7ViewWorkspace;

typedef struct CFGInfoScratch CFGInfoScratch;

typedef struct {
    NormValueKey  key;
    size_t        loop_id;
    size_t        loop_depth;
    NormOperand   replacement;
    size_t        next_same_mlir;
} M8ReplacementSlot;

typedef enum {
    M8_PLAN_IF,
    M8_PLAN_SWITCH,
    M8_PLAN_LOOP,
    M8_PLAN_TRANSFER,
    M8_PLAN_RETURN,
    M8_PLAN_UNREACHABLE,
} M8PlanKind;

typedef struct {
    size_t edge_id;
    size_t target_block;
    size_t target_node;
    bool   emit_source_block;
} M8TransferPlan;

typedef struct {
    NormOperand condition;
    size_t then_node;
    size_t else_node;
    size_t continuation_block;
    size_t continuation_node;
    MLIR_TypeHandle *result_types;
    size_t n_results;
} M8IfPlan;

typedef struct {
    NormOperand selector;
    int64_t *case_values;
    size_t *case_nodes;
    size_t n_cases;
    size_t default_node;
    size_t continuation_block;
    size_t continuation_node;
    MLIR_TypeHandle *result_types;
    size_t n_results;
} M8SwitchPlan;

typedef struct {
    size_t loop_id;
    size_t body_node;
    size_t continuation_node;
} M8LoopPlan;

typedef struct {
    M8PlanKind kind;
    size_t source_block;
    union {
        M8IfPlan if_plan;
        M8SwitchPlan switch_plan;
        M8LoopPlan loop_plan;
        M8TransferPlan transfer;
    } as;
} M8PlanNode;

typedef struct M8EmissionMetadata {
    CFGInfoScratch *scratch;
    size_t         *block_body_loop;
    size_t          n_blocks;
    M8ReplacementSlot *replacement_slots;
    size_t             n_replacement_slots;
    ValueIndexMap      replacement_by_mlir;
    M8PlanNode        *nodes;
    size_t             n_nodes;
    size_t             nodes_cap;
    size_t             root_node;
    bool               built;
} M8EmissionMetadata;

// View of one target's CFG snapshot + normalized graph in a shared arena pool.
typedef struct CFGInfoScratch {
    MLIR_Context     *ctx;
    CFGInfo           cfg;
    NormalizedCFG     norm;
    CFGValueIndex     values;
    M5ExitCombiner    m5;
    M7Result          m7;
    M7ViewWorkspace   m7_ws;
    M8EmissionMetadata m8;
} CFGInfoScratch;

// Lower a defined `llvm.func` into a `wasmssa.func` op appended to the
// module body (see lower_function_impl below CFG types).
static bool lower_function_impl(MLIR_Context *ctx, Arena *arena, ModCtx *mod,
                                MLIR_BlockHandle mod_body,
                                string fn_name, bool exported,
                                MLIR_OpHandle fn,
                                const uint8_t *param_types, size_t n_params,
                                const uint8_t *result_types, size_t n_results,
                                CFGInfoScratch *scratch) {
    FnCtx F;
    memset(&F, 0, sizeof F);
    F.ctx = ctx;
    F.arena = arena;
    F.mod = mod;
    F.n_params = n_params;
    F.sp_value = MLIR_INVALID_HANDLE;
    F.va_list_value = MLIR_INVALID_HANDLE;
    F.m8_plan = scratch ? &scratch->m8 : NULL;
    F.current_block = MLIR_INVALID_HANDLE;

    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    bool is_vararg = false;
    if (ftya != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
        is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
    }
    size_t orig_np = is_vararg ? n_params - 1 : n_params;

    MLIR_RegionHandle body = MLIR_GetOpRegion(fn, 0);
    MLIR_BlockHandle  entry = MLIR_GetRegionBlock(body, 0);

    F.body_block = MLIR_CreateBlock(ctx);
    MLIR_ValueHandle last_bv = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < n_params; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, param_types[i]);
        MLIR_ValueHandle bv = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                      MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, F.body_block, bv);
        if (i < orig_np) {
            vmap_set(&F, MLIR_GetBlockArg(entry, i), bv);
        }
        last_bv = bv;
    }
    if (is_vararg) F.va_list_value = last_bv;

    if (!prewalk_func(&F, body)) goto fail;

    if (F.frame_size > 0) {
        MLIR_ValueHandle sp_orig = emit_global_get(&F, /*sp*/0);
        MLIR_ValueHandle kf      = emit_const_i32(&F, (int32_t)F.frame_size);
        MLIR_ValueHandle sp_new  = emit_sub_i32(&F, sp_orig, kf);
        emit_global_set(&F, /*sp*/0, sp_new);
        F.sp_value = sp_new;
    }

    if (scratch) {
        if (!scratch->m8.built || !m8_emit_function_body(&F)) goto fail;
    } else {
        size_t nops = MLIR_GetBlockNumOps(entry);
        F.current_block = entry;
        for (size_t i = 0; i < nops; i++) {
            if (!lower_op(&F, MLIR_GetBlockOp(entry, i))) goto fail;
        }
    }

    {
        MLIR_AttributeHandle attrs[8];
        size_t na = 0;
        attrs[na++] = attr_s(ctx, "sym_name",
                             fn_name.str ? fn_name.str : "",
                             fn_name.str ? fn_name.size : 0);
        attrs[na++] = attr_s_hex(ctx, arena, "param_types", param_types, n_params);
        attrs[na++] = attr_s_hex(ctx, arena, "result_types", result_types, n_results);
        attrs[na++] = attr_b(ctx, "exported", exported);
        bool internal = false;
        MLIR_AttributeHandle linka = find_attr(fn, "llvm.linkage");
        if (linka != MLIR_INVALID_HANDLE) {
            string ls = MLIR_GetAttributeString(linka);
            const char *want = "#llvm.linkage<internal>";
            size_t wantn = strlen(want);
            if (ls.size == wantn && memcmp(ls.str, want, wantn) == 0) {
                internal = true;
            }
        }
        attrs[na++] = attr_b(ctx, "internal", internal);
        MLIR_AttributeHandle exa = find_attr(fn, "wasm.export_name");
        if (exa != MLIR_INVALID_HANDLE) {
            string es = MLIR_GetAttributeString(exa);
            attrs[na++] = attr_s(ctx, "export_name", es.str, es.size);
        }

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

    return true;
fail:
    return false;
}

// ---------------------------------------------------------------------------
// BlockIndexMap — O(1) MLIR_BlockHandle -> dense index.
// Upstream: dense block numbering used throughout CFGToSCF analyses.
// ---------------------------------------------------------------------------
// Look up dense block index for an MLIR block handle; SIZE_MAX if absent.
static size_t block_index_map_probe(const BlockIndexMap *map, MLIR_BlockHandle b) {
    if (map->cap == 0) return SIZE_MAX;
    uintptr_t key = (uintptr_t)b;
    size_t mask = map->cap - 1;
    size_t i = map_hash(key) & mask;
    while (map->keys[i] != 0) {
        if (map->keys[i] == key) return map->vals[i];
        i = (i + 1) & mask;
    }
    return SIZE_MAX;
}

// Initialize an empty BlockIndexMap with at least `min_cap` slots.
static void block_index_map_init(BlockIndexMap *map, Arena *arena, size_t min_cap) {
    memset(map, 0, sizeof(*map));
    size_t cap = 16;
    while (cap < min_cap) cap *= 2;
    map->cap = cap;
    map->keys = arena_new_array(arena, uintptr_t, cap);
    map->vals = arena_new_array(arena, size_t, cap);
    memset(map->keys, 0, cap * sizeof(uintptr_t));
}

// Double BlockIndexMap capacity and rehash all entries into a new table.
static void block_index_map_grow(BlockIndexMap *map, Arena *arena) {
    size_t ocap = map->cap;
    uintptr_t *okeys = map->keys;
    size_t *ovals = map->vals;
    size_t ncap = ocap ? ocap * 2 : 16;
    map->cap = ncap;
    map->keys = arena_new_array(arena, uintptr_t, ncap);
    map->vals = arena_new_array(arena, size_t, ncap);
    memset(map->keys, 0, ncap * sizeof(uintptr_t));
    map->n = 0;
    size_t mask = ncap - 1;
    for (size_t i = 0; i < ocap; ++i) {
        uintptr_t key = okeys[i];
        if (key == 0) continue;
        size_t j = map_hash(key) & mask;
        while (map->keys[j] != 0) j = (j + 1) & mask;
        map->keys[j] = key;
        map->vals[j] = ovals[i];
        map->n++;
    }
}

// Insert block->index mapping; first insert wins on duplicate keys.
static void block_index_map_put(BlockIndexMap *map, Arena *arena,
                                MLIR_BlockHandle b, size_t index) {
    uintptr_t key = (uintptr_t)b;
    if ((map->n + 1) * 4 >= map->cap * 3) block_index_map_grow(map, arena);
    size_t mask = map->cap - 1;
    size_t i = map_hash(key) & mask;
    while (map->keys[i] != 0) {
        if (map->keys[i] == key) return;  // first insert wins
        i = (i + 1) & mask;
    }
    map->keys[i] = key;
    map->vals[i] = index;
    map->n++;
}

// ---------------------------------------------------------------------------
// CFGInfo query helpers.
// ---------------------------------------------------------------------------
// Map an LLVM terminator op to CFGTermKind for snapshot classification.
static CFGTermKind cfg_term_kind_of(MLIR_OpHandle term) {
    if (term == MLIR_INVALID_HANDLE) return CFG_TERM_NONE;
    string n = MLIR_GetOpName(term);
    if (name_eq(n, "llvm.br")) return CFG_TERM_BR;
    if (name_eq(n, "llvm.cond_br")) return CFG_TERM_COND_BR;
    if (name_eq(n, "llvm.switch")) return CFG_TERM_SWITCH;
    if (name_eq(n, "llvm.return")) return CFG_TERM_RETURN;
    if (name_eq(n, "llvm.unreachable")) return CFG_TERM_UNREACHABLE;
    return CFG_TERM_NONE;
}

// Dense index of `block` in CFGInfo, or SIZE_MAX.
static size_t cfg_info_block_index(const CFGInfo *cfg, MLIR_BlockHandle block) {
    return block_index_map_probe(&cfg->block_to_index, block);
}

static const CFGEdge *cfg_info_edge_at(const CFGInfo *cfg, size_t edge_id) {
    if (edge_id >= cfg->n_edges) return NULL;
    return &cfg->edges[edge_id];
}

// Count of MLIR successor operands on snapshot edge `edge_id`.
static size_t cfg_info_edge_num_operands(const CFGInfo *cfg, size_t edge_id) {
    const CFGEdge *e = cfg_info_edge_at(cfg, edge_id);
    if (!e || e->from >= cfg->n_blocks) return 0;
    MLIR_OpHandle term = cfg->blocks[e->from].terminator;
    if (term == MLIR_INVALID_HANDLE) return 0;
    return MLIR_GetOpNumSuccessorOperands(term, e->successor_slot);
}

// MLIR successor operand at `op_idx` on snapshot edge `edge_id`.
static MLIR_ValueHandle cfg_info_edge_operand(const CFGInfo *cfg, size_t edge_id,
                                              size_t op_idx) {
    const CFGEdge *e = cfg_info_edge_at(cfg, edge_id);
    if (!e || e->from >= cfg->n_blocks) return MLIR_INVALID_HANDLE;
    MLIR_OpHandle term = cfg->blocks[e->from].terminator;
    if (term == MLIR_INVALID_HANDLE) return MLIR_INVALID_HANDLE;
    return MLIR_GetOpSuccessorOperand(term, e->successor_slot, op_idx);
}

// Resolve in-region successor to dense index; filters unreachable blocks.
static bool cfg_edge_dest_index(const CFGInfo *cfg, MLIR_RegionHandle region,
                                MLIR_BlockHandle succ, size_t *out_dest) {
    if (succ == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetBlockParentRegion(succ) != region) return false;
    size_t di = cfg_info_block_index(cfg, succ);
    if (di == SIZE_MAX || di >= cfg->n_blocks) return false;
    if (!cfg->blocks[di].reachable) return false;
    if (out_dest) *out_dest = di;
    return true;
}

// Build immutable CFGInfo: index blocks, BFS reachability, edge pool.
// Upstream: initial CFG snapshot before CFGToSCF rewrites.
static bool cfg_info_build(Arena *arena, MLIR_RegionHandle region,
                           CFGInfo *out) {
    memset(out, 0, sizeof(*out));
    size_t n = MLIR_GetRegionNumBlocks(region);
    if (n == 0) return false;

    out->n_blocks = n;
    out->blocks = arena_new_array(arena, CFGBlock, n);
    memset(out->blocks, 0, n * sizeof(CFGBlock));
    block_index_map_init(&out->block_to_index, arena, n * 2);

    for (size_t i = 0; i < n; ++i) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, i);
        out->blocks[i].block = b;
        out->blocks[i].terminator = MLIR_GetBlockTerminator(b);
        out->blocks[i].term_kind = cfg_term_kind_of(out->blocks[i].terminator);
        block_index_map_put(&out->block_to_index, arena, b, i);
    }

    bool *visited = arena_new_array(arena, bool, n);
    memset(visited, 0, n * sizeof(bool));
    size_t *queue = arena_new_array(arena, size_t, n);
    size_t head = 0, tail = 0;
    visited[0] = true;
    out->blocks[0].reachable = true;
    queue[tail++] = 0;
    while (head < tail) {
        size_t bi = queue[head++];
        MLIR_OpHandle term = out->blocks[bi].terminator;
        if (term == MLIR_INVALID_HANDLE) continue;
        size_t ns = MLIR_GetOpNumSuccessors(term);
        for (size_t i = 0; i < ns; ++i) {
            MLIR_BlockHandle s = MLIR_GetOpSuccessor(term, i);
            if (s == MLIR_INVALID_HANDLE) continue;
            if (MLIR_GetBlockParentRegion(s) != region) continue;
            size_t si = block_index_map_probe(&out->block_to_index, s);
            if (si == SIZE_MAX || visited[si]) continue;
            visited[si] = true;
            out->blocks[si].reachable = true;
            queue[tail++] = si;
        }
    }

    // Edge pass 1: classify terminators and count edges on reachable blocks.
    size_t total_edges = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!out->blocks[i].reachable) continue;
        CFGBlock *b = &out->blocks[i];
        MLIR_OpHandle term = b->terminator;
        size_t ns = (term == MLIR_INVALID_HANDLE) ? 0
                                                  : MLIR_GetOpNumSuccessors(term);
        b->n_successor_slots = ns;
        if (ns == 0) continue;
        for (size_t s = 0; s < ns; ++s) {
            size_t di;
            if (!cfg_edge_dest_index(out, region,
                                     MLIR_GetOpSuccessor(term, s), &di)) {
                continue;
            }
            total_edges++;
            b->n_out_edges++;
            out->blocks[di].n_in_edges++;
        }
    }

    out->n_edges = total_edges;
    out->edges = total_edges ? arena_new_array(arena, CFGEdge, total_edges) : NULL;

    size_t edge_id = 0;
    for (size_t i = 0; i < n; ++i) {
        CFGBlock *src = &out->blocks[i];
        if (!src->reachable) continue;
        MLIR_OpHandle term = src->terminator;
        if (term == MLIR_INVALID_HANDLE) continue;
        for (size_t s = 0; s < src->n_successor_slots; ++s) {
            size_t di;
            if (!cfg_edge_dest_index(out, region,
                                     MLIR_GetOpSuccessor(term, s), &di)) {
                continue;
            }
            out->edges[edge_id].from = i;
            out->edges[edge_id].to = di;
            out->edges[edge_id].successor_slot = s;
            edge_id++;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// NormalizedCFG — mutable graph layered on an immutable CFGInfo snapshot.
// Upstream: CFGToSCF graph mutation layer (mux insertion, edge redirect).
// ---------------------------------------------------------------------------

// --- Adjacency lists, operands, and block/terminator helpers ---

static void norm_ensure_block_slots(NormalizedCFG *n, Arena *arena, size_t need);

// True when edge and both endpoint blocks are active.
static bool norm_edge_is_active(const NormalizedCFG *n, size_t edge_id);
// True when `edge_id` is an active outgoing edge of `block_id` (O(1)).
static bool norm_edge_is_outgoing_from(const NormalizedCFG *n, size_t edge_id,
                                       size_t block_id);

// Begin single-pass iteration over active outgoing edges of `block_id`.
static NormEdgeIter norm_out_edges(const NormalizedCFG *n, size_t block_id) {
    NormEdgeIter it;
    it.cfg = n;
    it.i = 0;
    if (block_id >= n->n_blocks || !n->blocks[block_id].active) {
        it.adj = NULL;
        return it;
    }
    it.adj = &n->blocks[block_id].outgoing;
    return it;
}

// Begin single-pass iteration over active incoming edges of `block_id`.
static NormEdgeIter norm_in_edges(const NormalizedCFG *n, size_t block_id) {
    NormEdgeIter it;
    it.cfg = n;
    it.i = 0;
    if (block_id >= n->n_blocks || !n->blocks[block_id].active) {
        it.adj = NULL;
        return it;
    }
    it.adj = &n->blocks[block_id].incoming;
    return it;
}

// Advance edge iterator; returns next active edge id or false.
static bool norm_edge_iter_next(NormEdgeIter *it, size_t *out_edge_id) {
    if (!it->adj) return false;
    while (it->i < it->adj->n) {
        size_t eid = it->adj->edge_ids[it->i++];
        if (!norm_edge_is_active(it->cfg, eid)) continue;
        if (out_edge_id) *out_edge_id = eid;
        return true;
    }
    return false;
}

// Allocate exact-capacity adjacency list (snapshot build path).
static void norm_adj_prealloc(Arena *arena, NormAdjacency *adj, size_t cap) {
    adj->n = 0;
    adj->cap = 0;
    adj->edge_ids = NULL;
    if (cap == 0) return;
    adj->edge_ids = arena_new_array(arena, size_t, cap);
    adj->cap = cap;
}

// Append edge id assuming adjacency has spare capacity.
static void norm_adj_append(NormAdjacency *adj, size_t edge_id) {
    adj->edge_ids[adj->n++] = edge_id;
}

// Append edge id, growing adjacency geometrically when needed.
static void norm_adj_push(Arena *arena, NormAdjacency *adj, size_t edge_id) {
    if (adj->cap > 0 && adj->n < adj->cap) {
        norm_adj_append(adj, edge_id);
        return;
    }
    if (adj->n >= adj->cap) {
        size_t ncap = adj->cap ? adj->cap * 2 : 4;
        size_t *next = arena_new_array(arena, size_t, ncap);
        if (adj->n > 0) memcpy(next, adj->edge_ids, adj->n * sizeof(size_t));
        adj->edge_ids = next;
        adj->cap = ncap;
    }
    adj->edge_ids[adj->n++] = edge_id;
}

// Grow block storage so `n_blocks + extra` slots exist without realloc during commit.
static bool normalized_cfg_reserve_blocks(NormalizedCFG *n, size_t extra) {
    if (!n->graph_arena) return false;
    norm_ensure_block_slots(n, n->graph_arena, n->n_blocks + extra);
    return n->blocks_cap >= n->n_blocks + extra;
}

// Grow edge pool so `n_edges + extra` slots exist without realloc during commit.
static bool normalized_cfg_reserve_edges(NormalizedCFG *n, size_t extra) {
    if (!n->graph_arena) return false;
    size_t need = n->n_edges + extra;
    while (n->edges_cap < need) {
        size_t ncap = n->edges_cap ? n->edges_cap * 2 : 16;
        if (ncap < need) ncap = need;
        NormEdge *ne = arena_new_array(n->graph_arena, NormEdge, ncap);
        if (!ne) return false;
        if (n->n_edges > 0) {
            memcpy(ne, n->edges, n->n_edges * sizeof(NormEdge));
        }
        n->edges = ne;
        n->edges_cap = ncap;
    }
    return true;
}

// Reserve outgoing adjacency capacity on `block_id` for `extra` more edges.
static bool normalized_cfg_reserve_outgoing(NormalizedCFG *n, size_t block_id,
                                            size_t extra) {
    if (block_id >= n->n_blocks) return false;
    if (!n->graph_arena) return false;
    NormAdjacency *adj = &n->blocks[block_id].outgoing;
    size_t need = adj->n + extra;
    if (adj->cap >= need) return true;
    size_t *next = arena_new_array(n->graph_arena, size_t, need);
    if (!next) return false;
    if (adj->n > 0) {
        memcpy(next, adj->edge_ids, adj->n * sizeof(size_t));
    }
    adj->edge_ids = next;
    adj->cap = need;
    return true;
}

// O(1) swap-remove at `pos`; updates moved edge's position field.
static void norm_adj_remove_swap(NormAdjacency *adj, size_t pos,
                                 NormEdge *edges, bool is_outgoing) {
    if (pos >= adj->n) return;
    size_t last = adj->n - 1;
    if (pos != last) {
        size_t moved_eid = adj->edge_ids[last];
        adj->edge_ids[pos] = moved_eid;
        if (is_outgoing) {
            edges[moved_eid].out_position = pos;
        } else {
            edges[moved_eid].in_position = pos;
        }
    }
    adj->n--;
}

// Zero-init NormTerminator; default_edge_id = SIZE_MAX.
static void norm_terminator_init(NormTerminator *t) {
    memset(t, 0, sizeof(*t));
    t->default_edge_id = SIZE_MAX;
}

// Zero-init NormOperand (NORM_OPERAND_INVALID).
static void norm_operand_init(NormOperand *o) {
    memset(o, 0, sizeof(*o));
}

// Construct NORM_OPERAND_MLIR with type and MLIR value.
static NormOperand norm_operand_mlir(MLIR_TypeHandle ty, MLIR_ValueHandle v) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_MLIR;
    o.type = ty;
    o.as.mlir_value = v;
    return o;
}

// Construct NORM_OPERAND_BLOCK_ARG referencing a block formal.
static NormOperand norm_operand_block_arg(MLIR_TypeHandle ty, size_t block_id,
                                          size_t arg_index) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_BLOCK_ARG;
    o.type = ty;
    o.as.block_arg.block_id = block_id;
    o.as.block_arg.arg_index = arg_index;
    return o;
}

// Construct NORM_OPERAND_DISCRIMINATOR mux constant.
static NormOperand norm_operand_discriminator(MLIR_TypeHandle ty, int32_t disc) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_DISCRIMINATOR;
    o.type = ty;
    o.as.discriminator = disc;
    return o;
}

// Construct typed NORM_OPERAND_UNDEF padding value.
static NormOperand norm_operand_undef(MLIR_TypeHandle ty) {
    NormOperand o;
    norm_operand_init(&o);
    o.kind = NORM_OPERAND_UNDEF;
    o.type = ty;
    return o;
}

// False for zero-initialized / NORM_OPERAND_INVALID operands.
static bool norm_operand_is_valid(const NormOperand *o) {
    return o && o->kind != NORM_OPERAND_INVALID;
}

static void norm_value_key_init(NormValueKey *k) {
    memset(k, 0, sizeof(*k));
    k->kind = NORM_OPERAND_INVALID;
    k->mlir_value = MLIR_INVALID_HANDLE;
    k->block_id = SIZE_MAX;
    k->arg_index = SIZE_MAX;
}

static bool norm_value_key_is_valid(const NormValueKey *k) {
    if (!k || k->kind == NORM_OPERAND_INVALID) return false;
    if (k->kind == NORM_OPERAND_MLIR) {
        return k->mlir_value != MLIR_INVALID_HANDLE;
    }
    if (k->kind == NORM_OPERAND_BLOCK_ARG) {
        return k->block_id != SIZE_MAX;
    }
    return false;
}

static NormValueKey norm_value_key_mlir(MLIR_ValueHandle v) {
    NormValueKey k;
    norm_value_key_init(&k);
    k.kind = NORM_OPERAND_MLIR;
    k.mlir_value = v;
    return k;
}

static bool norm_value_key_from_operand(const NormOperand *op, NormValueKey *key) {
    if (!op || !norm_operand_is_valid(op) || !key) return false;
    norm_value_key_init(key);
    if (op->kind == NORM_OPERAND_MLIR) {
        key->kind = NORM_OPERAND_MLIR;
        key->mlir_value = op->as.mlir_value;
        return key->mlir_value != MLIR_INVALID_HANDLE;
    }
    if (op->kind == NORM_OPERAND_BLOCK_ARG) {
        key->kind = NORM_OPERAND_BLOCK_ARG;
        key->block_id = op->as.block_arg.block_id;
        key->arg_index = op->as.block_arg.arg_index;
        return true;
    }
    return false;
}

static bool norm_value_key_equal(const NormValueKey *a, const NormValueKey *b) {
    if (!a || !b || a->kind != b->kind) return false;
    if (a->kind == NORM_OPERAND_MLIR) {
        return a->mlir_value == b->mlir_value;
    }
    if (a->kind == NORM_OPERAND_BLOCK_ARG) {
        return a->block_id == b->block_id && a->arg_index == b->arg_index;
    }
    return false;
}

// Copy MLIR block argument types into NormBlockArg signature.
static void norm_block_init_original_signature(Arena *arena, NormBlock *nb,
                                               MLIR_BlockHandle b) {
    size_t nargs = MLIR_GetBlockNumArgs(b);
    if (nargs == 0) return;
    nb->args = arena_new_array(arena, NormBlockArg, nargs);
    nb->n_args = nargs;
    for (size_t i = 0; i < nargs; ++i) {
        MLIR_ValueHandle arg = MLIR_GetBlockArg(b, i);
        nb->args[i].type = MLIR_GetValueType(arg);
    }
}

// Attach NormMuxEntry destination layout table to a block.
static bool norm_block_set_mux_entries(NormBlock *nb, Arena *arena,
                                       const NormMuxEntry *entries, size_t n) {
    nb->mux_entries = NULL;
    nb->n_mux_entries = 0;
    if (n == 0) return true;
    nb->mux_entries = arena_new_array(arena, NormMuxEntry, n);
    if (!nb->mux_entries) return false;
    nb->n_mux_entries = n;
    memcpy(nb->mux_entries, entries, n * sizeof(NormMuxEntry));
    return true;
}

// False after outgoing adjacency/terminator finalization on a synthetic block.
static bool norm_synthetic_block_allows_outgoing_mutation(const NormBlock *b) {
    if (b->kind == NORM_BLOCK_ORIGINAL) return true;
    return !b->outgoing_finalized;
}

// Arena-copy variable-length NormTerminator arrays.
static bool norm_terminator_deep_copy(Arena *arena, const NormTerminator *src,
                                      NormTerminator *dst) {
    norm_terminator_init(dst);
    if (!src) return true;
    dst->kind = src->kind;
    dst->selector = src->selector;
    dst->default_edge_id = src->default_edge_id;
    dst->n_cases = src->n_cases;
    if (src->n_cases > 0) {
        if (!src->case_values || !src->case_edge_ids) return false;
        dst->case_values = arena_new_array(arena, int32_t, src->n_cases);
        dst->case_edge_ids = arena_new_array(arena, size_t, src->n_cases);
        if (!dst->case_values || !dst->case_edge_ids) return false;
        memcpy(dst->case_values, src->case_values, src->n_cases * sizeof(int32_t));
        memcpy(dst->case_edge_ids, src->case_edge_ids, src->n_cases * sizeof(size_t));
    }
    dst->n_return_values = src->n_return_values;
    if (src->n_return_values > 0) {
        if (!src->return_values) return false;
        dst->return_values = arena_new_array(arena, NormOperand, src->n_return_values);
        if (!dst->return_values) return false;
        memcpy(dst->return_values, src->return_values,
               src->n_return_values * sizeof(NormOperand));
    }
    return true;
}

// Arena-copy NormBlockArg formal signature onto a block.
static bool norm_block_set_signature(NormBlock *nb, Arena *arena,
                                     const NormBlockArg *args, size_t n_args) {
    nb->args = NULL;
    nb->n_args = 0;
    if (n_args == 0) return true;
    nb->args = arena_new_array(arena, NormBlockArg, n_args);
    if (!nb->args) return false;
    nb->n_args = n_args;
    memcpy(nb->args, args, n_args * sizeof(NormBlockArg));
    return true;
}

// Allocate block signature directly from MLIR type handles (one owned array).
static bool norm_block_set_signature_from_types(NormBlock *nb, Arena *arena,
                                                const MLIR_TypeHandle *types,
                                                size_t n_types) {
    nb->args = NULL;
    nb->n_args = 0;
    if (n_types == 0) return true;
    nb->args = arena_new_array(arena, NormBlockArg, n_types);
    if (!nb->args) return false;
    nb->n_args = n_types;
    for (size_t i = 0; i < n_types; ++i) {
        nb->args[i].type = types[i];
    }
    return true;
}

// Zero-init NormBlock including owned-terminator defaults.
static void norm_block_init(NormBlock *b) {
    memset(b, 0, sizeof(*b));
    norm_terminator_init(&b->normalized_term);
    b->term_source = NORM_TERM_SNAPSHOT;
    b->effective_term_kind = CFG_TERM_NONE;
}

// Grow NormalizedCFG blocks array, initializing new NormBlock slots.
static void norm_ensure_block_slots(NormalizedCFG *n, Arena *arena, size_t need) {
    while (n->blocks_cap < need) {
        size_t ocap = n->blocks_cap;
        size_t ncap = ocap ? ocap * 2 : 8;
        NormBlock *nb = arena_new_array(arena, NormBlock, ncap);
        if (ocap > 0) {
            memcpy(nb, n->blocks, ocap * sizeof(NormBlock));
        }
        for (size_t i = ocap; i < ncap; ++i) {
            norm_block_init(&nb[i]);
        }
        n->blocks = nb;
        n->blocks_cap = ncap;
    }
}

// Replace edge payload with owned synthetic NormOperand array.
static bool norm_edge_payload_set_operands(Arena *arena, NormEdgePayload *payload,
                                           const NormOperand *operands,
                                           size_t n_operands) {
    payload->kind = NORM_EDGE_SYNTHETIC;
    payload->snapshot_edge_id = SIZE_MAX;
    payload->operands = NULL;
    payload->n_operands = 0;
    if (n_operands == 0) return true;
    payload->operands = arena_new_array(arena, NormOperand, n_operands);
    if (!payload->operands) return false;
    payload->n_operands = n_operands;
    memcpy(payload->operands, operands, n_operands * sizeof(NormOperand));
    return true;
}

// Adopt an existing operand array (must live in graph_arena); no copy.
static void norm_edge_payload_adopt_operands(NormEdgePayload *payload,
                                           NormOperand *operands,
                                           size_t n_operands) {
    memset(payload, 0, sizeof(*payload));
    payload->kind = NORM_EDGE_SYNTHETIC;
    payload->snapshot_edge_id = SIZE_MAX;
    payload->operands = operands;
    payload->n_operands = n_operands;
}

// Mark payload as lazy snapshot reference (no operand copy).
static void norm_edge_payload_init_snapshot(NormEdgePayload *out,
                                            size_t snapshot_edge_id) {
    memset(out, 0, sizeof(*out));
    out->kind = NORM_EDGE_SNAPSHOT;
    out->snapshot_edge_id = snapshot_edge_id;
}

// Operand count; lazy-resolve snapshot edges via CFGInfo.
static size_t norm_edge_payload_num_operands(const NormalizedCFG *n,
                                             const NormEdgePayload *payload) {
    if (!payload) return 0;
    if (payload->kind == NORM_EDGE_SNAPSHOT && n->snapshot) {
        return cfg_info_edge_num_operands(n->snapshot, payload->snapshot_edge_id);
    }
    return payload->n_operands;
}

// Resolve one edge operand into caller-owned NormOperand.
static bool norm_edge_payload_operand(const NormalizedCFG *n,
                                      const NormEdgePayload *payload,
                                      size_t op_idx, NormOperand *out) {
    if (!payload || !out) return false;
    if (payload->kind == NORM_EDGE_SNAPSHOT && n->snapshot) {
        size_t sid = payload->snapshot_edge_id;
        if (op_idx >= cfg_info_edge_num_operands(n->snapshot, sid)) return false;
        MLIR_ValueHandle v = cfg_info_edge_operand(n->snapshot, sid, op_idx);
        *out = norm_operand_mlir(MLIR_GetValueType(v), v);
        return true;
    }
    if (op_idx >= payload->n_operands) return false;
    *out = payload->operands[op_idx];
    return true;
}

// Append edge to normalized graph; internal builder primitive.
static size_t normalized_cfg_add_edge(NormalizedCFG *n, Arena *arena,
                                      size_t from, size_t to,
                                      size_t succ_slot, size_t snapshot_edge_id) {
    if (n->n_edges >= n->edges_cap) {
        size_t ncap = n->edges_cap ? n->edges_cap * 2 : 16;
        NormEdge *ne = arena_new_array(arena, NormEdge, ncap);
        if (n->n_edges > 0) {
            memcpy(ne, n->edges, n->n_edges * sizeof(NormEdge));
        }
        n->edges = ne;
        n->edges_cap = ncap;
    }
    size_t eid = n->n_edges++;
    NormEdge *e = &n->edges[eid];
    memset(e, 0, sizeof(*e));
    e->from = from;
    e->to = to;
    e->successor_slot = succ_slot;
    e->active = true;
    if (snapshot_edge_id != SIZE_MAX && n->snapshot) {
        norm_edge_payload_init_snapshot(&e->payload, snapshot_edge_id);
    } else {
        if (!norm_edge_payload_set_operands(arena, &e->payload, NULL, 0)) {
            n->n_edges--;
            return SIZE_MAX;
        }
    }
    norm_ensure_block_slots(n, arena, from + 1);
    norm_ensure_block_slots(n, arena, to + 1);
    norm_adj_push(arena, &n->blocks[from].outgoing, eid);
    e->out_position = n->blocks[from].outgoing.n - 1;
    norm_adj_push(arena, &n->blocks[to].incoming, eid);
    e->in_position = n->blocks[to].incoming.n - 1;
    return eid;
}

// --- Graph construction and mutation (M5/M7/M8 rewrite primitives) ---

// Copy reachable CFGInfo into NormalizedCFG with exact adjacency.
// Upstream: graph copy before CFGToSCF mux/latch insertion.
static bool normalized_cfg_build_from_snapshot(CFGAnalysisArena *pool,
                                               const CFGInfo *cfg,
                                               NormalizedCFG *out) {
    memset(out, 0, sizeof(*out));
    if (!pool || !pool->graph_arena || !pool->phase_arena) {
        return false;
    }
    if (!cfg || cfg->n_blocks == 0) return false;
    out->snapshot = cfg;
    out->graph_arena = pool->graph_arena;
    out->phase_arena = pool->phase_arena;
    out->phase_base_pos = pool->phase_base_pos;
    out->entry_block = 0;
    if (!cfg->blocks[0].reachable) return false;

    Arena *arena = pool->graph_arena;
    out->n_blocks = cfg->n_blocks;
    norm_ensure_block_slots(out, arena, cfg->n_blocks);
    for (size_t i = 0; i < cfg->n_blocks; ++i) {
        const CFGBlock *sb = &cfg->blocks[i];
        NormBlock *nb = &out->blocks[i];
        norm_block_init(nb);
        nb->kind = NORM_BLOCK_ORIGINAL;
        nb->source_terminator = sb->terminator;
        nb->term_source = NORM_TERM_SNAPSHOT;
        nb->effective_term_kind = sb->term_kind;
        nb->mlir_block = sb->block;
        nb->active = sb->reachable;
        norm_block_init_original_signature(arena, nb, sb->block);
        if (!sb->reachable) continue;
        norm_adj_prealloc(arena, &nb->outgoing, sb->n_out_edges);
        norm_adj_prealloc(arena, &nb->incoming, sb->n_in_edges);
    }

    for (size_t eid = 0; eid < cfg->n_edges; ++eid) {
        const CFGEdge *se = &cfg->edges[eid];
        if (!out->blocks[se->from].active || !out->blocks[se->to].active) {
            continue;
        }
        normalized_cfg_add_edge(out, arena, se->from, se->to, se->successor_slot,
                                eid);
    }
    return true;
}

// Append synthetic block (mux, latch, shared return, etc.).
static size_t normalized_cfg_add_synthetic_block(NormalizedCFG *n, Arena *arena,
                                                 NormBlockKind kind,
                                                 CFGTermKind term_kind,
                                                 const NormBlockArg *args,
                                                 size_t n_args) {
    size_t id = n->n_blocks++;
    norm_ensure_block_slots(n, arena, n->n_blocks);
    NormBlock *b = &n->blocks[id];
    norm_block_init(b);
    b->kind = kind;
    b->term_source = NORM_TERM_OWNED;
    b->effective_term_kind = term_kind;
    b->normalized_term.kind = term_kind;
    b->mlir_block = MLIR_INVALID_HANDLE;
    b->source_terminator = MLIR_INVALID_HANDLE;
    b->active = true;
    if (!norm_block_set_signature(b, arena, args, n_args)) {
        n->n_blocks--;
        return SIZE_MAX;
    }
    return id;
}

// Install an owned terminator on any active block (original or synthetic).
static bool normalized_cfg_set_block_owned_terminator(NormalizedCFG *n,
                                                      Arena *arena,
                                                      size_t block_id,
                                                      const NormTerminator *term) {
    if (!term) return false;
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (!b->active) return false;
    if (b->kind != NORM_BLOCK_ORIGINAL && b->outgoing_finalized) return false;
    if (!norm_terminator_deep_copy(arena, term, &b->normalized_term)) return false;
    b->term_source = NORM_TERM_OWNED;
    b->effective_term_kind = term->kind;
    return true;
}

// Install mux destination->argument layout on a synthetic block.
static bool normalized_cfg_set_mux_entries(NormalizedCFG *n, Arena *arena,
                                           size_t block_id,
                                           const NormMuxEntry *entries,
                                           size_t n_entries) {
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (b->kind == NORM_BLOCK_ORIGINAL) return false;
    if (!b->active || b->outgoing_finalized) return false;
    if (!norm_block_set_mux_entries(b, arena, entries, n_entries)) return false;
    return true;
}

// Lock outgoing mutation after a synthetic block has been constructed.
// Incoming edges may still be added or redirected (M5 sites, M7/M8 exits).
static bool normalized_cfg_finalize_outgoing(NormalizedCFG *n, size_t block_id) {
    if (block_id >= n->n_blocks) return false;
    NormBlock *b = &n->blocks[block_id];
    if (b->kind == NORM_BLOCK_ORIGINAL) return false;
    if (!b->active || b->outgoing_finalized) return false;
    b->outgoing_finalized = true;
    return true;
}

// Add edge adopting pre-owned operands in graph_arena (no operand copy).
static bool normalized_cfg_add_synthetic_edge_adopt(NormalizedCFG *n,
                                                    size_t from, size_t to,
                                                    size_t succ_slot,
                                                    NormOperand *operands,
                                                    size_t n_operands,
                                                    size_t *out_edge_id) {
    Arena *arena = n->graph_arena;
    if (!arena) return false;
    if (out_edge_id) *out_edge_id = SIZE_MAX;
    if (from >= n->n_blocks || to >= n->n_blocks) return false;
    if (!n->blocks[from].active || !n->blocks[to].active) return false;
    if (!norm_synthetic_block_allows_outgoing_mutation(&n->blocks[from])) {
        return false;
    }
    if (n->n_edges >= n->edges_cap) return false;
    size_t eid = n->n_edges++;
    NormEdge *e = &n->edges[eid];
    memset(e, 0, sizeof(*e));
    e->from = from;
    e->to = to;
    e->successor_slot = succ_slot;
    e->active = true;
    norm_edge_payload_adopt_operands(&e->payload, operands, n_operands);
    norm_ensure_block_slots(n, arena, from + 1);
    norm_ensure_block_slots(n, arena, to + 1);
    norm_adj_push(arena, &n->blocks[from].outgoing, eid);
    e->out_position = n->blocks[from].outgoing.n - 1;
    norm_adj_push(arena, &n->blocks[to].incoming, eid);
    e->in_position = n->blocks[to].incoming.n - 1;
    if (out_edge_id) *out_edge_id = eid;
    return true;
}

// Replace only the operand payload of an active edge.
static bool normalized_cfg_replace_edge_payload(NormalizedCFG *n, Arena *arena,
                                              size_t edge_id,
                                              const NormOperand *operands,
                                              size_t n_operands) {
    if (edge_id >= n->n_edges) return false;
    NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (!norm_edge_payload_set_operands(arena, &e->payload, operands, n_operands)) {
        return false;
    }
    return true;
}

// Retarget edge and adopt pre-owned operands in graph_arena (single payload copy).
static bool normalized_cfg_redirect_edge_adopt_payload(NormalizedCFG *n,
                                                       size_t edge_id,
                                                       size_t new_target,
                                                       NormOperand *operands,
                                                       size_t n_operands) {
    if (edge_id >= n->n_edges) return false;
    NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (new_target >= n->n_blocks || !n->blocks[new_target].active) return false;
    norm_edge_payload_adopt_operands(&e->payload, operands, n_operands);
    Arena *arena = n->graph_arena;
    size_t old_to = e->to;
    if (old_to != new_target) {
        norm_ensure_block_slots(n, arena, new_target + 1);
        if (old_to < n->n_blocks) {
            norm_adj_remove_swap(&n->blocks[old_to].incoming, e->in_position,
                                 n->edges, false);
        }
        e->to = new_target;
        norm_adj_push(arena, &n->blocks[new_target].incoming, edge_id);
        e->in_position = n->blocks[new_target].incoming.n - 1;
    }
    return true;
}

// True when edge and both endpoint blocks are active.
static bool norm_edge_is_active(const NormalizedCFG *n, size_t edge_id) {
    if (edge_id >= n->n_edges) return false;
    const NormEdge *e = &n->edges[edge_id];
    if (!e->active) return false;
    if (e->from >= n->n_blocks || e->to >= n->n_blocks) return false;
    if (!n->blocks[e->from].active || !n->blocks[e->to].active) return false;
    return true;
}

// True when `edge_id` is an active outgoing edge of `block_id` (O(1)).
static bool norm_edge_is_outgoing_from(const NormalizedCFG *n, size_t edge_id,
                                       size_t block_id) {
    return norm_edge_is_active(n, edge_id) && n->edges[edge_id].from == block_id;
}

// Count active outgoing edges (O(degree); prefer norm_out_edges loop).
static size_t normalized_cfg_num_active_out_edges(const NormalizedCFG *n,
                                                  size_t block_id) {
    NormEdgeIter it = norm_out_edges(n, block_id);
    size_t count = 0, eid;
    while (norm_edge_iter_next(&it, &eid)) count++;
    return count;
}

// --- NormalizedCFG read-only queries ---

// Const pointer to edge pool entry `edge_id`.
static const NormEdge *normalized_cfg_edge_at(const NormalizedCFG *n,
                                              size_t edge_id) {
    if (edge_id >= n->n_edges) return NULL;
    return &n->edges[edge_id];
}

// Operand count on normalized edge (lazy for snapshot).
static size_t normalized_cfg_edge_num_operands(const NormalizedCFG *n,
                                               size_t edge_id) {
    const NormEdge *e = normalized_cfg_edge_at(n, edge_id);
    return norm_edge_payload_num_operands(n, e ? &e->payload : NULL);
}

// Resolve edge operand into caller-owned NormOperand (reentrant).
static bool normalized_cfg_edge_operand_resolve(const NormalizedCFG *n,
                                                size_t edge_id, size_t op_idx,
                                                NormOperand *out) {
    const NormEdge *e = normalized_cfg_edge_at(n, edge_id);
    if (!e) return false;
    return norm_edge_payload_operand(n, &e->payload, op_idx, out);
}

// Effective terminator kind for any block (snapshot or owned override).
static CFGTermKind normalized_cfg_block_term_kind(const NormalizedCFG *n,
                                                  size_t block_id) {
    if (block_id >= n->n_blocks) return CFG_TERM_NONE;
    const NormBlock *b = &n->blocks[block_id];
    if (!b->active) return CFG_TERM_NONE;
    return b->effective_term_kind;
}

// Owned terminator metadata when term_source is NORM_TERM_OWNED; else NULL.
static const NormTerminator *normalized_cfg_block_owned_terminator(
        const NormalizedCFG *n, size_t block_id) {
    if (block_id >= n->n_blocks) return NULL;
    const NormBlock *b = &n->blocks[block_id];
    if (!b->active || b->term_source != NORM_TERM_OWNED) return NULL;
    return &b->normalized_term;
}

// ---------------------------------------------------------------------------
// NormGraphView membership (epoch marks for O(1) block/edge queries).
// ---------------------------------------------------------------------------

static bool norm_view_edge_is_hidden(const NormGraphView *view, size_t edge_id) {
    if (!view || !view->hidden_edges) return false;
    for (size_t i = 0; i < view->n_hidden_edges; ++i) {
        if (view->hidden_edges[i] == edge_id) return true;
    }
    return false;
}

static void norm_view_membership_mark(NormViewMembership *m,
                                      const NormalizedCFG *cfg,
                                      const NormGraphView *view) {
    if (!m || !cfg || !view) return;
    if (m->current_epoch == SIZE_MAX) {
        if (m->block_epoch) memset(m->block_epoch, 0, m->n_blocks * sizeof(size_t));
        if (m->edge_epoch) memset(m->edge_epoch, 0, m->n_edges * sizeof(size_t));
        m->current_epoch = 1;
    } else {
        m->current_epoch++;
    }
    size_t epoch = m->current_epoch;
    for (size_t i = 0; i < view->n_blocks; ++i) {
        size_t bid = view->blocks[i];
        if (bid < m->n_blocks) m->block_epoch[bid] = epoch;
    }
    for (size_t i = 0; i < view->n_blocks; ++i) {
        size_t bid = view->blocks[i];
        if (bid >= cfg->n_blocks) continue;
        const NormAdjacency *out_adj = &cfg->blocks[bid].outgoing;
        for (size_t j = 0; j < out_adj->n; ++j) {
            size_t eid = out_adj->edge_ids[j];
            if (eid >= m->n_edges) continue;
            if (!norm_edge_is_active(cfg, eid)) continue;
            if (norm_view_edge_is_hidden(view, eid)) continue;
            size_t to = cfg->edges[eid].to;
            if (to >= m->n_blocks || m->block_epoch[to] != epoch) continue;
            m->edge_epoch[eid] = epoch;
        }
    }
}

static bool norm_view_contains_block(const NormViewMembership *m, size_t bid) {
    if (!m || !m->block_epoch || bid >= m->n_blocks) return false;
    return m->block_epoch[bid] == m->current_epoch;
}

static bool norm_view_contains_edge(const NormalizedCFG *cfg,
                                    const NormGraphView *view,
                                    const NormViewMembership *m,
                                    size_t edge_id) {
    if (!cfg || !m || !m->edge_epoch || edge_id >= m->n_edges) return false;
    if (!norm_edge_is_active(cfg, edge_id)) return false;
    if (norm_view_edge_is_hidden(view, edge_id)) return false;
    return m->edge_epoch[edge_id] == m->current_epoch;
}

static bool m7_view_workspace_grow_dense(M7ViewWorkspace *ws, size_t n_view) {
    if (!ws || !ws->persistent_arena) return false;
    if (n_view <= ws->view_dense_cap) return true;
    Arena *arena = ws->persistent_arena;
    size_t nc = ws->view_dense_cap ? ws->view_dense_cap * 2 : 16;
    while (nc < n_view) nc *= 2;
    size_t *vg = arena_new_array(arena, size_t, nc);
    bool *mark = arena_new_array(arena, bool, nc);
    size_t *idx = arena_new_array(arena, size_t, nc);
    size_t *low = arena_new_array(arena, size_t, nc);
    bool *stk = arena_new_array(arena, bool, nc);
    size_t *tstk = arena_new_array(arena, size_t, nc);
    NormSccFrame *frames = arena_new_array(arena, NormSccFrame, nc);
    if (!vg || !mark || !idx || !low || !stk || !tstk || !frames) return false;
    ws->view_to_global = vg;
    ws->view_mark = mark;
    ws->view_index_of = idx;
    ws->view_lowlink = low;
    ws->view_on_stack = stk;
    ws->view_tarjan_stk = tstk;
    ws->view_frames = frames;
    ws->view_dense_cap = nc;
    return true;
}

static bool m7_workspace_ensure_graph_capacity(M7ViewWorkspace *ws,
                                             size_t n_blocks,
                                             size_t n_edges) {
    if (!ws || !ws->persistent_arena) return false;
    if (n_blocks <= ws->membership.n_blocks &&
        n_edges <= ws->membership.n_edges) {
        return true;
    }

    Arena *arena = ws->persistent_arena;
    size_t nb = ws->membership.n_blocks ? ws->membership.n_blocks : 16;
    while (nb < n_blocks) nb *= 2;
    size_t ne = ws->membership.n_edges ? ws->membership.n_edges : 16;
    while (ne < n_edges) ne *= 2;

    size_t ob = ws->membership.n_blocks;
    size_t oe = ws->membership.n_edges;

    size_t *block_epoch = arena_new_array(arena, size_t, nb);
    size_t *edge_epoch = arena_new_array(arena, size_t, ne);
    size_t *global_to_view = arena_new_array(arena, size_t, nb);
    size_t *target_mark = arena_new_array(arena, size_t, nb);
    size_t *mux_dest_map = arena_new_array(arena, size_t, nb);
    if (!block_epoch || !edge_epoch || !global_to_view || !target_mark ||
        !mux_dest_map) {
        return false;
    }

    if (ob > 0 && ws->membership.block_epoch) {
        memcpy(block_epoch, ws->membership.block_epoch, ob * sizeof(size_t));
        memcpy(edge_epoch, ws->membership.edge_epoch, oe * sizeof(size_t));
        memcpy(global_to_view, ws->global_to_view, ob * sizeof(size_t));
        memcpy(target_mark, ws->target_mark, ob * sizeof(size_t));
        memcpy(mux_dest_map, ws->mux_dest_map, ob * sizeof(size_t));
    }
    memset(block_epoch + ob, 0, (nb - ob) * sizeof(size_t));
    memset(edge_epoch + oe, 0, (ne - oe) * sizeof(size_t));

    for (size_t i = ob; i < nb; ++i) {
        global_to_view[i] = SIZE_MAX;
        target_mark[i] = 0;
        mux_dest_map[i] = SIZE_MAX;
    }

    ws->membership.block_epoch = block_epoch;
    ws->membership.edge_epoch = edge_epoch;
    ws->membership.n_blocks = nb;
    ws->membership.n_edges = ne;
    ws->global_to_view = global_to_view;
    ws->target_mark = target_mark;
    ws->mux_dest_map = mux_dest_map;
    return true;
}

static bool m7_view_workspace_init(M7ViewWorkspace *ws, Arena *arena,
                                   size_t n_blocks, size_t n_edges) {
    memset(ws, 0, sizeof(*ws));
    if (!arena) return true;
    ws->persistent_arena = arena;
    ws->dense_map_view_id = SIZE_MAX;
    if (n_blocks == 0) return true;
    if (!m7_workspace_ensure_graph_capacity(ws, n_blocks, n_edges)) return false;
    ws->target_mark_epoch = 1;
    return true;
}

static void m7_view_workspace_build_dense_map(M7ViewWorkspace *ws,
                                              const NormGraphView *view) {
    if (!ws || !view) return;
    if (ws->dense_map_view_id == view->id) return;
    for (size_t i = 0; i < ws->membership.n_blocks; ++i) {
        ws->global_to_view[i] = SIZE_MAX;
    }
    if (!m7_view_workspace_grow_dense(ws, view->n_blocks)) return;
    for (size_t i = 0; i < view->n_blocks; ++i) {
        size_t bid = view->blocks[i];
        if (bid < ws->membership.n_blocks) {
            ws->global_to_view[bid] = i;
            ws->view_to_global[i] = bid;
        }
    }
    ws->dense_map_view_id = view->id;
}

// Collect every active block into a root view (no hidden edges).
static bool norm_build_root_view(Arena *arena, const NormalizedCFG *cfg,
                                 NormGraphView *out) {
    memset(out, 0, sizeof(*out));
    if (!arena || !cfg) return false;
    size_t n_active = 0;
    for (size_t bid = 0; bid < cfg->n_blocks; ++bid) {
        if (cfg->blocks[bid].active) n_active++;
    }
    if (n_active == 0) return true;
    out->blocks = arena_new_array(arena, size_t, n_active);
    size_t k = 0;
    for (size_t bid = 0; bid < cfg->n_blocks; ++bid) {
        if (!cfg->blocks[bid].active) continue;
        out->blocks[k++] = bid;
    }
    out->n_blocks = n_active;
    out->entry_block = cfg->entry_block;
    out->parent_loop = SIZE_MAX;
    return true;
}

// RPO restricted to a NormGraphView (DFS from view->entry_block).
static bool norm_compute_rpo_view(const NormalizedCFG *n,
                                  const NormGraphView *view,
                                  const NormViewMembership *membership,
                                  M7ViewWorkspace *ws,
                                  Arena *arena,
                                  RPOInfo *out) {
    memset(out, 0, sizeof(*out));
    if (!n || !view || !membership || !ws || !arena) return false;
    if (view->n_blocks == 0) return true;
    if (view->entry_block >= n->n_blocks ||
        !norm_view_contains_block(membership, view->entry_block)) {
        return true;
    }

    m7_view_workspace_build_dense_map(ws, view);
    if (!m7_view_workspace_grow_dense(ws, view->n_blocks)) {
        return false;
    }
    memset(ws->view_mark, 0, view->n_blocks * sizeof(bool));

    size_t *order = arena_new_array(arena, size_t, view->n_blocks);
    size_t post_n = 0;

    NormRpoFrame *stack = arena_new_array(arena, NormRpoFrame, view->n_blocks);
    size_t sp = 0;

    size_t entry_dense = ws->global_to_view[view->entry_block];
    if (entry_dense != SIZE_MAX) ws->view_mark[entry_dense] = true;
    stack[sp].bid = view->entry_block;
    stack[sp].edge_i = 0;
    sp++;

    while (sp > 0) {
        NormRpoFrame *f = &stack[sp - 1];
        const NormAdjacency *out_adj = &n->blocks[f->bid].outgoing;
        bool pushed = false;
        while (f->edge_i < out_adj->n) {
            size_t eid = out_adj->edge_ids[f->edge_i++];
            if (!norm_view_contains_edge(n, view, membership, eid)) continue;
            size_t succ = n->edges[eid].to;
            if (succ >= n->n_blocks) continue;
            size_t sd = ws->global_to_view[succ];
            if (sd == SIZE_MAX || ws->view_mark[sd]) continue;
            ws->view_mark[sd] = true;
            stack[sp].bid = succ;
            stack[sp].edge_i = 0;
            sp++;
            pushed = true;
            break;
        }
        if (!pushed) {
            sp--;
            order[post_n++] = stack[sp].bid;
        }
    }

    out->n = post_n;
    if (post_n == 0) return true;

    for (size_t i = 0, j = post_n - 1; i < j; ++i, --j) {
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    out->order = order;
    out->position = arena_new_array(arena, size_t, view->n_blocks);
    for (size_t i = 0; i < view->n_blocks; ++i) out->position[i] = SIZE_MAX;
    for (size_t i = 0; i < post_n; ++i) {
        size_t d = ws->global_to_view[out->order[i]];
        if (d < view->n_blocks) out->position[d] = i;
    }
    return true;
}

// Iterative Tarjan SCC over active edges in a NormGraphView.
static bool norm_compute_scc_view(const NormalizedCFG *cfg,
                                  const NormGraphView *view,
                                  const NormViewMembership *membership,
                                  M7ViewWorkspace *ws,
                                  Arena *arena,
                                  SCCInfo *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !view || !membership || !ws || !arena) return false;
    out->n = cfg->n_blocks;
    if (view->n_blocks == 0 || cfg->n_blocks == 0) return true;

    size_t n = cfg->n_blocks;
    out->component_id = arena_new_array(arena, size_t, n);
    for (size_t i = 0; i < n; ++i) out->component_id[i] = SIZE_MAX;

    size_t max_components = view->n_blocks;
    size_t *scc_starts_tmp = arena_new_array(arena, size_t, max_components + 1);
    bool   *scc_is_cycle_tmp = arena_new_array(arena, bool, max_components);
    size_t *members_tmp = arena_new_array(arena, size_t, view->n_blocks);
    size_t  n_sccs_out = 0;
    size_t  members_out = 0;

    m7_view_workspace_build_dense_map(ws, view);
    if (!m7_view_workspace_grow_dense(ws, view->n_blocks)) return false;
    size_t *index_of = ws->view_index_of;
    size_t *lowlink = ws->view_lowlink;
    bool   *on_stack = ws->view_on_stack;
    size_t *tarjan_stk = ws->view_tarjan_stk;
    NormSccFrame *frames = ws->view_frames;
    memset(index_of, 0, view->n_blocks * sizeof(size_t));
    memset(lowlink, 0, view->n_blocks * sizeof(size_t));
    memset(on_stack, 0, view->n_blocks * sizeof(bool));

    size_t tarjan_top = 0;
    size_t next_index = 1;

    for (size_t vi = 0; vi < view->n_blocks; ++vi) {
        size_t root = view->blocks[vi];
        if (!norm_view_contains_block(membership, root)) continue;
        size_t rd = ws->global_to_view[root];
        if (rd == SIZE_MAX || index_of[rd]) continue;

        size_t fp = 0;
        index_of[rd] = next_index;
        lowlink[rd] = next_index;
        next_index++;
        tarjan_stk[tarjan_top++] = root;
        on_stack[rd] = true;
        frames[fp].bid = root;
        frames[fp].edge_i = 0;
        fp++;

        while (fp > 0) {
            NormSccFrame *cur = &frames[fp - 1];
            size_t cur_d = ws->global_to_view[cur->bid];
            const NormAdjacency *cur_out = &cfg->blocks[cur->bid].outgoing;
            if (cur->edge_i < cur_out->n) {
                size_t eid = cur_out->edge_ids[cur->edge_i++];
                if (!norm_view_contains_edge(cfg, view, membership, eid)) continue;
                size_t w = cfg->edges[eid].to;
                if (!norm_view_contains_block(membership, w)) continue;
                size_t wd = ws->global_to_view[w];
                if (wd == SIZE_MAX) continue;
                if (index_of[wd] == 0) {
                    index_of[wd] = next_index;
                    lowlink[wd] = next_index;
                    next_index++;
                    tarjan_stk[tarjan_top++] = w;
                    on_stack[wd] = true;
                    frames[fp].bid = w;
                    frames[fp].edge_i = 0;
                    fp++;
                } else if (on_stack[wd]) {
                    if (index_of[wd] < lowlink[cur_d]) {
                        lowlink[cur_d] = index_of[wd];
                    }
                }
                continue;
            }
            size_t v = cur->bid;
            size_t vd = ws->global_to_view[v];
            size_t v_low = lowlink[vd];
            if (lowlink[vd] == index_of[vd]) {
                scc_starts_tmp[n_sccs_out] = members_out;
                size_t scc_size = 0;
                while (tarjan_top > 0) {
                    tarjan_top--;
                    size_t w = tarjan_stk[tarjan_top];
                    size_t wd = ws->global_to_view[w];
                    if (wd != SIZE_MAX) on_stack[wd] = false;
                    out->component_id[w] = n_sccs_out;
                    members_tmp[members_out++] = w;
                    scc_size++;
                    if (w == v) break;
                }
                bool cyclic = (scc_size >= 2);
                if (!cyclic) {
                    const NormAdjacency *v_out = &cfg->blocks[v].outgoing;
                    for (size_t ei = 0; ei < v_out->n; ++ei) {
                        size_t eid = v_out->edge_ids[ei];
                        if (!norm_view_contains_edge(cfg, view, membership, eid)) continue;
                        if (cfg->edges[eid].to == v) {
                            cyclic = true;
                            break;
                        }
                    }
                }
                scc_is_cycle_tmp[n_sccs_out] = cyclic;
                n_sccs_out++;
            }
            fp--;
            if (fp > 0) {
                NormSccFrame *par = &frames[fp - 1];
                size_t pd = ws->global_to_view[par->bid];
                if (pd != SIZE_MAX && v_low < lowlink[pd]) lowlink[pd] = v_low;
            }
        }
    }

    scc_starts_tmp[n_sccs_out] = members_out;
    out->n_components = n_sccs_out;
    if (n_sccs_out == 0) return true;

    out->component_offsets = arena_new_array(arena, size_t, n_sccs_out + 1);
    memcpy(out->component_offsets, scc_starts_tmp,
           (n_sccs_out + 1) * sizeof(size_t));
    out->members = arena_new_array(arena, size_t, members_out);
    memcpy(out->members, members_tmp, members_out * sizeof(size_t));
    out->is_cycle = arena_new_array(arena, bool, n_sccs_out);
    memcpy(out->is_cycle, scc_is_cycle_tmp, n_sccs_out * sizeof(bool));
    return true;
}

static size_t norm_dom_intersect_dense(const RPOInfo *rpo,
                                       const DominanceInfo *dom,
                                       size_t d1, size_t d2) {
    if (!rpo || !rpo->position || !dom || !dom->idom) return SIZE_MAX;
    while (d1 != d2) {
        while (rpo->position[d1] > rpo->position[d2]) {
            d1 = dom->idom[d1];
            if (d1 == SIZE_MAX) return SIZE_MAX;
        }
        while (rpo->position[d2] > rpo->position[d1]) {
            d2 = dom->idom[d2];
            if (d2 == SIZE_MAX) return SIZE_MAX;
        }
    }
    return d1;
}

static bool norm_build_dom_children(const size_t *idom, size_t n, Arena *arena,
                                    size_t **out_offsets, size_t **out_children) {
    size_t *child_count = arena_new_array(arena, size_t, n);
    if (!child_count) return false;
    memset(child_count, 0, n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) {
        size_t p = idom[i];
        if (p != i && p < n) child_count[p]++;
    }

    size_t *offsets = arena_new_array(arena, size_t, n + 1);
    if (!offsets) return false;
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) {
        offsets[i] = total;
        total += child_count[i];
    }
    offsets[n] = total;

    size_t *children = total > 0 ? arena_new_array(arena, size_t, total) : NULL;
    size_t *cursor = arena_new_array(arena, size_t, n);
    if (!cursor || (total > 0 && !children)) return false;
    for (size_t i = 0; i < n; ++i) cursor[i] = offsets[i];
    for (size_t i = 0; i < n; ++i) {
        size_t p = idom[i];
        if (p != i && p < n) {
            children[cursor[p]++] = i;
        }
    }

    *out_offsets = offsets;
    *out_children = children;
    return true;
}

static void norm_dom_label_tree_iter(size_t root, const size_t *child_offsets,
                                     const size_t *children, size_t stack_cap,
                                     size_t *stack_nodes, size_t *stack_phase,
                                     size_t *lo, size_t *hi, size_t *counter) {
    size_t sp = 0;
    stack_nodes[sp] = root;
    stack_phase[sp] = 0;
    sp++;

    while (sp > 0) {
        size_t v = stack_nodes[sp - 1];
        if (stack_phase[sp - 1] == 0) {
            lo[v] = ++(*counter);
            stack_phase[sp - 1] = 1;
            size_t begin = child_offsets[v];
            size_t end = child_offsets[v + 1];
            for (size_t ci = end; ci > begin; --ci) {
                if (sp >= stack_cap) return;
                stack_nodes[sp] = children[ci - 1];
                stack_phase[sp] = 0;
                sp++;
            }
        } else {
            hi[v] = *counter;
            sp--;
        }
    }
}

static bool norm_compute_dominance_view(const NormalizedCFG *cfg,
                                        const NormGraphView *view,
                                        const RPOInfo *rpo,
                                        const NormViewMembership *membership,
                                        M7ViewWorkspace *ws,
                                        Arena *arena,
                                        DominanceInfo *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !view || !rpo || !membership || !ws || !arena) return false;
    if (view->n_blocks == 0 || cfg->n_blocks == 0 || rpo->n == 0) return true;

    size_t entry = view->entry_block;
    if (!norm_view_contains_block(membership, entry)) return true;

    out->root = entry;
    out->n = view->n_blocks;
    out->idom = arena_new_array(arena, size_t, view->n_blocks);
    out->dom_lo = arena_new_array(arena, size_t, view->n_blocks);
    out->dom_hi = arena_new_array(arena, size_t, view->n_blocks);
    if (!out->idom || !out->dom_lo || !out->dom_hi) return false;
    for (size_t i = 0; i < view->n_blocks; ++i) out->idom[i] = SIZE_MAX;
    size_t entry_d = ws->global_to_view[entry];
    if (entry_d == SIZE_MAX) return true;
    out->idom[entry_d] = entry_d;

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t ri = 1; ri < rpo->n; ++ri) {
            size_t bid = rpo->order[ri];
            size_t bd = ws->global_to_view[bid];
            if (bd == SIZE_MAX) continue;

            size_t new_idom = SIZE_MAX;
            const NormAdjacency *in_adj = &cfg->blocks[bid].incoming;
            for (size_t j = 0; j < in_adj->n; ++j) {
                size_t eid = in_adj->edge_ids[j];
                if (!norm_view_contains_edge(cfg, view, membership, eid)) continue;
                size_t pred = cfg->edges[eid].from;
                if (!norm_view_contains_block(membership, pred)) continue;
                size_t pd = ws->global_to_view[pred];
                if (pd == SIZE_MAX || out->idom[pd] == SIZE_MAX) continue;
                if (new_idom == SIZE_MAX) {
                    new_idom = pd;
                } else {
                    new_idom = norm_dom_intersect_dense(rpo, out, new_idom, pd);
                }
            }
            if (new_idom != SIZE_MAX && out->idom[bd] != new_idom) {
                out->idom[bd] = new_idom;
                changed = true;
            }
        }
    }

    size_t counter = 0;
    size_t *child_offsets = NULL;
    size_t *children = NULL;
    if (!norm_build_dom_children(out->idom, view->n_blocks, arena,
                                 &child_offsets, &children)) {
        return false;
    }
    size_t *stk_nodes = arena_new_array(arena, size_t, view->n_blocks);
    size_t *stk_phase = arena_new_array(arena, size_t, view->n_blocks);
    if (!stk_nodes || !stk_phase) return false;
    norm_dom_label_tree_iter(entry_d, child_offsets, children,
                             view->n_blocks, stk_nodes, stk_phase,
                             out->dom_lo, out->dom_hi, &counter);
    return true;
}

static bool norm_dominates(const DominanceInfo *dom,
                           M7ViewWorkspace *ws,
                           size_t lhs_block,
                           size_t rhs_block) {
    if (!dom || !dom->idom || !ws || !dom->dom_lo || !dom->dom_hi) return false;
    size_t ld = ws->global_to_view[lhs_block];
    size_t rd = ws->global_to_view[rhs_block];
    if (ld == SIZE_MAX || rd == SIZE_MAX) return false;
    if (ld >= dom->n || rd >= dom->n) return false;
    return dom->dom_lo[ld] <= dom->dom_lo[rd] &&
           dom->dom_hi[rd] <= dom->dom_hi[ld];
}

// ---------------------------------------------------------------------------
// M5 — return normalization (CFGToSCF.cpp ReturnLikeExitCombiner port).
// Replace llvm.return terminators with virtual branches to shared exit blocks.
// Operates only on NormalizedCFG; LLVM MLIR is never mutated.
// ---------------------------------------------------------------------------

static bool m5_return_signature_matches_class(const M5ExitClass *cls,
                                              M5ExitFlavor flavor,
                                              MLIR_OpHandle ret) {
    if (cls->flavor != flavor) return false;
    size_t n = MLIR_GetOpNumOperands(ret);
    if (cls->n_operand_types != n) return false;
    for (size_t i = 0; i < n; ++i) {
        MLIR_ValueHandle v = MLIR_GetOpOperand(ret, i);
        if (cls->operand_types[i] != MLIR_GetValueType(v)) return false;
    }
    return true;
}

static bool m5_plan_grow_classes(M5Plan *plan, Arena *phase) {
    size_t new_cap = plan->classes_cap ? plan->classes_cap * 2 : 4;
    M5ExitClass *next = arena_new_array(phase, M5ExitClass, new_cap);
    if (!next) return false;
    if (plan->n_classes > 0) {
        memcpy(next, plan->classes, plan->n_classes * sizeof(M5ExitClass));
    }
    plan->classes = next;
    plan->classes_cap = new_cap;
    return true;
}

static size_t m5_plan_find_class_from_return(const M5Plan *plan,
                                             M5ExitFlavor flavor,
                                             MLIR_OpHandle ret) {
    for (size_t i = 0; i < plan->n_classes; ++i) {
        if (m5_return_signature_matches_class(&plan->classes[i], flavor, ret)) {
            return i;
        }
    }
    return SIZE_MAX;
}

static bool m5_plan_add_class_from_return(M5Plan *plan, Arena *phase, Arena *graph,
                                          M5ExitFlavor flavor, MLIR_OpHandle ret,
                                          size_t *out_class_id) {
    if (out_class_id) *out_class_id = SIZE_MAX;
    if (plan->n_classes >= plan->classes_cap && !m5_plan_grow_classes(plan, phase)) {
        return false;
    }

    size_t n_types = MLIR_GetOpNumOperands(ret);
    M5ExitClass candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.flavor = flavor;
    candidate.n_operand_types = n_types;
    candidate.n_sites = 0;
    candidate.shared_exit_block = SIZE_MAX;
    if (n_types > 0) {
        candidate.operand_types = arena_new_array(graph, MLIR_TypeHandle, n_types);
        if (!candidate.operand_types) return false;
        for (size_t i = 0; i < n_types; ++i) {
            candidate.operand_types[i] =
                MLIR_GetValueType(MLIR_GetOpOperand(ret, i));
        }
    }

    size_t class_id = plan->n_classes;
    plan->classes[class_id] = candidate;
    plan->n_classes++;
    if (out_class_id) *out_class_id = class_id;
    return true;
}

static bool m5_plan_grow_sites(M5Plan *plan, Arena *phase) {
    size_t new_cap = plan->sites_cap ? plan->sites_cap * 2 : 8;
    M5ExitSite *next = arena_new_array(phase, M5ExitSite, new_cap);
    if (!next) return false;
    if (plan->n_sites > 0) {
        memcpy(next, plan->sites, plan->n_sites * sizeof(M5ExitSite));
    }
    plan->sites = next;
    plan->sites_cap = new_cap;
    return true;
}

static bool m5_plan_add_site_from_return(M5Plan *plan, Arena *phase, Arena *graph,
                                       size_t source_block, size_t exit_class,
                                       MLIR_OpHandle ret) {
    if (plan->n_sites >= plan->sites_cap && !m5_plan_grow_sites(plan, phase)) {
        return false;
    }

    size_t n_values = MLIR_GetOpNumOperands(ret);
    M5ExitSite *site = &plan->sites[plan->n_sites++];
    site->source_block = source_block;
    site->exit_class = exit_class;
    site->n_return_values = n_values;
    site->return_values = NULL;
    if (n_values > 0) {
        site->return_values = arena_new_array(graph, NormOperand, n_values);
        if (!site->return_values) {
            plan->n_sites--;
            return false;
        }
        for (size_t i = 0; i < n_values; ++i) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(ret, i);
            site->return_values[i] = norm_operand_mlir(MLIR_GetValueType(v), v);
        }
    }
    return true;
}

// Discover return-like sites; no NormalizedCFG mutation.
static bool m5_prepare(NormalizedCFG *cfg, M5Plan *plan) {
    memset(plan, 0, sizeof(*plan));

    Arena *phase = cfg->phase_arena;
    Arena *graph = cfg->graph_arena;
    if (!phase || !graph) return false;

    size_t original_n_blocks = cfg->n_blocks;

    for (size_t bid = 0; bid < original_n_blocks; ++bid) {
        const NormBlock *block = &cfg->blocks[bid];

        if (!block->active ||
            block->kind != NORM_BLOCK_ORIGINAL ||
            block->term_source != NORM_TERM_SNAPSHOT) {
            continue;
        }

        M5ExitFlavor flavor;
        if (block->effective_term_kind == CFG_TERM_RETURN) {
            flavor = M5_EXIT_LLVM_RETURN;
        } else if (block->effective_term_kind == CFG_TERM_UNREACHABLE) {
            flavor = M5_EXIT_LLVM_UNREACHABLE;
        } else {
            continue;
        }

        if (block->outgoing.n != 0) return false;

        MLIR_OpHandle ret = block->source_terminator;
        if (ret == MLIR_INVALID_HANDLE) return false;

        size_t class_id = m5_plan_find_class_from_return(
            plan, flavor, ret);

        if (class_id == SIZE_MAX) {
            if (!m5_plan_add_class_from_return(plan, phase, graph,
                                               flavor, ret,
                                               &class_id)) {
                return false;
            }
        }

        if (!m5_plan_add_site_from_return(plan, phase, graph, bid, class_id,
                                          ret)) {
            return false;
        }

        plan->classes[class_id].n_sites++;
    }

    return true;
}

static bool m5_reserve_commit_capacity(NormalizedCFG *cfg, const M5Plan *plan) {
    if (!normalized_cfg_reserve_blocks(cfg, plan->n_classes)) return false;
    if (!normalized_cfg_reserve_edges(cfg, plan->n_sites)) return false;

    for (size_t i = 0; i < plan->n_sites; ++i) {
        size_t source = plan->sites[i].source_block;
        if (!normalized_cfg_reserve_outgoing(cfg, source, 1)) return false;
    }

    return true;
}

// Create one synthetic shared exit with a virtual terminal operation.
static bool m5_create_shared_exit_block(NormalizedCFG *cfg, M5ExitFlavor flavor,
                                        const MLIR_TypeHandle *types,
                                        size_t n_types, size_t n_incoming_sites,
                                        size_t *out_block_id) {
    if (out_block_id) *out_block_id = SIZE_MAX;
    Arena *arena = cfg->graph_arena;
    if (!arena) return false;

    CFGTermKind term_kind = flavor == M5_EXIT_LLVM_RETURN
        ? CFG_TERM_RETURN : CFG_TERM_UNREACHABLE;
    NormBlockKind block_kind = flavor == M5_EXIT_LLVM_RETURN
        ? NORM_BLOCK_SHARED_RETURN : NORM_BLOCK_SHARED_UNREACHABLE;
    size_t shared_id = normalized_cfg_add_synthetic_block(
        cfg, arena, block_kind, term_kind, NULL, 0);
    if (shared_id == SIZE_MAX) return false;

    NormBlock *nb = &cfg->blocks[shared_id];
    if (!norm_block_set_signature_from_types(nb, arena, types, n_types)) {
        return false;
    }

    norm_adj_prealloc(arena, &nb->incoming, n_incoming_sites);

    norm_terminator_init(&nb->normalized_term);
    nb->normalized_term.kind = term_kind;
    nb->normalized_term.n_return_values =
        term_kind == CFG_TERM_RETURN ? n_types : 0;
    if (term_kind == CFG_TERM_RETURN && n_types > 0) {
        nb->normalized_term.return_values =
            arena_new_array(arena, NormOperand, n_types);
        if (!nb->normalized_term.return_values) return false;
        for (size_t i = 0; i < n_types; ++i) {
            nb->normalized_term.return_values[i] =
                norm_operand_block_arg(types[i], shared_id, i);
        }
    }
    nb->term_source = NORM_TERM_OWNED;
    nb->effective_term_kind = term_kind;

    if (!normalized_cfg_finalize_outgoing(cfg, shared_id)) {
        return false;
    }
    if (out_block_id) *out_block_id = shared_id;
    return true;
}

static bool m5_commit_classes(NormalizedCFG *cfg, M5Plan *plan) {
    for (size_t i = 0; i < plan->n_classes; ++i) {
        M5ExitClass *cls = &plan->classes[i];
        if (!m5_create_shared_exit_block(
                cfg, cls->flavor, cls->operand_types, cls->n_operand_types,
                cls->n_sites, &cls->shared_exit_block)) {
            return false;
        }
    }
    return true;
}

static bool m5_rewrite_return_site(NormalizedCFG *cfg, const M5ExitSite *site,
                                   size_t shared_exit_id) {
    Arena *arena = cfg->graph_arena;
    if (!arena) return false;

    size_t edge_id;
    if (!normalized_cfg_add_synthetic_edge_adopt(
            cfg, site->source_block, shared_exit_id, 0,
            site->return_values, site->n_return_values, &edge_id)) {
        return false;
    }

    NormTerminator branch;
    norm_terminator_init(&branch);
    branch.kind = CFG_TERM_BR;

    if (!normalized_cfg_set_block_owned_terminator(
            cfg, arena, site->source_block, &branch)) {
        return false;
    }
    return true;
}

static bool m5_combiner_adopt_classes(M5ExitCombiner *combiner, Arena *graph,
                                    const M5Plan *plan) {
    if (!combiner || !graph) return false;
    if (plan->n_classes == 0) return true;

    M5ExitClass *owned = arena_new_array(graph, M5ExitClass, plan->n_classes);
    if (!owned) return false;
    memcpy(owned, plan->classes, plan->n_classes * sizeof(M5ExitClass));
    combiner->classes = owned;
    combiner->n_classes = plan->n_classes;
    return true;
}

static void normalized_cfg_reset_phase_storage(NormalizedCFG *cfg) {
    if (cfg->phase_arena) {
        arena_reset(cfg->phase_arena, cfg->phase_base_pos);
    }
}

static bool m5_normalize_returns(NormalizedCFG *cfg, M5ExitCombiner *out_combiner) {
    M5Plan plan;

    memset(out_combiner, 0, sizeof(*out_combiner));

    if (!m5_prepare(cfg, &plan)) return false;

    if (plan.n_sites == 0) return true;

    if (!m5_reserve_commit_capacity(cfg, &plan)) return false;

    bool ok = m5_commit_classes(cfg, &plan);

    for (size_t i = 0; ok && i < plan.n_sites; ++i) {
        M5ExitSite *site = &plan.sites[i];
        M5ExitClass *cls = &plan.classes[site->exit_class];
        ok = m5_rewrite_return_site(cfg, site, cls->shared_exit_block);
    }

    if (!ok) return false;

    if (!m5_combiner_adopt_classes(out_combiner, cfg->graph_arena, &plan)) {
        return false;
    }

    normalized_cfg_reset_phase_storage(cfg);
    return true;
}

// ---------------------------------------------------------------------------
// CFGValueIndex — one function-local value-use map (no repeated MLIR use scans).
// ---------------------------------------------------------------------------

static size_t value_index_map_probe(const ValueIndexMap *map, MLIR_ValueHandle v) {
    if (!map || map->cap == 0 || v == MLIR_INVALID_HANDLE) return SIZE_MAX;
    uintptr_t key = (uintptr_t)v;
    size_t mask = map->cap - 1;
    size_t i = map_hash(key) & mask;
    while (map->keys[i] != 0) {
        if (map->keys[i] == key) return map->vals[i];
        i = (i + 1) & mask;
    }
    return SIZE_MAX;
}

static void value_index_map_init(ValueIndexMap *map, Arena *arena, size_t min_cap) {
    memset(map, 0, sizeof(*map));
    size_t cap = 16;
    while (cap < min_cap) cap *= 2;
    map->cap = cap;
    map->keys = arena_new_array(arena, uintptr_t, cap);
    map->vals = arena_new_array(arena, size_t, cap);
    memset(map->keys, 0, cap * sizeof(uintptr_t));
}

static void value_index_map_grow(ValueIndexMap *map, Arena *arena) {
    size_t ocap = map->cap;
    uintptr_t *okeys = map->keys;
    size_t *ovals = map->vals;
    size_t ncap = ocap ? ocap * 2 : 16;
    map->cap = ncap;
    map->keys = arena_new_array(arena, uintptr_t, ncap);
    map->vals = arena_new_array(arena, size_t, ncap);
    memset(map->keys, 0, ncap * sizeof(uintptr_t));
    map->n = 0;
    size_t mask = ncap - 1;
    for (size_t i = 0; i < ocap; ++i) {
        uintptr_t key = okeys[i];
        if (key == 0) continue;
        size_t j = map_hash(key) & mask;
        while (map->keys[j] != 0) j = (j + 1) & mask;
        map->keys[j] = key;
        map->vals[j] = ovals[i];
        map->n++;
    }
}

static void value_index_map_put(ValueIndexMap *map, Arena *arena,
                                MLIR_ValueHandle v, size_t index) {
    if (v == MLIR_INVALID_HANDLE) return;
    uintptr_t key = (uintptr_t)v;
    if ((map->n + 1) * 4 >= map->cap * 3) value_index_map_grow(map, arena);
    size_t mask = map->cap - 1;
    size_t i = map_hash(key) & mask;
    while (map->keys[i] != 0) {
        if (map->keys[i] == key) return;
        i = (i + 1) & mask;
    }
    map->keys[i] = key;
    map->vals[i] = index;
    map->n++;
}

static bool cfg_value_info_add_use(CFGValueInfo *info, Arena *arena,
                                   size_t block_id) {
    if (info->last_use_block == block_id) return true;
    info->last_use_block = block_id;
    if (info->n_use_blocks >= info->use_blocks_cap) {
        size_t nc = info->use_blocks_cap ? info->use_blocks_cap * 2 : 4;
        size_t *next = arena_new_array(arena, size_t, nc);
        if (info->n_use_blocks > 0) {
            memcpy(next, info->use_blocks,
                   info->n_use_blocks * sizeof(size_t));
        }
        info->use_blocks = next;
        info->use_blocks_cap = nc;
    }
    info->use_blocks[info->n_use_blocks++] = block_id;
    return true;
}

static bool cfg_value_index_grow_values(CFGValueIndex *idx, Arena *arena) {
    size_t nc = idx->values_cap ? idx->values_cap * 2 : 16;
    CFGValueInfo *next = arena_new_array(arena, CFGValueInfo, nc);
    if (idx->n_values > 0) {
        memcpy(next, idx->values, idx->n_values * sizeof(CFGValueInfo));
    }
    idx->values = next;
    idx->values_cap = nc;
    return true;
}

static bool cfg_value_index_register(CFGValueIndex *idx, Arena *arena,
                                     MLIR_ValueHandle v, MLIR_TypeHandle ty,
                                     size_t defining_block) {
    if (v == MLIR_INVALID_HANDLE) return true;
    if (value_index_map_probe(&idx->value_to_index, v) != SIZE_MAX) return true;
    if (idx->n_values >= idx->values_cap &&
        !cfg_value_index_grow_values(idx, arena)) {
        return false;
    }
    size_t vi = idx->n_values;
    CFGValueInfo *info = &idx->values[vi];
    memset(info, 0, sizeof(*info));
    info->value = v;
    info->type = ty;
    info->defining_block = defining_block;
    info->last_use_block = SIZE_MAX;
    value_index_map_put(&idx->value_to_index, arena, v, vi);
    idx->n_values++;
    return true;
}

static void cfg_value_index_record_use(CFGValueIndex *idx, Arena *arena,
                                       MLIR_ValueHandle v,
                                       size_t use_block) {
    size_t vi = value_index_map_probe(&idx->value_to_index, v);
    if (vi == SIZE_MAX) return;
    cfg_value_info_add_use(&idx->values[vi], arena, use_block);
}

static bool cfg_value_index_build_defs_by_block(CFGValueIndex *idx,
                                                Arena *arena,
                                                size_t n_blocks) {
    idx->n_def_blocks = n_blocks;
    idx->defs_by_block_offsets =
        arena_new_array(arena, size_t, n_blocks + 1);
    if (!idx->defs_by_block_offsets) return false;

    size_t *counts = arena_new_array(arena, size_t, n_blocks);
    if (!counts) return false;
    memset(counts, 0, n_blocks * sizeof(size_t));
    for (size_t vi = 0; vi < idx->n_values; ++vi) {
        size_t bid = idx->values[vi].defining_block;
        if (bid < n_blocks) counts[bid]++;
    }

    idx->defs_by_block_offsets[0] = 0;
    for (size_t bid = 0; bid < n_blocks; ++bid) {
        idx->defs_by_block_offsets[bid + 1] =
            idx->defs_by_block_offsets[bid] + counts[bid];
    }

    size_t total = idx->defs_by_block_offsets[n_blocks];
    if (total == 0) return true;

    idx->defs_by_block_indices = arena_new_array(arena, size_t, total);
    if (!idx->defs_by_block_indices) return false;

    size_t *cursor = arena_new_array(arena, size_t, n_blocks);
    if (!cursor) return false;
    for (size_t bid = 0; bid < n_blocks; ++bid) {
        cursor[bid] = idx->defs_by_block_offsets[bid];
    }
    for (size_t vi = 0; vi < idx->n_values; ++vi) {
        size_t bid = idx->values[vi].defining_block;
        if (bid < n_blocks) {
            idx->defs_by_block_indices[cursor[bid]++] = vi;
        }
    }
    return true;
}

static bool cfg_value_index_build(const CFGInfo *cfg, Arena *arena,
                                  CFGValueIndex *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !arena || cfg->n_blocks == 0) return true;

    size_t cap = cfg->n_blocks * 8 + 64;
    out->values = arena_new_array(arena, CFGValueInfo, cap);
    out->values_cap = cap;
    value_index_map_init(&out->value_to_index, arena, cap);

    /* Pass 1: register every definition. */
    for (size_t bid = 0; bid < cfg->n_blocks; ++bid) {
        const CFGBlock *sb = &cfg->blocks[bid];
        if (!sb->reachable) continue;
        MLIR_BlockHandle b = sb->block;

        size_t nargs = MLIR_GetBlockNumArgs(b);
        for (size_t ai = 0; ai < nargs; ++ai) {
            MLIR_ValueHandle arg = MLIR_GetBlockArg(b, ai);
            if (!cfg_value_index_register(out, arena, arg,
                                          MLIR_GetValueType(arg), bid)) {
                return false;
            }
        }

        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t ri = 0; ri < nr; ++ri) {
                MLIR_ValueHandle res = MLIR_GetOpResult(op, ri);
                if (!cfg_value_index_register(out, arena, res,
                                              MLIR_GetValueType(res), bid)) {
                    return false;
                }
            }
        }
    }

    /* Pass 2: record every use. */
    for (size_t bid = 0; bid < cfg->n_blocks; ++bid) {
        const CFGBlock *sb = &cfg->blocks[bid];
        if (!sb->reachable) continue;
        MLIR_BlockHandle b = sb->block;

        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t opi = 0; opi < nop; ++opi) {
                cfg_value_index_record_use(out, arena,
                                           MLIR_GetOpOperand(op, opi), bid);
            }
        }

        MLIR_OpHandle term = sb->terminator;
        if (term != MLIR_INVALID_HANDLE) {
            size_t ns = MLIR_GetOpNumSuccessors(term);
            for (size_t si = 0; si < ns; ++si) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(term, si);
                for (size_t soi = 0; soi < nso; ++soi) {
                    cfg_value_index_record_use(
                        out, arena,
                        MLIR_GetOpSuccessorOperand(term, si, soi), bid);
                }
            }
        }
    }
    return cfg_value_index_build_defs_by_block(out, arena, cfg->n_blocks);
}

static const CFGValueInfo *cfg_value_index_find(const CFGValueIndex *idx,
                                                MLIR_ValueHandle v) {
    size_t vi = value_index_map_probe(&idx->value_to_index, v);
    if (vi == SIZE_MAX) return NULL;
    return &idx->values[vi];
}

// Append formal arguments to an existing synthetic block signature.
static bool norm_block_append_formal_args(NormBlock *nb, Arena *arena,
                                          const MLIR_TypeHandle *types,
                                          size_t n_types) {
    if (n_types == 0) return true;
    size_t new_n = nb->n_args + n_types;
    NormBlockArg *next = arena_new_array(arena, NormBlockArg, new_n);
    if (nb->n_args > 0 && nb->args) {
        memcpy(next, nb->args, nb->n_args * sizeof(NormBlockArg));
    }
    for (size_t i = 0; i < n_types; ++i) {
        next[nb->n_args + i].type = types[i];
    }
    nb->args = next;
    nb->n_args = new_n;
    return true;
}

// ---------------------------------------------------------------------------
// NormEdgeMux — shared edge multiplexer over NormalizedCFG (M7/M8).
// ---------------------------------------------------------------------------

static size_t norm_mux_find_entry(const NormMuxEntry *entries, size_t n,
                                  size_t dest_block) {
    for (size_t i = 0; i < n; ++i) {
        if (entries[i].destination_block == dest_block) return i;
    }
    return SIZE_MAX;
}

static bool norm_edge_mux_prepare(const NormalizedCFG *cfg,
                                  const size_t *edge_ids, size_t n_edges,
                                  const MLIR_TypeHandle *extra_types,
                                  size_t n_extra_types,
                                  MLIR_TypeHandle discriminator_type,
                                  size_t *dest_map,
                                  Arena *phase,
                                  NormEdgeMuxPlan *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !phase || !out) return false;
    if (n_edges == 0) return false;

    out->edge_ids = arena_new_array(phase, size_t, n_edges);
    out->dest_blocks = arena_new_array(phase, size_t, n_edges);
    if (!out->edge_ids || !out->dest_blocks) return false;
    memcpy(out->edge_ids, edge_ids, n_edges * sizeof(size_t));
    out->n_edges = n_edges;

    size_t max_entries = n_edges;
    size_t *distinct = arena_new_array(phase, size_t, max_entries);
    size_t *distinct_nargs = arena_new_array(phase, size_t, max_entries);
    size_t n_distinct = 0;
    size_t total_fwd = 0;

    if (dest_map) {
        for (size_t i = 0; i < n_edges; ++i) {
            size_t eid = edge_ids[i];
            if (eid < cfg->n_edges) {
                size_t dest = cfg->edges[eid].to;
                if (dest < cfg->n_blocks) dest_map[dest] = SIZE_MAX;
            }
        }
    }

    for (size_t i = 0; i < n_edges; ++i) {
        size_t eid = edge_ids[i];
        if (eid >= cfg->n_edges || !norm_edge_is_active(cfg, eid)) return false;
        size_t dest = cfg->edges[eid].to;
        out->dest_blocks[i] = dest;
        size_t found = SIZE_MAX;
        if (dest_map && dest < cfg->n_blocks) {
            found = dest_map[dest];
        } else {
            for (size_t j = 0; j < n_distinct; ++j) {
                if (distinct[j] == dest) { found = j; break; }
            }
        }
        if (found == SIZE_MAX) {
            if (dest >= cfg->n_blocks) return false;
            distinct[n_distinct] = dest;
            distinct_nargs[n_distinct] = cfg->blocks[dest].n_args;
            total_fwd += cfg->blocks[dest].n_args;
            if (dest_map && dest < cfg->n_blocks) {
                dest_map[dest] = n_distinct;
            }
            n_distinct++;
        }
    }

    out->entries = arena_new_array(phase, NormMuxEntry, n_distinct);
    if (!out->entries) return false;
    out->n_entries = n_distinct;
    size_t offset = 0;
    for (size_t i = 0; i < n_distinct; ++i) {
        out->entries[i].destination_block = distinct[i];
        out->entries[i].arg_offset = offset;
        out->entries[i].n_args = distinct_nargs[i];
        out->entries[i].discriminator = (int32_t)i;
        offset += distinct_nargs[i];
    }

    out->discriminator_arg = SIZE_MAX;
    if (n_distinct > 1) {
        out->discriminator_arg = offset;
        offset += 1;
    }
    out->extra_arg_offset = offset;
    out->n_extra_args = n_extra_types;
    if (n_extra_types > 0) {
        out->extra_types = arena_new_array(phase, MLIR_TypeHandle, n_extra_types);
        if (!out->extra_types) return false;
        memcpy(out->extra_types, extra_types, n_extra_types * sizeof(MLIR_TypeHandle));
    }
    offset += n_extra_types;
    out->n_mux_args = offset;
    out->discriminator_type = discriminator_type;
    return true;
}

static bool norm_edge_mux_create(NormalizedCFG *cfg, const NormEdgeMuxPlan *plan,
                                 NormBlockKind kind, NormEdgeMux *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !plan || !out) return false;
    Arena *arena = cfg->graph_arena;
    if (!arena) return false;

    NormBlockArg *sig = arena_new_array(arena, NormBlockArg, plan->n_mux_args);
    if (plan->n_mux_args > 0 && !sig) return false;

    for (size_t i = 0; i < plan->n_entries; ++i) {
        const NormMuxEntry *me = &plan->entries[i];
        if (me->destination_block >= cfg->n_blocks) return false;
        const NormBlock *dest = &cfg->blocks[me->destination_block];
        for (size_t j = 0; j < me->n_args; ++j) {
            sig[me->arg_offset + j].type = dest->args[j].type;
        }
    }
    if (plan->discriminator_arg != SIZE_MAX) {
        sig[plan->discriminator_arg].type = plan->discriminator_type;
    }
    for (size_t i = 0; i < plan->n_extra_args; ++i) {
        sig[plan->extra_arg_offset + i].type = plan->extra_types[i];
    }

    size_t mux_id = normalized_cfg_add_synthetic_block(
        cfg, arena, kind, CFG_TERM_NONE, sig, plan->n_mux_args);
    if (mux_id == SIZE_MAX) return false;

    if (!normalized_cfg_set_mux_entries(cfg, arena, mux_id,
                                        plan->entries, plan->n_entries)) {
        return false;
    }

    out->mux_block = mux_id;
    out->discriminator_arg = plan->discriminator_arg;
    out->extra_arg_offset = plan->extra_arg_offset;
    out->n_extra_args = plan->n_extra_args;
    out->n_mux_args = plan->n_mux_args;
    out->entries = cfg->blocks[mux_id].mux_entries;
    out->n_entries = cfg->blocks[mux_id].n_mux_entries;
    return true;
}

static bool norm_mux_build_redirect_operands(const NormalizedCFG *cfg,
                                             const NormEdgeMux *mux,
                                             size_t edge_id,
                                             size_t orig_dest,
                                             const NormOperand *extra_values,
                                             size_t n_extra_values,
                                             Arena *arena,
                                             NormOperand **out_ops,
                                             size_t *out_n) {
    if (!cfg || !mux || !arena || !out_ops || !out_n) return false;
    *out_ops = NULL;
    *out_n = 0;
    if (mux->n_mux_args == 0) return true;
    if (n_extra_values != mux->n_extra_args) return false;

    size_t entry_idx = norm_mux_find_entry(mux->entries, mux->n_entries, orig_dest);
    if (entry_idx == SIZE_MAX && orig_dest == SIZE_MAX && mux->n_entries > 0) {
        entry_idx = 0;
    }
    if (entry_idx == SIZE_MAX) return false;
    const NormMuxEntry *me = &mux->entries[entry_idx];

    NormOperand *ops = arena_new_array(arena, NormOperand, mux->n_mux_args);
    if (!ops) return false;

    for (size_t i = 0; i < mux->n_mux_args; ++i) {
        ops[i] = norm_operand_undef(
            cfg->blocks[mux->mux_block].args[i].type);
    }

    for (size_t j = 0; j < me->n_args; ++j) {
        NormOperand op;
        if (!norm_edge_payload_operand(cfg, &cfg->edges[edge_id].payload, j, &op)) {
            return false;
        }
        ops[me->arg_offset + j] = op;
    }

    if (mux->discriminator_arg != SIZE_MAX) {
        ops[mux->discriminator_arg] =
            norm_operand_discriminator(ops[mux->discriminator_arg].type,
                                       me->discriminator);
    }

    for (size_t i = 0; i < mux->n_extra_args; ++i) {
        ops[mux->extra_arg_offset + i] = extra_values[i];
    }

    *out_ops = ops;
    *out_n = mux->n_mux_args;
    return true;
}

static bool norm_edge_mux_redirect(NormalizedCFG *cfg, const NormEdgeMux *mux,
                                   size_t edge_id, size_t orig_dest,
                                   const NormOperand *extra_values,
                                   size_t n_extra_values) {
    Arena *phase = cfg->phase_arena;
    Arena *graph = cfg->graph_arena;
    if (!phase || !graph) return false;
    NormOperand *tmp = NULL;
    size_t n_ops = 0;
    if (!norm_mux_build_redirect_operands(cfg, mux, edge_id, orig_dest,
                                          extra_values, n_extra_values,
                                          phase, &tmp, &n_ops)) {
        return false;
    }
    if (n_ops == 0) {
        return normalized_cfg_redirect_edge_adopt_payload(
            cfg, edge_id, mux->mux_block, NULL, 0);
    }
    NormOperand *owned = arena_new_array(graph, NormOperand, n_ops);
    if (!owned) return false;
    memcpy(owned, tmp, n_ops * sizeof(NormOperand));
    return normalized_cfg_redirect_edge_adopt_payload(
        cfg, edge_id, mux->mux_block, owned, n_ops);
}

static bool norm_edge_mux_create_dispatch(NormalizedCFG *cfg,
                                          const NormEdgeMux *mux,
                                          size_t dispatch_block,
                                          size_t excluded_destination) {
    if (!cfg || !mux) return false;
    Arena *arena = cfg->graph_arena;
    if (dispatch_block >= cfg->n_blocks) return false;
    NormBlock *dispatch_b = &cfg->blocks[dispatch_block];

    size_t n_pick = 0;
    for (size_t i = 0; i < mux->n_entries; ++i) {
        if (mux->entries[i].destination_block != excluded_destination) {
            n_pick++;
        }
    }
    if (n_pick == 0) return false;

    if (n_pick == 1) {
        size_t pick_dest = SIZE_MAX;
        size_t pick_idx = SIZE_MAX;
        for (size_t i = 0; i < mux->n_entries; ++i) {
            if (mux->entries[i].destination_block != excluded_destination) {
                pick_dest = mux->entries[i].destination_block;
                pick_idx = i;
                break;
            }
        }
        const NormMuxEntry *me = &mux->entries[pick_idx];
        NormOperand *ops = me->n_args ? arena_new_array(arena, NormOperand, me->n_args) : NULL;
        for (size_t j = 0; j < me->n_args; ++j) {
            ops[j] = norm_operand_block_arg(
                dispatch_b->args[me->arg_offset + j].type,
                dispatch_block, me->arg_offset + j);
        }
        size_t eid = SIZE_MAX;
        if (!normalized_cfg_add_synthetic_edge_adopt(
                cfg, dispatch_block, pick_dest, 0, ops, me->n_args, &eid)) {
            return false;
        }
        NormTerminator term;
        norm_terminator_init(&term);
        term.kind = CFG_TERM_BR;
        term.default_edge_id = eid;
        return normalized_cfg_set_block_owned_terminator(
            cfg, arena, dispatch_block, &term);
    }

    NormTerminator term;
    norm_terminator_init(&term);
    term.kind = CFG_TERM_SWITCH;
    if (mux->discriminator_arg != SIZE_MAX) {
        term.selector = norm_operand_block_arg(
            dispatch_b->args[mux->discriminator_arg].type,
            dispatch_block, mux->discriminator_arg);
    } else {
        term.selector = norm_operand_discriminator(
            dispatch_b->args[0].type, 0);
    }

    term.n_cases = n_pick - 1;
    term.case_values = arena_new_array(arena, int32_t, term.n_cases);
    term.case_edge_ids = arena_new_array(arena, size_t, term.n_cases);

    size_t *pick_indices = arena_new_array(arena, size_t, n_pick);
    size_t pick_n = 0;
    for (size_t i = 0; i < mux->n_entries; ++i) {
        if (mux->entries[i].destination_block != excluded_destination) {
            pick_indices[pick_n++] = i;
        }
    }

    size_t default_idx = pick_indices[pick_n - 1];
    const NormMuxEntry *default_me = &mux->entries[default_idx];
    NormOperand *default_ops = default_me->n_args
        ? arena_new_array(arena, NormOperand, default_me->n_args) : NULL;
    for (size_t j = 0; j < default_me->n_args; ++j) {
        default_ops[j] = norm_operand_block_arg(
            dispatch_b->args[default_me->arg_offset + j].type,
            dispatch_block, default_me->arg_offset + j);
    }
    size_t default_eid = SIZE_MAX;
    if (!normalized_cfg_add_synthetic_edge_adopt(
            cfg, dispatch_block, default_me->destination_block,
            (size_t)(term.n_cases), default_ops, default_me->n_args,
            &default_eid)) {
        return false;
    }
    term.default_edge_id = default_eid;

    for (size_t ci = 0; ci < term.n_cases; ++ci) {
        const NormMuxEntry *me = &mux->entries[pick_indices[ci]];
        NormOperand *ops = me->n_args ? arena_new_array(arena, NormOperand, me->n_args) : NULL;
        for (size_t j = 0; j < me->n_args; ++j) {
            ops[j] = norm_operand_block_arg(
                dispatch_b->args[me->arg_offset + j].type,
                dispatch_block, me->arg_offset + j);
        }
        size_t eid = SIZE_MAX;
        if (!normalized_cfg_add_synthetic_edge_adopt(
                cfg, dispatch_block, me->destination_block, ci,
                ops, me->n_args, &eid)) {
            return false;
        }
        term.case_values[ci] = me->discriminator;
        term.case_edge_ids[ci] = eid;
    }
    return normalized_cfg_set_block_owned_terminator(
        cfg, arena, dispatch_block, &term);
}

// ---------------------------------------------------------------------------
// M7 cycle-edge classification and loop normalization driver.
// ---------------------------------------------------------------------------

static bool m7_member_in_component(const SCCInfo *scc, size_t component,
                                   size_t bid) {
    if (!scc || bid >= scc->n) return false;
    return scc->component_id[bid] == component;
}

static bool m7_push_u64(Arena *arena, size_t **items, size_t *n, size_t *cap,
                        size_t v) {
    if (*n >= *cap) {
        size_t nc = *cap ? *cap * 2 : 8;
        size_t *next = arena_new_array(arena, size_t, nc);
        if (*n > 0) memcpy(next, *items, *n * sizeof(size_t));
        *items = next;
        *cap = nc;
    }
    (*items)[(*n)++] = v;
    return true;
}

static bool m7_push_target_marked(M7ViewWorkspace *ws, Arena *arena,
                                  size_t **targets, size_t *n, size_t *cap,
                                  size_t bid) {
    if (!ws || bid >= ws->membership.n_blocks) return false;
    if (ws->target_mark[bid] == ws->target_mark_epoch) return true;
    ws->target_mark[bid] = ws->target_mark_epoch;
    return m7_push_u64(arena, targets, n, cap, bid);
}

static bool m7_push_target_unique(Arena *arena, size_t **targets, size_t *n,
                                  size_t *cap, size_t bid) {
    for (size_t i = 0; i < *n; ++i) {
        if ((*targets)[i] == bid) return true;
    }
    return m7_push_u64(arena, targets, n, cap, bid);
}

static bool m7_collect_cycle_edges(const NormalizedCFG *cfg,
                                   const NormGraphView *view,
                                   const SCCInfo *scc,
                                   size_t component,
                                   const NormViewMembership *membership,
                                   M7ViewWorkspace *ws,
                                   Arena *phase,
                                   M7CycleEdges *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !view || !scc || !membership || !phase) return false;
    if (component >= scc->n_components) return false;

    if (ws) {
        ws->target_mark_epoch++;
        if (ws->target_mark_epoch == SIZE_MAX) {
            memset(ws->target_mark, 0,
                   ws->membership.n_blocks * sizeof(size_t));
            ws->target_mark_epoch = 1;
        }
    }

    size_t off = scc->component_offsets[component];
    size_t end = scc->component_offsets[component + 1];
    out->n_members = end - off;
    if (out->n_members == 0) return false;

    out->members = arena_new_array(phase, size_t, out->n_members);
    if (!out->members) return false;
    memcpy(out->members, &scc->members[off], out->n_members * sizeof(size_t));

    size_t cap_e = out->n_members * 4 + 4;
    size_t cap_t = out->n_members + 4;
    out->entry_edges = arena_new_array(phase, size_t, cap_e);
    out->exit_edges = arena_new_array(phase, size_t, cap_e);
    out->back_edges = arena_new_array(phase, size_t, cap_e);
    out->entry_targets = arena_new_array(phase, size_t, cap_t);
    size_t cap_entry = cap_e;
    size_t cap_exit = cap_e;
    size_t cap_back = cap_e;
    size_t cap_targets = cap_t;

    for (size_t mi = 0; mi < out->n_members; ++mi) {
        size_t bid = out->members[mi];
        NormEdgeIter in_it = norm_in_edges(cfg, bid);
        size_t eid;
        while (norm_edge_iter_next(&in_it, &eid)) {
            if (!norm_view_contains_edge(cfg, view, membership, eid)) continue;
            size_t from = cfg->edges[eid].from;
            if (!m7_member_in_component(scc, component, from)) {
                if (!m7_push_u64(phase, &out->entry_edges, &out->n_entry_edges,
                                 &cap_entry, eid)) return false;
                if (ws) {
                    if (!m7_push_target_marked(ws, phase, &out->entry_targets,
                                               &out->n_entry_targets,
                                               &cap_targets, bid)) {
                        return false;
                    }
                } else if (!m7_push_target_unique(phase, &out->entry_targets,
                                                  &out->n_entry_targets,
                                                  &cap_targets, bid)) {
                    return false;
                }
            }
        }
        NormEdgeIter out_it = norm_out_edges(cfg, bid);
        while (norm_edge_iter_next(&out_it, &eid)) {
            if (!norm_view_contains_edge(cfg, view, membership, eid)) continue;
            size_t to = cfg->edges[eid].to;
            if (!m7_member_in_component(scc, component, to)) {
                if (!m7_push_u64(phase, &out->exit_edges, &out->n_exit_edges,
                                 &cap_exit, eid)) return false;
            }
        }
    }

    if (out->n_entry_edges == 0 &&
        view->entry_block < cfg->n_blocks &&
        m7_member_in_component(scc, component, view->entry_block)) {
        if (ws) {
            if (!m7_push_target_marked(ws, phase, &out->entry_targets,
                                       &out->n_entry_targets, &cap_targets,
                                       view->entry_block)) {
                return false;
            }
        } else if (!m7_push_target_unique(phase, &out->entry_targets,
                                          &out->n_entry_targets, &cap_targets,
                                          view->entry_block)) {
            return false;
        }
    }

    for (size_t mi = 0; mi < out->n_members; ++mi) {
        size_t bid = out->members[mi];
        NormEdgeIter out_it = norm_out_edges(cfg, bid);
        size_t eid;
        while (norm_edge_iter_next(&out_it, &eid)) {
            if (!norm_view_contains_edge(cfg, view, membership, eid)) continue;
            size_t to = cfg->edges[eid].to;
            if (!m7_member_in_component(scc, component, to)) continue;
            for (size_t ti = 0; ti < out->n_entry_targets; ++ti) {
                if (out->entry_targets[ti] == to) {
                    if (!m7_push_u64(phase, &out->back_edges, &out->n_back_edges,
                                     &cap_back, eid)) return false;
                    break;
                }
            }
        }
    }

    return out->n_back_edges > 0;
}

static size_t m7_mux_interface_arg_count(const NormEdgeMuxPlan *mux);

static bool m7_predict_header(const NormalizedCFG *cfg,
                              const NormGraphView *view,
                              M7CyclePlan *plan) {
    plan->needs_entry_mux = plan->edges.n_entry_edges > 1;
    if (!plan->needs_entry_mux) {
        if (plan->edges.n_entry_edges == 1) {
            plan->predicted_header =
                cfg->edges[plan->edges.entry_edges[0]].to;
        } else if (plan->edges.n_entry_edges == 0) {
            plan->predicted_header = view->entry_block;
        } else {
            return false;
        }
        if (plan->predicted_header >= cfg->n_blocks) return false;
        plan->base_header_nargs =
            cfg->blocks[plan->predicted_header].n_args;
    } else {
        plan->predicted_header = SIZE_MAX;
        plan->base_header_nargs = m7_mux_interface_arg_count(&plan->entry_mux);
    }
    return true;
}

static bool m7_prepare_latch_mux(const NormalizedCFG *cfg,
                                 size_t header,
                                 const M7CycleEdges *edges,
                                 const MLIR_TypeHandle *extra_types,
                                 size_t n_extra_types,
                                 MLIR_TypeHandle discriminator_type,
                                 Arena *phase,
                                 NormEdgeMuxPlan *out);

static bool m7_build_latch_mux_plan(const NormalizedCFG *cfg,
                                    M7CyclePlan *plan,
                                    const M7Config *config,
                                    Arena *phase) {
    size_t n_latch = plan->edges.n_back_edges + plan->edges.n_exit_edges;
    if (n_latch == 0) return false;
    MLIR_TypeHandle extra[1] = { config->flag_type };

    if (plan->predicted_header != SIZE_MAX) {
        return m7_prepare_latch_mux(cfg, plan->predicted_header, &plan->edges,
                                    extra, 1, config->flag_type, phase,
                                    &plan->latch_mux);
    }

    size_t *edge_ids = arena_new_array(phase, size_t, n_latch);
    size_t k = 0;
    for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
        edge_ids[k++] = plan->edges.back_edges[i];
    }
    for (size_t i = 0; i < plan->edges.n_exit_edges; ++i) {
        edge_ids[k++] = plan->edges.exit_edges[i];
    }

    size_t max_exits = plan->edges.n_exit_edges;
    size_t *exit_dests = arena_new_array(phase, size_t, max_exits);
    size_t *exit_nargs = arena_new_array(phase, size_t, max_exits);
    size_t n_exit_distinct = 0;
    for (size_t i = 0; i < plan->edges.n_exit_edges; ++i) {
        size_t dest = cfg->edges[plan->edges.exit_edges[i]].to;
        bool seen = false;
        for (size_t j = 0; j < n_exit_distinct; ++j) {
            if (exit_dests[j] == dest) { seen = true; break; }
        }
        if (seen) continue;
        exit_dests[n_exit_distinct] = dest;
        exit_nargs[n_exit_distinct] = cfg->blocks[dest].n_args;
        n_exit_distinct++;
    }

    size_t header_nargs = plan->entry_mux.n_mux_args;
    size_t n_entries = 1 + n_exit_distinct;
    NormMuxEntry *entries = arena_new_array(phase, NormMuxEntry, n_entries);
    entries[0].destination_block = SIZE_MAX;
    entries[0].arg_offset = 0;
    entries[0].n_args = header_nargs;
    entries[0].discriminator = 0;
    size_t offset = header_nargs;
    for (size_t i = 0; i < n_exit_distinct; ++i) {
        entries[i + 1].destination_block = exit_dests[i];
        entries[i + 1].arg_offset = offset;
        entries[i + 1].n_args = exit_nargs[i];
        entries[i + 1].discriminator = (int32_t)(i + 1);
        offset += exit_nargs[i];
    }

    memset(&plan->latch_mux, 0, sizeof(plan->latch_mux));
    plan->latch_mux.edge_ids = edge_ids;
    plan->latch_mux.dest_blocks = arena_new_array(phase, size_t, n_latch);
    for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
        plan->latch_mux.dest_blocks[i] = SIZE_MAX;
    }
    for (size_t i = 0; i < plan->edges.n_exit_edges; ++i) {
        plan->latch_mux.dest_blocks[plan->edges.n_back_edges + i] =
            cfg->edges[plan->edges.exit_edges[i]].to;
    }
    plan->latch_mux.n_edges = n_latch;
    plan->latch_mux.entries = entries;
    plan->latch_mux.n_entries = n_entries;
    plan->latch_mux.discriminator_arg = SIZE_MAX;
    if (n_entries > 1) {
        plan->latch_mux.discriminator_arg = offset;
        offset += 1;
    }
    plan->latch_mux.extra_arg_offset = offset;
    plan->latch_mux.n_extra_args = 1;
    plan->latch_mux.extra_types = arena_new_array(phase, MLIR_TypeHandle, 1);
    plan->latch_mux.extra_types[0] = config->flag_type;
    plan->latch_mux.discriminator_type = config->flag_type;
    offset += 1;
    plan->latch_mux.n_mux_args = offset;
    return true;
}

static bool m7_value_in_iteration_basis(const NormalizedCFG *cfg,
                                        const M7CyclePlan *plan,
                                        MLIR_ValueHandle v) {
    if (v == MLIR_INVALID_HANDLE || !plan) return false;
    for (size_t i = 0; i < plan->n_base_iterations; ++i) {
        if (plan->base_iteration_values[i] == v) return true;
    }
    if (cfg && !plan->needs_entry_mux &&
        plan->predicted_header < cfg->n_blocks) {
        const NormBlock *header = &cfg->blocks[plan->predicted_header];
        if (header->kind == NORM_BLOCK_ORIGINAL &&
            header->mlir_block != MLIR_INVALID_HANDLE) {
            size_t n_args = MLIR_GetBlockNumArgs(header->mlir_block);
            if (n_args > plan->n_base_iterations) {
                n_args = plan->n_base_iterations;
            }
            for (size_t i = 0; i < n_args; ++i) {
                if (MLIR_GetBlockArg(header->mlir_block, i) == v) return true;
            }
        }
    }
    return false;
}

static size_t m7_mux_interface_arg_count(const NormEdgeMuxPlan *mux) {
    if (!mux || mux->n_mux_args == 0) return 0;
    if (mux->n_extra_args > 0) return mux->extra_arg_offset;
    return mux->n_mux_args;
}

static bool m7_mux_carried_types(const NormalizedCFG *cfg,
                                 const NormEdgeMuxPlan *mux,
                                 size_t n_carried,
                                 MLIR_TypeHandle *types) {
    if (!cfg || !mux || !types || n_carried == 0) return true;
    for (size_t i = 0; i < mux->n_entries; ++i) {
        const NormMuxEntry *e = &mux->entries[i];
        if (e->destination_block >= cfg->n_blocks) continue;
        const NormBlock *dest = &cfg->blocks[e->destination_block];
        for (size_t j = 0; j < e->n_args; ++j) {
            size_t idx = e->arg_offset + j;
            if (idx >= n_carried || j >= dest->n_args) continue;
            types[idx] = dest->args[j].type;
        }
    }
    return true;
}

static bool m7_mux_interface_types(const NormalizedCFG *cfg,
                                   const NormEdgeMuxPlan *mux,
                                   size_t n_iface,
                                   MLIR_TypeHandle *types) {
    if (!mux || !types || n_iface == 0) return true;
    if (!m7_mux_carried_types(cfg, mux, n_iface, types)) return false;
    if (mux->discriminator_arg != SIZE_MAX &&
        mux->discriminator_arg < n_iface) {
        types[mux->discriminator_arg] = mux->discriminator_type;
    }
    if (mux->n_extra_args > 0) {
        for (size_t i = 0; i < mux->n_extra_args; ++i) {
            size_t idx = mux->extra_arg_offset + i;
            if (idx < n_iface) types[idx] = mux->extra_types[i];
        }
    }
    return true;
}

static bool m7_snapshot_iteration_basis(const NormalizedCFG *cfg,
                                        M7CyclePlan *plan,
                                        Arena *phase) {
    if (!cfg || !plan || plan->edges.n_back_edges == 0) return false;

    size_t n_carried = 0;
    MLIR_TypeHandle *types = NULL;
    MLIR_ValueHandle *values = NULL;

    if (plan->needs_entry_mux) {
        n_carried = m7_mux_interface_arg_count(&plan->entry_mux);
        if (n_carried == 0) {
            plan->n_base_iterations = 0;
            plan->base_header_nargs = 0;
            return true;
        }
        types = arena_new_array(phase, MLIR_TypeHandle, n_carried);
        values = arena_new_array(phase, MLIR_ValueHandle, n_carried);
        if (!types || !values) return false;
        for (size_t i = 0; i < n_carried; ++i) {
            values[i] = MLIR_INVALID_HANDLE;
        }
        if (!m7_mux_interface_types(cfg, &plan->entry_mux, n_carried, types)) {
            return false;
        }

        for (size_t bi = 0; bi < plan->edges.n_back_edges; ++bi) {
            size_t eid = plan->edges.back_edges[bi];
            size_t mux_edge_idx = plan->edges.n_entry_edges + bi;
            if (mux_edge_idx >= plan->entry_mux.n_edges) continue;
            size_t mux_dest = plan->entry_mux.dest_blocks[mux_edge_idx];
            for (size_t ei = 0; ei < plan->entry_mux.n_entries; ++ei) {
                const NormMuxEntry *me = &plan->entry_mux.entries[ei];
                if (me->destination_block != mux_dest) continue;
                size_t n_ops =
                    norm_edge_payload_num_operands(cfg, &cfg->edges[eid].payload);
                for (size_t j = 0; j < me->n_args && j < n_ops; ++j) {
                    size_t idx = me->arg_offset + j;
                    if (idx >= n_carried) continue;
                    NormOperand op;
                    if (!norm_edge_payload_operand(cfg, &cfg->edges[eid].payload, j,
                                                   &op)) {
                        return false;
                    }
                    types[idx] = op.type;
                    if (op.kind == NORM_OPERAND_MLIR) {
                        values[idx] = op.as.mlir_value;
                    }
                }
                break;
            }
        }
        plan->base_header_nargs = n_carried;
    } else {
        size_t eid = plan->edges.back_edges[0];
        n_carried =
            norm_edge_payload_num_operands(cfg, &cfg->edges[eid].payload);
        if (n_carried == 0) {
            plan->n_base_iterations = 0;
            return true;
        }

        types = arena_new_array(phase, MLIR_TypeHandle, n_carried);
        values = arena_new_array(phase, MLIR_ValueHandle, n_carried);
        if (!types || !values) return false;

        for (size_t i = 0; i < n_carried; ++i) {
            NormOperand op;
            if (!norm_edge_payload_operand(cfg, &cfg->edges[eid].payload, i,
                                           &op)) {
                return false;
            }
            types[i] = op.type;
            values[i] =
                (op.kind == NORM_OPERAND_MLIR) ? op.as.mlir_value
                                               : MLIR_INVALID_HANDLE;
        }
    }

    plan->n_base_iterations = n_carried;
    plan->base_iteration_types = types;
    plan->base_iteration_values = values;
    return true;
}

static bool m7_build_base_exit_passes(const NormalizedCFG *cfg,
                                      const CFGValueIndex *values,
                                      const DominanceInfo *dom,
                                      M7ViewWorkspace *ws,
                                      M7CyclePlan *plan,
                                      Arena *phase) {
    size_t n_base = plan->n_base_iterations;
    size_t n_exits = plan->edges.n_exit_edges;
    if (n_base == 0 || n_exits == 0) {
        plan->base_exit_passes = NULL;
        return true;
    }
    plan->base_exit_passes = arena_new_array(
        phase, bool, n_base * n_exits);
    if (!plan->base_exit_passes) return false;
    for (size_t i = 0; i < n_base; ++i) {
        const CFGValueInfo *info = cfg_value_index_find(
            values, plan->base_iteration_values[i]);
        for (size_t j = 0; j < n_exits; ++j) {
            size_t pred = cfg->edges[plan->edges.exit_edges[j]].from;
            plan->base_exit_passes[i * n_exits + j] =
                info && norm_dominates(dom, ws, info->defining_block, pred);
        }
    }
    return true;
}

static bool m7_find_additional_live_outs(const NormalizedCFG *cfg,
                                         const SCCInfo *scc,
                                         size_t component,
                                         const CFGValueIndex *values,
                                         const DominanceInfo *dom,
                                         M7ViewWorkspace *ws,
                                         M7CyclePlan *plan,
                                         Arena *phase) {
    size_t esc_cap = 8;
    plan->additional_values = arena_new_array(phase, CFGValueInfo *, esc_cap);
    plan->additional_forward = arena_new_array(phase, bool, esc_cap);
    plan->n_additional_values = 0;

    plan->n_latch_sources =
        plan->edges.n_back_edges + plan->edges.n_exit_edges;
    plan->latch_source_edges = arena_new_array(phase, size_t,
                                              plan->n_latch_sources);
    size_t ls = 0;
    for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
        plan->latch_source_edges[ls++] = plan->edges.back_edges[i];
    }
    for (size_t i = 0; i < plan->edges.n_exit_edges; ++i) {
        plan->latch_source_edges[ls++] = plan->edges.exit_edges[i];
    }

    bool use_def_index =
        values->defs_by_block_offsets && values->defs_by_block_indices;

    if (use_def_index) {
        for (size_t mi = 0; mi < plan->edges.n_members; ++mi) {
            size_t bid = plan->edges.members[mi];
            if (bid + 1 > values->n_def_blocks) continue;
            size_t off = values->defs_by_block_offsets[bid];
            size_t end = values->defs_by_block_offsets[bid + 1];
            for (size_t di = off; di < end; ++di) {
                size_t vi = values->defs_by_block_indices[di];
                CFGValueInfo *info = &values->values[vi];
                if (!m7_member_in_component(scc, component,
                                            info->defining_block)) {
                    continue;
                }
                if (m7_value_in_iteration_basis(cfg, plan, info->value)) continue;

                bool escapes = false;
                for (size_t ui = 0; ui < info->n_use_blocks; ++ui) {
                    if (!m7_member_in_component(scc, component,
                                                info->use_blocks[ui])) {
                        escapes = true;
                        break;
                    }
                }
                if (!escapes) continue;

                if (plan->n_additional_values >= esc_cap) {
                    esc_cap *= 2;
                    CFGValueInfo **next =
                        arena_new_array(phase, CFGValueInfo *, esc_cap);
                    bool *fnext = arena_new_array(phase, bool, esc_cap);
                    memcpy(next, plan->additional_values,
                           plan->n_additional_values * sizeof(CFGValueInfo *));
                    memcpy(fnext, plan->additional_forward,
                           plan->n_additional_values * sizeof(bool));
                    plan->additional_values = next;
                    plan->additional_forward = fnext;
                }
                plan->additional_values[plan->n_additional_values] = info;
                plan->additional_forward[plan->n_additional_values] = true;
                plan->n_additional_values++;
            }
        }
    } else {
        for (size_t vi = 0; vi < values->n_values; ++vi) {
            CFGValueInfo *info = &values->values[vi];
            if (!m7_member_in_component(scc, component, info->defining_block)) {
                continue;
            }
            if (m7_value_in_iteration_basis(cfg, plan, info->value)) continue;

            bool escapes = false;
            for (size_t ui = 0; ui < info->n_use_blocks; ++ui) {
                if (!m7_member_in_component(scc, component,
                                            info->use_blocks[ui])) {
                    escapes = true;
                    break;
                }
            }
            if (!escapes) continue;

            if (plan->n_additional_values >= esc_cap) {
                esc_cap *= 2;
                CFGValueInfo **next =
                    arena_new_array(phase, CFGValueInfo *, esc_cap);
                bool *fnext = arena_new_array(phase, bool, esc_cap);
                memcpy(next, plan->additional_values,
                       plan->n_additional_values * sizeof(CFGValueInfo *));
                memcpy(fnext, plan->additional_forward,
                       plan->n_additional_values * sizeof(bool));
                plan->additional_values = next;
                plan->additional_forward = fnext;
            }
            plan->additional_values[plan->n_additional_values] = info;
            plan->additional_forward[plan->n_additional_values] = true;
            plan->n_additional_values++;
        }
    }

    if (plan->n_additional_values > 0 && plan->n_latch_sources > 0) {
        size_t mat_n = plan->n_additional_values * plan->n_latch_sources;
        plan->additional_latch_passes = arena_new_array(phase, bool, mat_n);
        for (size_t i = 0; i < plan->n_additional_values; ++i) {
            CFGValueInfo *info = plan->additional_values[i];
            bool forward = true;
            for (size_t j = 0; j < plan->n_latch_sources; ++j) {
                size_t pred =
                    cfg->edges[plan->latch_source_edges[j]].from;
                bool pass =
                    norm_dominates(dom, ws, info->defining_block, pred);
                plan->additional_latch_passes[i * plan->n_latch_sources + j] =
                    pass;
                if (!pass) forward = false;
            }
            plan->additional_forward[i] = forward;
        }
    }
    return true;
}

static bool m7_refresh_mux_plans(NormalizedCFG *cfg,
                                 const NormGraphView *view,
                                 M7CyclePlan *plan,
                                 const M7Config *config,
                                 M7ViewWorkspace *ws,
                                 Arena *phase) {
    if (ws && !m7_workspace_ensure_graph_capacity(ws, cfg->n_blocks,
                                                cfg->n_edges)) {
        return false;
    }
    plan->needs_entry_mux = plan->edges.n_entry_edges > 1;
    if (plan->needs_entry_mux) {
        size_t n_entry_mux =
            plan->edges.n_entry_edges + plan->edges.n_back_edges;
        size_t *mux_edges = arena_new_array(phase, size_t, n_entry_mux);
        size_t k = 0;
        for (size_t i = 0; i < plan->edges.n_entry_edges; ++i) {
            mux_edges[k++] = plan->edges.entry_edges[i];
        }
        for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
            mux_edges[k++] = plan->edges.back_edges[i];
        }
        if (!norm_edge_mux_prepare(cfg, mux_edges, n_entry_mux, NULL, 0,
                                   config->flag_type, ws ? ws->mux_dest_map : NULL,
                                   phase, &plan->entry_mux)) {
            return false;
        }
        plan->predicted_header = SIZE_MAX;
        plan->base_header_nargs = m7_mux_interface_arg_count(&plan->entry_mux);
    } else {
        if (!m7_predict_header(cfg, view, plan)) return false;
    }
    return m7_build_latch_mux_plan(cfg, plan, config, phase);
}

static bool m7_plan_exits_to_plan(const NormalizedCFG *cfg,
                                  const M7CyclePlan *upstream,
                                  size_t downstream_plan,
                                  const size_t *block_to_plan) {
    for (size_t i = 0; i < upstream->edges.n_exit_edges; ++i) {
        size_t to = cfg->edges[upstream->edges.exit_edges[i]].to;
        if (to < cfg->n_blocks && block_to_plan[to] == downstream_plan) {
            return true;
        }
    }
    return false;
}

static bool m7_sort_plans_downstream_first(const NormalizedCFG *cfg,
                                           M7CyclePlan *plans,
                                           size_t n_plans,
                                           size_t *order) {
    if (!plans || !order || n_plans == 0) return false;
    if (n_plans == 1) {
        order[0] = 0;
        return true;
    }

    size_t *block_to_plan =
        arena_new_array(cfg->phase_arena, size_t, cfg->n_blocks);
    if (!block_to_plan) return false;
    for (size_t i = 0; i < cfg->n_blocks; ++i) block_to_plan[i] = SIZE_MAX;
    for (size_t i = 0; i < n_plans; ++i) {
        for (size_t m = 0; m < plans[i].edges.n_members; ++m) {
            size_t bid = plans[i].edges.members[m];
            if (bid < cfg->n_blocks) block_to_plan[bid] = i;
        }
    }

    size_t *in_degree = arena_new_array(cfg->phase_arena, size_t, n_plans);
    if (!in_degree) return false;
    memset(in_degree, 0, n_plans * sizeof(size_t));

    for (size_t j = 0; j < n_plans; ++j) {
        for (size_t pi = 0; pi < n_plans; ++pi) {
            if (j == pi) continue;
            if (m7_plan_exits_to_plan(cfg, &plans[j], pi, block_to_plan)) {
                in_degree[j]++;
            }
        }
    }

    size_t *queue = arena_new_array(cfg->phase_arena, size_t, n_plans);
    size_t qn = 0;
    for (size_t i = 0; i < n_plans; ++i) {
        if (in_degree[i] == 0) queue[qn++] = i;
    }

    size_t out_n = 0;
    while (qn > 0) {
        qn--;
        size_t pi = queue[qn];
        order[out_n++] = pi;
        for (size_t j = 0; j < n_plans; ++j) {
            if (j == pi) continue;
            if (m7_plan_exits_to_plan(cfg, &plans[j], pi, block_to_plan)) {
                if (--in_degree[j] == 0) queue[qn++] = j;
            }
        }
    }

    if (out_n != n_plans) {
        for (size_t i = 0; i < n_plans; ++i) order[i] = i;
    }
    return true;
}

static bool m7_prepare_latch_mux(const NormalizedCFG *cfg,
                                 size_t header,
                                 const M7CycleEdges *edges,
                                 const MLIR_TypeHandle *extra_types,
                                 size_t n_extra_types,
                                 MLIR_TypeHandle discriminator_type,
                                 Arena *phase,
                                 NormEdgeMuxPlan *out) {
    size_t n_latch_mux = edges->n_back_edges + edges->n_exit_edges;
    if (n_latch_mux == 0) return false;

    size_t *edge_ids = arena_new_array(phase, size_t, n_latch_mux);
    size_t *dest_blocks = arena_new_array(phase, size_t, n_latch_mux);
    size_t k = 0;
    for (size_t i = 0; i < edges->n_back_edges; ++i) {
        edge_ids[k] = edges->back_edges[i];
        dest_blocks[k] = header;
        k++;
    }
    for (size_t i = 0; i < edges->n_exit_edges; ++i) {
        edge_ids[k] = edges->exit_edges[i];
        dest_blocks[k] = cfg->edges[edges->exit_edges[i]].to;
        k++;
    }

    if (!norm_edge_mux_prepare(cfg, edge_ids, n_latch_mux, extra_types,
                               n_extra_types, discriminator_type, NULL, phase,
                               out)) {
        return false;
    }
    memcpy(out->dest_blocks, dest_blocks, n_latch_mux * sizeof(size_t));
    return true;
}

static bool m7_prepare_cycle(NormalizedCFG *cfg,
                             const NormGraphView *view,
                             const SCCInfo *scc,
                             size_t component,
                             const DominanceInfo *dom,
                             const CFGValueIndex *values,
                             const M7Config *config,
                             const NormViewMembership *membership,
                             M7ViewWorkspace *ws,
                             Arena *phase,
                             M7CyclePlan *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !view || !scc || !dom || !values || !config || !phase ||
        !membership) {
        return false;
    }

    if (!m7_collect_cycle_edges(cfg, view, scc, component, membership, ws,
                                phase, &out->edges)) {
        return true;
    }

    if (out->edges.n_entry_edges > 1) {
        size_t n_entry_mux =
            out->edges.n_entry_edges + out->edges.n_back_edges;
        size_t *mux_edges = arena_new_array(phase, size_t, n_entry_mux);
        size_t k = 0;
        for (size_t i = 0; i < out->edges.n_entry_edges; ++i) {
            mux_edges[k++] = out->edges.entry_edges[i];
        }
        for (size_t i = 0; i < out->edges.n_back_edges; ++i) {
            mux_edges[k++] = out->edges.back_edges[i];
        }
        if (!norm_edge_mux_prepare(cfg, mux_edges, n_entry_mux, NULL, 0,
                                   config->flag_type, NULL, phase,
                                   &out->entry_mux)) {
            return false;
        }
    }

    if (!m7_predict_header(cfg, view, out)) return false;

    if (!m7_snapshot_iteration_basis(cfg, out, phase)) return false;
    if (!m7_build_base_exit_passes(cfg, values, dom, ws, out, phase)) {
        return false;
    }

    if (!m7_find_additional_live_outs(cfg, scc, component, values, dom, ws,
                                      out, phase)) {
        return false;
    }

    return true;
}

static bool m7_prepare_all_cyclic_components(NormalizedCFG *cfg,
                                            const NormGraphView *view,
                                            const SCCInfo *scc,
                                            const DominanceInfo *dom,
                                            const CFGValueIndex *values,
                                            const M7Config *config,
                                            const NormViewMembership *membership,
                                            M7ViewWorkspace *ws,
                                            Arena *phase,
                                            M7CyclePlan **out_plans,
                                            size_t *out_n) {
    if (out_plans) *out_plans = NULL;
    if (out_n) *out_n = 0;
    if (!cfg || !view || !scc || !dom || !values || !config || !phase ||
        !membership) {
        return false;
    }

    size_t cap = scc->n_components;
    M7CyclePlan *plans = cap ? arena_new_array(phase, M7CyclePlan, cap) : NULL;
    size_t n = 0;
    for (size_t c = scc->n_components; c > 0; --c) {
        size_t ci = c - 1;
        if (!scc->is_cycle || !scc->is_cycle[ci]) continue;
        if (!m7_prepare_cycle(cfg, view, scc, ci, dom, values, config,
                              membership, ws, phase, &plans[n])) {
            return false;
        }
        if (plans[n].edges.n_back_edges == 0) continue;
        n++;
    }
    if (out_plans) *out_plans = plans;
    if (out_n) *out_n = n;
    return true;
}

static bool m7_commit_entry_normalization(NormalizedCFG *cfg,
                                          M7CyclePlan *plan,
                                          size_t *out_header) {
    if (out_header) *out_header = SIZE_MAX;
    if (!plan->needs_entry_mux) {
        *out_header = plan->predicted_header;
        return plan->predicted_header < cfg->n_blocks;
    }

    NormEdgeMux mux;
    if (!norm_edge_mux_create(cfg, &plan->entry_mux, NORM_BLOCK_ENTRY_MUX, &mux)) {
        return false;
    }

    for (size_t i = 0; i < plan->edges.n_entry_edges; ++i) {
        size_t eid = plan->edges.entry_edges[i];
        size_t orig = plan->entry_mux.dest_blocks[i];
        if (!norm_edge_mux_redirect(cfg, &mux, eid, orig, NULL, 0)) return false;
    }
    for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
        size_t eid = plan->edges.back_edges[i];
        size_t orig = plan->entry_mux.dest_blocks[plan->edges.n_entry_edges + i];
        if (!norm_edge_mux_redirect(cfg, &mux, eid, orig, NULL, 0)) return false;
    }

    if (!norm_edge_mux_create_dispatch(cfg, &mux, mux.mux_block, SIZE_MAX)) {
        return false;
    }
    if (!normalized_cfg_finalize_outgoing(cfg, mux.mux_block)) return false;

    *out_header = mux.mux_block;
    return true;
}

static bool m7_install_m5_infinite_exit(NormalizedCFG *cfg, size_t dispatch,
                                        const M5ExitCombiner *combiner) {
    Arena *arena = cfg->graph_arena;
    if (!combiner || combiner->n_classes == 0) {
        NormTerminator term;
        norm_terminator_init(&term);
        term.kind = CFG_TERM_UNREACHABLE;
        return normalized_cfg_set_block_owned_terminator(cfg, arena, dispatch,
                                                         &term);
    }

    const M5ExitClass *cls = NULL;
    for (size_t i = 0; i < combiner->n_classes; ++i) {
        if (combiner->classes[i].flavor == M5_EXIT_LLVM_RETURN) {
            cls = &combiner->classes[i];
            break;
        }
    }
    if (!cls) {
        NormTerminator term;
        norm_terminator_init(&term);
        term.kind = CFG_TERM_UNREACHABLE;
        return normalized_cfg_set_block_owned_terminator(cfg, arena, dispatch,
                                                         &term);
    }

    size_t n = cls->n_operand_types;
    NormOperand *ops = NULL;
    if (n > 0) {
        ops = arena_new_array(arena, NormOperand, n);
        for (size_t i = 0; i < n; ++i) {
            ops[i] = norm_operand_undef(cls->operand_types[i]);
        }
    }

    size_t edge_id = SIZE_MAX;
    if (!normalized_cfg_add_synthetic_edge_adopt(
            cfg, dispatch, cls->shared_exit_block, 0, ops, n, &edge_id)) {
        return false;
    }

    NormTerminator term;
    norm_terminator_init(&term);
    term.kind = CFG_TERM_BR;
    if (!normalized_cfg_set_block_owned_terminator(cfg, arena, dispatch, &term)) {
        return false;
    }
    return true;
}

static void m7_patch_latch_mux_header(M7CyclePlan *plan, size_t header) {
    if (!plan->needs_entry_mux) return;
    if (plan->latch_mux.n_entries > 0) {
        plan->latch_mux.entries[0].destination_block = header;
    }
    for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
        plan->latch_mux.dest_blocks[i] = header;
    }
}

static bool m7_apply_reduce_interface(NormalizedCFG *cfg,
                                      const M7CyclePlan *plan,
                                      size_t latch_block,
                                      size_t header,
                                      size_t dispatch,
                                      M7Loop *loop) {
    Arena *arena = cfg->graph_arena;
    NormBlock *hdr = &cfg->blocks[header];
    NormBlock *lch = &cfg->blocks[latch_block];
    NormBlock *disp = &cfg->blocks[dispatch];
    size_t dispatch_arg_offset = disp->n_args;

    for (size_t i = 0; i < plan->n_base_iterations; ++i) {
        MLIR_TypeHandle ty = plan->base_iteration_types[i];
        if (!norm_block_append_formal_args(disp, arena, &ty, 1)) {
            return false;
        }
    }

    loop->n_additional_live_outs = plan->n_additional_values;
    if (loop->n_additional_live_outs > 0) {
        loop->additional_live_outs =
            arena_new_array(arena, M7AdditionalLiveOut,
                            loop->n_additional_live_outs);
    }

    for (size_t i = 0; i < plan->n_additional_values; ++i) {
        CFGValueInfo *info = plan->additional_values[i];
        bool forward = plan->additional_forward[i];
        MLIR_TypeHandle ty = info->type;
        M7AdditionalLiveOut *alo = &loop->additional_live_outs[i];

        alo->source_value = info->value;
        alo->type = ty;
        norm_operand_init(&alo->latch_argument);

        if (!norm_block_append_formal_args(hdr, arena, &ty, 1) ||
            !norm_block_append_formal_args(disp, arena, &ty, 1)) {
            return false;
        }
        size_t hdr_idx = hdr->n_args - 1;
        size_t disp_idx = disp->n_args - 1;
        alo->header_argument = norm_operand_block_arg(ty, header, hdr_idx);
        alo->exit_argument = norm_operand_block_arg(ty, dispatch, disp_idx);

        if (!forward) {
            if (!norm_block_append_formal_args(lch, arena, &ty, 1)) {
                return false;
            }
            size_t latch_idx = lch->n_args - 1;
            alo->latch_argument = norm_operand_block_arg(ty, latch_block,
                                                         latch_idx);

            NormEdgeIter in_it = norm_in_edges(cfg, latch_block);
            size_t eid;
            while (norm_edge_iter_next(&in_it, &eid)) {
                NormEdge *e = &cfg->edges[eid];
                size_t n_ops =
                    norm_edge_payload_num_operands(cfg, &e->payload);
                NormOperand *ops =
                    arena_new_array(arena, NormOperand, n_ops + 1);
                for (size_t oi = 0; oi < n_ops; ++oi) {
                    norm_edge_payload_operand(cfg, &e->payload, oi, &ops[oi]);
                }
                bool pass_mlir = false;
                for (size_t sj = 0; sj < plan->n_latch_sources; ++sj) {
                    if (plan->latch_source_edges[sj] == eid) {
                        pass_mlir = plan->additional_latch_passes[
                            i * plan->n_latch_sources + sj];
                        break;
                    }
                }
                ops[n_ops] = pass_mlir
                    ? norm_operand_mlir(ty, info->value)
                    : norm_operand_undef(ty);
                if (!normalized_cfg_replace_edge_payload(cfg, arena, eid,
                                                         ops, n_ops + 1)) {
                    return false;
                }
            }
        }
    }

  /* Extend non-latch incoming edges to enlarged header. */
    NormEdgeIter hdr_in = norm_in_edges(cfg, header);
    size_t heid;
    while (norm_edge_iter_next(&hdr_in, &heid)) {
        if (cfg->edges[heid].from == latch_block) continue;
        size_t n_ops =
            norm_edge_payload_num_operands(cfg, &cfg->edges[heid].payload);
        if (n_ops == hdr->n_args) continue;
        NormOperand *ops = arena_new_array(arena, NormOperand, hdr->n_args);
        for (size_t oi = 0; oi < n_ops; ++oi) {
            norm_edge_payload_operand(cfg, &cfg->edges[heid].payload, oi,
                                      &ops[oi]);
        }
        for (size_t oi = n_ops; oi < hdr->n_args; ++oi) {
            ops[oi] = norm_operand_undef(hdr->args[oi].type);
        }
        if (!normalized_cfg_replace_edge_payload(cfg, arena, heid, ops,
                                                 hdr->n_args)) {
            return false;
        }
    }

    loop->n_iteration_values = plan->n_base_iterations;
    if (loop->n_iteration_values > 0) {
        loop->iteration_values =
            arena_new_array(arena, M7IterationValue, loop->n_iteration_values);
        for (size_t i = 0; i < loop->n_iteration_values; ++i) {
            M7IterationValue *iv = &loop->iteration_values[i];
            iv->type = plan->base_iteration_types[i];
            iv->header_argument =
                norm_operand_block_arg(iv->type, header, i);
            iv->replacement_source = plan->base_iteration_values[i];
            iv->exit_argument =
                norm_operand_block_arg(iv->type, dispatch,
                                       dispatch_arg_offset + i);
            norm_operand_init(&iv->next_value);
        }
    }

    return true;
}

static bool m7_build_back_edge_payload(const NormalizedCFG *cfg,
                                       const NormEdgeMux *latch_mux,
                                       size_t header,
                                       const M7CyclePlan *plan,
                                       Arena *arena,
                                       NormOperand **out_ops,
                                       size_t *out_n) {
    size_t hdr_entry =
        norm_mux_find_entry(latch_mux->entries, latch_mux->n_entries, header);
    if (hdr_entry == SIZE_MAX && latch_mux->n_entries > 0) hdr_entry = 0;
    if (hdr_entry == SIZE_MAX) return false;

    const NormMuxEntry *he = &latch_mux->entries[hdr_entry];
    const NormBlock *hdr = &cfg->blocks[header];
    size_t n_ops = hdr->n_args;
    NormOperand *ops = n_ops
        ? arena_new_array(arena, NormOperand, n_ops) : NULL;

    for (size_t j = 0; j < he->n_args; ++j) {
        ops[j] = norm_operand_block_arg(
            cfg->blocks[latch_mux->mux_block].args[he->arg_offset + j].type,
            latch_mux->mux_block, he->arg_offset + j);
    }
    for (size_t j = he->n_args; j < plan->base_header_nargs; ++j) {
        ops[j] = norm_operand_undef(hdr->args[j].type);
    }
    for (size_t i = 0; i < plan->n_additional_values; ++i) {
        size_t idx = plan->base_header_nargs + i;
        if (plan->additional_forward[i]) {
            ops[idx] = norm_operand_mlir(plan->additional_values[i]->type,
                                         plan->additional_values[i]->value);
        } else {
            size_t latch_idx = plan->latch_mux.n_mux_args;
            for (size_t j = 0; j < i; ++j) {
                if (!plan->additional_forward[j]) latch_idx++;
            }
            ops[idx] = norm_operand_block_arg(hdr->args[idx].type,
                                              latch_mux->mux_block, latch_idx);
        }
    }
    *out_ops = ops;
    *out_n = n_ops;
    return true;
}

static bool m7_build_exit_edge_payload(const NormalizedCFG *cfg,
                                       const NormEdgeMux *latch_mux,
                                       size_t latch_block,
                                       size_t header,
                                       size_t dispatch,
                                       const M7CyclePlan *plan,
                                       Arena *arena,
                                       NormOperand **out_ops,
                                       size_t *out_n) {
    const NormBlock *disp = &cfg->blocks[dispatch];
    size_t n_ops = disp->n_args;
    if (n_ops == 0) {
        *out_ops = NULL;
        *out_n = 0;
        return true;
    }
    NormOperand *ops = arena_new_array(arena, NormOperand, n_ops);
    size_t dispatch_arg_offset = latch_mux->extra_arg_offset;
    if (dispatch_arg_offset > n_ops) return false;
    for (size_t i = 0; i < dispatch_arg_offset; ++i) {
        ops[i] = norm_operand_block_arg(
            cfg->blocks[latch_block].args[i].type, latch_block, i);
    }

    size_t hdr_entry =
        norm_mux_find_entry(latch_mux->entries, latch_mux->n_entries, header);
    if (hdr_entry == SIZE_MAX && latch_mux->n_entries > 0) hdr_entry = 0;
    const NormMuxEntry *he = hdr_entry != SIZE_MAX
        ? &latch_mux->entries[hdr_entry] : NULL;

    for (size_t i = 0; i < plan->n_base_iterations; ++i) {
        size_t idx = dispatch_arg_offset + i;
        if (he && i < he->n_args) {
            ops[idx] = norm_operand_block_arg(
                cfg->blocks[latch_mux->mux_block].args[he->arg_offset + i].type,
                latch_mux->mux_block, he->arg_offset + i);
        } else {
            ops[idx] = norm_operand_undef(plan->base_iteration_types[i]);
        }
    }
    for (size_t i = 0; i < plan->n_additional_values; ++i) {
        size_t idx = dispatch_arg_offset + plan->n_base_iterations + i;
        if (plan->additional_forward[i]) {
            ops[idx] = norm_operand_mlir(plan->additional_values[i]->type,
                                         plan->additional_values[i]->value);
        } else {
            size_t latch_idx = plan->latch_mux.n_mux_args;
            for (size_t j = 0; j < i; ++j) {
                if (!plan->additional_forward[j]) latch_idx++;
            }
            ops[idx] = norm_operand_block_arg(disp->args[idx].type, latch_block,
                                              latch_idx);
        }
    }
    *out_ops = ops;
    *out_n = n_ops;
    return true;
}

static bool m7_create_single_exiting_latch(NormalizedCFG *cfg,
                                         const M7Config *config,
                                         size_t header,
                                         M7CyclePlan *plan,
                                         M7Loop *loop,
                                         const M5ExitCombiner *combiner,
                                         M7LatchResult *out) {
    memset(out, 0, sizeof(*out));
    m7_patch_latch_mux_header(plan, header);

    NormEdgeMux latch_mux;
    if (!norm_edge_mux_create(cfg, &plan->latch_mux, NORM_BLOCK_LOOP_LATCH,
                              &latch_mux)) {
        return false;
    }

    NormOperand repeat = norm_operand_discriminator(config->flag_type, 1);
    NormOperand stop = norm_operand_discriminator(config->flag_type, 0);

    for (size_t i = 0; i < plan->edges.n_back_edges; ++i) {
        size_t orig = plan->latch_mux.dest_blocks[i];
        if (!norm_edge_mux_redirect(cfg, &latch_mux, plan->edges.back_edges[i],
                                    orig, &repeat, 1)) {
            return false;
        }
    }
    size_t header_entry = norm_mux_find_entry(
        latch_mux.entries, latch_mux.n_entries, header);
    if (header_entry == SIZE_MAX) return false;
    const NormMuxEntry *header_mux = &latch_mux.entries[header_entry];
    for (size_t i = 0; i < plan->edges.n_exit_edges; ++i) {
        size_t edge_id = plan->edges.exit_edges[i];
        size_t orig =
            plan->latch_mux.dest_blocks[plan->edges.n_back_edges + i];
        if (!norm_edge_mux_redirect(cfg, &latch_mux, edge_id,
                                    orig, &stop, 1)) {
            return false;
        }

        size_t n_ops = norm_edge_payload_num_operands(
            cfg, &cfg->edges[edge_id].payload);
        NormOperand *ops = n_ops
            ? arena_new_array(cfg->phase_arena, NormOperand, n_ops) : NULL;
        if (n_ops && !ops) return false;
        for (size_t j = 0; j < n_ops; ++j) {
            if (!norm_edge_payload_operand(cfg, &cfg->edges[edge_id].payload,
                                           j, &ops[j])) {
                return false;
            }
        }
        size_t n_route = plan->n_base_iterations;
        if (n_route > header_mux->n_args) n_route = header_mux->n_args;
        for (size_t j = 0; j < n_route; ++j) {
            size_t idx = header_mux->arg_offset + j;
            if (idx >= n_ops || j >= cfg->blocks[header].n_args) return false;
            bool pass_next = plan->base_exit_passes &&
                plan->base_exit_passes[
                    j * plan->edges.n_exit_edges + i] &&
                plan->base_iteration_values[j] != MLIR_INVALID_HANDLE;
            ops[idx] = pass_next
                ? norm_operand_mlir(plan->base_iteration_types[j],
                                    plan->base_iteration_values[j])
                : norm_operand_block_arg(
                      cfg->blocks[header].args[j].type, header, j);
        }
        if (!normalized_cfg_replace_edge_payload(
                cfg, cfg->graph_arena, edge_id, ops, n_ops)) {
            return false;
        }
    }

    size_t latch_block = latch_mux.mux_block;
    size_t repeat_arg = cfg->blocks[latch_block].n_args - 1;
    NormOperand condition =
        norm_operand_block_arg(config->flag_type, latch_block, repeat_arg);

    size_t dispatch = normalized_cfg_add_synthetic_block(
        cfg, cfg->graph_arena, NORM_BLOCK_LOOP_EXIT_DISPATCH,
        CFG_TERM_NONE, NULL, 0);
    if (dispatch == SIZE_MAX) return false;

    for (size_t i = 0; i < latch_mux.extra_arg_offset; ++i) {
        MLIR_TypeHandle ty = cfg->blocks[latch_block].args[i].type;
        if (!norm_block_append_formal_args(&cfg->blocks[dispatch],
                                           cfg->graph_arena, &ty, 1)) {
            return false;
        }
    }

    if (!m7_apply_reduce_interface(cfg, plan, latch_block, header, dispatch,
                                   loop)) {
        return false;
    }

    NormOperand *back_ops = NULL;
    size_t n_back_ops = 0;
    if (!m7_build_back_edge_payload(cfg, &latch_mux, header, plan,
                                    cfg->graph_arena, &back_ops,
                                    &n_back_ops)) {
        return false;
    }
    size_t back_eid = SIZE_MAX;
    if (!normalized_cfg_add_synthetic_edge_adopt(
            cfg, latch_block, header, 0, back_ops, n_back_ops, &back_eid)) {
        return false;
    }

    NormOperand *exit_ops = NULL;
    size_t n_exit_ops = 0;
    if (!m7_build_exit_edge_payload(cfg, &latch_mux, latch_block, header,
                                    dispatch, plan,
                                    cfg->graph_arena, &exit_ops,
                                    &n_exit_ops)) {
        return false;
    }
    size_t exit_eid = SIZE_MAX;
    if (!normalized_cfg_add_synthetic_edge_adopt(
            cfg, latch_block, dispatch, 1, exit_ops, n_exit_ops, &exit_eid)) {
        return false;
    }

    NormTerminator latch_term;
    norm_terminator_init(&latch_term);
    latch_term.kind = CFG_TERM_COND_BR;
    latch_term.selector = condition;
    if (!normalized_cfg_set_block_owned_terminator(cfg, cfg->graph_arena,
                                                   latch_block, &latch_term)) {
        return false;
    }

    if (plan->edges.n_exit_edges > 0) {
        if (!norm_edge_mux_create_dispatch(cfg, &latch_mux, dispatch, header)) {
            return false;
        }
    } else {
        if (!m7_install_m5_infinite_exit(cfg, dispatch, combiner)) return false;
    }

    if (!normalized_cfg_finalize_outgoing(cfg, latch_block)) return false;
    if (!normalized_cfg_finalize_outgoing(cfg, dispatch)) return false;

    out->latch = latch_block;
    out->exit_dispatch = dispatch;
    out->back_edge = back_eid;
    out->exit_edge = exit_eid;
    out->condition = condition;
    return true;
}

static bool m7_populate_iteration_values(NormalizedCFG *cfg,
                                         const M7LatchResult *latch,
                                         M7Loop *loop) {
    if (loop->n_iteration_values == 0) return true;

    const NormEdge *back = &cfg->edges[latch->back_edge];
    for (size_t i = 0; i < loop->n_iteration_values; ++i) {
        M7IterationValue *iv = &loop->iteration_values[i];
        if (i < norm_edge_payload_num_operands(cfg, &back->payload)) {
            norm_edge_payload_operand(cfg, &back->payload, i,
                                      &iv->next_value);
        } else {
            iv->next_value = norm_operand_undef(iv->type);
        }
    }
    return true;
}

static bool m7_build_body_block_list(Arena *arena,
                                     const M7CyclePlan *plan,
                                     size_t header,
                                     size_t latch,
                                     size_t **out_blocks,
                                     size_t *out_n) {
    size_t n = plan->edges.n_members;
    size_t cap = n + 2;
    size_t *blocks = arena_new_array(arena, size_t, cap);
    if (!blocks) return false;

    memcpy(blocks, plan->edges.members, n * sizeof(size_t));

    bool have_hdr = false, have_latch = false;
    for (size_t j = 0; j < n; ++j) {
        if (blocks[j] == header) have_hdr = true;
        if (blocks[j] == latch) have_latch = true;
    }
    if (!have_hdr) blocks[n++] = header;
    if (!have_latch) blocks[n++] = latch;

    *out_blocks = blocks;
    *out_n = n;
    return true;
}

static bool m7_result_grow_replacements(M7Result *res, Arena *arena) {
    size_t nc = res->live_out_replacements_cap
        ? res->live_out_replacements_cap * 2 : 8;
    M7LiveOutReplacement *next =
        arena_new_array(arena, M7LiveOutReplacement, nc);
    if (res->n_live_out_replacements > 0) {
        memcpy(next, res->live_out_replacements,
               res->n_live_out_replacements * sizeof(M7LiveOutReplacement));
    }
    res->live_out_replacements = next;
    res->live_out_replacements_cap = nc;
    return true;
}

static bool m7_register_replacement_key(M7Result *res, Arena *arena,
                                        const NormValueKey *source, size_t loop_id,
                                        const NormOperand *replacement) {
    if (!norm_value_key_is_valid(source) || !norm_operand_is_valid(replacement)) {
        return true;
    }
    if (res->n_live_out_replacements >= res->live_out_replacements_cap &&
        !m7_result_grow_replacements(res, arena)) {
        return false;
    }
    M7LiveOutReplacement *r =
        &res->live_out_replacements[res->n_live_out_replacements++];
    r->source_key = *source;
    r->loop_id = loop_id;
    r->replacement = *replacement;
    return true;
}

static bool m7_register_loop_replacements(M7Result *res, Arena *arena,
                                          const NormalizedCFG *cfg,
                                          const M7Loop *loop,
                                          const M7CyclePlan *plan) {
    for (size_t i = 0; i < plan->n_base_iterations; ++i) {
        if (i >= loop->n_iteration_values) break;
        const M7IterationValue *iv = &loop->iteration_values[i];
        if (!norm_operand_is_valid(&iv->exit_argument)) continue;

        if (iv->replacement_source != MLIR_INVALID_HANDLE) {
            NormValueKey key = norm_value_key_mlir(iv->replacement_source);
            if (!m7_register_replacement_key(res, arena, &key, loop->id,
                                             &iv->exit_argument)) {
                return false;
            }
        }
        NormValueKey header_key;
        if (norm_value_key_from_operand(&iv->header_argument, &header_key) &&
            !m7_register_replacement_key(res, arena, &header_key, loop->id,
                                         &iv->exit_argument)) {
            return false;
        }
        if (cfg && loop->header < cfg->n_blocks) {
            const NormBlock *header = &cfg->blocks[loop->header];
            if (header->kind == NORM_BLOCK_ORIGINAL &&
                header->mlir_block != MLIR_INVALID_HANDLE &&
                i < MLIR_GetBlockNumArgs(header->mlir_block)) {
                NormValueKey key = norm_value_key_mlir(
                    MLIR_GetBlockArg(header->mlir_block, i));
                if (!m7_register_replacement_key(res, arena, &key, loop->id,
                                                 &iv->exit_argument)) {
                    return false;
                }
            }
        }
        NormValueKey next_key;
        if (norm_value_key_from_operand(&iv->next_value, &next_key) &&
            !m7_register_replacement_key(res, arena, &next_key, loop->id,
                                         &iv->exit_argument)) {
            return false;
        }
    }
    for (size_t i = 0; i < loop->n_additional_live_outs; ++i) {
        const M7AdditionalLiveOut *alo = &loop->additional_live_outs[i];
        if (!norm_operand_is_valid(&alo->exit_argument)) continue;

        if (alo->source_value != MLIR_INVALID_HANDLE) {
            NormValueKey key = norm_value_key_mlir(alo->source_value);
            if (!m7_register_replacement_key(res, arena, &key, loop->id,
                                             &alo->exit_argument)) {
                return false;
            }
        }
        NormValueKey hdr_key;
        if (norm_value_key_from_operand(&alo->header_argument, &hdr_key) &&
            !m7_register_replacement_key(res, arena, &hdr_key, loop->id,
                                         &alo->exit_argument)) {
            return false;
        }
    }
    return true;
}

// Parent-loop depth: root loops are 0, nested children increase depth.
static size_t m7_loop_depth(const M7Result *res, size_t loop_id) {
    size_t depth = 0;
    while (loop_id != SIZE_MAX && loop_id < res->n_loops) {
        size_t parent = res->loops[loop_id].parent_loop;
        if (parent == SIZE_MAX) break;
        depth++;
        loop_id = parent;
    }
    return depth;
}

static bool m8_block_in_loop_body(const M8EmissionMetadata *meta, size_t loop_id,
                                  size_t consumer_block) {
    if (!meta || !meta->scratch || consumer_block >= meta->n_blocks) {
        return false;
    }
    const M7Result *res = &meta->scratch->m7;
    size_t enc = meta->block_body_loop[consumer_block];
    while (enc != SIZE_MAX && enc < res->n_loops) {
        if (enc == loop_id) return true;
        enc = res->loops[enc].parent_loop;
    }
    return false;
}

typedef struct {
    size_t slot;
    MLIR_ValueHandle old_value;
    bool old_bound;
} M8BindingChange;

typedef enum {
    M8_LABEL_IF,
    M8_LABEL_BLOCK,
    M8_LABEL_LOOP,
} M8LabelKind;

typedef struct {
    M8LabelKind kind;
    size_t target_block;
    size_t loop_id;
} M8Label;

struct M8EmitCtx {
    FnCtx *F;
    const M8EmissionMetadata *meta;
    const NormalizedCFG *cfg;
    size_t *arg_offsets;
    MLIR_ValueHandle *arg_values;
    bool *arg_bound;
    size_t n_arg_slots;
    ValueIndexMap arg_slot_by_mlir;
    M8BindingChange *changes;
    size_t n_changes;
    size_t changes_cap;
    M8Label *labels;
    size_t n_labels;
    size_t labels_cap;
};

static bool m8_emit_lookup_arg(const M8EmitCtx *ec, size_t block_id,
                               size_t arg_index, MLIR_ValueHandle *out);

static bool m8_emit_lookup_current_mlir_arg(const M8EmitCtx *ec,
                                            MLIR_ValueHandle value,
                                            MLIR_ValueHandle *out) {
    if (!ec) return false;
    size_t slot = value_index_map_probe(&ec->arg_slot_by_mlir, value);
    if (slot == SIZE_MAX || slot >= ec->n_arg_slots ||
        !ec->arg_bound[slot]) {
        return false;
    }
    *out = ec->arg_values[slot];
    return *out != MLIR_INVALID_HANDLE;
}

static bool m8_emit_grow_changes(M8EmitCtx *ec) {
    size_t new_cap = ec->changes_cap ? ec->changes_cap * 2 : 32;
    M8BindingChange *next = (M8BindingChange *)arena_alloc(
        ec->F->arena, new_cap * sizeof(M8BindingChange));
    if (!next) return false;
    if (ec->n_changes) {
        memcpy(next, ec->changes,
               ec->n_changes * sizeof(M8BindingChange));
    }
    ec->changes = next;
    ec->changes_cap = new_cap;
    return true;
}

static size_t m8_emit_arg_slot(const M8EmitCtx *ec, size_t block_id,
                               size_t arg_index) {
    if (!ec || block_id >= ec->meta->n_blocks) return SIZE_MAX;
    size_t begin = ec->arg_offsets[block_id];
    size_t end = ec->arg_offsets[block_id + 1];
    if (arg_index >= end - begin) return SIZE_MAX;
    return begin + arg_index;
}

static bool m8_emit_lookup_arg(const M8EmitCtx *ec, size_t block_id,
                               size_t arg_index, MLIR_ValueHandle *out) {
    size_t slot = m8_emit_arg_slot(ec, block_id, arg_index);
    if (slot == SIZE_MAX || !ec->arg_bound[slot]) return false;
    *out = ec->arg_values[slot];
    return *out != MLIR_INVALID_HANDLE;
}

static bool m8_emit_bind_arg(M8EmitCtx *ec, size_t block_id, size_t arg_index,
                             MLIR_ValueHandle value) {
    size_t slot = m8_emit_arg_slot(ec, block_id, arg_index);
    if (slot == SIZE_MAX || value == MLIR_INVALID_HANDLE) {
        return false;
    }
    if (ec->n_changes >= ec->changes_cap && !m8_emit_grow_changes(ec)) {
        return false;
    }
    M8BindingChange *change = &ec->changes[ec->n_changes++];
    change->slot = slot;
    change->old_value = ec->arg_values[slot];
    change->old_bound = ec->arg_bound[slot];
    ec->arg_values[slot] = value;
    ec->arg_bound[slot] = true;
    return true;
}

static void m8_emit_restore_bindings(M8EmitCtx *ec, size_t checkpoint) {
    while (ec->n_changes > checkpoint) {
        ec->n_changes--;
        M8BindingChange *change = &ec->changes[ec->n_changes];
        ec->arg_values[change->slot] = change->old_value;
        ec->arg_bound[change->slot] = change->old_bound;
    }
}

static bool m8_resolve_live_out(const M8EmissionMetadata *meta,
                                size_t consumer_block,
                                const NormValueKey *source,
                                NormOperand *out) {
    if (!meta || !meta->built || !norm_value_key_is_valid(source) || !out) {
        return false;
    }
    norm_operand_init(out);

    size_t begin = 0;
    size_t end = meta->n_replacement_slots;
    bool use_chain = false;
    if (source->kind == NORM_OPERAND_MLIR) {
        size_t hint =
            value_index_map_probe(&meta->replacement_by_mlir, source->mlir_value);
        if (hint == SIZE_MAX) return false;
        begin = hint;
        use_chain = true;
    }

    size_t best_depth = SIZE_MAX;
    bool found = false;
    for (size_t i = begin; i < end;
         i = use_chain ? meta->replacement_slots[i].next_same_mlir : i + 1) {
        const M8ReplacementSlot *s = &meta->replacement_slots[i];
        if (!norm_value_key_equal(&s->key, source)) continue;
        if (m8_block_in_loop_body(meta, s->loop_id, consumer_block)) continue;
        if (!found || s->loop_depth < best_depth) {
            best_depth = s->loop_depth;
            *out = s->replacement;
            found = true;
        }
    }
    return found && norm_operand_is_valid(out);
}

static void m8_replacement_map_set(ValueIndexMap *map, Arena *arena,
                                   MLIR_ValueHandle value, size_t index) {
    size_t old = value_index_map_probe(map, value);
    if (old == SIZE_MAX) {
        value_index_map_put(map, arena, value, index);
        return;
    }
    uintptr_t key = (uintptr_t)value;
    size_t mask = map->cap - 1;
    size_t slot = map_hash(key) & mask;
    while (map->keys[slot] != key) slot = (slot + 1) & mask;
    map->vals[slot] = index;
}

static bool fn_emit_norm_operand(FnCtx *F, const NormOperand *op,
                                 MLIR_ValueHandle *out) {
    if (!F || !op || !out || !norm_operand_is_valid(op)) return false;
    switch (op->kind) {
    case NORM_OPERAND_MLIR:
        if (F->m8_emit &&
            m8_emit_lookup_current_mlir_arg(F->m8_emit,
                                            op->as.mlir_value, out)) {
            return true;
        }
        return vmap_get(F, op->as.mlir_value, out);
    case NORM_OPERAND_BLOCK_ARG:
        if (F->m8_emit) {
            return m8_emit_lookup_arg(F->m8_emit,
                                      op->as.block_arg.block_id,
                                      op->as.block_arg.arg_index, out);
        }
        return false;
    case NORM_OPERAND_DISCRIMINATOR:
        *out = emit_const_i32(F, op->as.discriminator);
        return *out != MLIR_INVALID_HANDLE;
    case NORM_OPERAND_UNDEF:
        *out = emit_const_i32(F, 0);
        return *out != MLIR_INVALID_HANDLE;
    default:
        return false;
    }
}

static bool m8_build_structured_plan(CFGInfoScratch *scratch);

static bool m8_build_emission_metadata(CFGInfoScratch *scratch) {
    if (!scratch) return false;
    M8EmissionMetadata *meta = &scratch->m8;
    M7Result *m7 = &scratch->m7;
    Arena *arena = scratch->norm.graph_arena;
    if (!arena) return false;

    memset(meta, 0, sizeof(*meta));
    meta->scratch = scratch;
    meta->n_blocks = scratch->norm.n_blocks;
    if (meta->n_blocks > 0) {
        meta->block_body_loop = arena_new_array(arena, size_t, meta->n_blocks);
        if (!meta->block_body_loop) return false;
        for (size_t i = 0; i < meta->n_blocks; ++i) {
            meta->block_body_loop[i] = SIZE_MAX;
        }

        size_t *body_depth = arena_new_array(arena, size_t, meta->n_blocks);
        if (!body_depth) return false;
        memset(body_depth, 0, meta->n_blocks * sizeof(size_t));

        for (size_t li = 0; li < m7->n_loops; ++li) {
            const M7Loop *loop = &m7->loops[li];
            size_t depth = m7_loop_depth(m7, li);
            for (size_t bi = 0; bi < loop->n_body_blocks; ++bi) {
                size_t bid = loop->body_blocks[bi];
                if (bid >= meta->n_blocks) continue;
                if (body_depth[bid] <= depth) {
                    body_depth[bid] = depth;
                    meta->block_body_loop[bid] = li;
                }
            }
        }
    }

    size_t n_slots = m7->n_live_out_replacements;
    size_t slots_cap = n_slots > 0 ? n_slots : 8;
    meta->replacement_slots =
        arena_new_array(arena, M8ReplacementSlot, slots_cap);
    if (!meta->replacement_slots) return false;

    if (n_slots > 0) {
        value_index_map_init(&meta->replacement_by_mlir, arena, n_slots);
    }

    for (size_t ri = 0; ri < m7->n_live_out_replacements; ++ri) {
        const M7LiveOutReplacement *r = &m7->live_out_replacements[ri];
        M8ReplacementSlot *s =
            &meta->replacement_slots[meta->n_replacement_slots++];
        s->key = r->source_key;
        s->loop_id = r->loop_id;
        s->loop_depth = m7_loop_depth(m7, r->loop_id);
        s->replacement = r->replacement;
        s->next_same_mlir = SIZE_MAX;
        if (s->key.kind == NORM_OPERAND_MLIR) {
            s->next_same_mlir = value_index_map_probe(
                &meta->replacement_by_mlir, s->key.mlir_value);
            m8_replacement_map_set(&meta->replacement_by_mlir, arena,
                                   s->key.mlir_value, ri);
        }
    }

    if (!m8_build_structured_plan(scratch)) return false;
    meta->built = true;
    return true;
}

static int fn_vmap_get(FnCtx *F, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    if (F->m8_plan && F->m8_plan->built &&
        F->current_block != MLIR_INVALID_HANDLE) {
        size_t cfg_bid = block_index_map_probe(
            &F->m8_plan->scratch->cfg.block_to_index, F->current_block);
        if (cfg_bid != SIZE_MAX) {
            NormValueKey source = norm_value_key_mlir(k);
            NormOperand repl;
            if (m8_resolve_live_out(F->m8_plan, cfg_bid, &source, &repl) &&
                fn_emit_norm_operand(F, &repl, out)) {
                return 1;
            }
        }
    }
    if (F->m8_emit &&
        m8_emit_lookup_current_mlir_arg(F->m8_emit, k, out)) {
        return 1;
    }
    return vmap_get(F, k, out);
}

static bool m7_result_grow_loops(M7Result *res, Arena *arena) {
    size_t nc = res->loops_cap ? res->loops_cap * 2 : 4;
    M7Loop *next = arena_new_array(arena, M7Loop, nc);
    if (res->n_loops > 0) {
        memcpy(next, res->loops, res->n_loops * sizeof(M7Loop));
    }
    res->loops = next;
    res->loops_cap = nc;
    return true;
}

static bool m7_result_grow_views(M7Result *res, Arena *arena) {
    size_t nc = res->views_cap ? res->views_cap * 2 : 4;
    NormGraphView *next = arena_new_array(arena, NormGraphView, nc);
    if (res->n_views > 0) {
        memcpy(next, res->views, res->n_views * sizeof(NormGraphView));
    }
    res->views = next;
    res->views_cap = nc;
    return true;
}

static bool m7_push_view(M7Result *res, Arena *arena, const NormGraphView *view,
                         size_t *out_id) {
    if (res->n_views >= res->views_cap && !m7_result_grow_views(res, arena)) {
        return false;
    }
    size_t id = res->n_views++;
    res->views[id] = *view;
    res->views[id].id = id;
    if (view->blocks && view->n_blocks > 0) {
        size_t *blocks = arena_new_array(arena, size_t, view->n_blocks);
        memcpy(blocks, view->blocks, view->n_blocks * sizeof(size_t));
        res->views[id].blocks = blocks;
    }
    if (view->hidden_edges && view->n_hidden_edges > 0) {
        size_t *hidden = arena_new_array(arena, size_t, view->n_hidden_edges);
        memcpy(hidden, view->hidden_edges,
               view->n_hidden_edges * sizeof(size_t));
        res->views[id].hidden_edges = hidden;
    }
    if (out_id) *out_id = id;
    return true;
}

static bool m7_worklist_push(M7Result *res, Arena *arena, size_t view_id) {
    if (res->worklist_n >= res->worklist_cap) {
        size_t nc = res->worklist_cap ? res->worklist_cap * 2 : 8;
        size_t *next = arena_new_array(arena, size_t, nc);
        if (res->worklist_n > 0) {
            memcpy(next, res->worklist, res->worklist_n * sizeof(size_t));
        }
        res->worklist = next;
        res->worklist_cap = nc;
    }
    res->worklist[res->worklist_n++] = view_id;
    return true;
}

static bool m7_worklist_empty(const M7Result *res) {
    return !res || res->worklist_n == 0;
}

static size_t m7_worklist_pop(M7Result *res) {
    res->worklist_n--;
    return res->worklist[res->worklist_n];
}

static bool m7_push_root_view(const NormalizedCFG *cfg, M7Result *res,
                              Arena *arena) {
    NormGraphView root;
    if (!norm_build_root_view(arena, cfg, &root)) return false;
    size_t id;
    if (!m7_push_view(res, arena, &root, &id)) return false;
    return m7_worklist_push(res, arena, id);
}

static bool m7_make_loop_body_view(NormalizedCFG *cfg,
                                   const M7LatchResult *latch,
                                   size_t parent_loop,
                                   M7Loop *loop,
                                   M7Result *result,
                                   size_t *out_view_id) {
    Arena *arena = cfg->graph_arena;
    NormGraphView child;
    memset(&child, 0, sizeof(child));
    child.entry_block = loop->header;
    child.n_blocks = loop->n_body_blocks;
    child.blocks = loop->body_blocks;
    child.n_hidden_edges = 2;
    child.hidden_edges = arena_new_array(arena, size_t, 2);
    child.hidden_edges[0] = latch->back_edge;
    child.hidden_edges[1] = latch->exit_edge;
    child.parent_loop = parent_loop;

    return m7_push_view(result, arena, &child, out_view_id);
}

static bool m7_reserve_all(NormalizedCFG *cfg, const M7CyclePlan *plans,
                           size_t n_plans) {
    if (!normalized_cfg_reserve_blocks(cfg, n_plans * 3)) return false;
    size_t edge_extra = 0;
    for (size_t i = 0; i < n_plans; ++i) {
        edge_extra += plans[i].edges.n_entry_edges +
                      plans[i].edges.n_back_edges +
                      plans[i].edges.n_exit_edges + 4;
    }
    if (!normalized_cfg_reserve_edges(cfg, edge_extra)) return false;
    return true;
}

static bool m7_commit_cycle(NormalizedCFG *cfg,
                            const M7Config *config,
                            const NormGraphView *view,
                            M7CyclePlan *plan,
                            const M5ExitCombiner *combiner,
                            M7Result *result) {
    Arena *arena = cfg->graph_arena;
    size_t header;
    if (!m7_commit_entry_normalization(cfg, plan, &header)) return false;

    if (result->n_loops >= result->loops_cap &&
        !m7_result_grow_loops(result, arena)) {
        return false;
    }
    size_t loop_id = result->n_loops++;
    M7Loop *loop = &result->loops[loop_id];
    memset(loop, 0, sizeof(*loop));
    loop->id = loop_id;
    loop->parent_loop = view ? view->parent_loop : SIZE_MAX;

    M7LatchResult latch;
    if (!m7_create_single_exiting_latch(cfg, config, header, plan, loop,
                                        combiner, &latch)) {
        result->n_loops--;
        return false;
    }

    loop->header = header;
    loop->latch = latch.latch;
    loop->exit_dispatch = latch.exit_dispatch;
    loop->back_edge = latch.back_edge;
    loop->exit_edge = latch.exit_edge;
    loop->condition = latch.condition;

    if (!m7_build_body_block_list(arena, plan, header, latch.latch,
                                  &loop->body_blocks,
                                  &loop->n_body_blocks)) {
        return false;
    }

    if (!m7_populate_iteration_values(cfg, &latch, loop)) {
        return false;
    }

    if (!m7_register_loop_replacements(result, arena, cfg, loop, plan)) {
        return false;
    }

    size_t child_view;
    if (!m7_make_loop_body_view(cfg, &latch, loop_id, loop, result,
                                &child_view)) {
        return false;
    }
    if (!m7_worklist_push(result, arena, child_view)) return false;

    return true;
}

static bool m7_normalize_cycles(NormalizedCFG *cfg,
                                M5ExitCombiner *exit_combiner,
                                const CFGValueIndex *values,
                                const M7Config *config,
                                M7ViewWorkspace *ws,
                                M7Result *out) {
    memset(out, 0, sizeof(*out));
    if (!cfg || !values || !config || !ws) return false;
    Arena *graph = cfg->graph_arena;
    if (!graph) return false;

    if (!normalized_cfg_reserve_blocks(cfg, 16)) return false;

    if (!m7_push_root_view(cfg, out, graph)) return false;

    while (!m7_worklist_empty(out)) {
        size_t view_id = m7_worklist_pop(out);
        NormGraphView *view = &out->views[view_id];

        normalized_cfg_reset_phase_storage(cfg);

        if (!m7_workspace_ensure_graph_capacity(ws, cfg->n_blocks,
                                              cfg->n_edges)) {
            return false;
        }

        norm_view_membership_mark(&ws->membership, cfg, view);

        SCCInfo scc;
        if (!norm_compute_scc_view(cfg, view, &ws->membership, ws,
                                   cfg->phase_arena, &scc)) {
            return false;
        }

        RPOInfo rpo;
        if (!norm_compute_rpo_view(cfg, view, &ws->membership, ws,
                                   cfg->phase_arena, &rpo)) {
            return false;
        }
        DominanceInfo dom;
        if (!norm_compute_dominance_view(cfg, view, &rpo, &ws->membership, ws,
                                         cfg->phase_arena, &dom)) {
            return false;
        }

        M7CyclePlan *plans = NULL;
        size_t n_plans = 0;
        if (!m7_prepare_all_cyclic_components(cfg, view, &scc, &dom, values,
                                              config, &ws->membership, ws,
                                              cfg->phase_arena,
                                              &plans, &n_plans)) {
            return false;
        }
        if (n_plans == 0) continue;

        if (!m7_reserve_all(cfg, plans, n_plans)) return false;

        size_t *order =
            arena_new_array(cfg->phase_arena, size_t, n_plans);
        if (!order || !m7_sort_plans_downstream_first(cfg, plans, n_plans,
                                                      order)) {
            return false;
        }

        bool ok = true;
        for (size_t i = 0; ok && i < n_plans; ++i) {
            M7CyclePlan *plan = &plans[order[i]];
            if (!m7_refresh_mux_plans(cfg, view, plan, config, ws,
                                      cfg->phase_arena)) {
                ok = false;
                break;
            }
            ok = m7_commit_cycle(cfg, config, view, plan, exit_combiner, out);
        }
        if (!ok) return false;
    }

    normalized_cfg_reset_phase_storage(cfg);

    out->header_to_loop = arena_new_array(graph, size_t, cfg->n_blocks);
    for (size_t i = 0; i < cfg->n_blocks; ++i) {
        out->header_to_loop[i] = SIZE_MAX;
    }
    for (size_t i = 0; i < out->n_loops; ++i) {
        size_t hdr = out->loops[i].header;
        if (hdr < cfg->n_blocks) {
            out->header_to_loop[hdr] = i;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// M8 -- acyclic CFG to a strict structured plan.
//
// M7 has made every cycle explicit as one latch -> header backedge.  Hiding
// those edges leaves a DAG.  M8 computes post-dominance on that DAG and builds
// a tree: every incoming CFG edge remains an explicit TRANSFER node carrying
// its normalized block arguments.  Acyclic paths, including complete M7 loop
// subplans, may be split when a non-tree join cannot be represented by a single
// enclosing continuation; hidden backedges still terminate at the owning latch.
// ---------------------------------------------------------------------------

typedef struct {
    M8EmissionMetadata *meta;
    NormalizedCFG      *cfg;
    M7Result           *m7;
    MLIR_Context       *ctx;
    Arena              *graph;
    Arena              *phase;
    bool               *hidden_edges;
    size_t             *post_idom;
    size_t             *post_position;
    bool               *post_terminal;
    size_t             *post_exits;
    size_t              n_post_exits;
    size_t              post_root;
    size_t              post_n;
    size_t             *reach_epoch;
    size_t             *reach_stack;
    size_t             *active_continuations;
    size_t              current_reach_epoch;
    size_t              reach_cached_target;
} M8BuildCtx;

typedef struct {
    size_t node;
    size_t next_index;
} M8PostFrame;

static bool m8_edge_hidden(const M8BuildCtx *bc, size_t edge_id) {
    return edge_id < bc->cfg->n_edges && bc->hidden_edges[edge_id];
}

static bool m8_is_loop_back_edge(const M8BuildCtx *bc, size_t edge_id) {
    for (size_t i = 0; i < bc->m7->n_loops; ++i) {
        if (bc->m7->loops[i].back_edge == edge_id) return true;
    }
    return false;
}

static bool m8_edge_visible(const M8BuildCtx *bc, size_t edge_id) {
    return norm_edge_is_active(bc->cfg, edge_id) &&
           !m8_edge_hidden(bc, edge_id);
}

static bool m8_post_terminal(const M8BuildCtx *bc, size_t block_id) {
    if (bc->post_terminal) return bc->post_terminal[block_id];
    NormEdgeIter it = norm_out_edges(bc->cfg, block_id);
    size_t edge_id;
    while (norm_edge_iter_next(&it, &edge_id)) {
        if (m8_edge_visible(bc, edge_id)) return false;
    }
    return true;
}

// Return the Nth successor in the reverse graph.  The virtual root reaches all
// DAG exits; a real block reaches its original predecessors.
static size_t m8_post_successor(const M8BuildCtx *bc, size_t node,
                                size_t *cursor) {
    if (node == bc->post_root) {
        if (*cursor >= bc->n_post_exits) return SIZE_MAX;
        return bc->post_exits[(*cursor)++];
    }

    const NormAdjacency *incoming = &bc->cfg->blocks[node].incoming;
    while (*cursor < incoming->n) {
        size_t edge_id = incoming->edge_ids[(*cursor)++];
        if (m8_edge_visible(bc, edge_id)) {
            return bc->cfg->edges[edge_id].from;
        }
    }
    return SIZE_MAX;
}

static size_t m8_post_intersect(const M8BuildCtx *bc, size_t lhs,
                                size_t rhs) {
    while (lhs != rhs) {
        while (bc->post_position[lhs] > bc->post_position[rhs]) {
            lhs = bc->post_idom[lhs];
            if (lhs == SIZE_MAX) return SIZE_MAX;
        }
        while (bc->post_position[rhs] > bc->post_position[lhs]) {
            rhs = bc->post_idom[rhs];
            if (rhs == SIZE_MAX) return SIZE_MAX;
        }
    }
    return lhs;
}

// Cooper-Harvey-Kennedy dominance on the reversed acyclic normalized graph.
static bool m8_compute_post_dominance(M8BuildCtx *bc) {
    size_t n_nodes = bc->cfg->n_blocks + 1;
    bc->post_root = bc->cfg->n_blocks;
    bc->post_n = n_nodes;

    bool *visited = arena_new_array(bc->phase, bool, n_nodes);
    size_t *postorder = arena_new_array(bc->phase, size_t, n_nodes);
    M8PostFrame *stack = arena_new_array(bc->phase, M8PostFrame, n_nodes);
    bc->post_position = arena_new_array(bc->phase, size_t, n_nodes);
    bc->post_idom = arena_new_array(bc->phase, size_t, n_nodes);
    bc->post_terminal = arena_new_array(bc->phase, bool, bc->cfg->n_blocks);
    bc->post_exits = arena_new_array(bc->phase, size_t, bc->cfg->n_blocks);
    if (!visited || !postorder || !stack || !bc->post_position ||
        !bc->post_idom || !bc->post_terminal || !bc->post_exits) {
        return false;
    }
    bc->n_post_exits = 0;
    for (size_t bid = 0; bid < bc->cfg->n_blocks; ++bid) {
        bool terminal = false;
        if (bc->cfg->blocks[bid].active) {
            terminal = true;
            NormEdgeIter it = norm_out_edges(bc->cfg, bid);
            size_t edge_id;
            while (norm_edge_iter_next(&it, &edge_id)) {
                if (m8_edge_visible(bc, edge_id)) {
                    terminal = false;
                    break;
                }
            }
        }
        bc->post_terminal[bid] = terminal;
        if (terminal) bc->post_exits[bc->n_post_exits++] = bid;
    }
    if (bc->n_post_exits == 0) return false;
    memset(visited, 0, n_nodes * sizeof(bool));
    for (size_t i = 0; i < n_nodes; ++i) {
        bc->post_position[i] = SIZE_MAX;
        bc->post_idom[i] = SIZE_MAX;
    }

    size_t sp = 0, post_n = 0;
    visited[bc->post_root] = true;
    stack[sp++] = (M8PostFrame){bc->post_root, 0};
    while (sp > 0) {
        M8PostFrame *frame = &stack[sp - 1];
        size_t succ = m8_post_successor(bc, frame->node,
                                        &frame->next_index);
        if (succ != SIZE_MAX) {
            if (!visited[succ]) {
                visited[succ] = true;
                stack[sp++] = (M8PostFrame){succ, 0};
            }
            continue;
        }
        postorder[post_n++] = frame->node;
        --sp;
    }
    if (post_n == 0) return false;

    size_t *rpo = arena_new_array(bc->phase, size_t, post_n);
    if (!rpo) return false;
    for (size_t i = 0; i < post_n; ++i) {
        rpo[i] = postorder[post_n - i - 1];
        bc->post_position[rpo[i]] = i;
    }
    bc->post_idom[bc->post_root] = bc->post_root;

    // The M7 backedge-hidden graph is a DAG, so reverse RPO is topological:
    // every reverse predecessor already has an idom and one CHK pass suffices.
    for (size_t ri = 1; ri < post_n; ++ri) {
        size_t bid = rpo[ri];
        size_t new_idom = SIZE_MAX;

        // Predecessors in the reverse graph are visible forward successors.
        NormEdgeIter it = norm_out_edges(bc->cfg, bid);
        size_t edge_id;
        while (norm_edge_iter_next(&it, &edge_id)) {
            if (!m8_edge_visible(bc, edge_id)) continue;
            size_t succ = bc->cfg->edges[edge_id].to;
            if (succ >= n_nodes || bc->post_idom[succ] == SIZE_MAX) continue;
            new_idom = new_idom == SIZE_MAX
                ? succ : m8_post_intersect(bc, new_idom, succ);
        }
        if (m8_post_terminal(bc, bid)) new_idom = bc->post_root;
        if (new_idom == SIZE_MAX) return false;
        bc->post_idom[bid] = new_idom;
    }

    // Every active block must reach an exit after loop backedges are hidden.
    for (size_t bid = 0; bid < bc->cfg->n_blocks; ++bid) {
        if (bc->cfg->blocks[bid].active && bc->post_idom[bid] == SIZE_MAX) {
            return false;
        }
    }
    return true;
}

static bool m8_plan_grow_nodes(M8BuildCtx *bc) {
    M8EmissionMetadata *meta = bc->meta;
    size_t new_cap = meta->nodes_cap ? meta->nodes_cap * 2 : 32;
    M8PlanNode *next = arena_new_array(bc->graph, M8PlanNode, new_cap);
    if (!next) return false;
    if (meta->n_nodes > 0) {
        memcpy(next, meta->nodes, meta->n_nodes * sizeof(M8PlanNode));
    }
    meta->nodes = next;
    meta->nodes_cap = new_cap;
    return true;
}

static bool m8_plan_new_node(M8BuildCtx *bc, M8PlanKind kind,
                             size_t source_block, size_t *out_node) {
    M8EmissionMetadata *meta = bc->meta;
    if (meta->n_nodes >= meta->nodes_cap && !m8_plan_grow_nodes(bc)) {
        return false;
    }
    size_t id = meta->n_nodes++;
    M8PlanNode *node = &meta->nodes[id];
    memset(node, 0, sizeof(*node));
    node->kind = kind;
    node->source_block = source_block;
    if (out_node) *out_node = id;
    return true;
}

static size_t m8_edge_for_successor_slot(const M8BuildCtx *bc,
                                         size_t block_id, size_t slot) {
    size_t hidden_fallback = SIZE_MAX;
    NormEdgeIter it = norm_out_edges(bc->cfg, block_id);
    size_t edge_id;
    while (norm_edge_iter_next(&it, &edge_id)) {
        if (!norm_edge_is_active(bc->cfg, edge_id)) continue;
        if (bc->cfg->edges[edge_id].successor_slot != slot) continue;
        if (m8_edge_visible(bc, edge_id)) return edge_id;
        hidden_fallback = edge_id;
    }
    return hidden_fallback;
}

static bool m8_block_selector(const M8BuildCtx *bc, size_t block_id,
                              NormOperand *out) {
    norm_operand_init(out);
    const NormBlock *block = &bc->cfg->blocks[block_id];
    const NormTerminator *owned =
        normalized_cfg_block_owned_terminator(bc->cfg, block_id);
    if (owned && norm_operand_is_valid(&owned->selector)) {
        *out = owned->selector;
        return true;
    }
    if (block->source_terminator == MLIR_INVALID_HANDLE ||
        MLIR_GetOpNumOperands(block->source_terminator) == 0) {
        return false;
    }
    MLIR_ValueHandle value = MLIR_GetOpOperand(block->source_terminator, 0);
    *out = norm_operand_mlir(MLIR_GetValueType(value), value);
    return true;
}

static bool m8_copy_block_result_types(M8BuildCtx *bc, size_t block_id,
                                       MLIR_TypeHandle **out_types,
                                       size_t *out_n) {
    *out_types = NULL;
    *out_n = 0;
    if (block_id == SIZE_MAX || block_id >= bc->cfg->n_blocks) return true;
    const NormBlock *block = &bc->cfg->blocks[block_id];
    if (block->n_args == 0) return true;
    MLIR_TypeHandle *types =
        arena_new_array(bc->graph, MLIR_TypeHandle, block->n_args);
    if (!types) return false;
    for (size_t i = 0; i < block->n_args; ++i) types[i] = block->args[i].type;
    *out_types = types;
    *out_n = block->n_args;
    return true;
}

static bool m8_parse_snapshot_switch_cases(M8BuildCtx *bc, size_t block_id,
                                           int64_t **out_values,
                                           size_t *out_n) {
    *out_values = NULL;
    *out_n = 0;
    const NormBlock *block = &bc->cfg->blocks[block_id];
    size_t n_out = normalized_cfg_num_active_out_edges(bc->cfg, block_id);
    if (n_out == 0) return false;
    size_t n_cases = n_out - 1;
    if (n_cases == 0) return true;

    int64_t *values = arena_new_array(bc->graph, int64_t, n_cases);
    if (!values) return false;
    MLIR_AttributeHandle attr = MLIR_GetOpAttributeByName(
        block->source_terminator, "case_values");
    if (attr == MLIR_INVALID_HANDLE) return false;
    string text = MLIR_GetAttributeAsString(bc->ctx, attr);
    size_t p = 0;
    while (p < text.size && text.str[p] != ':') ++p;
    if (p < text.size) ++p;
    size_t parsed = 0;
    while (p < text.size && parsed < n_cases) {
        while (p < text.size && (text.str[p] == ' ' || text.str[p] == ',')) ++p;
        if (p >= text.size || text.str[p] == '>') break;
        bool negative = false;
        if (text.str[p] == '-') { negative = true; ++p; }
        uint64_t magnitude = 0;
        bool digit = false;
        while (p < text.size && text.str[p] >= '0' && text.str[p] <= '9') {
            uint64_t next_digit = (uint64_t)(text.str[p++] - '0');
            uint64_t limit = negative
                ? (uint64_t)INT64_MAX + 1u : (uint64_t)INT64_MAX;
            if (magnitude > (limit - next_digit) / 10u) return false;
            magnitude = magnitude * 10u + next_digit;
            digit = true;
        }
        if (!digit) return false;
        if (negative && magnitude == (uint64_t)INT64_MAX + 1u) {
            values[parsed++] = INT64_MIN;
        } else {
            int64_t value = (int64_t)magnitude;
            values[parsed++] = negative ? -value : value;
        }
    }
    if (parsed != n_cases) return false;
    *out_values = values;
    *out_n = n_cases;
    return true;
}

static bool m8_plan_from(M8BuildCtx *bc, size_t entry, size_t stop,
                         size_t owner_loop, size_t *out_node);

static bool m8_reaches_block(M8BuildCtx *bc, size_t from, size_t target) {
    if (from == target) return true;
    if (from >= bc->cfg->n_blocks || target >= bc->cfg->n_blocks) return false;
    if (bc->reach_cached_target != target) {
        if (++bc->current_reach_epoch == SIZE_MAX) {
            memset(bc->reach_epoch, 0,
                   bc->cfg->n_blocks * sizeof(size_t));
            bc->current_reach_epoch = 1;
        }
        size_t epoch = bc->current_reach_epoch;
        size_t sp = 0;
        bc->reach_epoch[target] = epoch;
        bc->reach_stack[sp++] = target;
        while (sp > 0) {
            sp--;
            size_t block_id = bc->reach_stack[sp];
            NormEdgeIter it = norm_in_edges(bc->cfg, block_id);
            size_t edge_id;
            while (norm_edge_iter_next(&it, &edge_id)) {
                if (!m8_edge_visible(bc, edge_id)) continue;
                size_t pred = bc->cfg->edges[edge_id].from;
                if (bc->reach_epoch[pred] == epoch) continue;
                bc->reach_epoch[pred] = epoch;
                bc->reach_stack[sp++] = pred;
            }
        }
        bc->reach_cached_target = target;
    }
    return bc->reach_epoch[from] == bc->current_reach_epoch;
}

// Global post-dominance intentionally uses a virtual exit.  Inside an outer
// branch region, however, a mixed return/continue child must use the caller's
// stop block as its local continuation when any child path reaches that stop.
static bool m8_is_common_reachable_continuation(M8BuildCtx *bc,
                                                  size_t block_id,
                                                  size_t target) {
    if (target >= bc->cfg->n_blocks || target == block_id ||
        !bc->cfg->blocks[target].active) {
        return false;
    }
    bool saw_edge = false;
    NormEdgeIter it = norm_out_edges(bc->cfg, block_id);
    size_t edge_id;
    while (norm_edge_iter_next(&it, &edge_id)) {
        if (!m8_edge_visible(bc, edge_id)) continue;
        saw_edge = true;
        size_t successor = bc->cfg->edges[edge_id].to;
        if (successor != target &&
            !m8_reaches_block(bc, successor, target)) {
            return false;
        }
    }
    return saw_edge;
}

static size_t m8_nearest_common_continuation(M8BuildCtx *bc,
                                             size_t block_id,
                                             size_t limit) {
    size_t best = SIZE_MAX;
    for (size_t candidate = 0; candidate < bc->cfg->n_blocks; ++candidate) {
        if (candidate == block_id || !bc->cfg->blocks[candidate].active) {
            continue;
        }
        if (bc->active_continuations[candidate] && candidate != limit) {
            continue;
        }
        if (limit != SIZE_MAX && candidate != limit &&
            !m8_reaches_block(bc, candidate, limit)) {
            continue;
        }
        if (!m8_is_common_reachable_continuation(bc, block_id, candidate)) {
            continue;
        }
        if (best == SIZE_MAX ||
            (candidate != best && m8_reaches_block(bc, candidate, best))) {
            best = candidate;
        }
    }
    return best;
}

static size_t m8_select_continuation(M8BuildCtx *bc, size_t block_id,
                                     size_t stop) {
    size_t continuation = bc->post_idom[block_id];
    if (continuation != bc->post_root &&
        !bc->post_terminal[continuation] &&
        bc->cfg->blocks[continuation].kind != NORM_BLOCK_LOOP_LATCH) {
        return continuation;
    }
    bool has_local_stop =
        stop != SIZE_MAX && stop < bc->cfg->n_blocks;
    size_t limit = has_local_stop
        ? stop
        : (continuation != bc->post_root ? continuation : SIZE_MAX);
    size_t common = m8_nearest_common_continuation(bc, block_id, limit);
    if (common != SIZE_MAX) return common;
    if (has_local_stop) {
        NormEdgeIter it = norm_out_edges(bc->cfg, block_id);
        size_t edge_id;
        while (norm_edge_iter_next(&it, &edge_id)) {
            if (!m8_edge_visible(bc, edge_id)) continue;
            if (m8_reaches_block(bc, bc->cfg->edges[edge_id].to, stop)) {
                return stop;
            }
        }
    }
    return continuation != bc->post_root ? continuation : SIZE_MAX;
}

static bool m8_plan_transfer(M8BuildCtx *bc, size_t edge_id, size_t stop,
                             size_t owner_loop, bool emit_source_block,
                             size_t *out_node) {
    if (!norm_edge_is_active(bc->cfg, edge_id)) return false;
    const NormEdge *edge = &bc->cfg->edges[edge_id];
    bool is_back_edge = m8_is_loop_back_edge(bc, edge_id);
    bool targets_owner_latch =
        owner_loop < bc->m7->n_loops &&
        edge->to == bc->m7->loops[owner_loop].latch;
    if (m8_edge_hidden(bc, edge_id) &&
        !is_back_edge && edge->to != stop && !targets_owner_latch) {
        return false;
    }
    size_t node_id;
    if (!m8_plan_new_node(bc, M8_PLAN_TRANSFER, edge->from, &node_id)) {
        return false;
    }
    size_t target_node = SIZE_MAX;
    bool targets_active_continuation =
        edge->to < bc->cfg->n_blocks &&
        bc->active_continuations[edge->to] > 0;
    if (edge->to != stop && !targets_owner_latch && !is_back_edge &&
        !targets_active_continuation &&
        !m8_plan_from(bc, edge->to, stop, owner_loop, &target_node)) {
        return false;
    }
    M8PlanNode *node = &bc->meta->nodes[node_id];
    node->as.transfer.edge_id = edge_id;
    node->as.transfer.target_block = edge->to;
    node->as.transfer.target_node = target_node;
    node->as.transfer.emit_source_block = emit_source_block;
    *out_node = node_id;
    return true;
}

static bool m8_plan_loop(M8BuildCtx *bc, size_t loop_id, size_t stop,
                         size_t owner_loop, size_t *out_node) {
    if (loop_id >= bc->m7->n_loops) return false;
    const M7Loop *loop = &bc->m7->loops[loop_id];
    size_t node_id;
    if (!m8_plan_new_node(bc, M8_PLAN_LOOP, SIZE_MAX, &node_id)) {
        return false;
    }

    size_t body_node = SIZE_MAX;
    if (!m8_plan_from(bc, loop->header, loop->latch, loop_id, &body_node)) {
        return false;
    }
    size_t continuation_node = SIZE_MAX;
    if (loop->exit_dispatch != stop &&
        !m8_plan_from(bc, loop->exit_dispatch, stop, owner_loop,
                      &continuation_node)) {
        return false;
    }
    M8PlanNode *node = &bc->meta->nodes[node_id];
    node->as.loop_plan.loop_id = loop_id;
    node->as.loop_plan.body_node = body_node;
    node->as.loop_plan.continuation_node = continuation_node;
    *out_node = node_id;
    return true;
}

static bool m8_plan_if(M8BuildCtx *bc, size_t block_id, size_t stop,
                       size_t owner_loop, size_t node_id) {
    size_t continuation = m8_select_continuation(bc, block_id, stop);
    if (continuation != SIZE_MAX && continuation == block_id) return false;

    size_t then_edge = m8_edge_for_successor_slot(bc, block_id, 0);
    size_t else_edge = m8_edge_for_successor_slot(bc, block_id, 1);
    if (then_edge == SIZE_MAX || else_edge == SIZE_MAX) return false;

    NormOperand condition;
    if (!m8_block_selector(bc, block_id, &condition)) return false;
    MLIR_TypeHandle *result_types = NULL;
    size_t n_results = 0;
    if (!m8_copy_block_result_types(bc, continuation, &result_types,
                                    &n_results)) {
        return false;
    }

    bool continuation_was_active =
        continuation != SIZE_MAX &&
        bc->active_continuations[continuation] > 0;
    if (continuation != SIZE_MAX) {
        bc->active_continuations[continuation]++;
    }
    size_t then_node = SIZE_MAX, else_node = SIZE_MAX;
    bool branches_ok =
        m8_plan_transfer(bc, then_edge, continuation, owner_loop, false,
                         &then_node) &&
        m8_plan_transfer(bc, else_edge, continuation, owner_loop, false,
                         &else_node);
    if (continuation != SIZE_MAX) {
        bc->active_continuations[continuation]--;
    }
    if (!branches_ok) return false;

    size_t continuation_node = SIZE_MAX;
    if (continuation != SIZE_MAX && continuation != stop &&
        !continuation_was_active &&
        !m8_plan_from(bc, continuation, stop, owner_loop,
                      &continuation_node)) {
        return false;
    }

    M8PlanNode *node = &bc->meta->nodes[node_id];
    node->as.if_plan.condition = condition;
    node->as.if_plan.then_node = then_node;
    node->as.if_plan.else_node = else_node;
    node->as.if_plan.continuation_block = continuation;
    node->as.if_plan.continuation_node = continuation_node;
    node->as.if_plan.result_types = result_types;
    node->as.if_plan.n_results = n_results;
    return true;
}

static bool m8_plan_switch(M8BuildCtx *bc, size_t block_id, size_t stop,
                           size_t owner_loop, size_t node_id) {
    size_t continuation = m8_select_continuation(bc, block_id, stop);
    if (continuation != SIZE_MAX && continuation == block_id) return false;

    NormOperand selector;
    if (!m8_block_selector(bc, block_id, &selector)) return false;

    const NormTerminator *owned =
        normalized_cfg_block_owned_terminator(bc->cfg, block_id);
    int64_t *case_values = NULL;
    size_t n_cases = 0;
    size_t default_edge = SIZE_MAX;
    size_t *case_edges = NULL;
    if (owned) {
        n_cases = owned->n_cases;
        default_edge = owned->default_edge_id;
        if (n_cases > 0) {
            case_values = arena_new_array(bc->graph, int64_t, n_cases);
            case_edges = arena_new_array(bc->phase, size_t, n_cases);
            if (!case_values || !case_edges) return false;
            for (size_t i = 0; i < n_cases; ++i) {
                case_values[i] = owned->case_values[i];
            }
            memcpy(case_edges, owned->case_edge_ids,
                   n_cases * sizeof(size_t));
        }
    } else {
        if (!m8_parse_snapshot_switch_cases(bc, block_id, &case_values,
                                            &n_cases)) {
            return false;
        }
        size_t *slot_edges = arena_new_array(
            bc->phase, size_t, n_cases + 1);
        if (!slot_edges) return false;
        for (size_t i = 0; i <= n_cases; ++i) slot_edges[i] = SIZE_MAX;
        NormEdgeIter edge_it = norm_out_edges(bc->cfg, block_id);
        size_t edge_id;
        while (norm_edge_iter_next(&edge_it, &edge_id)) {
            if (!m8_edge_visible(bc, edge_id)) continue;
            size_t slot = bc->cfg->edges[edge_id].successor_slot;
            if (slot > n_cases || slot_edges[slot] != SIZE_MAX) return false;
            slot_edges[slot] = edge_id;
        }
        default_edge = slot_edges[0];
        if (n_cases > 0) {
            case_edges = arena_new_array(bc->phase, size_t, n_cases);
            if (!case_edges) return false;
            for (size_t i = 0; i < n_cases; ++i) {
                case_edges[i] = slot_edges[i + 1];
            }
        }
    }
    if (default_edge == SIZE_MAX) return false;

    size_t *case_nodes = n_cases
        ? arena_new_array(bc->graph, size_t, n_cases) : NULL;
    if (n_cases > 0 && !case_nodes) return false;
    bool continuation_was_active =
        continuation != SIZE_MAX &&
        bc->active_continuations[continuation] > 0;
    if (continuation != SIZE_MAX) {
        bc->active_continuations[continuation]++;
    }
    bool transfers_ok = true;
    for (size_t i = 0; i < n_cases; ++i) {
        if (case_edges[i] == SIZE_MAX ||
            !m8_plan_transfer(bc, case_edges[i], continuation, owner_loop,
                              false, &case_nodes[i])) {
            transfers_ok = false;
            break;
        }
    }
    size_t default_node = SIZE_MAX;
    if (transfers_ok &&
        !m8_plan_transfer(bc, default_edge, continuation, owner_loop, false,
                          &default_node)) {
        transfers_ok = false;
    }
    if (continuation != SIZE_MAX) {
        bc->active_continuations[continuation]--;
    }
    if (!transfers_ok) return false;

    size_t continuation_node = SIZE_MAX;
    if (continuation != SIZE_MAX && continuation != stop &&
        !continuation_was_active &&
        !m8_plan_from(bc, continuation, stop, owner_loop,
                      &continuation_node)) {
        return false;
    }
    MLIR_TypeHandle *result_types = NULL;
    size_t n_results = 0;
    if (!m8_copy_block_result_types(bc, continuation, &result_types,
                                    &n_results)) {
        return false;
    }

    M8PlanNode *node = &bc->meta->nodes[node_id];
    node->as.switch_plan.selector = selector;
    node->as.switch_plan.case_values = case_values;
    node->as.switch_plan.case_nodes = case_nodes;
    node->as.switch_plan.n_cases = n_cases;
    node->as.switch_plan.default_node = default_node;
    node->as.switch_plan.continuation_block = continuation;
    node->as.switch_plan.continuation_node = continuation_node;
    node->as.switch_plan.result_types = result_types;
    node->as.switch_plan.n_results = n_results;
    return true;
}

static bool m8_plan_from(M8BuildCtx *bc, size_t entry, size_t stop,
                         size_t owner_loop, size_t *out_node) {
    *out_node = SIZE_MAX;
    if (entry == stop) return true;
    if (entry >= bc->cfg->n_blocks || !bc->cfg->blocks[entry].active) {
        return false;
    }

    size_t nested_loop = entry < bc->cfg->n_blocks
        ? bc->m7->header_to_loop[entry] : SIZE_MAX;
    if (nested_loop != SIZE_MAX && nested_loop != owner_loop) {
        return m8_plan_loop(bc, nested_loop, stop, owner_loop, out_node);
    }

    CFGTermKind term_kind = normalized_cfg_block_term_kind(bc->cfg, entry);
    switch (term_kind) {
    case CFG_TERM_BR: {
        size_t edge_id = m8_edge_for_successor_slot(bc, entry, 0);
        if (edge_id == SIZE_MAX) return false;
        return m8_plan_transfer(bc, edge_id, stop, owner_loop, true, out_node);
    }
    case CFG_TERM_COND_BR: {
        size_t node_id;
        if (!m8_plan_new_node(bc, M8_PLAN_IF, entry, &node_id)) {
            return false;
        }
        if (!m8_plan_if(bc, entry, stop, owner_loop, node_id)) return false;
        *out_node = node_id;
        return true;
    }
    case CFG_TERM_SWITCH: {
        size_t node_id;
        if (!m8_plan_new_node(bc, M8_PLAN_SWITCH, entry, &node_id)) {
            return false;
        }
        if (!m8_plan_switch(bc, entry, stop, owner_loop, node_id)) {
            return false;
        }
        *out_node = node_id;
        return true;
    }
    case CFG_TERM_RETURN: {
        size_t node_id;
        if (!m8_plan_new_node(bc, M8_PLAN_RETURN, entry, &node_id)) {
            return false;
        }
        *out_node = node_id;
        return true;
    }
    case CFG_TERM_UNREACHABLE: {
        size_t node_id;
        if (!m8_plan_new_node(bc, M8_PLAN_UNREACHABLE, entry, &node_id)) {
            return false;
        }
        *out_node = node_id;
        return true;
    }
    default:
        return false;
    }
}

static bool m8_build_structured_plan(CFGInfoScratch *scratch) {
    if (!scratch) return false;
    M8EmissionMetadata *meta = &scratch->m8;
    NormalizedCFG *cfg = &scratch->norm;
    Arena *graph = cfg->graph_arena;
    Arena *phase = cfg->phase_arena;
    if (!graph || !phase || cfg->n_blocks == 0) return false;

    M8BuildCtx bc;
    memset(&bc, 0, sizeof(bc));
    bc.meta = meta;
    bc.cfg = cfg;
    bc.m7 = &scratch->m7;
    bc.ctx = scratch->ctx;
    bc.graph = graph;
    bc.phase = phase;
    bc.reach_cached_target = SIZE_MAX;

    meta->root_node = SIZE_MAX;
    bc.hidden_edges = arena_new_array(phase, bool, cfg->n_edges);
    bc.reach_epoch = arena_new_array(phase, size_t, cfg->n_blocks);
    bc.reach_stack = arena_new_array(phase, size_t, cfg->n_blocks);
    bc.active_continuations =
        arena_new_array(phase, size_t, cfg->n_blocks);
    if (cfg->n_edges > 0 && !bc.hidden_edges) {
        return false;
    }
    if (!bc.reach_epoch || !bc.reach_stack || !bc.active_continuations) {
        return false;
    }
    memset(bc.reach_epoch, 0, cfg->n_blocks * sizeof(size_t));
    memset(bc.active_continuations, 0,
           cfg->n_blocks * sizeof(size_t));
    if (cfg->n_edges > 0) {
        memset(bc.hidden_edges, 0, cfg->n_edges * sizeof(bool));
    }
    for (size_t i = 0; i < bc.m7->n_loops; ++i) {
        size_t back_edge = bc.m7->loops[i].back_edge;
        if (back_edge >= cfg->n_edges) return false;
        bc.hidden_edges[back_edge] = true;
    }

    if (!m8_compute_post_dominance(&bc)) return false;
    if (!m8_plan_from(&bc, cfg->entry_block, SIZE_MAX, SIZE_MAX,
                      &meta->root_node)) {
        return false;
    }
    normalized_cfg_reset_phase_storage(cfg);
    return meta->root_node != SIZE_MAX;
}

// ---------------------------------------------------------------------------
// M9 -- emit the strict M8 tree directly as structured WasmSSA.
// ---------------------------------------------------------------------------

static bool m8_emit_plan_node(M8EmitCtx *ec, size_t node_id);
static bool m8_emit_loop_latch(M8EmitCtx *ec, size_t loop_id);

static bool m8_emit_push_label(M8EmitCtx *ec, M8LabelKind kind,
                               size_t target_block, size_t loop_id) {
    if (ec->n_labels >= ec->labels_cap) {
        size_t new_cap = ec->labels_cap ? ec->labels_cap * 2 : 16;
        M8Label *next = (M8Label *)arena_alloc(
            ec->F->arena, new_cap * sizeof(M8Label));
        if (!next) return false;
        if (ec->n_labels) {
            memcpy(next, ec->labels, ec->n_labels * sizeof(M8Label));
        }
        ec->labels = next;
        ec->labels_cap = new_cap;
    }
    ec->labels[ec->n_labels++] = (M8Label){kind, target_block, loop_id};
    return true;
}

static void m8_emit_pop_label(M8EmitCtx *ec) {
    if (ec->n_labels > 0) --ec->n_labels;
}

static bool m8_emit_find_label(const M8EmitCtx *ec, size_t target_block,
                               uint32_t *out_depth, M8LabelKind *out_kind) {
    for (size_t i = ec->n_labels; i > 0; --i) {
        const M8Label *label = &ec->labels[i - 1];
        if (label->target_block != target_block) continue;
        *out_depth = (uint32_t)(ec->n_labels - i);
        if (out_kind) *out_kind = label->kind;
        return true;
    }
    return false;
}

static MLIR_ValueHandle m8_emit_zero(FnCtx *F, MLIR_TypeHandle type) {
    uint8_t vt = wasm_vt(F->ctx, type);
    if (vt == 0) return MLIR_INVALID_HANDLE;
    wasmssa_op_t op = {0};
    op.type = OP_TYPE_WASMSSA_CONST;
    op.valtype = vt;
    op.i_const = 0;
    op.has_result = true;
    return commit_op(F, &op);
}

static bool m8_emit_operand(M8EmitCtx *ec, const NormOperand *operand,
                            MLIR_ValueHandle *out) {
    if (operand->kind == NORM_OPERAND_MLIR) {
        return fn_vmap_get(ec->F, operand->as.mlir_value, out);
    }
    if (operand->kind == NORM_OPERAND_UNDEF) {
        *out = m8_emit_zero(ec->F, operand->type);
        return *out != MLIR_INVALID_HANDLE;
    }
    return fn_emit_norm_operand(ec->F, operand, out);
}

static bool m8_emit_result_vts(FnCtx *F, MLIR_TypeHandle *types, size_t n,
                               uint8_t **out) {
    *out = NULL;
    if (n == 0) return true;
    uint8_t *vts = (uint8_t *)arena_alloc(F->arena, n);
    if (!vts) return false;
    for (size_t i = 0; i < n; ++i) {
        vts[i] = wasm_vt(F->ctx, types[i]);
        if (vts[i] == 0) return false;
    }
    *out = vts;
    return true;
}

static bool m8_emit_edge_values(M8EmitCtx *ec, size_t edge_id,
                                MLIR_ValueHandle **out_values,
                                size_t *out_n) {
    *out_values = NULL;
    *out_n = 0;
    size_t n = normalized_cfg_edge_num_operands(ec->cfg, edge_id);
    if (n == 0) return true;
    MLIR_ValueHandle *values = (MLIR_ValueHandle *)arena_alloc(
        ec->F->arena, n * sizeof(MLIR_ValueHandle));
    if (!values) return false;
    for (size_t i = 0; i < n; ++i) {
        NormOperand operand;
        if (!normalized_cfg_edge_operand_resolve(ec->cfg, edge_id, i,
                                                  &operand) ||
            !m8_emit_operand(ec, &operand, &values[i])) {
            return false;
        }
    }
    *out_values = values;
    *out_n = n;
    return true;
}

static bool m8_emit_bind_values(M8EmitCtx *ec, size_t block_id,
                                MLIR_ValueHandle *values, size_t n) {
    if (block_id >= ec->cfg->n_blocks ||
        ec->cfg->blocks[block_id].n_args != n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!m8_emit_bind_arg(ec, block_id, i, values[i])) return false;
    }
    return true;
}

static bool m8_emit_block_data(M8EmitCtx *ec, size_t block_id) {
    if (block_id >= ec->cfg->n_blocks) return false;
    const NormBlock *block = &ec->cfg->blocks[block_id];
    if (block->kind != NORM_BLOCK_ORIGINAL) return true;
    if (block->mlir_block == MLIR_INVALID_HANDLE) return false;

    ec->F->current_block = block->mlir_block;
    size_t n_ops = MLIR_GetBlockNumOps(block->mlir_block);
    for (size_t i = 0; i < n_ops; ++i) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block->mlir_block, i);
        if (op == block->source_terminator) continue;
        if (!lower_op(ec->F, op)) return false;
    }
    return true;
}

static bool m8_emit_transfer(M8EmitCtx *ec, const M8TransferPlan *transfer,
                             size_t *out_next_node) {
    *out_next_node = SIZE_MAX;
    MLIR_ValueHandle *values = NULL;
    size_t n_values = 0;
    if (!m8_emit_edge_values(ec, transfer->edge_id, &values, &n_values)) {
        return false;
    }

    // Reaching the active loop latch performs the loop's exit/backedge split.
    for (size_t i = ec->n_labels; i > 0; --i) {
        const M8Label *label = &ec->labels[i - 1];
        if (label->kind != M8_LABEL_LOOP ||
            label->loop_id >= ec->meta->scratch->m7.n_loops) {
            continue;
        }
        const M7Loop *loop = &ec->meta->scratch->m7.loops[label->loop_id];
        if (loop->latch == transfer->target_block) {
            if (!m8_emit_bind_values(ec, transfer->target_block,
                                     values, n_values)) {
                return false;
            }
            return m8_emit_loop_latch(ec, label->loop_id);
        }
    }

    uint32_t depth;
    M8LabelKind kind;
    if (m8_emit_find_label(ec, transfer->target_block, &depth, &kind)) {
        if (kind == M8_LABEL_LOOP) {
            for (size_t i = ec->n_labels; i > 0; --i) {
                const M8Label *label = &ec->labels[i - 1];
                if (label->kind != M8_LABEL_LOOP ||
                    label->target_block != transfer->target_block ||
                    label->loop_id >= ec->meta->scratch->m7.n_loops) {
                    continue;
                }
                return m8_emit_loop_latch(ec, label->loop_id);
            }
        }
        if (kind == M8_LABEL_IF && depth == 0) {
            emit_block_return(ec->F, values, n_values);
        } else {
            emit_br_args(ec->F, depth, values, n_values);
        }
        return true;
    }

    if (!m8_emit_bind_values(ec, transfer->target_block, values, n_values)) {
        return false;
    }
    if (transfer->target_node == SIZE_MAX) return false;
    *out_next_node = transfer->target_node;
    return true;
}

static bool m8_emit_if_node(M8EmitCtx *ec, const M8IfPlan *plan) {
    FnCtx *F = ec->F;
    MLIR_ValueHandle condition;
    if (!m8_emit_operand(ec, &plan->condition, &condition)) return false;

    uint8_t *result_vts = NULL;
    if (!m8_emit_result_vts(F, plan->result_types, plan->n_results,
                            &result_vts)) {
        return false;
    }
    MLIR_BlockHandle saved = F->body_block;
    size_t checkpoint = ec->n_changes;

    MLIR_BlockHandle then_block = MLIR_CreateBlock(F->ctx);
    F->body_block = then_block;
    if (!m8_emit_push_label(ec, M8_LABEL_IF, plan->continuation_block,
                            SIZE_MAX) ||
        !m8_emit_plan_node(ec, plan->then_node)) {
        F->body_block = saved;
        return false;
    }
    m8_emit_pop_label(ec);
    m8_emit_restore_bindings(ec, checkpoint);

    MLIR_BlockHandle else_block = MLIR_CreateBlock(F->ctx);
    F->body_block = else_block;
    if (!m8_emit_push_label(ec, M8_LABEL_IF, plan->continuation_block,
                            SIZE_MAX) ||
        !m8_emit_plan_node(ec, plan->else_node)) {
        F->body_block = saved;
        return false;
    }
    m8_emit_pop_label(ec);
    m8_emit_restore_bindings(ec, checkpoint);
    F->body_block = saved;

    MLIR_RegionHandle then_region = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, then_region, then_block);
    MLIR_RegionHandle else_region = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, else_region, else_block);
    MLIR_RegionHandle regions[2] = {then_region, else_region};
    MLIR_ValueHandle cond_ops[1] = {condition};
    MLIR_ValueHandle *results = plan->n_results
        ? (MLIR_ValueHandle *)arena_alloc(
              F->arena, plan->n_results * sizeof(MLIR_ValueHandle))
        : NULL;
    MLIR_AttributeHandle attrs[1];
    size_t n_attrs = 0;
    if (plan->n_results) {
        attrs[n_attrs++] = attr_s_hex(F->ctx, F->arena, "result_types",
                                      result_vts, plan->n_results);
    }
    MLIR_OpHandle if_op = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_IF, attrs, n_attrs, cond_ops, 1, regions, 2,
        result_vts, plan->n_results, results);
    MLIR_AppendBlockOp(F->ctx, F->body_block, if_op);

    if (plan->continuation_block != SIZE_MAX) {
        if (!m8_emit_bind_values(ec, plan->continuation_block,
                                 results, plan->n_results)) {
            return false;
        }
        uint32_t depth;
        M8LabelKind kind;
        if (m8_emit_find_label(ec, plan->continuation_block,
                               &depth, &kind)) {
            if (kind == M8_LABEL_IF && depth == 0) {
                emit_block_return(F, results, plan->n_results);
            } else {
                emit_br_args(F, depth, results, plan->n_results);
            }
            return true;
        }
        if (plan->continuation_node != SIZE_MAX) {
            return m8_emit_plan_node(ec, plan->continuation_node);
        }
    }
    return true;
}

static MLIR_ValueHandle m8_emit_integer_constant(FnCtx *F, uint8_t vt,
                                                  int64_t value) {
    if (vt != WT_I32 && vt != WT_I64) return MLIR_INVALID_HANDLE;
    wasmssa_op_t op = {0};
    op.type = OP_TYPE_WASMSSA_CONST;
    op.valtype = vt;
    op.i_const = vt == WT_I32 ? (int64_t)(int32_t)value : value;
    op.has_result = true;
    return commit_op(F, &op);
}

static MLIR_ValueHandle m8_emit_integer_eq(FnCtx *F, uint8_t vt,
                                            MLIR_ValueHandle lhs,
                                            MLIR_ValueHandle rhs) {
    if (vt != WT_I32 && vt != WT_I64) return MLIR_INVALID_HANDLE;
    MLIR_ValueHandle operands[2] = {lhs, rhs};
    wasmssa_op_t op = {0};
    op.type = OP_TYPE_WASMSSA_BINOP;
    op.valtype = WT_I32;
    op.wasm_opcode = vt == WT_I32 ? 0x46 : 0x51;
    op.n_operands = 2;
    op.operands = operands;
    op.has_result = true;
    return commit_op(F, &op);
}

static bool m8_emit_switch_chain(M8EmitCtx *ec, const M8SwitchPlan *plan,
                                 size_t case_index,
                                 MLIR_ValueHandle selector,
                                 const uint8_t *result_vts,
                                 MLIR_ValueHandle *out_results) {
    if (case_index != 0 || plan->n_cases == 0) return false;
    FnCtx *F = ec->F;
    uint8_t selector_vt = wasm_vt(F->ctx, plan->selector.type);
    MLIR_BlockHandle saved = F->body_block;
    size_t base_checkpoint = ec->n_changes;

    // Materialize the default arm first. At runtime it is nested inside all
    // case tests, so account for every enclosing if label when computing br.
    MLIR_BlockHandle else_block = MLIR_CreateBlock(F->ctx);
    F->body_block = else_block;
    for (size_t i = 0; i < plan->n_cases; ++i) {
        if (!m8_emit_push_label(ec, M8_LABEL_IF,
                                plan->continuation_block, SIZE_MAX)) {
            F->body_block = saved;
            return false;
        }
    }
    if (!m8_emit_plan_node(ec, plan->default_node)) {
        F->body_block = saved;
        return false;
    }
    for (size_t i = 0; i < plan->n_cases; ++i) m8_emit_pop_label(ec);
    m8_emit_restore_bindings(ec, base_checkpoint);

    // Wrap from the last case to the first. This keeps C-stack use constant
    // even for switches with thousands of cases.
    for (size_t rev = plan->n_cases; rev > 0; --rev) {
        size_t i = rev - 1;
        MLIR_BlockHandle then_block = MLIR_CreateBlock(F->ctx);
        F->body_block = then_block;
        for (size_t depth = 0; depth <= i; ++depth) {
            if (!m8_emit_push_label(ec, M8_LABEL_IF,
                                    plan->continuation_block, SIZE_MAX)) {
                F->body_block = saved;
                return false;
            }
        }
        if (!m8_emit_plan_node(ec, plan->case_nodes[i])) {
            F->body_block = saved;
            return false;
        }
        for (size_t depth = 0; depth <= i; ++depth) m8_emit_pop_label(ec);
        m8_emit_restore_bindings(ec, base_checkpoint);

        MLIR_BlockHandle container = i == 0 ? saved : MLIR_CreateBlock(F->ctx);
        F->body_block = container;
        MLIR_ValueHandle constant = m8_emit_integer_constant(
            F, selector_vt, plan->case_values[i]);
        if (constant == MLIR_INVALID_HANDLE) return false;
        MLIR_ValueHandle compare = m8_emit_integer_eq(
            F, selector_vt, selector, constant);
        if (compare == MLIR_INVALID_HANDLE) return false;

        MLIR_RegionHandle then_region = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, then_region, then_block);
        MLIR_RegionHandle else_region = MLIR_CreateRegion(F->ctx);
        MLIR_AppendRegionBlock(F->ctx, else_region, else_block);
        MLIR_RegionHandle regions[2] = {then_region, else_region};
        MLIR_ValueHandle cond_ops[1] = {compare};
        MLIR_AttributeHandle attrs[1];
        size_t n_attrs = 0;
        if (plan->n_results) {
            attrs[n_attrs++] = attr_s_hex(
                F->ctx, F->arena, "result_types",
                result_vts, plan->n_results);
        }
        MLIR_ValueHandle *level_results = i == 0
            ? out_results
            : (plan->n_results
                ? (MLIR_ValueHandle *)arena_alloc(
                      F->arena,
                      plan->n_results * sizeof(MLIR_ValueHandle))
                : NULL);
        MLIR_OpHandle if_op = make_op_n(
            F->ctx, OP_TYPE_WASMSSA_IF, attrs, n_attrs, cond_ops, 1,
            regions, 2, result_vts, plan->n_results, level_results);
        MLIR_AppendBlockOp(F->ctx, container, if_op);
        if (i > 0) {
            emit_block_return(F, level_results, plan->n_results);
            else_block = container;
        }
    }
    F->body_block = saved;
    return true;
}

static bool m8_emit_switch_node(M8EmitCtx *ec, const M8SwitchPlan *plan) {
    if (plan->n_cases == 0) {
        const M8PlanNode *transfer =
            &ec->meta->nodes[plan->default_node];
        if (transfer->kind != M8_PLAN_TRANSFER) return false;

        // An LLVM switch with no explicit cases still owns a real default
        // edge. It usually falls through to the local continuation, but it
        // can also target an enclosing label or an active loop latch. Those
        // targets require the same structured branch emission as any other
        // transfer; merely binding their edge values would silently fall
        // through.
        bool targets_loop_latch = false;
        for (size_t i = ec->n_labels; i > 0; --i) {
            const M8Label *label = &ec->labels[i - 1];
            if (label->kind != M8_LABEL_LOOP ||
                label->loop_id >= ec->meta->scratch->m7.n_loops) {
                continue;
            }
            if (ec->meta->scratch->m7.loops[label->loop_id].latch ==
                transfer->as.transfer.target_block) {
                targets_loop_latch = true;
                break;
            }
        }
        uint32_t depth;
        M8LabelKind kind;
        if (transfer->as.transfer.target_node != SIZE_MAX ||
            targets_loop_latch ||
            m8_emit_find_label(ec, transfer->as.transfer.target_block,
                               &depth, &kind)) {
            size_t next_node = SIZE_MAX;
            if (!m8_emit_transfer(ec, &transfer->as.transfer, &next_node)) {
                return false;
            }
            return next_node == SIZE_MAX ||
                   m8_emit_plan_node(ec, next_node);
        }

        MLIR_ValueHandle *values = NULL;
        size_t n_values = 0;
        if (!m8_emit_edge_values(ec, transfer->as.transfer.edge_id,
                                 &values, &n_values) ||
            !m8_emit_bind_values(ec, transfer->as.transfer.target_block,
                                 values, n_values)) {
            return false;
        }
        if (transfer->as.transfer.target_node != SIZE_MAX &&
            !m8_emit_plan_node(ec, transfer->as.transfer.target_node)) {
            return false;
        }
        if (plan->continuation_node != SIZE_MAX) {
            return m8_emit_plan_node(ec, plan->continuation_node);
        }
        return true;
    }

    MLIR_ValueHandle selector;
    if (!m8_emit_operand(ec, &plan->selector, &selector)) return false;
    uint8_t selector_vt = wasm_vt(ec->F->ctx, plan->selector.type);
    if (selector_vt != WT_I32 && selector_vt != WT_I64) return false;
    uint8_t *result_vts = NULL;
    if (!m8_emit_result_vts(ec->F, plan->result_types, plan->n_results,
                            &result_vts)) {
        return false;
    }
    MLIR_ValueHandle *results = plan->n_results
        ? (MLIR_ValueHandle *)arena_alloc(
              ec->F->arena, plan->n_results * sizeof(MLIR_ValueHandle))
        : NULL;
    if (!m8_emit_switch_chain(ec, plan, 0, selector, result_vts, results)) {
        return false;
    }
    if (plan->continuation_block != SIZE_MAX &&
        !m8_emit_bind_values(ec, plan->continuation_block,
                             results, plan->n_results)) {
        return false;
    }
    if (plan->continuation_block != SIZE_MAX) {
        uint32_t depth;
        M8LabelKind kind;
        if (m8_emit_find_label(ec, plan->continuation_block,
                               &depth, &kind)) {
            if (kind == M8_LABEL_IF && depth == 0) {
                emit_block_return(ec->F, results, plan->n_results);
            } else {
                emit_br_args(ec->F, depth, results, plan->n_results);
            }
            return true;
        }
    }
    if (plan->continuation_node != SIZE_MAX) {
        return m8_emit_plan_node(ec, plan->continuation_node);
    }
    return true;
}

static bool m8_emit_loop_latch(M8EmitCtx *ec, size_t loop_id) {
    const M7Loop *loop = &ec->meta->scratch->m7.loops[loop_id];
    MLIR_ValueHandle condition;
    if (!m8_emit_operand(ec, &loop->condition, &condition)) return false;
    MLIR_ValueHandle exit_condition = emit_eqz(ec->F, condition);

    MLIR_ValueHandle *exit_values = NULL, *back_values = NULL;
    size_t n_exit = 0, n_back = 0;
    if (!m8_emit_edge_values(ec, loop->exit_edge, &exit_values, &n_exit) ||
        !m8_emit_edge_values(ec, loop->back_edge, &back_values, &n_back)) {
        return false;
    }

    FnCtx *F = ec->F;
    MLIR_BlockHandle loop_body = F->body_block;
    MLIR_BlockHandle exit_block = MLIR_CreateBlock(F->ctx);
    F->body_block = exit_block;
    if (!m8_emit_push_label(ec, M8_LABEL_IF, SIZE_MAX, SIZE_MAX)) return false;
    uint32_t exit_depth;
    if (!m8_emit_find_label(ec, loop->exit_dispatch, &exit_depth, NULL)) {
        return false;
    }
    emit_br_args(F, exit_depth, exit_values, n_exit);
    m8_emit_pop_label(ec);
    F->body_block = loop_body;

    MLIR_RegionHandle exit_region = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, exit_region, exit_block);
    MLIR_RegionHandle regions[1] = {exit_region};
    MLIR_ValueHandle cond_ops[1] = {exit_condition};
    MLIR_OpHandle exit_if = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_IF, NULL, 0, cond_ops, 1, regions, 1,
        NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, F->body_block, exit_if);

    uint32_t back_depth;
    if (!m8_emit_find_label(ec, loop->header, &back_depth, NULL)) return false;
    emit_br_args(F, back_depth, back_values, n_back);
    return true;
}

static bool m8_emit_loop_node(M8EmitCtx *ec, const M8LoopPlan *plan) {
    if (plan->loop_id >= ec->meta->scratch->m7.n_loops) return false;
    const M7Loop *loop = &ec->meta->scratch->m7.loops[plan->loop_id];
    const NormBlock *header = &ec->cfg->blocks[loop->header];
    const NormBlock *dispatch = &ec->cfg->blocks[loop->exit_dispatch];
    FnCtx *F = ec->F;

    size_t n_iter = header->n_args;
    MLIR_ValueHandle *init_values = n_iter
        ? (MLIR_ValueHandle *)arena_alloc(
              F->arena, n_iter * sizeof(MLIR_ValueHandle))
        : NULL;
    uint8_t *iter_vts = n_iter ? (uint8_t *)arena_alloc(F->arena, n_iter) : NULL;
    for (size_t i = 0; i < n_iter; ++i) {
        if (!m8_emit_lookup_arg(ec, loop->header, i, &init_values[i])) {
            return false;
        }
        iter_vts[i] = wasm_vt(F->ctx, header->args[i].type);
        if (iter_vts[i] == 0) return false;
    }

    MLIR_ValueHandle *loop_args = n_iter
        ? (MLIR_ValueHandle *)arena_alloc(
              F->arena, n_iter * sizeof(MLIR_ValueHandle))
        : NULL;
    MLIR_BlockHandle loop_body =
        make_block_with_args(F->ctx, iter_vts, n_iter, loop_args);
    MLIR_BlockHandle saved = F->body_block;
    size_t checkpoint = ec->n_changes;
    F->body_block = loop_body;
    for (size_t i = 0; i < n_iter; ++i) {
        if (!m8_emit_bind_arg(ec, loop->header, i, loop_args[i])) {
            F->body_block = saved;
            return false;
        }
    }
    if (!m8_emit_push_label(ec, M8_LABEL_BLOCK, loop->exit_dispatch,
                            plan->loop_id) ||
        !m8_emit_push_label(ec, M8_LABEL_LOOP, loop->header,
                            plan->loop_id) ||
        !m8_emit_plan_node(ec, plan->body_node)) {
        F->body_block = saved;
        return false;
    }
    m8_emit_pop_label(ec);
    m8_emit_pop_label(ec);
    m8_emit_restore_bindings(ec, checkpoint);
    F->body_block = saved;

    MLIR_RegionHandle loop_region = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, loop_region, loop_body);
    MLIR_OpHandle loop_op = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_LOOP, NULL, 0, init_values, n_iter,
        &loop_region, 1, NULL, 0, NULL);

    MLIR_BlockHandle outer_body = MLIR_CreateBlock(F->ctx);
    MLIR_AppendBlockOp(F->ctx, outer_body, loop_op);
    MLIR_OpHandle unreachable = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_UNREACHABLE, NULL, 0, NULL, 0,
        NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(F->ctx, outer_body, unreachable);
    MLIR_RegionHandle outer_region = MLIR_CreateRegion(F->ctx);
    MLIR_AppendRegionBlock(F->ctx, outer_region, outer_body);

    uint8_t *exit_vts = NULL;
    MLIR_TypeHandle *exit_types = dispatch->n_args
        ? (MLIR_TypeHandle *)arena_alloc(
              F->arena, dispatch->n_args * sizeof(MLIR_TypeHandle))
        : NULL;
    for (size_t i = 0; i < dispatch->n_args; ++i) {
        exit_types[i] = dispatch->args[i].type;
    }
    if (!m8_emit_result_vts(F, exit_types, dispatch->n_args, &exit_vts)) {
        return false;
    }
    MLIR_ValueHandle *results = dispatch->n_args
        ? (MLIR_ValueHandle *)arena_alloc(
              F->arena, dispatch->n_args * sizeof(MLIR_ValueHandle))
        : NULL;
    MLIR_OpHandle block_op = make_op_n(
        F->ctx, OP_TYPE_WASMSSA_BLOCK, NULL, 0, NULL, 0, &outer_region, 1,
        exit_vts, dispatch->n_args, results);
    MLIR_AppendBlockOp(F->ctx, F->body_block, block_op);

    if (!m8_emit_bind_values(ec, loop->exit_dispatch,
                             results, dispatch->n_args)) {
        return false;
    }
    if (plan->continuation_node != SIZE_MAX) {
        return m8_emit_plan_node(ec, plan->continuation_node);
    }
    return true;
}

static bool m8_emit_return_node(M8EmitCtx *ec, size_t block_id) {
    const NormTerminator *term =
        normalized_cfg_block_owned_terminator(ec->cfg, block_id);
    if (!term || term->kind != CFG_TERM_RETURN) return false;
    MLIR_ValueHandle *values = term->n_return_values
        ? (MLIR_ValueHandle *)arena_alloc(
              ec->F->arena,
              term->n_return_values * sizeof(MLIR_ValueHandle))
        : NULL;
    for (size_t i = 0; i < term->n_return_values; ++i) {
        if (!m8_emit_operand(ec, &term->return_values[i], &values[i])) {
            return false;
        }
    }
    if (ec->F->frame_size > 0 &&
        ec->F->sp_value != MLIR_INVALID_HANDLE) {
        MLIR_ValueHandle frame =
            emit_const_i32(ec->F, (int32_t)ec->F->frame_size);
        MLIR_ValueHandle restored =
            emit_add_i32(ec->F, ec->F->sp_value, frame);
        emit_global_set(ec->F, 0, restored);
    }
    MLIR_OpHandle ret = make_op_n(
        ec->F->ctx, OP_TYPE_WASMSSA_RETURN, NULL, 0,
        values, term->n_return_values, NULL, 0, NULL, 0, NULL);
    MLIR_AppendBlockOp(ec->F->ctx, ec->F->body_block, ret);
    return true;
}

static bool m8_emit_plan_node(M8EmitCtx *ec, size_t node_id) {
    while (node_id != SIZE_MAX) {
        if (node_id >= ec->meta->n_nodes) return false;
        const M8PlanNode *node = &ec->meta->nodes[node_id];
        if (node->source_block != SIZE_MAX &&
            (node->kind != M8_PLAN_TRANSFER ||
             node->as.transfer.emit_source_block) &&
            !m8_emit_block_data(ec, node->source_block)) {
            return false;
        }
        switch (node->kind) {
        case M8_PLAN_TRANSFER: {
            size_t next_node = SIZE_MAX;
            if (!m8_emit_transfer(ec, &node->as.transfer, &next_node)) {
                return false;
            }
            if (next_node == SIZE_MAX) return true;
            node_id = next_node;
            continue;
        }
        case M8_PLAN_IF:
            if (!m8_emit_if_node(ec, &node->as.if_plan)) {
                fprintf(stderr, "wasmssa-lower: failed to emit M8 if node %zu (block %zu)\n",
                        node_id, node->source_block);
                return false;
            }
            return true;
        case M8_PLAN_SWITCH:
            if (!m8_emit_switch_node(ec, &node->as.switch_plan)) {
                fprintf(stderr, "wasmssa-lower: failed to emit M8 switch node %zu (block %zu)\n",
                        node_id, node->source_block);
                return false;
            }
            return true;
        case M8_PLAN_LOOP:
            if (!m8_emit_loop_node(ec, &node->as.loop_plan)) {
                fprintf(stderr, "wasmssa-lower: failed to emit M8 loop node %zu\n",
                        node_id);
                return false;
            }
            return true;
        case M8_PLAN_RETURN:
            if (!m8_emit_return_node(ec, node->source_block)) {
                fprintf(stderr, "wasmssa-lower: failed to emit M8 return node %zu (block %zu)\n",
                        node_id, node->source_block);
                return false;
            }
            return true;
        case M8_PLAN_UNREACHABLE:
            emit_unreachable(ec->F);
            return true;
        default:
            return false;
        }
    }
    return false;
}

static bool m8_emit_function_body(FnCtx *F) {
    if (!F || !F->m8_plan || !F->m8_plan->built) return false;
    const M8EmissionMetadata *meta = F->m8_plan;
    const NormalizedCFG *cfg = &meta->scratch->norm;
    M8EmitCtx ec;
    memset(&ec, 0, sizeof(ec));
    ec.F = F;
    ec.meta = meta;
    ec.cfg = cfg;
    ec.arg_offsets = (size_t *)arena_alloc(
        F->arena, (cfg->n_blocks + 1) * sizeof(size_t));
    if (!ec.arg_offsets) return false;
    size_t total_args = 0;
    for (size_t i = 0; i < cfg->n_blocks; ++i) {
        ec.arg_offsets[i] = total_args;
        total_args += cfg->blocks[i].n_args;
    }
    ec.arg_offsets[cfg->n_blocks] = total_args;
    ec.n_arg_slots = total_args;
    ec.arg_values = total_args
        ? (MLIR_ValueHandle *)arena_alloc(
              F->arena, total_args * sizeof(MLIR_ValueHandle))
        : NULL;
    ec.arg_bound = total_args
        ? (bool *)arena_alloc(F->arena, total_args * sizeof(bool)) : NULL;
    if (total_args && (!ec.arg_values || !ec.arg_bound)) return false;
    if (total_args) {
        memset(ec.arg_values, 0, total_args * sizeof(MLIR_ValueHandle));
        memset(ec.arg_bound, 0, total_args * sizeof(bool));
    }
    value_index_map_init(&ec.arg_slot_by_mlir, F->arena,
                         total_args ? total_args * 2 : 16);
    for (size_t bid = 0; bid < cfg->n_blocks; ++bid) {
        const NormBlock *block = &cfg->blocks[bid];
        if (block->kind != NORM_BLOCK_ORIGINAL ||
            block->mlir_block == MLIR_INVALID_HANDLE) {
            continue;
        }
        size_t n_args = MLIR_GetBlockNumArgs(block->mlir_block);
        for (size_t i = 0; i < n_args; ++i) {
            value_index_map_put(&ec.arg_slot_by_mlir, F->arena,
                                MLIR_GetBlockArg(block->mlir_block, i),
                                ec.arg_offsets[bid] + i);
        }
    }
    ec.changes_cap = total_args * 2 + meta->n_nodes * 2 + 16;
    ec.changes = (M8BindingChange *)arena_alloc(
        F->arena, ec.changes_cap * sizeof(M8BindingChange));
    ec.labels_cap = meta->n_nodes + meta->scratch->m7.n_loops * 2 + 8;
    ec.labels = (M8Label *)arena_alloc(
        F->arena, ec.labels_cap * sizeof(M8Label));
    if (!ec.changes || !ec.labels) return false;

    size_t entry = cfg->entry_block;
    const NormBlock *entry_block = &cfg->blocks[entry];
    for (size_t i = 0; i < entry_block->n_args; ++i) {
        if (entry_block->mlir_block == MLIR_INVALID_HANDLE ||
            i >= MLIR_GetBlockNumArgs(entry_block->mlir_block)) {
            return false;
        }
        MLIR_ValueHandle mapped;
        if (!vmap_get(F, MLIR_GetBlockArg(entry_block->mlir_block, i),
                      &mapped) ||
            !m8_emit_bind_arg(&ec, entry, i, mapped)) {
            return false;
        }
    }

    F->m8_emit = &ec;
    bool ok = m8_emit_plan_node(&ec, meta->root_node);
    F->m8_emit = NULL;
    return ok;
}

// --- Arena pool and per-target scratch ---

// Create separate graph and phase bump allocators.
static bool cfg_analysis_arena_init(CFGAnalysisArena *pool) {
    memset(pool, 0, sizeof(*pool));
    pool->graph_arena = arena_create(80 * 1024);
    pool->phase_arena = arena_create(16 * 1024);
    if (!pool->graph_arena || !pool->phase_arena) {
        if (pool->graph_arena) arena_destroy(pool->graph_arena);
        if (pool->phase_arena) arena_destroy(pool->phase_arena);
        memset(pool, 0, sizeof(*pool));
        return false;
    }
    pool->graph_base_pos = arena_get_pos(pool->graph_arena);
    pool->phase_base_pos = arena_get_pos(pool->phase_arena);
    return true;
}

// Reset phase arena to scratch base position.
static void cfg_phase_arena_reset(CFGAnalysisArena *pool) {
    if (pool->phase_arena) {
        arena_reset(pool->phase_arena, pool->phase_base_pos);
    }
}

// Reset graph arena to snapshot+norm graph base position.
static void cfg_graph_arena_reset(CFGAnalysisArena *pool) {
    if (pool->graph_arena) arena_reset(pool->graph_arena, pool->graph_base_pos);
}

// Reset graph and phase arenas (new target build).
static void cfg_analysis_arena_reset(CFGAnalysisArena *pool) {
    cfg_graph_arena_reset(pool);
    cfg_phase_arena_reset(pool);
}

// Destroy all arenas in a CFGAnalysisArena pool.
static void cfg_analysis_arena_destroy(CFGAnalysisArena *pool) {
    if (pool->graph_arena) arena_destroy(pool->graph_arena);
    if (pool->phase_arena) arena_destroy(pool->phase_arena);
    memset(pool, 0, sizeof(*pool));
}

// Zero-init CFGInfoScratch view.
static void cfg_info_scratch_init(CFGInfoScratch *scratch) {
    memset(scratch, 0, sizeof(*scratch));
}

// Clear scratch view without freeing pool arenas.
static void cfg_info_scratch_clear(CFGInfoScratch *scratch) {
    scratch->ctx = NULL;
    memset(&scratch->cfg, 0, sizeof(scratch->cfg));
    memset(&scratch->norm, 0, sizeof(scratch->norm));
    memset(&scratch->values, 0, sizeof(scratch->values));
    memset(&scratch->m7, 0, sizeof(scratch->m7));
    memset(&scratch->m7_ws, 0, sizeof(scratch->m7_ws));
    memset(&scratch->m8, 0, sizeof(scratch->m8));
    memset(&scratch->m5, 0, sizeof(scratch->m5));
}

// Build CFGInfo + NormalizedCFG for one region in the shared pool.
static bool cfg_info_scratch_build(CFGInfoScratch *scratch,
                                   CFGAnalysisArena *pool,
                                   MLIR_Context *ctx,
                                   MLIR_RegionHandle region) {
    cfg_info_scratch_clear(scratch);
    if (!pool || !pool->graph_arena || !pool->phase_arena) {
        return false;
    }
    cfg_analysis_arena_reset(pool);
    scratch->ctx = ctx;
    if (!cfg_info_build(pool->graph_arena, region, &scratch->cfg)) {
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!scratch->cfg.blocks[0].reachable) {
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    // Direct CF lowering can leave dead blocks after terminal paths.  The
    // snapshot marks them unreachable and NormalizedCFG keeps them inactive.
    if (!normalized_cfg_build_from_snapshot(pool, &scratch->cfg,
                                            &scratch->norm)) {
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!m5_normalize_returns(&scratch->norm, &scratch->m5)) {
        fprintf(stderr, "wasmssa-cfg: M5 return normalization failed\n");
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!m7_view_workspace_init(&scratch->m7_ws, pool->graph_arena,
                                scratch->norm.n_blocks,
                                scratch->norm.n_edges)) {
        fprintf(stderr, "wasmssa-cfg: M7 view workspace init failed\n");
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    if (!cfg_value_index_build(&scratch->cfg, pool->graph_arena,
                               &scratch->values)) {
        fprintf(stderr, "wasmssa-cfg: CFG value index build failed\n");
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    M7Config m7_config;
    m7_config.flag_type = MLIR_CreateTypeInteger(ctx, 32, true);
    if (!m7_normalize_cycles(&scratch->norm, &scratch->m5, &scratch->values,
                               &m7_config, &scratch->m7_ws, &scratch->m7)) {
        fprintf(stderr, "wasmssa-cfg: M7 loop normalization failed\n");
        cfg_analysis_arena_reset(pool);
        cfg_info_scratch_clear(scratch);
        return false;
    }
    return true;
}

// --- CFG branch detection for per-function lowering ---

// True when op is llvm.func.
static bool op_is_llvm_func(MLIR_OpHandle op) {
    if (MLIR_GetOpType(op) == OP_TYPE_LLVM_FUNC) return true;
    return name_eq(MLIR_GetOpName(op), "llvm.func");
}

// True when op is builtin.module.
static bool op_is_builtin_module(MLIR_OpHandle op) {
    return MLIR_GetOpType(op) == OP_TYPE_MODULE;
}

// llvm.br / llvm.cond_br / llvm.switch — the LLVM CFG terminators we lift.
static bool op_is_llvm_cfg_branch(MLIR_OpHandle op) {
    string n = MLIR_GetOpName(op);
    return name_eq(n, "llvm.br") || name_eq(n, "llvm.cond_br") ||
           name_eq(n, "llvm.switch");
}

static bool llvm_func_region_has_cfg_branch(MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        MLIR_OpHandle term = MLIR_GetBlockTerminator(b);
        if (term != MLIR_INVALID_HANDLE && op_is_llvm_cfg_branch(term)) {
            return true;
        }
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            size_t nr = MLIR_GetOpNumRegions(op);
            for (size_t ri = 0; ri < nr; ++ri) {
                if (llvm_func_region_has_cfg_branch(MLIR_GetOpRegion(op, ri))) {
                    return true;
                }
            }
        }
    }
    return false;
}

// True when llvm.func has a non-empty entry block body.
static bool llvm_func_has_defined_body(MLIR_OpHandle fn) {
    if (!op_is_llvm_func(fn)) return false;
    if (MLIR_GetOpNumRegions(fn) == 0) return false;
    return MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}

// =============================================================================
// Module walker.
// =============================================================================
static bool sig_for_func(MLIR_Context *ctx, Arena *arena, MLIR_OpHandle fn,
                         uint8_t **out_p, size_t *out_np,
                         uint8_t **out_r, size_t *out_nr) {
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    if (ftya == MLIR_INVALID_HANDLE) return false;
    MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(ftya);
    size_t ni = MLIR_GetTypeFunctionNumInputs(fty);
    size_t no = MLIR_GetTypeFunctionNumResults(fty);
    bool is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
    size_t pn = ni + (is_vararg ? 1 : 0);
    uint8_t *p = (uint8_t *)arena_alloc(arena, pn ? pn : 1);
    for (size_t i = 0; i < ni; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionInput(fty, i));
        if (v == 0) return false;
        p[i] = v;
    }
    if (is_vararg) p[ni] = WT_I32;  // hidden va_list pointer
    uint8_t *r = (uint8_t *)arena_alloc(arena, no ? no : 1);
    for (size_t i = 0; i < no; i++) {
        uint8_t v = wasm_vt(ctx, MLIR_GetTypeFunctionResult(fty, i));
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
static void emit_import_global(MLIR_Context *ctx, MLIR_BlockHandle body,
                               string sym_name, uint32_t size,
                               uint32_t align_pow, bool is_const,
                               const uint8_t *data, const char *relocs_str,
                               size_t relocs_len) {
    MLIR_AttributeHandle attrs[16];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",
                         sym_name.str ? sym_name.str : "",
                         sym_name.str ? sym_name.size : 0);
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
}

// Per-slot scratch used by the insertvalue-chain folder in lower_global.
// Declared at file scope so tinyc (which lacks function-local struct
// declarations) accepts it.
typedef struct LowerGlobalCEntry {
    MLIR_ValueHandle v;
    uint64_t bits;
} LowerGlobalCEntry;

// Lower one llvm.mlir.global op directly into a wasmssa.import_global op
// appended to `body`. Returns false on failure (no op is appended).
static bool lower_global(MLIR_Context *ctx, Arena *arena,
                         MLIR_BlockHandle body, MLIR_OpHandle op) {
    MLIR_AttributeHandle sa = find_attr(op, "sym_name");
    if (sa == MLIR_INVALID_HANDLE) return false;
    string sym = MLIR_GetAttributeString(sa);

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

    MLIR_AttributeHandle va = find_attr(op, "value");
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind ak = MLIR_GetAttributeKind(va);
        if (ak == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(va);
            size = (uint32_t)s.size;
            emit_import_global(ctx, body, sym, size, align_pow,
                               is_const, (const uint8_t *)s.str, NULL, 0);
            return true;
        }
        if (ak == MLIR_ATTR_KIND_INTEGER || ak == MLIR_ATTR_KIND_FLOAT) {
            if (size == 0) return false;
            data = (uint8_t *)arena_alloc(arena, size);
            memset(data, 0, size);
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
                } else { return false; }
            }
            for (uint32_t i = 0; i < size && i < 8; i++)
                data[i] = (uint8_t)(bits >> (8 * i));
            emit_import_global(ctx, body, sym, size, align_pow,
                               is_const, data, NULL, 0);
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
            string pending_target = (string){0};
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
                    if (ts.size && ts.str[0] == '@') { ts.str++; ts.size--; }
                    pending_target = ts;
                } else if (name_eq(bn, "llvm.return")) {
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
                    emit_import_global(ctx, body, sym, size, align_pow,
                                       is_const, data, buf, off);
                    return true;
                }
            }
        }
    }

    // Region-init globals (alternative pattern): a chain of
    //   %u = llvm.mlir.undef : !llvm.array<N x iX>
    //   %c0 = llvm.mlir.constant : iX
    //   %1 = llvm.insertvalue %u, %c0 [0]
    //   ...
    //   llvm.return %k : !llvm.array<N x iX>
    // produced by MLIR_CreateLLVMGlobalArrayInit's upstream impl. Unpack
    // the constants into a raw byte buffer matching the global's layout.
    if (MLIR_GetOpNumRegions(op) > 0 && ga != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle gty = MLIR_GetAttributeTypeValue(ga);
        if (gty != MLIR_INVALID_HANDLE && MLIR_IsTypeLLVMArray(gty)) {
            uint64_t arr_n = MLIR_GetTypeLLVMArrayNumElements(gty);
            MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(gty);
            unsigned esz = type_size_bytes(ctx, et);
            if (esz > 0 && esz <= 8 && arr_n > 0 && size > 0 &&
                size == arr_n * esz) {
                MLIR_RegionHandle rgn = MLIR_GetOpRegion(op, 0);
                if (MLIR_GetRegionNumBlocks(rgn) > 0) {
                    MLIR_BlockHandle blk = MLIR_GetRegionBlock(rgn, 0);
                    size_t nb = MLIR_GetBlockNumOps(blk);
                    // Collect (value, int) pairs for llvm.mlir.constant ops.
                    LowerGlobalCEntry *cs = (LowerGlobalCEntry *)arena_alloc(arena,
                        sizeof(LowerGlobalCEntry) * (nb + 1));
                    size_t ncs = 0;
                    uint8_t *buf = (uint8_t *)arena_alloc(arena, size);
                    memset(buf, 0, size);
                    bool ok = true;
                    bool saw_undef_or_zero = false;
                    bool saw_return = false;
                    for (size_t bi = 0; bi < nb && ok; bi++) {
                        MLIR_OpHandle bop = MLIR_GetBlockOp(blk, bi);
                        string bn = MLIR_GetOpName(bop);
                        if (name_eq(bn, "llvm.mlir.undef") ||
                            name_eq(bn, "llvm.mlir.zero") ||
                            name_eq(bn, "llvm.mlir.poison")) {
                            saw_undef_or_zero = true;
                        } else if (name_eq(bn, "llvm.mlir.constant")) {
                            MLIR_AttributeHandle ca = find_attr(bop, "value");
                            if (ca == MLIR_INVALID_HANDLE) { ok = false; break; }
                            MLIR_AttrKind ck = MLIR_GetAttributeKind(ca);
                            uint64_t bits = 0;
                            if (ck == MLIR_ATTR_KIND_INTEGER) {
                                bits = (uint64_t)MLIR_GetAttributeInteger(ca);
                            } else if (ck == MLIR_ATTR_KIND_FLOAT) {
                                double d = MLIR_GetAttributeFloat(ca);
                                if (esz == 4) {
                                    float f = (float)d; uint32_t b32;
                                    memcpy(&b32, &f, 4); bits = b32;
                                } else if (esz == 8) {
                                    memcpy(&bits, &d, 8);
                                } else { ok = false; break; }
                            } else { ok = false; break; }
                            if (MLIR_GetOpNumResults(bop) > 0) {
                                cs[ncs].v = MLIR_GetOpResult(bop, 0);
                                cs[ncs].bits = bits;
                                ncs++;
                            }
                        } else if (name_eq(bn, "llvm.insertvalue")) {
                            if (MLIR_GetOpNumOperands(bop) < 2) { ok = false; break; }
                            // Position attr — printed as "array<i64: N, ...>".
                            MLIR_AttributeHandle pa = find_attr(bop, "position");
                            if (pa == MLIR_INVALID_HANDLE) { ok = false; break; }
                            string ps = MLIR_GetAttributeAsString(ctx, pa);
                            // Skip past first ':' (or '[').
                            const char *p = ps.str;
                            const char *end = ps.str + ps.size;
                            const char *colon = NULL;
                            for (const char *q = p; q < end; q++) {
                                if (*q == ':') { colon = q; break; }
                            }
                            if (colon) p = colon + 1;
                            else if (p < end && *p == '[') p++;
                            while (p < end && (*p == ' ' || *p == '\t')) p++;
                            int64_t idx = 0;
                            bool neg = false;
                            if (p < end && (*p == '-' || *p == '+')) { neg = (*p == '-'); p++; }
                            if (p >= end || *p < '0' || *p > '9') { ok = false; break; }
                            while (p < end && *p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); p++; }
                            if (neg) idx = -idx;
                            if (idx < 0 || (uint64_t)idx >= arr_n) { ok = false; break; }
                            // Look up the scalar value's integer literal.
                            MLIR_ValueHandle sv = MLIR_GetOpOperand(bop, 1);
                            bool found = false;
                            uint64_t bits = 0;
                            for (size_t i = 0; i < ncs; i++) {
                                if (cs[i].v == sv) { bits = cs[i].bits; found = true; break; }
                            }
                            if (!found) { ok = false; break; }
                            // Encode `bits` into buf at offset idx*esz (LE).
                            for (unsigned b2 = 0; b2 < esz; b2++)
                                buf[(uint64_t)idx * esz + b2] = (uint8_t)(bits >> (8 * b2));
                            // Track the insertvalue's result too so
                            // nested chains could be supported.
                            if (MLIR_GetOpNumResults(bop) > 0) {
                                cs[ncs].v = MLIR_GetOpResult(bop, 0);
                                cs[ncs].bits = 0;  // not a scalar
                                ncs++;
                            }
                        } else if (name_eq(bn, "llvm.return")) {
                            saw_return = true;
                            break;
                        }
                    }
                    if (ok && saw_undef_or_zero && saw_return) {
                        emit_import_global(ctx, body, sym, size, align_pow,
                                           is_const, buf, NULL, 0);
                        return true;
                    }
                }
            }
        }
    }

    // No initializer found (e.g. uninitialized scalar global). Zero-init.
    if (size == 0) return false;
    data = (uint8_t *)arena_alloc(arena, size);
    memset(data, 0, size);
    emit_import_global(ctx, body, sym, size, align_pow,
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
                             MLIR_BlockHandle body, string sym_name,
                             string import_module, string import_name,
                             const uint8_t *param_types, size_t n_params,
                             const uint8_t *result_types, size_t n_results) {
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",
                         sym_name.str ? sym_name.str : "",
                         sym_name.str ? sym_name.size : 0);
    attrs[na++] = attr_s_hex(ctx, arena, "param_types", param_types, n_params);
    attrs[na++] = attr_s_hex(ctx, arena, "result_types", result_types, n_results);
    attrs[na++] = attr_b(ctx, "exported", false);
    if (import_module.size > 0) {
        attrs[na++] = attr_s(ctx, "import_module", import_module.str, import_module.size);
    }
    if (import_name.size > 0) {
        attrs[na++] = attr_s(ctx, "import_name", import_name.str, import_name.size);
    }
    MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
                               attrs, na, NULL, 0, NULL, 0, 0, NULL);
    MLIR_AppendBlockOp(ctx, body, op);
}

// Recursively emit body-less llvm.func ops as wasmssa.import_func into
// `body`, including those under nested builtin.module ops.
static bool emit_import_funcs_in_region(MLIR_Context *ctx, Arena *arena,
                                        MLIR_BlockHandle body,
                                        MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    if (!emit_import_funcs_in_region(
                            ctx, arena, body, MLIR_GetOpRegion(op, 0))) {
                        return false;
                    }
                }
                continue;
            }
            if (!op_is_llvm_func(op)) continue;
            if (llvm_func_has_defined_body(op)) continue;
            uint8_t *p, *r;
            size_t np, nr;
            if (!sig_for_func(ctx, arena, op, &p, &np, &r, &nr)) return false;
            MLIR_AttributeHandle sa = find_attr(op, "sym_name");
            string nm = MLIR_GetAttributeString(sa);
            string imod = {0}, iname = {0};
            MLIR_AttributeHandle iam = find_attr(op, "wasm.import_module");
            if (iam != MLIR_INVALID_HANDLE) imod = MLIR_GetAttributeString(iam);
            MLIR_AttributeHandle ian = find_attr(op, "wasm.import_name");
            if (ian != MLIR_INVALID_HANDLE) iname = MLIR_GetAttributeString(ian);
            emit_import_func(ctx, arena, body, nm, imod, iname, p, np, r, nr);
        }
    }
    return true;
}

// Recursively lower defined llvm.func ops into wasmssa.func ops appended to
// `body`, including those under nested builtin.module ops.
static bool emit_defined_funcs_in_region(MLIR_Context *ctx, Arena *arena,
                                         ModCtx *mod, MLIR_BlockHandle body,
                                         MLIR_RegionHandle region,
                                         CFGAnalysisArena *cfg_pool) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    if (!emit_defined_funcs_in_region(
                            ctx, arena, mod, body,
                            MLIR_GetOpRegion(op, 0), cfg_pool)) {
                        return false;
                    }
                }
                continue;
            }
            if (!llvm_func_has_defined_body(op)) continue;
            uint8_t *p, *r;
            size_t np, nr;
            MLIR_AttributeHandle sa = find_attr(op, "sym_name");
            string sym = MLIR_GetAttributeString(sa);
            if (!sig_for_func(ctx, arena, op, &p, &np, &r, &nr)) {
                fprintf(stderr, "wasmssa-lower: unsupported signature for '%.*s'\n",
                        (int)sym.size, sym.str);
                return false;
            }
            bool is_main = (sym.size == 4 && memcmp(sym.str, "main", 4) == 0);
            string nm = is_main ? str_lit("__original_main") : sym;
            MLIR_RegionHandle fn_region = MLIR_GetOpRegion(op, 0);
            bool lowered = false;
            if (cfg_pool && llvm_func_region_has_cfg_branch(fn_region)) {
                CFGInfoScratch scratch;
                cfg_info_scratch_init(&scratch);
                if (!cfg_info_scratch_build(&scratch, cfg_pool, ctx, fn_region)) {
                    fprintf(stderr, "wasmssa-lower: CFG analysis failed for '%.*s'\n",
                            (int)sym.size, sym.str);
                    cfg_info_scratch_clear(&scratch);
                    return false;
                }
                if (!m8_build_emission_metadata(&scratch)) {
                    fprintf(stderr, "wasmssa-lower: M8 planning failed for '%.*s'\n",
                            (int)sym.size, sym.str);
                    cfg_analysis_arena_reset(cfg_pool);
                    cfg_info_scratch_clear(&scratch);
                    return false;
                }
                lowered = lower_function_impl(
                    ctx, arena, mod, body, nm, is_main, op, p, np, r, nr,
                    &scratch);
                cfg_analysis_arena_reset(cfg_pool);
                cfg_info_scratch_clear(&scratch);
            } else {
                lowered = lower_function_impl(ctx, arena, mod, body, nm, is_main,
                                            op, p, np, r, nr, NULL);
            }
            if (!lowered) {
                fprintf(stderr, "wasmssa-lower: emission failed for '%.*s'\n",
                        (int)sym.size, sym.str);
                return false;
            }
        }
    }
    return true;
}

// Recursively lower llvm.mlir.global ops into wasmssa.import_global ops
// appended to `body`, including those under nested builtin.module ops.
static bool emit_globals_in_region(MLIR_Context *ctx, Arena *arena,
                                   MLIR_BlockHandle body,
                                   MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    if (!emit_globals_in_region(
                            ctx, arena, body, MLIR_GetOpRegion(op, 0))) {
                        return false;
                    }
                }
                continue;
            }
            if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.global")) continue;
            if (!lower_global(ctx, arena, body, op)) {
                fprintf(stderr, "wasmssa-lower: failed to lower global\n");
                return false;
            }
        }
    }
    return true;
}

// When WASMSSA_LOWER_DEBUG=1, list top-level LLVM ops in the input module
// that are not lowered (only llvm.func bodies and llvm.mlir.global are
// consumed; anything else at module scope is intentionally ignored).
static void warn_skipped_module_ops(MLIR_RegionHandle region) {
    if (!wasmssa_lower_debug()) return;
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t bi = 0; bi < nb; ++bi) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t oi = 0; oi < no; ++oi) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, oi);
            if (op_is_builtin_module(op)) {
                if (MLIR_GetOpNumRegions(op) > 0) {
                    warn_skipped_module_ops(MLIR_GetOpRegion(op, 0));
                }
                continue;
            }
            if (op_is_llvm_func(op) || name_eq(MLIR_GetOpName(op), "llvm.mlir.global"))
                continue;
            string n = MLIR_GetOpName(op);
            if (n.size >= 5 && memcmp(n.str, "llvm.", 5) == 0) {
                fprintf(stderr,
                        "wasmssa-lower: module-scope LLVM op not lowered: "
                        "'%.*s'\n", (int)n.size, n.str);
            }
        }
    }
}

// =============================================================================
// Public stage 1 entry point: walk the LLVM-dialect module and emit a
// wasmssa-form `builtin.module` directly. Each function/global is emitted
// straight into the output module body — no module-level intermediate is
// retained. Walks nested builtin.module ops recursively so every llvm.func
// and llvm.mlir.global is lowered. Output ordering per scope: import_funcs,
// then defined funcs, then import_globals.
// =============================================================================
MLIR_OpHandle mlir_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);

    Arena *arena = MLIR_GetArenaAllocator(ctx);

    MLIR_BlockHandle  body  = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, body);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    ModCtx mod = {0};
    mod.ctx = ctx;
    mod.arena = arena;
    mod.body = body;

    // Pass 1: imported funcs (including nested builtin.module bodies).
    if (!emit_import_funcs_in_region(ctx, arena, body, mr)) {
        return MLIR_INVALID_HANDLE;
    }

    CFGAnalysisArena cfg_pool;
    if (!cfg_analysis_arena_init(&cfg_pool)) {
        return MLIR_INVALID_HANDLE;
    }

    // Pass 2: defined funcs in source order, recursing into nested modules.
    if (!emit_defined_funcs_in_region(ctx, arena, &mod, body, mr, &cfg_pool)) {
        cfg_analysis_arena_destroy(&cfg_pool);
        return MLIR_INVALID_HANDLE;
    }

    // Pass 3: globals last, recursing into nested modules.
    if (!emit_globals_in_region(ctx, arena, body, mr)) {
        cfg_analysis_arena_destroy(&cfg_pool);
        return MLIR_INVALID_HANDLE;
    }

    cfg_analysis_arena_destroy(&cfg_pool);

    warn_skipped_module_ops(mr);

    return out_module;
}

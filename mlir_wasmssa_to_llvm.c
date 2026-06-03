// See mlir_wasmssa_to_llvm.h for the role of this pass in the pipeline.
//
// Lifts a `wasmssa` builtin.module into the in-house `llvm` MLIR dialect so
// the single unified backend (mlir_llvm_to_aarch64.c) can serve the
// `--from-wasm` self-host path. This is the WASM-input counterpart to the
// C-frontend emit.c (which produces the `llvm` dialect directly).
//
// Coverage grows test-by-test, mirroring how mlir_wasmssa_to_wmir.c was
// built up. The current milestone handles single-block, straight-line
// integer functions (the macho_exit / macho_arith shape): module walk,
// import_func recognition, per-function locals-as-alloca lowering, and the
// const / local_get / local_set / add / sub / binop(arith) / extend_i32_s /
// call / return ops. Any unsupported op (control flow, floats, linear
// memory, globals, ...) makes the lowering fail cleanly with a diagnostic so
// the opt-in path degrades gracefully while the default `wmir` path is
// untouched.

#include "mlir_wasmssa_to_llvm.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// wasm valtype bytes.
#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// =============================================================================
// Attribute helpers.
// =============================================================================
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_ty(MLIR_Context *ctx, const char *name,
                                    MLIR_TypeHandle ty) {
    return MLIR_CreateAttributeType(ctx, str_from_cstr_view((char *)name), ty);
}

static int64_t at_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool at_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
static string at_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}

// Map a wasm valtype to its `llvm`-dialect carrier integer type. Milestone 1
// is integer-only; f32/f64 are deferred (the caller rejects them).
static MLIR_TypeHandle vt_to_llvm(MLIR_Context *ctx, uint8_t vt) {
    if (vt == WT_I64) return MLIR_CreateTypeInteger(ctx, 64, true);
    return MLIR_CreateTypeInteger(ctx, 32, true);
}
static bool vt_is_int(uint8_t vt) { return vt == WT_I32 || vt == WT_I64; }

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}
// Decode the i-th valtype byte from an ASCII-hex type string ("7e7f" = i64,i32).
static uint8_t type_byte_at(string s, size_t i) {
    if ((2 * i + 1) >= s.size) return 0;
    return (uint8_t)((hexval(s.str[2 * i]) << 4) | hexval(s.str[2 * i + 1]));
}

// wasm-ld's default global base (data segments start here in linmem).
#define WASM_DATA_BASE 1024
// Names of the synthesised module-level globals backing the lifted linear
// memory and runtime.
#define LINMEM_GLOBAL  "__wasm_linmem"
#define FMT_NL_GLOBAL  "__wasm_fmt_nl"

// =============================================================================
// OffsetMap: import_global sym_name -> fixed linear-memory byte offset. Built
// by the module-level walk and consulted by wasmssa.addressof lowering.
// =============================================================================
typedef struct {
    string   *names;
    int32_t  *offsets;
    size_t    n, cap;
} OffsetMap;

static void omap_add(OffsetMap *m, string name, int32_t off) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->names   = (string  *)realloc(m->names,   m->cap * sizeof(string));
        m->offsets = (int32_t *)realloc(m->offsets, m->cap * sizeof(int32_t));
    }
    m->names[m->n]   = name;
    m->offsets[m->n] = off;
    m->n++;
}
static int omap_get(const OffsetMap *m, string name, int32_t *out) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->names[i].size == name.size &&
            memcmp(m->names[i].str, name.str, name.size) == 0) {
            *out = m->offsets[i]; return 1;
        }
    }
    return 0;
}

// =============================================================================
// VMap: wasmssa result value -> lifted llvm value. Open-addressing hash keyed
// on the MLIR_ValueHandle (sentinel MLIR_INVALID_HANDLE == empty). Mirrors the
// wmir lifter's map; lookups only, so iteration order never affects output.
// =============================================================================
typedef struct {
    MLIR_ValueHandle *src;
    MLIR_ValueHandle *dst;
    size_t            n, cap;
} VMap;

static size_t vmap_hash(MLIR_ValueHandle k) {
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}
static void vmap_insert_raw(MLIR_ValueHandle *src, MLIR_ValueHandle *dst,
                            size_t cap, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    size_t mask = cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (src[i] != MLIR_INVALID_HANDLE && src[i] != k) i = (i + 1) & mask;
    src[i] = k;
    dst[i] = v;
}
static void vmap_grow(VMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    MLIR_ValueHandle *nsrc = (MLIR_ValueHandle *)calloc(ncap, sizeof(*nsrc));
    MLIR_ValueHandle *ndst = (MLIR_ValueHandle *)malloc(ncap * sizeof(*ndst));
    for (size_t i = 0; i < m->cap; i++)
        if (m->src[i] != MLIR_INVALID_HANDLE)
            vmap_insert_raw(nsrc, ndst, ncap, m->src[i], m->dst[i]);
    free(m->src);
    free(m->dst);
    m->src = nsrc;
    m->dst = ndst;
    m->cap = ncap;
}
static void vmap_set(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    if ((m->n + 1) * 4 >= m->cap * 3) vmap_grow(m);
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->src[i] != MLIR_INVALID_HANDLE && m->src[i] != k) i = (i + 1) & mask;
    if (m->src[i] == MLIR_INVALID_HANDLE) m->n++;
    m->src[i] = k;
    m->dst[i] = v;
}
static int vmap_get(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    if (m->cap == 0) return 0;
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->src[i] != MLIR_INVALID_HANDLE) {
        if (m->src[i] == k) { *out = m->dst[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

// =============================================================================
// Per-function lowering state.
//
// WASM structured control flow (block / loop / if + depth-relative br) is
// flattened into an explicit cf.br / cf.cond_br CFG. Each enclosing scope
// pushes a Frame recording two target blocks:
//   - br_target: where `br {depth}` to this scope jumps (loop header for a
//     loop; the scope's continuation for a block / if).
//   - ft_target: where a normal fall-through (block_return) goes (the
//     continuation for all three; differs from br_target only for loops).
// =============================================================================
typedef struct {
    MLIR_BlockHandle br_target;
    MLIR_BlockHandle ft_target;
} Frame;

typedef struct {
    MLIR_Context     *ctx;
    MLIR_RegionHandle dreg;      // destination region (new blocks appended here)
    MLIR_BlockHandle  cur;       // block currently being appended to
    bool              terminated;// cur already has a terminator
    VMap             *vmap;
    MLIR_ValueHandle *local_ptr; // alloca ptr per local index (params + decls)
    uint8_t          *local_vt;  // valtype per local index
    size_t            n_locals;  // params + declared locals
    Frame            *frames;
    size_t            n_frames, frames_cap;
    OffsetMap        *globals;   // import_global name -> linmem offset
} FLower;

// Create a fresh op result value of the given type.
static MLIR_ValueHandle mk_res(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, ty,
        (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}));
}
static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na,
                              MLIR_TypeHandle *rtys, size_t nr,
                              MLIR_ValueHandle *res,
                              MLIR_ValueHandle *ops, size_t no) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, rtys, nr, res, nr, ops, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Build an op with an explicit name string (for unregistered ops like
// llvm.select that have no MLIR_OpType enum).
static MLIR_OpHandle build_op_named(MLIR_Context *ctx, const char *nm,
                                    MLIR_AttributeHandle *attrs, size_t na,
                                    MLIR_TypeHandle *rtys, size_t nr,
                                    MLIR_ValueHandle *res,
                                    MLIR_ValueHandle *ops, size_t no) {
    return MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_from_cstr_view((char *)nm),
        attrs, na, rtys, nr, res, nr, ops, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Append `op` to the current block.
static void emit(FLower *L, MLIR_OpHandle op) {
    MLIR_AppendBlockOp(L->ctx, L->cur, op);
}

// Create a fresh CFG block appended to the destination region.
static MLIR_BlockHandle new_block(FLower *L) {
    MLIR_BlockHandle b = MLIR_CreateBlock(L->ctx);
    MLIR_AppendRegionBlock(L->ctx, L->dreg, b);
    return b;
}

// Emit an unconditional cf.br to `target` and mark cur terminated.
static void term_br(FLower *L, MLIR_BlockHandle target) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *sops[1] = { NULL };
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(L->ctx, OP_TYPE_CF_BR,
        op_type_to_string(OP_TYPE_CF_BR),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        succs, 1, sops, 0,
        MLIR_CreateLocationUnknown(L->ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    emit(L, op);
    L->terminated = true;
}

// Emit a cf.cond_br on `cond` to (t_blk, f_blk) and mark cur terminated.
static void term_cond_br(FLower *L, MLIR_ValueHandle cond,
                         MLIR_BlockHandle t_blk, MLIR_BlockHandle f_blk) {
    MLIR_BlockHandle succs[2] = { t_blk, f_blk };
    MLIR_ValueHandle cond_arr[1] = { cond };
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(L->ctx, OP_TYPE_CF_COND_BR,
        op_type_to_string(OP_TYPE_CF_COND_BR),
        NULL, 0, NULL, 0, NULL, 0, cond_arr, 1, NULL, 0,
        succs, 2, NULL, 0,
        MLIR_CreateLocationUnknown(L->ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    emit(L, op);
    L->terminated = true;
}

static void push_frame(FLower *L, MLIR_BlockHandle br_t, MLIR_BlockHandle ft_t) {
    if (L->n_frames == L->frames_cap) {
        L->frames_cap = L->frames_cap ? L->frames_cap * 2 : 8;
        L->frames = (Frame *)realloc(L->frames, L->frames_cap * sizeof(Frame));
    }
    L->frames[L->n_frames].br_target = br_t;
    L->frames[L->n_frames].ft_target = ft_t;
    L->n_frames++;
}
static void pop_frame(FLower *L) { L->n_frames--; }

// Map a wasmssa.binop wasm opcode to an `llvm`-dialect integer op. Returns
// false for opcodes outside the milestone-1 (arithmetic / bitwise) subset.
static bool binop_to_llvm(int64_t opc, MLIR_OpType *out) {
    switch (opc) {
        // i32
        case 0x6a: *out = OP_TYPE_LLVM_ADD;  return true;
        case 0x6b: *out = OP_TYPE_LLVM_SUB;  return true;
        case 0x6c: *out = OP_TYPE_LLVM_MUL;  return true;
        case 0x6d: *out = OP_TYPE_LLVM_SDIV; return true;
        case 0x6e: *out = OP_TYPE_LLVM_UDIV; return true;
        case 0x6f: *out = OP_TYPE_LLVM_SREM; return true;
        case 0x70: *out = OP_TYPE_LLVM_UREM; return true;
        case 0x71: *out = OP_TYPE_LLVM_AND;  return true;
        case 0x72: *out = OP_TYPE_LLVM_OR;   return true;
        case 0x73: *out = OP_TYPE_LLVM_XOR;  return true;
        case 0x74: *out = OP_TYPE_LLVM_SHL;  return true;
        case 0x75: *out = OP_TYPE_LLVM_ASHR; return true;
        case 0x76: *out = OP_TYPE_LLVM_LSHR; return true;
        // i64
        case 0x7c: *out = OP_TYPE_LLVM_ADD;  return true;
        case 0x7d: *out = OP_TYPE_LLVM_SUB;  return true;
        case 0x7e: *out = OP_TYPE_LLVM_MUL;  return true;
        case 0x7f: *out = OP_TYPE_LLVM_SDIV; return true;
        case 0x80: *out = OP_TYPE_LLVM_UDIV; return true;
        case 0x81: *out = OP_TYPE_LLVM_SREM; return true;
        case 0x82: *out = OP_TYPE_LLVM_UREM; return true;
        case 0x83: *out = OP_TYPE_LLVM_AND;  return true;
        case 0x84: *out = OP_TYPE_LLVM_OR;   return true;
        case 0x85: *out = OP_TYPE_LLVM_XOR;  return true;
        case 0x86: *out = OP_TYPE_LLVM_SHL;  return true;
        case 0x87: *out = OP_TYPE_LLVM_ASHR; return true;
        case 0x88: *out = OP_TYPE_LLVM_LSHR; return true;
        default: return false;
    }
}

// Map a wasmssa comparison opcode to an `llvm`-dialect icmp predicate integer
// (eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9).
// Returns false for non-comparison opcodes.
static bool cmp_to_icmp_pred(int64_t opc, int64_t *pred) {
    switch (opc) {
        // i32 comparisons
        case 0x46: *pred = 0; return true; // eq
        case 0x47: *pred = 1; return true; // ne
        case 0x48: *pred = 2; return true; // lt_s
        case 0x49: *pred = 6; return true; // lt_u
        case 0x4a: *pred = 4; return true; // gt_s
        case 0x4b: *pred = 8; return true; // gt_u
        case 0x4c: *pred = 3; return true; // le_s
        case 0x4d: *pred = 7; return true; // le_u
        case 0x4e: *pred = 5; return true; // ge_s
        case 0x4f: *pred = 9; return true; // ge_u
        // i64 comparisons
        case 0x51: *pred = 0; return true; // eq
        case 0x52: *pred = 1; return true; // ne
        case 0x53: *pred = 2; return true; // lt_s
        case 0x54: *pred = 6; return true; // lt_u
        case 0x55: *pred = 4; return true; // gt_s
        case 0x56: *pred = 8; return true; // gt_u
        case 0x57: *pred = 3; return true; // le_s
        case 0x58: *pred = 7; return true; // le_u
        case 0x59: *pred = 5; return true; // ge_s
        case 0x5a: *pred = 9; return true; // ge_u
        default: return false;
    }
}

// Build an llvm.icmp with an i32 boolean result (wasm comparison convention).
static MLIR_ValueHandle build_icmp(FLower *L, int64_t pred,
                                   MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_Context *ctx = L->ctx;
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_AttributeHandle pa = MLIR_CreateAttributeInteger(
        ctx, str_from_cstr_view((char *)"predicate"), pred, i64);
    MLIR_TypeHandle rt[1] = { i32 };
    MLIR_ValueHandle r[1] = { mk_res(ctx, i32) };
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_ICMP, &pa, 1, rt, 1, r, ops, 2);
    emit(L, out);
    return r[0];
}

static bool lower_op(FLower *L, MLIR_OpHandle op);

// =============================================================================
// Linear-memory lowering helpers.
//
// WASM linear memory is modelled as one big module-level native global
// (@__wasm_linmem). A wasm address is an i32 byte offset into that global;
// the host pointer for a load/store is materialised as
//   addressof(@__wasm_linmem) + zext(addr) + static_offset
// using ptrtoint / add / inttoptr (all supported by the unified backend).
// wasm globals (the shadow stack pointer etc.) become small mutable scalar
// globals @__wasm_g0..@__wasm_gN.
// =============================================================================

// Emit an integer constant of type `ity`; returns its value.
static MLIR_ValueHandle emit_const(FLower *L, MLIR_TypeHandle ity, int64_t v) {
    MLIR_AttributeHandle va = MLIR_CreateAttributeInteger(
        L->ctx, str_from_cstr_view((char *)"value"), v, ity);
    MLIR_TypeHandle rt[1] = { ity };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ity) };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
        &va, 1, rt, 1, r, NULL, 0);
    emit(L, out);
    return r[0];
}

// Emit `llvm.mlir.addressof @sym` -> !llvm.ptr.
static MLIR_ValueHandle emit_addressof(FLower *L, const char *sym) {
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(L->ctx);
    MLIR_AttributeHandle a = MLIR_CreateAttributeSymbolRef(
        L->ctx, str_from_cstr_view((char *)"global_name"),
        str_from_cstr_view((char *)sym));
    MLIR_TypeHandle rt[1] = { ptr };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ptr) };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_MLIR_ADDRESSOF,
        &a, 1, rt, 1, r, NULL, 0);
    emit(L, out);
    return r[0];
}

// Emit a single-operand cast op (zext / trunc) producing `toty`.
static MLIR_ValueHandle emit_cast(FLower *L, MLIR_OpType ot,
                                  MLIR_ValueHandle v, MLIR_TypeHandle toty) {
    MLIR_TypeHandle rt[1] = { toty };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, toty) };
    MLIR_ValueHandle ops[1] = { v };
    MLIR_OpHandle out = build_op(L->ctx, ot, NULL, 0, rt, 1, r, ops, 1);
    emit(L, out);
    return r[0];
}

// Emit a two-operand integer op (add / and) producing `ity`.
static MLIR_ValueHandle emit_binop2(FLower *L, MLIR_OpType ot,
                                    MLIR_ValueHandle a, MLIR_ValueHandle b,
                                    MLIR_TypeHandle ity) {
    MLIR_TypeHandle rt[1] = { ity };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ity) };
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_OpHandle out = build_op(L->ctx, ot, NULL, 0, rt, 1, r, ops, 2);
    emit(L, out);
    return r[0];
}

// Materialise the host pointer for wasm address `addr_i32` + static `off`:
//   p = inttoptr( ptrtoint(addressof(@__wasm_linmem)) + zext(addr) + off )
static MLIR_ValueHandle linmem_ptr(FLower *L, MLIR_ValueHandle addr_i32,
                                   int64_t off) {
    MLIR_Context *ctx = L->ctx;
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(ctx);
    MLIR_ValueHandle base_ptr = emit_addressof(L, LINMEM_GLOBAL);
    MLIR_ValueHandle base_i64 = emit_cast(L, OP_TYPE_LLVM_PTRTOINT, base_ptr, i64);
    MLIR_ValueHandle addr64 = emit_cast(L, OP_TYPE_LLVM_ZEXT, addr_i32, i64);
    MLIR_ValueHandle ea = emit_binop2(L, OP_TYPE_LLVM_ADD, base_i64, addr64, i64);
    if (off != 0) {
        MLIR_ValueHandle offc = emit_const(L, i64, off);
        ea = emit_binop2(L, OP_TYPE_LLVM_ADD, ea, offc, i64);
    }
    // inttoptr has no MLIR_OpType enum; build by name (unregistered).
    MLIR_TypeHandle rt[1] = { ptr };
    MLIR_ValueHandle r[1] = { mk_res(ctx, ptr) };
    MLIR_ValueHandle ops[1] = { ea };
    MLIR_OpHandle out = build_op_named(ctx, "llvm.inttoptr", NULL, 0, rt, 1, r, ops, 1);
    emit(L, out);
    return r[0];
}

static bool lower_region(FLower *L, MLIR_BlockHandle src_blk) {
    size_t n = MLIR_GetBlockNumOps(src_blk);
    for (size_t i = 0; i < n; i++) {
        if (L->terminated) break;
        MLIR_OpHandle o = MLIR_GetBlockOp(src_blk, i);
        if (!lower_op(L, o)) return false;
    }
    return true;
}

static bool lower_op(FLower *L, MLIR_OpHandle op) {
    MLIR_Context *ctx = L->ctx;
    MLIR_OpType t = MLIR_GetOpType(op);

    switch (t) {
    case OP_TYPE_WASMSSA_CONST: {
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: float const not yet supported\n");
            return false;
        }
        int64_t v = at_i(op, "value");
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        MLIR_AttributeHandle va = MLIR_CreateAttributeInteger(
            ctx, str_from_cstr_view((char *)"value"), v, ity);
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
            &va, 1, rt, 1, r, NULL, 0);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_LOCAL_GET: {
        int64_t idx = at_i(op, "local_idx");
        if (idx < 0 || (size_t)idx >= L->n_locals) {
            fprintf(stderr, "wasmssa->llvm: local_get idx %lld out of range\n",
                    (long long)idx);
            return false;
        }
        uint8_t vt = L->local_vt[idx];
        MLIR_TypeHandle ety = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ety };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ety) };
        MLIR_ValueHandle ops[1] = { L->local_ptr[idx] };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_LOAD,
            NULL, 0, rt, 1, r, ops, 1);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_LOCAL_SET: {
        int64_t idx = at_i(op, "local_idx");
        if (idx < 0 || (size_t)idx >= L->n_locals) {
            fprintf(stderr, "wasmssa->llvm: local_set idx %lld out of range\n",
                    (long long)idx);
            return false;
        }
        MLIR_ValueHandle v;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &v)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on local_set\n");
            return false;
        }
        MLIR_ValueHandle ops[2] = { v, L->local_ptr[idx] };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_STORE,
            NULL, 0, NULL, 0, NULL, ops, 2);
        emit(L, out);
        return true;
    }

    case OP_TYPE_WASMSSA_ADD:
    case OP_TYPE_WASMSSA_SUB: {
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: float add/sub not yet supported\n");
            return false;
        }
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on add/sub\n");
            return false;
        }
        MLIR_OpType ot = (t == OP_TYPE_WASMSSA_ADD) ? OP_TYPE_LLVM_ADD
                                                    : OP_TYPE_LLVM_SUB;
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_ValueHandle ops[2] = { a, b };
        MLIR_OpHandle out = build_op(ctx, ot, NULL, 0, rt, 1, r, ops, 2);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_BINOP: {
        int64_t opc = at_i(op, "wasm_opcode");
        int64_t pred;
        if (cmp_to_icmp_pred(opc, &pred)) {
            MLIR_ValueHandle a, b;
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
                !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on cmp\n");
                return false;
            }
            MLIR_ValueHandle r = build_icmp(L, pred, a, b);
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r);
            return true;
        }
        MLIR_OpType ot;
        if (!binop_to_llvm(opc, &ot)) {
            fprintf(stderr,
                "wasmssa->llvm: binop opcode 0x%llx not yet supported\n",
                (unsigned long long)opc);
            return false;
        }
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on binop\n");
            return false;
        }
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_ValueHandle ops[2] = { a, b };
        MLIR_OpHandle out = build_op(ctx, ot, NULL, 0, rt, 1, r, ops, 2);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_EXTEND_I32_S: {
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on extend_i32_s\n");
            return false;
        }
        MLIR_TypeHandle i64ty = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_TypeHandle rt[1] = { i64ty };
        MLIR_ValueHandle r[1] = { mk_res(ctx, i64ty) };
        MLIR_ValueHandle ops[1] = { a };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_SEXT,
            NULL, 0, rt, 1, r, ops, 1);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_CALL: {
        string callee = at_s(op, "target");
        if (callee.size == 0) {
            fprintf(stderr, "wasmssa->llvm: call without target\n");
            return false;
        }
        // WASI imports map to native libSystem stubs.
        if (callee.size == 9 && memcmp(callee.str, "proc_exit", 9) == 0)
            callee = str_lit("_exit");
        size_t no = MLIR_GetOpNumOperands(op);
        MLIR_ValueHandle *ops = (MLIR_ValueHandle *)malloc(
            (no ? no : 1) * sizeof(MLIR_ValueHandle));
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, k), &ops[k])) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on call\n");
                free(ops);
                return false;
            }
        }
        size_t nr = MLIR_GetOpNumResults(op);
        if (nr > 1) {
            fprintf(stderr, "wasmssa->llvm: multi-result call unsupported\n");
            free(ops);
            return false;
        }
        MLIR_AttributeHandle attrs[1] = {
            attr_s(ctx, "callee", callee.str, callee.size)
        };
        MLIR_TypeHandle rt[1];
        MLIR_ValueHandle r[1];
        if (nr == 1) {
            MLIR_TypeHandle ty = MLIR_GetValueType(MLIR_GetOpResult(op, 0));
            rt[0] = ty;
            r[0] = mk_res(ctx, ty);
        }
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_CALL,
            attrs, 1, rt, nr, r, ops, no);
        emit(L, out);
        if (nr == 1) vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        free(ops);
        return true;
    }

    case OP_TYPE_WASMSSA_RETURN: {
        size_t no = MLIR_GetOpNumOperands(op);
        if (no > 1) {
            fprintf(stderr, "wasmssa->llvm: multi-value return unsupported\n");
            return false;
        }
        MLIR_ValueHandle ops[1];
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, k), &ops[k])) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on return\n");
                return false;
            }
        }
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_RETURN,
            NULL, 0, NULL, 0, NULL, ops, no);
        emit(L, out);
        L->terminated = true;
        return true;
    }

    case OP_TYPE_WASMSSA_EQZ: {
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on eqz\n");
            return false;
        }
        // result = icmp eq (a, 0) : i32
        MLIR_TypeHandle aty = MLIR_GetValueType(a);
        MLIR_AttributeHandle za = MLIR_CreateAttributeInteger(
            ctx, str_from_cstr_view((char *)"value"), 0, aty);
        MLIR_TypeHandle zrt[1] = { aty };
        MLIR_ValueHandle zr[1] = { mk_res(ctx, aty) };
        MLIR_OpHandle zc = build_op(ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
            &za, 1, zrt, 1, zr, NULL, 0);
        emit(L, zc);
        MLIR_ValueHandle r = build_icmp(L, /*eq*/0, a, zr[0]);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r);
        return true;
    }

    case OP_TYPE_WASMSSA_SELECT: {
        // wasmssa.select(%a, %b, %cond) -> R   (cond is the LAST operand);
        // llvm.select wants (cond, tval, fval).
        MLIR_ValueHandle a, b, c;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 2), &c)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on select\n");
            return false;
        }
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        (void)vt;
        MLIR_TypeHandle ity = MLIR_GetValueType(MLIR_GetOpResult(op, 0));
        MLIR_TypeHandle rt[1] = { ity };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ity) };
        MLIR_ValueHandle ops[3] = { c, a, b };
        MLIR_OpHandle out = build_op_named(ctx, "llvm.select",
            NULL, 0, rt, 1, r, ops, 3);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_UNREACHABLE: {
        // No native trap op wired; a self-branch is a valid (dead) terminator.
        term_br(L, L->cur);
        return true;
    }

    case OP_TYPE_WASMSSA_BR: {
        if (MLIR_GetOpNumOperands(op) != 0) {
            fprintf(stderr,
                "wasmssa->llvm: value-carrying br not yet supported\n");
            return false;
        }
        int64_t depth = at_i(op, "depth");
        if (depth < 0 || (size_t)depth >= L->n_frames) {
            fprintf(stderr, "wasmssa->llvm: br depth %lld out of range\n",
                    (long long)depth);
            return false;
        }
        term_br(L, L->frames[L->n_frames - 1 - (size_t)depth].br_target);
        return true;
    }

    case OP_TYPE_WASMSSA_BR_IF: {
        int64_t depth = at_i(op, "depth");
        if (depth < 0 || (size_t)depth >= L->n_frames) {
            fprintf(stderr, "wasmssa->llvm: br_if depth %lld out of range\n",
                    (long long)depth);
            return false;
        }
        MLIR_ValueHandle cond;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &cond)) {
            fprintf(stderr, "wasmssa->llvm: unbound condition on br_if\n");
            return false;
        }
        MLIR_BlockHandle target = L->frames[L->n_frames - 1 - (size_t)depth].br_target;
        MLIR_BlockHandle fall = new_block(L);
        term_cond_br(L, cond, target, fall);
        L->cur = fall;
        L->terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_BLOCK_RETURN: {
        if (MLIR_GetOpNumOperands(op) != 0) {
            fprintf(stderr,
                "wasmssa->llvm: value-carrying block_return not yet supported\n");
            return false;
        }
        if (L->n_frames == 0) {
            fprintf(stderr, "wasmssa->llvm: block_return with no frame\n");
            return false;
        }
        term_br(L, L->frames[L->n_frames - 1].ft_target);
        return true;
    }

    case OP_TYPE_WASMSSA_BLOCK: {
        if (MLIR_GetOpNumResults(op) != 0) {
            fprintf(stderr,
                "wasmssa->llvm: value-carrying block not yet supported\n");
            return false;
        }
        if (MLIR_GetOpNumRegions(op) < 1) {
            fprintf(stderr, "wasmssa->llvm: block has no region\n");
            return false;
        }
        MLIR_BlockHandle merge = new_block(L);
        push_frame(L, merge, merge);
        L->terminated = false;
        if (!lower_region(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0))) {
            pop_frame(L);
            return false;
        }
        if (!L->terminated) term_br(L, merge);
        pop_frame(L);
        L->cur = merge;
        L->terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_IF: {
        if (MLIR_GetOpNumResults(op) != 0) {
            fprintf(stderr,
                "wasmssa->llvm: value-carrying if not yet supported\n");
            return false;
        }
        MLIR_ValueHandle cond;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &cond)) {
            fprintf(stderr, "wasmssa->llvm: unbound condition on if\n");
            return false;
        }
        size_t n_regions = MLIR_GetOpNumRegions(op);
        if (n_regions < 1) {
            fprintf(stderr, "wasmssa->llvm: if has no then region\n");
            return false;
        }
        bool has_else = (n_regions >= 2)
            && (MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 1)) > 0);

        MLIR_BlockHandle then_blk = new_block(L);
        MLIR_BlockHandle else_blk = has_else ? new_block(L) : MLIR_INVALID_HANDLE;
        MLIR_BlockHandle merge    = new_block(L);

        term_cond_br(L, cond, then_blk, has_else ? else_blk : merge);

        L->cur = then_blk;
        L->terminated = false;
        push_frame(L, merge, merge);
        if (!lower_region(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0))) {
            pop_frame(L);
            return false;
        }
        if (!L->terminated) term_br(L, merge);
        pop_frame(L);

        if (has_else) {
            L->cur = else_blk;
            L->terminated = false;
            push_frame(L, merge, merge);
            if (!lower_region(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0))) {
                pop_frame(L);
                return false;
            }
            if (!L->terminated) term_br(L, merge);
            pop_frame(L);
        }

        L->cur = merge;
        L->terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_LOOP: {
        if (MLIR_GetOpNumResults(op) != 0 || MLIR_GetOpNumOperands(op) != 0) {
            fprintf(stderr,
                "wasmssa->llvm: value-carrying loop not yet supported\n");
            return false;
        }
        if (MLIR_GetOpNumRegions(op) < 1) {
            fprintf(stderr, "wasmssa->llvm: loop has no region\n");
            return false;
        }
        MLIR_RegionHandle loop_region = MLIR_GetOpRegion(op, 0);
        if (MLIR_GetRegionNumBlocks(loop_region) < 1) {
            fprintf(stderr, "wasmssa->llvm: loop region empty\n");
            return false;
        }
        MLIR_BlockHandle src_loop_blk = MLIR_GetRegionBlock(loop_region, 0);
        if (MLIR_GetBlockNumArgs(src_loop_blk) != 0) {
            fprintf(stderr,
                "wasmssa->llvm: loop with block args not yet supported\n");
            return false;
        }
        MLIR_BlockHandle header = new_block(L);
        term_br(L, header);
        MLIR_BlockHandle post = new_block(L);
        L->cur = header;
        L->terminated = false;
        push_frame(L, header, post);
        if (!lower_region(L, src_loop_blk)) {
            pop_frame(L);
            return false;
        }
        if (!L->terminated) term_br(L, post);
        pop_frame(L);
        L->cur = post;
        L->terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_ADDRESSOF: {
        string tgt = at_s(op, "target");
        int32_t off;
        if (!L->globals || !omap_get(L->globals, tgt, &off)) {
            fprintf(stderr,
                "wasmssa->llvm: addressof unknown global '%.*s'\n",
                (int)tgt.size, tgt.str);
            return false;
        }
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_ValueHandle r = emit_const(L, i32, off);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r);
        return true;
    }

    case OP_TYPE_WASMSSA_GLOBAL_GET: {
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        int64_t idx = at_i(op, "global_idx");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: non-int global_get not supported\n");
            return false;
        }
        char gname[32];
        snprintf(gname, sizeof(gname), "__wasm_g%lld", (long long)idx);
        MLIR_ValueHandle p = emit_addressof(L, gname);
        MLIR_TypeHandle ety = vt_to_llvm(ctx, vt);
        MLIR_TypeHandle rt[1] = { ety };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ety) };
        MLIR_ValueHandle ops[1] = { p };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_LOAD, NULL, 0, rt, 1, r, ops, 1);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_GLOBAL_SET: {
        int64_t idx = at_i(op, "global_idx");
        MLIR_ValueHandle v;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &v)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on global_set\n");
            return false;
        }
        char gname[32];
        snprintf(gname, sizeof(gname), "__wasm_g%lld", (long long)idx);
        MLIR_ValueHandle p = emit_addressof(L, gname);
        MLIR_ValueHandle ops[2] = { v, p };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_STORE, NULL, 0, NULL, 0, NULL, ops, 2);
        emit(L, out);
        return true;
    }

    case OP_TYPE_WASMSSA_LOAD: {
        int64_t off = at_i(op, "memory_offset");
        int64_t sz  = at_i(op, "mem_size_bytes");
        uint8_t vt  = (uint8_t)at_i(op, "valtype");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: float load not yet supported\n");
            return false;
        }
        bool ok = (vt == WT_I32 && (sz == 1 || sz == 2 || sz == 4)) ||
                  (vt == WT_I64 && (sz == 1 || sz == 2 || sz == 4 || sz == 8));
        if (!ok) {
            fprintf(stderr,
                "wasmssa->llvm: load mem_size=%lld valtype=%u not supported\n",
                (long long)sz, (unsigned)vt);
            return false;
        }
        MLIR_ValueHandle addr;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &addr)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on load\n");
            return false;
        }
        MLIR_ValueHandle p = linmem_ptr(L, addr, off);
        bool is64 = (vt == WT_I64);
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
        // Choose a backend-supported load width (1/4/8 bytes); sub-word loads
        // are zero-extended into the i32/i64 result.
        MLIR_TypeHandle lty = (sz == 8) ? i64 : (sz == 1) ? MLIR_CreateTypeInteger(ctx, 8, true) : i32;
        MLIR_TypeHandle lrt[1] = { lty };
        MLIR_ValueHandle lr[1] = { mk_res(ctx, lty) };
        MLIR_ValueHandle lops[1] = { p };
        MLIR_OpHandle lop = build_op(ctx, OP_TYPE_LLVM_LOAD, NULL, 0, lrt, 1, lr, lops, 1);
        emit(L, lop);
        MLIR_ValueHandle v = lr[0];
        if (sz == 2) {
            // 4-byte load then mask to the low 16 bits.
            MLIR_ValueHandle m = emit_const(L, i32, 0xffff);
            v = emit_binop2(L, OP_TYPE_LLVM_AND, v, m, i32);
        }
        // Widen to the result type.
        MLIR_TypeHandle want = is64 ? i64 : i32;
        if (sz < 8 && is64) v = emit_cast(L, OP_TYPE_LLVM_ZEXT, v, i64);
        else if (sz == 1 && !is64) v = emit_cast(L, OP_TYPE_LLVM_ZEXT, v, i32);
        (void)want;
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), v);
        return true;
    }

    case OP_TYPE_WASMSSA_STORE: {
        int64_t off = at_i(op, "memory_offset");
        int64_t sz  = at_i(op, "mem_size_bytes");
        uint8_t vt  = (uint8_t)at_i(op, "valtype");
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: float store not yet supported\n");
            return false;
        }
        bool ok = (vt == WT_I32 && (sz == 1 || sz == 2 || sz == 4)) ||
                  (vt == WT_I64 && (sz == 1 || sz == 2 || sz == 4 || sz == 8));
        if (!ok) {
            fprintf(stderr,
                "wasmssa->llvm: store mem_size=%lld valtype=%u not supported\n",
                (long long)sz, (unsigned)vt);
            return false;
        }
        MLIR_ValueHandle addr, val;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &addr) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &val)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on store\n");
            return false;
        }
        MLIR_ValueHandle p = linmem_ptr(L, addr, off);
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        // Truncate the value to a backend-supported store width (1/4/8). A
        // 2-byte store uses a 4-byte store (matches the wmir backend).
        if (sz == 1) {
            MLIR_TypeHandle i8 = MLIR_CreateTypeInteger(ctx, 8, true);
            val = emit_cast(L, OP_TYPE_LLVM_TRUNC, val, i8);
        } else if (sz == 2 || sz == 4) {
            if (vt == WT_I64) val = emit_cast(L, OP_TYPE_LLVM_TRUNC, val, i32);
        }
        MLIR_ValueHandle ops[2] = { val, p };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_STORE, NULL, 0, NULL, 0, NULL, ops, 2);
        emit(L, out);
        return true;
    }

    case OP_TYPE_WASMSSA_MEMORY_SIZE: {
        MLIR_ValueHandle p = emit_addressof(L, "__wasm_mem_pages");
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle rt[1] = { i32 };
        MLIR_ValueHandle r[1] = { mk_res(ctx, i32) };
        MLIR_ValueHandle ops[1] = { p };
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_LOAD, NULL, 0, rt, 1, r, ops, 1);
        emit(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_MEMORY_GROW: {
        // No real growth: bump the page-count global and return the old value.
        MLIR_ValueHandle delta;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &delta)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on memory_grow\n");
            return false;
        }
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_ValueHandle p = emit_addressof(L, "__wasm_mem_pages");
        MLIR_TypeHandle lrt[1] = { i32 };
        MLIR_ValueHandle lr[1] = { mk_res(ctx, i32) };
        MLIR_ValueHandle lops[1] = { p };
        MLIR_OpHandle lop = build_op(ctx, OP_TYPE_LLVM_LOAD, NULL, 0, lrt, 1, lr, lops, 1);
        emit(L, lop);
        MLIR_ValueHandle nw = emit_binop2(L, OP_TYPE_LLVM_ADD, lr[0], delta, i32);
        MLIR_ValueHandle p2 = emit_addressof(L, "__wasm_mem_pages");
        MLIR_ValueHandle sops[2] = { nw, p2 };
        MLIR_OpHandle sop = build_op(ctx, OP_TYPE_LLVM_STORE, NULL, 0, NULL, 0, NULL, sops, 2);
        emit(L, sop);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), lr[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_UNOP: {
        int64_t opc = at_i(op, "wasm_opcode");
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on unop\n");
            return false;
        }
        MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
        MLIR_TypeHandle i16 = MLIR_CreateTypeInteger(ctx, 16, true);
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_ValueHandle res;
        switch (opc) {
        case 0xa7: // i32.wrap_i64
            res = emit_cast(L, OP_TYPE_LLVM_TRUNC, a, i32); break;
        case 0xac: // i64.extend_i32_s
            res = emit_cast(L, OP_TYPE_LLVM_SEXT, a, i64); break;
        case 0xad: // i64.extend_i32_u
            res = emit_cast(L, OP_TYPE_LLVM_ZEXT, a, i64); break;
        case 0xc0: // i32.extend8_s
            res = emit_cast(L, OP_TYPE_LLVM_SEXT,
                emit_cast(L, OP_TYPE_LLVM_TRUNC, a, i8), i32); break;
        case 0xc1: // i32.extend16_s
            res = emit_cast(L, OP_TYPE_LLVM_SEXT,
                emit_cast(L, OP_TYPE_LLVM_TRUNC, a, i16), i32); break;
        case 0xc2: // i64.extend8_s
            res = emit_cast(L, OP_TYPE_LLVM_SEXT,
                emit_cast(L, OP_TYPE_LLVM_TRUNC, a, i8), i64); break;
        case 0xc3: // i64.extend16_s
            res = emit_cast(L, OP_TYPE_LLVM_SEXT,
                emit_cast(L, OP_TYPE_LLVM_TRUNC, a, i16), i64); break;
        case 0xc4: // i64.extend32_s
            res = emit_cast(L, OP_TYPE_LLVM_SEXT,
                emit_cast(L, OP_TYPE_LLVM_TRUNC, a, i32), i64); break;
        default:
            fprintf(stderr,
                "wasmssa->llvm: unop opcode 0x%llx not yet supported\n",
                (long long)opc);
            return false;
        }
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), res);
        return true;
    }

    default:
    }
}

// Lift one wasmssa.func into an `llvm.func`. Returns MLIR_INVALID_HANDLE on
// failure (the caller aborts the whole module).
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src,
                                OffsetMap *globals) {
    string name     = at_s(src, "sym_name");
    bool   exported = at_b(src, "exported");
    string pt       = at_s(src, "param_types");
    string lt       = at_s(src, "local_types");

    // The backend synthesises `_start` -> `bl main`, so the exported wasm
    // entry (wasi_start) is renamed to `main`.
    if (exported) name = str_lit("main");

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wasmssa->llvm: func '%.*s' has no region\n",
                (int)name.size, name.str);
        return MLIR_INVALID_HANDLE;
    }
    MLIR_RegionHandle sreg = MLIR_GetOpRegion(src, 0);
    if (MLIR_GetRegionNumBlocks(sreg) != 1) {
        fprintf(stderr,
            "wasmssa->llvm: func '%.*s' unexpected multi-block source region\n",
            (int)name.size, name.str);
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle sblk = MLIR_GetRegionBlock(sreg, 0);

    size_t n_params = pt.size / 2;
    size_t n_decls  = lt.size / 2;
    size_t n_locals = n_params + n_decls;

    // Decode and validate all local valtypes up front (integer-only here).
    uint8_t *local_vt = (uint8_t *)malloc((n_locals ? n_locals : 1));
    for (size_t i = 0; i < n_params; i++) local_vt[i] = type_byte_at(pt, i);
    for (size_t i = 0; i < n_decls; i++)  local_vt[n_params + i] = type_byte_at(lt, i);
    for (size_t i = 0; i < n_locals; i++) {
        if (!vt_is_int(local_vt[i])) {
            fprintf(stderr,
                "wasmssa->llvm: func '%.*s' has a non-integer local "
                "(floats not yet supported)\n",
                (int)name.size, name.str);
            free(local_vt);
            return MLIR_INVALID_HANDLE;
        }
    }

    MLIR_RegionHandle dreg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, dreg, entry);

    VMap vmap = {0};
    FLower L = {0};
    L.ctx = ctx;
    L.dreg = dreg;
    L.cur = entry;
    L.terminated = false;
    L.vmap = &vmap;
    L.n_locals = n_locals;
    L.local_vt = local_vt;
    L.globals = globals;
    L.local_ptr = (MLIR_ValueHandle *)malloc(
        (n_locals ? n_locals : 1) * sizeof(MLIR_ValueHandle));

    // Function parameters become block args. Map each SOURCE block arg (the
    // wasmssa param value) to its dest counterpart so that ops referencing a
    // param value directly (not via local_get) resolve.
    size_t n_src_args = MLIR_GetBlockNumArgs(sblk);
    MLIR_ValueHandle *param_args = (MLIR_ValueHandle *)malloc(
        (n_params ? n_params : 1) * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < n_params; i++) {
        MLIR_TypeHandle ty = vt_to_llvm(ctx, local_vt[i]);
        MLIR_ValueHandle da = MLIR_CreateValueBlockArg(ctx, (string){0},
            (uint32_t)i, ty, MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, entry, da);
        param_args[i] = da;
        if (i < n_src_args)
            vmap_set(&vmap, MLIR_GetBlockArg(sblk, i), da);
    }

    // One i64 const=1 to serve as the element count for every alloca.
    MLIR_TypeHandle i64ty = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_AttributeHandle cnt_va = MLIR_CreateAttributeInteger(
        ctx, str_from_cstr_view((char *)"value"), 1, i64ty);
    MLIR_TypeHandle cnt_rt[1] = { i64ty };
    MLIR_ValueHandle cnt_r[1] = { mk_res(ctx, i64ty) };
    MLIR_OpHandle cnt_op = build_op(ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
        &cnt_va, 1, cnt_rt, 1, cnt_r, NULL, 0);
    MLIR_AppendBlockOp(ctx, entry, cnt_op);

    // One alloca per local index (params + declared locals).
    MLIR_TypeHandle ptr_ty = MLIR_CreateTypeLLVMPointer(ctx);
    for (size_t i = 0; i < n_locals; i++) {
        MLIR_TypeHandle ety = vt_to_llvm(ctx, local_vt[i]);
        MLIR_AttributeHandle a[1] = { attr_ty(ctx, "elem_type", ety) };
        MLIR_TypeHandle rt[1] = { ptr_ty };
        MLIR_ValueHandle r[1] = { mk_res(ctx, ptr_ty) };
        MLIR_ValueHandle ops[1] = { cnt_r[0] };
        MLIR_OpHandle aop = build_op(ctx, OP_TYPE_LLVM_ALLOCA,
            a, 1, rt, 1, r, ops, 1);
        MLIR_AppendBlockOp(ctx, entry, aop);
        L.local_ptr[i] = r[0];
    }

    // Store incoming params into their local allocas.
    for (size_t i = 0; i < n_params; i++) {
        MLIR_ValueHandle ops[2] = { param_args[i], L.local_ptr[i] };
        MLIR_OpHandle st = build_op(ctx, OP_TYPE_LLVM_STORE,
            NULL, 0, NULL, 0, NULL, ops, 2);
        MLIR_AppendBlockOp(ctx, entry, st);
    }

    // Lower the body. Control-flow ops append further blocks to dreg and
    // advance L.cur; lower_region skips dead ops after a terminator.
    bool ok = lower_region(&L, sblk);

    free(vmap.src);
    free(vmap.dst);
    free(local_vt);
    free(L.local_ptr);
    free(param_args);
    free(L.frames);
    if (!ok) return MLIR_INVALID_HANDLE;

    MLIR_AttributeHandle attrs[1] = {
        attr_s(ctx, "sym_name", name.str, name.size)
    };
    MLIR_RegionHandle regs[1] = { dreg };
    return MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Emit `llvm.alloca count x elem_ty` -> !llvm.ptr.
static MLIR_ValueHandle emit_alloca(FLower *L, MLIR_TypeHandle elem_ty, int64_t count) {
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(L->ctx, 64, true);
    MLIR_ValueHandle cnt = emit_const(L, i64, count);
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(L->ctx);
    MLIR_AttributeHandle a[1] = { attr_ty(L->ctx, "elem_type", elem_ty) };
    MLIR_TypeHandle rt[1] = { ptr };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ptr) };
    MLIR_ValueHandle ops[1] = { cnt };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_ALLOCA, a, 1, rt, 1, r, ops, 1);
    emit(L, out);
    return r[0];
}

// Compute `base_ptr + idx_i64` as a host pointer (byte offset).
static MLIR_ValueHandle emit_byte_ptr(FLower *L, MLIR_ValueHandle base, MLIR_ValueHandle idx) {
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(L->ctx, 64, true);
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(L->ctx);
    MLIR_ValueHandle bi = emit_cast(L, OP_TYPE_LLVM_PTRTOINT, base, i64);
    MLIR_ValueHandle ea = emit_binop2(L, OP_TYPE_LLVM_ADD, bi, idx, i64);
    MLIR_TypeHandle rt[1] = { ptr };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ptr) };
    MLIR_ValueHandle ops[1] = { ea };
    MLIR_OpHandle out = build_op_named(L->ctx, "llvm.inttoptr", NULL, 0, rt, 1, r, ops, 1);
    emit(L, out);
    return r[0];
}

static MLIR_ValueHandle emit_load_ty(FLower *L, MLIR_ValueHandle p, MLIR_TypeHandle ty) {
    MLIR_TypeHandle rt[1] = { ty };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ty) };
    MLIR_ValueHandle ops[1] = { p };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_LOAD, NULL, 0, rt, 1, r, ops, 1);
    emit(L, out);
    return r[0];
}
static void emit_store_v(FLower *L, MLIR_ValueHandle v, MLIR_ValueHandle p) {
    MLIR_ValueHandle ops[2] = { v, p };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_STORE, NULL, 0, NULL, 0, NULL, ops, 2);
    emit(L, out);
}

// Emit `_write(fd, ptr, len)` (result discarded).
static void emit_write_call(FLower *L, MLIR_ValueHandle fd, MLIR_ValueHandle p,
                            MLIR_ValueHandle len) {
    MLIR_AttributeHandle a[1] = {
        attr_s(L->ctx, "callee", (char *)"_write", 6)
    };
    MLIR_ValueHandle ops[3] = { fd, p, len };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_CALL, a, 1, NULL, 0, NULL, ops, 3);
    emit(L, out);
}

// void printNewline(): _write(1, "\n", 1).
static void synth_print_newline(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  blk = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, blk);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    FLower L = {0}; L.ctx = ctx; L.cur = blk;
    MLIR_ValueHandle p = emit_addressof(&L, FMT_NL_GLOBAL);
    emit_write_call(&L, emit_const(&L, i32, 1), p, emit_const(&L, i64, 1));
    MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, NULL, 0);
    emit(&L, ret);
    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)"printNewline", 12) };
    MLIR_RegionHandle regs[1] = { reg };
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}), MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, out_body, fn);
}

// void printI64(i64 v): format v as decimal into a stack buffer and _write it.
static void synth_print_i64(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);

    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);

    MLIR_ValueHandle v = MLIR_CreateValueBlockArg(ctx, (string){0}, 0, i64,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockArg(ctx, entry, v);

    FLower L = {0};
    L.ctx = ctx; L.dreg = reg; L.cur = entry;

    // Stack buffer (32 bytes) + cursor `pos` and working value `u` allocas.
    MLIR_ValueHandle buf = emit_alloca(&L, i8, 32);
    MLIR_ValueHandle ppr = emit_alloca(&L, i64, 1);
    MLIR_ValueHandle upr = emit_alloca(&L, i64, 1);
    MLIR_ValueHandle npr = emit_alloca(&L, i64, 1);
    MLIR_ValueHandle c0  = emit_const(&L, i64, 0);
    MLIR_ValueHandle c32 = emit_const(&L, i64, 32);
    emit_store_v(&L, c32, ppr);

    MLIR_BlockHandle negb = new_block(&L);
    MLIR_BlockHandle posb = new_block(&L);
    MLIR_BlockHandle head = new_block(&L);
    MLIR_BlockHandle signb = new_block(&L);
    MLIR_BlockHandle minusb = new_block(&L);
    MLIR_BlockHandle wblk = new_block(&L);

    MLIR_ValueHandle isneg = build_icmp(&L, /*slt*/2, v, c0);
    term_cond_br(&L, isneg, negb, posb);

    // negb: u = 0 - v; neg = 1
    L.cur = negb; L.terminated = false;
    MLIR_ValueHandle nv = emit_binop2(&L, OP_TYPE_LLVM_SUB, c0, v, i64);
    emit_store_v(&L, nv, upr);
    emit_store_v(&L, emit_const(&L, i64, 1), npr);
    term_br(&L, head);

    // posb: u = v; neg = 0
    L.cur = posb; L.terminated = false;
    emit_store_v(&L, v, upr);
    emit_store_v(&L, emit_const(&L, i64, 0), npr);
    term_br(&L, head);

    // head: do { digit = u%10; buf[--pos] = '0'+digit; u /= 10; } while (u != 0)
    L.cur = head; L.terminated = false;
    MLIR_ValueHandle u = emit_load_ty(&L, upr, i64);
    MLIR_ValueHandle ten = emit_const(&L, i64, 10);
    MLIR_ValueHandle rem = emit_binop2(&L, OP_TYPE_LLVM_SREM, u, ten, i64);
    MLIR_ValueHandle q   = emit_binop2(&L, OP_TYPE_LLVM_SDIV, u, ten, i64);
    MLIR_ValueHandle ch  = emit_binop2(&L, OP_TYPE_LLVM_ADD, rem, emit_const(&L, i64, 48), i64);
    MLIR_ValueHandle ch8 = emit_cast(&L, OP_TYPE_LLVM_TRUNC, ch, i8);
    MLIR_ValueHandle pos = emit_load_ty(&L, ppr, i64);
    MLIR_ValueHandle pos1 = emit_binop2(&L, OP_TYPE_LLVM_SUB, pos, emit_const(&L, i64, 1), i64);
    emit_store_v(&L, pos1, ppr);
    MLIR_ValueHandle bp = emit_byte_ptr(&L, buf, pos1);
    emit_store_v(&L, ch8, bp);
    emit_store_v(&L, q, upr);
    MLIR_ValueHandle nz = build_icmp(&L, /*ne*/1, q, c0);
    term_cond_br(&L, nz, head, signb);

    // signb: if neg, prepend '-'
    L.cur = signb; L.terminated = false;
    MLIR_ValueHandle neg = emit_load_ty(&L, npr, i64);
    MLIR_ValueHandle isn = build_icmp(&L, /*ne*/1, neg, c0);
    term_cond_br(&L, isn, minusb, wblk);

    L.cur = minusb; L.terminated = false;
    MLIR_ValueHandle mp = emit_load_ty(&L, ppr, i64);
    MLIR_ValueHandle mp1 = emit_binop2(&L, OP_TYPE_LLVM_SUB, mp, emit_const(&L, i64, 1), i64);
    emit_store_v(&L, mp1, ppr);
    MLIR_ValueHandle mbp = emit_byte_ptr(&L, buf, mp1);
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_TRUNC, emit_const(&L, i64, 45), i8), mbp);
    term_br(&L, wblk);

    // wblk: _write(1, buf+pos, 32-pos)
    L.cur = wblk; L.terminated = false;
    MLIR_ValueHandle fpos = emit_load_ty(&L, ppr, i64);
    MLIR_ValueHandle len  = emit_binop2(&L, OP_TYPE_LLVM_SUB, c32, fpos, i64);
    MLIR_ValueHandle wp   = emit_byte_ptr(&L, buf, fpos);
    emit_write_call(&L, emit_const(&L, i32, 1), wp, len);
    MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, NULL, 0);
    emit(&L, ret);

    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)"printI64", 8) };
    MLIR_RegionHandle regs[1] = { reg };
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}), MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, out_body, fn);
    free(L.frames);
}

// void printStr(i32 wasm_offset): scan a NUL-terminated string in linmem,
// _write it to stdout, then a newline (matches runtime.c printStr).
static void synth_print_str(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);

    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);

    MLIR_ValueHandle arg = MLIR_CreateValueBlockArg(ctx, (string){0}, 0, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockArg(ctx, entry, arg);

    FLower L = {0};
    L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle host = linmem_ptr(&L, arg, 0);
    MLIR_ValueHandle host_i = emit_cast(&L, OP_TYPE_LLVM_PTRTOINT, host, i64);
    MLIR_ValueHandle curpr = emit_alloca(&L, i64, 1);
    emit_store_v(&L, host_i, curpr);

    MLIR_BlockHandle loop = new_block(&L);
    MLIR_BlockHandle body = new_block(&L);
    MLIR_BlockHandle done = new_block(&L);
    term_br(&L, loop);

    // loop: b = *cur; if b==0 goto done else body
    L.cur = loop; L.terminated = false;
    MLIR_ValueHandle cur = emit_load_ty(&L, curpr, i64);
    MLIR_ValueHandle cp  = emit_byte_ptr(&L, host, emit_binop2(&L, OP_TYPE_LLVM_SUB, cur, host_i, i64));
    MLIR_ValueHandle b   = emit_load_ty(&L, cp, i8);
    MLIR_ValueHandle bz  = build_icmp(&L, /*eq*/0, emit_cast(&L, OP_TYPE_LLVM_ZEXT, b, i64), emit_const(&L, i64, 0));
    term_cond_br(&L, bz, done, body);

    // body: cur++
    L.cur = body; L.terminated = false;
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, cur, emit_const(&L, i64, 1), i64), curpr);
    term_br(&L, loop);

    // done: len = cur - host_i; write(1, host, len); write(1, "\n", 1)
    L.cur = done; L.terminated = false;
    MLIR_ValueHandle endv = emit_load_ty(&L, curpr, i64);
    MLIR_ValueHandle len  = emit_binop2(&L, OP_TYPE_LLVM_SUB, endv, host_i, i64);
    emit_write_call(&L, emit_const(&L, i32, 1), host, len);
    MLIR_ValueHandle nlp = emit_addressof(&L, FMT_NL_GLOBAL);
    emit_write_call(&L, emit_const(&L, i32, 1), nlp, emit_const(&L, i64, 1));
    MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, NULL, 0);
    emit(&L, ret);

    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)"printStr", 8) };
    MLIR_RegionHandle regs[1] = { reg };
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}), MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, out_body, fn);
    free(L.frames);
}
static void scan_max_global(MLIR_OpHandle op, int64_t *max_idx) {
    MLIR_OpType t = MLIR_GetOpType(op);
    if (t == OP_TYPE_WASMSSA_GLOBAL_GET || t == OP_TYPE_WASMSSA_GLOBAL_SET) {
        int64_t idx = at_i(op, "global_idx");
        if (idx > *max_idx) *max_idx = idx;
    }
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(op, r);
        size_t nb = MLIR_GetRegionNumBlocks(reg);
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(blk);
            for (size_t i = 0; i < no; i++)
                scan_max_global(MLIR_GetBlockOp(blk, i), max_idx);
        }
    }
}

MLIR_OpHandle mlir_wasmssa_to_llvm(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ssa_module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  out_body = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);

    // -- Linear-memory size (initial pages). --
    int64_t min_pages = 2;
    MLIR_AttributeHandle mpa = MLIR_GetOpAttributeByName(ssa_module, "memory_min_pages");
    if (mpa) min_pages = MLIR_GetAttributeInteger(mpa);
    if (min_pages < 1) min_pages = 1;

    // -- Pass 1: assign each import_global a fixed linmem offset. --
    OffsetMap globals = {0};
    int32_t cursor = (int32_t)WASM_DATA_BASE;
    int64_t data_end = WASM_DATA_BASE;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(top) != OP_TYPE_WASMSSA_IMPORT_GLOBAL) continue;
        string sn = at_s(top, "sym_name");
        string id = at_s(top, "init_data");
        int64_t sz = at_i(top, "size");
        int64_t ap = at_i(top, "align_pow");
        MLIR_AttributeHandle fa = MLIR_GetOpAttributeByName(top, "fixed_offset");
        if (sz <= 0) sz = (int64_t)id.size;
        int32_t align = (ap > 0) ? (int32_t)(1 << ap) : 1;
        if (fa) cursor = (int32_t)MLIR_GetAttributeInteger(fa);
        else    cursor = (cursor + align - 1) & ~(align - 1);
        omap_add(&globals, sn, cursor);
        cursor += (int32_t)sz;
        if (cursor > data_end) data_end = cursor;
    }

    int64_t linmem_total = (int64_t)min_pages * 65536;
    if (data_end > linmem_total) linmem_total = (data_end + 65535) & ~(int64_t)65535;

    // -- Pass 2: build the linmem image, applying relocs. --
    uint8_t *image = (uint8_t *)calloc((size_t)linmem_total, 1);
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(top) != OP_TYPE_WASMSSA_IMPORT_GLOBAL) continue;
        string sn = at_s(top, "sym_name");
        string id = at_s(top, "init_data");
        string rl = at_s(top, "relocs");
        int32_t my_off = 0;
        (void)omap_get(&globals, sn, &my_off);
        if (id.size > 0)
            memcpy(image + my_off, id.str, id.size);
        // Relocs: "off:target:addend,..." -> store 32-bit LE (target+addend).
        const char *p = rl.str, *e = rl.str + rl.size;
        while (p < e) {
            while (p < e && (*p == ',' || *p == ' ' || *p == '\t')) p++;
            if (p >= e) break;
            long off_local = 0;
            while (p < e && *p >= '0' && *p <= '9') { off_local = off_local*10 + (*p-'0'); p++; }
            if (p >= e || *p != ':') break;
            p++;
            const char *tname = p;
            while (p < e && *p != ':') p++;
            size_t tlen = (size_t)(p - tname);
            if (p >= e || *p != ':') break;
            p++;
            long addend = 0; int neg = 0;
            if (p < e && *p == '-') { neg = 1; p++; }
            while (p < e && *p >= '0' && *p <= '9') { addend = addend*10 + (*p-'0'); p++; }
            if (neg) addend = -addend;
            string tn = { tname, tlen };
            int32_t toff = 0;
            if (!omap_get(&globals, tn, &toff)) {
                fprintf(stderr,
                    "wasmssa->llvm: reloc references unknown global '%.*s'\n",
                    (int)tlen, tname);
                free(image); return MLIR_INVALID_HANDLE;
            }
            uint32_t val = (uint32_t)(toff + (int32_t)addend);
            if ((int64_t)my_off + off_local + 4 <= linmem_total) {
                image[my_off+off_local+0] = (uint8_t)(val);
                image[my_off+off_local+1] = (uint8_t)(val >> 8);
                image[my_off+off_local+2] = (uint8_t)(val >> 16);
                image[my_off+off_local+3] = (uint8_t)(val >> 24);
            }
        }
    }

    // -- Emit @__wasm_linmem (the linear-memory image). --
    MLIR_TypeHandle i8 = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_TypeHandle arr = MLIR_CreateTypeLLVMArray(ctx, i8, (uint64_t)linmem_total);
    string img_bytes = { (char *)image, (size_t)linmem_total };
    MLIR_OpHandle linmem_g = MLIR_CreateLLVMGlobalArrayInit(ctx,
        str_from_cstr_view((char *)LINMEM_GLOBAL), arr, false, img_bytes,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockOp(ctx, out_body, linmem_g);

    // -- Emit wasm globals @__wasm_g0..gN (g0 = shadow stack pointer). --
    int64_t max_global = -1;
    for (size_t i = 0; i < nops; i++)
        scan_max_global(MLIR_GetBlockOp(mb, i), &max_global);
    MLIR_TypeHandle i32t = MLIR_CreateTypeInteger(ctx, 32, true);
    for (int64_t g = 0; g <= max_global; g++) {
        char gname[32];
        snprintf(gname, sizeof(gname), "__wasm_g%lld", (long long)g);
        int64_t init = (g == 0) ? linmem_total : 0;
        MLIR_OpHandle gg = MLIR_CreateLLVMGlobal(ctx,
            str_from_cstr_view(gname), i32t, false, 0, init, 0.0, NULL,
            MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockOp(ctx, out_body, gg);
    }
    // -- Page-count global for memory.size / memory.grow. --
    MLIR_OpHandle pg = MLIR_CreateLLVMGlobal(ctx,
        str_from_cstr_view((char *)"__wasm_mem_pages"), i32t, false, 0, min_pages, 0.0, NULL,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockOp(ctx, out_body, pg);

    // -- Imports: allow the known WASI/print imports; reject others. --
    bool need_printI64 = false, need_printNewline = false, need_printStr = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSSA_IMPORT_FUNC) continue;
        string nm = at_s(op, "sym_name");
        if (nm.size == 9 && memcmp(nm.str, "proc_exit", 9) == 0) continue;
        if (nm.size == 8 && memcmp(nm.str, "printI64", 8) == 0) { need_printI64 = true; continue; }
        if (nm.size == 12 && memcmp(nm.str, "printNewline", 12) == 0) { need_printNewline = true; continue; }
        if (nm.size == 8 && memcmp(nm.str, "printStr", 8) == 0) { need_printStr = true; continue; }
        if (nm.size == 8 && memcmp(nm.str, "fd_write", 8) == 0) continue; // resolved if called
        fprintf(stderr,
            "wasmssa->llvm: import '%.*s' not yet supported\n",
            (int)nm.size, nm.str);
        free(image);
        free(globals.names); free(globals.offsets);
        return MLIR_INVALID_HANDLE;
    }

    // -- Lower functions. --
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSSA_FUNC) continue;
        MLIR_OpHandle fn = lower_func(ctx, op, &globals);
        if (fn == MLIR_INVALID_HANDLE) {
            free(image);
            free(globals.names); free(globals.offsets);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AppendBlockOp(ctx, out_body, fn);
    }

    // -- Synthesise print runtime (via _write) for the imports used. --
    if (need_printI64) synth_print_i64(ctx, out_body);
    if (need_printNewline || need_printStr) {
        char nl[2] = { '\n', '\0' };
        string b = { nl, 2 };
        MLIR_OpHandle g = MLIR_CreateLLVMGlobalString(ctx,
            str_from_cstr_view((char *)FMT_NL_GLOBAL), b,
            MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockOp(ctx, out_body, g);
    }
    if (need_printNewline) synth_print_newline(ctx, out_body);
    if (need_printStr) synth_print_str(ctx, out_body);

    free(globals.names); free(globals.offsets);
    free(image);

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

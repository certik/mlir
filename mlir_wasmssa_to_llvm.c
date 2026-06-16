// See mlir_wasmssa_to_llvm.h for the role of this pass in the pipeline.
//
// Lifts a `wasmssa` builtin.module into the in-house `llvm` MLIR dialect so
// the single unified backend (mlir_llvm_to_aarch64.c) can serve the
// `--from-wasm` self-host path. This is the WASM-input counterpart to the
// C-frontend emit.c (which produces the `llvm` dialect directly).
//
// Coverage was grown test-by-test. It handles the full self-host surface:
// module walk, import_func recognition, per-function locals-as-alloca
// lowering, integer/float ops, control flow (block/loop/if/br/br_if
// flattened to cf.br/cf.cond_br), linear memory + globals, and the WASI
// runtime shims. Any unsupported op makes the lowering fail cleanly with a
// diagnostic.

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

// Map a wasm valtype to its `llvm`-dialect type. Integers map to iN; f32/f64
// map to real float types (the unified backend supports float load/store,
// arith, compare and conversions end-to-end).
static MLIR_TypeHandle vt_to_llvm(MLIR_Context *ctx, uint8_t vt) {
    if (vt == WT_I64) return MLIR_CreateTypeInteger(ctx, 64, true);
    if (vt == WT_F32) return MLIR_CreateTypeFloat(ctx, 32, false);
    if (vt == WT_F64) return MLIR_CreateTypeFloat(ctx, 64, false);
    return MLIR_CreateTypeInteger(ctx, 32, true);
}
static bool vt_is_int(uint8_t vt) { return vt == WT_I32 || vt == WT_I64; }
static bool vt_is_float(uint8_t vt) { return vt == WT_F32 || vt == WT_F64; }

// Re-materialise a FRESH type in `ctx` matching the (possibly foreign-arena)
// tracked type `t`. Under --from-wasm the wasmssa module's arena is freed
// between passes, so a type handle that points into it dangles and later reads
// back as '?'. Decoding via the type string (valid while the source arena is
// alive) and recreating in `ctx` yields a stable handle. Defaults to i32.
static MLIR_TypeHandle fresh_type_like(MLIR_Context *ctx, MLIR_TypeHandle t) {
    string s = MLIR_GetTypeString(ctx, t);
    if (s.size == 3 && memcmp(s.str, "i64", 3) == 0)
        return MLIR_CreateTypeInteger(ctx, 64, true);
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0)
        return MLIR_CreateTypeFloat(ctx, 32, false);
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0)
        return MLIR_CreateTypeFloat(ctx, 64, false);
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

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
#define LINMEM_BASE_GLOBAL "__wasm_linmem_base"
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
// Function-pointer support. func_addr targets are
// interned into a slot table; call_indirect dispatches via a synthesised
// __dispatch_<sig> function that switches on the slot. FuncSigMap records each
// function's (param_types, result_types) wasm-type strings.
// =============================================================================
typedef struct {
    string   *names;
    int32_t  *slots;
    size_t    n, cap;
} FuncPtrMap;

static int32_t fpm_intern_with_slot(FuncPtrMap *m, string name, int32_t explicit_slot) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->names[i].size == name.size &&
            memcmp(m->names[i].str, name.str, name.size) == 0) {
            if (explicit_slot >= 0) m->slots[i] = explicit_slot;
            return m->slots[i];
        }
    }
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->names = (string  *)realloc(m->names, m->cap * sizeof(string));
        m->slots = (int32_t *)realloc(m->slots, m->cap * sizeof(int32_t));
    }
    int32_t s = (explicit_slot >= 0) ? explicit_slot : (int32_t)m->n;
    m->names[m->n] = name;
    m->slots[m->n] = s;
    m->n++;
    return s;
}
static int fpm_lookup(const FuncPtrMap *m, string name, int32_t *out) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->names[i].size == name.size &&
            memcmp(m->names[i].str, name.str, name.size) == 0) {
            *out = m->slots[i]; return 1;
        }
    }
    return 0;
}

typedef struct {
    string   *names;
    string   *param_types;
    string   *result_types;
    size_t    n, cap;
} FuncSigMap;

static void fsm_add(FuncSigMap *m, string name, string pt, string rt) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->names        = (string *)realloc(m->names,        m->cap * sizeof(string));
        m->param_types  = (string *)realloc(m->param_types,  m->cap * sizeof(string));
        m->result_types = (string *)realloc(m->result_types, m->cap * sizeof(string));
    }
    m->names[m->n] = name;
    m->param_types[m->n] = pt;
    m->result_types[m->n] = rt;
    m->n++;
}
static int fsm_get(const FuncSigMap *m, string name, string *pt, string *rt) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->names[i].size == name.size &&
            memcmp(m->names[i].str, name.str, name.size) == 0) {
            *pt = m->param_types[i]; *rt = m->result_types[i]; return 1;
        }
    }
    return 0;
}

// Read an integer attribute with a default if absent.
static int64_t at_i_or(MLIR_OpHandle op, const char *name, int64_t dflt) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : dflt;
}

// =============================================================================
// VMap: wasmssa result value -> lifted llvm value. Open-addressing hash keyed
// on the MLIR_ValueHandle (sentinel MLIR_INVALID_HANDLE == empty). Mirrors the
// the func-addr map; lookups only, so iteration order never affects output.
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
    // Within-block store-to-load forwarding ("local mem2reg-lite"). Wasm locals
    // are never address-taken, so within a single (flat, straight-line) block
    // the alloca for local `idx` holds exactly the value of the most recent
    // local.set; a subsequent local.get can reuse that SSA value with no load.
    // The whole map is invalidated lazily whenever the insertion block changes
    // (the lifter emits into each block once, contiguously, never revisiting).
    MLIR_ValueHandle *local_cur;     // last value stored to local idx this block
    bool             *local_cur_set; // whether local_cur[idx] is valid
    MLIR_BlockHandle  local_cur_blk; // block local_cur is valid for
    Frame            *frames;
    size_t            n_frames, frames_cap;
    OffsetMap        *globals;   // import_global name -> linmem offset
    FuncPtrMap       *fnptrs;    // func_addr target name -> slot index
    const FuncSigMap *sigs;      // callee name -> (param,result) wasm types
    MLIR_ValueHandle  linmem_base; // cached i64 load of @__wasm_linmem_base
    bool              have_base;   // whether linmem_base is populated
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

// Lazily invalidate the local store-to-load forwarding map when the insertion
// block has changed since it was last populated. Must be called at the top of
// every local.get / local.set handler before consulting/updating local_cur.
static void local_fwd_sync(FLower *L) {
    if (L->cur != L->local_cur_blk) {
        for (size_t i = 0; i < L->n_locals; i++) L->local_cur_set[i] = false;
        L->local_cur_blk = L->cur;
    }
}

// Forward declarations (defined later) used by linmem_ptr / memory.grow.
static MLIR_ValueHandle emit_load_ty(FLower *L, MLIR_ValueHandle p, MLIR_TypeHandle ty);
static void emit_store_v(FLower *L, MLIR_ValueHandle v, MLIR_ValueHandle p);
static MLIR_ValueHandle emit_libc_call(FLower *L, const char *name,
                                       MLIR_ValueHandle *args, size_t na,
                                       MLIR_TypeHandle rty);

// =============================================================================
// WASI host bridge — how the via-wasm native lowering reaches the OS.
//
// The WASI imports (fd_write, proc_exit, path_open, ...) are synthesised below
// as small `llvm`-dialect shims. The shims are deliberately THIN: they only
// bridge the wasm32 ABI of the lifted module (pointers are 32-bit linear-memory
// offsets; iovec/struct fields are 32-bit) to the native LP64 ABI — translate
// offsets to host pointers, repack the iovec, store results back to linmem —
// and then DELEGATE the actual OS work to corec's platform_<os>.c.
//
// Those platform functions (platform_fd_write / _read / _close / _seek / _tell
// / _path_open / _exit) are compiled LP64 and spliced into this module by the
// driver as __host_platform_* (see tinyc_compile_host_platform / the
// --host-platform flag). So there is ONE implementation of the platform per OS
// — the corec source — shared by the native and the via-wasm paths; the shims
// no longer re-implement it.
//
// Retargeting to Linux/Windows is then just: splice that OS's platform_<os>.c
// (its raw-syscall / kernel32 bodies) instead of platform_macos.c. The shim
// bodies here are OS-independent.
// =============================================================================


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

// Map a wasm float arithmetic opcode to its `llvm`-dialect op name (matched by
// name by the unified backend). f32: 0x92-0x95, f64: 0xa0-0xa3.
static const char *fbinop_name(int64_t opc) {
    switch (opc) {
        case 0x92: case 0xa0: return "llvm.fadd";
        case 0x93: case 0xa1: return "llvm.fsub";
        case 0x94: case 0xa2: return "llvm.fmul";
        case 0x95: case 0xa3: return "llvm.fdiv";
        default: return NULL;
    }
}

// Map a wasm float comparison opcode to an `llvm`-dialect fcmp predicate.
// Ordered predicates (oeq=1, ogt=2, oge=3, olt=4, ole=5, one=6) match the
// backend's fcmp_pred_to_cond table; `ne` maps to `one` which the aarch64
// backend lowers as an unordered-NE (NaN => true), matching wasm f.ne.
// f32: 0x5b-0x60, f64: 0x61-0x66.
static bool fcmp_to_pred(int64_t opc, int64_t *pred) {
    switch (opc) {
        case 0x5b: case 0x61: *pred = 1; return true; // eq
        case 0x5c: case 0x62: *pred = 6; return true; // ne
        case 0x5d: case 0x63: *pred = 4; return true; // lt
        case 0x5e: case 0x64: *pred = 2; return true; // gt
        case 0x5f: case 0x65: *pred = 5; return true; // le
        case 0x60: case 0x66: *pred = 3; return true; // ge
        default: return false;
    }
}

// Build an llvm.fcmp with an i32 boolean result (wasm comparison convention).
static MLIR_ValueHandle build_fcmp(FLower *L, int64_t pred,
                                   MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_Context *ctx = L->ctx;
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_AttributeHandle pa = MLIR_CreateAttributeInteger(
        ctx, str_from_cstr_view((char *)"predicate"), pred, i64);
    MLIR_TypeHandle rt[1] = { i32 };
    MLIR_ValueHandle r[1] = { mk_res(ctx, i32) };
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_OpHandle out = build_op_named(ctx, "llvm.fcmp", &pa, 1, rt, 1, r, ops, 2);
    emit(L, out);
    return r[0];
}

// Emit a single-operand cast op matched by name (fpext/fptrunc/sitofp/...).
static MLIR_ValueHandle emit_cast_named(FLower *L, const char *nm,
                                        MLIR_ValueHandle v, MLIR_TypeHandle toty) {
    MLIR_TypeHandle rt[1] = { toty };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, toty) };
    MLIR_ValueHandle ops[1] = { v };
    MLIR_OpHandle out = build_op_named(L->ctx, nm, NULL, 0, rt, 1, r, ops, 1);
    emit(L, out);
    return r[0];
}

// Emit a two-operand op matched by name (llvm.fadd/...) producing `ty`.
static MLIR_ValueHandle emit_binop_named(FLower *L, const char *nm,
                                         MLIR_ValueHandle a, MLIR_ValueHandle b,
                                         MLIR_TypeHandle ty) {
    MLIR_TypeHandle rt[1] = { ty };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ty) };
    MLIR_ValueHandle ops[2] = { a, b };
    MLIR_OpHandle out = build_op_named(L->ctx, nm, NULL, 0, rt, 1, r, ops, 2);
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

// Emit `llvm.select(cond, tval, fval)` -> `ty`. `cond` is an i32 (0/1).
static MLIR_ValueHandle emit_select(FLower *L, MLIR_ValueHandle cond,
                                    MLIR_ValueHandle tval, MLIR_ValueHandle fval,
                                    MLIR_TypeHandle ty) {
    MLIR_TypeHandle rt[1] = { ty };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ty) };
    MLIR_ValueHandle ops[3] = { cond, tval, fval };
    MLIR_OpHandle out = build_op_named(L->ctx, "llvm.select", NULL, 0, rt, 1, r, ops, 3);
    emit(L, out);
    return r[0];
}

// Materialise the host pointer for wasm address `addr_i32` + static `off`:
//   p = inttoptr( load(@__wasm_linmem_base) + zext(addr) + off )
// `__wasm_linmem_base` is set once at process start by the backend's `_start`
// to the base of a 4 GiB mmap'd region (see synth_start in
// mlir_llvm_to_aarch64.c); the initial data image is memcpy'd into its front.
static MLIR_ValueHandle linmem_ptr(FLower *L, MLIR_ValueHandle addr_i32,
                                   int64_t off) {
    MLIR_Context *ctx = L->ctx;
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(ctx);
    // The base pointer is loaded once per function (in the entry block) and
    // reused for every access; this avoids re-emitting addressof+load on each
    // memory operation, which otherwise dominates the lifted IR size. The
    // synth_* runtime helpers build their own FLower without a cached base, so
    // fall back to a per-access load there (they have few accesses).
    MLIR_ValueHandle base_i64;
    if (L->have_base) {
        base_i64 = L->linmem_base;
    } else {
        MLIR_ValueHandle base_slot = emit_addressof(L, LINMEM_BASE_GLOBAL);
        base_i64 = emit_load_ty(L, base_slot, i64);
    }
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
        if (vt_is_float(vt)) {
            // The `value` attr holds the raw IEEE-754 bit pattern as an integer
            // (f32 in the low 32 bits). Decode to a double and build a
            // float-typed llvm.mlir.constant.
            int64_t raw = at_i(op, "value");
            double d;
            if (vt == WT_F32) {
                uint32_t b32 = (uint32_t)raw;
                float f;
                memcpy(&f, &b32, 4);
                d = (double)f;
            } else {
                uint64_t b64 = (uint64_t)raw;
                memcpy(&d, &b64, 8);
            }
            MLIR_TypeHandle fty = vt_to_llvm(ctx, vt);
            MLIR_AttributeHandle va = MLIR_CreateAttributeFloat(
                ctx, str_from_cstr_view((char *)"value"), d, fty);
            MLIR_TypeHandle rt[1] = { fty };
            MLIR_ValueHandle r[1] = { mk_res(ctx, fty) };
            MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_MLIR_CONSTANT,
                &va, 1, rt, 1, r, NULL, 0);
            emit(L, out);
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
            return true;
        }
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: const valtype %u not supported\n",
                    (unsigned)vt);
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
        local_fwd_sync(L);
        if (L->local_cur_set[idx]) {
            // Reuse the value last stored to this local in the current block.
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), L->local_cur[idx]);
            return true;
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
        // The alloca now demonstrably holds r[0]; cache it so further gets of
        // this local in the same block also forward (still safe: no aliasing).
        L->local_cur[idx] = r[0];
        L->local_cur_set[idx] = true;
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
        // Record the stored value for within-block forwarding of later gets.
        local_fwd_sync(L);
        L->local_cur[idx] = v;
        L->local_cur_set[idx] = true;
        return true;
    }

    case OP_TYPE_WASMSSA_ADD:
    case OP_TYPE_WASMSSA_SUB: {
        uint8_t vt = (uint8_t)at_i(op, "valtype");
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on add/sub\n");
            return false;
        }
        MLIR_TypeHandle ity = vt_to_llvm(ctx, vt);
        if (vt_is_float(vt)) {
            const char *nm = (t == OP_TYPE_WASMSSA_ADD) ? "llvm.fadd"
                                                        : "llvm.fsub";
            MLIR_ValueHandle res = emit_binop_named(L, nm, a, b, ity);
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), res);
            return true;
        }
        MLIR_OpType ot = (t == OP_TYPE_WASMSSA_ADD) ? OP_TYPE_LLVM_ADD
                                                    : OP_TYPE_LLVM_SUB;
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
        if (fcmp_to_pred(opc, &pred)) {
            MLIR_ValueHandle a, b;
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
                !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on fcmp\n");
                return false;
            }
            MLIR_ValueHandle r = build_fcmp(L, pred, a, b);
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r);
            return true;
        }
        const char *fnm = fbinop_name(opc);
        if (fnm) {
            uint8_t vt = (uint8_t)at_i(op, "valtype");
            MLIR_ValueHandle a, b;
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &a) ||
                !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &b)) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on fbinop\n");
                return false;
            }
            MLIR_TypeHandle fty = vt_to_llvm(ctx, vt);
            MLIR_ValueHandle res = emit_binop_named(L, fnm, a, b, fty);
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), res);
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
        // WASI imports map to the host platform implementation (the
        // __host_platform_* funcs compiled from corec's platform_<os>.c and
        // spliced in by the driver via --host-platform).
        if (callee.size == 9 && memcmp(callee.str, "proc_exit", 9) == 0)
            callee = str_from_cstr_view((char *)"__host_platform_exit");
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
            // Under --from-wasm (no_def_use_tracking) the wasmssa.call result
            // value carries no reliable type; prefer the callee signature.
            MLIR_TypeHandle ty = MLIR_INVALID_HANDLE;
            string cpt, crt;
            if (L->sigs && fsm_get(L->sigs, callee, &cpt, &crt) && crt.size >= 2)
                ty = vt_to_llvm(ctx, type_byte_at(crt, 0));
            if (ty == MLIR_INVALID_HANDLE) {
                MLIR_TypeHandle tracked = MLIR_GetValueType(MLIR_GetOpResult(op, 0));
                ty = fresh_type_like(ctx, tracked);
            }
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
        // Result type follows the value operands; the result value itself is
        // untyped under no_def_use_tracking, so take it from operand `a`.
        MLIR_TypeHandle ity = MLIR_GetValueType(a);
        string itys = MLIR_GetTypeString(ctx, ity);
        if (itys.size == 0 || itys.str == NULL || itys.str[0] == '?' ||
            (itys.size == 7 && memcmp(itys.str, "unknown", 7) == 0)) {
            ity = MLIR_GetValueType(b);
            itys = MLIR_GetTypeString(ctx, ity);
            if (itys.size == 0 || itys.str == NULL || itys.str[0] == '?' ||
                (itys.size == 7 && memcmp(itys.str, "unknown", 7) == 0))
                ity = MLIR_CreateTypeInteger(ctx, 32, true);
        }
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
        if (vt_is_float(vt)) {
            if (!((vt == WT_F32 && sz == 4) || (vt == WT_F64 && sz == 8))) {
                fprintf(stderr,
                    "wasmssa->llvm: float load mem_size=%lld valtype=%u "
                    "not supported\n", (long long)sz, (unsigned)vt);
                return false;
            }
            MLIR_ValueHandle addr;
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &addr)) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on load\n");
                return false;
            }
            MLIR_ValueHandle p = linmem_ptr(L, addr, off);
            MLIR_TypeHandle fty = vt_to_llvm(ctx, vt);
            MLIR_TypeHandle lrt[1] = { fty };
            MLIR_ValueHandle lr[1] = { mk_res(ctx, fty) };
            MLIR_ValueHandle lops[1] = { p };
            MLIR_OpHandle lop = build_op(ctx, OP_TYPE_LLVM_LOAD,
                NULL, 0, lrt, 1, lr, lops, 1);
            emit(L, lop);
            vmap_set(L->vmap, MLIR_GetOpResult(op, 0), lr[0]);
            return true;
        }
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: load valtype %u not supported\n",
                    (unsigned)vt);
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
        if (vt_is_float(vt)) {
            if (!((vt == WT_F32 && sz == 4) || (vt == WT_F64 && sz == 8))) {
                fprintf(stderr,
                    "wasmssa->llvm: float store mem_size=%lld valtype=%u "
                    "not supported\n", (long long)sz, (unsigned)vt);
                return false;
            }
            MLIR_ValueHandle addr, val;
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &addr) ||
                !vmap_get(L->vmap, MLIR_GetOpOperand(op, 1), &val)) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on store\n");
                return false;
            }
            MLIR_ValueHandle p = linmem_ptr(L, addr, off);
            MLIR_ValueHandle ops[2] = { val, p };
            MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_STORE,
                NULL, 0, NULL, 0, NULL, ops, 2);
            emit(L, out);
            return true;
        }
        if (!vt_is_int(vt)) {
            fprintf(stderr, "wasmssa->llvm: store valtype %u not supported\n",
                    (unsigned)vt);
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
        // 2-byte store uses a 4-byte store.
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
        // Bump the page-count global by `delta`, capped at the wasm32 limit of
        // 65536 pages (the 4 GiB region `_start` reserved). Returns the old
        // page count on success, or -1 if the grow would exceed the cap. The
        // mmap is lazily backed so no real allocation is needed.
        MLIR_ValueHandle delta;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, 0), &delta)) {
            fprintf(stderr, "wasmssa->llvm: unbound operand on memory_grow\n");
            return false;
        }
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
        MLIR_ValueHandle p = emit_addressof(L, "__wasm_mem_pages");
        MLIR_ValueHandle old = emit_load_ty(L, p, i32);
        // new = old + delta computed in i64 to avoid i32 overflow.
        MLIR_ValueHandle old64 = emit_cast(L, OP_TYPE_LLVM_ZEXT, old, i64);
        MLIR_ValueHandle delta64 = emit_cast(L, OP_TYPE_LLVM_ZEXT, delta, i64);
        MLIR_ValueHandle new64 = emit_binop2(L, OP_TYPE_LLVM_ADD, old64, delta64, i64);
        MLIR_ValueHandle cap = emit_const(L, i64, 65536);
        MLIR_ValueHandle ok = build_icmp(L, /*ule*/7, new64, cap);
        MLIR_ValueHandle new32 = emit_cast(L, OP_TYPE_LLVM_TRUNC, new64, i32);
        // Keep old page count on failure; store the (possibly unchanged) value.
        MLIR_ValueHandle stored = emit_select(L, ok, new32, old, i32);
        emit_store_v(L, stored, emit_addressof(L, "__wasm_mem_pages"));
        // Result: old on success, -1 on failure.
        MLIR_ValueHandle neg1 = emit_const(L, i32, -1);
        MLIR_ValueHandle res = emit_select(L, ok, old, neg1, i32);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), res);
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
        MLIR_TypeHandle f32 = MLIR_CreateTypeFloat(ctx, 32, false);
        MLIR_TypeHandle f64 = MLIR_CreateTypeFloat(ctx, 64, false);
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
        // ---- float <-> int conversions ----
        // f -> i (signed): trunc_f{32,64}_s
        case 0xa8: res = emit_cast_named(L, "llvm.fptosi", a, i32); break;
        case 0xaa: res = emit_cast_named(L, "llvm.fptosi", a, i32); break;
        case 0xae: res = emit_cast_named(L, "llvm.fptosi", a, i64); break;
        case 0xb0: res = emit_cast_named(L, "llvm.fptosi", a, i64); break;
        // f -> i (unsigned): trunc_f{32,64}_u
        case 0xa9: res = emit_cast_named(L, "llvm.fptoui", a, i32); break;
        case 0xab: res = emit_cast_named(L, "llvm.fptoui", a, i32); break;
        case 0xaf: res = emit_cast_named(L, "llvm.fptoui", a, i64); break;
        case 0xb1: res = emit_cast_named(L, "llvm.fptoui", a, i64); break;
        // i -> f (signed): convert_i{32,64}_s
        case 0xb2: res = emit_cast_named(L, "llvm.sitofp", a, f32); break;
        case 0xb4: res = emit_cast_named(L, "llvm.sitofp", a, f32); break;
        case 0xb7: res = emit_cast_named(L, "llvm.sitofp", a, f64); break;
        case 0xb9: res = emit_cast_named(L, "llvm.sitofp", a, f64); break;
        // i -> f (unsigned): convert_i{32,64}_u
        case 0xb3: res = emit_cast_named(L, "llvm.uitofp", a, f32); break;
        case 0xb5: res = emit_cast_named(L, "llvm.uitofp", a, f32); break;
        case 0xb8: res = emit_cast_named(L, "llvm.uitofp", a, f64); break;
        case 0xba: res = emit_cast_named(L, "llvm.uitofp", a, f64); break;
        // f <-> f
        case 0xb6: res = emit_cast_named(L, "llvm.fptrunc", a, f32); break; // demote
        case 0xbb: res = emit_cast_named(L, "llvm.fpext",   a, f64); break; // promote
        // ---- float unary ops ----
        case 0x8c: res = emit_cast_named(L, "llvm.fneg",      a, f32); break; // f32.neg
        case 0x91: res = emit_cast_named(L, "llvm.intr.sqrt", a, f32); break; // f32.sqrt
        case 0x9a: res = emit_cast_named(L, "llvm.fneg",      a, f64); break; // f64.neg
        case 0x9f: res = emit_cast_named(L, "llvm.intr.sqrt", a, f64); break; // f64.sqrt
        default:
            fprintf(stderr,
                "wasmssa->llvm: unop opcode 0x%llx not yet supported\n",
                (long long)opc);
            return false;
        }
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), res);
        return true;
    }

    case OP_TYPE_WASMSSA_FUNC_ADDR: {
        // Returns the slot index assigned to the named function in the
        // module fnptr table, lowered to an i32 constant.
        string target = at_s(op, "target");
        int32_t slot = -1;
        if (!L->fnptrs || !fpm_lookup(L->fnptrs, target, &slot)) {
            fprintf(stderr,
                "wasmssa->llvm: func_addr target '%.*s' not in fnptr table\n",
                (int)target.size, target.str);
            return false;
        }
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_ValueHandle rv = emit_const(L, i32, slot);
        vmap_set(L->vmap, MLIR_GetOpResult(op, 0), rv);
        return true;
    }

    case OP_TYPE_WASMSSA_CALL_INDIRECT: {
        // Operands: (args..., slot). Lower to a call of the synthesised
        // dispatcher __dispatch_<params>_<results>, passing (slot, args...).
        size_t no = MLIR_GetOpNumOperands(op);
        if (no < 1) {
            fprintf(stderr, "wasmssa->llvm: call_indirect with no operands\n");
            return false;
        }
        MLIR_ValueHandle *ops_in  = (MLIR_ValueHandle *)malloc(no * sizeof(MLIR_ValueHandle));
        MLIR_ValueHandle *ops_out = (MLIR_ValueHandle *)malloc(no * sizeof(MLIR_ValueHandle));
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(op, k), &ops_in[k])) {
                fprintf(stderr, "wasmssa->llvm: unbound operand on call_indirect\n");
                free(ops_in); free(ops_out);
                return false;
            }
        }
        ops_out[0] = ops_in[no - 1];
        for (size_t k = 0; k + 1 < no; k++) ops_out[1 + k] = ops_in[k];

        string sig_p = at_s(op, "sig_params");
        string sig_r = at_s(op, "sig_results");
        size_t name_cap = sig_p.size + sig_r.size + 32;
        char *name_buf = (char *)malloc(name_cap);
        int nlen = snprintf(name_buf, name_cap, "__dispatch_%.*s_%.*s",
            (int)sig_p.size, sig_p.str, (int)sig_r.size, sig_r.str);
        if (nlen <= 0 || (size_t)nlen >= name_cap) {
            fprintf(stderr, "wasmssa->llvm: call_indirect dispatcher name too long\n");
            free(name_buf); free(ops_in); free(ops_out);
            return false;
        }
        size_t nr = MLIR_GetOpNumResults(op);
        if (nr > 1) {
            fprintf(stderr, "wasmssa->llvm: call_indirect multi-result unsupported\n");
            free(name_buf); free(ops_in); free(ops_out);
            return false;
        }
        MLIR_AttributeHandle attrs[1] = { attr_s(ctx, "callee", name_buf, (size_t)nlen) };
        MLIR_TypeHandle rt[1];
        MLIR_ValueHandle r[1];
        if (nr == 1) {
            MLIR_TypeHandle ty;
            if (sig_r.size >= 2) {
                ty = vt_to_llvm(ctx, type_byte_at(sig_r, 0));
            } else {
                MLIR_TypeHandle tracked = MLIR_GetValueType(MLIR_GetOpResult(op, 0));
                ty = fresh_type_like(ctx, tracked);
            }
            rt[0] = ty;
            r[0] = mk_res(ctx, ty);
        }
        MLIR_OpHandle out = build_op(ctx, OP_TYPE_LLVM_CALL,
            attrs, 1, rt, nr, r, ops_out, no);
        emit(L, out);
        if (nr == 1) vmap_set(L->vmap, MLIR_GetOpResult(op, 0), r[0]);
        // name_buf must outlive the attr (attr_s does not copy); intentionally
        // leaked, one per call_indirect site.
        free(ops_in); free(ops_out);
        return true;
    }

    default:
        fprintf(stderr,
            "wasmssa->llvm: op '%.*s' not yet supported\n",
            (int)MLIR_GetOpName(op).size, MLIR_GetOpName(op).str);
        return false;
    }
}

// =============================================================================
// Boolean-normalization peephole (the WMIR `wmir_simplify_bools` analogue).
//
// C frontends materialise a comparison as an explicit 0/1 integer and then
// re-test it for control flow, so the wasm (and hence the lifted llvm) carries
// chains like
//
//     %b = llvm.icmp ...                       ; already 0 or 1 (i32)
//     %s = llvm.select(%b, 1, 0)               ; %b ? 1 : 0   (identity on bool)
//     %n = llvm.icmp ne (%s, 0)                ; %s != 0       (identity on bool)
//     ... use %n ...
//
// and `!!c` idioms stack several select/icmp_ne pairs. wasmtime's Cranelift
// folds these; our straight-line backend otherwise emits a csel/cset round-trip
// per layer. Both `select(c,1,0)` and `icmp_ne(x,0)` equal their inner value
// whenever that value is a 0/1 boolean. A value is a boolean if it is an
// llvm.icmp / llvm.fcmp result OR is itself one of these folded redundancies
// (so `!!` chains collapse transitively in a single forward pass). We forward
// the result to the inner value at every use and erase the now-dead op.
// =============================================================================
typedef struct { MLIR_ValueHandle key, val; } BoolFwd;

static size_t bf_hash(MLIR_ValueHandle k, size_t mask) {
    return (size_t)(((uint64_t)k * 0x9E3779B97F4A7C15ull) >> 24) & mask;
}
static void bf_put(BoolFwd *m, size_t mask, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    size_t h = bf_hash(k, mask);
    while (m[h].key != MLIR_INVALID_HANDLE && m[h].key != k) h = (h + 1) & mask;
    m[h].key = k; m[h].val = v;
}
static bool bf_get(BoolFwd *m, size_t mask, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    size_t h = bf_hash(k, mask);
    while (m[h].key != MLIR_INVALID_HANDLE) {
        if (m[h].key == k) { *out = m[h].val; return true; }
        h = (h + 1) & mask;
    }
    return false;
}
static MLIR_ValueHandle bf_resolve(BoolFwd *m, size_t mask, MLIR_ValueHandle v) {
    MLIR_ValueHandle n;
    for (int g = 0; g < 64 && bf_get(m, mask, v, &n); g++) v = n;
    return v;
}
static bool sb_op_name_is(MLIR_OpHandle op, const char *nm) {
    string s = MLIR_GetOpName(op);
    size_t l = strlen(nm);
    return s.size == l && s.str != NULL && memcmp(s.str, nm, l) == 0;
}
static bool sb_is_cmp_def(MLIR_ValueHandle v) {
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(v);
    if (!d) return false;
    return sb_op_name_is(d, "llvm.icmp") || sb_op_name_is(d, "llvm.fcmp");
}
static bool sb_is_const(MLIR_ValueHandle v, int64_t want) {
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(v);
    if (!d || MLIR_GetOpType(d) != OP_TYPE_LLVM_MLIR_CONSTANT) return false;
    return at_i(d, "value") == want;
}
// Total operand slots (regular + per-successor block-arg operands), addressable
// by the flat index MLIR_GetOpOperand / MLIR_SetOpOperand use.
static size_t sb_flat_count(MLIR_OpHandle op) {
    size_t n = MLIR_GetOpNumOperands(op);
    size_t ns = MLIR_GetOpNumSuccessors(op);
    for (size_t s = 0; s < ns; s++) n += MLIR_GetOpNumSuccessorOperands(op, s);
    return n;
}
static MLIR_ValueHandle sb_flat_get(MLIR_OpHandle op, size_t i) {
    size_t nreg = MLIR_GetOpNumOperands(op);
    if (i < nreg) return MLIR_GetOpOperand(op, i);
    size_t rem = i - nreg, ns = MLIR_GetOpNumSuccessors(op);
    for (size_t s = 0; s < ns; s++) {
        size_t c = MLIR_GetOpNumSuccessorOperands(op, s);
        if (rem < c) return MLIR_GetOpSuccessorOperand(op, s, rem);
        rem -= c;
    }
    return MLIR_INVALID_HANDLE;
}

// `op` is a fold candidate; return the inner value it is equivalent to (using
// the in-progress map for the bool test so chains collapse), else INVALID.
static MLIR_ValueHandle sb_fold_inner(BoolFwd *m, size_t mask, MLIR_OpHandle op) {
    MLIR_ValueHandle dummy;
    if (sb_op_name_is(op, "llvm.select") && MLIR_GetOpNumOperands(op) == 3) {
        MLIR_ValueHandle c = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle t = MLIR_GetOpOperand(op, 1);
        MLIR_ValueHandle f = MLIR_GetOpOperand(op, 2);
        if (sb_is_const(t, 1) && sb_is_const(f, 0) &&
            (sb_is_cmp_def(c) || bf_get(m, mask, c, &dummy)))
            return bf_resolve(m, mask, c);
    } else if (MLIR_GetOpType(op) == OP_TYPE_LLVM_ICMP &&
               at_i(op, "predicate") == 1 /*ne*/ &&
               MLIR_GetOpNumOperands(op) == 2) {
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(op, 1);
        if (sb_is_const(b, 0) && (sb_is_cmp_def(a) || bf_get(m, mask, a, &dummy)))
            return bf_resolve(m, mask, a);
        if (sb_is_const(a, 0) && (sb_is_cmp_def(b) || bf_get(m, mask, b, &dummy)))
            return bf_resolve(m, mask, b);
    }
    return MLIR_INVALID_HANDLE;
}

static void simplify_bools(MLIR_Context *ctx, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    // Pass 1: upper-bound the candidate count (select/icmp-shaped ops) to size
    // the forwarding map and erase list.
    size_t ncand = 0;
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(blk);
        for (size_t oi = 0; oi < no; oi++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, oi);
            if (sb_op_name_is(op, "llvm.select") ||
                (MLIR_GetOpType(op) == OP_TYPE_LLVM_ICMP && at_i(op, "predicate") == 1))
                ncand++;
        }
    }
    if (ncand == 0) return;

    size_t cap = 16;
    while (cap < ncand * 4) cap <<= 1;
    size_t mask = cap - 1;
    BoolFwd *map = (BoolFwd *)calloc(cap, sizeof(*map));
    MLIR_OpHandle *erase = (MLIR_OpHandle *)malloc(ncand * sizeof(*erase));
    if (!map || !erase) { free(map); free(erase); return; }
    size_t nerase = 0;

    // Pass 2: forward pass in program order (defs precede uses), recording
    // result -> inner forwardings and the ops to drop.
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(blk);
        for (size_t oi = 0; oi < no; oi++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, oi);
            MLIR_ValueHandle inner = sb_fold_inner(map, mask, op);
            if (inner != MLIR_INVALID_HANDLE) {
                bf_put(map, mask, MLIR_GetOpResult(op, 0), inner);
                erase[nerase++] = op;
            }
        }
    }

    // Pass 3: repoint every operand (regular + successor block-arg) that
    // references a folded result onto the resolved inner value. Every operand
    // pre-dates its op, so the inner value dominates each rewritten use.
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(blk);
        for (size_t oi = 0; oi < no; oi++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, oi);
            size_t nf = sb_flat_count(op);
            for (size_t i = 0; i < nf; i++) {
                MLIR_ValueHandle cur = sb_flat_get(op, i);
                MLIR_ValueHandle r = bf_resolve(map, mask, cur);
                if (r != cur) MLIR_SetOpOperand(ctx, op, i, r);
            }
        }
    }

    // Pass 4: erase the now-dead folded ops. Their results have no remaining
    // users; the feeding 0/1 constants emit no code (rematerialised at use, now
    // unused) so they are left for the backend to drop.
    for (size_t i = 0; i < nerase; i++) MLIR_EraseOp(ctx, erase[i]);

    free(map); free(erase);
}

// Lift one wasmssa.func into an `llvm.func`. Returns MLIR_INVALID_HANDLE on
// failure (the caller aborts the whole module).
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src,
                                OffsetMap *globals, FuncPtrMap *fnptrs,
                                const FuncSigMap *sigs) {
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
    // The wasmssa `local_types` attr already enumerates the FULL local index
    // space — params first, then declared locals — matching the wasm local
    // numbering (see mlir_wasm_to_macho.c local_types construction). Do NOT
    // prepend params again; index local_vt[i] directly from local_types.
    size_t n_lt     = lt.size / 2;
    size_t n_locals = n_lt > n_params ? n_lt : n_params;

    // Decode and validate all local valtypes up front (integer-only here).
    uint8_t *local_vt = (uint8_t *)malloc((n_locals ? n_locals : 1));
    for (size_t i = 0; i < n_locals; i++)
        local_vt[i] = (i < n_lt) ? type_byte_at(lt, i) : type_byte_at(pt, i);
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
    L.fnptrs = fnptrs;
    L.sigs = sigs;
    L.local_ptr = (MLIR_ValueHandle *)malloc(
        (n_locals ? n_locals : 1) * sizeof(MLIR_ValueHandle));
    L.local_cur = (MLIR_ValueHandle *)malloc(
        (n_locals ? n_locals : 1) * sizeof(MLIR_ValueHandle));
    L.local_cur_set = (bool *)calloc(
        (n_locals ? n_locals : 1), sizeof(bool));
    L.local_cur_blk = MLIR_INVALID_HANDLE;

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

    // Load the linear-memory base pointer once in the entry block; linmem_ptr
    // reuses this cached value for every access (the entry block dominates the
    // whole body, so the value is valid everywhere). L.cur is still the entry
    // block here (body lowering has not started).
    {
        MLIR_ValueHandle bslot = emit_addressof(&L, LINMEM_BASE_GLOBAL);
        L.linmem_base = emit_load_ty(&L, bslot, i64ty);
        L.have_base = true;
    }

    // Lower the body. Control-flow ops append further blocks to dreg and
    // advance L.cur; lower_region skips dead ops after a terminator.
    bool ok = lower_region(&L, sblk);

    free(vmap.src);
    free(vmap.dst);
    free(local_vt);
    free(L.local_ptr);
    free(L.local_cur);
    free(L.local_cur_set);
    free(param_args);
    free(L.frames);
    if (!ok) return MLIR_INVALID_HANDLE;

    // Boolean-normalization peephole: collapse select(1,0,bool) / icmp_ne(bool,0)
    // chains the frontend emits around every comparison. Gated for A/B.
    if (!getenv("TINYC_NO_SIMPLIFY_BOOLS"))
        simplify_bools(ctx, dreg);

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

// Emit `_write(fd, ptr, len)` (result discarded). Used by the printI64 /
// printStr runtime shims (tinyC's own print imports, not the platform layer),
// which write to stdout via the libSystem `_write` stub directly.
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

// i32 fd_write(i32 fd, i32 iovs, i32 iovs_len, i32 nwritten): WASI shim. Walks
// `iovs_len` (buf_ofs, len) iovec pairs in linmem starting at `iovs`, _write's
// each chunk, accumulates the byte count to linmem[nwritten], returns 0.
static void synth_fd_write(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);

    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);

    MLIR_ValueHandle pfd = MLIR_CreateValueBlockArg(ctx, (string){0}, 0, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_ValueHandle piov = MLIR_CreateValueBlockArg(ctx, (string){0}, 1, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_ValueHandle plen = MLIR_CreateValueBlockArg(ctx, (string){0}, 2, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_ValueHandle pnw = MLIR_CreateValueBlockArg(ctx, (string){0}, 3, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockArg(ctx, entry, pfd);
    MLIR_AppendBlockArg(ctx, entry, piov);
    MLIR_AppendBlockArg(ctx, entry, plen);
    MLIR_AppendBlockArg(ctx, entry, pnw);

    FLower L = {0};
    L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle iovpr = emit_alloca(&L, i32, 1); // current iovec offset
    MLIR_ValueHandle rempr = emit_alloca(&L, i32, 1); // remaining count
    MLIR_ValueHandle totpr = emit_alloca(&L, i64, 1); // bytes-written accumulator
    MLIR_ValueHandle cvbuf = emit_alloca(&L, i8, 16); // one native ciovec {ptr,i64}
    MLIR_ValueHandle nwchunk = emit_alloca(&L, i64, 1); // per-chunk *nwritten
    emit_store_v(&L, piov, iovpr);
    emit_store_v(&L, plen, rempr);
    emit_store_v(&L, emit_const(&L, i64, 0), totpr);

    MLIR_BlockHandle loop = new_block(&L);
    MLIR_BlockHandle body = new_block(&L);
    MLIR_BlockHandle done = new_block(&L);
    term_br(&L, loop);

    // loop: if rem==0 goto done
    L.cur = loop; L.terminated = false;
    MLIR_ValueHandle rem = emit_load_ty(&L, rempr, i32);
    MLIR_ValueHandle rz = build_icmp(&L, /*eq*/0, rem, emit_const(&L, i32, 0));
    term_cond_br(&L, rz, done, body);

    // body: read iovec (buf_ofs, len), _write, accumulate, advance
    L.cur = body; L.terminated = false;
    MLIR_ValueHandle iov = emit_load_ty(&L, iovpr, i32);
    MLIR_ValueHandle bufp = linmem_ptr(&L, iov, 0);
    MLIR_ValueHandle bufofs = emit_load_ty(&L, bufp, i32);
    MLIR_ValueHandle lenp = linmem_ptr(&L, iov, 4);
    MLIR_ValueHandle clen = emit_load_ty(&L, lenp, i32);
    MLIR_ValueHandle host = linmem_ptr(&L, bufofs, 0);
    MLIR_ValueHandle clen64 = emit_cast(&L, OP_TYPE_LLVM_ZEXT, clen, i64);
    // Build a 1-element native ciovec {buf=host, buf_len=clen64} and call the
    // real platform_fd_write. *nwritten (size_t) receives the bytes written.
    emit_store_v(&L, host, cvbuf);
    emit_store_v(&L, clen64, emit_byte_ptr(&L, cvbuf, emit_const(&L, i64, 8)));
    emit_store_v(&L, emit_const(&L, i64, 0), nwchunk);
    MLIR_ValueHandle hargs[4] = { pfd, cvbuf, emit_const(&L, i64, 1), nwchunk };
    emit_libc_call(&L, "__host_platform_fd_write", hargs, 4, i32);
    MLIR_ValueHandle nbytes = emit_load_ty(&L, nwchunk, i64);
    MLIR_ValueHandle tot = emit_load_ty(&L, totpr, i64);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, tot, nbytes, i64), totpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, iov, emit_const(&L, i32, 8), i32), iovpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_SUB, rem, emit_const(&L, i32, 1), i32), rempr);
    term_br(&L, loop);

    // done: linmem[nwritten] = (i32)total; return 0
    L.cur = done; L.terminated = false;
    MLIR_ValueHandle ftot = emit_load_ty(&L, totpr, i64);
    MLIR_ValueHandle nwp = linmem_ptr(&L, pnw, 0);
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_TRUNC, ftot, i32), nwp);
    MLIR_ValueHandle zero = emit_const(&L, i32, 0);
    MLIR_ValueHandle rops[1] = { zero };
    MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, rops, 1);
    emit(&L, ret);

    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)"fd_write", 8) };
    MLIR_RegionHandle regs[1] = { reg };
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}), MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, out_body, fn);
    free(L.frames);
}

// =============================================================================
// WASI file-I/O + args shims (`--from-wasm` selfhost path).
//
// Each is emitted as an `llvm.func` that calls libc (_read/_lseek/_open/_close)
// with host pointers, translating wasm linear-memory offsets via `linmem_ptr`.
// They provide the WASI host shims at the `llvm`-dialect level. argc/argv are
// recovered from the @__wasm_argc / @__wasm_argv globals that `_start` fills.
// =============================================================================

// Append a block argument of type `ty` at index `idx` to `entry`.
static MLIR_ValueHandle mk_arg(MLIR_Context *ctx, MLIR_BlockHandle entry,
                               int idx, MLIR_TypeHandle ty) {
    MLIR_ValueHandle v = MLIR_CreateValueBlockArg(ctx, (string){0}, idx, ty,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockArg(ctx, entry, v);
    return v;
}

// Wrap region `reg` as an `llvm.func` named `name` and append to out_body.
static void finish_func(MLIR_Context *ctx, MLIR_BlockHandle out_body,
                        MLIR_RegionHandle reg, const char *name) {
    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)name, strlen(name)) };
    MLIR_RegionHandle regs[1] = { reg };
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}), MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, out_body, fn);
}

// Emit `llvm.return %v`.
static void emit_ret_v(FLower *L, MLIR_ValueHandle v) {
    MLIR_ValueHandle ops[1] = { v };
    emit(L, build_op(L->ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, ops, 1));
}

// Emit `llvm.call @name(args...)` returning `rty`.
static MLIR_ValueHandle emit_libc_call(FLower *L, const char *name,
                                       MLIR_ValueHandle *args, size_t na,
                                       MLIR_TypeHandle rty) {
    MLIR_AttributeHandle a[1] = { attr_s(L->ctx, "callee", (char *)name, strlen(name)) };
    MLIR_TypeHandle rt[1] = { rty };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, rty) };
    MLIR_OpHandle out = build_op(L->ctx, OP_TYPE_LLVM_CALL, a, 1, rt, 1, r, args, na);
    emit(L, out);
    return r[0];
}

// inttoptr(i64) -> !llvm.ptr (a raw host pointer, no linmem base added).
static MLIR_ValueHandle emit_i2p(FLower *L, MLIR_ValueHandle i64v) {
    MLIR_TypeHandle ptr = MLIR_CreateTypeLLVMPointer(L->ctx);
    MLIR_TypeHandle rt[1] = { ptr };
    MLIR_ValueHandle r[1] = { mk_res(L->ctx, ptr) };
    MLIR_ValueHandle ops[1] = { i64v };
    emit(L, build_op_named(L->ctx, "llvm.inttoptr", NULL, 0, rt, 1, r, ops, 1));
    return r[0];
}

// i32 fd_close(i32 fd): platform_fd_close(fd); return 0.
static void synth_fd_close(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_ValueHandle fd = mk_arg(ctx, entry, 0, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;
    MLIR_ValueHandle a[1] = { fd };
    emit_libc_call(&L, "__host_platform_fd_close", a, 1, i32);
    emit_ret_v(&L, emit_const(&L, i32, 0));
    finish_func(ctx, out_body, reg, "fd_close");
}

// i32 fd_read(i32 fd, i32 iovs, i32 iovs_len, i32 nread): walk (buf,len)
// iovecs in linmem, _read each, accumulate into *nread. Stops on short read.
static void synth_fd_read(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_ValueHandle pfd  = mk_arg(ctx, entry, 0, i32);
    MLIR_ValueHandle piov = mk_arg(ctx, entry, 1, i32);
    MLIR_ValueHandle plen = mk_arg(ctx, entry, 2, i32);
    MLIR_ValueHandle pnw  = mk_arg(ctx, entry, 3, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle iovpr = emit_alloca(&L, i32, 1);
    MLIR_ValueHandle rempr = emit_alloca(&L, i32, 1);
    MLIR_ValueHandle totpr = emit_alloca(&L, i64, 1);
    MLIR_ValueHandle cvbuf = emit_alloca(&L, i8, 16);  // native ciovec {ptr,i64}
    MLIR_ValueHandle nrchunk = emit_alloca(&L, i64, 1); // per-chunk *nread
    emit_store_v(&L, piov, iovpr);
    emit_store_v(&L, plen, rempr);
    emit_store_v(&L, emit_const(&L, i64, 0), totpr);

    MLIR_BlockHandle loop = new_block(&L);
    MLIR_BlockHandle body = new_block(&L);
    MLIR_BlockHandle cont = new_block(&L);
    MLIR_BlockHandle done = new_block(&L);
    MLIR_BlockHandle err  = new_block(&L);
    term_br(&L, loop);

    L.cur = loop; L.terminated = false;
    MLIR_ValueHandle rem = emit_load_ty(&L, rempr, i32);
    MLIR_ValueHandle rz = build_icmp(&L, /*eq*/0, rem, emit_const(&L, i32, 0));
    term_cond_br(&L, rz, done, body);

    L.cur = body; L.terminated = false;
    MLIR_ValueHandle iov = emit_load_ty(&L, iovpr, i32);
    MLIR_ValueHandle bufofs = emit_load_ty(&L, linmem_ptr(&L, iov, 0), i32);
    MLIR_ValueHandle clen = emit_load_ty(&L, linmem_ptr(&L, iov, 4), i32);
    MLIR_ValueHandle host = linmem_ptr(&L, bufofs, 0);
    MLIR_ValueHandle clen64 = emit_cast(&L, OP_TYPE_LLVM_ZEXT, clen, i64);
    // Build a 1-element native iovec {host, clen64} and call platform_fd_read.
    // Returns 0 (with bytes in *nread) or errno.
    emit_store_v(&L, host, cvbuf);
    emit_store_v(&L, clen64, emit_byte_ptr(&L, cvbuf, emit_const(&L, i64, 8)));
    emit_store_v(&L, emit_const(&L, i64, 0), nrchunk);
    MLIR_ValueHandle rargs[4] = { pfd, cvbuf, emit_const(&L, i64, 1), nrchunk };
    MLIR_ValueHandle rc = emit_libc_call(&L, "__host_platform_fd_read", rargs, 4, i32);
    MLIR_ValueHandle iserr = build_icmp(&L, /*ne*/1, rc, emit_const(&L, i32, 0));
    term_cond_br(&L, iserr, err, cont);

    L.cur = cont; L.terminated = false;
    MLIR_ValueHandle n = emit_load_ty(&L, nrchunk, i64);
    MLIR_ValueHandle tot = emit_load_ty(&L, totpr, i64);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, tot, n, i64), totpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, iov, emit_const(&L, i32, 8), i32), iovpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_SUB, rem, emit_const(&L, i32, 1), i32), rempr);
    // Short read (n < requested): stop the whole call.
    MLIR_ValueHandle shortr = build_icmp(&L, /*slt*/2, n, clen64);
    term_cond_br(&L, shortr, done, loop);

    L.cur = done; L.terminated = false;
    MLIR_ValueHandle ftot = emit_load_ty(&L, totpr, i64);
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_TRUNC, ftot, i32), linmem_ptr(&L, pnw, 0));
    emit_ret_v(&L, emit_const(&L, i32, 0));

    L.cur = err; L.terminated = false;
    emit_ret_v(&L, emit_const(&L, i32, 8));

    finish_func(ctx, out_body, reg, "fd_read");
    free(L.frames);
}

// i32 fd_seek(i32 fd, i64 offset, i32 whence, i32 newoff): _lseek; store the
// resulting 64-bit offset to *newoff; return 0 (or 8 on error).
static void synth_fd_seek(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_ValueHandle pfd = mk_arg(ctx, entry, 0, i32);
    MLIR_ValueHandle poff = mk_arg(ctx, entry, 1, i64);
    MLIR_ValueHandle pwh = mk_arg(ctx, entry, 2, i32);
    MLIR_ValueHandle pno = mk_arg(ctx, entry, 3, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_TypeHandle i64p = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_ValueHandle nofs = emit_alloca(&L, i64p, 1);  // native uint64 newoffset
    MLIR_ValueHandle sargs[4] = { pfd, poff, pwh, nofs };
    MLIR_ValueHandle rc = emit_libc_call(&L, "__host_platform_fd_seek", sargs, 4, i32);
    MLIR_BlockHandle ok = new_block(&L);
    MLIR_BlockHandle err = new_block(&L);
    MLIR_ValueHandle iserr = build_icmp(&L, /*ne*/1, rc, emit_const(&L, i32, 0));
    term_cond_br(&L, iserr, err, ok);

    L.cur = ok; L.terminated = false;
    emit_store_v(&L, emit_load_ty(&L, nofs, i64), linmem_ptr(&L, pno, 0));
    emit_ret_v(&L, emit_const(&L, i32, 0));

    L.cur = err; L.terminated = false;
    emit_ret_v(&L, emit_const(&L, i32, 8));

    finish_func(ctx, out_body, reg, "fd_seek");
    free(L.frames);
}

// i32 fd_tell(fd, offset_ptr): write the current file position to *offset_ptr.
// POSIX: lseek(fd, 0, SEEK_CUR=1).
static void synth_fd_tell(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_ValueHandle pfd = mk_arg(ctx, entry, 0, i32);
    MLIR_ValueHandle pno = mk_arg(ctx, entry, 1, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle ofs = emit_alloca(&L, i64, 1);   // native uint64 offset
    MLIR_ValueHandle targs[2] = { pfd, ofs };
    MLIR_ValueHandle rc = emit_libc_call(&L, "__host_platform_fd_tell", targs, 2, i32);
    MLIR_BlockHandle ok = new_block(&L);
    MLIR_BlockHandle err = new_block(&L);
    MLIR_ValueHandle iserr = build_icmp(&L, /*ne*/1, rc, emit_const(&L, i32, 0));
    term_cond_br(&L, iserr, err, ok);

    L.cur = ok; L.terminated = false;
    emit_store_v(&L, emit_load_ty(&L, ofs, i64), linmem_ptr(&L, pno, 0));
    emit_ret_v(&L, emit_const(&L, i32, 0));

    L.cur = err; L.terminated = false;
    emit_ret_v(&L, emit_const(&L, i32, 8));

    finish_func(ctx, out_body, reg, "fd_tell");
    free(L.frames);
}
//               rights_inh:i64, fdflags, opened_fd): copy the path to a stack
// buffer, translate WASI flags to POSIX open() flags, _open, store the fd to
// *opened_fd. Ignores dirfd (relies on the process cwd matching the preopen).
static void synth_path_open(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    (void)mk_arg(ctx, entry, 0, i32);                 // dirfd (unused)
    (void)mk_arg(ctx, entry, 1, i32);                 // dirflags (unused)
    MLIR_ValueHandle ppath = mk_arg(ctx, entry, 2, i32);
    MLIR_ValueHandle plen  = mk_arg(ctx, entry, 3, i32);
    MLIR_ValueHandle poflags = mk_arg(ctx, entry, 4, i32);
    MLIR_ValueHandle prights = mk_arg(ctx, entry, 5, i64);
    (void)mk_arg(ctx, entry, 6, i64);                 // rights_inh (unused)
    MLIR_ValueHandle pfdflags = mk_arg(ctx, entry, 7, i32);
    MLIR_ValueHandle popened = mk_arg(ctx, entry, 8, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle buf = emit_alloca(&L, i8, 1056);
    MLIR_BlockHandle copy_init = new_block(&L);
    MLIR_BlockHandle err = new_block(&L);
    // Guard against overlong paths (buffer is 1056 incl. nul).
    MLIR_ValueHandle toolong = build_icmp(&L, /*uge*/9, plen, emit_const(&L, i32, 1055));
    term_cond_br(&L, toolong, err, copy_init);

    L.cur = copy_init; L.terminated = false;
    MLIR_ValueHandle ipr = emit_alloca(&L, i32, 1);
    emit_store_v(&L, emit_const(&L, i32, 0), ipr);
    MLIR_BlockHandle cploop = new_block(&L);
    MLIR_BlockHandle cpbody = new_block(&L);
    MLIR_BlockHandle cpdone = new_block(&L);
    term_br(&L, cploop);

    L.cur = cploop; L.terminated = false;
    MLIR_ValueHandle ci = emit_load_ty(&L, ipr, i32);
    MLIR_ValueHandle more = build_icmp(&L, /*ult*/6, ci, plen);
    term_cond_br(&L, more, cpbody, cpdone);

    L.cur = cpbody; L.terminated = false;
    MLIR_ValueHandle saddr = emit_binop2(&L, OP_TYPE_LLVM_ADD, ppath, ci, i32);
    MLIR_ValueHandle byte = emit_load_ty(&L, linmem_ptr(&L, saddr, 0), i8);
    MLIR_ValueHandle dst = emit_byte_ptr(&L, buf, emit_cast(&L, OP_TYPE_LLVM_ZEXT, ci, i64));
    emit_store_v(&L, byte, dst);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, ci, emit_const(&L, i32, 1), i32), ipr);
    term_br(&L, cploop);

    L.cur = cpdone; L.terminated = false;
    // nul terminator at buf[path_len].
    MLIR_ValueHandle nuldst = emit_byte_ptr(&L, buf,
        emit_cast(&L, OP_TYPE_LLVM_ZEXT, plen, i64));
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_TRUNC, emit_const(&L, i64, 0), i8), nuldst);

    // platform_path_open does the WASI rights/oflags -> POSIX translation
    // internally, so pass the WASI values straight through.
    MLIR_ValueHandle plen64 = emit_cast(&L, OP_TYPE_LLVM_ZEXT, plen, i64);
    MLIR_ValueHandle oargs[4] = { buf, plen64, prights, poflags };
    MLIR_ValueHandle fd = emit_libc_call(&L, "__host_platform_path_open", oargs, 4, i32);
    MLIR_BlockHandle ook = new_block(&L);
    MLIR_ValueHandle ofail = build_icmp(&L, /*slt*/2, fd, emit_const(&L, i32, 0));
    term_cond_br(&L, ofail, err, ook);

    L.cur = ook; L.terminated = false;
    emit_store_v(&L, fd, linmem_ptr(&L, popened, 0));
    // Darwin arm64: open()'s mode is a *variadic* argument and must be passed
    // on the stack, but tinyC's call ABI passes args in registers, so a freshly
    // created file ends up with a garbage mode (e.g. ---------x) — true inside
    // platform_path_open's own open() too. fchmod() is non-variadic, so fix the
    // mode explicitly to 0644 whenever O_CREAT (WASI oflags bit0) was requested.
    MLIR_ValueHandle created = build_icmp(&L, /*ne*/1,
        emit_binop2(&L, OP_TYPE_LLVM_AND, poflags, emit_const(&L, i32, 1), i32),
        emit_const(&L, i32, 0));
    MLIR_BlockHandle do_chmod = new_block(&L);
    MLIR_BlockHandle ret_ok = new_block(&L);
    term_cond_br(&L, created, do_chmod, ret_ok);

    L.cur = do_chmod; L.terminated = false;
    MLIR_ValueHandle cargs[2] = { fd, emit_const(&L, i32, 0644) };
    (void)emit_libc_call(&L, "_fchmod", cargs, 2, i32);
    term_br(&L, ret_ok);

    L.cur = ret_ok; L.terminated = false;
    emit_ret_v(&L, emit_const(&L, i32, 0));

    L.cur = err; L.terminated = false;
    emit_ret_v(&L, emit_const(&L, i32, 44));   // WASI ENOENT

    finish_func(ctx, out_body, reg, "path_open");
    free(L.frames);
}

// Load argc (i32) and the host argv base pointer (from @__wasm_argv).
static void load_argc_argv(FLower *L, MLIR_ValueHandle *argc_out,
                           MLIR_ValueHandle *argvp_out) {
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(L->ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(L->ctx, 64, true);
    *argc_out = emit_load_ty(L, emit_addressof(L, "__wasm_argc"), i32);
    *argvp_out = emit_i2p(L, emit_load_ty(L, emit_addressof(L, "__wasm_argv"), i64));
}

// host char* of argv[i] (a host pointer), read from the argv vector.
static MLIR_ValueHandle argv_str_ptr(FLower *L, MLIR_ValueHandle argvp,
                                     MLIR_ValueHandle i_i32) {
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(L->ctx, 64, true);
    MLIR_ValueHandle off = emit_binop2(L, OP_TYPE_LLVM_MUL,
        emit_cast(L, OP_TYPE_LLVM_ZEXT, i_i32, i64), emit_const(L, i64, 8), i64);
    MLIR_ValueHandle slot = emit_byte_ptr(L, argvp, off);
    return emit_i2p(L, emit_load_ty(L, slot, i64));
}

// i32 args_sizes_get(i32 out_argc, i32 out_buf_size): write argc and the total
// byte size of all argv strings (incl. nul terminators) into linmem.
static void synth_args_sizes_get(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_ValueHandle poutc = mk_arg(ctx, entry, 0, i32);
    MLIR_ValueHandle poutsz = mk_arg(ctx, entry, 1, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle argc, argvp;
    load_argc_argv(&L, &argc, &argvp);
    emit_store_v(&L, argc, linmem_ptr(&L, poutc, 0));
    MLIR_ValueHandle totpr = emit_alloca(&L, i64, 1);
    MLIR_ValueHandle ipr = emit_alloca(&L, i32, 1);
    MLIR_ValueHandle sppr = emit_alloca(&L, i64, 1);
    emit_store_v(&L, emit_const(&L, i64, 0), totpr);
    emit_store_v(&L, emit_const(&L, i32, 0), ipr);

    MLIR_BlockHandle outer = new_block(&L);
    MLIR_BlockHandle scan = new_block(&L);
    MLIR_BlockHandle walk = new_block(&L);
    MLIR_BlockHandle wcont = new_block(&L);
    MLIR_BlockHandle nextarg = new_block(&L);
    MLIR_BlockHandle done = new_block(&L);
    term_br(&L, outer);

    L.cur = outer; L.terminated = false;
    MLIR_ValueHandle i = emit_load_ty(&L, ipr, i32);
    MLIR_ValueHandle more = build_icmp(&L, /*ult*/6, i, argc);
    term_cond_br(&L, more, scan, done);

    L.cur = scan; L.terminated = false;
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_PTRTOINT,
        argv_str_ptr(&L, argvp, i), i64), sppr);
    term_br(&L, walk);

    L.cur = walk; L.terminated = false;
    MLIR_ValueHandle sp = emit_load_ty(&L, sppr, i64);
    MLIR_ValueHandle ch = emit_load_ty(&L, emit_i2p(&L, sp), i8);
    MLIR_ValueHandle isnul = build_icmp(&L, /*eq*/0, ch,
        emit_cast(&L, OP_TYPE_LLVM_TRUNC, emit_const(&L, i64, 0), i8));
    term_cond_br(&L, isnul, nextarg, wcont);

    L.cur = wcont; L.terminated = false;
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD,
        emit_load_ty(&L, totpr, i64), emit_const(&L, i64, 1), i64), totpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, sp, emit_const(&L, i64, 1), i64), sppr);
    term_br(&L, walk);

    L.cur = nextarg; L.terminated = false;
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD,
        emit_load_ty(&L, totpr, i64), emit_const(&L, i64, 1), i64), totpr);  // nul
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD,
        emit_load_ty(&L, ipr, i32), emit_const(&L, i32, 1), i32), ipr);
    term_br(&L, outer);

    L.cur = done; L.terminated = false;
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_TRUNC,
        emit_load_ty(&L, totpr, i64), i32), linmem_ptr(&L, poutsz, 0));
    emit_ret_v(&L, emit_const(&L, i32, 0));

    finish_func(ctx, out_body, reg, "args_sizes_get");
    free(L.frames);
}

// i32 args_get(i32 argv_ofs_arr, i32 argv_buf_ofs): fill argv_ofs_arr[i] with
// the linmem offset of argv[i] and copy each string (incl. nul) into linmem.
static void synth_args_get(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_ValueHandle parr = mk_arg(ctx, entry, 0, i32);
    MLIR_ValueHandle pbuf = mk_arg(ctx, entry, 1, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle argc, argvp;
    load_argc_argv(&L, &argc, &argvp);
    MLIR_ValueHandle curpr = emit_alloca(&L, i32, 1);  // current linmem write offset
    MLIR_ValueHandle ipr = emit_alloca(&L, i32, 1);
    MLIR_ValueHandle sppr = emit_alloca(&L, i64, 1);
    emit_store_v(&L, pbuf, curpr);
    emit_store_v(&L, emit_const(&L, i32, 0), ipr);

    MLIR_BlockHandle outer = new_block(&L);
    MLIR_BlockHandle setup = new_block(&L);
    MLIR_BlockHandle copy = new_block(&L);
    MLIR_BlockHandle nextarg = new_block(&L);
    MLIR_BlockHandle done = new_block(&L);
    term_br(&L, outer);

    L.cur = outer; L.terminated = false;
    MLIR_ValueHandle i = emit_load_ty(&L, ipr, i32);
    MLIR_ValueHandle more = build_icmp(&L, /*ult*/6, i, argc);
    term_cond_br(&L, more, setup, done);

    L.cur = setup; L.terminated = false;
    // argv_ofs_arr[i] = cur.
    MLIR_ValueHandle slot = emit_binop2(&L, OP_TYPE_LLVM_ADD, parr,
        emit_binop2(&L, OP_TYPE_LLVM_MUL, i, emit_const(&L, i32, 4), i32), i32);
    emit_store_v(&L, emit_load_ty(&L, curpr, i32), linmem_ptr(&L, slot, 0));
    emit_store_v(&L, emit_cast(&L, OP_TYPE_LLVM_PTRTOINT,
        argv_str_ptr(&L, argvp, i), i64), sppr);
    term_br(&L, copy);

    L.cur = copy; L.terminated = false;
    MLIR_ValueHandle sp = emit_load_ty(&L, sppr, i64);
    MLIR_ValueHandle ch = emit_load_ty(&L, emit_i2p(&L, sp), i8);
    MLIR_ValueHandle cur = emit_load_ty(&L, curpr, i32);
    emit_store_v(&L, ch, linmem_ptr(&L, cur, 0));
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, cur, emit_const(&L, i32, 1), i32), curpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, sp, emit_const(&L, i64, 1), i64), sppr);
    MLIR_ValueHandle isnul = build_icmp(&L, /*eq*/0, ch,
        emit_cast(&L, OP_TYPE_LLVM_TRUNC, emit_const(&L, i64, 0), i8));
    term_cond_br(&L, isnul, nextarg, copy);

    L.cur = nextarg; L.terminated = false;
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, i, emit_const(&L, i32, 1), i32), ipr);
    term_br(&L, outer);

    L.cur = done; L.terminated = false;
    emit_ret_v(&L, emit_const(&L, i32, 0));

    finish_func(ctx, out_body, reg, "args_get");
    free(L.frames);
}

// i32 environ_sizes_get(i32 out_count, i32 out_buf_size): no env vars.
static void synth_environ_sizes_get(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_ValueHandle pc = mk_arg(ctx, entry, 0, i32);
    MLIR_ValueHandle pb = mk_arg(ctx, entry, 1, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;
    emit_store_v(&L, emit_const(&L, i32, 0), linmem_ptr(&L, pc, 0));
    emit_store_v(&L, emit_const(&L, i32, 0), linmem_ptr(&L, pb, 0));
    emit_ret_v(&L, emit_const(&L, i32, 0));
    finish_func(ctx, out_body, reg, "environ_sizes_get");
}

// i32 environ_get(i32 environ, i32 buf): no env vars.
static void synth_environ_get(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);
    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    (void)mk_arg(ctx, entry, 0, i32);
    (void)mk_arg(ctx, entry, 1, i32);
    FLower L = {0}; L.ctx = ctx; L.dreg = reg; L.cur = entry;
    emit_ret_v(&L, emit_const(&L, i32, 0));
    finish_func(ctx, out_body, reg, "environ_get");
}

// tinyc_va_arg_i32/ptr(i32 ap) -> i32  and  tinyc_va_arg_i64/f64(i32 ap) -> i64
// `ap` is a wasm offset to a 4-byte cell holding the current va_list cursor
// (itself a wasm offset to the next arg in linmem). Read the value at the
// cursor (8-byte-aligning the cursor first for the 64-bit forms), then advance
// the stored cursor past it.
static void synth_va_arg_scalar(MLIR_Context *ctx, MLIR_BlockHandle out_body,
                                const char *name, size_t namelen, bool is64) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);

    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_TypeHandle vty = is64 ? i64 : i32;

    MLIR_ValueHandle ap = MLIR_CreateValueBlockArg(ctx, (string){0}, 0, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockArg(ctx, entry, ap);

    FLower L = {0};
    L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle app = linmem_ptr(&L, ap, 0);
    MLIR_ValueHandle cur = emit_load_ty(&L, app, i32);
    if (is64) {
        cur = emit_binop2(&L, OP_TYPE_LLVM_ADD, cur, emit_const(&L, i32, 7), i32);
        cur = emit_binop2(&L, OP_TYPE_LLVM_AND, cur, emit_const(&L, i32, -8), i32);
    }
    MLIR_ValueHandle vp  = linmem_ptr(&L, cur, 0);
    MLIR_ValueHandle res = emit_load_ty(&L, vp, vty);
    MLIR_ValueHandle nxt = emit_binop2(&L, OP_TYPE_LLVM_ADD, cur,
        emit_const(&L, i32, is64 ? 8 : 4), i32);
    emit_store_v(&L, nxt, app);

    MLIR_ValueHandle rops[1] = { res };
    MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, rops, 1);
    emit(&L, ret);

    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)name, namelen) };
    MLIR_RegionHandle regs[1] = { reg };
    MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
        op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}), MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, out_body, fn);
    free(L.frames);
}

// tinyc_va_arg_struct(i32 ap, i32 out, i64 size) -> void: copy `(size+7)/8`
// 8-byte words from the (8-byte-aligned) va_list cursor into linmem[out],
// advancing the stored cursor. Mirrors synth_tinyc_va_arg_struct.
static void synth_va_arg_struct(MLIR_Context *ctx, MLIR_BlockHandle out_body) {
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, reg, entry);

    MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);

    MLIR_ValueHandle ap = MLIR_CreateValueBlockArg(ctx, (string){0}, 0, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_ValueHandle outo = MLIR_CreateValueBlockArg(ctx, (string){0}, 1, i32,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_ValueHandle size = MLIR_CreateValueBlockArg(ctx, (string){0}, 2, i64,
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_AppendBlockArg(ctx, entry, ap);
    MLIR_AppendBlockArg(ctx, entry, outo);
    MLIR_AppendBlockArg(ctx, entry, size);

    FLower L = {0};
    L.ctx = ctx; L.dreg = reg; L.cur = entry;

    MLIR_ValueHandle app  = linmem_ptr(&L, ap, 0);
    MLIR_ValueHandle words = emit_binop2(&L, OP_TYPE_LLVM_LSHR,
        emit_binop2(&L, OP_TYPE_LLVM_ADD, size, emit_const(&L, i64, 7), i64),
        emit_const(&L, i64, 3), i64);
    MLIR_ValueHandle curpr = emit_alloca(&L, i32, 1);
    MLIR_ValueHandle outpr = emit_alloca(&L, i32, 1);
    MLIR_ValueHandle wpr   = emit_alloca(&L, i64, 1);
    emit_store_v(&L, emit_load_ty(&L, app, i32), curpr);
    emit_store_v(&L, outo, outpr);
    emit_store_v(&L, words, wpr);

    MLIR_BlockHandle loop = new_block(&L);
    MLIR_BlockHandle body = new_block(&L);
    MLIR_BlockHandle done = new_block(&L);
    term_br(&L, loop);

    // loop: if words==0 goto done
    L.cur = loop; L.terminated = false;
    MLIR_ValueHandle w = emit_load_ty(&L, wpr, i64);
    MLIR_ValueHandle wz = build_icmp(&L, /*eq*/0, w, emit_const(&L, i64, 0));
    term_cond_br(&L, wz, done, body);

    // body: align cur to 8, copy one word, advance all cursors
    L.cur = body; L.terminated = false;
    MLIR_ValueHandle cur = emit_load_ty(&L, curpr, i32);
    cur = emit_binop2(&L, OP_TYPE_LLVM_ADD, cur, emit_const(&L, i32, 7), i32);
    cur = emit_binop2(&L, OP_TYPE_LLVM_AND, cur, emit_const(&L, i32, -8), i32);
    MLIR_ValueHandle o = emit_load_ty(&L, outpr, i32);
    MLIR_ValueHandle srcp = linmem_ptr(&L, cur, 0);
    MLIR_ValueHandle dstp = linmem_ptr(&L, o, 0);
    emit_store_v(&L, emit_load_ty(&L, srcp, i64), dstp);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, cur, emit_const(&L, i32, 8), i32), curpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_ADD, o, emit_const(&L, i32, 8), i32), outpr);
    emit_store_v(&L, emit_binop2(&L, OP_TYPE_LLVM_SUB, w, emit_const(&L, i64, 1), i64), wpr);
    term_br(&L, loop);

    // done: write the final cursor back to linmem[ap], return void
    L.cur = done; L.terminated = false;
    emit_store_v(&L, emit_load_ty(&L, curpr, i32), app);
    MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN, NULL, 0, NULL, 0, NULL, NULL, 0);
    emit(&L, ret);

    MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", (char *)"tinyc_va_arg_struct", 19) };
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

// Recursively intern every wasmssa.func_addr target into the fnptr table,
// honouring an explicit `slot` attr (the wasm table index).
static void scan_func_addrs(MLIR_OpHandle op, FuncPtrMap *fnptrs) {
    if (MLIR_GetOpType(op) == OP_TYPE_WASMSSA_FUNC_ADDR) {
        string tgt = at_s(op, "target");
        int32_t es = (int32_t)at_i_or(op, "slot", -1);
        fpm_intern_with_slot(fnptrs, tgt, es);
    }
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(op, r);
        size_t nb = MLIR_GetRegionNumBlocks(reg);
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(blk);
            for (size_t i = 0; i < no; i++)
                scan_func_addrs(MLIR_GetBlockOp(blk, i), fnptrs);
        }
    }
}

// Map a wasm valtype byte to its llvm carrier integer type (i64 for i64/f64,
// i32 otherwise).
static MLIR_TypeHandle carrier_for_byte(MLIR_Context *ctx, unsigned b) {
    if (b == 0x7e || b == 0x7c)
        return MLIR_CreateTypeInteger(ctx, 64, true);
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

// Synthesise one __dispatch_<pt>_<rt> llvm.func per signature referenced by a
// func_addr target. The dispatcher takes (slot:i32, args...) and switches on
// slot to directly call the matching addressed function.
static bool synth_dispatchers(MLIR_Context *ctx, MLIR_BlockHandle out_body,
                              FuncPtrMap *fnptrs, FuncSigMap *sigs) {
    for (size_t pi = 0; pi < fnptrs->n; pi++) {
        string pname = fnptrs->names[pi];
        string ppt, prt;
        if (!fsm_get(sigs, pname, &ppt, &prt)) continue;

        // Skip if an earlier target already produced this signature's dispatcher.
        bool dup = false;
        for (size_t qi = 0; qi < pi; qi++) {
            string qpt, qrt;
            if (!fsm_get(sigs, fnptrs->names[qi], &qpt, &qrt)) continue;
            if (qpt.size == ppt.size && qrt.size == prt.size &&
                memcmp(qpt.str, ppt.str, ppt.size) == 0 &&
                memcmp(qrt.str, prt.str, prt.size) == 0) { dup = true; break; }
        }
        if (dup) continue;

        size_t dname_cap = ppt.size + prt.size + 32;
        char *dname = (char *)malloc(dname_cap);
        int dlen = snprintf(dname, dname_cap, "__dispatch_%.*s_%.*s",
            (int)ppt.size, ppt.str, (int)prt.size, prt.str);
        if (dlen <= 0 || (size_t)dlen >= dname_cap) { free(dname); return false; }

        size_t n_pp = ppt.size / 2;
        if (n_pp > 15) {
            fprintf(stderr, "wasmssa->llvm: dispatcher signature too long\n");
            free(dname); return false;
        }
        bool has_result = (prt.size >= 2);

        MLIR_RegionHandle dreg = MLIR_CreateRegion(ctx);
        MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, dreg, entry);

        FLower L = {0};
        L.ctx = ctx; L.dreg = dreg; L.cur = entry;

        // Entry block args: slot (i32) + params.
        size_t n_dp = n_pp + 1;
        MLIR_ValueHandle pvals[16];
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        pvals[0] = MLIR_CreateValueBlockArg(ctx, (string){0}, 0, i32,
            MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, entry, pvals[0]);
        for (size_t k = 0; k < n_pp; k++) {
            MLIR_TypeHandle ty = carrier_for_byte(ctx, type_byte_at(ppt, k));
            pvals[k + 1] = MLIR_CreateValueBlockArg(ctx, (string){0},
                (uint32_t)(k + 1), ty, MLIR_CreateLocationUnknown(ctx, (string){0}));
            MLIR_AppendBlockArg(ctx, entry, pvals[k + 1]);
        }
        MLIR_TypeHandle rty = has_result
            ? carrier_for_byte(ctx, type_byte_at(prt, 0)) : MLIR_INVALID_HANDLE;

        // Collect matching addressed funcs, in slot order.
        for (size_t qi = 0; qi < fnptrs->n; qi++) {
            string qpt, qrt;
            if (!fsm_get(sigs, fnptrs->names[qi], &qpt, &qrt)) continue;
            if (qpt.size != ppt.size || qrt.size != prt.size) continue;
            if (memcmp(qpt.str, ppt.str, ppt.size) != 0) continue;
            if (memcmp(qrt.str, prt.str, prt.size) != 0) continue;

            int32_t target_slot = fnptrs->slots[qi];
            string  target_name = fnptrs->names[qi];

            MLIR_ValueHandle sc = emit_const(&L, i32, target_slot);
            MLIR_ValueHandle eq = build_icmp(&L, /*eq*/0, pvals[0], sc);
            MLIR_BlockHandle call_blk = new_block(&L);
            MLIR_BlockHandle next_blk = new_block(&L);
            term_cond_br(&L, eq, call_blk, next_blk);

            // call_blk: r = call target(args...); return [r]
            L.cur = call_blk; L.terminated = false;
            MLIR_AttributeHandle ca[1] = {
                attr_s(ctx, "callee", target_name.str, target_name.size)
            };
            MLIR_TypeHandle crt[1]; MLIR_ValueHandle cr[1];
            if (has_result) { crt[0] = rty; cr[0] = mk_res(ctx, rty); }
            MLIR_OpHandle call = build_op(ctx, OP_TYPE_LLVM_CALL,
                ca, 1, crt, has_result ? 1 : 0, cr, &pvals[1], n_dp - 1);
            emit(&L, call);
            MLIR_ValueHandle rops[1];
            if (has_result) rops[0] = cr[0];
            MLIR_OpHandle ret = build_op(ctx, OP_TYPE_LLVM_RETURN,
                NULL, 0, NULL, 0, NULL, has_result ? rops : NULL, has_result ? 1 : 0);
            emit(&L, ret);

            L.cur = next_blk; L.terminated = false;
        }
        // Fall-through (no slot matched): dead self-branch terminator.
        term_br(&L, L.cur);

        MLIR_AttributeHandle fa[1] = { attr_s(ctx, "sym_name", dname, (size_t)dlen) };
        MLIR_RegionHandle regs[1] = { dreg };
        MLIR_OpHandle fn = MLIR_CreateOp(ctx, OP_TYPE_LLVM_FUNC,
            op_type_to_string(OP_TYPE_LLVM_FUNC), fa, 1, NULL, 0, NULL, 0, NULL, 0,
            regs, 1, MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, out_body, fn);
        free(L.frames);
        // dname must outlive the attr (attr_s does not copy); leaked.
    }
    return true;
}

MLIR_OpHandle mlir_wasmssa_to_llvm(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {    MLIR_RegionHandle mr = MLIR_GetOpRegion(ssa_module, 0);
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

    // -- Emit @__wasm_linmem_base / @__wasm_argc / @__wasm_argv. --
    // `_start` (backend) mmaps the 4 GiB linear memory and stores its base
    // here; it also stashes the process argc (i32) and argv (host char**, i64).
    MLIR_TypeHandle i64t = MLIR_CreateTypeInteger(ctx, 64, true);
    MLIR_TypeHandle i32g = MLIR_CreateTypeInteger(ctx, 32, true);
    MLIR_AppendBlockOp(ctx, out_body, MLIR_CreateLLVMGlobal(ctx,
        str_from_cstr_view((char *)LINMEM_BASE_GLOBAL), i64t, false, 0, 0, 0.0, NULL,
        MLIR_CreateLocationUnknown(ctx, (string){0})));
    MLIR_AppendBlockOp(ctx, out_body, MLIR_CreateLLVMGlobal(ctx,
        str_from_cstr_view((char *)"__wasm_argc"), i32g, false, 0, 0, 0.0, NULL,
        MLIR_CreateLocationUnknown(ctx, (string){0})));
    MLIR_AppendBlockOp(ctx, out_body, MLIR_CreateLLVMGlobal(ctx,
        str_from_cstr_view((char *)"__wasm_argv"), i64t, false, 0, 0, 0.0, NULL,
        MLIR_CreateLocationUnknown(ctx, (string){0})));

    // -- Emit wasm globals @__wasm_g0..gN (g0 = shadow stack pointer). --
    int64_t max_global = -1;
    for (size_t i = 0; i < nops; i++)
        scan_max_global(MLIR_GetBlockOp(mb, i), &max_global);
    // g0 (the shadow stack pointer) must start at wasm-ld's __stack_pointer
    // (the top of the stack region; the heap grows up from there). Use the
    // value propagated from the wasm GLOBAL section. Initialising it to the
    // top of linear memory instead would make the stack grow down INTO the
    // heap and clobber early heap allocations.
    int64_t g0_init = linmem_total;
    MLIR_AttributeHandle a_g0 = MLIR_GetOpAttributeByName(ssa_module, "global0_init");
    if (a_g0) g0_init = MLIR_GetAttributeInteger(a_g0);
    MLIR_TypeHandle i32t = MLIR_CreateTypeInteger(ctx, 32, true);
    for (int64_t g = 0; g <= max_global; g++) {
        char gname[32];
        snprintf(gname, sizeof(gname), "__wasm_g%lld", (long long)g);
        int64_t init = (g == 0) ? g0_init : 0;
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
    bool need_fd_write = false;
    bool need_fd_read = false, need_fd_seek = false, need_fd_close = false;
    bool need_fd_tell = false;
    bool need_path_open = false, need_args_get = false, need_args_sizes = false;
    bool need_environ_get = false, need_environ_sizes = false;
    bool need_va_i32 = false, need_va_i64 = false, need_va_ptr = false;
    bool need_va_f64 = false, need_va_struct = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSSA_IMPORT_FUNC) continue;
        string nm = at_s(op, "sym_name");
        if (nm.size == 9 && memcmp(nm.str, "proc_exit", 9) == 0) continue;
        if (nm.size == 8 && memcmp(nm.str, "printI64", 8) == 0) { need_printI64 = true; continue; }
        if (nm.size == 12 && memcmp(nm.str, "printNewline", 12) == 0) { need_printNewline = true; continue; }
        if (nm.size == 8 && memcmp(nm.str, "printStr", 8) == 0) { need_printStr = true; continue; }
        if (nm.size == 8 && memcmp(nm.str, "fd_write", 8) == 0) { need_fd_write = true; continue; }
        if (nm.size == 7 && memcmp(nm.str, "fd_read", 7) == 0) { need_fd_read = true; continue; }
        if (nm.size == 7 && memcmp(nm.str, "fd_seek", 7) == 0) { need_fd_seek = true; continue; }
        if (nm.size == 7 && memcmp(nm.str, "fd_tell", 7) == 0) { need_fd_tell = true; continue; }
        if (nm.size == 8 && memcmp(nm.str, "fd_close", 8) == 0) { need_fd_close = true; continue; }
        if (nm.size == 9 && memcmp(nm.str, "path_open", 9) == 0) { need_path_open = true; continue; }
        if (nm.size == 8 && memcmp(nm.str, "args_get", 8) == 0) { need_args_get = true; continue; }
        if (nm.size == 14 && memcmp(nm.str, "args_sizes_get", 14) == 0) { need_args_sizes = true; continue; }
        if (nm.size == 11 && memcmp(nm.str, "environ_get", 11) == 0) { need_environ_get = true; continue; }
        if (nm.size == 17 && memcmp(nm.str, "environ_sizes_get", 17) == 0) { need_environ_sizes = true; continue; }
        if (nm.size == 16 && memcmp(nm.str, "tinyc_va_arg_i32", 16) == 0) { need_va_i32 = true; continue; }
        if (nm.size == 16 && memcmp(nm.str, "tinyc_va_arg_i64", 16) == 0) { need_va_i64 = true; continue; }
        if (nm.size == 16 && memcmp(nm.str, "tinyc_va_arg_ptr", 16) == 0) { need_va_ptr = true; continue; }
        if (nm.size == 16 && memcmp(nm.str, "tinyc_va_arg_f64", 16) == 0) { need_va_f64 = true; continue; }
        if (nm.size == 19 && memcmp(nm.str, "tinyc_va_arg_struct", 19) == 0) { need_va_struct = true; continue; }
        fprintf(stderr,
            "wasmssa->llvm: import '%.*s' not yet supported\n",
            (int)nm.size, nm.str);
        free(image);
        free(globals.names); free(globals.offsets);
        return MLIR_INVALID_HANDLE;
    }

    // -- Build signature + fnptr-slot maps (pre-pass for func pointers). --
    FuncSigMap sigs = {0};
    FuncPtrMap fnptrs = {0};
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t == OP_TYPE_WASMSSA_FUNC || t == OP_TYPE_WASMSSA_IMPORT_FUNC) {
            string fnm = at_s(top, "sym_name");
            if (t == OP_TYPE_WASMSSA_FUNC && at_b(top, "exported"))
                fnm = str_lit("main");
            fsm_add(&sigs, fnm, at_s(top, "param_types"), at_s(top, "result_types"));
        }
    }
    for (size_t i = 0; i < nops; i++)
        scan_func_addrs(MLIR_GetBlockOp(mb, i), &fnptrs);

    // -- Lower functions. --
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSSA_FUNC) continue;
        MLIR_OpHandle fn = lower_func(ctx, op, &globals, &fnptrs, &sigs);
        if (fn == MLIR_INVALID_HANDLE) {
            free(image);
            free(globals.names); free(globals.offsets);
            free(fnptrs.names); free(fnptrs.slots);
            free(sigs.names); free(sigs.param_types); free(sigs.result_types);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AppendBlockOp(ctx, out_body, fn);
    }

    // -- Synthesise call_indirect dispatchers. --
    if (!synth_dispatchers(ctx, out_body, &fnptrs, &sigs)) {
        free(image);
        free(globals.names); free(globals.offsets);
        free(fnptrs.names); free(fnptrs.slots);
        free(sigs.names); free(sigs.param_types); free(sigs.result_types);
        return MLIR_INVALID_HANDLE;
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
    if (need_fd_write) synth_fd_write(ctx, out_body);
    if (need_fd_read) synth_fd_read(ctx, out_body);
    if (need_fd_seek) synth_fd_seek(ctx, out_body);
    if (need_fd_tell) synth_fd_tell(ctx, out_body);
    if (need_fd_close) synth_fd_close(ctx, out_body);
    if (need_path_open) synth_path_open(ctx, out_body);
    if (need_args_get) synth_args_get(ctx, out_body);
    if (need_args_sizes) synth_args_sizes_get(ctx, out_body);
    if (need_environ_get) synth_environ_get(ctx, out_body);
    if (need_environ_sizes) synth_environ_sizes_get(ctx, out_body);
    if (need_va_i32) synth_va_arg_scalar(ctx, out_body, "tinyc_va_arg_i32", 16, false);
    if (need_va_ptr) synth_va_arg_scalar(ctx, out_body, "tinyc_va_arg_ptr", 16, false);
    if (need_va_i64) synth_va_arg_scalar(ctx, out_body, "tinyc_va_arg_i64", 16, true);
    if (need_va_f64) synth_va_arg_scalar(ctx, out_body, "tinyc_va_arg_f64", 16, true);
    if (need_va_struct) synth_va_arg_struct(ctx, out_body);

    free(globals.names); free(globals.offsets);
    free(fnptrs.names); free(fnptrs.slots);
    free(sigs.names); free(sigs.param_types); free(sigs.result_types);
    free(image);

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

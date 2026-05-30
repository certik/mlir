// wasmssa -> wmir lowering. See mlir_wasmssa_to_wmir.h for the public
// API and rationale.
//
// This pass flattens wasmssa's structured control flow (block / loop /
// if regions) into a CFG of wmir blocks connected by `wmir.br` and
// `wmir.cond_br`. Loop / block targets are tracked through a small
// frame stack so wasmssa.br with a `depth=N` attribute resolves to
// the right successor.
//
// Per-op handlers either:
//   * emit a single side-effect-free wmir op into the current block
//     (most arithmetic / memory ops), or
//   * unpack a region-bearing wasmssa op into fresh wmir blocks and
//     leave the Lowerer's `cur` pointed at the post-merge / post-loop
//     block where subsequent ops should go.
//
// Imports (`wasmssa.import_func`) are dropped: this slice of the
// pipeline never targets them. They'll come back as `wmir.import_func`
// declarations when a test needs an actual external call.

#include "mlir_wasmssa_to_wmir.h"
#include "mlir_wmir_mem2reg.h"

#include <stdarg.h>
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

// =============================================================================
// Wasm value-type byte constants (mirrors the WT_* table used elsewhere
// in the pipeline).
// =============================================================================
#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// Default linear-memory data-segment base offset (matches wasm-ld's
// default --global-base of 1024). Used when laying out import_global
// data into the wmir linmem image. Defined as a #define (rather than
// a local `enum`) so the file compiles cleanly under tinyc, which
// only supports enum bodies at module scope.
#define WASM_DATA_BASE 1024

// =============================================================================
// Attribute / op helpers.
// =============================================================================
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}

static MLIR_OpHandle build_op_simple(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_TypeHandle *result_types, size_t n_results,
                                     MLIR_ValueHandle *results,
                                     MLIR_ValueHandle *operands, size_t n_operands) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, result_types, n_results, results, n_results,
        operands, n_operands, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static MLIR_OpHandle build_op_br(MLIR_Context *ctx,
                                 MLIR_BlockHandle target,
                                 MLIR_ValueHandle *args, size_t n_args) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *succ_ops[1] = { args };
    size_t           n_succ_ops[1] = { n_args };
    return MLIR_CreateOpWithSuccessors(ctx, OP_TYPE_WMIR_BR,
        op_type_to_string(OP_TYPE_WMIR_BR),
        NULL, 0, NULL, 0, NULL, 0,
        NULL, 0, NULL, 0,
        succs, 1, succ_ops, n_succ_ops,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static MLIR_OpHandle build_op_cond_br(MLIR_Context *ctx,
                                      MLIR_ValueHandle cond,
                                      MLIR_BlockHandle t_blk,
                                      MLIR_BlockHandle f_blk) {
    MLIR_BlockHandle succs[2] = { t_blk, f_blk };
    MLIR_ValueHandle *succ_ops[2] = { NULL, NULL };
    size_t           n_succ_ops[2] = { 0, 0 };
    MLIR_ValueHandle cond_arr[1] = { cond };
    return MLIR_CreateOpWithSuccessors(ctx, OP_TYPE_WMIR_COND_BR,
        op_type_to_string(OP_TYPE_WMIR_COND_BR),
        NULL, 0, NULL, 0, NULL, 0,
        cond_arr, 1, NULL, 0,
        succs, 2, succ_ops, n_succ_ops,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
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
// Returns an int attribute value or `dflt` if the attribute is absent.
static int64_t at_i_or(MLIR_OpHandle op, const char *name, int64_t dflt) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : dflt;
}

// Translate a wasm value-type byte to the matching MLIR integer / float
// type used inside wmir. *Note*: f32 / f64 wasm values are carried by
// i32 / i64 bit-pattern wmir values throughout the pipeline. The actual
// float-ness is recorded per-op (e.g. via the `fwidth` attribute on
// wmir.fbinop). This lets the existing i32/i64-aware spill/reload
// machinery in wmir->aarch64 just work; FP arithmetic detours through
// a V register only for the duration of the FP instruction.
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_F64: return MLIR_CreateTypeInteger(ctx, 64, true);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

// Width of a wasm value type in bytes (4 or 8). Used to pick the
// integer carrier width for f32/f64.
static uint32_t vt_width(uint8_t vt) {
    return (vt == WT_I64 || vt == WT_F64) ? 8 : 4;
}

// Normalise an MLIR type to its integer carrier: f32/f64 collapse to
// i32/i64 so wmir block args, params, and result types only ever see
// integer types. Integer / index / function / pointer types pass
// through unchanged.
static MLIR_TypeHandle normalise_carrier(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (!ty) return ty;
    string ts = MLIR_GetTypeString(ctx, ty);
    if (ts.size == 3 && memcmp(ts.str, "f32", 3) == 0)
        return MLIR_CreateTypeInteger(ctx, 32, true);
    if (ts.size == 3 && memcmp(ts.str, "f64", 3) == 0)
        return MLIR_CreateTypeInteger(ctx, 64, true);
    // Re-create integer carriers in the *current* context arena rather than
    // returning the input handle. The input type lives in the wasmssa
    // module's arena, which the wasm->macho pipeline frees right after this
    // pass; handing that handle to a wmir value would leave it dangling. A
    // fresh CreateTypeInteger interns into the current (wmir) arena so the
    // produced module is self-contained. Integer types print as "i{width}".
    if (ts.size >= 2 && ts.str[0] == 'i') {
        int64_t w = 0;
        bool ok = true;
        for (size_t i = 1; i < ts.size; i++) {
            if (ts.str[i] < '0' || ts.str[i] > '9') { ok = false; break; }
            w = w * 10 + (ts.str[i] - '0');
        }
        if (ok && w > 0) return MLIR_CreateTypeInteger(ctx, (uint32_t)w, true);
    }
    return ty;
}

// =============================================================================
// SSA value remapping: each wasmssa value maps to its replacement in
// the wmir module. Open-addressing hash table (linear probing) keyed on
// the source handle, so per-op operand lookups stay O(1). A linear scan
// is O(n) per lookup and O(n^2) over a block; self-host functions have
// thousands of values, so the hash table is required to keep the
// wasmssa->wmir pass roughly linear.
//
// The empty-slot sentinel is MLIR_INVALID_HANDLE (0), which is never a
// valid source value (block args and op results are always non-zero).
// =============================================================================
typedef struct {
    MLIR_ValueHandle *src;   // hash slots; 0 == empty
    MLIR_ValueHandle *dst;
    size_t            n, cap; // cap is always a power of two (or 0)
} VMap;

static size_t vmap_hash(MLIR_ValueHandle k) {
    // 32-bit-safe integer mix (no 64-bit literals): works identically on
    // the wasm32 self-host (uintptr_t == 32-bit) and the 64-bit host. We
    // only ever look up by key, never iterate, so bucket order never
    // affects the emitted module -- the self-host stays deterministic.
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}

static void vmap_insert_raw(MLIR_ValueHandle *src, MLIR_ValueHandle *dst,
                            size_t cap, MLIR_ValueHandle k,
                            MLIR_ValueHandle v) {
    size_t mask = cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (src[i] != MLIR_INVALID_HANDLE && src[i] != k) {
        i = (i + 1) & mask;
    }
    src[i] = k;
    dst[i] = v;
}

static void vmap_grow(VMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    MLIR_ValueHandle *nsrc =
        (MLIR_ValueHandle *)calloc(ncap, sizeof(*nsrc));
    MLIR_ValueHandle *ndst =
        (MLIR_ValueHandle *)malloc(ncap * sizeof(*ndst));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->src[i] != MLIR_INVALID_HANDLE) {
            vmap_insert_raw(nsrc, ndst, ncap, m->src[i], m->dst[i]);
        }
    }
    free(m->src);
    free(m->dst);
    m->src = nsrc;
    m->dst = ndst;
    m->cap = ncap;
}

static void vmap_set(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    // Grow at 75% load factor to keep probe chains short.
    if ((m->n + 1) * 4 >= m->cap * 3) {
        vmap_grow(m);
    }
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->src[i] != MLIR_INVALID_HANDLE && m->src[i] != k) {
        i = (i + 1) & mask;
    }
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
// Frame stack for resolving wasmssa.br {depth=N}. Each enclosing
// block / loop / if pushes one frame; depth=0 refers to the innermost.
//
// Two targets must be tracked separately:
//   * `br_target`   — destination of `wasmssa.br` / `wasmssa.br_if` at this
//                     depth. For block/if it is the merge block; for loop
//                     it is the loop HEADER (so `br depth=0` continues
//                     the loop, per wasm spec).
//   * `fall_target` — destination of `wasmssa.block_return` (the implicit
//                     fall-off-end terminator emitted by the lifter at
//                     every region's end). For block/if it is the merge
//                     block (same as br_target); for loop it is the
//                     POST-LOOP exit block, so falling off the end of a
//                     loop exits the loop instead of looping forever.
//
// `n_args` records the number of operands on the br/br_if path (block
// args of br_target) so the lowering can sanity-check operand counts.
// `n_fall_args` records the number of operands on the fall-off path
// (block args of fall_target); for loops this is always 0 because
// wasm loops cannot produce values via fall-through (the verifier
// requires falling off a value-producing loop to be unreachable, and
// in practice tinyc never produces such loops).
// =============================================================================
typedef struct {
    MLIR_BlockHandle br_target;
    MLIR_BlockHandle fall_target;
    size_t           n_args;
    size_t           n_fall_args;
    bool             is_loop;
} Frame;

// Module-level layout of import_global data in linmem. Used to lower
// wasmssa.addressof to a wmir.const that yields the wasm linear-memory
// offset of the named symbol.
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

// Maps a function name to its slot index in the indirect-call dispatch
// table. Slots are assigned in the order wasmssa.func_addr targets are
// first encountered during the pre-pass. Used to lower
// wasmssa.func_addr -> wmir.const slot, and to drive the synthesised
// __dispatch_<sig> functions.
typedef struct {
    string   *names;
    int32_t  *slots;
    size_t    n, cap;
} FuncPtrMap;

// Optional explicit slot: if `slot >= 0`, the entry is added with that
// exact slot (used by the wasm-lifter path, where the wasm table
// layout determines the slot number — the dispatcher and call sites
// must both agree on the same integer). Returns the slot ultimately
// assigned.
static int32_t fpm_intern_with_slot(FuncPtrMap *m, string name,
                                    int32_t explicit_slot) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->names[i].size == name.size &&
            memcmp(m->names[i].str, name.str, name.size) == 0) {
            // If an explicit slot is provided and the existing entry
            // has a different one, prefer the explicit slot.
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
static int32_t fpm_intern(FuncPtrMap *m, string name) {
    return fpm_intern_with_slot(m, name, -1);
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

// Maps a function name to its (param_types, result_types) signature
// strings (Wasm value-type tags, e.g. "7f7f" for (i32,i32)). Built
// from the wasmssa.func ops in a pre-pass.
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
    m->names[m->n]        = name;
    m->param_types[m->n]  = pt;
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

typedef struct {
    MLIR_Context     *ctx;
    MLIR_RegionHandle dst_region;
    MLIR_BlockHandle  cur;          // current insertion point
    VMap             *vmap;
    Frame            *frames;
    size_t            n_frames, c_frames;
    bool              cur_terminated; // last appended op was a terminator
    const OffsetMap  *globals;        // module-level data globals offset map
    const FuncPtrMap *fnptrs;         // target name -> slot
    const FuncSigMap *sigs;           // func name -> signature
} Lowerer;

static void L_push_frame(Lowerer *L, MLIR_BlockHandle br_target,
                         MLIR_BlockHandle fall_target,
                         size_t n_args, size_t n_fall_args, bool is_loop) {
    if (L->n_frames == L->c_frames) {
        L->c_frames = L->c_frames ? L->c_frames * 2 : 4;
        L->frames = (Frame *)realloc(L->frames, L->c_frames * sizeof(Frame));
    }
    L->frames[L->n_frames].br_target   = br_target;
    L->frames[L->n_frames].fall_target = fall_target;
    L->frames[L->n_frames].n_args      = n_args;
    L->frames[L->n_frames].n_fall_args = n_fall_args;
    L->frames[L->n_frames].is_loop     = is_loop;
    L->n_frames++;
}
static void L_pop_frame(Lowerer *L) { L->n_frames--; }

static MLIR_BlockHandle L_new_block(Lowerer *L) {
    MLIR_BlockHandle b = MLIR_CreateBlock(L->ctx);
    MLIR_AppendRegionBlock(L->ctx, L->dst_region, b);
    return b;
}

static void L_append(Lowerer *L, MLIR_OpHandle op) {
    MLIR_AppendBlockOp(L->ctx, L->cur, op);
}
static void L_emit_br(Lowerer *L, MLIR_BlockHandle target,
                      MLIR_ValueHandle *args, size_t n_args) {
    L_append(L, build_op_br(L->ctx, target, args, n_args));
    L->cur_terminated = true;
}
static void L_emit_cond_br(Lowerer *L, MLIR_ValueHandle cond,
                           MLIR_BlockHandle t, MLIR_BlockHandle f) {
    L_append(L, build_op_cond_br(L->ctx, cond, t, f));
    L->cur_terminated = true;
}

// Forward decls.
static bool lower_wasmssa_op(Lowerer *L, MLIR_OpHandle src_op);
static bool lower_region(Lowerer *L, MLIR_RegionHandle r);

// =============================================================================
// Compare/binop tables. Map raw wasm opcode → string predicate for
// wmir.icmp. Returns NULL if the opcode is not a compare.
// =============================================================================
static const char *icmp_pred_for_binop(int64_t opc) {
    switch (opc) {
        // i32 compares
        case 0x46: return "eq";
        case 0x47: return "ne";
        case 0x48: return "slt";
        case 0x49: return "ult";
        case 0x4a: return "sgt";
        case 0x4b: return "ugt";
        case 0x4c: return "sle";
        case 0x4d: return "ule";
        case 0x4e: return "sge";
        case 0x4f: return "uge";
        // i64 compares
        case 0x51: return "eq";
        case 0x52: return "ne";
        case 0x53: return "slt";
        case 0x54: return "ult";
        case 0x55: return "sgt";
        case 0x56: return "ugt";
        case 0x57: return "sle";
        case 0x58: return "ule";
        case 0x59: return "sge";
        case 0x5a: return "uge";
    }
    return NULL;
}

// Float comparisons that wasmssa.binop encodes the same way as integer
// ones. Returns the wmir.fcmp pred + fwidth, or NULL if not a float
// compare. Caller checks return value to dispatch.
//   f32: 0x5b eq, 0x5c ne, 0x5d lt, 0x5e gt, 0x5f le, 0x60 ge
//   f64: 0x61 eq, 0x62 ne, 0x63 lt, 0x64 gt, 0x65 le, 0x66 ge
static const char *fcmp_pred_for_binop(int64_t opc, int *fwidth_out) {
    if (opc >= 0x5b && opc <= 0x60) { *fwidth_out = 32; }
    else if (opc >= 0x61 && opc <= 0x66) { *fwidth_out = 64; }
    else return NULL;
    int rel = (int)((opc - 0x5b) % 6);
    switch (rel) {
        case 0: return "oeq";
        case 1: return "une";
        case 2: return "olt";
        case 3: return "ogt";
        case 4: return "ole";
        case 5: return "oge";
    }
    return NULL;
}

// Float binops that produce a same-typed float result.
//   f32: 0x92 fadd, 0x93 fsub, 0x94 fmul, 0x95 fdiv
//   f64: 0xa0 fadd, 0xa1 fsub, 0xa2 fmul, 0xa3 fdiv
static const char *fbinop_kind(int64_t opc, int *fwidth_out) {
    static const char *kinds[] = { "fadd", "fsub", "fmul", "fdiv" };
    if (opc >= 0x92 && opc <= 0x95) { *fwidth_out = 32; return kinds[opc - 0x92]; }
    if (opc >= 0xa0 && opc <= 0xa3) { *fwidth_out = 64; return kinds[opc - 0xa0]; }
    return NULL;
}

// =============================================================================
// Per-op lowering. Appends ops to L->cur and updates L->vmap as
// needed. Returns false on an unsupported op.
// =============================================================================
static bool lower_wasmssa_op(Lowerer *L, MLIR_OpHandle src_op) {
    MLIR_Context *ctx = L->ctx;
    MLIR_OpType t = MLIR_GetOpType(src_op);
    switch (t) {

    case OP_TYPE_WASMSSA_CONST: {
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        int64_t v  = at_i(src_op, "value");
        MLIR_AttributeHandle attrs[1];
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        attrs[0] = MLIR_CreateAttributeInteger(ctx, str_lit("value"), v, res_ty[0]);
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_CONST,
            attrs, 1, res_ty, 1, res, NULL, 0);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_ADD:
    case OP_TYPE_WASMSSA_SUB: {
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 1), &b)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.%s\n",
                    t == OP_TYPE_WASMSSA_ADD ? "add" : "sub");
            return false;
        }
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt ? vt : 0x7f) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_ValueHandle ops[2] = { a, b };
        MLIR_OpType out_t = (t == OP_TYPE_WASMSSA_ADD)
                          ? OP_TYPE_WMIR_IADD : OP_TYPE_WMIR_ISUB;
        MLIR_OpHandle out = build_op_simple(ctx, out_t,
            NULL, 0, res_ty, 1, res, ops, 2);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_BINOP: {
        int64_t opc = at_i(src_op, "wasm_opcode");
        const char *pred = icmp_pred_for_binop(opc);
        MLIR_ValueHandle a, b;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 1), &b)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.binop\n");
            return false;
        }
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_ValueHandle ops[2] = { a, b };
        if (pred) {
            MLIR_AttributeHandle attrs[1] = { attr_s(ctx, "pred", pred, strlen(pred)) };
            MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_ICMP,
                attrs, 1, res_ty, 1, res, ops, 2);
            L_append(L, out);
            vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
            return true;
        }
        int fwidth = 0;
        const char *fpred = fcmp_pred_for_binop(opc, &fwidth);
        if (fpred) {
            // wmir.fcmp produces an i32 boolean (0 or 1), matching the
            // wasm convention for compare opcodes.
            MLIR_TypeHandle ity = MLIR_CreateTypeInteger(ctx, 32, true);
            MLIR_TypeHandle rty[1] = { ity };
            MLIR_ValueHandle r2[1] = {
                MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, ity,
                    (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
            };
            MLIR_AttributeHandle attrs[2] = {
                attr_s  (ctx, "pred",   fpred, strlen(fpred)),
                attr_i32(ctx, "fwidth", fwidth),
            };
            MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_FCMP,
                attrs, 2, rty, 1, r2, ops, 2);
            L_append(L, out);
            vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), r2[0]);
            return true;
        }
        const char *fkind = fbinop_kind(opc, &fwidth);
        if (fkind) {
            MLIR_AttributeHandle attrs[2] = {
                attr_s  (ctx, "kind",   fkind, strlen(fkind)),
                attr_i32(ctx, "fwidth", fwidth),
            };
            MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_FBINOP,
                attrs, 2, res_ty, 1, res, ops, 2);
            L_append(L, out);
            vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
            return true;
        }
        MLIR_OpType out_t;
        switch (opc) {
            // i32 arithmetic
            case 0x6a: out_t = OP_TYPE_WMIR_IADD; break;
            case 0x6b: out_t = OP_TYPE_WMIR_ISUB; break;
            case 0x6c: out_t = OP_TYPE_WMIR_IMUL; break;
            case 0x6d: out_t = OP_TYPE_WMIR_SDIV; break;
            case 0x6e: out_t = OP_TYPE_WMIR_UDIV; break;
            case 0x6f: out_t = OP_TYPE_WMIR_SREM; break;
            case 0x70: out_t = OP_TYPE_WMIR_UREM; break;
            case 0x71: out_t = OP_TYPE_WMIR_IAND; break;
            case 0x72: out_t = OP_TYPE_WMIR_IOR;  break;
            case 0x73: out_t = OP_TYPE_WMIR_IXOR; break;
            case 0x74: out_t = OP_TYPE_WMIR_ISHL; break;
            case 0x75: out_t = OP_TYPE_WMIR_SSHR; break;
            case 0x76: out_t = OP_TYPE_WMIR_USHR; break;
            // i64 arithmetic
            case 0x7c: out_t = OP_TYPE_WMIR_IADD; break;
            case 0x7d: out_t = OP_TYPE_WMIR_ISUB; break;
            case 0x7e: out_t = OP_TYPE_WMIR_IMUL; break;
            case 0x7f: out_t = OP_TYPE_WMIR_SDIV; break;
            case 0x80: out_t = OP_TYPE_WMIR_UDIV; break;
            case 0x81: out_t = OP_TYPE_WMIR_SREM; break;
            case 0x82: out_t = OP_TYPE_WMIR_UREM; break;
            case 0x83: out_t = OP_TYPE_WMIR_IAND; break;
            case 0x84: out_t = OP_TYPE_WMIR_IOR;  break;
            case 0x85: out_t = OP_TYPE_WMIR_IXOR; break;
            case 0x86: out_t = OP_TYPE_WMIR_ISHL; break;
            case 0x87: out_t = OP_TYPE_WMIR_SSHR; break;
            case 0x88: out_t = OP_TYPE_WMIR_USHR; break;
            default:
                fprintf(stderr,
                    "wmir: wasmssa.binop opcode 0x%llx not yet supported\n",
                    (unsigned long long)opc);
                return false;
        }
        MLIR_OpHandle out = build_op_simple(ctx, out_t,
            NULL, 0, res_ty, 1, res, ops, 2);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_EQZ: {
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &a)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.eqz\n");
            return false;
        }
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_EQZ,
            NULL, 0, res_ty, 1, res, &a, 1);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    // wasmssa.extend_i32_s %x : i32 -> i64 (sign-extend). The valtype
    // attribute carries the result type byte (0x7e for i64). Lowers to
    // wmir.sext.
    case OP_TYPE_WASMSSA_EXTEND_I32_S: {
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &a)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.extend_i32_s\n");
            return false;
        }
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_SEXT,
            NULL, 0, res_ty, 1, res, &a, 1);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    // wasmssa.unop {wasm_opcode = ...}. We only handle the integer
    // conversion opcodes here; floats are deferred.
    //   0xa7 (167) i32.wrap_i64       -> wmir.trunc i64 -> i32
    //   0xac (172) i64.extend_i32_s   -> wmir.sext  i32 -> i64
    //   0xad (173) i64.extend_i32_u   -> wmir.zext  i32 -> i64
    //   0xc0 (192) i32.extend8_s      -> trunc to i8 then sext to i32 (we
    //                                    just emit a sext via i8 conceptually;
    //                                    for now masked via and+sxtb path)
    case OP_TYPE_WASMSSA_UNOP: {
        int64_t opc = at_i(src_op, "wasm_opcode");
        MLIR_ValueHandle a;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &a)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.unop\n");
            return false;
        }
        // Float unops (fabs/fneg/fsqrt) become wmir.funop preserving
        // the bit-pattern carrier width.
        {
            const char *fkind = NULL;
            int fwidth = 0;
            switch (opc) {
            case 0x8b: fkind = "fabs";  fwidth = 32; break;
            case 0x8c: fkind = "fneg";  fwidth = 32; break;
            case 0x91: fkind = "fsqrt"; fwidth = 32; break;
            case 0x99: fkind = "fabs";  fwidth = 64; break;
            case 0x9a: fkind = "fneg";  fwidth = 64; break;
            case 0x9f: fkind = "fsqrt"; fwidth = 64; break;
            }
            if (fkind) {
                MLIR_TypeHandle rty[1] = {
                    MLIR_CreateTypeInteger(ctx, fwidth, true)
                };
                MLIR_ValueHandle r[1] = {
                    MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                        rty[0], (string){0},
                        MLIR_CreateLocationUnknown(ctx, (string){0}))
                };
                MLIR_AttributeHandle attrs[2] = {
                    attr_s  (ctx, "kind",   fkind, strlen(fkind)),
                    attr_i32(ctx, "fwidth", fwidth),
                };
                MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_FUNOP,
                    attrs, 2, rty, 1, r, &a, 1);
                L_append(L, out);
                vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), r[0]);
                return true;
            }
        }
        // FP <-> int conversion family (wmir.fconv).
        //   src_w / dst_w in {32, 64}, kind one of "f2f"/"f2i"/"i2f",
        //   sign matters for f2i / i2f (signed vs unsigned variants).
        {
            const char *fkind = NULL;
            int src_w = 0, dst_w = 0;
            bool sign = false;
            switch (opc) {
            // i32.trunc_f32_s..i64.trunc_f64_u
            case 0xa8: fkind="f2i"; src_w=32; dst_w=32; sign=true;  break;
            case 0xa9: fkind="f2i"; src_w=32; dst_w=32; sign=false; break;
            case 0xaa: fkind="f2i"; src_w=64; dst_w=32; sign=true;  break;
            case 0xab: fkind="f2i"; src_w=64; dst_w=32; sign=false; break;
            case 0xae: fkind="f2i"; src_w=32; dst_w=64; sign=true;  break;
            case 0xaf: fkind="f2i"; src_w=32; dst_w=64; sign=false; break;
            case 0xb0: fkind="f2i"; src_w=64; dst_w=64; sign=true;  break;
            case 0xb1: fkind="f2i"; src_w=64; dst_w=64; sign=false; break;
            // f32/f64 convert_i32/64_s/u
            case 0xb2: fkind="i2f"; src_w=32; dst_w=32; sign=true;  break;
            case 0xb3: fkind="i2f"; src_w=32; dst_w=32; sign=false; break;
            case 0xb4: fkind="i2f"; src_w=64; dst_w=32; sign=true;  break;
            case 0xb5: fkind="i2f"; src_w=64; dst_w=32; sign=false; break;
            case 0xb7: fkind="i2f"; src_w=32; dst_w=64; sign=true;  break;
            case 0xb8: fkind="i2f"; src_w=32; dst_w=64; sign=false; break;
            case 0xb9: fkind="i2f"; src_w=64; dst_w=64; sign=true;  break;
            case 0xba: fkind="i2f"; src_w=64; dst_w=64; sign=false; break;
            // f32.demote_f64 / f64.promote_f32
            case 0xb6: fkind="f2f"; src_w=64; dst_w=32; break;
            case 0xbb: fkind="f2f"; src_w=32; dst_w=64; break;
            // *.reinterpret_* — pure bitcast; in our bit-pattern model
            // this is the identity (operand and result already share the
            // same i32/i64 carrier). Forward the operand directly.
            case 0xbc: case 0xbd: case 0xbe: case 0xbf:
                vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), a);
                return true;
            }
            if (fkind) {
                MLIR_TypeHandle rty[1] = {
                    MLIR_CreateTypeInteger(ctx, dst_w, true)
                };
                MLIR_ValueHandle r[1] = {
                    MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                        rty[0], (string){0},
                        MLIR_CreateLocationUnknown(ctx, (string){0}))
                };
                MLIR_AttributeHandle attrs[4] = {
                    attr_s  (ctx, "kind",  fkind, strlen(fkind)),
                    attr_i32(ctx, "src_w", src_w),
                    attr_i32(ctx, "dst_w", dst_w),
                    attr_b  (ctx, "sign",  sign),
                };
                MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_FCONV,
                    attrs, 4, rty, 1, r, &a, 1);
                L_append(L, out);
                vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), r[0]);
                return true;
            }
        }
        MLIR_OpType k;
        MLIR_TypeHandle rt;
        int32_t src_bits = 0; // 0 == default (handled per-op)
        switch (opc) {
        case 0xa7: k = OP_TYPE_WMIR_TRUNC;
                   rt = MLIR_CreateTypeInteger(ctx, 32, true); break;
        case 0xac: k = OP_TYPE_WMIR_SEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 64, true);
                   src_bits = 32; break;
        case 0xad: k = OP_TYPE_WMIR_ZEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 64, true);
                   src_bits = 32; break;
        case 0xc0: k = OP_TYPE_WMIR_SEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 32, true);
                   src_bits = 8; break;
        case 0xc1: k = OP_TYPE_WMIR_SEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 32, true);
                   src_bits = 16; break;
        case 0xc2: k = OP_TYPE_WMIR_SEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 64, true);
                   src_bits = 8; break;
        case 0xc3: k = OP_TYPE_WMIR_SEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 64, true);
                   src_bits = 16; break;
        case 0xc4: k = OP_TYPE_WMIR_SEXT;
                   rt = MLIR_CreateTypeInteger(ctx, 64, true);
                   src_bits = 32; break;
        default:
            fprintf(stderr,
                "wmir: wasmssa.unop opcode 0x%llx not yet supported\n",
                (long long)opc);
            return false;
        }
        MLIR_TypeHandle res_ty[1] = { rt };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, rt,
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_AttributeHandle attrs[1];
        size_t na = 0;
        if (src_bits) attrs[na++] = attr_i32(ctx, "src_bits", src_bits);
        MLIR_OpHandle out = build_op_simple(ctx, k,
            attrs, na, res_ty, 1, res, &a, 1);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    // wasmssa.addressof {target = ".str.N"} -> wmir.const i32 = <offset>.
    // The offset has been pre-assigned by the module-level walk that
    // also emitted wmir.data_init ops. Unknown targets are an error.
    case OP_TYPE_WASMSSA_ADDRESSOF: {
        string tgt = at_s(src_op, "target");
        int32_t off;
        if (!L->globals || !omap_get(L->globals, tgt, &off)) {
            fprintf(stderr,
                "wmir: wasmssa.addressof unknown global '%.*s'\n",
                (int)tgt.size, tgt.str);
            return false;
        }
        MLIR_AttributeHandle attrs[1] = { attr_i32(ctx, "value", off) };
        MLIR_TypeHandle res_ty[1] = { MLIR_CreateTypeInteger(ctx, 32, true) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_CONST,
            attrs, 1, res_ty, 1, res, NULL, 0);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_SELECT: {
        // wasmssa.select(%a, %b, %cond) -> R   (cond is the LAST operand;
        // see mlir_llvm_to_wasmssa.c which orders args as a,b,cond).
        MLIR_ValueHandle a, b, c;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &a) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 1), &b) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 2), &c)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.select\n");
            return false;
        }
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_ValueHandle ops[3] = { a, b, c };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_SELECT,
            NULL, 0, res_ty, 1, res, ops, 3);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_GLOBAL_GET: {
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        int64_t idx = at_i(src_op, "global_idx");
        MLIR_AttributeHandle attrs[1] = { attr_i32(ctx, "global_idx", idx) };
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_GLOBAL_GET,
            attrs, 1, res_ty, 1, res, NULL, 0);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_LOCAL_GET: {
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        int64_t idx = at_i(src_op, "local_idx");
        MLIR_AttributeHandle attrs[2] = {
            attr_i32(ctx, "local_idx", idx),
            attr_i32(ctx, "valtype",   vt),
        };
        MLIR_TypeHandle res_ty[1] = { vt_to_type(ctx, vt) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_LOCAL_GET,
            attrs, 2, res_ty, 1, res, NULL, 0);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_LOCAL_SET: {
        uint8_t vt = (uint8_t)at_i(src_op, "valtype");
        int64_t idx = at_i(src_op, "local_idx");
        MLIR_ValueHandle v;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &v)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.local_set\n");
            return false;
        }
        MLIR_AttributeHandle attrs[2] = {
            attr_i32(ctx, "local_idx", idx),
            attr_i32(ctx, "valtype",   vt),
        };
        MLIR_ValueHandle ops[1] = { v };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_LOCAL_SET,
            attrs, 2, NULL, 0, NULL, ops, 1);
        L_append(L, out);
        return true;
    }

    case OP_TYPE_WASMSSA_GLOBAL_SET: {
        int64_t idx = at_i(src_op, "global_idx");
        MLIR_ValueHandle v;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &v)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.global_set\n");
            return false;
        }
        MLIR_AttributeHandle attrs[1] = { attr_i32(ctx, "global_idx", idx) };
        MLIR_ValueHandle ops[1] = { v };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_GLOBAL_SET,
            attrs, 1, NULL, 0, NULL, ops, 1);
        L_append(L, out);
        return true;
    }

    case OP_TYPE_WASMSSA_LOAD: {
        int64_t off = at_i(src_op, "memory_offset");
        int64_t sz  = at_i(src_op, "mem_size_bytes");
        uint8_t vt  = (uint8_t)at_i(src_op, "valtype");
        // Accept any (size, valtype) the operand-side semantics allow.
        // Result type is whatever valtype dictates — for sub-word loads
        // (sz < width) the load is zero-extended into the result. f32/
        // f64 are modelled as i32/i64 bit patterns at the wmir level.
        bool ok =
            (vt == WT_I32 && (sz == 1 || sz == 2 || sz == 4)) ||
            (vt == WT_I64 && (sz == 1 || sz == 2 || sz == 4 || sz == 8)) ||
            (vt == WT_F32 && sz == 4) ||
            (vt == WT_F64 && sz == 8);
        if (!ok) {
            fprintf(stderr,
                "wmir: wasmssa.load mem_size=%lld valtype=%u not yet supported\n",
                (long long)sz, (unsigned)vt);
            return false;
        }
        MLIR_ValueHandle addr;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &addr)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.load\n");
            return false;
        }
        MLIR_AttributeHandle attrs[2] = {
            attr_i32(ctx, "memory_offset", off),
            attr_i32(ctx, "mem_size",      sz),
        };
        bool is64 = (vt == WT_I64 || vt == WT_F64);
        MLIR_TypeHandle res_ty[1] = {
            is64 ? MLIR_CreateTypeInteger(ctx, 64, true)
                 : MLIR_CreateTypeInteger(ctx, 32, true)
        };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_ValueHandle ops[1] = { addr };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_LOAD,
            attrs, 2, res_ty, 1, res, ops, 1);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_MEMORY_SIZE: {
        // i32 result.
        MLIR_TypeHandle res_ty[1] = {
            MLIR_CreateTypeInteger(ctx, 32, true)
        };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_MEMORY_SIZE,
            NULL, 0, res_ty, 1, res, NULL, 0);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_MEMORY_GROW: {
        MLIR_ValueHandle delta;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &delta)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.memory_grow\n");
            return false;
        }
        MLIR_TypeHandle res_ty[1] = {
            MLIR_CreateTypeInteger(ctx, 32, true)
        };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_ValueHandle ops[1] = { delta };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_MEMORY_GROW,
            NULL, 0, res_ty, 1, res, ops, 1);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_STORE: {
        int64_t off = at_i(src_op, "memory_offset");
        int64_t sz  = at_i(src_op, "mem_size_bytes");
        uint8_t vt  = (uint8_t)at_i(src_op, "valtype");
        bool ok =
            (vt == WT_I32 && (sz == 1 || sz == 2 || sz == 4)) ||
            (vt == WT_I64 && (sz == 1 || sz == 2 || sz == 4 || sz == 8)) ||
            (vt == WT_F32 && sz == 4) ||
            (vt == WT_F64 && sz == 8);
        if (!ok) {
            fprintf(stderr,
                "wmir: wasmssa.store mem_size=%lld valtype=%u not yet supported\n",
                (long long)sz, (unsigned)vt);
            return false;
        }
        MLIR_ValueHandle addr, val;
        if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 0), &addr) ||
            !vmap_get(L->vmap, MLIR_GetOpOperand(src_op, 1), &val)) {
            fprintf(stderr, "wmir: unbound operand on wasmssa.store\n");
            return false;
        }
        MLIR_AttributeHandle attrs[2] = {
            attr_i32(ctx, "memory_offset", off),
            attr_i32(ctx, "mem_size",      sz),
        };
        MLIR_ValueHandle ops[2] = { addr, val };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_STORE,
            attrs, 2, NULL, 0, NULL, ops, 2);
        L_append(L, out);
        return true;
    }

    case OP_TYPE_WASMSSA_CALL: {
        string callee = at_s(src_op, "target");
        if (callee.size == 0) {
            fprintf(stderr, "wmir: wasmssa.call without `target` attribute\n");
            return false;
        }
        size_t no = MLIR_GetOpNumOperands(src_op);
        MLIR_ValueHandle *ops = NULL;
        MLIR_ValueHandle ops_inline[16];
        if (no <= 16) {
            ops = ops_inline;
        } else {
            ops = (MLIR_ValueHandle *)malloc(no * sizeof(MLIR_ValueHandle));
            if (!ops) {
                fprintf(stderr, "wmir: oom allocating wasmssa.call operands\n");
                return false;
            }
        }
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, k), &ops[k])) {
                MLIR_ValueHandle ov = MLIR_GetOpOperand(src_op, k);
                MLIR_TypeHandle ot = MLIR_GetValueType(ov);
                string ots = MLIR_GetTypeString(ctx, ot);
                fprintf(stderr,
                    "wmir: unbound operand %zu/%zu on wasmssa.call "
                    "(target='%.*s', operand_type='%.*s')\n",
                    k, no, (int)callee.size, callee.str,
                    (int)ots.size, ots.str);
                if (ops != ops_inline) free(ops);
                return false;
            }
        }
        size_t nr = MLIR_GetOpNumResults(src_op);
        if (nr > 1) {
            fprintf(stderr, "wmir: wasmssa.call multi-result not yet supported\n");
            if (ops != ops_inline) free(ops);
            return false;
        }
        MLIR_AttributeHandle attrs[1];
        attrs[0] = attr_s(ctx, "target", callee.str, callee.size);
        MLIR_TypeHandle res_ty[1];
        MLIR_ValueHandle res[1];
        if (nr == 1) {
            MLIR_ValueHandle rv = MLIR_GetOpResult(src_op, 0);
            MLIR_TypeHandle rt = normalise_carrier(ctx, MLIR_GetValueType(rv));
            res_ty[0] = rt;
            res[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, rt,
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}));
        }
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_CALL,
            attrs, 1, res_ty, nr, res, ops, no);
        L_append(L, out);
        if (nr == 1) vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        if (ops != ops_inline) free(ops);
        return true;
    }

    case OP_TYPE_WASMSSA_FUNC_ADDR: {
        // Returns the slot index assigned to the named function in the
        // module-level fnptr table. Lowered to wmir.const slot.
        string target = at_s(src_op, "target");
        int32_t slot = -1;
        if (!L->fnptrs || !fpm_lookup(L->fnptrs, target, &slot)) {
            fprintf(stderr,
                "wmir: wasmssa.func_addr target '%.*s' not in fnptr table\n",
                (int)target.size, target.str);
            return false;
        }
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_AttributeHandle a[1];
        a[0] = MLIR_CreateAttributeInteger(ctx, str_lit("value"),
            (int64_t)slot, i32);
        MLIR_ValueHandle rv = MLIR_CreateValueOpResult(ctx,
            MLIR_INVALID_HANDLE, 0, i32, (string){0},
            MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_TypeHandle rty[1] = { i32 };
        MLIR_ValueHandle res[1] = { rv };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_CONST,
            a, 1, rty, 1, res, NULL, 0);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), rv);
        return true;
    }

    case OP_TYPE_WASMSSA_CALL_INDIRECT: {
        // Operand layout in wasmssa: (args..., slot_idx).
        // Lower to a regular wmir.call to a synthesised dispatcher
        // function whose name is derived from the signature, passing
        // (slot, args...) — slot moves to the front.
        size_t no = MLIR_GetOpNumOperands(src_op);
        if (no < 1) {
            fprintf(stderr, "wmir: wasmssa.call_indirect with no operands\n");
            return false;
        }
        MLIR_ValueHandle ops_in_inline[16];
        MLIR_ValueHandle ops_out_inline[16];
        MLIR_ValueHandle *ops_in;
        MLIR_ValueHandle *ops_out;
        bool ops_heap = no > 16;
        if (ops_heap) {
            ops_in  = (MLIR_ValueHandle *)malloc(no * sizeof(MLIR_ValueHandle));
            ops_out = (MLIR_ValueHandle *)malloc(no * sizeof(MLIR_ValueHandle));
            if (!ops_in || !ops_out) {
                fprintf(stderr, "wmir: oom allocating call_indirect operands\n");
                free(ops_in); free(ops_out);
                return false;
            }
        } else {
            ops_in  = ops_in_inline;
            ops_out = ops_out_inline;
        }
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, k), &ops_in[k])) {
                fprintf(stderr, "wmir: unbound operand on wasmssa.call_indirect\n");
                if (ops_heap) { free(ops_in); free(ops_out); }
                return false;
            }
        }
        // Slot is last; reorder to (slot, args...).
        ops_out[0] = ops_in[no - 1];
        for (size_t k = 0; k + 1 < no; k++) ops_out[1 + k] = ops_in[k];

        // Build dispatcher name __dispatch_<sig>: e.g. "__dispatch_7f7f_7f"
        // where the signature uses underscores as a delimiter so it is
        // safe to use as a C identifier. The buffer must outlive the
        // attribute (attr_s does not copy), so it is heap-allocated and
        // intentionally leaked — only one per call_indirect site, lives
        // until program exit.
        string sig_p = at_s(src_op, "sig_params");
        string sig_r = at_s(src_op, "sig_results");
        size_t name_cap = sig_p.size + sig_r.size + 32;
        char *name_buf = (char *)malloc(name_cap);
        int nlen = snprintf(name_buf, name_cap,
            "__dispatch_%.*s_%.*s",
            (int)sig_p.size, sig_p.str,
            (int)sig_r.size, sig_r.str);
        if (nlen <= 0 || (size_t)nlen >= name_cap) {
            fprintf(stderr, "wmir: call_indirect dispatcher name too long\n");
            free(name_buf);
            if (ops_heap) { free(ops_in); free(ops_out); }
            return false;
        }

        size_t nr = MLIR_GetOpNumResults(src_op);
        if (nr > 1) {
            fprintf(stderr,
                "wmir: wasmssa.call_indirect multi-result not supported\n");
            if (ops_heap) { free(ops_in); free(ops_out); }
            return false;
        }
        MLIR_AttributeHandle attrs[1];
        attrs[0] = attr_s(ctx, "target", name_buf, (size_t)nlen);
        MLIR_TypeHandle res_ty[1];
        MLIR_ValueHandle res[1];
        if (nr == 1) {
            MLIR_ValueHandle rv = MLIR_GetOpResult(src_op, 0);
            MLIR_TypeHandle rt = normalise_carrier(ctx, MLIR_GetValueType(rv));
            res_ty[0] = rt;
            res[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, rt,
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}));
        }
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_CALL,
            attrs, 1, res_ty, nr, res, ops_out, no);
        L_append(L, out);
        if (nr == 1) vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        if (ops_heap) { free(ops_in); free(ops_out); }
        return true;
    }

    case OP_TYPE_WASMSSA_RETURN: {
        size_t n_ops = MLIR_GetOpNumOperands(src_op);
        MLIR_ValueHandle operands[8];
        if (n_ops > 8) {
            fprintf(stderr, "wmir: wasmssa.return with >8 results unsupported\n");
            return false;
        }
        for (size_t k = 0; k < n_ops; k++) {
            MLIR_ValueHandle sv = MLIR_GetOpOperand(src_op, k);
            if (!vmap_get(L->vmap, sv, &operands[k])) {
                fprintf(stderr, "wmir: unbound operand on wasmssa.return\n");
                return false;
            }
        }
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_RETURN,
            NULL, 0, NULL, 0, NULL, operands, n_ops);
        L_append(L, out);
        L->cur_terminated = true;
        return true;
    }

    case OP_TYPE_WASMSSA_UNREACHABLE: {
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_UNREACHABLE,
            NULL, 0, NULL, 0, NULL, NULL, 0);
        L_append(L, out);
        L->cur_terminated = true;
        return true;
    }

    case OP_TYPE_WASMSSA_BR: {
        // wasmssa.br(%v...) {depth=D}: jump to frames[top-D].br_target.
        // For loop frames this targets the loop header (continue);
        // for block/if frames this targets the merge block (exit).
        int64_t depth = at_i(src_op, "depth");
        if (depth < 0 || (size_t)depth >= L->n_frames) {
            fprintf(stderr,
                "wmir: wasmssa.br depth=%lld out of range (n_frames=%zu)\n",
                (long long)depth, L->n_frames);
            return false;
        }
        Frame *f = &L->frames[L->n_frames - 1 - (size_t)depth];
        size_t no = MLIR_GetOpNumOperands(src_op);
        MLIR_ValueHandle args[8];
        if (no > 8) {
            fprintf(stderr, "wmir: wasmssa.br with >8 args unsupported\n");
            return false;
        }
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, k), &args[k])) {
                fprintf(stderr, "wmir: unbound operand on wasmssa.br\n");
                return false;
            }
        }
        L_emit_br(L, f->br_target, args, no);
        return true;
    }

    case OP_TYPE_WASMSSA_BR_IF: {
        // wasmssa.br_if(%cond) {depth=D}: if cond != 0, jump to
        // frames[top-D].br_target with no operands; otherwise fall
        // through. The lifter emits value-carrying conditional
        // branches as `wasmssa.if (%cond) { wasmssa.br D (vals) }`
        // so we only ever see br_if with the bare cond and zero
        // operands here.
        int64_t depth = at_i(src_op, "depth");
        if (depth < 0 || (size_t)depth >= L->n_frames) {
            fprintf(stderr,
                "wmir: wasmssa.br_if depth=%lld out of range (n_frames=%zu)\n",
                (long long)depth, L->n_frames);
            return false;
        }
        Frame *f = &L->frames[L->n_frames - 1 - (size_t)depth];
        MLIR_ValueHandle cond_src = MLIR_GetOpOperand(src_op, 0);
        MLIR_ValueHandle cond;
        if (!vmap_get(L->vmap, cond_src, &cond)) {
            fprintf(stderr, "wmir: unbound condition on wasmssa.br_if\n");
            return false;
        }
        MLIR_BlockHandle fall = L_new_block(L);
        L_emit_cond_br(L, cond, f->br_target, fall);
        L->cur = fall;
        L->cur_terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_BLOCK_RETURN: {
        // Terminator of the innermost wasmssa.block / .loop / .if
        // region. Always emitted by the lifter at every region's
        // end; semantically means "fall off end of region".
        //
        // For block/if frames: fall_target == merge (same as br_target).
        // For loop frames:     fall_target == post-loop exit block
        //                      (DIFFERENT from br_target = header).
        //
        // Using f->br_target here used to cause an infinite loop in
        // any function whose body is `loop ... br_if 0 ... end`,
        // because the implicit fall-off-end would branch back to the
        // loop header instead of exiting. See e.g. strlen, strcmp,
        // memcmp, memchr — and any inlined-loop tinyc helper.
        if (L->n_frames == 0) {
            fprintf(stderr, "wmir: wasmssa.block_return with no frame\n");
            return false;
        }
        Frame *f = &L->frames[L->n_frames - 1];
        size_t no = MLIR_GetOpNumOperands(src_op);
        MLIR_ValueHandle args[8];
        if (no > 8) {
            fprintf(stderr,
                "wmir: wasmssa.block_return with >8 args unsupported\n");
            return false;
        }
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, k), &args[k])) {
                fprintf(stderr, "wmir: unbound operand on wasmssa.block_return\n");
                return false;
            }
        }
        L_emit_br(L, f->fall_target, args, no);
        return true;
    }

    case OP_TYPE_WASMSSA_BLOCK: {
        // wasmssa.block() -> R...   { body }
        // body's br depth=0 (and a final implicit fall-through, if any)
        // jumps to ^merge with the block's result values.
        size_t n_results = MLIR_GetOpNumResults(src_op);
        MLIR_BlockHandle merge = L_new_block(L);
        for (size_t i = 0; i < n_results; i++) {
            MLIR_ValueHandle src_res = MLIR_GetOpResult(src_op, i);
            MLIR_TypeHandle ty = normalise_carrier(ctx, MLIR_GetValueType(src_res));
            MLIR_ValueHandle ba = MLIR_CreateValueBlockArg(ctx, (string){0},
                (uint32_t)i, ty,
                MLIR_CreateLocationUnknown(ctx, (string){0}));
            MLIR_AppendBlockArg(ctx, merge, ba);
            vmap_set(L->vmap, src_res, ba);
        }
        if (MLIR_GetOpNumRegions(src_op) < 1) {
            fprintf(stderr, "wmir: wasmssa.block has no region\n");
            return false;
        }
        L_push_frame(L, merge, merge, n_results, n_results, /*is_loop=*/false);
        bool saved_term = L->cur_terminated;
        L->cur_terminated = false;
        if (!lower_region(L, MLIR_GetOpRegion(src_op, 0))) {
            L_pop_frame(L);
            return false;
        }
        // If body falls through (no terminator), implicit br to merge with
        // no args (the wasm "fall off end of block" path; only well-typed
        // when the block has no result types).
        if (!L->cur_terminated) {
            L_emit_br(L, merge, NULL, 0);
        }
        L_pop_frame(L);
        L->cur = merge;
        L->cur_terminated = saved_term;  // restore (merge starts non-terminated)
        L->cur_terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_IF: {
        // wasmssa.if(%cond) -> R... { then } else? { else }
        MLIR_ValueHandle cond_src = MLIR_GetOpOperand(src_op, 0);
        MLIR_ValueHandle cond;
        if (!vmap_get(L->vmap, cond_src, &cond)) {
            fprintf(stderr, "wmir: unbound condition on wasmssa.if\n");
            return false;
        }
        size_t n_regions = MLIR_GetOpNumRegions(src_op);
        if (n_regions < 1) {
            fprintf(stderr, "wmir: wasmssa.if has no then region\n");
            return false;
        }
        bool has_else = (n_regions >= 2)
            && (MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(src_op, 1)) > 0);

        size_t n_results = MLIR_GetOpNumResults(src_op);

        MLIR_BlockHandle then_blk = L_new_block(L);
        MLIR_BlockHandle else_blk = has_else ? L_new_block(L) : MLIR_INVALID_HANDLE;
        MLIR_BlockHandle merge    = L_new_block(L);

        for (size_t i = 0; i < n_results; i++) {
            MLIR_ValueHandle src_res = MLIR_GetOpResult(src_op, i);
            MLIR_TypeHandle ty = normalise_carrier(ctx, MLIR_GetValueType(src_res));
            MLIR_ValueHandle ba = MLIR_CreateValueBlockArg(ctx, (string){0},
                (uint32_t)i, ty,
                MLIR_CreateLocationUnknown(ctx, (string){0}));
            MLIR_AppendBlockArg(ctx, merge, ba);
            vmap_set(L->vmap, src_res, ba);
        }

        MLIR_BlockHandle f_target = has_else ? else_blk : merge;
        L_emit_cond_br(L, cond, then_blk, f_target);

        // then-region
        L->cur = then_blk;
        L->cur_terminated = false;
        L_push_frame(L, merge, merge, n_results, n_results, /*is_loop=*/false);
        if (!lower_region(L, MLIR_GetOpRegion(src_op, 0))) {
            L_pop_frame(L);
            return false;
        }
        if (!L->cur_terminated) {
            // Fall-off-end: implicit br to merge with no args.
            L_emit_br(L, merge, NULL, 0);
        }
        L_pop_frame(L);

        if (has_else) {
            L->cur = else_blk;
            L->cur_terminated = false;
            L_push_frame(L, merge, merge, n_results, n_results, /*is_loop=*/false);
            if (!lower_region(L, MLIR_GetOpRegion(src_op, 1))) {
                L_pop_frame(L);
                return false;
            }
            if (!L->cur_terminated) {
                L_emit_br(L, merge, NULL, 0);
            }
            L_pop_frame(L);
        }

        L->cur = merge;
        L->cur_terminated = false;
        return true;
    }

    case OP_TYPE_WASMSSA_LOOP: {
        // wasmssa.loop(%init...) {
        //   ^bb0(%arg0...):
        //     body
        // }
        // `br depth=0` inside the body jumps back to ^bb0 with new args.
        // Falling off the end terminates the loop iteration: control
        // flows to the loop's `post_loop` block.
        if (MLIR_GetOpNumRegions(src_op) < 1) {
            fprintf(stderr, "wmir: wasmssa.loop has no region\n");
            return false;
        }
        MLIR_RegionHandle loop_region = MLIR_GetOpRegion(src_op, 0);
        if (MLIR_GetRegionNumBlocks(loop_region) < 1) {
            fprintf(stderr, "wmir: wasmssa.loop region empty\n");
            return false;
        }
        MLIR_BlockHandle src_loop_blk = MLIR_GetRegionBlock(loop_region, 0);

        size_t no = MLIR_GetOpNumOperands(src_op);
        MLIR_ValueHandle init_args[8];
        if (no > 8) {
            fprintf(stderr, "wmir: wasmssa.loop with >8 init args unsupported\n");
            return false;
        }
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, k), &init_args[k])) {
                fprintf(stderr, "wmir: unbound init operand on wasmssa.loop\n");
                return false;
            }
        }

        MLIR_BlockHandle header = L_new_block(L);
        size_t n_loop_args = MLIR_GetBlockNumArgs(src_loop_blk);
        for (size_t i = 0; i < n_loop_args; i++) {
            MLIR_ValueHandle sa = MLIR_GetBlockArg(src_loop_blk, i);
            MLIR_TypeHandle ty = normalise_carrier(ctx, MLIR_GetValueType(sa));
            MLIR_ValueHandle ba = MLIR_CreateValueBlockArg(ctx, (string){0},
                (uint32_t)i, ty,
                MLIR_CreateLocationUnknown(ctx, (string){0}));
            MLIR_AppendBlockArg(ctx, header, ba);
            vmap_set(L->vmap, sa, ba);
        }

        L_emit_br(L, header, init_args, no);

        // Create post-loop fall-through block BEFORE pushing the frame,
        // so that any `wasmssa.block_return` inside the loop body (the
        // implicit fall-off-end terminator emitted by the lifter) can
        // target it. Using the header here used to cause an infinite
        // loop in any function whose body is `loop ... br_if 0 ... end`,
        // because the implicit fall-off-end would branch back to the
        // loop header instead of exiting.
        //
        // NOTE: post has no block args. wasmssa.loop with non-zero
        // result arity (a loop producing values via fall-through) is
        // not currently supported — tinyc never generates such loops.
        MLIR_BlockHandle post = L_new_block(L);

        L->cur = header;
        L->cur_terminated = false;
        L_push_frame(L, /*br_target=*/header, /*fall_target=*/post,
                     n_loop_args, /*n_fall_args=*/0, /*is_loop=*/true);
        if (!lower_region(L, loop_region)) {
            L_pop_frame(L);
            return false;
        }
        L_pop_frame(L);

        // If the body fell through without emitting a terminator,
        // branch to post explicitly. (This path is hit when the body
        // ends with a non-terminator op; the lifter normally appends
        // a block_return at end-of-region, which already lowers to a
        // br to post.)
        if (!L->cur_terminated) {
            L_emit_br(L, post, NULL, 0);
        }
        L->cur = post;
        L->cur_terminated = false;
        return true;
    }

    default: {
        string nm = MLIR_GetOpName(src_op);
        fprintf(stderr,
            "wmir lowering: unsupported wasmssa op '%.*s' (kind=%d)\n",
            (int)nm.size, nm.str, (int)t);
        return false;
    }
    }
}

// =============================================================================
// Lower a region (just its entry block — wasmssa regions only ever
// have one block before flattening).
// =============================================================================
static bool lower_region(Lowerer *L, MLIR_RegionHandle r) {
    if (MLIR_GetRegionNumBlocks(r) < 1) return true;
    MLIR_BlockHandle blk = MLIR_GetRegionBlock(r, 0);
    size_t n = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        if (!lower_wasmssa_op(L, op)) return false;
    }
    return true;
}

// =============================================================================
// Lower one wasmssa.func to one wmir.func.
// =============================================================================
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src,
                                const OffsetMap *globals,
                                const FuncPtrMap *fnptrs,
                                const FuncSigMap *sigs) {
    string name      = at_s(src, "sym_name");
    bool   exported  = at_b(src, "exported");
    string pt        = at_s(src, "param_types");
    string rt        = at_s(src, "result_types");
    string lt        = at_s(src, "local_types");

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wmir lowering: wasmssa.func has no region\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(src, 0);
    if (MLIR_GetRegionNumBlocks(src_region) < 1) {
        fprintf(stderr, "wmir lowering: wasmssa.func has no entry block\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, 0);

    MLIR_RegionHandle dst_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle  entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, dst_region, entry);

    VMap vmap = {0};
    Lowerer L = {0};
    L.ctx        = ctx;
    L.dst_region = dst_region;
    L.cur        = entry;
    L.vmap       = &vmap;
    L.globals    = globals;
    L.fnptrs     = fnptrs;
    L.sigs       = sigs;

    size_t n_params = MLIR_GetBlockNumArgs(src_blk);
    for (size_t i = 0; i < n_params; i++) {
        MLIR_ValueHandle sa = MLIR_GetBlockArg(src_blk, i);
        MLIR_TypeHandle  ty = normalise_carrier(ctx, MLIR_GetValueType(sa));
        MLIR_ValueHandle da = MLIR_CreateValueBlockArg(ctx, (string){0},
            (uint32_t)i, ty,
            MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, entry, da);
        vmap_set(&vmap, sa, da);
    }
    size_t n_ops = MLIR_GetBlockNumOps(src_blk);
    bool ok = true;
    for (size_t i = 0; i < n_ops; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(src_blk, i);
        if (!lower_wasmssa_op(&L, bo)) { ok = false; break; }
    }
    free(vmap.src); free(vmap.dst);
    free(L.frames);
    if (!ok) return MLIR_INVALID_HANDLE;

    // Promote wasm locals to SSA values + block-arg phis (mem2reg). After this
    // the body holds no wmir.local_get/local_set ops, so the frame allocates no
    // local slots: emit local_types="" below.
    bool promoted = false;
    if (lt.size > 0)
        promoted = mlir_wmir_mem2reg(ctx, dst_region, lt.str, lt.size);

    MLIR_AttributeHandle attrs[6];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",     name.str, name.size);
    attrs[na++] = attr_s(ctx, "param_types",  pt.str,   pt.size);
    attrs[na++] = attr_s(ctx, "result_types", rt.str,   rt.size);
    if (promoted)
        attrs[na++] = attr_s(ctx, "local_types", "", 0);
    else
        attrs[na++] = attr_s(ctx, "local_types", lt.str, lt.size);
    attrs[na++] = attr_b(ctx, "exported",     exported);

    MLIR_RegionHandle regs[1] = { dst_region };
    return MLIR_CreateOp(ctx, OP_TYPE_WMIR_FUNC,
        op_type_to_string(OP_TYPE_WMIR_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Top-level: walk the wasmssa module body, build a fresh wmir module.
// =============================================================================
MLIR_OpHandle mlir_wasmssa_to_wmir(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    if (!ssa_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(ssa_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ssa_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };

    // Propagate `memory_min_pages` from the wasmssa module so the
    // wmir -> aarch64 backend can size the linear memory image
    // correctly (see mlir_wasm_to_wasmstack.c for the rationale).
    MLIR_AttributeHandle mod_attrs[1];
    size_t n_mod_attrs = 0;
    MLIR_AttributeHandle a_min_pages = MLIR_GetOpAttributeByName(
        ssa_module, "memory_min_pages");
    if (a_min_pages) {
        MLIR_AttributeHandle aa = MLIR_CreateAttributeInteger(ctx,
            str_from_cstr_view((char *)"memory_min_pages"),
            MLIR_GetAttributeInteger(a_min_pages),
            MLIR_CreateTypeInteger(ctx, 32, true));
        mod_attrs[n_mod_attrs++] = aa;
    }

    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        mod_attrs, n_mod_attrs, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    // -----------------------------------------------------------------
    // First pass: walk import_global ops and assign each a fixed offset
    // in linmem starting at WASM_DATA_BASE (matches wasm-ld's default
    // global-base of 1024). Honour the global's align_pow attribute.
    // Emit one wmir.data_init at module level per global and build a
    // name -> offset table for the later wasmssa.addressof handler.
    //
    // We make TWO passes over globals:
    //   1. Assign offsets to all of them up-front.
    //   2. Emit data_init ops with `relocs` attribute applied to the
    //      init_data bytes (since a global's relocs may reference any
    //      other global, including ones declared later in source order).
    // -----------------------------------------------------------------
    // Note: WASM_DATA_BASE is a #define (not a local enum), because
    // tinyc — used during wasm self-hosting — only allows enum bodies
    // at module scope.
    //
    // For the wasm-lifter pipeline, import_globals may carry an
    // explicit `fixed_offset` attribute encoding the linmem byte
    // address chosen by wasm-ld. When present we honour that offset
    // verbatim (because the lifted code embeds it as `i32.const N`
    // operands to load/store). When absent we fall back to the
    // cursor-based layout used by the C-frontend pipeline.
    OffsetMap globals = {0};
    int32_t cursor = (int32_t)WASM_DATA_BASE;
    size_t n_top = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(top) != OP_TYPE_WASMSSA_IMPORT_GLOBAL) continue;
        string sn = at_s(top, "sym_name");
        string id = at_s(top, "init_data");
        int64_t sz = at_i(top, "size");
        int64_t ap = at_i(top, "align_pow");
        MLIR_AttributeHandle fa = MLIR_GetOpAttributeByName(top, "fixed_offset");
        if (sz <= 0) sz = (int64_t)id.size;
        int32_t align = (ap > 0) ? (int32_t)(1 << ap) : 1;
        if (fa) {
            int32_t fo = (int32_t)MLIR_GetAttributeInteger(fa);
            cursor = fo;
        } else {
            cursor = (cursor + align - 1) & ~(align - 1);
        }
        omap_add(&globals, sn, cursor);
        cursor += (int32_t)sz;
    }
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(top) != OP_TYPE_WASMSSA_IMPORT_GLOBAL) continue;
        string sn = at_s(top, "sym_name");
        string id = at_s(top, "init_data");
        string rl = at_s(top, "relocs");
        int32_t my_off = 0;
        (void)omap_get(&globals, sn, &my_off);
        // Make an editable copy of init_data so we can apply relocs.
        size_t cap = id.size;
        char *buf = (char *)malloc(cap > 0 ? cap : 1);
        if (cap > 0) memcpy(buf, id.str, cap);
        // Parse relocs of the form "off:target:addend,off:target:addend,..."
        // (commas separate entries; colons separate fields). Each entry
        // means: at local `off` within this global's bytes, store the
        // 32-bit LE address of `target + addend`.
        const char *p = rl.str;
        const char *e = rl.str + rl.size;
        while (p < e) {
            // skip leading commas/whitespace
            while (p < e && (*p == ',' || *p == ' ' || *p == '\t')) p++;
            if (p >= e) break;
            // parse offset
            long off_local = 0;
            while (p < e && *p >= '0' && *p <= '9') {
                off_local = off_local * 10 + (*p - '0'); p++;
            }
            if (p >= e || *p != ':') break;
            p++;
            // parse target name (until next ':')
            const char *tname = p;
            while (p < e && *p != ':') p++;
            size_t tlen = (size_t)(p - tname);
            if (p >= e || *p != ':') break;
            p++;
            // parse addend (may be negative)
            long addend = 0;
            int neg = 0;
            if (p < e && *p == '-') { neg = 1; p++; }
            while (p < e && *p >= '0' && *p <= '9') {
                addend = addend * 10 + (*p - '0'); p++;
            }
            if (neg) addend = -addend;
            string tn = { tname, tlen };
            int32_t toff = 0;
            if (!omap_get(&globals, tn, &toff)) {
                fprintf(stderr,
                    "wasmssa->wmir: reloc references unknown global '%.*s'\n",
                    (int)tlen, tname);
                free(buf);
                return MLIR_INVALID_HANDLE;
            }
            uint32_t val = (uint32_t)(toff + (int32_t)addend);
            if ((size_t)off_local + 4 > cap) {
                fprintf(stderr,
                    "wasmssa->wmir: reloc offset %ld out of range for "
                    "global '%.*s' (size %zu)\n",
                    off_local, (int)sn.size, sn.str, cap);
                free(buf);
                return MLIR_INVALID_HANDLE;
            }
            buf[off_local + 0] = (char)((val >>  0) & 0xff);
            buf[off_local + 1] = (char)((val >>  8) & 0xff);
            buf[off_local + 2] = (char)((val >> 16) & 0xff);
            buf[off_local + 3] = (char)((val >> 24) & 0xff);
        }
        MLIR_AttributeHandle a[3];
        a[0] = attr_s  (ctx, "sym_name",  sn.str, sn.size);
        a[1] = attr_i32(ctx, "offset",    my_off);
        a[2] = attr_s  (ctx, "init_data", buf, cap);
        MLIR_OpHandle di = MLIR_CreateOp(ctx, OP_TYPE_WMIR_DATA_INIT,
            op_type_to_string(OP_TYPE_WMIR_DATA_INIT),
            a, 3, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
            MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, out_body, di);
        // Intentionally leak `buf`: attr_s above does not copy, so the
        // memory must outlive this function.
    }

    // -----------------------------------------------------------------
    // Pre-pass over all wasmssa.func ops to build:
    //   - FuncSigMap: function name -> (param_types, result_types).
    //   - FuncPtrMap: function name -> slot index, assigned in the
    //     encounter order of wasmssa.func_addr targets across the
    //     entire module. Used by both wasmssa.func_addr lowering and
    //     by the synthesised __dispatch_<sig> functions below.
    // -----------------------------------------------------------------
    FuncSigMap sigs   = {0};
    FuncPtrMap fnptrs = {0};
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t == OP_TYPE_WASMSSA_FUNC) {
            string nm = at_s(top, "sym_name");
            string pt = at_s(top, "param_types");
            string rt = at_s(top, "result_types");
            fsm_add(&sigs, nm, pt, rt);
        }
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC) {
            string nm = at_s(top, "sym_name");
            string pt = at_s(top, "param_types");
            string rt = at_s(top, "result_types");
            fsm_add(&sigs, nm, pt, rt);
        }
    }
    // Walk every wasmssa.func_addr in the module to intern its target.
    // We walk the full op-tree of each function looking for func_addr.
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(top) != OP_TYPE_WASMSSA_FUNC) continue;
        if (MLIR_GetOpNumRegions(top) < 1) continue;
        MLIR_RegionHandle r = MLIR_GetOpRegion(top, 0);
        size_t nb = MLIR_GetRegionNumBlocks(r);
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(r, b);
            size_t nops = MLIR_GetBlockNumOps(blk);
            for (size_t o = 0; o < nops; o++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(blk, o);
                if (MLIR_GetOpType(op) == OP_TYPE_WASMSSA_FUNC_ADDR) {
                    string tgt = at_s(op, "target");
                    int32_t es = (int32_t)at_i_or(op, "slot", -1);
                    fpm_intern_with_slot(&fnptrs, tgt, es);
                }
                // Nested regions (block/loop/if) may also contain
                // func_addr ops. Recurse one level — wasmssa regions
                // only have one block before flattening.
                size_t nr = MLIR_GetOpNumRegions(op);
                for (size_t k = 0; k < nr; k++) {
                    MLIR_RegionHandle rr = MLIR_GetOpRegion(op, k);
                    size_t nrb = MLIR_GetRegionNumBlocks(rr);
                    for (size_t bb = 0; bb < nrb; bb++) {
                        MLIR_BlockHandle bl = MLIR_GetRegionBlock(rr, bb);
                        size_t nbo = MLIR_GetBlockNumOps(bl);
                        for (size_t bo = 0; bo < nbo; bo++) {
                            MLIR_OpHandle op2 = MLIR_GetBlockOp(bl, bo);
                            if (MLIR_GetOpType(op2) == OP_TYPE_WASMSSA_FUNC_ADDR) {
                                string tgt2 = at_s(op2, "target");
                                int32_t es2 = (int32_t)at_i_or(op2, "slot", -1);
                                fpm_intern_with_slot(&fnptrs, tgt2, es2);
                            }
                            // Recurse one more level for nested if/then/else.
                            size_t nr2 = MLIR_GetOpNumRegions(op2);
                            for (size_t k2 = 0; k2 < nr2; k2++) {
                                MLIR_RegionHandle rr2 = MLIR_GetOpRegion(op2, k2);
                                size_t nrb2 = MLIR_GetRegionNumBlocks(rr2);
                                for (size_t bb2 = 0; bb2 < nrb2; bb2++) {
                                    MLIR_BlockHandle bl2 = MLIR_GetRegionBlock(rr2, bb2);
                                    size_t nbo2 = MLIR_GetBlockNumOps(bl2);
                                    for (size_t bo2 = 0; bo2 < nbo2; bo2++) {
                                        MLIR_OpHandle op3 = MLIR_GetBlockOp(bl2, bo2);
                                        if (MLIR_GetOpType(op3) == OP_TYPE_WASMSSA_FUNC_ADDR) {
                                            string tgt3 = at_s(op3, "target");
                                            int32_t es3 = (int32_t)at_i_or(op3, "slot", -1);
                                            fpm_intern_with_slot(&fnptrs, tgt3, es3);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC ||
            t == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            continue;
        }
        if (t == OP_TYPE_WASMSSA_FUNC) {
            MLIR_OpHandle out_op = lower_func(ctx, top, &globals,
                                              &fnptrs, &sigs);
            if (!out_op) {
                free(globals.names); free(globals.offsets);
                free(fnptrs.names);  free(fnptrs.slots);
                free(sigs.names); free(sigs.param_types); free(sigs.result_types);
                return MLIR_INVALID_HANDLE;
            }
            MLIR_AppendBlockOp(ctx, out_body, out_op);
            continue;
        }
        string nm = MLIR_GetOpName(top);
        fprintf(stderr,
            "wmir lowering: unexpected top-level op '%.*s'\n",
            (int)nm.size, nm.str);
        free(globals.names); free(globals.offsets);
        free(fnptrs.names);  free(fnptrs.slots);
        free(sigs.names); free(sigs.param_types); free(sigs.result_types);
        return MLIR_INVALID_HANDLE;
    }

    // -----------------------------------------------------------------
    // Synthesise __dispatch_<sig> functions for every signature that
    // some call_indirect site uses. The dispatcher takes
    // (slot:i32, args...) and switches on slot to call the addressed
    // function with the matching signature.
    //
    // We generate one dispatcher per signature that appears in the
    // fnptr-addressed set. The pre-pass populated fnptrs in slot order;
    // we walk it once per signature and pick out the matching ones.
    // -----------------------------------------------------------------
    bool sig_emitted[64] = {0};
    for (size_t pi = 0; pi < fnptrs.n; pi++) {
        string pname = fnptrs.names[pi];
        string ppt, prt;
        if (!fsm_get(&sigs, pname, &ppt, &prt)) continue;

        // Skip if dispatcher for this signature already emitted.
        bool dup = false;
        for (size_t qi = 0; qi < pi; qi++) {
            string qpt, qrt;
            if (!fsm_get(&sigs, fnptrs.names[qi], &qpt, &qrt)) continue;
            if (qpt.size == ppt.size && qrt.size == prt.size &&
                memcmp(qpt.str, ppt.str, ppt.size) == 0 &&
                memcmp(qrt.str, prt.str, prt.size) == 0) {
                dup = true; break;
            }
        }
        if (dup) continue;
        if (pi < 64) sig_emitted[pi] = true;

        // Build the dispatcher: function name __dispatch_<pt>_<rt>.
        // Heap-allocated and intentionally leaked: attr_s does not copy
        // the string, and the name must outlive the attribute.
        size_t dname_cap = ppt.size + prt.size + 32;
        char *dname = (char *)malloc(dname_cap);
        int dlen = snprintf(dname, dname_cap,
            "__dispatch_%.*s_%.*s",
            (int)ppt.size, ppt.str,
            (int)prt.size, prt.str);

        // Parse the signature into a list of MLIR types for params.
        // Wasm value type bytes: 0x7f=i32, 0x7e=i64, 0x7d=f32, 0x7c=f64.
        // pt is the hex string e.g. "7f7f".
        MLIR_TypeHandle ptys[16];
        size_t n_pp = ppt.size / 2;
        if (n_pp > 15) {
            fprintf(stderr,
                "wmir: dispatcher signature has too many params\n");
            free(globals.names); free(globals.offsets);
            free(fnptrs.names);  free(fnptrs.slots);
            free(sigs.names); free(sigs.param_types); free(sigs.result_types);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_TypeHandle i32 = MLIR_CreateTypeInteger(ctx, 32, true);
        MLIR_TypeHandle i64 = MLIR_CreateTypeInteger(ctx, 64, true);
        ptys[0] = i32; // slot
        for (size_t k = 0; k < n_pp; k++) {
            unsigned hi = (unsigned)ppt.str[k * 2];
            unsigned lo = (unsigned)ppt.str[k * 2 + 1];
            // Parse one hex byte (2 chars).
            unsigned b = 0;
            #define HEX_NIB(c) ((c) >= '0' && (c) <= '9' ? (c) - '0' : \
                                (c) >= 'a' && (c) <= 'f' ? (c) - 'a' + 10 : \
                                (c) >= 'A' && (c) <= 'F' ? (c) - 'A' + 10 : 0)
            b = (HEX_NIB(hi) << 4) | HEX_NIB(lo);
            #undef HEX_NIB
            ptys[k + 1] = (b == 0x7e || b == 0x7c) ? i64 : i32;
        }
        size_t n_dp = n_pp + 1;
        MLIR_TypeHandle rty;
        bool has_result = (prt.size >= 2);
        if (has_result) {
            unsigned hi = (unsigned)prt.str[0];
            unsigned lo = (unsigned)prt.str[1];
            #define HEX_NIB(c) ((c) >= '0' && (c) <= '9' ? (c) - '0' : \
                                (c) >= 'a' && (c) <= 'f' ? (c) - 'a' + 10 : \
                                (c) >= 'A' && (c) <= 'F' ? (c) - 'A' + 10 : 0)
            unsigned b = (HEX_NIB(hi) << 4) | HEX_NIB(lo);
            #undef HEX_NIB
            rty = (b == 0x7e || b == 0x7c) ? i64 : i32;
        }

        // Build region with one block per slot test + call. We use one
        // entry block doing nested icmp+cond_br to a fall-through chain.
        MLIR_RegionHandle dreg = MLIR_CreateRegion(ctx);
        // Entry block: receives all params.
        MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, dreg, entry);
        MLIR_ValueHandle pvals[16];
        for (size_t k = 0; k < n_dp; k++) {
            pvals[k] = MLIR_CreateValueBlockArg(ctx, (string){0},
                (uint32_t)k, ptys[k],
                MLIR_CreateLocationUnknown(ctx, (string){0}));
            MLIR_AppendBlockArg(ctx, entry, pvals[k]);
        }

        // Build the chain of check / call / next blocks.
        MLIR_BlockHandle cur = entry;
        // Collect matching addressed funcs.
        size_t n_match = 0;
        size_t match_idx[64];
        for (size_t qi = 0; qi < fnptrs.n; qi++) {
            string qpt, qrt;
            if (!fsm_get(&sigs, fnptrs.names[qi], &qpt, &qrt)) continue;
            if (qpt.size != ppt.size || qrt.size != prt.size) continue;
            if (memcmp(qpt.str, ppt.str, ppt.size) != 0) continue;
            if (memcmp(qrt.str, prt.str, prt.size) != 0) continue;
            if (n_match >= 64) break;
            match_idx[n_match++] = qi;
        }

        for (size_t mi = 0; mi < n_match; mi++) {
            int32_t target_slot = fnptrs.slots[match_idx[mi]];
            string  target_name = fnptrs.names[match_idx[mi]];

            // %slot_const = wmir.const target_slot
            MLIR_ValueHandle slot_cst = MLIR_CreateValueOpResult(ctx,
                MLIR_INVALID_HANDLE, 0, i32, (string){0},
                MLIR_CreateLocationUnknown(ctx, (string){0}));
            {
                MLIR_AttributeHandle a[1];
                a[0] = MLIR_CreateAttributeInteger(ctx, str_lit("value"),
                    (int64_t)target_slot, i32);
                MLIR_TypeHandle rty1[1] = { i32 };
                MLIR_ValueHandle res1[1] = { slot_cst };
                MLIR_OpHandle cst = build_op_simple(ctx, OP_TYPE_WMIR_CONST,
                    a, 1, rty1, 1, res1, NULL, 0);
                MLIR_AppendBlockOp(ctx, cur, cst);
            }
            // %eq = wmir.icmp(eq, pvals[0], slot_cst) -> i32
            MLIR_ValueHandle eq = MLIR_CreateValueOpResult(ctx,
                MLIR_INVALID_HANDLE, 0, i32, (string){0},
                MLIR_CreateLocationUnknown(ctx, (string){0}));
            {
                MLIR_AttributeHandle a[1];
                a[0] = attr_s(ctx, "pred", "eq", 2);
                MLIR_TypeHandle rty1[1] = { i32 };
                MLIR_ValueHandle res1[1] = { eq };
                MLIR_ValueHandle ops1[2] = { pvals[0], slot_cst };
                MLIR_OpHandle ic = build_op_simple(ctx, OP_TYPE_WMIR_ICMP,
                    a, 1, rty1, 1, res1, ops1, 2);
                MLIR_AppendBlockOp(ctx, cur, ic);
            }
            MLIR_BlockHandle call_blk = MLIR_CreateBlock(ctx);
            MLIR_AppendRegionBlock(ctx, dreg, call_blk);
            MLIR_BlockHandle next_blk = MLIR_CreateBlock(ctx);
            MLIR_AppendRegionBlock(ctx, dreg, next_blk);

            // wmir.cond_br %eq, ^call_blk, ^next_blk
            MLIR_OpHandle cbr = build_op_cond_br(ctx, eq, call_blk, next_blk);
            MLIR_AppendBlockOp(ctx, cur, cbr);

            // call_blk: %r = wmir.call target(pvals[1..]); wmir.return %r
            MLIR_ValueHandle call_args[16];
            for (size_t k = 0; k + 1 < n_dp; k++) call_args[k] = pvals[k + 1];
            MLIR_OpHandle call_op;
            MLIR_ValueHandle call_res_v = MLIR_INVALID_HANDLE;
            {
                MLIR_AttributeHandle a[1];
                a[0] = attr_s(ctx, "target", target_name.str, target_name.size);
                MLIR_TypeHandle res_ty[1];
                MLIR_ValueHandle res_v[1];
                size_t nr = has_result ? 1 : 0;
                if (has_result) {
                    res_ty[0] = rty;
                    res_v[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE,
                        0, rty, (string){0},
                        MLIR_CreateLocationUnknown(ctx, (string){0}));
                    call_res_v = res_v[0];
                }
                call_op = build_op_simple(ctx, OP_TYPE_WMIR_CALL,
                    a, 1, res_ty, nr, res_v, call_args, n_dp - 1);
                MLIR_AppendBlockOp(ctx, call_blk, call_op);
            }
            // wmir.return [%r]
            if (has_result) {
                MLIR_ValueHandle ret_ops[1] = { call_res_v };
                MLIR_OpHandle ret_op = build_op_simple(ctx, OP_TYPE_WMIR_RETURN,
                    NULL, 0, NULL, 0, NULL, ret_ops, 1);
                MLIR_AppendBlockOp(ctx, call_blk, ret_op);
            } else {
                MLIR_OpHandle ret_op = build_op_simple(ctx, OP_TYPE_WMIR_RETURN,
                    NULL, 0, NULL, 0, NULL, NULL, 0);
                MLIR_AppendBlockOp(ctx, call_blk, ret_op);
            }

            cur = next_blk;
        }
        // Trailing block: unreachable (slot didn't match any target).
        {
            MLIR_OpHandle un = build_op_simple(ctx, OP_TYPE_WMIR_UNREACHABLE,
                NULL, 0, NULL, 0, NULL, NULL, 0);
            MLIR_AppendBlockOp(ctx, cur, un);
        }

        // Wrap as a wmir.func.
        MLIR_AttributeHandle fattrs[4];
        fattrs[0] = attr_s(ctx, "sym_name",     dname, (size_t)dlen);
        fattrs[1] = attr_s(ctx, "param_types",  "",    0);
        fattrs[2] = attr_s(ctx, "result_types", "",    0);
        fattrs[3] = attr_b(ctx, "exported",     false);
        MLIR_RegionHandle regs[1] = { dreg };
        MLIR_OpHandle dfunc = MLIR_CreateOp(ctx, OP_TYPE_WMIR_FUNC,
            op_type_to_string(OP_TYPE_WMIR_FUNC),
            fattrs, 4, NULL, 0, NULL, 0, NULL, 0, regs, 1,
            MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, out_body, dfunc);
    }

    free(globals.names); free(globals.offsets);
    free(fnptrs.names);  free(fnptrs.slots);
    free(sigs.names); free(sigs.param_types); free(sigs.result_types);
    return out_module;
}

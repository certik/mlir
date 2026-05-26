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
    return ty;
}

// =============================================================================
// SSA value remapping: each wasmssa value maps to its replacement in
// the wmir module. Linear scan; functions are small.
// =============================================================================
typedef struct {
    MLIR_ValueHandle *src;
    MLIR_ValueHandle *dst;
    size_t            n, cap;
} VMap;

static void vmap_set(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle v) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->src = (MLIR_ValueHandle *)realloc(m->src, m->cap * sizeof(*m->src));
        m->dst = (MLIR_ValueHandle *)realloc(m->dst, m->cap * sizeof(*m->dst));
    }
    m->src[m->n] = k;
    m->dst[m->n] = v;
    m->n++;
}
static int vmap_get(VMap *m, MLIR_ValueHandle k, MLIR_ValueHandle *out) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->src[i] == k) { *out = m->dst[i]; return 1; }
    }
    return 0;
}

// =============================================================================
// Frame stack for resolving wasmssa.br {depth=N}. Each enclosing
// block / loop / if pushes one frame; depth=0 refers to the innermost.
//   * For wasmssa.block / wasmssa.if: br targets the merge block.
//   * For wasmssa.loop:               br targets the loop header.
// The number of target block args is recorded so we can sanity-check
// operand counts when lowering wasmssa.br.
// =============================================================================
typedef struct {
    MLIR_BlockHandle target;
    size_t           n_args;
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

typedef struct {
    MLIR_Context     *ctx;
    MLIR_RegionHandle dst_region;
    MLIR_BlockHandle  cur;          // current insertion point
    VMap             *vmap;
    Frame            *frames;
    size_t            n_frames, c_frames;
    bool              cur_terminated; // last appended op was a terminator
    const OffsetMap  *globals;        // module-level data globals offset map
} Lowerer;

static void L_push_frame(Lowerer *L, MLIR_BlockHandle target,
                         size_t n_args, bool is_loop) {
    if (L->n_frames == L->c_frames) {
        L->c_frames = L->c_frames ? L->c_frames * 2 : 4;
        L->frames = (Frame *)realloc(L->frames, L->c_frames * sizeof(Frame));
    }
    L->frames[L->n_frames].target  = target;
    L->frames[L->n_frames].n_args  = n_args;
    L->frames[L->n_frames].is_loop = is_loop;
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
        MLIR_TypeHandle res_ty[1] = { MLIR_CreateTypeInteger(ctx, 32, true) };
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
        MLIR_ValueHandle ops[16];
        if (no > 16) {
            fprintf(stderr, "wmir: wasmssa.call with >16 args unsupported\n");
            return false;
        }
        for (size_t k = 0; k < no; k++) {
            if (!vmap_get(L->vmap, MLIR_GetOpOperand(src_op, k), &ops[k])) {
                fprintf(stderr, "wmir: unbound operand on wasmssa.call\n");
                return false;
            }
        }
        size_t nr = MLIR_GetOpNumResults(src_op);
        if (nr > 1) {
            fprintf(stderr, "wmir: wasmssa.call multi-result not yet supported\n");
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
        // wasmssa.br(%v...) {depth=D}: jump to frames[top-D].target.
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
        L_emit_br(L, f->target, args, no);
        return true;
    }

    case OP_TYPE_WASMSSA_BLOCK_RETURN: {
        // Terminator of the innermost wasmssa.block / .if region.
        // Targets the topmost frame's exit block.
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
        L_emit_br(L, f->target, args, no);
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
        L_push_frame(L, merge, n_results, /*is_loop=*/false);
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
        L_push_frame(L, merge, n_results, /*is_loop=*/false);
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
            L_push_frame(L, merge, n_results, /*is_loop=*/false);
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

        L->cur = header;
        L->cur_terminated = false;
        L_push_frame(L, header, n_loop_args, /*is_loop=*/true);
        if (!lower_region(L, loop_region)) {
            L_pop_frame(L);
            return false;
        }
        L_pop_frame(L);

        // Post-loop fall-through block. If the body terminated explicitly
        // (br/return/unreachable), this block is dead but harmless.
        MLIR_BlockHandle post = L_new_block(L);
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
                                const OffsetMap *globals) {
    string name      = at_s(src, "sym_name");
    bool   exported  = at_b(src, "exported");
    string pt        = at_s(src, "param_types");
    string rt        = at_s(src, "result_types");

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

    MLIR_AttributeHandle attrs[6];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",     name.str, name.size);
    attrs[na++] = attr_s(ctx, "param_types",  pt.str,   pt.size);
    attrs[na++] = attr_s(ctx, "result_types", rt.str,   rt.size);
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
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    // -----------------------------------------------------------------
    // First pass: walk import_global ops and assign each a fixed offset
    // in linmem starting at WASM_DATA_BASE (matches wasm-ld's default
    // global-base of 1024). Honour the global's align_pow attribute.
    // Emit one wmir.data_init at module level per global and build a
    // name -> offset table for the later wasmssa.addressof handler.
    // -----------------------------------------------------------------
    enum { WASM_DATA_BASE = 1024 };
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
        if (sz <= 0) sz = (int64_t)id.size;
        int32_t align = (ap > 0) ? (int32_t)(1 << ap) : 1;
        cursor = (cursor + align - 1) & ~(align - 1);
        omap_add(&globals, sn, cursor);
        MLIR_AttributeHandle a[3];
        a[0] = attr_s  (ctx, "sym_name",  sn.str, sn.size);
        a[1] = attr_i32(ctx, "offset",    cursor);
        a[2] = attr_s  (ctx, "init_data", id.str, id.size);
        MLIR_OpHandle di = MLIR_CreateOp(ctx, OP_TYPE_WMIR_DATA_INIT,
            op_type_to_string(OP_TYPE_WMIR_DATA_INIT),
            a, 3, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
            MLIR_CreateLocationUnknown(ctx, (string){0}),
            MLIR_INVALID_HANDLE, (string){0}, -1);
        MLIR_AppendBlockOp(ctx, out_body, di);
        cursor += (int32_t)sz;
    }

    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC ||
            t == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            continue;
        }
        if (t == OP_TYPE_WASMSSA_FUNC) {
            MLIR_OpHandle out_op = lower_func(ctx, top, &globals);
            if (!out_op) {
                free(globals.names); free(globals.offsets);
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
        return MLIR_INVALID_HANDLE;
    }
    free(globals.names); free(globals.offsets);
    return out_module;
}

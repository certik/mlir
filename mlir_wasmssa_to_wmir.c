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
// type used inside wmir.
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
        case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
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

typedef struct {
    MLIR_Context     *ctx;
    MLIR_RegionHandle dst_region;
    MLIR_BlockHandle  cur;          // current insertion point
    VMap             *vmap;
    Frame            *frames;
    size_t            n_frames, c_frames;
    bool              cur_terminated; // last appended op was a terminator
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
    }
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
        MLIR_OpType out_t;
        switch (opc) {
            case 0x6a: out_t = OP_TYPE_WMIR_IADD; break;
            case 0x6b: out_t = OP_TYPE_WMIR_ISUB; break;
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
        if (!(sz == 4 && vt == WT_I32)) {
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
        MLIR_AttributeHandle attrs[1] = { attr_i32(ctx, "memory_offset", off) };
        MLIR_TypeHandle res_ty[1] = { MLIR_CreateTypeInteger(ctx, 32, true) };
        MLIR_ValueHandle res[1] = {
            MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, res_ty[0],
                (string){0}, MLIR_CreateLocationUnknown(ctx, (string){0}))
        };
        MLIR_ValueHandle ops[1] = { addr };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_LOAD,
            attrs, 1, res_ty, 1, res, ops, 1);
        L_append(L, out);
        vmap_set(L->vmap, MLIR_GetOpResult(src_op, 0), res[0]);
        return true;
    }

    case OP_TYPE_WASMSSA_STORE: {
        int64_t off = at_i(src_op, "memory_offset");
        int64_t sz  = at_i(src_op, "mem_size_bytes");
        uint8_t vt  = (uint8_t)at_i(src_op, "valtype");
        if (!(sz == 4 && vt == WT_I32)) {
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
        MLIR_AttributeHandle attrs[1] = { attr_i32(ctx, "memory_offset", off) };
        MLIR_ValueHandle ops[2] = { addr, val };
        MLIR_OpHandle out = build_op_simple(ctx, OP_TYPE_WMIR_STORE,
            attrs, 1, NULL, 0, NULL, ops, 2);
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
            MLIR_TypeHandle rt = MLIR_GetValueType(rv);
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
            MLIR_TypeHandle ty = MLIR_GetValueType(src_res);
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
            MLIR_TypeHandle ty = MLIR_GetValueType(src_res);
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
            MLIR_TypeHandle ty = MLIR_GetValueType(sa);
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
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src) {
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

    size_t n_params = MLIR_GetBlockNumArgs(src_blk);
    for (size_t i = 0; i < n_params; i++) {
        MLIR_ValueHandle sa = MLIR_GetBlockArg(src_blk, i);
        MLIR_TypeHandle  ty = MLIR_GetValueType(sa);
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

    size_t n_top = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC ||
            t == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            continue;
        }
        if (t == OP_TYPE_WASMSSA_FUNC) {
            MLIR_OpHandle out_op = lower_func(ctx, top);
            if (!out_op) return MLIR_INVALID_HANDLE;
            MLIR_AppendBlockOp(ctx, out_body, out_op);
            continue;
        }
        string nm = MLIR_GetOpName(top);
        fprintf(stderr,
            "wmir lowering: unexpected top-level op '%.*s'\n",
            (int)nm.size, nm.str);
        return MLIR_INVALID_HANDLE;
    }
    return out_module;
}

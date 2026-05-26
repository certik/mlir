// wasmstack -> wasmssa lifter.
// See mlir_wasmstack_to_wasmssa.h for the public API and rationale.
//
// First-light scope (macho_exit): linear functions with no structured
// control flow. Walks each wasmstack.func's flat opcode sequence,
// simulating the operand stack and per-function local variables, and
// produces wasmssa SSA values.
//
// Block/loop/if support arrives in a later iteration.

#include "mlir_wasmstack_to_wasmssa.h"

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

#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// =============================================================================
// Attr / hex helpers (mirrors wasmssa_to_wasmstack).
// =============================================================================
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

static uint8_t hex_nibble(char c) {
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f') return (uint8_t)(10 + c - 'a');
    if (c >= 'A' && c <= 'F') return (uint8_t)(10 + c - 'A');
    return 0;
}
static uint8_t *hex_decode_arena(Arena *a, string s, size_t *out_n) {
    size_t n = s.size / 2;
    uint8_t *p = (uint8_t *)arena_alloc(a, n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        p[i] = (uint8_t)((hex_nibble(s.str[i*2]) << 4) | hex_nibble(s.str[i*2+1]));
    }
    *out_n = n;
    return p;
}

// =============================================================================
// Type mapping.
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

// =============================================================================
// Operand stack (SSA values).
// =============================================================================
typedef struct {
    MLIR_ValueHandle *data;
    size_t            n, cap;
} Stack;

static void st_push(Stack *s, MLIR_ValueHandle v) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->data = (MLIR_ValueHandle *)realloc(s->data, s->cap * sizeof(*s->data));
    }
    s->data[s->n++] = v;
}
static MLIR_ValueHandle st_pop(Stack *s) {
    if (s->n == 0) return MLIR_INVALID_HANDLE;
    return s->data[--s->n];
}
static MLIR_ValueHandle st_peek(Stack *s) {
    if (s->n == 0) return MLIR_INVALID_HANDLE;
    return s->data[s->n - 1];
}
static void st_free(Stack *s) { free(s->data); s->data = NULL; s->n = s->cap = 0; }

// =============================================================================
// Op build helpers.
// =============================================================================
static MLIR_OpHandle build_op_no_results(MLIR_Context *ctx, MLIR_OpType t,
                                         MLIR_AttributeHandle *attrs, size_t na,
                                         MLIR_ValueHandle *operands, size_t no) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, operands, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Build an op with a single result of `result_ty`. Returns the op's
// fresh SSA result via `out_result`.
static MLIR_OpHandle build_op_1res(MLIR_Context *ctx, MLIR_OpType t,
                                   MLIR_AttributeHandle *attrs, size_t na,
                                   MLIR_ValueHandle *operands, size_t no,
                                   MLIR_TypeHandle result_ty,
                                   MLIR_ValueHandle *out_result) {
    MLIR_TypeHandle rts[1] = { result_ty };
    MLIR_ValueHandle rvs[1];
    rvs[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
        result_ty, (string){0},
        MLIR_CreateLocationUnknown(ctx, (string){0}));
    MLIR_OpHandle op = MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, rts, 1, rvs, 1, operands, no, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    *out_result = rvs[0];
    return op;
}

// =============================================================================
// Lift one wasmstack.func to a wasmssa.func.
// =============================================================================
static MLIR_OpHandle lift_func(MLIR_Context *ctx, Arena *arena,
                               MLIR_OpHandle src, MLIR_BlockHandle siblings_blk) {
    string sym_name      = at_s(src, "sym_name");
    string pt_s          = at_s(src, "param_types");
    string rt_s          = at_s(src, "result_types");
    bool   exported      = at_b(src, "exported");
    bool   internal      = at_b(src, "internal");
    string local_types_s = at_s(src, "local_types");

    size_t np = 0, nr = 0, nl_total = 0;
    uint8_t *params  = hex_decode_arena(arena, pt_s, &np);
    uint8_t *results = hex_decode_arena(arena, rt_s, &nr);
    uint8_t *locals  = hex_decode_arena(arena, local_types_s, &nl_total);

    // Entry block with one block arg per function parameter.
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_ValueHandle *local_vals = (MLIR_ValueHandle *)arena_alloc(arena,
        (nl_total ? nl_total : 1) * sizeof(MLIR_ValueHandle));

    for (size_t i = 0; i < np; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, params[i]);
        MLIR_ValueHandle a = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
            MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, entry, a);
        local_vals[i] = a;
    }
    // Locals beyond the parameters start at zero.
    for (size_t i = np; i < nl_total; i++) {
        MLIR_AttributeHandle as[2];
        as[0] = attr_i32(ctx, "valtype", locals[i]);
        as[1] = attr_i64(ctx, "value", 0);
        MLIR_TypeHandle ty = vt_to_type(ctx, locals[i]);
        MLIR_ValueHandle rv;
        MLIR_AppendBlockOp(ctx, entry, build_op_1res(ctx,
            OP_TYPE_WASMSSA_CONST, as, 2, NULL, 0, ty, &rv));
        local_vals[i] = rv;
    }

    Stack S = {0};

    size_t n_ops = MLIR_GetBlockNumOps(MLIR_GetRegionBlock(MLIR_GetOpRegion(src, 0), 0));
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(src, 0), 0);

    bool emitted_terminator = false;
    for (size_t i = 0; i < n_ops && !emitted_terminator; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(src_blk, i);
        MLIR_OpType t = MLIR_GetOpType(bo);
        uint8_t valtype = (uint8_t)at_i(bo, "valtype");

        switch (t) {
        case OP_TYPE_WASMSTACK_LOCAL_GET: {
            uint32_t li = (uint32_t)at_i(bo, "local_idx");
            if (li >= nl_total) {
                fprintf(stderr, "wasmstack->wasmssa: local.get %u out of range (%zu)\n", li, nl_total);
                st_free(&S); return MLIR_INVALID_HANDLE;
            }
            st_push(&S, local_vals[li]);
            break;
        }
        case OP_TYPE_WASMSTACK_LOCAL_SET: {
            uint32_t li = (uint32_t)at_i(bo, "local_idx");
            MLIR_ValueHandle v = st_pop(&S);
            if (li >= nl_total) {
                fprintf(stderr, "wasmstack->wasmssa: local.set %u out of range\n", li);
                st_free(&S); return MLIR_INVALID_HANDLE;
            }
            local_vals[li] = v;
            break;
        }
        case OP_TYPE_WASMSTACK_LOCAL_TEE: {
            uint32_t li = (uint32_t)at_i(bo, "local_idx");
            MLIR_ValueHandle v = st_peek(&S);
            if (li >= nl_total) {
                fprintf(stderr, "wasmstack->wasmssa: local.tee %u out of range\n", li);
                st_free(&S); return MLIR_INVALID_HANDLE;
            }
            local_vals[li] = v;
            break;
        }
        case OP_TYPE_WASMSTACK_CONST: {
            int64_t v = at_i(bo, "value");
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", valtype);
            as[1] = attr_i64(ctx, "value", v);
            MLIR_TypeHandle ty = vt_to_type(ctx, valtype);
            MLIR_ValueHandle rv;
            MLIR_AppendBlockOp(ctx, entry, build_op_1res(ctx,
                OP_TYPE_WASMSSA_CONST, as, 2, NULL, 0, ty, &rv));
            st_push(&S, rv);
            break;
        }
        case OP_TYPE_WASMSTACK_RETURN: {
            // Pop `nr` values, emit wasmssa.return.
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(arena,
                (nr ? nr : 1) * sizeof(MLIR_ValueHandle));
            // Wasm stack order: values pushed in declaration order; rightmost = top.
            for (size_t k = nr; k > 0; k--) opnds[k - 1] = st_pop(&S);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, entry, build_op_no_results(ctx,
                OP_TYPE_WASMSSA_RETURN, as, 1, opnds, nr));
            emitted_terminator = true;
            break;
        }
        case OP_TYPE_WASMSTACK_CALL: {
            string tgt = at_s(bo, "target");
            // Resolve callee signature by scanning module-level siblings.
            string cpt = (string){0}, crt = (string){0};
            {
                size_t nm = MLIR_GetBlockNumOps(siblings_blk);
                for (size_t k = 0; k < nm; k++) {
                    MLIR_OpHandle cand = MLIR_GetBlockOp(siblings_blk, k);
                    string nm_s = at_s(cand, "sym_name");
                    if (nm_s.size == tgt.size &&
                        memcmp(nm_s.str, tgt.str, tgt.size) == 0) {
                        cpt = at_s(cand, "param_types");
                        crt = at_s(cand, "result_types");
                        break;
                    }
                }
            }
            if (cpt.str == NULL && crt.str == NULL && tgt.size > 0) {
                fprintf(stderr,
                    "wasmstack->wasmssa: call target '%.*s' not found\n",
                    (int)tgt.size, tgt.str);
                st_free(&S); return MLIR_INVALID_HANDLE;
            }
            size_t cnp = 0, cnr = 0;
            uint8_t *cp = hex_decode_arena(arena, cpt, &cnp);
            uint8_t *cr = hex_decode_arena(arena, crt, &cnr);
            // Pop `cnp` args (top of stack = last param).
            MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(arena,
                (cnp ? cnp : 1) * sizeof(MLIR_ValueHandle));
            for (size_t k = cnp; k > 0; k--) opnds[k - 1] = st_pop(&S);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_s(ctx, "target", tgt.str, tgt.size);
            if (cnr == 0) {
                MLIR_AppendBlockOp(ctx, entry, build_op_no_results(ctx,
                    OP_TYPE_WASMSSA_CALL, as, 2, opnds, cnp));
            } else if (cnr == 1) {
                MLIR_TypeHandle rty = vt_to_type(ctx, cr[0]);
                MLIR_ValueHandle rv;
                MLIR_AppendBlockOp(ctx, entry, build_op_1res(ctx,
                    OP_TYPE_WASMSSA_CALL, as, 2, opnds, cnp, rty, &rv));
                st_push(&S, rv);
            } else {
                fprintf(stderr, "wasmstack->wasmssa: multi-result call not yet supported\n");
                st_free(&S); return MLIR_INVALID_HANDLE;
            }
            break;
        }
        case OP_TYPE_WASMSTACK_END:
            // Function-level end: implicit return of any remaining stack values.
            if (!emitted_terminator) {
                MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(arena,
                    (nr ? nr : 1) * sizeof(MLIR_ValueHandle));
                for (size_t k = nr; k > 0; k--) opnds[k - 1] = st_pop(&S);
                MLIR_AttributeHandle as[1];
                as[0] = attr_i32(ctx, "valtype", 0);
                MLIR_AppendBlockOp(ctx, entry, build_op_no_results(ctx,
                    OP_TYPE_WASMSSA_RETURN, as, 1, opnds, nr));
                emitted_terminator = true;
            }
            break;
        case OP_TYPE_WASMSTACK_UNREACHABLE: {
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, entry, build_op_no_results(ctx,
                OP_TYPE_WASMSSA_UNREACHABLE, as, 1, NULL, 0));
            emitted_terminator = true;
            break;
        }
        default: {
            string nm = MLIR_GetOpName(bo);
            fprintf(stderr,
                "wasmstack->wasmssa: unsupported wasmstack op '%.*s' (enum=%d) in '%.*s'\n",
                (int)nm.size, nm.str, (int)t, (int)sym_name.size, sym_name.str);
            st_free(&S); return MLIR_INVALID_HANDLE;
        }
        }
    }
    if (!emitted_terminator) {
        // Defensive fallback: emit empty return.
        MLIR_ValueHandle *opnds = (MLIR_ValueHandle *)arena_alloc(arena,
            (nr ? nr : 1) * sizeof(MLIR_ValueHandle));
        for (size_t k = nr; k > 0; k--) opnds[k - 1] = st_pop(&S);
        MLIR_AttributeHandle as[1];
        as[0] = attr_i32(ctx, "valtype", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op_no_results(ctx,
            OP_TYPE_WASMSSA_RETURN, as, 1, opnds, nr));
    }
    st_free(&S);

    // Build the wasmssa.func wrapper.
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", sym_name.str, sym_name.size);
    attrs[na++] = attr_s(ctx, "param_types", pt_s.str, pt_s.size);
    attrs[na++] = attr_s(ctx, "result_types", rt_s.str, rt_s.size);
    attrs[na++] = attr_b(ctx, "exported", exported);
    attrs[na++] = attr_b(ctx, "internal", internal);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, entry);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_FUNC,
        op_type_to_string(OP_TYPE_WASMSSA_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Lift one wasmstack.import_func to a wasmssa.import_func.
// =============================================================================
static MLIR_OpHandle lift_import_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string sym_name = at_s(src, "sym_name");
    string pt_s = at_s(src, "param_types");
    string rt_s = at_s(src, "result_types");
    bool exported = at_b(src, "exported");
    MLIR_AttributeHandle attrs[4];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", sym_name.str, sym_name.size);
    attrs[na++] = attr_s(ctx, "param_types", pt_s.str, pt_s.size);
    attrs[na++] = attr_s(ctx, "result_types", rt_s.str, rt_s.size);
    attrs[na++] = attr_b(ctx, "exported", exported);
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
        op_type_to_string(OP_TYPE_WASMSSA_IMPORT_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Top-level entry point.
// =============================================================================
MLIR_OpHandle mlir_wasmstack_to_wasmssa(MLIR_Context *ctx,
                                        MLIR_OpHandle stack_module) {
    if (!stack_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(stack_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(stack_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);

    Arena *arena = MLIR_GetArenaAllocator(ctx);

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
        MLIR_OpHandle out_op = MLIR_INVALID_HANDLE;
        if (t == OP_TYPE_WASMSTACK_IMPORT_FUNC) {
            out_op = lift_import_func(ctx, top);
        } else if (t == OP_TYPE_WASMSTACK_FUNC) {
            out_op = lift_func(ctx, arena, top, mb);
        } else {
            string nm = MLIR_GetOpName(top);
            fprintf(stderr, "wasmstack->wasmssa: unexpected top-level op '%.*s'\n",
                    (int)nm.size, nm.str);
            return MLIR_INVALID_HANDLE;
        }
        if (!out_op) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, out_op);
    }
    return out_module;
}

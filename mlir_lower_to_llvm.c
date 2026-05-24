// Native lowering pass: rewrites a `builtin.module` from the high-level
// dialects (func, arith, cf, vector, memref, scf) down to the `llvm`
// dialect. Uses ONLY the public mlir_api.h surface — no upstream MLIR
// types — so this same translation unit is linked into both the
// upstream-backed and the native-backed builds.
//
// Strategy: walk every region top-down. For each op encountered, if it
// has a known lowering, build the replacement LLVM-dialect op(s),
// redirect uses of the old results, and erase the old op. Ops that are
// already in the LLVM dialect (or have no lowering needed) are left
// alone.
//
// This file is C, linked into both the upstream-backed and native-backed
// builds (no upstream MLIR headers), so the same translation unit
// supplies the agnostic `MLIR_LowerToLLVMDialect` in both binaries.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

static bool name_eq(string s, const char *cstr) {
    size_t n = 0;
    while (cstr[n]) n++;
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static MLIR_TypeHandle ty_i64(MLIR_Context *ctx) {
    return MLIR_CreateTypeInteger(ctx, 64, false);
}
static MLIR_TypeHandle ty_llvm_void(MLIR_Context *ctx) {
    return MLIR_CreateTypeLLVMVoid(ctx);
}

// Allocate a fresh value handle (op-result) that becomes the result of
// some op we are about to create.
static MLIR_ValueHandle make_result_value(MLIR_Context *ctx,
                                          MLIR_TypeHandle type,
                                          MLIR_LocationHandle loc) {
    return MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, type,
                                    str_lit(""), loc);
}

// Convenience for ops that take no successors and no special blocks. Use
// for everything except branch-like ops.
static MLIR_OpHandle create_simple_op(
        MLIR_Context *ctx, MLIR_OpType type, string opname,
        MLIR_AttributeHandle *attrs, size_t n_attrs,
        MLIR_TypeHandle *result_types, size_t n_result_types,
        MLIR_ValueHandle *results, size_t n_results,
        MLIR_ValueHandle *operands, size_t n_operands,
        MLIR_RegionHandle *regions, size_t n_regions,
        MLIR_LocationHandle loc) {
    return MLIR_CreateOp(ctx, type, opname,
                         attrs, n_attrs,
                         result_types, n_result_types,
                         results, n_results,
                         operands, n_operands,
                         regions, n_regions,
                         loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

// -----------------------------------------------------------------------------
// State shared across the whole lowering pass
// -----------------------------------------------------------------------------

typedef struct LowerState {
    MLIR_Context *ctx;
    MLIR_OpHandle module;
    MLIR_BlockHandle module_body;

    // vector.print runtime helpers, declared lazily once per pass.
    bool vp_decl_i64;
    bool vp_decl_newline;
    bool vp_decl_f32;
    bool vp_decl_f64;

    // When true, the walker preserves scf.* control-flow ops and skips
    // any cf->llvm rewrites: the wasm pipeline expects the post-lift
    // scf-form to survive into the wasmssa lowering. Set by
    // MLIR_LowerToLLVMDialectForWasm.
    bool keep_scf;
} LowerState;

// Append an `llvm.func` declaration (no body) to the module body. Used
// to declare runtime helpers like `printI64` referenced by the lowered
// `vector.print`. Always void-returning for now (covers our needs).
static void append_llvm_func_decl(LowerState *st, const char *name,
                                  MLIR_TypeHandle *params, size_t n_params) {
    MLIR_TypeHandle fn_ty = MLIR_CreateTypeLLVMFunction(
        st->ctx, ty_llvm_void(st->ctx), params, n_params, false);
    size_t name_len = 0; while (name[name_len]) name_len++;
    string name_str = str_from_cstr_len_view_const(name, name_len);
    MLIR_AttributeHandle attrs[2];
    attrs[0] = MLIR_CreateAttributeString(st->ctx, str_lit("sym_name"),
                                          name_str);
    attrs[1] = MLIR_CreateAttributeType(st->ctx, str_lit("function_type"),
                                        fn_ty);
    MLIR_RegionHandle empty_region = MLIR_CreateRegion(st->ctx);
    MLIR_OpHandle decl = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
        attrs, 2, NULL, 0, NULL, 0, NULL, 0,
        &empty_region, 1,
        MLIR_CreateLocationUnknown(st->ctx, str_lit("")));
    MLIR_AppendBlockOp(st->ctx, st->module_body, decl);
}

static void ensure_vp_decl_i64(LowerState *st) {
    if (st->vp_decl_i64) return;
    MLIR_TypeHandle params[1] = { ty_i64(st->ctx) };
    append_llvm_func_decl(st, "printI64", params, 1);
    st->vp_decl_i64 = true;
}

static void ensure_vp_decl_f32(LowerState *st) {
    if (st->vp_decl_f32) return;
    MLIR_TypeHandle params[1] = { MLIR_CreateTypeFloat(st->ctx, 32, false) };
    append_llvm_func_decl(st, "printF32", params, 1);
    st->vp_decl_f32 = true;
}

static void ensure_vp_decl_f64(LowerState *st) {
    if (st->vp_decl_f64) return;
    MLIR_TypeHandle params[1] = { MLIR_CreateTypeFloat(st->ctx, 64, false) };
    append_llvm_func_decl(st, "printF64", params, 1);
    st->vp_decl_f64 = true;
}

static void ensure_vp_decl_newline(LowerState *st) {
    if (st->vp_decl_newline) return;
    append_llvm_func_decl(st, "printNewline", NULL, 0);
    st->vp_decl_newline = true;
}

// -----------------------------------------------------------------------------
// Per-op lowering helpers
// -----------------------------------------------------------------------------

// Lower a `vector.print %x : T`. Sign-extends T to i64 (when needed),
// calls `printI64`, then `printNewline`. Matches what the upstream
// VectorToLLVM pass produces for tinyC's `vector.print<i32>`.
static bool lower_vector_print(LowerState *st, MLIR_OpHandle op,
                               MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_ValueHandle x = MLIR_GetOpOperand(op, 0);
    MLIR_TypeHandle xty = MLIR_GetValueType(x);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);

    string ts = MLIR_GetTypeString(st->ctx, xty);
    bool is_f32 = ts.size == 3 && ts.str[0] == 'f' && ts.str[1] == '3' && ts.str[2] == '2';
    bool is_f64 = ts.size == 3 && ts.str[0] == 'f' && ts.str[1] == '6' && ts.str[2] == '4';

    const char *callee;
    MLIR_ValueHandle arg = x;
    if (is_f32) {
        ensure_vp_decl_f32(st);
        callee = "printF32";
    } else if (is_f64) {
        ensure_vp_decl_f64(st);
        callee = "printF64";
    } else {
        ensure_vp_decl_i64(st);
        callee = "printI64";
        // Sign-extend integers to i64 unless already i64.
        if (MLIR_IsTypeInteger(xty) &&
            !(ts.size == 3 && ts.str[0] == 'i' && ts.str[1] == '6' && ts.str[2] == '4')) {
            MLIR_TypeHandle i64 = ty_i64(st->ctx);
            MLIR_ValueHandle res = make_result_value(st->ctx, i64, loc);
            MLIR_TypeHandle rts[1] = { i64 };
            MLIR_ValueHandle results[1] = { res };
            MLIR_ValueHandle ops[1] = { x };
            MLIR_OpHandle sext = create_simple_op(
                st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.sext"),
                NULL, 0, rts, 1, results, 1, ops, 1, NULL, 0, loc);
            MLIR_InsertBlockOpAtIndex(st->ctx, parent, sext, pos++);
            arg = res;
        }
    }
    ensure_vp_decl_newline(st);

    {
        MLIR_AttributeHandle attrs[1];
        size_t cl = 0; while (callee[cl]) cl++;
        attrs[0] = MLIR_CreateAttributeSymbolRef(
            st->ctx, str_lit("callee"),
            str_from_cstr_len_view_const(callee, cl));
        MLIR_ValueHandle ops[1] = { arg };
        MLIR_OpHandle call = create_simple_op(
            st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
            attrs, 1, NULL, 0, NULL, 0, ops, 1, NULL, 0, loc);
        MLIR_InsertBlockOpAtIndex(st->ctx, parent, call, pos++);
    }

    {
        MLIR_AttributeHandle attrs[1];
        attrs[0] = MLIR_CreateAttributeSymbolRef(
            st->ctx, str_lit("callee"), str_lit("printNewline"));
        MLIR_OpHandle call = create_simple_op(
            st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
            attrs, 1, NULL, 0, NULL, 0, NULL, 0, NULL, 0, loc);
        MLIR_InsertBlockOpAtIndex(st->ctx, parent, call, pos++);
    }

    return true;
}

// Lower an `arith.constant value : T` to `llvm.mlir.constant(value : T) : T`.
// Same operand/result shape — just re-create with the right name and the
// "value" attribute kept as-is (the original op's attribute is already
// named "value", but its name in the user-attr table on the upstream
// backend is what matters).
static bool lower_arith_constant(LowerState *st, MLIR_OpHandle op,
                                 MLIR_BlockHandle parent, size_t pos) {
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle old_res = MLIR_GetOpResult(op, 0);
    MLIR_TypeHandle ty = MLIR_GetValueType(old_res);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);

    // Find the "value" attribute on the source op.
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle val_attr = MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "value")) { val_attr = a; break; }
    }
    if (val_attr == MLIR_INVALID_HANDLE) return false;

    MLIR_ValueHandle new_res = make_result_value(st->ctx, ty, loc);
    MLIR_AttributeHandle attrs[1] = { val_attr };
    MLIR_TypeHandle rts[1] = { ty };
    MLIR_ValueHandle results[1] = { new_res };
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_MLIR_CONSTANT, str_lit("llvm.mlir.constant"),
        attrs, 1, rts, 1, results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    MLIR_ReplaceAllUsesOfValue(st->ctx, old_res, new_res);
    return true;
}

// Lower `func.return [%v ...]` to `llvm.return [%v ...]` (1:1).
static bool lower_func_return(LowerState *st, MLIR_OpHandle op,
                              MLIR_BlockHandle parent, size_t pos) {
    size_t no = MLIR_GetOpNumOperands(op);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_ValueHandle *operands = NULL;
    MLIR_ValueHandle stk[4];
    if (no <= 4) operands = stk;
    else operands = (MLIR_ValueHandle *)arena_alloc(
        MLIR_GetArenaAllocator(st->ctx), no * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_LLVM_RETURN, str_lit("llvm.return"),
        NULL, 0, NULL, 0, NULL, 0, operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    return true;
}

// Lower a `func.func` to an `llvm.func`, taking ownership of its body
// region so block args, ops, and edges are preserved verbatim.
static bool lower_func_func(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos) {
    // Pull the body region off (may be empty for declarations).
    size_t nr = MLIR_GetOpNumRegions(op);
    MLIR_RegionHandle body = (nr > 0)
        ? MLIR_TakeOpRegion(st->ctx, op, 0)
        : MLIR_CreateRegion(st->ctx);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);

    // Copy attributes that matter: sym_name, function_type, sym_visibility,
    // arg_attrs, res_attrs. We don't need `passthrough` at this stage.
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle attrs_buf[16];
    size_t n_attrs = 0;
    for (size_t i = 0; i < na && n_attrs < 16; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(a);
        if (name_eq(an, "sym_name") ||
            name_eq(an, "sym_visibility") ||
            name_eq(an, "arg_attrs") ||
            name_eq(an, "res_attrs") ||
            name_eq(an, "llvm.linkage") ||
            name_eq(an, "wasm.import_module") ||
            name_eq(an, "wasm.import_name") ||
            name_eq(an, "wasm.export_name")) {
            attrs_buf[n_attrs++] = a;
        } else if (name_eq(an, "function_type")) {
            // Convert FunctionType -> LLVMFunctionType (which carries
            // is_var_arg and uses LLVM void instead of zero-results).
            MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(a);
            size_t ni = MLIR_GetTypeFunctionNumInputs(ft);
            size_t nr_ = MLIR_GetTypeFunctionNumResults(ft);
            MLIR_TypeHandle ins_stk[16];
            MLIR_TypeHandle *ins = ins_stk;
            if (ni > 16) ins = (MLIR_TypeHandle *)arena_alloc(
                MLIR_GetArenaAllocator(st->ctx),
                ni * sizeof(MLIR_TypeHandle));
            for (size_t k = 0; k < ni; k++) ins[k] = MLIR_GetTypeFunctionInput(ft, k);
            MLIR_TypeHandle ret_ty = (nr_ == 0)
                ? ty_llvm_void(st->ctx)
                : MLIR_GetTypeFunctionResult(ft, 0);
            MLIR_TypeHandle llvmft = MLIR_CreateTypeLLVMFunction(
                st->ctx, ret_ty, ins, ni, false);
            attrs_buf[n_attrs++] = MLIR_CreateAttributeType(
                st->ctx, str_lit("function_type"), llvmft);
        }
    }

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
        attrs_buf, n_attrs, NULL, 0, NULL, 0, NULL, 0,
        &body, 1, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    return true;
}

// Lower `func.call @callee(args) : (...) -> (...)` to `llvm.call`.
static bool lower_func_call(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);

    MLIR_ValueHandle stk[8];
    MLIR_ValueHandle *operands = stk;
    if (no > 8) operands = (MLIR_ValueHandle *)arena_alloc(
        MLIR_GetArenaAllocator(st->ctx), no * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_TypeHandle rts_stk[2];
    MLIR_ValueHandle res_stk[2];
    MLIR_TypeHandle *rts = (nr <= 2) ? rts_stk
        : (MLIR_TypeHandle *)arena_alloc(MLIR_GetArenaAllocator(st->ctx),
                                         nr * sizeof(MLIR_TypeHandle));
    MLIR_ValueHandle *results = (nr <= 2) ? res_stk
        : (MLIR_ValueHandle *)arena_alloc(MLIR_GetArenaAllocator(st->ctx),
                                          nr * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        rts[i] = MLIR_GetValueType(old);
        results[i] = make_result_value(st->ctx, rts[i], loc);
    }

    // Copy `callee` attribute through.
    size_t na = MLIR_GetOpNumAttributes(op);
    MLIR_AttributeHandle attrs_buf[2];
    size_t n_attrs = 0;
    for (size_t i = 0; i < na && n_attrs < 2; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "callee")) {
            attrs_buf[n_attrs++] = a;
        }
    }

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
        attrs_buf, n_attrs, rts, nr, results, nr,
        operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, results[i]);
    }
    return true;
}

// Generic "rename op": build a new op with `new_name` that copies all
// operands, results, and attributes from `op` 1:1. Used for the family
// of arith→llvm conversions where the semantics are identical and only
// the op name changes (arith.addi→llvm.add, arith.cmpi→llvm.icmp, ...).
static bool lower_rename(LowerState *st, MLIR_OpHandle op,
                         MLIR_BlockHandle parent, size_t pos,
                         string new_name, MLIR_OpType new_type) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);
    size_t na = MLIR_GetOpNumAttributes(op);

    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle *operands = no ? (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_TypeHandle *rts = nr ? (MLIR_TypeHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_TypeHandle)) : NULL;
    MLIR_ValueHandle *results = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        rts[i] = MLIR_GetValueType(old);
        results[i] = make_result_value(st->ctx, rts[i], loc);
    }

    MLIR_AttributeHandle *attrs = na ? (MLIR_AttributeHandle *)arena_alloc(
        alloc, na * sizeof(MLIR_AttributeHandle)) : NULL;
    for (size_t i = 0; i < na; i++) attrs[i] = MLIR_GetOpAttribute(op, i);

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, new_type, new_name,
        attrs, na, rts, nr, results, nr,
        operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, results[i]);
    }
    return true;
}

// Lower `cf.br` / `cf.cond_br` to `llvm.br` / `llvm.cond_br`. These ops
// carry block successors plus per-successor operand lists; we have to
// use the successor-aware op constructor.
static bool lower_cf_branch(LowerState *st, MLIR_OpHandle op,
                            MLIR_BlockHandle parent, size_t pos,
                            string new_name) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t na = MLIR_GetOpNumAttributes(op);
    size_t ns = MLIR_GetOpNumSuccessors(op);

    MLIR_ValueHandle *operands = no ? (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_AttributeHandle *attrs = na ? (MLIR_AttributeHandle *)arena_alloc(
        alloc, na * sizeof(MLIR_AttributeHandle)) : NULL;
    for (size_t i = 0; i < na; i++) attrs[i] = MLIR_GetOpAttribute(op, i);

    MLIR_BlockHandle *succs = ns ? (MLIR_BlockHandle *)arena_alloc(
        alloc, ns * sizeof(MLIR_BlockHandle)) : NULL;
    MLIR_ValueHandle **succ_ops = ns ? (MLIR_ValueHandle **)arena_alloc(
        alloc, ns * sizeof(MLIR_ValueHandle *)) : NULL;
    size_t *n_succ_ops = ns ? (size_t *)arena_alloc(
        alloc, ns * sizeof(size_t)) : NULL;
    for (size_t s = 0; s < ns; s++) {
        succs[s] = MLIR_GetOpSuccessor(op, s);
        size_t k = MLIR_GetOpNumSuccessorOperands(op, s);
        n_succ_ops[s] = k;
        succ_ops[s] = k ? (MLIR_ValueHandle *)arena_alloc(
            alloc, k * sizeof(MLIR_ValueHandle)) : NULL;
        for (size_t j = 0; j < k; j++) {
            succ_ops[s][j] = MLIR_GetOpSuccessorOperand(op, s, j);
        }
    }

    MLIR_OpHandle nop = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, new_name,
        attrs, na, NULL, 0, NULL, 0,
        operands, no, NULL, 0,
        succs, ns, succ_ops, n_succ_ops,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    return true;
}

// Lower `func.constant @sym : T` to `llvm.mlir.addressof @sym : !llvm.ptr`.
static bool lower_func_constant(LowerState *st, MLIR_OpHandle op,
                                MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    if (MLIR_GetOpNumResults(op) != 1) return false;

    // Find the symbol-ref `value` attribute and extract its name as a
    // C string.
    size_t na = MLIR_GetOpNumAttributes(op);
    string sym_name = {0};
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (name_eq(MLIR_GetAttributeName(a), "value")) {
            string s = MLIR_GetAttributeAsString(st->ctx, a);
            // Format is "@name" (FlatSymbolRefAttr) — strip leading '@'.
            if (s.size > 0 && s.str[0] == '@') {
                sym_name.str = s.str + 1;
                sym_name.size = s.size - 1;
            } else {
                sym_name = s;
            }
            break;
        }
    }
    if (sym_name.size == 0) return false;

    MLIR_TypeHandle ptr_ty = MLIR_CreateTypeLLVMPointer(st->ctx);
    MLIR_ValueHandle new_res = make_result_value(st->ctx, ptr_ty, loc);
    MLIR_TypeHandle rts[1] = { ptr_ty };
    MLIR_ValueHandle results[1] = { new_res };
    MLIR_AttributeHandle attrs[1];
    attrs[0] = MLIR_CreateAttributeSymbolRef(
        st->ctx, str_lit("global_name"), sym_name);
    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.mlir.addressof"),
        attrs, 1, rts, 1, results, 1, NULL, 0, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    MLIR_ReplaceAllUsesOfValue(st->ctx, MLIR_GetOpResult(op, 0), new_res);
    return true;
}

// Lower `builtin.unrealized_conversion_cast %x : T1 to T2`. After our
// other lowerings produce LLVM-dialect ops at both endpoints, all
// remaining casts are no-op type punning (typically !llvm.ptr↔function
// type around indirect calls). Just RAUW the result with the operand
// and let the downstream consumer use the operand directly.
static bool lower_unrealized_cast(LowerState *st, MLIR_OpHandle op,
                                  MLIR_BlockHandle parent, size_t pos) {
    (void)st; (void)parent; (void)pos;
    if (MLIR_GetOpNumOperands(op) != 1 || MLIR_GetOpNumResults(op) != 1) {
        return false;
    }
    MLIR_ReplaceAllUsesOfValue(st->ctx,
                               MLIR_GetOpResult(op, 0),
                               MLIR_GetOpOperand(op, 0));
    return true;
}

// Lower `func.call_indirect %fn(args)` to `llvm.call %fn(args)`. The
// callee is operand[0] (a !llvm.ptr after prior lowerings); remaining
// operands are the arguments.
static bool lower_func_call_indirect(LowerState *st, MLIR_OpHandle op,
                                     MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    size_t no = MLIR_GetOpNumOperands(op);
    size_t nr = MLIR_GetOpNumResults(op);
    if (no < 1) return false;

    MLIR_ValueHandle *operands = (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(op, i);

    MLIR_TypeHandle *rts = nr ? (MLIR_TypeHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_TypeHandle)) : NULL;
    MLIR_ValueHandle *results = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        rts[i] = MLIR_GetValueType(old);
        results[i] = make_result_value(st->ctx, rts[i], loc);
    }

    // Build a `var_callee_type` LLVMFunctionType attribute describing
    // the indirect call's signature (return + arg types).
    MLIR_TypeHandle ret_ty = (nr == 0) ? ty_llvm_void(st->ctx) : rts[0];
    size_t n_args = no - 1;
    MLIR_TypeHandle *arg_tys = n_args ? (MLIR_TypeHandle *)arena_alloc(
        alloc, n_args * sizeof(MLIR_TypeHandle)) : NULL;
    for (size_t i = 0; i < n_args; i++) {
        arg_tys[i] = MLIR_GetValueType(operands[i + 1]);
    }
    MLIR_TypeHandle llvmft = MLIR_CreateTypeLLVMFunction(
        st->ctx, ret_ty, arg_tys, n_args, false);
    MLIR_AttributeHandle attrs[1];
    attrs[0] = MLIR_CreateAttributeType(
        st->ctx, str_lit("var_callee_type"), llvmft);

    MLIR_OpHandle nop = create_simple_op(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
        attrs, 1, rts, nr, results, nr,
        operands, no, NULL, 0, loc);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, nop, pos);
    for (size_t i = 0; i < nr; i++) {
        MLIR_ReplaceAllUsesOfValue(st->ctx, MLIR_GetOpResult(op, i), results[i]);
    }
    return true;
}

// Replace the trailing `scf.yield %vs` of `blk` with `llvm.br ^cont(%vs)`.
static void rewrite_yield_to_br(LowerState *st, MLIR_BlockHandle blk,
                                MLIR_BlockHandle cont) {
    size_t n = MLIR_GetBlockNumOps(blk);
    if (n == 0) return;
    MLIR_OpHandle term = MLIR_GetBlockOp(blk, n - 1);
    if (!name_eq(MLIR_GetOpName(term), "scf.yield")) return;
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    size_t no = MLIR_GetOpNumOperands(term);
    MLIR_ValueHandle *operands = no ? (MLIR_ValueHandle *)arena_alloc(
        alloc, no * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < no; i++) operands[i] = MLIR_GetOpOperand(term, i);
    MLIR_BlockHandle succs[1] = { cont };
    MLIR_ValueHandle *succ_ops_arr[1] = { operands };
    size_t n_succ_ops_arr[1] = { no };
    MLIR_LocationHandle term_loc = MLIR_GetOpLocation(term);
    MLIR_OpHandle br = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.br"),
        NULL, 0, NULL, 0, NULL, 0,
        NULL, 0, NULL, 0,
        succs, 1, succ_ops_arr, n_succ_ops_arr,
        term_loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_AppendBlockOp(st->ctx, blk, br);
    MLIR_EraseOp(st->ctx, term);
}

// Lower `scf.if %cond -> (Ts) { thenRegion } else { elseRegion }` to a
// CFG: the parent block ends with `llvm.cond_br %cond, ^then, ^else`;
// then/else regions are inlined into the parent region; their terminating
// `scf.yield %vs` becomes `llvm.br ^cont(%vs)`; `cont` is a fresh block
// holding everything that came after the scf.if in the parent block,
// with block args matching the scf.if's results (used to RAUW them).
//
// Limitations: only handles single-block regions (which is what tinyC's
// `if`/`else` produce — scf.if regions with no nested control flow).
static bool lower_scf_if(LowerState *st, MLIR_OpHandle op,
                         MLIR_BlockHandle parent, size_t pos) {
    MLIR_LocationHandle loc = MLIR_GetOpLocation(op);
    MLIR_RegionHandle parent_region = MLIR_GetBlockParentRegion(parent);
    if (parent_region == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumOperands(op) < 1) return false;
    if (MLIR_GetOpNumRegions(op) < 1) return false;

    MLIR_ValueHandle cond = MLIR_GetOpOperand(op, 0);
    size_t nr = MLIR_GetOpNumResults(op);

    MLIR_RegionHandle then_reg = MLIR_TakeOpRegion(st->ctx, op, 0);
    MLIR_RegionHandle else_reg = MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(op) >= 2) {
        else_reg = MLIR_TakeOpRegion(st->ctx, op, 1);
    }
    if (then_reg == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetRegionNumBlocks(then_reg) < 1) return false;
    bool has_else = (else_reg != MLIR_INVALID_HANDLE) &&
                    (MLIR_GetRegionNumBlocks(else_reg) >= 1);

    // Entry blocks are first block of each region; the branch target.
    MLIR_BlockHandle then_blk = MLIR_GetRegionBlock(then_reg, 0);
    MLIR_BlockHandle else_blk = has_else ? MLIR_GetRegionBlock(else_reg, 0)
                                         : MLIR_INVALID_HANDLE;

    // Continuation block with one block-arg per scf.if result.
    MLIR_BlockHandle cont = MLIR_CreateBlock(st->ctx);
    Arena *alloc = MLIR_GetArenaAllocator(st->ctx);
    MLIR_ValueHandle *cont_args = nr ? (MLIR_ValueHandle *)arena_alloc(
        alloc, nr * sizeof(MLIR_ValueHandle)) : NULL;
    for (size_t i = 0; i < nr; i++) {
        MLIR_ValueHandle old = MLIR_GetOpResult(op, i);
        MLIR_TypeHandle ty = MLIR_GetValueType(old);
        cont_args[i] = MLIR_CreateValueBlockArg(
            st->ctx, str_lit(""), (uint32_t)i, ty, loc);
        MLIR_AppendBlockArg(st->ctx, cont, cont_args[i]);
        MLIR_ReplaceAllUsesOfValue(st->ctx, old, cont_args[i]);
    }

    // Move all ops AFTER the scf.if from parent to cont.
    while (MLIR_GetBlockNumOps(parent) > pos + 1) {
        MLIR_OpHandle tail = MLIR_GetBlockOp(parent, pos + 1);
        MLIR_MoveOpToBlockEnd(st->ctx, tail, cont);
    }

    // Rewrite scf.yield in any block of the regions to llvm.br ^cont(args).
    // Inner control-flow lowering may have produced multi-block regions,
    // but scf.yield only ever appears as a block terminator, so walking
    // the blocks of the moved region and matching on the terminator is
    // sufficient.
    size_t nthen = MLIR_GetRegionNumBlocks(then_reg);
    for (size_t bi = 0; bi < nthen; bi++) {
        rewrite_yield_to_br(st, MLIR_GetRegionBlock(then_reg, bi), cont);
    }
    if (has_else) {
        size_t nelse = MLIR_GetRegionNumBlocks(else_reg);
        for (size_t bi = 0; bi < nelse; bi++) {
            rewrite_yield_to_br(st, MLIR_GetRegionBlock(else_reg, bi), cont);
        }
    }

    // Move all blocks of each region into parent_region (in original order).
    // Each MoveBlockToRegionEnd removes from src region, so re-fetch index 0.
    while (MLIR_GetRegionNumBlocks(then_reg) > 0) {
        MLIR_MoveBlockToRegionEnd(st->ctx,
            MLIR_GetRegionBlock(then_reg, 0), parent_region);
    }
    if (has_else) {
        while (MLIR_GetRegionNumBlocks(else_reg) > 0) {
            MLIR_MoveBlockToRegionEnd(st->ctx,
                MLIR_GetRegionBlock(else_reg, 0), parent_region);
        }
    }
    MLIR_MoveBlockToRegionEnd(st->ctx, cont, parent_region);

    MLIR_BlockHandle false_target = has_else ? else_blk : cont;
    MLIR_BlockHandle succs[2] = { then_blk, false_target };
    MLIR_ValueHandle *empty_ops[2] = { NULL, NULL };
    size_t n_empty_ops[2] = { 0, 0 };
    MLIR_ValueHandle cond_arr[1] = { cond };
    MLIR_OpHandle cbr = MLIR_CreateOpWithSuccessors(
        st->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.cond_br"),
        NULL, 0, NULL, 0, NULL, 0,
        cond_arr, 1, NULL, 0,
        succs, 2, empty_ops, n_empty_ops,
        loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
    MLIR_InsertBlockOpAtIndex(st->ctx, parent, cbr, pos);
    return true;
}

// Return values from try_lower_op:
//   LOWER_NONE      — op not handled; walker should leave it in place.
//   LOWER_REPLACED  — lowering inserted N replacement ops at position
//                     `pos` and the original op is still attached at the
//                     end of that run. Walker erases original and skips
//                     past the inserts.
//   LOWER_DONE_BLOCK — lowering rewrote the block's tail (e.g. scf.if
//                     turned the block into a cond_br); the original op
//                     has already been erased and there is nothing more
//                     to walk in this block.
typedef enum {
    LOWER_NONE = 0,
    LOWER_REPLACED,
    LOWER_DONE_BLOCK,
} LowerResult;

// Returns one of LowerResult values. See enum docs above.
static int try_lower_op(LowerState *st, MLIR_OpHandle op,
                        MLIR_BlockHandle parent, size_t pos) {
    string name = MLIR_GetOpName(op);
    if (name_eq(name, "scf.if")) {
        if (st->keep_scf) return LOWER_NONE;
        // scf.if rewrites the parent block's tail: it erases its own
        // original op (after taking its regions and uses) — actually it
        // doesn't erase here; we handle that in the walker by checking
        // whether the original is still in the block. Easier: have
        // scf.if erase its own original and return DONE_BLOCK.
        if (lower_scf_if(st, op, parent, pos)) {
            MLIR_EraseOp(st->ctx, op);
            return LOWER_DONE_BLOCK;
        }
        return LOWER_NONE;
    }
    bool ok = false;
    if (name_eq(name, "func.func"))      ok = lower_func_func(st, op, parent, pos);
    else if (name_eq(name, "func.return") ||
             name_eq(name, "return"))    ok = lower_func_return(st, op, parent, pos);
    else if (name_eq(name, "func.call")) ok = lower_func_call(st, op, parent, pos);
    else if (name_eq(name, "func.constant")) ok = lower_func_constant(st, op, parent, pos);
    else if (name_eq(name, "func.call_indirect")) ok = lower_func_call_indirect(st, op, parent, pos);
    else if (name_eq(name, "builtin.unrealized_conversion_cast") ||
             name_eq(name, "unrealized_conversion_cast"))
                                         ok = lower_unrealized_cast(st, op, parent, pos);
    else if (name_eq(name, "arith.constant")) ok = lower_arith_constant(st, op, parent, pos);
    else if (name_eq(name, "vector.print"))   ok = lower_vector_print(st, op, parent, pos);
    else if (name_eq(name, "arith.addi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.add"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.subi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.sub"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.muli"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.mul"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.divsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.sdiv"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.divui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.udiv"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.remsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.srem"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.remui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.urem"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.andi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.and"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.ori"))   ok = lower_rename(st, op, parent, pos, str_lit("llvm.or"),   OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.xori"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.xor"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.shli"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.shl"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.shrsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.ashr"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.shrui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.lshr"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.addf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fadd"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.subf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fsub"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.mulf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fmul"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.divf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fdiv"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.cmpi"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.icmp"), OP_TYPE_LLVM_ICMP);
    else if (name_eq(name, "arith.cmpf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fcmp"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.select"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.select"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.extsi")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.sext"),   OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.extui")) ok = lower_rename(st, op, parent, pos, str_lit("llvm.zext"),   OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.trunci"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.trunc"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.extf"))  ok = lower_rename(st, op, parent, pos, str_lit("llvm.fpext"),  OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.truncf"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.fptrunc"),OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.sitofp"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.sitofp"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.uitofp"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.uitofp"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.fptosi"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.fptosi"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.fptoui"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.fptoui"), OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "arith.bitcast"))ok = lower_rename(st, op, parent, pos, str_lit("llvm.bitcast"),OP_TYPE_UNREGISTERED);
    else if (name_eq(name, "cf.br"))      ok = st->keep_scf ? false : lower_cf_branch(st, op, parent, pos, str_lit("llvm.br"));
    else if (name_eq(name, "cf.cond_br")) ok = st->keep_scf ? false : lower_cf_branch(st, op, parent, pos, str_lit("llvm.cond_br"));

    return ok ? LOWER_REPLACED : LOWER_NONE;
}

static void walk_block(LowerState *st, MLIR_BlockHandle block);

static void walk_op(LowerState *st, MLIR_OpHandle op) {
    size_t nr = MLIR_GetOpNumRegions(op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(op, r);
        // Recompute block count each iteration: lowerings (e.g. scf.if)
        // can add sibling blocks to this region during the walk.
        for (size_t b = 0; b < MLIR_GetRegionNumBlocks(reg); b++) {
            walk_block(st, MLIR_GetRegionBlock(reg, b));
        }
    }
}

static void walk_block(LowerState *st, MLIR_BlockHandle block) {
    size_t i = 0;
    while (i < MLIR_GetBlockNumOps(block)) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        walk_op(st, op);
        size_t before = MLIR_GetBlockNumOps(block);
        int res = try_lower_op(st, op, block, i);
        if (res == LOWER_NONE) { i++; continue; }
        if (res == LOWER_DONE_BLOCK) {
            // The lowering restructured this block (and likely created
            // sibling blocks); nothing more to walk here. The new blocks
            // are appended to the parent region and will be visited by
            // the outer walker as it enumerates them.
            return;
        }
        // LOWER_REPLACED: simple in-place rewrite. Original is still
        // attached at position i + inserted; erase it and skip the inserts.
        size_t after = MLIR_GetBlockNumOps(block);
        size_t inserted = after - before;
        MLIR_EraseOp(st->ctx, op);
        i += inserted;
    }
}

bool MLIR_LowerToLLVMDialect(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (module == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpNumRegions(module) == 0) return false;
    MLIR_RegionHandle body_region = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(body_region) == 0) return false;

    LowerState st = {0};
    st.ctx = ctx;
    st.module = module;
    st.module_body = MLIR_GetRegionBlock(body_region, 0);

    walk_block(&st, st.module_body);
    return true;
}

// In-tree LLVM-dialect lowering tailored for the wasm pipeline. First
// lifts cf.br / cf.cond_br into scf.if / scf.while via MLIR_LiftCfToScf,
// then runs the regular lowering with `keep_scf = true` so the scf
// operations survive into the wasmssa stage (which expects structured
// control flow). cf->llvm.br rewrites are also skipped: any cf op
// surviving the lift is a hard error caught later by wasmssa-lower.
bool MLIR_LowerToLLVMDialectForWasm(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (module == MLIR_INVALID_HANDLE) return false;
    if (!MLIR_LiftCfToScf(ctx, module)) return false;
    if (MLIR_GetOpNumRegions(module) == 0) return false;
    MLIR_RegionHandle body_region = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(body_region) == 0) return false;

    LowerState st = {0};
    st.ctx = ctx;
    st.module = module;
    st.module_body = MLIR_GetRegionBlock(body_region, 0);
    st.keep_scf = true;

    walk_block(&st, st.module_body);
    return true;
}

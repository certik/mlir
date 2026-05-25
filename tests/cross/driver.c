// tests/cross/driver.c
//
// Cross-implementation smoke test for the public MLIR C API in
// `mlir_api.h`. Builds a small but non-trivial module containing a
// func.func with two block arguments, an arith.addi, and a
// func.return, then prints it via the generic printer. Compiled twice:
//
//   * `cross_native` — backed by mlir_api_impl.c (corec arena, no libc).
//   * `cross_upstream` — backed by mlir_api_impl_upstream.cpp (upstream
//     LLVM/MLIR C++, system libc).
//
// Both binaries must produce byte-identical output.

#include <base/arena.h>
#include <base/string.h>
#include <base/io.h>
#include <platform/platform.h>
#include "mlir_api.h"
#include "mlir_generic_printer.h"

static void print_string(string s) {
    ciovec_t iov;
    iov.buf = s.str;
    iov.buf_len = s.size;
    write_all(PLATFORM_STDOUT_FD, &iov, 1);
}

int app_main(void) {
    Arena *arena = arena_create(1 * 1024 * 1024);
    MLIR_Context ctx;
    MLIR_SetArenaAllocator(&ctx, arena);

    MLIR_TypeHandle i32_ty = MLIR_CreateTypeInteger(&ctx, 32, true);

    // Verify that semantically-equivalent simple types share a single
    // MLIR_TypeHandle. The upstream backend has always done this (MLIR
    // interns IntegerType / IndexType / ptr / etc. via `::get`); the
    // native backend used to allocate a fresh IR_Type per call, which
    // broke type-handle equality checks in the cf->scf lifter
    // (get_undef_value / get_switch_value caches) and produced extra
    // llvm.mlir.undef ops when the same i32 was looked up by handle.
    // See mlir_api_impl.c's intern_type for the native fix.
    {
        MLIR_TypeHandle i32_a = MLIR_CreateTypeInteger(&ctx, 32, true);
        MLIR_TypeHandle i32_b = MLIR_CreateTypeInteger(&ctx, 32, true);
        MLIR_TypeHandle i64_a = MLIR_CreateTypeInteger(&ctx, 64, false);
        MLIR_TypeHandle i64_b = MLIR_CreateTypeInteger(&ctx, 64, false);
        MLIR_TypeHandle idx_a = MLIR_CreateTypeIndex(&ctx);
        MLIR_TypeHandle idx_b = MLIR_CreateTypeIndex(&ctx);
        MLIR_TypeHandle ptr_a = MLIR_CreateTypeLLVMPointer(&ctx);
        MLIR_TypeHandle ptr_b = MLIR_CreateTypeLLVMPointer(&ctx);
        print_string(i32_a == i32_b ? str_lit("intern i32: yes\n") : str_lit("intern i32: no\n"));
        print_string(i64_a == i64_b ? str_lit("intern i64: yes\n") : str_lit("intern i64: no\n"));
        print_string(idx_a == idx_b ? str_lit("intern index: yes\n") : str_lit("intern index: no\n"));
        print_string(ptr_a == ptr_b ? str_lit("intern !llvm.ptr: yes\n") : str_lit("intern !llvm.ptr: no\n"));
        print_string(i32_a == i64_a ? str_lit("intern i32==i64: yes\n") : str_lit("intern i32==i64: no\n"));
    }

    // Verify that the flat-operand contract of MLIR_GetValueUseOwner /
    // MLIR_SetOpOperand also covers successor-operand uses. The cf->scf
    // lifter's `replace_use` helper round-trips a use through
    // SetOpOperand using the synthetic index returned by GetValueUseOwner,
    // which on the upstream backend addresses both regular and
    // successor-operand slots of a BranchOpInterface op uniformly. The
    // native backend used to silently no-op when idx >= n_operands,
    // which left cf.switch / cf.cond_br successor operands pointing to
    // stale values across the cf->scf RAUW chain and produced
    // semantically-equivalent but byte-different wasm. See
    // mlir_api_impl.c's MLIR_SetOpOperand for the fix.
    {
        MLIR_RegionHandle r = MLIR_CreateRegion(&ctx);
        MLIR_BlockHandle entry = MLIR_CreateBlock(&ctx);
        MLIR_BlockHandle tgt = MLIR_CreateBlock(&ctx);
        MLIR_AppendRegionBlock(&ctx, r, entry);
        MLIR_AppendRegionBlock(&ctx, r, tgt);

        MLIR_ValueHandle defv = MLIR_CreateValueBlockArg(
            &ctx, str_lit("%d"), 0, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(&ctx, entry, defv);
        MLIR_ValueHandle replv = MLIR_CreateValueBlockArg(
            &ctx, str_lit("%r"), 1, i32_ty, MLIR_INVALID_HANDLE);
        MLIR_AppendBlockArg(&ctx, entry, replv);

        // Build cf.br tgt(defv : i32) — zero regular operands, one successor
        // with one successor-operand.
        MLIR_BlockHandle *succs = arena_new_array(arena, MLIR_BlockHandle, 1);
        succs[0] = tgt;
        MLIR_ValueHandle **sops = arena_new_array(arena, MLIR_ValueHandle *, 1);
        sops[0] = arena_new_array(arena, MLIR_ValueHandle, 1);
        sops[0][0] = defv;
        size_t *snums = arena_new_array(arena, size_t, 1);
        snums[0] = 1;
        MLIR_OpHandle br = MLIR_CreateOpWithSuccessors(
            &ctx, OP_TYPE_CF_BR, str_lit("cf.br"),
            NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
            succs, 1, sops, snums,
            MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(&ctx, entry, br);

        size_t n_uses = MLIR_GetValueNumUses(&ctx, defv);
        print_string(n_uses == 1 ? str_lit("cf.br succ_op use registered: yes\n")
                                 : str_lit("cf.br succ_op use registered: no\n"));

        size_t op_idx = (size_t)-1;
        MLIR_OpHandle owner = MLIR_GetValueUseOwner(&ctx, defv, 0, &op_idx);
        print_string(owner == br ? str_lit("use owner is cf.br: yes\n")
                                 : str_lit("use owner is cf.br: no\n"));

        // Round-trip the synthetic flat index through SetOpOperand. After
        // the call, cf.br's successor operand at slot 0 must point to
        // `replv` instead of `defv`.
        MLIR_SetOpOperand(&ctx, br, op_idx, replv);
        MLIR_ValueHandle after = MLIR_GetOpSuccessorOperand(br, 0, 0);
        print_string(after == replv ? str_lit("SetOpOperand updates succ_op: yes\n")
                                    : str_lit("SetOpOperand updates succ_op: no\n"));
        print_string(MLIR_GetValueNumUses(&ctx, defv) == 0
                         ? str_lit("old value uses cleared: yes\n")
                         : str_lit("old value uses cleared: no\n"));
        print_string(MLIR_GetValueNumUses(&ctx, replv) == 1
                         ? str_lit("new value uses registered: yes\n")
                         : str_lit("new value uses registered: no\n"));
    }

    // ---- module region/block ------------------------------------------------
    MLIR_RegionHandle module_region = MLIR_CreateRegion(&ctx);
    MLIR_BlockHandle  module_block  = MLIR_CreateBlock(&ctx);
    MLIR_AppendRegionBlock(&ctx, module_region, module_block);

    // ---- func.func body -----------------------------------------------------
    MLIR_RegionHandle func_region = MLIR_CreateRegion(&ctx);
    MLIR_BlockHandle  func_block  = MLIR_CreateBlock(&ctx);

    MLIR_ValueHandle arg0 = MLIR_CreateValueBlockArg(
        &ctx, str_lit("%arg0"), 0, i32_ty, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(&ctx, func_block, arg0);
    MLIR_ValueHandle arg1 = MLIR_CreateValueBlockArg(
        &ctx, str_lit("%arg1"), 1, i32_ty, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(&ctx, func_block, arg1);

    // %0 = arith.addi(%arg0, %arg1) -> i32
    MLIR_ValueHandle add_result = MLIR_CreateValueOpResult(
        &ctx, MLIR_INVALID_HANDLE, 0, i32_ty,
        str_lit(""), MLIR_INVALID_HANDLE);

    MLIR_ValueHandle *add_operands = arena_new_array(arena, MLIR_ValueHandle, 2);
    add_operands[0] = arg0;
    add_operands[1] = arg1;
    MLIR_TypeHandle  *add_result_types = arena_new_array(arena, MLIR_TypeHandle, 1);
    add_result_types[0] = i32_ty;
    MLIR_ValueHandle *add_results = arena_new_array(arena, MLIR_ValueHandle, 1);
    add_results[0] = add_result;

    MLIR_OpHandle add_op = MLIR_CreateOp(
        &ctx, OP_TYPE_ARITH_ADDI, str_lit("arith.addi"),
        NULL, 0,
        add_result_types, 1,
        add_results, 1,
        add_operands, 2,
        NULL, 0,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    // func.return(%0)
    MLIR_ValueHandle *ret_operands = arena_new_array(arena, MLIR_ValueHandle, 1);
    ret_operands[0] = add_result;
    MLIR_OpHandle ret_op = MLIR_CreateOp(
        &ctx, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
        NULL, 0, NULL, 0, NULL, 0,
        ret_operands, 1,
        NULL, 0,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    MLIR_AppendBlockOp(&ctx, func_block, add_op);
    MLIR_AppendBlockOp(&ctx, func_block, ret_op);
    MLIR_AppendRegionBlock(&ctx, func_region, func_block);

    // ---- func.func op --------------------------------------------------------
    MLIR_AttributeHandle sym_name_attr =
        MLIR_CreateAttributeString(&ctx, str_lit("sym_name"), str_lit("add"));
    MLIR_AttributeHandle *func_attrs = arena_new_array(arena, MLIR_AttributeHandle, 1);
    func_attrs[0] = sym_name_attr;
    MLIR_RegionHandle *func_regions = arena_new_array(arena, MLIR_RegionHandle, 1);
    func_regions[0] = func_region;

    MLIR_OpHandle func_op = MLIR_CreateOp(
        &ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
        func_attrs, 1,
        NULL, 0, NULL, 0, NULL, 0,
        func_regions, 1,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);
    MLIR_AppendBlockOp(&ctx, module_block, func_op);

    // ---- module op ----------------------------------------------------------
    MLIR_RegionHandle *module_regions = arena_new_array(arena, MLIR_RegionHandle, 1);
    module_regions[0] = module_region;
    MLIR_OpHandle module_op = MLIR_CreateOp(
        &ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        module_regions, 1,
        MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE,
        str_lit(""), -1);

    MLIR_InitApi(&ctx, module_op);
    print_string(print_operation_generic(&ctx, 0, module_op));

    arena_destroy(arena);
    return 0;
}


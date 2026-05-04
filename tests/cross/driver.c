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


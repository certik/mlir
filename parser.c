#include <stdbool.h>
#include <stdio.h>

#include "tokenizer.h"
#include <base/arena.h>
#include <base/io.h>
#include "mlir_api.h"
#include "mlir_generic_printer.h"
#include "mlir_classic_printer.h"
#include "mlir_parser.h"
#include <base/hashtable.h>

void tokenizer_print_all_tokens(Arena *arena, const string input_code) {
    unsigned char *string_start;
    string_start = (unsigned char*)input_code.str;
    uint64_t cur=0;
    while (true) {
        TokenType token_type;
        uint64_t first, last;
        first = cur;
        tokenizer_get_next_token(string_start, &cur, &token_type);
        last = cur-1;
        bool debug = false;
        if (debug) {
            string token_str = str_substr(input_code, first, last-first+1);
            if (token_type == TK_WHITESPACE || token_type == TK_NEWLINE
                    || token_type == TK_EOF) {
                token_str = str_lit("");
            }
            println(arena, str_lit("Token({}, \"{}\", {},{})"),
                str_from_cstr_view((char*)mlir_tokentype_to_string(token_type)), token_str, first, last);
        }
        if (token_type == TK_EOF) {
            return;
        }
    }
}


MLIR_OpHandle construct_test_module_full(void) {
    // Create types
    MLIR_TypeHandle i32_type = MLIR_CreateTypeInteger(32, true);

    // Create module operation
    MLIR_RegionHandle module_region = MLIR_CreateRegion();
    MLIR_BlockHandle module_block = MLIR_CreateBlock();
    MLIR_AppendRegionBlock(module_region, module_block);

    Arena *arena = arena_create(1024*1024);  // Small arena for temp allocations

    // Set regions
    MLIR_RegionHandle *module_regions = arena_alloc_array(arena, MLIR_RegionHandle, 1);
    module_regions[0] = module_region;

    MLIR_OpHandle module = MLIR_CreateOp(OP_TYPE_MODULE, str_lit("module"), NULL, 0, NULL, 0, NULL, 0, NULL, 0, module_regions, 1, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);

    // Function attributes (sym_name)
    MLIR_AttributeHandle sym_name_attr = MLIR_CreateAttributeString(str_lit("sym_name"), str_lit("example_func"));
    MLIR_AttributeHandle *func_attrs = arena_alloc_array(arena, MLIR_AttributeHandle, 1);
    func_attrs[0] = sym_name_attr;

    // Create function region and block
    MLIR_RegionHandle func_region = MLIR_CreateRegion();
    MLIR_BlockHandle func_block = MLIR_CreateBlock();

    // Function block arguments (%arg0, %arg1)
    MLIR_ValueHandle arg0 = MLIR_CreateValueBlockArg(str_lit("%arg0"), 0, i32_type, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(func_block, arg0);

    MLIR_ValueHandle arg1 = MLIR_CreateValueBlockArg(str_lit("%arg1"), 1, i32_type, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(func_block, arg1);

    // %0 = arith.constant 5 : i32
    MLIR_ValueHandle const_result = MLIR_CreateValueOpResult(MLIR_INVALID_HANDLE, 0, i32_type, str_lit("%0"), MLIR_INVALID_HANDLE);

    MLIR_TypeHandle *const_result_types = arena_alloc_array(arena, MLIR_TypeHandle, 1);
    const_result_types[0] = i32_type;

    MLIR_AttributeHandle value_attr = MLIR_CreateAttributeInteger(str_lit("value"), 5);
    MLIR_AttributeHandle *const_attrs = arena_alloc_array(arena, MLIR_AttributeHandle, 1);
    const_attrs[0] = value_attr;

    MLIR_ValueHandle *const_results = arena_alloc_array(arena, MLIR_ValueHandle, 1);
    const_results[0] = const_result;

    MLIR_OpHandle const_op = MLIR_CreateOp(OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"), const_attrs, 1, const_result_types, 1, const_results, 1, NULL, 0, NULL, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);

    // %1 = arith.addi %arg0, %arg1 : i32
    MLIR_ValueHandle add_result = MLIR_CreateValueOpResult(MLIR_INVALID_HANDLE, 1, i32_type, str_lit("%1"), MLIR_INVALID_HANDLE);

    MLIR_ValueHandle *add_operands = arena_alloc_array(arena, MLIR_ValueHandle, 2);
    add_operands[0] = MLIR_GetBlockArg(func_block, 0);
    add_operands[1] = MLIR_GetBlockArg(func_block, 1);

    MLIR_TypeHandle *add_result_types = arena_alloc_array(arena, MLIR_TypeHandle, 1);
    add_result_types[0] = i32_type;

    MLIR_ValueHandle *add_results = arena_alloc_array(arena, MLIR_ValueHandle, 1);
    add_results[0] = add_result;

    MLIR_OpHandle add_op = MLIR_CreateOp(OP_TYPE_ARITH_ADDI, str_lit(""), NULL, 0, add_result_types, 1, add_results, 1, add_operands, 2, NULL, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);

    // %2 = arith.muli %1, %0 : i32
    MLIR_ValueHandle mul_result = MLIR_CreateValueOpResult(MLIR_INVALID_HANDLE, 2, i32_type, str_lit("%2"), MLIR_INVALID_HANDLE);

    MLIR_ValueHandle *mul_operands = arena_alloc_array(arena, MLIR_ValueHandle, 2);
    mul_operands[0] = add_result;
    mul_operands[1] = const_result;

    MLIR_TypeHandle *mul_result_types = arena_alloc_array(arena, MLIR_TypeHandle, 1);
    mul_result_types[0] = i32_type;

    MLIR_ValueHandle *mul_results = arena_alloc_array(arena, MLIR_ValueHandle, 1);
    mul_results[0] = mul_result;

    MLIR_OpHandle mul_op = MLIR_CreateOp(OP_TYPE_ARITH_MULI, str_lit(""), NULL, 0, mul_result_types, 1, mul_results, 1, mul_operands, 2, NULL, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);

    // func.return %2 : i32
    MLIR_ValueHandle *ret_operands = arena_alloc_array(arena, MLIR_ValueHandle, 1);
    ret_operands[0] = mul_result;
    MLIR_OpHandle ret_op = MLIR_CreateOp(OP_TYPE_FUNC_RETURN, str_lit(""), NULL, 0, NULL, 0, NULL, 0, ret_operands, 1, NULL, 0, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);

    MLIR_AppendBlockOp(func_block, const_op);
    MLIR_AppendBlockOp(func_block, add_op);
    MLIR_AppendBlockOp(func_block, mul_op);
    MLIR_AppendBlockOp(func_block, ret_op);

    MLIR_AppendRegionBlock(func_region, func_block);

    MLIR_RegionHandle *func_regions = arena_alloc_array(arena, MLIR_RegionHandle, 1);
    func_regions[0] = func_region;

    MLIR_OpHandle func_op = MLIR_CreateOp(OP_TYPE_FUNC_FUNC, str_lit("func.func"), func_attrs, 1, NULL, 0, NULL, 0, NULL, 0, func_regions, 1, MLIR_INVALID_HANDLE, MLIR_INVALID_HANDLE, str_lit(""), -1);

    MLIR_AppendBlockOp(module_block, func_op);

    return module;
}

int main(int argc, char *argv[]) {
    bool use_construction = false;
    bool use_classic_printer = false;
    bool verbose = false;
    char *input_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
            break;
        }
    }

    if (verbose) printf("Starting main...\n");
    Arena *arena = arena_create(50*1024*1024);
    if (verbose) printf("Arena created...\n");

    MLIR_SetArena(arena);
    if (verbose) printf("Global arena set...\n");

    if (verbose) printf("Parsing args...\n");
    for (int i = 1; i < argc; i++) {
        if (verbose) printf("Arg %d: %s\n", i, argv[i]);
        if (strcmp(argv[i], "--construct") == 0) {
            use_construction = true;
            if (verbose) printf("Construction mode enabled\n");
        } else if (strcmp(argv[i], "--classic") == 0 || strcmp(argv[i], "-c") == 0) {
            use_classic_printer = true;
            if (verbose) printf("Classic printer enabled\n");
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
            if (verbose) printf("Verbose mode enabled\n");
        } else if (argv[i][0] != '-') {
            input_file = argv[i];
        }
    }
    if (verbose) printf("Done parsing args. use_construction=%d, use_classic_printer=%d\n", use_construction, use_classic_printer);

    MLIR_OpHandle op;

    int exit_code;
    if (use_construction) {
        if (verbose) printf("Creating module...\n");
        op = construct_test_module_full();
        if (verbose) printf("Module created successfully.\n");

        if (verbose) printf("=== Generic Printer Test ===\n");
        if (verbose) printf("About to print operation...\n");
        string result;
        if (use_classic_printer) {
            result = print_operation_classic(arena, 0, op);
        } else {
            result = print_operation_generic(arena, 0, op);
        }
        if (verbose) printf("Printing result...\n");
        println(arena, str_lit("{}"), result);

        const char *expected =
            "module() {\n"
            "^bb0:\n"
            "    func.func() {sym_name = \"example_func\"} {\n"
            "    ^bb0(%arg0: i32, %arg1: i32):\n"
            "        %0 = arith.constant() {value = 5} -> i32\n"
            "        %1 = arith.addi(%arg0: i32, %arg1: i32) -> i32\n"
            "        %2 = arith.muli(%1: i32, %0: i32) -> i32\n"
            "        func.return(%2: i32)\n"
            "    }\n"
            "}\n";

        if (str_eq(result, str_from_cstr_view((char*)expected))) {
            if (verbose) printf("✅ Generic mode test PASSED\n");
            exit_code = 0;
        } else {
            if (verbose) printf("❌ Generic mode test FAILED\n");
            if (verbose) printf("Expected:\n%s\n", expected);
            if (verbose) printf("Actual:\n");
            println(arena, str_lit("{}"), result);
            exit_code = 1;
        }
    } else {
        string mlir_code = str_lit("module {\n"
                                "  %0 = \"std.constant\"() {value = 42} : () -> i32\n"
                                "  \"std.return\"(%0) : (i32) -> ()\n"
                                "}");

        if (input_file) {
            mlir_code = read_file_ok(arena, str_from_cstr_view(input_file));
        }

        tokenizer_print_all_tokens(arena, mlir_code);

        MLIR_LocationMap *locmap = NULL;
        op = mlir_parse_module(arena, (const char*)mlir_code.str, mlir_code.size, &locmap);
        if (verbose) println(arena, str_lit("MLIR:"));
        if (use_classic_printer) {
            println(arena, str_lit("{}"), print_module_classic(arena, op, locmap));
        } else {
            println(arena, str_lit("{}"), print_operation_generic(arena, 0, op));
        }
        exit_code = 0;
    }

    arena_free(arena);
    return exit_code;
}

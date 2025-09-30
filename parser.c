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


MlirOperation* construct_test_module_full(Arena *arena) {
    // Create types
    MlirType *i32_type = mlir_type_create_integer(arena, 32, true);
    //MlirType *i64_type = mlir_type_create_integer(arena, 64, true);

    // Create module operation
    MlirOperation *module = mlir_op_create(arena, OP_TYPE_MODULE, str_lit("module"), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Create module region
    MlirRegion *module_region = mlir_region_create(arena);
    MlirBlock *module_block = mlir_block_create(arena);
    // Add block to region
    mlir_region_add_block(arena, module_region, module_block);

    // Add region to module
    mlir_op_add_region(arena, module, module_region);

    // Create function operation
    MlirOperation *func_op = mlir_op_create(arena, OP_TYPE_FUNC_FUNC, str_lit("func.func"), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Function attributes (sym_name)
    MlirAttribute *sym_name_attr = mlir_attribute_create_string(arena, "example_func", 12);
    mlir_attribute_set_name(sym_name_attr, "sym_name", 8);
    MlirAttribute **func_attrs = arena_alloc_array(arena, MlirAttribute*, 1);
    func_attrs[0] = sym_name_attr;
    mlir_operation_set_attributes(func_op, func_attrs, 1);

    // Create function region and block
    MlirRegion *func_region = mlir_region_create(arena);
    MlirBlock *func_block = mlir_block_create(arena);

    // Function block arguments (%arg0, %arg1)
    MlirValue *arg0 = mlir_value_create(arena, BLOCK_ARG);
    mlir_value_set_result_index(arg0, 0);
    mlir_value_set_type(arg0, i32_type);
    mlir_value_set_register_name(arg0, "%arg0", 5);
    mlir_block_add_argument(arena, func_block, arg0);

    MlirValue *arg1 = mlir_value_create(arena, BLOCK_ARG);
    mlir_value_set_result_index(arg1, 1);
    mlir_value_set_type(arg1, i32_type);
    mlir_value_set_register_name(arg1, "%arg1", 5);
    mlir_block_add_argument(arena, func_block, arg1);

    // Create operations in function block
    // %0 = arith.constant 5 : i32
    MlirOperation *const_op = mlir_op_create(arena, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Set result types
    MlirType **const_result_types = arena_alloc_array(arena, MlirType*, 1);
    const_result_types[0] = i32_type;

    // Set attributes
    MlirAttribute *value_attr = mlir_attribute_create_integer(arena, 5);
    mlir_attribute_set_name(value_attr, "value", 5);
    MlirAttribute **const_attrs = arena_alloc_array(arena, MlirAttribute*, 1);
    const_attrs[0] = value_attr;
    mlir_operation_set_attributes(const_op, const_attrs, 1);

    // Create const_result
    MlirValue *const_result = mlir_value_create(arena, OP_RESULT);
    mlir_value_set_def(const_result, const_op);
    mlir_value_set_result_index(const_result, 0);
    mlir_value_set_type(const_result, i32_type);
    mlir_value_set_register_name(const_result, "%0", 2);

    // Set operation results and types together
    MlirValue **const_results = arena_alloc_array(arena, MlirValue*, 1);
    const_results[0] = const_result;
    mlir_operation_set_results_with_types(const_op, const_results, const_result_types, 1);

    // %1 = arith.addi %arg0, %arg1 : i32
    MlirOperation *add_op = mlir_op_create(arena, OP_TYPE_ARITH_ADDI, str_lit(""), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Set operands
    MlirValue **add_operands = arena_alloc_array(arena, MlirValue*, 2);
    add_operands[0] = mlir_block_get_argument(func_block, 0);
    add_operands[1] = mlir_block_get_argument(func_block, 1);
    mlir_operation_set_operands(add_op, add_operands, 2);

    // Set result types
    MlirType **add_result_types = arena_alloc_array(arena, MlirType*, 1);
    add_result_types[0] = i32_type;

    // Create add_result
    MlirValue *add_result = mlir_value_create(arena, OP_RESULT);
    mlir_value_set_def(add_result, add_op);
    mlir_value_set_result_index(add_result, 1);
    mlir_value_set_type(add_result, i32_type);
    mlir_value_set_register_name(add_result, "%1", 2);

    // Set operation results and types together
    MlirValue **add_results = arena_alloc_array(arena, MlirValue*, 1);
    add_results[0] = add_result;
    mlir_operation_set_results_with_types(add_op, add_results, add_result_types, 1);

    // %2 = arith.muli %1, %0 : i32 (add_result and const_result already created above)

    // Create mul operation
    MlirOperation *mul_op = mlir_op_create(arena, OP_TYPE_ARITH_MULI, str_lit(""), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Set operands
    MlirValue **mul_operands = arena_alloc_array(arena, MlirValue*, 2);
    mul_operands[0] = add_result;
    mul_operands[1] = const_result;
    mlir_operation_set_operands(mul_op, mul_operands, 2);

    // Set result types
    MlirType **mul_result_types = arena_alloc_array(arena, MlirType*, 1);
    mul_result_types[0] = i32_type;

    // Create mul_result
    MlirValue *mul_result = mlir_value_create(arena, OP_RESULT);
    mlir_value_set_def(mul_result, mul_op);
    mlir_value_set_result_index(mul_result, 2);
    mlir_value_set_type(mul_result, i32_type);
    mlir_value_set_register_name(mul_result, "%2", 2);

    // Set operation results and types together
    MlirValue **mul_results = arena_alloc_array(arena, MlirValue*, 1);
    mul_results[0] = mul_result;
    mlir_operation_set_results_with_types(mul_op, mul_results, mul_result_types, 1);

    // func.return %2 : i32
    MlirOperation *ret_op = mlir_op_create(arena, OP_TYPE_FUNC_RETURN, str_lit(""), NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Set operands
    MlirValue **ret_operands = arena_alloc_array(arena, MlirValue*, 1);
    ret_operands[0] = mul_result;
    mlir_operation_set_operands(ret_op, ret_operands, 1);

    // Link operations to function block
    mlir_block_add_operation(arena, func_block, const_op);
    mlir_block_add_operation(arena, func_block, add_op);
    mlir_block_add_operation(arena, func_block, mul_op);
    mlir_block_add_operation(arena, func_block, ret_op);

    // Link function block to function region
    mlir_region_add_block(arena, func_region, func_block);

    // Link function region to function operation
    mlir_op_add_region(arena, func_op, func_region);

    // Link function operation to module block
    mlir_block_add_operation(arena, module_block, func_op);

    // Link module block to module region
    // module_region already has block

    // Link module region to module operation
    // module already has region

    return module;
}

int main(int argc, char *argv[]) {
    // Check for options first
    bool use_construction = false;
    bool use_classic_printer = false;
    bool verbose = false;
    char *input_file = NULL;

    // Parse arguments first to determine verbose mode
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
            break;
        }
    }

    if (verbose) printf("Starting main...\n");
    Arena *arena = arena_create(50*1024*1024);  // Increase arena size
    if (verbose) printf("Arena created...\n");

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

    MlirOperation* op;

    int exit_code;
    if (use_construction) {
        // Use constructed test module
        if (verbose) printf("Creating module...\n");
        op = construct_test_module_full(arena);
        if (verbose) printf("Module created successfully.\n");

        // Test generic printing with expected output comparison
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

        // Reference expected output for generic mode
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

        // Compare output
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
        // Use parser mode
        string mlir_code = str_lit("module {\n"
                                "  %0 = \"std.constant\"() {value = 42} : () -> i32\n"
                                "  \"std.return\"(%0) : (i32) -> ()\n"
                                "}");

        if (input_file) {
            mlir_code = read_file_ok(arena, str_from_cstr_view(input_file));
        }

        tokenizer_print_all_tokens(arena, mlir_code);

        MlirLocationMap *locmap = NULL;
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

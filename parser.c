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


MLIR_Op* construct_test_module_full(void) {
    // Create types
    MLIR_Type *i32_type = MLIR_CreateTypeInteger(32, true);
    //MLIR_Type *i64_type = MLIR_CreateTypeInteger(64, true);

    // Create module operation
    // Create module region
    MLIR_Region *module_region = MLIR_CreateRegion();
    MLIR_Block *module_block = MLIR_CreateBlock();
    // Add block to region
    MLIR_AppendRegionBlock(module_region, module_block);

    // Note: We still need the local arena for temporary allocations like arrays
    Arena *arena = arena_create(1024*1024);  // Small arena for temp allocations
    
    // Set regions
    MLIR_Region **module_regions = arena_alloc_array(arena, MLIR_Region*, 1);
    module_regions[0] = module_region;

    MLIR_Op *module = MLIR_CreateOp(OP_TYPE_MODULE, str_lit("module"), NULL, 0, NULL, 0, NULL, 0, NULL, 0, module_regions, 1, NULL, NULL, str_lit(""), -1);

    // Function attributes (sym_name)
    MLIR_Attribute *sym_name_attr = MLIR_CreateAttributeString(str_lit("sym_name"), str_lit("example_func"));
    MLIR_Attribute **func_attrs = arena_alloc_array(arena, MLIR_Attribute*, 1);
    func_attrs[0] = sym_name_attr;

    // Create function region and block
    MLIR_Region *func_region = MLIR_CreateRegion();
    MLIR_Block *func_block = MLIR_CreateBlock();

    // Function block arguments (%arg0, %arg1)
    MLIR_Value *arg0 = MLIR_CreateValueBlockArg(str_lit("%arg0"), 0, i32_type, NULL);
    MLIR_AppendBlockArg(func_block, arg0);

    MLIR_Value *arg1 = MLIR_CreateValueBlockArg(str_lit("%arg1"), 1, i32_type, NULL);
    MLIR_AppendBlockArg(func_block, arg1);

    // Create operations in function block
    // %0 = arith.constant 5 : i32

    // Create const_result first (def will be set after operation creation)
    MLIR_Value *const_result = MLIR_CreateValueOpResult(NULL, 0, i32_type, str_lit("%0"), NULL);

    // Set result types
    MLIR_Type **const_result_types = arena_alloc_array(arena, MLIR_Type*, 1);
    const_result_types[0] = i32_type;

    // Set attributes
    MLIR_Attribute *value_attr = MLIR_CreateAttributeInteger(str_lit("value"), 5);
    MLIR_Attribute **const_attrs = arena_alloc_array(arena, MLIR_Attribute*, 1);
    const_attrs[0] = value_attr;

    // Set results
    MLIR_Value **const_results = arena_alloc_array(arena, MLIR_Value*, 1);
    const_results[0] = const_result;

    MLIR_Op *const_op = MLIR_CreateOp(OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"), const_attrs, 1, const_result_types, 1, const_results, 1, NULL, 0, NULL, 0, NULL, NULL, str_lit(""), -1);

    // %1 = arith.addi %arg0, %arg1 : i32

    // Create add_result first (def will be set after operation creation)
    MLIR_Value *add_result = MLIR_CreateValueOpResult(NULL, 1, i32_type, str_lit("%1"), NULL);

    // Set operands
    MLIR_Value **add_operands = arena_alloc_array(arena, MLIR_Value*, 2);
    add_operands[0] = MLIR_GetBlockArg(func_block, 0);
    add_operands[1] = MLIR_GetBlockArg(func_block, 1);

    // Set result types
    MLIR_Type **add_result_types = arena_alloc_array(arena, MLIR_Type*, 1);
    add_result_types[0] = i32_type;

    // Set results
    MLIR_Value **add_results = arena_alloc_array(arena, MLIR_Value*, 1);
    add_results[0] = add_result;

    MLIR_Op *add_op = MLIR_CreateOp(OP_TYPE_ARITH_ADDI, str_lit(""), NULL, 0, add_result_types, 1, add_results, 1, add_operands, 2, NULL, 0, NULL, NULL, str_lit(""), -1);

    // %2 = arith.muli %1, %0 : i32 (add_result and const_result already created above)

    // Create mul_result first (def will be set after operation creation)
    MLIR_Value *mul_result = MLIR_CreateValueOpResult(NULL, 2, i32_type, str_lit("%2"), NULL);

    // Set operands
    MLIR_Value **mul_operands = arena_alloc_array(arena, MLIR_Value*, 2);
    mul_operands[0] = add_result;
    mul_operands[1] = const_result;

    // Set result types
    MLIR_Type **mul_result_types = arena_alloc_array(arena, MLIR_Type*, 1);
    mul_result_types[0] = i32_type;

    // Set results
    MLIR_Value **mul_results = arena_alloc_array(arena, MLIR_Value*, 1);
    mul_results[0] = mul_result;

    MLIR_Op *mul_op = MLIR_CreateOp(OP_TYPE_ARITH_MULI, str_lit(""), NULL, 0, mul_result_types, 1, mul_results, 1, mul_operands, 2, NULL, 0, NULL, NULL, str_lit(""), -1);

    // func.return %2 : i32
    MLIR_Value **ret_operands = arena_alloc_array(arena, MLIR_Value*, 1);
    ret_operands[0] = mul_result;
    MLIR_Op *ret_op = MLIR_CreateOp(OP_TYPE_FUNC_RETURN, str_lit(""), NULL, 0, NULL, 0, NULL, 0, ret_operands, 1, NULL, 0, NULL, NULL, str_lit(""), -1);

    // Link operations to function block
    MLIR_AppendBlockOp(func_block, const_op);
    MLIR_AppendBlockOp(func_block, add_op);
    MLIR_AppendBlockOp(func_block, mul_op);
    MLIR_AppendBlockOp(func_block, ret_op);

    // Link function block to function region
    MLIR_AppendRegionBlock(func_region, func_block);

    // Set regions for function operation
    MLIR_Region **func_regions = arena_alloc_array(arena, MLIR_Region*, 1);
    func_regions[0] = func_region;

    // Create function operation
    MLIR_Op *func_op = MLIR_CreateOp(OP_TYPE_FUNC_FUNC, str_lit("func.func"), func_attrs, 1, NULL, 0, NULL, 0, NULL, 0, func_regions, 1, NULL, NULL, str_lit(""), -1);

    // Link function operation to module block
    MLIR_AppendBlockOp(module_block, func_op);

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
    
    // Set the global arena for MLIR API
    MLIR_SetGlobalArenaAllocator(arena);
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

    MLIR_Op* op;

    int exit_code;
    if (use_construction) {
        // Use constructed test module
        if (verbose) printf("Creating module...\n");
        op = construct_test_module_full();
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

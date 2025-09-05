#include <stdbool.h>
#include <stdio.h>

#include "tokenizer.h"
#include <base/arena.h>
#include <base/io.h>
#include "mlir_parser.h"
#include "mlir_generic_printer.h"
#include "mlir_classic_printer.h"
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
                tokentype_to_string(token_type), token_str, first, last);
        }
        if (token_type == TK_EOF) {
            return;
        }
    }
}


Operation* construct_test_module_full(Arena *arena) {
    // Create types
    Type *i32_type = arena_alloc(arena, Type);
    i32_type->kind = TYPE_KIND_INTEGER;
    i32_type->data.integer.width = 32;
    i32_type->data.integer.is_signed = true;

    Type *i64_type = arena_alloc(arena, Type);
    i64_type->kind = TYPE_KIND_INTEGER;
    i64_type->data.integer.width = 64;
    i64_type->data.integer.is_signed = true;

    // Create module operation
    Operation *module = arena_alloc(arena, Operation);
    module->op_type = OP_TYPE_MODULE;
    module->operands = NULL;
    module->n_operands = 0;
    module->result_types = NULL;
    module->n_result_types = 0;
    module->attributes = NULL;
    module->n_attributes = 0;
    module->results = NULL;
    module->n_results = 0;
    module->opname = str_lit("module");

    // Create module region
    Region *module_region = arena_alloc(arena, Region);
    Block *module_block = arena_alloc(arena, Block);
    module_block->arguments = NULL;
    module_block->n_arguments = 0;

    // Create function operation
    Operation *func_op = arena_alloc(arena, Operation);
    func_op->op_type = OP_TYPE_FUNC_FUNC;
    func_op->operands = NULL;
    func_op->n_operands = 0;
    func_op->result_types = NULL;
    func_op->n_result_types = 0;
    func_op->results = NULL;
    func_op->n_results = 0;
    func_op->opname = str_lit("func.func");

    // Function attributes (sym_name)
    func_op->n_attributes = 1;
    func_op->attributes = arena_alloc_array(arena, Attribute*, 1);
    func_op->attributes[0] = arena_alloc(arena, Attribute);
    func_op->attributes[0]->kind = ATTR_KIND_STRING;
    func_op->attributes[0]->data.string_value = str_lit("example_func");
    func_op->attributes[0]->name = str_lit("sym_name");

    // Create function region and block
    Region *func_region = arena_alloc(arena, Region);
    Block *func_block = arena_alloc(arena, Block);

    // Function block arguments (%arg0, %arg1)
    func_block->n_arguments = 2;
    func_block->arguments = arena_alloc_array(arena, ValueRef*, 2);
    func_block->arguments[0] = create_value_ref(arena, BLOCK_ARG);
    func_block->arguments[0]->result_index = 0;
    func_block->arguments[0]->type = i32_type;
    func_block->arguments[0]->register_name = str_lit("%arg0");
    func_block->arguments[1] = create_value_ref(arena, BLOCK_ARG);
    func_block->arguments[1]->result_index = 1;
    func_block->arguments[1]->type = i32_type;
    func_block->arguments[1]->register_name = str_lit("%arg1");

    // Create operations in function block
    func_block->n_operations = 4;
    func_block->operations = arena_alloc_array(arena, Operation*, 4);

    // %0 = arith.constant 5 : i32
    Operation *const_op = arena_alloc(arena, Operation);
    const_op->op_type = OP_TYPE_ARITH_CONSTANT;
    const_op->operands = NULL;
    const_op->n_operands = 0;
    const_op->n_result_types = 1;
    const_op->result_types = arena_alloc_array(arena, Type*, 1);
    const_op->result_types[0] = i32_type;
    const_op->n_attributes = 1;
    const_op->attributes = arena_alloc_array(arena, Attribute*, 1);
    const_op->attributes[0] = arena_alloc(arena, Attribute);
    const_op->attributes[0]->kind = ATTR_KIND_INTEGER;
    const_op->attributes[0]->data.integer_value = 5;
    const_op->attributes[0]->name = str_lit("value");
    const_op->regions = NULL;
    const_op->n_regions = 0;

    // Create const_result before linking
    ValueRef *const_result = create_value_ref(arena, OP_RESULT);
    const_result->def = const_op;
    const_result->result_index = 0;
    const_result->type = i32_type;
    const_result->register_name = str_lit("%0");

    const_op->results = arena_alloc_array(arena, ValueRef*, 1);
    const_op->results[0] = const_result;
    const_op->n_results = 1;
    const_op->opname = str_lit("arith.constant");

    // %1 = arith.addi %arg0, %arg1 : i32
    Operation *add_op = arena_alloc(arena, Operation);
    add_op->op_type = OP_TYPE_ARITH_ADDI;
    add_op->n_operands = 2;
    add_op->operands = arena_alloc_array(arena, ValueRef*, 2);
    add_op->operands[0] = func_block->arguments[0];
    add_op->operands[1] = func_block->arguments[1];
    add_op->n_result_types = 1;
    add_op->result_types = arena_alloc_array(arena, Type*, 1);
    add_op->result_types[0] = i32_type;
    add_op->attributes = NULL;
    add_op->n_attributes = 0;
    add_op->regions = NULL;
    add_op->n_regions = 0;
    // Create add_result before linking
    ValueRef *add_result = create_value_ref(arena, OP_RESULT);
    add_result->def = add_op;
    add_result->result_index = 1;
    add_result->type = i32_type;
    add_result->register_name = str_lit("%1");

    add_op->results = arena_alloc_array(arena, ValueRef*, 1);
    add_op->results[0] = add_result;
    add_op->n_results = 1;
    add_op->opname = str_lit("arith.addi");

    // %2 = arith.muli %1, %0 : i32 (add_result and const_result already created above)

    // Create mul_result before creating mul_op
    ValueRef *mul_result = create_value_ref(arena, OP_RESULT);
    mul_result->result_index = 2;  // This will be %2
    mul_result->type = i32_type;
    mul_result->register_name = str_lit("%2");

    Operation *mul_op = arena_alloc(arena, Operation);
    mul_op->op_type = OP_TYPE_ARITH_MULI;
    mul_op->n_operands = 2;
    mul_op->operands = arena_alloc_array(arena, ValueRef*, 2);
    mul_op->operands[0] = add_result;
    mul_op->operands[1] = const_result;
    mul_op->n_result_types = 1;
    mul_op->result_types = arena_alloc_array(arena, Type*, 1);
    mul_op->result_types[0] = i32_type;
    mul_op->attributes = NULL;
    mul_op->n_attributes = 0;
    mul_op->regions = NULL;
    mul_op->n_regions = 0;
    mul_op->results = arena_alloc_array(arena, ValueRef*, 1);
    mul_op->results[0] = mul_result;
    mul_op->n_results = 1;
    mul_op->opname = str_lit("arith.muli");

    // Set def pointer for mul_result now that mul_op exists
    mul_result->def = mul_op;

    // func.return %2 : i32

    Operation *ret_op = arena_alloc(arena, Operation);
    ret_op->op_type = OP_TYPE_FUNC_RETURN;
    ret_op->n_operands = 1;
    ret_op->operands = arena_alloc_array(arena, ValueRef*, 1);
    ret_op->operands[0] = mul_result;
    ret_op->result_types = NULL;
    ret_op->n_result_types = 0;
    ret_op->attributes = NULL;
    ret_op->n_attributes = 0;
    ret_op->regions = NULL;
    ret_op->n_regions = 0;
    ret_op->results = NULL;
    ret_op->n_results = 0;
    ret_op->opname = str_lit("func.return");

    // Link operations to function block
    func_block->operations[0] = const_op;
    func_block->operations[1] = add_op;
    func_block->operations[2] = mul_op;
    func_block->operations[3] = ret_op;

    // Link function block to function region
    func_region->n_blocks = 1;
    func_region->blocks = arena_alloc_array(arena, Block*, 1);
    func_region->blocks[0] = func_block;

    // Link function region to function operation
    func_op->n_regions = 1;
    func_op->regions = arena_alloc_array(arena, Region*, 1);
    func_op->regions[0] = func_region;

    // Link function operation to module block
    module_block->n_operations = 1;
    module_block->operations = arena_alloc_array(arena, Operation*, 1);
    module_block->operations[0] = func_op;

    // Link module block to module region
    module_region->n_blocks = 1;
    module_region->blocks = arena_alloc_array(arena, Block*, 1);
    module_region->blocks[0] = module_block;

    // Link module region to module operation
    module->n_regions = 1;
    module->regions = arena_alloc_array(arena, Region*, 1);
    module->regions[0] = module_region;

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

    Operation* op;

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

        Parser parser;
        parser_init(arena, &parser, mlir_code);
        op = parse_module(&parser);
        if (verbose) println(arena, str_lit("MLIR:"));
        if (use_classic_printer) {
            println(arena, str_lit("{}"), print_module_classic(arena, op, &parser.location_map));
        } else {
            println(arena, str_lit("{}"), print_operation_generic(arena, 0, op));
        }
        exit_code = 0;
    }

    arena_free(arena);
    return exit_code;
}

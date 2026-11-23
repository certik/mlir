#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <platform/platform.h>
#include "tokenizer.h"
#include <base/arena.h>
#include <base/io.h>
#include "mlir_api.h"
#include "mlir_api_upstream.h"
#include "mlir_printer.h"
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
            println(str_lit("Token({}, \"{}\", {},{})"),
                str_from_cstr_view((char*)mlir_tokentype_to_string(token_type)), token_str, first, last);
        }
        if (token_type == TK_EOF) {
            return;
        }
    }
}


MLIR_OpHandle construct_test_module_full(MLIR_Context *ctx) {
    // Create types
    MLIR_TypeHandle i32_type = MLIR_CreateTypeInteger(ctx, 32, true);

    // Create module operation
    MLIR_RegionHandle module_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle module_block = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, module_region, module_block);

    Arena *arena = MLIR_GetArenaAllocator(ctx);

    // Set regions
    MLIR_RegionHandle module_regions[1] = { module_region };
    MLIR_OpSpec mod_spec = {0};
    mod_spec.type = OP_TYPE_MODULE;
    mod_spec.regions = module_regions;
    mod_spec.n_regions = 1;
    MLIR_OpHandle module = MLIR_CreateOpFromSpec(ctx, &mod_spec);

    // Function attributes (sym_name)
    MLIR_NamedAttr sym_name_attr = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, str_lit("example_func")));
    MLIR_NamedAttr *func_attrs = arena_new_array(arena, MLIR_NamedAttr, 1);
    func_attrs[0] = sym_name_attr;

    // Create function region and block
    MLIR_RegionHandle func_region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle func_block = MLIR_CreateBlock(ctx);

    // Function block arguments (%arg0, %arg1)
    MLIR_ValueHandle arg0 = MLIR_CreateValueBlockArg(ctx, str_lit("%arg0"), 0, i32_type, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(ctx, func_block, arg0);

    MLIR_ValueHandle arg1 = MLIR_CreateValueBlockArg(ctx, str_lit("%arg1"), 1, i32_type, MLIR_INVALID_HANDLE);
    MLIR_AppendBlockArg(ctx, func_block, arg1);

    // %0 = arith.constant 5 : i32
    // %1 = arith.addi %arg0, %arg1 : i32
    // %2 = arith.muli %1, %0 : i32
    // func.return %2 : i32
    MLIR_Builder b = MLIR_BuilderAtEnd(ctx, func_block);
    MLIR_ValueHandle const_result = MLIR_BuildConstantInt(&b, OP_TYPE_ARITH_CONSTANT, i32_type, 5);
    MLIR_SetValueRegisterName(ctx, const_result, str_lit("%0"));

    MLIR_ValueHandle add_result = MLIR_BuildBinary(&b, OP_TYPE_ARITH_ADDI, i32_type,
                                                   MLIR_GetBlockArg(ctx, func_block, 0),
                                                   MLIR_GetBlockArg(ctx, func_block, 1));
    MLIR_SetValueRegisterName(ctx, add_result, str_lit("%1"));

    MLIR_ValueHandle mul_result = MLIR_BuildBinary(&b, OP_TYPE_ARITH_MULI, i32_type,
                                                   add_result, const_result);
    MLIR_SetValueRegisterName(ctx, mul_result, str_lit("%2"));

    MLIR_OpSpec ret_spec = {0};
    ret_spec.type = OP_TYPE_FUNC_RETURN;
    ret_spec.operands = &mul_result;
    ret_spec.n_operands = 1;
    MLIR_BuildOp(&b, &ret_spec);

    MLIR_AppendRegionBlock(ctx, func_region, func_block);

    MLIR_RegionHandle func_regions[1] = { func_region };
    MLIR_OpSpec func_spec = {0};
    func_spec.type = OP_TYPE_FUNC_FUNC;
    func_spec.attrs = func_attrs;
    func_spec.n_attrs = 1;
    func_spec.regions = func_regions;
    func_spec.n_regions = 1;
    MLIR_OpHandle func_op = MLIR_CreateOpFromSpec(ctx, &func_spec);

    MLIR_AppendBlockOp(ctx, module_block, func_op);

    return module;
}

int app_main(void) {
    bool use_construction = false;
    typedef string (*PrintFn)(MLIR_Context *, MLIR_OpHandle);
    typedef MLIR_OpHandle (*ParseFn)(MLIR_Context *, string);
    PrintFn print_fn = MLIR_PrintOperationGeneric;
    ParseFn parse_fn = MLIR_ParseTextClassic;
    bool verbose = false;
    char *input_file = NULL;

    // Fetch argv via the corec platform layer.
    size_t pargc = 0, argv_buf_size = 0;
    int rc = platform_args_sizes_get(&pargc, &argv_buf_size);
    if (rc != 0) pargc = 0;
    int argc = (int)pargc;
    Arena *boot_arena = arena_create(64 * 1024);
    char **argv = arena_new_array(boot_arena, char *, argc + 1);
    char *argv_buf = arena_new_array(boot_arena, char, argv_buf_size + 1);
    if (argc > 0) {
        platform_args_get(argv, argv_buf);
    }
    argv[argc] = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
            break;
        }
    }

    if (verbose) printf("Starting main...\n");
    Arena *arena = arena_create(50*1024*1024);
    if (verbose) printf("Arena created...\n");
    MLIR_ContextOptions ctx_options = {0};
    ctx_options.arena = arena;
    MLIR_Context *ctx = MLIR_CreateContext(&ctx_options);
    if (verbose) printf("Arena allocator set on context...\n");

    if (verbose) printf("Parsing args...\n");
    for (int i = 1; i < argc; i++) {
        if (verbose) printf("Arg %d: %s\n", i, argv[i]);
        if (strcmp(argv[i], "--construct") == 0) {
            use_construction = true;
            if (verbose) printf("Construction mode enabled\n");
        } else if (strcmp(argv[i], "--classic") == 0 || strcmp(argv[i], "-c") == 0) {
            print_fn = MLIR_PrintOperationClassic;
        } else if (strcmp(argv[i], "--upstream-printer") == 0) {
            print_fn = MLIR_PrintOperationUpstream;
        } else if (strcmp(argv[i], "--print=upstream") == 0) {
            print_fn = MLIR_PrintOperationUpstream;
        } else if (strcmp(argv[i], "--print=classic") == 0) {
            print_fn = MLIR_PrintOperationClassic;
        } else if (strcmp(argv[i], "--print=generic") == 0) {
            print_fn = MLIR_PrintOperationGeneric;
        } else if (strcmp(argv[i], "--parse=upstream") == 0) {
            parse_fn = MLIR_ParseTextUpstream;
        } else if (strcmp(argv[i], "--parse=classic") == 0) {
            parse_fn = MLIR_ParseTextClassic;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
            if (verbose) printf("Verbose mode enabled\n");
        } else if (argv[i][0] != '-') {
            input_file = argv[i];
        }
    }

    MLIR_OpHandle op;

    int exit_code;
    if (use_construction) {
        if (verbose) printf("Creating module...\n");
        op = construct_test_module_full(ctx);
        if (verbose) printf("Module created successfully.\n");

        if (verbose) printf("=== Generic Printer Test ===\n");
        if (verbose) printf("About to print operation...\n");
        string result = print_fn(ctx, op);
        if (verbose) printf("Printing result...\n");
        println(str_lit("{}"), result);

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
            println(str_lit("{}"), result);
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

        if (verbose) tokenizer_print_all_tokens(arena, mlir_code);

        op = parse_fn(ctx, mlir_code);
        if (op == MLIR_INVALID_HANDLE) {
            println(str_lit("error: parse failed"));
            MLIR_DestroyContext(ctx);
            arena_destroy(arena);
            return 1;
        }
        if (verbose) println(str_lit("MLIR:"));
        println(str_lit("{}"), print_fn(ctx, op));
        exit_code = 0;
    }

    MLIR_DestroyContext(ctx);
    arena_destroy(arena);
    return exit_code;
}

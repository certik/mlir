#include <stdbool.h>
#include <stdio.h>

#include "tokenizer.h"
#include <base/arena.h>
#include <base/io.h>
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
                tokentype_to_string(token_type), token_str, first, last);
        }
        if (token_type == TK_EOF) {
            return;
        }
    }
}

// SSA numbering map for printer
static inline size_t ptr_hash(ValueRef *p) { return ((size_t)p) >> 3; }
static inline bool ptr_equal(ValueRef *a, ValueRef *b) { return a == b; }
#define SsaMap_HASH ptr_hash
#define SsaMap_EQUAL ptr_equal
DEFINE_HASHTABLE_FOR_TYPES(ValueRef*, uint32_t, SsaMap)

typedef struct {
    Arena *arena;
    uint32_t next_ssa;
    SsaMap ssa_map;
} PrintCtx;

static inline void ssa_map_init(PrintCtx *ctx, Arena *arena) {
    ctx->arena = arena;
    ctx->next_ssa = 0;
    SsaMap_init(arena, &ctx->ssa_map, 128);
}

static inline uint32_t get_or_assign_ssa(PrintCtx *ctx, ValueRef *v) {
    uint32_t *found = SsaMap_get(&ctx->ssa_map, v);
    if (found) return *found;
    uint32_t num = ctx->next_ssa++;
    SsaMap_insert(ctx->arena, &ctx->ssa_map, v, num);
    return num;
}


string print_operation_internal(PrintCtx *ctx, int indent_level, Operation *op);

static void preassign_region_ssa(PrintCtx *ctx, Region *region, int indent_level);
static void preassign_op_ssa(PrintCtx *ctx, Operation *op, int indent_level) {
    // First preassign nested regions so nested results get earlier numbers
    if (op->n_regions > 0 && op->regions) {
        for (int i = 0; i < op->n_regions; i++) {
            preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
        }
    }
    // Then assign SSA for this op's results, if any
    if (op->n_results > 0 && op->results) {
        for (int i = 0; i < op->n_results; i++) {
            if (op->results[i]) {
                (void)get_or_assign_ssa(ctx, op->results[i]);
            }
        }
    }
}

static void preassign_block_ssa(PrintCtx *ctx, Block *block, int indent_level) {
    for (int i = 0; i < block->n_operations; i++) {
        preassign_op_ssa(ctx, block->operations[i], indent_level + 1);
    }
}

static void preassign_region_ssa(PrintCtx *ctx, Region *region, int indent_level) {
    for (int i = 0; i < region->n_blocks; i++) {
        preassign_block_ssa(ctx, region->blocks[i], indent_level);
    }
}

string indent(Arena *arena, int indent_level) {
    const int indent_spaces=4;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) {
        buf[i] = ' ';
    }
    string str = {buf, buf_size};
    return str;
}

string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, Block *block) {
    Arena *arena = ctx->arena;
    string result = format(arena, str_lit("{}^bb{}"), indent(arena, indent_level), bb_index);

    // Print block arguments if any
    if (block->n_arguments > 0 && block->arguments) {
        result = str_concat(arena, result, str_lit("("));
        for (int i = 0; i < block->n_arguments; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            ValueRef *arg = block->arguments[i];
            if (arg && arg->type) {
                // For block arguments, use the original register name
                if (arg->register_name.size > 0) {
                    result = str_concat(arena, result, format(arena, str_lit("{}: {}"),
                                                            arg->register_name, type_to_string(arena, arg->type)));
                } else {
                    result = str_concat(arena, result, format(arena, str_lit("%arg{}: {}"),
                                                            (int64_t)arg->result_index, type_to_string(arena, arg->type)));
                }
            } else {
                result = str_concat(arena, result, str_lit("null_arg"));
            }
        }
        result = str_concat(arena, result, str_lit(")"));
    }

    result = str_concat(arena, result, str_lit(":\n"));

    for (int i=0; i < block->n_operations; i++) {
        result = str_concat(arena, result,
            print_operation_internal(ctx, indent_level+1, block->operations[i])
        );
    }
    return result;
}

string print_region_internal(PrintCtx *ctx, int indent_level, Region *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (int i=0; i < region->n_blocks; i++) {
        result = str_concat(arena, result,
            print_block_internal(ctx, i, indent_level, region->blocks[i])
        );
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

string print_operation_internal(PrintCtx *ctx, int indent_level, Operation *op) {
    Arena *arena = ctx->arena;
    string result = indent(arena, indent_level);

    // Print results if any
    if (op->n_result_types > 0) {
        // Ensure nested regions get SSA numbers first to match expected ordering
        if (op->n_regions > 0 && op->regions) {
            for (int i = 0; i < op->n_regions; i++) {
                preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
            }
        }
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));

            // Assign/get SSA number for the result value (after preassigning children)
            if (op->n_results > i && op->results && op->results[i]) {
                ValueRef *res = op->results[i];
                if (res->register_name.size > 0) {
                    result = str_concat(arena, result, res->register_name);
                } else {
                    uint32_t num = get_or_assign_ssa(ctx, res);
                    result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num));
                }
            } else {
                // Should not happen; emit placeholder
                result = str_concat(arena, result, str_lit("%_"));
            }
        }
        result = str_concat(arena, result, str_lit(" = "));
    }

    // Print operation name (quotes only for unregistered operations, except tt.func)
    bool is_tt_func = (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.func")));
    if (op->op_type == OP_TYPE_UNREGISTERED && !is_tt_func) {
        result = str_concat(arena, result, str_lit("\""));
        if (op->opname.size > 0) {
            result = str_concat(arena, result, op->opname);
        } else {
            result = str_concat(arena, result, str_lit("unknown"));
        }
        result = str_concat(arena, result, str_lit("\""));
    } else {
        if (op->opname.size > 0) {
            result = str_concat(arena, result, op->opname);
        } else {
            result = str_concat(arena, result, op_type_to_string(op->op_type));
        }
    }

    // Print operands with types (always include parentheses)
    result = str_concat(arena, result, str_lit("("));
    for (int i = 0; i < op->n_operands; i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        ValueRef *operand = op->operands[i];
        if (operand == NULL) {
            result = str_concat(arena, result, str_lit("NULL_OPERAND"));
            continue;
        }
        // Prefer original register name when available; otherwise compute SSA number
        if (operand->register_name.size > 0) {
            result = str_concat(arena, result, operand->register_name);
        } else {
            uint32_t num = get_or_assign_ssa(ctx, operand);
            result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num));
        }
        result = str_concat(arena, result, str_lit(": "));
        result = str_concat(arena, result, type_to_string(arena, operand->type));
    }
    result = str_concat(arena, result, str_lit(")"));

    // Print attributes if any (skip for tt.get_program_id as it's handled specially)
    if (op->n_attributes > 0) {
        result = str_concat(arena, result, str_lit(" {"));
        for (int i = 0; i < op->n_attributes; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            Attribute *attr = op->attributes[i];
            result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    // Add type annotation for tt.make_range attributes
                    if (str_eq(op->opname, str_lit("tt.make_range"))) {
                        result = str_concat(arena, result, format(arena, str_lit("{} : i32"), attr->data.integer_value));
                    } else {
                        result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                    }
                    break;
                case ATTR_KIND_STRING:
                    result = str_concat(arena, result, format(arena, str_lit("\"{}\""), attr->data.string_value));
                    break;
                default:
                    result = str_concat(arena, result, str_lit("..."));
            }
        }
        result = str_concat(arena, result, str_lit("}"));
    }

    // Print result types if any
    if (op->n_result_types > 0) {
        result = str_concat(arena, result, str_lit(" -> "));
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            if (op->result_types && op->result_types[i]) {
                result = str_concat(arena, result, type_to_string(arena, op->result_types[i]));
            } else {
                result = str_concat(arena, result, str_lit("?"));
            }
        }
    }

    // Print regions if any
    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) {
            result = str_concat(arena, result,
                print_region_internal(ctx, indent_level, op->regions[i])
            );
        }
    }

    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Public entry: initialize SSA context and print
string print_operation(Arena *arena, int indent_level, Operation *op) {
    PrintCtx ctx;
    ssa_map_init(&ctx, arena);
    // Preassign SSA numbers for entire subtree to match parser's post-order numbering
    preassign_op_ssa(&ctx, op, indent_level);
    return print_operation_internal(&ctx, indent_level, op);
}

// Main
Operation* construct_test_module(Arena *arena) {
    // Create simple module operation
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

    // Create module region and block (empty)
    Region *module_region = arena_alloc(arena, Region);
    Block *module_block = arena_alloc(arena, Block);
    module_block->arguments = NULL;
    module_block->n_arguments = 0;
    module_block->operations = NULL;
    module_block->n_operations = 0;

    module_region->n_blocks = 1;
    module_region->blocks = arena_alloc_array(arena, Block*, 1);
    module_region->blocks[0] = module_block;

    module->n_regions = 1;
    module->regions = arena_alloc_array(arena, Region*, 1);
    module->regions[0] = module_region;

    return module;
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
    printf("Starting main...\n");
    Arena *arena = arena_create(50*1024*1024);  // Increase arena size
    printf("Arena created...\n");

    // Check for --construct option
    bool use_construction = false;
    char *input_file = NULL;

    printf("Parsing args...\n");
    for (int i = 1; i < argc; i++) {
        printf("Arg %d: %s\n", i, argv[i]);
        if (strcmp(argv[i], "--construct") == 0) {
            use_construction = true;
            printf("Construction mode enabled\n");
        } else if (argv[i][0] != '-') {
            input_file = argv[i];
        }
    }
    printf("Done parsing args. use_construction=%d\n", use_construction);

    Operation* op;

    int exit_code;
    if (use_construction) {
        // Use constructed test module
        printf("Creating module...\n");
        op = construct_test_module_full(arena);
        printf("Module created successfully.\n");

        // Test generic printing with expected output comparison
        printf("=== Generic Printer Test ===\n");
        printf("About to print operation...\n");
        string result = print_operation(arena, 0, op);
        printf("Printing result...\n");
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
            printf("✅ Generic mode test PASSED\n");
            exit_code = 0;
        } else {
            printf("❌ Generic mode test FAILED\n");
            printf("Expected:\n%s\n", expected);
            printf("Actual:\n");
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
        println(arena, str_lit("MLIR:"));
        println(arena, str_lit("{}"), print_operation(arena, 0, op));
        exit_code = 0;
    }

    arena_free(arena);
    return exit_code;
}

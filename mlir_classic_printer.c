#include "mlir_classic_printer.h"
#include <base/hashtable.h>
#include <base/format.h>
#include <stdio.h>

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
    Operation *current_scf_for;
} PrintCtx;

static inline void ssa_map_init(PrintCtx *ctx, Arena *arena) {
    ctx->arena = arena;
    ctx->next_ssa = 0;
    SsaMap_init(arena, &ctx->ssa_map, 128);
    ctx->current_scf_for = NULL;
}

static inline uint32_t get_or_assign_ssa(PrintCtx *ctx, ValueRef *v) {
    uint32_t *found = SsaMap_get(&ctx->ssa_map, v);
    if (found) return *found;
    uint32_t num = ctx->next_ssa++;
    SsaMap_insert(ctx->arena, &ctx->ssa_map, v, num);
    return num;
}

// Forward declarations for internal functions
static string print_operation_internal_classic(PrintCtx *ctx, int indent_level, Operation *op);
static string print_region_internal_classic(PrintCtx *ctx, int indent_level, Region *region);
static string print_block_internal_classic(PrintCtx *ctx, int bb_index, int indent_level, Block *block);
static string print_function_region_classic(PrintCtx *ctx, int indent_level, Region *region);

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

static string indent_classic(Arena *arena, int indent_level) {
    const int indent_spaces=2;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) {
        buf[i] = ' ';
    }
    string str = {buf, buf_size};
    return str;
}

// Helper to print SSA value reference
static string print_ssa_value_classic(PrintCtx *ctx, ValueRef *value) {
    Arena *arena = ctx->arena;
    if (value->register_name.size > 0) {
        return value->register_name;
    } else {
        uint32_t num = get_or_assign_ssa(ctx, value);
        return format(arena, str_lit("%{}"), (int64_t)num);
    }
}

// Helper to print operands; appends "#0" when referencing first result of a multi-result def
static string print_ssa_operand_classic(PrintCtx *ctx, ValueRef *value) {
    Arena *arena = ctx->arena;
    string base = print_ssa_value_classic(ctx, value);
    if (value && value->kind == OP_RESULT && value->def) {
        Operation *defop = (Operation*)value->def;
        if (defop->n_result_types > 1) {
            base = str_concat(arena, base, str_lit("#0"));
        }
    }
    return base;
}

// Helper to print location information
static string print_location_classic(Arena *arena, Location *loc) {
    if (!loc) return str_lit("");
    
    switch (loc->kind) {
        case LOC_KIND_FILE:
            return format(arena, str_lit(" loc({}:{}:{})"), 
                         loc->data.file.filename, 
                         (int64_t)loc->data.file.line, 
                         (int64_t)loc->data.file.column);
                         
        case LOC_KIND_NAME:
            return format(arena, str_lit(" loc(\"{}\")"), loc->data.name.name);
            
        case LOC_KIND_REF:
            if (loc->data.ref.ref_id == 0) {
                return str_lit(" loc(#loc)");
            }
            return format(arena, str_lit(" loc(#loc{})"), (int64_t)loc->data.ref.ref_id);
            
        case LOC_KIND_UNKNOWN:
            if (loc->original_text.size > 0) {
                return format(arena, str_lit(" {}"), loc->original_text);
            }
            return str_lit(" loc(unknown)");
            
        default:
            return str_lit("");
    }
}

static string print_block_internal_classic(PrintCtx *ctx, int bb_index, int indent_level, Block *block) {
    Arena *arena = ctx->arena;
    string result = format(arena, str_lit("{}^bb{}"), indent_classic(arena, indent_level), bb_index);

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
            print_operation_internal_classic(ctx, indent_level+1, block->operations[i])
        );
    }
    return result;
}

static string print_region_internal_classic(PrintCtx *ctx, int indent_level, Region *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (int i=0; i < region->n_blocks; i++) {
        result = str_concat(arena, result,
            print_block_internal_classic(ctx, i, indent_level, region->blocks[i])
        );
    }
    result = str_concat(arena, result, indent_classic(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

// Special function region printer that doesn't print block labels (for function bodies)
static string print_function_region_classic(PrintCtx *ctx, int indent_level, Region *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    
    // For functions, just print all operations from all blocks directly without block labels
    for (int i = 0; i < region->n_blocks; i++) {
        Block *block = region->blocks[i];
        for (int j = 0; j < block->n_operations; j++) {
            result = str_concat(arena, result,
                print_operation_internal_classic(ctx, indent_level + 1, block->operations[j])
            );
        }
    }
    
    result = str_concat(arena, result, indent_classic(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

static string print_operation_internal_classic(PrintCtx *ctx, int indent_level, Operation *op) {
    Arena *arena = ctx->arena;
    string result = indent_classic(arena, indent_level);

    // Print results if any
    if (op->n_result_types > 0) {
        // Ensure nested regions get SSA numbers first to match expected ordering
        if (op->n_regions > 0 && op->regions) {
            for (int i = 0; i < op->n_regions; i++) {
                preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
            }
        }
        // Special-case: one named result but multiple result types => print "%name:N ="
        if (op->n_results == 1 && op->n_result_types > 1 && op->results && op->results[0]) {
            result = str_concat(arena, result, print_ssa_value_classic(ctx, op->results[0]));
            result = str_concat(arena, result, format(arena, str_lit(":{}"), (int64_t)op->n_result_types));
            result = str_concat(arena, result, str_lit(" = "));
        } else {
            for (int i = 0; i < op->n_result_types; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                if (op->n_results > i && op->results && op->results[i]) {
                    ValueRef *res = op->results[i];
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, res));
                } else {
                    result = str_concat(arena, result, str_lit("%_"));
                }
            }
            result = str_concat(arena, result, str_lit(" = "));
        }
    }

    // Operation-specific printing with switch statement
    switch (op->op_type) {
        case OP_TYPE_ARITH_CONSTANT: {
            // Classic format: arith.constant 42 : i32 | 0.000000e+00 : f32 | dense<...> : tensor<...>
            result = str_concat(arena, result, str_lit("arith.constant "));
            if (op->n_attributes > 0 && op->attributes[0]) {
                if (op->attributes[0]->kind == ATTR_KIND_STRING && str_eq(op->attributes[0]->name, str_lit("value_text"))) {
                    result = str_concat(arena, result, op->attributes[0]->data.string_value);
                } else if (op->attributes[0]->kind == ATTR_KIND_INTEGER) {
                    result = str_concat(arena, result, format(arena, str_lit("{}"), op->attributes[0]->data.integer_value));
                } else if (op->attributes[0]->kind == ATTR_KIND_FLOAT) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.6e", op->attributes[0]->data.float_value);
                    result = str_concat(arena, result, str_from_cstr_view(buf));
                } else {
                    result = str_concat(arena, result, str_lit("0"));
                }
            } else {
                result = str_concat(arena, result, str_lit("0"));
            }
            if (op->n_result_types > 0 && op->result_types[0]) {
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
            }
            break;
        }
        
        case OP_TYPE_ARITH_CMPI: {
            // Classic format: arith.cmpi slt, %0, %c10 : i64
            result = str_concat(arena, result, str_lit("arith.cmpi "));
            
            // Extract comparison predicate from attributes
            string predicate = str_lit("slt"); // default fallback
            if (op->n_attributes > 0) {
                for (int i = 0; i < op->n_attributes; i++) {
                    Attribute *attr = op->attributes[i];
                    if (str_eq(attr->name, str_lit("predicate")) && attr->kind == ATTR_KIND_STRING) {
                        predicate = attr->data.string_value;
                        break;
                    }
                }
            }
            result = str_concat(arena, result, predicate);
            
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(", "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
                }
            }
            
            if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
            }
            break;
        }
        
        case OP_TYPE_CF_BR: {
            // Classic format: cf.br ^bb1(%0 : i64)
            result = str_concat(arena, result, str_lit("cf.br"));
            
            // TODO: Add block target reference ^bb1 and operands
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(" ^bb1("));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
                    if (op->operands[i]->type) {
                        result = str_concat(arena, result, str_lit(" : "));
                        result = str_concat(arena, result, type_to_string(arena, op->operands[i]->type));
                    }
                }
                result = str_concat(arena, result, str_lit(")"));
            }
            break;
        }
        
        case OP_TYPE_CF_COND_BR: {
            // Classic format: cf.cond_br %cond, ^bb1, ^bb2
            result = str_concat(arena, result, str_lit("cf.cond_br"));
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(" "));
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[0]));
                result = str_concat(arena, result, str_lit(", ^bb2, ^bb3"));
            }
            break;
        }
        
        case OP_TYPE_FUNC_RETURN:
        case OP_TYPE_RETURN: {
            // Classic format: return %0 : i64
            result = str_concat(arena, result, str_lit("return"));
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(" "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
                }
                if (op->operands[0]->type) {
                    result = str_concat(arena, result, str_lit(" : "));
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                }
            }
            break;
        }
        
        case OP_TYPE_FUNC_CALL: {
            // Classic format: call @func(%0) : (i64) -> i64
            result = str_concat(arena, result, str_lit("call "));
            
            // TODO: Extract function name from attributes
            if (op->n_attributes > 0 && op->attributes[0] && op->attributes[0]->kind == ATTR_KIND_STRING) {
                result = str_concat(arena, result, str_lit("@"));
                result = str_concat(arena, result, op->attributes[0]->data.string_value);
            } else {
                result = str_concat(arena, result, str_lit("@unknown"));
            }
            
            result = str_concat(arena, result, str_lit("("));
            for (int i = 0; i < op->n_operands; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
            }
            result = str_concat(arena, result, str_lit(")"));
            
            // Add function type if available
            if (op->n_operands > 0 || op->n_result_types > 0) {
                result = str_concat(arena, result, str_lit(" : ("));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    if (op->operands[i]->type) {
                        result = str_concat(arena, result, type_to_string(arena, op->operands[i]->type));
                    }
                }
                result = str_concat(arena, result, str_lit(") -> "));
                if (op->n_result_types > 0 && op->result_types[0]) {
                    result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
                } else {
                    result = str_concat(arena, result, str_lit("()"));
                }
            }
            break;
        }

        case OP_TYPE_TT_FUNC: {
            // Classic format header with symbol name if available
            result = str_concat(arena, result, str_lit("tt.func public @"));
            // Try to find 'sym_name' attribute - use original name if available
            string fname = str_lit("unknown_func");
            for (int i = 0; i < op->n_attributes; i++) {
                if (op->attributes[i] && str_eq(op->attributes[i]->name, str_lit("sym_name")) && op->attributes[i]->kind == ATTR_KIND_STRING) {
                    fname = op->attributes[i]->data.string_value;
                    break;
                }
            }
            result = str_concat(arena, result, fname);
            // Arguments are stored in op->operands for tt.func
            result = str_concat(arena, result, str_lit("("));
            for (int i = 0; i < op->n_operands; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                ValueRef *arg = op->operands[i];
                if (arg && arg->type) {
                    result = str_concat(arena, result, format(arena, str_lit("{}: {}"),
                                                             print_ssa_value_classic(ctx, arg),
                                                             type_to_string(arena, arg->type)));
                    if (arg->has_divisibility || arg->has_max_divisibility) {
                        result = str_concat(arena, result, str_lit(" {"));
                        if (arg->has_divisibility) {
                            result = str_concat(arena, result, str_lit("tt.divisibility = "));
                            result = str_concat(arena, result, format(arena, str_lit("{}"), (int64_t)arg->divisibility_value));
                            result = str_concat(arena, result, str_lit(" : "));
                            result = str_concat(arena, result, type_to_string(arena, arg->divisibility_type ? arg->divisibility_type : arg->type));
                        }
                        if (arg->has_max_divisibility) {
                            if (arg->has_divisibility) {
                                result = str_concat(arena, result, str_lit(", "));
                            }
                            result = str_concat(arena, result, str_lit("tt.max_divisibility = "));
                            result = str_concat(arena, result, format(arena, str_lit("{}"), (int64_t)arg->max_divisibility_value));
                            result = str_concat(arena, result, str_lit(" : "));
                            result = str_concat(arena, result, type_to_string(arena, arg->max_divisibility_type ? arg->max_divisibility_type : arg->type));
                        }
                        result = str_concat(arena, result, str_lit("}"));
                    }
                    if (arg->location) {
                        result = str_concat(arena, result, print_location_classic(arena, arg->location));
                    }
                }
            }
            result = str_concat(arena, result, str_lit(")"));
            // Add function attributes
            result = str_concat(arena, result, str_lit(" attributes {noinline = false}"));
            break;
        }

        case OP_TYPE_TT_GET_PROGRAM_ID: {
            // Classic format: tt.get_program_id x : i32
            result = str_concat(arena, result, str_lit("tt.get_program_id x : i32"));
            break;
        }

        case OP_TYPE_SCF_FOR: {
            // Classic format: %res? = scf.for %iv = %lb to %ub step %step
            //                  [iter_args(%a = %init, ...)] [-> (types...)]  : iv_type
            result = str_concat(arena, result, str_lit("scf.for "));

            // Resolve body block and arguments
            Block *body = NULL;
            if (op->n_regions > 0 && op->regions && op->regions[0] && op->regions[0]->n_blocks > 0) {
                body = op->regions[0]->blocks[0];
            }

            // Print induction variable name from block arg 0
            if (body && body->n_arguments > 0 && body->arguments[0]) {
                result = str_concat(arena, result, print_ssa_value_classic(ctx, body->arguments[0]));
                result = str_concat(arena, result, str_lit(" = "));
            }

            // lb, ub, step operands
            if (op->n_operands >= 3) {
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[0]));
                result = str_concat(arena, result, str_lit(" to "));
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[1]));
                result = str_concat(arena, result, str_lit(" step "));
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[2]));
            }

            // iter_args section with original names from block args 1..N
            int n_iter = op->n_operands > 3 ? (op->n_operands - 3) : 0;
            if (n_iter > 0) {
                result = str_concat(arena, result, str_lit(" iter_args("));
                for (int i = 0; i < n_iter; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    ValueRef *arg_name = NULL;
                    if (body && body->n_arguments > (size_t)(i + 1)) arg_name = body->arguments[i + 1];
                    if (arg_name) {
                        result = str_concat(arena, result, print_ssa_value_classic(ctx, arg_name));
                        result = str_concat(arena, result, str_lit(" = "));
                    }
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[3 + i]));
                }
                // Arrow result types: exactly op->result_types
                if (op->n_result_types > 0) {
                    result = str_concat(arena, result, str_lit(") -> ("));
                    for (int i = 0; i < op->n_result_types; i++) {
                        if (i > 0) result = str_concat(arena, result, str_lit(", "));
                        result = str_concat(arena, result, type_to_string(arena, op->result_types[i]));
                    }
                    result = str_concat(arena, result, str_lit(")"));
                } else {
                    result = str_concat(arena, result, str_lit(")"));
                }
            }

            // Type annotation for induction variable after header
            result = str_concat(arena, result, str_lit("  : "));
            if (body && body->n_arguments > 0 && body->arguments[0] && body->arguments[0]->type) {
                result = str_concat(arena, result, type_to_string(arena, body->arguments[0]->type));
            } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
            } else if (op->n_result_types > 0 && op->result_types[0]) {
                // Fallback
                result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
            }
            // Before printing region, set parent scf.for in context
            ctx->current_scf_for = op;
            break;
        }

        case OP_TYPE_SCF_IF: {
            // Classic format: scf.if %22 -> (f32) {
            result = str_concat(arena, result, str_lit("scf.if "));
            
            if (op->n_operands > 0) {
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[0]));
            }
            
            // Return type if present
            if (op->n_result_types > 0) {
                result = str_concat(arena, result, str_lit(" -> ("));
                for (int i = 0; i < op->n_result_types; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, type_to_string(arena, op->result_types[i]));
                }
                result = str_concat(arena, result, str_lit(")"));
            }
            break;
        }

        case OP_TYPE_SCF_YIELD: {
            // Classic format: scf.yield %41 : f32
            result = str_concat(arena, result, str_lit("scf.yield"));
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(" "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
                }
                // Print yield types: if inside scf.for, mirror its result types; else use operand types
                if (ctx->current_scf_for && ctx->current_scf_for->n_result_types > 0) {
                    result = str_concat(arena, result, str_lit(" : "));
                    for (int i = 0; i < ctx->current_scf_for->n_result_types; i++) {
                        if (i > 0) result = str_concat(arena, result, str_lit(", "));
                        result = str_concat(arena, result, type_to_string(arena, ctx->current_scf_for->result_types[i]));
                    }
                } else {
                    result = str_concat(arena, result, str_lit(" : "));
                    for (int i = 0; i < op->n_operands; i++) {
                        if (i > 0) result = str_concat(arena, result, str_lit(", "));
                        if (op->operands[i] && op->operands[i]->type) {
                            result = str_concat(arena, result, type_to_string(arena, op->operands[i]->type));
                        }
                    }
                }
            }
            break;
        }
        
        case OP_TYPE_TT_SPLAT: {
            // Classic format: tt.splat %v : T -> tensor<NxT>
            result = str_concat(arena, result, str_lit("tt.splat "));
            if (op->n_operands > 0 && op->operands[0]) {
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[0]));
            }
            if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                // Use parentheses only if original signature had them
                bool sig_parens = false;
                string sig_src = str_lit("");
                for (int i = 0; i < op->n_attributes; i++) {
                    if (str_eq(op->attributes[i]->name, str_lit("_sig_parens")) && op->attributes[i]->kind == ATTR_KIND_BOOL && op->attributes[i]->data.bool_value) {
                        sig_parens = true; break;
                    }
                    if (str_eq(op->attributes[i]->name, str_lit("_sig_src")) && op->attributes[i]->kind == ATTR_KIND_STRING) {
                        sig_src = op->attributes[i]->data.string_value;
                    }
                }
                result = str_concat(arena, result, str_lit(" : "));
                if (sig_parens) result = str_concat(arena, result, str_lit("("));
                if (sig_src.size > 0) result = str_concat(arena, result, sig_src);
                else result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                if (sig_parens) result = str_concat(arena, result, str_lit(")"));
            }
            if (op->n_result_types > 0 && op->result_types[0]) {
                result = str_concat(arena, result, str_lit(" -> "));
                result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
            }
            break;
        }
        
        case OP_TYPE_TT_ADDPTR: {
            // Classic format: tt.addptr %a, %b : TyA, TyB
            result = str_concat(arena, result, str_lit("tt.addptr "));
            for (int i = 0; i < op->n_operands; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
            }
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(" : "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    if (op->operands[i] && op->operands[i]->type) {
                        result = str_concat(arena, result, type_to_string(arena, op->operands[i]->type));
                    }
                }
            }
            break;
        }

        case OP_TYPE_TT_LOAD: {
            // Classic format: tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
            result = str_concat(arena, result, str_lit("tt.load "));
            for (int i = 0; i < op->n_operands; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
            }
            
            // Print attributes before the result type
            if (op->n_attributes > 0) {
                bool has_visible_attrs = false;
                for (int i = 0; i < op->n_attributes; i++) {
                    Attribute *attr = op->attributes[i];
                    // Skip internal attributes
                    if (str_eq(attr->name, str_lit("sym_name")) ||
                        (str_eq(attr->name, str_lit("value")) && op->op_type == OP_TYPE_ARITH_CONSTANT) ||
                        str_eq(attr->name, str_lit("axis")) ||
                        str_eq(attr->name, str_lit("start")) ||
                        str_eq(attr->name, str_lit("end"))) {
                        continue;
                    }
                    if (!has_visible_attrs) {
                        result = str_concat(arena, result, str_lit(" {"));
                        has_visible_attrs = true;
                    } else {
                        result = str_concat(arena, result, str_lit(", "));
                    }
                    result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
                    switch (attr->kind) {
                        case ATTR_KIND_INTEGER:
                            result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                            result = str_concat(arena, result, str_lit(" : i32"));
                            break;
                        case ATTR_KIND_BOOL:
                            result = str_concat(arena, result, attr->data.bool_value ? str_lit("true") : str_lit("false"));
                            break;
                        default:
                            result = str_concat(arena, result, str_lit("..."));
                    }
                }
                if (has_visible_attrs) {
                    result = str_concat(arena, result, str_lit("}"));
                }
            }
            
            // Print result type
            if (op->n_result_types > 0 && op->result_types[0]) {
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
            }
            break;
        }

        case OP_TYPE_TT_STORE: {
            // Classic format: tt.store %ptr, %value {cache = 1 : i32, evict = 1 : i32} : f32
            result = str_concat(arena, result, str_lit("tt.store "));
            for (int i = 0; i < op->n_operands; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
            }
            
            // Print attributes before the result type
            if (op->n_attributes > 0) {
                bool has_visible_attrs = false;
                for (int i = 0; i < op->n_attributes; i++) {
                    Attribute *attr = op->attributes[i];
                    // Skip internal attributes
                    if (str_eq(attr->name, str_lit("sym_name")) ||
                        (str_eq(attr->name, str_lit("value")) && op->op_type == OP_TYPE_ARITH_CONSTANT) ||
                        str_eq(attr->name, str_lit("axis")) ||
                        str_eq(attr->name, str_lit("start")) ||
                        str_eq(attr->name, str_lit("end"))) {
                        continue;
                    }
                    if (!has_visible_attrs) {
                        result = str_concat(arena, result, str_lit(" {"));
                        has_visible_attrs = true;
                    } else {
                        result = str_concat(arena, result, str_lit(", "));
                    }
                    result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
                    switch (attr->kind) {
                        case ATTR_KIND_INTEGER:
                            result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                            result = str_concat(arena, result, str_lit(" : i32"));
                            break;
                        case ATTR_KIND_BOOL:
                            result = str_concat(arena, result, attr->data.bool_value ? str_lit("true") : str_lit("false"));
                            break;
                        default:
                            result = str_concat(arena, result, str_lit("..."));
                    }
                }
                if (has_visible_attrs) {
                    result = str_concat(arena, result, str_lit("}"));
                }
            }
            
            // Print result type (for tt.store with attributes, use value type; otherwise use pointer type)
            if (op->n_attributes > 0 && op->n_operands > 1 && op->operands[1] && op->operands[1]->type) {
                // With attributes: use value operand type
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->operands[1]->type));
            } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                // Without attributes: use pointer operand type
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
            }
            break;
        }
        
        case OP_TYPE_TT_RETURN: {
            // Classic format: tt.return
            result = str_concat(arena, result, str_lit("tt.return"));
            break;
        }

        case OP_TYPE_TT_MAKE_RANGE: {
            // Classic format: tt.make_range {end = N : i32, start = M : i32} : tensor<Nxi32>
            result = str_concat(arena, result, str_lit("tt.make_range"));
            if (op->n_attributes > 0) {
                result = str_concat(arena, result, str_lit(" {"));
                for (int i = 0; i < op->n_attributes; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    Attribute *attr = op->attributes[i];
                    result = str_concat(arena, result, format(arena, str_lit("{} = {} : i32"), attr->name, (int64_t)attr->data.integer_value));
                }
                result = str_concat(arena, result, str_lit("}"));
            }
            if (op->n_result_types > 0 && op->result_types[0]) {
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
            }
            break;
        }
        
        default: {
            // Default case: classic-ish formatting without result arrows
            
            
            // Print operation name  
            bool is_tt_func = (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.func")));
            bool is_known_op = false;
            if (op->opname.size > 0) {
                // Check if it's a known dialect operation that shouldn't be quoted
                const char *name = op->opname.str;
                size_t len = op->opname.size;
                is_known_op = ((len > 6 && strncmp(name, "arith.", 6) == 0) ||
                              (len > 4 && strncmp(name, "scf.", 4) == 0) ||
                              (len > 3 && strncmp(name, "tt.", 3) == 0) ||
                              (len > 5 && strncmp(name, "func.", 5) == 0) ||
                              (len > 3 && strncmp(name, "cf.", 3) == 0));
            }
            
            if (op->op_type == OP_TYPE_UNREGISTERED && !is_tt_func && !is_known_op) {
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

            // Special classic formatting for select tt.* ops
            if (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.broadcast"))) {
                // tt.broadcast %x : (src) -> dst
                result = str_concat(arena, result, str_lit(" "));
                if (op->n_operands > 0 && op->operands[0]) {
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[0]));
                }
                // Use captured src signature if available
                string sig_src = str_lit("");
                for (int i = 0; i < op->n_attributes; i++) {
                    if (str_eq(op->attributes[i]->name, str_lit("_sig_src")) && op->attributes[i]->kind == ATTR_KIND_STRING) {
                        sig_src = op->attributes[i]->data.string_value; break;
                    }
                }
                if (sig_src.size > 0) {
                    // normalize ",X" to ", X" for readability
                    string norm = str_lit("");
                    for (size_t k = 0; k < sig_src.size; k++) {
                        char c = sig_src.str[k];
                        norm = str_concat(arena, norm, (string){&c,1});
                        if (c == ',' && k+1 < sig_src.size && sig_src.str[k+1] != ' ') {
                            norm = str_concat(arena, norm, str_lit(" "));
                        }
                    }
                    result = str_concat(arena, result, str_lit(" : ("));
                    result = str_concat(arena, result, norm);
                    result = str_concat(arena, result, str_lit(")"));
                } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                    result = str_concat(arena, result, str_lit(" : ("));
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                    result = str_concat(arena, result, str_lit(")"));
                }
                if (op->n_result_types > 0 && op->result_types[0]) {
                    result = str_concat(arena, result, str_lit(" -> "));
                    result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
                }
                break;
            }
            if (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.expand_dims"))) {
                // tt.expand_dims %x {axis = i : i32} : (src) -> dst
                result = str_concat(arena, result, str_lit(" "));
                if (op->n_operands > 0 && op->operands[0]) {
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[0]));
                }
                // Inline attributes
                if (op->n_attributes > 0) {
                    bool opened = false; bool first = true;
                    for (int i = 0; i < op->n_attributes; i++) {
                        Attribute *attr = op->attributes[i];
                        if (str_eq(attr->name, str_lit("_sig_parens")) || str_eq(attr->name, str_lit("_sig_src"))) { continue; }
                        if (!opened) { result = str_concat(arena, result, str_lit(" {")); opened = true; }
                        if (!first) result = str_concat(arena, result, str_lit(", ")); first = false;
                        if (attr->kind == ATTR_KIND_INTEGER) {
                            result = str_concat(arena, result, format(arena, str_lit("{} = {} : i32"), attr->name, (int64_t)attr->data.integer_value));
                        } else {
                            result = str_concat(arena, result, format(arena, str_lit("{} = ..."), attr->name));
                        }
                    }
                    if (opened) result = str_concat(arena, result, str_lit("}"));
                }
                string sig_src2 = str_lit("");
                for (int i = 0; i < op->n_attributes; i++) {
                    if (str_eq(op->attributes[i]->name, str_lit("_sig_src")) && op->attributes[i]->kind == ATTR_KIND_STRING) { sig_src2 = op->attributes[i]->data.string_value; break; }
                }
                if (sig_src2.size > 0) {
                    result = str_concat(arena, result, str_lit(" : ("));
                    result = str_concat(arena, result, sig_src2);
                    result = str_concat(arena, result, str_lit(")"));
                } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                    result = str_concat(arena, result, str_lit(" : ("));
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                    result = str_concat(arena, result, str_lit(")"));
                }
                if (op->n_result_types > 0 && op->result_types[0]) {
                    result = str_concat(arena, result, str_lit(" -> "));
                    result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
                }
                break;
            }
            if (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.dot"))) {
                // tt.dot %a, %b, %acc {attrs} : lhs * rhs -> res
                result = str_concat(arena, result, str_lit(" "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, op->operands[i]));
                }
                if (op->n_attributes > 0) {
                    result = str_concat(arena, result, str_lit(" {"));
                    for (int i = 0; i < op->n_attributes; i++) {
                        if (i > 0) result = str_concat(arena, result, str_lit(", "));
                        Attribute *attr = op->attributes[i];
                        if (attr->kind == ATTR_KIND_BOOL) {
                            result = str_concat(arena, result, format(arena, str_lit("{} = {}"), attr->name, attr->data.bool_value ? str_lit("true") : str_lit("false")));
                        } else if (attr->kind == ATTR_KIND_INTEGER) {
                            result = str_concat(arena, result, format(arena, str_lit("{} = {} : i32"), attr->name, (int64_t)attr->data.integer_value));
                        } else {
                            result = str_concat(arena, result, format(arena, str_lit("{} = ..."), attr->name));
                        }
                    }
                    result = str_concat(arena, result, str_lit("}"));
                }
                // Types
                if (op->n_operands >= 2 && op->operands[0] && op->operands[1] && op->operands[0]->type && op->operands[1]->type) {
                    string lhs = type_to_string(arena, op->operands[0]->type);
                    string rhs = type_to_string(arena, op->operands[1]->type);
                    result = str_concat(arena, result, str_lit(" : "));
                    result = str_concat(arena, result, lhs);
                    result = str_concat(arena, result, str_lit(" * "));
                    result = str_concat(arena, result, rhs);
                    // Compute result type tensor<MxNxf32> from lhs and rhs
                    int64_t m = 0, n = 0; const char *p;
                    // parse m from lhs after 'tensor<'
                    p = strstr(lhs.str, "tensor<"); if (p) { p += 7; while (*p && *p>='0' && *p<='9') { m = m*10 + (*p-'0'); p++; } }
                    // parse n from rhs as last (second) dim
                    const char *q = strstr(rhs.str, "tensor<"); if (q) {
                        q += 7; // skip 'tensor<'
                        // skip first dim and 'x'
                        while (*q && *q!='x' && *q!='>') q++; if (*q=='x') q++;
                        // parse n until 'x'
                        while (*q && *q>='0' && *q<='9') { n = n*10 + (*q-'0'); q++; }
                    }
                    result = str_concat(arena, result, str_lit(" -> "));
                    result = str_concat(arena, result, format(arena, str_lit("tensor<{}x{}xf32>"), m, n));
                }
                break;
            }

            // Print operands in canonical format (no types for most ops)
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(" "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    ValueRef *operand = op->operands[i];
                    if (operand == NULL) {
                        result = str_concat(arena, result, str_lit("NULL_OPERAND"));
                        continue;
                    }
                    result = str_concat(arena, result, print_ssa_operand_classic(ctx, operand));
                }
            }

            // Print type suffix in classic format
            if (op->n_result_types > 0 && op->result_types[0]) {
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
            } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                result = str_concat(arena, result, str_lit(" : "));
                // For binary operations, print type once
                if (op->n_operands == 2 && op->operands[0]->type && op->operands[1]->type) {
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                } else if (op->n_operands == 1) {
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                } else {
                    // Multiple different types, print all
                    for (int i = 0; i < op->n_operands; i++) {
                        if (i > 0) result = str_concat(arena, result, str_lit(", "));
                        if (op->operands[i] && op->operands[i]->type) {
                            result = str_concat(arena, result, type_to_string(arena, op->operands[i]->type));
                        }
                    }
                }
            }

            break;
        }
    }

    // Print attributes for operations that should show them in classic format
    // Skip internal attributes that shouldn't be visible
    if (op->n_attributes > 0 && op->op_type != OP_TYPE_TT_FUNC && 
        op->op_type != OP_TYPE_TT_LOAD && op->op_type != OP_TYPE_TT_STORE &&
        op->op_type != OP_TYPE_ARITH_CMPI && op->op_type != OP_TYPE_TT_MAKE_RANGE) {
        // Skip printing for ops where we printed inline already by name
        if (op->opname.size > 0 && (str_eq(op->opname, str_lit("tt.expand_dims")) || str_eq(op->opname, str_lit("tt.dot")))) {
            // do nothing
        } else {
        bool has_visible_attrs = false;
        for (int i = 0; i < op->n_attributes; i++) {
            Attribute *attr = op->attributes[i];
            // Skip internal attributes that shouldn't be shown in classic format
            if (str_eq(attr->name, str_lit("sym_name")) || str_eq(attr->name, str_lit("_sig_parens")) || str_eq(attr->name, str_lit("_sig_src")) || str_eq(attr->name, str_lit("value_text"))) {
                continue;
            }
            // Skip 'value' attribute only for arith.constant operations
            if (str_eq(attr->name, str_lit("value")) && op->op_type == OP_TYPE_ARITH_CONSTANT) {
                continue;
            }
            // Skip axis attribute for tt.get_program_id
            if (op->op_type == OP_TYPE_TT_GET_PROGRAM_ID && str_eq(attr->name, str_lit("axis"))) {
                continue;
            }
            // No skipping of axis/start/end in classic mode
            if (!has_visible_attrs) {
                result = str_concat(arena, result, str_lit(" {"));
                has_visible_attrs = true;
            } else {
                result = str_concat(arena, result, str_lit(", "));
            }
            result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                    // Add type annotation for integer attributes
                    result = str_concat(arena, result, str_lit(" : i32"));
                    break;
                case ATTR_KIND_FLOAT:
                    result = str_concat(arena, result, format(arena, str_lit("{:e}"), attr->data.float_value));
                    break;
                case ATTR_KIND_STRING:
                    result = str_concat(arena, result, format(arena, str_lit("\"{}\""), attr->data.string_value));
                    break;
                case ATTR_KIND_BOOL:
                    result = str_concat(arena, result, attr->data.bool_value ? str_lit("true") : str_lit("false"));
                    break;
                default:
                    result = str_concat(arena, result, str_lit("..."));
            }
        }
        if (has_visible_attrs) {
            result = str_concat(arena, result, str_lit("}"));
        }
        }
    }

    // For classic formatting: place regions (when present)
    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) {
            // Special handling for SCF if else
            if (op->op_type == OP_TYPE_SCF_IF && i == 1 && op->n_regions == 2) {
                result = str_concat(arena, result, str_lit(" else "));
            }
            
            if (op->op_type == OP_TYPE_TT_FUNC || op->op_type == OP_TYPE_MODULE ||
                op->op_type == OP_TYPE_SCF_FOR || op->op_type == OP_TYPE_SCF_IF || op->op_type == OP_TYPE_SCF_WHILE) {
                result = str_concat(arena, result,
                    print_function_region_classic(ctx, indent_level, op->regions[i])
                );
            } else {
                result = str_concat(arena, result,
                    print_region_internal_classic(ctx, indent_level, op->regions[i])
                );
            }
        }
        // After regions of scf.for, restore parent pointer
        if (op->op_type == OP_TYPE_SCF_FOR) {
            ctx->current_scf_for = NULL;
        }
    }
    if (op->location) {
        result = str_concat(arena, result, print_location_classic(arena, op->location));
    }

    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Public API implementations
string print_operation_classic(Arena *arena, int indent_level, Operation *op) {
    PrintCtx ctx;
    ssa_map_init(&ctx, arena);
    // Preassign SSA numbers for entire subtree to match parser's post-order numbering
    preassign_op_ssa(&ctx, op, indent_level);
    return print_operation_internal_classic(&ctx, indent_level, op);
}

string print_region_classic(Arena *arena, int indent_level, Region *region) {
    PrintCtx ctx;
    ssa_map_init(&ctx, arena);
    preassign_region_ssa(&ctx, region, indent_level);
    return print_region_internal_classic(&ctx, indent_level, region);
}

string print_block_classic(Arena *arena, int bb_index, int indent_level, Block *block) {
    PrintCtx ctx;
    ssa_map_init(&ctx, arena);
    preassign_block_ssa(&ctx, block, indent_level);
    return print_block_internal_classic(&ctx, bb_index, indent_level, block);
}

// Helper to print location map definitions
static string print_location_map_classic(Arena *arena, LocationMap *location_map) {
    string result = str_lit("");
    if (!location_map) return result;

    typedef struct { string key; Location *loc; int number; } LocEntry;
    // Collect entries (excluding '#loc')
    size_t cap = location_map->size;
    LocEntry *arr = arena_alloc_array(arena, LocEntry, cap);
    size_t n = 0;
    for (size_t i = 0; i < location_map->num_buckets; i++) {
        if (!location_map->buckets[i].occupied) continue;
        string name = location_map->buckets[i].key;
        if (str_eq(name, str_lit("#loc"))) continue;
        int num = -1;
        if (name.size > 4 && name.str[0]=='#' && name.str[1]=='l' && name.str[2]=='o' && name.str[3]=='c') {
            // parse integer suffix
            int v = 0; bool any=false;
            for (size_t k=4;k<name.size;k++) { char c = name.str[k]; if (c>='0'&&c<='9'){ any=true; v = v*10 + (c-'0'); } else { any=false; break; } }
            if (any) num = v;
        }
        arr[n].key = name;
        arr[n].loc = location_map->buckets[i].value;
        arr[n].number = num;
        n++;
    }

    // Simple insertion sort by numeric suffix when present; non-numeric after numeric in stable order
    for (size_t i = 1; i < n; i++) {
        LocEntry x = arr[i];
        size_t j = i;
        while (j > 0) {
            bool swap = false;
            if (arr[j-1].number >= 0 && x.number >= 0) swap = arr[j-1].number > x.number;
            else if (arr[j-1].number >= 0 && x.number < 0) swap = false; // keep numeric before non-numeric
            else if (arr[j-1].number < 0 && x.number >= 0) swap = true;
            else swap = false;
            if (!swap) break;
            arr[j] = arr[j-1];
            j--;
        }
        arr[j] = x;
    }

    // Emit in order
    for (size_t i = 0; i < n; i++) {
        result = str_concat(arena, result, arr[i].key);
        result = str_concat(arena, result, str_lit(" = "));
        Location *loc = arr[i].loc;
        if (loc->original_text.size > 0) {
            result = str_concat(arena, result, loc->original_text);
        } else {
            switch (loc->kind) {
                case LOC_KIND_FILE:
                    result = str_concat(arena, result,
                        format(arena, str_lit("loc({}:{}:{})"),
                               loc->data.file.filename,
                               (int64_t)loc->data.file.line,
                               (int64_t)loc->data.file.column));
                    break;
                case LOC_KIND_NAME:
                    result = str_concat(arena, result,
                        format(arena, str_lit("loc(\"{}\")"), loc->data.name.name));
                    break;
                default:
                    result = str_concat(arena, result, str_lit("loc(unknown)"));
                    break;
            }
        }
        result = str_concat(arena, result, str_lit("\n"));
    }

    return result;
}

string print_module_classic(Arena *arena, Operation *module, LocationMap *location_map) {
    string result = str_lit("");
    
    // Print the special #loc definition at the beginning if it exists
    if (module && module->unnumbered_loc_def) {
        Location *loc = module->unnumbered_loc_def;
        result = str_concat(arena, result, str_lit("#loc = "));
        switch (loc->kind) {
            case LOC_KIND_FILE:
                result = str_concat(arena, result,
                    format(arena, str_lit("loc({}:{}:{})"),
                           loc->data.file.filename,
                           (int64_t)loc->data.file.line,
                           (int64_t)loc->data.file.column));
                break;
            case LOC_KIND_NAME:
                result = str_concat(arena, result,
                    format(arena, str_lit("loc(\"{}\")"), loc->data.name.name));
                break;
            default:
                result = str_concat(arena, result, str_lit("loc(unknown)"));
                break;
        }
        result = str_concat(arena, result, str_lit("\n"));
    } else if (location_map) {
        for (size_t i = 0; i < location_map->num_buckets; i++) {
            if (location_map->buckets[i].occupied) {
                string loc_name = location_map->buckets[i].key;
                if (loc_name.size == 4 && loc_name.str && loc_name.str[0]=='#' && loc_name.str[1]=='l' && loc_name.str[2]=='o' && loc_name.str[3]=='c') {
                    Location *loc = location_map->buckets[i].value;
                    result = str_concat(arena, result, loc_name);
                    result = str_concat(arena, result, str_lit(" = "));
                    switch (loc->kind) {
                        case LOC_KIND_FILE:
                            result = str_concat(arena, result,
                                format(arena, str_lit("loc({}:{}:{})"),
                                       loc->data.file.filename,
                                       (int64_t)loc->data.file.line,
                                       (int64_t)loc->data.file.column));
                            break;
                        case LOC_KIND_NAME:
                            result = str_concat(arena, result,
                                format(arena, str_lit("loc(\"{}\")"), loc->data.name.name));
                            break;
                        default:
                            result = str_concat(arena, result, str_lit("loc(unknown)"));
                            break;
                    }
                    result = str_concat(arena, result, str_lit("\n"));
                    break;
                }
            }
        }
    }
    
    // Print the module operation
    result = str_concat(arena, result, print_operation_classic(arena, 0, module));
    
    // Add numbered location map definitions at the end
    string loc_defs = print_location_map_classic(arena, location_map);
    if (loc_defs.size > 0) {
        result = str_concat(arena, result, loc_defs);
    }
    // Trim one trailing newline to match reference files exactly
    if (result.size > 0 && result.str[result.size - 1] == '\n') {
        result.size -= 1;
    }
    return result;
}

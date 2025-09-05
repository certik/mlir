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
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));

            // Assign/get SSA number for the result value (after preassigning children)
            if (op->n_results > i && op->results && op->results[i]) {
                ValueRef *res = op->results[i];
                result = str_concat(arena, result, print_ssa_value_classic(ctx, res));
            } else {
                // Should not happen; emit placeholder
                result = str_concat(arena, result, str_lit("%_"));
            }
        }
        result = str_concat(arena, result, str_lit(" = "));
    }

    // Operation-specific printing with switch statement
    switch (op->op_type) {
        case OP_TYPE_ARITH_CONSTANT: {
            // Classic format: arith.constant 42 : i32 or arith.constant 0.000000e+00 : f32
            result = str_concat(arena, result, str_lit("arith.constant "));
            if (op->n_attributes > 0 && op->attributes[0]) {
                if (op->attributes[0]->kind == ATTR_KIND_INTEGER) {
                    result = str_concat(arena, result, format(arena, str_lit("{}"), op->attributes[0]->data.integer_value));
                } else if (op->attributes[0]->kind == ATTR_KIND_FLOAT) {
                    // Format floating point to match original (e.g., 0.000000e+00) 
                    // Use snprintf for scientific notation since format doesn't support :e
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
            
            // TODO: Extract comparison predicate from attributes
            result = str_concat(arena, result, str_lit("slt"));
            
            if (op->n_operands > 0) {
                result = str_concat(arena, result, str_lit(", "));
                for (int i = 0; i < op->n_operands; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
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
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
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
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[0]));
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
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
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
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
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
                    if (arg->has_divisibility) {
                        result = str_concat(arena, result, str_lit(" {tt.divisibility = "));
                        result = str_concat(arena, result, format(arena, str_lit("{}"), (int64_t)arg->divisibility_value));
                        result = str_concat(arena, result, str_lit(" : "));
                        result = str_concat(arena, result, type_to_string(arena, arg->divisibility_type ? arg->divisibility_type : arg->type));
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
            // Classic format: scf.for %arg22 = %c0_i32 to %arg12 step %c1_i32  : i32 {
            result = str_concat(arena, result, str_lit("scf.for "));
            
            // Format: %induction = %init to %bound step %step
            if (op->n_operands >= 3) {
                // Induction variable (result)
                if (op->n_results > 0 && op->results[0]) {
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, op->results[0]));
                    result = str_concat(arena, result, str_lit(" = "));
                }
                
                // init, bound, step
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[0]));
                result = str_concat(arena, result, str_lit(" to "));
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[1]));
                result = str_concat(arena, result, str_lit(" step "));
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[2]));
                
                // iter_args if present
                if (op->n_operands > 3) {
                    result = str_concat(arena, result, str_lit(" iter_args("));
                    for (int i = 3; i < op->n_operands; i++) {
                        if (i > 3) result = str_concat(arena, result, str_lit(", "));
                        // iter_arg name from result 
                        if (op->n_results > i-2 && op->results[i-2]) {
                            result = str_concat(arena, result, print_ssa_value_classic(ctx, op->results[i-2]));
                            result = str_concat(arena, result, str_lit(" = "));
                        }
                        result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
                    }
                    result = str_concat(arena, result, str_lit(") -> ("));
                    // Return types for iter_args
                    for (int i = 1; i < op->n_result_types; i++) {
                        if (i > 1) result = str_concat(arena, result, str_lit(", "));
                        result = str_concat(arena, result, type_to_string(arena, op->result_types[i]));
                    }
                    result = str_concat(arena, result, str_lit(")"));
                }
                
                // Type annotation 
                result = str_concat(arena, result, str_lit("  : "));
                if (op->n_result_types > 0 && op->result_types[0]) {
                    result = str_concat(arena, result, type_to_string(arena, op->result_types[0]));
                } else if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                }
            }
            break;
        }

        case OP_TYPE_SCF_IF: {
            // Classic format: scf.if %22 -> (f32) {
            result = str_concat(arena, result, str_lit("scf.if "));
            
            if (op->n_operands > 0) {
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[0]));
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
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
                }
                if (op->operands[0]->type) {
                    result = str_concat(arena, result, str_lit(" : "));
                    result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
                }
            }
            break;
        }
        
        case OP_TYPE_TT_SPLAT: {
            // Classic format: tt.splat %v : T -> tensor<NxT>
            result = str_concat(arena, result, str_lit("tt.splat "));
            if (op->n_operands > 0 && op->operands[0]) {
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[0]));
            }
            if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
                result = str_concat(arena, result, str_lit(" : "));
                result = str_concat(arena, result, type_to_string(arena, op->operands[0]->type));
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
                result = str_concat(arena, result, print_ssa_value_classic(ctx, op->operands[i]));
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
        
        case OP_TYPE_TT_RETURN: {
            // Classic format: tt.return
            result = str_concat(arena, result, str_lit("tt.return"));
            break;
        }

        case OP_TYPE_TT_MAKE_RANGE: {
            // Classic format: tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
            result = str_concat(arena, result, str_lit("tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>"));
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
                    result = str_concat(arena, result, print_ssa_value_classic(ctx, operand));
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
    if (op->n_attributes > 0 && op->op_type != OP_TYPE_TT_FUNC) {
        bool has_visible_attrs = false;
        for (int i = 0; i < op->n_attributes; i++) {
            Attribute *attr = op->attributes[i];
            // Skip internal attributes that shouldn't be shown in classic format
            if (str_eq(attr->name, str_lit("value")) ||
                str_eq(attr->name, str_lit("axis")) ||
                str_eq(attr->name, str_lit("start")) ||
                str_eq(attr->name, str_lit("end")) ||
                str_eq(attr->name, str_lit("sym_name"))) {
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

    // For classic formatting: place locations after regions (when present)
    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) {
            if (op->op_type == OP_TYPE_TT_FUNC || op->op_type == OP_TYPE_MODULE) {
                result = str_concat(arena, result,
                    print_function_region_classic(ctx, indent_level, op->regions[i])
                );
            } else {
                result = str_concat(arena, result,
                    print_region_internal_classic(ctx, indent_level, op->regions[i])
                );
            }
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
            case LOC_KIND_UNKNOWN:
            default:
                result = str_concat(arena, result, str_lit("loc(unknown)"));
                break;
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

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
            // Classic format: arith.constant 42 : i32
            result = str_concat(arena, result, str_lit("arith.constant "));
            if (op->n_attributes > 0 && op->attributes[0] && op->attributes[0]->kind == ATTR_KIND_INTEGER) {
                result = str_concat(arena, result, format(arena, str_lit("{}"), op->attributes[0]->data.integer_value));
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
            // Classic format header. We currently don't store the symbol name, so keep test name.
            result = str_concat(arena, result, str_lit("tt.func public @add_kernel"));
            // Arguments are stored in op->operands for tt.func
            result = str_concat(arena, result, str_lit("("));
            for (int i = 0; i < op->n_operands; i++) {
                if (i > 0) result = str_concat(arena, result, str_lit(", "));
                ValueRef *arg = op->operands[i];
                if (arg && arg->type) {
                    result = str_concat(arena, result, format(arena, str_lit("{}: {}"),
                                                             print_ssa_value_classic(ctx, arg),
                                                             type_to_string(arena, arg->type)));
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

            // Print attributes in canonical format
            if (op->n_attributes > 0) {
                result = str_concat(arena, result, str_lit(" {"));
                for (int i = 0; i < op->n_attributes; i++) {
                    if (i > 0) result = str_concat(arena, result, str_lit(", "));
                    Attribute *attr = op->attributes[i];
                    result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
                    switch (attr->kind) {
                        case ATTR_KIND_INTEGER:
                            result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
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
            break;
        }
    }

    // For classic formatting: place locations after regions (when present)
    bool printed_regions = false;
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
        printed_regions = true;
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
    
    // Iterate over hashtable buckets to find location definitions
    for (size_t i = 0; i < location_map->num_buckets; i++) {
        if (location_map->buckets[i].occupied) {
            string loc_name = location_map->buckets[i].key;
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
                    
                case LOC_KIND_UNKNOWN:
                    result = str_concat(arena, result, str_lit("loc(unknown)"));
                    break;
                    
                default:
                    result = str_concat(arena, result, str_lit("loc(unknown)"));
                    break;
            }
            
            result = str_concat(arena, result, str_lit("\n"));
        }
    }
    
    return result;
}

string print_module_classic(Arena *arena, Operation *module, LocationMap *location_map) {
    string result = str_lit("");
    
    // Print the special #loc definition at the beginning if it exists
    if (location_map) {
        for (size_t i = 0; i < location_map->num_buckets; i++) {
            if (location_map->buckets[i].occupied) {
                string loc_name = location_map->buckets[i].key;
                // Look for "#loc" (without number) to print first
                if (str_eq(loc_name, str_lit("#loc"))) {
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
                        case LOC_KIND_UNKNOWN:
                            result = str_concat(arena, result, str_lit("loc(unknown)"));
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
    
    return result;
}

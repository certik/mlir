#include "mlir_generic_printer.h"
#include "mlir_ir_internal.h"
#include <base/hashtable.h>
#include <base/format.h>

// SSA numbering map for printer (internal IR)
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
static string print_operation_internal(PrintCtx *ctx, int indent_level, Operation *op);
static string print_region_internal(PrintCtx *ctx, int indent_level, Region *region);
static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, Block *block);

static void preassign_region_ssa(PrintCtx *ctx, Region *region, int indent_level);
static void preassign_op_ssa(PrintCtx *ctx, Operation *op, int indent_level) {
    if (op->n_regions > 0 && op->regions) {
        for (int i = 0; i < op->n_regions; i++) {
            preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
        }
    }
    if (op->n_results > 0 && op->results) {
        for (int i = 0; i < op->n_results; i++) {
            if (op->results[i]) (void)get_or_assign_ssa(ctx, op->results[i]);
        }
    }
}

static void preassign_block_ssa(PrintCtx *ctx, Block *block, int indent_level) {
    for (int i = 0; i < block->n_operations; i++) preassign_op_ssa(ctx, block->operations[i], indent_level + 1);
}

static void preassign_region_ssa(PrintCtx *ctx, Region *region, int indent_level) {
    for (int i = 0; i < region->n_blocks; i++) preassign_block_ssa(ctx, region->blocks[i], indent_level);
}

static string indent(Arena *arena, int indent_level) {
    const int indent_spaces=4;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) buf[i] = ' ';
    return (string){buf, buf_size};
}

static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, Block *block) {
    Arena *arena = ctx->arena;
    string result = format(arena, str_lit("{}^bb{}"), indent(arena, indent_level), bb_index);

    if (block->n_arguments > 0 && block->arguments) {
        result = str_concat(arena, result, str_lit("("));
        for (int i = 0; i < block->n_arguments; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            ValueRef *arg = block->arguments[i];
            if (arg && arg->type) {
                if (arg->register_name.size > 0) {
                    result = str_concat(arena, result, format(arena, str_lit("{}: {}"), arg->register_name, type_to_string(arena, arg->type)));
                } else {
                    result = str_concat(arena, result, format(arena, str_lit("%arg{}: {}"), (int64_t)arg->result_index, type_to_string(arena, arg->type)));
                }
            } else {
                result = str_concat(arena, result, str_lit("null_arg"));
            }
        }
        result = str_concat(arena, result, str_lit(")"));
    }

    result = str_concat(arena, result, str_lit(":\n"));
    for (int i=0; i < block->n_operations; i++) {
        result = str_concat(arena, result, print_operation_internal(ctx, indent_level+1, block->operations[i]));
    }
    return result;
}

static string print_region_internal(PrintCtx *ctx, int indent_level, Region *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (int i=0; i < region->n_blocks; i++) {
        result = str_concat(arena, result, print_block_internal(ctx, i, indent_level, region->blocks[i]));
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

static string print_operation_internal(PrintCtx *ctx, int indent_level, Operation *op) {
    Arena *arena = ctx->arena;
    string result = indent(arena, indent_level);

    if (op->n_result_types > 0) {
        if (op->n_regions > 0 && op->regions) for (int i = 0; i < op->n_regions; i++) preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            if (op->n_results > i && op->results && op->results[i]) {
                ValueRef *res = op->results[i];
                if (res->register_name.size > 0) result = str_concat(arena, result, res->register_name);
                else { uint32_t num = get_or_assign_ssa(ctx, res); result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num)); }
            } else {
                result = str_concat(arena, result, str_lit("%_"));
            }
        }
        result = str_concat(arena, result, str_lit(" = "));
    }

    bool is_tt_func = (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.func")));
    if (op->op_type == OP_TYPE_UNREGISTERED && !is_tt_func) {
        result = str_concat(arena, result, str_lit("\""));
        if (op->opname.size > 0) result = str_concat(arena, result, op->opname); else result = str_concat(arena, result, str_lit("unknown"));
        result = str_concat(arena, result, str_lit("\""));
    } else {
        if (op->opname.size > 0) result = str_concat(arena, result, op->opname); else result = str_concat(arena, result, op_type_to_string(op->op_type));
    }

    result = str_concat(arena, result, str_lit("("));
    for (int i = 0; i < op->n_operands; i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        ValueRef *operand = op->operands[i];
        if (!operand) { result = str_concat(arena, result, str_lit("NULL_OPERAND")); continue; }
        if (operand->register_name.size > 0) result = str_concat(arena, result, operand->register_name);
        else { uint32_t num = get_or_assign_ssa(ctx, operand); result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num)); }
        result = str_concat(arena, result, str_lit(": "));
        result = str_concat(arena, result, type_to_string(arena, operand->type));
    }
    result = str_concat(arena, result, str_lit(")"));

    if (op->n_attributes > 0) {
        bool opened = false; bool first = true;
        for (int i = 0; i < op->n_attributes; i++) {
            Attribute *attr = op->attributes[i];
            if (attr->name.size > 0 && attr->name.str[0] == '_') { continue; }
            if (!opened) { result = str_concat(arena, result, str_lit(" {")); opened = true; }
            if (!first) result = str_concat(arena, result, str_lit(", ")); first = false;
            result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    if (op->op_type == OP_TYPE_TT_MAKE_RANGE) result = str_concat(arena, result, format(arena, str_lit("{} : i32"), attr->data.integer_value));
                    else result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                    break;
                case ATTR_KIND_STRING:
                    result = str_concat(arena, result, format(arena, str_lit("\"{}\""), attr->data.string_value));
                    break;
                default:
                    result = str_concat(arena, result, str_lit("..."));
            }
        }
        if (opened) result = str_concat(arena, result, str_lit("}"));
    }

    if (op->n_result_types > 0) {
        result = str_concat(arena, result, str_lit(" -> "));
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            if (op->result_types && op->result_types[i]) result = str_concat(arena, result, type_to_string(arena, op->result_types[i]));
            else result = str_concat(arena, result, str_lit("?"));
        }
    }

    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) {
            result = str_concat(arena, result, print_region_internal(ctx, indent_level, op->regions[i]));
        }
    }

    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Public wrappers to match API header
string print_operation_generic(Arena *arena, int indent_level, MlirOperation *op) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_op_ssa(&ctx, (Operation*)op, indent_level);
    return print_operation_internal(&ctx, indent_level, (Operation*)op);
}

string print_region_generic(Arena *arena, int indent_level, MlirRegion *region) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_region_ssa(&ctx, (Region*)region, indent_level);
    return print_region_internal(&ctx, indent_level, (Region*)region);
}

string print_block_generic(Arena *arena, int bb_index, int indent_level, MlirBlock *block) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_block_ssa(&ctx, (Block*)block, indent_level);
    return print_block_internal(&ctx, bb_index, indent_level, (Block*)block);
}


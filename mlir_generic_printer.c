#include "mlir_generic_printer.h"
#include <base/hashtable.h>
#include <base/format.h>

// SSA numbering map for printer (API values)
static inline size_t ptr_hash(MlirValue *p) { return ((size_t)p) >> 3; }
static inline bool ptr_equal(MlirValue *a, MlirValue *b) { return a == b; }
#define SsaMap_HASH ptr_hash
#define SsaMap_EQUAL ptr_equal
DEFINE_HASHTABLE_FOR_TYPES(MlirValue*, uint32_t, SsaMap)

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

static inline uint32_t get_or_assign_ssa(PrintCtx *ctx, MlirValue *v) {
    uint32_t *found = SsaMap_get(&ctx->ssa_map, v);
    if (found) return *found;
    uint32_t num = ctx->next_ssa++;
    SsaMap_insert(ctx->arena, &ctx->ssa_map, v, num);
    return num;
}

// Forward declarations (API-based)
static string print_operation_internal(PrintCtx *ctx, int indent_level, MlirOperation *op);
static string print_region_internal(PrintCtx *ctx, int indent_level, MlirRegion *region);
static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, MlirBlock *block);

static void preassign_region_ssa(PrintCtx *ctx, MlirRegion *region, int indent_level);
static void preassign_op_ssa(PrintCtx *ctx, MlirOperation *op, int indent_level) {
    for (size_t i = 0; i < mlir_operation_num_regions(op); i++) {
        preassign_region_ssa(ctx, mlir_operation_get_region(op, i), indent_level + 1);
    }
    for (size_t i = 0; i < mlir_operation_num_results(op); i++) {
        MlirValue *res = mlir_operation_get_result(op, i);
        if (res) (void)get_or_assign_ssa(ctx, res);
    }
}

static void preassign_block_ssa(PrintCtx *ctx, MlirBlock *block, int indent_level) {
    for (size_t i = 0; i < mlir_block_num_operations(block); i++) {
        preassign_op_ssa(ctx, mlir_block_get_operation(block, i), indent_level + 1);
    }
}

static void preassign_region_ssa(PrintCtx *ctx, MlirRegion *region, int indent_level) {
    for (size_t i = 0; i < mlir_region_num_blocks(region); i++) {
        preassign_block_ssa(ctx, mlir_region_get_block(region, i), indent_level);
    }
}

static string indent(Arena *arena, int indent_level) {
    const int indent_spaces=4;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) buf[i] = ' ';
    return (string){buf, buf_size};
}

static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, MlirBlock *block) {
    Arena *arena = ctx->arena;
    string result = format(arena, str_lit("{}^bb{}"), indent(arena, indent_level), bb_index);

    if (mlir_block_num_arguments(block) > 0) {
        result = str_concat(arena, result, str_lit("("));
        for (size_t i = 0; i < mlir_block_num_arguments(block); i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            MlirValue *arg = mlir_block_get_argument(block, i);
            MlirType *ty = mlir_value_get_type(arg);
            if (arg && ty) {
                string reg = mlir_value_get_register_name(arg);
                if (reg.size > 0) {
                    result = str_concat(arena, result, format(arena, str_lit("{}: {}"), reg, mlir_type_to_string(arena, ty)));
                } else {
                    result = str_concat(arena, result, format(arena, str_lit("%arg{}: {}"), (int64_t)mlir_value_get_result_index(arg), mlir_type_to_string(arena, ty)));
                }
            } else {
                result = str_concat(arena, result, str_lit("null_arg"));
            }
        }
        result = str_concat(arena, result, str_lit(")"));
    }

    result = str_concat(arena, result, str_lit(":\n"));
    for (size_t i=0; i < mlir_block_num_operations(block); i++) {
        result = str_concat(arena, result, print_operation_internal(ctx, indent_level+1, mlir_block_get_operation(block, i)));
    }
    return result;
}

static string print_region_internal(PrintCtx *ctx, int indent_level, MlirRegion *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (size_t i=0; i < mlir_region_num_blocks(region); i++) {
        result = str_concat(arena, result, print_block_internal(ctx, (int)i, indent_level, mlir_region_get_block(region, i)));
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

static string print_operation_internal(PrintCtx *ctx, int indent_level, MlirOperation *op) {
    Arena *arena = ctx->arena;
    string result = indent(arena, indent_level);

    if (mlir_operation_num_result_types(op) > 0) {
        for (size_t i = 0; i < mlir_operation_num_regions(op); i++) {
            preassign_region_ssa(ctx, mlir_operation_get_region(op, i), indent_level + 1);
        }
        for (size_t i = 0; i < mlir_operation_num_results(op); i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            MlirValue *res = mlir_operation_get_result(op, i);
            string rn = mlir_value_get_register_name(res);
            if (rn.size > 0) result = str_concat(arena, result, rn);
            else {
                uint32_t num = get_or_assign_ssa(ctx, res);
                result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num));
            }
        }
        result = str_concat(arena, result, str_lit(" = "));
    }

    if (mlir_operation_get_type(op) == OP_TYPE_UNREGISTERED) {
        result = str_concat(arena, result, str_lit("\""));
        const char *nm = mlir_operation_get_name(op); if (!nm) nm = "unknown";
        result = str_concat(arena, result, str_from_cstr_view((char*)nm));
        result = str_concat(arena, result, str_lit("\""));
    } else {
        result = str_concat(arena, result, str_from_cstr_view((char*)mlir_op_type_to_string(mlir_operation_get_type(op))));
    }

    result = str_concat(arena, result, str_lit("("));
    for (size_t i = 0; i < mlir_operation_num_operands(op); i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        MlirValue *operand = mlir_operation_get_operand(op, i);
        if (!operand) { result = str_concat(arena, result, str_lit("NULL_OPERAND")); continue; }
        string rn2 = mlir_value_get_register_name(operand);
        if (rn2.size > 0) result = str_concat(arena, result, rn2);
        else { uint32_t num = get_or_assign_ssa(ctx, operand); result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num)); }
        result = str_concat(arena, result, str_lit(": "));
        result = str_concat(arena, result, mlir_type_to_string(arena, mlir_value_get_type(operand)));
    }
    result = str_concat(arena, result, str_lit(")"));

    if (mlir_operation_num_attributes(op) > 0) {
        bool opened = false; bool first = true;
        for (size_t i = 0; i < mlir_operation_num_attributes(op); i++) {
            MlirAttribute *attr = mlir_operation_get_attribute(op, i);
            string an = mlir_attribute_get_name(attr);
            if (an.size > 0 && an.str[0] == '_') { continue; }
            if (!opened) { result = str_concat(arena, result, str_lit(" {")); opened = true; }
            if (!first) result = str_concat(arena, result, str_lit(", ")); first = false;
            result = str_concat(arena, result, format(arena, str_lit("{} = "), an));
            MlirAttrKind ak = mlir_attribute_get_kind(attr);
            if (ak == MLIR_ATTR_KIND_INTEGER) {
                if (mlir_operation_get_type(op) == OP_TYPE_TT_MAKE_RANGE) result = str_concat(arena, result, format(arena, str_lit("{} : i32"), mlir_attribute_get_integer(attr)));
                else result = str_concat(arena, result, format(arena, str_lit("{}"), mlir_attribute_get_integer(attr)));
            } else if (ak == MLIR_ATTR_KIND_STRING) {
                result = str_concat(arena, result, format(arena, str_lit("\"{}\""), mlir_attribute_get_string(attr)));
            } else {
                result = str_concat(arena, result, str_lit("..."));
            }
        }
        if (opened) result = str_concat(arena, result, str_lit("}"));
    }

    if (mlir_operation_num_result_types(op) > 0) {
        result = str_concat(arena, result, str_lit(" -> "));
        for (size_t i = 0; i < mlir_operation_num_result_types(op); i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            MlirType *ty = mlir_operation_get_result_type(op, i);
            if (ty) result = str_concat(arena, result, mlir_type_to_string(arena, ty)); else result = str_concat(arena, result, str_lit("?"));
        }
    }

    if (mlir_operation_num_regions(op) > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (size_t i = 0; i < mlir_operation_num_regions(op); i++) {
            result = str_concat(arena, result, print_region_internal(ctx, indent_level, mlir_operation_get_region(op, i)));
        }
    }

    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Public API implementations
string print_operation_generic(Arena *arena, int indent_level, MlirOperation *op) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_op_ssa(&ctx, op, indent_level);
    return print_operation_internal(&ctx, indent_level, op);
}

string print_region_generic(Arena *arena, int indent_level, MlirRegion *region) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_region_ssa(&ctx, region, indent_level);
    return print_region_internal(&ctx, indent_level, region);
}

string print_block_generic(Arena *arena, int bb_index, int indent_level, MlirBlock *block) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_block_ssa(&ctx, block, indent_level);
    return print_block_internal(&ctx, bb_index, indent_level, block);
}


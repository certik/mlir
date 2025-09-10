// Generic printer (internal IR for now, wrapped for API)
#include "mlir_generic_printer.h"
#include "mlir_ir_internal.h"
#include <base/hashtable.h>
#include <base/format.h>

// During migration, keep SSA numbering keyed by API values (MlirValue*).
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

static string print_operation_internal(PrintCtx *ctx, int indent_level, Operation *op);
static string print_region_internal(PrintCtx *ctx, int indent_level, Region *region);
static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, Block *block);

static void preassign_region_ssa(PrintCtx *ctx, Region *region, int indent_level);
static void preassign_op_ssa(PrintCtx *ctx, Operation *op, int indent_level) {
    if (op->n_regions > 0 && op->regions) {
        for (int i = 0; i < op->n_regions; i++) preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
    }
    if (op->n_results > 0 && op->results) {
        for (int i = 0; i < op->n_results; i++) if (op->results[i]) (void)get_or_assign_ssa(ctx, (MlirValue*)op->results[i]);
    }
}

static void preassign_block_ssa(PrintCtx *ctx, Block *block, int indent_level) {
    MlirBlock *b = (MlirBlock*)block;
    size_t n = mlir_block_num_operations(b);
    for (size_t i = 0; i < n; i++) {
        MlirOperation *op = mlir_block_get_operation(b, i);
        preassign_op_ssa(ctx, (Operation*)op, indent_level + 1);
    }
}

static void preassign_region_ssa(PrintCtx *ctx, Region *region, int indent_level) {
    MlirRegion *r = (MlirRegion*)region;
    size_t n = mlir_region_num_blocks(r);
    for (size_t i = 0; i < n; i++) {
        MlirBlock *b = mlir_region_get_block(r, i);
        preassign_block_ssa(ctx, (Block*)b, indent_level);
    }
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
    size_t n_args = mlir_block_num_arguments((MlirBlock*)block);
    if (n_args > 0) {
        result = str_concat(arena, result, str_lit("("));
        for (size_t i = 0; i < n_args; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            MlirValue *arg = mlir_block_get_argument((MlirBlock*)block, i);
            MlirType *arg_ty = arg ? mlir_value_get_type(arg) : NULL;
            if (arg && arg_ty) {
                string rname = mlir_value_get_register_name(arg);
                if (rname.size > 0) result = str_concat(arena, result, format(arena, str_lit("{}: {}"), rname, mlir_type_to_string(arena, arg_ty)));
                else result = str_concat(arena, result, format(arena, str_lit("%arg{}: {}"), (int64_t)mlir_value_get_result_index(arg), mlir_type_to_string(arena, arg_ty)));
            } else {
                result = str_concat(arena, result, str_lit("null_arg"));
            }
        }
        result = str_concat(arena, result, str_lit(")"));
    }
    result = str_concat(arena, result, str_lit(":\n"));
    for (size_t i = 0, e = mlir_block_num_operations((MlirBlock*)block); i < e; i++) {
        MlirOperation *op = mlir_block_get_operation((MlirBlock*)block, i);
        result = str_concat(arena, result, print_operation_internal(ctx, indent_level+1, (Operation*)op));
    }
    return result;
}

static string print_region_internal(PrintCtx *ctx, int indent_level, Region *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (size_t i = 0, e = mlir_region_num_blocks((MlirRegion*)region); i < e; i++) {
        MlirBlock *b = mlir_region_get_block((MlirRegion*)region, i);
        result = str_concat(arena, result, print_block_internal(ctx, (int)i, indent_level, (Block*)b));
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

static string print_operation_internal(PrintCtx *ctx, int indent_level, Operation *op) {
    Arena *arena = ctx->arena;
    string result = indent(arena, indent_level);
    size_t api_num_result_types = mlir_operation_num_result_types((MlirOperation*)op);
    if (api_num_result_types > 0) {
        if (op->n_regions > 0 && op->regions) for (int i = 0; i < op->n_regions; i++) preassign_region_ssa(ctx, op->regions[i], indent_level + 1);
        for (size_t i = 0; i < api_num_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            size_t api_num_results = mlir_operation_num_results((MlirOperation*)op);
            if (api_num_results > i) {
                MlirValue *res = mlir_operation_get_result((MlirOperation*)op, i);
                if (res) {
                    string name = mlir_value_get_register_name(res);
                    if (name.size > 0) result = str_concat(arena, result, name);
                    else { uint32_t num = get_or_assign_ssa(ctx, res); result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num)); }
                } else {
                    result = str_concat(arena, result, str_lit("%_"));
                }
            } else {
                result = str_concat(arena, result, str_lit("%_"));
            }
        }
        result = str_concat(arena, result, str_lit(" = "));
    }
    if (op->op_type == OP_TYPE_UNREGISTERED) {
        result = str_concat(arena, result, str_lit("\""));
        if (op->opname.size > 0) result = str_concat(arena, result, op->opname); else result = str_concat(arena, result, str_lit("unknown"));
        result = str_concat(arena, result, str_lit("\""));
    } else {
        if (op->opname.size > 0) result = str_concat(arena, result, op->opname); else result = str_concat(arena, result, op_type_to_string(op->op_type));
    }
    result = str_concat(arena, result, str_lit("("));
    for (size_t i = 0, e = mlir_operation_num_operands((MlirOperation*)op); i < e; i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        MlirValue *operand = mlir_operation_get_operand((MlirOperation*)op, i);
        if (!operand) { result = str_concat(arena, result, str_lit("NULL_OPERAND")); continue; }
        string name = mlir_value_get_register_name(operand);
        if (name.size > 0) result = str_concat(arena, result, name);
        else { uint32_t num = get_or_assign_ssa(ctx, operand); result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num)); }
        result = str_concat(arena, result, str_lit(": "));
        MlirType *ot = mlir_value_get_type(operand);
        result = str_concat(arena, result, mlir_type_to_string(arena, ot));
    }
    result = str_concat(arena, result, str_lit(")"));
    {
        size_t n_attrs = mlir_operation_num_attributes((MlirOperation*)op);
        if (n_attrs > 0) {
            bool opened = false; bool first = true;
            OpType opty = mlir_operation_get_type((MlirOperation*)op);
            for (size_t i = 0; i < n_attrs; i++) {
                MlirAttribute *attr = mlir_operation_get_attribute((MlirOperation*)op, i);
                string name = mlir_attribute_get_name(attr);
                if (name.size > 0 && name.str[0] == '_') { continue; }
                if (!opened) { result = str_concat(arena, result, str_lit(" {")); opened = true; }
                if (!first) result = str_concat(arena, result, str_lit(", ")); first = false;
                result = str_concat(arena, result, format(arena, str_lit("{} = "), name));
                switch (mlir_attribute_get_kind(attr)) {
                    case MLIR_ATTR_KIND_INTEGER: {
                        int64_t v = mlir_attribute_get_integer(attr);
                        if (opty == OP_TYPE_TT_MAKE_RANGE) result = str_concat(arena, result, format(arena, str_lit("{} : i32"), v));
                        else result = str_concat(arena, result, format(arena, str_lit("{}"), v));
                        break;
                    }
                    case MLIR_ATTR_KIND_STRING: {
                        string s = mlir_attribute_get_string(attr);
                        result = str_concat(arena, result, format(arena, str_lit("\"{}\""), s));
                        break;
                    }
                    default:
                        result = str_concat(arena, result, str_lit("..."));
                }
            }
            if (opened) result = str_concat(arena, result, str_lit("}"));
        }
    }
    if (api_num_result_types > 0) {
        result = str_concat(arena, result, str_lit(" -> "));
        for (size_t i = 0; i < api_num_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            MlirType *rt = mlir_operation_get_result_type((MlirOperation*)op, i);
            if (rt) result = str_concat(arena, result, mlir_type_to_string(arena, rt)); else result = str_concat(arena, result, str_lit("?"));
        }
    }
    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) result = str_concat(arena, result, print_region_internal(ctx, indent_level, op->regions[i]));
    }
    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Public wrappers to match API header while still using internal IR during migration
// duplicate wrappers removed
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

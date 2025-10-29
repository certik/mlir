// Generic printer (internal IR for now, wrapped for API)
#include "mlir_generic_printer.h"
#include <base/hashtable.h>
#include <base/format.h>

// During migration, keep SSA numbering keyed by API values (MLIR_Value*).
static inline size_t ptr_hash(MLIR_Value *p) { return ((size_t)p) >> 3; }
static inline bool ptr_equal(MLIR_Value *a, MLIR_Value *b) { return a == b; }
#define SsaMap_HASH ptr_hash
#define SsaMap_EQUAL ptr_equal
DEFINE_HASHTABLE_FOR_TYPES(MLIR_Value*, uint32_t, SsaMap)

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

static inline uint32_t get_or_assign_ssa(PrintCtx *ctx, MLIR_Value *v) {
    uint32_t *found = SsaMap_get(&ctx->ssa_map, v);
    if (found) return *found;
    uint32_t num = ctx->next_ssa++;
    SsaMap_insert(ctx->arena, &ctx->ssa_map, v, num);
    return num;
}

static string print_operation_internal(PrintCtx *ctx, int indent_level, MLIR_Op *op);
static string print_region_internal(PrintCtx *ctx, int indent_level, MLIR_Region *region);
static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, MLIR_Block *block);

static void preassign_region_ssa(PrintCtx *ctx, MLIR_Region *region, int indent_level);
static void preassign_op_ssa(PrintCtx *ctx, MLIR_Op *op, int indent_level) {
    size_t n_regions = MLIR_GetOpNumRegions(op);
    for (size_t i = 0; i < n_regions; i++) {
        MLIR_Region *region = MLIR_GetOpRegion(op, i);
        if (region) preassign_region_ssa(ctx, region, indent_level + 1);
    }
    size_t n_results = MLIR_GetOpNumResults(op);
    for (size_t i = 0; i < n_results; i++) {
        MLIR_Value *result = MLIR_GetOpResult(op, i);
        if (result) (void)get_or_assign_ssa(ctx, result);
    }
}

static void preassign_block_ssa(PrintCtx *ctx, MLIR_Block *block, int indent_level) {
    MLIR_Block *b = block;
    size_t n = MLIR_GetBlockNumOps(b);
    for (size_t i = 0; i < n; i++) {
        MLIR_Op *op = MLIR_GetBlockOp(b, i);
        preassign_op_ssa(ctx, op, indent_level + 1);
    }
}

static void preassign_region_ssa(PrintCtx *ctx, MLIR_Region *region, int indent_level) {
    MLIR_Region *r = region;
    size_t n = MLIR_GetRegionNumBlocks(r);
    for (size_t i = 0; i < n; i++) {
        MLIR_Block *b = MLIR_GetRegionBlock(r, i);
        preassign_block_ssa(ctx, b, indent_level);
    }
}

static string indent(Arena *arena, int indent_level) {
    const int indent_spaces=4;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) buf[i] = ' ';
    return (string){buf, buf_size};
}

static string print_block_internal(PrintCtx *ctx, int bb_index, int indent_level, MLIR_Block *block) {
    Arena *arena = ctx->arena;
    string result = format(arena, str_lit("{}^bb{}"), indent(arena, indent_level), bb_index);
    size_t n_args = MLIR_GetBlockNumArgs(block);
    if (n_args > 0) {
        result = str_concat(arena, result, str_lit("("));
        for (size_t i = 0; i < n_args; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            MLIR_Value *arg = MLIR_GetBlockArg(block, i);
            MLIR_Type *arg_ty = arg ? MLIR_GetValueType(arg) : NULL;
            if (arg && arg_ty) {
                string rname = MLIR_GetValueRegisterName(arg);
                if (rname.size > 0) result = str_concat(arena, result, format(arena, str_lit("{}: {}"), rname, MLIR_GetTypeString(arena, arg_ty)));
                else result = str_concat(arena, result, format(arena, str_lit("%arg{}: {}"), (int64_t)MLIR_GetValueResultIndex(arg), MLIR_GetTypeString(arena, arg_ty)));
            } else {
                result = str_concat(arena, result, str_lit("null_arg"));
            }
        }
        result = str_concat(arena, result, str_lit(")"));
    }
    result = str_concat(arena, result, str_lit(":\n"));
    for (size_t i = 0, e = MLIR_GetBlockNumOps(block); i < e; i++) {
        MLIR_Op *op = MLIR_GetBlockOp(block, i);
        result = str_concat(arena, result, print_operation_internal(ctx, indent_level+1, op));
    }
    return result;
}

static string print_region_internal(PrintCtx *ctx, int indent_level, MLIR_Region *region) {
    Arena *arena = ctx->arena;
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (size_t i = 0, e = MLIR_GetRegionNumBlocks((MLIR_Region*)region); i < e; i++) {
        MLIR_Block *b = MLIR_GetRegionBlock((MLIR_Region*)region, i);
        result = str_concat(arena, result, print_block_internal(ctx, (int)i, indent_level, b));
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

static string print_operation_internal(PrintCtx *ctx, int indent_level, MLIR_Op *op) {
    Arena *arena = ctx->arena;
    string result = indent(arena, indent_level);
    size_t api_num_result_types = MLIR_GetOpNumResultTypes(op);
    if (api_num_result_types > 0) {
        size_t n_regions = MLIR_GetOpNumRegions(op);
        for (size_t i = 0; i < n_regions; i++) {
            MLIR_Region *region = MLIR_GetOpRegion(op, i);
            if (region) preassign_region_ssa(ctx, region, indent_level + 1);
        }
        for (size_t i = 0; i < api_num_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            size_t api_num_results = MLIR_GetOpNumResults(op);
            if (api_num_results > i) {
                MLIR_Value *res = MLIR_GetOpResult(op, i);
                if (res) {
                    string name = MLIR_GetValueRegisterName(res);
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
    MLIR_OpType op_type = MLIR_GetOpType(op);
    string opname = MLIR_GetOpName_string(op);
    if (op_type == OP_TYPE_UNREGISTERED) {
        result = str_concat(arena, result, str_lit("\""));
        if (opname.size > 0) result = str_concat(arena, result, opname);
        else result = str_concat(arena, result, str_lit("unknown"));
        result = str_concat(arena, result, str_lit("\""));
    } else {
        if (opname.size > 0) result = str_concat(arena, result, opname);
        else result = str_concat(arena, result, MLIR_MLIR_OpTypeToString(op_type));
    }
    result = str_concat(arena, result, str_lit("("));
    for (size_t i = 0, e = MLIR_GetOpNumOperands(op); i < e; i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        MLIR_Value *operand = MLIR_GetOpOperand(op, i);
        if (!operand) { result = str_concat(arena, result, str_lit("NULL_OPERAND")); continue; }
        string name = MLIR_GetValueRegisterName(operand);
        if (name.size > 0) result = str_concat(arena, result, name);
        else { uint32_t num = get_or_assign_ssa(ctx, operand); result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)num)); }
        result = str_concat(arena, result, str_lit(": "));
        MLIR_Type *ot = MLIR_GetValueType(operand);
        result = str_concat(arena, result, MLIR_GetTypeString(arena, ot));
    }
    result = str_concat(arena, result, str_lit(")"));
    {
        size_t n_attrs = MLIR_GetOpNumAttributes(op);
        if (n_attrs > 0) {
            bool opened = false; bool first = true;
            MLIR_OpType opty = MLIR_GetOpType(op);
            for (size_t i = 0; i < n_attrs; i++) {
                MLIR_Attribute *attr = MLIR_GetOpAttribute(op, i);
                string name = MLIR_GetAttributeName(attr);
                if (name.size > 0 && name.str[0] == '_') { continue; }
                if (!opened) { result = str_concat(arena, result, str_lit(" {")); opened = true; }
                if (!first) result = str_concat(arena, result, str_lit(", ")); first = false;
                result = str_concat(arena, result, format(arena, str_lit("{} = "), name));
                switch (MLIR_GetAttributeKind(attr)) {
                    case MLIR_ATTR_KIND_INTEGER: {
                        int64_t v = MLIR_GetAttributeInteger(attr);
                        if (opty == OP_TYPE_TT_MAKE_RANGE) result = str_concat(arena, result, format(arena, str_lit("{} : i32"), v));
                        else result = str_concat(arena, result, format(arena, str_lit("{}"), v));
                        break;
                    }
                    case MLIR_ATTR_KIND_STRING: {
                        string s = MLIR_GetAttributeString(attr);
                        result = str_concat(arena, result, format(arena, str_lit("\"{}\""), s));
                        break;
                    }
                    case MLIR_ATTR_KIND_ARRAY: {
                        result = str_concat(arena, result, str_lit("["));
                        size_t arr_size = MLIR_GetAttributeArraySize(attr);
                        for (size_t j = 0; j < arr_size; j++) {
                            if (j > 0) result = str_concat(arena, result, str_lit(", "));
                            MLIR_Attribute *elem = MLIR_GetAttributeArrayElement(attr, j);
                            if (elem) {
                                // Recursively print the element
                                if (MLIR_GetAttributeKind(elem) == MLIR_ATTR_KIND_DICT) {
                                    // Print dictionary inline
                                    result = str_concat(arena, result, str_lit("{"));
                                    size_t dict_size = MLIR_GetAttributeDictSize(elem);
                                    for (size_t k = 0; k < dict_size; k++) {
                                        if (k > 0) result = str_concat(arena, result, str_lit(", "));
                                        MLIR_Attribute *dict_elem = MLIR_GetAttributeDictElement(elem, k);
                                        if (dict_elem) {
                                            string elem_name = MLIR_GetAttributeName(dict_elem);
                                            result = str_concat(arena, result, elem_name);
                                            result = str_concat(arena, result, str_lit(" = "));
                                            int64_t val = MLIR_GetAttributeInteger(dict_elem);
                                            result = str_concat(arena, result, format(arena, str_lit("{} : i32"), val));
                                        }
                                    }
                                    result = str_concat(arena, result, str_lit("}"));
                                } else {
                                    result = str_concat(arena, result, str_lit("..."));
                                }
                            }
                        }
                        result = str_concat(arena, result, str_lit("]"));
                        break;
                    }
                    case MLIR_ATTR_KIND_DICT: {
                        result = str_concat(arena, result, str_lit("{"));
                        size_t dict_size = MLIR_GetAttributeDictSize(attr);
                        for (size_t j = 0; j < dict_size; j++) {
                            if (j > 0) result = str_concat(arena, result, str_lit(", "));
                            MLIR_Attribute *elem = MLIR_GetAttributeDictElement(attr, j);
                            if (elem) {
                                string elem_name = MLIR_GetAttributeName(elem);
                                result = str_concat(arena, result, elem_name);
                                result = str_concat(arena, result, str_lit(" = "));
                                int64_t val = MLIR_GetAttributeInteger(elem);
                                result = str_concat(arena, result, format(arena, str_lit("{} : i32"), val));
                            }
                        }
                        result = str_concat(arena, result, str_lit("}"));
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
            MLIR_Type *rt = MLIR_GetOpResult_type(op, i);
            if (rt) result = str_concat(arena, result, MLIR_GetTypeString(arena, rt)); else result = str_concat(arena, result, str_lit("?"));
        }
    }
    size_t n_regions = MLIR_GetOpNumRegions(op);
    if (n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (size_t i = 0; i < n_regions; i++) {
            MLIR_Region *region = MLIR_GetOpRegion(op, i);
            if (region) result = str_concat(arena, result, print_region_internal(ctx, indent_level, region));
        }
    }
    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Public wrappers to match API header while still using internal IR during migration
// duplicate wrappers removed
string print_operation_generic(Arena *arena, int indent_level, MLIR_Op *op) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_op_ssa(&ctx, op, indent_level);
    return print_operation_internal(&ctx, indent_level, op);
}

string print_region_generic(Arena *arena, int indent_level, MLIR_Region *region) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_region_ssa(&ctx, region, indent_level);
    return print_region_internal(&ctx, indent_level, region);
}

string print_block_generic(Arena *arena, int bb_index, int indent_level, MLIR_Block *block) {
    PrintCtx ctx; ssa_map_init(&ctx, arena);
    preassign_block_ssa(&ctx, block, indent_level);
    return print_block_internal(&ctx, bb_index, indent_level, block);
}

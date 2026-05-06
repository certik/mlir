#include <stdio.h>
#include <base/arena.h>
#include <base/string.h>
#include <base/format.h>
#include <base/hashtable.h>
#include <base/vector.h>
#include "tokenizer.h"
#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_parser.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Internal concrete IR definitions

typedef enum {
    TYPE_KIND_UNKNOWN,
    TYPE_KIND_OPAQUE,
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX,
    TYPE_KIND_POINTER
} TypeKind;

typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,
    LOC_KIND_NAME,
    LOC_KIND_CALLSITE,
    LOC_KIND_FUSED,
    LOC_KIND_REF
} LocationKind;

// Internal struct definitions using handle fields

typedef struct IR_Type {
    TypeKind kind;
    union {
        struct {
            uint32_t width;
            bool is_signed;
        } integer;
        struct {
            uint32_t width;
            bool is_bfloat;
        } floating;
        struct {
            MLIR_TypeHandle element_type;
            int64_t *shape;
            uint32_t rank;
        } shaped;
        struct {
            MLIR_TypeHandle element_type;
            uint32_t address_space;
            bool has_address_space;
        } pointer;
        struct {
            MLIR_TypeHandle *inputs;
            size_t n_inputs;
            MLIR_TypeHandle *results;
            size_t n_results;
        } function;
    } data;
} IR_Type;

typedef struct IR_Attribute {
    enum {
        ATTR_KIND_INTEGER,
        ATTR_KIND_FLOAT,
        ATTR_KIND_STRING,
        ATTR_KIND_BOOL,
        ATTR_KIND_ARRAY,
        ATTR_KIND_DICT,
        ATTR_KIND_TYPE
    } kind;
    union {
        int64_t integer_value;
        double float_value;
        string string_value;
        bool bool_value;
        struct {
            MLIR_AttributeHandle *elements;
            size_t count;
        } array;
        MLIR_TypeHandle type_value;
    } data;
    string name;
    // MLIR type of the attribute's value. Required for ATTR_KIND_INTEGER
    // and ATTR_KIND_FLOAT (and will apply to ATTR_KIND_DENSE_ELEMENTS
    // when added). For other kinds it is unused and must be
    // MLIR_INVALID_HANDLE (the zero value, set by `IR_Attribute a = {0}`).
    MLIR_TypeHandle type;
} IR_Attribute;

typedef struct IR_Location {
    LocationKind kind;
    union {
        struct { string filename; int line; int column; } file;
        struct { string name; } name;
        struct { int ref_id; } ref;
    } data;
    string original_text;
} IR_Location;

typedef struct IR_Value {
    MLIR_ValueKind kind;
    uintptr_t def_handle;  // MLIR_OpHandle if OP_RESULT, MLIR_BlockHandle if BLOCK_ARG
    uint32_t result_index;
    MLIR_TypeHandle type;
    string register_name;
    MLIR_LocationHandle location;
} IR_Value;

typedef struct IR_Op {
    MLIR_OpType op_type;
    MLIR_ValueHandle *operands;
    uint64_t n_operands;
    MLIR_TypeHandle *result_types;
    uint64_t n_result_types;
    MLIR_AttributeHandle *attributes;
    uint64_t n_attributes;
    MLIR_RegionHandle *regions;
    uint64_t n_regions;
    MLIR_BlockHandle *successors;
    uint64_t n_successors;
    MLIR_ValueHandle **successor_operands;
    uint64_t *n_successor_operands;
    string opname;
    MLIR_ValueHandle *results;
    uint64_t n_results;
    MLIR_LocationHandle location;
    MLIR_LocationHandle unnumbered_loc_def;
    string trailing_comment;
    int64_t source_line_start;
} IR_Op;

typedef struct IR_Block {
    MLIR_OpHandle *operations;
    uint64_t n_operations;
    MLIR_ValueHandle *arguments;
    uint64_t n_arguments;
    MLIR_RegionHandle parent_region;
} IR_Block;

typedef struct IR_Region {
    MLIR_BlockHandle *blocks;
    uint64_t n_blocks;
} IR_Region;

static inline IR_Op *resolve_op(MLIR_OpHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Op *)(uintptr_t)h;
}

static inline IR_Region *resolve_region(MLIR_RegionHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Region *)(uintptr_t)h;
}

static inline IR_Block *resolve_block(MLIR_BlockHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Block *)(uintptr_t)h;
}

static inline IR_Value *resolve_value(MLIR_ValueHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Value *)(uintptr_t)h;
}

static inline IR_Type *resolve_type(MLIR_TypeHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Type *)(uintptr_t)h;
}

static inline IR_Attribute *resolve_attr(MLIR_AttributeHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Attribute *)(uintptr_t)h;
}

static inline IR_Location *resolve_loc(MLIR_LocationHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Location *)(uintptr_t)h;
}

static inline MLIR_OpHandle alloc_op(MLIR_Context *ctx, IR_Op op) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Op *slot = arena_new(ctx->arena, IR_Op);
    *slot = op;
    return (MLIR_OpHandle)(uintptr_t)slot;
}

static inline MLIR_RegionHandle alloc_region(MLIR_Context *ctx, IR_Region r) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Region *slot = arena_new(ctx->arena, IR_Region);
    *slot = r;
    return (MLIR_RegionHandle)(uintptr_t)slot;
}

static inline MLIR_BlockHandle alloc_block(MLIR_Context *ctx, IR_Block b) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Block *slot = arena_new(ctx->arena, IR_Block);
    *slot = b;
    return (MLIR_BlockHandle)(uintptr_t)slot;
}

static inline MLIR_ValueHandle alloc_value(MLIR_Context *ctx, IR_Value v) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Value *slot = arena_new(ctx->arena, IR_Value);
    *slot = v;
    return (MLIR_ValueHandle)(uintptr_t)slot;
}

static inline MLIR_TypeHandle alloc_type(MLIR_Context *ctx, IR_Type t) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Type *slot = arena_new(ctx->arena, IR_Type);
    *slot = t;
    return (MLIR_TypeHandle)(uintptr_t)slot;
}

static inline MLIR_AttributeHandle alloc_attr_obj(MLIR_Context *ctx, IR_Attribute a) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Attribute *slot = arena_new(ctx->arena, IR_Attribute);
    *slot = a;
    return (MLIR_AttributeHandle)(uintptr_t)slot;
}

static inline MLIR_LocationHandle alloc_loc(MLIR_Context *ctx, IR_Location l) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Location *slot = arena_new(ctx->arena, IR_Location);
    *slot = l;
    return (MLIR_LocationHandle)(uintptr_t)slot;
}

// API lifecycle
void MLIR_InitApi(MLIR_Context *ctx, MLIR_OpHandle root) {
    (void)ctx;
    (void)root;
}

void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena) {
    if (!ctx) return;
    ctx->arena = arena;
}

Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx) {
    return ctx ? ctx->arena : NULL;
}

// Operation creation
MLIR_OpHandle MLIR_CreateOpWithSuccessors(
    MLIR_Context *ctx,
    MLIR_OpType type,
    string opname,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_BlockHandle *successors, size_t n_successors,
    MLIR_ValueHandle **successor_operands,
    size_t *n_successor_operands,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {

    IR_Op op = {0};
    op.op_type = type;
    op.opname = opname;
    op.attributes = attributes;
    op.n_attributes = n_attributes;
    op.result_types = result_types;
    op.n_result_types = n_result_types;
    op.results = results;
    op.n_results = n_results;
    op.operands = operands;
    op.n_operands = n_operands;
    op.regions = regions;
    op.n_regions = n_regions;
    op.successors = successors;
    op.n_successors = n_successors;
    op.successor_operands = successor_operands;
    // The native IR_Op stores n_successor_operands as uint64_t* for layout
    // stability, but the public API uses size_t* (caller-owned arena array).
    // On 64-bit targets these are identical; we simply alias the pointer.
    op.n_successor_operands = (uint64_t *)n_successor_operands;
    op.location = location;
    op.unnumbered_loc_def = unnumbered_loc_def;
    op.trailing_comment = trailing_comment;
    op.source_line_start = source_line_start;

    MLIR_OpHandle handle = alloc_op(ctx, op);

    // Set def_handle on result values
    for (size_t i = 0; i < n_results; i++) {
        IR_Value *v = resolve_value(results[i]);
        if (v) {
            v->def_handle = handle;
            if (i < n_result_types && result_types[i] != MLIR_INVALID_HANDLE && v->type == MLIR_INVALID_HANDLE) {
                v->type = result_types[i];
            }
        }
    }

    return handle;
}

MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *ctx,
    MLIR_OpType type,
    string opname,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {
    return MLIR_CreateOpWithSuccessors(
        ctx, type, opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands, n_operands,
        regions, n_regions,
        NULL, 0, NULL, NULL,
        location, unnumbered_loc_def, trailing_comment, source_line_start);
}

void MLIR_AppendBlockOp(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_OpHandle op) {
    IR_Block *block = resolve_block(bh);
    if (!block || !ctx || !ctx->arena) return;
    Arena *arena = ctx->arena;
    MLIR_OpHandle *new_ops = arena_new_array(arena, MLIR_OpHandle, block->n_operations + 1);
    if (block->operations) memcpy(new_ops, block->operations, block->n_operations * sizeof(MLIR_OpHandle));
    new_ops[block->n_operations] = op;
    block->operations = new_ops;
    block->n_operations++;
}

void MLIR_AppendBlockArg(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_ValueHandle arg) {
    IR_Block *block = resolve_block(bh);
    if (!block || !ctx || !ctx->arena) return;
    Arena *arena = ctx->arena;
    MLIR_ValueHandle *new_args = arena_new_array(arena, MLIR_ValueHandle, block->n_arguments + 1);
    if (block->arguments) memcpy(new_args, block->arguments, block->n_arguments * sizeof(MLIR_ValueHandle));
    new_args[block->n_arguments] = arg;
    block->arguments = new_args;
    block->n_arguments++;

    // Set def_handle to block for block arguments
    IR_Value *v = resolve_value(arg);
    if (v) v->def_handle = bh;
}

void MLIR_AppendRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle rh, MLIR_BlockHandle block) {
    IR_Region *region = resolve_region(rh);
    if (!region || !ctx || !ctx->arena) return;
    Arena *arena = ctx->arena;
    MLIR_BlockHandle *new_blocks = arena_new_array(arena, MLIR_BlockHandle, region->n_blocks + 1);
    if (region->blocks) memcpy(new_blocks, region->blocks, region->n_blocks * sizeof(MLIR_BlockHandle));
    new_blocks[region->n_blocks] = block;
    region->blocks = new_blocks;
    region->n_blocks++;
    IR_Block *b = resolve_block(block);
    if (b) b->parent_region = rh;
}

size_t MLIR_GetRegionNumBlocks(MLIR_RegionHandle rh) {
    IR_Region *r = resolve_region(rh);
    return r ? r->n_blocks : 0;
}

MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_RegionHandle rh, size_t idx) {
    IR_Region *r = resolve_region(rh);
    if (!r || idx >= r->n_blocks) return MLIR_INVALID_HANDLE;
    return r->blocks[idx];
}

size_t MLIR_GetBlockNumOps(MLIR_BlockHandle bh) {
    IR_Block *b = resolve_block(bh);
    return b ? b->n_operations : 0;
}

MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle bh, size_t idx) {
    IR_Block *b = resolve_block(bh);
    if (!b || idx >= b->n_operations) return MLIR_INVALID_HANDLE;
    return b->operations[idx];
}

MLIR_OpType MLIR_GetOpType(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->op_type : OP_TYPE_UNREGISTERED;
}

size_t MLIR_GetOpNumRegions(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_regions : 0;
}

MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_regions) return MLIR_INVALID_HANDLE;
    return op->regions[idx];
}

string MLIR_GetOpTrailingComment(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->trailing_comment : str_lit("");
}

size_t MLIR_GetOpNumAttributes(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_attributes : 0;
}

MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_attributes) return MLIR_INVALID_HANDLE;
    return op->attributes[idx];
}

size_t MLIR_GetOpNumOperands(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_operands : 0;
}

MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_operands) return MLIR_INVALID_HANDLE;
    return op->operands[idx];
}

size_t MLIR_GetOpNumResults(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_results : 0;
}

MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_results) return MLIR_INVALID_HANDLE;
    return op->results[idx];
}

MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->location : MLIR_INVALID_HANDLE;
}

string MLIR_GetOpName(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->opname : str_lit("");
}

string MLIR_GetOpName_string(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->opname : str_lit("");
}

size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_result_types : 0;
}

MLIR_TypeHandle MLIR_GetOpResult_type(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_result_types) return MLIR_INVALID_HANDLE;
    return op->result_types[idx];
}

// Native backend doesn't track successors/successor-operands separately;
// the classic parser stores _target/_ntrue/_nfalse + interleaves operands.
// Return 0/invalid so the printer falls back to its attribute-based path.
size_t MLIR_GetOpNumSuccessors(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->n_successors : 0;
}
MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_OpHandle oh, size_t idx) {
    IR_Op *op = resolve_op(oh);
    if (!op || idx >= op->n_successors) return MLIR_INVALID_HANDLE;
    return op->successors[idx];
}
size_t MLIR_GetOpNumSuccessorOperands(MLIR_OpHandle oh, size_t s) {
    IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors || !op->n_successor_operands) return 0;
    return (size_t)op->n_successor_operands[s];
}
MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_OpHandle oh, size_t s, size_t i) {
    IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors || !op->successor_operands) return MLIR_INVALID_HANDLE;
    if (i >= (size_t)op->n_successor_operands[s]) return MLIR_INVALID_HANDLE;
    return op->successor_operands[s][i];
}

size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle bh) {
    IR_Block *b = resolve_block(bh);
    return b ? b->n_arguments : 0;
}

MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle bh, size_t idx) {
    IR_Block *b = resolve_block(bh);
    if (!b || idx >= b->n_arguments) return MLIR_INVALID_HANDLE;
    return b->arguments[idx];
}

size_t MLIR_GetBlockIndex(MLIR_BlockHandle bh) {
    IR_Block *b = resolve_block(bh);
    if (!b) return (size_t)-1;
    IR_Region *r = resolve_region(b->parent_region);
    if (!r) return (size_t)-1;
    for (size_t i = 0; i < r->n_blocks; i++) {
        if (r->blocks[i] == bh) return i;
    }
    return (size_t)-1;
}

MLIR_LocationHandle MLIR_GetValueLocation(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->location : MLIR_INVALID_HANDLE;
}

// Type to string
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op) {
    (void)ctx; (void)op;
    return str_lit("error: MLIR_PrintOperationUpstream requires the upstream backend\n");
}

MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text) {
    (void)ctx; (void)text;
    return MLIR_INVALID_HANDLE;
}

string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle th) {
    IR_Type *type = resolve_type(th);
    if (!type) return str_lit("null");
    if (!ctx || !ctx->arena) return str_lit("null");
    Arena *arena = ctx->arena;

    switch (type->kind) {
        case TYPE_KIND_UNKNOWN:
            return str_lit("?");
        case TYPE_KIND_OPAQUE:
            return str_lit("unknown");
        case TYPE_KIND_INTEGER:
            if (type->data.integer.is_signed) {
                return format(arena, str_lit("i{}"), (int64_t)type->data.integer.width);
            } else {
                return format(arena, str_lit("ui{}"), (int64_t)type->data.integer.width);
            }
        case TYPE_KIND_FLOAT:
            if (type->data.floating.is_bfloat && type->data.floating.width == 16) {
                return str_lit("bf16");
            }
            return format(arena, str_lit("f{}"), (int64_t)type->data.floating.width);
        case TYPE_KIND_TENSOR: {
            MLIR_TypeHandle elem_h = type->data.shaped.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    string shape_str = str_lit("");
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            shape_str = str_concat(arena, shape_str, str_lit("?x"));
                        } else {
                            shape_str = str_concat(arena, shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("tensor<{}{}>"), shape_str, elem_str);
                } else {
                    return format(arena, str_lit("tensor<{}>"), elem_str);
                }
            }
            return str_lit("tensor<?>");
        }
        case TYPE_KIND_MEMREF: {
            MLIR_TypeHandle elem_h = type->data.shaped.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    string shape_str = str_lit("");
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            shape_str = str_concat(arena, shape_str, str_lit("?x"));
                        } else {
                            shape_str = str_concat(arena, shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("memref<{}{}>"), shape_str, elem_str);
                } else {
                    return format(arena, str_lit("memref<{}>"), elem_str);
                }
            }
            return str_lit("memref<?>");
        }
        case TYPE_KIND_POINTER: {
            MLIR_TypeHandle elem_h = type->data.pointer.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.pointer.has_address_space) {
                    return format(arena, str_lit("!tt.ptr<{}, {}>"), elem_str, (int64_t)type->data.pointer.address_space);
                } else {
                    return format(arena, str_lit("!tt.ptr<{}>"), elem_str);
                }
            }
            return str_lit("!tt.ptr<?>");
        }
        case TYPE_KIND_INDEX:
            return str_lit("index");
        case TYPE_KIND_FUNCTION: {
            string in_str = str_lit("");
            for (size_t i = 0; i < type->data.function.n_inputs; i++) {
                if (i > 0) in_str = str_concat(arena, in_str, str_lit(", "));
                in_str = str_concat(arena, in_str,
                                    MLIR_GetTypeString(ctx, type->data.function.inputs[i]));
            }
            string out_str = str_lit("");
            size_t nr = type->data.function.n_results;
            if (nr == 1) {
                out_str = MLIR_GetTypeString(ctx, type->data.function.results[0]);
            } else {
                string body = str_lit("");
                for (size_t i = 0; i < nr; i++) {
                    if (i > 0) body = str_concat(arena, body, str_lit(", "));
                    body = str_concat(arena, body,
                                      MLIR_GetTypeString(ctx, type->data.function.results[i]));
                }
                out_str = format(arena, str_lit("({})"), body);
            }
            return format(arena, str_lit("({}) -> {}"), in_str, out_str);
        }
        default:
            return str_lit("unknown");
    }
}

// Type creation
MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *ctx, uint32_t width, bool is_signed) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_INTEGER;
    t.data.integer.width = width;
    t.data.integer.is_signed = is_signed;
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *ctx, uint32_t width, bool is_bfloat) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_FLOAT;
    t.data.floating.width = width;
    t.data.floating.is_bfloat = is_bfloat;
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_INDEX;
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_UNKNOWN;
    return alloc_type(ctx, t);
}

static void copy_shape_to_arena(MLIR_Context *ctx, IR_Type *type, const int64_t *shape, size_t rank) {
    type->data.shaped.rank = (uint32_t)rank;
    if (rank > 0 && shape) {
        if (!ctx || !ctx->arena) {
            type->data.shaped.shape = NULL;
            type->data.shaped.rank = 0;
            return;
        }
        type->data.shaped.shape = arena_new_array(ctx->arena, int64_t, rank);
        for (size_t i = 0; i < rank; i++) {
            type->data.shaped.shape[i] = shape[i];
        }
    } else {
        type->data.shaped.shape = NULL;
    }
}

MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_TENSOR;
    t.data.shaped.element_type = element_type;
    copy_shape_to_arena(ctx, &t, shape, rank);
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_MEMREF;
    t.data.shaped.element_type = element_type;
    copy_shape_to_arena(ctx, &t, shape, rank);
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_POINTER;
    t.data.pointer.element_type = element_type;
    t.data.pointer.has_address_space = has_address_space;
    t.data.pointer.address_space = address_space;
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx) {
    // Native classic/generic backend doesn't currently model LLVM dialect
    // types; tinyC and other LLVM-dialect users go through the upstream
    // backend. Return an opaque marker so unrelated tests still link.
    IR_Type t = {0};
    t.kind = TYPE_KIND_OPAQUE;
    (void)ctx;
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *ctx, string name) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_OPAQUE;
    (void)name;
    return alloc_type(ctx, t);
}

void MLIR_SetTypeLLVMStructBody(MLIR_Context *ctx, MLIR_TypeHandle struct_ty,
                                 const MLIR_TypeHandle *fields, size_t n_fields) {
    (void)ctx; (void)struct_ty; (void)fields; (void)n_fields;
}

MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *ctx, MLIR_TypeHandle elem, uint64_t count) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_OPAQUE;
    (void)elem; (void)count;
    return alloc_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *ctx,
                                         const MLIR_TypeHandle *inputs, size_t n_inputs,
                                         const MLIR_TypeHandle *results, size_t n_results) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_FUNCTION;
    if (ctx && ctx->arena) {
        if (n_inputs > 0 && inputs) {
            t.data.function.inputs = arena_new_array(ctx->arena, MLIR_TypeHandle, n_inputs);
            for (size_t i = 0; i < n_inputs; i++) t.data.function.inputs[i] = inputs[i];
            t.data.function.n_inputs = n_inputs;
        }
        if (n_results > 0 && results) {
            t.data.function.results = arena_new_array(ctx->arena, MLIR_TypeHandle, n_results);
            for (size_t i = 0; i < n_results; i++) t.data.function.results[i] = results[i];
            t.data.function.n_results = n_results;
        }
    }
    return alloc_type(ctx, t);
}

bool MLIR_IsTypeFunction(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_FUNCTION;
}

size_t MLIR_GetTypeFunctionNumInputs(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_FUNCTION) return 0;
    return t->data.function.n_inputs;
}

MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle th, size_t idx) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_FUNCTION) return MLIR_INVALID_HANDLE;
    if (idx >= t->data.function.n_inputs || !t->data.function.inputs) return MLIR_INVALID_HANDLE;
    return t->data.function.inputs[idx];
}

size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_FUNCTION) return 0;
    return t->data.function.n_results;
}

MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle th, size_t idx) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_FUNCTION) return MLIR_INVALID_HANDLE;
    if (idx >= t->data.function.n_results || !t->data.function.results) return MLIR_INVALID_HANDLE;
    return t->data.function.results[idx];
}

MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_TENSOR || t->kind == TYPE_KIND_MEMREF) {
        return t->data.shaped.element_type;
    }
    return MLIR_INVALID_HANDLE;
}

void MLIR_SetTypeIntegerProperties(MLIR_TypeHandle th, uint32_t width, bool is_signed) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_INTEGER;
    t->data.integer.width = width;
    t->data.integer.is_signed = is_signed;
}

void MLIR_SetTypeFloatProperties(MLIR_TypeHandle th, uint32_t width, bool is_bfloat) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_FLOAT;
    t->data.floating.width = width;
    t->data.floating.is_bfloat = is_bfloat;
}

// Attribute creation
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, string name, int64_t value, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_INTEGER;
    a.name = name;
    a.data.integer_value = value;
    a.type = type;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string name, string value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    a.data.string_value = value;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, string name, double value, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_FLOAT;
    a.name = name;
    a.data.float_value = value;
    a.type = type;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, string name, bool value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_BOOL;
    a.name = name;
    a.data.bool_value = value;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_ARRAY;
    a.name = name;
    a.data.array.elements = elements;
    a.data.array.count = count;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_DICT;
    a.name = name;
    a.data.array.elements = elements;
    a.data.array.count = count;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *ctx, string name, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_TYPE;
    a.name = name;
    a.data.type_value = type;
    return alloc_attr_obj(ctx, a);
}

// Native backend stores SymbolRef as a plain string attribute with the
// symbol name (without leading `@`). This is sufficient for printing —
// the classic printer prepends `@` when emitting `callee = @sym_name`
// for func.call. Verifier-style validation is not performed natively.
MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *ctx, string name, string value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    a.data.string_value = value;
    return alloc_attr_obj(ctx, a);
}

// Native classic/generic backend doesn't model DenseI32ArrayAttr; stash
// nothing useful — only the upstream backend currently consumes this.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *ctx, string name,
                                                       const int32_t *values, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = name;
    (void)values; (void)count;
    return alloc_attr_obj(ctx, a);
}

// Value creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location) {
    IR_Value v = {0};
    v.kind = BLOCK_ARG;
    v.register_name = register_name;
    v.result_index = result_index;
    v.type = type;
    v.location = location;
    v.def_handle = MLIR_INVALID_HANDLE;
    return alloc_value(ctx, v);
}

MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *ctx, MLIR_OpHandle def, uint32_t result_index, MLIR_TypeHandle type, string register_name, MLIR_LocationHandle location) {
    IR_Value v = {0};
    v.kind = OP_RESULT;
    v.def_handle = def;
    v.result_index = result_index;
    v.type = type;
    v.register_name = register_name;
    v.location = location;
    return alloc_value(ctx, v);
}

// Block and Region creation
MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *ctx) {
    IR_Block b = {0};
    return alloc_block(ctx, b);
}

MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *ctx) {
    IR_Region r = {0};
    return alloc_region(ctx, r);
}

void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_AttributeHandle attr) {
    IR_Op *op = resolve_op(oh);
    if (!op || !ctx || !ctx->arena) return;
    size_t new_count = op->n_attributes + 1;
    Arena *arena = ctx->arena;
    MLIR_AttributeHandle *new_attrs = arena_new_array(arena, MLIR_AttributeHandle, new_count);
    if (op->attributes) {
        memcpy(new_attrs, op->attributes, op->n_attributes * sizeof(MLIR_AttributeHandle));
    }
    new_attrs[new_count - 1] = attr;
    op->attributes = new_attrs;
    op->n_attributes = new_count;
}

int64_t MLIR_GetOpSourceLineStart(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->source_line_start : -1;
}

MLIR_LocationHandle MLIR_GetOpUnnumberedLocationDef(MLIR_OpHandle oh) {
    IR_Op *op = resolve_op(oh);
    return op ? op->unnumbered_loc_def : MLIR_INVALID_HANDLE;
}

// Attribute accessors
MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return MLIR_ATTR_KIND_INTEGER;
    switch (attr->kind) {
        case ATTR_KIND_INTEGER: return MLIR_ATTR_KIND_INTEGER;
        case ATTR_KIND_FLOAT:   return MLIR_ATTR_KIND_FLOAT;
        case ATTR_KIND_STRING:  return MLIR_ATTR_KIND_STRING;
        case ATTR_KIND_BOOL:    return MLIR_ATTR_KIND_BOOL;
        case ATTR_KIND_ARRAY:   return MLIR_ATTR_KIND_ARRAY;
        case ATTR_KIND_DICT:    return MLIR_ATTR_KIND_DICT;
        case ATTR_KIND_TYPE:    return MLIR_ATTR_KIND_TYPE;
        default:                return MLIR_ATTR_KIND_DICT;
    }
}

string MLIR_GetAttributeName(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->name : str_lit("");
}

// Native parser never produces "other" attrs; this is just for API parity.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    (void)ctx; (void)ah;
    return str_lit("");
}

int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.integer_value : 0;
}

string MLIR_GetAttributeString(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.string_value : str_lit("");
}

double MLIR_GetAttributeFloat(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.float_value : 0.0;
}

MLIR_TypeHandle MLIR_GetAttributeType(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return MLIR_INVALID_HANDLE;
    if (attr->kind != ATTR_KIND_INTEGER && attr->kind != ATTR_KIND_FLOAT)
        return MLIR_INVALID_HANDLE;
    return attr->type;
}

bool MLIR_GetAttributeBool(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.bool_value : false;
}

size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.array.count : 0;
}

MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle ah, size_t idx) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || idx >= attr->data.array.count) return MLIR_INVALID_HANDLE;
    return attr->data.array.elements[idx];
}

size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.array.count : 0;
}

MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle ah, size_t idx) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || idx >= attr->data.array.count) return MLIR_INVALID_HANDLE;
    return attr->data.array.elements[idx];
}

MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_AttributeHandle ah) {
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || attr->kind != ATTR_KIND_TYPE) return MLIR_INVALID_HANDLE;
    return attr->data.type_value;
}

// Value accessors
MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->kind : BLOCK_ARG;
}

MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->type : MLIR_INVALID_HANDLE;
}

string MLIR_GetValueRegisterName(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->register_name : str_lit("");
}

uint32_t MLIR_GetValueResultIndex(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    return v ? v->result_index : 0;
}

MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_ValueHandle vh) {
    IR_Value *v = resolve_value(vh);
    if (!v || v->kind != OP_RESULT) return MLIR_INVALID_HANDLE;
    return (MLIR_OpHandle)v->def_handle;
}

string MLIR_MLIR_OpTypeToString(MLIR_OpType type) {
    return op_type_to_string(type);
}

// Location accessors
MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    if (!loc) return MLIR_LOC_UNKNOWN;
    switch (loc->kind) {
        case LOC_KIND_FILE: return MLIR_LOC_FILE;
        case LOC_KIND_NAME: return MLIR_LOC_NAME;
        case LOC_KIND_CALLSITE: return MLIR_LOC_CALLSITE;
        case LOC_KIND_FUSED: return MLIR_LOC_FUSED;
        case LOC_KIND_REF: return MLIR_LOC_REF;
        case LOC_KIND_UNKNOWN: default: return MLIR_LOC_UNKNOWN;
    }
}

string MLIR_GetLocationOriginalText(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->original_text : str_lit("");
}

string MLIR_GetLocationFileFilename(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.filename : str_lit("");
}

int MLIR_GetLocationFileLine(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.line : 0;
}

int MLIR_GetLocationFileColumn(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.column : 0;
}

string MLIR_GetLocationName(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.name.name : str_lit("");
}

int MLIR_GetLocationRefId(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.ref.ref_id : 0;
}

MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *ctx, string original_text) {
    IR_Location l = {0};
    l.kind = LOC_KIND_UNKNOWN;
    l.original_text = original_text;
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *ctx, string filename, int line, int column) {
    IR_Location l = {0};
    l.kind = LOC_KIND_FILE;
    l.data.file.filename = filename;
    l.data.file.line = line;
    l.data.file.column = column;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc({}:{}:{})"), filename, (int64_t)line, (int64_t)column);
    }
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *ctx, string name) {
    IR_Location l = {0};
    l.kind = LOC_KIND_NAME;
    l.data.name.name = name;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc(\"{}\")"), name);
    }
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *ctx, int ref_id) {
    IR_Location l = {0};
    l.kind = LOC_KIND_REF;
    l.data.ref.ref_id = ref_id;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc(#loc{})"), (int64_t)ref_id);
    }
    return alloc_loc(ctx, l);
}

MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name) {
    (void)name;
    IR_Type t = {0};
    t.kind = TYPE_KIND_OPAQUE;
    return alloc_type(ctx, t);
}

void MLIR_SetTypeTensorProperties(MLIR_TypeHandle th, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_TENSOR;
    t->data.shaped.element_type = element_type;
    t->data.shaped.shape = (int64_t*)shape;
    t->data.shaped.rank = (uint32_t)rank;
}

void MLIR_SetTypeMemrefProperties(MLIR_TypeHandle th, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_MEMREF;
    t->data.shaped.element_type = element_type;
    t->data.shaped.shape = (int64_t*)shape;
    t->data.shaped.rank = (uint32_t)rank;
}

void MLIR_SetTypePointerProperties(MLIR_TypeHandle th, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space) {
    IR_Type *t = resolve_type(th);
    if (!t) return;
    t->kind = TYPE_KIND_POINTER;
    t->data.pointer.element_type = element_type;
    t->data.pointer.has_address_space = has_address_space;
    t->data.pointer.address_space = address_space;
}

// Type introspection
bool MLIR_IsTypeInteger(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_INTEGER;
}

bool MLIR_IsTypeFloat(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_FLOAT;
}

bool MLIR_IsTypeTensor(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_TENSOR;
}

bool MLIR_IsTypeMemref(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_MEMREF;
}

bool MLIR_IsTypePointer(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_POINTER;
}

bool MLIR_IsTypeIndex(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_INDEX;
}

bool MLIR_IsTypeUnknown(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_UNKNOWN;
}

bool MLIR_IsTypeOpaque(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_OPAQUE;
}

// Lowering / LLVM IR translation are upstream-only. Stub them on the
// native backend so binaries that link only the native impl still resolve.
bool MLIR_LowerToLLVMDialect(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx; (void)module;
    return false;
}

string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx; (void)module;
    return str_lit("");
}

#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <base/arena.h>
#include <base/string.h>
#include <base/format.h>
#include <base/hashtable.h>
#include <base/vector.h>
#include "tokenizer.h"
#include "mlir_api.h"
#include "mlir_parser.h"
#include <string.h>

/*
TODO:
* Currently we have a vector of pointers
* Instead, have a vector of indices, store all operations in one vector
* An idea: one can inline the structs at the end of
the parent struct, but that will make it hard to add more later, and each struct will have a different size, so not possible to uniformly store them in a vector by value
* Best is to have one vector for all operations
  and then use either indices or pointers.
* An issue with using pointers is that you can't double and copy the vector allocation. Thus using indices is probably the best way to do that.
* One can use a pool allocator to reuse space when operations are removed. One can move things around.
* We still need an arena to allocate a vector of indices for each Operation/Region/Block
* Indices also allow easy serializing/deserializing
*/

#ifdef __cplusplus
extern "C" {
#endif

// Internal concrete IR definitions (inlined from the former mlir_ir_internal.h)
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

struct MlirType {
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
            struct MlirType *element_type;
            int64_t *shape;
            uint32_t rank;
        } shaped;
        struct {
            struct MlirType *element_type;
            uint32_t address_space;
            bool has_address_space;
        } pointer;
    } data;
};

struct MlirAttribute {
    enum {
        ATTR_KIND_INTEGER,
        ATTR_KIND_FLOAT,
        ATTR_KIND_STRING,
        ATTR_KIND_BOOL,
        ATTR_KIND_ARRAY,
        ATTR_KIND_DICT
    } kind;
    union {
        int64_t integer_value;
        double float_value;
        string string_value;
        bool bool_value;
        struct {
            struct MlirAttribute **elements;
            size_t count;
        } array;
    } data;
    string name;
};

struct MlirLocation {
    LocationKind kind;
    union {
        struct { string filename; int line; int column; } file;
        struct { string name; } name;
        struct { int ref_id; } ref;
    } data;
    string original_text;
};

struct MlirValue {
    ValueKind kind;
    void *def;
    uint32_t result_index;
    struct MlirType *type;
    string register_name;
    struct MlirLocation *location;
    bool has_divisibility;
    int64_t divisibility_value;
    struct MlirType *divisibility_type;
    bool has_max_divisibility;
    int64_t max_divisibility_value;
    struct MlirType *max_divisibility_type;
};

struct MlirOperation {
    OpType op_type;
    struct MlirValue **operands;
    uint64_t n_operands;
    struct MlirType **result_types;
    uint64_t n_result_types;
    struct MlirAttribute **attributes;
    uint64_t n_attributes;
    struct MlirRegion **regions;
    uint64_t n_regions;
    string opname;
    struct MlirValue **results;
    uint64_t n_results;
    struct MlirLocation *location;
    struct MlirLocation *unnumbered_loc_def;
    string trailing_comment;
    int64_t source_line_start;
};

struct MlirBlock {
    struct MlirOperation **operations;
    uint64_t n_operations;
    struct MlirValue **arguments;
    uint64_t n_arguments;
};

struct MlirRegion {
    struct MlirBlock **blocks;
    uint64_t n_blocks;
};

// string_hash and string_equal are provided inline in mlir_parser.h

// Use SymbolTable, ScopedSymbolTable, and LocationMap from mlir_parser.h

typedef Parser Parser; // use Parser from mlir_parser.h

// Forward declarations for parser entry points implemented in mlir_parser.c
void parser_init(Arena *arena, Parser *parser, string text);
struct MlirOperation; // forward decl
struct MlirOperation* parse_module(Parser *parser);

// Use Operation/Block/Region/Type/Attribute/Location/ValueRef and vectors from mlir_parser.h


// API wrapper functions that cast between API types and concrete types
void mlir_api_init(MlirOperation *root) {
    // Cast API type to concrete type
    (void)root;
    // No initialization required for the native C implementation.
}

MlirOperation *mlir_op_create(
    Arena *arena,
    OpType type,
    string opname,
    MlirAttribute **attributes, size_t n_attributes,
    MlirType **result_types, size_t n_result_types,
    MlirValue **results, size_t n_results,
    MlirValue **operands, size_t n_operands,
    MlirRegion **regions, size_t n_regions,
    MlirLocation *location,
    MlirLocation *unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {
    MlirOperation *op = arena_alloc(arena, MlirOperation);
    *op = (MlirOperation){0};
    op->op_type = type;
    op->opname = opname;
    op->attributes = attributes;
    op->n_attributes = n_attributes;
    op->result_types = result_types;
    op->n_result_types = n_result_types;
    op->results = results;
    op->n_results = n_results;
    op->operands = operands;
    op->n_operands = n_operands;
    op->regions = regions;
    op->n_regions = n_regions;
    op->location = location;
    op->unnumbered_loc_def = unnumbered_loc_def;
    op->trailing_comment = trailing_comment;
    op->source_line_start = source_line_start;
    return op;
}

void mlir_op_add_region(Arena *arena, MlirOperation *op, MlirRegion *region) {
    MlirRegion **new_regions = arena_alloc_array(arena, MlirRegion*, op->n_regions + 1);
    if (op->regions) memcpy(new_regions, op->regions, op->n_regions * sizeof(MlirRegion*));
    new_regions[op->n_regions] = region;
    op->regions = new_regions;
    op->n_regions++;
}

void mlir_op_add_operand(Arena *arena, MlirOperation *op, MlirValue *operand) {
    struct MlirValue **new_operands = arena_alloc_array(arena, struct MlirValue*, op->n_operands + 1);
    if (op->operands) memcpy(new_operands, op->operands, op->n_operands * sizeof(struct MlirValue*));
    new_operands[op->n_operands] = operand;
    op->operands = new_operands;
    op->n_operands++;
}

void mlir_op_add_result(Arena *arena, MlirOperation *op, MlirValue *result) {
    struct MlirValue **new_results = arena_alloc_array(arena, struct MlirValue*, op->n_results + 1);
    if (op->results) memcpy(new_results, op->results, op->n_results * sizeof(struct MlirValue*));
    new_results[op->n_results] = result;
    op->results = new_results;
    op->n_results++;
}

void mlir_block_add_operation(Arena *arena, MlirBlock *block, MlirOperation *op) {
    MlirOperation **new_ops = arena_alloc_array(arena, MlirOperation*, block->n_operations + 1);
    if (block->operations) memcpy(new_ops, block->operations, block->n_operations * sizeof(MlirOperation*));
    new_ops[block->n_operations] = op;
    block->operations = new_ops;
    block->n_operations++;
}

void mlir_block_add_argument(Arena *arena, MlirBlock *block, MlirValue *arg) {
    struct MlirValue **new_args = arena_alloc_array(arena, struct MlirValue*, block->n_arguments + 1);
    if (block->arguments) memcpy(new_args, block->arguments, block->n_arguments * sizeof(struct MlirValue*));
    new_args[block->n_arguments] = arg;
    block->arguments = new_args;
    block->n_arguments++;
}

void mlir_region_add_block(Arena *arena, MlirRegion *region, MlirBlock *block) {
    MlirBlock **new_blocks = arena_alloc_array(arena, MlirBlock*, region->n_blocks + 1);
    if (region->blocks) memcpy(new_blocks, region->blocks, region->n_blocks * sizeof(MlirBlock*));
    new_blocks[region->n_blocks] = block;
    region->blocks = new_blocks;
    region->n_blocks++;
}

size_t mlir_region_num_blocks(const MlirRegion *region) {
    return region->n_blocks;
}

MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx) {
    return region->blocks[idx];
}

size_t mlir_block_num_operations(const MlirBlock *block) {
    return block->n_operations;
}

MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx) {
    return block->operations[idx];
}

OpType mlir_operation_get_type(const MlirOperation *op) {
    return op->op_type;
}

size_t mlir_operation_num_regions(const MlirOperation *op) {
    return op->n_regions;
}

MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx) {
    return op->regions[idx];
}

string mlir_operation_get_trailing_comment(const MlirOperation *op) {
    return op->trailing_comment;
}

// Additional accessors for printer support
size_t mlir_operation_num_attributes(const MlirOperation *op) {
    return op->n_attributes;
}

MlirAttribute *mlir_operation_get_attribute(const MlirOperation *op, size_t idx) {
    return op->attributes[idx];
}

size_t mlir_operation_num_operands(const MlirOperation *op) {
    return op->n_operands;
}

MlirValue *mlir_operation_get_operand(const MlirOperation *op, size_t idx) {
    return op->operands[idx];
}

size_t mlir_operation_num_results(const MlirOperation *op) {
    return op->n_results;
}

MlirValue *mlir_operation_get_result(const MlirOperation *op, size_t idx) {
    return op->results[idx];
}

MlirLocation *mlir_operation_get_location(const MlirOperation *op) {
    return op->location;
}

const char *mlir_operation_get_name(const MlirOperation *op) {
    return op->opname.str;
}

string mlir_operation_get_name_string(const MlirOperation *op) {
    return op->opname;
}

size_t mlir_operation_num_result_types(const MlirOperation *op) {
    return op->n_result_types;
}

MlirType *mlir_operation_get_result_type(const MlirOperation *op, size_t idx) {
    return op->result_types[idx];
}

size_t mlir_block_num_arguments(const MlirBlock *block) {
    return block->n_arguments;
}

MlirValue *mlir_block_get_argument(const MlirBlock *block, size_t idx) {
    return block->arguments[idx];
}

// Value metadata accessors
MlirLocation *mlir_value_get_location(const MlirValue *value) {
    return value->location;
}

bool mlir_value_has_divisibility(const MlirValue *value) {
    return value->has_divisibility;
}

int64_t mlir_value_get_divisibility_value(const MlirValue *value) {
    return value->divisibility_value;
}

MlirType *mlir_value_get_divisibility_type(const MlirValue *value) {
    return value->divisibility_type;
}

bool mlir_value_has_max_divisibility(const MlirValue *value) {
    return value->has_max_divisibility;
}

int64_t mlir_value_get_max_divisibility_value(const MlirValue *value) {
    return value->max_divisibility_value;
}

MlirType *mlir_value_get_max_divisibility_type(const MlirValue *value) {
    return value->max_divisibility_type;
}

// Type to string
string mlir_type_to_string(Arena *arena, MlirType *type) {
    if (!type) {
        return str_lit("null");
    }
    

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
        case TYPE_KIND_TENSOR:
            if (type->data.shaped.element_type) {
                string elem_str = mlir_type_to_string(arena, type->data.shaped.element_type);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    // Build shape string like "4x" or "4x2x"
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
        case TYPE_KIND_MEMREF:
            if (type->data.shaped.element_type) {
                string elem_str = mlir_type_to_string(arena, type->data.shaped.element_type);
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
        case TYPE_KIND_POINTER:
            if (type->data.pointer.element_type) {
                string elem_str = mlir_type_to_string(arena, type->data.pointer.element_type);
                // Show address space if it was explicitly present in the input
                if (type->data.pointer.has_address_space) {
                    return format(arena, str_lit("!tt.ptr<{}, {}>"), elem_str, (int64_t)type->data.pointer.address_space);
                } else {
                    return format(arena, str_lit("!tt.ptr<{}>"), elem_str);
                }
            }
            return str_lit("!tt.ptr<?>");
        case TYPE_KIND_INDEX:
            return str_lit("index");
        case TYPE_KIND_FUNCTION:
            return str_lit("function");
        default:
            return str_lit("unknown");
    }
}

// Type creation and manipulation
MlirType *mlir_type_create_integer(Arena *arena, uint32_t width, bool is_signed) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_INTEGER;
    type->data.integer.width = width;
    type->data.integer.is_signed = is_signed;
    return type;
}

MlirType *mlir_type_create_float(Arena *arena, uint32_t width, bool is_bfloat) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_FLOAT;
    type->data.floating.width = width;
    type->data.floating.is_bfloat = is_bfloat;
    return type;
}

MlirType *mlir_type_create_index(Arena *arena) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_INDEX;
    return type;
}

MlirType *mlir_type_create_unknown(Arena *arena) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_UNKNOWN;
    return type;
}

static void copy_shape(Arena *arena, struct MlirType *type, const int64_t *shape, size_t rank) {
    type->data.shaped.rank = (uint32_t)rank;
    if (rank > 0 && shape) {
        type->data.shaped.shape = arena_alloc_array(arena, int64_t, rank);
        for (size_t i = 0; i < rank; i++) {
            type->data.shaped.shape[i] = shape[i];
        }
    } else {
        type->data.shaped.shape = NULL;
    }
}

MlirType *mlir_type_create_tensor(Arena *arena, const int64_t *shape, size_t rank, MlirType *element_type) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_TENSOR;
    type->data.shaped.element_type = element_type;
    copy_shape(arena, type, shape, rank);
    return type;
}

MlirType *mlir_type_create_memref(Arena *arena, const int64_t *shape, size_t rank, MlirType *element_type) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_MEMREF;
    type->data.shaped.element_type = element_type;
    copy_shape(arena, type, shape, rank);
    return type;
}

MlirType *mlir_type_create_pointer(Arena *arena, MlirType *element_type, bool has_address_space, uint32_t address_space) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_POINTER;
    type->data.pointer.element_type = element_type;
    type->data.pointer.has_address_space = has_address_space;
    type->data.pointer.address_space = address_space;
    return type;
}

void mlir_type_set_integer_properties(MlirType *type, uint32_t width, bool is_signed) {
    type->kind = TYPE_KIND_INTEGER;
    type->data.integer.width = width;
    type->data.integer.is_signed = is_signed;
}

void mlir_type_set_float_properties(MlirType *type, uint32_t width, bool is_bfloat) {
    type->kind = TYPE_KIND_FLOAT;
    type->data.floating.width = width;
    type->data.floating.is_bfloat = is_bfloat;
}

// Attribute creation and manipulation
MlirAttribute *mlir_attribute_create_integer(Arena *arena, int64_t value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_INTEGER;
    attr->data.integer_value = value;
    return attr;
}

MlirAttribute *mlir_attribute_create_string(Arena *arena, const char *str, size_t len) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_STRING;
    attr->data.string_value = (string){(char*)str, len};
    return attr;
}

MlirAttribute *mlir_attribute_create_float(Arena *arena, double value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_FLOAT;
    attr->data.float_value = value;
    return attr;
}

MlirAttribute *mlir_attribute_create_bool(Arena *arena, bool value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_BOOL;
    attr->data.bool_value = value;
    return attr;
}

void mlir_attribute_set_name(MlirAttribute *attr, const char *name, size_t name_len) {
    attr->name = (string){(char*)name, name_len};
}

// Value creation and manipulation
MlirValue *mlir_value_create(Arena *arena, ValueKind value_kind) {
    struct MlirValue *value = arena_alloc(arena, struct MlirValue);
    *value = (struct MlirValue){0};
    value->kind = value_kind;
    return value;
}

void mlir_value_set_type(MlirValue *value, MlirType *type) {
    value->type = type;
}

void mlir_value_set_register_name(MlirValue *value, const char *name, size_t name_len) {
    value->register_name = (string){(char*)name, name_len};
}

void mlir_value_set_result_index(MlirValue *value, uint32_t index) {
    value->result_index = index;
}

void mlir_value_set_def(MlirValue *value, void *def) {
    value->def = def;
}

void mlir_value_set_divisibility(MlirValue *value, bool has_value, int64_t div_value, MlirType *type) {
    value->has_divisibility = has_value;
    value->divisibility_value = div_value;
    value->divisibility_type = type;
}

void mlir_value_set_max_divisibility(MlirValue *value, bool has_value, int64_t div_value, MlirType *type) {
    value->has_max_divisibility = has_value;
    value->max_divisibility_value = div_value;
    value->max_divisibility_type = type;
}

// Block and Region creation
MlirBlock *mlir_block_create(Arena *arena) {
    struct MlirBlock *block = arena_alloc(arena, struct MlirBlock);
    *block = (struct MlirBlock){0};
    return block;
}

MlirRegion *mlir_region_create(Arena *arena) {
    struct MlirRegion *region = arena_alloc(arena, struct MlirRegion);
    *region = (struct MlirRegion){0};
    return region;
}

// Operation properties
void mlir_operation_set_name(MlirOperation *op, const char *name, size_t name_len) {
    op->opname = (string){(char*)name, name_len};
}

// Consolidated setter for results and their types
void mlir_operation_set_results_with_types(MlirOperation *op, MlirValue **results, MlirType **result_types, size_t count) {
    op->results = results;
    op->n_results = count;
    op->result_types = result_types;
    op->n_result_types = count;
}

void mlir_operation_set_attributes(MlirOperation *op, MlirAttribute **attrs, size_t count) {
    op->attributes = attrs;
    op->n_attributes = count;
}

// Internal helper for parser - sets just result types during incremental parsing
void set_op_result_types(MlirOperation *op, MlirType **types, size_t count) {
    op->result_types = types;
    op->n_result_types = count;
}

void mlir_operation_set_operands(MlirOperation *op, MlirValue **operands, size_t count) {
    op->operands = operands;
    op->n_operands = count;
}

void mlir_operation_set_location(MlirOperation *op, MlirLocation *loc) {
    op->location = loc;
}

void mlir_operation_set_trailing_comment(MlirOperation *op, const char *comment, size_t comment_len) {
    op->trailing_comment = (string){(char*)comment, comment_len};
}

void mlir_operation_set_source_line_start(MlirOperation *op, int64_t line_start) {
    op->source_line_start = line_start;
}

void mlir_operation_set_unnumbered_loc_def(MlirOperation *op, MlirLocation *loc) {
    op->unnumbered_loc_def = loc;
}

void mlir_operation_append_attribute(Arena *arena, MlirOperation *op, MlirAttribute *attr) {
    size_t new_count = op->n_attributes + 1;
    struct MlirAttribute **new_attrs = arena_alloc_array(arena, struct MlirAttribute*, new_count);
    if (op->attributes) {
        memcpy(new_attrs, op->attributes, op->n_attributes * sizeof(struct MlirAttribute*));
    }
    new_attrs[new_count - 1] = attr;
    op->attributes = new_attrs;
    op->n_attributes = new_count;
}

int64_t mlir_operation_get_source_line_start(const MlirOperation *op) {
    return op->source_line_start;
}

MlirLocation *mlir_operation_get_unnumbered_loc_def(const MlirOperation *op) {
    return op->unnumbered_loc_def;
}

// Attribute accessors
MlirAttrKind mlir_attribute_get_kind(const MlirAttribute *attr) {
    switch (attr->kind) {
        case ATTR_KIND_INTEGER: return MLIR_ATTR_KIND_INTEGER;
        case ATTR_KIND_FLOAT:   return MLIR_ATTR_KIND_FLOAT;
        case ATTR_KIND_STRING:  return MLIR_ATTR_KIND_STRING;
        case ATTR_KIND_BOOL:    return MLIR_ATTR_KIND_BOOL;
        case ATTR_KIND_ARRAY:   return MLIR_ATTR_KIND_ARRAY;
        case ATTR_KIND_DICT:    return MLIR_ATTR_KIND_DICT;
        default:                return MLIR_ATTR_KIND_DICT;
    }
}

string mlir_attribute_get_name(const MlirAttribute *attr) {
    return attr->name;
}

int64_t mlir_attribute_get_integer(const MlirAttribute *attr) {
    return attr->data.integer_value;
}

string mlir_attribute_get_string(const MlirAttribute *attr) {
    return attr->data.string_value;
}

double mlir_attribute_get_float(const MlirAttribute *attr) {
    return attr->data.float_value;
}

bool mlir_attribute_get_bool(const MlirAttribute *attr) {
    return attr->data.bool_value;
}

// Value accessors
ValueKind mlir_value_get_kind(const MlirValue *value) {
    return value->kind;
}

MlirType *mlir_value_get_type(const MlirValue *value) {
    return value->type;
}

string mlir_value_get_register_name(const MlirValue *value) {
    return value->register_name;
}

uint32_t mlir_value_get_result_index(const MlirValue *value) {
    return value->result_index;
}

MlirOperation *mlir_value_get_def_op(const MlirValue *value) {
    return (MlirOperation*)value->def;
}

const char *mlir_op_type_to_string(OpType type) {
    string s = op_type_to_string(type);
    return s.str;
}

// Location accessors
MlirLocationKind mlir_location_get_kind(const MlirLocation *loc) {
    switch (loc->kind) {
        case LOC_KIND_FILE: return MLIR_LOC_FILE;
        case LOC_KIND_NAME: return MLIR_LOC_NAME;
        case LOC_KIND_CALLSITE: return MLIR_LOC_CALLSITE;
        case LOC_KIND_FUSED: return MLIR_LOC_FUSED;
        case LOC_KIND_REF: return MLIR_LOC_REF;
        case LOC_KIND_UNKNOWN: default: return MLIR_LOC_UNKNOWN;
    }
}

string mlir_location_get_original_text(const MlirLocation *loc) {
    return loc->original_text;
}

string mlir_location_get_file_filename(const MlirLocation *loc) {
    return loc->data.file.filename;
}

int mlir_location_get_file_line(const MlirLocation *loc) {
    return loc->data.file.line;
}

int mlir_location_get_file_column(const MlirLocation *loc) {
    return loc->data.file.column;
}

string mlir_location_get_name(const MlirLocation *loc) {
    return loc->data.name.name;
}

int mlir_location_get_ref_id(const MlirLocation *loc) {
    return loc->data.ref.ref_id;
}

MlirLocation *mlir_location_create(Arena *arena) {
    struct MlirLocation *loc = arena_alloc(arena, struct MlirLocation);
    *loc = (struct MlirLocation){0};
    loc->kind = LOC_KIND_UNKNOWN;
    return loc;
}

void mlir_location_set_kind(MlirLocation *loc, MlirLocationKind kind) {
    switch (kind) {
        case MLIR_LOC_FILE: loc->kind = LOC_KIND_FILE; break;
        case MLIR_LOC_NAME: loc->kind = LOC_KIND_NAME; break;
        case MLIR_LOC_CALLSITE: loc->kind = LOC_KIND_CALLSITE; break;
        case MLIR_LOC_FUSED: loc->kind = LOC_KIND_FUSED; break;
        case MLIR_LOC_REF: loc->kind = LOC_KIND_REF; break;
        case MLIR_LOC_UNKNOWN: default: loc->kind = LOC_KIND_UNKNOWN; break;
    }
}

void mlir_location_set_original_text(MlirLocation *loc, string text) {
    loc->original_text = text;
}

void mlir_location_set_file_data(MlirLocation *loc, string filename, int line, int column) {
    loc->kind = LOC_KIND_FILE;
    loc->data.file.filename = filename;
    loc->data.file.line = line;
    loc->data.file.column = column;
}

void mlir_location_set_name_data(MlirLocation *loc, string name) {
    loc->kind = LOC_KIND_NAME;
    loc->data.name.name = name;
}

void mlir_location_set_ref_id(MlirLocation *loc, int ref_id) {
    loc->kind = LOC_KIND_REF;
    loc->data.ref.ref_id = ref_id;
}

void mlir_value_set_location(MlirValue *value, MlirLocation *loc) {
    value->location = loc;
}

// New API functions for parser compatibility

MlirType *mlir_type_create_opaque(Arena *arena, string name) {
    struct MlirType *type = arena_alloc(arena, struct MlirType);
    *type = (struct MlirType){0};
    type->kind = TYPE_KIND_OPAQUE;
    // Store the name in a way that can be retrieved later if needed
    return type;
}

void mlir_type_set_tensor_properties(MlirType *type, const int64_t *shape, size_t rank, MlirType *element_type) {
    type->kind = TYPE_KIND_TENSOR;
    type->data.shaped.element_type = element_type;
    // Note: These functions expect the shape to be pre-allocated, they just set the pointer
    type->data.shaped.shape = (int64_t*)shape;
    type->data.shaped.rank = (uint32_t)rank;
}

void mlir_type_set_memref_properties(MlirType *type, const int64_t *shape, size_t rank, MlirType *element_type) {
    type->kind = TYPE_KIND_MEMREF;
    type->data.shaped.element_type = element_type;
    // Note: These functions expect the shape to be pre-allocated, they just set the pointer
    type->data.shaped.shape = (int64_t*)shape;
    type->data.shaped.rank = (uint32_t)rank;
}

void mlir_type_set_pointer_properties(MlirType *type, MlirType *element_type, bool has_address_space, uint32_t address_space) {
    type->kind = TYPE_KIND_POINTER;
    type->data.pointer.element_type = element_type;
    type->data.pointer.has_address_space = has_address_space;
    type->data.pointer.address_space = address_space;
}

// Type introspection functions
bool mlir_type_is_integer(const MlirType *type) {
    return type->kind == TYPE_KIND_INTEGER;
}

bool mlir_type_is_float(const MlirType *type) {
    return type->kind == TYPE_KIND_FLOAT;
}

bool mlir_type_is_tensor(const MlirType *type) {
    return type->kind == TYPE_KIND_TENSOR;
}

bool mlir_type_is_memref(const MlirType *type) {
    return type->kind == TYPE_KIND_MEMREF;
}

bool mlir_type_is_pointer(const MlirType *type) {
    return type->kind == TYPE_KIND_POINTER;
}

bool mlir_type_is_index(const MlirType *type) {
    return type->kind == TYPE_KIND_INDEX;
}

bool mlir_type_is_unknown(const MlirType *type) {
    return type->kind == TYPE_KIND_UNKNOWN;
}

bool mlir_type_is_opaque(const MlirType *type) {
    return type->kind == TYPE_KIND_OPAQUE;
}

void mlir_operation_set_type(MlirOperation *op, OpType type) {
    op->op_type = type;
}

#ifdef __cplusplus
}
#endif

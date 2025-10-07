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

    // Automatically set the def field on all result values
    // Also sync type from result_types if value has NULL type
    for (size_t i = 0; i < n_results; i++) {
        if (results[i]) {
            results[i]->def = op;
            // Auto-sync type from result_types if not already set
            if (i < n_result_types && result_types[i] && !results[i]->type) {
                results[i]->type = result_types[i];
            }
        }
    }

    return op;
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

    // Automatically set the def field to point to the block for block arguments
    if (arg) {
        arg->def = block;
    }
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

OpType mlir_op_get_type(const MlirOperation *op) {
    return op->op_type;
}

size_t mlir_op_num_regions(const MlirOperation *op) {
    return op->n_regions;
}

MlirRegion *mlir_op_get_region(const MlirOperation *op, size_t idx) {
    return op->regions[idx];
}

string mlir_op_get_trailing_comment(const MlirOperation *op) {
    return op->trailing_comment;
}

// Additional accessors for printer support
size_t mlir_op_num_attributes(const MlirOperation *op) {
    return op->n_attributes;
}

MlirAttribute *mlir_op_get_attribute(const MlirOperation *op, size_t idx) {
    return op->attributes[idx];
}

size_t mlir_op_num_operands(const MlirOperation *op) {
    return op->n_operands;
}

MlirValue *mlir_op_get_operand(const MlirOperation *op, size_t idx) {
    return op->operands[idx];
}

size_t mlir_op_num_results(const MlirOperation *op) {
    return op->n_results;
}

MlirValue *mlir_op_get_result(const MlirOperation *op, size_t idx) {
    return op->results[idx];
}

MlirLocation *mlir_op_get_location(const MlirOperation *op) {
    return op->location;
}

string mlir_op_get_name(const MlirOperation *op) {
    return op->opname;
}

string mlir_op_get_name_string(const MlirOperation *op) {
    return op->opname;
}

size_t mlir_op_num_result_types(const MlirOperation *op) {
    return op->n_result_types;
}

MlirType *mlir_op_get_result_type(const MlirOperation *op, size_t idx) {
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
MlirAttribute *mlir_attribute_create_integer(Arena *arena, string name, int64_t value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_INTEGER;
    attr->name = name;
    attr->data.integer_value = value;
    return attr;
}

MlirAttribute *mlir_attribute_create_string(Arena *arena, string name, string value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_STRING;
    attr->name = name;
    attr->data.string_value = value;
    return attr;
}

MlirAttribute *mlir_attribute_create_float(Arena *arena, string name, double value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_FLOAT;
    attr->name = name;
    attr->data.float_value = value;
    return attr;
}

MlirAttribute *mlir_attribute_create_bool(Arena *arena, string name, bool value) {
    struct MlirAttribute *attr = arena_alloc(arena, struct MlirAttribute);
    *attr = (struct MlirAttribute){0};
    attr->kind = ATTR_KIND_BOOL;
    attr->name = name;
    attr->data.bool_value = value;
    return attr;
}

// Value creation and manipulation
MlirValue *mlir_value_create_block_arg(Arena *arena, string register_name, uint32_t result_index, MlirType *type, MlirLocation *location) {
    struct MlirValue *value = arena_alloc(arena, struct MlirValue);
    *value = (struct MlirValue){0};
    value->kind = BLOCK_ARG;
    value->register_name = register_name;
    value->result_index = result_index;
    value->type = type;
    value->location = location;
    value->def = NULL;
    return value;
}

MlirValue *mlir_value_create_op_result(Arena *arena, void *def, uint32_t result_index, MlirType *type, string register_name, MlirLocation *location) {
    struct MlirValue *value = arena_alloc(arena, struct MlirValue);
    *value = (struct MlirValue){0};
    value->kind = OP_RESULT;
    value->def = def;
    value->result_index = result_index;
    value->type = type;
    value->register_name = register_name;
    value->location = location;
    return value;
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

void mlir_op_append_attribute(Arena *arena, MlirOperation *op, MlirAttribute *attr) {
    size_t new_count = op->n_attributes + 1;
    struct MlirAttribute **new_attrs = arena_alloc_array(arena, struct MlirAttribute*, new_count);
    if (op->attributes) {
        memcpy(new_attrs, op->attributes, op->n_attributes * sizeof(struct MlirAttribute*));
    }
    new_attrs[new_count - 1] = attr;
    op->attributes = new_attrs;
    op->n_attributes = new_count;
}

int64_t mlir_op_get_source_line_start(const MlirOperation *op) {
    return op->source_line_start;
}

MlirLocation *mlir_op_get_unnumbered_loc_def(const MlirOperation *op) {
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

string mlir_op_type_to_string(OpType type) {
    return op_type_to_string(type);
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

MlirLocation *mlir_location_create_unknown(Arena *arena, string original_text) {
    struct MlirLocation *loc = arena_alloc(arena, struct MlirLocation);
    *loc = (struct MlirLocation){0};
    loc->kind = LOC_KIND_UNKNOWN;
    loc->original_text = original_text;
    return loc;
}

MlirLocation *mlir_location_create_file(Arena *arena, string filename, int line, int column) {
    struct MlirLocation *loc = arena_alloc(arena, struct MlirLocation);
    *loc = (struct MlirLocation){0};
    loc->kind = LOC_KIND_FILE;
    loc->data.file.filename = filename;
    loc->data.file.line = line;
    loc->data.file.column = column;
    loc->original_text = format(arena, str_lit("loc({}:{}:{})"), filename, (int64_t)line, (int64_t)column);
    return loc;
}

MlirLocation *mlir_location_create_name(Arena *arena, string name) {
    struct MlirLocation *loc = arena_alloc(arena, struct MlirLocation);
    *loc = (struct MlirLocation){0};
    loc->kind = LOC_KIND_NAME;
    loc->data.name.name = name;
    loc->original_text = format(arena, str_lit("loc(\"{}\")"), name);
    return loc;
}

MlirLocation *mlir_location_create_ref(Arena *arena, int ref_id) {
    struct MlirLocation *loc = arena_alloc(arena, struct MlirLocation);
    *loc = (struct MlirLocation){0};
    loc->kind = LOC_KIND_REF;
    loc->data.ref.ref_id = ref_id;
    loc->original_text = format(arena, str_lit("loc(#loc{})"), (int64_t)ref_id);
    return loc;
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

#ifdef __cplusplus
}
#endif

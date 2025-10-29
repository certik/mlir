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

struct MLIR_Type {
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
            struct MLIR_Type *element_type;
            int64_t *shape;
            uint32_t rank;
        } shaped;
        struct {
            struct MLIR_Type *element_type;
            uint32_t address_space;
            bool has_address_space;
        } pointer;
    } data;
};

struct MLIR_Attribute {
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
            struct MLIR_Attribute **elements;
            size_t count;
        } array;
    } data;
    string name;
};

struct MLIR_Location {
    LocationKind kind;
    union {
        struct { string filename; int line; int column; } file;
        struct { string name; } name;
        struct { int ref_id; } ref;
    } data;
    string original_text;
};

struct MLIR_Value {
    MLIR_ValueKind kind;
    void *def;
    uint32_t result_index;
    struct MLIR_Type *type;
    string register_name;
    struct MLIR_Location *location;
};

struct MLIR_Op {
    MLIR_OpType op_type;
    struct MLIR_Value **operands;
    uint64_t n_operands;
    struct MLIR_Type **result_types;
    uint64_t n_result_types;
    struct MLIR_Attribute **attributes;
    uint64_t n_attributes;
    struct MLIR_Region **regions;
    uint64_t n_regions;
    string opname;
    struct MLIR_Value **results;
    uint64_t n_results;
    struct MLIR_Location *location;
    struct MLIR_Location *unnumbered_loc_def;
    string trailing_comment;
    int64_t source_line_start;
};

struct MLIR_Block {
    struct MLIR_Op **operations;
    uint64_t n_operations;
    struct MLIR_Value **arguments;
    uint64_t n_arguments;
};

struct MLIR_Region {
    struct MLIR_Block **blocks;
    uint64_t n_blocks;
};

// string_hash and string_equal are provided inline in mlir_parser.h

// Use SymbolTable, ScopedSymbolTable, and LocationMap from mlir_parser.h

typedef Parser Parser; // use Parser from mlir_parser.h

// Forward declarations for parser entry points implemented in mlir_parser.c
void parser_init(Arena *arena, Parser *parser, string text);
struct MLIR_Op; // forward decl
struct MLIR_Op* parse_module(Parser *parser);

// Use Operation/Block/Region/Type/Attribute/Location/ValueRef and vectors from mlir_parser.h


// API wrapper functions that cast between API types and concrete types
void MLIR_InitApi(MLIR_Op *root) {
    // Cast API type to concrete type
    (void)root;
    // No initialization required for the native C implementation.
}

MLIR_Op *MLIR_CreateOp(
    Arena *arena,
    MLIR_OpType type,
    string opname,
    MLIR_Attribute **attributes, size_t n_attributes,
    MLIR_Type **result_types, size_t n_result_types,
    MLIR_Value **results, size_t n_results,
    MLIR_Value **operands, size_t n_operands,
    MLIR_Region **regions, size_t n_regions,
    MLIR_Location *location,
    MLIR_Location *unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {
    MLIR_Op *op = arena_alloc(arena, MLIR_Op);
    *op = (MLIR_Op){0};
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

void MLIR_AppendBlockOp(Arena *arena, MLIR_Block *block, MLIR_Op *op) {
    MLIR_Op **new_ops = arena_alloc_array(arena, MLIR_Op*, block->n_operations + 1);
    if (block->operations) memcpy(new_ops, block->operations, block->n_operations * sizeof(MLIR_Op*));
    new_ops[block->n_operations] = op;
    block->operations = new_ops;
    block->n_operations++;
}

void MLIR_AppendBlockArg(Arena *arena, MLIR_Block *block, MLIR_Value *arg) {
    struct MLIR_Value **new_args = arena_alloc_array(arena, struct MLIR_Value*, block->n_arguments + 1);
    if (block->arguments) memcpy(new_args, block->arguments, block->n_arguments * sizeof(struct MLIR_Value*));
    new_args[block->n_arguments] = arg;
    block->arguments = new_args;
    block->n_arguments++;

    // Automatically set the def field to point to the block for block arguments
    if (arg) {
        arg->def = block;
    }
}

void MLIR_AppendRegionBlock(Arena *arena, MLIR_Region *region, MLIR_Block *block) {
    MLIR_Block **new_blocks = arena_alloc_array(arena, MLIR_Block*, region->n_blocks + 1);
    if (region->blocks) memcpy(new_blocks, region->blocks, region->n_blocks * sizeof(MLIR_Block*));
    new_blocks[region->n_blocks] = block;
    region->blocks = new_blocks;
    region->n_blocks++;
}

size_t MLIR_GetRegionNumBlocks(const MLIR_Region *region) {
    return region->n_blocks;
}

MLIR_Block *MLIR_GetRegionBlock(const MLIR_Region *region, size_t idx) {
    return region->blocks[idx];
}

size_t MLIR_GetBlockNumOps(const MLIR_Block *block) {
    return block->n_operations;
}

MLIR_Op *MLIR_GetBlockOp(const MLIR_Block *block, size_t idx) {
    return block->operations[idx];
}

MLIR_OpType MLIR_GetOpType(const MLIR_Op *op) {
    return op->op_type;
}

size_t MLIR_GetOpNumRegions(const MLIR_Op *op) {
    return op->n_regions;
}

MLIR_Region *MLIR_GetOpRegion(const MLIR_Op *op, size_t idx) {
    return op->regions[idx];
}

string MLIR_GetOpTrailingComment(const MLIR_Op *op) {
    return op->trailing_comment;
}

// Additional accessors for printer support
size_t MLIR_GetOpNumAttributes(const MLIR_Op *op) {
    return op->n_attributes;
}

MLIR_Attribute *MLIR_GetOpAttribute(const MLIR_Op *op, size_t idx) {
    return op->attributes[idx];
}

size_t MLIR_GetOpNumOperands(const MLIR_Op *op) {
    return op->n_operands;
}

MLIR_Value *MLIR_GetOpOperand(const MLIR_Op *op, size_t idx) {
    return op->operands[idx];
}

size_t MLIR_GetOpNumResults(const MLIR_Op *op) {
    return op->n_results;
}

MLIR_Value *MLIR_GetOpResult(const MLIR_Op *op, size_t idx) {
    return op->results[idx];
}

MLIR_Location *MLIR_GetOpLocation(const MLIR_Op *op) {
    return op->location;
}

string MLIR_GetOpName(const MLIR_Op *op) {
    return op->opname;
}

string MLIR_GetOpName_string(const MLIR_Op *op) {
    return op->opname;
}

size_t MLIR_GetOpNumResultTypes(const MLIR_Op *op) {
    return op->n_result_types;
}

MLIR_Type *MLIR_GetOpResult_type(const MLIR_Op *op, size_t idx) {
    return op->result_types[idx];
}

size_t MLIR_GetBlockNumArgs(const MLIR_Block *block) {
    return block->n_arguments;
}

MLIR_Value *MLIR_GetBlockArg(const MLIR_Block *block, size_t idx) {
    return block->arguments[idx];
}

// Value metadata accessors
MLIR_Location *MLIR_GetValueLocation(const MLIR_Value *value) {
    return value->location;
}

// Type to string
string MLIR_GetTypeString(Arena *arena, MLIR_Type *type) {
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
                string elem_str = MLIR_GetTypeString(arena, type->data.shaped.element_type);
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
                string elem_str = MLIR_GetTypeString(arena, type->data.shaped.element_type);
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
                string elem_str = MLIR_GetTypeString(arena, type->data.pointer.element_type);
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
MLIR_Type *MLIR_CreateTypeInteger(Arena *arena, uint32_t width, bool is_signed) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_INTEGER;
    type->data.integer.width = width;
    type->data.integer.is_signed = is_signed;
    return type;
}

MLIR_Type *MLIR_CreateTypeFloat(Arena *arena, uint32_t width, bool is_bfloat) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_FLOAT;
    type->data.floating.width = width;
    type->data.floating.is_bfloat = is_bfloat;
    return type;
}

MLIR_Type *MLIR_CreateTypeIndex(Arena *arena) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_INDEX;
    return type;
}

MLIR_Type *MLIR_CreateTypeUnknown(Arena *arena) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_UNKNOWN;
    return type;
}

static void copy_shape(Arena *arena, struct MLIR_Type *type, const int64_t *shape, size_t rank) {
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

MLIR_Type *MLIR_CreateTypeTensor(Arena *arena, const int64_t *shape, size_t rank, MLIR_Type *element_type) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_TENSOR;
    type->data.shaped.element_type = element_type;
    copy_shape(arena, type, shape, rank);
    return type;
}

MLIR_Type *MLIR_CreateTypeMemref(Arena *arena, const int64_t *shape, size_t rank, MLIR_Type *element_type) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_MEMREF;
    type->data.shaped.element_type = element_type;
    copy_shape(arena, type, shape, rank);
    return type;
}

MLIR_Type *MLIR_CreateTypePointer(Arena *arena, MLIR_Type *element_type, bool has_address_space, uint32_t address_space) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_POINTER;
    type->data.pointer.element_type = element_type;
    type->data.pointer.has_address_space = has_address_space;
    type->data.pointer.address_space = address_space;
    return type;
}

void MLIR_SetTypeIntegerProperties(MLIR_Type *type, uint32_t width, bool is_signed) {
    type->kind = TYPE_KIND_INTEGER;
    type->data.integer.width = width;
    type->data.integer.is_signed = is_signed;
}

void MLIR_SetTypeFloatProperties(MLIR_Type *type, uint32_t width, bool is_bfloat) {
    type->kind = TYPE_KIND_FLOAT;
    type->data.floating.width = width;
    type->data.floating.is_bfloat = is_bfloat;
}

// Attribute creation and manipulation
MLIR_Attribute *MLIR_CreateAttributeInteger(Arena *arena, string name, int64_t value) {
    struct MLIR_Attribute *attr = arena_alloc(arena, struct MLIR_Attribute);
    *attr = (struct MLIR_Attribute){0};
    attr->kind = ATTR_KIND_INTEGER;
    attr->name = name;
    attr->data.integer_value = value;
    return attr;
}

MLIR_Attribute *MLIR_CreateAttributeString(Arena *arena, string name, string value) {
    struct MLIR_Attribute *attr = arena_alloc(arena, struct MLIR_Attribute);
    *attr = (struct MLIR_Attribute){0};
    attr->kind = ATTR_KIND_STRING;
    attr->name = name;
    attr->data.string_value = value;
    return attr;
}

MLIR_Attribute *MLIR_CreateAttributeFloat(Arena *arena, string name, double value) {
    struct MLIR_Attribute *attr = arena_alloc(arena, struct MLIR_Attribute);
    *attr = (struct MLIR_Attribute){0};
    attr->kind = ATTR_KIND_FLOAT;
    attr->name = name;
    attr->data.float_value = value;
    return attr;
}

MLIR_Attribute *MLIR_CreateAttributeBool(Arena *arena, string name, bool value) {
    struct MLIR_Attribute *attr = arena_alloc(arena, struct MLIR_Attribute);
    *attr = (struct MLIR_Attribute){0};
    attr->kind = ATTR_KIND_BOOL;
    attr->name = name;
    attr->data.bool_value = value;
    return attr;
}

MLIR_Attribute *MLIR_CreateAttributeArray(Arena *arena, string name, MLIR_Attribute **elements, size_t count) {
    struct MLIR_Attribute *attr = arena_alloc(arena, struct MLIR_Attribute);
    *attr = (struct MLIR_Attribute){0};
    attr->kind = ATTR_KIND_ARRAY;
    attr->name = name;
    attr->data.array.elements = elements;
    attr->data.array.count = count;
    return attr;
}

MLIR_Attribute *MLIR_CreateAttributeDict(Arena *arena, string name, MLIR_Attribute **elements, size_t count) {
    struct MLIR_Attribute *attr = arena_alloc(arena, struct MLIR_Attribute);
    *attr = (struct MLIR_Attribute){0};
    attr->kind = ATTR_KIND_DICT;
    attr->name = name;
    // Dictionary uses the same storage as array
    attr->data.array.elements = elements;
    attr->data.array.count = count;
    return attr;
}

// Value creation and manipulation
MLIR_Value *MLIR_CreateValueBlockArg(Arena *arena, string register_name, uint32_t result_index, MLIR_Type *type, MLIR_Location *location) {
    struct MLIR_Value *value = arena_alloc(arena, struct MLIR_Value);
    *value = (struct MLIR_Value){0};
    value->kind = BLOCK_ARG;
    value->register_name = register_name;
    value->result_index = result_index;
    value->type = type;
    value->location = location;
    value->def = NULL;
    return value;
}

MLIR_Value *MLIR_CreateValueOpResult(Arena *arena, void *def, uint32_t result_index, MLIR_Type *type, string register_name, MLIR_Location *location) {
    struct MLIR_Value *value = arena_alloc(arena, struct MLIR_Value);
    *value = (struct MLIR_Value){0};
    value->kind = OP_RESULT;
    value->def = def;
    value->result_index = result_index;
    value->type = type;
    value->register_name = register_name;
    value->location = location;
    return value;
}

// Block and Region creation
MLIR_Block *MLIR_CreateBlock(Arena *arena) {
    struct MLIR_Block *block = arena_alloc(arena, struct MLIR_Block);
    *block = (struct MLIR_Block){0};
    return block;
}

MLIR_Region *MLIR_CreateRegion(Arena *arena) {
    struct MLIR_Region *region = arena_alloc(arena, struct MLIR_Region);
    *region = (struct MLIR_Region){0};
    return region;
}

void MLIR_AppendOpAttribute(Arena *arena, MLIR_Op *op, MLIR_Attribute *attr) {
    size_t new_count = op->n_attributes + 1;
    struct MLIR_Attribute **new_attrs = arena_alloc_array(arena, struct MLIR_Attribute*, new_count);
    if (op->attributes) {
        memcpy(new_attrs, op->attributes, op->n_attributes * sizeof(struct MLIR_Attribute*));
    }
    new_attrs[new_count - 1] = attr;
    op->attributes = new_attrs;
    op->n_attributes = new_count;
}

int64_t MLIR_GetOpSourceLineStart(const MLIR_Op *op) {
    return op->source_line_start;
}

MLIR_Location *MLIR_GetOpUnnumberedLocationDef(const MLIR_Op *op) {
    return op->unnumbered_loc_def;
}

// Attribute accessors
MLIR_AttrKind MLIR_GetAttributeKind(const MLIR_Attribute *attr) {
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

string MLIR_GetAttributeName(const MLIR_Attribute *attr) {
    return attr->name;
}

int64_t MLIR_GetAttributeInteger(const MLIR_Attribute *attr) {
    return attr->data.integer_value;
}

string MLIR_GetAttributeString(const MLIR_Attribute *attr) {
    return attr->data.string_value;
}

double MLIR_GetAttributeFloat(const MLIR_Attribute *attr) {
    return attr->data.float_value;
}

bool MLIR_GetAttributeBool(const MLIR_Attribute *attr) {
    return attr->data.bool_value;
}

size_t MLIR_GetAttributeArraySize(const MLIR_Attribute *attr) {
    return attr->data.array.count;
}

MLIR_Attribute *MLIR_GetAttributeArrayElement(const MLIR_Attribute *attr, size_t idx) {
    if (idx >= attr->data.array.count) return NULL;
    return attr->data.array.elements[idx];
}

size_t MLIR_GetAttributeDictSize(const MLIR_Attribute *attr) {
    return attr->data.array.count;
}

MLIR_Attribute *MLIR_GetAttributeDictElement(const MLIR_Attribute *attr, size_t idx) {
    if (idx >= attr->data.array.count) return NULL;
    return attr->data.array.elements[idx];
}

// Value accessors
MLIR_ValueKind MLIR_GetValueKind(const MLIR_Value *value) {
    return value->kind;
}

MLIR_Type *MLIR_GetValueType(const MLIR_Value *value) {
    return value->type;
}

string MLIR_GetValueRegisterName(const MLIR_Value *value) {
    return value->register_name;
}

uint32_t MLIR_GetValueResultIndex(const MLIR_Value *value) {
    return value->result_index;
}

MLIR_Op *MLIR_GetValueDefiningOp(const MLIR_Value *value) {
    return (MLIR_Op*)value->def;
}

string MLIR_MLIR_OpTypeToString(MLIR_OpType type) {
    return op_type_to_string(type);
}

// Location accessors
MLIR_LocationKind MLIR_GetLocationKind(const MLIR_Location *loc) {
    switch (loc->kind) {
        case LOC_KIND_FILE: return MLIR_LOC_FILE;
        case LOC_KIND_NAME: return MLIR_LOC_NAME;
        case LOC_KIND_CALLSITE: return MLIR_LOC_CALLSITE;
        case LOC_KIND_FUSED: return MLIR_LOC_FUSED;
        case LOC_KIND_REF: return MLIR_LOC_REF;
        case LOC_KIND_UNKNOWN: default: return MLIR_LOC_UNKNOWN;
    }
}

string MLIR_GetLocationOriginalText(const MLIR_Location *loc) {
    return loc->original_text;
}

string MLIR_GetLocationFileFilename(const MLIR_Location *loc) {
    return loc->data.file.filename;
}

int MLIR_GetLocationFileLine(const MLIR_Location *loc) {
    return loc->data.file.line;
}

int MLIR_GetLocationFileColumn(const MLIR_Location *loc) {
    return loc->data.file.column;
}

string MLIR_GetLocationName(const MLIR_Location *loc) {
    return loc->data.name.name;
}

int MLIR_GetLocationRefId(const MLIR_Location *loc) {
    return loc->data.ref.ref_id;
}

MLIR_Location *MLIR_CreateLocationUnknown(Arena *arena, string original_text) {
    struct MLIR_Location *loc = arena_alloc(arena, struct MLIR_Location);
    *loc = (struct MLIR_Location){0};
    loc->kind = LOC_KIND_UNKNOWN;
    loc->original_text = original_text;
    return loc;
}

MLIR_Location *MLIR_CreateLocationFile(Arena *arena, string filename, int line, int column) {
    struct MLIR_Location *loc = arena_alloc(arena, struct MLIR_Location);
    *loc = (struct MLIR_Location){0};
    loc->kind = LOC_KIND_FILE;
    loc->data.file.filename = filename;
    loc->data.file.line = line;
    loc->data.file.column = column;
    loc->original_text = format(arena, str_lit("loc({}:{}:{})"), filename, (int64_t)line, (int64_t)column);
    return loc;
}

MLIR_Location *MLIR_CreateLocationName(Arena *arena, string name) {
    struct MLIR_Location *loc = arena_alloc(arena, struct MLIR_Location);
    *loc = (struct MLIR_Location){0};
    loc->kind = LOC_KIND_NAME;
    loc->data.name.name = name;
    loc->original_text = format(arena, str_lit("loc(\"{}\")"), name);
    return loc;
}

MLIR_Location *MLIR_CreateLocationRef(Arena *arena, int ref_id) {
    struct MLIR_Location *loc = arena_alloc(arena, struct MLIR_Location);
    *loc = (struct MLIR_Location){0};
    loc->kind = LOC_KIND_REF;
    loc->data.ref.ref_id = ref_id;
    loc->original_text = format(arena, str_lit("loc(#loc{})"), (int64_t)ref_id);
    return loc;
}

// New API functions for parser compatibility

MLIR_Type *MLIR_CreateTypeOpaque(Arena *arena, string name) {
    struct MLIR_Type *type = arena_alloc(arena, struct MLIR_Type);
    *type = (struct MLIR_Type){0};
    type->kind = TYPE_KIND_OPAQUE;
    // Store the name in a way that can be retrieved later if needed
    return type;
}

void MLIR_SetTypeTensorProperties(MLIR_Type *type, const int64_t *shape, size_t rank, MLIR_Type *element_type) {
    type->kind = TYPE_KIND_TENSOR;
    type->data.shaped.element_type = element_type;
    // Note: These functions expect the shape to be pre-allocated, they just set the pointer
    type->data.shaped.shape = (int64_t*)shape;
    type->data.shaped.rank = (uint32_t)rank;
}

void MLIR_SetTypeMemrefProperties(MLIR_Type *type, const int64_t *shape, size_t rank, MLIR_Type *element_type) {
    type->kind = TYPE_KIND_MEMREF;
    type->data.shaped.element_type = element_type;
    // Note: These functions expect the shape to be pre-allocated, they just set the pointer
    type->data.shaped.shape = (int64_t*)shape;
    type->data.shaped.rank = (uint32_t)rank;
}

void MLIR_SetTypePointerProperties(MLIR_Type *type, MLIR_Type *element_type, bool has_address_space, uint32_t address_space) {
    type->kind = TYPE_KIND_POINTER;
    type->data.pointer.element_type = element_type;
    type->data.pointer.has_address_space = has_address_space;
    type->data.pointer.address_space = address_space;
}

// Type introspection functions
bool MLIR_IsTypeInteger(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_INTEGER;
}

bool MLIR_IsTypeFloat(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_FLOAT;
}

bool MLIR_IsTypeTensor(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_TENSOR;
}

bool MLIR_IsTypeMemref(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_MEMREF;
}

bool MLIR_IsTypePointer(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_POINTER;
}

bool MLIR_IsTypeIndex(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_INDEX;
}

bool MLIR_IsTypeUnknown(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_UNKNOWN;
}

bool MLIR_IsTypeOpaque(const MLIR_Type *type) {
    return type->kind == TYPE_KIND_OPAQUE;
}

#ifdef __cplusplus
}
#endif

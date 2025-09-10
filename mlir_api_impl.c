#include <base/arena.h>
#include <base/string.h>
#include <base/hashtable.h>
#include <base/vector.h>
#include "tokenizer.h"
#include "mlir_api.h"
#include "mlir_parser.h"
#include "mlir_ir_internal.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// string_hash and string_equal are provided inline in mlir_parser.h

// Use SymbolTable, ScopedSymbolTable, and LocationMap from mlir_parser.h

typedef Parser Parser; // use Parser from mlir_parser.h

// Forward declarations for parser entry points implemented in mlir_parser.c
void parser_init(Arena *arena, Parser *parser, string text);
struct Operation; // forward decl
struct Operation* parse_module(Parser *parser);

// Use Operation/Block/Region/Type/Attribute/Location/ValueRef and vectors from mlir_parser.h


// API wrapper functions that cast between API types and concrete types
void mlir_api_init(MlirOperation *root) {
    // Cast API type to concrete type
    Operation *concrete_root = (Operation*)root;
    (void)concrete_root;
    // No initialization required for the native C implementation.
}

MlirOperation *mlir_op_create(Arena *arena, OpType type) {
    // Cast Arena* to Arena*
    Arena *concrete_arena = (Arena*)arena;
    Operation *op = arena_alloc(concrete_arena, Operation);
    *op = (Operation){0};
    op->op_type = type;
    // Cast concrete type to API type
    return (MlirOperation*)op;
}

void mlir_op_add_region(Arena *arena, MlirOperation *op, MlirRegion *region) {
    Arena *concrete_arena = (Arena*)arena;
    Operation *concrete_op = (Operation*)op;
    Region *concrete_region = (Region*)region;
    Region **new_regions = arena_alloc_array(concrete_arena, Region*, concrete_op->n_regions + 1);
    if (concrete_op->regions) memcpy(new_regions, concrete_op->regions, concrete_op->n_regions * sizeof(Region*));
    new_regions[concrete_op->n_regions] = concrete_region;
    concrete_op->regions = new_regions;
    concrete_op->n_regions++;
}

void mlir_op_add_operand(Arena *arena, MlirOperation *op, MlirValue *operand) {
    Arena *concrete_arena = (Arena*)arena;
    Operation *concrete_op = (Operation*)op;
    ValueRef *concrete_operand = (ValueRef*)operand;
    ValueRef **new_operands = arena_alloc_array(concrete_arena, ValueRef*, concrete_op->n_operands + 1);
    if (concrete_op->operands) memcpy(new_operands, concrete_op->operands, concrete_op->n_operands * sizeof(ValueRef*));
    new_operands[concrete_op->n_operands] = concrete_operand;
    concrete_op->operands = new_operands;
    concrete_op->n_operands++;
}

void mlir_op_add_result(Arena *arena, MlirOperation *op, MlirValue *result) {
    Arena *concrete_arena = (Arena*)arena;
    Operation *concrete_op = (Operation*)op;
    ValueRef *concrete_result = (ValueRef*)result;
    ValueRef **new_results = arena_alloc_array(concrete_arena, ValueRef*, concrete_op->n_results + 1);
    if (concrete_op->results) memcpy(new_results, concrete_op->results, concrete_op->n_results * sizeof(ValueRef*));
    new_results[concrete_op->n_results] = concrete_result;
    concrete_op->results = new_results;
    concrete_op->n_results++;
}

void mlir_block_add_operation(Arena *arena, MlirBlock *block, MlirOperation *op) {
    Arena *concrete_arena = (Arena*)arena;
    Block *concrete_block = (Block*)block;
    Operation *concrete_op = (Operation*)op;
    Operation **new_ops = arena_alloc_array(concrete_arena, Operation*, concrete_block->n_operations + 1);
    if (concrete_block->operations) memcpy(new_ops, concrete_block->operations, concrete_block->n_operations * sizeof(Operation*));
    new_ops[concrete_block->n_operations] = concrete_op;
    concrete_block->operations = new_ops;
    concrete_block->n_operations++;
}

void mlir_block_add_argument(Arena *arena, MlirBlock *block, MlirValue *arg) {
    Arena *concrete_arena = (Arena*)arena;
    Block *concrete_block = (Block*)block;
    ValueRef *concrete_arg = (ValueRef*)arg;
    ValueRef **new_args = arena_alloc_array(concrete_arena, ValueRef*, concrete_block->n_arguments + 1);
    if (concrete_block->arguments) memcpy(new_args, concrete_block->arguments, concrete_block->n_arguments * sizeof(ValueRef*));
    new_args[concrete_block->n_arguments] = concrete_arg;
    concrete_block->arguments = new_args;
    concrete_block->n_arguments++;
}

void mlir_region_add_block(Arena *arena, MlirRegion *region, MlirBlock *block) {
    Arena *concrete_arena = (Arena*)arena;
    Region *concrete_region = (Region*)region;
    Block *concrete_block = (Block*)block;
    Block **new_blocks = arena_alloc_array(concrete_arena, Block*, concrete_region->n_blocks + 1);
    if (concrete_region->blocks) memcpy(new_blocks, concrete_region->blocks, concrete_region->n_blocks * sizeof(Block*));
    new_blocks[concrete_region->n_blocks] = concrete_block;
    concrete_region->blocks = new_blocks;
    concrete_region->n_blocks++;
}

size_t mlir_region_num_blocks(const MlirRegion *region) {
    const Region *concrete_region = (const Region*)region;
    return concrete_region->n_blocks;
}

MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx) {
    const Region *concrete_region = (const Region*)region;
    return (MlirBlock*)concrete_region->blocks[idx];
}

size_t mlir_block_num_operations(const MlirBlock *block) {
    const Block *concrete_block = (const Block*)block;
    return concrete_block->n_operations;
}

MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx) {
    const Block *concrete_block = (const Block*)block;
    return (MlirOperation*)concrete_block->operations[idx];
}

OpType mlir_operation_get_type(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->op_type;
}

size_t mlir_operation_num_regions(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_regions;
}

MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirRegion*)concrete_op->regions[idx];
}

// Additional accessors for printer support
size_t mlir_operation_num_attributes(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_attributes;
}

MlirAttribute *mlir_operation_get_attribute(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirAttribute*)concrete_op->attributes[idx];
}

size_t mlir_operation_num_operands(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_operands;
}

MlirValue *mlir_operation_get_operand(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirValue*)concrete_op->operands[idx];
}

size_t mlir_operation_num_results(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_results;
}

MlirValue *mlir_operation_get_result(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirValue*)concrete_op->results[idx];
}

MlirLocation *mlir_operation_get_location(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirLocation*)concrete_op->location;
}

const char *mlir_operation_get_name(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->opname.str;
}

size_t mlir_operation_num_result_types(const MlirOperation *op) {
    const Operation *concrete_op = (const Operation*)op;
    return concrete_op->n_result_types;
}

MlirType *mlir_operation_get_result_type(const MlirOperation *op, size_t idx) {
    const Operation *concrete_op = (const Operation*)op;
    return (MlirType*)concrete_op->result_types[idx];
}

size_t mlir_block_num_arguments(const MlirBlock *block) {
    const Block *concrete_block = (const Block*)block;
    return concrete_block->n_arguments;
}

MlirValue *mlir_block_get_argument(const MlirBlock *block, size_t idx) {
    const Block *concrete_block = (const Block*)block;
    return (MlirValue*)concrete_block->arguments[idx];
}

// Type to string
string mlir_type_to_string(Arena *arena, MlirType *type) {
    return type_to_string(arena, (Type*)type);
}

// Type creation and manipulation
MlirType *mlir_type_create_integer(Arena *arena, uint32_t width, bool is_signed) {
    Arena *concrete_arena = (Arena*)arena;
    Type *type = arena_alloc(concrete_arena, Type);
    *type = (Type){0};
    type->kind = TYPE_KIND_INTEGER;
    type->data.integer.width = width;
    type->data.integer.is_signed = is_signed;
    return (MlirType*)type;
}

MlirType *mlir_type_create_float(Arena *arena, uint32_t width, bool is_bfloat) {
    Arena *concrete_arena = (Arena*)arena;
    Type *type = arena_alloc(concrete_arena, Type);
    *type = (Type){0};
    type->kind = TYPE_KIND_FLOAT;
    type->data.floating.width = width;
    type->data.floating.is_bfloat = is_bfloat;
    return (MlirType*)type;
}

void mlir_type_set_integer_properties(MlirType *type, uint32_t width, bool is_signed) {
    Type *concrete_type = (Type*)type;
    concrete_type->kind = TYPE_KIND_INTEGER;
    concrete_type->data.integer.width = width;
    concrete_type->data.integer.is_signed = is_signed;
}

void mlir_type_set_float_properties(MlirType *type, uint32_t width, bool is_bfloat) {
    Type *concrete_type = (Type*)type;
    concrete_type->kind = TYPE_KIND_FLOAT;
    concrete_type->data.floating.width = width;
    concrete_type->data.floating.is_bfloat = is_bfloat;
}

// Attribute creation and manipulation
MlirAttribute *mlir_attribute_create_integer(Arena *arena, int64_t value) {
    Arena *concrete_arena = (Arena*)arena;
    Attribute *attr = arena_alloc(concrete_arena, Attribute);
    *attr = (Attribute){0};
    attr->kind = ATTR_KIND_INTEGER;
    attr->data.integer_value = value;
    return (MlirAttribute*)attr;
}

MlirAttribute *mlir_attribute_create_string(Arena *arena, const char *str, size_t len) {
    Arena *concrete_arena = (Arena*)arena;
    Attribute *attr = arena_alloc(concrete_arena, Attribute);
    *attr = (Attribute){0};
    attr->kind = ATTR_KIND_STRING;
    attr->data.string_value = (string){(char*)str, len};
    return (MlirAttribute*)attr;
}

void mlir_attribute_set_name(MlirAttribute *attr, const char *name, size_t name_len) {
    Attribute *concrete_attr = (Attribute*)attr;
    concrete_attr->name = (string){(char*)name, name_len};
}

// Value creation and manipulation
MlirValue *mlir_value_create(Arena *arena, int value_kind) {
    Arena *concrete_arena = (Arena*)arena;
    ValueRef *value = arena_alloc(concrete_arena, ValueRef);
    *value = (ValueRef){0};
    value->kind = (ValueKind)value_kind;
    return (MlirValue*)value;
}

void mlir_value_set_type(MlirValue *value, MlirType *type) {
    ValueRef *concrete_value = (ValueRef*)value;
    Type *concrete_type = (Type*)type;
    concrete_value->type = concrete_type;
}

void mlir_value_set_register_name(MlirValue *value, const char *name, size_t name_len) {
    ValueRef *concrete_value = (ValueRef*)value;
    concrete_value->register_name = (string){(char*)name, name_len};
}

void mlir_value_set_result_index(MlirValue *value, uint32_t index) {
    ValueRef *concrete_value = (ValueRef*)value;
    concrete_value->result_index = index;
}

void mlir_value_set_def(MlirValue *value, void *def) {
    ValueRef *concrete_value = (ValueRef*)value;
    concrete_value->def = def;
}

// Block and Region creation
MlirBlock *mlir_block_create(Arena *arena) {
    Arena *concrete_arena = (Arena*)arena;
    Block *block = arena_alloc(concrete_arena, Block);
    *block = (Block){0};
    return (MlirBlock*)block;
}

MlirRegion *mlir_region_create(Arena *arena) {
    Arena *concrete_arena = (Arena*)arena;
    Region *region = arena_alloc(concrete_arena, Region);
    *region = (Region){0};
    return (MlirRegion*)region;
}

// Operation properties
void mlir_operation_set_name(MlirOperation *op, const char *name, size_t name_len) {
    Operation *concrete_op = (Operation*)op;
    concrete_op->opname = (string){(char*)name, name_len};
}

void mlir_operation_set_result_types(MlirOperation *op, MlirType **types, size_t count) {
    Operation *concrete_op = (Operation*)op;
    concrete_op->result_types = (Type**)types;
    concrete_op->n_result_types = count;
}

void mlir_operation_set_attributes(MlirOperation *op, MlirAttribute **attrs, size_t count) {
    Operation *concrete_op = (Operation*)op;
    concrete_op->attributes = (Attribute**)attrs;
    concrete_op->n_attributes = count;
}

void mlir_operation_set_results(MlirOperation *op, MlirValue **results, size_t count) {
    Operation *concrete_op = (Operation*)op;
    concrete_op->results = (ValueRef**)results;
    concrete_op->n_results = count;
}

void mlir_operation_set_operands(MlirOperation *op, MlirValue **operands, size_t count) {
    Operation *concrete_op = (Operation*)op;
    concrete_op->operands = (ValueRef**)operands;
    concrete_op->n_operands = count;
}

// Attribute accessors
MlirAttrKind mlir_attribute_get_kind(const MlirAttribute *attr) {
    const Attribute *a = (const Attribute*)attr;
    switch (a->kind) {
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
    const Attribute *a = (const Attribute*)attr;
    return a->name;
}

int64_t mlir_attribute_get_integer(const MlirAttribute *attr) {
    const Attribute *a = (const Attribute*)attr;
    return a->data.integer_value;
}

string mlir_attribute_get_string(const MlirAttribute *attr) {
    const Attribute *a = (const Attribute*)attr;
    return a->data.string_value;
}

double mlir_attribute_get_float(const MlirAttribute *attr) {
    const Attribute *a = (const Attribute*)attr;
    return a->data.float_value;
}

bool mlir_attribute_get_bool(const MlirAttribute *attr) {
    const Attribute *a = (const Attribute*)attr;
    return a->data.bool_value;
}

// Value accessors
int mlir_value_get_kind(const MlirValue *value) {
    const ValueRef *v = (const ValueRef*)value;
    return (int)v->kind;
}

MlirType *mlir_value_get_type(const MlirValue *value) {
    const ValueRef *v = (const ValueRef*)value;
    return (MlirType*)v->type;
}

string mlir_value_get_register_name(const MlirValue *value) {
    const ValueRef *v = (const ValueRef*)value;
    return v->register_name;
}

uint32_t mlir_value_get_result_index(const MlirValue *value) {
    const ValueRef *v = (const ValueRef*)value;
    return v->result_index;
}

MlirOperation *mlir_value_get_def_op(const MlirValue *value) {
    const ValueRef *v = (const ValueRef*)value;
    return (MlirOperation*)v->def;
}

const char *mlir_op_type_to_string(OpType type) {
    string s = op_type_to_string(type);
    return s.str;
}

// Parser functions
MlirParser *mlir_parser_create(Arena *arena) {
    Arena *concrete_arena = (Arena*)arena;
    Parser *parser = arena_alloc(concrete_arena, Parser);
    return (MlirParser*)parser;
}

void mlir_parser_init(Arena *arena, MlirParser *parser, const char *input, size_t input_len) {
    Arena *concrete_arena = (Arena*)arena;
    Parser *concrete_parser = (Parser*)parser;
    string input_string = {
        .str = (char*)input,
        .size = input_len
    };
    parser_init(concrete_arena, concrete_parser, input_string);
}

MlirOperation *mlir_parse_module(MlirParser *parser) {
    Parser *concrete_parser = (Parser*)parser;
    Operation *module = parse_module(concrete_parser);
    return (MlirOperation*)module;
}

void mlir_parser_get_location_map(MlirParser *parser, void **location_map) {
    Parser *concrete_parser = (Parser*)parser;
    *location_map = &concrete_parser->location_map;
}

const char *mlir_tokentype_to_string(int token_type) {
    // Reuse existing tokenizer pretty-printer
    string s = tokentype_to_string((TokenType)token_type);
    return (const char*)s.str;
}

// Location accessors
MlirLocationKind mlir_location_get_kind(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    switch (l->kind) {
        case LOC_KIND_FILE: return MLIR_LOC_FILE;
        case LOC_KIND_NAME: return MLIR_LOC_NAME;
        case LOC_KIND_CALLSITE: return MLIR_LOC_CALLSITE;
        case LOC_KIND_FUSED: return MLIR_LOC_FUSED;
        case LOC_KIND_REF: return MLIR_LOC_REF;
        case LOC_KIND_UNKNOWN: default: return MLIR_LOC_UNKNOWN;
    }
}

string mlir_location_get_original_text(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    return l->original_text;
}

string mlir_location_get_file_filename(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    return l->data.file.filename;
}

int mlir_location_get_file_line(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    return l->data.file.line;
}

int mlir_location_get_file_column(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    return l->data.file.column;
}

string mlir_location_get_name(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    return l->data.name.name;
}

int mlir_location_get_ref_id(const MlirLocation *loc) {
    const Location *l = (const Location*)loc;
    return l->data.ref.ref_id;
}

#ifdef __cplusplus
}
#endif

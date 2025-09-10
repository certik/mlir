#pragma once

#include <base/arena.h>
#include <base/string.h>
#include <base/vector.h>
#include "mlir_api.h"
#include "mlir_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal concrete IR node definitions. Not exposed via public API.

// Type kinds for MLIR type system
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

typedef struct Type {
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
            struct Type *element_type;
            int64_t *shape;
            uint32_t rank;
        } shaped;  // memref/tensor
        struct {
            struct Type *element_type;
            uint32_t address_space;
            bool has_address_space;
        } pointer;
    } data;
} Type;

typedef struct Attribute {
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
            struct Attribute **elements;
            size_t count;
        } array;
    } data;
    string name;
} Attribute;

typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,
    LOC_KIND_NAME,
    LOC_KIND_CALLSITE,
    LOC_KIND_FUSED,
    LOC_KIND_REF
} LocationKind;

typedef struct Location {
    LocationKind kind;
    union {
        struct { string filename; int line; int column; } file;
        struct { string name; } name;
        struct { int ref_id; } ref;
    } data;
    string original_text;
} Location;

typedef struct ValueRef {
    int /*ValueKind*/ kind;
    void* def;               // Operation* or Block*
    uint32_t result_index;
    Type *type;
    string register_name;

    Location *location;
    bool has_divisibility;
    int64_t divisibility_value;
    Type *divisibility_type;
    bool has_max_divisibility;
    int64_t max_divisibility_value;
    Type *max_divisibility_type;
} ValueRef;

typedef struct Operation Operation;
typedef struct Block Block;
typedef struct Region Region;

struct Operation {
    OpType op_type;
    ValueRef **operands;
    uint64_t n_operands;
    Type **result_types;
    uint64_t n_result_types;
    Attribute **attributes;
    uint64_t n_attributes;
    Region **regions;
    uint64_t n_regions;
    string opname;
    ValueRef **results;
    uint64_t n_results;
    Location *location;
    Location *unnumbered_loc_def;
    string trailing_comment;
    int64_t source_line_start;
};

struct Block {
    Operation **operations;
    uint64_t n_operations;
    ValueRef **arguments;
    uint64_t n_arguments;
};

struct Region {
    Block **blocks;
    uint64_t n_blocks;
};

// Common vectors used internally
DEFINE_VECTOR_FOR_TYPE(Operation*, VecOperation)
DEFINE_VECTOR_FOR_TYPE(ValueRef*, VecValueRef)
DEFINE_VECTOR_FOR_TYPE(Block*, VecBlock)

#ifdef __cplusplus
}
#endif

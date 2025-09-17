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
        } shaped;  // memref/tensor
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

typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,
    LOC_KIND_NAME,
    LOC_KIND_CALLSITE,
    LOC_KIND_FUSED,
    LOC_KIND_REF
} LocationKind;

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
    int /*ValueKind*/ kind;
    void* def;               // Operation* or Block*
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

// Common vectors used internally
DEFINE_VECTOR_FOR_TYPE(struct MlirOperation*, VecOperation)
DEFINE_VECTOR_FOR_TYPE(struct MlirValue*, VecValue)
DEFINE_VECTOR_FOR_TYPE(struct MlirBlock*, VecBlock)

#ifdef __cplusplus
}
#endif

// Public MLIR C API declarations without exposing internal data structures.
// Implementations can back these APIs with different MLIR representations.

#pragma once

#include <stddef.h>

#include <base/arena.h>
#include <base/string.h>
#include <base/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Opaque forward declarations
// -----------------------------------------------------------------------------

typedef struct MlirOperation MlirOperation;
typedef struct MlirRegion MlirRegion;
typedef struct MlirBlock MlirBlock;
typedef struct MlirValue MlirValue;
typedef struct MlirType MlirType;
typedef struct MlirAttribute MlirAttribute;
typedef struct MlirLocation MlirLocation;

typedef struct MlirLocationMap {
    void *impl;
} MlirLocationMap;

size_t mlir_location_map_size(const MlirLocationMap *location_map);
size_t mlir_location_map_collect(const MlirLocationMap *location_map, string *out_keys, MlirLocation **out_locs, size_t max);

// Vector helpers used by the parser implementation
DEFINE_VECTOR_FOR_TYPE(MlirOperation*, VecOperation)
DEFINE_VECTOR_FOR_TYPE(MlirValue*, VecValue)
DEFINE_VECTOR_FOR_TYPE(MlirBlock*, VecBlock)

// -----------------------------------------------------------------------------
// Operation kinds
// -----------------------------------------------------------------------------

typedef enum {
    // Core ops
    OP_TYPE_UNREGISTERED = 0,
    OP_TYPE_MODULE,

    // Arithmetic dialect
    OP_TYPE_ARITH_ADDI,
    OP_TYPE_ARITH_SUBI,
    OP_TYPE_ARITH_MULI,
    OP_TYPE_ARITH_DIVI,
    OP_TYPE_ARITH_ADDF,
    OP_TYPE_ARITH_SUBF,
    OP_TYPE_ARITH_MULF,
    OP_TYPE_ARITH_DIVF,
    OP_TYPE_ARITH_CONSTANT,
    OP_TYPE_ARITH_CMPI,
    OP_TYPE_ARITH_CMPF,
    OP_TYPE_ARITH_SELECT,
    OP_TYPE_ARITH_BITCAST,
    OP_TYPE_ARITH_SITOFP,
    OP_TYPE_ARITH_EXTSI,
    OP_TYPE_ARITH_TRUNCI,
    OP_TYPE_ARITH_EXTF,
    OP_TYPE_ARITH_TRUNCF,
    OP_TYPE_ARITH_EXTUI,
    OP_TYPE_ARITH_MAXF,
    OP_TYPE_ARITH_DIVSI,
    OP_TYPE_ARITH_REMSI,
    OP_TYPE_ARITH_ORI,
    OP_TYPE_ARITH_MINSI,
    OP_TYPE_ARITH_ANDI,

    // Math dialect
    OP_TYPE_MATH_EXP,
    OP_TYPE_MATH_LOG,

    // Memory dialect
    OP_TYPE_MEMREF_LOAD,
    OP_TYPE_MEMREF_STORE,
    OP_TYPE_MEMREF_ALLOC,
    OP_TYPE_MEMREF_DEALLOC,

    // Control flow
    OP_TYPE_CF_BR,
    OP_TYPE_CF_COND_BR,
    OP_TYPE_CF_SWITCH,

    // Function dialect
    OP_TYPE_FUNC_FUNC,
    OP_TYPE_FUNC_RETURN,
    OP_TYPE_FUNC_CALL,

    // SCF dialect
    OP_TYPE_SCF_FOR,
    OP_TYPE_SCF_WHILE,
    OP_TYPE_SCF_IF,
    OP_TYPE_SCF_YIELD,

    // Triton dialect
    OP_TYPE_TT_GET_PROGRAM_ID,
    OP_TYPE_TT_LOAD,
    OP_TYPE_TT_STORE,
    OP_TYPE_TT_MAKE_RANGE,
    OP_TYPE_TT_SPLAT,
    OP_TYPE_TT_ADDPTR,
    OP_TYPE_TT_RETURN,
    OP_TYPE_TT_FUNC,
    OP_TYPE_TT_CALL,
    OP_TYPE_TT_REDUCE,
    OP_TYPE_TT_BROADCAST,
    OP_TYPE_TT_EXPAND_DIMS,
    OP_TYPE_TT_DOT,
    OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE,

    // GPU dialect
    OP_TYPE_GPU_LAUNCH,

    // Affine dialect
    OP_TYPE_AFFINE_FOR,
    OP_TYPE_AFFINE_LOAD,

    // Vector dialect
    OP_TYPE_VECTOR_PRINT,

    // Standard dialect
    OP_TYPE_STD_CONSTANT,
    OP_TYPE_STD_RETURN,

    // Tensor dialect
    OP_TYPE_TENSOR_EXTRACT,
    OP_TYPE_TENSOR_SPLAT,
    OP_TYPE_TENSOR_COLLAPSE_SHAPE,

    // Linalg dialect
    OP_TYPE_LINALG_FILL,

    // Index dialect
    OP_TYPE_INDEX_CONSTANT,

    // LLVM dialect
    OP_TYPE_LLVM_MLIR_UNDEF,

    // Return operations
    OP_TYPE_RETURN,
    OP_TYPE_TT_REDUCE_RETURN,

    OP_TYPE_COUNT
} OpType;

const char *mlir_op_type_to_string(OpType type);

// -----------------------------------------------------------------------------
// API lifecycle
// -----------------------------------------------------------------------------

void mlir_api_init(MlirOperation *root);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
MlirOperation *mlir_op_create(Arena *arena, OpType type);
void mlir_op_add_region(Arena *arena, MlirOperation *op, MlirRegion *region);
void mlir_op_add_operand(Arena *arena, MlirOperation *op, MlirValue *operand);
void mlir_op_add_result(Arena *arena, MlirOperation *op, MlirValue *result);

// Property setters
void mlir_operation_set_type(MlirOperation *op, OpType type);
void mlir_operation_set_name(MlirOperation *op, const char *name, size_t name_len);
void mlir_operation_set_operands(MlirOperation *op, MlirValue **operands, size_t count);
void mlir_operation_set_results(MlirOperation *op, MlirValue **results, size_t count);
void mlir_operation_set_result_types(MlirOperation *op, MlirType **types, size_t count);
void mlir_operation_set_attributes(MlirOperation *op, MlirAttribute **attrs, size_t count);
void mlir_operation_append_attribute(Arena *arena, MlirOperation *op, MlirAttribute *attr);
void mlir_operation_set_location(MlirOperation *op, MlirLocation *loc);
void mlir_operation_set_trailing_comment(MlirOperation *op, const char *comment, size_t comment_len);
void mlir_operation_set_source_line_start(MlirOperation *op, int64_t line_start);
void mlir_operation_set_unnumbered_loc_def(MlirOperation *op, MlirLocation *loc);

// Accessors
OpType mlir_operation_get_type(const MlirOperation *op);
const char *mlir_operation_get_name(const MlirOperation *op);
string mlir_operation_get_name_string(const MlirOperation *op);
MlirLocation *mlir_operation_get_location(const MlirOperation *op);
string mlir_operation_get_trailing_comment(const MlirOperation *op);
int64_t mlir_operation_get_source_line_start(const MlirOperation *op);
MlirLocation *mlir_operation_get_unnumbered_loc_def(const MlirOperation *op);
size_t mlir_operation_num_operands(const MlirOperation *op);
MlirValue *mlir_operation_get_operand(const MlirOperation *op, size_t idx);
size_t mlir_operation_num_results(const MlirOperation *op);
MlirValue *mlir_operation_get_result(const MlirOperation *op, size_t idx);
size_t mlir_operation_num_result_types(const MlirOperation *op);
MlirType *mlir_operation_get_result_type(const MlirOperation *op, size_t idx);
size_t mlir_operation_num_attributes(const MlirOperation *op);
MlirAttribute *mlir_operation_get_attribute(const MlirOperation *op, size_t idx);
size_t mlir_operation_num_regions(const MlirOperation *op);
MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx);

// -----------------------------------------------------------------------------
// Region API
// -----------------------------------------------------------------------------

MlirRegion *mlir_region_create(Arena *arena);
void mlir_region_add_block(Arena *arena, MlirRegion *region, MlirBlock *block);
size_t mlir_region_num_blocks(const MlirRegion *region);
MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx);

// -----------------------------------------------------------------------------
// Block API
// -----------------------------------------------------------------------------

MlirBlock *mlir_block_create(Arena *arena);
void mlir_block_add_operation(Arena *arena, MlirBlock *block, MlirOperation *op);
void mlir_block_add_argument(Arena *arena, MlirBlock *block, MlirValue *arg);
size_t mlir_block_num_operations(const MlirBlock *block);
MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx);
size_t mlir_block_num_arguments(const MlirBlock *block);
MlirValue *mlir_block_get_argument(const MlirBlock *block, size_t idx);

// -----------------------------------------------------------------------------
// Value API
// -----------------------------------------------------------------------------

#define MLIR_VALUE_BLOCK_ARG 0
#define MLIR_VALUE_OP_RESULT 1

// Creation & mutation
MlirValue *mlir_value_create(Arena *arena, int value_kind);
void mlir_value_set_def(MlirValue *value, void *def);
void mlir_value_set_type(MlirValue *value, MlirType *type);
void mlir_value_set_register_name(MlirValue *value, const char *name, size_t name_len);
void mlir_value_set_result_index(MlirValue *value, uint32_t index);
void mlir_value_set_location(MlirValue *value, MlirLocation *loc);
void mlir_value_set_divisibility(MlirValue *value, bool has_value, int64_t div_value, MlirType *type);
void mlir_value_set_max_divisibility(MlirValue *value, bool has_value, int64_t div_value, MlirType *type);

// Accessors
int mlir_value_get_kind(const MlirValue *value);
MlirType *mlir_value_get_type(const MlirValue *value);
string mlir_value_get_register_name(const MlirValue *value);
uint32_t mlir_value_get_result_index(const MlirValue *value);
MlirOperation *mlir_value_get_def_op(const MlirValue *value);
MlirLocation *mlir_value_get_location(const MlirValue *value);
bool mlir_value_has_divisibility(const MlirValue *value);
int64_t mlir_value_get_divisibility_value(const MlirValue *value);
MlirType *mlir_value_get_divisibility_type(const MlirValue *value);
bool mlir_value_has_max_divisibility(const MlirValue *value);
int64_t mlir_value_get_max_divisibility_value(const MlirValue *value);
MlirType *mlir_value_get_max_divisibility_type(const MlirValue *value);

// -----------------------------------------------------------------------------
// Type API
// -----------------------------------------------------------------------------

// Creation
MlirType *mlir_type_create_integer(Arena *arena, uint32_t width, bool is_signed);
MlirType *mlir_type_create_float(Arena *arena, uint32_t width, bool is_bfloat);
MlirType *mlir_type_create_index(Arena *arena);
MlirType *mlir_type_create_unknown(Arena *arena);
MlirType *mlir_type_create_tensor(Arena *arena, const int64_t *shape, size_t rank, MlirType *element_type);
MlirType *mlir_type_create_memref(Arena *arena, const int64_t *shape, size_t rank, MlirType *element_type);
MlirType *mlir_type_create_pointer(Arena *arena, MlirType *element_type, bool has_address_space, uint32_t address_space);
MlirType *mlir_type_create_opaque(Arena *arena, string name);
MlirType *mlir_type_create_from_string(Arena *arena, string type_str);

// Mutation
void mlir_type_set_integer_properties(MlirType *type, uint32_t width, bool is_signed);
void mlir_type_set_float_properties(MlirType *type, uint32_t width, bool is_bfloat);
void mlir_type_set_tensor_properties(MlirType *type, const int64_t *shape, size_t rank, MlirType *element_type);
void mlir_type_set_memref_properties(MlirType *type, const int64_t *shape, size_t rank, MlirType *element_type);
void mlir_type_set_pointer_properties(MlirType *type, MlirType *element_type, bool has_address_space, uint32_t address_space);

// Introspection & formatting
bool mlir_type_is_integer(const MlirType *type);
bool mlir_type_is_float(const MlirType *type);
bool mlir_type_is_tensor(const MlirType *type);
bool mlir_type_is_memref(const MlirType *type);
bool mlir_type_is_pointer(const MlirType *type);
bool mlir_type_is_index(const MlirType *type);
bool mlir_type_is_unknown(const MlirType *type);
bool mlir_type_is_opaque(const MlirType *type);
string mlir_type_to_string(Arena *arena, MlirType *type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
MlirAttribute *mlir_attribute_create_integer(Arena *arena, int64_t value);
MlirAttribute *mlir_attribute_create_float(Arena *arena, double value);
MlirAttribute *mlir_attribute_create_bool(Arena *arena, bool value);
MlirAttribute *mlir_attribute_create_string(Arena *arena, const char *str, size_t len);
void mlir_attribute_set_name(MlirAttribute *attr, const char *name, size_t name_len);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT
} MlirAttrKind;

MlirAttrKind mlir_attribute_get_kind(const MlirAttribute *attr);
string mlir_attribute_get_name(const MlirAttribute *attr);
int64_t mlir_attribute_get_integer(const MlirAttribute *attr);
double mlir_attribute_get_float(const MlirAttribute *attr);
bool mlir_attribute_get_bool(const MlirAttribute *attr);
string mlir_attribute_get_string(const MlirAttribute *attr);

// -----------------------------------------------------------------------------
// Location API
// -----------------------------------------------------------------------------

typedef enum {
    MLIR_LOC_UNKNOWN,
    MLIR_LOC_FILE,
    MLIR_LOC_NAME,
    MLIR_LOC_CALLSITE,
    MLIR_LOC_FUSED,
    MLIR_LOC_REF
} MlirLocationKind;

// Creation & mutation
MlirLocation *mlir_location_create(Arena *arena);
void mlir_location_set_kind(MlirLocation *loc, MlirLocationKind kind);
void mlir_location_set_original_text(MlirLocation *loc, string text);
void mlir_location_set_file_data(MlirLocation *loc, string filename, int line, int column);
void mlir_location_set_name_data(MlirLocation *loc, string name);
void mlir_location_set_ref_id(MlirLocation *loc, int ref_id);

// Accessors
MlirLocationKind mlir_location_get_kind(const MlirLocation *loc);
string mlir_location_get_original_text(const MlirLocation *loc);
string mlir_location_get_file_filename(const MlirLocation *loc);
int mlir_location_get_file_line(const MlirLocation *loc);
int mlir_location_get_file_column(const MlirLocation *loc);
string mlir_location_get_name(const MlirLocation *loc);
int mlir_location_get_ref_id(const MlirLocation *loc);

#ifdef __cplusplus
}
#endif

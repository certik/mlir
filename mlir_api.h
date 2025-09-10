// Public MLIR C API declarations without exposing internal data structures.
// The actual implementation is provided separately and can be backed by
// different underlying MLIR representations.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of opaque structures
struct Arena;

typedef struct MlirOperation MlirOperation;
typedef struct MlirRegion MlirRegion;
typedef struct MlirBlock MlirBlock;
typedef struct MlirValue MlirValue;
typedef struct MlirType MlirType;
typedef struct MlirAttribute MlirAttribute;
typedef struct MlirLocation MlirLocation;

// Operation type enumeration.
typedef enum {
    // Core ops
    OP_TYPE_UNREGISTERED = 0,  // For dynamic/unregistered operations
    OP_TYPE_MODULE,            // Module operation

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

    OP_TYPE_COUNT  // Total number of operation types
} OpType;

// Initialization for implementations that require setup. When used with an
// upstream MLIR implementation, root should point to the top-level
// operation (typically a module) so that the implementation can pre-compute
// any necessary lookup tables.
void mlir_api_init(MlirOperation *root);

// Construction helpers
MlirOperation *mlir_op_create(struct Arena *arena, OpType type);
void mlir_op_add_region(struct Arena *arena, MlirOperation *op, MlirRegion *region);
void mlir_op_add_operand(struct Arena *arena, MlirOperation *op, MlirValue *operand);
void mlir_op_add_result(struct Arena *arena, MlirOperation *op, MlirValue *result);
void mlir_block_add_operation(struct Arena *arena, MlirBlock *block, MlirOperation *op);
void mlir_block_add_argument(struct Arena *arena, MlirBlock *block, MlirValue *arg);
void mlir_region_add_block(struct Arena *arena, MlirRegion *region, MlirBlock *block);

// Traversal helpers
size_t mlir_region_num_blocks(const MlirRegion *region);
MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx);
size_t mlir_block_num_operations(const MlirBlock *block);
MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx);
OpType mlir_operation_get_type(const MlirOperation *op);
size_t mlir_operation_num_regions(const MlirOperation *op);
MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx);

// Additional accessors for printer support
size_t mlir_operation_num_attributes(const MlirOperation *op);
MlirAttribute *mlir_operation_get_attribute(const MlirOperation *op, size_t idx);
size_t mlir_operation_num_operands(const MlirOperation *op);
MlirValue *mlir_operation_get_operand(const MlirOperation *op, size_t idx);
size_t mlir_operation_num_results(const MlirOperation *op);
MlirValue *mlir_operation_get_result(const MlirOperation *op, size_t idx);
MlirLocation *mlir_operation_get_location(const MlirOperation *op);
const char *mlir_operation_get_name(const MlirOperation *op);

size_t mlir_block_num_arguments(const MlirBlock *block);
MlirValue *mlir_block_get_argument(const MlirBlock *block, size_t idx);

#ifdef __cplusplus
}
#endif

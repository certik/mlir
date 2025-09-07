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
#ifndef MLIR_API_TYPES_DEFINED
typedef struct MlirOperation MlirOperation;
typedef struct MlirRegion MlirRegion;
typedef struct MlirBlock MlirBlock;
typedef struct MlirValue MlirValue;
#define MLIR_API_TYPES_DEFINED 1
#endif

// Operation type enumeration. This mirrors the internal OpType enum and must
// remain in sync with any implementation.
#ifndef MLIR_OP_TYPE_DEFINED
#define MLIR_OP_TYPE_DEFINED 1
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

    // Return operations
    OP_TYPE_RETURN,
    OP_TYPE_TT_REDUCE_RETURN,

    OP_TYPE_COUNT
} OpType;
#endif

// Initialization for implementations that require setup. When used with an
// upstream MLIR implementation, \`root\` should point to the top-level
// operation (typically a module) so that the implementation can pre-compute
// any necessary lookup tables.
static inline void mlir_api_init(MlirOperation *root);

// Construction helpers
static inline MlirOperation *mlir_op_create(Arena *arena, OpType type);
static inline void mlir_op_add_region(Arena *arena, MlirOperation *op, MlirRegion *region);
static inline void mlir_op_add_operand(Arena *arena, MlirOperation *op, MlirValue *operand);
static inline void mlir_op_add_result(Arena *arena, MlirOperation *op, MlirValue *result);
static inline void mlir_block_add_operation(Arena *arena, MlirBlock *block, MlirOperation *op);
static inline void mlir_block_add_argument(Arena *arena, MlirBlock *block, MlirValue *arg);
static inline void mlir_region_add_block(Arena *arena, MlirRegion *region, MlirBlock *block);

// Traversal helpers
static inline size_t mlir_region_num_blocks(const MlirRegion *region);
static inline MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx);
static inline size_t mlir_block_num_operations(const MlirBlock *block);
static inline MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx);
static inline OpType mlir_operation_get_type(const MlirOperation *op);
static inline size_t mlir_operation_num_regions(const MlirOperation *op);
static inline MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx);

#ifdef __cplusplus
}
#endif



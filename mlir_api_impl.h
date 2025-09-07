#pragma once

// Map API type names to our internal structures via macros.
#define MlirOperation Operation
#define MlirRegion Region
#define MlirBlock Block
#define MlirValue ValueRef
#define MlirAttribute Attribute
#define MlirType Type
#define MlirLocation Location

#define MLIR_API_TYPES_DEFINED 1
#include "mlir_parser.h"
#include "mlir_api.h"
#include <string.h>

static inline void mlir_api_init(MlirOperation *root) {
    (void)root;
    // No initialization required for the native C implementation.
}

static inline MlirOperation *mlir_op_create(Arena *arena, OpType type) {
    MlirOperation *op = arena_alloc(arena, MlirOperation);
    *op = (MlirOperation){0};
    op->op_type = type;
    return op;
}

static inline void mlir_op_add_region(Arena *arena, MlirOperation *op, MlirRegion *region) {
    MlirRegion **new_regions = arena_alloc_array(arena, MlirRegion*, op->n_regions + 1);
    if (op->regions) memcpy(new_regions, op->regions, op->n_regions * sizeof(MlirRegion*));
    new_regions[op->n_regions] = region;
    op->regions = new_regions;
    op->n_regions++;
}

static inline void mlir_op_add_operand(Arena *arena, MlirOperation *op, MlirValue *operand) {
    MlirValue **new_operands = arena_alloc_array(arena, MlirValue*, op->n_operands + 1);
    if (op->operands) memcpy(new_operands, op->operands, op->n_operands * sizeof(MlirValue*));
    new_operands[op->n_operands] = operand;
    op->operands = new_operands;
    op->n_operands++;
}

static inline void mlir_op_add_result(Arena *arena, MlirOperation *op, MlirValue *result) {
    MlirValue **new_results = arena_alloc_array(arena, MlirValue*, op->n_results + 1);
    if (op->results) memcpy(new_results, op->results, op->n_results * sizeof(MlirValue*));
    new_results[op->n_results] = result;
    op->results = new_results;
    op->n_results++;
}

static inline void mlir_block_add_operation(Arena *arena, MlirBlock *block, MlirOperation *op) {
    MlirOperation **new_ops = arena_alloc_array(arena, MlirOperation*, block->n_operations + 1);
    if (block->operations) memcpy(new_ops, block->operations, block->n_operations * sizeof(MlirOperation*));
    new_ops[block->n_operations] = op;
    block->operations = new_ops;
    block->n_operations++;
}

static inline void mlir_block_add_argument(Arena *arena, MlirBlock *block, MlirValue *arg) {
    MlirValue **new_args = arena_alloc_array(arena, MlirValue*, block->n_arguments + 1);
    if (block->arguments) memcpy(new_args, block->arguments, block->n_arguments * sizeof(MlirValue*));
    new_args[block->n_arguments] = arg;
    block->arguments = new_args;
    block->n_arguments++;
}

static inline void mlir_region_add_block(Arena *arena, MlirRegion *region, MlirBlock *block) {
    MlirBlock **new_blocks = arena_alloc_array(arena, MlirBlock*, region->n_blocks + 1);
    if (region->blocks) memcpy(new_blocks, region->blocks, region->n_blocks * sizeof(MlirBlock*));
    new_blocks[region->n_blocks] = block;
    region->blocks = new_blocks;
    region->n_blocks++;
}

static inline size_t mlir_region_num_blocks(const MlirRegion *region) {
    return region->n_blocks;
}

static inline MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx) {
    return region->blocks[idx];
}

static inline size_t mlir_block_num_operations(const MlirBlock *block) {
    return block->n_operations;
}

static inline MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx) {
    return block->operations[idx];
}

static inline OpType mlir_operation_get_type(const MlirOperation *op) {
    return op->op_type;
}

static inline size_t mlir_operation_num_regions(const MlirOperation *op) {
    return op->n_regions;
}

static inline MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx) {
    return op->regions[idx];
}


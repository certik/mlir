#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

typedef struct MlirLocationMap MlirLocationMap;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Classic MLIR printer that produces upstream MLIR textual format.
 *
 * This printer handles all MLIR constructs with classic formatting conventions:
 * - Inline attributes (arith.constant 42 : i32)
 * - No empty parentheses
 * - Block references for branches
 * - Operation-specific formatting
 */

/**
 * Print an MLIR operation and its nested content to string using classic format.
 *
 * @param arena Memory arena for allocating the result string
 * @param indent_level Base indentation level (0 = no indent, 1 = 4 spaces, etc.)
 * @param op Operation to print
 * @return String containing the printed MLIR representation
 */
string print_operation_classic(Arena *arena, int indent_level, MlirOperation *op);

/**
 * Print an MLIR region to string using classic format.
 *
 * @param arena Memory arena for allocating the result string
 * @param indent_level Base indentation level
 * @param region Region to print
 * @return String containing the printed region with braces and blocks
 */
string print_region_classic(Arena *arena, int indent_level, MlirRegion *region);

/**
 * Print an MLIR basic block to string using classic format.
 *
 * @param arena Memory arena for allocating the result string
 * @param bb_index Block index (for ^bb0, ^bb1, etc.)
 * @param indent_level Base indentation level
 * @param block Block to print
 * @return String containing the printed block with label and operations
 */
string print_block_classic(Arena *arena, int bb_index, int indent_level, MlirBlock *block);

/**
 * Print a complete MLIR module with location map definitions.
 *
 * @param arena Memory arena for allocating the result string
 * @param module Module operation to print
 * @param location_map Location map containing #locN definitions
 * @return String containing the complete printed module with location definitions
 */
string print_module_classic(Arena *arena, MlirOperation *module, MlirLocationMap *location_map);

#ifdef __cplusplus
}
#endif

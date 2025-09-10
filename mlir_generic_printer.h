#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

// Forward declarations for internal concrete types used by printers
typedef struct Operation Operation;
typedef struct Region Region;
typedef struct Block Block;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Generic MLIR printer that produces standard MLIR textual format.
 * 
 * This printer handles all MLIR constructs including operations, regions, blocks,
 * SSA values, types, attributes, and maintains proper indentation and SSA numbering.
 */

/**
 * Print an MLIR operation and its nested content to string.
 * 
 * @param arena Memory arena for allocating the result string
 * @param indent_level Base indentation level (0 = no indent, 1 = 4 spaces, etc.)
 * @param op Operation to print
 * @return String containing the printed MLIR representation
 */
string print_operation_generic(Arena *arena, int indent_level, Operation *op);

/**
 * Print an MLIR region to string.
 * 
 * @param arena Memory arena for allocating the result string
 * @param indent_level Base indentation level
 * @param region Region to print
 * @return String containing the printed region with braces and blocks
 */
string print_region_generic(Arena *arena, int indent_level, Region *region);

/**
 * Print an MLIR basic block to string.
 * 
 * @param arena Memory arena for allocating the result string
 * @param bb_index Block index (for ^bb0, ^bb1, etc.)
 * @param indent_level Base indentation level
 * @param block Block to print
 * @return String containing the printed block with label and operations
 */
string print_block_generic(Arena *arena, int bb_index, int indent_level, Block *block);

#ifdef __cplusplus
}
#endif

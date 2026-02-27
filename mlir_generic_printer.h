#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Generic MLIR printer that produces standard MLIR textual format.
 */

string print_operation_generic(MLIR_Context *ctx, int indent_level, MLIR_OpHandle op);
string print_region_generic(MLIR_Context *ctx, int indent_level, MLIR_RegionHandle region);
string print_block_generic(MLIR_Context *ctx, int bb_index, int indent_level, MLIR_BlockHandle block);

#ifdef __cplusplus
}
#endif

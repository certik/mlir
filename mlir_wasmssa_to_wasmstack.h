// Stage 2 of the native LLVM-dialect MLIR -> WASM pipeline.
//
//   wasmssa builtin.module  -->  wasmstack builtin.module
//
// Linearises SSA results onto the wasm value stack and inserts
// block/loop/if/end markers.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns MLIR_INVALID_HANDLE on failure.
MLIR_OpHandle mlir_wasmssa_to_wasmstack(MLIR_Context *ctx,
                                        MLIR_OpHandle ssa_module);

#ifdef __cplusplus
}
#endif

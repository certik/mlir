// Stage 3 of the native LLVM-dialect MLIR -> WASM pipeline.
//
//   wasmstack builtin.module  -->  wasm32 .wasm.o relocatable object bytes
//
// Emits type/import/function/table/memory/global/export/element/code/
// data sections plus relocation metadata consumed by `wasm-ld`.

#pragma once

#include <base/string.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the wasm32 relocatable object bytes (allocated in `ctx`'s
// arena) or an empty string on failure.
string mlir_wasmstack_to_bin(MLIR_Context *ctx, MLIR_OpHandle stk_module);

#ifdef __cplusplus
}
#endif

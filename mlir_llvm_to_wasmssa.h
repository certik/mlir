// Stage 1 of the native LLVM-dialect MLIR -> WASM pipeline.
//
//   LLVM-dialect builtin.module  -->  wasmssa builtin.module
//
// Also declares the WebAssembly value-type byte encodings shared by
// all three pipeline stages.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// WebAssembly value-type byte encodings.
enum {
    WT_I32 = 0x7f,
    WT_I64 = 0x7e,
    WT_F32 = 0x7d,
    WT_F64 = 0x7c,
};

// Returns MLIR_INVALID_HANDLE on failure (a diagnostic is printed to
// stderr).
MLIR_OpHandle mlir_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif

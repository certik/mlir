// Public interface for the three-stage native LLVM-dialect MLIR -> WASM
// pipeline. Each pipeline stage takes and/or returns an MLIR_OpHandle
// (a `builtin.module` containing unregistered wasmssa.* / wasmstack.*
// ops). Internal stages may use private working representations but
// must not leak them through this header.
//
//   stage 1 (mlir_lower_llvm_to_wasmssa.c):
//                 LLVM-dialect builtin.module  -->  wasmssa  builtin.module
//   stage 2 (mlir_stackify_wasmssa.c):
//                 wasmssa     builtin.module  -->  wasmstack builtin.module
//   stage 3 (mlir_translate_wasmstack_to_binary.c):
//                 wasmstack   builtin.module  -->  wasm32 .wasm.o bytes

#pragma once

#include <base/string.h>

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

// Stage 1. Returns MLIR_INVALID_HANDLE on failure (a diagnostic is
// printed to stderr).
MLIR_OpHandle mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx,
                                         MLIR_OpHandle module);

// Stage 2. Returns MLIR_INVALID_HANDLE on failure.
MLIR_OpHandle mlir_stackify_wasmssa(MLIR_Context *ctx,
                                    MLIR_OpHandle ssa_module);

// Stage 3. Returns the wasm32 relocatable object bytes (allocated in
// `ctx`'s arena) or an empty string on failure.
string mlir_translate_wasmstack_to_binary(MLIR_Context *ctx,
                                          MLIR_OpHandle stk_module);

// Disassemble wasm32 binary bytes (e.g. stage 3 output) to a WAT-like
// human-readable text form. Lives in mlir_wasm_binary_to_wat.c.
string mlir_wasm_binary_to_wat(MLIR_Context *ctx, string bin);

#ifdef __cplusplus
}
#endif

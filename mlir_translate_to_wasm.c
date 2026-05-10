// Public entry point for the native LLVM-dialect MLIR -> WASM translator.
//
// This file is the thin façade that chains the three stages of the
// pipeline; the heavy lifting lives in:
//
//   mlir_lower_llvm_to_wasmssa.c          (stage 1: LLVM -> wasmssa)
//   mlir_stackify_wasmssa.c               (stage 2: wasmssa -> wasmstack)
//   mlir_translate_wasmstack_to_binary.c  (stage 3: wasmstack -> .wasm.o)
//
// The shared C-struct intermediate-form definitions live in
// mlir_wasm_dialect.h / mlir_wasm_dialect.c.

#include <stddef.h>
#include <stdio.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_wasm_dialect.h"

string mlir_translate_to_wasm_native(MLIR_Context *ctx, MLIR_OpHandle module);

string mlir_translate_to_wasm_native(MLIR_Context *ctx, MLIR_OpHandle module) {
    string fail = {0};

    wasmssa_module_t *ssa = mlir_lower_llvm_to_wasmssa(ctx, module);
    if (!ssa) return fail;

    wasmstack_module_t *stk = mlir_stackify_wasmssa(ssa);
    wasmssa_module_free(ssa);
    if (!stk) return fail;

    string r = mlir_translate_wasmstack_to_binary(ctx, stk);
    wasmstack_module_free(stk);
    return r;
}

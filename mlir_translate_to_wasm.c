// Public entry point for the native LLVM-dialect MLIR -> WASM translator.
// Thin façade chaining the three stages of the pipeline.

#include <stddef.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_wasm_pipeline.h"

string mlir_translate_to_wasm_native(MLIR_Context *ctx, MLIR_OpHandle module);

string mlir_translate_to_wasm_native(MLIR_Context *ctx, MLIR_OpHandle module) {
    string fail = {0};
    MLIR_OpHandle ssa = mlir_lower_llvm_to_wasmssa(ctx, module);
    if (!ssa) return fail;
    MLIR_OpHandle stk = mlir_stackify_wasmssa(ctx, ssa);
    if (!stk) return fail;
    return mlir_translate_wasmstack_to_binary(ctx, stk);
}

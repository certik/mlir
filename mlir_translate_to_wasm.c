// Native implementation of MLIR_TranslateModuleToWasm.
//
// Chains the three in-tree stages
//   llvm.dialect MLIR  ->  mlir_llvm_to_wasmssa   ->
//   wasmssa MLIR        ->  mlir_wasmssa_to_wasmstack  ->
//   wasmstack MLIR      ->  mlir_wasmstack_to_bin  -> wasm32-wasi bytes
//
// Uses only the public mlir_api.h surface (plus the dedicated stage
// headers, which are themselves agnostic). This translation unit is C
// and links into both the upstream-backed and native-backed tinyc
// builds, providing the backend-agnostic
// `MLIR_TranslateModuleToWasm`.

#include <stddef.h>

#include "mlir_api.h"
#include "mlir_llvm_to_wasmssa.h"
#include "mlir_wasmssa_to_wasmstack.h"
#include "mlir_wasmstack_to_bin.h"

string MLIR_TranslateModuleToWasm(MLIR_Context *ctx, MLIR_OpHandle module) {
    string fail = {0};
    MLIR_OpHandle ssa = mlir_llvm_to_wasmssa(ctx, module);
    if (ssa == MLIR_INVALID_HANDLE) return fail;
    MLIR_OpHandle stk = mlir_wasmssa_to_wasmstack(ctx, ssa);
    if (stk == MLIR_INVALID_HANDLE) return fail;
    return mlir_wasmstack_to_bin(ctx, stk);
}

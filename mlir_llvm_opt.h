// LLVM-dialect optimization passes for -O1/-O2 (LICM, etc.).
#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loop-Invariant Code Motion (LICM) on flat `llvm`-dialect CFGs.
// Hoists pure, non-trapping, single-result ops whose operands are all defined
// outside a natural loop into the loop's unique unconditional preheader.
void mlir_llvm_licm(MLIR_Context *ctx, MLIR_OpHandle module);

// Small-function inliner on flat `llvm`-dialect CFGs.
void mlir_llvm_inline(MLIR_Context *ctx, MLIR_OpHandle module);

// Tail-recursion elimination (direct + accumulator ADD/MUL) on flat `llvm` CFGs.
void mlir_llvm_tre(MLIR_Context *ctx, MLIR_OpHandle module);

// Simplify-CFG: thread branches through empty forwarding blocks & remove dead blocks.
void mlir_llvm_simplify_cfg(MLIR_Context *ctx, MLIR_OpHandle module);

// Constant folding, algebraic identities, and constant-condition branch folding.
void mlir_llvm_constfold(MLIR_Context *ctx, MLIR_OpHandle module);

// Exact-trip-count loop unroller (complete unroll <= 8, 4x unroll for trip % 4 == 0).
void mlir_llvm_unroll(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif

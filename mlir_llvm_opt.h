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

#ifdef __cplusplus
}
#endif

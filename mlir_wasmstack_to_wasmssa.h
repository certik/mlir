// wasmstack -> wasmssa lifter.
//
// Reverse of mlir_wasmssa_to_wasmstack.c. Walks each wasmstack func's
// flat opcode sequence, simulates the operand stack and the local
// variable array, and reconstructs SSA values for downstream
// consumption by the wasmssa -> llvm pipeline.
//
// First-light scope (macho_exit): linear functions with no structured
// control flow. Each wasmstack op pops its operands, produces (at
// most) one SSA result, and that result is bound either to a local
// (on local.set/tee) or pushed back on the simulated stack.

#ifndef MLIR_WASMSTACK_TO_WASMSSA_H
#define MLIR_WASMSTACK_TO_WASMSSA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_OpHandle mlir_wasmstack_to_wasmssa(MLIR_Context *ctx,
                                        MLIR_OpHandle stack_module);

#ifdef __cplusplus
}
#endif

#endif

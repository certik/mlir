#ifndef MLIR_LLVM_DCE_H
#define MLIR_LLVM_DCE_H

#include "mlir_api.h"

// Whole-function dead-code elimination on the wasm->llvm->aarch64 (from-wasm)
// path. The tinyC front end emits the *value* of every C expression even in
// statement context (e.g. the `(char)v` result of `d[i] = v;`), which survives
// as a zero-use pure SSA value after mem2reg/CSE/GVN. The backend has no DCE,
// so it assigns each such dead value a frame slot and emits a spill store --
// in hot loops (memcpy/memset/memcmp/strlen and friends) that is a wasted
// store *every iteration*. This pass removes any side-effect-free single-result
// op whose result has no uses, transitively, so the dead computation and its
// backend slot/store disappear at the source.
//
// Honors the pipeline's no_def_use_tracking discipline: it computes its own use
// counts by sweeping every op operand and terminator successor-operand, then
// runs a worklist that erases zero-use pure ops and re-queues their operand
// definitions. Side-effecting ops (store/call/load/terminators/return) are
// never removed. Deterministic, so self-host stays bit-identical.
//
// Disable with TINYC_NO_DCE=1 (A/B aid).
void mlir_llvm_dce(MLIR_Context *ctx, MLIR_OpHandle module);

#endif

// llvm dialect -> aarch64 lowering (Option 1: the unified native backend).
//
// This is the future replacement for the wasmssa -> wmir -> aarch64 path.
// It lowers the in-house `llvm` MLIR dialect directly to the existing
// (physical-register) `aarch64` dialect, which `mlir_aarch64_to_macho.c`
// then encodes. Internally it follows the LLVM GlobalISel / Cranelift
// shape:
//
//     llvm  --select-->  a64ssa (virtual registers, real opcodes,
//                                 operand constraints)
//           --regalloc-> aarch64 (physical registers, spills, frame)
//
// The pipeline is being built up op-by-op (see the session plan). Until a
// construct is supported each stage returns MLIR_INVALID_HANDLE with a
// clear diagnostic on stderr. It is reached only via the opt-in
// `--macho-backend=llvm` driver flag, so it cannot affect the established
// wasm or wmir backends.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lower an `llvm`-dialect module to an `aarch64`-dialect module suitable
// for `mlir_aarch64_to_macho`. Returns MLIR_INVALID_HANDLE on failure or
// on an as-yet-unsupported construct (with diagnostics on stderr).
MLIR_OpHandle mlir_llvm_to_aarch64(MLIR_Context *ctx,
                                   MLIR_OpHandle llvm_module);

#ifdef __cplusplus
}
#endif

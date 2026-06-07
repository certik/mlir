// llvm dialect -> aarch64 lowering (Option 1: the unified native backend).
//
// This is the unified native backend; it replaced the former
// wasmssa -> wmir -> aarch64 path.
// It lowers the in-house `llvm` MLIR dialect directly to the existing
// (physical-register) `aarch64` dialect, which `mlir_aarch64_to_macho.c`
// then encodes. Internally it follows the LLVM GlobalISel / Cranelift
// shape:
//
//     llvm  --select-->  a64ssa (virtual registers, real opcodes,
//                                 operand constraints)
//           --regalloc-> aarch64 (physical registers, spills, frame)
//
// The pipeline is built up op-by-op (see the session plan). Until a
// construct is supported each stage returns MLIR_INVALID_HANDLE with a
// clear diagnostic on stderr. It is the default Mach-O backend for the
// from-wasm path and is also reachable via the `--macho-backend=llvm`
// driver flag for native C input.

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

// ---------------------------------------------------------------------------
// Streaming selection API for the low-memory per-function backend.
//
// The whole-module entry above builds the aarch64 IR for every function at
// once, which (with the trivial spill-everything allocator) balloons peak RSS
// past the 4GB self-host budget. The Mach-O encoder instead drives this API to
// lower ONE function at a time into a throwaway arena (see mlir_llvm_to_macho
// in mlir_aarch64_to_macho.c), keeping peak memory ~ one function's worth of
// aarch64 IR rather than the whole module's.
// ---------------------------------------------------------------------------
typedef struct LlvmSelState LlvmSelState;

// Begin streaming selection over `llvm_module`. Pre-collects module-level
// globals into one native data blob, returned via *out_gblob / *out_gblob_len
// (caller frees *out_gblob). Returns a heap state (free with mlir_llvm_sel_end)
// or NULL on failure (with a diagnostic on stderr).
LlvmSelState *mlir_llvm_sel_begin(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                                  uint8_t **out_gblob, uint32_t *out_gblob_len);

// Number of defined functions to lower (declarations are excluded).
size_t mlir_llvm_sel_num_funcs(LlvmSelState *st);

// Build the synthetic `_start` aarch64 func into the CURRENT ctx arena.
MLIR_OpHandle mlir_llvm_sel_synth_start(MLIR_Context *ctx, LlvmSelState *st);

// Select defined function `idx` into the CURRENT ctx arena. Returns
// MLIR_INVALID_HANDLE on an unsupported construct.
MLIR_OpHandle mlir_llvm_sel_func(MLIR_Context *ctx, LlvmSelState *st, size_t idx);

// True iff a defined `main` was seen (required for a valid program).
bool mlir_llvm_sel_saw_main(LlvmSelState *st);

void mlir_llvm_sel_end(LlvmSelState *st);

// Stream an `llvm`-dialect module straight to a Mach-O image, lowering one
// function at a time to bound peak memory. Equivalent in output to
// mlir_llvm_to_aarch64 followed by mlir_aarch64_to_macho, but with a far
// smaller peak RSS. Returns false on failure (diagnostics on stderr).
bool mlir_llvm_to_macho(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                        uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

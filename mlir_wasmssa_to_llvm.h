// Stage 2 of the unified LLVM-dialect MLIR -> Mach-O pipeline for
// WASM-sourced input:
//
//   wasmssa builtin.module  -->  llvm builtin.module
//
// This is the WASM-input counterpart to the C-frontend `emit.c`, which
// produces the in-house `llvm` dialect directly. Lifting WASM to the same
// `llvm` dialect lets the single unified backend (`mlir_llvm_to_aarch64.c`)
// serve BOTH the C-frontend native path and the `--from-wasm` self-host
// path. This is what replaced the separate `wmir` backend.
//
// Unlike the C frontend (which emits structured control flow in a single
// block per function), the lifted WASM is a flattened multi-block CFG using
// `llvm.br` / `llvm.cond_br`. Linear memory is modelled as i32 byte offsets
// into a single host-mapped region; only at `load`/`store` boundaries is a
// real host pointer formed as `linmem_base + zext(i32 addr)`.
//
// On unsupported wasmssa ops the lowering returns
// MLIR_INVALID_HANDLE and prints a diagnostic to stderr identifying the op.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// The WASI shims (fd_write/path_open/args_*/environ_*) are not synthesised
// here; a C adapter (corec/wasm/wasi_adapter.c) is spliced into the module by
// the driver (--wasi-adapter) to provide them.
MLIR_OpHandle mlir_wasmssa_to_llvm(MLIR_Context *ctx,
                                   MLIR_OpHandle ssa_module);

#ifdef __cplusplus
}
#endif

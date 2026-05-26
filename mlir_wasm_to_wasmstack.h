// wasm bytecode -> wasmstack builtin.module (MLIR).
//
// First-light scaffold: parses a linked wasm32 module produced by
// wasm-ld and emits a wasmstack MLIR module compatible with the
// existing wasmstack -> wasmssa -> wmir -> aarch64 -> macho pipeline.
//
// Function naming convention:
//   * Imports get `<module>_<name>` (e.g. "wasi_snapshot_preview1_proc_exit").
//     The recognised WASI imports use just their unprefixed name (e.g.
//     "proc_exit", "fd_write") so the wmir synth helpers can match them.
//   * The function exported as "_start" is renamed to "wasi_start" so
//     that the wmir backend's synth_start does not collide with it.
//   * Other defined functions: "func_<idx>" where <idx> is the linked
//     wasm function index. Callers reference targets by that name.
//
// Globals / data / tables: only handled in later iterations as tests
// require them.

#ifndef MLIR_WASM_TO_WASMSTACK_H
#define MLIR_WASM_TO_WASMSTACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parse `wasm_bytes` and emit a wasmstack `builtin.module` op into `ctx`.
// Returns MLIR_INVALID_HANDLE on error (and prints a diagnostic to
// stderr).
MLIR_OpHandle mlir_wasm_to_wasmstack(MLIR_Context *ctx,
                                     const uint8_t *wasm_bytes,
                                     size_t wasm_size);

#ifdef __cplusplus
}
#endif

#endif

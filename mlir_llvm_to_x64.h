// llvm dialect (flat CFG, LP64) -> static x86_64 ELF executable.
// Linux-native code-generation backend; see mlir_llvm_to_x64.c.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lower a flat `llvm`-dialect module (produced by MLIR_LowerToLLVMDialect) to a
// statically-linked x86_64 ELF executable. On success `*out_data`/`*out_size`
// receive a malloc'd image the caller owns. Returns false on failure or on an
// as-yet-unsupported construct (diagnostics on stderr).
//
// `wasm_lifted` says where the module came from, and selects the crt0:
//   false - a native module (tinyC source compiled straight to LP64). _start
//           calls platform_init(argc, argv, envp) then `main`.
//   true  - a module lifted from wasm (`--from-wasm`, mlir_wasmssa_to_llvm) with
//           the host platform/WASI helpers spliced in. _start maps the wasm
//           linear memory, records argc/argv in __wasm_argc/__wasm_argv, calls
//           __host_platform_init and then app_main (or the lifted wasm entry).
// The caller must say which; the backend does not guess from symbol names.
bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      bool wasm_lifted,
                      uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

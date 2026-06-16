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
bool mlir_llvm_to_elf(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                      uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

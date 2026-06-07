// aarch64 (MLIR dialect) -> Mach-O ARM64 binary translator.
//
// Pairs with mlir_llvm_to_aarch64.{h,c}: that pass produces an MLIR
// module whose top-level ops are `aarch64.func`, each containing a
// flat sequence of aarch64 instruction ops (1:1 with the binary
// encoding). This file is the "dumb" byte emitter that turns that
// module into a runnable Mach-O ARM64 executable.
//
// First-light scope: minimal Mach-O envelope (no __DATA, no libSystem
// stubs, no GOT) — just enough for `int main() { return 42; }` to
// run and exit with code 42 via a direct `svc #0x80` syscall.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Translate an MLIR module whose body is aarch64.* ops into a Mach-O
// ARM64 executable. On success, `*out_data` / `*out_size` receive a
// freshly-malloc'd buffer that the caller owns and must `free()`.
// Returns true on success; on failure prints diagnostics to stderr.
bool mlir_aarch64_to_macho(MLIR_Context *ctx, MLIR_OpHandle aarch64_module,
                           uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

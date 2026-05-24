// WASM (linked) -> Mach-O ARM64 native binary translator. Backs
// `tinyc --emit=macho`.
//
// Takes the bytes of a fully-linked WebAssembly module (the output of
// `MLIR_WasmLink`) and produces a Mach-O ARM64 executable. The
// translation is a single pass: WASM stack ops are emitted as ARM64
// instructions that use the CPU stack as the value stack — no
// register allocation, no SSA, no analysis. Linear memory and
// `__stack_pointer` get host-side backing stores; the two WASI
// imports `proc_exit` / `fd_write` are bridged to libSystem
// `_exit` / `_write` by synthesised shim functions.
//
// This file compiles on every platform; the binary it produces is
// only runnable on Apple Silicon (CPU_TYPE_ARM64). Tests gate by
// platform.
//
// On the C++ reference side, `test_wasm/write_macho.cpp` produces a
// byte-identical baseline binary that we use to validate the
// envelope.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Translate a fully-linked WASM module to a Mach-O ARM64 executable.
//
// `wasm_bytes` is the raw byte buffer of the linked module. On
// success the freshly-malloc'd output bytes are returned in
// `*out_data` / `*out_size`; caller owns and must `free()` the
// buffer. Diagnostics go to stderr. Returns true on success.
bool MLIR_WasmToMachoArm64(const uint8_t *wasm_bytes, size_t wasm_size,
                           uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

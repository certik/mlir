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

// Code generation only, for one translation unit: run the x86_64 code
// generator on EVERY defined function (no entry point needed, no reachability
// trimming, references to undefined symbols allowed) and return the
// concatenated raw machine code of all functions (malloc'd; no ELF container,
// no crt0/thunks, cross-function call and data relocations left unpatched).
// This is the per-TU cost an object-file (`-c`) compile would pay minus the
// object writer, which this backend does not have yet; it is used to benchmark
// codegen file by file and to inspect code with
// `objdump -D -b binary -m i386:x86-64`.
bool mlir_llvm_to_x64_code(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                           uint8_t **out_data, size_t *out_size);

// Lower a flat `llvm`-dialect module to an `x64.*` target-dialect MLIR module
// (for `--emit=x64-mlir` inspection and testing). Returns MLIR_INVALID_HANDLE
// on failure.
MLIR_OpHandle mlir_llvm_to_x64_mlir(MLIR_Context *ctx, MLIR_OpHandle llvm_module);

// Optional phase timing (`tinyc --time-stages`). While `t` is non-NULL, every
// mlir_llvm_to_elf / mlir_llvm_to_x64_code call ADDS the wall time (ns) of its
// phases into *t. NULL (the default) disables timing; the only cost is then a
// pointer test per function.
typedef struct MLIR_X64StageTimes {
    uint64_t setup_ns;        // data section, entry, reachability, crt0
    uint64_t regalloc_ns;     // liveness + linear-scan register allocation
    uint64_t isel_encode_ns;  // frame layout + instruction selection, which
                              // encodes the bytes directly (one fused phase)
    uint64_t link_ns;         // thunks + ELF layout / relocation (container)
} MLIR_X64StageTimes;
void mlir_llvm_to_x64_set_stage_times(MLIR_X64StageTimes *t);

// Optimization level (0 = -O0 fast block-local RA, 1+ = -O1/-O2 global RA + opts).
void mlir_llvm_to_x64_set_opt_level(int level);

#ifdef __cplusplus
}
#endif

// In-tree minimal wasm linker. Replaces the host `wasm-ld` invocation
// used by `examples/tinyc/run_tinyc_tests.py` to combine a tinyc-emitted
// relocatable wasm32 object with `runtime_wasm.o` and `start_wasm.o`
// into a single executable wasm module.
//
// Scope: handles exactly the slice of the wasm linking spec that the
// tinyc pipeline exercises today — TYPE / IMPORT / FUNCTION / TABLE /
// MEMORY / GLOBAL / EXPORT / ELEMENT / CODE / DATA sections, plus the
// custom sections `linking`, `reloc.CODE`, and `reloc.DATA`. Other
// custom sections (e.g. `name`, DWARF) are intentionally stripped from
// the output.
//
// API (memory model): caller passes the raw bytes of each input object
// and receives the linked module as a freshly-malloc'd buffer in
// `*out_data` / `*out_size`. Caller owns the result and must `free()`
// it.
//
// Diagnostics (errors) are written to stderr-equivalent via the corec
// `write_all` primitive so this file links cleanly in both the native
// build (libc-backed) and the wasm32-wasi build (corec-stdlib-backed).

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t *data;
    size_t         size;
    const char    *name; // for diagnostics only
} MLIR_WasmLinkInput;

// Link N wasm relocatable objects into a single executable wasm module.
// `entry_export` is the name to add to the Export section pointing at
// the resolved symbol of the same name (e.g. "_start"). Returns true on
// success, false on error. On success, *out_data is malloc'd; caller
// frees it.
bool MLIR_WasmLink(const MLIR_WasmLinkInput *inputs, size_t n_inputs,
                   const char *entry_export,
                   uint8_t **out_data, size_t *out_size);

#ifdef __cplusplus
}
#endif

// Disassemble wasm32 binary bytes (e.g. stage 3 output) to a WAT-like
// human-readable text form. Backs `tinyc --emit=wat`.

#pragma once

#include <base/string.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates the result string in `ctx`'s arena.
string mlir_wasm_binary_to_wat(MLIR_Context *ctx, string bin);

#ifdef __cplusplus
}
#endif

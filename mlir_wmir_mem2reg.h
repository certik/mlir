// wmir mem2reg: promote wasm locals (wmir.local_get / wmir.local_set) into
// SSA values + block-argument phis, using Braun et al. (2013) simple SSA
// construction over the already-built wmir CFG.
//
// Wasm locals are never address-taken, so every local is promotable. After
// this pass the region contains no wmir.local_get / wmir.local_set ops; reads
// resolve directly to the defining SSA value (or a block-arg phi at merges),
// which the register allocator then keeps in registers instead of emitting an
// unconditional ldr/str to the stack frame for every access.
//
// Operates on a single wmir.func body region. `lt_hex` is the func's
// `local_types` attribute (hex-encoded valtype byte per local, params first);
// nlocals = lt_hex_len / 2. Returns true on success (region mutated in place).
#ifndef MLIR_WMIR_MEM2REG_H
#define MLIR_WMIR_MEM2REG_H

#include "mlir_api.h"

bool mlir_wmir_mem2reg(MLIR_Context *ctx, MLIR_RegionHandle region,
                       const char *lt_hex, size_t lt_hex_len);

#endif

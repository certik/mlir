// LLVM-dialect load CSE / value numbering (EarlyCSE-style).
//
// Runs after mem2reg on the flat `cf`/`llvm` CFG produced by the
// wasm -> wasmssa -> llvm lifter. It performs a dominator-tree walk with a
// scoped value-number table that eliminates redundant `llvm.load`s of a
// structurally identical (pure-address-expression) pointer when no
// memory-clobbering op (store / call) lies between a dominating load and the
// redundant one. The detached `inttoptr(add(base, zext(idx)))` address chains
// the lifter emits for each eliminated linear-memory access are then removed
// by a local dead-code sweep.
//
// To stay sound WITHOUT MemorySSA, reuse is restricted to single-entry regions
// (extended basic blocks): the entry of every multi-predecessor block is
// treated as a memory clobber, so an ancestor load is never forwarded across a
// CFG join where a sibling path might have stored.
//
// This removes the repeated `mem[i]` reloads (and their spilled index reloads)
// that short-circuit `||` comparison chains in the lexer/parser generate, which
// dominate the runtime of the code we emit. Skippable via TINYC_NO_LOAD_CSE for
// A/B debugging (and TINYC_NO_LOAD_CSE_DCE to keep the dead address chains).
#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

void mlir_llvm_load_cse(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif

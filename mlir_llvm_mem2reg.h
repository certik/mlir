// LLVM-dialect mem2reg: promote non-escaping `llvm.alloca` slots into SSA
// values + block-argument phis, using Braun et al. (2013) simple SSA
// construction over the function's flat `cf` CFG.
//
// tinyC's emit.c materialises every C local as one entry-block `llvm.alloca`
// (clang -O0 style) and threads all cross-block data flow through
// `llvm.load` / `llvm.store`. mlir_llvm_to_wasmssa later lowers each alloca to
// a linear-memory shadow-stack slot, so every local access becomes an
// unconditional load/store the register allocator can never promote. Running
// this pass BEFORE that lowering converts the non-address-taken locals back to
// SSA, leaving only genuinely address-taken locals in the shadow stack.
//
// An alloca is promotable iff every use of its pointer result is the pointer
// operand of an `llvm.load` (operand 0) or `llvm.store` (operand 1), and every
// such load result / store value has the same type as the alloca's element
// type. Any other use (address-of, GEP, call argument, storing the pointer as
// a value, a type-punned load/store) leaves the alloca untouched.
//
// Operates on a whole `builtin.module`; each `func.func` body is promoted in
// place. Must run on the flat cf-dialect CFG produced by emit.c, before any
// cf->scf lifting. Safe to run before the standard/wasm lowering.
#ifndef MLIR_LLVM_MEM2REG_H
#define MLIR_LLVM_MEM2REG_H

#include "mlir_api.h"

void mlir_llvm_mem2reg(MLIR_Context *ctx, MLIR_OpHandle module);

#endif

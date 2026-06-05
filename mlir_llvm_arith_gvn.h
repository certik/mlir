#ifndef MLIR_LLVM_ARITH_GVN_H
#define MLIR_LLVM_ARITH_GVN_H

#include "mlir_api.h"

// Intra-block value numbering (CSE) of pure integer arithmetic on the
// wasm->llvm->aarch64 (from-wasm) path. Ported from the removed WMIR backend's
// `wmir_value_number` pass. Within each basic block, identical pure integer
// computations (add/sub/mul/and/or/xor/shl/lshr/ashr/sdiv/udiv/srem/urem/icmp)
// are computed once and later occurrences forwarded to the first. This removes
// the repeated index/address arithmetic the wasm frontend leaves behind and,
// more importantly, shrinks the number of simultaneously-live values, relieving
// the register pressure that drives the backend's spill traffic.
//
// Scope is intra-block (the value table resets at each block entry), so a
// cached definition always precedes (and therefore dominates) its reuse -- no
// dominator or alias analysis is needed and the pass is always correct.
//
// The address spine the backend fuses into a `[x28, Widx, UXTW]` addressing
// mode (`inttoptr(add(base, zext(idx)))`) is deliberately NOT value-numbered:
// the `add` feeding any `inttoptr` is excluded so each linmem access keeps its
// own single-use spine and the fusion is preserved. Conversions (zext/sext/
// trunc/inttoptr/ptrtoint/bitcast) are not numbered either.
//
// Honors the pipeline's no_def_use_tracking discipline: it never relies on
// RAUW; redundant values are rewritten by a manual operand sweep over every op
// operand and terminator successor-operand, then detached chains are removed by
// manual use-count DCE.
//
// Disable with TINYC_NO_ARITH_GVN=1 (A/B aid).
void mlir_llvm_arith_gvn(MLIR_Context *ctx, MLIR_OpHandle module);

#endif

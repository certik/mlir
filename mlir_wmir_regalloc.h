// wmir register allocator. Computes a per-SSA-value home (physical
// register or stack slot) for one wmir.func.
//
// Strategy: SSA-aware linear scan over the linearised CFG.
//
//   1. Linearise the function: each op gets a unique position; each
//      block records its [first_pos, last_pos] range.
//   2. Liveness: backward fixpoint dataflow over the CFG. For every
//      SSA value compute its live interval [def_pos, last_use_pos]
//      and the `crosses_call` flag.
//   3. Allocate: walk live intervals in start-position order. Pool =
//      caller-saved {x12..x15} plus callee-saved {x19..x26}. A value
//      whose live range crosses a call may use only the call-safe
//      callee-saved registers x24/x25 (the other callee-saved regs are
//      clobbered by hand-written runtime helpers); otherwise it falls
//      back to a stack slot. When all eligible pool regs are busy at a
//      def point, evict the active interval with the latest end
//      position and spill it. Callee-saved registers are preferred last
//      (caller-saved first) so leaf/call-light functions pay no
//      save/restore cost. The set of callee-saved registers actually
//      assigned is reported via `used_callee_mask` so the lowering can
//      save/restore exactly those in the prologue/epilogue.
//
// FP values (f32/f64), call-crossing values that cannot get x24/x25,
// and any value the lowering reads/writes through a fixed register
// (e.g. call args in x0..x7) map to stack slots. The allocator is
// invisible to those code paths.

#ifndef MLIR_WMIR_REGALLOC_H
#define MLIR_WMIR_REGALLOC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Where an SSA value lives after allocation.
typedef enum {
    HOME_SLOT  = 0,  // 8-byte stack slot at offset `idx * 8` from sp
    HOME_REG   = 1,  // physical AArch64 register `idx` (0..30; W or X chosen by type)
    HOME_CONST = 2,  // rematerializable integer constant; never occupies a
                     // register or slot. The lowering re-emits a `mov`-immediate
                     // at each use site, so identical constants cost no spill/
                     // reload and add no register pressure.
} HomeKind;

typedef struct {
    uint8_t  kind;    // HomeKind
    uint8_t  is_i64;  // HOME_CONST: 1 if the constant is i64, else 0
    uint16_t pad2;
    uint32_t idx;     // register number (0..30) or slot index (0..n_slots-1)
    int64_t  cval;    // HOME_CONST: the constant's integer value
} ValueHome;

typedef struct {
    // Parallel arrays. `values[i]` is the SSA value; `homes[i]` is
    // its home. Look up via `wmir_regalloc_lookup`. `n_values` is
    // the count; `cap` is the underlying capacity.
    MLIR_ValueHandle *values;
    ValueHome        *homes;
    size_t            n_values;
    size_t            cap;
    // Hash index: value handle -> index into values/homes (open addressing,
    // linear probing, power-of-two size, key 0 == empty). Built once after
    // allocation so `wmir_regalloc_lookup` is O(1) instead of O(n_values).
    MLIR_ValueHandle *hkeys;
    uint32_t         *hidx;
    size_t            hcap;
    // Number of 8-byte stack slots allocated. Frame size in bytes
    // (before 16-byte alignment) is `n_slots * 8`.
    uint32_t          n_slots;
    // Bitmask of callee-saved registers actually assigned to a value.
    // Bit i (0..7) corresponds to physical register x(19+i). The
    // lowering must save these in the prologue and restore them in the
    // epilogue. Zero for leaf functions that only needed caller-saved
    // registers.
    uint32_t          used_callee_mask;
} WmirRegAlloc;

// Compute the assignment for one `wmir.func` op. Returns a heap-
// allocated WmirRegAlloc that the caller must free with
// `wmir_regalloc_free`. Returns NULL on internal error.
WmirRegAlloc *wmir_regalloc_run(MLIR_Context *ctx, MLIR_OpHandle func);

// Look up the home of an SSA value. Returns true if `v` was found
// (and writes the home to *out); false if not (caller should treat
// this as an internal error — every defined value should be
// allocated).
bool wmir_regalloc_lookup(const WmirRegAlloc *ra,
                          MLIR_ValueHandle v,
                          ValueHome *out);

void wmir_regalloc_free(WmirRegAlloc *ra);

#ifdef __cplusplus
}
#endif

#endif  // MLIR_WMIR_REGALLOC_H

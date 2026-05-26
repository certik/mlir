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
//      {x11..x18}. A value with `crosses_call=true` always goes to a
//      stack slot (v1 doesn't yet use callee-saved registers). When
//      all pool regs are busy at a def point, evict the active
//      interval with the latest end position and spill it.
//
// FP values (f32/f64), values whose live range crosses a call, and
// any value the lowering reads/writes through a fixed register (e.g.
// call args in x0..x7) all map to stack slots — same as today. The
// allocator is invisible to those code paths.

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
    HOME_SLOT = 0,  // 8-byte stack slot at offset `idx * 8` from sp
    HOME_REG  = 1,  // physical AArch64 register `idx` (0..30; W or X chosen by type)
} HomeKind;

typedef struct {
    uint8_t  kind;  // HomeKind
    uint8_t  pad;
    uint16_t idx;   // register number (0..30) or slot index (0..n_slots-1)
} ValueHome;

typedef struct {
    // Parallel arrays. `values[i]` is the SSA value; `homes[i]` is
    // its home. Look up via `wmir_regalloc_lookup`. `n_values` is
    // the count; `cap` is the underlying capacity.
    MLIR_ValueHandle *values;
    ValueHome        *homes;
    size_t            n_values;
    size_t            cap;
    // Number of 8-byte stack slots allocated. Frame size in bytes
    // (before 16-byte alignment) is `n_slots * 8`.
    uint16_t          n_slots;
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

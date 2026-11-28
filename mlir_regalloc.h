// Shared, target-parameterized register allocator. See mlir_regalloc.c.
//
// The allocators are target-independent: they consume an `llvm`-dialect region
// and write a value->physical-register map (`rm`, a SlotMap). The register pool
// and the per-function fold/fusion maps are supplied via the data-only
// `RegTarget` (no function pointers, so this header stays parseable by the tinyC
// self-host). The aarch64 backend builds a RegTarget from its existing pool +
// fusion maps; the x64 backend supplies its own pool and NULL fusion maps.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returned by mlir_regalloc_global when the function is too large for the
// whole-function path (the caller then falls back to mlir_regalloc_cfg).
#define RA_BAIL 0xFFFFFFFFu

// Upper bound on a RegTarget's allocatable pool size (sizes stack arrays in the
// allocators without C99 VLAs). Must be >= every backend's npool.
#define RA_MAXPOOL 32

// Value handle -> int32 map (slot index / physical register / use count),
// open-addressed and arena-backed.
typedef struct { uintptr_t key; int32_t slot; } SlotEnt;
typedef struct {
    SlotEnt *t;
    size_t cap;
    size_t n;
    Arena *arena;
    // Optional dense direct-indexed table (empty = -1) for native dense handles.
    int32_t *dense;
    uintptr_t dense_base;
    size_t dense_len;
} SlotMap;
void sm_put(SlotMap *m, MLIR_ValueHandle k, int32_t slot);
bool sm_get(SlotMap *m, MLIR_ValueHandle k, int32_t *out);
void uc_inc(SlotMap *m, MLIR_ValueHandle k);

static inline bool sm_get_fast(SlotMap *m, MLIR_ValueHandle k, int32_t *out) {
    if (m->dense) {
        uintptr_t idx = (uintptr_t)k - m->dense_base;
        if (idx < m->dense_len) {
            int32_t v = m->dense[idx];
            if (v >= 0) { *out = v; return true; }
        }
        if (m->cap == 0) return false;
    }
    return sm_get(m, k, out);
}

// Fold/const/cast hooks the allocators call, defined by the aarch64 backend
// (mlir_llvm_to_aarch64.c). x64 reuses const_int_val / cast_src (generic) and
// passes NULL fusion maps.
bool const_int_val(MLIR_Context *ctx, MLIR_OpHandle op, int64_t *val, uint8_t *is64);
bool cast_src(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_ValueHandle *src);
bool a64_is_spine(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *memspine);
bool a64_memfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *memfuse,
                      MLIR_ValueHandle *base, MLIR_ValueHandle *idx);
bool a64_shiftfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *shiftfuse,
                        MLIR_ValueHandle *xv);

// Register-pool + fold/fusion description for one target. Data only.
//   pool[0..ncaller)        caller-saved physical registers
//   pool[ncaller..npool)    callee-saved physical registers
//   nhome                   # of top callee-saved regs reserved as unique
//                           block-arg homes (block-local path)
//   memfuse/memspine/shiftfuse  per-function fold/fusion maps, or NULL.
typedef struct RegTarget {
    int npool;
    int ncaller;
    int nhome;
    uint8_t pool[RA_MAXPOOL];
    SlotMap *memfuse, *memspine, *shiftfuse;
    SlotMap *remat_skip;
    SlotMap *call_save_mask;    // optional: (ValueHandle)call_op -> live caller-saved pool bitmask
    uint32_t *call_save_used;   // optional: union of all masks written to call_save_mask
    bool remat_alloca;
} RegTarget;

// Fast block-local linear-scan allocator for -O0 / default mode: O(ops) with
// zero global liveness dataflow, zero crosses_call backwards walk, and zero
// sort. Cross-block values and block args get dedicated frame slots; block-local
// values are allocated into callee-saved regs (or full pool in call-free blocks)
// with per-block spill-slot reuse. Returns callee-saved use mask, or RA_BAIL.
uint32_t mlir_regalloc_fast(MLIR_Context *ctx, MLIR_RegionHandle reg,
                            size_t n_blocks, SlotMap *rm,
                            SlotMap *sm, int32_t *out_nslots,
                            const RegTarget *T);
const MLIR_OpHandle *mlir_regalloc_fast_allocas(size_t *out_n);
uint32_t mlir_regalloc_fast_max_edge(void);

// Whole-function linear scan. Fills `rm` (value->phys reg) and `sm` (reused
// frame slots for spilled values), sets *out_nslots, and returns the callee-
// saved use mask, or RA_BAIL if the function is too large.
uint32_t mlir_regalloc_global(MLIR_Context *ctx, MLIR_RegionHandle reg,
                              size_t n_blocks, SlotMap *rm,
                              SlotMap *sm, int32_t *out_nslots,
                              const RegTarget *T);

// Block-local fallback: homes block-local + block-arg values into callee-saved
// registers. Fills `rm`; returns the callee-saved use mask.
uint32_t mlir_regalloc_cfg(MLIR_Context *ctx, MLIR_RegionHandle reg,
                           size_t n_blocks, SlotMap *rm,
                           const RegTarget *T);

#ifdef __cplusplus
}
#endif

// Whole-function dead-code elimination for the wasm->llvm->aarch64 (from-wasm)
// pipeline. See header for rationale.
//
// Runs in the no_def_use_tracking pipeline (examples/tinyc/driver.c), so it may
// NOT rely on MLIR_GetValueNumUses / RAUW. Instead it computes use counts by an
// explicit sweep over every op operand and terminator successor-operand, then a
// worklist erases zero-use pure ops and decrements their operands, re-queueing
// the operand-defining ops (which may themselves become dead).
#include "mlir_llvm_dce.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// uintptr -> int64 open-addressing map (value handle -> use count). Key 0 is
// reserved as "empty" (no valid value handle is 0).
// ---------------------------------------------------------------------------
typedef struct { uintptr_t *keys; int64_t *vals; size_t cap, n; } DceMap;

static void dcemap_init(DceMap *m, size_t hint) {
    m->cap = 64; while (m->cap < hint * 2) m->cap <<= 1;
    m->keys = (uintptr_t *)calloc(m->cap, sizeof(uintptr_t));
    m->vals = (int64_t *)calloc(m->cap, sizeof(int64_t));
    m->n = 0;
}
static void dcemap_free(DceMap *m) { free(m->keys); free(m->vals); }
static size_t dcemap_slot(uintptr_t *keys, size_t cap, uintptr_t k) {
    size_t i = (size_t)((k * 0x9E3779B97F4A7C15ull) & (cap - 1));
    while (keys[i] && keys[i] != k) i = (i + 1) & (cap - 1);
    return i;
}
static void dcemap_grow(DceMap *m) {
    size_t nc = m->cap << 1;
    uintptr_t *nk = (uintptr_t *)calloc(nc, sizeof(uintptr_t));
    int64_t *nv = (int64_t *)calloc(nc, sizeof(int64_t));
    for (size_t i = 0; i < m->cap; i++) if (m->keys[i]) {
        size_t j = dcemap_slot(nk, nc, m->keys[i]);
        nk[j] = m->keys[i]; nv[j] = m->vals[i];
    }
    free(m->keys); free(m->vals);
    m->keys = nk; m->vals = nv; m->cap = nc;
}
static void dcemap_put(DceMap *m, uintptr_t k, int64_t v) {
    if ((m->n + 1) * 2 >= m->cap) dcemap_grow(m);
    size_t i = dcemap_slot(m->keys, m->cap, k);
    if (!m->keys[i]) { m->keys[i] = k; m->n++; }
    m->vals[i] = v;
}
static bool dcemap_get(DceMap *m, uintptr_t k, int64_t *out) {
    size_t i = dcemap_slot(m->keys, m->cap, k);
    if (!m->keys[i]) return false;
    if (out) *out = m->vals[i];
    return true;
}

// ---------------------------------------------------------------------------
// Removable iff side-effect-free with a single result. Conservative: matches
// exactly the pure ops the lifter/backend can drop without observable effect.
// Loads, stores, calls, terminators and anything unrecognized are NEVER
// removed. Mirrors arith_gvn's ag_is_dce_pure, kept independent so each pass is
// self-contained.
// ---------------------------------------------------------------------------
static bool dce_name_eq(string s, const char *lit) {
    size_t n = strlen(lit);
    if (s.size != n) return false;
    return memcmp(s.str, lit, n) == 0;
}
static bool dce_is_pure(MLIR_OpHandle op) {
    switch (MLIR_GetOpType(op)) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_UDIV: case OP_TYPE_LLVM_SDIV:
    case OP_TYPE_LLVM_UREM: case OP_TYPE_LLVM_SREM:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR: case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_ICMP:
    case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_TRUNC:
    case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ADDRESSOF:
        return true;
    default: break;
    }
    string nm = MLIR_GetOpName(op);
    return dce_name_eq(nm, "llvm.inttoptr") ||
           dce_name_eq(nm, "llvm.bitcast") ||
           dce_name_eq(nm, "llvm.getelementptr") ||
           dce_name_eq(nm, "llvm.select") ||
           dce_name_eq(nm, "llvm.fadd") || dce_name_eq(nm, "llvm.fsub") ||
           dce_name_eq(nm, "llvm.fmul") || dce_name_eq(nm, "llvm.fdiv") ||
           dce_name_eq(nm, "llvm.fneg") || dce_name_eq(nm, "llvm.fcmp") ||
           dce_name_eq(nm, "llvm.fpext") || dce_name_eq(nm, "llvm.fptrunc") ||
           dce_name_eq(nm, "llvm.sitofp") || dce_name_eq(nm, "llvm.uitofp") ||
           dce_name_eq(nm, "llvm.fptosi") || dce_name_eq(nm, "llvm.fptoui");
}

// Worklist DCE driver: erase zero-use pure single-result ops, decrement their
// operands in `uses`, and re-queue operand-defining ops. `seeds` is the initial
// worklist (any op handle; non-pure / live ops are filtered inside).
static void dce_worklist(MLIR_Context *ctx, DceMap *uses,
                         MLIR_OpHandle *seeds, size_t nseed) {
    MLIR_OpHandle *wl = NULL; size_t nwl = 0, cwl = 0;
    for (size_t i = 0; i < nseed; i++) {
        if (seeds[i] == MLIR_INVALID_HANDLE) continue;
        if (nwl == cwl) { cwl = cwl ? cwl * 2 : 256; wl = (MLIR_OpHandle *)realloc(wl, cwl * sizeof(MLIR_OpHandle)); }
        wl[nwl++] = seeds[i];
    }
    while (nwl > 0) {
        MLIR_OpHandle op = wl[--nwl];
        if (op == MLIR_INVALID_HANDLE) continue;
        if (MLIR_GetOpParentBlock(op) == MLIR_INVALID_HANDLE) continue;  // already erased
        if (MLIR_GetOpNumResults(op) != 1 || !dce_is_pure(op)) continue;
        int64_t cnt = 0; dcemap_get(uses, (uintptr_t)MLIR_GetOpResult(op, 0), &cnt);
        if (cnt != 0) continue;
        size_t non = MLIR_GetOpNumOperands(op);
        MLIR_OpHandle *defs = (MLIR_OpHandle *)malloc((non ? non : 1) * sizeof(MLIR_OpHandle));
        for (size_t kk = 0; kk < non; kk++) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, kk);
            defs[kk] = (MLIR_GetValueKind(v) == OP_RESULT) ? MLIR_GetValueDefiningOp(v) : MLIR_INVALID_HANDLE;
            int64_t c0 = 0; dcemap_get(uses, (uintptr_t)v, &c0);
            if (c0 > 0) dcemap_put(uses, (uintptr_t)v, c0 - 1);
        }
        MLIR_EraseOp(ctx, op);
        for (size_t kk = 0; kk < non; kk++) if (defs[kk] != MLIR_INVALID_HANDLE) {
            if (nwl == cwl) { cwl = cwl ? cwl * 2 : 256; wl = (MLIR_OpHandle *)realloc(wl, cwl * sizeof(MLIR_OpHandle)); }
            wl[nwl++] = defs[kk];
        }
        free(defs);
    }
    free(wl);
}

// uintptr -> index map for block-handle -> region-block-index lookups.
static bool dce_blkidx(MLIR_RegionHandle region, size_t nb,
                       MLIR_BlockHandle b, size_t *out) {
    for (size_t i = 0; i < nb; i++)
        if (MLIR_GetRegionBlock(region, i) == b) { *out = i; return true; }
    return false;
}

// Remove block arguments that have no uses, fixing up every predecessor edge's
// successor-operand list to match. The tinyC loop lowering threads a dead
// loop-result value through a block argument that is stored (an edge copy /
// spill) every iteration but never read; removing it deletes one dead store per
// loop iteration. Returns DCE seeds (defining ops of the removed edge operands,
// which may now be dead) via `*seeds`/`*nseed` (caller frees).
static void dce_dead_block_args(MLIR_Context *ctx, MLIR_RegionHandle region,
                                size_t nb, DceMap *uses,
                                MLIR_OpHandle **seeds, size_t *nseed) {
    *seeds = NULL; *nseed = 0; size_t cseed = 0;
    // Block-handle -> region-block-index, built once (O(nb)).
    DceMap b2i; dcemap_init(&b2i, nb);
    for (size_t bi = 0; bi < nb; bi++)
        dcemap_put(&b2i, (uintptr_t)MLIR_GetRegionBlock(region, bi), (int64_t)bi);
    // Predecessor list per block: parallel arrays of (op, succ_idx). Edges are
    // stable across this pass (only operands are removed, never successors), so
    // build once. Indexed by region-block index.
    typedef struct { MLIR_OpHandle *ops; size_t *succ; size_t n, cap; } PredList;
    PredList *preds = (PredList *)calloc(nb, sizeof(PredList));
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, bi);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                int64_t ti;
                if (!dcemap_get(&b2i, (uintptr_t)MLIR_GetOpSuccessor(op, s), &ti)) continue;
                PredList *p = &preds[ti];
                if (p->n == p->cap) {
                    p->cap = p->cap ? p->cap * 2 : 4;
                    p->ops = (MLIR_OpHandle *)realloc(p->ops, p->cap * sizeof(MLIR_OpHandle));
                    p->succ = (size_t *)realloc(p->succ, p->cap * sizeof(size_t));
                }
                p->ops[p->n] = op; p->succ[p->n] = s; p->n++;
            }
        }
    }

    // Skip block 0 (function entry: its args are the ABI parameters).
    for (size_t bi = 1; bi < nb; bi++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, bi);
        size_t na = MLIR_GetBlockNumArgs(bk);
        // High -> low so erasing an arg never shifts a not-yet-processed index.
        for (size_t ai = na; ai-- > 0; ) {
            MLIR_ValueHandle arg = MLIR_GetBlockArg(bk, ai);
            int64_t cnt = 0; dcemap_get(uses, (uintptr_t)arg, &cnt);
            if (cnt != 0) continue;                       // live block-arg
            // Rewrite each predecessor edge: drop the operand at index `ai`.
            PredList *p = &preds[bi];
            for (size_t k = 0; k < p->n; k++) {
                MLIR_OpHandle op = p->ops[k]; size_t s = p->succ[k];
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                if (ai >= nso) continue;                  // defensive
                MLIR_ValueHandle *nv = (MLIR_ValueHandle *)malloc((nso ? nso : 1) * sizeof(MLIR_ValueHandle));
                size_t m = 0;
                for (size_t j = 0; j < nso; j++) {
                    MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(op, s, j);
                    if (j == ai) {
                        int64_t c0 = 0; dcemap_get(uses, (uintptr_t)v, &c0);
                        if (c0 > 0) dcemap_put(uses, (uintptr_t)v, c0 - 1);
                        if (MLIR_GetValueKind(v) == OP_RESULT) {
                            if (*nseed == cseed) { cseed = cseed ? cseed * 2 : 64; *seeds = (MLIR_OpHandle *)realloc(*seeds, cseed * sizeof(MLIR_OpHandle)); }
                            (*seeds)[(*nseed)++] = MLIR_GetValueDefiningOp(v);
                        }
                        continue;
                    }
                    nv[m++] = v;
                }
                MLIR_SetOpSuccessorOperands(ctx, op, s, nv, m);
                free(nv);
            }
            MLIR_EraseBlockArguments(ctx, bk, ai, 1);
        }
    }

    for (size_t bi = 0; bi < nb; bi++) { free(preds[bi].ops); free(preds[bi].succ); }
    free(preds);
    dcemap_free(&b2i);
}

static void dce_run_region(MLIR_Context *ctx, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    if (nb == 0) return;

    // 1. Count uses of every value across regular operands AND terminator
    //    successor-operands (block-arg edge values).
    DceMap uses; dcemap_init(&uses, 1024);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t kk = 0; kk < non; kk++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, kk);
                int64_t c0 = 0; dcemap_get(&uses, (uintptr_t)v, &c0);
                dcemap_put(&uses, (uintptr_t)v, c0 + 1);
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t kk = 0; kk < nso; kk++) {
                    MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(op, s, kk);
                    int64_t c0 = 0; dcemap_get(&uses, (uintptr_t)v, &c0);
                    dcemap_put(&uses, (uintptr_t)v, c0 + 1);
                }
            }
        }
    }

    // 2. Seed a worklist with every op and erase zero-use pure ops transitively.
    MLIR_OpHandle *all = NULL; size_t nall = 0, call = 0;
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            if (nall == call) { call = call ? call * 2 : 256; all = (MLIR_OpHandle *)realloc(all, call * sizeof(MLIR_OpHandle)); }
            all[nall++] = MLIR_GetBlockOp(bk, i);
        }
    }
    dce_worklist(ctx, &uses, all, nall);
    free(all);

    // 3. Dead block-argument elimination (gated): removes loop-carried dead
    //    block args (and their per-iteration edge stores), then re-DCEs the now
    //    detached values that fed those edges.
    if (!getenv("TINYC_NO_DEADARG")) {
        MLIR_OpHandle *seeds = NULL; size_t nseed = 0;
        dce_dead_block_args(ctx, region, nb, &uses, &seeds, &nseed);
        if (nseed) dce_worklist(ctx, &uses, seeds, nseed);
        free(seeds);
    }

    dcemap_free(&uses);
}

void mlir_llvm_dce(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_DCE")) return;
    if (MLIR_GetOpNumRegions(module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        MLIR_OpType ft = MLIR_GetOpType(op);
        if (ft != OP_TYPE_FUNC_FUNC && ft != OP_TYPE_LLVM_FUNC) continue;
        if (MLIR_GetOpNumRegions(op) == 0) continue;
        dce_run_region(ctx, MLIR_GetOpRegion(op, 0));
    }
}

// LLVM-dialect mem2reg — see mlir_llvm_mem2reg.h for the high-level description.
//
// Braun et al. (2013) "Simple and Efficient Construction of Static Single
// Assignment Form", specialised to a function's flat cf-dialect CFG (every
// block's predecessors are known, so all blocks are treated as sealed and no
// incomplete-phi bookkeeping is needed).
//
// Like the wasm-lifter pass, this is def-use-tracking-agnostic: it never relies
// on use lists or MLIR_ReplaceAllUsesOfValue. It builds its own value->value
// remap table (llvm.load result -> reaching SSA value) and, after rewriting,
// sweeps every op operand and terminator successor-operand to resolve through
// it.

#include "mlir_llvm_mem2reg.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// --- open-addressing uintptr_t -> uintptr_t map -----------------------------
// Key 0 (MLIR_INVALID_HANDLE) marks an empty slot, so 0 is never a valid key.
typedef struct {
    uintptr_t *keys;
    uintptr_t *vals;
    size_t cap;   // power of two
    size_t n;
} HMap;

static void hmap_init(HMap *m, size_t hint) {
    size_t cap = 16;
    while (cap < hint * 2) cap <<= 1;
    m->keys = (uintptr_t *)calloc(cap, sizeof(uintptr_t));
    m->vals = (uintptr_t *)calloc(cap, sizeof(uintptr_t));
    m->cap = cap;
    m->n = 0;
}

static void hmap_free(HMap *m) {
    free(m->keys);
    free(m->vals);
    m->keys = NULL;
    m->vals = NULL;
    m->cap = 0;
    m->n = 0;
}

static size_t hmap_slot(uintptr_t *keys, size_t cap, uintptr_t key) {
    uintptr_t h = key;
    h ^= h >> 16;
    h *= (uintptr_t)0x9E3779B97F4A7C15ull;
    h ^= h >> 16;
    size_t i = (size_t)h & (cap - 1);
    while (keys[i] != 0 && keys[i] != key)
        i = (i + 1) & (cap - 1);
    return i;
}

static void hmap_put(HMap *m, uintptr_t key, uintptr_t val);

static void hmap_grow(HMap *m) {
    size_t ncap = m->cap << 1;
    uintptr_t *nk = (uintptr_t *)calloc(ncap, sizeof(uintptr_t));
    uintptr_t *nv = (uintptr_t *)calloc(ncap, sizeof(uintptr_t));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->keys[i] != 0) {
            size_t s = hmap_slot(nk, ncap, m->keys[i]);
            nk[s] = m->keys[i];
            nv[s] = m->vals[i];
        }
    }
    free(m->keys);
    free(m->vals);
    m->keys = nk;
    m->vals = nv;
    m->cap = ncap;
}

static void hmap_put(HMap *m, uintptr_t key, uintptr_t val) {
    if ((m->n + 1) * 4 >= m->cap * 3) hmap_grow(m);
    size_t i = hmap_slot(m->keys, m->cap, key);
    if (m->keys[i] == 0) {
        m->keys[i] = key;
        m->n++;
    }
    m->vals[i] = val;
}

static bool hmap_get(HMap *m, uintptr_t key, uintptr_t *out) {
    if (m->cap == 0) return false;
    size_t i = hmap_slot(m->keys, m->cap, key);
    if (m->keys[i] == 0) return false;
    *out = m->vals[i];
    return true;
}

// --- pass context -----------------------------------------------------------
//
// Phis are modelled lazily as lightweight integer ids (NOT IR block arguments)
// during construction; only the survivors of triviality elimination become
// real block arguments. Bit 0 (set) tags a phi id; IR handles are >=2-byte
// aligned so bit 0 is free for the tag, valid for 32-bit uintptr_t too.
#define L2R_PHI_TAG ((uintptr_t)1)
#define L2R_IS_PHI(r) (((uintptr_t)(r) & L2R_PHI_TAG) != 0)
#define L2R_PHI_ID(r) ((size_t)((uintptr_t)(r) >> 1))
#define L2R_MK_PHI(id) ((((uintptr_t)(id)) << 1) | L2R_PHI_TAG)

typedef struct {
    MLIR_Context *ctx;
    size_t nb;                  // number of blocks
    size_t na;                  // number of candidate allocas
    MLIR_TypeHandle *elemty;    // element type per candidate alloca (na)
    bool *promotable;           // per candidate alloca (na)
    MLIR_BlockHandle *blocks;   // blocks[nb]
    MLIR_OpHandle   *terms;     // terminator op per block (0 if none)
    size_t *pred_off;           // nb+1 prefix offsets into pred_blk/pred_slot
    size_t *pred_blk;           // predecessor block index per edge
    size_t *pred_slot;          // predecessor successor-slot per edge
    // Lightweight phi table (heap, reclaimed at end).
    size_t pcount, pcap;
    size_t *p_block;            // block index of each phi
    size_t *p_alloca;           // candidate-alloca index of each phi
    uintptr_t **p_ops;          // p_ops[pid][k] = vref from k-th pred
    size_t *p_nops;             // number of operands == preds of p_block
    uintptr_t *p_rep;           // representative vref (self phi-ref if surviving)
    MLIR_ValueHandle *p_handle; // materialised block-arg handle (0 until done)
    HMap defOut;                // (bi*na+ai+1) -> last value stored in block
    HMap entryDef;              // (bi*na+ai+1) -> memoised reaching vref at entry
    HMap bidx;                  // block handle -> block index
    HMap aidx;                  // alloca result handle -> candidate index
    HMap remap;                 // llvm.load result handle -> reaching vref
} M;

static MLIR_LocationHandle l2r_unkloc(MLIR_Context *ctx) {
    return MLIR_CreateLocationUnknown(ctx, (string){0});
}

static size_t l2r_block_index(M *m, MLIR_BlockHandle b) {
    uintptr_t v;
    if (hmap_get(&m->bidx, (uintptr_t)b, &v)) return (size_t)v;
    return (size_t)-1;
}

// Returns candidate-alloca index of `v` (an alloca pointer result), or -1.
static size_t l2r_alloca_index(M *m, MLIR_ValueHandle v) {
    uintptr_t a;
    if (hmap_get(&m->aidx, (uintptr_t)v, &a)) return (size_t)a;
    return (size_t)-1;
}

// Recursively mark allocas as non-promotable when their pointer escapes.
// An alloca is promotable only if every use of its result is a TOP-LEVEL
// (direct child of the function region) llvm.load (operand 0) or llvm.store
// (operand 1) whose value type matches the alloca's element type. Any other
// use — including ANY use nested inside an scf/structured region, where the
// top-level CFG phi machinery cannot reach — disqualifies it.
static void l2r_mark_escapes_block(M *m, MLIR_BlockHandle blk, bool nested) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t j = 0; j < nops; j++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
        MLIR_OpType t = MLIR_GetOpType(op);
        size_t no = MLIR_GetOpNumOperands(op);
        for (size_t k = 0; k < no; k++) {
            size_t ai = l2r_alloca_index(m, MLIR_GetOpOperand(op, k));
            if (ai == (size_t)-1) continue;
            bool ok = false;
            if (!nested && t == OP_TYPE_LLVM_LOAD && k == 0) {
                ok = (MLIR_GetValueType(MLIR_GetOpResult(op, 0)) == m->elemty[ai]);
            } else if (!nested && t == OP_TYPE_LLVM_STORE && k == 1) {
                ok = (MLIR_GetValueType(MLIR_GetOpOperand(op, 0)) == m->elemty[ai]);
            }
            if (!ok) m->promotable[ai] = false;
        }
        // A terminator could carry the pointer as a successor operand.
        size_t nsuc = MLIR_GetOpNumSuccessors(op);
        for (size_t s = 0; s < nsuc; s++) {
            size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
            for (size_t i = 0; i < nso; i++) {
                size_t ai = l2r_alloca_index(m, MLIR_GetOpSuccessorOperand(op, s, i));
                if (ai != (size_t)-1) m->promotable[ai] = false;
            }
        }
        // Recurse into nested regions: any alloca touched there cannot be
        // promoted by the top-level SSA construction.
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, r);
            size_t rnb = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < rnb; b++)
                l2r_mark_escapes_block(m, MLIR_GetRegionBlock(rg, b), true);
        }
    }
}

// Defensive zero materialisation for a read with no reaching def (an
// uninitialized local on some path — UB in C; zero is a valid defined choice).
static MLIR_ValueHandle l2r_make_zero(M *m, size_t bi, size_t ai) {
    MLIR_Context *ctx = m->ctx;
    MLIR_TypeHandle ty = m->elemty[ai];
    MLIR_LocationHandle loc = l2r_unkloc(ctx);
    MLIR_AttributeHandle a = MLIR_IsTypeFloat(ty)
        ? MLIR_CreateAttributeFloat(ctx, str_lit("value"), 0.0, ty)
        : MLIR_CreateAttributeInteger(ctx, str_lit("value"), 0, ty);
    MLIR_ValueHandle res =
        MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, ty,
                                 (string){0}, loc);
    MLIR_TypeHandle rty[1] = { ty };
    MLIR_ValueHandle rv[1] = { res };
    MLIR_AttributeHandle attrs[1] = { a };
    MLIR_OpHandle op = MLIR_CreateOp(
        ctx, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
        attrs, 1, rty, 1, rv, 1, NULL, 0, NULL, 0,
        loc, MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_InsertBlockOpAtIndex(ctx, m->blocks[bi], op, 0);
    return res;
}

static uintptr_t l2r_live_in(M *m, size_t ai, size_t bi);

static uintptr_t l2r_live_out(M *m, size_t ai, size_t bi) {
    uintptr_t d;
    if (hmap_get(&m->defOut, (uintptr_t)(bi * m->na + ai + 1), &d))
        return d;
    return l2r_live_in(m, ai, bi);
}

static size_t l2r_new_phi(M *m, size_t bi, size_t ai, size_t np) {
    if (m->pcount == m->pcap) {
        size_t nc = m->pcap ? m->pcap * 2 : 1024;
        m->p_block  = (size_t *)realloc(m->p_block, nc * sizeof(size_t));
        m->p_alloca = (size_t *)realloc(m->p_alloca, nc * sizeof(size_t));
        m->p_ops    = (uintptr_t **)realloc(m->p_ops, nc * sizeof(uintptr_t *));
        m->p_nops   = (size_t *)realloc(m->p_nops, nc * sizeof(size_t));
        m->p_rep    = (uintptr_t *)realloc(m->p_rep, nc * sizeof(uintptr_t));
        m->p_handle = (MLIR_ValueHandle *)realloc(m->p_handle,
                                                  nc * sizeof(MLIR_ValueHandle));
        m->pcap = nc;
    }
    size_t pid = m->pcount++;
    m->p_block[pid]  = bi;
    m->p_alloca[pid] = ai;
    m->p_nops[pid]   = np;
    m->p_ops[pid]    = (uintptr_t *)malloc((np ? np : 1) * sizeof(uintptr_t));
    m->p_rep[pid]    = L2R_MK_PHI(pid);  // assume surviving until proven trivial
    m->p_handle[pid] = 0;
    return pid;
}

static uintptr_t l2r_live_in(M *m, size_t ai, size_t bi) {
    uintptr_t key = (uintptr_t)(bi * m->na + ai + 1);
    uintptr_t e;
    if (hmap_get(&m->entryDef, key, &e)) return e;

    size_t po = m->pred_off[bi];
    size_t pe = m->pred_off[bi + 1];
    size_t np = pe - po;

    if (np == 0) {
        uintptr_t v = (uintptr_t)l2r_make_zero(m, bi, ai);
        hmap_put(&m->entryDef, key, v);
        return v;
    }
    if (np == 1) {
        uintptr_t v = l2r_live_out(m, ai, m->pred_blk[po]);
        hmap_put(&m->entryDef, key, v);
        return v;
    }

    // Merge point: create a lightweight phi id and memoise it BEFORE recursing
    // so back-edges in loops terminate.
    size_t pid = l2r_new_phi(m, bi, ai, np);
    uintptr_t ref = L2R_MK_PHI(pid);
    hmap_put(&m->entryDef, key, ref);
    for (size_t e2 = po; e2 < pe; e2++)
        m->p_ops[pid][e2 - po] = l2r_live_out(m, ai, m->pred_blk[e2]);
    return ref;
}

// Follow the phi-representative chain to a real handle or a surviving phi-ref.
static uintptr_t l2r_resolve_phi(M *m, uintptr_t r) {
    for (int guard = 0; guard < (1 << 26) && L2R_IS_PHI(r); guard++) {
        uintptr_t nr = m->p_rep[L2R_PHI_ID(r)];
        if (nr == r) break;
        r = nr;
    }
    return r;
}

// Resolve a value reference (possibly a load result via remap and/or a
// phi-ref) to a concrete materialised value handle.
static MLIR_ValueHandle l2r_concrete(M *m, uintptr_t v) {
    uintptr_t cur = v;
    // Alternate remap-chasing (erased load result -> its value) and trivial-phi
    // resolution until a fixpoint. Both are needed because a promoted alloca's
    // stored value can itself be an erased load of another promoted alloca
    // (e.g. `T *p = s;` where `s` is also promoted), and a trivial phi can
    // resolve to such an erased load result.
    for (int guard = 0; guard < (1 << 26); guard++) {
        uintptr_t before = cur;
        for (int g2 = 0; g2 < (1 << 24); g2++) {
            if (L2R_IS_PHI(cur)) break;
            uintptr_t nx;
            if (!hmap_get(&m->remap, cur, &nx)) break;
            if (nx == cur) break;
            cur = nx;
        }
        cur = l2r_resolve_phi(m, cur);
        if (cur == before) break;
    }
    if (L2R_IS_PHI(cur)) return m->p_handle[L2R_PHI_ID(cur)];
    return (MLIR_ValueHandle)cur;
}

// Recursively resolve every operand in a block (and its nested regions)
// through the remap/phi tables to its concrete value.
static void l2r_sweep_block(M *m, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t j = 0; j < nops; j++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
        size_t no = MLIR_GetOpNumOperands(op);
        for (size_t k = 0; k < no; k++) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
            MLIR_ValueHandle r = l2r_concrete(m, (uintptr_t)v);
            if (r != v) MLIR_SetOpOperand(m->ctx, op, k, r);
        }
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t rr = 0; rr < nr; rr++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, rr);
            size_t rnb = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < rnb; b++)
                l2r_sweep_block(m, MLIR_GetRegionBlock(rg, b));
        }
    }
}

// Promote one function body region (flat cf CFG) in place.
static void l2r_promote_region(MLIR_Context *ctx, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    if (nb == 0) return;

    M m;
    memset(&m, 0, sizeof(m));
    m.ctx = ctx;
    m.nb = nb;

    m.blocks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    m.terms  = (MLIR_OpHandle *)malloc(nb * sizeof(MLIR_OpHandle));
    hmap_init(&m.bidx, nb);
    for (size_t bi = 0; bi < nb; bi++) {
        m.blocks[bi] = MLIR_GetRegionBlock(region, bi);
        m.terms[bi]  = MLIR_GetBlockTerminator(m.blocks[bi]);
        hmap_put(&m.bidx, (uintptr_t)m.blocks[bi], (uintptr_t)bi);
    }

    // ---- Collect candidate allocas (every llvm.alloca in the region) -------
    hmap_init(&m.aidx, 64);
    size_t cap = 64;
    m.elemty     = (MLIR_TypeHandle *)malloc(cap * sizeof(MLIR_TypeHandle));
    m.promotable = (bool *)malloc(cap * sizeof(bool));
    MLIR_ValueHandle *alloca_res = (MLIR_ValueHandle *)malloc(cap * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle    *alloca_op  = (MLIR_OpHandle *)malloc(cap * sizeof(MLIR_OpHandle));
    size_t na = 0;
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = m.blocks[bi];
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            if (MLIR_GetOpType(op) != OP_TYPE_LLVM_ALLOCA) continue;
            MLIR_AttributeHandle ea = MLIR_GetOpAttributeByName(op, "elem_type");
            if (ea == MLIR_INVALID_HANDLE) continue;
            if (na == cap) {
                cap <<= 1;
                m.elemty     = (MLIR_TypeHandle *)realloc(m.elemty, cap * sizeof(MLIR_TypeHandle));
                m.promotable = (bool *)realloc(m.promotable, cap * sizeof(bool));
                alloca_res   = (MLIR_ValueHandle *)realloc(alloca_res, cap * sizeof(MLIR_ValueHandle));
                alloca_op    = (MLIR_OpHandle *)realloc(alloca_op, cap * sizeof(MLIR_OpHandle));
            }
            m.elemty[na]     = MLIR_GetAttributeTypeValue(ea);
            m.promotable[na] = true;
            alloca_res[na]   = MLIR_GetOpResult(op, 0);
            alloca_op[na]    = op;
            hmap_put(&m.aidx, (uintptr_t)alloca_res[na], (uintptr_t)na);
            na++;
        }
    }
    m.na = na;
    if (na == 0) {
        free(m.blocks); free(m.terms);
        free(m.elemty); free(m.promotable); free(alloca_res); free(alloca_op);
        hmap_free(&m.bidx); hmap_free(&m.aidx);
        return;
    }

    // ---- Escape / type-mismatch analysis -----------------------------------
    // An alloca pointer is promotable iff every use is a top-level llvm.load
    // (op 0) / llvm.store (op 1) of matching element type. Recurses into
    // nested scf/structured regions to disqualify allocas used there.
    for (size_t bi = 0; bi < nb; bi++)
        l2r_mark_escapes_block(&m, m.blocks[bi], false);

    bool any = false;
    for (size_t ai = 0; ai < na; ai++) if (m.promotable[ai]) { any = true; break; }
    if (!any) {
        free(m.blocks); free(m.terms);
        free(m.elemty); free(m.promotable); free(alloca_res); free(alloca_op);
        hmap_free(&m.bidx); hmap_free(&m.aidx);
        return;
    }

    hmap_init(&m.defOut, 256);
    hmap_init(&m.entryDef, 256);
    hmap_init(&m.remap, 256);

    // ---- defOut: last value stored to each promotable alloca per block -----
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = m.blocks[bi];
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            if (MLIR_GetOpType(op) != OP_TYPE_LLVM_STORE) continue;
            size_t ai = l2r_alloca_index(&m, MLIR_GetOpOperand(op, 1));
            if (ai == (size_t)-1 || !m.promotable[ai]) continue;
            hmap_put(&m.defOut, (uintptr_t)(bi * na + ai + 1),
                     (uintptr_t)MLIR_GetOpOperand(op, 0));
        }
    }

    // ---- Predecessor edges (one per terminator successor slot) -------------
    m.pred_off = (size_t *)calloc(nb + 1, sizeof(size_t));
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_OpHandle t = m.terms[bi];
        if (t == 0) continue;
        size_t ns = MLIR_GetOpNumSuccessors(t);
        for (size_t s = 0; s < ns; s++) {
            size_t k = l2r_block_index(&m, MLIR_GetOpSuccessor(t, s));
            if (k != (size_t)-1) m.pred_off[k + 1]++;
        }
    }
    for (size_t bi = 0; bi < nb; bi++) m.pred_off[bi + 1] += m.pred_off[bi];
    size_t nedges = m.pred_off[nb];
    m.pred_blk  = (size_t *)malloc((nedges ? nedges : 1) * sizeof(size_t));
    m.pred_slot = (size_t *)malloc((nedges ? nedges : 1) * sizeof(size_t));
    {
        size_t *fill = (size_t *)calloc(nb, sizeof(size_t));
        for (size_t bi = 0; bi < nb; bi++) {
            MLIR_OpHandle t = m.terms[bi];
            if (t == 0) continue;
            size_t ns = MLIR_GetOpNumSuccessors(t);
            for (size_t s = 0; s < ns; s++) {
                size_t k = l2r_block_index(&m, MLIR_GetOpSuccessor(t, s));
                if (k == (size_t)-1) continue;
                size_t pos = m.pred_off[k] + fill[k]++;
                m.pred_blk[pos] = bi;
                m.pred_slot[pos] = s;
            }
        }
        free(fill);
    }

    // ---- Rewrite: build the remap table and lazily create phi ids ----------
    uintptr_t *cur = (uintptr_t *)malloc(na * sizeof(uintptr_t));
    size_t erase_cap = 256, erase_n = 0;
    MLIR_OpHandle *erase = (MLIR_OpHandle *)malloc(erase_cap * sizeof(MLIR_OpHandle));
    #define L2R_ERASE(o) do { \
        if (erase_n == erase_cap) { erase_cap <<= 1; \
            erase = (MLIR_OpHandle *)realloc(erase, erase_cap * sizeof(MLIR_OpHandle)); } \
        erase[erase_n++] = (o); } while (0)

    for (size_t bi = 0; bi < nb; bi++) {
        for (size_t ai = 0; ai < na; ai++) cur[ai] = 0;
        MLIR_BlockHandle blk = m.blocks[bi];
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            MLIR_OpType t = MLIR_GetOpType(op);
            if (t == OP_TYPE_LLVM_STORE) {
                size_t ai = l2r_alloca_index(&m, MLIR_GetOpOperand(op, 1));
                if (ai == (size_t)-1 || !m.promotable[ai]) continue;
                cur[ai] = (uintptr_t)MLIR_GetOpOperand(op, 0);
            } else if (t == OP_TYPE_LLVM_LOAD) {
                size_t ai = l2r_alloca_index(&m, MLIR_GetOpOperand(op, 0));
                if (ai == (size_t)-1 || !m.promotable[ai]) continue;
                uintptr_t val = cur[ai];
                if (val == 0) { val = l2r_live_in(&m, ai, bi); cur[ai] = val; }
                hmap_put(&m.remap, (uintptr_t)MLIR_GetOpResult(op, 0), val);
            } else {
                continue;
            }
            L2R_ERASE(op);
            // Creating phis may insert zero consts into this block.
            nops = MLIR_GetBlockNumOps(blk);
        }
    }
    // The promotable alloca ops are now dead.
    for (size_t ai = 0; ai < na; ai++)
        if (m.promotable[ai]) L2R_ERASE(alloca_op[ai]);

    // ---- Trivial-phi elimination (Braun et al.) to a fixpoint --------------
    for (size_t pass = 0; pass < m.pcount + 2; pass++) {
        bool changed = false;
        for (size_t pid = 0; pid < m.pcount; pid++) {
            if (m.p_rep[pid] != L2R_MK_PHI(pid)) continue;
            uintptr_t self = L2R_MK_PHI(pid);
            uintptr_t same = 0;
            bool trivial = true;
            for (size_t o = 0; o < m.p_nops[pid]; o++) {
                uintptr_t r = l2r_resolve_phi(&m, m.p_ops[pid][o]);
                if (r == self) continue;
                if (same == 0) same = r;
                else if (r != same) { trivial = false; break; }
            }
            if (trivial && same != 0) { m.p_rep[pid] = same; changed = true; }
        }
        if (!changed) break;
    }

    // ---- Materialise surviving phis as block arguments (pid order) ---------
    size_t *surv_off = (size_t *)calloc(nb + 1, sizeof(size_t));
    for (size_t pid = 0; pid < m.pcount; pid++) {
        if (l2r_resolve_phi(&m, L2R_MK_PHI(pid)) != L2R_MK_PHI(pid)) continue;
        MLIR_TypeHandle ty = m.elemty[m.p_alloca[pid]];
        m.p_handle[pid] =
            MLIR_AddBlockArgument(ctx, m.blocks[m.p_block[pid]], ty, l2r_unkloc(ctx));
        surv_off[m.p_block[pid] + 1]++;
    }
    for (size_t bi = 0; bi < nb; bi++) surv_off[bi + 1] += surv_off[bi];
    size_t nsurv = surv_off[nb];
    size_t *surv_pid = (size_t *)malloc((nsurv ? nsurv : 1) * sizeof(size_t));
    {
        size_t *fill = (size_t *)calloc(nb, sizeof(size_t));
        for (size_t pid = 0; pid < m.pcount; pid++) {
            if (l2r_resolve_phi(&m, L2R_MK_PHI(pid)) != L2R_MK_PHI(pid)) continue;
            size_t bi = m.p_block[pid];
            surv_pid[surv_off[bi] + fill[bi]++] = pid;
        }
        free(fill);
    }

    // ---- Final sweep: resolve every regular operand to a concrete value ----
    // Recurses into nested scf/structured regions so any reference to a
    // remapped (now-erased) load result is updated wherever it appears.
    for (size_t bi = 0; bi < nb; bi++)
        l2r_sweep_block(&m, m.blocks[bi]);

    // ---- Edge wiring: append each surviving phi's incoming value -----------
    size_t sobuf_cap = 8;
    MLIR_ValueHandle *sobuf =
        (MLIR_ValueHandle *)malloc(sobuf_cap * sizeof(MLIR_ValueHandle));
    for (size_t k = 0; k < nb; k++) {
        size_t add = surv_off[k + 1] - surv_off[k];
        for (size_t e = m.pred_off[k]; e < m.pred_off[k + 1]; e++) {
            MLIR_OpHandle term = m.terms[m.pred_blk[e]];
            size_t slot = m.pred_slot[e];
            size_t nso = MLIR_GetOpNumSuccessorOperands(term, slot);
            size_t total = nso + add;
            if (total == 0) continue;
            if (total > sobuf_cap) {
                while (sobuf_cap < total) sobuf_cap <<= 1;
                sobuf = (MLIR_ValueHandle *)realloc(
                    sobuf, sobuf_cap * sizeof(MLIR_ValueHandle));
            }
            bool changed = (add > 0);
            for (size_t i = 0; i < nso; i++) {
                MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(term, slot, i);
                MLIR_ValueHandle r = l2r_concrete(&m, (uintptr_t)v);
                sobuf[i] = r;
                if (r != v) changed = true;
            }
            size_t eidx = e - m.pred_off[k];
            for (size_t s = 0; s < add; s++) {
                size_t pid = surv_pid[surv_off[k] + s];
                sobuf[nso + s] = l2r_concrete(&m, m.p_ops[pid][eidx]);
            }
            if (changed)
                MLIR_SetOpSuccessorOperands(ctx, term, slot, sobuf, total);
        }
    }

    // ---- Erase the now-dead load / store / alloca ops ----------------------
    for (size_t i = 0; i < erase_n; i++)
        MLIR_EraseOp(ctx, erase[i]);

    #undef L2R_ERASE
    free(sobuf);
    free(surv_off);
    free(surv_pid);
    free(erase);
    free(cur);
    for (size_t pid = 0; pid < m.pcount; pid++) free(m.p_ops[pid]);
    free(m.p_block);
    free(m.p_alloca);
    free(m.p_ops);
    free(m.p_nops);
    free(m.p_rep);
    free(m.p_handle);
    free(m.pred_blk);
    free(m.pred_slot);
    free(m.pred_off);
    free(m.blocks);
    free(m.terms);
    free(m.elemty);
    free(m.promotable);
    free(alloca_res);
    free(alloca_op);
    hmap_free(&m.defOut);
    hmap_free(&m.entryDef);
    hmap_free(&m.bidx);
    hmap_free(&m.aidx);
    hmap_free(&m.remap);
}

// ===========================================================================
// SROA — scalar replacement of aggregates (runs before the scalar promotion)
// ===========================================================================
// tinyC materialises every C local — including aggregates like `string`,
// `Type`, `EVal` — as one entry-block `llvm.alloca`, and accesses each field
// via a constant-index `llvm.getelementptr` + scalar `llvm.load`/`llvm.store`.
// The scalar mem2reg above only promotes allocas whose element type is loaded/
// stored as a whole, so every struct local would otherwise stay in the linear-
// memory shadow stack. This pass splits a non-escaping aggregate alloca into
// one fresh scalar alloca per accessed leaf field, after which the scalar pass
// promotes those scalars to SSA + block-arg phis.
//
// An aggregate alloca is splittable iff the ONLY uses of its pointer are
// `llvm.getelementptr`s (as the base, operand 0) whose indices are all
// constant, whose leading index is 0, and whose leaf type is a scalar; and the
// ONLY uses of each such GEP result are an `llvm.load` (operand 0) or
// `llvm.store` (operand 1) of that exact leaf type. Any other use — the
// pointer escaping to a call/arith/store-as-value, a whole-aggregate load/
// store, a dynamic or chained GEP, or any use inside a nested region — leaves
// the alloca untouched.

#define SROA_MAX_PATH 8

typedef struct {
    MLIR_OpHandle   op;          // the llvm.getelementptr op
    MLIR_ValueHandle res;        // its result (the field pointer)
    size_t          ai;          // owning aggregate-alloca index
    MLIR_TypeHandle leaf;        // scalar leaf type at this path
    int32_t         path[SROA_MAX_PATH];
    int             plen;
    bool            valid;       // path parsed & leaf is scalar
    MLIR_ValueHandle scalar;     // assigned per-field scalar alloca (apply)
} SroaGep;

typedef struct {
    MLIR_Context *ctx;
    // aggregate allocas
    MLIR_OpHandle    *a_op;
    MLIR_BlockHandle *a_blk;
    MLIR_ValueHandle *a_res;
    MLIR_ValueHandle *a_size;    // alloca element-count operand
    MLIR_TypeHandle  *a_elem;    // aggregate element type
    MLIR_TypeHandle  *a_ptrty;   // alloca result type (!llvm.ptr)
    bool             *a_split;   // still splittable
    // per-alloca list of created (path -> scalar) for dedup
    int              *a_nfield;
    int32_t          *a_fpath;   // flat: entry e occupies [e*SROA_MAX_PATH ..]
    int              *a_fplen;
    MLIR_ValueHandle *a_fval;
    int              *a_foff;    // start index into the flat field arrays
    size_t na, acap;
    HMap aidx;                   // alloca result -> ai
    // geps
    SroaGep *geps;
    size_t   ng, gcap;
    HMap gidx;                   // gep result -> gi
} S;

// Parse "array<i32: 0, 1, ...>" into out[]; returns false on overflow/garbage.
static bool sroa_parse_indices(string s, int32_t *out, int cap, int *n) {
    const char *p = s.str;
    const char *e = s.str + s.size;
    while (p < e && *p != ':') p++;
    if (p >= e) return false;
    p++; // skip ':'
    int cnt = 0;
    while (p < e) {
        while (p < e && (*p == ' ' || *p == ',')) p++;
        if (p >= e || *p == '>') break;
        bool neg = false;
        if (*p == '-') { neg = true; p++; }
        if (p >= e || *p < '0' || *p > '9') return false;
        long v = 0;
        while (p < e && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
        if (neg) v = -v;
        if (cnt >= cap) return false;
        out[cnt++] = (int32_t)v;
    }
    *n = cnt;
    return cnt > 0;
}

// Walk an aggregate type along a constant index path (path[0] must be 0, the
// element selector for the single-object alloca). Returns the scalar leaf type
// via *out, or false if the path is dynamic, out of range, terminates on an
// aggregate, or is just the whole object.
static bool sroa_leaf_type(MLIR_TypeHandle agg, const int32_t *idx, int n,
                           MLIR_TypeHandle *out) {
    if (n < 2 || idx[0] != 0) return false;
    MLIR_TypeHandle cur = agg;
    for (int i = 1; i < n; i++) {
        int32_t c = idx[i];
        if (c == (int32_t)0x80000000) return false; // dynamic
        if (MLIR_IsTypeLLVMStruct(cur)) {
            size_t nf = MLIR_GetTypeLLVMStructNumFields(cur);
            if (c < 0 || (size_t)c >= nf) return false;
            cur = MLIR_GetTypeLLVMStructField(cur, (size_t)c);
        } else if (MLIR_IsTypeLLVMArray(cur)) {
            uint64_t ne = MLIR_GetTypeLLVMArrayNumElements(cur);
            if (c < 0 || (uint64_t)c >= ne) return false;
            cur = MLIR_GetTypeLLVMArrayElement(cur);
        } else {
            return false; // indexing into a scalar
        }
    }
    if (MLIR_IsTypeLLVMStruct(cur) || MLIR_IsTypeLLVMArray(cur)) return false;
    *out = cur;
    return true;
}

static size_t sroa_aidx(S *s, MLIR_ValueHandle v) {
    uintptr_t a;
    if (hmap_get(&s->aidx, (uintptr_t)v, &a)) return (size_t)a;
    return (size_t)-1;
}
static size_t sroa_gidx(S *s, MLIR_ValueHandle v) {
    uintptr_t g;
    if (hmap_get(&s->gidx, (uintptr_t)v, &g)) return (size_t)g;
    return (size_t)-1;
}

// Phase 1: classify direct uses of each aggregate alloca; record candidate GEPs.
static void sroa_scan_uses(S *s, MLIR_BlockHandle blk, bool nested) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t j = 0; j < nops; j++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
        MLIR_OpType t = MLIR_GetOpType(op);
        size_t no = MLIR_GetOpNumOperands(op);
        if (!nested && t == OP_TYPE_LLVM_GEP && no >= 1) {
            size_t ai = sroa_aidx(s, MLIR_GetOpOperand(op, 0));
            if (ai != (size_t)-1) {
                // Any aggregate alloca appearing as a non-base operand (a
                // dynamic-index value) would be an escape.
                for (size_t k = 1; k < no; k++)
                    if (sroa_aidx(s, MLIR_GetOpOperand(op, k)) != (size_t)-1)
                        s->a_split[sroa_aidx(s, MLIR_GetOpOperand(op, k))] = false;
                MLIR_AttributeHandle ria = MLIR_GetOpAttributeByName(op, "rawConstantIndices");
                int32_t idx[SROA_MAX_PATH]; int nidx = 0;
                bool okpath = ria != MLIR_INVALID_HANDLE &&
                    sroa_parse_indices(MLIR_GetAttributeAsString(s->ctx, ria),
                                       idx, SROA_MAX_PATH, &nidx);
                MLIR_TypeHandle leaf = MLIR_INVALID_HANDLE;
                // A constant-only GEP has exactly one operand (the base); every
                // index lives in rawConstantIndices. Dynamic indices add operands.
                bool allconst = (no == 1) && okpath;
                bool good = allconst &&
                    sroa_leaf_type(s->a_elem[ai], idx, nidx, &leaf);
                if (!good) { s->a_split[ai] = false; continue; }
                if (s->ng == s->gcap) {
                    s->gcap = s->gcap ? s->gcap * 2 : 256;
                    s->geps = (SroaGep *)realloc(s->geps, s->gcap * sizeof(SroaGep));
                }
                SroaGep *g = &s->geps[s->ng];
                memset(g, 0, sizeof(*g));
                g->op = op;
                g->res = MLIR_GetOpResult(op, 0);
                g->ai = ai;
                g->leaf = leaf;
                g->plen = nidx;
                for (int k = 0; k < nidx; k++) g->path[k] = idx[k];
                g->valid = true;
                hmap_put(&s->gidx, (uintptr_t)g->res, (uintptr_t)s->ng);
                s->ng++;
                continue;
            }
        }
        // Generic case: any operand that is an aggregate alloca (and is not the
        // base of a recognised constant GEP handled above) escapes.
        for (size_t k = 0; k < no; k++) {
            size_t ai = sroa_aidx(s, MLIR_GetOpOperand(op, k));
            if (ai != (size_t)-1) s->a_split[ai] = false;
        }
        size_t nsuc = MLIR_GetOpNumSuccessors(op);
        for (size_t su = 0; su < nsuc; su++) {
            size_t nso = MLIR_GetOpNumSuccessorOperands(op, su);
            for (size_t i = 0; i < nso; i++) {
                size_t ai = sroa_aidx(s, MLIR_GetOpSuccessorOperand(op, su, i));
                if (ai != (size_t)-1) s->a_split[ai] = false;
            }
        }
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, r);
            size_t rnb = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < rnb; b++)
                sroa_scan_uses(s, MLIR_GetRegionBlock(rg, b), true);
        }
    }
}

// Phase 2: every use of a candidate GEP result must be a matching scalar
// load/store; anything else disqualifies the owning alloca.
static void sroa_scan_gep_uses(S *s, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t j = 0; j < nops; j++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
        MLIR_OpType t = MLIR_GetOpType(op);
        size_t no = MLIR_GetOpNumOperands(op);
        for (size_t k = 0; k < no; k++) {
            size_t gi = sroa_gidx(s, MLIR_GetOpOperand(op, k));
            if (gi == (size_t)-1) continue;
            SroaGep *g = &s->geps[gi];
            bool ok = false;
            if (t == OP_TYPE_LLVM_LOAD && k == 0)
                ok = (MLIR_GetValueType(MLIR_GetOpResult(op, 0)) == g->leaf);
            else if (t == OP_TYPE_LLVM_STORE && k == 1)
                ok = (MLIR_GetValueType(MLIR_GetOpOperand(op, 0)) == g->leaf);
            if (!ok) s->a_split[g->ai] = false;
        }
        size_t nsuc = MLIR_GetOpNumSuccessors(op);
        for (size_t su = 0; su < nsuc; su++) {
            size_t nso = MLIR_GetOpNumSuccessorOperands(op, su);
            for (size_t i = 0; i < nso; i++) {
                size_t gi = sroa_gidx(s, MLIR_GetOpSuccessorOperand(op, su, i));
                if (gi != (size_t)-1) s->a_split[s->geps[gi].ai] = false;
            }
        }
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, r);
            size_t rnb = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < rnb; b++)
                sroa_scan_gep_uses(s, MLIR_GetRegionBlock(rg, b));
        }
    }
}

static size_t sroa_block_op_index(MLIR_BlockHandle blk, MLIR_OpHandle op) {
    size_t n = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < n; i++)
        if (MLIR_GetBlockOp(blk, i) == op) return i;
    return (size_t)-1;
}

// Get-or-create the scalar alloca for (ai, path), inserting it right after the
// original aggregate alloca so the element-count operand still dominates it.
static MLIR_ValueHandle sroa_field_alloca(S *s, size_t ai, const int32_t *path,
                                          int plen, MLIR_TypeHandle leaf) {
    int base = s->a_foff[ai];
    for (int f = 0; f < s->a_nfield[ai]; f++) {
        int e = base + f;
        if (s->a_fplen[e] == plen) {
            bool eq = true;
            for (int k = 0; k < plen; k++)
                if (s->a_fpath[e*SROA_MAX_PATH + k] != path[k]) { eq = false; break; }
            if (eq) return s->a_fval[e];
        }
    }
    MLIR_Context *ctx = s->ctx;
    MLIR_LocationHandle loc = l2r_unkloc(ctx);
    MLIR_ValueHandle res = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                                                    s->a_ptrty[ai], (string){0}, loc);
    MLIR_AttributeHandle ea = MLIR_CreateAttributeType(ctx, str_lit("elem_type"), leaf);
    MLIR_TypeHandle rty[1] = { s->a_ptrty[ai] };
    MLIR_ValueHandle rv[1] = { res };
    MLIR_ValueHandle ops[1] = { s->a_size[ai] };
    MLIR_AttributeHandle attrs[1] = { ea };
    MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_LLVM_ALLOCA, str_lit("llvm.alloca"),
                                     attrs, 1, rty, 1, rv, 1, ops, 1, NULL, 0,
                                     loc, MLIR_INVALID_HANDLE, (string){0}, -1);
    size_t at = sroa_block_op_index(s->a_blk[ai], s->a_op[ai]);
    MLIR_InsertBlockOpAtIndex(ctx, s->a_blk[ai], op, at + 1);
    int e = base + s->a_nfield[ai];
    s->a_fplen[e] = plen;
    for (int k = 0; k < plen; k++) s->a_fpath[e*SROA_MAX_PATH + k] = path[k];
    s->a_fval[e] = res;
    s->a_nfield[ai]++;
    return res;
}

// Phase 4: repoint loads/stores from field GEPs to the per-field scalar alloca.
static void sroa_rewrite(S *s, MLIR_BlockHandle blk) {
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t j = 0; j < nops; j++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
        MLIR_OpType t = MLIR_GetOpType(op);
        if (t == OP_TYPE_LLVM_LOAD) {
            size_t gi = sroa_gidx(s, MLIR_GetOpOperand(op, 0));
            if (gi != (size_t)-1 && s->geps[gi].scalar != MLIR_INVALID_HANDLE)
                MLIR_SetOpOperand(s->ctx, op, 0, s->geps[gi].scalar);
        } else if (t == OP_TYPE_LLVM_STORE && MLIR_GetOpNumOperands(op) >= 2) {
            size_t gi = sroa_gidx(s, MLIR_GetOpOperand(op, 1));
            if (gi != (size_t)-1 && s->geps[gi].scalar != MLIR_INVALID_HANDLE)
                MLIR_SetOpOperand(s->ctx, op, 1, s->geps[gi].scalar);
        }
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, r);
            size_t rnb = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < rnb; b++)
                sroa_rewrite(s, MLIR_GetRegionBlock(rg, b));
        }
    }
}

static void sroa_region(MLIR_Context *ctx, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    if (nb == 0) return;

    S s;
    memset(&s, 0, sizeof(s));
    s.ctx = ctx;
    s.acap = 64;
    s.a_op    = malloc(s.acap * sizeof(*s.a_op));
    s.a_blk   = malloc(s.acap * sizeof(*s.a_blk));
    s.a_res   = malloc(s.acap * sizeof(*s.a_res));
    s.a_size  = malloc(s.acap * sizeof(*s.a_size));
    s.a_elem  = malloc(s.acap * sizeof(*s.a_elem));
    s.a_ptrty = malloc(s.acap * sizeof(*s.a_ptrty));
    s.a_split = malloc(s.acap * sizeof(*s.a_split));
    hmap_init(&s.aidx, 64);
    hmap_init(&s.gidx, 256);

    // Collect aggregate (struct/array) allocas with a constant element count.
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(region, bi);
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            if (MLIR_GetOpType(op) != OP_TYPE_LLVM_ALLOCA) continue;
            if (MLIR_GetOpNumOperands(op) < 1) continue;
            MLIR_AttributeHandle ea = MLIR_GetOpAttributeByName(op, "elem_type");
            if (ea == MLIR_INVALID_HANDLE) continue;
            MLIR_TypeHandle et = MLIR_GetAttributeTypeValue(ea);
            if (!MLIR_IsTypeLLVMStruct(et) && !MLIR_IsTypeLLVMArray(et)) continue;
            if (s.na == s.acap) {
                s.acap <<= 1;
                s.a_op    = realloc(s.a_op,    s.acap * sizeof(*s.a_op));
                s.a_blk   = realloc(s.a_blk,   s.acap * sizeof(*s.a_blk));
                s.a_res   = realloc(s.a_res,   s.acap * sizeof(*s.a_res));
                s.a_size  = realloc(s.a_size,  s.acap * sizeof(*s.a_size));
                s.a_elem  = realloc(s.a_elem,  s.acap * sizeof(*s.a_elem));
                s.a_ptrty = realloc(s.a_ptrty, s.acap * sizeof(*s.a_ptrty));
                s.a_split = realloc(s.a_split, s.acap * sizeof(*s.a_split));
            }
            size_t ai = s.na++;
            MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
            s.a_op[ai]    = op;
            s.a_blk[ai]   = blk;
            s.a_res[ai]   = res;
            s.a_size[ai]  = MLIR_GetOpOperand(op, 0);
            s.a_elem[ai]  = et;
            s.a_ptrty[ai] = MLIR_GetValueType(res);
            s.a_split[ai] = true;
            hmap_put(&s.aidx, (uintptr_t)res, (uintptr_t)ai);
        }
    }
    if (s.na == 0) goto done;

    for (size_t bi = 0; bi < nb; bi++)
        sroa_scan_uses(&s, MLIR_GetRegionBlock(region, bi), false);
    for (size_t bi = 0; bi < nb; bi++)
        sroa_scan_gep_uses(&s, MLIR_GetRegionBlock(region, bi));

    bool any = false;
    for (size_t ai = 0; ai < s.na; ai++)
        if (s.a_split[ai]) { any = true; break; }
    if (!any) goto done;

    // Per-alloca field bookkeeping. Total distinct fields across all allocas is
    // bounded by the number of candidate GEPs, so size the flat arrays at ng and
    // give each alloca a contiguous slice sized by its own GEP count.
    s.a_nfield = calloc(s.na, sizeof(int));
    s.a_foff   = malloc(s.na * sizeof(int));
    {
        int *cnt = calloc(s.na, sizeof(int));
        for (size_t gi = 0; gi < s.ng; gi++)
            if (s.geps[gi].valid) cnt[s.geps[gi].ai]++;
        int off = 0;
        for (size_t ai = 0; ai < s.na; ai++) { s.a_foff[ai] = off; off += cnt[ai]; }
        free(cnt);
        size_t tot = (size_t)off + 1;
        s.a_fpath = malloc(tot * SROA_MAX_PATH * sizeof(int32_t));
        s.a_fplen = malloc(tot * sizeof(int));
        s.a_fval  = malloc(tot * sizeof(*s.a_fval));
    }

    // Assign each splittable GEP its per-field scalar alloca (creating allocas).
    for (size_t gi = 0; gi < s.ng; gi++) {
        SroaGep *g = &s.geps[gi];
        if (!g->valid || !s.a_split[g->ai]) { g->scalar = MLIR_INVALID_HANDLE; continue; }
        g->scalar = sroa_field_alloca(&s, g->ai, g->path, g->plen, g->leaf);
    }

    for (size_t bi = 0; bi < nb; bi++)
        sroa_rewrite(&s, MLIR_GetRegionBlock(region, bi));

    // Erase now-dead field GEPs, then the split aggregate allocas.
    for (size_t gi = 0; gi < s.ng; gi++) {
        SroaGep *g = &s.geps[gi];
        if (g->valid && s.a_split[g->ai]) MLIR_EraseOp(ctx, g->op);
    }
    for (size_t ai = 0; ai < s.na; ai++)
        if (s.a_split[ai]) MLIR_EraseOp(ctx, s.a_op[ai]);

    free(s.a_nfield); free(s.a_foff);
    free(s.a_fpath); free(s.a_fplen); free(s.a_fval);
done:
    free(s.a_op); free(s.a_blk); free(s.a_res); free(s.a_size);
    free(s.a_elem); free(s.a_ptrty); free(s.a_split);
    free(s.geps);
    hmap_free(&s.aidx); hmap_free(&s.gidx);
}

void mlir_llvm_mem2reg(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (MLIR_GetOpNumRegions(module) == 0) return;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(mr) == 0) return;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        MLIR_OpType ft = MLIR_GetOpType(op);
        // `func.func` is a registered op (matched by type), but `llvm.func`
        // is UNREGISTERED in the native MLIR API impl, so MLIR_GetOpType
        // returns OP_TYPE_UNREGISTERED for it there — only the upstream C++
        // impl maps it to OP_TYPE_LLVM_FUNC. Variadic functions are emitted
        // as `llvm.func`, so matching by name keeps mem2reg behaviour
        // identical across both API impls (see verify_native_upstream_identical).
        bool is_func = (ft == OP_TYPE_FUNC_FUNC || ft == OP_TYPE_LLVM_FUNC);
        if (!is_func) {
            string on = MLIR_GetOpName(op);
            is_func = (on.size == 9 && memcmp(on.str, "llvm.func", 9) == 0);
        }
        if (!is_func) continue;
        if (MLIR_GetOpNumRegions(op) == 0) continue;
        MLIR_RegionHandle body = MLIR_GetOpRegion(op, 0);
        sroa_region(ctx, body);
        l2r_promote_region(ctx, body);
    }
}

// wmir mem2reg — see mlir_wmir_mem2reg.h for the high-level description.
//
// Braun et al. (2013) "Simple and Efficient Construction of Static Single
// Assignment Form", specialised to the wmir CFG which is already fully built
// (every block's predecessors are known), so all blocks are treated as sealed
// and no incomplete-phi bookkeeping is needed.
//
// The pass is def-use-tracking-agnostic: it never relies on use lists or
// MLIR_ReplaceAllUsesOfValue. Instead it builds its own value->value remap
// table (local_get result -> reaching SSA value) and, after rewriting, sweeps
// every op operand and terminator successor-operand to resolve through it.

#include "mlir_wmir_mem2reg.h"
#include "mlir_op_names.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// --- wasm valtype bytes -----------------------------------------------------
#define M2R_WT_I64 0x7e
#define M2R_WT_F64 0x7c

static MLIR_TypeHandle m2r_vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    if (vt == M2R_WT_I64 || vt == M2R_WT_F64)
        return MLIR_CreateTypeInteger(ctx, 64, true);
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

static int m2r_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

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
    // Fibonacci-ish hash of the pointer-sized key.
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
// during construction. A "value reference" (vref) is a uintptr_t that is either
// a real MLIR_ValueHandle (bit 0 clear: IR handles are 16-byte aligned) or a
// phi id tagged in bit 0. After construction a triviality fixpoint collapses
// redundant phis; only the survivors are materialised as real block arguments.
// This keeps the IR arena cost proportional to the SURVIVING (non-trivial)
// phis, not the (often 50x larger) set of phis the on-demand SSA construction
// would otherwise create. Bit 0 is used (not a high bit) so the scheme is valid
// for 32-bit uintptr_t under the wasm32 self-host build.
#define M2R_PHI_TAG ((uintptr_t)1)
#define M2R_IS_PHI(r) (((uintptr_t)(r) & M2R_PHI_TAG) != 0)
#define M2R_PHI_ID(r) ((size_t)((uintptr_t)(r) >> 1))
#define M2R_MK_PHI(id) ((((uintptr_t)(id)) << 1) | M2R_PHI_TAG)

typedef struct {
    MLIR_Context *ctx;
    size_t nb;                 // number of blocks
    size_t nl;                 // number of locals
    const uint8_t *vt;         // valtype per local (length nl)
    MLIR_BlockHandle *blocks;  // blocks[nb]
    MLIR_OpHandle   *terms;    // terminator op per block (0 if none)
    size_t *pred_off;          // nb+1 prefix offsets into pred_blk/pred_slot
    size_t *pred_blk;          // predecessor block index per edge
    size_t *pred_slot;         // predecessor successor-slot per edge
    // Lightweight phi table (heap, reclaimed at end).
    size_t pcount, pcap;
    size_t *p_block;           // block index of each phi
    size_t *p_local;           // local index of each phi
    uintptr_t **p_ops;         // p_ops[pid][k] = vref from k-th pred (in pred order)
    size_t *p_nops;            // number of operands == preds of p_block
    uintptr_t *p_rep;          // representative vref (self phi-ref if surviving)
    MLIR_ValueHandle *p_handle;// materialised block-arg handle (0 until done)
    HMap defOut;               // (bi*nl+i+1) -> last value (real handle) stored in block
    HMap entryDef;             // (bi*nl+i+1) -> memoised reaching vref at block entry
    HMap bidx;                 // block handle -> block index
    HMap remap;                // local_get result handle -> reaching vref
} M;

static MLIR_LocationHandle m2r_unkloc(MLIR_Context *ctx) {
    return MLIR_CreateLocationUnknown(ctx, (string){0});
}

static size_t m2r_block_index(M *m, MLIR_BlockHandle b) {
    uintptr_t v;
    if (hmap_get(&m->bidx, (uintptr_t)b, &v)) return (size_t)v;
    return (size_t)-1;
}

// Defensive zero materialisation for an unreachable block read (should not
// trigger on a well-formed CFG: the entry block always has its locals seeded
// by the lift prologue, and every other read has a predecessor).
static MLIR_ValueHandle m2r_make_zero(M *m, size_t bi, size_t i) {
    MLIR_Context *ctx = m->ctx;
    MLIR_TypeHandle ty = m2r_vt_to_type(ctx, m->vt[i]);
    MLIR_LocationHandle loc = m2r_unkloc(ctx);
    MLIR_AttributeHandle a =
        MLIR_CreateAttributeInteger(ctx, str_lit("value"), 0, ty);
    MLIR_ValueHandle res =
        MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0, ty,
                                 (string){0}, loc);
    MLIR_TypeHandle rty[1] = { ty };
    MLIR_ValueHandle rv[1] = { res };
    MLIR_AttributeHandle attrs[1] = { a };
    MLIR_OpHandle op = MLIR_CreateOp(
        ctx, OP_TYPE_WMIR_CONST, op_type_to_string(OP_TYPE_WMIR_CONST),
        attrs, 1, rty, 1, rv, 1, NULL, 0, NULL, 0,
        loc, MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_InsertBlockOpAtIndex(ctx, m->blocks[bi], op, 0);
    return res;
}

static uintptr_t m2r_live_in(M *m, size_t i, size_t bi);

static uintptr_t m2r_live_out(M *m, size_t i, size_t bi) {
    uintptr_t d;
    if (hmap_get(&m->defOut, (uintptr_t)(bi * m->nl + i + 1), &d))
        return d;
    return m2r_live_in(m, i, bi);
}

static size_t m2r_new_phi(M *m, size_t bi, size_t i, size_t np) {
    if (m->pcount == m->pcap) {
        size_t nc = m->pcap ? m->pcap * 2 : 1024;
        m->p_block  = (size_t *)realloc(m->p_block, nc * sizeof(size_t));
        m->p_local  = (size_t *)realloc(m->p_local, nc * sizeof(size_t));
        m->p_ops    = (uintptr_t **)realloc(m->p_ops, nc * sizeof(uintptr_t *));
        m->p_nops   = (size_t *)realloc(m->p_nops, nc * sizeof(size_t));
        m->p_rep    = (uintptr_t *)realloc(m->p_rep, nc * sizeof(uintptr_t));
        m->p_handle = (MLIR_ValueHandle *)realloc(m->p_handle,
                                                  nc * sizeof(MLIR_ValueHandle));
        m->pcap = nc;
    }
    size_t pid = m->pcount++;
    m->p_block[pid] = bi;
    m->p_local[pid] = i;
    m->p_nops[pid]  = np;
    m->p_ops[pid]   = (uintptr_t *)malloc((np ? np : 1) * sizeof(uintptr_t));
    m->p_rep[pid]   = M2R_MK_PHI(pid);  // assume surviving until proven trivial
    m->p_handle[pid] = 0;
    return pid;
}

static uintptr_t m2r_live_in(M *m, size_t i, size_t bi) {
    uintptr_t key = (uintptr_t)(bi * m->nl + i + 1);
    uintptr_t e;
    if (hmap_get(&m->entryDef, key, &e)) return e;

    size_t po = m->pred_off[bi];
    size_t pe = m->pred_off[bi + 1];
    size_t np = pe - po;

    if (np == 0) {
        uintptr_t v = (uintptr_t)m2r_make_zero(m, bi, i);
        hmap_put(&m->entryDef, key, v);
        return v;
    }
    if (np == 1) {
        uintptr_t v = m2r_live_out(m, i, m->pred_blk[po]);
        hmap_put(&m->entryDef, key, v);
        return v;
    }

    // Merge point: create a lightweight phi id and memoise it BEFORE recursing
    // so back-edges in loops terminate. Real block arguments are only created
    // later for the phis that survive triviality elimination.
    size_t pid = m2r_new_phi(m, bi, i, np);
    uintptr_t ref = M2R_MK_PHI(pid);
    hmap_put(&m->entryDef, key, ref);
    for (size_t e2 = po; e2 < pe; e2++)
        m->p_ops[pid][e2 - po] = m2r_live_out(m, i, m->pred_blk[e2]);
    return ref;
}

// Follow the phi-representative chain to a real handle or a surviving phi-ref.
static uintptr_t m2r_resolve_phi(M *m, uintptr_t r) {
    for (int guard = 0; guard < (1 << 26) && M2R_IS_PHI(r); guard++) {
        uintptr_t nr = m->p_rep[M2R_PHI_ID(r)];
        if (nr == r) break;
        r = nr;
    }
    return r;
}

// Resolve a value reference that may be a local_get result (via remap) and/or
// a phi-ref, to a concrete materialised value handle.
static MLIR_ValueHandle m2r_concrete(M *m, uintptr_t v) {
    uintptr_t cur = v;
    // Alternate remap-chasing (erased local_get result -> its reaching value)
    // and trivial-phi resolution until a fixpoint. Both are needed because a
    // promoted local's stored value can itself be an erased local_get of
    // another promoted local (e.g. `local.set x, (local.get y)`), and a
    // trivial phi can resolve to such an erased local_get result.
    for (int guard = 0; guard < (1 << 26); guard++) {
        uintptr_t before = cur;
        for (int g2 = 0; g2 < (1 << 24); g2++) {
            if (M2R_IS_PHI(cur)) break;
            uintptr_t nx;
            if (!hmap_get(&m->remap, cur, &nx)) break;
            if (nx == cur) break;
            cur = nx;
        }
        cur = m2r_resolve_phi(m, cur);
        if (cur == before) break;
    }
    if (M2R_IS_PHI(cur)) return m->p_handle[M2R_PHI_ID(cur)];
    return (MLIR_ValueHandle)cur;
}

bool mlir_wmir_mem2reg(MLIR_Context *ctx, MLIR_RegionHandle region,
                       const char *lt_hex, size_t lt_hex_len) {
    size_t nl = lt_hex_len / 2;
    if (nl == 0) return true;  // no locals: nothing to promote

    size_t nb = MLIR_GetRegionNumBlocks(region);
    if (nb == 0) return true;

    M m;
    memset(&m, 0, sizeof(m));
    m.ctx = ctx;
    m.nb = nb;
    m.nl = nl;

    // Decode the hex-encoded valtype byte per local.
    uint8_t *vt = (uint8_t *)malloc(nl ? nl : 1);
    for (size_t i = 0; i < nl; i++)
        vt[i] = (uint8_t)((m2r_hex(lt_hex[2 * i]) << 4) | m2r_hex(lt_hex[2 * i + 1]));
    m.vt = vt;

    m.blocks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    m.terms  = (MLIR_OpHandle *)malloc(nb * sizeof(MLIR_OpHandle));
    hmap_init(&m.defOut, 256);
    hmap_init(&m.entryDef, 256);
    hmap_init(&m.bidx, nb);
    hmap_init(&m.remap, 256);

    for (size_t bi = 0; bi < nb; bi++) {
        m.blocks[bi] = MLIR_GetRegionBlock(region, bi);
        m.terms[bi]  = MLIR_GetBlockTerminator(m.blocks[bi]);
        hmap_put(&m.bidx, (uintptr_t)m.blocks[bi], (uintptr_t)bi);
    }

    // Precompute defOut: last value stored to each local within each block.
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = m.blocks[bi];
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            if (MLIR_GetOpType(op) == OP_TYPE_WMIR_LOCAL_SET) {
                MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, "local_idx");
                size_t idx = a ? (size_t)MLIR_GetAttributeInteger(a) : 0;
                if (idx < nl)
                    hmap_put(&m.defOut, (uintptr_t)(bi * nl + idx + 1),
                             (uintptr_t)MLIR_GetOpOperand(op, 0));
            }
        }
    }

    // Precompute predecessor edges (one per terminator successor slot).
    m.pred_off = (size_t *)calloc(nb + 1, sizeof(size_t));
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_OpHandle t = m.terms[bi];
        if (t == 0) continue;
        size_t ns = MLIR_GetOpNumSuccessors(t);
        for (size_t s = 0; s < ns; s++) {
            size_t k = m2r_block_index(&m, MLIR_GetOpSuccessor(t, s));
            if (k != (size_t)-1) m.pred_off[k + 1]++;
        }
    }
    for (size_t bi = 0; bi < nb; bi++)
        m.pred_off[bi + 1] += m.pred_off[bi];
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
                size_t k = m2r_block_index(&m, MLIR_GetOpSuccessor(t, s));
                if (k == (size_t)-1) continue;
                size_t pos = m.pred_off[k] + fill[k]++;
                m.pred_blk[pos] = bi;
                m.pred_slot[pos] = s;
            }
        }
        free(fill);
    }

    // Rewrite pass: build the remap table and lazily create phi ids.
    uintptr_t *cur = (uintptr_t *)malloc(nl * sizeof(uintptr_t));
    // Collect ops to erase.
    size_t erase_cap = 256, erase_n = 0;
    MLIR_OpHandle *erase = (MLIR_OpHandle *)malloc(erase_cap * sizeof(MLIR_OpHandle));

    for (size_t bi = 0; bi < nb; bi++) {
        for (size_t i = 0; i < nl; i++) cur[i] = 0;
        MLIR_BlockHandle blk = m.blocks[bi];
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            MLIR_OpType t = MLIR_GetOpType(op);
            if (t == OP_TYPE_WMIR_LOCAL_SET) {
                MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, "local_idx");
                size_t idx = a ? (size_t)MLIR_GetAttributeInteger(a) : 0;
                if (idx < nl) cur[idx] = MLIR_GetOpOperand(op, 0);
            } else if (t == OP_TYPE_WMIR_LOCAL_GET) {
                MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, "local_idx");
                size_t idx = a ? (size_t)MLIR_GetAttributeInteger(a) : 0;
                uintptr_t val;
                if (idx < nl) {
                    val = cur[idx];
                    if (val == 0) {
                        val = m2r_live_in(&m, idx, bi);
                        cur[idx] = val;
                    }
                } else {
                    val = (uintptr_t)m2r_make_zero(&m, bi, 0);
                }
                hmap_put(&m.remap, (uintptr_t)MLIR_GetOpResult(op, 0), val);
            } else {
                continue;
            }
            if (erase_n == erase_cap) {
                erase_cap <<= 1;
                erase = (MLIR_OpHandle *)realloc(erase, erase_cap * sizeof(MLIR_OpHandle));
            }
            erase[erase_n++] = op;
            // Creating phis may have inserted ops (zero consts) into other
            // blocks; this block's op count is unaffected by such inserts,
            // but refresh defensively in case bi itself got a zero const.
            nops = MLIR_GetBlockNumOps(blk);
        }
    }

    // Trivial-phi elimination (Braun et al.): a phi whose operands are all the
    // same value (ignoring self-references) is redundant and resolves to that
    // value. Run to a fixpoint over the lightweight phi table; only the phis
    // that survive become real block arguments. This keeps the phi count
    // near-linear and the IR small.
    for (size_t pass = 0; pass < m.pcount + 2; pass++) {
        bool changed = false;
        for (size_t pid = 0; pid < m.pcount; pid++) {
            if (m.p_rep[pid] != M2R_MK_PHI(pid)) continue;  // already collapsed
            uintptr_t self = M2R_MK_PHI(pid);
            uintptr_t same = 0;
            bool trivial = true;
            for (size_t o = 0; o < m.p_nops[pid]; o++) {
                uintptr_t r = m2r_resolve_phi(&m, m.p_ops[pid][o]);
                if (r == self) continue;
                if (same == 0) same = r;
                else if (r != same) { trivial = false; break; }
            }
            if (trivial && same != 0) {
                m.p_rep[pid] = same;
                changed = true;
            }
        }
        if (!changed) break;
    }

    // Materialise surviving phis as block arguments (in pid order so the
    // per-block CSR built below matches the block-argument order).
    size_t *surv_off = (size_t *)calloc(nb + 1, sizeof(size_t));
    for (size_t pid = 0; pid < m.pcount; pid++) {
        if (m2r_resolve_phi(&m, M2R_MK_PHI(pid)) != M2R_MK_PHI(pid)) continue;
        MLIR_TypeHandle ty = m2r_vt_to_type(ctx, m.vt[m.p_local[pid]]);
        m.p_handle[pid] =
            MLIR_AddBlockArgument(ctx, m.blocks[m.p_block[pid]], ty,
                                  m2r_unkloc(ctx));
        surv_off[m.p_block[pid] + 1]++;
    }
    for (size_t bi = 0; bi < nb; bi++) surv_off[bi + 1] += surv_off[bi];
    size_t nsurv = surv_off[nb];
    size_t *surv_pid = (size_t *)malloc((nsurv ? nsurv : 1) * sizeof(size_t));
    {
        size_t *fill = (size_t *)calloc(nb, sizeof(size_t));
        for (size_t pid = 0; pid < m.pcount; pid++) {
            if (m2r_resolve_phi(&m, M2R_MK_PHI(pid)) != M2R_MK_PHI(pid)) continue;
            size_t bi = m.p_block[pid];
            surv_pid[surv_off[bi] + fill[bi]++] = pid;
        }
        free(fill);
    }

    // Final sweep: resolve every regular operand to its concrete value.
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = m.blocks[bi];
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t j = 0; j < nops; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, j);
            size_t no = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < no; k++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
                MLIR_ValueHandle r = m2r_concrete(&m, (uintptr_t)v);
                if (r != v) MLIR_SetOpOperand(ctx, op, k, r);
            }
        }
    }

    // Edge wiring: for every CFG edge into block k, append (in survivor order)
    // the concrete value each surviving phi of k expects from that edge.
    // p_ops[pid][edge-local-index] holds the live-out value the phi recorded
    // for that predecessor (built over pred_off[k]..pred_off[k+1]).
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
                MLIR_ValueHandle r = m2r_concrete(&m, (uintptr_t)v);
                sobuf[i] = r;
                if (r != v) changed = true;
            }
            size_t eidx = e - m.pred_off[k];
            for (size_t s = 0; s < add; s++) {
                size_t pid = surv_pid[surv_off[k] + s];
                sobuf[nso + s] = m2r_concrete(&m, m.p_ops[pid][eidx]);
            }
            if (changed)
                MLIR_SetOpSuccessorOperands(ctx, term, slot, sobuf, total);
        }
    }

    // Erase the now-dead local_get / local_set ops.
    for (size_t i = 0; i < erase_n; i++)
        MLIR_EraseOp(ctx, erase[i]);

    free(sobuf);
    free(surv_off);
    free(surv_pid);
    free(erase);
    free(cur);
    for (size_t pid = 0; pid < m.pcount; pid++) free(m.p_ops[pid]);
    free(m.p_block);
    free(m.p_local);
    free(m.p_ops);
    free(m.p_nops);
    free(m.p_rep);
    free(m.p_handle);
    free(m.pred_blk);
    free(m.pred_slot);
    free(m.pred_off);
    free(m.blocks);
    free(m.terms);
    hmap_free(&m.defOut);
    hmap_free(&m.entryDef);
    hmap_free(&m.bidx);
    hmap_free(&m.remap);
    free(vt);
    return true;
}

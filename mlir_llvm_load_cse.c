// LLVM-dialect load CSE / value numbering (EarlyCSE-style). See header.
//
// IMPORTANT: this runs in the wasm->macho pipeline where the context has
// `no_def_use_tracking = true` (examples/tinyc/driver.c). In that mode
// MLIR_ReplaceAllUsesOfValue is a no-op and MLIR_GetValueNumUses is
// unreliable, so — like mlir_llvm_mem2reg.c — this pass rewrites operands
// itself (an explicit value->value replacement map + a final sweep) and
// computes its own use counts for the dead-code elimination of detached
// address chains.
#include "mlir_llvm_load_cse.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ===========================================================================
// uintptr -> int64 open-addressing map (replacement map, use counts).
// Key 0 is reserved as "empty".
// ===========================================================================
typedef struct { uintptr_t *keys; int64_t *vals; size_t cap, n; } PMap;

static void pmap_init(PMap *m, size_t hint) {
    size_t c = 16; while (c < hint * 2 + 16) c <<= 1;
    m->cap = c; m->n = 0;
    m->keys = (uintptr_t *)calloc(c, sizeof(uintptr_t));
    m->vals = (int64_t *)calloc(c, sizeof(int64_t));
}
static void pmap_free(PMap *m) { free(m->keys); free(m->vals); }
static size_t pmap_slot(uintptr_t *keys, size_t cap, uintptr_t k) {
    size_t i = (size_t)((k >> 4) * 11400714819323198485ull) & (cap - 1);
    while (keys[i] && keys[i] != k) i = (i + 1) & (cap - 1);
    return i;
}
static void pmap_grow(PMap *m) {
    size_t nc = m->cap << 1;
    uintptr_t *nk = (uintptr_t *)calloc(nc, sizeof(uintptr_t));
    int64_t *nv = (int64_t *)calloc(nc, sizeof(int64_t));
    for (size_t i = 0; i < m->cap; i++) if (m->keys[i]) {
        size_t s = pmap_slot(nk, nc, m->keys[i]);
        nk[s] = m->keys[i]; nv[s] = m->vals[i];
    }
    free(m->keys); free(m->vals);
    m->keys = nk; m->vals = nv; m->cap = nc;
}
static void pmap_put(PMap *m, uintptr_t k, int64_t v) {
    if (!k) return;
    if (m->n * 2 >= m->cap) pmap_grow(m);
    size_t s = pmap_slot(m->keys, m->cap, k);
    if (!m->keys[s]) { m->keys[s] = k; m->n++; }
    m->vals[s] = v;
}
static bool pmap_get(PMap *m, uintptr_t k, int64_t *out) {
    if (!k || m->cap == 0) return false;
    size_t s = pmap_slot(m->keys, m->cap, k);
    if (!m->keys[s]) return false;
    *out = m->vals[s]; return true;
}

// ===========================================================================
// Scoped value-number map (chained, push/pop on the dominator-tree walk).
// ===========================================================================
typedef struct {
    char            *key;
    size_t           keylen;
    MLIR_ValueHandle val;
    uint64_t         gen;
    int              next;
} LcEnt;
typedef struct { LcEnt *ents; size_t n, cap; int *buckets; size_t nbuckets; } LcMap;

static uint64_t lcse_hash(const char *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)p[i]; h *= 1099511628211ull; }
    return h;
}
static void lcse_map_init(LcMap *m, size_t nops) {
    size_t nb = 16; while (nb < nops * 2 + 16) nb <<= 1;
    m->nbuckets = nb;
    m->buckets = (int *)malloc(nb * sizeof(int));
    for (size_t i = 0; i < nb; i++) m->buckets[i] = -1;
    m->cap = nops * 2 + 16;
    m->ents = (LcEnt *)malloc(m->cap * sizeof(LcEnt));
    m->n = 0;
}
static void lcse_map_free(LcMap *m) {
    for (size_t i = 0; i < m->n; i++) free(m->ents[i].key);
    free(m->ents); free(m->buckets);
}
static int lcse_lookup(LcMap *m, const char *key, size_t keylen) {
    uint64_t h = lcse_hash(key, keylen) & (m->nbuckets - 1);
    for (int e = m->buckets[h]; e != -1; e = m->ents[e].next)
        if (m->ents[e].keylen == keylen && memcmp(m->ents[e].key, key, keylen) == 0)
            return e;
    return -1;
}
static void lcse_insert(LcMap *m, const char *key, size_t keylen,
                        MLIR_ValueHandle val, uint64_t gen) {
    if (m->n == m->cap) { m->cap *= 2; m->ents = (LcEnt *)realloc(m->ents, m->cap * sizeof(LcEnt)); }
    uint64_t h = lcse_hash(key, keylen) & (m->nbuckets - 1);
    LcEnt *e = &m->ents[m->n];
    e->key = (char *)malloc(keylen ? keylen : 1);
    memcpy(e->key, key, keylen);
    e->keylen = keylen; e->val = val; e->gen = gen;
    e->next = m->buckets[h]; m->buckets[h] = (int)m->n; m->n++;
}
static void lcse_pop_to(LcMap *m, size_t scope) {
    while (m->n > scope) {
        m->n--;
        LcEnt *e = &m->ents[m->n];
        uint64_t h = lcse_hash(e->key, e->keylen) & (m->nbuckets - 1);
        m->buckets[h] = e->next;
        free(e->key);
    }
}

// ===========================================================================
// Key builder.
// ===========================================================================
typedef struct { char *p; size_t n, cap; } KB;
static void kb_init(KB *k) { k->cap = 256; k->p = (char *)malloc(k->cap); k->n = 0; }
static void kb_free(KB *k) { free(k->p); }
static void kb_reset(KB *k) { k->n = 0; }
static void kb_bytes(KB *k, const char *b, size_t n) {
    if (k->n + n > k->cap) { while (k->n + n > k->cap) k->cap *= 2; k->p = (char *)realloc(k->p, k->cap); }
    memcpy(k->p + k->n, b, n); k->n += n;
}
static void kb_u64(KB *k, uint64_t v) { kb_bytes(k, (char *)&v, sizeof(v)); }
static void kb_i64(KB *k, int64_t v)  { kb_bytes(k, (char *)&v, sizeof(v)); }
static void kb_str(KB *k, string s)   { kb_bytes(k, s.str, s.size); kb_bytes(k, "\0", 1); }

// ===========================================================================
// Op classification.
// ===========================================================================
static int lc_name_eq(string s, const char *c) {
    size_t n = strlen(c); return s.size == n && memcmp(s.str, c, n) == 0;
}

// No memory / control side effect: never clobbers an available load. Not
// value-numbered itself (CSE-ing the linmem address chain would break the
// addressing-mode fusion the backend performs); only recognised so it does
// not bump the memory generation.
static bool lc_is_pure(MLIR_OpHandle op) {
    switch (MLIR_GetOpType(op)) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR:  case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_SDIV: case OP_TYPE_LLVM_UDIV:
    case OP_TYPE_LLVM_SREM: case OP_TYPE_LLVM_UREM:
    case OP_TYPE_LLVM_ICMP:
    case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_TRUNC:
    case OP_TYPE_LLVM_PTRTOINT:
    case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ADDRESSOF:
        return true;
    default: break;
    }
    string nm = MLIR_GetOpName(op);
    return lc_name_eq(nm, "llvm.inttoptr") || lc_name_eq(nm, "llvm.select") ||
           lc_name_eq(nm, "llvm.fcmp") || lc_name_eq(nm, "llvm.getelementptr") ||
           lc_name_eq(nm, "llvm.bitcast");
}

// Pure, deterministic value op recursed through when structurally keying a
// load address, and removable by DCE once its address chain is detached.
static bool lc_is_transparent(MLIR_OpHandle op) {
    switch (MLIR_GetOpType(op)) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR:  case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_TRUNC:
    case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_MLIR_CONSTANT:
        return true;
    default: break;
    }
    string nm = MLIR_GetOpName(op);
    return lc_name_eq(nm, "llvm.inttoptr") || lc_name_eq(nm, "llvm.getelementptr") ||
           lc_name_eq(nm, "llvm.bitcast");
}
static bool lc_is_load(MLIR_OpHandle op) { return MLIR_GetOpType(op) == OP_TYPE_LLVM_LOAD; }
static bool lc_is_store(MLIR_OpHandle op) { return MLIR_GetOpType(op) == OP_TYPE_LLVM_STORE; }

// True iff `op` bumps the memory generation (clobbers all available values)
// when encountered during the walk. MUST match lc_process_block exactly: a
// store clobbers; a malformed load (!=1 result) clobbers; a call / unknown
// (non-pure) op clobbers; the block terminator and pure / normal-load ops do
// not. Shared by lc_process_block and the join-clobber precompute so the two
// can never disagree (a weaker precompute would forward stale values).
static bool lc_op_bumps_gen(MLIR_OpHandle op, MLIR_OpHandle term) {
    if (op == term) return false;
    if (lc_is_load(op)) return MLIR_GetOpNumResults(op) != 1;
    if (lc_is_store(op)) return true;
    return !lc_is_pure(op);
}

// Follow the replacement map to the canonical value (flat: replacement targets
// are leaders that are never themselves replaced, but loop defensively).
static MLIR_ValueHandle lc_canon(PMap *repl, MLIR_ValueHandle v) {
    int64_t r; int guard = 0;
    while (pmap_get(repl, (uintptr_t)v, &r) && (MLIR_ValueHandle)r != v && guard++ < 64)
        v = (MLIR_ValueHandle)r;
    return v;
}

static void lc_emit_op_ident(KB *k, MLIR_Context *ctx, MLIR_OpHandle op) {
    kb_str(k, MLIR_GetOpName(op));
    size_t na = MLIR_GetOpNumAttributes(op);
    kb_u64(k, na);
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        kb_str(k, MLIR_GetAttributeName(a));
        kb_u64(k, (uint64_t)MLIR_GetAttributeKind(a));
        kb_i64(k, MLIR_GetAttributeInteger(a));
        kb_str(k, MLIR_GetAttributeAsString(ctx, a));
    }
}

// Encode the structure of the expression producing `v` (canonicalised through
// `repl`). Recursion stops at non-transparent defs (block args, loads, calls),
// encoded as opaque SSA-handle leaves.
static void lc_emit_value(KB *k, MLIR_Context *ctx, PMap *repl, MLIR_ValueHandle v, int depth) {
    v = lc_canon(repl, v);
    MLIR_OpHandle d = (v == MLIR_INVALID_HANDLE) ? MLIR_INVALID_HANDLE : MLIR_GetValueDefiningOp(v);
    if (depth <= 0 || d == MLIR_INVALID_HANDLE || !lc_is_transparent(d)) {
        kb_bytes(k, "#", 1); kb_u64(k, (uint64_t)v); return;
    }
    kb_bytes(k, "(", 1);
    lc_emit_op_ident(k, ctx, d);
    kb_u64(k, (uint64_t)MLIR_GetOpResult_type(d, 0));
    size_t no = MLIR_GetOpNumOperands(d);
    kb_u64(k, no);
    for (size_t i = 0; i < no; i++)
        lc_emit_value(k, ctx, repl, MLIR_GetOpOperand(d, i), depth - 1);
    kb_bytes(k, ")", 1);
}
static void lc_build_load_key(KB *k, MLIR_Context *ctx, PMap *repl, MLIR_OpHandle op) {
    kb_reset(k);
    kb_bytes(k, "L", 1);
    kb_u64(k, (uint64_t)MLIR_GetOpResult_type(op, 0));
    size_t no = MLIR_GetOpNumOperands(op);
    kb_u64(k, no);
    for (size_t i = 0; i < no; i++)
        lc_emit_value(k, ctx, repl, MLIR_GetOpOperand(op, i), 24);
}

// Build the key a single-operand `llvm.load` from this store's address (operand
// 1), producing the stored value's type, WOULD have. A later load with the same
// structural address and type then forwards to the stored value (operand 0).
// Mirrors lc_build_load_key for a 1-operand load exactly so the keys collide.
static void lc_build_store_key(KB *k, MLIR_Context *ctx, PMap *repl, MLIR_OpHandle op) {
    kb_reset(k);
    kb_bytes(k, "L", 1);
    kb_u64(k, (uint64_t)MLIR_GetValueType(MLIR_GetOpOperand(op, 0)));
    kb_u64(k, 1);
    lc_emit_value(k, ctx, repl, MLIR_GetOpOperand(op, 1), 24);
}

// ===========================================================================
// Per-region CFG / dominator state.
// ===========================================================================
typedef struct {
    MLIR_Context     *ctx;
    size_t            nb;
    MLIR_BlockHandle *blocks;
    MLIR_OpHandle    *terms;
    size_t *succ_off, *succ, *pred_off, *pred;
    size_t *rpo; size_t nrpo;
    size_t *po_num, *idom, *dom_off, *dom_child;
    char *blk_write;   // [nb]: block contains a memory clobber (lc_op_bumps_gen)
    char *join_clob;   // [nb]: a >1-pred block whose idom..block region clobbers
} Cfg;

static size_t cfg_bidx(Cfg *c, MLIR_BlockHandle b) {
    for (size_t i = 0; i < c->nb; i++) if (c->blocks[i] == b) return i;
    return SIZE_MAX;
}
static void cfg_rpo(Cfg *c) {
    char *vis = (char *)calloc(c->nb, 1);
    size_t *stk = (size_t *)malloc(c->nb * sizeof(size_t));
    size_t *it = (size_t *)malloc(c->nb * sizeof(size_t));
    size_t *po = (size_t *)malloc(c->nb * sizeof(size_t));
    size_t npo = 0, sp = 0;
    if (c->nb > 0) { stk[sp] = 0; it[sp] = 0; vis[0] = 1; sp++; }
    while (sp > 0) {
        size_t b = stk[sp - 1];
        size_t e0 = c->succ_off[b], e1 = c->succ_off[b + 1];
        if (it[sp - 1] < e1 - e0) {
            size_t s = c->succ[e0 + it[sp - 1]]; it[sp - 1]++;
            if (!vis[s]) { vis[s] = 1; stk[sp] = s; it[sp] = 0; sp++; }
        } else { po[npo++] = b; sp--; }
    }
    for (size_t i = 0; i < c->nb; i++) c->po_num[i] = SIZE_MAX;
    for (size_t i = 0; i < npo; i++) c->po_num[po[i]] = i;
    c->nrpo = npo;
    for (size_t i = 0; i < npo; i++) c->rpo[i] = po[npo - 1 - i];
    free(vis); free(stk); free(it); free(po);
}
static size_t cfg_intersect(Cfg *c, size_t a, size_t b) {
    while (a != b) {
        while (c->po_num[a] < c->po_num[b]) a = c->idom[a];
        while (c->po_num[b] < c->po_num[a]) b = c->idom[b];
    }
    return a;
}
static void cfg_doms(Cfg *c) {
    for (size_t i = 0; i < c->nb; i++) c->idom[i] = SIZE_MAX;
    if (c->nrpo == 0) { c->dom_off = (size_t *)calloc(c->nb + 1, sizeof(size_t));
                        c->dom_child = (size_t *)malloc(sizeof(size_t)); return; }
    size_t entry = c->rpo[0];
    c->idom[entry] = entry;
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < c->nrpo; i++) {
            size_t b = c->rpo[i];
            if (b == entry) continue;
            size_t newidom = SIZE_MAX;
            for (size_t e = c->pred_off[b]; e < c->pred_off[b + 1]; e++) {
                size_t p = c->pred[e];
                if (c->po_num[p] == SIZE_MAX || c->idom[p] == SIZE_MAX) continue;
                newidom = (newidom == SIZE_MAX) ? p : cfg_intersect(c, p, newidom);
            }
            if (newidom != SIZE_MAX && c->idom[b] != newidom) { c->idom[b] = newidom; changed = true; }
        }
    }
    size_t *cnt = (size_t *)calloc(c->nb + 1, sizeof(size_t));
    for (size_t b = 0; b < c->nb; b++)
        if (c->idom[b] != SIZE_MAX && b != entry) cnt[c->idom[b]]++;
    c->dom_off = (size_t *)malloc((c->nb + 1) * sizeof(size_t));
    c->dom_off[0] = 0;
    for (size_t i = 0; i < c->nb; i++) c->dom_off[i + 1] = c->dom_off[i] + cnt[i];
    c->dom_child = (size_t *)malloc((c->dom_off[c->nb] ? c->dom_off[c->nb] : 1) * sizeof(size_t));
    size_t *fill = (size_t *)calloc(c->nb, sizeof(size_t));
    for (size_t b = 0; b < c->nb; b++) {
        if (c->idom[b] == SIZE_MAX || b == entry) continue;
        size_t d = c->idom[b];
        c->dom_child[c->dom_off[d] + fill[d]++] = b;
    }
    free(cnt); free(fill);
}

// Precompute, for every join (>1-predecessor) block b, whether the CFG region
// strictly between idom(b) and b contains a memory clobber. Without MemorySSA
// the walk must, at a join, drop any ancestor value that a store on a sibling
// path could have clobbered before the merge. The unconditional "clobber at
// every join" that this replaces is needlessly pessimistic: it also kills
// store-to-load forwarding across `a||b||c||d` short-circuit chains, whose
// merge blocks are joins but whose sibling paths contain no stores at all.
//
// The between-region is computed by reverse-BFS from b's predecessors, stopping
// before expanding d=idom(b). Because d dominates b, every path entry->b passes
// through d, so the BFS stays within blocks dominated by d and always
// terminates at d. CRUCIAL for loops: a join reached via a back-edge (a loop
// header) is itself re-reached by the reverse walk through the latch, so the
// loop body AND the header are scanned -- loop-carried stores correctly force a
// clobber. Any uncertainty (unknown idom, an unreachable block on the way) is
// resolved conservatively to "clobbers".
static void cfg_join_clob(Cfg *c) {
    c->blk_write = (char *)calloc(c->nb ? c->nb : 1, 1);
    c->join_clob = (char *)calloc(c->nb ? c->nb : 1, 1);
    for (size_t b = 0; b < c->nb; b++) {
        MLIR_OpHandle term = c->terms[b];
        size_t no = MLIR_GetBlockNumOps(c->blocks[b]);
        for (size_t i = 0; i < no; i++)
            if (lc_op_bumps_gen(MLIR_GetBlockOp(c->blocks[b], i), term)) {
                c->blk_write[b] = 1; break;
            }
    }
    char *vis = (char *)malloc(c->nb ? c->nb : 1);
    size_t *q = (size_t *)malloc((c->nb ? c->nb : 1) * sizeof(size_t));
    for (size_t b = 0; b < c->nb; b++) {
        if (c->pred_off[b + 1] - c->pred_off[b] <= 1) continue;   // not a join
        size_t d = c->idom[b];
        if (d == SIZE_MAX) { c->join_clob[b] = 1; continue; }
        memset(vis, 0, c->nb);
        size_t head = 0, tail = 0; bool clob = false;
        for (size_t e = c->pred_off[b]; e < c->pred_off[b + 1] && !clob; e++) {
            size_t p = c->pred[e];
            if (p == d) continue;                       // d's stores already in gen
            if (c->po_num[p] == SIZE_MAX) { clob = true; break; }  // unreachable
            if (!vis[p]) { vis[p] = 1; q[tail++] = p; }
        }
        while (tail > head && !clob) {
            size_t x = q[head++];
            if (c->blk_write[x]) { clob = true; break; }
            for (size_t e = c->pred_off[x]; e < c->pred_off[x + 1]; e++) {
                size_t p = c->pred[e];
                if (p == d) continue;                   // stop at idom
                if (c->po_num[p] == SIZE_MAX) { clob = true; break; }
                if (!vis[p]) { vis[p] = 1; q[tail++] = p; }
            }
        }
        c->join_clob[b] = clob ? 1 : 0;
    }
    free(vis); free(q);
}

// ===========================================================================
// Dominator-tree walk: collect load replacements (no IR mutation yet).
// ===========================================================================
typedef struct {
    Cfg   *c;
    LcMap  map;
    KB     kb;
    PMap   repl;                 // redundant load result -> leader load result
    MLIR_OpHandle *erase; size_t n_erase, cap_erase;   // redundant load ops
    uint64_t gen;
    bool store_fwd;
    bool join_refine;            // conditional join clobber (TINYC_NO_JOIN_REFINE off)
} Walk;

static void walk_mark_erase(Walk *w, MLIR_OpHandle op) {
    if (w->n_erase == w->cap_erase) {
        w->cap_erase = w->cap_erase ? w->cap_erase * 2 : 64;
        w->erase = (MLIR_OpHandle *)realloc(w->erase, w->cap_erase * sizeof(MLIR_OpHandle));
    }
    w->erase[w->n_erase++] = op;
}

static void lc_process_block(Walk *w, size_t bi) {
    Cfg *c = w->c;
    // Restrict load reuse across joins. Without MemorySSA a store on a sibling
    // CFG path that re-merges at a multi-predecessor block can clobber an
    // ancestor value, so a join must drop available values UNLESS the region
    // between idom(b) and b is provably store-free (cfg_join_clob). The
    // store-free case is exactly the `||` short-circuit merge we want to keep
    // forwarding through. Single-predecessor chains are never clobbered here.
    if (c->pred_off[bi + 1] - c->pred_off[bi] > 1 &&
        (!w->join_refine || c->join_clob[bi]))
        w->gen++;
    MLIR_BlockHandle blk = c->blocks[bi];
    size_t nops = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        if (op == c->terms[bi]) continue;               // terminator: no value, no clobber
        if (lc_is_load(op)) {
            if (MLIR_GetOpNumResults(op) != 1) { w->gen++; continue; }
            lc_build_load_key(&w->kb, c->ctx, &w->repl, op);
            int e = lcse_lookup(&w->map, w->kb.p, w->kb.n);
            if (e != -1 && w->map.ents[e].gen == w->gen) {
                pmap_put(&w->repl, (uintptr_t)MLIR_GetOpResult(op, 0),
                         (int64_t)w->map.ents[e].val);
                walk_mark_erase(w, op);
            } else {
                lcse_insert(&w->map, w->kb.p, w->kb.n, MLIR_GetOpResult(op, 0), w->gen);
            }
            continue;
        }
        if (lc_is_store(op)) {
            w->gen++;                                   // store clobbers all available values
            // Store-to-load forwarding: republish addr -> stored value at the
            // new generation so a later must-alias load reuses the value
            // instead of reloading. The gen bump above invalidates every prior
            // entry, so only this store's (addr,value) is forwardable until the
            // next clobber -- exactly must-alias semantics, no alias analysis.
            if (w->store_fwd && MLIR_GetOpNumOperands(op) == 2) {
                // The stored value may itself be a load we CSE'd away; forward
                // to its canonical leader, never the soon-to-be-erased value.
                MLIR_ValueHandle sv = lc_canon(&w->repl, MLIR_GetOpOperand(op, 0));
                // Only forward scalar int/float values. The memfuse addressing
                // spine carries slotless `!llvm.ptr` values; never republish one
                // as a load result (it has no register slot). In the wasm-lifted
                // IR stored values are always i32/i64/f32/f64, but enforce it.
                MLIR_TypeHandle st = MLIR_GetValueType(sv);
                if (MLIR_IsTypeInteger(st) || MLIR_IsTypeFloat(st)) {
                    lc_build_store_key(&w->kb, c->ctx, &w->repl, op);
                    lcse_insert(&w->map, w->kb.p, w->kb.n, sv, w->gen);
                }
            }
            continue;
        }
        if (!lc_is_pure(op)) w->gen++;                  // call / unknown: clobber
    }
}

typedef struct { size_t bi, scope; uint64_t saved_gen; bool exit; } Frame;
static void lc_walk(Walk *w, size_t root) {
    Cfg *c = w->c;
    size_t cap = 64, sp = 0;
    Frame *st = (Frame *)malloc(cap * sizeof(Frame));
    st[sp++] = (Frame){ root, 0, 0, false };
    while (sp > 0) {
        Frame f = st[--sp];
        if (f.exit) { lcse_pop_to(&w->map, f.scope); w->gen = f.saved_gen; continue; }
        size_t scope = w->map.n; uint64_t saved_gen = w->gen;
        lc_process_block(w, f.bi);
        size_t nchild = c->dom_off[f.bi + 1] - c->dom_off[f.bi];
        if (sp + nchild + 2 > cap) { while (sp + nchild + 2 > cap) cap *= 2;
                                     st = (Frame *)realloc(st, cap * sizeof(Frame)); }
        st[sp++] = (Frame){ f.bi, scope, saved_gen, true };
        for (size_t e = c->dom_off[f.bi]; e < c->dom_off[f.bi + 1]; e++)
            st[sp++] = (Frame){ c->dom_child[e], 0, 0, false };
    }
    free(st);
}

// ===========================================================================
// Apply: rewrite operands through `repl`, erase redundant loads, DCE the
// detached address chains. All manual because the context has no def-use
// tracking.
// ===========================================================================
static void lc_sweep_operands(Cfg *c, PMap *repl) {
    for (size_t b = 0; b < c->nb; b++) {
        MLIR_BlockHandle bk = c->blocks[b];
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < non; k++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
                MLIR_ValueHandle r = lc_canon(repl, v);
                if (r != v) MLIR_SetOpOperand(c->ctx, op, k, r);
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                bool any = false;
                for (size_t k = 0; k < nso; k++)
                    if (lc_canon(repl, MLIR_GetOpSuccessorOperand(op, s, k)) !=
                        MLIR_GetOpSuccessorOperand(op, s, k)) { any = true; break; }
                if (!any) continue;
                MLIR_ValueHandle *vals = (MLIR_ValueHandle *)malloc((nso ? nso : 1) * sizeof(MLIR_ValueHandle));
                for (size_t k = 0; k < nso; k++)
                    vals[k] = lc_canon(repl, MLIR_GetOpSuccessorOperand(op, s, k));
                MLIR_SetOpSuccessorOperands(c->ctx, op, s, vals, nso);
                free(vals);
            }
        }
    }
}

// Manual reference counts (op-result value -> number of operand / succ-operand
// uses across the region), then worklist DCE of unused transparent ops seeded
// at the detached address-chain roots.
static void lc_dce(Cfg *c, MLIR_OpHandle *seed_defs, size_t n_seed) {
    PMap uses; pmap_init(&uses, 1024);
    for (size_t b = 0; b < c->nb; b++) {
        MLIR_BlockHandle bk = c->blocks[b];
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < non; k++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
                if (MLIR_GetValueKind(v) == OP_RESULT) {
                    int64_t c0 = 0; pmap_get(&uses, (uintptr_t)v, &c0);
                    pmap_put(&uses, (uintptr_t)v, c0 + 1);
                }
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(op, s, k);
                    if (MLIR_GetValueKind(v) == OP_RESULT) {
                        int64_t c0 = 0; pmap_get(&uses, (uintptr_t)v, &c0);
                        pmap_put(&uses, (uintptr_t)v, c0 + 1);
                    }
                }
            }
        }
    }
    MLIR_OpHandle *wl = NULL; size_t nwl = 0, cwl = 0;
    for (size_t i = 0; i < n_seed; i++) {
        if (seed_defs[i] == MLIR_INVALID_HANDLE) continue;
        if (nwl == cwl) { cwl = cwl ? cwl * 2 : 64; wl = (MLIR_OpHandle *)realloc(wl, cwl * sizeof(MLIR_OpHandle)); }
        wl[nwl++] = seed_defs[i];
    }
    while (nwl > 0) {
        MLIR_OpHandle op = wl[--nwl];
        if (op == MLIR_INVALID_HANDLE) continue;
        if (MLIR_GetOpParentBlock(op) == MLIR_INVALID_HANDLE) continue;   // already erased
        if (MLIR_GetOpNumResults(op) != 1 || !lc_is_transparent(op)) continue;
        int64_t cnt = 0; pmap_get(&uses, (uintptr_t)MLIR_GetOpResult(op, 0), &cnt);
        if (cnt != 0) continue;
        size_t non = MLIR_GetOpNumOperands(op);
        MLIR_OpHandle *defs = (MLIR_OpHandle *)malloc((non ? non : 1) * sizeof(MLIR_OpHandle));
        for (size_t k = 0; k < non; k++) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
            defs[k] = (MLIR_GetValueKind(v) == OP_RESULT) ? MLIR_GetValueDefiningOp(v) : MLIR_INVALID_HANDLE;
            if (MLIR_GetValueKind(v) == OP_RESULT) {
                int64_t c0 = 0; pmap_get(&uses, (uintptr_t)v, &c0);
                if (c0 > 0) pmap_put(&uses, (uintptr_t)v, c0 - 1);
            }
        }
        MLIR_EraseOp(c->ctx, op);
        for (size_t k = 0; k < non; k++) if (defs[k] != MLIR_INVALID_HANDLE) {
            if (nwl == cwl) { cwl = cwl ? cwl * 2 : 64; wl = (MLIR_OpHandle *)realloc(wl, cwl * sizeof(MLIR_OpHandle)); }
            wl[nwl++] = defs[k];
        }
        free(defs);
    }
    free(wl); pmap_free(&uses);
}

static void lc_run_region(MLIR_Context *ctx, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    if (nb == 0) return;

    Cfg c; memset(&c, 0, sizeof(c));
    c.ctx = ctx; c.nb = nb;
    c.blocks = (MLIR_BlockHandle *)malloc(nb * sizeof(MLIR_BlockHandle));
    c.terms  = (MLIR_OpHandle *)malloc(nb * sizeof(MLIR_OpHandle));
    for (size_t i = 0; i < nb; i++) {
        c.blocks[i] = MLIR_GetRegionBlock(region, i);
        c.terms[i]  = MLIR_GetBlockTerminator(c.blocks[i]);
    }
    // Resolve successors -> indices.
    size_t total_ops = 0;
    size_t *rscnt = (size_t *)calloc(nb, sizeof(size_t));
    size_t **rsucc = (size_t **)malloc(nb * sizeof(size_t *));
    for (size_t i = 0; i < nb; i++) {
        total_ops += MLIR_GetBlockNumOps(c.blocks[i]);
        size_t ns = (c.terms[i] != MLIR_INVALID_HANDLE) ? MLIR_GetOpNumSuccessors(c.terms[i]) : 0;
        rsucc[i] = (size_t *)malloc((ns ? ns : 1) * sizeof(size_t));
        for (size_t s = 0; s < ns; s++) {
            size_t si = cfg_bidx(&c, MLIR_GetOpSuccessor(c.terms[i], s));
            if (si != SIZE_MAX) rsucc[i][rscnt[i]++] = si;
        }
    }
    c.succ_off = (size_t *)malloc((nb + 1) * sizeof(size_t));
    c.succ_off[0] = 0;
    for (size_t i = 0; i < nb; i++) c.succ_off[i + 1] = c.succ_off[i] + rscnt[i];
    c.succ = (size_t *)malloc((c.succ_off[nb] ? c.succ_off[nb] : 1) * sizeof(size_t));
    size_t *pcnt = (size_t *)calloc(nb + 1, sizeof(size_t));
    for (size_t i = 0; i < nb; i++)
        for (size_t s = 0; s < rscnt[i]; s++) { c.succ[c.succ_off[i] + s] = rsucc[i][s]; pcnt[rsucc[i][s]]++; }
    c.pred_off = (size_t *)malloc((nb + 1) * sizeof(size_t));
    c.pred_off[0] = 0;
    for (size_t i = 0; i < nb; i++) c.pred_off[i + 1] = c.pred_off[i] + pcnt[i];
    c.pred = (size_t *)malloc((c.pred_off[nb] ? c.pred_off[nb] : 1) * sizeof(size_t));
    size_t *pfill = (size_t *)calloc(nb, sizeof(size_t));
    for (size_t i = 0; i < nb; i++)
        for (size_t e = c.succ_off[i]; e < c.succ_off[i + 1]; e++) {
            size_t s = c.succ[e]; c.pred[c.pred_off[s] + pfill[s]++] = i;
        }
    for (size_t i = 0; i < nb; i++) free(rsucc[i]);
    free(rsucc); free(rscnt);

    c.rpo = (size_t *)malloc(nb * sizeof(size_t));
    c.po_num = (size_t *)malloc(nb * sizeof(size_t));
    c.idom = (size_t *)malloc(nb * sizeof(size_t));
    cfg_rpo(&c);
    cfg_doms(&c);
    cfg_join_clob(&c);

    Walk w; memset(&w, 0, sizeof(w));
    w.c = &c; w.gen = 0;
    w.store_fwd = (getenv("TINYC_NO_STORE_FWD") == NULL);
    w.join_refine = (getenv("TINYC_NO_JOIN_REFINE") == NULL);
    lcse_map_init(&w.map, total_ops);
    kb_init(&w.kb);
    pmap_init(&w.repl, total_ops);
    if (c.nrpo > 0) lc_walk(&w, c.rpo[0]);

    if (w.n_erase > 0) {
        // Capture each redundant load's address-chain root before erasing it,
        // then rewrite all uses, erase the loads, and DCE the dead chains.
        MLIR_OpHandle *roots = (MLIR_OpHandle *)malloc(w.n_erase * sizeof(MLIR_OpHandle));
        for (size_t i = 0; i < w.n_erase; i++) {
            MLIR_ValueHandle p = MLIR_GetOpOperand(w.erase[i], 0);
            roots[i] = (MLIR_GetValueKind(p) == OP_RESULT) ? MLIR_GetValueDefiningOp(p) : MLIR_INVALID_HANDLE;
        }
        lc_sweep_operands(&c, &w.repl);
        for (size_t i = 0; i < w.n_erase; i++) MLIR_EraseOp(ctx, w.erase[i]);
        if (!getenv("TINYC_NO_LOAD_CSE_DCE")) lc_dce(&c, roots, w.n_erase);
        free(roots);
    }

    lcse_map_free(&w.map);
    kb_free(&w.kb);
    pmap_free(&w.repl);
    free(w.erase);
    free(c.blocks); free(c.terms);
    free(pcnt); free(pfill);
    free(c.succ_off); free(c.succ);
    free(c.pred_off); free(c.pred);
    free(c.rpo); free(c.po_num); free(c.idom);
    free(c.dom_off); free(c.dom_child);
    free(c.blk_write); free(c.join_clob);
}

void mlir_llvm_load_cse(MLIR_Context *ctx, MLIR_OpHandle module) {
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
        lc_run_region(ctx, MLIR_GetOpRegion(op, 0));
    }
}

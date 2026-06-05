// LLVM-dialect intra-block arithmetic value numbering (CSE). See header.
//
// IMPORTANT: this runs in the wasm->macho pipeline where the context has
// `no_def_use_tracking = true` (examples/tinyc/driver.c). In that mode
// MLIR_ReplaceAllUsesOfValue is a no-op and MLIR_GetValueNumUses is
// unreliable, so -- like mlir_llvm_mem2reg.c and mlir_llvm_load_cse.c -- this
// pass rewrites operands itself (an explicit value->value replacement map + a
// final sweep) and computes its own use counts for dead-code elimination.
#include "mlir_llvm_arith_gvn.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ===========================================================================
// uintptr -> uintptr open-addressing map (replacement map / spine set / uses).
// Key 0 is reserved as "empty".
// ===========================================================================
typedef struct { uintptr_t *keys; int64_t *vals; size_t cap, n; } AgMap;

static void agmap_init(AgMap *m, size_t hint) {
    m->cap = 64; while (m->cap < hint * 2) m->cap <<= 1;
    m->keys = (uintptr_t *)calloc(m->cap, sizeof(uintptr_t));
    m->vals = (int64_t *)calloc(m->cap, sizeof(int64_t));
    m->n = 0;
}
static void agmap_free(AgMap *m) { free(m->keys); free(m->vals); }
static size_t agmap_slot(uintptr_t *keys, size_t cap, uintptr_t k) {
    size_t i = (size_t)((k * 0x9E3779B97F4A7C15ull) & (cap - 1));
    while (keys[i] && keys[i] != k) i = (i + 1) & (cap - 1);
    return i;
}
static void agmap_grow(AgMap *m) {
    size_t nc = m->cap << 1;
    uintptr_t *nk = (uintptr_t *)calloc(nc, sizeof(uintptr_t));
    int64_t *nv = (int64_t *)calloc(nc, sizeof(int64_t));
    for (size_t i = 0; i < m->cap; i++) if (m->keys[i]) {
        size_t j = agmap_slot(nk, nc, m->keys[i]);
        nk[j] = m->keys[i]; nv[j] = m->vals[i];
    }
    free(m->keys); free(m->vals);
    m->keys = nk; m->vals = nv; m->cap = nc;
}
static void agmap_put(AgMap *m, uintptr_t k, int64_t v) {
    if ((m->n + 1) * 2 >= m->cap) agmap_grow(m);
    size_t i = agmap_slot(m->keys, m->cap, k);
    if (!m->keys[i]) { m->keys[i] = k; m->n++; }
    m->vals[i] = v;
}
static bool agmap_get(AgMap *m, uintptr_t k, int64_t *out) {
    size_t i = agmap_slot(m->keys, m->cap, k);
    if (!m->keys[i]) return false;
    if (out) *out = m->vals[i];
    return true;
}

// ===========================================================================
// Key buffer + string hash map (expression key -> leader value).
// ===========================================================================
typedef struct { char *p; size_t n, cap; } AgKB;
static void agkb_init(AgKB *k) { k->cap = 256; k->p = (char *)malloc(k->cap); k->n = 0; }
static void agkb_free(AgKB *k) { free(k->p); }
static void agkb_reset(AgKB *k) { k->n = 0; }
static void agkb_bytes(AgKB *k, const char *b, size_t n) {
    if (k->n + n > k->cap) { while (k->n + n > k->cap) k->cap <<= 1; k->p = (char *)realloc(k->p, k->cap); }
    memcpy(k->p + k->n, b, n); k->n += n;
}
static void agkb_u64(AgKB *k, uint64_t v) { agkb_bytes(k, (char *)&v, sizeof(v)); }
static void agkb_i64(AgKB *k, int64_t v)  { agkb_bytes(k, (char *)&v, sizeof(v)); }
static void agkb_str(AgKB *k, string s)   { agkb_bytes(k, s.str, s.size); agkb_bytes(k, "\0", 1); }

typedef struct { uint64_t hash; size_t koff, klen; int64_t leader; } AgEnt;
typedef struct {
    AgEnt *ents; size_t n, cap;
    char  *kbytes; size_t kn, kcap;
    int   *buckets; size_t nbuckets;
} AgHash;

static uint64_t ag_hash(const char *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)p[i]; h *= 1099511628211ull; }
    return h;
}
static void aghash_init(AgHash *m, size_t nops) {
    m->cap = 64; m->n = 0;
    m->ents = (AgEnt *)malloc(m->cap * sizeof(AgEnt));
    m->kcap = 1024; m->kn = 0; m->kbytes = (char *)malloc(m->kcap);
    m->nbuckets = 256; while (m->nbuckets < nops * 2) m->nbuckets <<= 1;
    m->buckets = (int *)malloc(m->nbuckets * sizeof(int));
    for (size_t i = 0; i < m->nbuckets; i++) m->buckets[i] = -1;
}
static void aghash_free(AgHash *m) { free(m->ents); free(m->kbytes); free(m->buckets); }
static void aghash_reset(AgHash *m) {
    m->n = 0; m->kn = 0;
    for (size_t i = 0; i < m->nbuckets; i++) m->buckets[i] = -1;
}
// Look up `key`; if present return its leader (out!=NULL), else insert with
// `leader` and return false.
static bool aghash_get_or_put(AgHash *m, const char *key, size_t klen,
                              int64_t leader, int64_t *found) {
    uint64_t h = ag_hash(key, klen);
    size_t b = (size_t)(h & (m->nbuckets - 1));
    // linear scan of the bucket chain via open addressing on buckets[]
    for (size_t probe = b; ; probe = (probe + 1) & (m->nbuckets - 1)) {
        int idx = m->buckets[probe];
        if (idx < 0) {
            // empty bucket -> insert here
            if ((size_t)m->n + 1 > m->cap) { m->cap <<= 1; m->ents = (AgEnt *)realloc(m->ents, m->cap * sizeof(AgEnt)); }
            if (m->kn + klen > m->kcap) { while (m->kn + klen > m->kcap) m->kcap <<= 1; m->kbytes = (char *)realloc(m->kbytes, m->kcap); }
            size_t koff = m->kn; memcpy(m->kbytes + koff, key, klen); m->kn += klen;
            m->ents[m->n].hash = h; m->ents[m->n].koff = koff; m->ents[m->n].klen = klen;
            m->ents[m->n].leader = leader;
            m->buckets[probe] = (int)m->n; m->n++;
            return false;
        }
        AgEnt *e = &m->ents[idx];
        if (e->hash == h && e->klen == klen && memcmp(m->kbytes + e->koff, key, klen) == 0) {
            if (found) *found = e->leader;
            return true;
        }
    }
}

// ===========================================================================
// Op classification.
// ===========================================================================
static int ag_name_eq(string s, const char *c) {
    size_t n = strlen(c); return s.size == n && memcmp(s.str, c, n) == 0;
}

// Pure integer arithmetic that is value-numbered. Conversions and the address
// spine are intentionally excluded (see header).
static bool ag_is_arith(MLIR_OpHandle op) {
    switch (MLIR_GetOpType(op)) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR:  case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_SDIV: case OP_TYPE_LLVM_UDIV:
    case OP_TYPE_LLVM_SREM: case OP_TYPE_LLVM_UREM:
    case OP_TYPE_LLVM_ICMP:
        return true;
    default: return false;
    }
}

// Commutative ops whose operand order is canonicalized in the key.
static bool ag_is_commutative(MLIR_OpHandle op) {
    switch (MLIR_GetOpType(op)) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR: case OP_TYPE_LLVM_XOR:
        return true;
    default: return false;
    }
}

// Side-effect-free ops removable by DCE once detached.
static bool ag_is_dce_pure(MLIR_OpHandle op) {
    if (ag_is_arith(op)) return true;
    switch (MLIR_GetOpType(op)) {
    case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_TRUNC:
    case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ADDRESSOF:
        return true;
    default: break;
    }
    string nm = MLIR_GetOpName(op);
    return ag_name_eq(nm, "llvm.inttoptr") || ag_name_eq(nm, "llvm.bitcast") ||
           ag_name_eq(nm, "llvm.getelementptr");
}

static MLIR_ValueHandle ag_canon(AgMap *repl, MLIR_ValueHandle v) {
    int64_t r; int guard = 0;
    while (agmap_get(repl, (uintptr_t)v, &r) && (MLIR_ValueHandle)r != v && guard++ < 64)
        v = (MLIR_ValueHandle)r;
    return v;
}

// If `v` is produced by an integer `llvm.mlir.constant`, return true and its
// value/type so the key can canonicalize it by VALUE (each constant in the
// llvm dialect is a distinct SSA op, so keying by handle would defeat CSE of
// `mul x, 8` against another `mul x, 8`).
static bool ag_const_operand(MLIR_Context *ctx, MLIR_ValueHandle v,
                             int64_t *val, uint64_t *tykey) {
    MLIR_OpHandle d = (MLIR_GetValueKind(v) == OP_RESULT) ? MLIR_GetValueDefiningOp(v)
                                                          : MLIR_INVALID_HANDLE;
    if (d == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetOpType(d) != OP_TYPE_LLVM_MLIR_CONSTANT) return false;
    if (MLIR_GetOpNumResults(d) != 1) return false;
    MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(d, "value");
    if (va == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetAttributeKind(va) != MLIR_ATTR_KIND_INTEGER) return false;
    *val = MLIR_GetAttributeInteger(va);
    *tykey = (uint64_t)MLIR_GetValueType(MLIR_GetOpResult(d, 0));
    return true;
}

// Encode one operand into the key: integer constants by (type,value), all
// other values by their canonicalized SSA handle.
static void ag_emit_operand(AgKB *k, MLIR_Context *ctx, AgMap *repl, MLIR_ValueHandle v) {
    MLIR_ValueHandle c = ag_canon(repl, v);
    int64_t cv; uint64_t ck;
    if (ag_const_operand(ctx, c, &cv, &ck)) {
        agkb_bytes(k, "K", 1); agkb_u64(k, ck); agkb_i64(k, cv);
    } else {
        agkb_bytes(k, "#", 1); agkb_u64(k, (uint64_t)c);
    }
}

static void ag_emit_op_ident(AgKB *k, MLIR_Context *ctx, MLIR_OpHandle op) {
    agkb_str(k, MLIR_GetOpName(op));
    size_t na = MLIR_GetOpNumAttributes(op);
    agkb_u64(k, na);
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        agkb_str(k, MLIR_GetAttributeName(a));
        agkb_u64(k, (uint64_t)MLIR_GetAttributeKind(a));
        agkb_i64(k, MLIR_GetAttributeInteger(a));
        agkb_str(k, MLIR_GetAttributeAsString(ctx, a));
    }
}

// Build the value-number key for an arithmetic op: opcode + attrs + result type
// + canonicalized operand encodings (commutative ops sort their two operands).
static void ag_build_key(AgKB *k, MLIR_Context *ctx, AgMap *repl, MLIR_OpHandle op) {
    agkb_reset(k);
    agkb_bytes(k, "A", 1);
    ag_emit_op_ident(k, ctx, op);
    agkb_u64(k, (uint64_t)MLIR_GetOpResult_type(op, 0));
    size_t no = MLIR_GetOpNumOperands(op);
    agkb_u64(k, no);
    if (no == 2 && ag_is_commutative(op)) {
        // Encode each operand to a scratch buffer, emit in sorted byte order so
        // `a op b` and `b op a` hash identically.
        AgKB t0, t1; agkb_init(&t0); agkb_init(&t1);
        ag_emit_operand(&t0, ctx, repl, MLIR_GetOpOperand(op, 0));
        ag_emit_operand(&t1, ctx, repl, MLIR_GetOpOperand(op, 1));
        AgKB *lo = &t0, *hi = &t1;
        size_t mn = t0.n < t1.n ? t0.n : t1.n;
        int cmp = memcmp(t0.p, t1.p, mn);
        if (cmp > 0 || (cmp == 0 && t0.n > t1.n)) { lo = &t1; hi = &t0; }
        agkb_bytes(k, lo->p, lo->n);
        agkb_bytes(k, hi->p, hi->n);
        agkb_free(&t0); agkb_free(&t1);
    } else {
        for (size_t i = 0; i < no; i++)
            ag_emit_operand(k, ctx, repl, MLIR_GetOpOperand(op, i));
    }
}

// ===========================================================================
// Region driver.
// ===========================================================================
typedef struct { MLIR_OpHandle *v; size_t n, cap; } AgList;
static void aglist_push(AgList *l, MLIR_OpHandle op) {
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 64; l->v = (MLIR_OpHandle *)realloc(l->v, l->cap * sizeof(MLIR_OpHandle)); }
    l->v[l->n++] = op;
}

// Manual RAUW: rewrite every op operand and terminator successor-operand
// through the replacement map.
static void ag_sweep_operands(MLIR_Context *ctx, MLIR_RegionHandle region, AgMap *repl) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t kk = 0; kk < non; kk++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, kk);
                MLIR_ValueHandle r = ag_canon(repl, v);
                if (r != v) MLIR_SetOpOperand(ctx, op, kk, r);
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                bool any = false;
                for (size_t kk = 0; kk < nso; kk++)
                    if (ag_canon(repl, MLIR_GetOpSuccessorOperand(op, s, kk)) !=
                        MLIR_GetOpSuccessorOperand(op, s, kk)) { any = true; break; }
                if (!any) continue;
                MLIR_ValueHandle *vals = (MLIR_ValueHandle *)malloc((nso ? nso : 1) * sizeof(MLIR_ValueHandle));
                for (size_t kk = 0; kk < nso; kk++)
                    vals[kk] = ag_canon(repl, MLIR_GetOpSuccessorOperand(op, s, kk));
                MLIR_SetOpSuccessorOperands(ctx, op, s, vals, nso);
                free(vals);
            }
        }
    }
}

// Manual use-count DCE: count uses across the region, then erase zero-use pure
// ops transitively, seeded at the operands of the erased redundant ops.
static void ag_dce(MLIR_Context *ctx, MLIR_RegionHandle region,
                   MLIR_OpHandle *seeds, size_t nseed) {
    AgMap uses; agmap_init(&uses, 1024);
    size_t nb = MLIR_GetRegionNumBlocks(region);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t kk = 0; kk < non; kk++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, kk);
                if (MLIR_GetValueKind(v) == OP_RESULT) {
                    int64_t c0 = 0; agmap_get(&uses, (uintptr_t)v, &c0);
                    agmap_put(&uses, (uintptr_t)v, c0 + 1);
                }
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t kk = 0; kk < nso; kk++) {
                    MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(op, s, kk);
                    if (MLIR_GetValueKind(v) == OP_RESULT) {
                        int64_t c0 = 0; agmap_get(&uses, (uintptr_t)v, &c0);
                        agmap_put(&uses, (uintptr_t)v, c0 + 1);
                    }
                }
            }
        }
    }
    MLIR_OpHandle *wl = NULL; size_t nwl = 0, cwl = 0;
    for (size_t i = 0; i < nseed; i++) {
        if (seeds[i] == MLIR_INVALID_HANDLE) continue;
        if (nwl == cwl) { cwl = cwl ? cwl * 2 : 64; wl = (MLIR_OpHandle *)realloc(wl, cwl * sizeof(MLIR_OpHandle)); }
        wl[nwl++] = seeds[i];
    }
    while (nwl > 0) {
        MLIR_OpHandle op = wl[--nwl];
        if (op == MLIR_INVALID_HANDLE) continue;
        if (MLIR_GetOpParentBlock(op) == MLIR_INVALID_HANDLE) continue;   // erased
        if (MLIR_GetOpNumResults(op) != 1 || !ag_is_dce_pure(op)) continue;
        int64_t cnt = 0; agmap_get(&uses, (uintptr_t)MLIR_GetOpResult(op, 0), &cnt);
        if (cnt != 0) continue;
        size_t non = MLIR_GetOpNumOperands(op);
        MLIR_OpHandle *defs = (MLIR_OpHandle *)malloc((non ? non : 1) * sizeof(MLIR_OpHandle));
        for (size_t kk = 0; kk < non; kk++) {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, kk);
            defs[kk] = (MLIR_GetValueKind(v) == OP_RESULT) ? MLIR_GetValueDefiningOp(v) : MLIR_INVALID_HANDLE;
            if (MLIR_GetValueKind(v) == OP_RESULT) {
                int64_t c0 = 0; agmap_get(&uses, (uintptr_t)v, &c0);
                if (c0 > 0) agmap_put(&uses, (uintptr_t)v, c0 - 1);
            }
        }
        MLIR_EraseOp(ctx, op);
        for (size_t kk = 0; kk < non; kk++) if (defs[kk] != MLIR_INVALID_HANDLE) {
            if (nwl == cwl) { cwl = cwl ? cwl * 2 : 64; wl = (MLIR_OpHandle *)realloc(wl, cwl * sizeof(MLIR_OpHandle)); }
            wl[nwl++] = defs[kk];
        }
        free(defs);
    }
    free(wl); agmap_free(&uses);
}

static void ag_run_region(MLIR_Context *ctx, MLIR_RegionHandle region) {
    size_t nb = MLIR_GetRegionNumBlocks(region);
    if (nb == 0) return;
    size_t total_ops = 0;
    for (size_t b = 0; b < nb; b++) total_ops += MLIR_GetBlockNumOps(MLIR_GetRegionBlock(region, b));
    if (total_ops == 0) return;

    // Spine-exclusion set: every value that is the operand of an `llvm.inttoptr`
    // (the `add(base, zext(idx))` the backend fuses). Numbering those would make
    // the spine multi-use and disable the `[x28, Widx, UXTW]` fusion.
    AgMap spine; agmap_init(&spine, 64);
    for (size_t b = 0; b < nb; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (!ag_name_eq(MLIR_GetOpName(op), "llvm.inttoptr")) continue;
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t kk = 0; kk < non; kk++)
                agmap_put(&spine, (uintptr_t)MLIR_GetOpOperand(op, kk), 1);
        }
    }

    AgMap repl; agmap_init(&repl, total_ops);
    AgHash tbl; aghash_init(&tbl, total_ops);
    AgKB kb; agkb_init(&kb);
    AgList redundant; memset(&redundant, 0, sizeof(redundant));

    for (size_t b = 0; b < nb; b++) {
        aghash_reset(&tbl);                       // intra-block: reset per block
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(region, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (!ag_is_arith(op) || MLIR_GetOpNumResults(op) != 1) continue;
            MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
            int64_t dummy;
            if (agmap_get(&spine, (uintptr_t)res, &dummy)) continue;   // spine add
            ag_build_key(&kb, ctx, &repl, op);
            int64_t leader;
            if (aghash_get_or_put(&tbl, kb.p, kb.n, (int64_t)res, &leader)) {
                // Redundant: forward this result to the earlier identical value.
                agmap_put(&repl, (uintptr_t)res, leader);
                aglist_push(&redundant, op);
            }
        }
    }

    if (redundant.n > 0) {
        // Seed DCE with the operands of every redundant op (their chains may
        // become dead once the redundant ops are erased).
        AgList seedlist; memset(&seedlist, 0, sizeof(seedlist));
        for (size_t i = 0; i < redundant.n; i++) {
            MLIR_OpHandle op = redundant.v[i];
            size_t non = MLIR_GetOpNumOperands(op);
            for (size_t kk = 0; kk < non; kk++) {
                MLIR_ValueHandle v = MLIR_GetOpOperand(op, kk);
                if (MLIR_GetValueKind(v) == OP_RESULT)
                    aglist_push(&seedlist, MLIR_GetValueDefiningOp(v));
            }
        }
        ag_sweep_operands(ctx, region, &repl);
        for (size_t i = 0; i < redundant.n; i++) MLIR_EraseOp(ctx, redundant.v[i]);
        ag_dce(ctx, region, seedlist.v, seedlist.n);
        free(seedlist.v);
    }

    free(redundant.v);
    agkb_free(&kb);
    aghash_free(&tbl);
    agmap_free(&repl);
    agmap_free(&spine);
}

void mlir_llvm_arith_gvn(MLIR_Context *ctx, MLIR_OpHandle module) {
    if (getenv("TINYC_NO_ARITH_GVN")) return;
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
        ag_run_region(ctx, MLIR_GetOpRegion(op, 0));
    }
}

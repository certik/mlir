// The op-type table of a context (mlir_op_names.h): the names of the static
// types of mlir_ops.def and of the dynamic types the context registers, and
// the hash that maps a name back to its type.

#include <string.h>
#include <base/arena.h>
#include <base/assert.h>
#include <base/string.h>
#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_printer.h"

static uint32_t op_name_hash(string name) {
    uint32_t h = 2166136261u;  // FNV-1a
    for (size_t i = 0; i < name.size; i++) {
        h ^= (uint8_t)name.str[i];
        h *= 16777619u;
    }
    return h;
}

// The name a hash slot stands for: a type's name or an alias.
static string op_type_table_slot_name(const OpTypeTable *t, uint32_t slot) {
    if (slot & OP_TYPE_ALIAS_SLOT) return t->alias_names[slot & ~OP_TYPE_ALIAS_SLOT];
    return t->names[slot];
}

// Adds `slot` (a type, or an alias) under its name to the hash.
static void op_type_table_insert(OpTypeTable *t, uint32_t slot) {
    string name = op_type_table_slot_name(t, slot);
    uint32_t mask = t->n_slots - 1;
    uint32_t i = op_name_hash(name) & mask;
    while (t->slots[i] != 0) {
        // Two entries with one name would make the name ambiguous.
        assert(!str_eq(op_type_table_slot_name(t, t->slots[i]), name));
        i = (i + 1) & mask;
    }
    t->slots[i] = slot;
}

static void op_type_table_rehash(OpTypeTable *t, uint32_t n_slots) {
    t->slots = arena_new_array(t->arena, uint32_t, n_slots);
    memset(t->slots, 0, sizeof(uint32_t) * (size_t)n_slots);
    t->n_slots = n_slots;
    for (uint32_t type = 1; type < t->n_types; type++) op_type_table_insert(t, type);
    for (uint32_t a = 0; a < t->n_aliases; a++) op_type_table_insert(t, OP_TYPE_ALIAS_SLOT | a);
}

static void op_type_table_add_alias(OpTypeTable *t, MLIR_OpType type, string name) {
    assert(t->n_aliases < OP_TYPE_ALIAS_CAPACITY);
    t->alias_names[t->n_aliases] = name;
    t->alias_types[t->n_aliases] = type;
    t->n_aliases++;
}

// The names of the static types, in enum order, each followed by a NUL. A
// string rather than an array of strings: tinyC, which compiles this file
// when self-hosting, accepts no pointer in a global initializer.
static const char g_static_op_names[] =
#define MLIR_OP(enumerator, name, traits) name "\0"
#include "mlir_ops.def"
    ;

// The trait bitmask of each static op type (dynamic types return MLIR_T_NONE).
// Uses a switch rather than a global initializer array because tinyC's
// self-host parser does not accept bitwise expressions in global array inits;
// optimizing compilers lower this dense switch to a static lookup table.
MLIR_OpTraits MLIR_GetOpTypeTraits(MLIR_OpType type) {
    switch (type) {
#define MLIR_OP(enumerator, name, traits) case enumerator: return (MLIR_OpTraits)(traits);
#include "mlir_ops.def"
    default:
        return MLIR_T_NONE;
    }
}

void op_type_table_init(OpTypeTable *t, Arena *arena) {
    t->arena = arena;
    t->cap_types = (uint32_t)OP_TYPE_COUNT + 64;
    t->names = arena_new_array(arena, string, t->cap_types);
    t->n_types = (uint32_t)OP_TYPE_COUNT;
    t->names[OP_TYPE_UNREGISTERED].str = (char *)g_static_op_names + sizeof(g_static_op_names) - 1;
    t->names[OP_TYPE_UNREGISTERED].size = 0;
    const char *p = g_static_op_names;
    for (uint32_t type = 1; type < t->n_types; type++) {
        size_t n = strlen(p);
        t->names[type].str = (char *)p;
        t->names[type].size = n;
        p += n + 1;
    }
    // One name per enumerator: the walk ends at the string's final NUL.
    assert(p == g_static_op_names + sizeof(g_static_op_names) - 1);
    t->alias_names = arena_new_array(arena, string, OP_TYPE_ALIAS_CAPACITY);
    t->alias_types = arena_new_array(arena, MLIR_OpType, OP_TYPE_ALIAS_CAPACITY);
    t->n_aliases = 0;
#define MLIR_OP_ALIAS(enumerator, name) op_type_table_add_alias(t, enumerator, str_lit(name));
#include "mlir_ops.def"
    op_type_table_rehash(t, 1024);
}

MLIR_OpType op_type_table_lookup(const OpTypeTable *t, string name) {
    if (name.size == 0) return OP_TYPE_UNREGISTERED;
    uint32_t mask = t->n_slots - 1;
    uint32_t i = op_name_hash(name) & mask;
    uint32_t slot;
    while ((slot = t->slots[i]) != 0) {
        if (str_eq(op_type_table_slot_name(t, slot), name))
            return (slot & OP_TYPE_ALIAS_SLOT)
                ? t->alias_types[slot & ~OP_TYPE_ALIAS_SLOT] : (MLIR_OpType)slot;
        i = (i + 1) & mask;
    }
    return OP_TYPE_UNREGISTERED;
}

MLIR_OpType op_type_table_register(OpTypeTable *t, string name) {
    MLIR_OpType found = op_type_table_lookup(t, name);
    if (found != OP_TYPE_UNREGISTERED || name.size == 0) return found;
    if (t->n_types == t->cap_types) {
        uint32_t cap = 2 * t->cap_types;
        string *names = arena_new_array(t->arena, string, cap);
        memcpy(names, t->names, sizeof(string) * (size_t)t->n_types);
        t->names = names;
        t->cap_types = cap;
    }
    char *copy = (char *)arena_alloc(t->arena, name.size);
    memcpy(copy, name.str, name.size);
    uint32_t type = t->n_types++;
    t->names[type].str = copy;
    t->names[type].size = name.size;
    if (2 * (t->n_types + t->n_aliases) > t->n_slots) op_type_table_rehash(t, 2 * t->n_slots);
    else op_type_table_insert(t, type);
    return (MLIR_OpType)type;
}

// -----------------------------------------------------------------------------
// Attribute keys (mlir_op_names.h).
// -----------------------------------------------------------------------------

static void attr_key_table_insert(AttrKeyTable *t, uint32_t key) {
    string name = t->names[key];
    uint32_t mask = t->n_slots - 1;
    uint32_t i = op_name_hash(name) & mask;
    while (t->slots[i] != 0) {
        assert(!str_eq(t->names[t->slots[i]], name));
        i = (i + 1) & mask;
    }
    t->slots[i] = key;
}

static void attr_key_table_rehash(AttrKeyTable *t, uint32_t n_slots) {
    t->slots = arena_new_array(t->arena, uint32_t, n_slots);
    memset(t->slots, 0, sizeof(uint32_t) * (size_t)n_slots);
    t->n_slots = n_slots;
    for (uint32_t k = 1; k < t->n_keys; k++) attr_key_table_insert(t, k);
}

// The names of the static keys, in enum order, each followed by a NUL (no
// pointers in a global initializer, for tinyC).
static const char g_static_key_names[] =
#define MLIR_ATTR_KEY(enumerator, name) name "\0"
#include "mlir_attr_keys.def"
    ;

void attr_key_table_init(AttrKeyTable *t, Arena *arena) {
    t->arena = arena;
    t->cap_keys = (uint32_t)MLIR_KEY_COUNT + 64;
    t->names = arena_new_array(arena, string, t->cap_keys);
    t->n_keys = (uint32_t)MLIR_KEY_COUNT;
    t->names[MLIR_KEY_INVALID].str = (char *)g_static_key_names + sizeof(g_static_key_names) - 1;
    t->names[MLIR_KEY_INVALID].size = 0;
    const char *p = g_static_key_names;
    for (uint32_t k = 1; k < t->n_keys; k++) {
        size_t n = strlen(p);
        t->names[k].str = (char *)p;
        t->names[k].size = n;
        p += n + 1;
    }
    assert(p == g_static_key_names + sizeof(g_static_key_names) - 1);
    attr_key_table_rehash(t, 512);
}

void attr_key_table_init_strings(AttrKeyTable *t, Arena *arena) {
    t->arena = arena;
    t->cap_keys = 64;
    t->names = arena_new_array(arena, string, t->cap_keys);
    t->n_keys = 1;
    t->names[0].str = (char *)g_static_key_names + sizeof(g_static_key_names) - 1;
    t->names[0].size = 0;
    attr_key_table_rehash(t, 128);
}

MLIR_AttrKey attr_key_table_lookup(const AttrKeyTable *t, string name) {
    if (name.size == 0) return MLIR_KEY_INVALID;
    uint32_t mask = t->n_slots - 1;
    uint32_t i = op_name_hash(name) & mask;
    uint32_t k;
    while ((k = t->slots[i]) != 0) {
        if (str_eq(t->names[k], name)) return k;
        i = (i + 1) & mask;
    }
    return MLIR_KEY_INVALID;
}

MLIR_AttrKey attr_key_table_intern(AttrKeyTable *t, string name) {
    MLIR_AttrKey found = attr_key_table_lookup(t, name);
    if (found != MLIR_KEY_INVALID || name.size == 0) return found;
    if (t->n_keys == t->cap_keys) {
        uint32_t cap = 2 * t->cap_keys;
        string *names = arena_new_array(t->arena, string, cap);
        memcpy(names, t->names, sizeof(string) * (size_t)t->n_keys);
        t->names = names;
        t->cap_keys = cap;
    }
    char *copy = (char *)arena_alloc(t->arena, name.size);
    memcpy(copy, name.str, name.size);
    uint32_t k = t->n_keys++;
    t->names[k].str = copy;
    t->names[k].size = name.size;
    if (2 * t->n_keys > t->n_slots) attr_key_table_rehash(t, 2 * t->n_slots);
    else attr_key_table_insert(t, k);
    return k;
}

string attr_key_table_name(const AttrKeyTable *t, MLIR_AttrKey key) {
    if (key < t->n_keys) return t->names[key];
    return t->names[MLIR_KEY_INVALID];
}

// -----------------------------------------------------------------------------
// Parser-only op kinds (mlir_printer.h). Here rather than inline in the
// header because tinyC emits every static inline function of a header into
// each file that includes it.
// -----------------------------------------------------------------------------

void parser_op_types_init(MLIR_Context *ctx, ParserOpTypes *p) {
#define MLIR_PARSER_OP(kind, name) \
    p->type[(int)kind - (int)OP_TYPE_COUNT] = MLIR_RegisterOpType(ctx, str_lit(name));
#define MLIR_PRINTER_OP(kind, name) \
    p->type[(int)kind - (int)OP_TYPE_COUNT] = MLIR_RegisterOpType(ctx, str_lit(name));
#include "mlir_ops.def"
    int n = (int)PARSER_OP_END_ - (int)OP_TYPE_COUNT;
    p->consecutive = true;
    for (int i = 0; i < n; i++)
        if ((int)p->type[i] != (int)p->type[0] + i) p->consecutive = false;
}

int parser_op_kind(const ParserOpTypes *p, MLIR_OpType t) {
    if ((unsigned)t < (unsigned)OP_TYPE_COUNT) return (int)t;
    int n = (int)PARSER_OP_END_ - (int)OP_TYPE_COUNT;
    if (p->consecutive) {
        unsigned i = (unsigned)t - (unsigned)p->type[0];
        return i < (unsigned)n ? (int)OP_TYPE_COUNT + (int)i : (int)OP_TYPE_UNREGISTERED;
    }
    for (int i = 0; i < n; i++)
        if (p->type[i] == t) return (int)OP_TYPE_COUNT + i;
    return (int)OP_TYPE_UNREGISTERED;
}

// -----------------------------------------------------------------------------
// Backend-agnostic traversal, walks and CFG info (#352)
// -----------------------------------------------------------------------------

MLIR_ValueHandle MLIR_GetIncomingValue(MLIR_Context *ctx, MLIR_PredHandle pred, size_t arg_idx) {
    MLIR_OpHandle term = MLIR_GetPredTerminator(ctx, pred);
    size_t s = MLIR_GetPredSuccessorIndex(ctx, pred);
    return MLIR_GetOpSuccessorOperand(ctx, term, s, arg_idx);
}

void MLIR_SetIncomingValue(MLIR_Context *ctx, MLIR_PredHandle pred,
                           size_t arg_idx, MLIR_ValueHandle value) {
    MLIR_OpHandle term = MLIR_GetPredTerminator(ctx, pred);
    size_t s = MLIR_GetPredSuccessorIndex(ctx, pred);
    size_t first = 0, count = 0;
    MLIR_GetOpSuccessorOperandRange(ctx, term, s, &first, &count);
    assert(arg_idx < count);
    MLIR_SetOpOperand(ctx, term, first + arg_idx, value);
}

MLIR_WalkResult MLIR_Walk(MLIR_Context *ctx, MLIR_OpHandle root,
                          MLIR_WalkOrder order, MLIR_WalkFn fn, void *user) {
    if (order == MLIR_WALK_PRE) {
        MLIR_WalkResult r = fn(ctx, root, user);
        if (r == MLIR_WALK_INTERRUPT) return MLIR_WALK_INTERRUPT;
        if (r == MLIR_WALK_SKIP) return MLIR_WALK_ADVANCE;
    }
    size_t n_reg = MLIR_GetOpNumRegions(ctx, root);
    for (size_t ri = 0; ri < n_reg; ri++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, root, ri);
        for (MLIR_BlockHandle blk = MLIR_GetRegionFirstBlock(ctx, reg);
             blk != MLIR_INVALID_HANDLE;
             blk = MLIR_GetBlockNext(ctx, blk)) {
            MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, blk);
            while (op != MLIR_INVALID_HANDLE) {
                MLIR_OpHandle next = MLIR_GetOpNext(ctx, op);
                MLIR_WalkResult r = MLIR_Walk(ctx, op, order, fn, user);
                if (r == MLIR_WALK_INTERRUPT) return MLIR_WALK_INTERRUPT;
                op = next;
            }
        }
    }
    if (order == MLIR_WALK_POST) {
        MLIR_WalkResult r = fn(ctx, root, user);
        if (r == MLIR_WALK_INTERRUPT) return MLIR_WALK_INTERRUPT;
    }
    return MLIR_WALK_ADVANCE;
}

// Small open-addressing hash map from MLIR_BlockHandle -> int32_t (0..N-1).
typedef struct {
    MLIR_BlockHandle *keys;
    int32_t          *vals;
    size_t            cap; // power of two
} CfgBlkMap;

static size_t cfg_blk_hash(MLIR_BlockHandle b) {
    uint64_t x = (uint64_t)b;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return (size_t)x;
}

static void cfg_map_init(CfgBlkMap *m, Arena *a, size_t n_hint) {
    size_t cap = 16;
    while (cap < n_hint * 2 + 4) cap *= 2;
    m->keys = arena_new_array(a, MLIR_BlockHandle, cap);
    m->vals = arena_new_array(a, int32_t, cap);
    memset(m->keys, 0, cap * sizeof(MLIR_BlockHandle));
    m->cap = cap;
}

static int32_t cfg_map_get(const CfgBlkMap *m, MLIR_BlockHandle b) {
    size_t mask = m->cap - 1;
    size_t i = cfg_blk_hash(b) & mask;
    while (m->keys[i] != MLIR_INVALID_HANDLE) {
        if (m->keys[i] == b) return m->vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}

static void cfg_map_put(CfgBlkMap *m, MLIR_BlockHandle b, int32_t val) {
    size_t mask = m->cap - 1;
    size_t i = cfg_blk_hash(b) & mask;
    while (m->keys[i] != MLIR_INVALID_HANDLE && m->keys[i] != b) {
        i = (i + 1) & mask;
    }
    m->keys[i] = b;
    m->vals[i] = val;
}

size_t MLIR_GetBlockOrder(MLIR_Context *ctx, MLIR_RegionHandle region,
                          MLIR_BlockOrder order, Arena *out,
                          MLIR_BlockHandle **out_blocks) {
    MLIR_BlockHandle entry = MLIR_GetRegionFirstBlock(ctx, region);
    if (entry == MLIR_INVALID_HANDLE) {
        if (out_blocks) *out_blocks = NULL;
        return 0;
    }
    size_t n_total = 0;
    for (MLIR_BlockHandle b = entry; b != MLIR_INVALID_HANDLE; b = MLIR_GetBlockNext(ctx, b))
        n_total++;

    CfgBlkMap visited;
    cfg_map_init(&visited, out, n_total);
    MLIR_BlockHandle *stk_blk = arena_new_array(out, MLIR_BlockHandle, n_total);
    size_t *stk_succ = arena_new_array(out, size_t, n_total);
    MLIR_BlockHandle *po = arena_new_array(out, MLIR_BlockHandle, n_total);
    size_t sp = 0, n_po = 0;

    cfg_map_put(&visited, entry, 1);
    stk_blk[sp] = entry;
    stk_succ[sp] = 0;
    sp++;

    while (sp > 0) {
        MLIR_BlockHandle cur = stk_blk[sp - 1];
        MLIR_OpHandle term = MLIR_GetBlockLastOp(ctx, cur);
        size_t ns = (term != MLIR_INVALID_HANDLE) ? MLIR_GetOpNumSuccessors(ctx, term) : 0;
        if (stk_succ[sp - 1] < ns) {
            MLIR_BlockHandle succ = MLIR_GetOpSuccessor(ctx, term, stk_succ[sp - 1]++);
            if (succ != MLIR_INVALID_HANDLE && cfg_map_get(&visited, succ) < 0) {
                cfg_map_put(&visited, succ, 1);
                stk_blk[sp] = succ;
                stk_succ[sp] = 0;
                sp++;
            }
        } else {
            sp--;
            po[n_po++] = cur;
        }
    }

    if (order == MLIR_ORDER_RPO) {
        for (size_t i = 0; i < n_po / 2; i++) {
            MLIR_BlockHandle tmp = po[i];
            po[i] = po[n_po - 1 - i];
            po[n_po - 1 - i] = tmp;
        }
    }
    if (out_blocks) *out_blocks = po;
    return n_po;
}

static int32_t cfg_intersect(const int32_t *idom, int32_t b1, int32_t b2) {
    while (b1 != b2) {
        while (b1 > b2) b1 = idom[b1];
        while (b2 > b1) b2 = idom[b2];
    }
    return b1;
}

MLIR_CFGInfo *MLIR_ComputeCFGInfo(MLIR_Context *ctx, MLIR_RegionHandle region, Arena *out) {
    MLIR_CFGInfo *cfg = arena_new(out, MLIR_CFGInfo);
    cfg->n_blocks = 0;
    cfg->rpo = NULL;
    cfg->idom = NULL;
    cfg->loop_depth = NULL;
    cfg->is_loop_header = NULL;

    MLIR_BlockHandle *rpo = NULL;
    size_t n = MLIR_GetBlockOrder(ctx, region, MLIR_ORDER_RPO, out, &rpo);
    if (n == 0) return cfg;

    CfgBlkMap rpo_idx;
    cfg_map_init(&rpo_idx, out, n);
    for (size_t i = 0; i < n; i++) cfg_map_put(&rpo_idx, rpo[i], (int32_t)i);

    int32_t *idom = arena_new_array(out, int32_t, n);
    for (size_t i = 0; i < n; i++) idom[i] = -1;
    idom[0] = 0;

    // Cooper-Harvey-Kennedy iterative dominators on RPO.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < n; i++) {
            int32_t new_idom = -1;
            for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, rpo[i]);
                 p != MLIR_INVALID_HANDLE; p = MLIR_GetPredNext(ctx, p)) {
                MLIR_BlockHandle pb = MLIR_GetPredBlock(ctx, p);
                if (pb == MLIR_INVALID_HANDLE) continue;
                int32_t pi = cfg_map_get(&rpo_idx, pb);
                if (pi < 0 || idom[pi] < 0) continue;
                if (new_idom < 0) new_idom = pi;
                else new_idom = cfg_intersect(idom, pi, new_idom);
            }
            if (new_idom >= 0 && idom[i] != new_idom) {
                idom[i] = new_idom;
                changed = true;
            }
        }
    }
    idom[0] = -1; // entry has no immediate dominator

    uint32_t *loop_depth = arena_new_array(out, uint32_t, n);
    bool *is_header = arena_new_array(out, bool, n);
    memset(loop_depth, 0, n * sizeof(uint32_t));
    memset(is_header, 0, n * sizeof(bool));

    cfg->n_blocks = n;
    cfg->rpo = rpo;
    cfg->idom = idom;
    cfg->loop_depth = loop_depth;
    cfg->is_loop_header = is_header;

    // Natural loops: for each back-edge u -> h (where h dominates u),
    // collect all blocks reaching u without going through h.
    int32_t *wl = arena_new_array(out, int32_t, n);
    uint32_t *seen = arena_new_array(out, uint32_t, n);
    memset(seen, 0, n * sizeof(uint32_t));
    uint32_t stamp = 0;

    for (size_t hi = 0; hi < n; hi++) {
        bool found_backedge = false;
        stamp++;
        size_t wn = 0;
        for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, rpo[hi]);
             p != MLIR_INVALID_HANDLE; p = MLIR_GetPredNext(ctx, p)) {
            MLIR_BlockHandle pb = MLIR_GetPredBlock(ctx, p);
            if (pb == MLIR_INVALID_HANDLE) continue;
            int32_t ui = cfg_map_get(&rpo_idx, pb);
            if (ui >= 0 && MLIR_CFGDominates(cfg, (int32_t)hi, ui)) {
                found_backedge = true;
                if (seen[ui] != stamp) {
                    seen[ui] = stamp;
                    wl[wn++] = ui;
                }
            }
        }
        if (!found_backedge) continue;
        is_header[hi] = true;
        seen[hi] = stamp;
        loop_depth[hi]++;
        while (wn > 0) {
            int32_t cur = wl[--wn];
            if (cur != (int32_t)hi) loop_depth[cur]++;
            for (MLIR_PredHandle p = MLIR_GetBlockFirstPred(ctx, rpo[cur]);
                 p != MLIR_INVALID_HANDLE; p = MLIR_GetPredNext(ctx, p)) {
                MLIR_BlockHandle pb = MLIR_GetPredBlock(ctx, p);
                if (pb == MLIR_INVALID_HANDLE) continue;
                int32_t pi = cfg_map_get(&rpo_idx, pb);
                if (pi >= 0 && seen[pi] != stamp) {
                    seen[pi] = stamp;
                    wl[wn++] = pi;
                }
            }
        }
    }

    return cfg;
}

int32_t MLIR_CFGGetRPOIndex(const MLIR_CFGInfo *cfg, MLIR_BlockHandle block) {
    if (!cfg) return -1;
    for (size_t i = 0; i < cfg->n_blocks; i++) {
        if (cfg->rpo[i] == block) return (int32_t)i;
    }
    return -1;
}

bool MLIR_CFGDominates(const MLIR_CFGInfo *cfg, int32_t a_rpo, int32_t b_rpo) {
    if (!cfg || a_rpo < 0 || b_rpo < 0 ||
        (size_t)a_rpo >= cfg->n_blocks || (size_t)b_rpo >= cfg->n_blocks)
        return false;
    int32_t cur = b_rpo;
    while (cur > a_rpo && cur >= 0) cur = cfg->idom[cur];
    return cur == a_rpo;
}

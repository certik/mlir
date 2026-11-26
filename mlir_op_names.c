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
// Populated once into a zero-initialized static array rather than a global
// initializer (tinyC's self-host parser does not accept bitwise expressions in
// global array inits) or a 260-case switch (which exceeds wasm->macho's 256
// block nesting limit when compiled by tinyC).
static MLIR_OpTraits g_static_op_traits[OP_TYPE_COUNT];
static bool g_static_op_traits_init = false;

static void init_static_op_traits(void) {
#define MLIR_OP(enumerator, name, traits) g_static_op_traits[enumerator] = (MLIR_OpTraits)(traits);
#include "mlir_ops.def"
    g_static_op_traits_init = true;
}

MLIR_OpTraits MLIR_GetOpTypeTraits(MLIR_OpType type) {
    if (!g_static_op_traits_init) init_static_op_traits();
    if ((unsigned)type < (unsigned)OP_TYPE_COUNT) return g_static_op_traits[type];
    return MLIR_T_NONE;
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

MLIR_CFGInfo *mlir_compute_cfg_info_generic(MLIR_Context *ctx, MLIR_RegionHandle region, Arena *out) {
    MLIR_CFGInfo *cfg = arena_new(out, MLIR_CFGInfo);
    cfg->n_blocks = 0;
    cfg->rpo = NULL;
    cfg->idom = NULL;
    cfg->loop_depth = NULL;
    cfg->is_loop_header = NULL;

    MLIR_BlockHandle entry = MLIR_GetRegionFirstBlock(ctx, region);
    if (entry == MLIR_INVALID_HANDLE) return cfg;

    size_t n_total = MLIR_GetRegionNumBlocks(ctx, region);

    // Single fused DFS pass: discovers reachable blocks, computes postorder,
    // caches terminator + successor count on the DFS stack, and records all
    // CFG edges in dense discovery-id coordinates with only ONE CfgBlkMap.
    CfgBlkMap dfs_id;
    cfg_map_init(&dfs_id, out, n_total);
    MLIR_BlockHandle *stk_blk = arena_new_array(out, MLIR_BlockHandle, n_total);
    MLIR_OpHandle *stk_term = arena_new_array(out, MLIR_OpHandle, n_total);
    int32_t *stk_id = arena_new_array(out, int32_t, n_total);
    uint32_t *stk_succ = arena_new_array(out, uint32_t, n_total);
    uint32_t *stk_ns = arena_new_array(out, uint32_t, n_total);
    MLIR_BlockHandle *po_blk = arena_new_array(out, MLIR_BlockHandle, n_total);
    int32_t *po_id = arena_new_array(out, int32_t, n_total);
    int32_t *dfs_to_rpo = arena_new_array(out, int32_t, n_total);

    size_t edge_cap = n_total * 2 + 8;
    int32_t *edge_u = arena_new_array(out, int32_t, edge_cap);
    int32_t *edge_v = arena_new_array(out, int32_t, edge_cap);
    size_t n_edges = 0;

    size_t sp = 0, n_po = 0;
    int32_t next_id = 0;
    {
        MLIR_OpHandle term = MLIR_GetBlockLastOp(ctx, entry);
        uint32_t ns = (term != MLIR_INVALID_HANDLE) ? (uint32_t)MLIR_GetOpNumSuccessors(ctx, term) : 0;
        cfg_map_put(&dfs_id, entry, next_id);
        stk_blk[0] = entry;
        stk_term[0] = term;
        stk_id[0] = next_id++;
        stk_succ[0] = 0;
        stk_ns[0] = ns;
        sp = 1;
    }

    while (sp > 0) {
        size_t top = sp - 1;
        uint32_t si = stk_succ[top];
        if (si < stk_ns[top]) {
            stk_succ[top] = si + 1;
            MLIR_BlockHandle succ = MLIR_GetOpSuccessor(ctx, stk_term[top], si);
            if (succ != MLIR_INVALID_HANDLE) {
                int32_t cur_id = stk_id[top];
                int32_t sid = cfg_map_get(&dfs_id, succ);
                if (sid < 0) {
                    sid = next_id++;
                    cfg_map_put(&dfs_id, succ, sid);
                    MLIR_OpHandle sterm = MLIR_GetBlockLastOp(ctx, succ);
                    uint32_t sns = (sterm != MLIR_INVALID_HANDLE)
                        ? (uint32_t)MLIR_GetOpNumSuccessors(ctx, sterm) : 0;
                    stk_blk[sp] = succ;
                    stk_term[sp] = sterm;
                    stk_id[sp] = sid;
                    stk_succ[sp] = 0;
                    stk_ns[sp] = sns;
                    sp++;
                }
                if (n_edges == edge_cap) {
                    size_t new_cap = edge_cap * 2;
                    int32_t *nu = arena_new_array(out, int32_t, new_cap);
                    int32_t *nv = arena_new_array(out, int32_t, new_cap);
                    memcpy(nu, edge_u, n_edges * sizeof(int32_t));
                    memcpy(nv, edge_v, n_edges * sizeof(int32_t));
                    edge_u = nu;
                    edge_v = nv;
                    edge_cap = new_cap;
                }
                edge_u[n_edges] = cur_id;
                edge_v[n_edges] = sid;
                n_edges++;
            }
        } else {
            sp = top;
            po_blk[n_po] = stk_blk[top];
            po_id[n_po] = stk_id[top];
            n_po++;
        }
    }

    size_t n = n_po;
    if (n == 0) return cfg;

    // Reverse postorder -> RPO and build dense dfs_id -> rpo_idx table.
    MLIR_BlockHandle *rpo = arena_new_array(out, MLIR_BlockHandle, n);
    for (size_t i = 0; i < n; i++) {
        size_t ri = n - 1 - i;
        rpo[ri] = po_blk[i];
        dfs_to_rpo[po_id[i]] = (int32_t)ri;
    }

    // Build compact CSR predecessor array in RPO coordinates (zero hash lookups,
    // zero second pass over ops/successors).
    uint32_t *pred_off = arena_new_array(out, uint32_t, n + 1);
    memset(pred_off, 0, (n + 1) * sizeof(uint32_t));
    for (size_t e = 0; e < n_edges; e++) {
        int32_t u = dfs_to_rpo[edge_u[e]];
        int32_t v = dfs_to_rpo[edge_v[e]];
        edge_u[e] = u;
        edge_v[e] = v;
        pred_off[(size_t)v + 1]++;
    }
    for (size_t i = 0; i < n; i++) pred_off[i + 1] += pred_off[i];
    int32_t *pred_rpo = arena_new_array(out, int32_t, n_edges > 0 ? n_edges : 1);
    uint32_t *pred_cur = arena_new_array(out, uint32_t, n);
    memcpy(pred_cur, pred_off, n * sizeof(uint32_t));
    for (size_t e = 0; e < n_edges; e++) {
        int32_t v = edge_v[e];
        pred_rpo[pred_cur[v]++] = edge_u[e];
    }

    int32_t *idom = arena_new_array(out, int32_t, n);
    for (size_t i = 0; i < n; i++) idom[i] = -1;
    idom[0] = 0;

    // Cooper-Harvey-Kennedy iterative dominators on RPO.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < n; i++) {
            uint32_t ps = pred_off[i];
            uint32_t pe = pred_off[i + 1];
            int32_t new_idom;
            if (pe - ps == 1) {
                new_idom = pred_rpo[ps];
                if (idom[new_idom] < 0) continue;
            } else {
                new_idom = -1;
                for (uint32_t p = ps; p < pe; p++) {
                    int32_t pi = pred_rpo[p];
                    if (idom[pi] < 0) continue;
                    if (new_idom < 0) new_idom = pi;
                    else new_idom = cfg_intersect(idom, pi, new_idom);
                }
                if (new_idom < 0) continue;
            }
            if (idom[i] != new_idom) {
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

    // Natural loops: for each back-edge u -> h (where h dominates u, so u >= h
    // in RPO), collect all blocks reaching u without going through h.
    int32_t *wl = arena_new_array(out, int32_t, n);
    uint32_t *seen = arena_new_array(out, uint32_t, n);
    memset(seen, 0, n * sizeof(uint32_t));
    uint32_t stamp = 0;

    for (size_t hi = 0; hi < n; hi++) {
        bool found_backedge = false;
        stamp++;
        size_t wn = 0;
        uint32_t pe = pred_off[hi + 1];
        for (uint32_t p = pred_off[hi]; p < pe; p++) {
            int32_t ui = pred_rpo[p];
            if (ui < (int32_t)hi) continue;
            int32_t cur = ui;
            while (cur > (int32_t)hi) cur = idom[cur];
            if (cur == (int32_t)hi) {
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
            if (cur == (int32_t)hi) continue;
            loop_depth[cur]++;
            uint32_t cpe = pred_off[cur + 1];
            for (uint32_t p = pred_off[cur]; p < cpe; p++) {
                int32_t pi = pred_rpo[p];
                if (seen[pi] != stamp) {
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

// -----------------------------------------------------------------------------
// Backend-agnostic LLVM-level completeness helpers (doc/mlir_api_design.md §4).
// -----------------------------------------------------------------------------

MLIR_ValueHandle MLIR_BuildConstantInt(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, int64_t v) {
    MLIR_AttributeHandle val_attr;
    if (MLIR_GetTypeKind(b->ctx, rt) == MLIR_TYPE_KIND_VECTOR) {
        MLIR_TypeHandle elem = MLIR_GetTypeElement(b->ctx, rt);
        val_attr = MLIR_CreateAttributeSplat(b->ctx, rt,
            MLIR_CreateAttributeInteger(b->ctx, v, elem));
    } else {
        val_attr = MLIR_CreateAttributeInteger(b->ctx, v, rt);
    }
    MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_VALUE, val_attr);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = &attr;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildConstantFloat(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, double v) {
    MLIR_AttributeHandle val_attr;
    if (MLIR_GetTypeKind(b->ctx, rt) == MLIR_TYPE_KIND_VECTOR) {
        MLIR_TypeHandle elem = MLIR_GetTypeElement(b->ctx, rt);
        val_attr = MLIR_CreateAttributeSplat(b->ctx, rt,
            MLIR_CreateAttributeFloat(b->ctx, v, elem));
    } else {
        val_attr = MLIR_CreateAttributeFloat(b->ctx, v, rt);
    }
    MLIR_NamedAttr attr = MLIR_Named(MLIR_KEY_VALUE, val_attr);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = &attr;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildFBinary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y, MLIR_FastMath fmf) {
    MLIR_ValueHandle ops[2];
    ops[0] = x;
    ops[1] = y;
    MLIR_NamedAttr attr;
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &rt;
    s.n_results = 1;
    if (fmf != MLIR_FMF_NONE) {
        attr = MLIR_Named(MLIR_KEY_FASTMATH_FLAGS, MLIR_CreateAttributeFastMath(b->ctx, fmf));
        s.attrs = &attr;
        s.n_attrs = 1;
    }
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildFUnary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_FastMath fmf) {
    MLIR_NamedAttr attr;
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = &x;
    s.n_operands = 1;
    s.result_types = &rt;
    s.n_results = 1;
    if (fmf != MLIR_FMF_NONE) {
        attr = MLIR_Named(MLIR_KEY_FASTMATH_FLAGS, MLIR_CreateAttributeFastMath(b->ctx, fmf));
        s.attrs = &attr;
        s.n_attrs = 1;
    }
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildUndef(MLIR_Builder *b, MLIR_TypeHandle rt) {
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = OP_TYPE_LLVM_MLIR_UNDEF;
    s.result_types = &rt;
    s.n_results = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildPoison(MLIR_Builder *b, MLIR_TypeHandle rt) {
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = OP_TYPE_LLVM_MLIR_POISON;
    s.result_types = &rt;
    s.n_results = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildZero(MLIR_Builder *b, MLIR_TypeHandle rt) {
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = OP_TYPE_LLVM_MLIR_ZERO;
    s.result_types = &rt;
    s.n_results = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildCallIntrinsic(MLIR_Builder *b, string intrin_name,
        MLIR_TypeHandle rt, const MLIR_ValueHandle *args, uint32_t n_args,
        MLIR_FastMath fmf) {
    MLIR_NamedAttr attrs[2];
    uint32_t n_attrs = 0;
    attrs[n_attrs++] = MLIR_Named(MLIR_KEY_INTRIN,
        MLIR_CreateAttributeString(b->ctx, intrin_name));
    if (fmf != MLIR_FMF_NONE) {
        attrs[n_attrs++] = MLIR_Named(MLIR_KEY_FASTMATH_FLAGS,
            MLIR_CreateAttributeFastMath(b->ctx, fmf));
    }
    bool has_result = (rt != MLIR_INVALID_HANDLE &&
                       MLIR_GetTypeKind(b->ctx, rt) != MLIR_TYPE_KIND_LLVM_VOID);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = OP_TYPE_LLVM_CALL_INTRINSIC;
    s.operands = args;
    s.n_operands = n_args;
    if (has_result) {
        s.result_types = &rt;
        s.n_results = 1;
    }
    s.attrs = attrs;
    s.n_attrs = n_attrs;
    MLIR_OpHandle op = MLIR_BuildOp(b, &s);
    return has_result ? MLIR_GetOpResult(b->ctx, op, 0) : MLIR_INVALID_HANDLE;
}

string MLIR_GetCallee(MLIR_Context *ctx, MLIR_OpHandle op) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, MLIR_KEY_CALLEE);
    if (a != MLIR_INVALID_HANDLE) {
        string s = MLIR_GetAttributeString(ctx, a);
        if (s.size > 0 && s.str[0] == '@') {
            string bare;
            bare.str = s.str + 1;
            bare.size = s.size - 1;
            return bare;
        }
        return s;
    }
    a = MLIR_GetOpAttr(ctx, op, MLIR_KEY_INTRIN);
    if (a != MLIR_INVALID_HANDLE) {
        return MLIR_GetAttributeString(ctx, a);
    }
    return str_lit("");
}

bool MLIR_IsFunctionDeclaration(MLIR_Context *ctx, MLIR_OpHandle fn) {
    if (MLIR_GetOpNumRegions(ctx, fn) == 0) return true;
    return MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, fn, 0)) == 0;
}

static bool passthrough_entry_matches(MLIR_Context *ctx, MLIR_AttributeHandle elem,
                                      string key, string *out_value) {
    MLIR_AttrKind k = MLIR_GetAttributeKind(ctx, elem);
    if (k == MLIR_ATTR_KIND_STRING) {
        if (str_eq(MLIR_GetAttributeString(ctx, elem), key)) {
            if (out_value) *out_value = str_lit("");
            return true;
        }
    } else if (k == MLIR_ATTR_KIND_ARRAY && MLIR_GetAttributeArraySize(ctx, elem) >= 2) {
        MLIR_AttributeHandle k_attr = MLIR_GetAttributeArrayElement(ctx, elem, 0);
        if (MLIR_GetAttributeKind(ctx, k_attr) == MLIR_ATTR_KIND_STRING &&
            str_eq(MLIR_GetAttributeString(ctx, k_attr), key)) {
            if (out_value) {
                MLIR_AttributeHandle v_attr = MLIR_GetAttributeArrayElement(ctx, elem, 1);
                if (MLIR_GetAttributeKind(ctx, v_attr) == MLIR_ATTR_KIND_STRING)
                    *out_value = MLIR_GetAttributeString(ctx, v_attr);
                else
                    *out_value = str_lit("");
            }
            return true;
        }
    }
    return false;
}

string MLIR_GetFunctionPassthrough(MLIR_Context *ctx, MLIR_OpHandle fn, string key) {
    MLIR_AttributeHandle pt = MLIR_GetOpAttr(ctx, fn, MLIR_KEY_PASSTHROUGH);
    if (pt == MLIR_INVALID_HANDLE || MLIR_GetAttributeKind(ctx, pt) != MLIR_ATTR_KIND_ARRAY)
        return str_lit("");
    size_t n = MLIR_GetAttributeArraySize(ctx, pt);
    for (size_t i = 0; i < n; i++) {
        string val;
        if (passthrough_entry_matches(ctx, MLIR_GetAttributeArrayElement(ctx, pt, i), key, &val))
            return val;
    }
    return str_lit("");
}

bool MLIR_HasFunctionPassthrough(MLIR_Context *ctx, MLIR_OpHandle fn, string key) {
    MLIR_AttributeHandle pt = MLIR_GetOpAttr(ctx, fn, MLIR_KEY_PASSTHROUGH);
    if (pt == MLIR_INVALID_HANDLE || MLIR_GetAttributeKind(ctx, pt) != MLIR_ATTR_KIND_ARRAY)
        return false;
    size_t n = MLIR_GetAttributeArraySize(ctx, pt);
    for (size_t i = 0; i < n; i++) {
        if (passthrough_entry_matches(ctx, MLIR_GetAttributeArrayElement(ctx, pt, i), key, NULL))
            return true;
    }
    return false;
}

void MLIR_SetFunctionPassthrough(MLIR_Context *ctx, MLIR_OpHandle fn,
                                 string key, string value) {
    MLIR_AttributeHandle entry;
    if (value.size > 0) {
        MLIR_AttributeHandle pair[2];
        pair[0] = MLIR_CreateAttributeString(ctx, key);
        pair[1] = MLIR_CreateAttributeString(ctx, value);
        entry = MLIR_CreateAttributeArray(ctx, pair, 2);
    } else {
        entry = MLIR_CreateAttributeString(ctx, key);
    }
    MLIR_AttributeHandle pt = MLIR_GetOpAttr(ctx, fn, MLIR_KEY_PASSTHROUGH);
    size_t n = 0;
    if (pt != MLIR_INVALID_HANDLE && MLIR_GetAttributeKind(ctx, pt) == MLIR_ATTR_KIND_ARRAY)
        n = MLIR_GetAttributeArraySize(ctx, pt);
    MLIR_AttributeHandle stack_elems[16];
    MLIR_AttributeHandle *elems = (n + 1 <= 16)
        ? stack_elems
        : (MLIR_AttributeHandle *)arena_alloc(MLIR_GetArenaAllocator(ctx),
                                              (n + 1) * sizeof(MLIR_AttributeHandle));
    bool replaced = false;
    for (size_t i = 0; i < n; i++) {
        MLIR_AttributeHandle e = MLIR_GetAttributeArrayElement(ctx, pt, i);
        if (!replaced && passthrough_entry_matches(ctx, e, key, NULL)) {
            elems[i] = entry;
            replaced = true;
        } else {
            elems[i] = e;
        }
    }
    if (!replaced) elems[n++] = entry;
    MLIR_SetOpAttr(ctx, fn, MLIR_KEY_PASSTHROUGH,
                   MLIR_CreateAttributeArray(ctx, elems, n));
}

static bool str_starts_with(string s, const char *prefix) {
    size_t plen = strlen(prefix);
    if (s.size < plen) return false;
    for (size_t i = 0; i < plen; i++)
        if (s.str[i] != prefix[i]) return false;
    return true;
}

MLIR_TargetInfo MLIR_TargetFromTriple(string triple, uint64_t features) {
    MLIR_TargetInfo t;
    t.arch = MLIR_ARCH_UNKNOWN;
    t.features = features;
    t.triple = triple;
    if (str_starts_with(triple, "x86_64") || str_starts_with(triple, "amd64"))
        t.arch = MLIR_ARCH_X86_64;
    else if (str_starts_with(triple, "aarch64") || str_starts_with(triple, "arm64"))
        t.arch = MLIR_ARCH_AARCH64;
    else if (str_starts_with(triple, "wasm32") || str_starts_with(triple, "wasm64"))
        t.arch = MLIR_ARCH_WASM32;
    return t;
}

static uint64_t feature_name_bit(const char *name, size_t len) {
    string s;
    s.str = (char *)name;
    s.size = len;
    if (str_eq(s, str_lit("sse2"))) return MLIR_FEATURE_SSE2;
    if (str_eq(s, str_lit("sse4.1"))) return MLIR_FEATURE_SSE4_1;
    if (str_eq(s, str_lit("avx"))) return MLIR_FEATURE_AVX;
    if (str_eq(s, str_lit("avx2"))) return MLIR_FEATURE_AVX2;
    if (str_eq(s, str_lit("fma"))) return MLIR_FEATURE_FMA;
    if (str_eq(s, str_lit("avx512f"))) return MLIR_FEATURE_AVX512F;
    if (str_eq(s, str_lit("neon"))) return MLIR_FEATURE_NEON;
    if (str_eq(s, str_lit("simd128"))) return MLIR_FEATURE_SIMD128;
    return 0;
}

static void apply_feature_string(uint64_t *features, string s) {
    size_t i = 0;
    while (i < s.size) {
        while (i < s.size && (s.str[i] == ',' || s.str[i] == ' ' || s.str[i] == '"' ||
                              s.str[i] == '[' || s.str[i] == ']'))
            i++;
        if (i >= s.size) break;
        char sign = s.str[i];
        if (sign == '+' || sign == '-') {
            i++;
            size_t start = i;
            while (i < s.size && s.str[i] != ',' && s.str[i] != ' ' &&
                   s.str[i] != '"' && s.str[i] != ']' && s.str[i] != '>')
                i++;
            uint64_t bit = feature_name_bit(s.str + start, i - start);
            if (bit) {
                if (sign == '+') *features |= bit;
                else *features &= ~bit;
            }
        } else {
            while (i < s.size && s.str[i] != ',' && s.str[i] != ' ' &&
                   s.str[i] != '"' && s.str[i] != ']' && s.str[i] != '>')
                i++;
        }
    }
}

bool MLIR_TargetHasFeature(MLIR_Context *ctx, const MLIR_TargetInfo *t,
                           MLIR_OpHandle fn, uint64_t feature) {
    if (!t || feature == 0) return false;
    uint64_t eff = t->features;
    if (ctx && fn != MLIR_INVALID_HANDLE) {
        string pt = MLIR_GetFunctionPassthrough(ctx, fn, str_lit("target-features"));
        if (pt.size > 0) apply_feature_string(&eff, pt);
        MLIR_AttributeHandle tf = MLIR_GetOpAttr(ctx, fn, MLIR_KEY_TARGET_FEATURES);
        if (tf != MLIR_INVALID_HANDLE) {
            MLIR_AttrKind k = MLIR_GetAttributeKind(ctx, tf);
            if (k == MLIR_ATTR_KIND_STRING) {
                apply_feature_string(&eff, MLIR_GetAttributeString(ctx, tf));
            } else if (k == MLIR_ATTR_KIND_ARRAY) {
                size_t n = MLIR_GetAttributeArraySize(ctx, tf);
                for (size_t i = 0; i < n; i++) {
                    MLIR_AttributeHandle e = MLIR_GetAttributeArrayElement(ctx, tf, i);
                    if (MLIR_GetAttributeKind(ctx, e) == MLIR_ATTR_KIND_STRING)
                        apply_feature_string(&eff, MLIR_GetAttributeString(ctx, e));
                }
            } else {
                string txt = MLIR_GetAttributeAsString(ctx, tf);
                if (txt.size > 0) apply_feature_string(&eff, txt);
            }
        }
    }
    return (eff & feature) == feature;
}

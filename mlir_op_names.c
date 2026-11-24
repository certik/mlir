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

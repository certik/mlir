#pragma once

// The op-type table of a context: the name of every op type and the reverse
// map, used by both backends (doc/mlir_api_design.md section 2.5). The static
// types come from mlir_ops.def; a context adds dynamic types for other names
// (MLIR_RegisterOpType), numbered from OP_TYPE_COUNT up.

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OP_TYPE_ALIAS_CAPACITY 8
#define OP_TYPE_ALIAS_SLOT 0x80000000u

typedef struct OpTypeTable {
    Arena *arena;        // owned by the context; holds the arrays and names
    // names[t] for every type t < n_types; names[OP_TYPE_UNREGISTERED] is "".
    string *names;
    uint32_t n_types;    // OP_TYPE_COUNT + the number of dynamic types
    uint32_t cap_types;
    // Other names of static types (MLIR_OP_ALIAS in mlir_ops.def), in
    // arrays of OP_TYPE_ALIAS_CAPACITY entries. (Pointers rather than member
    // arrays: tinyC cannot return a struct element of a member array.)
    string *alias_names;
    MLIR_OpType *alias_types;
    uint32_t n_aliases;
    // Open-addressing hash of the names and aliases. A slot holds a type,
    // OP_TYPE_ALIAS_SLOT | i for alias i, or OP_TYPE_UNREGISTERED (0) when
    // it is empty. n_slots is a power of two and at least twice the number
    // of entries.
    uint32_t *slots;
    uint32_t n_slots;
} OpTypeTable;

// Fills `t` with the static types, allocating from `arena`.
void op_type_table_init(OpTypeTable *t, Arena *arena);
// The type named `name` (a name or an alias), or OP_TYPE_UNREGISTERED.
MLIR_OpType op_type_table_lookup(const OpTypeTable *t, string name);
// The type named `name`; adds a dynamic type, with a copy of the name, if
// there is none.
MLIR_OpType op_type_table_register(OpTypeTable *t, string name);

// The name of `type`, or "" for OP_TYPE_UNREGISTERED and unknown values.
void op_type_table_ensure_names(OpTypeTable *t);
static inline string op_type_table_name(const OpTypeTable *t, MLIR_OpType type) {
    if (!t->names) op_type_table_ensure_names((OpTypeTable *)t);
    if ((uint32_t)type < t->n_types) return t->names[type];
    return t->names[OP_TYPE_UNREGISTERED];
}

// The attribute-key table of a context (mlir_api.h, "Attribute keys"): the
// names of the static keys of mlir_attr_keys.def and of the dynamic keys the
// context interns, numbered from MLIR_KEY_COUNT up, and the hash that maps a
// name back to its key. Shared by both backends.
typedef struct AttrKeyTable {
    Arena *arena;      // owned by the context; holds the arrays and names
    string *names;     // names[k] for every key k < n_keys; names[0] is ""
    uint32_t n_keys;
    uint32_t cap_keys;
    uint32_t *slots;   // open addressing, keys, 0 = empty; a power of two
    uint32_t n_slots;
} AttrKeyTable;

void attr_key_table_init(AttrKeyTable *t, Arena *arena);
// An empty table, without the static keys: interns any strings to small
// integers from 1 (the native backend's location file names and names).
void attr_key_table_init_strings(AttrKeyTable *t, Arena *arena);
// The key of `name`, or MLIR_KEY_INVALID.
MLIR_AttrKey attr_key_table_lookup(const AttrKeyTable *t, string name);
// The key of `name`; adds a dynamic key, with a copy of the name, if there
// is none.
MLIR_AttrKey attr_key_table_intern(AttrKeyTable *t, string name);
// The name of `key`, or "" for MLIR_KEY_INVALID and unknown values.
string attr_key_table_name(const AttrKeyTable *t, MLIR_AttrKey key);

// Generic backend-agnostic CFG + dominators + natural loops implementation.
MLIR_CFGInfo *mlir_compute_cfg_info_generic(MLIR_Context *ctx, MLIR_RegionHandle region, Arena *out);

#ifdef __cplusplus
}
#endif

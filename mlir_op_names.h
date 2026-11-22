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

typedef struct OpTypeTable {
    Arena *arena;        // owned by the context; holds the arrays and names
    // names[t] for every type t < n_types; names[OP_TYPE_UNREGISTERED] is "".
    string *names;
    uint32_t n_types;    // OP_TYPE_COUNT + the number of dynamic types
    uint32_t cap_types;
    // Open-addressing hash of the names. A slot holds a type, or
    // OP_TYPE_UNREGISTERED (0) when it is empty. n_slots is a power of two
    // and at least twice n_types.
    uint32_t *slots;
    uint32_t n_slots;
} OpTypeTable;

// Fills `t` with the static types, allocating from `arena`.
void op_type_table_init(OpTypeTable *t, Arena *arena);
// The type named `name`, or OP_TYPE_UNREGISTERED.
MLIR_OpType op_type_table_lookup(const OpTypeTable *t, string name);
// The type named `name`; adds a dynamic type, with a copy of the name, if
// there is none.
MLIR_OpType op_type_table_register(OpTypeTable *t, string name);

// The name of `type`, or "" for OP_TYPE_UNREGISTERED and unknown values.
static inline string op_type_table_name(const OpTypeTable *t, MLIR_OpType type) {
    if ((uint32_t)type < t->n_types) return t->names[type];
    return t->names[OP_TYPE_UNREGISTERED];
}

#ifdef __cplusplus
}
#endif

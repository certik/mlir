#pragma once

#include <base/string.h>
#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

string op_type_to_string(MLIR_OpType type);

// The op type registered under `name`, or OP_TYPE_UNREGISTERED if there is
// none. A hash lookup; the table is built on first use.
MLIR_OpType op_type_from_string(string name);

// Size for global tables indexed by MLIR_OpType. A literal rather than
// OP_TYPE_COUNT because tinyC, which compiles these sources when
// self-hosting, does not accept an enumerator as a global array length.
#define OP_TYPE_TABLE_CAPACITY 512
#ifndef __TINYC__
// Compile-time check (a negative array size if the enum outgrows the
// tables): raise OP_TYPE_TABLE_CAPACITY, and OP_NAME_TABLE_SIZE in
// mlir_op_names.c with it.
typedef char op_type_table_capacity_ok[(OP_TYPE_COUNT <= OP_TYPE_TABLE_CAPACITY) ? 1 : -1];
#endif

#ifdef __cplusplus
}
#endif

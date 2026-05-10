// Shared C-struct intermediate representation for the wasmssa and
// wasmstack dialects. These structs are produced and consumed by the
// three-stage native LLVM->WASM pipeline:
//
//   stage 1: mlir_lower_llvm_to_wasmssa.c
//              LLVM-dialect MLIR ----> wasmssa_module_t
//   stage 2: mlir_stackify_wasmssa.c
//              wasmssa_module_t   ----> wasmstack_module_t
//   stage 3: mlir_translate_wasmstack_to_binary.c
//              wasmstack_module_t ----> .wasm.o relocatable bytes
//
// Each op carries a MLIR_OpType discriminator from one of the
// OP_TYPE_WASMSSA_* / OP_TYPE_WASMSTACK_* enum ranges, so dispatch in
// every stage is enum-based (no string comparisons).

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/string.h>

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// WebAssembly value-type byte encodings (also used as MLIR-side
// discriminators here).
enum {
    WT_I32 = 0x7f,
    WT_I64 = 0x7e,
    WT_F32 = 0x7d,
    WT_F64 = 0x7c,
};

// =============================================================================
// wasmssa: SSA-form intermediate.
// =============================================================================
//
// Each op references its operands by ssa_def index — a 0-based index
// into the surrounding function's `ops` array. The result of op N (if
// `has_result` is true) is the SSA value with index N.
//
// Operand-order convention for non-commutative ops:
//   sub:   operands[0] = lhs, operands[1] = rhs
//   load:  operands[0] = address
//   store: operands[0] = address, operands[1] = value
//   global_set: operands[0] = value
//   call:  operands[0..n-1] = arguments in declared order

typedef struct {
    MLIR_OpType type;             // OP_TYPE_WASMSSA_*
    uint8_t     valtype;          // WT_*. For ops producing/operating on a
                                  // typed value (const/add/sub/load/store/
                                  // global.get/global.set/call result).

    // Op-specific data. Only fields relevant to `type` are used.
    int64_t  i_const;             // CONST: integer immediate
    uint32_t global_idx;          // GLOBAL_GET / GLOBAL_SET: 0 = __stack_pointer
    uint32_t memory_offset;       // LOAD / STORE
    uint32_t memory_align_log2;   // LOAD / STORE
    uint32_t mem_size_bytes;      // LOAD / STORE (1/2/4/8)
    char    *call_target;         // CALL: function symbol name (xstrdup'd)
    uint8_t  wasm_opcode;         // BINOP / UNOP: actual wasm bytecode byte
    uint32_t br_depth;            // BR / BR_IF: relative target depth
    uint32_t carrier_id;          // CARRIER_SET / CARRIER_GET

    // Operands as ssa_def indices into the parent function's `ops`.
    int   *operands;              // length n_operands; -1 means "unbound"
    int    n_operands;

    bool has_result;              // true if this op defines an SSA value
} wasmssa_op_t;

typedef struct {
    char    *name;                // unique symbol name (xstrdup'd)
    bool     imported;
    bool     exported;            // true if visible to linker (e.g. __original_main)
    uint8_t *param_types;         // length n_params (WT_*)
    size_t   n_params;
    uint8_t *result_types;        // length n_results (WT_*)
    size_t   n_results;

    // Carriers: per-function shared "registers" used to materialize
    // scf.yield-style values across structured-CF boundaries. Stage 2
    // turns each carrier into one wasm local of the recorded valtype.
    uint8_t *carrier_vts;
    size_t   n_carriers;

    // Function body. Empty for imports.
    wasmssa_op_t *ops;
    size_t        n_ops;
    size_t        c_ops;
} wasmssa_func_t;

typedef struct {
    wasmssa_func_t *funcs;
    size_t          n_funcs;
    size_t          c_funcs;
    // The imported env.__stack_pointer global is implicit; emitted by stage 3.
} wasmssa_module_t;

// =============================================================================
// wasmstack: stack-machine intermediate (1:1 with wasm bytecode).
// =============================================================================

typedef struct {
    MLIR_OpType type;             // OP_TYPE_WASMSTACK_*

    // Op-specific data.
    uint8_t  valtype;             // WT_* (for typed ops)
    int64_t  i_const;             // CONST
    uint32_t local_idx;           // LOCAL_GET / LOCAL_SET / LOCAL_TEE
    uint32_t global_idx;          // GLOBAL_GET / GLOBAL_SET
    uint32_t memory_offset;       // LOAD / STORE
    uint32_t memory_align_log2;   // LOAD / STORE
    uint32_t mem_size_bytes;      // LOAD / STORE
    char    *call_target;         // CALL
    uint8_t  wasm_opcode;         // BINOP / UNOP
    uint32_t br_depth;            // BR / BR_IF
} wasmstack_op_t;

typedef struct {
    char    *name;
    bool     imported;
    bool     exported;
    uint8_t *param_types;         // length n_params
    size_t   n_params;
    uint8_t *result_types;
    size_t   n_results;

    // Locals beyond the params (params occupy local indices 0..n_params-1;
    // declared locals occupy n_params..n_params+n_locals-1).
    uint8_t *local_types;
    size_t   n_locals;
    size_t   c_locals;

    // Body: sequence of stack-machine ops. Nothing implicit; `end` byte
    // is emitted by stage 3 at the close of the function.
    wasmstack_op_t *ops;
    size_t          n_ops;
    size_t          c_ops;
} wasmstack_func_t;

typedef struct {
    wasmstack_func_t *funcs;
    size_t            n_funcs;
    size_t            c_funcs;
} wasmstack_module_t;

// =============================================================================
// Constructors / destructors.
// =============================================================================

wasmssa_module_t   *wasmssa_module_new(void);
void                wasmssa_module_free(wasmssa_module_t *m);
wasmssa_func_t     *wasmssa_module_add_func(wasmssa_module_t *m);
wasmssa_op_t       *wasmssa_func_add_op(wasmssa_func_t *f);

wasmstack_module_t *wasmstack_module_new(void);
void                wasmstack_module_free(wasmstack_module_t *m);
wasmstack_func_t   *wasmstack_module_add_func(wasmstack_module_t *m);
wasmstack_op_t     *wasmstack_func_add_op(wasmstack_func_t *f);
uint32_t            wasmstack_func_add_local(wasmstack_func_t *f, uint8_t vt);

// =============================================================================
// Pipeline stages.
// =============================================================================

// Stage 1: lower an LLVM-dialect builtin.module to wasmssa form. Returns
// NULL on failure (a diagnostic is printed to stderr).
wasmssa_module_t *mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx,
                                             MLIR_OpHandle module);

// Stage 2: stackification — convert SSA-form wasmssa into stack-machine
// wasmstack. v1 implementation: naive (every SSA value materializes
// into a fresh local; nothing rides the operand stack). Always succeeds
// on a well-formed input; never returns NULL.
wasmstack_module_t *mlir_stackify_wasmssa(wasmssa_module_t *m);

// Stage 3: serialize a wasmstack module to a wasm32 relocatable object.
// Returns the bytes (allocated in `ctx`'s arena) or an empty string on
// failure.
string mlir_translate_wasmstack_to_binary(MLIR_Context *ctx,
                                          wasmstack_module_t *m);

#ifdef __cplusplus
}
#endif

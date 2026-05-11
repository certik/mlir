// Internal (implementation-only) header for the native LLVM->WASM
// pipeline. Stages 1/2/3 use these C-struct working representations
// for their in-flight bookkeeping. The public interface — taking and
// returning MLIR_OpHandles — is declared in mlir_wasm_pipeline.h.
//
// The on-disk IR between stages is now plain MLIR (unregistered
// wasmssa.* / wasmstack.* ops); these structs are transient and never
// escape the .c files that include this header.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_wasm_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Globals (llvm.mlir.global -> wasm DATA segment + symbol).
// =============================================================================
typedef struct {
    uint32_t offset;          // byte offset within the segment's data
    char    *target;          // target global symbol name (xstrdup'd)
    int32_t  addend;
} wasm_data_reloc_t;

typedef struct {
    char    *name;
    uint8_t *data;
    uint32_t size;
    uint32_t align_pow;       // log2 alignment
    bool     is_const;
    wasm_data_reloc_t *relocs;
    size_t n_relocs, c_relocs;
} wasm_global_t;

// =============================================================================
// wasmssa: SSA-form intermediate (transient working form).
// =============================================================================
typedef struct {
    MLIR_OpType type;
    uint8_t     valtype;

    int64_t  i_const;
    uint32_t global_idx;
    uint32_t memory_offset;
    uint32_t memory_align_log2;
    uint32_t mem_size_bytes;
    char    *call_target;
    uint8_t  wasm_opcode;
    uint32_t br_depth;
    uint32_t carrier_id;

    uint8_t *sig_params;
    uint8_t *sig_results;
    size_t   n_sig_params, n_sig_results;

    int   *operands;
    int    n_operands;

    bool has_result;
} wasmssa_op_t;

typedef struct {
    char    *name;
    bool     imported;
    bool     exported;
    uint8_t *param_types;
    size_t   n_params;
    uint8_t *result_types;
    size_t   n_results;

    uint8_t *carrier_vts;
    size_t   n_carriers;

    wasmssa_op_t *ops;
    size_t        n_ops;
    size_t        c_ops;
} wasmssa_func_t;

typedef struct {
    wasmssa_func_t *funcs;
    size_t          n_funcs;
    size_t          c_funcs;
    wasm_global_t  *globals;
    size_t          n_globals;
    size_t          c_globals;
} wasmssa_module_t;

// =============================================================================
// wasmstack: stack-machine intermediate (transient working form).
// =============================================================================
typedef struct {
    MLIR_OpType type;

    uint8_t  valtype;
    int64_t  i_const;
    uint32_t local_idx;
    uint32_t global_idx;
    uint32_t memory_offset;
    uint32_t memory_align_log2;
    uint32_t mem_size_bytes;
    char    *call_target;
    uint8_t  wasm_opcode;
    uint32_t br_depth;

    uint8_t *sig_params;
    uint8_t *sig_results;
    size_t   n_sig_params, n_sig_results;
} wasmstack_op_t;

typedef struct {
    char    *name;
    bool     imported;
    bool     exported;
    uint8_t *param_types;
    size_t   n_params;
    uint8_t *result_types;
    size_t   n_results;

    uint8_t *local_types;
    size_t   n_locals;
    size_t   c_locals;

    wasmstack_op_t *ops;
    size_t          n_ops;
    size_t          c_ops;
} wasmstack_func_t;

typedef struct {
    wasmstack_func_t *funcs;
    size_t            n_funcs;
    size_t            c_funcs;
    wasm_global_t    *globals;
    size_t            n_globals;
    size_t            c_globals;
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

wasm_global_t      *wasmssa_module_add_global(wasmssa_module_t *m);
wasm_global_t      *wasmstack_module_add_global(wasmstack_module_t *m);
void                wasm_global_add_reloc(wasm_global_t *g, uint32_t off,
                                          const char *target, int32_t addend);

// =============================================================================
// Struct <-> MLIR converters. Used by the public adapter wrappers in
// mlir_wasm_pipeline_internal.c. Each builds (or reads) a builtin.module
// op containing flat wasmssa.* / wasmstack.* ops as direct children.
// =============================================================================

// Build an MLIR `builtin.module` op containing wasmssa.* ops from a
// wasmssa_module_t. Allocates into ctx's arena (string copies, etc.).
MLIR_OpHandle wasmssa_module_to_mlir(MLIR_Context *ctx,
                                     const wasmssa_module_t *m);

// Build a wasmssa_module_t (malloc-owned) by reading an MLIR
// `builtin.module` containing wasmssa.* ops. Returns NULL on failure.
wasmssa_module_t *mlir_to_wasmssa_module(MLIR_Context *ctx,
                                         MLIR_OpHandle module);

// Same for wasmstack.
MLIR_OpHandle wasmstack_module_to_mlir(MLIR_Context *ctx,
                                       const wasmstack_module_t *m);
wasmstack_module_t *mlir_to_wasmstack_module(MLIR_Context *ctx,
                                             MLIR_OpHandle module);

// Internal (struct-returning) stage entry points used by the adapter
// wrappers. The public MLIR-handle-based forms are in mlir_wasm_pipeline.h.
wasmssa_module_t   *mlir_lower_llvm_to_wasmssa_struct(MLIR_Context *ctx,
                                                      MLIR_OpHandle module);
wasmstack_module_t *mlir_stackify_wasmssa_struct(wasmssa_module_t *m);
string              mlir_translate_wasmstack_to_binary_struct(
                        MLIR_Context *ctx, wasmstack_module_t *m);

#ifdef __cplusplus
}
#endif

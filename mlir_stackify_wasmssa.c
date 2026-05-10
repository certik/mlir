// Stage 2 of the native LLVM->WASM pipeline:
// wasmssa_module_t (SSA) -> wasmstack_module_t (stack-machine).
//
// This is the "stackification" stage. v1 implementation is naive:
// every wasmssa SSA value materializes into a fresh wasmstack local.
// Nothing rides the operand stack between adjacent ops. The wasm
// runtime's JIT trivially optimizes back the resulting redundant
// local.set/local.get pairs.
//
// Naive scheme (per wasmssa op, in declaration order):
//   for each operand ssa_def_idx:
//       emit  wasmstack.local.get vmap[ssa_def_idx]
//   emit the corresponding wasmstack op (no operands; immediates copied)
//   if op produces a result:
//       allocate new local of the result's valtype
//       emit  wasmstack.local.set new_local
//       vmap[n_params + op_index] = new_local
//
// SSA value indexing (shared with stage 1):
//   ssa_def_idx in [0, n_params)              -> function parameter i
//   ssa_def_idx in [n_params, n_params + n_ops) -> result of op (idx - n_params)

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlir_wasm_dialect.h"

wasmstack_module_t *mlir_stackify_wasmssa(wasmssa_module_t *m);

// Helpers to append simple wasmstack ops.
static void emit_local_get(wasmstack_func_t *f, uint32_t idx) {
    wasmstack_op_t *o = wasmstack_func_add_op(f);
    o->type = OP_TYPE_WASMSTACK_LOCAL_GET;
    o->local_idx = idx;
}
static void emit_local_set(wasmstack_func_t *f, uint32_t idx) {
    wasmstack_op_t *o = wasmstack_func_add_op(f);
    o->type = OP_TYPE_WASMSTACK_LOCAL_SET;
    o->local_idx = idx;
}

// wasmssa op type -> wasmstack op type (the parallel-named counterparts).
// Returns 0 on unknown.
static MLIR_OpType ssa_to_stack(MLIR_OpType t) {
    switch (t) {
    case OP_TYPE_WASMSSA_CONST:      return OP_TYPE_WASMSTACK_CONST;
    case OP_TYPE_WASMSSA_ADD:        return OP_TYPE_WASMSTACK_ADD;
    case OP_TYPE_WASMSSA_SUB:        return OP_TYPE_WASMSTACK_SUB;
    case OP_TYPE_WASMSSA_BINOP:      return OP_TYPE_WASMSTACK_BINOP;
    case OP_TYPE_WASMSSA_UNOP:       return OP_TYPE_WASMSTACK_UNOP;
    case OP_TYPE_WASMSSA_LOAD:       return OP_TYPE_WASMSTACK_LOAD;
    case OP_TYPE_WASMSSA_STORE:      return OP_TYPE_WASMSTACK_STORE;
    case OP_TYPE_WASMSSA_GLOBAL_GET: return OP_TYPE_WASMSTACK_GLOBAL_GET;
    case OP_TYPE_WASMSSA_GLOBAL_SET: return OP_TYPE_WASMSTACK_GLOBAL_SET;
    case OP_TYPE_WASMSSA_EXTEND_I32_S: return OP_TYPE_WASMSTACK_EXTEND_I32_S;
    case OP_TYPE_WASMSSA_RETURN:     return OP_TYPE_WASMSTACK_RETURN;
    case OP_TYPE_WASMSSA_CALL:       return OP_TYPE_WASMSTACK_CALL;
    case OP_TYPE_WASMSSA_BLOCK_BEGIN: return OP_TYPE_WASMSTACK_BLOCK;
    case OP_TYPE_WASMSSA_LOOP_BEGIN:  return OP_TYPE_WASMSTACK_LOOP;
    case OP_TYPE_WASMSSA_IF_BEGIN:    return OP_TYPE_WASMSTACK_IF;
    case OP_TYPE_WASMSSA_IF_ELSE:     return OP_TYPE_WASMSTACK_ELSE;
    case OP_TYPE_WASMSSA_END:         return OP_TYPE_WASMSTACK_END;
    case OP_TYPE_WASMSSA_BR:          return OP_TYPE_WASMSTACK_BR;
    case OP_TYPE_WASMSSA_BR_IF:       return OP_TYPE_WASMSTACK_BR_IF;
    case OP_TYPE_WASMSSA_SELECT:      return OP_TYPE_WASMSTACK_SELECT;
    case OP_TYPE_WASMSSA_EQZ:         return OP_TYPE_WASMSTACK_EQZ;
    case OP_TYPE_WASMSSA_ADDRESSOF:   return OP_TYPE_WASMSTACK_ADDRESSOF;
    default: return (MLIR_OpType)0;
    }
}

// Stackify a single wasmssa func into the supplied wasmstack func.
static bool stackify_func(const wasmssa_func_t *src, wasmstack_func_t *dst) {
    // Mirror header (name, imports, params, results).
    dst->name = strdup(src->name);
    dst->imported = src->imported;
    dst->exported = src->exported;
    dst->n_params = src->n_params;
    dst->param_types = (uint8_t *)malloc(src->n_params ? src->n_params : 1);
    if (src->n_params) memcpy(dst->param_types, src->param_types, src->n_params);
    dst->n_results = src->n_results;
    dst->result_types = (uint8_t *)malloc(src->n_results ? src->n_results : 1);
    if (src->n_results) memcpy(dst->result_types, src->result_types, src->n_results);
    if (src->imported) return true;

    // vmap: ssa_def_idx -> wasmstack local idx.
    size_t n_ssa = src->n_params + src->n_ops;
    uint32_t *vmap = (uint32_t *)malloc((n_ssa ? n_ssa : 1) * sizeof(uint32_t));
    bool *bound = (bool *)calloc(n_ssa ? n_ssa : 1, sizeof(bool));
    for (size_t i = 0; i < src->n_params; i++) {
        vmap[i] = (uint32_t)i;
        bound[i] = true;
    }

    // Carrier id -> wasm local idx (allocated lazily on first use).
    uint32_t *cmap = NULL;
    bool     *cbound = NULL;
    if (src->n_carriers) {
        cmap   = (uint32_t *)malloc(src->n_carriers * sizeof(uint32_t));
        cbound = (bool *)calloc(src->n_carriers, sizeof(bool));
    }

    for (size_t i = 0; i < src->n_ops; i++) {
        const wasmssa_op_t *op = &src->ops[i];

        // Carriers desugar to local.set / local.get directly: don't push
        // operands via local.get for them (handled inline below).
        if (op->type == OP_TYPE_WASMSSA_CARRIER_SET) {
            uint32_t cid = op->carrier_id;
            if (cid >= src->n_carriers) {
                fprintf(stderr, "stackify: carrier id %u out of range\n", cid);
                goto fail;
            }
            if (!cbound[cid]) {
                cmap[cid] = wasmstack_func_add_local(dst, src->carrier_vts[cid]);
                cbound[cid] = true;
            }
            // Push operand value, then local.set carrier-local.
            int sidx = op->operands[0];
            if (sidx < 0 || (size_t)sidx >= n_ssa || !bound[sidx]) goto unbound;
            emit_local_get(dst, vmap[sidx]);
            emit_local_set(dst, cmap[cid]);
            continue;
        }
        if (op->type == OP_TYPE_WASMSSA_CARRIER_GET) {
            uint32_t cid = op->carrier_id;
            if (cid >= src->n_carriers) {
                fprintf(stderr, "stackify: carrier id %u out of range\n", cid);
                goto fail;
            }
            if (!cbound[cid]) {
                // Possible only if the producing region was unreachable;
                // allocate anyway with the recorded valtype.
                cmap[cid] = wasmstack_func_add_local(dst, src->carrier_vts[cid]);
                cbound[cid] = true;
            }
            emit_local_get(dst, cmap[cid]);
            // CARRIER_GET produces an SSA result: stash into a fresh local.
            uint32_t li = wasmstack_func_add_local(dst, op->valtype);
            emit_local_set(dst, li);
            vmap[src->n_params + i] = li;
            bound[src->n_params + i] = true;
            continue;
        }

        // Push each operand by local.get vmap[idx].
        for (int k = 0; k < op->n_operands; k++) {
            int sidx = op->operands[k];
            if (sidx < 0 || (size_t)sidx >= n_ssa || !bound[sidx]) {
            unbound:
                fprintf(stderr,
                        "stackify: unbound operand (op %zu, opnd %d, sidx %d)\n",
                        i, k, sidx);
                goto fail;
            }
            emit_local_get(dst, vmap[sidx]);
        }

        // Emit the corresponding stack op with immediates copied through.
        MLIR_OpType st = ssa_to_stack(op->type);
        if (st == 0) {
            fprintf(stderr, "stackify: unknown wasmssa op (enum=%d)\n",
                    (int)op->type);
            goto fail;
        }
        wasmstack_op_t *out = wasmstack_func_add_op(dst);
        out->type              = st;
        out->valtype           = op->valtype;
        out->i_const           = op->i_const;
        out->global_idx        = op->global_idx;
        out->memory_offset     = op->memory_offset;
        out->memory_align_log2 = op->memory_align_log2;
        out->mem_size_bytes    = op->mem_size_bytes;
        out->wasm_opcode       = op->wasm_opcode;
        out->br_depth          = op->br_depth;
        if (op->call_target) out->call_target = strdup(op->call_target);

        // Stash the result into a fresh local.
        if (op->has_result) {
            uint32_t li = wasmstack_func_add_local(dst, op->valtype);
            emit_local_set(dst, li);
            vmap[src->n_params + i] = li;
            bound[src->n_params + i] = true;
        }
    }

    free(vmap); free(bound); free(cmap); free(cbound);
    return true;
fail:
    free(vmap); free(bound); free(cmap); free(cbound);
    return false;
}

wasmstack_module_t *mlir_stackify_wasmssa(wasmssa_module_t *m) {
    if (!m) return NULL;
    wasmstack_module_t *out = wasmstack_module_new();
    for (size_t i = 0; i < m->n_funcs; i++) {
        wasmstack_func_t *df = wasmstack_module_add_func(out);
        if (!stackify_func(&m->funcs[i], df)) {
            wasmstack_module_free(out);
            return NULL;
        }
    }
    // Move globals over verbatim (transfer ownership of name/data/relocs).
    for (size_t i = 0; i < m->n_globals; i++) {
        wasm_global_t *src = &m->globals[i];
        wasm_global_t *dst = wasmstack_module_add_global(out);
        dst->name = src->name;       src->name = NULL;
        dst->data = src->data;       src->data = NULL;
        dst->size = src->size;
        dst->align_pow = src->align_pow;
        dst->is_const = src->is_const;
        dst->relocs = src->relocs;   src->relocs = NULL;
        dst->n_relocs = src->n_relocs; src->n_relocs = 0;
        dst->c_relocs = src->c_relocs; src->c_relocs = 0;
    }
    return out;
}

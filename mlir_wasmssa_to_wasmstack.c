// Stage 2 of the native LLVM->WASM pipeline:
// wasmssa builtin.module (MLIR) -> wasmstack builtin.module (MLIR).
//
// This is the "stackification" stage. v1 implementation is naive:
// every wasmssa SSA value materializes into a fresh wasmstack local.
// Nothing rides the operand stack between adjacent ops.
//
// Per wasmssa op (in declaration order):
//   for each operand value:
//       emit  wasmstack.local.get vmap[operand]
//   emit the corresponding wasmstack op (no operands; immediates copied)
//   if op produces a result:
//       allocate new local of the result's valtype
//       emit  wasmstack.local.set new_local
//       vmap[op_result_value] = new_local
//
// Structured CF ops (`wasmssa.block`/`loop`/`if`) open/close wasmstack
// labels with blocktype 0x40 (empty signature). Per-region "result locals"
// receive values from `wasmssa.block_return` and from `wasmssa.br` to that
// label; loops additionally allocate "entry-arg locals" that receive
// back-edge values.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_llvm_to_wasmssa.h"
#include "mlir_wasmssa_to_wasmstack.h"

// =============================================================================
// MLIR builder helpers.
// =============================================================================
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
        case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_s_cstr(MLIR_Context *ctx, const char *name,
                                        const char *v) {
    return attr_s(ctx, name, v, v ? strlen(v) : 0);
}
static char *hex_encode_arena(Arena *arena, const uint8_t *p, size_t n) {
    char *out = (char *)arena_alloc(arena, n ? n * 2 : 1);
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i*2 + 0] = d[(p[i] >> 4) & 0xf];
        out[i*2 + 1] = d[p[i] & 0xf];
    }
    return out;
}
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n) {
    return attr_s(ctx, name, hex_encode_arena(arena, p, n), n * 2);
}
static uint8_t *hex_decode(string s, size_t *out_n) {
    if (s.size & 1) { *out_n = 0; return NULL; }
    size_t n = s.size / 2;
    uint8_t *out = (uint8_t *)malloc(n ? n : 1);
    for (size_t i = 0; i < n; i++) {
        int hi = -1, lo = -1;
        char a = s.str[i*2], b = s.str[i*2+1];
        if (a >= '0' && a <= '9') hi = a - '0';
        else if (a >= 'a' && a <= 'f') hi = a - 'a' + 10;
        if (b >= '0' && b <= '9') lo = b - '0';
        else if (b >= 'a' && b <= 'f') lo = b - 'a' + 10;
        if (hi < 0 || lo < 0) { free(out); *out_n = 0; return NULL; }
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    *out_n = n;
    return out;
}

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static int64_t at_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool at_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
static string at_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}

// wasmssa -> wasmstack op type. Returns 0 on unknown.
static MLIR_OpType ssa_to_stack(MLIR_OpType t) {
    switch (t) {
    case OP_TYPE_WASMSSA_CONST:        return OP_TYPE_WASMSTACK_CONST;
    case OP_TYPE_WASMSSA_ADD:          return OP_TYPE_WASMSTACK_ADD;
    case OP_TYPE_WASMSSA_SUB:          return OP_TYPE_WASMSTACK_SUB;
    case OP_TYPE_WASMSSA_BINOP:        return OP_TYPE_WASMSTACK_BINOP;
    case OP_TYPE_WASMSSA_UNOP:         return OP_TYPE_WASMSTACK_UNOP;
    case OP_TYPE_WASMSSA_LOAD:         return OP_TYPE_WASMSTACK_LOAD;
    case OP_TYPE_WASMSSA_STORE:        return OP_TYPE_WASMSTACK_STORE;
    case OP_TYPE_WASMSSA_GLOBAL_GET:   return OP_TYPE_WASMSTACK_GLOBAL_GET;
    case OP_TYPE_WASMSSA_GLOBAL_SET:   return OP_TYPE_WASMSTACK_GLOBAL_SET;
    case OP_TYPE_WASMSSA_EXTEND_I32_S: return OP_TYPE_WASMSTACK_EXTEND_I32_S;
    case OP_TYPE_WASMSSA_RETURN:       return OP_TYPE_WASMSTACK_RETURN;
    case OP_TYPE_WASMSSA_CALL:         return OP_TYPE_WASMSTACK_CALL;
    case OP_TYPE_WASMSSA_BR:           return OP_TYPE_WASMSTACK_BR;
    case OP_TYPE_WASMSSA_BR_IF:        return OP_TYPE_WASMSTACK_BR_IF;
    case OP_TYPE_WASMSSA_SELECT:       return OP_TYPE_WASMSTACK_SELECT;
    case OP_TYPE_WASMSSA_EQZ:          return OP_TYPE_WASMSTACK_EQZ;
    case OP_TYPE_WASMSSA_ADDRESSOF:    return OP_TYPE_WASMSTACK_ADDRESSOF;
    case OP_TYPE_WASMSSA_FUNC_ADDR:    return OP_TYPE_WASMSTACK_FUNC_ADDR;
    case OP_TYPE_WASMSSA_CALL_INDIRECT: return OP_TYPE_WASMSTACK_CALL_INDIRECT;
    case OP_TYPE_WASMSSA_MEMORY_SIZE:   return OP_TYPE_WASMSTACK_MEMORY_SIZE;
    case OP_TYPE_WASMSSA_MEMORY_GROW:   return OP_TYPE_WASMSTACK_MEMORY_GROW;
    default: return (MLIR_OpType)0;
    }
}

// Whether the wasmssa op produces an SSA result.
static bool ssa_op_has_result(MLIR_OpType t) {
    switch (t) {
    case OP_TYPE_WASMSSA_CONST:
    case OP_TYPE_WASMSSA_ADD:
    case OP_TYPE_WASMSSA_SUB:
    case OP_TYPE_WASMSSA_BINOP:
    case OP_TYPE_WASMSSA_UNOP:
    case OP_TYPE_WASMSSA_LOAD:
    case OP_TYPE_WASMSSA_GLOBAL_GET:
    case OP_TYPE_WASMSSA_EXTEND_I32_S:
    case OP_TYPE_WASMSSA_SELECT:
    case OP_TYPE_WASMSSA_EQZ:
    case OP_TYPE_WASMSSA_ADDRESSOF:
    case OP_TYPE_WASMSSA_FUNC_ADDR:
    case OP_TYPE_WASMSSA_MEMORY_SIZE:
    case OP_TYPE_WASMSSA_MEMORY_GROW:
        return true;
    default:
        return false;
    }
}

// =============================================================================
// Value->local map. Open-addressing hash table (key 0 == empty sentinel;
// wasmssa value handles are always non-zero). Replaces an earlier linear scan
// that dominated the wasmssa->wasmstack pass (O(n^2) per function).
// =============================================================================
typedef struct {
    MLIR_ValueHandle *keys;   // hash slots; 0 == empty
    uint32_t *vals;
    size_t n, cap;            // cap is 0 or a power of two
} VMap;

static size_t vmap_hash(MLIR_ValueHandle k) {
    // 32-bit-safe mix (identical on the wasm32 self-host and 64-bit host).
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}
static void vmap_grow(VMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 32;
    MLIR_ValueHandle *nk = (MLIR_ValueHandle *)calloc(ncap, sizeof(*nk));
    uint32_t *nv = (uint32_t *)malloc(ncap * sizeof(*nv));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < m->cap; i++) {
        if (m->keys[i] == MLIR_INVALID_HANDLE) continue;
        size_t j = vmap_hash(m->keys[i]) & mask;
        while (nk[j] != MLIR_INVALID_HANDLE) j = (j + 1) & mask;
        nk[j] = m->keys[i];
        nv[j] = m->vals[i];
    }
    free(m->keys);
    free(m->vals);
    m->keys = nk;
    m->vals = nv;
    m->cap = ncap;
}
static void vmap_set(VMap *m, MLIR_ValueHandle k, uint32_t v) {
    if ((m->n + 1) * 4 >= m->cap * 3) vmap_grow(m);
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->keys[i] != MLIR_INVALID_HANDLE) {
        if (m->keys[i] == k) return;  // first insert wins
        i = (i + 1) & mask;
    }
    m->keys[i] = k;
    m->vals[i] = v;
    m->n++;
}
static int vmap_get(VMap *m, MLIR_ValueHandle k, uint32_t *out) {
    if (m->cap == 0) return 0;
    size_t mask = m->cap - 1;
    size_t i = vmap_hash(k) & mask;
    while (m->keys[i] != MLIR_INVALID_HANDLE) {
        if (m->keys[i] == k) { *out = m->vals[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

// =============================================================================
// Per-function locals collector.
// =============================================================================
typedef struct {
    uint8_t *types;
    size_t n, cap;
    size_t n_params;
} Locals;

static uint32_t locals_add(Locals *L, uint8_t vt) {
    if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 16;
        L->types = (uint8_t *)realloc(L->types, L->cap);
    }
    L->types[L->n] = vt;
    return (uint32_t)(L->n_params + L->n++);
}

// =============================================================================
// Region-stack: tracks open wasmstack labels (block/loop/if) so that
// wasmssa.br {depth=N} (vals) can route values into the right per-target
// locals.
// =============================================================================
typedef struct RegionFrame {
    MLIR_OpType  op_type;            // OP_TYPE_WASMSSA_BLOCK/LOOP/IF
    uint32_t    *result_locals;      // for fall-through values out of label
    size_t       n_results;
    uint32_t    *entry_arg_locals;   // loop only: locals for entry-block args
    size_t       n_entry_args;
} RegionFrame;

typedef struct {
    RegionFrame *data;
    size_t       n, cap;
} RegionStack;

static void rs_push(RegionStack *rs, RegionFrame f) {
    if (rs->n == rs->cap) {
        rs->cap = rs->cap ? rs->cap * 2 : 8;
        rs->data = (RegionFrame *)realloc(rs->data, rs->cap * sizeof(RegionFrame));
    }
    rs->data[rs->n++] = f;
}
static void rs_pop(RegionStack *rs) { rs->n--; }
static RegionFrame *rs_at_depth(RegionStack *rs, uint32_t depth) {
    if (depth >= rs->n) return NULL;
    return &rs->data[rs->n - 1 - depth];
}

// =============================================================================
// Op emitters into a wasmstack func body block.
// =============================================================================
static void emit_local_get(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t li) {
    MLIR_AttributeHandle as[2];
    as[0] = attr_i32(ctx, "valtype", 0);
    as[1] = attr_i32(ctx, "local_idx", (int64_t)li);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_WASMSTACK_LOCAL_GET, as, 2));
}
static void emit_local_set(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t li) {
    MLIR_AttributeHandle as[2];
    as[0] = attr_i32(ctx, "valtype", 0);
    as[1] = attr_i32(ctx, "local_idx", (int64_t)li);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_WASMSTACK_LOCAL_SET, as, 2));
}

// Copy immediates from a wasmssa op `bo` onto the parallel wasmstack op
// (with all the same attribute encodings).
static void copy_imms(MLIR_Context *ctx, Arena *arena, MLIR_OpHandle bo,
                      MLIR_AttributeHandle *as, size_t *pna) {
    MLIR_OpType t = MLIR_GetOpType(bo);
    size_t nas = *pna;
    switch (t) {
    case OP_TYPE_WASMSSA_CONST:
        as[nas++] = attr_i64(ctx, "value", at_i(bo, "value"));
        break;
    case OP_TYPE_WASMSSA_BINOP:
    case OP_TYPE_WASMSSA_UNOP:
        as[nas++] = attr_i32(ctx, "wasm_opcode", at_i(bo, "wasm_opcode"));
        break;
    case OP_TYPE_WASMSSA_LOAD:
    case OP_TYPE_WASMSSA_STORE:
        as[nas++] = attr_i32(ctx, "memory_offset",     at_i(bo, "memory_offset"));
        as[nas++] = attr_i32(ctx, "memory_align_log2", at_i(bo, "memory_align_log2"));
        as[nas++] = attr_i32(ctx, "mem_size_bytes",    at_i(bo, "mem_size_bytes"));
        break;
    case OP_TYPE_WASMSSA_GLOBAL_GET:
    case OP_TYPE_WASMSSA_GLOBAL_SET:
        as[nas++] = attr_i32(ctx, "global_idx", at_i(bo, "global_idx"));
        break;
    case OP_TYPE_WASMSSA_CALL:
    case OP_TYPE_WASMSSA_ADDRESSOF:
    case OP_TYPE_WASMSSA_FUNC_ADDR: {
        string tgt = at_s(bo, "target");
        as[nas++] = attr_s(ctx, "target", tgt.str, tgt.size);
        break;
    }
    case OP_TYPE_WASMSSA_CALL_INDIRECT: {
        string sp = at_s(bo, "sig_params");
        string sr = at_s(bo, "sig_results");
        as[nas++] = attr_s(ctx, "sig_params", sp.str, sp.size);
        as[nas++] = attr_s(ctx, "sig_results", sr.str, sr.size);
        (void)arena;
        break;
    }
    case OP_TYPE_WASMSSA_BR:
    case OP_TYPE_WASMSSA_BR_IF:
        as[nas++] = attr_i32(ctx, "depth", at_i(bo, "depth"));
        break;
    default: break;
    }
    *pna = nas;
}

// =============================================================================
// Stackify a wasmssa.func body region into a wasmstack.func body block.
// =============================================================================

// Determine the wasm valtype of an MLIR Value (i32/i64/f32/f64).
static uint8_t value_to_vt(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle ty = MLIR_GetValueType(v);
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return WT_I32;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return WT_I32;
    if (s.size == 5 && memcmp(s.str, "index", 5) == 0) return WT_I32;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return WT_F32;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return WT_F64;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8 || w == 16 || w == 32) return WT_I32;
        if (w == 64) return WT_I64;
    }
    return WT_I32;
}

// Emit `wasmstack.<op>` with no operands (any source operands have already
// been pushed via local.get). copy_imms handles the per-op attrs.
static void emit_stack_simple(MLIR_Context *ctx, Arena *arena, MLIR_BlockHandle blk,
                              MLIR_OpHandle bo, MLIR_OpType st, uint8_t valtype) {
    MLIR_AttributeHandle as[8];
    size_t nas = 0;
    as[nas++] = attr_i32(ctx, "valtype", valtype);
    copy_imms(ctx, arena, bo, as, &nas);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, st, as, nas));
}

// Walk one wasmssa block, emitting wasmstack ops. Recursively descends
// into region-bearing ops (BLOCK/LOOP/IF).
static bool stackify_walk_block(MLIR_Context *ctx, Arena *arena,
                                MLIR_BlockHandle src_blk, MLIR_BlockHandle dst_blk,
                                VMap *vmap, Locals *L, RegionStack *rs);

// Open a region-bearing wasmssa op: emit its wasmstack opener (block/loop/if),
// recursively walk the inner block(s), emit `wasmstack.end`, and bind the
// wasmssa op's results to fresh locals. Returns false on error.
static bool stackify_region_op(MLIR_Context *ctx, Arena *arena,
                               MLIR_OpHandle bo, MLIR_BlockHandle dst_blk,
                               VMap *vmap, Locals *L, RegionStack *rs) {
    MLIR_OpType t = MLIR_GetOpType(bo);
    size_t n_results = MLIR_GetOpNumResults(bo);
    size_t n_operands = MLIR_GetOpNumOperands(bo);

    // Pre-allocate result locals.
    RegionFrame fr = {0};
    fr.op_type = t;
    fr.n_results = n_results;
    if (n_results) {
        fr.result_locals = (uint32_t *)arena_alloc(arena, n_results * sizeof(uint32_t));
        for (size_t i = 0; i < n_results; i++) {
            MLIR_ValueHandle r = MLIR_GetOpResult(bo, i);
            fr.result_locals[i] = locals_add(L, value_to_vt(ctx, r));
        }
    }

    if (t == OP_TYPE_WASMSSA_LOOP) {
        // Loop: copy init operands into entry-arg locals before opening.
        fr.n_entry_args = n_operands;
        if (n_operands) {
            fr.entry_arg_locals = (uint32_t *)arena_alloc(arena, n_operands * sizeof(uint32_t));
        }
        if (MLIR_GetOpNumRegions(bo) < 1) return false;
        MLIR_BlockHandle entry = MLIR_GetRegionBlock(MLIR_GetOpRegion(bo, 0), 0);
        if (MLIR_GetBlockNumArgs(entry) != n_operands) return false;
        for (size_t i = 0; i < n_operands; i++) {
            MLIR_ValueHandle ba = MLIR_GetBlockArg(entry, i);
            uint8_t vt = value_to_vt(ctx, ba);
            fr.entry_arg_locals[i] = locals_add(L, vt);
            uint32_t li;
            if (!vmap_get(vmap, MLIR_GetOpOperand(bo, i), &li)) return false;
            emit_local_get(ctx, dst_blk, li);
            emit_local_set(ctx, dst_blk, fr.entry_arg_locals[i]);
            vmap_set(vmap, ba, fr.entry_arg_locals[i]);
        }
    } else if (t == OP_TYPE_WASMSSA_IF) {
        // If: push condition local before the wasmstack.if op.
        if (n_operands != 1) return false;
        uint32_t cli;
        if (!vmap_get(vmap, MLIR_GetOpOperand(bo, 0), &cli)) return false;
        emit_local_get(ctx, dst_blk, cli);
    }

    // Open the wasmstack label. Always blocktype 0x40 (empty); cross-CF
    // values flow via the per-region locals we allocated above.
    MLIR_OpType opener = (t == OP_TYPE_WASMSSA_BLOCK) ? OP_TYPE_WASMSTACK_BLOCK
                       : (t == OP_TYPE_WASMSSA_LOOP)  ? OP_TYPE_WASMSTACK_LOOP
                       :                                 OP_TYPE_WASMSTACK_IF;
    {
        MLIR_AttributeHandle as[1];
        as[0] = attr_i32(ctx, "valtype", 0x40);
        MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, opener, as, 1));
    }

    rs_push(rs, fr);

    // Walk inner region(s).
    if (t == OP_TYPE_WASMSSA_IF) {
        // then-region.
        if (MLIR_GetOpNumRegions(bo) < 1) { rs_pop(rs); return false; }
        MLIR_BlockHandle then_blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(bo, 0), 0);
        if (!stackify_walk_block(ctx, arena, then_blk, dst_blk, vmap, L, rs)) {
            rs_pop(rs); return false;
        }
        bool has_else = MLIR_GetOpNumRegions(bo) >= 2 &&
                        MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(bo, 1)) > 0;
        if (has_else) {
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_ELSE, as, 1));
            MLIR_BlockHandle else_blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(bo, 1), 0);
            if (!stackify_walk_block(ctx, arena, else_blk, dst_blk, vmap, L, rs)) {
                rs_pop(rs); return false;
            }
        }
    } else {
        // block / loop: single region.
        if (MLIR_GetOpNumRegions(bo) < 1) { rs_pop(rs); return false; }
        MLIR_BlockHandle inner = MLIR_GetRegionBlock(MLIR_GetOpRegion(bo, 0), 0);
        if (!stackify_walk_block(ctx, arena, inner, dst_blk, vmap, L, rs)) {
            rs_pop(rs); return false;
        }
    }

    rs_pop(rs);

    // Close the label.
    {
        MLIR_AttributeHandle as[1];
        as[0] = attr_i32(ctx, "valtype", 0);
        MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_END, as, 1));
    }

    // Bind wasmssa op's results to result_locals.
    for (size_t i = 0; i < n_results; i++) {
        vmap_set(vmap, MLIR_GetOpResult(bo, i), fr.result_locals[i]);
    }
    return true;
}

static bool stackify_walk_block(MLIR_Context *ctx, Arena *arena,
                                MLIR_BlockHandle src_blk, MLIR_BlockHandle dst_blk,
                                VMap *vmap, Locals *L, RegionStack *rs) {
    size_t n_body = MLIR_GetBlockNumOps(src_blk);
    for (size_t i = 0; i < n_body; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(src_blk, i);
        MLIR_OpType t = MLIR_GetOpType(bo);
        uint8_t valtype = (uint8_t)at_i(bo, "valtype");

        // ---- Region-bearing structured CF ops ----
        if (t == OP_TYPE_WASMSSA_BLOCK || t == OP_TYPE_WASMSSA_LOOP || t == OP_TYPE_WASMSSA_IF) {
            if (!stackify_region_op(ctx, arena, bo, dst_blk, vmap, L, rs)) {
                return false;
            }
            continue;
        }
        if (t == OP_TYPE_WASMSSA_BLOCK_RETURN) {
            // Copy yield operands into the enclosing region's result_locals.
            RegionFrame *top = rs->n ? &rs->data[rs->n - 1] : NULL;
            if (!top) { fprintf(stderr, "stackify: block_return with no enclosing region\n"); return false; }
            size_t no = MLIR_GetOpNumOperands(bo);
            if (no != top->n_results) {
                fprintf(stderr, "stackify: block_return arity %zu != region results %zu\n", no, top->n_results);
                return false;
            }
            for (size_t k = 0; k < no; k++) {
                uint32_t li;
                if (!vmap_get(vmap, MLIR_GetOpOperand(bo, k), &li)) {
                    fprintf(stderr, "stackify: unbound block_return operand %zu\n", k); return false;
                }
                emit_local_get(ctx, dst_blk, li);
                emit_local_set(ctx, dst_blk, top->result_locals[k]);
            }
            continue;
        }
        if (t == OP_TYPE_WASMSSA_UNREACHABLE) {
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_UNREACHABLE, as, 1));
            continue;
        }

        // ---- wasmssa.br with operands: transfer values into target locals ----
        if (t == OP_TYPE_WASMSSA_BR && MLIR_GetOpNumOperands(bo) > 0) {
            uint32_t depth = (uint32_t)at_i(bo, "depth");
            RegionFrame *target = rs_at_depth(rs, depth);
            if (!target) { fprintf(stderr, "stackify: br depth %u out of range\n", depth); return false; }
            uint32_t *dst_locals;
            size_t n_dst;
            if (target->op_type == OP_TYPE_WASMSSA_LOOP) {
                // br to a loop label = back-edge: feed entry-arg locals.
                dst_locals = target->entry_arg_locals; n_dst = target->n_entry_args;
            } else {
                dst_locals = target->result_locals; n_dst = target->n_results;
            }
            size_t no = MLIR_GetOpNumOperands(bo);
            if (no != n_dst) {
                fprintf(stderr, "stackify: br arity %zu != target slot count %zu\n", no, n_dst);
                return false;
            }
            for (size_t k = 0; k < no; k++) {
                uint32_t li;
                if (!vmap_get(vmap, MLIR_GetOpOperand(bo, k), &li)) {
                    fprintf(stderr, "stackify: unbound br operand %zu\n", k); return false;
                }
                emit_local_get(ctx, dst_blk, li);
                emit_local_set(ctx, dst_blk, dst_locals[k]);
            }
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_i32(ctx, "depth", (int64_t)depth);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_BR, as, 2));
            continue;
        }

        // ---- Generic: push operands as local.get, emit op, stash result ----
        size_t n_ops = MLIR_GetOpNumOperands(bo);
        for (size_t k = 0; k < n_ops; k++) {
            uint32_t li;
            MLIR_ValueHandle ov = MLIR_GetOpOperand(bo, k);
            if (!vmap_get(vmap, ov, &li)) {
                fprintf(stderr, "stackify: unbound operand on op (idx %zu)\n", k);
                return false;
            }
            emit_local_get(ctx, dst_blk, li);
        }

        MLIR_OpType st = ssa_to_stack(t);
        if (!st) {
            fprintf(stderr, "stackify: unknown wasmssa op enum=%d\n", (int)t);
            return false;
        }
        emit_stack_simple(ctx, arena, dst_blk, bo, st, valtype);

        if (MLIR_GetOpNumResults(bo) > 0) {
            uint32_t li = locals_add(L, valtype);
            emit_local_set(ctx, dst_blk, li);
            vmap_set(vmap, MLIR_GetOpResult(bo, 0), li);
        }
    }
    return true;
}

static bool stackify_body(MLIR_Context *ctx, Arena *arena,
                          MLIR_OpHandle src_func, MLIR_BlockHandle dst_blk,
                          const uint8_t *param_types, size_t n_params,
                          Locals *L) {
    (void)param_types;
    L->n_params = n_params;

    if (MLIR_GetOpNumRegions(src_func) < 1) return true;
    MLIR_RegionHandle r = MLIR_GetOpRegion(src_func, 0);
    if (MLIR_GetRegionNumBlocks(r) < 1) return true;
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(r, 0);

    VMap vmap = {0};
    size_t n_args = MLIR_GetBlockNumArgs(src_blk);
    for (size_t i = 0; i < n_args; i++) {
        vmap_set(&vmap, MLIR_GetBlockArg(src_blk, i), (uint32_t)i);
    }

    RegionStack rs = {0};
    bool ok = stackify_walk_block(ctx, arena, src_blk, dst_blk, &vmap, L, &rs);

    free(rs.data);
    free(vmap.keys); free(vmap.vals);
    return ok;
}

// =============================================================================
// Per-top-level-op converters.
// =============================================================================
static MLIR_OpHandle convert_func(MLIR_Context *ctx, Arena *arena, MLIR_OpHandle src) {
    string name = at_s(src, "sym_name");
    bool exported = at_b(src, "exported");
    bool internal = at_b(src, "internal");
    string pt = at_s(src, "param_types");
    string rt = at_s(src, "result_types");
    size_t n_params = pt.size / 2;
    uint8_t *param_types = hex_decode(pt, &n_params);
    // Optional wasm-attribute forwarding: import_module/import_name on
    // imports, export_name on definitions.
    string imod = at_s(src, "import_module");
    string iname = at_s(src, "import_name");
    string ename = at_s(src, "export_name");

    MLIR_AttributeHandle attrs[12];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", name.str, name.size);
    attrs[na++] = attr_s(ctx, "param_types", pt.str, pt.size);
    attrs[na++] = attr_s(ctx, "result_types", rt.str, rt.size);
    attrs[na++] = attr_b(ctx, "exported", exported);
    attrs[na++] = attr_b(ctx, "internal", internal);
    if (imod.size > 0)  attrs[na++] = attr_s(ctx, "import_module", imod.str, imod.size);
    if (iname.size > 0) attrs[na++] = attr_s(ctx, "import_name", iname.str, iname.size);
    if (ename.size > 0) attrs[na++] = attr_s(ctx, "export_name", ename.str, ename.size);

    bool is_import = MLIR_GetOpType(src) == OP_TYPE_WASMSSA_IMPORT_FUNC;
    if (is_import) {
        free(param_types);
        return build_op(ctx, OP_TYPE_WASMSTACK_IMPORT_FUNC, attrs, na);
    }

    MLIR_BlockHandle dst_blk = MLIR_CreateBlock(ctx);
    Locals L = {0};
    bool ok = stackify_body(ctx, arena, src, dst_blk, param_types, n_params, &L);
    free(param_types);
    if (!ok) { free(L.types); return MLIR_INVALID_HANDLE; }

    attrs[na++] = attr_s_hex(ctx, arena, "local_types", L.types, L.n);
    free(L.types);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, dst_blk);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSTACK_FUNC,
        op_type_to_string(OP_TYPE_WASMSTACK_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

static MLIR_OpHandle convert_global(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name    = at_s(src, "sym_name");
    string idata   = at_s(src, "init_data");
    string relocs  = at_s(src, "relocs");
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name",  name.str, name.size);
    attrs[na++] = attr_i32(ctx, "size",      at_i(src, "size"));
    attrs[na++] = attr_i32(ctx, "align_pow", at_i(src, "align_pow"));
    attrs[na++] = attr_b(ctx, "is_const",    at_b(src, "is_const"));
    attrs[na++] = attr_s(ctx, "init_data", idata.str, idata.size);
    if (relocs.size) attrs[na++] = attr_s(ctx, "relocs", relocs.str, relocs.size);
    return build_op(ctx, OP_TYPE_WASMSTACK_IMPORT_GLOBAL, attrs, na);
}

// =============================================================================
// Top-level: walk input module body, build output module body.
// =============================================================================
MLIR_OpHandle mlir_wasmssa_to_wasmstack(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    if (!ssa_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(ssa_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ssa_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);

    Arena *arena = MLIR_GetArenaAllocator(ctx);

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    size_t n_top = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        MLIR_OpHandle out_op = MLIR_INVALID_HANDLE;
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC || t == OP_TYPE_WASMSSA_FUNC) {
            out_op = convert_func(ctx, arena, top);
            if (!out_op) return MLIR_INVALID_HANDLE;
        } else if (t == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            out_op = convert_global(ctx, top);
        } else {
            string nm = MLIR_GetOpName(top);
            fprintf(stderr, "stackify: unexpected top-level op '%.*s'\n",
                    (int)nm.size, nm.str);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AppendBlockOp(ctx, out_body, out_op);
    }
    return out_module;
}

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
static MLIR_NamedAttr attr_i32(MLIR_Context *ctx, MLIR_AttrKey key, int64_t v) {
    return MLIR_Named(key, MLIR_CreateAttributeInteger(ctx, v,
                                       MLIR_CreateTypeInteger(ctx, 32, true)));
}
static MLIR_NamedAttr attr_i64(MLIR_Context *ctx, MLIR_AttrKey key, int64_t v) {
    return MLIR_Named(key, MLIR_CreateAttributeInteger(ctx, v,
                                       MLIR_CreateTypeInteger(ctx, 64, true)));
}
static MLIR_NamedAttr attr_b(MLIR_Context *ctx, MLIR_AttrKey key, bool v) {
    return MLIR_Named(key, MLIR_CreateAttributeBool(ctx, v));
}
static MLIR_NamedAttr attr_s(MLIR_Context *ctx, MLIR_AttrKey key,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_Named(key, MLIR_CreateAttributeString(ctx, sv));
}
static MLIR_NamedAttr attr_s_cstr(MLIR_Context *ctx, MLIR_AttrKey key,
                                        const char *v) {
    return attr_s(ctx, key, v, v ? strlen(v) : 0);
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
static MLIR_NamedAttr attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       MLIR_AttrKey key,
                                       const uint8_t *p, size_t n) {
    return attr_s(ctx, key, hex_encode_arena(arena, p, n), n * 2);
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
                              MLIR_NamedAttr *attrs, size_t na) {
    MLIR_OpSpec spec = {0};
    spec.type = t;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)na;
    return MLIR_CreateOpFromSpec(ctx, &spec);
}

static int64_t at_i(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, key);
    return a ? MLIR_GetAttributeInteger(ctx, a) : 0;
}
static bool at_b(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, key);
    return a ? MLIR_GetAttributeBool(ctx, a) : false;
}
static string at_s(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, key);
    return a ? MLIR_GetAttributeString(ctx, a) : (string){0};
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
// Per-module cache for hot types, attributes, and reusable scratch tables.
// =============================================================================
#define SC_LOCAL_CACHE_N 512

typedef struct {
    MLIR_TypeHandle i32_ty;
    MLIR_TypeHandle i64_ty;
    MLIR_AttributeHandle attr_i32_0;
    MLIR_AttributeHandle attr_i32_40;
    MLIR_AttributeHandle attr_i64_0;
    MLIR_AttributeHandle attr_local[SC_LOCAL_CACHE_N];
    VMap vmap;
    RegionStack rs;
    Locals locals;
} StackCache;

static void sc_init(MLIR_Context *ctx, StackCache *sc) {
    memset(sc, 0, sizeof(*sc));
    sc->i32_ty = MLIR_CreateTypeInteger(ctx, 32, true);
    sc->i64_ty = MLIR_CreateTypeInteger(ctx, 64, true);
    sc->attr_i32_0  = MLIR_CreateAttributeInteger(ctx, 0, sc->i32_ty);
    sc->attr_i32_40 = MLIR_CreateAttributeInteger(ctx, 0x40, sc->i32_ty);
    sc->attr_i64_0  = MLIR_CreateAttributeInteger(ctx, 0, sc->i64_ty);
}

static void sc_destroy(StackCache *sc) {
    free(sc->vmap.keys);
    free(sc->vmap.vals);
    free(sc->rs.data);
    free(sc->locals.types);
}

static MLIR_AttributeHandle sc_attr_local(MLIR_Context *ctx, StackCache *sc, uint32_t li) {
    if (li < SC_LOCAL_CACHE_N) {
        MLIR_AttributeHandle a = sc->attr_local[li];
        if (!a) {
            a = MLIR_CreateAttributeInteger(ctx, (int64_t)li, sc->i32_ty);
            sc->attr_local[li] = a;
        }
        return a;
    }
    return MLIR_CreateAttributeInteger(ctx, (int64_t)li, sc->i32_ty);
}

static MLIR_NamedAttr fwd_attr_i32(MLIR_Context *ctx, StackCache *sc,
                                   MLIR_OpHandle bo, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, bo, key);
    return MLIR_Named(key, a ? a : sc->attr_i32_0);
}
static MLIR_NamedAttr fwd_attr_i64(MLIR_Context *ctx, StackCache *sc,
                                   MLIR_OpHandle bo, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, bo, key);
    return MLIR_Named(key, a ? a : sc->attr_i64_0);
}
static MLIR_NamedAttr fwd_attr_s(MLIR_Context *ctx, MLIR_OpHandle bo, MLIR_AttrKey key) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, bo, key);
    if (a) return MLIR_Named(key, a);
    return attr_s(ctx, key, "", 0);
}

// =============================================================================
// Op emitters into a wasmstack func body block.
// =============================================================================
static void emit_local_get(MLIR_Context *ctx, StackCache *sc, MLIR_BlockHandle blk, uint32_t li) {
    MLIR_NamedAttr as[2];
    as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_0);
    as[1] = MLIR_Named(MLIR_KEY_LOCAL_IDX, sc_attr_local(ctx, sc, li));
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_WASMSTACK_LOCAL_GET, as, 2));
}
static void emit_local_set(MLIR_Context *ctx, StackCache *sc, MLIR_BlockHandle blk, uint32_t li) {
    MLIR_NamedAttr as[2];
    as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_0);
    as[1] = MLIR_Named(MLIR_KEY_LOCAL_IDX, sc_attr_local(ctx, sc, li));
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_WASMSTACK_LOCAL_SET, as, 2));
}

// Copy immediates from a wasmssa op `bo` onto the parallel wasmstack op
// by forwarding existing AttributeHandles directly (zero re-interning).
static void copy_imms(MLIR_Context *ctx, StackCache *sc, Arena *arena,
                      MLIR_OpHandle bo, MLIR_OpType t,
                      MLIR_NamedAttr *as, size_t *pna) {
    size_t nas = *pna;
    (void)arena;
    switch (t) {
    case OP_TYPE_WASMSSA_CONST:
        as[nas++] = fwd_attr_i64(ctx, sc, bo, MLIR_KEY_VALUE);
        break;
    case OP_TYPE_WASMSSA_BINOP:
    case OP_TYPE_WASMSSA_UNOP:
        as[nas++] = fwd_attr_i32(ctx, sc, bo, MLIR_KEY_WASM_OPCODE);
        break;
    case OP_TYPE_WASMSSA_LOAD:
    case OP_TYPE_WASMSSA_STORE:
        as[nas++] = fwd_attr_i32(ctx, sc, bo, MLIR_KEY_MEMORY_OFFSET);
        as[nas++] = fwd_attr_i32(ctx, sc, bo, MLIR_KEY_MEMORY_ALIGN_LOG2);
        as[nas++] = fwd_attr_i32(ctx, sc, bo, MLIR_KEY_MEM_SIZE_BYTES);
        break;
    case OP_TYPE_WASMSSA_GLOBAL_GET:
    case OP_TYPE_WASMSSA_GLOBAL_SET:
        as[nas++] = fwd_attr_i32(ctx, sc, bo, MLIR_KEY_GLOBAL_IDX);
        break;
    case OP_TYPE_WASMSSA_CALL:
    case OP_TYPE_WASMSSA_ADDRESSOF:
    case OP_TYPE_WASMSSA_FUNC_ADDR:
        as[nas++] = fwd_attr_s(ctx, bo, MLIR_KEY_TARGET);
        break;
    case OP_TYPE_WASMSSA_CALL_INDIRECT:
        as[nas++] = fwd_attr_s(ctx, bo, MLIR_KEY_SIG_PARAMS);
        as[nas++] = fwd_attr_s(ctx, bo, MLIR_KEY_SIG_RESULTS);
        break;
    case OP_TYPE_WASMSSA_BR:
    case OP_TYPE_WASMSSA_BR_IF:
        as[nas++] = fwd_attr_i32(ctx, sc, bo, MLIR_KEY_DEPTH);
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
    MLIR_TypeHandle ty = MLIR_GetValueType(ctx, v);
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    uint32_t w = MLIR_GetTypeWidth(ctx, ty);
    if (k == MLIR_TYPE_KIND_FLOAT && w == 32) return WT_F32;
    if (k == MLIR_TYPE_KIND_FLOAT && w == 64) return WT_F64;
    if (k == MLIR_TYPE_KIND_INTEGER && w == 64) return WT_I64;
    // Pointers, index and the narrower integers (and anything else).
    return WT_I32;
}

// Emit `wasmstack.<op>` with no operands (any source operands have already
// been pushed via local.get). copy_imms handles the per-op attrs.
static void emit_stack_simple(MLIR_Context *ctx, StackCache *sc, Arena *arena,
                              MLIR_BlockHandle blk, MLIR_OpHandle bo,
                              MLIR_OpType bo_t, MLIR_OpType st,
                              MLIR_AttributeHandle vt_attr) {
    MLIR_NamedAttr as[8];
    size_t nas = 0;
    as[nas++] = MLIR_Named(MLIR_KEY_VALTYPE, vt_attr ? vt_attr : sc->attr_i32_0);
    copy_imms(ctx, sc, arena, bo, bo_t, as, &nas);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, st, as, nas));
}

// Walk one wasmssa block, emitting wasmstack ops. Recursively descends
// into region-bearing ops (BLOCK/LOOP/IF).
static bool stackify_walk_block(MLIR_Context *ctx, StackCache *sc, Arena *arena,
                                MLIR_BlockHandle src_blk, MLIR_BlockHandle dst_blk,
                                VMap *vmap, Locals *L, RegionStack *rs);

// Open a region-bearing wasmssa op: emit its wasmstack opener (block/loop/if),
// recursively walk the inner block(s), emit `wasmstack.end`, and bind the
// wasmssa op's results to fresh locals. Returns false on error.
static bool stackify_region_op(MLIR_Context *ctx, StackCache *sc, Arena *arena,
                               MLIR_OpHandle bo, MLIR_OpType t,
                               MLIR_BlockHandle dst_blk,
                               VMap *vmap, Locals *L, RegionStack *rs) {
    size_t n_results = MLIR_GetOpNumResults(ctx, bo);
    size_t n_operands = MLIR_GetOpNumOperands(ctx, bo);

    // Pre-allocate result locals.
    RegionFrame fr = {0};
    fr.op_type = t;
    fr.n_results = n_results;
    if (n_results) {
        fr.result_locals = (uint32_t *)arena_alloc(arena, n_results * sizeof(uint32_t));
        for (size_t i = 0; i < n_results; i++) {
            MLIR_ValueHandle r = MLIR_GetOpResult(ctx, bo, i);
            fr.result_locals[i] = locals_add(L, value_to_vt(ctx, r));
        }
    }

    if (t == OP_TYPE_WASMSSA_LOOP) {
        // Loop: copy init operands into entry-arg locals before opening.
        fr.n_entry_args = n_operands;
        if (n_operands) {
            fr.entry_arg_locals = (uint32_t *)arena_alloc(arena, n_operands * sizeof(uint32_t));
        }
        if (MLIR_GetOpNumRegions(ctx, bo) < 1) return false;
        MLIR_BlockHandle entry = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 0), 0);
        if (MLIR_GetBlockNumArgs(ctx, entry) != n_operands) return false;
        for (size_t i = 0; i < n_operands; i++) {
            MLIR_ValueHandle ba = MLIR_GetBlockArg(ctx, entry, i);
            uint8_t vt = value_to_vt(ctx, ba);
            fr.entry_arg_locals[i] = locals_add(L, vt);
            uint32_t li;
            if (!vmap_get(vmap, MLIR_GetOpOperand(ctx, bo, i), &li)) return false;
            emit_local_get(ctx, sc, dst_blk, li);
            emit_local_set(ctx, sc, dst_blk, fr.entry_arg_locals[i]);
            vmap_set(vmap, ba, fr.entry_arg_locals[i]);
        }
    } else if (t == OP_TYPE_WASMSSA_IF) {
        // If: push condition local before the wasmstack.if op.
        if (n_operands != 1) return false;
        uint32_t cli;
        if (!vmap_get(vmap, MLIR_GetOpOperand(ctx, bo, 0), &cli)) return false;
        emit_local_get(ctx, sc, dst_blk, cli);
    }

    // Open the wasmstack label. Always blocktype 0x40 (empty); cross-CF
    // values flow via the per-region locals we allocated above.
    MLIR_OpType opener = (t == OP_TYPE_WASMSSA_BLOCK) ? OP_TYPE_WASMSTACK_BLOCK
                       : (t == OP_TYPE_WASMSSA_LOOP)  ? OP_TYPE_WASMSTACK_LOOP
                       :                                 OP_TYPE_WASMSTACK_IF;
    {
        MLIR_NamedAttr as[1];
        as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_40);
        MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, opener, as, 1));
    }

    rs_push(rs, fr);

    // Walk inner region(s).
    if (t == OP_TYPE_WASMSSA_IF) {
        // then-region.
        if (MLIR_GetOpNumRegions(ctx, bo) < 1) { rs_pop(rs); return false; }
        MLIR_BlockHandle then_blk = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 0), 0);
        if (!stackify_walk_block(ctx, sc, arena, then_blk, dst_blk, vmap, L, rs)) {
            rs_pop(rs); return false;
        }
        bool has_else = MLIR_GetOpNumRegions(ctx, bo) >= 2 &&
                        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, bo, 1)) > 0;
        if (has_else) {
            MLIR_NamedAttr as[1];
            as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_ELSE, as, 1));
            MLIR_BlockHandle else_blk = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 1), 0);
            if (!stackify_walk_block(ctx, sc, arena, else_blk, dst_blk, vmap, L, rs)) {
                rs_pop(rs); return false;
            }
        }
    } else {
        // block / loop: single region.
        if (MLIR_GetOpNumRegions(ctx, bo) < 1) { rs_pop(rs); return false; }
        MLIR_BlockHandle inner = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 0), 0);
        if (!stackify_walk_block(ctx, sc, arena, inner, dst_blk, vmap, L, rs)) {
            rs_pop(rs); return false;
        }
    }

    rs_pop(rs);

    // Close the label.
    {
        MLIR_NamedAttr as[1];
        as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_0);
        MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_END, as, 1));
    }

    // Bind wasmssa op's results to result_locals.
    for (size_t i = 0; i < n_results; i++) {
        vmap_set(vmap, MLIR_GetOpResult(ctx, bo, i), fr.result_locals[i]);
    }
    return true;
}

static bool stackify_walk_block(MLIR_Context *ctx, StackCache *sc, Arena *arena,
                                MLIR_BlockHandle src_blk, MLIR_BlockHandle dst_blk,
                                VMap *vmap, Locals *L, RegionStack *rs) {
    size_t n_body = MLIR_GetBlockNumOps(ctx, src_blk);
    for (size_t i = 0; i < n_body; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(ctx, src_blk, i);
        MLIR_OpType t = MLIR_GetOpType(ctx, bo);

        // ---- Region-bearing structured CF ops ----
        if (t == OP_TYPE_WASMSSA_BLOCK || t == OP_TYPE_WASMSSA_LOOP || t == OP_TYPE_WASMSSA_IF) {
            if (!stackify_region_op(ctx, sc, arena, bo, t, dst_blk, vmap, L, rs)) {
                return false;
            }
            continue;
        }
        if (t == OP_TYPE_WASMSSA_BLOCK_RETURN) {
            // Copy yield operands into the enclosing region's result_locals.
            RegionFrame *top = rs->n ? &rs->data[rs->n - 1] : NULL;
            if (!top) { fprintf(stderr, "stackify: block_return with no enclosing region\n"); return false; }
            size_t no = MLIR_GetOpNumOperands(ctx, bo);
            if (no != top->n_results) {
                fprintf(stderr, "stackify: block_return arity %zu != region results %zu\n", no, top->n_results);
                return false;
            }
            for (size_t k = 0; k < no; k++) {
                uint32_t li;
                if (!vmap_get(vmap, MLIR_GetOpOperand(ctx, bo, k), &li)) {
                    fprintf(stderr, "stackify: unbound block_return operand %zu\n", k); return false;
                }
                emit_local_get(ctx, sc, dst_blk, li);
                emit_local_set(ctx, sc, dst_blk, top->result_locals[k]);
            }
            continue;
        }
        if (t == OP_TYPE_WASMSSA_UNREACHABLE) {
            MLIR_NamedAttr as[1];
            as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_UNREACHABLE, as, 1));
            continue;
        }

        // ---- wasmssa.br with operands: transfer values into target locals ----
        if (t == OP_TYPE_WASMSSA_BR && MLIR_GetOpNumOperands(ctx, bo) > 0) {
            MLIR_AttributeHandle depth_attr = MLIR_GetOpAttr(ctx, bo, MLIR_KEY_DEPTH);
            uint32_t depth = depth_attr ? (uint32_t)MLIR_GetAttributeInteger(ctx, depth_attr) : 0;
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
            size_t no = MLIR_GetOpNumOperands(ctx, bo);
            if (no != n_dst) {
                fprintf(stderr, "stackify: br arity %zu != target slot count %zu\n", no, n_dst);
                return false;
            }
            for (size_t k = 0; k < no; k++) {
                uint32_t li;
                if (!vmap_get(vmap, MLIR_GetOpOperand(ctx, bo, k), &li)) {
                    fprintf(stderr, "stackify: unbound br operand %zu\n", k); return false;
                }
                emit_local_get(ctx, sc, dst_blk, li);
                emit_local_set(ctx, sc, dst_blk, dst_locals[k]);
            }
            MLIR_NamedAttr as[2];
            as[0] = MLIR_Named(MLIR_KEY_VALTYPE, sc->attr_i32_0);
            as[1] = MLIR_Named(MLIR_KEY_DEPTH, depth_attr ? depth_attr : sc->attr_i32_0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_BR, as, 2));
            continue;
        }

        // ---- Generic: push operands as local.get, emit op, stash result ----
        size_t n_ops = MLIR_GetOpNumOperands(ctx, bo);
        for (size_t k = 0; k < n_ops; k++) {
            uint32_t li;
            MLIR_ValueHandle ov = MLIR_GetOpOperand(ctx, bo, k);
            if (!vmap_get(vmap, ov, &li)) {
                fprintf(stderr, "stackify: unbound operand on op (idx %zu)\n", k);
                return false;
            }
            emit_local_get(ctx, sc, dst_blk, li);
        }

        MLIR_OpType st = ssa_to_stack(t);
        if (!st) {
            fprintf(stderr, "stackify: unknown wasmssa op enum=%d\n", (int)t);
            return false;
        }
        MLIR_AttributeHandle vt_attr = MLIR_GetOpAttr(ctx, bo, MLIR_KEY_VALTYPE);
        emit_stack_simple(ctx, sc, arena, dst_blk, bo, t, st, vt_attr);

        if (MLIR_GetOpNumResults(ctx, bo) > 0) {
            uint8_t valtype = vt_attr ? (uint8_t)MLIR_GetAttributeInteger(ctx, vt_attr) : 0;
            uint32_t li = locals_add(L, valtype);
            emit_local_set(ctx, sc, dst_blk, li);
            vmap_set(vmap, MLIR_GetOpResult(ctx, bo, 0), li);
        }
    }
    return true;
}

static bool stackify_body(MLIR_Context *ctx, StackCache *sc, Arena *arena,
                          MLIR_OpHandle src_func, MLIR_BlockHandle dst_blk,
                          const uint8_t *param_types, size_t n_params,
                          Locals *L) {
    (void)param_types;
    L->n = 0;
    L->n_params = n_params;

    if (MLIR_GetOpNumRegions(ctx, src_func) < 1) return true;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, src_func, 0);
    if (MLIR_GetRegionNumBlocks(ctx, r) < 1) return true;
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(ctx, r, 0);

    VMap *vmap = &sc->vmap;
    if (vmap->cap > 0) {
        memset(vmap->keys, 0, vmap->cap * sizeof(*vmap->keys));
        vmap->n = 0;
    }
    size_t n_args = MLIR_GetBlockNumArgs(ctx, src_blk);
    for (size_t i = 0; i < n_args; i++) {
        vmap_set(vmap, MLIR_GetBlockArg(ctx, src_blk, i), (uint32_t)i);
    }

    sc->rs.n = 0;
    return stackify_walk_block(ctx, sc, arena, src_blk, dst_blk, vmap, L, &sc->rs);
}

// =============================================================================
// Per-top-level-op converters.
// =============================================================================
static MLIR_OpHandle convert_func(MLIR_Context *ctx, StackCache *sc, Arena *arena, MLIR_OpHandle src) {
    string pt = at_s(ctx, src, MLIR_KEY_PARAM_TYPES);
    size_t n_params = pt.size / 2;

    MLIR_NamedAttr attrs[12];
    size_t na = 0;
    attrs[na++] = fwd_attr_s(ctx, src, MLIR_KEY_SYM_NAME);
    attrs[na++] = fwd_attr_s(ctx, src, MLIR_KEY_PARAM_TYPES);
    attrs[na++] = fwd_attr_s(ctx, src, MLIR_KEY_RESULT_TYPES);
    attrs[na++] = MLIR_Named(MLIR_KEY_EXPORTED, MLIR_GetOpAttr(ctx, src, MLIR_KEY_EXPORTED) ? MLIR_GetOpAttr(ctx, src, MLIR_KEY_EXPORTED) : MLIR_CreateAttributeBool(ctx, false));
    attrs[na++] = MLIR_Named(MLIR_KEY_INTERNAL, MLIR_GetOpAttr(ctx, src, MLIR_KEY_INTERNAL) ? MLIR_GetOpAttr(ctx, src, MLIR_KEY_INTERNAL) : MLIR_CreateAttributeBool(ctx, false));
    MLIR_AttributeHandle imod = MLIR_GetOpAttr(ctx, src, MLIR_KEY_IMPORT_MODULE);
    MLIR_AttributeHandle iname = MLIR_GetOpAttr(ctx, src, MLIR_KEY_IMPORT_NAME);
    MLIR_AttributeHandle ename = MLIR_GetOpAttr(ctx, src, MLIR_KEY_EXPORT_NAME);
    if (imod)  attrs[na++] = MLIR_Named(MLIR_KEY_IMPORT_MODULE, imod);
    if (iname) attrs[na++] = MLIR_Named(MLIR_KEY_IMPORT_NAME, iname);
    if (ename) attrs[na++] = MLIR_Named(MLIR_KEY_EXPORT_NAME, ename);

    bool is_import = MLIR_GetOpType(ctx, src) == OP_TYPE_WASMSSA_IMPORT_FUNC;
    if (is_import) {
        return build_op(ctx, OP_TYPE_WASMSTACK_IMPORT_FUNC, attrs, na);
    }

    MLIR_BlockHandle dst_blk = MLIR_CreateBlock(ctx);
    bool ok = stackify_body(ctx, sc, arena, src, dst_blk, NULL, n_params, &sc->locals);
    if (!ok) return MLIR_INVALID_HANDLE;

    attrs[na++] = attr_s_hex(ctx, arena, MLIR_KEY_LOCAL_TYPES, sc->locals.types, sc->locals.n);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, dst_blk);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpSpec spec = {0};
    spec.type = OP_TYPE_WASMSTACK_FUNC;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)na;
    spec.regions = regs;
    spec.n_regions = 1;
    return MLIR_CreateOpFromSpec(ctx, &spec);
}

static MLIR_OpHandle convert_global(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name    = at_s(ctx, src, MLIR_KEY_SYM_NAME);
    string idata   = at_s(ctx, src, MLIR_KEY_INIT_DATA);
    string relocs  = at_s(ctx, src, MLIR_KEY_RELOCS);
    MLIR_NamedAttr attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, MLIR_KEY_SYM_NAME,  name.str, name.size);
    attrs[na++] = attr_i32(ctx, MLIR_KEY_SIZE,      at_i(ctx, src, MLIR_KEY_SIZE));
    attrs[na++] = attr_i32(ctx, MLIR_KEY_ALIGN_POW, at_i(ctx, src, MLIR_KEY_ALIGN_POW));
    attrs[na++] = attr_b(ctx, MLIR_KEY_IS_CONST,    at_b(ctx, src, MLIR_KEY_IS_CONST));
    attrs[na++] = attr_s(ctx, MLIR_KEY_INIT_DATA, idata.str, idata.size);
    if (relocs.size) attrs[na++] = attr_s(ctx, MLIR_KEY_RELOCS, relocs.str, relocs.size);
    return build_op(ctx, OP_TYPE_WASMSTACK_IMPORT_GLOBAL, attrs, na);
}

// =============================================================================
// Top-level: walk input module body, build output module body.
// =============================================================================
MLIR_OpHandle mlir_wasmssa_to_wasmstack(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    if (!ssa_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(ctx, ssa_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, ssa_module, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);

    Arena *arena = MLIR_GetArenaAllocator(ctx);

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };
    MLIR_OpSpec mod_spec = {0};
    mod_spec.type = OP_TYPE_MODULE;
    mod_spec.regions = out_regs;
    mod_spec.n_regions = 1;
    MLIR_OpHandle out_module = MLIR_CreateOpFromSpec(ctx, &mod_spec);

    StackCache sc;
    sc_init(ctx, &sc);

    size_t n_top = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(ctx, mb, i);
        MLIR_OpType t = MLIR_GetOpType(ctx, top);
        MLIR_OpHandle out_op = MLIR_INVALID_HANDLE;
        if (t == OP_TYPE_WASMSSA_IMPORT_FUNC || t == OP_TYPE_WASMSSA_FUNC) {
            out_op = convert_func(ctx, &sc, arena, top);
            if (!out_op) { sc_destroy(&sc); return MLIR_INVALID_HANDLE; }
        } else if (t == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            out_op = convert_global(ctx, top);
        } else {
            string nm = MLIR_GetOpTypeName(ctx, t);
            fprintf(stderr, "stackify: unexpected top-level op '%.*s'\n",
                    (int)nm.size, nm.str);
            sc_destroy(&sc);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_AppendBlockOp(ctx, out_body, out_op);
    }
    sc_destroy(&sc);
    return out_module;
}

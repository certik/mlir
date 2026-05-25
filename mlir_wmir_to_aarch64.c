// wmir -> aarch64 lowering. See mlir_wmir_to_aarch64.h for the public
// API and rationale.
//
// Strategy (still pre-register-allocation, expanded for multi-block
// control flow):
//   * Pre-pass: walk every block of each wmir.func body. Number every
//     block argument and every op result with a unique slot index
//     (0, 1, 2, …). Each slot is 8 bytes (over-aligned even for i32).
//   * Function prologue:
//         stp x29, x30, [sp, #-16]!
//         mov x29, sp
//         sub sp, sp, #frame_size
//         str w0,  [sp, #slot(param0)*8]    ; spill parameters
//   * Each non-branch op materialises operands into scratch registers
//     (w9, w10, w11), computes the result into w9, then stores it back.
//   * Branches (wmir.br / wmir.cond_br) preserve the wmir block
//     structure: each wmir block becomes an aarch64 block, and the
//     aarch64 branch op carries its successor handle so the Mach-O
//     backend can resolve the PC-relative displacement once block
//     offsets are known.
//   * `wmir.br ^bb(%v1, %v2)` lowers to a sequence of `ldr/str` pairs
//     that copy each operand into the target block's argument slot,
//     followed by an unconditional `aarch64.b`.
//   * `wmir.cond_br %c, ^t, ^f` (no successor args; the wasmssa->wmir
//     pass arranges that cond_br targets never receive operands)
//     lowers to:
//         ldr w9, [sp, #cond_slot]
//         cbnz w9, ^t
//         b    ^f
//
// Module-level metadata (n_globals, linmem_size, …) is attached as
// attributes on the aarch64 builtin.module so the downstream Mach-O
// backend can size the __DATA segment without re-parsing wmir.

#include "mlir_wmir_to_aarch64.h"

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

// =============================================================================
// Module-wide constants (must match the wasm-link conventions so the
// new pipeline is observationally identical to the existing one).
// =============================================================================
enum {
    DEFAULT_STACK_SIZE        = 4u * 1024u * 1024u,
    DEFAULT_GLOBAL_BASE_OFFS  = 1024u,
    WASM_PAGE_SIZE            = 65536u,
};

// =============================================================================
// Attribute helpers.
// =============================================================================
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
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

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Build a 0-result op with one block successor (no successor operands).
static MLIR_OpHandle build_branch_op(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_BlockHandle target) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *succ_ops[1] = { NULL };
    size_t           n_succ_ops[1] = { 0 };
    return MLIR_CreateOpWithSuccessors(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0,
        NULL, 0, NULL, 0,
        succs, 1, succ_ops, n_succ_ops,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Emit helpers — one per aarch64.* op kind.
// =============================================================================
static void emit_movz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVZ, a, 4));
}
static void emit_movk(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "imm16", imm16);
    a[2] = attr_i32(ctx, "hw", hw);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOVK, a, 4));
}
static void emit_bl(MLIR_Context *ctx, MLIR_BlockHandle blk, string callee) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_s(ctx, "callee", callee.str, callee.size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BL, a, 1));
}
static void emit_svc(MLIR_Context *ctx, MLIR_BlockHandle blk, uint16_t imm16) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "imm16", imm16);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SVC, a, 1));
}
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_RET, NULL, 0));
}
static void emit_brk(MLIR_Context *ctx, MLIR_BlockHandle blk, uint16_t imm16) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "imm16", imm16);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BRK, a, 1));
}
static void emit_add_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_REG, a, 4));
}
static void emit_sub_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SUB_REG, a, 4));
}
static void emit_ldr_w(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_LDR_W, a, 3));
}
static void emit_str_w(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_STR_W, a, 3));
}
static void emit_adrp_data(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, const char *target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_s(ctx, "target", target, strlen(target));
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADRP_DATA, a, 2));
}
static void emit_add_data_lo(MLIR_Context *ctx, MLIR_BlockHandle blk,
                             uint8_t rd, uint8_t rn, const char *target) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_s(ctx, "target", target, strlen(target));
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_DATA_LO, a, 3));
}
static void emit_prologue(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint16_t frame_size) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", frame_size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_PROLOGUE, a, 1));
}
static void emit_epilogue(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint16_t frame_size) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", frame_size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_EPILOGUE, a, 1));
}
static void emit_cmp_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rn", rn);
    a[1] = attr_i32(ctx, "rm", rm);
    a[2] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CMP_REG, a, 3));
}
static void emit_cmp_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rn, uint16_t imm12, bool sf) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rn", rn);
    a[1] = attr_i32(ctx, "imm12", imm12);
    a[2] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CMP_IMM, a, 3));
}
static void emit_cset(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t cond, bool sf) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "cond", cond);
    a[2] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CSET, a, 3));
}
static void emit_csel(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn, uint8_t rm,
                      uint8_t cond, bool sf) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_i32(ctx, "cond", cond);
    a[4] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CSEL, a, 5));
}
static void emit_b(MLIR_Context *ctx, MLIR_BlockHandle blk,
                   MLIR_BlockHandle target) {
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_B, NULL, 0, target));
}
static void emit_cbz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                     uint8_t rt, bool sf, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_CBZ, a, 2, target));
}
static void emit_cbnz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rt, bool sf, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_CBNZ, a, 2, target));
}

// Materialise a 32-bit immediate into Wn.
static void emit_mov_imm32(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, uint32_t v) {
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffff), 0, /*sf=*/false);
    if ((v >> 16) != 0) {
        emit_movk(ctx, blk, rd, (uint16_t)((v >> 16) & 0xffff), 1, /*sf=*/false);
    }
}

// =============================================================================
// Slot map: SSA value -> stack-slot index.
// =============================================================================
typedef struct {
    MLIR_ValueHandle *vs;
    uint16_t         *slot;
    size_t            size;
    size_t            cap;
} SlotMap;

static void sm_set(SlotMap *m, MLIR_ValueHandle v, uint16_t s) {
    if (m->size == m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 16;
        m->vs   = realloc(m->vs,   nc * sizeof(MLIR_ValueHandle));
        m->slot = realloc(m->slot, nc * sizeof(uint16_t));
        m->cap  = nc;
    }
    m->vs[m->size]   = v;
    m->slot[m->size] = s;
    m->size++;
}
static int sm_get(const SlotMap *m, MLIR_ValueHandle v, uint16_t *out) {
    for (size_t i = 0; i < m->size; i++) {
        if (m->vs[i] == v) { *out = m->slot[i]; return 1; }
    }
    return 0;
}
static void sm_free(SlotMap *m) { free(m->vs); free(m->slot); memset(m, 0, sizeof(*m)); }

// =============================================================================
// Block map: source wmir block -> destination aarch64 block.
// =============================================================================
typedef struct {
    MLIR_BlockHandle *src;
    MLIR_BlockHandle *dst;
    size_t            n, cap;
} BlockMap;
static void bm_set(BlockMap *m, MLIR_BlockHandle k, MLIR_BlockHandle v) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->src = realloc(m->src, m->cap * sizeof(*m->src));
        m->dst = realloc(m->dst, m->cap * sizeof(*m->dst));
    }
    m->src[m->n] = k; m->dst[m->n] = v; m->n++;
}
static MLIR_BlockHandle bm_get(const BlockMap *m, MLIR_BlockHandle k) {
    for (size_t i = 0; i < m->n; i++) {
        if (m->src[i] == k) return m->dst[i];
    }
    return MLIR_INVALID_HANDLE;
}
static void bm_free(BlockMap *m) { free(m->src); free(m->dst); memset(m, 0, sizeof(*m)); }

// =============================================================================
// AArch64 condition codes (subset).
// =============================================================================
enum {
    COND_EQ = 0, COND_NE = 1, COND_CS = 2, COND_CC = 3,
    COND_HI = 8, COND_LS = 9, COND_GE = 10, COND_LT = 11,
    COND_GT = 12, COND_LE = 13,
};

static uint8_t cond_for_pred(string pred) {
    if (pred.size == 2 && memcmp(pred.str, "eq", 2) == 0) return COND_EQ;
    if (pred.size == 2 && memcmp(pred.str, "ne", 2) == 0) return COND_NE;
    if (pred.size == 3 && memcmp(pred.str, "slt", 3) == 0) return COND_LT;
    if (pred.size == 3 && memcmp(pred.str, "sgt", 3) == 0) return COND_GT;
    if (pred.size == 3 && memcmp(pred.str, "sle", 3) == 0) return COND_LE;
    if (pred.size == 3 && memcmp(pred.str, "sge", 3) == 0) return COND_GE;
    if (pred.size == 3 && memcmp(pred.str, "ult", 3) == 0) return COND_CC;
    if (pred.size == 3 && memcmp(pred.str, "ugt", 3) == 0) return COND_HI;
    if (pred.size == 3 && memcmp(pred.str, "ule", 3) == 0) return COND_LS;
    if (pred.size == 3 && memcmp(pred.str, "uge", 3) == 0) return COND_CS;
    return COND_EQ;
}

// =============================================================================
// Per-function lowering.
// =============================================================================
static MLIR_OpHandle lower_func(MLIR_Context *ctx, MLIR_OpHandle src) {
    string name     = at_s(src, "sym_name");
    bool   exported = at_b(src, "exported");
    string pt       = at_s(src, "param_types");

    if (MLIR_GetOpNumRegions(src) < 1) {
        fprintf(stderr, "wmir->aarch64: wmir.func has no region\n");
        return MLIR_INVALID_HANDLE;
    }
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(src, 0);
    size_t n_blocks = MLIR_GetRegionNumBlocks(src_region);
    if (n_blocks < 1) {
        fprintf(stderr, "wmir->aarch64: wmir.func has no entry block\n");
        return MLIR_INVALID_HANDLE;
    }

    // ---------- Pre-pass: assign a slot to every block arg + op result.
    SlotMap sm = {0};
    uint16_t next_slot = 0;
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle b = MLIR_GetRegionBlock(src_region, bi);
        size_t na = MLIR_GetBlockNumArgs(b);
        for (size_t i = 0; i < na; i++) {
            sm_set(&sm, MLIR_GetBlockArg(b, i), next_slot++);
        }
        size_t no = MLIR_GetBlockNumOps(b);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(b, i);
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t k = 0; k < nr; k++) {
                sm_set(&sm, MLIR_GetOpResult(op, k), next_slot++);
            }
        }
    }
    uint32_t frame_size = (uint32_t)next_slot * 8u;
    frame_size = (frame_size + 15u) & ~15u;
    if (frame_size > 0xfff) {
        fprintf(stderr,
            "wmir->aarch64: function '%.*s' frame size %u exceeds imm12 budget\n",
            (int)name.size, name.str, frame_size);
        sm_free(&sm);
        return MLIR_INVALID_HANDLE;
    }

    // ---------- Create destination region + blocks (one per src block).
    MLIR_RegionHandle dst_region = MLIR_CreateRegion(ctx);
    BlockMap bm = {0};
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle s = MLIR_GetRegionBlock(src_region, bi);
        MLIR_BlockHandle d = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, dst_region, d);
        bm_set(&bm, s, d);
    }
    MLIR_BlockHandle entry_dst = bm_get(&bm, MLIR_GetRegionBlock(src_region, 0));

    // ---------- Entry prologue + parameter spill.
    emit_prologue(ctx, entry_dst, (uint16_t)frame_size);
    {
        MLIR_BlockHandle src_entry = MLIR_GetRegionBlock(src_region, 0);
        size_t n_params = MLIR_GetBlockNumArgs(src_entry);
        for (size_t i = 0; i < n_params; i++) {
            uint16_t s; sm_get(&sm, MLIR_GetBlockArg(src_entry, i), &s);
            emit_str_w(ctx, entry_dst, (uint8_t)i, 31, (uint16_t)(s * 8u));
        }
    }

    // ---------- Walk each block and lower its ops.
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, bi);
        MLIR_BlockHandle dst_blk = bm_get(&bm, src_blk);

        #define LD_OPERAND(REG, IDX)                                       \
            do {                                                           \
                MLIR_ValueHandle _v = MLIR_GetOpOperand(op, (IDX));        \
                uint16_t _s;                                               \
                if (!sm_get(&sm, _v, &_s)) {                               \
                    fprintf(stderr,                                        \
                        "wmir->aarch64: operand %zu unbound\n",            \
                        (size_t)(IDX));                                    \
                    sm_free(&sm); bm_free(&bm);                            \
                    return MLIR_INVALID_HANDLE;                            \
                }                                                          \
                emit_ldr_w(ctx, dst_blk, (REG), 31, (uint16_t)(_s * 8u));  \
            } while (0)
        #define ST_RESULT(REG, IDX)                                        \
            do {                                                           \
                MLIR_ValueHandle _v = MLIR_GetOpResult(op, (IDX));         \
                uint16_t _s;                                               \
                sm_get(&sm, _v, &_s);                                      \
                emit_str_w(ctx, dst_blk, (REG), 31, (uint16_t)(_s * 8u));  \
            } while (0)

        size_t n_ops = MLIR_GetBlockNumOps(src_blk);
        for (size_t i = 0; i < n_ops; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(src_blk, i);
            MLIR_OpType  t  = MLIR_GetOpType(op);
            switch (t) {
            case OP_TYPE_WMIR_CONST: {
                MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
                MLIR_TypeHandle  ty = MLIR_GetValueType(r);
                string ts = MLIR_GetTypeString(ctx, ty);
                if (!(ts.size == 3 && memcmp(ts.str, "i32", 3) == 0)) {
                    if (ts.size == 3 && memcmp(ts.str, "i64", 3) == 0) continue;
                    fprintf(stderr,
                        "wmir->aarch64: wmir.const of unsupported type '%.*s'\n",
                        (int)ts.size, ts.str);
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                int64_t v = at_i(op, "value");
                emit_mov_imm32(ctx, dst_blk, 9, (uint32_t)v);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_IADD: {
                LD_OPERAND(9, 0); LD_OPERAND(10, 1);
                emit_add_reg(ctx, dst_blk, 9, 9, 10, /*sf=*/false);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_ISUB: {
                LD_OPERAND(9, 0); LD_OPERAND(10, 1);
                emit_sub_reg(ctx, dst_blk, 9, 9, 10, /*sf=*/false);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_ICMP: {
                // cmp Wn, Wm; cset Wd, <pred>.
                LD_OPERAND(9, 0); LD_OPERAND(10, 1);
                emit_cmp_reg(ctx, dst_blk, 9, 10, /*sf=*/false);
                string pred = at_s(op, "pred");
                uint8_t cond = cond_for_pred(pred);
                emit_cset(ctx, dst_blk, 9, cond, /*sf=*/false);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_EQZ: {
                LD_OPERAND(9, 0);
                emit_cmp_imm(ctx, dst_blk, 9, 0, /*sf=*/false);
                emit_cset(ctx, dst_blk, 9, COND_EQ, /*sf=*/false);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_SELECT: {
                // result = cond != 0 ? a : b.
                // ldr w9,a / ldr w10,b / ldr w11,cond; cmp w11,#0;
                // csel w9, w9, w10, NE.
                LD_OPERAND(9, 0);  // a
                LD_OPERAND(10, 1); // b
                LD_OPERAND(11, 2); // cond
                emit_cmp_imm(ctx, dst_blk, 11, 0, /*sf=*/false);
                emit_csel(ctx, dst_blk, 9, 9, 10, COND_NE, /*sf=*/false);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_GLOBAL_GET: {
                int64_t gi = at_i(op, "global_idx");
                emit_ldr_w(ctx, dst_blk, 9, 27, (uint16_t)(gi * 8));
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_GLOBAL_SET: {
                int64_t gi = at_i(op, "global_idx");
                LD_OPERAND(9, 0);
                emit_str_w(ctx, dst_blk, 9, 27, (uint16_t)(gi * 8));
                break;
            }
            case OP_TYPE_WMIR_LOAD: {
                int64_t off = at_i(op, "memory_offset");
                LD_OPERAND(9, 0);
                emit_add_reg(ctx, dst_blk, 10, 28, 9, /*sf=*/true);
                emit_ldr_w(ctx, dst_blk, 9, 10, (uint16_t)off);
                ST_RESULT(9, 0);
                break;
            }
            case OP_TYPE_WMIR_STORE: {
                int64_t off = at_i(op, "memory_offset");
                LD_OPERAND(9, 0); LD_OPERAND(11, 1);
                emit_add_reg(ctx, dst_blk, 10, 28, 9, /*sf=*/true);
                emit_str_w(ctx, dst_blk, 11, 10, (uint16_t)off);
                break;
            }
            case OP_TYPE_WMIR_CALL: {
                size_t na = MLIR_GetOpNumOperands(op);
                if (na > 8) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.call with %zu args (>8) not "
                        "yet supported\n", na);
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                for (size_t k = 0; k < na; k++) LD_OPERAND((uint8_t)k, k);
                string callee = at_s(op, "target");
                if (callee.size == 0) callee = at_s(op, "callee");
                emit_bl(ctx, dst_blk, callee);
                if (MLIR_GetOpNumResults(op) > 0) ST_RESULT(0, 0);
                break;
            }
            case OP_TYPE_WMIR_RETURN: {
                size_t no = MLIR_GetOpNumOperands(op);
                if (no > 1) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.return with %zu results not yet "
                        "supported\n", no);
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                if (no == 1) LD_OPERAND(0, 0);
                emit_epilogue(ctx, dst_blk, (uint16_t)frame_size);
                emit_ret(ctx, dst_blk);
                break;
            }
            case OP_TYPE_WMIR_UNREACHABLE: {
                emit_brk(ctx, dst_blk, 1);
                break;
            }
            case OP_TYPE_WMIR_BR: {
                // Copy each successor operand into the corresponding
                // target block-arg slot, then branch. Successor operands
                // are NOT regular operands — they are attached to the
                // successor edge and queried separately.
                MLIR_BlockHandle s_target = MLIR_GetOpSuccessor(op, 0);
                MLIR_BlockHandle d_target = bm_get(&bm, s_target);
                if (d_target == MLIR_INVALID_HANDLE) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.br target block not mapped\n");
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                size_t n_sops = MLIR_GetOpNumSuccessorOperands(op, 0);
                size_t n_args = MLIR_GetBlockNumArgs(s_target);
                if (n_sops != n_args) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.br operand count %zu != target "
                        "arg count %zu\n", n_sops, n_args);
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                for (size_t k = 0; k < n_sops; k++) {
                    MLIR_ValueHandle src_v = MLIR_GetOpSuccessorOperand(op, 0, k);
                    uint16_t src_s; sm_get(&sm, src_v, &src_s);
                    MLIR_ValueHandle tgt_a = MLIR_GetBlockArg(s_target, k);
                    uint16_t tgt_s; sm_get(&sm, tgt_a, &tgt_s);
                    if (src_s == tgt_s) continue;
                    emit_ldr_w(ctx, dst_blk, 9, 31, (uint16_t)(src_s * 8u));
                    emit_str_w(ctx, dst_blk, 9, 31, (uint16_t)(tgt_s * 8u));
                }
                emit_b(ctx, dst_blk, d_target);
                break;
            }
            case OP_TYPE_WMIR_COND_BR: {
                // Two successors, no successor operands by construction.
                if (MLIR_GetOpNumSuccessors(op) < 2) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.cond_br needs 2 successors\n");
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                MLIR_BlockHandle t_src = MLIR_GetOpSuccessor(op, 0);
                MLIR_BlockHandle f_src = MLIR_GetOpSuccessor(op, 1);
                MLIR_BlockHandle t_dst = bm_get(&bm, t_src);
                MLIR_BlockHandle f_dst = bm_get(&bm, f_src);
                if (t_dst == MLIR_INVALID_HANDLE || f_dst == MLIR_INVALID_HANDLE) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.cond_br target block not mapped\n");
                    sm_free(&sm); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                // Load cond into w9; test against zero. Wasm boolean
                // convention: 0 = false, non-zero = true.
                LD_OPERAND(9, 0);
                emit_cbnz(ctx, dst_blk, 9, /*sf=*/false, t_dst);
                emit_b(ctx, dst_blk, f_dst);
                break;
            }
            default: {
                string nm = MLIR_GetOpName(op);
                fprintf(stderr,
                    "wmir->aarch64: unsupported wmir op '%.*s' (kind=%d)\n",
                    (int)nm.size, nm.str, (int)t);
                sm_free(&sm); bm_free(&bm);
                return MLIR_INVALID_HANDLE;
            }
            }
        }

        #undef LD_OPERAND
        #undef ST_RESULT
    }

    sm_free(&sm);
    bm_free(&bm);

    MLIR_AttributeHandle attrs[4];
    size_t naf = 0;
    attrs[naf++] = attr_s(ctx, "sym_name", name.str, name.size);
    attrs[naf++] = attr_b(ctx, "exported", exported);
    attrs[naf++] = attr_s(ctx, "param_types", pt.str, pt.size);

    MLIR_RegionHandle regs[1] = { dst_region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, naf, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// _start synthesis. Sets up x26/x27/x28 then calls main, then svc-exits.
// =============================================================================
static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name,
                                 bool use_data_priv, bool use_globals,
                                 bool use_linmem) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);

    if (use_data_priv) {
        emit_adrp_data(ctx, blk, 26, "data_priv");
        emit_add_data_lo(ctx, blk, 26, 26, "data_priv");
    }
    if (use_globals) {
        emit_adrp_data(ctx, blk, 27, "globals");
        emit_add_data_lo(ctx, blk, 27, 27, "globals");
    }
    if (use_linmem) {
        emit_adrp_data(ctx, blk, 28, "linmem");
        emit_add_data_lo(ctx, blk, 28, 28, "linmem");
    }

    emit_bl(ctx, blk, main_name);
    emit_movz(ctx, blk, /*rd=*/16, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_svc(ctx, blk, 0x80);

    MLIR_AttributeHandle attrs[2];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", "_start", 6);
    attrs[na++] = attr_b(ctx, "exported", true);

    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Walk the module to compute the highest global_idx referenced and
// whether linmem is touched.
// =============================================================================
typedef struct {
    int32_t n_globals;
    bool    uses_linmem;
} ModInfo;

static void scan_block(MLIR_BlockHandle blk, ModInfo *mi) {
    size_t n = MLIR_GetBlockNumOps(blk);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        MLIR_OpType t = MLIR_GetOpType(op);
        if (t == OP_TYPE_WMIR_GLOBAL_GET || t == OP_TYPE_WMIR_GLOBAL_SET) {
            int32_t gi = (int32_t)at_i(op, "global_idx");
            if (gi + 1 > mi->n_globals) mi->n_globals = gi + 1;
        } else if (t == OP_TYPE_WMIR_LOAD || t == OP_TYPE_WMIR_STORE) {
            mi->uses_linmem = true;
        }
        size_t nr = MLIR_GetOpNumRegions(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_RegionHandle rh = MLIR_GetOpRegion(op, r);
            size_t nb = MLIR_GetRegionNumBlocks(rh);
            for (size_t b = 0; b < nb; b++)
                scan_block(MLIR_GetRegionBlock(rh, b), mi);
        }
    }
}

// =============================================================================
// Top-level pass.
// =============================================================================
MLIR_OpHandle mlir_wmir_to_aarch64(MLIR_Context *ctx, MLIR_OpHandle wmir_module) {
    if (!wmir_module) return MLIR_INVALID_HANDLE;
    if (MLIR_GetOpNumRegions(wmir_module) < 1) return MLIR_INVALID_HANDLE;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(wmir_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return MLIR_INVALID_HANDLE;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);

    ModInfo mi = {0};
    scan_block(mb, &mi);

    uint32_t n_globals = (uint32_t)mi.n_globals;
    uint64_t global0_init = DEFAULT_STACK_SIZE;
    uint64_t lm_bytes = (uint64_t)DEFAULT_STACK_SIZE + DEFAULT_GLOBAL_BASE_OFFS;
    uint64_t lm_pages = (lm_bytes + WASM_PAGE_SIZE - 1) / WASM_PAGE_SIZE;
    uint64_t linmem_size = lm_pages * WASM_PAGE_SIZE;

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };

    MLIR_AttributeHandle mattrs[4];
    size_t nma = 0;
    mattrs[nma++] = attr_i32(ctx, "n_globals",     (int64_t)n_globals);
    mattrs[nma++] = attr_i64(ctx, "global0_init",  (int64_t)global0_init);
    mattrs[nma++] = attr_i64(ctx, "linmem_size",   (int64_t)linmem_size);
    mattrs[nma++] = attr_i32(ctx, "linmem_pages",  (int64_t)lm_pages);

    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        mattrs, nma, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    string entry_name = {0};

    size_t n_top = MLIR_GetBlockNumOps(mb);
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
        MLIR_OpType t = MLIR_GetOpType(top);
        if (t != OP_TYPE_WMIR_FUNC) {
            string nm = MLIR_GetOpName(top);
            fprintf(stderr,
                "wmir->aarch64: unexpected top-level op '%.*s'\n",
                (int)nm.size, nm.str);
            return MLIR_INVALID_HANDLE;
        }
        MLIR_OpHandle out_op = lower_func(ctx, top);
        if (!out_op) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, out_op);

        if (at_b(top, "exported") && entry_name.size == 0) {
            entry_name = at_s(top, "sym_name");
        }
    }

    if (entry_name.size == 0) {
        fprintf(stderr,
            "wmir->aarch64: module has no exported function to use as entry\n");
        return MLIR_INVALID_HANDLE;
    }

    bool use_globals = n_globals > 0;
    bool use_linmem  = mi.uses_linmem;
    MLIR_OpHandle start = synth_start(ctx, entry_name,
        /*use_data_priv=*/false, use_globals, use_linmem);
    if (!start) return MLIR_INVALID_HANDLE;
    MLIR_AppendBlockOp(ctx, out_body, start);

    return out_module;
}

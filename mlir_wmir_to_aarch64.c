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
#include "mlir_wmir_regalloc.h"

// =============================================================================
// Module-wide constants (must match the wasm-link conventions so the
// new pipeline is observationally identical to the existing one).
// =============================================================================
enum {
    DEFAULT_STACK_SIZE        = 4u * 1024u * 1024u,
    DEFAULT_GLOBAL_BASE_OFFS  = 1024u,
    WASM_PAGE_SIZE            = 65536u,
    // Maximum linmem size, in wasm pages (64 KiB each). We pre-reserve
    // this much virtual address space as a zero-fill BSS section in the
    // Mach-O envelope; macOS lazily commits pages as they are touched,
    // so the on-disk binary size is unaffected. The wasm-side
    // `memory.size` / `memory.grow` then bump a counter against this
    // cap, mirroring real WASI semantics. 4096 pages = 256 MiB; small
    // enough that dyld is happy with the __DATA segment vmsize, large
    // enough for the entire tinyC test suite. Selfhost (which needs
    // much more) bumps this via the upstream wasm-backend's path or
    // via a per-module override (TODO).
    MAX_LINMEM_PAGES          = 24576u,
    // Slot offsets within linmem used by synthesised libc shims.
    // See the full linmem-layout doc later in the file. Hoisted here
    // because lowering of `memory.size` / `memory.grow` (which uses
    // MEM_PAGES_SLOT) appears earlier than the doc.
    MALLOC_HEAP_OFF_SLOT      = 16,
    MEM_PAGES_SLOT            = 24,
    ARGC_SLOT                 = 40,
    ARGV_SLOT                 = 48,
    MALLOC_HEAP_BASE_BYTES    = 1u * 1024u * 1024u,
};

// Set by mlir_wmir_to_aarch64() once per module; read by lower_func() when
// expanding wmir.memory_size into a movz of the static page count.
static uint32_t g_linmem_pages = 0;

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

// Returns true if `v` has type `i64`. Used to pick X-form vs W-form
// instructions for binops and loads/stores into local slots.
static bool is_i64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle ty = MLIR_GetValueType(v);
    string s = MLIR_GetTypeString(ctx, ty);
    return s.size == 3 && memcmp(s.str, "i64", 3) == 0;
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
static void emit_mov_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOV_X, a, 2));
}
static void emit_add_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "imm12", imm12);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_IMM, a, 4));
}
static void emit_sub_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "imm12", imm12);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SUB_IMM, a, 4));
}
static void emit_b_cond(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        uint8_t cond, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "cond", cond);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_B_COND, a, 1, target));
}
static void emit_strb_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_STRB_IMM, a, 3));
}
static void emit_ldrb_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_LDRB_IMM, a, 3));
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
// Choose an addressing form that fits a 12-bit scaled-imm offset.
//
//   `scale` is 4 for W loads/stores and 8 for X loads/stores. The
//   final hardware encoding holds `off_bytes / scale` in 12 bits, so
//   the largest representable byte offset is `4095 * scale`.
//
// For larger offsets we materialise the byte offset into x16, add it
// to the base register (returning the sum in x16), and return
// `*p_rn = 16`, `*p_off_bytes = 0`. The caller then emits the load
// or store using `[x16, #0]`. x16 (IP0) is reserved by the linker
// for stubs but is otherwise free between BL boundaries, which is
// always the case for these helpers (we never call between
// the prologue computation and the load/store).
static void rebase_for_imm12(MLIR_Context *ctx, MLIR_BlockHandle blk,
                             uint8_t *p_rn, uint32_t *p_off_bytes,
                             uint32_t scale) {
    uint32_t off = *p_off_bytes;
    if (off <= (uint32_t)(4095u * scale) && (off % scale) == 0) return;
    // The shifted-register form of ADD/SUB treats register encoding
    // 31 as XZR (the zero register), NOT as SP. So when the base is
    // SP and the offset exceeds imm12, we must materialise SP via
    // the immediate form first (where Rn=31 IS SP, an architectural
    // quirk of the ADD instruction family). Materialise the offset
    // in x17 (also a linker scratch reg, free between BL boundaries
    // like x16) so we can then combine `(mov x16, sp) + x17` into
    // x16 via a register ADD where both inputs are real registers.
    if (*p_rn == 31) {
        emit_movz(ctx, blk, 17, (uint16_t)(off & 0xffffu), 0, /*sf=*/true);
        if ((off >> 16) != 0) {
            emit_movk(ctx, blk, 17, (uint16_t)((off >> 16) & 0xffffu), 1, /*sf=*/true);
        }
        emit_add_imm(ctx, blk, /*rd=*/16, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
        emit_add_reg(ctx, blk, /*rd=*/16, /*rn=*/16, /*rm=*/17, /*sf=*/true);
    } else {
        // movz x16, low; movk x16, high (if needed); add x16, rn, x16
        emit_movz(ctx, blk, 16, (uint16_t)(off & 0xffffu), 0, /*sf=*/true);
        if ((off >> 16) != 0) {
            emit_movk(ctx, blk, 16, (uint16_t)((off >> 16) & 0xffffu), 1, /*sf=*/true);
        }
        emit_add_reg(ctx, blk, /*rd=*/16, /*rn=*/*p_rn, /*rm=*/16, /*sf=*/true);
    }
    *p_rn = 16;
    *p_off_bytes = 0;
}

static void emit_ldr_w(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off_bytes) {
    rebase_for_imm12(ctx, blk, &rn, &off_bytes, /*scale=*/4);
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", (int64_t)off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_LDR_W, a, 3));
}
static void emit_str_w(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off_bytes) {
    rebase_for_imm12(ctx, blk, &rn, &off_bytes, /*scale=*/4);
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", (int64_t)off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_STR_W, a, 3));
}
static void emit_ldr_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off_bytes) {
    rebase_for_imm12(ctx, blk, &rn, &off_bytes, /*scale=*/8);
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", (int64_t)off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_LDR_X, a, 3));
}
static void emit_str_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off_bytes) {
    rebase_for_imm12(ctx, blk, &rn, &off_bytes, /*scale=*/8);
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", (int64_t)off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_STR_X, a, 3));
}
static void emit_3reg(MLIR_Context *ctx, MLIR_BlockHandle blk, MLIR_OpType t,
                      uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 4));
}
static void emit_msub(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, bool sf) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_i32(ctx, "ra", ra);
    a[4] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MSUB, a, 5));
}
static void emit_2reg_no_sf(MLIR_Context *ctx, MLIR_BlockHandle blk, MLIR_OpType t,
                            uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 2));
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
                          uint32_t frame_size) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)frame_size);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_PROLOGUE, a, 1));
}
static void emit_epilogue(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint32_t frame_size) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)frame_size);
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
// Materialise a 64-bit immediate into Xn (worst case 4 instructions).
static void emit_mov_imm64(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, uint64_t v) {
    bool emitted = false;
    for (uint8_t hw = 0; hw < 4; hw++) {
        uint16_t chunk = (uint16_t)((v >> (hw * 16)) & 0xffffu);
        if (chunk == 0 && emitted) continue;
        if (!emitted) {
            emit_movz(ctx, blk, rd, chunk, hw, /*sf=*/true);
            emitted = true;
        } else {
            emit_movk(ctx, blk, rd, chunk, hw, /*sf=*/true);
        }
    }
    if (!emitted) emit_movz(ctx, blk, rd, 0, 0, /*sf=*/true);
}

// =============================================================================
// Floating-point op emitters. All take *V register* numbers for rd/rn/rm
// where the slot is V, and *GP register* numbers (W/X) for GP slots.
// =============================================================================
// FMOV gp <-> V. dir_to_v=true: GP->V; false: V->GP. sf=false picks
// W<->S, sf=true picks X<->D.
static void emit_fmov_gp_v(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           bool dir_to_v, bool sf, uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_b  (ctx, "dir_to_v", dir_to_v);
    a[1] = attr_b  (ctx, "sf",       sf);
    a[2] = attr_i32(ctx, "rd",       rd);
    a[3] = attr_i32(ctx, "rn",       rn);
    MLIR_AppendBlockOp(ctx, blk,
        build_op(ctx, OP_TYPE_AARCH64_FMOV_GP_V, a, 4));
}
// FADD/FSUB/FMUL/FDIV (V regs). kind = "fadd"|"fsub"|"fmul"|"fdiv".
static void emit_fp_binop(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          const char *kind, int fwidth,
                          uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_s  (ctx, "kind",   kind, strlen(kind));
    a[1] = attr_i32(ctx, "fwidth", fwidth);
    a[2] = attr_i32(ctx, "rd",     rd);
    a[3] = attr_i32(ctx, "rn",     rn);
    a[4] = attr_i32(ctx, "rm",     rm);
    MLIR_AppendBlockOp(ctx, blk,
        build_op(ctx, OP_TYPE_AARCH64_FP_BINOP, a, 5));
}
// FNEG/FABS/FSQRT.
static void emit_fp_unop(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         const char *kind, int fwidth,
                         uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_s  (ctx, "kind",   kind, strlen(kind));
    a[1] = attr_i32(ctx, "fwidth", fwidth);
    a[2] = attr_i32(ctx, "rd",     rd);
    a[3] = attr_i32(ctx, "rn",     rn);
    MLIR_AppendBlockOp(ctx, blk,
        build_op(ctx, OP_TYPE_AARCH64_FP_UNOP, a, 4));
}
// FCMP Sn/Dn, Sm/Dm (sets NZCV, no result reg).
static void emit_fcmp(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      int fwidth, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "fwidth", fwidth);
    a[1] = attr_i32(ctx, "rn",     rn);
    a[2] = attr_i32(ctx, "rm",     rm);
    MLIR_AppendBlockOp(ctx, blk,
        build_op(ctx, OP_TYPE_AARCH64_FCMP, a, 3));
}
// FP conversion. kind ∈ {"f2f","f2i","i2f"}. sign matters only for
// f2i / i2f; for f2f it should be set to false.
static void emit_fp_cvt(MLIR_Context *ctx, MLIR_BlockHandle blk,
                        const char *kind, int src_w, int dst_w,
                        bool sign, uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[6];
    a[0] = attr_s  (ctx, "kind",  kind, strlen(kind));
    a[1] = attr_i32(ctx, "src_w", src_w);
    a[2] = attr_i32(ctx, "dst_w", dst_w);
    a[3] = attr_b  (ctx, "sign",  sign);
    a[4] = attr_i32(ctx, "rd",    rd);
    a[5] = attr_i32(ctx, "rn",    rn);
    MLIR_AppendBlockOp(ctx, blk,
        build_op(ctx, OP_TYPE_AARCH64_FP_CVT, a, 6));
}

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
    COND_MI = 4, COND_PL = 5, COND_VS = 6, COND_VC = 7,
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
// Register-allocator-aware operand/result accessors.
//
// `ra` is the per-function `WmirRegAlloc` produced by `wmir_regalloc_run`.
// Every SSA value defined in the function has a `ValueHome`: either a
// physical register (HOME_REG) or a stack slot (HOME_SLOT).
//
// `ld_operand_into(... idx, scratch)` ensures the operand `idx` of `op`
// is available in some register and returns that register. If the
// operand's home is a register, no instruction is emitted and that
// register is returned. Otherwise a `ldr` from the slot into `scratch`
// is emitted and `scratch` is returned.
//
// `pick_result_reg(ra, op, idx, scratch)` returns the register a
// computation should write its result into. If the result is assigned
// to a register, that register is returned; otherwise `scratch`.
//
// `st_result(... idx, produced_reg)` finalises a result. If the home
// is a register and matches `produced_reg`, no instruction is emitted.
// If the home is a register but differs, an `mov` is emitted. If the
// home is a slot, a `str` is emitted.
// =============================================================================
static uint8_t ld_operand_into(MLIR_Context *ctx, MLIR_BlockHandle blk,
                               const WmirRegAlloc *ra, MLIR_OpHandle op,
                               size_t idx, uint8_t scratch) {
    MLIR_ValueHandle v = MLIR_GetOpOperand(op, idx);
    ValueHome h;
    if (!wmir_regalloc_lookup(ra, v, &h)) {
        fprintf(stderr, "wmir->aarch64: unbound operand %zu\n", idx);
        return scratch;
    }
    if (h.kind == HOME_REG) return h.idx;
    if (is_i64(ctx, v))
        emit_ldr_x(ctx, blk, scratch, 31, (uint32_t)(h.idx * 8u));
    else
        emit_ldr_w(ctx, blk, scratch, 31, (uint32_t)(h.idx * 8u));
    return scratch;
}

static uint8_t pick_result_reg(const WmirRegAlloc *ra, MLIR_OpHandle op,
                               size_t idx, uint8_t scratch) {
    MLIR_ValueHandle v = MLIR_GetOpResult(op, idx);
    ValueHome h;
    if (!wmir_regalloc_lookup(ra, v, &h)) return scratch;
    if (h.kind == HOME_REG) return h.idx;
    return scratch;
}

static void st_result(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      const WmirRegAlloc *ra, MLIR_OpHandle op,
                      size_t idx, uint8_t produced_reg) {
    MLIR_ValueHandle v = MLIR_GetOpResult(op, idx);
    ValueHome h;
    if (!wmir_regalloc_lookup(ra, v, &h)) return;
    if (h.kind == HOME_REG) {
        if (h.idx != produced_reg) {
            // mov dst, src. We use the X-form (mov_x): for i32 the
            // upper bits are don't-care, and aarch64 mov_x is the
            // alias `orr xd, xzr, xn` — always a single instruction.
            emit_mov_x(ctx, blk, h.idx, produced_reg);
        }
        return;
    }
    if (is_i64(ctx, v))
        emit_str_x(ctx, blk, produced_reg, 31, (uint32_t)(h.idx * 8u));
    else
        emit_str_w(ctx, blk, produced_reg, 31, (uint32_t)(h.idx * 8u));
}

// Force `op`'s operand `idx` into a specific physical register `reg`.
// Used for call-arg passing (x0..x7) and return-value materialisation.
// Emits `mov` from a register home or `ldr` from a slot home as
// needed; no-op if already in `reg`.
static void ld_operand_into_fixed(MLIR_Context *ctx, MLIR_BlockHandle blk,
                                  const WmirRegAlloc *ra, MLIR_OpHandle op,
                                  size_t idx, uint8_t reg) {
    MLIR_ValueHandle v = MLIR_GetOpOperand(op, idx);
    ValueHome h;
    if (!wmir_regalloc_lookup(ra, v, &h)) {
        fprintf(stderr, "wmir->aarch64: unbound operand %zu\n", idx);
        return;
    }
    if (h.kind == HOME_REG) {
        if (h.idx != reg) emit_mov_x(ctx, blk, reg, h.idx);
        return;
    }
    if (is_i64(ctx, v))
        emit_ldr_x(ctx, blk, reg, 31, (uint32_t)(h.idx * 8u));
    else
        emit_ldr_w(ctx, blk, reg, 31, (uint32_t)(h.idx * 8u));
}

// Persist `reg` into the home of result `idx` of `op`. If the home is
// the same register, no-op. Used for call results (x0) materialisation.
static void st_result_from_fixed(MLIR_Context *ctx, MLIR_BlockHandle blk,
                                 const WmirRegAlloc *ra, MLIR_OpHandle op,
                                 size_t idx, uint8_t reg) {
    st_result(ctx, blk, ra, op, idx, reg);
}

// Emit the per-branch parallel-copy that moves successor operands
// into the target block's argument homes. Handles the four
// {reg,slot} × {reg,slot} combinations and resolves reg→reg cycles
// (e.g. swap) by routing the cycle break through scratch register x9.
#define MAX_PAIRS 16
typedef struct {
    ValueHome src;
    ValueHome dst;
    bool      is_i64;
    bool      done;
} BranchArgPair;
static void emit_branch_arg_copies(MLIR_Context *ctx, MLIR_BlockHandle blk,
                                   const WmirRegAlloc *ra,
                                   MLIR_OpHandle op, size_t succ_idx,
                                   MLIR_BlockHandle target_src_block) {
    size_t n = MLIR_GetOpNumSuccessorOperands(op, succ_idx);
    if (n == 0) return;
    if (n != MLIR_GetBlockNumArgs(target_src_block)) {
        // mismatched — caller already verifies/diagnoses this; bail.
        return;
    }
    BranchArgPair pairs[MAX_PAIRS];
    if (n > MAX_PAIRS) {
        // Fallback: route every pair through scratch r9. Same as the
        // pre-regalloc lowering. Safe for arbitrary fan-in.
        for (size_t k = 0; k < n; k++) {
            MLIR_ValueHandle src_v = MLIR_GetOpSuccessorOperand(op, succ_idx, k);
            MLIR_ValueHandle dst_v = MLIR_GetBlockArg(target_src_block, k);
            ValueHome sh, dh;
            wmir_regalloc_lookup(ra, src_v, &sh);
            wmir_regalloc_lookup(ra, dst_v, &dh);
            bool i64 = is_i64(ctx, dst_v);
            uint8_t r = 9;
            if (sh.kind == HOME_REG) {
                r = sh.idx;
            } else {
                if (i64) emit_ldr_x(ctx, blk, r, 31, (uint32_t)(sh.idx * 8u));
                else     emit_ldr_w(ctx, blk, r, 31, (uint32_t)(sh.idx * 8u));
            }
            if (dh.kind == HOME_REG) {
                if (dh.idx != r) emit_mov_x(ctx, blk, dh.idx, r);
            } else {
                if (i64) emit_str_x(ctx, blk, r, 31, (uint32_t)(dh.idx * 8u));
                else     emit_str_w(ctx, blk, r, 31, (uint32_t)(dh.idx * 8u));
            }
        }
        return;
    }
    for (size_t k = 0; k < n; k++) {
        MLIR_ValueHandle src_v = MLIR_GetOpSuccessorOperand(op, succ_idx, k);
        MLIR_ValueHandle dst_v = MLIR_GetBlockArg(target_src_block, k);
        wmir_regalloc_lookup(ra, src_v, &pairs[k].src);
        wmir_regalloc_lookup(ra, dst_v, &pairs[k].dst);
        pairs[k].is_i64 = is_i64(ctx, dst_v);
        pairs[k].done   = false;
    }

    // Phase 1: emit every pair whose destination is a slot. These
    // never participate in conflicts (slot writes can't clobber a
    // register source another pair is waiting on).
    for (size_t k = 0; k < n; k++) {
        if (pairs[k].dst.kind != HOME_SLOT) continue;
        uint8_t r = 9;
        if (pairs[k].src.kind == HOME_REG) {
            r = pairs[k].src.idx;
        } else {
            if (pairs[k].is_i64)
                emit_ldr_x(ctx, blk, r, 31, (uint32_t)(pairs[k].src.idx * 8u));
            else
                emit_ldr_w(ctx, blk, r, 31, (uint32_t)(pairs[k].src.idx * 8u));
        }
        if (pairs[k].is_i64)
            emit_str_x(ctx, blk, r, 31, (uint32_t)(pairs[k].dst.idx * 8u));
        else
            emit_str_w(ctx, blk, r, 31, (uint32_t)(pairs[k].dst.idx * 8u));
        pairs[k].done = true;
    }

    // Phase 2: emit reg-dest pairs in topological order. A pair is
    // safe to emit when no other un-done pair still reads its dst
    // register. When all remaining pairs participate in a cycle,
    // break the cycle by spilling the chosen src into scratch x9.
    for (;;) {
        bool any_remaining = false;
        for (size_t k = 0; k < n; k++) {
            if (!pairs[k].done) { any_remaining = true; break; }
        }
        if (!any_remaining) return;

        bool progress = false;
        for (size_t k = 0; k < n; k++) {
            if (pairs[k].done) continue;
            // pairs[k].dst is HOME_REG (phase 1 handled HOME_SLOT).
            uint8_t dst_reg = pairs[k].dst.idx;
            bool blocked = false;
            for (size_t j = 0; j < n; j++) {
                if (j == k || pairs[j].done) continue;
                if (pairs[j].src.kind == HOME_REG && pairs[j].src.idx == dst_reg) {
                    blocked = true; break;
                }
            }
            if (blocked) continue;
            if (pairs[k].src.kind == HOME_SLOT) {
                if (pairs[k].is_i64)
                    emit_ldr_x(ctx, blk, dst_reg, 31, (uint32_t)(pairs[k].src.idx * 8u));
                else
                    emit_ldr_w(ctx, blk, dst_reg, 31, (uint32_t)(pairs[k].src.idx * 8u));
            } else {
                if (pairs[k].src.idx != dst_reg)
                    emit_mov_x(ctx, blk, dst_reg, pairs[k].src.idx);
            }
            pairs[k].done = true;
            progress = true;
        }
        if (progress) continue;
        // Cycle: pick any unfinished pair whose src is a reg, save
        // that src to scratch x9, rewrite all remaining pairs that
        // read that src to read x9 instead, then loop.
        int cyc = -1;
        for (size_t k = 0; k < n; k++) {
            if (!pairs[k].done && pairs[k].src.kind == HOME_REG) { cyc = (int)k; break; }
        }
        if (cyc < 0) return;  // shouldn't reach here
        uint8_t cyc_src = pairs[cyc].src.idx;
        emit_mov_x(ctx, blk, 9, cyc_src);
        for (size_t k = 0; k < n; k++) {
            if (!pairs[k].done && pairs[k].src.kind == HOME_REG && pairs[k].src.idx == cyc_src) {
                pairs[k].src.idx = 9;
            }
        }
    }
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

    // ---------- Pre-pass: register allocation. Each SSA value gets a
    // ---------- home (HOME_REG or HOME_SLOT) via linear-scan.
    WmirRegAlloc *ra = wmir_regalloc_run(ctx, src);
    if (!ra) {
        fprintf(stderr, "wmir->aarch64: register allocation failed for '%.*s'\n",
                (int)name.size, name.str);
        return MLIR_INVALID_HANDLE;
    }
    uint32_t frame_size = (uint32_t)ra->n_slots * 8u;
    frame_size = (frame_size + 15u) & ~15u;
    // SUB SP, SP, #imm12 [LSL #12] supports up to ~16 MiB. The LDR/STR
    // helpers above transparently rematerialise large offsets via x16,
    // so any frame that fits the prologue's reach is reachable from
    // any slot.
    if (frame_size > 0xfff000) {
        fprintf(stderr,
            "wmir->aarch64: function '%.*s' frame size %u exceeds 16 MiB budget\n",
            (int)name.size, name.str, frame_size);
        wmir_regalloc_free(ra);
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
    emit_prologue(ctx, entry_dst, frame_size);
    {
        MLIR_BlockHandle src_entry = MLIR_GetRegionBlock(src_region, 0);
        size_t n_params = MLIR_GetBlockNumArgs(src_entry);
        for (size_t i = 0; i < n_params; i++) {
            MLIR_ValueHandle pv = MLIR_GetBlockArg(src_entry, i);
            ValueHome h;
            if (!wmir_regalloc_lookup(ra, pv, &h)) continue;
            bool i64 = is_i64(ctx, pv);
            if (i < 8) {
                // Param i arrives in xI (i64) or wI (i32). Move it to
                // its home (mov if reg, str if slot). No-op if reg==i.
                if (h.kind == HOME_REG) {
                    if (h.idx != (uint8_t)i)
                        emit_mov_x(ctx, entry_dst, h.idx, (uint8_t)i);
                } else {
                    // i64 params must be spilled as full 8 bytes; w0..w7 alone
                    // would drop the upper half (caused ternary_i64 regression).
                    if (i64)
                        emit_str_x(ctx, entry_dst, (uint8_t)i, 31, (uint32_t)(h.idx * 8u));
                    else
                        emit_str_w(ctx, entry_dst, (uint8_t)i, 31, (uint32_t)(h.idx * 8u));
                }
            } else {
                // Stack-passed param: caller put it at [old_sp, #(i-8)*8],
                // which is at [fp, #16 + (i-8)*8] from our perspective
                // (fp -> saved x29, fp+8 -> saved x30, fp+16 -> first
                // stack arg). Bring it into a scratch reg first, then
                // store / move to its home.
                uint16_t fp_off = (uint16_t)(16u + (i - 8u) * 8u);
                uint8_t scratch = 9;
                if (h.kind == HOME_REG) scratch = h.idx;
                if (i64) emit_ldr_x(ctx, entry_dst, scratch, /*rn=*/29, fp_off);
                else     emit_ldr_w(ctx, entry_dst, scratch, /*rn=*/29, fp_off);
                if (h.kind == HOME_SLOT) {
                    if (i64)
                        emit_str_x(ctx, entry_dst, scratch, 31, (uint32_t)(h.idx * 8u));
                    else
                        emit_str_w(ctx, entry_dst, scratch, 31, (uint32_t)(h.idx * 8u));
                }
            }
        }
    }

    // ---------- Walk each block and lower its ops.
    for (size_t bi = 0; bi < n_blocks; bi++) {
        MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, bi);
        MLIR_BlockHandle dst_blk = bm_get(&bm, src_blk);

        // Macro form for the common pattern: read operand IDX into
        // a register (returning the actual reg used — either the
        // operand's assigned home reg or the scratch if it had to be
        // loaded from a slot). For the result side use PICK_RES /
        // ST_RES — see comments on the underlying helpers above.
        #define LD_OPERAND(SCRATCH, IDX) \
            ld_operand_into(ctx, dst_blk, ra, op, (IDX), (SCRATCH))
        #define PICK_RES(SCRATCH, IDX) \
            pick_result_reg(ra, op, (IDX), (SCRATCH))
        #define ST_RESULT(PRODUCED_REG, IDX) \
            st_result(ctx, dst_blk, ra, op, (IDX), (PRODUCED_REG))

        size_t n_ops = MLIR_GetBlockNumOps(src_blk);
        for (size_t i = 0; i < n_ops; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(src_blk, i);
            MLIR_OpType  t  = MLIR_GetOpType(op);
            switch (t) {
            case OP_TYPE_WMIR_CONST: {
                MLIR_ValueHandle r = MLIR_GetOpResult(op, 0);
                MLIR_TypeHandle  ty = MLIR_GetValueType(r);
                string ts = MLIR_GetTypeString(ctx, ty);
                int64_t v = at_i(op, "value");
                uint8_t rd = PICK_RES(9, 0);
                if (ts.size == 3 && memcmp(ts.str, "i32", 3) == 0) {
                    emit_mov_imm32(ctx, dst_blk, rd, (uint32_t)v);
                    ST_RESULT(rd, 0);
                } else if (ts.size == 3 && memcmp(ts.str, "i64", 3) == 0) {
                    emit_mov_imm64(ctx, dst_blk, rd, (uint64_t)v);
                    ST_RESULT(rd, 0);
                } else {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.const of unsupported type '%.*s'\n",
                        (int)ts.size, ts.str);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                break;
            }
            case OP_TYPE_WMIR_MEMORY_SIZE: {
                // Load current wasm page count from linmem[MEM_PAGES_SLOT].
                // x28 is the callee-saved linmem base register
                // established at function entry.
                uint8_t rd = PICK_RES(9, 0);
                emit_ldr_w(ctx, dst_blk, rd, /*rn=*/28, /*off=*/MEM_PAGES_SLOT);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_MEMORY_GROW: {
                // memory.grow(delta) — bump linmem[MEM_PAGES_SLOT] by
                // delta wasm pages, capping at MAX_LINMEM_PAGES. The
                // backing virtual address space is already reserved by
                // the Mach-O envelope (large __linmem_bss S_ZEROFILL
                // section); macOS commits real pages lazily on touch.
                //
                // Returns the previous page count on success, or -1
                // (cast to i32) on failure.
                //
                // ABI:
                //   in:  r_delta (i32) = delta pages requested
                //   out: rd     (i32) = previous count, or 0xffffffff
                //
                // Codegen (uses scratch x9/x10/x11; lowering scratch):
                //   ldr  w9 , [x28, #MEM_PAGES_SLOT]   ; old
                //   add  w10, w9 , w_delta              ; new
                //   mov  w11, #MAX_LINMEM_PAGES
                //   cmp  w10, w11
                //   b.hi ^fail
                //   str  w10, [x28, #MEM_PAGES_SLOT]   ; commit
                //   mov  w_rd, w9                       ; return old
                //   b    ^done
                // ^fail:
                //   mov  w_rd, #-1
                //   b    ^done
                // ^done:
                uint8_t r_delta = LD_OPERAND(10, 0);
                uint8_t rd      = PICK_RES(9, 0);

                MLIR_BlockHandle fail_blk = MLIR_CreateBlock(ctx);
                MLIR_BlockHandle done_blk = MLIR_CreateBlock(ctx);
                MLIR_AppendRegionBlock(ctx, dst_region, fail_blk);
                MLIR_AppendRegionBlock(ctx, dst_region, done_blk);

                emit_ldr_w(ctx, dst_blk, /*rt=*/9 , /*rn=*/28, MEM_PAGES_SLOT);
                emit_add_reg(ctx, dst_blk, /*rd=*/10, /*rn=*/9, /*rm=*/r_delta,
                             /*sf=*/false);
                emit_mov_imm32(ctx, dst_blk, /*rd=*/11, MAX_LINMEM_PAGES);
                emit_cmp_reg(ctx, dst_blk, /*rn=*/10, /*rm=*/11, /*sf=*/false);
                // b.hi (unsigned higher) — new > cap.
                emit_b_cond(ctx, dst_blk, /*cond=*/0x8 /* HI */, fail_blk);
                emit_str_w(ctx, dst_blk, /*rt=*/10, /*rn=*/28, MEM_PAGES_SLOT);
                emit_mov_x(ctx, dst_blk, /*rd=*/rd, /*rm=*/9);
                emit_b(ctx, dst_blk, done_blk);

                emit_mov_imm32(ctx, fail_blk, /*rd=*/rd, (uint32_t)-1);
                emit_b(ctx, fail_blk, done_blk);

                dst_blk = done_blk;
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_IADD: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_add_reg(ctx, dst_blk, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_ISUB: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_sub_reg(ctx, dst_blk, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_IMUL: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_MUL, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_SDIV: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_SDIV, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_UDIV: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_UDIV, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_SREM: {
                // rem = a - (a / b) * b  ==  msub(sdiv(a,b), b, a).
                // Uses scratch x9 for the intermediate division result.
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);   // a — may be x9 if loaded
                uint8_t r1 = LD_OPERAND(10, 1);  // b — may be x10 if loaded
                // Need a temp that doesn't collide with r0 or r1.
                // x9 is the load scratch; if r0 happened to come from x9,
                // we'd clobber it before msub reads it. Use x10 as the
                // temp if r0 == 9, otherwise x9.
                uint8_t tmp = (r0 == 9 || r1 == 9) ? (r0 == 10 || r1 == 10 ? 11 : 10) : 9;
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_SDIV, tmp, r0, r1, sf);
                uint8_t rd = PICK_RES(9, 0);
                // msub reads tmp, r1, r0 then writes rd — reads-before-writes.
                emit_msub(ctx, dst_blk, rd, tmp, r1, r0, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_UREM: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t tmp = (r0 == 9 || r1 == 9) ? (r0 == 10 || r1 == 10 ? 11 : 10) : 9;
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_UDIV, tmp, r0, r1, sf);
                uint8_t rd = PICK_RES(9, 0);
                emit_msub(ctx, dst_blk, rd, tmp, r1, r0, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_IAND: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_AND_REG, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_IOR: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_ORR_REG, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_IXOR: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_EOR_REG, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_ISHL: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_LSL_REG, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_USHR: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_LSR_REG, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_SSHR: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                uint8_t rd = PICK_RES(9, 0);
                emit_3reg(ctx, dst_blk, OP_TYPE_AARCH64_ASR_REG, rd, r0, r1, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_SEXT: {
                // Sign-extend the low `src_bits` bits of operand[0] into
                // result type. src_bits defaults to 32 for the common
                // i32->i64 case.
                int64_t sb = at_i(op, "src_bits");
                if (sb == 0) sb = 32;
                bool dst_is64 = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t rd = PICK_RES(9, 0);
                if (sb == 8) {
                    MLIR_AttributeHandle a[3];
                    a[0] = attr_i32(ctx, "rd", rd);
                    a[1] = attr_i32(ctx, "rn", r0);
                    a[2] = attr_b(ctx, "sf", dst_is64);
                    MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_AARCH64_SXTB, a, 3));
                } else if (sb == 16) {
                    MLIR_AttributeHandle a[3];
                    a[0] = attr_i32(ctx, "rd", rd);
                    a[1] = attr_i32(ctx, "rn", r0);
                    a[2] = attr_b(ctx, "sf", dst_is64);
                    MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_AARCH64_SXTH, a, 3));
                } else {
                    // src_bits == 32 (or unspecified): sxtw is X-form only.
                    emit_2reg_no_sf(ctx, dst_blk, OP_TYPE_AARCH64_SXTW, rd, r0);
                }
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_ZEXT: {
                // ldr_w already zero-extends; for byte/half-width source
                // values stored as i32, the producer (e.g. ldrb) already
                // zeroed the upper bits. See pre-regalloc comment for
                // history. For now ZEXT is a value-bearing pass-through:
                // mov rd, r0 if regs differ; nothing if same.
                int64_t sb = at_i(op, "src_bits");
                if (sb == 0) sb = 32;
                (void)sb;
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t rd = PICK_RES(9, 0);
                if (rd != r0) emit_mov_x(ctx, dst_blk, rd, r0);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_TRUNC: {
                // i64 -> i32 truncate. We store only the low 32 bits via
                // the i32-width store, so a mov w_rd, w_r0 (zero-extending)
                // is sufficient.
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t rd = PICK_RES(9, 0);
                if (rd != r0) emit_mov_x(ctx, dst_blk, rd, r0);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_ICMP: {
                bool sf = is_i64(ctx, MLIR_GetOpOperand(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                emit_cmp_reg(ctx, dst_blk, r0, r1, sf);
                string pred = at_s(op, "pred");
                uint8_t cond = cond_for_pred(pred);
                uint8_t rd = PICK_RES(9, 0);
                emit_cset(ctx, dst_blk, rd, cond, /*sf=*/false);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_EQZ: {
                bool sf = is_i64(ctx, MLIR_GetOpOperand(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);
                emit_cmp_imm(ctx, dst_blk, r0, 0, sf);
                uint8_t rd = PICK_RES(9, 0);
                emit_cset(ctx, dst_blk, rd, COND_EQ, /*sf=*/false);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_SELECT: {
                bool sf = is_i64(ctx, MLIR_GetOpResult(op, 0));
                uint8_t r0 = LD_OPERAND(9, 0);   // a
                uint8_t r1 = LD_OPERAND(10, 1);  // b
                uint8_t r2 = LD_OPERAND(11, 2);  // cond (always i32)
                emit_cmp_imm(ctx, dst_blk, r2, 0, /*sf=*/false);
                uint8_t rd = PICK_RES(9, 0);
                emit_csel(ctx, dst_blk, rd, r0, r1, COND_NE, sf);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_FBINOP: {
                string k = at_s(op, "kind");
                int64_t fw = at_i(op, "fwidth");
                bool sf = (fw == 64);
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true, sf,
                               /*rd=*/0, /*rn=*/r0);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true, sf,
                               /*rd=*/1, /*rn=*/r1);
                const char *kind_lit = NULL;
                if      (k.size == 4 && memcmp(k.str, "fadd", 4) == 0) kind_lit = "fadd";
                else if (k.size == 4 && memcmp(k.str, "fsub", 4) == 0) kind_lit = "fsub";
                else if (k.size == 4 && memcmp(k.str, "fmul", 4) == 0) kind_lit = "fmul";
                else if (k.size == 4 && memcmp(k.str, "fdiv", 4) == 0) kind_lit = "fdiv";
                else {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.fbinop unknown kind '%.*s'\n",
                        (int)k.size, k.str);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                emit_fp_binop(ctx, dst_blk, kind_lit, (int)fw,
                              /*rd=*/0, /*rn=*/0, /*rm=*/1);
                uint8_t rd = PICK_RES(9, 0);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/false, sf,
                               /*rd=*/rd, /*rn=*/0);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_FUNOP: {
                string k = at_s(op, "kind");
                int64_t fw = at_i(op, "fwidth");
                bool sf = (fw == 64);
                uint8_t r0 = LD_OPERAND(9, 0);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true, sf,
                               /*rd=*/0, /*rn=*/r0);
                const char *kind_lit = NULL;
                if      (k.size == 4 && memcmp(k.str, "fabs", 4) == 0)  kind_lit = "fabs";
                else if (k.size == 4 && memcmp(k.str, "fneg", 4) == 0)  kind_lit = "fneg";
                else if (k.size == 5 && memcmp(k.str, "fsqrt", 5) == 0) kind_lit = "fsqrt";
                else {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.funop unknown kind '%.*s'\n",
                        (int)k.size, k.str);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                emit_fp_unop(ctx, dst_blk, kind_lit, (int)fw,
                             /*rd=*/0, /*rn=*/0);
                uint8_t rd = PICK_RES(9, 0);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/false, sf,
                               /*rd=*/rd, /*rn=*/0);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_FCMP: {
                string pred = at_s(op, "pred");
                int64_t fw = at_i(op, "fwidth");
                bool sf = (fw == 64);
                uint8_t r0 = LD_OPERAND(9, 0);
                uint8_t r1 = LD_OPERAND(10, 1);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true, sf,
                               /*rd=*/0, /*rn=*/r0);
                emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true, sf,
                               /*rd=*/1, /*rn=*/r1);
                emit_fcmp(ctx, dst_blk, (int)fw, /*rn=*/0, /*rm=*/1);
                uint8_t cond = COND_EQ;
                if      (pred.size == 3 && memcmp(pred.str, "oeq", 3) == 0) cond = COND_EQ;
                else if (pred.size == 3 && memcmp(pred.str, "une", 3) == 0) cond = COND_NE;
                else if (pred.size == 3 && memcmp(pred.str, "olt", 3) == 0) cond = COND_CC;
                else if (pred.size == 3 && memcmp(pred.str, "ole", 3) == 0) cond = COND_LS;
                else if (pred.size == 3 && memcmp(pred.str, "ogt", 3) == 0) cond = COND_HI;
                else if (pred.size == 3 && memcmp(pred.str, "oge", 3) == 0) cond = COND_CS;
                uint8_t rd = PICK_RES(9, 0);
                emit_cset(ctx, dst_blk, rd, cond, /*sf=*/false);
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_FCONV: {
                string k = at_s(op, "kind");
                int64_t src_w = at_i(op, "src_w");
                int64_t dst_w = at_i(op, "dst_w");
                bool sign = at_b(op, "sign");
                if (k.size == 3 && memcmp(k.str, "f2f", 3) == 0) {
                    uint8_t r0 = LD_OPERAND(9, 0);
                    emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true,
                                   /*sf=*/src_w == 64, /*rd=*/0, /*rn=*/r0);
                    emit_fp_cvt(ctx, dst_blk, "f2f", (int)src_w, (int)dst_w,
                                /*sign=*/false, /*rd=*/0, /*rn=*/0);
                    uint8_t rd = PICK_RES(9, 0);
                    emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/false,
                                   /*sf=*/dst_w == 64, /*rd=*/rd, /*rn=*/0);
                    ST_RESULT(rd, 0);
                } else if (k.size == 3 && memcmp(k.str, "f2i", 3) == 0) {
                    uint8_t r0 = LD_OPERAND(9, 0);
                    emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/true,
                                   /*sf=*/src_w == 64, /*rd=*/0, /*rn=*/r0);
                    uint8_t rd = PICK_RES(9, 0);
                    emit_fp_cvt(ctx, dst_blk, "f2i", (int)src_w, (int)dst_w,
                                sign, /*rd=*/rd, /*rn=*/0);
                    ST_RESULT(rd, 0);
                } else if (k.size == 3 && memcmp(k.str, "i2f", 3) == 0) {
                    uint8_t r0 = LD_OPERAND(9, 0);
                    emit_fp_cvt(ctx, dst_blk, "i2f", (int)src_w, (int)dst_w,
                                sign, /*rd=*/0, /*rn=*/r0);
                    uint8_t rd = PICK_RES(9, 0);
                    emit_fmov_gp_v(ctx, dst_blk, /*dir_to_v=*/false,
                                   /*sf=*/dst_w == 64, /*rd=*/rd, /*rn=*/0);
                    ST_RESULT(rd, 0);
                } else {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.fconv unknown kind '%.*s'\n",
                        (int)k.size, k.str);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                break;
            }
            case OP_TYPE_WMIR_GLOBAL_GET: {
                int64_t gi = at_i(op, "global_idx");
                uint8_t rd = PICK_RES(9, 0);
                emit_ldr_w(ctx, dst_blk, rd, 27, (uint32_t)(gi * 8));
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_GLOBAL_SET: {
                int64_t gi = at_i(op, "global_idx");
                uint8_t r0 = LD_OPERAND(9, 0);
                emit_str_w(ctx, dst_blk, r0, 27, (uint32_t)(gi * 8));
                break;
            }
            case OP_TYPE_WMIR_LOAD: {
                int64_t off = at_i(op, "memory_offset");
                int64_t sz  = at_i(op, "mem_size");
                if (sz == 0) sz = 4;
                uint8_t r0 = LD_OPERAND(9, 0);
                // x10 is scratch (not in pool, not r0). Use it for the
                // heap-addr computation: addr = linmem(x28) + r0.
                emit_add_reg(ctx, dst_blk, 10, 28, r0, /*sf=*/true);
                uint8_t rd = PICK_RES(9, 0);
                if (sz == 8) {
                    emit_ldr_x(ctx, dst_blk, rd, 10, (uint32_t)off);
                } else if (sz == 1) {
                    emit_ldrb_imm(ctx, dst_blk, rd, 10, (uint16_t)off);
                } else {
                    emit_ldr_w(ctx, dst_blk, rd, 10, (uint32_t)off);
                }
                ST_RESULT(rd, 0);
                break;
            }
            case OP_TYPE_WMIR_STORE: {
                int64_t off = at_i(op, "memory_offset");
                int64_t sz  = at_i(op, "mem_size");
                if (sz == 0) sz = 4;
                uint8_t r0 = LD_OPERAND(9, 0);   // index
                uint8_t r1 = LD_OPERAND(11, 1);  // value (use x11 scratch
                                                 // to leave x10 free for
                                                 // the heap_addr compute)
                emit_add_reg(ctx, dst_blk, 10, 28, r0, /*sf=*/true);
                if (sz == 8) {
                    emit_str_x(ctx, dst_blk, r1, 10, (uint32_t)off);
                } else if (sz == 1) {
                    emit_strb_imm(ctx, dst_blk, r1, 10, (uint16_t)off);
                } else {
                    emit_str_w(ctx, dst_blk, r1, 10, (uint32_t)off);
                }
                break;
            }
            case OP_TYPE_WMIR_CALL: {
                size_t na = MLIR_GetOpNumOperands(op);
                // Args 0..7 go in x0..x7. Args 8+ are passed on the
                // stack at [sp, #(k-8)*8]. The PCS requires SP to remain
                // 16-byte aligned across the call.
                size_t stack_args = (na > 8) ? na - 8 : 0;
                size_t reg_args   = (na > 8) ? 8 : na;
                uint32_t stack_bytes =
                    (uint32_t)((stack_args * 8u + 15u) & ~(size_t)15);
                if (stack_bytes > 0xfff) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.call needs %u bytes of stack "
                        "args (>4080)\n", stack_bytes);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                // ABI arg regs x0..x7 are outside our allocation pool
                // (which is x11..x18), so no operand's home register
                // can collide with the target arg reg. Load reg args
                // FIRST while SP is still at its "local slots" position.
                for (size_t k = 0; k < reg_args; k++) {
                    ld_operand_into_fixed(ctx, dst_blk, ra, op, k, (uint8_t)k);
                }
                if (stack_bytes > 0) {
                    // Shift SP down to expose the outgoing-arg area.
                    emit_sub_imm(ctx, dst_blk, /*rd=*/31, /*rn=*/31,
                                 (uint16_t)stack_bytes, /*sf=*/true);
                    // Now [sp, #0..stack_bytes-1] is the outgoing arg
                    // area, and any HOME_SLOT operand sits at
                    // [sp, #stack_bytes + slot*8] because we moved sp.
                    for (size_t k = 8; k < na; k++) {
                        MLIR_ValueHandle v = MLIR_GetOpOperand(op, k);
                        ValueHome h;
                        if (!wmir_regalloc_lookup(ra, v, &h)) {
                            fprintf(stderr,
                                "wmir->aarch64: unbound operand %zu in call\n", k);
                            wmir_regalloc_free(ra); bm_free(&bm);
                            return MLIR_INVALID_HANDLE;
                        }
                        bool i64 = is_i64(ctx, v);
                        uint8_t src_reg;
                        if (h.kind == HOME_REG) {
                            src_reg = h.idx;
                        } else {
                            src_reg = 9;
                            uint32_t off = (uint32_t)stack_bytes + (uint32_t)h.idx * 8u;
                            if (i64) emit_ldr_x(ctx, dst_blk, src_reg, 31, off);
                            else     emit_ldr_w(ctx, dst_blk, src_reg, 31, off);
                        }
                        uint32_t dst_off = (uint32_t)((k - 8) * 8u);
                        if (i64) emit_str_x(ctx, dst_blk, src_reg, 31, dst_off);
                        else     emit_str_w(ctx, dst_blk, src_reg, 31, dst_off);
                    }
                }
                string callee = at_s(op, "target");
                if (callee.size == 0) callee = at_s(op, "callee");
                emit_bl(ctx, dst_blk, callee);
                if (stack_bytes > 0) {
                    emit_add_imm(ctx, dst_blk, /*rd=*/31, /*rn=*/31,
                                 (uint16_t)stack_bytes, /*sf=*/true);
                }
                if (MLIR_GetOpNumResults(op) > 0) {
                    // Result arrives in x0; persist to its home.
                    st_result_from_fixed(ctx, dst_blk, ra, op, 0, 0);
                }
                break;
            }
            case OP_TYPE_WMIR_RETURN: {
                size_t no = MLIR_GetOpNumOperands(op);
                if (no > 1) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.return with %zu results not yet "
                        "supported\n", no);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                if (no == 1) {
                    ld_operand_into_fixed(ctx, dst_blk, ra, op, 0, /*reg=*/0);
                }
                emit_epilogue(ctx, dst_blk, frame_size);
                emit_ret(ctx, dst_blk);
                break;
            }
            case OP_TYPE_WMIR_UNREACHABLE: {
                emit_brk(ctx, dst_blk, 1);
                break;
            }
            case OP_TYPE_WMIR_BR: {
                // Copy each successor operand into the corresponding
                // target block-arg home, then branch. emit_branch_arg_copies
                // resolves reg→reg cycles via scratch x9.
                MLIR_BlockHandle s_target = MLIR_GetOpSuccessor(op, 0);
                MLIR_BlockHandle d_target = bm_get(&bm, s_target);
                if (d_target == MLIR_INVALID_HANDLE) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.br target block not mapped\n");
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                size_t n_sops = MLIR_GetOpNumSuccessorOperands(op, 0);
                size_t n_args = MLIR_GetBlockNumArgs(s_target);
                if (n_sops != n_args) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.br operand count %zu != target "
                        "arg count %zu\n", n_sops, n_args);
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                emit_branch_arg_copies(ctx, dst_blk, ra, op, 0, s_target);
                emit_b(ctx, dst_blk, d_target);
                break;
            }
            case OP_TYPE_WMIR_COND_BR: {
                // Two successors, no successor operands by construction.
                if (MLIR_GetOpNumSuccessors(op) < 2) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.cond_br needs 2 successors\n");
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                MLIR_BlockHandle t_src = MLIR_GetOpSuccessor(op, 0);
                MLIR_BlockHandle f_src = MLIR_GetOpSuccessor(op, 1);
                MLIR_BlockHandle t_dst = bm_get(&bm, t_src);
                MLIR_BlockHandle f_dst = bm_get(&bm, f_src);
                if (t_dst == MLIR_INVALID_HANDLE || f_dst == MLIR_INVALID_HANDLE) {
                    fprintf(stderr,
                        "wmir->aarch64: wmir.cond_br target block not mapped\n");
                    wmir_regalloc_free(ra); bm_free(&bm);
                    return MLIR_INVALID_HANDLE;
                }
                uint8_t r0 = LD_OPERAND(9, 0);
                emit_cbnz(ctx, dst_blk, r0, /*sf=*/false, t_dst);
                emit_b(ctx, dst_blk, f_dst);
                break;
            }
            default: {
                string nm = MLIR_GetOpName(op);
                fprintf(stderr,
                    "wmir->aarch64: unsupported wmir op '%.*s' (kind=%d)\n",
                    (int)nm.size, nm.str, (int)t);
                wmir_regalloc_free(ra); bm_free(&bm);
                return MLIR_INVALID_HANDLE;
            }
            }
        }

        #undef LD_OPERAND
        #undef PICK_RES
        #undef ST_RESULT
    }

    wmir_regalloc_free(ra);
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
        // Stash argc / argv (host pointer) into linmem so the WASI
        // args_get / args_sizes_get shims can recover them. macOS uses
        // LC_MAIN: dyld calls our entry as `int main(argc, argv, envp,
        // apple)` with argc in x0 and argv in x1 (standard C ABI), NOT
        // via the kernel-style user stack layout (which has argc at
        // [sp,#0]; that layout is only used for LC_UNIXTHREAD entries).
        // Done BEFORE the bl so the callee can read them. Offsets must
        // match ARGC_SLOT/ARGV_SLOT (defined later in the file alongside
        // the linmem layout doc).
        emit_str_w   (ctx, blk, /*rt=*/ 0, /*rn=*/28, /*off=*/40);
        emit_str_x   (ctx, blk, /*rt=*/ 1, /*rn=*/28, /*off=*/48);
        // Initialise mem_pages slot with the static page count. The
        // `wmir.memory_size` op loads from this slot, and
        // `wmir.memory_grow` updates it (capped at MAX_LINMEM_PAGES).
        // See the linmem layout doc for MEM_PAGES_SLOT.
        emit_mov_imm32(ctx, blk, /*rd=*/9, g_linmem_pages);
        emit_str_w   (ctx, blk, /*rt=*/ 9, /*rn=*/28, /*off=*/24);
    }

    emit_bl(ctx, blk, main_name);
    // After main returns, exit with x0 (main's return value) as the
    // status code. Calls libSystem _exit; noreturn.
    emit_bl(ctx, blk, str_lit("_exit"));

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
// Synthesise printI64(x0: i64) -> void by formatting digits into a
// 32-byte stack buffer and writing via the macOS BSD `write` syscall
// (SYS_write = 4, x16 = 4, svc #0x80). Handles negative numbers (with a
// leading '-') for values where -v is representable; INT64_MIN is a
// known corner case left unhandled.
//
// Block layout:
//   entry           prologue; cmp x0, #0; b.ge ^pos
//   neg             mov x10, #1; sub x0, xzr, x0; b ^pos
//   pos             add x11, sp, #24; b ^loop
//   loop            divmod by 10; strb digit; cbnz x0, ^loop; b ^after
//   after           cbz x10, ^skip; strb '-'; b ^skip
//   skip            compute count; setup x0/x1/x2/x16; svc; epilogue; ret
// =============================================================================
static MLIR_OpHandle synth_print_i64(MLIR_Context *ctx) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bneg  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bpos  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bloop = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bafter= MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bskip = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, entry);
    MLIR_AppendRegionBlock(ctx, region, bneg);
    MLIR_AppendRegionBlock(ctx, region, bpos);
    MLIR_AppendRegionBlock(ctx, region, bloop);
    MLIR_AppendRegionBlock(ctx, region, bafter);
    MLIR_AppendRegionBlock(ctx, region, bskip);

    emit_prologue(ctx, entry, /*frame_size=*/32);
    // Default x10 = 0 (no sign). The neg-path overrides this with 1.
    emit_movz(ctx, entry, /*rd=*/10, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_cmp_imm(ctx, entry, /*rn=*/0, /*imm12=*/0, /*sf=*/true);
    emit_b_cond(ctx, entry, COND_GE, bpos);
    emit_b(ctx, entry, bneg);

    emit_movz(ctx, bneg, /*rd=*/10, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    // sub Xd, XZR, Xn -> Xd = -Xn   (rn=31 = XZR for shifted-register form)
    {
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 0);
        a[1] = attr_i32(ctx, "rn", 31);
        a[2] = attr_i32(ctx, "rm", 0);
        a[3] = attr_b(ctx, "sf", true);
        MLIR_AppendBlockOp(ctx, bneg,
            build_op(ctx, OP_TYPE_AARCH64_SUB_REG, a, 4));
    }
    emit_b(ctx, bneg, bpos);

    // bpos: x11 = sp + 24  (one past end of digit buffer)
    emit_add_imm(ctx, bpos, /*rd=*/11, /*rn=*/31, /*imm12=*/24, /*sf=*/true);
    emit_b(ctx, bpos, bloop);

    // bloop: divide by 10, store digit, dec ptr, loop while non-zero
    emit_movz(ctx, bloop, /*rd=*/12, /*imm16=*/10, /*hw=*/0, /*sf=*/true);
    // udiv x13, x0, x12
    {
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 13);
        a[1] = attr_i32(ctx, "rn", 0);
        a[2] = attr_i32(ctx, "rm", 12);
        a[3] = attr_b(ctx, "sf", true);
        MLIR_AppendBlockOp(ctx, bloop,
            build_op(ctx, OP_TYPE_AARCH64_UDIV, a, 4));
    }
    // msub x14, x13, x12, x0  -> x14 = x0 - x13*x12 = x0 % 10
    {
        MLIR_AttributeHandle a[5];
        a[0] = attr_i32(ctx, "rd", 14);
        a[1] = attr_i32(ctx, "rn", 13);
        a[2] = attr_i32(ctx, "rm", 12);
        a[3] = attr_i32(ctx, "ra", 0);
        a[4] = attr_b(ctx, "sf", true);
        MLIR_AppendBlockOp(ctx, bloop,
            build_op(ctx, OP_TYPE_AARCH64_MSUB, a, 5));
    }
    // add w14, w14, #'0'
    emit_add_imm(ctx, bloop, /*rd=*/14, /*rn=*/14,
                 /*imm12=*/0x30, /*sf=*/false);
    // sub x11, x11, #1
    emit_sub_imm(ctx, bloop, /*rd=*/11, /*rn=*/11,
                 /*imm12=*/1, /*sf=*/true);
    // strb w14, [x11, #0]
    emit_strb_imm(ctx, bloop, /*rt=*/14, /*rn=*/11, /*off=*/0);
    // mov x0, x13
    emit_mov_x(ctx, bloop, /*rd=*/0, /*rn=*/13);
    // cbnz x0, ^loop
    emit_cbnz(ctx, bloop, /*rt=*/0, /*sf=*/true, bloop);
    emit_b(ctx, bloop, bafter);

    // bafter: if x10 == 0, jump to skip; else prepend '-'
    emit_cbz(ctx, bafter, /*rt=*/10, /*sf=*/true, bskip);
    emit_movz(ctx, bafter, /*rd=*/14, /*imm16=*/0x2d, /*hw=*/0, /*sf=*/false);
    emit_sub_imm(ctx, bafter, /*rd=*/11, /*rn=*/11,
                 /*imm12=*/1, /*sf=*/true);
    emit_strb_imm(ctx, bafter, /*rt=*/14, /*rn=*/11, /*off=*/0);
    emit_b(ctx, bafter, bskip);

    // bskip: setup syscall write(1, x11, count); count = (sp+24) - x11
    emit_add_imm(ctx, bskip, /*rd=*/12, /*rn=*/31,
                 /*imm12=*/24, /*sf=*/true);
    // sub x2, x12, x11
    {
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 2);
        a[1] = attr_i32(ctx, "rn", 12);
        a[2] = attr_i32(ctx, "rm", 11);
        a[3] = attr_b(ctx, "sf", true);
        MLIR_AppendBlockOp(ctx, bskip,
            build_op(ctx, OP_TYPE_AARCH64_SUB_REG, a, 4));
    }
    emit_movz(ctx, bskip, /*rd=*/0, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_mov_x(ctx, bskip, /*rd=*/1, /*rn=*/11);
    emit_bl(ctx, bskip, str_lit("_write"));
    emit_epilogue(ctx, bskip, /*frame_size=*/32);
    emit_ret(ctx, bskip);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", "printI64", 8);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Synthesise printNewline() -> void by writing a single '\n' via the BSD
// write syscall. Uses one byte of the 16-byte stack frame as the buffer.
// =============================================================================
static MLIR_OpHandle synth_print_newline(MLIR_Context *ctx) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);

    emit_prologue(ctx, blk, /*frame_size=*/16);
    emit_movz(ctx, blk, /*rd=*/9, /*imm16=*/0x0a, /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, blk, /*rt=*/9, /*rn=*/31, /*off=*/0);
    emit_movz(ctx, blk, /*rd=*/0, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    // x1 = sp  ->  add x1, sp, #0
    emit_add_imm(ctx, blk, /*rd=*/1, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_movz(ctx, blk, /*rd=*/2, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, blk, str_lit("_write"));
    emit_epilogue(ctx, blk, /*frame_size=*/16);
    emit_ret(ctx, blk);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", "printNewline", 12);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Synthesise printStr(i32 wasm_offset) -> void:
//   * w0 holds a wasm linmem offset; the host pointer is x28 + zext(w0).
//   * Loop scanning bytes until a NUL is found to compute the length.
//   * Issue write(1, host_ptr, len) via the BSD `write` syscall (#4).
//
// Block layout (each becomes its own aarch64 block):
//   entry: prologue; x10 = host_ptr (start); x11 = x10 (scan cursor); b loop
//   loop : ldrb w12, [x11]; cbz w12, ^done; add x11, x11, #1; b ^loop
//   done : x2 = x11 - x10; x1 = x10; x0 = 1; x16 = 4; svc #0x80; epi; ret
// =============================================================================
static MLIR_OpHandle synth_print_str(MLIR_Context *ctx) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bloop = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bdone = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, entry);
    MLIR_AppendRegionBlock(ctx, region, bloop);
    MLIR_AppendRegionBlock(ctx, region, bdone);

    // entry
    emit_prologue(ctx, entry, /*frame_size=*/16);
    // mov w9, w0   (UXTW: zero-extends w0 into x9)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9);
        a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    // add x10, x28, x9        ; host pointer = linmem + offset
    emit_add_reg(ctx, entry, /*rd=*/10, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // mov x11, x10            ; scan cursor
    emit_mov_x(ctx, entry, /*rd=*/11, /*rn=*/10);
    emit_b(ctx, entry, bloop);

    // loop
    emit_ldrb_imm(ctx, bloop, /*rt=*/12, /*rn=*/11, /*off=*/0);
    emit_cbz(ctx, bloop, /*rt=*/12, /*sf=*/false, bdone);
    emit_add_imm(ctx, bloop, /*rd=*/11, /*rn=*/11, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, bloop, bloop);

    // done: x2 = x11 - x10; x1 = x10; x0 = 1; x16 = 4; svc
    emit_sub_reg(ctx, bdone, /*rd=*/2, /*rn=*/11, /*rm=*/10, /*sf=*/true);
    emit_mov_x(ctx, bdone, /*rd=*/1, /*rn=*/10);
    emit_movz(ctx, bdone, /*rd=*/0, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, bdone, str_lit("_write"));
    // Append '\n' (matches runtime.c semantics: printStr always emits
    // a trailing newline). Reuse the prologue stack slot at [sp, #0].
    emit_movz(ctx, bdone, /*rd=*/9, /*imm16=*/0x0a, /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bdone, /*rt=*/9, /*rn=*/31, /*off=*/0);
    emit_movz(ctx, bdone, /*rd=*/0, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_add_imm(ctx, bdone, /*rd=*/1, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_movz(ctx, bdone, /*rd=*/2, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, bdone, str_lit("_write"));
    emit_epilogue(ctx, bdone, /*frame_size=*/16);
    emit_ret(ctx, bdone);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", "printStr", 8);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Helper used by synth_printf: emit a `write(1, x_ptr_reg, x_len_reg)` BSD
// syscall sequence. Clobbers x0, x1, x2, x16.
// =============================================================================
static void emit_write_stdout(MLIR_Context *ctx, MLIR_BlockHandle blk,
                              uint8_t ptr_reg, uint8_t len_reg) {
    if (len_reg != 2) emit_mov_x(ctx, blk, /*rd=*/2, /*rn=*/len_reg);
    if (ptr_reg != 1) emit_mov_x(ctx, blk, /*rd=*/1, /*rn=*/ptr_reg);
    emit_movz(ctx, blk, /*rd=*/0,  /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, blk, str_lit("_write"));
}

// =============================================================================
// Synthesise `printf(i32 fmt_ofs, i32 args_ofs) -> i32`.
//
// The C `printf(const char *fmt, ...)` is desugared by tinyC's frontend
// into a two-argument call: `printf(fmt, &args_blob)`, where `args_blob`
// is a contiguous region in linmem that the caller has populated with the
// variadic arguments (each i32 arg as 4 bytes, each i64 arg as 8 bytes,
// with natural alignment for 8-byte slots). The Mach-O backend then sees
// this as a regular call with two i32 arguments.
//
// Implemented format directives (matches `runtime_wasm.c`'s vprintf_impl
// minimum subset used by the tinyC test corpus):
//
//   %%           literal '%'
//   %c           int -> single byte
//   %d  %i       int (4-byte slot) decimal
//   %ld %li      long (i32 on wasm32, 4-byte slot) decimal
//   %lld %lli    long long (8-byte slot, aligned to 8) decimal
//   %u           unsigned int (4-byte slot) decimal
//   %lu          unsigned long (4-byte slot) decimal
//   %llu         unsigned long long (8-byte slot) decimal
//   %s           const char * (4-byte slot, wasm offset)
//
// Anything else (precision specs like %.2f, hex %x, pointer %p, floats
// %f/%g) is silently consumed for now — those few tinyC tests fail with
// "unsupported printf spec" rather than producing garbage output.
//
// Output happens directly via `write(2)` syscalls on fd 1. Decimal
// printing for both i32 and i64 paths reuses the existing `printI64`
// helper (which only ever clobbers x0-x18 and never touches x19-x28).
// =============================================================================
static MLIR_OpHandle synth_printf(MLIR_Context *ctx) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry        = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle loop_top     = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle write_lit    = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle saw_pct      = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_top     = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_inc_l   = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_pct     = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_decimal = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_d_word  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_d_ll    = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_s       = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_s_loop  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_s_done  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_c       = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_f       = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle spec_skip    = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle done         = MLIR_CreateBlock(ctx);

    MLIR_AppendRegionBlock(ctx, region, entry);
    MLIR_AppendRegionBlock(ctx, region, loop_top);
    MLIR_AppendRegionBlock(ctx, region, write_lit);
    MLIR_AppendRegionBlock(ctx, region, saw_pct);
    MLIR_AppendRegionBlock(ctx, region, spec_top);
    MLIR_AppendRegionBlock(ctx, region, spec_inc_l);
    MLIR_AppendRegionBlock(ctx, region, spec_pct);
    MLIR_AppendRegionBlock(ctx, region, spec_decimal);
    MLIR_AppendRegionBlock(ctx, region, spec_d_word);
    MLIR_AppendRegionBlock(ctx, region, spec_d_ll);
    MLIR_AppendRegionBlock(ctx, region, spec_s);
    MLIR_AppendRegionBlock(ctx, region, spec_s_loop);
    MLIR_AppendRegionBlock(ctx, region, spec_s_done);
    MLIR_AppendRegionBlock(ctx, region, spec_c);
    MLIR_AppendRegionBlock(ctx, region, spec_f);
    MLIR_AppendRegionBlock(ctx, region, spec_skip);
    MLIR_AppendRegionBlock(ctx, region, done);

    // ---------- entry ----------
    // Prologue + 32-byte scratch frame. [sp+0] is a 1-byte buffer for
    // emitting literal chars via the write(2) syscall.
    emit_prologue(ctx, entry, /*frame_size=*/32);
    // x19 = x28 + zext(w0)   (fmt cursor)
    // x20 = x28 + zext(w1)   (args cursor)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9);
        a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/19, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9);
        a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/20, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    emit_b(ctx, entry, loop_top);

    // ---------- loop_top ----------
    // c = *fmt++; if c == 0 done; else if c == '%' saw_pct; else emit c.
    emit_ldrb_imm(ctx, loop_top, /*rt=*/9, /*rn=*/19, /*off=*/0);
    emit_add_imm(ctx, loop_top, /*rd=*/19, /*rn=*/19, /*imm12=*/1, /*sf=*/true);
    emit_cbz(ctx, loop_top, /*rt=*/9, /*sf=*/false, done);
    emit_cmp_imm(ctx, loop_top, /*rn=*/9, /*imm12=*/'%', /*sf=*/false);
    emit_b_cond(ctx, loop_top, COND_EQ, saw_pct);
    emit_b(ctx, loop_top, write_lit);

    // ---------- write_lit ----------
    // [sp+0] = c; write(1, sp, 1).
    emit_strb_imm(ctx, write_lit, /*rt=*/9, /*rn=*/31, /*off=*/0);
    emit_movz(ctx, write_lit, /*rd=*/2, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_add_imm(ctx, write_lit, /*rd=*/1, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_movz(ctx, write_lit, /*rd=*/0,  /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, write_lit, str_lit("_write"));
    emit_b(ctx, write_lit, loop_top);

    // ---------- saw_pct ----------
    // ll_count = 0 (x21); fall through to spec_top.
    emit_movz(ctx, saw_pct, /*rd=*/21, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_b(ctx, saw_pct, spec_top);

    // ---------- spec_top ----------
    // c = *fmt++; dispatch.
    emit_ldrb_imm(ctx, spec_top, /*rt=*/9, /*rn=*/19, /*off=*/0);
    emit_add_imm(ctx, spec_top, /*rd=*/19, /*rn=*/19, /*imm12=*/1, /*sf=*/true);
    emit_cbz(ctx, spec_top, /*rt=*/9, /*sf=*/false, done);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'l', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_inc_l);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'%', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_pct);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'d', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_decimal);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'i', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_decimal);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'u', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_decimal);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'s', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_s);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'c', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_c);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'f', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_f);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'g', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_f);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'e', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_f);
    // Precision / width specifiers: skip '.' and any decimal digits and
    // come back around for the actual conversion character.
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'.', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_EQ, spec_skip);
    // '0' .. '9' → skip (treat as width/precision digit)
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'0', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_LT, loop_top);
    emit_cmp_imm(ctx, spec_top, /*rn=*/9, /*imm12=*/'9', /*sf=*/false);
    emit_b_cond(ctx, spec_top, COND_LE, spec_skip);
    // Unknown spec char: silently skip back to top-level fmt loop.
    emit_b(ctx, spec_top, loop_top);

    // ---------- spec_inc_l ----------
    // ll_count++; back to spec_top.
    emit_add_imm(ctx, spec_inc_l, /*rd=*/21, /*rn=*/21,
                 /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, spec_inc_l, spec_top);

    // ---------- spec_pct ----------
    // Write '%' and resume.
    emit_movz(ctx, spec_pct, /*rd=*/9, /*imm16=*/'%', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, spec_pct, /*rt=*/9, /*rn=*/31, /*off=*/0);
    emit_movz(ctx, spec_pct, /*rd=*/2, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_add_imm(ctx, spec_pct, /*rd=*/1, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_movz(ctx, spec_pct, /*rd=*/0,  /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, spec_pct, str_lit("_write"));
    emit_b(ctx, spec_pct, loop_top);

    // ---------- spec_decimal ----------
    // Branch on ll_count: ll_count == 2 -> spec_d_ll, else spec_d_word.
    emit_cmp_imm(ctx, spec_decimal, /*rn=*/21, /*imm12=*/2, /*sf=*/true);
    emit_b_cond(ctx, spec_decimal, COND_GE, spec_d_ll);
    emit_b(ctx, spec_decimal, spec_d_word);

    // ---------- spec_d_word ----------
    // i32/long arg: read 4 bytes from args cursor, sext to i64, bl printI64.
    emit_ldr_w(ctx, spec_d_word, /*rt=*/9, /*rn=*/20, /*off=*/0);
    emit_add_imm(ctx, spec_d_word, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/4, /*sf=*/true);
    // sxtw x0, w9
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 0);
        a[1] = attr_i32(ctx, "rn", 9);
        MLIR_AppendBlockOp(ctx, spec_d_word,
            build_op(ctx, OP_TYPE_AARCH64_SXTW, a, 2));
    }
    emit_bl(ctx, spec_d_word, str_lit("printI64"));
    emit_b(ctx, spec_d_word, loop_top);

    // ---------- spec_d_ll ----------
    // i64 arg: align cursor up to 8, read 8 bytes, bl printI64.
    // Alignment: x20 = ((x20 + 7) >> 3) << 3.
    emit_add_imm(ctx, spec_d_ll, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/7, /*sf=*/true);
    emit_movz(ctx, spec_d_ll, /*rd=*/12, /*imm16=*/3, /*hw=*/0, /*sf=*/true);
    emit_3reg(ctx, spec_d_ll, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/20, /*rn=*/20, /*rm=*/12, /*sf=*/true);
    emit_3reg(ctx, spec_d_ll, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/20, /*rn=*/20, /*rm=*/12, /*sf=*/true);
    emit_ldr_x(ctx, spec_d_ll, /*rt=*/0, /*rn=*/20, /*off=*/0);
    emit_add_imm(ctx, spec_d_ll, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/8, /*sf=*/true);
    emit_bl(ctx, spec_d_ll, str_lit("printI64"));
    emit_b(ctx, spec_d_ll, loop_top);

    // ---------- spec_s ----------
    // Read i32 (wasm offset) from args, advance, set x9 = x28 + offset (host),
    // x10 = x9 (cursor).
    emit_ldr_w(ctx, spec_s, /*rt=*/9, /*rn=*/20, /*off=*/0);
    emit_add_imm(ctx, spec_s, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/4, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9);
        a[1] = attr_i32(ctx, "rn", 9);
        MLIR_AppendBlockOp(ctx, spec_s,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    // x10 = x28 + x9 (host ptr start); x11 = x10 (scan cursor)
    emit_add_reg(ctx, spec_s, /*rd=*/10, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    emit_mov_x(ctx, spec_s, /*rd=*/11, /*rn=*/10);
    emit_b(ctx, spec_s, spec_s_loop);

    // ---------- spec_s_loop ----------
    // while (*x11) x11++;
    emit_ldrb_imm(ctx, spec_s_loop, /*rt=*/12, /*rn=*/11, /*off=*/0);
    emit_cbz(ctx, spec_s_loop, /*rt=*/12, /*sf=*/false, spec_s_done);
    emit_add_imm(ctx, spec_s_loop, /*rd=*/11, /*rn=*/11,
                 /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, spec_s_loop, spec_s_loop);

    // ---------- spec_s_done ----------
    // write(1, x10, x11 - x10).
    emit_sub_reg(ctx, spec_s_done, /*rd=*/2, /*rn=*/11, /*rm=*/10, /*sf=*/true);
    emit_mov_x(ctx, spec_s_done, /*rd=*/1, /*rn=*/10);
    emit_movz(ctx, spec_s_done, /*rd=*/0,  /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, spec_s_done, str_lit("_write"));
    emit_b(ctx, spec_s_done, loop_top);

    // ---------- spec_c ----------
    // Read i32 from args, write low byte.
    emit_ldr_w(ctx, spec_c, /*rt=*/9, /*rn=*/20, /*off=*/0);
    emit_add_imm(ctx, spec_c, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/4, /*sf=*/true);
    emit_strb_imm(ctx, spec_c, /*rt=*/9, /*rn=*/31, /*off=*/0);
    emit_movz(ctx, spec_c, /*rd=*/2, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_add_imm(ctx, spec_c, /*rd=*/1, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_movz(ctx, spec_c, /*rd=*/0,  /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, spec_c, str_lit("_write"));
    emit_b(ctx, spec_c, loop_top);

    // ---------- spec_f ----------
    // %f / %g / %e: read a double (8 bytes, aligned to 8) from args and
    // BL printF64. The double's i64 bit pattern is passed in x0 as the
    // wmir carrier convention.
    emit_add_imm(ctx, spec_f, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/7, /*sf=*/true);
    emit_movz(ctx, spec_f, /*rd=*/12, /*imm16=*/3, /*hw=*/0, /*sf=*/true);
    emit_3reg(ctx, spec_f, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/20, /*rn=*/20, /*rm=*/12, /*sf=*/true);
    emit_3reg(ctx, spec_f, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/20, /*rn=*/20, /*rm=*/12, /*sf=*/true);
    emit_ldr_x(ctx, spec_f, /*rt=*/0, /*rn=*/20, /*off=*/0);
    emit_add_imm(ctx, spec_f, /*rd=*/20, /*rn=*/20,
                 /*imm12=*/8, /*sf=*/true);
    emit_bl(ctx, spec_f, str_lit("printF64"));
    emit_b(ctx, spec_f, loop_top);

    // ---------- spec_skip ----------
    // Width / precision digit or '.': consume but don't reset ll_count.
    emit_b(ctx, spec_skip, spec_top);

    // ---------- done ----------
    // Return 0 (we don't bother tracking the actual byte count).
    emit_movz(ctx, done, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, done, /*frame_size=*/32);
    emit_ret(ctx, done);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", "printf", 6);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Helper: emit a function header for a leaf-ish synthesised helper.
// Creates the region, appends `entry` as the first block, emits the
// 16-byte-frame prologue, and returns the entry block.
// =============================================================================
static MLIR_RegionHandle synth_leaf_begin(MLIR_Context *ctx,
                                         MLIR_BlockHandle *entry_out,
                                         uint16_t frame_size) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, entry);
    emit_prologue(ctx, entry, frame_size);
    *entry_out = entry;
    return region;
}
static MLIR_OpHandle synth_leaf_finish(MLIR_Context *ctx,
                                      MLIR_RegionHandle region,
                                      const char *name, size_t name_len) {
    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", name, name_len);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Linmem layout convention for synthesised libc helpers:
//
//   [ 0..15 ]   reserved (matches the wasm/wasm-ld preamble)
//   [16..19]    `malloc_heap_off` — 4-byte counter, the next free byte
//               offset within the synthesised heap. Initialised lazily
//               to MALLOC_HEAP_BASE on the first malloc call (zero is
//               the sentinel "uninitialised" value).
//   [24..27]    `mem_pages` (i32) — current wasm linear-memory size in
//               pages; initialised by `_start` to the static page
//               count derived from the wasm module's MEMORY section
//               (`g_linmem_pages`), and bumped by `memory_grow` up to
//               MAX_LINMEM_PAGES. `memory_size` returns this value.
//   [40..43]    `argc` (i32) — captured by `_start` from x0 on entry.
//   [48..55]    `argv` (host ptr, 8 bytes) — captured by `_start` from
//               x1 on entry (LC_MAIN ABI; LC_UNIXTHREAD would use
//               sp+0 / sp+8 instead). Used by synth_args_get /
//               synth_args_sizes_get.
//   [20..1023]  reserved (modulo the slots above).
//   [ 1024 .. WASM_DATA_BASE + linmem_init_size ] static data globals
//   [ ... .. MALLOC_HEAP_BASE )                  BSS-equivalent zero pages
//   [ MALLOC_HEAP_BASE .. MALLOC_HEAP_TOP )      bump-allocated heap
//   [ MALLOC_HEAP_TOP .. DEFAULT_STACK_SIZE )    free / wasm stack
//
// MALLOC_HEAP_BASE = 1 MiB and MALLOC_HEAP_TOP = 2 MiB. That leaves the
// upper 2 MiB of linmem for the wasm stack (global 0 starts at the very
// top, growing down). 1 MiB of heap is plenty for the tinyC test suite.
//
// The slot offsets above (MALLOC_HEAP_OFF_SLOT, MEM_PAGES_SLOT,
// ARGC_SLOT, ARGV_SLOT) and MALLOC_HEAP_BASE_BYTES are defined in the
// hoisted enum near the top of this file (they're referenced before
// this comment by the `wmir.memory_size` / `memory.grow` lowering).
// =============================================================================

// -----------------------------------------------------------------------------
// malloc(i32 size) -> i32: bump-allocate (size+15 & ~15) bytes from the
// synthesised heap, return the wasm offset. No error path — overflow
// silently wraps for now; tinyC tests never come close to the cap.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_malloc(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, init_done;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    init_done = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, init_done);

    // w9 = *heap_off (load from linmem[16])
    emit_ldr_w(ctx, entry, /*rt=*/9, /*rn=*/28,
               /*off=*/MALLOC_HEAP_OFF_SLOT);
    // If zero, initialise; else fall through to init_done.
    emit_cbnz(ctx, entry, /*rt=*/9, /*sf=*/false, init_done);
    // movz w9, #16, lsl #16  -> 0x100000 = 1 MiB
    emit_movz(ctx, entry, /*rd=*/9,
              /*imm16=*/MALLOC_HEAP_BASE_BYTES >> 16,
              /*hw=*/1, /*sf=*/false);
    emit_b(ctx, entry, init_done);

    // init_done: align w9 up to 16, advance heap_off, return aligned w9.
    emit_add_imm(ctx, init_done, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/15, /*sf=*/false);
    emit_movz(ctx, init_done, /*rd=*/12, /*imm16=*/4, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, init_done, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/9, /*rn=*/9, /*rm=*/12, /*sf=*/false);
    emit_3reg(ctx, init_done, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/9, /*rn=*/9, /*rm=*/12, /*sf=*/false);
    // w10 = aligned + size
    emit_add_reg(ctx, init_done, /*rd=*/10, /*rn=*/9, /*rm=*/0, /*sf=*/false);
    emit_str_w(ctx, init_done, /*rt=*/10, /*rn=*/28,
               /*off=*/MALLOC_HEAP_OFF_SLOT);
    // Return aligned offset.
    emit_mov_x(ctx, init_done, /*rd=*/0, /*rn=*/9);
    emit_epilogue(ctx, init_done, /*frame_size=*/16);
    emit_ret(ctx, init_done);

    return synth_leaf_finish(ctx, region, "malloc", 6);
}

// -----------------------------------------------------------------------------
// free(i32 p): no-op (the bump allocator never reclaims).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_free(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "free", 4);
}

// -----------------------------------------------------------------------------
// strlen(i32 ofs) -> i32. Walks until NUL, returns scanned count.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_strlen(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, done;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    done = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, done);

    // x9 = host ptr; x10 = scan cursor
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9);
        a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    emit_mov_x(ctx, entry, /*rd=*/10, /*rn=*/9);
    emit_b(ctx, entry, loop);

    emit_ldrb_imm(ctx, loop, /*rt=*/11, /*rn=*/10, /*off=*/0);
    emit_cbz(ctx, loop, /*rt=*/11, /*sf=*/false, done);
    emit_add_imm(ctx, loop, /*rd=*/10, /*rn=*/10, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, loop, loop);

    // w0 = w10 - w9  (length fits in 32 bits)
    emit_sub_reg(ctx, done, /*rd=*/0, /*rn=*/10, /*rm=*/9, /*sf=*/false);
    emit_epilogue(ctx, done, /*frame_size=*/16);
    emit_ret(ctx, done);

    return synth_leaf_finish(ctx, region, "strlen", 6);
}

// -----------------------------------------------------------------------------
// strcmp(i32 a_ofs, i32 b_ofs) -> i32. Returns (unsigned char)*a - (unsigned
// char)*b at the first mismatch, or 0 at both NULs.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_strcmp(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, diff, equal;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    diff  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, diff);
    equal = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, equal);

    // x9 = host(a); x10 = host(b)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_b(ctx, entry, loop);

    emit_ldrb_imm(ctx, loop, /*rt=*/11, /*rn=*/9,  /*off=*/0);
    emit_ldrb_imm(ctx, loop, /*rt=*/12, /*rn=*/10, /*off=*/0);
    emit_cmp_reg(ctx, loop, /*rn=*/11, /*rm=*/12, /*sf=*/false);
    emit_b_cond(ctx, loop, COND_NE, diff);
    emit_cbz(ctx, loop, /*rt=*/11, /*sf=*/false, equal);
    emit_add_imm(ctx, loop, /*rd=*/9,  /*rn=*/9,  /*imm12=*/1, /*sf=*/true);
    emit_add_imm(ctx, loop, /*rd=*/10, /*rn=*/10, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, loop, loop);

    emit_sub_reg(ctx, diff, /*rd=*/0, /*rn=*/11, /*rm=*/12, /*sf=*/false);
    emit_epilogue(ctx, diff, /*frame_size=*/16);
    emit_ret(ctx, diff);

    emit_movz(ctx, equal, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, equal, /*frame_size=*/16);
    emit_ret(ctx, equal);

    return synth_leaf_finish(ctx, region, "strcmp", 6);
}

// -----------------------------------------------------------------------------
// memcmp(i32 a_ofs, i32 b_ofs, i32 n) -> i32. Same as strcmp but bounded.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_memcmp(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, diff, equal;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    diff  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, diff);
    equal = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, equal);

    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_b(ctx, entry, loop);

    // if n == 0: equal
    emit_cbz(ctx, loop, /*rt=*/2, /*sf=*/false, equal);
    emit_ldrb_imm(ctx, loop, /*rt=*/11, /*rn=*/9,  /*off=*/0);
    emit_ldrb_imm(ctx, loop, /*rt=*/12, /*rn=*/10, /*off=*/0);
    emit_cmp_reg(ctx, loop, /*rn=*/11, /*rm=*/12, /*sf=*/false);
    emit_b_cond(ctx, loop, COND_NE, diff);
    emit_add_imm(ctx, loop, /*rd=*/9,  /*rn=*/9,  /*imm12=*/1, /*sf=*/true);
    emit_add_imm(ctx, loop, /*rd=*/10, /*rn=*/10, /*imm12=*/1, /*sf=*/true);
    emit_sub_imm(ctx, loop, /*rd=*/2,  /*rn=*/2,  /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, loop, loop);

    emit_sub_reg(ctx, diff, /*rd=*/0, /*rn=*/11, /*rm=*/12, /*sf=*/false);
    emit_epilogue(ctx, diff, /*frame_size=*/16);
    emit_ret(ctx, diff);

    emit_movz(ctx, equal, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, equal, /*frame_size=*/16);
    emit_ret(ctx, equal);

    return synth_leaf_finish(ctx, region, "memcmp", 6);
}

// -----------------------------------------------------------------------------
// memchr(i32 p_ofs, i32 c, i32 n) -> i32 (offset or 0 == NULL).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_memchr(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, match, nomatch;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop    = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    match   = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, match);
    nomatch = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, region, nomatch);

    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    emit_b(ctx, entry, loop);

    emit_cbz(ctx, loop, /*rt=*/2, /*sf=*/false, nomatch);
    emit_ldrb_imm(ctx, loop, /*rt=*/11, /*rn=*/9, /*off=*/0);
    emit_cmp_reg(ctx, loop, /*rn=*/11, /*rm=*/1, /*sf=*/false);
    emit_b_cond(ctx, loop, COND_EQ, match);
    emit_add_imm(ctx, loop, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/true);
    emit_sub_imm(ctx, loop, /*rd=*/2, /*rn=*/2, /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, loop, loop);

    // Return offset = host - linmem_base, truncated to i32.
    emit_sub_reg(ctx, match, /*rd=*/0, /*rn=*/9, /*rm=*/28, /*sf=*/false);
    emit_epilogue(ctx, match, /*frame_size=*/16);
    emit_ret(ctx, match);

    emit_movz(ctx, nomatch, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, nomatch, /*frame_size=*/16);
    emit_ret(ctx, nomatch);

    return synth_leaf_finish(ctx, region, "memchr", 6);
}

// -----------------------------------------------------------------------------
// fd_write(i32 fd, i32 iovs_ofs, i32 iovs_len, i32 nwritten_ofs) -> i32 errno
//
// Iterates `iovs_len` (ptr, len) pairs starting at linmem[iovs_ofs] and
// issues a write(2) syscall per chunk. We do not currently write back to
// `nwritten_ofs` (no tinyC test cares about the byte count); returns 0
// on success.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_fd_write(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, done;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    done = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, done);

    // Stash fd in x19 (we clobber x0 for the syscall on each iteration).
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 19); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    // x9 = host(iovs)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // x20 = iovs_len
    emit_mov_x(ctx, entry, /*rd=*/20, /*rn=*/2);
    // x21 = total bytes written accumulator (must be zero so callers
    // — notably write_all in corec/base/io.c, which loops while
    // nwritten==0 — see the iovec-bytes back and make progress).
    emit_movz(ctx, entry, /*rd=*/21, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    // x22 = nwritten_ofs (w3): wasm linmem offset where the caller
    // expects us to deposit the total byte count. Stash now because we
    // clobber x3 with the iovec walk and want a stable copy at `done`.
    emit_mov_x(ctx, entry, /*rd=*/22, /*rn=*/3);
    emit_b(ctx, entry, loop);

    emit_cbz(ctx, loop, /*rt=*/20, /*sf=*/false, done);
    // w11 = ptr offset, w12 = len
    emit_ldr_w(ctx, loop, /*rt=*/11, /*rn=*/9, /*off=*/0);
    emit_ldr_w(ctx, loop, /*rt=*/12, /*rn=*/9, /*off=*/4);
    // x11 = host(ptr)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 11);
        MLIR_AppendBlockOp(ctx, loop,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, loop, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    // write(fd=x19, buf=x11, count=x12)
    emit_mov_x(ctx, loop, /*rd=*/0, /*rn=*/19);
    emit_mov_x(ctx, loop, /*rd=*/1, /*rn=*/11);
    emit_mov_x(ctx, loop, /*rd=*/2, /*rn=*/12);
    emit_bl(ctx, loop, str_lit("_write"));
    // Accumulate bytes-written. x0 holds ssize_t (negative on error). We
    // intentionally don't branch on error here — the simplest tinyc
    // pipelines all just call write to a known-good fd, and dropping a
    // negative return into the accumulator (instead of into the host
    // *nwritten) at least keeps the loop converging.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, loop, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, loop, /*rd=*/21, /*rn=*/21, /*rm=*/10, /*sf=*/true);
    // Advance: iovs += 8, len -= 1
    emit_add_imm(ctx, loop, /*rd=*/9,  /*rn=*/9,  /*imm12=*/8, /*sf=*/true);
    emit_sub_imm(ctx, loop, /*rd=*/20, /*rn=*/20, /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, loop, loop);

    // done: store total bytes written back to *nwritten_ptr (i32 on
    // wasm32, matching size_t there), then return errno=0.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 22);
        MLIR_AppendBlockOp(ctx, done, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, done, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_str_w(ctx, done, /*rt=*/21, /*rn=*/10, /*off=*/0);
    emit_movz(ctx, done, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, done, /*frame_size=*/16);
    emit_ret(ctx, done);

    return synth_leaf_finish(ctx, region, "fd_write", 8);
}

// -----------------------------------------------------------------------------
// WASI proc_exit(i32 exit_code) -> noreturn: bl _exit.
// Used by the wasm->wasmstack->wasmssa lifter pipeline: wasm-ld synthesises
// a `_start` that calls `__original_main` then `proc_exit`, and the lifter
// preserves that call. The C-frontend pipeline doesn't generate this call
// at all (the wmir backend's own synth_start does the libSystem call
// directly).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_proc_exit(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // exit_code already in x0/w0 per AAPCS; call libSystem _exit which
    // is noreturn. The bl never returns but we emit a clean epilogue +
    // ret for IR well-formedness.
    emit_bl(ctx, entry, str_lit("_exit"));
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "proc_exit", 9);
}

// -----------------------------------------------------------------------------
// tinyc_va_arg_i32(i32 ap_ofs) -> i32: read i32 at *ap, advance *ap by 4.
// `ap` is a wasm offset to a 4-byte cell that itself stores the current
// va_list cursor (a wasm offset to the next arg in linmem).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_tinyc_va_arg_i32(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // x9 = host(ap)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // w10 = *ap
    emit_ldr_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    // x11 = host(*ap)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 10);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    emit_ldr_w(ctx, entry, /*rt=*/0, /*rn=*/11, /*off=*/0);
    // *ap += 4
    emit_add_imm(ctx, entry, /*rd=*/10, /*rn=*/10, /*imm12=*/4, /*sf=*/false);
    emit_str_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "tinyc_va_arg_i32", 16);
}

// -----------------------------------------------------------------------------
// tinyc_va_arg_i64(i32 ap_ofs) -> i64: align *ap up to 8, read 8 bytes,
// advance *ap by 8.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_tinyc_va_arg_i64(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // x9 = host(ap)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // w10 = *ap; align up to 8.
    emit_ldr_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_add_imm(ctx, entry, /*rd=*/10, /*rn=*/10, /*imm12=*/7, /*sf=*/false);
    emit_movz(ctx, entry, /*rd=*/12, /*imm16=*/3, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, entry, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/10, /*rn=*/10, /*rm=*/12, /*sf=*/false);
    emit_3reg(ctx, entry, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/10, /*rn=*/10, /*rm=*/12, /*sf=*/false);
    // x11 = host(*ap)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 10);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    emit_ldr_x(ctx, entry, /*rt=*/0, /*rn=*/11, /*off=*/0);
    // *ap = aligned + 8
    emit_add_imm(ctx, entry, /*rd=*/10, /*rn=*/10, /*imm12=*/8, /*sf=*/false);
    emit_str_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "tinyc_va_arg_i64", 16);
}

// -----------------------------------------------------------------------------
// tinyc_va_arg_ptr(i32 ap_ofs) -> i32: same as i32 (wasm32 pointers are i32).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_tinyc_va_arg_ptr(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    emit_ldr_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 10);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    emit_ldr_w(ctx, entry, /*rt=*/0, /*rn=*/11, /*off=*/0);
    emit_add_imm(ctx, entry, /*rd=*/10, /*rn=*/10, /*imm12=*/4, /*sf=*/false);
    emit_str_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "tinyc_va_arg_ptr", 16);
}

// -----------------------------------------------------------------------------
// tinyc_va_arg_f64(i32 ap_ofs) -> i64: identical to tinyc_va_arg_i64 — the
// wmir carrier convention treats f64 as its i64 bit-pattern, and the
// va_list always stores f64 args as 8 raw bytes 8-byte-aligned.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_tinyc_va_arg_f64(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // x9 = host(ap)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // w10 = *ap; align up to 8.
    emit_ldr_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_add_imm(ctx, entry, /*rd=*/10, /*rn=*/10, /*imm12=*/7, /*sf=*/false);
    emit_movz(ctx, entry, /*rd=*/12, /*imm16=*/3, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, entry, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/10, /*rn=*/10, /*rm=*/12, /*sf=*/false);
    emit_3reg(ctx, entry, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/10, /*rn=*/10, /*rm=*/12, /*sf=*/false);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 10);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    emit_ldr_x(ctx, entry, /*rt=*/0, /*rn=*/11, /*off=*/0);
    emit_add_imm(ctx, entry, /*rd=*/10, /*rn=*/10, /*imm12=*/8, /*sf=*/false);
    emit_str_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "tinyc_va_arg_f64", 16);
}

// -----------------------------------------------------------------------------
// printF64(double v) -> void  (v arrives as the i64 bit-pattern in X0).
//
// Implements a fixed-decimal-style float formatter equivalent to
// runtime_wasm.c's fmt_f64 + wasm_write(buf,len). Output:
//   - "nan" if NaN
//   - "-<rest>" if negative
//   - "inf" if magnitude > 1e308
//   - "0" if magnitude exactly zero
//   - Otherwise 6 significant digits, trailing zeros trimmed, fixed
//     notation when -4 <= exp10 < 6, scientific notation otherwise.
//
// All cross-block state lives in callee-saved-style registers within
// this single function frame, since synth_print_f64 is a leaf except
// for the trailing write(2) syscall which doesn't clobber GPRs we use.
//
// Frame layout (96 bytes, 16-aligned):
//   [ 0..63 ] 64-byte ASCII output buffer (no NUL)
//   [64..71 ] 8-byte digit ring used during extraction
//
// Live registers (preserved across all internal blocks):
//   X13 = buf base   X14 = digits ptr
//   W9  = out        W10 = exp10 (signed)   W11 = ndig
//   W12 = neg (0/1)  D8  = m (current mantissa)
//   D9  = 10.0       D10 = 1.0              D11 = 0.0
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_print_f64(MLIR_Context *ctx) {
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle entry        = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bnan         = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bcheck_neg   = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bdo_neg      = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bcheck_inf   = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle binf         = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bcheck_zero  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bzero        = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bnorm_up_hd  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bnorm_up_bd  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bnorm_dn_hd  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bnorm_dn_bd  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bextract_hd  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bextract_bd  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle btrim_hd     = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle btrim_bd     = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfmt_select  = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_pos_hd= MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_pos_bd= MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_pos_dot = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_pos_fr_hd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_pos_fr_bd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_neg_lead_hd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_neg_lead_bd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_neg_digits_hd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bfixed_neg_digits_bd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bsci         = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bsci_frac_hd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bsci_frac_bd = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bsci_exp     = MLIR_CreateBlock(ctx);
    MLIR_BlockHandle bdo_write    = MLIR_CreateBlock(ctx);

    MLIR_AppendRegionBlock(ctx, region, entry);
    MLIR_AppendRegionBlock(ctx, region, bnan);
    MLIR_AppendRegionBlock(ctx, region, bcheck_neg);
    MLIR_AppendRegionBlock(ctx, region, bdo_neg);
    MLIR_AppendRegionBlock(ctx, region, bcheck_inf);
    MLIR_AppendRegionBlock(ctx, region, binf);
    MLIR_AppendRegionBlock(ctx, region, bcheck_zero);
    MLIR_AppendRegionBlock(ctx, region, bzero);
    MLIR_AppendRegionBlock(ctx, region, bnorm_up_hd);
    MLIR_AppendRegionBlock(ctx, region, bnorm_up_bd);
    MLIR_AppendRegionBlock(ctx, region, bnorm_dn_hd);
    MLIR_AppendRegionBlock(ctx, region, bnorm_dn_bd);
    MLIR_AppendRegionBlock(ctx, region, bextract_hd);
    MLIR_AppendRegionBlock(ctx, region, bextract_bd);
    MLIR_AppendRegionBlock(ctx, region, btrim_hd);
    MLIR_AppendRegionBlock(ctx, region, btrim_bd);
    MLIR_AppendRegionBlock(ctx, region, bfmt_select);
    MLIR_AppendRegionBlock(ctx, region, bfixed_pos_hd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_pos_bd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_pos_dot);
    MLIR_AppendRegionBlock(ctx, region, bfixed_pos_fr_hd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_pos_fr_bd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_neg_lead_hd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_neg_lead_bd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_neg_digits_hd);
    MLIR_AppendRegionBlock(ctx, region, bfixed_neg_digits_bd);
    MLIR_AppendRegionBlock(ctx, region, bsci);
    MLIR_AppendRegionBlock(ctx, region, bsci_frac_hd);
    MLIR_AppendRegionBlock(ctx, region, bsci_frac_bd);
    MLIR_AppendRegionBlock(ctx, region, bsci_exp);
    MLIR_AppendRegionBlock(ctx, region, bdo_write);

    emit_prologue(ctx, entry, /*frame_size=*/96);
    // D0 <- X0 (FP carrier convention: i64 bit-pattern in X0 is the
    // double we want to format).
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/true, /*sf=*/true,
                   /*rd=*/0, /*rn=*/0);
    // X13 = sp + 0 (buf base)
    emit_add_imm(ctx, entry, /*rd=*/13, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    // X14 = X13 + 64 (digits ptr)
    emit_add_imm(ctx, entry, /*rd=*/14, /*rn=*/13, /*imm12=*/64, /*sf=*/true);
    // W9 = 0 (out), W12 = 0 (neg)
    emit_movz(ctx, entry, /*rd=*/9, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_movz(ctx, entry, /*rd=*/12, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    // D9 = 10.0  (0x4024_0000_0000_0000)
    emit_movz(ctx, entry, /*rd=*/15, /*imm16=*/0x4024, /*hw=*/3, /*sf=*/true);
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/true, /*sf=*/true,
                   /*rd=*/9,  /*rn=*/15);
    // D10 = 1.0   (0x3FF0_0000_0000_0000)
    emit_movz(ctx, entry, /*rd=*/15, /*imm16=*/0x3FF0, /*hw=*/3, /*sf=*/true);
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/true, /*sf=*/true,
                   /*rd=*/10, /*rn=*/15);
    // D11 = 0.0
    emit_movz(ctx, entry, /*rd=*/15, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/true, /*sf=*/true,
                   /*rd=*/11, /*rn=*/15);
    // D8 = D0 (working copy)
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/false, /*sf=*/true,
                   /*rd=*/15, /*rn=*/0);
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/true,  /*sf=*/true,
                   /*rd=*/8,  /*rn=*/15);
    // NaN check: FCMP D0,D0; B.NE bnan
    emit_fcmp(ctx, entry, /*fwidth=*/64, /*rn=*/0, /*rm=*/0);
    emit_b_cond(ctx, entry, COND_NE, bnan);
    emit_b(ctx, entry, bcheck_neg);

    // ---- bnan: write "nan" (3 bytes) directly at buf[0..2], out=3
    emit_movz(ctx, bnan, /*rd=*/15, /*imm16=*/'n', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bnan, /*rt=*/15, /*rn=*/13, /*off=*/0);
    emit_movz(ctx, bnan, /*rd=*/15, /*imm16=*/'a', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bnan, /*rt=*/15, /*rn=*/13, /*off=*/1);
    emit_movz(ctx, bnan, /*rd=*/15, /*imm16=*/'n', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bnan, /*rt=*/15, /*rn=*/13, /*off=*/2);
    emit_movz(ctx, bnan, /*rd=*/9,  /*imm16=*/3, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, bnan, bdo_write);

    // ---- bcheck_neg: if D8 < 0 -> write '-', D8 = -D8, neg=1, out=1
    emit_fcmp(ctx, bcheck_neg, /*fwidth=*/64, /*rn=*/8, /*rm=*/11);
    emit_b_cond(ctx, bcheck_neg, COND_MI, bdo_neg);
    emit_b(ctx, bcheck_neg, bcheck_inf);

    // ---- bdo_neg
    emit_movz(ctx, bdo_neg, /*rd=*/15, /*imm16=*/'-', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bdo_neg, /*rt=*/15, /*rn=*/13, /*off=*/0);
    emit_movz(ctx, bdo_neg, /*rd=*/9,  /*imm16=*/1, /*hw=*/0, /*sf=*/false);
    emit_movz(ctx, bdo_neg, /*rd=*/12, /*imm16=*/1, /*hw=*/0, /*sf=*/false);
    emit_fp_unop(ctx, bdo_neg, "fneg", /*fwidth=*/64, /*rd=*/8, /*rn=*/8);
    emit_b(ctx, bdo_neg, bcheck_inf);

    // ---- bcheck_inf: load 1e308 into D1, compare. D8 > 1e308 -> binf
    //   1e308 ~= 0x7FE1CCF385EBC8A0 (more precisely 1.0e308 = 0x7FE1CCF385EBC8A0)
    //   We'll build that constant via two MOVZ + two MOVK into X15 then FMOV.
    emit_movz(ctx, bcheck_inf, /*rd=*/15, /*imm16=*/0xc8a0, /*hw=*/0, /*sf=*/true);
    emit_movk(ctx, bcheck_inf, /*rd=*/15, /*imm16=*/0x85eb, /*hw=*/1, /*sf=*/true);
    emit_movk(ctx, bcheck_inf, /*rd=*/15, /*imm16=*/0xccf3, /*hw=*/2, /*sf=*/true);
    emit_movk(ctx, bcheck_inf, /*rd=*/15, /*imm16=*/0x7fe1, /*hw=*/3, /*sf=*/true);
    emit_fmov_gp_v(ctx, bcheck_inf, /*dir_to_v=*/true, /*sf=*/true,
                   /*rd=*/1, /*rn=*/15);
    emit_fcmp(ctx, bcheck_inf, /*fwidth=*/64, /*rn=*/8, /*rm=*/1);
    emit_b_cond(ctx, bcheck_inf, COND_GT, binf);
    emit_b(ctx, bcheck_inf, bcheck_zero);

    // ---- binf: write "inf" at buf[out..out+2], out += 3
    //  X15 = X13 + W9 (host write head)
    {
        // sxtw not needed: out fits in low 32 bits and W9 is sign-extendable.
        // Use add Xd, Xn, Wm UXTW: but our helpers don't expose that. Use a
        // simple ADD with X9 (we treat w9 as a 64-bit byte offset; out is
        // small so the upper bits are zero from previous movz/add_imm).
        emit_add_reg(ctx, binf, /*rd=*/15, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    }
    emit_movz(ctx, binf, /*rd=*/16, /*imm16=*/'i', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, binf, /*rt=*/16, /*rn=*/15, /*off=*/0);
    emit_movz(ctx, binf, /*rd=*/16, /*imm16=*/'n', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, binf, /*rt=*/16, /*rn=*/15, /*off=*/1);
    emit_movz(ctx, binf, /*rd=*/16, /*imm16=*/'f', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, binf, /*rt=*/16, /*rn=*/15, /*off=*/2);
    emit_add_imm(ctx, binf, /*rd=*/9, /*rn=*/9, /*imm12=*/3, /*sf=*/false);
    emit_b(ctx, binf, bdo_write);

    // ---- bcheck_zero: FCMP D8, D11; B.EQ bzero
    emit_fcmp(ctx, bcheck_zero, /*fwidth=*/64, /*rn=*/8, /*rm=*/11);
    emit_b_cond(ctx, bcheck_zero, COND_EQ, bzero);
    emit_b(ctx, bcheck_zero, bnorm_up_hd);

    // ---- bzero: write '0' at buf[out], out += 1
    emit_add_reg(ctx, bzero, /*rd=*/15, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_movz(ctx, bzero, /*rd=*/16, /*imm16=*/'0', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bzero, /*rt=*/16, /*rn=*/15, /*off=*/0);
    emit_add_imm(ctx, bzero, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bzero, bdo_write);

    // ---- normalize up: while D8 >= 10: D8 /= 10; exp10++
    emit_movz(ctx, bnorm_up_hd, /*rd=*/10, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, bnorm_up_hd, bnorm_up_bd);
    // body: if D8 < 10 -> goto norm_dn_hd else divide
    emit_fcmp(ctx, bnorm_up_bd, /*fwidth=*/64, /*rn=*/8, /*rm=*/9);
    emit_b_cond(ctx, bnorm_up_bd, COND_MI, bnorm_dn_hd);
    emit_fp_binop(ctx, bnorm_up_bd, "fdiv", 64, /*rd=*/8, /*rn=*/8, /*rm=*/9);
    emit_add_imm(ctx, bnorm_up_bd, /*rd=*/10, /*rn=*/10,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bnorm_up_bd, bnorm_up_bd);

    // ---- normalize down: while D8 < 1: D8 *= 10; exp10--
    emit_b(ctx, bnorm_dn_hd, bnorm_dn_bd);
    emit_fcmp(ctx, bnorm_dn_bd, /*fwidth=*/64, /*rn=*/8, /*rm=*/10);
    // if !(D8 < 1), exit. B.GE bextract_hd
    emit_b_cond(ctx, bnorm_dn_bd, COND_GE, bextract_hd);
    emit_fp_binop(ctx, bnorm_dn_bd, "fmul", 64, /*rd=*/8, /*rn=*/8, /*rm=*/9);
    emit_sub_imm(ctx, bnorm_dn_bd, /*rd=*/10, /*rn=*/10,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bnorm_dn_bd, bnorm_dn_bd);

    // ---- extract 6 digits: for i in 0..5
    //   d = (int)D8 (clamp 0..9)
    //   digits[i] = '0' + d
    //   D8 = (D8 - d) * 10
    // i counter = X15 (scratch within block; but we need cross-block too).
    // Use W11 for the counter (we'll reset W11 below to ndig).
    emit_movz(ctx, bextract_hd, /*rd=*/11, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, bextract_hd, bextract_bd);
    // loop body
    // if W11 == 6 -> btrim_hd
    {
        emit_cmp_imm(ctx, bextract_bd, /*rn=*/11, /*imm12=*/6, /*sf=*/false);
        emit_b_cond(ctx, bextract_bd, COND_GE, btrim_hd);
    }
    // d = fcvtzs Wd, D8 (truncate toward zero); use V1 as int holder
    emit_fp_cvt(ctx, bextract_bd, "f2i", /*src_w=*/64, /*dst_w=*/32,
                /*sign=*/true, /*rd=*/15, /*rn=*/8);
    // clamp: d = max(0, min(9, d))
    emit_cmp_imm(ctx, bextract_bd, /*rn=*/15, /*imm12=*/9, /*sf=*/false);
    // csel d, d, #9 if d > 9
    {
        // Use cset/csel; simpler: if d > 9, w15 = 9. Synthesize via
        // movz w16, #9; csel w15, w16, w15, gt.
        emit_movz(ctx, bextract_bd, /*rd=*/16, /*imm16=*/9, /*hw=*/0, /*sf=*/false);
        emit_csel(ctx, bextract_bd, /*rd=*/15, /*rn=*/16, /*rm=*/15,
                  /*cond=*/COND_GT, /*sf=*/false);
    }
    emit_cmp_imm(ctx, bextract_bd, /*rn=*/15, /*imm12=*/0, /*sf=*/false);
    {
        emit_movz(ctx, bextract_bd, /*rd=*/16, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
        emit_csel(ctx, bextract_bd, /*rd=*/15, /*rn=*/16, /*rm=*/15,
                  /*cond=*/COND_LT, /*sf=*/false);
    }
    // digits[i] = '0' + d
    emit_add_imm(ctx, bextract_bd, /*rd=*/16, /*rn=*/15,
                 /*imm12=*/'0', /*sf=*/false);
    // digit_addr = X14 + W11
    emit_add_reg(ctx, bextract_bd, /*rd=*/0, /*rn=*/14, /*rm=*/11, /*sf=*/true);
    emit_strb_imm(ctx, bextract_bd, /*rt=*/16, /*rn=*/0, /*off=*/0);
    // D2 = (double)d
    emit_fp_cvt(ctx, bextract_bd, "i2f", /*src_w=*/32, /*dst_w=*/64,
                /*sign=*/true, /*rd=*/2, /*rn=*/15);
    // D8 = (D8 - D2) * 10
    emit_fp_binop(ctx, bextract_bd, "fsub", 64, /*rd=*/8, /*rn=*/8, /*rm=*/2);
    emit_fp_binop(ctx, bextract_bd, "fmul", 64, /*rd=*/8, /*rn=*/8, /*rm=*/9);
    emit_add_imm(ctx, bextract_bd, /*rd=*/11, /*rn=*/11,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bextract_bd, bextract_bd);

    // ---- trim trailing zeros: ndig = 6; while ndig > 1 && digits[ndig-1] == '0': ndig--
    emit_movz(ctx, btrim_hd, /*rd=*/11, /*imm16=*/6, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, btrim_hd, btrim_bd);
    // body
    emit_cmp_imm(ctx, btrim_bd, /*rn=*/11, /*imm12=*/1, /*sf=*/false);
    emit_b_cond(ctx, btrim_bd, COND_LE, bfmt_select);
    // x15 = X14 + (W11 - 1)
    emit_sub_imm(ctx, btrim_bd, /*rd=*/15, /*rn=*/11,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_reg(ctx, btrim_bd, /*rd=*/15, /*rn=*/14, /*rm=*/15, /*sf=*/true);
    emit_ldrb_imm(ctx, btrim_bd, /*rt=*/16, /*rn=*/15, /*off=*/0);
    emit_cmp_imm(ctx, btrim_bd, /*rn=*/16, /*imm12=*/'0', /*sf=*/false);
    emit_b_cond(ctx, btrim_bd, COND_NE, bfmt_select);
    emit_sub_imm(ctx, btrim_bd, /*rd=*/11, /*rn=*/11,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, btrim_bd, btrim_bd);

    // ---- fmt_select: if exp10 in [-4, 6): fixed; else scientific
    // exp10 is signed in W10. We need to test:
    //   if (exp10 >= -4 && exp10 < 6).
    emit_cmp_imm(ctx, bfmt_select, /*rn=*/10, /*imm12=*/6, /*sf=*/false);
    emit_b_cond(ctx, bfmt_select, COND_GE, bsci);
    // exp10 < 6 — now check exp10 >= -4 i.e. exp10 >= 0xFFFFFFFC (signed -4).
    // We do: cmn w10, #4 (i.e. cmp w10, #-4 with reverse). Lacking cmn,
    // use add w15, w10, #4; cmp w15, #0; b.lt sci.
    emit_add_imm(ctx, bfmt_select, /*rd=*/15, /*rn=*/10,
                 /*imm12=*/4, /*sf=*/false);
    emit_cmp_imm(ctx, bfmt_select, /*rn=*/15, /*imm12=*/0, /*sf=*/false);
    emit_b_cond(ctx, bfmt_select, COND_LT, bsci);
    // exp10 >= -4 and exp10 < 6 - choose fixed-pos or fixed-neg.
    emit_cmp_imm(ctx, bfmt_select, /*rn=*/10, /*imm12=*/0, /*sf=*/false);
    emit_b_cond(ctx, bfmt_select, COND_LT, bfixed_neg_lead_hd);
    emit_b(ctx, bfmt_select, bfixed_pos_hd);

    // ---- fixed_pos: int_digits = exp10 + 1
    //   for i in 0..int_digits: buf[out++] = (i < ndig) ? digits[i] : '0'
    //   if (int_digits < ndig): buf[out++] = '.'; for i in int_digits..ndig: buf[out++] = digits[i]
    // We use X15 as i counter, X16 (caller-saved) for tmp.
    emit_add_imm(ctx, bfixed_pos_hd, /*rd=*/0, /*rn=*/10,
                 /*imm12=*/1, /*sf=*/false); // X0 = int_digits
    emit_movz(ctx, bfixed_pos_hd, /*rd=*/15, /*imm16=*/0, /*hw=*/0, /*sf=*/false); // i = 0
    emit_b(ctx, bfixed_pos_hd, bfixed_pos_bd);

    emit_cmp_reg(ctx, bfixed_pos_bd, /*rn=*/15, /*rm=*/0, /*sf=*/false);
    emit_b_cond(ctx, bfixed_pos_bd, COND_GE, bfixed_pos_dot);
    // ch = (i < ndig) ? digits[i] : '0'
    emit_cmp_reg(ctx, bfixed_pos_bd, /*rn=*/15, /*rm=*/11, /*sf=*/false);
    // load digits[i] into w16; '0' into w17; csel
    emit_add_reg(ctx, bfixed_pos_bd, /*rd=*/2, /*rn=*/14, /*rm=*/15, /*sf=*/true);
    emit_ldrb_imm(ctx, bfixed_pos_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_movz(ctx, bfixed_pos_bd, /*rd=*/17, /*imm16=*/'0', /*hw=*/0, /*sf=*/false);
    emit_csel(ctx, bfixed_pos_bd, /*rd=*/16, /*rn=*/16, /*rm=*/17,
              /*cond=*/COND_LT, /*sf=*/false);
    // buf[out] = ch; out++
    emit_add_reg(ctx, bfixed_pos_bd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_strb_imm(ctx, bfixed_pos_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_pos_bd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_imm(ctx, bfixed_pos_bd, /*rd=*/15, /*rn=*/15,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bfixed_pos_bd, bfixed_pos_bd);

    // dot: if (int_digits >= ndig) -> bdo_write
    emit_cmp_reg(ctx, bfixed_pos_dot, /*rn=*/0, /*rm=*/11, /*sf=*/false);
    emit_b_cond(ctx, bfixed_pos_dot, COND_GE, bdo_write);
    // emit '.'
    emit_movz(ctx, bfixed_pos_dot, /*rd=*/16, /*imm16=*/'.', /*hw=*/0, /*sf=*/false);
    emit_add_reg(ctx, bfixed_pos_dot, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_strb_imm(ctx, bfixed_pos_dot, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_pos_dot, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    // i = int_digits (= X0)
    emit_mov_x(ctx, bfixed_pos_dot, /*rd=*/15, /*rn=*/0);
    emit_b(ctx, bfixed_pos_dot, bfixed_pos_fr_hd);

    emit_b(ctx, bfixed_pos_fr_hd, bfixed_pos_fr_bd);
    emit_cmp_reg(ctx, bfixed_pos_fr_bd, /*rn=*/15, /*rm=*/11, /*sf=*/false);
    emit_b_cond(ctx, bfixed_pos_fr_bd, COND_GE, bdo_write);
    emit_add_reg(ctx, bfixed_pos_fr_bd, /*rd=*/2, /*rn=*/14, /*rm=*/15, /*sf=*/true);
    emit_ldrb_imm(ctx, bfixed_pos_fr_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_reg(ctx, bfixed_pos_fr_bd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_strb_imm(ctx, bfixed_pos_fr_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_pos_fr_bd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_imm(ctx, bfixed_pos_fr_bd, /*rd=*/15, /*rn=*/15,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bfixed_pos_fr_bd, bfixed_pos_fr_bd);

    // ---- fixed_neg: buf[out++]='0'; buf[out++]='.';
    //                  for i in 0..(-exp10-1): buf[out++]='0';
    //                  for i in 0..ndig:       buf[out++]=digits[i];
    emit_add_reg(ctx, bfixed_neg_lead_hd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_movz(ctx, bfixed_neg_lead_hd, /*rd=*/16, /*imm16=*/'0', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bfixed_neg_lead_hd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_neg_lead_hd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_reg(ctx, bfixed_neg_lead_hd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_movz(ctx, bfixed_neg_lead_hd, /*rd=*/16, /*imm16=*/'.', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bfixed_neg_lead_hd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_neg_lead_hd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    // i_max = -exp10 - 1.  W15 used as i.
    // W0 = (-exp10) - 1 = -(exp10+1).
    emit_movz(ctx, bfixed_neg_lead_hd, /*rd=*/15, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    // w0 = 0 - exp10 - 1 = -(exp10 + 1)
    emit_add_imm(ctx, bfixed_neg_lead_hd, /*rd=*/0, /*rn=*/10,
                 /*imm12=*/1, /*sf=*/false);  // w0 = exp10 + 1 (signed)
    // negate w0: sub w0, wzr, w0 (rn=31 = wzr)
    {
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 0);
        a[1] = attr_i32(ctx, "rn", 31);
        a[2] = attr_i32(ctx, "rm", 0);
        a[3] = attr_b  (ctx, "sf", false);
        MLIR_AppendBlockOp(ctx, bfixed_neg_lead_hd,
            build_op(ctx, OP_TYPE_AARCH64_SUB_REG, a, 4));
    }
    emit_b(ctx, bfixed_neg_lead_hd, bfixed_neg_lead_bd);

    emit_cmp_reg(ctx, bfixed_neg_lead_bd, /*rn=*/15, /*rm=*/0, /*sf=*/false);
    emit_b_cond(ctx, bfixed_neg_lead_bd, COND_GE, bfixed_neg_digits_hd);
    emit_add_reg(ctx, bfixed_neg_lead_bd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_movz(ctx, bfixed_neg_lead_bd, /*rd=*/16, /*imm16=*/'0', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bfixed_neg_lead_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_neg_lead_bd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_imm(ctx, bfixed_neg_lead_bd, /*rd=*/15, /*rn=*/15,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bfixed_neg_lead_bd, bfixed_neg_lead_bd);

    // digits loop: for i in 0..ndig: buf[out++] = digits[i]
    emit_movz(ctx, bfixed_neg_digits_hd, /*rd=*/15, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, bfixed_neg_digits_hd, bfixed_neg_digits_bd);

    emit_cmp_reg(ctx, bfixed_neg_digits_bd, /*rn=*/15, /*rm=*/11, /*sf=*/false);
    emit_b_cond(ctx, bfixed_neg_digits_bd, COND_GE, bdo_write);
    emit_add_reg(ctx, bfixed_neg_digits_bd, /*rd=*/2, /*rn=*/14, /*rm=*/15, /*sf=*/true);
    emit_ldrb_imm(ctx, bfixed_neg_digits_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_reg(ctx, bfixed_neg_digits_bd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_strb_imm(ctx, bfixed_neg_digits_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bfixed_neg_digits_bd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_imm(ctx, bfixed_neg_digits_bd, /*rd=*/15, /*rn=*/15,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bfixed_neg_digits_bd, bfixed_neg_digits_bd);

    // ---- scientific: buf[out++] = digits[0]; if ndig>1: buf[out++]='.', then digits[1..ndig]
    //   buf[out++]='e'; sign; |exp10| in decimal (>=2 chars)
    emit_add_reg(ctx, bsci, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_ldrb_imm(ctx, bsci, /*rt=*/16, /*rn=*/14, /*off=*/0);
    emit_strb_imm(ctx, bsci, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bsci, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
    emit_cmp_imm(ctx, bsci, /*rn=*/11, /*imm12=*/1, /*sf=*/false);
    emit_b_cond(ctx, bsci, COND_LE, bsci_exp);
    // emit dot
    emit_add_reg(ctx, bsci, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_movz(ctx, bsci, /*rd=*/16, /*imm16=*/'.', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bsci, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bsci, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
    emit_movz(ctx, bsci, /*rd=*/15, /*imm16=*/1, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, bsci, bsci_frac_hd);

    emit_b(ctx, bsci_frac_hd, bsci_frac_bd);
    emit_cmp_reg(ctx, bsci_frac_bd, /*rn=*/15, /*rm=*/11, /*sf=*/false);
    emit_b_cond(ctx, bsci_frac_bd, COND_GE, bsci_exp);
    emit_add_reg(ctx, bsci_frac_bd, /*rd=*/2, /*rn=*/14, /*rm=*/15, /*sf=*/true);
    emit_ldrb_imm(ctx, bsci_frac_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_reg(ctx, bsci_frac_bd, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_strb_imm(ctx, bsci_frac_bd, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bsci_frac_bd, /*rd=*/9, /*rn=*/9,
                 /*imm12=*/1, /*sf=*/false);
    emit_add_imm(ctx, bsci_frac_bd, /*rd=*/15, /*rn=*/15,
                 /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, bsci_frac_bd, bsci_frac_bd);

    // bsci_exp: emit 'e', sign, 2-digit (or more) decimal of |exp10|
    emit_add_reg(ctx, bsci_exp, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_movz(ctx, bsci_exp, /*rd=*/16, /*imm16=*/'e', /*hw=*/0, /*sf=*/false);
    emit_strb_imm(ctx, bsci_exp, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bsci_exp, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
    emit_cmp_imm(ctx, bsci_exp, /*rn=*/10, /*imm12=*/0, /*sf=*/false);
    // We'll emit sign char then digits. Negate exp10 if < 0.
    {
        // w16 = (exp10 < 0) ? '-' : '+'
        emit_movz(ctx, bsci_exp, /*rd=*/16, /*imm16=*/'+', /*hw=*/0, /*sf=*/false);
        emit_movz(ctx, bsci_exp, /*rd=*/17, /*imm16=*/'-', /*hw=*/0, /*sf=*/false);
        emit_csel(ctx, bsci_exp, /*rd=*/16, /*rn=*/17, /*rm=*/16,
                  /*cond=*/COND_LT, /*sf=*/false);
    }
    emit_add_reg(ctx, bsci_exp, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
    emit_strb_imm(ctx, bsci_exp, /*rt=*/16, /*rn=*/2, /*off=*/0);
    emit_add_imm(ctx, bsci_exp, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
    // w0 = |exp10|: if exp10 < 0 then -exp10 else exp10
    {
        // w15 = -exp10  (sub w15, wzr, w10)
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 15);
        a[1] = attr_i32(ctx, "rn", 31);
        a[2] = attr_i32(ctx, "rm", 10);
        a[3] = attr_b  (ctx, "sf", false);
        MLIR_AppendBlockOp(ctx, bsci_exp,
            build_op(ctx, OP_TYPE_AARCH64_SUB_REG, a, 4));
        emit_cmp_imm(ctx, bsci_exp, /*rn=*/10, /*imm12=*/0, /*sf=*/false);
        emit_csel(ctx, bsci_exp, /*rd=*/0, /*rn=*/15, /*rm=*/10,
                  /*cond=*/COND_LT, /*sf=*/false);
    }
    // Emit 2 digits min: tens then ones (exp10 magnitude <= 308 fits in 3 chars,
    // but we just emit modulo-10 always until done, then pad to 2).
    //   ones = w0 % 10; tens = w0 / 10; if tens > 0 emit (tens digits then ones); else emit "0<ones>"
    // Simple: hundreds = w0/100; tens = (w0/10)%10; ones = w0%10.
    //   if hundreds != 0 emit it; always emit tens and ones.
    emit_movz(ctx, bsci_exp, /*rd=*/1, /*imm16=*/10, /*hw=*/0, /*sf=*/false);
    // w15 = w0/10
    {
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 15);
        a[1] = attr_i32(ctx, "rn", 0);
        a[2] = attr_i32(ctx, "rm", 1);
        a[3] = attr_b  (ctx, "sf", false);
        MLIR_AppendBlockOp(ctx, bsci_exp,
            build_op(ctx, OP_TYPE_AARCH64_UDIV, a, 4));
    }
    // w16 = w0 - w15*10 = w0 % 10
    emit_msub(ctx, bsci_exp, /*rd=*/16, /*rn=*/15, /*rm=*/1,
              /*ra=*/0, /*sf=*/false);
    // w17 = w15/10  (hundreds)
    {
        MLIR_AttributeHandle a[4];
        a[0] = attr_i32(ctx, "rd", 17);
        a[1] = attr_i32(ctx, "rn", 15);
        a[2] = attr_i32(ctx, "rm", 1);
        a[3] = attr_b  (ctx, "sf", false);
        MLIR_AppendBlockOp(ctx, bsci_exp,
            build_op(ctx, OP_TYPE_AARCH64_UDIV, a, 4));
    }
    // w18 = w15 - w17*10 = tens
    emit_msub(ctx, bsci_exp, /*rd=*/18, /*rn=*/17, /*rm=*/1,
              /*ra=*/15, /*sf=*/false);
    // Emit hundreds if non-zero
    emit_cmp_imm(ctx, bsci_exp, /*rn=*/17, /*imm12=*/0, /*sf=*/false);
    {
        MLIR_BlockHandle skip_h = MLIR_CreateBlock(ctx);
        MLIR_BlockHandle do_h   = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, region, skip_h);
        MLIR_AppendRegionBlock(ctx, region, do_h);
        emit_b_cond(ctx, bsci_exp, COND_EQ, skip_h);
        emit_b(ctx, bsci_exp, do_h);

        // do_h: emit hundreds digit
        emit_add_imm(ctx, do_h, /*rd=*/16, /*rn=*/17,
                     /*imm12=*/'0', /*sf=*/false);
        emit_add_reg(ctx, do_h, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
        emit_strb_imm(ctx, do_h, /*rt=*/16, /*rn=*/2, /*off=*/0);
        emit_add_imm(ctx, do_h, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
        emit_b(ctx, do_h, skip_h);

        // skip_h: emit tens
        emit_add_imm(ctx, skip_h, /*rd=*/15, /*rn=*/18,
                     /*imm12=*/'0', /*sf=*/false);
        emit_add_reg(ctx, skip_h, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
        emit_strb_imm(ctx, skip_h, /*rt=*/15, /*rn=*/2, /*off=*/0);
        emit_add_imm(ctx, skip_h, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
        // emit ones (still in w0%10 which we put in w16-original; reload via msub)
        // We've clobbered w16. Re-derive ones from w0 % 10:
        emit_movz(ctx, skip_h, /*rd=*/1, /*imm16=*/10, /*hw=*/0, /*sf=*/false);
        {
            MLIR_AttributeHandle a[4];
            a[0] = attr_i32(ctx, "rd", 15);
            a[1] = attr_i32(ctx, "rn", 0);
            a[2] = attr_i32(ctx, "rm", 1);
            a[3] = attr_b  (ctx, "sf", false);
            MLIR_AppendBlockOp(ctx, skip_h,
                build_op(ctx, OP_TYPE_AARCH64_UDIV, a, 4));
        }
        emit_msub(ctx, skip_h, /*rd=*/16, /*rn=*/15, /*rm=*/1,
                  /*ra=*/0, /*sf=*/false);
        emit_add_imm(ctx, skip_h, /*rd=*/16, /*rn=*/16,
                     /*imm12=*/'0', /*sf=*/false);
        emit_add_reg(ctx, skip_h, /*rd=*/2, /*rn=*/13, /*rm=*/9, /*sf=*/true);
        emit_strb_imm(ctx, skip_h, /*rt=*/16, /*rn=*/2, /*off=*/0);
        emit_add_imm(ctx, skip_h, /*rd=*/9, /*rn=*/9, /*imm12=*/1, /*sf=*/false);
        emit_b(ctx, skip_h, bdo_write);
    }

    // ---- bdo_write: write(1, buf=X13, W9 bytes), epilogue, ret
    emit_movz(ctx, bdo_write, /*rd=*/0, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_mov_x(ctx, bdo_write, /*rd=*/1, /*rn=*/13);
    emit_mov_x(ctx, bdo_write, /*rd=*/2, /*rn=*/9);
    emit_bl(ctx, bdo_write, str_lit("_write"));
    emit_epilogue(ctx, bdo_write, /*frame_size=*/96);
    emit_ret(ctx, bdo_write);

    return synth_leaf_finish(ctx, region, "printF64", 8);
}

// -----------------------------------------------------------------------------
// printF32(float v) -> void  (v arrives as the i32 bit-pattern in W0).
// Convert to double, then tail-call printF64.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_print_f32(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // S0 <- W0 (32-bit GP->V)
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/true, /*sf=*/false,
                   /*rd=*/0, /*rn=*/0);
    // FCVT D0, S0
    emit_fp_cvt(ctx, entry, "f2f", /*src_w=*/32, /*dst_w=*/64,
                /*sign=*/false, /*rd=*/0, /*rn=*/0);
    // X0 <- D0 (carrier for the call to printF64)
    emit_fmov_gp_v(ctx, entry, /*dir_to_v=*/false, /*sf=*/true,
                   /*rd=*/0, /*rn=*/0);
    emit_bl(ctx, entry, str_from_cstr_view("printF64"));
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "printF32", 8);
}


// -----------------------------------------------------------------------------
// tinyc_va_arg_struct(i32 ap_ofs, i32 out_ofs, i64 size) -> void:
//   long long *o = (long long *)out;
//   long long words = (size + 7) / 8;
//   for (long long i = 0; i < words; i++) o[i] = va_arg(*ap, long long);
//
// ap_ofs in W0, out_ofs in W1, size in X2.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_tinyc_va_arg_struct(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, done;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    done = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, done);

    // x9 = host(ap_ofs)  (pointer to the i32 cursor stored in linmem)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // x11 = host(out_ofs)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    // x12 = words = (size + 7) >> 3
    emit_add_imm(ctx, entry, /*rd=*/12, /*rn=*/2, /*imm12=*/7, /*sf=*/true);
    emit_movz(ctx, entry, /*rd=*/13, /*imm16=*/3, /*hw=*/0, /*sf=*/true);
    emit_3reg(ctx, entry, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/12, /*rn=*/12, /*rm=*/13, /*sf=*/true);
    // w10 = *ap (i32 cursor)
    emit_ldr_w(ctx, entry, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_b(ctx, entry, loop);

    // loop_top: if x12 == 0 -> done; else copy one word, advance.
    emit_cbz(ctx, loop, /*rt=*/12, /*sf=*/true, done);
    // Align w10 up to 8: w10 = (w10 + 7) & ~7.
    emit_add_imm(ctx, loop, /*rd=*/10, /*rn=*/10, /*imm12=*/7, /*sf=*/false);
    emit_movz(ctx, loop, /*rd=*/14, /*imm16=*/3, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, loop, OP_TYPE_AARCH64_LSR_REG,
              /*rd=*/10, /*rn=*/10, /*rm=*/14, /*sf=*/false);
    emit_3reg(ctx, loop, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/10, /*rn=*/10, /*rm=*/14, /*sf=*/false);
    // x13 = host(cursor) = x28 + zext(w10)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 13); a[1] = attr_i32(ctx, "rn", 10);
        MLIR_AppendBlockOp(ctx, loop,
            build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, loop, /*rd=*/13, /*rn=*/28, /*rm=*/13, /*sf=*/true);
    // x15 = *cursor;  *out = x15
    emit_ldr_x(ctx, loop, /*rt=*/15, /*rn=*/13, /*off=*/0);
    emit_str_x(ctx, loop, /*rt=*/15, /*rn=*/11, /*off=*/0);
    // cursor += 8; out += 8; words -= 1
    emit_add_imm(ctx, loop, /*rd=*/10, /*rn=*/10, /*imm12=*/8, /*sf=*/false);
    emit_add_imm(ctx, loop, /*rd=*/11, /*rn=*/11, /*imm12=*/8, /*sf=*/true);
    emit_sub_imm(ctx, loop, /*rd=*/12, /*rn=*/12, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, loop, loop);

    // done: write back cursor; return.
    emit_str_w(ctx, done, /*rt=*/10, /*rn=*/9, /*off=*/0);
    emit_epilogue(ctx, done, /*frame_size=*/16);
    emit_ret(ctx, done);

    return synth_leaf_finish(ctx, region, "tinyc_va_arg_struct", 19);
}

// =============================================================================
// WASI host imports for the wmir Mach-O backend.
//
// These shims translate the WASI calling convention into Mach BSD syscalls
// (svc #0x80). All shims:
//   - take wasm linmem offsets (i32) as pointer arguments, translated to
//     host pointers via uxtw + add with x28 (linmem base);
//   - return 0 on success and a non-zero WASI errno value on failure
//     (we use ENOENT=44 or generic 8 for IO/EBADF — close enough for
//     tinyc which mostly cares about ==0 / !=0);
//   - never longjmp / never propagate errno globally.
//
// argc/argv source: synth_start stashes argc into linmem[ARGC_SLOT] and the
// kernel argv vector base pointer into linmem[ARGV_SLOT]. The args_get /
// args_sizes_get shims read those back.
// =============================================================================

// -----------------------------------------------------------------------------
// args_sizes_get(out_argc_ofs, out_buf_size_ofs) -> errno
// Reads argc from linmem[ARGC_SLOT] and walks the host argv vector to total
// up the byte size of every argv string (including nul terminator).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_args_sizes_get(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, outer, inner, advance, done;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    outer   = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, outer);
    inner   = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, inner);
    advance = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, advance);
    done    = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, done);

    // x9 = argc; x10 = host argv base; x11 = host(out_argc); x12 = host(out_buf_size).
    emit_ldr_w(ctx, entry, /*rt=*/9,  /*rn=*/28, /*off=*/ARGC_SLOT);
    emit_ldr_x(ctx, entry, /*rt=*/10, /*rn=*/28, /*off=*/ARGV_SLOT);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 12); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/12, /*rn=*/28, /*rm=*/12, /*sf=*/true);
    // *out_argc = argc (i32).
    emit_str_w(ctx, entry, /*rt=*/9, /*rn=*/11, /*off=*/0);
    // x13 = total accumulator; x14 = i.
    emit_movz(ctx, entry, /*rd=*/13, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_movz(ctx, entry, /*rd=*/14, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_b(ctx, entry, outer);

    // outer: while (i < argc) { scan argv[i]; }
    emit_cmp_reg(ctx, outer, /*rn=*/14, /*rm=*/9, /*sf=*/true);
    emit_b_cond(ctx, outer, COND_GE, done);
    // x15 = host argv[i] = *(argv_base + i*8).
    emit_movz(ctx, outer, /*rd=*/0, /*imm16=*/3, /*hw=*/0, /*sf=*/true);
    emit_3reg(ctx, outer, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/15, /*rn=*/14, /*rm=*/0, /*sf=*/true);
    emit_add_reg(ctx, outer, /*rd=*/15, /*rn=*/10, /*rm=*/15, /*sf=*/true);
    emit_ldr_x(ctx, outer, /*rt=*/15, /*rn=*/15, /*off=*/0);
    emit_b(ctx, outer, inner);

    // inner: while (*x15) { x13++; x15++; }; x13++ (nul).
    emit_ldrb_imm(ctx, inner, /*rt=*/0, /*rn=*/15, /*off=*/0);
    emit_cbz(ctx, inner, /*rt=*/0, /*sf=*/false, advance);
    emit_add_imm(ctx, inner, /*rd=*/15, /*rn=*/15, /*imm12=*/1, /*sf=*/true);
    emit_add_imm(ctx, inner, /*rd=*/13, /*rn=*/13, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, inner, inner);

    // advance: account for trailing nul, ++i, loop.
    emit_add_imm(ctx, advance, /*rd=*/13, /*rn=*/13, /*imm12=*/1, /*sf=*/true);
    emit_add_imm(ctx, advance, /*rd=*/14, /*rn=*/14, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, advance, outer);

    // done: *out_buf_size = total (i32); return 0.
    emit_str_w(ctx, done, /*rt=*/13, /*rn=*/12, /*off=*/0);
    emit_movz(ctx, done, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, done, /*frame_size=*/16);
    emit_ret(ctx, done);

    return synth_leaf_finish(ctx, region, "args_sizes_get", 14);
}

// -----------------------------------------------------------------------------
// args_get(argv_ofs_arr, argv_buf_ofs) -> errno
// argv_ofs_arr[i] (i32 at linmem[argv_ofs_arr + i*4]) is filled in with the
// wasm linmem offset of argv[i]'s string; the bytes of argv[i] (with the
// nul terminator) are copied into linmem starting at argv_buf_ofs.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_args_get(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, outer, copy, copy_step, advance, done;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    outer     = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, outer);
    copy      = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, copy);
    copy_step = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, copy_step);
    advance   = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, advance);
    done      = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, done);

    // x9 = argc; x10 = host argv base.
    emit_ldr_w(ctx, entry, /*rt=*/9,  /*rn=*/28, /*off=*/ARGC_SLOT);
    emit_ldr_x(ctx, entry, /*rt=*/10, /*rn=*/28, /*off=*/ARGV_SLOT);
    // x11 = host(argv_ofs_arr); w12 = current_buf_offset (linmem-relative,
    // starts at the caller-provided argv_buf_ofs and is what we write into
    // argv_ofs_arr[i]); x14 = i.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 12); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_movz(ctx, entry, /*rd=*/14, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_b(ctx, entry, outer);

    // outer: if (i >= argc) done. Else compute argv_ofs_arr[i] slot host
    // address into x13, write current_buf_offset there, prep src pointer
    // in x15 and dst pointer (host(linmem + buf_off)) in x0.
    emit_cmp_reg(ctx, outer, /*rn=*/14, /*rm=*/9, /*sf=*/true);
    emit_b_cond(ctx, outer, COND_GE, done);
    // x13 = x11 + i*4
    emit_movz(ctx, outer, /*rd=*/0, /*imm16=*/2, /*hw=*/0, /*sf=*/true);
    emit_3reg(ctx, outer, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/13, /*rn=*/14, /*rm=*/0, /*sf=*/true);
    emit_add_reg(ctx, outer, /*rd=*/13, /*rn=*/11, /*rm=*/13, /*sf=*/true);
    emit_str_w(ctx, outer, /*rt=*/12, /*rn=*/13, /*off=*/0);
    // x15 = host argv[i] = *(x10 + i*8)
    emit_movz(ctx, outer, /*rd=*/0, /*imm16=*/3, /*hw=*/0, /*sf=*/true);
    emit_3reg(ctx, outer, OP_TYPE_AARCH64_LSL_REG,
              /*rd=*/15, /*rn=*/14, /*rm=*/0, /*sf=*/true);
    emit_add_reg(ctx, outer, /*rd=*/15, /*rn=*/10, /*rm=*/15, /*sf=*/true);
    emit_ldr_x(ctx, outer, /*rt=*/15, /*rn=*/15, /*off=*/0);
    emit_b(ctx, outer, copy);

    // copy: w0 = *src; strb to host(linmem + buf_off); buf_off++; src++;
    // if (w0 == 0) advance else loop.
    emit_ldrb_imm(ctx, copy, /*rt=*/0, /*rn=*/15, /*off=*/0);
    // dst host = x28 + uxtw(w12)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 13); a[1] = attr_i32(ctx, "rn", 12);
        MLIR_AppendBlockOp(ctx, copy, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, copy, /*rd=*/13, /*rn=*/28, /*rm=*/13, /*sf=*/true);
    emit_strb_imm(ctx, copy, /*rt=*/0, /*rn=*/13, /*off=*/0);
    emit_add_imm(ctx, copy, /*rd=*/12, /*rn=*/12, /*imm12=*/1, /*sf=*/false);
    emit_add_imm(ctx, copy, /*rd=*/15, /*rn=*/15, /*imm12=*/1, /*sf=*/true);
    emit_cbz(ctx, copy, /*rt=*/0, /*sf=*/false, advance);
    emit_b(ctx, copy, copy);

    // copy_step: unused (kept for readability). Just jump.
    emit_b(ctx, copy_step, copy);

    // advance: ++i; loop.
    emit_add_imm(ctx, advance, /*rd=*/14, /*rn=*/14, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, advance, outer);

    emit_movz(ctx, done, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, done, /*frame_size=*/16);
    emit_ret(ctx, done);

    return synth_leaf_finish(ctx, region, "args_get", 8);
}

// -----------------------------------------------------------------------------
// environ_sizes_get(out_count_ofs, out_buf_size_ofs) -> errno
// Stub: report 0 environment variables, 0 bytes.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_environ_sizes_get(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // x9 = host(out_count); x10 = host(out_buf_size).
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    // Store zeros (wzr = reg 31).
    emit_str_w(ctx, entry, /*rt=*/31, /*rn=*/9,  /*off=*/0);
    emit_str_w(ctx, entry, /*rt=*/31, /*rn=*/10, /*off=*/0);
    emit_movz(ctx, entry, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "environ_sizes_get", 17);
}

// -----------------------------------------------------------------------------
// environ_get(environ_ofs, buf_ofs) -> errno
// Stub: no environment variables to copy.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_environ_get(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    emit_movz(ctx, entry, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "environ_get", 11);
}

// -----------------------------------------------------------------------------
// fd_close(fd) -> errno
// Issues SYS_close (#6). Always returns 0.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_fd_close(MLIR_Context *ctx) {
    MLIR_BlockHandle entry;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    // x0 already has fd (low 32 bits valid). svc clobbers x16.
    emit_bl(ctx, entry, str_lit("_close"));
    emit_movz(ctx, entry, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, entry, /*frame_size=*/16);
    emit_ret(ctx, entry);
    return synth_leaf_finish(ctx, region, "fd_close", 8);
}

// -----------------------------------------------------------------------------
// fd_read(fd, iovs_ofs, iovs_len, nread_ofs) -> errno
// Iterates (ptr, len) pairs, issues SYS_read for each, accumulates total bytes
// read into *nread_ofs.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_fd_read(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, loop, after_rd, done, err;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    loop     = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, loop);
    after_rd = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, after_rd);
    done     = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, done);
    err      = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, err);

    // Stash fd (w0) in x19, host(iovs) in x9, iovs_len in x20.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 19); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 1);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    emit_mov_x(ctx, entry, /*rd=*/20, /*rn=*/2);
    // x21 = total bytes read
    emit_movz(ctx, entry, /*rd=*/21, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    // Save nread_ofs (w3) into x22 (we'll need it after the loop).
    emit_mov_x(ctx, entry, /*rd=*/22, /*rn=*/3);
    emit_b(ctx, entry, loop);

    // loop: if (iovs_len == 0) done; else read one chunk.
    emit_cbz(ctx, loop, /*rt=*/20, /*sf=*/false, done);
    emit_ldr_w(ctx, loop, /*rt=*/11, /*rn=*/9, /*off=*/0);  // ptr offset
    emit_ldr_w(ctx, loop, /*rt=*/12, /*rn=*/9, /*off=*/4);  // len
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 11); a[1] = attr_i32(ctx, "rn", 11);
        MLIR_AppendBlockOp(ctx, loop, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, loop, /*rd=*/11, /*rn=*/28, /*rm=*/11, /*sf=*/true);
    // read(fd=x19, buf=x11, count=x12)
    emit_mov_x(ctx, loop, /*rd=*/0, /*rn=*/19);
    emit_mov_x(ctx, loop, /*rd=*/1, /*rn=*/11);
    emit_mov_x(ctx, loop, /*rd=*/2, /*rn=*/12);
    emit_bl(ctx, loop, str_lit("_read"));
    // x0 = bytes read (or -errno). Check sign bit.
    emit_cmp_imm(ctx, loop, /*rn=*/0, /*imm12=*/0, /*sf=*/false);
    emit_b_cond(ctx, loop, COND_LT, err);
    emit_b(ctx, loop, after_rd);

    // after_rd: accumulate, advance iovs, decrement iovs_len, loop.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, after_rd, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, after_rd, /*rd=*/21, /*rn=*/21, /*rm=*/10, /*sf=*/true);
    emit_add_imm(ctx, after_rd, /*rd=*/9, /*rn=*/9, /*imm12=*/8, /*sf=*/true);
    emit_sub_imm(ctx, after_rd, /*rd=*/20, /*rn=*/20, /*imm12=*/1, /*sf=*/false);
    emit_b(ctx, after_rd, loop);

    // done: store nread back, return 0. (size_t* is i32 on wasm32.)
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 22);
        MLIR_AppendBlockOp(ctx, done, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, done, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_str_w(ctx, done, /*rt=*/21, /*rn=*/10, /*off=*/0);
    emit_movz(ctx, done, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, done, /*frame_size=*/16);
    emit_ret(ctx, done);

    // err: return errno (we use 8 ~ EBADF; close enough for tinyc).
    emit_movz(ctx, err, /*rd=*/0, /*imm16=*/8, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, err, /*frame_size=*/16);
    emit_ret(ctx, err);

    return synth_leaf_finish(ctx, region, "fd_read", 7);
}

// -----------------------------------------------------------------------------
// fd_seek(fd, offset_i64, whence_i32, newoffset_ofs) -> errno
// SYS_lseek (#199). PLATFORM_SEEK_* values match POSIX SEEK_*.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_fd_seek(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, ok, err;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    ok  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, ok);
    err = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, err);

    // Stash newoffset_ofs (w3) into x19 before clobbering w3.
    emit_mov_x(ctx, entry, /*rd=*/19, /*rn=*/3);
    // lseek(fd=x0(uxtw), offset=x1(i64 already), whence=x2(uxtw))
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 0); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 2); a[1] = attr_i32(ctx, "rn", 2);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_bl(ctx, entry, str_lit("_lseek"));
    // x0 = new offset on success, or -1 on error (kernel returns -1 with
    // C flag set). Check sign bit on the 64-bit result.
    emit_cmp_imm(ctx, entry, /*rn=*/0, /*imm12=*/0, /*sf=*/true);
    emit_b_cond(ctx, entry, COND_LT, err);
    emit_b(ctx, entry, ok);

    // ok: store newoffset (8 bytes, u64) to *newoffset_ofs; return 0.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 19);
        MLIR_AppendBlockOp(ctx, ok, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, ok, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_str_x(ctx, ok, /*rt=*/0, /*rn=*/10, /*off=*/0);
    emit_movz(ctx, ok, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, ok, /*frame_size=*/16);
    emit_ret(ctx, ok);

    emit_movz(ctx, err, /*rd=*/0, /*imm16=*/8, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, err, /*frame_size=*/16);
    emit_ret(ctx, err);

    return synth_leaf_finish(ctx, region, "fd_seek", 7);
}

// -----------------------------------------------------------------------------
// fd_tell(fd, newoffset_ofs) -> errno
// Equivalent to fd_seek(fd, 0, SEEK_CUR, newoffset_ofs).
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_fd_tell(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, ok, err;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, 16);
    ok  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, ok);
    err = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, err);

    // x19 = newoffset_ofs.
    emit_mov_x(ctx, entry, /*rd=*/19, /*rn=*/1);
    // lseek(fd=uxtw(w0), offset=0, whence=1).
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 0); a[1] = attr_i32(ctx, "rn", 0);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_movz(ctx, entry, /*rd=*/1, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_movz(ctx, entry, /*rd=*/2, /*imm16=*/1, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, entry, str_lit("_lseek"));
    emit_cmp_imm(ctx, entry, /*rn=*/0, /*imm12=*/0, /*sf=*/true);
    emit_b_cond(ctx, entry, COND_LT, err);
    emit_b(ctx, entry, ok);

    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 19);
        MLIR_AppendBlockOp(ctx, ok, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, ok, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_str_x(ctx, ok, /*rt=*/0, /*rn=*/10, /*off=*/0);
    emit_movz(ctx, ok, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, ok, /*frame_size=*/16);
    emit_ret(ctx, ok);

    emit_movz(ctx, err, /*rd=*/0, /*imm16=*/8, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, err, /*frame_size=*/16);
    emit_ret(ctx, err);

    return synth_leaf_finish(ctx, region, "fd_tell", 7);
}

// -----------------------------------------------------------------------------
// path_open(dirfd, dirflags, path_ofs, path_len, oflags, rights,
//           rights_inheriting, fdflags, opened_fd_ofs) -> errno
//
// Translates the WASI rights/oflags bitmasks into POSIX open() flags, then
// issues SYS_open. Writes the resulting fd back to *opened_fd_ofs.
//
// Path is copied into a 1024-byte stack scratch buffer + nul terminator.
// -----------------------------------------------------------------------------
static MLIR_OpHandle synth_path_open(MLIR_Context *ctx) {
    MLIR_BlockHandle entry, cpchk, cploop, cpdone;
    MLIR_BlockHandle f_check_w, f_write_only, f_check_creat;
    MLIR_BlockHandle f_check_trunc, f_check_excl, f_check_append;
    MLIR_BlockHandle f_after_w, f_done_flags;
    MLIR_BlockHandle do_open, ok, err;
    MLIR_RegionHandle region = synth_leaf_begin(ctx, &entry, /*frame=*/1024);
    cpchk         = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, cpchk);
    cploop        = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, cploop);
    cpdone        = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, cpdone);
    f_check_w     = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_check_w);
    f_write_only  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_write_only);
    f_after_w     = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_after_w);
    f_check_creat = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_check_creat);
    f_check_trunc = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_check_trunc);
    f_check_excl  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_check_excl);
    f_check_append= MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_check_append);
    f_done_flags  = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, f_done_flags);
    do_open       = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, do_open);
    ok            = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, ok);
    err           = MLIR_CreateBlock(ctx); MLIR_AppendRegionBlock(ctx, region, err);

    // Stash inputs we still need after the syscall:
    //   x19 = opened_fd_ofs (from caller stack [fp,#16]).
    //   x20 = path_len (used by null-terminator placement).
    //   x21 = oflags (we'll mask piece-by-piece for O_CREAT/TRUNC).
    //   x22 = rights (we'll mask for FD_READ / FD_WRITE).
    //   x23 = fdflags (for O_APPEND).
    emit_ldr_w(ctx, entry, /*rt=*/19, /*rn=*/29, /*off=*/16);
    emit_mov_x(ctx, entry, /*rd=*/20, /*rn=*/3);
    emit_mov_x(ctx, entry, /*rd=*/21, /*rn=*/4);
    emit_mov_x(ctx, entry, /*rd=*/22, /*rn=*/5);
    emit_mov_x(ctx, entry, /*rd=*/23, /*rn=*/7);
    // host(path) = x28 + uxtw(w2).
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 9); a[1] = attr_i32(ctx, "rn", 2);
        MLIR_AppendBlockOp(ctx, entry, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, entry, /*rd=*/9, /*rn=*/28, /*rm=*/9, /*sf=*/true);
    // x10 = i = 0; sp = path buffer base.
    emit_movz(ctx, entry, /*rd=*/10, /*imm16=*/0, /*hw=*/0, /*sf=*/true);
    emit_b(ctx, entry, cpchk);

    // cpchk: if (i >= path_len) cpdone.
    emit_cmp_reg(ctx, cpchk, /*rn=*/10, /*rm=*/20, /*sf=*/true);
    emit_b_cond(ctx, cpchk, COND_GE, cpdone);
    emit_b(ctx, cpchk, cploop);

    // cploop: byte = src[i]; sp[i] = byte; i++; cpchk.
    // src + i:
    emit_add_reg(ctx, cploop, /*rd=*/11, /*rn=*/9, /*rm=*/10, /*sf=*/true);
    emit_ldrb_imm(ctx, cploop, /*rt=*/12, /*rn=*/11, /*off=*/0);
    // dst (sp + i): materialise sp into x13 first (shifted-reg form
    // of `add` treats Rn=31 as XZR, not SP — must use add-imm to
    // get sp into a real register, then add the offset).
    emit_add_imm(ctx, cploop, /*rd=*/13, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_add_reg(ctx, cploop, /*rd=*/13, /*rn=*/13, /*rm=*/10, /*sf=*/true);
    emit_strb_imm(ctx, cploop, /*rt=*/12, /*rn=*/13, /*off=*/0);
    emit_add_imm(ctx, cploop, /*rd=*/10, /*rn=*/10, /*imm12=*/1, /*sf=*/true);
    emit_b(ctx, cploop, cpchk);

    // cpdone: nul terminator at sp + path_len; start flag translation.
    emit_add_imm(ctx, cpdone, /*rd=*/13, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_add_reg(ctx, cpdone, /*rd=*/13, /*rn=*/13, /*rm=*/20, /*sf=*/true);
    emit_strb_imm(ctx, cpdone, /*rt=*/31, /*rn=*/13, /*off=*/0);
    // w14 = os_flags accumulator = 0.
    emit_movz(ctx, cpdone, /*rd=*/14, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, cpdone, f_check_w);

    // f_check_w: if (rights & FD_WRITE) check_read else after_w (RDONLY).
    emit_movz(ctx, f_check_w, /*rd=*/15, /*imm16=*/0x40, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_w, OP_TYPE_AARCH64_AND_REG,
              /*rd=*/11, /*rn=*/22, /*rm=*/15, /*sf=*/false);
    emit_cbz(ctx, f_check_w, /*rt=*/11, /*sf=*/false, f_after_w);
    // has_write: check read bit. If also read -> O_RDWR (=2). Else WO (=1).
    emit_movz(ctx, f_check_w, /*rd=*/15, /*imm16=*/0x02, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_w, OP_TYPE_AARCH64_AND_REG,
              /*rd=*/11, /*rn=*/22, /*rm=*/15, /*sf=*/false);
    emit_cbz(ctx, f_check_w, /*rt=*/11, /*sf=*/false, f_write_only);
    emit_movz(ctx, f_check_w, /*rd=*/14, /*imm16=*/2, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, f_check_w, f_after_w);

    emit_movz(ctx, f_write_only, /*rd=*/14, /*imm16=*/1, /*hw=*/0, /*sf=*/false);
    emit_b(ctx, f_write_only, f_after_w);

    // f_after_w: apply O_CREAT if oflags bit 0.
    emit_b(ctx, f_after_w, f_check_creat);

    emit_movz(ctx, f_check_creat, /*rd=*/15, /*imm16=*/1, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_creat, OP_TYPE_AARCH64_AND_REG,
              /*rd=*/11, /*rn=*/21, /*rm=*/15, /*sf=*/false);
    emit_cbz(ctx, f_check_creat, /*rt=*/11, /*sf=*/false, f_check_trunc);
    emit_movz(ctx, f_check_creat, /*rd=*/15, /*imm16=*/0x200, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_creat, OP_TYPE_AARCH64_ORR_REG,
              /*rd=*/14, /*rn=*/14, /*rm=*/15, /*sf=*/false);
    emit_b(ctx, f_check_creat, f_check_trunc);

    // f_check_trunc: O_TRUNC if oflags bit 3 (PLATFORM_O_TRUNC = 8).
    emit_movz(ctx, f_check_trunc, /*rd=*/15, /*imm16=*/8, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_trunc, OP_TYPE_AARCH64_AND_REG,
              /*rd=*/11, /*rn=*/21, /*rm=*/15, /*sf=*/false);
    emit_cbz(ctx, f_check_trunc, /*rt=*/11, /*sf=*/false, f_check_excl);
    emit_movz(ctx, f_check_trunc, /*rd=*/15, /*imm16=*/0x400, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_trunc, OP_TYPE_AARCH64_ORR_REG,
              /*rd=*/14, /*rn=*/14, /*rm=*/15, /*sf=*/false);
    emit_b(ctx, f_check_trunc, f_check_excl);

    // f_check_excl: O_EXCL if oflags bit 2 (WASI EXCL = 4); macOS O_EXCL=0x800.
    emit_movz(ctx, f_check_excl, /*rd=*/15, /*imm16=*/4, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_excl, OP_TYPE_AARCH64_AND_REG,
              /*rd=*/11, /*rn=*/21, /*rm=*/15, /*sf=*/false);
    emit_cbz(ctx, f_check_excl, /*rt=*/11, /*sf=*/false, f_check_append);
    emit_movz(ctx, f_check_excl, /*rd=*/15, /*imm16=*/0x800, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_excl, OP_TYPE_AARCH64_ORR_REG,
              /*rd=*/14, /*rn=*/14, /*rm=*/15, /*sf=*/false);
    emit_b(ctx, f_check_excl, f_check_append);

    // f_check_append: O_APPEND if fdflags bit 0.
    emit_movz(ctx, f_check_append, /*rd=*/15, /*imm16=*/1, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_append, OP_TYPE_AARCH64_AND_REG,
              /*rd=*/11, /*rn=*/23, /*rm=*/15, /*sf=*/false);
    emit_cbz(ctx, f_check_append, /*rt=*/11, /*sf=*/false, f_done_flags);
    emit_movz(ctx, f_check_append, /*rd=*/15, /*imm16=*/0x08, /*hw=*/0, /*sf=*/false);
    emit_3reg(ctx, f_check_append, OP_TYPE_AARCH64_ORR_REG,
              /*rd=*/14, /*rn=*/14, /*rm=*/15, /*sf=*/false);
    emit_b(ctx, f_check_append, f_done_flags);

    emit_b(ctx, f_done_flags, do_open);

    // do_open: x0 = sp (path), x1 = os_flags, x2 = 0o644.
    emit_add_imm(ctx, do_open, /*rd=*/0, /*rn=*/31, /*imm12=*/0, /*sf=*/true);
    emit_mov_x(ctx, do_open, /*rd=*/1, /*rn=*/14);
    emit_movz(ctx, do_open, /*rd=*/2, /*imm16=*/0x1a4, /*hw=*/0, /*sf=*/true);
    emit_bl(ctx, do_open, str_lit("_open"));
    emit_cmp_imm(ctx, do_open, /*rn=*/0, /*imm12=*/0, /*sf=*/false);
    emit_b_cond(ctx, do_open, COND_LT, err);
    emit_b(ctx, do_open, ok);

    // ok: *opened_fd_ofs = w0; return 0.
    {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "rd", 10); a[1] = attr_i32(ctx, "rn", 19);
        MLIR_AppendBlockOp(ctx, ok, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
    }
    emit_add_reg(ctx, ok, /*rd=*/10, /*rn=*/28, /*rm=*/10, /*sf=*/true);
    emit_str_w(ctx, ok, /*rt=*/0, /*rn=*/10, /*off=*/0);
    emit_movz(ctx, ok, /*rd=*/0, /*imm16=*/0, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, ok, /*frame_size=*/1024);
    emit_ret(ctx, ok);

    // err: return 44 (WASI ENOENT). tinyc treats !=0 as failure.
    emit_movz(ctx, err, /*rd=*/0, /*imm16=*/44, /*hw=*/0, /*sf=*/false);
    emit_epilogue(ctx, err, /*frame_size=*/1024);
    emit_ret(ctx, err);

    return synth_leaf_finish(ctx, region, "path_open", 9);
}


typedef struct {
    int32_t n_globals;
    bool    uses_linmem;
    bool    needs_printI64;
    bool    needs_printNewline;
    bool    needs_printStr;
    bool    needs_printf;
    bool    needs_malloc;
    bool    needs_free;
    bool    needs_strlen;
    bool    needs_strcmp;
    bool    needs_memcmp;
    bool    needs_memchr;
    bool    needs_fd_write;
    bool    needs_proc_exit;
    bool    needs_va_arg_i32;
    bool    needs_va_arg_i64;
    bool    needs_va_arg_ptr;
    bool    needs_va_arg_struct;
    bool    needs_va_arg_f64;
    bool    needs_print_f32;
    bool    needs_print_f64;
    bool    needs_path_open;
    bool    needs_fd_close;
    bool    needs_fd_read;
    bool    needs_fd_seek;
    bool    needs_fd_tell;
    bool    needs_args_get;
    bool    needs_args_sizes_get;
    bool    needs_environ_get;
    bool    needs_environ_sizes_get;
    // user_provides_* are set when the wmir module already defines a
    // function with the corresponding name. Synth helpers must NOT be
    // emitted in that case — they would shadow the user's function and
    // (worse) be picked as the first match by the mach-o backend's
    // name-based reloc resolver. tinyc selfhost compiles
    // corec-stdlib/stdlib.c which provides malloc/free/realloc;
    // synth_malloc is a header-less bump allocator, so realloc's
    // `hdr->size` read would land on uninitialised bytes if synth_malloc
    // shadowed the real malloc.
    bool    user_provides_malloc;
    bool    user_provides_free;
    bool    user_provides_strlen;
    bool    user_provides_strcmp;
    bool    user_provides_memcmp;
    bool    user_provides_memchr;
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
        } else if (t == OP_TYPE_WMIR_DATA_INIT) {
            // Implies a __DATA segment that needs to be loaded into
            // linmem, so we need both x28 (linmem base) and a
            // file-backed init region.
            mi->uses_linmem = true;
        } else if (t == OP_TYPE_WMIR_CALL) {
            string callee = at_s(op, "target");
            if (callee.size == 0) callee = at_s(op, "callee");
            #define EQ_LIT(s, lit) \
                ((s).size == (sizeof(lit) - 1) && \
                 memcmp((s).str, (lit), sizeof(lit) - 1) == 0)
            if (EQ_LIT(callee, "printI64"))      mi->needs_printI64    = true;
            if (EQ_LIT(callee, "printNewline"))  mi->needs_printNewline = true;
            if (EQ_LIT(callee, "printStr"))      mi->needs_printStr    = true;
            if (EQ_LIT(callee, "printf"))        mi->needs_printf      = true;
            if (EQ_LIT(callee, "malloc"))        mi->needs_malloc      = true;
            if (EQ_LIT(callee, "free"))          mi->needs_free        = true;
            if (EQ_LIT(callee, "strlen"))        mi->needs_strlen      = true;
            if (EQ_LIT(callee, "strcmp"))        mi->needs_strcmp      = true;
            if (EQ_LIT(callee, "memcmp"))        mi->needs_memcmp      = true;
            if (EQ_LIT(callee, "memchr"))        mi->needs_memchr      = true;
            if (EQ_LIT(callee, "fd_write"))      mi->needs_fd_write    = true;
            if (EQ_LIT(callee, "proc_exit"))     mi->needs_proc_exit   = true;
            if (EQ_LIT(callee, "tinyc_va_arg_i32")) mi->needs_va_arg_i32 = true;
            if (EQ_LIT(callee, "tinyc_va_arg_i64")) mi->needs_va_arg_i64 = true;
            if (EQ_LIT(callee, "tinyc_va_arg_ptr")) mi->needs_va_arg_ptr = true;
            if (EQ_LIT(callee, "tinyc_va_arg_struct")) mi->needs_va_arg_struct = true;
            if (EQ_LIT(callee, "tinyc_va_arg_f64")) mi->needs_va_arg_f64 = true;
            if (EQ_LIT(callee, "printF32"))      mi->needs_print_f32   = true;
            if (EQ_LIT(callee, "printF64"))      mi->needs_print_f64   = true;
            if (EQ_LIT(callee, "path_open"))     { mi->needs_path_open   = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "fd_close"))      mi->needs_fd_close    = true;
            if (EQ_LIT(callee, "fd_read"))       { mi->needs_fd_read     = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "fd_seek"))       { mi->needs_fd_seek     = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "fd_tell"))       { mi->needs_fd_tell     = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "args_get"))      { mi->needs_args_get    = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "args_sizes_get")) { mi->needs_args_sizes_get = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "environ_get"))   { mi->needs_environ_get = true; mi->uses_linmem = true; }
            if (EQ_LIT(callee, "environ_sizes_get")) { mi->needs_environ_sizes_get = true; mi->uses_linmem = true; }
            #undef EQ_LIT
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

    // Scan top-level WMIR_FUNC ops to record which "helper" names the
    // user wasm module already provides. We must not synthesize over
    // these or the mach-o name-based reloc resolver will pick the
    // synth (header-less bump allocator) and break realloc, which
    // expects a header-bearing malloc.
    {
        size_t n_pre = MLIR_GetBlockNumOps(mb);
        for (size_t i = 0; i < n_pre; i++) {
            MLIR_OpHandle top = MLIR_GetBlockOp(mb, i);
            if (MLIR_GetOpType(top) != OP_TYPE_WMIR_FUNC) continue;
            string sn = at_s(top, "sym_name");
            #define EQ_LIT2(s, lit) \
                ((s).size == (sizeof(lit) - 1) && \
                 memcmp((s).str, (lit), sizeof(lit) - 1) == 0)
            if (EQ_LIT2(sn, "malloc")) mi.user_provides_malloc = true;
            if (EQ_LIT2(sn, "free"))   mi.user_provides_free   = true;
            if (EQ_LIT2(sn, "strlen")) mi.user_provides_strlen = true;
            if (EQ_LIT2(sn, "strcmp")) mi.user_provides_strcmp = true;
            if (EQ_LIT2(sn, "memcmp")) mi.user_provides_memcmp = true;
            if (EQ_LIT2(sn, "memchr")) mi.user_provides_memchr = true;
            #undef EQ_LIT2
        }
    }

    uint32_t n_globals = (uint32_t)mi.n_globals;
    uint64_t global0_init = DEFAULT_STACK_SIZE;
    uint64_t lm_bytes = (uint64_t)DEFAULT_STACK_SIZE + DEFAULT_GLOBAL_BASE_OFFS;
    uint64_t lm_pages = (lm_bytes + WASM_PAGE_SIZE - 1) / WASM_PAGE_SIZE;
    // If the upstream wasm module declared a memory MIN larger than
    // our default (e.g. tinyc selfhost: ~8.5 MiB of static data needs
    // ~130 pages), grow linmem to fit. Otherwise __heap_base would
    // land past the end of linmem and platform_heap_size() (which is
    // `memory.size * 64KiB - __heap_base`) would underflow to a
    // huge unsigned value, sending the buddy allocator into the void.
    MLIR_AttributeHandle a_min_pages = MLIR_GetOpAttributeByName(
        wmir_module, "memory_min_pages");
    if (a_min_pages) {
        uint64_t min_pages = (uint64_t)MLIR_GetAttributeInteger(a_min_pages);
        if (min_pages > lm_pages) {
            lm_pages = min_pages;
        }
    }
    g_linmem_pages = (uint32_t)lm_pages;
    // The Mach-O envelope reserves the whole MAX_LINMEM_PAGES range as
    // an S_ZEROFILL BSS section (zero on-disk cost, lazy commit at
    // touch time). The wasm-side `memory.size` returns the CURRENT
    // page count tracked in linmem[MEM_PAGES_SLOT], initialised to
    // g_linmem_pages and grown by `memory.grow` up to MAX_LINMEM_PAGES.
    uint64_t linmem_size = (uint64_t)MAX_LINMEM_PAGES * WASM_PAGE_SIZE;

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
        if (t == OP_TYPE_WMIR_DATA_INIT) {
            // Pass through unchanged as an aarch64.data_init op so the
            // macho backend can overlay the bytes onto the linmem image.
            string sn = at_s(top, "sym_name");
            string id = at_s(top, "init_data");
            int64_t off = at_i(top, "offset");
            MLIR_AttributeHandle a[3];
            a[0] = attr_s  (ctx, "sym_name",  sn.str, sn.size);
            a[1] = attr_i32(ctx, "offset",    off);
            a[2] = attr_s  (ctx, "init_data", id.str, id.size);
            MLIR_OpHandle di = MLIR_CreateOp(ctx, OP_TYPE_AARCH64_DATA_INIT,
                op_type_to_string(OP_TYPE_AARCH64_DATA_INIT),
                a, 3, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                MLIR_CreateLocationUnknown(ctx, (string){0}),
                MLIR_INVALID_HANDLE, (string){0}, -1);
            MLIR_AppendBlockOp(ctx, out_body, di);
            continue;
        }
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
    bool use_linmem  = mi.uses_linmem || mi.needs_printStr || mi.needs_printf ||
        mi.needs_malloc || mi.needs_free || mi.needs_strlen ||
        mi.needs_strcmp || mi.needs_memcmp || mi.needs_memchr ||
        mi.needs_fd_write || mi.needs_va_arg_i32 || mi.needs_va_arg_i64 ||
        mi.needs_va_arg_ptr || mi.needs_va_arg_struct ||
        mi.needs_va_arg_f64;
    MLIR_OpHandle start = synth_start(ctx, entry_name,
        /*use_data_priv=*/false, use_globals, use_linmem);
    if (!start) return MLIR_INVALID_HANDLE;
    MLIR_AppendBlockOp(ctx, out_body, start);

    if (mi.needs_printI64 || mi.needs_printf) {
        MLIR_OpHandle p = synth_print_i64(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_printNewline) {
        MLIR_OpHandle p = synth_print_newline(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_printStr) {
        MLIR_OpHandle p = synth_print_str(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_printf) {
        MLIR_OpHandle p = synth_printf(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_malloc && !mi.user_provides_malloc) {
        MLIR_OpHandle p = synth_malloc(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_free && !mi.user_provides_free) {
        MLIR_OpHandle p = synth_free(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_strlen && !mi.user_provides_strlen) {
        MLIR_OpHandle p = synth_strlen(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_strcmp && !mi.user_provides_strcmp) {
        MLIR_OpHandle p = synth_strcmp(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_memcmp && !mi.user_provides_memcmp) {
        MLIR_OpHandle p = synth_memcmp(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_memchr && !mi.user_provides_memchr) {
        MLIR_OpHandle p = synth_memchr(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_fd_write) {
        MLIR_OpHandle p = synth_fd_write(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_proc_exit) {
        MLIR_OpHandle p = synth_proc_exit(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_va_arg_i32) {
        MLIR_OpHandle p = synth_tinyc_va_arg_i32(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_va_arg_i64) {
        MLIR_OpHandle p = synth_tinyc_va_arg_i64(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_va_arg_ptr) {
        MLIR_OpHandle p = synth_tinyc_va_arg_ptr(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_va_arg_struct) {
        MLIR_OpHandle p = synth_tinyc_va_arg_struct(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_va_arg_f64) {
        MLIR_OpHandle p = synth_tinyc_va_arg_f64(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_print_f32 || mi.needs_print_f64 || mi.needs_printf) {
        // printF32 and the synthesised printf both call printF64; emit
        // printF64 if any consumer is present.
        MLIR_OpHandle p = synth_print_f64(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_print_f32) {
        MLIR_OpHandle p = synth_print_f32(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_path_open) {
        MLIR_OpHandle p = synth_path_open(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_fd_close) {
        MLIR_OpHandle p = synth_fd_close(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_fd_read) {
        MLIR_OpHandle p = synth_fd_read(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_fd_seek) {
        MLIR_OpHandle p = synth_fd_seek(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_fd_tell) {
        MLIR_OpHandle p = synth_fd_tell(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_args_get) {
        MLIR_OpHandle p = synth_args_get(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_args_sizes_get) {
        MLIR_OpHandle p = synth_args_sizes_get(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_environ_get) {
        MLIR_OpHandle p = synth_environ_get(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }
    if (mi.needs_environ_sizes_get) {
        MLIR_OpHandle p = synth_environ_sizes_get(ctx);
        if (!p) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, p);
    }

    return out_module;
}

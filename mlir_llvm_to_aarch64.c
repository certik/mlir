// llvm dialect -> aarch64 lowering. See mlir_llvm_to_aarch64.h for the
// public API, rationale, and the staged build-up plan.
//
// Status: Step 2 (in progress) — straight-line integer codegen. Lowers a
// single-block `llvm.func` of integer operations directly to the
// physical-register `aarch64` dialect, then relies on the existing
// `mlir_aarch64_to_macho` encoder. A synthesised `_start` calls `main`
// and exits with its return value via libSystem `_exit`.
//
// Supported so far: parameters (<=8 integer args), llvm.mlir.constant,
// integer binops (add/sub/mul/sdiv/udiv/srem/urem/and/or/xor/shl/lshr/
// ashr), llvm.icmp, llvm.select, direct llvm.call (<=8 args), llvm.return.
// NOT yet: structured control flow (scf.if/while/index_switch), memory
// (alloca/load/store/getelementptr), globals/strings, indirect & variadic
// calls, floating point. Each lands a clear stderr diagnostic.
//
// Register model: a trivial "spill everything" allocator — every SSA value
// gets its own 8-byte frame slot; operands are reloaded into scratch regs
// (x9/x10/x11) per instruction and results stored back immediately. Always
// correct and call-safe (no value kept live across an op), if slow. The
// real linear-scan allocator + a materialised virtual-register `a64ssa`
// tier arrive together in Step 3, where keeping values in registers pays
// off. Emitting `aarch64.*` directly here is the GlobalISel "fast isel +
// trivial regalloc" shape.
//
// Nothing here is on the path of the existing wasm-wrapper backend; it is
// reached only via the opt-in `--macho-backend=llvm` driver flag.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "mlir_llvm_to_aarch64.h"
#include "mlir_op_names.h"

// ---------------------------------------------------------------------------
// Small attribute / op builders (mirrors of the former wmir backend's helpers).
// ---------------------------------------------------------------------------
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name,
                                     int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

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
static void emit_ret(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_RET, NULL, 0));
}
// blr Xn — indirect branch-and-link via register (for indirect calls).
static void emit_blr(MLIR_Context *ctx, MLIR_BlockHandle blk, uint8_t rn) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_BLR, a, 1));
}
// ADRP rd, <section page>  /  ADD rd, rn, #<section lo12 + addend>. The
// encoder resolves `target` (a section kind or, for native globals, the
// "linmem_template" data section) and adds `addend` to the base address.
static void emit_adrp_data(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, string target, uint32_t addend) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_s(ctx, "target", target.str, target.size);
    a[2] = attr_i32(ctx, "addend", (int32_t)addend);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADRP_DATA, a, 3));
}
static void emit_add_data_lo(MLIR_Context *ctx, MLIR_BlockHandle blk,
                             uint8_t rd, uint8_t rn, string target,
                             uint32_t addend) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_s(ctx, "target", target.str, target.size);
    a[3] = attr_i32(ctx, "addend", (int32_t)addend);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_DATA_LO, a, 4));
}

// ---- Floating-point op builders (mirror the former wmir->aarch64 backend) --------
// FMOV between a GP and a V register. dir_to_v=true: GP->V; false: V->GP.
// sf=true picks X<->D (moves all 64 bits); sf=false picks W<->S.
static void emit_fmov_gp_v(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           bool dir_to_v, bool sf, uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_b  (ctx, "dir_to_v", dir_to_v);
    a[1] = attr_b  (ctx, "sf",       sf);
    a[2] = attr_i32(ctx, "rd",       rd);
    a[3] = attr_i32(ctx, "rn",       rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_FMOV_GP_V, a, 4));
}
// FADD/FSUB/FMUL/FDIV on V registers. kind = "fadd"|"fsub"|"fmul"|"fdiv".
static void emit_fp_binop(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          const char *kind, int fwidth,
                          uint8_t rd, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_s  (ctx, "kind",   kind, strlen(kind));
    a[1] = attr_i32(ctx, "fwidth", fwidth);
    a[2] = attr_i32(ctx, "rd",     rd);
    a[3] = attr_i32(ctx, "rn",     rn);
    a[4] = attr_i32(ctx, "rm",     rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_FP_BINOP, a, 5));
}
// FNEG/FABS/FSQRT on V registers.
static void emit_fp_unop(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         const char *kind, int fwidth,
                         uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_s  (ctx, "kind",   kind, strlen(kind));
    a[1] = attr_i32(ctx, "fwidth", fwidth);
    a[2] = attr_i32(ctx, "rd",     rd);
    a[3] = attr_i32(ctx, "rn",     rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_FP_UNOP, a, 4));
}
// FCMP Sn/Dn, Sm/Dm (sets NZCV, no result reg).
static void emit_fcmp(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      int fwidth, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "fwidth", fwidth);
    a[1] = attr_i32(ctx, "rn",     rn);
    a[2] = attr_i32(ctx, "rm",     rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_FCMP, a, 3));
}
// FP conversion. kind ∈ {"f2f","f2i","i2f"}; sign matters for f2i/i2f only.
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
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_FP_CVT, a, 6));
}

// Materialise a 64-bit immediate into `rd` with a movz + movk chain,
// emitting only the non-zero 16-bit lanes (movz seeds the lowest lane so
// a zero value still produces `movz rd, #0`).
static void emit_load_imm(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint8_t rd, uint64_t v, bool sf) {
    int lanes = sf ? 4 : 2;
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffffu), 0, sf);
    for (int hw = 1; hw < lanes; hw++) {
        uint16_t chunk = (uint16_t)((v >> (16 * hw)) & 0xffffu);
        if (chunk != 0) emit_movk(ctx, blk, rd, chunk, (uint8_t)hw, sf);
    }
}

static void emit_mov_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_MOV_X, a, 2));
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
// Immediate shift (lsl/lsr/asr #shift), used for mul-by-2^k strength reduction
// and constant-amount shifts.
static void emit_shift_imm(MLIR_Context *ctx, MLIR_BlockHandle blk, MLIR_OpType t,
                           uint8_t rd, uint8_t rn, uint8_t shift, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "shift", shift);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 4));
}
// add rd, rn, rm, LSL #lsl  (the encoder reads the optional "lsl" attribute,
// defaulting to 0 for every other ADD_REG emission). Used to fold a power-of-2
// gep stride or a single-use shift-by-constant addend into one shifted-add
// instead of a mov+mul+add / lsl+add sequence. `sf` picks Xd (true) / Wd.
static void emit_add_reg_lsl(MLIR_Context *ctx, MLIR_BlockHandle blk,
                             uint8_t rd, uint8_t rn, uint8_t rm, uint8_t lsl,
                             bool sf) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_b(ctx, "sf", sf);
    a[4] = attr_i32(ctx, "lsl", lsl);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_ADD_REG, a, 5));
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
static void emit_ldst_x(MLIR_Context *ctx, MLIR_BlockHandle blk, MLIR_OpType t,
                        uint8_t rt, uint8_t rn, uint32_t off_bytes) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "off_bytes", (int64_t)off_bytes);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 3));
}
static void emit_ldr_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off) {
    emit_ldst_x(ctx, blk, OP_TYPE_AARCH64_LDR_X, rt, rn, off);
}
static void emit_str_x(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t rt, uint8_t rn, uint32_t off) {
    emit_ldst_x(ctx, blk, OP_TYPE_AARCH64_STR_X, rt, rn, off);
}
// Register-offset 64-bit load/store: ldr/str Xt, [Xn, Xm, LSL #0].
static void emit_ldst_x_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                            MLIR_OpType t, uint8_t rt, uint8_t rn, uint8_t rm) {
    MLIR_AttributeHandle a[3];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, t, a, 3));
}
// Materialise an unsigned byte offset into register `rd`.
static void emit_mov_imm32(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rd, uint32_t v) {
    emit_movz(ctx, blk, rd, (uint16_t)(v & 0xffffu), 0, true);
    if (v >> 16)
        emit_movk(ctx, blk, rd, (uint16_t)((v >> 16) & 0xffffu), 1, true);
}
// 64-bit load from [base + byte_off] into rt, handling offsets beyond the
// 12-bit scaled immediate range via a materialised register offset (x17).
static void emit_ldr_x_off(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rt, uint8_t base, uint32_t byte_off) {
    if (byte_off <= 32760u && (byte_off & 7u) == 0u) {
        emit_ldr_x(ctx, blk, rt, base, byte_off);
    } else {
        emit_mov_imm32(ctx, blk, 17, byte_off);
        emit_ldst_x_reg(ctx, blk, OP_TYPE_AARCH64_LDR_X_REG, rt, base, 17);
    }
}
// 64-bit store of rt into [base + byte_off], large-offset safe (scratch x17).
static void emit_str_x_off(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rt, uint8_t base, uint32_t byte_off) {
    if (byte_off <= 32760u && (byte_off & 7u) == 0u) {
        emit_str_x(ctx, blk, rt, base, byte_off);
    } else {
        emit_mov_imm32(ctx, blk, 17, byte_off);
        emit_ldst_x_reg(ctx, blk, OP_TYPE_AARCH64_STR_X_REG, rt, base, 17);
    }
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
// Mask the low `w` bits: `and rd, rn, #((1<<w)-1)` in a SINGLE instruction
// (the aarch64 logical bitmask immediate covers every contiguous-low-ones
// mask), replacing the 2-instr `mov scratch,#mask; and rd,rn,scratch`. Caller
// must ensure 1 <= w <= (sf?63:31); w==0/>=64 has no single-AND form.
static void emit_and_imm_lowbits(MLIR_Context *ctx, MLIR_BlockHandle blk,
                                 uint8_t rd, uint8_t rn, uint8_t w, bool sf) {
    MLIR_AttributeHandle a[4];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "w", w);
    a[3] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_AND_IMM, a, 4));
}
// Add a signed constant byte offset to the running gep address in x9, using a
// single add/sub immediate when it fits 12 bits, else materialising it.
static void emit_gep_const_off(MLIR_Context *ctx, MLIR_BlockHandle blk,
                               int64_t off) {
    if (off == 0) return;
    if (off > 0 && off <= 0xFFF) {
        emit_add_imm(ctx, blk, 9, 9, (uint16_t)off, true);
    } else if (off < 0 && -off <= 0xFFF) {
        emit_sub_imm(ctx, blk, 9, 9, (uint16_t)(-off), true);
    } else {
        emit_load_imm(ctx, blk, 10, (uint64_t)off, true);
        emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, 9, 9, 10, true);
    }
}
static bool emit_mem_load(MLIR_Context *ctx, MLIR_BlockHandle blk,
                          uint8_t rt, uint8_t base, unsigned sz) {
    MLIR_OpType t = sz == 1 ? OP_TYPE_AARCH64_LDRB_IMM
                  : sz == 4 ? OP_TYPE_AARCH64_LDR_W
                  : sz == 8 ? OP_TYPE_AARCH64_LDR_X
                            : OP_TYPE_AARCH64_LDR_X;
    if (sz != 1 && sz != 4 && sz != 8) return false;
    emit_ldst_x(ctx, blk, t, rt, base, 0);
    return true;
}
static bool emit_mem_store(MLIR_Context *ctx, MLIR_BlockHandle blk,
                           uint8_t rt, uint8_t base, unsigned sz) {
    MLIR_OpType t = sz == 1 ? OP_TYPE_AARCH64_STRB_IMM
                  : sz == 4 ? OP_TYPE_AARCH64_STR_W
                  : sz == 8 ? OP_TYPE_AARCH64_STR_X
                            : OP_TYPE_AARCH64_STR_X;
    if (sz != 1 && sz != 4 && sz != 8) return false;
    emit_ldst_x(ctx, blk, t, rt, base, 0);
    return true;
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
                      uint8_t rd, uint8_t rn, uint8_t rm, uint8_t cond, bool sf) {
    MLIR_AttributeHandle a[5];
    a[0] = attr_i32(ctx, "rd", rd);
    a[1] = attr_i32(ctx, "rn", rn);
    a[2] = attr_i32(ctx, "rm", rm);
    a[3] = attr_i32(ctx, "cond", cond);
    a[4] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_CSEL, a, 5));
}
static void emit_prologue(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t fs) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)fs);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_PROLOGUE, a, 1));
}
static void emit_epilogue(MLIR_Context *ctx, MLIR_BlockHandle blk, uint32_t fs) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "frame_size", (int32_t)fs);
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_EPILOGUE, a, 1));
}
static MLIR_OpHandle build_branch_op(MLIR_Context *ctx, MLIR_OpType t,
                                     MLIR_AttributeHandle *attrs, size_t na,
                                     MLIR_BlockHandle target) {
    MLIR_BlockHandle succs[1] = { target };
    MLIR_ValueHandle *succ_ops[1] = { NULL };
    size_t           n_succ_ops[1] = { 0 };
    return MLIR_CreateOpWithSuccessors(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        succs, 1, succ_ops, n_succ_ops,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}
static void emit_b(MLIR_Context *ctx, MLIR_BlockHandle blk,
                   MLIR_BlockHandle target) {
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_B, NULL, 0, target));
}
static void emit_bcond(MLIR_Context *ctx, MLIR_BlockHandle blk,
                       uint8_t cond, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[1];
    a[0] = attr_i32(ctx, "cond", cond);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_B_COND, a, 1, target));
}
static void emit_cbnz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rt, bool sf, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_CBNZ, a, 2, target));
}
static void emit_cbz(MLIR_Context *ctx, MLIR_BlockHandle blk,
                     uint8_t rt, bool sf, MLIR_BlockHandle target) {
    MLIR_AttributeHandle a[2];
    a[0] = attr_i32(ctx, "rt", rt);
    a[1] = attr_b(ctx, "sf", sf);
    MLIR_AppendBlockOp(ctx, blk,
        build_branch_op(ctx, OP_TYPE_AARCH64_CBZ, a, 2, target));
}

// ---------------------------------------------------------------------------
// Helpers over the input `llvm` dialect module.
// ---------------------------------------------------------------------------
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static unsigned a64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty);

// True when a value occupies a full 64-bit general-purpose register, i.e. it
// is an i64 OR a pointer (!llvm.ptr / ptr) — both must use 64-bit (sf=1)
// AArch64 ops. Deciding this from the literal "i64" string alone is WRONG: a
// pointer-typed value would then be lowered with sf=0, and ops like CSEL/CMP/
// ADD would zero the upper 32 bits, truncating the pointer to its low half.
static bool type_is_gp64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    return a64_type_size(ctx, MLIR_GetValueType(v)) == 8;
}

// Bit width of an integer-typed value ("i1"/"i8"/"i16"/"i32"/"i64"). 0 if not
// a recognised integer type.
static int int_type_bits(MLIR_Context *ctx, MLIR_ValueHandle v) {
    string s = MLIR_GetTypeString(ctx, MLIR_GetValueType(v));
    if (s.size < 2 || s.str[0] != 'i') return 0;
    int w = 0;
    for (size_t i = 1; i < s.size; i++) {
        if (s.str[i] < '0' || s.str[i] > '9') return 0;
        w = w * 10 + (s.str[i] - '0');
    }
    return w;
}

// Forward decl: extract an integer llvm.mlir.constant's value (defined below).
static bool const_int_val(MLIR_Context *ctx, MLIR_OpHandle op,
                          int64_t *val, uint8_t *is64);

// Helper: is `addop` an `llvm.add(base, zext(idx_i32))` (zext on either side)?
// On success fills *base (i64), *idx (i32) and *zop (the zext op).
static bool match_base_zext(MLIR_Context *ctx, MLIR_OpHandle addop,
                            MLIR_ValueHandle *base, MLIR_ValueHandle *idx,
                            MLIR_OpHandle *zop) {
    if (addop == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(addop), "llvm.add")) return false;
    if (MLIR_GetOpNumOperands(addop) != 2) return false;
    MLIR_ValueHandle a[2] = { MLIR_GetOpOperand(addop, 0),
                              MLIR_GetOpOperand(addop, 1) };
    for (int k = 0; k < 2; k++) {
        MLIR_OpHandle zdef = MLIR_GetValueDefiningOp(a[k]);
        if (zdef == MLIR_INVALID_HANDLE) continue;
        if (!name_eq(MLIR_GetOpName(zdef), "llvm.zext")) continue;
        if (MLIR_GetOpNumOperands(zdef) != 1) continue;
        MLIR_ValueHandle src = MLIR_GetOpOperand(zdef, 0);
        if (int_type_bits(ctx, src) != 32) continue;   // UXTW reads 32 bits
        *base = a[1 - k];
        *idx  = src;
        *zop  = zdef;
        return true;
    }
    return false;
}

// Match the linmem-pointer spine feeding `P`. Two shapes are recognised
// (the second is the struct-field / static-offset case, where the lifter emits
// an extra `add ea, const`; see linmem_ptr in mlir_wasmssa_to_llvm.c):
//   off==0:  p = inttoptr( add(base, zext(idx)) )
//   off!=0:  p = inttoptr( add( add(base, zext(idx)), const ) )
// On success fills *base/*idx/*zop (the inner add's zext), *eop (the inner
// `base+zext` add), *pop (the inttoptr), *coff (the constant byte offset, 0 if
// none), and *oadd (the outer `+const` add to skip, INVALID if none). The whole
// spine collapses to `add Wtmp,Widx,#off ; Rt,[base,Wtmp,UXTW]` (off!=0) or a
// single `Rt,[base,Widx,UXTW]` (off==0). Folding the offset into the 32-bit
// index is sound: a valid wasm access has idx+off < memsize <= 2^32, so the
// 32-bit add never wraps, and UXTW(idx+off) == zext(idx)+off. coff/oadd may be
// NULL when the caller only needs base/idx.
static bool memfuse_match(MLIR_Context *ctx, MLIR_ValueHandle P,
                          MLIR_ValueHandle *base, MLIR_ValueHandle *idx,
                          MLIR_OpHandle *zop, MLIR_OpHandle *eop,
                          MLIR_OpHandle *pop, int64_t *coff,
                          MLIR_OpHandle *oadd) {
    if (coff) *coff = 0;
    if (oadd) *oadd = MLIR_INVALID_HANDLE;
    MLIR_OpHandle pdef = MLIR_GetValueDefiningOp(P);
    if (pdef == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(pdef), "llvm.inttoptr")) return false;
    if (MLIR_GetOpNumOperands(pdef) != 1) return false;
    MLIR_ValueHandle E = MLIR_GetOpOperand(pdef, 0);
    MLIR_OpHandle edef = MLIR_GetValueDefiningOp(E);
    if (edef == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(edef), "llvm.add")) return false;
    if (MLIR_GetOpNumOperands(edef) != 2) return false;
    // Shape 1: inttoptr(add(base, zext(idx))).
    if (match_base_zext(ctx, edef, base, idx, zop)) {
        *eop = edef;
        *pop = pdef;
        return true;
    }
    // Shape 2: inttoptr(add(add(base, zext(idx)), const)).
    MLIR_ValueHandle o[2] = { MLIR_GetOpOperand(edef, 0),
                              MLIR_GetOpOperand(edef, 1) };
    for (int k = 0; k < 2; k++) {
        MLIR_OpHandle cdef = MLIR_GetValueDefiningOp(o[k]);
        int64_t cv; uint8_t c64;
        if (cdef == MLIR_INVALID_HANDLE) continue;
        if (!const_int_val(ctx, cdef, &cv, &c64)) continue;
        MLIR_OpHandle xdef = MLIR_GetValueDefiningOp(o[1 - k]);
        if (!match_base_zext(ctx, xdef, base, idx, zop)) continue;
        *eop = xdef;        // inner base+zext add (skipped spine)
        *pop = pdef;
        if (coff) *coff = cv;
        if (oadd) *oadd = edef;   // outer +const add (skipped spine)
        return true;
    }
    return false;
}

// True if `op` materialises the address of the `__wasm_linmem_base` global
// (the i64 linear-memory base slot written by synth_start). Used to recognise
// the per-function base load so it can be pinned to the reserved x28 register
// instead of being reloaded from memory on every linmem access.
static bool is_linmem_base_addressof(MLIR_Context *ctx, MLIR_OpHandle op) {
    if (op == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.addressof")) return false;
    MLIR_AttributeHandle ga = MLIR_GetOpAttributeByName(op, "global_name");
    if (ga == MLIR_INVALID_HANDLE) return false;
    string g = MLIR_GetAttributeAsString(ctx, ga);
    if (g.size > 0 && g.str[0] == '@') { g.str++; g.size--; }
    const char *target = "__wasm_linmem_base";
    size_t tlen = 18; // strlen("__wasm_linmem_base")
    return g.size == tlen && memcmp(g.str, target, tlen) == 0;
}

// True if `op` is `llvm.load` of `addressof @__wasm_linmem_base` (the linmem
// base load). On success *aof receives the addressof op (which can then also be
// skipped from emission, being single-use feeding only this load).
static bool is_linmem_base_load(MLIR_Context *ctx, MLIR_OpHandle op,
                                MLIR_OpHandle *aof) {
    if (op == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(op), "llvm.load")) return false;
    if (MLIR_GetOpNumOperands(op) != 1) return false;
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(op, 0));
    if (!is_linmem_base_addressof(ctx, d)) return false;
    if (aof) *aof = d;
    return true;
}

// FP width of an f32/f64-typed value: 32, 64, or 0 if not a float type.
static int fp_width(MLIR_Context *ctx, MLIR_ValueHandle v) {
    string s = MLIR_GetTypeString(ctx, MLIR_GetValueType(v));
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 32;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 64;
    return 0;
}

// LLVM FCmpPredicate (oeq=1,ogt=2,oge=3,olt=4,ole=5,one=6) -> ARM cond code
// after FCMP. Ordered predicates only (matches the wasm backend). -1 if
// unsupported.
static int fcmp_pred_to_cond(int64_t pred) {
    switch (pred) {
    case 1: return 0;   // oeq -> EQ
    case 2: return 12;  // ogt -> GT
    case 3: return 10;  // oge -> GE
    case 4: return 4;   // olt -> MI
    case 5: return 9;   // ole -> LS
    case 6: return 1;   // one -> NE
    default: return -1;
    }
}

static unsigned a64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty);

static unsigned a64_align_up(unsigned x, unsigned a) {
    return (x + a - 1) & ~(a - 1);
}
static unsigned a64_struct_size(MLIR_Context *ctx, MLIR_TypeHandle ty);
static unsigned a64_type_align(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    if (MLIR_IsTypeLLVMArray(ty))
        return a64_type_align(ctx, MLIR_GetTypeLLVMArrayElement(ty));
    if (MLIR_IsTypeLLVMStruct(ty)) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
        unsigned ma = 1;
        for (size_t i = 0; i < nf; i++) {
            unsigned fa = a64_type_align(ctx, MLIR_GetTypeLLVMStructField(ty, i));
            if (fa > ma) ma = fa;
        }
        return ma;
    }
    unsigned sz = a64_type_size(ctx, ty);
    return sz ? sz : 1;
}
static unsigned a64_struct_size(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
    unsigned off = 0, max_align = 1;
    for (size_t i = 0; i < nf; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ty, i);
        unsigned fsz = a64_type_size(ctx, ft);
        unsigned fal = a64_type_align(ctx, ft);
        if (fsz == 0 || fal == 0) return 0;
        off = a64_align_up(off, fal);
        off += fsz;
        if (fal > max_align) max_align = fal;
    }
    return a64_align_up(off, max_align);
}
// Byte offset of the i-th field within an LLVM struct (native 64-bit layout).
static unsigned a64_struct_field_offset(MLIR_Context *ctx, MLIR_TypeHandle sty,
                                        size_t fld) {
    unsigned off = 0;
    for (size_t i = 0; i <= fld; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(sty, i);
        off = a64_align_up(off, a64_type_align(ctx, ft));
        if (i == fld) return off;
        off += a64_type_size(ctx, ft);
    }
    return off;
}
// Size in bytes of an LLVM-dialect type on aarch64 (pointers are 8 bytes).
static unsigned a64_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return 8;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return 8;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 8;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    if (MLIR_IsTypeLLVMArray(ty)) {
        unsigned esz = a64_type_size(ctx, MLIR_GetTypeLLVMArrayElement(ty));
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeLLVMArrayNumElements(ty);
    }
    if (MLIR_IsTypeLLVMStruct(ty)) return a64_struct_size(ctx, ty);
    return 0;
}

static bool func_has_body(MLIR_OpHandle fn) {
    return MLIR_GetOpNumRegions(fn) > 0 &&
           MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;
}
// Map an LLVM-dialect icmp predicate (eq=0, ne=1, slt=2, sle=3, sgt=4,
// sge=5, ult=6, ule=7, ugt=8, uge=9) to an AArch64 condition code. -1 if
// out of range.
static int icmp_pred_to_cond(int64_t p) {
    switch (p) {
        case 0: return 0;   // eq  -> EQ
        case 1: return 1;   // ne  -> NE
        case 2: return 11;  // slt -> LT
        case 3: return 13;  // sle -> LE
        case 4: return 12;  // sgt -> GT
        case 5: return 10;  // sge -> GE
        case 6: return 3;   // ult -> CC/LO
        case 7: return 9;   // ule -> LS
        case 8: return 8;   // ugt -> HI
        case 9: return 2;   // uge -> CS/HS
        default: return -1;
    }
}

// ---------------------------------------------------------------------------
// Value -> stack-slot map (open addressing keyed by the SSA value handle).
// Step 2 uses a trivial "spill everything" allocator: each SSA value lives
// in its own 8-byte frame slot; operands are reloaded into scratch
// registers (x9/x10/x11) for each instruction and the result is stored
// back immediately. This is always correct regardless of value count and
// survives calls (no value is kept live in a register across an op). The
// real linear-scan allocator + materialised a64ssa vreg tier arrive in
// Step 3, where keeping values in registers actually pays off.
// ---------------------------------------------------------------------------
typedef struct { uintptr_t key; int32_t slot; } SlotEnt;
typedef struct { SlotEnt *t; size_t cap; size_t n; Arena *arena; } SlotMap;

static size_t sm_hash(uintptr_t k) {
    k *= 0x9E3779B97F4A7C15ull;
    return (size_t)(k >> 32);
}
static void sm_grow(SlotMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    SlotEnt *nt = (SlotEnt *)arena_alloc(m->arena, ncap * sizeof(SlotEnt));
    memset(nt, 0, ncap * sizeof(SlotEnt));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->t[i].key == 0) continue;
        size_t j = sm_hash(m->t[i].key) & (ncap - 1);
        while (nt[j].key != 0) j = (j + 1) & (ncap - 1);
        nt[j] = m->t[i];
    }
    m->t = nt;
    m->cap = ncap;
}
static void sm_put(SlotMap *m, MLIR_ValueHandle k, int32_t slot) {
    if ((m->n + 1) * 4 >= m->cap * 3) sm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) return;
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].slot = slot;
    m->n++;
}
static bool sm_get(SlotMap *m, MLIR_ValueHandle k, int32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) { *out = m->t[i].slot; return true; }
        i = (i + 1) & mask;
    }
    return false;
}

// Increment the use count of value `k` (used to build a use-count map keyed by
// SSA value handle, reusing the SlotMap storage with `slot` = count).
static void uc_inc(SlotMap *m, MLIR_ValueHandle k) {
    if (!k) return;
    if ((m->n + 1) * 4 >= m->cap * 3) sm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) { m->t[i].slot++; return; }
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].slot = 1;
    m->n++;
}

// Reload an operand value from its frame slot into register `rd`.
static bool load_value(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                       MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_ldr_x_off(ctx, blk, rd, 31, (uint32_t)slot * 8u);
    return true;
}
// Store register `rd` into the frame slot for result value `v`.
static bool store_value(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                        MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_str_x_off(ctx, blk, rd, 31, (uint32_t)slot * 8u);
    return true;
}

// ---------------------------------------------------------------------------
// Constant rematerialization (the wmir HOME_CONST analogue). An integer
// `llvm.mlir.constant` never occupies a frame slot or register: instead a
// `mov`-immediate is re-emitted at every use site. Identical constants cost no
// spill/reload and add no register pressure. Float constants are excluded (they
// keep a frame slot). A remat is byte-identical to the value the constant's own
// lowering would have produced, so it is purely a code-placement change.
// ---------------------------------------------------------------------------
typedef struct { uintptr_t key; int64_t val; uint8_t is64; } ConstEnt;
typedef struct { ConstEnt *t; size_t cap; size_t n; Arena *arena; } ConstMap;

static void cm_grow(ConstMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    ConstEnt *nt = (ConstEnt *)arena_alloc(m->arena, ncap * sizeof(ConstEnt));
    memset(nt, 0, ncap * sizeof(ConstEnt));
    for (size_t i = 0; i < m->cap; i++) {
        if (m->t[i].key == 0) continue;
        size_t j = sm_hash(m->t[i].key) & (ncap - 1);
        while (nt[j].key != 0) j = (j + 1) & (ncap - 1);
        nt[j] = m->t[i];
    }
    m->t = nt;
    m->cap = ncap;
}
static void cm_put(ConstMap *m, MLIR_ValueHandle k, int64_t val, uint8_t is64) {
    if ((m->n + 1) * 4 >= m->cap * 3) cm_grow(m);
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) return;
        i = (i + 1) & mask;
    }
    m->t[i].key = (uintptr_t)k;
    m->t[i].val = val;
    m->t[i].is64 = is64;
    m->n++;
}
static bool cm_get(ConstMap *m, MLIR_ValueHandle k, int64_t *val, uint8_t *is64) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1;
    size_t i = sm_hash((uintptr_t)k) & mask;
    while (m->t[i].key != 0) {
        if (m->t[i].key == (uintptr_t)k) {
            *val = m->t[i].val; *is64 = m->t[i].is64; return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

// True if `op` is an integer llvm.mlir.constant eligible for remat. Mirrors the
// integer branch of the llvm.mlir.constant lowering so the re-emitted immediate
// is byte-identical to the value that branch would have produced.
static bool const_int_val(MLIR_Context *ctx, MLIR_OpHandle op,
                          int64_t *val, uint8_t *is64) {
    if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.constant")) return false;
    if (MLIR_GetOpNumResults(op) != 1) return false;
    MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
    if (fp_width(ctx, res) != 0) return false;
    MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
    if (va == MLIR_INVALID_HANDLE) return false;
    *val = MLIR_GetAttributeInteger(va);
    *is64 = type_is_gp64(ctx, res) ? 1 : 0;
    return true;
}

// Recursively record every remat-eligible integer constant in `cm`.
static void build_const_map(MLIR_Context *ctx, ConstMap *cm,
                            MLIR_BlockHandle block) {
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        int64_t v; uint8_t is64;
        if (const_int_val(ctx, op, &v, &is64))
            cm_put(cm, MLIR_GetOpResult(op, 0), v, is64);
        size_t ng = MLIR_GetOpNumRegions(op);
        for (size_t g = 0; g < ng; g++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, g);
            size_t nbk = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < nbk; b++)
                build_const_map(ctx, cm, MLIR_GetRegionBlock(rg, b));
        }
    }
}

// Integer binops whose AArch64 form is a plain 3-register op.
static bool simple_binop_optype(string on, MLIR_OpType *out) {
    if (name_eq(on, "llvm.add"))  { *out = OP_TYPE_AARCH64_ADD_REG; return true; }
    if (name_eq(on, "llvm.sub"))  { *out = OP_TYPE_AARCH64_SUB_REG; return true; }
    if (name_eq(on, "llvm.mul"))  { *out = OP_TYPE_AARCH64_MUL;     return true; }
    if (name_eq(on, "llvm.sdiv")) { *out = OP_TYPE_AARCH64_SDIV;    return true; }
    if (name_eq(on, "llvm.udiv")) { *out = OP_TYPE_AARCH64_UDIV;    return true; }
    if (name_eq(on, "llvm.and"))  { *out = OP_TYPE_AARCH64_AND_REG; return true; }
    if (name_eq(on, "llvm.or"))   { *out = OP_TYPE_AARCH64_ORR_REG; return true; }
    if (name_eq(on, "llvm.xor"))  { *out = OP_TYPE_AARCH64_EOR_REG; return true; }
    if (name_eq(on, "llvm.shl"))  { *out = OP_TYPE_AARCH64_LSL_REG; return true; }
    if (name_eq(on, "llvm.lshr")) { *out = OP_TYPE_AARCH64_LSR_REG; return true; }
    if (name_eq(on, "llvm.ashr")) { *out = OP_TYPE_AARCH64_ASR_REG; return true; }
    return false;
}

#define A64_FAIL(...) do { fprintf(stderr, __VA_ARGS__); return MLIR_INVALID_HANDLE; } while (0)

// ---------------------------------------------------------------------------
// Lowering context. The selector walks the structured `llvm`+`scf` body and
// emits a flat CFG of `aarch64` blocks into `out_region`. `cur` is the block
// currently being appended to; control-flow ops create new blocks and move
// `cur`. `ok` is cleared (with a stderr diagnostic) on the first failure.
// ---------------------------------------------------------------------------
typedef struct { string name; uint32_t off; uint32_t size; } GlobalEnt;
typedef struct { GlobalEnt *e; size_t n; } GlobalMap;

static bool gmap_get(GlobalMap *g, string nm, uint32_t *out) {
    for (size_t i = 0; i < g->n; i++)
        if (g->e[i].name.size == nm.size &&
            memcmp(g->e[i].name.str, nm.str, nm.size) == 0) {
            *out = g->e[i].off; return true;
        }
    return false;
}

// Look up a global by its NUL-terminated name, returning its byte offset in
// the data blob and its size. Used by `synth_start` to locate the wasm
// linear-memory template / argc / argv slots.
static bool gmap_get_cstr(GlobalMap *g, const char *nm,
                          uint32_t *off, uint32_t *size) {
    size_t len = strlen(nm);
    for (size_t i = 0; i < g->n; i++)
        if (g->e[i].name.size == len &&
            memcmp(g->e[i].name.str, nm, len) == 0) {
            if (off)  *off  = g->e[i].off;
            if (size) *size = g->e[i].size;
            return true;
        }
    return false;
}

// x27 anchor offset: the byte offset within __data that x27 points at. The
// scalar wasm globals (__wasm_linmem_base, __wasm_argc/argv, __wasm_g0 the hot
// shadow-stack pointer, __wasm_mem_pages) are clustered at the very end of the
// data blob (the big linmem template precedes them), at offsets ~4.39 MB which
// do NOT fit a scaled imm12 from offset 0. We anchor x27 at the first such
// global (__wasm_linmem_base) so the rest sit at tiny positive deltas that fit.
static uint32_t gfuse_anchor(GlobalMap *gm) {
    uint32_t a = 0;
    if (gm && gmap_get_cstr(gm, "__wasm_linmem_base", &a, NULL)) return a;
    return 0;
}

// If `ptr` is the result of a single-use `llvm.mlir.addressof @G` where G is a
// data global whose byte offset, relative to the x27 anchor, fits the unsigned
// scaled imm12 range for access size `sz` (1/4/8), return true and set *off to
// that relative offset. Such a load/store folds to a single `ldr/str [x27,
// #off]`: x27 is the reserved globals-cluster base set once in synth_start
// (mirrors x28 = linmem base), so the wasm-global access drops from adrp+add+ldr
// (3 insns) to one. `uc` is the value use-count map; only single-use addressofs
// fold (else a sibling use would still need it). `__wasm_linmem_base` itself is
// excluded — its loads are owned by the x28 base-pin (a64_is_linmem_base_load).
static bool gfuse_match(MLIR_Context *ctx, GlobalMap *gm, SlotMap *uc,
                        MLIR_ValueHandle ptr, unsigned sz, uint32_t anchor,
                        uint32_t *off) {
    if (gm == NULL) return false;
    if (sz != 1 && sz != 4 && sz != 8) return false;
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(ptr);
    if (d == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(d), "llvm.mlir.addressof")) return false;
    MLIR_AttributeHandle ga = MLIR_GetOpAttributeByName(d, "global_name");
    if (ga == MLIR_INVALID_HANDLE) return false;
    string gnm = MLIR_GetAttributeAsString(ctx, ga);
    if (gnm.size > 0 && gnm.str[0] == '@') { gnm.str++; gnm.size--; }
    if (gnm.size == 18 && memcmp(gnm.str, "__wasm_linmem_base", 18) == 0)
        return false;       // owned by the x28 base-pin
    uint32_t o = 0;
    if (!gmap_get(gm, gnm, &o)) return false;       // must be a data global
    if (o < anchor) return false;                   // below the x27 anchor
    uint32_t rel = o - anchor;
    if ((rel & (sz - 1u)) != 0) return false;       // natural alignment
    if (rel / sz > 4095u) return false;             // scaled imm12 range
    int32_t c;
    if (!sm_get(uc, ptr, &c) || c != 1) return false;  // addressof single-use
    *off = rel;
    return true;
}

typedef struct {
    MLIR_Context     *ctx;
    SlotMap          *sm;
    MLIR_RegionHandle out_region;
    MLIR_BlockHandle  cur;
    string            sym;
    uint32_t          frame_size;
    SlotMap          *am;          // alloca result value -> byte offset in frame
    uint32_t          slot_bytes;  // size of the slot region (allocas sit above)
    GlobalMap        *gm;          // global symbol name -> byte offset in __data
    size_t            n_fixed;     // number of fixed (named) params of this func
    SlotMap          *rm;          // value -> home physical reg (x19..x28); NULL = none
    ConstMap         *cm;          // value -> remat integer constant; NULL = none
    SlotMap          *skip;        // op-result values whose ops are not lowered
                                   // (folded into a fused branch); NULL = none
    MLIR_OpHandle     fuse_root;   // icmp op lowered "cmp-only" (flags feed the
                                   // terminator's b.cond); INVALID = none
    SlotMap          *memfuse;     // inttoptr result value (a linmem host ptr
                                   // = base + zext(i32 idx)) whose zext/add/
                                   // inttoptr spine is folded into a register-
                                   // offset load/store [base, Widx, UXTW]; the
                                   // spine ops are in `skip`. NULL = none.
    SlotMap          *shiftfuse;   // llvm.add result value whose single-use
                                   // shl/mul-by-2^k operand is folded into a
                                   // shifted-register `add Rd,Ra,Rx,lsl #amt`;
                                   // the shl/mul op is in `skip`. NULL = none.
                                   // Map value = (amt<<1)|shift_operand_index.
    SlotMap          *gfuse;       // scalar-global addressof result value -> byte
                                   // offset (relative to the x27 anchor); the
                                   // load/store folds to a single `ldr/str [x27,
                                   // #off]` (x27 = reserved globals-cluster base).
                                   // The addressof is in skip/memspine. NULL=none.
    bool              ok;
} LowerCtx;

// mov xd, xn  (orr xd, xzr, xn). 64-bit; safe for our full-register slot model.
static void emit_mov_reg(MLIR_Context *ctx, MLIR_BlockHandle blk,
                         uint8_t rd, uint8_t rn) {
    if (rd == rn) return;
    emit_3reg(ctx, blk, OP_TYPE_AARCH64_ORR_REG, rd, 31, rn, true);
}
// Single-instruction width extensions (replace movz+and / movz+lsl+asr triples).
// UXTW/SXTW are 2-register (rd,rn); SXTB/SXTH take sf to pick Wd/Xd.
static void emit_uxtw(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2] = { attr_i32(ctx, "rd", rd), attr_i32(ctx, "rn", rn) };
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_UXTW, a, 2));
}
static void emit_sxtw(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn) {
    MLIR_AttributeHandle a[2] = { attr_i32(ctx, "rd", rd), attr_i32(ctx, "rn", rn) };
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SXTW, a, 2));
}
static void emit_sxtb(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn, bool sf) {
    MLIR_AttributeHandle a[3] = { attr_i32(ctx, "rd", rd), attr_i32(ctx, "rn", rn),
                                  attr_b(ctx, "sf", sf) };
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SXTB, a, 3));
}
static void emit_sxth(MLIR_Context *ctx, MLIR_BlockHandle blk,
                      uint8_t rd, uint8_t rn, bool sf) {
    MLIR_AttributeHandle a[3] = { attr_i32(ctx, "rd", rd), attr_i32(ctx, "rn", rn),
                                  attr_b(ctx, "sf", sf) };
    MLIR_AppendBlockOp(ctx, blk, build_op(ctx, OP_TYPE_AARCH64_SXTH, a, 3));
}

// Allocation-aware operand read: returns the register holding `v`. If `v` has a
// home register, it is returned with no code emitted; otherwise `v` is loaded
// from its frame slot into `scratch`. With L->rm == NULL this is exactly the
// old `load_value(v, scratch)` path (byte-identical).
// Bool-returning remat-aware load of `v` into `dst` (home-reg mov, constant
// immediate, or frame-slot ldr). Returns false only when `v` has no home, no
// remat constant, and no frame slot (a genuine internal error). Used at sites
// that keep their own diagnostic.
static bool mat_into(LowerCtx *L, MLIR_ValueHandle v, uint8_t dst) {
    int32_t reg;
    if (L->rm && sm_get(L->rm, v, &reg)) {
        emit_mov_reg(L->ctx, L->cur, dst, (uint8_t)reg);
        return true;
    }
    int64_t cv; uint8_t c64;
    if (L->cm && cm_get(L->cm, v, &cv, &c64)) {
        emit_load_imm(L->ctx, L->cur, dst, (uint64_t)cv, c64 != 0);
        return true;
    }
    return load_value(L->ctx, L->cur, L->sm, v, dst);
}
static uint8_t use_val(LowerCtx *L, MLIR_ValueHandle v, uint8_t scratch) {
    int32_t reg;
    if (L->rm && sm_get(L->rm, v, &reg)) return (uint8_t)reg;
    int64_t cv; uint8_t c64;
    if (L->cm && cm_get(L->cm, v, &cv, &c64)) {
        emit_load_imm(L->ctx, L->cur, scratch, (uint64_t)cv, c64 != 0);
        return scratch;
    }
    if (!load_value(L->ctx, L->cur, L->sm, v, scratch)) {
        fprintf(stderr, "llvm->aarch64: undefined operand in '%.*s'\n",
                (int)L->sym.size, L->sym.str);
        L->ok = false;
    }
    return scratch;
}
// Ensure value `v` is materialised in register `dst` (mov from its home reg,
// constant immediate, or ldr from its frame slot).
static void load_into(LowerCtx *L, MLIR_ValueHandle v, uint8_t dst) {
    if (!mat_into(L, v, dst)) {
        fprintf(stderr, "llvm->aarch64: undefined operand in '%.*s'\n",
                (int)L->sym.size, L->sym.str);
        L->ok = false;
    }
}
// Allocation-aware result target: the register the caller should write `v`'s
// result into (its home register if allocated, else `scratch`).
static uint8_t def_val(LowerCtx *L, MLIR_ValueHandle v, uint8_t scratch) {
    int32_t reg;
    if (L->rm && sm_get(L->rm, v, &reg)) return (uint8_t)reg;
    return scratch;
}
// Finalise `v`'s result, produced in register `produced`: allocated -> move
// into the home register if needed; spilled -> store to the frame slot. With
// L->rm == NULL this is exactly the old `store_value(v, produced)`.
static void fin_val(LowerCtx *L, MLIR_ValueHandle v, uint8_t produced) {
    int32_t reg;
    if (L->rm && sm_get(L->rm, v, &reg)) {
        if ((uint8_t)reg != produced)
            emit_mov_reg(L->ctx, L->cur, (uint8_t)reg, produced);
        return;
    }
    if (!store_value(L->ctx, L->cur, L->sm, v, produced)) L->ok = false;
}

#define LFAIL(...) do { fprintf(stderr, __VA_ARGS__); L->ok = false; return; } while (0)

// True if `v` is a rematerialized integer constant that fits an AArch64 12-bit
// unsigned immediate (0..4095), so add/sub/cmp can fold it into their immediate
// form instead of materialising it into a scratch register. Negative or large
// constants fall back to the register form.
static bool operand_const_u12(LowerCtx *L, MLIR_ValueHandle v, uint16_t *imm) {
    int64_t cv; uint8_t c64;
    if (!L->cm || !cm_get(L->cm, v, &cv, &c64)) return false;
    uint64_t u = c64 ? (uint64_t)cv : (uint64_t)(uint32_t)cv;
    if (u > 0xFFFu) return false;
    *imm = (uint16_t)u;
    return true;
}

// Read the full integer value of a remat-constant operand (any magnitude).
static bool operand_const_any(LowerCtx *L, MLIR_ValueHandle v,
                              int64_t *cv, uint8_t *c64) {
    if (!L->cm) return false;
    return cm_get(L->cm, v, cv, c64);
}

// True if `v` is a rematerialized integer constant equal to zero. In that case
// the AArch64 zero register (wzr/xzr, encoding 31) can stand in for `v` as a
// DATA-SOURCE operand with no materialising `mov #0`. Only valid where register
// 31 denotes ZR -- i.e. a store transfer register (Rt) or an ALU/mov source --
// NOT where it denotes SP (an address base or add/sub Rn). Float constants are
// never in the ConstMap, so this only fires for integer zero.
static bool operand_is_zero(LowerCtx *L, MLIR_ValueHandle v) {
    int64_t cv; uint8_t c64;
    if (!L->cm || !cm_get(L->cm, v, &cv, &c64)) return false;
    return cv == 0;
}

// Operand read for a position where register 31 means ZR (store Rt / mov src):
// a zero remat-constant returns wzr/xzr (31) with no code; everything else is
// the ordinary `use_val`.
static uint8_t use_val_zr(LowerCtx *L, MLIR_ValueHandle v, uint8_t scratch) {
    if (operand_is_zero(L, v)) return 31;
    return use_val(L, v, scratch);
}

static MLIR_BlockHandle new_block(LowerCtx *L) {
    MLIR_BlockHandle b = MLIR_CreateBlock(L->ctx);
    MLIR_AppendRegionBlock(L->ctx, L->out_region, b);
    return b;
}

// True if `v`'s producing op already leaves the upper 32 bits of its 64-bit
// home/slot zero (all i32 results are produced in W-form data-processing
// instructions, W-form load/csel/cset, or the zero-extending casts, and spills
// are full 8-byte str/ldr). For such a source a `zext i32 -> i64` is a no-op, so
// it can be lowered as a plain copy (which copy-coalescing then elides) instead
// of an explicit UXTW. Conservative: an unrecognised or absent producer
// (function params, block args, sign-extends, index casts) keeps the UXTW.
static bool i32_src_zero_extended(MLIR_ValueHandle v) {
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(v);
    if (d == MLIR_INVALID_HANDLE) return false;
    string nm = MLIR_GetOpName(d);
    return name_eq(nm, "llvm.add")  || name_eq(nm, "llvm.sub")  ||
           name_eq(nm, "llvm.mul")  || name_eq(nm, "llvm.sdiv") ||
           name_eq(nm, "llvm.udiv") || name_eq(nm, "llvm.srem") ||
           name_eq(nm, "llvm.urem") || name_eq(nm, "llvm.and")  ||
           name_eq(nm, "llvm.or")   || name_eq(nm, "llvm.xor")  ||
           name_eq(nm, "llvm.shl")  || name_eq(nm, "llvm.lshr") ||
           name_eq(nm, "llvm.ashr") || name_eq(nm, "llvm.icmp") ||
           name_eq(nm, "llvm.zext") || name_eq(nm, "llvm.trunc")||
           name_eq(nm, "llvm.select") || name_eq(nm, "llvm.load") ||
           name_eq(nm, "llvm.inttoptr") || name_eq(nm, "llvm.ptrtoint");
}

// True when a `zext`/sub-width mask to `w` low bits is REDUNDANT because the
// source value already has every bit at/above position w clear. A byte/half/
// word `llvm.load` zero-extends to its declared width in hardware (ldrb/ldrh/
// ldr-w) and our slots keep that result zero-extended, so an i<=w load has bits
// >= w clear. Restricted to loads (unlike i32 W-form ops, generic i8/i16
// arithmetic does NOT clear the high bits, so masking those IS required).
static bool src_low_bits_clear(MLIR_Context *ctx, MLIR_ValueHandle v, int w) {
    MLIR_OpHandle d = MLIR_GetValueDefiningOp(v);
    if (d == MLIR_INVALID_HANDLE) return false;
    if (!name_eq(MLIR_GetOpName(d), "llvm.load")) return false;
    int lw = int_type_bits(ctx, v);
    return lw > 0 && lw <= w;
}

// Slot-to-slot copy (block-arg / phi resolution and yield forwarding).
static void copy_slot(LowerCtx *L, MLIR_ValueHandle src, MLIR_ValueHandle dst) {
    int64_t cv; uint8_t c64;
    if (L->cm && cm_get(L->cm, src, &cv, &c64)) {
        uint8_t r = 9;
        if (cv == 0) r = 31;                         // str xzr: no `mov #0`
        else emit_load_imm(L->ctx, L->cur, 9, (uint64_t)cv, c64 != 0);
        if (!store_value(L->ctx, L->cur, L->sm, dst, r))
            LFAIL("llvm->aarch64: undefined value in copy (%.*s)\n",
                  (int)L->sym.size, L->sym.str);
        return;
    }
    if (!load_value(L->ctx, L->cur, L->sm, src, 9) ||
        !store_value(L->ctx, L->cur, L->sm, dst, 9))
        LFAIL("llvm->aarch64: undefined value in copy (%.*s)\n",
              (int)L->sym.size, L->sym.str);
}

// Emit the phi edge copies for successor `s` of branch `term` into L->cur:
// each successor operand is copied into the matching block-argument slot of
// the destination block. Under the spill-everything model every value (incl.
// block args) owns a stable frame slot, and a destination block-arg slot is
// always disjoint from any source-value slot, so these sequential memory
// copies need no parallel-copy / swap handling.
//
// With block-argument register homing (alloc_regs_cfg), some destination block
// args live in a reserved callee-saved register and some sources may live in a
// register too, so the edge becomes a genuine parallel copy over abstract
// locations (homed reg / remat constant / frame slot). Register destinations
// can form cycles (a homed arg whose source is another homed arg); these are
// broken with scratch x9. Frame-slot destinations are never the source of
// another copy (block-arg slots are disjoint from every source value's slot),
// so only register destinations ever participate in a cycle.
typedef enum { LK_REG, LK_SLOT, LK_CONST, LK_NONE } LocKind;
typedef struct { LocKind k; uint8_t reg; int32_t slot; int64_t cval; uint8_t c64; } Loc;

static Loc edge_src_loc(LowerCtx *L, MLIR_ValueHandle v) {
    Loc loc; int32_t r; int64_t cv; uint8_t c64; int32_t slot;
    if (L->rm && sm_get(L->rm, v, &r)) { loc.k = LK_REG; loc.reg = (uint8_t)r; return loc; }
    if (L->cm && cm_get(L->cm, v, &cv, &c64)) {
        loc.k = LK_CONST; loc.cval = cv; loc.c64 = c64; return loc;
    }
    if (sm_get(L->sm, v, &slot)) { loc.k = LK_SLOT; loc.slot = slot; return loc; }
    loc.k = LK_NONE; return loc;
}
static Loc edge_dst_loc(LowerCtx *L, MLIR_ValueHandle v) {
    Loc loc; int32_t r, slot;
    if (L->rm && sm_get(L->rm, v, &r)) { loc.k = LK_REG; loc.reg = (uint8_t)r; return loc; }
    if (sm_get(L->sm, v, &slot)) { loc.k = LK_SLOT; loc.slot = slot; return loc; }
    loc.k = LK_NONE; return loc;
}
static bool loc_eq(Loc a, Loc b) {
    if (a.k != b.k) return false;
    if (a.k == LK_REG) return a.reg == b.reg;
    if (a.k == LK_SLOT) return a.slot == b.slot;
    return false;                                 // constants are leaf sources
}
// Materialise source `s` into destination `d` (no aliasing assumptions). Frame
// slots are addressed relative to `sb` (normally sp/x31, but the >8-arg call
// path passes the saved original sp after lowering sp for the stack-arg area).
static void emit_loc_move(LowerCtx *L, Loc s, Loc d, uint8_t sb) {
    if (loc_eq(s, d)) return;
    if (d.k == LK_REG) {
        if (s.k == LK_REG) emit_mov_reg(L->ctx, L->cur, d.reg, s.reg);
        else if (s.k == LK_CONST) emit_load_imm(L->ctx, L->cur, d.reg, (uint64_t)s.cval, s.c64 != 0);
        else emit_ldr_x_off(L->ctx, L->cur, d.reg, sb, (uint32_t)s.slot * 8u);
    } else {                                       // d.k == LK_SLOT
        uint8_t r;
        if (s.k == LK_REG) r = s.reg;
        else if (s.k == LK_CONST) {
            if (s.cval == 0) r = 31;                  // str xzr: no `mov #0`
            else { emit_load_imm(L->ctx, L->cur, 9, (uint64_t)s.cval, s.c64 != 0); r = 9; }
        }
        else { emit_ldr_x_off(L->ctx, L->cur, 9, sb, (uint32_t)s.slot * 8u); r = 9; }
        emit_str_x_off(L->ctx, L->cur, r, sb, (uint32_t)d.slot * 8u);
    }
}
// Resolve a parallel move: simultaneously move each src[k] into ds[k]. Register
// destinations can form cycles (broken with scratch x10); slot/const sources use
// x9 as transfer scratch (emit_loc_move). Slot locations are addressed relative
// to `sb`. `done[k]` may be pre-marked for identity moves. Used by the block-arg
// edge resolver, entry-param homing, and call-argument setup.
static void resolve_parallel(LowerCtx *L, Loc *src, Loc *ds, uint8_t *done,
                             size_t n, uint8_t sb) {
    size_t remaining = 0;
    for (size_t k = 0; k < n; k++)
        if (!done[k]) remaining++;
    while (remaining) {
        int progress = 0;
        for (size_t k = 0; k < n; k++) {
            if (done[k]) continue;
            int blocked = 0;                       // dst still read by a pending src?
            for (size_t j = 0; j < n; j++)
                if (!done[j] && j != k && loc_eq(src[j], ds[k])) { blocked = 1; break; }
            if (blocked) continue;
            emit_loc_move(L, src[k], ds[k], sb);
            done[k] = 1; remaining--; progress = 1;
        }
        if (progress) continue;
        // No progress => a cycle remains. Break one edge by saving its
        // destination into x10 (NOT x9 — emit_loc_move uses x9 as slot/const
        // scratch). Destinations may be registers OR frame slots, so load
        // whichever into x10 and redirect every pending reader of that
        // destination to x10. The broken cycle fully drains before the next
        // break, so a single x10 suffices even with multiple cycles.
        for (size_t k = 0; k < n; k++) {
            if (done[k]) continue;
            if (ds[k].k == LK_REG) emit_mov_reg(L->ctx, L->cur, 10, ds[k].reg);
            else emit_ldr_x_off(L->ctx, L->cur, 10, sb, (uint32_t)ds[k].slot * 8u);
            for (size_t j = 0; j < n; j++)
                if (!done[j] && loc_eq(src[j], ds[k])) {
                    src[j].k = LK_REG; src[j].reg = 10;
                }
            break;                                 // re-drain before next break
        }
    }
}
static void emit_edge_copies(LowerCtx *L, MLIR_OpHandle term, size_t s) {
    MLIR_BlockHandle dst = MLIR_GetOpSuccessor(term, s);
    size_t n = MLIR_GetOpNumSuccessorOperands(term, s);
    if (n == 0) return;
    // Fast path: no homing in this function -> all dsts are slots, no cycles,
    // byte-identical to the original sequential copy_slot loop.
    if (L->rm == NULL) {
        for (size_t k = 0; k < n; k++) {
            copy_slot(L, MLIR_GetOpSuccessorOperand(term, s, k),
                      MLIR_GetBlockArg(dst, k));
            if (!L->ok) return;
        }
        return;
    }
    Arena *ar = MLIR_GetArenaAllocator(L->ctx);
    Loc *src = (Loc *)arena_alloc(ar, n * sizeof(Loc));
    Loc *ds  = (Loc *)arena_alloc(ar, n * sizeof(Loc));
    uint8_t *done = (uint8_t *)arena_alloc(ar, n * sizeof(uint8_t));
    for (size_t k = 0; k < n; k++) {
        src[k] = edge_src_loc(L, MLIR_GetOpSuccessorOperand(term, s, k));
        ds[k]  = edge_dst_loc(L, MLIR_GetBlockArg(dst, k));
        if (src[k].k == LK_NONE || ds[k].k == LK_NONE)
            LFAIL("llvm->aarch64: undefined value in edge copy (%.*s)\n",
                  (int)L->sym.size, L->sym.str);
        done[k] = loc_eq(src[k], ds[k]) ? 1 : 0;   // identity copies are no-ops
    }
    resolve_parallel(L, src, ds, done, n, /*sb=*/31);
}

// Store an scf.yield's operands into the enclosing op's result slots.
static void store_yield(LowerCtx *L, MLIR_OpHandle yield, MLIR_OpHandle owner) {
    size_t n = MLIR_GetOpNumOperands(yield);
    for (size_t i = 0; i < n; i++) {
        copy_slot(L, MLIR_GetOpOperand(yield, i), MLIR_GetOpResult(owner, i));
        if (!L->ok) return;
    }
}

static void lower_op(LowerCtx *L, MLIR_OpHandle op);
static bool shiftfuse_decode(MLIR_Context *ctx, MLIR_OpHandle addop, SlotMap *sfm,
                             MLIR_ValueHandle *xv, MLIR_ValueHandle *av,
                             unsigned *amt);

// Lower every non-terminator op of `block` into the current CFG and return
// the block's terminator op (the caller interprets it).
static MLIR_OpHandle lower_block_ops(LowerCtx *L, MLIR_BlockHandle block) {
    size_t n = MLIR_GetBlockNumOps(block);
    if (n == 0) { L->ok = false; return MLIR_INVALID_HANDLE; }
    for (size_t i = 0; i + 1 < n; i++) {
        lower_op(L, MLIR_GetBlockOp(block, i));
        if (!L->ok) return MLIR_INVALID_HANDLE;
    }
    return MLIR_GetBlockOp(block, n - 1);
}

// Parse a `cases` attribute that prints as "array<i64: 1, 2, 3>".
static bool parse_cases(string cs, int64_t *out, size_t n) {
    size_t p = 0;
    while (p < cs.size && cs.str[p] != ':') p++;
    if (p < cs.size) p++;
    size_t got = 0;
    while (p < cs.size && got < n) {
        while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
        if (p >= cs.size || cs.str[p] == '>') break;
        int64_t sign = 1;
        if (cs.str[p] == '-') { sign = -1; p++; }
        int64_t v = 0;
        while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9')
            v = v * 10 + (cs.str[p++] - '0');
        out[got++] = sign * v;
    }
    return got == n;
}

// Parse "array<i32: v0, v1, ...>" into `out` (capacity `cap`); returns count.
static size_t parse_i32_array(string cs, int32_t *out, size_t cap) {
    size_t p = 0, got = 0;
    while (p < cs.size && cs.str[p] != ':') p++;
    if (p < cs.size) p++;
    while (p < cs.size && got < cap) {
        while (p < cs.size && (cs.str[p] == ' ' || cs.str[p] == ',')) p++;
        if (p >= cs.size || cs.str[p] == '>') break;
        int64_t sign = 1;
        if (cs.str[p] == '-') { sign = -1; p++; }
        int64_t v = 0;
        while (p < cs.size && cs.str[p] >= '0' && cs.str[p] <= '9')
            v = v * 10 + (cs.str[p++] - '0');
        out[got++] = (int32_t)(sign * v);
    }
    return got;
}

static void lower_scf_if(LowerCtx *L, MLIR_OpHandle op) {
    bool he = MLIR_GetOpNumRegions(op) >= 2 &&
              MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 1)) > 0;
    if (!mat_into(L, MLIR_GetOpOperand(op, 0), 9))
        LFAIL("llvm->aarch64: undefined scf.if condition\n");

    MLIR_BlockHandle then_blk = new_block(L);
    MLIR_BlockHandle else_blk = he ? new_block(L) : MLIR_INVALID_HANDLE;
    MLIR_BlockHandle end_blk  = new_block(L);

    emit_cbnz(L->ctx, L->cur, 9, false, then_blk);
    emit_b(L->ctx, L->cur, he ? else_blk : end_blk);

    L->cur = then_blk;
    MLIR_OpHandle yt = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0));
    if (!L->ok) return;
    store_yield(L, yt, op);
    if (!L->ok) return;
    emit_b(L->ctx, L->cur, end_blk);

    if (he) {
        L->cur = else_blk;
        MLIR_OpHandle ye = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0));
        if (!L->ok) return;
        store_yield(L, ye, op);
        if (!L->ok) return;
        emit_b(L->ctx, L->cur, end_blk);
    }
    L->cur = end_blk;
}

static void lower_scf_while(LowerCtx *L, MLIR_OpHandle op) {
    MLIR_BlockHandle before_src = MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0);
    MLIR_BlockHandle after_src  = MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 1), 0);
    size_t nb = MLIR_GetBlockNumArgs(before_src);

    // init operands -> before-region block-arg slots.
    for (size_t i = 0; i < nb; i++) {
        copy_slot(L, MLIR_GetOpOperand(op, i), MLIR_GetBlockArg(before_src, i));
        if (!L->ok) return;
    }
    MLIR_BlockHandle before_blk = new_block(L);
    emit_b(L->ctx, L->cur, before_blk);

    L->cur = before_blk;
    MLIR_OpHandle cterm = lower_block_ops(L, before_src);   // scf.condition
    if (!L->ok) return;
    size_t nf = MLIR_GetOpNumOperands(cterm) - 1;           // forwarded values
    if (!mat_into(L, MLIR_GetOpOperand(cterm, 0), 9))
        LFAIL("llvm->aarch64: undefined scf.condition value\n");

    MLIR_BlockHandle exit_store = new_block(L);
    MLIR_BlockHandle cont_blk   = new_block(L);
    MLIR_BlockHandle exit_blk   = new_block(L);
    emit_cbnz(L->ctx, L->cur, 9, false, cont_blk);          // cond true -> after
    emit_b(L->ctx, L->cur, exit_store);                     // cond false -> exit

    // exit: forwarded values become the scf.while results.
    L->cur = exit_store;
    for (size_t i = 0; i < nf; i++) {
        copy_slot(L, MLIR_GetOpOperand(cterm, i + 1), MLIR_GetOpResult(op, i));
        if (!L->ok) return;
    }
    emit_b(L->ctx, L->cur, exit_blk);

    // continue: forwarded values become the after-region block args.
    L->cur = cont_blk;
    for (size_t i = 0; i < nf; i++) {
        copy_slot(L, MLIR_GetOpOperand(cterm, i + 1), MLIR_GetBlockArg(after_src, i));
        if (!L->ok) return;
    }
    MLIR_OpHandle yterm = lower_block_ops(L, after_src);    // scf.yield
    if (!L->ok) return;
    for (size_t i = 0; i < nb; i++) {                       // yield -> before args
        copy_slot(L, MLIR_GetOpOperand(yterm, i), MLIR_GetBlockArg(before_src, i));
        if (!L->ok) return;
    }
    emit_b(L->ctx, L->cur, before_blk);

    L->cur = exit_blk;
}

static void lower_scf_index_switch(LowerCtx *L, MLIR_OpHandle op) {
    size_t n_regions = MLIR_GetOpNumRegions(op);
    if (n_regions < 1) LFAIL("llvm->aarch64: malformed scf.index_switch\n");
    size_t n_cases = n_regions - 1;
    int64_t case_vals[256];
    if (n_cases > 256) LFAIL("llvm->aarch64: scf.index_switch too large\n");
    if (n_cases > 0) {
        MLIR_AttributeHandle ca = MLIR_GetOpAttributeByName(op, "cases");
        if (ca == MLIR_INVALID_HANDLE ||
            !parse_cases(MLIR_GetAttributeAsString(L->ctx, ca), case_vals, n_cases))
            LFAIL("llvm->aarch64: cannot parse scf.index_switch cases\n");
    }
    if (!mat_into(L, MLIR_GetOpOperand(op, 0), 9))
        LFAIL("llvm->aarch64: undefined scf.index_switch selector\n");

    MLIR_BlockHandle end_blk = new_block(L);
    MLIR_BlockHandle def_blk = new_block(L);
    MLIR_BlockHandle case_blk[256];
    for (size_t i = 0; i < n_cases; i++) case_blk[i] = new_block(L);

    // Compare chain: cmp selector, #case; b.eq case_blk[i]. Selector stays
    // in x9 throughout this (single) block.
    for (size_t i = 0; i < n_cases; i++) {
        emit_load_imm(L->ctx, L->cur, 10, (uint64_t)case_vals[i], false);
        emit_cmp_reg(L->ctx, L->cur, 9, 10, false);
        emit_bcond(L->ctx, L->cur, /*EQ*/0, case_blk[i]);
    }
    emit_b(L->ctx, L->cur, def_blk);

    L->cur = def_blk;
    MLIR_OpHandle dt = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0));
    if (!L->ok) return;
    store_yield(L, dt, op);
    if (!L->ok) return;
    emit_b(L->ctx, L->cur, end_blk);

    for (size_t i = 0; i < n_cases; i++) {
        L->cur = case_blk[i];
        MLIR_OpHandle ct = lower_block_ops(L, MLIR_GetRegionBlock(MLIR_GetOpRegion(op, i + 1), 0));
        if (!L->ok) return;
        store_yield(L, ct, op);
        if (!L->ok) return;
        emit_b(L->ctx, L->cur, end_blk);
    }
    L->cur = end_blk;
}

// Lower a single non-terminator op into the current block / CFG.
static void lower_op(LowerCtx *L, MLIR_OpHandle op) {
    MLIR_Context     *ctx = L->ctx;
    MLIR_BlockHandle  blk = L->cur;
    SlotMap          *sm  = L->sm;
    string            on  = MLIR_GetOpName(op);
    MLIR_OpType       bt;

    if (name_eq(on, "llvm.mlir.constant")) {
        MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
        if (va == MLIR_INVALID_HANDLE || MLIR_GetOpNumResults(op) != 1)
            LFAIL("llvm->aarch64: malformed llvm.mlir.constant\n");
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, res);
        // Integer constants are rematerialized at each use (no slot/reg/def);
        // see ConstMap. Only float constants still need a def here.
        if (!fw) return;
        uint8_t rd = def_val(L, res, 9);
        // Float constant: materialise the IEEE bit pattern into a GP slot.
        double d = MLIR_GetAttributeFloat(va);
        uint64_t bits = 0;
        if (fw == 32) { float f = (float)d; uint32_t b32;
                        memcpy(&b32, &f, 4); bits = b32; }
        else          { memcpy(&bits, &d, 8); }
        emit_load_imm(ctx, blk, rd, bits, /*sf=*/true);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.add")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        MLIR_ValueHandle xv, av; unsigned amt;
        if (shiftfuse_decode(ctx, op, L->shiftfuse, &xv, &av, &amt)) {
            // add Rd, Ra, Rx, lsl #amt  (the shl/mul op is skipped).
            uint8_t ra = use_val(L, av, 9);
            uint8_t rx = use_val(L, xv, 10);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_add_reg_lsl(ctx, blk, rd, ra, rx, (uint8_t)amt, sf);
            fin_val(L, res, rd);
            return;
        }
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(op, 1);
        uint16_t imm;
        if (operand_const_u12(L, a1, &imm) || operand_const_u12(L, a0, &imm)) {
            // add is commutative: fold whichever operand is a small constant,
            // keeping the other in a register.
            MLIR_ValueHandle rv = operand_const_u12(L, a1, &imm) ? a0 : a1;
            uint8_t r = use_val(L, rv, 9);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_add_imm(ctx, blk, rd, r, imm, sf);
            fin_val(L, res, rd);
        } else {
            uint8_t r0 = use_val(L, a0, 9);
            uint8_t r1 = use_val(L, a1, 10);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, rd, r0, r1, sf);
            fin_val(L, res, rd);
        }

    } else if (name_eq(on, "llvm.sub")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(op, 1);
        uint16_t imm;
        if (operand_const_u12(L, a1, &imm)) {   // sub is not commutative
            uint8_t r0 = use_val(L, a0, 9);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_sub_imm(ctx, blk, rd, r0, imm, sf);
            fin_val(L, res, rd);
        } else {
            uint8_t r0 = use_val(L, a0, 9);
            uint8_t r1 = use_val(L, a1, 10);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_SUB_REG, rd, r0, r1, sf);
            fin_val(L, res, rd);
        }

    } else if (name_eq(on, "llvm.mul")) {
        // Strength-reduce mul by a constant: x*0 -> 0, x*1 -> copy,
        // x*2^k -> lsl #k. Other constants / non-constant keep register mul.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(op, 1);
        int64_t cv; uint8_t c64;
        MLIR_ValueHandle rv = MLIR_INVALID_HANDLE;   // non-constant operand
        bool have_c = false;
        if (!getenv("TINYC_NO_MUL_STRENGTH")) {
            if (operand_const_any(L, a1, &cv, &c64))      { rv = a0; have_c = true; }
            else if (operand_const_any(L, a0, &cv, &c64)) { rv = a1; have_c = true; }
        }
        uint64_t uc = have_c ? (sf ? (uint64_t)cv : (uint64_t)(uint32_t)cv) : 0;
        if (have_c && uc == 0) {
            uint8_t rd = def_val(L, res, 9);
            emit_load_imm(ctx, blk, rd, 0, sf);
            fin_val(L, res, rd);
        } else if (have_c && uc == 1) {
            uint8_t r = use_val(L, rv, 9);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            // i32 result must keep the upper 32 bits zero (the W-form mul did);
            // a 64-bit mov would not, breaking downstream zext elision.
            if (sf) emit_mov_reg(ctx, blk, rd, r);
            else    emit_uxtw(ctx, blk, rd, r);
            fin_val(L, res, rd);
        } else if (have_c && uc >= 2 && (uc & (uc - 1)) == 0) {
            uint8_t k = 0; while ((uc >> k) != 1u) k++;
            uint8_t r = use_val(L, rv, 9);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_shift_imm(ctx, blk, OP_TYPE_AARCH64_LSL_IMM, rd, r, k, sf);
            fin_val(L, res, rd);
        } else {
            uint8_t r0 = use_val(L, a0, 9);
            uint8_t r1 = use_val(L, a1, 10);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_MUL, rd, r0, r1, sf);
            fin_val(L, res, rd);
        }

    } else if (name_eq(on, "llvm.shl") || name_eq(on, "llvm.lshr") ||
               name_eq(on, "llvm.ashr")) {
        // Constant-amount shift -> immediate shift (no scratch mov of the count).
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        MLIR_ValueHandle a0 = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle a1 = MLIR_GetOpOperand(op, 1);
        uint8_t ds = sf ? 64 : 32;
        int64_t cv; uint8_t c64;
        bool have_c = !getenv("TINYC_NO_MUL_STRENGTH") &&
                      operand_const_any(L, a1, &cv, &c64);
        uint64_t sh = have_c ? (uint64_t)cv : 0;
        if (have_c && sh != 0 && sh < ds) {
            uint8_t r = use_val(L, a0, 9);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            MLIR_OpType it = name_eq(on, "llvm.shl") ? OP_TYPE_AARCH64_LSL_IMM
                           : name_eq(on, "llvm.lshr") ? OP_TYPE_AARCH64_LSR_IMM
                                                       : OP_TYPE_AARCH64_ASR_IMM;
            emit_shift_imm(ctx, blk, it, rd, r, (uint8_t)sh, sf);
            fin_val(L, res, rd);
        } else if (have_c && sh == 0) {
            uint8_t r = use_val(L, a0, 9);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            if (sf) emit_mov_reg(ctx, blk, rd, r);
            else    emit_uxtw(ctx, blk, rd, r);
            fin_val(L, res, rd);
        } else {
            uint8_t r0 = use_val(L, a0, 9);
            uint8_t r1 = use_val(L, a1, 10);
            if (!L->ok) return;
            uint8_t rd = def_val(L, res, 9);
            MLIR_OpType rt = name_eq(on, "llvm.shl") ? OP_TYPE_AARCH64_LSL_REG
                           : name_eq(on, "llvm.lshr") ? OP_TYPE_AARCH64_LSR_REG
                                                       : OP_TYPE_AARCH64_ASR_REG;
            emit_3reg(ctx, blk, rt, rd, r0, r1, sf);
            fin_val(L, res, rd);
        }

    } else if (simple_binop_optype(on, &bt)) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
        if (!L->ok) return;
        uint8_t rd = def_val(L, res, 9);
        emit_3reg(ctx, blk, bt, rd, r0, r1, sf);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.srem") || name_eq(on, "llvm.urem")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
        if (!L->ok) return;
        MLIR_OpType dt = name_eq(on, "llvm.srem") ? OP_TYPE_AARCH64_SDIV
                                                  : OP_TYPE_AARCH64_UDIV;
        uint8_t rd = def_val(L, res, 9);
        emit_3reg(ctx, blk, dt, 11, r0, r1, sf);
        emit_msub(ctx, blk, rd, 11, r1, r0, sf);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.icmp")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: llvm.icmp without predicate\n");
        int cond = icmp_pred_to_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0)
            LFAIL("llvm->aarch64: unsupported icmp predicate %lld\n",
                  (long long)MLIR_GetAttributeInteger(pa));
        bool sf = type_is_gp64(ctx, MLIR_GetOpOperand(op, 0));
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        if (op == L->fuse_root) {
            // Fused branch: emit only the comparison; the NZCV flags are
            // consumed by the terminator's b.cond. No cset, no result store.
            uint16_t imm;
            if (operand_const_u12(L, MLIR_GetOpOperand(op, 1), &imm)) {
                emit_cmp_imm(ctx, blk, r0, imm, sf);
            } else {
                uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
                if (!L->ok) return;
                emit_cmp_reg(ctx, blk, r0, r1, sf);
            }
            return;
        }
        uint8_t rd = def_val(L, res, 9);
        uint16_t imm;
        if (operand_const_u12(L, MLIR_GetOpOperand(op, 1), &imm)) {
            emit_cmp_imm(ctx, blk, r0, imm, sf);
        } else {
            uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
            if (!L->ok) return;
            emit_cmp_reg(ctx, blk, r0, r1, sf);
        }
        emit_cset(ctx, blk, rd, (uint8_t)cond, false);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.select")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_gp64(ctx, res);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
        uint8_t r2 = use_val(L, MLIR_GetOpOperand(op, 2), 11);
        if (!L->ok) return;
        uint8_t rd = def_val(L, res, 9);
        emit_cmp_imm(ctx, blk, r0, 0, false);
        emit_csel(ctx, blk, rd, r1, r2, /*NE*/1, sf);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.call")) {
        MLIR_AttributeHandle callee = MLIR_GetOpAttributeByName(op, "callee");
        bool is_indirect = (callee == MLIR_INVALID_HANDLE);
        MLIR_AttributeHandle vct = MLIR_GetOpAttributeByName(op, "var_callee_type");
        bool is_variadic = false;
        size_t n_fixed = 0;
        if (vct != MLIR_INVALID_HANDLE) {
            MLIR_TypeHandle ft = MLIR_GetAttributeTypeValue(vct);
            if (MLIR_GetTypeFunctionIsVarArg(ft)) {
                is_variadic = true;
                n_fixed = MLIR_GetTypeFunctionNumInputs(ft);
            }
        }
        // For an indirect call operand 0 is the function-pointer value; the
        // actual arguments start at operand 1. Materialise the target into
        // x16 (an intra-procedure scratch reg not used for arguments) BEFORE
        // any stack adjustment, since the slot is sp-relative.
        size_t arg_base = is_indirect ? 1 : 0;
        if (is_indirect) {
            if (!mat_into(L, MLIR_GetOpOperand(op, 0), 16))
                LFAIL("llvm->aarch64: undefined indirect callee in '%.*s'\n",
                      (int)L->sym.size, L->sym.str);
        }
        size_t no = MLIR_GetOpNumOperands(op) - arg_base;
        if (no > 64)
            LFAIL("llvm->aarch64: call with %zu args (>64 not supported)\n", no);
        string nm = (string){0};
        if (!is_indirect) {
            nm = MLIR_GetAttributeAsString(ctx, callee);
            if (nm.size > 0 && nm.str[0] == '@') { nm.str++; nm.size--; }
        }
        // n_reg = args passed in x0..x7. Darwin arm64 passes ALL variadic args
        // on the stack, so only the fixed params (capped at 8) go in registers.
        size_t n_reg = no;
        if (is_variadic)   n_reg = n_fixed < 8 ? n_fixed : 8;
        else if (no > 8)   n_reg = 8;
        if (n_reg > no)    n_reg = no;
        if (no == n_reg) {
            if (no) {
                Arena *ar = MLIR_GetArenaAllocator(ctx);
                Loc *src = (Loc *)arena_alloc(ar, no * sizeof(Loc));
                Loc *ds  = (Loc *)arena_alloc(ar, no * sizeof(Loc));
                uint8_t *done = (uint8_t *)arena_alloc(ar, no * sizeof(uint8_t));
                for (size_t k = 0; k < no; k++) {
                    src[k] = edge_src_loc(L, MLIR_GetOpOperand(op, arg_base + k));
                    if (src[k].k == LK_NONE)
                        LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                              (int)L->sym.size, L->sym.str);
                    ds[k].k = LK_REG; ds[k].reg = (uint8_t)k;
                    done[k] = loc_eq(src[k], ds[k]) ? 1 : 0;
                }
                resolve_parallel(L, src, ds, done, no, /*sb=*/31);
            }
            if (is_indirect) emit_blr(ctx, blk, 16);
            else             emit_bl(ctx, blk, nm);
        } else {
            // AAPCS64: first n_reg args in x0..x{n_reg-1}, the rest on the stack
            // at the call sp. Reserve a 16-aligned arg area below sp; address
            // frame-slot sources via a stable copy of the original sp in x11.
            // (x12 is now an allocator pool register that may hold a live arg,
            // and x17 is the large-offset scratch used by emit_*_x_off, so a big
            // stack-slot source read would clobber it -- x11 is free here: the
            // resolver uses x9/x10, emit_*_x_off uses x17, blr uses x16.)
            // Store the stack args FIRST (while x0..x7 still hold their incoming
            // values), then do the register args as a parallel move into
            // x0..x{n_reg-1} (sources may alias destination arg registers now
            // that x0..x8 are allocatable).
            size_t n_stack = no - n_reg;
            uint32_t area = (uint32_t)((n_stack * 8u + 15u) & ~15u);
            emit_sub_imm(ctx, blk, 31, 31, (uint16_t)area, true);
            emit_add_imm(ctx, blk, 11, 31, (uint16_t)area, true); // x11 = orig sp
            for (size_t k = n_reg; k < no; k++) {
                Loc s = edge_src_loc(L, MLIR_GetOpOperand(op, arg_base + k));
                if (s.k == LK_NONE)
                    LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                          (int)L->sym.size, L->sym.str);
                uint8_t r;
                if (s.k == LK_REG) r = s.reg;
                else if (s.k == LK_CONST) {
                    if (s.cval == 0) r = 31;          // str xzr: no `mov #0`
                    else { emit_load_imm(ctx, blk, 9, (uint64_t)s.cval, s.c64 != 0); r = 9; }
                } else {
                    emit_ldr_x_off(ctx, blk, 9, 11, (uint32_t)s.slot * 8u); r = 9;
                }
                emit_str_x_off(ctx, blk, r, 31, (uint32_t)(k - n_reg) * 8u);
            }
            if (n_reg) {
                Arena *ar = MLIR_GetArenaAllocator(ctx);
                Loc *src = (Loc *)arena_alloc(ar, n_reg * sizeof(Loc));
                Loc *ds  = (Loc *)arena_alloc(ar, n_reg * sizeof(Loc));
                uint8_t *done = (uint8_t *)arena_alloc(ar, n_reg * sizeof(uint8_t));
                for (size_t k = 0; k < n_reg; k++) {
                    src[k] = edge_src_loc(L, MLIR_GetOpOperand(op, arg_base + k));
                    if (src[k].k == LK_NONE)
                        LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                              (int)L->sym.size, L->sym.str);
                    ds[k].k = LK_REG; ds[k].reg = (uint8_t)k;
                    done[k] = loc_eq(src[k], ds[k]) ? 1 : 0;
                }
                resolve_parallel(L, src, ds, done, n_reg, /*sb=*/11);
            }
            if (is_indirect) emit_blr(ctx, blk, 16);
            else             emit_bl(ctx, blk, nm);
            emit_add_imm(ctx, blk, 31, 31, (uint16_t)area, true); // restore sp
        }
        if (MLIR_GetOpNumResults(op) == 1)
            fin_val(L, MLIR_GetOpResult(op, 0), 0);

    } else if (name_eq(on, "llvm.intr.vastart")) {
        // Darwin arm64: all variadic args are passed on the stack and va_list is
        // just a pointer to the first one. Incoming stack args of this function
        // sit at [x29, #16 + max(0, n_fixed-8)*8] (after the saved {fp,lr} pair
        // and any fixed stack args). Store that address into the va_list buffer.
        uint32_t base_off = 16u +
            (L->n_fixed > 8 ? (uint32_t)(L->n_fixed - 8) * 8u : 0u);
        uint8_t rp = use_val(L, MLIR_GetOpOperand(op, 0), 10);
        if (!L->ok) return;
        emit_add_imm(ctx, blk, 9, 29, (uint16_t)base_off, true);
        emit_str_x(ctx, blk, 9, rp, 0);

    } else if (name_eq(on, "llvm.intr.vaend")) {
        // No teardown needed for the stack-based va_list.

    } else if (name_eq(on, "llvm.mlir.undef") ||
               name_eq(on, "llvm.mlir.zero")) {
        // mlir.zero materialises a typed zero (null pointer / zeroed
        // aggregate scalar); mlir.undef is don't-care — 0 is fine for both.
        if (MLIR_GetOpNumResults(op) == 1) {
            MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
            uint8_t rd = def_val(L, res, 9);
            emit_load_imm(ctx, blk, rd, 0, false);
            fin_val(L, res, rd);
        }

    } else if (name_eq(on, "llvm.alloca")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int32_t off;
        if (!sm_get(L->am, res, &off))
            LFAIL("llvm->aarch64: alloca without a frame offset\n");
        uint32_t addr = L->slot_bytes + (uint32_t)off;
        uint8_t rd = def_val(L, res, 9);
        if (addr <= 4095) {
            emit_add_imm(ctx, blk, rd, 31, (uint16_t)addr, true);
        } else {
            // Offset exceeds the 12-bit ADD immediate: materialise it in a
            // scratch register and add it to a copy of sp.
            emit_add_imm(ctx, blk, rd, 31, 0, true);            // rd = sp
            emit_load_imm(ctx, blk, 10, (uint64_t)addr, true);  // x10 = addr
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, rd, rd, 10, true);
        }
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.getelementptr")) {
        MLIR_ValueHandle res  = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle base = MLIR_GetOpOperand(op, 0);
        MLIR_AttributeHandle eta = MLIR_GetOpAttributeByName(op, "elem_type");
        MLIR_AttributeHandle ria = MLIR_GetOpAttributeByName(op, "rawConstantIndices");
        if (eta == MLIR_INVALID_HANDLE || ria == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: getelementptr missing elem_type/indices\n");
        MLIR_TypeHandle elem_ty = MLIR_GetAttributeTypeValue(eta);
        int32_t cidx[64];
        size_t n_idx = parse_i32_array(MLIR_GetAttributeAsString(ctx, ria),
                                       cidx, 64);
        if (n_idx == 0)
            LFAIL("llvm->aarch64: getelementptr with no indices\n");
        load_into(L, base, 9);   // running address in x9
        if (!L->ok) return;
        size_t op_idx = 1;
        MLIR_TypeHandle cur_ty = elem_ty;
        for (size_t i = 0; i < n_idx; i++) {
            bool is_dyn = (cidx[i] == (int32_t)0x80000000);
            unsigned stride;
            int64_t  cst = 0;
            if (i == 0) {
                stride = a64_type_size(ctx, elem_ty);
            } else if (MLIR_IsTypeLLVMStruct(cur_ty)) {
                if (is_dyn)
                    LFAIL("llvm->aarch64: dynamic struct index in gep\n");
                unsigned foff = a64_struct_field_offset(ctx, cur_ty, (size_t)cidx[i]);
                cur_ty = MLIR_GetTypeLLVMStructField(cur_ty, (size_t)cidx[i]);
                emit_gep_const_off(ctx, blk, (int64_t)foff);
                continue;
            } else if (MLIR_IsTypeLLVMArray(cur_ty)) {
                MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(cur_ty);
                stride = a64_type_size(ctx, et);
                cur_ty = et;
            } else {
                LFAIL("llvm->aarch64: gep into non-aggregate type\n");
            }
            if (stride == 0) continue;
            if (is_dyn) {
                if (op_idx >= MLIR_GetOpNumOperands(op))
                    LFAIL("llvm->aarch64: gep dynamic index operand missing\n");
                uint8_t ri = use_val(L, MLIR_GetOpOperand(op, op_idx++), 10);
                if (!L->ok) return;
                // Power-of-2 stride folds into a single shifted-add; otherwise
                // materialise the stride and multiply.
                if (stride != 0 && (stride & (stride - 1)) == 0) {
                    unsigned sh = 0; while ((1u << sh) != stride) sh++;
                    emit_add_reg_lsl(ctx, blk, 9, 9, ri, (uint8_t)sh, true);
                } else {
                    emit_load_imm(ctx, blk, 11, (uint64_t)stride, true);
                    emit_3reg(ctx, blk, OP_TYPE_AARCH64_MUL, 10, ri, 11, true);
                    emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, 9, 9, 10, true);
                }
            } else if ((cst = cidx[i]) != 0) {
                emit_gep_const_off(ctx, blk, cst * (int64_t)stride);
            }
        }
        fin_val(L, res, 9);

    } else if (name_eq(on, "llvm.store")) {
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 1);
        unsigned sz = a64_type_size(ctx, MLIR_GetValueType(val));
        if (sz != 1 && sz != 4 && sz != 8)
            LFAIL("llvm->aarch64: unsupported store size %u\n", sz);
        int32_t mtag;
        if (L->memfuse && sm_get(L->memfuse, ptr, &mtag)) {
            // ptr = inttoptr(base + zext(idx_i32) [+ mtag]): one register-offset
            // store str Rt, [base, Widx, UXTW]. With a static offset (mtag>0) the
            // offset folds into the 32-bit index: add Wtmp,Widx,#mtag first. The
            // zext/add(/+const)/inttoptr spine is skipped.
            MLIR_ValueHandle mbase, midx; MLIR_OpHandle z, e, p;
            if (!memfuse_match(ctx, ptr, &mbase, &midx, &z, &e, &p, NULL, NULL))
                LFAIL("llvm->aarch64: memfuse store spine mismatch\n");
            uint8_t rv = use_val_zr(L, val, 9);
            uint8_t rn = use_val(L, mbase, 10);
            uint8_t rm = use_val(L, midx, 11);
            if (!L->ok) return;
            if (mtag != 0) {                         // fold static offset into idx
                emit_add_imm(ctx, blk, 11, rm, (uint16_t)mtag, false);
                rm = 11;
            }
            MLIR_OpType t = sz == 1 ? OP_TYPE_AARCH64_STRB_REG
                          : sz == 4 ? OP_TYPE_AARCH64_STR_W_REG
                                    : OP_TYPE_AARCH64_STR_X_REG;
            emit_ldst_x_reg(ctx, blk, t, rv, rn, rm);
            return;
        }
        int32_t goff;
        if (L->gfuse && sm_get(L->gfuse, ptr, &goff)) {
            // ptr = addressof @global: one [x27,#off] store (x27 = globals base).
            uint8_t rv = use_val_zr(L, val, 9);
            if (!L->ok) return;
            MLIR_OpType t = sz == 1 ? OP_TYPE_AARCH64_STRB_IMM
                          : sz == 4 ? OP_TYPE_AARCH64_STR_W
                                    : OP_TYPE_AARCH64_STR_X;
            emit_ldst_x(ctx, blk, t, rv, 27, (uint32_t)goff);
            return;
        }
        uint8_t rv = use_val_zr(L, val, 9);
        uint8_t rp = use_val(L, ptr, 10);
        if (!L->ok) return;
        emit_mem_store(ctx, blk, rv, rp, sz);

    } else if (name_eq(on, "llvm.load")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 0);
        unsigned sz = a64_type_size(ctx, MLIR_GetValueType(res));
        if (sz != 1 && sz != 4 && sz != 8)
            LFAIL("llvm->aarch64: unsupported load size %u\n", sz);
        int32_t mtag;
        if (L->memfuse && sm_get(L->memfuse, ptr, &mtag)) {
            MLIR_ValueHandle mbase, midx; MLIR_OpHandle z, e, p;
            if (!memfuse_match(ctx, ptr, &mbase, &midx, &z, &e, &p, NULL, NULL))
                LFAIL("llvm->aarch64: memfuse load spine mismatch\n");
            uint8_t rn = use_val(L, mbase, 10);
            uint8_t rm = use_val(L, midx, 11);
            if (!L->ok) return;
            if (mtag != 0) {                         // fold static offset into idx
                emit_add_imm(ctx, blk, 11, rm, (uint16_t)mtag, false);
                rm = 11;
            }
            uint8_t rd = def_val(L, res, 9);
            MLIR_OpType t = sz == 1 ? OP_TYPE_AARCH64_LDRB_REG
                          : sz == 4 ? OP_TYPE_AARCH64_LDR_W_REG
                                    : OP_TYPE_AARCH64_LDR_X_REG;
            emit_ldst_x_reg(ctx, blk, t, rd, rn, rm);
            fin_val(L, res, rd);
            return;
        }
        int32_t goff;
        if (L->gfuse && sm_get(L->gfuse, ptr, &goff)) {
            // ptr = addressof @global: one [x27,#off] load (x27 = globals base).
            uint8_t rd = def_val(L, res, 9);
            MLIR_OpType t = sz == 1 ? OP_TYPE_AARCH64_LDRB_IMM
                          : sz == 4 ? OP_TYPE_AARCH64_LDR_W
                                    : OP_TYPE_AARCH64_LDR_X;
            emit_ldst_x(ctx, blk, t, rd, 27, (uint32_t)goff);
            fin_val(L, res, rd);
            return;
        }
        uint8_t rp = use_val(L, ptr, 10);
        if (!L->ok) return;
        uint8_t rd = def_val(L, res, 9);
        emit_mem_load(ctx, blk, rd, rp, sz);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.ptrtoint") || name_eq(on, "llvm.inttoptr")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        int w = int_type_bits(ctx, res);   // 0 for ptr (64-bit, no mask)
        uint8_t rd = def_val(L, res, 9);
        if (w == 32) {
            emit_uxtw(ctx, blk, rd, r0);
        } else if (w > 0 && w < 64) {
            emit_and_imm_lowbits(ctx, blk, rd, r0, (uint8_t)w, true);
        } else {
            emit_mov_reg(ctx, blk, rd, r0);
        }
        fin_val(L, res, rd);

    } else if (name_eq(on, "arith.index_cast") ||
               name_eq(on, "arith.index_castui")) {
        // index <-> integer: a plain copy in our 64-bit slot model.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        uint8_t rd = def_val(L, res, 9);
        emit_mov_reg(ctx, blk, rd, r0);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.trunc") || name_eq(on, "llvm.zext")) {
        // Both keep the low N bits (our values are stored zero-extended); a
        // mask to the destination (trunc) / source (zext) width suffices.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = name_eq(on, "llvm.trunc") ? int_type_bits(ctx, res)
                                          : int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        uint8_t rd = def_val(L, res, 9);
        if (w == 32) {
            // zext i32->i64: UXTW (mov wd,wn) hardware-zeroes the top half. When
            // the source is already zero-extended (its producer wrote a W-form
            // result), the extension is a no-op: emit a plain copy so copy-
            // coalescing can elide it entirely.
            if (name_eq(on, "llvm.zext") &&
                i32_src_zero_extended(MLIR_GetOpOperand(op, 0)))
                emit_mov_reg(ctx, blk, rd, r0);
            else
                emit_uxtw(ctx, blk, rd, r0);        // mov wd,wn -> hardware zero-ext
        } else if (w > 0 && w < 64) {
            // A `zext` whose source already has bits >= w clear (a narrow load)
            // needs no mask at all: emit a plain copy for copy-coalescing to
            // elide. Otherwise mask the low w bits in ONE `and rd,r0,#(1<<w)-1`.
            if (name_eq(on, "llvm.zext") &&
                src_low_bits_clear(ctx, MLIR_GetOpOperand(op, 0), w))
                emit_mov_reg(ctx, blk, rd, r0);
            else
                emit_and_imm_lowbits(ctx, blk, rd, r0, (uint8_t)w, true);
        } else {
            emit_mov_reg(ctx, blk, rd, r0);
        }
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.sext")) {
        // Sign-extend from the source width via a left-then-arithmetic-right
        // shift pair in 64-bit registers.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        uint8_t rd = def_val(L, res, 9);
        if (w == 8) {
            emit_sxtb(ctx, blk, rd, r0, true);
        } else if (w == 16) {
            emit_sxth(ctx, blk, rd, r0, true);
        } else if (w == 32) {
            emit_sxtw(ctx, blk, rd, r0);
        } else if (w > 0 && w < 64) {
            emit_load_imm(ctx, blk, 10, (uint64_t)(64 - w), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_LSL_REG, rd, r0, 10, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ASR_REG, rd, rd, 10, true);
        } else {
            emit_mov_reg(ctx, blk, rd, r0);
        }
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.mlir.addressof")) {
        // Materialise the address of a module-level global: ADRP+ADD against
        // the native data section, offset by the global's slot.
        MLIR_AttributeHandle ga = MLIR_GetOpAttributeByName(op, "global_name");
        if (ga == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: addressof missing global_name\n");
        string gnm = MLIR_GetAttributeAsString(ctx, ga);
        if (gnm.size > 0 && gnm.str[0] == '@') { gnm.str++; gnm.size--; }
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t rd = def_val(L, res, 9);
        uint32_t off = 0;
        if (L->gm && gmap_get(L->gm, gnm, &off)) {
            string tgt = str_from_cstr_view("linmem_template");
            emit_adrp_data(ctx, blk, rd, tgt, off);
            emit_add_data_lo(ctx, blk, rd, rd, tgt, off);
        } else {
            // Not a data global — materialise the address of a function
            // symbol (ADRP+ADD resolved against the function's text address
            // by the Mach-O encoder).
            emit_adrp_data(ctx, blk, rd, gnm, 0);
            emit_add_data_lo(ctx, blk, rd, rd, gnm, 0);
        }
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.fadd") || name_eq(on, "llvm.fsub") ||
               name_eq(on, "llvm.fmul") || name_eq(on, "llvm.fdiv")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, res);
        const char *kind = name_eq(on, "llvm.fadd") ? "fadd"
                         : name_eq(on, "llvm.fsub") ? "fsub"
                         : name_eq(on, "llvm.fmul") ? "fmul" : "fdiv";
        if (fw == 0) LFAIL("llvm->aarch64: %.*s non-float result\n",
                           (int)on.size, on.str);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
        if (!L->ok) return;
        emit_fmov_gp_v(ctx, blk, /*to_v=*/true, true, 0, r0);
        emit_fmov_gp_v(ctx, blk, /*to_v=*/true, true, 1, r1);
        emit_fp_binop(ctx, blk, kind, fw, 0, 0, 1);
        uint8_t rd = def_val(L, res, 9);
        emit_fmov_gp_v(ctx, blk, /*to_v=*/false, true, rd, 0);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.fneg") || name_eq(on, "llvm.intr.sqrt")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, res);
        const char *kind = name_eq(on, "llvm.fneg") ? "fneg" : "fsqrt";
        if (fw == 0) LFAIL("llvm->aarch64: %.*s non-float result\n",
                           (int)on.size, on.str);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        emit_fmov_gp_v(ctx, blk, true, true, 0, r0);
        emit_fp_unop(ctx, blk, kind, fw, 0, 0);
        uint8_t rd = def_val(L, res, 9);
        emit_fmov_gp_v(ctx, blk, false, true, rd, 0);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.fcmp")) {
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: fcmp missing predicate\n");
        int cond = fcmp_pred_to_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0)
            LFAIL("llvm->aarch64: unsupported fcmp predicate\n");
        int fw = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        if (fw == 0) LFAIL("llvm->aarch64: fcmp non-float operand\n");
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        uint8_t r1 = use_val(L, MLIR_GetOpOperand(op, 1), 10);
        if (!L->ok) return;
        emit_fmov_gp_v(ctx, blk, true, true, 0, r0);
        emit_fmov_gp_v(ctx, blk, true, true, 1, r1);
        emit_fcmp(ctx, blk, fw, 0, 1);
        uint8_t rd = def_val(L, res, 9);
        emit_cset(ctx, blk, rd, (uint8_t)cond, false);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.fptosi") || name_eq(on, "llvm.fptoui")) {
        // FP -> int. Result lands in a GP reg; narrow results must be masked
        // back to the slot's zero-extended invariant.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sign = name_eq(on, "llvm.fptosi");
        int src_w = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        if (src_w == 0)
            LFAIL("llvm->aarch64: fptosi/fptoui non-float source\n");
        int rw = int_type_bits(ctx, res);
        int dst_w = rw > 32 ? 64 : 32;
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        emit_fmov_gp_v(ctx, blk, true, true, 0, r0);
        uint8_t rd = def_val(L, res, 9);
        emit_fp_cvt(ctx, blk, "f2i", src_w, dst_w, sign, /*rd GP*/rd, /*rn V*/0);
        if (rw > 0 && rw < 32) {
            emit_and_imm_lowbits(ctx, blk, rd, rd, (uint8_t)rw, true);
        }
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.sitofp") || name_eq(on, "llvm.uitofp")) {
        // int -> FP. Slots hold zero-extended ints: uitofp can read the full
        // 64-bit slot directly; sitofp must first sign-extend the source width.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sign = name_eq(on, "llvm.sitofp");
        int sw = int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        int dst_w = fp_width(ctx, res);
        if (dst_w == 0)
            LFAIL("llvm->aarch64: sitofp/uitofp non-float result\n");
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        if (sign && sw > 0 && sw < 64) {
            emit_load_imm(ctx, blk, 10, (uint64_t)(64 - sw), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_LSL_REG, 9, r0, 10, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ASR_REG, 9, 9, 10, true);
            r0 = 9;
        }
        emit_fp_cvt(ctx, blk, "i2f", /*src_w=*/64, dst_w, sign, /*rd V*/0, /*rn GP*/r0);
        uint8_t rd = def_val(L, res, 9);
        emit_fmov_gp_v(ctx, blk, false, true, rd, 0);
        fin_val(L, res, rd);

    } else if (name_eq(on, "llvm.fpext") || name_eq(on, "llvm.fptrunc")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int src_w = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        int dst_w = fp_width(ctx, res);
        if (src_w == 0 || dst_w == 0)
            LFAIL("llvm->aarch64: fpext/fptrunc non-float type\n");
        uint8_t r0 = use_val(L, MLIR_GetOpOperand(op, 0), 9);
        if (!L->ok) return;
        emit_fmov_gp_v(ctx, blk, true, true, 0, r0);
        emit_fp_cvt(ctx, blk, "f2f", src_w, dst_w, false, 0, 0);
        uint8_t rd = def_val(L, res, 9);
        emit_fmov_gp_v(ctx, blk, false, true, rd, 0);
        fin_val(L, res, rd);

    } else if (name_eq(on, "scf.if")) {
        lower_scf_if(L, op);
    } else if (name_eq(on, "scf.while")) {
        lower_scf_while(L, op);
    } else if (name_eq(on, "scf.index_switch")) {
        lower_scf_index_switch(L, op);

    } else {
        LFAIL("llvm->aarch64: unsupported op '%.*s' in '%.*s'\n",
              (int)on.size, on.str, (int)L->sym.size, L->sym.str);
    }
}

// Recursively assign an 8-byte frame slot to every block arg and op result
// reachable in `block` (including nested scf regions).
static void assign_slots_block(MLIR_Context *ctx, SlotMap *sm,
                               MLIR_BlockHandle block, int32_t *nslots) {
    size_t na = MLIR_GetBlockNumArgs(block);
    for (size_t i = 0; i < na; i++)
        sm_put(sm, MLIR_GetBlockArg(block, i), (*nslots)++);
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        int64_t cv; uint8_t c64;
        // Integer constants are rematerialized at each use (no frame slot).
        if (const_int_val(ctx, op, &cv, &c64)) continue;
        size_t nr = MLIR_GetOpNumResults(op);
        for (size_t r = 0; r < nr; r++)
            sm_put(sm, MLIR_GetOpResult(op, r), (*nslots)++);
        size_t ng = MLIR_GetOpNumRegions(op);
        for (size_t g = 0; g < ng; g++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, g);
            size_t nbk = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < nbk; b++)
                assign_slots_block(ctx, sm, MLIR_GetRegionBlock(rg, b), nslots);
        }
    }
}

// Post-pass for the global allocator path (alloc_regs_global already filled
// `sm` with reused slots for spilled values and `rm` with register homes). Walk
// every value and give a UNIQUE slot ONLY to those the allocator did not place:
//   * already in `sm` (spilled, reused slot) or in `rm` (register-homed): skip;
//   * integer constants: rematerialized, no slot;
//   * anything else: a fresh unique slot. This covers (a) entry params when
//     TINYC_NO_PARAM_HOME disables param homing (the allocator never numbers
//     them) and (b) memfuse-spine / base-pin values in the `skip` set, which
//     emit no code but, per b7ba0fb, must occupy isolated storage that never
//     ALIASES a live spilled slot (sharing a single scratch slot among them
//     re-triggers a latent miscompile -> infinite recursion in self-host).
static void assign_post_slots(MLIR_Context *ctx, SlotMap *sm, SlotMap *rm,
                              MLIR_BlockHandle block, int32_t *nslots) {
    int32_t dummy;
    size_t na = MLIR_GetBlockNumArgs(block);
    for (size_t i = 0; i < na; i++) {
        MLIR_ValueHandle a = MLIR_GetBlockArg(block, i);
        if (sm_get(sm, a, &dummy) || (rm && sm_get(rm, a, &dummy))) continue;
        sm_put(sm, a, (*nslots)++);
    }
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        int64_t cv; uint8_t c64;
        if (const_int_val(ctx, op, &cv, &c64)) continue;
        size_t nr = MLIR_GetOpNumResults(op);
        for (size_t r = 0; r < nr; r++) {
            MLIR_ValueHandle rv = MLIR_GetOpResult(op, r);
            if (sm_get(sm, rv, &dummy) || (rm && sm_get(rm, rv, &dummy))) continue;
            sm_put(sm, rv, (*nslots)++);
        }
    }
}

// Recursive prepass: assign each llvm.alloca result a byte offset within the
// alloca region (placed above the slot region). Returns total alloca bytes.
static bool collect_allocas(MLIR_Context *ctx, SlotMap *am,
                            MLIR_BlockHandle block, uint32_t *bytes) {
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        if (name_eq(MLIR_GetOpName(op), "llvm.alloca") &&
            MLIR_GetOpNumResults(op) == 1) {
            MLIR_AttributeHandle eta = MLIR_GetOpAttributeByName(op, "elem_type");
            if (eta == MLIR_INVALID_HANDLE) return false;
            MLIR_TypeHandle et = MLIR_GetAttributeTypeValue(eta);
            unsigned esz = a64_type_size(ctx, et);
            if (esz == 0) return false;
            int64_t cnt = 1;
            if (MLIR_GetOpNumOperands(op) >= 1) {
                MLIR_OpHandle cd = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(op, 0));
                if (cd == MLIR_INVALID_HANDLE ||
                    !name_eq(MLIR_GetOpName(cd), "llvm.mlir.constant"))
                    return false;
                cnt = MLIR_GetAttributeInteger(
                          MLIR_GetOpAttributeByName(cd, "value"));
            }
            unsigned al = a64_type_align(ctx, et);
            if (al < 8) al = 8;
            *bytes = a64_align_up(*bytes, al);
            sm_put(am, MLIR_GetOpResult(op, 0), (int32_t)*bytes);
            *bytes += (uint32_t)(esz * cnt);
        }
        size_t ng = MLIR_GetOpNumRegions(op);
        for (size_t g = 0; g < ng; g++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, g);
            size_t nbk = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < nbk; b++)
                if (!collect_allocas(ctx, am, MLIR_GetRegionBlock(rg, b), bytes))
                    return false;
        }
    }
    return true;
}

typedef struct { uint32_t slot_off; uint32_t target_off; } PtrReloc;

// ---------------------------------------------------------------------------
// Pre-scan the module for `llvm.mlir.global` ops. Each global gets an
// 8-aligned byte offset in a single native data blob (emitted as one
// `aarch64.data_init` record into the __linmem_template data section). The
// name->offset map drives `llvm.mlir.addressof` lowering. Scalar globals
// store their integer/float `value`; aggregate globals (e.g. string
// literals) store the raw bytes of a string `value`; pointer globals
// initialised with the address of another global (region: addressof +
// return) become an internal data->data pointer reloc (emitted as an
// `aarch64.data_ptr`). Absent values are zero-filled. Returns false on an
// unsupported global.
static bool collect_globals(MLIR_Context *ctx, MLIR_BlockHandle mb,
                            GlobalMap *gm, uint8_t **blob_out,
                            uint32_t *blob_len_out,
                            PtrReloc **relocs_out, size_t *n_relocs_out) {
    size_t nops = MLIR_GetBlockNumOps(mb);
    size_t cap = 0;
    for (size_t i = 0; i < nops; i++)
        if (name_eq(MLIR_GetOpName(MLIR_GetBlockOp(mb, i)), "llvm.mlir.global"))
            cap++;
    gm->e = cap ? (GlobalEnt *)calloc(cap, sizeof(GlobalEnt)) : NULL;
    gm->n = 0;

    // Pending pointer relocs hold the target global NAME (it may be a
    // forward reference), resolved to an offset after all globals are placed.
    typedef struct { uint32_t slot_off; string target; } PendingReloc;
    PendingReloc *pend = cap ? (PendingReloc *)calloc(cap, sizeof(PendingReloc))
                             : NULL;
    size_t n_pend = 0;

    uint8_t *blob = NULL;
    uint32_t blen = 0, bcap = 0;

    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.global")) continue;

        MLIR_AttributeHandle na = MLIR_GetOpAttributeByName(op, "sym_name");
        MLIR_AttributeHandle ta = MLIR_GetOpAttributeByName(op, "global_type");
        if (na == MLIR_INVALID_HANDLE || ta == MLIR_INVALID_HANDLE) {
            free(blob); free(pend); return false;
        }
        string name = MLIR_GetAttributeString(na);
        unsigned sz = a64_type_size(ctx, MLIR_GetAttributeTypeValue(ta));
        if (sz == 0) { free(blob); free(pend); return false; }

        uint32_t off = (blen + 7u) & ~7u;
        uint32_t need = off + sz;
        if (need > bcap) {
            uint32_t ncap = bcap ? bcap * 2 : 64;
            while (ncap < need) ncap *= 2;
            blob = (uint8_t *)realloc(blob, ncap);
            memset(blob + bcap, 0, ncap - bcap);
            bcap = ncap;
        }
        memset(blob + blen, 0, off - blen);   // alignment padding
        memset(blob + off, 0, sz);

        MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
        if (va != MLIR_INVALID_HANDLE) {
            MLIR_AttrKind vk = MLIR_GetAttributeKind(va);
            if (vk == MLIR_ATTR_KIND_STRING) {
                string s = MLIR_GetAttributeAsString(ctx, va);
                size_t n = s.size < sz ? s.size : sz;
                memcpy(blob + off, s.str, n);
            } else if (vk == MLIR_ATTR_KIND_INTEGER ||
                       vk == MLIR_ATTR_KIND_BOOL) {
                uint64_t v = (uint64_t)MLIR_GetAttributeInteger(va);
                for (unsigned b = 0; b < sz && b < 8; b++)
                    blob[off + b] = (uint8_t)(v >> (8 * b));
            } else if (vk == MLIR_ATTR_KIND_FLOAT) {
                double d = MLIR_GetAttributeFloat(va);
                uint64_t bits = 0;
                if (sz == 4) { float f = (float)d; uint32_t b32;
                               memcpy(&b32, &f, 4); bits = b32; }
                else if (sz == 8) { memcpy(&bits, &d, 8); }
                else { free(blob); free(pend); return false; }
                for (unsigned b = 0; b < sz && b < 8; b++)
                    blob[off + b] = (uint8_t)(bits >> (8 * b));
            } else {
                free(blob); free(pend); return false;
            }
        } else if (MLIR_GetOpNumRegions(op) > 0 &&
                   MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(op, 0)) > 0) {
            // Region-init pointer global: scan for addressof @target + return.
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(MLIR_GetOpRegion(op, 0), 0);
            size_t nb = MLIR_GetBlockNumOps(blk);
            string target = (string){0};
            bool only_trivial = true;
            for (size_t bi = 0; bi < nb; bi++) {
                MLIR_OpHandle bop = MLIR_GetBlockOp(blk, bi);
                string opn = MLIR_GetOpName(bop);
                if (name_eq(opn, "llvm.mlir.addressof")) {
                    MLIR_AttributeHandle ga =
                        MLIR_GetOpAttributeByName(bop, "global_name");
                    if (ga == MLIR_INVALID_HANDLE) { free(blob); free(pend); return false; }
                    string ts = MLIR_GetAttributeAsString(ctx, ga);
                    if (ts.size && ts.str[0] == '@') { ts.str++; ts.size--; }
                    target = ts;
                } else if (!name_eq(opn, "llvm.mlir.zero") &&
                           !name_eq(opn, "llvm.mlir.undef") &&
                           !name_eq(opn, "llvm.return")) {
                    // Anything other than zero/undef/return/addressof means a
                    // non-trivial initializer we can't lower here.
                    only_trivial = false;
                }
            }
            if (target.str) {
                // Pointer global initialised to the address of another global.
                if (sz < 8) { free(blob); free(pend); return false; }
                pend[n_pend].slot_off = off;
                pend[n_pend].target   = target;
                n_pend++;
            } else if (!only_trivial) {
                // Unsupported non-zero aggregate/scalar initializer.
                free(blob); free(pend); return false;
            }
            // else: zeroinitializer / undef — blob is already zero-filled.
        }
        blen = need;

        gm->e[gm->n].name = name;
        gm->e[gm->n].off  = off;
        gm->e[gm->n].size = sz;
        gm->n++;
    }

    // Resolve pending pointer relocs now that every global has an offset.
    PtrReloc *relocs = n_pend ? (PtrReloc *)calloc(n_pend, sizeof(PtrReloc))
                              : NULL;
    for (size_t k = 0; k < n_pend; k++) {
        uint32_t toff = 0;
        if (!gmap_get(gm, pend[k].target, &toff)) {
            free(blob); free(pend); free(relocs); return false;
        }
        relocs[k].slot_off   = pend[k].slot_off;
        relocs[k].target_off = toff;
    }
    free(pend);

    *blob_out = blob;
    *blob_len_out = blen;
    *relocs_out = relocs;
    *n_relocs_out = n_pend;
    return true;
}


// ---------------------------------------------------------------------------
// Multi-block CFG lowering. Used for functions whose body is an explicit
// `cf.br` / `cf.cond_br` control-flow graph (e.g. the `--from-wasm` lifter
// output) rather than single-block structured `scf`. The WASM-sourced CFG
// has an empty operand stack at every block boundary (all cross-block state
// flows through `llvm.alloca` locals), so branch ops carry NO block-arg
// operands — there is no phi / edge-copy to resolve. Each source block maps
// 1:1 to an output `aarch64` block; non-terminator ops reuse `lower_op`.
// ---------------------------------------------------------------------------
static MLIR_BlockHandle map_block(MLIR_BlockHandle *src, MLIR_BlockHandle *dst,
                                  size_t n, MLIR_BlockHandle s) {
    for (size_t i = 0; i < n; i++)
        if (src[i] == s) return dst[i];
    return MLIR_INVALID_HANDLE;
}

// Callee-saved register pool for the linear-scan allocator: x19..x26. x27 and
// x28 are RESERVED (x28 = linear-memory base, x27 = data/globals base), each
// set once in synth_start, then preserved for the lifetime of the program —
// every function treats them as callee-saved and never clobbers them because
// they are outside the pool, so the base values flow through the whole call
// tree with no per-function reload). Linmem loads/stores read base from x28
// directly (see base-pinning in select_func_cfg); wasm-global loads/stores read
// the data base from x27 (see global-pinning in select_func_cfg). Mirrors the
// deleted WMIR backend, which dedicated x28 to linmem and x27 to wasm globals.
#define A64_NSAVED 8
#define A64_SAVE_BASE 19

// Register-allocation pool. Caller-saved registers come FIRST so non-call-
// crossing values prefer them (they need no prologue save/restore); callee-
// saved come LAST so they are reached only under pressure, kept available for
// call-crossing values that can live nowhere else.
//   pool index 0..8   -> x0..x8    (caller-saved: AAPCS arg/result regs)
//   pool index 9..12  -> x12..x15  (caller-saved: temporaries)
//   pool index 13..20 -> x19..x26  (callee-saved: x27=data base, x28=linmem
//                                   base both reserved out of the pool)
// x0..x8 are clobbered by every `bl`, so a call-crossing value may not live
// there, but the vast majority of values are short-lived and call-free, so
// adding them roughly doubles the usable pool and cuts spill traffic. The
// call-arg / entry-param / call-result paths route through a general parallel-
// move resolver (resolve_parallel), so an operand homed in an arg register
// never clobbers another live arg. x9/x10/x11 stay reserved as lowering scratch
// (and x16/x17 for the indirect-call target and >8-arg stack-pointer save), so
// they are NOT in the pool. Mirrors the deleted WMIR backend's 21-register pool.
// a64_pool_reg() maps pool index -> phys reg by arithmetic (no brace-init array,
// which the self-hosting tinyC front end does not lower).
#define A64_NPOOL  21
#define A64_NCALLER 13
// Number of block-arg home registers reserved from the top of the callee-saved
// range (pool 20 -> x26, 19 -> x25, ... down). Tuned empirically (frozen bench):
// 6 homes the hottest loop-carried args. The edge parallel-move resolver
// (emit_edge_copies) handles arbitrary cycles, so any count is correct.
#define A64_NHOME  6
static inline uint8_t a64_pool_reg(int pk) {
    if (pk < 9)  return (uint8_t)pk;                  // x0..x8
    if (pk < 13) return (uint8_t)(12 + (pk - 9));     // x12..x15
    return (uint8_t)(A64_SAVE_BASE + (pk - A64_NCALLER)); // x19..x26
}

// Save (or, with restore=true, reload) the callee-saved registers selected by
// `mask` (bit k = x(19+k)) to/from the per-frame save area at byte offset
// `base` and up. Emitted just after the prologue and just before each epilogue.
static void emit_callee_saves(MLIR_Context *ctx, MLIR_BlockHandle blk,
                              uint32_t mask, uint32_t base, bool restore) {
    uint32_t off = base;
    for (int k = 0; k < A64_NSAVED; k++) {
        if (!(mask & (1u << k))) continue;
        uint8_t reg = (uint8_t)(A64_SAVE_BASE + k);
        if (restore) emit_ldr_x_off(ctx, blk, reg, 31, off);
        else         emit_str_x_off(ctx, blk, reg, 31, off);
        off += 8;
    }
}

// Returns true and sets *src to operand 0 if `op` is a single-source cast that
// lowers to instructions reading only operand 0 and writing the result (so the
// result may safely share operand 0's home register: every branch in these
// handlers is correct when rd == r0, and the pure-move cases collapse to a
// `mov rd,rd` that emit_mov_reg drops entirely). Drives copy coalescing.
static bool cast_src(MLIR_OpHandle op, MLIR_ValueHandle *src) {
    string nm = MLIR_GetOpName(op);
    if (name_eq(nm, "llvm.inttoptr") || name_eq(nm, "llvm.ptrtoint") ||
        name_eq(nm, "arith.index_cast") || name_eq(nm, "arith.index_castui") ||
        name_eq(nm, "llvm.trunc") || name_eq(nm, "llvm.zext") ||
        name_eq(nm, "llvm.sext") || name_eq(nm, "llvm.bitcast")) {
        if (MLIR_GetOpNumOperands(op) != 1) return false;
        *src = MLIR_GetOpOperand(op, 0);
        return true;
    }
    return false;
}

// ===========================================================================
// Global (whole-function) linear-scan register allocator.
//
// A faithful port of the deleted WMIR allocator (mlir_wmir_regalloc.c) over
// this lowering's existing pool: caller-saved x12..x15 then callee-saved
// x19..x28 (a64_pool_reg). Unlike the block-local alloc_regs_cfg it keeps
// cross-block and loop-carried values (block args / induction vars /
// accumulators) resident in registers across their whole live range via
// iterative dataflow liveness + loop-depth-weighted spilling, then resolves
// block-arg homes at edges through the existing parallel-move resolver
// (emit_edge_copies). The win over block-local is exactly the hot loop
// values that block-local forces to memory.
//
// This path reuses the lowering's pre-assigned frame slots (assign_slots_block
// gives EVERY value a slot), so there is no slot allocation here: a value is
// either homed in a register (written to `rm`) or left in its slot. "Eviction"
// simply un-homes a value (clears its tentative register); the final `rm` is
// written once from the per-value home decision, so a partially-built scan
// never corrupts state.
//
// Memory is malloc/free'd (NOT arena) so the per-function liveness bitsets are
// reclaimed; a size gate (below) bails to block-local for very large functions
// (the self-host's giant dispatch function) so the wasm32 buddy heap never
// OOMs. Bailing returns A64_GLOBAL_BAIL with `rm` untouched.
// ===========================================================================
#define A64_GLOBAL_BAIL 0xFFFFFFFFu

static inline size_t a64_bs_words(size_t n) { return (n + 63u) / 64u; }
static inline void a64_bs_set(uint64_t *bs, uint32_t i) { bs[i >> 6] |= (1ull << (i & 63u)); }
static inline void a64_bs_clear(uint64_t *bs, uint32_t i) { bs[i >> 6] &= ~(1ull << (i & 63u)); }
static inline bool a64_bs_test(const uint64_t *bs, uint32_t i) { return (bs[i >> 6] >> (i & 63u)) & 1u; }
static inline void a64_bs_zero(uint64_t *bs, size_t n) { memset(bs, 0, a64_bs_words(n) * 8u); }
static inline bool a64_bs_equal(const uint64_t *a, const uint64_t *b, size_t n) {
    return memcmp(a, b, a64_bs_words(n) * 8u) == 0;
}
static inline void a64_bs_or(uint64_t *d, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t w = a64_bs_words(n); for (size_t i = 0; i < w; i++) d[i] = a[i] | b[i];
}
static inline void a64_bs_diff(uint64_t *d, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t w = a64_bs_words(n); for (size_t i = 0; i < w; i++) d[i] = a[i] & ~b[i];
}

// Open-addressing handle -> uint32 hash (malloc-backed, pre-sized, no grow).
// Key 0 (MLIR_INVALID_HANDLE) marks empty; handles are non-zero. Probe order
// is irrelevant to output determinism (only get/set by key are used).
typedef struct { MLIR_ValueHandle *k; uint32_t *v; size_t cap; } A64HHash;
static uint32_t a64_hh_hash(MLIR_ValueHandle v) {
    uint32_t h = (uint32_t)v ^ (uint32_t)(v >> 16);
    h ^= h >> 15; h *= 2654435761u; h ^= h >> 13; return h;
}
static void a64_hh_init(A64HHash *m, size_t want) {
    size_t c = 16; while (c < want * 2u) c *= 2;
    m->cap = c;
    m->k = (MLIR_ValueHandle *)calloc(c, sizeof(*m->k));
    m->v = (uint32_t *)malloc(c * sizeof(*m->v));
}
static void a64_hh_put(A64HHash *m, MLIR_ValueHandle key, uint32_t val) {
    size_t mask = m->cap - 1, p = (size_t)a64_hh_hash(key) & mask;
    while (m->k[p] != 0) { if (m->k[p] == key) { m->v[p] = val; return; } p = (p + 1) & mask; }
    m->k[p] = key; m->v[p] = val;
}
static bool a64_hh_get(const A64HHash *m, MLIR_ValueHandle key, uint32_t *out) {
    if (m->cap == 0) return false;
    size_t mask = m->cap - 1, p = (size_t)a64_hh_hash(key) & mask;
    while (m->k[p] != 0) { if (m->k[p] == key) { *out = m->v[p]; return true; } p = (p + 1) & mask; }
    return false;
}
static void a64_hh_free(A64HHash *m) { free(m->k); free(m->v); m->k = NULL; m->v = NULL; m->cap = 0; }

// True if `op` is a folded linmem spine op (single result in memspine): it
// emits no code, so its operand uses must NOT extend liveness (base+idx are
// re-injected at the fused load/store instead).
static bool a64_is_spine(MLIR_OpHandle op, SlotMap *memspine) {
    if (!memspine || MLIR_GetOpNumResults(op) != 1) return false;
    int32_t mt;
    return sm_get(memspine, MLIR_GetOpResult(op, 0), &mt);
}
// If `op` is a memfuse'd load/store, output its base+idx register operands.
static bool a64_memfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *memfuse,
                             MLIR_ValueHandle *base, MLIR_ValueHandle *idx) {
    if (!memfuse) return false;
    string opn = MLIR_GetOpName(op);
    size_t nop = MLIR_GetOpNumOperands(op);
    MLIR_ValueHandle P = MLIR_INVALID_HANDLE;
    if (name_eq(opn, "llvm.load") && nop >= 1)        P = MLIR_GetOpOperand(op, 0);
    else if (name_eq(opn, "llvm.store") && nop >= 2)  P = MLIR_GetOpOperand(op, 1);
    if (P == MLIR_INVALID_HANDLE) return false;
    int32_t mt;
    if (!sm_get(memfuse, P, &mt)) return false;
    MLIR_OpHandle z, e, p;
    return memfuse_match(ctx, P, base, idx, &z, &e, &p, NULL, NULL);
}

// Shift-fusion: does the `llvm.add` `addop` have an operand defined by a
// single-use shift-by-constant (`llvm.shl(x, C)` or `llvm.mul(x, 2^C)`) that can
// fold into one `add Rd, Ra, Rx, lsl #amt`? On success sets *sidx (the add
// operand index holding the shift, 0 or 1) and *amt (the shift amount). The
// shift source x must be non-constant, the amount in [1, 31]/[1, 63], and the
// shift result single-use (uc==1) so its op can be skipped without losing a
// reader.
static bool shiftfuse_match(MLIR_Context *ctx, MLIR_OpHandle addop, SlotMap *uc,
                            int *sidx, unsigned *amt) {
    if (MLIR_GetOpNumResults(addop) != 1 || MLIR_GetOpNumOperands(addop) < 2)
        return false;
    unsigned maxsh = type_is_gp64(ctx, MLIR_GetOpResult(addop, 0)) ? 63u : 31u;
    for (int k = 1; k >= 0; k--) {        // prefer operand 1 (matches add bias)
        MLIR_OpHandle sop = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(addop, k));
        if (!sop || MLIR_GetOpNumOperands(sop) < 2) continue;
        string opn = MLIR_GetOpName(sop);
        MLIR_OpHandle d0 = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(sop, 0));
        MLIR_OpHandle d1 = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(sop, 1));
        int64_t c0, c1; uint8_t b0, b1;
        bool c0k = d0 && const_int_val(ctx, d0, &c0, &b0);
        bool c1k = d1 && const_int_val(ctx, d1, &c1, &b1);
        unsigned a;
        if (name_eq(opn, "llvm.shl")) {
            if (!c1k || c0k) continue;            // shl(x, C), x non-constant
            if (c1 < 1 || (uint64_t)c1 > maxsh) continue;
            a = (unsigned)c1;
        } else if (name_eq(opn, "llvm.mul")) {
            int64_t cc;                           // mul(x, 2^a) or mul(2^a, x)
            if (c1k && !c0k)      cc = c1;
            else if (c0k && !c1k) cc = c0;
            else continue;
            if (cc < 2 || (cc & (cc - 1)) != 0) continue;   // power of 2, >= 2
            a = 0; while (((int64_t)1 << a) != cc) a++;
            if (a > maxsh) continue;
        } else continue;
        int32_t u;
        if (!sm_get(uc, MLIR_GetOpResult(sop, 0), &u) || u != 1) continue;
        *sidx = k; *amt = a;
        return true;
    }
    return false;
}

// Decode a recorded shift-fused add (sfm value = (amt<<1)|sidx). Outputs the
// shift source x, the other addend a, and the shift amount. x is the
// NON-constant operand of the shift/mul op (the constant may be on either side
// for `mul`). Used by both the emitter and the register allocators.
static bool shiftfuse_decode(MLIR_Context *ctx, MLIR_OpHandle addop, SlotMap *sfm,
                             MLIR_ValueHandle *xv, MLIR_ValueHandle *av,
                             unsigned *amt) {
    if (!sfm || MLIR_GetOpNumResults(addop) != 1) return false;
    int32_t v;
    if (!sm_get(sfm, MLIR_GetOpResult(addop, 0), &v)) return false;
    int sidx = v & 1;
    if (amt) *amt = (unsigned)(v >> 1);
    if (av)  *av  = MLIR_GetOpOperand(addop, 1 - sidx);
    MLIR_OpHandle sop = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(addop, sidx));
    if (!sop || MLIR_GetOpNumOperands(sop) < 2) return false;
    MLIR_ValueHandle o0 = MLIR_GetOpOperand(sop, 0);
    int64_t cv; uint8_t c64;
    MLIR_OpHandle d0 = MLIR_GetValueDefiningOp(o0);
    if (d0 && const_int_val(ctx, d0, &cv, &c64))
        *xv = MLIR_GetOpOperand(sop, 1);   // const is operand 0 (mul(C,x))
    else
        *xv = o0;                          // shl(x,C) / mul(x,C): x is operand 0
    return true;
}

// If `op` is a shift-fused add, output the shift-source operand x so the
// allocator re-injects it as a live use at the add (the skipped shl/mul op no
// longer carries x's liveness). Mirrors a64_memfuse_uses.
static bool a64_shiftfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op,
                               SlotMap *shiftfuse, MLIR_ValueHandle *xv) {
    return shiftfuse_decode(ctx, op, shiftfuse, xv, NULL, NULL);
}

static uint32_t alloc_regs_global(MLIR_Context *ctx, MLIR_RegionHandle reg,
                                  size_t n_blocks, SlotMap *rm,
                                  SlotMap *memfuse, SlotMap *memspine,
                                  SlotMap *shiftfuse,
                                  SlotMap *sm, int32_t *out_nslots) {
    *out_nslots = 0;
    if (n_blocks == 0) return A64_GLOBAL_BAIL;

    // --- 0. Count register-candidate values; bail on nested regions. Entry
    //        block args (incoming params) ARE allocated like any other value:
    //        the prologue moves them from x0..x7 (or the caller's stack) into
    //        their assigned home register, so a param used throughout the
    //        function (e.g. a parser/lexer state pointer) stays resident instead
    //        of being reloaded from its slot at every use. Param homing is in
    //        the existing pool (x12..x28), none of which alias the incoming
    //        x0..x7, so the prologue move is a safe sequential copy (no parallel
    //        move needed). Disable via TINYC_NO_PARAM_HOME for A/B measurement.
    //        Constants and folded spine results get no value number
    //        (rematerialized / fused). Positions still advance per op. ---
    bool home_params = !getenv("TINYC_NO_PARAM_HOME");
    size_t nv_max = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        if (b > 0 || home_params) nv_max += MLIR_GetBlockNumArgs(bk);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (MLIR_GetOpNumRegions(op) > 0) return A64_GLOBAL_BAIL;
            nv_max += MLIR_GetOpNumResults(op);
        }
    }
    if (nv_max == 0) return 0;
    // Memory gate: bound the liveness bitsets (n_blocks * bw words, four of
    // them) so a truly pathological function falls back to block-local rather
    // than exhausting the wasm32 heap during self-host (stage1 runs this backend
    // under wasmtime). The deleted WMIR allocator ran the SAME O(n_blocks*bw)
    // liveness with NO gate and self-hosted, so these thresholds are set high
    // enough that every real tinyC function (largest ~2628 blocks / ~60K values
    // / ~60MB peak, freed per function) gets global allocation + slot reuse.
    {
        size_t bwg = a64_bs_words(nv_max);
        if (nv_max > 100000u || (uint64_t)n_blocks * (uint64_t)bwg > 4000000ull)
            return A64_GLOBAL_BAIL;
    }

    // Half-open live intervals. With one integer position per op, an operand's
    // last read and a result's def land on the SAME position, so the strict
    // active-list retirement (last_use < first_pos) cannot retire the dying
    // operand before the consumer is allocated -- they falsely overlap and
    // demand two registers for what is logically one value (the induction
    // `%next = %a + 1` / loop-carried case). Numbering operand reads at 2g+1,
    // result defs at 2g+2, and block args at 2g makes a value dying at op g
    // retire (2g+1 < 2g+2) before the same-op result, so the consumer reuses
    // its register; and a block arg (2g) sits strictly before its block's first
    // op read (2g+1) so a block arg consumed by the first op is no longer
    // falsely flagged dead-on-def. Safe because every aarch64 op this backend
    // emits reads all source registers before writing its result register.
    // TINYC_NO_HALFOPEN restores integer positions.
    bool halfopen = (getenv("TINYC_NO_HALFOPEN") == NULL);
    #define A64_USEP(g)  (halfopen ? (2u * (uint32_t)(g) + 1u) : (uint32_t)(g))
    #define A64_DEFP(g)  (halfopen ? (2u * (uint32_t)(g) + 2u) : (uint32_t)(g))
    #define A64_BARGP(g) (halfopen ? (2u * (uint32_t)(g))      : (uint32_t)(g))
    #define A64_COVL(g)  (halfopen ? (2u * (uint32_t)(g) + 1u) : (uint32_t)(g))
    #define A64_COVF(g)  (halfopen ? (2u * (uint32_t)(g))      : (uint32_t)(g))

    // --- 1. Linearise + number values. One position per op (incl. const/spine
    //        ops, so is_call[] and use positions stay consistent). ---
    uint32_t *block_first = (uint32_t *)malloc(n_blocks * sizeof(uint32_t));
    uint32_t *block_last  = (uint32_t *)malloc(n_blocks * sizeof(uint32_t));
    MLIR_ValueHandle *handle = (MLIR_ValueHandle *)malloc(nv_max * sizeof(MLIR_ValueHandle));
    uint32_t *def_pos   = (uint32_t *)malloc(nv_max * sizeof(uint32_t));
    uint32_t *first_pos = (uint32_t *)malloc(nv_max * sizeof(uint32_t));
    uint32_t *last_use  = (uint32_t *)malloc(nv_max * sizeof(uint32_t));
    uint64_t *weight    = (uint64_t *)malloc(nv_max * sizeof(uint64_t));
    uint8_t  *crosses   = (uint8_t  *)malloc(nv_max * sizeof(uint8_t));
    int8_t   *home_pk   = (int8_t   *)malloc(nv_max * sizeof(int8_t));
    A64HHash vmap; a64_hh_init(&vmap, nv_max);
    A64HHash bmap; a64_hh_init(&bmap, n_blocks);

    for (size_t b = 0; b < n_blocks; b++)
        a64_hh_put(&bmap, (MLIR_ValueHandle)MLIR_GetRegionBlock(reg, b), (uint32_t)b);

    uint32_t pos = 0, nv = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        block_first[b] = pos;
        if (b > 0 || home_params) {
            size_t na = MLIR_GetBlockNumArgs(bk);
            for (size_t a = 0; a < na; a++) {
                MLIR_ValueHandle arg = MLIR_GetBlockArg(bk, a);
                handle[nv] = arg; def_pos[nv] = A64_BARGP(pos); first_pos[nv] = A64_BARGP(pos);
                last_use[nv] = 0; weight[nv] = 0; crosses[nv] = 0; home_pk[nv] = -1;
                a64_hh_put(&vmap, arg, nv); nv++;
            }
        }
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            int64_t cv; uint8_t c64;
            bool skip = const_int_val(ctx, op, &cv, &c64) || a64_is_spine(op, memspine);
            if (!skip) {
                size_t nr = MLIR_GetOpNumResults(op);
                for (size_t r = 0; r < nr; r++) {
                    MLIR_ValueHandle rv = MLIR_GetOpResult(op, r);
                    handle[nv] = rv; def_pos[nv] = A64_DEFP(pos); first_pos[nv] = A64_DEFP(pos);
                    last_use[nv] = 0; weight[nv] = 0; crosses[nv] = 0; home_pk[nv] = -1;
                    a64_hh_put(&vmap, rv, nv); nv++;
                }
            }
            pos++;
        }
        block_last[b] = pos;
    }

    // --- 1b. Per-block loop depth via back-edge difference array. ---
    uint32_t *loop_depth = (uint32_t *)calloc(n_blocks, sizeof(uint32_t));
    {
        int64_t *delta = (int64_t *)calloc(n_blocks + 1, sizeof(int64_t));
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    uint32_t ti;
                    if (a64_hh_get(&bmap, (MLIR_ValueHandle)MLIR_GetOpSuccessor(op, s), &ti) &&
                        ti <= (uint32_t)b) { delta[ti] += 1; delta[b + 1] -= 1; }
                }
            }
        }
        int64_t acc = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            acc += delta[b]; loop_depth[b] = acc < 0 ? 0 : (uint32_t)acc;
        }
        free(delta);
    }
    #define A64_USE_WEIGHT_AT(d) (1ull << (((d) * 3u) < 60u ? ((d) * 3u) : 60u))

    // --- 2. Liveness dataflow: gen/kill then live_in/live_out fixpoint. ---
    size_t bw = a64_bs_words(nv ? nv : 1);
    uint64_t *live_in  = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *live_out = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *gen      = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *kill     = (uint64_t *)calloc(n_blocks * bw, sizeof(uint64_t));
    uint64_t *scratch  = (uint64_t *)calloc(bw, sizeof(uint64_t));

    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        uint64_t *G = gen + b * bw, *K = kill + b * bw;
        if (b > 0) {
            size_t na = MLIR_GetBlockNumArgs(bk);
            for (size_t a = 0; a < na; a++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetBlockArg(bk, a), &vi)) a64_bs_set(K, vi);
            }
        }
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (a64_is_spine(op, memspine)) continue;
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi) && !a64_bs_test(K, vi))
                    a64_bs_set(G, vi);
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi) &&
                        !a64_bs_test(K, vi)) a64_bs_set(G, vi);
                }
            }
            MLIR_ValueHandle mbase, midx;
            if (a64_memfuse_uses(ctx, op, memfuse, &mbase, &midx)) {
                MLIR_ValueHandle uu[2] = { mbase, midx };
                for (int u = 0; u < 2; u++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, uu[u], &vi) && !a64_bs_test(K, vi)) a64_bs_set(G, vi);
                }
            }
            MLIR_ValueHandle sfx;
            if (a64_shiftfuse_uses(ctx, op, shiftfuse, &sfx)) {
                uint32_t vi;
                if (a64_hh_get(&vmap, sfx, &vi) && !a64_bs_test(K, vi)) a64_bs_set(G, vi);
            }
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t r = 0; r < nr; r++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetOpResult(op, r), &vi)) a64_bs_set(K, vi);
            }
        }
    }

    bool changed = true; int iters = 0;
    while (changed && iters++ < 4096) {
        changed = false;
        for (size_t br = 0; br < n_blocks; br++) {
            size_t b = n_blocks - 1 - br;
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            uint64_t *LO = live_out + b * bw;
            a64_bs_zero(LO, nv);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    uint32_t si;
                    if (a64_hh_get(&bmap, (MLIR_ValueHandle)MLIR_GetOpSuccessor(op, s), &si))
                        a64_bs_or(LO, LO, live_in + si * bw, nv);
                }
            }
            uint64_t *LI = live_in + b * bw;
            a64_bs_diff(scratch, LO, kill + b * bw, nv);
            a64_bs_or(scratch, scratch, gen + b * bw, nv);
            if (!a64_bs_equal(scratch, LI, nv)) {
                memcpy(LI, scratch, bw * sizeof(uint64_t)); changed = true;
            }
        }
    }

    // --- 3. Per-value first_pos / last_use_pos / use_weight. ---
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        uint64_t *LO = live_out + b * bw, *LI = live_in + b * bw;
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (a64_bs_test(LO, vi) && A64_COVL(block_last[b]) > last_use[vi])  last_use[vi]  = A64_COVL(block_last[b]);
            if (a64_bs_test(LI, vi) && A64_COVF(block_first[b]) < first_pos[vi]) first_pos[vi] = A64_COVF(block_first[b]);
        }
        uint32_t op_pos = block_first[b];
        uint64_t w = A64_USE_WEIGHT_AT(loop_depth[b]);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (a64_is_spine(op, memspine)) { op_pos++; continue; }
            uint32_t up = A64_USEP(op_pos);
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                uint32_t vi;
                if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += w;
                }
            }
            size_t ns = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < ns; s++) {
                MLIR_BlockHandle succ = MLIR_GetOpSuccessor(op, s);
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi)) {
                        if (up > last_use[vi]) last_use[vi] = up;
                        if (up < first_pos[vi]) first_pos[vi] = up;
                        weight[vi] += w;
                    }
                    // The branch writes the k-th successor block arg's home at
                    // THIS position; extend that arg's interval start to cover
                    // the write (forward edges define the arg before its own
                    // block-entry position).
                    uint32_t avi;
                    if (a64_hh_get(&vmap, MLIR_GetBlockArg(succ, k), &avi) &&
                        up < first_pos[avi]) first_pos[avi] = up;
                }
            }
            MLIR_ValueHandle mbase, midx;
            if (a64_memfuse_uses(ctx, op, memfuse, &mbase, &midx)) {
                MLIR_ValueHandle uu[2] = { mbase, midx };
                for (int u = 0; u < 2; u++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, uu[u], &vi)) {
                        if (up > last_use[vi]) last_use[vi] = up;
                        if (up < first_pos[vi]) first_pos[vi] = up;
                        weight[vi] += w;
                    }
                }
            }
            MLIR_ValueHandle sfx;
            if (a64_shiftfuse_uses(ctx, op, shiftfuse, &sfx)) {
                uint32_t vi;
                if (a64_hh_get(&vmap, sfx, &vi)) {
                    if (up > last_use[vi]) last_use[vi] = up;
                    if (up < first_pos[vi]) first_pos[vi] = up;
                    weight[vi] += w;
                }
            }
            op_pos++;
        }
    }

    // crosses_call: a value must live in a callee-saved register iff it is live
    // across a call -- i.e. live immediately AFTER some llvm.call and not that
    // call's own result (the result is born at the call, it does not cross it).
    // This is computed PRECISELY from the per-block live_out by a backward walk:
    // `live_out(call) \ {result}` is exactly the set of values live across the
    // call, which is sound (a caller-saved reg live across a call is clobbered)
    // and exact. The legacy linear first_pos..last_use span (kept under
    // TINYC_LINEAR_CROSSES for A/B) over-approximates across mutually-exclusive
    // CFG paths, falsely forcing phi/block-arg values into the 9 callee-saved
    // registers and inflating spill pressure.
    if (getenv("TINYC_LINEAR_CROSSES")) {
        uint8_t *is_call = (uint8_t *)calloc(pos ? pos : 1, sizeof(uint8_t));
        uint32_t op_pos = 0;
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                if (name_eq(MLIR_GetOpName(MLIR_GetBlockOp(bk, i)), "llvm.call"))
                    is_call[op_pos] = 1;
                op_pos++;
            }
        }
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (last_use[vi] < first_pos[vi]) continue;
            uint32_t lo = halfopen ? first_pos[vi] / 2u : first_pos[vi];
            uint32_t hi = halfopen ? last_use[vi]  / 2u : last_use[vi];
            for (uint32_t g = lo; g <= hi && g < pos; g++) {
                if (!is_call[g]) continue;
                uint32_t cs = A64_USEP(g);
                if (cs >= first_pos[vi] && cs < last_use[vi]) { crosses[vi] = 1; break; }
            }
        }
        free(is_call);
    } else {
        uint64_t *live = (uint64_t *)calloc(bw, sizeof(uint64_t));
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            memcpy(live, live_out + b * bw, bw * sizeof(uint64_t));
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t ii = no; ii-- > 0; ) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, ii);
                if (a64_is_spine(op, memspine)) continue;
                // `live` holds the live set immediately AFTER op.
                if (name_eq(MLIR_GetOpName(op), "llvm.call")) {
                    uint32_t rvi = 0; bool has_r = false;
                    if (MLIR_GetOpNumResults(op) &&
                        a64_hh_get(&vmap, MLIR_GetOpResult(op, 0), &rvi)) has_r = true;
                    for (uint32_t vi = 0; vi < nv; vi++)
                        if (a64_bs_test(live, vi) && !(has_r && vi == rvi))
                            crosses[vi] = 1;
                }
                // Transform to live-in(op): live = (live \ defs) U uses.
                size_t nr = MLIR_GetOpNumResults(op);
                for (size_t r = 0; r < nr; r++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpResult(op, r), &vi)) a64_bs_clear(live, vi);
                }
                size_t nop = MLIR_GetOpNumOperands(op);
                for (size_t k = 0; k < nop; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) a64_bs_set(live, vi);
                }
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vi;
                        if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi))
                            a64_bs_set(live, vi);
                    }
                }
                MLIR_ValueHandle mbase, midx;
                if (a64_memfuse_uses(ctx, op, memfuse, &mbase, &midx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, mbase, &vi)) a64_bs_set(live, vi);
                    if (a64_hh_get(&vmap, midx, &vi))  a64_bs_set(live, vi);
                }
                MLIR_ValueHandle sfx;
                if (a64_shiftfuse_uses(ctx, op, shiftfuse, &sfx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, sfx, &vi)) a64_bs_set(live, vi);
                }
            }
        }
        free(live);
    }

    // --- 3c. Phi edge-copy coalescing hints. -----------------------------------
    // For each block-arg B, find a predecessor edge whose source value V (a) is
    // used exactly once (its sole use is that edge) and (b) dies exactly where B
    // is born (last_use[V] == first_pos[B], the defining edge). If the linear
    // scan then gives B the SAME register V occupies, the edge `mov R_B, R_V`
    // becomes an identity copy that emit_edge_copies elides -- removing a link
    // from the loop-carried dependency chain (the IPC lever). The copies for
    // each cf.cond_br successor run in their own isolated landing block, so the
    // only unsoundness is passing B's own value through the same terminator to
    // another arg (then V's def would clobber it before the move): guard by
    // requiring B not appear as any successor operand of the terminator.
    int32_t *hint_src = (int32_t *)malloc((nv ? nv : 1) * sizeof(int32_t));
    for (uint32_t i = 0; i < nv; i++) hint_src[i] = -1;
    if (!getenv("TINYC_NO_PHI_COALESCE")) {
        uint32_t *nuse = (uint32_t *)calloc(nv ? nv : 1, sizeof(uint32_t));
        // Pass A: count every use of every value (regular operands, successor
        // operands, and re-injected memfuse/shiftfuse operands).
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                if (a64_is_spine(op, memspine)) continue;
                size_t nop = MLIR_GetOpNumOperands(op);
                for (size_t k = 0; k < nop; k++) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, MLIR_GetOpOperand(op, k), &vi)) nuse[vi]++;
                }
                size_t ns = MLIR_GetOpNumSuccessors(op);
                for (size_t s = 0; s < ns; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vi;
                        if (a64_hh_get(&vmap, MLIR_GetOpSuccessorOperand(op, s, k), &vi))
                            nuse[vi]++;
                    }
                }
                MLIR_ValueHandle mbase, midx;
                if (a64_memfuse_uses(ctx, op, memfuse, &mbase, &midx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, mbase, &vi)) nuse[vi]++;
                    if (a64_hh_get(&vmap, midx, &vi))  nuse[vi]++;
                }
                MLIR_ValueHandle sfx;
                if (a64_shiftfuse_uses(ctx, op, shiftfuse, &sfx)) {
                    uint32_t vi;
                    if (a64_hh_get(&vmap, sfx, &vi)) nuse[vi]++;
                }
            }
        }
        // Pass B: build hints from sole-use edge sources.
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
            size_t no = MLIR_GetBlockNumOps(bk);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                size_t ns = MLIR_GetOpNumSuccessors(op);
                if (ns == 0) continue;
                for (size_t s = 0; s < ns; s++) {
                    MLIR_BlockHandle succ = MLIR_GetOpSuccessor(op, s);
                    size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                    for (size_t k = 0; k < nso; k++) {
                        uint32_t vV, vB;
                        MLIR_ValueHandle Vh = MLIR_GetOpSuccessorOperand(op, s, k);
                        MLIR_ValueHandle Bh = MLIR_GetBlockArg(succ, k);
                        if (!a64_hh_get(&vmap, Vh, &vV)) continue;
                        if (!a64_hh_get(&vmap, Bh, &vB)) continue;
                        if (nuse[vV] != 1) continue;            // sole use = this edge
                        if (last_use[vV] != first_pos[vB]) continue; // V dies into B
                        if (hint_src[vB] >= 0) continue;        // already hinted
                        // Guard: B's own value must not be passed through this
                        // terminator to another arg (would be clobbered by V's def).
                        bool through = false;
                        for (size_t s2 = 0; s2 < ns && !through; s2++) {
                            size_t n2 = MLIR_GetOpNumSuccessorOperands(op, s2);
                            for (size_t k2 = 0; k2 < n2; k2++)
                                if (MLIR_GetOpSuccessorOperand(op, s2, k2) == Bh) {
                                    through = true; break;
                                }
                        }
                        if (through) continue;
                        hint_src[vB] = (int32_t)vV;
                    }
                }
            }
        }
        free(nuse);
    }

    // --- 4. Linear scan over the pool, loop-weighted eviction. ---
    uint32_t *order = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < nv; i++) order[i] = i;
    {   // stable bottom-up merge sort by first_pos (ties keep numbering order).
        uint32_t *tmp = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        for (size_t width = 1; width < nv; width *= 2) {
            for (size_t lo = 0; lo < nv; lo += 2 * width) {
                size_t mid = lo + width;     if (mid > nv) mid = nv;
                size_t hi  = lo + 2 * width; if (hi  > nv) hi  = nv;
                size_t a = lo, bb = mid, o = lo;
                while (a < mid && bb < hi)
                    tmp[o++] = (first_pos[order[bb]] < first_pos[order[a]]) ? order[bb++] : order[a++];
                while (a < mid) tmp[o++] = order[a++];
                while (bb < hi) tmp[o++] = order[bb++];
            }
            uint32_t *sw = order; order = tmp; tmp = sw;
        }
        free(tmp);
    }

    // Active list: (value index, pool index). reg_busy[pk].
    uint32_t *act_vi = (uint32_t *)malloc(A64_NPOOL * sizeof(uint32_t));
    uint8_t  *act_pk = (uint8_t  *)malloc(A64_NPOOL * sizeof(uint8_t));
    size_t nact = 0;
    bool reg_busy[A64_NPOOL]; for (int i = 0; i < A64_NPOOL; i++) reg_busy[i] = false;

    for (uint32_t k = 0; k < nv; k++) {
        uint32_t vi = order[k];
        // Retire active intervals ending before this one starts (strict, so an
        // operand whose last use == this def keeps its register).
        for (size_t a = 0; a < nact; ) {
            if (last_use[act_vi[a]] < first_pos[vi]) {
                reg_busy[act_pk[a]] = false;
                act_vi[a] = act_vi[nact - 1]; act_pk[a] = act_pk[nact - 1]; nact--;
            } else a++;
        }
        // Dead-on-def / never-used values keep their slot.
        if (last_use[vi] <= first_pos[vi]) continue;
        bool callee_only = crosses[vi];
        // Phi coalescing: if vi (a block arg) has a hint source V that is still
        // active and dies exactly at vi's birth (last_use[V] == first_pos[vi],
        // and V has no other use), retire V now and let vi inherit V's register.
        // The edge copy then sees edge_src_loc(V) == edge_dst_loc(vi) and is
        // elided. Sound: V is truly dead at this point (sole use is this edge),
        // so transferring its register to vi clobbers nothing; the register was
        // exclusive to V's interval [V.def, edge] and becomes exclusive to vi
        // from the edge onward (contiguous, no gap).
        if (hint_src[vi] >= 0) {
            int32_t vV = hint_src[vi];
            for (size_t a = 0; a < nact; a++) {
                if (act_vi[a] != (uint32_t)vV) continue;
                if (last_use[vV] != first_pos[vi]) break;   // not a clean death
                if (callee_only && act_pk[a] < A64_NCALLER) break; // reg class
                uint8_t pk = act_pk[a];
                home_pk[vi] = (int8_t)pk;        // inherit V's physical register
                act_vi[a] = vi;                   // ownership transfers; reg stays busy
                break;
            }
            if (home_pk[vi] >= 0) continue;
        }
        int found = -1;
        for (int pk = 0; pk < A64_NPOOL; pk++) {
            if (reg_busy[pk]) continue;
            if (callee_only && pk < A64_NCALLER) continue;
            found = pk; break;
        }
        if (found >= 0) {
            reg_busy[found] = true;
            act_vi[nact] = vi; act_pk[nact] = (uint8_t)found; nact++;
            home_pk[vi] = (int8_t)found;
            continue;
        }
        // No free eligible reg: evict the active interval with the lowest use
        // weight (cheapest to reload), tie-break latest end — but only if it is
        // strictly cheaper than the current value, else spill the current one.
        size_t evict_a = (size_t)-1; uint64_t best_w = 0; uint32_t best_end = 0;
        for (size_t a = 0; a < nact; a++) {
            if (callee_only && act_pk[a] < A64_NCALLER) continue;
            uint64_t aw = weight[act_vi[a]]; uint32_t ae = last_use[act_vi[a]];
            if (evict_a == (size_t)-1 || aw < best_w || (aw == best_w && ae > best_end)) {
                evict_a = a; best_w = aw; best_end = ae;
            }
        }
        if (evict_a != (size_t)-1 && best_w >= weight[vi]) evict_a = (size_t)-1;
        if (evict_a != (size_t)-1) {
            home_pk[act_vi[evict_a]] = -1;          // evicted -> slot
            act_vi[evict_a] = vi;                   // reuse its register
            home_pk[vi] = (int8_t)act_pk[evict_a];
        }
        // else: current value stays in its slot (home_pk already -1).
    }

    // --- 4b. Assign frame slots to spilled values with WMIR-style reuse.
    //         Walking values in first_pos order (the existing `order[]`), a slot
    //         whose occupant's interval ended strictly before the current
    //         value's first_pos is reclaimed onto a free list and handed to the
    //         current value. This is the SAME non-overlap invariant the register
    //         active-list retires on (last_use < first_pos), reusing the very
    //         intervals the allocator already trusts, so sharing is sound. Each
    //         value is processed at ITS OWN first_pos (never at an eviction
    //         point), so reused slots are always free as of that position and no
    //         WMIR-style "fresh, no-reuse" special case is needed. The frame is
    //         then sized to PEAK simultaneous spills, not TOTAL spills, keeping
    //         most offsets inside aarch64's scaled imm12 load/store range. ---
    int32_t next_slot = 0;
    {
        bool no_reuse = getenv("TINYC_NO_SLOT_REUSE") != NULL;
        uint32_t *sl_slot = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        uint32_t *sl_end  = (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        uint32_t *freelist= (uint32_t *)malloc((nv ? nv : 1) * sizeof(uint32_t));
        size_t n_sact = 0, n_free = 0;
        for (uint32_t k = 0; k < nv; k++) {
            uint32_t vi = order[k];
            for (size_t s = 0; s < n_sact; ) {
                if (sl_end[s] < first_pos[vi]) {
                    freelist[n_free++] = sl_slot[s];
                    sl_slot[s] = sl_slot[n_sact - 1];
                    sl_end[s]  = sl_end[n_sact - 1];
                    n_sact--;
                } else s++;
            }
            if (home_pk[vi] >= 0) continue;           // register-homed: no slot
            uint32_t end = last_use[vi] < first_pos[vi] ? first_pos[vi]
                                                        : last_use[vi];
            uint32_t slot = (!no_reuse && n_free > 0) ? freelist[--n_free]
                                                      : (uint32_t)next_slot++;
            sl_slot[n_sact] = slot; sl_end[n_sact] = end; n_sact++;
            sm_put(sm, handle[vi], (int32_t)slot);
        }
        free(sl_slot); free(sl_end); free(freelist);
    }
    if (getenv("TINYC_SPILL_STATS")) {
        for (uint32_t vi = 0; vi < nv; vi++) {
            if (home_pk[vi] >= 0) continue;
            MLIR_OpHandle dop = MLIR_GetValueDefiningOp(handle[vi]);
            const char *nm = "(blockarg)";
            int nl = 10;
            if (dop) { string s = MLIR_GetOpName(dop); nm = s.str; nl = (int)s.size; }
            const char *dead = (last_use[vi] <= first_pos[vi]) ? "deaddef" : "live";
            char ops[128]; int op_off = 0; ops[0] = '\0';
            if (dop) {
                size_t non = MLIR_GetOpNumOperands(dop);
                for (size_t k = 0; k < non && op_off < 100; k++) {
                    MLIR_ValueHandle ov = MLIR_GetOpOperand(dop, k);
                    MLIR_OpHandle odef = MLIR_GetValueDefiningOp(ov);
                    int64_t cvv; uint8_t c64v;
                    const char *on; int onl;
                    if (!odef) { on = "barg"; onl = 4; }
                    else if (const_int_val(ctx, odef, &cvv, &c64v)) { on = "const"; onl = 5; }
                    else { string os = MLIR_GetOpName(odef); on = os.str; onl = (int)os.size; }
                    op_off += snprintf(ops + op_off, sizeof(ops) - op_off,
                                       "%s%.*s", k ? "," : "", onl, on);
                }
            }
            fprintf(stderr, "SPILL\t%s\t%s\t%.*s\t%s\n",
                    weight[vi] > 1 ? "hot" : "cold", dead, nl, nm, ops);
        }
    }
    *out_nslots = next_slot;

    // --- 5. Emit results: write rm for register homes; report used_mask. ---
    uint32_t used_mask = 0;
    for (uint32_t vi = 0; vi < nv; vi++) {
        if (home_pk[vi] < 0) continue;
        int pk = home_pk[vi];
        sm_put(rm, handle[vi], a64_pool_reg(pk));
        if (pk >= A64_NCALLER) used_mask |= (1u << (pk - A64_NCALLER));
    }

    #undef A64_USE_WEIGHT_AT
    #undef A64_USEP
    #undef A64_DEFP
    #undef A64_BARGP
    #undef A64_COVL
    #undef A64_COVF
    free(order); free(act_vi); free(act_pk);
    free(live_in); free(live_out); free(gen); free(kill); free(scratch);
    free(block_first); free(block_last); free(handle);
    free(def_pos); free(first_pos); free(last_use);
    free(weight); free(crosses); free(home_pk); free(loop_depth);
    free(hint_src);
    a64_hh_free(&vmap); a64_hh_free(&bmap);
    return used_mask;
}

// Linear-scan register allocator for the CFG (from-wasm) lowering path. Assigns
// block-local SSA values (def and all uses in one block, never a block arg, and
// never consumed by a terminator or an llvm.call — those paths read frame slots
// directly) to callee-saved registers x19..x28. Everything else keeps its frame
// slot. Fills `rm` (value -> home reg) and returns the set of used callee-saved
// regs as a bitmask over x19..x28 (bit k = x(19+k)).
//
// Allocation is deterministic (purely structural block/op order, ascending
// register preference), which is required for the bit-identical self-host gate.
// Callee-saved homes survive `bl`, so a value live across a call needs no
// spill-around-call handling. Functions containing ops with nested regions are
// not allocated (returns 0) — the CFG path is flat in practice.
static uint32_t alloc_regs_cfg(MLIR_Context *ctx, MLIR_RegionHandle reg,
                               size_t n_blocks, SlotMap *rm,
                               SlotMap *memfuse, SlotMap *memspine,
                               SlotMap *shiftfuse) {
    Arena *ar = MLIR_GetArenaAllocator(ctx);
    size_t cap = 0;
    size_t maxops = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        if (no > maxops) maxops = no;
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            if (MLIR_GetOpNumRegions(op) > 0) return 0;   // bail: nested regions
            cap += MLIR_GetOpNumResults(op);
        }
    }
    if (cap == 0) return 0;

    int32_t *def_block = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    int32_t *def_pos   = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    int32_t *last_use  = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    uint8_t *disq      = (uint8_t *)arena_alloc(ar, cap * sizeof(uint8_t));
    uint8_t *usedv     = (uint8_t *)arena_alloc(ar, cap * sizeof(uint8_t));
    uint8_t *crosses   = (uint8_t *)arena_alloc(ar, cap * sizeof(uint8_t));
    int8_t  *home_pk   = (int8_t *)arena_alloc(ar, cap * sizeof(int8_t));
    // Phi edge-copy coalescing bookkeeping: total use count of each value, and
    // (when it is used exactly once as a branch successor operand) the
    // destination block-arg value + the terminator carrying that edge.
    int32_t *use_cnt   = (int32_t *)arena_alloc(ar, cap * sizeof(int32_t));
    MLIR_ValueHandle *edge_arg =
        (MLIR_ValueHandle *)arena_alloc(ar, cap * sizeof(MLIR_ValueHandle));
    MLIR_OpHandle *edge_term =
        (MLIR_OpHandle *)arena_alloc(ar, cap * sizeof(MLIR_OpHandle));
    MLIR_ValueHandle *vals =
        (MLIR_ValueHandle *)arena_alloc(ar, cap * sizeof(MLIR_ValueHandle));
    SlotMap idx = {0}; idx.arena = ar;   // value -> array index
    size_t nv = 0;

    // Pass 1: record every op-result definition with its block and position.
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            int64_t cv; uint8_t c64;
            if (const_int_val(ctx, op, &cv, &c64)) continue; // remat'd, no home
            // Folded linmem spine ops (zext/add/inttoptr) emit no code and get
            // no home register; their result is consumed by the fused load/store.
            if (memspine && MLIR_GetOpNumResults(op) == 1) {
                int32_t mt;
                if (sm_get(memspine, MLIR_GetOpResult(op, 0), &mt)) continue;
            }
            size_t nr = MLIR_GetOpNumResults(op);
            for (size_t r = 0; r < nr; r++) {
                vals[nv] = MLIR_GetOpResult(op, r);
                def_block[nv] = (int32_t)b; def_pos[nv] = (int32_t)i;
                last_use[nv] = -1; disq[nv] = 0; usedv[nv] = 0; crosses[nv] = 0;
                home_pk[nv] = -1;
                use_cnt[nv] = 0; edge_arg[nv] = MLIR_INVALID_HANDLE;
                edge_term[nv] = MLIR_INVALID_HANDLE;
                sm_put(&idx, vals[nv], (int32_t)nv);
                nv++;
            }
        }
    }

    // Pass 2: scan operand uses. Disqualify on cross-block use, or use by a
    // terminator (incl. successor/edge operands) or an llvm.call (those paths
    // read frame slots, not home registers). Track the last in-block use.
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        for (size_t i = 0; i < no; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
            bool is_term = (i + 1 == no);
            bool is_call = name_eq(MLIR_GetOpName(op), "llvm.call");
            // A folded spine op (zext/add/inttoptr) emits no code: its operand
            // reads must NOT extend liveness. The fused load/store re-registers
            // base+idx as live uses at its own position (below).
            if (memspine && MLIR_GetOpNumResults(op) == 1) {
                int32_t mt;
                if (sm_get(memspine, MLIR_GetOpResult(op, 0), &mt)) continue;
            }
            size_t nop = MLIR_GetOpNumOperands(op);
            for (size_t k = 0; k < nop; k++) {
                int32_t vi;
                if (!sm_get(&idx, MLIR_GetOpOperand(op, k), &vi)) continue;
                usedv[vi] = 1; use_cnt[vi]++;
                if ((int32_t)b != def_block[vi] || is_term || is_call)
                    disq[vi] = 1;
                else if ((int32_t)i > last_use[vi])
                    last_use[vi] = (int32_t)i;
            }
            // Branch successor (block-arg) operands are a separate operand list
            // resolved via slot-to-slot edge copies (copy_slot reads the frame
            // slot), so any value passed across an edge must stay in memory.
            size_t nsucc = MLIR_GetOpNumSuccessors(op);
            for (size_t s = 0; s < nsucc; s++) {
                size_t nso = MLIR_GetOpNumSuccessorOperands(op, s);
                for (size_t k = 0; k < nso; k++) {
                    int32_t vi;
                    if (!sm_get(&idx, MLIR_GetOpSuccessorOperand(op, s, k), &vi))
                        continue;
                    usedv[vi] = 1; disq[vi] = 1; use_cnt[vi]++;
                    // Record the destination block-arg + terminator so a sole-use
                    // edge operand can later be coalesced into the block-arg's
                    // home register (the mov vanishes in emit_edge_copies).
                    edge_arg[vi]  = MLIR_GetBlockArg(MLIR_GetOpSuccessor(op, s), k);
                    edge_term[vi] = op;
                }
            }
            // Fused linmem access: the rewritten load/store reads `base` and
            // `idx` directly (not the skipped inttoptr spine). Register those as
            // live uses at this position so the allocator keeps them in regs /
            // models their lifetimes correctly (base is cross-block -> spilled,
            // idx is block-local -> stays live until here, never aliased).
            if (memfuse) {
                string opn = MLIR_GetOpName(op);
                MLIR_ValueHandle P = MLIR_INVALID_HANDLE;
                if (name_eq(opn, "llvm.load") && nop >= 1)
                    P = MLIR_GetOpOperand(op, 0);
                else if (name_eq(opn, "llvm.store") && nop >= 2)
                    P = MLIR_GetOpOperand(op, 1);
                int32_t mt;
                if (P != MLIR_INVALID_HANDLE && sm_get(memfuse, P, &mt)) {
                    MLIR_ValueHandle mb, mi; MLIR_OpHandle z, e, p;
                    if (memfuse_match(ctx, P, &mb, &mi, &z, &e, &p, NULL, NULL)) {
                        MLIR_ValueHandle uu[2] = { mb, mi };
                        for (int u = 0; u < 2; u++) {
                            int32_t vi;
                            if (!sm_get(&idx, uu[u], &vi)) continue;
                            usedv[vi] = 1; use_cnt[vi]++;
                            if ((int32_t)b != def_block[vi] || is_call)
                                disq[vi] = 1;
                            else if ((int32_t)i > last_use[vi])
                                last_use[vi] = (int32_t)i;
                        }
                    }
                }
            }
            // Shift-fused add: the shifted-register add reads the shift source x
            // directly (the skipped shl/mul result is not a real operand). Mark
            // x live here, mirroring the memfuse re-injection above.
            if (shiftfuse) {
                MLIR_ValueHandle sfx;
                if (a64_shiftfuse_uses(ctx, op, shiftfuse, &sfx)) {
                    int32_t vi;
                    if (sm_get(&idx, sfx, &vi)) {
                        usedv[vi] = 1; use_cnt[vi]++;
                        if ((int32_t)b != def_block[vi] || is_call)
                            disq[vi] = 1;
                        else if ((int32_t)i > last_use[vi])
                            last_use[vi] = (int32_t)i;
                    }
                }
            }
        }
    }

    // Pass 3: mark block-local values whose live range [def, last_use] spans an
    // llvm.call. Such values are clobbered by the `bl`, so they may only be
    // homed in callee-saved registers (never the caller-saved x12..x15 pool).
    int32_t *callpos = (int32_t *)arena_alloc(ar, (maxops ? maxops : 1) *
                                              sizeof(int32_t));
    for (size_t b = 0; b < n_blocks; b++) {
        MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
        size_t no = MLIR_GetBlockNumOps(bk);
        size_t ncall = 0;
        for (size_t i = 0; i < no; i++)
            if (name_eq(MLIR_GetOpName(MLIR_GetBlockOp(bk, i)), "llvm.call"))
                callpos[ncall++] = (int32_t)i;
        if (ncall == 0) continue;
        for (size_t vi = 0; vi < nv; vi++) {
            if (def_block[vi] != (int32_t)b) continue;
            // Strict: a value used only as a call arg (last_use == the call) is
            // not crossing -- read into the param reg before the `bl`.
            for (size_t c = 0; c < ncall; c++)
                if (def_pos[vi] < callpos[c] && callpos[c] < last_use[vi]) {
                    crosses[vi] = 1; break;
                }
        }
    }

    // Per-block linear scan over the pool (caller-saved x12..x15 then callee-
    // saved x19..x28). Block-local values never overlap across blocks, so each
    // block independently reuses the whole pool. Expiry uses a strict
    // `end < start` test so an operand whose last use coincides with a result's
    // definition keeps its register — preventing a multi-instruction lowering
    // from clobbering an operand it still reads. `used_mask` reports only the
    // callee-saved registers actually used (bit k = x(19+k)), driving the
    // prologue/epilogue save/restore; caller-saved homes need none. State is
    // kept in pool-index space (0..A64_NPOOL-1), mapped to physical regs via
    // A64_POOL; allocation is deterministic for the bit-identical self-host.
    uint32_t used_mask = 0;

    // ---- Block-argument register homing ----
    // Keep loop-carried SSA values (mem2reg promotes them to block args) in
    // registers across blocks. Collect non-entry block args, score them by use
    // frequency, and give the top A64_NHOME a UNIQUE reserved callee-saved
    // register (x28 downward). Because each homed arg gets a distinct register,
    // no two homed args ever share a register and correctness needs no global
    // liveness analysis (so none of the prior global-allocator O(blocks*values)
    // bitset blowup). Reserved regs are excluded from the block-local pool
    // below and saved/restored via used_mask; operand reads/writes already
    // route through `rm` (use_val/def_val/fin_val), and edges into homed args
    // are resolved by the allocation-aware parallel mover in emit_edge_copies.
    // Selection is deterministic (score desc, then block/arg order) for the
    // bit-identical self-host gate.
    uint32_t reserved_mask = 0;
    if (!getenv("TINYC_NO_HOMING")) {
        size_t nba = 0;
        for (size_t b = 1; b < n_blocks; b++)
            nba += MLIR_GetBlockNumArgs(MLIR_GetRegionBlock(reg, b));
        if (nba) {
            MLIR_ValueHandle *bv =
                (MLIR_ValueHandle *)arena_alloc(ar, nba * sizeof(MLIR_ValueHandle));
            int32_t *bscore = (int32_t *)arena_alloc(ar, nba * sizeof(int32_t));
            SlotMap bidx = {0}; bidx.arena = ar;
            size_t m = 0;
            for (size_t b = 1; b < n_blocks; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
                size_t na = MLIR_GetBlockNumArgs(bk);
                for (size_t a = 0; a < na; a++) {
                    bv[m] = MLIR_GetBlockArg(bk, a); bscore[m] = 0;
                    sm_put(&bidx, bv[m], (int32_t)m); m++;
                }
            }
            for (size_t b = 0; b < n_blocks; b++) {
                MLIR_BlockHandle bk = MLIR_GetRegionBlock(reg, b);
                size_t no = MLIR_GetBlockNumOps(bk);
                for (size_t i = 0; i < no; i++) {
                    MLIR_OpHandle op = MLIR_GetBlockOp(bk, i);
                    size_t nop = MLIR_GetOpNumOperands(op);
                    for (size_t k = 0; k < nop; k++) {
                        int32_t bi;
                        if (sm_get(&bidx, MLIR_GetOpOperand(op, k), &bi)) bscore[bi]++;
                    }
                    size_t nsucc = MLIR_GetOpNumSuccessors(op);
                    for (size_t sx = 0; sx < nsucc; sx++) {
                        size_t nso = MLIR_GetOpNumSuccessorOperands(op, sx);
                        for (size_t k = 0; k < nso; k++) {
                            int32_t bi;
                            if (sm_get(&bidx, MLIR_GetOpSuccessorOperand(op, sx, k), &bi))
                                bscore[bi]++;
                        }
                    }
                }
            }
            for (int h = 0; h < A64_NHOME; h++) {
                int best = -1;
                for (size_t j = 0; j < nba; j++) {
                    if (bscore[j] <= 0) continue;
                    if (best < 0 || bscore[j] > bscore[best]) best = (int)j;
                }
                if (best < 0) break;
                int pk = A64_NPOOL - 1 - h;          // 20 -> x26, 19 -> x25, ...
                sm_put(rm, bv[best], a64_pool_reg(pk));
                reserved_mask |= (1u << pk);
                used_mask |= (1u << (pk - A64_NCALLER));
                bscore[best] = -1;                   // consumed
            }
        }
    }

    // ---- Phi edge-copy coalescing (ISEL-time direct-write) ----
    // A value whose SOLE use is one branch successor-operand feeding a
    // register-homed block-arg is normally forced to a frame slot (disq above),
    // so emit_edge_copies materialises a `mov R_arg, R_val` (or a slot reload)
    // at the edge -- a link on the loop-carried dependency chain. If instead we
    // home the value DIRECTLY in the block-arg's reserved register, the edge
    // copy becomes an identity move (edge_src_loc == edge_dst_loc) and vanishes,
    // shortening the chain (the IPC lever).
    //
    // Safe only under tight conditions: the value's def is the last op before an
    // UNCONDITIONAL (single-successor) terminator, it has exactly one use (that
    // edge), and the destination register is not read by any OTHER operand of
    // the same edge (i.e. the old block-arg value is not also passed through).
    // Under these the block-arg register is dead from the def point onward
    // except for this edge, so writing the new value into it early is sound; the
    // value's own def-op may freely read the register as an input (the isel
    // already tolerates result-into-home-reg aliasing for every homed value).
    if (!getenv("TINYC_NO_PHI_COALESCE")) {
        for (size_t vi = 0; vi < nv; vi++) {
            if (!disq[vi] || use_cnt[vi] != 1) continue;
            MLIR_OpHandle term = edge_term[vi];
            MLIR_ValueHandle ba = edge_arg[vi];
            if (term == MLIR_INVALID_HANDLE || ba == MLIR_INVALID_HANDLE) continue;
            // Single-successor (unconditional) terminator only.
            if (MLIR_GetOpNumSuccessors(term) != 1) continue;
            // Value must be defined immediately before its block's terminator.
            MLIR_BlockHandle db = MLIR_GetRegionBlock(reg, def_block[vi]);
            size_t dno = MLIR_GetBlockNumOps(db);
            if (dno < 2 || def_pos[vi] != (int32_t)dno - 2) continue;
            if (MLIR_GetBlockOp(db, dno - 1) != term) continue;
            // Destination block-arg must be register-homed.
            int32_t R;
            if (!sm_get(rm, ba, &R)) continue;
            // The block-arg's register must not be read by any OTHER operand of
            // this edge (no pass-through of the old phi value into R).
            bool conflict = false;
            size_t nso = MLIR_GetOpNumSuccessorOperands(term, 0);
            for (size_t k = 0; k < nso && !conflict; k++) {
                MLIR_ValueHandle o = MLIR_GetOpSuccessorOperand(term, 0, k);
                if (o == ba) { conflict = true; break; }   // old arg passed through
                if (o == vals[vi]) continue;                // the value itself
                int32_t or_;
                if (sm_get(rm, o, &or_) && or_ == R) conflict = true;
            }
            if (conflict) continue;
            // Home the value directly in the block-arg's register.
            sm_put(rm, vals[vi], R);
        }
    }

    for (size_t b = 0; b < n_blocks; b++) {
        uint32_t freemask = ((1u << A64_NPOOL) - 1u) & ~reserved_mask;
        int32_t act_end[A64_NPOOL]; uint8_t act_pi[A64_NPOOL]; size_t nact = 0;
        for (size_t vi = 0; vi < nv; vi++) {
            if (def_block[vi] != (int32_t)b) continue;
            if (disq[vi] || !usedv[vi] || last_use[vi] < def_pos[vi]) continue;
            int32_t s = def_pos[vi];
            for (size_t a = 0; a < nact; ) {
                if (act_end[a] < s) {
                    freemask |= (1u << act_pi[a]);
                    act_pi[a]  = act_pi[nact - 1];
                    act_end[a] = act_end[nact - 1];
                    nact--;
                } else a++;
            }
            // Copy coalescing: if this value is the result of a single-source
            // cast whose source is a pool-homed, block-local value dying exactly
            // at this op, reuse the source's register. For the pure-move casts
            // this turns emit_mov_reg into a no-op (the copy disappears); for the
            // extend casts it merely lowers register pressure. Provably safe:
            // the source's live range ends here and these handlers tolerate
            // rd == r0, so no value is clobbered before its last read.
            if (!getenv("TINYC_NO_COALESCE")) {
                MLIR_ValueHandle csrc;
                MLIR_OpHandle dop = MLIR_GetValueDefiningOp(vals[vi]);
                int32_t ui;
                if (dop != MLIR_INVALID_HANDLE && cast_src(dop, &csrc) &&
                    sm_get(&idx, csrc, &ui) && def_block[ui] == (int32_t)b &&
                    home_pk[ui] >= 0 && last_use[ui] == s &&
                    !(crosses[vi] && home_pk[ui] < A64_NCALLER)) {
                    int pk = home_pk[ui];
                    // The source still occupies pk (strict expiry keeps an entry
                    // whose end == s active); extend it to cover this value.
                    bool ext = false;
                    for (size_t a = 0; a < nact; a++)
                        if (act_pi[a] == (uint8_t)pk) {
                            act_end[a] = last_use[vi]; ext = true; break;
                        }
                    if (ext) {
                        home_pk[vi] = (int8_t)pk;
                        sm_put(rm, vals[vi], a64_pool_reg(pk));
                        if (pk >= A64_NCALLER)
                            used_mask |= (1u << (pk - A64_NCALLER));
                        continue;
                    }
                }
            }
            // Values that cross a call may only use callee-saved pool entries.
            uint32_t avail = crosses[vi]
                ? (freemask & ~((1u << A64_NCALLER) - 1u))
                : freemask;
            if (avail == 0) continue;        // no eligible reg: stay in memory
            int pk = 0; while (!(avail & (1u << pk))) pk++;
            freemask &= ~(1u << pk);
            uint8_t regn = a64_pool_reg(pk);
            sm_put(rm, vals[vi], regn);
            home_pk[vi] = (int8_t)pk;
            if (pk >= A64_NCALLER) used_mask |= (1u << (pk - A64_NCALLER));
            act_end[nact] = last_use[vi]; act_pi[nact] = (uint8_t)pk; nact++;
        }
    }
    return used_mask;
}

// ---------------------------------------------------------------------------
// Within-block redundant-load elimination (a64 peephole).
//
// After isel+regalloc the emitted block stream contains many `str Xr,[sp,#off];
// ... ; ldr Xr,[sp,#off]` pairs (terminator-condition spills, block-arg edge
// copies) where the loaded slot is still resident in the very register the load
// targets. We track, per sp-relative frame slot, which physical register
// currently equals that slot's in-memory value, and erase a load that would
// reload a slot into the register that already holds it.
//
// Safety: the cache only ever records sp-relative (base x31) immediate-offset
// slots, which are compiler-private spill slots whose address is never taken,
// so no pointer store can alias them. Any op we do not explicitly model
// invalidates the whole cache, and any uncertain attribute read falls back to a
// full invalidation -- the pass is conservative by construction, so the only
// way it could miscompile is by FAILING to invalidate, which is audited per
// op-type below. The pass is purely structural/deterministic, so it preserves
// the bit-identical self-host fixed point.
// ---------------------------------------------------------------------------
#define A64_SLOTCACHE_CAP 32
typedef struct { int32_t off; uint8_t reg; uint8_t width; } A64SlotReg;

static int a64_attr_i(MLIR_OpHandle op, const char *name, int32_t *out) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    if (a == MLIR_INVALID_HANDLE) return 0;
    *out = (int32_t)MLIR_GetAttributeInteger(a);
    return 1;
}

static int a64sc_find(A64SlotReg *c, size_t n, int32_t off) {
    for (size_t i = 0; i < n; i++) if (c[i].off == off) return (int)i;
    return -1;
}
static void a64sc_drop(A64SlotReg *c, size_t *n, int idx) {
    c[idx] = c[*n - 1]; (*n)--;
}
// Record that slot `off` (of byte width `w`) now equals register `reg`.
static void a64sc_set(A64SlotReg *c, size_t *n, int32_t off, uint8_t reg,
                      uint8_t w) {
    int i = a64sc_find(c, *n, off);
    if (i >= 0) { c[i].reg = reg; c[i].width = w; return; }
    if (*n >= A64_SLOTCACHE_CAP) a64sc_drop(c, n, 0); // FIFO-ish evict
    c[*n].off = off; c[*n].reg = reg; c[*n].width = w; (*n)++;
}
// A register was redefined: any slot believed to live in it is now stale.
static void a64sc_kill_reg(A64SlotReg *c, size_t *n, uint8_t reg) {
    for (size_t i = 0; i < *n; ) {
        if (c[i].reg == reg) a64sc_drop(c, n, (int)i); else i++;
    }
}
// A call clobbers all caller-saved GP registers (x0..x18).
static void a64sc_kill_caller(A64SlotReg *c, size_t *n) {
    for (size_t i = 0; i < *n; ) {
        if (c[i].reg <= 18) a64sc_drop(c, n, (int)i); else i++;
    }
}

static uint8_t a64_ldst_width(MLIR_OpType t) {
    switch (t) {
        case OP_TYPE_AARCH64_LDR_X: case OP_TYPE_AARCH64_STR_X:   return 8;
        case OP_TYPE_AARCH64_LDR_W: case OP_TYPE_AARCH64_STR_W:   return 4;
        case OP_TYPE_AARCH64_LDRB_IMM: case OP_TYPE_AARCH64_STRB_IMM: return 1;
        default: return 0;
    }
}

static void peephole_block(MLIR_Context *ctx, MLIR_BlockHandle blk) {
    A64SlotReg cache[A64_SLOTCACHE_CAP];
    size_t nc = 0;
    size_t nops = MLIR_GetBlockNumOps(blk);
    MLIR_OpHandle *erase = (MLIR_OpHandle *)malloc(nops * sizeof(MLIR_OpHandle));
    size_t nerase = 0;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        MLIR_OpType t = MLIR_GetOpType(op);
        switch (t) {
        case OP_TYPE_AARCH64_LDR_X:
        case OP_TYPE_AARCH64_LDR_W:
        case OP_TYPE_AARCH64_LDRB_IMM: {
            int32_t rn, rt, off;
            uint8_t w = a64_ldst_width(t);
            if (a64_attr_i(op, "rn", &rn) && rn == 31 &&
                a64_attr_i(op, "rt", &rt) && a64_attr_i(op, "off_bytes", &off)) {
                int idx = a64sc_find(cache, nc, off);
                if (idx >= 0 && cache[idx].width == w &&
                    cache[idx].reg == (uint8_t)rt) {
                    erase[nerase++] = op;        // reloads a still-resident slot
                    break;                       // rt unchanged, cache intact
                }
                a64sc_kill_reg(cache, &nc, (uint8_t)rt);
                a64sc_set(cache, &nc, off, (uint8_t)rt, w);
            } else if (a64_attr_i(op, "rt", &rt)) {
                a64sc_kill_reg(cache, &nc, (uint8_t)rt);
            } else {
                nc = 0;
            }
            break;
        }
        case OP_TYPE_AARCH64_STR_X:
        case OP_TYPE_AARCH64_STR_W:
        case OP_TYPE_AARCH64_STRB_IMM: {
            int32_t rn, rt, off;
            if (a64_attr_i(op, "rn", &rn) && rn == 31 &&
                a64_attr_i(op, "rt", &rt) && a64_attr_i(op, "off_bytes", &off)) {
                a64sc_set(cache, &nc, off, (uint8_t)rt, a64_ldst_width(t));
            }
            // non-sp store (pointer store) cannot alias spill slots; no def.
            break;
        }
        case OP_TYPE_AARCH64_LDR_X_REG:
        case OP_TYPE_AARCH64_LDR_W_REG:
        case OP_TYPE_AARCH64_LDRB_REG: {
            int32_t rt;                          // register-offset load: defs rt,
            if (a64_attr_i(op, "rt", &rt))       // unknown slot => just kill rt
                a64sc_kill_reg(cache, &nc, (uint8_t)rt);
            else nc = 0;
            break;
        }
        case OP_TYPE_AARCH64_ADD_IMM: case OP_TYPE_AARCH64_SUB_IMM:
        case OP_TYPE_AARCH64_ADD_REG: case OP_TYPE_AARCH64_SUB_REG:
        case OP_TYPE_AARCH64_AND_IMM: {
            // Redundant low-bits mask: `and Rd,Rd,#((1<<w)-1)` (w>=8) right
            // after a `ldrb Rd,[...]` is a no-op -- ldrb already zero-extends
            // bits 8..63, so masking the low byte (or more) changes nothing.
            // This `and` sits on the hot lexer load->compare dependency chain.
            int32_t rd, rn, w;
            if (!getenv("TINYC_NO_MASK_ELIDE") && i > 0 &&
                a64_attr_i(op, "rd", &rd) && a64_attr_i(op, "rn", &rn) &&
                rd == rn && a64_attr_i(op, "w", &w) && w >= 8) {
                MLIR_OpHandle prev = MLIR_GetBlockOp(blk, i - 1);
                MLIR_OpType pt = MLIR_GetOpType(prev);
                int32_t prt;
                if ((pt == OP_TYPE_AARCH64_LDRB_IMM ||
                     pt == OP_TYPE_AARCH64_LDRB_REG) &&
                    a64_attr_i(prev, "rt", &prt) && prt == rd) {
                    erase[nerase++] = op;   // value already byte-zero-extended
                    break;                  // Rd unchanged; cache stays valid
                }
            }
            int32_t krd;
            if (a64_attr_i(op, "rd", &krd)) a64sc_kill_reg(cache, &nc, (uint8_t)krd);
            else nc = 0;
            break;
        }
        case OP_TYPE_AARCH64_AND_REG: case OP_TYPE_AARCH64_ORR_REG:
        case OP_TYPE_AARCH64_EOR_REG: case OP_TYPE_AARCH64_ASR_REG:
        case OP_TYPE_AARCH64_LSL_REG: case OP_TYPE_AARCH64_LSR_REG:
        case OP_TYPE_AARCH64_MUL: case OP_TYPE_AARCH64_MSUB:
        case OP_TYPE_AARCH64_SDIV: case OP_TYPE_AARCH64_UDIV:
        case OP_TYPE_AARCH64_LSL_IMM: case OP_TYPE_AARCH64_LSR_IMM:
        case OP_TYPE_AARCH64_ASR_IMM:
        case OP_TYPE_AARCH64_CSEL: case OP_TYPE_AARCH64_CSET:
        case OP_TYPE_AARCH64_MOV_X: case OP_TYPE_AARCH64_MOVZ:
        case OP_TYPE_AARCH64_MOVK: case OP_TYPE_AARCH64_ADRP_DATA:
        case OP_TYPE_AARCH64_ADD_DATA_LO:
        case OP_TYPE_AARCH64_SXTB: case OP_TYPE_AARCH64_SXTH:
        case OP_TYPE_AARCH64_SXTW: case OP_TYPE_AARCH64_UXTW: {
            int32_t rd;
            if (a64_attr_i(op, "rd", &rd)) a64sc_kill_reg(cache, &nc, (uint8_t)rd);
            else nc = 0;
            break;
        }
        case OP_TYPE_AARCH64_BL:
        case OP_TYPE_AARCH64_BLR:
            a64sc_kill_caller(cache, &nc);
            break;
        // flags-only / control-flow ops: no GP def, no slot write.
        case OP_TYPE_AARCH64_CMP_IMM: case OP_TYPE_AARCH64_CMP_REG:
        case OP_TYPE_AARCH64_FCMP:
        case OP_TYPE_AARCH64_B: case OP_TYPE_AARCH64_B_COND:
        case OP_TYPE_AARCH64_CBNZ: case OP_TYPE_AARCH64_CBZ:
        case OP_TYPE_AARCH64_RET: case OP_TYPE_AARCH64_LABEL:
            break;
        default:                                 // unmodeled => safe full reset
            nc = 0;
            break;
        }
    }
    for (size_t i = 0; i < nerase; i++) MLIR_EraseOp(ctx, erase[i]);
    free(erase);
}

static void peephole_region(MLIR_Context *ctx, MLIR_RegionHandle reg) {
    if (getenv("TINYC_NO_PEEPHOLE")) return;
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    for (size_t b = 0; b < nb; b++)
        peephole_block(ctx, MLIR_GetRegionBlock(reg, b));
}

// True if value `v` is an integer constant equal to `want`.
static bool operand_is_const_val(LowerCtx *L, MLIR_ValueHandle v, int64_t want) {
    int64_t cv; uint8_t c64;
    return L->cm && cm_get(L->cm, v, &cv, &c64) && cv == want;
}

// Branch-condition fusion plan for a cf.cond_br terminator. tinyC lowers a
// comparison condition as `icmp <pred>` -> `select(.,1,0)` -> `icmp ne(.,0)`
// -> cond_br, ~10 instructions where `cmp; b.<cond>` suffices. We walk the
// condition backward through the redundant boolean ops (each used exactly once
// and defined in this block) to a genuine comparison "root" icmp, accumulating
// a branch-sense inversion, then emit the root as a bare `cmp` and let the
// terminator branch directly on the resulting NZCV flags.
typedef struct {
    bool          fuse;   // valid plan?
    MLIR_OpHandle root;   // icmp lowered cmp-only
    uint8_t       cond;   // ARM cond code for the b.cond (inversion applied)
    // Zero-test fast path: the root is a `icmp eq/ne(x,0)` whose `x` is not
    // further fusible. Instead of `cmp x,#0; b.cond` emit a single `cbz`/`cbnz`
    // on `x` (no flags needed). `root` still names the icmp (skipped, emits
    // nothing); the terminator materialises `zt_val` and branches.
    bool             zt;
    MLIR_ValueHandle zt_val;  // the value `x` tested against zero
    bool             zt_sf;   // 64-bit form (x is i64/ptr) ?
    bool             zt_nz;   // true -> cbnz (branch if non-zero); false -> cbz
} FusePlan;

static FusePlan analyze_cond_fusion(LowerCtx *L, MLIR_OpHandle term,
                                    MLIR_BlockHandle sblk, SlotMap *uc,
                                    SlotMap *skipset_out) {
    FusePlan p = { false, MLIR_INVALID_HANDLE, 0,
                   false, MLIR_INVALID_HANDLE, false, false };
    MLIR_ValueHandle cur = MLIR_GetOpOperand(term, 0);
    int inv = 0;
    MLIR_ValueHandle tmpskip[8];
    int n_skip = 0;
    MLIR_OpHandle root = MLIR_INVALID_HANDLE;
    uint8_t root_cond = 0;
    for (int step = 0; step < 8; step++) {
        int32_t c;
        if (!sm_get(uc, cur, &c) || c != 1) break;   // used only by the chain
        MLIR_OpHandle D = MLIR_GetValueDefiningOp(cur);
        if (D == MLIR_INVALID_HANDLE) break;          // block arg / cross-block
        string dn = MLIR_GetOpName(D);
        if (name_eq(dn, "llvm.icmp")) {
            MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(D, "predicate");
            if (pa == MLIR_INVALID_HANDLE) break;
            int64_t pred = MLIR_GetAttributeInteger(pa);
            bool z1 = operand_is_const_val(L, MLIR_GetOpOperand(D, 1), 0);
            bool z0 = operand_is_const_val(L, MLIR_GetOpOperand(D, 0), 0);
            if ((pred == 0 || pred == 1) && (z0 || z1)) {
                // Redundant zero-test icmp eq/ne (x,0). Forward to x (eq == !x)
                // only when x continues a fusible chain (select/icmp used once
                // in this block). Otherwise emit a single `cbz`/`cbnz` on x
                // (one instruction, no cmp), with the operand width taken from
                // x's type.
                MLIR_ValueHandle x = z1 ? MLIR_GetOpOperand(D, 0)
                                        : MLIR_GetOpOperand(D, 1);
                MLIR_OpHandle XD = MLIR_GetValueDefiningOp(x);
                int32_t xc;
                bool x_fusible = XD != MLIR_INVALID_HANDLE &&
                    sm_get(uc, x, &xc) && xc == 1 &&
                    (name_eq(MLIR_GetOpName(XD), "llvm.select") ||
                     name_eq(MLIR_GetOpName(XD), "llvm.icmp"));
                if (x_fusible) {
                    if (pred == 0) inv ^= 1;
                    if (n_skip >= 8) break;
                    tmpskip[n_skip++] = cur;
                    cur = x;
                    continue;
                }
                int cc0 = icmp_pred_to_cond(pred);  // eq->EQ(0), ne->NE(1)
                if (cc0 < 0) break;
                root = D;
                root_cond = (uint8_t)(inv ? (cc0 ^ 1) : cc0);
                if (!getenv("TINYC_NO_ZEROTEST_CBZ")) {
                    p.zt     = true;
                    p.zt_val = x;
                    p.zt_sf  = type_is_gp64(L->ctx, x);
                    p.zt_nz  = (root_cond == 1);  // NE -> cbnz ; EQ -> cbz
                }
                break;
            }
            int cc = icmp_pred_to_cond(pred);
            if (cc < 0) break;
            root = D;
            root_cond = (uint8_t)(inv ? (cc ^ 1) : cc);
            break;
        } else if (name_eq(dn, "llvm.select")) {
            int64_t tv, fv; uint8_t t64, f64;
            if (!(L->cm && cm_get(L->cm, MLIR_GetOpOperand(D, 1), &tv, &t64)
                       && cm_get(L->cm, MLIR_GetOpOperand(D, 2), &fv, &f64)))
                break;
            if (tv == 1 && fv == 0) { /* sense unchanged */ }
            else if (tv == 0 && fv == 1) { inv ^= 1; }
            else break;
            if (n_skip >= 8) break;
            tmpskip[n_skip++] = cur;
            cur = MLIR_GetOpOperand(D, 0);
            continue;
        }
        break;
    }
    if (root == MLIR_INVALID_HANDLE) return p;
    // Positional check: between `root` and the terminator every op must be a
    // skipped chain op or an llvm.mlir.constant (both emit nothing), so the
    // root cmp's NZCV flags reach the branch. Also confirms root is in sblk.
    size_t no = MLIR_GetBlockNumOps(sblk);
    bool reached = false;
    for (size_t i = no - 1; i > 0; ) {
        i--;
        MLIR_OpHandle o = MLIR_GetBlockOp(sblk, i);
        if (o == root) { reached = true; break; }
        if (name_eq(MLIR_GetOpName(o), "llvm.mlir.constant")) continue;
        bool is_skip = false;
        if (MLIR_GetOpNumResults(o) == 1) {
            MLIR_ValueHandle rv = MLIR_GetOpResult(o, 0);
            for (int k = 0; k < n_skip; k++)
                if (tmpskip[k] == rv) { is_skip = true; break; }
        }
        if (!is_skip) return p;   // a flag-clobbering op intervenes
    }
    if (!reached) return p;
    for (int k = 0; k < n_skip; k++) sm_put(skipset_out, tmpskip[k], 1);
    // Zero-test: the root icmp itself emits nothing (the terminator does the
    // cbz/cbnz on x), so skip it entirely rather than lowering it cmp-only.
    if (p.zt) sm_put(skipset_out, MLIR_GetOpResult(root, 0), 1);
    p.fuse = true;
    p.root = root;
    p.cond = root_cond;
    return p;
}

static MLIR_OpHandle select_func_cfg(MLIR_Context *ctx, MLIR_OpHandle fn,
                                     string sym, GlobalMap *gm) {
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(fn, 0);
    size_t n_blocks = MLIR_GetRegionNumBlocks(src_region);
    MLIR_BlockHandle entry_src = MLIR_GetRegionBlock(src_region, 0);
    size_t nargs = MLIR_GetBlockNumArgs(entry_src);
    if (nargs > 64)
        A64_FAIL("llvm->aarch64: function '%.*s' has %zu parameters "
                 "(>64 not supported)\n", (int)sym.size, sym.str, nargs);

    SlotMap sm = {0};
    sm.arena = MLIR_GetArenaAllocator(ctx);
    int32_t nslots = 0;       // frame slots assigned AFTER register allocation

    SlotMap am = {0};
    am.arena = MLIR_GetArenaAllocator(ctx);
    uint32_t alloca_bytes = 0;
    for (size_t b = 0; b < n_blocks; b++)
        if (!collect_allocas(ctx, &am, MLIR_GetRegionBlock(src_region, b),
                             &alloca_bytes))
            A64_FAIL("llvm->aarch64: function '%.*s' has an unsupported "
                     "alloca\n", (int)sym.size, sym.str);

    // Branch-condition fusion + linmem addressing-mode fusion both need a
    // function-wide use-count map (operands plus successor block-arg operands).
    // Build it BEFORE register allocation so the allocator can model the
    // post-fusion liveness of memfuse'd loads/stores.
    bool do_fuse = !getenv("TINYC_NO_FUSE");
    bool do_memfuse = do_fuse && !getenv("TINYC_NO_MEMFUSE");
    SlotMap uc = {0};   uc.arena = MLIR_GetArenaAllocator(ctx);
    SlotMap skip = {0}; skip.arena = MLIR_GetArenaAllocator(ctx);
    if (do_fuse) {
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle sb = MLIR_GetRegionBlock(src_region, b);
            size_t no = MLIR_GetBlockNumOps(sb);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle o = MLIR_GetBlockOp(sb, i);
                size_t non = MLIR_GetOpNumOperands(o);
                for (size_t k = 0; k < non; k++)
                    uc_inc(&uc, MLIR_GetOpOperand(o, k));
                size_t nsucc = MLIR_GetOpNumSuccessors(o);
                for (size_t s = 0; s < nsucc; s++) {
                    size_t nso = MLIR_GetOpNumSuccessorOperands(o, s);
                    for (size_t k = 0; k < nso; k++)
                        uc_inc(&uc, MLIR_GetOpSuccessorOperand(o, s, k));
                }
            }
        }
    }

    // memfuse pre-pass: find linmem loads/stores whose host pointer is the
    // single-use inttoptr(base + zext(i32 idx)) spine. Record the inttoptr
    // result in `memfuse`, the three spine results in `memspine` (no home, no
    // emitted code), and add the spine results to the lowering `skip` set.
    SlotMap memfuse = {0};  memfuse.arena = MLIR_GetArenaAllocator(ctx);
    SlotMap memspine = {0}; memspine.arena = MLIR_GetArenaAllocator(ctx);
    if (do_memfuse) {
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle sb = MLIR_GetRegionBlock(src_region, b);
            size_t no = MLIR_GetBlockNumOps(sb);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle o = MLIR_GetBlockOp(sb, i);
                string opn = MLIR_GetOpName(o);
                MLIR_ValueHandle P = MLIR_INVALID_HANDLE; unsigned vsz = 0;
                if (name_eq(opn, "llvm.load") && MLIR_GetOpNumOperands(o) >= 1) {
                    P = MLIR_GetOpOperand(o, 0);
                    vsz = a64_type_size(ctx, MLIR_GetValueType(MLIR_GetOpResult(o, 0)));
                } else if (name_eq(opn, "llvm.store") && MLIR_GetOpNumOperands(o) >= 2) {
                    P = MLIR_GetOpOperand(o, 1);
                    vsz = a64_type_size(ctx, MLIR_GetValueType(MLIR_GetOpOperand(o, 0)));
                }
                if (P == MLIR_INVALID_HANDLE) continue;
                if (vsz != 1 && vsz != 4 && vsz != 8) continue; // no LDRH_REG
                MLIR_ValueHandle mb, mi; MLIR_OpHandle z, e, p;
                int64_t coff; MLIR_OpHandle oadd;
                if (!memfuse_match(ctx, P, &mb, &mi, &z, &e, &p, &coff, &oadd))
                    continue;
                // The static offset folds into the 32-bit index via one extra
                // `add Wtmp,Widx,#coff`, so it must fit the add-immediate range.
                if (coff < 0 || coff > 4095) continue;
                // Only fold when the spine is single-use (no other reader of the
                // inttoptr / +const add / base+zext add / zext results); else the
                // spine ops still emit (e.g. base+zext shared across two fields).
                int32_t c;
                if (!sm_get(&uc, P, &c) || c != 1) continue;
                if (oadd != MLIR_INVALID_HANDLE &&
                    (!sm_get(&uc, MLIR_GetOpResult(oadd, 0), &c) || c != 1))
                    continue;
                if (!sm_get(&uc, MLIR_GetOpResult(e, 0), &c) || c != 1) continue;
                if (!sm_get(&uc, MLIR_GetOpResult(z, 0), &c) || c != 1) continue;
                sm_put(&memfuse, P, (int32_t)coff);
                sm_put(&memspine, P, 1);
                sm_put(&memspine, MLIR_GetOpResult(e, 0), 1);
                sm_put(&memspine, MLIR_GetOpResult(z, 0), 1);
                sm_put(&skip, P, 1);
                sm_put(&skip, MLIR_GetOpResult(e, 0), 1);
                sm_put(&skip, MLIR_GetOpResult(z, 0), 1);
                if (oadd != MLIR_INVALID_HANDLE) {
                    sm_put(&memspine, MLIR_GetOpResult(oadd, 0), 1);
                    sm_put(&skip, MLIR_GetOpResult(oadd, 0), 1);
                }
            }
        }
    }

    // shift-fusion pre-pass: fold a single-use `shl(x,C)` / `mul(x,2^C)` addend
    // of an `llvm.add` into one `add Rd, Ra, Rx, lsl #amt`. Record the add
    // result in `shiftfuse` ((amt<<1)|operand_index) and add the shl/mul result
    // to `skip` (no code) + `memspine` (no home, operand reads suppressed; x is
    // re-injected as a live use at the add via a64_shiftfuse_uses).
    SlotMap shiftfuse = {0}; shiftfuse.arena = MLIR_GetArenaAllocator(ctx);
    bool do_shiftfuse = do_fuse && !getenv("TINYC_NO_SHIFTFUSE");
    if (do_shiftfuse) {
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle sb = MLIR_GetRegionBlock(src_region, b);
            size_t no = MLIR_GetBlockNumOps(sb);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle o = MLIR_GetBlockOp(sb, i);
                if (!name_eq(MLIR_GetOpName(o), "llvm.add")) continue;
                if (MLIR_GetOpNumResults(o) != 1) continue;
                MLIR_ValueHandle ares = MLIR_GetOpResult(o, 0);
                int32_t st;
                if (sm_get(&skip, ares, &st)) continue;  // already a fused spine add
                int sidx; unsigned amt;
                if (!shiftfuse_match(ctx, o, &uc, &sidx, &amt)) continue;
                MLIR_OpHandle sop =
                    MLIR_GetValueDefiningOp(MLIR_GetOpOperand(o, sidx));
                MLIR_ValueHandle sres = MLIR_GetOpResult(sop, 0);
                if (sm_get(&memspine, sres, &st)) continue;  // shift already folded
                sm_put(&shiftfuse, ares, (int32_t)((amt << 1) | (unsigned)sidx));
                sm_put(&skip, sres, 1);
                sm_put(&memspine, sres, 1);
            }
        }
    }

    // global-pinning pre-pass: fold a load/store whose pointer is a single-use
    // `addressof @G` of a small-offset data global into one `ldr/str [x27,#off]`
    // (x27 = reserved data base, set in synth_start). Drops the per-access
    // adrp+add (3 insns -> 1). The addressof result goes in skip (no emission)
    // and memspine (no allocation); the offset is recorded in `gfuse`. Mirrors
    // the deleted WMIR backend's x27 globals base. Gated on a data section
    // existing so x27 is always validly set up.
    SlotMap gfuse = {0}; gfuse.arena = MLIR_GetArenaAllocator(ctx);
    bool do_globalpin = do_fuse && !getenv("TINYC_NO_GLOBALPIN") &&
                        gm && gm->n > 0;
    uint32_t gp_anchor = gfuse_anchor(gm);
    if (do_globalpin) {
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle sb = MLIR_GetRegionBlock(src_region, b);
            size_t no = MLIR_GetBlockNumOps(sb);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle o = MLIR_GetBlockOp(sb, i);
                string opn = MLIR_GetOpName(o);
                MLIR_ValueHandle P = MLIR_INVALID_HANDLE; unsigned vsz = 0;
                if (name_eq(opn, "llvm.load") && MLIR_GetOpNumOperands(o) >= 1) {
                    P = MLIR_GetOpOperand(o, 0);
                    vsz = a64_type_size(ctx, MLIR_GetValueType(MLIR_GetOpResult(o, 0)));
                } else if (name_eq(opn, "llvm.store") && MLIR_GetOpNumOperands(o) >= 2) {
                    P = MLIR_GetOpOperand(o, 1);
                    vsz = a64_type_size(ctx, MLIR_GetValueType(MLIR_GetOpOperand(o, 0)));
                }
                if (P == MLIR_INVALID_HANDLE) continue;
                uint32_t goff;
                if (!gfuse_match(ctx, gm, &uc, P, vsz, gp_anchor, &goff))
                    continue;
                sm_put(&gfuse, P, (int32_t)goff);
                sm_put(&memspine, P, 1);   // addressof result: no home register
                sm_put(&skip, P, 1);       // addressof op: not emitted
            }
        }
    }

    // Linear-scan register allocation over callee-saved x19..x26 (CFG path).
    SlotMap rm = {0};
    rm.arena = MLIR_GetArenaAllocator(ctx);

    // Base-register pinning: the wasm->llvm lifter loads the linear-memory base
    // (`llvm.load` of `addressof @__wasm_linmem_base`) once per function and
    // every linmem access adds an index to it. synth_start sets x28 = base and,
    // because x28 is reserved out of the allocation pool (above), no generated
    // function ever clobbers it — so the base value persists program-wide. Pin
    // every base load's result to x28 (rm) and drop the load + its addressof
    // from emission (skip) and from allocation (memspine), eliminating a
    // per-function base load plus a per-access reload in functions that spill.
    // Gated on do_fuse so the allocator (memspine) and the lowering skip-check
    // stay consistent (the skip-check is do_fuse-gated below).
    if (do_fuse && !getenv("TINYC_NO_BASEPIN")) {
        for (size_t b = 0; b < n_blocks; b++) {
            MLIR_BlockHandle sb = MLIR_GetRegionBlock(src_region, b);
            size_t no = MLIR_GetBlockNumOps(sb);
            for (size_t i = 0; i < no; i++) {
                MLIR_OpHandle o = MLIR_GetBlockOp(sb, i);
                MLIR_OpHandle aof = MLIR_INVALID_HANDLE;
                if (!is_linmem_base_load(ctx, o, &aof)) continue;
                MLIR_ValueHandle lr = MLIR_GetOpResult(o, 0);
                sm_put(&rm, lr, 28);
                sm_put(&memspine, lr, 1);
                sm_put(&skip, lr, 1);
                // Drop the addressof too when it feeds only this load.
                int32_t c;
                MLIR_ValueHandle ar = MLIR_GetOpResult(aof, 0);
                if (sm_get(&uc, ar, &c) && c == 1) {
                    sm_put(&memspine, ar, 1);
                    sm_put(&skip, ar, 1);
                }
            }
        }
    }

    uint32_t saved_mask = 0;
    bool global_ok = false;
    if (!getenv("TINYC_NO_REGALLOC")) {
        // Global (whole-function) linear scan by default; it keeps cross-block
        // and loop-carried values resident across their live range. Very large
        // functions bail (A64_GLOBAL_BAIL) and fall back to the block-local
        // allocator, as does TINYC_BLOCK_REGALLOC=1 (A/B debugging aid).
        saved_mask = A64_GLOBAL_BAIL;
        if (!getenv("TINYC_BLOCK_REGALLOC"))
            saved_mask = alloc_regs_global(ctx, src_region, n_blocks, &rm,
                                           do_memfuse ? &memfuse : NULL,
                                           &memspine,
                                           do_shiftfuse ? &shiftfuse : NULL,
                                           &sm, &nslots);
        global_ok = (saved_mask != A64_GLOBAL_BAIL);
        if (!global_ok)
            saved_mask = alloc_regs_cfg(ctx, src_region, n_blocks, &rm,
                                        do_memfuse ? &memfuse : NULL,
                                        &memspine,
                                        do_shiftfuse ? &shiftfuse : NULL);
    }

    // Assign frame slots. On the global path alloc_regs_global already filled
    // `sm` with REUSED slots for spilled values and set `nslots`; the post-pass
    // only places allocator-missed values (params under no-homing, and the
    // emit-no-code skip/memspine values) into isolated unique slots. On every
    // other path (block-local, global bail, or regalloc disabled) fall back to
    // the simple unique-slot-per-value assignment (no reuse) which the
    // block-local path and giant bailed functions still use.
    if (global_ok) {
        for (size_t b = 0; b < n_blocks; b++)
            assign_post_slots(ctx, &sm, &rm,
                              MLIR_GetRegionBlock(src_region, b), &nslots);
    } else {
        for (size_t b = 0; b < n_blocks; b++)
            assign_slots_block(ctx, &sm, MLIR_GetRegionBlock(src_region, b),
                               &nslots);
    }
    if (getenv("TINYC_SLOT_STATS"))
        fprintf(stderr, "SLOTSTAT %.*s nslots=%d blocks=%zu global=%d\n",
                (int)sym.size, sym.str, nslots, n_blocks, global_ok);
    if (nslots > (1 << 20))
        A64_FAIL("llvm->aarch64: function '%.*s' needs %d slots "
                 "(frame too large for the trivial allocator)\n",
                 (int)sym.size, sym.str, nslots);
    uint32_t slot_bytes = (uint32_t)nslots * 8u;
    uint32_t base_bytes = slot_bytes + alloca_bytes;

    uint32_t num_saved = 0;
    for (uint32_t mtmp = saved_mask; mtmp; mtmp &= mtmp - 1) num_saved++;
    uint32_t save_base = (base_bytes + 7u) & ~7u;
    uint32_t frame_size = (save_base + num_saved * 8u + 15u) & ~15u;
    if (frame_size > 0xff0000u)
        A64_FAIL("llvm->aarch64: function '%.*s' frame %u too large\n",
                 (int)sym.size, sym.str, frame_size);

    MLIR_RegionHandle out_reg = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle *src_blks = (MLIR_BlockHandle *)malloc(
        n_blocks * sizeof(MLIR_BlockHandle));
    MLIR_BlockHandle *out_blks = (MLIR_BlockHandle *)malloc(
        n_blocks * sizeof(MLIR_BlockHandle));
    for (size_t b = 0; b < n_blocks; b++) {
        src_blks[b] = MLIR_GetRegionBlock(src_region, b);
        out_blks[b] = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, out_reg, out_blks[b]);
    }

    // Build the constant-rematerialisation map, then the LowerCtx (needed by the
    // entry-param parallel move below).
    ConstMap cm = {0};
    cm.arena = MLIR_GetArenaAllocator(ctx);
    for (size_t b = 0; b < n_blocks; b++)
        build_const_map(ctx, &cm, MLIR_GetRegionBlock(src_region, b));

    LowerCtx L = { ctx, &sm, out_reg, out_blks[0], sym, frame_size,
                   &am, slot_bytes, gm, nargs, &rm, &cm,
                   do_fuse ? &skip : NULL, MLIR_INVALID_HANDLE,
                   do_memfuse ? &memfuse : NULL,
                   do_shiftfuse ? &shiftfuse : NULL,
                   do_globalpin ? &gfuse : NULL, true };

    // Prologue + incoming-param placement in the entry block. Now that the
    // allocator pool includes the argument registers x0..x8, a param may be
    // homed in an argument register that ALSO carries another incoming param,
    // so the reg-param placement is a general parallel move (sources x0..x7,
    // destinations homed reg or frame slot) resolved through resolve_parallel.
    // Stack params (i>=8) arrive in memory at [x29,#16+...], so their loads
    // never alias an arg register and stay sequential. emit_callee_saves ran
    // first, so a param homed in a callee-saved register overwrites an
    // already-saved value.
    emit_prologue(ctx, out_blks[0], frame_size);
    emit_callee_saves(ctx, out_blks[0], saved_mask, save_base, false);
    L.cur = out_blks[0];
    {
        size_t nreg = nargs < 8 ? nargs : 8;
        if (nreg) {
            Arena *par = MLIR_GetArenaAllocator(ctx);
            Loc *psrc = (Loc *)arena_alloc(par, nreg * sizeof(Loc));
            Loc *pds  = (Loc *)arena_alloc(par, nreg * sizeof(Loc));
            uint8_t *pdone = (uint8_t *)arena_alloc(par, nreg * sizeof(uint8_t));
            for (size_t i = 0; i < nreg; i++) {
                MLIR_ValueHandle pv = MLIR_GetBlockArg(entry_src, i);
                psrc[i].k = LK_REG; psrc[i].reg = (uint8_t)i;
                pds[i] = edge_dst_loc(&L, pv);
                if (pds[i].k == LK_NONE)
                    A64_FAIL("llvm->aarch64: undefined param in '%.*s'\n",
                             (int)sym.size, sym.str);
                pdone[i] = loc_eq(psrc[i], pds[i]) ? 1 : 0;
            }
            resolve_parallel(&L, psrc, pds, pdone, nreg, /*sb=*/31);
        }
    }
    for (size_t i = 8; i < nargs; i++) {
        MLIR_ValueHandle pv = MLIR_GetBlockArg(entry_src, i);
        int32_t hr;
        if (sm_get(&rm, pv, &hr)) {
            emit_ldr_x_off(ctx, out_blks[0], (uint8_t)hr, 29,
                           (uint32_t)(16u + (i - 8) * 8u));
        } else {
            emit_ldr_x_off(ctx, out_blks[0], 9, 29, (uint32_t)(16u + (i - 8) * 8u));
            store_value(ctx, out_blks[0], &sm, pv, 9);
        }
    }

    for (size_t b = 0; b < n_blocks; b++) {
        L.cur = out_blks[b];
        MLIR_BlockHandle sb = src_blks[b];
        size_t no = MLIR_GetBlockNumOps(sb);
        if (no == 0) {
            fprintf(stderr, "llvm->aarch64: empty block in '%.*s'\n",
                    (int)sym.size, sym.str);
            free(src_blks); free(out_blks);
            return MLIR_INVALID_HANDLE;
        }
        // Compute the branch-fusion plan for this block's terminator before
        // lowering its ops (so folded chain ops are skipped and the root icmp
        // is lowered cmp-only).
        FusePlan plan = { false, MLIR_INVALID_HANDLE, 0 };
        MLIR_OpHandle term0 = MLIR_GetBlockOp(sb, no - 1);
        if (do_fuse && name_eq(MLIR_GetOpName(term0), "cf.cond_br") &&
            MLIR_GetOpNumOperands(term0) >= 1) {
            plan = analyze_cond_fusion(&L, term0, sb, &uc, &skip);
        }
        L.fuse_root = plan.fuse ? plan.root : MLIR_INVALID_HANDLE;
        for (size_t i = 0; i + 1 < no; i++) {
            MLIR_OpHandle o = MLIR_GetBlockOp(sb, i);
            if (do_fuse && MLIR_GetOpNumResults(o) == 1) {
                int32_t dummy;
                if (sm_get(&skip, MLIR_GetOpResult(o, 0), &dummy)) continue;
            }
            lower_op(&L, o);
            if (!L.ok) { free(src_blks); free(out_blks); return MLIR_INVALID_HANDLE; }
        }
        MLIR_OpHandle term = term0;
        string tn = MLIR_GetOpName(term);
        if (name_eq(tn, "llvm.return")) {
            size_t nr = MLIR_GetOpNumOperands(term);
            if (nr == 1) {
                if (!mat_into(&L, MLIR_GetOpOperand(term, 0), 0)) {
                    fprintf(stderr, "llvm->aarch64: undefined return value "
                            "in '%.*s'\n", (int)sym.size, sym.str);
                    free(src_blks); free(out_blks);
                    return MLIR_INVALID_HANDLE;
                }
            } else if (nr > 1) {
                fprintf(stderr, "llvm->aarch64: multi-value return\n");
                free(src_blks); free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
            emit_callee_saves(ctx, L.cur, saved_mask, save_base, true);
            emit_epilogue(ctx, L.cur, frame_size);
            emit_ret(ctx, L.cur);
        } else if (name_eq(tn, "cf.br")) {
            emit_edge_copies(&L, term, 0);
            if (!L.ok) { free(src_blks); free(out_blks); return MLIR_INVALID_HANDLE; }
            MLIR_BlockHandle d = map_block(src_blks, out_blks, n_blocks,
                                           MLIR_GetOpSuccessor(term, 0));
            emit_b(ctx, L.cur, d);
        } else if (name_eq(tn, "cf.cond_br")) {
            if (MLIR_GetOpNumOperands(term) < 1) {
                fprintf(stderr, "llvm->aarch64: cf.cond_br missing condition "
                        "in '%.*s'\n", (int)sym.size, sym.str);
                free(src_blks); free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
            // When the condition was fused, the root cmp has already been
            // emitted (cmp-only) into this block and NZCV is live; branch on
            // the flags. Otherwise materialise the condition and test for !=0.
            if (!plan.fuse) {
                if (!mat_into(&L, MLIR_GetOpOperand(term, 0), 9)) {
                    fprintf(stderr, "llvm->aarch64: undefined cond_br condition "
                            "in '%.*s'\n", (int)sym.size, sym.str);
                    free(src_blks); free(out_blks);
                    return MLIR_INVALID_HANDLE;
                }
            }
            MLIR_BlockHandle real_t = map_block(src_blks, out_blks, n_blocks,
                                                MLIR_GetOpSuccessor(term, 0));
            MLIR_BlockHandle real_f = map_block(src_blks, out_blks, n_blocks,
                                                MLIR_GetOpSuccessor(term, 1));
            size_t nso_t = MLIR_GetOpNumSuccessorOperands(term, 0);
            size_t nso_f = MLIR_GetOpNumSuccessorOperands(term, 1);
            // Edges that carry block-arg operands get a landing block that
            // performs the copies before jumping to the real successor; this
            // splits exactly the operand-carrying edges (no critical-edge
            // hazard). Operand-free edges branch straight to the successor.
            MLIR_BlockHandle br_t = real_t, br_f = real_f;
            MLIR_BlockHandle cur = L.cur;
            if (nso_t) {
                br_t = MLIR_CreateBlock(ctx);
                MLIR_AppendRegionBlock(ctx, out_reg, br_t);
            }
            if (nso_f) {
                br_f = MLIR_CreateBlock(ctx);
                MLIR_AppendRegionBlock(ctx, out_reg, br_f);
            }
            if (plan.fuse) {
                if (plan.zt) {
                    uint8_t r = use_val(&L, plan.zt_val, 9);
                    if (!L.ok) { free(src_blks); free(out_blks); return MLIR_INVALID_HANDLE; }
                    if (plan.zt_nz) emit_cbnz(ctx, cur, r, plan.zt_sf, br_t);
                    else            emit_cbz(ctx, cur, r, plan.zt_sf, br_t);
                } else {
                    emit_bcond(ctx, cur, plan.cond, br_t);
                }
            } else {
                emit_cbnz(ctx, cur, 9, false, br_t);
            }
            emit_b(ctx, cur, br_f);
            if (nso_t) {
                L.cur = br_t;
                emit_edge_copies(&L, term, 0);
                if (!L.ok) { free(src_blks); free(out_blks); return MLIR_INVALID_HANDLE; }
                emit_b(ctx, br_t, real_t);
            }
            if (nso_f) {
                L.cur = br_f;
                emit_edge_copies(&L, term, 1);
                if (!L.ok) { free(src_blks); free(out_blks); return MLIR_INVALID_HANDLE; }
                emit_b(ctx, br_f, real_f);
            }
        } else {
            fprintf(stderr, "llvm->aarch64: block in '%.*s' ends in "
                    "non-terminator '%.*s'\n",
                    (int)sym.size, sym.str, (int)tn.size, tn.str);
            free(src_blks); free(out_blks);
            return MLIR_INVALID_HANDLE;
        }
    }
    free(src_blks);
    free(out_blks);

    peephole_region(ctx, out_reg);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", sym.str, sym.size);
    MLIR_RegionHandle regs[1] = { out_reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// line integer code plus structured control flow (scf.if / scf.while /
// scf.index_switch). Uses the trivial spill-everything allocator: every SSA
// value (incl. block args / scf results) lives in a frame slot, so phi /
// block-arg resolution is a slot-to-slot copy at each edge.
// ---------------------------------------------------------------------------
static MLIR_OpHandle select_func(MLIR_Context *ctx, MLIR_OpHandle fn,
                                 string sym, GlobalMap *gm) {
    MLIR_RegionHandle src_region = MLIR_GetOpRegion(fn, 0);
    if (MLIR_GetRegionNumBlocks(src_region) > 1)
        return select_func_cfg(ctx, fn, sym, gm);
    if (MLIR_GetRegionNumBlocks(src_region) != 1) {
        A64_FAIL("llvm->aarch64: function '%.*s' has a non-structured CFG "
                 "(expected a single entry block with scf control flow)\n",
                 (int)sym.size, sym.str);
    }
    MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(src_region, 0);
    size_t nargs = MLIR_GetBlockNumArgs(src_blk);
    if (nargs > 64) {
        A64_FAIL("llvm->aarch64: function '%.*s' has %zu parameters "
                 "(>64 not supported)\n",
                 (int)sym.size, sym.str, nargs);
    }

    MLIR_BlockHandle  out_blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_reg, out_blk);

    SlotMap sm = {0};
    sm.arena = MLIR_GetArenaAllocator(ctx);
    int32_t nslots = 0;
    assign_slots_block(ctx, &sm, src_blk, &nslots);
    if (nslots > (1 << 20)) {
        A64_FAIL("llvm->aarch64: function '%.*s' needs %d slots "
                 "(frame too large for the trivial allocator)\n",
                 (int)sym.size, sym.str, nslots);
    }
    uint32_t slot_bytes = (uint32_t)nslots * 8u;

    SlotMap am = {0};
    am.arena = MLIR_GetArenaAllocator(ctx);
    uint32_t alloca_bytes = 0;
    if (!collect_allocas(ctx, &am, src_blk, &alloca_bytes)) {
        A64_FAIL("llvm->aarch64: function '%.*s' has an unsupported alloca\n",
                 (int)sym.size, sym.str);
    }
    uint32_t frame_size = (slot_bytes + alloca_bytes + 15u) & ~15u;
    if (frame_size > 0xff0000u)
        A64_FAIL("llvm->aarch64: function '%.*s' frame %u too large\n",
                 (int)sym.size, sym.str, frame_size);

    emit_prologue(ctx, out_blk, frame_size);
    for (size_t i = 0; i < nargs && i < 8; i++)
        store_value(ctx, out_blk, &sm, MLIR_GetBlockArg(src_blk, i), (uint8_t)i);
    // Args 8.. arrive on the caller's stack. After the prologue x29 (fp) points
    // at the saved {fp,lr} pair, so the first stack arg is at [x29, #16].
    for (size_t i = 8; i < nargs; i++) {
        emit_ldr_x_off(ctx, out_blk, 9, 29, (uint32_t)(16u + (i - 8) * 8u));
        store_value(ctx, out_blk, &sm, MLIR_GetBlockArg(src_blk, i), 9);
    }

    ConstMap cm = {0};
    cm.arena = MLIR_GetArenaAllocator(ctx);
    build_const_map(ctx, &cm, src_blk);

    LowerCtx L = { ctx, &sm, out_reg, out_blk, sym, frame_size,
                   &am, slot_bytes, gm, nargs, NULL, &cm,
                   NULL, MLIR_INVALID_HANDLE, NULL, NULL, NULL, true };
    MLIR_OpHandle term = lower_block_ops(&L, src_blk);
    if (!L.ok) return MLIR_INVALID_HANDLE;

    // Function terminator: llvm.return.
    if (!name_eq(MLIR_GetOpName(term), "llvm.return"))
        A64_FAIL("llvm->aarch64: function '%.*s' does not end in llvm.return\n",
                 (int)sym.size, sym.str);
    size_t no = MLIR_GetOpNumOperands(term);
    if (no == 1) {
        if (!mat_into(&L, MLIR_GetOpOperand(term, 0), 0))
            A64_FAIL("llvm->aarch64: undefined return value in '%.*s'\n",
                     (int)sym.size, sym.str);
    } else if (no > 1) {
        A64_FAIL("llvm->aarch64: multi-value return\n");
    }
    emit_epilogue(ctx, L.cur, frame_size);
    emit_ret(ctx, L.cur);

    peephole_region(ctx, out_reg);

    MLIR_AttributeHandle attrs[1];
    attrs[0] = attr_s(ctx, "sym_name", sym.str, sym.size);
    MLIR_RegionHandle regs[1] = { out_reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 1, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// Synthesise `_start`: initialise pointer globals (PC-relative, so they work
// under ASLR), then `bl main; bl _exit`. Exported and emitted first so the
// Mach-O encoder lands LC_MAIN.entryoff on it.
//
// For the `--from-wasm` path the lifter emits a `__wasm_linmem_base` i64
// global; when present we additionally:
//   * mmap a 4 GiB lazily-backed anonymous region (the wasm32 linear memory),
//     store its base into `__wasm_linmem_base`, and memcpy the initialised
//     data image (the `__wasm_linmem` template) into its front;
//   * stash the process argc/argv into `__wasm_argc` / `__wasm_argv` so the
//     WASI args_get / args_sizes_get shims can recover them.
// The 4 GiB reservation means `memory.grow` never needs real allocation
// (pages fault in on first touch), mirroring a conventional crt0.
static MLIR_OpHandle synth_start(MLIR_Context *ctx, string main_name,
                                 PtrReloc *relocs, size_t n_relocs,
                                 GlobalMap *gm) {
    MLIR_BlockHandle  blk = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle reg = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, reg, blk);

    string tgt = str_from_cstr_view("linmem_template");

    // wasm linear-memory setup (from-wasm path only).
    uint32_t base_off = 0, base_sz = 0;
    bool has_linmem = gm && gmap_get_cstr(gm, "__wasm_linmem_base",
                                          &base_off, &base_sz);
    uint32_t tmpl_off = 0, tmpl_sz = 0;
    bool has_tmpl = gm && gmap_get_cstr(gm, "__wasm_linmem",
                                        &tmpl_off, &tmpl_sz);
    uint32_t argc_off = 0, argv_off = 0;
    bool has_argc = gm && gmap_get_cstr(gm, "__wasm_argc", &argc_off, NULL) &&
                    gmap_get_cstr(gm, "__wasm_argv", &argv_off, NULL);

    if (has_linmem) {
        // Save dyld-supplied argc (x0) / argv (x1) across the mmap/memcpy
        // calls (which clobber x0..x18). 16-byte frame: [sp,#0]=argc, [sp,#8]=argv.
        emit_prologue(ctx, blk, 16);
        emit_str_x(ctx, blk, 0, 31, 0);
        emit_str_x(ctx, blk, 1, 31, 8);

        // mmap(NULL, 4 GiB, PROT_READ|WRITE, MAP_ANON|MAP_PRIVATE, -1, 0).
        emit_movz(ctx, blk, 0, 0, 0, true);        // x0 = NULL
        emit_movz(ctx, blk, 1, 1, 2, true);        // x1 = 1<<32 = 4 GiB
        emit_movz(ctx, blk, 2, 3, 0, false);       // w2 = PROT_READ|PROT_WRITE
        emit_movz(ctx, blk, 3, 0x1002, 0, false);  // w3 = MAP_ANON|MAP_PRIVATE
        emit_movz(ctx, blk, 4, 0xffff, 0, false);  // w4 = -1 (fd)
        emit_movk(ctx, blk, 4, 0xffff, 1, false);
        emit_movz(ctx, blk, 5, 0, 0, true);        // x5 = 0 (offset)
        emit_bl(ctx, blk, str_lit("_mmap"));
        emit_mov_x(ctx, blk, 28, 0);               // x28 = base (callee-saved)

        // __wasm_linmem_base = base.
        emit_adrp_data(ctx, blk, 10, tgt, base_off);
        emit_add_data_lo(ctx, blk, 10, 10, tgt, base_off);
        emit_str_x(ctx, blk, 28, 10, 0);

        // memcpy(base, __wasm_linmem_template, init_size).
        if (has_tmpl && tmpl_sz > 0) {
            emit_mov_x(ctx, blk, 0, 28);
            emit_adrp_data(ctx, blk, 1, tgt, tmpl_off);
            emit_add_data_lo(ctx, blk, 1, 1, tgt, tmpl_off);
            emit_movz(ctx, blk, 2, (uint16_t)(tmpl_sz & 0xffff), 0, true);
            if (tmpl_sz >> 16)
                emit_movk(ctx, blk, 2, (uint16_t)((tmpl_sz >> 16) & 0xffff), 1, true);
            emit_bl(ctx, blk, str_lit("_memcpy"));
        }

        // Reload argc/argv and stash into their globals.
        if (has_argc) {
            emit_ldr_x(ctx, blk, 0, 31, 0);
            emit_ldr_x(ctx, blk, 1, 31, 8);
            emit_adrp_data(ctx, blk, 10, tgt, argc_off);
            emit_add_data_lo(ctx, blk, 10, 10, tgt, argc_off);
            emit_ldst_x(ctx, blk, OP_TYPE_AARCH64_STR_W, 0, 10, 0);
            emit_adrp_data(ctx, blk, 10, tgt, argv_off);
            emit_add_data_lo(ctx, blk, 10, 10, tgt, argv_off);
            emit_str_x(ctx, blk, 1, 10, 0);
        }
        emit_epilogue(ctx, blk, 16);
    }

    // For each pointer global, compute target address (PC-relative) and the
    // slot address (PC-relative), then store target into the slot.
    for (size_t k = 0; k < n_relocs; k++) {
        emit_adrp_data(ctx, blk, 9, tgt, relocs[k].target_off);
        emit_add_data_lo(ctx, blk, 9, 9, tgt, relocs[k].target_off);
        emit_adrp_data(ctx, blk, 10, tgt, relocs[k].slot_off);
        emit_add_data_lo(ctx, blk, 10, 10, tgt, relocs[k].slot_off);
        emit_str_x(ctx, blk, 9, 10, 0);
    }

    // Pin x27 = the wasm-globals cluster base (mirrors x28 = linmem base). x27 is
    // reserved out of the allocation pool and callee-saved, so this single setup
    // flows program-wide; every scalar wasm-global access then folds to one
    // ldr/str [x27,#off] (see global-pinning in select_func_cfg). The cluster
    // sits at a ~4.39 MB offset (past the linmem template), so x27 is anchored at
    // that offset (gfuse_anchor), not at the data-section start. Emitted last,
    // after all external startup calls (mmap/memcpy preserve x27 anyway), before
    // `bl main`. Only when a data section exists, so the relocation resolves.
    if (gm && gm->n > 0) {
        uint32_t anchor = gfuse_anchor(gm);
        emit_adrp_data(ctx, blk, 27, tgt, anchor);
        emit_add_data_lo(ctx, blk, 27, 27, tgt, anchor);
    }

    emit_bl(ctx, blk, main_name);
    emit_bl(ctx, blk, str_lit("_exit"));

    MLIR_AttributeHandle attrs[2];
    attrs[0] = attr_s(ctx, "sym_name", "_start", 6);
    attrs[1] = attr_b(ctx, "exported", true);
    MLIR_RegionHandle regs[1] = { reg };
    return MLIR_CreateOp(ctx, OP_TYPE_AARCH64_FUNC,
        op_type_to_string(OP_TYPE_AARCH64_FUNC),
        attrs, 2, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

MLIR_OpHandle mlir_llvm_to_aarch64(MLIR_Context *ctx,
                                   MLIR_OpHandle llvm_module) {
    MLIR_RegionHandle mr = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    MLIR_BlockHandle  out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);

    // Pre-scan module-level globals into a single native data blob.
    GlobalMap gm = {0};
    uint8_t  *gblob = NULL;
    uint32_t  gblob_len = 0;
    PtrReloc *grelocs = NULL;
    size_t    n_grelocs = 0;
    if (!collect_globals(ctx, mb, &gm, &gblob, &gblob_len,
                         &grelocs, &n_grelocs)) {
        fprintf(stderr, "llvm->aarch64: unsupported module-level global\n");
        free(gm.e); free(gblob); free(grelocs);
        return MLIR_INVALID_HANDLE;
    }

    // `_start` first (entry point); it initialises pointer globals then
    // calls main. PC-relative init keeps things correct under ASLR.
    MLIR_AppendBlockOp(ctx, out_body,
        synth_start(ctx, str_lit("main"), grelocs, n_grelocs, &gm));

    bool saw_main = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        if (!func_has_body(op)) continue;  // skip declarations (malloc/free).
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) {
            fprintf(stderr, "llvm->aarch64: llvm.func without sym_name\n");
            free(gm.e); free(gblob); free(grelocs);
            return MLIR_INVALID_HANDLE;
        }
        string sym = MLIR_GetAttributeString(sa);
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) saw_main = true;
        MLIR_OpHandle fn = select_func(ctx, op, sym, &gm);
        if (fn == MLIR_INVALID_HANDLE) { free(gm.e); free(gblob); free(grelocs); return MLIR_INVALID_HANDLE; }
        MLIR_AppendBlockOp(ctx, out_body, fn);
    }

    // Emit the global data blob as one data_init record (the encoder copies
    // these bytes into the __linmem_template data section).
    if (gblob_len > 0) {
        MLIR_AttributeHandle a[2];
        a[0] = attr_i32(ctx, "offset", 0);
        a[1] = attr_s(ctx, "init_data", (const char *)gblob, gblob_len);
        MLIR_AppendBlockOp(ctx, out_body,
            build_op(ctx, OP_TYPE_AARCH64_DATA_INIT, a, 2));
    }
    free(gm.e); free(gblob); free(grelocs);

    if (!saw_main) {
        fprintf(stderr, "llvm->aarch64: no defined 'main' function\n");
        return MLIR_INVALID_HANDLE;
    }

    MLIR_RegionHandle regs[1] = { out_region };
    return MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// ===========================================================================
// Streaming selection API (see mlir_llvm_to_aarch64.h). Reuses the static
// collect_globals / select_func / synth_start helpers above; the only new
// state is the precomputed global blob/relocs and the list of defined funcs.
// ===========================================================================
struct LlvmSelState {
    MLIR_BlockHandle mb;
    GlobalMap        gm;
    uint8_t         *gblob;
    uint32_t         gblob_len;
    PtrReloc        *grelocs;
    size_t           n_grelocs;
    MLIR_OpHandle   *funcs;   // defined functions, in source order
    string          *syms;    // their sym_name strings (point into llvm IR)
    size_t           n_funcs;
    bool             saw_main;
};

LlvmSelState *mlir_llvm_sel_begin(MLIR_Context *ctx, MLIR_OpHandle llvm_module,
                                  uint8_t **out_gblob, uint32_t *out_gblob_len) {
    if (out_gblob) *out_gblob = NULL;
    if (out_gblob_len) *out_gblob_len = 0;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(llvm_module, 0);
    MLIR_BlockHandle  mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    LlvmSelState *st = (LlvmSelState *)calloc(1, sizeof(LlvmSelState));
    if (!st) return NULL;
    st->mb = mb;

    if (!collect_globals(ctx, mb, &st->gm, &st->gblob, &st->gblob_len,
                         &st->grelocs, &st->n_grelocs)) {
        fprintf(stderr, "llvm->aarch64: unsupported module-level global\n");
        free(st->gm.e); free(st->gblob); free(st->grelocs); free(st);
        return NULL;
    }

    st->funcs = (MLIR_OpHandle *)calloc(nops ? nops : 1, sizeof(MLIR_OpHandle));
    st->syms  = (string *)calloc(nops ? nops : 1, sizeof(string));
    if (!st->funcs || !st->syms) {
        free(st->funcs); free(st->syms);
        free(st->gm.e); free(st->gblob); free(st->grelocs); free(st);
        return NULL;
    }
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (!name_eq(MLIR_GetOpName(op), "llvm.func")) continue;
        if (!func_has_body(op)) continue;  // skip declarations (malloc/free).
        MLIR_AttributeHandle sa = MLIR_GetOpAttributeByName(op, "sym_name");
        if (sa == MLIR_INVALID_HANDLE) {
            fprintf(stderr, "llvm->aarch64: llvm.func without sym_name\n");
            free(st->funcs); free(st->syms);
            free(st->gm.e); free(st->gblob); free(st->grelocs); free(st);
            return NULL;
        }
        string sym = MLIR_GetAttributeString(sa);
        if (sym.size == 4 && memcmp(sym.str, "main", 4) == 0) st->saw_main = true;
        st->funcs[st->n_funcs] = op;
        st->syms[st->n_funcs]  = sym;
        st->n_funcs++;
    }

    if (out_gblob) *out_gblob = st->gblob;
    if (out_gblob_len) *out_gblob_len = st->gblob_len;
    return st;
}

size_t mlir_llvm_sel_num_funcs(LlvmSelState *st) {
    return st ? st->n_funcs : 0;
}

bool mlir_llvm_sel_saw_main(LlvmSelState *st) {
    return st ? st->saw_main : false;
}

MLIR_OpHandle mlir_llvm_sel_synth_start(MLIR_Context *ctx, LlvmSelState *st) {
    return synth_start(ctx, str_lit("main"), st->grelocs, st->n_grelocs,
                       &st->gm);
}

MLIR_OpHandle mlir_llvm_sel_func(MLIR_Context *ctx, LlvmSelState *st,
                                 size_t idx) {
    if (idx >= st->n_funcs) return MLIR_INVALID_HANDLE;
    return select_func(ctx, st->funcs[idx], st->syms[idx], &st->gm);
}

void mlir_llvm_sel_end(LlvmSelState *st) {
    if (!st) return;
    // Note: st->gblob ownership was handed to the caller via out_gblob; the
    // caller frees it. Everything else is owned here.
    free(st->gm.e);
    free(st->grelocs);
    free(st->funcs);
    free(st->syms);
    free(st);
}

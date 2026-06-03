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
// Nothing here is on the path of the existing wasm/wmir backends; it is
// reached only via the opt-in `--macho-backend=llvm` driver flag.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "mlir_llvm_to_aarch64.h"
#include "mlir_op_names.h"

// ---------------------------------------------------------------------------
// Small attribute / op builders (mirrors of the wmir backend's helpers).
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

// ---- Floating-point op builders (mirror the wmir->aarch64 backend) --------
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

// ---------------------------------------------------------------------------
// Helpers over the input `llvm` dialect module.
// ---------------------------------------------------------------------------
static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}

static bool type_is_i64(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle ty = MLIR_GetValueType(v);
    string s = MLIR_GetTypeString(ctx, ty);
    return s.size == 3 && memcmp(s.str, "i64", 3) == 0;
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

// Reload an operand value from its frame slot into register `rd`.
static bool load_value(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                       MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_ldr_x(ctx, blk, rd, 31, (uint32_t)slot * 8u);
    return true;
}
// Store register `rd` into the frame slot for result value `v`.
static bool store_value(MLIR_Context *ctx, MLIR_BlockHandle blk, SlotMap *sm,
                        MLIR_ValueHandle v, uint8_t rd) {
    int32_t slot;
    if (!sm_get(sm, v, &slot)) return false;
    emit_str_x(ctx, blk, rd, 31, (uint32_t)slot * 8u);
    return true;
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
    bool              ok;
} LowerCtx;

#define LFAIL(...) do { fprintf(stderr, __VA_ARGS__); L->ok = false; return; } while (0)

static MLIR_BlockHandle new_block(LowerCtx *L) {
    MLIR_BlockHandle b = MLIR_CreateBlock(L->ctx);
    MLIR_AppendRegionBlock(L->ctx, L->out_region, b);
    return b;
}

// Slot-to-slot copy (block-arg / phi resolution and yield forwarding).
static void copy_slot(LowerCtx *L, MLIR_ValueHandle src, MLIR_ValueHandle dst) {
    if (!load_value(L->ctx, L->cur, L->sm, src, 9) ||
        !store_value(L->ctx, L->cur, L->sm, dst, 9))
        LFAIL("llvm->aarch64: undefined value in copy (%.*s)\n",
              (int)L->sym.size, L->sym.str);
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
    if (!load_value(L->ctx, L->cur, L->sm, MLIR_GetOpOperand(op, 0), 9))
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
    if (!load_value(L->ctx, L->cur, L->sm, MLIR_GetOpOperand(cterm, 0), 9))
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
    if (!load_value(L->ctx, L->cur, L->sm, MLIR_GetOpOperand(op, 0), 9))
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
        if (fw) {
            // Float constant: materialise the IEEE bit pattern into a GP slot.
            double d = MLIR_GetAttributeFloat(va);
            uint64_t bits = 0;
            if (fw == 32) { float f = (float)d; uint32_t b32;
                            memcpy(&b32, &f, 4); bits = b32; }
            else          { memcpy(&bits, &d, 8); }
            emit_load_imm(ctx, blk, 9, bits, /*sf=*/true);
        } else {
            emit_load_imm(ctx, blk, 9, (uint64_t)MLIR_GetAttributeInteger(va),
                          type_is_i64(ctx, res));
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (simple_binop_optype(on, &bt)) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_i64(ctx, res);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_3reg(ctx, blk, bt, 9, 9, 10, sf);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.srem") || name_eq(on, "llvm.urem")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_i64(ctx, res);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        MLIR_OpType dt = name_eq(on, "llvm.srem") ? OP_TYPE_AARCH64_SDIV
                                                  : OP_TYPE_AARCH64_UDIV;
        emit_3reg(ctx, blk, dt, 11, 9, 10, sf);
        emit_msub(ctx, blk, 9, 11, 10, 9, sf);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.icmp")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: llvm.icmp without predicate\n");
        int cond = icmp_pred_to_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0)
            LFAIL("llvm->aarch64: unsupported icmp predicate %lld\n",
                  (long long)MLIR_GetAttributeInteger(pa));
        bool sf = type_is_i64(ctx, MLIR_GetOpOperand(op, 0));
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_cmp_reg(ctx, blk, 9, 10, sf);
        emit_cset(ctx, blk, 9, (uint8_t)cond, false);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.select")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sf = type_is_i64(ctx, res);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9)  ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 2), 11))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_cmp_imm(ctx, blk, 9, 0, false);
        emit_csel(ctx, blk, 9, 10, 11, /*NE*/1, sf);
        store_value(ctx, blk, sm, res, 9);

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
            if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 16))
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
            for (size_t k = 0; k < no; k++)
                if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, arg_base + k), (uint8_t)k))
                    LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                          (int)L->sym.size, L->sym.str);
            if (is_indirect) emit_blr(ctx, blk, 16);
            else             emit_bl(ctx, blk, nm);
        } else {
            // AAPCS64: first n_reg args in x0..x{n_reg-1}, the rest on the stack
            // at the call sp. Reserve a 16-aligned arg area below sp; address
            // slots via a stable copy of the original sp in x12 (the sub shifts
            // sp-relative slot offsets).
            size_t n_stack = no - n_reg;
            uint32_t area = (uint32_t)((n_stack * 8u + 15u) & ~15u);
            emit_sub_imm(ctx, blk, 31, 31, (uint16_t)area, true);
            emit_add_imm(ctx, blk, 12, 31, (uint16_t)area, true); // x12 = orig sp
            for (size_t k = 0; k < n_reg; k++) {
                int32_t slot;
                if (!sm_get(sm, MLIR_GetOpOperand(op, arg_base + k), &slot))
                    LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                          (int)L->sym.size, L->sym.str);
                emit_ldr_x(ctx, blk, (uint8_t)k, 12, (uint32_t)slot * 8u);
            }
            for (size_t k = n_reg; k < no; k++) {
                int32_t slot;
                if (!sm_get(sm, MLIR_GetOpOperand(op, arg_base + k), &slot))
                    LFAIL("llvm->aarch64: undefined call arg in '%.*s'\n",
                          (int)L->sym.size, L->sym.str);
                emit_ldr_x(ctx, blk, 9, 12, (uint32_t)slot * 8u);
                emit_str_x(ctx, blk, 9, 31, (uint32_t)(k - n_reg) * 8u);
            }
            if (is_indirect) emit_blr(ctx, blk, 16);
            else             emit_bl(ctx, blk, nm);
            emit_add_imm(ctx, blk, 31, 31, (uint16_t)area, true); // restore sp
        }
        if (MLIR_GetOpNumResults(op) == 1)
            store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 0);

    } else if (name_eq(on, "llvm.intr.vastart")) {
        // Darwin arm64: all variadic args are passed on the stack and va_list is
        // just a pointer to the first one. Incoming stack args of this function
        // sit at [x29, #16 + max(0, n_fixed-8)*8] (after the saved {fp,lr} pair
        // and any fixed stack args). Store that address into the va_list buffer.
        uint32_t base_off = 16u +
            (L->n_fixed > 8 ? (uint32_t)(L->n_fixed - 8) * 8u : 0u);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 10))
            LFAIL("llvm->aarch64: undefined va_list in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_add_imm(ctx, blk, 9, 29, (uint16_t)base_off, true);
        emit_str_x(ctx, blk, 9, 10, 0);

    } else if (name_eq(on, "llvm.intr.vaend")) {
        // No teardown needed for the stack-based va_list.

    } else if (name_eq(on, "llvm.mlir.undef") ||
               name_eq(on, "llvm.mlir.zero")) {
        // mlir.zero materialises a typed zero (null pointer / zeroed
        // aggregate scalar); mlir.undef is don't-care — 0 is fine for both.
        if (MLIR_GetOpNumResults(op) == 1) {
            emit_load_imm(ctx, blk, 9, 0, false);
            store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 9);
        }

    } else if (name_eq(on, "llvm.alloca")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int32_t off;
        if (!sm_get(L->am, res, &off))
            LFAIL("llvm->aarch64: alloca without a frame offset\n");
        uint32_t addr = L->slot_bytes + (uint32_t)off;
        if (addr <= 4095) {
            emit_add_imm(ctx, blk, 9, 31, (uint16_t)addr, true);
        } else {
            // Offset exceeds the 12-bit ADD immediate: materialise it in a
            // scratch register and add it to a copy of sp.
            emit_add_imm(ctx, blk, 9, 31, 0, true);             // x9 = sp
            emit_load_imm(ctx, blk, 10, (uint64_t)addr, true);  // x10 = addr
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

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
        if (!load_value(ctx, blk, sm, base, 9))   // running address in x9
            LFAIL("llvm->aarch64: undefined gep base in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
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
                if (foff) { emit_load_imm(ctx, blk, 10, (uint64_t)foff, true);
                            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, 9, 9, 10, true); }
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
                if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, op_idx++), 10))
                    LFAIL("llvm->aarch64: undefined gep index in '%.*s'\n",
                          (int)L->sym.size, L->sym.str);
                emit_load_imm(ctx, blk, 11, (uint64_t)stride, true);
                emit_3reg(ctx, blk, OP_TYPE_AARCH64_MUL, 10, 10, 11, true);
                emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, 9, 9, 10, true);
            } else if ((cst = cidx[i]) != 0) {
                emit_load_imm(ctx, blk, 10, (uint64_t)(cst * (int64_t)stride), true);
                emit_3reg(ctx, blk, OP_TYPE_AARCH64_ADD_REG, 9, 9, 10, true);
            }
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.store")) {
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 1);
        unsigned sz = a64_type_size(ctx, MLIR_GetValueType(val));
        if (sz != 1 && sz != 4 && sz != 8)
            LFAIL("llvm->aarch64: unsupported store size %u\n", sz);
        if (!load_value(ctx, blk, sm, val, 9) ||
            !load_value(ctx, blk, sm, ptr, 10))
            LFAIL("llvm->aarch64: undefined store operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_mem_store(ctx, blk, 9, 10, sz);

    } else if (name_eq(on, "llvm.load")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        unsigned sz = a64_type_size(ctx, MLIR_GetValueType(res));
        if (sz != 1 && sz != 4 && sz != 8)
            LFAIL("llvm->aarch64: unsupported load size %u\n", sz);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 10))
            LFAIL("llvm->aarch64: undefined load operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        emit_mem_load(ctx, blk, 9, 10, sz);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.ptrtoint") || name_eq(on, "llvm.inttoptr")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        int w = int_type_bits(ctx, res);   // 0 for ptr (64-bit, no mask)
        if (w > 0 && w < 64) {
            emit_load_imm(ctx, blk, 10, (1ull << w) - 1, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_AND_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "arith.index_cast") ||
               name_eq(on, "arith.index_castui")) {
        // index <-> integer: a plain copy in our 64-bit slot model.
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 9);

    } else if (name_eq(on, "llvm.trunc") || name_eq(on, "llvm.zext")) {
        // Both keep the low N bits (our values are stored zero-extended); a
        // mask to the destination (trunc) / source (zext) width suffices.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = name_eq(on, "llvm.trunc") ? int_type_bits(ctx, res)
                                          : int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        if (w > 0 && w < 64) {
            emit_load_imm(ctx, blk, 10, (w >= 64) ? ~0ull : ((1ull << w) - 1), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_AND_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.sext")) {
        // Sign-extend from the source width via a left-then-arithmetic-right
        // shift pair in 64-bit registers.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int w = int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: undefined operand in '%.*s'\n",
                  (int)L->sym.size, L->sym.str);
        if (w > 0 && w < 64) {
            emit_load_imm(ctx, blk, 10, (uint64_t)(64 - w), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_LSL_REG, 9, 9, 10, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ASR_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.mlir.addressof")) {
        // Materialise the address of a module-level global: ADRP+ADD against
        // the native data section, offset by the global's slot.
        MLIR_AttributeHandle ga = MLIR_GetOpAttributeByName(op, "global_name");
        if (ga == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: addressof missing global_name\n");
        string gnm = MLIR_GetAttributeAsString(ctx, ga);
        if (gnm.size > 0 && gnm.str[0] == '@') { gnm.str++; gnm.size--; }
        uint32_t off = 0;
        if (L->gm && gmap_get(L->gm, gnm, &off)) {
            string tgt = str_from_cstr_view("linmem_template");
            emit_adrp_data(ctx, blk, 9, tgt, off);
            emit_add_data_lo(ctx, blk, 9, 9, tgt, off);
        } else {
            // Not a data global — materialise the address of a function
            // symbol (ADRP+ADD resolved against the function's text address
            // by the Mach-O encoder).
            emit_adrp_data(ctx, blk, 9, gnm, 0);
            emit_add_data_lo(ctx, blk, 9, 9, gnm, 0);
        }
        store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 9);

    } else if (name_eq(on, "llvm.fadd") || name_eq(on, "llvm.fsub") ||
               name_eq(on, "llvm.fmul") || name_eq(on, "llvm.fdiv")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, res);
        const char *kind = name_eq(on, "llvm.fadd") ? "fadd"
                         : name_eq(on, "llvm.fsub") ? "fsub"
                         : name_eq(on, "llvm.fmul") ? "fmul" : "fdiv";
        if (fw == 0) LFAIL("llvm->aarch64: %.*s non-float result\n",
                           (int)on.size, on.str);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: %.*s operand not materialised\n",
                  (int)on.size, on.str);
        emit_fmov_gp_v(ctx, blk, /*to_v=*/true, true, 0, 9);
        emit_fmov_gp_v(ctx, blk, /*to_v=*/true, true, 1, 10);
        emit_fp_binop(ctx, blk, kind, fw, 0, 0, 1);
        emit_fmov_gp_v(ctx, blk, /*to_v=*/false, true, 9, 0);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.fneg") || name_eq(on, "llvm.intr.sqrt")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int fw = fp_width(ctx, res);
        const char *kind = name_eq(on, "llvm.fneg") ? "fneg" : "fsqrt";
        if (fw == 0) LFAIL("llvm->aarch64: %.*s non-float result\n",
                           (int)on.size, on.str);
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: %.*s operand not materialised\n",
                  (int)on.size, on.str);
        emit_fmov_gp_v(ctx, blk, true, true, 0, 9);
        emit_fp_unop(ctx, blk, kind, fw, 0, 0);
        emit_fmov_gp_v(ctx, blk, false, true, 9, 0);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.fcmp")) {
        MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "predicate");
        if (pa == MLIR_INVALID_HANDLE)
            LFAIL("llvm->aarch64: fcmp missing predicate\n");
        int cond = fcmp_pred_to_cond(MLIR_GetAttributeInteger(pa));
        if (cond < 0)
            LFAIL("llvm->aarch64: unsupported fcmp predicate\n");
        int fw = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        if (fw == 0) LFAIL("llvm->aarch64: fcmp non-float operand\n");
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9) ||
            !load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 1), 10))
            LFAIL("llvm->aarch64: fcmp operand not materialised\n");
        emit_fmov_gp_v(ctx, blk, true, true, 0, 9);
        emit_fmov_gp_v(ctx, blk, true, true, 1, 10);
        emit_fcmp(ctx, blk, fw, 0, 1);
        emit_cset(ctx, blk, 9, (uint8_t)cond, false);
        store_value(ctx, blk, sm, MLIR_GetOpResult(op, 0), 9);

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
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: fptosi/fptoui operand not materialised\n");
        emit_fmov_gp_v(ctx, blk, true, true, 0, 9);
        emit_fp_cvt(ctx, blk, "f2i", src_w, dst_w, sign, /*rd GP*/9, /*rn V*/0);
        if (rw > 0 && rw < 32) {
            emit_load_imm(ctx, blk, 10, (1ull << rw) - 1, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_AND_REG, 9, 9, 10, true);
        }
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.sitofp") || name_eq(on, "llvm.uitofp")) {
        // int -> FP. Slots hold zero-extended ints: uitofp can read the full
        // 64-bit slot directly; sitofp must first sign-extend the source width.
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        bool sign = name_eq(on, "llvm.sitofp");
        int sw = int_type_bits(ctx, MLIR_GetOpOperand(op, 0));
        int dst_w = fp_width(ctx, res);
        if (dst_w == 0)
            LFAIL("llvm->aarch64: sitofp/uitofp non-float result\n");
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: sitofp/uitofp operand not materialised\n");
        if (sign && sw > 0 && sw < 64) {
            emit_load_imm(ctx, blk, 10, (uint64_t)(64 - sw), true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_LSL_REG, 9, 9, 10, true);
            emit_3reg(ctx, blk, OP_TYPE_AARCH64_ASR_REG, 9, 9, 10, true);
        }
        emit_fp_cvt(ctx, blk, "i2f", /*src_w=*/64, dst_w, sign, /*rd V*/0, /*rn GP*/9);
        emit_fmov_gp_v(ctx, blk, false, true, 9, 0);
        store_value(ctx, blk, sm, res, 9);

    } else if (name_eq(on, "llvm.fpext") || name_eq(on, "llvm.fptrunc")) {
        MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
        int src_w = fp_width(ctx, MLIR_GetOpOperand(op, 0));
        int dst_w = fp_width(ctx, res);
        if (src_w == 0 || dst_w == 0)
            LFAIL("llvm->aarch64: fpext/fptrunc non-float type\n");
        if (!load_value(ctx, blk, sm, MLIR_GetOpOperand(op, 0), 9))
            LFAIL("llvm->aarch64: fpext/fptrunc operand not materialised\n");
        emit_fmov_gp_v(ctx, blk, true, true, 0, 9);
        emit_fp_cvt(ctx, blk, "f2f", src_w, dst_w, false, 0, 0);
        emit_fmov_gp_v(ctx, blk, false, true, 9, 0);
        store_value(ctx, blk, sm, res, 9);

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
static void assign_slots_block(SlotMap *sm, MLIR_BlockHandle block,
                               int32_t *nslots) {
    size_t na = MLIR_GetBlockNumArgs(block);
    for (size_t i = 0; i < na; i++)
        sm_put(sm, MLIR_GetBlockArg(block, i), (*nslots)++);
    size_t no = MLIR_GetBlockNumOps(block);
    for (size_t i = 0; i < no; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(block, i);
        size_t nr = MLIR_GetOpNumResults(op);
        for (size_t r = 0; r < nr; r++)
            sm_put(sm, MLIR_GetOpResult(op, r), (*nslots)++);
        size_t ng = MLIR_GetOpNumRegions(op);
        for (size_t g = 0; g < ng; g++) {
            MLIR_RegionHandle rg = MLIR_GetOpRegion(op, g);
            size_t nbk = MLIR_GetRegionNumBlocks(rg);
            for (size_t b = 0; b < nbk; b++)
                assign_slots_block(sm, MLIR_GetRegionBlock(rg, b), nslots);
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
    int32_t nslots = 0;
    for (size_t b = 0; b < n_blocks; b++)
        assign_slots_block(&sm, MLIR_GetRegionBlock(src_region, b), &nslots);
    if (nslots > 4095)
        A64_FAIL("llvm->aarch64: function '%.*s' needs %d slots "
                 "(frame too large for the trivial allocator)\n",
                 (int)sym.size, sym.str, nslots);
    uint32_t slot_bytes = (uint32_t)nslots * 8u;

    SlotMap am = {0};
    am.arena = MLIR_GetArenaAllocator(ctx);
    uint32_t alloca_bytes = 0;
    for (size_t b = 0; b < n_blocks; b++)
        if (!collect_allocas(ctx, &am, MLIR_GetRegionBlock(src_region, b),
                             &alloca_bytes))
            A64_FAIL("llvm->aarch64: function '%.*s' has an unsupported "
                     "alloca\n", (int)sym.size, sym.str);
    uint32_t frame_size = (slot_bytes + alloca_bytes + 15u) & ~15u;

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

    // Prologue + incoming-param spill in the entry block.
    emit_prologue(ctx, out_blks[0], frame_size);
    for (size_t i = 0; i < nargs && i < 8; i++)
        store_value(ctx, out_blks[0], &sm, MLIR_GetBlockArg(entry_src, i),
                    (uint8_t)i);
    for (size_t i = 8; i < nargs; i++) {
        emit_ldr_x(ctx, out_blks[0], 9, 29, (uint32_t)(16u + (i - 8) * 8u));
        store_value(ctx, out_blks[0], &sm, MLIR_GetBlockArg(entry_src, i), 9);
    }

    LowerCtx L = { ctx, &sm, out_reg, out_blks[0], sym, frame_size,
                   &am, slot_bytes, gm, nargs, true };

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
        for (size_t i = 0; i + 1 < no; i++) {
            lower_op(&L, MLIR_GetBlockOp(sb, i));
            if (!L.ok) { free(src_blks); free(out_blks); return MLIR_INVALID_HANDLE; }
        }
        MLIR_OpHandle term = MLIR_GetBlockOp(sb, no - 1);
        string tn = MLIR_GetOpName(term);
        if (name_eq(tn, "llvm.return")) {
            size_t nr = MLIR_GetOpNumOperands(term);
            if (nr == 1) {
                if (!load_value(ctx, L.cur, &sm, MLIR_GetOpOperand(term, 0), 0)) {
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
            emit_epilogue(ctx, L.cur, frame_size);
            emit_ret(ctx, L.cur);
        } else if (name_eq(tn, "cf.br")) {
            MLIR_BlockHandle d = map_block(src_blks, out_blks, n_blocks,
                                           MLIR_GetOpSuccessor(term, 0));
            emit_b(ctx, L.cur, d);
        } else if (name_eq(tn, "cf.cond_br")) {
            if (!load_value(ctx, L.cur, &sm, MLIR_GetOpOperand(term, 0), 9)) {
                fprintf(stderr, "llvm->aarch64: undefined cond_br condition "
                        "in '%.*s'\n", (int)sym.size, sym.str);
                free(src_blks); free(out_blks);
                return MLIR_INVALID_HANDLE;
            }
            MLIR_BlockHandle tb = map_block(src_blks, out_blks, n_blocks,
                                            MLIR_GetOpSuccessor(term, 0));
            MLIR_BlockHandle fb = map_block(src_blks, out_blks, n_blocks,
                                            MLIR_GetOpSuccessor(term, 1));
            emit_cbnz(ctx, L.cur, 9, false, tb);
            emit_b(ctx, L.cur, fb);
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
    assign_slots_block(&sm, src_blk, &nslots);
    if (nslots > 4095) {
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

    emit_prologue(ctx, out_blk, frame_size);
    for (size_t i = 0; i < nargs && i < 8; i++)
        store_value(ctx, out_blk, &sm, MLIR_GetBlockArg(src_blk, i), (uint8_t)i);
    // Args 8.. arrive on the caller's stack. After the prologue x29 (fp) points
    // at the saved {fp,lr} pair, so the first stack arg is at [x29, #16].
    for (size_t i = 8; i < nargs; i++) {
        emit_ldr_x(ctx, out_blk, 9, 29, (uint32_t)(16u + (i - 8) * 8u));
        store_value(ctx, out_blk, &sm, MLIR_GetBlockArg(src_blk, i), 9);
    }

    LowerCtx L = { ctx, &sm, out_reg, out_blk, sym, frame_size,
                   &am, slot_bytes, gm, nargs, true };
    MLIR_OpHandle term = lower_block_ops(&L, src_blk);
    if (!L.ok) return MLIR_INVALID_HANDLE;

    // Function terminator: llvm.return.
    if (!name_eq(MLIR_GetOpName(term), "llvm.return"))
        A64_FAIL("llvm->aarch64: function '%.*s' does not end in llvm.return\n",
                 (int)sym.size, sym.str);
    size_t no = MLIR_GetOpNumOperands(term);
    if (no == 1) {
        if (!load_value(ctx, L.cur, &sm, MLIR_GetOpOperand(term, 0), 0))
            A64_FAIL("llvm->aarch64: undefined return value in '%.*s'\n",
                     (int)sym.size, sym.str);
    } else if (no > 1) {
        A64_FAIL("llvm->aarch64: multi-value return\n");
    }
    emit_epilogue(ctx, L.cur, frame_size);
    emit_ret(ctx, L.cur);

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
// (pages fault in on first touch), mirroring the wmir backend's crt0.
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

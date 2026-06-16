// aarch64 instruction assembler: aarch64-dialect ops -> machine-code bytes +
// relocation records (the MachineFunc object model). Split out of
// mlir_aarch64_to_macho.c so the architecture assembler is separated from the
// Mach-O binary-format container (which now consumes MachineFunc via
// mlir_aarch64_asm.h). See mlir_aarch64_asm.h.

#include "mlir_aarch64_asm.h"

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
// AArch64 instruction encoders. Each helper returns the 32-bit little-
// endian instruction word for the given operands. These are the only
// ops the first-light slice supports; the table grows op-by-op.
// =============================================================================
static uint32_t arm64_movz(uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    // 5283/d283 0000 base. sf=1 sets bit 31.
    uint32_t base = sf ? 0xd2800000u : 0x52800000u;
    return base | ((uint32_t)(hw & 0x3) << 21) | ((uint32_t)imm16 << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_movk(uint8_t rd, uint16_t imm16, uint8_t hw, bool sf) {
    uint32_t base = sf ? 0xf2800000u : 0x72800000u;
    return base | ((uint32_t)(hw & 0x3) << 21) | ((uint32_t)imm16 << 5)
                | (uint32_t)(rd & 0x1f);
}
// `mov Xd, Xn` (alias for ORR Xd, XZR, Xn): aa0003e0 | Rn<<16 | Rd.
static uint32_t arm64_mov_x(uint8_t rd, uint8_t rn) {
    return 0xaa0003e0u | ((uint32_t)(rn & 0x1f) << 16) | (uint32_t)(rd & 0x1f);
}
// `bl <pc-rel>` (the imm26 carries `byte_offset / 4`, sign-extended).
uint32_t arm64_bl(int32_t imm26) {
    return 0x94000000u | ((uint32_t)imm26 & 0x03ffffffu);
}
static uint32_t arm64_svc(uint16_t imm16) {
    return 0xd4000001u | ((uint32_t)imm16 << 5);
}
// `blr Xn` — indirect branch-and-link via register.
static uint32_t arm64_blr(uint8_t rn) {
    return 0xd63f0000u | ((uint32_t)(rn & 0x1f) << 5);
}
static uint32_t arm64_ret(void) { return 0xd65f03c0u; }

// ---- compare + cset/csel + branches ------------------------------
// CMP Wn, Wm  ==  SUBS Wzr, Wn, Wm (W form base 0x6B00001F; X 0xEB00001F)
static uint32_t arm64_cmp_reg(uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0xeb00001fu : 0x6b00001fu;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5);
}
// CMP Wn, #imm12  ==  SUBS Wzr, Wn, #imm12 (W base 0x7100001F; X 0xF100001F)
static uint32_t arm64_cmp_imm(uint8_t rn, uint16_t imm12, bool sf) {
    uint32_t base = sf ? 0xf100001fu : 0x7100001fu;
    return base | (((uint32_t)imm12 & 0xfffu) << 10)
                | ((uint32_t)(rn & 0x1f) << 5);
}
// CSET Wd, COND == CSINC Wd, WZR, WZR, invert(COND).
// CSINC encoding: sf 0 0 1101 0100 Rm 4-bit-cond 0 1 Rn Rd
//   W form base: 0x1A9F07E0 (Rm=WZR=31, Rn=WZR=31; cond field at bits[15:12]).
//   X form base: 0x9A9F07E0.
// The condition we pass here is the ORIGINAL (uninverted) condition;
// CSET inverts it internally to feed CSINC.
static uint32_t arm64_cset(uint8_t rd, uint8_t cond, bool sf) {
    uint32_t base = sf ? 0x9a9f07e0u : 0x1a9f07e0u;
    uint8_t  invc = (uint8_t)(cond ^ 1);
    return base | (((uint32_t)invc & 0xfu) << 12) | (uint32_t)(rd & 0x1f);
}
// CSEL Wd, Wn, Wm, COND: sf 0 0 1101 0100 Rm 4-bit-cond 0 0 Rn Rd
//   W base: 0x1A800000.  X base: 0x9A800000.
static uint32_t arm64_csel(uint8_t rd, uint8_t rn, uint8_t rm,
                           uint8_t cond, bool sf) {
    uint32_t base = sf ? 0x9a800000u : 0x1a800000u;
    return base | ((uint32_t)(rm & 0x1f) << 16)
                | (((uint32_t)cond & 0xfu) << 12)
                | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// B  imm26 ; PC-relative (imm26 holds byte_offset / 4, sign-extended).
static uint32_t arm64_b(int32_t imm26) {
    return 0x14000000u | ((uint32_t)imm26 & 0x03ffffffu);
}
// B.cond imm19 ; PC-relative (imm19 holds byte_offset / 4, sign-extended).
static uint32_t arm64_b_cond(int32_t imm19, uint8_t cond) {
    return 0x54000000u | (((uint32_t)imm19 & 0x7ffffu) << 5)
                       | ((uint32_t)cond & 0xfu);
}
// CBZ/CBNZ Wn, imm19 (W base 0x34000000/0x35000000; X 0xB4.../0xB5...).
static uint32_t arm64_cbz(uint8_t rt, int32_t imm19, bool sf) {
    uint32_t base = sf ? 0xb4000000u : 0x34000000u;
    return base | (((uint32_t)imm19 & 0x7ffffu) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_cbnz(uint8_t rt, int32_t imm19, bool sf) {
    uint32_t base = sf ? 0xb5000000u : 0x35000000u;
    return base | (((uint32_t)imm19 & 0x7ffffu) << 5) | (uint32_t)(rt & 0x1f);
}
// BRK #imm16: 0xD4200000 | (imm16 << 5).
static uint32_t arm64_brk(uint16_t imm16) {
    return 0xd4200000u | ((uint32_t)imm16 << 5);
}

// ---- frame ops ----------------------------------------------------
static uint32_t arm64_stp_fp_lr_pre(void)   { return 0xa9bf7bfdu; } // stp x29,x30,[sp,#-16]!
static uint32_t arm64_ldp_fp_lr_post(void)  { return 0xa8c17bfdu; } // ldp x29,x30,[sp],#16
static uint32_t arm64_mov_fp_sp(void)       { return 0x910003fdu; } // add x29,sp,#0
// shift=0: imm12 in bits 21:10 unshifted. shift=1: same imm12 but LSL #12 (sh bit 22).
static uint32_t arm64_add_sp_imm_sh(uint16_t imm12, bool lsl12) {
    return 0x910003ffu | (lsl12 ? 0x00400000u : 0u)
                       | (((uint32_t)imm12 & 0xfffu) << 10);
}
static uint32_t arm64_sub_sp_imm_sh(uint16_t imm12, bool lsl12) {
    return 0xd10003ffu | (lsl12 ? 0x00400000u : 0u)
                       | (((uint32_t)imm12 & 0xfffu) << 10);
}

// ---- arithmetic ---------------------------------------------------
// add/sub imm (X or W). When rn or rd is 31 it means SP, not XZR.
uint32_t arm64_add_imm(uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
    uint32_t base = sf ? 0x91000000u : 0x11000000u;
    return base | (((uint32_t)imm12 & 0xfffu) << 10)
                | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_sub_imm(uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
    uint32_t base = sf ? 0xd1000000u : 0x51000000u;
    return base | (((uint32_t)imm12 & 0xfffu) << 10)
                | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// add/sub Xd, Xn, Xm (shifted-register, LSL #0). With sf=false we get
// the W form. NB: for our load lowering we use sf=true on values
// produced by W-form loads, which zero-extend into the full X register
// — so a plain X-form add is equivalent to "add Xd, Xn, Wm, UXTW".
static uint32_t arm64_add_reg(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x8b000000u : 0x0b000000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_sub_reg(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0xcb000000u : 0x4b000000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}

// ---- memory access ------------------------------------------------
// LDR Wt, [Xn, #pimm]   pimm is unsigned byte offset, divisible by 4.
static uint32_t arm64_ldr_w_uoff(uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    return 0xb9400000u | (((uint32_t)(off_bytes >> 2) & 0xfffu) << 10)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_str_w_uoff(uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    return 0xb9000000u | (((uint32_t)(off_bytes >> 2) & 0xfffu) << 10)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
// LDR Xt, [Xn, #pimm] / STR Xt, [Xn, #pimm]  (8-byte aligned).
static uint32_t arm64_ldr_x_uoff(uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    return 0xf9400000u | (((uint32_t)(off_bytes >> 3) & 0xfffu) << 10)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_str_x_uoff(uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    return 0xf9000000u | (((uint32_t)(off_bytes >> 3) & 0xfffu) << 10)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
// STRB Wt, [Xn, #pimm]  (1-byte store, unscaled imm12).
static uint32_t arm64_strb_imm(uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    return 0x39000000u | (((uint32_t)off_bytes & 0xfffu) << 10)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
// LDRB Wt, [Xn, #pimm] (1-byte zero-extend load, unscaled imm12).
static uint32_t arm64_ldrb_imm(uint8_t rt, uint8_t rn, uint16_t off_bytes) {
    return 0x39400000u | (((uint32_t)off_bytes & 0xfffu) << 10)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
// ---- register-offset loads/stores: <op> Rt, [Xn, Wm, UXTW #0] -------
// Rm carries a wasm i32 linear-memory index. It must be treated as a
// 32-bit value zero-extended to 64 bits (option=UXTW(010), S=0), NOT as
// a full 64-bit X register: nothing guarantees the high half of the
// producing register is clear (e.g. mem2reg can route an index through a
// register still holding stale upper bits), and using LSL(011) would add
// that garbage to the base x28, faulting far outside linear memory.
static uint32_t arm64_ldr_w_reg(uint8_t rt, uint8_t rn, uint8_t rm) {
    return 0xb8604800u | ((uint32_t)(rm & 0x1f) << 16)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_str_w_reg(uint8_t rt, uint8_t rn, uint8_t rm) {
    return 0xb8204800u | ((uint32_t)(rm & 0x1f) << 16)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_ldr_x_reg(uint8_t rt, uint8_t rn, uint8_t rm) {
    return 0xf8604800u | ((uint32_t)(rm & 0x1f) << 16)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_str_x_reg(uint8_t rt, uint8_t rn, uint8_t rm) {
    return 0xf8204800u | ((uint32_t)(rm & 0x1f) << 16)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_ldrb_reg(uint8_t rt, uint8_t rn, uint8_t rm) {
    return 0x38604800u | ((uint32_t)(rm & 0x1f) << 16)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
static uint32_t arm64_strb_reg(uint8_t rt, uint8_t rn, uint8_t rm) {
    return 0x38204800u | ((uint32_t)(rm & 0x1f) << 16)
                       | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rt & 0x1f);
}
// MUL Wd, Wn, Wm == MADD Wd, Wn, Wm, WZR
//   W base: 0x1B007C00, X base: 0x9B007C00
static uint32_t arm64_mul(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x9b007c00u : 0x1b007c00u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
// SDIV / UDIV.
//   sdiv W base: 0x1AC00C00, X base: 0x9AC00C00
//   udiv W base: 0x1AC00800, X base: 0x9AC00800
static uint32_t arm64_sdiv(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x9ac00c00u : 0x1ac00c00u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_udiv(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x9ac00800u : 0x1ac00800u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
// UBFM / SBFM (bitfield moves). LSL/LSR are UBFM aliases, ASR is SBFM.
//   UBFM W base: 0x53000000, X base (N=1): 0xD3400000
//   SBFM W base: 0x13000000, X base (N=1): 0x93400000
static uint32_t arm64_ubfm(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms, bool sf) {
    uint32_t base = sf ? 0xD3400000u : 0x53000000u;
    return base | ((uint32_t)(immr & 0x3f) << 16) | ((uint32_t)(imms & 0x3f) << 10)
                | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_sbfm(uint8_t rd, uint8_t rn, uint8_t immr, uint8_t imms, bool sf) {
    uint32_t base = sf ? 0x93400000u : 0x13000000u;
    return base | ((uint32_t)(immr & 0x3f) << 16) | ((uint32_t)(imms & 0x3f) << 10)
                | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// MSUB Wd, Wn, Wm, Wa  (Wd = Wa - Wn*Wm). Used to compute remainder
// after a div: rem = a - (a/b)*b = msub(a/b, b, a).
//   W base: 0x1B008000, X base: 0x9B008000  (Ra at bits[14:10])
static uint32_t arm64_msub(uint8_t rd, uint8_t rn, uint8_t rm, uint8_t ra, bool sf) {
    uint32_t base = sf ? 0x9b008000u : 0x1b008000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(ra & 0x1f) << 10)
                | ((uint32_t)(rn & 0x1f) << 5)  | (uint32_t)(rd & 0x1f);
}
// AND/ORR/EOR (shifted register, LSL #0).
//   AND W base: 0x0A000000, X base: 0x8A000000
//   ORR W base: 0x2A000000, X base: 0xAA000000
//   EOR W base: 0x4A000000, X base: 0xCA000000
static uint32_t arm64_and_reg(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x8a000000u : 0x0a000000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
// AND (immediate) with a low-bits mask #((1<<w)-1), 1 <= w <= (sf?63:31).
// (1<<w)-1 is `w` contiguous low ones, always a valid aarch64 logical bitmask
// immediate: element size = register size (N=sf), no rotation (immr=0), run
// length = w (imms = w-1). 64-bit base 0x92400000 (N=1), 32-bit base 0x12000000.
static uint32_t arm64_and_imm_lowbits(uint8_t rd, uint8_t rn, uint8_t w, bool sf) {
    uint32_t base = sf ? 0x92400000u : 0x12000000u;   // sf=1 sets N via 0x400000
    uint32_t imms = (uint32_t)(w - 1) & 0x3f;          // immr = 0
    return base | (imms << 10) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_orr_reg(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0xaa000000u : 0x2a000000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_eor_reg(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0xca000000u : 0x4a000000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
// LSLV / LSRV / ASRV (variable shift by register, lsl/lsr/asr Rm).
//   LSLV W base: 0x1AC02000, X base: 0x9AC02000
//   LSRV W base: 0x1AC02400, X base: 0x9AC02400
//   ASRV W base: 0x1AC02800, X base: 0x9AC02800
static uint32_t arm64_lslv(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x9ac02000u : 0x1ac02000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_lsrv(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x9ac02400u : 0x1ac02400u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
static uint32_t arm64_asrv(uint8_t rd, uint8_t rn, uint8_t rm, bool sf) {
    uint32_t base = sf ? 0x9ac02800u : 0x1ac02800u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
// SXTW Xd, Wn  ==  SBFM Xd, Xn, #0, #31 :: 0x93407C00 | Rn<<5 | Rd
static uint32_t arm64_sxtw(uint8_t rd, uint8_t rn) {
    return 0x93407c00u | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// SXTB Wd, Wn  == SBFM Wd, Wn, #0, #7   :: 0x13001C00 | Rn<<5 | Rd
// SXTB Xd, Wn  == SBFM Xd, Xn, #0, #7   :: 0x93401C00 | Rn<<5 | Rd
static uint32_t arm64_sxtb(uint8_t rd, uint8_t rn, bool sf) {
    uint32_t base = sf ? 0x93401c00u : 0x13001c00u;
    return base | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// SXTH Wd, Wn  == SBFM Wd, Wn, #0, #15  :: 0x13003C00 | Rn<<5 | Rd
// SXTH Xd, Wn  == SBFM Xd, Xn, #0, #15  :: 0x93403C00 | Rn<<5 | Rd
static uint32_t arm64_sxth(uint8_t rd, uint8_t rn, bool sf) {
    uint32_t base = sf ? 0x93403c00u : 0x13003c00u;
    return base | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// UXTW: zero-extend W to X. == ORR Wd, WZR, Wn (writes to W which zeros
// the upper half of X). 0x2A0003E0 | Rn<<16 | Rd  (sf=0).
static uint32_t arm64_uxtw(uint8_t rd, uint8_t rn) {
    return 0x2a0003e0u | ((uint32_t)(rn & 0x1f) << 16) | (uint32_t)(rd & 0x1f);
}

// ---- Floating-point encodings ------------------------------------
// FMOV Sd, Wn (GP -> V, 32-bit): 0x1E270000 | (Rn<<5) | Rd.
// FMOV Dd, Xn (GP -> V, 64-bit): 0x9E670000 | ...
static uint32_t arm64_fmov_gp_to_v(uint8_t rd, uint8_t rn, bool sf) {
    uint32_t base = sf ? 0x9e670000u : 0x1e270000u;
    return base | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// FMOV Wd, Sn / Xd, Dn: 0x1E260000 / 0x9E660000.
static uint32_t arm64_fmov_v_to_gp(uint8_t rd, uint8_t rn, bool sf) {
    uint32_t base = sf ? 0x9e660000u : 0x1e260000u;
    return base | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// FADD/FSUB/FMUL/FDIV Sd/Dd, Sn/Dn, Sm/Dm.
//   ftype at bit 22 (0=single, 1=double).
//   opcode at bits[13:12]: fmul=00, fdiv=01, fadd=10, fsub=11.
// Base for FMUL Sd: 0x1E200800.
static uint32_t arm64_fp_binop(uint8_t kind, int fwidth,
                               uint8_t rd, uint8_t rn, uint8_t rm) {
    // kind: 0=fmul, 1=fdiv, 2=fadd, 3=fsub.
    uint32_t base = 0x1e200800u;
    if (fwidth == 64) base |= 0x00400000u;
    base |= ((uint32_t)kind & 0x3u) << 12;
    return base | ((uint32_t)(rm & 0x1f) << 16)
                | ((uint32_t)(rn & 0x1f) << 5)
                | (uint32_t)(rd & 0x1f);
}
// FABS/FNEG/FSQRT Sd/Dd, Sn/Dn.
//   opcode at bits[20:15] (only bits 15..17 used here):
//     fabs=000001, fneg=000010, fsqrt=000011.
// Base for FABS Sd: 0x1E20C000.
static uint32_t arm64_fp_unop(uint8_t kind, int fwidth,
                              uint8_t rd, uint8_t rn) {
    // kind: 1=fabs, 2=fneg, 3=fsqrt.
    uint32_t base = 0x1e204000u;
    if (fwidth == 64) base |= 0x00400000u;
    base |= ((uint32_t)kind & 0x3fu) << 15;
    return base | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// FCMP Sn/Dn, Sm/Dm (no result reg; writes NZCV).
//   Base FCMP Sn, Sm: 0x1E202000 | Rm<<16 | Rn<<5.
static uint32_t arm64_fcmp(int fwidth, uint8_t rn, uint8_t rm) {
    uint32_t base = 0x1e202000u;
    if (fwidth == 64) base |= 0x00400000u;
    return base | ((uint32_t)(rm & 0x1f) << 16) | ((uint32_t)(rn & 0x1f) << 5);
}
// FCVT Dd, Sn (single -> double): 0x1E22C000 | Rn<<5 | Rd.
// FCVT Sd, Dn (double -> single): 0x1E624000 | Rn<<5 | Rd.
static uint32_t arm64_fcvt_f2f(int src_w, int dst_w, uint8_t rd, uint8_t rn) {
    uint32_t insn;
    if (src_w == 32 && dst_w == 64) insn = 0x1e22c000u;
    else                            insn = 0x1e624000u;
    return insn | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// FCVTZS / FCVTZU (FP -> integer; round toward zero).
//   FCVTZS Wd, Sn: 0x1E380000
//   FCVTZS Wd, Dn: 0x1E780000
//   FCVTZS Xd, Sn: 0x9E380000
//   FCVTZS Xd, Dn: 0x9E780000
//   FCVTZU: add 0x10000 (opcode bit 16).
static uint32_t arm64_fp_to_int(bool sign, int src_w, int dst_w,
                                uint8_t rd, uint8_t rn) {
    uint32_t insn = 0x1e380000u;
    if (src_w == 64) insn |= 0x00400000u;
    if (dst_w == 64) insn |= 0x80000000u;
    if (!sign)       insn |= 0x00010000u;
    return insn | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}
// SCVTF / UCVTF (integer -> FP).
//   SCVTF Sd, Wn: 0x1E220000
//   SCVTF Dd, Wn: 0x1E620000
//   SCVTF Sd, Xn: 0x9E220000
//   SCVTF Dd, Xn: 0x9E620000
//   UCVTF: add 0x10000 (opcode bit 16).
static uint32_t arm64_int_to_fp(bool sign, int src_w, int dst_w,
                                uint8_t rd, uint8_t rn) {
    uint32_t insn = 0x1e220000u;
    if (dst_w == 64) insn |= 0x00400000u;
    if (src_w == 64) insn |= 0x80000000u;
    if (!sign)       insn |= 0x00010000u;
    return insn | ((uint32_t)(rn & 0x1f) << 5) | (uint32_t)(rd & 0x1f);
}

// ---- ADRP / ADD ---------------------------------------------------
// ADRP Xd, #rel_pages*4096 (rel_pages is a signed 21-bit value).
uint32_t arm64_adrp(uint8_t rd, int64_t rel_pages) {
    uint32_t immlo = (uint32_t)(rel_pages & 0x3);
    uint32_t immhi = (uint32_t)((rel_pages >> 2) & 0x7ffffu);
    return 0x90000000u | (immlo << 29) | (immhi << 5) | (uint32_t)(rd & 0x1f);
}

static void emit_word(Buf *b, uint32_t w) { buf_le32(b, w); }


static void ef_add_reloc(MachineFunc *e, string callee, uint32_t off) {
    if (e->n_relocs == e->c_relocs) {
        e->c_relocs = e->c_relocs ? e->c_relocs * 2 : 4;
        e->relocs = (MachineCallReloc *)realloc(e->relocs, e->c_relocs * sizeof(MachineCallReloc));
    }
    e->relocs[e->n_relocs].callee = callee;
    e->relocs[e->n_relocs].fn_off = off;
    e->n_relocs++;
}
static void ef_add_dr(MachineFunc *e, string kind, bool is_add_lo,
                      uint8_t rd, uint8_t rn, uint32_t off, uint32_t addend) {
    if (e->n_dr == e->c_dr) {
        e->c_dr = e->c_dr ? e->c_dr * 2 : 4;
        e->dr = (MachineDataReloc *)realloc(e->dr, e->c_dr * sizeof(MachineDataReloc));
    }
    e->dr[e->n_dr].kind      = kind;
    e->dr[e->n_dr].is_add_lo = is_add_lo;
    e->dr[e->n_dr].rd        = rd;
    e->dr[e->n_dr].rn        = rn;
    e->dr[e->n_dr].fn_off    = off;
    e->dr[e->n_dr].addend    = addend;
    e->n_dr++;
}
static void ef_add_br(MachineFunc *e, int kind, MLIR_BlockHandle target,
                      uint32_t off, uint8_t cond_or_rt, bool sf) {
    if (e->n_br == e->c_br) {
        e->c_br = e->c_br ? e->c_br * 2 : 4;
        e->br = (MachineBranchReloc *)realloc(e->br, e->c_br * sizeof(MachineBranchReloc));
    }
    e->br[e->n_br].kind       = kind;
    e->br[e->n_br].target     = target;
    e->br[e->n_br].fn_off     = off;
    e->br[e->n_br].cond_or_rt = cond_or_rt;
    e->br[e->n_br].sf         = sf;
    e->n_br++;
}
static void ef_add_bp(MachineFunc *e, MLIR_BlockHandle blk, uint32_t off) {
    if (e->n_bp == e->c_bp) {
        e->c_bp = e->c_bp ? e->c_bp * 2 : 4;
        e->bp = (MachineBlockPos *)realloc(e->bp, e->c_bp * sizeof(MachineBlockPos));
    }
    e->bp[e->n_bp].blk    = blk;
    e->bp[e->n_bp].fn_off = off;
    e->n_bp++;
}

int64_t attr_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool attr_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
string attr_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}


bool emit_aarch64_func(MLIR_OpHandle fn, MachineFunc *out) {
    out->name     = attr_s(fn, "sym_name");
    out->exported = attr_b(fn, "exported");

    if (MLIR_GetOpNumRegions(fn) < 1) {
        fprintf(stderr, "aarch64->macho: aarch64.func has no region\n");
        return false;
    }
    MLIR_RegionHandle reg = MLIR_GetOpRegion(fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    // Fallthrough elision: a terminator `aarch64.b` whose target is the very
    // next block in layout is a no-op jump to PC+4. Dropping it lets control
    // fall through. The lifted `cf` CFG (and its cond_br trampoline blocks)
    // produces many such branches; eliding them removes ~1.5% of all
    // instructions, concentrated in hot loops. Skippable for A/B measurement.
    bool elide_fallthrough = (getenv("TINYC_NO_FALLTHROUGH") == NULL);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(reg, bi);
        MLIR_BlockHandle next_blk = (bi + 1 < nb)
            ? MLIR_GetRegionBlock(reg, bi + 1) : MLIR_INVALID_HANDLE;
        // Record the position of this block's first instruction.
        ef_add_bp(out, blk, (uint32_t)out->code.len);
        size_t n = MLIR_GetBlockNumOps(blk);
        for (size_t i = 0; i < n; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
            MLIR_OpType  t  = MLIR_GetOpType(op);
            switch (t) {
            case OP_TYPE_AARCH64_MOVZ: {
                uint8_t rd  = (uint8_t)attr_i(op, "rd");
                uint16_t im = (uint16_t)attr_i(op, "imm16");
                uint8_t hw  = (uint8_t)attr_i(op, "hw");
                bool   sf   = attr_b(op, "sf");
                emit_word(&out->code, arm64_movz(rd, im, hw, sf));
                break;
            }
            case OP_TYPE_AARCH64_MOVK: {
                uint8_t rd  = (uint8_t)attr_i(op, "rd");
                uint16_t im = (uint16_t)attr_i(op, "imm16");
                uint8_t hw  = (uint8_t)attr_i(op, "hw");
                bool   sf   = attr_b(op, "sf");
                emit_word(&out->code, arm64_movk(rd, im, hw, sf));
                break;
            }
            case OP_TYPE_AARCH64_MOV_X: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                emit_word(&out->code, arm64_mov_x(rd, rn));
                break;
            }
            case OP_TYPE_AARCH64_BL: {
                string callee = attr_s(op, "callee");
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_bl(0));
                ef_add_reloc(out, callee, off);
                break;
            }
            case OP_TYPE_AARCH64_BLR: {
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                emit_word(&out->code, arm64_blr(rn));
                break;
            }
            case OP_TYPE_AARCH64_SVC: {
                uint16_t im = (uint16_t)attr_i(op, "imm16");
                emit_word(&out->code, arm64_svc(im));
                break;
            }
            case OP_TYPE_AARCH64_RET:
                emit_word(&out->code, arm64_ret());
                break;
            case OP_TYPE_AARCH64_BRK: {
                uint16_t im = (uint16_t)attr_i(op, "imm16");
                emit_word(&out->code, arm64_brk(im));
                break;
            }
            case OP_TYPE_AARCH64_PROLOGUE: {
                uint32_t fs = (uint32_t)attr_i(op, "frame_size");
                emit_word(&out->code, arm64_stp_fp_lr_pre());
                emit_word(&out->code, arm64_mov_fp_sp());
                uint32_t lo = fs & 0xfffu;
                uint32_t hi = (fs >> 12) & 0xfffu;
                // Emit high-12-bit part first (subtract larger amount),
                // then low-12 part. Both can be 0; we only emit non-zero
                // chunks. Note: high is encoded as LSL #12.
                if (hi) emit_word(&out->code, arm64_sub_sp_imm_sh((uint16_t)hi, /*lsl12=*/true));
                if (lo) emit_word(&out->code, arm64_sub_sp_imm_sh((uint16_t)lo, /*lsl12=*/false));
                break;
            }
            case OP_TYPE_AARCH64_EPILOGUE: {
                uint32_t fs = (uint32_t)attr_i(op, "frame_size");
                uint32_t lo = fs & 0xfffu;
                uint32_t hi = (fs >> 12) & 0xfffu;
                if (lo) emit_word(&out->code, arm64_add_sp_imm_sh((uint16_t)lo, /*lsl12=*/false));
                if (hi) emit_word(&out->code, arm64_add_sp_imm_sh((uint16_t)hi, /*lsl12=*/true));
                emit_word(&out->code, arm64_ldp_fp_lr_post());
                break;
            }
            case OP_TYPE_AARCH64_ADD_IMM: {
                uint8_t  rd = (uint8_t)attr_i(op, "rd");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t im = (uint16_t)attr_i(op, "imm12");
                bool     sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_add_imm(rd, rn, im, sf));
                break;
            }
            case OP_TYPE_AARCH64_SUB_IMM: {
                uint8_t  rd = (uint8_t)attr_i(op, "rd");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t im = (uint16_t)attr_i(op, "imm12");
                bool     sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_sub_imm(rd, rn, im, sf));
                break;
            }
            case OP_TYPE_AARCH64_ADD_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                uint8_t lsl = (uint8_t)attr_i(op, "lsl");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code,
                          arm64_add_reg(rd, rn, rm, sf) | ((uint32_t)(lsl & 0x3f) << 10));
                break;
            }
            case OP_TYPE_AARCH64_SUB_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_sub_reg(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_LDR_W: {
                uint8_t  rt = (uint8_t)attr_i(op, "rt");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t of = (uint16_t)attr_i(op, "off_bytes");
                emit_word(&out->code, arm64_ldr_w_uoff(rt, rn, of));
                break;
            }
            case OP_TYPE_AARCH64_STR_W: {
                uint8_t  rt = (uint8_t)attr_i(op, "rt");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t of = (uint16_t)attr_i(op, "off_bytes");
                emit_word(&out->code, arm64_str_w_uoff(rt, rn, of));
                break;
            }
            case OP_TYPE_AARCH64_LDR_X: {
                uint8_t  rt = (uint8_t)attr_i(op, "rt");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t of = (uint16_t)attr_i(op, "off_bytes");
                emit_word(&out->code, arm64_ldr_x_uoff(rt, rn, of));
                break;
            }
            case OP_TYPE_AARCH64_STR_X: {
                uint8_t  rt = (uint8_t)attr_i(op, "rt");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t of = (uint16_t)attr_i(op, "off_bytes");
                emit_word(&out->code, arm64_str_x_uoff(rt, rn, of));
                break;
            }
            case OP_TYPE_AARCH64_STRB_IMM: {
                uint8_t  rt = (uint8_t)attr_i(op, "rt");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t of = (uint16_t)attr_i(op, "off_bytes");
                emit_word(&out->code, arm64_strb_imm(rt, rn, of));
                break;
            }
            case OP_TYPE_AARCH64_LDRB_IMM: {
                uint8_t  rt = (uint8_t)attr_i(op, "rt");
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t of = (uint16_t)attr_i(op, "off_bytes");
                emit_word(&out->code, arm64_ldrb_imm(rt, rn, of));
                break;
            }
            case OP_TYPE_AARCH64_LDR_W_REG: {
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_ldr_w_reg(rt, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_STR_W_REG: {
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_str_w_reg(rt, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_LDR_X_REG: {
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_ldr_x_reg(rt, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_STR_X_REG: {
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_str_x_reg(rt, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_LDRB_REG: {
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_ldrb_reg(rt, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_STRB_REG: {
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_strb_reg(rt, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_MUL: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_mul(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_SDIV: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_sdiv(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_UDIV: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_udiv(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_MSUB: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                uint8_t ra = (uint8_t)attr_i(op, "ra");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_msub(rd, rn, rm, ra, sf));
                break;
            }
            case OP_TYPE_AARCH64_AND_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_and_reg(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_AND_IMM: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t w  = (uint8_t)attr_i(op, "w");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_and_imm_lowbits(rd, rn, w, sf));
                break;
            }
            case OP_TYPE_AARCH64_ORR_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_orr_reg(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_EOR_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_eor_reg(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_LSL_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_lslv(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_LSR_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_lsrv(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_ASR_REG: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_asrv(rd, rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_LSL_IMM: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t sh = (uint8_t)attr_i(op, "shift");
                bool    sf = attr_b(op, "sf");
                uint8_t ds = sf ? 64 : 32;
                uint8_t immr = (uint8_t)((ds - sh) & (ds - 1));
                uint8_t imms = (uint8_t)(ds - 1 - sh);
                emit_word(&out->code, arm64_ubfm(rd, rn, immr, imms, sf));
                break;
            }
            case OP_TYPE_AARCH64_LSR_IMM: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t sh = (uint8_t)attr_i(op, "shift");
                bool    sf = attr_b(op, "sf");
                uint8_t ds = sf ? 64 : 32;
                emit_word(&out->code, arm64_ubfm(rd, rn, sh, (uint8_t)(ds - 1), sf));
                break;
            }
            case OP_TYPE_AARCH64_ASR_IMM: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t sh = (uint8_t)attr_i(op, "shift");
                bool    sf = attr_b(op, "sf");
                uint8_t ds = sf ? 64 : 32;
                emit_word(&out->code, arm64_sbfm(rd, rn, sh, (uint8_t)(ds - 1), sf));
                break;
            }
            case OP_TYPE_AARCH64_SXTW: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                emit_word(&out->code, arm64_sxtw(rd, rn));
                break;
            }
            case OP_TYPE_AARCH64_SXTB: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_sxtb(rd, rn, sf));
                break;
            }
            case OP_TYPE_AARCH64_SXTH: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_sxth(rd, rn, sf));
                break;
            }
            case OP_TYPE_AARCH64_UXTW: {
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                emit_word(&out->code, arm64_uxtw(rd, rn));
                break;
            }
            case OP_TYPE_AARCH64_ADRP_DATA: {
                uint8_t rd      = (uint8_t)attr_i(op, "rd");
                string  target  = attr_s(op, "target");
                uint32_t addend = (uint32_t)attr_i(op, "addend");
                uint32_t off    = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_adrp(rd, 0));
                ef_add_dr(out, target, /*is_add_lo=*/false, rd, /*rn=*/0, off, addend);
                break;
            }
            case OP_TYPE_AARCH64_ADD_DATA_LO: {
                uint8_t rd     = (uint8_t)attr_i(op, "rd");
                uint8_t rn     = (uint8_t)attr_i(op, "rn");
                string  target = attr_s(op, "target");
                uint32_t addend = (uint32_t)attr_i(op, "addend");
                uint32_t off   = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_add_imm(rd, rn, 0, /*sf=*/true));
                ef_add_dr(out, target, /*is_add_lo=*/true, rd, rn, off, addend);
                break;
            }
            case OP_TYPE_AARCH64_CMP_REG: {
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_cmp_reg(rn, rm, sf));
                break;
            }
            case OP_TYPE_AARCH64_CMP_IMM: {
                uint8_t  rn = (uint8_t)attr_i(op, "rn");
                uint16_t im = (uint16_t)attr_i(op, "imm12");
                bool     sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_cmp_imm(rn, im, sf));
                break;
            }
            case OP_TYPE_AARCH64_CSET: {
                uint8_t rd   = (uint8_t)attr_i(op, "rd");
                uint8_t cond = (uint8_t)attr_i(op, "cond");
                bool    sf   = attr_b(op, "sf");
                emit_word(&out->code, arm64_cset(rd, cond, sf));
                break;
            }
            case OP_TYPE_AARCH64_CSEL: {
                uint8_t rd   = (uint8_t)attr_i(op, "rd");
                uint8_t rn   = (uint8_t)attr_i(op, "rn");
                uint8_t rm   = (uint8_t)attr_i(op, "rm");
                uint8_t cond = (uint8_t)attr_i(op, "cond");
                bool    sf   = attr_b(op, "sf");
                emit_word(&out->code, arm64_csel(rd, rn, rm, cond, sf));
                break;
            }
            case OP_TYPE_AARCH64_B: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                // Elide a terminator branch that just falls through to the
                // next block in layout.
                if (elide_fallthrough && i + 1 == n && tgt == next_blk)
                    break;
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_b(0));
                ef_add_br(out, BR_B, tgt, off, 0, false);
                break;
            }
            // Conditional branches (B.cond / CBZ / CBNZ) only reach ±1 MiB
            // (a signed 19-bit word offset), which overflows in large
            // self-host functions and silently wraps to a wrong target. Emit
            // every conditional branch as a fixed inverted skip (imm19 = 2,
            // i.e. PC+8) over an unconditional B, whose imm26 reaches ±128 MiB
            // and is enough for any function we emit.
            case OP_TYPE_AARCH64_B_COND: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint8_t cond = (uint8_t)attr_i(op, "cond");
                emit_word(&out->code, arm64_b_cond(2, (uint8_t)(cond ^ 1u)));
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_b(0));
                ef_add_br(out, BR_B_COND, tgt, off, cond, false);
                break;
            }
            case OP_TYPE_AARCH64_CBZ: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_cbnz(rt, 2, sf));
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_b(0));
                ef_add_br(out, BR_CBZ, tgt, off, rt, sf);
                break;
            }
            case OP_TYPE_AARCH64_CBNZ: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_cbz(rt, 2, sf));
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_b(0));
                ef_add_br(out, BR_CBNZ, tgt, off, rt, sf);
                break;
            }
            case OP_TYPE_AARCH64_LABEL:
                // Pseudo: marks a position; emits no bytes. Block boundary
                // tracking already happened above.
                break;
            case OP_TYPE_AARCH64_FMOV_GP_V: {
                bool dir_to_v = attr_b(op, "dir_to_v");
                bool sf       = attr_b(op, "sf");
                uint8_t rd    = (uint8_t)attr_i(op, "rd");
                uint8_t rn    = (uint8_t)attr_i(op, "rn");
                emit_word(&out->code, dir_to_v
                    ? arm64_fmov_gp_to_v(rd, rn, sf)
                    : arm64_fmov_v_to_gp(rd, rn, sf));
                break;
            }
            case OP_TYPE_AARCH64_FP_BINOP: {
                string k = attr_s(op, "kind");
                int fwidth = (int)attr_i(op, "fwidth");
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                uint8_t kind = 0;
                if      (k.size == 4 && memcmp(k.str, "fmul", 4) == 0) kind = 0;
                else if (k.size == 4 && memcmp(k.str, "fdiv", 4) == 0) kind = 1;
                else if (k.size == 4 && memcmp(k.str, "fadd", 4) == 0) kind = 2;
                else if (k.size == 4 && memcmp(k.str, "fsub", 4) == 0) kind = 3;
                else {
                    fprintf(stderr,
                        "aarch64->macho: aarch64.fp_binop unknown kind '%.*s'\n",
                        (int)k.size, k.str);
                    return false;
                }
                emit_word(&out->code, arm64_fp_binop(kind, fwidth, rd, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_FP_UNOP: {
                string k = attr_s(op, "kind");
                int fwidth = (int)attr_i(op, "fwidth");
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t kind = 0;
                if      (k.size == 4 && memcmp(k.str, "fabs", 4) == 0)  kind = 1;
                else if (k.size == 4 && memcmp(k.str, "fneg", 4) == 0)  kind = 2;
                else if (k.size == 5 && memcmp(k.str, "fsqrt", 5) == 0) kind = 3;
                else {
                    fprintf(stderr,
                        "aarch64->macho: aarch64.fp_unop unknown kind '%.*s'\n",
                        (int)k.size, k.str);
                    return false;
                }
                emit_word(&out->code, arm64_fp_unop(kind, fwidth, rd, rn));
                break;
            }
            case OP_TYPE_AARCH64_FCMP: {
                int fwidth = (int)attr_i(op, "fwidth");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint8_t rm = (uint8_t)attr_i(op, "rm");
                emit_word(&out->code, arm64_fcmp(fwidth, rn, rm));
                break;
            }
            case OP_TYPE_AARCH64_FP_CVT: {
                string k = attr_s(op, "kind");
                int src_w = (int)attr_i(op, "src_w");
                int dst_w = (int)attr_i(op, "dst_w");
                bool sign = attr_b(op, "sign");
                uint8_t rd = (uint8_t)attr_i(op, "rd");
                uint8_t rn = (uint8_t)attr_i(op, "rn");
                uint32_t insn;
                if      (k.size == 3 && memcmp(k.str, "f2f", 3) == 0)
                    insn = arm64_fcvt_f2f(src_w, dst_w, rd, rn);
                else if (k.size == 3 && memcmp(k.str, "f2i", 3) == 0)
                    insn = arm64_fp_to_int(sign, src_w, dst_w, rd, rn);
                else if (k.size == 3 && memcmp(k.str, "i2f", 3) == 0)
                    insn = arm64_int_to_fp(sign, src_w, dst_w, rd, rn);
                else {
                    fprintf(stderr,
                        "aarch64->macho: aarch64.fp_cvt unknown kind '%.*s'\n",
                        (int)k.size, k.str);
                    return false;
                }
                emit_word(&out->code, insn);
                break;
            }
            default: {
                string nm = MLIR_GetOpName(op);
                fprintf(stderr,
                    "aarch64->macho: unsupported aarch64 op '%.*s' "
                    "(kind=%d)\n", (int)nm.size, nm.str, (int)t);
                return false;
            }
            }
        }
    }
    return true;
}

// Stable bottom-up mergesort of an index array by a uint64 key array.
static void a64_sort_idx_u64(uint32_t *idx, size_t n, const uint64_t *key) {
    if (n < 2) return;
    uint32_t *tmp = (uint32_t *)malloc(n * sizeof(uint32_t));
    for (size_t w = 1; w < n; w *= 2) {
        for (size_t lo = 0; lo < n; lo += 2 * w) {
            size_t mid = lo + w;     if (mid > n) mid = n;
            size_t hi  = lo + 2 * w; if (hi  > n) hi  = n;
            size_t a = lo, b = mid, o = lo;
            while (a < mid && b < hi)
                tmp[o++] = (key[idx[b]] < key[idx[a]]) ? idx[b++] : idx[a++];
            while (a < mid) tmp[o++] = idx[a++];
            while (b < hi)  tmp[o++] = idx[b++];
        }
        for (size_t i = 0; i < n; i++) idx[i] = tmp[i];
    }
    free(tmp);
}

// Patch a 32-bit little-endian instruction word into the code buffer.
static void a64_write_word(MachineFunc *e, uint32_t off, uint32_t insn) {
    e->code.data[off + 0] = (uint8_t)(insn      );
    e->code.data[off + 1] = (uint8_t)(insn >>  8);
    e->code.data[off + 2] = (uint8_t)(insn >> 16);
    e->code.data[off + 3] = (uint8_t)(insn >> 24);
}

// Count of sorted-ascending `a[]` entries strictly less than `key` (lower
// bound index). Used to map a pre-compaction byte offset to its position after
// some earlier words have been deleted: newoff(o) = o - 4*count(del < o).
static size_t a64_lower_count(const uint32_t *a, size_t n, uint32_t key) {
    size_t lo = 0, hi = n;
    while (lo < hi) { size_t m = (lo + hi) / 2; if (a[m] < key) lo = m + 1; else hi = m; }
    return lo;
}

// Resolve intra-function branch targets to PC-relative immediates. Done
// AFTER all blocks have been emitted (and their offsets recorded) but
// BEFORE the function is laid out within __text — branches are
// function-local, so we don't need text_off here.
//
// Conditional-branch form chosen by patch_branches (see the compaction logic).
enum { FORM_B = 0, FORM_DIRECT = 1, FORM_FALLBACK = 2 };

// Branch threading: a "pure trampoline" block — one whose entire body is a
// single unconditional `b TARGET` (exactly 4 bytes, no edge-copy movs) — adds
// a hop to every branch that lands on it. Such a block carries no block-arg
// forwarding (forwarding would require movs, making it >4 bytes), so any
// branch to it can be redirected straight to TARGET. Chains are followed (with
// a cycle cap) so a branch reaches its ultimate destination in one hop. This
// removes the per-iteration trampoline bounce that the structured-CFG lowering
// emits in hot loops. Trampoline blocks are left in place (now cold). Gated by
// TINYC_NO_THREAD for A/B.
bool patch_branches(MachineFunc *e) {
    size_t nbp = e->n_bp;
    uint32_t *byhandle = NULL;   // bp indices sorted by block handle
    int32_t  *fwd = NULL;        // bp index -> bp index it trampolines to (-1 none)
    if (nbp > 0) {
        byhandle = (uint32_t *)malloc(nbp * sizeof(uint32_t));
        uint64_t *hkey = (uint64_t *)malloc(nbp * sizeof(uint64_t));
        for (size_t k = 0; k < nbp; k++) {
            byhandle[k] = (uint32_t)k; hkey[k] = (uint64_t)e->bp[k].blk;
        }
        a64_sort_idx_u64(byhandle, nbp, hkey);
        free(hkey);

        if (getenv("TINYC_NO_THREAD") == NULL) {
            // Offset order, to compute block lengths and look a block up by its
            // start offset.
            uint32_t *byoff = (uint32_t *)malloc(nbp * sizeof(uint32_t));
            uint64_t *okey = (uint64_t *)malloc(nbp * sizeof(uint64_t));
            for (size_t k = 0; k < nbp; k++) {
                byoff[k] = (uint32_t)k; okey[k] = e->bp[k].fn_off;
            }
            a64_sort_idx_u64(byoff, nbp, okey);
            free(okey);
            uint32_t *blen = (uint32_t *)malloc(nbp * sizeof(uint32_t));
            for (size_t i = 0; i < nbp; i++) {
                uint32_t o   = e->bp[byoff[i]].fn_off;
                uint32_t end = (i + 1 < nbp) ? e->bp[byoff[i + 1]].fn_off
                                             : (uint32_t)e->code.len;
                blen[byoff[i]] = end - o;
            }
            fwd = (int32_t *)malloc(nbp * sizeof(int32_t));
            for (size_t k = 0; k < nbp; k++) fwd[k] = -1;
            for (size_t r = 0; r < e->n_br; r++) {
                if (e->br[r].kind != BR_B) continue;
                uint32_t roff = e->br[r].fn_off;
                // The block whose single `b` sits at roff (binary search byoff).
                // Several blocks can share roff when zero-length fallthrough-
                // elided blocks precede the real one; among an equal-offset run
                // only the actually-emitted block has nonzero length, so scan
                // the run for the len==4 entry.
                size_t lo = 0, hi = nbp; int blk_i = -1;
                while (lo < hi) {
                    size_t m = (lo + hi) / 2;
                    uint32_t mo = e->bp[byoff[m]].fn_off;
                    if (mo == roff) {
                        size_t s = m;
                        while (s > 0 && e->bp[byoff[s - 1]].fn_off == roff) s--;
                        for (; s < nbp && e->bp[byoff[s]].fn_off == roff; s++)
                            if (blen[byoff[s]] == 4) { blk_i = (int)byoff[s]; break; }
                        break;
                    }
                    if (mo < roff) lo = m + 1; else hi = m;
                }
                if (blk_i < 0) continue;
                // Forward target = bp index of this `b`'s target block.
                MLIR_BlockHandle t = e->br[r].target;
                size_t l2 = 0, h2 = nbp; int t_i = -1;
                while (l2 < h2) {
                    size_t m = (l2 + h2) / 2;
                    MLIR_BlockHandle mh = e->bp[byhandle[m]].blk;
                    if (mh == t) { t_i = (int)byhandle[m]; break; }
                    if (mh < t) l2 = m + 1; else h2 = m;
                }
                if (t_i >= 0) fwd[blk_i] = t_i;
            }
            free(byoff); free(blen);
        }
    }

    bool ok = true;

    // ---------------------------------------------------------------------
    // Resolve + compact. Conditional branches are emitted as a two-word
    // `b.!cond +8 ; b target` placeholder so the unconditional `b` gives
    // imm26 reach. When `target` fits the direct conditional imm19 reach the
    // first word becomes a single `b.cond target` and the second word is dead.
    // Rather than leave a NOP there (~3% of all instructions, one per in-range
    // conditional branch -- WMIR emits none), DELETE the dead word and compact
    // the function, remapping every later offset. KEY INVARIANT: deleting words
    // only ever shrinks |displacement|, so a branch that fits imm19 in the
    // padded layout still fits after compaction -- the form decided here with
    // padded offsets stays valid and no NOP is ever reintroduced. Gated by
    // TINYC_NO_COMPACT for A/B (then the dead word is kept as a NOP).
    //
    // Phase A: resolve each branch's threaded target offset and classify its
    // form using the *padded* offsets, writing nothing yet.
    size_t    nbr     = e->n_br;
    uint8_t  *form    = (uint8_t  *)malloc((nbr ? nbr : 1) * sizeof(uint8_t));
    uint32_t *tgt_old = (uint32_t *)malloc((nbr ? nbr : 1) * sizeof(uint32_t));
    bool      no_direct = (getenv("TINYC_NO_DIRECT_COND") != NULL);
    for (size_t i = 0; i < nbr; i++) {
        MachineBranchReloc *r = &e->br[i];
        int idx = -1;
        size_t lo = 0, hi = nbp;
        while (lo < hi) {
            size_t m = (lo + hi) / 2;
            MLIR_BlockHandle mh = e->bp[byhandle[m]].blk;
            if (mh == r->target) { idx = (int)byhandle[m]; break; }
            if (mh < r->target) lo = m + 1; else hi = m;
        }
        uint32_t to = (uint32_t)-1;
        if (idx >= 0) {
            if (fwd) {
                size_t steps = 0;
                while (fwd[idx] >= 0 && fwd[idx] != idx && steps < nbp) {
                    idx = fwd[idx]; steps++;
                }
            }
            to = e->bp[idx].fn_off;
        }
        if (to == (uint32_t)-1) {
            fprintf(stderr,
                "aarch64->macho: branch target block has no recorded offset\n");
            ok = false;
            break;
        }
        tgt_old[i] = to;
        if (r->kind == BR_B) {
            form[i] = FORM_B;
        } else {
            uint32_t cs = r->fn_off - 4;
            int32_t  imm_c = ((int32_t)to - (int32_t)cs) >> 2;
            bool fits19 = !no_direct && imm_c >= -(1 << 18) && imm_c <= (1 << 18) - 1;
            form[i] = fits19 ? FORM_DIRECT : FORM_FALLBACK;
        }
    }

    if (ok) {
        // Phase B: deletion set = dead second slots of direct conditionals.
        // br[] is appended in emission order, so br[i].fn_off is strictly
        // increasing and the filtered list is already sorted ascending.
        bool do_compact = (getenv("TINYC_NO_COMPACT") == NULL);
        uint32_t *del = (uint32_t *)malloc((nbr ? nbr : 1) * sizeof(uint32_t));
        size_t ndel = 0;
        if (do_compact) {
            for (size_t i = 0; i < nbr; i++)
                if (form[i] == FORM_DIRECT) {
                    if (ndel > 0 && e->br[i].fn_off <= del[ndel - 1]) {
                        do_compact = false; ndel = 0; break;  // not monotonic: bail safely
                    }
                    del[ndel++] = e->br[i].fn_off;
                }
        }
        bool compacted = do_compact && ndel > 0;

        // Phase C: compact the code buffer in place (drop every deleted word).
        if (compacted) {
            size_t w = 0, di = 0;
            for (uint32_t o = 0; o < e->code.len; o += 4) {
                if (di < ndel && del[di] == o) { di++; continue; }
                if (w != o) memcpy(e->code.data + w, e->code.data + o, 4);
                w += 4;
            }
            e->code.len = w;
            // Remap the only intra-function offsets consumed downstream.
            for (size_t i = 0; i < e->n_relocs; i++)
                e->relocs[i].fn_off -= 4u * (uint32_t)a64_lower_count(del, ndel, e->relocs[i].fn_off);
            for (size_t i = 0; i < e->n_dr; i++)
                e->dr[i].fn_off -= 4u * (uint32_t)a64_lower_count(del, ndel, e->dr[i].fn_off);
        }

        // Phase D: write each branch at its (possibly remapped) offset against
        // remapped target offsets. With compaction off, the dead direct slot is
        // kept as an explicit NOP exactly as before.
        #define NEWOFF(o) ((uint32_t)((o) - (compacted ? 4u * (uint32_t)a64_lower_count(del, ndel, (o)) : 0u)))
        for (size_t i = 0; i < nbr && ok; i++) {
            MachineBranchReloc *r = &e->br[i];
            uint32_t tnew = NEWOFF(tgt_old[i]);
            if (form[i] == FORM_B) {
                uint32_t fo = NEWOFF(r->fn_off);
                a64_write_word(e, fo, arm64_b((int32_t)(tnew - fo) >> 2));
            } else if (form[i] == FORM_DIRECT) {
                uint32_t cs = NEWOFF(r->fn_off - 4);
                int32_t  imm_c = (int32_t)(tnew - cs) >> 2;
                uint32_t c = 0;
                switch (r->kind) {
                    case BR_B_COND: c = arm64_b_cond(imm_c, r->cond_or_rt); break;
                    case BR_CBZ:    c = arm64_cbz(r->cond_or_rt, imm_c, r->sf); break;
                    case BR_CBNZ:   c = arm64_cbnz(r->cond_or_rt, imm_c, r->sf); break;
                    default: break;
                }
                a64_write_word(e, cs, c);
                if (!compacted) a64_write_word(e, NEWOFF(r->fn_off), 0xD503201Fu); // NOP
            } else {  // FORM_FALLBACK: inverted skip (+8) over an unconditional b
                uint32_t cs = NEWOFF(r->fn_off - 4);
                uint32_t fo = NEWOFF(r->fn_off);
                uint32_t skip = 0;
                switch (r->kind) {
                    case BR_B_COND:
                        skip = arm64_b_cond(2, (uint8_t)(r->cond_or_rt ^ 1u)); break;
                    case BR_CBZ:
                        skip = arm64_cbnz(r->cond_or_rt, 2, r->sf); break;
                    case BR_CBNZ:
                        skip = arm64_cbz(r->cond_or_rt, 2, r->sf); break;
                    default: break;
                }
                a64_write_word(e, cs, skip);
                a64_write_word(e, fo, arm64_b((int32_t)(tnew - fo) >> 2));
            }
        }
        #undef NEWOFF
        free(del);
    }

    free(form); free(tgt_old);
    free(byhandle); free(fwd);
    return ok;
}

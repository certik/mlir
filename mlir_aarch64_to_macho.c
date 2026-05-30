// aarch64 (MLIR dialect) -> Mach-O ARM64 binary translator.
//
// First-light scope: lower a module whose top-level ops are
// aarch64.func (with a flat sequence of aarch64.movz / movk / mov_x /
// bl / svc / ret instruction ops inside) into a runnable Mach-O ARM64
// binary that does not require any libSystem stubs or GOT. The
// special function name `_start` is the program entry; it is placed
// first in the `__text` section so LC_MAIN.entryoff lands on it.
//
// The envelope mirrors the macho_exit reference layout used by
// mlir_wasm_to_macho.c (the n_stubs == 0 / no __DATA shape), so we
// know the load-command sequence and ad-hoc signature format are
// accepted by macOS / AMFI. The only difference from the existing
// backend is that the contents of __text are produced by walking the
// aarch64 dialect rather than the WASM bytecode.
//
// Everything outside the first-light op set returns `false` with a
// diagnostic; coverage grows as new aarch64.* ops are added.

#include "mlir_aarch64_to_macho.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/string.h>

#include "mlir_api.h"
#include "mlir_op_names.h"

// =============================================================================
// Growable byte buffer + endian helpers + SHA-256. These mirror the
// helpers in mlir_wasm_to_macho.c (which keeps them file-local), so we
// duplicate them here rather than tying the two backends together.
// =============================================================================
typedef struct { uint8_t *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 1024;
    while (b->len + add > nc) nc *= 2;
    b->data = (uint8_t *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_u8(Buf *b, uint8_t v) { buf_grow(b, 1); b->data[b->len++] = v; }
static void buf_append(Buf *b, const void *p, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void buf_pad_to(Buf *b, size_t target) {
    if (target <= b->len) return;
    buf_grow(b, target - b->len);
    memset(b->data + b->len, 0, target - b->len);
    b->len = target;
}
static void buf_le16(Buf *b, uint16_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
}
static void buf_le32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)(v & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
}
static void buf_le64(Buf *b, uint64_t v) {
    for (int i = 0; i < 8; i++) buf_u8(b, (uint8_t)((v >> (8 * i)) & 0xff));
}
static void buf_be32(Buf *b, uint32_t v) {
    buf_u8(b, (uint8_t)((v >> 24) & 0xff));
    buf_u8(b, (uint8_t)((v >> 16) & 0xff));
    buf_u8(b, (uint8_t)((v >> 8) & 0xff));
    buf_u8(b, (uint8_t)(v & 0xff));
}
static void buf_uleb(Buf *b, uint64_t v) {
    do {
        uint8_t byte = (uint8_t)(v & 0x7f);
        v >>= 7;
        if (v) byte |= 0x80;
        buf_u8(b, byte);
    } while (v);
}
static void buf_cstr(Buf *b, const char *s) {
    while (*s) buf_u8(b, (uint8_t)*s++);
    buf_u8(b, 0);
}
static void buf_patch_le32(Buf *b, size_t pos, uint32_t v) {
    b->data[pos + 0] = (uint8_t)(v & 0xff);
    b->data[pos + 1] = (uint8_t)((v >> 8) & 0xff);
    b->data[pos + 2] = (uint8_t)((v >> 16) & 0xff);
    b->data[pos + 3] = (uint8_t)((v >> 24) & 0xff);
}
static void buf_patch_le64(Buf *b, size_t pos, uint64_t v) {
    for (int k = 0; k < 8; k++)
        b->data[pos + (size_t)k] = (uint8_t)(v >> (8 * k));
}

#define SHA256_DIGEST_LEN 32
typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint32_t datalen;
    uint8_t  data[64];
} Sha256;
static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
static inline uint32_t rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
static void sha256_xform(Sha256 *s, const uint8_t *d) {
    uint32_t a,b,c,e,f,g,h,t1,t2,m[64];
    uint32_t dd;
    for (int i = 0, j = 0; i < 16; i++, j += 4) {
        m[i] = ((uint32_t)(d[j]   & 0xFFu) << 24) | ((uint32_t)(d[j+1] & 0xFFu) << 16) |
               ((uint32_t)(d[j+2] & 0xFFu) << 8)  |  (uint32_t)(d[j+3] & 0xFFu);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(m[i-15], 7) ^ rotr32(m[i-15], 18) ^ (m[i-15] >> 3);
        uint32_t s1 = rotr32(m[i-2], 17) ^ rotr32(m[i-2], 19)  ^ (m[i-2] >> 10);
        m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    a = s->state[0]; b = s->state[1]; c = s->state[2]; dd = s->state[3];
    e = s->state[4]; f = s->state[5]; g = s->state[6]; h = s->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        t1 = h + S1 + ch + SHA256_K[i] + m[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + mj;
        h = g; g = f; f = e; e = dd + t1;
        dd = c; c = b; b = a; a = t1 + t2;
    }
    s->state[0] += a; s->state[1] += b; s->state[2] += c; s->state[3] += dd;
    s->state[4] += e; s->state[5] += f; s->state[6] += g; s->state[7] += h;
}
static void sha256(const uint8_t *data, size_t n, uint8_t out[32]) {
    Sha256 s;
    s.state[0] = 0x6a09e667; s.state[1] = 0xbb67ae85;
    s.state[2] = 0x3c6ef372; s.state[3] = 0xa54ff53a;
    s.state[4] = 0x510e527f; s.state[5] = 0x9b05688c;
    s.state[6] = 0x1f83d9ab; s.state[7] = 0x5be0cd19;
    s.bitlen = 0; s.datalen = 0;
    while (n) {
        size_t take = 64 - s.datalen;
        if (take > n) take = n;
        memcpy(s.data + s.datalen, data, take);
        s.datalen += (uint32_t)take; data += take; n -= take;
        if (s.datalen == 64) { sha256_xform(&s, s.data); s.bitlen += 512; s.datalen = 0; }
    }
    uint32_t i = s.datalen;
    if (s.datalen < 56) {
        s.data[i++] = 0x80;
        while (i < 56) s.data[i++] = 0;
    } else {
        s.data[i++] = 0x80;
        while (i < 64) s.data[i++] = 0;
        sha256_xform(&s, s.data);
        memset(s.data, 0, 56);
    }
    s.bitlen += (uint64_t)s.datalen * 8;
    for (int k = 7; k >= 0; k--) s.data[56 + (7 - k)] = (uint8_t)(s.bitlen >> (8 * k));
    sha256_xform(&s, s.data);
    for (int k = 0; k < 8; k++) {
        out[4*k + 0] = (uint8_t)(s.state[k] >> 24);
        out[4*k + 1] = (uint8_t)(s.state[k] >> 16);
        out[4*k + 2] = (uint8_t)(s.state[k] >> 8);
        out[4*k + 3] = (uint8_t)(s.state[k]);
    }
}

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
static uint32_t arm64_bl(int32_t imm26) {
    return 0x94000000u | ((uint32_t)imm26 & 0x03ffffffu);
}
static uint32_t arm64_svc(uint16_t imm16) {
    return 0xd4000001u | ((uint32_t)imm16 << 5);
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
static uint32_t arm64_add_sp_imm(uint16_t imm12) {
    return arm64_add_sp_imm_sh(imm12, /*lsl12=*/false);
}
static uint32_t arm64_sub_sp_imm(uint16_t imm12) {
    return arm64_sub_sp_imm_sh(imm12, /*lsl12=*/false);
}

// ---- arithmetic ---------------------------------------------------
// add/sub imm (X or W). When rn or rd is 31 it means SP, not XZR.
static uint32_t arm64_add_imm(uint8_t rd, uint8_t rn, uint16_t imm12, bool sf) {
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
// the W form. NB: for our wmir.load lowering we use sf=true on values
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
static uint32_t arm64_adrp(uint8_t rd, int64_t rel_pages) {
    uint32_t immlo = (uint32_t)(rel_pages & 0x3);
    uint32_t immhi = (uint32_t)((rel_pages >> 2) & 0x7ffffu);
    return 0x90000000u | (immlo << 29) | (immhi << 5) | (uint32_t)(rd & 0x1f);
}

static void emit_word(Buf *b, uint32_t w) { buf_le32(b, w); }

// =============================================================================
// Per-function emission. A pass over an aarch64.func produces a byte
// buffer + a list of `bl` call sites that need PC-relative patching
// once all function offsets are known.
// =============================================================================
typedef struct {
    string   callee;        // referenced symbol name
    uint32_t fn_off;        // byte offset within the function's code
} BlReloc;

// ADRP / ADD imm12 relocs. `kind` is one of "data_priv" / "globals" /
// "linmem"; the patcher computes the page-relative ADRP imm21 and the
// low-12 ADD imm12 once segment VM addresses are known.
typedef struct {
    string   kind;          // "data_priv" / "globals" / "linmem"
    bool     is_add_lo;     // false = ADRP, true = ADD imm12
    uint8_t  rd;
    uint8_t  rn;
    uint32_t fn_off;
} DataReloc;

// Branch reloc. Identifies a placeholder branch instruction emitted
// for an aarch64.b / b_cond / cbz / cbnz op so we can resolve it to a
// PC-relative imm once all blocks have known function offsets.
enum BranchKind { BR_B, BR_B_COND, BR_CBZ, BR_CBNZ };
typedef struct {
    int              kind;            // enum BranchKind
    MLIR_BlockHandle target;
    uint32_t         fn_off;          // offset of the branch insn within fn
    uint8_t          cond_or_rt;      // cond for B_COND, rt for CBZ/CBNZ
    bool             sf;              // for CBZ/CBNZ
} BranchReloc;

// Position of a block within the function's code buffer. Filled in as
// blocks are emitted, consumed by the branch patcher.
typedef struct {
    MLIR_BlockHandle blk;
    uint32_t         fn_off;
} BlockPos;

typedef struct {
    string       name;
    bool         exported;
    Buf          code;
    BlReloc     *relocs;
    size_t       n_relocs, c_relocs;
    DataReloc   *dr;
    size_t       n_dr, c_dr;
    BranchReloc *br;
    size_t       n_br, c_br;
    BlockPos    *bp;
    size_t       n_bp, c_bp;
    uint32_t     text_off;   // assigned after layout
} EmittedFunc;

static void ef_add_reloc(EmittedFunc *e, string callee, uint32_t off) {
    if (e->n_relocs == e->c_relocs) {
        e->c_relocs = e->c_relocs ? e->c_relocs * 2 : 4;
        e->relocs = (BlReloc *)realloc(e->relocs, e->c_relocs * sizeof(BlReloc));
    }
    e->relocs[e->n_relocs].callee = callee;
    e->relocs[e->n_relocs].fn_off = off;
    e->n_relocs++;
}
static void ef_add_dr(EmittedFunc *e, string kind, bool is_add_lo,
                      uint8_t rd, uint8_t rn, uint32_t off) {
    if (e->n_dr == e->c_dr) {
        e->c_dr = e->c_dr ? e->c_dr * 2 : 4;
        e->dr = (DataReloc *)realloc(e->dr, e->c_dr * sizeof(DataReloc));
    }
    e->dr[e->n_dr].kind      = kind;
    e->dr[e->n_dr].is_add_lo = is_add_lo;
    e->dr[e->n_dr].rd        = rd;
    e->dr[e->n_dr].rn        = rn;
    e->dr[e->n_dr].fn_off    = off;
    e->n_dr++;
}
static void ef_add_br(EmittedFunc *e, int kind, MLIR_BlockHandle target,
                      uint32_t off, uint8_t cond_or_rt, bool sf) {
    if (e->n_br == e->c_br) {
        e->c_br = e->c_br ? e->c_br * 2 : 4;
        e->br = (BranchReloc *)realloc(e->br, e->c_br * sizeof(BranchReloc));
    }
    e->br[e->n_br].kind       = kind;
    e->br[e->n_br].target     = target;
    e->br[e->n_br].fn_off     = off;
    e->br[e->n_br].cond_or_rt = cond_or_rt;
    e->br[e->n_br].sf         = sf;
    e->n_br++;
}
static void ef_add_bp(EmittedFunc *e, MLIR_BlockHandle blk, uint32_t off) {
    if (e->n_bp == e->c_bp) {
        e->c_bp = e->c_bp ? e->c_bp * 2 : 4;
        e->bp = (BlockPos *)realloc(e->bp, e->c_bp * sizeof(BlockPos));
    }
    e->bp[e->n_bp].blk    = blk;
    e->bp[e->n_bp].fn_off = off;
    e->n_bp++;
}

static int64_t attr_i(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeInteger(a) : 0;
}
static bool attr_b(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeBool(a) : false;
}
static string attr_s(MLIR_OpHandle op, const char *name) {
    MLIR_AttributeHandle a = MLIR_GetOpAttributeByName(op, name);
    return a ? MLIR_GetAttributeString(a) : (string){0};
}

// =============================================================================
// libSystem stub table.
//
// macOS only commits to libSystem.B.dylib as a stable ABI; raw BSD
// syscalls (`svc #0x80`) are private/unstable. Every external call the
// backend wants to make therefore goes through a `bl _<name>` to a
// PC-relative stub in __TEXT,__stubs that does `ADRP x16, GOT_page;
// LDR x16, [x16, GOT_lo12]; BR x16`. dyld fills the corresponding
// __DATA_CONST,__got slot at load time via chained fixups.
//
// LibSysSym enumerates every libSystem symbol the wmir backend may
// reference from its synth_* shims. The actual set used by any given
// program is discovered by walking BL relocs after function emission
// (see `n_libsys_stubs` further down). Enum order matters: it controls
// the dense stub index assigned to each symbol, which in turn
// determines the order of entries in __stubs / __got / chained-fixups
// imports / dysymtab indirect-syms.
//
// The lookup helpers are written as if/else cascades rather than as a
// `static const char *names[N] = {...};` table because tinyc (which
// must compile this file for selfhost) does not yet support string-
// literal initialisers in global arrays of `char *`, nor `sizeof(arr)`
// in a constant expression.
// =============================================================================
typedef enum {
    LS_EXIT  = 0,
    LS_WRITE,
    LS_READ,
    LS_OPEN,
    LS_CLOSE,
    LS_LSEEK,
    LS_ERRNO,
    LS_MMAP,
    LS_MEMCPY
} LibSysSym;
#define LS_COUNT 9

static const char *libsys_name(int sym) {
    if (sym == LS_EXIT)   return "_exit";
    if (sym == LS_WRITE)  return "_write";
    if (sym == LS_READ)   return "_read";
    if (sym == LS_OPEN)   return "_open";
    if (sym == LS_CLOSE)  return "_close";
    if (sym == LS_LSEEK)  return "_lseek";
    if (sym == LS_ERRNO)  return "___error";
    if (sym == LS_MMAP)   return "_mmap";
    if (sym == LS_MEMCPY) return "_memcpy";
    return "";
}

// Returns the LibSysSym value matching `callee`, or -1 if `callee` is
// not a known libSystem symbol.
static int libsys_lookup(string callee) {
    for (int i = 0; i < LS_COUNT; i++) {
        const char *nm = libsys_name(i);
        size_t kl = strlen(nm);
        if (callee.size == kl && memcmp(callee.str, nm, kl) == 0) {
            return i;
        }
    }
    return -1;
}

// Per-translation registry: which LibSysSym values are referenced, and
// in what dense order. Populated by a discovery pass before layout.
typedef struct {
    bool     used[LS_COUNT];
    uint32_t stub_index[LS_COUNT];   // dense index, valid iff used[i]
    uint32_t n_stubs;                 // total libSystem symbols used
} LibSysRegistry;

static bool emit_aarch64_func(MLIR_OpHandle fn, EmittedFunc *out) {
    out->name     = attr_s(fn, "sym_name");
    out->exported = attr_b(fn, "exported");

    if (MLIR_GetOpNumRegions(fn) < 1) {
        fprintf(stderr, "aarch64->macho: aarch64.func has no region\n");
        return false;
    }
    MLIR_RegionHandle reg = MLIR_GetOpRegion(fn, 0);
    size_t nb = MLIR_GetRegionNumBlocks(reg);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(reg, bi);
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
                bool    sf = attr_b(op, "sf");
                emit_word(&out->code, arm64_add_reg(rd, rn, rm, sf));
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
                uint32_t off    = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_adrp(rd, 0));
                ef_add_dr(out, target, /*is_add_lo=*/false, rd, /*rn=*/0, off);
                break;
            }
            case OP_TYPE_AARCH64_ADD_DATA_LO: {
                uint8_t rd     = (uint8_t)attr_i(op, "rd");
                uint8_t rn     = (uint8_t)attr_i(op, "rn");
                string  target = attr_s(op, "target");
                uint32_t off   = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_add_imm(rd, rn, 0, /*sf=*/true));
                ef_add_dr(out, target, /*is_add_lo=*/true, rd, rn, off);
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
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_b(0));
                ef_add_br(out, BR_B, tgt, off, 0, false);
                break;
            }
            case OP_TYPE_AARCH64_B_COND: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint8_t cond = (uint8_t)attr_i(op, "cond");
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_b_cond(0, cond));
                ef_add_br(out, BR_B_COND, tgt, off, cond, false);
                break;
            }
            case OP_TYPE_AARCH64_CBZ: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                bool    sf = attr_b(op, "sf");
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_cbz(rt, 0, sf));
                ef_add_br(out, BR_CBZ, tgt, off, rt, sf);
                break;
            }
            case OP_TYPE_AARCH64_CBNZ: {
                MLIR_BlockHandle tgt = MLIR_GetOpSuccessor(op, 0);
                uint8_t rt = (uint8_t)attr_i(op, "rt");
                bool    sf = attr_b(op, "sf");
                uint32_t off = (uint32_t)out->code.len;
                emit_word(&out->code, arm64_cbnz(rt, 0, sf));
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

// Resolve intra-function branch targets to PC-relative immediates. Done
// AFTER all blocks have been emitted (and their offsets recorded) but
// BEFORE the function is laid out within __text — branches are
// function-local, so we don't need text_off here.
static bool patch_branches(EmittedFunc *e) {
    for (size_t i = 0; i < e->n_br; i++) {
        BranchReloc *r = &e->br[i];
        uint32_t tgt_off = (uint32_t)-1;
        for (size_t k = 0; k < e->n_bp; k++) {
            if (e->bp[k].blk == r->target) { tgt_off = e->bp[k].fn_off; break; }
        }
        if (tgt_off == (uint32_t)-1) {
            fprintf(stderr,
                "aarch64->macho: branch target block has no recorded offset\n");
            return false;
        }
        int32_t rel = (int32_t)tgt_off - (int32_t)r->fn_off;
        int32_t imm = rel >> 2;
        uint32_t insn = 0;
        switch (r->kind) {
            case BR_B:      insn = arm64_b(imm); break;
            case BR_B_COND: insn = arm64_b_cond(imm, r->cond_or_rt); break;
            case BR_CBZ:    insn = arm64_cbz(r->cond_or_rt, imm, r->sf); break;
            case BR_CBNZ:   insn = arm64_cbnz(r->cond_or_rt, imm, r->sf); break;
        }
        e->code.data[r->fn_off + 0] = (uint8_t)(insn      );
        e->code.data[r->fn_off + 1] = (uint8_t)(insn >>  8);
        e->code.data[r->fn_off + 2] = (uint8_t)(insn >> 16);
        e->code.data[r->fn_off + 3] = (uint8_t)(insn >> 24);
    }
    return true;
}

// =============================================================================
// Mach-O envelope constants (mirror mlir_wasm_to_macho.c).
// =============================================================================
#define MH_MAGIC_64            0xfeedfacfu
#define CPU_ARCH_ABI64         0x01000000u
#define CPU_TYPE_ARM           12u
#define CPU_TYPE_ARM64         (CPU_TYPE_ARM | CPU_ARCH_ABI64)

#define LC_REQ_DYLD            0x80000000u
#define LC_SEGMENT_64          0x19u
#define LC_SYMTAB              0x02u
#define LC_DYSYMTAB            0x0bu
#define LC_LOAD_DYLIB          0x0cu
#define LC_LOAD_DYLINKER       0x0eu
#define LC_UUID                0x1bu
#define LC_CODE_SIGNATURE      0x1du
#define LC_FUNCTION_STARTS     0x26u
#define LC_DATA_IN_CODE        0x29u
#define LC_SOURCE_VERSION      0x2au
#define LC_BUILD_VERSION       0x32u
#define LC_MAIN                (0x28u | LC_REQ_DYLD)
#define LC_DYLD_EXPORTS_TRIE   (0x33u | LC_REQ_DYLD)
#define LC_DYLD_CHAINED_FIXUPS (0x34u | LC_REQ_DYLD)

#define MH_EXECUTE             2u
#define MH_FLAGS_EXEC          2097285u

#define VM_PROT_READ           1u
#define VM_PROT_WRITE          2u
#define VM_PROT_EXECUTE        4u

#define TEXT_VM_BASE           0x100000000ULL
#define TEXT_FILE_BASE         0u
#define VMSEG_SIZE             0x4000u   // 16 KiB minimum mach-o page

// =============================================================================
// Top-level translator.
// =============================================================================
bool mlir_aarch64_to_macho(MLIR_Context *ctx, MLIR_OpHandle module,
                           uint8_t **out_data, size_t *out_size) {
    (void)ctx;
    *out_data = NULL; *out_size = 0;
    if (!module) return false;
    if (MLIR_GetOpNumRegions(module) < 1) return false;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(MLIR_GetOpRegion(module, 0), 0);
    size_t n_top = MLIR_GetBlockNumOps(mb);

    // -----------------------------------------------------------------
    // Read module-level layout attributes set by mlir_wmir_to_aarch64.
    // -----------------------------------------------------------------
    uint32_t n_globals    = (uint32_t)attr_i(module, "n_globals");
    uint64_t global0_init = (uint64_t)attr_i(module, "global0_init");
    uint64_t linmem_size  = (uint64_t)attr_i(module, "linmem_size");

    uint32_t globals_bytes  = n_globals * 8u;
    uint32_t globals_padded = (globals_bytes + 15u) & ~15u;
    bool     has_data_seg   = (n_globals > 0) || (linmem_size > 0);
    uint32_t data_priv_size = has_data_seg ? 32u : 0u;

    // -----------------------------------------------------------------
    // Collect functions, find `_start`. `_start` must be placed first
    // in __text so LC_MAIN.entryoff equals text_section_off.
    // -----------------------------------------------------------------
    EmittedFunc *efs = (EmittedFunc *)calloc(n_top, sizeof(EmittedFunc));
    size_t n_funcs = 0;
    size_t start_idx = (size_t)-1;
    // -----------------------------------------------------------------
    // Walk top-level for data_init ops first: collect the contributions
    // to the linmem __DATA section. linmem_init_size is the high-water
    // mark across all (offset+size) records.
    // -----------------------------------------------------------------
    typedef struct { uint32_t offset; string bytes; } LinInit;
    LinInit  *inits = NULL;
    size_t    n_inits = 0;
    uint32_t  linmem_init_size = 0;
    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        if (MLIR_GetOpType(op) != OP_TYPE_AARCH64_DATA_INIT) continue;
        int64_t off = attr_i(op, "offset");
        string  bs  = attr_s(op, "init_data");
        inits = (LinInit *)realloc(inits, (n_inits + 1) * sizeof(LinInit));
        inits[n_inits].offset = (uint32_t)off;
        inits[n_inits].bytes  = bs;
        n_inits++;
        uint32_t end = (uint32_t)off + (uint32_t)bs.size;
        if (end > linmem_init_size) linmem_init_size = end;
    }
    if (linmem_init_size > 0) {
        // Round up to 16 for ARM64 alignment expectations.
        linmem_init_size = (linmem_init_size + 15u) & ~15u;
    }

    for (size_t i = 0; i < n_top; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        MLIR_OpType ot = MLIR_GetOpType(op);
        if (ot == OP_TYPE_AARCH64_DATA_INIT) continue;
        if (ot != OP_TYPE_AARCH64_FUNC) {
            string nm = MLIR_GetOpName(op);
            fprintf(stderr,
                "aarch64->macho: unexpected top-level op '%.*s'\n",
                (int)nm.size, nm.str);
            free(efs); free(inits); return false;
        }
        if (!emit_aarch64_func(op, &efs[n_funcs])) {
            for (size_t k = 0; k <= n_funcs; k++) {
                free(efs[k].code.data); free(efs[k].relocs); free(efs[k].dr);
                free(efs[k].br); free(efs[k].bp);
            }
            free(efs); free(inits); return false;
        }
        if (!patch_branches(&efs[n_funcs])) {
            for (size_t k = 0; k <= n_funcs; k++) {
                free(efs[k].code.data); free(efs[k].relocs); free(efs[k].dr);
                free(efs[k].br); free(efs[k].bp);
            }
            free(efs); free(inits); return false;
        }
        if (efs[n_funcs].name.size == 6
            && memcmp(efs[n_funcs].name.str, "_start", 6) == 0) {
            start_idx = n_funcs;
        }
        n_funcs++;
    }
    if (start_idx == (size_t)-1) {
        fprintf(stderr, "aarch64->macho: no `_start` function in module\n");
        for (size_t k = 0; k < n_funcs; k++) {
            free(efs[k].code.data); free(efs[k].relocs); free(efs[k].dr);
            free(efs[k].br); free(efs[k].bp);
        }
        free(efs); free(inits); return false;
    }

    // -----------------------------------------------------------------
    // Layout: _start first, then everything else in source order.
    // -----------------------------------------------------------------
    uint32_t cursor = 0;
    efs[start_idx].text_off = cursor; cursor += (uint32_t)efs[start_idx].code.len;
    for (size_t i = 0; i < n_funcs; i++) {
        if (i == start_idx) continue;
        efs[i].text_off = cursor;
        cursor += (uint32_t)efs[i].code.len;
    }
    uint32_t text_size = cursor;
    while (text_size % 4) text_size++;

    // -----------------------------------------------------------------
    // Discovery pass: walk every BL reloc and record which libSystem
    // names are referenced. Each unique referenced name gets a dense
    // stub index in first-seen order, which determines its position
    // in __stubs / __got / chained-fixups imports / dysymtab
    // indirect-syms.
    //
    // Done BEFORE layout because n_libsys_stubs feeds into
    // stubs_size, got_size, sizeofcmds (via section count and load-cmd
    // body sizes), and the strtab. Done AFTER text layout because
    // unreachable BL relocs in functions that ended up at non-zero
    // text_off are still real references that need stubs.
    // -----------------------------------------------------------------
    LibSysRegistry libsys = {0};
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            BlReloc *r = &efs[i].relocs[k];
            int ls = libsys_lookup(r->callee);
            if (ls < 0) continue;
            if (!libsys.used[ls]) {
                libsys.used[ls] = true;
                libsys.stub_index[ls] = libsys.n_stubs++;
            }
        }
    }

    // -----------------------------------------------------------------
    // Patch `bl` PC-relative displacements. Each reloc identifies its
    // callee by symbol name. Resolve against (in order):
    //   1. local functions  — branch to efs[j].text_off
    //   2. libSystem stubs   — branch to text_size + stub_index * 12
    // anything else is a hard error (no inter-module-text linking yet).
    // -----------------------------------------------------------------
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            BlReloc *r = &efs[i].relocs[k];
            uint32_t dst_pc;
            bool resolved = false;
            for (size_t j = 0; j < n_funcs; j++) {
                if (efs[j].name.size == r->callee.size
                    && memcmp(efs[j].name.str, r->callee.str,
                              r->callee.size) == 0) {
                    dst_pc = efs[j].text_off;
                    resolved = true;
                    break;
                }
            }
            if (!resolved) {
                int ls = libsys_lookup(r->callee);
                if (ls >= 0 && libsys.used[ls]) {
                    dst_pc = text_size + libsys.stub_index[ls] * 12u;
                    resolved = true;
                }
            }
            if (!resolved) {
                fprintf(stderr,
                    "aarch64->macho: bl to unknown symbol '%.*s'\n",
                    (int)r->callee.size, r->callee.str);
                for (size_t k2 = 0; k2 < n_funcs; k2++) {
                    free(efs[k2].code.data); free(efs[k2].relocs); free(efs[k2].dr);
                    free(efs[k2].br); free(efs[k2].bp);
                }
                free(efs); return false;
            }
            uint32_t src_pc = efs[i].text_off + r->fn_off;
            int32_t  rel    = (int32_t)dst_pc - (int32_t)src_pc;
            int32_t  imm26  = rel >> 2;
            uint32_t insn   = arm64_bl(imm26);
            efs[i].code.data[r->fn_off + 0] = (uint8_t)(insn      );
            efs[i].code.data[r->fn_off + 1] = (uint8_t)(insn >>  8);
            efs[i].code.data[r->fn_off + 2] = (uint8_t)(insn >> 16);
            efs[i].code.data[r->fn_off + 3] = (uint8_t)(insn >> 24);
        }
    }

    // -----------------------------------------------------------------
    // Layout constants. n_stubs is the count of libSystem imports the
    // BL discovery pass found. See mlir_wasm_to_macho.c for the full
    // annotated walk-through of the load-command shape.
    // -----------------------------------------------------------------
    const uint32_t n_stubs = libsys.n_stubs;
    const uint32_t stub_size = 12;
    const uint32_t got_size  = n_stubs * 8;

    // sizeofcmds varies with __DATA presence. We use up to 2 sections in
    // __DATA: __data + __linmem_template. Linear memory itself is
    // allocated dynamically by `_start` via mmap (see synth_start in
    // mlir_wmir_to_aarch64.c), so the binary no longer reserves a
    // multi-GiB __linmem_bss zerofill section — that placement is
    // incompatible with the dyld shared cache on macOS arm64.
    uint32_t n_data_sections  = 0;
    if (has_data_seg) n_data_sections = 1;
    if (linmem_init_size > 0) n_data_sections = 2;
    uint32_t data_seg_lc_size = has_data_seg
        ? (72u + n_data_sections * 80u)
        : 0u;
    const uint32_t n_cmds      = has_data_seg ? 18u : 17u;
    const uint32_t sizeofcmds  = 976u + data_seg_lc_size;
    uint32_t text_section_off  = (32u + sizeofcmds + 15u) & ~15u;
    if (text_section_off < 1040u) text_section_off = 1040u;

    const uint32_t stubs_off    = text_section_off + text_size;
    const uint32_t stubs_size   = n_stubs * stub_size;
    const uint32_t cstring_off  = stubs_off + stubs_size;
    const uint32_t cstring_size = 0;

    uint32_t text_seg_end  = cstring_off + cstring_size;
    uint32_t text_seg_size = (text_seg_end + (VMSEG_SIZE - 1u)) & ~(VMSEG_SIZE - 1u);
    if (text_seg_size < VMSEG_SIZE) text_seg_size = VMSEG_SIZE;
    const uint64_t data_const_vm_base   = TEXT_VM_BASE + text_seg_size;
    const uint32_t data_const_file_base = text_seg_size;

    // __DATA segment (between __DATA_CONST and __LINKEDIT). Up to 2
    // sections, all file-backed and laid out sequentially:
    //   __data            S_REGULAR: data_priv (32B) + globals
    //   __linmem_template S_REGULAR: byte-for-byte image of the wasm
    //                                data segments; copied into the
    //                                mmap-allocated linmem by `_start`
    //                                (only present if linmem_init_size > 0)
    uint32_t data_section_payload = data_priv_size + globals_padded;
    uint32_t data_seg_payload    = data_section_payload + linmem_init_size;
    uint64_t data_seg_filesize_v = has_data_seg
        ? (((uint64_t)data_seg_payload + (VMSEG_SIZE - 1u)) & ~(uint64_t)(VMSEG_SIZE - 1u))
        : 0u;
    uint64_t data_seg_vmsize     = data_seg_filesize_v;

    const uint64_t data_vm_base       = data_const_vm_base + VMSEG_SIZE;
    const uint32_t data_file_base     = data_const_file_base + VMSEG_SIZE;
    const uint64_t data_priv_vmaddr   = data_vm_base;
    const uint64_t globals_vmaddr     = data_priv_vmaddr + data_priv_size;
    const uint64_t linmem_tpl_vmaddr  = globals_vmaddr + globals_padded;

    const uint64_t linkedit_vm_base   = has_data_seg
        ? (data_vm_base + data_seg_vmsize)
        : (data_const_vm_base + VMSEG_SIZE);
    const uint32_t linkedit_file_base = has_data_seg
        ? (data_file_base + (uint32_t)data_seg_filesize_v)
        : (data_const_file_base + VMSEG_SIZE);

    // -----------------------------------------------------------------
    // Build the image.
    // -----------------------------------------------------------------
    Buf img = {0};
    img.cap = 1 << 15; img.data = (uint8_t *)malloc(img.cap);

    // mach_header_64
    buf_le32(&img, MH_MAGIC_64);
    buf_le32(&img, CPU_TYPE_ARM64);
    buf_le32(&img, 0);
    buf_le32(&img, MH_EXECUTE);
    buf_le32(&img, n_cmds);
    buf_le32(&img, sizeofcmds);
    buf_le32(&img, MH_FLAGS_EXEC);
    buf_le32(&img, 0);

    // LC_SEGMENT_64 __PAGEZERO
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 72);
    { static const char SEG[16] = "__PAGEZERO"; buf_append(&img, SEG, 16); }
    buf_le64(&img, 0);
    buf_le64(&img, TEXT_VM_BASE);
    buf_le64(&img, 0); buf_le64(&img, 0);
    buf_le32(&img, 0); buf_le32(&img, 0);
    buf_le32(&img, 0); buf_le32(&img, 0);

    // LC_SEGMENT_64 __TEXT (3 sections: __text, __stubs, __cstring)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 312);
    { static const char SEG[16] = "__TEXT"; buf_append(&img, SEG, 16); }
    buf_le64(&img, TEXT_VM_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le64(&img, TEXT_FILE_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, 3);
    buf_le32(&img, 0);
    {
        static const char SN[16] = "__text";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + text_section_off);
        buf_le64(&img, (uint64_t)text_size);
        buf_le32(&img, text_section_off);
        buf_le32(&img, 4);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000400u);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0);
    }
    {
        static const char SN[16] = "__stubs";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + stubs_off);
        buf_le64(&img, (uint64_t)stubs_size);
        buf_le32(&img, stubs_off);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000408u);
        buf_le32(&img, n_stubs);
        buf_le32(&img, stub_size);
        buf_le32(&img, 0);
    }
    {
        static const char SN[16] = "__cstring";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + cstring_off);
        buf_le64(&img, (uint64_t)cstring_size);
        buf_le32(&img, cstring_off);
        buf_le32(&img, 0);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0);
    }

    // LC_SEGMENT_64 __DATA_CONST (1 section: __got, empty)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 152);
    { static const char SEG[16] = "__DATA_CONST"; buf_append(&img, SEG, 16); }
    buf_le64(&img, data_const_vm_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, (uint64_t)data_const_file_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, 1);
    buf_le32(&img, 16);                  // SG_READ_ONLY
    {
        static const char SN[16] = "__got";
        static const char SG[16] = "__DATA_CONST";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, data_const_vm_base);
        buf_le64(&img, (uint64_t)got_size);
        buf_le32(&img, data_const_file_base);
        buf_le32(&img, 3);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 6);
        buf_le32(&img, 0);
        buf_le32(&img, 0);
        buf_le32(&img, 0);
    }

    // LC_SEGMENT_64 __DATA (optional). Up to two sections, both
    // file-backed and laid out sequentially:
    //   __data            : data_priv + globals
    //   __linmem_template : init bytes copied by `_start` into mmap'd linmem
    if (has_data_seg) {
        buf_le32(&img, LC_SEGMENT_64);
        buf_le32(&img, 72u + n_data_sections * 80u);
        { static const char SEG[16] = "__DATA"; buf_append(&img, SEG, 16); }
        buf_le64(&img, data_vm_base);
        buf_le64(&img, data_seg_vmsize);
        buf_le64(&img, (uint64_t)data_file_base);
        buf_le64(&img, data_seg_filesize_v);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, n_data_sections);
        buf_le32(&img, 0);
        {
            static const char SN[16] = "__data";
            static const char SG[16] = "__DATA";
            buf_append(&img, SN, 16); buf_append(&img, SG, 16);
            buf_le64(&img, data_vm_base);
            buf_le64(&img, (uint64_t)data_section_payload);
            buf_le32(&img, data_file_base);
            buf_le32(&img, 3);                // align = 2^3 = 8
            buf_le32(&img, 0); buf_le32(&img, 0);
            buf_le32(&img, 0);                // S_REGULAR
            buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
        }
        if (linmem_init_size > 0) {
            static const char SN[16] = "__linmem_tpl";
            static const char SG[16] = "__DATA";
            buf_append(&img, SN, 16); buf_append(&img, SG, 16);
            buf_le64(&img, linmem_tpl_vmaddr);
            buf_le64(&img, (uint64_t)linmem_init_size);
            buf_le32(&img, data_file_base + data_section_payload);
            buf_le32(&img, 4);                // align = 2^4 = 16
            buf_le32(&img, 0); buf_le32(&img, 0);
            buf_le32(&img, 0);                // S_REGULAR
            buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
        }
    }

    // LC_SEGMENT_64 __LINKEDIT (filesize patched at end)
    size_t pos_linkedit_seg = img.len;
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 72);
    { static const char SEG[16] = "__LINKEDIT"; buf_append(&img, SEG, 16); }
    buf_le64(&img, linkedit_vm_base);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, linkedit_file_base);
    buf_le64(&img, 0);                   // PLACEHOLDER filesize
    buf_le32(&img, VM_PROT_READ);
    buf_le32(&img, VM_PROT_READ);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_linkedit_filesize = pos_linkedit_seg + 8 + 16 + 8 + 8 + 8;

    size_t pos_lc_chained_fixups   = img.len;
    buf_le32(&img, LC_DYLD_CHAINED_FIXUPS); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_exports_trie     = img.len;
    buf_le32(&img, LC_DYLD_EXPORTS_TRIE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_symtab           = img.len;
    buf_le32(&img, LC_SYMTAB); buf_le32(&img, 24);
    buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_dysymtab         = img.len;
    buf_le32(&img, LC_DYSYMTAB); buf_le32(&img, 80);
    for (int k = 0; k < 18; k++) buf_le32(&img, 0);

    // LC_LOAD_DYLINKER
    buf_le32(&img, LC_LOAD_DYLINKER);
    buf_le32(&img, 32);
    buf_le32(&img, 12);
    { static const char nm[20] = "/usr/lib/dyld"; buf_append(&img, nm, 20); }

    // LC_UUID
    buf_le32(&img, LC_UUID); buf_le32(&img, 24);
    {
        static const uint8_t U[16] = {
            0x27,0x07,0xdd,0x62,0x09,0x67,0x3c,0xc0,
            0xb2,0xac,0xef,0xc3,0x2b,0x1c,0xf6,0x3a};
        buf_append(&img, U, 16);
    }

    // LC_BUILD_VERSION
    buf_le32(&img, LC_BUILD_VERSION); buf_le32(&img, 32);
    buf_le32(&img, 1);
    buf_le32(&img, 0x000f0700);
    buf_le32(&img, 0);
    buf_le32(&img, 1);
    buf_le32(&img, 3);
    buf_le32(&img, 0x04ce0100);

    // LC_SOURCE_VERSION
    buf_le32(&img, LC_SOURCE_VERSION); buf_le32(&img, 16);
    buf_le64(&img, 0);

    // LC_MAIN
    // Initial stack size: 256 MB. This is large because the wmir
    // backend currently doesn't promote wasm locals to physical
    // registers — every local lives in an 8-byte stack cell, and
    // the regalloc spills aggressively (no live-range splitting).
    // Real-world tinyc functions like emit_expr end up with ~225 KB
    // per-call frames, so the parser only gets ~35 levels of
    // recursion on the macOS default 8 MB stack and OOMs on
    // deeply-nested expressions. 256 MB gives ~1100 emit_expr
    // levels which is plenty for tinyc's selfhost workload.
    buf_le32(&img, LC_MAIN); buf_le32(&img, 24);
    buf_le64(&img, (uint64_t)text_section_off);
    buf_le64(&img, 256ULL * 1024 * 1024);

    // LC_LOAD_DYLIB libSystem (kept so the load-command layout matches
    // the macho_exit reference even though we don't import anything).
    buf_le32(&img, LC_LOAD_DYLIB); buf_le32(&img, 56);
    buf_le32(&img, 24);
    buf_le32(&img, 2);
    buf_le32(&img, 0x054c0000);
    buf_le32(&img, 0x00010000);
    { static const char nm[32] = "/usr/lib/libSystem.B.dylib"; buf_append(&img, nm, 32); }

    size_t pos_lc_function_starts  = img.len;
    buf_le32(&img, LC_FUNCTION_STARTS); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_data_in_code     = img.len;
    buf_le32(&img, LC_DATA_IN_CODE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);
    size_t pos_lc_code_sig         = img.len;
    buf_le32(&img, LC_CODE_SIGNATURE); buf_le32(&img, 16);
    buf_le32(&img, 0); buf_le32(&img, 0);

    buf_pad_to(&img, text_section_off);

    // __text — _start first, then everything else.
    buf_append(&img, efs[start_idx].code.data, efs[start_idx].code.len);
    for (size_t i = 0; i < n_funcs; i++) {
        if (i == start_idx) continue;
        buf_append(&img, efs[i].code.data, efs[i].code.len);
    }
    buf_pad_to(&img, stubs_off);

    // __stubs: one ADRP/LDR/BR triple per libSystem import. Stub `i`
    // points at __got slot `i`; dyld populates the slot at load time
    // via the chained-fixups blob we emit below.
    {
        uint64_t data_const_vm_base_64 = TEXT_VM_BASE + (uint64_t)text_seg_size;
        for (uint32_t i = 0; i < n_stubs; i++) {
            uint64_t got_target = data_const_vm_base_64 + 8ULL * i;
            uint64_t stub_addr  = TEXT_VM_BASE + (uint64_t)stubs_off
                                + (uint64_t)i * stub_size;
            uint64_t page_dst   = got_target & ~0xfffULL;
            uint64_t page_src   = stub_addr  & ~0xfffULL;
            int64_t  page_diff  = (int64_t)(page_dst - page_src);
            int64_t  page_imm   = page_diff >> 12;
            uint32_t immlo = (uint32_t)(page_imm & 0x3);
            uint32_t immhi = (uint32_t)((page_imm >> 2) & 0x7ffff);
            uint32_t adrp  = 0x90000010u | (immlo << 29) | (immhi << 5);
            uint32_t lo12  = (uint32_t)(got_target & 0xfffu);
            uint32_t ldr   = 0xf9400210u | (((lo12 >> 3) & 0xfffu) << 10);
            uint32_t br    = 0xd61f0200u;  // br x16
            buf_le32(&img, adrp);
            buf_le32(&img, ldr);
            buf_le32(&img, br);
        }
    }
    buf_pad_to(&img, cstring_off);
    // __cstring empty.

    // Pad to __DATA_CONST.
    buf_pad_to(&img, data_const_file_base);
    // __got: one chained-fixup-format pointer per libSystem import. The
    // format is DYLD_CHAINED_PTR_64_OFFSET (pointer_format=6). Layout
    // (LSB-first):
    //   bits  0..23   ordinal (= imports[] index)
    //   bits 24..31   addend
    //   bits 32..50   reserved
    //   bits 51..62   next (stride 4; step 2 to reach the next 8B slot)
    //   bit      63   bind (1)
    for (uint32_t i = 0; i < n_stubs; i++) {
        uint64_t v = (1ULL << 63) | (uint64_t)i;
        if (i + 1 < n_stubs) v |= (2ULL << 51);
        buf_le64(&img, v);
    }

    // -----------------------------------------------------------------
    // __DATA file content (if any): __data section (data_priv + globals)
    // followed by __linmem_template bytes (file-backed wasm init data
    // that `_start` will memcpy into mmap'd linmem). No zerofill — the
    // remainder of linmem is anonymous mmap'd RAM.
    // -----------------------------------------------------------------
    if (has_data_seg) {
        buf_pad_to(&img, data_file_base);
        // data_priv: 32 zero bytes.
        for (uint32_t k = 0; k < data_priv_size; k++) buf_u8(&img, 0);
        // globals: 8 bytes each. global[0] = global0_init; rest = 0.
        for (uint32_t k = 0; k < n_globals; k++) {
            uint64_t v = (k == 0) ? global0_init : 0;
            buf_le64(&img, v);
        }
        // Pad globals to 16.
        while ((img.len - (data_file_base + data_priv_size)) < globals_padded) {
            buf_u8(&img, 0);
        }
        // __linmem_template section: zero-initialise the whole window
        // then overlay each data_init record's bytes at its
        // (wasm-offset relative) position. _start memcpys this into
        // mmap'd linmem at runtime.
        if (linmem_init_size > 0) {
            uint32_t init_start = (uint32_t)img.len;
            for (uint32_t k = 0; k < linmem_init_size; k++) buf_u8(&img, 0);
            for (size_t k = 0; k < n_inits; k++) {
                uint32_t off = inits[k].offset;
                string   bs  = inits[k].bytes;
                if ((uint32_t)(off + bs.size) > linmem_init_size) {
                    fprintf(stderr,
                        "aarch64->macho: data_init overflows init region\n");
                    free(efs); free(inits); free(img.data); return false;
                }
                memcpy(img.data + init_start + off, bs.str, bs.size);
            }
        }
        // Pad to data_seg_filesize (VMSEG boundary).
        buf_pad_to(&img, data_file_base + (uint32_t)data_seg_filesize_v);
    }

    buf_pad_to(&img, linkedit_file_base);

    // -----------------------------------------------------------------
    // Patch ADRP / ADD imm12 for data-segment references. Done after
    // text content lives in `img` so we know its absolute offset.
    // -----------------------------------------------------------------
    for (size_t i = 0; i < n_funcs; i++) {
        for (size_t k = 0; k < efs[i].n_dr; k++) {
            DataReloc *dr = &efs[i].dr[k];
            uint64_t dst_vm;
            if (dr->kind.size == 9 && memcmp(dr->kind.str, "data_priv", 9) == 0) {
                dst_vm = data_priv_vmaddr;
            } else if (dr->kind.size == 7 && memcmp(dr->kind.str, "globals", 7) == 0) {
                dst_vm = globals_vmaddr;
            } else if (dr->kind.size == 15 && memcmp(dr->kind.str, "linmem_template", 15) == 0) {
                dst_vm = linmem_tpl_vmaddr;
            } else {
                fprintf(stderr,
                    "aarch64->macho: unknown data reloc kind '%.*s'\n",
                    (int)dr->kind.size, dr->kind.str);
                for (size_t k2 = 0; k2 < n_funcs; k2++) {
                    free(efs[k2].code.data); free(efs[k2].relocs); free(efs[k2].dr);
                }
                free(efs); free(inits); free(img.data); return false;
            }
            uint64_t src_pc = TEXT_VM_BASE + (uint64_t)text_section_off
                            + (uint64_t)efs[i].text_off + (uint64_t)dr->fn_off;
            size_t   img_off = text_section_off + efs[i].text_off + dr->fn_off;
            uint32_t insn;
            if (!dr->is_add_lo) {
                int64_t dst_page = (int64_t)(dst_vm >> 12);
                int64_t src_page = (int64_t)(src_pc >> 12);
                int64_t rel_pages = dst_page - src_page;
                insn = arm64_adrp(dr->rd, rel_pages);
            } else {
                uint16_t imm12 = (uint16_t)(dst_vm & 0xfffu);
                insn = arm64_add_imm(dr->rd, dr->rn, imm12, /*sf=*/true);
            }
            img.data[img_off + 0] = (uint8_t)(insn      );
            img.data[img_off + 1] = (uint8_t)(insn >>  8);
            img.data[img_off + 2] = (uint8_t)(insn >> 16);
            img.data[img_off + 3] = (uint8_t)(insn >> 24);
        }
    }

    // =====================================================================
    // __LINKEDIT
    // =====================================================================
    size_t linkedit_start = img.len;

    // ---- chained fixups blob (no imports) ----
    size_t chained_start = img.len;
    uint32_t fx_imports_off = 0x50u;
    uint32_t fx_symbols_off = fx_imports_off + n_stubs * 4u;
    buf_le32(&img, 0);
    buf_le32(&img, 0x20);
    buf_le32(&img, fx_imports_off);
    buf_le32(&img, fx_symbols_off);
    buf_le32(&img, n_stubs);
    buf_le32(&img, 1);
    buf_le32(&img, 0);
    buf_le32(&img, 0);

    uint32_t fx_seg_count = 4u;
    buf_le32(&img, fx_seg_count);
    buf_le32(&img, 0); buf_le32(&img, 0);
    buf_le32(&img, 0x18);
    buf_le32(&img, 0);
    buf_le32(&img, 0);

    buf_le32(&img, 0x18);
    buf_le16(&img, 0x4000);
    buf_le16(&img, 6);
    buf_le64(&img, (uint64_t)data_const_file_base);
    buf_le32(&img, 0);
    buf_le16(&img, 1);
    // page_start[0]: 0 = chain starts at byte 0 of the page; 0xffff =
    // DYLD_CHAINED_PTR_START_NONE (no fixups on this page). When n_stubs
    // is 0 there's nothing to fix up so we mark the page as having no
    // chain to avoid dyld walking uninitialised bytes.
    buf_le16(&img, n_stubs > 0 ? 0 : 0xffff);

    // imports table: one DYLD_CHAINED_IMPORT (4B) per stub.
    //   bits  0..7   lib_ordinal (= 1 for libSystem.B.dylib, the only
    //                  LC_LOAD_DYLIB load command)
    //   bit       8  weak_import (0)
    //   bits  9..31  name_offset into symbols table
    //
    // For each stub_index i, we find which libsys name has stub_index==i
    // and emit it in that order — the linker builds the chained-fixup
    // chain in this order, so import[i] must match GOT slot i.
    uint32_t libsys_name_offsets[LS_COUNT] = {0};
    {
        // Pre-compute the byte offset of each libsys name in the
        // symbol-names blob. Leading 0 byte is at offset 0; first name
        // at offset 1.
        uint32_t off = 1;  // skip the leading 0
        for (uint32_t i = 0; i < n_stubs; i++) {
            // Find which libsys name has stub_index == i.
            for (size_t k = 0; k < LS_COUNT; k++) {
                if (libsys.used[k] && libsys.stub_index[k] == i) {
                    libsys_name_offsets[k] = off;
                    off += (uint32_t)strlen(libsys_name(k)) + 1u;
                    break;
                }
            }
        }
    }
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                uint32_t lib_ordinal = 1;
                uint32_t name_off    = libsys_name_offsets[k];
                buf_le32(&img,
                    (lib_ordinal & 0xffu)
                    | ((name_off & 0x7fffffu) << 9));
                break;
            }
        }
    }

    // symbols table: leading 0, then each used libsys name in
    // stub-index order, NUL-terminated. Padded to multiple of 8 below.
    buf_u8(&img, 0);
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                const char *nm = libsys_name(k);
                size_t nl = strlen(nm);
                for (size_t j = 0; j < nl; j++) buf_u8(&img, (uint8_t)nm[j]);
                buf_u8(&img, 0);
                break;
            }
        }
    }
    while ((img.len - chained_start) % 8) buf_u8(&img, 0);
    uint32_t chained_size = (uint32_t)(img.len - chained_start);

    // ---- exports trie ----
    size_t exports_start = img.len;
    buf_u8(&img, 0x00); buf_u8(&img, 0x01);
    buf_cstr(&img, "_");
    buf_uleb(&img, 0x12);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x03); buf_u8(&img, 0x00);
    buf_uleb(&img, (uint64_t)text_section_off);
    buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_cstr(&img, "_mh_execute_header"); buf_uleb(&img, 0x09);
    buf_cstr(&img, "main"); buf_uleb(&img, 0x0d);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00);
    while ((img.len - exports_start) % 8) buf_u8(&img, 0);
    uint32_t exports_size = (uint32_t)(img.len - exports_start);

    // ---- function starts ----
    size_t fs_start = img.len;
    buf_uleb(&img, (uint64_t)text_section_off);
    buf_u8(&img, 0);
    while ((img.len - fs_start) % 8) buf_u8(&img, 0);
    uint32_t fs_size = (uint32_t)(img.len - fs_start);

    // ---- symtab + strtab ----
    // Layout: [defined syms (mh, main)] + [undef syms (libsys imports)],
    // then strtab. Undef syms come AFTER defined ones because LC_DYSYMTAB
    // describes them as contiguous ranges (iundefsym = 2 for our two
    // defined syms). Indirect-syms references __stubs section, one
    // entry per stub, pointing at the symtab index of its undef sym.
    Buf strtab = {0};
    buf_u8(&strtab, 0x20);
    buf_u8(&strtab, 0x00);
    uint32_t str_mh   = (uint32_t)strtab.len; buf_cstr(&strtab, "__mh_execute_header");
    uint32_t str_main = (uint32_t)strtab.len; buf_cstr(&strtab, "_main");
    // libsys symbol names in stub-index order.
    uint32_t libsys_str_off[LS_COUNT] = {0};
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                libsys_str_off[k] = (uint32_t)strtab.len;
                buf_cstr(&strtab, libsys_name(k));
                break;
            }
        }
    }
    while (strtab.len % 8) buf_u8(&strtab, 0);

    Buf symtab = {0};
    // __mh_execute_header
    buf_le32(&symtab, str_mh);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0010);
    buf_le64(&symtab, TEXT_VM_BASE);
    // _main
    buf_le32(&symtab, str_main);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0000);
    buf_le64(&symtab, TEXT_VM_BASE + text_section_off);
    // libsys undefs (n_type=N_UNDF|N_EXT=0x01, n_sect=0, n_desc=lib_ordinal<<8,
    // n_value=0). Order MUST match stub_index so indirect-syms can
    // point at symtab index (2 + stub_index).
    for (uint32_t i = 0; i < n_stubs; i++) {
        for (size_t k = 0; k < LS_COUNT; k++) {
            if (libsys.used[k] && libsys.stub_index[k] == i) {
                buf_le32(&symtab, libsys_str_off[k]);
                buf_u8(&symtab, 0x01);          // N_UNDF | N_EXT
                buf_u8(&symtab, 0);             // n_sect = NO_SECT
                // n_desc: lib_ordinal in high byte (libSystem = 1).
                // Bits: REFERENCE_FLAG_UNDEFINED_NON_LAZY (0) + ordinal.
                buf_le16(&symtab, (uint16_t)(1u << 8));
                buf_le64(&symtab, 0);
                break;
            }
        }
    }

    uint32_t n_syms       = 2u + n_stubs;
    uint32_t n_undefs     = n_stubs;
    uint32_t iundefsym    = 2;

    // Indirect-symbols table: 2*n_stubs entries. The first n_stubs are
    // for the __got section (reserved1=0 in the section header); the
    // next n_stubs are for __stubs (reserved1=n_stubs). Both blocks
    // hold the same values: indirect_sym[i] = symtab index of the
    // undef sym that backs slot i. Our undefs start at symtab[2].
    Buf indsyms = {0};
    for (uint32_t i = 0; i < n_stubs; i++) buf_le32(&indsyms, 2u + i);  // GOT
    for (uint32_t i = 0; i < n_stubs; i++) buf_le32(&indsyms, 2u + i);  // stubs

    size_t symtab_off  = img.len;
    buf_append(&img, symtab.data, symtab.len);
    size_t indsyms_off = img.len;
    buf_append(&img, indsyms.data, indsyms.len);
    size_t strtab_off  = img.len;
    buf_append(&img, strtab.data, strtab.len);

    while ((img.len - linkedit_start) % 16) buf_u8(&img, 0);

    // ---- code signature ----
    uint32_t code_limit = (uint32_t)img.len;
    size_t code_sig_off = img.len;

    const uint32_t page_size  = 4096;
    const uint32_t page_shift = 12;
    const char *ident = "tinyc.out";
    uint32_t ident_len = (uint32_t)strlen(ident);
    const uint32_t n_slots = (code_limit + page_size - 1) / page_size;
    const uint32_t ident_offset = 88;
    const uint32_t n_special_slots = 2;
    const uint32_t hash_offset = ident_offset + ident_len + 1
                               + n_special_slots * SHA256_DIGEST_LEN;
    const uint32_t cd_len  = hash_offset + n_slots * SHA256_DIGEST_LEN;
    const uint32_t req_len = 12;
    const uint32_t cms_len = 8;
    const uint32_t sb_header = 12 + 3 * 8;
    const uint32_t sb_len_unpadded = sb_header + cd_len + req_len + cms_len;
    const uint32_t code_sig_size = (sb_len_unpadded + 15u) & ~15u;

    buf_patch_le32(&img, pos_lc_chained_fixups   + 8,  (uint32_t)chained_start);
    buf_patch_le32(&img, pos_lc_chained_fixups   + 12, chained_size);
    buf_patch_le32(&img, pos_lc_exports_trie     + 8,  (uint32_t)exports_start);
    buf_patch_le32(&img, pos_lc_exports_trie     + 12, exports_size);
    buf_patch_le32(&img, pos_lc_symtab           + 8,  (uint32_t)symtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 12, n_syms);
    buf_patch_le32(&img, pos_lc_symtab           + 16, (uint32_t)strtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 20, (uint32_t)strtab.len);
    buf_patch_le32(&img, pos_lc_dysymtab         + 8,  0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 12, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 16, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 20, 2);
    buf_patch_le32(&img, pos_lc_dysymtab         + 24, iundefsym);
    buf_patch_le32(&img, pos_lc_dysymtab         + 28, n_undefs);
    buf_patch_le32(&img, pos_lc_dysymtab         + 32, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 36, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 40, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 44, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 48, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 52, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 56, (uint32_t)indsyms_off);
    buf_patch_le32(&img, pos_lc_dysymtab         + 60, 2u * n_stubs);
    buf_patch_le32(&img, pos_lc_dysymtab         + 64, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 68, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 72, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 76, 0);
    buf_patch_le32(&img, pos_lc_function_starts  + 8,  (uint32_t)fs_start);
    buf_patch_le32(&img, pos_lc_function_starts  + 12, fs_size);
    buf_patch_le32(&img, pos_lc_data_in_code     + 8,  (uint32_t)fs_start + fs_size);
    buf_patch_le32(&img, pos_lc_data_in_code     + 12, 0);
    buf_patch_le32(&img, pos_lc_code_sig         + 8,  (uint32_t)code_sig_off);
    buf_patch_le32(&img, pos_lc_code_sig         + 12, code_sig_size);

    uint64_t linkedit_filesize_v =
        (uint64_t)(code_limit + code_sig_size) - linkedit_file_base;
    uint64_t linkedit_vmsize_v =
        (linkedit_filesize_v + (VMSEG_SIZE - 1u)) & ~(uint64_t)(VMSEG_SIZE - 1u);
    if (linkedit_vmsize_v < VMSEG_SIZE) linkedit_vmsize_v = VMSEG_SIZE;
    buf_patch_le64(&img, pos_linkedit_filesize,        linkedit_filesize_v);
    buf_patch_le64(&img, pos_linkedit_filesize - 16,   linkedit_vmsize_v);

    free(symtab.data); free(indsyms.data); free(strtab.data);

    Buf cs = {0};
    {
        Buf req = {0};
        buf_be32(&req, 0xfade0c01);
        buf_be32(&req, req_len);
        buf_be32(&req, 0);

        uint8_t req_hash[SHA256_DIGEST_LEN];
        sha256(req.data, req.len, req_hash);

        Buf cd = {0};
        buf_be32(&cd, 0xfade0c02);
        buf_be32(&cd, cd_len);
        buf_be32(&cd, 0x00020400);
        buf_be32(&cd, 0x00000002);          // CS_ADHOC
        buf_be32(&cd, hash_offset);
        buf_be32(&cd, ident_offset);
        buf_be32(&cd, n_special_slots);
        buf_be32(&cd, n_slots);
        buf_be32(&cd, code_limit);
        buf_u8(&cd, SHA256_DIGEST_LEN);
        buf_u8(&cd, 2);
        buf_u8(&cd, 0);
        buf_u8(&cd, page_shift);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        while (cd.len < 64) buf_u8(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, text_seg_size);
        buf_be32(&cd, 0);
        buf_be32(&cd, 1);                   // CS_EXECSEG_MAIN_BINARY
        while (cd.len < ident_offset) buf_u8(&cd, 0);
        buf_append(&cd, ident, ident_len);
        buf_u8(&cd, 0);
        buf_append(&cd, req_hash, SHA256_DIGEST_LEN);
        for (int z = 0; z < SHA256_DIGEST_LEN; z++) buf_u8(&cd, 0);
        for (uint32_t i = 0; i < n_slots; i++) {
            uint32_t start  = i * page_size;
            uint32_t remain = code_limit - start;
            uint32_t len    = remain < page_size ? remain : page_size;
            uint8_t  d[SHA256_DIGEST_LEN];
            sha256(img.data + start, len, d);
            buf_append(&cd, d, sizeof(d));
        }

        const uint32_t cd_off  = sb_header;
        const uint32_t req_off = cd_off + cd_len;
        const uint32_t cms_off = req_off + req_len;
        buf_be32(&cs, 0xfade0cc0);
        buf_be32(&cs, sb_len_unpadded);
        buf_be32(&cs, 3);
        buf_be32(&cs, 0x00000000);
        buf_be32(&cs, cd_off);
        buf_be32(&cs, 0x00000002);
        buf_be32(&cs, req_off);
        buf_be32(&cs, 0x00010000);
        buf_be32(&cs, cms_off);
        buf_append(&cs, cd.data, cd.len);
        buf_append(&cs, req.data, req.len);
        buf_be32(&cs, 0xfade0b01);
        buf_be32(&cs, cms_len);
        free(cd.data);
        free(req.data);
    }
    while (cs.len < code_sig_size) buf_u8(&cs, 0);
    buf_append(&img, cs.data, cs.len);
    free(cs.data);

    for (size_t i = 0; i < n_funcs; i++) {
        free(efs[i].code.data);
        free(efs[i].relocs);
        free(efs[i].dr);
        free(efs[i].br);
        free(efs[i].bp);
    }
    free(efs);
    free(inits);

    *out_data = img.data;
    *out_size = img.len;
    return true;
}

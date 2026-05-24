// WASM (linked) -> Mach-O ARM64 translator. See mlir_wasm_to_macho.h
// for the public API. The Mach-O envelope mirrors the byte layout of
// `test_wasm/write_macho.cpp` (the reference C++ implementation we
// validated against), and the codegen layer does the "literal stack"
// WASM->ARM64 translation described in the header.

#include "mlir_wasm_to_macho.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// Growable byte buffer.
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
// Patch a little-endian uint32_t at an existing buffer offset.
static void buf_patch_le32(Buf *b, size_t pos, uint32_t v) {
    b->data[pos + 0] = (uint8_t)(v & 0xff);
    b->data[pos + 1] = (uint8_t)((v >> 8) & 0xff);
    b->data[pos + 2] = (uint8_t)((v >> 16) & 0xff);
    b->data[pos + 3] = (uint8_t)((v >> 24) & 0xff);
}
static void buf_patch_le64(Buf *b, size_t pos, uint64_t v) {
    for (int k = 0; k < 8; k++) b->data[pos + (size_t)k] = (uint8_t)(v >> (8 * k));
}

// =============================================================================
// SHA-256 (self-contained; CommonCrypto isn't available everywhere).
// =============================================================================
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
        m[i] = ((uint32_t)d[j] << 24) | ((uint32_t)d[j+1] << 16) |
               ((uint32_t)d[j+2] << 8) | (uint32_t)d[j+3];
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
// WASM mini-reader (independent of mlir_wasm_to_wat.c).
// =============================================================================
typedef struct {
    const uint8_t *p, *end;
    bool overflow;
} Rd;

static uint8_t rd_u8(Rd *r) {
    if (r->p >= r->end) { r->overflow = true; return 0; }
    return *r->p++;
}
static uint64_t rd_uleb(Rd *r) {
    uint64_t v = 0; int shift = 0;
    while (1) {
        if (r->p >= r->end) { r->overflow = true; return v; }
        uint8_t b = *r->p++;
        v |= ((uint64_t)(b & 0x7f)) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift > 63) { r->overflow = true; return v; }
    }
    return v;
}
static int64_t rd_sleb(Rd *r) {
    int64_t v = 0; int shift = 0; uint8_t b = 0;
    while (1) {
        if (r->p >= r->end) { r->overflow = true; return v; }
        b = *r->p++;
        v |= ((int64_t)(b & 0x7f)) << shift;
        shift += 7;
        if (!(b & 0x80)) break;
        if (shift > 63) { r->overflow = true; return v; }
    }
    if (shift < 64 && (b & 0x40)) v |= -((int64_t)1 << shift);
    return v;
}

// =============================================================================
// WASM module summary.
//
// We pull out only what the backend needs:
//   * the type table (for function arities),
//   * the imports (so func indices line up; we additionally detect
//     `wasi_snapshot_preview1.proc_exit`),
//   * the FUNCTION section (defined-func index -> type idx),
//   * the bodies in CODE,
//   * the export named `_start`.
// =============================================================================
typedef struct {
    const uint8_t *code;
    size_t         code_size;
    uint32_t       type_idx;
} WasmFunc;

typedef struct {
    uint8_t  valtype;     // 0x7f i32, 0x7e i64
    uint8_t  mut;
    int64_t  init_value;  // result of constant init expression
} WasmGlobal;

typedef struct {
    uint32_t       offset;   // linear-memory offset where bytes go
    const uint8_t *bytes;
    uint32_t       size;
} WasmDataSeg;

typedef struct {
    const uint8_t *types_blob;     // points just past the count uleb
    uint32_t       n_types;

    uint32_t       n_imported_funcs;
    uint32_t       import_idx_proc_exit;

    WasmFunc      *funcs;
    uint32_t       n_funcs;

    uint32_t       start_export_idx;   // combined index

    bool           has_memory;
    uint32_t       memory_min_pages;   // pages of 64 KiB

    WasmGlobal    *globals;
    uint32_t       n_globals;

    WasmDataSeg   *datas;
    uint32_t       n_datas;

    // Filled in by MLIR_WasmToMachoArm64 after layout.
    uint64_t       linmem_vmaddr;
    uint64_t       globals_vmaddr;
    uint64_t       entry_vmaddr;       // VM address of _start's first instruction
} WasmModule;

static void wasm_module_free(WasmModule *m) {
    free(m->funcs); free(m->globals); free(m->datas);
}

static bool wasm_parse(const uint8_t *bytes, size_t size, WasmModule *out) {
    memset(out, 0, sizeof(*out));
    out->import_idx_proc_exit = UINT32_MAX;
    out->start_export_idx     = UINT32_MAX;

    if (size < 8 || memcmp(bytes, "\x00" "asm" "\x01\x00\x00\x00", 8) != 0) {
        fprintf(stderr, "wasm->macho: not a wasm module\n");
        return false;
    }

    Rd r = { bytes + 8, bytes + size, false };
    while (r.p < r.end && !r.overflow) {
        uint8_t id = rd_u8(&r);
        uint32_t sz = (uint32_t)rd_uleb(&r);
        const uint8_t *sec_end = r.p + sz;
        if (sec_end > r.end) { r.overflow = true; break; }
        Rd s = { r.p, sec_end, false };

        switch (id) {
            case 1: { // TYPE
                out->n_types = (uint32_t)rd_uleb(&s);
                out->types_blob = s.p;
                break;
            }
            case 2: { // IMPORT
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n && !s.overflow; i++) {
                    uint32_t mod_len = (uint32_t)rd_uleb(&s);
                    const uint8_t *mod_p = s.p;
                    if (s.p + mod_len > s.end) { s.overflow = true; break; }
                    s.p += mod_len;
                    uint32_t nm_len = (uint32_t)rd_uleb(&s);
                    const uint8_t *nm_p = s.p;
                    if (s.p + nm_len > s.end) { s.overflow = true; break; }
                    s.p += nm_len;
                    uint8_t kind = rd_u8(&s);
                    if (kind == 0) {
                        uint32_t fi = out->n_imported_funcs;
                        (void)rd_uleb(&s);
                        if (mod_len == 22 &&
                            memcmp(mod_p, "wasi_snapshot_preview1", 22) == 0 &&
                            nm_len == 9 && memcmp(nm_p, "proc_exit", 9) == 0) {
                            out->import_idx_proc_exit = fi;
                        }
                        out->n_imported_funcs++;
                    } else if (kind == 1) {
                        rd_u8(&s); uint8_t fl = rd_u8(&s);
                        (void)rd_uleb(&s); if (fl & 1) (void)rd_uleb(&s);
                    } else if (kind == 2) {
                        uint8_t fl = rd_u8(&s);
                        (void)rd_uleb(&s); if (fl & 1) (void)rd_uleb(&s);
                    } else if (kind == 3) {
                        rd_u8(&s); rd_u8(&s);
                    }
                }
                break;
            }
            case 3: { // FUNCTION
                uint32_t n = (uint32_t)rd_uleb(&s);
                out->funcs = (WasmFunc *)calloc(n, sizeof(WasmFunc));
                out->n_funcs = n;
                for (uint32_t i = 0; i < n && !s.overflow; i++) {
                    out->funcs[i].type_idx = (uint32_t)rd_uleb(&s);
                }
                break;
            }
            case 5: { // MEMORY
                uint32_t n = (uint32_t)rd_uleb(&s);
                if (n > 0) {
                    uint8_t fl = rd_u8(&s);
                    uint32_t mn = (uint32_t)rd_uleb(&s);
                    if (fl & 1) (void)rd_uleb(&s);
                    out->has_memory = true;
                    out->memory_min_pages = mn;
                }
                // Additional memories (multi-memory) ignored.
                for (uint32_t i = 1; i < n && !s.overflow; i++) {
                    uint8_t fl = rd_u8(&s);
                    (void)rd_uleb(&s); if (fl & 1) (void)rd_uleb(&s);
                }
                break;
            }
            case 6: { // GLOBAL
                uint32_t n = (uint32_t)rd_uleb(&s);
                out->globals = (WasmGlobal *)calloc(
                    n ? n : 1, sizeof(WasmGlobal));
                out->n_globals = n;
                for (uint32_t i = 0; i < n && !s.overflow; i++) {
                    uint8_t vt  = rd_u8(&s);
                    uint8_t mut = rd_u8(&s);
                    // Init expr: one constant instruction followed by 0x0b.
                    int64_t v = 0;
                    uint8_t op = rd_u8(&s);
                    if (op == 0x41 /* i32.const */) {
                        v = rd_sleb(&s);
                    } else if (op == 0x42 /* i64.const */) {
                        v = rd_sleb(&s);
                    } else {
                        fprintf(stderr,
                                "wasm->macho: global %u has unsupported "
                                "init opcode 0x%02x\n", i, op);
                        s.overflow = true;
                        break;
                    }
                    uint8_t e = rd_u8(&s);
                    if (e != 0x0b) {
                        fprintf(stderr,
                                "wasm->macho: global %u init expr does not "
                                "end with 0x0b (got 0x%02x)\n", i, e);
                        s.overflow = true;
                        break;
                    }
                    out->globals[i].valtype    = vt;
                    out->globals[i].mut        = mut;
                    out->globals[i].init_value = v;
                }
                break;
            }
            case 7: { // EXPORT
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n && !s.overflow; i++) {
                    uint32_t nm_len = (uint32_t)rd_uleb(&s);
                    const uint8_t *nm_p = s.p;
                    if (s.p + nm_len > s.end) { s.overflow = true; break; }
                    s.p += nm_len;
                    uint8_t kind = rd_u8(&s);
                    uint32_t idx  = (uint32_t)rd_uleb(&s);
                    if (kind == 0 && nm_len == 6 && memcmp(nm_p, "_start", 6) == 0) {
                        out->start_export_idx = idx;
                    }
                }
                break;
            }
            case 10: { // CODE
                uint32_t n = (uint32_t)rd_uleb(&s);
                uint32_t lim = n < out->n_funcs ? n : out->n_funcs;
                for (uint32_t i = 0; i < lim && !s.overflow; i++) {
                    uint32_t body_sz = (uint32_t)rd_uleb(&s);
                    const uint8_t *body_p = s.p;
                    if (s.p + body_sz > s.end) { s.overflow = true; break; }
                    out->funcs[i].code      = body_p;
                    out->funcs[i].code_size = body_sz;
                    s.p += body_sz;
                }
                break;
            }
            case 11: { // DATA
                uint32_t n = (uint32_t)rd_uleb(&s);
                out->datas = (WasmDataSeg *)calloc(
                    n ? n : 1, sizeof(WasmDataSeg));
                out->n_datas = n;
                for (uint32_t i = 0; i < n && !s.overflow; i++) {
                    uint32_t mode = (uint32_t)rd_uleb(&s);
                    uint32_t off = 0;
                    if (mode == 0) {
                        // Active, memory 0, with i32 init expr.
                        uint8_t op = rd_u8(&s);
                        if (op != 0x41) {
                            fprintf(stderr,
                                    "wasm->macho: data %u uses unsupported "
                                    "init opcode 0x%02x\n", i, op);
                            s.overflow = true; break;
                        }
                        off = (uint32_t)(int32_t)rd_sleb(&s);
                        uint8_t e = rd_u8(&s);
                        if (e != 0x0b) {
                            fprintf(stderr,
                                    "wasm->macho: data %u init expr does not "
                                    "end with 0x0b (got 0x%02x)\n", i, e);
                            s.overflow = true; break;
                        }
                    } else if (mode == 2) {
                        // Active, explicit memory index, with i32 init expr.
                        (void)rd_uleb(&s); // memidx
                        uint8_t op = rd_u8(&s);
                        if (op != 0x41) {
                            fprintf(stderr,
                                    "wasm->macho: data %u mode 2 has "
                                    "unsupported init opcode 0x%02x\n", i, op);
                            s.overflow = true; break;
                        }
                        off = (uint32_t)(int32_t)rd_sleb(&s);
                        uint8_t e = rd_u8(&s);
                        if (e != 0x0b) { s.overflow = true; break; }
                    } else {
                        fprintf(stderr,
                                "wasm->macho: data %u uses unsupported "
                                "mode %u (passive segments not supported)\n",
                                i, mode);
                        s.overflow = true; break;
                    }
                    uint32_t blen = (uint32_t)rd_uleb(&s);
                    if (s.p + blen > s.end) { s.overflow = true; break; }
                    out->datas[i].offset = off;
                    out->datas[i].bytes  = s.p;
                    out->datas[i].size   = blen;
                    s.p += blen;
                }
                break;
            }
            default: break;
        }
        r.p = sec_end;
    }
    return !r.overflow;
}

// Look up (nparams, nresults) for a function type by index.
static bool type_arity(const WasmModule *m, uint32_t type_idx,
                       uint32_t *out_np, uint32_t *out_nr) {
    if (type_idx >= m->n_types) return false;
    Rd s = { m->types_blob, m->types_blob + (1 << 30), false };
    // We don't know the blob's true end; we just walk past type_idx
    // entries assuming the input was well-formed (the WASM parser
    // already validated it).
    for (uint32_t i = 0; i <= type_idx; i++) {
        rd_u8(&s); // 0x60
        uint32_t np = (uint32_t)rd_uleb(&s);
        for (uint32_t k = 0; k < np; k++) rd_u8(&s);
        uint32_t nr = (uint32_t)rd_uleb(&s);
        if (i == type_idx) {
            *out_np = np; *out_nr = nr;
            return true;
        }
        for (uint32_t k = 0; k < nr; k++) rd_u8(&s);
    }
    return false;
}

// Variant of type_arity that also writes the param valtype bytes into
// `out_param_types[0..np)`. Caller must provide an array large enough.
static bool type_arity_params(const WasmModule *m, uint32_t type_idx,
                              uint32_t *out_np, uint32_t *out_nr,
                              uint8_t *out_param_types) {
    if (type_idx >= m->n_types) return false;
    Rd s = { m->types_blob, m->types_blob + (1 << 30), false };
    for (uint32_t i = 0; i <= type_idx; i++) {
        rd_u8(&s); // 0x60
        uint32_t np = (uint32_t)rd_uleb(&s);
        if (i == type_idx) {
            for (uint32_t k = 0; k < np; k++) out_param_types[k] = rd_u8(&s);
            uint32_t nr = (uint32_t)rd_uleb(&s);
            *out_np = np; *out_nr = nr;
            return true;
        }
        for (uint32_t k = 0; k < np; k++) rd_u8(&s);
        uint32_t nr = (uint32_t)rd_uleb(&s);
        for (uint32_t k = 0; k < nr; k++) rd_u8(&s);
    }
    return false;
}

// =============================================================================
// ARM64 instruction encoders.
// =============================================================================
static void emit_word(Buf *b, uint32_t w) { buf_le32(b, w); }
static uint32_t arm64_movz_w(uint32_t rd, uint16_t imm16) {
    return 0x52800000u | ((uint32_t)imm16 << 5) | (rd & 0x1fu);
}
static uint32_t arm64_movk_w(uint32_t rd, uint16_t imm16, uint32_t hw) {
    return 0x72800000u | ((hw & 3u) << 21) | ((uint32_t)imm16 << 5) | (rd & 0x1fu);
}
static uint32_t arm64_movz_x(uint32_t rd, uint16_t imm16, uint32_t hw) {
    return 0xd2800000u | ((hw & 3u) << 21) | ((uint32_t)imm16 << 5) | (rd & 0x1fu);
}
static uint32_t arm64_movk_x(uint32_t rd, uint16_t imm16, uint32_t hw) {
    return 0xf2800000u | ((hw & 3u) << 21) | ((uint32_t)imm16 << 5) | (rd & 0x1fu);
}
static uint32_t arm64_ret(void)             { return 0xd65f03c0u; }
static uint32_t arm64_stp_fp_lr_pre(void)   { return 0xa9bf7bfdu; } // stp x29,x30,[sp,#-16]!
static uint32_t arm64_ldp_fp_lr_post(void)  { return 0xa8c17bfdu; } // ldp x29,x30,[sp],#16
// `mov x29, sp` (encoded as `add x29, sp, #0`).
static uint32_t arm64_mov_fp_sp(void)       { return 0x910003fdu; }
// `mov sp, x29` (encoded as `add sp, x29, #0`).
static uint32_t arm64_mov_sp_fp(void)       { return 0x910003bfu; }
static uint32_t arm64_add_sp_imm(uint16_t imm12) {
    return 0x910003ffu | (((uint32_t)imm12 & 0xfffu) << 10);
}
static uint32_t arm64_sub_sp_imm(uint16_t imm12) {
    return 0xd10003ffu | (((uint32_t)imm12 & 0xfffu) << 10);
}
static uint32_t arm64_str_w_sp(uint32_t rt, uint32_t imm) {
    return 0xb90003e0u | ((imm >> 2) << 10) | (rt & 0x1fu);
}
static uint32_t arm64_ldr_w_sp(uint32_t rt, uint32_t imm) {
    return 0xb94003e0u | ((imm >> 2) << 10) | (rt & 0x1fu);
}
static uint32_t arm64_str_x_sp(uint32_t rt, uint32_t imm) {
    return 0xf90003e0u | ((imm >> 3) << 10) | (rt & 0x1fu);
}
static uint32_t arm64_ldr_x_sp(uint32_t rt, uint32_t imm) {
    return 0xf94003e0u | ((imm >> 3) << 10) | (rt & 0x1fu);
}
// STUR/LDUR <Wt|Xt>, [x29, #simm9]. simm9 is in range [-256, 255].
// Used to address locals at negative offsets from the frame pointer.
static uint32_t arm64_stur_w_fp(uint32_t rt, int32_t off) {
    return 0xb80003a0u | (((uint32_t)off & 0x1ffu) << 12) | (rt & 0x1fu);
}
static uint32_t arm64_ldur_w_fp(uint32_t rt, int32_t off) {
    return 0xb84003a0u | (((uint32_t)off & 0x1ffu) << 12) | (rt & 0x1fu);
}
static uint32_t arm64_stur_x_fp(uint32_t rt, int32_t off) {
    return 0xf80003a0u | (((uint32_t)off & 0x1ffu) << 12) | (rt & 0x1fu);
}
static uint32_t arm64_ldur_x_fp(uint32_t rt, int32_t off) {
    return 0xf84003a0u | (((uint32_t)off & 0x1ffu) << 12) | (rt & 0x1fu);
}
// add Wd, Wn, Wm   (shifted register, lsl #0).
static uint32_t arm64_add_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x0b000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// sub Wd, Wn, Wm   (shifted register, lsl #0).
static uint32_t arm64_sub_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x4b000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// add Xd, Xn, Wm, UXTW #0  (zero-extend Wm to 64, add to Xn).
static uint32_t arm64_add_x_xn_wm_uxtw(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x8b204000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// LDR Wt, [Xn, #pimm]   pimm is unsigned imm12 scaled by 4 (0..16380).
static uint32_t arm64_ldr_w_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xb9400000u | (((imm >> 2) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// STR Wt, [Xn, #pimm]   pimm scaled by 4 (0..16380).
static uint32_t arm64_str_w_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xb9000000u | (((imm >> 2) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// ADD Xd, Xn, #imm12 (LSL #0).
static uint32_t arm64_add_x_imm(uint32_t rd, uint32_t rn, uint16_t imm12) {
    return 0x91000000u | (((uint32_t)imm12 & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// ADRP Xd, #rel_pages*4096  (rel_pages is a signed 21-bit value).
static uint32_t arm64_adrp(uint32_t rd, int64_t rel_pages) {
    uint32_t immlo = (uint32_t)(rel_pages & 0x3);
    uint32_t immhi = (uint32_t)((rel_pages >> 2) & 0x7ffffu);
    return 0x90000000u | (immlo << 29) | (immhi << 5) | (rd & 0x1fu);
}
// bl <pc-relative byte offset>
static uint32_t arm64_bl(int32_t off_bytes) {
    int32_t imm26 = off_bytes >> 2;
    return 0x94000000u | ((uint32_t)imm26 & 0x03ffffffu);
}
// b <pc-relative byte offset>
static uint32_t arm64_b(int32_t off_bytes) {
    int32_t imm26 = off_bytes >> 2;
    return 0x14000000u | ((uint32_t)imm26 & 0x03ffffffu);
}

static void emit_mov_w_imm32(Buf *b, uint32_t rd, uint32_t imm32) {
    emit_word(b, arm64_movz_w(rd, (uint16_t)(imm32 & 0xffff)));
    uint16_t hi = (uint16_t)((imm32 >> 16) & 0xffff);
    if (hi != 0) emit_word(b, arm64_movk_w(rd, hi, 1));
}

// Emit MOVZ/MOVK to materialize a 64-bit constant in xd. Two's-complement
// negative values still work because we treat the value bit-wise.
static void emit_mov_x_imm64(Buf *b, uint32_t rd, uint64_t imm64) {
    emit_word(b, arm64_movz_x(rd, (uint16_t)(imm64 & 0xffffu), 0));
    for (int hw = 1; hw < 4; hw++) {
        uint16_t v = (uint16_t)((imm64 >> (hw * 16)) & 0xffffu);
        if (v != 0) emit_word(b, arm64_movk_x(rd, v, (uint32_t)hw));
    }
}

// =============================================================================
// Per-function code emission with deferred call relocations.
//
// We emit each WASM function into its own Buf. Calls become `bl`
// instructions with a placeholder imm26; we record a CallReloc with
// the call site offset and the target function index. After we lay
// out functions in __text we know each function's offset and can
// patch the imm26 to a PC-relative byte offset.
// =============================================================================
typedef struct {
    uint32_t site_off;  // offset (in bytes) within the EmittedFunc's code Buf
    uint32_t target;    // target function index in combined space, or the
                        // sentinel UINT32_MAX meaning "the _exit stub".
    uint8_t  is_b;      // 1 == use `b` (tail-call), 0 == `bl`
} CallReloc;

typedef struct {
    Buf        code;
    CallReloc *relocs;
    size_t     n_relocs, cap_relocs;
    uint32_t   text_off;  // assigned during layout
    bool       emitted;
} EmittedFunc;

static void ef_push_reloc(EmittedFunc *e, CallReloc r) {
    if (e->n_relocs == e->cap_relocs) {
        e->cap_relocs = e->cap_relocs ? e->cap_relocs * 2 : 8;
        e->relocs = (CallReloc *)realloc(e->relocs,
            e->cap_relocs * sizeof(CallReloc));
    }
    e->relocs[e->n_relocs++] = r;
}

// Advance `r` past one WASM instruction. Knows the immediate-operand
// layout of every MVP opcode plus the 0xfc-prefixed extensions
// (saturating conversions, bulk memory) so we can walk function
// bodies for analysis without needing to actually understand them.
// On unknown opcodes we still abort — but this covers everything
// the wasm32-wasi runtime / tinyc-emitted code we link against uses.
//
// Returns 0 normally, +1 if this instruction was `end` (0x0b — for
// callers that want to stop at the matching close), -1 on error.
static int wasm_skip_one(Rd *r) {
    if (r->p >= r->end) return -1;
    uint8_t op = rd_u8(r);
    switch (op) {
        // Control flow.
        case 0x00: case 0x01: return 0;            // unreachable, nop
        case 0x02: case 0x03: case 0x04:           // block, loop, if
            rd_u8(r);
            return 0;
        case 0x05: return 0;                       // else
        case 0x0b: return 1;                       // end
        case 0x0c: case 0x0d:                       // br, br_if
            (void)rd_uleb(r);
            return 0;
        case 0x0e: {                               // br_table
            uint32_t n = (uint32_t)rd_uleb(r);
            for (uint32_t i = 0; i < n; i++) (void)rd_uleb(r);
            (void)rd_uleb(r);
            return 0;
        }
        case 0x0f: return 0;                       // return
        case 0x10: (void)rd_uleb(r); return 0;     // call funcidx
        case 0x11:                                  // call_indirect
            (void)rd_uleb(r); (void)rd_uleb(r);
            return 0;

        case 0x1a: case 0x1b: return 0;            // drop, select

        // Variable instructions: local.{get,set,tee}, global.{get,set}.
        case 0x20: case 0x21: case 0x22:
        case 0x23: case 0x24:
            (void)rd_uleb(r);
            return 0;

        // Memory loads/stores 0x28..0x3e: (align uleb)(offset uleb).
        case 0x28: case 0x29: case 0x2a: case 0x2b: case 0x2c: case 0x2d:
        case 0x2e: case 0x2f: case 0x30: case 0x31: case 0x32: case 0x33:
        case 0x34: case 0x35:
        case 0x36: case 0x37: case 0x38: case 0x39:
        case 0x3a: case 0x3b: case 0x3c: case 0x3d: case 0x3e:
            (void)rd_uleb(r); (void)rd_uleb(r);
            return 0;
        case 0x3f: case 0x40: rd_u8(r); return 0;  // memory.size, memory.grow

        // Constants.
        case 0x41: (void)rd_sleb(r); return 0;     // i32.const
        case 0x42: (void)rd_sleb(r); return 0;     // i64.const
        case 0x43:                                  // f32.const
            for (int i = 0; i < 4 && r->p < r->end; i++) r->p++;
            return 0;
        case 0x44:                                  // f64.const
            for (int i = 0; i < 8 && r->p < r->end; i++) r->p++;
            return 0;

        // Prefix 0xfc — saturating conversions + bulk memory.
        case 0xfc: {
            uint32_t sub = (uint32_t)rd_uleb(r);
            switch (sub) {
                case 0: case 1: case 2: case 3:
                case 4: case 5: case 6: case 7:    // i{32,64}.trunc_sat_f{32,64}_{s,u}
                    return 0;
                case 8:                             // memory.init seg, 0x00
                    (void)rd_uleb(r); rd_u8(r);
                    return 0;
                case 9:                             // data.drop seg
                    (void)rd_uleb(r);
                    return 0;
                case 10:                            // memory.copy 0 0
                    rd_u8(r); rd_u8(r);
                    return 0;
                case 11:                            // memory.fill 0
                    rd_u8(r);
                    return 0;
                default: return -1;
            }
        }
    }
    // Everything in 0x45..0xc4 is a simple (no-immediates) opcode:
    // comparisons, int/float arithmetic, conversions, sign extends.
    if (op >= 0x45 && op <= 0xc4) return 0;
    return -1;
}

// Walk a function body once to collect every callee index. Used by
// reachability analysis to find the closure of functions
// transitively callable from `_start`. Returns false only on a
// malformed body — unknown opcodes are *tolerated* (the function
// may be unreachable; if it ends up reachable, emit_function will
// reject it later with a clearer message).
static bool scan_callees(const WasmFunc *wf, uint32_t **callees,
                         size_t *n_callees, size_t *cap_callees) {
    Rd r = { wf->code, wf->code + wf->code_size, false };
    // Skip locals decls.
    uint32_t ngroups = (uint32_t)rd_uleb(&r);
    for (uint32_t g = 0; g < ngroups; g++) {
        (void)rd_uleb(&r);
        rd_u8(&r);
    }
    while (r.p < r.end && !r.overflow) {
        // Peek the opcode so we can record `call` targets, then
        // delegate operand skipping to wasm_skip_one.
        uint8_t op = *r.p;
        if (op == 0x10) {
            r.p++;
            uint32_t idx = (uint32_t)rd_uleb(&r);
            if (*n_callees == *cap_callees) {
                *cap_callees = *cap_callees ? *cap_callees * 2 : 8;
                *callees = (uint32_t *)realloc(*callees,
                    *cap_callees * sizeof(uint32_t));
            }
            (*callees)[(*n_callees)++] = idx;
            continue;
        }
        int sr = wasm_skip_one(&r);
        if (sr < 0) {
            // Unknown opcode in a possibly-unreachable function: bail
            // out of *this* scan only. The reachable-function emitter
            // will produce a clearer error if this function turns out
            // to matter.
            return true;
        }
    }
    return !r.overflow;
}

// Emit ARM64 for one defined WASM function.
//
// Frame layout established by the prologue:
//
//   [caller's frame …]
//   [saved x29, x30]       <- x29 (fp) after prologue
//   [local 0 slot, 16 B]   <- fp - 16
//   [local 1 slot, 16 B]   <- fp - 32
//   …
//   [local N-1 slot]       <- fp - N*16
//   [value-stack slot k-1] <- top of value stack
//   …
//   [value-stack slot 0]   <- sp (initial value stack is empty,
//                              i.e. sp == fp - N*16 right after the
//                              prologue)
//
// Each local and each value-stack slot is 16 bytes wide so that sp
// stays 16-byte aligned and a slot can hold either an i32 or an i64
// without changing layout.
static bool emit_function(const WasmModule *wm, uint32_t fidx,
                          EmittedFunc *e) {
    const WasmFunc *wf = &wm->funcs[fidx];
    Rd r = { wf->code, wf->code + wf->code_size, false };

    // Parse the locals declaration: a series of (count, valtype) groups.
    // The total `n_locals` counts only the explicitly-declared locals;
    // function params occupy local slots 0..nparams-1 *before* these.
    uint32_t ngroups = (uint32_t)rd_uleb(&r);
    uint32_t n_decl_locals = 0;
    // Note the per-group valtypes so we know how to copy from
    // x0..x7 (params) into the local slab in the prologue, and how
    // to load/store i32 vs i64 values for local.{get,set,tee}.
    // We record the (count, valtype) pairs and walk them later when
    // computing each local's type.
    uint32_t local_groups[64][2];
    if (ngroups > 64) {
        fprintf(stderr,
                "wasm->macho: too many local groups (%u) in func %u\n",
                ngroups, fidx);
        return false;
    }
    for (uint32_t g = 0; g < ngroups; g++) {
        uint32_t cnt = (uint32_t)rd_uleb(&r);
        uint8_t  ty  = rd_u8(&r);
        local_groups[g][0] = cnt;
        local_groups[g][1] = ty;
        n_decl_locals += cnt;
    }

    uint32_t np_self, nr_self;
    if (!type_arity(wm, wf->type_idx, &np_self, &nr_self)) return false;
    if (np_self > 8) {
        fprintf(stderr,
                "wasm->macho: function %u has %u params (>8 not supported)\n",
                fidx, np_self);
        return false;
    }

    uint32_t n_locals_total = np_self + n_decl_locals;
    if (n_locals_total > 15) {
        // STUR/LDUR offsets are 9-bit signed: -256..255. Locals start
        // at fp-16 and decrement by 16, so we can address up to local
        // index 15 (offset -256). For more we'd need an extra base
        // register; not yet supported.
        fprintf(stderr,
                "wasm->macho: function %u uses %u locals (>15 not supported)\n",
                fidx, n_locals_total);
        return false;
    }

    // Local index -> WASM valtype byte (0x7f i32, 0x7e i64, …).
    // Params come first; their types come from the function's type
    // signature.
    uint8_t local_types[16] = {0};
    uint8_t param_types_buf[8] = {0};
    {
        uint32_t np_check, nr_check;
        if (!type_arity_params(wm, wf->type_idx, &np_check, &nr_check,
                               param_types_buf)) return false;
        for (uint32_t k = 0; k < np_self; k++)
            local_types[k] = param_types_buf[k];
    }
    {
        uint32_t li = np_self;
        for (uint32_t g = 0; g < ngroups; g++) {
            for (uint32_t k = 0; k < local_groups[g][0]; k++)
                local_types[li++] = (uint8_t)local_groups[g][1];
        }
    }

    // Prologue: save fp/lr, set fp, reserve the local slab.
    // If this is the program entry (_start), first materialize the
    // fixed bases for linear memory (x28) and globals (x27). We use
    // ADRP + ADD so the addresses are PC-relative and survive the
    // ASLR slide that dyld applies at load time. These AArch64
    // callee-saved registers are never clobbered by any emitted
    // function, so the program-entry initialization is the only
    // setup needed for the lifetime of the program.
    uint32_t fidx_combined = wm->n_imported_funcs + fidx;
    bool is_program_entry = (fidx_combined == wm->start_export_idx);
    if (is_program_entry && (wm->has_memory || wm->n_globals > 0)) {
        // page_of(__DATA) relative to page_of(_start) (both are
        // 4 KiB-aligned values, _start lies on the first page of
        // __TEXT in current layouts so the diff is just
        // (DATA_VM_BASE - TEXT_VM_BASE) / 4096 = 8).
        uint64_t pc_page     = wm->entry_vmaddr & ~0xfffULL;
        uint64_t target_page = wm->globals_vmaddr & ~0xfffULL;
        int64_t  rel_pages   = (int64_t)(target_page - pc_page) >> 12;
        uint32_t off_in_page = (uint32_t)(wm->globals_vmaddr & 0xfffULL);
        // x27 = page_of(__DATA) + off_in_page  =  globals_vmaddr.
        emit_word(&e->code, arm64_adrp(27, rel_pages));
        if (off_in_page != 0) {
            emit_word(&e->code, arm64_add_x_imm(27, 27, (uint16_t)off_in_page));
        }
        if (wm->has_memory) {
            uint32_t globals_padded = (uint32_t)(wm->linmem_vmaddr
                                                - wm->globals_vmaddr);
            if (globals_padded == 0) {
                // linmem and globals share base; x28 = x27.
                emit_word(&e->code,
                    0x91000000u | (27u << 5) | 28u);   // add x28, x27, #0
            } else {
                emit_word(&e->code,
                    arm64_add_x_imm(28, 27, (uint16_t)globals_padded));
            }
        }
    }
    emit_word(&e->code, arm64_stp_fp_lr_pre());
    emit_word(&e->code, arm64_mov_fp_sp());
    if (n_locals_total > 0) {
        emit_word(&e->code, arm64_sub_sp_imm((uint16_t)(n_locals_total * 16)));
    }

    // Copy params (in x0..x7) into local slots 0..np_self-1, and
    // zero-initialise the declared locals (WASM spec requires).
    for (uint32_t k = 0; k < np_self; k++) {
        int32_t off = -(int32_t)(k + 1) * 16;
        // Store the full 64-bit register; an i32 param's low 4 bytes
        // still occupy the slot correctly.
        emit_word(&e->code, arm64_stur_x_fp(k, off));
    }
    for (uint32_t k = 0; k < n_decl_locals; k++) {
        int32_t off = -(int32_t)(np_self + k + 1) * 16;
        emit_word(&e->code, arm64_stur_x_fp(31 /* xzr */, off));
    }

    // Helper macro: emit the function's epilogue (used by both
    // `end` falling through and explicit `return`).
    // Pops the return value into x0/w0 if any, restores sp+fp+lr, ret.
    bool ended = false;
    while (r.p < r.end && !r.overflow && !ended) {
        uint8_t op = rd_u8(&r);
        switch (op) {
            case 0x0b: ended = true; break;          // end
            case 0x0f: {                              // return
                if (nr_self == 1) {
                    emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                }
                emit_word(&e->code, arm64_mov_sp_fp());
                emit_word(&e->code, arm64_ldp_fp_lr_post());
                emit_word(&e->code, arm64_ret());
                break;
            }
            case 0x20:                                // local.get
            case 0x21:                                // local.set
            case 0x22: {                              // local.tee
                uint32_t li = (uint32_t)rd_uleb(&r);
                if (li >= n_locals_total) {
                    fprintf(stderr,
                            "wasm->macho: local %u out of range in func %u\n",
                            li, fidx);
                    return false;
                }
                int32_t off = -(int32_t)(li + 1) * 16;
                bool is_i64 = (local_types[li] == 0x7e);
                if (op == 0x20) {                      // local.get
                    if (is_i64) emit_word(&e->code, arm64_ldur_x_fp(0, off));
                    else        emit_word(&e->code, arm64_ldur_w_fp(0, off));
                    emit_word(&e->code, arm64_sub_sp_imm(16));
                    if (is_i64) emit_word(&e->code, arm64_str_x_sp(0, 0));
                    else        emit_word(&e->code, arm64_str_w_sp(0, 0));
                } else {                              // set / tee
                    if (is_i64) emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                    else        emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                    if (op == 0x21) {                  // set: pop
                        emit_word(&e->code, arm64_add_sp_imm(16));
                    }
                    if (is_i64) emit_word(&e->code, arm64_stur_x_fp(0, off));
                    else        emit_word(&e->code, arm64_stur_w_fp(0, off));
                }
                break;
            }
            case 0x41: { // i32.const sleb
                int64_t v = rd_sleb(&r);
                emit_mov_w_imm32(&e->code, 0, (uint32_t)(int32_t)v);
                emit_word(&e->code, arm64_sub_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0x42: { // i64.const sleb
                int64_t v = rd_sleb(&r);
                emit_mov_x_imm64(&e->code, 0, (uint64_t)v);
                emit_word(&e->code, arm64_sub_sp_imm(16));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }
            case 0x6a: { // i32.add
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // top   -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // below -> w0
                emit_word(&e->code, arm64_add_w_reg(0, 0, 1));
                emit_word(&e->code, arm64_add_sp_imm(16));   // pop one slot
                emit_word(&e->code, arm64_str_w_sp(0, 0));   // overwrite remaining
                break;
            }
            case 0x6b: { // i32.sub
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // top   -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // below -> w0
                emit_word(&e->code, arm64_sub_w_reg(0, 0, 1));
                emit_word(&e->code, arm64_add_sp_imm(16));   // pop one slot
                emit_word(&e->code, arm64_str_w_sp(0, 0));   // overwrite remaining
                break;
            }
            case 0x23: {                                     // global.get
                uint32_t gi = (uint32_t)rd_uleb(&r);
                if (gi >= wm->n_globals) {
                    fprintf(stderr,
                            "wasm->macho: global.get %u out of range in "
                            "func %u (n_globals=%u)\n",
                            gi, fidx, wm->n_globals);
                    return false;
                }
                bool gi64 = (wm->globals[gi].valtype == 0x7e);
                uint32_t slot_off = gi * 8;
                if (gi64) emit_word(&e->code,
                    0xf9400000u | (((slot_off >> 3) & 0xfffu) << 10)
                                | (27u << 5) | 0u);          // ldr x0, [x27, #slot]
                else      emit_word(&e->code,
                    arm64_ldr_w_xn(0, 27, slot_off));
                emit_word(&e->code, arm64_sub_sp_imm(16));
                if (gi64) emit_word(&e->code, arm64_str_x_sp(0, 0));
                else      emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0x24: {                                     // global.set
                uint32_t gi = (uint32_t)rd_uleb(&r);
                if (gi >= wm->n_globals) {
                    fprintf(stderr,
                            "wasm->macho: global.set %u out of range in "
                            "func %u (n_globals=%u)\n",
                            gi, fidx, wm->n_globals);
                    return false;
                }
                bool gi64 = (wm->globals[gi].valtype == 0x7e);
                uint32_t slot_off = gi * 8;
                if (gi64) emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                else      emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_add_sp_imm(16));
                if (gi64) emit_word(&e->code,
                    0xf9000000u | (((slot_off >> 3) & 0xfffu) << 10)
                                | (27u << 5) | 0u);          // str x0, [x27, #slot]
                else      emit_word(&e->code,
                    arm64_str_w_xn(0, 27, slot_off));
                break;
            }
            case 0x28: {                                     // i32.load
                (void)rd_uleb(&r);                            // align
                uint32_t off = (uint32_t)rd_uleb(&r);         // offset
                if (off > 16380u) {
                    fprintf(stderr,
                            "wasm->macho: i32.load offset %u too large "
                            "in func %u\n", off, fidx);
                    return false;
                }
                if (!wm->has_memory) {
                    fprintf(stderr,
                            "wasm->macho: i32.load with no memory in "
                            "func %u\n", fidx);
                    return false;
                }
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));   // pop addr -> w0
                emit_word(&e->code,
                    arm64_add_x_xn_wm_uxtw(10, 28, 0));      // x10 = x28 + (u32)w0
                emit_word(&e->code,
                    arm64_ldr_w_xn(0, 10, off));             // ldr w0, [x10, #off]
                emit_word(&e->code, arm64_str_w_sp(0, 0));   // write top of stack
                break;
            }
            case 0x36: {                                     // i32.store
                (void)rd_uleb(&r);                            // align
                uint32_t off = (uint32_t)rd_uleb(&r);         // offset
                if (off > 16380u) {
                    fprintf(stderr,
                            "wasm->macho: i32.store offset %u too large "
                            "in func %u\n", off, fidx);
                    return false;
                }
                if (!wm->has_memory) {
                    fprintf(stderr,
                            "wasm->macho: i32.store with no memory in "
                            "func %u\n", fidx);
                    return false;
                }
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // value -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // addr  -> w0
                emit_word(&e->code, arm64_add_sp_imm(32));
                emit_word(&e->code,
                    arm64_add_x_xn_wm_uxtw(10, 28, 0));      // x10 = x28 + (u32)addr
                emit_word(&e->code,
                    arm64_str_w_xn(1, 10, off));             // str w1, [x10, #off]
                break;
            }
            case 0x10: { // call funcidx
                uint32_t cidx = (uint32_t)rd_uleb(&r);
                uint32_t np = 0, nr = 0;
                if (cidx < wm->n_imported_funcs) {
                    if (cidx == wm->import_idx_proc_exit) { np = 1; nr = 0; }
                    else {
                        fprintf(stderr,
                                "wasm->macho: unsupported import call (idx %u)\n",
                                cidx);
                        return false;
                    }
                } else {
                    uint32_t cfidx = cidx - wm->n_imported_funcs;
                    if (cfidx >= wm->n_funcs) {
                        fprintf(stderr,
                                "wasm->macho: call out of range: %u\n", cidx);
                        return false;
                    }
                    if (!type_arity(wm, wm->funcs[cfidx].type_idx, &np, &nr))
                        return false;
                }
                if (np > 8) {
                    fprintf(stderr,
                            "wasm->macho: call with %u params not supported\n",
                            np);
                    return false;
                }
                // Pop args from CPU stack into x(np-1)..x0 (top of WASM
                // stack -> highest-numbered arg). We pop the full
                // 64-bit slot so i64 args are handled correctly.
                for (int k = (int)np - 1; k >= 0; k--) {
                    emit_word(&e->code, arm64_ldr_x_sp((uint32_t)k, 0));
                    emit_word(&e->code, arm64_add_sp_imm(16));
                }
                // Place a `bl` placeholder.
                CallReloc rel = {
                    .site_off = (uint32_t)e->code.len,
                    .target = cidx,
                    .is_b = 0,
                };
                ef_push_reloc(e, rel);
                emit_word(&e->code, arm64_bl(0));
                if (nr == 1) {
                    emit_word(&e->code, arm64_sub_sp_imm(16));
                    emit_word(&e->code, arm64_str_x_sp(0, 0));
                } else if (nr > 1) {
                    fprintf(stderr,
                            "wasm->macho: call with %u results not supported\n",
                            nr);
                    return false;
                }
                break;
            }
            default:
                fprintf(stderr,
                        "wasm->macho: unsupported opcode 0x%02x in func %u\n",
                        op, fidx);
                return false;
        }
    }

    // Trailing fall-through epilogue.
    if (nr_self == 1) {
        emit_word(&e->code, arm64_ldr_x_sp(0, 0));
    } else if (nr_self > 1) return false;
    emit_word(&e->code, arm64_mov_sp_fp());
    emit_word(&e->code, arm64_ldp_fp_lr_post());
    emit_word(&e->code, arm64_ret());

    return !r.overflow;
}

// Emit the proc_exit shim: tail-call _exit (which never returns).
//   b _exit_stub
static void emit_proc_exit_shim(EmittedFunc *e) {
    CallReloc rel = {
        .site_off = (uint32_t)e->code.len,
        .target = UINT32_MAX,   // _exit stub
        .is_b = 1,
    };
    ef_push_reloc(e, rel);
    emit_word(&e->code, arm64_b(0));
}

// Patch a single bl/b at site_off so that PC-relative addressing
// lands on absolute __text offset `target_text_off`.
static void patch_branch(EmittedFunc *e, uint32_t target_text_off,
                         CallReloc *cr) {
    int32_t from = (int32_t)e->text_off + (int32_t)cr->site_off;
    int32_t off  = (int32_t)target_text_off - from;
    uint32_t imm26 = ((uint32_t)(off >> 2)) & 0x03ffffffu;
    uint32_t op_base = cr->is_b ? 0x14000000u : 0x94000000u;
    uint32_t enc = op_base | imm26;
    e->code.data[cr->site_off + 0] = (uint8_t)(enc & 0xff);
    e->code.data[cr->site_off + 1] = (uint8_t)((enc >> 8) & 0xff);
    e->code.data[cr->site_off + 2] = (uint8_t)((enc >> 16) & 0xff);
    e->code.data[cr->site_off + 3] = (uint8_t)((enc >> 24) & 0xff);
}

// =============================================================================
// Mach-O constants we use.
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
// Same flags word as write_macho.cpp (decimal 2097285).
#define MH_FLAGS_EXEC          2097285u

#define VM_PROT_READ           1u
#define VM_PROT_WRITE          2u
#define VM_PROT_EXECUTE        4u

#define TEXT_VM_BASE           0x100000000ULL
#define DATA_CONST_VM_BASE     0x100004000ULL
#define DATA_VM_BASE           0x100008000ULL
#define TEXT_FILE_BASE         0u
#define DATA_CONST_FILE_BASE   16384u
#define DATA_FILE_BASE         32768u
#define VMSEG_SIZE             0x4000u  // 16 KB per non-__DATA segment

// =============================================================================
// The main translator.
// =============================================================================
bool MLIR_WasmToMachoArm64(const uint8_t *wasm_bytes, size_t wasm_size,
                           uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;

    WasmModule wm;
    if (!wasm_parse(wasm_bytes, wasm_size, &wm)) return false;
    if (wm.start_export_idx == UINT32_MAX) {
        fprintf(stderr, "wasm->macho: input has no `_start` export\n");
        wasm_module_free(&wm); return false;
    }
    if (wm.import_idx_proc_exit == UINT32_MAX) {
        fprintf(stderr,
                "wasm->macho: input does not import wasi proc_exit\n");
        wasm_module_free(&wm); return false;
    }

    // -----------------------------------------------------------------
    // Compute layout of the optional __DATA segment that holds the
    // wasm linear memory and wasm globals. The segment is omitted
    // entirely when the module declares neither memory nor globals
    // (the `macho_exit` case), which keeps the simpler binaries small.
    // Layout inside __DATA: [globals (n_globals * 8B)] [linear memory].
    // globals are 8B-aligned so the i64 case works without extra
    // padding; linmem starts at the next 16-byte boundary.
    // -----------------------------------------------------------------
    uint64_t globals_bytes = (uint64_t)wm.n_globals * 8ULL;
    uint64_t globals_padded = (globals_bytes + 15ULL) & ~15ULL;
    uint64_t linmem_size = 0;
    if (wm.has_memory) {
        linmem_size = (uint64_t)wm.memory_min_pages * 65536ULL;
        for (uint32_t i = 0; i < wm.n_datas; i++) {
            uint64_t end_off = (uint64_t)wm.datas[i].offset
                             + (uint64_t)wm.datas[i].size;
            if (end_off > linmem_size) linmem_size = end_off;
        }
    }
    uint64_t data_seg_payload = globals_padded + linmem_size;
    uint64_t data_seg_vmsize  = (data_seg_payload + 0x3fffULL) & ~0x3fffULL;
    bool     has_data_seg     = (data_seg_vmsize > 0);
    if (data_seg_vmsize > 0xffffffffULL) {
        fprintf(stderr,
                "wasm->macho: linear memory + globals too large (%llu)\n",
                (unsigned long long)data_seg_vmsize);
        wasm_module_free(&wm); return false;
    }
    uint32_t data_seg_size32 = (uint32_t)data_seg_vmsize;

    wm.globals_vmaddr = DATA_VM_BASE;
    wm.linmem_vmaddr  = DATA_VM_BASE + globals_padded;

    uint64_t linkedit_vm_base   = DATA_VM_BASE + (has_data_seg ? data_seg_vmsize : 0);
    uint32_t linkedit_file_base = DATA_FILE_BASE + (has_data_seg ? data_seg_size32 : 0);

    // ncmds / sizeofcmds. Adding __DATA contributes one LC_SEGMENT_64
    // of size 72 (segment header) + 80 (one section header) = 152.
    uint32_t base_ncmds      = 17;
    uint32_t base_sizeofcmds = 976;
    uint32_t n_cmds      = base_ncmds + (has_data_seg ? 1u : 0u);
    uint32_t sizeofcmds  = base_sizeofcmds + (has_data_seg ? 152u : 0u);

    // text_section_off must come after the mach header + LCs.
    // We pad to 16 bytes (and never below the original 1040 so
    // macho_exit's layout stays unchanged).
    uint32_t text_section_off = (32u + sizeofcmds + 15u) & ~15u;
    if (text_section_off < 1040u) text_section_off = 1040u;

    wm.entry_vmaddr = TEXT_VM_BASE + text_section_off;

    // -----------------------------------------------------------------
    // Reachability analysis: BFS from _start over `call` edges. Only
    // functions in the reachable set get translated; the rest stay
    // out of the output. (wasm-ld may have dragged in tens of
    // unrelated runtime functions; many use opcodes we don't yet
    // support but they're never invoked from main, so skipping them
    // is safe.)
    // -----------------------------------------------------------------
    uint32_t n_total = wm.n_imported_funcs + wm.n_funcs;
    bool *reachable = (bool *)calloc(n_total, sizeof(bool));
    uint32_t *worklist = (uint32_t *)malloc(sizeof(uint32_t) * n_total);
    size_t   wlen = 0;
    reachable[wm.start_export_idx] = true;
    worklist[wlen++] = wm.start_export_idx;
    while (wlen) {
        uint32_t f = worklist[--wlen];
        if (f < wm.n_imported_funcs) continue;        // imports have no body
        const WasmFunc *wf = &wm.funcs[f - wm.n_imported_funcs];
        uint32_t *callees = NULL; size_t nc = 0, cc = 0;
        if (!scan_callees(wf, &callees, &nc, &cc)) {
            free(callees); free(worklist); free(reachable);
            wasm_module_free(&wm);
            return false;
        }
        for (size_t k = 0; k < nc; k++) {
            uint32_t c = callees[k];
            if (c >= n_total) continue;
            if (!reachable[c]) {
                reachable[c] = true;
                worklist[wlen++] = c;
            }
        }
        free(callees);
    }
    free(worklist);

    // -----------------------------------------------------------------
    // Emit per-function bytes for: every reachable import shim + every
    // reachable defined function. We index by combined func space.
    // -----------------------------------------------------------------
    EmittedFunc *efs = (EmittedFunc *)calloc(n_total, sizeof(EmittedFunc));

    if (reachable[wm.import_idx_proc_exit]) {
        emit_proc_exit_shim(&efs[wm.import_idx_proc_exit]);
        efs[wm.import_idx_proc_exit].emitted = true;
    }

    for (uint32_t i = 0; i < wm.n_funcs; i++) {
        uint32_t fidx_combined = wm.n_imported_funcs + i;
        if (!reachable[fidx_combined]) continue;
        EmittedFunc *e = &efs[fidx_combined];
        if (!emit_function(&wm, i, e)) {
            for (uint32_t j = 0; j < n_total; j++) {
                free(efs[j].code.data); free(efs[j].relocs);
            }
            free(efs); free(reachable);
            wasm_module_free(&wm);
            return false;
        }
        e->emitted = true;
    }
    free(reachable);

    // -----------------------------------------------------------------
    // Layout: __text contains [start_export] then every other
    // emitted function in index order. We place `_start` first so the
    // LC_MAIN entry_offset equals TEXT_SECTION_OFF.
    // -----------------------------------------------------------------
    uint32_t cursor = 0;
    // Place start export first.
    {
        EmittedFunc *e = &efs[wm.start_export_idx];
        e->text_off = cursor;
        cursor += (uint32_t)e->code.len;
    }
    for (uint32_t i = 0; i < n_total; i++) {
        if (i == wm.start_export_idx) continue;
        if (!efs[i].emitted) continue;
        efs[i].text_off = cursor;
        cursor += (uint32_t)efs[i].code.len;
    }
    uint32_t text_size = cursor;
    // Pad to a multiple of 4 (already true; instructions are word-aligned)
    // and to a multiple of 4 for the stubs section start (it's
    // align=2^2=4).
    while (text_size % 4) text_size++;

    // __stubs: one entry per libSystem import. For now just `_exit`.
    const uint32_t n_stubs = 1;
    const uint32_t stub_size = 12;
    const uint32_t stubs_off  = text_section_off + text_size;
    const uint32_t stubs_size = n_stubs * stub_size;

    // __cstring: empty for macho_exit.
    const uint32_t cstring_off  = stubs_off + stubs_size;
    const uint32_t cstring_size = 0;

    // GOT in __DATA_CONST.
    const uint32_t got_size = n_stubs * 8;

    // -----------------------------------------------------------------
    // Patch every emitted function's call relocations.
    // -----------------------------------------------------------------
    // Address of the _exit stub (within file = its offset within __text
    // = stubs_off - TEXT_SECTION_OFF, treated as if part of __text for
    // the PC-relative math — but stubs follows __text so this is just
    // text_size + (stub_idx * stub_size).
    uint32_t exit_stub_text_off = text_size + 0 * stub_size;
    for (uint32_t i = 0; i < n_total; i++) {
        if (!efs[i].emitted) continue;
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            CallReloc *cr = &efs[i].relocs[k];
            uint32_t target_off = (cr->target == UINT32_MAX)
                                  ? exit_stub_text_off
                                  : efs[cr->target].text_off;
            patch_branch(&efs[i], target_off, cr);
        }
    }

    // -----------------------------------------------------------------
    // Build the Mach-O image.
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
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, TEXT_FILE_BASE);
    buf_le64(&img, VMSEG_SIZE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_EXECUTE);
    buf_le32(&img, 3);
    buf_le32(&img, 0);
    {
        // __text
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
        buf_le32(&img, 0);   // struct trailing pad (reserved3 in ABI)
    }
    {
        // __stubs
        static const char SN[16] = "__stubs";
        static const char SG[16] = "__TEXT";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, TEXT_VM_BASE + stubs_off);
        buf_le64(&img, (uint64_t)stubs_size);
        buf_le32(&img, stubs_off);
        buf_le32(&img, 2);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 0x80000408u);
        buf_le32(&img, 0);
        buf_le32(&img, stub_size);
        buf_le32(&img, 0);   // struct trailing pad
    }
    {
        // __cstring (empty, but keep it so the section count matches
        // the reference layout).
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
        buf_le32(&img, 0);   // struct trailing pad
    }

    // LC_SEGMENT_64 __DATA_CONST (1 section: __got)
    buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 152);
    { static const char SEG[16] = "__DATA_CONST"; buf_append(&img, SEG, 16); }
    buf_le64(&img, DATA_CONST_VM_BASE);
    buf_le64(&img, VMSEG_SIZE);
    buf_le64(&img, DATA_CONST_FILE_BASE);
    buf_le64(&img, VMSEG_SIZE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
    buf_le32(&img, 1);
    buf_le32(&img, 16);                  // SG_READ_ONLY
    {
        static const char SN[16] = "__got";
        static const char SG[16] = "__DATA_CONST";
        buf_append(&img, SN, 16); buf_append(&img, SG, 16);
        buf_le64(&img, DATA_CONST_VM_BASE);
        buf_le64(&img, (uint64_t)got_size);
        buf_le32(&img, DATA_CONST_FILE_BASE);
        buf_le32(&img, 3);
        buf_le32(&img, 0); buf_le32(&img, 0);
        buf_le32(&img, 6);                // S_NON_LAZY_SYMBOL_POINTERS
        buf_le32(&img, 0);                // reserved1 = start indirect-sym idx
        buf_le32(&img, 0);
        buf_le32(&img, 0);                // struct trailing pad
    }

    // LC_SEGMENT_64 __DATA (optional, with one section holding both
    // wasm globals and wasm linear memory).
    if (has_data_seg) {
        buf_le32(&img, LC_SEGMENT_64); buf_le32(&img, 152);
        { static const char SEG[16] = "__DATA"; buf_append(&img, SEG, 16); }
        buf_le64(&img, DATA_VM_BASE);
        buf_le64(&img, data_seg_vmsize);
        buf_le64(&img, DATA_FILE_BASE);
        buf_le64(&img, data_seg_vmsize);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, 1);                // nsects
        buf_le32(&img, 0);                // flags
        {
            static const char SN[16] = "__data";
            static const char SG[16] = "__DATA";
            buf_append(&img, SN, 16); buf_append(&img, SG, 16);
            buf_le64(&img, DATA_VM_BASE);
            buf_le64(&img, (uint64_t)data_seg_payload);
            buf_le32(&img, DATA_FILE_BASE);
            buf_le32(&img, 3);            // align = 2^3 = 8
            buf_le32(&img, 0); buf_le32(&img, 0);
            buf_le32(&img, 0);            // S_REGULAR
            buf_le32(&img, 0);
            buf_le32(&img, 0);
            buf_le32(&img, 0);
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
    size_t pos_linkedit_filesize = pos_linkedit_seg + 8 /*cmd+cmdsize*/ +
                                   16 /*segname*/ + 8 /*vmaddr*/ +
                                   8 /*vmsize*/ + 8 /*fileoff*/;

    // Linkedit data load commands — fill in offsets/sizes after we
    // emit the linkedit blob.
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

    // LC_LOAD_DYLINKER /usr/lib/dyld
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
    buf_le32(&img, 1);                  // platform = macOS
    buf_le32(&img, 0x000f0700);          // minos 15.7
    buf_le32(&img, 0);                  // sdk
    buf_le32(&img, 1);                  // ntools
    buf_le32(&img, 3);                  // tool = ld
    buf_le32(&img, 0x04ce0100);

    // LC_SOURCE_VERSION
    buf_le32(&img, LC_SOURCE_VERSION); buf_le32(&img, 16);
    buf_le64(&img, 0);

    // LC_MAIN (entryoff = TEXT_SECTION_OFF since _start is first in
    // __text).
    buf_le32(&img, LC_MAIN); buf_le32(&img, 24);
    buf_le64(&img, (uint64_t)text_section_off);
    buf_le64(&img, 0);

    // LC_LOAD_DYLIB /usr/lib/libSystem.B.dylib
    buf_le32(&img, LC_LOAD_DYLIB); buf_le32(&img, 56);
    buf_le32(&img, 24);                  // name offset
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

    // Pad header+LCs to TEXT_SECTION_OFF.
    buf_pad_to(&img, text_section_off);

    // __text
    {
        EmittedFunc *e = &efs[wm.start_export_idx];
        buf_append(&img, e->code.data, e->code.len);
    }
    for (uint32_t i = 0; i < n_total; i++) {
        if (i == wm.start_export_idx) continue;
        if (!efs[i].emitted) continue;
        buf_append(&img, efs[i].code.data, efs[i].code.len);
    }
    buf_pad_to(&img, stubs_off);

    // __stubs: one entry per libSystem import. Stub 0 is _exit.
    for (uint32_t i = 0; i < n_stubs; i++) {
        uint64_t got_target = DATA_CONST_VM_BASE + 8ULL * i;
        uint64_t stub_addr  = TEXT_VM_BASE + stubs_off + (uint64_t)i * stub_size;
        uint64_t page_dst   = got_target & ~0xfffULL;
        uint64_t page_src   = stub_addr  & ~0xfffULL;
        int64_t  page_diff  = (int64_t)(page_dst - page_src);
        int64_t  page_imm   = page_diff >> 12;
        uint32_t immlo = (uint32_t)(page_imm & 0x3);
        uint32_t immhi = (uint32_t)((page_imm >> 2) & 0x7ffff);
        uint32_t adrp  = 0x90000010u | (immlo << 29) | (immhi << 5);
        uint32_t lo12  = (uint32_t)(got_target & 0xfffu);
        uint32_t ldr   = 0xf9400210u | (((lo12 >> 3) & 0xfffu) << 10);
        uint32_t br    = 0xd61f0200u;
        emit_word(&img, adrp);
        emit_word(&img, ldr);
        emit_word(&img, br);
    }
    // __cstring is empty for macho_exit.

    // Pad to __DATA_CONST.
    buf_pad_to(&img, DATA_CONST_FILE_BASE);
    // __got: one chained-fixup-format pointer per import. The format
    // is DYLD_CHAINED_PTR_64_OFFSET (pointer_format=6). For a single
    // import:
    //   bit63=1 (is_bind), next=0 (end of chain), ordinal=0.
    // The "ordinal=0" indexes the imports table where we place
    // _exit at index 0.
    buf_le64(&img, 0x8000000000000000ULL);

    // __DATA segment payload (optional): [globals bytes][linmem bytes].
    if (has_data_seg) {
        buf_pad_to(&img, DATA_FILE_BASE);
        size_t data_seg_start = img.len;
        for (uint32_t i = 0; i < wm.n_globals; i++) {
            // Each global stored as 8 bytes (zero-extended for i32).
            buf_le64(&img, (uint64_t)wm.globals[i].init_value);
        }
        // Pad globals region to globals_padded.
        while ((img.len - data_seg_start) < globals_padded) buf_u8(&img, 0);
        // Linear memory: zeros first, then overlay each data segment
        // at its declared offset.
        size_t linmem_start = img.len;
        while ((img.len - linmem_start) < linmem_size) buf_u8(&img, 0);
        for (uint32_t i = 0; i < wm.n_datas; i++) {
            const WasmDataSeg *d = &wm.datas[i];
            size_t dst = linmem_start + (size_t)d->offset;
            if (dst + d->size > img.len) {
                fprintf(stderr,
                        "wasm->macho: data segment %u extends past "
                        "allocated linear memory\n", i);
                free(reachable); /* fall through after manual cleanup */
                // Best-effort cleanup before returning.
                for (uint32_t j = 0; j < n_total; j++) {
                    free(efs[j].code.data); free(efs[j].relocs);
                }
                free(efs);
                free(img.data);
                wasm_module_free(&wm);
                return false;
            }
            memcpy(img.data + dst, d->bytes, d->size);
        }
    }

    // Pad to __LINKEDIT.
    buf_pad_to(&img, linkedit_file_base);

    // =====================================================================
    // __LINKEDIT contents
    // =====================================================================
    size_t linkedit_start = img.len;

    // ---- chained fixups blob ----
    size_t chained_start = img.len;
    // Layout inside the blob (offsets are from chained_start):
    //   0x00            header (32 B)
    //   0x20            starts_in_image (4 + N*4 B) + optional 4 B align pad
    //   0x38            starts_in_segment for __DATA_CONST (24 B)
    //   0x50            imports table (4 B per import)
    //   imports + 4*N   symbols (leading 0 + cstrs + pad to 8 B)
    //
    // dyld_chained_starts_in_segment starts at offset 0x38 in BOTH cases
    // (seg_count=4 needs a 4 B align pad after the offsets table,
    //  seg_count=5 already lands on an 8 B boundary without padding).
    uint32_t fx_imports_off = 0x50u;
    uint32_t fx_symbols_off = fx_imports_off + n_stubs * 4u;
    buf_le32(&img, 0);                    // fixups_version
    buf_le32(&img, 0x20);                  // starts_offset
    buf_le32(&img, fx_imports_off);        // imports_offset
    buf_le32(&img, fx_symbols_off);        // symbols_offset
    buf_le32(&img, n_stubs);               // imports_count
    buf_le32(&img, 1);                    // imports_format
    buf_le32(&img, 0);                    // symbols_format
    buf_le32(&img, 0);                    // pad to starts_offset (0x20)

    // dyld_chained_starts_in_image. The seg_info_offset for
    // __DATA_CONST is always 0x18 (= 24): for seg_count=4 we have
    // 20 B of struct + 4 B of trailing align pad to reach an 8-byte
    // boundary; for seg_count=5 we have 24 B of struct and no pad.
    uint32_t fx_seg_count = has_data_seg ? 5u : 4u;
    buf_le32(&img, fx_seg_count);          // seg_count
    buf_le32(&img, 0);                    // __PAGEZERO
    buf_le32(&img, 0);                    // __TEXT
    buf_le32(&img, 0x18);                  // __DATA_CONST: offset to starts_in_segment
    if (has_data_seg) buf_le32(&img, 0);  // __DATA (no fixups in linmem)
    buf_le32(&img, 0);                    // __LINKEDIT
    if (!has_data_seg) buf_le32(&img, 0); // align pad to 8 B (only when seg_count=4)

    // dyld_chained_starts_in_segment (for __DATA_CONST)
    buf_le32(&img, 0x18);                  // size
    buf_le16(&img, 0x4000);                // page_size
    buf_le16(&img, 6);                    // pointer_format (DYLD_CHAINED_PTR_64_OFFSET)
    buf_le64(&img, 0x4000);                // segment_offset
    buf_le32(&img, 0);                    // max_valid_pointer
    buf_le16(&img, 1);                    // page_count
    buf_le16(&img, 0);                    // page_start[0]

    // imports table: one dyld_chained_import per import.
    //   { lib_ordinal: 8, weak_import: 1, name_offset: 23 }
    // ordinal=1 (the only LC_LOAD_DYLIB, libSystem). Name "_exit"
    // sits at offset 1 in the symbols table (after the leading 0).
    {
        uint32_t entry = 1u | (0u << 8) | (1u << 9);
        buf_le32(&img, entry);
    }
    // symbols table: leading 0 + cstrs + trailing 0s for alignment.
    buf_u8(&img, 0);
    buf_cstr(&img, "_exit");
    buf_u8(&img, 0);
    buf_u8(&img, 0);
    while ((img.len - chained_start) % 8) buf_u8(&img, 0);
    uint32_t chained_size = (uint32_t)(img.len - chained_start);

    // ---- exports trie ----
    //
    // Mirrors the layout that `ld` emits for two exports
    // (__mh_execute_header at 0, _main at LC_MAIN entry offset).
    // The byte sequence is taken verbatim from
    // test_wasm/write_macho.cpp's build_exports_trie_blob().
    size_t exports_start = img.len;
    buf_u8(&img, 0x00); buf_u8(&img, 0x01);
    buf_cstr(&img, "_");
    buf_uleb(&img, 0x12);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x00); buf_u8(&img, 0x02);
    buf_u8(&img, 0x00); buf_u8(&img, 0x00); buf_u8(&img, 0x00);

    buf_u8(&img, 0x03); buf_u8(&img, 0x00);
    buf_uleb(&img, (uint64_t)text_section_off);   // _main entry offset
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
    //
    // We use a layout close to the reference but smaller:
    //   locals: 0
    //   extdefs: 2  (__mh_execute_header, _main)
    //   undefs:  1  (_exit)
    //
    // String table layout:
    //   [0]   = ' '   (preserves the reference's leading byte)
    //   [1]   = 0
    //   [2..] = "__mh_execute_header\0_main\0_exit\0" + padding
    //
    // The leading-byte choice doesn't matter to dyld, but matching
    // the reference makes diffing easier.
    Buf strtab = {0};
    buf_u8(&strtab, 0x20);
    buf_u8(&strtab, 0x00);
    uint32_t str_mh   = (uint32_t)strtab.len; buf_cstr(&strtab, "__mh_execute_header");
    uint32_t str_main = (uint32_t)strtab.len; buf_cstr(&strtab, "_main");
    uint32_t str_exit = (uint32_t)strtab.len; buf_cstr(&strtab, "_exit");
    while (strtab.len % 8) buf_u8(&strtab, 0);

    Buf symtab = {0};
    // __mh_execute_header (n_type=0x0f N_SECT|N_EXT, n_sect=1, n_desc=REF_DYNAMICALLY)
    buf_le32(&symtab, str_mh);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0010);
    buf_le64(&symtab, TEXT_VM_BASE);
    // _main
    buf_le32(&symtab, str_main);
    buf_u8(&symtab, 0x0f); buf_u8(&symtab, 1);
    buf_le16(&symtab, 0x0000);
    buf_le64(&symtab, TEXT_VM_BASE + text_section_off);
    // _exit (undefined, dynamic-lookup)
    buf_le32(&symtab, str_exit);
    buf_u8(&symtab, 0x01); buf_u8(&symtab, 0);
    buf_le16(&symtab, 0x0100);
    buf_le64(&symtab, 0);

    // Indirect symbols: one per __got entry and one per __stubs entry.
    // The reference puts indirect_syms BEFORE strtab; we'll do the
    // same. Layout: __got indirect (1 entry), then __stubs indirect
    // (1 entry). Indirect-sym value = index into symtab of the
    // undefined symbol that backs that slot. Our symtab is:
    //   0: __mh_execute_header
    //   1: _main
    //   2: _exit
    // so both indirects reference index 2.
    Buf indsyms = {0};
    buf_le32(&indsyms, 2);   // __got[0]
    buf_le32(&indsyms, 2);   // __stubs[0]

    // Emit symtab, indsyms, strtab in that order (matches the
    // reference's offset ordering and the dysymtab indirectsymoff
    // we'll fill in).
    size_t symtab_off = img.len;
    buf_append(&img, symtab.data, symtab.len);
    size_t indsyms_off = img.len;
    buf_append(&img, indsyms.data, indsyms.len);
    size_t strtab_off = img.len;
    buf_append(&img, strtab.data, strtab.len);

    // Pad to a 16-byte boundary before the code signature.
    while ((img.len - linkedit_start) % 16) buf_u8(&img, 0);

    // ---- code signature ----
    //
    // Important: every byte we hash here must be in its final form.
    // That includes the LC_CODE_SIGNATURE and __LINKEDIT-filesize
    // fields in the header (page 0). So we compute the signature
    // *size* first (which only depends on code_limit + identifier
    // length), patch every load command including LC_CODE_SIGNATURE
    // and __LINKEDIT-filesize, *then* hash the pages, and finally
    // append the signature blob.
    uint32_t code_limit = (uint32_t)img.len;
    size_t code_sig_off = img.len;

    const uint32_t page_size  = 4096;
    const uint32_t page_shift = 12;
    const char *ident = "tinyc.out";
    uint32_t ident_len = (uint32_t)strlen(ident);
    const uint32_t n_slots = (code_limit + page_size - 1) / page_size;
    const uint32_t ident_offset = 88;
    const uint32_t hash_offset = ident_offset + ident_len + 1;
    const uint32_t cd_len = hash_offset + n_slots * SHA256_DIGEST_LEN;
    const uint32_t sb_len_unpadded = 20 + cd_len;
    const uint32_t code_sig_size = (sb_len_unpadded + 15u) & ~15u;

    // Patch all linkedit-data LCs (including LC_CODE_SIGNATURE) so
    // the header is final before we hash.
    buf_patch_le32(&img, pos_lc_chained_fixups   + 8,  (uint32_t)chained_start);
    buf_patch_le32(&img, pos_lc_chained_fixups   + 12, chained_size);
    buf_patch_le32(&img, pos_lc_exports_trie     + 8,  (uint32_t)exports_start);
    buf_patch_le32(&img, pos_lc_exports_trie     + 12, exports_size);
    buf_patch_le32(&img, pos_lc_symtab           + 8,  (uint32_t)symtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 12, 3);                       // nsyms
    buf_patch_le32(&img, pos_lc_symtab           + 16, (uint32_t)strtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 20, (uint32_t)strtab.len);
    buf_patch_le32(&img, pos_lc_dysymtab         + 8,  0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 12, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 16, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 20, 2);
    buf_patch_le32(&img, pos_lc_dysymtab         + 24, 2);
    buf_patch_le32(&img, pos_lc_dysymtab         + 28, 1);
    buf_patch_le32(&img, pos_lc_dysymtab         + 32, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 36, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 40, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 44, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 48, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 52, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 56, (uint32_t)indsyms_off);
    buf_patch_le32(&img, pos_lc_dysymtab         + 60, 2);
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

    // Patch __LINKEDIT segment filesize (data + code signature).
    buf_patch_le64(&img, pos_linkedit_filesize,
                   (uint64_t)(code_limit + code_sig_size) - linkedit_file_base);

    free(symtab.data); free(indsyms.data); free(strtab.data);

    // Now build and append the code signature blob, hashing pages of
    // the now-final image bytes [0, code_limit).
    Buf cs = {0};
    {
        Buf cd = {0};
        buf_be32(&cd, 0xfade0c02);
        buf_be32(&cd, cd_len);
        buf_be32(&cd, 0x00020400);
        buf_be32(&cd, 0x00020002);
        buf_be32(&cd, hash_offset);
        buf_be32(&cd, ident_offset);
        buf_be32(&cd, 0);
        buf_be32(&cd, n_slots);
        buf_be32(&cd, code_limit);
        buf_u8(&cd, SHA256_DIGEST_LEN);
        buf_u8(&cd, 2);
        buf_u8(&cd, 0);
        buf_u8(&cd, page_shift);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        buf_be32(&cd, 0);
        while (cd.len < 76) buf_u8(&cd, 0);
        buf_be32(&cd, 0x1c);
        buf_be32(&cd, 0x00);
        buf_be32(&cd, 0x01);
        while (cd.len < ident_offset) buf_u8(&cd, 0);
        buf_append(&cd, ident, ident_len);
        buf_u8(&cd, 0);
        for (uint32_t i = 0; i < n_slots; i++) {
            uint32_t start  = i * page_size;
            uint32_t remain = code_limit - start;
            uint32_t len    = remain < page_size ? remain : page_size;
            uint8_t  d[SHA256_DIGEST_LEN];
            sha256(img.data + start, len, d);
            buf_append(&cd, d, sizeof(d));
        }

        buf_be32(&cs, 0xfade0cc0);
        buf_be32(&cs, (uint32_t)(20 + cd.len));
        buf_be32(&cs, 1);
        buf_be32(&cs, 0);
        buf_be32(&cs, 20);
        buf_append(&cs, cd.data, cd.len);
        free(cd.data);
    }
    while (cs.len < code_sig_size) buf_u8(&cs, 0);
    buf_append(&img, cs.data, cs.len);
    free(cs.data);

    // Cleanup.
    for (uint32_t i = 0; i < n_total; i++) {
        free(efs[i].code.data);
        free(efs[i].relocs);
    }
    free(efs);
    wasm_module_free(&wm);

    *out_data = img.data;
    *out_size = img.len;
    return true;
}

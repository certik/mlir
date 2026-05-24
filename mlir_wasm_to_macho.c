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
    uint32_t       import_idx_fd_write;

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
    out->import_idx_fd_write  = UINT32_MAX;
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
                        } else if (mod_len == 22 &&
                            memcmp(mod_p, "wasi_snapshot_preview1", 22) == 0 &&
                            nm_len == 8 && memcmp(nm_p, "fd_write", 8) == 0) {
                            out->import_idx_fd_write = fi;
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
// `out_param_types[0..np)` and the result valtype bytes into
// `out_result_types[0..nr)`. Either array pointer may be NULL.
static bool type_arity_params(const WasmModule *m, uint32_t type_idx,
                              uint32_t *out_np, uint32_t *out_nr,
                              uint8_t *out_param_types,
                              uint8_t *out_result_types) {
    if (type_idx >= m->n_types) return false;
    Rd s = { m->types_blob, m->types_blob + (1 << 30), false };
    for (uint32_t i = 0; i <= type_idx; i++) {
        rd_u8(&s); // 0x60
        uint32_t np = (uint32_t)rd_uleb(&s);
        if (i == type_idx) {
            for (uint32_t k = 0; k < np; k++) {
                uint8_t v = rd_u8(&s);
                if (out_param_types) out_param_types[k] = v;
            }
            uint32_t nr = (uint32_t)rd_uleb(&s);
            for (uint32_t k = 0; k < nr; k++) {
                uint8_t v = rd_u8(&s);
                if (out_result_types) out_result_types[k] = v;
            }
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
// LDR Xt, [Xn|SP, #pimm]   pimm scaled by 8 (0..32760).
static uint32_t arm64_ldr_x_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xf9400000u | (((imm >> 3) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// STR Xt, [Xn|SP, #pimm]   pimm scaled by 8 (0..32760).
static uint32_t arm64_str_x_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xf9000000u | (((imm >> 3) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// CMP Wn, Wm   alias for SUBS WZR, Wn, Wm.
static uint32_t arm64_cmp_w(uint32_t rn, uint32_t rm) {
    return 0x6b00001fu | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5);
}
// CSET Wd, <cond>   alias for CSINC Wd, WZR, WZR, invert(<cond>). The
// caller passes the *inverted* condition code (e.g. EQ=0 to make CSET
// produce 1 when NE).
static uint32_t arm64_cset_w(uint32_t rd, uint32_t cond_inv) {
    return 0x1a800400u | (31u << 16) | ((cond_inv & 0xfu) << 12)
         | (31u << 5) | (rd & 0x1fu);
}
// CBZ Wt, <pc-relative byte offset>. Encodes a 19-bit signed
// instruction-count offset.
static uint32_t arm64_cbz_w(uint32_t rt, int32_t off_bytes) {
    int32_t imm19 = off_bytes >> 2;
    return 0x34000000u | (((uint32_t)imm19 & 0x7ffffu) << 5) | (rt & 0x1fu);
}
// CSEL Wd, Wn, Wm, cond. If cond holds, Wd = Wn else Wd = Wm.
static uint32_t arm64_csel_w(uint32_t rd, uint32_t rn, uint32_t rm,
                             uint32_t cond) {
    return 0x1a800000u | ((rm & 0x1fu) << 16) | ((cond & 0xfu) << 12)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// CMP Wn, #0   alias for SUBS WZR, Wn, #0 (32-bit sub immediate).
static uint32_t arm64_cmp_w_imm0(uint32_t rn) {
    return 0x7100001fu | ((rn & 0x1fu) << 5);
}
// MOV Xd, SP   alias for ADD Xd, SP, #0 (only valid when Rd != 31).
static uint32_t arm64_mov_x_sp(uint32_t rd) {
    return 0x91000000u | (31u << 5) | (rd & 0x1fu);
}
// ADD Xd, Xn, #imm12 (LSL #0).
static uint32_t arm64_add_x_imm(uint32_t rd, uint32_t rn, uint16_t imm12) {
    return 0x91000000u | (((uint32_t)imm12 & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// ADD Xd, Xn, #imm12, LSL #12.
static uint32_t arm64_add_x_imm_lsl12(uint32_t rd, uint32_t rn,
                                      uint16_t imm12) {
    return 0x91400000u | (((uint32_t)imm12 & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// ADRP Xd, #rel_pages*4096  (rel_pages is a signed 21-bit value).
static uint32_t arm64_adrp(uint32_t rd, int64_t rel_pages) {
    uint32_t immlo = (uint32_t)(rel_pages & 0x3);
    uint32_t immhi = (uint32_t)((rel_pages >> 2) & 0x7ffffu);
    return 0x90000000u | (immlo << 29) | (immhi << 5) | (rd & 0x1fu);
}
// MOV Wd, Wm   alias for ORR Wd, WZR, Wm.
static uint32_t arm64_mov_w_reg(uint32_t rd, uint32_t rm) {
    return 0x2a000000u | ((rm & 0x1fu) << 16) | (31u << 5) | (rd & 0x1fu);
}
// ADD Wd, Wn, #imm12 (LSL #0).
static uint32_t arm64_add_w_imm(uint32_t rd, uint32_t rn, uint16_t imm12) {
    return 0x11000000u | (((uint32_t)imm12 & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// SUB Wd, Wn, #imm12 (LSL #0).
static uint32_t arm64_sub_w_imm(uint32_t rd, uint32_t rn, uint16_t imm12) {
    return 0x51000000u | (((uint32_t)imm12 & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// ADD Xd, Xn, Xm (LSL #0).
static uint32_t arm64_add_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x8b000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// SUB Xd, Xn, Xm (LSL #0).
static uint32_t arm64_sub_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0xcb000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// CMP Xn, Xm   alias for SUBS XZR, Xn, Xm.
static uint32_t arm64_cmp_x(uint32_t rn, uint32_t rm) {
    return 0xeb00001fu | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5);
}
// CMP Xn, #0   alias for SUBS XZR, Xn, #0.
static uint32_t arm64_cmp_x_imm0(uint32_t rn) {
    return 0xf100001fu | ((rn & 0x1fu) << 5);
}
// ORR Wd, Wn, Wm (LSL #0).
static uint32_t arm64_orr_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x2a000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// ORR Xd, Xn, Xm (LSL #0).
static uint32_t arm64_orr_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0xaa000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// EOR Wd, Wn, Wm (LSL #0).
static uint32_t arm64_eor_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x4a000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// EOR Xd, Xn, Xm (LSL #0).
static uint32_t arm64_eor_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0xca000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// AND Wd, Wn, Wm (LSL #0).
static uint32_t arm64_and_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x0a000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// AND Xd, Xn, Xm (LSL #0).
static uint32_t arm64_and_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x8a000000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// MUL Wd, Wn, Wm   alias for MADD Wd, Wn, Wm, WZR.
static uint32_t arm64_mul_w(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1b007c00u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// MUL Xd, Xn, Xm.
static uint32_t arm64_mul_x(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x9b007c00u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// UDIV Wd, Wn, Wm.
static uint32_t arm64_udiv_w(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1ac00800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// UDIV Xd, Xn, Xm.
static uint32_t arm64_udiv_x(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x9ac00800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// SDIV Wd, Wn, Wm.
static uint32_t arm64_sdiv_w(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1ac00c00u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// SDIV Xd, Xn, Xm.
static uint32_t arm64_sdiv_x(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x9ac00c00u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// LSL Wd, Wn, Wm (variable shift, alias of LSLV).
static uint32_t arm64_lsl_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1ac02000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// LSL Xd, Xn, Xm.
static uint32_t arm64_lsl_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x9ac02000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// LSR Wd, Wn, Wm (variable, LSRV).
static uint32_t arm64_lsr_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1ac02400u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// LSR Xd, Xn, Xm.
static uint32_t arm64_lsr_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x9ac02400u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// ASR Wd, Wn, Wm (variable, ASRV).
static uint32_t arm64_asr_w_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1ac02800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// ASR Xd, Xn, Xm.
static uint32_t arm64_asr_x_reg(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x9ac02800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
// SXTW Xd, Wn   alias for SBFM Xd, Xn, #0, #31.
static uint32_t arm64_sxtw(uint32_t rd, uint32_t rn) {
    return 0x93407c00u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// LDRB Wt, [Xn, #imm12]  (unsigned offset, scaled by 1).
static uint32_t arm64_ldrb_w_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0x39400000u | ((imm12 & 0xfffu) << 10) | ((rn & 0x1fu) << 5)
         | (rt & 0x1fu);
}
// STRB Wt, [Xn, #imm12]  (unsigned offset, scaled by 1).
static uint32_t arm64_strb_w_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0x39000000u | ((imm12 & 0xfffu) << 10) | ((rn & 0x1fu) << 5)
         | (rt & 0x1fu);
}
// STRH Wt, [Xn, #imm12]  (unsigned offset, scaled by 2 — imm12 is in
// halfword units, i.e. byte offset / 2).
static uint32_t arm64_strh_w_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0x79000000u | ((imm12 & 0xfffu) << 10) | ((rn & 0x1fu) << 5)
         | (rt & 0x1fu);
}
// LDRSB Wt, [Xn, #imm12]  (unsigned offset, scaled by 1, sign-extended
// into Wt).
static uint32_t arm64_ldrsb_w_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0x39c00000u | ((imm12 & 0xfffu) << 10) | ((rn & 0x1fu) << 5)
         | (rt & 0x1fu);
}
// LDRH Wt, [Xn, #imm12]  (unsigned offset, scaled by 2).
static uint32_t arm64_ldrh_w_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0x79400000u | ((imm12 & 0xfffu) << 10) | ((rn & 0x1fu) << 5)
         | (rt & 0x1fu);
}
// LDRSH Wt, [Xn, #imm12]  (unsigned offset, scaled by 2,
// sign-extended into Wt).
static uint32_t arm64_ldrsh_w_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0x79c00000u | ((imm12 & 0xfffu) << 10) | ((rn & 0x1fu) << 5)
         | (rt & 0x1fu);
}
// LDRSW Xt, [Xn, #imm12]  (unsigned offset, scaled by 4,
// sign-extended into Xt).
static uint32_t arm64_ldrsw_x_xn(uint32_t rt, uint32_t rn, uint32_t imm12) {
    return 0xb9800000u | (((imm12 >> 2) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// SXTB Wd, Wn   alias for SBFM Wd, Wn, #0, #7.
static uint32_t arm64_sxtb_w(uint32_t rd, uint32_t rn) {
    return 0x13001c00u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// SXTH Wd, Wn   alias for SBFM Wd, Wn, #0, #15.
static uint32_t arm64_sxth_w(uint32_t rd, uint32_t rn) {
    return 0x13003c00u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// SXTB Xd, Wn   alias for SBFM Xd, Xn, #0, #7 (with sf=1, N=1).
static uint32_t arm64_sxtb_x(uint32_t rd, uint32_t rn) {
    return 0x93401c00u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// SXTH Xd, Wn   alias for SBFM Xd, Xn, #0, #15.
static uint32_t arm64_sxth_x(uint32_t rd, uint32_t rn) {
    return 0x93403c00u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// MSUB Wd, Wn, Wm, Wa.
static uint32_t arm64_msub_w(uint32_t rd, uint32_t rn, uint32_t rm,
                             uint32_t ra) {
    return 0x1b008000u | ((rm & 0x1fu) << 16) | ((ra & 0x1fu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// MSUB Xd, Xn, Xm, Xa.
static uint32_t arm64_msub_x(uint32_t rd, uint32_t rn, uint32_t rm,
                             uint32_t ra) {
    return 0x9b008000u | ((rm & 0x1fu) << 16) | ((ra & 0x1fu) << 10)
         | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// BRK #imm16 — break (raises trap on macOS).
static uint32_t arm64_brk(uint16_t imm16) {
    return 0xd4200000u | ((uint32_t)imm16 << 5);
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

// ---- ARM64 FP/SIMD scalar encoders ---------------------------------------
// All operate on the lower lane of V0..V31 (Dn = double, Sn = float).

// LDR Dt, [Xn|SP, #pimm]  pimm scaled by 8.
static uint32_t arm64_ldr_d_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xfd400000u | (((imm >> 3) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// STR Dt, [Xn|SP, #pimm].
static uint32_t arm64_str_d_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xfd000000u | (((imm >> 3) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// LDR St, [Xn|SP, #pimm]  pimm scaled by 4.
static uint32_t arm64_ldr_s_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xbd400000u | (((imm >> 2) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
// STR St, [Xn|SP, #pimm].
static uint32_t arm64_str_s_xn(uint32_t rt, uint32_t rn, uint32_t imm) {
    return 0xbd000000u | (((imm >> 2) & 0xfffu) << 10)
         | ((rn & 0x1fu) << 5) | (rt & 0x1fu);
}
static uint32_t arm64_ldr_d_sp(uint32_t rt, uint32_t imm) {
    return arm64_ldr_d_xn(rt, 31, imm);
}
static uint32_t arm64_str_d_sp(uint32_t rt, uint32_t imm) {
    return arm64_str_d_xn(rt, 31, imm);
}
static uint32_t arm64_ldr_s_sp(uint32_t rt, uint32_t imm) {
    return arm64_ldr_s_xn(rt, 31, imm);
}
static uint32_t arm64_str_s_sp(uint32_t rt, uint32_t imm) {
    return arm64_str_s_xn(rt, 31, imm);
}
// FMOV Dd, Xn — move 64-bit GPR into D-reg, bit-preserving.
static uint32_t arm64_fmov_d_x(uint32_t rd, uint32_t rn) {
    return 0x9e670000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// FMOV Sd, Wn — move 32-bit GPR into S-reg, bit-preserving.
static uint32_t arm64_fmov_s_w(uint32_t rd, uint32_t rn) {
    return 0x1e270000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// FMOV Xd, Dn — bit-preserving reverse (used by reinterpret).
static uint32_t arm64_fmov_x_d(uint32_t rd, uint32_t rn) {
    return 0x9e660000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
// FMOV Wd, Sn — bit-preserving reverse for 32-bit.
static uint32_t arm64_fmov_w_s(uint32_t rd, uint32_t rn) {
    return 0x1e260000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}

// Binary scalar FP arithmetic (Dd = Dn OP Dm or Sd = Sn OP Sm).
//   FADD d/s, FSUB d/s, FMUL d/s, FDIV d/s.
static uint32_t arm64_fadd_d(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e602800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fsub_d(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e603800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fmul_d(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e600800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fdiv_d(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e601800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fadd_s(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e202800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fsub_s(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e203800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fmul_s(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e200800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}
static uint32_t arm64_fdiv_s(uint32_t rd, uint32_t rn, uint32_t rm) {
    return 0x1e201800u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5)
         | (rd & 0x1fu);
}

// Unary scalar FP (FABS, FNEG, FSQRT).
static uint32_t arm64_fabs_d(uint32_t rd, uint32_t rn) {
    return 0x1e60c000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fneg_d(uint32_t rd, uint32_t rn) {
    return 0x1e614000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fsqrt_d(uint32_t rd, uint32_t rn) {
    return 0x1e61c000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fabs_s(uint32_t rd, uint32_t rn) {
    return 0x1e20c000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fneg_s(uint32_t rd, uint32_t rn) {
    return 0x1e214000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fsqrt_s(uint32_t rd, uint32_t rn) {
    return 0x1e21c000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}

// FCMP Dn, Dm   /  FCMP Sn, Sm.
static uint32_t arm64_fcmp_d(uint32_t rn, uint32_t rm) {
    return 0x1e602000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5);
}
static uint32_t arm64_fcmp_s(uint32_t rn, uint32_t rm) {
    return 0x1e202000u | ((rm & 0x1fu) << 16) | ((rn & 0x1fu) << 5);
}

// FP↔int conversions. All use round-toward-zero (FCVTZS/FCVTZU) for
// truncation, which matches the wasm trunc_* semantics.
static uint32_t arm64_fcvtzs_w_d(uint32_t rd, uint32_t rn) {
    return 0x1e780000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzs_x_d(uint32_t rd, uint32_t rn) {
    return 0x9e780000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzs_w_s(uint32_t rd, uint32_t rn) {
    return 0x1e380000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzs_x_s(uint32_t rd, uint32_t rn) {
    return 0x9e380000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzu_w_d(uint32_t rd, uint32_t rn) {
    return 0x1e790000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzu_x_d(uint32_t rd, uint32_t rn) {
    return 0x9e790000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzu_w_s(uint32_t rd, uint32_t rn) {
    return 0x1e390000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvtzu_x_s(uint32_t rd, uint32_t rn) {
    return 0x9e390000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_scvtf_d_w(uint32_t rd, uint32_t rn) {
    return 0x1e620000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_scvtf_s_w(uint32_t rd, uint32_t rn) {
    return 0x1e220000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_scvtf_d_x(uint32_t rd, uint32_t rn) {
    return 0x9e620000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_scvtf_s_x(uint32_t rd, uint32_t rn) {
    return 0x9e220000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_ucvtf_d_w(uint32_t rd, uint32_t rn) {
    return 0x1e630000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_ucvtf_s_w(uint32_t rd, uint32_t rn) {
    return 0x1e230000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_ucvtf_d_x(uint32_t rd, uint32_t rn) {
    return 0x9e630000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_ucvtf_s_x(uint32_t rd, uint32_t rn) {
    return 0x9e230000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}

// FCVT: precision change. FCVT Sd, Dn  (demote d→s);  FCVT Dd, Sn (promote).
static uint32_t arm64_fcvt_s_d(uint32_t rd, uint32_t rn) {
    return 0x1e624000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}
static uint32_t arm64_fcvt_d_s(uint32_t rd, uint32_t rn) {
    return 0x1e22c000u | ((rn & 0x1fu) << 5) | (rd & 0x1fu);
}

// Materialize the effective address of a wasm load/store entirely into
// x10. Assumes the popped wasm address (i32) is in w0. The caller then
// uses #0 as the immediate offset on the LDR/STR.
//
// Handles offsets up to 16 MB (12-bit high + 12-bit low) by emitting one
// or two ADD-imm instructions. Tinyc-emitted code keeps offsets well
// below this — globals live in pages 1+ of linear memory (≤ a few MB).
//
// Returns false if `off` exceeds 16 MB; the caller surfaces a clear
// diagnostic in that case.
static bool emit_compute_mem_addr(Buf *buf, uint32_t off) {
    // x10 = x28 + (uint32)w0  (linmem base + wasm addr).
    emit_word(buf, arm64_add_x_xn_wm_uxtw(10, 28, 0));
    if (off == 0) return true;
    if (off > 0xffffffu) return false;
    uint32_t hi = (off >> 12) & 0xfffu;
    uint32_t lo = off & 0xfffu;
    if (hi != 0)
        emit_word(buf, arm64_add_x_imm_lsl12(10, 10, (uint16_t)hi));
    if (lo != 0)
        emit_word(buf, arm64_add_x_imm(10, 10, (uint16_t)lo));
    return true;
}

// Patch a CBZ Wt at site_off so the imm19 field encodes
// (target_off - site_off) in instructions.
static void patch_cbz_local(uint8_t *code_data, uint32_t site_off,
                            uint32_t target_off) {
    int32_t off_bytes = (int32_t)target_off - (int32_t)site_off;
    int32_t imm19 = off_bytes >> 2;
    uint8_t *p = code_data + site_off;
    uint32_t enc = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8)
                 | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    enc &= ~((uint32_t)0x7ffffu << 5);
    enc |= ((uint32_t)imm19 & 0x7ffffu) << 5;
    p[0] = (uint8_t)(enc & 0xff);
    p[1] = (uint8_t)((enc >> 8) & 0xff);
    p[2] = (uint8_t)((enc >> 16) & 0xff);
    p[3] = (uint8_t)((enc >> 24) & 0xff);
}
// Patch an unconditional B at site_off so its imm26 field encodes
// (target_off - site_off) in instructions.
static void patch_b_local(uint8_t *code_data, uint32_t site_off,
                          uint32_t target_off) {
    int32_t off_bytes = (int32_t)target_off - (int32_t)site_off;
    int32_t imm26 = off_bytes >> 2;
    uint8_t *p = code_data + site_off;
    uint32_t enc = 0x14000000u | ((uint32_t)imm26 & 0x03ffffffu);
    p[0] = (uint8_t)(enc & 0xff);
    p[1] = (uint8_t)((enc >> 8) & 0xff);
    p[2] = (uint8_t)((enc >> 16) & 0xff);
    p[3] = (uint8_t)((enc >> 24) & 0xff);
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
//
// `target` values:
//   < n_total       — wasm-space function index (resolves to efs[target]).
//   STUB_EXIT       — call the `_exit` libSystem stub (stub index 0).
//   STUB_WRITE      — call the `_write` libSystem stub (stub index 1).
// =============================================================================
#define STUB_EXIT   UINT32_MAX
#define STUB_WRITE  (UINT32_MAX - 1u)

typedef struct {
    uint32_t site_off;  // offset (in bytes) within the EmittedFunc's code Buf
    uint32_t target;    // wasm funcidx OR STUB_EXIT / STUB_WRITE sentinel
    uint8_t  is_b;      // 1 == use `b` (tail-call), 0 == `bl`
} CallReloc;

typedef struct {
    Buf        code;
    CallReloc *relocs;
    size_t     n_relocs, cap_relocs;
    uint32_t   text_off;  // assigned during layout
    uint32_t   adrp_off;  // byte offset (within code) of program-entry
                          // ADRP x27, page_of(globals_vmaddr); or
                          // UINT32_MAX if this function is not _start
                          // or the module has no globals/memory.
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
    // With x25-relative LDR/STR (unsigned imm12 scaled by 4 for w,
    // by 8 for x), we can index up to (4095*4) / 16 ≈ 1023 i32 locals,
    // or roughly twice that for i64. Cap conservatively below the i32
    // limit; the cap leaves room for the saved-x25 slot at the bottom
    // of the slab.
    if (n_locals_total > 1022) {
        fprintf(stderr,
                "wasm->macho: function %u uses %u locals (>1022 "
                "not supported)\n", fidx, n_locals_total);
        return false;
    }

    // Local index -> WASM valtype byte (0x7f i32, 0x7e i64, …).
    // Params come first; their types come from the function's type
    // signature.
    uint8_t *local_types = (uint8_t *)calloc(
        n_locals_total ? n_locals_total : 1, 1);
    uint8_t param_types_buf[8] = {0};
    uint8_t result_type_buf[1] = {0};
    {
        uint32_t np_check, nr_check;
        if (!type_arity_params(wm, wf->type_idx, &np_check, &nr_check,
                               param_types_buf, result_type_buf)) {
            free(local_types); return false;
        }
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
        // page_of(__DATA) relative to page_of(_start). Both targets
        // are 4 KiB-aligned. The actual addresses are finalised after
        // all functions are emitted (we need the exact text size to
        // size __TEXT and place __DATA_CONST / __DATA), so we just
        // record the offset of the ADRP instruction here and patch
        // its immediate in a post-pass.
        uint64_t pc_page     = wm->entry_vmaddr & ~0xfffULL;
        uint64_t target_page = wm->globals_vmaddr & ~0xfffULL;
        int64_t  rel_pages   = (int64_t)(target_page - pc_page) >> 12;
        uint32_t off_in_page = (uint32_t)(wm->globals_vmaddr & 0xfffULL);
        e->adrp_off = (uint32_t)e->code.len;
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
    // Reserve (n_locals_total + 1) 16-byte slots when we have any
    // locals: N for the locals themselves plus one for the caller's
    // x25, which we spill before claiming x25 as our locals base.
    // The saved x25 lives at the lowest address (== sp post-prologue
    // == new x25 + 0), and local i lives at new x25 + (N - i) * 16
    // so it grows upward in offset as i decreases — matching the
    // existing fp - (i+1)*16 layout. For functions with no locals
    // we skip the x25 dance entirely.
    if (n_locals_total > 0) {
        uint32_t reserved_slots = n_locals_total + 1;
        emit_word(&e->code, arm64_sub_sp_imm(
                (uint16_t)(reserved_slots * 16)));
        // str x25, [sp, #0]   (save caller's x25 at lowest slot)
        emit_word(&e->code, arm64_str_x_xn(25, 31 /* SP */, 0));
        // mov x25, sp
        emit_word(&e->code, arm64_mov_x_sp(25));
    }

    // Copy params (received in x0..x7 / d0..d7 via the ARM64 PCS) into
    // local slots 0..np_self-1, and zero-initialise the declared locals
    // (WASM spec requires).  Local i lives at [x25, #(n_locals_total - i) * 16].
    // The PCS uses *independent* sequences for GPR and FPR args, so we
    // walk left-to-right tracking each.
    {
        uint32_t gpr_idx = 0, fpr_idx = 0;
        for (uint32_t k = 0; k < np_self; k++) {
            uint32_t off = (n_locals_total - k) * 16;
            uint8_t t = param_types_buf[k];
            if (t == 0x7c) // f64
                emit_word(&e->code, arm64_str_d_xn(fpr_idx++, 25, off));
            else if (t == 0x7d) // f32
                emit_word(&e->code, arm64_str_s_xn(fpr_idx++, 25, off));
            else // i32/i64 — store the full 64-bit register
                emit_word(&e->code, arm64_str_x_xn(gpr_idx++, 25, off));
        }
    }
    for (uint32_t k = 0; k < n_decl_locals; k++) {
        uint32_t li = np_self + k;
        uint32_t off = (n_locals_total - li) * 16;
        emit_word(&e->code, arm64_str_x_xn(31 /* xzr */, 25, off));
    }

    // Block frame stack — tracks structured control-flow nesting.
    // Three kinds:
    //   0 = block (forward branch target == after matching `end`)
    //   1 = if    (also a forward target; pops a cond, may have else)
    //   2 = loop  (backward branch target == loop_header_off)
    // Each frame remembers the value-stack depth (in 16-byte slots)
    // at entry so that `br N` can drop the right number of
    // intermediate values before branching.  For block/if we also
    // accumulate a list of branch fixups that we patch when the
    // matching `end` is reached.
    typedef struct {
        uint8_t  kind;
        uint8_t  has_else;         // for if
        uint8_t  arity;            // result arity (only 0 supported today)
        uint32_t entry_depth;      // value-stack depth (slots) at frame entry
        uint32_t cond_branch_site; // for if: CBZ over the `then` body
        uint32_t else_jump_site;   // for if-with-else: B over the `else` body
        uint32_t loop_header_off;  // for loop: code offset to branch back to
        uint32_t fixups[32];       // for block/if: br/br_if B sites to patch
        uint32_t n_fixups;
    } BlockFrame;
    BlockFrame block_stack[64];
    uint32_t block_depth = 0;
    // Running value-stack depth (in 16-byte slots) since the start of
    // this function's body.  Each opcode handler updates this in
    // lock-step with the SP adjustments it emits, so `br`/`br_if` can
    // compute how many slots to deallocate before branching.
    uint32_t value_depth = 0;

    // Helper to emit the epilogue (used by both explicit `return` and
    // the fall-through end-of-function). When the function has locals
    // we restore the caller's x25 from the bottom of our local slab.
    #define EMIT_EPILOGUE() do {                                          \
        if (nr_self == 1) {                                               \
            if (result_type_buf[0] == 0x7c)                               \
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));                \
            else if (result_type_buf[0] == 0x7d)                          \
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));                \
            else                                                          \
                emit_word(&e->code, arm64_ldr_x_sp(0, 0));                \
        }                                                                 \
        if (n_locals_total > 0)                                           \
            emit_word(&e->code, arm64_ldr_x_xn(25, 25, 0));               \
        emit_word(&e->code, arm64_mov_sp_fp());                           \
        emit_word(&e->code, arm64_ldp_fp_lr_post());                      \
        emit_word(&e->code, arm64_ret());                                 \
    } while (0)

    bool ended = false;
    while (r.p < r.end && !r.overflow && !ended) {
        uint8_t op = rd_u8(&r);
        switch (op) {
            case 0x0b: {                              // end
                if (block_depth == 0) {
                    ended = true;
                    break;
                }
                BlockFrame *frame = &block_stack[--block_depth];
                uint32_t here = (uint32_t)e->code.len;
                if (frame->kind == 1) {
                    // if/else: patch the CBZ (or the B at end-of-then)
                    // to land here.
                    if (frame->has_else) {
                        patch_b_local(e->code.data,
                                      frame->else_jump_site, here);
                    } else {
                        patch_cbz_local(e->code.data,
                                        frame->cond_branch_site, here);
                    }
                } else if (frame->kind == 0) {
                    // block: patch every recorded br/br_if branch
                    // site to land just past `end`.
                    for (uint32_t fi = 0; fi < frame->n_fixups; fi++) {
                        patch_b_local(e->code.data,
                                      frame->fixups[fi], here);
                    }
                }
                // loop frame: br targets a backward edge already
                // patched at the branch site; nothing to do here.
                value_depth = frame->entry_depth + frame->arity;
                break;
            }
            case 0x02:                                // block blocktype
            case 0x03: {                              // loop  blocktype
                uint8_t bt = rd_u8(&r);
                if (bt != 0x40) {
                    fprintf(stderr,
                            "wasm->macho: only void block-type (0x40) "
                            "supported (got 0x%02x) in func %u\n",
                            bt, fidx);
                    free(local_types); return false;
                }
                if (block_depth >= 64) {
                    fprintf(stderr,
                            "wasm->macho: block depth >64 in func %u\n",
                            fidx);
                    free(local_types); return false;
                }
                BlockFrame *frame = &block_stack[block_depth++];
                frame->kind = (op == 0x02) ? 0 : 2;
                frame->has_else = 0;
                frame->arity = 0;
                frame->entry_depth = value_depth;
                frame->cond_branch_site = 0;
                frame->else_jump_site = 0;
                frame->loop_header_off = (uint32_t)e->code.len;
                frame->n_fixups = 0;
                break;
            }
            case 0x04: {                              // if blocktype
                uint8_t bt = rd_u8(&r);
                if (bt != 0x40) {
                    fprintf(stderr,
                            "wasm->macho: only void if-type (0x40) "
                            "supported (got 0x%02x) in func %u\n",
                            bt, fidx);
                    free(local_types); return false;
                }
                // Pop the condition off the value stack into w0.
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_add_sp_imm(16));
                value_depth -= 1;
                // CBZ placeholder: branch to `else` (or `end` if no
                // `else`), patched when we get there.
                if (block_depth >= 64) {
                    fprintf(stderr,
                            "wasm->macho: block depth >64 in func %u\n",
                            fidx);
                    free(local_types); return false;
                }
                uint32_t site = (uint32_t)e->code.len;
                emit_word(&e->code, arm64_cbz_w(0, 0));
                BlockFrame *frame = &block_stack[block_depth++];
                frame->kind = 1;
                frame->has_else = 0;
                frame->arity = 0;
                frame->entry_depth = value_depth;
                frame->cond_branch_site = site;
                frame->else_jump_site = 0;
                frame->loop_header_off = 0;
                frame->n_fixups = 0;
                break;
            }
            case 0x05: {                              // else
                if (block_depth == 0
                    || block_stack[block_depth-1].kind != 1) {
                    fprintf(stderr,
                            "wasm->macho: stray else in func %u\n", fidx);
                    free(local_types); return false;
                }
                BlockFrame *frame = &block_stack[block_depth-1];
                // Emit an unconditional jump that skips the `else`
                // body once the `then` body finishes executing.
                uint32_t jump_site = (uint32_t)e->code.len;
                emit_word(&e->code, arm64_b(0));
                // Patch the CBZ to land here (start of `else` body).
                patch_cbz_local(e->code.data, frame->cond_branch_site,
                                (uint32_t)e->code.len);
                frame->has_else = 1;
                frame->else_jump_site = jump_site;
                // Reset depth to entry for the else body.
                value_depth = frame->entry_depth;
                break;
            }
            case 0x0c:                                // br labelidx
            case 0x0d: {                              // br_if labelidx
                uint32_t labelidx = (uint32_t)rd_uleb(&r);
                if (labelidx >= block_depth) {
                    fprintf(stderr,
                            "wasm->macho: br label %u out of range "
                            "(depth=%u) in func %u\n",
                            labelidx, block_depth, fidx);
                    free(local_types); return false;
                }
                BlockFrame *target =
                    &block_stack[block_depth - 1 - labelidx];
                // Branch arity is the loop's *input* arity (only 0 is
                // supported today), or the block's result arity.
                uint32_t branch_arity = target->arity;
                if (op == 0x0d) {
                    // Pop the condition into w0 first.
                    emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                    emit_word(&e->code, arm64_add_sp_imm(16));
                    value_depth -= 1;
                }
                int32_t drop = (int32_t)value_depth
                             - (int32_t)target->entry_depth
                             - (int32_t)branch_arity;
                if (drop < 0) {
                    fprintf(stderr,
                            "wasm->macho: br N=%u underflow drop=%d "
                            "in func %u (depth=%u target_entry=%u)\n",
                            labelidx, drop, fidx,
                            value_depth, target->entry_depth);
                    free(local_types); return false;
                }
                uint32_t skip_site = 0;
                if (op == 0x0d) {
                    skip_site = (uint32_t)e->code.len;
                    // CBZ over the {drop, b} bypass: if cond==0,
                    // skip the branch and fall through.
                    emit_word(&e->code, arm64_cbz_w(0, 0));
                }
                if (drop > 0) {
                    uint32_t bytes = (uint32_t)drop * 16u;
                    if (bytes > 4095) {
                        fprintf(stderr,
                                "wasm->macho: br drop %d slots exceeds "
                                "12-bit imm in func %u\n", drop, fidx);
                        free(local_types); return false;
                    }
                    emit_word(&e->code,
                        arm64_add_sp_imm((uint16_t)bytes));
                }
                uint32_t b_site = (uint32_t)e->code.len;
                if (target->kind == 2) {
                    // Backward branch to loop header (concrete).
                    int32_t off = (int32_t)target->loop_header_off
                                - (int32_t)b_site;
                    emit_word(&e->code, arm64_b(off));
                } else {
                    // Forward branch to matching end; record fixup.
                    if (target->n_fixups >= 32) {
                        fprintf(stderr,
                                "wasm->macho: too many br fixups in "
                                "func %u\n", fidx);
                        free(local_types); return false;
                    }
                    target->fixups[target->n_fixups++] = b_site;
                    emit_word(&e->code, arm64_b(0));
                }
                if (op == 0x0d) {
                    patch_cbz_local(e->code.data, skip_site,
                                    (uint32_t)e->code.len);
                }
                // For unconditional br, subsequent code until the
                // matching `end` is unreachable; we still emit it but
                // value_depth tracking will resync at `end`.
                break;
            }
            case 0x0f: {                              // return
                EMIT_EPILOGUE();
                break;
            }
            case 0x1a: {                              // drop
                emit_word(&e->code, arm64_add_sp_imm(16));
                value_depth -= 1;
                break;
            }
            case 0x1b: {                              // select
                // [val_true, val_false, cond] on the stack (cond is
                // top). Result is val_true when cond != 0, else
                // val_false. We pop cond and val_false, then
                // overwrite the (now-top) val_true slot.
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));   // cond -> w0
                emit_word(&e->code, arm64_ldr_w_sp(1, 16));  // false -> w1
                emit_word(&e->code, arm64_ldr_w_sp(2, 32));  // true  -> w2
                emit_word(&e->code, arm64_add_sp_imm(32));   // pop 2 slots
                emit_word(&e->code, arm64_cmp_w_imm0(0));
                emit_word(&e->code, arm64_csel_w(2, 2, 1, 1)); // NE
                emit_word(&e->code, arm64_str_w_sp(2, 0));
                value_depth -= 2;
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
                    free(local_types); return false;
                }
                uint32_t off = (n_locals_total - li) * 16;
                bool is_64 = (local_types[li] == 0x7e) // i64
                          || (local_types[li] == 0x7c); // f64
                if (op == 0x20) {                      // local.get
                    if (is_64) emit_word(&e->code, arm64_ldr_x_xn(0, 25, off));
                    else       emit_word(&e->code, arm64_ldr_w_xn(0, 25, off));
                    emit_word(&e->code, arm64_sub_sp_imm(16));
                    if (is_64) emit_word(&e->code, arm64_str_x_sp(0, 0));
                    else       emit_word(&e->code, arm64_str_w_sp(0, 0));
                    value_depth += 1;
                } else {                              // set / tee
                    if (is_64) emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                    else       emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                    if (op == 0x21) {                  // set: pop
                        emit_word(&e->code, arm64_add_sp_imm(16));
                        value_depth -= 1;
                    }
                    if (is_64) emit_word(&e->code, arm64_str_x_xn(0, 25, off));
                    else       emit_word(&e->code, arm64_str_w_xn(0, 25, off));
                }
                break;
            }
            case 0x41: { // i32.const sleb
                int64_t v = rd_sleb(&r);
                emit_mov_w_imm32(&e->code, 0, (uint32_t)(int32_t)v);
                emit_word(&e->code, arm64_sub_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth += 1;
                break;
            }
            case 0x42: { // i64.const sleb
                int64_t v = rd_sleb(&r);
                emit_mov_x_imm64(&e->code, 0, (uint64_t)v);
                emit_word(&e->code, arm64_sub_sp_imm(16));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                value_depth += 1;
                break;
            }
            case 0x45: { // i32.eqz
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_cmp_w_imm0(0));
                emit_word(&e->code, arm64_cset_w(0, 1));      // cset w0, eq
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0x48: { // i32.lt_s
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // top   -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // below -> w0
                emit_word(&e->code, arm64_cmp_w(0, 1));
                emit_word(&e->code, arm64_cset_w(0, 0xa));    // cset w0, lt
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }
            case 0x6a: { // i32.add
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // top   -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // below -> w0
                emit_word(&e->code, arm64_add_w_reg(0, 0, 1));
                emit_word(&e->code, arm64_add_sp_imm(16));   // pop one slot
                emit_word(&e->code, arm64_str_w_sp(0, 0));   // overwrite remaining
                value_depth -= 1;
                break;
            }
            case 0x6b: { // i32.sub
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // top   -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // below -> w0
                emit_word(&e->code, arm64_sub_w_reg(0, 0, 1));
                emit_word(&e->code, arm64_add_sp_imm(16));   // pop one slot
                emit_word(&e->code, arm64_str_w_sp(0, 0));   // overwrite remaining
                value_depth -= 1;
                break;
            }
            case 0x47: { // i32.ne
                // CSET writes 1 when the inverted condition is *false*;
                // pass EQ so the result is 1 iff w0 != w1.
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));   // top   -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));  // below -> w0
                emit_word(&e->code, arm64_cmp_w(0, 1));
                emit_word(&e->code, arm64_cset_w(0, 0));     // cset w0, ne
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }
            case 0x23: {                                     // global.get
                uint32_t gi = (uint32_t)rd_uleb(&r);
                if (gi >= wm->n_globals) {
                    fprintf(stderr,
                            "wasm->macho: global.get %u out of range in "
                            "func %u (n_globals=%u)\n",
                            gi, fidx, wm->n_globals);
                    free(local_types); return false;
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
                value_depth += 1;
                break;
            }
            case 0x24: {                                     // global.set
                uint32_t gi = (uint32_t)rd_uleb(&r);
                if (gi >= wm->n_globals) {
                    fprintf(stderr,
                            "wasm->macho: global.set %u out of range in "
                            "func %u (n_globals=%u)\n",
                            gi, fidx, wm->n_globals);
                    free(local_types); return false;
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
                value_depth -= 1;
                break;
            }
            // --- Memory loads (1 pop addr, 1 push value).
            //
            // Every load goes through `emit_compute_mem_addr` so the
            // final LDR uses #0 immediate offset. This sidesteps the
            // per-opcode LDR scale limits and supports any wasm offset
            // up to 16 MB, which covers all realistic data layouts.
            case 0x28:   // i32.load    (zero-fill upper half of slot)
            case 0x29:   // i64.load
            case 0x2a:   // f32.load    (bit-identical to i32.load)
            case 0x2b:   // f64.load    (bit-identical to i64.load)
            case 0x2c:   // i32.load8_s
            case 0x2d:   // i32.load8_u
            case 0x2e:   // i32.load16_s
            case 0x2f:   // i32.load16_u
            case 0x30:   // i64.load8_s
            case 0x31:   // i64.load8_u
            case 0x32:   // i64.load16_s
            case 0x33:   // i64.load16_u
            case 0x34:   // i64.load32_s
            case 0x35: { // i64.load32_u
                (void)rd_uleb(&r);                       // align
                uint32_t off = (uint32_t)rd_uleb(&r);    // offset
                if (!wm->has_memory) {
                    fprintf(stderr,
                            "wasm->macho: load 0x%02x with no memory in "
                            "func %u\n", op, fidx);
                    free(local_types); return false;
                }
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));   // addr -> w0
                if (!emit_compute_mem_addr(&e->code, off)) {
                    fprintf(stderr,
                            "wasm->macho: load 0x%02x offset %u exceeds "
                            "16 MB in func %u\n", op, off, fidx);
                    free(local_types); return false;
                }
                // Issue the load; result -> w0 / x0.
                switch (op) {
                    case 0x28: case 0x2a:
                        emit_word(&e->code, arm64_ldr_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_w_sp(0, 0));
                        break;
                    case 0x29: case 0x2b:
                        emit_word(&e->code, arm64_ldr_x_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                    case 0x2c:
                        emit_word(&e->code, arm64_ldrsb_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_w_sp(0, 0));
                        break;
                    case 0x2d:
                        emit_word(&e->code, arm64_ldrb_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_w_sp(0, 0));
                        break;
                    case 0x2e:
                        emit_word(&e->code, arm64_ldrsh_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_w_sp(0, 0));
                        break;
                    case 0x2f:
                        emit_word(&e->code, arm64_ldrh_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_w_sp(0, 0));
                        break;
                    case 0x30:
                        // ldrsb w0; sxtb x0,w0; str x0 -> slot.
                        emit_word(&e->code, arm64_ldrsb_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_sxtb_x(0, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                    case 0x31:
                        emit_word(&e->code, arm64_ldrb_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                    case 0x32:
                        emit_word(&e->code, arm64_ldrsh_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_sxth_x(0, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                    case 0x33:
                        emit_word(&e->code, arm64_ldrh_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                    case 0x34:
                        emit_word(&e->code, arm64_ldrsw_x_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                    case 0x35:
                        emit_word(&e->code, arm64_ldr_w_xn(0, 10, 0));
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                        break;
                }
                break;
            }

            // --- Memory stores (2 pop: addr below, value above).
            case 0x36:   // i32.store
            case 0x37:   // i64.store
            case 0x38:   // f32.store    (bit-identical to i32.store)
            case 0x39:   // f64.store    (bit-identical to i64.store)
            case 0x3a:   // i32.store8
            case 0x3b: { // i32.store16
                (void)rd_uleb(&r);
                uint32_t off = (uint32_t)rd_uleb(&r);
                if (!wm->has_memory) {
                    fprintf(stderr,
                            "wasm->macho: store 0x%02x with no memory in "
                            "func %u\n", op, fidx);
                    free(local_types); return false;
                }
                // Pop value (full 64-bit slot) into x1, addr into w0.
                emit_word(&e->code, arm64_ldr_x_sp(1, 0));
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));
                emit_word(&e->code, arm64_add_sp_imm(32));
                if (!emit_compute_mem_addr(&e->code, off)) {
                    fprintf(stderr,
                            "wasm->macho: store 0x%02x offset %u exceeds "
                            "16 MB in func %u\n", op, off, fidx);
                    free(local_types); return false;
                }
                switch (op) {
                    case 0x36: case 0x38:
                        emit_word(&e->code, arm64_str_w_xn(1, 10, 0));
                        break;
                    case 0x37: case 0x39:
                        emit_word(&e->code, arm64_str_x_xn(1, 10, 0));
                        break;
                    case 0x3a:
                        emit_word(&e->code, arm64_strb_w_xn(1, 10, 0));
                        break;
                    case 0x3b:
                        emit_word(&e->code, arm64_strh_w_xn(1, 10, 0));
                        break;
                }
                value_depth -= 2;
                break;
            }
            case 0x10: { // call funcidx
                uint32_t cidx = (uint32_t)rd_uleb(&r);
                uint32_t np = 0, nr = 0;
                uint8_t param_types[8] = {0};
                uint8_t result_types[1] = {0};
                if (cidx < wm->n_imported_funcs) {
                    if (cidx == wm->import_idx_proc_exit) {
                        np = 1; nr = 0;
                        param_types[0] = 0x7f;
                    }
                    else if (cidx == wm->import_idx_fd_write) {
                        np = 4; nr = 1;
                        param_types[0] = 0x7f; param_types[1] = 0x7f;
                        param_types[2] = 0x7f; param_types[3] = 0x7f;
                        result_types[0] = 0x7f;
                    }
                    else {
                        fprintf(stderr,
                                "wasm->macho: unsupported import call (idx %u)\n",
                                cidx);
                        free(local_types); return false;
                    }
                } else {
                    uint32_t cfidx = cidx - wm->n_imported_funcs;
                    if (cfidx >= wm->n_funcs) {
                        fprintf(stderr,
                                "wasm->macho: call out of range: %u\n", cidx);
                        free(local_types); return false;
                    }
                    if (!type_arity_params(wm, wm->funcs[cfidx].type_idx,
                                           &np, &nr,
                                           param_types, result_types)) {
                        free(local_types); return false;
                    }
                }
                if (np > 8) {
                    fprintf(stderr,
                            "wasm->macho: call with %u params not supported\n",
                            np);
                    free(local_types); return false;
                }
                if (nr > 1) {
                    fprintf(stderr,
                            "wasm->macho: call with %u results not supported\n",
                            nr);
                    free(local_types); return false;
                }
                // ARM64 PCS: GPRs (x0..x7) and FPRs (d0..d7) are
                // *independent* sequences. Assign each param its
                // register *first* by walking left-to-right, then pop
                // them off the value stack in right-to-left order.
                uint8_t arg_reg[8] = {0};
                uint8_t arg_is_fp[8] = {0};
                uint32_t n_gpr = 0, n_fpr = 0;
                for (uint32_t k = 0; k < np; k++) {
                    uint8_t t = param_types[k];
                    bool fp = (t == 0x7c) || (t == 0x7d);
                    arg_is_fp[k] = fp;
                    if (fp) {
                        if (n_fpr >= 8) {
                            fprintf(stderr,
                                    "wasm->macho: too many fp args\n");
                            free(local_types); return false;
                        }
                        arg_reg[k] = (uint8_t)n_fpr++;
                    } else {
                        if (n_gpr >= 8) {
                            fprintf(stderr,
                                    "wasm->macho: too many int args\n");
                            free(local_types); return false;
                        }
                        arg_reg[k] = (uint8_t)n_gpr++;
                    }
                }
                // Pop args off the CPU stack from top (= last param)
                // down to bottom (= first param), loading each into
                // its assigned register.
                for (int k = (int)np - 1; k >= 0; k--) {
                    uint8_t t = param_types[k];
                    uint32_t r_idx = arg_reg[k];
                    if (t == 0x7c)
                        emit_word(&e->code, arm64_ldr_d_sp(r_idx, 0));
                    else if (t == 0x7d)
                        emit_word(&e->code, arm64_ldr_s_sp(r_idx, 0));
                    else
                        emit_word(&e->code, arm64_ldr_x_sp(r_idx, 0));
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
                    uint8_t rt = result_types[0];
                    if (rt == 0x7c)
                        emit_word(&e->code, arm64_str_d_sp(0, 0));
                    else if (rt == 0x7d)
                        emit_word(&e->code, arm64_str_s_sp(0, 0));
                    else
                        emit_word(&e->code, arm64_str_x_sp(0, 0));
                }
                value_depth = value_depth - np + nr;
                break;
            }

            // --- i32 comparisons (2 pop, 1 push). Pattern:
            //     ldr w1,[sp,#0]; ldr w0,[sp,#16]; cmp w0,w1; cset w0,COND;
            //     add sp,#16; str w0,[sp].
            case 0x46:   // i32.eq      cset cond_inv = NE = 1
            case 0x49:   // i32.lt_u    cset cond_inv = HS = 2
            case 0x4a:   // i32.gt_s    cset cond_inv = LE = 0xd
            case 0x4b:   // i32.gt_u    cset cond_inv = LS = 9
            case 0x4c:   // i32.le_s    cset cond_inv = GT = 0xc
            case 0x4d:   // i32.le_u    cset cond_inv = HI = 8
            case 0x4e:   // i32.ge_s    cset cond_inv = LT = 0xb
            case 0x4f: { // i32.ge_u    cset cond_inv = LO = 3
                uint32_t cond_inv;
                switch (op) {
                    case 0x46: cond_inv = 1;    break;
                    case 0x49: cond_inv = 2;    break;
                    case 0x4a: cond_inv = 0xd;  break;
                    case 0x4b: cond_inv = 9;    break;
                    case 0x4c: cond_inv = 0xc;  break;
                    case 0x4d: cond_inv = 8;    break;
                    case 0x4e: cond_inv = 0xb;  break;
                    case 0x4f: cond_inv = 3;    break;
                    default:   cond_inv = 0;    break;
                }
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));
                emit_word(&e->code, arm64_cmp_w(0, 1));
                emit_word(&e->code, arm64_cset_w(0, cond_inv));
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- i32 binary arithmetic (2 pop, 1 push). Same envelope as
            // i32.add; only the middle instruction differs.
            case 0x6c:   // i32.mul
            case 0x6d:   // i32.div_s
            case 0x6e:   // i32.div_u
            case 0x71:   // i32.and
            case 0x72:   // i32.or
            case 0x73:   // i32.xor
            case 0x74:   // i32.shl
            case 0x75:   // i32.shr_s
            case 0x76: { // i32.shr_u
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));
                uint32_t insn = 0;
                switch (op) {
                    case 0x6c: insn = arm64_mul_w   (0, 0, 1); break;
                    case 0x6d: insn = arm64_sdiv_w  (0, 0, 1); break;
                    case 0x6e: insn = arm64_udiv_w  (0, 0, 1); break;
                    case 0x71: insn = arm64_and_w_reg(0, 0, 1); break;
                    case 0x72: insn = arm64_orr_w_reg(0, 0, 1); break;
                    case 0x73: insn = arm64_eor_w_reg(0, 0, 1); break;
                    case 0x74: insn = arm64_lsl_w_reg(0, 0, 1); break;
                    case 0x75: insn = arm64_asr_w_reg(0, 0, 1); break;
                    case 0x76: insn = arm64_lsr_w_reg(0, 0, 1); break;
                }
                emit_word(&e->code, insn);
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- i64.eqz (1 pop, 1 push).
            case 0x50: {
                emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                emit_word(&e->code, arm64_cmp_x_imm0(0));
                emit_word(&e->code, arm64_cset_w(0, 1));  // cset w0, eq
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }

            // --- i64 comparisons (2 pop, 1 push). Same as the i32 set but
            // 64-bit operands. Result is an i32 (0/1).
            case 0x51:   // i64.eq    cond_inv = NE = 1
            case 0x52:   // i64.ne    cond_inv = EQ = 0
            case 0x53:   // i64.lt_s  cond_inv = GE = 0xa
            case 0x54:   // i64.lt_u  cond_inv = HS = 2
            case 0x55:   // i64.gt_s  cond_inv = LE = 0xd
            case 0x56:   // i64.gt_u  cond_inv = LS = 9
            case 0x57:   // i64.le_s  cond_inv = GT = 0xc
            case 0x58:   // i64.le_u  cond_inv = HI = 8
            case 0x59:   // i64.ge_s  cond_inv = LT = 0xb
            case 0x5a: { // i64.ge_u  cond_inv = LO = 3
                uint32_t cond_inv;
                switch (op) {
                    case 0x51: cond_inv = 1;    break;
                    case 0x52: cond_inv = 0;    break;
                    case 0x53: cond_inv = 0xa;  break;
                    case 0x54: cond_inv = 2;    break;
                    case 0x55: cond_inv = 0xd;  break;
                    case 0x56: cond_inv = 9;    break;
                    case 0x57: cond_inv = 0xc;  break;
                    case 0x58: cond_inv = 8;    break;
                    case 0x59: cond_inv = 0xb;  break;
                    case 0x5a: cond_inv = 3;    break;
                    default:   cond_inv = 0;    break;
                }
                emit_word(&e->code, arm64_ldr_x_sp(1, 0));
                emit_word(&e->code, arm64_ldr_x_sp(0, 16));
                emit_word(&e->code, arm64_cmp_x(0, 1));
                emit_word(&e->code, arm64_cset_w(0, cond_inv));
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- i64 binary arithmetic (2 pop, 1 push). 64-bit operands +
            // result.
            case 0x7c:   // i64.add
            case 0x7d:   // i64.sub
            case 0x7e:   // i64.mul
            case 0x7f:   // i64.div_s
            case 0x80:   // i64.div_u
            case 0x83:   // i64.and
            case 0x84:   // i64.or
            case 0x85:   // i64.xor
            case 0x86:   // i64.shl
            case 0x87:   // i64.shr_s
            case 0x88: { // i64.shr_u
                emit_word(&e->code, arm64_ldr_x_sp(1, 0));
                emit_word(&e->code, arm64_ldr_x_sp(0, 16));
                uint32_t insn = 0;
                switch (op) {
                    case 0x7c: insn = arm64_add_x_reg(0, 0, 1); break;
                    case 0x7d: insn = arm64_sub_x_reg(0, 0, 1); break;
                    case 0x7e: insn = arm64_mul_x    (0, 0, 1); break;
                    case 0x7f: insn = arm64_sdiv_x   (0, 0, 1); break;
                    case 0x80: insn = arm64_udiv_x   (0, 0, 1); break;
                    case 0x83: insn = arm64_and_x_reg(0, 0, 1); break;
                    case 0x84: insn = arm64_orr_x_reg(0, 0, 1); break;
                    case 0x85: insn = arm64_eor_x_reg(0, 0, 1); break;
                    case 0x86: insn = arm64_lsl_x_reg(0, 0, 1); break;
                    case 0x87: insn = arm64_asr_x_reg(0, 0, 1); break;
                    case 0x88: insn = arm64_lsr_x_reg(0, 0, 1); break;
                }
                emit_word(&e->code, insn);
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- i32.wrap_i64: 1 pop (i64), 1 push (i32). The top slot
            // holds the i64; consumers that read it as i32 only look at
            // the low 32 bits, so codegen is effectively a no-op.
            case 0xa7: {
                break;
            }

            // --- i64.extend_i32_s: sign-extend the low 32 bits.
            case 0xac: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));   // w0 = low32
                emit_word(&e->code, arm64_sxtw(0, 0));        // x0 = sxtw w0
                emit_word(&e->code, arm64_str_x_sp(0, 0));   // overwrite
                break;
            }

            // --- i64.extend_i32_u: zero-extend the low 32 bits. ldr w0
            // already zero-extends into x0, so just rewrite the slot.
            case 0xad: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }

            // --- unreachable: trap. Emitted by tinyc lowering for cases
            // the producer believes cannot run (e.g. after a noreturn
            // call). Use BRK to raise a debugger / kernel trap.
            case 0x00: {
                emit_word(&e->code, arm64_brk(0));
                break;
            }

            // --- i32.rem_s / i32.rem_u: r = a - (a/b)*b   via sdiv/udiv
            // + msub. ARM64 has no integer modulo instruction.
            case 0x6f:   // i32.rem_s
            case 0x70: { // i32.rem_u
                emit_word(&e->code, arm64_ldr_w_sp(1, 0));      // b -> w1
                emit_word(&e->code, arm64_ldr_w_sp(0, 16));     // a -> w0
                uint32_t divisor = (op == 0x6f)
                    ? arm64_sdiv_w(2, 0, 1)
                    : arm64_udiv_w(2, 0, 1);
                emit_word(&e->code, divisor);                   // w2 = a/b
                emit_word(&e->code, arm64_msub_w(0, 2, 1, 0));  // w0 = a - w2*b
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- i64.rem_s / i64.rem_u: 64-bit version.
            case 0x81:   // i64.rem_s
            case 0x82: { // i64.rem_u
                emit_word(&e->code, arm64_ldr_x_sp(1, 0));
                emit_word(&e->code, arm64_ldr_x_sp(0, 16));
                uint32_t divisor = (op == 0x81)
                    ? arm64_sdiv_x(2, 0, 1)
                    : arm64_udiv_x(2, 0, 1);
                emit_word(&e->code, divisor);
                emit_word(&e->code, arm64_msub_x(0, 2, 1, 0));
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- i32.extend8_s / i32.extend16_s: sign-extend low N bits
            // into i32. One-instruction sxtb/sxth.
            case 0xc0: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_sxtb_w(0, 0));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0xc1: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_sxth_w(0, 0));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }

            // --- i64.extend8_s / extend16_s / extend32_s.
            case 0xc2: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_sxtb_x(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }
            case 0xc3: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_sxth_x(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }
            case 0xc4: {
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_sxtw(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }

            // --- f32.const / f64.const: read raw IEEE-754 bytes and
            // push as bit-identical 32-bit / 64-bit value.
            case 0x43: { // f32.const (4-byte LE)
                if (r.p + 4 > r.end) { r.overflow = true; break; }
                uint32_t bits = (uint32_t)r.p[0]
                              | ((uint32_t)r.p[1] << 8)
                              | ((uint32_t)r.p[2] << 16)
                              | ((uint32_t)r.p[3] << 24);
                r.p += 4;
                emit_mov_w_imm32(&e->code, 0, bits);
                emit_word(&e->code, arm64_sub_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth += 1;
                break;
            }
            case 0x44: { // f64.const (8-byte LE)
                if (r.p + 8 > r.end) { r.overflow = true; break; }
                uint64_t bits = 0;
                for (int i = 0; i < 8; i++)
                    bits |= ((uint64_t)r.p[i]) << (i * 8);
                r.p += 8;
                emit_mov_x_imm64(&e->code, 0, bits);
                emit_word(&e->code, arm64_sub_sp_imm(16));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                value_depth += 1;
                break;
            }

            // --- f32 / f64 comparisons (2 pop, 1 push i32).
            //   FCMP sets NZCV; CSET emits 1 when COND holds, else 0.
            // The COND used is the *inverse* of the wasm op, matching
            // the i32 cmp handler pattern.
            case 0x5b:   // f32.eq    cond_inv = NE = 1
            case 0x5c:   // f32.ne    cond_inv = EQ = 0
            case 0x5d:   // f32.lt    cond_inv = PL = 5  (N==0)  -- NOTE: see below
            case 0x5e:   // f32.gt    cond_inv = LE = 0xd
            case 0x5f:   // f32.le    cond_inv = GT = 0xc
            case 0x60:   // f32.ge    cond_inv = LT = 0xb
            case 0x61:   // f64.eq
            case 0x62:   // f64.ne
            case 0x63:   // f64.lt
            case 0x64:   // f64.gt
            case 0x65:   // f64.le
            case 0x66: { // f64.ge
                bool is_double = (op >= 0x61);
                int sub = (op >= 0x61) ? (op - 0x61) : (op - 0x5b);
                // For floating-point comparisons we need ordered
                // semantics: lt/gt/le/ge must return 0 when either
                // operand is NaN; eq must return 0; ne must return 1.
                // ARM64 FCMP sets NZCV with V=1 on unordered (NaN).
                // The conditions LT/LE/GT/GE on FP use special codes:
                //   FP-LT = "MI" (N==1) = cond 4
                //   FP-LE = "LS" (C==0 || Z==1) = cond 9
                //   FP-GT = "GT" (Z==0 && N==V) = cond 0xc
                //   FP-GE = "GE" (N==V) = cond 0xa
                //   FP-EQ = "EQ" = cond 0
                //   FP-NE = "NE" = cond 1
                // (See ARM ARM table "Condition flags for FP".)
                // We invert because the existing cset helper uses the
                // *inverse* of the desired truth value.
                uint32_t cond_for_true;
                switch (sub) {
                    case 0: cond_for_true = 0;    break; // eq
                    case 1: cond_for_true = 1;    break; // ne
                    case 2: cond_for_true = 4;    break; // lt (MI)
                    case 3: cond_for_true = 0xc;  break; // gt (GT)
                    case 4: cond_for_true = 9;    break; // le (LS)
                    case 5: cond_for_true = 0xa;  break; // ge (GE)
                    default: cond_for_true = 0;   break;
                }
                if (is_double) {
                    emit_word(&e->code, arm64_ldr_d_sp(1, 0));
                    emit_word(&e->code, arm64_ldr_d_sp(0, 16));
                    emit_word(&e->code, arm64_fcmp_d(0, 1));
                } else {
                    emit_word(&e->code, arm64_ldr_s_sp(1, 0));
                    emit_word(&e->code, arm64_ldr_s_sp(0, 16));
                    emit_word(&e->code, arm64_fcmp_s(0, 1));
                }
                // cset Wd, cond  encodes the *inverse* of the desired
                // condition, matching the i32-cmp scheme used above.
                uint32_t cond_inv = cond_for_true ^ 1u;
                emit_word(&e->code, arm64_cset_w(0, cond_inv));
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- f32 binary ops: add / sub / mul / div.
            case 0x92:   // f32.add
            case 0x93:   // f32.sub
            case 0x94:   // f32.mul
            case 0x95: { // f32.div
                emit_word(&e->code, arm64_ldr_s_sp(1, 0));
                emit_word(&e->code, arm64_ldr_s_sp(0, 16));
                uint32_t insn;
                switch (op) {
                    case 0x92: insn = arm64_fadd_s(0, 0, 1); break;
                    case 0x93: insn = arm64_fsub_s(0, 0, 1); break;
                    case 0x94: insn = arm64_fmul_s(0, 0, 1); break;
                    case 0x95: insn = arm64_fdiv_s(0, 0, 1); break;
                    default: insn = 0; break;
                }
                emit_word(&e->code, insn);
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- f32 unary: abs / neg / sqrt.
            case 0x8b:   // f32.abs
            case 0x8c:   // f32.neg
            case 0x91: { // f32.sqrt
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));
                uint32_t insn;
                switch (op) {
                    case 0x8b: insn = arm64_fabs_s (0, 0); break;
                    case 0x8c: insn = arm64_fneg_s (0, 0); break;
                    case 0x91: insn = arm64_fsqrt_s(0, 0); break;
                    default: insn = 0; break;
                }
                emit_word(&e->code, insn);
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                break;
            }

            // --- f64 binary ops: add / sub / mul / div.
            case 0xa0:   // f64.add
            case 0xa1:   // f64.sub
            case 0xa2:   // f64.mul
            case 0xa3: { // f64.div
                emit_word(&e->code, arm64_ldr_d_sp(1, 0));
                emit_word(&e->code, arm64_ldr_d_sp(0, 16));
                uint32_t insn;
                switch (op) {
                    case 0xa0: insn = arm64_fadd_d(0, 0, 1); break;
                    case 0xa1: insn = arm64_fsub_d(0, 0, 1); break;
                    case 0xa2: insn = arm64_fmul_d(0, 0, 1); break;
                    case 0xa3: insn = arm64_fdiv_d(0, 0, 1); break;
                    default: insn = 0; break;
                }
                emit_word(&e->code, insn);
                emit_word(&e->code, arm64_add_sp_imm(16));
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                value_depth -= 1;
                break;
            }

            // --- f64 unary: abs / neg / sqrt.
            case 0x99:   // f64.abs
            case 0x9a:   // f64.neg
            case 0x9f: { // f64.sqrt
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));
                uint32_t insn;
                switch (op) {
                    case 0x99: insn = arm64_fabs_d (0, 0); break;
                    case 0x9a: insn = arm64_fneg_d (0, 0); break;
                    case 0x9f: insn = arm64_fsqrt_d(0, 0); break;
                    default: insn = 0; break;
                }
                emit_word(&e->code, insn);
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                break;
            }

            // --- FP→int conversions (round-toward-zero).
            case 0xa8: { // i32.trunc_f32_s
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzs_w_s(0, 0));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0xa9: { // i32.trunc_f32_u
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzu_w_s(0, 0));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0xaa: { // i32.trunc_f64_s
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzs_w_d(0, 0));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0xab: { // i32.trunc_f64_u
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzu_w_d(0, 0));
                emit_word(&e->code, arm64_str_w_sp(0, 0));
                break;
            }
            case 0xae: { // i64.trunc_f32_s
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzs_x_s(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }
            case 0xaf: { // i64.trunc_f32_u
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzu_x_s(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }
            case 0xb0: { // i64.trunc_f64_s
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzs_x_d(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }
            case 0xb1: { // i64.trunc_f64_u
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));
                emit_word(&e->code, arm64_fcvtzu_x_d(0, 0));
                emit_word(&e->code, arm64_str_x_sp(0, 0));
                break;
            }

            // --- int→FP conversions.
            case 0xb2: { // f32.convert_i32_s
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_scvtf_s_w(0, 0));
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                break;
            }
            case 0xb3: { // f32.convert_i32_u
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_ucvtf_s_w(0, 0));
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                break;
            }
            case 0xb4: { // f32.convert_i64_s
                emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                emit_word(&e->code, arm64_scvtf_s_x(0, 0));
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                break;
            }
            case 0xb5: { // f32.convert_i64_u
                emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                emit_word(&e->code, arm64_ucvtf_s_x(0, 0));
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                break;
            }
            case 0xb7: { // f64.convert_i32_s
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_scvtf_d_w(0, 0));
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                break;
            }
            case 0xb8: { // f64.convert_i32_u
                emit_word(&e->code, arm64_ldr_w_sp(0, 0));
                emit_word(&e->code, arm64_ucvtf_d_w(0, 0));
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                break;
            }
            case 0xb9: { // f64.convert_i64_s
                emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                emit_word(&e->code, arm64_scvtf_d_x(0, 0));
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                break;
            }
            case 0xba: { // f64.convert_i64_u
                emit_word(&e->code, arm64_ldr_x_sp(0, 0));
                emit_word(&e->code, arm64_ucvtf_d_x(0, 0));
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                break;
            }

            // --- f32 ↔ f64 precision conversions.
            case 0xb6: { // f32.demote_f64
                emit_word(&e->code, arm64_ldr_d_sp(0, 0));
                emit_word(&e->code, arm64_fcvt_s_d(0, 0));
                emit_word(&e->code, arm64_str_s_sp(0, 0));
                break;
            }
            case 0xbb: { // f64.promote_f32
                emit_word(&e->code, arm64_ldr_s_sp(0, 0));
                emit_word(&e->code, arm64_fcvt_d_s(0, 0));
                emit_word(&e->code, arm64_str_d_sp(0, 0));
                break;
            }

            // --- Bit-identical reinterprets (no-op for our slot layout).
            // The bits are already in place; we just acknowledge them.
            case 0xbc:   // i32.reinterpret_f32
            case 0xbd:   // i64.reinterpret_f64
            case 0xbe:   // f32.reinterpret_i32
            case 0xbf: { // f64.reinterpret_i64
                break;
            }

            default:
                fprintf(stderr,
                        "wasm->macho: unsupported opcode 0x%02x in func %u\n",
                        op, fidx);
                free(local_types); return false;
        }
    }

    if (block_depth != 0) {
        fprintf(stderr,
                "wasm->macho: unclosed block (depth %u) in func %u\n",
                block_depth, fidx);
        free(local_types); return false;
    }

    // Trailing fall-through epilogue.
    if (nr_self > 1) { free(local_types); return false; }
    EMIT_EPILOGUE();
    #undef EMIT_EPILOGUE

    free(local_types);
    return !r.overflow;
}

// Emit the proc_exit shim: tail-call _exit (which never returns).
//   b _exit_stub
static void emit_proc_exit_shim(EmittedFunc *e) {
    CallReloc rel = {
        .site_off = (uint32_t)e->code.len,
        .target = STUB_EXIT,
        .is_b = 1,
    };
    ef_push_reloc(e, rel);
    emit_word(&e->code, arm64_b(0));
}

// Emit the fd_write shim. WASI signature:
//
//   i32 fd_write(i32 fd, i32 iovs_ptr, i32 iovs_len, i32 nwritten_ptr);
//
// Iterates over the iovec array in linear memory, calls libSystem
// `_write(fd, linmem+buf_ptr, buf_len)` for each chunk, accumulates
// the total bytes written into `*nwritten_ptr`, and returns 0.
// Errors from `_write` are not propagated to errno — sufficient for
// the tinyC test suite which writes to stdout only.
//
// Register usage (callee-saved):
//   x19  fd                  x22  total written
//   x20  iovs_ptr (wasm32)   x23  nwritten_ptr (wasm32)
//   x21  iovs_len
//
// x28 holds the linear-memory base (set up at `_start`) and is
// preserved by AArch64 PCS across libSystem calls.
static void emit_fd_write_shim(EmittedFunc *e) {
    // Prologue: stp fp, lr / mov fp, sp / sub sp, sp, #48
    emit_word(&e->code, arm64_stp_fp_lr_pre());
    emit_word(&e->code, arm64_mov_fp_sp());
    emit_word(&e->code, arm64_sub_sp_imm(48));
    emit_word(&e->code, arm64_str_x_sp(19, 0));
    emit_word(&e->code, arm64_str_x_sp(20, 8));
    emit_word(&e->code, arm64_str_x_sp(21, 16));
    emit_word(&e->code, arm64_str_x_sp(22, 24));
    emit_word(&e->code, arm64_str_x_sp(23, 32));

    // Move args into callee-saved registers; total = 0.
    emit_word(&e->code, arm64_mov_w_reg(19, 0));   // fd
    emit_word(&e->code, arm64_mov_w_reg(20, 1));   // iovs_ptr
    emit_word(&e->code, arm64_mov_w_reg(21, 2));   // iovs_len
    emit_word(&e->code, arm64_mov_w_reg(23, 3));   // nwritten_ptr
    emit_word(&e->code, arm64_movz_x(22, 0, 0));   // total = 0

    // loop_top:
    uint32_t loop_top = (uint32_t)e->code.len;
    uint32_t cbz_site = (uint32_t)e->code.len;
    emit_word(&e->code, arm64_cbz_w(21, 0));        // patched -> loop_done
    // x10 = linmem + iovs_ptr
    emit_word(&e->code, arm64_add_x_xn_wm_uxtw(10, 28, 20));
    emit_word(&e->code, arm64_ldr_w_xn(11, 10, 0));  // buf_ptr
    emit_word(&e->code, arm64_ldr_w_xn(12, 10, 4));  // buf_len
    // _write(fd, linmem+buf_ptr, buf_len)
    emit_word(&e->code, arm64_mov_w_reg(0, 19));
    emit_word(&e->code, arm64_add_x_xn_wm_uxtw(1, 28, 11));
    emit_word(&e->code, arm64_mov_w_reg(2, 12));
    CallReloc rel = {
        .site_off = (uint32_t)e->code.len,
        .target = STUB_WRITE,
        .is_b = 0,
    };
    ef_push_reloc(e, rel);
    emit_word(&e->code, arm64_bl(0));
    // total += x0  (treat negative returns as 0-ish; minimal handling)
    emit_word(&e->code, arm64_add_x_reg(22, 22, 0));
    // iovs_ptr += 8; iovs_len -= 1
    emit_word(&e->code, arm64_add_w_imm(20, 20, 8));
    emit_word(&e->code, arm64_sub_w_imm(21, 21, 1));
    // b loop_top
    int32_t back = (int32_t)loop_top - (int32_t)e->code.len;
    emit_word(&e->code, arm64_b(back));

    // loop_done:
    patch_cbz_local(e->code.data, cbz_site, (uint32_t)e->code.len);
    // *nwritten_ptr = (i32)total
    emit_word(&e->code, arm64_add_x_xn_wm_uxtw(10, 28, 23));
    emit_word(&e->code, arm64_str_w_xn(22, 10, 0));

    // Epilogue
    emit_word(&e->code, arm64_movz_w(0, 0));        // return 0
    emit_word(&e->code, arm64_ldr_x_sp(19, 0));
    emit_word(&e->code, arm64_ldr_x_sp(20, 8));
    emit_word(&e->code, arm64_ldr_x_sp(21, 16));
    emit_word(&e->code, arm64_ldr_x_sp(22, 24));
    emit_word(&e->code, arm64_ldr_x_sp(23, 32));
    emit_word(&e->code, arm64_add_sp_imm(48));
    emit_word(&e->code, arm64_ldp_fp_lr_post());
    emit_word(&e->code, arm64_ret());
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
#define TEXT_FILE_BASE         0u
#define VMSEG_SIZE             0x4000u  // 16 KB minimum mach-o page

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

    wm.globals_vmaddr = TEXT_VM_BASE + 2 * VMSEG_SIZE;
    wm.linmem_vmaddr  = wm.globals_vmaddr + globals_padded;
    // NOTE: globals_vmaddr / linmem_vmaddr above are *placeholders*
    // used during function emission so the prologue ADRP gets a
    // well-formed (if eventually wrong) immediate. They are
    // finalised below after the actual __TEXT size is known, and
    // the ADRP is then patched in-place.

    uint64_t linkedit_vm_base   = 0; // finalised after text_seg_size known
    uint32_t linkedit_file_base = 0;

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
    for (uint32_t i = 0; i < n_total; i++) efs[i].adrp_off = UINT32_MAX;

    if (reachable[wm.import_idx_proc_exit]) {
        emit_proc_exit_shim(&efs[wm.import_idx_proc_exit]);
        efs[wm.import_idx_proc_exit].emitted = true;
    }
    bool need_write_stub = false;
    if (wm.import_idx_fd_write != UINT32_MAX
        && reachable[wm.import_idx_fd_write]) {
        emit_fd_write_shim(&efs[wm.import_idx_fd_write]);
        efs[wm.import_idx_fd_write].emitted = true;
        need_write_stub = true;
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

    // __stubs: one entry per reachable libSystem import.
    //   stub 0: _exit
    //   stub 1: _write (only when fd_write is reachable)
    const uint32_t n_stubs = need_write_stub ? 2u : 1u;
    const uint32_t stub_size = 12;
    const uint32_t stubs_off  = text_section_off + text_size;
    const uint32_t stubs_size = n_stubs * stub_size;

    // __cstring: empty for macho_exit.
    const uint32_t cstring_off  = stubs_off + stubs_size;
    const uint32_t cstring_size = 0;

    // GOT in __DATA_CONST.
    const uint32_t got_size = n_stubs * 8;

    // -----------------------------------------------------------------
    // Finalise segment layout now that we know the real text size.
    //
    // __TEXT segment file/vm size = round_up(text_section_off + text +
    //                                        stubs + cstring, 16 KiB)
    // and never below VMSEG_SIZE so the simplest macho_exit case is
    // unchanged byte-for-byte.
    // __DATA_CONST and __DATA each occupy one VMSEG_SIZE-aligned slab
    // right after __TEXT in both VM and file space.
    // -----------------------------------------------------------------
    uint32_t text_seg_end = cstring_off + cstring_size;
    uint32_t text_seg_size = (text_seg_end + (VMSEG_SIZE - 1u))
                             & ~(VMSEG_SIZE - 1u);
    if (text_seg_size < VMSEG_SIZE) text_seg_size = VMSEG_SIZE;
    const uint64_t data_const_vm_base   = TEXT_VM_BASE + text_seg_size;
    const uint32_t data_const_file_base = text_seg_size;
    const uint64_t data_vm_base         = data_const_vm_base + VMSEG_SIZE;
    const uint32_t data_file_base       = data_const_file_base + VMSEG_SIZE;

    // Finalise globals/linmem placement.
    wm.globals_vmaddr = data_vm_base;
    wm.linmem_vmaddr  = data_vm_base + globals_padded;

    linkedit_vm_base   = data_vm_base + (has_data_seg ? data_seg_vmsize : 0);
    linkedit_file_base = data_file_base + (has_data_seg ? data_seg_size32 : 0);

    // Patch _start's prologue ADRP to materialise the now-final
    // globals_vmaddr. The matching `add x28, x27, #globals_padded`
    // and the (optional) `add x27, x27, #off_in_page` are independent
    // of where __DATA lands because globals_padded is fixed and
    // globals_vmaddr is page-aligned (off_in_page == 0).
    {
        EmittedFunc *e = &efs[wm.start_export_idx];
        if (e->adrp_off != UINT32_MAX) {
            uint64_t pc_pc       = wm.entry_vmaddr
                                 + (uint64_t)e->adrp_off;
            uint64_t pc_page     = pc_pc & ~0xfffULL;
            uint64_t target_page = wm.globals_vmaddr & ~0xfffULL;
            int64_t  rel_pages   = (int64_t)(target_page - pc_page) >> 12;
            uint32_t adrp = arm64_adrp(27, rel_pages);
            e->code.data[e->adrp_off + 0] = (uint8_t)(adrp >>  0);
            e->code.data[e->adrp_off + 1] = (uint8_t)(adrp >>  8);
            e->code.data[e->adrp_off + 2] = (uint8_t)(adrp >> 16);
            e->code.data[e->adrp_off + 3] = (uint8_t)(adrp >> 24);
        }
    }

    // -----------------------------------------------------------------
    // Patch every emitted function's call relocations.
    // -----------------------------------------------------------------
    // Each stub lives at __text-relative offset `text_size + i*stub_size`.
    uint32_t exit_stub_text_off  = text_size + 0u * stub_size;
    uint32_t write_stub_text_off = text_size + 1u * stub_size;
    for (uint32_t i = 0; i < n_total; i++) {
        if (!efs[i].emitted) continue;
        for (size_t k = 0; k < efs[i].n_relocs; k++) {
            CallReloc *cr = &efs[i].relocs[k];
            uint32_t target_off;
            if (cr->target == STUB_EXIT)       target_off = exit_stub_text_off;
            else if (cr->target == STUB_WRITE) target_off = write_stub_text_off;
            else                               target_off = efs[cr->target].text_off;
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
    buf_le64(&img, (uint64_t)text_seg_size);
    buf_le64(&img, TEXT_FILE_BASE);
    buf_le64(&img, (uint64_t)text_seg_size);
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
        buf_le32(&img, n_stubs);             // reserved1 = idx in indirect syms
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
        buf_le64(&img, data_vm_base);
        buf_le64(&img, data_seg_vmsize);
        buf_le64(&img, (uint64_t)data_file_base);
        buf_le64(&img, data_seg_vmsize);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, VM_PROT_READ | VM_PROT_WRITE);
        buf_le32(&img, 1);                // nsects
        buf_le32(&img, 0);                // flags
        {
            static const char SN[16] = "__data";
            static const char SG[16] = "__DATA";
            buf_append(&img, SN, 16); buf_append(&img, SG, 16);
            buf_le64(&img, data_vm_base);
            buf_le64(&img, (uint64_t)data_seg_payload);
            buf_le32(&img, data_file_base);
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
        uint64_t got_target = data_const_vm_base + 8ULL * i;
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
    buf_pad_to(&img, data_const_file_base);
    // __got: one chained-fixup-format pointer per import. The format
    // is DYLD_CHAINED_PTR_64_OFFSET (pointer_format=6). Layout
    // (LSB-first):
    //   bits  0..23  ordinal (= imports[] index)
    //   bits 24..31  addend
    //   bits 32..50  reserved
    //   bits 51..62  next (stride is 4 bytes -> step 2 to reach the next
    //                       8-byte GOT pointer)
    //   bit       63 bind (1)
    for (uint32_t i = 0; i < n_stubs; i++) {
        uint64_t v = (1ULL << 63) | (uint64_t)i;
        if (i + 1 < n_stubs) v |= (2ULL << 51);    // next = 2 -> +8 bytes
        buf_le64(&img, v);
    }

    // __DATA segment payload (optional): [globals bytes][linmem bytes].
    if (has_data_seg) {
        buf_pad_to(&img, data_file_base);
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
    buf_le64(&img, (uint64_t)data_const_file_base); // segment_offset (file)
    buf_le32(&img, 0);                    // max_valid_pointer
    buf_le16(&img, 1);                    // page_count
    buf_le16(&img, 0);                    // page_start[0]

    // imports table: one dyld_chained_import per import.
    //   { lib_ordinal: 8, weak_import: 1, name_offset: 23 }
    // ordinal=1 (the only LC_LOAD_DYLIB, libSystem). Symbols are
    // listed in the same order as stubs: _exit, then _write (if
    // present). Each name_offset points into the symbols table
    // following the leading 0.
    {
        uint32_t name_off = 1;            // skip leading NUL
        uint32_t entry = 1u | (0u << 8) | (name_off << 9);
        buf_le32(&img, entry);
    }
    if (need_write_stub) {
        uint32_t name_off = 1 + (uint32_t)strlen("_exit") + 1;
        uint32_t entry = 1u | (0u << 8) | (name_off << 9);
        buf_le32(&img, entry);
    }
    // symbols table: leading 0 + cstrs + trailing 0s for alignment.
    buf_u8(&img, 0);
    buf_cstr(&img, "_exit");
    if (need_write_stub) buf_cstr(&img, "_write");
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
    uint32_t str_mh    = (uint32_t)strtab.len; buf_cstr(&strtab, "__mh_execute_header");
    uint32_t str_main  = (uint32_t)strtab.len; buf_cstr(&strtab, "_main");
    uint32_t str_exit  = (uint32_t)strtab.len; buf_cstr(&strtab, "_exit");
    uint32_t str_write = 0;
    if (need_write_stub) {
        str_write = (uint32_t)strtab.len;
        buf_cstr(&strtab, "_write");
    }
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
    if (need_write_stub) {
        // _write (undefined, dynamic-lookup)
        buf_le32(&symtab, str_write);
        buf_u8(&symtab, 0x01); buf_u8(&symtab, 0);
        buf_le16(&symtab, 0x0100);
        buf_le64(&symtab, 0);
    }
    uint32_t n_syms       = 2u + n_stubs;
    uint32_t n_undefs     = n_stubs;
    uint32_t iundefsym    = 2;
    uint32_t first_undef  = iundefsym;   // first undefined symbol index

    // Indirect symbols: one per __got entry and one per __stubs entry.
    // Layout: __got[0..n_stubs-1] then __stubs[0..n_stubs-1]. Each value
    // is an index into the symtab pointing at the undefined symbol
    // that backs that GOT / stub slot. Our symtab undefs are laid out
    // in the same order as stubs (stub i -> undef symbol first_undef+i).
    Buf indsyms = {0};
    for (uint32_t i = 0; i < n_stubs; i++) buf_le32(&indsyms, first_undef + i); // GOT
    for (uint32_t i = 0; i < n_stubs; i++) buf_le32(&indsyms, first_undef + i); // stubs

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
    buf_patch_le32(&img, pos_lc_symtab           + 12, n_syms);                  // nsyms
    buf_patch_le32(&img, pos_lc_symtab           + 16, (uint32_t)strtab_off);
    buf_patch_le32(&img, pos_lc_symtab           + 20, (uint32_t)strtab.len);
    buf_patch_le32(&img, pos_lc_dysymtab         + 8,  0);                       // ilocalsym
    buf_patch_le32(&img, pos_lc_dysymtab         + 12, 0);                       // nlocalsym
    buf_patch_le32(&img, pos_lc_dysymtab         + 16, 0);                       // iextdefsym
    buf_patch_le32(&img, pos_lc_dysymtab         + 20, 2);                       // nextdefsym
    buf_patch_le32(&img, pos_lc_dysymtab         + 24, iundefsym);               // iundefsym
    buf_patch_le32(&img, pos_lc_dysymtab         + 28, n_undefs);                // nundefsym
    buf_patch_le32(&img, pos_lc_dysymtab         + 32, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 36, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 40, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 44, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 48, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 52, 0);
    buf_patch_le32(&img, pos_lc_dysymtab         + 56, (uint32_t)indsyms_off);
    buf_patch_le32(&img, pos_lc_dysymtab         + 60, 2u * n_stubs);            // nindirectsyms
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

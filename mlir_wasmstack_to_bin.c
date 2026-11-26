// Stage 3 of the native LLVM->WASM pipeline:
// wasmstack builtin.module (MLIR) -> wasm32 relocatable object bytes.
//
// This stage walks the input wasmstack-MLIR module directly: each
// top-level op is `wasmstack.import_func`, `wasmstack.func`, or
// `wasmstack.import_global`, and the bodies of `wasmstack.func` are
// flat sequences of `wasmstack.*` ops. Each op dispatches on its
// MLIR_OpType discriminator and writes the corresponding bytecode.
//
// The output is wasm-ld-compatible: TYPE / IMPORT / FUNCTION / CODE
// sections plus the conventional `linking` and `reloc.CODE` custom
// sections.

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
#include "mlir_wasmstack_to_bin.h"

// =============================================================================
// Growable byte buffer + LEB128 encoders.
// =============================================================================
typedef struct { uint8_t *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 256;
    while (b->len + add > nc) nc *= 2;
    b->data = (uint8_t *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_putc(Buf *b, uint8_t c) { buf_grow(b, 1); b->data[b->len++] = c; }
static void buf_append(Buf *b, const void *p, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}
static void buf_free(Buf *b) { free(b->data); b->data = NULL; b->len = b->cap = 0; }

static void leb_u(Buf *b, uint64_t v) {
    do {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (v) byte |= 0x80;
        buf_putc(b, byte);
    } while (v);
}
static void leb_s(Buf *b, int64_t v) {
    bool more = true;
    while (more) {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        bool sign = (byte & 0x40) != 0;
        if ((v == 0 && !sign) || (v == -1 && sign)) more = false;
        else byte |= 0x80;
        buf_putc(b, byte);
    }
}
static void leb_u_pad5(Buf *b, uint32_t v) {
    for (int i = 0; i < 5; i++) {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (i < 4) byte |= 0x80;
        buf_putc(b, byte);
    }
}
static void leb_s_pad5(Buf *b, int32_t v) {
    for (int i = 0; i < 5; i++) {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (i < 4) byte |= 0x80;
        buf_putc(b, byte);
    }
}

// =============================================================================
// WASM constants.
// =============================================================================
enum {
    SEC_CUSTOM   = 0,
    SEC_TYPE     = 1,
    SEC_IMPORT   = 2,
    SEC_FUNCTION = 3,
    SEC_ELEMENT  = 9,
    SEC_CODE     = 10,
    SEC_DATA     = 11,

    IMP_FUNC   = 0,
    IMP_TABLE  = 1,
    IMP_MEMORY = 2,
    IMP_GLOBAL = 3,

    LINK_VERSION             = 2,
    LINK_SUB_SEGMENT_INFO    = 5,
    LINK_SUB_SYMBOL_TABLE    = 8,

    SYM_FUNCTION = 0,
    SYM_DATA     = 1,
    SYM_GLOBAL   = 2,
    SYM_TABLE    = 5,

    SYMF_BINDING_LOCAL     = 0x02,
    SYMF_VISIBILITY_HIDDEN = 0x04,
    SYMF_UNDEFINED         = 0x10,
    SYMF_NO_STRIP          = 0x80,

    R_WASM_FUNCTION_INDEX_LEB = 0,
    R_WASM_TABLE_INDEX_SLEB   = 1,
    R_WASM_MEMORY_ADDR_LEB    = 3,
    R_WASM_MEMORY_ADDR_SLEB   = 4,
    R_WASM_MEMORY_ADDR_I32    = 5,
    R_WASM_TYPE_INDEX_LEB     = 6,
    R_WASM_GLOBAL_INDEX_LEB   = 7,
};

// =============================================================================
// MLIR attribute readers.
// =============================================================================
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

// Hex-string -> byte vector decoder (caller owns the malloc'd result).
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

// Parse "off:target:addend,off2:target2:addend2,..." reloc string.
typedef struct { uint32_t offset; char *target; int32_t addend; } DataReloc;
static DataReloc *parse_relocs(string s, size_t *out_n) {
    *out_n = 0;
    if (s.size == 0) return NULL;
    DataReloc *out = NULL; size_t n = 0, c = 0;
    size_t i = 0;
    while (i < s.size) {
        uint32_t off = 0;
        while (i < s.size && s.str[i] >= '0' && s.str[i] <= '9') {
            off = off * 10 + (uint32_t)(s.str[i] - '0'); i++;
        }
        if (i >= s.size || s.str[i] != ':') { free(out); return NULL; }
        i++;
        size_t ts = i;
        while (i < s.size && s.str[i] != ':') i++;
        if (i >= s.size) { free(out); return NULL; }
        size_t tlen = i - ts;
        char *tgt = (char *)malloc(tlen + 1);
        memcpy(tgt, s.str + ts, tlen); tgt[tlen] = 0;
        i++;
        bool neg = false;
        if (i < s.size && s.str[i] == '-') { neg = true; i++; }
        int32_t ad = 0;
        while (i < s.size && s.str[i] >= '0' && s.str[i] <= '9') {
            ad = ad * 10 + (int32_t)(s.str[i] - '0'); i++;
        }
        if (neg) ad = -ad;
        if (n == c) { c = c ? c * 2 : 4; out = (DataReloc *)realloc(out, c * sizeof(DataReloc)); }
        out[n++] = (DataReloc){ off, tgt, ad };
        if (i < s.size && s.str[i] == ',') i++;
    }
    *out_n = n;
    return out;
}

// =============================================================================
// Function-signature interner (functype dedup for the TYPE section).
// =============================================================================
typedef struct {
    uint8_t *params;  size_t nparams;
    uint8_t *results; size_t nresults;
} Sig;

typedef struct { Sig *e; size_t n, cap; } SigTab;

static bool sig_eq(const Sig *a, const Sig *b) {
    if (a->nparams != b->nparams || a->nresults != b->nresults) return false;
    if (a->nparams && memcmp(a->params, b->params, a->nparams) != 0) return false;
    if (a->nresults && memcmp(a->results, b->results, a->nresults) != 0) return false;
    return true;
}
static uint32_t sig_intern(SigTab *t, const uint8_t *p, size_t np,
                           const uint8_t *r, size_t nr) {
    Sig probe = { (uint8_t *)p, np, (uint8_t *)r, nr };
    for (size_t i = 0; i < t->n; i++) if (sig_eq(&t->e[i], &probe)) return (uint32_t)i;
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 8;
        t->e = (Sig *)realloc(t->e, t->cap * sizeof(Sig));
    }
    Sig *s = &t->e[t->n];
    s->nparams = np;
    s->params = (uint8_t *)malloc(np ? np : 1); if (np) memcpy(s->params, p, np);
    s->nresults = nr;
    s->results = (uint8_t *)malloc(nr ? nr : 1); if (nr) memcpy(s->results, r, nr);
    return (uint32_t)t->n++;
}
static void sigtab_free(SigTab *t) {
    for (size_t i = 0; i < t->n; i++) { free(t->e[i].params); free(t->e[i].results); }
    free(t->e);
}

// =============================================================================
// Per-function emission state and global table.
// =============================================================================
typedef struct {
    uint8_t  type;
    uint32_t body_offset;
    uint32_t sym_idx;
    int32_t  addend;
} Reloc;

typedef struct {
    char    *name;
    uint32_t sig;
    bool     imported;
    bool     exported;
    bool     internal;
    Buf      body;
    uint32_t func_index;
    uint32_t sym_index;
    Reloc   *relocs;
    size_t   n_relocs, c_relocs;
    // Source MLIR func op (for body emission of defined funcs).
    MLIR_OpHandle src_op;
    // Locals layout (decoded from local_types attr).
    uint8_t *local_types; size_t n_locals;
    // Optional wasm.* attribute overrides (from
    // `__attribute__((__import_module__("...")))` etc.). NULL means
    // "use the default" (env / sym name).
    char *import_module;
    char *import_name;
    char *export_name;
} EmFunc;

typedef struct {
    char    *name;
    uint8_t *data;
    uint32_t size;
    uint32_t align_pow;
    bool     is_const;
    DataReloc *relocs;
    size_t   n_relocs;
} EmGlobal;

static void emfunc_add_reloc(EmFunc *f, uint8_t type, uint32_t off, uint32_t sym) {
    if (f->n_relocs == f->c_relocs) {
        f->c_relocs = f->c_relocs ? f->c_relocs * 2 : 4;
        f->relocs = (Reloc *)realloc(f->relocs, f->c_relocs * sizeof(Reloc));
    }
    f->relocs[f->n_relocs++] = (Reloc){type, off, sym, 0};
}

// =============================================================================
// Helpers.
// =============================================================================
static char *strdup_str(string s) {
    char *r = (char *)malloc(s.size + 1);
    if (s.size) memcpy(r, s.str, s.size);
    r[s.size] = 0;
    return r;
}
typedef struct {
    const char **keys;
    size_t      *klens;
    int         *vals;
    size_t       cap;
} NameMap;

static uint32_t nm_hash(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}
static void nm_init(NameMap *m, size_t n) {
    size_t cap = 16;
    while (cap < n * 2u + 1u) cap <<= 1;
    m->keys  = (const char **)calloc(cap, sizeof(const char *));
    m->klens = (size_t *)calloc(cap, sizeof(size_t));
    m->vals  = (int *)malloc(cap * sizeof(int));
    m->cap   = cap;
}
static void nm_put(NameMap *m, const char *k, int val) {
    size_t klen = strlen(k);
    size_t mask = m->cap - 1;
    size_t i = nm_hash(k, klen) & mask;
    while (m->keys[i]) {
        if (m->klens[i] == klen && memcmp(m->keys[i], k, klen) == 0) return;
        i = (i + 1) & mask;
    }
    m->keys[i] = k;
    m->klens[i] = klen;
    m->vals[i] = val;
}
static int nm_get(const NameMap *m, string nm) {
    if (m->cap == 0) return -1;
    size_t mask = m->cap - 1;
    size_t i = nm_hash(nm.str, nm.size) & mask;
    while (m->keys[i]) {
        if (m->klens[i] == nm.size && memcmp(m->keys[i], nm.str, nm.size) == 0)
            return m->vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}
static void nm_free(NameMap *m) {
    free(m->keys); free(m->klens); free(m->vals);
    m->keys = NULL; m->klens = NULL; m->vals = NULL; m->cap = 0;
}

static int find_func_by_name(const NameMap *fmap, string nm) {
    return nm_get(fmap, nm);
}
static int find_global_by_name(const NameMap *gmap, string nm) {
    return nm_get(gmap, nm);
}

// =============================================================================
// Per-op encoding. Reads attrs from the MLIR op and writes bytecode.
// =============================================================================
static bool encode_op_typed(MLIR_Context *ctx, EmFunc *F, MLIR_OpHandle op,
                            MLIR_OpType t, uint8_t valtype,
                            uint32_t sp_sym_idx,
                            EmFunc *all_funcs, const NameMap *fmap,
                            const NameMap *gmap,
                            uint32_t global_sym_base, SigTab *sigs) {
    Buf *b = &F->body;

    switch (t) {
    case OP_TYPE_WASMSTACK_LOCAL_GET:
        buf_putc(b, 0x20); leb_u(b, (uint64_t)at_i(ctx, op, MLIR_KEY_LOCAL_IDX)); return true;
    case OP_TYPE_WASMSTACK_LOCAL_SET:
        buf_putc(b, 0x21); leb_u(b, (uint64_t)at_i(ctx, op, MLIR_KEY_LOCAL_IDX)); return true;
    case OP_TYPE_WASMSTACK_LOCAL_TEE:
        buf_putc(b, 0x22); leb_u(b, (uint64_t)at_i(ctx, op, MLIR_KEY_LOCAL_IDX)); return true;

    case OP_TYPE_WASMSTACK_GLOBAL_GET: {
        uint32_t gidx = (uint32_t)at_i(ctx, op, MLIR_KEY_GLOBAL_IDX);
        buf_putc(b, 0x23);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, gidx);
        if (gidx == 0) emfunc_add_reloc(F, R_WASM_GLOBAL_INDEX_LEB, off, sp_sym_idx);
        return true;
    }
    case OP_TYPE_WASMSTACK_GLOBAL_SET: {
        uint32_t gidx = (uint32_t)at_i(ctx, op, MLIR_KEY_GLOBAL_IDX);
        buf_putc(b, 0x24);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, gidx);
        if (gidx == 0) emfunc_add_reloc(F, R_WASM_GLOBAL_INDEX_LEB, off, sp_sym_idx);
        return true;
    }

    case OP_TYPE_WASMSTACK_CONST: {
        int64_t iv = at_i(ctx, op, MLIR_KEY_VALUE);
        if (valtype == WT_I32)      { buf_putc(b, 0x41); leb_s(b, (int32_t)iv); }
        else if (valtype == WT_I64) { buf_putc(b, 0x42); leb_s(b, iv); }
        else if (valtype == WT_F32) {
            buf_putc(b, 0x43);
            uint32_t bits = (uint32_t)(uint64_t)iv;
            for (int i = 0; i < 4; i++) buf_putc(b, (uint8_t)(bits >> (8*i)));
        }
        else if (valtype == WT_F64) {
            buf_putc(b, 0x44);
            uint64_t bits = (uint64_t)iv;
            for (int i = 0; i < 8; i++) buf_putc(b, (uint8_t)(bits >> (8*i)));
        }
        else { fprintf(stderr, "wasm-emit: const of unsupported valtype\n"); return false; }
        return true;
    }

    case OP_TYPE_WASMSTACK_ADD:
        if (valtype == WT_I32)      buf_putc(b, 0x6a);
        else if (valtype == WT_I64) buf_putc(b, 0x7c);
        else { fprintf(stderr, "wasm-emit: add of unsupported valtype\n"); return false; }
        return true;

    case OP_TYPE_WASMSTACK_SUB:
        if (valtype == WT_I32)      buf_putc(b, 0x6b);
        else if (valtype == WT_I64) buf_putc(b, 0x7d);
        else { fprintf(stderr, "wasm-emit: sub of unsupported valtype\n"); return false; }
        return true;

    case OP_TYPE_WASMSTACK_LOAD: {
        uint32_t mem_size = (uint32_t)at_i(ctx, op, MLIR_KEY_MEM_SIZE_BYTES);
        uint32_t mem_align = (uint32_t)at_i(ctx, op, MLIR_KEY_MEMORY_ALIGN_LOG2);
        uint32_t mem_off = (uint32_t)at_i(ctx, op, MLIR_KEY_MEMORY_OFFSET);
        uint8_t opc;
        if (valtype == WT_I32) {
            // For sub-i32 loads we always use the zero-extending opcode
            // (i32.load8_u / i32.load16_u). Sign extension is then handled
            // explicitly by the following `llvm.sext` lowering, which
            // emits an `i32.extend8_s` / `i32.extend16_s`. Defaulting to
            // zero-extension is necessary so that a `load i8` followed
            // by `llvm.zext i8 to i32` (e.g. for `uint8_t` operands)
            // produces the correct zero-extended value — the `zext` is a
            // no-op at the wasmssa level because wasm has no sub-i32
            // result type.
            if (mem_size == 4)      opc = 0x28;
            else if (mem_size == 2) opc = 0x2f;
            else if (mem_size == 1) opc = 0x2d;
            else return false;
        } else if (valtype == WT_I64) {
            opc = 0x29;
        } else if (valtype == WT_F32) opc = 0x2a;
        else if (valtype == WT_F64)   opc = 0x2b;
        else return false;
        buf_putc(b, opc);
        leb_u(b, mem_align);
        leb_u(b, mem_off);
        return true;
    }
    case OP_TYPE_WASMSTACK_STORE: {
        uint32_t mem_size = (uint32_t)at_i(ctx, op, MLIR_KEY_MEM_SIZE_BYTES);
        uint32_t mem_align = (uint32_t)at_i(ctx, op, MLIR_KEY_MEMORY_ALIGN_LOG2);
        uint32_t mem_off = (uint32_t)at_i(ctx, op, MLIR_KEY_MEMORY_OFFSET);
        uint8_t opc;
        if (valtype == WT_I32) {
            if (mem_size == 4)      opc = 0x36;
            else if (mem_size == 2) opc = 0x3b;
            else if (mem_size == 1) opc = 0x3a;
            else return false;
        } else if (valtype == WT_I64) {
            opc = 0x37;
        } else if (valtype == WT_F32) opc = 0x38;
        else if (valtype == WT_F64)   opc = 0x39;
        else return false;
        buf_putc(b, opc);
        leb_u(b, mem_align);
        leb_u(b, mem_off);
        return true;
    }

    case OP_TYPE_WASMSTACK_RETURN:
        buf_putc(b, 0x0f); return true;

    case OP_TYPE_WASMSTACK_EXTEND_I32_S:
        buf_putc(b, 0xac); return true;

    case OP_TYPE_WASMSTACK_BINOP:
    case OP_TYPE_WASMSTACK_UNOP:
        buf_putc(b, (uint8_t)at_i(ctx, op, MLIR_KEY_WASM_OPCODE)); return true;

    case OP_TYPE_WASMSTACK_BLOCK:
        buf_putc(b, 0x02); buf_putc(b, valtype ? valtype : 0x40); return true;
    case OP_TYPE_WASMSTACK_LOOP:
        buf_putc(b, 0x03); buf_putc(b, valtype ? valtype : 0x40); return true;
    case OP_TYPE_WASMSTACK_IF:
        buf_putc(b, 0x04); buf_putc(b, valtype ? valtype : 0x40); return true;
    case OP_TYPE_WASMSTACK_ELSE:
        buf_putc(b, 0x05); return true;
    case OP_TYPE_WASMSTACK_END:
        buf_putc(b, 0x0b); return true;
    case OP_TYPE_WASMSTACK_BR:
        buf_putc(b, 0x0c); leb_u(b, (uint64_t)at_i(ctx, op, MLIR_KEY_DEPTH)); return true;
    case OP_TYPE_WASMSTACK_BR_IF:
        buf_putc(b, 0x0d); leb_u(b, (uint64_t)at_i(ctx, op, MLIR_KEY_DEPTH)); return true;
    case OP_TYPE_WASMSTACK_UNREACHABLE:
        buf_putc(b, 0x00); return true;
    case OP_TYPE_WASMSTACK_SELECT:
        buf_putc(b, 0x1b); return true;
    case OP_TYPE_WASMSTACK_EQZ:
        buf_putc(b, valtype == WT_I64 ? 0x50 : 0x45); return true;

    case OP_TYPE_WASMSTACK_CALL: {
        string tgt = at_s(ctx, op, MLIR_KEY_TARGET);
        int idx = find_func_by_name(fmap, tgt);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: call to unknown function '%.*s'\n",
                    (int)tgt.size, tgt.str);
            return false;
        }
        buf_putc(b, 0x10);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, all_funcs[idx].func_index);
        emfunc_add_reloc(F, R_WASM_FUNCTION_INDEX_LEB, off,
                         all_funcs[idx].sym_index);
        return true;
    }

    case OP_TYPE_WASMSTACK_ADDRESSOF: {
        string tgt = at_s(ctx, op, MLIR_KEY_TARGET);
        int idx = find_global_by_name(gmap, tgt);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: addressof unknown global '%.*s'\n",
                    (int)tgt.size, tgt.str);
            return false;
        }
        buf_putc(b, 0x41);
        uint32_t off = (uint32_t)b->len;
        leb_s_pad5(b, 0);
        emfunc_add_reloc(F, R_WASM_MEMORY_ADDR_SLEB, off,
                         global_sym_base + (uint32_t)idx);
        return true;
    }

    case OP_TYPE_WASMSTACK_FUNC_ADDR: {
        string tgt = at_s(ctx, op, MLIR_KEY_TARGET);
        int idx = find_func_by_name(fmap, tgt);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: func_addr unknown function '%.*s'\n",
                    (int)tgt.size, tgt.str);
            return false;
        }
        buf_putc(b, 0x41);
        uint32_t off = (uint32_t)b->len;
        leb_s_pad5(b, 0);
        emfunc_add_reloc(F, R_WASM_TABLE_INDEX_SLEB, off,
                         all_funcs[idx].sym_index);
        return true;
    }

    case OP_TYPE_WASMSTACK_CALL_INDIRECT: {
        size_t snp, snr;
        uint8_t *sp = hex_decode(at_s(ctx, op, MLIR_KEY_SIG_PARAMS), &snp);
        uint8_t *sr = hex_decode(at_s(ctx, op, MLIR_KEY_SIG_RESULTS), &snr);
        uint32_t tidx = sig_intern(sigs, sp, snp, sr, snr);
        free(sp); free(sr);
        buf_putc(b, 0x11);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, tidx);
        emfunc_add_reloc(F, R_WASM_TYPE_INDEX_LEB, off, tidx);
        buf_putc(b, 0x00);
        return true;
    }

    case OP_TYPE_WASMSTACK_MEMORY_SIZE:
        // wasm `memory.size` (0x3F) takes a single immediate memory
        // index byte (always 0 for the default memory) and returns
        // the current size in pages on the stack.
        buf_putc(b, 0x3f); buf_putc(b, 0x00); return true;

    case OP_TYPE_WASMSTACK_MEMORY_GROW:
        // wasm `memory.grow` (0x40) pops a page count and pushes
        // either the previous size in pages or -1.
        buf_putc(b, 0x40); buf_putc(b, 0x00); return true;

    default: {
        string nm = MLIR_GetOpTypeName(ctx, t);
        fprintf(stderr, "wasm-emit: unsupported wasmstack op '%.*s' (enum=%d)\n",
                (int)nm.size, nm.str, (int)t);
        return false;
    }
    }
}

// =============================================================================
// Function body builder: locals prelude + ops + 0x0b end.
// =============================================================================
static bool build_body(MLIR_Context *ctx, EmFunc *F, uint32_t sp_sym_idx,
                       EmFunc *all_funcs, const NameMap *fmap,
                       const NameMap *gmap,
                       uint32_t global_sym_base, SigTab *sigs) {
    // Walk MLIR ops within the func op's body region/block.
    if (MLIR_GetOpNumRegions(ctx, F->src_op) < 1) return false;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, F->src_op, 0);
    if (MLIR_GetRegionNumBlocks(ctx, r) < 1) return false;
    MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, r, 0);
    size_t nops = MLIR_GetBlockNumOps(ctx, blk);

    Buf insns = {0};
    Buf saved = F->body;
    F->body = insns;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blk, i);
        MLIR_OpType t = MLIR_GetOpType(ctx, op);
        uint8_t valtype = (uint8_t)at_i(ctx, op, MLIR_KEY_VALTYPE);
        if (!encode_op_typed(ctx, F, op, t, valtype, sp_sym_idx, all_funcs, fmap,
                             gmap, global_sym_base, sigs)) {
            buf_free(&F->body);
            F->body = saved;
            return false;
        }
    }
    buf_putc(&F->body, 0x0b);  // end
    Buf body_insns = F->body;
    F->body = saved;

    // Locals prelude.
    Buf prelude = {0};
    {
        size_t i = 0, groups = 0;
        while (i < F->n_locals) {
            uint8_t t = F->local_types[i];
            size_t j = i + 1;
            while (j < F->n_locals && F->local_types[j] == t) j++;
            groups++; i = j;
        }
        leb_u(&prelude, groups);
        i = 0;
        while (i < F->n_locals) {
            uint8_t t = F->local_types[i];
            size_t j = i + 1;
            while (j < F->n_locals && F->local_types[j] == t) j++;
            leb_u(&prelude, (uint32_t)(j - i));
            buf_putc(&prelude, t);
            i = j;
        }
    }

    buf_append(&F->body, prelude.data, prelude.len);
    uint32_t shift = (uint32_t)prelude.len;
    buf_append(&F->body, body_insns.data, body_insns.len);
    for (size_t i = 0; i < F->n_relocs; i++) {
        F->relocs[i].body_offset += shift;
    }
    buf_free(&prelude);
    buf_free(&body_insns);
    return true;
}

// =============================================================================
// Fused wasmssa -> bytecode body builder (avoids materializing wasmstack ops).
// =============================================================================
typedef struct {
    MLIR_ValueHandle *keys;
    uint32_t *vals;
    size_t n, cap;
} SsaVMap;

static size_t ssavmap_hash(MLIR_ValueHandle k) {
    size_t h = (size_t)k;
    h ^= h >> 15;
    h *= 2654435761u;
    h ^= h >> 13;
    return h;
}
static void ssavmap_grow(SsaVMap *m) {
    size_t ncap = m->cap ? m->cap * 2 : 64;
    MLIR_ValueHandle *nk = (MLIR_ValueHandle *)calloc(ncap, sizeof(*nk));
    uint32_t *nv = (uint32_t *)malloc(ncap * sizeof(*nv));
    size_t mask = ncap - 1;
    for (size_t i = 0; i < m->cap; i++) {
        if (m->keys[i] == MLIR_INVALID_HANDLE) continue;
        size_t j = ssavmap_hash(m->keys[i]) & mask;
        while (nk[j] != MLIR_INVALID_HANDLE) j = (j + 1) & mask;
        nk[j] = m->keys[i];
        nv[j] = m->vals[i];
    }
    free(m->keys); free(m->vals);
    m->keys = nk; m->vals = nv; m->cap = ncap;
}
static void ssavmap_set(SsaVMap *m, MLIR_ValueHandle k, uint32_t v) {
    if ((m->n + 1) * 4 >= m->cap * 3) ssavmap_grow(m);
    size_t mask = m->cap - 1;
    size_t i = ssavmap_hash(k) & mask;
    while (m->keys[i] != MLIR_INVALID_HANDLE) {
        if (m->keys[i] == k) return;
        i = (i + 1) & mask;
    }
    m->keys[i] = k; m->vals[i] = v; m->n++;
}
static int ssavmap_get(const SsaVMap *m, MLIR_ValueHandle k, uint32_t *out) {
    if (m->cap == 0) return 0;
    size_t mask = m->cap - 1;
    size_t i = ssavmap_hash(k) & mask;
    while (m->keys[i] != MLIR_INVALID_HANDLE) {
        if (m->keys[i] == k) { *out = m->vals[i]; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

typedef struct {
    MLIR_OpType op_type;
    uint32_t   *result_locals;
    size_t      n_results;
    uint32_t   *entry_arg_locals;
    size_t      n_entry_args;
} SsaFrame;

typedef struct {
    SsaFrame *data;
    size_t    n, cap;
} SsaRStack;

typedef struct {
    uint8_t *types;
    size_t   n, cap;
    size_t   n_params;
} SsaLocals;

typedef struct {
    MLIR_Context *ctx;
    Arena        *arena;
    EmFunc       *F;
    uint32_t      sp_sym_idx;
    EmFunc       *all_funcs;
    const NameMap *fmap;
    const NameMap *gmap;
    uint32_t      global_sym_base;
    SigTab       *sigs;
    SsaVMap      *vmap;
    SsaLocals    *L;
    SsaRStack    *rs;
    bool         *need_indirect_table;
    uint32_t    **addr_taken;
    size_t       *n_addr_taken;
} SsaEmitCtx;

static uint32_t ssalocals_add(SsaLocals *L, uint8_t vt) {
    if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 32;
        L->types = (uint8_t *)realloc(L->types, L->cap);
    }
    L->types[L->n] = vt;
    return (uint32_t)(L->n_params + L->n++);
}
static uint8_t ssa_val_to_vt(MLIR_Context *ctx, MLIR_ValueHandle v) {
    MLIR_TypeHandle ty = MLIR_GetValueType(ctx, v);
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    uint32_t w = MLIR_GetTypeWidth(ctx, ty);
    if (k == MLIR_TYPE_KIND_FLOAT && w == 32) return WT_F32;
    if (k == MLIR_TYPE_KIND_FLOAT && w == 64) return WT_F64;
    if (k == MLIR_TYPE_KIND_INTEGER && w == 64) return WT_I64;
    return WT_I32;
}
static MLIR_OpType ssa_to_stack_op(MLIR_OpType t) {
    switch (t) {
    case OP_TYPE_WASMSSA_CONST:         return OP_TYPE_WASMSTACK_CONST;
    case OP_TYPE_WASMSSA_ADD:           return OP_TYPE_WASMSTACK_ADD;
    case OP_TYPE_WASMSSA_SUB:           return OP_TYPE_WASMSTACK_SUB;
    case OP_TYPE_WASMSSA_BINOP:         return OP_TYPE_WASMSTACK_BINOP;
    case OP_TYPE_WASMSSA_UNOP:          return OP_TYPE_WASMSTACK_UNOP;
    case OP_TYPE_WASMSSA_LOAD:          return OP_TYPE_WASMSTACK_LOAD;
    case OP_TYPE_WASMSSA_STORE:         return OP_TYPE_WASMSTACK_STORE;
    case OP_TYPE_WASMSSA_GLOBAL_GET:    return OP_TYPE_WASMSTACK_GLOBAL_GET;
    case OP_TYPE_WASMSSA_GLOBAL_SET:    return OP_TYPE_WASMSTACK_GLOBAL_SET;
    case OP_TYPE_WASMSSA_EXTEND_I32_S:  return OP_TYPE_WASMSTACK_EXTEND_I32_S;
    case OP_TYPE_WASMSSA_RETURN:        return OP_TYPE_WASMSTACK_RETURN;
    case OP_TYPE_WASMSSA_CALL:          return OP_TYPE_WASMSTACK_CALL;
    case OP_TYPE_WASMSSA_BR:            return OP_TYPE_WASMSTACK_BR;
    case OP_TYPE_WASMSSA_BR_IF:         return OP_TYPE_WASMSTACK_BR_IF;
    case OP_TYPE_WASMSSA_SELECT:        return OP_TYPE_WASMSTACK_SELECT;
    case OP_TYPE_WASMSSA_EQZ:           return OP_TYPE_WASMSTACK_EQZ;
    case OP_TYPE_WASMSSA_ADDRESSOF:     return OP_TYPE_WASMSTACK_ADDRESSOF;
    case OP_TYPE_WASMSSA_FUNC_ADDR:     return OP_TYPE_WASMSTACK_FUNC_ADDR;
    case OP_TYPE_WASMSSA_CALL_INDIRECT: return OP_TYPE_WASMSTACK_CALL_INDIRECT;
    case OP_TYPE_WASMSSA_MEMORY_SIZE:   return OP_TYPE_WASMSTACK_MEMORY_SIZE;
    case OP_TYPE_WASMSSA_MEMORY_GROW:   return OP_TYPE_WASMSTACK_MEMORY_GROW;
    default: return (MLIR_OpType)0;
    }
}

static bool ssa_emit_block(SsaEmitCtx *E, MLIR_BlockHandle src_blk);

static bool ssa_emit_region_op(SsaEmitCtx *E, MLIR_OpHandle bo, MLIR_OpType t) {
    MLIR_Context *ctx = E->ctx;
    Buf *b = &E->F->body;
    size_t n_results = MLIR_GetOpNumResults(ctx, bo);
    size_t n_operands = MLIR_GetOpNumOperands(ctx, bo);

    SsaFrame fr = {0};
    fr.op_type = t;
    fr.n_results = n_results;
    if (n_results) {
        fr.result_locals = (uint32_t *)arena_alloc(E->arena, n_results * sizeof(uint32_t));
        for (size_t i = 0; i < n_results; i++) {
            fr.result_locals[i] = ssalocals_add(E->L, ssa_val_to_vt(ctx, MLIR_GetOpResult(ctx, bo, i)));
        }
    }
    if (t == OP_TYPE_WASMSSA_LOOP) {
        fr.n_entry_args = n_operands;
        if (n_operands) {
            fr.entry_arg_locals = (uint32_t *)arena_alloc(E->arena, n_operands * sizeof(uint32_t));
        }
        if (MLIR_GetOpNumRegions(ctx, bo) < 1) return false;
        MLIR_BlockHandle entry = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 0), 0);
        if (MLIR_GetBlockNumArgs(ctx, entry) != n_operands) return false;
        for (size_t i = 0; i < n_operands; i++) {
            MLIR_ValueHandle ba = MLIR_GetBlockArg(ctx, entry, i);
            fr.entry_arg_locals[i] = ssalocals_add(E->L, ssa_val_to_vt(ctx, ba));
            uint32_t li;
            if (!ssavmap_get(E->vmap, MLIR_GetOpOperand(ctx, bo, i), &li)) return false;
            buf_putc(b, 0x20); leb_u(b, li);
            buf_putc(b, 0x21); leb_u(b, fr.entry_arg_locals[i]);
            ssavmap_set(E->vmap, ba, fr.entry_arg_locals[i]);
        }
    } else if (t == OP_TYPE_WASMSSA_IF) {
        if (n_operands != 1) return false;
        uint32_t cli;
        if (!ssavmap_get(E->vmap, MLIR_GetOpOperand(ctx, bo, 0), &cli)) return false;
        buf_putc(b, 0x20); leb_u(b, cli);
    }

    buf_putc(b, (t == OP_TYPE_WASMSSA_BLOCK) ? 0x02 : (t == OP_TYPE_WASMSSA_LOOP) ? 0x03 : 0x04);
    buf_putc(b, 0x40);

    if (E->rs->n == E->rs->cap) {
        E->rs->cap = E->rs->cap ? E->rs->cap * 2 : 16;
        E->rs->data = (SsaFrame *)realloc(E->rs->data, E->rs->cap * sizeof(SsaFrame));
    }
    E->rs->data[E->rs->n++] = fr;

    if (t == OP_TYPE_WASMSSA_IF) {
        if (MLIR_GetOpNumRegions(ctx, bo) < 1) { E->rs->n--; return false; }
        MLIR_BlockHandle then_blk = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 0), 0);
        if (!ssa_emit_block(E, then_blk)) { E->rs->n--; return false; }
        bool has_else = MLIR_GetOpNumRegions(ctx, bo) >= 2 &&
                        MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, bo, 1)) > 0;
        if (has_else) {
            buf_putc(b, 0x05);
            MLIR_BlockHandle else_blk = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 1), 0);
            if (!ssa_emit_block(E, else_blk)) { E->rs->n--; return false; }
        }
    } else {
        if (MLIR_GetOpNumRegions(ctx, bo) < 1) { E->rs->n--; return false; }
        MLIR_BlockHandle inner = MLIR_GetRegionBlock(ctx, MLIR_GetOpRegion(ctx, bo, 0), 0);
        if (!ssa_emit_block(E, inner)) { E->rs->n--; return false; }
    }
    E->rs->n--;

    buf_putc(b, 0x0b);
    for (size_t i = 0; i < n_results; i++) {
        ssavmap_set(E->vmap, MLIR_GetOpResult(ctx, bo, i), fr.result_locals[i]);
    }
    return true;
}

static bool ssa_emit_block(SsaEmitCtx *E, MLIR_BlockHandle src_blk) {
    MLIR_Context *ctx = E->ctx;
    Buf *b = &E->F->body;
    size_t n_body = MLIR_GetBlockNumOps(ctx, src_blk);
    for (size_t i = 0; i < n_body; i++) {
        MLIR_OpHandle bo = MLIR_GetBlockOp(ctx, src_blk, i);
        MLIR_OpType t = MLIR_GetOpType(ctx, bo);

        if (t == OP_TYPE_WASMSSA_BLOCK || t == OP_TYPE_WASMSSA_LOOP || t == OP_TYPE_WASMSSA_IF) {
            if (!ssa_emit_region_op(E, bo, t)) return false;
            continue;
        }
        if (t == OP_TYPE_WASMSSA_BLOCK_RETURN) {
            SsaFrame *top = E->rs->n ? &E->rs->data[E->rs->n - 1] : NULL;
            if (!top) return false;
            size_t no = MLIR_GetOpNumOperands(ctx, bo);
            if (no != top->n_results) return false;
            for (size_t k = 0; k < no; k++) {
                uint32_t li;
                if (!ssavmap_get(E->vmap, MLIR_GetOpOperand(ctx, bo, k), &li)) return false;
                buf_putc(b, 0x20); leb_u(b, li);
                buf_putc(b, 0x21); leb_u(b, top->result_locals[k]);
            }
            continue;
        }
        if (t == OP_TYPE_WASMSSA_UNREACHABLE) {
            buf_putc(b, 0x00);
            continue;
        }
        if (t == OP_TYPE_WASMSSA_BR && MLIR_GetOpNumOperands(ctx, bo) > 0) {
            uint32_t depth = (uint32_t)at_i(ctx, bo, MLIR_KEY_DEPTH);
            if (depth >= E->rs->n) return false;
            SsaFrame *target = &E->rs->data[E->rs->n - 1 - depth];
            uint32_t *dst_locals = (target->op_type == OP_TYPE_WASMSSA_LOOP)
                                 ? target->entry_arg_locals : target->result_locals;
            size_t n_dst = (target->op_type == OP_TYPE_WASMSSA_LOOP)
                         ? target->n_entry_args : target->n_results;
            size_t no = MLIR_GetOpNumOperands(ctx, bo);
            if (no != n_dst) return false;
            for (size_t k = 0; k < no; k++) {
                uint32_t li;
                if (!ssavmap_get(E->vmap, MLIR_GetOpOperand(ctx, bo, k), &li)) return false;
                buf_putc(b, 0x20); leb_u(b, li);
                buf_putc(b, 0x21); leb_u(b, dst_locals[k]);
            }
            buf_putc(b, 0x0c); leb_u(b, depth);
            continue;
        }

        if (t == OP_TYPE_WASMSSA_CALL_INDIRECT) {
            *E->need_indirect_table = true;
        } else if (t == OP_TYPE_WASMSSA_FUNC_ADDR) {
            *E->need_indirect_table = true;
            string tgt = at_s(ctx, bo, MLIR_KEY_TARGET);
            int idx = find_func_by_name(E->fmap, tgt);
            if (idx >= 0) {
                uint32_t fi = E->all_funcs[idx].func_index;
                bool dup = false;
                for (size_t k = 0; k < *E->n_addr_taken; k++)
                    if ((*E->addr_taken)[k] == fi) { dup = true; break; }
                if (!dup) {
                    *E->addr_taken = (uint32_t *)realloc(
                        *E->addr_taken, (*E->n_addr_taken + 1) * sizeof(uint32_t));
                    (*E->addr_taken)[(*E->n_addr_taken)++] = fi;
                }
            }
        }

        size_t n_ops = MLIR_GetOpNumOperands(ctx, bo);
        for (size_t k = 0; k < n_ops; k++) {
            uint32_t li;
            if (!ssavmap_get(E->vmap, MLIR_GetOpOperand(ctx, bo, k), &li)) return false;
            buf_putc(b, 0x20); leb_u(b, li);
        }

        MLIR_OpType st = ssa_to_stack_op(t);
        if (!st) return false;
        uint8_t valtype = (uint8_t)at_i(ctx, bo, MLIR_KEY_VALTYPE);
        if (!encode_op_typed(ctx, E->F, bo, st, valtype, E->sp_sym_idx,
                             E->all_funcs, E->fmap, E->gmap,
                             E->global_sym_base, E->sigs)) {
            return false;
        }

        if (MLIR_GetOpNumResults(ctx, bo) > 0) {
            uint32_t li = ssalocals_add(E->L, valtype);
            buf_putc(b, 0x21); leb_u(b, li);
            ssavmap_set(E->vmap, MLIR_GetOpResult(ctx, bo, 0), li);
        }
    }
    return true;
}

static bool build_body_from_wasmssa(MLIR_Context *ctx, Arena *arena, EmFunc *F,
                                    size_t n_params, uint32_t sp_sym_idx,
                                    EmFunc *all_funcs, const NameMap *fmap,
                                    const NameMap *gmap,
                                    uint32_t global_sym_base, SigTab *sigs,
                                    SsaVMap *vmap, SsaRStack *rs, SsaLocals *L,
                                    bool *need_indirect_table,
                                    uint32_t **addr_taken, size_t *n_addr_taken) {
    L->n = 0;
    L->n_params = n_params;
    rs->n = 0;
    if (vmap->cap > 0) {
        memset(vmap->keys, 0, vmap->cap * sizeof(*vmap->keys));
        vmap->n = 0;
    }

    Buf insns = {0};
    Buf saved = F->body;
    F->body = insns;

    if (MLIR_GetOpNumRegions(ctx, F->src_op) >= 1) {
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, F->src_op, 0);
        if (MLIR_GetRegionNumBlocks(ctx, r) >= 1) {
            MLIR_BlockHandle src_blk = MLIR_GetRegionBlock(ctx, r, 0);
            size_t n_args = MLIR_GetBlockNumArgs(ctx, src_blk);
            for (size_t i = 0; i < n_args; i++) {
                ssavmap_set(vmap, MLIR_GetBlockArg(ctx, src_blk, i), (uint32_t)i);
            }
            SsaEmitCtx E = {
                ctx, arena, F, sp_sym_idx, all_funcs, fmap, gmap,
                global_sym_base, sigs, vmap, L, rs,
                need_indirect_table, addr_taken, n_addr_taken
            };
            if (!ssa_emit_block(&E, src_blk)) {
                buf_free(&F->body);
                F->body = saved;
                return false;
            }
        }
    }
    buf_putc(&F->body, 0x0b);
    Buf body_insns = F->body;
    F->body = saved;

    Buf prelude = {0};
    {
        size_t i = 0, groups = 0;
        while (i < L->n) {
            uint8_t t = L->types[i];
            size_t j = i + 1;
            while (j < L->n && L->types[j] == t) j++;
            groups++; i = j;
        }
        leb_u(&prelude, groups);
        i = 0;
        while (i < L->n) {
            uint8_t t = L->types[i];
            size_t j = i + 1;
            while (j < L->n && L->types[j] == t) j++;
            leb_u(&prelude, (uint32_t)(j - i));
            buf_putc(&prelude, t);
            i = j;
        }
    }
    buf_append(&F->body, prelude.data, prelude.len);
    uint32_t shift = (uint32_t)prelude.len;
    buf_append(&F->body, body_insns.data, body_insns.len);
    for (size_t i = 0; i < F->n_relocs; i++) {
        F->relocs[i].body_offset += shift;
    }
    buf_free(&prelude);
    buf_free(&body_insns);
    return true;
}

// =============================================================================
// Section serialization.
// =============================================================================
static void emit_section(Buf *out, uint8_t id, const Buf *payload) {
    buf_putc(out, id);
    leb_u(out, payload->len);
    buf_append(out, payload->data, payload->len);
}
static void emit_string(Buf *out, const char *s) {
    size_t n = strlen(s);
    leb_u(out, n);
    buf_append(out, s, n);
}

static void build_linking_section(EmFunc *funcs, size_t n_funcs,
                                  uint32_t sp_global_idx,
                                  const EmGlobal *globals, size_t n_globals,
                                  bool include_indirect_table,
                                  uint32_t indirect_table_idx,
                                  Buf *out) {
    Buf body = {0};
    leb_u(&body, LINK_VERSION);

    Buf sub = {0};
    leb_u(&sub, 1u + (uint32_t)n_funcs + (uint32_t)n_globals
                + (include_indirect_table ? 1u : 0u));

    buf_putc(&sub, SYM_GLOBAL);
    leb_u(&sub, SYMF_UNDEFINED);
    leb_u(&sub, sp_global_idx);

    for (size_t i = 0; i < n_funcs; i++) {
        EmFunc *f = &funcs[i];
        buf_putc(&sub, SYM_FUNCTION);
        // Imported functions are referenced by name (UNDEFINED).
        // Static / file-local C functions get BINDING_LOCAL +
        // VISIBILITY_HIDDEN so the wasm linker won't merge two
        // same-named statics from different TUs into one global
        // symbol (which causes signature-mismatch link errors).
        uint32_t flags;
        if (f->imported) {
            flags = SYMF_UNDEFINED;
        } else if (f->internal) {
            flags = SYMF_BINDING_LOCAL | SYMF_VISIBILITY_HIDDEN;
        } else {
            flags = 0;
        }
        leb_u(&sub, flags);
        leb_u(&sub, f->func_index);
        if (!f->imported) emit_string(&sub, f->name);
    }

    for (size_t i = 0; i < n_globals; i++) {
        buf_putc(&sub, SYM_DATA);
        leb_u(&sub, SYMF_BINDING_LOCAL | SYMF_VISIBILITY_HIDDEN);
        emit_string(&sub, globals[i].name);
        leb_u(&sub, (uint32_t)i);
        leb_u(&sub, 0);
        leb_u(&sub, globals[i].size);
    }

    if (include_indirect_table) {
        buf_putc(&sub, SYM_TABLE);
        leb_u(&sub, SYMF_NO_STRIP | SYMF_UNDEFINED);
        leb_u(&sub, indirect_table_idx);
    }

    buf_putc(&body, LINK_SUB_SYMBOL_TABLE);
    leb_u(&body, sub.len);
    buf_append(&body, sub.data, sub.len);
    buf_free(&sub);

    if (n_globals) {
        Buf seg = {0};
        leb_u(&seg, (uint32_t)n_globals);
        for (size_t i = 0; i < n_globals; i++) {
            const char *nm = globals[i].name && globals[i].name[0]
                              ? globals[i].name : ".rodata";
            char *full = (char *)malloc(strlen(nm) + 16);
            sprintf(full, "%s%s", globals[i].is_const ? ".rodata." : ".data.", nm);
            emit_string(&seg, full);
            free(full);
            leb_u(&seg, globals[i].align_pow);
            leb_u(&seg, 0);
        }
        buf_putc(&body, LINK_SUB_SEGMENT_INFO);
        leb_u(&body, seg.len);
        buf_append(&body, seg.data, seg.len);
        buf_free(&seg);
    }

    Buf payload = {0};
    emit_string(&payload, "linking");
    buf_append(&payload, body.data, body.len);
    buf_free(&body);

    emit_section(out, SEC_CUSTOM, &payload);
    buf_free(&payload);
}

static bool reloc_has_addend(uint8_t t) {
    return t == R_WASM_MEMORY_ADDR_LEB ||
           t == R_WASM_MEMORY_ADDR_SLEB ||
           t == R_WASM_MEMORY_ADDR_I32;
}

static void build_reloc_code_section(EmFunc *funcs, size_t n_funcs,
                                     uint8_t code_section_index,
                                     const uint32_t *func_body_off,
                                     Buf *out) {
    typedef struct { uint8_t type; uint32_t off; uint32_t sym; int32_t addend; } R;
    R *all = NULL; size_t na = 0, ca = 0;
    size_t k = 0;
    for (size_t i = 0; i < n_funcs; i++) {
        EmFunc *f = &funcs[i];
        if (f->imported) continue;
        for (size_t j = 0; j < f->n_relocs; j++) {
            if (na == ca) { ca = ca ? ca * 2 : 8; all = (R *)realloc(all, ca * sizeof(R)); }
            all[na++] = (R){
                f->relocs[j].type,
                func_body_off[k] + f->relocs[j].body_offset,
                f->relocs[j].sym_idx,
                f->relocs[j].addend,
            };
        }
        k++;
    }

    Buf body = {0};
    leb_u(&body, code_section_index);
    leb_u(&body, na);
    for (size_t i = 0; i < na; i++) {
        buf_putc(&body, all[i].type);
        leb_u(&body, all[i].off);
        leb_u(&body, all[i].sym);
        if (reloc_has_addend(all[i].type)) leb_s(&body, all[i].addend);
    }
    free(all);

    Buf payload = {0};
    emit_string(&payload, "reloc.CODE");
    buf_append(&payload, body.data, body.len);
    buf_free(&body);
    emit_section(out, SEC_CUSTOM, &payload);
    buf_free(&payload);
}

// =============================================================================
// Top-level translator. Walks either a wasmstack or wasmssa MLIR module.
// =============================================================================
static string wasm_module_to_bin_impl(MLIR_Context *ctx,
                                      MLIR_OpHandle mod,
                                      bool from_ssa) {
    string fail = {0};
    if (!mod) return fail;

    if (MLIR_GetOpNumRegions(ctx, mod) < 1) return fail;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, mod, 0);
    if (MLIR_GetRegionNumBlocks(ctx, mr) < 1) return fail;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t n_top = MLIR_GetBlockNumOps(ctx, mb);

    MLIR_OpType ty_imp_func   = from_ssa ? OP_TYPE_WASMSSA_IMPORT_FUNC   : OP_TYPE_WASMSTACK_IMPORT_FUNC;
    MLIR_OpType ty_func       = from_ssa ? OP_TYPE_WASMSSA_FUNC          : OP_TYPE_WASMSTACK_FUNC;
    MLIR_OpType ty_imp_global = from_ssa ? OP_TYPE_WASMSSA_IMPORT_GLOBAL : OP_TYPE_WASMSTACK_IMPORT_GLOBAL;

    SigTab sigs = {0};

    // ---- Materialize EmFuncs: imports first, then defs. -------------------
    EmFunc *funcs = NULL;
    size_t n_funcs = 0;
    size_t *func_nparams = NULL;

    // Pass A: imports.
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, ti);
        if (MLIR_GetOpType(ctx, op) != ty_imp_func) continue;
        size_t np, nr;
        uint8_t *p = hex_decode(at_s(ctx, op, MLIR_KEY_PARAM_TYPES), &np);
        uint8_t *r = hex_decode(at_s(ctx, op, MLIR_KEY_RESULT_TYPES), &nr);
        funcs = (EmFunc *)realloc(funcs, (n_funcs + 1) * sizeof(EmFunc));
        func_nparams = (size_t *)realloc(func_nparams, (n_funcs + 1) * sizeof(size_t));
        memset(&funcs[n_funcs], 0, sizeof(EmFunc));
        func_nparams[n_funcs] = np;
        funcs[n_funcs].name = strdup_str(at_s(ctx, op, MLIR_KEY_SYM_NAME));
        funcs[n_funcs].sig = sig_intern(&sigs, p, np, r, nr);
        funcs[n_funcs].imported = true;
        funcs[n_funcs].func_index = (uint32_t)n_funcs;
        string imod = at_s(ctx, op, MLIR_KEY_IMPORT_MODULE);
        if (imod.size) funcs[n_funcs].import_module = strdup_str(imod);
        string inm = at_s(ctx, op, MLIR_KEY_IMPORT_NAME);
        if (inm.size) funcs[n_funcs].import_name = strdup_str(inm);
        free(p); free(r);
        n_funcs++;
    }
    // Pass B: defs.
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, ti);
        if (MLIR_GetOpType(ctx, op) != ty_func) continue;
        size_t np, nr;
        uint8_t *p = hex_decode(at_s(ctx, op, MLIR_KEY_PARAM_TYPES), &np);
        uint8_t *r = hex_decode(at_s(ctx, op, MLIR_KEY_RESULT_TYPES), &nr);
        size_t nl = 0;
        uint8_t *ll = from_ssa ? NULL : hex_decode(at_s(ctx, op, MLIR_KEY_LOCAL_TYPES), &nl);
        funcs = (EmFunc *)realloc(funcs, (n_funcs + 1) * sizeof(EmFunc));
        func_nparams = (size_t *)realloc(func_nparams, (n_funcs + 1) * sizeof(size_t));
        memset(&funcs[n_funcs], 0, sizeof(EmFunc));
        func_nparams[n_funcs] = np;
        funcs[n_funcs].name = strdup_str(at_s(ctx, op, MLIR_KEY_SYM_NAME));
        funcs[n_funcs].sig = sig_intern(&sigs, p, np, r, nr);
        funcs[n_funcs].imported = false;
        funcs[n_funcs].exported = at_b(ctx, op, MLIR_KEY_EXPORTED);
        funcs[n_funcs].internal = at_b(ctx, op, MLIR_KEY_INTERNAL);
        funcs[n_funcs].func_index = (uint32_t)n_funcs;
        funcs[n_funcs].src_op = op;
        funcs[n_funcs].local_types = ll;
        funcs[n_funcs].n_locals = nl;
        string en = at_s(ctx, op, MLIR_KEY_EXPORT_NAME);
        if (en.size) funcs[n_funcs].export_name = strdup_str(en);
        free(p); free(r);
        n_funcs++;
    }

    // ---- Materialize EmGlobals --------------------------------------------
    EmGlobal *globals = NULL;
    size_t n_globals = 0;
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, ti);
        if (MLIR_GetOpType(ctx, op) != ty_imp_global) continue;
        globals = (EmGlobal *)realloc(globals, (n_globals + 1) * sizeof(EmGlobal));
        memset(&globals[n_globals], 0, sizeof(EmGlobal));
        globals[n_globals].name = strdup_str(at_s(ctx, op, MLIR_KEY_SYM_NAME));
        globals[n_globals].size = (uint32_t)at_i(ctx, op, MLIR_KEY_SIZE);
        globals[n_globals].align_pow = (uint32_t)at_i(ctx, op, MLIR_KEY_ALIGN_POW);
        globals[n_globals].is_const = at_b(ctx, op, MLIR_KEY_IS_CONST);
        string id = at_s(ctx, op, MLIR_KEY_INIT_DATA);
        if (id.size) {
            globals[n_globals].data = (uint8_t *)malloc(id.size);
            memcpy(globals[n_globals].data, id.str, id.size);
        }
        globals[n_globals].relocs = parse_relocs(at_s(ctx, op, MLIR_KEY_RELOCS),
                                                 &globals[n_globals].n_relocs);
        n_globals++;
    }

    NameMap fmap = {0}, gmap = {0};
    nm_init(&fmap, n_funcs);
    for (size_t i = 0; i < n_funcs; i++) nm_put(&fmap, funcs[i].name, (int)i);
    nm_init(&gmap, n_globals);
    for (size_t i = 0; i < n_globals; i++) nm_put(&gmap, globals[i].name, (int)i);

    uint32_t sp_sym = 0;
    for (size_t i = 0; i < n_funcs; i++) funcs[i].sym_index = (uint32_t)(i + 1);
    uint32_t global_sym_base = 1u + (uint32_t)n_funcs;

    // ---- Detect address-taken funcs and emit defined function bodies -------
    bool need_indirect_table = false;
    uint32_t *addr_taken = NULL;
    size_t n_addr_taken = 0;
    if (!from_ssa) {
        for (size_t i = 0; i < n_funcs; i++) {
            if (funcs[i].imported) continue;
            MLIR_RegionHandle rg = MLIR_GetOpRegion(ctx, funcs[i].src_op, 0);
            MLIR_BlockHandle  bl = MLIR_GetRegionBlock(ctx, rg, 0);
            size_t nb = MLIR_GetBlockNumOps(ctx, bl);
            for (size_t j = 0; j < nb; j++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, bl, j);
                MLIR_OpType t = MLIR_GetOpType(ctx, op);
                if (t == OP_TYPE_WASMSTACK_CALL_INDIRECT) {
                    need_indirect_table = true;
                } else if (t == OP_TYPE_WASMSTACK_FUNC_ADDR) {
                    need_indirect_table = true;
                    string tgt = at_s(ctx, op, MLIR_KEY_TARGET);
                    int idx = find_func_by_name(&fmap, tgt);
                    if (idx < 0) continue;
                    bool dup = false;
                    for (size_t k = 0; k < n_addr_taken; k++)
                        if (addr_taken[k] == funcs[idx].func_index) { dup = true; break; }
                    if (!dup) {
                        addr_taken = (uint32_t *)realloc(
                            addr_taken, (n_addr_taken + 1) * sizeof(uint32_t));
                        addr_taken[n_addr_taken++] = funcs[idx].func_index;
                    }
                }
            }
        }
        for (size_t i = 0; i < n_funcs; i++) {
            if (funcs[i].imported) continue;
            if (!build_body(ctx, &funcs[i], sp_sym, funcs, &fmap,
                            &gmap, global_sym_base, &sigs)) {
                goto fail_after_funcs;
            }
        }
    } else {
        Arena *arena = MLIR_GetArenaAllocator(ctx);
        SsaVMap vmap = {0};
        SsaRStack rs = {0};
        SsaLocals L = {0};
        bool ok = true;
        for (size_t i = 0; i < n_funcs; i++) {
            if (funcs[i].imported) continue;
            if (!build_body_from_wasmssa(ctx, arena, &funcs[i], func_nparams[i],
                                         sp_sym, funcs, &fmap, &gmap,
                                         global_sym_base, &sigs,
                                         &vmap, &rs, &L,
                                         &need_indirect_table,
                                         &addr_taken, &n_addr_taken)) {
                ok = false;
                break;
            }
        }
        free(vmap.keys); free(vmap.vals);
        free(rs.data);
        free(L.types);
        if (!ok) goto fail_after_funcs;
    }

    // ---- Build sections -----------------------------------------------------
    Buf type_payload = {0};
    leb_u(&type_payload, sigs.n);
    for (size_t i = 0; i < sigs.n; i++) {
        buf_putc(&type_payload, 0x60);
        leb_u(&type_payload, sigs.e[i].nparams);
        buf_append(&type_payload, sigs.e[i].params, sigs.e[i].nparams);
        leb_u(&type_payload, sigs.e[i].nresults);
        buf_append(&type_payload, sigs.e[i].results, sigs.e[i].nresults);
    }

    Buf import_payload = {0};
    {
        uint32_t n_func_imports = 0;
        for (size_t i = 0; i < n_funcs; i++) if (funcs[i].imported) n_func_imports++;
        uint32_t n_imports_total = 2 + n_func_imports + (need_indirect_table ? 1u : 0u);
        leb_u(&import_payload, n_imports_total);

        emit_string(&import_payload, "env");
        emit_string(&import_payload, "__linear_memory");
        buf_putc(&import_payload, IMP_MEMORY);
        buf_putc(&import_payload, 0x00);
        leb_u(&import_payload, 0);

        emit_string(&import_payload, "env");
        emit_string(&import_payload, "__stack_pointer");
        buf_putc(&import_payload, IMP_GLOBAL);
        buf_putc(&import_payload, WT_I32);
        buf_putc(&import_payload, 0x01);

        for (size_t i = 0; i < n_funcs; i++) {
            if (!funcs[i].imported) continue;
            // Default to "env" but honor `__attribute__((__import_module__("...")))`
            // (e.g. `wasi_snapshot_preview1` for WASI imports).
            const char *mod = funcs[i].import_module ? funcs[i].import_module : "env";
            const char *nm  = funcs[i].import_name   ? funcs[i].import_name   : funcs[i].name;
            emit_string(&import_payload, mod);
            emit_string(&import_payload, nm);
            buf_putc(&import_payload, IMP_FUNC);
            leb_u(&import_payload, funcs[i].sig);
        }

        if (need_indirect_table) {
            emit_string(&import_payload, "env");
            emit_string(&import_payload, "__indirect_function_table");
            buf_putc(&import_payload, IMP_TABLE);
            buf_putc(&import_payload, 0x70);
            buf_putc(&import_payload, 0x00);
            leb_u(&import_payload, 1);
        }
    }

    Buf function_payload = {0};
    {
        uint32_t n_def = 0;
        for (size_t i = 0; i < n_funcs; i++) if (!funcs[i].imported) n_def++;
        leb_u(&function_payload, n_def);
        for (size_t i = 0; i < n_funcs; i++) {
            if (funcs[i].imported) continue;
            leb_u(&function_payload, funcs[i].sig);
        }
    }

    Buf code_payload = {0};
    uint32_t *body_offsets = NULL;
    {
        uint32_t n_def = 0;
        for (size_t i = 0; i < n_funcs; i++) if (!funcs[i].imported) n_def++;
        body_offsets = (uint32_t *)calloc(n_def ? n_def : 1, sizeof(uint32_t));
        leb_u(&code_payload, n_def);
        size_t k = 0;
        for (size_t i = 0; i < n_funcs; i++) {
            if (funcs[i].imported) continue;
            leb_u(&code_payload, funcs[i].body.len);
            body_offsets[k++] = (uint32_t)code_payload.len;
            buf_append(&code_payload, funcs[i].body.data, funcs[i].body.len);
        }
    }

    Buf img = {0};
    static const uint8_t magic[8] = {0,'a','s','m', 1,0,0,0};
    buf_append(&img, magic, 8);
    emit_section(&img, SEC_TYPE,     &type_payload);
    emit_section(&img, SEC_IMPORT,   &import_payload);
    emit_section(&img, SEC_FUNCTION, &function_payload);

    Buf elem_payload = {0};
    if (n_addr_taken) {
        leb_u(&elem_payload, 1);
        leb_u(&elem_payload, 0);
        buf_putc(&elem_payload, 0x41);
        leb_s(&elem_payload, 1);
        buf_putc(&elem_payload, 0x0b);
        leb_u(&elem_payload, (uint32_t)n_addr_taken);
        for (size_t i = 0; i < n_addr_taken; i++)
            leb_u(&elem_payload, addr_taken[i]);
        emit_section(&img, SEC_ELEMENT, &elem_payload);
    }

    uint8_t code_section_index = (uint8_t)(n_addr_taken ? 4 : 3);
    emit_section(&img, SEC_CODE,     &code_payload);
    uint8_t data_section_index = (uint8_t)(code_section_index + 1);

    Buf data_payload = {0};
    uint32_t *data_offsets = NULL;
    if (n_globals) {
        data_offsets = (uint32_t *)calloc(n_globals, sizeof(uint32_t));
        leb_u(&data_payload, (uint32_t)n_globals);
        for (size_t i = 0; i < n_globals; i++) {
            leb_u(&data_payload, 0);
            buf_putc(&data_payload, 0x41); leb_s(&data_payload, 0);
            buf_putc(&data_payload, 0x0b);
            leb_u(&data_payload, globals[i].size);
            data_offsets[i] = (uint32_t)data_payload.len;
            if (globals[i].size) buf_append(&data_payload, globals[i].data, globals[i].size);
        }
        emit_section(&img, SEC_DATA, &data_payload);
    }

    build_linking_section(funcs, n_funcs, /*sp_global_idx=*/0,
                          globals, n_globals,
                          need_indirect_table, /*indirect_table_idx=*/0,
                          &img);
    build_reloc_code_section(funcs, n_funcs, code_section_index, body_offsets, &img);

    if (n_globals) {
        size_t total = 0;
        for (size_t i = 0; i < n_globals; i++) total += globals[i].n_relocs;
        if (total) {
            Buf body = {0};
            leb_u(&body, data_section_index);
            leb_u(&body, (uint32_t)total);
            for (size_t i = 0; i < n_globals; i++) {
                for (size_t j = 0; j < globals[i].n_relocs; j++) {
                    int tidx = -1;
                    for (size_t k = 0; k < n_globals; k++) {
                        if (strcmp(globals[k].name, globals[i].relocs[j].target) == 0) {
                            tidx = (int)k; break;
                        }
                    }
                    if (tidx < 0) {
                        fprintf(stderr,
                                "wasm-emit: reloc.DATA target '%s' not found\n",
                                globals[i].relocs[j].target);
                        free(data_offsets);
                        buf_free(&data_payload);
                        buf_free(&body);
                        goto fail_after_funcs;
                    }
                    buf_putc(&body, R_WASM_MEMORY_ADDR_I32);
                    leb_u(&body, data_offsets[i] + globals[i].relocs[j].offset);
                    leb_u(&body, global_sym_base + (uint32_t)tidx);
                    leb_s(&body, globals[i].relocs[j].addend);
                }
            }
            Buf payload = {0};
            emit_string(&payload, "reloc.DATA");
            buf_append(&payload, body.data, body.len);
            buf_free(&body);
            emit_section(&img, SEC_CUSTOM, &payload);
            buf_free(&payload);
        }
    }
    free(data_offsets);
    buf_free(&data_payload);
    (void)data_section_index;

    free(body_offsets);
    buf_free(&type_payload);
    buf_free(&import_payload);
    buf_free(&function_payload);
    buf_free(&code_payload);
    buf_free(&elem_payload);

    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *out = (char *)arena_alloc(arena, img.len);
    memcpy(out, img.data, img.len);
    string result = { out, img.len };
    buf_free(&img);

    for (size_t i = 0; i < n_funcs; i++) {
        free(funcs[i].name);
        free(funcs[i].local_types);
        buf_free(&funcs[i].body);
        free(funcs[i].relocs);
    }
    free(funcs);
    free(func_nparams);
    free(addr_taken);
    nm_free(&fmap);
    nm_free(&gmap);
    for (size_t i = 0; i < n_globals; i++) {
        free(globals[i].name);
        free(globals[i].data);
        for (size_t j = 0; j < globals[i].n_relocs; j++) free(globals[i].relocs[j].target);
        free(globals[i].relocs);
    }
    free(globals);
    sigtab_free(&sigs);
    return result;

fail_after_funcs:
    for (size_t i = 0; i < n_funcs; i++) {
        free(funcs[i].name);
        free(funcs[i].local_types);
        buf_free(&funcs[i].body);
        free(funcs[i].relocs);
    }
    free(funcs);
    free(func_nparams);
    free(addr_taken);
    nm_free(&fmap);
    nm_free(&gmap);
    for (size_t i = 0; i < n_globals; i++) {
        free(globals[i].name);
        free(globals[i].data);
        for (size_t j = 0; j < globals[i].n_relocs; j++) free(globals[i].relocs[j].target);
        free(globals[i].relocs);
    }
    free(globals);
    sigtab_free(&sigs);
    return fail;
}

string mlir_wasmstack_to_bin(MLIR_Context *ctx, MLIR_OpHandle stk_module) {
    return wasm_module_to_bin_impl(ctx, stk_module, false);
}

string mlir_wasmssa_to_bin(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    return wasm_module_to_bin_impl(ctx, ssa_module, true);
}

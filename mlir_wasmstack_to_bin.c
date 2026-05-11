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
#include "mlir_op_names.h"
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
    Buf      body;
    uint32_t func_index;
    uint32_t sym_index;
    Reloc   *relocs;
    size_t   n_relocs, c_relocs;
    // Source MLIR func op (for body emission of defined funcs).
    MLIR_OpHandle src_op;
    // Locals layout (decoded from local_types attr).
    uint8_t *local_types; size_t n_locals;
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
static int find_func_by_name(EmFunc *funcs, size_t n, string nm) {
    for (size_t i = 0; i < n; i++) {
        if (strlen(funcs[i].name) == nm.size &&
            memcmp(funcs[i].name, nm.str, nm.size) == 0) return (int)i;
    }
    return -1;
}
static int find_global_by_name(const EmGlobal *gs, size_t n, string nm) {
    for (size_t i = 0; i < n; i++) {
        if (strlen(gs[i].name) == nm.size &&
            memcmp(gs[i].name, nm.str, nm.size) == 0) return (int)i;
    }
    return -1;
}

// =============================================================================
// Per-op encoding. Reads attrs from the MLIR op and writes bytecode.
// =============================================================================
static bool encode_op(EmFunc *F, MLIR_OpHandle op, uint32_t sp_sym_idx,
                      EmFunc *all_funcs, size_t all_funcs_n,
                      const EmGlobal *globals, size_t n_globals,
                      uint32_t global_sym_base, SigTab *sigs) {
    Buf *b = &F->body;
    MLIR_OpType t = MLIR_GetOpType(op);
    uint8_t valtype = (uint8_t)at_i(op, "valtype");

    switch (t) {
    case OP_TYPE_WASMSTACK_LOCAL_GET:
        buf_putc(b, 0x20); leb_u(b, (uint64_t)at_i(op, "local_idx")); return true;
    case OP_TYPE_WASMSTACK_LOCAL_SET:
        buf_putc(b, 0x21); leb_u(b, (uint64_t)at_i(op, "local_idx")); return true;
    case OP_TYPE_WASMSTACK_LOCAL_TEE:
        buf_putc(b, 0x22); leb_u(b, (uint64_t)at_i(op, "local_idx")); return true;

    case OP_TYPE_WASMSTACK_GLOBAL_GET: {
        uint32_t gidx = (uint32_t)at_i(op, "global_idx");
        buf_putc(b, 0x23);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, gidx);
        if (gidx == 0) emfunc_add_reloc(F, R_WASM_GLOBAL_INDEX_LEB, off, sp_sym_idx);
        return true;
    }
    case OP_TYPE_WASMSTACK_GLOBAL_SET: {
        uint32_t gidx = (uint32_t)at_i(op, "global_idx");
        buf_putc(b, 0x24);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, gidx);
        if (gidx == 0) emfunc_add_reloc(F, R_WASM_GLOBAL_INDEX_LEB, off, sp_sym_idx);
        return true;
    }

    case OP_TYPE_WASMSTACK_CONST: {
        int64_t iv = at_i(op, "value");
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
        uint32_t mem_size = (uint32_t)at_i(op, "mem_size_bytes");
        uint32_t mem_align = (uint32_t)at_i(op, "memory_align_log2");
        uint32_t mem_off = (uint32_t)at_i(op, "memory_offset");
        uint8_t opc;
        if (valtype == WT_I32) {
            if (mem_size == 4)      opc = 0x28;
            else if (mem_size == 2) opc = 0x2e;
            else if (mem_size == 1) opc = 0x2c;
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
        uint32_t mem_size = (uint32_t)at_i(op, "mem_size_bytes");
        uint32_t mem_align = (uint32_t)at_i(op, "memory_align_log2");
        uint32_t mem_off = (uint32_t)at_i(op, "memory_offset");
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
        buf_putc(b, (uint8_t)at_i(op, "wasm_opcode")); return true;

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
        buf_putc(b, 0x0c); leb_u(b, (uint64_t)at_i(op, "depth")); return true;
    case OP_TYPE_WASMSTACK_BR_IF:
        buf_putc(b, 0x0d); leb_u(b, (uint64_t)at_i(op, "depth")); return true;
    case OP_TYPE_WASMSTACK_SELECT:
        buf_putc(b, 0x1b); return true;
    case OP_TYPE_WASMSTACK_EQZ:
        buf_putc(b, valtype == WT_I64 ? 0x50 : 0x45); return true;

    case OP_TYPE_WASMSTACK_CALL: {
        string tgt = at_s(op, "target");
        int idx = find_func_by_name(all_funcs, all_funcs_n, tgt);
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
        string tgt = at_s(op, "target");
        int idx = find_global_by_name(globals, n_globals, tgt);
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
        string tgt = at_s(op, "target");
        int idx = find_func_by_name(all_funcs, all_funcs_n, tgt);
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
        uint8_t *sp = hex_decode(at_s(op, "sig_params"), &snp);
        uint8_t *sr = hex_decode(at_s(op, "sig_results"), &snr);
        uint32_t tidx = sig_intern(sigs, sp, snp, sr, snr);
        free(sp); free(sr);
        buf_putc(b, 0x11);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, tidx);
        emfunc_add_reloc(F, R_WASM_TYPE_INDEX_LEB, off, tidx);
        buf_putc(b, 0x00);
        return true;
    }

    default: {
        string nm = MLIR_GetOpName(op);
        fprintf(stderr, "wasm-emit: unsupported wasmstack op '%.*s' (enum=%d)\n",
                (int)nm.size, nm.str, (int)t);
        return false;
    }
    }
}

// =============================================================================
// Function body builder: locals prelude + ops + 0x0b end.
// =============================================================================
static bool build_body(EmFunc *F, uint32_t sp_sym_idx,
                       EmFunc *all_funcs, size_t all_funcs_n,
                       const EmGlobal *globals, size_t n_globals,
                       uint32_t global_sym_base, SigTab *sigs) {
    // Walk MLIR ops within the func op's body region/block.
    if (MLIR_GetOpNumRegions(F->src_op) < 1) return false;
    MLIR_RegionHandle r = MLIR_GetOpRegion(F->src_op, 0);
    if (MLIR_GetRegionNumBlocks(r) < 1) return false;
    MLIR_BlockHandle blk = MLIR_GetRegionBlock(r, 0);
    size_t nops = MLIR_GetBlockNumOps(blk);

    Buf insns = {0};
    Buf saved = F->body;
    F->body = insns;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        if (!encode_op(F, op, sp_sym_idx, all_funcs, all_funcs_n,
                       globals, n_globals, global_sym_base, sigs)) {
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
        uint32_t flags = f->imported ? SYMF_UNDEFINED : 0;
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
// Top-level translator. Walks the input wasmstack-MLIR module directly.
// =============================================================================
string mlir_wasmstack_to_bin(MLIR_Context *ctx,
                             MLIR_OpHandle stk_module) {
    string fail = {0};
    if (!stk_module) return fail;

    if (MLIR_GetOpNumRegions(stk_module) < 1) return fail;
    MLIR_RegionHandle mr = MLIR_GetOpRegion(stk_module, 0);
    if (MLIR_GetRegionNumBlocks(mr) < 1) return fail;
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t n_top = MLIR_GetBlockNumOps(mb);

    SigTab sigs = {0};

    // ---- Materialize EmFuncs: imports first, then defs. -------------------
    EmFunc *funcs = NULL;
    size_t n_funcs = 0;

    // Pass A: imports.
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, ti);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSTACK_IMPORT_FUNC) continue;
        size_t np, nr;
        uint8_t *p = hex_decode(at_s(op, "param_types"), &np);
        uint8_t *r = hex_decode(at_s(op, "result_types"), &nr);
        funcs = (EmFunc *)realloc(funcs, (n_funcs + 1) * sizeof(EmFunc));
        memset(&funcs[n_funcs], 0, sizeof(EmFunc));
        funcs[n_funcs].name = strdup_str(at_s(op, "sym_name"));
        funcs[n_funcs].sig = sig_intern(&sigs, p, np, r, nr);
        funcs[n_funcs].imported = true;
        funcs[n_funcs].func_index = (uint32_t)n_funcs;
        free(p); free(r);
        n_funcs++;
    }
    // Pass B: defs.
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, ti);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSTACK_FUNC) continue;
        size_t np, nr;
        uint8_t *p = hex_decode(at_s(op, "param_types"), &np);
        uint8_t *r = hex_decode(at_s(op, "result_types"), &nr);
        size_t nl;
        uint8_t *ll = hex_decode(at_s(op, "local_types"), &nl);
        funcs = (EmFunc *)realloc(funcs, (n_funcs + 1) * sizeof(EmFunc));
        memset(&funcs[n_funcs], 0, sizeof(EmFunc));
        funcs[n_funcs].name = strdup_str(at_s(op, "sym_name"));
        funcs[n_funcs].sig = sig_intern(&sigs, p, np, r, nr);
        funcs[n_funcs].imported = false;
        funcs[n_funcs].exported = at_b(op, "exported");
        funcs[n_funcs].func_index = (uint32_t)n_funcs;
        funcs[n_funcs].src_op = op;
        funcs[n_funcs].local_types = ll;
        funcs[n_funcs].n_locals = nl;
        free(p); free(r);
        n_funcs++;
    }

    // ---- Materialize EmGlobals --------------------------------------------
    EmGlobal *globals = NULL;
    size_t n_globals = 0;
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, ti);
        if (MLIR_GetOpType(op) != OP_TYPE_WASMSTACK_IMPORT_GLOBAL) continue;
        globals = (EmGlobal *)realloc(globals, (n_globals + 1) * sizeof(EmGlobal));
        memset(&globals[n_globals], 0, sizeof(EmGlobal));
        globals[n_globals].name = strdup_str(at_s(op, "sym_name"));
        globals[n_globals].size = (uint32_t)at_i(op, "size");
        globals[n_globals].align_pow = (uint32_t)at_i(op, "align_pow");
        globals[n_globals].is_const = at_b(op, "is_const");
        string id = at_s(op, "init_data");
        if (id.size) {
            globals[n_globals].data = (uint8_t *)malloc(id.size);
            memcpy(globals[n_globals].data, id.str, id.size);
        }
        globals[n_globals].relocs = parse_relocs(at_s(op, "relocs"),
                                                 &globals[n_globals].n_relocs);
        n_globals++;
    }

    uint32_t sp_sym = 0;
    for (size_t i = 0; i < n_funcs; i++) funcs[i].sym_index = (uint32_t)(i + 1);
    uint32_t global_sym_base = 1u + (uint32_t)n_funcs;

    // ---- Detect address-taken funcs and indirect-call sites ----------------
    bool need_indirect_table = false;
    uint32_t *addr_taken = NULL;
    size_t n_addr_taken = 0;
    for (size_t i = 0; i < n_funcs; i++) {
        if (funcs[i].imported) continue;
        MLIR_RegionHandle rg = MLIR_GetOpRegion(funcs[i].src_op, 0);
        MLIR_BlockHandle  bl = MLIR_GetRegionBlock(rg, 0);
        size_t nb = MLIR_GetBlockNumOps(bl);
        for (size_t j = 0; j < nb; j++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(bl, j);
            MLIR_OpType t = MLIR_GetOpType(op);
            if (t == OP_TYPE_WASMSTACK_CALL_INDIRECT) {
                need_indirect_table = true;
            } else if (t == OP_TYPE_WASMSTACK_FUNC_ADDR) {
                need_indirect_table = true;
                string tgt = at_s(op, "target");
                int idx = find_func_by_name(funcs, n_funcs, tgt);
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

    // ---- Emit each defined function's body ---------------------------------
    for (size_t i = 0; i < n_funcs; i++) {
        if (funcs[i].imported) continue;
        if (!build_body(&funcs[i], sp_sym, funcs, n_funcs,
                        globals, n_globals, global_sym_base, &sigs)) {
            goto fail_after_funcs;
        }
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
            emit_string(&import_payload, "env");
            emit_string(&import_payload, funcs[i].name);
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
    free(addr_taken);
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
    free(addr_taken);
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

// Stage 3 of the native LLVM->WASM pipeline:
// wasmstack_module_t -> wasm32 relocatable object bytes (.wasm.o).
//
// This stage is intentionally near-mechanical. It walks the
// wasmstack form, dispatches on each op's MLIR_OpType (no string
// compares) and writes the corresponding bytecode. All the
// non-trivial decisions (function shape, op order, local layout,
// shadow-stack lowering) were made by stages 1 and 2.
//
// The output is wasm-ld-compatible: TYPE / IMPORT / FUNCTION / CODE
// sections plus the conventional `linking` and `reloc.CODE` custom
// sections. Symbols and reloc records are produced inline as ops are
// emitted.

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
#include "mlir_wasm_dialect.h"

string mlir_translate_wasmstack_to_binary(MLIR_Context *ctx,
                                          wasmstack_module_t *m);

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
// Five-byte zero-padded uleb128: used inside the CODE section for
// any value referenced by a relocation. The linker rewrites the LEB
// in place so the byte width must be invariant.
static void leb_u_pad5(Buf *b, uint32_t v) {
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
    SEC_DATACOUNT = 12,

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

    SYMF_BINDING_LOCAL = 0x02,
    SYMF_VISIBILITY_HIDDEN = 0x04,
    SYMF_UNDEFINED = 0x10,
    SYMF_EXPLICIT_NAME = 0x40,
    SYMF_NO_STRIP  = 0x80,

    R_WASM_FUNCTION_INDEX_LEB = 0,
    R_WASM_TABLE_INDEX_SLEB   = 1,
    R_WASM_MEMORY_ADDR_LEB    = 3,
    R_WASM_MEMORY_ADDR_SLEB   = 4,
    R_WASM_MEMORY_ADDR_I32    = 5,
    R_WASM_TYPE_INDEX_LEB     = 6,
    R_WASM_GLOBAL_INDEX_LEB   = 7,
};

// 5-byte zero-padded sleb128 (signed counterpart of leb_u_pad5).
static void leb_s_pad5(Buf *b, int32_t v) {
    for (int i = 0; i < 5; i++) {
        uint8_t byte = v & 0x7f;
        v >>= 7;
        if (i < 4) byte |= 0x80;
        buf_putc(b, byte);
    }
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
// Function record (intermediate, post-flattening).
// =============================================================================
typedef struct {
    uint8_t  type;
    uint32_t body_offset;   // relative to start of `body` while building
    uint32_t sym_idx;
    int32_t  addend;        // only for memory_addr reloc kinds
} Reloc;

typedef struct {
    char    *name;
    uint32_t sig;
    bool     imported;
    bool     exported;
    Buf      body;          // CODE-section body (locals prelude + insns + 0x0b end)
    uint32_t func_index;
    uint32_t sym_index;
    Reloc   *relocs;
    size_t   n_relocs, c_relocs;
} EmFunc;

static void emfunc_add_reloc(EmFunc *f, uint8_t type, uint32_t off, uint32_t sym) {
    if (f->n_relocs == f->c_relocs) {
        f->c_relocs = f->c_relocs ? f->c_relocs * 2 : 4;
        f->relocs = (Reloc *)realloc(f->relocs, f->c_relocs * sizeof(Reloc));
    }
    f->relocs[f->n_relocs++] = (Reloc){type, off, sym, 0};
}
static void emfunc_add_reloc_a(EmFunc *f, uint8_t type, uint32_t off,
                               uint32_t sym, int32_t addend) {
    if (f->n_relocs == f->c_relocs) {
        f->c_relocs = f->c_relocs ? f->c_relocs * 2 : 4;
        f->relocs = (Reloc *)realloc(f->relocs, f->c_relocs * sizeof(Reloc));
    }
    f->relocs[f->n_relocs++] = (Reloc){type, off, sym, addend};
}

// =============================================================================
// Helpers.
// =============================================================================
static char *xstrdupc(const char *s) {
    size_t n = strlen(s);
    char *r = (char *)malloc(n + 1);
    memcpy(r, s, n + 1);
    return r;
}
static int find_func_by_name(EmFunc *funcs, size_t n, const char *nm) {
    for (size_t i = 0; i < n; i++) if (strcmp(funcs[i].name, nm) == 0) return (int)i;
    return -1;
}

// =============================================================================
// Per-op encoding. Dispatches on MLIR_OpType. Returns false on unknown
// op or invalid encoding.
// =============================================================================
static int find_global_by_name(const wasm_global_t *gs, size_t n,
                               const char *nm) {
    for (size_t i = 0; i < n; i++)
        if (strcmp(gs[i].name, nm) == 0) return (int)i;
    return -1;
}

static bool encode_op(EmFunc *F, const wasmstack_op_t *op, uint32_t sp_sym_idx,
                      EmFunc *all_funcs, size_t all_funcs_n,
                      const wasm_global_t *globals, size_t n_globals,
                      uint32_t global_sym_base, SigTab *sigs) {
    Buf *b = &F->body;
    switch (op->type) {
    case OP_TYPE_WASMSTACK_LOCAL_GET:
        buf_putc(b, 0x20); leb_u(b, op->local_idx); return true;
    case OP_TYPE_WASMSTACK_LOCAL_SET:
        buf_putc(b, 0x21); leb_u(b, op->local_idx); return true;
    case OP_TYPE_WASMSTACK_LOCAL_TEE:
        buf_putc(b, 0x22); leb_u(b, op->local_idx); return true;

    case OP_TYPE_WASMSTACK_GLOBAL_GET: {
        buf_putc(b, 0x23);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, op->global_idx);
        // global_idx 0 is the imported __stack_pointer; emit a reloc.
        if (op->global_idx == 0) {
            emfunc_add_reloc(F, R_WASM_GLOBAL_INDEX_LEB, off, sp_sym_idx);
        }
        return true;
    }
    case OP_TYPE_WASMSTACK_GLOBAL_SET: {
        buf_putc(b, 0x24);
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, op->global_idx);
        if (op->global_idx == 0) {
            emfunc_add_reloc(F, R_WASM_GLOBAL_INDEX_LEB, off, sp_sym_idx);
        }
        return true;
    }

    case OP_TYPE_WASMSTACK_CONST:
        if (op->valtype == WT_I32)      { buf_putc(b, 0x41); leb_s(b, (int32_t)op->i_const); }
        else if (op->valtype == WT_I64) { buf_putc(b, 0x42); leb_s(b, op->i_const); }
        else if (op->valtype == WT_F32) {
            buf_putc(b, 0x43);
            uint32_t bits = (uint32_t)(uint64_t)op->i_const;
            for (int i = 0; i < 4; i++) buf_putc(b, (uint8_t)(bits >> (8*i)));
        }
        else if (op->valtype == WT_F64) {
            buf_putc(b, 0x44);
            uint64_t bits = (uint64_t)op->i_const;
            for (int i = 0; i < 8; i++) buf_putc(b, (uint8_t)(bits >> (8*i)));
        }
        else { fprintf(stderr, "wasm-emit: const of unsupported valtype\n"); return false; }
        return true;

    case OP_TYPE_WASMSTACK_ADD:
        if (op->valtype == WT_I32)      buf_putc(b, 0x6a);
        else if (op->valtype == WT_I64) buf_putc(b, 0x7c);
        else { fprintf(stderr, "wasm-emit: add of unsupported valtype\n"); return false; }
        return true;

    case OP_TYPE_WASMSTACK_SUB:
        if (op->valtype == WT_I32)      buf_putc(b, 0x6b);
        else if (op->valtype == WT_I64) buf_putc(b, 0x7d);
        else { fprintf(stderr, "wasm-emit: sub of unsupported valtype\n"); return false; }
        return true;

    case OP_TYPE_WASMSTACK_LOAD: {
        uint8_t opc;
        if (op->valtype == WT_I32) {
            if (op->mem_size_bytes == 4)      opc = 0x28;
            else if (op->mem_size_bytes == 2) opc = 0x2e;  // i32.load16_s
            else if (op->mem_size_bytes == 1) opc = 0x2c;  // i32.load8_s
            else return false;
        } else if (op->valtype == WT_I64) {
            opc = 0x29;
        } else if (op->valtype == WT_F32) { opc = 0x2a; }
        else if (op->valtype == WT_F64)   { opc = 0x2b; }
        else return false;
        buf_putc(b, opc);
        leb_u(b, op->memory_align_log2);
        leb_u(b, op->memory_offset);
        return true;
    }
    case OP_TYPE_WASMSTACK_STORE: {
        uint8_t opc;
        if (op->valtype == WT_I32) {
            if (op->mem_size_bytes == 4)      opc = 0x36;
            else if (op->mem_size_bytes == 2) opc = 0x3b;  // i32.store16
            else if (op->mem_size_bytes == 1) opc = 0x3a;  // i32.store8
            else return false;
        } else if (op->valtype == WT_I64) {
            opc = 0x37;
        } else if (op->valtype == WT_F32) { opc = 0x38; }
        else if (op->valtype == WT_F64)   { opc = 0x39; }
        else return false;
        buf_putc(b, opc);
        leb_u(b, op->memory_align_log2);
        leb_u(b, op->memory_offset);
        return true;
    }

    case OP_TYPE_WASMSTACK_RETURN:
        buf_putc(b, 0x0f);
        return true;

    case OP_TYPE_WASMSTACK_EXTEND_I32_S:
        buf_putc(b, 0xac);  // i64.extend_i32_s
        return true;

    case OP_TYPE_WASMSTACK_BINOP:
    case OP_TYPE_WASMSTACK_UNOP:
        buf_putc(b, op->wasm_opcode);
        return true;

    case OP_TYPE_WASMSTACK_BLOCK:
        buf_putc(b, 0x02); buf_putc(b, op->valtype ? op->valtype : 0x40);
        return true;
    case OP_TYPE_WASMSTACK_LOOP:
        buf_putc(b, 0x03); buf_putc(b, op->valtype ? op->valtype : 0x40);
        return true;
    case OP_TYPE_WASMSTACK_IF:
        buf_putc(b, 0x04); buf_putc(b, op->valtype ? op->valtype : 0x40);
        return true;
    case OP_TYPE_WASMSTACK_ELSE:
        buf_putc(b, 0x05);
        return true;
    case OP_TYPE_WASMSTACK_END:
        buf_putc(b, 0x0b);
        return true;
    case OP_TYPE_WASMSTACK_BR:
        buf_putc(b, 0x0c); leb_u(b, op->br_depth); return true;
    case OP_TYPE_WASMSTACK_BR_IF:
        buf_putc(b, 0x0d); leb_u(b, op->br_depth); return true;
    case OP_TYPE_WASMSTACK_SELECT:
        buf_putc(b, 0x1b); return true;
    case OP_TYPE_WASMSTACK_EQZ:
        buf_putc(b, op->valtype == WT_I64 ? 0x50 : 0x45); return true;

    case OP_TYPE_WASMSTACK_CALL: {
        int idx = find_func_by_name(all_funcs, all_funcs_n, op->call_target);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: call to unknown function '%s'\n",
                    op->call_target);
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
        int idx = find_global_by_name(globals, n_globals, op->call_target);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: addressof unknown global '%s'\n",
                    op->call_target);
            return false;
        }
        buf_putc(b, 0x41);  // i32.const
        uint32_t off = (uint32_t)b->len;
        leb_s_pad5(b, 0);
        emfunc_add_reloc(F, R_WASM_MEMORY_ADDR_SLEB, off,
                         global_sym_base + (uint32_t)idx);
        return true;
    }

    case OP_TYPE_WASMSTACK_FUNC_ADDR: {
        int idx = find_func_by_name(all_funcs, all_funcs_n, op->call_target);
        if (idx < 0) {
            fprintf(stderr, "wasm-emit: func_addr unknown function '%s'\n",
                    op->call_target);
            return false;
        }
        buf_putc(b, 0x41);  // i32.const
        uint32_t off = (uint32_t)b->len;
        leb_s_pad5(b, 0);
        emfunc_add_reloc(F, R_WASM_TABLE_INDEX_SLEB, off,
                         all_funcs[idx].sym_index);
        return true;
    }

    case OP_TYPE_WASMSTACK_CALL_INDIRECT: {
        // Intern the callee signature so the TYPE section has it.
        uint32_t tidx = sig_intern(sigs, op->sig_params, op->n_sig_params,
                                         op->sig_results, op->n_sig_results);
        buf_putc(b, 0x11);  // call_indirect
        uint32_t off = (uint32_t)b->len;
        leb_u_pad5(b, tidx);
        emfunc_add_reloc(F, R_WASM_TYPE_INDEX_LEB, off, tidx);
        buf_putc(b, 0x00);  // table 0
        return true;
    }

    default:
        fprintf(stderr, "wasm-emit: unsupported wasmstack op (enum=%d)\n",
                (int)op->type);
        return false;
    }
}

// =============================================================================
// Function-body builder: locals prelude + ops + 0x0b end.
// =============================================================================
static bool build_body(EmFunc *F, const wasmstack_func_t *src,
                       uint32_t sp_sym_idx,
                       EmFunc *all_funcs, size_t all_funcs_n,
                       const wasm_global_t *globals, size_t n_globals,
                       uint32_t global_sym_base, SigTab *sigs) {
    // Build instruction stream.
    Buf insns = {0};
    Buf saved = F->body;
    F->body = insns;
    for (size_t i = 0; i < src->n_ops; i++) {
        if (!encode_op(F, &src->ops[i], sp_sym_idx, all_funcs, all_funcs_n,
                       globals, n_globals, global_sym_base, sigs)) {
            buf_free(&F->body);
            F->body = saved;
            return false;
        }
    }
    buf_putc(&F->body, 0x0b);  // end

    // Stash insns; we'll prepend the locals prelude.
    Buf body_insns = F->body;
    F->body = saved;

    // Locals prelude: count of (count, type) groups, then each group.
    Buf prelude = {0};
    {
        size_t i = 0, groups = 0;
        while (i < src->n_locals) {
            uint8_t t = src->local_types[i];
            size_t j = i + 1;
            while (j < src->n_locals && src->local_types[j] == t) j++;
            groups++; i = j;
        }
        leb_u(&prelude, groups);
        i = 0;
        while (i < src->n_locals) {
            uint8_t t = src->local_types[i];
            size_t j = i + 1;
            while (j < src->n_locals && src->local_types[j] == t) j++;
            leb_u(&prelude, (uint32_t)(j - i));
            buf_putc(&prelude, t);
            i = j;
        }
    }

    // Splice prelude + insns into F->body and shift reloc offsets.
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

// `linking` custom section: SYMBOL_TABLE subsection + optional SEGMENT_INFO.
//   sym 0       : __stack_pointer (imported global)
//   sym 1..n    : functions in EmFunc order (imports first, then defs)
//   sym n+1..   : data symbols (one per global)
static void build_linking_section(EmFunc *funcs, size_t n_funcs,
                                  uint32_t sp_global_idx,
                                  const wasm_global_t *globals, size_t n_globals,
                                  bool include_indirect_table,
                                  uint32_t indirect_table_idx,
                                  Buf *out) {
    Buf body = {0};
    leb_u(&body, LINK_VERSION);

    Buf sub = {0};
    leb_u(&sub, 1u + (uint32_t)n_funcs + (uint32_t)n_globals
                + (include_indirect_table ? 1u : 0u));

    // Sym 0: __stack_pointer (imported global).
    buf_putc(&sub, SYM_GLOBAL);
    leb_u(&sub, SYMF_UNDEFINED);
    leb_u(&sub, sp_global_idx);
    // No name: UNDEFINED && !EXPLICIT_NAME

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
        // Mark all data symbols as local + hidden so the linker treats
        // them as private to this object.
        leb_u(&sub, SYMF_BINDING_LOCAL | SYMF_VISIBILITY_HIDDEN);
        emit_string(&sub, globals[i].name);
        // For SYM_DATA, defined symbols carry segment index, offset, size.
        leb_u(&sub, (uint32_t)i);   // segment_index
        leb_u(&sub, 0);             // offset within segment
        leb_u(&sub, globals[i].size);
    }

    if (include_indirect_table) {
        buf_putc(&sub, SYM_TABLE);
        leb_u(&sub, SYMF_NO_STRIP | SYMF_UNDEFINED);
        leb_u(&sub, indirect_table_idx);
        // No name: UNDEFINED && !EXPLICIT_NAME → name comes from import.
    }

    buf_putc(&body, LINK_SUB_SYMBOL_TABLE);
    leb_u(&body, sub.len);
    buf_append(&body, sub.data, sub.len);
    buf_free(&sub);

    if (n_globals) {
        Buf seg = {0};
        leb_u(&seg, (uint32_t)n_globals);
        for (size_t i = 0; i < n_globals; i++) {
            // Sanitize: lld accepts arbitrary names but require non-empty.
            const char *nm = globals[i].name && globals[i].name[0]
                              ? globals[i].name : ".rodata";
            // Use a conventional segment name prefix per constness.
            // Many existing toolchains use ".rodata.<name>" / ".data.<name>".
            char *full = (char *)malloc(strlen(nm) + 16);
            sprintf(full, "%s%s", globals[i].is_const ? ".rodata." : ".data.", nm);
            emit_string(&seg, full);
            free(full);
            leb_u(&seg, globals[i].align_pow);
            leb_u(&seg, 0);  // flags
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

// Helper: true if this reloc type carries an addend (sleb128).
static bool reloc_has_addend(uint8_t t) {
    return t == R_WASM_MEMORY_ADDR_LEB ||
           t == R_WASM_MEMORY_ADDR_SLEB ||
           t == R_WASM_MEMORY_ADDR_I32;
}

// `reloc.CODE` custom section.
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
// Top-level translator.
// =============================================================================
string mlir_translate_wasmstack_to_binary(MLIR_Context *ctx,
                                          wasmstack_module_t *m) {
    string fail = {0};
    if (!m) return fail;

    SigTab sigs = {0};

    // ---- Materialize EmFuncs (one per src func) -----------------------------
    // Order: imports first, then defs. This matches the wasm function-index
    // space convention.
    EmFunc *funcs = NULL;
    size_t  n_funcs = 0;

    // Pass A: imports.
    for (size_t i = 0; i < m->n_funcs; i++) {
        if (!m->funcs[i].imported) continue;
        funcs = (EmFunc *)realloc(funcs, (n_funcs + 1) * sizeof(EmFunc));
        memset(&funcs[n_funcs], 0, sizeof(EmFunc));
        funcs[n_funcs].name = xstrdupc(m->funcs[i].name);
        funcs[n_funcs].sig = sig_intern(&sigs, m->funcs[i].param_types, m->funcs[i].n_params,
                                              m->funcs[i].result_types, m->funcs[i].n_results);
        funcs[n_funcs].imported = true;
        funcs[n_funcs].func_index = (uint32_t)n_funcs;
        n_funcs++;
    }
    // Pass B: defs.
    for (size_t i = 0; i < m->n_funcs; i++) {
        if (m->funcs[i].imported) continue;
        funcs = (EmFunc *)realloc(funcs, (n_funcs + 1) * sizeof(EmFunc));
        memset(&funcs[n_funcs], 0, sizeof(EmFunc));
        funcs[n_funcs].name = xstrdupc(m->funcs[i].name);
        funcs[n_funcs].sig = sig_intern(&sigs, m->funcs[i].param_types, m->funcs[i].n_params,
                                              m->funcs[i].result_types, m->funcs[i].n_results);
        funcs[n_funcs].imported = false;
        funcs[n_funcs].exported = m->funcs[i].exported;
        funcs[n_funcs].func_index = (uint32_t)n_funcs;
        n_funcs++;
    }

    // Sym indices: 0 = __stack_pointer, 1..N = funcs, N+1.. = data globals.
    uint32_t sp_sym = 0;
    for (size_t i = 0; i < n_funcs; i++) funcs[i].sym_index = (uint32_t)(i + 1);
    uint32_t global_sym_base = 1u + (uint32_t)n_funcs;

    // ---- Collect address-taken functions and detect indirect calls --------
    // Walk all (defined) function bodies looking for FUNC_ADDR ops (gives the
    // set of address-taken function names) and CALL_INDIRECT ops (signals
    // that we need the indirect_function_table import even if no FUNC_ADDR is
    // local to this object).
    bool need_indirect_table = false;
    uint32_t *addr_taken = NULL;        // function-index list
    size_t n_addr_taken = 0;
    {
        for (size_t i = 0; i < m->n_funcs; i++) {
            const wasmstack_func_t *sf = &m->funcs[i];
            for (size_t j = 0; j < sf->n_ops; j++) {
                const wasmstack_op_t *op = &sf->ops[j];
                if (op->type == OP_TYPE_WASMSTACK_CALL_INDIRECT) {
                    need_indirect_table = true;
                } else if (op->type == OP_TYPE_WASMSTACK_FUNC_ADDR) {
                    need_indirect_table = true;
                    int idx = find_func_by_name(funcs, n_funcs, op->call_target);
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
    }

    // ---- Emit each defined function's body ---------------------------------
    // We have to walk the source funcs in their original order to find each
    // EmFunc's body source.
    size_t ei = 0;
    for (size_t pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < m->n_funcs; i++) {
            bool imp = m->funcs[i].imported;
            if (pass == 0 && !imp) continue;
            if (pass == 1 && imp) continue;
            if (pass == 1) {
                if (!build_body(&funcs[ei], &m->funcs[i], sp_sym, funcs, n_funcs,
                                m->globals, m->n_globals, global_sym_base, &sigs)) {
                    goto fail_after_funcs;
                }
            }
            ei++;
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

        // env.__linear_memory : memory 0
        emit_string(&import_payload, "env");
        emit_string(&import_payload, "__linear_memory");
        buf_putc(&import_payload, IMP_MEMORY);
        buf_putc(&import_payload, 0x00);
        leb_u(&import_payload, 0);

        // env.__stack_pointer : global i32 mut
        emit_string(&import_payload, "env");
        emit_string(&import_payload, "__stack_pointer");
        buf_putc(&import_payload, IMP_GLOBAL);
        buf_putc(&import_payload, WT_I32);
        buf_putc(&import_payload, 0x01);  // mutable

        for (size_t i = 0; i < n_funcs; i++) {
            if (!funcs[i].imported) continue;
            emit_string(&import_payload, "env");
            emit_string(&import_payload, funcs[i].name);
            buf_putc(&import_payload, IMP_FUNC);
            leb_u(&import_payload, funcs[i].sig);
        }

        if (need_indirect_table) {
            // env.__indirect_function_table : table funcref min=1
            emit_string(&import_payload, "env");
            emit_string(&import_payload, "__indirect_function_table");
            buf_putc(&import_payload, IMP_TABLE);
            buf_putc(&import_payload, 0x70);  // funcref
            buf_putc(&import_payload, 0x00);  // limits: min only
            leb_u(&import_payload, 1);        // min=1 (slot 0 reserved)
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

    // ---- ELEM section ------------------------------------------------------
    // One active segment in the imported __indirect_function_table at
    // initial offset i32.const 1 (slot 0 reserved for null-fnptr).
    Buf elem_payload = {0};
    if (n_addr_taken) {
        leb_u(&elem_payload, 1);            // 1 segment
        leb_u(&elem_payload, 0);            // flag=0 (active, table 0, funcref impl.)
        buf_putc(&elem_payload, 0x41);      // i32.const
        leb_s(&elem_payload, 1);            //   1
        buf_putc(&elem_payload, 0x0b);      // end
        leb_u(&elem_payload, (uint32_t)n_addr_taken);
        for (size_t i = 0; i < n_addr_taken; i++)
            leb_u(&elem_payload, addr_taken[i]);
        emit_section(&img, SEC_ELEMENT, &elem_payload);
    }

    uint8_t code_section_index = (uint8_t)(n_addr_taken ? 4 : 3);
    emit_section(&img, SEC_CODE,     &code_payload);
    uint8_t data_section_index = (uint8_t)(code_section_index + 1);

    // ---- DATA section + reloc.DATA -----------------------------------------
    // One active segment per global (memory 0, offset i32.const 0). Each
    // segment header bytes occupy a few bytes before the data payload;
    // we track each global's data start within the payload buffer to
    // place reloc records for ptr-init globals.
    Buf data_payload = {0};
    uint32_t *data_offsets = NULL;
    if (m->n_globals) {
        data_offsets = (uint32_t *)calloc(m->n_globals, sizeof(uint32_t));
        leb_u(&data_payload, (uint32_t)m->n_globals);
        for (size_t i = 0; i < m->n_globals; i++) {
            const wasm_global_t *g = &m->globals[i];
            // active segment in memory 0 with offset = i32.const 0
            leb_u(&data_payload, 0);     // flags=0 (active, memory 0)
            buf_putc(&data_payload, 0x41); leb_s(&data_payload, 0); // i32.const 0
            buf_putc(&data_payload, 0x0b);                          // end
            leb_u(&data_payload, g->size);
            data_offsets[i] = (uint32_t)data_payload.len;
            if (g->size) buf_append(&data_payload, g->data, g->size);
        }
        // Required for relocatable wasm with active segments? Not strictly
        // — wasm-ld doesn't insist on a DataCount for relocatable inputs
        // when memory.init isn't used. Skip.
        emit_section(&img, SEC_DATA, &data_payload);
    }

    build_linking_section(funcs, n_funcs, /*sp_global_idx=*/0,
                          m->globals, m->n_globals,
                          need_indirect_table, /*indirect_table_idx=*/0,
                          &img);
    build_reloc_code_section(funcs, n_funcs, code_section_index, body_offsets, &img);

    // reloc.DATA for ptr-init globals.
    if (m->n_globals) {
        size_t total = 0;
        for (size_t i = 0; i < m->n_globals; i++) total += m->globals[i].n_relocs;
        if (total) {
            Buf body = {0};
            leb_u(&body, data_section_index);
            leb_u(&body, (uint32_t)total);
            for (size_t i = 0; i < m->n_globals; i++) {
                const wasm_global_t *g = &m->globals[i];
                for (size_t j = 0; j < g->n_relocs; j++) {
                    int tidx = -1;
                    for (size_t k = 0; k < m->n_globals; k++)
                        if (strcmp(m->globals[k].name, g->relocs[j].target) == 0) {
                            tidx = (int)k; break;
                        }
                    if (tidx < 0) {
                        fprintf(stderr,
                                "wasm-emit: reloc.DATA target '%s' not found\n",
                                g->relocs[j].target);
                        free(data_offsets);
                        buf_free(&data_payload);
                        buf_free(&body);
                        goto fail_after_funcs;
                    }
                    buf_putc(&body, R_WASM_MEMORY_ADDR_I32);
                    leb_u(&body, data_offsets[i] + g->relocs[j].offset);
                    leb_u(&body, global_sym_base + (uint32_t)tidx);
                    leb_s(&body, g->relocs[j].addend);
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

    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *out = (char *)arena_alloc(arena, img.len);
    memcpy(out, img.data, img.len);
    string r = { out, img.len };
    buf_free(&img);

    for (size_t i = 0; i < n_funcs; i++) {
        free(funcs[i].name);
        buf_free(&funcs[i].body);
        free(funcs[i].relocs);
    }
    free(funcs);
    sigtab_free(&sigs);
    return r;

fail_after_funcs:
    for (size_t i = 0; i < n_funcs; i++) {
        free(funcs[i].name);
        buf_free(&funcs[i].body);
        free(funcs[i].relocs);
    }
    free(funcs);
    sigtab_free(&sigs);
    return fail;
}

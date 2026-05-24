// In-tree minimal wasm linker. See mlir_wasm_link.h for the public API
// and scope. Implements what wasm-ld does for the tinyc test pipeline:
// merge N relocatable wasm objects, resolve symbols, apply
// R_WASM_* relocations, and emit a single executable wasm module.
//
// The structure mirrors lld/wasm/Writer.cpp in spirit but is roughly
// ~50x shorter because we only support the slice of the spec our
// inputs actually use:
//
//   * One memory (imported by every input as env.__linear_memory; the
//     linker materialises it and assigns initial pages large enough
//     for stack + data + heap headroom).
//   * One indirect function table (env.__indirect_function_table),
//     materialised on demand by the linker.
//   * One stack pointer global (env.__stack_pointer), materialised by
//     the linker. Stack lives in [0, STACK_SIZE); data follows.
//   * Function and global symbol resolution by name. Undefined symbols
//     with a non-"env" import_module are treated as host imports
//     (e.g. wasi_snapshot_preview1.*).
//   * The relocation types tinyc + clang emit for our inputs (see
//     reloc_table below).
//
// Layout of the produced wasm module:
//
//   magic / version
//   Type section          (deduplicated across inputs)
//   Import section        (host imports only: wasi + similar)
//   Function section      (defined functions, in object/file order)
//   Table section         (one funcref table sized to element segments)
//   Memory section        (one memory, computed page count)
//   Global section        (defined globals incl. linker-materialised
//                          __stack_pointer)
//   Export section        (just `entry_export`, plus memory for
//                          wasi-libc-style hosts)
//   Element section       (function table init from object element
//                          segments)
//   Code section          (function bodies with relocations applied)
//   Data section          (data segments at computed addresses with
//                          relocations applied)
//
// No custom sections are emitted; the output is stripped.

#include "mlir_wasm_link.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <base/io.h>
#include <base/string.h>
#include <platform/platform.h>

// =============================================================================
// Diagnostics
// =============================================================================
static void link_err(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    // We avoid pulling in vsnprintf via libc; the corec/corec-stdlib
    // builds may or may not have it. Fall back to a manual two-step
    // concat that handles "%s" and "%u" only — enough for our error
    // messages.
    size_t n = 0;
    const char *p = fmt;
    while (*p && n + 1 < sizeof(buf)) {
        if (p[0] == '%' && p[1] == 's') {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            while (*s && n + 1 < sizeof(buf)) buf[n++] = *s++;
            p += 2;
        } else if (p[0] == '%' && p[1] == 'u') {
            unsigned u = va_arg(ap, unsigned);
            char tmp[16];
            int  tn = 0;
            if (u == 0) tmp[tn++] = '0';
            while (u) { tmp[tn++] = (char)('0' + (u % 10u)); u /= 10u; }
            while (tn > 0 && n + 1 < sizeof(buf)) buf[n++] = tmp[--tn];
            p += 2;
        } else if (p[0] == '%' && p[1] == 'x') {
            unsigned u = va_arg(ap, unsigned);
            char tmp[16];
            int  tn = 0;
            if (u == 0) tmp[tn++] = '0';
            while (u) {
                unsigned d = u & 0xfu;
                tmp[tn++] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
                u >>= 4;
            }
            while (tn > 0 && n + 1 < sizeof(buf)) buf[n++] = tmp[--tn];
            p += 2;
        } else {
            buf[n++] = *p++;
        }
    }
    if (n + 1 < sizeof(buf)) buf[n++] = '\n';
    va_end(ap);
    ciovec_t iov = {.buf = buf, .buf_len = n};
    write_all(PLATFORM_STDERR_FD, &iov, 1);
}

// =============================================================================
// Growable byte buffer
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

static void emit_section(Buf *out, uint8_t id, const Buf *payload) {
    buf_putc(out, id);
    leb_u(out, payload->len);
    buf_append(out, payload->data, payload->len);
}
static void emit_string(Buf *b, const char *s) {
    size_t n = strlen(s);
    leb_u(b, n);
    buf_append(b, s, n);
}
static void emit_string_view(Buf *b, const uint8_t *s, size_t n) {
    leb_u(b, n);
    buf_append(b, s, n);
}

// =============================================================================
// Wasm spec constants
// =============================================================================
enum {
    SEC_CUSTOM   = 0,
    SEC_TYPE     = 1,
    SEC_IMPORT   = 2,
    SEC_FUNCTION = 3,
    SEC_TABLE    = 4,
    SEC_MEMORY   = 5,
    SEC_GLOBAL   = 6,
    SEC_EXPORT   = 7,
    SEC_START    = 8,
    SEC_ELEMENT  = 9,
    SEC_CODE     = 10,
    SEC_DATA     = 11,
    SEC_DATACNT  = 12,

    IMP_FUNC   = 0,
    IMP_TABLE  = 1,
    IMP_MEMORY = 2,
    IMP_GLOBAL = 3,

    // Linking-section subsection IDs.
    LINK_SUB_SEGMENT_INFO    = 5,
    LINK_SUB_INIT_FUNCS      = 6,
    LINK_SUB_COMDAT_INFO     = 7,
    LINK_SUB_SYMBOL_TABLE    = 8,

    // Symbol kinds.
    SYM_FUNCTION = 0,
    SYM_DATA     = 1,
    SYM_GLOBAL   = 2,
    SYM_SECTION  = 3,
    SYM_EVENT    = 4,
    SYM_TABLE    = 5,

    // Symbol flags.
    SYMF_BINDING_WEAK      = 0x01,
    SYMF_BINDING_LOCAL     = 0x02,
    SYMF_VISIBILITY_HIDDEN = 0x04,
    SYMF_UNDEFINED         = 0x10,
    SYMF_EXPORTED          = 0x20,
    SYMF_EXPLICIT_NAME     = 0x40,
    SYMF_NO_STRIP          = 0x80,
    SYMF_TLS               = 0x100,
    SYMF_ABS               = 0x200,

    // Relocation kinds (subset we apply; everything else errors out).
    R_WASM_FUNCTION_INDEX_LEB = 0,
    R_WASM_TABLE_INDEX_SLEB   = 1,
    R_WASM_TABLE_INDEX_I32    = 2,
    R_WASM_MEMORY_ADDR_LEB    = 3,
    R_WASM_MEMORY_ADDR_SLEB   = 4,
    R_WASM_MEMORY_ADDR_I32    = 5,
    R_WASM_TYPE_INDEX_LEB     = 6,
    R_WASM_GLOBAL_INDEX_LEB   = 7,
    R_WASM_FUNCTION_OFFSET_I32 = 8,
    R_WASM_SECTION_OFFSET_I32  = 9,
    R_WASM_TAG_INDEX_LEB       = 10,
    R_WASM_MEMORY_ADDR_REL_SLEB = 11,
    R_WASM_TABLE_INDEX_REL_SLEB = 12,
    R_WASM_GLOBAL_INDEX_I32    = 13,
    R_WASM_MEMORY_ADDR_LEB64   = 14,
    R_WASM_MEMORY_ADDR_SLEB64  = 15,
    R_WASM_MEMORY_ADDR_I64     = 16,
    R_WASM_TABLE_INDEX_SLEB64  = 18,
    R_WASM_TABLE_INDEX_I64     = 19,
    R_WASM_TABLE_NUMBER_LEB    = 20,
    R_WASM_FUNCTION_OFFSET_I64 = 22,
    R_WASM_MEMORY_ADDR_LOCREL_I32 = 23,
    R_WASM_TABLE_INDEX_REL_SLEB64 = 24,
};

static bool reloc_has_addend(uint8_t t) {
    switch (t) {
        case R_WASM_MEMORY_ADDR_LEB:
        case R_WASM_MEMORY_ADDR_SLEB:
        case R_WASM_MEMORY_ADDR_I32:
        case R_WASM_MEMORY_ADDR_REL_SLEB:
        case R_WASM_FUNCTION_OFFSET_I32:
        case R_WASM_SECTION_OFFSET_I32:
        case R_WASM_MEMORY_ADDR_LEB64:
        case R_WASM_MEMORY_ADDR_SLEB64:
        case R_WASM_MEMORY_ADDR_I64:
        case R_WASM_MEMORY_ADDR_LOCREL_I32:
        case R_WASM_FUNCTION_OFFSET_I64:
            return true;
        default:
            return false;
    }
}

// Resolved-value class: every reloc binds to a "target" of one of
// these forms.
typedef enum {
    TGT_FUNC,       // final function index (incl. imports first)
    TGT_GLOBAL,     // final global index
    TGT_DATA_ADDR,  // 32-bit linear-memory address
    TGT_TYPE,       // final type index
    TGT_TABLE,      // final table index
} TargetKind;

// =============================================================================
// LEB readers (return -1 if input ran out)
// =============================================================================
typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    bool ok;
} Reader;

static Reader rd_make(const uint8_t *p, size_t n) {
    return (Reader){.p = p, .end = p + n, .ok = true};
}
static bool rd_avail(const Reader *r, size_t n) {
    return r->ok && (size_t)(r->end - r->p) >= n;
}
static uint8_t rd_u8(Reader *r) {
    if (!rd_avail(r, 1)) { r->ok = false; return 0; }
    return *r->p++;
}
static uint32_t rd_u32_le(Reader *r) {
    if (!rd_avail(r, 4)) { r->ok = false; return 0; }
    uint32_t v = (uint32_t)r->p[0] | ((uint32_t)r->p[1] << 8)
               | ((uint32_t)r->p[2] << 16) | ((uint32_t)r->p[3] << 24);
    r->p += 4;
    return v;
}
static uint64_t rd_leb_u(Reader *r) {
    uint64_t v = 0; unsigned shift = 0;
    while (1) {
        if (!rd_avail(r, 1)) { r->ok = false; return 0; }
        uint8_t b = *r->p++;
        v |= ((uint64_t)(b & 0x7f)) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
        if (shift > 63) { r->ok = false; return 0; }
    }
    return v;
}
static int64_t rd_leb_s(Reader *r) {
    int64_t v = 0; unsigned shift = 0; uint8_t b = 0;
    while (1) {
        if (!rd_avail(r, 1)) { r->ok = false; return 0; }
        b = *r->p++;
        v |= ((int64_t)(b & 0x7f)) << shift;
        shift += 7;
        if ((b & 0x80) == 0) break;
        if (shift > 63) { r->ok = false; return 0; }
    }
    if (shift < 64 && (b & 0x40)) v |= -((int64_t)1 << shift);
    return v;
}

// Skip one constant init expression and verify it ends with 0x0b. The
// init expr is a single constant instruction followed by the `end`
// opcode (0x0b). We can't byte-walk to the first 0x0b because const
// payload bytes can legitimately equal 0x0b (e.g. `i32.const 11`
// encodes as 0x41 0x0b 0x0b).
//
// Returns true on success; on success, `r->p` points past the trailing
// 0x0b. Caller can compute the expression range from the saved `start`.
static bool skip_init_expr(Reader *r) {
    if (!rd_avail(r, 1)) return false;
    uint8_t op = *r->p++;
    switch (op) {
        case 0x41: (void)rd_leb_s(r); break;   // i32.const
        case 0x42: (void)rd_leb_s(r); break;   // i64.const
        case 0x43:                              // f32.const
            if (!rd_avail(r, 4)) return false;
            r->p += 4; break;
        case 0x44:                              // f64.const
            if (!rd_avail(r, 8)) return false;
            r->p += 8; break;
        case 0x23: (void)rd_leb_u(r); break;   // global.get
        case 0xd0:                              // ref.null t
            if (!rd_avail(r, 1)) return false;
            r->p += 1; break;
        case 0xd2: (void)rd_leb_u(r); break;   // ref.func
        default:
            // Unknown opcode in an init_expr; we can't safely skip it
            // without knowing its operand layout.
            r->ok = false;
            return false;
    }
    if (!r->ok) return false;
    if (!rd_avail(r, 1) || *r->p != 0x0b) { r->ok = false; return false; }
    r->p++;
    return true;
}

// =============================================================================
// Per-input parsed-object IR
// =============================================================================

typedef struct {
    uint8_t *params;     uint32_t n_params;
    uint8_t *results;    uint32_t n_results;
} FuncType;

typedef enum {
    F_IMPORT,    // env.foo / wasi.foo — needs resolution
    F_DEFINED,   // function body in this object's Code section
} FuncOrigin;

typedef struct {
    FuncOrigin origin;
    uint32_t   type_idx;      // local type idx (pre-remap)
    // Imports:
    char      *imp_module;    // strdup'd
    char      *imp_name;      // strdup'd
    // Defined:
    uint32_t   code_off;      // offset of body within object code section
    uint32_t   code_len;      // body length incl. locals + expr
    // Final resolution:
    uint32_t   final_idx;     // assigned in the linked module
    bool       is_host_import;// true if kept as a wasi-style import
    uint32_t   sym_idx_in_obj;// symbol index (linking section) defining this
} ObjFunc;

typedef struct {
    bool      is_import;
    char     *imp_module;
    char     *imp_name;
    uint8_t   value_type;     // 0x7f i32 / 0x7e i64 / 0x7d f32 / 0x7c f64
    bool      is_mutable;
    // For defined: raw init-expr bytes (incl. terminating 0x0b).
    uint8_t  *init_expr;
    uint32_t  init_expr_len;
    // Final resolution:
    uint32_t  final_idx;
    bool      is_host_import;
} ObjGlobal;

typedef struct {
    uint8_t  type;          // R_WASM_*
    uint32_t section_off;   // byte offset within the section payload
    uint32_t sym_idx;       // index into ObjSymbol[]
    int64_t  addend;
} ObjReloc;

typedef enum {
    K_FUNC = SYM_FUNCTION,
    K_DATA = SYM_DATA,
    K_GLOBAL = SYM_GLOBAL,
    K_SECTION = SYM_SECTION,
    K_EVENT = SYM_EVENT,
    K_TABLE = SYM_TABLE,
} SymKind;

typedef struct {
    SymKind   kind;
    uint32_t  flags;          // SYMF_*
    char     *name;           // strdup'd; NULL means no explicit name
    // For FUNCTION/GLOBAL/EVENT/TABLE: object-local element index.
    uint32_t  idx;
    // For DATA: segment idx + byte offset + size within segment.
    uint32_t  data_seg;
    uint32_t  data_off;
    uint32_t  data_size;
    // Resolution result (filled by global symbol merge):
    // index into the global Symbol[] table that "owns" this symbol.
    int32_t   global_sym;
} ObjSymbol;

typedef struct {
    char     *name;           // strdup'd or NULL
    uint32_t  flags;
    uint8_t  *data;           // raw bytes
    uint32_t  size;
    bool      is_bss;         // .bss.* — zero-init, not emitted
    // Final base address in linear memory.
    uint32_t  final_addr;
} ObjDataSeg;

typedef struct {
    // Tag indicating whether this is an active (table_idx + offset)
    // segment or a passive one. We only support active segments with
    // a constant i32 offset.
    bool      is_active;
    uint32_t  table_idx;
    int32_t   offset_const;   // resolved at link time if reloc'd
    bool      offset_is_reloc;// true: offset replaced by linker base
    uint32_t *func_ids;       // local function indices (pre-remap)
    uint32_t  n_funcs;
} ObjElemSeg;

typedef struct {
    const MLIR_WasmLinkInput *src;

    // Section payloads (pointers into src->data + offsets in the
    // source object; not owned).
    const uint8_t *type_p; size_t type_n;
    const uint8_t *import_p; size_t import_n;
    const uint8_t *function_p; size_t function_n;
    const uint8_t *table_p; size_t table_n;
    const uint8_t *memory_p; size_t memory_n;
    const uint8_t *global_p; size_t global_n;
    const uint8_t *export_p; size_t export_n;
    const uint8_t *element_p; size_t element_n;
    const uint8_t *code_p; size_t code_n;
    const uint8_t *code_section_payload_start; // start of code payload after count
    const uint8_t *data_p; size_t data_n;
    const uint8_t *linking_p; size_t linking_n;
    const uint8_t *reloc_code_p; size_t reloc_code_n;
    const uint8_t *reloc_data_p; size_t reloc_data_n;

    FuncType   *types;     uint32_t n_types;
    uint32_t   *type_remap;// local type idx -> global type idx

    ObjFunc    *funcs;     uint32_t n_funcs;
    ObjGlobal  *globals;   uint32_t n_globals;
    ObjSymbol  *syms;      uint32_t n_syms;
    ObjDataSeg *datasegs;  uint32_t n_datasegs;
    ObjElemSeg *elemsegs;  uint32_t n_elemsegs;

    // Imported table names, indexed by local table index. Tables are
    // referenced by SYM_TABLE symbols and reloc.CODE entries via the
    // local table index space.
    char     **table_imp_names; uint32_t n_tables;

    ObjReloc   *code_relocs; uint32_t n_code_relocs;
    ObjReloc   *data_relocs; uint32_t n_data_relocs;
} Obj;

// =============================================================================
// Read an object file
// =============================================================================
static bool read_func_type(Reader *r, FuncType *t) {
    if (rd_u8(r) != 0x60) return false;
    uint32_t np = (uint32_t)rd_leb_u(r);
    if (!r->ok || !rd_avail(r, np)) return false;
    t->n_params = np;
    t->params = (uint8_t *)malloc(np ? np : 1);
    memcpy(t->params, r->p, np);
    r->p += np;
    uint32_t nr = (uint32_t)rd_leb_u(r);
    if (!r->ok || !rd_avail(r, nr)) return false;
    t->n_results = nr;
    t->results = (uint8_t *)malloc(nr ? nr : 1);
    memcpy(t->results, r->p, nr);
    r->p += nr;
    return true;
}

static char *strdup_bytes(const uint8_t *p, size_t n) {
    char *s = (char *)malloc(n + 1);
    memcpy(s, p, n);
    s[n] = 0;
    return s;
}

// Parse the top-level structure: locate every wasm section by id, plus
// the three custom sections we care about. Leaves payload pointers in
// `o` referring to bytes inside `src->data`.
static bool obj_locate_sections(Obj *o) {
    Reader r = rd_make(o->src->data, o->src->size);
    if (rd_u32_le(&r) != 0x6d736100u) { link_err("bad magic in %s", o->src->name); return false; }
    if (rd_u32_le(&r) != 1u) { link_err("bad version in %s", o->src->name); return false; }
    while (r.ok && r.p < r.end) {
        uint8_t  id = rd_u8(&r);
        uint64_t sz = rd_leb_u(&r);
        if (!r.ok || !rd_avail(&r, sz)) { link_err("truncated section in %s", o->src->name); return false; }
        const uint8_t *payload = r.p;
        const uint8_t *next    = r.p + sz;
        if (id == SEC_CUSTOM) {
            Reader cr = rd_make(payload, sz);
            uint32_t nlen = (uint32_t)rd_leb_u(&cr);
            if (!cr.ok || !rd_avail(&cr, nlen)) goto skip;
            const char *cname = (const char *)cr.p;
            size_t       cn   = nlen;
            cr.p += nlen;
            const uint8_t *body = cr.p;
            size_t bn = (size_t)(next - cr.p);
            if (cn == 7 && memcmp(cname, "linking", 7) == 0) {
                o->linking_p = body; o->linking_n = bn;
            } else if (cn == 10 && memcmp(cname, "reloc.CODE", 10) == 0) {
                o->reloc_code_p = body; o->reloc_code_n = bn;
            } else if (cn == 10 && memcmp(cname, "reloc.DATA", 10) == 0) {
                o->reloc_data_p = body; o->reloc_data_n = bn;
            }
        } else if (id == SEC_TYPE)    { o->type_p = payload; o->type_n = sz; }
        else if (id == SEC_IMPORT)    { o->import_p = payload; o->import_n = sz; }
        else if (id == SEC_FUNCTION)  { o->function_p = payload; o->function_n = sz; }
        else if (id == SEC_TABLE)     { o->table_p = payload; o->table_n = sz; }
        else if (id == SEC_MEMORY)    { o->memory_p = payload; o->memory_n = sz; }
        else if (id == SEC_GLOBAL)    { o->global_p = payload; o->global_n = sz; }
        else if (id == SEC_EXPORT)    { o->export_p = payload; o->export_n = sz; }
        else if (id == SEC_ELEMENT)   { o->element_p = payload; o->element_n = sz; }
        else if (id == SEC_CODE)      { o->code_p = payload; o->code_n = sz; }
        else if (id == SEC_DATA)      { o->data_p = payload; o->data_n = sz; }
        // ignore SEC_START, SEC_DATACNT, and any unknown ids.
    skip:
        r.p = next;
    }
    return true;
}

static bool obj_parse_types(Obj *o) {
    if (!o->type_p) return true;
    Reader r = rd_make(o->type_p, o->type_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    o->types = (FuncType *)calloc(n ? n : 1, sizeof(FuncType));
    o->n_types = n;
    for (uint32_t i = 0; i < n; i++) {
        if (!read_func_type(&r, &o->types[i])) {
            link_err("type parse fail at %u in %s", i, o->src->name);
            return false;
        }
    }
    return true;
}

static bool obj_parse_imports(Obj *o) {
    if (!o->import_p) return true;
    Reader r = rd_make(o->import_p, o->import_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t mod_n = (uint32_t)rd_leb_u(&r);
        if (!rd_avail(&r, mod_n)) return false;
        const uint8_t *mod_p = r.p; r.p += mod_n;
        uint32_t name_n = (uint32_t)rd_leb_u(&r);
        if (!rd_avail(&r, name_n)) return false;
        const uint8_t *name_p = r.p; r.p += name_n;
        uint8_t kind = rd_u8(&r);
        if (kind == IMP_FUNC) {
            uint32_t tidx = (uint32_t)rd_leb_u(&r);
            o->funcs = (ObjFunc *)realloc(o->funcs, (o->n_funcs + 1) * sizeof(ObjFunc));
            ObjFunc *f = &o->funcs[o->n_funcs++];
            memset(f, 0, sizeof(*f));
            f->origin = F_IMPORT;
            f->type_idx = tidx;
            f->imp_module = strdup_bytes(mod_p, mod_n);
            f->imp_name = strdup_bytes(name_p, name_n);
        } else if (kind == IMP_TABLE) {
            (void)rd_u8(&r);            // reftype
            uint8_t lim = rd_u8(&r);
            (void)rd_leb_u(&r);         // min
            if (lim & 0x01) (void)rd_leb_u(&r); // max
            o->table_imp_names = (char **)realloc(
                o->table_imp_names,
                (o->n_tables + 1) * sizeof(char *));
            o->table_imp_names[o->n_tables++] = strdup_bytes(name_p, name_n);
            // The linker provides one materialised table at index 0.
        } else if (kind == IMP_MEMORY) {
            uint8_t lim = rd_u8(&r);
            (void)rd_leb_u(&r);
            if (lim & 0x01) (void)rd_leb_u(&r);
            // Same: one materialised memory in the output.
        } else if (kind == IMP_GLOBAL) {
            uint8_t vt = rd_u8(&r);
            uint8_t mut = rd_u8(&r);
            o->globals = (ObjGlobal *)realloc(o->globals, (o->n_globals + 1) * sizeof(ObjGlobal));
            ObjGlobal *g = &o->globals[o->n_globals++];
            memset(g, 0, sizeof(*g));
            g->is_import = true;
            g->imp_module = strdup_bytes(mod_p, mod_n);
            g->imp_name = strdup_bytes(name_p, name_n);
            g->value_type = vt;
            g->is_mutable = (mut != 0);
        } else {
            link_err("unsupported import kind %u in %s", kind, o->src->name);
            return false;
        }
        if (!r.ok) return false;
    }
    return true;
}

static bool obj_parse_function_section(Obj *o) {
    if (!o->function_p) return true;
    Reader r = rd_make(o->function_p, o->function_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t tidx = (uint32_t)rd_leb_u(&r);
        o->funcs = (ObjFunc *)realloc(o->funcs, (o->n_funcs + 1) * sizeof(ObjFunc));
        ObjFunc *f = &o->funcs[o->n_funcs++];
        memset(f, 0, sizeof(*f));
        f->origin = F_DEFINED;
        f->type_idx = tidx;
    }
    return r.ok;
}

static bool obj_parse_globals(Obj *o) {
    if (!o->global_p) return true;
    Reader r = rd_make(o->global_p, o->global_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t vt = rd_u8(&r);
        uint8_t mut = rd_u8(&r);
        // Opcode-aware skip of the init expression (handles const
        // payloads that contain 0x0b bytes).
        const uint8_t *start = r.p;
        if (!skip_init_expr(&r)) {
            link_err("bad global init expr at %u in %s", i, o->src->name);
            return false;
        }
        const uint8_t *end = r.p; // already past trailing 0x0b
        o->globals = (ObjGlobal *)realloc(o->globals, (o->n_globals + 1) * sizeof(ObjGlobal));
        ObjGlobal *g = &o->globals[o->n_globals++];
        memset(g, 0, sizeof(*g));
        g->is_import = false;
        g->value_type = vt;
        g->is_mutable = (mut != 0);
        size_t len = (size_t)(end - start);
        g->init_expr = (uint8_t *)malloc(len);
        memcpy(g->init_expr, start, len);
        g->init_expr_len = (uint32_t)len;
    }
    return true;
}

static bool obj_parse_code(Obj *o) {
    if (!o->code_p) return true;
    Reader r = rd_make(o->code_p, o->code_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    o->code_section_payload_start = o->code_p; // section payload incl. count

    // Find the first defined function in o->funcs and walk bodies.
    uint32_t fi = 0;
    while (fi < o->n_funcs && o->funcs[fi].origin == F_IMPORT) fi++;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t sz = (uint32_t)rd_leb_u(&r);
        if (!rd_avail(&r, sz)) return false;
        if (fi + i >= o->n_funcs || o->funcs[fi + i].origin != F_DEFINED) {
            link_err("code/funcsec mismatch in %s", o->src->name);
            return false;
        }
        // code_off = offset within the SECTION PAYLOAD (i.e. starting
        // from the count LEB). This is what reloc.CODE offsets are
        // relative to.
        o->funcs[fi + i].code_off = (uint32_t)(r.p - o->code_p);
        o->funcs[fi + i].code_len = sz;
        r.p += sz;
    }
    return r.ok;
}

static bool obj_parse_data(Obj *o) {
    if (!o->data_p) return true;
    Reader r = rd_make(o->data_p, o->data_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    o->datasegs = (ObjDataSeg *)calloc(n ? n : 1, sizeof(ObjDataSeg));
    o->n_datasegs = n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t flags = (uint32_t)rd_leb_u(&r);
        if (flags == 0) {
            // active: mem 0, i32.const offset, end
            if (rd_u8(&r) != 0x41) return false; (void)rd_leb_s(&r); if (rd_u8(&r) != 0x0b) return false;
        } else if (flags == 2) {
            // active with explicit memory idx
            (void)rd_leb_u(&r);
            if (rd_u8(&r) != 0x41) return false; (void)rd_leb_s(&r); if (rd_u8(&r) != 0x0b) return false;
        }
        // passive (flags & 1): no offset expr.
        uint32_t sz = (uint32_t)rd_leb_u(&r);
        if (!rd_avail(&r, sz)) return false;
        o->datasegs[i].size = sz;
        o->datasegs[i].data = (uint8_t *)malloc(sz ? sz : 1);
        memcpy(o->datasegs[i].data, r.p, sz);
        r.p += sz;
    }
    return r.ok;
}

static bool obj_parse_elements(Obj *o) {
    if (!o->element_p) return true;
    Reader r = rd_make(o->element_p, o->element_n);
    uint32_t n = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    o->elemsegs = (ObjElemSeg *)calloc(n ? n : 1, sizeof(ObjElemSeg));
    o->n_elemsegs = n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t flags = (uint32_t)rd_leb_u(&r);
        ObjElemSeg *e = &o->elemsegs[i];
        if (flags == 0) {
            e->is_active = true;
            e->table_idx = 0;
            // i32.const N (0x41, sleb) end (0x0b)
            if (rd_u8(&r) != 0x41) return false;
            e->offset_const = (int32_t)rd_leb_s(&r);
            if (rd_u8(&r) != 0x0b) return false;
            uint32_t nf = (uint32_t)rd_leb_u(&r);
            e->n_funcs = nf;
            e->func_ids = (uint32_t *)malloc(nf * sizeof(uint32_t));
            for (uint32_t j = 0; j < nf; j++) e->func_ids[j] = (uint32_t)rd_leb_u(&r);
        } else {
            link_err("unsupported element segment flags %u in %s",
                     flags, o->src->name);
            return false;
        }
    }
    return r.ok;
}

static bool obj_parse_linking(Obj *o) {
    if (!o->linking_p) return true; // some hand-written objects may omit it
    Reader r = rd_make(o->linking_p, o->linking_n);
    uint32_t version = (uint32_t)rd_leb_u(&r);
    if (!r.ok || version != 2) {
        link_err("unsupported linking version %u in %s", version, o->src->name);
        return false;
    }
    while (r.ok && r.p < r.end) {
        uint8_t sub_id = rd_u8(&r);
        uint32_t sub_n = (uint32_t)rd_leb_u(&r);
        if (!rd_avail(&r, sub_n)) return false;
        const uint8_t *sub_end = r.p + sub_n;
        if (sub_id == LINK_SUB_SYMBOL_TABLE) {
            uint32_t nsym = (uint32_t)rd_leb_u(&r);
            o->syms = (ObjSymbol *)calloc(nsym ? nsym : 1, sizeof(ObjSymbol));
            o->n_syms = nsym;
            for (uint32_t i = 0; i < nsym; i++) {
                ObjSymbol *s = &o->syms[i];
                s->kind = (SymKind)rd_u8(&r);
                s->flags = (uint32_t)rd_leb_u(&r);
                if (s->kind == SYM_FUNCTION || s->kind == SYM_GLOBAL
                    || s->kind == SYM_EVENT || s->kind == SYM_TABLE) {
                    s->idx = (uint32_t)rd_leb_u(&r);
                    bool has_explicit_name = (s->flags & SYMF_UNDEFINED) == 0
                        || (s->flags & SYMF_EXPLICIT_NAME);
                    if (has_explicit_name) {
                        uint32_t nl = (uint32_t)rd_leb_u(&r);
                        if (!rd_avail(&r, nl)) return false;
                        s->name = strdup_bytes(r.p, nl);
                        r.p += nl;
                    } else {
                        // Name comes implicitly from the import entry
                        // this symbol refers to. Fill it in so symbol
                        // resolution has something to key on.
                        if (s->kind == SYM_FUNCTION && s->idx < o->n_funcs
                            && o->funcs[s->idx].imp_name) {
                            s->name = strdup_bytes(
                                (const uint8_t *)o->funcs[s->idx].imp_name,
                                strlen(o->funcs[s->idx].imp_name));
                        } else if (s->kind == SYM_GLOBAL && s->idx < o->n_globals
                            && o->globals[s->idx].imp_name) {
                            s->name = strdup_bytes(
                                (const uint8_t *)o->globals[s->idx].imp_name,
                                strlen(o->globals[s->idx].imp_name));
                        } else if (s->kind == SYM_TABLE && s->idx < o->n_tables
                            && o->table_imp_names[s->idx]) {
                            s->name = strdup_bytes(
                                (const uint8_t *)o->table_imp_names[s->idx],
                                strlen(o->table_imp_names[s->idx]));
                        }
                    }
                } else if (s->kind == SYM_DATA) {
                    uint32_t nl = (uint32_t)rd_leb_u(&r);
                    if (!rd_avail(&r, nl)) return false;
                    s->name = strdup_bytes(r.p, nl);
                    r.p += nl;
                    if ((s->flags & SYMF_UNDEFINED) == 0) {
                        s->data_seg = (uint32_t)rd_leb_u(&r);
                        s->data_off = (uint32_t)rd_leb_u(&r);
                        s->data_size = (uint32_t)rd_leb_u(&r);
                    }
                } else if (s->kind == SYM_SECTION) {
                    s->idx = (uint32_t)rd_leb_u(&r);
                } else {
                    link_err("unsupported symbol kind %u in %s",
                             s->kind, o->src->name);
                    return false;
                }
                s->global_sym = -1;
            }
        } else if (sub_id == LINK_SUB_SEGMENT_INFO) {
            uint32_t nseg = (uint32_t)rd_leb_u(&r);
            for (uint32_t i = 0; i < nseg; i++) {
                uint32_t nl = (uint32_t)rd_leb_u(&r);
                if (!rd_avail(&r, nl)) return false;
                if (i < o->n_datasegs) {
                    o->datasegs[i].name = strdup_bytes(r.p, nl);
                    // Mark BSS segments by name prefix (lld
                    // convention): contents are zero so we skip
                    // emitting them — wasm linear memory is
                    // zero-initialised.
                    if (nl >= 5 && memcmp(r.p, ".bss.", 5) == 0) {
                        o->datasegs[i].is_bss = true;
                    } else if (nl >= 4 && memcmp(r.p, ".bss", 4) == 0
                               && (nl == 4 || r.p[4] == '.')) {
                        o->datasegs[i].is_bss = true;
                    }
                }
                r.p += nl;
                (void)rd_leb_u(&r); // align
                uint32_t f = (uint32_t)rd_leb_u(&r);
                if (i < o->n_datasegs) o->datasegs[i].flags = f;
            }
        }
        // Skip everything else (INIT_FUNCS, COMDAT, …)
        r.p = sub_end;
    }
    return r.ok;
}

static bool obj_parse_relocs_one(const uint8_t *p, size_t n,
                                  ObjReloc **out, uint32_t *out_n,
                                  const char *src_name, uint32_t *out_section_idx) {
    Reader r = rd_make(p, n);
    *out_section_idx = (uint32_t)rd_leb_u(&r);
    uint32_t nrel = (uint32_t)rd_leb_u(&r);
    if (!r.ok) return false;
    *out = (ObjReloc *)calloc(nrel ? nrel : 1, sizeof(ObjReloc));
    *out_n = nrel;
    for (uint32_t i = 0; i < nrel; i++) {
        ObjReloc *e = &(*out)[i];
        e->type = rd_u8(&r);
        e->section_off = (uint32_t)rd_leb_u(&r);
        e->sym_idx = (uint32_t)rd_leb_u(&r);
        if (reloc_has_addend(e->type)) e->addend = rd_leb_s(&r);
        if (!r.ok) { link_err("reloc parse fail in %s", src_name); return false; }
    }
    return true;
}

static bool obj_parse_relocs(Obj *o) {
    if (o->reloc_code_p) {
        uint32_t sec_idx;
        if (!obj_parse_relocs_one(o->reloc_code_p, o->reloc_code_n,
                                   &o->code_relocs, &o->n_code_relocs,
                                   o->src->name, &sec_idx)) return false;
    }
    if (o->reloc_data_p) {
        uint32_t sec_idx;
        if (!obj_parse_relocs_one(o->reloc_data_p, o->reloc_data_n,
                                   &o->data_relocs, &o->n_data_relocs,
                                   o->src->name, &sec_idx)) return false;
    }
    return true;
}

static bool parse_object(Obj *o, const MLIR_WasmLinkInput *in) {
    memset(o, 0, sizeof(*o));
    o->src = in;
    if (!obj_locate_sections(o))         return false;
    if (!obj_parse_types(o))             return false;
    if (!obj_parse_imports(o))           return false;
    if (!obj_parse_function_section(o))  return false;
    if (!obj_parse_globals(o))           return false;
    if (!obj_parse_code(o))              return false;
    if (!obj_parse_data(o))              return false;
    if (!obj_parse_elements(o))          return false;
    if (!obj_parse_linking(o))           return false;
    if (!obj_parse_relocs(o))            return false;
    return true;
}

static void free_object(Obj *o) {
    for (uint32_t i = 0; i < o->n_types; i++) {
        free(o->types[i].params); free(o->types[i].results);
    }
    free(o->types);
    free(o->type_remap);
    for (uint32_t i = 0; i < o->n_funcs; i++) {
        free(o->funcs[i].imp_module); free(o->funcs[i].imp_name);
    }
    free(o->funcs);
    for (uint32_t i = 0; i < o->n_globals; i++) {
        free(o->globals[i].imp_module); free(o->globals[i].imp_name);
        free(o->globals[i].init_expr);
    }
    free(o->globals);
    for (uint32_t i = 0; i < o->n_syms; i++) free(o->syms[i].name);
    free(o->syms);
    for (uint32_t i = 0; i < o->n_datasegs; i++) {
        free(o->datasegs[i].name); free(o->datasegs[i].data);
    }
    free(o->datasegs);
    for (uint32_t i = 0; i < o->n_elemsegs; i++) free(o->elemsegs[i].func_ids);
    free(o->elemsegs);
    free(o->code_relocs);
    free(o->data_relocs);
}

// =============================================================================
// Symbol merge: build a global symbol table keyed by name. Each defined
// symbol becomes a "global symbol"; undefined symbols look up the
// global table by name. wasi-style imports (anything whose
// import_module is not "env") are kept as host imports.
// =============================================================================

typedef struct {
    char     *name;
    SymKind   kind;
    // Which object owns the definition and which obj-local symbol it
    // is. For host-imported funcs, owner is the FIRST object that
    // imported it (we keep its import_module/import_name).
    int32_t   def_obj;       // -1 if still undefined (error after merge)
    uint32_t  def_sym;       // index into Obj.syms[]
    bool      is_host_import;
    // Set after final index assignment:
    uint32_t  final_idx;     // function/global/table idx in linked module
    uint32_t  final_addr;    // for K_DATA — base address in memory
} GlobalSym;

typedef struct {
    GlobalSym *e;
    uint32_t   n;
    uint32_t   cap;
} GlobalSymTab;

static int gst_find(GlobalSymTab *g, const char *name) {
    for (uint32_t i = 0; i < g->n; i++) {
        if (strcmp(g->e[i].name, name) == 0) return (int)i;
    }
    return -1;
}
static int gst_add(GlobalSymTab *g, const char *name, SymKind kind) {
    if (g->n == g->cap) {
        g->cap = g->cap ? g->cap * 2 : 64;
        g->e = (GlobalSym *)realloc(g->e, g->cap * sizeof(GlobalSym));
    }
    GlobalSym *s = &g->e[g->n];
    memset(s, 0, sizeof(*s));
    s->name = (char *)malloc(strlen(name) + 1);
    strcpy(s->name, name);
    s->kind = kind;
    s->def_obj = -1;
    return (int)g->n++;
}
static void gst_free(GlobalSymTab *g) {
    for (uint32_t i = 0; i < g->n; i++) free(g->e[i].name);
    free(g->e);
}

// "env" placeholder imports are materialised by the linker, not kept
// as host imports.
static bool is_env_placeholder(const char *module, const char *name) {
    if (!module || strcmp(module, "env") != 0) return false;
    return strcmp(name, "__linear_memory") == 0
        || strcmp(name, "__stack_pointer") == 0
        || strcmp(name, "__indirect_function_table") == 0
        || strcmp(name, "__memory_base") == 0
        || strcmp(name, "__table_base") == 0
        || strcmp(name, "__heap_base") == 0
        || strcmp(name, "__data_end") == 0;
}

static bool merge_symbols(Obj *objs, uint32_t n_objs, GlobalSymTab *gst,
                          bool *need_indirect_table) {
    // Pass 1: add every defined non-local symbol to the global table.
    for (uint32_t oi = 0; oi < n_objs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t si = 0; si < o->n_syms; si++) {
            ObjSymbol *s = &o->syms[si];
            if (s->flags & SYMF_BINDING_LOCAL) continue;
            if (s->flags & SYMF_UNDEFINED) continue;
            if (!s->name) continue; // section symbols
            int gi = gst_find(gst, s->name);
            if (gi < 0) {
                gi = gst_add(gst, s->name, s->kind);
                gst->e[gi].def_obj = (int32_t)oi;
                gst->e[gi].def_sym = si;
            } else {
                bool cur_weak = (objs[gst->e[gi].def_obj].syms[gst->e[gi].def_sym].flags & SYMF_BINDING_WEAK) != 0;
                bool new_weak = (s->flags & SYMF_BINDING_WEAK) != 0;
                if (cur_weak && !new_weak) {
                    gst->e[gi].def_obj = (int32_t)oi;
                    gst->e[gi].def_sym = si;
                }
                // else: same/strong-vs-weak → keep first; duplicate
                // strong defs are tolerated (matches lld --allow-multiple-definition behavior for our limited cases).
            }
            s->global_sym = gi;
        }
    }
    // Pass 2: undefined symbols.
    for (uint32_t oi = 0; oi < n_objs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t si = 0; si < o->n_syms; si++) {
            ObjSymbol *s = &o->syms[si];
            if ((s->flags & SYMF_UNDEFINED) == 0) continue;
            if (!s->name) continue;
            int gi = gst_find(gst, s->name);
            if (gi >= 0) {
                s->global_sym = gi;
                continue;
            }
            // Undefined and not yet seen — possibly a host import.
            // For host imports we need an explicit non-"env" module
            // (which lives on the corresponding ObjFunc/ObjGlobal).
            if (s->kind == SYM_FUNCTION) {
                if (s->idx >= o->n_funcs) {
                    link_err("undefined func sym %s with bad idx in %s",
                             s->name, o->src->name);
                    return false;
                }
                ObjFunc *f = &o->funcs[s->idx];
                if (f->origin == F_IMPORT && f->imp_module
                    && !is_env_placeholder(f->imp_module, f->imp_name)) {
                    gi = gst_add(gst, s->name, s->kind);
                    gst->e[gi].def_obj = (int32_t)oi;
                    gst->e[gi].def_sym = si;
                    gst->e[gi].is_host_import = true;
                    s->global_sym = gi;
                    continue;
                }
                if (f->origin == F_IMPORT && f->imp_module
                    && is_env_placeholder(f->imp_module, f->imp_name)) {
                    // env.* function placeholder — linker materialises
                    // (or it's a forward-reference that another object
                    // defines). The forward-reference case is already
                    // handled by Pass 1 above; if we reach here it's
                    // a true env placeholder, which has no body — we
                    // just mark as unresolved zero.
                    continue;
                }
            } else if (s->kind == SYM_GLOBAL) {
                if (s->idx >= o->n_globals) {
                    link_err("undefined global sym %s with bad idx in %s",
                             s->name, o->src->name);
                    return false;
                }
                ObjGlobal *g = &o->globals[s->idx];
                if (g->is_import && g->imp_module
                    && !is_env_placeholder(g->imp_module, g->imp_name)) {
                    gi = gst_add(gst, s->name, s->kind);
                    gst->e[gi].def_obj = (int32_t)oi;
                    gst->e[gi].def_sym = si;
                    gst->e[gi].is_host_import = true;
                    s->global_sym = gi;
                    continue;
                }
                if (g->is_import && g->imp_module
                    && is_env_placeholder(g->imp_module, g->imp_name)) {
                    // env.__stack_pointer / env.__memory_base / etc.
                    // Resolve against the materialised entity by name.
                    // Add a synthetic global entry so reloc.CODE has
                    // something to point at; final_idx will be set
                    // below after we assign material indices.
                    int egi = gst_find(gst, s->name);
                    if (egi < 0) {
                        egi = gst_add(gst, s->name, s->kind);
                        gst->e[egi].def_obj = (int32_t)oi;
                        gst->e[egi].def_sym = si;
                        gst->e[egi].is_host_import = false;
                    }
                    s->global_sym = egi;
                    continue;
                }
            } else if (s->kind == SYM_TABLE) {
                // env.__indirect_function_table — linker materialises.
                *need_indirect_table = true;
                int egi = gst_find(gst, s->name ? s->name : "__indirect_function_table");
                if (egi < 0) {
                    egi = gst_add(gst, s->name ? s->name : "__indirect_function_table", s->kind);
                    gst->e[egi].def_obj = (int32_t)oi;
                    gst->e[egi].def_sym = si;
                    gst->e[egi].final_idx = 0;
                }
                s->global_sym = egi;
                continue;
            }
            // Weak undefined → unresolved is OK (resolves to 0 / null);
            // non-weak undefined that wasn't resolved is an error.
            if ((s->flags & SYMF_BINDING_WEAK) == 0
                && !(s->kind == SYM_DATA && (s->flags & SYMF_BINDING_WEAK))) {
                link_err("undefined symbol: %s (from %s)",
                         s->name, o->src->name);
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Type deduplication
// =============================================================================
static bool ftype_eq(const FuncType *a, const FuncType *b) {
    if (a->n_params != b->n_params || a->n_results != b->n_results) return false;
    if (a->n_params && memcmp(a->params, b->params, a->n_params)) return false;
    if (a->n_results && memcmp(a->results, b->results, a->n_results)) return false;
    return true;
}

typedef struct {
    FuncType *e;
    uint32_t  n;
    uint32_t  cap;
} TypeTab;

static uint32_t ttab_intern(TypeTab *t, const FuncType *ft) {
    for (uint32_t i = 0; i < t->n; i++) {
        if (ftype_eq(&t->e[i], ft)) return i;
    }
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 32;
        t->e = (FuncType *)realloc(t->e, t->cap * sizeof(FuncType));
    }
    FuncType *dst = &t->e[t->n];
    dst->n_params = ft->n_params;
    dst->params = (uint8_t *)malloc(ft->n_params ? ft->n_params : 1);
    memcpy(dst->params, ft->params, ft->n_params);
    dst->n_results = ft->n_results;
    dst->results = (uint8_t *)malloc(ft->n_results ? ft->n_results : 1);
    memcpy(dst->results, ft->results, ft->n_results);
    return t->n++;
}
static void ttab_free(TypeTab *t) {
    for (uint32_t i = 0; i < t->n; i++) {
        free(t->e[i].params); free(t->e[i].results);
    }
    free(t->e);
}

// =============================================================================
// Relocation application: patch a 5-byte padded LEB at section_off
// inside the per-function code body or per-segment data bytes.
// =============================================================================
static void patch_u32_leb5(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 5; i++) {
        uint8_t b = v & 0x7f;
        v >>= 7;
        if (i < 4) b |= 0x80;
        p[i] = b;
    }
}
static void patch_s32_leb5(uint8_t *p, int32_t v) {
    uint32_t u = (uint32_t)v;
    for (int i = 0; i < 5; i++) {
        uint8_t b = u & 0x7f;
        u >>= 7;
        if (i < 4) b |= 0x80;
        p[i] = b;
    }
}
static void patch_i32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v); p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

// Function symbols referenced via a TABLE_INDEX reloc must resolve to the
// indirect-function-table slot, not the final function index. The slot
// map is keyed by final func idx and is built once during link layout.
// Returns the original value when the symbol/reloc combination doesn't
// require remapping, or when the function hasn't been assigned a slot.
static uint32_t remap_func_to_table_slot(uint32_t value,
                                         uint8_t reloc_type,
                                         SymKind sym_kind,
                                         const uint32_t *slot_map,
                                         uint32_t slot_map_n) {
    if (sym_kind != SYM_FUNCTION) return value;
    if (reloc_type != R_WASM_TABLE_INDEX_SLEB
        && reloc_type != R_WASM_TABLE_INDEX_I32
        && reloc_type != R_WASM_TABLE_INDEX_REL_SLEB) return value;
    if (value < slot_map_n && slot_map[value] != 0) return slot_map[value];
    return value;
}

// Resolve a symbol reference to its final value.
//   - SYM_FUNCTION → final function index
//   - SYM_GLOBAL   → final global index
//   - SYM_DATA     → final linear-memory address (without addend; caller adds it)
//   - SYM_TABLE    → final table index
// Local symbols (BINDING_LOCAL) bypass the GlobalSymTab and resolve
// directly against the parent object's tables. Weak undefined symbols
// resolve to 0. Returns false on unresolvable strong reference.
static bool resolve_sym_value(const Obj *o, const ObjSymbol *s,
                              const GlobalSymTab *gst,
                              uint32_t *value_out) {
    // For undefined function imports we keep the redirect target on the
    // local ObjFunc entry (so per-caller signature-mismatch stubs are
    // applied). For all other globally-resolved symbols, defer to the
    // GlobalSymTab.
    if ((s->flags & SYMF_UNDEFINED) && s->kind == SYM_FUNCTION
        && s->idx < o->n_funcs) {
        *value_out = o->funcs[s->idx].final_idx;
        return true;
    }
    if (s->global_sym >= 0) {
        if (s->kind == SYM_DATA) {
            *value_out = gst->e[s->global_sym].final_addr;
        } else {
            *value_out = gst->e[s->global_sym].final_idx;
        }
        return true;
    }
    // No global-sym entry — must be a local definition or weak undef.
    if ((s->flags & SYMF_UNDEFINED) == 0) {
        switch (s->kind) {
            case SYM_FUNCTION:
                if (s->idx < o->n_funcs) {
                    *value_out = o->funcs[s->idx].final_idx;
                    return true;
                }
                break;
            case SYM_GLOBAL:
                if (s->idx < o->n_globals) {
                    *value_out = o->globals[s->idx].final_idx;
                    return true;
                }
                break;
            case SYM_DATA:
                if (s->data_seg < o->n_datasegs) {
                    *value_out = o->datasegs[s->data_seg].final_addr
                               + s->data_off;
                    return true;
                }
                break;
            case SYM_TABLE:
                *value_out = 0;
                return true;
            case SYM_SECTION:
                *value_out = 0;
                return true;
            default: break;
        }
    }
    if (s->flags & SYMF_BINDING_WEAK) {
        *value_out = 0;
        return true;
    }
    *value_out = 0;
    return false;
}

// =============================================================================
// Top-level link()
// =============================================================================

// Configuration constants.
enum {
    PAGE_SIZE     = 65536u,
    STACK_SIZE    = 64u * 1024u,    // 64 KB stack, matches wasm-ld default
    INITIAL_PAGES = 2u,             // grown as needed below
    GLOBAL_BASE_OFFSET = 1024u,     // wasm-ld default: leave low 1 KB unused
};

bool MLIR_WasmLink(const MLIR_WasmLinkInput *inputs, size_t n_inputs,
                   const char *entry_export,
                   uint8_t **out_data, size_t *out_size) {
    *out_data = NULL; *out_size = 0;
    if (n_inputs == 0) return false;

    Obj *objs = (Obj *)calloc(n_inputs, sizeof(Obj));
    bool ok = true;
    for (size_t i = 0; i < n_inputs && ok; i++) {
        if (!parse_object(&objs[i], &inputs[i])) ok = false;
    }
    if (!ok) goto done;

    // ---- Symbol table merge ---------------------------------------------
    GlobalSymTab gst = {0};
    bool need_indirect_table = false;
    if (!merge_symbols(objs, (uint32_t)n_inputs, &gst, &need_indirect_table)) {
        ok = false; goto done;
    }

    // ---- Type table dedup ----------------------------------------------
    TypeTab ttab = {0};
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        o->type_remap = (uint32_t *)calloc(o->n_types ? o->n_types : 1,
                                            sizeof(uint32_t));
        for (uint32_t i = 0; i < o->n_types; i++) {
            o->type_remap[i] = ttab_intern(&ttab, &o->types[i]);
        }
    }

    // ---- Assign final function indices ---------------------------------
    // Layout: host imports first, then defined functions in object order.
    // Walk global symbols to enumerate host imports; assign final_idx in
    // declaration order across objects.
    uint32_t next_func_idx = 0;

    // 1) Host-imported functions, deduplicated by global symbol.
    for (uint32_t gi = 0; gi < gst.n; gi++) {
        if (gst.e[gi].kind != K_FUNC || !gst.e[gi].is_host_import) continue;
        gst.e[gi].final_idx = next_func_idx++;
    }
    // 2) Defined functions: walk each object in order, walk its
    // funcs[] in order, skip imports (those resolve via the global
    // symbol they reference), assign final_idx to defined ones.
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t fi = 0; fi < o->n_funcs; fi++) {
            if (o->funcs[fi].origin != F_DEFINED) continue;
            o->funcs[fi].final_idx = next_func_idx++;
            // If a global symbol points at this defined function,
            // record final_idx on the symbol too.
            for (uint32_t si = 0; si < o->n_syms; si++) {
                ObjSymbol *s = &o->syms[si];
                if (s->kind != SYM_FUNCTION) continue;
                if (s->idx != fi) continue;
                if (s->flags & SYMF_UNDEFINED) continue;
                if (s->global_sym >= 0) {
                    gst.e[s->global_sym].final_idx = o->funcs[fi].final_idx;
                }
            }
        }
    }
    // 3) Resolve each object's import functions to a final_idx by
    // looking up the symbol pointing at the import. If the caller-side
    // import signature differs from the resolved definition's signature,
    // route through a trapping stub (matches wasm-ld's
    // `signature_mismatch:<name>` behaviour).
    typedef struct {
        int32_t  global_sym;     // gst sym this stub redirects from
        uint32_t expected_type;  // final type idx the caller expects
        uint32_t final_idx;      // assigned final func idx
    } StubFunc;
    StubFunc *stubs = NULL;
    uint32_t  n_stubs = 0;
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t si = 0; si < o->n_syms; si++) {
            ObjSymbol *s = &o->syms[si];
            if (s->kind != SYM_FUNCTION) continue;
            if ((s->flags & SYMF_UNDEFINED) == 0) continue;
            if (s->global_sym < 0) {
                if (s->idx < o->n_funcs) o->funcs[s->idx].final_idx = 0;
                continue;
            }
            if (s->idx >= o->n_funcs) continue;
            GlobalSym *gs = &gst.e[s->global_sym];
            uint32_t caller_type_final = o->type_remap[o->funcs[s->idx].type_idx];
            bool mismatched = false;
            if (!gs->is_host_import && gs->def_obj >= 0) {
                ObjSymbol *dsym = &objs[gs->def_obj].syms[gs->def_sym];
                if (dsym->idx < objs[gs->def_obj].n_funcs) {
                    ObjFunc *df = &objs[gs->def_obj].funcs[dsym->idx];
                    uint32_t real_type_final = objs[gs->def_obj].type_remap[df->type_idx];
                    if (real_type_final != caller_type_final) mismatched = true;
                }
            }
            if (mismatched) {
                int32_t stub_idx = -1;
                for (uint32_t k = 0; k < n_stubs; k++) {
                    if (stubs[k].global_sym == s->global_sym
                        && stubs[k].expected_type == caller_type_final) {
                        stub_idx = (int32_t)k; break;
                    }
                }
                if (stub_idx < 0) {
                    stubs = (StubFunc *)realloc(stubs, (n_stubs + 1) * sizeof(StubFunc));
                    stubs[n_stubs].global_sym = s->global_sym;
                    stubs[n_stubs].expected_type = caller_type_final;
                    stubs[n_stubs].final_idx = next_func_idx++;
                    stub_idx = (int32_t)n_stubs++;
                }
                o->funcs[s->idx].final_idx = stubs[stub_idx].final_idx;
                o->funcs[s->idx].is_host_import = false;
            } else {
                o->funcs[s->idx].final_idx = gs->final_idx;
                o->funcs[s->idx].is_host_import = gs->is_host_import;
            }
        }
    }

    // ---- Assign final global indices -----------------------------------
    // Layout: linker-materialised __stack_pointer at idx 0, then host
    // global imports, then defined globals.
    uint32_t sp_global_idx = 0;
    uint32_t next_global_idx = 1;
    for (uint32_t gi = 0; gi < gst.n; gi++) {
        if (gst.e[gi].kind != K_GLOBAL || !gst.e[gi].is_host_import) continue;
        gst.e[gi].final_idx = next_global_idx++;
    }
    uint32_t first_defined_global = next_global_idx;
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_globals; i++) {
            if (o->globals[i].is_import) continue;
            o->globals[i].final_idx = next_global_idx++;
            for (uint32_t si = 0; si < o->n_syms; si++) {
                ObjSymbol *s = &o->syms[si];
                if (s->kind != SYM_GLOBAL) continue;
                if (s->idx != i) continue;
                if (s->flags & SYMF_UNDEFINED) continue;
                if (s->global_sym >= 0) {
                    gst.e[s->global_sym].final_idx = o->globals[i].final_idx;
                }
            }
        }
    }
    // Resolve env.__stack_pointer imports to sp_global_idx.
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_globals; i++) {
            ObjGlobal *g = &o->globals[i];
            if (!g->is_import) continue;
            if (g->imp_module && strcmp(g->imp_module, "env") == 0
                && g->imp_name && strcmp(g->imp_name, "__stack_pointer") == 0) {
                g->final_idx = sp_global_idx;
            } else if (g->is_import) {
                // Resolve via symbol table.
                for (uint32_t si = 0; si < o->n_syms; si++) {
                    ObjSymbol *s = &o->syms[si];
                    if (s->kind != SYM_GLOBAL) continue;
                    if (s->idx != i) continue;
                    if (s->global_sym >= 0) {
                        g->final_idx = gst.e[s->global_sym].final_idx;
                    }
                }
            }
        }
    }
    // Propagate __stack_pointer's materialised idx into gst so that
    // R_WASM_GLOBAL_INDEX_LEB relocations against it resolve correctly.
    for (uint32_t gi = 0; gi < gst.n; gi++) {
        if (gst.e[gi].kind != K_GLOBAL) continue;
        if (gst.e[gi].name && strcmp(gst.e[gi].name, "__stack_pointer") == 0
            && !gst.e[gi].is_host_import) {
            gst.e[gi].final_idx = sp_global_idx;
        }
    }

    // ---- Assign data segment base addresses ----------------------------
    // Layout: stack [0, STACK_SIZE); data starts at GLOBAL_BASE_OFFSET
    // past the stack — wasm-ld uses --stack-first so the stack lives
    // below data. Match that.
    uint32_t cur_addr = STACK_SIZE + GLOBAL_BASE_OFFSET;
    // Round up to 16-byte alignment for each segment, since some
    // platforms expect at least 16B aligned heap blocks.
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_datasegs; i++) {
            cur_addr = (cur_addr + 15u) & ~15u;
            o->datasegs[i].final_addr = cur_addr;
            cur_addr += o->datasegs[i].size;
        }
    }
    // Resolve K_DATA symbols to their final_addr = seg.final_addr + data_off
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t si = 0; si < o->n_syms; si++) {
            ObjSymbol *s = &o->syms[si];
            if (s->kind != SYM_DATA) continue;
            if (s->flags & SYMF_UNDEFINED) continue;
            if (s->data_seg >= o->n_datasegs) continue;
            uint32_t addr = o->datasegs[s->data_seg].final_addr + s->data_off;
            if (s->global_sym >= 0) gst.e[s->global_sym].final_addr = addr;
        }
    }
    uint32_t data_end = cur_addr;
    uint32_t heap_base = (data_end + 15u) & ~15u;
    uint32_t total_pages = ((heap_base + PAGE_SIZE - 1u) / PAGE_SIZE) + 1u;
    if (total_pages < INITIAL_PAGES) total_pages = INITIAL_PAGES;

    // ---- Assign indirect-function-table slots --------------------------
    // Each function referenced by any input element segment gets a
    // unique table slot starting at 1 (slot 0 reserved as null). The
    // slot map is keyed by final function index.
    uint32_t  table_slot_map_n = next_func_idx;
    uint32_t *table_slot_map = (uint32_t *)calloc(table_slot_map_n ? table_slot_map_n : 1, sizeof(uint32_t));
    uint32_t  table_slots_used = 1; // slot 0 reserved
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_elemsegs; i++) {
            ObjElemSeg *es = &o->elemsegs[i];
            for (uint32_t j = 0; j < es->n_funcs; j++) {
                uint32_t local = es->func_ids[j];
                if (local >= o->n_funcs) continue;
                uint32_t fin = o->funcs[local].final_idx;
                if (fin < table_slot_map_n && table_slot_map[fin] == 0) {
                    table_slot_map[fin] = table_slots_used++;
                }
            }
        }
    }
    if (table_slots_used > 1) need_indirect_table = true;

    // ---- Apply relocations ---------------------------------------------
    // We need a mutable copy of every defined-function code body and
    // every data segment so we can patch in place. Patching happens in
    // the emit pass below (per-function and per-segment).

reloc_fail:
    if (!ok) {
        ttab_free(&ttab);
        gst_free(&gst);
        goto done;
    }

    // ---- Emit final wasm module ----------------------------------------
    Buf img = {0};
    static const uint8_t magic[8] = {0,'a','s','m', 1,0,0,0};
    buf_append(&img, magic, 8);

    // Type section.
    Buf type_pl = {0};
    leb_u(&type_pl, ttab.n);
    for (uint32_t i = 0; i < ttab.n; i++) {
        buf_putc(&type_pl, 0x60);
        leb_u(&type_pl, ttab.e[i].n_params);
        buf_append(&type_pl, ttab.e[i].params, ttab.e[i].n_params);
        leb_u(&type_pl, ttab.e[i].n_results);
        buf_append(&type_pl, ttab.e[i].results, ttab.e[i].n_results);
    }
    emit_section(&img, SEC_TYPE, &type_pl);
    buf_free(&type_pl);

    // Import section. Only host imports; "env" placeholders are
    // materialised (memory, indirect_function_table, __stack_pointer
    // are emitted as defined entities below).
    Buf imp_pl = {0};
    uint32_t n_imports = 0;
    for (uint32_t gi = 0; gi < gst.n; gi++) {
        if (gst.e[gi].is_host_import) n_imports++;
    }
    leb_u(&imp_pl, n_imports);
    // Iterate in the same order as final_idx assignment (which we did
    // for K_FUNC host imports first, then K_GLOBAL). For now collect
    // and sort by (kind, final_idx).
    // K_FUNC host imports first.
    for (uint32_t pass = 0; pass < 2; pass++) {
        SymKind want = (pass == 0) ? K_FUNC : K_GLOBAL;
        for (uint32_t gi = 0; gi < gst.n; gi++) {
            if (!gst.e[gi].is_host_import) continue;
            if (gst.e[gi].kind != want) continue;
            // Locate the original Obj's imp_module/imp_name.
            Obj *o = &objs[gst.e[gi].def_obj];
            ObjSymbol *s = &o->syms[gst.e[gi].def_sym];
            const char *mod = "env", *nm = gst.e[gi].name;
            if (s->kind == SYM_FUNCTION && s->idx < o->n_funcs) {
                mod = o->funcs[s->idx].imp_module;
                nm  = o->funcs[s->idx].imp_name;
            } else if (s->kind == SYM_GLOBAL && s->idx < o->n_globals) {
                mod = o->globals[s->idx].imp_module;
                nm  = o->globals[s->idx].imp_name;
            }
            emit_string(&imp_pl, mod);
            emit_string(&imp_pl, nm);
            if (want == K_FUNC) {
                buf_putc(&imp_pl, IMP_FUNC);
                ObjFunc *f = &o->funcs[s->idx];
                leb_u(&imp_pl, o->type_remap[f->type_idx]);
            } else {
                buf_putc(&imp_pl, IMP_GLOBAL);
                ObjGlobal *g = &o->globals[s->idx];
                buf_putc(&imp_pl, g->value_type);
                buf_putc(&imp_pl, g->is_mutable ? 1 : 0);
            }
        }
    }
    emit_section(&img, SEC_IMPORT, &imp_pl);
    buf_free(&imp_pl);

    // Function section: list of type indices for each defined function
    // in final_idx order.
    Buf fn_pl = {0};
    uint32_t n_host_func_imports = 0;
    for (uint32_t gi = 0; gi < gst.n; gi++)
        if (gst.e[gi].kind == K_FUNC && gst.e[gi].is_host_import) n_host_func_imports++;
    uint32_t n_def_funcs = next_func_idx - n_host_func_imports;
    leb_u(&fn_pl, n_def_funcs);
    // Walk objects in order, emitting type idx for each defined func.
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_funcs; i++) {
            if (o->funcs[i].origin != F_DEFINED) continue;
            leb_u(&fn_pl, o->type_remap[o->funcs[i].type_idx]);
        }
    }
    // Stubs come last in final_idx order.
    for (uint32_t k = 0; k < n_stubs; k++) {
        leb_u(&fn_pl, stubs[k].expected_type);
    }
    emit_section(&img, SEC_FUNCTION, &fn_pl);
    buf_free(&fn_pl);

    // Table section.
    if (need_indirect_table) {
        Buf t_pl = {0};
        leb_u(&t_pl, 1);
        buf_putc(&t_pl, 0x70);          // funcref
        buf_putc(&t_pl, 0x01);          // limits flags: has_max
        leb_u(&t_pl, table_slots_used);
        leb_u(&t_pl, table_slots_used);
        emit_section(&img, SEC_TABLE, &t_pl);
        buf_free(&t_pl);
    }

    // Memory section.
    Buf m_pl = {0};
    leb_u(&m_pl, 1);
    buf_putc(&m_pl, 0x00);  // no max
    leb_u(&m_pl, total_pages);
    emit_section(&img, SEC_MEMORY, &m_pl);
    buf_free(&m_pl);

    // Global section. __stack_pointer (mutable i32) at the head, then
    // defined globals from each object (init_expr copied as-is — but
    // if it contains a relocation, we apply it).
    Buf g_pl = {0};
    uint32_t n_globals_total = 1; // __stack_pointer
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_globals; i++) if (!o->globals[i].is_import) n_globals_total++;
    }
    leb_u(&g_pl, n_globals_total);
    // __stack_pointer = i32 mut, init = i32.const STACK_SIZE
    buf_putc(&g_pl, 0x7f); buf_putc(&g_pl, 0x01);
    buf_putc(&g_pl, 0x41); leb_s(&g_pl, (int32_t)STACK_SIZE); buf_putc(&g_pl, 0x0b);
    for (size_t oi = 0; oi < n_inputs; oi++) {
        Obj *o = &objs[oi];
        for (uint32_t i = 0; i < o->n_globals; i++) {
            if (o->globals[i].is_import) continue;
            buf_putc(&g_pl, o->globals[i].value_type);
            buf_putc(&g_pl, o->globals[i].is_mutable ? 1 : 0);
            buf_append(&g_pl, o->globals[i].init_expr, o->globals[i].init_expr_len);
        }
    }
    emit_section(&img, SEC_GLOBAL, &g_pl);
    buf_free(&g_pl);
    (void)first_defined_global;

    // Export section.
    Buf e_pl = {0};
    int entry_gi = gst_find(&gst, entry_export);
    uint32_t n_exports = 1; // memory
    if (entry_gi >= 0) n_exports++;
    leb_u(&e_pl, n_exports);
    emit_string(&e_pl, "memory");
    buf_putc(&e_pl, 0x02); leb_u(&e_pl, 0);
    if (entry_gi >= 0) {
        emit_string(&e_pl, entry_export);
        buf_putc(&e_pl, 0x00);
        leb_u(&e_pl, gst.e[entry_gi].final_idx);
    }
    emit_section(&img, SEC_EXPORT, &e_pl);
    buf_free(&e_pl);

    // Element section (one merged active segment at slot 1).
    if (table_slots_used > 1) {
        Buf el_pl = {0};
        leb_u(&el_pl, 1);
        leb_u(&el_pl, 0); // flags=0 active table 0
        buf_putc(&el_pl, 0x41);
        leb_s(&el_pl, 1); // start at slot 1
        buf_putc(&el_pl, 0x0b);
        leb_u(&el_pl, table_slots_used - 1);
        for (uint32_t slot = 1; slot < table_slots_used; slot++) {
            for (uint32_t fi = 0; fi < table_slot_map_n; fi++) {
                if (table_slot_map[fi] == slot) { leb_u(&el_pl, fi); break; }
            }
        }
        emit_section(&img, SEC_ELEMENT, &el_pl);
        buf_free(&el_pl);
    }

    // Code section. Emit each defined func body, applying its code
    // relocations on a per-body copy.
    Buf code_pl = {0};
    leb_u(&code_pl, n_def_funcs);
    for (size_t oi = 0; oi < n_inputs && ok; oi++) {
        Obj *o = &objs[oi];
        // Allocate per-body writable copies.
        for (uint32_t fi = 0; fi < o->n_funcs && ok; fi++) {
            ObjFunc *f = &o->funcs[fi];
            if (f->origin != F_DEFINED) continue;
            uint8_t *body = (uint8_t *)malloc(f->code_len ? f->code_len : 1);
            memcpy(body, o->code_p + f->code_off, f->code_len);
            // Apply all reloc.CODE entries that fall inside this body.
            for (uint32_t ri = 0; ri < o->n_code_relocs && ok; ri++) {
                ObjReloc *r = &o->code_relocs[ri];
                if (r->section_off < f->code_off || r->section_off >= f->code_off + f->code_len) continue;
                uint32_t off = r->section_off - f->code_off;
                if (r->sym_idx >= o->n_syms) {
                    link_err("reloc bad sym in %s", o->src->name); ok = false; break;
                }
                ObjSymbol *s = &o->syms[r->sym_idx];
                uint32_t value = 0;
                int32_t  svalue = 0;
                if (r->type == R_WASM_TYPE_INDEX_LEB) {
                    // Type-index relocs reference the local type table
                    // directly via r->sym_idx (per the linking ABI,
                    // type relocs use a type index rather than a sym
                    // index), so we remap straight from type_remap.
                    if (r->sym_idx < o->n_types) {
                        value = o->type_remap[r->sym_idx];
                    }
                } else {
                    if (!resolve_sym_value(o, s, &gst, &value)) {
                        link_err("reloc to unresolved sym %s in %s",
                                 s->name ? s->name : "(null)", o->src->name);
                        ok = false; break;
                    }
                    if (s->kind == SYM_DATA) value += (uint32_t)r->addend;
                    // For table-index relocations to function symbols,
                    // remap from final-func-idx to its slot in the
                    // merged indirect_function_table.
                    value = remap_func_to_table_slot(value, r->type, s->kind,
                                                     table_slot_map, table_slot_map_n);
                    svalue = (int32_t)value;
                }
                switch (r->type) {
                    case R_WASM_FUNCTION_INDEX_LEB:
                    case R_WASM_TYPE_INDEX_LEB:
                    case R_WASM_GLOBAL_INDEX_LEB:
                    case R_WASM_MEMORY_ADDR_LEB:
                    case R_WASM_TAG_INDEX_LEB:
                    case R_WASM_TABLE_NUMBER_LEB:
                        patch_u32_leb5(body + off, value);
                        break;
                    case R_WASM_TABLE_INDEX_SLEB:
                    case R_WASM_MEMORY_ADDR_SLEB:
                    case R_WASM_MEMORY_ADDR_REL_SLEB:
                    case R_WASM_TABLE_INDEX_REL_SLEB:
                        patch_s32_leb5(body + off, svalue);
                        break;
                    case R_WASM_TABLE_INDEX_I32:
                    case R_WASM_MEMORY_ADDR_I32:
                    case R_WASM_GLOBAL_INDEX_I32:
                    case R_WASM_FUNCTION_OFFSET_I32:
                    case R_WASM_SECTION_OFFSET_I32:
                    case R_WASM_MEMORY_ADDR_LOCREL_I32:
                        patch_i32_le(body + off, value);
                        break;
                    default:
                        link_err("unsupported reloc type %u in %s",
                                 r->type, o->src->name);
                        ok = false; break;
                }
            }
            leb_u(&code_pl, f->code_len);
            buf_append(&code_pl, body, f->code_len);
            free(body);
        }
    }
    // Stub bodies: 3-byte function body (locals=0, unreachable, end).
    for (uint32_t k = 0; k < n_stubs; k++) {
        (void)stubs;
        leb_u(&code_pl, 3);
        buf_putc(&code_pl, 0x00); // 0 local decls
        buf_putc(&code_pl, 0x00); // unreachable
        buf_putc(&code_pl, 0x0b); // end
    }
    if (!ok) { buf_free(&code_pl); ttab_free(&ttab); gst_free(&gst); free(table_slot_map); free(stubs); goto done; }
    emit_section(&img, SEC_CODE, &code_pl);
    buf_free(&code_pl);

    // Data section. Emit each non-BSS segment as an active segment at
    // (i32.const final_addr). BSS segments are zero-initialised
    // implicitly by the wasm runtime, so we skip them.
    uint32_t total_segments = 0;
    for (size_t oi = 0; oi < n_inputs; oi++) {
        for (uint32_t i = 0; i < objs[oi].n_datasegs; i++) {
            if (!objs[oi].datasegs[i].is_bss) total_segments++;
        }
    }
    if (total_segments) {
        Buf d_pl = {0};
        leb_u(&d_pl, total_segments);
        // -- Compute reloc-segment mapping --
        // Parse the data section once more per object to get each
        // segment's payload offset within the section payload.
        for (size_t oi = 0; oi < n_inputs && ok; oi++) {
            Obj *o = &objs[oi];
            if (!o->data_p) continue;
            uint32_t *body_off = (uint32_t *)calloc(o->n_datasegs ? o->n_datasegs : 1, sizeof(uint32_t));
            {
                Reader r = rd_make(o->data_p, o->data_n);
                (void)rd_leb_u(&r); // count
                for (uint32_t i = 0; i < o->n_datasegs; i++) {
                    uint32_t flags = (uint32_t)rd_leb_u(&r);
                    if (flags == 0) { rd_u8(&r); rd_leb_s(&r); rd_u8(&r); }
                    else if (flags == 2) { rd_leb_u(&r); rd_u8(&r); rd_leb_s(&r); rd_u8(&r); }
                    uint32_t sz = (uint32_t)rd_leb_u(&r);
                    body_off[i] = (uint32_t)(r.p - o->data_p);
                    r.p += sz;
                }
            }
            // Patch and emit each non-BSS segment.
            for (uint32_t i = 0; i < o->n_datasegs; i++) {
                if (o->datasegs[i].is_bss) continue;
                uint8_t *bytes = o->datasegs[i].data; // already a malloc'd copy
                // Process data relocs that fall within this segment.
                for (uint32_t ri = 0; ri < o->n_data_relocs && ok; ri++) {
                    ObjReloc *r = &o->data_relocs[ri];
                    if (r->section_off < body_off[i]
                        || r->section_off >= body_off[i] + o->datasegs[i].size) continue;
                    uint32_t off = r->section_off - body_off[i];
                    if (r->sym_idx >= o->n_syms) { ok = false; break; }
                    ObjSymbol *s = &o->syms[r->sym_idx];
                    uint32_t value = 0;
                    if (r->type == R_WASM_TYPE_INDEX_LEB) {
                        if (r->sym_idx < o->n_types) value = o->type_remap[r->sym_idx];
                    } else {
                        if (!resolve_sym_value(o, s, &gst, &value)) {
                            link_err("data reloc to unresolved sym %s in %s",
                                     s->name ? s->name : "(null)", o->src->name);
                            ok = false; break;
                        }
                        if (s->kind == SYM_DATA) value += (uint32_t)r->addend;
                        // For table-index relocations to function symbols
                        // in *data* (e.g. a static initializer that stores
                        // &fn), patch with the table slot, not the final
                        // function index. Matches the CODE-path remap so
                        // indirect calls through that slot land on the
                        // right body.
                        value = remap_func_to_table_slot(value, r->type, s->kind,
                                                         table_slot_map, table_slot_map_n);
                    }
                    switch (r->type) {
                        case R_WASM_MEMORY_ADDR_I32:
                        case R_WASM_TABLE_INDEX_I32:
                        case R_WASM_GLOBAL_INDEX_I32:
                            patch_i32_le(bytes + off, value);
                            break;
                        case R_WASM_MEMORY_ADDR_LEB:
                        case R_WASM_FUNCTION_INDEX_LEB:
                        case R_WASM_TYPE_INDEX_LEB:
                        case R_WASM_GLOBAL_INDEX_LEB:
                            patch_u32_leb5(bytes + off, value);
                            break;
                        case R_WASM_TABLE_INDEX_SLEB:
                        case R_WASM_MEMORY_ADDR_SLEB:
                            patch_s32_leb5(bytes + off, (int32_t)value);
                            break;
                        default:
                            link_err("unsupported data reloc %u in %s",
                                     r->type, o->src->name);
                            ok = false; break;
                    }
                }
                leb_u(&d_pl, 0);  // flags=0 active mem 0
                buf_putc(&d_pl, 0x41); leb_s(&d_pl, (int32_t)o->datasegs[i].final_addr); buf_putc(&d_pl, 0x0b);
                leb_u(&d_pl, o->datasegs[i].size);
                buf_append(&d_pl, bytes, o->datasegs[i].size);
            }
            free(body_off);
        }
        if (ok) emit_section(&img, SEC_DATA, &d_pl);
        buf_free(&d_pl);
    }

    if (!ok) { ttab_free(&ttab); gst_free(&gst); free(table_slot_map); free(stubs); goto done; }

    *out_data = img.data;
    *out_size = img.len;
    img.data = NULL; img.len = img.cap = 0;
    ttab_free(&ttab);
    gst_free(&gst);
    free(table_slot_map);
    free(stubs);

done:
    for (size_t i = 0; i < n_inputs; i++) free_object(&objs[i]);
    free(objs);
    return ok;
}


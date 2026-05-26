// wasm bytecode -> wasmstack builtin.module (MLIR).
// See mlir_wasm_to_wasmstack.h for the public API and conventions.

#include "mlir_wasm_to_wasmstack.h"

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

// =============================================================================
// Wasm value-type constants.
// =============================================================================
#define WT_I32 0x7f
#define WT_I64 0x7e
#define WT_F32 0x7d
#define WT_F64 0x7c

// =============================================================================
// Byte-reader with overflow tracking.
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
static uint32_t rd_f32_bits(Rd *r) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= ((uint32_t)rd_u8(r)) << (8*i);
    return v;
}
static uint64_t rd_f64_bits(Rd *r) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)rd_u8(r)) << (8*i);
    return v;
}

// =============================================================================
// Parsed wasm module.
// =============================================================================
typedef struct {
    uint8_t  n_params, n_results;
    uint8_t *params;   // owned by Arena
    uint8_t *results;  // owned by Arena
} WasmType;

typedef struct {
    uint32_t       type_idx;
    const uint8_t *code;   // function body bytes (locals + opcodes + 0x0b)
    uint32_t       code_size;
    char          *name;   // arena-owned, set after export+import pass
    bool           exported;
} WasmFunc;

typedef struct {
    uint8_t valtype;
    uint8_t mut;
    int64_t init_value;
} WasmGlobal;

typedef struct {
    uint32_t offset;        // linmem byte offset (resolved const)
    const uint8_t *data;    // owned by input wasm bytes (still valid)
    uint32_t size;
} WasmDataSeg;

typedef struct {
    WasmType  *types;
    uint32_t   n_types;

    char     **import_func_names;  // per-imported-func, arena-owned full name
    uint32_t  *import_func_types;
    uint32_t   n_imported_funcs;

    WasmFunc  *funcs;       // size = n_imported_funcs + n_def_funcs
    uint32_t   n_funcs;     // total (imports + defined)
    uint32_t   n_def_funcs; // just defined

    WasmGlobal *globals;
    uint32_t    n_globals;

    bool       has_memory;
    uint32_t   memory_min_pages;

    uint32_t   start_export_idx;  // UINT32_MAX if no _start export

    WasmDataSeg *data_segs;
    uint32_t     n_data_segs;

    // Element-segment contents: elem_funcs[slot] = func_index that
    // wasm-ld placed at that table slot, or UINT32_MAX if the slot is
    // unused. Indexed by wasm table slot (0..n_elem_funcs).
    uint32_t  *elem_funcs;
    uint32_t   n_elem_funcs;     // highest slot+1 written
    uint32_t   elem_funcs_cap;   // current capacity
} WasmModule;

// =============================================================================
// Arena helpers.
// =============================================================================
static char *arena_strdup(Arena *a, const char *s) {
    size_t n = strlen(s);
    char *o = (char *)arena_alloc(a, n + 1);
    memcpy(o, s, n + 1);
    return o;
}
static char *arena_strndup(Arena *a, const char *s, size_t n) {
    char *o = (char *)arena_alloc(a, n + 1);
    memcpy(o, s, n);
    o[n] = '\0';
    return o;
}
static char *arena_printf(Arena *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[128];
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    char *o = (char *)arena_alloc(a, (size_t)n + 1);
    memcpy(o, tmp, (size_t)n);
    o[n] = '\0';
    return o;
}

// =============================================================================
// Parse wasm module into our internal structure.
// =============================================================================
static bool wasm_parse(Arena *a, const uint8_t *bytes, size_t size,
                       WasmModule *out) {
    memset(out, 0, sizeof(*out));
    out->start_export_idx = UINT32_MAX;

    if (size < 8 || memcmp(bytes, "\x00" "asm" "\x01\x00\x00\x00", 8) != 0) {
        fprintf(stderr, "wasm->wasmstack: not a wasm module\n");
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
            uint32_t n = (uint32_t)rd_uleb(&s);
            out->types = (WasmType *)arena_alloc(a, n * sizeof(WasmType));
            out->n_types = n;
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint8_t tag = rd_u8(&s);
                if (tag != 0x60) {
                    fprintf(stderr, "wasm->wasmstack: type %u bad tag 0x%02x\n", i, tag);
                    return false;
                }
                uint32_t np = (uint32_t)rd_uleb(&s);
                uint8_t *params = (uint8_t *)arena_alloc(a, np ? np : 1);
                for (uint32_t k = 0; k < np; k++) params[k] = rd_u8(&s);
                uint32_t nr = (uint32_t)rd_uleb(&s);
                uint8_t *results = (uint8_t *)arena_alloc(a, nr ? nr : 1);
                for (uint32_t k = 0; k < nr; k++) results[k] = rd_u8(&s);
                out->types[i].n_params = (uint8_t)np;
                out->types[i].n_results = (uint8_t)nr;
                out->types[i].params = params;
                out->types[i].results = results;
            }
            break;
        }
        case 2: { // IMPORT
            uint32_t n = (uint32_t)rd_uleb(&s);
            out->import_func_names = (char **)arena_alloc(a, n * sizeof(char *));
            out->import_func_types = (uint32_t *)arena_alloc(a, n * sizeof(uint32_t));
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint32_t ml = (uint32_t)rd_uleb(&s);
                const uint8_t *mp = s.p;
                if (s.p + ml > s.end) { s.overflow = true; break; }
                s.p += ml;
                uint32_t nl = (uint32_t)rd_uleb(&s);
                const uint8_t *np = s.p;
                if (s.p + nl > s.end) { s.overflow = true; break; }
                s.p += nl;
                uint8_t kind = rd_u8(&s);
                if (kind == 0) {
                    uint32_t tidx = (uint32_t)rd_uleb(&s);
                    uint32_t fi = out->n_imported_funcs;
                    out->import_func_types[fi] = tidx;
                    // Use the unprefixed import name when it's a known
                    // WASI import — the wmir backend matches by name.
                    bool is_wasi = (ml == 22 &&
                        memcmp(mp, "wasi_snapshot_preview1", 22) == 0);
                    if (is_wasi) {
                        out->import_func_names[fi] = arena_strndup(a,
                            (const char *)np, nl);
                    } else {
                        out->import_func_names[fi] = arena_printf(a,
                            "%.*s_%.*s", (int)ml, mp, (int)nl, np);
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
            out->n_def_funcs = n;
            // Allocate slots for imports first, then defined.
            out->n_funcs = out->n_imported_funcs + n;
            out->funcs = (WasmFunc *)arena_alloc(a, out->n_funcs * sizeof(WasmFunc));
            memset(out->funcs, 0, out->n_funcs * sizeof(WasmFunc));
            for (uint32_t i = 0; i < out->n_imported_funcs; i++) {
                out->funcs[i].type_idx = out->import_func_types[i];
                out->funcs[i].name = out->import_func_names[i];
            }
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                out->funcs[out->n_imported_funcs + i].type_idx =
                    (uint32_t)rd_uleb(&s);
            }
            break;
        }
        case 9: { // ELEM
            // Element segments populate the wasm function table at
            // load time. wasm-ld emits one ACTIVE segment per
            // address-taken function, e.g.:
            //   (elem (i32.const N) func $foo)
            // We collect (table_slot, func_index) pairs so the wmir
            // lowering can pre-populate the call_indirect dispatcher
            // with real table slots (rather than auto-interned ones).
            uint32_t n = (uint32_t)rd_uleb(&s);
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint32_t flags = (uint32_t)rd_uleb(&s);
                // We handle only flags=0: active segment for table 0
                // with an i32.const offset expression and a vec of
                // func indices (elemkind=funcref omitted because
                // active-segment encoding uses opcode form).
                if (flags != 0) {
                    fprintf(stderr,
                        "wasm->wasmstack: elem segment %u flags=%u "
                        "not supported (only active/table0)\n",
                        i, flags);
                    return false;
                }
                uint32_t off = 0;
                uint8_t op = rd_u8(&s);
                if (op == 0x41) {
                    off = (uint32_t)rd_sleb(&s);
                } else {
                    fprintf(stderr,
                        "wasm->wasmstack: elem seg %u bad init 0x%02x\n",
                        i, op);
                    return false;
                }
                uint8_t e = rd_u8(&s);
                if (e != 0x0b) {
                    fprintf(stderr,
                        "wasm->wasmstack: elem seg %u missing end\n", i);
                    return false;
                }
                uint32_t cnt = (uint32_t)rd_uleb(&s);
                // Grow elem_funcs to fit slot range [off, off+cnt).
                uint32_t need = off + cnt;
                if (need > out->elem_funcs_cap) {
                    uint32_t nc = out->elem_funcs_cap ? out->elem_funcs_cap : 8;
                    while (nc < need) nc *= 2;
                    uint32_t *nf = (uint32_t *)arena_alloc(a,
                        nc * sizeof(uint32_t));
                    for (uint32_t k = 0; k < out->elem_funcs_cap; k++)
                        nf[k] = out->elem_funcs[k];
                    for (uint32_t k = out->elem_funcs_cap; k < nc; k++)
                        nf[k] = UINT32_MAX;
                    out->elem_funcs = nf;
                    out->elem_funcs_cap = nc;
                }
                if (need > out->n_elem_funcs) out->n_elem_funcs = need;
                for (uint32_t k = 0; k < cnt && !s.overflow; k++) {
                    uint32_t fi = (uint32_t)rd_uleb(&s);
                    out->elem_funcs[off + k] = fi;
                }
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
            for (uint32_t i = 1; i < n && !s.overflow; i++) {
                uint8_t fl = rd_u8(&s);
                (void)rd_uleb(&s); if (fl & 1) (void)rd_uleb(&s);
            }
            break;
        }
        case 6: { // GLOBAL
            uint32_t n = (uint32_t)rd_uleb(&s);
            out->globals = (WasmGlobal *)arena_alloc(a, n ? n * sizeof(WasmGlobal) : 1);
            out->n_globals = n;
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint8_t vt = rd_u8(&s);
                uint8_t mut = rd_u8(&s);
                int64_t v = 0;
                uint8_t op = rd_u8(&s);
                if (op == 0x41 || op == 0x42) v = rd_sleb(&s);
                else {
                    fprintf(stderr, "wasm->wasmstack: global %u bad init 0x%02x\n", i, op);
                    return false;
                }
                uint8_t e = rd_u8(&s);
                if (e != 0x0b) {
                    fprintf(stderr, "wasm->wasmstack: global %u init missing end\n", i);
                    return false;
                }
                out->globals[i].valtype = vt;
                out->globals[i].mut = mut;
                out->globals[i].init_value = v;
            }
            break;
        }
        case 7: { // EXPORT
            uint32_t n = (uint32_t)rd_uleb(&s);
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint32_t nl = (uint32_t)rd_uleb(&s);
                const uint8_t *np = s.p;
                if (s.p + nl > s.end) { s.overflow = true; break; }
                s.p += nl;
                uint8_t kind = rd_u8(&s);
                uint32_t idx = (uint32_t)rd_uleb(&s);
                if (kind == 0) {
                    if (nl == 6 && memcmp(np, "_start", 6) == 0) {
                        out->start_export_idx = idx;
                    }
                    if (idx < out->n_funcs) {
                        // Remember the export name in case the name
                        // section is absent. Don't overwrite a name we
                        // already assigned from a prior export.
                        if (!out->funcs[idx].name) {
                            out->funcs[idx].name = arena_strndup(a,
                                (const char *)np, nl);
                        }
                        out->funcs[idx].exported = true;
                    }
                }
            }
            break;
        }
        case 11: { // DATA
            uint32_t n = (uint32_t)rd_uleb(&s);
            out->data_segs = (WasmDataSeg *)arena_alloc(a,
                (n ? n : 1) * sizeof(WasmDataSeg));
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint32_t flags = (uint32_t)rd_uleb(&s);
                uint32_t off = 0;
                if (flags == 0) {
                    uint8_t op = rd_u8(&s);
                    if (op == 0x41) {
                        off = (uint32_t)rd_sleb(&s);
                    } else if (op == 0x23) {
                        (void)rd_uleb(&s);  // global.get gidx
                    } else {
                        fprintf(stderr,
                            "wasm->wasmstack: data seg %u bad init 0x%02x\n",
                            i, op);
                        return false;
                    }
                    uint8_t e = rd_u8(&s);
                    if (e != 0x0b) {
                        fprintf(stderr,
                            "wasm->wasmstack: data seg %u missing end\n", i);
                        return false;
                    }
                } else if (flags == 1) {
                    // Passive: offset unused. Treat as 0.
                } else if (flags == 2) {
                    (void)rd_uleb(&s); // memidx
                    uint8_t op = rd_u8(&s);
                    if (op == 0x41) off = (uint32_t)rd_sleb(&s);
                    rd_u8(&s); // end
                } else {
                    fprintf(stderr, "wasm->wasmstack: data seg %u bad flags %u\n",
                        i, flags);
                    return false;
                }
                uint32_t dl = (uint32_t)rd_uleb(&s);
                if (s.p + dl > s.end) { s.overflow = true; break; }
                out->data_segs[out->n_data_segs].offset = off;
                out->data_segs[out->n_data_segs].data = s.p;
                out->data_segs[out->n_data_segs].size = dl;
                out->n_data_segs++;
                s.p += dl;
            }
            break;
        }
        case 0: { // CUSTOM
            // Only the "name" custom section is consumed. Everything else
            // (producers, target_features, ...) is silently skipped.
            uint32_t nl = (uint32_t)rd_uleb(&s);
            if (s.p + nl > s.end) { s.overflow = true; break; }
            bool is_name = (nl == 4 && memcmp(s.p, "name", 4) == 0);
            s.p += nl;
            if (!is_name) break;
            while (s.p < s.end && !s.overflow) {
                uint8_t sub_id = rd_u8(&s);
                uint32_t sub_len = (uint32_t)rd_uleb(&s);
                const uint8_t *sub_end = s.p + sub_len;
                if (sub_end > s.end) { s.overflow = true; break; }
                if (sub_id == 1) {
                    // Function names.
                    uint32_t cnt = (uint32_t)rd_uleb(&s);
                    for (uint32_t i = 0; i < cnt && !s.overflow; i++) {
                        uint32_t fi = (uint32_t)rd_uleb(&s);
                        uint32_t fnl = (uint32_t)rd_uleb(&s);
                        if (s.p + fnl > sub_end) { s.overflow = true; break; }
                        if (fi < out->n_funcs) {
                            // Name section takes priority over export name
                            // for non-exported functions. For exported ones
                            // the export name is already set.
                            if (!out->funcs[fi].name) {
                                out->funcs[fi].name = arena_strndup(a,
                                    (const char *)s.p, fnl);
                            }
                        }
                        s.p += fnl;
                    }
                }
                s.p = sub_end;
            }
            break;
        }
        case 10: { // CODE
            uint32_t n = (uint32_t)rd_uleb(&s);
            for (uint32_t i = 0; i < n && !s.overflow; i++) {
                uint32_t bsz = (uint32_t)rd_uleb(&s);
                const uint8_t *bp = s.p;
                if (s.p + bsz > s.end) { s.overflow = true; break; }
                uint32_t fi = out->n_imported_funcs + i;
                if (fi < out->n_funcs) {
                    out->funcs[fi].code = bp;
                    out->funcs[fi].code_size = bsz;
                }
                s.p += bsz;
            }
            break;
        }
        default: break;
        }

        r.p = sec_end;
    }
    if (r.overflow) {
        fprintf(stderr, "wasm->wasmstack: parse overflow\n");
        return false;
    }

    // Name defined functions. Priority: explicit name from EXPORT/name
    // section > placeholder `func_N`. The wasm `_start` export is
    // renamed to `wasi_start` so the wmir backend's synth_start (which
    // generates a fresh `_start` from `__original_main`) doesn't
    // collide.
    for (uint32_t i = out->n_imported_funcs; i < out->n_funcs; i++) {
        if (out->funcs[i].name && strcmp(out->funcs[i].name, "_start") == 0) {
            out->funcs[i].name = arena_strdup(a, "wasi_start");
        } else if (!out->funcs[i].name) {
            if (out->funcs[i].exported && i == out->start_export_idx) {
                out->funcs[i].name = arena_strdup(a, "wasi_start");
            } else {
                out->funcs[i].name = arena_printf(a, "func_%u", i);
            }
        }
    }

    // Disambiguate duplicate function names. Two static C functions
    // named "foo" in different translation units linked together by
    // wasm-ld both end up in the wasm name section as "foo", but they
    // are distinct functions at the wasm-binary level (distinct func
    // indices). The wasmstack→wasmssa lifter resolves calls by name,
    // so name collisions cause the wrong signature to be picked and
    // either stack underflow or wrong arity. Append ".<index>" to
    // every duplicate so each function has a unique symbol.
    for (uint32_t i = 0; i < out->n_funcs; i++) {
        if (!out->funcs[i].name) continue;
        for (uint32_t j = 0; j < i; j++) {
            if (!out->funcs[j].name) continue;
            if (strcmp(out->funcs[i].name, out->funcs[j].name) == 0) {
                out->funcs[i].name = arena_printf(a, "%s.%u",
                    out->funcs[i].name, i);
                break;
            }
        }
    }

    return true;
}

// =============================================================================
// MLIR builder helpers.
// =============================================================================
static MLIR_AttributeHandle attr_i32(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 32, true));
}
static MLIR_AttributeHandle attr_i64(MLIR_Context *ctx, const char *name, int64_t v) {
    return MLIR_CreateAttributeInteger(ctx, str_from_cstr_view((char *)name), v,
                                       MLIR_CreateTypeInteger(ctx, 64, true));
}
static MLIR_AttributeHandle attr_b(MLIR_Context *ctx, const char *name, bool v) {
    return MLIR_CreateAttributeBool(ctx, str_from_cstr_view((char *)name), v);
}
static MLIR_AttributeHandle attr_s(MLIR_Context *ctx, const char *name,
                                   const char *v, size_t vlen) {
    string sv = { (char *)v, vlen };
    return MLIR_CreateAttributeString(ctx, str_from_cstr_view((char *)name), sv);
}
static char *hex_encode_arena(Arena *arena, const uint8_t *p, size_t n) {
    char *out = (char *)arena_alloc(arena, n ? n * 2 : 1);
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i*2 + 0] = d[(p[i] >> 4) & 0xf];
        out[i*2 + 1] = d[p[i] & 0xf];
    }
    return out;
}
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n) {
    char *h = hex_encode_arena(arena, p, n);
    return attr_s(ctx, name, h, n * 2);
}

static MLIR_OpHandle build_op(MLIR_Context *ctx, MLIR_OpType t,
                              MLIR_AttributeHandle *attrs, size_t na) {
    return MLIR_CreateOp(ctx, t, op_type_to_string(t),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Per-function: decode the body opcodes into wasmstack ops.
//
// First-light scope: opcodes appearing in macho_exit's linked wasm
// (local_get/set/tee, i32/i64.const, return, call, end, plus the
// `end` terminator). Other opcodes return false with a diagnostic so
// we can grow this op-by-op.
// =============================================================================
typedef struct {
    uint32_t  local_idx;  // function-local index
    uint8_t   valtype;
} LocalEntry;

static MLIR_OpHandle emit_op_simple(MLIR_Context *ctx, MLIR_OpType t,
                                    uint8_t valtype) {
    MLIR_AttributeHandle as[1];
    as[0] = attr_i32(ctx, "valtype", valtype);
    return build_op(ctx, t, as, 1);
}

static bool decode_body(MLIR_Context *ctx, Arena *arena,
                        const WasmModule *m,
                        const WasmFunc *fn,
                        MLIR_BlockHandle dst_blk,
                        uint8_t **local_types_out, uint32_t *n_locals_out) {
    Rd r = { fn->code, fn->code + fn->code_size, false };

    // Locals prelude: count of (run_count, valtype) pairs.
    uint32_t n_local_runs = (uint32_t)rd_uleb(&r);
    // First, the function's parameters are also locals[0..n_params).
    const WasmType *fty = &m->types[fn->type_idx];
    uint32_t n_total = fty->n_params;
    // Pre-walk runs to compute total.
    Rd pre = r;
    for (uint32_t i = 0; i < n_local_runs && !pre.overflow; i++) {
        uint32_t c = (uint32_t)rd_uleb(&pre);
        (void)rd_u8(&pre);
        n_total += c;
    }
    if (pre.overflow) return false;

    uint8_t *local_types = (uint8_t *)arena_alloc(arena, n_total ? n_total : 1);
    uint32_t li = 0;
    for (uint32_t i = 0; i < fty->n_params; i++) local_types[li++] = fty->params[i];
    for (uint32_t i = 0; i < n_local_runs && !r.overflow; i++) {
        uint32_t c = (uint32_t)rd_uleb(&r);
        uint8_t vt = rd_u8(&r);
        for (uint32_t k = 0; k < c; k++) local_types[li++] = vt;
    }
    *local_types_out = local_types;
    *n_locals_out = n_total;

    // Helper: emit local.get / local.set / local.tee.
    #define EMIT_LOCAL_OP(T, IDX) do { \
        MLIR_AttributeHandle as[2]; \
        as[0] = attr_i32(ctx, "valtype", 0); \
        as[1] = attr_i32(ctx, "local_idx", (int64_t)(IDX)); \
        MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, (T), as, 2)); \
    } while (0)

    // Walk opcodes until the trailing 0x0b (function end).
    while (r.p < r.end && !r.overflow) {
        uint8_t op = rd_u8(&r);
        switch (op) {
        case 0x00: { // unreachable
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_UNREACHABLE, as, 1));
            break;
        }
        case 0x0b: { // end (terminator)
            if (r.p != r.end) {
                // Inner end (for block/loop/if). Emit ENDand keep going.
                MLIR_AttributeHandle as[1];
                as[0] = attr_i32(ctx, "valtype", 0);
                MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_END, as, 1));
            } else {
                // Function-terminator end — implied return. We don't
                // emit a `wasmstack.return` here; the wasmstack.func
                // body simply ends. The wasmstack -> wasmssa lifter
                // will materialize an SSA return using the function's
                // type result.
                MLIR_AttributeHandle as[1];
                as[0] = attr_i32(ctx, "valtype", 0);
                MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_END, as, 1));
            }
            break;
        }
        case 0x0f: { // return
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk, build_op(ctx, OP_TYPE_WASMSTACK_RETURN, as, 1));
            break;
        }
        case 0x20: { // local.get N
            uint32_t idx = (uint32_t)rd_uleb(&r);
            EMIT_LOCAL_OP(OP_TYPE_WASMSTACK_LOCAL_GET, idx);
            break;
        }
        case 0x21: { // local.set N
            uint32_t idx = (uint32_t)rd_uleb(&r);
            EMIT_LOCAL_OP(OP_TYPE_WASMSTACK_LOCAL_SET, idx);
            break;
        }
        case 0x22: { // local.tee N
            uint32_t idx = (uint32_t)rd_uleb(&r);
            EMIT_LOCAL_OP(OP_TYPE_WASMSTACK_LOCAL_TEE, idx);
            break;
        }
        case 0x41: { // i32.const
            int64_t v = rd_sleb(&r);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I32);
            as[1] = attr_i64(ctx, "value", v);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_CONST, as, 2));
            break;
        }
        case 0x42: { // i64.const
            int64_t v = rd_sleb(&r);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            as[1] = attr_i64(ctx, "value", v);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_CONST, as, 2));
            break;
        }
        case 0x43: { // f32.const
            uint32_t bits = rd_f32_bits(&r);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F32);
            as[1] = attr_i64(ctx, "value", (int64_t)(int32_t)bits);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_CONST, as, 2));
            break;
        }
        case 0x44: { // f64.const
            uint64_t bits = rd_f64_bits(&r);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F64);
            as[1] = attr_i64(ctx, "value", (int64_t)bits);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_CONST, as, 2));
            break;
        }
        case 0x10: { // call funcidx
            uint32_t fi = (uint32_t)rd_uleb(&r);
            if (fi >= m->n_funcs) {
                fprintf(stderr, "wasm->wasmstack: call: bad funcidx %u\n", fi);
                return false;
            }
            const char *nm = m->funcs[fi].name;
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_s(ctx, "target", nm, strlen(nm));
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_CALL, as, 2));
            break;
        }
        case 0x11: { // call_indirect typeidx 0x00
            uint32_t tidx = (uint32_t)rd_uleb(&r);
            (void)rd_u8(&r); // table index byte (always 0)
            if (tidx >= m->n_types) {
                fprintf(stderr, "wasm->wasmstack: call_indirect: bad typeidx %u\n", tidx);
                return false;
            }
            const WasmType *ft = &m->types[tidx];
            char *sp = hex_encode_arena(arena, ft->params, ft->n_params);
            char *sr = hex_encode_arena(arena, ft->results, ft->n_results);
            MLIR_AttributeHandle as[3];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_s(ctx, "sig_params", sp, ft->n_params * 2);
            as[2] = attr_s(ctx, "sig_results", sr, ft->n_results * 2);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_CALL_INDIRECT, as, 3));
            break;
        }
        case 0x1b: { // select
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_SELECT, as, 1));
            break;
        }
        case 0x23: { // global.get
            uint32_t gi = (uint32_t)rd_uleb(&r);
            uint8_t vt = (gi < m->n_globals) ? m->globals[gi].valtype : WT_I32;
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", vt);
            as[1] = attr_i32(ctx, "global_idx", (int64_t)gi);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_GLOBAL_GET, as, 2));
            break;
        }
        case 0x24: { // global.set
            uint32_t gi = (uint32_t)rd_uleb(&r);
            uint8_t vt = (gi < m->n_globals) ? m->globals[gi].valtype : WT_I32;
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", vt);
            as[1] = attr_i32(ctx, "global_idx", (int64_t)gi);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_GLOBAL_SET, as, 2));
            break;
        }
        case 0x28: case 0x29: case 0x2a: case 0x2b:
        case 0x2c: case 0x2d: case 0x2e: case 0x2f:
        case 0x30: case 0x31: case 0x32: case 0x33:
        case 0x34: case 0x35: { // loads
            uint32_t al = (uint32_t)rd_uleb(&r);
            uint32_t off = (uint32_t)rd_uleb(&r);
            uint8_t vt; uint32_t sz; bool sign_extend = false;
            switch (op) {
            case 0x28: vt = WT_I32; sz = 4; break;
            case 0x29: vt = WT_I64; sz = 8; break;
            case 0x2a: vt = WT_F32; sz = 4; break;
            case 0x2b: vt = WT_F64; sz = 8; break;
            case 0x2c: vt = WT_I32; sz = 1; sign_extend = true; break;
            case 0x2d: vt = WT_I32; sz = 1; break;
            case 0x2e: vt = WT_I32; sz = 2; sign_extend = true; break;
            case 0x2f: vt = WT_I32; sz = 2; break;
            case 0x30: vt = WT_I64; sz = 1; sign_extend = true; break;
            case 0x31: vt = WT_I64; sz = 1; break;
            case 0x32: vt = WT_I64; sz = 2; sign_extend = true; break;
            case 0x33: vt = WT_I64; sz = 2; break;
            case 0x34: vt = WT_I64; sz = 4; sign_extend = true; break;
            case 0x35: vt = WT_I64; sz = 4; break;
            default:   vt = WT_I32; sz = 4; break;
            }
            MLIR_AttributeHandle as[4];
            as[0] = attr_i32(ctx, "valtype", vt);
            as[1] = attr_i32(ctx, "memory_offset", (int64_t)off);
            as[2] = attr_i32(ctx, "memory_align_log2", (int64_t)al);
            as[3] = attr_i32(ctx, "mem_size_bytes", (int64_t)sz);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_LOAD, as, 4));
            // For signed sub-word loads, emit an explicit sign-extension
            // op after the zero-extending load. This matches the forward
            // pipeline which represents `i32.load8_s` as `load8_u +
            // i32.extend8_s` (the wasm UNOP with wasm_opcode 0xc0/0xc1
            // for i32; 0xc2/0xc3/0xc4 for i64).
            if (sign_extend) {
                uint8_t uop_byte = 0;
                if (vt == WT_I32) uop_byte = (sz == 1) ? 0xc0 : 0xc1;
                else              uop_byte = (sz == 1) ? 0xc2 : (sz == 2 ? 0xc3 : 0xc4);
                MLIR_AttributeHandle bs[2];
                bs[0] = attr_i32(ctx, "valtype", vt);
                bs[1] = attr_i32(ctx, "wasm_opcode", (int64_t)uop_byte);
                MLIR_AppendBlockOp(ctx, dst_blk,
                    build_op(ctx, OP_TYPE_WASMSTACK_UNOP, bs, 2));
            }
            break;
        }
        case 0x36: case 0x37: case 0x38: case 0x39:
        case 0x3a: case 0x3b:
        case 0x3c: case 0x3d: case 0x3e: { // stores
            uint32_t al = (uint32_t)rd_uleb(&r);
            uint32_t off = (uint32_t)rd_uleb(&r);
            uint8_t vt; uint32_t sz;
            switch (op) {
            case 0x36: vt = WT_I32; sz = 4; break;
            case 0x37: vt = WT_I64; sz = 8; break;
            case 0x38: vt = WT_F32; sz = 4; break;
            case 0x39: vt = WT_F64; sz = 8; break;
            case 0x3a: vt = WT_I32; sz = 1; break;
            case 0x3b: vt = WT_I32; sz = 2; break;
            case 0x3c: vt = WT_I64; sz = 1; break;
            case 0x3d: vt = WT_I64; sz = 2; break;
            case 0x3e: vt = WT_I64; sz = 4; break;
            default:   vt = WT_I32; sz = 4; break;
            }
            MLIR_AttributeHandle as[4];
            as[0] = attr_i32(ctx, "valtype", vt);
            as[1] = attr_i32(ctx, "memory_offset", (int64_t)off);
            as[2] = attr_i32(ctx, "memory_align_log2", (int64_t)al);
            as[3] = attr_i32(ctx, "mem_size_bytes", (int64_t)sz);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_STORE, as, 4));
            break;
        }
        case 0x3f: { // memory.size 0x00
            (void)rd_u8(&r);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_MEMORY_SIZE, as, 1));
            break;
        }
        case 0x40: { // memory.grow 0x00
            (void)rd_u8(&r);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_MEMORY_GROW, as, 1));
            break;
        }
        case 0x45: { // i32.eqz
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I32);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_EQZ, as, 1));
            break;
        }
        case 0x50: { // i64.eqz
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_EQZ, as, 1));
            break;
        }
        case 0x6a: { // i32.add — special-case
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I32);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_ADD, as, 1));
            break;
        }
        case 0x6b: { // i32.sub
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I32);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_SUB, as, 1));
            break;
        }
        case 0x7c: { // i64.add
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_ADD, as, 1));
            break;
        }
        case 0x7d: { // i64.sub
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_SUB, as, 1));
            break;
        }
        case 0xac: { // i64.extend_i32_s
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_EXTEND_I32_S, as, 1));
            break;
        }
        case 0x02: { // block btype
            uint8_t bt = rd_u8(&r);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", bt == 0x40 ? 0 : bt);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BLOCK, as, 1));
            break;
        }
        case 0x03: { // loop btype
            uint8_t bt = rd_u8(&r);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", bt == 0x40 ? 0 : bt);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_LOOP, as, 1));
            break;
        }
        case 0x04: { // if btype
            uint8_t bt = rd_u8(&r);
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", bt == 0x40 ? 0 : bt);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_IF, as, 1));
            break;
        }
        case 0x05: { // else
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_ELSE, as, 1));
            break;
        }
        case 0x0c: { // br depth
            uint32_t d = (uint32_t)rd_uleb(&r);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_i32(ctx, "depth", (int64_t)d);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BR, as, 2));
            break;
        }
        case 0x0d: { // br_if depth
            uint32_t d = (uint32_t)rd_uleb(&r);
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_i32(ctx, "depth", (int64_t)d);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BR_IF, as, 2));
            break;
        }
        // Remaining wasm binops/unops: emit as WASMSTACK_BINOP / UNOP
        // with the raw `wasm_opcode` byte attribute. The wasmstack ->
        // wasmssa lifter knows the standard (vt, vt) -> vt shape for
        // these, so this passes through cleanly.
        //
        // i32 family (vt=i32, takes/returns i32):
        case 0x46: case 0x47: case 0x48: case 0x49: case 0x4a: case 0x4b:
        case 0x4c: case 0x4d: case 0x4e: case 0x4f:
        case 0x6c: case 0x6d: case 0x6e: case 0x6f: case 0x70:
        case 0x71: case 0x72: case 0x73:
        case 0x74: case 0x75: case 0x76:
        case 0x77: case 0x78: { // i32 binops (mul/div/rem/and/or/xor/shl/shr/rotl/rotr) + i32 cmps
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I32);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        case 0x67: case 0x68: case 0x69: { // i32 unops clz/ctz/popcnt
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I32);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_UNOP, as, 2));
            break;
        }
        case 0x79: case 0x7a: case 0x7b: { // i64 clz/ctz/popcnt
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_UNOP, as, 2));
            break;
        }
        case 0x51: case 0x52: case 0x53: case 0x54: case 0x55: case 0x56:
        case 0x57: case 0x58: case 0x59: case 0x5a: { // i64 cmps
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        case 0x7e: case 0x7f: case 0x80: case 0x81: case 0x82:
        case 0x83: case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x89: case 0x8a: { // i64 binops (mul/div/rem/and/or/xor/shl/shr/rotl/rotr)
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_I64);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        // f32 cmps (0x5b..0x60):
        case 0x5b: case 0x5c: case 0x5d: case 0x5e: case 0x5f: case 0x60: {
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F32);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        // f64 cmps (0x61..0x66):
        case 0x61: case 0x62: case 0x63: case 0x64: case 0x65: case 0x66: {
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F64);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        // f32 unops (abs/neg/ceil/floor/trunc/nearest/sqrt: 0x8b..0x91):
        case 0x8b: case 0x8c: case 0x8d: case 0x8e: case 0x8f:
        case 0x90: case 0x91: {
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F32);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_UNOP, as, 2));
            break;
        }
        // f32 binops (add/sub/mul/div/min/max/copysign: 0x92..0x98):
        case 0x92: case 0x93: case 0x94: case 0x95: case 0x96:
        case 0x97: case 0x98: {
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F32);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        // f64 unops (0x99..0x9f):
        case 0x99: case 0x9a: case 0x9b: case 0x9c: case 0x9d:
        case 0x9e: case 0x9f: {
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F64);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_UNOP, as, 2));
            break;
        }
        // f64 binops (0xa0..0xa6):
        case 0xa0: case 0xa1: case 0xa2: case 0xa3: case 0xa4:
        case 0xa5: case 0xa6: {
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", WT_F64);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BINOP, as, 2));
            break;
        }
        // Conversions (0xa7..0xc4):
        // wrap/extend/trunc/convert/reinterpret. We treat all of these
        // as UNOPs with a wasm_opcode attribute carrying the byte.
        // Result valtype derived per opcode.
        case 0xa7: case 0xa8: case 0xa9: case 0xaa: case 0xab:
        /* 0xac handled above as EXTEND_I32_S */
        case 0xad: case 0xae: case 0xaf: case 0xb0: case 0xb1:
        case 0xb2: case 0xb3: case 0xb4: case 0xb5: case 0xb6:
        case 0xb7: case 0xb8: case 0xb9: case 0xba: case 0xbb:
        case 0xbc: case 0xbd: case 0xbe: case 0xbf:
        case 0xc0: case 0xc1: case 0xc2: case 0xc3: case 0xc4: {
            // Result valtype per opcode. Source valtype isn't stored on
            // the op since the wasm stack already carries the typed
            // value the unop reads from.
            uint8_t res_vt;
            switch (op) {
            case 0xa7: res_vt = WT_I32; break;  // i32.wrap_i64
            case 0xa8: case 0xa9: res_vt = WT_I32; break;  // i32.trunc_f32_s/u
            case 0xaa: case 0xab: res_vt = WT_I32; break;  // i32.trunc_f64_s/u
            case 0xad: case 0xae: res_vt = WT_I64; break;  // i64.extend_i32_u (0xad), trunc_f32_s (0xae)
            case 0xaf: case 0xb0: case 0xb1: res_vt = WT_I64; break;  // various
            case 0xb2: case 0xb3: case 0xb4: case 0xb5: res_vt = WT_F32; break;
            case 0xb6: res_vt = WT_F32; break;  // f32.demote_f64
            case 0xb7: case 0xb8: case 0xb9: case 0xba: res_vt = WT_F64; break;
            case 0xbb: res_vt = WT_F64; break;  // f64.promote_f32
            case 0xbc: res_vt = WT_I32; break;  // i32.reinterpret_f32
            case 0xbd: res_vt = WT_I64; break;  // i64.reinterpret_f64
            case 0xbe: res_vt = WT_F32; break;  // f32.reinterpret_i32
            case 0xbf: res_vt = WT_F64; break;  // f64.reinterpret_i64
            case 0xc0: case 0xc1: res_vt = WT_I32; break;  // i32.extend8_s/16_s
            case 0xc2: case 0xc3: case 0xc4: res_vt = WT_I64; break;
            default:   res_vt = WT_I32;
            }
            MLIR_AttributeHandle as[2];
            as[0] = attr_i32(ctx, "valtype", res_vt);
            as[1] = attr_i32(ctx, "wasm_opcode", (int64_t)op);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_UNOP, as, 2));
            break;
        }
        case 0x1a: { // drop — pop one value, no result.
            MLIR_AttributeHandle as[1];
            as[0] = attr_i32(ctx, "valtype", 0);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_DROP, as, 1));
            break;
        }
        case 0x0e: { // br_table — list of N depths, then default depth.
            uint32_t n = (uint32_t)rd_uleb(&r);
            // Collect depths into a comma-separated string.
            char tbuf[1024];
            size_t tlen = 0;
            for (uint32_t k = 0; k < n; k++) {
                uint32_t d = (uint32_t)rd_uleb(&r);
                int wr = snprintf(tbuf + tlen, sizeof(tbuf) - tlen,
                    k == 0 ? "%u" : ",%u", d);
                if (wr < 0 || (size_t)wr >= sizeof(tbuf) - tlen) {
                    fprintf(stderr, "wasm->wasmstack: br_table too many targets\n");
                    return false;
                }
                tlen += (size_t)wr;
            }
            uint32_t dflt = (uint32_t)rd_uleb(&r);
            char *targets = arena_strndup(arena, tbuf, tlen);
            MLIR_AttributeHandle as[3];
            as[0] = attr_i32(ctx, "valtype", 0);
            as[1] = attr_s(ctx, "targets", targets, tlen);
            as[2] = attr_i32(ctx, "default", (int64_t)dflt);
            MLIR_AppendBlockOp(ctx, dst_blk,
                build_op(ctx, OP_TYPE_WASMSTACK_BR_TABLE, as, 3));
            break;
        }
        default:
            fprintf(stderr,
                "wasm->wasmstack: unsupported opcode 0x%02x in function '%s' "
                "(offset %ld)\n",
                op, fn->name ? fn->name : "?",
                (long)(r.p - 1 - fn->code));
            return false;
        }
    }
    #undef EMIT_LOCAL_OP

    if (r.overflow) {
        fprintf(stderr, "wasm->wasmstack: body overflow in function '%s'\n",
                fn->name ? fn->name : "?");
        return false;
    }
    return true;
}

// =============================================================================
// Convert one function (import or defined) to a wasmstack op.
// =============================================================================
static MLIR_OpHandle convert_func(MLIR_Context *ctx, Arena *arena,
                                  const WasmModule *m, uint32_t fi) {
    const WasmFunc *fn = &m->funcs[fi];
    const WasmType *fty = &m->types[fn->type_idx];

    char *pt_hex = hex_encode_arena(arena, fty->params, fty->n_params);
    char *rt_hex = hex_encode_arena(arena, fty->results, fty->n_results);

    bool is_import = (fi < m->n_imported_funcs);

    MLIR_AttributeHandle attrs[10];
    size_t na = 0;
    attrs[na++] = attr_s(ctx, "sym_name", fn->name, strlen(fn->name));
    attrs[na++] = attr_s(ctx, "param_types", pt_hex, fty->n_params * 2);
    attrs[na++] = attr_s(ctx, "result_types", rt_hex, fty->n_results * 2);
    attrs[na++] = attr_b(ctx, "exported", fn->exported);
    attrs[na++] = attr_b(ctx, "internal", false);

    if (is_import) {
        return build_op(ctx, OP_TYPE_WASMSTACK_IMPORT_FUNC, attrs, na);
    }

    MLIR_BlockHandle dst_blk = MLIR_CreateBlock(ctx);
    uint8_t *local_types = NULL;
    uint32_t n_locals = 0;
    if (!decode_body(ctx, arena, m, fn, dst_blk, &local_types, &n_locals)) {
        return MLIR_INVALID_HANDLE;
    }

    attrs[na++] = attr_s_hex(ctx, arena, "local_types", local_types, n_locals);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, dst_blk);
    MLIR_RegionHandle regs[1] = { region };
    return MLIR_CreateOp(ctx, OP_TYPE_WASMSTACK_FUNC,
        op_type_to_string(OP_TYPE_WASMSTACK_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
}

// =============================================================================
// Top-level entry point.
// =============================================================================
MLIR_OpHandle mlir_wasm_to_wasmstack(MLIR_Context *ctx,
                                     const uint8_t *wasm_bytes,
                                     size_t wasm_size) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    WasmModule m;
    if (!wasm_parse(arena, wasm_bytes, wasm_size, &m)) return MLIR_INVALID_HANDLE;

    MLIR_BlockHandle out_body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle out_region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, out_region, out_body);
    MLIR_RegionHandle out_regs[1] = { out_region };

    // Propagate the wasm MEMORY section's `min_pages` so downstream
    // passes (wmir -> aarch64 -> Mach-O) can size the linear memory
    // image correctly. Without this the wmir backend hardcoded
    // a tiny 4 MiB default and any module with >4 MiB of static
    // data would silently overflow __heap_base past the end of
    // linmem, causing platform_heap_size() to wrap to a huge value
    // and the buddy allocator to scribble all over the binary.
    MLIR_AttributeHandle mod_attrs[1];
    size_t n_mod_attrs = 0;
    if (m.has_memory) {
        mod_attrs[n_mod_attrs++] = attr_i32(ctx, "memory_min_pages",
                                            (int64_t)m.memory_min_pages);
    }
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        mod_attrs, n_mod_attrs, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    // Emit module-level globals (the wasm GLOBAL section).
    for (uint32_t i = 0; i < m.n_globals; i++) {
        MLIR_AttributeHandle as[4];
        as[0] = attr_i32(ctx, "global_idx", (int64_t)i);
        as[1] = attr_i32(ctx, "valtype", (int64_t)m.globals[i].valtype);
        as[2] = attr_b(ctx, "mut", m.globals[i].mut != 0);
        as[3] = attr_i64(ctx, "init_value", m.globals[i].init_value);
        MLIR_AppendBlockOp(ctx, out_body,
            build_op(ctx, OP_TYPE_WASMSTACK_GLOBAL_DECL, as, 4));
    }

    // Emit module-level data segments (the wasm DATA section).
    for (uint32_t i = 0; i < m.n_data_segs; i++) {
        MLIR_AttributeHandle as[3];
        as[0] = attr_i32(ctx, "offset", (int64_t)m.data_segs[i].offset);
        as[1] = attr_s(ctx, "init_data",
            (const char *)m.data_segs[i].data, m.data_segs[i].size);
        as[2] = attr_i32(ctx, "size", (int64_t)m.data_segs[i].size);
        MLIR_AppendBlockOp(ctx, out_body,
            build_op(ctx, OP_TYPE_WASMSTACK_DATA_SEGMENT, as, 3));
    }

    // Emit module-level function-pointer table entries (the wasm ELEM
    // section). Each entry says "wasm table slot N now holds a
    // reference to function f". The wmir lowering uses these to build
    // call_indirect dispatchers keyed on the actual wasm table slot
    // (rather than an auto-interned slot — that mismatch would make
    // the slot value the program sees at runtime not match the slot
    // the dispatcher expects).
    for (uint32_t k = 0; k < m.n_elem_funcs; k++) {
        uint32_t fi = m.elem_funcs[k];
        if (fi == UINT32_MAX) continue;
        if (fi >= m.n_funcs || !m.funcs[fi].name) continue;
        MLIR_AttributeHandle as[2];
        as[0] = attr_i32(ctx, "slot", (int64_t)k);
        as[1] = attr_s(ctx, "target", m.funcs[fi].name,
            strlen(m.funcs[fi].name));
        MLIR_AppendBlockOp(ctx, out_body,
            build_op(ctx, OP_TYPE_WASMSTACK_FUNC_ADDR_DECL, as, 2));
    }

    for (uint32_t i = 0; i < m.n_funcs; i++) {
        MLIR_OpHandle op = convert_func(ctx, arena, &m, i);
        if (!op) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, op);
    }
    return out_module;
}

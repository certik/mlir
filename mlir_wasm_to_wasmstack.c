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
                if (kind == 0 && nl == 6 && memcmp(np, "_start", 6) == 0) {
                    out->start_export_idx = idx;
                    if (idx < out->n_funcs) out->funcs[idx].exported = true;
                }
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

    // Name defined functions.
    for (uint32_t i = out->n_imported_funcs; i < out->n_funcs; i++) {
        if (out->funcs[i].exported && i == out->start_export_idx) {
            // The wasm _start gets renamed to wasi_start so the wmir
            // backend's synth_start doesn't collide.
            out->funcs[i].name = arena_strdup(a, "wasi_start");
        } else {
            out->funcs[i].name = arena_printf(a, "func_%u", i);
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
    MLIR_OpHandle out_module = MLIR_CreateOp(ctx, OP_TYPE_MODULE,
        str_lit("module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, out_regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);

    for (uint32_t i = 0; i < m.n_funcs; i++) {
        MLIR_OpHandle op = convert_func(ctx, arena, &m, i);
        if (!op) return MLIR_INVALID_HANDLE;
        MLIR_AppendBlockOp(ctx, out_body, op);
    }
    return out_module;
}

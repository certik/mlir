// Constructors / destructors for the wasmssa and wasmstack C-struct
// working representations declared in mlir_wasm_pipeline_internal.h,
// plus the converters that turn each form into / out of MLIR ops, and
// the public adapter wrappers that implement the MLIR-handle-based
// stage API declared in mlir_wasm_pipeline.h.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_op_names.h"
#include "mlir_wasm_pipeline_internal.h"

// ---- generic growable-array helper -----------------------------------------
// Note: we embed length+capacity into each container struct; helpers below
// just amortized-double when needed.

#define ENSURE(arr, n, c, T) do { \
    if ((n) == (c)) { \
        (c) = (c) ? (c) * 2 : 8; \
        (arr) = (T *)realloc((arr), (c) * sizeof(T)); \
    } \
} while (0)

// ============================================================================
// wasmssa
// ============================================================================
wasmssa_module_t *wasmssa_module_new(void) {
    wasmssa_module_t *m = (wasmssa_module_t *)calloc(1, sizeof(*m));
    return m;
}

static void wasmssa_func_free(wasmssa_func_t *f) {
    free(f->name);
    free(f->param_types);
    free(f->result_types);
    free(f->carrier_vts);
    for (size_t i = 0; i < f->n_ops; i++) {
        free(f->ops[i].operands);
        free(f->ops[i].call_target);
        free(f->ops[i].sig_params);
        free(f->ops[i].sig_results);
    }
    free(f->ops);
}

void wasmssa_module_free(wasmssa_module_t *m) {
    if (!m) return;
    for (size_t i = 0; i < m->n_funcs; i++) wasmssa_func_free(&m->funcs[i]);
    free(m->funcs);
    for (size_t i = 0; i < m->n_globals; i++) {
        free(m->globals[i].name);
        free(m->globals[i].data);
        for (size_t j = 0; j < m->globals[i].n_relocs; j++)
            free(m->globals[i].relocs[j].target);
        free(m->globals[i].relocs);
    }
    free(m->globals);
    free(m);
}

wasmssa_func_t *wasmssa_module_add_func(wasmssa_module_t *m) {
    ENSURE(m->funcs, m->n_funcs, m->c_funcs, wasmssa_func_t);
    wasmssa_func_t *f = &m->funcs[m->n_funcs++];
    memset(f, 0, sizeof(*f));
    return f;
}

wasmssa_op_t *wasmssa_func_add_op(wasmssa_func_t *f) {
    ENSURE(f->ops, f->n_ops, f->c_ops, wasmssa_op_t);
    wasmssa_op_t *o = &f->ops[f->n_ops++];
    memset(o, 0, sizeof(*o));
    return o;
}

// ============================================================================
// wasmstack
// ============================================================================
wasmstack_module_t *wasmstack_module_new(void) {
    wasmstack_module_t *m = (wasmstack_module_t *)calloc(1, sizeof(*m));
    return m;
}

static void wasmstack_func_free(wasmstack_func_t *f) {
    free(f->name);
    free(f->param_types);
    free(f->result_types);
    free(f->local_types);
    for (size_t i = 0; i < f->n_ops; i++) {
        free(f->ops[i].call_target);
        free(f->ops[i].sig_params);
        free(f->ops[i].sig_results);
    }
    free(f->ops);
}

void wasmstack_module_free(wasmstack_module_t *m) {
    if (!m) return;
    for (size_t i = 0; i < m->n_funcs; i++) wasmstack_func_free(&m->funcs[i]);
    free(m->funcs);
    for (size_t i = 0; i < m->n_globals; i++) {
        free(m->globals[i].name);
        free(m->globals[i].data);
        for (size_t j = 0; j < m->globals[i].n_relocs; j++)
            free(m->globals[i].relocs[j].target);
        free(m->globals[i].relocs);
    }
    free(m->globals);
    free(m);
}

wasmstack_func_t *wasmstack_module_add_func(wasmstack_module_t *m) {
    ENSURE(m->funcs, m->n_funcs, m->c_funcs, wasmstack_func_t);
    wasmstack_func_t *f = &m->funcs[m->n_funcs++];
    memset(f, 0, sizeof(*f));
    return f;
}

wasmstack_op_t *wasmstack_func_add_op(wasmstack_func_t *f) {
    ENSURE(f->ops, f->n_ops, f->c_ops, wasmstack_op_t);
    wasmstack_op_t *o = &f->ops[f->n_ops++];
    memset(o, 0, sizeof(*o));
    return o;
}

uint32_t wasmstack_func_add_local(wasmstack_func_t *f, uint8_t vt) {
    ENSURE(f->local_types, f->n_locals, f->c_locals, uint8_t);
    f->local_types[f->n_locals] = vt;
    return (uint32_t)(f->n_params + f->n_locals++);
}

// ============================================================================
// globals
// ============================================================================
wasm_global_t *wasmssa_module_add_global(wasmssa_module_t *m) {
    ENSURE(m->globals, m->n_globals, m->c_globals, wasm_global_t);
    wasm_global_t *g = &m->globals[m->n_globals++];
    memset(g, 0, sizeof(*g));
    return g;
}
wasm_global_t *wasmstack_module_add_global(wasmstack_module_t *m) {
    ENSURE(m->globals, m->n_globals, m->c_globals, wasm_global_t);
    wasm_global_t *g = &m->globals[m->n_globals++];
    memset(g, 0, sizeof(*g));
    return g;
}
void wasm_global_add_reloc(wasm_global_t *g, uint32_t off,
                           const char *target, int32_t addend) {
    ENSURE(g->relocs, g->n_relocs, g->c_relocs, wasm_data_reloc_t);
    wasm_data_reloc_t *r = &g->relocs[g->n_relocs++];
    r->offset = off;
    size_t n = strlen(target);
    r->target = (char *)malloc(n + 1);
    memcpy(r->target, target, n + 1);
    r->addend = addend;
}

// =============================================================================
// Hex encoding for byte arrays (param/result/local types, signatures).
// =============================================================================
// Each WT_* byte becomes two lowercase hex chars; binary-safe through the
// MLIR string attribute layer (which stores explicit-length bytes).
static char *hex_encode_alloc_arena(Arena *arena, const uint8_t *p, size_t n) {
    char *out = (char *)arena_alloc(arena, n ? n * 2 : 1);
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[i*2 + 0] = d[(p[i] >> 4) & 0xf];
        out[i*2 + 1] = d[p[i] & 0xf];
    }
    return out;
}
static uint8_t *hex_decode_alloc(string s, size_t *out_n) {
    if (s.size & 1) { *out_n = 0; return (uint8_t *)malloc(1); }
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

// =============================================================================
// Helpers: build MLIR ops with a uniform interface.
// =============================================================================
static MLIR_TypeHandle vt_to_type(MLIR_Context *ctx, uint8_t vt) {
    switch (vt) {
        case WT_I32: return MLIR_CreateTypeInteger(ctx, 32, true);
        case WT_I64: return MLIR_CreateTypeInteger(ctx, 64, true);
        case WT_F32: return MLIR_CreateTypeFloat(ctx, 32, false);
        case WT_F64: return MLIR_CreateTypeFloat(ctx, 64, false);
    }
    return MLIR_CreateTypeInteger(ctx, 32, true);
}

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
static MLIR_AttributeHandle attr_s_cstr(MLIR_Context *ctx, const char *name,
                                        const char *v) {
    return attr_s(ctx, name, v, v ? strlen(v) : 0);
}
static MLIR_AttributeHandle attr_s_hex(MLIR_Context *ctx, Arena *arena,
                                       const char *name,
                                       const uint8_t *p, size_t n) {
    char *enc = hex_encode_alloc_arena(arena, p, n);
    return attr_s(ctx, name, enc, n * 2);
}

// Construct an op with up to `n_attrs` attributes, returning the op
// handle and (if has_result) populating *out_result with the result
// value handle of the requested valtype.
static MLIR_OpHandle make_op(MLIR_Context *ctx,
                             MLIR_OpType type,
                             MLIR_AttributeHandle *attrs, size_t n_attrs,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_RegionHandle *regions, size_t n_regions,
                             uint8_t result_vt /* 0 if none */,
                             MLIR_ValueHandle *out_result) {
    MLIR_TypeHandle res_tys[1];
    MLIR_ValueHandle res_vals[1];
    size_t n_res = 0;
    if (result_vt) {
        res_tys[0] = vt_to_type(ctx, result_vt);
        res_vals[0] = MLIR_CreateValueOpResult(ctx, MLIR_INVALID_HANDLE, 0,
                                               res_tys[0], (string){0},
                                               MLIR_CreateLocationUnknown(ctx, (string){0}));
        n_res = 1;
    }
    MLIR_OpHandle op = MLIR_CreateOp(ctx, type, op_type_to_string(type),
        attrs, n_attrs, res_tys, n_res, res_vals, n_res,
        operands, n_operands, regions, n_regions,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    if (out_result) *out_result = n_res ? res_vals[0] : MLIR_INVALID_HANDLE;
    return op;
}

// =============================================================================
// Read-back attribute helpers.
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
static char *strdup_str_nul(string s) {
    char *r = (char *)malloc(s.size + 1);
    if (s.size) memcpy(r, s.str, s.size);
    r[s.size] = 0;
    return r;
}

// =============================================================================
// Common helpers for building module/region/block.
// =============================================================================
static MLIR_OpHandle make_module(MLIR_Context *ctx, MLIR_BlockHandle *out_body) {
    MLIR_BlockHandle body = MLIR_CreateBlock(ctx);
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, body);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("builtin.module"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    *out_body = body;
    return op;
}

// =============================================================================
// wasmssa_module_t -> MLIR
// =============================================================================
//
// Body op operands are wired by ssa_def index: index in [0, n_params)
// is param i; index in [n_params, n_params + i_op) is op i_op's result.
// We construct block args for the params and then walk ops, recording
// each op's result value (if any) into a parallel array indexed by
// (n_params + op_index).

static void emit_global_op(MLIR_Context *ctx, Arena *arena,
                           MLIR_BlockHandle body, const wasm_global_t *g,
                           MLIR_OpType op_type) {
    // For relocs we encode as a single string attribute "offset:target:addend,..."
    // (target names are arbitrary identifiers; we just don't allow ':' or ','
    // in them, which is fine since they come from C identifiers).
    MLIR_AttributeHandle attrs[16];
    size_t na = 0;
    attrs[na++] = attr_s_cstr(ctx, "sym_name", g->name ? g->name : "");
    attrs[na++] = attr_i32(ctx, "size", g->size);
    attrs[na++] = attr_i32(ctx, "align_pow", g->align_pow);
    attrs[na++] = attr_b(ctx, "is_const", g->is_const);
    if (g->size) {
        attrs[na++] = attr_s(ctx, "init_data", (const char *)g->data, g->size);
    } else {
        attrs[na++] = attr_s(ctx, "init_data", "", 0);
    }
    if (g->n_relocs) {
        // Compute required text buffer size.
        size_t cap = 1;
        for (size_t i = 0; i < g->n_relocs; i++) {
            cap += strlen(g->relocs[i].target ? g->relocs[i].target : "") + 32;
        }
        char *buf = (char *)arena_alloc(arena, cap);
        size_t off = 0;
        for (size_t i = 0; i < g->n_relocs; i++) {
            int n = snprintf(buf + off, cap - off, "%s%u:%s:%d",
                             i ? "," : "",
                             g->relocs[i].offset,
                             g->relocs[i].target ? g->relocs[i].target : "",
                             g->relocs[i].addend);
            if (n < 0) break;
            off += (size_t)n;
        }
        attrs[na++] = attr_s(ctx, "relocs", buf, off);
    }
    MLIR_OpHandle op = make_op(ctx, op_type, attrs, na, NULL, 0, NULL, 0, 0, NULL);
    MLIR_AppendBlockOp(ctx, body, op);
}

static void emit_func_op(MLIR_Context *ctx, Arena *arena,
                         MLIR_BlockHandle body,
                         const wasmssa_func_t *f);

MLIR_OpHandle wasmssa_module_to_mlir(MLIR_Context *ctx,
                                     const wasmssa_module_t *m) {
    if (!m) return MLIR_INVALID_HANDLE;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    MLIR_BlockHandle body;
    MLIR_OpHandle module = make_module(ctx, &body);

    for (size_t i = 0; i < m->n_funcs; i++) {
        emit_func_op(ctx, arena, body, &m->funcs[i]);
    }
    for (size_t i = 0; i < m->n_globals; i++) {
        emit_global_op(ctx, arena, body, &m->globals[i],
                       OP_TYPE_WASMSSA_IMPORT_GLOBAL);
    }
    return module;
}

static void emit_func_op(MLIR_Context *ctx, Arena *arena,
                         MLIR_BlockHandle body,
                         const wasmssa_func_t *f) {
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s_cstr(ctx, "sym_name", f->name ? f->name : "");
    attrs[na++] = attr_s_hex(ctx, arena, "param_types", f->param_types, f->n_params);
    attrs[na++] = attr_s_hex(ctx, arena, "result_types", f->result_types, f->n_results);
    attrs[na++] = attr_b(ctx, "exported", f->exported);

    if (f->imported) {
        MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSSA_IMPORT_FUNC,
                                   attrs, na, NULL, 0, NULL, 0, 0, NULL);
        MLIR_AppendBlockOp(ctx, body, op);
        return;
    }
    attrs[na++] = attr_s_hex(ctx, arena, "carrier_types", f->carrier_vts, f->n_carriers);

    // Build the function body region with one block; block args = params.
    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    MLIR_ValueHandle *params = NULL;
    if (f->n_params) {
        params = (MLIR_ValueHandle *)arena_alloc(arena, f->n_params * sizeof(MLIR_ValueHandle));
    }
    for (size_t i = 0; i < f->n_params; i++) {
        MLIR_TypeHandle ty = vt_to_type(ctx, f->param_types[i]);
        MLIR_ValueHandle bv = MLIR_CreateValueBlockArg(ctx, (string){0}, (uint32_t)i, ty,
                                                      MLIR_CreateLocationUnknown(ctx, (string){0}));
        MLIR_AppendBlockArg(ctx, blk, bv);
        params[i] = bv;
    }

    // ssa_def_idx -> value handle.
    size_t n_ssa = f->n_params + f->n_ops;
    MLIR_ValueHandle *vmap = (MLIR_ValueHandle *)calloc(n_ssa ? n_ssa : 1,
                                                       sizeof(MLIR_ValueHandle));
    for (size_t i = 0; i < f->n_params; i++) vmap[i] = params[i];

    for (size_t i = 0; i < f->n_ops; i++) {
        const wasmssa_op_t *o = &f->ops[i];
        MLIR_ValueHandle ops_buf[16];
        MLIR_ValueHandle *ops = ops_buf;
        size_t n_ops = (size_t)(o->n_operands < 0 ? 0 : o->n_operands);
        if (n_ops > 16) {
            ops = (MLIR_ValueHandle *)arena_alloc(arena, n_ops * sizeof(MLIR_ValueHandle));
        }
        for (size_t k = 0; k < n_ops; k++) {
            int idx = o->operands[k];
            ops[k] = (idx >= 0 && (size_t)idx < n_ssa) ? vmap[idx] : MLIR_INVALID_HANDLE;
        }
        // Build attribute list based on op type.
        MLIR_AttributeHandle as[8];
        size_t nas = 0;
        as[nas++] = attr_i32(ctx, "valtype", o->valtype);
        switch (o->type) {
        case OP_TYPE_WASMSSA_CONST:
            as[nas++] = attr_i64(ctx, "value", o->i_const);
            break;
        case OP_TYPE_WASMSSA_BINOP:
        case OP_TYPE_WASMSSA_UNOP:
            as[nas++] = attr_i32(ctx, "wasm_opcode", o->wasm_opcode);
            break;
        case OP_TYPE_WASMSSA_LOAD:
        case OP_TYPE_WASMSSA_STORE:
            as[nas++] = attr_i32(ctx, "memory_offset", (int64_t)o->memory_offset);
            as[nas++] = attr_i32(ctx, "memory_align_log2", (int64_t)o->memory_align_log2);
            as[nas++] = attr_i32(ctx, "mem_size_bytes", (int64_t)o->mem_size_bytes);
            break;
        case OP_TYPE_WASMSSA_GLOBAL_GET:
        case OP_TYPE_WASMSSA_GLOBAL_SET:
            as[nas++] = attr_i32(ctx, "global_idx", (int64_t)o->global_idx);
            break;
        case OP_TYPE_WASMSSA_CALL:
        case OP_TYPE_WASMSSA_ADDRESSOF:
        case OP_TYPE_WASMSSA_FUNC_ADDR:
            as[nas++] = attr_s_cstr(ctx, "target", o->call_target ? o->call_target : "");
            break;
        case OP_TYPE_WASMSSA_CALL_INDIRECT:
            as[nas++] = attr_s_hex(ctx, arena, "sig_params", o->sig_params, o->n_sig_params);
            as[nas++] = attr_s_hex(ctx, arena, "sig_results", o->sig_results, o->n_sig_results);
            break;
        case OP_TYPE_WASMSSA_BR:
        case OP_TYPE_WASMSSA_BR_IF:
            as[nas++] = attr_i32(ctx, "depth", (int64_t)o->br_depth);
            break;
        case OP_TYPE_WASMSSA_CARRIER_SET:
        case OP_TYPE_WASMSSA_CARRIER_GET:
            as[nas++] = attr_i32(ctx, "carrier_id", (int64_t)o->carrier_id);
            break;
        default: break;
        }
        MLIR_ValueHandle res = MLIR_INVALID_HANDLE;
        MLIR_OpHandle mop = make_op(ctx, o->type, as, nas, ops, n_ops, NULL, 0,
                                    o->has_result ? o->valtype : 0,
                                    &res);
        MLIR_AppendBlockOp(ctx, blk, mop);
        if (o->has_result) vmap[f->n_params + i] = res;
    }
    free(vmap);

    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_WASMSSA_FUNC,
        op_type_to_string(OP_TYPE_WASMSSA_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, body, op);
}

// =============================================================================
// MLIR -> wasmssa_module_t
// =============================================================================

static bool parse_relocs(string s, wasm_global_t *g) {
    if (s.size == 0) return true;
    size_t i = 0;
    while (i < s.size) {
        // offset
        uint32_t off = 0;
        while (i < s.size && s.str[i] >= '0' && s.str[i] <= '9') {
            off = off * 10 + (uint32_t)(s.str[i] - '0'); i++;
        }
        if (i >= s.size || s.str[i] != ':') return false; i++;
        // target
        size_t ts = i;
        while (i < s.size && s.str[i] != ':') i++;
        if (i >= s.size) return false;
        size_t tlen = i - ts;
        char *tgt = (char *)malloc(tlen + 1);
        memcpy(tgt, s.str + ts, tlen); tgt[tlen] = 0;
        i++; // skip ':'
        bool neg = false;
        if (i < s.size && s.str[i] == '-') { neg = true; i++; }
        int32_t ad = 0;
        while (i < s.size && s.str[i] >= '0' && s.str[i] <= '9') {
            ad = ad * 10 + (int32_t)(s.str[i] - '0'); i++;
        }
        if (neg) ad = -ad;
        wasm_global_add_reloc(g, off, tgt, ad);
        free(tgt);
        if (i < s.size && s.str[i] == ',') i++;
    }
    return true;
}

wasmssa_module_t *mlir_to_wasmssa_module(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx;
    if (!module) return NULL;
    wasmssa_module_t *m = wasmssa_module_new();
    if (MLIR_GetOpNumRegions(module) < 1) return m;
    MLIR_RegionHandle r = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(r) < 1) return m;
    MLIR_BlockHandle body = MLIR_GetRegionBlock(r, 0);
    size_t n_top = MLIR_GetBlockNumOps(body);
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(body, ti);
        MLIR_OpType tt = MLIR_GetOpType(top);
        if (tt == OP_TYPE_WASMSSA_IMPORT_FUNC || tt == OP_TYPE_WASMSSA_FUNC) {
            wasmssa_func_t *f = wasmssa_module_add_func(m);
            string nm = at_s(top, "sym_name");
            f->name = strdup_str_nul(nm);
            f->imported = (tt == OP_TYPE_WASMSSA_IMPORT_FUNC);
            f->exported = at_b(top, "exported");
            string pt = at_s(top, "param_types");
            f->param_types = hex_decode_alloc(pt, &f->n_params);
            string rt = at_s(top, "result_types");
            f->result_types = hex_decode_alloc(rt, &f->n_results);
            if (!f->imported) {
                string ct = at_s(top, "carrier_types");
                f->carrier_vts = hex_decode_alloc(ct, &f->n_carriers);
                if (MLIR_GetOpNumRegions(top) < 1) continue;
                MLIR_RegionHandle fr = MLIR_GetOpRegion(top, 0);
                if (MLIR_GetRegionNumBlocks(fr) < 1) continue;
                MLIR_BlockHandle fb = MLIR_GetRegionBlock(fr, 0);
                size_t n_args = MLIR_GetBlockNumArgs(fb);
                size_t n_body = MLIR_GetBlockNumOps(fb);
                // Build a value->ssa_def_idx map.
                size_t n_slots = n_args + n_body;
                MLIR_ValueHandle *keys = (MLIR_ValueHandle *)calloc(n_slots ? n_slots : 1,
                                                                    sizeof(MLIR_ValueHandle));
                int *idxs = (int *)calloc(n_slots ? n_slots : 1, sizeof(int));
                size_t nk = 0;
                for (size_t i = 0; i < n_args; i++) {
                    keys[nk] = MLIR_GetBlockArg(fb, i);
                    idxs[nk] = (int)i;
                    nk++;
                }
                for (size_t i = 0; i < n_body; i++) {
                    MLIR_OpHandle bo = MLIR_GetBlockOp(fb, i);
                    wasmssa_op_t *o = wasmssa_func_add_op(f);
                    o->type = MLIR_GetOpType(bo);
                    o->valtype = (uint8_t)at_i(bo, "valtype");
                    size_t n_ops = MLIR_GetOpNumOperands(bo);
                    if (n_ops) {
                        o->n_operands = (int)n_ops;
                        o->operands = (int *)malloc(n_ops * sizeof(int));
                        for (size_t k = 0; k < n_ops; k++) {
                            MLIR_ValueHandle ov = MLIR_GetOpOperand(bo, k);
                            int found = -1;
                            for (size_t s = 0; s < nk; s++) {
                                if (keys[s] == ov) { found = idxs[s]; break; }
                            }
                            o->operands[k] = found;
                        }
                    }
                    switch (o->type) {
                    case OP_TYPE_WASMSSA_CONST:
                        o->i_const = at_i(bo, "value");
                        o->has_result = true;
                        break;
                    case OP_TYPE_WASMSSA_BINOP:
                    case OP_TYPE_WASMSSA_UNOP:
                        o->wasm_opcode = (uint8_t)at_i(bo, "wasm_opcode");
                        o->has_result = true;
                        break;
                    case OP_TYPE_WASMSSA_LOAD:
                        o->memory_offset = (uint32_t)at_i(bo, "memory_offset");
                        o->memory_align_log2 = (uint32_t)at_i(bo, "memory_align_log2");
                        o->mem_size_bytes = (uint32_t)at_i(bo, "mem_size_bytes");
                        o->has_result = true;
                        break;
                    case OP_TYPE_WASMSSA_STORE:
                        o->memory_offset = (uint32_t)at_i(bo, "memory_offset");
                        o->memory_align_log2 = (uint32_t)at_i(bo, "memory_align_log2");
                        o->mem_size_bytes = (uint32_t)at_i(bo, "mem_size_bytes");
                        break;
                    case OP_TYPE_WASMSSA_GLOBAL_GET:
                        o->global_idx = (uint32_t)at_i(bo, "global_idx");
                        o->has_result = true;
                        break;
                    case OP_TYPE_WASMSSA_GLOBAL_SET:
                        o->global_idx = (uint32_t)at_i(bo, "global_idx");
                        break;
                    case OP_TYPE_WASMSSA_CALL: {
                        string tgt = at_s(bo, "target");
                        o->call_target = strdup_str_nul(tgt);
                        o->has_result = (MLIR_GetOpNumResults(bo) > 0);
                        break;
                    }
                    case OP_TYPE_WASMSSA_ADDRESSOF:
                    case OP_TYPE_WASMSSA_FUNC_ADDR: {
                        string tgt = at_s(bo, "target");
                        o->call_target = strdup_str_nul(tgt);
                        o->has_result = true;
                        break;
                    }
                    case OP_TYPE_WASMSSA_CALL_INDIRECT: {
                        string sp = at_s(bo, "sig_params");
                        string sr = at_s(bo, "sig_results");
                        o->sig_params = hex_decode_alloc(sp, &o->n_sig_params);
                        o->sig_results = hex_decode_alloc(sr, &o->n_sig_results);
                        o->has_result = (MLIR_GetOpNumResults(bo) > 0);
                        break;
                    }
                    case OP_TYPE_WASMSSA_BR:
                    case OP_TYPE_WASMSSA_BR_IF:
                        o->br_depth = (uint32_t)at_i(bo, "depth");
                        break;
                    case OP_TYPE_WASMSSA_CARRIER_SET:
                        o->carrier_id = (uint32_t)at_i(bo, "carrier_id");
                        break;
                    case OP_TYPE_WASMSSA_CARRIER_GET:
                        o->carrier_id = (uint32_t)at_i(bo, "carrier_id");
                        o->has_result = true;
                        break;
                    case OP_TYPE_WASMSSA_ADD:
                    case OP_TYPE_WASMSSA_SUB:
                    case OP_TYPE_WASMSSA_SELECT:
                    case OP_TYPE_WASMSSA_EQZ:
                    case OP_TYPE_WASMSSA_EXTEND_I32_S:
                        o->has_result = true;
                        break;
                    default: break;
                    }
                    if (o->has_result) {
                        MLIR_ValueHandle rv = MLIR_GetOpResult(bo, 0);
                        keys[nk] = rv;
                        idxs[nk] = (int)(f->n_params + i);
                        nk++;
                    }
                }
                free(keys); free(idxs);
            }
        } else if (tt == OP_TYPE_WASMSSA_IMPORT_GLOBAL) {
            wasm_global_t *g = wasmssa_module_add_global(m);
            string nm = at_s(top, "sym_name");
            g->name = strdup_str_nul(nm);
            g->size = (uint32_t)at_i(top, "size");
            g->align_pow = (uint32_t)at_i(top, "align_pow");
            g->is_const = at_b(top, "is_const");
            string id = at_s(top, "init_data");
            if (id.size) {
                g->data = (uint8_t *)malloc(id.size);
                memcpy(g->data, id.str, id.size);
            } else {
                g->data = (uint8_t *)malloc(1);
            }
            string rl = at_s(top, "relocs");
            parse_relocs(rl, g);
        }
    }
    return m;
}

// =============================================================================
// wasmstack_module_t -> MLIR
// =============================================================================
static void emit_stack_func_op(MLIR_Context *ctx, Arena *arena,
                               MLIR_BlockHandle body,
                               const wasmstack_func_t *f) {
    MLIR_AttributeHandle attrs[8];
    size_t na = 0;
    attrs[na++] = attr_s_cstr(ctx, "sym_name", f->name ? f->name : "");
    attrs[na++] = attr_s_hex(ctx, arena, "param_types", f->param_types, f->n_params);
    attrs[na++] = attr_s_hex(ctx, arena, "result_types", f->result_types, f->n_results);
    attrs[na++] = attr_b(ctx, "exported", f->exported);
    if (f->imported) {
        MLIR_OpHandle op = make_op(ctx, OP_TYPE_WASMSTACK_IMPORT_FUNC,
                                   attrs, na, NULL, 0, NULL, 0, 0, NULL);
        MLIR_AppendBlockOp(ctx, body, op);
        return;
    }
    attrs[na++] = attr_s_hex(ctx, arena, "local_types", f->local_types, f->n_locals);

    MLIR_BlockHandle blk = MLIR_CreateBlock(ctx);
    for (size_t i = 0; i < f->n_ops; i++) {
        const wasmstack_op_t *o = &f->ops[i];
        MLIR_AttributeHandle as[8];
        size_t nas = 0;
        as[nas++] = attr_i32(ctx, "valtype", o->valtype);
        switch (o->type) {
        case OP_TYPE_WASMSTACK_LOCAL_GET:
        case OP_TYPE_WASMSTACK_LOCAL_SET:
        case OP_TYPE_WASMSTACK_LOCAL_TEE:
            as[nas++] = attr_i32(ctx, "local_idx", (int64_t)o->local_idx);
            break;
        case OP_TYPE_WASMSTACK_GLOBAL_GET:
        case OP_TYPE_WASMSTACK_GLOBAL_SET:
            as[nas++] = attr_i32(ctx, "global_idx", (int64_t)o->global_idx);
            break;
        case OP_TYPE_WASMSTACK_CONST:
            as[nas++] = attr_i64(ctx, "value", o->i_const);
            break;
        case OP_TYPE_WASMSTACK_BINOP:
        case OP_TYPE_WASMSTACK_UNOP:
            as[nas++] = attr_i32(ctx, "wasm_opcode", o->wasm_opcode);
            break;
        case OP_TYPE_WASMSTACK_LOAD:
        case OP_TYPE_WASMSTACK_STORE:
            as[nas++] = attr_i32(ctx, "memory_offset", (int64_t)o->memory_offset);
            as[nas++] = attr_i32(ctx, "memory_align_log2", (int64_t)o->memory_align_log2);
            as[nas++] = attr_i32(ctx, "mem_size_bytes", (int64_t)o->mem_size_bytes);
            break;
        case OP_TYPE_WASMSTACK_CALL:
        case OP_TYPE_WASMSTACK_ADDRESSOF:
        case OP_TYPE_WASMSTACK_FUNC_ADDR:
            as[nas++] = attr_s_cstr(ctx, "target", o->call_target ? o->call_target : "");
            break;
        case OP_TYPE_WASMSTACK_CALL_INDIRECT:
            as[nas++] = attr_s_hex(ctx, arena, "sig_params", o->sig_params, o->n_sig_params);
            as[nas++] = attr_s_hex(ctx, arena, "sig_results", o->sig_results, o->n_sig_results);
            break;
        case OP_TYPE_WASMSTACK_BR:
        case OP_TYPE_WASMSTACK_BR_IF:
            as[nas++] = attr_i32(ctx, "depth", (int64_t)o->br_depth);
            break;
        default: break;
        }
        MLIR_OpHandle mop = make_op(ctx, o->type, as, nas, NULL, 0, NULL, 0, 0, NULL);
        MLIR_AppendBlockOp(ctx, blk, mop);
    }
    MLIR_RegionHandle region = MLIR_CreateRegion(ctx);
    MLIR_AppendRegionBlock(ctx, region, blk);
    MLIR_RegionHandle regs[1] = { region };
    MLIR_OpHandle op = MLIR_CreateOp(ctx, OP_TYPE_WASMSTACK_FUNC,
        op_type_to_string(OP_TYPE_WASMSTACK_FUNC),
        attrs, na, NULL, 0, NULL, 0, NULL, 0, regs, 1,
        MLIR_CreateLocationUnknown(ctx, (string){0}),
        MLIR_INVALID_HANDLE, (string){0}, -1);
    MLIR_AppendBlockOp(ctx, body, op);
}

MLIR_OpHandle wasmstack_module_to_mlir(MLIR_Context *ctx,
                                       const wasmstack_module_t *m) {
    if (!m) return MLIR_INVALID_HANDLE;
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    MLIR_BlockHandle body;
    MLIR_OpHandle module = make_module(ctx, &body);
    for (size_t i = 0; i < m->n_funcs; i++) {
        emit_stack_func_op(ctx, arena, body, &m->funcs[i]);
    }
    for (size_t i = 0; i < m->n_globals; i++) {
        emit_global_op(ctx, arena, body, &m->globals[i],
                       OP_TYPE_WASMSTACK_IMPORT_GLOBAL);
    }
    return module;
}

// =============================================================================
// MLIR -> wasmstack_module_t
// =============================================================================
wasmstack_module_t *mlir_to_wasmstack_module(MLIR_Context *ctx, MLIR_OpHandle module) {
    (void)ctx;
    if (!module) return NULL;
    wasmstack_module_t *m = wasmstack_module_new();
    if (MLIR_GetOpNumRegions(module) < 1) return m;
    MLIR_RegionHandle r = MLIR_GetOpRegion(module, 0);
    if (MLIR_GetRegionNumBlocks(r) < 1) return m;
    MLIR_BlockHandle body = MLIR_GetRegionBlock(r, 0);
    size_t n_top = MLIR_GetBlockNumOps(body);
    for (size_t ti = 0; ti < n_top; ti++) {
        MLIR_OpHandle top = MLIR_GetBlockOp(body, ti);
        MLIR_OpType tt = MLIR_GetOpType(top);
        if (tt == OP_TYPE_WASMSTACK_IMPORT_FUNC || tt == OP_TYPE_WASMSTACK_FUNC) {
            wasmstack_func_t *f = wasmstack_module_add_func(m);
            f->name = strdup_str_nul(at_s(top, "sym_name"));
            f->imported = (tt == OP_TYPE_WASMSTACK_IMPORT_FUNC);
            f->exported = at_b(top, "exported");
            f->param_types = hex_decode_alloc(at_s(top, "param_types"), &f->n_params);
            f->result_types = hex_decode_alloc(at_s(top, "result_types"), &f->n_results);
            if (!f->imported) {
                f->local_types = hex_decode_alloc(at_s(top, "local_types"), &f->n_locals);
                f->c_locals = f->n_locals;
                if (MLIR_GetOpNumRegions(top) < 1) continue;
                MLIR_RegionHandle fr = MLIR_GetOpRegion(top, 0);
                if (MLIR_GetRegionNumBlocks(fr) < 1) continue;
                MLIR_BlockHandle fb = MLIR_GetRegionBlock(fr, 0);
                size_t n_body = MLIR_GetBlockNumOps(fb);
                for (size_t i = 0; i < n_body; i++) {
                    MLIR_OpHandle bo = MLIR_GetBlockOp(fb, i);
                    wasmstack_op_t *o = wasmstack_func_add_op(f);
                    o->type = MLIR_GetOpType(bo);
                    o->valtype = (uint8_t)at_i(bo, "valtype");
                    switch (o->type) {
                    case OP_TYPE_WASMSTACK_LOCAL_GET:
                    case OP_TYPE_WASMSTACK_LOCAL_SET:
                    case OP_TYPE_WASMSTACK_LOCAL_TEE:
                        o->local_idx = (uint32_t)at_i(bo, "local_idx");
                        break;
                    case OP_TYPE_WASMSTACK_GLOBAL_GET:
                    case OP_TYPE_WASMSTACK_GLOBAL_SET:
                        o->global_idx = (uint32_t)at_i(bo, "global_idx");
                        break;
                    case OP_TYPE_WASMSTACK_CONST:
                        o->i_const = at_i(bo, "value");
                        break;
                    case OP_TYPE_WASMSTACK_BINOP:
                    case OP_TYPE_WASMSTACK_UNOP:
                        o->wasm_opcode = (uint8_t)at_i(bo, "wasm_opcode");
                        break;
                    case OP_TYPE_WASMSTACK_LOAD:
                    case OP_TYPE_WASMSTACK_STORE:
                        o->memory_offset = (uint32_t)at_i(bo, "memory_offset");
                        o->memory_align_log2 = (uint32_t)at_i(bo, "memory_align_log2");
                        o->mem_size_bytes = (uint32_t)at_i(bo, "mem_size_bytes");
                        break;
                    case OP_TYPE_WASMSTACK_CALL:
                    case OP_TYPE_WASMSTACK_ADDRESSOF:
                    case OP_TYPE_WASMSTACK_FUNC_ADDR:
                        o->call_target = strdup_str_nul(at_s(bo, "target"));
                        break;
                    case OP_TYPE_WASMSTACK_CALL_INDIRECT:
                        o->sig_params = hex_decode_alloc(at_s(bo, "sig_params"),
                                                         &o->n_sig_params);
                        o->sig_results = hex_decode_alloc(at_s(bo, "sig_results"),
                                                          &o->n_sig_results);
                        break;
                    case OP_TYPE_WASMSTACK_BR:
                    case OP_TYPE_WASMSTACK_BR_IF:
                        o->br_depth = (uint32_t)at_i(bo, "depth");
                        break;
                    default: break;
                    }
                }
            }
        } else if (tt == OP_TYPE_WASMSTACK_IMPORT_GLOBAL) {
            wasm_global_t *g = wasmstack_module_add_global(m);
            g->name = strdup_str_nul(at_s(top, "sym_name"));
            g->size = (uint32_t)at_i(top, "size");
            g->align_pow = (uint32_t)at_i(top, "align_pow");
            g->is_const = at_b(top, "is_const");
            string id = at_s(top, "init_data");
            if (id.size) {
                g->data = (uint8_t *)malloc(id.size);
                memcpy(g->data, id.str, id.size);
            } else {
                g->data = (uint8_t *)malloc(1);
            }
            parse_relocs(at_s(top, "relocs"), g);
        }
    }
    return m;
}

// =============================================================================
// Public adapter wrappers: each chains struct-form stage with converters.
// =============================================================================
MLIR_OpHandle mlir_lower_llvm_to_wasmssa(MLIR_Context *ctx, MLIR_OpHandle module) {
    wasmssa_module_t *ssa = mlir_lower_llvm_to_wasmssa_struct(ctx, module);
    if (!ssa) return MLIR_INVALID_HANDLE;
    MLIR_OpHandle out = wasmssa_module_to_mlir(ctx, ssa);
    wasmssa_module_free(ssa);
    return out;
}

MLIR_OpHandle mlir_stackify_wasmssa(MLIR_Context *ctx, MLIR_OpHandle ssa_module) {
    wasmssa_module_t *ssa = mlir_to_wasmssa_module(ctx, ssa_module);
    if (!ssa) return MLIR_INVALID_HANDLE;
    wasmstack_module_t *stk = mlir_stackify_wasmssa_struct(ssa);
    wasmssa_module_free(ssa);
    if (!stk) return MLIR_INVALID_HANDLE;
    MLIR_OpHandle out = wasmstack_module_to_mlir(ctx, stk);
    wasmstack_module_free(stk);
    return out;
}

string mlir_translate_wasmstack_to_binary(MLIR_Context *ctx, MLIR_OpHandle stk_module) {
    string fail = {0};
    wasmstack_module_t *stk = mlir_to_wasmstack_module(ctx, stk_module);
    if (!stk) return fail;
    string r = mlir_translate_wasmstack_to_binary_struct(ctx, stk);
    wasmstack_module_free(stk);
    return r;
}

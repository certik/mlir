// Pretty-printers for the wasmssa / wasmstack C-struct IRs and a
// minimal disassembler for the wasm32 relocatable object bytes
// produced by stage 3. Used by `tinyc --emit=wasmssa|wasmstack|wat`
// to inspect intermediate stages of the native LLVM -> WASM pipeline.
//
// All three entry points allocate their final result string into the
// MLIR_Context's arena. They use a private malloc-backed growable
// Buf internally (same pattern as mlir_translate_to_llvm_ir.c /
// mlir_translate_wasmstack_to_binary.c) and copy out at the end.

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
#include "mlir_wasm_dialect.h"

// =============================================================================
// Growable byte buffer.
// =============================================================================

typedef struct { char *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 1024;
    while (b->len + add > nc) nc *= 2;
    b->data = (char *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_putc(Buf *b, char c) { buf_grow(b, 1); b->data[b->len++] = c; }
static void buf_append(Buf *b, const char *s, size_t n) {
    buf_grow(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
}
static void buf_cstr(Buf *b, const char *s) { buf_append(b, s, strlen(s)); }
static void buf_str(Buf *b, string s)       { buf_append(b, s.str, s.size); }
static void buf_printf(Buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[256];
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return; }
    if ((size_t)n < sizeof(tmp)) {
        buf_append(b, tmp, (size_t)n);
    } else {
        char *big = (char *)malloc((size_t)n + 1);
        vsnprintf(big, (size_t)n + 1, fmt, ap2);
        buf_append(b, big, (size_t)n);
        free(big);
    }
    va_end(ap2);
}
static void buf_indent(Buf *b, int n) { for (int i = 0; i < n; i++) buf_putc(b, ' '); }

static string buf_to_arena_string(MLIR_Context *ctx, Buf *b) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *out = (char *)arena_alloc(arena, b->len ? b->len : 1);
    if (b->len) memcpy(out, b->data, b->len);
    string r = { out, b->len };
    free(b->data);
    b->data = NULL; b->len = b->cap = 0;
    return r;
}

// =============================================================================
// Shared helpers.
// =============================================================================

static const char *vt_name(uint8_t vt) {
    switch (vt) {
        case WT_I32: return "i32";
        case WT_I64: return "i64";
        case WT_F32: return "f32";
        case WT_F64: return "f64";
        default: return "?";
    }
}

static void print_func_type(Buf *b,
                            const uint8_t *params, size_t n_params,
                            const uint8_t *results, size_t n_results) {
    buf_putc(b, '(');
    for (size_t i = 0; i < n_params; i++) {
        if (i) buf_cstr(b, ", ");
        buf_cstr(b, vt_name(params[i]));
    }
    buf_cstr(b, ") -> ");
    if (n_results == 0) {
        buf_cstr(b, "()");
    } else if (n_results == 1) {
        buf_cstr(b, vt_name(results[0]));
    } else {
        buf_putc(b, '(');
        for (size_t i = 0; i < n_results; i++) {
            if (i) buf_cstr(b, ", ");
            buf_cstr(b, vt_name(results[i]));
        }
        buf_putc(b, ')');
    }
}

static void print_op_name_quoted(Buf *b, MLIR_OpType t) {
    string s = op_type_to_string(t);
    buf_putc(b, '"');
    buf_append(b, s.str, s.size);
    buf_putc(b, '"');
}

static void print_data_bytes_escaped(Buf *b, const uint8_t *p, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        uint8_t c = p[i];
        if (c == '\\' || c == '"') { buf_putc(b, '\\'); buf_putc(b, (char)c); }
        else if (c >= 0x20 && c < 0x7f) { buf_putc(b, (char)c); }
        else { buf_printf(b, "\\%02x", c); }
    }
    buf_putc(b, '"');
}

// =============================================================================
// wasmssa printer.
// =============================================================================

static void wasmssa_print_operand(Buf *b, int op_idx, int n_params) {
    if (op_idx < 0) { buf_cstr(b, "<unbound>"); return; }
    if (op_idx < n_params) buf_printf(b, "%%arg%d", op_idx);
    else                   buf_printf(b, "%%%d", op_idx - n_params);
}

static void wasmssa_print_op(Buf *b, wasmssa_func_t *f, size_t op_index,
                             wasmssa_op_t *op, int indent) {
    int n_params = (int)f->n_params;
    buf_indent(b, indent);
    if (op->has_result) {
        buf_printf(b, "%%%zu = ", op_index);
    }
    print_op_name_quoted(b, op->type);

    // Operands.
    buf_putc(b, '(');
    for (int i = 0; i < op->n_operands; i++) {
        if (i) buf_cstr(b, ", ");
        wasmssa_print_operand(b, op->operands[i], n_params);
    }
    buf_putc(b, ')');

    // Attribute dict (op-specific, only fields relevant to type).
    Buf attrs = {0};
    bool first = true;
    #define KV(...) do { if (!first) buf_cstr(&attrs, ", "); first = false; \
                         buf_printf(&attrs, __VA_ARGS__); } while (0)
    switch (op->type) {
        case OP_TYPE_WASMSSA_CONST:
            KV("value = %lld : %s", (long long)op->i_const, vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_BINOP:
        case OP_TYPE_WASMSSA_UNOP:
            KV("wasm_opcode = 0x%02x : i8", op->wasm_opcode);
            KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_LOAD:
        case OP_TYPE_WASMSSA_STORE:
            KV("memory_offset = %u : i32", op->memory_offset);
            KV("memory_align_log2 = %u : i32", op->memory_align_log2);
            KV("mem_size_bytes = %u : i32", op->mem_size_bytes);
            KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_GLOBAL_GET:
        case OP_TYPE_WASMSSA_GLOBAL_SET:
            KV("global_idx = %u : i32", op->global_idx);
            KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_CALL:
            if (op->call_target) KV("callee = @%s", op->call_target);
            if (op->valtype) KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_CALL_INDIRECT:
            KV("sig = ");
            print_func_type(&attrs, op->sig_params, op->n_sig_params,
                            op->sig_results, op->n_sig_results);
            break;
        case OP_TYPE_WASMSSA_BR:
        case OP_TYPE_WASMSSA_BR_IF:
            KV("br_depth = %u : i32", op->br_depth);
            break;
        case OP_TYPE_WASMSSA_CARRIER_GET:
        case OP_TYPE_WASMSSA_CARRIER_SET:
            KV("carrier_id = %u : i32", op->carrier_id);
            if (op->valtype) KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_ADDRESSOF:
        case OP_TYPE_WASMSSA_FUNC_ADDR:
            if (op->call_target) KV("symbol = @%s", op->call_target);
            break;
        case OP_TYPE_WASMSSA_BLOCK_BEGIN:
        case OP_TYPE_WASMSSA_LOOP_BEGIN:
        case OP_TYPE_WASMSSA_IF_BEGIN:
            if (op->valtype) KV("result_type = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSSA_EXTEND_I32_S:
        case OP_TYPE_WASMSSA_EQZ:
        case OP_TYPE_WASMSSA_SELECT:
        case OP_TYPE_WASMSSA_ADD:
        case OP_TYPE_WASMSSA_SUB:
            if (op->valtype) KV("valtype = %s", vt_name(op->valtype));
            break;
        default:
            break;
    }
    #undef KV
    if (!first) {
        buf_cstr(b, " {");
        buf_append(b, attrs.data, attrs.len);
        buf_putc(b, '}');
    }
    free(attrs.data);

    // Type signature.
    buf_cstr(b, " : (");
    for (int i = 0; i < op->n_operands; i++) {
        if (i) buf_cstr(b, ", ");
        // We don't track per-operand types here; print the op's valtype if
        // it's typed, else "i32" as a placeholder. This matches what stage 2
        // assumes (every SSA value has one well-defined valtype).
        buf_cstr(b, op->valtype ? vt_name(op->valtype) : "i32");
    }
    buf_cstr(b, ") -> ");
    if (op->has_result) buf_cstr(b, op->valtype ? vt_name(op->valtype) : "i32");
    else                buf_cstr(b, "()");
    buf_putc(b, '\n');
}

static void wasmssa_print_global(Buf *b, wasm_global_t *g, int indent) {
    buf_indent(b, indent);
    buf_cstr(b, "\"wasmssa.global\"() {");
    buf_printf(b, "sym_name = \"%s\", size = %u : i32, align_pow = %u : i32, is_const = %s",
               g->name ? g->name : "", g->size, g->align_pow,
               g->is_const ? "true" : "false");
    if (g->n_relocs) {
        buf_cstr(b, ", relocs = [");
        for (size_t i = 0; i < g->n_relocs; i++) {
            if (i) buf_cstr(b, ", ");
            buf_printf(b, "{offset = %u, target = @%s, addend = %d}",
                       g->relocs[i].offset,
                       g->relocs[i].target ? g->relocs[i].target : "",
                       g->relocs[i].addend);
        }
        buf_putc(b, ']');
    }
    if (g->data && g->size) {
        buf_cstr(b, ", init = ");
        print_data_bytes_escaped(b, g->data, g->size);
    }
    buf_cstr(b, "} : () -> ()\n");
}

static void wasmssa_print_func(Buf *b, wasmssa_func_t *f, int indent) {
    buf_indent(b, indent);
    if (f->imported) {
        buf_cstr(b, "\"wasmssa.import_func\"() {");
        buf_printf(b, "sym_name = \"%s\", function_type = ", f->name ? f->name : "");
        print_func_type(b, f->param_types, f->n_params, f->result_types, f->n_results);
        buf_cstr(b, "} : () -> ()\n");
        return;
    }
    buf_cstr(b, "\"wasmssa.func\"() ({\n");
    buf_indent(b, indent + 2);
    buf_cstr(b, "^bb0");
    if (f->n_params > 0) {
        buf_putc(b, '(');
        for (size_t i = 0; i < f->n_params; i++) {
            if (i) buf_cstr(b, ", ");
            buf_printf(b, "%%arg%zu: %s", i, vt_name(f->param_types[i]));
        }
        buf_putc(b, ')');
    }
    buf_cstr(b, ":\n");
    for (size_t i = 0; i < f->n_ops; i++) {
        wasmssa_print_op(b, f, i, &f->ops[i], indent + 4);
    }
    buf_indent(b, indent);
    buf_cstr(b, "}) {");
    buf_printf(b, "sym_name = \"%s\", function_type = ", f->name ? f->name : "");
    print_func_type(b, f->param_types, f->n_params, f->result_types, f->n_results);
    if (f->exported) buf_cstr(b, ", exported = true");
    if (f->n_carriers) {
        buf_printf(b, ", n_carriers = %zu : i32", f->n_carriers);
        buf_cstr(b, ", carrier_types = [");
        for (size_t i = 0; i < f->n_carriers; i++) {
            if (i) buf_cstr(b, ", ");
            buf_cstr(b, vt_name(f->carrier_vts[i]));
        }
        buf_putc(b, ']');
    }
    buf_cstr(b, "} : () -> ()\n");
}

string wasmssa_module_print(MLIR_Context *ctx, wasmssa_module_t *m) {
    Buf b = {0};
    buf_cstr(&b, "\"builtin.module\"() ({\n");
    for (size_t i = 0; i < m->n_globals; i++) {
        wasmssa_print_global(&b, &m->globals[i], 2);
    }
    for (size_t i = 0; i < m->n_funcs; i++) {
        wasmssa_print_func(&b, &m->funcs[i], 2);
    }
    buf_cstr(&b, "}) : () -> ()\n");
    return buf_to_arena_string(ctx, &b);
}

// =============================================================================
// wasmstack printer.
// =============================================================================

static void wasmstack_print_op(Buf *b, wasmstack_op_t *op, int indent) {
    buf_indent(b, indent);
    print_op_name_quoted(b, op->type);

    Buf attrs = {0};
    bool first = true;
    #define KV(...) do { if (!first) buf_cstr(&attrs, ", "); first = false; \
                         buf_printf(&attrs, __VA_ARGS__); } while (0)
    switch (op->type) {
        case OP_TYPE_WASMSTACK_CONST:
            KV("value = %lld : %s", (long long)op->i_const, vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSTACK_LOCAL_GET:
        case OP_TYPE_WASMSTACK_LOCAL_SET:
        case OP_TYPE_WASMSTACK_LOCAL_TEE:
            KV("local_idx = %u : i32", op->local_idx);
            break;
        case OP_TYPE_WASMSTACK_GLOBAL_GET:
        case OP_TYPE_WASMSTACK_GLOBAL_SET:
            KV("global_idx = %u : i32", op->global_idx);
            if (op->valtype) KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSTACK_LOAD:
        case OP_TYPE_WASMSTACK_STORE:
            KV("memory_offset = %u : i32", op->memory_offset);
            KV("memory_align_log2 = %u : i32", op->memory_align_log2);
            KV("mem_size_bytes = %u : i32", op->mem_size_bytes);
            KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSTACK_BINOP:
        case OP_TYPE_WASMSTACK_UNOP:
            KV("wasm_opcode = 0x%02x : i8", op->wasm_opcode);
            KV("valtype = %s", vt_name(op->valtype));
            break;
        case OP_TYPE_WASMSTACK_CALL:
            if (op->call_target) KV("callee = @%s", op->call_target);
            break;
        case OP_TYPE_WASMSTACK_CALL_INDIRECT:
            KV("sig = ");
            print_func_type(&attrs, op->sig_params, op->n_sig_params,
                            op->sig_results, op->n_sig_results);
            break;
        case OP_TYPE_WASMSTACK_BR:
        case OP_TYPE_WASMSTACK_BR_IF:
            KV("br_depth = %u : i32", op->br_depth);
            break;
        case OP_TYPE_WASMSTACK_ADDRESSOF:
        case OP_TYPE_WASMSTACK_FUNC_ADDR:
            if (op->call_target) KV("symbol = @%s", op->call_target);
            break;
        case OP_TYPE_WASMSTACK_BLOCK:
        case OP_TYPE_WASMSTACK_LOOP:
        case OP_TYPE_WASMSTACK_IF:
            if (op->valtype) KV("result_type = %s", vt_name(op->valtype));
            break;
        default:
            if (op->valtype) KV("valtype = %s", vt_name(op->valtype));
            break;
    }
    #undef KV
    if (!first) {
        buf_cstr(b, " {");
        buf_append(b, attrs.data, attrs.len);
        buf_putc(b, '}');
    }
    free(attrs.data);
    buf_putc(b, '\n');
}

static void wasmstack_print_func(Buf *b, wasmstack_func_t *f, int indent) {
    buf_indent(b, indent);
    if (f->imported) {
        buf_cstr(b, "\"wasmstack.import_func\"() {");
        buf_printf(b, "sym_name = \"%s\", function_type = ", f->name ? f->name : "");
        print_func_type(b, f->param_types, f->n_params, f->result_types, f->n_results);
        buf_cstr(b, "} : () -> ()\n");
        return;
    }
    buf_cstr(b, "\"wasmstack.func\"() ({\n");
    buf_indent(b, indent + 2);
    buf_printf(b, "// sym_name = \"%s\", function_type = ", f->name ? f->name : "");
    print_func_type(b, f->param_types, f->n_params, f->result_types, f->n_results);
    buf_putc(b, '\n');
    if (f->n_locals) {
        buf_indent(b, indent + 2);
        buf_cstr(b, "// locals = [");
        for (size_t i = 0; i < f->n_locals; i++) {
            if (i) buf_cstr(b, ", ");
            buf_cstr(b, vt_name(f->local_types[i]));
        }
        buf_cstr(b, "]\n");
    }
    int nest = indent + 4;
    for (size_t i = 0; i < f->n_ops; i++) {
        wasmstack_op_t *op = &f->ops[i];
        int line_indent = nest;
        // End / else dedent before printing.
        if (op->type == OP_TYPE_WASMSTACK_END ||
            op->type == OP_TYPE_WASMSTACK_ELSE) {
            line_indent = nest - 2;
            if (line_indent < indent + 4) line_indent = indent + 4;
        }
        wasmstack_print_op(b, op, line_indent);
        // Adjust nest after block / loop / if / else / end.
        if (op->type == OP_TYPE_WASMSTACK_BLOCK ||
            op->type == OP_TYPE_WASMSTACK_LOOP  ||
            op->type == OP_TYPE_WASMSTACK_IF    ||
            op->type == OP_TYPE_WASMSTACK_ELSE) {
            nest += 2;
        } else if (op->type == OP_TYPE_WASMSTACK_END) {
            nest -= 2;
            if (nest < indent + 4) nest = indent + 4;
        }
    }
    buf_indent(b, indent);
    buf_cstr(b, "}) {");
    buf_printf(b, "sym_name = \"%s\"", f->name ? f->name : "");
    if (f->exported) buf_cstr(b, ", exported = true");
    buf_cstr(b, "} : () -> ()\n");
}

string wasmstack_module_print(MLIR_Context *ctx, wasmstack_module_t *m) {
    Buf b = {0};
    buf_cstr(&b, "\"builtin.module\"() ({\n");
    for (size_t i = 0; i < m->n_globals; i++) {
        wasm_global_t *g = &m->globals[i];
        buf_indent(&b, 2);
        buf_cstr(&b, "\"wasmstack.global\"() {");
        buf_printf(&b, "sym_name = \"%s\", size = %u : i32, align_pow = %u : i32, is_const = %s",
                   g->name ? g->name : "", g->size, g->align_pow,
                   g->is_const ? "true" : "false");
        if (g->data && g->size) {
            buf_cstr(&b, ", init = ");
            print_data_bytes_escaped(&b, g->data, g->size);
        }
        buf_cstr(&b, "} : () -> ()\n");
    }
    for (size_t i = 0; i < m->n_funcs; i++) {
        wasmstack_print_func(&b, &m->funcs[i], 2);
    }
    buf_cstr(&b, "}) : () -> ()\n");
    return buf_to_arena_string(ctx, &b);
}

// =============================================================================
// Wasm binary -> WAT decoder.
// =============================================================================

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    bool overflow;
} Rd;

static uint8_t rd_u8(Rd *r) {
    if (r->p >= r->end) { r->overflow = true; return 0; }
    return *r->p++;
}
static uint64_t rd_uleb(Rd *r) {
    uint64_t v = 0;
    int shift = 0;
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
    int64_t v = 0;
    int shift = 0;
    uint8_t b = 0;
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
static uint32_t rd_u32_le(Rd *r) {
    if (r->p + 4 > r->end) { r->overflow = true; r->p = r->end; return 0; }
    uint32_t v = (uint32_t)r->p[0] | ((uint32_t)r->p[1] << 8) |
                 ((uint32_t)r->p[2] << 16) | ((uint32_t)r->p[3] << 24);
    r->p += 4;
    return v;
}
static float rd_f32(Rd *r) {
    uint32_t bits = rd_u32_le(r);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
static double rd_f64(Rd *r) {
    if (r->p + 8 > r->end) { r->overflow = true; r->p = r->end; return 0.0; }
    uint64_t bits = 0;
    for (int i = 0; i < 8; i++) bits |= ((uint64_t)r->p[i]) << (i * 8);
    r->p += 8;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

typedef struct { const uint8_t *str; uint32_t size; } RdName;
static RdName rd_name(Rd *r) {
    uint32_t n = (uint32_t)rd_uleb(r);
    if (r->p + n > r->end) { r->overflow = true; n = (uint32_t)(r->end - r->p); }
    RdName rn = { r->p, n };
    r->p += n;
    return rn;
}

static const char *vt_name_byte(uint8_t b) {
    switch (b) {
        case 0x7f: return "i32";
        case 0x7e: return "i64";
        case 0x7d: return "f32";
        case 0x7c: return "f64";
        case 0x70: return "funcref";
        case 0x6f: return "externref";
        default:   return "?";
    }
}

// Name custom section: function-name subsection (id 1) -> idx -> name.
typedef struct { uint32_t idx; char *name; } NameEnt;
typedef struct { NameEnt *e; size_t n, cap; } NameMap;

static void namemap_put(NameMap *m, uint32_t idx, const uint8_t *p, uint32_t n) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->e = (NameEnt *)realloc(m->e, m->cap * sizeof(NameEnt));
    }
    char *s = (char *)malloc(n + 1);
    if (n) memcpy(s, p, n);
    s[n] = 0;
    m->e[m->n].idx = idx;
    m->e[m->n].name = s;
    m->n++;
}
static const char *namemap_get(NameMap *m, uint32_t idx) {
    for (size_t i = 0; i < m->n; i++) if (m->e[i].idx == idx) return m->e[i].name;
    return NULL;
}
static void namemap_free(NameMap *m) {
    for (size_t i = 0; i < m->n; i++) free(m->e[i].name);
    free(m->e);
}

static void scan_name_section(NameMap *fnames, NameMap *gnames,
                              const uint8_t *p, size_t n) {
    Rd r = { p, p + n, false };
    while (r.p < r.end && !r.overflow) {
        uint8_t sub = rd_u8(&r);
        uint32_t sz = (uint32_t)rd_uleb(&r);
        if (r.overflow || r.p + sz > r.end) return;
        const uint8_t *sub_end = r.p + sz;
        if (sub == 1 || sub == 7) {
            uint32_t cnt = (uint32_t)rd_uleb(&r);
            for (uint32_t i = 0; i < cnt && r.p < sub_end && !r.overflow; i++) {
                uint32_t idx = (uint32_t)rd_uleb(&r);
                RdName nm = rd_name(&r);
                namemap_put(sub == 1 ? fnames : gnames, idx, nm.str, nm.size);
            }
        }
        r.p = sub_end;
    }
}

static void fmt_funcref(Buf *b, NameMap *fnames, uint32_t idx) {
    const char *nm = namemap_get(fnames, idx);
    if (nm && nm[0]) buf_printf(b, " $%s", nm);
    else             buf_printf(b, " %u", idx);
}

// Decode one instruction. Returns the new nesting delta (+1 for
// block/loop/if/else-open, -1 for end). `nest` is updated by caller for
// indentation. `*ended` set true on `end` byte.
//
// Writes a complete line (without trailing newline; caller adds one).
static int decode_instr(Buf *b, Rd *r, NameMap *fnames, bool *ended) {
    *ended = false;
    if (r->p >= r->end) return 0;
    uint8_t op = rd_u8(r);
    switch (op) {
        case 0x00: buf_cstr(b, "unreachable"); return 0;
        case 0x01: buf_cstr(b, "nop"); return 0;
        case 0x02: case 0x03: case 0x04: {
            const char *kw = op == 0x02 ? "block" : op == 0x03 ? "loop" : "if";
            uint8_t bt = rd_u8(r);
            if (bt == 0x40) buf_printf(b, "%s", kw);
            else            buf_printf(b, "%s (result %s)", kw, vt_name_byte(bt));
            return 1;
        }
        case 0x05: buf_cstr(b, "else"); return 0; // caller handles dedent/indent
        case 0x0b: buf_cstr(b, "end"); *ended = true; return -1;
        case 0x0c: buf_printf(b, "br %u", (uint32_t)rd_uleb(r)); return 0;
        case 0x0d: buf_printf(b, "br_if %u", (uint32_t)rd_uleb(r)); return 0;
        case 0x0e: {
            uint32_t n = (uint32_t)rd_uleb(r);
            buf_cstr(b, "br_table");
            for (uint32_t i = 0; i < n; i++) buf_printf(b, " %u", (uint32_t)rd_uleb(r));
            buf_printf(b, " %u", (uint32_t)rd_uleb(r));
            return 0;
        }
        case 0x0f: buf_cstr(b, "return"); return 0;
        case 0x10: {
            uint32_t idx = (uint32_t)rd_uleb(r);
            buf_cstr(b, "call");
            fmt_funcref(b, fnames, idx);
            return 0;
        }
        case 0x11: {
            uint32_t tyidx = (uint32_t)rd_uleb(r);
            uint32_t tbl   = (uint32_t)rd_uleb(r);
            buf_printf(b, "call_indirect (type %u) %u", tyidx, tbl);
            return 0;
        }
        case 0x1a: buf_cstr(b, "drop"); return 0;
        case 0x1b: buf_cstr(b, "select"); return 0;
        case 0x20: buf_printf(b, "local.get %u",  (uint32_t)rd_uleb(r)); return 0;
        case 0x21: buf_printf(b, "local.set %u",  (uint32_t)rd_uleb(r)); return 0;
        case 0x22: buf_printf(b, "local.tee %u",  (uint32_t)rd_uleb(r)); return 0;
        case 0x23: buf_printf(b, "global.get %u", (uint32_t)rd_uleb(r)); return 0;
        case 0x24: buf_printf(b, "global.set %u", (uint32_t)rd_uleb(r)); return 0;
        // Loads / stores 0x28..0x3e: each is `(align uleb)(offset uleb)`.
        case 0x28: case 0x29: case 0x2a: case 0x2b: case 0x2c: case 0x2d:
        case 0x2e: case 0x2f: case 0x30: case 0x31: case 0x32: case 0x33:
        case 0x34: case 0x35:
        case 0x36: case 0x37: case 0x38: case 0x39:
        case 0x3a: case 0x3b: case 0x3c: case 0x3d: case 0x3e: {
            static const char *names[] = {
                "i32.load","i64.load","f32.load","f64.load",
                "i32.load8_s","i32.load8_u","i32.load16_s","i32.load16_u",
                "i64.load8_s","i64.load8_u","i64.load16_s","i64.load16_u",
                "i64.load32_s","i64.load32_u",
                "i32.store","i64.store","f32.store","f64.store",
                "i32.store8","i32.store16",
                "i64.store8","i64.store16","i64.store32"
            };
            uint32_t align = (uint32_t)rd_uleb(r);
            uint32_t off   = (uint32_t)rd_uleb(r);
            buf_printf(b, "%s offset=%u align=%u", names[op - 0x28], off, 1u << align);
            return 0;
        }
        case 0x3f: rd_u8(r); buf_cstr(b, "memory.size"); return 0;
        case 0x40: rd_u8(r); buf_cstr(b, "memory.grow"); return 0;
        case 0x41: buf_printf(b, "i32.const %lld", (long long)rd_sleb(r)); return 0;
        case 0x42: buf_printf(b, "i64.const %lld", (long long)rd_sleb(r)); return 0;
        case 0x43: buf_printf(b, "f32.const %g", (double)rd_f32(r)); return 0;
        case 0x44: buf_printf(b, "f64.const %g", rd_f64(r)); return 0;
        case 0xfc: {
            uint32_t sub = (uint32_t)rd_uleb(r);
            switch (sub) {
                case 0:  buf_cstr(b, "i32.trunc_sat_f32_s"); break;
                case 1:  buf_cstr(b, "i32.trunc_sat_f32_u"); break;
                case 2:  buf_cstr(b, "i32.trunc_sat_f64_s"); break;
                case 3:  buf_cstr(b, "i32.trunc_sat_f64_u"); break;
                case 4:  buf_cstr(b, "i64.trunc_sat_f32_s"); break;
                case 5:  buf_cstr(b, "i64.trunc_sat_f32_u"); break;
                case 6:  buf_cstr(b, "i64.trunc_sat_f64_s"); break;
                case 7:  buf_cstr(b, "i64.trunc_sat_f64_u"); break;
                case 8:  { uint32_t seg = (uint32_t)rd_uleb(r); rd_u8(r);
                           buf_printf(b, "memory.init %u", seg); break; }
                case 9:  buf_printf(b, "data.drop %u", (uint32_t)rd_uleb(r)); break;
                case 10: rd_u8(r); rd_u8(r); buf_cstr(b, "memory.copy"); break;
                case 11: rd_u8(r);            buf_cstr(b, "memory.fill"); break;
                default: buf_printf(b, "(unsupported 0xfc %u)", sub); break;
            }
            return 0;
        }
        default: {
            // Simple opcodes with no immediates: comparisons, arith, conversions,
            // sign-extensions.
            #define X(O, S) case O: buf_cstr(b, S); return 0
            switch (op) {
                X(0x45,"i32.eqz"); X(0x46,"i32.eq"); X(0x47,"i32.ne");
                X(0x48,"i32.lt_s"); X(0x49,"i32.lt_u"); X(0x4a,"i32.gt_s");
                X(0x4b,"i32.gt_u"); X(0x4c,"i32.le_s"); X(0x4d,"i32.le_u");
                X(0x4e,"i32.ge_s"); X(0x4f,"i32.ge_u");
                X(0x50,"i64.eqz"); X(0x51,"i64.eq"); X(0x52,"i64.ne");
                X(0x53,"i64.lt_s"); X(0x54,"i64.lt_u"); X(0x55,"i64.gt_s");
                X(0x56,"i64.gt_u"); X(0x57,"i64.le_s"); X(0x58,"i64.le_u");
                X(0x59,"i64.ge_s"); X(0x5a,"i64.ge_u");
                X(0x5b,"f32.eq"); X(0x5c,"f32.ne"); X(0x5d,"f32.lt");
                X(0x5e,"f32.gt"); X(0x5f,"f32.le"); X(0x60,"f32.ge");
                X(0x61,"f64.eq"); X(0x62,"f64.ne"); X(0x63,"f64.lt");
                X(0x64,"f64.gt"); X(0x65,"f64.le"); X(0x66,"f64.ge");
                X(0x67,"i32.clz"); X(0x68,"i32.ctz"); X(0x69,"i32.popcnt");
                X(0x6a,"i32.add"); X(0x6b,"i32.sub"); X(0x6c,"i32.mul");
                X(0x6d,"i32.div_s"); X(0x6e,"i32.div_u"); X(0x6f,"i32.rem_s");
                X(0x70,"i32.rem_u"); X(0x71,"i32.and"); X(0x72,"i32.or");
                X(0x73,"i32.xor"); X(0x74,"i32.shl"); X(0x75,"i32.shr_s");
                X(0x76,"i32.shr_u"); X(0x77,"i32.rotl"); X(0x78,"i32.rotr");
                X(0x79,"i64.clz"); X(0x7a,"i64.ctz"); X(0x7b,"i64.popcnt");
                X(0x7c,"i64.add"); X(0x7d,"i64.sub"); X(0x7e,"i64.mul");
                X(0x7f,"i64.div_s"); X(0x80,"i64.div_u"); X(0x81,"i64.rem_s");
                X(0x82,"i64.rem_u"); X(0x83,"i64.and"); X(0x84,"i64.or");
                X(0x85,"i64.xor"); X(0x86,"i64.shl"); X(0x87,"i64.shr_s");
                X(0x88,"i64.shr_u"); X(0x89,"i64.rotl"); X(0x8a,"i64.rotr");
                X(0x8b,"f32.abs"); X(0x8c,"f32.neg"); X(0x8d,"f32.ceil");
                X(0x8e,"f32.floor"); X(0x8f,"f32.trunc"); X(0x90,"f32.nearest");
                X(0x91,"f32.sqrt"); X(0x92,"f32.add"); X(0x93,"f32.sub");
                X(0x94,"f32.mul"); X(0x95,"f32.div"); X(0x96,"f32.min");
                X(0x97,"f32.max"); X(0x98,"f32.copysign");
                X(0x99,"f64.abs"); X(0x9a,"f64.neg"); X(0x9b,"f64.ceil");
                X(0x9c,"f64.floor"); X(0x9d,"f64.trunc"); X(0x9e,"f64.nearest");
                X(0x9f,"f64.sqrt"); X(0xa0,"f64.add"); X(0xa1,"f64.sub");
                X(0xa2,"f64.mul"); X(0xa3,"f64.div"); X(0xa4,"f64.min");
                X(0xa5,"f64.max"); X(0xa6,"f64.copysign");
                X(0xa7,"i32.wrap_i64");
                X(0xa8,"i32.trunc_f32_s"); X(0xa9,"i32.trunc_f32_u");
                X(0xaa,"i32.trunc_f64_s"); X(0xab,"i32.trunc_f64_u");
                X(0xac,"i64.extend_i32_s"); X(0xad,"i64.extend_i32_u");
                X(0xae,"i64.trunc_f32_s"); X(0xaf,"i64.trunc_f32_u");
                X(0xb0,"i64.trunc_f64_s"); X(0xb1,"i64.trunc_f64_u");
                X(0xb2,"f32.convert_i32_s"); X(0xb3,"f32.convert_i32_u");
                X(0xb4,"f32.convert_i64_s"); X(0xb5,"f32.convert_i64_u");
                X(0xb6,"f32.demote_f64");
                X(0xb7,"f64.convert_i32_s"); X(0xb8,"f64.convert_i32_u");
                X(0xb9,"f64.convert_i64_s"); X(0xba,"f64.convert_i64_u");
                X(0xbb,"f64.promote_f32");
                X(0xbc,"i32.reinterpret_f32"); X(0xbd,"i64.reinterpret_f64");
                X(0xbe,"f32.reinterpret_i32"); X(0xbf,"f64.reinterpret_i64");
                X(0xc0,"i32.extend8_s"); X(0xc1,"i32.extend16_s");
                X(0xc2,"i64.extend8_s"); X(0xc3,"i64.extend16_s");
                X(0xc4,"i64.extend32_s");
                default: buf_printf(b, "(unknown 0x%02x)", op); return 0;
            }
            #undef X
        }
    }
}

static void decode_code_section(Buf *out, Rd *r, NameMap *fnames,
                                uint32_t func_index_base, int indent) {
    uint32_t nfuncs = (uint32_t)rd_uleb(r);
    for (uint32_t i = 0; i < nfuncs; i++) {
        uint32_t body_size = (uint32_t)rd_uleb(r);
        const uint8_t *body_end = r->p + body_size;
        if (body_end > r->end) body_end = r->end;

        uint32_t fidx = func_index_base + i;
        const char *fname = namemap_get(fnames, fidx);
        buf_indent(out, indent);
        if (fname && fname[0]) buf_printf(out, "(func $%s", fname);
        else                   buf_printf(out, "(func ;; %u", fidx);
        buf_putc(out, '\n');

        // Local groups.
        uint32_t ngroups = (uint32_t)rd_uleb(r);
        for (uint32_t g = 0; g < ngroups; g++) {
            uint32_t cnt = (uint32_t)rd_uleb(r);
            uint8_t  vt  = rd_u8(r);
            buf_indent(out, indent + 2);
            buf_printf(out, "(local");
            for (uint32_t k = 0; k < cnt; k++) buf_printf(out, " %s", vt_name_byte(vt));
            buf_cstr(out, ")\n");
        }

        // Instructions.
        int nest = indent + 2;
        while (r->p < body_end && !r->overflow) {
            // Peek at byte to handle else/end indentation.
            uint8_t pk = *r->p;
            int line_indent = nest;
            if (pk == 0x0b || pk == 0x05) {
                line_indent = nest - 2;
                if (line_indent < indent + 2) line_indent = indent + 2;
            }
            buf_indent(out, line_indent);
            bool ended = false;
            int delta = decode_instr(out, r, fnames, &ended);
            buf_putc(out, '\n');
            if (ended) {
                nest -= 2;
                if (nest < indent + 2) {
                    // Function-terminating end. Stop walking the body.
                    break;
                }
            } else if (pk == 0x05) {
                // else: nest stays the same (just dedent/indent inside if).
                (void)delta;
            } else {
                nest += delta * 2;
            }
        }
        // If body has extra bytes, skip them.
        r->p = body_end;

        buf_indent(out, indent);
        buf_cstr(out, ")\n");
    }
}

string mlir_wasm_binary_to_wat(MLIR_Context *ctx, string bin) {
    Buf out = {0};
    NameMap fnames = {0}, gnames = {0};

    if (bin.size < 8 ||
        memcmp(bin.str, "\x00" "asm" "\x01\x00\x00\x00", 8) != 0) {
        buf_cstr(&out, ";; (not a wasm binary: missing magic+version)\n");
        namemap_free(&fnames); namemap_free(&gnames);
        return buf_to_arena_string(ctx, &out);
    }

    // First pass: locate and parse the "name" custom section (if any) so
    // we can use function names in the second pass.
    {
        Rd r = { (const uint8_t *)bin.str + 8, (const uint8_t *)bin.str + bin.size, false };
        while (r.p < r.end && !r.overflow) {
            uint8_t id = rd_u8(&r);
            uint32_t sz = (uint32_t)rd_uleb(&r);
            if (r.p + sz > r.end) break;
            if (id == 0) {
                Rd s = { r.p, r.p + sz, false };
                RdName nm = rd_name(&s);
                if (nm.size == 4 && memcmp(nm.str, "name", 4) == 0) {
                    scan_name_section(&fnames, &gnames, s.p, (size_t)(s.end - s.p));
                }
            }
            r.p += sz;
        }
    }

    buf_cstr(&out, "(module\n");

    uint32_t n_imported_funcs = 0;

    Rd r = { (const uint8_t *)bin.str + 8, (const uint8_t *)bin.str + bin.size, false };
    while (r.p < r.end && !r.overflow) {
        uint8_t id = rd_u8(&r);
        uint32_t sz = (uint32_t)rd_uleb(&r);
        const uint8_t *sec_end = r.p + sz;
        if (sec_end > r.end) sec_end = r.end;
        Rd s = { r.p, sec_end, false };

        switch (id) {
            case 0: { // SEC_CUSTOM
                RdName nm = rd_name(&s);
                buf_indent(&out, 2);
                if (nm.size == 4 && memcmp(nm.str, "name", 4) == 0) {
                    buf_printf(&out, ";; custom section \"name\" (%u bytes)\n", sz);
                } else if (nm.size && (memcmp(nm.str, "linking", nm.size < 7 ? nm.size : 7) == 0)) {
                    buf_printf(&out, ";; custom section \"linking\" (%u bytes)\n", sz);
                } else if (nm.size >= 6 && memcmp(nm.str, "reloc.", 6) == 0) {
                    buf_printf(&out, ";; custom section \"%.*s\" (%u bytes)\n",
                               (int)nm.size, (const char *)nm.str, sz);
                } else {
                    buf_printf(&out, ";; custom section \"%.*s\" (%u bytes)\n",
                               (int)nm.size, (const char *)nm.str, sz);
                }
                break;
            }
            case 1: { // SEC_TYPE
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint8_t tag = rd_u8(&s);
                    (void)tag; // expect 0x60
                    buf_indent(&out, 2);
                    buf_printf(&out, "(type (;%u;) (func", i);
                    uint32_t np = (uint32_t)rd_uleb(&s);
                    if (np) {
                        buf_cstr(&out, " (param");
                        for (uint32_t k = 0; k < np; k++)
                            buf_printf(&out, " %s", vt_name_byte(rd_u8(&s)));
                        buf_putc(&out, ')');
                    }
                    uint32_t nr = (uint32_t)rd_uleb(&s);
                    if (nr) {
                        buf_cstr(&out, " (result");
                        for (uint32_t k = 0; k < nr; k++)
                            buf_printf(&out, " %s", vt_name_byte(rd_u8(&s)));
                        buf_putc(&out, ')');
                    }
                    buf_cstr(&out, "))\n");
                }
                break;
            }
            case 2: { // SEC_IMPORT
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    RdName mod = rd_name(&s);
                    RdName nm  = rd_name(&s);
                    uint8_t kind = rd_u8(&s);
                    buf_indent(&out, 2);
                    buf_printf(&out, "(import \"%.*s\" \"%.*s\" ",
                               (int)mod.size, (const char *)mod.str,
                               (int)nm.size,  (const char *)nm.str);
                    switch (kind) {
                        case 0: {
                            uint32_t ti = (uint32_t)rd_uleb(&s);
                            const char *fn = namemap_get(&fnames, n_imported_funcs);
                            if (fn && fn[0]) buf_printf(&out, "(func $%s (type %u))", fn, ti);
                            else             buf_printf(&out, "(func (;%u;) (type %u))", n_imported_funcs, ti);
                            n_imported_funcs++;
                            break;
                        }
                        case 1: {
                            uint8_t et = rd_u8(&s);
                            uint8_t fl = rd_u8(&s);
                            uint32_t mn = (uint32_t)rd_uleb(&s);
                            if (fl & 1) {
                                uint32_t mx = (uint32_t)rd_uleb(&s);
                                buf_printf(&out, "(table %u %u %s)", mn, mx, vt_name_byte(et));
                            } else {
                                buf_printf(&out, "(table %u %s)", mn, vt_name_byte(et));
                            }
                            break;
                        }
                        case 2: {
                            uint8_t fl = rd_u8(&s);
                            uint32_t mn = (uint32_t)rd_uleb(&s);
                            if (fl & 1) {
                                uint32_t mx = (uint32_t)rd_uleb(&s);
                                buf_printf(&out, "(memory %u %u)", mn, mx);
                            } else {
                                buf_printf(&out, "(memory %u)", mn);
                            }
                            break;
                        }
                        case 3: {
                            uint8_t vt = rd_u8(&s);
                            uint8_t mu = rd_u8(&s);
                            buf_printf(&out, "(global %s%s%s)",
                                       mu ? "(mut " : "",
                                       vt_name_byte(vt),
                                       mu ? ")" : "");
                            break;
                        }
                        default: buf_printf(&out, "(?)"); break;
                    }
                    buf_cstr(&out, ")\n");
                }
                break;
            }
            case 3: { // SEC_FUNCTION
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t ti = (uint32_t)rd_uleb(&s);
                    uint32_t fidx = n_imported_funcs + i;
                    const char *fn = namemap_get(&fnames, fidx);
                    buf_indent(&out, 2);
                    if (fn && fn[0]) buf_printf(&out, ";; func $%s (type %u)\n", fn, ti);
                    else             buf_printf(&out, ";; func (;%u;) (type %u)\n", fidx, ti);
                }
                break;
            }
            case 4: { // SEC_TABLE
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint8_t et = rd_u8(&s);
                    uint8_t fl = rd_u8(&s);
                    uint32_t mn = (uint32_t)rd_uleb(&s);
                    buf_indent(&out, 2);
                    if (fl & 1) {
                        uint32_t mx = (uint32_t)rd_uleb(&s);
                        buf_printf(&out, "(table (;%u;) %u %u %s)\n", i, mn, mx, vt_name_byte(et));
                    } else {
                        buf_printf(&out, "(table (;%u;) %u %s)\n", i, mn, vt_name_byte(et));
                    }
                }
                break;
            }
            case 5: { // SEC_MEMORY
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint8_t fl = rd_u8(&s);
                    uint32_t mn = (uint32_t)rd_uleb(&s);
                    buf_indent(&out, 2);
                    if (fl & 1) {
                        uint32_t mx = (uint32_t)rd_uleb(&s);
                        buf_printf(&out, "(memory (;%u;) %u %u)\n", i, mn, mx);
                    } else {
                        buf_printf(&out, "(memory (;%u;) %u)\n", i, mn);
                    }
                }
                break;
            }
            case 6: { // SEC_GLOBAL
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint8_t vt = rd_u8(&s);
                    uint8_t mu = rd_u8(&s);
                    buf_indent(&out, 2);
                    buf_printf(&out, "(global (;%u;) %s%s%s ",
                               i,
                               mu ? "(mut " : "",
                               vt_name_byte(vt),
                               mu ? ")" : "");
                    // init expr — decode until end (0x0b).
                    buf_cstr(&out, "(");
                    bool ended;
                    decode_instr(&out, &s, &fnames, &ended);
                    // Skip up to next end (the initializer expression is
                    // terminated by 0x0b).
                    while (!ended && s.p < s.end && !s.overflow) {
                        buf_putc(&out, ' ');
                        decode_instr(&out, &s, &fnames, &ended);
                    }
                    buf_cstr(&out, "))\n");
                }
                break;
            }
            case 7: { // SEC_EXPORT
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    RdName nm = rd_name(&s);
                    uint8_t kind = rd_u8(&s);
                    uint32_t idx = (uint32_t)rd_uleb(&s);
                    const char *kn = kind == 0 ? "func" :
                                     kind == 1 ? "table" :
                                     kind == 2 ? "memory" :
                                     kind == 3 ? "global" : "?";
                    buf_indent(&out, 2);
                    buf_printf(&out, "(export \"%.*s\" (%s %u))\n",
                               (int)nm.size, (const char *)nm.str, kn, idx);
                }
                break;
            }
            case 8: { // SEC_START
                uint32_t idx = (uint32_t)rd_uleb(&s);
                buf_indent(&out, 2);
                buf_printf(&out, "(start %u)\n", idx);
                break;
            }
            case 9: { // SEC_ELEMENT
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t flags = (uint32_t)rd_uleb(&s);
                    buf_indent(&out, 2);
                    buf_printf(&out, "(elem (;%u;) flags=0x%x ", i, flags);
                    if ((flags & 0x03) == 0) {
                        // active, table 0, offset expr
                        buf_cstr(&out, "(offset");
                        bool ended;
                        do {
                            buf_putc(&out, ' ');
                            decode_instr(&out, &s, &fnames, &ended);
                        } while (!ended && s.p < s.end && !s.overflow);
                        buf_cstr(&out, ") func");
                        uint32_t cnt = (uint32_t)rd_uleb(&s);
                        for (uint32_t k = 0; k < cnt; k++) {
                            uint32_t fi = (uint32_t)rd_uleb(&s);
                            fmt_funcref(&out, &fnames, fi);
                        }
                    } else {
                        // Just print bytes count.
                        buf_printf(&out, "...");
                    }
                    buf_cstr(&out, ")\n");
                }
                break;
            }
            case 10: { // SEC_CODE
                decode_code_section(&out, &s, &fnames, n_imported_funcs, 2);
                break;
            }
            case 11: { // SEC_DATA
                uint32_t n = (uint32_t)rd_uleb(&s);
                for (uint32_t i = 0; i < n; i++) {
                    uint32_t flags = (uint32_t)rd_uleb(&s);
                    buf_indent(&out, 2);
                    buf_printf(&out, "(data (;%u;)", i);
                    if ((flags & 1) == 0) {
                        // active with memory 0 (or explicit memidx if flags & 2)
                        if (flags & 2) {
                            uint32_t mi = (uint32_t)rd_uleb(&s);
                            buf_printf(&out, " (memory %u)", mi);
                        }
                        buf_cstr(&out, " (offset");
                        bool ended;
                        do {
                            buf_putc(&out, ' ');
                            decode_instr(&out, &s, &fnames, &ended);
                        } while (!ended && s.p < s.end && !s.overflow);
                        buf_putc(&out, ')');
                    }
                    uint32_t dn = (uint32_t)rd_uleb(&s);
                    buf_putc(&out, ' ');
                    if (s.p + dn > s.end) dn = (uint32_t)(s.end - s.p);
                    print_data_bytes_escaped(&out, s.p, dn);
                    s.p += dn;
                    buf_cstr(&out, ")\n");
                }
                break;
            }
            case 12: { // SEC_DATACOUNT
                uint32_t dc = (uint32_t)rd_uleb(&s);
                buf_indent(&out, 2);
                buf_printf(&out, ";; data count = %u\n", dc);
                break;
            }
            default:
                buf_indent(&out, 2);
                buf_printf(&out, ";; unknown section %u (%u bytes)\n", id, sz);
                break;
        }

        r.p = sec_end;
    }

    buf_cstr(&out, ")\n");

    namemap_free(&fnames);
    namemap_free(&gnames);
    return buf_to_arena_string(ctx, &out);
}

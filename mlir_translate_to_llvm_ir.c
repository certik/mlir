// Native implementation of MLIR_TranslateModuleToLLVMIR.
//
// Walks an `llvm`-dialect builtin.module via the public mlir_api.h
// surface and emits LLVM IR text suitable for `llc`. This file is C,
// linked into both the upstream-backed and native-backed builds (no
// upstream MLIR headers), so the same translation unit supplies the
// agnostic `MLIR_TranslateModuleToLLVMIR` in both binaries.

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
#include "mlir_passes.h"
#include "mlir_llvm_type_layout.h"

// -----------------------------------------------------------------------------
// Growable byte buffer (heap-backed; result is copied into the arena at the
// end).
// -----------------------------------------------------------------------------

typedef struct { char *data; size_t len, cap; } Buf;

static void buf_grow(Buf *b, size_t add) {
    if (b->len + add <= b->cap) return;
    size_t nc = b->cap ? b->cap : 1024;
    while (b->len + add > nc) nc *= 2;
    b->data = (char *)realloc(b->data, nc);
    b->cap = nc;
}
static void buf_putc(Buf *b, char c)            { buf_grow(b, 1); b->data[b->len++] = c; }
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

// -----------------------------------------------------------------------------
// Handle → name maps (linear, function-scoped).
// -----------------------------------------------------------------------------

typedef struct { uintptr_t key; char *val; } Entry;
typedef struct { Entry *e; size_t n, cap; } Map;

static const char *map_get(Map *m, uintptr_t k) {
    for (size_t i = 0; i < m->n; i++) if (m->e[i].key == k) return m->e[i].val;
    return NULL;
}
static void map_put(Map *m, uintptr_t k, char *v) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->e = (Entry *)realloc(m->e, m->cap * sizeof(Entry));
    }
    m->e[m->n].key = k;
    m->e[m->n].val = v;
    m->n++;
}
static void map_free(Map *m) {
    for (size_t i = 0; i < m->n; i++) free(m->e[i].val);
    free(m->e);
    m->e = NULL; m->n = m->cap = 0;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    memcpy(r, s, n);
    return r;
}
static char *fmt_alloc(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    va_list ap2; va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    char *out = (char *)malloc((size_t)n + 1);
    vsnprintf(out, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    return out;
}

// Scalar type tests by kind and width (no type strings).
static bool is_float_w(MLIR_Context *ctx, MLIR_TypeHandle ty, uint32_t width) {
    return MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_FLOAT && MLIR_GetTypeWidth(ctx, ty) == width;
}
static bool is_llvm_ptr(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_POINTER;
}
// The width of an integer type, 0 for any other type.
static uint32_t int_width(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    return MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_INTEGER ? MLIR_GetTypeWidth(ctx, ty) : 0;
}

// Format a floating-point constant as LLVM's hexadecimal form
// (`0x<16 hex digits of the IEEE-754 double>`), which round-trips every bit.
// A decimal `%e` keeps only 7 significant digits, which silently truncated
// every `double` constant to roughly `float` precision. For an f32-typed
// constant LLVM still spells the value as a double, but it must be exactly
// representable as a float, so round it first.
static char *fmt_fp_literal(MLIR_Context *ctx, MLIR_TypeHandle ty, double v) {
    if (is_float_w(ctx, ty, 32)) v = (double)(float)v;
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    return fmt_alloc("0x%016llX", (unsigned long long)bits);
}

// -----------------------------------------------------------------------------
// String / op-name helpers.
// -----------------------------------------------------------------------------

static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && memcmp(s.str, cstr, n) == 0;
}
static bool name_starts_with(string s, const char *p) {
    size_t n = strlen(p);
    return s.size >= n && memcmp(s.str, p, n) == 0;
}

// -----------------------------------------------------------------------------
// Type printing.
// -----------------------------------------------------------------------------
//
// We rely on MLIR_GetTypeString for a textual rendering of MLIR types and
// translate that to LLVM-IR-text form. Keeps the implementation small at
// the cost of being slightly fragile to MLIR's printer changes.

static void print_llvm_type_text(Buf *out, const char *s, size_t n);

static void print_type(Buf *out, MLIR_Context *ctx, MLIR_TypeHandle ty) {
    string s = MLIR_GetTypeString(ctx, ty);
    print_llvm_type_text(out, s.str, s.size);
}

static void print_llvm_type_text(Buf *out, const char *s, size_t n) {
    // Trim leading spaces.
    while (n > 0 && (*s == ' ' || *s == '\t')) { s++; n--; }
    // Trim trailing spaces.
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t')) n--;

    if (n == 0) { buf_cstr(out, "void"); return; }

    // Integer: i<width> (or ui<width> / si<width> from the native printer
    // — LLVM IR is signedness-agnostic so collapse them all to i<width>).
    {
        const char *p = s; size_t pn = n;
        if (pn > 2 && (p[0] == 'u' || p[0] == 's') && p[1] == 'i') { p += 1; pn -= 1; }
        if (p[0] == 'i' && pn > 1) {
            bool all_dig = true;
            for (size_t i = 1; i < pn; i++) if (p[i] < '0' || p[i] > '9') { all_dig = false; break; }
            if (all_dig) { buf_append(out, p, pn); return; }
        }
    }
    // Float keywords
    if (n == 3 && memcmp(s, "f16", 3) == 0) { buf_cstr(out, "half"); return; }
    if (n == 3 && memcmp(s, "f32", 3) == 0) { buf_cstr(out, "float"); return; }
    if (n == 3 && memcmp(s, "f64", 3) == 0) { buf_cstr(out, "double"); return; }

    // !llvm.ptr OR ptr
    if ((n >= 9 && memcmp(s, "!llvm.ptr", 9) == 0 &&
         (n == 9 || s[9] == ' ' || s[9] == '<')) ||
        (n == 3 && memcmp(s, "ptr", 3) == 0)) {
        buf_cstr(out, "ptr");
        return;
    }
    // !llvm.void OR void
    if ((n == 10 && memcmp(s, "!llvm.void", 10) == 0) ||
        (n == 4 && memcmp(s, "void", 4) == 0)) { buf_cstr(out, "void"); return; }
    // !llvm.array<N x T> OR array<N x T>
    {
        const char *body_in = NULL; size_t bn = 0;
        if (n > 12 && memcmp(s, "!llvm.array<", 12) == 0 && s[n-1] == '>') {
            body_in = s + 12; bn = n - 13;
        } else if (n > 6 && memcmp(s, "array<", 6) == 0 && s[n-1] == '>') {
            body_in = s + 6; bn = n - 7;
        }
        if (body_in) {
            const char *sp = NULL;
            for (size_t i = 0; i + 2 < bn; i++) {
                if (body_in[i] == ' ' && body_in[i+1] == 'x' && body_in[i+2] == ' ') { sp = body_in + i; break; }
            }
            if (sp) {
                size_t cn = (size_t)(sp - body_in);
                buf_putc(out, '[');
                buf_append(out, body_in, cn);
                buf_cstr(out, " x ");
                print_llvm_type_text(out, sp + 3, bn - cn - 3);
                buf_putc(out, ']');
                return;
            }
        }
    }
    // !llvm.struct<...>  OR  struct<...>  (nested form omits the !llvm. prefix)
    {
        const char *body_in = NULL; size_t bn = 0;
        if (n > 13 && memcmp(s, "!llvm.struct<", 13) == 0 && s[n-1] == '>') {
            body_in = s + 13; bn = n - 14;
        } else if (n > 7 && memcmp(s, "struct<", 7) == 0 && s[n-1] == '>') {
            body_in = s + 7; bn = n - 8;
        }
        if (body_in) {
            if (bn > 0 && body_in[0] == '"') {
                size_t i = 1;
                while (i < bn && body_in[i] != '"') i++;
                buf_putc(out, '%');
                buf_append(out, body_in + 1, i - 1);
                return;
            }
            if (bn > 0 && body_in[0] == '(' && body_in[bn-1] == ')') {
                buf_cstr(out, "{ ");
                const char *p = body_in + 1; size_t left = bn - 2;
                bool first = true;
                while (left > 0) {
                    int depth = 0; size_t j = 0;
                    while (j < left) {
                        char c = p[j];
                        if (c == '<' || c == '(' || c == '[' || c == '{') depth++;
                        else if (c == '>' || c == ')' || c == ']' || c == '}') depth--;
                        else if (c == ',' && depth == 0) break;
                        j++;
                    }
                    if (!first) buf_cstr(out, ", ");
                    print_llvm_type_text(out, p, j);
                    first = false;
                    if (j < left && p[j] == ',') { j++; while (j < left && p[j] == ' ') j++; }
                    p += j; left -= j;
                }
                buf_cstr(out, " }");
                return;
            }
        }
    }
    // !llvm.func<RT (params)>
    if (n > 11 && memcmp(s, "!llvm.func<", 11) == 0 && s[n-1] == '>') {
        const char *body = s + 11; size_t bn = n - 12;
        // Pass through as-is; MLIR's syntax already matches LLVM's "RT (P1, P2)" form,
        // except inner !llvm.* sub-types — but we don't currently use nested function types.
        buf_append(out, body, bn);
        return;
    }

    // Fallback: pass through unchanged.
    buf_append(out, s, n);
}

// Best-effort alignment for store/load. Matches what the upstream printer
// emits for the tinyc surface.
static unsigned type_align(MLIR_Context *ctx, MLIR_TypeHandle ty) {
    uint32_t w = int_width(ctx, ty);
    if (w == 1 || w == 8) return 1;
    if (w == 16 || w == 32 || w == 64) return w / 8;
    if (is_float_w(ctx, ty, 16)) return 2;
    if (is_float_w(ctx, ty, 32)) return 4;
    return 8;
}

// -----------------------------------------------------------------------------
// Per-function naming + emit.
// -----------------------------------------------------------------------------

typedef struct {
    MLIR_Context *ctx;
    Buf *out;
    Map vmap;   // value handle -> "%v3" or "%arg0" or "<inline literal>"
    Map bmap;   // block handle -> "bb1"
    int next_v;
    int next_b;
} FnCtx;

static const char *vname(FnCtx *F, MLIR_ValueHandle v) {
    const char *s = map_get(&F->vmap, v);
    return s ? s : "<unbound>";
}
static const char *bname(FnCtx *F, MLIR_BlockHandle b) {
    const char *s = map_get(&F->bmap, b);
    return s ? s : "<unbound>";
}

// Emit a use of `v`: either a constant inlined as a literal, or a reference
// to its SSA name.
static void emit_use(FnCtx *F, MLIR_ValueHandle v) {
    const char *n = map_get(&F->vmap, v);
    buf_cstr(F->out, n ? n : "<unbound>");
}

// If `v` is the result of a constant-like op (llvm.mlir.constant /
// llvm.mlir.zero / llvm.mlir.addressof / llvm.mlir.null / llvm.mlir.undef /
// llvm.mlir.poison), assign it an inline literal in `F->vmap` instead of an
// SSA name.
static char *inline_literal_for(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_TypeHandle vty) {
    MLIR_OpType ty = MLIR_GetOpType(ctx, op);
    bool is_undef = (ty == OP_TYPE_LLVM_MLIR_UNDEF);
    bool is_poison = (ty == OP_TYPE_LLVM_MLIR_POISON);
    if (ty == OP_TYPE_LLVM_MLIR_CONSTANT) {
        // Value is the "value" attribute. It's printed as e.g. "3 : i32" or
        // "1.5e+00 : f32".
        size_t na = MLIR_GetOpNumAttributes(ctx, op);
        for (size_t i = 0; i < na; i++) {
            MLIR_AttributeHandle a = MLIR_GetOpAttribute(ctx, op, i);
            string an = MLIR_GetAttrKeyName(ctx, MLIR_GetOpAttributeKey(ctx, op, i));
            if (name_eq(an, "value")) {
                MLIR_AttrKind k = MLIR_GetAttributeKind(ctx, a);
                if (k == MLIR_ATTR_KIND_INTEGER) {
                    int64_t iv = MLIR_GetAttributeInteger(ctx, a);
                    return fmt_alloc("%lld", (long long)iv);
                } else if (k == MLIR_ATTR_KIND_FLOAT) {
                    return fmt_fp_literal(ctx, vty, MLIR_GetAttributeFloat(ctx, a));
                } else if (k == MLIR_ATTR_KIND_BOOL) {
                    bool bv = MLIR_GetAttributeBool(ctx, a);
                    return xstrdup(bv ? "true" : "false");
                } else {
                    string s = MLIR_GetAttributeAsString(ctx, a);
                    // s is "<value> : <type>"; strip the type suffix.
                    size_t end = s.size;
                    for (size_t j = 0; j + 2 < s.size; j++) {
                        if (s.str[j] == ' ' && s.str[j+1] == ':' && s.str[j+2] == ' ') {
                            end = j; break;
                        }
                    }
                    char *r = (char *)malloc(end + 1);
                    memcpy(r, s.str, end); r[end] = 0;
                    return r;
                }
            }
        }
        (void)vty;
        return xstrdup("0");
    }
    if (ty == OP_TYPE_LLVM_MLIR_ZERO || is_undef || is_poison) {
        // Type-driven: ptr -> null, anything aggregate -> zeroinitializer,
        // scalar -> 0 / 0.0.
        MLIR_TypeKind vk = MLIR_GetTypeKind(ctx, vty);
        if (vk == MLIR_TYPE_KIND_LLVM_POINTER) {
            return xstrdup(is_undef ? "undef" : is_poison ? "poison" : "null");
        }
        if (is_undef) return xstrdup("undef");
        if (is_poison) return xstrdup("poison");
        // zero
        if (vk == MLIR_TYPE_KIND_INTEGER || vk == MLIR_TYPE_KIND_INDEX) return xstrdup("0");
        if (is_float_w(ctx, vty, 32) || is_float_w(ctx, vty, 64))
            return xstrdup("0.000000e+00");
        return xstrdup("zeroinitializer");
    }
    if (ty == OP_TYPE_LLVM_MLIR_NULL) return xstrdup("null");
    if (ty == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
        size_t na = MLIR_GetOpNumAttributes(ctx, op);
        for (size_t i = 0; i < na; i++) {
            MLIR_AttributeHandle a = MLIR_GetOpAttribute(ctx, op, i);
            string an = MLIR_GetAttrKeyName(ctx, MLIR_GetOpAttributeKey(ctx, op, i));
            if (name_eq(an, "global_name") || name_eq(an, "function_name") ||
                name_eq(an, "symbol")) {
                string s = MLIR_GetAttributeAsString(ctx, a);
                // s is "@<name>" already
                return fmt_alloc("%.*s", (int)s.size, s.str);
            }
        }
    }
    return NULL;
}

// Pre-pass: assign names to all values & blocks in the function body.
static void assign_names(FnCtx *F, MLIR_RegionHandle body) {
    size_t nb = MLIR_GetRegionNumBlocks(F->ctx, body);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(F->ctx, body, bi);
        // Block name: "entry" for block 0, "bbN" for the rest.
        char *bn = bi == 0 ? xstrdup("entry") : fmt_alloc("bb%d", F->next_b++);
        map_put(&F->bmap, blk, bn);
        // Block arg names: "%aN" — for entry, these are also the function
        // parameters and are referred to as %0, %1, ...
        size_t na = MLIR_GetBlockNumArgs(F->ctx, blk);
        for (size_t ai = 0; ai < na; ai++) {
            MLIR_ValueHandle a = MLIR_GetBlockArg(F->ctx, blk, ai);
            char *vn = bi == 0 ? fmt_alloc("%%a%zu", ai) : fmt_alloc("%%v%d", F->next_v++);
            map_put(&F->vmap, a, vn);
        }
        // Op results.
        size_t nops = MLIR_GetBlockNumOps(F->ctx, blk);
        for (size_t oi = 0; oi < nops; oi++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(F->ctx, blk, oi);
            size_t nres = MLIR_GetOpNumResults(F->ctx, op);
            for (size_t ri = 0; ri < nres; ri++) {
                MLIR_ValueHandle r = MLIR_GetOpResult(F->ctx, op, ri);
                MLIR_TypeHandle rty = MLIR_GetValueType(F->ctx, r);
                char *lit = inline_literal_for(F->ctx, op, rty);
                if (lit) map_put(&F->vmap, r, lit);
                else     map_put(&F->vmap, r, fmt_alloc("%%v%d", F->next_v++));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Op emission.
// -----------------------------------------------------------------------------

static MLIR_AttributeHandle find_attr(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttrKey key) {
    size_t na = MLIR_GetOpNumAttributes(ctx, op);
    for (size_t i = 0; i < na; i++) {
        if (MLIR_GetOpAttributeKey(ctx, op, i) == key) return MLIR_GetOpAttribute(ctx, op, i);
    }
    return MLIR_INVALID_HANDLE;
}

static void emit_call(FnCtx *F, MLIR_OpHandle op) {
    Buf *out = F->out;
    size_t nres = MLIR_GetOpNumResults(F->ctx, op);
    MLIR_TypeHandle rty = nres ? MLIR_GetOpResultType(F->ctx, op, 0) : MLIR_INVALID_HANDLE;
    bool is_void = (nres == 0);

    // Find callee (symbol) attr — "callee" for direct, none for indirect (the
    // callee value is then the first operand).
    MLIR_AttributeHandle callee = find_attr(F->ctx, op, MLIR_KEY_CALLEE);
    MLIR_AttributeHandle var_callee_type = find_attr(F->ctx, op, MLIR_KEY_VAR_CALLEE_TYPE);
    bool indirect = (callee == MLIR_INVALID_HANDLE);

    // Operands.
    size_t nops = MLIR_GetOpNumOperands(F->ctx, op);
    size_t arg_start = indirect ? 1 : 0;

    buf_cstr(out, "  ");
    if (!is_void) { emit_use(F, MLIR_GetOpResult(F->ctx, op, 0)); buf_cstr(out, " = "); }
    buf_cstr(out, "call ");

    // Vararg form: emit the full function signature `<ret> (<arg-types>, ...)`.
    // For non-variadic indirect calls, just the return type is sufficient
    // (LLVM IR allows `call <ret> %fnptr(...)` when the function is not
    // variadic). Emitting the variadic form unconditionally for indirect
    // calls produces calls with the wrong signature on targets like wasm32
    // where variadic and fixed-arity ABIs differ.
    if (var_callee_type != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(F->ctx, var_callee_type);
        bool is_vararg = MLIR_GetTypeFunctionIsVarArg(F->ctx, fty);
        if (is_vararg) {
            size_t ni = MLIR_GetTypeFunctionNumInputs(F->ctx, fty);
            size_t no = MLIR_GetTypeFunctionNumResults(F->ctx, fty);
            if (no == 0) buf_cstr(out, "void");
            else         print_type(out, F->ctx, MLIR_GetTypeFunctionResult(F->ctx, fty, 0));
            buf_cstr(out, " (");
            for (size_t i = 0; i < ni; i++) {
                if (i) buf_cstr(out, ", ");
                print_type(out, F->ctx, MLIR_GetTypeFunctionInput(F->ctx, fty, i));
            }
            buf_cstr(out, ", ...)");
        } else {
            if (is_void) buf_cstr(out, "void");
            else         print_type(out, F->ctx, rty);
        }
    } else {
        if (is_void) buf_cstr(out, "void");
        else         print_type(out, F->ctx, rty);
    }
    buf_putc(out, ' ');

    if (indirect) emit_use(F, MLIR_GetOpOperand(F->ctx, op, 0));
    else {
        string s = MLIR_GetAttributeAsString(F->ctx, callee);
        buf_str(out, s);
    }
    buf_putc(out, '(');
    for (size_t i = arg_start; i < nops; i++) {
        if (i > arg_start) buf_cstr(out, ", ");
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, i);
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, a));
        buf_putc(out, ' ');
        emit_use(F, a);
    }
    buf_cstr(out, ")\n");
}

static void emit_phis(FnCtx *F, MLIR_BlockHandle blk,
                      MLIR_OpHandle *all_ops, size_t all_n);

// Emit phi nodes at the start of `blk` for each block argument, gathering
// values from all predecessor terminators.
static void emit_phis(FnCtx *F, MLIR_BlockHandle blk,
                      MLIR_OpHandle *all_ops, size_t all_n) {
    size_t na = MLIR_GetBlockNumArgs(F->ctx, blk);
    if (na == 0) return;
    for (size_t ai = 0; ai < na; ai++) {
        MLIR_ValueHandle arg = MLIR_GetBlockArg(F->ctx, blk, ai);
        MLIR_TypeHandle ty = MLIR_GetValueType(F->ctx, arg);
        buf_cstr(F->out, "  ");
        emit_use(F, arg);
        buf_cstr(F->out, " = phi ");
        print_type(F->out, F->ctx, ty);
        buf_putc(F->out, ' ');
        bool first = true;
        for (size_t i = 0; i < all_n; i++) {
            MLIR_OpHandle term = all_ops[i];
            size_t nsucc = MLIR_GetOpNumSuccessors(F->ctx, term);
            for (size_t s = 0; s < nsucc; s++) {
                MLIR_BlockHandle dst = MLIR_GetOpSuccessor(F->ctx, term, s);
                if (dst != blk) continue;
                MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(F->ctx, term, s, ai);
                MLIR_BlockHandle src = MLIR_GetOpParentBlock(F->ctx, term);
                if (!first) buf_cstr(F->out, ", ");
                first = false;
                buf_cstr(F->out, "[ ");
                emit_use(F, v);
                buf_cstr(F->out, ", %");
                buf_cstr(F->out, bname(F, src));
                buf_cstr(F->out, " ]");
            }
        }
        if (first) {
            // No predecessors found — emit "[ undef, %entry ]" as a placeholder
            // so the IR is at least parseable.
            buf_cstr(F->out, "[ undef, %entry ]");
        }
        buf_putc(F->out, '\n');
    }
}

// Returns true if the op is a constant-like producer that we inline at uses.
static bool is_inline_constant(MLIR_OpHandle op, MLIR_OpType ty) {
    switch (ty) {
    case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ZERO:
    case OP_TYPE_LLVM_MLIR_UNDEF:
    case OP_TYPE_LLVM_MLIR_NULL:
    case OP_TYPE_LLVM_MLIR_ADDRESSOF:
    case OP_TYPE_LLVM_MLIR_POISON:
        return true;
    default:
        return false;
    }
}

// Map of llvm.* arith ops to LLVM IR opcodes (1 result, 2 operands,
// printed as "%r = OP T a, b\n").
static const char *llvm_binop_opcode(MLIR_OpType ty) {
    switch (ty) {
    case OP_TYPE_LLVM_ADD:    return "add";
    case OP_TYPE_LLVM_SUB:    return "sub";
    case OP_TYPE_LLVM_MUL:    return "mul";
    case OP_TYPE_LLVM_SDIV:   return "sdiv";
    case OP_TYPE_LLVM_UDIV:   return "udiv";
    case OP_TYPE_LLVM_SREM:   return "srem";
    case OP_TYPE_LLVM_UREM:   return "urem";
    case OP_TYPE_LLVM_AND:    return "and";
    case OP_TYPE_LLVM_OR:     return "or";
    case OP_TYPE_LLVM_XOR:    return "xor";
    case OP_TYPE_LLVM_SHL:    return "shl";
    case OP_TYPE_LLVM_LSHR:   return "lshr";
    case OP_TYPE_LLVM_ASHR:   return "ashr";
    case OP_TYPE_LLVM_FADD:   return "fadd";
    case OP_TYPE_LLVM_FSUB:   return "fsub";
    case OP_TYPE_LLVM_FMUL:   return "fmul";
    case OP_TYPE_LLVM_FDIV:   return "fdiv";
    case OP_TYPE_LLVM_FREM:   return "frem";
    default: return NULL;
    }
}

static const char *llvm_cast_opcode(MLIR_OpHandle op, MLIR_OpType ty) {
    switch (ty) {
    case OP_TYPE_LLVM_SEXT:      return "sext";
    case OP_TYPE_LLVM_ZEXT:      return "zext";
    case OP_TYPE_LLVM_TRUNC:     return "trunc";
    case OP_TYPE_LLVM_FPEXT:     return "fpext";
    case OP_TYPE_LLVM_FPTRUNC:   return "fptrunc";
    case OP_TYPE_LLVM_SITOFP:    return "sitofp";
    case OP_TYPE_LLVM_UITOFP:    return "uitofp";
    case OP_TYPE_LLVM_FPTOSI:    return "fptosi";
    case OP_TYPE_LLVM_FPTOUI:    return "fptoui";
    case OP_TYPE_LLVM_BITCAST:   return "bitcast";
    case OP_TYPE_LLVM_PTRTOINT:  return "ptrtoint";
    case OP_TYPE_LLVM_INTTOPTR:  return "inttoptr";
    case OP_TYPE_LLVM_ADDRSPACECAST: return "addrspacecast";
    default: return NULL;
    }
}

static void emit_op(FnCtx *F, MLIR_OpHandle op) {
    Buf *out = F->out;
    // The op type is derived from the op name by MLIR_CreateOp, so the ops
    // with an enumerator are selected by one switch; the rest (intrinsics,
    // llvm.unreachable) are matched by name after it.
    MLIR_OpType ty = MLIR_GetOpType(F->ctx, op);

    if (is_inline_constant(op, ty)) return; // inlined at uses

    const char *bop;
    if ((bop = llvm_binop_opcode(ty)) != NULL) {
        // %r = OP T a, b
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = ");
        buf_cstr(out, bop);
        buf_putc(out, ' ');
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(F->ctx, op, 1);
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, ", ");
        emit_use(F, b);
        buf_putc(out, '\n');
        return;
    }
    if (ty == OP_TYPE_LLVM_FNEG) {
        // %r = fneg T a
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = fneg ");
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 0);
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_putc(out, '\n');
        return;
    }

    const char *cop;
    if ((cop = llvm_cast_opcode(op, ty)) != NULL) {
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = ");
        buf_cstr(out, cop);
        buf_putc(out, ' ');
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 0);
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, " to ");
        print_type(out, F->ctx, MLIR_GetOpResultType(F->ctx, op, 0));
        buf_putc(out, '\n');
        return;
    }

    switch (ty) {
    case OP_TYPE_LLVM_ICMP: case OP_TYPE_LLVM_FCMP: {
        // %r = icmp pred T a, b
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = ");
        buf_cstr(out, ty == OP_TYPE_LLVM_ICMP ? "icmp " : "fcmp ");
        MLIR_AttributeHandle pa = find_attr(F->ctx, op, MLIR_KEY_PREDICATE);
        const char *predtbl_i[] = {"eq","ne","slt","sle","sgt","sge","ult","ule","ugt","uge"};
        const char *predtbl_f[] = {
            "false","oeq","ogt","oge","olt","ole","one","ord",
            "ueq","ugt","uge","ult","ule","une","uno","true"
        };
        int pi = 0;
        if (pa != MLIR_INVALID_HANDLE) pi = (int)MLIR_GetAttributeInteger(F->ctx, pa);
        if (ty == OP_TYPE_LLVM_ICMP) {
            if (pi >= 0 && pi < 10) buf_cstr(out, predtbl_i[pi]); else buf_cstr(out, "eq");
        } else {
            if (pi >= 0 && pi < 16) buf_cstr(out, predtbl_f[pi]); else buf_cstr(out, "oeq");
        }
        buf_putc(out, ' ');
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(F->ctx, op, 1);
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, ", ");
        emit_use(F, b);
        buf_putc(out, '\n');
        return;
    }

    case OP_TYPE_LLVM_SELECT: {
        // %r = select i1 c, T a, T b
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = select i1 ");
        emit_use(F, MLIR_GetOpOperand(F->ctx, op, 0));
        buf_cstr(out, ", ");
        MLIR_ValueHandle a = MLIR_GetOpOperand(F->ctx, op, 1);
        MLIR_ValueHandle b = MLIR_GetOpOperand(F->ctx, op, 2);
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, ", ");
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, b));
        buf_putc(out, ' ');
        emit_use(F, b);
        buf_putc(out, '\n');
        return;
    }

    case OP_TYPE_LLVM_ALLOCA: {
        // %r = alloca T, i64 N, align A
        // The alloca element type is in the "elem_type" attribute; the
        // operand is the count (i64 typically).
        MLIR_AttributeHandle ea = find_attr(F->ctx, op, MLIR_KEY_ELEM_TYPE);
        MLIR_TypeHandle et = MLIR_GetAttributeTypeValue(F->ctx, ea);
        MLIR_ValueHandle cnt = MLIR_GetOpOperand(F->ctx, op, 0);
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = alloca ");
        print_type(out, F->ctx, et);
        buf_cstr(out, ", ");
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, cnt));
        buf_putc(out, ' ');
        emit_use(F, cnt);
        buf_printf(out, ", align %u\n", type_align(F->ctx, et));
        return;
    }

    case OP_TYPE_LLVM_STORE: {
        // store T val, ptr p, align A
        MLIR_ValueHandle val = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(F->ctx, op, 1);
        buf_cstr(out, "  store ");
        print_type(out, F->ctx, MLIR_GetValueType(F->ctx, val));
        buf_putc(out, ' ');
        emit_use(F, val);
        buf_cstr(out, ", ptr ");
        emit_use(F, ptr);
        buf_printf(out, ", align %u\n", type_align(F->ctx, MLIR_GetValueType(F->ctx, val)));
        return;
    }
    case OP_TYPE_LLVM_LOAD: {
        // %r = load T, ptr p, align A
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_TypeHandle rty = MLIR_GetOpResultType(F->ctx, op, 0);
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = load ");
        print_type(out, F->ctx, rty);
        buf_cstr(out, ", ptr ");
        emit_use(F, ptr);
        buf_printf(out, ", align %u\n", type_align(F->ctx, rty));
        return;
    }

    case OP_TYPE_LLVM_GEP: {
        // %r = getelementptr ELEM_TY, ptr base, INDICES
        // MLIR form: llvm.getelementptr %p[0, 1, %v] : ... -> !llvm.ptr, ELEM_TY
        // The constant indices come from the "rawConstantIndices" attribute
        // (DenseI32ArrayAttr); -2147483648 means "use next operand".
        MLIR_AttributeHandle eta = find_attr(F->ctx, op, MLIR_KEY_ELEM_TYPE);
        MLIR_TypeHandle elem = MLIR_GetAttributeTypeValue(F->ctx, eta);
        MLIR_AttributeHandle ria = find_attr(F->ctx, op, MLIR_KEY_RAW_CONSTANT_INDICES);
        MLIR_ValueHandle base = MLIR_GetOpOperand(F->ctx, op, 0);
        size_t nops = MLIR_GetOpNumOperands(F->ctx, op);
        size_t next_op = 1;

        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_cstr(out, " = getelementptr ");
        print_type(out, F->ctx, elem);
        buf_cstr(out, ", ptr ");
        emit_use(F, base);

        if (ria != MLIR_INVALID_HANDLE) {
            // A dense i32 array; 0x80000000 marks a dynamic index.
            size_t n_ria = MLIR_GetAttributeDenseSize(F->ctx, ria);
            for (size_t i = 0; i < n_ria; i++) {
                long long v = (long long)MLIR_GetAttributeDenseElement(F->ctx, ria, i);
                if (v == (long long)(int32_t)0x80000000) {
                    // Dynamic — pull from operands.
                    MLIR_ValueHandle dv = MLIR_GetOpOperand(F->ctx, op, next_op++);
                    buf_cstr(out, ", ");
                    print_type(out, F->ctx, MLIR_GetValueType(F->ctx, dv));
                    buf_putc(out, ' ');
                    emit_use(F, dv);
                } else {
                    buf_printf(out, ", i32 %lld", v);
                }
            }
        }
        // Any remaining operands are dynamic indices that weren't covered by
        // the constant-indices attr.
        for (size_t j = next_op; j < nops; j++) {
            MLIR_ValueHandle dv = MLIR_GetOpOperand(F->ctx, op, j);
            buf_cstr(out, ", ");
            print_type(out, F->ctx, MLIR_GetValueType(F->ctx, dv));
            buf_putc(out, ' ');
            emit_use(F, dv);
        }
        buf_putc(out, '\n');
        return;
    }

    case OP_TYPE_LLVM_CALL: emit_call(F, op); return;

    case OP_TYPE_LLVM_RETURN: {
        if (MLIR_GetOpNumOperands(F->ctx, op) == 0) buf_cstr(out, "  ret void\n");
        else {
            MLIR_ValueHandle v = MLIR_GetOpOperand(F->ctx, op, 0);
            buf_cstr(out, "  ret ");
            print_type(out, F->ctx, MLIR_GetValueType(F->ctx, v));
            buf_putc(out, ' ');
            emit_use(F, v);
            buf_putc(out, '\n');
        }
        return;
    }

    case OP_TYPE_LLVM_BR: {
        MLIR_BlockHandle dst = MLIR_GetOpSuccessor(F->ctx, op, 0);
        buf_printf(out, "  br label %%%s\n", bname(F, dst));
        return;
    }
    case OP_TYPE_LLVM_COND_BR: {
        MLIR_ValueHandle c = MLIR_GetOpOperand(F->ctx, op, 0);
        MLIR_BlockHandle t = MLIR_GetOpSuccessor(F->ctx, op, 0);
        MLIR_BlockHandle f = MLIR_GetOpSuccessor(F->ctx, op, 1);
        buf_cstr(out, "  br i1 ");
        emit_use(F, c);
        buf_printf(out, ", label %%%s, label %%%s\n", bname(F, t), bname(F, f));
        return;
    }
    case OP_TYPE_LLVM_UNREACHABLE: {
        buf_cstr(out, "  unreachable\n");
        return;
    }
    case OP_TYPE_LLVM_INTR_STACKSAVE: case OP_TYPE_LLVM_INTR_STACKRESTORE: {
        // Skip for now — tinyc doesn't rely on these.
        return;
    }
    case OP_TYPE_LLVM_INTR_VASTART: {
        buf_cstr(out, "  call void @llvm.va_start.p0(ptr ");
        emit_use(F, MLIR_GetOpOperand(F->ctx, op, 0));
        buf_cstr(out, ")\n");
        return;
    }
    case OP_TYPE_LLVM_INTR_VAEND: {
        buf_cstr(out, "  call void @llvm.va_end.p0(ptr ");
        emit_use(F, MLIR_GetOpOperand(F->ctx, op, 0));
        buf_cstr(out, ")\n");
        return;
    }
    case OP_TYPE_LLVM_INTR_VACOPY: {
        buf_cstr(out, "  call void @llvm.va_copy.p0(ptr ");
        emit_use(F, MLIR_GetOpOperand(F->ctx, op, 0));
        buf_cstr(out, ", ptr ");
        emit_use(F, MLIR_GetOpOperand(F->ctx, op, 1));
        buf_cstr(out, ")\n");
        return;
    }
    case OP_TYPE_LLVM_INTR_WASM_MEMORY_SIZE: {
        if (MLIR_GetOpNumResults(F->ctx, op) == 1) {
            buf_cstr(out, "  ");
            emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
            buf_cstr(out, " = load i32, ptr @__wasm_mem_pages, align 4\n");
        }
        return;
    }
    case OP_TYPE_LLVM_INTR_WASM_MEMORY_GROW: {
        if (MLIR_GetOpNumResults(F->ctx, op) != 1) return;
        char *old = fmt_alloc("%%v%d", F->next_v++);
        char *newv = fmt_alloc("%%v%d", F->next_v++);
        char *new64 = fmt_alloc("%%v%d", F->next_v++);
        char *ok = fmt_alloc("%%v%d", F->next_v++);
        char *stored = fmt_alloc("%%v%d", F->next_v++);
        buf_printf(out, "  %s = load i32, ptr @__wasm_mem_pages, align 4\n", old);
        buf_printf(out, "  %s = add i32 %s, ", newv, old);
        emit_use(F, MLIR_GetOpOperand(F->ctx, op, 0));
        buf_putc(out, '\n');
        buf_printf(out, "  %s = zext i32 %s to i64\n", new64, newv);
        buf_printf(out, "  %s = icmp ule i64 %s, 65536\n", ok, new64);
        buf_printf(out, "  %s = select i1 %s, i32 %s, i32 %s\n", stored, ok, newv, old);
        buf_cstr(out, "  store i32 ");
        buf_cstr(out, stored);
        buf_cstr(out, ", ptr @__wasm_mem_pages, align 4\n");
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(F->ctx, op, 0));
        buf_printf(out, " = select i1 %s, i32 %s, i32 -1\n", ok, old);
        return;
    }
    default: break;
    }

    string opn = MLIR_GetOpTypeName(F->ctx, ty);
    // Float math intrinsics: llvm.intr.sqrt, and llvm.intr.sin / cos / fabs /
    // ... from parsed or upstream-lowered IR, which are dynamic op types. The
    // call is named after the op, so only those compare the name.
    if (ty == OP_TYPE_LLVM_INTR_SQRT ||
        ((unsigned)ty >= (unsigned)OP_TYPE_COUNT && name_starts_with(opn, "llvm.intr."))) {
        // Generic shape: %r = call T @llvm.NAME.f32/f64(T %x[, T %y, ...])
        const char *iname = opn.str + 10;
        size_t in = opn.size - 10;
        bool has_res = MLIR_GetOpNumResults(F->ctx, op) > 0;
        MLIR_TypeHandle rty = has_res ? MLIR_GetOpResultType(F->ctx, op, 0) :
                              (MLIR_GetOpNumOperands(F->ctx, op) > 0 ? MLIR_GetValueType(F->ctx, MLIR_GetOpOperand(F->ctx, op, 0)) : MLIR_INVALID_HANDLE);
        // Pick suffix from the first operand/result type (f32/f64/etc.)
        const char *suffix = "f64";
        if (is_float_w(F->ctx, rty, 32)) suffix = "f32";
        else if (is_float_w(F->ctx, rty, 16)) suffix = "f16";
        buf_cstr(out, "  ");
        if (has_res) { emit_use(F, MLIR_GetOpResult(F->ctx, op, 0)); buf_cstr(out, " = "); }
        buf_cstr(out, "call ");
        if (has_res) print_type(out, F->ctx, rty); else buf_cstr(out, "void");
        buf_cstr(out, " @llvm.");
        buf_append(out, iname, in);
        buf_putc(out, '.');
        buf_cstr(out, suffix);
        buf_putc(out, '(');
        size_t no = MLIR_GetOpNumOperands(F->ctx, op);
        for (size_t i = 0; i < no; i++) {
            if (i) buf_cstr(out, ", ");
            MLIR_ValueHandle v = MLIR_GetOpOperand(F->ctx, op, i);
            print_type(out, F->ctx, MLIR_GetValueType(F->ctx, v));
            buf_putc(out, ' ');
            emit_use(F, v);
        }
        buf_cstr(out, ")\n");
        return;
    }

    // Fallback: emit as a comment so we can spot misses.
    buf_cstr(out, "  ; UNHANDLED ");
    buf_str(out, opn);
    buf_putc(out, '\n');
}

// -----------------------------------------------------------------------------
// Function emission.
// -----------------------------------------------------------------------------

static void collect_all_ops(MLIR_Context *ctx, MLIR_RegionHandle body, MLIR_OpHandle **out, size_t *n, size_t *cap) {
    size_t nb = MLIR_GetRegionNumBlocks(ctx, body);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, body, bi);
        size_t nops = MLIR_GetBlockNumOps(ctx, blk);
        for (size_t oi = 0; oi < nops; oi++) {
            if (*n == *cap) {
                *cap = *cap ? *cap * 2 : 64;
                *out = (MLIR_OpHandle *)realloc(*out, *cap * sizeof(MLIR_OpHandle));
            }
            (*out)[(*n)++] = MLIR_GetBlockOp(ctx, blk, oi);
        }
    }
}

// Emit an LLVM IR string literal with quotes and hex escapes if needed.
static void emit_attr_quoted(Buf *out, string s) {
    buf_putc(out, '"');
    for (size_t i = 0; i < s.size; i++) {
        unsigned char c = (unsigned char)s.str[i];
        if (c == '"' || c == '\\' || c < 0x20 || c >= 0x7f)
            buf_printf(out, "\\%02X", c);
        else
            buf_putc(out, (char)c);
    }
    buf_putc(out, '"');
}

static bool has_nonempty_passthrough(MLIR_Context *ctx, MLIR_OpHandle fn) {
    MLIR_AttributeHandle pt = find_attr(ctx, fn, MLIR_KEY_PASSTHROUGH);
    return pt != MLIR_INVALID_HANDLE &&
           MLIR_GetAttributeKind(ctx, pt) == MLIR_ATTR_KIND_ARRAY &&
           MLIR_GetAttributeArraySize(ctx, pt) > 0;
}

static void emit_passthrough_group(MLIR_Context *ctx, Buf *out, MLIR_OpHandle fn, size_t group_id) {
    MLIR_AttributeHandle pt = find_attr(ctx, fn, MLIR_KEY_PASSTHROUGH);
    if (pt == MLIR_INVALID_HANDLE || MLIR_GetAttributeKind(ctx, pt) != MLIR_ATTR_KIND_ARRAY)
        return;
    size_t n = MLIR_GetAttributeArraySize(ctx, pt);
    buf_printf(out, "attributes #%zu = { ", group_id);
    for (size_t i = 0; i < n; i++) {
        MLIR_AttributeHandle e = MLIR_GetAttributeArrayElement(ctx, pt, i);
        MLIR_AttrKind k = MLIR_GetAttributeKind(ctx, e);
        if (k == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(ctx, e);
            emit_attr_quoted(out, s);
            buf_putc(out, ' ');
        } else if (k == MLIR_ATTR_KIND_ARRAY && MLIR_GetAttributeArraySize(ctx, e) >= 2) {
            MLIR_AttributeHandle ka = MLIR_GetAttributeArrayElement(ctx, e, 0);
            MLIR_AttributeHandle va = MLIR_GetAttributeArrayElement(ctx, e, 1);
            if (MLIR_GetAttributeKind(ctx, ka) == MLIR_ATTR_KIND_STRING &&
                MLIR_GetAttributeKind(ctx, va) == MLIR_ATTR_KIND_STRING) {
                emit_attr_quoted(out, MLIR_GetAttributeString(ctx, ka));
                buf_putc(out, '=');
                emit_attr_quoted(out, MLIR_GetAttributeString(ctx, va));
                buf_putc(out, ' ');
            }
        }
    }
    buf_cstr(out, "}\n");
}

static void emit_function(MLIR_Context *ctx, Buf *out, MLIR_OpHandle fn, int attr_group_id) {
    string sym = {0};
    MLIR_TypeHandle fty = MLIR_INVALID_HANDLE;
    MLIR_AttributeHandle visa = find_attr(ctx, fn, MLIR_KEY_SYM_VISIBILITY);
    bool is_private = false;
    if (visa != MLIR_INVALID_HANDLE) {
        string v = MLIR_GetAttributeString(ctx, visa);
        if (name_eq(v, "private")) is_private = true;
    }
    MLIR_AttributeHandle syma = find_attr(ctx, fn, MLIR_KEY_SYM_NAME);
    if (syma != MLIR_INVALID_HANDLE) sym = MLIR_GetAttributeString(ctx, syma);
    MLIR_AttributeHandle ftya = find_attr(ctx, fn, MLIR_KEY_FUNCTION_TYPE);
    if (ftya != MLIR_INVALID_HANDLE) fty = MLIR_GetAttributeTypeValue(ctx, ftya);

    size_t ni = fty != MLIR_INVALID_HANDLE ? MLIR_GetTypeFunctionNumInputs(ctx, fty) : 0;
    size_t no = fty != MLIR_INVALID_HANDLE ? MLIR_GetTypeFunctionNumResults(ctx, fty) : 0;

    bool varargs = MLIR_GetTypeFunctionIsVarArg(ctx, fty);

    bool has_body = MLIR_GetOpNumRegions(ctx, fn) > 0 &&
                    MLIR_GetRegionNumBlocks(ctx, MLIR_GetOpRegion(ctx, fn, 0)) > 0;

    if (!has_body) {
        buf_cstr(out, "declare ");
        if (no == 0) buf_cstr(out, "void");
        else         print_type(out, ctx, MLIR_GetTypeFunctionResult(ctx, fty, 0));
        buf_printf(out, " @%.*s(", (int)sym.size, sym.str);
        for (size_t i = 0; i < ni; i++) {
            if (i) buf_cstr(out, ", ");
            print_type(out, ctx, MLIR_GetTypeFunctionInput(ctx, fty, i));
        }
        if (varargs) { if (ni) buf_cstr(out, ", "); buf_cstr(out, "..."); }
        if (attr_group_id >= 0) buf_printf(out, ") #%d\n", attr_group_id);
        else                    buf_cstr(out, ")\n");
        return;
    }

    buf_cstr(out, "define ");
    if (is_private) buf_cstr(out, "internal ");
    if (no == 0) buf_cstr(out, "void");
    else         print_type(out, ctx, MLIR_GetTypeFunctionResult(ctx, fty, 0));
    buf_printf(out, " @%.*s(", (int)sym.size, sym.str);
    // Parameters use the names assigned by assign_names ("%a0" etc.).
    FnCtx F; memset(&F, 0, sizeof F);
    F.ctx = ctx; F.out = out;
    MLIR_RegionHandle body = MLIR_GetOpRegion(ctx, fn, 0);
    assign_names(&F, body);
    for (size_t i = 0; i < ni; i++) {
        if (i) buf_cstr(out, ", ");
        print_type(out, ctx, MLIR_GetTypeFunctionInput(ctx, fty, i));
        buf_putc(out, ' ');
        MLIR_BlockHandle entry = MLIR_GetRegionBlock(ctx, body, 0);
        MLIR_ValueHandle a = MLIR_GetBlockArg(ctx, entry, i);
        emit_use(&F, a);
    }
    if (varargs) { if (ni) buf_cstr(out, ", "); buf_cstr(out, "..."); }
    if (attr_group_id >= 0) buf_printf(out, ") #%d {\n", attr_group_id);
    else                    buf_cstr(out, ") {\n");

    // Collect all ops (for predecessor scans).
    MLIR_OpHandle *all = NULL; size_t an = 0, ac = 0;
    collect_all_ops(ctx, body, &all, &an, &ac);

    size_t nb = MLIR_GetRegionNumBlocks(ctx, body);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, body, bi);
        // Always emit the block label, including for the entry block: after
        // mem2reg a phi in a successor block may name the entry block as an
        // incoming predecessor (e.g. "[ 1, %entry ]"), which requires the
        // entry block to carry an "entry:" label. LLVM permits this — the
        // entry block simply may not be used as a branch *target*.
        buf_printf(out, "%s:\n", bname(&F, blk));
        if (bi != 0) emit_phis(&F, blk, all, an);
        size_t nops = MLIR_GetBlockNumOps(ctx, blk);
        for (size_t oi = 0; oi < nops; oi++) {
            emit_op(&F, MLIR_GetBlockOp(ctx, blk, oi));
        }
    }

    free(all);
    buf_cstr(out, "}\n\n");

    map_free(&F.vmap);
    map_free(&F.bmap);
}

// -----------------------------------------------------------------------------
// Globals.
// -----------------------------------------------------------------------------

// Print the constant of type `ty` whose memory image (laid out with 8-byte
// pointers, see mlir_llvm_type_layout.h) is `b`, as LLVM IR: e.g.
// `{ i32 1, [2 x i8] [i8 104, i8 0] }`. An all-zero value prints as
// `zeroinitializer`; a pointer can only be null.
static void print_const_bytes(Buf *out, MLIR_Context *ctx, MLIR_TypeHandle ty,
                              const uint8_t *b) {
    unsigned sz = llvm_type_size(ctx, ty, 8);
    bool zero = true;
    for (unsigned i = 0; i < sz; i++) {
        if (b[i]) { zero = false; break; }
    }
    if ((MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_ARRAY) || (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_STRUCT)) {
        if (zero) { buf_cstr(out, "zeroinitializer"); return; }
        bool arr = (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_ARRAY);
        size_t n = arr ? (size_t)MLIR_GetTypeNumElements(ctx, ty)
                       : MLIR_GetTypeLLVMStructNumFields(ctx, ty);
        buf_cstr(out, arr ? "[" : "{ ");
        unsigned off = 0;  // the field's offset, kept running for a struct
        for (size_t i = 0; i < n; i++) {
            MLIR_TypeHandle et;
            if (arr) {
                et = MLIR_GetTypeElement(ctx, ty);
                off = (unsigned)i * llvm_type_size(ctx, et, 8);
            } else {
                et = MLIR_GetTypeLLVMStructField(ctx, ty, i);
                if (i > 0) {
                    MLIR_TypeHandle pt = MLIR_GetTypeLLVMStructField(ctx, ty, i - 1);
                    off = llvm_align_up(off + llvm_type_size(ctx, pt, 8),
                                        llvm_type_align(ctx, et, 8));
                }
            }
            if (i) buf_cstr(out, ", ");
            print_type(out, ctx, et);
            buf_putc(out, ' ');
            print_const_bytes(out, ctx, et, b + off);
        }
        buf_cstr(out, arr ? "]" : " }");
        return;
    }
    uint64_t bits = 0;
    for (unsigned i = 0; i < sz && i < 8; i++) bits |= (uint64_t)b[i] << (8 * i);
    if (is_llvm_ptr(ctx, ty)) {
        buf_cstr(out, "null");
    } else if (is_float_w(ctx, ty, 32)) {
        uint32_t b32 = (uint32_t)bits;
        float f;
        memcpy(&f, &b32, 4);
        char *l = fmt_fp_literal(ctx, ty, (double)f);
        buf_cstr(out, l);
        free(l);
    } else if (is_float_w(ctx, ty, 64)) {
        double d;
        memcpy(&d, &bits, 8);
        char *l = fmt_fp_literal(ctx, ty, d);
        buf_cstr(out, l);
        free(l);
    } else {
        // An integer, sign-extended from its width.
        if (sz < 8 && (bits >> (8 * sz - 1)) & 1) bits |= ~(uint64_t)0 << (8 * sz);
        buf_printf(out, "%lld", (long long)(int64_t)bits);
    }
}

static void emit_global(MLIR_Context *ctx, Buf *out, MLIR_OpHandle gop) {
    MLIR_AttributeHandle syma = find_attr(ctx, gop, MLIR_KEY_SYM_NAME);
    string sym = syma != MLIR_INVALID_HANDLE ? MLIR_GetAttributeString(ctx, syma) : (string){0,0};
    MLIR_AttributeHandle gtya = find_attr(ctx, gop, MLIR_KEY_GLOBAL_TYPE);
    MLIR_TypeHandle gty = MLIR_INVALID_HANDLE;
    if (gtya != MLIR_INVALID_HANDLE) gty = MLIR_GetAttributeTypeValue(ctx, gtya);

    // Linkage: "linkage" attr is an LLVM Linkage attr; we read it via
    // GetAttributeAsString and map common values. A separate flag tracks
    // "external" because it's a declaration form in LLVM IR text (no
    // initializer is permitted/expected) rather than a linkage prefix.
    const char *linkage = "";
    bool is_external = false;
    MLIR_AttributeHandle linka = find_attr(ctx, gop, MLIR_KEY_LINKAGE);
    if (linka != MLIR_INVALID_HANDLE) {
        string s = MLIR_GetAttributeAsString(ctx, linka);
        if      (name_starts_with(s, "#llvm.linkage<internal>")) linkage = "internal ";
        else if (name_starts_with(s, "#llvm.linkage<private>"))  linkage = "private ";
        else if (name_starts_with(s, "#llvm.linkage<external>")) is_external = true;
        else if (name_starts_with(s, "#llvm.linkage<common>"))   linkage = "common ";
        else if (name_starts_with(s, "#llvm.linkage<weak>"))     linkage = "weak ";
    }
    bool is_constant = false;
    MLIR_AttributeHandle ca = find_attr(ctx, gop, MLIR_KEY_CONSTANT);
    if (ca != MLIR_INVALID_HANDLE) is_constant = true;

    // Initial value: either the "value" attribute (string/numeric), or a
    // body region whose llvm.return yields the constant.
    char *init = NULL;
    MLIR_AttributeHandle va = find_attr(ctx, gop, MLIR_KEY_VALUE);
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind k = MLIR_GetAttributeKind(ctx, va);
        if (k == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(ctx, va);
            // For !llvm.array<N x iX> with X > 8, LLVM IR cannot use a
            // c"..." byte string — that form is reserved for i8 arrays.
            // Unpack the raw bytes into an `[N x iX] [iX v1, iX v2, ...]`
            // typed array constant.
            unsigned elem_w = 0;
            uint64_t arr_n = 0;
            if (gty != MLIR_INVALID_HANDLE && (MLIR_GetTypeKind(ctx, gty) == MLIR_TYPE_KIND_LLVM_ARRAY)) {
                arr_n = MLIR_GetTypeNumElements(ctx, gty);
                MLIR_TypeHandle et = MLIR_GetTypeElement(ctx, gty);
                uint32_t w = int_width(ctx, et);
                if (w == 8 || w == 16 || w == 32 || w == 64) elem_w = w;
            }
            if (elem_w > 8 && arr_n > 0 &&
                s.size == arr_n * (elem_w / 8)) {
                unsigned esz = elem_w / 8;
                Buf tmp = {0};
                buf_putc(&tmp, '[');
                const unsigned char *bs = (const unsigned char *)s.str;
                for (uint64_t i = 0; i < arr_n; i++) {
                    if (i) buf_cstr(&tmp, ", ");
                    uint64_t v = 0;
                    for (unsigned b2 = 0; b2 < esz; b2++) {
                        v |= (uint64_t)bs[i * esz + b2] << (8 * b2);
                    }
                    int64_t sv;
                    if      (esz == 2) sv = (int64_t)(int16_t)v;
                    else if (esz == 4) sv = (int64_t)(int32_t)v;
                    else               sv = (int64_t)v;
                    buf_printf(&tmp, "i%u %lld", elem_w, (long long)sv);
                }
                buf_putc(&tmp, ']');
                init = (char *)malloc(tmp.len + 1);
                memcpy(init, tmp.data, tmp.len); init[tmp.len] = 0;
                free(tmp.data);
            } else {
                // Build c"..." with escapes (i8 array or generic string).
                Buf tmp = {0};
                buf_cstr(&tmp, "c\"");
                for (size_t i = 0; i < s.size; i++) {
                    unsigned char c = (unsigned char)s.str[i];
                    if (c == '\\' || c == '"' || c < 32 || c >= 127) {
                        buf_printf(&tmp, "\\%02X", c);
                    } else {
                        buf_putc(&tmp, (char)c);
                    }
                }
                buf_cstr(&tmp, "\"");
                init = (char *)malloc(tmp.len + 1);
                memcpy(init, tmp.data, tmp.len); init[tmp.len] = 0;
                free(tmp.data);
            }
        } else if (k == MLIR_ATTR_KIND_INTEGER) {
            init = fmt_alloc("%lld", (long long)MLIR_GetAttributeInteger(ctx, va));
        } else if (k == MLIR_ATTR_KIND_FLOAT) {
            init = fmt_fp_literal(ctx, gty, MLIR_GetAttributeFloat(ctx, va));
        }
    }
    if (init == NULL && MLIR_GetOpNumRegions(ctx, gop) > 0) {
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, gop, 0);
        if (MLIR_GetRegionNumBlocks(ctx, r) > 0) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, r, 0);
            size_t nops = MLIR_GetBlockNumOps(ctx, blk);
            // Find the llvm.return; its operand's defining op is the
            // initializer expression.
            for (size_t i = 0; i < nops; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blk, i);
                if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_RETURN &&
                    MLIR_GetOpNumOperands(ctx, op) > 0) {
                    MLIR_ValueHandle v = MLIR_GetOpOperand(ctx, op, 0);
                    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
                    char *lit = inline_literal_for(ctx, def, MLIR_GetValueType(ctx, v));
                    if (lit) { init = lit; break; }
                    // A constant aggregate built by an llvm.insertvalue
                    // chain (a struct initializer, or an array from
                    // MLIR_CreateLLVMGlobalArrayInit's upstream impl).
                    unsigned sz = gty != MLIR_INVALID_HANDLE ? llvm_type_size(ctx, gty, 8) : 0;
                    uint8_t *img = sz ? (uint8_t *)calloc(sz, 1) : NULL;
                    if (img && llvm_global_init_bytes(ctx, gop, gty, 8, img, sz)) {
                        Buf tmp = {0};
                        print_const_bytes(&tmp, ctx, gty, img);
                        buf_putc(&tmp, '\0');
                        init = tmp.data;
                    }
                    free(img);
                    break;
                }
            }
        }
    }

    // External linkage with no initializer is a declaration:
    //   @name = external global T
    // Otherwise emit as a definition with an initializer (zeroinitializer
    // by default).
    if (is_external && init == NULL) {
        buf_printf(out, "@%.*s = external %s ", (int)sym.size, sym.str,
                   is_constant ? "constant" : "global");
        print_type(out, ctx, gty);
        buf_putc(out, '\n');
        return;
    }
    if (init == NULL) init = xstrdup("zeroinitializer");

    buf_printf(out, "@%.*s = %s%s ", (int)sym.size, sym.str,
               linkage, is_constant ? "constant" : "global");
    print_type(out, ctx, gty);
    buf_putc(out, ' ');
    buf_cstr(out, init);
    buf_putc(out, '\n');
    free(init);
}

// -----------------------------------------------------------------------------
// Named struct collection + declaration.
// -----------------------------------------------------------------------------
//
// Walk every type referenced by every op/value in the module. When we see
// !llvm.struct<"Name", (...)>, record (name, body-text) once and emit a
// `%Name = type { ... }` line at the top.

typedef struct { const char *name; size_t name_n; const char *body; size_t body_n; } StructDecl;
// `recent` are the types scanned last: an insertvalue chain repeats one
// aggregate type (and its scalar types) thousands of times, and formatting
// it each time is quadratic for a wide struct.
typedef struct {
    StructDecl *e;
    size_t n, cap;
    MLIR_TypeHandle recent[4];
    size_t next_recent;
} StructList;

static bool struct_seen(StructList *L, const char *name, size_t name_n) {
    for (size_t i = 0; i < L->n; i++) {
        if (L->e[i].name_n == name_n && memcmp(L->e[i].name, name, name_n) == 0) return true;
    }
    return false;
}
static void scan_type_for_structs(MLIR_Context *ctx, MLIR_TypeHandle ty, StructList *L);

static void scan_type_text(MLIR_Context *ctx, const char *s, size_t n, StructList *L) {
    // Find all occurrences of `struct<"Name", (body)>`. The leading
    // `!llvm.` prefix is optional — MLIR omits it for struct fields nested
    // inside another `!llvm.struct<>`.
    size_t i = 0;
    while (i + 7 < n) {
        bool has_prefix = (i + 13 <= n) && memcmp(s + i, "!llvm.struct<", 13) == 0;
        bool no_prefix  = !has_prefix && memcmp(s + i, "struct<", 7) == 0;
        if (!has_prefix && !no_prefix) { i++; continue; }
        size_t hdr = has_prefix ? 13 : 7;
        size_t j = i + hdr;
        if (j < n && s[j] == '"') {
            size_t name_start = j + 1;
            size_t name_end = name_start;
            while (name_end < n && s[name_end] != '"') name_end++;
            int depth = 1; size_t k = name_end + 1;
            while (k < n && depth > 0) {
                if (s[k] == '<') depth++;
                else if (s[k] == '>') { depth--; if (depth == 0) break; }
                k++;
            }
            size_t bs = name_end + 1;
            while (bs < k && (s[bs] == ',' || s[bs] == ' ')) bs++;
            if (!struct_seen(L, s + name_start, name_end - name_start)) {
                if (L->n == L->cap) {
                    L->cap = L->cap ? L->cap * 2 : 8;
                    L->e = (StructDecl *)realloc(L->e, L->cap * sizeof(StructDecl));
                }
                L->e[L->n].name = s + name_start;
                L->e[L->n].name_n = name_end - name_start;
                L->e[L->n].body = s + bs;
                L->e[L->n].body_n = k - bs;
                L->n++;
            }
            // Recurse into the body so we pick up nested struct decls too.
            scan_type_text(ctx, s + bs, k - bs, L);
            i = k + 1;
        } else {
            i++;
        }
    }
}
static void scan_type_for_structs(MLIR_Context *ctx, MLIR_TypeHandle ty, StructList *L) {
    if (ty == MLIR_INVALID_HANDLE) return;
    for (size_t i = 0; i < 4; i++) {
        if (L->recent[i] == ty) return;
    }
    L->recent[L->next_recent] = ty;
    L->next_recent = (L->next_recent + 1) % 4;
    string s = MLIR_GetTypeString(ctx, ty);
    scan_type_text(ctx, s.str, s.size, L);
}

static void scan_op_for_structs(MLIR_Context *ctx, MLIR_OpHandle op, StructList *L) {
    size_t nr = MLIR_GetOpNumResults(ctx, op);
    for (size_t i = 0; i < nr; i++) scan_type_for_structs(ctx, MLIR_GetOpResultType(ctx, op, i), L);
    size_t no = MLIR_GetOpNumOperands(ctx, op);
    for (size_t i = 0; i < no; i++) scan_type_for_structs(ctx, MLIR_GetValueType(ctx, MLIR_GetOpOperand(ctx, op, i)), L);
    size_t na = MLIR_GetOpNumAttributes(ctx, op);
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(ctx, op, i);
        if (MLIR_GetAttributeKind(ctx, a) == MLIR_ATTR_KIND_TYPE) {
            scan_type_for_structs(ctx, MLIR_GetAttributeTypeValue(ctx, a), L);
        } else {
            // Scan the printed form for embedded type strings.
            string s = MLIR_GetAttributeAsString(ctx, a);
            scan_type_text(ctx, s.str, s.size, L);
        }
    }
    size_t nrg = MLIR_GetOpNumRegions(ctx, op);
    for (size_t ri = 0; ri < nrg; ri++) {
        MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, op, ri);
        size_t nb = MLIR_GetRegionNumBlocks(ctx, r);
        for (size_t bi = 0; bi < nb; bi++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, r, bi);
            size_t nops = MLIR_GetBlockNumOps(ctx, blk);
            for (size_t oi = 0; oi < nops; oi++) {
                scan_op_for_structs(ctx, MLIR_GetBlockOp(ctx, blk, oi), L);
            }
        }
    }
}

static bool module_has_global(MLIR_Context *ctx, MLIR_BlockHandle mb, const char *gn) {
    size_t n = MLIR_GetBlockNumOps(ctx, mb);
    for (size_t i = 0; i < n; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) != OP_TYPE_LLVM_MLIR_GLOBAL) continue;
        MLIR_AttributeHandle syma = find_attr(ctx, op, MLIR_KEY_SYM_NAME);
        if (syma == MLIR_INVALID_HANDLE) continue;
        string sym = MLIR_GetAttributeString(ctx, syma);
        if (sym.size == strlen(gn) && memcmp(sym.str, gn, sym.size) == 0)
            return true;
    }
    return false;
}

static bool op_uses_wasm_memory(MLIR_Context *ctx, MLIR_OpHandle op) {
    MLIR_OpType ty = MLIR_GetOpType(ctx, op);
    if (ty == OP_TYPE_LLVM_INTR_WASM_MEMORY_SIZE ||
        ty == OP_TYPE_LLVM_INTR_WASM_MEMORY_GROW)
        return true;
    size_t nr = MLIR_GetOpNumRegions(ctx, op);
    for (size_t r = 0; r < nr; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, op, r);
        size_t nb = MLIR_GetRegionNumBlocks(ctx, reg);
        for (size_t b = 0; b < nb; b++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t no = MLIR_GetBlockNumOps(ctx, blk);
            for (size_t i = 0; i < no; i++)
                if (op_uses_wasm_memory(ctx, MLIR_GetBlockOp(ctx, blk, i))) return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// Module entry.
// -----------------------------------------------------------------------------

string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module) {
    Buf out = {0};
    buf_cstr(&out, "; ModuleID = 'LLVMDialectModule'\n");
    buf_cstr(&out, "source_filename = \"LLVMDialectModule\"\n\n");

    // Module body region 0 contains a single block with all top-level ops.
    MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(ctx, mr, 0);
    size_t nops = MLIR_GetBlockNumOps(ctx, mb);

    // Gather struct decls.
    StructList sl = {0};
    for (size_t i = 0; i < nops; i++) {
        scan_op_for_structs(ctx, MLIR_GetBlockOp(ctx, mb, i), &sl);
    }
    for (size_t i = 0; i < sl.n; i++) {
        // Translate the body text (a tuple "(T1, T2)") to LLVM "{ T1, T2 }".
        Buf tb = {0};
        const char *b = sl.e[i].body; size_t bn = sl.e[i].body_n;
        // body should be like "(i32, i32)"
        if (bn >= 2 && b[0] == '(' && b[bn-1] == ')') {
            buf_cstr(&tb, "{ ");
            const char *p = b + 1; size_t left = bn - 2;
            bool first = true;
            while (left > 0) {
                int depth = 0; size_t j = 0;
                while (j < left) {
                    char c = p[j];
                    if (c == '<' || c == '(' || c == '[' || c == '{') depth++;
                    else if (c == '>' || c == ')' || c == ']' || c == '}') depth--;
                    else if (c == ',' && depth == 0) break;
                    j++;
                }
                if (!first) buf_cstr(&tb, ", ");
                print_llvm_type_text(&tb, p, j);
                first = false;
                if (j < left && p[j] == ',') { j++; while (j < left && p[j] == ' ') j++; }
                p += j; left -= j;
            }
            buf_cstr(&tb, " }");
        } else {
            buf_append(&tb, b, bn);
        }
        buf_printf(&out, "%%%.*s = type %.*s\n",
                   (int)sl.e[i].name_n, sl.e[i].name,
                   (int)tb.len, tb.data);
        free(tb.data);
    }
    if (sl.n) buf_putc(&out, '\n');
    free(sl.e);

    bool need_wasm_pages = false;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FUNC && op_uses_wasm_memory(ctx, op))
            need_wasm_pages = true;
    }
    if (need_wasm_pages && !module_has_global(ctx, mb, "__wasm_mem_pages"))
        buf_cstr(&out, "@__wasm_mem_pages = global i32 0, align 4\n\n");

    // Emit globals first, then functions (order doesn't really matter to llc).
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_MLIR_GLOBAL) emit_global(ctx, &out, op);
    }
    int next_attr_group = 0;
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
        if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FUNC) {
            int gid = has_nonempty_passthrough(ctx, op) ? next_attr_group++ : -1;
            emit_function(ctx, &out, op, gid);
        }
    }
    if (next_attr_group > 0) {
        size_t gid = 0;
        for (size_t i = 0; i < nops; i++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(ctx, mb, i);
            if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FUNC && has_nonempty_passthrough(ctx, op))
                emit_passthrough_group(ctx, &out, op, gid++);
        }
    }

    // Copy result into the arena.
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    char *buf = (char *)arena_alloc(arena, out.len);
    memcpy(buf, out.data, out.len);
    free(out.data);
    string r;
    r.str = buf;
    r.size = out.len;
    return r;
}

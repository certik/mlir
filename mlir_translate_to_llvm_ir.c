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
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w*10 + (s.str[i]-'0');
            else { w = 0; break; }
        }
        if (w == 1) return 1;
        if (w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    if (s.size == 3 && memcmp(s.str, "f16", 3) == 0) return 2;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 8;
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return 8;
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
    string opn = MLIR_GetOpName(op);
    if (name_eq(opn, "llvm.mlir.constant")) {
        // Value is the "value" attribute. It's printed as e.g. "3 : i32" or
        // "1.5e+00 : f32".
        size_t na = MLIR_GetOpNumAttributes(op);
        for (size_t i = 0; i < na; i++) {
            MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
            string an = MLIR_GetAttributeName(a);
            if (name_eq(an, "value")) {
                MLIR_AttrKind k = MLIR_GetAttributeKind(a);
                if (k == MLIR_ATTR_KIND_INTEGER) {
                    int64_t iv = MLIR_GetAttributeInteger(a);
                    return fmt_alloc("%lld", (long long)iv);
                } else if (k == MLIR_ATTR_KIND_FLOAT) {
                    double fv = MLIR_GetAttributeFloat(a);
                    // Match MLIR/LLVM "0x..." style for non-finite would be
                    // overkill; plain printf %e suffices for tinyc.
                    return fmt_alloc("%e", fv);
                } else if (k == MLIR_ATTR_KIND_BOOL) {
                    bool bv = MLIR_GetAttributeBool(a);
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
    if (name_eq(opn, "llvm.mlir.zero") || name_eq(opn, "llvm.mlir.undef") ||
        name_eq(opn, "llvm.mlir.poison")) {
        // Type-driven: ptr -> null, anything aggregate -> zeroinitializer,
        // scalar -> 0 / 0.0.
        string tys = MLIR_GetTypeString(ctx, vty);
        if (tys.size >= 9 && memcmp(tys.str, "!llvm.ptr", 9) == 0) {
            return xstrdup(name_eq(opn, "llvm.mlir.undef") ? "undef" :
                           name_eq(opn, "llvm.mlir.poison") ? "poison" : "null");
        }
        if (name_eq(opn, "llvm.mlir.undef")) return xstrdup("undef");
        if (name_eq(opn, "llvm.mlir.poison")) return xstrdup("poison");
        // zero
        if (tys.size >= 1 && tys.str[0] == 'i') return xstrdup("0");
        if (tys.size == 3 && (memcmp(tys.str, "f32", 3) == 0 || memcmp(tys.str, "f64", 3) == 0))
            return xstrdup("0.000000e+00");
        return xstrdup("zeroinitializer");
    }
    if (name_eq(opn, "llvm.mlir.null")) return xstrdup("null");
    if (name_eq(opn, "llvm.mlir.addressof")) {
        size_t na = MLIR_GetOpNumAttributes(op);
        for (size_t i = 0; i < na; i++) {
            MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
            string an = MLIR_GetAttributeName(a);
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

// Parse an integer at `*p` (up to `end`), advancing `*p` past it. Skips
// leading whitespace. Returns false if no integer is present.
static bool parse_int64(const char **p, const char *end, int64_t *out) {
    const char *q = *p;
    while (q < end && (*q == ' ' || *q == '\t')) q++;
    bool neg = false;
    if (q < end && (*q == '-' || *q == '+')) { neg = (*q == '-'); q++; }
    if (q >= end || *q < '0' || *q > '9') return false;
    int64_t v = 0;
    while (q < end && *q >= '0' && *q <= '9') { v = v * 10 + (*q - '0'); q++; }
    *out = neg ? -v : v;
    *p = q;
    return true;
}

// Extract the first integer from a `position` attribute printed as
// "array<i64: N, ...>" (DenseI64ArrayAttr) or "[N, ...]" (ArrayAttr of
// IntegerAttr). For nested aggregate insertions the chain inserts at
// successive single-level positions, so we only look at the first index.
static bool parse_position_first(string s, int64_t *out) {
    const char *p = s.str;
    const char *end = s.str + s.size;
    // The printed form is either "array<iN: V, ...>" or "[V, ...]".
    // For the former we must skip past the ":" so we don't mistake the
    // element-width (e.g. "i64") for the position.
    const char *colon = NULL;
    for (const char *q = p; q < end; q++) {
        if (*q == ':') { colon = q; break; }
    }
    if (colon) p = colon + 1;
    else if (p < end && *p == '[') p++;
    return parse_int64(&p, end, out);
}

// Per-slot state used by fold_aggregate_init_chain. Declared at file
// scope so tinyc (which lacks function-local struct decls) accepts it.
typedef struct FoldSlot {
    int64_t pos;
    MLIR_ValueHandle scalar;
    bool set;
} FoldSlot;

// Fold an `llvm.mlir.undef` + chain of `llvm.insertvalue` into a typed
// LLVM IR aggregate constant literal. Returns a malloc'd string on
// success (e.g. "[i32 10, i32 20, i32 30, i32 40]") or NULL if the
// pattern doesn't fit. Only handles single-level !llvm.array<N x iX>
// aggregates today — what MLIR_CreateLLVMGlobalArrayInit produces.
static char *fold_aggregate_init_chain(MLIR_Context *ctx, MLIR_OpHandle root,
                                       MLIR_TypeHandle agg_ty) {
    if (agg_ty == MLIR_INVALID_HANDLE || !MLIR_IsTypeLLVMArray(agg_ty))
        return NULL;
    uint64_t n = MLIR_GetTypeLLVMArrayNumElements(agg_ty);
    if (n == 0 || n > (1u << 20)) return NULL;
    MLIR_TypeHandle elem_ty = MLIR_GetTypeLLVMArrayElement(agg_ty);
    string ets = MLIR_GetTypeString(ctx, elem_ty);
    // Only handle integer element types iX (X in {8,16,32,64}).
    unsigned elem_w = 0;
    if (ets.size >= 2 && ets.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < ets.size; i++) {
            if (ets.str[i] >= '0' && ets.str[i] <= '9') w = w * 10 + (ets.str[i] - '0');
            else { w = 0; break; }
        }
        if (w == 8 || w == 16 || w == 32 || w == 64) elem_w = (unsigned)w;
    }
    if (elem_w == 0) return NULL;

    // Collect insertvalues by walking back operand[0] from `root`.
    // Each insertvalue contributes (position, scalar-value-handle).
    FoldSlot *slots = (FoldSlot *)calloc(n, sizeof(FoldSlot));
    if (!slots) return NULL;
    MLIR_OpHandle cur = root;
    for (;;) {
        string opn = MLIR_GetOpName(cur);
        if (name_eq(opn, "llvm.insertvalue")) {
            if (MLIR_GetOpNumOperands(cur) < 2) { free(slots); return NULL; }
            // Find position attribute.
            int64_t pos = -1;
            size_t na = MLIR_GetOpNumAttributes(cur);
            for (size_t i = 0; i < na; i++) {
                MLIR_AttributeHandle a = MLIR_GetOpAttribute(cur, i);
                if (name_eq(MLIR_GetAttributeName(a), "position")) {
                    string ps = MLIR_GetAttributeAsString(ctx, a);
                    if (!parse_position_first(ps, &pos)) pos = -1;
                    break;
                }
            }
            if (pos < 0 || (uint64_t)pos >= n) { free(slots); return NULL; }
            MLIR_ValueHandle scalar = MLIR_GetOpOperand(cur, 1);
            if (!slots[pos].set) {
                slots[pos].pos = pos;
                slots[pos].scalar = scalar;
                slots[pos].set = true;
            }
            MLIR_ValueHandle prev = MLIR_GetOpOperand(cur, 0);
            cur = MLIR_GetValueDefiningOp(prev);
            if (cur == MLIR_INVALID_HANDLE) { free(slots); return NULL; }
            continue;
        }
        if (name_eq(opn, "llvm.mlir.undef") || name_eq(opn, "llvm.mlir.zero") ||
            name_eq(opn, "llvm.mlir.poison")) {
            break;
        }
        // Unknown base op — bail.
        free(slots); return NULL;
    }
    string base_opn = MLIR_GetOpName(cur);
    const char *default_lit =
        name_eq(base_opn, "llvm.mlir.undef")   ? "undef" :
        name_eq(base_opn, "llvm.mlir.poison")  ? "poison" : "0";

    // Resolve each slot's scalar to a literal by inspecting its
    // defining op. Only accept constant-like ops.
    char **lits = (char **)calloc(n, sizeof(char *));
    if (!lits) { free(slots); return NULL; }
    bool ok = true;
    for (uint64_t i = 0; i < n; i++) {
        if (!slots[i].set) {
            lits[i] = xstrdup(default_lit);
            continue;
        }
        MLIR_OpHandle sd = MLIR_GetValueDefiningOp(slots[i].scalar);
        if (sd == MLIR_INVALID_HANDLE) { ok = false; break; }
        char *l = inline_literal_for(ctx, sd, MLIR_GetValueType(slots[i].scalar));
        if (!l) { ok = false; break; }
        lits[i] = l;
    }
    char *result = NULL;
    if (ok) {
        Buf tmp = {0};
        buf_putc(&tmp, '[');
        for (uint64_t i = 0; i < n; i++) {
            if (i) buf_cstr(&tmp, ", ");
            buf_printf(&tmp, "i%u %s", elem_w, lits[i]);
        }
        buf_putc(&tmp, ']');
        result = (char *)malloc(tmp.len + 1);
        memcpy(result, tmp.data, tmp.len); result[tmp.len] = 0;
        free(tmp.data);
    }
    for (uint64_t i = 0; i < n; i++) free(lits[i]);
    free(lits);
    free(slots);
    return result;
}


// Pre-pass: assign names to all values & blocks in the function body.
static void assign_names(FnCtx *F, MLIR_RegionHandle body) {
    size_t nb = MLIR_GetRegionNumBlocks(body);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(body, bi);
        // Block name: "entry" for block 0, "bbN" for the rest.
        char *bn = bi == 0 ? xstrdup("entry") : fmt_alloc("bb%d", F->next_b++);
        map_put(&F->bmap, blk, bn);
        // Block arg names: "%aN" — for entry, these are also the function
        // parameters and are referred to as %0, %1, ...
        size_t na = MLIR_GetBlockNumArgs(blk);
        for (size_t ai = 0; ai < na; ai++) {
            MLIR_ValueHandle a = MLIR_GetBlockArg(blk, ai);
            char *vn = bi == 0 ? fmt_alloc("%%a%zu", ai) : fmt_alloc("%%v%d", F->next_v++);
            map_put(&F->vmap, a, vn);
        }
        // Op results.
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t oi = 0; oi < nops; oi++) {
            MLIR_OpHandle op = MLIR_GetBlockOp(blk, oi);
            size_t nres = MLIR_GetOpNumResults(op);
            for (size_t ri = 0; ri < nres; ri++) {
                MLIR_ValueHandle r = MLIR_GetOpResult(op, ri);
                MLIR_TypeHandle rty = MLIR_GetValueType(r);
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

static MLIR_AttributeHandle find_attr(MLIR_OpHandle op, const char *name) {
    size_t na = MLIR_GetOpNumAttributes(op);
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        string an = MLIR_GetAttributeName(a);
        if (name_eq(an, name)) return a;
    }
    return MLIR_INVALID_HANDLE;
}

static void emit_call(FnCtx *F, MLIR_OpHandle op) {
    Buf *out = F->out;
    size_t nres = MLIR_GetOpNumResults(op);
    MLIR_TypeHandle rty = nres ? MLIR_GetOpResult_type(op, 0) : MLIR_INVALID_HANDLE;
    bool is_void = (nres == 0);

    // Find callee (symbol) attr — "callee" for direct, none for indirect (the
    // callee value is then the first operand).
    MLIR_AttributeHandle callee = find_attr(op, "callee");
    MLIR_AttributeHandle var_callee_type = find_attr(op, "var_callee_type");
    bool indirect = (callee == MLIR_INVALID_HANDLE);

    // Operands.
    size_t nops = MLIR_GetOpNumOperands(op);
    size_t arg_start = indirect ? 1 : 0;

    buf_cstr(out, "  ");
    if (!is_void) { emit_use(F, MLIR_GetOpResult(op, 0)); buf_cstr(out, " = "); }
    buf_cstr(out, "call ");

    // Vararg form: emit the full function signature `<ret> (<arg-types>, ...)`.
    // For non-variadic indirect calls, just the return type is sufficient
    // (LLVM IR allows `call <ret> %fnptr(...)` when the function is not
    // variadic). Emitting the variadic form unconditionally for indirect
    // calls produces calls with the wrong signature on targets like wasm32
    // where variadic and fixed-arity ABIs differ.
    if (var_callee_type != MLIR_INVALID_HANDLE) {
        MLIR_TypeHandle fty = MLIR_GetAttributeTypeValue(var_callee_type);
        bool is_vararg = MLIR_GetTypeFunctionIsVarArg(fty);
        if (is_vararg) {
            size_t ni = MLIR_GetTypeFunctionNumInputs(fty);
            size_t no = MLIR_GetTypeFunctionNumResults(fty);
            if (no == 0) buf_cstr(out, "void");
            else         print_type(out, F->ctx, MLIR_GetTypeFunctionResult(fty, 0));
            buf_cstr(out, " (");
            for (size_t i = 0; i < ni; i++) {
                if (i) buf_cstr(out, ", ");
                print_type(out, F->ctx, MLIR_GetTypeFunctionInput(fty, i));
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

    if (indirect) emit_use(F, MLIR_GetOpOperand(op, 0));
    else {
        string s = MLIR_GetAttributeAsString(F->ctx, callee);
        buf_str(out, s);
    }
    buf_putc(out, '(');
    for (size_t i = arg_start; i < nops; i++) {
        if (i > arg_start) buf_cstr(out, ", ");
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, i);
        print_type(out, F->ctx, MLIR_GetValueType(a));
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
    size_t na = MLIR_GetBlockNumArgs(blk);
    if (na == 0) return;
    for (size_t ai = 0; ai < na; ai++) {
        MLIR_ValueHandle arg = MLIR_GetBlockArg(blk, ai);
        MLIR_TypeHandle ty = MLIR_GetValueType(arg);
        buf_cstr(F->out, "  ");
        emit_use(F, arg);
        buf_cstr(F->out, " = phi ");
        print_type(F->out, F->ctx, ty);
        buf_putc(F->out, ' ');
        bool first = true;
        for (size_t i = 0; i < all_n; i++) {
            MLIR_OpHandle term = all_ops[i];
            size_t nsucc = MLIR_GetOpNumSuccessors(term);
            for (size_t s = 0; s < nsucc; s++) {
                MLIR_BlockHandle dst = MLIR_GetOpSuccessor(term, s);
                if (dst != blk) continue;
                MLIR_ValueHandle v = MLIR_GetOpSuccessorOperand(term, s, ai);
                MLIR_BlockHandle src = MLIR_GetOpParentBlock(term);
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
static bool is_inline_constant(string opn) {
    return name_eq(opn, "llvm.mlir.constant") ||
           name_eq(opn, "llvm.mlir.zero") ||
           name_eq(opn, "llvm.mlir.undef") ||
           name_eq(opn, "llvm.mlir.poison") ||
           name_eq(opn, "llvm.mlir.null") ||
           name_eq(opn, "llvm.mlir.addressof");
}

// Map of llvm.* arith op names to LLVM IR opcodes (1 result, 2 operands,
// printed as "%r = OP T a, b\n").
static const char *llvm_binop_opcode(string opn) {
    struct { const char *mlir; const char *llvm; } tbl[] = {
        {"llvm.add",  "add"},  {"llvm.sub", "sub"},  {"llvm.mul", "mul"},
        {"llvm.sdiv", "sdiv"}, {"llvm.udiv","udiv"},
        {"llvm.srem", "srem"}, {"llvm.urem","urem"},
        {"llvm.and",  "and"},  {"llvm.or",  "or"},   {"llvm.xor", "xor"},
        {"llvm.shl",  "shl"},  {"llvm.lshr","lshr"}, {"llvm.ashr","ashr"},
        {"llvm.fadd", "fadd"}, {"llvm.fsub","fsub"}, {"llvm.fmul","fmul"},
        {"llvm.fdiv", "fdiv"}, {"llvm.frem","frem"},
        {NULL, NULL},
    };
    for (int i = 0; tbl[i].mlir; i++) if (name_eq(opn, tbl[i].mlir)) return tbl[i].llvm;
    return NULL;
}

static const char *llvm_cast_opcode(string opn) {
    struct { const char *mlir; const char *llvm; } tbl[] = {
        {"llvm.sext", "sext"},     {"llvm.zext", "zext"},
        {"llvm.trunc","trunc"},
        {"llvm.fpext","fpext"},    {"llvm.fptrunc","fptrunc"},
        {"llvm.sitofp","sitofp"},  {"llvm.uitofp","uitofp"},
        {"llvm.fptosi","fptosi"},  {"llvm.fptoui","fptoui"},
        {"llvm.bitcast","bitcast"},{"llvm.ptrtoint","ptrtoint"},
        {"llvm.inttoptr","inttoptr"}, {"llvm.addrspacecast","addrspacecast"},
        {NULL, NULL},
    };
    for (int i = 0; tbl[i].mlir; i++) if (name_eq(opn, tbl[i].mlir)) return tbl[i].llvm;
    return NULL;
}

static void emit_op(FnCtx *F, MLIR_OpHandle op) {
    Buf *out = F->out;
    string opn = MLIR_GetOpName(op);

    if (is_inline_constant(opn)) return; // inlined at uses

    const char *bop;
    if ((bop = llvm_binop_opcode(opn)) != NULL) {
        // %r = OP T a, b
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = ");
        buf_cstr(out, bop);
        buf_putc(out, ' ');
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(op, 1);
        print_type(out, F->ctx, MLIR_GetValueType(a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, ", ");
        emit_use(F, b);
        buf_putc(out, '\n');
        return;
    }
    const char *cop;
    if ((cop = llvm_cast_opcode(opn)) != NULL) {
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = ");
        buf_cstr(out, cop);
        buf_putc(out, ' ');
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        print_type(out, F->ctx, MLIR_GetValueType(a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, " to ");
        print_type(out, F->ctx, MLIR_GetOpResult_type(op, 0));
        buf_putc(out, '\n');
        return;
    }

    if (name_eq(opn, "llvm.icmp") || name_eq(opn, "llvm.fcmp")) {
        // %r = icmp pred T a, b
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = ");
        buf_cstr(out, name_eq(opn, "llvm.icmp") ? "icmp " : "fcmp ");
        MLIR_AttributeHandle pa = find_attr(op, "predicate");
        const char *predtbl_i[] = {"eq","ne","slt","sle","sgt","sge","ult","ule","ugt","uge"};
        const char *predtbl_f[] = {
            "false","oeq","ogt","oge","olt","ole","one","ord",
            "ueq","ugt","uge","ult","ule","une","uno","true"
        };
        int pi = 0;
        if (pa != MLIR_INVALID_HANDLE) pi = (int)MLIR_GetAttributeInteger(pa);
        if (name_eq(opn, "llvm.icmp")) {
            if (pi >= 0 && pi < 10) buf_cstr(out, predtbl_i[pi]); else buf_cstr(out, "eq");
        } else {
            if (pi >= 0 && pi < 16) buf_cstr(out, predtbl_f[pi]); else buf_cstr(out, "oeq");
        }
        buf_putc(out, ' ');
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle b = MLIR_GetOpOperand(op, 1);
        print_type(out, F->ctx, MLIR_GetValueType(a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, ", ");
        emit_use(F, b);
        buf_putc(out, '\n');
        return;
    }

    if (name_eq(opn, "llvm.select")) {
        // %r = select i1 c, T a, T b
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = select i1 ");
        emit_use(F, MLIR_GetOpOperand(op, 0));
        buf_cstr(out, ", ");
        MLIR_ValueHandle a = MLIR_GetOpOperand(op, 1);
        MLIR_ValueHandle b = MLIR_GetOpOperand(op, 2);
        print_type(out, F->ctx, MLIR_GetValueType(a));
        buf_putc(out, ' ');
        emit_use(F, a);
        buf_cstr(out, ", ");
        print_type(out, F->ctx, MLIR_GetValueType(b));
        buf_putc(out, ' ');
        emit_use(F, b);
        buf_putc(out, '\n');
        return;
    }

    if (name_eq(opn, "llvm.alloca")) {
        // %r = alloca T, i64 N, align A
        // The alloca element type is in the "elem_type" attribute; the
        // operand is the count (i64 typically).
        MLIR_AttributeHandle ea = find_attr(op, "elem_type");
        MLIR_TypeHandle et = MLIR_GetAttributeTypeValue(ea);
        MLIR_ValueHandle cnt = MLIR_GetOpOperand(op, 0);
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = alloca ");
        print_type(out, F->ctx, et);
        buf_cstr(out, ", ");
        print_type(out, F->ctx, MLIR_GetValueType(cnt));
        buf_putc(out, ' ');
        emit_use(F, cnt);
        buf_printf(out, ", align %u\n", type_align(F->ctx, et));
        return;
    }

    if (name_eq(opn, "llvm.store")) {
        // store T val, ptr p, align A
        MLIR_ValueHandle val = MLIR_GetOpOperand(op, 0);
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 1);
        buf_cstr(out, "  store ");
        print_type(out, F->ctx, MLIR_GetValueType(val));
        buf_putc(out, ' ');
        emit_use(F, val);
        buf_cstr(out, ", ptr ");
        emit_use(F, ptr);
        buf_printf(out, ", align %u\n", type_align(F->ctx, MLIR_GetValueType(val)));
        return;
    }
    if (name_eq(opn, "llvm.load")) {
        // %r = load T, ptr p, align A
        MLIR_ValueHandle ptr = MLIR_GetOpOperand(op, 0);
        MLIR_TypeHandle rty = MLIR_GetOpResult_type(op, 0);
        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = load ");
        print_type(out, F->ctx, rty);
        buf_cstr(out, ", ptr ");
        emit_use(F, ptr);
        buf_printf(out, ", align %u\n", type_align(F->ctx, rty));
        return;
    }

    if (name_eq(opn, "llvm.getelementptr")) {
        // %r = getelementptr ELEM_TY, ptr base, INDICES
        // MLIR form: llvm.getelementptr %p[0, 1, %v] : ... -> !llvm.ptr, ELEM_TY
        // The constant indices come from the "rawConstantIndices" attribute
        // (DenseI32ArrayAttr); -2147483648 means "use next operand".
        MLIR_AttributeHandle eta = find_attr(op, "elem_type");
        MLIR_TypeHandle elem = MLIR_GetAttributeTypeValue(eta);
        MLIR_AttributeHandle ria = find_attr(op, "rawConstantIndices");
        MLIR_ValueHandle base = MLIR_GetOpOperand(op, 0);
        size_t nops = MLIR_GetOpNumOperands(op);
        size_t next_op = 1;

        buf_cstr(out, "  ");
        emit_use(F, MLIR_GetOpResult(op, 0));
        buf_cstr(out, " = getelementptr ");
        print_type(out, F->ctx, elem);
        buf_cstr(out, ", ptr ");
        emit_use(F, base);

        if (ria != MLIR_INVALID_HANDLE) {
            // Parse the printed array form: "array<i32: 0, 1>" or "[0, 1]".
            string s = MLIR_GetAttributeAsString(F->ctx, ria);
            // Find first '[' or ':' then walk numbers separated by commas.
            size_t i = 0;
            while (i < s.size && s.str[i] != '[' && s.str[i] != ':') i++;
            if (i < s.size && s.str[i] == ':') i++;     // "array<i32: ..."
            else if (i < s.size && s.str[i] == '[') i++; // "[..."
            while (i < s.size) {
                while (i < s.size && (s.str[i] == ' ' || s.str[i] == ',')) i++;
                if (i >= s.size || s.str[i] == ']' || s.str[i] == '>') break;
                // Read signed integer.
                long long v = 0; int sign = 1;
                if (s.str[i] == '-') { sign = -1; i++; }
                while (i < s.size && s.str[i] >= '0' && s.str[i] <= '9') {
                    v = v*10 + (s.str[i]-'0'); i++;
                }
                v *= sign;
                if (v == (long long)(int32_t)0x80000000) {
                    // Dynamic — pull from operands.
                    MLIR_ValueHandle dv = MLIR_GetOpOperand(op, next_op++);
                    buf_cstr(out, ", ");
                    print_type(out, F->ctx, MLIR_GetValueType(dv));
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
            MLIR_ValueHandle dv = MLIR_GetOpOperand(op, j);
            buf_cstr(out, ", ");
            print_type(out, F->ctx, MLIR_GetValueType(dv));
            buf_putc(out, ' ');
            emit_use(F, dv);
        }
        buf_putc(out, '\n');
        return;
    }

    if (name_eq(opn, "llvm.call")) { emit_call(F, op); return; }

    if (name_eq(opn, "llvm.return")) {
        if (MLIR_GetOpNumOperands(op) == 0) buf_cstr(out, "  ret void\n");
        else {
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, 0);
            buf_cstr(out, "  ret ");
            print_type(out, F->ctx, MLIR_GetValueType(v));
            buf_putc(out, ' ');
            emit_use(F, v);
            buf_putc(out, '\n');
        }
        return;
    }

    if (name_eq(opn, "llvm.br")) {
        MLIR_BlockHandle dst = MLIR_GetOpSuccessor(op, 0);
        buf_printf(out, "  br label %%%s\n", bname(F, dst));
        return;
    }
    if (name_eq(opn, "llvm.cond_br")) {
        MLIR_ValueHandle c = MLIR_GetOpOperand(op, 0);
        MLIR_BlockHandle t = MLIR_GetOpSuccessor(op, 0);
        MLIR_BlockHandle f = MLIR_GetOpSuccessor(op, 1);
        buf_cstr(out, "  br i1 ");
        emit_use(F, c);
        buf_printf(out, ", label %%%s, label %%%s\n", bname(F, t), bname(F, f));
        return;
    }
    if (name_eq(opn, "llvm.unreachable")) {
        buf_cstr(out, "  unreachable\n");
        return;
    }
    if (name_eq(opn, "llvm.intr.stacksave") || name_eq(opn, "llvm.intr.stackrestore")) {
        // Skip for now — tinyc doesn't rely on these.
        return;
    }

    if (name_eq(opn, "llvm.intr.vastart") || name_eq(opn, "llvm.intr.va_start")) {
        buf_cstr(out, "  call void @llvm.va_start.p0(ptr ");
        emit_use(F, MLIR_GetOpOperand(op, 0));
        buf_cstr(out, ")\n");
        return;
    }
    if (name_eq(opn, "llvm.intr.vaend") || name_eq(opn, "llvm.intr.va_end")) {
        buf_cstr(out, "  call void @llvm.va_end.p0(ptr ");
        emit_use(F, MLIR_GetOpOperand(op, 0));
        buf_cstr(out, ")\n");
        return;
    }
    if (name_eq(opn, "llvm.intr.vacopy") || name_eq(opn, "llvm.intr.va_copy")) {
        buf_cstr(out, "  call void @llvm.va_copy.p0(ptr ");
        emit_use(F, MLIR_GetOpOperand(op, 0));
        buf_cstr(out, ", ptr ");
        emit_use(F, MLIR_GetOpOperand(op, 1));
        buf_cstr(out, ")\n");
        return;
    }
    // Float math intrinsics: llvm.intr.sqrt / sin / cos / fabs / ...
    if (name_starts_with(opn, "llvm.intr.")) {
        // Generic shape: %r = call T @llvm.NAME.f32/f64(T %x[, T %y, ...])
        const char *iname = opn.str + 10;
        size_t in = opn.size - 10;
        bool has_res = MLIR_GetOpNumResults(op) > 0;
        MLIR_TypeHandle rty = has_res ? MLIR_GetOpResult_type(op, 0) :
                              (MLIR_GetOpNumOperands(op) > 0 ? MLIR_GetValueType(MLIR_GetOpOperand(op, 0)) : MLIR_INVALID_HANDLE);
        // Pick suffix from the first operand/result type (f32/f64/etc.)
        const char *suffix = "f64";
        if (rty != MLIR_INVALID_HANDLE) {
            string ts = MLIR_GetTypeString(F->ctx, rty);
            if (ts.size == 3 && memcmp(ts.str, "f32", 3) == 0) suffix = "f32";
            else if (ts.size == 3 && memcmp(ts.str, "f64", 3) == 0) suffix = "f64";
            else if (ts.size == 3 && memcmp(ts.str, "f16", 3) == 0) suffix = "f16";
        }
        buf_cstr(out, "  ");
        if (has_res) { emit_use(F, MLIR_GetOpResult(op, 0)); buf_cstr(out, " = "); }
        buf_cstr(out, "call ");
        if (has_res) print_type(out, F->ctx, rty); else buf_cstr(out, "void");
        buf_cstr(out, " @llvm.");
        buf_append(out, iname, in);
        buf_putc(out, '.');
        buf_cstr(out, suffix);
        buf_putc(out, '(');
        size_t no = MLIR_GetOpNumOperands(op);
        for (size_t i = 0; i < no; i++) {
            if (i) buf_cstr(out, ", ");
            MLIR_ValueHandle v = MLIR_GetOpOperand(op, i);
            print_type(out, F->ctx, MLIR_GetValueType(v));
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

static void collect_all_ops(MLIR_RegionHandle body, MLIR_OpHandle **out, size_t *n, size_t *cap) {
    size_t nb = MLIR_GetRegionNumBlocks(body);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(body, bi);
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t oi = 0; oi < nops; oi++) {
            if (*n == *cap) {
                *cap = *cap ? *cap * 2 : 64;
                *out = (MLIR_OpHandle *)realloc(*out, *cap * sizeof(MLIR_OpHandle));
            }
            (*out)[(*n)++] = MLIR_GetBlockOp(blk, oi);
        }
    }
}

static void emit_function(MLIR_Context *ctx, Buf *out, MLIR_OpHandle fn) {
    string sym = {0};
    MLIR_TypeHandle fty = MLIR_INVALID_HANDLE;
    MLIR_AttributeHandle visa = find_attr(fn, "sym_visibility");
    bool is_private = false;
    if (visa != MLIR_INVALID_HANDLE) {
        string v = MLIR_GetAttributeString(visa);
        if (name_eq(v, "private")) is_private = true;
    }
    MLIR_AttributeHandle syma = find_attr(fn, "sym_name");
    if (syma != MLIR_INVALID_HANDLE) sym = MLIR_GetAttributeString(syma);
    MLIR_AttributeHandle ftya = find_attr(fn, "function_type");
    if (ftya != MLIR_INVALID_HANDLE) fty = MLIR_GetAttributeTypeValue(ftya);

    size_t ni = fty != MLIR_INVALID_HANDLE ? MLIR_GetTypeFunctionNumInputs(fty) : 0;
    size_t no = fty != MLIR_INVALID_HANDLE ? MLIR_GetTypeFunctionNumResults(fty) : 0;

    // Detect varargs by scanning the printed type (the function type
    // attribute records "..." when present).
    bool varargs = false;
    if (fty != MLIR_INVALID_HANDLE) {
        string ts = MLIR_GetTypeString(ctx, fty);
        for (size_t i = 0; i + 2 < ts.size; i++) {
            if (ts.str[i] == '.' && ts.str[i+1] == '.' && ts.str[i+2] == '.') { varargs = true; break; }
        }
    }

    bool has_body = MLIR_GetOpNumRegions(fn) > 0 &&
                    MLIR_GetRegionNumBlocks(MLIR_GetOpRegion(fn, 0)) > 0;

    if (!has_body) {
        buf_cstr(out, "declare ");
        if (no == 0) buf_cstr(out, "void");
        else         print_type(out, ctx, MLIR_GetTypeFunctionResult(fty, 0));
        buf_printf(out, " @%.*s(", (int)sym.size, sym.str);
        for (size_t i = 0; i < ni; i++) {
            if (i) buf_cstr(out, ", ");
            print_type(out, ctx, MLIR_GetTypeFunctionInput(fty, i));
        }
        if (varargs) { if (ni) buf_cstr(out, ", "); buf_cstr(out, "..."); }
        buf_cstr(out, ")\n");
        return;
    }

    buf_cstr(out, "define ");
    if (is_private) buf_cstr(out, "internal ");
    if (no == 0) buf_cstr(out, "void");
    else         print_type(out, ctx, MLIR_GetTypeFunctionResult(fty, 0));
    buf_printf(out, " @%.*s(", (int)sym.size, sym.str);
    // Parameters use the names assigned by assign_names ("%a0" etc.).
    FnCtx F; memset(&F, 0, sizeof F);
    F.ctx = ctx; F.out = out;
    MLIR_RegionHandle body = MLIR_GetOpRegion(fn, 0);
    assign_names(&F, body);
    for (size_t i = 0; i < ni; i++) {
        if (i) buf_cstr(out, ", ");
        print_type(out, ctx, MLIR_GetTypeFunctionInput(fty, i));
        buf_putc(out, ' ');
        MLIR_BlockHandle entry = MLIR_GetRegionBlock(body, 0);
        MLIR_ValueHandle a = MLIR_GetBlockArg(entry, i);
        emit_use(&F, a);
    }
    if (varargs) { if (ni) buf_cstr(out, ", "); buf_cstr(out, "..."); }
    buf_cstr(out, ") {\n");

    // Collect all ops (for predecessor scans).
    MLIR_OpHandle *all = NULL; size_t an = 0, ac = 0;
    collect_all_ops(body, &all, &an, &ac);

    size_t nb = MLIR_GetRegionNumBlocks(body);
    for (size_t bi = 0; bi < nb; bi++) {
        MLIR_BlockHandle blk = MLIR_GetRegionBlock(body, bi);
        // Always emit the block label, including for the entry block: after
        // mem2reg a phi in a successor block may name the entry block as an
        // incoming predecessor (e.g. "[ 1, %entry ]"), which requires the
        // entry block to carry an "entry:" label. LLVM permits this — the
        // entry block simply may not be used as a branch *target*.
        buf_printf(out, "%s:\n", bname(&F, blk));
        if (bi != 0) emit_phis(&F, blk, all, an);
        size_t nops = MLIR_GetBlockNumOps(blk);
        for (size_t oi = 0; oi < nops; oi++) {
            emit_op(&F, MLIR_GetBlockOp(blk, oi));
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

static void emit_global(MLIR_Context *ctx, Buf *out, MLIR_OpHandle gop) {
    MLIR_AttributeHandle syma = find_attr(gop, "sym_name");
    string sym = syma != MLIR_INVALID_HANDLE ? MLIR_GetAttributeString(syma) : (string){0,0};
    MLIR_AttributeHandle gtya = find_attr(gop, "global_type");
    MLIR_TypeHandle gty = MLIR_INVALID_HANDLE;
    if (gtya != MLIR_INVALID_HANDLE) gty = MLIR_GetAttributeTypeValue(gtya);

    // Linkage: "linkage" attr is an LLVM Linkage attr; we read it via
    // GetAttributeAsString and map common values. A separate flag tracks
    // "external" because it's a declaration form in LLVM IR text (no
    // initializer is permitted/expected) rather than a linkage prefix.
    const char *linkage = "";
    bool is_external = false;
    MLIR_AttributeHandle linka = find_attr(gop, "linkage");
    if (linka != MLIR_INVALID_HANDLE) {
        string s = MLIR_GetAttributeAsString(ctx, linka);
        if      (name_starts_with(s, "#llvm.linkage<internal>")) linkage = "internal ";
        else if (name_starts_with(s, "#llvm.linkage<private>"))  linkage = "private ";
        else if (name_starts_with(s, "#llvm.linkage<external>")) is_external = true;
        else if (name_starts_with(s, "#llvm.linkage<common>"))   linkage = "common ";
        else if (name_starts_with(s, "#llvm.linkage<weak>"))     linkage = "weak ";
    }
    bool is_constant = false;
    MLIR_AttributeHandle ca = find_attr(gop, "constant");
    if (ca != MLIR_INVALID_HANDLE) is_constant = true;

    // Initial value: either the "value" attribute (string/numeric), or a
    // body region whose llvm.return yields the constant.
    char *init = NULL;
    MLIR_AttributeHandle va = find_attr(gop, "value");
    if (va != MLIR_INVALID_HANDLE) {
        MLIR_AttrKind k = MLIR_GetAttributeKind(va);
        if (k == MLIR_ATTR_KIND_STRING) {
            string s = MLIR_GetAttributeString(va);
            // For !llvm.array<N x iX> with X > 8, LLVM IR cannot use a
            // c"..." byte string — that form is reserved for i8 arrays.
            // Unpack the raw bytes into an `[N x iX] [iX v1, iX v2, ...]`
            // typed array constant.
            unsigned elem_w = 0;
            uint64_t arr_n = 0;
            if (gty != MLIR_INVALID_HANDLE && MLIR_IsTypeLLVMArray(gty)) {
                arr_n = MLIR_GetTypeLLVMArrayNumElements(gty);
                MLIR_TypeHandle et = MLIR_GetTypeLLVMArrayElement(gty);
                string ets = MLIR_GetTypeString(ctx, et);
                if (ets.size >= 2 && ets.str[0] == 'i') {
                    int w = 0;
                    for (size_t i = 1; i < ets.size; i++) {
                        if (ets.str[i] >= '0' && ets.str[i] <= '9') {
                            w = w * 10 + (ets.str[i] - '0');
                        } else { w = 0; break; }
                    }
                    if (w == 8 || w == 16 || w == 32 || w == 64) elem_w = (unsigned)w;
                }
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
            init = fmt_alloc("%lld", (long long)MLIR_GetAttributeInteger(va));
        } else if (k == MLIR_ATTR_KIND_FLOAT) {
            init = fmt_alloc("%e", MLIR_GetAttributeFloat(va));
        }
    }
    if (init == NULL && MLIR_GetOpNumRegions(gop) > 0) {
        MLIR_RegionHandle r = MLIR_GetOpRegion(gop, 0);
        if (MLIR_GetRegionNumBlocks(r) > 0) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(r, 0);
            size_t nops = MLIR_GetBlockNumOps(blk);
            // Find the llvm.return; its operand's defining op is the
            // initializer expression.
            for (size_t i = 0; i < nops; i++) {
                MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
                if (name_eq(MLIR_GetOpName(op), "llvm.return") &&
                    MLIR_GetOpNumOperands(op) > 0) {
                    MLIR_ValueHandle v = MLIR_GetOpOperand(op, 0);
                    MLIR_OpHandle def = MLIR_GetValueDefiningOp(v);
                    char *lit = inline_literal_for(ctx, def, MLIR_GetValueType(v));
                    if (lit) { init = lit; break; }
                    // Try to fold an `llvm.mlir.undef` + chain of
                    // `llvm.insertvalue` (which is how
                    // MLIR_CreateLLVMGlobalArrayInit's upstream impl
                    // builds an aggregate-array initializer). Walk back
                    // through operand[0] of each insertvalue; collect
                    // (position, scalar-literal) pairs from operand[1].
                    char *folded = fold_aggregate_init_chain(ctx, def, gty);
                    if (folded) init = folded;
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
typedef struct { StructDecl *e; size_t n, cap; } StructList;

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
    string s = MLIR_GetTypeString(ctx, ty);
    scan_type_text(ctx, s.str, s.size, L);
}

static void scan_op_for_structs(MLIR_Context *ctx, MLIR_OpHandle op, StructList *L) {
    size_t nr = MLIR_GetOpNumResults(op);
    for (size_t i = 0; i < nr; i++) scan_type_for_structs(ctx, MLIR_GetOpResult_type(op, i), L);
    size_t no = MLIR_GetOpNumOperands(op);
    for (size_t i = 0; i < no; i++) scan_type_for_structs(ctx, MLIR_GetValueType(MLIR_GetOpOperand(op, i)), L);
    size_t na = MLIR_GetOpNumAttributes(op);
    for (size_t i = 0; i < na; i++) {
        MLIR_AttributeHandle a = MLIR_GetOpAttribute(op, i);
        if (MLIR_GetAttributeKind(a) == MLIR_ATTR_KIND_TYPE) {
            scan_type_for_structs(ctx, MLIR_GetAttributeTypeValue(a), L);
        } else {
            // Scan the printed form for embedded type strings.
            string s = MLIR_GetAttributeAsString(ctx, a);
            scan_type_text(ctx, s.str, s.size, L);
        }
    }
    size_t nrg = MLIR_GetOpNumRegions(op);
    for (size_t ri = 0; ri < nrg; ri++) {
        MLIR_RegionHandle r = MLIR_GetOpRegion(op, ri);
        size_t nb = MLIR_GetRegionNumBlocks(r);
        for (size_t bi = 0; bi < nb; bi++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(r, bi);
            size_t nops = MLIR_GetBlockNumOps(blk);
            for (size_t oi = 0; oi < nops; oi++) {
                scan_op_for_structs(ctx, MLIR_GetBlockOp(blk, oi), L);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Module entry.
// -----------------------------------------------------------------------------

string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module) {
    Buf out = {0};
    buf_cstr(&out, "; ModuleID = 'LLVMDialectModule'\n");
    buf_cstr(&out, "source_filename = \"LLVMDialectModule\"\n\n");

    // Module body region 0 contains a single block with all top-level ops.
    MLIR_RegionHandle mr = MLIR_GetOpRegion(module, 0);
    MLIR_BlockHandle mb = MLIR_GetRegionBlock(mr, 0);
    size_t nops = MLIR_GetBlockNumOps(mb);

    // Gather struct decls.
    StructList sl = {0};
    for (size_t i = 0; i < nops; i++) {
        scan_op_for_structs(ctx, MLIR_GetBlockOp(mb, i), &sl);
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

    // Emit globals first, then functions (order doesn't really matter to llc).
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        string n = MLIR_GetOpName(op);
        if (name_eq(n, "llvm.mlir.global")) emit_global(ctx, &out, op);
    }
    for (size_t i = 0; i < nops; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(mb, i);
        string n = MLIR_GetOpName(op);
        if (name_eq(n, "llvm.func")) emit_function(ctx, &out, op);
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
